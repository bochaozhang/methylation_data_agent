"""
Agent 1: DatabaseAgent

Searches TCGA (GDC) and GEO databases for methylation datasets
matching the parsed user intent, then downloads them.

Operates as a LangGraph node (function that takes/returns MethyAgentState).
Uses a ReAct-style loop internally: search → filter → register → download.

GEO filtering pipeline (v4):
  Every GSE returned by esearch goes through PubMed cross-verification:
    1. Fetch the linked PubMed abstract (NCBI efetch)
    2. LLM compares GEO summary/design vs abstract
    3. Datasets where abstract contradicts GEO metadata are dropped (keep=False)
    4. Datasets with no PMID are kept by default (conservative)
  This replaces the previous GSM-level LLM judge.
"""
import asyncio
import re
import os
from typing import Any, Dict, List, Optional, Tuple


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from registry.registry import Registry
from state.graph_state import MethyAgentState
from tools.download_tools import (
    DownloadEngine,
    build_geo_download_tasks,
    build_tcga_download_tasks,
)
from tools.geo_tools import GEOClient
from tools.parser_tools import SAMPLE_TYPE_RELATED, TCGA_CODE_TO_ENGLISH
from tools.tcga_tools import GDCClient
from utils.logger import get_logger
from utils.llm_factory import get_llm

logger = get_logger(__name__)

# NOTE: Keep this string CONSTANT across all calls.
# Z.AI (GLM) automatically caches identical system prompts — cached tokens are
# billed at a lower rate and reduce latency. Any change to this string invalidates
# the cache. Dynamic content (query, dataset info) must go in the user message only.
SYSTEM_PROMPT = """You are DatabaseAgent, a specialized AI for discovering and downloading DNA methylation datasets from TCGA and GEO databases.

Your workflow:
1. Use the search tools to find relevant methylation datasets based on the parsed intent
2. Filter results to only include methylation data (450K, EPIC arrays, WGBS, RRBS)
3. Check the registry to avoid re-downloading existing datasets
4. Download new datasets and update the registry

Always prefer:
- Illumina 450K/EPIC array data (most common in TCGA/GEO)
- Level 3 processed data (beta values) over raw data
- Datasets with clear cancer type annotations

Report your findings clearly, including how many datasets were found, filtered, and downloaded.
"""

# LLM judge system prompt — kept constant for Z.AI system cache.
# NOTE: This prompt is retained for reference but _llm_judge_datasets_concurrent
# is no longer called in the main pipeline (replaced by PubMed verification).
LLM_JUDGE_SYSTEM_PROMPT = """You are a biomedical data curator specializing in DNA methylation datasets.

Given metadata for a GEO dataset (title, summary, sample source, molecule type, characteristics),
decide whether it is a valid methylation dataset matching the user's sample type requirement.

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{
  "keep": true or false,
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence explaining the decision",
  "detected_sample_type": "tumor|adjacent|normal|wbc|cfdna|plasma|serum|whole_blood|cell_line|unknown"
}

Rules:
- keep=true  if the dataset matches the requested sample type
- keep=false if it clearly does NOT match (e.g. cell lines when plasma requested,
              tissue gDNA when cfDNA requested)
- keep=true  if evidence is ambiguous or insufficient (conservative — prefer false positives)
- Cell lines / organoids / in-vitro models → always keep=false
"""

# PubMed verification system prompt — kept CONSTANT for Z.AI system cache.
# This is the PRIMARY filter for GEO datasets: every GSE goes through this step.
# The LLM cross-checks GEO summary/design against the paper abstract and decides
# whether the dataset is consistent and usable.
LLM_VERIFY_SYSTEM_PROMPT = """You are a biomedical data curator specializing in DNA methylation datasets.

You will receive:
1. GEO dataset metadata: accession, title, summary, overall design, platform, sample count, sample type, cancer type
2. The abstract of the linked publication (fetched from PubMed)

Your task: cross-check whether the GEO metadata is consistent with the paper abstract, and decide if the dataset is usable.

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{
  "keep": true or false,
  "confirmed_sample_type": "plasma|tumor|adjacent|normal|wbc|cfdna|serum|whole_blood|cell_line|unknown",
  "confirmed_cancer_type": "canonical English cancer name, e.g. colorectal cancer",
  "sample_count_in_paper": null or integer,
  "stage_treatment": "staging/treatment info from abstract, or null",
  "accession_mentioned": true or false,
  "consistency": "consistent|minor_discrepancy|major_discrepancy",
  "recommended_action": "download" | "review" | "skip",
  "reason": "one sentence summarising the verification result",
  "notes": "any discrepancy between GEO metadata and abstract, or empty string"
}

Decision rules for `keep`:
- keep=true  if the abstract confirms the dataset matches the GEO metadata (cancer type, sample type, methylation data)
- keep=true  if the abstract is unavailable or uninformative — give benefit of the doubt
- keep=false ONLY if the abstract clearly contradicts the GEO metadata, e.g.:
    * GEO says plasma cfDNA but abstract describes tumor tissue only
    * GEO says cancer X but abstract is about a completely different cancer
    * Abstract confirms the dataset is cell lines / organoids / in-vitro only
    * Abstract confirms the data is NOT methylation (e.g. RNA-seq, proteomics)
- When in doubt, keep=true (conservative — prefer false positives over false negatives)

Rules for other fields:
- confirmed_sample_type: use the abstract's description of biological material
- sample_count_in_paper: extract the n= number for the main patient cohort; null if not stated
- accession_mentioned: true only if the GEO accession (e.g. GSE220160) appears explicitly in the abstract
- consistency: consistent if GEO and abstract agree; minor_discrepancy if small differences; major_discrepancy if fundamental mismatch
- recommended_action: skip only if keep=false with high confidence; review if minor discrepancy; download otherwise
- notes: record any discrepancy (e.g. sample count mismatch, different cancer subtype)
"""


class DatabaseAgent:
    """
    Agent 1: Searches GEO and TCGA for methylation datasets and downloads them.

    Args:
        config: Full settings dict from settings.yaml.
        registry: Shared Registry instance.
    """

    def __init__(self, config: Dict[str, Any], registry: Registry):
        self.config = config
        self.registry = registry
        self.llm = get_llm(config["llm"])

        # Initialize API clients
        ncbi_key = os.environ.get(config["geo"].get("api_key_env", ""), "")
        # Proxy: env var NCBI_PROXY takes priority, then settings.yaml geo.proxy
        ncbi_proxy = os.environ.get("NCBI_PROXY", "") or config.get("geo", {}).get("proxy", "")
        self.geo_client = GEOClient(api_key=ncbi_key or None, proxy=ncbi_proxy or None)

        gdc_token = os.environ.get(config["tcga"].get("gdc_token_env", ""), "")
        self.gdc_client = GDCClient(token=gdc_token or None)

        self.downloader = DownloadEngine(
            output_dir=config["download"]["output_dir"],
            max_concurrent=config["download"]["max_concurrent"],
            retry_attempts=config["download"]["retry_attempts"],
            retry_delay=config["download"]["retry_delay"],
            chunk_size_mb=config["download"]["chunk_size_mb"],
            timeout=config["download"]["timeout"],
        )

    # ------------------------------------------------------------------ #
    #  LangGraph node entry point                                          #
    # ------------------------------------------------------------------ #

    def run(self, state: MethyAgentState) -> MethyAgentState:
        """
        Main LangGraph node function.
        Takes the current state, runs the database search+download pipeline,
        and returns the updated state.
        """
        intent = state.get("parsed_intent", {})
        logger.info(f"DatabaseAgent starting. Intent: {intent}")

        candidates = []
        downloaded = []
        failed = []
        skipped = []
        errors = list(state.get("error_log", []))

        # ---- Step 1: Handle explicit accession mode ----
        accessions_dict = intent.get("accessions", {})
        explicit_geo = accessions_dict.get("geo", []) if isinstance(accessions_dict, dict) else []
        explicit_tcga = accessions_dict.get("tcga", []) if isinstance(accessions_dict, dict) else []

        if explicit_geo or explicit_tcga:
            logger.info(f"Accession mode: GEO={explicit_geo}, TCGA={explicit_tcga}")
            geo_candidates = self._fetch_geo_by_accessions(explicit_geo)
            tcga_candidates = self._fetch_tcga_by_accessions(explicit_tcga)
            candidates = geo_candidates + tcga_candidates
        else:
            # ---- Step 2: Semantic search mode ----
            logger.info("Semantic search mode")
            geo_candidates = self._search_geo(intent)
            tcga_candidates = self._search_tcga(intent)
            candidates = geo_candidates + tcga_candidates

        logger.info(f"Total candidates found: {len(candidates)}")

        # ---- Step 3: Dedup against registry ----
        new_candidates = []
        for c in candidates:
            acc = c.get("accession", "")
            if not acc:
                continue
            if self.registry.exists(acc):
                logger.info(f"Skipping {acc}: already in registry")
                skipped.append(acc)
            else:
                new_candidates.append(c)

        logger.info(f"New datasets to download: {len(new_candidates)} (skipped {len(skipped)})")

        # ---- Step 4: Register and download ----
        download_tasks = []
        for c in new_candidates:
            acc = c["accession"]
            # Register with pending status
            self.registry.upsert_dataset(
                accession=acc,
                source=c.get("source", "GEO"),
                discovered_by="agent1",
                data_type=c.get("data_type"),
                cancer_type=c.get("cancer_type"),
                platform=c.get("platform_canonical") or c.get("platform"),
                sample_count=c.get("sample_count"),
                year=c.get("year"),
                title=c.get("title"),
                sample_type=c.get("sample_type"),
                download_status="pending",
            )
            self.registry.log_event(acc, "start", f"Registered by DatabaseAgent")

            # Build download tasks
            if c.get("source") == "TCGA":
                tasks = build_tcga_download_tasks(
                    c,
                    self.config["download"]["output_dir"],
                    self.config["tcga"]["gdc_api_base"],
                )
            else:
                tasks = build_geo_download_tasks(c, self.config["download"]["output_dir"])

            download_tasks.extend(tasks)

        # ---- Step 5: Execute downloads ----
        if download_tasks:
            self.registry.update_status(
                download_tasks[0]["accession"], "downloading"
            )
            results = self.downloader.download_many_sync(download_tasks)

            # Aggregate results by accession
            acc_results: Dict[str, List] = {}
            for r in results:
                acc = r["accession"]
                acc_results.setdefault(acc, []).append(r)

            for acc, acc_res in acc_results.items():
                all_done = all(r["status"] == "done" for r in acc_res)
                any_done = any(r["status"] == "done" for r in acc_res)

                if all_done:
                    local_path = acc_res[0]["local_path"]
                    file_size = sum(r.get("file_size_bytes", 0) for r in acc_res)
                    self.registry.update_status(
                        acc, "done",
                        local_path=str(local_path),
                        file_size_bytes=file_size,
                    )
                    self.registry.log_event(acc, "done", f"Downloaded {len(acc_res)} files")
                    downloaded.append(acc)
                else:
                    error_msgs = [r.get("error", "") for r in acc_res if r["status"] == "failed"]
                    self.registry.update_status(acc, "failed")
                    self.registry.log_event(acc, "error", "; ".join(error_msgs))
                    failed.append(acc)
                    errors.append(f"DatabaseAgent: {acc} failed: {'; '.join(error_msgs)}")

        # ---- Step 6: LLM summary message ----
        summary_msg = self._generate_summary_message(
            candidates, new_candidates, downloaded, failed, skipped
        )

        return {
            **state,
            "db_candidates": candidates,
            "db_downloaded": downloaded,
            "db_failed": failed,
            "db_skipped": skipped,
            "error_log": errors,
            "messages": [AIMessage(content=summary_msg, name="DatabaseAgent")],
        }

    # ------------------------------------------------------------------ #
    #  GEO search methods                                                  #
    # ------------------------------------------------------------------ #

    def _search_geo(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search GEO using the parsed intent.

        Pipeline:
          1. Build NCBI E-utilities query string (methylation platform filters included)
          2. esearch → GSE UID list
          3. filter_methylation_datasets() → platform/year filter + batch esummary
          4. Inject cancer_type from intent (GEO metadata doesn't carry it)
          5. _pubmed_verify_datasets_concurrent() — PRIMARY filter:
               - Every GSE fetches its linked PubMed abstract (NCBI efetch)
               - LLM cross-checks GEO summary/design vs abstract
               - Returns keep=True/False + corrected fields
               - Datasets with no PMID: kept by default (conservative)
        """
        from tools.parser_tools import build_geo_search_string

        # Always use build_geo_search_string to ensure methylation platform filters
        # (GPL13534/GPL21145/etc.) are included. LLM-provided geo_search_query is
        # intentionally ignored here because it typically omits GPL filters, causing
        # the search to return RNA-seq and other non-methylation datasets.
        search_query = build_geo_search_string(intent)
        if not search_query:
            return []

        try:
            accessions = self.geo_client.search_gse(search_query, max_results=2000)
            if not accessions:
                return []

            platform_filter = intent.get("platform")
            year_start = intent.get("year_start")
            year_end = intent.get("year_end")

            datasets = self.geo_client.filter_methylation_datasets(
                accessions,
                platform_filter=platform_filter,
                year_start=year_start,
                year_end=year_end,
            )

            # Inject cancer_type from intent into each GEO record
            # (GEO metadata doesn't carry cancer type; we infer it from the search intent)
            ct = intent.get("cancer_type")
            if ct:
                cancer_label = ct.get("display") if isinstance(ct, dict) else str(ct)
                for d in datasets:
                    if not d.get("cancer_type"):
                        d["cancer_type"] = cancer_label
            elif intent.get("cancer_type_code") and intent["cancer_type_code"] in TCGA_CODE_TO_ENGLISH:
                cancer_label = TCGA_CODE_TO_ENGLISH[intent["cancer_type_code"]]
                for d in datasets:
                    if not d.get("cancer_type"):
                        d["cancer_type"] = cancer_label

            # PubMed cross-verification — PRIMARY filter.
            # Every GSE fetches its linked PubMed abstract and the LLM checks
            # whether the GEO summary/design is consistent with the paper.
            # Datasets that pass (keep=True) are returned; others are logged and dropped.
            datasets = self._pubmed_verify_datasets_concurrent(datasets)

            return datasets
        except Exception as e:
            logger.error(f"GEO search failed: {e}")
            return []

    def _fetch_geo_by_accessions(self, accessions: List[str]) -> List[Dict[str, Any]]:
        """Fetch metadata for explicit GEO accessions."""
        results = []
        for acc in accessions:
            try:
                meta = self.geo_client.get_series_metadata(acc)
                if not meta.get("error"):
                    results.append(meta)
            except Exception as e:
                logger.error(f"Failed to fetch GEO metadata for {acc}: {e}")
        return results

    # ------------------------------------------------------------------ #
    #  LLM-based dataset judge (retained, not used in main pipeline)      #
    # ------------------------------------------------------------------ #

    def _enrich_dataset_for_llm(
        self,
        ds: Dict[str, Any],
        wanted_sample_type: str = "",
    ) -> Dict[str, Any]:
        """
        Enrich a GEO dataset dict with representative GSM-level details for LLM evaluation.

        Calls get_representative_gsm_details() which:
          - Reads ALL sample titles from esummary (zero extra API calls)
          - Groups GSMs by biological material type (plasma_cfdna / tissue / wbc_blood / ...)
          - Selects representative GSMs per group (not just the first N):
              <= 30 samples  -> all GSMs
              31-200         -> 2 per group, cap 10
              > 200          -> 1 per group, cap 5
          - efetch MiniML only for selected representatives

        Args:
            ds: Dataset metadata dict.
            wanted_sample_type: The sample type the caller is looking for.

        Returns the dataset dict with an added 'gsm_details' key.
        """
        acc = ds.get("accession", "")
        if not acc:
            return ds
        try:
            gsm_details = self.geo_client.get_representative_gsm_details(
                acc, wanted_sample_type=wanted_sample_type
            )
            ds = {**ds, "gsm_details": gsm_details}
        except Exception as e:
            logger.debug(f"_enrich_dataset_for_llm({acc}): {e}")
            ds = {**ds, "gsm_details": []}
        return ds

    def _llm_judge_dataset(
        self,
        ds: Dict[str, Any],
        wanted_sample_type: str,
    ) -> Tuple[bool, str, str]:
        """
        Ask the LLM to judge whether a single GEO dataset matches the wanted
        sample type. Uses LLM_JUDGE_SYSTEM_PROMPT (constant) for Z.AI system cache.

        NOTE: Not used in the main pipeline (replaced by PubMed verification).
        Retained for ad-hoc use or fallback.
        """
        import json as _json

        acc = ds.get("accession", "?")

        gsm_lines = []
        for g in ds.get("gsm_details", [])[:5]:
            ch_str = "; ".join(f"{k}={v}" for k, v in g.get("characteristics", {}).items())
            gsm_lines.append(
                f"  GSM {g['gsm']}: source={g.get('source_name','?')!r}, "
                f"molecule={g.get('molecule','?')!r}, characteristics={{{ch_str}}}"
            )
        gsm_block = "\n".join(gsm_lines) if gsm_lines else "  (no GSM details available)"

        user_msg = (
            f"Wanted sample type: {wanted_sample_type}\n\n"
            f"Dataset: {acc}\n"
            f"Title: {ds.get('title','')[:200]}\n"
            f"Summary: {ds.get('summary','')[:400]}\n"
            f"Platform: {ds.get('platform_canonical') or ds.get('platforms',[])}\n"
            f"Sample count: {ds.get('sample_count')}\n"
            f"Sample titles (first 5): {ds.get('sample_titles', [])[:5]}\n"
            f"GSM details (first 5):\n{gsm_block}\n"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=LLM_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            raw = response.content.strip()

            usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {})
            cached = 0
            if isinstance(usage, dict):
                cached = (
                    usage.get("cached_tokens")
                    or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    or 0
                )
            if cached:
                logger.debug(f"LLM judge {acc}: cached_tokens={cached}")

            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            verdict = _json.loads(raw)
            keep = bool(verdict.get("keep", True))
            confidence = verdict.get("confidence", "low")
            reason = verdict.get("reason", "")
            detected = verdict.get("detected_sample_type", "unknown")

            logger.info(
                f"LLM judge {acc}: keep={keep} conf={confidence} "
                f"detected={detected} reason={reason[:80]}"
            )
            return keep, confidence, reason

        except Exception as e:
            logger.warning(f"LLM judge failed for {acc}: {e} — keeping dataset")
            return True, "low", f"LLM judge error: {e}"

    def _llm_judge_datasets_concurrent(
        self,
        datasets: List[Dict[str, Any]],
        wanted_sample_type: str,
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run LLM judgment on all datasets with up to max_concurrent parallel calls.

        NOTE: Not used in the main pipeline (replaced by PubMed verification).
        Retained for ad-hoc use or fallback.
        """
        if not datasets or not wanted_sample_type:
            return datasets

        sem = asyncio.Semaphore(max_concurrent)

        async def _judge_one_async(ds: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                loop = asyncio.get_event_loop()
                enriched = await loop.run_in_executor(
                    None, self._enrich_dataset_for_llm, ds, wanted_sample_type
                )
                keep, conf, reason = await loop.run_in_executor(
                    None, self._llm_judge_dataset, enriched, wanted_sample_type
                )
                return {
                    **enriched,
                    "llm_keep": keep,
                    "llm_confidence": conf,
                    "llm_reason": reason,
                }

        async def _run_all() -> List[Dict[str, Any]]:
            tasks = [_judge_one_async(ds) for ds in datasets]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _run_all())
                    judged = future.result()
            else:
                judged = loop.run_until_complete(_run_all())
        except RuntimeError:
            judged = asyncio.run(_run_all())

        kept = [d for d in judged if d.get("llm_keep", True)]
        rejected = [d for d in judged if not d.get("llm_keep", True)]

        logger.info(
            f"LLM judge (concurrent={max_concurrent}): "
            f"{len(kept)}/{len(datasets)} kept, "
            f"{len(rejected)} rejected"
        )
        for d in rejected:
            logger.info(
                f"  Rejected {d.get('accession','?')}: "
                f"conf={d.get('llm_confidence')} reason={d.get('llm_reason','')[:80]}"
            )

        return kept

    # ------------------------------------------------------------------ #
    #  PubMed cross-verification — PRIMARY GEO filter                     #
    # ------------------------------------------------------------------ #

    def _pubmed_verify_dataset(self, ds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-check a GEO dataset against its associated PubMed abstract.

        This is the PRIMARY filter for GEO datasets. Every GSE goes through this
        step regardless of sample type or other metadata.

        Workflow:
          1. Take the first PMID from ds["pubmed_ids"] (primary publication).
          2. Fetch the abstract via GEOClient.fetch_pubmed_abstract() (NCBI efetch).
          3. Send GEO metadata (title, summary, overall_design, platform, sample_count,
             sample_type, cancer_type) + abstract to LLM with LLM_VERIFY_SYSTEM_PROMPT
             (constant → Z.AI system cache).
          4. LLM returns:
               keep                 → whether the dataset is consistent with the abstract
               confirmed_sample_type → corrected sample type from abstract
               confirmed_cancer_type → corrected cancer type from abstract
               sample_count_in_paper → n= from abstract (corrects GEO if >20% diff)
               stage_treatment       → staging/treatment info from abstract
               consistency           → consistent / minor_discrepancy / major_discrepancy
               recommended_action    → download / review / skip
               reason                → one-sentence summary
               notes                 → discrepancy details

        Fallback behaviour (conservative — prefer false positives):
          - No PMID available     → keep=True, notes="no_pubmed_link"
          - Abstract fetch fails  → keep=True, notes="abstract_unavailable"
          - LLM/parse error       → keep=True, notes="verify_error: ..."

        Returns updated ds dict with added fields:
          pubmed_verified  (bool)
          pubmed_keep      (bool)  — used by _pubmed_verify_datasets_concurrent to filter
          paper_pmid       (str)
        """
        import json as _json

        acc = ds.get("accession", "?")
        pmids = ds.get("pubmed_ids", [])

        # ---- No PMID: keep by default ----
        if not pmids:
            logger.debug(f"pubmed_verify {acc}: no PMID — keeping (conservative)")
            existing_notes = ds.get("notes") or ""
            return {
                **ds,
                "pubmed_verified": False,
                "pubmed_keep": True,
                "notes": (existing_notes + "; no_pubmed_link").lstrip("; "),
            }

        pmid = str(pmids[0])
        abstract = self.geo_client.fetch_pubmed_abstract(pmid)

        # ---- Abstract unavailable: keep by default ----
        if not abstract:
            logger.debug(f"pubmed_verify {acc}: abstract unavailable for PMID {pmid} — keeping")
            existing_notes = ds.get("notes") or ""
            return {
                **ds,
                "pubmed_verified": False,
                "pubmed_keep": True,
                "notes": (existing_notes + f"; abstract_unavailable(PMID={pmid})").lstrip("; "),
            }

        # ---- Build user message (dynamic content only) ----
        user_msg = (
            f"GEO Accession: {acc}\n"
            f"Title: {ds.get('title', '')[:200]}\n"
            f"Summary: {ds.get('summary', '')[:400]}\n"
            f"Overall Design: {ds.get('overall_design', '')[:300]}\n"
            f"Platform: {ds.get('platform_canonical') or ds.get('platforms', [])}\n"
            f"Sample count (GEO): {ds.get('sample_count')}\n"
            f"Sample type (GEO): {ds.get('sample_type')}\n"
            f"Cancer type (GEO): {ds.get('cancer_type')}\n"
            f"PMID: {pmid}\n\n"
            f"Abstract:\n{abstract[:2500]}\n"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=LLM_VERIFY_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            raw = response.content.strip()

            # Log cached tokens (Z.AI system cache telemetry)
            usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {})
            if isinstance(usage, dict):
                cached = (
                    usage.get("cached_tokens")
                    or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    or 0
                )
                if cached:
                    logger.debug(f"pubmed_verify {acc}: cached_tokens={cached}")

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            v = _json.loads(raw)

            # ---- Apply verified fields ----
            updated = dict(ds)
            updated["pubmed_verified"] = True
            updated["pubmed_keep"] = bool(v.get("keep", True))
            updated["paper_pmid"] = pmid

            if v.get("confirmed_sample_type") and v["confirmed_sample_type"] != "unknown":
                updated["sample_type"] = v["confirmed_sample_type"]

            if v.get("confirmed_cancer_type"):
                updated["cancer_type"] = v["confirmed_cancer_type"]

            if v.get("stage_treatment"):
                updated["stage_treatment"] = v["stage_treatment"]

            # Correct sample_count if paper value differs by >20% from GEO value
            paper_n = v.get("sample_count_in_paper")
            geo_n = ds.get("sample_count")
            if paper_n and geo_n and geo_n > 0:
                if abs(paper_n - geo_n) / geo_n > 0.20:
                    updated["sample_count"] = paper_n
                    discrepancy_note = f"sample_count GEO={geo_n} paper={paper_n}"
                    existing_notes = updated.get("notes") or ""
                    updated["notes"] = (existing_notes + "; " + discrepancy_note).lstrip("; ")
            elif paper_n and not geo_n:
                updated["sample_count"] = paper_n

            # usable / recommended_action / reason — paper verification takes priority
            updated["usable"] = int(bool(v.get("keep", True)))
            if v.get("recommended_action"):
                updated["recommended_action"] = v["recommended_action"]
            if v.get("reason"):
                updated["reason"] = v["reason"]
            if v.get("consistency"):
                updated["consistency"] = v["consistency"]

            # Append notes (never overwrite existing notes)
            if v.get("notes"):
                existing_notes = updated.get("notes") or ""
                updated["notes"] = (existing_notes + "; " + v["notes"]).lstrip("; ")

            logger.info(
                f"pubmed_verify {acc} (PMID={pmid}): "
                f"keep={updated['pubmed_keep']} "
                f"consistency={v.get('consistency','?')} "
                f"sample_type={updated.get('sample_type')} "
                f"action={updated.get('recommended_action')} "
                f"reason={v.get('reason','')[:80]}"
            )
            return updated

        except Exception as e:
            # On any error: keep the dataset (conservative)
            logger.warning(f"pubmed_verify {acc}: LLM/parse error — {e} — keeping dataset")
            existing_notes = ds.get("notes") or ""
            return {
                **ds,
                "pubmed_verified": False,
                "pubmed_keep": True,
                "notes": (existing_notes + f"; verify_error: {e}").lstrip("; "),
            }

    def _pubmed_verify_datasets_concurrent(
        self,
        datasets: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run PubMed cross-verification on ALL datasets with up to max_concurrent
        parallel calls (asyncio + Semaphore), then filter on keep=True.

        Every dataset goes through this step:
          - Has a PMID → fetch abstract (NCBI efetch) + LLM verify
          - No PMID    → kept by default (conservative), notes="no_pubmed_link"
          - Abstract unavailable → kept by default
          - LLM/parse error → kept by default

        After verification, datasets where pubmed_keep=False are dropped and logged.

        Args:
            datasets: List of GEO metadata dicts from filter_methylation_datasets().
            max_concurrent: Max parallel efetch + LLM calls (default 5).

        Returns:
            Filtered list of datasets where pubmed_keep=True, each enriched with
            verified/corrected fields (sample_type, cancer_type, sample_count,
            stage_treatment, recommended_action, reason, notes).
        """
        if not datasets:
            return datasets

        sem = asyncio.Semaphore(max_concurrent)

        async def _verify_one_async(ds: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._pubmed_verify_dataset, ds)

        async def _run_all() -> List[Dict[str, Any]]:
            tasks = [_verify_one_async(ds) for ds in datasets]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _run_all())
                    verified = future.result()
            else:
                verified = loop.run_until_complete(_run_all())
        except RuntimeError:
            verified = asyncio.run(_run_all())

        # Filter: keep only datasets where pubmed_keep=True
        kept = [d for d in verified if d.get("pubmed_keep", True)]
        rejected = [d for d in verified if not d.get("pubmed_keep", True)]

        n_with_pmid = sum(1 for d in verified if d.get("pubmed_verified"))
        n_no_pmid = sum(1 for d in verified if "no_pubmed_link" in (d.get("notes") or ""))
        n_error = sum(1 for d in verified if "verify_error" in (d.get("notes") or ""))

        logger.info(
            f"pubmed_verify: {len(datasets)} total → "
            f"{n_with_pmid} verified, {n_no_pmid} no-PMID, {n_error} errors | "
            f"{len(kept)} kept, {len(rejected)} rejected"
        )
        for d in rejected:
            logger.info(
                f"  Rejected {d.get('accession','?')} "
                f"(PMID={d.get('paper_pmid','?')}): "
                f"reason={d.get('reason','')[:100]}"
            )

        return kept

    # ------------------------------------------------------------------ #
    #  TCGA search methods                                                 #
    # ------------------------------------------------------------------ #

    def _search_tcga(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search TCGA/GDC using the parsed intent."""
        # TCGA only contains tumor tissue and adjacent normal samples.
        # If the user wants cfDNA/plasma/serum/WBC, skip TCGA entirely.
        sample_type = intent.get("sample_type")
        tcga_compatible_types = {"tumor", "adjacent", "normal", "non_cancer", None}
        if sample_type and sample_type not in tcga_compatible_types:
            logger.info(
                f"Skipping TCGA search: sample_type='{sample_type}' is not available in TCGA "
                f"(TCGA only has tumor/adjacent normal tissue)"
            )
            return []

        cancer_type = intent.get("cancer_type", {})
        if isinstance(cancer_type, dict):
            cancer_code = cancer_type.get("tcga_code")
        else:
            cancer_code = cancer_type

        platform = intent.get("platform")
        year_start = intent.get("year_start")
        year_end = intent.get("year_end")

        try:
            files = self.gdc_client.search_methylation_files(
                cancer_type_code=cancer_code,
                platform=platform,
                year_start=year_start,
                year_end=year_end,
                max_results=500,
            )
            if not files:
                return []

            return self.gdc_client.files_to_dataset_records(files, cancer_code or "UNKNOWN")
        except Exception as e:
            logger.error(f"TCGA search failed: {e}")
            return []

    def _fetch_tcga_by_accessions(self, accessions: List[str]) -> List[Dict[str, Any]]:
        """Fetch TCGA project info for explicit TCGA accessions."""
        results = []
        for acc in accessions:
            try:
                project_id = acc if acc.startswith("TCGA-") else f"TCGA-{acc}"
                summary = self.gdc_client.get_project_summary(project_id)
                if summary:
                    results.append({
                        "accession": project_id,
                        "source": "TCGA",
                        "title": summary.get("name", project_id),
                        "cancer_type": acc.replace("TCGA-", ""),
                        "platform": None,
                        "data_type": "array",
                        "sample_count": None,
                        "year": None,
                        "file_ids": [],
                    })
            except Exception as e:
                logger.error(f"Failed to fetch TCGA info for {acc}: {e}")
        return results

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def _generate_summary_message(
        self,
        candidates: List,
        new_candidates: List,
        downloaded: List,
        failed: List,
        skipped: List,
    ) -> str:
        return (
            f"DatabaseAgent completed.\n"
            f"  Candidates found: {len(candidates)} "
            f"(GEO: {sum(1 for c in candidates if c.get('source') == 'GEO')}, "
            f"TCGA: {sum(1 for c in candidates if c.get('source') == 'TCGA')})\n"
            f"  Already in registry (skipped): {len(skipped)}\n"
            f"  New datasets registered: {len(new_candidates)}\n"
            f"  Successfully downloaded: {len(downloaded)}\n"
            f"  Failed: {len(failed)}\n"
            f"  Downloaded accessions: {downloaded}\n"
            f"  Failed accessions: {failed}"
        )
