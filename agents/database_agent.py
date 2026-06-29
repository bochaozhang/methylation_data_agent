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

# GEO metadata screening prompt — Step 1 of the two-step filter.
# Screens datasets using ONLY GEO metadata (title, summary, overall_design).
# No PubMed abstract needed. Runs before PubMed verification to eliminate
# obvious mismatches cheaply (cell lines, wrong cancer type, wrong sample type).
# Kept CONSTANT for Z.AI system cache.
LLM_SCREEN_SYSTEM_PROMPT = """You are a biomedical data curator specializing in DNA methylation datasets.

You will receive:
1. The user's search intent (requested cancer type, sample type, platform)
2. GEO dataset metadata: accession, title, summary, overall design, platform, sample count

Your task: decide whether this dataset is a plausible match for the user's request,
based ONLY on the GEO metadata (no paper abstract available yet).

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{
  "keep": true or false,
  "detected_cancer_type": "canonical English cancer name, or null if unclear",
  "detected_sample_type": "plasma|tumor|adjacent|normal|wbc|cfdna|serum|whole_blood|cell_line|unknown",
  "reason": "one sentence explaining the decision"
}

Decision rules — evaluate in ORDER, stop at the first matching rule:

RULE 1 (HARD REJECT — cell lines / in-vitro, no exceptions):
- keep=false if the dataset is clearly cell lines / organoids / in-vitro models
  * Triggered by: "cell line", "cell culture", "organoid", "in vitro", "in-vitro",
    "cultured cells", specific cell line names (HCT116, SW480, MCF7, A549, etc.)
  * Applies regardless of cancer type or any other factor

RULE 2 (HARD REJECT — data type mismatch):
- keep=false if the data is clearly NOT DNA methylation
  * e.g. title/summary mentions RNA-seq, transcriptome, proteomics, ChIP-seq, ATAC-seq

RULE 3 (HARD REJECT — cancer type mismatch):
- keep=false if the dataset cancer type clearly does NOT match the requested cancer type
  * e.g. requested colorectal cancer but title/summary clearly describes lung / breast / leukemia
  * Minor subtype differences are acceptable (e.g. "colon adenocarcinoma" ≈ "colorectal cancer")
  * Skip this rule if requested cancer type is null/unknown
  * When uncertain (cancer type not mentioned in metadata), keep=true

RULE 4 (HARD REJECT — sample type mismatch):
- keep=false if the dataset sample type clearly does NOT match the requested sample type
  * e.g. requested cfDNA/plasma but title/summary clearly describes tumor tissue only
  * e.g. requested cfDNA/plasma but summary clearly describes WBC / whole blood only
  * Skip this rule if requested sample type is null/unknown
  * When uncertain (sample type not mentioned in metadata), keep=true

RULE 5 (DEFAULT):
- keep=true if none of rules 1-4 triggered
- Be LIBERAL at this stage — false positives are acceptable here because
  PubMed abstract verification (Step 2) will do a more rigorous check.
  Only reject when the mismatch is OBVIOUS from the GEO metadata alone.
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

Decision rules for `keep` — evaluate in ORDER, stop at the first matching rule:

RULE 1 (HARD REJECT — highest priority, no exceptions):
- keep=false if the dataset is cell lines / organoids / in-vitro models
    * This applies regardless of cancer type match or any other factor
    * Triggered when: GEO sample type is "cell_line", OR abstract mentions "cell line", "cell culture",
      "organoid", "in vitro", "in-vitro", "cultured cells" as the primary biological material
    * Example: "colon cancer cell lines HCT116 and SW480" → keep=false even if cancer type matches

RULE 2 (HARD REJECT — data type mismatch):
- keep=false if the data is NOT DNA methylation
    * e.g. abstract confirms RNA-seq, proteomics, ChIP-seq, ATAC-seq, WGS (non-bisulfite)

RULE 3 (HARD REJECT — cancer type mismatch):
- keep=false if the dataset cancer type clearly does NOT match the requested cancer type
    * e.g. requested colorectal cancer but dataset is lung cancer / breast cancer / leukemia
    * Minor subtype differences are acceptable (e.g. "colon adenocarcinoma" ≈ "colorectal cancer")
    * Skip this rule if requested cancer type is null/unknown

RULE 4 (HARD REJECT — sample type mismatch):
- keep=false if the dataset sample type clearly does NOT match the requested sample type
    * e.g. requested cfDNA/plasma but dataset is tumor tissue only (abstract confirms no liquid biopsy)
    * e.g. requested cfDNA/plasma but dataset is WBC / whole blood only
    * Skip this rule if requested sample type is null/unknown

RULE 5 (DEFAULT — only if none of the above triggered):
- keep=true if none of rules 1-4 triggered
- keep=true if the abstract is unavailable or uninformative (benefit of the doubt)
- When genuinely uncertain about rules 3 or 4, keep=true is acceptable
- Rules 1 and 2 are never uncertain — cell lines and non-methylation data are always keep=false

Rules for other fields:
- confirmed_sample_type: use the abstract's description of biological material
- sample_count_in_paper: extract the n= number for the main patient cohort; null if not stated
- accession_mentioned: true only if the GEO accession (e.g. GSE220160) appears explicitly in the abstract
- consistency: consistent if GEO and abstract agree; minor_discrepancy if small differences; major_discrepancy if fundamental mismatch
- recommended_action: skip only if keep=false with high confidence; review if minor discrepancy; download otherwise
- notes: record any discrepancy (e.g. sample count mismatch, different cancer subtype, cancer type mismatch with request)
"""

# GSM-level include/exclude judge prompt — Call 1 of the two-step sample metadata pipeline.
# Input: intent + single GSM characteristics. Output: include bool + reason.
# Kept CONSTANT for Z.AI system cache.
LLM_GSM_JUDGE_SYSTEM_PROMPT = """You are a biomedical data curator specializing in DNA methylation datasets.

You will receive:
1. The user's search intent (cancer type, sample type, query detail)
2. A single GEO sample (GSM) with its source_name, molecule, and characteristics

Your task: decide whether this individual sample matches the user's requested sample type.

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{
  "include": true or false,
  "reason": null or "one sentence explaining why excluded"
}

Rules:
- include=true  if the sample matches the requested sample type
  * reason MUST be null when include=true
- include=false if the sample clearly does NOT match
  * reason MUST be a concise explanation (e.g. "tumor tissue, not cfDNA")
- When in doubt (ambiguous characteristics), include=true (conservative)
- Focus ONLY on sample type match — do not judge cancer type or platform here
"""

# Dataset-level keep/unsure/reject prompt — Call 2 of the two-step sample metadata pipeline.
# Input: include/exclude statistics + GEO summary. Output: dataset_keep tri-state.
# Kept CONSTANT for Z.AI system cache.
LLM_DATASET_KEEP_SYSTEM_PROMPT = """You are a biomedical data curator specializing in DNA methylation datasets.

You will receive:
1. The user's search intent (cancer type, sample type, query detail)
2. GEO dataset summary (title, summary, overall_design)
3. Sample-level statistics: how many samples were judged include vs exclude

Your task: decide the fate of this dataset based on the statistics and GEO summary.

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{
  "dataset_keep": "true" | "false" | "unsure",
  "reason": "one sentence explaining the decision"
}

Decision rules — evaluate in ORDER, stop at the first matching rule:

RULE 1 — dataset_keep="false" (clear reject):
- include_fraction is very low (< 5%) AND GEO summary confirms the dataset is not the requested type
- e.g. requested cfDNA/plasma but >95% samples are tumor tissue and summary confirms tissue study

RULE 2 — dataset_keep="true" (clear accept):
- include_fraction is reasonably high (>= 20%) AND GEO summary is consistent with the request
- e.g. requested cfDNA/plasma, 60% samples are plasma cfDNA, summary mentions liquid biopsy

RULE 3 — dataset_keep="unsure" (needs PubMed confirmation):
- include_fraction is borderline (5–20%), OR
- GEO summary is ambiguous or contradicts the statistics, OR
- Characteristics fields were mostly missing/empty (statistics may be unreliable)

RULE 4 — DEFAULT:
- dataset_keep="unsure" if none of the above rules clearly apply
- Be conservative: prefer "unsure" over "false" when uncertain
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

        # ---- Step 4: Register as awaiting_approval ----
        # Downloads are NOT triggered here. Datasets are queued for human
        # approval in the Web UI. The daemon download loop picks them up
        # after the user confirms via POST /datasets/approve.
        queued = []
        for c in new_candidates:
            acc = c["accession"]
            _notes = c.get("notes") or ""
            _no_pubmed = "no_pubmed_link" in _notes
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
                paper_pmid=c.get("paper_pmid"),
                notes=c.get("notes"),
                no_pubmed_link=_no_pubmed,
                sample_metadata_path=c.get("sample_metadata_path"),
                download_status="awaiting_approval",
            )
            self.registry.log_event(
                acc, "queued",
                "Registered by DatabaseAgent — awaiting human approval to download"
            )
            queued.append(acc)

        # ---- Step 5: Summary message ----
        summary_msg = self._generate_summary_message(
            candidates, new_candidates, queued, skipped
        )

        return {
            **state,
            "db_candidates": candidates,
            "db_downloaded": [],
            "db_failed": [],
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

        Three-step filtering pipeline:
          1. esearch → GSE UID list (NCBI E-utilities keyword search)
          2. filter_methylation_datasets() → platform/year filter + batch esummary
          3. Inject cancer_type from intent
          4. Step 1 — GEO metadata screening (_geo_screen_datasets_concurrent):
               - LLM reads title/summary/overall_design vs intent
               - Drops obvious mismatches (cell lines, wrong cancer, wrong sample type)
               - No NCBI API calls needed; cheap and fast
          5. Step 2 — GSM sample metadata judge (_sample_metadata_judge_concurrent):
               - Full efetch all GSMs (no cap), cached to sample_metadata.csv
               - LLM Call 1: per-GSM include/exclude
               - LLM Call 2: dataset_keep (true/false/unsure) from statistics + GEO summary
               - true  → awaiting_approval (skip Step 3)
               - false → discard
               - unsure → Step 3
          6. Step 3 — PubMed cross-verification (only for unsure datasets):
               - Has PMID + abstract → LLM verify
               - No PMID or no abstract → pubmed_keep=False (discard)
        """
        from tools.parser_tools import build_geo_search_string

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

            logger.info(f"GEO: {len(datasets)} datasets after platform/year filter")

            # ---- Step 1: GEO metadata screening (no NCBI calls, LLM only) ----
            datasets = self._geo_screen_datasets_concurrent(datasets, intent=intent)
            logger.info(f"GEO Step 1: {len(datasets)} datasets passed metadata screen")

            # ---- Step 2: GSM sample metadata judge (full efetch + two LLM calls) ----
            keep_list, reject_list, unsure_list = self._sample_metadata_judge_concurrent(
                datasets, intent
            )
            logger.info(
                f"GEO Step 2: keep={len(keep_list)} reject={len(reject_list)} "
                f"unsure={len(unsure_list)}"
            )

            # ---- Step 3: PubMed cross-verification (only for unsure datasets) ----
            if unsure_list:
                pubmed_results = self._pubmed_verify_datasets_concurrent(
                    unsure_list, intent=intent
                )
                keep_list += [d for d in pubmed_results if d.get("pubmed_keep")]
                reject_list += [d for d in pubmed_results if not d.get("pubmed_keep")]
                logger.info(
                    f"GEO Step 3 (PubMed): {sum(1 for d in pubmed_results if d.get('pubmed_keep'))} "
                    f"kept from {len(unsure_list)} unsure"
                )

            logger.info(f"GEO final: {len(keep_list)} datasets → awaiting_approval")
            return keep_list
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
    #  Step 2: GSM sample metadata judge (two LLM calls)                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_sample_df(gsm_list: List[Dict[str, Any]]):
        """
        Build a pandas DataFrame from a list of GSM dicts returned by
        get_all_gsm_metadata(). Characteristics are expanded into columns.
        """
        import pandas as pd

        rows = []
        for g in gsm_list:
            row = {
                "gsm": g.get("gsm", ""),
                "source_name": g.get("source_name", ""),
                "molecule": g.get("molecule", ""),
                "group": g.get("group", "other"),
            }
            for k, v in (g.get("characteristics") or {}).items():
                row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def _judge_single_gsm(
        self,
        gsm_dict: Dict[str, Any],
        intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        LLM Call 1: judge a single GSM sample — include or exclude.

        Input: intent + one GSM's source_name, molecule, characteristics.
        Output: {"gsm": str, "include": bool, "reason": str|None}

        On any error: returns include=True (conservative).
        """
        import json as _json

        gsm_id = gsm_dict.get("gsm", "?")

        # Build intent block
        ct = intent.get("cancer_type")
        ct_label = (ct.get("display") if isinstance(ct, dict) else str(ct)) if ct else                    intent.get("cancer_type_display") or "not specified"
        sample_types = intent.get("sample_types") or []
        primary_st = intent.get("sample_type") or "not specified"
        st_label = f"{primary_st} (all: {sample_types})" if sample_types else primary_st
        detail = intent.get("sample_type_detail") or ""
        raw_query = intent.get("raw_query") or ""

        ch = gsm_dict.get("characteristics") or {}
        ch_str = "; ".join(f"{k}: {v}" for k, v in ch.items()) if ch else "(none)"

        user_msg = (
            f"=== USER REQUEST ===\n"
            f"Cancer type: {ct_label}\n"
            f"Sample type: {st_label}\n"
            + (f"Detail: {detail}\n" if detail else "")
            + (f"Query: {raw_query[:200]}\n" if raw_query else "")
            + f"\n=== GSM SAMPLE ===\n"
            f"GSM: {gsm_id}\n"
            f"source_name: {gsm_dict.get('source_name', '')}\n"
            f"molecule: {gsm_dict.get('molecule', '')}\n"
            f"characteristics: {ch_str}\n"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=LLM_GSM_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            v = _json.loads(raw)
            include = bool(v.get("include", True))
            reason = v.get("reason") if not include else None
            return {"gsm": gsm_id, "include": include, "reason": reason}
        except Exception as e:
            logger.debug(f"_judge_single_gsm({gsm_id}): LLM/parse error — {e} — including (conservative)")
            return {"gsm": gsm_id, "include": True, "reason": None}

    def _judge_all_gsms_concurrent(
        self,
        gsm_list: List[Dict[str, Any]],
        intent: Dict[str, Any],
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run LLM Call 1 (per-GSM include/exclude) concurrently for all GSMs.

        Uses asyncio + Semaphore(max_concurrent). Each call is short (~100-300 tokens).
        Returns list of {"gsm", "include", "reason"} dicts.
        """
        if not gsm_list:
            return []

        sem = asyncio.Semaphore(max_concurrent)

        async def _judge_one_async(g: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._judge_single_gsm, g, intent
                )

        async def _run_all() -> List[Dict[str, Any]]:
            tasks = [_judge_one_async(g) for g in gsm_list]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _run_all())
                    results = future.result()
            else:
                results = loop.run_until_complete(_run_all())
        except RuntimeError:
            results = asyncio.run(_run_all())

        n_include = sum(1 for r in results if r.get("include"))
        logger.info(
            f"_judge_all_gsms_concurrent: {n_include}/{len(results)} GSMs included"
        )
        return results

    def _judge_dataset_keep(
        self,
        ds: Dict[str, Any],
        gsm_verdicts: List[Dict[str, Any]],
        intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        LLM Call 2: decide dataset_keep (true/false/unsure) based on
        include/exclude statistics + GEO summary.

        Input: aggregated stats from gsm_verdicts + ds title/summary/overall_design.
        Output: {"dataset_keep": "true"|"false"|"unsure", "reason": str}

        On any error: returns dataset_keep="unsure" (conservative).
        """
        import json as _json

        acc = ds.get("accession", "?")
        n_total = len(gsm_verdicts)
        n_include = sum(1 for v in gsm_verdicts if v.get("include"))
        n_exclude = n_total - n_include
        include_frac = n_include / n_total if n_total > 0 else 0.0
        exclude_frac = n_exclude / n_total if n_total > 0 else 0.0

        # Build intent block
        ct = intent.get("cancer_type")
        ct_label = (ct.get("display") if isinstance(ct, dict) else str(ct)) if ct else                    intent.get("cancer_type_display") or "not specified"
        primary_st = intent.get("sample_type") or "not specified"
        detail = intent.get("sample_type_detail") or ""
        raw_query = intent.get("raw_query") or ""

        user_msg = (
            f"=== USER REQUEST ===\n"
            f"Cancer type: {ct_label}\n"
            f"Sample type: {primary_st}\n"
            + (f"Detail: {detail}\n" if detail else "")
            + (f"Query: {raw_query[:200]}\n" if raw_query else "")
            + f"\n=== DATASET ===\n"
            f"Accession: {acc}\n"
            f"Title: {ds.get('title', '')[:200]}\n"
            f"Summary: {ds.get('summary', '')[:500]}\n"
            f"Overall Design: {ds.get('overall_design', '')[:300]}\n"
            f"\n=== SAMPLE STATISTICS ===\n"
            f"Total samples: {n_total}\n"
            f"Include count: {n_include}  (include_fraction: {include_frac:.1%})\n"
            f"Exclude count: {n_exclude}  (exclude_fraction: {exclude_frac:.1%})\n"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=LLM_DATASET_KEEP_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            v = _json.loads(raw)
            dataset_keep = v.get("dataset_keep", "unsure")
            if dataset_keep not in ("true", "false", "unsure"):
                dataset_keep = "unsure"
            reason = v.get("reason", "")
            logger.info(
                f"_judge_dataset_keep({acc}): dataset_keep={dataset_keep} "
                f"include={n_include}/{n_total} ({include_frac:.1%}) reason={reason[:80]}"
            )
            return {"dataset_keep": dataset_keep, "reason": reason}
        except Exception as e:
            logger.warning(
                f"_judge_dataset_keep({acc}): LLM/parse error — {e} — returning unsure (conservative)"
            )
            return {"dataset_keep": "unsure", "reason": f"judge_error: {e}"}

    def _sample_metadata_judge(
        self,
        ds: Dict[str, Any],
        intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Full Step 2 pipeline for one dataset:
          1. Cache check: reuse existing sample_metadata.csv if present
          2. Full efetch all GSMs (get_all_gsm_metadata) if no cache
          3. LLM Call 1: per-GSM include/exclude (concurrent)
          4. Write query column to CSV (full, no truncation)
          5. LLM Call 2: dataset_keep based on statistics + GEO summary

        Returns ds enriched with:
          dataset_keep       ("true" | "false" | "unsure")
          sample_reason      (str)
          sample_metadata_path (str)
        """
        import pandas as pd
        from pathlib import Path

        acc = ds.get("accession", "?")
        data_dir = self.config.get("download", {}).get("output_dir", "/data")
        csv_path = Path(data_dir) / acc / "sample_metadata.csv"
        query_col = (intent.get("raw_query") or "")[:80]

        # ---- 1. Cache check ----
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Reconstruct gsm_list from CSV for LLM Call 1
                # Fixed columns that are NOT characteristics
                _fixed_cols = {"gsm", "source_name", "molecule", "group"}
                gsm_list = []
                for _, row in df.iterrows():
                    ch = {
                        k: str(v) for k, v in row.items()
                        if k not in _fixed_cols
                        and str(v) != "nan"
                        and len(k) <= 40  # skip old query cols (long names > 40 chars)
                    }
                    gsm_list.append({
                        "gsm": str(row.get("gsm", "")),
                        "source_name": str(row.get("source_name", "")),
                        "molecule": str(row.get("molecule", "")),
                        "group": str(row.get("group", "other")),
                        "characteristics": ch,
                    })
                logger.info(f"_sample_metadata_judge({acc}): reusing cached CSV ({len(gsm_list)} GSMs)")
            except Exception as e:
                logger.warning(f"_sample_metadata_judge({acc}): CSV read failed ({e}), re-efetching")
                gsm_list = self.geo_client.get_all_gsm_metadata(acc)
                df = self._build_sample_df(gsm_list)
        else:
            # ---- 2. Full efetch ----
            gsm_list = self.geo_client.get_all_gsm_metadata(acc)
            if not gsm_list:
                logger.warning(f"_sample_metadata_judge({acc}): no GSMs fetched — returning unsure")
                return {**ds, "dataset_keep": "unsure", "sample_reason": "no GSMs fetched",
                        "sample_metadata_path": None}
            df = self._build_sample_df(gsm_list)

        # ---- 3. LLM Call 1: per-GSM include/exclude (concurrent) ----
        gsm_verdicts = self._judge_all_gsms_concurrent(gsm_list, intent)

        # ---- 4. Write query column to CSV (full, no truncation) ----
        verdict_map = {v["gsm"]: v for v in gsm_verdicts}
        df[query_col] = df["gsm"].map(lambda g: (
            "include"
            if verdict_map.get(str(g), {}).get("include", True)
            else f"exclude: {verdict_map.get(str(g), {}).get('reason', 'unknown')}"
        ))
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"_sample_metadata_judge({acc}): wrote CSV → {csv_path}")
        except Exception as e:
            logger.warning(f"_sample_metadata_judge({acc}): CSV write failed: {e}")

        # ---- 5. LLM Call 2: dataset_keep ----
        keep_verdict = self._judge_dataset_keep(ds, gsm_verdicts, intent)

        return {
            **ds,
            "dataset_keep": keep_verdict["dataset_keep"],
            "sample_reason": keep_verdict["reason"],
            "sample_metadata_path": str(csv_path),
        }

    def _sample_metadata_judge_concurrent(
        self,
        datasets: List[Dict[str, Any]],
        intent: Dict[str, Any],
        max_concurrent: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run Step 2 (_sample_metadata_judge) on all datasets concurrently.
        Semaphore(3) because each call triggers many efetch requests.

        Returns:
            (keep_list, reject_list, unsure_list)
            keep_list   — dataset_keep="true"
            reject_list — dataset_keep="false"
            unsure_list — dataset_keep="unsure"
        """
        if not datasets:
            return [], [], []

        sem = asyncio.Semaphore(max_concurrent)

        async def _judge_one_async(ds: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._sample_metadata_judge, ds, intent
                )

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

        keep_list = [d for d in judged if d.get("dataset_keep") == "true"]
        reject_list = [d for d in judged if d.get("dataset_keep") == "false"]
        unsure_list = [d for d in judged if d.get("dataset_keep") == "unsure"]

        logger.info(
            f"Step 2 sample_metadata_judge: "
            f"keep={len(keep_list)} reject={len(reject_list)} unsure={len(unsure_list)} "
            f"/ total={len(datasets)}"
        )
        for d in reject_list:
            logger.info(
                f"  Step2-rejected {d.get('accession','?')}: {d.get('sample_reason','')[:80]}"
            )
        return keep_list, reject_list, unsure_list

    # ------------------------------------------------------------------ #
    #  Step 1: GEO metadata screening (no PubMed needed)                  #
    # ------------------------------------------------------------------ #

    def _geo_screen_dataset(self, ds: Dict[str, Any], intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Step 1 filter: screen a GEO dataset using ONLY its metadata
        (title, summary, overall_design) against the user intent.

        No NCBI API calls needed — uses only fields already fetched by
        filter_methylation_datasets(). Cheap and fast.

        Returns ds enriched with:
          screen_keep  (bool)  — False → drop before PubMed verification
          screen_reason (str)  — one-sentence explanation
          detected_sample_type (str)
          detected_cancer_type (str)
        """
        import json as _json

        acc = ds.get("accession", "?")

        # ---- Build intent block ----
        intent_lines = []
        if intent:
            ct = intent.get("cancer_type")
            ct_label = (ct.get("display") if isinstance(ct, dict) else str(ct)) if ct else                        intent.get("cancer_type_display") or "not specified"
            intent_lines.append(f"Requested cancer type: {ct_label}")

            sample_types = intent.get("sample_types") or []
            primary_st = intent.get("sample_type") or ""
            if sample_types:
                intent_lines.append(f"Requested sample type(s): {primary_st} (all: {sample_types})")
            elif primary_st:
                intent_lines.append(f"Requested sample type: {primary_st}")
            else:
                intent_lines.append("Requested sample type: not specified")

            platform_req = intent.get("platform") or "not specified"
            intent_lines.append(f"Requested platform: {platform_req}")

            detail = intent.get("sample_type_detail") or ""
            if detail:
                intent_lines.append(f"Sample type detail: {detail}")

        intent_block = "\n".join(intent_lines) if intent_lines else "not specified"

        user_msg = (
            f"=== USER REQUEST ===\n"
            f"{intent_block}\n\n"
            f"=== GEO DATASET METADATA ===\n"
            f"GEO Accession: {acc}\n"
            f"Title: {ds.get('title', '')[:200]}\n"
            f"Summary: {ds.get('summary', '')[:500]}\n"
            f"Overall Design: {ds.get('overall_design', '')[:400]}\n"
            f"Platform: {ds.get('platform_canonical') or ds.get('platforms', [])}\n"
            f"Sample count (GEO): {ds.get('sample_count')}\n"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=LLM_SCREEN_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            v = _json.loads(raw)

            updated = dict(ds)
            updated["screen_keep"] = bool(v.get("keep", True))
            updated["screen_reason"] = v.get("reason", "")
            if v.get("detected_sample_type"):
                updated["detected_sample_type"] = v["detected_sample_type"]
            if v.get("detected_cancer_type"):
                updated["detected_cancer_type"] = v["detected_cancer_type"]

            logger.info(
                f"geo_screen {acc}: keep={updated['screen_keep']} "
                f"cancer={v.get('detected_cancer_type','?')} "
                f"sample={v.get('detected_sample_type','?')} "
                f"reason={v.get('reason','')[:80]}"
            )
            return updated

        except Exception as e:
            logger.warning(f"geo_screen {acc}: LLM/parse error — {e} — keeping (conservative)")
            return {**ds, "screen_keep": True, "screen_reason": f"screen_error: {e}"}

    def _geo_screen_datasets_concurrent(
        self,
        datasets: List[Dict[str, Any]],
        max_concurrent: int = 5,
        intent: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run Step 1 GEO metadata screening on all datasets concurrently.
        Drops datasets where screen_keep=False before PubMed verification.
        """
        if not datasets:
            return datasets

        sem = asyncio.Semaphore(max_concurrent)

        async def _screen_one_async(ds: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._geo_screen_dataset, ds, intent
                )

        async def _run_all() -> List[Dict[str, Any]]:
            tasks = [_screen_one_async(ds) for ds in datasets]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _run_all())
                    screened = future.result()
            else:
                screened = loop.run_until_complete(_run_all())
        except RuntimeError:
            screened = asyncio.run(_run_all())

        kept = [d for d in screened if d.get("screen_keep", True)]
        rejected = [d for d in screened if not d.get("screen_keep", True)]

        logger.info(
            f"geo_screen (Step 1): {len(kept)}/{len(datasets)} passed, "
            f"{len(rejected)} rejected"
        )
        for d in rejected:
            logger.info(
                f"  Step1-rejected {d.get('accession','?')}: {d.get('screen_reason','')[:80]}"
            )
        return kept

    # ------------------------------------------------------------------ #
    #  Step 2: PubMed cross-verification                                  #
    # ------------------------------------------------------------------ #

    def _pubmed_verify_dataset(self, ds: Dict[str, Any], intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
          - No PMID available     → keep=False (unsure + no paper = discard)
          - Abstract fetch fails  → keep=False (unsure + no abstract = discard)
          - LLM/parse error       → keep=False (conservative reject for unsure datasets)

        Returns updated ds dict with added fields:
          pubmed_verified  (bool)
          pubmed_keep      (bool)  — used by _pubmed_verify_datasets_concurrent to filter
          paper_pmid       (str)
        """
        import json as _json

        acc = ds.get("accession", "?")
        pmids = ds.get("pubmed_ids", [])

        # ---- No PMID: discard (unsure + no paper = cannot confirm = don't download) ----
        if not pmids:
            existing_notes = ds.get("notes") or ""
            logger.info(
                f"pubmed_verify {acc}: no PMID — pubmed_keep=False (unsure, no paper to confirm)"
            )
            return {
                **ds,
                "pubmed_verified": False,
                "pubmed_keep": False,
                "notes": (existing_notes + "; no_pubmed_link").lstrip("; "),
            }

        pmid = str(pmids[0])
        abstract = self.geo_client.fetch_pubmed_abstract(pmid)

        # ---- Abstract unavailable: discard (unsure + no abstract = cannot confirm) ----
        if not abstract:
            existing_notes = ds.get("notes") or ""
            logger.info(
                f"pubmed_verify {acc}: abstract unavailable (PMID={pmid}) — "
                f"pubmed_keep=False (unsure, no abstract to confirm)"
            )
            return {
                **ds,
                "pubmed_verified": False,
                "pubmed_keep": False,
                "paper_pmid": pmid,
                "notes": (existing_notes + f"; abstract_unavailable(PMID={pmid})").lstrip("; "),
            }

        # ---- Build intent summary for LLM (what the user actually wants) ----
        intent_lines = []
        if intent:
            # Cancer type
            ct = intent.get("cancer_type")
            if ct:
                ct_label = ct.get("display") if isinstance(ct, dict) else str(ct)
            else:
                ct_label = intent.get("cancer_type_display") or "not specified"
            intent_lines.append(f"Requested cancer type: {ct_label}")

            # Sample types
            sample_types = intent.get("sample_types") or []
            primary_st = intent.get("sample_type") or ""
            if sample_types:
                intent_lines.append(f"Requested sample type(s): {primary_st} (all: {sample_types})")
            elif primary_st:
                intent_lines.append(f"Requested sample type: {primary_st}")
            else:
                intent_lines.append("Requested sample type: not specified")

            # Platform
            platform_req = intent.get("platform") or "not specified"
            intent_lines.append(f"Requested platform: {platform_req}")

            # Year range
            yr_start = intent.get("year_start")
            yr_end = intent.get("year_end")
            if yr_start or yr_end:
                intent_lines.append(f"Requested year range: {yr_start or 'any'} – {yr_end or 'any'}")

            # Free-text detail if available
            detail = intent.get("sample_type_detail") or ""
            if detail:
                intent_lines.append(f"Sample type detail: {detail}")

        intent_block = "\n".join(intent_lines) if intent_lines else "not specified"

        # ---- Build user message (dynamic content only) ----
        user_msg = (
            f"=== USER REQUEST ===\n"
            f"{intent_block}\n\n"
            f"=== GEO DATASET METADATA ===\n"
            f"GEO Accession: {acc}\n"
            f"Title: {ds.get('title', '')[:200]}\n"
            f"Summary: {ds.get('summary', '')[:400]}\n"
            f"Overall Design: {ds.get('overall_design', '')[:300]}\n"
            f"Platform: {ds.get('platform_canonical') or ds.get('platforms', [])}\n"
            f"Sample count (GEO): {ds.get('sample_count')}\n"
            f"Sample type (GEO): {ds.get('sample_type')}\n"
            f"Cancer type (GEO): {ds.get('cancer_type')}\n"
            f"PMID: {pmid}\n\n"
            f"=== PUBMED ABSTRACT ===\n"
            f"{abstract[:2500]}\n"
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
            logger.warning(f"pubmed_verify {acc}: LLM/parse error — {e} — rejecting (conservative for unsure)")
            existing_notes = ds.get("notes") or ""
            return {
                **ds,
                "pubmed_verified": False,
                "pubmed_keep": False,
                "notes": (existing_notes + f"; verify_error: {e}").lstrip("; "),
            }

    def _pubmed_verify_datasets_concurrent(
        self,
        datasets: List[Dict[str, Any]],
        max_concurrent: int = 5,
        intent: Optional[Dict[str, Any]] = None,
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
                return await loop.run_in_executor(
                    None, self._pubmed_verify_dataset, ds, intent
                )

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
        queued: List,
        skipped: List,
    ) -> str:
        return (
            f"DatabaseAgent completed.\n"
            f"  Candidates found: {len(candidates)} "
            f"(GEO: {sum(1 for c in candidates if c.get('source') == 'GEO')}, "
            f"TCGA: {sum(1 for c in candidates if c.get('source') == 'TCGA')})\n"
            f"  Already in registry (skipped): {len(skipped)}\n"
            f"  New datasets queued for approval: {len(queued)}\n"
            f"  Queued accessions: {queued}\n"
            f"  (Datasets will be downloaded after human approval in Web UI)"
        )
