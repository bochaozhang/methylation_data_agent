"""
Agent 1: DatabaseAgent

Searches TCGA (GDC) and GEO databases for methylation datasets
matching the parsed user intent, then downloads them.

Operates as a LangGraph node (function that takes/returns MethyAgentState).
Uses a ReAct-style loop internally: search → filter → register → download.

Sample type filtering:
  When the user specifies a sample type (e.g. cfDNA, plasma, WBC),
  DatabaseAgent applies a two-stage filter:
    1. Search query: adds sample type terms to GEO/TCGA search string
    2. Post-retrieval: filters metadata results by sample type keywords
       in title/summary, removing datasets that clearly don't match
       (e.g. tumor tissue datasets when user asked for cfDNA).
"""
import os
import time
import xml.etree.ElementTree as ET

import requests
from typing import Any, Dict, List, Optional

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
        self.geo_client = GEOClient(api_key=ncbi_key or None)

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

        # ---- Step 2b: Paper-based verification for datasets with no sample type signal ----
        # For GEO datasets where title+summary+sample_titles gave no sample type signal
        # (detected_sample_types is empty), fetch the linked PubMed abstract and check
        # for wanted/excluded keywords. This rescues false negatives like GSE124600
        # where the GEO summary omits sample type info but the paper abstract mentions plasma.
        sample_type = intent.get("sample_type")
        sample_types = intent.get("sample_types", [])
        if sample_type or sample_types:
            candidates = self._verify_candidates_via_papers(candidates, sample_type, sample_types)

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
        """Search GEO using the parsed intent."""
        from tools.parser_tools import build_geo_search_string

        # Always use build_geo_search_string to ensure methylation platform filters
        # (GPL13534/GPL21145/etc.) are included. LLM-provided geo_search_query is
        # intentionally ignored here because it typically omits GPL filters, causing
        # the search to return RNA-seq and other non-methylation datasets.
        search_query = build_geo_search_string(intent)
        if not search_query:
            return []

        try:
            accessions = self.geo_client.search_gse(search_query, max_results=100)
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
                # Rule-based parser: use canonical English name
                cancer_label = TCGA_CODE_TO_ENGLISH[intent["cancer_type_code"]]
                for d in datasets:
                    if not d.get("cancer_type"):
                        d["cancer_type"] = cancer_label

            # Apply sample type post-retrieval filter
            sample_type = intent.get("sample_type")
            sample_types = intent.get("sample_types", [])
            if sample_type or sample_types:
                datasets = self._filter_by_sample_type(datasets, sample_type, sample_types)

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
    #  Sample type filtering                                               #
    # ------------------------------------------------------------------ #

    def _filter_by_sample_type(
        self,
        datasets: List[Dict[str, Any]],
        primary_sample_type: Optional[str] = None,
        sample_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Post-retrieval filter: remove datasets whose metadata clearly indicates
        an incompatible sample type.

        Three-layer verification strategy:
          Layer 1 — title + summary keyword matching (fast, no extra API calls)
          Layer 2 — sample_titles keyword matching (GSM titles from esummary,
                    already fetched; resolves ambiguous cases where summary
                    mentions cfDNA as research topic but samples are tissue)
          Layer 3 — in-vitro exclusion (cell lines, organoids, supernatant)
                    applied globally regardless of wanted/excluded sets

        Decision logic:
          - If Layer 3 in-vitro signal found → exclude (cell lines are never
            valid cfDNA/plasma/tissue patient samples)
          - If Layer 1 summary matches ONLY excluded type → exclude
          - If Layer 1 summary matches BOTH wanted and excluded (ambiguous):
              → check Layer 2 sample_titles
              → if sample_titles match excluded → exclude
              → if sample_titles match wanted or are ambiguous → keep
          - If no signal → keep (conservative, prefer false positives)

        Args:
            datasets: List of GEO metadata dicts (must include 'sample_titles').
            primary_sample_type: The main sample type the user wants.
            sample_types: All sample types mentioned in the query.

        Returns:
            Filtered list of datasets, each tagged with 'detected_sample_types'.
        """
        if not primary_sample_type and not sample_types:
            return datasets

        # Build wanted and excluded sets
        wanted = set(sample_types or [])
        if primary_sample_type:
            wanted.add(primary_sample_type)
            wanted.update(SAMPLE_TYPE_RELATED.get(primary_sample_type, set()))

        all_types = {
            "tumor", "adjacent", "normal", "non_cancer",
            "wbc", "cfdna", "plasma", "serum", "whole_blood",
        }
        excluded = all_types - wanted

        # Keywords for each sample type (for text matching)
        sample_type_keywords = {
            "tumor": [
                "tumor", "tumour", "cancer tissue", "primary tumor", "malignant",
                # "genomic dna" is a strong signal for tissue/cell gDNA extraction,
                # as opposed to cfDNA which is fragmented cell-free DNA from plasma.
                "genomic dna", "gdna", "tissue dna", "dna from tissue",
                "formalin-fixed", "ffpe", "fresh frozen tissue",
                "肿瘤组织", "癌组织", "癌症组织", "原发灶", "基因组dna",
            ],
            "adjacent": ["adjacent normal", "paratumor", "peritumoral", "margin",
                         "癌旁", "旁组织"],
            "normal": ["normal tissue", "healthy tissue", "healthy control",
                       "正常组织", "健康组织"],
            "non_cancer": ["non-cancer", "benign", "noncancerous", "control tissue",
                           "非癌", "良性"],
            "wbc": ["wbc", "leukocyte", "buffy coat", "pbmc",
                    "peripheral blood mononuclear", "白细胞", "血细胞", "外周血单个核"],
            "cfdna": ["cfdna", "cell-free dna", "circulating dna", "ctdna",
                      "游离dna", "循环dna", "循环肿瘤dna"],
            "plasma": ["plasma", "blood plasma", "血浆"],
            "serum": ["serum", "blood serum", "血清"],
            "whole_blood": ["whole blood", "全血"],
        }

        # Layer 3 keywords: in-vitro models are never valid patient samples.
        # Applied globally — a dataset from cell lines or organoids is excluded
        # regardless of whether cfDNA/tissue is mentioned in the summary.
        IN_VITRO_KEYWORDS = [
            "cell line", "cell lines", "cancer cell", "cancer cells",
            "organoid", "organoids", "patient-derived organoid",
            "in vitro", "supernatant", "culture medium",
            "细胞系", "类器官", "体外",
        ]

        def _matches_keywords(text: str, kw_list: List[str]) -> bool:
            return any(kw in text for kw in kw_list)

        def _check_types(text: str, type_set) -> Optional[str]:
            """Return first matching type from type_set, or None."""
            for t in type_set:
                if _matches_keywords(text, sample_type_keywords.get(t, [])):
                    return t
            return None

        filtered = []
        for ds in datasets:
            acc = ds.get("accession", "?")
            title = (ds.get("title") or "").lower()
            summary = (ds.get("summary") or "").lower()
            # Include sample_titles in layer1_text so that informative GSM titles
            # (e.g. "genomic DNA from CRC patient") are checked alongside summary.
            # This resolves cases where summary mentions cfDNA as a research topic
            # but the actual samples are tissue gDNA.
            sample_titles = ds.get("sample_titles") or []
            sample_titles_text = " ".join(t.lower() for t in sample_titles)
            layer1_text = title + " " + summary + " " + sample_titles_text

            # ---- Layer 3: in-vitro exclusion (highest priority) ----
            if _matches_keywords(layer1_text, IN_VITRO_KEYWORDS):
                logger.info(
                    f"Sample type filter [L3-vitro]: excluding {acc} "
                    f"— in-vitro model detected, title: {title[:80]}"
                )
                continue

            # ---- Layer 1: title + summary + sample_titles keyword matching ----
            excluded_match = _check_types(layer1_text, excluded)
            wanted_match = _check_types(layer1_text, wanted)

            if excluded_match and not wanted_match:
                # Clearly excluded, no wanted signal
                logger.info(
                    f"Sample type filter [L1]: excluding {acc} "
                    f"— matches excluded '{excluded_match}', title: {title[:80]}"
                )
                continue

            if excluded_match and wanted_match:
                # Ambiguous — escalate to Layer 2: sample titles
                # (sample_titles_text already computed above)
                if sample_titles_text:
                    titles_excluded = _check_types(sample_titles_text, excluded)
                    titles_wanted = _check_types(sample_titles_text, wanted)

                    if titles_excluded and not titles_wanted:
                        # Sample titles confirm excluded type — reject
                        logger.info(
                            f"Sample type filter [L2]: excluding {acc} "
                            f"— summary ambiguous but sample titles match "
                            f"excluded '{titles_excluded}': {sample_titles[:3]}"
                        )
                        continue
                    elif titles_wanted:
                        logger.debug(
                            f"Sample type filter [L2]: keeping {acc} "
                            f"— sample titles confirm wanted '{titles_wanted}'"
                        )
                    else:
                        # Sample titles also ambiguous — keep (conservative)
                        logger.debug(
                            f"Sample type filter [L2]: keeping {acc} "
                            f"— sample titles inconclusive: {sample_titles[:3]}"
                        )
                else:
                    # No sample titles available — keep (conservative)
                    logger.debug(
                        f"Sample type filter [L1-ambiguous]: keeping {acc} "
                        f"— no sample titles to resolve ambiguity"
                    )

            # Tag the dataset with detected sample types for downstream use
            detected_types = []
            for st_type, keywords in sample_type_keywords.items():
                if _matches_keywords(layer1_text, keywords):
                    detected_types.append(st_type)
            ds["detected_sample_types"] = detected_types
            if not ds.get("sample_type") and detected_types:
                ds["sample_type"] = detected_types[0]

            filtered.append(ds)

        logger.info(
            f"Sample type filter: {len(filtered)}/{len(datasets)} datasets passed "
            f"(wanted={wanted}, excluded={excluded})"
        )
        return filtered

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

    # ------------------------------------------------------------------ #
    #  Paper-based sample type verification                                #
    # ------------------------------------------------------------------ #

    def _verify_candidates_via_papers(
        self,
        candidates: List[Dict[str, Any]],
        primary_sample_type: Optional[str],
        sample_types: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        For GEO datasets with no sample type signal (detected_sample_types is empty),
        fetch the linked PubMed abstract and verify sample type from the abstract text.

        This rescues false negatives where the GEO summary omits sample type info
        but the paper abstract clearly mentions the sample material (e.g. GSE124600:
        GEO summary is method-focused but abstract says "plasma samples").

        Only runs for datasets that:
          1. Are GEO datasets (source == "GEO")
          2. Have detected_sample_types == [] (no signal from title/summary/titles)
          3. Have pubmed_ids available in metadata

        Datasets with existing sample type signal are returned unchanged.
        """
        result = []
        for ds in candidates:
            # Only verify GEO datasets with no sample type signal
            if ds.get("source") != "GEO" or ds.get("detected_sample_types"):
                result.append(ds)
                continue

            pubmed_ids = ds.get("pubmed_ids") or []
            if not pubmed_ids:
                # No PMID — keep conservatively
                result.append(ds)
                continue

            acc = ds.get("accession", "?")
            decision, reason = self._verify_dataset_via_paper(
                acc, pubmed_ids, primary_sample_type, sample_types
            )

            if decision == "excluded":
                logger.info(
                    f"Sample type filter [paper-verify]: excluding {acc} "
                    f"— paper abstract confirms excluded type: {reason}"
                )
                continue
            elif decision == "wanted":
                logger.info(
                    f"Sample type filter [paper-verify]: keeping {acc} "
                    f"— paper abstract confirms wanted type: {reason}"
                )
                ds["detected_sample_types"] = [reason]
                if not ds.get("sample_type"):
                    ds["sample_type"] = reason
            else:
                # Unknown — keep conservatively
                logger.debug(
                    f"Sample type filter [paper-verify]: keeping {acc} "
                    f"— paper abstract inconclusive"
                )

            result.append(ds)

        return result

    def _verify_dataset_via_paper(
        self,
        accession: str,
        pubmed_ids: List[str],
        primary_sample_type: Optional[str],
        sample_types: Optional[List[str]],
    ) -> tuple:
        """
        Fetch the PubMed abstract for the first available PMID and check
        whether it confirms a wanted or excluded sample type.

        Uses the same keyword lists as _filter_by_sample_type for consistency.

        Args:
            accession: GEO accession (for logging).
            pubmed_ids: List of PMIDs linked to this dataset.
            primary_sample_type: Primary sample type from user intent.
            sample_types: All sample types from user intent.

        Returns:
            Tuple of (decision, reason) where:
              decision: "wanted" | "excluded" | "unknown"
              reason: matched sample type string or "" for unknown
        """
        # Build wanted/excluded sets (same logic as _filter_by_sample_type)
        wanted = set(sample_types or [])
        if primary_sample_type:
            wanted.add(primary_sample_type)
            wanted.update(SAMPLE_TYPE_RELATED.get(primary_sample_type, set()))

        all_types = {
            "tumor", "adjacent", "normal", "non_cancer",
            "wbc", "cfdna", "plasma", "serum", "whole_blood",
        }
        excluded = all_types - wanted

        sample_type_keywords = {
            "tumor": [
                "tumor", "tumour", "cancer tissue", "primary tumor", "malignant",
                "genomic dna", "gdna", "tissue dna", "formalin-fixed", "ffpe",
                "fresh frozen tissue", "肿瘤组织", "癌组织", "基因组dna",
            ],
            "adjacent": ["adjacent normal", "paratumor", "peritumoral", "癌旁"],
            "normal": ["normal tissue", "healthy tissue", "正常组织"],
            "non_cancer": ["non-cancer", "benign", "noncancerous", "非癌", "良性"],
            "wbc": ["wbc", "leukocyte", "buffy coat", "pbmc",
                    "peripheral blood mononuclear", "白细胞"],
            "cfdna": ["cfdna", "cell-free dna", "circulating dna", "ctdna",
                      "游离dna", "循环dna"],
            "plasma": ["plasma", "blood plasma", "血浆"],
            "serum": ["serum", "blood serum", "血清"],
            "whole_blood": ["whole blood", "全血"],
        }

        pmid = str(pubmed_ids[0])
        try:
            # Fetch PubMed abstract via efetch
            time.sleep(0.35)  # NCBI rate limit (no API key)
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "xml",
                    "rettype": "abstract",
                },
                timeout=20,
            )
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            abstract_parts = root.findall(".//AbstractText")
            abstract = " ".join(
                (p.text or "") for p in abstract_parts if p.text
            ).lower()

            if not abstract:
                logger.debug(
                    f"Paper verify {accession} PMID {pmid}: no abstract available"
                )
                return ("unknown", "")

            # Check wanted types first (positive signal).
            # Use a fixed priority order so the most specific type wins
            # (e.g. "cfdna" before "plasma" before "non_cancer") regardless
            # of Python set iteration order.
            WANTED_PRIORITY = ["cfdna", "plasma", "serum", "wbc", "whole_blood",
                               "tumor", "adjacent", "normal", "non_cancer"]
            for st in WANTED_PRIORITY:
                if st not in wanted:
                    continue
                for kw in sample_type_keywords.get(st, []):
                    if kw in abstract:
                        return ("wanted", st)

            # Check excluded types
            EXCLUDED_PRIORITY = ["tumor", "adjacent", "normal", "non_cancer",
                                  "wbc", "whole_blood", "serum", "plasma", "cfdna"]
            for st in EXCLUDED_PRIORITY:
                if st not in excluded:
                    continue
                for kw in sample_type_keywords.get(st, []):
                    if kw in abstract:
                        return ("excluded", st)

            return ("unknown", "")

        except Exception as e:
            logger.debug(
                f"Paper verify {accession} PMID {pmid}: fetch failed ({e})"
            )
            return ("unknown", "")
