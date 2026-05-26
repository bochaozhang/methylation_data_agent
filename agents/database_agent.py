"""
Agent 1: DatabaseAgent

Searches TCGA (GDC) and GEO databases for methylation datasets
matching the parsed user intent, then downloads them.

Operates as a LangGraph node (function that takes/returns MethyAgentState).
Uses a ReAct-style loop internally: search → filter → register → download.
"""
import os
from typing import Any, Dict, List

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

        search_query = intent.get("geo_search_query") or build_geo_search_string(intent)
        if not search_query:
            return []

        try:
            accessions = self.geo_client.search_gse(search_query, max_results=100)
            if not accessions:
                return []

            platform_filter = intent.get("platform")
            year_start = intent.get("year_start")
            year_end = intent.get("year_end")

            return self.geo_client.filter_methylation_datasets(
                accessions,
                platform_filter=platform_filter,
                year_start=year_start,
                year_end=year_end,
            )
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
    #  TCGA search methods                                                 #
    # ------------------------------------------------------------------ #

    def _search_tcga(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search TCGA/GDC using the parsed intent."""
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
