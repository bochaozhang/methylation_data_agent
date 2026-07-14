"""
geo-search skill — GEO candidate recall (deterministic).

Owns the synonym data (skills/geo_search/synonyms.yaml) and the deterministic
query construction (delegates to tools.parser_tools.build_geo_search_string).
Does NOT judge usability — that is geo-filter's job. GEO-only; TCGA is a
separate module.

Input  (state): parsed_intent (SearchIntent)
Output (state): candidate_gse_list (full filter_methylation_datasets dicts),
                search_queries, search_log
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from tools.geo_tools import GEOClient
from tools.parser_tools import build_geo_search_string
from utils.logger import get_logger

logger = get_logger(__name__)


class SearchSkill:
    """GEO recall skill. Deterministic — no LLM call."""

    name = "geo-search"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ncbi_key = os.environ.get(config.get("geo", {}).get("api_key_env", ""), "") or None
        ncbi_proxy = (
            os.environ.get("NCBI_PROXY", "")
            or config.get("geo", {}).get("proxy", "")
            or None
        )
        self.geo_client = GEOClient(api_key=ncbi_key or None, proxy=ncbi_proxy or None)

    # ------------------------------------------------------------------ #

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent = state.get("parsed_intent") or {}
        mode = intent.get("mode", "semantic")
        geo_acc = (intent.get("accessions") or {}).get("geo", [])

        if mode == "accession" and geo_acc:
            candidates = self._fetch_by_accessions(geo_acc)
            search_queries = [f"accession mode: {geo_acc}"]
            logger.info(f"geo-search[accession]: {len(candidates)} datasets from {geo_acc}")
        else:
            query = build_geo_search_string(intent)
            search_queries = [query] if query else []
            candidates = self._semantic_search(intent, query)
            logger.info(f"geo-search[semantic]: query len={len(query)}, {len(candidates)} candidates")

        self._inject_cancer_type(candidates, intent)

        return {
            "candidate_gse_list": candidates,
            "search_queries": search_queries,
            "search_log": (
                f"geo-search mode={mode}, queries={len(search_queries)}, "
                f"candidates={len(candidates)}"
            ),
        }

    # ------------------------------------------------------------------ #

    def _semantic_search(self, intent: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        if not query:
            return []
        try:
            accessions = self.geo_client.search_gse(query, max_results=2000)
            if not accessions:
                return []
            return self.geo_client.filter_methylation_datasets(
                accessions,
                platform_filter=intent.get("platform"),
                year_start=intent.get("year_start"),
                year_end=intent.get("year_end"),
            )
        except Exception as e:
            logger.error(f"geo-search semantic failed: {e}")
            return []

    def _fetch_by_accessions(self, accessions: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for acc in accessions:
            try:
                meta = self.geo_client.get_series_metadata(acc)
                if not meta.get("error"):
                    results.append(meta)
            except Exception as e:
                logger.error(f"geo-search fetch {acc} failed: {e}")
        return results

    @staticmethod
    def _inject_cancer_type(candidates: List[Dict[str, Any]], intent: Dict[str, Any]) -> None:
        """Stamp the requested cancer_type onto each candidate (if absent)."""
        ct = intent.get("cancer_type")
        label = None
        if isinstance(ct, dict):
            label = ct.get("display")
        if not label:
            return
        for d in candidates:
            if not d.get("cancer_type"):
                d["cancer_type"] = label
