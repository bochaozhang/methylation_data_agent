"""
TCGA recall module (NOT a skill) — parallel GEO branch, search-only.

Per design: TCGA is trivial to search (one cancer keyword) and its Level-3 beta
values are standard usable data, so it bypasses geo-filter entirely — match the
cancer keyword → GDC search → return candidates. NO download here (downloads are
unified in the daemon after the bulk "待下载" confirm). Liquid-biopsy requests
(cfDNA/plasma/serum/WBC) skip TCGA entirely (TCGA only has tumor/adjacent/normal
tissue).

Input  (state): parsed_intent
Output (state): tcga_candidates (dataset records), tcga_log
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from tools.tcga_tools import GDCClient
from utils.logger import get_logger

logger = get_logger(__name__)

# Sample types TCGA cannot serve (it only has tumor/adjacent/normal tissue).
_LIQUID_TYPES = {"cfdna", "plasma", "serum", "wbc", "whole_blood"}


def run_tcga_direct(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Search GDC by cancer keyword and return candidate dataset records (no download)."""
    intent = state.get("parsed_intent") or {}

    # Resolve cancer code.
    ct = intent.get("cancer_type")
    cancer_code = ct.get("tcga_code") if isinstance(ct, dict) else None
    if not cancer_code:
        return {"tcga_candidates": [], "tcga_log": "tcga-direct: no cancer_type.tcga_code, skipped"}

    # Skip liquid-biopsy requests — TCGA has no cfDNA/plasma/serum/WBC.
    sample_type = intent.get("sample_type")
    if sample_type in _LIQUID_TYPES:
        return {"tcga_candidates": [],
                "tcga_log": f"tcga-direct: skipped (sample_type={sample_type} not in TCGA)"}

    gdc_token = os.environ.get(config.get("tcga", {}).get("gdc_token_env", ""), "") or None
    gdc = GDCClient(token=gdc_token or None)

    try:
        files = gdc.search_methylation_files(
            cancer_type_code=cancer_code,
            platform=intent.get("platform"),
            year_start=intent.get("year_start"),
            year_end=intent.get("year_end"),
            max_results=500,
        )
    except Exception as e:
        logger.error(f"tcga-direct: GDC search failed: {e}")
        return {"tcga_candidates": [], "tcga_log": f"tcga-direct: GDC search failed: {e}"}
    if not files:
        return {"tcga_candidates": [], "tcga_log": f"tcga-direct: no GDC files for {cancer_code}"}

    candidates = gdc.files_to_dataset_records(files, cancer_code)
    for c in candidates:
        c["source"] = "TCGA"
    logger.info(f"tcga-direct: {len(candidates)} candidate record(s) for {cancer_code}")

    return {
        "tcga_candidates": candidates,
        "tcga_log": f"tcga-direct: cancer={cancer_code}, {len(candidates)} candidate(s)",
    }
