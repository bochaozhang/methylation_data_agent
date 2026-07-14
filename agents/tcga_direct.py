"""
TCGA direct module (NOT a skill) — parallel recall+download branch.

Per design: TCGA is trivial to search (one cancer keyword) and its Level-3 beta
values are standard usable data, so it bypasses geo-filter entirely — match the
cancer keyword → GDC search → download directly, no filter, no approval.

Run in parallel with the GEO chain (geo-search → geo-filter → geo-download);
both branches' results are merged into the registry by the pipeline's register
node. Liquid-biopsy requests (cfDNA/plasma/serum/WBC) skip TCGA entirely
(TCGA only has tumor/adjacent/normal tissue).

Input  (state): parsed_intent, output_dir
Output (state): tcga_results, tcga_log
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from tools.download_tools import DownloadEngine, build_tcga_download_tasks
from tools.tcga_tools import GDCClient
from utils.logger import get_logger

logger = get_logger(__name__)

# Sample types TCGA cannot serve (it only has tumor/adjacent/normal tissue).
_LIQUID_TYPES = {"cfdna", "plasma", "serum", "wbc", "whole_blood"}


def run_tcga_direct(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Search GDC by cancer keyword and download directly. Returns state update."""
    intent = state.get("parsed_intent") or {}
    output_dir = state.get("output_dir") or config["download"]["output_dir"]

    # Resolve cancer code.
    ct = intent.get("cancer_type")
    cancer_code = ct.get("tcga_code") if isinstance(ct, dict) else None
    if not cancer_code:
        return {"tcga_results": [], "tcga_log": "tcga-direct: no cancer_type.tcga_code, skipped"}

    # Skip liquid-biopsy requests — TCGA has no cfDNA/plasma/serum/WBC.
    sample_type = intent.get("sample_type")
    if sample_type in _LIQUID_TYPES:
        return {"tcga_results": [],
                "tcga_log": f"tcga-direct: skipped (sample_type={sample_type} not in TCGA)"}

    # Clients.
    gdc_token = os.environ.get(config.get("tcga", {}).get("gdc_token_env", ""), "") or None
    gdc = GDCClient(token=gdc_token or None)
    dl_cfg = config["download"]
    downloader = DownloadEngine(
        output_dir=dl_cfg["output_dir"],
        max_concurrent=dl_cfg["max_concurrent"],
        retry_attempts=dl_cfg["retry_attempts"],
        retry_delay=dl_cfg["retry_delay"],
        chunk_size_mb=dl_cfg["chunk_size_mb"],
        timeout=dl_cfg["timeout"],
    )
    gdc_api_base = config.get("tcga", {}).get("gdc_api_base", "https://api.gdc.cancer.gov")

    # Search.
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
        return {"tcga_results": [], "tcga_log": f"tcga-direct: GDC search failed: {e}"}
    if not files:
        return {"tcga_results": [], "tcga_log": f"tcga-direct: no GDC files for {cancer_code}"}

    records = gdc.files_to_dataset_records(files, cancer_code)
    logger.info(f"tcga-direct: {len(records)} dataset record(s) for {cancer_code}")

    # Build + run download tasks.
    tasks: List[Dict[str, Any]] = []
    for rec in records:
        try:
            tasks.extend(build_tcga_download_tasks(rec, output_dir, gdc_api_base))
        except Exception as e:
            logger.error(f"tcga-direct: build tasks failed for {rec.get('accession')}: {e}")

    results: List[Dict[str, Any]] = []
    if tasks:
        try:
            dl_results = downloader.download_many_sync(tasks)
        except Exception as e:
            logger.error(f"tcga-direct: download engine raised: {e}")
            dl_results = []
        results = _aggregate(dl_results)

    n_ok = sum(1 for r in results if r.get("outcome_final") == "download_success")
    return {
        "tcga_results": results,
        "tcga_log": (
            f"tcga-direct: cancer={cancer_code}, {len(records)} record(s), "
            f"{len(tasks)} task(s), {n_ok} succeeded"
        ),
    }


def _aggregate(dl_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_acc: Dict[str, List[Dict[str, Any]]] = {}
    for r in dl_results:
        by_acc.setdefault(r.get("accession", "?"), []).append(r)
    out: List[Dict[str, Any]] = []
    for acc, res in by_acc.items():
        done = [r for r in res if r.get("status") == "done"]
        all_done = bool(done) and all(r.get("status") == "done" for r in res)
        out.append({
            "accession": acc,
            "source": "TCGA",
            "files_downloaded": [
                {"name": (r.get("local_path") or "").split("/")[-1],
                 "local_path": r.get("local_path"),
                 "size_bytes": r.get("file_size_bytes"),
                 "qc_passed": True}
                for r in done
            ],
            "outcome_final": "download_success" if all_done else "failed",
            "notes": "" if all_done else "; ".join(
                r.get("error", "") for r in res if r.get("status") != "done"),
        })
    return out
