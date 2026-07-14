"""
geo-download skill — download + verify GEO datasets (Phase 1 skeleton).

Receives geo-filter's `download_list` (records whose outcome=download, each
carrying a metadata-inferred `files[]`), downloads the actual supplementary
files via DownloadEngine (+ md5), and emits `download_results`.

Phase 1 scope: download + md5. NO post-download content QC yet (value-type /
sample-column / GSM-mapping / disease-group verification, delete-on-fail,
quarantine, outcome revert). Those land in Phase 2 — the output schema already
reserves the fields (`files_failed_qc`, `outcome_final` revert values) so Phase 2
only fills in implementation, not contract.

Input  (state): download_list, output_dir
Output (state): download_results, download_log
"""
from __future__ import annotations

from typing import Any, Dict, List

from tools.download_tools import DownloadEngine, build_geo_download_tasks
from utils.logger import get_logger

logger = get_logger(__name__)


class DownloadSkill:
    """GEO download skill (Phase 1: download + md5, no content QC)."""

    name = "geo-download"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        dl = config["download"]
        self.output_dir = dl["output_dir"]
        self.downloader = DownloadEngine(
            output_dir=dl["output_dir"],
            max_concurrent=dl["max_concurrent"],
            retry_attempts=dl["retry_attempts"],
            retry_delay=dl["retry_delay"],
            chunk_size_mb=dl["chunk_size_mb"],
            timeout=dl["timeout"],
        )

    # ------------------------------------------------------------------ #

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        download_list = state.get("download_list") or []
        output_dir = state.get("output_dir") or self.output_dir

        results: List[Dict[str, Any]] = []
        tasks: List[Dict[str, Any]] = []

        # Build download tasks from each record's real GEO supplementary_files.
        for rec in download_list:
            acc = rec.get("accession", "?")
            try:
                tasks.extend(build_geo_download_tasks(rec, output_dir))
            except Exception as e:
                logger.error(f"geo-download: build tasks failed for {acc}: {e}")
                results.append({
                    "accession": acc,
                    "files_downloaded": [],
                    "files_failed_qc": [],
                    "outcome_final": "failed",
                    "flags": rec.get("flags", ""),
                    "notes": f"task build error: {e}",
                })

        # Execute downloads.
        if tasks:
            try:
                dl_results = self.downloader.download_many_sync(tasks)
            except Exception as e:
                logger.error(f"geo-download: download engine raised: {e}")
                dl_results = [{"accession": t.get("accession"), "status": "failed",
                               "error": str(e)} for t in tasks]
            results.extend(self._aggregate(dl_results, download_list))
        else:
            logger.info("geo-download: no tasks (download_list empty or no supplementary files)")

        n_ok = sum(1 for r in results if r.get("outcome_final") == "download_success")
        return {
            "download_results": results,
            "download_log": (
                f"geo-download: {len(download_list)} record(s), {len(tasks)} task(s), "
                f"{n_ok} succeeded"
            ),
        }

    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate(dl_results: List[Dict[str, Any]],
                   download_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group raw DownloadEngine results by accession → download_results records."""
        flags_by_acc = {r.get("accession"): r.get("flags", "") for r in download_list}
        by_acc: Dict[str, List[Dict[str, Any]]] = {}
        for r in dl_results:
            by_acc.setdefault(r.get("accession", "?"), []).append(r)

        out: List[Dict[str, Any]] = []
        for acc, res in by_acc.items():
            done = [r for r in res if r.get("status") == "done"]
            files_downloaded = [
                {
                    "name": r.get("filename") or (r.get("local_path") or "").split("/")[-1],
                    "local_path": r.get("local_path"),
                    "size_bytes": r.get("file_size_bytes"),
                    "qc_passed": True,  # Phase 1: md5/download success only; Phase 2 adds content QC
                    "data_form": None,
                    "provenance": {"source_url": _task_url(r), "checksum_md5": r.get("checksum_md5")},
                }
                for r in done
            ]
            all_done = bool(done) and all(r.get("status") == "done" for r in res)
            out.append({
                "accession": acc,
                "files_downloaded": files_downloaded,
                "files_failed_qc": [],  # Phase 2
                "outcome_final": "download_success" if all_done else "failed",
                "flags": flags_by_acc.get(acc, ""),
                "notes": "" if all_done else "; ".join(
                    r.get("error", "") for r in res if r.get("status") != "done"
                ),
            })
        return out


def _task_url(result: Dict[str, Any]) -> str:
    """Best-effort recover of the source URL from a DownloadEngine result."""
    return result.get("url") or result.get("source_url") or ""
