"""
geo-download skill — download + cancer-subset GEO datasets (Phase 2).

Per accession (from geo-filter's download_list):
  1. Build sample_metadata.csv with a `cancer` column (Phase 2b) — labels each
     GSM as query_cancer / control / unclear via heuristic matching.
  2. Download the methylation files (Phase 1: build_geo_download_tasks + md5).
  3. Cancer-subset (Phase 2c): for multi-cancer datasets where per-GSM labels are
     reliable, write a {acc}_query_subset.txt.gz with only the query-cancer GSM
     columns; discard the rest. If labels are mostly unclear → outcome reverts to
     manual_review (human labels/subsets).

File-form A-level verification already happened upstream in the filter (Phase 2a,
核验前置); this skill does NOT re-verify file form — only a light landing check
(file exists, size > 0) + the cancer subset.

Input  (state): download_list, parsed_intent, output_dir
Output (state): download_results, download_log
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from skills.geo_download.cancer_label import (
    build_sample_metadata_with_cancer,
    query_cancer_terms,
)
from tools.download_tools import DownloadEngine, build_geo_download_tasks
from tools.geo_tools import GEOClient
from utils.logger import get_logger

logger = get_logger(__name__)

# If more than this fraction of GSMs are "unclear", send to manual_review.
_UNCLEAR_MANUAL_REVIEW_THRESHOLD = 0.5


def _cancer_matches(cancer_type: Optional[str], query_terms: List[str]) -> bool:
    """
    Does the dataset's cancer_type match the query cancer? Used by the
    single-cancer fallback: if per-GSM labels are unclear but the dataset's own
    cancer matches the query, assume the whole file is single-cancer.
    """
    if not cancer_type or not query_terms:
        return False
    ct = cancer_type.lower()
    return any(t and (t in ct or ct in t) for t in query_terms)


class DownloadSkill:
    """GEO download + cancer-subset skill (Phase 2)."""

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
        # geo_client for full GSM fetch (sample_metadata cancer labeling).
        ncbi_key = os.environ.get(config.get("geo", {}).get("api_key_env", ""), "") or None
        ncbi_proxy = (
            os.environ.get("NCBI_PROXY", "")
            or config.get("geo", {}).get("proxy", "")
            or None
        )
        self.geo_client = GEOClient(api_key=ncbi_key or None, proxy=ncbi_proxy or None)

    # ------------------------------------------------------------------ #

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        download_list = state.get("download_list") or []
        intent = state.get("parsed_intent") or {}
        output_dir = state.get("output_dir") or self.output_dir
        query_terms = query_cancer_terms(intent)

        results = [
            self.process_dataset(rec, query_terms, output_dir)
            for rec in download_list
        ]
        n_ok = sum(1 for r in results if r.get("outcome_final") == "download_success")
        return {
            "download_results": results,
            "download_log": (
                f"geo-download: {len(download_list)} record(s), {n_ok} succeeded, "
                f"{sum(1 for r in results if 'manual_review' in (r.get('outcome_final') or ''))} manual_review"
            ),
        }

    # ------------------------------------------------------------------ #

    def process_dataset(self, rec: Dict[str, Any], query_terms: List[str],
                        output_dir: str) -> Dict[str, Any]:
        """
        Download + cancer-subset ONE dataset record. Public so the daemon's
        run_pending_downloads (human-approved items) can reuse the same path as
        the pipeline. `rec` needs at least accession + supplementary_files
        (re-fetch via get_series_metadata if not available).
        """
        acc = rec.get("accession", "?")
        flags = rec.get("flags", "")

        # Phase 2b: per-GSM cancer labels → sample_metadata.csv (also used for subset).
        sm = build_sample_metadata_with_cancer(acc, self.geo_client, query_terms, output_dir)

        # Phase 1: download.
        try:
            tasks = build_geo_download_tasks(rec, output_dir)
        except Exception as e:
            logger.error(f"geo-download: build tasks failed for {acc}: {e}")
            return self._result(acc, [], [], "failed", flags,
                                notes=f"task build error: {e}", subset_path=None)
        dl_results = self.downloader.download_many_sync(tasks) if tasks else []
        done = [r for r in dl_results if r.get("status") == "done"]

        # Phase 2c: cancer subset (or single-cancer fallback / manual_review).
        subset_path, subset_note, forced_outcome = self._subset_by_cancer(
            acc, done, sm, output_dir,
            query_terms=query_terms, cancer_type=rec.get("cancer_type"))

        files_downloaded = [
            {
                "name": (r.get("local_path") or "").split("/")[-1],
                "local_path": r.get("local_path"),
                "size_bytes": r.get("file_size_bytes"),
                "qc_passed": bool(r.get("local_path")),  # light landing check
                "data_form": rec.get("available_file_type"),
                "provenance": {"source_url": r.get("url"), "checksum_md5": r.get("checksum_md5")},
            }
            for r in done
        ]
        outcome = forced_outcome or ("download_success" if done else "failed")
        notes = subset_note
        if not done:
            notes = "; ".join(r.get("error", "") for r in dl_results if r.get("status") != "done")
        return self._result(acc, files_downloaded, [], outcome, flags,
                            notes=notes, subset_path=subset_path)

    # ------------------------------------------------------------------ #
    #  Cancer subset (Phase 2c)                                          #
    # ------------------------------------------------------------------ #

    def _subset_by_cancer(self, acc: str, done_results: List[Dict[str, Any]],
                          sm: Optional[pd.DataFrame], output_dir: str,
                          query_terms: List[str] = None, cancer_type: str = None
                          ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Decide whether to subset the downloaded matrix to query-cancer GSMs.

        Returns (subset_path, note, forced_outcome):
          - forced_outcome="qc_failed_reverted_manual_review" when cancer labels
            are mostly unclear AND the dataset's cancer can't be confirmed as the
            query cancer (human must label/subset).
          - subset_path set when a query-cancer subset file was written.
        """
        if sm is None or sm.empty or not done_results:
            return None, "no sample_metadata / no downloaded file", None

        total = len(sm)
        if "cancer" not in sm.columns:
            return None, "sample_metadata has no cancer column", None
        counts = sm["cancer"].value_counts().to_dict()
        n_query = int(counts.get("query_cancer", 0))
        n_unclear = int(counts.get("unclear", 0))

        # Mostly unclear → try the single-cancer fallback: if per-GSM cancer
        # labels are unavailable BUT the dataset's own cancer_type matches the
        # query cancer (the filter already confirmed it's the target cancer),
        # assume the whole file is single-cancer (no subset needed, success).
        # Only send to manual_review when we can't even confirm the dataset is
        # the query cancer.
        if total and n_unclear / total > _UNCLEAR_MANUAL_REVIEW_THRESHOLD:
            if _cancer_matches(cancer_type, query_terms):
                logger.info(
                    f"geo-download {acc}: {n_unclear}/{total} GSMs cancer-unclear, "
                    f"but dataset cancer_type='{cancer_type}' matches query → "
                    f"single-cancer assumed (no subset)")
                return None, (
                    f"single-cancer assumed (per-GSM cancer labels unavailable; "
                    f"{n_unclear}/{total} unclear; dataset cancer={cancer_type})"
                ), None
            logger.info(f"geo-download {acc}: {n_unclear}/{total} GSMs cancer-unclear → manual_review")
            return None, (
                f"cancer labels unclear for {n_unclear}/{total} GSMs; "
                f"needs manual subset (counts={counts})"
            ), "qc_failed_reverted_manual_review"

        # Single-cancer dataset (all/most query_cancer) → keep whole file, no subset.
        query_gsms = set(sm.loc[sm["cancer"] == "query_cancer", "gsm"].astype(str))
        if n_query == 0:
            # No query-cancer GSM identified but labels not mostly-unclear → manual review.
            return None, f"no query-cancer GSM identified (counts={counts})", \
                   "qc_failed_reverted_manual_review"
        if n_query >= total * 0.9:
            return None, f"single-cancer ({n_query}/{total} query) — whole file kept", None

        # Multi-cancer → subset the largest downloaded file to query-cancer GSM columns.
        target = max(done_results, key=lambda r: r.get("file_size_bytes") or 0)
        local_path = target.get("local_path")
        if not local_path:
            return None, "no local file to subset", None
        subset_path, n_kept, n_cols, note = _write_query_subset(
            local_path, str(Path(output_dir) / acc), acc, query_gsms)
        return subset_path, (note or "subset ok"), None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _result(accession: str, files_downloaded: List[Dict], files_failed_qc: List[Dict],
                outcome_final: str, flags: str, notes: str, subset_path: Optional[str]) -> Dict[str, Any]:
        return {
            "accession": accession,
            "files_downloaded": files_downloaded,
            "files_failed_qc": files_failed_qc,
            "outcome_final": outcome_final,
            "flags": flags,
            "subset_path": subset_path,
            "notes": notes or "",
        }


# ---------------------------------------------------------------------- #
#  Matrix subset helper                                                   #
# ---------------------------------------------------------------------- #

def _write_query_subset(local_path: str, out_dir: str, accession: str,
                        query_gsms: set) -> Tuple[Optional[str], int, int, str]:
    """
    Best-effort: read the (gzip) matrix, keep the first column (feature id) +
    columns whose header contains a query-cancer GSM, write a subset file.

    Returns (subset_path, n_kept_columns, n_total_columns, note).
    """
    try:
        comp = "gzip" if local_path.endswith(".gz") else None
        # Read just the header to find columns (skip GEO SOFT '!' metadata lines).
        header_row = None
        with _open_maybe_gz(local_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("!"):
                    continue
                header_row = s
                break
        if not header_row:
            return None, 0, 0, "no header line found"

        sep = "\t" if "\t" in header_row else ","
        cols = [c.strip().strip('"') for c in header_row.split(sep)]
        # keep first column (feature id) + any column whose name contains a query GSM
        keep = [cols[0]] + [c for c in cols[1:] if any(g in c for g in query_gsms)]
        keep = _dedup_preserve(keep)
        if len(keep) <= 1:
            return None, 0, len(cols) - 1, "no query-cancer GSM columns matched in header"

        df = pd.read_csv(local_path, sep=sep, usecols=keep, compression=comp,
                         low_memory=False)
        subset_path = Path(out_dir) / f"{accession}_query_subset.txt.gz"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(subset_path, sep="\t", index=False, compression="gzip")
        return str(subset_path), len(keep) - 1, len(cols) - 1, \
            f"subset: kept {len(keep) - 1}/{len(cols) - 1} query-cancer sample columns"
    except Exception as e:
        return None, 0, 0, f"subset failed: {e}"


def _open_maybe_gz(path: str):
    import gzip
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") \
        if path.endswith(".gz") else open(path, "rt", encoding="utf-8", errors="replace")


def _dedup_preserve(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
