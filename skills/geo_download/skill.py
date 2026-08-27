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

import hashlib
import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage

from skills.geo_download.cancer_label import (
    build_sample_metadata_with_cancer,
    query_cancer_terms,
)
from skills.geo_download.archive_extract import (
    detect_per_sample_set,
    extract_archive_members,
    is_archive,
)
from skills.geo_download.verify import (
    build_gsm_column_map,
    check_column_counts,
    check_group_completeness,
    parse_sample_columns,
    parse_series_matrix_samples_local,
    quarantine_files,
)
from skills.geo_filter.file_inspect import inspect_matrix_head
from tools.download_tools import DownloadEngine, build_geo_download_tasks
from tools.geo_tools import GEOClient
from utils.llm_factory import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)

# If more than this fraction of GSMs are "unclear", send to manual_review.
_UNCLEAR_MANUAL_REVIEW_THRESHOLD = 0.5

# GSM→column mapping coverage below this → manual_review FLAG (files stay;
# report-only philosophy — user decision 2026-08-21).
_MAP_COVERAGE_REVIEW = 0.5

# Filename carries a GSM id → one-sample-per-file (Tier-3 .cov/.bed forms).
_GSM_IN_NAME_RE = re.compile(r"GSM\d+", re.IGNORECASE)


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

    def __init__(self, config: Dict[str, Any], registry=None):
        self.config = config
        self.registry = registry
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
        # LLM for Tier-2 sample-type-aware file selection (None if unconfigured →
        # the selector falls back to keeping all non-junk files + manual_review).
        try:
            # JSON mode: the file-selection invoke returns {files:[...]} JSON.
            self.llm = get_llm(config.get("llm") or {}, json_mode=True)
        except Exception as exc:
            logger.warning(f"geo-download: LLM unavailable ({exc}); file selection will keep all non-junk")
            self.llm = None

    # ------------------------------------------------------------------ #

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        download_list = state.get("download_list") or []
        intent = state.get("parsed_intent") or {}
        output_dir = state.get("output_dir") or self.output_dir
        query_terms = query_cancer_terms(intent)
        task_id = state.get("task_id")

        results = [
            self.process_dataset(rec, query_terms, output_dir, task_id=task_id)
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
                        output_dir: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Download + cancer-subset ONE dataset record. Three-tier fallback:
          1. series_matrix has data → download series_matrix (all samples in one file)
          2. supplementary files exist → download those
          3. neither → scrape GSM pages for per-sample beta tables (only "download" GSMs)

        task_id routes the per-GSM "download" set via the registry (GSM-grain
        query_sample_map) instead of guessing from a CSV column — essential when a
        GSE is hit by multiple queries. Falls back to the CSV when task_id is
        absent or the registry has no rows for this task.
        """
        acc = rec.get("accession", "?")
        flags = rec.get("flags", "")

        # Phase 2b: read existing sample_metadata.csv (written by filter).
        sm = _read_sample_metadata(acc, output_dir)

        # Resolve THIS task's "download" GSM set once: registry-first, CSV fallback.
        # Used by Tier 3 (which GSM pages to scrape) and Tier 2 (target-sample summary
        # for LLM file selection) so both honour task routing.
        download_gsms = _read_downloadable_gsms(
            acc, output_dir, task_id=task_id, registry=self.registry)

        # ---- Three-tier download task building ----
        tier_used = "?"
        try:
            # Tier 1: series_matrix has data?
            if self.geo_client.series_matrix_has_data(acc):
                tier_used = "1(series_matrix)"
                tasks = _build_series_matrix_task(acc, output_dir)
                logger.info(f"geo-download {acc}: Tier 1 (series_matrix has data)")
            # Tier 2: supplementary files?
            elif rec.get("supplementary_files"):
                tier_used = "2(supplementary)"
                # Download ALL non-RAW supp files; content-based keep/discard happens
                # after download (download_all_non_raw skips the filename keyword gate).
                tasks = build_geo_download_tasks(
                    rec, output_dir, download_all_non_raw=True)
                logger.info(f"geo-download {acc}: Tier 2 (supplementary files)")
            # Tier 3: GSM page scraping (only "download" GSMs)
            else:
                tier_used = "3(gsm_scrape)"
                gsm_list = download_gsms
                logger.info(f"geo-download {acc}: Tier 3 (scraping {len(gsm_list)} GSM pages)")
                tasks = []
                for gsm_id in gsm_list:
                    info = self.geo_client.fetch_gsm_supplementary_file(gsm_id)
                    if info:
                        url = info["url"].replace("ftp://", "https://", 1)
                        # GSM-page hrefs are URL-encoded (%5F for _); decode so
                        # the on-disk name (and GSM-in-filename detection) is sane.
                        from urllib.parse import unquote
                        tasks.append({
                            "accession": acc,
                            "url": url,
                            "filename": unquote(info["filename"]),
                            "subdir": f"{acc}/{gsm_id}",
                        })
                    else:
                        logger.debug(f"geo-download {acc}: no supp file for {gsm_id}")
                if not tasks:
                    logger.warning(f"geo-download {acc}: Tier 3 found no GSM supp files")
        except Exception as e:
            logger.error(f"geo-download: build tasks failed for {acc}: {e}")
            return self._result(acc, [], [], "failed", flags,
                                notes=f"task build error ({tier_used}): {e}", subset_path=None)

        # ---- Download ----
        dl_results = self.downloader.download_many_sync(tasks) if tasks else []
        done = [r for r in dl_results if r.get("status") == "done"]

        # ---- Phase-2 ⑤: archive member extraction ----
        # Must run BEFORE the junk filter: a compressed archive read as raw
        # bytes looks "unparseable" to inspect_matrix_head and would be
        # junk-dropped, losing every member inside. Failure → archive kept whole.
        done = self._expand_archives(acc, done, output_dir)

        # ---- Tier 2/3 file selection ----
        # We downloaded every non-RAW supp file (Tier 2) or per-GSM supp file
        # (Tier 3). Now select which to KEEP:
        #   1. junk-filter (demoted inspect_matrix_head) drops clear non-data files
        #      (empty/README, p-value/logFC tables) — NOT a value-range A-level gate.
        #   2. the LLM picks files whose SAMPLE TYPE matches the query (e.g. plasma
        #      cfDNA vs tissue), using the query + the target-sample set + each file's
        #      head. Conservative fallback keeps all non-junk + manual_review.
        # This is where "keep which data" is decided now that geo-filter no longer
        # gates on file format (3-state outcome; usability judged post-download).
        discarded: List[Dict[str, Any]] = []
        selection_note = ""
        sel_forced: Optional[str] = None
        if tier_used.startswith(("2", "3")):
            done, discarded, selection_note, sel_forced = self._select_relevant_files(
                acc, done, rec, sm, download_gsms=download_gsms)

        # ---- Phase-2 ①②③: post-download verification ----
        # Column count vs download-GSM count (quarantines on fail), GSM→column
        # mapping (title-index cascade; low coverage → manual_review flag,
        # files stay), group completeness (report-only, notes).
        verify_note = ""
        verify_failed_qc: List[Dict[str, Any]] = []
        qc_forced: Optional[str] = None
        column_map: Dict[str, Dict[str, str]] = {}
        if done:
            (verify_note, verify_failed_qc, qc_forced, column_map
             ) = self._verify_dataset(acc, done, sm, output_dir,
                                      tier_used=tier_used,
                                      download_gsms=download_gsms)

        # ---- Phase-2 ④: quarantine on column-count fail ----
        if verify_failed_qc:
            outcome_pre = "qc_failed_reverted_manual_review"
            logger.warning(f"geo-download {acc}: QC fail → quarantine "
                           f"({len(verify_failed_qc)} file(s)): {verify_note}")
        else:
            outcome_pre = None

        # ---- Cancer subset (uses the verified column→GSM map when available) ----
        subset_path, subset_note, forced_outcome = self._subset_by_cancer(
            acc, done, sm, output_dir,
            query_terms=query_terms, cancer_type=rec.get("cancer_type"),
            column_map=column_map)
        forced_outcome = forced_outcome or sel_forced  # selection fallback may flag review

        files_downloaded = [
            {
                "name": (r.get("local_path") or "").split("/")[-1],
                "local_path": r.get("local_path"),
                "size_bytes": r.get("file_size_bytes"),
                "qc_passed": bool(r.get("local_path")) and not verify_failed_qc,
                "data_form": r.get("data_form") or rec.get("available_file_type"),
                "needs_processing": r.get("needs_processing"),
                "provenance": {"source_url": r.get("url"), "checksum_md5": r.get("checksum_md5")},
            }
            for r in done
        ]
        outcome = outcome_pre or forced_outcome or (
            "download_success" if done else "failed")
        notes = subset_note + f" [tier={tier_used}]"
        if selection_note:
            notes += f"; {selection_note}"
        if verify_note:
            notes += f"; {verify_note}"
        if discarded:
            reasons = ", ".join(sorted({
                f"{d['name']}: {d.get('reason', '')}" for d in discarded}))
            notes += (f"; tier2 file-select: kept {len(files_downloaded)}, "
                      f"discarded {len(discarded)} ({reasons})")
        if not done:
            dl_done = [r for r in dl_results if r.get("status") == "done"]
            if not tasks:
                # No files found in any tier — not a download error, just unavailable
                outcome = "no_files"
                notes = f"no downloadable files found [tier={tier_used}]"
            elif not dl_done:
                # Tasks existed but none downloaded successfully
                outcome = "failed"
                notes = ("; ".join(r.get("error", "") for r in dl_results
                                   if r.get("status") != "done")
                         + f" [tier={tier_used}]")
            elif discarded:
                # Files downloaded but every one was discarded
                outcome = "no_files"
                notes = (f"downloaded {len(discarded)} supp file(s), all discarded "
                         f"[tier={tier_used}]")
            else:
                outcome = "failed"
                notes = ("; ".join(r.get("error", "") for r in dl_results
                                   if r.get("status") != "done")
                         + f" [tier={tier_used}]")
        return self._result(acc, files_downloaded, verify_failed_qc, outcome, flags,
                            notes=notes, subset_path=subset_path,
                            files_discarded=discarded)

    # ------------------------------------------------------------------ #
    #  Phase-2 ⑤: archive member extraction                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _expand_archives(acc: str, done_results: List[Dict[str, Any]],
                         output_dir: str) -> List[Dict[str, Any]]:
        """
        Expand tar/zip supp files into their members (in place of the archive).
        Runs BEFORE the junk filter — an archive's raw bytes would otherwise
        be junk-dropped as "unparseable". Conservative: extraction failure or
        an empty result keeps the archive as one candidate file.
        """
        out: List[Dict[str, Any]] = []
        for r in done_results:
            local_path = r.get("local_path")
            if not local_path or not is_archive(local_path):
                out.append(r)
                continue
            members = extract_archive_members(
                local_path, str(Path(output_dir) / acc / "members"))
            if members:
                logger.info(f"geo-download {acc}: expanded {len(members)} member(s) "
                            f"from {(local_path or '').split('/')[-1]}")
                _delete_file(local_path, acc, (local_path or "").split("/")[-1])
                for m in members:
                    out.append({
                        "accession": acc, "status": "done", "local_path": m,
                        "file_size_bytes": os.path.getsize(m),
                        "checksum_md5": None, "url": r.get("url"),
                        "extracted_from": (local_path or "").split("/")[-1],
                    })
            else:
                out.append(r)  # keep the archive whole (judged downstream)
        return out

    # ------------------------------------------------------------------ #
    #  Phase-2 ①②③: post-download verification                          #
    # ------------------------------------------------------------------ #

    def _verify_dataset(self, acc: str, done_results: List[Dict[str, Any]],
                        sm: Optional[pd.DataFrame], output_dir: str,
                        tier_used: str = "",
                        download_gsms: Optional[List[str]] = None,
                        ) -> Tuple[str, List[Dict[str, Any]], Optional[str],
                                   Dict[str, Dict[str, str]]]:
        """
        Verify the kept files against the task's download-GSM set:

          ① sample-column count (aggregate across files; tolerance
             max(5, 10%)) — fail → quarantine records returned;
          ② GSM→column mapping (gsm-id regex → series !Sample_title index →
             prefix) — low coverage → manual_review FLAG (files stay);
          ③ disease-group completeness — report-only (notes).

        Returns (note, files_failed_qc, forced_outcome, column_map).
        files_failed_qc non-empty ⇒ caller quarantines (④).
        """
        n_gsms = len(download_gsms or [])
        per_file: Dict[str, Dict[str, Any]] = {}
        matrix_results: List[Dict[str, Any]] = []
        # Per-sample form first: filenames carrying GSM ids (Tier-3 per-GSM
        # downloads like .cov/.bed/.bsmap files) are one-sample-per-file BY
        # CONSTRUCTION — parse_sample_columns would miscount their coordinate
        # columns as 6 "samples" each. Detect by GSM-in-filename BEFORE
        # matrix parsing, and exclude them from column aggregation.
        per_sample_paths: List[str] = []
        for r in done_results:
            name = (r.get("local_path") or "").split("/")[-1]
            if name and _GSM_IN_NAME_RE.search(name):
                per_sample_paths.append(r["local_path"])
                r["needs_processing"] = "merge_per_sample"
        is_per_sample = detect_per_sample_set(per_sample_paths, n_gsms)

        for r in done_results:
            if r.get("needs_processing"):
                continue  # per-sample files don't contribute matrix columns
            parsed = parse_sample_columns(r.get("local_path") or "")
            if parsed is None:
                continue  # unparseable files don't contribute columns either
            r["_parsed_cols"] = parsed
            per_file[(r.get("local_path") or "").split("/")[-1]] = parsed
            matrix_results.append(r)

        col_check = check_column_counts(
            list(per_file.values()), n_gsms,
            per_sample_files=per_sample_paths if is_per_sample else None)
        if not col_check["passed"]:
            reason = f"column-count check failed: {col_check['reason']}"
            return (f"[verify] {reason}", quarantine_files(
                acc, done_results, output_dir, reason),
                "qc_failed_reverted_manual_review", {})

        # ② mapping (needs series sample titles; Tier 1 parses the local
        # series_matrix, Tier 2/3 fetch it once per accession).
        series_samples = self._series_samples(acc, done_results, output_dir,
                                              tier_used)
        col_names_by_file = {name: p.get("col_names") or []
                             for name, p in per_file.items()}
        mapres = build_gsm_column_map(col_names_by_file, series_samples)
        column_map = mapres["column_map"]

        parts = [f"[verify] {col_check['reason']}"]
        forced: Optional[str] = None
        if mapres["n_cols"]:
            parts.append(f"map {mapres['level']} {mapres['n_mapped']}/{mapres['n_cols']}")
            if mapres["coverage"] < _MAP_COVERAGE_REVIEW:
                forced = "manual_review_gsm_mapping"
                parts.append(f"(low coverage → {forced})")

        # ③ group completeness (report-only).
        if column_map and sm is not None:
            grp = check_group_completeness(column_map, sm, download_gsms or [])
            parts.append(f"groups {grp['summary']}")

        return "; ".join(parts), [], forced, column_map

    def _series_samples(self, acc: str, done_results: List[Dict[str, Any]],
                        output_dir: str, tier_used: str) -> List[Dict[str, str]]:
        """
        Per-sample {gsm,title,source_name} rows for the mapping cascade.
        Parse a LOCAL series_matrix first (downloaded by the tier probe, or on
        disk from an earlier run — NCBI fetches can be throttled, and the local
        copy is the same data); fall back to fetching once per accession (the
        !Sample_* rows exist even when the series_matrix has no data table).
        Best-effort: [] on failure.
        """
        cached = getattr(self, "_series_samples_cache", None)
        if cached is None:
            cached = self._series_samples_cache = {}
        if acc in cached:
            return cached[acc]
        samples: List[Dict[str, str]] = []
        try:
            # 1) Any series_matrix among this run's downloaded files (Tier 1).
            for r in done_results:
                name = (r.get("local_path") or "")
                if "series_matrix" in name.lower():
                    samples = parse_series_matrix_samples_local(name)
                    break
            # 2) A leftover series_matrix on disk from an earlier run/probe
            #    (covers Tier 2/3 where the tier check downloaded it but the
            #    download list went elsewhere).
            if not samples:
                for cand in sorted(
                        (Path(output_dir) / acc).glob("*series_matrix*")):
                    samples = parse_series_matrix_samples_local(str(cand))
                    if samples:
                        break
            # 3) Network fetch as last resort.
            if not samples and self.geo_client is not None:
                fetched = self.geo_client.fetch_series_matrix_sample_info(acc)
                if fetched:
                    samples = [{"gsm": s.get("gsm", ""), "title": s.get("title", ""),
                                "source_name": s.get("source_name", "")}
                               for s in fetched]
        except Exception as e:
            logger.debug(f"geo-download {acc}: series sample info unavailable: {e}")
        cached[acc] = samples
        return samples

    # ------------------------------------------------------------------ #
    #  Cancer subset (Phase 2c)                                          #
    # ------------------------------------------------------------------ #

    def _subset_by_cancer(self, acc: str, done_results: List[Dict[str, Any]],
                          sm: Optional[pd.DataFrame], output_dir: str,
                          query_terms: List[str] = None, cancer_type: str = None,
                          column_map: Optional[Dict[str, Dict[str, str]]] = None,
                          ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Decide whether to subset the downloaded matrix to query-cancer GSMs.

        column_map (from Phase-2 verification, ②) resolves submitter-coded
        column names (Pcrc90...) to GSMs — when present, the subset selects
        columns THROUGH the map instead of GSM-substring matching in the
        header (which fails on submitter-coded names).

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
        # Route through the verified column→GSM map when it covers this file's
        # columns (submitter-coded names); else the legacy GSM-substring match.
        query_cols: Optional[Set[str]] = None
        if column_map:
            query_cols = {col for col, v in column_map.items()
                          if v.get("gsm") in query_gsms}
        subset_path, n_kept, n_cols, note = _write_query_subset(
            local_path, str(Path(output_dir) / acc), acc, query_gsms,
            query_cols=query_cols)
        if query_cols and subset_path:
            note = (note or "subset ok") + " (via column map)"
        return subset_path, (note or "subset ok"), None

    # ------------------------------------------------------------------ #
    #  Tier-2 file selection (junk filter + LLM sample-type selection)   #
    # ------------------------------------------------------------------ #

    def _select_relevant_files(
        self, acc: str, done_results: List[Dict[str, Any]], rec: Dict[str, Any],
        sm: Optional[pd.DataFrame], download_gsms: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[str]]:
        """
        Select which downloaded Tier-2 supp files to keep.

        1. Junk-filter (demoted inspect_matrix_head): auto-drop clear non-data files
           (empty/README/binary, p-value/logFC statistical tables). NOT a value-range
           A-level gate — MCTA-Seq counts / unusual matrices pass through.
        2. LLM picks files whose SAMPLE TYPE matches the query (e.g. plasma cfDNA vs
           tissue), using the query + the target-sample set + each file's head.
        3. Conservative fallback: no LLM / parse error / LLM keeps none → keep ALL
           non-junk files and flag manual_review (never silently delete real data).

        Returns (kept_results, discarded_info, note, forced_outcome).
        """
        candidates, junk = self._junk_filter(acc, done_results)

        if not candidates:
            return [], junk, "all supp files were junk (no data)", None

        if self.llm is None:
            # No LLM → keep everything that isn't obvious junk, flag for review.
            return candidates, junk, (
                f"no LLM → kept all {len(candidates)} non-junk file(s) (manual_review)"
            ), "manual_review_file_selection"

        try:
            kept, llm_discarded, llm_note = self._llm_select_files(
                acc, candidates, rec, sm, download_gsms=download_gsms)
        except Exception as exc:
            logger.warning(f"geo-download {acc}: LLM file-selection failed ({exc}) → keep all non-junk")
            return candidates, junk, (
                f"LLM selection error ({exc}) → kept all {len(candidates)} non-junk (manual_review)"
            ), "manual_review_file_selection"

        # Safety: if the LLM kept nothing but candidates existed, don't trust it —
        # revert to keeping all non-junk + flag review. (Files are NOT deleted yet —
        # _llm_select_files only classifies; deletion happens below on the commit path.)
        if not kept:
            logger.warning(f"geo-download {acc}: LLM kept no files → revert to all non-junk (manual_review)")
            return candidates, junk, (
                f"LLM kept none → kept all {len(candidates)} non-junk (manual_review); "
                f"{llm_note}"
            ), "manual_review_file_selection"

        # Commit: delete the LLM-discarded files now that we're keeping ≥1.
        for d in llm_discarded:
            _delete_file(d.get("local_path"), acc, d.get("name"))
        # Strip the local_path from the discard records (provenance keeps md5+url).
        for d in llm_discarded:
            d.pop("local_path", None)
        return kept, junk + llm_discarded, llm_note, None

    # -------- junk pre-filter (demoted inspect_matrix_head) ------------- #

    def _junk_filter(self, acc: str, done_results: List[Dict[str, Any]]
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Auto-discard ONLY clear non-data files. Returns (kept, junk_discarded).
        A file is junk iff:
          - inspect returns unknown with empty/no-lines/no-numeric reason (README/empty/binary), or
          - inspect returns non_methylation with 'statistical columns' (p-value/logFC diff table).
        Everything else (beta/M-value/ratio/paired-counts, integer read-counts with sample
        columns, unknown-with-numeric-range) is kept for the LLM to judge.
        """
        kept: List[Dict[str, Any]] = []
        junk: List[Dict[str, Any]] = []
        for r in done_results:
            local_path = r.get("local_path")
            name = (local_path or "").split("/")[-1]
            if not local_path or not os.path.exists(local_path):
                continue
            # series_matrix → always keep (GEO-compiled), don't even inspect.
            if "series_matrix" in name.lower():
                r["data_form"] = "series_matrix"
                kept.append(r)
                continue
            try:
                with open(local_path, "rb") as f:
                    info = inspect_matrix_head(f.read(1 << 20))
            except Exception as e:
                info = {"value_type": "unknown", "reason": f"inspect error: {e}"}
            vt, reason = info.get("value_type", "unknown"), info.get("reason", "")
            is_junk = (
                (vt == "unknown" and any(s in reason for s in
                    ("empty", "no lines", "no numeric", "unparseable")))
                or (vt == "non_methylation" and reason.startswith("statistical columns"))
            )
            if is_junk:
                md5 = _md5_file(local_path)
                _delete_file(local_path, acc, name)
                junk.append({"name": name, "value_type": vt, "reason": f"junk: {reason}",
                             "md5": md5, "source_url": r.get("url")})
                logger.info(f"geo-download {acc}: junk-drop {name} ({vt}: {reason})")
            else:
                kept.append(r)
        return kept, junk

    # -------- LLM sample-type-aware selection --------------------------- #

    def _llm_select_files(self, acc: str, candidates: List[Dict[str, Any]],
                          rec: Dict[str, Any], sm: Optional[pd.DataFrame],
                          download_gsms: Optional[List[str]] = None,
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Ask the LLM which candidate files contain samples matching the query sample
        type. Returns (kept, discarded, note). Raises on parse/error (caller falls back).
        """
        # Target-sample summary from sample_metadata (the LLM-selected "download" set).
        target_summary = self._target_sample_summary(sm, download_gsms=download_gsms)

        file_blocks = []
        name_to_result: Dict[str, Dict[str, Any]] = {}
        for r in candidates:
            name = (r.get("local_path") or "").split("/")[-1]
            name_to_result[name] = r
            head = _read_head_text(r.get("local_path"), max_lines=8)
            file_blocks.append(f"--- FILE: {name} ---\n{head}")
        files_text = "\n\n".join(file_blocks)

        user_msg = (
            f"=== USER REQUEST ===\n"
            f"raw query: {rec.get('raw_query') or '(not recorded)'}\n"
            f"requested sample type: {rec.get('sample_type') or '(unspecified)'}\n"
            f"cancer: {rec.get('cancer_type') or '(unspecified)'}\n\n"
            f"=== DATASET ===\n"
            f"accession: {acc}\n"
            f"title: {(rec.get('title') or '')[:200]}\n"
            f"data type: {rec.get('data_type') or 'unknown'}\n\n"
            f"=== TARGET SAMPLES (already selected for this query) ===\n"
            f"{target_summary}\n\n"
            f"=== CANDIDATE SUPPLEMENTARY FILES (head of each) ===\n"
            f"{files_text}\n"
        )

        resp = self.llm.invoke([
            SystemMessage(content=_FILE_SELECTION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        verdict = _safe_json(raw)
        files_verdict = verdict.get("files") or []
        keep_set = {
            str(f.get("name", "")).strip(): f for f in files_verdict
            if isinstance(f, dict)
        }

        kept: List[Dict[str, Any]] = []
        discarded: List[Dict[str, Any]] = []
        for name, r in name_to_result.items():
            v = keep_set.get(name)
            local_path = r.get("local_path")
            if v and bool(v.get("keep")):
                r["data_form"] = v.get("sample_type") or v.get("data_form") or "selected"
                kept.append(r)
                logger.info(f"geo-download {acc}: LLM keep {name} "
                            f"({v.get('sample_type', '')}: {v.get('reason', '')})")
            else:
                # Classify only — DO NOT delete here. The caller commits deletion only
                # after the keep-none safety check passes (else we'd lose data on a
                # bad LLM response that we then revert).
                reason = (v or {}).get("reason", "not selected by LLM")
                stype = (v or {}).get("sample_type", "")
                discarded.append({"name": name, "local_path": local_path,
                                  "sample_type": stype, "reason": reason,
                                  "md5": _md5_file(local_path), "source_url": r.get("url")})
                logger.info(f"geo-download {acc}: LLM discard {name} ({stype}: {reason})")

        reasoning = (verdict.get("reasoning") or "").replace("\n", " ")[:200]
        note = (f"LLM selected {len(kept)}/{len(candidates)} file(s)"
                + (f": {reasoning}" if reasoning else ""))
        return kept, discarded, note

    @staticmethod
    def _target_sample_summary(sm: Optional[pd.DataFrame],
                               download_gsms: Optional[List[str]] = None) -> str:
        """One-line summary of the LLM-selected 'download' sample set for the prompt.

        If download_gsms is provided (task-routed from the registry), summarize exactly
        those GSMs. Otherwise fall back to the rightmost per-task CSV column — the old
        behaviour, kept for callers without a task_id.
        """
        if sm is None or sm.empty or "gsm" not in sm.columns:
            return "(sample_metadata unavailable)"

        if download_gsms:
            dl = sm[sm["gsm"].astype(str).isin(set(download_gsms))]
            if dl.empty:
                return f"0 samples matched task download set ({len(download_gsms)} ids)"
        else:
            base = {"gsm", "source_name", "molecule", "group", "cancer"}
            tid_cols = [c for c in sm.columns if c not in base and len(str(c)) == 8]
            if not tid_cols:
                return f"{len(sm)} sample(s) (no per-task download column)"
            col = tid_cols[-1]
            dl = sm[sm[col] == "download"]
            if dl.empty:
                return f"0 samples marked download (col={col})"

        parts = [f"{len(dl)} sample(s) marked download"]
        if "source_name" in dl.columns:
            dist = dl["source_name"].astype(str).value_counts().head(5)
            parts.append("source_name: " + ", ".join(f"{k}={v}" for k, v in dist.items()))
        if "group" in dl.columns:
            gdist = dl["group"].astype(str).value_counts().head(5)
            parts.append("group: " + ", ".join(f"{k}={v}" for k, v in gdist.items()))
        return "; ".join(parts)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _result(accession: str, files_downloaded: List[Dict], files_failed_qc: List[Dict],
                outcome_final: str, flags: str, notes: str, subset_path: Optional[str],
                files_discarded: Optional[List[Dict]] = None) -> Dict[str, Any]:
        return {
            "accession": accession,
            "files_downloaded": files_downloaded,
            "files_failed_qc": files_failed_qc,
            "files_discarded": files_discarded or [],
            "outcome_final": outcome_final,
            "flags": flags,
            "subset_path": subset_path,
            "notes": notes or "",
        }


# ---------------------------------------------------------------------- #
#  Matrix subset helper                                                   #
# ---------------------------------------------------------------------- #

def _write_query_subset(local_path: str, out_dir: str, accession: str,
                        query_gsms: set,
                        query_cols: Optional[Set[str]] = None,
                        ) -> Tuple[Optional[str], int, int, str]:
    """
    Best-effort: read the (gzip) matrix, keep the first column (feature id) +
    the query-cancer sample columns, write a subset file.

    Column selection: query_cols (exact names, from the verified column→GSM
    map — handles submitter-coded names) when given; else any column whose
    header contains a query GSM (the legacy heuristic).

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
        if query_cols is not None:
            # Exact-name selection via the map. Mixed-delimiter headers
            # (sample names space-separated inside one tab field) cannot be
            # column-selected with pandas usecols — stream those manually.
            flat = _flatten_header_cols(header_row, sep)
            if flat and len(flat) > len(cols):
                return _write_query_subset_flat(
                    local_path, out_dir, accession, flat, query_cols)
            keep = _dedup_preserve([cols[0]] + [c for c in cols[1:]
                                                if c in query_cols])
        else:
            # keep first column (feature id) + any column whose name contains a query GSM
            keep = [cols[0]] + [c for c in cols[1:] if any(g in c for g in query_gsms)]
            keep = _dedup_preserve(keep)
        if len(keep) <= 1:
            return None, 0, len(cols) - 1, "no query-cancer GSM columns matched in header"

        df = pd.read_csv(local_path, sep=sep, usecols=keep,
                         compression=comp, low_memory=False)
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


def _flatten_header_cols(header_row: str, sep: str) -> List[str]:
    """
    Flatten a possibly mixed-delimiter header (tab fields whose tail field
    packs space-separated sample names) into the true column-name list.
    Returns [] when the plain sep-split already accounts for every token.
    """
    from skills.geo_download.verify import _flatten_header
    flat = _flatten_header(header_row)
    plain = [c.strip().strip('"') for c in header_row.split(sep)]
    return flat if len(flat) > len(plain) else []


def _write_query_subset_flat(local_path: str, out_dir: str, accession: str,
                             flat_cols: List[str], query_cols: Set[str],
                             ) -> Tuple[Optional[str], int, int, str]:
    """
    Subset a MIXED-delimiter matrix (annotation cols tab-separated, sample
    values tab-separated in data rows but names space-packed in the header).
    pandas usecols cannot address these columns, so stream lines: keep the
    leading tab-fields that precede the sample block + the sample fields
    whose flattened names are query columns. Field COUNTS come from the data
    rows (the header undercounts by packing).
    """
    try:
        keep_idx: List[int] = []
        n_ann = 0
        for i, c in enumerate(flat_cols):
            if c in query_cols:
                keep_idx.append(i)
            elif not keep_idx:
                n_ann = i + 1  # leading annotation block
        if not keep_idx:
            return None, 0, len(flat_cols) - n_ann, \
                "no query-cancer columns matched in flattened header"

        subset_path = Path(out_dir) / f"{accession}_query_subset.txt.gz"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        n_kept = len(keep_idx)
        with _open_maybe_gz(local_path) as src, \
                gzip.open(subset_path, "wt", encoding="utf-8") as dst:
            wrote_header = False
            for line in src:
                s = line.rstrip("\n")
                if not s or s.startswith("!"):
                    continue
                if not wrote_header:
                    # Re-emit a clean tab header: annotation names + kept samples.
                    names = [c for c in flat_cols[:n_ann]] + \
                            [flat_cols[i] for i in keep_idx]
                    dst.write("\t".join(names) + "\n")
                    wrote_header = True
                    continue
                fields = s.split("\t")
                # Data rows: first n_ann-1 tab fields are annotation values,
                # then one tab field per sample column (tab-separated).
                ann_vals = fields[:n_ann - 1]
                sample_vals = fields[n_ann - 1:]
                kept_vals = [sample_vals[i - n_ann + 1]
                             for i in keep_idx if i - n_ann + 1 < len(sample_vals)]
                dst.write("\t".join(ann_vals + kept_vals) + "\n")
        return str(subset_path), n_kept, len(flat_cols) - n_ann, \
            f"subset: kept {n_kept}/{len(flat_cols) - n_ann} query-cancer sample columns"
    except Exception as e:
        return None, 0, 0, f"subset failed: {e}"


def _md5_file(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """Stream-md5 a file; returns None on error (best-effort provenance)."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _delete_file(path: Optional[str], acc: str, name: str) -> None:
    """Best-effort delete; logs a warning on failure (never raises)."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError as e:
        logger.warning(f"geo-download {acc}: could not delete {name}: {e}")


def _read_head_text(path: Optional[str], max_lines: int = 10,
                    max_bytes: int = 1 << 20) -> str:
    """
    Decompress (gzip-tolerant) the head of a local file and return the first
    max_lines non-empty lines (truncated to ~2 KB) as plain text — for LLM context.
    """
    if not path:
        return "(no file)"
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes)
        from skills.geo_filter.file_inspect import _decompress_head
        text = _decompress_head(raw)
    except Exception as e:
        return f"(could not read head: {e})"
    lines = [ln for ln in text.splitlines() if ln.strip()][:max_lines]
    out = "\n".join(lines)
    return out[:2048] if out else "(empty)"


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _safe_json(raw: str) -> Dict[str, Any]:
    """Parse JSON, tolerating leading/trailing text and code fences."""
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


# System prompt for Tier-2 sample-type-aware supplementary file selection.
_FILE_SELECTION_SYSTEM_PROMPT = """\
You are a bioinformatics curator selecting which GEO supplementary file(s) to KEEP
for a DNA-methylation analysis.

For each candidate file you are given its decompressed head (header row + first data
rows). Decide KEEP vs DROP based on whether the file's COLUMNS are samples of the
type the user requested (e.g. the query wants plasma/cfDNA → keep plasma-sample
files; DROP tissue-only files even if they are valid methylation matrices).

Reason from:
- the requested sample type and the TARGET SAMPLES summary (the sample set already
  selected for this query — its source_name/group tells you what specimen type is
  wanted);
- each file's header: column names often encode submitter codes whose prefix/label
  indicates specimen (e.g. tissue tumor Tcrc/Tnm vs plasma Pcrc/Pn); use the study
  design + sample counts to map columns → specimen type;
- value shape only to reject obvious non-data (a marker list, a differential p-value
  table with no sample columns). Do NOT reject a file just because values are integer
  read-counts or a non-0–1 methylation score — those are valid methylation matrices.

If multiple files together comprise the requested cohort (e.g. one file per
processing batch), KEEP all of them. If unsure whether a file matches, prefer KEEP.

Output ONLY valid JSON:
{
  "reasoning": "<one or two sentences>",
  "files": [
    {"name": "<exact filename>", "keep": true|false,
     "sample_type": "plasma|tissue|wbc|cell_line|mixed|unknown",
     "reason": "<short reason>"}
  ]
}
"""


def _dedup_preserve(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _read_sample_metadata(accession: str, output_dir: str) -> Optional[pd.DataFrame]:
    """Read existing sample_metadata.csv (written by filter). Returns None if missing."""
    from pathlib import Path
    csv_path = Path(output_dir) / accession / "sample_metadata.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.debug(f"_read_sample_metadata({accession}): {e}")
        return None


def _read_downloadable_gsms(accession: str, output_dir: str,
                            task_id: Optional[str] = None,
                            registry=None) -> List[str]:
    """
    Resolve the GSM IDs marked "download" for this download job.

    task_id + registry → query the GSM-grain query_sample_map (task-routed, correct
    when a GSE is hit by multiple queries). Otherwise fall back to reading the
    rightmost per-task column of sample_metadata.csv (the historical behaviour,
    used when no task_id is available or the registry has no rows yet).
    Only used in Tier 3 (GSM page scraping).
    """
    # Registry-first: exact task routing.
    if task_id and registry is not None:
        try:
            gsms = registry.get_downloadable_gsms(task_id, accession)
            if gsms:
                logger.info(
                    f"_read_downloadable_gsms({accession}): {len(gsms)} GSMs via "
                    f"registry (task={task_id[:8]})")
                return gsms
            # Empty registry result → still fall through to CSV (task may predate the
            # GSM-level registry, i.e. not backfilled). A genuine "0 downloads" verdict
            # is indistinguishable here, so the CSV is the safer tiebreaker.
        except Exception as e:
            logger.debug(f"_read_downloadable_gsms({accession}): registry lookup failed: {e}")

    # CSV fallback: rightmost 8-char task column.
    from pathlib import Path
    csv_path = Path(output_dir) / accession / "sample_metadata.csv"
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
        # Find the latest task_id column (8-char names, rightmost)
        base_cols = {"gsm", "source_name", "molecule", "group", "cancer"}
        tid_cols = [c for c in df.columns if c not in base_cols and len(c) == 8]
        if not tid_cols:
            return []
        col = tid_cols[-1]  # latest = rightmost
        downloadable = df[df[col] == "download"]["gsm"].tolist()
        logger.info(f"_read_downloadable_gsms({accession}): {len(downloadable)} GSMs marked download (col={col})")
        return [str(g) for g in downloadable if pd.notna(g)]
    except Exception as e:
        logger.warning(f"_read_downloadable_gsms({accession}): {e}")
        return []


def _build_series_matrix_task(accession: str, output_dir: str) -> List[Dict[str, Any]]:
    """Build a single download task for the series_matrix (Tier 1)."""
    prefix = accession[:-3] + "nnn"
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/"
           f"matrix/{accession}_series_matrix.txt.gz")
    return [{
        "accession": accession,
        "url": url,
        "filename": f"{accession}_series_matrix.txt.gz",
        "subdir": accession,
    }]
