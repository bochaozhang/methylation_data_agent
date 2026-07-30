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
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage

from skills.geo_download.cancer_label import (
    build_sample_metadata_with_cancer,
    query_cancer_terms,
)
from skills.geo_filter.file_inspect import inspect_matrix_head
from tools.download_tools import DownloadEngine, build_geo_download_tasks
from tools.geo_tools import GEOClient
from utils.llm_factory import get_llm
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
        # LLM for Tier-2 sample-type-aware file selection (None if unconfigured →
        # the selector falls back to keeping all non-junk files + manual_review).
        try:
            self.llm = get_llm(config.get("llm") or {})
        except Exception as exc:
            logger.warning(f"geo-download: LLM unavailable ({exc}); file selection will keep all non-junk")
            self.llm = None

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
        Download + cancer-subset ONE dataset record. Three-tier fallback:
          1. series_matrix has data → download series_matrix (all samples in one file)
          2. supplementary files exist → download those
          3. neither → scrape GSM pages for per-sample beta tables (only "download" GSMs)
        """
        acc = rec.get("accession", "?")
        flags = rec.get("flags", "")

        # Phase 2b: read existing sample_metadata.csv (written by filter).
        sm = _read_sample_metadata(acc, output_dir)

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
                gsm_list = _read_downloadable_gsms(acc, output_dir)
                logger.info(f"geo-download {acc}: Tier 3 (scraping {len(gsm_list)} GSM pages)")
                tasks = []
                for gsm_id in gsm_list:
                    info = self.geo_client.fetch_gsm_supplementary_file(gsm_id)
                    if info:
                        url = info["url"].replace("ftp://", "https://", 1)
                        tasks.append({
                            "accession": acc,
                            "url": url,
                            "filename": info["filename"],
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

        # ---- Tier 2 file selection ----
        # We downloaded every non-RAW supp file. Now select which to KEEP:
        #   1. junk-filter (demoted inspect_matrix_head) drops clear non-data files
        #      (empty/README, p-value/logFC tables) — NOT a value-range A-level gate.
        #   2. the LLM picks files whose SAMPLE TYPE matches the query (e.g. plasma
        #      cfDNA vs tissue), using the query + the target-sample set + each file's
        #      head. Conservative fallback keeps all non-junk + manual_review.
        discarded: List[Dict[str, Any]] = []
        selection_note = ""
        sel_forced: Optional[str] = None
        if tier_used.startswith("2"):
            done, discarded, selection_note, sel_forced = self._select_relevant_files(
                acc, done, rec, sm)

        # ---- Cancer subset ----
        subset_path, subset_note, forced_outcome = self._subset_by_cancer(
            acc, done, sm, output_dir,
            query_terms=query_terms, cancer_type=rec.get("cancer_type"))
        forced_outcome = forced_outcome or sel_forced  # selection fallback may flag review

        files_downloaded = [
            {
                "name": (r.get("local_path") or "").split("/")[-1],
                "local_path": r.get("local_path"),
                "size_bytes": r.get("file_size_bytes"),
                "qc_passed": bool(r.get("local_path")),
                "data_form": r.get("data_form") or rec.get("available_file_type"),
                "provenance": {"source_url": r.get("url"), "checksum_md5": r.get("checksum_md5")},
            }
            for r in done
        ]
        outcome = forced_outcome or ("download_success" if done else "failed")
        notes = subset_note + f" [tier={tier_used}]"
        if selection_note:
            notes += f"; {selection_note}"
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
        return self._result(acc, files_downloaded, [], outcome, flags,
                            notes=notes, subset_path=subset_path,
                            files_discarded=discarded)

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
    #  Tier-2 file selection (junk filter + LLM sample-type selection)   #
    # ------------------------------------------------------------------ #

    def _select_relevant_files(
        self, acc: str, done_results: List[Dict[str, Any]], rec: Dict[str, Any],
        sm: Optional[pd.DataFrame],
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
            kept, llm_discarded, llm_note = self._llm_select_files(acc, candidates, rec, sm)
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
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Ask the LLM which candidate files contain samples matching the query sample
        type. Returns (kept, discarded, note). Raises on parse/error (caller falls back).
        """
        # Target-sample summary from sample_metadata (the LLM-selected "download" set).
        target_summary = self._target_sample_summary(sm)

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
    def _target_sample_summary(sm: Optional[pd.DataFrame]) -> str:
        """One-line summary of the LLM-selected 'download' sample set for the prompt."""
        if sm is None or sm.empty or "gsm" not in sm.columns:
            return "(sample_metadata unavailable)"
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


def _read_downloadable_gsms(accession: str, output_dir: str) -> List[str]:
    """
    Read sample_metadata.csv and return GSM IDs marked "download" in the
    latest task_id column. Only used in Tier 3 (GSM page scraping).
    """
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
