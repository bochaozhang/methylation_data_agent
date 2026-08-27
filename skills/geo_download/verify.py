"""
geo-download Phase-2 verification — column count, GSM→column mapping,
group completeness, quarantine.

Pure(ish) helpers invoked by DownloadSkill.process_dataset after the LLM file
selection. Design inputs (verified on production data, GSE124600):

  * Sample-column counting must use DATA rows: submitter files can have a
    header whose sample names are space-separated inside one tab field while
    the data rows are pure-tab (8 vs 275 fields on the same file).
  * Column counts aggregate ACROSS kept files (6 + 268 = 274 = the task's
    download-GSM count).
  * series_matrix !Sample_title ↔ !Sample_geo_accession rows are
    column-index-aligned, giving a deterministic mapping for submitter-coded
    column names (Pcrc90, Pn43_m, Pcrc1_m_dup1.5, ...).

Conservative philosophy (user decision 2026-08-21): generous tolerance
(max(5, 10%)) for the column-count check; group-completeness gaps and
low-confidence mappings are REPORTED ONLY (notes / manual_review flag) —
only the column-count check can quarantine.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# Column-count tolerance: |n_cols - n_gsms| <= max(5, 10% of n_gsms).
_COL_TOLERANCE_ABS = 5
_COL_TOLERANCE_FRAC = 0.10

# Mapping coverage below this (when column names exist to map) → manual_review
# flag (report-only, files stay in place).
_MAP_COVERAGE_REVIEW = 0.5

_GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)

# Annotation / coordinate prefix columns that are NOT samples. Matched on the
# whole (case-insensitive) column name — feature/coordinate vocabulary seen in
# GEO methylation supplements (MCTA-Seq coordinate block, ID_REF, probe ids).
_ANNOTATION_COL_RE = re.compile(
    r"^(#)?(chr|chrom|chromosome|start|end|str|strand|pos|position|locus|locusid|"
    r"cgi_num|cgcgcgg_num|cg_num|cpg_num|island|probe|probeid|id_ref|idref|"
    r"target|targetid|region|regionid|gene|geneid|gene_symbol|symbol|"
    r"feature|featureid|row_names|rownames|name|id)$",
    re.IGNORECASE,
)

# Separator characters used to flatten a header field into column names.
_WS_SPLIT_RE = re.compile(r"[,\s;]+")


# ---------------------------------------------------------------------- #
#  ① Sample-column counting                                              #
# ---------------------------------------------------------------------- #

def parse_sample_columns(path: str, max_scan_rows: int = 20,
                         max_bytes: int = 8 << 20) -> Optional[Dict[str, Any]]:
    """
    Count a matrix file's SAMPLE columns (annotation/coordinate columns excluded).

    Returns {"n_total_fields", "col_names", "n_annotation", "n_sample_cols",
             "n_gsm_cols", "basis"} or None when nothing parseable was found.
    Columns come from the flattened header (tab-split, then whitespace-split
    inside each field — tolerates the mixed-delimiter header form); the count
    basis is the MODE of the first max_scan_rows data-row field counts, so a
    header with collapsed fields cannot undercount.
    """
    try:
        with _open_text(path) as f:
            lines: List[str] = []
            for line in f:
                lines.append(line.rstrip("\n"))
                if len(lines) >= max_scan_rows:
                    break
    except Exception as e:
        logger.debug(f"parse_sample_columns({os.path.basename(path)}): {e}")
        return None

    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("!")]
    if not data_lines:
        return None
    header, data_lines = data_lines[0], data_lines[1:]
    if not data_lines:
        # Header only — best effort from flattened header.
        col_names = _flatten_header(header)
        n_ann = sum(1 for c in col_names if _ANNOTATION_COL_RE.match(c))
        return {"n_total_fields": len(col_names), "col_names": col_names,
                "n_annotation": n_ann, "n_sample_cols": len(col_names) - n_ann,
                "n_gsm_cols": sum(1 for c in col_names if _GSM_RE.search(c)),
                "basis": "header_only"}

    def _n_fields(line: str) -> int:
        return len(line.split("\t")) if "\t" in line else len(_WS_SPLIT_RE.split(line.strip()))

    counts = Counter(_n_fields(ln) for ln in data_lines)
    n_total, n_rows = counts.most_common(1)[0]
    col_names = _flatten_header(header)
    # A collapsed header (fewer fields than data rows) cannot name the columns —
    # fall back to unknown names but keep the data-row count.
    if len(col_names) < n_total:
        col_names = []

    if col_names:
        n_ann = sum(1 for c in col_names if _ANNOTATION_COL_RE.match(c))
        n_sample = len(col_names) - n_ann
        names = [c for c in col_names if not _ANNOTATION_COL_RE.match(c)]
    else:
        # Unknown names: assume exactly one annotation/feature leading column
        # (the overwhelmingly common form: ID_REF + samples).
        n_ann = 1
        n_sample = n_total - n_ann
        names = []
    return {"n_total_fields": n_total, "col_names": names,
            "n_annotation": n_ann, "n_sample_cols": n_sample,
            "n_gsm_cols": sum(1 for c in names if _GSM_RE.search(c)),
            "basis": f"data_rows({n_rows})"}


def _flatten_header(header: str) -> List[str]:
    """Split a header into column names: tab first, then any whitespace/comma
    runs inside each field (mixed-delimiter headers)."""
    fields: List[str] = []
    for part in header.split("\t"):
        fields.extend(x for x in _WS_SPLIT_RE.split(part.strip().strip('"')) if x)
    return fields


def check_column_counts(per_file: List[Dict[str, Any]], n_gsms: int,
                        per_sample_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Aggregate the per-file sample-column counts against the task's download-GSM
    count. Merged form dedups column NAMES across files (two files may share
    samples); without names, falls back to the sum (generous). Per-sample form
    (one file = one sample) compares FILE counts.

    Returns {"passed", "n_cols", "n_expected", "tolerance", "reason", "form"}.
    n_gsms==0/None → skip (passed, reason="no gsm set").
    """
    per_sample_files = per_sample_files or []
    if not n_gsms:
        return {"passed": True, "n_cols": None, "n_expected": n_gsms,
                "tolerance": None, "reason": "no gsm set", "form": "skip"}

    tol = max(_COL_TOLERANCE_ABS, int(_COL_TOLERANCE_FRAC * n_gsms))
    if per_sample_files:
        n_files = len(per_sample_files)
        ok = abs(n_files - n_gsms) <= tol
        return {"passed": ok, "n_cols": n_files, "n_expected": n_gsms,
                "tolerance": tol,
                "reason": (f"per-sample files {n_files} vs {n_gsms} gsms "
                           f"(tol {tol})"),
                "form": "per_sample"}

    parsed = [p for p in per_file if p]
    if not parsed:
        return {"passed": True, "n_cols": None, "n_expected": n_gsms,
                "tolerance": tol, "reason": "no parseable matrix files", "form": "skip"}

    name_sets: List[List[str]] = [p.get("col_names") or [] for p in parsed]
    if any(name_sets):
        union: Set[str] = set()
        sum_cols = 0
        for names in name_sets:
            sum_cols += len(names)
            union.update(names)
        # Named columns somewhere → union across files (dedup overlap); files
        # without names contribute nothing (cannot identify overlap).
        n_cols = len(union) if union else sum_cols
        form = "merged_names"
    else:
        n_cols = sum(p["n_sample_cols"] for p in parsed)
        form = "merged_sum"
    ok = abs(n_cols - n_gsms) <= tol
    return {"passed": ok, "n_cols": n_cols, "n_expected": n_gsms,
            "tolerance": tol,
            "reason": f"cols {n_cols} vs {n_gsms} gsms (tol {tol})",
            "form": form}


# ---------------------------------------------------------------------- #
#  ② GSM→column mapping                                                  #
# ---------------------------------------------------------------------- #

def parse_series_matrix_samples_local(path: str) -> List[Dict[str, str]]:
    """
    Parse a LOCAL series_matrix file's !Sample_* rows into per-sample dicts
    with keys gsm / title / source_name (same shape as
    GEOClient.fetch_series_matrix_sample_info, without re-downloading).
    """
    out: Dict[str, List[str]] = {}
    try:
        with _open_text(path) as f:
            for line in f:
                if not line.startswith("!Sample_"):
                    if line.startswith("!series_matrix_table_begin"):
                        break  # sample metadata never follows the data table
                    continue
                parts = line.rstrip("\n").split("\t")
                key = {"!Sample_geo_accession": "gsm",
                       "!Sample_title": "title",
                       "!Sample_source_name_ch1": "source_name"}.get(parts[0])
                if not key:
                    continue
                out[key] = [v.strip().strip('"') for v in parts[1:] if v.strip()]
    except Exception as e:
        logger.debug(f"parse_series_matrix_samples_local({os.path.basename(path)}): {e}")
        return []
    n = min((len(v) for v in out.values()), default=0)
    if not n:
        return []
    return [{k: out[k][i] for k in out} for i in range(n)]


def build_gsm_column_map(col_names_by_file: Dict[str, List[str]],
                         series_samples: List[Dict[str, str]],
                         ) -> Dict[str, Any]:
    """
    Map column names → GSMs, cheapest strategy first:

      L1 gsm_id    — column name contains a GSM id (series_matrix form);
      L2 title     — column name equals a series !Sample_title (exact →
                     case/quote-normalized → unique title+suffix prefix);
      L3 none      — unmapped; the caller flags manual_review when coverage
                     is low (report-only; no LLM fallback this phase).

    Returns {"column_map": {col: {"gsm","source"}}, "coverage": float,
             "level": "gsm_id|title|none", "n_cols", "n_mapped"}.
    """
    title_to_gsm: Dict[str, List[str]] = {}
    for s in series_samples:
        t = (s.get("title") or "").strip()
        g = (s.get("gsm") or "").strip()
        if t and g:
            title_to_gsm.setdefault(_norm_token(t), []).append(g)

    column_map: Dict[str, Dict[str, str]] = {}
    level = "none"
    all_cols: List[str] = []
    for cols in col_names_by_file.values():
        all_cols.extend(cols)
    n_cols = len(set(all_cols))
    if not n_cols:
        return {"column_map": {}, "coverage": 0.0, "level": "none",
                "n_cols": 0, "n_mapped": 0}

    # L1: GSM ids embedded in column names.
    for col in set(all_cols):
        m = _GSM_RE.search(col)
        if m:
            column_map[col] = {"gsm": m.group(0).upper(), "source": "gsm_id"}
    if column_map:
        level = "gsm_id"

    # L2: series titles (covers submitter-coded names like Pcrc90 / Pn43_m /
    # Pcrc1_m_dup1.5 — the suffix variants hit the prefix rule).
    if len(column_map) < n_cols and title_to_gsm:
        for col in set(all_cols):
            if col in column_map:
                continue
            nc = _norm_token(col)
            hit: Optional[List[str]] = None
            if nc in title_to_gsm:
                hit = title_to_gsm[nc]
            else:
                # col == title + separator + suffix, but only when UNIQUE
                # (a prefix shared by several titles is ambiguous → skip).
                cands = [g for t, g in title_to_gsm.items()
                         if nc.startswith(t + "\x1f") or _is_title_prefix(nc, t)]
                if len(set(tuple(g) for g in cands)) == 1:
                    hit = cands[0]
            if hit and len(set(hit)) == 1:
                column_map[col] = {"gsm": hit[0], "source": "title"}
                level = level if level == "gsm_id" else "title"

    n_mapped = len(column_map)
    return {"column_map": column_map, "coverage": n_mapped / n_cols,
            "level": level if n_mapped else "none",
            "n_cols": n_cols, "n_mapped": n_mapped}


def _is_title_prefix(col_norm: str, title_norm: str) -> bool:
    """col starts with title followed by a separator-flanked suffix."""
    if not col_norm.startswith(title_norm):
        return False
    rest = col_norm[len(title_norm):]
    return bool(rest) and rest[0] in "_-."


def _norm_token(s: str) -> str:
    """Lowercase, strip quotes/space — for exact-title and prefix matching."""
    return s.strip().strip('"').strip("'").lower().replace(" ", "_")


# ---------------------------------------------------------------------- #
#  ③ Disease-group completeness (report-only)                            #
# ---------------------------------------------------------------------- #

def check_group_completeness(column_map: Dict[str, Dict[str, str]],
                             samples_df,
                             download_gsms: List[str]) -> Dict[str, Any]:
    """
    Given the column→GSM map and the task's download set, report how many
    expected query-cancer / control samples are actually locatable in the
    kept columns. REPORT-ONLY by design (user decision): gaps go to notes,
    never quarantine.

    samples_df: sample_metadata.csv DataFrame (needs gsm + cancer columns).
    Returns {"groups": {label: {"expected","mapped"}}, "gaps": [gsm,...],
             "summary": "q=142/142 c=132/132"}.
    """
    mapped_gsms = {v["gsm"] for v in column_map.values()}
    dl = set(download_gsms or [])
    groups: Dict[str, Dict[str, int]] = {}
    gaps: List[str] = []
    if samples_df is None or samples_df.empty or "gsm" not in getattr(samples_df, "columns", []):
        mapped_dl = mapped_gsms & dl
        return {"groups": {}, "gaps": [],
                "summary": f"{len(mapped_dl)}/{len(dl)} download gsms mapped"}

    sub = samples_df[samples_df["gsm"].astype(str).isin(dl)]
    if "cancer" not in samples_df.columns:
        mapped_dl = mapped_gsms & dl
        return {"groups": {}, "gaps": [],
                "summary": f"{len(mapped_dl)}/{len(dl)} download gsms mapped"}
    for label, grp in sub.groupby("cancer"):
        expected = set(grp["gsm"].astype(str))
        mapped = expected & mapped_gsms
        groups[str(label)] = {"expected": len(expected), "mapped": len(mapped)}
        gaps.extend(sorted(expected - mapped))
    summary = " ".join(
        f"{_short(k)}={v['mapped']}/{v['expected']}" for k, v in groups.items())
    return {"groups": groups, "gaps": gaps,
            "summary": summary or "no labelled download gsms"}


def _short(label: str) -> str:
    return {"query_cancer": "q", "control": "c"}.get(label, label[:8])


# ---------------------------------------------------------------------- #
#  ④ Quarantine                                                          #
# ---------------------------------------------------------------------- #

def quarantine_files(acc: str, results: List[Dict[str, Any]], output_dir: str,
                     reason: str) -> List[Dict[str, Any]]:
    """
    MOVE kept files (download-engine result dicts) to
    {output_dir}/quarantine/{acc}/ after recording md5+url. Moves, never
    deletes; a failed move keeps the file in place (record says so). The
    results' local_path is updated so downstream registry updates stay honest.
    Returns files_failed_qc records.
    """
    qdir = Path(output_dir) / "quarantine" / acc
    records: List[Dict[str, Any]] = []
    for r in results:
        local_path = r.get("local_path")
        name = (local_path or "").split("/")[-1]
        if not local_path or not os.path.exists(local_path):
            continue
        md5 = _md5(local_path)  # before the move — the source vanishes after
        try:
            qdir.mkdir(parents=True, exist_ok=True)
            dest = str(qdir / name)
            shutil.move(local_path, dest)
            r["local_path"] = dest
            moved = True
        except OSError as e:
            logger.warning(f"quarantine {acc}: could not move {name}: {e}")
            dest, moved = local_path, False
        records.append({
            "name": name, "reason": reason, "md5": md5,
            "source_url": r.get("url"),
            "quarantine_path": dest if moved else local_path,
        })
    return records


def _md5(path: str, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _open_text(path: str):
    """Open plain or gzip text for reading (utf-8, replacement)."""
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") \
        if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8",
                                               errors="replace")
