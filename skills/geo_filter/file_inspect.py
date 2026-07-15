"""
File-form inspection for the geo_filter skill (Phase 2, "核验前置").

`inspect_matrix_head(raw_bytes)` decompresses (gzip, truncation-tolerant) the
head of a GEO supplementary file and classifies it as a methylation VALUE matrix
(β / M-value / ratio / paired counts) or not — without downloading the whole
file. Used by the filter node to upgrade files[] from "metadata prediction" to
"file-head evidence" (A-level or not) BEFORE the download skill fetches it.

Heuristic, not exhaustive: when the head cannot be parsed (compressed format we
don't handle, binary, empty), it returns value_type="unknown" and the caller
falls back to the metadata-level prediction.
"""
from __future__ import annotations

import gzip
import io
import re
import zlib
from typing import Any, Dict, List

_GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)

_A_LEVEL_VALUE_TYPES = {"beta", "m_value", "ratio", "paired_counts"}

# Header-name hints.
_BETA_HINTS = ("beta", "β", "methylation level", "beta_value", "avgbeta")
_MVALUE_HINTS = ("m_value", "m-value", "mvalue")
_RATIO_HINTS = ("ratio", "fraction", "percent", "proportion")
_PAIRED_HINTS = ("methylated", "unmethylated", "meth_count", "unmeth_count",
                 "met_count", "unmet_count", "n_meth", "n_unmeth")
_NON_METH_HINTS = ("pvalue", "p_value", "padj", "fdr", "logfc", "log2fc",
                   "chrom", "chr", "pos", "position", "coordinate", "count")  # count alone is ambiguous


def _decompress_head(raw: bytes, limit: int = 1 << 20) -> str:
    """Decode head bytes to text; gunzip if needed, tolerating truncation."""
    if not raw:
        return ""
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        # Raw-deflate (skip the 10-byte gzip header) tolerates truncation best.
        try:
            d = zlib.decompressobj(-zlib.MAX_WBITS)
            out = d.decompress(raw[10:], limit)
            if out:
                return out.decode("utf-8", errors="replace")
        except Exception:
            pass
        # Fallback: gzip module (may raise on truncation; keep what it read).
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
                return f.read(limit).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return raw.decode("utf-8", errors="replace")


def _detect_delim(line: str) -> str:
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return None


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def inspect_matrix_head(raw: bytes) -> Dict[str, Any]:
    """
    Inspect the head of a (possibly gzip) GEO supplementary file.

    Returns:
        {value_type, n_columns, header_fields, gsm_in_columns, is_A_level, reason}
        value_type ∈ {beta | m_value | ratio | paired_counts | non_methylation | unknown}
        is_A_level True when it looks like a methylation VALUE matrix with
        sample-like columns.
    """
    text = _decompress_head(raw)
    if not text.strip():
        return _result("unknown", 0, [], False, False, "empty / unparseable head")

    lines = [ln for ln in text.splitlines() if ln.strip()][:25]
    if not lines:
        return _result("unknown", 0, [], False, "no lines")

    delim = _detect_delim(lines[0])
    rows: List[List[str]] = [ln.split(delim) if delim else [ln.strip()] for ln in lines]
    header = [c.strip().strip('"').strip("'") for c in rows[0]]
    n_cols = len(header)
    header_lower = " ".join(header).lower()
    data_rows = rows[1:]

    # Numeric values from data cells (skip the first column — usually probe/gene id).
    vals: List[float] = []
    for r in data_rows:
        for cell in r[1:]:
            if _is_float(cell):
                vals.append(float(cell))

    gsm_in_columns = any(_GSM_RE.search(h) for h in header)

    # ---- paired counts (methylated/unmethylated column pairs) ----
    if vals and any(h in header_lower for h in _PAIRED_HINTS):
        return _result("paired_counts", n_cols, header, gsm_in_columns, True,
                       "paired methylated/unmethylated counts")

    if not vals:
        return _result("unknown", n_cols, header, gsm_in_columns, False,
                       "no numeric values in head")

    vmin, vmax = min(vals), max(vals)
    in_01 = all(0.0 <= v <= 1.0 for v in vals)
    in_0100 = all(0.0 <= v <= 100.0 for v in vals)
    has_negative = vmin < 0

    # ---- clearly NOT methylation values (statistical / coordinate columns) ----
    if any(h in header_lower for h in ("pvalue", "p_value", "padj", "fdr", "logfc", "log2fc")):
        return _result("non_methylation", n_cols, header, gsm_in_columns, False,
                       "statistical columns (p/logFC), not methylation values")

    # ---- β / ratio (0–1 range) ----
    if in_01:
        if any(h in header_lower for h in _BETA_HINTS):
            return _result("beta", n_cols, header, gsm_in_columns, True, "β-values 0–1")
        if any(h in header_lower for h in _RATIO_HINTS):
            return _result("ratio", n_cols, header, gsm_in_columns, True, "ratio 0–1")
        # 0–1 with sample-ish columns → treat as β (most common methylation form)
        if _looks_like_sample_matrix(n_cols, gsm_in_columns):
            return _result("beta", n_cols, header, gsm_in_columns, True,
                           "0–1 values, sample columns (assumed β)")
        return _result("unknown", n_cols, header, gsm_in_columns, False,
                       "0–1 values but not sample-shaped")

    # ---- M-value (any real, often negative) ----
    if has_negative or (vmax > 1.0 and vmin < 0.0):
        if any(h in header_lower for h in _MVALUE_HINTS) or _looks_like_sample_matrix(n_cols, gsm_in_columns):
            return _result("m_value", n_cols, header, gsm_in_columns, True,
                           "M-value range (negatives / >1)")
        return _result("unknown", n_cols, header, gsm_in_columns, False,
                       "real values, not clearly M-value")

    # ---- ratio 0–100 ----
    if in_0100 and any(h in header_lower for h in _RATIO_HINTS):
        return _result("ratio", n_cols, header, gsm_in_columns, True, "ratio 0–100")

    # ---- large integers / single counts (raw read counts, enrichment) ----
    mostly_int = sum(1 for v in vals if abs(v - round(v)) < 1e-9) > 0.8 * len(vals)
    if mostly_int and vmax > 50:
        return _result("non_methylation", n_cols, header, gsm_in_columns, False,
                       "integer counts (raw / enrichment), not methylation level")

    return _result("unknown", n_cols, header, gsm_in_columns, False,
                   f"unrecognized value range [{vmin:.3g},{vmax:.3g}]")


def _looks_like_sample_matrix(n_cols: int, gsm_in_columns: bool) -> bool:
    """Heuristic: enough columns to be a samples×features matrix."""
    return gsm_in_columns or n_cols >= 4


def _result(value_type: str, n_cols: int, header: List[str],
            gsm_in_columns: bool, is_A_level: bool, reason: str) -> Dict[str, Any]:
    return {
        "value_type": value_type,
        "n_columns": n_cols,
        "header_fields": header[:12],
        "gsm_in_columns": gsm_in_columns,
        "is_A_level": is_A_level and value_type in _A_LEVEL_VALUE_TYPES,
        "reason": reason,
    }


def verify_a_level_files(supplementary_files: List[str], geo_client: Any,
                         max_files: int = 4):
    """
    Inspect the heads of a candidate's real supplementary file URLs to determine
    which are A-level methylation matrices. series_matrix files are TRUSTED as
    A-level (GEO-compiled β matrix) without fetching.

    Returns (has_A_level, files_list, a_level_data_form) where files_list items
    are {name, url, is_A_level, download, data_form, reason}.
    """
    files: List[Dict[str, Any]] = []
    has_A = False
    a_level_form = None
    for url in (supplementary_files or [])[:max_files]:
        name = url.rsplit("/", 1)[-1] or url
        if "series_matrix" in name:
            files.append({"name": name, "url": url, "is_A_level": True,
                          "download": True, "data_form": "series_matrix",
                          "reason": "GEO-compiled series matrix (trusted)"})
            has_A = True
            a_level_form = a_level_form or "series_matrix"
            continue
        raw = geo_client.fetch_file_head(url)
        if not raw:
            files.append({"name": name, "url": url, "is_A_level": False,
                          "download": False, "data_form": "unknown",
                          "reason": "could not fetch/inspect head"})
            continue
        info = inspect_matrix_head(raw)
        is_A = bool(info["is_A_level"])
        files.append({"name": name, "url": url, "is_A_level": is_A,
                      "download": is_A,
                      "data_form": info["value_type"] if is_A else "unknown",
                      "reason": info["reason"]})
        if is_A:
            has_A = True
            a_level_form = a_level_form or info["value_type"]
    return has_A, files, a_level_form
