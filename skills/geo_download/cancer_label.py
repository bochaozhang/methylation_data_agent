"""
Per-GSM cancer labeling for the geo_download skill (Phase 2b).

Builds sample_metadata.csv with a `cancer` column so the download skill can
subset a multi-cancer matrix to the query cancer's GSMs. Labeling is heuristic:
match the query cancer's terms (display name + synonyms from
skills/geo_search/synonyms.yaml) against each GSM's `characteristics` values.
GSMs that don't match → "unclear" (the download skill sends unclear-majority
datasets to manual_review for a human to label).

No per-GSM LLM (too expensive at hundreds of GSMs); "尽量补" per the user —
heuristic first, unclear → manual review.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from tools.parser_tools import _get_cancer_synonyms, _load_cancer_synonyms
from utils.logger import get_logger

logger = get_logger(__name__)

_CONTROL_TERMS = (
    "healthy", "normal", "control", "benign",
    "non-cancer", "non_cancer", "non-tumor", "nontumor",
)


def query_cancer_terms(intent: Dict[str, Any]) -> List[str]:
    """Lowercased search terms for the query cancer (display + synonyms)."""
    ct = intent.get("cancer_type") or {}
    if not isinstance(ct, dict):
        ct = {}
    raw: List[str] = []
    if ct.get("display"):
        raw.append(str(ct["display"]))
    code = ct.get("tcga_code")
    if code:
        raw.extend(_get_cancer_synonyms(code))
    seen, out = set(), []
    for t in raw:
        tl = t.lower()
        if tl and tl not in seen:
            seen.add(tl)
            out.append(tl)
    return out


def query_terms_from_label(label: str) -> List[str]:
    """
    Reverse-lookup query cancer terms from a free-text cancer label (e.g. the
    registry's cancer_type column) — used when downloading a human-approved
    dataset outside the original query context. Falls back to just the label.
    """
    from tools.parser_tools import TCGA_CODE_TO_ENGLISH

    label = (label or "").lower().strip()
    if not label:
        return []
    data = _load_cancer_synonyms()
    raw = [label]
    for code, syns in data.get("cancer_synonyms", {}).items():
        english = TCGA_CODE_TO_ENGLISH.get(code, "").lower()
        bucket = [english] + [str(s).lower() for s in syns]
        if label in bucket or any(label in s or s in label for s in bucket if s):
            raw.extend(syns)
            break
    seen, out = set(), []
    for t in raw:
        tl = t.lower()
        if tl and tl not in seen:
            seen.add(tl)
            out.append(tl)
    return out


def label_gsm_cancer(characteristics: Dict[str, Any], query_terms: List[str]) -> str:
    """
    Label one GSM's cancer from its characteristics.

    Returns:
        "<query cancer>" if characteristics mention a query-cancer term,
        "control" if they mention a healthy/normal/control term,
        "unclear" otherwise (could be another cancer or unlabeled).
    """
    vals = " ".join(str(v) for v in (characteristics or {}).values()).lower()
    if not vals.strip():
        return "unclear"
    if any(t and t in vals for t in query_terms):
        return "query_cancer"
    if any(c in vals for c in _CONTROL_TERMS):
        return "control"
    return "unclear"


def build_sample_metadata_with_cancer(
    accession: str,
    geo_client: Any,
    query_terms: List[str],
    output_dir: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch all GSMs for an accession, label each GSM's cancer, and write
    {output_dir}/{accession}/sample_metadata.csv with a `cancer` column.

    Returns the DataFrame (or None if no GSMs could be fetched).
    """
    try:
        gsm_list = geo_client.get_all_gsm_metadata(accession)
    except Exception as e:
        logger.warning(f"build_sample_metadata_with_cancer({accession}): GSM fetch failed: {e}")
        return None
    if not gsm_list:
        return None

    rows = []
    for g in gsm_list:
        rows.append({
            "gsm": g.get("gsm", ""),
            "source_name": g.get("source_name", ""),
            "molecule": g.get("molecule", ""),
            "group": g.get("group", "unknown"),
            "cancer": label_gsm_cancer(g.get("characteristics") or {}, query_terms),
        })
    df = pd.DataFrame(rows)

    try:
        csv_path = Path(output_dir) / accession / "sample_metadata.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"sample_metadata({accession}): {len(df)} GSMs, "
                    f"cancer counts={df['cancer'].value_counts().to_dict()}")
    except Exception as e:
        logger.warning(f"sample_metadata({accession}): write failed: {e}")
    return df
