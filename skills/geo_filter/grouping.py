"""
GSM sample grouping for the geo_filter skill.

The canonical keyword table lives in tools/geo_tools.py:SAMPLE_GROUP_KEYWORDS
and is already consumed by GEOClient.get_representative_gsm_details() and
get_all_gsm_metadata(). We re-export it here so the skill can classify titles
and build the per-sample annotation column without duplicating the table.

Single source of truth → edit keywords in ONE place (geo_tools.py).
"""
from typing import Dict, List

# Re-export the canonical keyword table (do NOT duplicate).
from tools.geo_tools import SAMPLE_GROUP_KEYWORDS  # noqa: F401

# Stable ordering for CSV/summary output.
GROUP_ORDER: List[str] = [
    "plasma_cfdna", "tissue", "wbc_blood", "normal", "cell_line", "other",
]


def classify_group(title: str) -> str:
    """
    Assign a single GSM title to one biological-material group.

    Returns one of: plasma_cfdna | tissue | wbc_blood | normal | cell_line | unknown
    (matches the 'group' values produced by GEOClient).
    """
    t = (title or "").lower()
    for group_name, keywords in SAMPLE_GROUP_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return group_name
    return "unknown"


def group_summary(gsm_list: List[Dict]) -> Dict[str, int]:
    """Count GSMs per group using each item's 'group' field (or reclassify)."""
    counts: Dict[str, int] = {g: 0 for g in GROUP_ORDER}
    counts["unknown"] = 0
    for g in gsm_list:
        grp = g.get("group") or classify_group(g.get("source_name", "") + " " + g.get("title", ""))
        counts[grp] = counts.get(grp, 0) + 1
    return counts
