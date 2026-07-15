"""
Agent1 pipeline state + canonical SearchIntent.

The agent1 pipeline (agents/agent1_pipeline.py) is a deterministic LangGraph:
  parse → { geo-search → geo-filter → geo-download  //  tcga-direct } → register

`Agent1State` is the single state object flowing through that graph; nodes add
fields as they run. `SearchIntent` is the canonical structured query produced by
the `parse` node — it converges the two inconsistent shapes returned by
`parse_query_with_llm` (rich/nested) and `parse_query_rules` (flat/missing),
so downstream skills (geo-search, geo-filter) never have to do `isinstance`
duck-typing.

`normalize_intent(raw_query, parsed)` maps either parser's output onto
SearchIntent. The dead LLM-byproduct fields `geo_search_query` /
`pubmed_search_query` are dropped (geo-search rebuilds the GEO query
deterministically via build_geo_search_string).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from typing_extensions import TypedDict

# ---------------------------------------------------------------------- #
#  Canonical search intent                                               #
# ---------------------------------------------------------------------- #

_LIQUID_SAMPLE_TYPES = {"cfdna", "plasma", "serum", "whole_blood", "wbc"}
_TISSUE_SAMPLE_TYPES = {"tumor", "adjacent", "normal", "non_cancer"}


class CancerType(TypedDict, total=False):
    display: str            # "colorectal cancer" / "结直肠癌"
    tcga_code: str          # "COAD" — drives synonym expansion + TCGA search
    mesh_term: Optional[str]  # "Colorectal Neoplasms" (PubMed, optional)


class Accessions(TypedDict, total=False):
    geo: List[str]          # ["GSE124600", ...]
    tcga: List[str]         # ["TCGA-COAD", ...]


class SearchIntent(TypedDict):
    # identity
    raw_query: str
    mode: Literal["accession", "semantic"]
    parse_method: Literal["llm", "rules_fallback"]
    # accession mode
    accessions: Accessions
    # semantic drivers
    cancer_type: Optional[CancerType]
    platform: Optional[str]               # "450K" | "EPIC" | "WGBS" | "RRBS" | None
    data_type: Optional[str]              # "array" | "sequencing" | "both" | None
    sample_type: Optional[str]            # primary requested sample type
    sample_types: List[str]               # all requested sample types
    sample_type_detail: str               # free-text elaboration ("" if none)
    focus: Literal["liquid_biopsy", "tissue", "both"]
    year_start: Optional[int]
    year_end: Optional[int]
    extra_keywords: List[str]             # user-supplied extra terms ([])
    notes: str                            # special instructions ("")


def _derive_focus(sample_types: List[str]) -> Literal["liquid_biopsy", "tissue", "both"]:
    """Coarse focus hint derived from fine-grained sample_types."""
    has_liquid = any(s in _LIQUID_SAMPLE_TYPES for s in sample_types or [])
    has_tissue = any(s in _TISSUE_SAMPLE_TYPES for s in sample_types or [])
    if has_liquid and has_tissue:
        return "both"
    if has_liquid:
        return "liquid_biopsy"
    if has_tissue:
        return "tissue"
    return "both"


def normalize_intent(raw_query: str, parsed: Dict[str, Any]) -> SearchIntent:
    """
    Map either parse_query_with_llm() or parse_query_rules() output onto the
    canonical SearchIntent. Always sets raw_query; unifies cancer_type to a
    nested dict; fills missing fields with defaults.
    """
    # ---- accessions: tolerate dict {geo,tcga,...} or other shapes ----
    raw_acc = parsed.get("accessions") or {}
    if isinstance(raw_acc, dict):
        geo_acc = list(raw_acc.get("geo") or [])
        tcga_acc = list(raw_acc.get("tcga") or [])
    else:
        geo_acc, tcga_acc = [], []

    # ---- cancer_type: nested dict (LLM) vs flat fields (rules) ----
    ct = parsed.get("cancer_type")
    if isinstance(ct, dict):
        cancer_type: Optional[CancerType] = {
            "display": ct.get("display") or parsed.get("cancer_type_display"),
            "tcga_code": ct.get("tcga_code") or parsed.get("cancer_type_code"),
            "mesh_term": ct.get("mesh_term"),
        }
        if not cancer_type["display"] and not cancer_type["tcga_code"]:
            cancer_type = None
    elif parsed.get("cancer_type_display") or parsed.get("cancer_type_code"):
        cancer_type = {
            "display": parsed.get("cancer_type_display"),
            "tcga_code": parsed.get("cancer_type_code"),
            "mesh_term": None,
        }
    else:
        cancer_type = None

    sample_types = list(parsed.get("sample_types") or [])
    if not sample_types and parsed.get("sample_type"):
        sample_types = [parsed["sample_type"]]

    return SearchIntent(
        raw_query=raw_query,
        mode=parsed.get("mode", "semantic"),
        parse_method=parsed.get("parse_method", "rules_fallback"),
        accessions={"geo": geo_acc, "tcga": tcga_acc},
        cancer_type=cancer_type,
        platform=parsed.get("platform"),
        data_type=parsed.get("data_type"),
        sample_type=parsed.get("sample_type"),
        sample_types=sample_types,
        sample_type_detail=parsed.get("sample_type_detail") or "",
        focus=_derive_focus(sample_types),
        year_start=parsed.get("year_start"),
        year_end=parsed.get("year_end"),
        extra_keywords=list(parsed.get("extra_keywords") or []),
        notes=parsed.get("notes") or "",
    )


# ---------------------------------------------------------------------- #
#  Pipeline state                                                        #
# ------------------------------------------------------------------ #


class Agent1State(TypedDict, total=False):
    """Accumulative state for the agent1 pipeline graph."""

    # ---- input (set by the caller / daemon) ----
    raw_query: str
    config: Dict[str, Any]
    registry: Any
    output_dir: str

    # ---- parse node ----
    parsed_intent: SearchIntent

    # ---- geo-search node ----
    candidate_gse_list: List[Dict[str, Any]]
    search_queries: List[str]
    search_log: str

    # ---- geo-filter node (four-state) ----
    download_list: List[Dict[str, Any]]
    lead_list: List[Dict[str, Any]]
    exclude_list: List[Dict[str, Any]]
    manual_review_list: List[Dict[str, Any]]
    filter_log: str

    # ---- geo-download node (removed: no inline download; downloads happen in
    #      the daemon after the bulk "待下载" confirm) ----

    # ---- tcga-direct node (search-only; returns candidates, no download) ----
    tcga_candidates: List[Dict[str, Any]]

    # ---- global ----
    error_log: List[str]
