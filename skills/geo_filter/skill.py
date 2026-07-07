"""
geo_filter skill — spec-driven, threshold-free GEO dataset filtering.

The canonical procedure is skills/geo_filter/SPEC.md (a verbatim copy of the
human-authored "GEO 数据检索注意事项 v3"). It is loaded once as a module-level
constant and used as the LLM system prompt, so:

  * it triggers provider prompt caching (DeepSeek / Z.AI implicit cache), and
  * updating filtering behaviour means editing SPEC.md — one file, no code.

Unlike the legacy pipeline, there are NO hardcoded thresholds (the old
5% / 20% / 50% include_fraction rules). The LLM applies the SPEC's qualitative
rules directly and returns the structured verdict defined in SPEC's
"输出格式要求" section.

Public surface:
  - filter_dataset(...): one LLM call → verdict dict (used directly by DatabaseAgent)
  - GeoFilterSkill: Skill wrapper for the orchestrator (Phase 3)
  - apply_verdict(ds, verdict): map a verdict onto a dataset dict for registry upsert
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from skills.base import Skill, SkillContext, register_skill
from skills.geo_filter.grouping import group_summary
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------- #
#  Canonical SPEC — single source of truth for GEO filtering             #
# ---------------------------------------------------------------------- #

_SPEC_PATH = Path(__file__).resolve().parent / "SPEC.md"
SPEC: str = _SPEC_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------- #
#  Machine output contract (appended to SPEC as the system prompt)       #
#  Keeps SPEC.md clean/human-readable; the JSON contract lives here.     #
# ---------------------------------------------------------------------- #

_OUTPUT_CONTRACT = """

========================================
OUTPUT CONTRACT (machine-readable)
========================================
You are also an API. After applying the procedure above, respond with ONLY a
single JSON object (no markdown fences, no prose outside the JSON). The object
MUST have exactly these keys:

{
  "reasoning": "<REQUIRED: step-by-step logic chain — see below>",
  "usable": "yes" | "no" | "partial" | "unclear",
  "recommended_action": "keep" | "exclude" | "article_only" | "manual_review",
  "confirmed_sample_type": "plasma|tumor|adjacent|normal|wbc|cfdna|serum|whole_blood|cell_line|other|unknown",
  "confirmed_cancer_type": "canonical English cancer name, or null",
  "sample_count_in_paper": <integer or null>,
  "stage_treatment": "staging / treatment status from the data or article, or null",
  "consistency": "consistent" | "minor_discrepancy" | "major_discrepancy" | "unknown",
  "sample_level_annotation": "yes" | "no" | "unclear",
  "available_file_type": "best-guess of the usable file type, or null",
  "disease_groups": "case/control breakdown as free text, or null",
  "reason": "one sentence: what the samples are and why keep/exclude/review",
  "notes": "any discrepancy or caveat (sample count mismatch, pooled cfDNA, ...); empty string if none",
  "gsm_includes": [
    {"gsm": "<gsm id>", "include": true|false, "reason": null | "one sentence if excluded"}
  ]
}

Field guidance:
- reasoning (REQUIRED, fill FIRST): a short step-by-step logic chain that walks through, in order:
    1. What biological samples this dataset actually contains (use GSM details + summary + abstract).
    2. Human / target cancer type? (per SPEC)
    3. Sample type vs request — note that plasma / serum ARE cell-free DNA (cfDNA).
    4. Are there non-cancer / control samples as the request needs?
    5. File type / data type (is it methylation? downloadable?)
    6. Therefore → the verdict.
  This chain is logged for human audit, so it MUST make your logic explicit and
  internally consistent. If your chain shows the requested samples ARE present,
  the verdict MUST follow that — do not contradict your own reasoning.
- recommended_action values:
    keep           → usable data matching the request; queue for approval
    exclude        → cell line / organoid / animal / in-vitro / treated / non-target /
                     only marker list / not downloadable — do NOT queue
    article_only   → the article mentions data but GEO does not actually provide it
                     (e.g. validation by qPCR/panel in paper only)
    manual_review  → genuinely ambiguous, conflicting metadata, or cannot confirm
- usable: yes when the dataset provides downloadable sample-level methylation data
  matching the request; partial when only a subset is usable; no when excluded.
- gsm_includes: classify ONLY the representative GSM samples you were given.
  include=true when the sample matches the requested sample type; reason MUST be null.
  include=false when it does not; reason MUST be a short explanation.
- Apply the SPEC's qualitative rules. Do NOT invent numeric thresholds.
"""


SYSTEM_PROMPT: str = SPEC + "\n" + _OUTPUT_CONTRACT


def _parse_spec_name(spec_text: str) -> str:
    """Return the SPEC document name from its first '# ' heading.

    The user manages versioning by renaming the source doc (e.g. updating the
    title to '... v4'); the logger simply records whatever this name is.
    """
    for line in spec_text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("# ").strip() or "unknown"
        elif line:
            break
    return "unknown"


# Name of the filtering SPEC (auto-derived from SPEC.md heading, e.g.
# 'GEO 数据检索注意事项 v3'). Recorded in the per-query CSV log.
SPEC_NAME: str = _parse_spec_name(SPEC)


# ---------------------------------------------------------------------- #
#  Intent → human-readable block                                         #
# ---------------------------------------------------------------------- #

def _intent_block(intent: Dict[str, Any]) -> str:
    ct = intent.get("cancer_type")
    ct_label = (ct.get("display") if isinstance(ct, dict) else str(ct)) if ct else \
        intent.get("cancer_type_display") or "not specified"

    sample_types = intent.get("sample_types") or []
    primary_st = intent.get("sample_type") or ""
    if sample_types:
        st_line = f"Requested sample type(s): {primary_st} (all: {sample_types})"
    elif primary_st:
        st_line = f"Requested sample type: {primary_st}"
    else:
        st_line = "Requested sample type: not specified"

    platform_req = intent.get("platform") or "not specified"

    yr_start = intent.get("year_start")
    yr_end = intent.get("year_end")
    year_line = ""
    if yr_start or yr_end:
        year_line = f"\nRequested year range: {yr_start or 'any'} – {yr_end or 'any'}"

    detail = intent.get("sample_type_detail") or ""
    detail_line = f"\nSample type detail: {detail}" if detail else ""

    raw_q = (intent.get("raw_query") or "").strip()
    query_line = f"\nOriginal user query: {raw_q[:200]}" if raw_q else ""

    return (
        f"Requested cancer type: {ct_label}\n"
        f"{st_line}\n"
        f"Requested platform: {platform_req}"
        f"{year_line}{detail_line}{query_line}"
    )


def _gsm_block(gsm_details: List[Dict[str, Any]]) -> str:
    """Render representative GSM details + group counts as evidence."""
    if not gsm_details:
        return "(no representative GSM details available)"

    counts = group_summary(gsm_details)
    counts_line = ", ".join(f"{g}={n}" for g, n in counts.items() if n)

    lines = [f"Representative samples (group counts: {counts_line}):"]
    for g in gsm_details:
        ch = g.get("characteristics") or {}
        ch_str = "; ".join(f"{k}: {v}" for k, v in ch.items()) if ch else "(none)"
        lines.append(
            f"  - GSM {g.get('gsm', '?')} [group={g.get('group', '?')}]: "
            f"source_name={g.get('source_name', '')!r}, "
            f"molecule={g.get('molecule', '')!r}, characteristics={{{ch_str}}}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
#  Core: one spec-driven LLM call → verdict                              #
# ---------------------------------------------------------------------- #

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
        # Fall back to the first {...} balanced block.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def filter_dataset(
    llm: Any,
    ds: Dict[str, Any],
    intent: Dict[str, Any],
    gsm_details: List[Dict[str, Any]],
    abstract: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply the SPEC to one GEO dataset and return the structured verdict.

    Args:
        llm:         LangChain chat model.
        ds:          GEO dataset metadata dict (accession, title, summary, ...).
        intent:      Parsed user intent dict.
        gsm_details: Representative GSM details from GEOClient.get_representative_gsm_details().
        abstract:    Optional PubMed abstract (extra evidence per SPEC "文章反向追踪").

    Returns:
        Verdict dict (see _OUTPUT_CONTRACT). On any LLM/parse error, returns a
        conservative manual_review verdict with the error in notes — never
        silently keeps or excludes (SPEC: cannot confirm → manual_review).
    """
    acc = ds.get("accession", "?")
    pmids = ds.get("pubmed_ids", [])

    user_msg = (
        f"=== USER REQUEST ===\n"
        f"{_intent_block(intent)}\n\n"
        f"=== GEO DATASET METADATA ===\n"
        f"GEO Accession: {acc}\n"
        f"Title: {ds.get('title', '')[:200]}\n"
        f"Summary: {ds.get('summary', '')[:600]}\n"
        f"Overall Design: {ds.get('overall_design', '')[:400]}\n"
        f"Platform: {ds.get('platform_canonical') or ds.get('platforms', [])}\n"
        f"Sample count (GEO): {ds.get('sample_count')}\n"
        f"Sample type (GEO): {ds.get('sample_type')}\n"
        f"Cancer type (GEO): {ds.get('cancer_type')}\n"
        f"PubMed IDs: {pmids}\n\n"
        f"=== REPRESENTATIVE GSM SAMPLES ===\n"
        f"{_gsm_block(gsm_details)}\n"
    )
    if abstract:
        user_msg += f"\n=== PUBMED ABSTRACT (PMID {pmids[0] if pmids else '?'}) ===\n{abstract[:2500]}\n"

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        verdict = _safe_json(raw)

        # Capture token usage + API-returned model (for the per-query CSV log).
        usage = _extract_usage(response)
        if usage.get("cached_tokens"):
            logger.debug(f"geo_filter {acc}: cached_tokens={usage['cached_tokens']}")
        verdict = _normalise_verdict(verdict, gsm_details)
        verdict["_usage"] = usage
        # Evidence snapshot (logged so a human can see WHAT the model was given
        # alongside HOW it reasoned — e.g. all-unknown GSM groups + no abstract
        # = weak evidence that often explains a bad verdict).
        verdict["_evidence"] = {
            "gsm_groups": group_summary(gsm_details),
            "n_representative_gsm": len(gsm_details),
            "had_abstract": bool(abstract),
            "geo_sample_count": ds.get("sample_count"),
        }
        logger.info(
            f"geo_filter {acc}: action={verdict.get('recommended_action')} "
            f"usable={verdict.get('usable')} sample={verdict.get('confirmed_sample_type')} "
            f"reason={(verdict.get('reason') or '')[:80]}"
        )
        return verdict

    except Exception as e:
        logger.warning(f"geo_filter {acc}: LLM/parse error — {e} — manual_review (conservative)")
        return {
            "usable": "unclear",
            "recommended_action": "manual_review",
            "confirmed_sample_type": ds.get("sample_type", "unknown"),
            "confirmed_cancer_type": None,
            "sample_count_in_paper": None,
            "stage_treatment": None,
            "consistency": "unknown",
            "sample_level_annotation": "unclear",
            "available_file_type": None,
            "disease_groups": None,
            "reason": f"filter_error: {e}",
            "notes": f"filter_error: {e}",
            "reasoning": f"filter_error: {e}",
            "gsm_includes": [],
        }


def _extract_usage(response: Any) -> Dict[str, Any]:
    """
    Pull token-usage + the API-returned model name from an LLM response.

    Handles both LangChain's standardized `usage_metadata` (input_tokens /
    output_tokens / total_tokens / input_token_details.cache_read) and the raw
    OpenAI-style `response_metadata` (prompt_tokens / completion_tokens /
    prompt_tokens_details.cached_tokens). Returns 0s if unavailable.
    """
    um = getattr(response, "usage_metadata", None) or {}
    if not isinstance(um, dict):
        um = {}
    rm = getattr(response, "response_metadata", None) or {}
    if not isinstance(rm, dict):
        rm = {}

    itd = um.get("input_token_details") or {}
    cached = (
        itd.get("cache_read")
        or itd.get("cached_tokens")
        or um.get("cached_tokens")
        or rm.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        or 0
    )
    prompt = um.get("input_tokens") or um.get("prompt_tokens") or rm.get("prompt_tokens") or 0
    completion = um.get("output_tokens") or um.get("completion_tokens") or rm.get("completion_tokens") or 0
    total = um.get("total_tokens") or rm.get("total_tokens") or (prompt + completion) or 0
    api_model = rm.get("model_name") or rm.get("model") or ""

    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "cached_tokens": int(cached),
        "api_model": str(api_model),
    }


# Enum normalisation so downstream code never sees an unexpected value.
_ACTION_VALUES = {"keep", "exclude", "article_only", "manual_review"}
_USABLE_VALUES = {"yes", "no", "partial", "unclear"}


def _normalise_verdict(verdict: Dict[str, Any], gsm_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    action = verdict.get("recommended_action")
    if action not in _ACTION_VALUES:
        verdict["recommended_action"] = "manual_review"

    usable = verdict.get("usable")
    if usable not in _USABLE_VALUES:
        verdict["usable"] = "unclear"

    # Ensure the reasoning chain is a non-empty string (default "" so the CSV
    # column always exists; an empty value signals the model skipped it).
    if not isinstance(verdict.get("reasoning"), str):
        verdict["reasoning"] = ""

    # Ensure gsm_includes is a list of well-formed dicts covering the representatives.
    raw_inc = verdict.get("gsm_includes") or []
    norm: List[Dict[str, Any]] = []
    for item in raw_inc:
        if isinstance(item, dict):
            norm.append({
                "gsm": str(item.get("gsm", "")),
                "include": bool(item.get("include", True)),
                "reason": None if item.get("include") else (item.get("reason") or "excluded"),
            })
    verdict["gsm_includes"] = norm
    return verdict


# ---------------------------------------------------------------------- #
#  Verdict → dataset dict (for registry upsert)                          #
# ---------------------------------------------------------------------- #

def apply_verdict(ds: Dict[str, Any], verdict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of `ds` with verdict fields mapped onto registry columns.
    The output is ready for Registry.upsert_dataset(...) (interface unchanged).
    """
    action = verdict.get("recommended_action", "manual_review")
    # usable column is INTEGER 0/1: yes/partial → 1 (benefit of the doubt for unclear).
    usable_map = {"yes": 1, "partial": 1, "no": 0, "unclear": 1}
    usable_int = usable_map.get(verdict.get("usable", "unclear"), 1)

    updated = dict(ds)
    updated["recommended_action"] = action
    updated["usable"] = usable_int
    updated["reason"] = verdict.get("reason", "")
    updated["consistency"] = verdict.get("consistency", "unknown")

    if verdict.get("confirmed_sample_type") and verdict["confirmed_sample_type"] != "unknown":
        updated["sample_type"] = verdict["confirmed_sample_type"]
    if verdict.get("confirmed_cancer_type"):
        updated["cancer_type"] = verdict["confirmed_cancer_type"]
    if verdict.get("stage_treatment"):
        updated["stage_treatment"] = verdict["stage_treatment"]
    if verdict.get("available_file_type"):
        updated["available_file_type"] = verdict["available_file_type"]
    if verdict.get("sample_level_annotation"):
        updated["sample_level_annotation"] = verdict["sample_level_annotation"]
    if verdict.get("disease_groups"):
        updated["disease_groups"] = verdict["disease_groups"]

    # Correct sample count if the paper states a materially different n.
    paper_n = verdict.get("sample_count_in_paper")
    geo_n = ds.get("sample_count")
    if paper_n and isinstance(paper_n, int):
        if not geo_n or (geo_n and abs(paper_n - geo_n) / geo_n > 0.20):
            existing = updated.get("notes") or ""
            note = f"sample_count GEO={geo_n} paper={paper_n}"
            updated["notes"] = (existing + "; " + note).lstrip("; ")
            updated["sample_count"] = paper_n

    # Append notes (never overwrite existing).
    if verdict.get("notes"):
        existing = updated.get("notes") or ""
        updated["notes"] = (existing + "; " + verdict["notes"]).lstrip("; ")

    updated["_verdict"] = verdict  # full verdict retained for reporting/debug
    return updated


# ---------------------------------------------------------------------- #
#  Skill wrapper (for the dynamic orchestrator — Phase 3)                #
# ---------------------------------------------------------------------- #

class _FilterArgs(BaseModel):
    accession: str = Field(..., description="GSE accession to filter")
    fetch_abstract: bool = Field(
        True, description="Attempt to fetch the linked PubMed abstract as extra evidence."
    )


class GeoFilterSkill(Skill):
    name = "geo_filter"
    description = (
        "Apply the GEO filtering SPEC to one dataset. Given a GSE accession (already "
        "discovered via search), fetch representative GSM details (+ optional PubMed "
        "abstract) and return a keep/exclude/article_only/manual_review verdict with "
        "reasoning. Use this to decide whether a GEO dataset is usable for the user's "
        "cancer-early-detection methylation request."
    )
    args_schema = _FilterArgs

    def run(self, ctx: SkillContext, accession: str, fetch_abstract: bool = True) -> Dict[str, Any]:
        geo = ctx.geo_client
        if geo is None:
            raise RuntimeError("geo_filter requires ctx.geo_client")

        ds = geo.get_series_metadata(accession)
        gsm_details = geo.get_representative_gsm_details(
            accession, wanted_sample_type=(ctx.state.get("parsed_intent", {}) or {}).get("sample_type", "")
        )

        abstract = None
        if fetch_abstract and ds.get("pubmed_ids"):
            abstract = geo.fetch_pubmed_abstract(str(ds["pubmed_ids"][0]))

        verdict = filter_dataset(ctx.llm, ds, ctx.state.get("parsed_intent", {}), gsm_details, abstract)
        return apply_verdict(ds, verdict)


register_skill(GeoFilterSkill())
