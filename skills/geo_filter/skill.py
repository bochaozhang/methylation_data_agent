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
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
  "outcome": "download" | "exclude" | "manual_review",
  "confirmed_sample_type": "plasma|tumor|adjacent|normal|wbc|cfdna|serum|whole_blood|cell_line|other|unknown",
  "confirmed_cancer_type": "canonical English cancer name, or null",
  "technology": "450K|850K/EPIC|RRBS|WGBS|MCTA|MeDIP|panel|qMSP|null",
  "platform": "GPL id(s) or null",
  "sample_size": "case N, control M, ... or null",
  "stage_or_treatment_status": "staging / treatment status, or null",
  "disease_groups": "case/control/precursor breakdown, or null",
  "sample_level_annotation": "yes|no|unclear",
  "annotation_source": "GSM_characteristics|sample_title|paper_table|mixed|unclear|null",
  "download_tier": 1 | 2 | 3,
  "exclude_reason": "cell_line|animal_model|non_target_unsplittable|no_reference_value|non_methylation|locked|article_only|null",
  "flags": "case_only|pooled|cross_platform|tissue_only|no_control|pan_cancer_needs_split|...|empty string",
  "sample_count_in_paper": <integer or null>,
  "consistency": "consistent|minor_discrepancy|major_discrepancy|unknown",
  "reason": "one sentence: what the samples are and why this outcome",
  "notes": "caveats (sample count mismatch, pooled cfDNA, ...); empty string if none",
  "gsm_includes": [
    {"gsm": "<gsm id>", "include": true|false, "reason": null | "one sentence if excluded"}
  ]
}

Field guidance:
- reasoning (REQUIRED, fill FIRST): step-by-step chain, in order:
    1. What biological samples this dataset actually contains (GSM details + summary + abstract).
    2. Human / target cancer type?
    3. Sample type vs request — plasma / serum ARE cell-free DNA (cfDNA).
    4. Are there non-cancer / control samples?
    5. Download tier from metadata (Tier 1/2/3 — see download_tier; do NOT open files).
    6. → outcome.
  Must be internally consistent: if the chain shows the requested samples are present
  and GEO provides downloadable data, outcome MUST be download.
- outcome (three states):
    download      → matches the request. NO file-format requirement — whether the
                    downloaded data is usable is judged AFTER download by the
                    geo-download skill, not here.
    exclude       → cell line / organoid / animal / in-vitro / treated / metastasis-only /
                    non-target-unsplittable / non-methylation / locked (data locked or
                    application required — nothing obtainable now) / article_only (data
                    exists only in the paper, not deposited in GEO).
    manual_review → ambiguous, GEO-vs-article metadata conflict, or cannot confirm
                    sample type / controls.
- download_tier (which download fallback tier the GEO metadata implies):
    1 → series_matrix has data (GEO-compiled value matrix).
    2 → series_matrix empty, but supplementary files other than RAW bundles exist.
    3 → series_matrix empty and no non-RAW supplementary files (per-GSM scraping).
- exclude_reason: fill when outcome is exclude (else null). locked = data locked /
  application required; article_only = only in the paper, not deposited in GEO.
- flags: free-text caveats (case_only, pooled, cross_platform, tissue_only, no_control,
  pan_cancer_needs_split, ...); empty string if none.
- gsm_includes: classify ONLY the representative GSM samples you were given.
  include=true → reason MUST be null; include=false → reason MUST be a short explanation.
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


# ---------------------------------------------------------------------- #
#  Characteristics noise filter (used by _dedup_gsm_combos)              #
# ---------------------------------------------------------------------- #
#
# GEO !Sample_characteristics columns are free-text and mix biological
# labels with per-sample noise. If every column participates in the dedup key,
# any per-patient field (patient id / age / batch / sex / date) defeats collapse
# — each GSM becomes its own combo and the LLM sees N near-duplicate rows.
#
# Policy: drop columns whose name is unambiguously NON-biological, keep
# everything else (disease state / tissue / sample type / treatment / stage /
# grade / condition / source / ...). This is a CONSERVATIVE blocklist: it can
# only fail to collapse (never wrongly merge distinct sample types), because a
# column is dropped only when its name matches a known-noise token.
#
# This single constant is the one place to tune the noise policy. Matched
# case-insensitively on whole tokens of the column name, e.g.
#   "patient id" / "Age at diagnosis" / "Batch number" / "Sex"  → dropped
#   "disease state" / "tissue" / "treatment" / "tumor stage"    → kept
_NOISE_CHAR_TOKENS = frozenset({
    # submitter / per-patient identifiers
    "patient", "subject", "donor", "individual", "id", "code",
    # demographics
    "age", "sex", "gender", "race", "ethnicity", "ethnic", "birth",
    # batch / wet-lab technical
    "batch", "plate", "well", "lane", "flowcell", "flowcells",
    "barcode", "index", "indices", "adapter", "primer",
    # sequencing / library prep
    "library", "sequencer", "instrument", "machine", "reads", "insert",
    "concentration", "rin",
    # dates / time
    "date", "time", "day", "month", "year", "timestamp",
    # cell culture
    "passage",
})

_CHAR_KEY_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _is_noise_characteristic_key(key: str) -> bool:
    """True if a characteristics column name is non-biological noise (per the
    token blocklist above). Whole-token match on the lower-cased name."""
    if not key:
        return False
    tokens = _CHAR_KEY_TOKEN_RE.split(str(key).strip().lower())
    return any(tok and tok in _NOISE_CHAR_TOKENS for tok in tokens)


def _biological_characteristics(ch: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `ch` with noise columns (patient id / age / batch / sex /
    dates / sequencing-technical) removed, so dedup collapses biologically
    identical samples. Conservative — only matches the noise blocklist."""
    return {k: v for k, v in (ch or {}).items()
            if not _is_noise_characteristic_key(k)}


def _dedup_gsm_combos(gsm_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate GSM samples by (source_name, molecule, group, characteristics).

    ``characteristics`` is first passed through ``_biological_characteristics``
    so per-sample noise columns (patient id / age / batch / ...) don't split
    biologically identical samples into separate combos. Affects BOTH the dedup
    key and the characteristics shown to the LLM / used for verdict expansion.

    Returns a list of unique combos, each with a 'count' field showing how many
    GSMs share this combination. The 'gsm' field shows a representative GSM id.
    """
    seen: Dict[tuple, Dict[str, Any]] = {}
    for g in gsm_details:
        ch = _biological_characteristics(g.get("characteristics") or {})
        ch_key = tuple(sorted(ch.items()))
        key = (g.get("source_name", ""), g.get("molecule", ""), g.get("group", "unknown"), ch_key)
        if key not in seen:
            seen[key] = {
                "gsm": g.get("gsm", ""),
                "source_name": g.get("source_name", ""),
                "molecule": g.get("molecule", ""),
                "group": g.get("group", "unknown"),
                "characteristics": ch,
                "count": 1,
                "gsm_ids": [g.get("gsm", "")],
            }
        else:
            seen[key]["count"] += 1
            seen[key]["gsm_ids"].append(g.get("gsm", ""))
    return list(seen.values())


def _expand_gsm_includes(verdict: Dict[str, Any], gsm_details: List[Dict[str, Any]]) -> None:
    """
    Expand per-combo gsm_includes verdicts back to per-GSM.

    The LLM returns gsm_includes for the deduplicated combos. This function
    expands each combo's include/exclude verdict to ALL GSMs sharing that combo.
    Mutates verdict["gsm_includes"] in place.
    """
    raw_includes = verdict.get("gsm_includes") or []
    if not raw_includes:
        return

    # Build combo verdict map: combo_key → include (bool)
    combo_verdicts: Dict[str, bool] = {}
    for item in raw_includes:
        gsm_id = str(item.get("gsm", ""))
        combo_verdicts[gsm_id] = bool(item.get("include", True))

    # Expand: for each GSM, find its combo representative and inherit the verdict
    combos = _dedup_gsm_combos(gsm_details)
    expanded = []
    for g in gsm_details:
        gsm_id = str(g.get("gsm", ""))
        # Find which combo this GSM belongs to
        for combo in combos:
            if gsm_id in combo.get("gsm_ids", []):
                rep_id = str(combo.get("gsm", ""))
                include = combo_verdicts.get(rep_id, combo_verdicts.get(gsm_id, False))
                expanded.append({
                    "gsm": gsm_id,
                    "include": include,
                    "reason": None if include else f"belongs to combo {rep_id} (count={combo['count']})",
                })
                break
        else:
            expanded.append({"gsm": gsm_id, "include": False, "reason": "no combo match"})

    verdict["gsm_includes"] = expanded


def _gsm_block(gsm_details: List[Dict[str, Any]]) -> str:
    """Render deduplicated sample combos + counts as evidence."""
    if not gsm_details:
        return "(no sample details available)"

    combos = _dedup_gsm_combos(gsm_details)
    n_total = len(gsm_details)
    counts = group_summary(gsm_details)
    counts_line = ", ".join(f"{g}={n}" for g, n in counts.items() if n)

    lines = [f"Sample details ({n_total} total in {len(combos)} unique combos, "
            f"group counts: {counts_line}):"]
    for combo in combos:
        ch = combo.get("characteristics") or {}
        ch_str = "; ".join(f"{k}: {v}" for k, v in ch.items()) if ch else "(none)"
        lines.append(
            f"  - GSM {combo.get('gsm', '?')} [×{combo['count']}, group={combo.get('group', '?')}]: "
            f"source_name={combo.get('source_name', '')!r}, "
            f"molecule={combo.get('molecule', '')!r}, characteristics={{{ch_str}}}"
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


# ---------------------------------------------------------------------- #
#  JSON-parse tier counters (feeds the §3 model-stability benchmark)     #
# ---------------------------------------------------------------------- #
# How much recovery _safe_json needed. "clean" = bare JSON parsed first
# try — what json_mode makes the norm. Tracked per-process (thread-safe).
_JSON_PARSE_STATS: Dict[str, int] = {"clean": 0, "fenced": 0, "extracted": 0, "failed": 0}
_JSON_PARSE_LOCK = threading.Lock()


def get_json_parse_stats() -> Dict[str, int]:
    """Snapshot of cumulative geo_filter JSON-parse outcomes for this process.

    clean     = bare JSON, parsed first-try (json_mode target → ~100%).
    fenced    = needed ```-fence stripping (the old glm default).
    extracted = needed the first balanced {...} block (most degraded).
    failed    = unparseable → caller falls back to manual_review.
    """
    with _JSON_PARSE_LOCK:
        return dict(_JSON_PARSE_STATS)


def reset_json_parse_stats() -> None:
    """Zero the counters (e.g. at the start of a benchmark run)."""
    with _JSON_PARSE_LOCK:
        for k in _JSON_PARSE_STATS:
            _JSON_PARSE_STATS[k] = 0


def _bump_json_stat(tier: str) -> None:
    with _JSON_PARSE_LOCK:
        _JSON_PARSE_STATS[tier] = _JSON_PARSE_STATS.get(tier, 0) + 1


def _safe_json(raw: str) -> Tuple[Dict[str, Any], str]:
    """Tolerant JSON parse → (parsed_dict, tier).

    Tries bare JSON first (tier "clean"), then ```-fence stripping ("fenced"),
    then the first balanced {...} block ("extracted"); bumps exactly one counter
    per call. Raises json.JSONDecodeError (tier "failed") only when all tiers
    miss — the caller (filter_dataset) catches that → conservative manual_review.
    """
    stripped = raw.strip()
    # Tier clean: bare JSON, no fence stripping.
    try:
        out = json.loads(stripped)
        _bump_json_stat("clean")
        return out, "clean"
    except json.JSONDecodeError:
        pass
    # Tier fenced: strip markdown fences, retry.
    unfenced = _strip_fences(raw)
    if unfenced != stripped:
        try:
            out = json.loads(unfenced)
            _bump_json_stat("fenced")
            return out, "fenced"
        except json.JSONDecodeError:
            pass
    # Tier extracted: first balanced {...} block.
    start = unfenced.find("{")
    end = unfenced.rfind("}")
    if start != -1 and end > start:
        try:
            out = json.loads(unfenced[start:end + 1])
            _bump_json_stat("extracted")
            return out, "extracted"
        except json.JSONDecodeError:
            pass
    _bump_json_stat("failed")
    raise json.JSONDecodeError("geo_filter: could not parse JSON", stripped, 0)


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
        verdict, json_tier = _safe_json(raw)
        verdict["_json_parse_tier"] = json_tier
        if json_tier != "clean":
            logger.info(
                f"geo_filter {acc}: JSON parse needed '{json_tier}' recovery "
                f"(json_mode off, or model wrapped/truncated the JSON)"
            )

        # Capture token usage + API-returned model (for the per-query CSV log).
        usage = _extract_usage(response)
        if usage.get("cached_tokens"):
            logger.debug(f"geo_filter {acc}: cached_tokens={usage['cached_tokens']}")
        verdict = _normalise_verdict(verdict, gsm_details)
        # Expand per-combo gsm_includes back to per-GSM (LLM saw deduped combos)
        _expand_gsm_includes(verdict, gsm_details)
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
            f"geo_filter {acc}: outcome={verdict.get('outcome')} "
            f"sample={verdict.get('confirmed_sample_type')} "
            f"reason={(verdict.get('reason') or '')[:80]}"
        )
        return verdict

    except Exception as e:
        logger.warning(f"geo_filter {acc}: LLM/parse error — {e} — manual_review (conservative)")
        return {
            "outcome": "manual_review",
            "recommended_action": "manual_review",
            "usable": "unclear",
            "confirmed_sample_type": ds.get("sample_type", "unknown"),
            "confirmed_cancer_type": None,
            "sample_count_in_paper": None,
            "stage_or_treatment_status": None,
            "consistency": "unknown",
            "sample_level_annotation": "unclear",
            "disease_groups": None,
            "download_tier": None,
            "flags": "",
            "exclude_reason": None,
            "reason": f"filter_error: {e}",
            "notes": f"filter_error: {e}",
            "reasoning": f"filter_error: {e}",
            "gsm_includes": [],
            "_json_parse_tier": "failed",
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


# Three-state outcome (relevance only; file-format usability is judged post-download
# by geo_download). Phase 1: metadata-level inference, no per-file verification.
_OUTCOME_VALUES = {"download", "exclude", "manual_review"}

# outcome → (recommended_action stored in registry, usable label).
# recommended_action mirrors the true 3-state outcome so the Web UI "Outcome"
# column shows download/exclude/manual_review (not a collapsed legacy value).
_OUTCOME_TO_LEGACY: Dict[str, tuple] = {
    "download": ("download", "yes"),
    "exclude": ("exclude", "no"),
    "manual_review": ("manual_review", "unclear"),
}


def _normalise_verdict(verdict: Dict[str, Any], gsm_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    outcome = verdict.get("outcome")
    # Defensive: the retired "lead" outcome (old prompts/caches) maps to download —
    # format usability is judged after download anyway.
    if outcome == "lead":
        outcome = "download"
        verdict["reason"] = (
            (verdict.get("reason") or "")
            + "; legacy lead → download (format judged post-download)"
        ).strip("; ")
    if outcome not in _OUTCOME_VALUES:
        outcome = "manual_review"
    verdict["outcome"] = outcome

    # Derive legacy recommended_action + usable so old consumers keep working.
    rec_action, usable = _OUTCOME_TO_LEGACY.get(outcome, ("manual_review", "unclear"))
    verdict["recommended_action"] = rec_action
    verdict["usable"] = usable

    # Reasoning chain (non-empty string; "" signals the model skipped it).
    if not isinstance(verdict.get("reasoning"), str):
        verdict["reasoning"] = ""

    # download_tier — integer 1|2|3 (which fallback tier GEO metadata implies).
    # Accept loose forms (float/int/str); anything unparseable → None (download side
    # re-derives the tier from live GEO queries anyway).
    tier = verdict.get("download_tier")
    try:
        tier = int(tier)
        if tier not in (1, 2, 3):
            tier = None
    except (TypeError, ValueError):
        tier = None
    verdict["download_tier"] = tier

    # files[] — legacy per-file shape from old prompts; no longer part of the
    # contract (replaced by download_tier). Dropped here so downstream never sees
    # stale metadata-inferred file verdicts.
    verdict.pop("files", None)
    verdict.pop("lead_type", None)

    if not isinstance(verdict.get("flags"), str):
        verdict["flags"] = ""
    verdict.setdefault("exclude_reason", None)

    # gsm_includes — list of well-formed dicts covering the representatives.
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
    Return a copy of `ds` with verdict fields mapped onto registry columns +
    the three-state record fields attached for the pipeline. Ready for
    Registry.upsert_dataset(...) (interface unchanged) and for split_by_outcome().
    """
    action = verdict.get("recommended_action", "manual_review")
    # usable column is INTEGER 0/1: yes → 1 (benefit of the doubt for unclear).
    usable_map = {"yes": 1, "no": 0, "unclear": 1}
    usable_int = usable_map.get(verdict.get("usable", "unclear"), 1)

    updated = dict(ds)
    updated["outcome"] = verdict.get("outcome", "manual_review")
    updated["recommended_action"] = action  # legacy compat (derived from outcome)
    updated["usable"] = usable_int
    updated["reason"] = verdict.get("reason", "")
    updated["consistency"] = verdict.get("consistency", "unknown")

    if verdict.get("confirmed_sample_type") and verdict["confirmed_sample_type"] != "unknown":
        updated["sample_type"] = verdict["confirmed_sample_type"]
    if verdict.get("confirmed_cancer_type"):
        updated["cancer_type"] = verdict["confirmed_cancer_type"]
    # accept both the new and the legacy key name
    stage = verdict.get("stage_or_treatment_status") or verdict.get("stage_treatment")
    if stage:
        updated["stage_treatment"] = stage

    # Download tier implied by GEO metadata (1|2|3; the download skill re-derives
    # the actual tier from live queries — this is the metadata-level expectation).
    if verdict.get("download_tier"):
        updated["download_tier"] = verdict["download_tier"]

    if verdict.get("technology"):
        updated["technology"] = verdict["technology"]
    if verdict.get("platform"):
        updated["platform_filter_hint"] = verdict["platform"]
    if verdict.get("sample_level_annotation"):
        updated["sample_level_annotation"] = verdict["sample_level_annotation"]
    if verdict.get("disease_groups"):
        updated["disease_groups"] = verdict["disease_groups"]

    # Attach the full three-state record fields (used by split_by_outcome / pipeline).
    updated["flags"] = verdict.get("flags", "")
    if verdict.get("exclude_reason"):
        updated["exclude_reason"] = verdict["exclude_reason"]

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


def split_by_outcome(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split enriched dataset records (from apply_verdict) into the three-state lists
    consumed by the pipeline: download_list / exclude_list / manual_review_list.
    Each record carries its full three-state fields (outcome, download_tier, flags,
    exclude_reason, reason, ...).
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "download_list": [],
        "exclude_list": [],
        "manual_review_list": [],
    }
    _key = {"download": "download_list",
            "exclude": "exclude_list", "manual_review": "manual_review_list"}
    for r in records:
        outcome = r.get("outcome", "manual_review")
        buckets[_key.get(outcome, "manual_review_list")].append(r)
    return buckets


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
