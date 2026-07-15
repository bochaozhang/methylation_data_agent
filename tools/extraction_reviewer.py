"""
LLM Reviewer Agent for MethyAgent — second-pass consistency check on the
output of extract_paper_structured() (tools/query_clarifier.py).

Why a second LLM call instead of extending the existing regex validators
(_validate_auc_sample_type_cooccurrence, _exclude_reference_datasets in
tools/query_clarifier.py):

  - The regex same-sentence check for AUC/sample_type co-occurrence is
    brittle: a correctly-linked AUC value that happens to be described one
    sentence away from its sample-type mention gets nulled as a false
    positive, and a wrong value that happens to share a sentence with the
    right sample-type keyword slips through as a false negative.
  - The reference-dataset keyword list ("reference", "background",
    "normalization", ...) only catches datasets whose exclusion is signaled
    by one of those literal words nearby; it has no actual understanding of
    what the dataset was used for.

An LLM given the full abstract and the draft extraction can reason about
both questions the way a human reviewer would, at the cost of one extra LLM
call per paper. This directly addresses Bug 2 (tissue/cfDNA AUC confusion,
PMID 40860669) and Bug 3 (reference-dataset misattribution, same paper), and
functions as the reflection/verification step ZhipuAI's 2026-06-30
consultation flagged as missing (no multi-turn reasoning).

This module does NOT replace the existing regex validators — both still run
inside extract_paper_structured(). review_extraction() runs after them, as
an independent second opinion; it can restore/correct fields the regex
pass got wrong in either direction. review_extraction() also runs its own
regex backstop (_check_auc_unsupported) alongside the LLM check, since
unlike the two query_clarifier.py validators it previously had no
non-LLM fallback.

review_geo_verdict() is a second reviewer entry point in this same module:
an independent second opinion on evaluate_geo_dataset()'s verdict (species /
methylation-data-type / tissue-vs-cfDNA), re-checked against the raw GEO
metadata rather than an abstract.

Public API:
    review_extraction(abstract, draft_extraction, llm) -> dict
        (returns a corrected draft_extraction record; see docstring)
    review_geo_verdict(verdict, geo_metadata, llm) -> dict
        (returns {risk_level, flags, corrected_fields, needs_human_review, reason})
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from tools.query_clarifier import _auc_value_candidates
from utils.logger import get_logger

logger = get_logger(__name__)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


_REVIEWER_SYSTEM = """You are a biomedical literature review assistant. Another model has \
already extracted structured fields from a paper abstract; your job is to independently \
verify two specific things by re-reading the abstract, and to correct the draft if it is wrong.

CHECK A — AUC / sample_type consistency:
For each non-null value in performance_metrics (auc_training, auc_validation, auc_external),
find where that number is reported in the abstract and determine which sample type it was
actually measured on (tissue/tumor vs plasma cfDNA vs serum cfDNA vs WBC/whole blood, etc).
If that sample type does NOT match the draft's "sample_type" field, the value is wrong for
this record — set it to null in your output. If it matches, keep it. If you cannot locate the
value in the abstract at all, set it to null (do not guess).

CHECK B — dataset_id attribution:
For each accession in dataset_ids, decide whether the abstract describes it as the study's
own primary/analysis data (including cohorts explicitly used to compute a reported
performance metric, e.g. an external validation cohort), OR whether it is cited only as a
reference/background/normalization/noise-filtering panel that the authors used to pre-process
or filter their own data but did not directly analyze as a study cohort. Drop any accession
of the second kind from your output dataset_ids list.

Output ONLY valid JSON, no markdown fences, matching this shape exactly:
{
  "performance_metrics": {
    "auc_training": null,
    "auc_validation": null,
    "auc_external": null
  },
  "dataset_ids": ["<accessions you determined are primary/analysis data>"],
  "needs_human_review": false,
  "reason": "one or two sentences: what you checked, what (if anything) you changed and why"
}

Include every key from performance_metrics that was present in the draft, even if you are
keeping its value unchanged. Set needs_human_review=true if you changed anything, or if you
are not confident about a value even after your check."""


def _build_review_payload(draft_extraction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_type": draft_extraction.get("sample_type"),
        "performance_metrics": draft_extraction.get("performance_metrics") or {},
        "dataset_ids": draft_extraction.get("dataset_ids") or [],
    }


# Bug 1 regex backstop: a reported AUC value should actually be labeled as
# such somewhere near its occurrence in the source text, not just a bare
# number the extractor mistook for one (e.g. a sample size or p-value).
_AUC_LABEL_RE = re.compile(r"auc|auroc|\broc\b|c-statistic|c-index", re.IGNORECASE)
_AUC_UNSUPPORTED_WINDOW = 150


def _check_auc_unsupported(source_text: str, metrics: Dict[str, Any]) -> List[str]:
    """
    For each non-null AUC value in metrics, verify it's labeled AUC/AUROC/ROC/
    C-statistic/C-index within ~150 chars of its occurrence in source_text.
    Returns the metric keys that failed this check (caller nulls them).
    """
    if not source_text or not isinstance(metrics, dict):
        return []
    unsupported: List[str] = []
    for key in ("auc_training", "auc_validation", "auc_external"):
        value = metrics.get(key)
        if value is None:
            continue
        candidates = _auc_value_candidates(value)
        if not candidates:
            continue
        idx = -1
        for candidate in candidates:
            idx = source_text.find(candidate)
            if idx != -1:
                break
        if idx == -1:
            unsupported.append(key)  # can't even locate the value in the text
            continue
        window = source_text[max(0, idx - _AUC_UNSUPPORTED_WINDOW): idx + _AUC_UNSUPPORTED_WINDOW]
        if not _AUC_LABEL_RE.search(window):
            unsupported.append(key)
    return unsupported


def review_extraction(
    abstract: str,
    draft_extraction: Dict[str, Any],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Independently re-check a draft extraction's AUC/sample_type consistency
    (Bug 2) and dataset_id attribution (Bug 3) against the source abstract.

    Args:
        abstract:         The paper abstract text used to produce draft_extraction.
        draft_extraction: Output dict from extract_paper_structured().
        llm:              LangChain chat model.

    Returns:
        A copy of draft_extraction with performance_metrics/dataset_ids
        corrected where the reviewer found a mismatch, needs_human_review
        set when anything changed, and a "review_report" key holding
        {risk_level, flags, corrected_fields, needs_human_review, reason}
        for traceability. On reviewer failure (bad JSON, no metrics/
        dataset_ids to check), returns draft_extraction unchanged except for
        a needs_human_review flag and a review_report noting the failure.
    """
    result = dict(draft_extraction)

    metrics = result.get("performance_metrics")
    dataset_ids = result.get("dataset_ids")
    has_metrics_to_check = isinstance(metrics, dict) and any(
        metrics.get(k) is not None for k in ("auc_training", "auc_validation", "auc_external")
    )
    has_datasets_to_check = isinstance(dataset_ids, list) and len(dataset_ids) > 0

    if not abstract or not (has_metrics_to_check or has_datasets_to_check):
        result["review_report"] = {
            "risk_level": "low", "flags": [], "corrected_fields": {},
            "needs_human_review": False, "reason": "nothing to check",
        }
        return result  # nothing for the reviewer to usefully check

    payload = _build_review_payload(result)
    context = (
        f"Abstract:\n{abstract}\n\n"
        f"Draft extraction (fields under review only):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(content=_REVIEWER_SYSTEM),
        HumanMessage(content=context),
    ]

    try:
        response = llm.invoke(messages)
        verdict = json.loads(_strip_json_fences(response.content))
    except json.JSONDecodeError:
        result["needs_human_review"] = True
        note = "Reviewer response was not valid JSON; corrections skipped."
        result["reason"] = f"{result.get('reason') or ''} {note}".strip()
        result["review_report"] = {
            "risk_level": "medium", "flags": ["reviewer_error"], "corrected_fields": {},
            "needs_human_review": True, "reason": note,
        }
        logger.warning(f"extraction_reviewer: unparseable response (PMID {result.get('pmid', '')})")
        return result
    except Exception as e:
        result["needs_human_review"] = True
        note = f"Reviewer call failed ({e}); corrections skipped."
        result["reason"] = f"{result.get('reason') or ''} {note}".strip()
        result["review_report"] = {
            "risk_level": "medium", "flags": ["reviewer_error"], "corrected_fields": {},
            "needs_human_review": True, "reason": note,
        }
        logger.warning(f"extraction_reviewer: call failed for PMID {result.get('pmid', '')}: {e}")
        return result

    changes: List[str] = []
    flags: List[str] = []
    corrected_fields: Dict[str, Any] = {}

    # CHECK A (LLM): AUC / sample_type consistency
    reviewed_metrics = verdict.get("performance_metrics")
    if isinstance(reviewed_metrics, dict) and isinstance(result.get("performance_metrics"), dict):
        orig_metrics = result["performance_metrics"]
        metrics_changed = False
        for key in ("auc_training", "auc_validation", "auc_external"):
            if key not in reviewed_metrics:
                continue
            new_val = reviewed_metrics[key]
            if new_val != orig_metrics.get(key):
                changes.append(f"{key}: {orig_metrics.get(key)!r} -> {new_val!r}")
                orig_metrics[key] = new_val
                metrics_changed = True
        if metrics_changed:
            flags.append("sample_type_mismatch")
            corrected_fields["performance_metrics"] = dict(orig_metrics)

    # CHECK B (LLM): dataset_id attribution
    reviewed_dataset_ids = verdict.get("dataset_ids")
    if isinstance(reviewed_dataset_ids, list):
        orig_ids = result.get("dataset_ids") or []
        removed = [d for d in orig_ids if d not in reviewed_dataset_ids]
        if removed:
            result["dataset_ids"] = reviewed_dataset_ids or None
            existing_excluded = result.get("excluded_reference_datasets") or []
            result["excluded_reference_datasets"] = existing_excluded + [
                d for d in removed if d not in existing_excluded
            ]
            changes.append(f"dataset_ids: removed reference-only {removed}")
            flags.append("reference_accession")
            corrected_fields["dataset_ids"] = result["dataset_ids"]

    # Bug 1 regex backstop — runs regardless of what the LLM said, on
    # whatever metrics remain non-null after CHECK A's correction.
    auc_unsupported_keys = _check_auc_unsupported(abstract, result.get("performance_metrics") or {})
    if auc_unsupported_keys:
        om = result["performance_metrics"]
        for key in auc_unsupported_keys:
            if om.get(key) is not None:
                changes.append(f"{key}: {om.get(key)!r} -> None (auc_unsupported)")
                om[key] = None
        flags.append("auc_unsupported")
        corrected_fields["performance_metrics"] = dict(om)

    if changes:
        logger.info(f"extraction_reviewer corrected PMID {result.get('pmid', '')}: {'; '.join(changes)}")

    needs_human_review = bool(changes) or bool(verdict.get("needs_human_review"))
    if needs_human_review:
        result["needs_human_review"] = True
        reviewer_reason = verdict.get("reason") or "; ".join(changes)
        result["reason"] = f"{result.get('reason') or ''} [reviewer] {reviewer_reason}".strip()

    if "auc_unsupported" in flags or len(flags) >= 2:
        risk_level = "high"
    elif flags or verdict.get("needs_human_review"):
        risk_level = "medium"
    else:
        risk_level = "low"

    result["review_report"] = {
        "risk_level": risk_level,
        "flags": flags,
        "corrected_fields": corrected_fields,
        "needs_human_review": needs_human_review,
        "reason": verdict.get("reason") or ("; ".join(changes) if changes else "no issues found"),
    }
    return result


# ============================================================
# review_geo_verdict — second reviewer entry point: independently re-checks
# an evaluate_geo_dataset() verdict against the raw GEO metadata (species,
# is-this-actually-methylation-data, tissue-vs-cfDNA sample type).
# ============================================================

_GEO_REVIEWER_SYSTEM = """You are a GEO dataset triage auditor. Another model has already \
judged whether a GEO series is usable for a specific research query; your job is to \
independently re-check three things by re-reading the raw GEO metadata, and flag — do not \
assume the first judgment was correct.

CHECK 1 — species: is this dataset human data? If the metadata indicates a non-human organism
(mouse, rat, zebrafish, cell line of non-human origin, etc.), the verdict should have rejected
it; flag if it didn't.

CHECK 2 — data type is DNA methylation: does the metadata (title/summary/platform) actually
describe DNA methylation data (bisulfite sequencing, methylation array such as 450K/850K/EPIC,
WGBS, RRBS, etc)? If the metadata instead describes a different assay (RNA-seq, ChIP-seq,
ATAC-seq, genotyping, etc.) with no methylation content, flag it.

CHECK 3 — sample type: does the metadata's actual sample material (tissue/tumor/biopsy/FFPE vs
plasma/serum cfDNA vs whole blood/WBC/PBMC) match what the verdict's sample_type field claims,
and is it consistent with what the verdict's recommended_action implies was checked? If the
metadata clearly indicates tissue/FFPE samples but the verdict did not reject or flag on that
basis (e.g. it recommended "keep" for a cfDNA-specific query), flag it.

Output ONLY valid JSON, no markdown fences:
{
  "species_ok": true,
  "data_type_ok": true,
  "sample_type_ok": true,
  "corrected_recommended_action": null,
  "needs_human_review": false,
  "reason": "one or two sentences: what you checked and why you flagged (or didn't)"
}
Set corrected_recommended_action to "manual_review" if any check fails and you believe the
original recommended_action should change; otherwise leave it null. Set needs_human_review=true
if you disagree with any part of the original verdict, even if you're not certain."""

# Regex/keyword backstops — run regardless of LLM outcome, so a single
# egregious mismatch is still caught if the LLM call fails or agrees.
_NON_HUMAN_RE = re.compile(
    r"\b(mouse|mice|murine|mus musculus|rat|rattus|zebrafish|danio rerio|drosophila|"
    r"c\.\s?elegans|xenopus|porcine|bovine|canine)\b",
    re.IGNORECASE,
)
_METHYLATION_RE = re.compile(
    r"methylat|bisulfite|450k|850k|\bepic\b|wgbs|rrbs|infinium|\bcpg\b",
    re.IGNORECASE,
)
_TISSUE_INDICATOR_RE = re.compile(
    r"\b(tissue|tumou?r|biopsy|ffpe|resection|surgical specimen)\b",
    re.IGNORECASE,
)
_CFDNA_INDICATOR_RE = re.compile(
    r"\b(cfdna|cell-free dna|cell free dna|ctdna|circulating|plasma|serum)\b",
    re.IGNORECASE,
)


def _format_geo_metadata_for_review(geo_metadata: Dict[str, Any]) -> str:
    lines = [f"Accession: {geo_metadata.get('accession', '(unknown)')}"]
    if geo_metadata.get("title"):
        lines.append(f"Title: {geo_metadata['title']}")
    if geo_metadata.get("summary"):
        lines.append(f"Summary: {geo_metadata['summary']}")
    if geo_metadata.get("sample_titles"):
        lines.append("Sample titles: " + "; ".join(geo_metadata["sample_titles"]))
    platform = geo_metadata.get("platform_canonical") or geo_metadata.get("platforms")
    if platform:
        lines.append(f"Platform: {platform}")
    if geo_metadata.get("data_type"):
        lines.append(f"Data type: {geo_metadata['data_type']}")
    if geo_metadata.get("sample_count"):
        lines.append(f"Sample count: {geo_metadata['sample_count']}")
    return "\n".join(lines)


def _regex_backstop_geo(verdict: Dict[str, Any], geo_metadata: Dict[str, Any]) -> List[str]:
    """Keyword-based backstop — catches egregious mismatches even if the LLM
    call fails or agrees with a wrong verdict."""
    text = " ".join(str(geo_metadata.get(k) or "") for k in ("title", "summary"))
    text += " " + " ".join(geo_metadata.get("sample_titles") or [])
    flags: List[str] = []

    if _NON_HUMAN_RE.search(text):
        flags.append("species_mismatch")

    platform_and_type = str(
        geo_metadata.get("platform_canonical") or geo_metadata.get("platforms") or ""
    ) + " " + str(geo_metadata.get("data_type") or "")
    if text.strip() and not _METHYLATION_RE.search(text + " " + platform_and_type):
        flags.append("data_type_mismatch")

    verdict_sample_type = str(verdict.get("sample_type") or "").lower()
    verdict_rejected = verdict.get("recommended_action") in ("exclude", "manual_review")
    if (
        _TISSUE_INDICATOR_RE.search(text)
        and not _CFDNA_INDICATOR_RE.search(text)
        and "tissue" not in verdict_sample_type
        and not verdict_rejected
    ):
        flags.append("sample_type_mismatch")

    return flags


def review_geo_verdict(
    verdict: Dict[str, Any],
    geo_metadata: Dict[str, Any],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Independently re-check an evaluate_geo_dataset() verdict against the raw
    GEO metadata: species, whether the data is actually DNA methylation, and
    whether the sample type (tissue vs cfDNA/plasma/serum) matches what the
    verdict assumed.

    Args:
        verdict:      Output dict from evaluate_geo_dataset() (reads
                       sample_type, recommended_action, reason, usable).
        geo_metadata: Output dict from GEOClient.get_series_metadata().
        llm:          LangChain chat model.

    Returns:
        {"risk_level": "low"|"medium"|"high", "flags": [...],
         "corrected_fields": {...}, "needs_human_review": bool, "reason": str}
    """
    if not geo_metadata or geo_metadata.get("error"):
        return {
            "risk_level": "medium",
            "flags": ["metadata_unavailable"],
            "corrected_fields": {},
            "needs_human_review": True,
            "reason": "GEO metadata unavailable; cannot independently verify verdict.",
        }

    regex_flags = _regex_backstop_geo(verdict, geo_metadata)

    context = (
        "Original verdict:\n"
        + json.dumps(
            {k: verdict.get(k) for k in ("sample_type", "recommended_action", "reason", "usable")},
            ensure_ascii=False, indent=2,
        )
        + "\n\nRaw GEO metadata:\n"
        + _format_geo_metadata_for_review(geo_metadata)
    )
    messages = [
        SystemMessage(content=_GEO_REVIEWER_SYSTEM),
        HumanMessage(content=context),
    ]

    llm_flags: List[str] = []
    llm_needs_review = False
    llm_reason = ""
    corrected_action = None

    try:
        response = llm.invoke(messages)
        llm_verdict = json.loads(_strip_json_fences(response.content))
        if llm_verdict.get("species_ok") is False:
            llm_flags.append("species_mismatch")
        if llm_verdict.get("data_type_ok") is False:
            llm_flags.append("data_type_mismatch")
        if llm_verdict.get("sample_type_ok") is False:
            llm_flags.append("sample_type_mismatch")
        llm_needs_review = bool(llm_verdict.get("needs_human_review"))
        llm_reason = llm_verdict.get("reason") or ""
        corrected_action = llm_verdict.get("corrected_recommended_action")
    except Exception as e:
        logger.warning(
            f"review_geo_verdict: LLM call/parse failed for "
            f"{geo_metadata.get('accession', '')}: {e}"
        )
        llm_reason = f"Reviewer call failed ({e}); regex backstop only."

    flags = sorted(set(regex_flags) | set(llm_flags))
    needs_human_review = bool(flags) or llm_needs_review

    corrected_fields: Dict[str, Any] = {}
    if needs_human_review and verdict.get("recommended_action") not in ("exclude", "manual_review"):
        corrected_fields["recommended_action"] = corrected_action or "manual_review"

    if "species_mismatch" in flags or "sample_type_mismatch" in flags:
        risk_level = "high"
    elif flags or llm_needs_review:
        risk_level = "medium"
    else:
        risk_level = "low"

    reason = llm_reason or (
        f"Regex backstop flagged: {', '.join(regex_flags)}" if regex_flags else "no issues found"
    )

    return {
        "risk_level": risk_level,
        "flags": flags,
        "corrected_fields": corrected_fields,
        "needs_human_review": needs_human_review,
        "reason": reason,
    }


# ============================================================
# Quick test. Two modes:
#   python -m tools.extraction_reviewer        -> stubbed LLM, verifies
#       merge/parsing logic only (works with no API key / no network).
#   python -m tools.extraction_reviewer --live -> real get_llm(config["llm"])
#       from config/settings.yaml, verifies actual reviewer reasoning
#       quality. Requires a configured API key (.env) and network access.
# ============================================================

if __name__ == "__main__":
    # PMID 40860669 real abstract (Gao et al., PeerJ 2025) and the known-bad
    # draft extraction observed from the live pipeline (docs/6_30/NCBI_Test_out.rtf).
    _ABSTRACT = (
        "BACKGROUND: Preoperative identification of breast cancer (BC) subtypes is "
        "essential for optimizing treatment strategies and improving patient outcomes. "
        "This study aimed to identify circulating cell-free DNA (cfDNA) methylation "
        "signatures to differentiate triple-negative breast cancer (TNBC) from other BC "
        "subtypes (non-TNBC). METHODS: We initially performed a genome-wide analysis to "
        "identify differentially methylated CpG sites (DMCs; |Δβ| > 0.10 and P < 0.05) between "
        "five TNBC and nine non-TNBC tissues using the Infinium HumanMethylationEPIC "
        "BeadChip. These DMCs were further validated using large-scale data from the "
        "Cancer Genome Atlas (TCGA, n = 774; |Δβ| > 0.25 and P < 0.05), and only CpG "
        "sites with average β values > 0.90 or < 0.10 in white blood cells (GSE50132, n = "
        "233) were retained to minimize potential background methylation interference. "
        "Least absolute shrinkage and selection operator (LASSO) regression was applied "
        "to select optimal markers. Diagnostic performance was assessed by the area under "
        "the receiver operating characteristic curve (AUC), and prognostic value was "
        "evaluated using Cox regression analysis. A multiplex digital droplet PCR "
        "(mddPCR) assay was developed to simultaneously detect cg06268921 and cg23247845 "
        "in cfDNA from TNBC (n = 33) and non-TNBC (n = 80) patients. "
        "RESULTS: We identified 113 DMCs, of which eight were selected as optimal "
        "markers. They effectively discriminated TNBC from non-TNBC tissues. Then an "
        "eight-marker diagnostic panel was developed with an AUC of 0.922 in TCGA and "
        "0.875 in GSE69914. Among them, cg06268921 was significantly associated with "
        "overall survival (hazard ratio = 0.249, P = 0.044) and disease-free survival "
        "(hazard ratio = 0.194, P = 0.015) in the TCGA-TNBC cohort. In the cfDNA cohort, "
        "cg06268921 significantly differentiated TNBC from non-TNBC (P < 0.001), and the "
        "combination of both markers yielded an AUC of 0.728. The findings demonstrated "
        "the potential of methylation signatures as non-invasive diagnostic tools for "
        "TNBC. Future research with larger cohorts is warranted."
    )

    _DRAFT = {
        "pmid": "40860669",
        "sample_type": "plasma_cfdna",
        "performance_metrics": {
            "auc_training": None,
            "auc_validation": 0.922,
            "auc_external": None,
        },
        "dataset_ids": ["GSE50132", "TCGA", "GSE69914"],
        "confidence_level": "medium",
        "needs_human_review": False,
        "reason": "",
    }

    class _FakeLLM:
        """Stub chat model — no network/API key available in this sandbox.

        Returns the response a correct reviewer SHOULD produce for the fixture
        above, so this test exercises review_extraction()'s parsing/merge
        logic end-to-end. It does NOT validate the reviewer prompt's actual
        reasoning quality — that requires a live LLM call against a real
        backend (see README note printed below).
        """

        def invoke(self, messages):
            class _Resp:
                content = json.dumps({
                    "performance_metrics": {
                        "auc_training": None,
                        "auc_validation": None,  # 0.922 is the TCGA/tissue AUC, not cfDNA
                        "auc_external": None,
                    },
                    "dataset_ids": ["TCGA", "GSE69914"],  # GSE50132 is a WBC background filter panel
                    "needs_human_review": True,
                    "reason": (
                        "auc_validation=0.922 is reported for the TCGA tissue cohort, not the "
                        "plasma_cfdna sample_type assigned to this record (real cfDNA AUC is "
                        "0.728, in a different sentence); nulled. GSE50132 is cited only as a "
                        "background/noise-filtering WBC panel, not analyzed data; removed."
                    ),
                })
            return _Resp()

    if "--live" in sys.argv:
        import yaml
        from utils.llm_factory import get_llm

        cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        llm = get_llm(cfg["llm"])
        print(f"=== extraction_reviewer LIVE test (PMID 40860669) — backend={cfg['llm']['backend']} ===\n")
    else:
        llm = _FakeLLM()
        print("=== extraction_reviewer mock test (PMID 40860669) ===")
        print("NOTE: no LLM API key is available in this sandbox — using a stubbed LLM that")
        print("returns the expected-correct reviewer verdict, to exercise the merge/parsing")
        print("logic only. Re-run with `--live` on a machine with API access to validate")
        print("actual reviewer reasoning quality.\n")

    corrected = review_extraction(_ABSTRACT, _DRAFT, llm)
    print(json.dumps(corrected, ensure_ascii=False, indent=2))

    if "--live" not in sys.argv:
        assert corrected["performance_metrics"]["auc_validation"] is None, "auc_validation should be nulled"
        assert "GSE50132" not in (corrected["dataset_ids"] or []), "GSE50132 should be removed"
        assert "GSE50132" in corrected.get("excluded_reference_datasets", []), "GSE50132 should be logged as excluded"
        assert corrected["needs_human_review"] is True
        print("\nAll assertions passed (merge/parsing logic verified against stubbed response).")
    else:
        print("\n--- Manually check above: did auc_validation get nulled? Was GSE50132 removed? ---")
