"""
Checklist reviewer for MethyAgent — a per-check restructuring of the
second-pass review done by tools/extraction_reviewer.py review_extraction().

WHY THIS EXISTS (Z.ai evaluation team, 2026-08):
    The current reviewer renders ONE holistic judgment over a whole record and
    returns it as {risk_level, flags, corrected_fields, needs_human_review,
    reason}. That has two costs:

      1. Humans can only audit the record-level outcome. There is no per-claim
         verdict to sign off on, so a wrong null and a right null look the same
         from the outside.
      2. Check scope bleeds. review_extraction()'s CHECK A asks "is this AUC
         measured on the sample type this record claims?", and in practice the
         model widens that into "is this the training or the validation cohort?"
         — a question nobody asked it. Valid AUCs get nulled on cohort-
         attribution grounds (over-nulling), which the regex backstop then
         cannot undo.

    This module renders a SEPARATE verdict per check (C1..C7), each with its
    own narrow scope, its own verbatim evidence quote, and its own confidence.
    Humans audit each verdict individually (see scripts/eval_reviewer.py, which
    emits a one-row-per-verdict audit worksheet).

DROP-IN COMPATIBILITY:
    review_by_checklist() returns the same record-shaped result as
    review_extraction(): a copy of the extraction with corrections applied
    in-place and result["review_report"] = {risk_level, flags,
    corrected_fields, needs_human_review, reason}. The per-check detail is
    additionally exposed at result["checklist"] (and mirrored at
    result["review_report"]["checklist"]).

    tools/extraction_reviewer.py is NOT modified by this module; both reviewers
    can run on the same record and be compared.

Public API:
    review_by_checklist(source_text, extraction, llm) -> dict
    check_family(check_id) -> str          (strip the ":<target>" suffix)
    CHECK_FAMILIES                          (ordered C1..C7 ids)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from tools.query_clarifier import _auc_value_candidates
from utils.logger import get_logger

logger = get_logger(__name__)

_AUC_KEYS = ("auc_training", "auc_validation", "auc_external")

# ------------------------------------------------------------------ #
#  Check vocabulary                                                   #
# ------------------------------------------------------------------ #
# A check_id is "<family>" or "<family>:<target>". The target disambiguates
# which AUC key / accession / marker the verdict is about, WITHOUT adding keys
# to the per-check object (the contract is exactly five keys).
C1_AUC_PRESENT = "C1_auc_present"
C2_AUC_TERMINOLOGY = "C2_auc_terminology"
C3_AUC_SAMPLE_MATCH = "C3_auc_sample_match"
C4_SAMPLE_TYPE_CORRECT = "C4_sample_type_correct"
C5_DATASET_PROVENANCE = "C5_dataset_provenance"
C6_SAMPLE_SIZE_PRESENT = "C6_sample_size_present"
C7_MARKER_PRESENT = "C7_marker_present"

CHECK_FAMILIES: Tuple[str, ...] = (
    C1_AUC_PRESENT,
    C2_AUC_TERMINOLOGY,
    C3_AUC_SAMPLE_MATCH,
    C4_SAMPLE_TYPE_CORRECT,
    C5_DATASET_PROVENANCE,
    C6_SAMPLE_SIZE_PRESENT,
    C7_MARKER_PRESENT,
)

_VERDICTS = ("PASS", "FAIL", "NOT_APPLICABLE")
_CONFIDENCES = ("high", "medium", "low")

# Failing family -> flag emitted in review_report["flags"]. The first three
# reuse extraction_reviewer.py's existing flag vocabulary so downstream
# consumers (orchestrator, review queue) need no changes.
_FAMILY_TO_FLAG = {
    C1_AUC_PRESENT: "auc_unsupported",
    C2_AUC_TERMINOLOGY: "auc_unsupported",
    C3_AUC_SAMPLE_MATCH: "sample_type_mismatch",
    C4_SAMPLE_TYPE_CORRECT: "sample_type_mismatch",
    C5_DATASET_PROVENANCE: "reference_accession",
    C6_SAMPLE_SIZE_PRESENT: "sample_size_unsupported",
    C7_MARKER_PRESENT: "marker_unsupported",
}


def check_family(check_id: str) -> str:
    """"C1_auc_present:auc_validation" -> "C1_auc_present"."""
    return (check_id or "").split(":", 1)[0]


def check_target(check_id: str) -> Optional[str]:
    """"C5_dataset_provenance:GSE50132" -> "GSE50132"; None if no target."""
    parts = (check_id or "").split(":", 1)
    return parts[1] if len(parts) == 2 else None


def _strip_json_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ------------------------------------------------------------------ #
#  System prompt                                                      #
# ------------------------------------------------------------------ #
# The SCOPE RULE below is the whole point of this module — see the C2 example.
_CHECKLIST_SYSTEM = """You are a biomedical literature verification assistant working through \
a CHECKLIST. Another model extracted structured fields from a source text (usually a paper \
abstract). You do NOT re-extract anything and you do NOT rewrite the record. You render one \
independent verdict per checklist item, and a human will audit each of your verdicts \
individually.

You are given the source text, the extracted record, and an explicit list of checks to render.
Answer EVERY check in that list, once each, using the exact check_id given to you.

=== CRITICAL SCOPE RULE (read twice) ===
Each check answers ONLY its own question. Nothing else.
  * Do not import a concern from one check into another.
  * Never FAIL a check on grounds outside that check's stated scope. If you notice a problem
    that belongs to a different check, mention it in that OTHER check, not this one.
  * COHORT ATTRIBUTION IS NOT ON THIS CHECKLIST. Whether a number belongs to the training,
    validation, internal, or external cohort is NOT something you are being asked. It is not
    a reason to FAIL any check here. In particular, C2 asks SOLELY whether the number is
    termed an AUC/AUROC/ROC/C-statistic in the source text; if the text calls it an AUC, C2
    is a PASS even if you suspect it came from a different cohort than the field name says.
  * ABSENT EVIDENCE IS NOT CONTRADICTORY EVIDENCE. If the source text simply does not say
    enough to decide, return NOT_APPLICABLE — not FAIL. FAIL means the source text actively
    contradicts the record.
  * Silence about a sample type, a cohort, or a dataset's role is silence, not a violation.

=== CHECK DEFINITIONS ===
C1_auc_present         Does the reported AUC value appear verbatim in the source text (as the
                       same number, or the same number written as a percentage)? This is a
                       pure string-presence question. PASS if the number is there in any form;
                       FAIL if the number is not in the text at all. Do not consider what the
                       number is called or which cohort it is from.
C2_auc_terminology     Is that number actually called an AUC, AUROC, ROC / "area under the
                       receiver operating characteristic curve", C-statistic, or C-index in
                       the source text? FAIL if the text labels that number as something else
                       (sensitivity, specificity, accuracy, a "diagnostic score", a p-value, a
                       hazard ratio, a beta value, a sample count). NOT_APPLICABLE if the
                       number cannot be located at all. Cohort is irrelevant here.
C3_auc_sample_match    Is that AUC measured in the sample type the record claims (the record's
                       sample_type field)? FAIL only if the source text states the AUC was
                       measured on a DIFFERENT material (e.g. record says plasma cfDNA but the
                       text attributes that AUC to tissue/tumour). If the text does not say
                       which material that AUC came from, return NOT_APPLICABLE. Training vs
                       validation vs external is NOT a sample type — ignore that distinction.
C4_sample_type_correct Does the record's declared sample_type match the material the source
                       text describes as the study's primary analysed material? FAIL on a
                       direct contradiction (record says tissue, text is a plasma cfDNA study,
                       or vice versa). NOT_APPLICABLE if the text is genuinely ambiguous or
                       describes several materials without a clear primary.
C5_dataset_provenance  For THIS accession: does the source text present it as the study's own
                       primary/analysis data (including a cohort used to compute a reported
                       metric, e.g. an external validation cohort), or ONLY as a
                       reference/background/normalization/noise-filtering/annotation panel
                       used to pre-process or filter the authors' own data? PASS = primary or
                       analysis data. FAIL = reference/background panel only. NOT_APPLICABLE =
                       the accession does not appear in the source text at all, or its role is
                       not stated.
C6_sample_size_present Do the record's case/control sample-size integers appear in the source
                       text? PASS if each stated integer appears (as "n = 33", "33 patients",
                       "33 TNBC", etc). FAIL if a number was invented. NOT_APPLICABLE if the
                       record states no sample sizes.
C7_marker_present      Does this marker / gene / CpG ID appear in the source text? PASS if
                       present (case-insensitive, allow "cg06268921" vs "CG06268921").
                       FAIL if absent from the text. NOT_APPLICABLE if the record lists no
                       markers.

=== OUTPUT FORMAT ===
Output ONLY valid JSON, no markdown fences:
{
  "checks": [
    {
      "check_id": "<exactly the check_id you were given>",
      "verdict": "PASS" | "FAIL" | "NOT_APPLICABLE",
      "evidence": "<verbatim quote from the source text supporting the verdict, or null>",
      "confidence": "high" | "medium" | "low",
      "note": "<one sentence explaining this verdict, and nothing about other checks>"
    }
  ]
}
Each check object must have EXACTLY these five keys. "evidence" must be copied verbatim from
the source text (no paraphrase); use null only when no quote applies. Return one object per
check_id you were given — no extras, no omissions."""


# ------------------------------------------------------------------ #
#  Deterministic backstops (string-presence questions)                #
# ------------------------------------------------------------------ #
# C1/C6/C7 are literally "does this string occur in the text". Where code can
# decide that objectively it does, and the LLM verdict is recorded alongside so
# a human auditor can see any disagreement.

def _value_in_text(source_text: str, value: Any) -> Optional[str]:
    """
    Return the matched textual form of a numeric value in source_text, or None.

    Uses a digit boundary so "0.7" does not spuriously match inside "0.75"
    (a plain substring search would, and would wrongly clear a fabricated value).
    """
    if not source_text or value is None:
        return None
    for candidate in sorted(_auc_value_candidates(value), key=len, reverse=True):
        pattern = r"(?<![\d.])" + re.escape(candidate) + r"(?![\d])"
        if re.search(pattern, source_text):
            return candidate
    return None


_INT_TOKEN_RE_TMPL = r"(?<!\d){}(?!\d)"


def _int_in_text(source_text: str, value: Any) -> bool:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return False
    return bool(re.search(_INT_TOKEN_RE_TMPL.format(n), source_text or ""))


def _marker_label(marker: Any) -> str:
    """markers_or_panel entries are {"id","gene","type"} dicts or bare strings."""
    if isinstance(marker, dict):
        return str(marker.get("id") or marker.get("gene") or "").strip()
    return str(marker or "").strip()


def _marker_in_text(source_text: str, marker: Any) -> bool:
    text = (source_text or "").lower()
    if isinstance(marker, dict):
        parts = [marker.get("id"), marker.get("gene")]
    else:
        parts = [marker]
    for part in parts:
        part = str(part or "").strip().lower()
        if part and part in text:
            return True
    return False


# ------------------------------------------------------------------ #
#  Check planning                                                     #
# ------------------------------------------------------------------ #

def _plan_checks(source_text: str, extraction: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build the explicit list of checks to render for this record. Only checks
    that have something to look at are planned; anything the record does not
    assert is simply not on the list (rather than being asked and answered
    NOT_APPLICABLE by the model at extra cost).
    """
    planned: List[Dict[str, str]] = []
    metrics = extraction.get("performance_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    for key in _AUC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        subject = f"{key} = {value}"
        planned.append({
            "check_id": f"{C1_AUC_PRESENT}:{key}",
            "subject": subject,
            "question": f"Does the number {value} (reported as {key}) appear verbatim in the source text?",
        })
        planned.append({
            "check_id": f"{C2_AUC_TERMINOLOGY}:{key}",
            "subject": subject,
            "question": (
                f"Is the number {value} called an AUC/AUROC/ROC/C-statistic in the source text "
                f"(as opposed to a sensitivity, specificity, accuracy, diagnostic score, "
                f"p-value, hazard ratio or count)? Do not consider which cohort it came from."
            ),
        })
        planned.append({
            "check_id": f"{C3_AUC_SAMPLE_MATCH}:{key}",
            "subject": subject,
            "question": (
                f"Does the source text state that the AUC {value} was measured on "
                f"{extraction.get('sample_type') or 'the record’s declared sample type'}? "
                f"FAIL only on a stated different material; NOT_APPLICABLE if unstated."
            ),
        })

    planned.append({
        "check_id": C4_SAMPLE_TYPE_CORRECT,
        "subject": f"sample_type = {extraction.get('sample_type')!r}",
        "question": (
            f"Does the record's declared sample_type "
            f"({extraction.get('sample_type')!r}) match the primary material described in the "
            f"source text?"
        ),
    })

    for accession in (extraction.get("dataset_ids") or []):
        if not isinstance(accession, str) or not accession.strip():
            continue
        planned.append({
            "check_id": f"{C5_DATASET_PROVENANCE}:{accession}",
            "subject": f"dataset_id {accession}",
            "question": (
                f"In the source text, is {accession} the study's own primary/analysis data "
                f"(PASS), or only a reference/background/normalization/annotation panel (FAIL)?"
            ),
        })

    sizes = {k: extraction.get(k) for k in ("sample_size_case", "sample_size_control")}
    if any(v is not None for v in sizes.values()):
        planned.append({
            "check_id": C6_SAMPLE_SIZE_PRESENT,
            "subject": f"sample_size_case={sizes['sample_size_case']}, sample_size_control={sizes['sample_size_control']}",
            "question": "Do these case/control integers appear in the source text?",
        })

    markers = extraction.get("markers_or_panel") or []
    for marker in markers[:12]:
        label = _marker_label(marker)
        if not label:
            continue
        planned.append({
            "check_id": f"{C7_MARKER_PRESENT}:{label}",
            "subject": f"marker {label}",
            "question": f"Does the marker/gene/CpG {label} appear in the source text?",
        })

    return planned


def _record_view(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of the record the checklist actually reasons about."""
    return {
        "sample_type": extraction.get("sample_type"),
        "performance_metrics": extraction.get("performance_metrics") or {},
        "dataset_ids": extraction.get("dataset_ids") or [],
        "sample_size_case": extraction.get("sample_size_case"),
        "sample_size_control": extraction.get("sample_size_control"),
        "markers_or_panel": extraction.get("markers_or_panel") or [],
    }


def _blank_check(check_id: str, note: str, verdict: str = "NOT_APPLICABLE") -> Dict[str, Any]:
    return {
        "check_id": check_id,
        "verdict": verdict,
        "evidence": None,
        "confidence": "low",
        "note": note,
    }


def _normalize_check(raw: Any, check_id: str) -> Dict[str, Any]:
    """Coerce one model-produced check object into the exact five-key contract."""
    if not isinstance(raw, dict):
        return _blank_check(check_id, "Reviewer returned no usable verdict for this check.")
    verdict = str(raw.get("verdict") or "").strip().upper().replace(" ", "_")
    if verdict in ("NA", "N/A", "NOT_APPLICABLE", "NOTAPPLICABLE"):
        verdict = "NOT_APPLICABLE"
    if verdict not in _VERDICTS:
        verdict = "NOT_APPLICABLE"
    confidence = str(raw.get("confidence") or "").strip().lower()
    if confidence not in _CONFIDENCES:
        confidence = "low"
    evidence = raw.get("evidence")
    if evidence is not None:
        evidence = str(evidence).strip() or None
    note = str(raw.get("note") or "").strip() or "(no note given)"
    return {
        "check_id": check_id,
        "verdict": verdict,
        "evidence": evidence,
        "confidence": confidence,
        "note": note,
    }


def _apply_deterministic_backstops(
    checks: List[Dict[str, Any]],
    source_text: str,
    extraction: Dict[str, Any],
) -> None:
    """
    Override/confirm the purely mechanical checks in place.

    C1 is decidable in code in BOTH directions (the value is either in the text
    or it is not). C6/C7 are only allowed to force a PASS — if code cannot find
    the string, the model's reading (which tolerates "thirty-three" or a
    reformatted gene name) is kept, so this never manufactures a FAIL.
    """
    metrics = extraction.get("performance_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    for check in checks:
        family, target = check_family(check["check_id"]), check_target(check["check_id"])

        if family == C1_AUC_PRESENT and target in _AUC_KEYS:
            matched = _value_in_text(source_text, metrics.get(target))
            deterministic = "PASS" if matched else "FAIL"
            if check["verdict"] != deterministic:
                check["note"] = (
                    f"{check['note']} [string-match backstop: value "
                    f"{'found' if matched else 'not found'} in source text; "
                    f"model said {check['verdict']}, overridden to {deterministic}]"
                )
                check["verdict"] = deterministic
                check["confidence"] = "high"
                if matched and not check.get("evidence"):
                    check["evidence"] = _quote_around(source_text, matched)
                elif not matched:
                    check["evidence"] = None
            else:
                check["confidence"] = "high"

        elif family == C6_SAMPLE_SIZE_PRESENT and check["verdict"] == "FAIL":
            values = [extraction.get("sample_size_case"), extraction.get("sample_size_control")]
            stated = [v for v in values if v is not None]
            if stated and all(_int_in_text(source_text, v) for v in stated):
                check["verdict"] = "PASS"
                check["note"] = f"{check['note']} [string-match backstop: all stated integers occur in the source text]"

        elif family == C7_MARKER_PRESENT and check["verdict"] == "FAIL" and target:
            if target.lower() in (source_text or "").lower():
                check["verdict"] = "PASS"
                check["note"] = f"{check['note']} [string-match backstop: marker string occurs in the source text]"


def _quote_around(source_text: str, needle: str, window: int = 90) -> Optional[str]:
    idx = source_text.find(needle)
    if idx == -1:
        return None
    return source_text[max(0, idx - window): idx + len(needle) + window].strip()


# ------------------------------------------------------------------ #
#  Aggregation                                                        #
# ------------------------------------------------------------------ #

def _aggregate(
    checks: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Turn per-check verdicts into the review_report shape and apply corrections
    to `result` in place. Returns (review_report, changes).

    Correction policy — deliberately conservative, because over-nulling is the
    bug this module exists to fix:
      * only a FAIL ever corrects anything (NOT_APPLICABLE never does);
      * a low-confidence FAIL flags for human review but does NOT null/drop.
    """
    changes: List[str] = []
    flags: List[str] = []
    corrected_fields: Dict[str, Any] = {}
    failing_notes: List[str] = []

    metrics = result.get("performance_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    auc_keys_to_null: List[Tuple[str, str]] = []   # (auc_key, reason check_id)
    accessions_to_drop: List[Tuple[str, str]] = []

    for check in checks:
        if check["verdict"] != "FAIL":
            continue
        family, target = check_family(check["check_id"]), check_target(check["check_id"])
        flag = _FAMILY_TO_FLAG.get(family)
        if flag and flag not in flags:
            flags.append(flag)
        failing_notes.append(f"{check['check_id']}: {check['note']}")

        actionable = check["confidence"] in ("high", "medium")
        if not actionable:
            continue
        if family in (C1_AUC_PRESENT, C2_AUC_TERMINOLOGY, C3_AUC_SAMPLE_MATCH) and target in _AUC_KEYS:
            auc_keys_to_null.append((target, check["check_id"]))
        elif family == C5_DATASET_PROVENANCE and target:
            accessions_to_drop.append((target, check["check_id"]))

    for key, check_id in auc_keys_to_null:
        if metrics.get(key) is not None:
            changes.append(f"{key}: {metrics[key]!r} -> None ({check_id})")
            metrics[key] = None
            corrected_fields["performance_metrics"] = dict(metrics)

    if accessions_to_drop:
        orig_ids = result.get("dataset_ids") or []
        drop = {a for a, _ in accessions_to_drop}
        kept = [d for d in orig_ids if d not in drop]
        removed = [d for d in orig_ids if d in drop]
        if removed:
            result["dataset_ids"] = kept or None
            existing_excluded = result.get("excluded_reference_datasets") or []
            result["excluded_reference_datasets"] = existing_excluded + [
                d for d in removed if d not in existing_excluded
            ]
            changes.append(f"dataset_ids: removed reference-only {removed}")
            corrected_fields["dataset_ids"] = result["dataset_ids"]

    n_fail = sum(1 for c in checks if c["verdict"] == "FAIL")
    needs_human_review = bool(n_fail) or bool(changes)

    if "auc_unsupported" in flags or len(flags) >= 2:
        risk_level = "high"
    elif flags:
        risk_level = "medium"
    else:
        risk_level = "low"

    if n_fail:
        reason = (
            f"{n_fail} of {len(checks)} checks failed — "
            + "; ".join(failing_notes[:4])
            + ("; ..." if len(failing_notes) > 4 else "")
        )
    else:
        n_na = sum(1 for c in checks if c["verdict"] == "NOT_APPLICABLE")
        reason = f"all {len(checks)} checks passed or were not applicable ({n_na} N/A); no corrections made"

    report = {
        "risk_level": risk_level,
        "flags": flags,
        "corrected_fields": corrected_fields,
        "needs_human_review": needs_human_review,
        "reason": reason,
        "checklist": checks,
    }
    return report, changes


# ------------------------------------------------------------------ #
#  Public entry point                                                 #
# ------------------------------------------------------------------ #

def review_by_checklist(
    source_text: str,
    extraction: Dict[str, Any],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Review a draft extraction as an explicit checklist, one verdict per check.

    Args:
        source_text: The text the extraction was derived from (abstract, or
                     abstract + relevant full-text section).
        extraction:  Output dict from extract_paper_structured() (or any dict
                     with the same field names).
        llm:         LangChain chat model.

    Returns:
        A copy of `extraction` with corrections applied in place, plus:
          result["review_report"] = {risk_level, flags, corrected_fields,
                                     needs_human_review, reason, checklist}
          result["checklist"]     = [ {check_id, verdict, evidence,
                                       confidence, note}, ... ]
        This is drop-in compatible with
        tools.extraction_reviewer.review_extraction(); the extra "checklist"
        key is additive.

        Each check_id is "<family>" or "<family>:<target>" where family is one
        of CHECK_FAMILIES and target names the AUC key / accession / marker the
        verdict is about. Use check_family()/check_target() to split them.

        On reviewer failure (bad JSON, LLM exception) every planned check is
        returned as NOT_APPLICABLE with a "reviewer_error" flag and
        needs_human_review=True; the record is NOT modified.
    """
    result = dict(extraction)
    if isinstance(result.get("performance_metrics"), dict):
        result["performance_metrics"] = dict(result["performance_metrics"])

    planned = _plan_checks(source_text or "", result)

    if not source_text or not planned:
        report = {
            "risk_level": "low", "flags": [], "corrected_fields": {},
            "needs_human_review": False,
            "reason": "nothing to check (no source text or no assertions in the record)",
            "checklist": [],
        }
        result["review_report"] = report
        result["checklist"] = []
        return result

    context = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"EXTRACTED RECORD (fields under review):\n"
        f"{json.dumps(_record_view(result), ensure_ascii=False, indent=2)}\n\n"
        f"CHECKS TO RENDER ({len(planned)}) — answer each one, using its exact check_id:\n"
        f"{json.dumps(planned, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(content=_CHECKLIST_SYSTEM),
        HumanMessage(content=context),
    ]

    try:
        response = llm.invoke(messages)
        payload = json.loads(_strip_json_fences(response.content))
        raw_checks = payload.get("checks") if isinstance(payload, dict) else payload
        if not isinstance(raw_checks, list):
            raise ValueError("response has no 'checks' list")
    except Exception as e:
        note = f"Checklist reviewer failed ({type(e).__name__}: {e}); no verdicts rendered."
        checks = [_blank_check(p["check_id"], note) for p in planned]
        result["needs_human_review"] = True
        result["reason"] = f"{result.get('reason') or ''} {note}".strip()
        report = {
            "risk_level": "medium", "flags": ["reviewer_error"], "corrected_fields": {},
            "needs_human_review": True, "reason": note, "checklist": checks,
        }
        result["review_report"] = report
        result["checklist"] = checks
        logger.warning(f"checklist_reviewer: {note} (PMID {result.get('pmid', '')})")
        return result

    by_id: Dict[str, Any] = {}
    for raw in raw_checks:
        if isinstance(raw, dict) and raw.get("check_id"):
            by_id.setdefault(str(raw["check_id"]).strip(), raw)

    checks: List[Dict[str, Any]] = []
    for plan in planned:
        cid = plan["check_id"]
        raw = by_id.get(cid)
        if raw is None:
            # tolerate a model that dropped the ":<target>" suffix when the
            # family has only one instance
            family_matches = [v for k, v in by_id.items() if check_family(k) == check_family(cid)]
            raw = family_matches[0] if len(family_matches) == 1 else None
        if raw is None:
            checks.append(_blank_check(cid, "Reviewer did not return a verdict for this check."))
        else:
            checks.append(_normalize_check(raw, cid))

    _apply_deterministic_backstops(checks, source_text, result)

    report, changes = _aggregate(checks, result)
    result["review_report"] = report
    result["checklist"] = checks

    if report["needs_human_review"]:
        result["needs_human_review"] = True
        result["reason"] = f"{result.get('reason') or ''} [checklist] {report['reason']}".strip()

    if changes:
        logger.info(
            f"checklist_reviewer corrected PMID {result.get('pmid', '')}: {'; '.join(changes)}"
        )
    return result


# ============================================================
# Quick test (mirrors tools/extraction_reviewer.py's two modes):
#   python -m tools.checklist_reviewer          -> stubbed LLM, no network
#   python -m tools.checklist_reviewer --live   -> real get_llm(config["llm"])
# ============================================================

if __name__ == "__main__":
    _ABSTRACT = (
        "BACKGROUND: Preoperative identification of breast cancer (BC) subtypes is "
        "essential for optimizing treatment strategies. This study aimed to identify "
        "circulating cell-free DNA (cfDNA) methylation signatures to differentiate "
        "triple-negative breast cancer (TNBC) from other BC subtypes (non-TNBC). "
        "METHODS: We initially performed a genome-wide analysis to identify "
        "differentially methylated CpG sites between five TNBC and nine non-TNBC "
        "tissues using the Infinium HumanMethylationEPIC BeadChip. These DMCs were "
        "further validated using large-scale data from the Cancer Genome Atlas (TCGA, "
        "n = 774), and only CpG sites with average beta values > 0.90 or < 0.10 in "
        "white blood cells (GSE50132, n = 233) were retained to minimize potential "
        "background methylation interference. A multiplex digital droplet PCR assay "
        "was developed to simultaneously detect cg06268921 and cg23247845 in cfDNA "
        "from TNBC (n = 33) and non-TNBC (n = 80) patients. "
        "RESULTS: An eight-marker diagnostic panel was developed with an AUC of 0.922 "
        "in TCGA and 0.875 in GSE69914. In the cfDNA cohort, the combination of both "
        "markers yielded an AUC of 0.728."
    )

    _DRAFT = {
        "pmid": "40860669",
        "sample_type": "plasma_cfdna",
        "performance_metrics": {"auc_training": None, "auc_validation": 0.728, "auc_external": 0.911},
        "dataset_ids": ["GSE50132", "TCGA", "GSE69914"],
        "sample_size_case": 33,
        "sample_size_control": 80,
        "markers_or_panel": [{"id": "cg06268921", "gene": None, "type": "CpG"}],
        "confidence_level": "medium",
        "needs_human_review": False,
        "reason": "",
    }

    class _FakeLLM:
        """Stub chat model: returns the verdicts a correctly-scoped reviewer
        should produce for the fixture above. Exercises planning, parsing,
        backstops and aggregation without network/API access."""

        def invoke(self, messages):
            planned = json.loads(messages[1].content.split("CHECKS TO RENDER")[1].split("\n", 1)[1])
            out = []
            for p in planned:
                cid = p["check_id"]
                fam, tgt = check_family(cid), check_target(cid)
                if fam == C1_AUC_PRESENT:
                    # deliberately wrong on the fabricated 0.911 — the string
                    # backstop must override it to FAIL
                    out.append({"check_id": cid, "verdict": "PASS", "evidence": "AUC of 0.728",
                                "confidence": "medium", "note": "value seen in text"})
                elif fam == C2_AUC_TERMINOLOGY:
                    out.append({"check_id": cid, "verdict": "PASS", "evidence": "yielded an AUC of 0.728",
                                "confidence": "high", "note": "the text calls this number an AUC"})
                elif fam == C3_AUC_SAMPLE_MATCH:
                    out.append({"check_id": cid, "verdict": "PASS", "evidence": "In the cfDNA cohort",
                                "confidence": "high", "note": "0.728 is the cfDNA cohort AUC"})
                elif fam == C4_SAMPLE_TYPE_CORRECT:
                    out.append({"check_id": cid, "verdict": "PASS", "evidence": "cfDNA methylation signatures",
                                "confidence": "medium", "note": "the study's assay is run on cfDNA"})
                elif fam == C5_DATASET_PROVENANCE:
                    ref = tgt == "GSE50132"
                    out.append({"check_id": cid, "verdict": "FAIL" if ref else "PASS",
                                "evidence": "GSE50132, n = 233) were retained to minimize potential background methylation interference" if ref else f"{tgt}",
                                "confidence": "high",
                                "note": "background-filter panel only" if ref else "analysis cohort"})
                elif fam == C6_SAMPLE_SIZE_PRESENT:
                    out.append({"check_id": cid, "verdict": "PASS", "evidence": "TNBC (n = 33) and non-TNBC (n = 80)",
                                "confidence": "high", "note": "both integers occur in the text"})
                else:
                    out.append({"check_id": cid, "verdict": "PASS", "evidence": "cg06268921",
                                "confidence": "high", "note": "marker occurs in the text"})

            class _Resp:
                content = json.dumps({"checks": out})
            return _Resp()

    if "--live" in sys.argv:
        import yaml
        from utils.llm_factory import get_llm

        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "settings.yaml"))
        llm = get_llm(cfg["llm"], json_mode=True)
        print(f"=== checklist_reviewer LIVE test (PMID 40860669) — backend={cfg['llm']['backend']} ===\n")
    else:
        llm = _FakeLLM()
        print("=== checklist_reviewer mock test (PMID 40860669) ===\n")

    reviewed = review_by_checklist(_ABSTRACT, _DRAFT, llm)
    for c in reviewed["checklist"]:
        print(f"  [{c['verdict']:14s}] {c['check_id']:40s} ({c['confidence']}) {c['note'][:90]}")
    print("\nreview_report:")
    print(json.dumps({k: v for k, v in reviewed["review_report"].items() if k != "checklist"},
                     ensure_ascii=False, indent=2))
    print("\nrecord after corrections:")
    print(json.dumps({k: reviewed.get(k) for k in
                      ("performance_metrics", "dataset_ids", "excluded_reference_datasets",
                       "needs_human_review")}, ensure_ascii=False, indent=2))

    if "--live" not in sys.argv:
        pm = reviewed["performance_metrics"]
        assert pm["auc_validation"] == 0.728, "0.728 is in the text and correctly scoped — must NOT be nulled"
        assert pm["auc_external"] is None, "0.911 is absent from the text — C1 backstop must null it"
        assert "GSE50132" not in (reviewed["dataset_ids"] or []), "GSE50132 is a background panel — must be dropped"
        assert "GSE50132" in reviewed.get("excluded_reference_datasets", [])
        assert reviewed["review_report"]["needs_human_review"] is True
        assert {c["check_id"] for c in reviewed["checklist"]} >= {
            f"{C1_AUC_PRESENT}:auc_validation", f"{C5_DATASET_PROVENANCE}:GSE50132", C4_SAMPLE_TYPE_CORRECT}
        print("\nAll assertions passed (planning / backstop / aggregation verified against stubbed verdicts).")
