#!/usr/bin/env python3
"""
Field-level grounding evaluation for MethyAgent extractions.

No gold standard is needed: the source text the extractor was shown IS the
reference. For every extracted field we ask one question — "is this value
actually supported by that text?" — and answer it with deterministic string /
regex checks, reusing the validators that already live in tools/.

Verdicts per field instance:
    GROUNDED        the value is present in the source text (and, for AUC,
                    labelled as an AUC/AUROC/ROC/C-statistic nearby)
    UNGROUNDED      the value is absent, or present but unsupported
    NOT_APPLICABLE  nothing to check (field null/empty, or not a checkable
                    claim — e.g. sample_type "mixed" has no synonym set)

Checks implemented:
    1. AUC values             number verbatim AND an AUC term within ~120 chars
    2. sample_size_case/control  integer appears in the source text
    3. dataset_ids            accession appears; context PRIMARY vs REFERENCE
    4. markers_or_panel       each CpG id / gene symbol appears
    5. sample_type            a synonym of the assigned enum appears

WHY THIS SCRIPT RE-RUNS THE STAGE-2 LOOP INSTEAD OF CALLING search_and_extract():
stage2_extract() picks its source text per record (PMC full text when
available, else the abstract) and records only *which kind* it used, in
`source_basis` — the actual ~12k-char extraction_text is discarded. Grounding a
"fulltext" record against its abstract would be measuring the wrong reference.
So we reproduce that loop here from the same imported primitives and keep the
text, writing it into the output as `_source_text`. That is also what makes
`--from-json` a genuinely network-free re-analysis.

No module under tools/ is modified; everything is imported.

Usage:
    python scripts/eval_grounding.py --query "CRC plasma cfDNA methylation" --top-n 10
    python scripts/eval_grounding.py --from-json data/eval/grounding_20260812_101500.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_dotenv() -> None:
    """Populate os.environ from .env without clobbering the real environment.
    Same helper as scripts/eval_consistency.py."""
    p = _REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

# --- imported, never modified -------------------------------------------------
from tools.extraction_reviewer import (          # noqa: E402
    _AUC_LABEL_RE,
    _AUC_UNSUPPORTED_WINDOW,
    _check_auc_unsupported,
    review_extraction,
)
from tools.ncbi_search import (                  # noqa: E402
    _fetch_fulltext_safe,
    _prepare_fulltext_for_extraction,
    _resolve_proxy,
    fetch_pubmed_records,
    stage1_filter,
)
from tools.parser_tools import parse_query_rules  # noqa: E402
from tools.query_clarifier import (              # noqa: E402
    _REFERENCE_DATASET_KEYWORDS,
    _SAMPLE_TYPE_SYNONYMS,
    _auc_value_candidates,
    build_pubmed_query_with_controls,
    extract_paper_structured,
)

GROUNDED = "GROUNDED"
UNGROUNDED = "UNGROUNDED"
NOT_APPLICABLE = "NOT_APPLICABLE"

# Graded window for check 1. The spec calls for ~120 chars; the module backstop
# _check_auc_unsupported() uses its own _AUC_UNSUPPORTED_WINDOW (150). We grade
# at 120 and report the module's verdict alongside, flagging any divergence,
# rather than silently preferring one threshold over the other.
_DEFAULT_AUC_WINDOW = 120
_SNIPPET_CHARS = 80          # +/- chars of evidence around a hit
_DATASET_CONTEXT_WORDS = 15  # mirrors _exclude_reference_datasets' window

# _AUC_LABEL_RE (auc|auroc|\broc\b|c-statistic|c-index) was tuned for abstracts,
# where "AUC" is nearly always abbreviated. Full text spells it out -- "the area
# under the receiver operating characteristic curve of 0.891" contains no token
# the module regex matches (\broc\b does not fire inside "characteristic"). Left
# unaddressed, that systematically under-scores fulltext records and corrupts the
# headline "did full text improve grounding?" comparison. We therefore accept the
# spelled-out forms too, count how many values are grounded *only* that way, and
# still report the pure-module verdict per value as `module_backstop`.
# --strict-auc-label restores module-regex-only grading.
_AUC_LABEL_EXPANDED_RE = re.compile(
    r"area under the (receiver operating characteristic |roc )?curve"
    r"|receiver operating characteristic"
    r"|concordance (statistic|index)",
    re.IGNORECASE,
)

_AUC_KEYS = ("auc_training", "auc_validation", "auc_external")

_FIELDS = ("auc", "sample_size", "dataset_ids", "markers_or_panel", "sample_type")

# markers_or_panel entry classification
_CPG_RE = re.compile(r"^cg\d{6,}$", re.IGNORECASE)
_GENE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{1,14}$")


# ------------------------------------------------------------------ #
#  Proxy                                                              #
# ------------------------------------------------------------------ #

def resolve_proxy() -> str:
    """
    config/settings.yaml ships geo.proxy INTENTIONALLY BLANK -- the real value is
    per-machine and lives in the environment. A naive cfg["geo"]["proxy"] read
    therefore runs unproxied and gets rate-limit-blocked by NCBI. Resolve as
    NCBI_PROXY -> HTTPS_PROXY -> config, and publish the winner back into
    NCBI_PROXY so ncbi_search._resolve_proxy() (which checks NCBI_PROXY first,
    and caches its answer) picks it up without us editing that module.

    Same contract as resolve_proxy() in scripts/eval_consistency.py.
    """
    cfg_proxy = ""
    try:
        import yaml
        with open(_REPO_ROOT / "config" / "settings.yaml") as fh:
            cfg = yaml.safe_load(fh) or {}
        cfg_proxy = (cfg.get("geo") or {}).get("proxy") or ""
    except Exception:
        pass

    proxy = (
        os.environ.get("NCBI_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or cfg_proxy
        or ""
    )
    if proxy and not os.environ.get("NCBI_PROXY"):
        os.environ["NCBI_PROXY"] = proxy
    return proxy


# ------------------------------------------------------------------ #
#  Evidence helpers                                                   #
# ------------------------------------------------------------------ #

def _snippet(text: str, idx: int, span: int = 0, pad: int = _SNIPPET_CHARS) -> str:
    """+/-`pad` chars of context around text[idx:idx+span], whitespace-collapsed."""
    if idx < 0:
        return ""
    start = max(0, idx - pad)
    end = min(len(text), idx + span + pad)
    frag = " ".join(text[start:end].split())
    return ("..." if start > 0 else "") + frag + ("..." if end < len(text) else "")


def _find_all(haystack: str, needle: str) -> List[int]:
    out, start = [], 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return out
        out.append(i)
        start = i + 1


def _find_number_occurrences(text: str, form: str) -> List[int]:
    """
    Occurrences of a numeric form with digit boundaries enforced.

    _auc_value_candidates() emits progressively lossier renderings (0.95 ->
    "0.950", "0.95", "0.9"), and a bare .find() lets the 1-decimal form match
    inside an unrelated longer number -- "0.9" hits the "0.90" of a beta-value
    threshold and reports a fabricated AUC as grounded. Require the form not to
    be preceded by a digit/decimal point nor followed by another digit.
    """
    pat = re.compile(r"(?<![\d.])" + re.escape(form) + r"(?!\d)")
    return [m.start() for m in pat.finditer(text)]


def _verdict(field: str, key: str, value: Any, verdict: str, reason: str,
             evidence: str = "", **extra: Any) -> Dict[str, Any]:
    item = {
        "field": field,
        "key": key,
        "value": value,
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }
    item.update(extra)
    return item


# ------------------------------------------------------------------ #
#  Check 1 — AUC values                                               #
# ------------------------------------------------------------------ #

def check_auc(record: Dict[str, Any], text: str, window: int,
              strict_label: bool = False) -> List[Dict[str, Any]]:
    """
    An AUC is GROUNDED when the number appears verbatim in the source text AND
    an AUC/AUROC/ROC/C-statistic term appears within +/-`window` chars.

    Reuses _AUC_LABEL_RE and _auc_value_candidates from the existing validators,
    with two deliberate refinements over _check_auc_unsupported():

      * every occurrence of every candidate form is scanned, not just the first
        locatable one, so a value written twice (once bare, once labelled) is not
        failed on the bare mention;
      * numeric forms are matched with digit boundaries (see
        _find_number_occurrences) so the lossy 1-decimal candidate cannot match
        inside an unrelated longer number.

    Each verdict also carries `module_backstop`, the unmodified
    _check_auc_unsupported() result at its native 150-char window, so any
    divergence is visible rather than silently resolved.
    """
    metrics = record.get("performance_metrics")
    out: List[Dict[str, Any]] = []

    if not isinstance(metrics, dict):
        return [_verdict("auc", k, None, NOT_APPLICABLE, "no performance_metrics on record")
                for k in _AUC_KEYS]

    module_unsupported = set(_check_auc_unsupported(text, metrics)) if text else set()

    for key in _AUC_KEYS:
        value = metrics.get(key)
        if value is None:
            out.append(_verdict("auc", key, None, NOT_APPLICABLE, "value is null"))
            continue

        candidates = _auc_value_candidates(value)
        if not candidates:
            out.append(_verdict("auc", key, value, UNGROUNDED,
                                "value is not numeric; no textual form to search for"))
            continue

        module_agrees_grounded = key not in module_unsupported

        # (form, index, module_label, expanded_label)
        hits: List[Tuple[str, int, bool, bool]] = []
        for form in candidates:
            for idx in _find_number_occurrences(text, form):
                lo = max(0, idx - window)
                hi = min(len(text), idx + len(form) + window)
                ctx = text[lo:hi]
                hits.append((form, idx,
                             bool(_AUC_LABEL_RE.search(ctx)),
                             bool(_AUC_LABEL_EXPANDED_RE.search(ctx))))

        if not hits:
            out.append(_verdict(
                "auc", key, value, UNGROUNDED,
                f"none of {candidates} appears verbatim in the source text",
                matched_form=None, label_within_chars=window,
                module_backstop="supported" if module_agrees_grounded else "unsupported",
                module_backstop_window=_AUC_UNSUPPORTED_WINDOW,
                divergence=module_agrees_grounded,
            ))
            continue

        def _rank(h: Tuple[str, int, bool, bool]) -> int:
            if h[2]:
                return 0                       # abbreviated AUC/ROC label
            if h[3] and not strict_label:
                return 1                       # spelled-out label
            return 2

        form, idx, mod_label, exp_label = min(hits, key=_rank)
        labelled = mod_label or (exp_label and not strict_label)
        verdict = GROUNDED if labelled else UNGROUNDED
        expanded_only = labelled and not mod_label

        if mod_label:
            reason = f"'{form}' found at char {idx} with an AUC/ROC term within {window} chars"
        elif expanded_only:
            reason = (f"'{form}' found at char {idx} with a spelled-out "
                      f"'area under the ... curve' label within {window} chars "
                      f"(_AUC_LABEL_RE does not match this phrasing)")
        else:
            reason = (f"'{form}' found at char {idx} but no AUC/AUROC/ROC/C-statistic term "
                      f"within {window} chars ({len(hits)} occurrence(s) checked)")

        out.append(_verdict(
            "auc", key, value, verdict, reason,
            _snippet(text, idx, len(form)),
            matched_form=form, char_offset=idx, occurrences=len(hits),
            label_within_chars=window,
            # the lossiest candidate forms are rounded; flag when one of those matched
            loose_match=(form != candidates[0]),
            expanded_label_only=expanded_only,
            module_backstop="supported" if module_agrees_grounded else "unsupported",
            module_backstop_window=_AUC_UNSUPPORTED_WINDOW,
            divergence=(verdict == GROUNDED) != module_agrees_grounded,
        ))
    return out


# ------------------------------------------------------------------ #
#  Check 2 — sample sizes                                             #
# ------------------------------------------------------------------ #

def _int_forms(n: int) -> List[str]:
    """Plain and comma-grouped renderings, e.g. 1143 -> ['1143', '1,143']."""
    forms = [str(n)]
    grouped = f"{n:,}"
    if grouped != forms[0]:
        forms.append(grouped)
    return forms


def check_sample_sizes(record: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """
    GROUNDED when the integer appears in the source text as a standalone number
    (not embedded in a longer number or a decimal such as an AUC of 0.143).
    """
    out: List[Dict[str, Any]] = []
    for key in ("sample_size_case", "sample_size_control"):
        raw = record.get(key)
        if raw is None:
            out.append(_verdict("sample_size", key, None, NOT_APPLICABLE, "value is null"))
            continue
        try:
            n = int(raw)
        except (TypeError, ValueError):
            out.append(_verdict("sample_size", key, raw, UNGROUNDED,
                                "value is not an integer; cannot locate in text"))
            continue

        found = None
        for form in _int_forms(n):
            # Not preceded by a digit / decimal point, not followed by a digit,
            # a decimal fraction, or a '%' -- those are other quantities.
            pat = re.compile(r"(?<![\d.,])" + re.escape(form) + r"(?![\d]|\.\d|%)")
            m = pat.search(text)
            if m:
                found = (form, m.start())
                break

        if found is None:
            out.append(_verdict("sample_size", key, n, UNGROUNDED,
                                f"integer {n} does not appear as a standalone number in the source text",
                                searched_forms=_int_forms(n)))
        else:
            form, idx = found
            out.append(_verdict("sample_size", key, n, GROUNDED,
                                f"'{form}' found at char {idx}",
                                _snippet(text, idx, len(form)),
                                matched_form=form, char_offset=idx))
    return out


# ------------------------------------------------------------------ #
#  Check 3 — dataset accessions                                       #
# ------------------------------------------------------------------ #

def _dataset_context(text: str, accession: str) -> Tuple[str, str]:
    """
    Classify each mention's context as PRIMARY or REFERENCE using the same
    keyword set and +/-15-word window as _exclude_reference_datasets().
    Returns (context, matched_keyword).
    """
    words = text.split()
    lower_words = [w.lower() for w in words]
    for idx, word in enumerate(words):
        if accession.lower() not in word.lower():
            continue
        window = " ".join(lower_words[max(0, idx - _DATASET_CONTEXT_WORDS):
                                      idx + _DATASET_CONTEXT_WORDS + 1])
        for kw in _REFERENCE_DATASET_KEYWORDS:
            if kw in window:
                return "REFERENCE", kw
    return "PRIMARY", ""


def check_dataset_ids(record: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """
    GROUNDED when the accession string appears in the source text.
    Presence is the grounding question; PRIMARY-vs-REFERENCE is reported as a
    separate `context` attribute, since a REFERENCE accession is a real (and
    present) mention that was simply mis-attributed as study data.
    """
    ids = record.get("dataset_ids")
    if not isinstance(ids, list) or not ids:
        return [_verdict("dataset_ids", "dataset_ids", None, NOT_APPLICABLE,
                         "no dataset_ids extracted")]

    out: List[Dict[str, Any]] = []
    lower_text = text.lower()
    for acc in ids:
        if not isinstance(acc, str) or not acc.strip():
            out.append(_verdict("dataset_ids", "accession", acc, UNGROUNDED,
                                "empty or non-string accession"))
            continue
        acc = acc.strip()
        idx = lower_text.find(acc.lower())
        if idx == -1:
            out.append(_verdict("dataset_ids", "accession", acc, UNGROUNDED,
                                "accession does not appear in the source text",
                                context="ABSENT"))
            continue
        context, kw = _dataset_context(text, acc)
        out.append(_verdict(
            "dataset_ids", "accession", acc, GROUNDED,
            f"found at char {idx}; context classified {context}"
            + (f" (keyword '{kw}' within {_DATASET_CONTEXT_WORDS} words)" if kw else ""),
            _snippet(text, idx, len(acc)),
            char_offset=idx, context=context, context_keyword=kw or None,
        ))
    return out


# ------------------------------------------------------------------ #
#  Check 4 — markers / panel                                          #
# ------------------------------------------------------------------ #

def _marker_tokens(entry: Any) -> Tuple[List[str], List[str]]:
    """
    Return (checkable_tokens, freetext_tokens) for one markers_or_panel entry.

    Entries are {id, gene, type} dicts (per the extractor schema) but plain
    strings are tolerated. A token is checkable when it looks like a CpG id or a
    gene symbol; a multi-word panel description ("eight-marker panel") is not a
    verbatim identifier and is reported NOT_APPLICABLE rather than counted as a
    grounding failure.
    """
    if isinstance(entry, str):
        raw = [entry]
    elif isinstance(entry, dict):
        raw = [entry.get("id"), entry.get("gene")]
    else:
        raw = []

    checkable, freetext = [], []
    for tok in raw:
        if not isinstance(tok, str):
            continue
        tok = tok.strip()
        if not tok:
            continue
        if _CPG_RE.match(tok) or _GENE_RE.match(tok):
            checkable.append(tok)
        else:
            freetext.append(tok)
    return checkable, freetext


def check_markers(record: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """GROUNDED when the gene symbol / CpG id appears in the source text."""
    markers = record.get("markers_or_panel")
    if not isinstance(markers, list) or not markers:
        return [_verdict("markers_or_panel", "markers_or_panel", None, NOT_APPLICABLE,
                         "no markers extracted")]

    out: List[Dict[str, Any]] = []
    for entry in markers:
        checkable, freetext = _marker_tokens(entry)
        mtype = entry.get("type") if isinstance(entry, dict) else None

        for tok in freetext:
            out.append(_verdict("markers_or_panel", "marker", tok, NOT_APPLICABLE,
                                "free-text panel description, not a verbatim identifier",
                                marker_type=mtype))

        for tok in checkable:
            # Case-sensitive first: a 3-letter gene symbol matched case-blind
            # would hit ordinary English words.
            m = re.search(r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])", text)
            case_sensitive = m is not None
            if m is None:
                m = re.search(r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])",
                              text, re.IGNORECASE)
            if m is None:
                out.append(_verdict("markers_or_panel", "marker", tok, UNGROUNDED,
                                    "identifier does not appear in the source text",
                                    marker_type=mtype))
            else:
                out.append(_verdict(
                    "markers_or_panel", "marker", tok, GROUNDED,
                    f"found at char {m.start()}"
                    + ("" if case_sensitive else " (case-insensitive match)"),
                    _snippet(text, m.start(), len(tok)),
                    marker_type=mtype, char_offset=m.start(),
                    case_sensitive_match=case_sensitive,
                ))
    return out


# ------------------------------------------------------------------ #
#  Check 5 — sample_type                                              #
# ------------------------------------------------------------------ #

def check_sample_type(record: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """
    GROUNDED when at least one synonym of the assigned enum value appears in the
    source text. Reuses _SAMPLE_TYPE_SYNONYMS, which deliberately omits
    mixed/unknown -- those get NOT_APPLICABLE, there is no synonym set to check.
    """
    st = record.get("sample_type")
    if st in (None, "", "unknown"):
        return [_verdict("sample_type", "sample_type", st, NOT_APPLICABLE,
                         "sample_type is null/unknown")]

    synonyms = _SAMPLE_TYPE_SYNONYMS.get(st)
    if not synonyms:
        return [_verdict("sample_type", "sample_type", st, NOT_APPLICABLE,
                         f"'{st}' has no synonym set (mixed/unknown or unrecognised enum value)")]

    lower_text = text.lower()
    for syn in synonyms:
        idx = lower_text.find(syn)
        if idx != -1:
            return [_verdict("sample_type", "sample_type", st, GROUNDED,
                             f"synonym '{syn}' found at char {idx}",
                             _snippet(text, idx, len(syn)),
                             matched_synonym=syn, char_offset=idx,
                             synonyms_checked=list(synonyms))]
    return [_verdict("sample_type", "sample_type", st, UNGROUNDED,
                     f"none of {list(synonyms)} appears in the source text",
                     synonyms_checked=list(synonyms))]


# ------------------------------------------------------------------ #
#  Per-record grading                                                 #
# ------------------------------------------------------------------ #

def _source_text_of(record: Dict[str, Any]) -> str:
    """
    Recover the text the extractor actually saw. Records written by this script
    carry `_source_text`; records from other prior-result JSON files fall back to
    whatever text field they do carry (abstract), which is only the true
    reference when source_basis == "abstract".
    """
    for key in ("_source_text", "source_text", "extraction_text"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    val = record.get("abstract")
    return val if isinstance(val, str) else ""


def grade_record(record: Dict[str, Any], auc_window: int,
                 strict_label: bool = False) -> Dict[str, Any]:
    text = _source_text_of(record)
    basis = record.get("source_basis") or "unknown"

    if not text:
        checks = [_verdict(f, f, None, NOT_APPLICABLE,
                           "no source text available on this record; cannot ground")
                  for f in _FIELDS]
        source_note = "MISSING"
    else:
        checks = (
            check_auc(record, text, auc_window, strict_label)
            + check_sample_sizes(record, text)
            + check_dataset_ids(record, text)
            + check_markers(record, text)
            + check_sample_type(record, text)
        )
        source_note = "OK"
        if record.get("_source_text") is None and basis == "fulltext":
            # Grading a full-text extraction against an abstract measures the
            # wrong reference; say so instead of reporting a bogus rate.
            source_note = "MISMATCH: source_basis=fulltext but only abstract available"

    counts = {
        "grounded": sum(1 for c in checks if c["verdict"] == GROUNDED),
        "ungrounded": sum(1 for c in checks if c["verdict"] == UNGROUNDED),
        "not_applicable": sum(1 for c in checks if c["verdict"] == NOT_APPLICABLE),
    }
    applicable = counts["grounded"] + counts["ungrounded"]
    counts["grounded_rate"] = round(counts["grounded"] / applicable, 4) if applicable else None

    return {
        "pmid": record.get("pmid", ""),
        "title": record.get("title", ""),
        "source_basis": basis,
        "source_chars": len(text),
        "source_text_status": source_note,
        "confidence_level": record.get("confidence_level"),
        "needs_human_review": record.get("needs_human_review"),
        "checks": checks,
        "counts": counts,
    }


# ------------------------------------------------------------------ #
#  Aggregation                                                        #
# ------------------------------------------------------------------ #

def _blank() -> Dict[str, Any]:
    return {"grounded": 0, "ungrounded": 0, "not_applicable": 0, "rate": None}


def _tally(bucket: Dict[str, Any], verdict: str) -> None:
    if verdict == GROUNDED:
        bucket["grounded"] += 1
    elif verdict == UNGROUNDED:
        bucket["ungrounded"] += 1
    else:
        bucket["not_applicable"] += 1


def _finalise(bucket: Dict[str, Any]) -> Dict[str, Any]:
    applicable = bucket["grounded"] + bucket["ungrounded"]
    bucket["applicable"] = applicable
    bucket["rate"] = round(bucket["grounded"] / applicable, 4) if applicable else None
    return bucket


def aggregate(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_field = {f: _blank() for f in _FIELDS}
    overall = _blank()
    by_basis: Dict[str, Dict[str, Any]] = {}
    dataset_context = {"PRIMARY": 0, "REFERENCE": 0, "ABSENT": 0}
    auc_divergences: List[Dict[str, Any]] = []
    auc_expanded_only: List[Dict[str, Any]] = []
    auc_loose: List[Dict[str, Any]] = []

    for rec in graded:
        basis = rec["source_basis"]
        by_basis.setdefault(basis, {"records": 0, "overall": _blank(),
                                    "per_field": {f: _blank() for f in _FIELDS}})
        by_basis[basis]["records"] += 1

        for check in rec["checks"]:
            field, verdict = check["field"], check["verdict"]
            _tally(per_field[field], verdict)
            _tally(overall, verdict)
            _tally(by_basis[basis]["per_field"][field], verdict)
            _tally(by_basis[basis]["overall"], verdict)

            if field == "dataset_ids" and check.get("context"):
                dataset_context[check["context"]] = dataset_context.get(check["context"], 0) + 1
            if field == "auc" and check.get("divergence"):
                auc_divergences.append({
                    "pmid": rec["pmid"], "key": check["key"], "value": check["value"],
                    "graded": verdict, "module_backstop": check.get("module_backstop"),
                })
            if field == "auc" and check.get("expanded_label_only"):
                auc_expanded_only.append({
                    "pmid": rec["pmid"], "key": check["key"], "value": check["value"],
                    "source_basis": basis, "evidence": check.get("evidence", ""),
                })
            if field == "auc" and check.get("loose_match") and verdict == GROUNDED:
                auc_loose.append({
                    "pmid": rec["pmid"], "key": check["key"], "value": check["value"],
                    "matched_form": check.get("matched_form"),
                })

    for f in _FIELDS:
        _finalise(per_field[f])
    _finalise(overall)
    for basis in by_basis:
        _finalise(by_basis[basis]["overall"])
        for f in _FIELDS:
            _finalise(by_basis[basis]["per_field"][f])

    return {
        "per_field": per_field,
        "overall": overall,
        "by_source_basis": by_basis,
        "dataset_context": dataset_context,
        "auc_window_divergences": auc_divergences,
        "auc_grounded_via_spelled_out_label_only": auc_expanded_only,
        "auc_grounded_via_rounded_form": auc_loose,
    }


# ------------------------------------------------------------------ #
#  Reporting                                                          #
# ------------------------------------------------------------------ #

def _pct(rate: Optional[float]) -> str:
    return "   n/a" if rate is None else f"{rate * 100:5.1f}%"


def print_report(graded: List[Dict[str, Any]], agg: Dict[str, Any], meta: Dict[str, Any]) -> None:
    line = "=" * 78
    print(f"\n{line}")
    print("GROUNDING EVALUATION — is each extracted field supported by the source text?")
    print(line)
    print(f"  mode              : {meta['mode']}")
    if meta.get("query"):
        print(f"  query             : {meta['query']}")
    if meta.get("source_json"):
        print(f"  source json       : {meta['source_json']}")
    print(f"  records graded    : {len(graded)}")
    print(f"  AUC label window  : +/-{meta['auc_window']} chars "
          f"(module backstop uses +/-{_AUC_UNSUPPORTED_WINDOW})")
    print(f"  evidence snippet  : +/-{_SNIPPET_CHARS} chars")

    print(f"\n{'-' * 78}\nPER-RECORD\n{'-' * 78}")
    print(f"{'PMID':<11}{'BASIS':<10}{'CHARS':>7}  {'GRD':>4}{'UNG':>4}{'N/A':>4}   {'RATE':>6}  TITLE")
    for rec in graded:
        c = rec["counts"]
        print(f"{rec['pmid']:<11}{rec['source_basis']:<10}{rec['source_chars']:>7}  "
              f"{c['grounded']:>4}{c['ungrounded']:>4}{c['not_applicable']:>4}   "
              f"{_pct(c['grounded_rate'])}  {rec['title'][:26]}")
        if rec["source_text_status"] not in ("OK",):
            print(f"{'':<11}  !! {rec['source_text_status']}")

    print(f"\n{'-' * 78}\nPER-FIELD GROUNDED RATE (all records)\n{'-' * 78}")
    print(f"{'FIELD':<20}{'GROUNDED':>9}{'UNGROUND':>9}{'N/A':>6}{'APPLIC':>8}{'RATE':>9}")
    for f in _FIELDS:
        b = agg["per_field"][f]
        print(f"{f:<20}{b['grounded']:>9}{b['ungrounded']:>9}{b['not_applicable']:>6}"
              f"{b['applicable']:>8}{_pct(b['rate']):>9}")
    o = agg["overall"]
    print("-" * 78)
    print(f"{'OVERALL':<20}{o['grounded']:>9}{o['ungrounded']:>9}{o['not_applicable']:>6}"
          f"{o['applicable']:>8}{_pct(o['rate']):>9}")

    print(f"\n{'-' * 78}\nBREAKDOWN BY source_basis  (did full text improve grounding?)\n{'-' * 78}")
    print(f"{'BASIS':<12}{'RECS':>5}{'GROUNDED':>10}{'UNGROUND':>10}{'APPLIC':>8}{'RATE':>9}")
    for basis in sorted(agg["by_source_basis"]):
        b = agg["by_source_basis"][basis]
        ov = b["overall"]
        print(f"{basis:<12}{b['records']:>5}{ov['grounded']:>10}{ov['ungrounded']:>10}"
              f"{ov['applicable']:>8}{_pct(ov['rate']):>9}")

    ft = agg["by_source_basis"].get("fulltext", {}).get("overall", {}).get("rate")
    ab = agg["by_source_basis"].get("abstract", {}).get("overall", {}).get("rate")
    if ft is not None and ab is not None:
        delta = (ft - ab) * 100
        verdict = "full text IMPROVED" if delta > 0 else ("full text DEGRADED" if delta < 0 else "no change")
        print(f"\n  delta (fulltext - abstract): {delta:+.1f} pp  ->  {verdict} grounding")
    else:
        print("\n  delta: not computable (only one source_basis present in this run)")

    if len(agg["by_source_basis"]) > 1:
        print(f"\n  per-field rate by basis:")
        header = "  " + f"{'FIELD':<20}" + "".join(f"{b:>12}" for b in sorted(agg["by_source_basis"]))
        print(header)
        for f in _FIELDS:
            row = "  " + f"{f:<20}"
            for basis in sorted(agg["by_source_basis"]):
                row += f"{_pct(agg['by_source_basis'][basis]['per_field'][f]['rate']):>12}"
            print(row)

    dc = agg["dataset_context"]
    print(f"\n{'-' * 78}\nDATASET ACCESSION CONTEXT\n{'-' * 78}")
    print(f"  PRIMARY (study data)       : {dc.get('PRIMARY', 0)}")
    print(f"  REFERENCE (bg/norm panel)  : {dc.get('REFERENCE', 0)}   <- present in text but "
          f"likely mis-attributed")
    print(f"  ABSENT (not in text)       : {dc.get('ABSENT', 0)}")

    if agg["auc_window_divergences"]:
        print(f"\n{'-' * 78}\nAUC DIVERGENCES vs _check_auc_unsupported() backstop\n{'-' * 78}")
        for d in agg["auc_window_divergences"]:
            print(f"  PMID {d['pmid']}  {d['key']}={d['value']}  "
                  f"graded={d['graded']}  module={d['module_backstop']}")
        print(f"  (graded window +/-{meta['auc_window']} chars with digit-boundary matching; "
              f"module uses +/-{_AUC_UNSUPPORTED_WINDOW} and first-candidate .find())")

    exp = agg["auc_grounded_via_spelled_out_label_only"]
    if exp:
        print(f"\n{'-' * 78}\nAUC LABELLED ONLY BY A SPELLED-OUT PHRASE ({len(exp)})\n{'-' * 78}")
        print("  _AUC_LABEL_RE misses these; they are real AUCs written out in prose.")
        print("  Actionable: extend _AUC_LABEL_RE in tools/extraction_reviewer.py before")
        print("  trusting it on full text. Re-run with --strict-auc-label to exclude them.")
        for d in exp:
            print(f"  PMID {d['pmid']}  {d['key']}={d['value']}  basis={d['source_basis']}")
            print(f"      {d['evidence'][:150]}")

    loose = agg["auc_grounded_via_rounded_form"]
    if loose:
        print(f"\n{'-' * 78}\nAUC GROUNDED VIA A ROUNDED FORM ({len(loose)}) — verify manually\n{'-' * 78}")
        for d in loose:
            print(f"  PMID {d['pmid']}  {d['key']}={d['value']}  matched as '{d['matched_form']}'")

    print(f"\n{'-' * 78}\nWORST UNGROUNDED FIELDS (top 10 by evidence)\n{'-' * 78}")
    shown = 0
    for rec in graded:
        for check in rec["checks"]:
            if check["verdict"] != UNGROUNDED or shown >= 10:
                continue
            print(f"  PMID {rec['pmid']:<10} {check['field']}.{check['key']} = {check['value']!r}")
            print(f"      {check['reason']}")
            shown += 1
    if shown == 0:
        print("  (none)")
    print(f"{line}\n")


# ------------------------------------------------------------------ #
#  Collection (network path)                                          #
# ------------------------------------------------------------------ #

def collect_records(query: str, top_n: int, review: bool, use_fulltext: bool) -> List[Dict[str, Any]]:
    """
    Reproduce stage2_extract()'s loop from its own primitives so the source text
    survives into the output. All NCBI traffic goes through the imported
    fetch_pubmed_records / _fetch_fulltext_safe, which carry the existing
    backoff, rate-limit delay and per-PMID full-text cache.
    """
    import yaml
    from utils.llm_factory import get_llm

    with open(_REPO_ROOT / "config" / "settings.yaml") as fh:
        cfg = yaml.safe_load(fh)
    llm = get_llm(cfg["llm"])

    intent = parse_query_rules(query)
    print(f"  intent: cancer={intent.get('cancer_type_code')} "
          f"sample={intent.get('sample_types')}")

    queries = build_pubmed_query_with_controls(intent)
    seen: set = set()
    raw_all: List[Dict[str, str]] = []
    for variant, q in queries.items():
        print(f"  [esearch] variant={variant}")
        for rec in fetch_pubmed_records(q, max_results=top_n):
            pmid = rec.get("pmid", "")
            if pmid and pmid not in seen:
                seen.add(pmid)
                raw_all.append(rec)
    print(f"  fetched {len(raw_all)} unique records")

    passed = stage1_filter(raw_all)
    print(f"  stage 1 passed: {len(passed)}")

    capped = passed[:top_n]
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(capped, 1):
        pmid = rec.get("pmid", "")
        full_text = _fetch_fulltext_safe(pmid) if use_fulltext else None
        if full_text:
            text = _prepare_fulltext_for_extraction(full_text)
            basis = "fulltext"
        else:
            text = rec.get("abstract", "")
            basis = "abstract"

        print(f"  [extract] {i}/{len(capped)} ({basis}) PMID={pmid} ...", flush=True)
        result = extract_paper_structured(abstract=text, llm=llm, pmid=pmid,
                                          title=rec.get("title", ""))
        result.setdefault("pmid", pmid)
        result.setdefault("title", rec.get("title", ""))
        result["source_basis"] = basis

        if review:
            print(f"  [review]  {i}/{len(capped)} PMID={pmid} ...", flush=True)
            result = review_extraction(abstract=text, draft_extraction=result, llm=llm)

        result["_source_text"] = text     # <- what makes --from-json network-free
        result["abstract"] = rec.get("abstract", "")
        out.append(result)
    return out


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Accept either this script's own output or a bare list/dict of records."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "extractions", "results"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    raise SystemExit(f"Could not find a record list in {path} "
                     f"(looked for a top-level list, or a 'records'/'extractions'/'results' key)")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Field-level grounding accuracy for MethyAgent extractions.")
    ap.add_argument("--query", help="Natural-language query; runs the live search+extract pipeline.")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Max PMIDs per query variant and max records extracted (default 10).")
    ap.add_argument("--from-json", dest="from_json",
                    help="Re-analyse a prior results JSON. No network is touched.")
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "data" / "eval"))
    ap.add_argument("--auc-window", type=int, default=_DEFAULT_AUC_WINDOW,
                    help=f"Chars around an AUC value to search for an AUC label "
                         f"(default {_DEFAULT_AUC_WINDOW}).")
    ap.add_argument("--strict-auc-label", action="store_true",
                    help="Grade AUC labels with _AUC_LABEL_RE only, rejecting spelled-out "
                         "'area under the receiver operating characteristic curve' phrasing.")
    ap.add_argument("--no-review", action="store_true",
                    help="Skip the extraction_reviewer second LLM call (live mode only).")
    ap.add_argument("--no-fulltext", action="store_true",
                    help="Force abstract-only extraction (live mode only).")
    ap.add_argument("--embed-source", dest="embed_source", action="store_true", default=True,
                    help="Embed _source_text in the output so --from-json can re-grade (default).")
    ap.add_argument("--no-embed-source", dest="embed_source", action="store_false",
                    help="Drop _source_text from the output (smaller file, breaks --from-json).")
    args = ap.parse_args()

    if bool(args.query) == bool(args.from_json):
        ap.error("provide exactly one of --query or --from-json")

    ts = time.strftime("%Y%m%d_%H%M%S")

    if args.from_json:
        src = Path(args.from_json)
        if not src.exists():
            raise SystemExit(f"No such file: {src}")
        print(f"[eval_grounding] re-analysing {src} (offline, no network)")
        records = load_records(src)
        mode = "from-json"
        proxy = "(not used — offline re-analysis)"
    else:
        proxy = resolve_proxy()
        print(f"[eval_grounding] proxy resolved: {proxy or '(NONE — expect NCBI rate-limit blocks)'}")
        print(f"[eval_grounding] module proxy:   {_resolve_proxy() or '(none)'}")
        if not proxy:
            print("  WARNING: no NCBI_PROXY / HTTPS_PROXY set. config/settings.yaml keeps "
                  "geo.proxy blank on purpose; export it in the shell before running.")
        records = collect_records(args.query, args.top_n,
                                  review=not args.no_review,
                                  use_fulltext=not args.no_fulltext)
        mode = "live"

    if not records:
        print("No records to grade.")
        return 1

    graded = [grade_record(r, args.auc_window, args.strict_auc_label) for r in records]
    agg = aggregate(graded)

    meta = {
        "timestamp": ts,
        "mode": mode,
        "query": args.query,
        "top_n": args.top_n if mode == "live" else None,
        "source_json": str(args.from_json) if args.from_json else None,
        "auc_window": args.auc_window,
        "module_auc_window": _AUC_UNSUPPORTED_WINDOW,
        "strict_auc_label": args.strict_auc_label,
        "snippet_chars": _SNIPPET_CHARS,
        "proxy_resolved": proxy,
        "n_records": len(records),
        "review_enabled": (not args.no_review) if mode == "live" else None,
        "fulltext_enabled": (not args.no_fulltext) if mode == "live" else None,
    }

    out_records = []
    for rec in records:
        copy = dict(rec)
        if not args.embed_source:
            copy.pop("_source_text", None)
        out_records.append(copy)

    payload = {
        "meta": meta,
        "aggregate": agg,
        "graded": graded,
        "records": out_records,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grounding_{ts}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print_report(graded, agg, meta)
    print(f"Wrote {out_path}")
    if args.embed_source:
        print(f"Re-analyse offline with:\n  python scripts/eval_grounding.py --from-json {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
