"""
Controlled-injection meta-evaluation of the two LLM reviewers.

    tools/extraction_reviewer.py  review_extraction()      (baseline, holistic)
    tools/checklist_reviewer.py   review_by_checklist()    (new, per-check)

WHAT THIS MEASURES
    We take REAL abstracts + REAL extractions, then inject exactly one known
    error per copy. Because we know what we broke, we know which check ought to
    fail — so we can measure, per check:

      * detection rate (recall) on the MUTATED arm — does the reviewer catch
        the error we planted?
      * FALSE POSITIVE rate on the CLEAN arm — does the reviewer flag/null a
        record we did not touch? This is the number that quantifies the
        over-nulling problem (valid AUCs nulled because the reviewer wandered
        into cohort attribution).
      * check-attribution accuracy — when it does fail something, is it the
        check that SHOULD have failed, or a different one?

    The baseline reviewer has no per-check verdicts, so its record-level flags
    are mapped onto check families (see _BASELINE_FLAG_FAMILIES). That mapping
    is many-to-one and is reported as such: an auc_unsupported flag cannot
    distinguish C1 from C2, and a sample_type_mismatch flag cannot distinguish
    C3 from C4. That imprecision IS the finding, not a measurement artifact.

INJECTED ERROR TYPES (one per mutated copy)
    auc_fabricated       an AUC value that does not occur in the abstract   -> expect C1 FAIL
    auc_mislabeled       a real non-AUC number relabelled as an AUC         -> expect C2 FAIL
    sample_type_swap     plasma_cfdna <-> tissue, against the text          -> expect C4 FAIL
    reference_accession  an accession injected in background-panel context  -> expect C5 FAIL
    (clean)              unmodified copy                                    -> expect NO FAIL

USAGE
    python scripts/eval_reviewer.py --base <records.json> --n-per-type 5 [--model <name>]

    --base accepts:
      * a saved pipeline output JSON (list of records, or a dict with
        records/papers/results/items holding them). Each record needs an
        abstract/source text plus extracted fields.
      * the literal "gold" — builds base records from scripts/gold_standard.py
        GOLD_STANDARD by fetching abstracts through tools.ncbi_search
        (proxy-aware, backoff/caching already in that module) and running
        extract_paper_structured() once per PMID. The result is CACHED to
        data/eval/base_records_from_gold.json; pass that file to --base
        afterwards so re-analysis never re-hits NCBI.

    Offline harness check (no API key, no network):
      python scripts/eval_reviewer.py --base gold --self-test

OUTPUTS
    data/eval/reviewer_eval_<ts>.json    full results + metrics + comparison
    data/eval/reviewer_audit_<ts>.csv    ONE ROW PER CHECK VERDICT for human
                                         audit: pmid, check_id, verdict,
                                         evidence, model_note, ground_truth,
                                         and an empty human_judgment column
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.checklist_reviewer import (  # noqa: E402
    C1_AUC_PRESENT,
    C2_AUC_TERMINOLOGY,
    C3_AUC_SAMPLE_MATCH,
    C4_SAMPLE_TYPE_CORRECT,
    C5_DATASET_PROVENANCE,
    C6_SAMPLE_SIZE_PRESENT,
    C7_MARKER_PRESENT,
    CHECK_FAMILIES,
    check_family,
    review_by_checklist,
    _value_in_text,
)
from tools.extraction_reviewer import _AUC_LABEL_RE, review_extraction  # noqa: E402

_REPO_ROOT = Path(__file__).parent.parent
_OUT_DIR = _REPO_ROOT / "data" / "eval"
_GOLD_CACHE = _OUT_DIR / "base_records_from_gold.json"
_AUC_KEYS = ("auc_training", "auc_validation", "auc_external")

# Mutation type -> the check family that SHOULD fail for it.
_EXPECTED_CHECK = {
    "auc_fabricated": C1_AUC_PRESENT,
    "auc_mislabeled": C2_AUC_TERMINOLOGY,
    "sample_type_swap": C4_SAMPLE_TYPE_CORRECT,
    "reference_accession": C5_DATASET_PROVENANCE,
}
_MUTATION_TYPES = tuple(_EXPECTED_CHECK.keys())

# The baseline reviewer emits record-level flags, not per-check verdicts.
# This maps each flag onto the check families it could correspond to. The
# mapping is deliberately many-to-one — that ambiguity is the point.
_BASELINE_FLAG_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "auc_unsupported": (C1_AUC_PRESENT, C2_AUC_TERMINOLOGY),
    "sample_type_mismatch": (C3_AUC_SAMPLE_MATCH, C4_SAMPLE_TYPE_CORRECT),
    "reference_accession": (C5_DATASET_PROVENANCE,),
    "reviewer_error": (),
}


# ------------------------------------------------------------------ #
#  Proxy / config                                                     #
# ------------------------------------------------------------------ #

def _load_cfg() -> Dict[str, Any]:
    with open(_REPO_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _resolve_proxy(cfg: Dict[str, Any]) -> str:
    """
    PROXY RULE: config/settings.yaml keeps geo.proxy intentionally BLANK; the
    real value is per-machine in the environment. Reading cfg["geo"]["proxy"]
    alone runs unproxied and gets rate-limit-blocked by NCBI.
    """
    return (
        os.environ.get("NCBI_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or (cfg.get("geo") or {}).get("proxy")
        or ""
    )


def _ensure_proxy_env(cfg: Dict[str, Any]) -> str:
    """Export the resolved proxy as NCBI_PROXY so tools.ncbi_search picks it up."""
    proxy = _resolve_proxy(cfg)
    if proxy and not os.environ.get("NCBI_PROXY"):
        os.environ["NCBI_PROXY"] = proxy
    return proxy


def _load_llm(cfg: Dict[str, Any], model: Optional[str]):
    from utils.llm_factory import get_llm

    llm_cfg = dict(cfg["llm"])
    if model:
        llm_cfg["model"] = model
        # env wins over config inside get_llm(), so set the env names too
        for env_name in ("ZHIPU_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL", "ANTHROPIC_MODEL"):
            os.environ[env_name] = model
    # Both reviewers parse strict JSON; ask the backend for it natively.
    return get_llm(llm_cfg, json_mode=True)


# ------------------------------------------------------------------ #
#  Base record loading                                                #
# ------------------------------------------------------------------ #

_TEXT_KEYS = ("source_text", "abstract", "paper_abstract", "abstract_text", "text")
_LIST_KEYS = ("records", "papers", "results", "extractions", "items", "data")


def _looks_like_extraction(d: Any) -> bool:
    return isinstance(d, dict) and any(
        k in d for k in ("performance_metrics", "sample_type", "dataset_ids", "markers_or_panel")
    )


def _find_text(d: Dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for nest_key in ("paper", "source", "record"):
        nested = d.get(nest_key)
        if isinstance(nested, dict):
            text = _find_text(nested)
            if text:
                return text
    return ""


def _candidate_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(obj, dict):
        for key in _LIST_KEYS:
            if isinstance(obj.get(key), list):
                yield from _candidate_dicts(obj[key])
                return
        if _looks_like_extraction(obj) or _find_text(obj):
            yield obj


def load_base_records(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Normalize a saved pipeline output JSON into [{pmid, source_text, extraction}]."""
    with open(path) as f:
        raw = json.load(f)

    out: List[Dict[str, Any]] = []
    for cand in _candidate_dicts(raw):
        text = _find_text(cand)
        extraction = cand.get("extraction") if _looks_like_extraction(cand.get("extraction")) else None
        if extraction is None and _looks_like_extraction(cand):
            extraction = {k: v for k, v in cand.items() if k not in _TEXT_KEYS}
        if not text or extraction is None:
            continue
        pmid = str(cand.get("pmid") or extraction.get("pmid") or f"rec{len(out)}")
        out.append({"pmid": pmid, "source_text": text, "extraction": copy.deepcopy(extraction)})
        if limit and len(out) >= limit:
            break

    if not out:
        raise SystemExit(
            f"No usable base records in {path}. Each record needs an abstract/source text "
            f"(one of {_TEXT_KEYS}) plus extracted fields (performance_metrics / sample_type / "
            f"dataset_ids)."
        )
    return out


def build_base_records_from_gold(llm, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Build base records from scripts/gold_standard.py GOLD_STANDARD.

    Fetches abstracts through tools.ncbi_search.efetch_abstracts() (which owns
    the proxy resolution, retry/backoff and batching — we never call NCBI
    endpoints directly here) and runs extract_paper_structured() once per PMID.
    The result is cached to data/eval/base_records_from_gold.json so subsequent
    runs can pass --base <that file> and never touch the network again.
    """
    from scripts.gold_standard import GOLD_STANDARD
    from tools.ncbi_search import efetch_abstracts
    from tools.query_clarifier import extract_paper_structured

    pmids = [g["pmid"] for g in GOLD_STANDARD if g.get("pmid")]
    if limit:
        pmids = pmids[:limit]
    print(f"[base] fetching {len(pmids)} abstracts from PubMed (proxied, batched)...")
    fetched = efetch_abstracts(pmids)
    print(f"[base] got {len(fetched)} abstracts; extracting...")

    records: List[Dict[str, Any]] = []
    for rec in fetched:
        abstract = (rec.get("abstract") or "").strip()
        if not abstract:
            continue
        extraction = extract_paper_structured(
            abstract=abstract, llm=llm, pmid=rec.get("pmid", ""), title=rec.get("title", "")
        )
        records.append({
            "pmid": str(rec.get("pmid") or ""),
            "source_text": abstract,
            "extraction": extraction,
        })

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_GOLD_CACHE, "w") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)
    print(f"[base] cached {len(records)} base records -> {_GOLD_CACHE}")
    print(f"[base] re-run with --base {_GOLD_CACHE} to skip fetching/extracting entirely.")
    return records


# ------------------------------------------------------------------ #
#  Mutators — exactly one injected error per copy                     #
# ------------------------------------------------------------------ #

_FABRICATION_POOL = (0.837, 0.914, 0.762, 0.688, 0.953, 0.719, 0.884, 0.641, 0.976, 0.803)
_CFDNA_TERMS = ("cfdna", "cell-free dna", "cell free dna", "ctdna", "plasma", "serum", "circulating")
_TISSUE_TERMS = ("tissue", "tumour", "tumor", "biopsy", "ffpe", "resection", "surgical specimen")


def _first_auc_key(extraction: Dict[str, Any]) -> Optional[str]:
    metrics = extraction.get("performance_metrics") or {}
    for key in _AUC_KEYS:
        if metrics.get(key) is not None:
            return key
    return None


def _ensure_metrics(extraction: Dict[str, Any]) -> Dict[str, Any]:
    metrics = extraction.get("performance_metrics")
    if not isinstance(metrics, dict):
        metrics = {k: None for k in _AUC_KEYS}
        extraction["performance_metrics"] = metrics
    return metrics


def mutate_auc_fabricated(text: str, extraction: Dict[str, Any], idx: int):
    """Put an AUC value into the record that does not occur in the abstract."""
    metrics = _ensure_metrics(extraction)
    key = _first_auc_key(extraction) or "auc_validation"
    for offset in range(len(_FABRICATION_POOL)):
        value = _FABRICATION_POOL[(idx + offset) % len(_FABRICATION_POOL)]
        if _value_in_text(text, value) is None:
            metrics[key] = value
            return text, extraction, f"set {key}={value} (absent from abstract)", key
    return None


_PCT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d)?)%")
_DEC_RE = re.compile(r"(?<![\d.])(0\.\d{2,3})(?![\d])")


def _unlabelled_numbers(text: str) -> List[Tuple[float, str]]:
    """
    Numbers in the abstract that are NOT described as an AUC anywhere near their
    occurrence — i.e. exactly the kind of value C2 exists to catch when it gets
    relabelled as an AUC. Returns [(value_as_fraction, matched_text)].
    """
    found: List[Tuple[float, str]] = []
    for regex, scale in ((_DEC_RE, 1.0), (_PCT_RE, 0.01)):
        for m in regex.finditer(text):
            window = text[max(0, m.start() - 150): m.end() + 150]
            if _AUC_LABEL_RE.search(window):
                continue          # it IS called an AUC nearby — not usable
            try:
                value = round(float(m.group(1)) * scale, 4)
            except ValueError:
                continue
            if not (0.0 < value <= 1.0):
                continue
            if _value_in_text(text, value) is None:
                continue          # must still be locatable verbatim, or C1 muddies C2
            found.append((value, m.group(0)))
    return found


def mutate_auc_mislabeled(text: str, extraction: Dict[str, Any], idx: int):
    """Relabel a real non-AUC number (sensitivity, HR, beta, ...) as an AUC."""
    candidates = _unlabelled_numbers(text)
    if not candidates:
        return None
    value, matched = candidates[idx % len(candidates)]
    metrics = _ensure_metrics(extraction)
    key = _first_auc_key(extraction) or "auc_validation"
    metrics[key] = value
    return text, extraction, f"set {key}={value} from non-AUC text {matched!r}", key


def mutate_sample_type_swap(text: str, extraction: Dict[str, Any], idx: int):
    """Flip the declared sample_type so it contradicts the abstract."""
    lowered = text.lower()
    has_cfdna = any(t in lowered for t in _CFDNA_TERMS)
    has_tissue = any(t in lowered for t in _TISSUE_TERMS)
    current = (extraction.get("sample_type") or "").lower()

    if current in ("plasma_cfdna", "serum_cfdna"):
        new = "tissue"
    elif current == "tissue":
        new = "plasma_cfdna"
    elif has_cfdna and not has_tissue:
        new = "tissue"
    elif has_tissue and not has_cfdna:
        new = "plasma_cfdna"
    else:
        return None      # text supports neither cleanly — a swap wouldn't be ground truth
    if new == current:
        return None
    extraction["sample_type"] = new
    return text, extraction, f"sample_type {current or 'unknown'!r} -> {new!r} (contradicts text)", None


_REFERENCE_SENTENCE = (
    " Probes were additionally filtered by retaining only CpG sites with average beta values "
    ">0.90 or <0.10 in whole blood samples from {acc} (n = 233); {acc} was used solely as a "
    "background-methylation reference panel to minimise interference and was not analysed as a "
    "study cohort."
)


def mutate_reference_accession(text: str, extraction: Dict[str, Any], idx: int):
    """Add an accession to the record that the text presents as a background panel only."""
    for offset in range(20):
        acc = f"GSE{910000 + idx * 7 + offset}"
        if acc not in text and acc not in (extraction.get("dataset_ids") or []):
            break
    else:
        return None
    new_text = text.rstrip() + _REFERENCE_SENTENCE.format(acc=acc)
    ids = list(extraction.get("dataset_ids") or [])
    ids.append(acc)
    extraction["dataset_ids"] = ids
    return new_text, extraction, f"added {acc} in background-panel context", acc


_MUTATORS = {
    "auc_fabricated": mutate_auc_fabricated,
    "auc_mislabeled": mutate_auc_mislabeled,
    "sample_type_swap": mutate_sample_type_swap,
    "reference_accession": mutate_reference_accession,
}


def build_eval_set(base: List[Dict[str, Any]], n_per_type: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build the mutated + clean arms. Base records are cycled so that, where the
    corpus allows it, the N copies of a mutation type land on N different papers.
    """
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for mut_type in _MUTATION_TYPES:
        made = 0
        attempts = 0
        while made < n_per_type and attempts < len(base) * 3:
            src = base[attempts % len(base)]
            attempts += 1
            text = src["source_text"]
            extraction = copy.deepcopy(src["extraction"])
            result = _MUTATORS[mut_type](text, extraction, made)
            if result is None:
                skipped.append(f"{mut_type} not applicable to PMID {src['pmid']}")
                continue
            new_text, new_extraction, note, target = result
            items.append({
                "record_id": f"{mut_type}#{made}",
                "arm": "mutated",
                "mutation_type": mut_type,
                "expected_check": _EXPECTED_CHECK[mut_type],
                "expected_target": target,
                "pmid": src["pmid"],
                "source_text": new_text,
                "extraction": new_extraction,
                "injection_note": note,
            })
            made += 1
        if made < n_per_type:
            skipped.append(f"{mut_type}: only {made}/{n_per_type} copies could be built")

    for i in range(n_per_type):
        src = base[i % len(base)]
        items.append({
            "record_id": f"clean#{i}",
            "arm": "clean",
            "mutation_type": "clean",
            "expected_check": None,
            "expected_target": None,
            "pmid": src["pmid"],
            "source_text": src["source_text"],
            "extraction": copy.deepcopy(src["extraction"]),
            "injection_note": "unmodified",
        })

    return items, skipped


# ------------------------------------------------------------------ #
#  Running the two reviewers                                          #
# ------------------------------------------------------------------ #

def _auc_nulled(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    pre = (before.get("performance_metrics") or {})
    post = (after.get("performance_metrics") or {})
    return [k for k in _AUC_KEYS if pre.get(k) is not None and post.get(k) is None]


def _datasets_dropped(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    pre = set(before.get("dataset_ids") or [])
    post = set(after.get("dataset_ids") or [])
    return sorted(pre - post)


def run_checklist(item: Dict[str, Any], llm) -> Dict[str, Any]:
    before = copy.deepcopy(item["extraction"])
    t0 = time.time()
    reviewed = review_by_checklist(item["source_text"], copy.deepcopy(before), llm)
    checks = reviewed.get("checklist") or []
    report = reviewed.get("review_report") or {}
    failed = sorted({check_family(c["check_id"]) for c in checks if c["verdict"] == "FAIL"})
    return {
        "reviewer": "checklist",
        "elapsed_s": round(time.time() - t0, 2),
        "checks": checks,
        "failed_check_ids": [c["check_id"] for c in checks if c["verdict"] == "FAIL"],
        "flags": report.get("flags") or [],
        "risk_level": report.get("risk_level"),
        "needs_human_review": bool(report.get("needs_human_review")),
        "reason": report.get("reason"),
        "failed_families": failed,
        "auc_nulled": _auc_nulled(before, reviewed),
        "datasets_dropped": _datasets_dropped(before, reviewed),
        "error": "reviewer_error" in (report.get("flags") or []),
    }


def run_baseline(item: Dict[str, Any], llm) -> Dict[str, Any]:
    before = copy.deepcopy(item["extraction"])
    t0 = time.time()
    reviewed = review_extraction(item["source_text"], copy.deepcopy(before), llm)
    report = reviewed.get("review_report") or {}
    flags = report.get("flags") or []
    failed: set = set()
    for flag in flags:
        failed.update(_BASELINE_FLAG_FAMILIES.get(flag, ()))
    return {
        "reviewer": "baseline",
        "elapsed_s": round(time.time() - t0, 2),
        "checks": [],           # baseline renders no per-check verdicts
        "failed_check_ids": [],  # ...and therefore names no target either
        "flags": flags,
        "risk_level": report.get("risk_level"),
        "needs_human_review": bool(report.get("needs_human_review")),
        "reason": report.get("reason"),
        "failed_families": sorted(failed),
        "auc_nulled": _auc_nulled(before, reviewed),
        "datasets_dropped": _datasets_dropped(before, reviewed),
        "error": "reviewer_error" in flags,
    }


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #

def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 3) if den else None


def compute_metrics(results: List[Dict[str, Any]], reviewer: str) -> Dict[str, Any]:
    rows = [r for r in results if r["reviewer"] == reviewer]
    mutated = [r for r in rows if r["arm"] == "mutated"]
    clean = [r for r in rows if r["arm"] == "clean"]

    # --- detection rate (recall), per expected check ---
    recall: Dict[str, Any] = {}
    for mut_type, family in _EXPECTED_CHECK.items():
        subset = [r for r in mutated if r["mutation_type"] == mut_type]
        hits = [r for r in subset if family in r["failed_families"]]
        recall[family] = {
            "mutation_type": mut_type,
            "n": len(subset),
            "detected": len(hits),
            "recall": _rate(len(hits), len(subset)),
        }

    # --- false positive rate on the clean arm, per check family ---
    false_positives: Dict[str, Any] = {}
    for family in CHECK_FAMILIES:
        fp = [r for r in clean if family in r["failed_families"]]
        false_positives[family] = {
            "n_clean": len(clean),
            "false_fails": len(fp),
            "fp_rate": _rate(len(fp), len(clean)),
        }

    clean_any_fail = [r for r in clean if r["failed_families"]]
    clean_nulled = [r for r in clean if r["auc_nulled"]]
    clean_dropped = [r for r in clean if r["datasets_dropped"]]

    # --- check-attribution accuracy on the mutated arm ---
    flagged = [r for r in mutated if r["failed_families"]]
    hit = [r for r in flagged if r["expected_check"] in r["failed_families"]]
    exact = [r for r in flagged if r["failed_families"] == [r["expected_check"]]]
    collateral = [
        r for r in mutated
        if [f for f in r["failed_families"] if f != r["expected_check"]]
    ]

    # --- target-level attribution: right check, right *item*? ---
    # Only meaningful for a reviewer that names a target (checklist). The
    # baseline flags the record, not the AUC key or the accession, so this is
    # reported as null for it.
    targeted = [r for r in mutated if r["expected_target"] and r["reviewer"] == "checklist"]
    target_hit = [
        r for r in targeted
        if any(cid == f"{r['expected_check']}:{r['expected_target']}" for cid in r["failed_check_ids"])
    ]
    target_wrong = [
        r for r in targeted
        if any(
            check_family(cid) == r["expected_check"]
            and cid != f"{r['expected_check']}:{r['expected_target']}"
            for cid in r["failed_check_ids"]
        )
    ]

    return {
        "reviewer": reviewer,
        "n_mutated": len(mutated),
        "n_clean": len(clean),
        "reviewer_errors": sum(1 for r in rows if r["error"]),
        "per_check_recall": recall,
        "overall_recall": _rate(
            sum(1 for r in mutated if r["expected_check"] in r["failed_families"]), len(mutated)
        ),
        "per_check_false_positive_rate": false_positives,
        "clean_arm": {
            "any_check_failed_rate": _rate(len(clean_any_fail), len(clean)),
            "auc_nulled_rate": _rate(len(clean_nulled), len(clean)),          # <- over-nulling
            "dataset_dropped_rate": _rate(len(clean_dropped), len(clean)),
            "needs_human_review_rate": _rate(
                sum(1 for r in clean if r["needs_human_review"]), len(clean)
            ),
        },
        "attribution": {
            "n_flagged": len(flagged),
            "hit": len(hit),
            "hit_rate": _rate(len(hit), len(flagged)),
            "exact": len(exact),
            "exact_rate": _rate(len(exact), len(flagged)),
            "collateral_failures": len(collateral),
            "collateral_rate": _rate(len(collateral), len(mutated)),
            "n_targeted": len(targeted),
            "target_hit": len(target_hit),
            "target_hit_rate": _rate(len(target_hit), len(targeted)),
            "wrong_target_same_check": len(target_wrong),
            "wrong_target_rate": _rate(len(target_wrong), len(targeted)),
        },
        "mean_latency_s": round(sum(r["elapsed_s"] for r in rows) / len(rows), 2) if rows else None,
    }


def print_comparison(metrics_by_reviewer: Dict[str, Dict[str, Any]]) -> None:
    base = metrics_by_reviewer.get("baseline")
    chk = metrics_by_reviewer.get("checklist")
    print("\n" + "=" * 88)
    print("SIDE-BY-SIDE:  baseline review_extraction()   vs   new review_by_checklist()")
    print("=" * 88)

    def cell(m, path, default="—"):
        if m is None:
            return default
        cur: Any = m
        for key in path:
            cur = (cur or {}).get(key) if isinstance(cur, dict) else None
        return default if cur is None else f"{cur}"

    print(f"\n{'metric':52s} {'baseline':>15s} {'checklist':>15s}")
    print("-" * 88)
    for mut_type, family in _EXPECTED_CHECK.items():
        print(f"{'recall  ' + family + '  (' + mut_type + ')':52s} "
              f"{cell(base, ['per_check_recall', family, 'recall']):>15s} "
              f"{cell(chk, ['per_check_recall', family, 'recall']):>15s}")
    print(f"{'recall  OVERALL (mutated arm)':52s} "
          f"{cell(base, ['overall_recall']):>15s} {cell(chk, ['overall_recall']):>15s}")
    print("-" * 88)
    for family in CHECK_FAMILIES:
        print(f"{'false-positive rate (clean arm)  ' + family:52s} "
              f"{cell(base, ['per_check_false_positive_rate', family, 'fp_rate']):>15s} "
              f"{cell(chk, ['per_check_false_positive_rate', family, 'fp_rate']):>15s}")
    print(f"{'clean arm: ANY check failed':52s} "
          f"{cell(base, ['clean_arm', 'any_check_failed_rate']):>15s} "
          f"{cell(chk, ['clean_arm', 'any_check_failed_rate']):>15s}")
    print(f"{'clean arm: valid AUC nulled  <-- over-nulling':52s} "
          f"{cell(base, ['clean_arm', 'auc_nulled_rate']):>15s} "
          f"{cell(chk, ['clean_arm', 'auc_nulled_rate']):>15s}")
    print(f"{'clean arm: dataset dropped':52s} "
          f"{cell(base, ['clean_arm', 'dataset_dropped_rate']):>15s} "
          f"{cell(chk, ['clean_arm', 'dataset_dropped_rate']):>15s}")
    print("-" * 88)
    print(f"{'attribution hit rate (expected check failed)':52s} "
          f"{cell(base, ['attribution', 'hit_rate']):>15s} {cell(chk, ['attribution', 'hit_rate']):>15s}")
    print(f"{'attribution exact rate (ONLY expected failed)':52s} "
          f"{cell(base, ['attribution', 'exact_rate']):>15s} {cell(chk, ['attribution', 'exact_rate']):>15s}")
    print(f"{'target hit rate (right check AND right item)':52s} "
          f"{cell(base, ['attribution', 'target_hit_rate']):>15s} "
          f"{cell(chk, ['attribution', 'target_hit_rate']):>15s}")
    print(f"{'wrong-target rate (right check, wrong item)':52s} "
          f"{cell(base, ['attribution', 'wrong_target_rate']):>15s} "
          f"{cell(chk, ['attribution', 'wrong_target_rate']):>15s}")
    print(f"{'collateral failure rate (mutated arm)':52s} "
          f"{cell(base, ['attribution', 'collateral_rate']):>15s} "
          f"{cell(chk, ['attribution', 'collateral_rate']):>15s}")
    print(f"{'reviewer errors':52s} "
          f"{cell(base, ['reviewer_errors']):>15s} {cell(chk, ['reviewer_errors']):>15s}")
    print(f"{'mean latency (s/record)':52s} "
          f"{cell(base, ['mean_latency_s']):>15s} {cell(chk, ['mean_latency_s']):>15s}")
    print("\nNOTE: the baseline renders no per-check verdicts. Its record-level flags are mapped")
    print("onto check families (auc_unsupported -> C1+C2, sample_type_mismatch -> C3+C4,")
    print("reference_accession -> C5), so its per-check numbers are inherently coarse: it cannot")
    print("tell C1 from C2 or C3 from C4. That is exactly what the checklist restructuring fixes.")


# ------------------------------------------------------------------ #
#  Audit worksheet (one row per check verdict)                        #
# ------------------------------------------------------------------ #

_CSV_FIELDS = [
    "reviewer", "arm", "mutation_type", "record_id", "pmid", "injection_note",
    "check_id", "verdict", "confidence", "evidence", "model_note",
    "ground_truth", "agrees_with_ground_truth", "human_judgment", "human_comment",
]


def build_audit_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        common = {
            "reviewer": r["reviewer"], "arm": r["arm"], "mutation_type": r["mutation_type"],
            "record_id": r["record_id"], "pmid": r["pmid"], "injection_note": r["injection_note"],
        }
        if r["reviewer"] == "checklist":
            for check in r["checks"]:
                family = check_family(check["check_id"])
                should_fail = (r["expected_check"] == family)
                gt = "SHOULD_FAIL" if should_fail else "SHOULD_NOT_FAIL"
                agrees = (check["verdict"] == "FAIL") == should_fail
                rows.append({
                    **common,
                    "check_id": check["check_id"],
                    "verdict": check["verdict"],
                    "confidence": check["confidence"],
                    "evidence": check["evidence"] or "",
                    "model_note": check["note"],
                    "ground_truth": gt,
                    "agrees_with_ground_truth": "yes" if agrees else "NO",
                    "human_judgment": "",
                    "human_comment": "",
                })
        else:
            # The baseline has no per-check verdicts; emit one row per flag it
            # raised (plus a single row when it raised none) so a human can
            # audit the same records side by side.
            if r["flags"]:
                for flag in r["flags"]:
                    families = _BASELINE_FLAG_FAMILIES.get(flag, ())
                    should_fail = r["expected_check"] in families
                    rows.append({
                        **common,
                        "check_id": f"baseline_flag:{flag} (maps to {'+'.join(families) or 'n/a'})",
                        "verdict": "FAIL",
                        "confidence": "n/a",
                        "evidence": "",
                        "model_note": r["reason"] or "",
                        "ground_truth": "SHOULD_FAIL" if should_fail else "SHOULD_NOT_FAIL",
                        "agrees_with_ground_truth": "yes" if should_fail else "NO",
                        "human_judgment": "",
                        "human_comment": "",
                    })
            else:
                should_fail = r["expected_check"] is not None
                rows.append({
                    **common,
                    "check_id": "baseline_flag:(none)",
                    "verdict": "PASS",
                    "confidence": "n/a",
                    "evidence": "",
                    "model_note": r["reason"] or "",
                    "ground_truth": "SHOULD_FAIL" if should_fail else "SHOULD_NOT_FAIL",
                    "agrees_with_ground_truth": "NO" if should_fail else "yes",
                    "human_judgment": "",
                    "human_comment": "",
                })
    return rows


# ------------------------------------------------------------------ #
#  Offline self-test stub                                             #
# ------------------------------------------------------------------ #

class _SelfTestLLM:
    """
    Scripted model for --self-test: exercises the whole harness (mutation,
    scoring, CSV/JSON emission) with no API key and no network. It answers the
    checklist honestly from string matching, and answers the baseline reviewer
    with a deliberately over-eager response (nulls every AUC) so the clean-arm
    false-positive machinery is visibly exercised.
    """

    def invoke(self, messages):
        system = messages[0].content
        body = messages[1].content

        class _Resp:
            content = ""

        if "CHECKLIST" in system:
            planned = json.loads(body.split("CHECKS TO RENDER")[1].split("\n", 1)[1])
            source = body.split("SOURCE TEXT:\n", 1)[1].split("\n\nEXTRACTED RECORD", 1)[0]
            checks = []
            for plan in planned:
                cid = plan["check_id"]
                subject = plan.get("subject", "")
                verdict, note = "PASS", "string-matched"
                if cid.startswith("C5_dataset_provenance:"):
                    acc = cid.split(":", 1)[1]
                    pos = source.find(acc)
                    window = source[max(0, pos - 200): pos + 250] if pos != -1 else ""
                    is_reference = "reference panel" in window or "background" in window
                    verdict = "FAIL" if is_reference else "PASS"
                    note = "background panel only" if is_reference else "analysis or unstated role"
                elif cid.startswith("C2_auc_terminology:"):
                    raw_value = subject.split("=")[-1].strip()
                    # locate the value in whichever textual form it appears
                    # (0.878 may be written "87.8%") — same helper the real
                    # reviewer's C1 backstop uses
                    try:
                        matched = _value_in_text(source, float(raw_value))
                    except ValueError:
                        matched = None
                    idx = source.find(matched) if matched else -1
                    if idx == -1:
                        verdict, note = "NOT_APPLICABLE", "value not locatable"
                    else:
                        window = source[max(0, idx - 150): idx + len(matched) + 150]
                        labelled = bool(_AUC_LABEL_RE.search(window))
                        verdict = "PASS" if labelled else "FAIL"
                        note = "called an AUC" if labelled else "not termed an AUC in the text"
                elif cid == "C4_sample_type_correct":
                    declared = subject.split("=", 1)[1].strip().strip("'\"")
                    low = source.lower()
                    cf = any(t in low for t in _CFDNA_TERMS)
                    ts = any(t in low for t in _TISSUE_TERMS)
                    if declared in ("plasma_cfdna", "serum_cfdna") and ts and not cf:
                        verdict, note = "FAIL", "text describes tissue"
                    elif declared == "tissue" and cf and not ts:
                        verdict, note = "FAIL", "text describes cfDNA"
                    else:
                        verdict, note = "PASS", "consistent with the text"
                checks.append({"check_id": cid, "verdict": verdict, "evidence": None,
                               "confidence": "high", "note": note})
            _Resp.content = json.dumps({"checks": checks})
        else:
            payload = json.loads(body.split("Draft extraction (fields under review only):\n", 1)[1])
            _Resp.content = json.dumps({
                "performance_metrics": {k: None for k in _AUC_KEYS},
                "dataset_ids": payload.get("dataset_ids") or [],
                "needs_human_review": True,
                "reason": "self-test stub: nulls every AUC (simulates the over-nulling baseline)",
            })
        return _Resp()


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled-injection meta-evaluation of the extraction reviewers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", required=True,
                        help='Saved pipeline output JSON, or "gold" to build base records '
                             'from scripts/gold_standard.py (cached to data/eval/).')
    parser.add_argument("--n-per-type", type=int, default=5,
                        help="Copies per mutation type, and per clean arm (default 5).")
    parser.add_argument("--model", default=None, help="Override the LLM model name.")
    parser.add_argument("--limit-base", type=int, default=None,
                        help="Use at most this many base records.")
    parser.add_argument("--reviewers", default="both", choices=("both", "checklist", "baseline"))
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    parser.add_argument("--self-test", action="store_true",
                        help="Run the harness with a scripted stub LLM (no API key, no network).")
    args = parser.parse_args()

    cfg = _load_cfg()
    proxy = _ensure_proxy_env(cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        llm = _SelfTestLLM()
        backend, model_name = "self-test-stub", "self-test-stub"
    else:
        llm = _load_llm(cfg, args.model)
        backend = cfg["llm"].get("backend")
        model_name = args.model or os.environ.get("ZHIPU_MODEL") or cfg["llm"].get("model") or "?"

    print(f"[setup] backend={backend} model={model_name} proxy={proxy or '(none)'}")

    # ---- base records ----
    if args.base == "gold":
        if args.self_test and not _GOLD_CACHE.exists():
            raise SystemExit(
                "--self-test with --base gold needs a cached base file. Run once with a real "
                f"LLM/network first, or pass --base <records.json>. (looked for {_GOLD_CACHE})"
            )
        if _GOLD_CACHE.exists():
            print(f"[base] reusing cached base records from {_GOLD_CACHE} (no network)")
            base = load_base_records(_GOLD_CACHE, args.limit_base)
        else:
            base = build_base_records_from_gold(llm, args.limit_base)
    else:
        base = load_base_records(Path(args.base), args.limit_base)
    print(f"[base] {len(base)} base records: {', '.join(r['pmid'] for r in base)}")

    # ---- build arms ----
    items, skipped = build_eval_set(base, args.n_per_type)
    n_mut = sum(1 for i in items if i["arm"] == "mutated")
    n_clean = sum(1 for i in items if i["arm"] == "clean")
    print(f"[eval] {n_mut} mutated + {n_clean} clean records")
    for note in skipped:
        print(f"  [skip] {note}")

    reviewers = ["baseline", "checklist"] if args.reviewers == "both" else [args.reviewers]

    # ---- run ----
    results: List[Dict[str, Any]] = []
    total = len(items) * len(reviewers)
    done = 0
    for item in items:
        for reviewer in reviewers:
            done += 1
            runner = run_checklist if reviewer == "checklist" else run_baseline
            try:
                outcome = runner(item, llm)
            except Exception as e:                       # never lose a whole run to one record
                outcome = {
                    "reviewer": reviewer, "elapsed_s": 0.0, "checks": [], "failed_check_ids": [],
                    "flags": ["reviewer_error"],
                    "risk_level": "medium", "needs_human_review": True,
                    "reason": f"harness caught {type(e).__name__}: {e}",
                    "failed_families": [], "auc_nulled": [], "datasets_dropped": [], "error": True,
                }
            row = {
                "record_id": item["record_id"], "arm": item["arm"],
                "mutation_type": item["mutation_type"], "expected_check": item["expected_check"],
                "expected_target": item["expected_target"],
                "pmid": item["pmid"], "injection_note": item["injection_note"], **outcome,
            }
            results.append(row)
            mark = "!" if row["error"] else ("FAIL:" + ",".join(row["failed_families"]) if row["failed_families"] else "clean")
            print(f"  [{done}/{total}] {reviewer:9s} {item['record_id']:24s} pmid={item['pmid']:9s} -> {mark}")

    # ---- score ----
    metrics_by_reviewer = {rev: compute_metrics(results, rev) for rev in reviewers}
    print_comparison(metrics_by_reviewer)

    # ---- emit ----
    json_path = out_dir / f"reviewer_eval_{ts}.json"
    csv_path = out_dir / f"reviewer_audit_{ts}.csv"

    with open(json_path, "w") as f:
        json.dump({
            "generated_at": ts,
            "config": {
                "base": args.base, "n_per_type": args.n_per_type, "backend": backend,
                "model": model_name, "reviewers": reviewers, "self_test": args.self_test,
                "n_base_records": len(base), "base_pmids": [r["pmid"] for r in base],
            },
            "skipped": skipped,
            "metrics": metrics_by_reviewer,
            "baseline_flag_to_check_mapping": {k: list(v) for k, v in _BASELINE_FLAG_FAMILIES.items()},
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    rows = build_audit_rows(results)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[out] {json_path}")
    print(f"[out] {csv_path}   ({len(rows)} rows — one per check verdict; "
          f"fill in the human_judgment column)")


if __name__ == "__main__":
    main()
