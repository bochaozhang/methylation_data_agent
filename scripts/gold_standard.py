"""
Gold-standard evaluation set for tools/query_clarifier.py extract_paper_structured().

Usage (on SSH server):
    set -a && source .env && set +a
    export NCBI_PROXY=socks5://127.0.0.1:1080
    # proxy must already be running: bash /home/ubuntu/bochaozhang/proxy.sh
    source .venv/bin/activate
    python scripts/gold_standard.py

For each PMID in GOLD_STANDARD:
    1. Fetch the real abstract via tools.ncbi_search.efetch_abstracts()
    2. Run extract_paper_structured() on it
    3. Compare predicted fields against the manually-verified gold values
    4. Print per-record and per-field accuracy

Only fill in a record's gold fields once they have been manually verified
against the actual PubMed listing. Records with gold_verified=False are
fetched/extracted for inspection but excluded from the accuracy summary.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ncbi_search import efetch_abstracts
from tools.query_clarifier import extract_paper_structured
from utils.llm_factory import get_llm


# ------------------------------------------------------------------ #
#  Gold-standard records                                              #
# ------------------------------------------------------------------ #
# gold fields are compared against extract_paper_structured() output.
#   - scalar fields (cancer_type, sample_type, sample_size_case, ...) → exact match
#   - performance_metrics: {...}                                      → match per sub-key present
#   - dataset_ids_exclude: [...]                                      → none of these may appear
#     in predicted dataset_ids
#   - dataset_ids_include: [...]                                      → all of these must appear
#     in predicted dataset_ids
#
# Leave a field out entirely if there is no verified ground truth for it yet
# (it will simply be skipped when scoring, not counted as a miss).

GOLD_STANDARD: List[Dict[str, Any]] = [
    {
        "pmid": "41796341",
        "gold_verified": True,
        "cancer_type": "CRC",
        "sample_type": "plasma_cfdna",
        "sample_size_case": 636,
        "performance_metrics": {
            "auc_validation": None,  # abstract only reports sensitivity 87.82% / specificity
                                       # 91.88% — no AUC. LLM previously hallucinated AUC=0.91
                                       # by confusing it with specificity.
        },
        "notes": "Known bug: LLM reported auc_validation=0.91, but no AUC is in the abstract.",
    },
    {
        "pmid": "40860669",
        "gold_verified": True,
        "sample_type": "tissue",  # NOT plasma_cfdna — abstract's primary cohort is tissue
        "performance_metrics": {
            "auc_validation": 0.922,  # this is the TISSUE AUC, not the cfDNA AUC (0.728, n=33)
        },
        "dataset_ids_exclude": ["GSE50132"],  # mouse WBC reference panel, not primary data
        "notes": (
            "Known bug: LLM reported sample_type=plasma_cfdna with AUC=0.922 (tissue AUC "
            "mislabeled as cfDNA), and included GSE50132 (background-filter reference panel) "
            "as a dataset_id."
        ),
    },
    # --- Remaining 8 records: TODO — paste in PMID + gold-verified fields below,
    #     set gold_verified=True once confirmed against PubMed. ---
    {"pmid": "", "gold_verified": False},
    {"pmid": "", "gold_verified": False},
    {"pmid": "", "gold_verified": False},
    {"pmid": "", "gold_verified": False},
    {"pmid": "", "gold_verified": False},
    {"pmid": "", "gold_verified": False},
    {"pmid": "", "gold_verified": False},
    {"pmid": "", "gold_verified": False},
]

_SCALAR_FIELDS = [
    "cancer_type", "sample_type", "has_normal_control", "has_cancer_samples",
    "technology", "sample_size_case", "sample_size_control", "early_stage_count",
    "has_external_validation", "data_availability", "confidence_level",
]


# ------------------------------------------------------------------ #
#  Comparison                                                         #
# ------------------------------------------------------------------ #

def _norm(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def compare_record(gold: Dict[str, Any], predicted: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of {field, gold, predicted, match} for every gold field present."""
    rows: List[Dict[str, Any]] = []

    for field in _SCALAR_FIELDS:
        if field not in gold:
            continue
        pred_val = predicted.get(field)
        rows.append({
            "field": field,
            "gold": gold[field],
            "predicted": pred_val,
            "match": _norm(gold[field]) == _norm(pred_val),
        })

    if "performance_metrics" in gold:
        pred_metrics = predicted.get("performance_metrics") or {}
        for sub_key, gold_val in gold["performance_metrics"].items():
            pred_val = pred_metrics.get(sub_key)
            rows.append({
                "field": f"performance_metrics.{sub_key}",
                "gold": gold_val,
                "predicted": pred_val,
                "match": gold_val == pred_val,
            })

    if "dataset_ids_exclude" in gold:
        pred_ids = set(predicted.get("dataset_ids") or [])
        for ds_id in gold["dataset_ids_exclude"]:
            rows.append({
                "field": f"dataset_ids excludes {ds_id}",
                "gold": "absent",
                "predicted": "present" if ds_id in pred_ids else "absent",
                "match": ds_id not in pred_ids,
            })

    if "dataset_ids_include" in gold:
        pred_ids = set(predicted.get("dataset_ids") or [])
        for ds_id in gold["dataset_ids_include"]:
            rows.append({
                "field": f"dataset_ids includes {ds_id}",
                "gold": "present",
                "predicted": "present" if ds_id in pred_ids else "absent",
                "match": ds_id in pred_ids,
            })

    return rows


# ------------------------------------------------------------------ #
#  Runner                                                             #
# ------------------------------------------------------------------ #

def load_llm():
    cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    llm = get_llm(cfg["llm"])
    print(f"[setup] backend={cfg['llm']['backend']}  model={cfg['llm'].get('model', '?')}\n")
    return llm


def run_evaluation() -> None:
    llm = load_llm()
    verified = [g for g in GOLD_STANDARD if g.get("gold_verified") and g.get("pmid")]

    if not verified:
        print("No verified gold-standard records yet. Add gold fields to GOLD_STANDARD first.")
        return

    all_rows: List[Dict[str, Any]] = []

    for gold in verified:
        pmid = gold["pmid"]
        print(f"\n{'=' * 60}")
        print(f"PMID {pmid}")
        print('=' * 60)

        fetched = efetch_abstracts([pmid])
        if not fetched:
            print(f"  [SKIP] could not fetch abstract for PMID {pmid}")
            continue
        rec = fetched[0]

        predicted = extract_paper_structured(
            abstract=rec.get("abstract", ""),
            llm=llm,
            pmid=pmid,
            title=rec.get("title", ""),
        )

        rows = compare_record(gold, predicted)
        for row in rows:
            status = "PASS" if row["match"] else "FAIL"
            print(f"  [{status}] {row['field']}: gold={row['gold']!r}  predicted={row['predicted']!r}")
        all_rows.extend(rows)

    # ------------------------------------------------------------------ #
    #  Per-field accuracy summary                                         #
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("Per-field accuracy summary")
    print('=' * 60)

    by_field: Dict[str, List[bool]] = {}
    for row in all_rows:
        by_field.setdefault(row["field"], []).append(row["match"])

    total_correct = sum(r["match"] for r in all_rows)
    total = len(all_rows)

    for field, matches in sorted(by_field.items()):
        acc = sum(matches) / len(matches) * 100
        print(f"  {field:40s} {sum(matches)}/{len(matches)}  ({acc:.0f}%)")

    overall = total_correct / total * 100 if total else 0.0
    print(f"\n  OVERALL: {total_correct}/{total}  ({overall:.0f}%)")
    print(f"  Records evaluated: {len(verified)} / {len(GOLD_STANDARD)} total gold-standard slots")


if __name__ == "__main__":
    run_evaluation()
