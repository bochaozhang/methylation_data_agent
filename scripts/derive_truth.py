#!/usr/bin/env python3
"""
Derive per-query truth sets from gold labels + query predicates (no LLM).

gold/labels/v1/labels_v1.jsonl (facts per combo)  ×  gold/queries.yaml
    → gold/truth/v1/{query_id}.json

Truth semantics (see gold/queries.yaml header):
  - GSE matches when gse_all holds AND every requirement group has ≥1
    satisfying combo (a group is a conjunction over ONE combo's fields).
  - Truth GSM set = union of gsm_ids of combos satisfying ALL target:true
    groups simultaneously (the samples a perfect agent would download).
  - `expect_empty: true` queries are false-positive probes — their truth set
    should be ~empty for the *agent's* request semantics; here we still derive
    the literal predicate match and report it (eval treats positives on these
    as the agent's false positives only when the agent's query intent equals
    the text — see eval_agent.py).

Usage:
    python scripts/derive_truth.py                       # labels_v1 → truth/v1/
    python scripts/derive_truth.py --check               # + sanity assertions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = _ROOT / "gold" / "queries.yaml"
LABELS_FILE = _ROOT / "gold" / "labels" / "v1" / "labels_v1.jsonl"
POOL_FILE = _ROOT / "gold" / "pool" / "pool_v1.jsonl"
TRUTH_DIR = _ROOT / "gold" / "truth" / "v1"

_CONF_ORDER = {"low": 0, "medium": 1, "high": 2}


# ---------------------------------------------------------------------- #
#  Predicate matching                                                     #
# ---------------------------------------------------------------------- #

def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def _field_matches(field: str, constraint: Any, combo: Dict[str, Any],
                   disease_groups: Dict[str, List[str]]) -> bool:
    """One combo field vs one constraint (scalar, list, or disease_group)."""
    if field == "disease_group":
        # resolve group name(s) → synonym list; match combo's disease by substring
        names = constraint if isinstance(constraint, list) else [constraint]
        synonyms: List[str] = []
        for name in names:
            if name in disease_groups:
                synonyms.extend(disease_groups[name])
            else:
                synonyms.append(name)
        disease = _norm(combo.get("disease"))
        return any(syn in disease or disease in syn for syn in map(_norm, synonyms) if syn)

    value = combo.get(field)
    if field in ("multi_organism", "decidable_from_metadata"):
        return bool(value) == bool(constraint)  # boolean fields

    allowed = constraint if isinstance(constraint, list) else [constraint]
    if field in ("disease", "organism", "specimen", "analyte", "sample_kind",
                 "disease_role", "treatment", "lesion"):
        return _norm(value) in {_norm(a) for a in allowed}
    return str(value) in {str(a) for a in allowed}  # exact (e.g. technology)


def _combo_satisfies(group: Dict[str, Any], combo: Dict[str, Any],
                     disease_groups: Dict[str, List[str]], min_conf: str) -> bool:
    if _CONF_ORDER.get(_norm(combo.get("confidence")), 0) < _CONF_ORDER[min_conf]:
        return False
    return all(
        _field_matches(field, constraint, combo, disease_groups)
        for field, constraint in (group.get("fields") or {}).items()
    )


def derive_query(query: Dict[str, Any], labels_by_acc: Dict[str, Dict],
                 disease_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    pred = query.get("predicate") or {}
    gse_all = pred.get("gse_all") or {}
    groups = pred.get("groups") or []
    min_conf = _norm(pred.get("min_confidence", "medium"))
    target_groups = [g for g in groups if g.get("target")]

    gse_positive: List[str] = []
    gsm_positive: Dict[str, List[str]] = {}
    rationale: Dict[str, str] = {}
    n_partial = 0

    for acc, rec in sorted(labels_by_acc.items()):
        gse_labels = rec.get("gse_labels") or {}
        combo_labels = rec.get("combo_labels") or []
        if rec.get("coverage") == "partial":
            n_partial += 1

        # expand combos → per-combo GSM ids (labels store combo_gsm only; the
        # expansion lives in the evidence cache snapshot embedded at export time)
        combo_gsm_ids: Dict[str, List[str]] = {
            c.get("combo_gsm"): (c.get("gsm_ids") or [c.get("combo_gsm")])
            for c in combo_labels
        }

        # GSE-level gate
        if not all(_field_matches(f, v, gse_labels, disease_groups)
                   for f, v in gse_all.items()):
            continue

        # requirement groups
        group_hits: Dict[str, List[Dict]] = {}
        ok = True
        for g in groups:
            hits = [c for c in combo_labels
                    if _combo_satisfies(g, c, disease_groups, min_conf)]
            group_hits[g["name"]] = hits
            if not hits:
                ok = False
                break
        if not ok:
            continue

        # truth GSM set = union over target groups of each group's satisfying
        # combos. (NOT intersection: "tumor tissue AND adjacent tissue" are two
        # different sample populations — both are wanted, in the same GSE but
        # never in the same combo.)
        selected: List[Dict] = []
        for g in target_groups:
            selected.extend(group_hits[g["name"]])
        # dedupe by combo_gsm, keep order
        seen_ids = set()
        dedup: List[Dict] = []
        for c in selected:
            key = c.get("combo_gsm")
            if key not in seen_ids:
                seen_ids.add(key)
                dedup.append(c)
        selected = dedup
        # no target groups (pure existence query) → GSE-level truth only
        gse_positive.append(acc)
        gsms = sorted({g for c in selected for g in combo_gsm_ids.get(c.get("combo_gsm"), [])})
        if gsms:
            gsm_positive[acc] = gsms
        parts = [f"{g['name']}: {len(group_hits[g['name']])} combo(s)" for g in groups]
        rationale[acc] = "; ".join(parts) + f" → {len(gsms)} target GSM (union of target groups)"

    pool_accs = set(labels_by_acc)
    return {
        "query_id": query["id"],
        "text": query.get("text"),
        "expect_empty": bool(query.get("expect_empty")),
        "stats": {
            "n_labeled_gse": len(pool_accs),
            "n_positive": len(gse_positive),
            "n_gsm_positive": sum(len(v) for v in gsm_positive.values()),
            "n_partial_coverage": n_partial,
        },
        "gse_positive": gse_positive,
        "gse_negative": sorted(pool_accs - set(gse_positive)),
        "gsm_positive": gsm_positive,
        "per_gse_rationale": rationale,
    }


# ---------------------------------------------------------------------- #
#  Main                                                                   #
# ---------------------------------------------------------------------- #

def load_labels(path: Path) -> Dict[str, Dict]:
    labels: Dict[str, Dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            labels[rec["accession"]] = rec
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive truth sets from labels × predicates.")
    ap.add_argument("--queries", default=str(QUERIES_FILE))
    ap.add_argument("--labels", default=str(LABELS_FILE))
    ap.add_argument("--out-dir", default=str(TRUTH_DIR))
    ap.add_argument("--check", action="store_true", help="Print sanity report.")
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        qcfg = yaml.safe_load(f)
    disease_groups = {k: [str(s) for s in v] for k, v in (qcfg.get("disease_groups") or {}).items()}
    labels = load_labels(Path(args.labels))
    if not labels:
        print(f"[truth] no labels at {args.labels} — run build_gold_labels.py --export first")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for query in qcfg.get("queries", []):
        truth = derive_query(query, labels, disease_groups)
        out = out_dir / f"{query['id']}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(truth, f, ensure_ascii=False, indent=1)
        flag = " (expect_empty)" if truth["expect_empty"] else ""
        print(f"[truth] {query['id']:24s} {truth['stats']['n_positive']:3d} GSE positive, "
              f"{truth['stats']['n_gsm_positive']:4d} GSM{flag}")

    if args.check:
        print("\n[sanity] per-query positives by pool stratum:")
        pool_stratum: Dict[str, str] = {}
        pool_path = Path(POOL_FILE)
        if pool_path.exists():
            with open(pool_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    pool_stratum[rec["accession"]] = rec["stratum"]
        from collections import Counter
        for query in qcfg.get("queries", []):
            truth = json.loads((out_dir / f"{query['id']}.json").read_text())
            strat = Counter(pool_stratum.get(a, "?") for a in truth["gse_positive"])
            print(f"  {query['id']:24s} {dict(strat)}")
            if truth["expect_empty"] and truth["stats"]["n_positive"] > 0:
                print(f"    ⚠ expect_empty query has {truth['stats']['n_positive']} "
                      f"positives: {truth['gse_positive'][:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
