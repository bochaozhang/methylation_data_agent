#!/usr/bin/env python3
"""
Human calibration of gold labels: export a stratified sample for review, then
score agreement (Cohen's kappa per field).

Two modes:
  --export : sample n combos stratified by (pool stratum × confidence), write
             a CSV with LLM label columns + blank human_* columns to fill in.
  --report : after the human columns are filled, compute per-field
             percent-agreement + Cohen's kappa (hand-rolled — no sklearn dep).
             κ < 0.6 is flagged: revise the labeler prompt and re-label.

Usage:
    python scripts/calibration_report.py --export --n 60
    # fill human_* columns in gold/calibration/calibration_v1.csv
    python scripts/calibration_report.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

POOL_FILE = _ROOT / "gold" / "pool" / "pool_v1.jsonl"
LABELS_FILE = _ROOT / "gold" / "labels" / "v1" / "labels_v1.jsonl"
CALIB_DIR = _ROOT / "gold" / "calibration"
DEFAULT_CSV = CALIB_DIR / "calibration_v1.csv"

# Combo-level fields to calibrate (GSE-level fields are too few to kappa).
FIELDS = ["sample_kind", "specimen", "analyte", "disease", "disease_role",
          "treatment", "lesion", "decidable_from_metadata"]


def _load_labels_and_strata() -> List[Dict[str, Any]]:
    stratum: Dict[str, str] = {}
    with open(POOL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                stratum[rec["accession"]] = rec["stratum"]
    out: List[Dict[str, Any]] = []
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for c in rec.get("combo_labels", []):
                row = dict(c)
                row["accession"] = rec["accession"]
                row["stratum"] = stratum.get(rec["accession"], "?")
                row["gse_data_type"] = (rec.get("gse_labels") or {}).get("data_type")
                out.append(row)
    return out


def export_sample(n: int, seed: int, out_csv: Path) -> int:
    combos = _load_labels_and_strata()
    rng = random.Random(seed)
    # stratify by (stratum, confidence), allocate n proportionally, ≥1 per stratum
    by_strat: Dict[tuple, List[Dict]] = defaultdict(list)
    for c in combos:
        by_strat[(c["stratum"], c.get("confidence", "unknown"))].append(c)
    total = len(combos)
    sample: List[Dict] = []
    for key, items in sorted(by_strat.items()):
        k = max(1, round(n * len(items) / total))
        sample.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(sample)

    cols = (["accession", "combo_gsm", "stratum", "gse_data_type", "evidence",
             "source_name", "molecule", "characteristics"]
            + FIELDS + ["human_" + f for f in FIELDS] + ["human_comment"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in sample:
            ch = c.get("characteristics") or {}
            row = {k: c.get(k, "") for k in cols}
            row["characteristics"] = "; ".join(f"{k}: {v}" for k, v in ch.items())
            row["evidence"] = c.get("evidence", "")
            w.writerow(row)
    print(f"[calib] exported {len(sample)} combos → {out_csv}")
    print(f"[calib] fill the human_* columns (same enumerations as LLM columns), "
          f"then run --report")
    return len(sample)


def cohen_kappa(a: List[str], b: List[str]) -> float | None:
    """Hand-rolled Cohen's kappa (unweighted). None if either rater is constant."""
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[cat] / n * cb[cat] / n for cat in cats)
    if pe == 1.0:
        return None  # both constant & identical — kappa undefined
    return (po - pe) / (1 - pe)


def report(out_csv: Path) -> int:
    with open(out_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("human_sample_kind", "").strip()]
    if not rows:
        print(f"[calib] no filled human_* rows in {out_csv} — fill at least "
              f"human_sample_kind (and ideally all human_ fields) first")
        return 1
    print(f"[calib] {len(rows)} calibrated combos\n")
    print(f"{'field':26s} {'agree':>6s} {'kappa':>6s}  flag")
    print("-" * 50)
    n_flag = 0
    for field in FIELDS:
        hkey = "human_" + field
        pairs = [(r[field].strip().lower(), r[hkey].strip().lower())
                 for r in rows if r.get(hkey, "").strip()]
        if not pairs:
            print(f"{field:26s} {'—':>6s} {'—':>6s}  (not filled)")
            continue
        a, b = zip(*pairs)
        agree = sum(x == y for x, y in pairs) / len(pairs)
        kap = cohen_kappa(list(a), list(b))
        flag = ""
        if kap is not None and kap < 0.6:
            flag = "⚠ κ<0.6 — revise labeler prompt & re-label"
            n_flag += 1
        print(f"{field:26s} {agree * 100:5.1f}% {kap if kap is not None else float('nan'):6.3f}  {flag}")
    if n_flag:
        print(f"\n{n_flag} field(s) below κ=0.6 — the labeler prompt needs work "
              f"before trusting these labels.")
    else:
        print("\nall calibrated fields κ≥0.6 — labels look trustworthy at this sample size.")
    return 0


def main() -> int:
    global POOL_FILE, LABELS_FILE
    ap = argparse.ArgumentParser(description="Gold-label human calibration (kappa).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--pool", default=str(POOL_FILE))
    ap.add_argument("--labels", default=str(LABELS_FILE))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--n", type=int, default=60, help="Sample size for --export.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    POOL_FILE = Path(args.pool)
    LABELS_FILE = Path(args.labels)

    if args.export:
        export_sample(args.n, args.seed, Path(args.csv))
        return 0
    return report(Path(args.csv))


if __name__ == "__main__":
    sys.exit(main())
