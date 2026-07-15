#!/usr/bin/env python3
"""
Compare benchmark query_log CSVs across LLM × SPEC combos.

Reads all query_*.csv in a directory, groups by (llm_model, SPEC name), and
produces:
  ① 四态分布表 (stdout markdown): per combo → download/lead/exclude/manual_review counts + tokens.
  ② 逐 GSE 一致性矩阵 (gse_outcome_matrix.csv): rows=GSE, columns=combo, cells=outcome.
  ③ token 汇总 (token_summary.csv): per combo → total/avg/cached tokens.
  ④ 准确率 (accuracy.csv, if --gold): per combo → match rate against gold standard.

Usage:
  python scripts/compare_benchmarks.py --dir data/benchmark/query_logs
  python scripts/compare_benchmarks.py --dir data/benchmark/query_logs --gold gold.csv

gold.csv format: accession,outcome (the correct 4-state outcome per GSE).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------- #
#  Parse a query_log CSV                                                 #
# ---------------------------------------------------------------------- #

def parse_query_log(path: str) -> Optional[Dict[str, Any]]:
    """Parse one query_log CSV → {combo, llm_model, spec, total_tokens, rows}."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  skip {path}: {e}")
        return None

    # Metadata from # comments.
    meta = {}
    data_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            # "# key: value"
            if ":" in s:
                k, v = s[1:].split(":", 1)
                meta[k.strip()] = v.strip()
        elif s:
            data_lines.append(line)

    if not data_lines:
        return None

    llm_model = meta.get("llm_model", "unknown")
    spec = meta.get("注意事项", meta.get("spec_version", "unknown"))
    total_tokens = int(meta.get("total_tokens", "0") or "0")
    combo = f"{llm_model} + {spec}"

    # Data rows.
    reader = csv.DictReader(io.StringIO("".join(data_lines)))
    rows = []
    for r in reader:
        rows.append({
            "accession": r.get("accession", ""),
            "outcome": r.get("recommended_action", r.get("outcome", "")),
            "reasoning": r.get("reasoning", ""),
            "total_tokens": r.get("total_tokens", ""),
        })
    return {"combo": combo, "llm_model": llm_model, "spec": spec,
            "total_tokens": total_tokens, "rows": rows, "path": path}


# ---------------------------------------------------------------------- #
#  Reports                                                               #
# ---------------------------------------------------------------------- #

def report_outcome_distribution(logs: List[Dict]) -> None:
    """① 四态分布表 (markdown)."""
    print("\n## ① 四态分布\n")
    print("| combo (LLM + SPEC) | download | lead | exclude | manual_review | total_tokens |")
    print("|---|---|---|---|---|---|")
    for log in logs:
        counts = {"download": 0, "lead": 0, "exclude": 0, "manual_review": 0}
        for r in log["rows"]:
            oc = r["outcome"]
            if oc in counts:
                counts[oc] += 1
        n = len(log["rows"])
        print(f"| {log['combo']} | {counts['download']} | {counts['lead']} | "
              f"{counts['exclude']} | {counts['manual_review']} | {log['total_tokens']} |")


def report_gse_matrix(logs: List[Dict], out_dir: str) -> None:
    """② 逐 GSE 一致性矩阵 (CSV)."""
    # Collect all accessions + combo names.
    all_gse = sorted(set(r["accession"] for log in logs for r in log["rows"]))
    combo_names = [log["combo"] for log in logs]

    out_path = os.path.join(out_dir, "gse_outcome_matrix.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["accession"] + combo_names + ["flips"])
        for gse in all_gse:
            outcomes = []
            for log in logs:
                match = next((r["outcome"] for r in log["rows"] if r["accession"] == gse), "—")
                outcomes.append(match)
            unique = set(o for o in outcomes if o != "—")
            flips = "⚠️" if len(unique) > 1 else ""
            w.writerow([gse] + outcomes + [flips])
    n_flips = sum(1 for gse in all_gse for log in [None] if len(set(
        next((r["outcome"] for r in log2["rows"] if r["accession"] == gse), "—")
        for log2 in logs if next((r["outcome"] for r in log2["rows"] if r["accession"] == gse), "—") != "—"
    )) > 1) if len(logs) > 1 else 0
    print(f"\n## ② GSE 一致性矩阵 → {out_path}")
    print(f"   {len(all_gse)} GSE × {len(combo_names)} combos. 翻转(不一致)的 GSE 标 ⚠️.")
    # Print a few flips
    flip_rows = []
    for gse in all_gse:
        outcomes = [next((r["outcome"] for r in log["rows"] if r["accession"] == gse), "—") for log in logs]
        if len(set(o for o in outcomes if o != "—")) > 1:
            flip_rows.append((gse, outcomes))
    if flip_rows:
        print(f"   翻转 GSE ({len(flip_rows)} 个):")
        for gse, oc in flip_rows[:10]:
            print(f"     {gse}: {' vs '.join(oc)}")
        if len(flip_rows) > 10:
            print(f"     ... 共 {len(flip_rows)} 个")


def report_tokens(logs: List[Dict], out_dir: str) -> None:
    """③ token 汇总 (CSV)."""
    out_path = os.path.join(out_dir, "token_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo", "total_tokens", "n_datasets", "avg_tokens_per_dataset"])
        for log in logs:
            n = len(log["rows"])
            avg = log["total_tokens"] // n if n else 0
            w.writerow([log["combo"], log["total_tokens"], n, avg])
    print(f"\n## ③ Token 汇总 → {out_path}")
    for log in logs:
        n = len(log["rows"])
        avg = log["total_tokens"] // n if n else 0
        print(f"   {log['combo']}: {log['total_tokens']} total / {n} datasets = {avg}/dataset")


def report_accuracy(logs: List[Dict], gold_path: str, out_dir: str) -> None:
    """④ 准确率 (if gold standard provided)."""
    gold = {}
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gold[r["accession"]] = r["outcome"].strip()
    except Exception as e:
        print(f"\n## ④ 准确率: 无法读取 gold ({e})")
        return

    out_path = os.path.join(out_dir, "accuracy.csv")
    print(f"\n## ④ 准确率 (gold: {gold_path})\n")
    print("| combo | correct | total_gold | accuracy | mismatched GSEs |")
    print("|---|---|---|---|---|")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo", "correct", "total_gold", "accuracy_pct", "mismatched"])
        for log in logs:
            correct = 0
            mismatched = []
            for acc, gold_outcome in gold.items():
                pred = next((r["outcome"] for r in log["rows"] if r["accession"] == acc), None)
                if pred == gold_outcome:
                    correct += 1
                else:
                    mismatched.append(f"{acc}({pred}≠{gold_outcome})")
            total = len(gold)
            acc_pct = round(100 * correct / total, 1) if total else 0
            mm_str = "; ".join(mismatched[:5]) + ("..." if len(mismatched) > 5 else "")
            print(f"| {log['combo']} | {correct} | {total} | {acc_pct}% | {mm_str} |")
            w.writerow([log["combo"], correct, total, acc_pct, "; ".join(mismatched)])
    print(f"   → {out_path}")


# ---------------------------------------------------------------------- #
#  Main                                                                  #
# ---------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Compare benchmark query_log CSVs.")
    ap.add_argument("--dir", required=True, help="Directory with query_*.csv files.")
    ap.add_argument("--gold", default=None, help="Gold standard CSV (accession,outcome).")
    args = ap.parse_args()

    query_dir = args.dir
    csv_files = sorted(Path(query_dir).glob("query_*.csv"))
    if not csv_files:
        print(f"No query_*.csv files in {query_dir}")
        return 1

    print(f"Found {len(csv_files)} query_log file(s) in {query_dir}")

    logs = []
    for f in csv_files:
        log = parse_query_log(str(f))
        if log:
            logs.append(log)
            print(f"  {f.name}: {log['combo']} ({len(log['rows'])} rows)")

    if not logs:
        print("No valid query_logs found.")
        return 1

    # Reports
    report_outcome_distribution(logs)
    report_gse_matrix(logs, query_dir)
    report_tokens(logs, query_dir)
    if args.gold:
        report_accuracy(logs, args.gold, query_dir)

    print(f"\nDone. Output files in {query_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
