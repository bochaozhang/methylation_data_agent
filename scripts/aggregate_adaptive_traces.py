#!/usr/bin/env python3
"""
Aggregate the §2 (adaptive guardrails) + §3 (JSON parse stability) benchmark
metrics into one report.

Reads two artifacts that the pipeline already writes:
  {output_dir}/adaptive_traces.jsonl    one record per adaptive run
                                         (skills/adaptive_evidence/trace_log.py)
  {output_dir}/query_logs/query_*.csv   json_parse_tier column
                                         (utils/query_logger.py)

Prints: adaptive resolution rate, guard-trigger histogram, step/fetch
distribution, outcome transitions, and the JSON first-pass parse rate.
Optionally writes a JSON summary (--json).

Stdlib-only (csv + json + yaml for the default output_dir).

  python scripts/aggregate_adaptive_traces.py
  python scripts/aggregate_adaptive_traces.py --output-dir ./data --json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_output_dir(config_path: str) -> str:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("download", {}).get("output_dir", "./data")
    except Exception:
        return "./data"


def read_traces(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_json_tiers(query_logs_dir: Path) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    files = sorted(glob.glob(str(query_logs_dir / "query_*.csv")))
    for fp in files:
        try:
            with open(fp, encoding="utf-8-sig") as f:
                lines = [ln for ln in f.read().splitlines()
                         if ln.strip() and not ln.startswith("#")]
            reader = csv.DictReader(lines)
            if "json_parse_tier" not in (reader.fieldnames or []):
                continue  # older logs predate the column
            for row in reader:
                tier = (row.get("json_parse_tier") or "").strip()
                if tier:
                    counts[tier] += 1
        except Exception:
            continue
    return counts, len(files)


def _dist(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 2),
        "median": statistics.median(vals),
        "max": max(vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate adaptive + JSON-parse benchmark metrics.")
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--output-dir", default=None,
                    help="Data root (default: config download.output_dir).")
    ap.add_argument("--traces", default=None,
                    help="Override path to adaptive_traces.jsonl.")
    ap.add_argument("--query-logs", default=None,
                    help="Override directory containing query_*.csv.")
    ap.add_argument("--json", action="store_true", help="Also write a JSON summary to stdout-path.")
    args = ap.parse_args()

    output_dir = args.output_dir or load_output_dir(args.config)
    traces_path = Path(args.traces) if args.traces else Path(output_dir) / "adaptive_traces.jsonl"
    qlogs_dir = Path(args.query_logs) if args.query_logs else Path(output_dir) / "query_logs"

    traces = read_traces(traces_path)
    tiers, n_csv = read_json_tiers(qlogs_dir)

    # ---- adaptive (§2) ----
    total = len(traces)
    resolved = sum(1 for r in traces if r.get("resolved"))
    fallback = sum(1 for r in traces if r.get("fallback"))
    event_hist: Counter = Counter()
    transitions: Dict[str, Counter] = defaultdict(Counter)
    for r in traces:
        for e, c in (r.get("event_counts") or {}).items():
            event_hist[e] += c
        transitions[r.get("outcome_before", "?")][r.get("outcome_after", "?")] += 1
    steps = [int(r.get("n_steps", 0)) for r in traces]
    fetches = [int(r.get("n_fetches", 0)) for r in traces]

    # ---- json parse (§3) ----
    tier_total = sum(tiers.values())
    clean = tiers.get("clean", 0)

    # ---- report ----
    hr = "=" * 70
    print(hr)
    print(f"Stability report   (output_dir={output_dir})")
    print(hr)

    print("\n## Adaptive evidence (§2 guardrails)")
    if total == 0:
        print(f"  no traces at {traces_path} (adaptive never fired, or file absent)")
    else:
        print(f"  adaptive runs        : {total}")
        print(f"  resolution rate      : {resolved}/{total} = {resolved/total:.1%}  "
              f"(manual_review → download/exclude)")
        print(f"  fallback rate        : {fallback}/{total} = {fallback/total:.1%}  "
              f"(degraded to first_verdict; should be rare)")
        print(f"  steps distribution   : {_dist(steps)}")
        print(f"  fetches distribution : {_dist(fetches)}")
        print(f"  guard / event histogram:")
        for e, c in event_hist.most_common():
            print(f"      {e:<22} {c}")
        print(f"  outcome transitions (before → after):")
        for before, afters in sorted(transitions.items()):
            parts = ", ".join(f"{a}={n}" for a, n in afters.most_common())
            print(f"      {before:<16} → {parts}")

    print("\n## JSON parse stability (§3)")
    if tier_total == 0:
        print(f"  no json_parse_tier data in {qlogs_dir} (no query_*.csv with the column)")
    else:
        print(f"  query_logs scanned   : {n_csv}")
        print(f"  total verdicts       : {tier_total}")
        for tier in ("clean", "fenced", "extracted", "failed"):
            c = tiers.get(tier, 0)
            print(f"      {tier:<10} {c:>6}  ({c/tier_total:5.1%})")
        print(f"  first-pass clean rate: {clean}/{tier_total} = {clean/tier_total:.1%}  "
              f"(json_mode target ≈ 100%; P0 gate ≥ 99%)")

    if args.json:
        summary = {
            "output_dir": output_dir,
            "adaptive": {
                "runs": total,
                "resolution_rate": (resolved / total) if total else None,
                "fallback_rate": (fallback / total) if total else None,
                "steps": _dist(steps),
                "fetches": _dist(fetches),
                "event_histogram": dict(event_hist),
                "transitions": {b: dict(a) for b, a in transitions.items()},
            },
            "json_parse": {
                "query_logs_scanned": n_csv,
                "total": tier_total,
                "tiers": dict(tiers),
                "clean_rate": (clean / tier_total) if tier_total else None,
            },
        }
        out_path = Path(output_dir) / "stability_report.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nJSON summary written: {out_path}")
        except Exception as e:
            print(f"\n[warn] could not write JSON summary: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
