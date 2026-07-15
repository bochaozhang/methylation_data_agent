#!/usr/bin/env python3
"""
Benchmark runner — quickly swap LLM backend + model, run the pipeline, produce
a query_log CSV for comparison.

SPEC version is controlled manually (cp SPEC_vN.md SPEC.md). The runner records
which SPEC was used (SPEC_NAME auto-derived from the SPEC.md heading).

Usage:
  # Single query, deepseek
  python scripts/benchmark.py --llm deepseek --model deepseek-chat \
      --query "colorectal cancer和非癌对照的cfDNA甲基化数据" --output-dir data/benchmark

  # Multiple queries from a file (one per line)
  python scripts/benchmark.py --llm zhipu --model glm-4-flash \
      --queries-file queries.txt --output-dir data/benchmark

  # Then compare:
  python scripts/compare_benchmarks.py --dir data/benchmark/query_logs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make project root importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

# LLM backend → env var that holds the model name (get_llm reads env with priority).
_LLM_MODEL_ENV = {
    "deepseek": "DEEPSEEK_MODEL",
    "zhipu": "ZHIPU_MODEL",
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark runner: swap LLM, run pipeline, produce query_log.")
    ap.add_argument("--llm", required=True, help="LLM backend: deepseek | zhipu | openai | anthropic")
    ap.add_argument("--model", default=None, help="Model name (e.g. deepseek-chat, glm-4-flash)")
    ap.add_argument("--query", default=None, help="Single query string.")
    ap.add_argument("--queries-file", default=None, help="File with one query per line.")
    ap.add_argument("--output-dir", default="data/benchmark", help="Output dir for query_logs.")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args()

    # Collect queries.
    queries = []
    if args.query:
        queries.append(args.query)
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries.extend(line.strip() for line in f if line.strip())
    if not queries:
        ap.error("Provide --query or --queries-file")

    # Load config and override LLM + output_dir.
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["llm"]["backend"] = args.llm
    if args.model:
        env_var = _LLM_MODEL_ENV.get(args.llm)
        if env_var:
            os.environ[env_var] = args.model
            print(f"[benchmark] LLM={args.llm} model={args.model} (env {env_var} set)")
        else:
            print(f"[benchmark] LLM={args.llm} model={args.model} (no env override; using config)")
    else:
        print(f"[benchmark] LLM={args.llm} (model from env/config)")

    # Override output_dir so query_logs go to the benchmark dir.
    config["download"]["output_dir"] = args.output_dir
    print(f"[benchmark] output_dir={args.output_dir}")

    # SPEC version (info only; user controls by cp SPEC_vN.md SPEC.md).
    from skills.geo_filter import SPEC_NAME
    print(f"[benchmark] SPEC={SPEC_NAME}")

    # Run pipeline per query (registry=None: no production writes).
    from agents.agent1_pipeline import run_agent1_pipeline
    for i, query in enumerate(queries):
        print(f"\n[benchmark] query {i+1}/{len(queries)}: {query[:60]}...")
        try:
            final = run_agent1_pipeline(query, config, registry=None)
            qlog = final.get("query_logger") if isinstance(final, dict) else None
            if qlog and qlog.path:
                print(f"[benchmark] query_log: {qlog.path}")
            else:
                print("[benchmark] WARNING: no query_log produced")
        except Exception as e:
            print(f"[benchmark] FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[benchmark] Done. {len(queries)} query(ies) run.")
    print(f"[benchmark] Compare: python scripts/compare_benchmarks.py --dir {args.output_dir}/query_logs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
