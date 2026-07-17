#!/usr/bin/env python3
"""
Benchmark runner — quickly swap LLM backend + model + SPEC version, run the
pipeline, produce a query_log CSV for comparison.

SPEC version can be swapped via --skill (reads the file, overrides the filter's
SPEC at runtime, and labels the query_log with the filename).

Usage:
  # deepseek + SPEC_v3
  python scripts/benchmark.py --llm deepseek --model deepseek-chat \
      --skill skills/geo_filter/SPEC_v3.md \
      --query "colorectal cancer和非癌对照的cfDNA甲基化数据" --output-dir data/benchmark

  # zhipu + SPEC_v4
  python scripts/benchmark.py --llm zhipu --model glm-4-flash \
      --skill skills/geo_filter/SPEC_v4.md \
      --query "..." --output-dir data/benchmark

  # No --skill: uses whatever skills/geo_filter/SPEC.md currently is
  python scripts/benchmark.py --llm deepseek --query "..." --output-dir data/benchmark

  # Compare:
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
    "qwen": "QWEN_MODEL",
}


def override_spec(skill_path: str) -> str:
    """
    Read a SPEC file and override the filter's SPEC/SYSTEM_PROMPT/SPEC_NAME at
    runtime. Returns the filename (for labeling).

    Must be called BEFORE importing agent1_pipeline (so its `from skills.geo_filter
    import SPEC_NAME` picks up the overridden value).
    """
    p = Path(skill_path)
    if not p.exists():
        # Try relative to skills/geo_filter/
        p = _ROOT / "skills" / "geo_filter" / skill_path
    if not p.exists():
        raise FileNotFoundError(f"SPEC file not found: {skill_path}")

    new_spec = p.read_text(encoding="utf-8")
    filename = p.name

    import skills.geo_filter.skill as _fs
    import skills.geo_filter as _pkg
    _fs.SPEC = new_spec
    _fs.SYSTEM_PROMPT = new_spec + "\n" + _fs._OUTPUT_CONTRACT
    parsed = _fs._parse_spec_name(new_spec)
    _fs.SPEC_NAME = f"{parsed} ({filename})"
    # Also update the package-level SPEC_NAME (from __init__.py) so that
    # `from skills.geo_filter import SPEC_NAME` (in agent1_pipeline) gets it.
    _pkg.SPEC_NAME = _fs.SPEC_NAME

    print(f"[benchmark] SPEC overridden: {filename} → SPEC_NAME={_fs.SPEC_NAME}")
    return filename


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark runner: swap LLM + SPEC, run pipeline, produce query_log.")
    ap.add_argument("--llm", required=True, help="LLM backend: deepseek | zhipu | openai | anthropic")
    ap.add_argument("--model", default=None, help="Model name (e.g. deepseek-chat, glm-4-flash)")
    ap.add_argument("--skill", default=None,
                    help="SPEC file to use (e.g. SPEC_v3.md or skills/geo_filter/SPEC_v3.md). "
                         "Overrides the filter's SPEC at runtime + labels query_log with filename.")
    ap.add_argument("--query", default=None, help="Single query string.")
    ap.add_argument("--queries-file", default=None, help="File with one query per line.")
    ap.add_argument("--output-dir", default="data/benchmark", help="Output dir for query_logs.")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args()

    # Override SPEC BEFORE importing agent1_pipeline (so SPEC_NAME flows through).
    skill_file = None
    if args.skill:
        skill_file = override_spec(args.skill)

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

    # SPEC info (from override or current SPEC.md).
    from skills.geo_filter import SPEC_NAME
    print(f"[benchmark] SPEC_NAME={SPEC_NAME}")

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
