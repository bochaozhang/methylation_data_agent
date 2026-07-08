#!/usr/bin/env python3
"""
MethyAgent CLI Entry Point

Usage examples:
  # Semantic search (natural language)
  python main.py --query "EPIC平台在2024年的乳腺癌相关数据"

  # Exact accession download
  python main.py --query "下载GEO编号GSE124600的所有数据"

  # English query
  python main.py --query "breast cancer WGBS methylation 2022-2023"

  # Run only DatabaseAgent (skip literature mining)
  python main.py --query "乳腺癌EPIC甲基化" --agent db-only

  # Run only LiteratureAgent (skip database search)
  python main.py --query "乳腺癌EPIC甲基化" --agent lit-only

  # View current registry status
  python main.py --status

  # Use a custom config file
  python main.py --query "..." --config /path/to/settings.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="methyagent",
        description="MethyAgent: Automated methylation data acquisition system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query (natural language or accession number)",
    )
    parser.add_argument(
        "--agent",
        choices=["both", "db-only", "lit-only"],
        default="both",
        help="Which agents to run (default: both)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current registry status and exit",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override download output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse query and show intent without downloading",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # ---- Setup logging ----
    import logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---- Load config ----
    if not Path(args.config).exists():
        print(f"[ERROR] Config file not found: {args.config}")
        print("  Run from the methyagent/ directory, or specify --config path.")
        sys.exit(1)

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override output dir if specified
    if args.output_dir:
        config["download"]["output_dir"] = args.output_dir

    # ---- Status mode ----
    if args.status:
        _show_status(config)
        return

    # ---- Query required for all other modes ----
    if not args.query:
        parser.print_help()
        sys.exit(1)

    # ---- Dry run mode ----
    if args.dry_run:
        _dry_run(args.query, config)
        return

    # ---- Check API key ----
    _check_api_key(config)

    # ---- Pick orchestrator version ----
    # Priority: ORCHESTRATOR_VERSION env var > orchestrator.version in settings.yaml > "v1".
    # v1 = agents/orchestrator.py (fixed sequential pipeline, in production use).
    # v2 = agents/orchestrator_v2.py (agentic tool-calling graph, new/experimental).
    orchestrator_version = (
        os.environ.get("ORCHESTRATOR_VERSION")
        or config.get("orchestrator", {}).get("version", "v1")
    ).lower()

    # ---- Run MethyAgent ----
    print(f"\n{'='*60}")
    print(f"  MethyAgent Starting")
    print(f"  Query       : {args.query}")
    print(f"  Mode        : {args.agent}")
    print(f"  Orchestrator: {orchestrator_version}")
    print(f"  Output      : {config['download']['output_dir']}")
    print(f"{'='*60}\n")

    try:
        if orchestrator_version == "v2":
            from agents.orchestrator_v2 import run_methyagent_v2
            final_state = run_methyagent_v2(query=args.query, config_path=args.config)
        else:
            from agents.orchestrator import run_methyagent
            final_state = run_methyagent(
                query=args.query,
                config_path=args.config,
                agent_mode=args.agent,
            )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] MethyAgent stopped by user.")
        print("  Partial results are saved in the registry.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] MethyAgent failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # ---- Print summary ----
    if orchestrator_version == "v2":
        _print_summary_v2(final_state)
    else:
        _print_summary(final_state)


def _show_status(config: dict):
    """Display current registry status."""
    from registry.registry import Registry

    db_path = config["registry"]["db_path"]
    if not Path(db_path).exists():
        print(f"Registry not found at: {db_path}")
        print("No downloads have been performed yet.")
        return

    registry = Registry(db_path)
    summary = registry.get_summary()
    datasets = registry.get_all()

    print(f"\n{'='*60}")
    print(f"  MethyAgent Registry Status")
    print(f"  Database: {db_path}")
    print(f"{'='*60}")
    print(f"\nTotal datasets: {summary['total']}")
    print(f"\nBy status:")
    for status, count in summary.get("by_status", {}).items():
        print(f"  {status:15s}: {count}")
    print(f"\nBy source:")
    for source, count in summary.get("by_source", {}).items():
        print(f"  {source:15s}: {count}")
    print(f"\nBy data type:")
    for dtype, count in summary.get("by_data_type", {}).items():
        print(f"  {dtype:15s}: {count}")

    if datasets:
        print(f"\nRecent datasets (last 10):")
        print(f"  {'Accession':<15} {'Source':<8} {'Platform':<8} {'Status':<12} {'Year'}")
        print(f"  {'-'*60}")
        for ds in datasets[-10:]:
            print(
                f"  {ds.get('accession',''):<15} "
                f"{ds.get('source',''):<8} "
                f"{(ds.get('platform') or '-'):<8} "
                f"{ds.get('download_status',''):<12} "
                f"{ds.get('year') or '-'}"
            )


def _dry_run(query: str, config: dict):
    """Parse query and show intent without downloading."""
    from tools.parser_tools import parse_query_rules, parse_query_with_llm
    from utils.llm_factory import get_llm

    print(f"\n[DRY RUN] Parsing query: {query}\n")

    # Rule-based parse
    rule_intent = parse_query_rules(query)
    print("Rule-based parse result:")
    print(json.dumps(rule_intent, ensure_ascii=False, indent=2))

    # LLM parse (if API key available)
    api_key_env = config["llm"].get("api_key_env", "")
    if os.environ.get(api_key_env):
        print("\nLLM-enhanced parse result:")
        try:
            llm = get_llm(config["llm"])
            llm_intent = parse_query_with_llm(query, llm)
            print(json.dumps(llm_intent, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"  LLM parse failed: {e}")
    else:
        print(f"\n[NOTE] Set {api_key_env} to enable LLM-enhanced parsing.")


def _check_api_key(config: dict):
    """Warn if LLM API key is not set."""
    api_key_env = config["llm"].get("api_key_env", "")
    backend = config["llm"].get("backend", "openai")

    if backend != "ollama" and not os.environ.get(api_key_env):
        print(f"[WARNING] {api_key_env} is not set.")
        print(f"  The system will use rule-based query parsing (no LLM).")
        print(f"  Set the environment variable for full LLM-powered parsing:\n")
        print(f"    export {api_key_env}=your_api_key_here\n")


def _print_summary_v2(report: dict):
    """Print a human-readable summary of an orchestrator_v2 (agentic) run."""
    print(f"\n{'='*60}")
    print(f"  MethyAgent (v2, agentic) Completed")
    print(f"{'='*60}")
    print(f"\nQuery: {report.get('query', '')}")
    print(f"\nResults:")
    print(f"  Papers found        : {report.get('papers_found', 0)}")
    print(f"  GEO datasets evaluated : {len(report.get('gse_evaluated', []))}")
    print(f"  Registry writes     : {len(report.get('registry_writes', []))}")
    if report.get("agent_summary"):
        print(f"\nAgent summary:\n  {report['agent_summary']}")
    if report.get("log_path"):
        print(f"\nRun log saved: {report['log_path']}")


def _print_summary(state: dict):
    """Print a human-readable summary of the run."""
    report = state.get("final_report", {})
    summary = report.get("summary", {})
    a1 = report.get("agent1", {})
    a2 = report.get("agent2", {})

    print(f"\n{'='*60}")
    print(f"  MethyAgent Completed")
    print(f"{'='*60}")
    print(f"\nQuery: {report.get('query', '')}")
    print(f"\nResults:")
    print(f"  Total datasets in registry : {summary.get('total', 0)}")
    print(f"  Agent 1 (Database) found   : {summary.get('agent1_discovered', 0)}")
    print(f"  Agent 2 (Literature) added : {summary.get('agent2_discovered', 0)}")
    print(f"  Successfully downloaded    : {summary.get('by_status', {}).get('done', 0)}")
    print(f"  Failed                     : {summary.get('by_status', {}).get('failed', 0)}")

    output_dir = Path(state.get("config", {}).get("download", {}).get("output_dir", "./data"))
    reports = sorted(output_dir.glob("report_*.md"))
    if reports:
        print(f"\nReport saved: {reports[-1]}")

    errors = state.get("error_log", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more (see report JSON)")


if __name__ == "__main__":
    main()
