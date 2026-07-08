#!/usr/bin/env python3
"""
Debug harness for the geo_filter skill — judge ONE dataset, fast.

Takes a query + one or more GSE accessions and runs the EXACT same path as
production (parse_query → series metadata → representative GSMs → optional
PubMed abstract → filter_dataset), then prints the full reasoning chain,
the evidence the model was given, and the token cost.

Use this to iterate on skills/geo_filter/SPEC.md without waiting ~10 minutes
to re-run a full 100+ dataset batch:

    .venv/bin/python scripts/debug_geo_filter.py \\
        --query "colorectal cancer和非癌对照的cfDNA甲基化数据" \\
        --accession GSE124600

    # multiple accessions at once:
    --accession GSE124600 GSE110185 GSE79277

    # skip the PubMed abstract fetch:
    --no-abstract

    # also write a one-row-per-accession CSV (same format as the prod log):
    --csv

Reads config from config/settings.yaml. Honours NCBI_PROXY env / geo.proxy
(so it can reuse the host's ssh -D 1080 tunnel) and the LLM keys from .env.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Make the project root importable when run as `python scripts/debug_geo_filter.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from skills.geo_filter import SPEC_NAME, filter_dataset
from skills.geo_filter.grouping import group_summary
from tools.geo_tools import GEOClient
from tools.parser_tools import parse_query_rules, parse_query_with_llm
from utils.llm_factory import get_llm
from utils.logger import get_logger
from utils.query_logger import QueryLogger

logger = get_logger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hr(char: str = "=", width: int = 78) -> str:
    return char * width


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug geo_filter on a single dataset.")
    ap.add_argument("--query", required=True, help="Natural-language query (same as Web UI).")
    ap.add_argument("--accession", nargs="+", required=True,
                    help="One or more GSE accessions, e.g. GSE124600 GSE110185.")
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--proxy", default=None,
                    help="Override NCBI proxy, e.g. socks5h://127.0.0.1:1080.")
    ap.add_argument("--no-abstract", action="store_true",
                    help="Skip fetching the PubMed abstract.")
    ap.add_argument("--csv", action="store_true",
                    help="Also write a CSV (one row per accession) via QueryLogger.")
    ap.add_argument("--use-rules-parser", action="store_true",
                    help="Parse the query with the rule-based parser (no LLM call).")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show INFO/DEBUG library logs (default: WARNING only, clean output).")
    args = ap.parse_args()

    # Keep the printed output clean unless --verbose.
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    config = load_config(args.config)

    # ---- LLM ----
    llm = get_llm(config["llm"])
    model_name = (
        getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    )

    # ---- GEO client (mirror DatabaseAgent: env NCBI_PROXY > settings geo.proxy) ----
    ncbi_key = os.environ.get(config["geo"].get("api_key_env", ""), "") or None
    ncbi_proxy = args.proxy or os.environ.get("NCBI_PROXY", "") or config.get("geo", {}).get("proxy", "") or None
    geo = GEOClient(api_key=ncbi_key or None, proxy=ncbi_proxy or None)

    # ---- Parse query → intent ----
    print(_hr())
    print(f"QUERY: {args.query}")
    print(f"MODEL: {model_name}    SPEC: {SPEC_NAME}    PROXY: {ncbi_proxy or '(none)'}")
    print(_hr())
    if args.use_rules_parser:
        intent = parse_query_rules(args.query)
        print("[intent] parsed via RULES")
    else:
        try:
            intent = parse_query_with_llm(args.query, llm)
            print("[intent] parsed via LLM")
        except Exception as e:
            print(f"[intent] LLM parse failed ({e}), falling back to RULES")
            intent = parse_query_rules(args.query)
    intent["raw_query"] = args.query
    wanted = intent.get("sample_type", "") or ""
    print(f"  cancer_type : {intent.get('cancer_type')}")
    print(f"  sample_type : {wanted}  (all: {intent.get('sample_types')})")
    print(f"  platform    : {intent.get('platform')}")
    print()

    qlog: "QueryLogger | None" = None
    if args.csv:
        qlog = QueryLogger(
            query=args.query,
            model_name=model_name,
            spec_name=SPEC_NAME,
            output_dir=config.get("download", {}).get("output_dir", "./data/methylation"),
        )

    rc = 0
    for acc in args.accession:
        ok = run_one(acc, geo, llm, intent, wanted, fetch_abstract=not args.no_abstract, qlog=qlog)
        if not ok:
            rc = 1

    if qlog is not None:
        path = qlog.finalize()
        print(_hr("-"))
        print(f"CSV written: {path}")

    return rc


def run_one(
    accession: str,
    geo: GEOClient,
    llm: Any,
    intent: Dict[str, Any],
    wanted_sample_type: str,
    fetch_abstract: bool,
    qlog: "QueryLogger | None",
) -> bool:
    """Run the full geo_filter path on one accession and print the reasoning."""
    print(_hr())
    print(f"ACCESSION: {accession}")
    print(_hr())
    t0 = time.time()

    # 1) series metadata
    try:
        ds = geo.get_series_metadata(accession)
    except Exception as e:
        print(f"  [ERROR] get_series_metadata failed: {e}")
        return False
    if ds.get("error") or not ds.get("accession"):
        print(f"  [ERROR] no metadata for {accession}: {ds.get('error', 'unknown')}")
        return False

    print(f"  title       : {(ds.get('title') or '')[:100]}")
    print(f"  platform    : {ds.get('platform_canonical') or ds.get('platforms')}")
    print(f"  sample_count: {ds.get('sample_count')}")
    print(f"  pubmed_ids  : {ds.get('pubmed_ids', [])}")
    print(f"  summary     : {(ds.get('summary') or '')[:160]}...")
    print()

    # 2) representative GSMs
    gsm_details = geo.get_representative_gsm_details(accession, wanted_sample_type=wanted_sample_type)
    groups = group_summary(gsm_details)
    groups_str = ", ".join(f"{g}={n}" for g, n in groups.items() if n) or "(none)"
    print(f"[evidence] representative GSMs: {len(gsm_details)}  (groups: {groups_str})")
    for g in gsm_details[:6]:
        ch = g.get("characteristics") or {}
        ch_str = "; ".join(f"{k}={v}" for k, v in list(ch.items())[:4])
        print(f"    {g.get('gsm')} [{g.get('group','?')}]: "
              f"source={g.get('source_name','')!r} mol={g.get('molecule','')!r} {ch_str}")
    if len(gsm_details) > 6:
        print(f"    ... ({len(gsm_details) - 6} more)")
    print()

    # 3) optional abstract
    abstract = None
    pmids = ds.get("pubmed_ids") or []
    if fetch_abstract and pmids:
        try:
            abstract = geo.fetch_pubmed_abstract(str(pmids[0]))
            print(f"[evidence] PubMed abstract (PMID {pmids[0]}): "
                  f"{len(abstract)} chars" if abstract else
                  f"[evidence] PubMed abstract (PMID {pmids[0]}): UNAVAILABLE")
        except Exception as e:
            print(f"[evidence] PubMed abstract fetch failed: {e}")
    else:
        print("[evidence] PubMed abstract: skipped (no PMID or --no-abstract)")
    print()

    # 4) filter_dataset → verdict (the single LLM call)
    verdict = filter_dataset(llm, ds, intent, gsm_details, abstract=abstract)
    elapsed = time.time() - t0
    usage = verdict.get("_usage") or {}

    # 5) print the verdict + full reasoning chain
    print(_hr("."))
    print(f"VERDICT: {verdict.get('recommended_action')}  "
          f"(usable={verdict.get('usable')}, sample={verdict.get('confirmed_sample_type')})")
    print(_hr("."))
    print(f"REASON : {verdict.get('reason')}")
    print()
    print("REASONING CHAIN:")
    print(_hr("-"))
    reasoning = verdict.get("reasoning") or "(model returned no reasoning)"
    print(reasoning if reasoning.strip() else "(empty)")
    print(_hr("-"))
    print()
    if verdict.get("notes"):
        print(f"NOTES  : {verdict['notes']}")
    print()
    print(f"TOKENS : prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
          f"total={usage.get('total_tokens')} cached={usage.get('cached_tokens')} "
          f"api_model={usage.get('api_model')}")
    print(f"TIME   : {elapsed:.1f}s")
    print()

    if qlog is not None:
        qlog.log_dataset(ds, verdict)

    return True


if __name__ == "__main__":
    sys.exit(main())
