"""
Standalone prototype: search → GEO evaluation → registry write.

Chains three pieces that already exist but are NOT wired together in
agents/orchestrator.py or agents/literature_agent.py:
    search_and_extract()   (tools/ncbi_search.py)
        -> evaluate_geo_dataset()   (tools/query_clarifier.py)
        -> registry.upsert_dataset() (registry/registry.py)

Does NOT touch agents/orchestrator.py or agents/literature_agent.py — this is
a separate, standalone path for demoing/validating the missing pipeline steps
(items 4-6 from the 2026-06-24 meeting) without risking the teammate's
already-working orchestrator flow. See docs/6_30/orchestrator_integration_finding_2026-06-30.md
for the integration decision this still needs.

Usage (on SSH server):
    set -a && source .env && set +a
    source .venv/bin/activate
    export HTTPS_PROXY=... HTTP_PROXY=... ALL_PROXY=... NCBI_PROXY=...
    bash /home/ubuntu/bochaozhang/proxy.sh
    python scripts/pipeline_prototype.py "结直肠癌血浆cfDNA甲基化，需要健康对照"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.geo_tools import GEOClient
from tools.ncbi_search import search_and_extract
from tools.parser_tools import parse_query_rules
from tools.query_clarifier import evaluate_geo_dataset
from registry.registry import Registry
from utils.llm_factory import get_llm


def _build_dataset_info(accession: str, meta: Dict[str, Any]) -> str:
    """Format GEOClient.get_series_metadata() output as free text for evaluate_geo_dataset()."""
    lines = [f"{accession} — Title: {meta.get('title', '(no title)')}"]
    if meta.get("summary"):
        lines.append(f"Summary: {meta['summary']}")
    if meta.get("sample_titles"):
        lines.append("Sample titles: " + "; ".join(meta["sample_titles"]))
    if meta.get("platform_canonical") or meta.get("platforms"):
        lines.append(f"Platform: {meta.get('platform_canonical') or meta.get('platforms')}")
    if meta.get("sample_count"):
        lines.append(f"Sample count: {meta['sample_count']}")
    if meta.get("data_type"):
        lines.append(f"Data type: {meta['data_type']}")
    return "\n".join(lines)


def run_prototype(query: str, top_n: int = 5) -> None:
    cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    llm = get_llm(cfg["llm"])

    ncbi_key = os.environ.get(cfg["geo"].get("api_key_env", ""), "")
    ncbi_proxy = os.environ.get("NCBI_PROXY", "") or cfg.get("geo", {}).get("proxy", "")
    geo_client = GEOClient(api_key=ncbi_key or None, proxy=ncbi_proxy or None)
    registry = Registry(db_path=cfg["registry"]["db_path"])

    print(f"[setup] backend={cfg['llm']['backend']}  NCBI_PROXY={ncbi_proxy or '(not set)'}\n")

    # --- Step 1: search + Stage1/Stage2 extraction ---
    intent = parse_query_rules(query)
    print(f"Query: {query}")
    papers = search_and_extract(intent, llm, top_n=top_n)
    print(f"\n{len(papers)} structured papers returned.\n")

    # --- Step 2 + 3: evaluate each unique GSE, write kept/reviewed ones to registry ---
    seen_gse: set = set()
    counts = {"keep": 0, "exclude": 0, "manual_review": 0, "article_only": 0, "error": 0}

    for paper in papers:
        pmid = paper.get("pmid", "")
        cancer_type = paper.get("cancer_type") or "unknown"
        sample_type = paper.get("sample_type")
        sample_types = [sample_type] if sample_type else []

        for ds_id in (paper.get("dataset_ids") or []):
            if not isinstance(ds_id, str) or not ds_id.upper().startswith("GSE"):
                continue  # this prototype only evaluates GEO series; skip TCGA/other IDs
            if ds_id in seen_gse:
                continue
            seen_gse.add(ds_id)

            meta = geo_client.get_series_metadata(ds_id)
            if meta.get("error"):
                print(f"  [skip] {ds_id}: {meta['error']}")
                counts["error"] += 1
                continue

            dataset_info = _build_dataset_info(ds_id, meta)
            judgment = evaluate_geo_dataset(dataset_info, cancer_type, sample_types, llm)
            action = judgment.get("recommended_action", "manual_review")
            counts[action] = counts.get(action, 0) + 1

            print(f"  GSE={ds_id}  (from PMID={pmid})  action={action}")
            print(f"    reason: {(judgment.get('reason') or '')[:150]}")

            registry.upsert_dataset(
                accession=ds_id,
                source="GEO",
                discovered_by="pipeline_prototype",
                data_type=meta.get("data_type"),
                cancer_type=cancer_type,
                platform=meta.get("platform_canonical"),
                sample_count=meta.get("sample_count"),
                year=meta.get("year"),
                title=meta.get("title"),
                paper_pmid=pmid,
                download_status="pending",
                needs_review=(action == "manual_review"),
                llm_evidence=judgment.get("reason"),
                sample_type=sample_type,
                usable=1 if action in ("keep", "manual_review") else 0,
                recommended_action=action,
                reason=judgment.get("reason"),
            )
            print(f"    -> written to registry (needs_review={action == 'manual_review'})")

    print(f"\n{'=' * 60}")
    print("Prototype summary")
    print('=' * 60)
    print(f"  Unique GSE evaluated: {len(seen_gse)}")
    for action, n in counts.items():
        print(f"    {action:15s} {n}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "结直肠癌血浆cfDNA甲基化，需要健康对照"
    run_prototype(query)
