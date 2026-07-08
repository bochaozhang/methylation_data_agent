"""
P1-2 — CRC cfDNA methylation biomarker recall test (docs/07_03).

NOT runnable in this sandbox: requires a real LLM API key (search_and_extract's
Stage 2 + reviewer both call llm.invoke()) and outbound NCBI network access,
neither available here. Written so it's ready to run as-is on the SSH server:

    set -a && source .env && set +a
    source .venv/bin/activate
    export HTTPS_PROXY=... HTTP_PROXY=... ALL_PROXY=... NCBI_PROXY=...
    bash /home/ubuntu/bochaozhang/proxy.sh
    python -m scripts.biomarker_recall_test

Also note: mtg/07:02/03.文章标志物整理.xlsx (sheet "CRC-仅基因"), referenced as
the ground truth to diff against, is not in this repo — it needs to be supplied
separately (e.g. dropped under docs/07_03/) before the overlap report can run;
this script prints the found markers either way so that comparison can be done
by hand if the xlsx isn't available yet.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ncbi_search import search_and_extract
from tools.parser_tools import parse_query_rules
from utils.llm_factory import get_llm
import yaml

QUERY = (
    "colorectal cancer blood-based cfDNA DNA methylation biomarker studies"
)

KNOWN_CRC_MARKERS = [
    "SFRP2", "VIM", "MGMT", "SEPTIN9", "APC", "NDRG4",
    "CDKN2A", "MLH1", "SDC2", "BMP3", "TFPI2", "HLTF", "WIF1",
]

# Optional: path to the ground-truth xlsx, if present locally.
GOLD_XLSX = Path(__file__).parent.parent / "docs" / "07_03" / "文章标志物整理.xlsx"
GOLD_SHEET = "CRC-仅基因"


def _extract_marker_names(paper: dict) -> list:
    names = []
    for m in paper.get("markers_or_panel") or []:
        if isinstance(m, dict):
            gene = m.get("gene") or m.get("id")
            if gene:
                names.append(str(gene).upper())
        elif isinstance(m, str):
            names.append(m.upper())
    return names


def main() -> None:
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "settings.yaml"))
    llm = get_llm(cfg["llm"])

    intent = parse_query_rules(QUERY)
    print(f"Query: {QUERY}")
    papers = search_and_extract(intent, llm, top_n=10)
    print(f"\n{len(papers)} structured papers returned.\n")

    found_counter: Counter = Counter()
    for paper in papers:
        markers = _extract_marker_names(paper)
        print(f"  PMID={paper.get('pmid')}  markers_or_panel={markers}")
        found_counter.update(markers)

    print(f"\n{'=' * 60}")
    print("Known high-frequency CRC methylation marker recall")
    print(f"{'=' * 60}")
    hits = [m for m in KNOWN_CRC_MARKERS if m in found_counter]
    misses = [m for m in KNOWN_CRC_MARKERS if m not in found_counter]
    for m in KNOWN_CRC_MARKERS:
        mark = "✓" if m in found_counter else " "
        print(f"  [{mark}] {m:10s} (mentions: {found_counter.get(m, 0)})")
    print(f"\nRecall: {len(hits)}/{len(KNOWN_CRC_MARKERS)} known markers found "
          f"across {len(papers)} papers.")

    if GOLD_XLSX.exists():
        try:
            import pandas as pd
            gold_df = pd.read_excel(GOLD_XLSX, sheet_name=GOLD_SHEET)
            print(f"\nLoaded gold standard from {GOLD_XLSX} (sheet {GOLD_SHEET!r}), "
                  f"{len(gold_df)} rows — cross-check columns manually, schema unknown.")
        except Exception as e:
            print(f"\n[WARN] Could not read gold xlsx: {e}")
    else:
        print(f"\n[NOTE] Gold-standard xlsx not found at {GOLD_XLSX} — "
              f"copy it there (or point GOLD_XLSX at its real path) to auto-diff "
              f"against sheet {GOLD_SHEET!r}.")

    out_path = Path(__file__).parent.parent / "docs" / "07_03" / "biomarker_recall_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "query": QUERY,
            "papers_found": len(papers),
            "known_markers_checked": KNOWN_CRC_MARKERS,
            "found": hits,
            "missing": misses,
            "mention_counts": dict(found_counter),
            "papers": papers,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
