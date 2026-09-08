#!/usr/bin/env python3
"""
Build the gold-standard GSE pool via deliberately broad, query-independent
GEO crawls (gold/pool/strata_v1.yaml → gold/pool/pool_v1.jsonl).

The pool MUST NOT be built with the agent's own synonym-expanded search
(tools/parser_tools.build_geo_search_string) — otherwise it inherits the
search layer's blind spots and search recall is measured against itself.
Instead each stratum uses a raw esearch term + fixed-seed random sampling.

Usage (on SSH server, proxy must be running):
    set -a && source .env && set +a
    export NCBI_PROXY=socks5://127.0.0.1:1080
    source .venv/bin/activate
    python scripts/build_gold_pool.py                     # full pool
    python scripts/build_gold_pool.py --limit-per-stratum 2 --out gold/pool/pool_smoke.jsonl

Pool immutability: once pool_v1.jsonl exists it is NEVER regenerated (refuse
without --force) — labels and truth are derived from it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Make project root importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.geo_tools import GEOClient
from utils.logger import get_logger

logger = get_logger(__name__)

STRATA_FILE = _ROOT / "gold" / "pool" / "strata_v1.yaml"
POOL_FILE = _ROOT / "gold" / "pool" / "pool_v1.jsonl"

# How many UIDs to pull per stratum before random-sampling down to n_target.
# esearch relevance order is NOT the agent's ranking — sampling from the top
# 1000 is a documented, acceptable bias (see docs/gold_standard_design.md).
SEARCH_WINDOW = 1000


def crawl_stratum(geo: GEOClient, stratum: Dict[str, Any], seed: int,
                  limit: int | None) -> Dict[str, Any]:
    """Run one stratum's esearch (or explicit accession list) → sample.
    Returns a result dict with either 'accessions' or 'error'."""
    name = stratum["name"]
    term = stratum.get("esearch_term", "")
    n_target = limit if limit is not None else int(stratum["n_target"])

    # Explicit-accession stratum (mode: explicit) — no esearch, no sampling.
    # Used for agent-verified positives / canonical cases the broad strata missed.
    if stratum.get("mode") == "explicit":
        accessions = [str(a).upper() for a in stratum.get("accessions", [])
                      if str(a).upper().startswith("GSE")]
        if not accessions:
            return {"stratum": name, "error": "explicit stratum with no valid accessions"}
        return {"stratum": name, "esearch_term": "(explicit list)",
                "n_found": len(accessions), "accessions": accessions}

    try:
        found = geo.search_gse(term, max_results=SEARCH_WINDOW)
    except Exception as e:  # noqa: BLE001 — one stratum failing must not kill the pool
        return {"stratum": name, "error": f"esearch failed: {e}"}

    # search_gse retries without the GSE filter when the primary returns empty;
    # that fallback can return non-GSE accessions. Guard.
    accessions = [a for a in found if str(a).upper().startswith("GSE")]
    if not accessions:
        return {"stratum": name, "error": f"esearch returned 0 GSE accessions "
                                          f"({len(found)} raw results)"}
    if len(accessions) < n_target:
        logger.warning(f"stratum {name}: only {len(accessions)} accessions "
                       f"(< n_target={n_target}) — taking all")

    rng = random.Random(f"{seed}:{name}")  # stratum-stable seeding
    if len(accessions) > n_target:
        sampled = rng.sample(accessions, n_target)
    else:
        sampled = list(accessions)

    return {
        "stratum": name,
        "esearch_term": term,
        "n_found": len(accessions),
        "accessions": sampled,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the gold GSE pool from broad strata.")
    ap.add_argument("--strata-file", default=str(STRATA_FILE))
    ap.add_argument("--out", default=str(POOL_FILE))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-stratum", type=int, default=None,
                    help="Smoke mode: cap each stratum at N GSE.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing pool file.")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"[pool] {out_path} already exists — pools are immutable once built. "
              f"Use --force (and a new --out) if you really mean it.")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.strata_file, encoding="utf-8") as f:
        strata_cfg = yaml.safe_load(f)
    strata = strata_cfg.get("strata", [])
    seed = args.seed if args.seed != 42 else int(strata_cfg.get("seed", 42))

    # Factorial config (strata_v2-factorial.yaml): expand cancers × specimens into
    # per-cell strata. Each cell's term is narrowed enough that total hits < the
    # 1000-UID window, so within-cell random sampling is uniform over the WHOLE
    # combination — the fix for v1's relevance-window bias.
    if "cancers" in strata_cfg and "specimens" in strata_cfg:
        cancers = strata_cfg["cancers"]
        specimens = strata_cfg["specimens"]
        n_default = int(strata_cfg.get("n_target_default", 8))
        overrides = strata_cfg.get("n_target_overrides", {}) or {}
        factorial = []
        for cancer_name, cancer_term in cancers.items():
            for spec_name, spec_term in specimens.items():
                cell_name = f"{cancer_name}__{spec_name}"
                factorial.append({
                    "name": cell_name,
                    "esearch_term": (
                        f"methylation[All Fields] AND {cancer_term} AND {spec_term} "
                        f"AND Homo sapiens[orgn]"
                    ),
                    "n_target": int(overrides.get(f"{cancer_name}|{spec_name}", n_default)),
                })
        factorial.extend(strata_cfg.get("extra_cells", []))
        # factorial cells come first so cell provenance wins cross-stratum dedup
        strata = factorial + strata
        print(f"[pool] factorial expansion: {len(factorial)} cells "
              f"({len(cancers)} cancers × {len(specimens)} specimens + extras)")

    # --- GEO client, constructed the same way as agent1_pipeline ---
    import os
    with open(_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    geo_cfg = config.get("geo", {})
    api_key = os.environ.get(geo_cfg.get("api_key_env", ""), "") or None
    proxy = os.environ.get("NCBI_PROXY", "") or geo_cfg.get("proxy", "") or None
    geo = GEOClient(api_key=api_key, proxy=proxy)

    # --- crawl each stratum ---
    results: List[Dict[str, Any]] = []
    for stratum in strata:
        logger.info(f"[pool] crawling stratum {stratum['name']} ...")
        results.append(crawl_stratum(geo, stratum, seed, args.limit_per_stratum))

    ok = [r for r in results if "accessions" in r]
    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"[pool] ⚠ stratum {r['stratum']} FAILED: {r['error']}")

    # --- cross-stratum dedup: first stratum wins ---
    # Each row records its stratum's sampling frame (n_found) and kept count so
    # downstream recall can be inverse-probability weighted / stratified: the
    # factorial pool samples cells at DIFFERENT rates (e.g. 12/33 plasma vs
    # 8/620 tissue), and unweighted pooled recall would over-represent
    # densely-sampled cells. (docs/gold_standard_pool_gap.md §5.5)
    stratum_stats: Dict[str, Dict[str, int]] = {
        r["stratum"]: {"n_found": r.get("n_found", 0), "n_kept": len(r["accessions"])}
        for r in ok
    }
    seen: Dict[str, str] = {}  # accession → stratum
    order: List[str] = []
    for r in ok:
        for acc in r["accessions"]:
            if acc not in seen:
                seen[acc] = r["stratum"]
                order.append(acc)
    print(f"[pool] {len(ok)}/{len(results)} strata ok, "
          f"{len(seen)} unique GSE (after cross-stratum dedup)")

    if not seen:
        print("[pool] no accessions at all — aborting")
        return 1

    # --- batch metadata (title/sample_count/data_type/year/platforms) ---
    # Per-accession failures (NCBI abuse redirect etc.) degrade to a metadata-less
    # pool row rather than killing the crawl — the labeling step re-fetches
    # metadata per GSE anyway.
    meta_by_acc: Dict[str, Dict[str, Any]] = {}
    accessions = list(seen.keys())
    try:
        for meta in geo.batch_get_series_metadata(accessions):
            acc = (meta.get("accession") or "").upper()
            if acc:
                meta_by_acc[acc] = meta
    except Exception as e:  # noqa: BLE001 — NCBI abuse redirect: write pool without metadata
        print(f"[pool] ⚠ batch metadata failed ({e}) — writing pool rows without "
              f"metadata (labeling re-fetches per GSE)")
    n_meta_fail = sum(1 for m in meta_by_acc.values() if m.get("error"))
    print(f"[pool] metadata fetched for {len(meta_by_acc)}/{len(accessions)}"
          + (f" ({n_meta_fail} with errors — rows keep metadata_error)" if n_meta_fail else ""))

    # --- write pool ---
    with open(out_path, "w", encoding="utf-8") as f:
        for acc in order:
            meta = meta_by_acc.get(acc, {})
            rec = {
                "accession": acc,
                "stratum": seen[acc],
                "esearch_term": next(r["esearch_term"] for r in ok if r["stratum"] == seen[acc]),
                # sampling frame for this stratum: n_found hits in the esearch
                # window, n_kept sampled → inclusion prob = n_kept/n_found (for
                # uniform-outcome weighting of pool-conditioned recall).
                "stratum_n_found": stratum_stats[seen[acc]]["n_found"],
                "stratum_n_kept": stratum_stats[seen[acc]]["n_kept"],
                "seed": seed,
                "selected_at": date.today().isoformat(),
                "title": meta.get("title", ""),
                "sample_count": meta.get("sample_count"),
                "data_type": meta.get("data_type"),
                "year": meta.get("year"),
                "platforms": meta.get("platforms", []),
                "pubmed_ids": meta.get("pubmed_ids", []),
            }
            if meta.get("error"):
                rec["metadata_error"] = meta["error"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- summary ---
    from collections import Counter
    by_stratum = Counter(seen.values())
    print(f"[pool] wrote {len(order)} GSE → {out_path}")
    for r in results:
        if "accessions" in r:
            kept = by_stratum.get(r["stratum"], 0)
            print(f"  {r['stratum']:20s} found={r['n_found']:5d} kept={kept}")
        else:
            print(f"  {r['stratum']:20s} FAILED")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
