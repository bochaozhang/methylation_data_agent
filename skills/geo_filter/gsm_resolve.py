"""
GSM-evidence resolver — picks the cheapest complete source of per-sample metadata.

Both filter pipelines (agents/agent1_pipeline.py and agents/database_agent.py skill
path) need per-GSM metadata to feed skills.geo_filter.filter_dataset(), which
deduplicates the samples and makes ONE LLM call. The evidence can come from three
sources, in order of preference:

  1. series_matrix  — one download yields ALL samples with structured fields
                      (the "Path A" the user wants the no-series_matrix path to
                      mirror). Cheapest and most complete when available.
  2. JSON cache     — {output_dir}/{accession}/gsm_metadata_cache.json, an exact
                      snapshot of a prior get_all_gsm_metadata() result. Lets a
                      re-run skip the O(N) efetch.
  3. get_all_gsm_metadata — efetch MiniML for EVERY GSM (the new "Path B" default,
                      replacing the old representative sample of ≤30). Bounded for
                      very large series by a soft cap that falls back to
                      representative sampling.

The cache is a dedicated JSON file (not sample_metadata.csv) because the two
existing CSV writers have different, non-round-trippable schemas for
`characteristics` — reconstructing from them would mis-classify the cancer/task/
query columns. JSON gives an exact round-trip and keeps sample_metadata.csv as the
human-readable verdict artifact.

No new network code: this only wires together existing GEOClient methods.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CAP = 600


def _cache_path(output_dir: str, accession: str) -> Path:
    return Path(output_dir) / accession / "gsm_metadata_cache.json"


def read_gsm_cache(cache_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Return the cached gsm_list, or None if missing/corrupt/unreadable.

    None (not []) signals "no cache — fall through to fetch", so an empty-but-
    valid cached list is distinguishable from a cache miss. In practice the list
    is never empty when written (get_all_gsm_metadata returns [] only on failure,
    which we don't cache).
    """
    try:
        if not cache_path.exists():
            return None
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning(f"read_gsm_cache({cache_path}): unexpected payload type — ignoring")
        return None
    except Exception as e:  # corrupt JSON, permission, ...
        logger.debug(f"read_gsm_cache({cache_path}): failed ({e}) — ignoring")
        return None


def write_gsm_cache(cache_path: Path, gsm_list: List[Dict[str, Any]]) -> None:
    """Atomically persist the gsm_list so a concurrent reader never sees a partial file.

    Best-effort: a write failure is logged but never raised (the caller already has
    the in-memory list and can proceed; only the next run's cache benefit is lost).
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(gsm_list, f, ensure_ascii=False)
        os.replace(tmp, cache_path)
    except Exception as e:  # noqa: BLE001 — never break filtering over a cache write
        logger.warning(f"write_gsm_cache({cache_path}): failed ({e}) — proceeding without cache")


def resolve_gsm_details(
    geo_client: Any,
    accession: str,
    ds: Dict[str, Any],
    output_dir: str,
    wanted_sample_type: str = "",
    max_all_fetch: int = _DEFAULT_CAP,
) -> List[Dict[str, Any]]:
    """
    Resolve per-sample GSM metadata via the cheapest complete source.

    Policy (first non-empty source wins):
      1. series_matrix            — geo_client.fetch_series_matrix_sample_info(accession)
      2. JSON cache hit           — {output_dir}/{accession}/gsm_metadata_cache.json
      3. over the soft cap        — representative sampling fallback (avoids O(N) efetch
                                    on very large series); logged, NOT cached
      4. otherwise                — geo_client.get_all_gsm_metadata(accession), then
                                    the result is cached for future runs

    Args:
        geo_client:        GEOClient instance (or any duck-typed equivalent).
        accession:         GSE accession, e.g. 'GSE124600'.
        ds:                dataset metadata dict; only ds.get('sample_count') is read
                           for the soft-cap check (already populated by the search step).
        output_dir:        root data dir; the cache lives under {output_dir}/{accession}/.
        wanted_sample_type: sample type hint for the representative fallback only.
        max_all_fetch:     soft cap on sample_count above which Path B uses representative
                           sampling instead of efetching every GSM.

    Returns:
        List of dicts with keys gsm, source_name, molecule, characteristics (dict),
        group — the schema filter_dataset / _dedup_gsm_combos expect. Returns [] only
        if every source fails (filter_dataset treats [] as "no sample details").
    """
    cache_path = _cache_path(output_dir, accession)

    # 1. series_matrix — one download, all samples, structured fields.
    try:
        sm = geo_client.fetch_series_matrix_sample_info(accession)
    except Exception as e:  # noqa: BLE001 — network/head failure → fall through
        logger.debug(f"resolve_gsm_details({accession}): series_matrix fetch failed ({e})")
        sm = None
    if sm:
        logger.info(f"resolve_gsm_details({accession}): series_matrix {len(sm)} samples")
        return sm

    # 2. cache hit — skip the O(N) efetch on re-runs.
    cached = read_gsm_cache(cache_path)
    if cached:
        logger.info(f"resolve_gsm_details({accession}): cache hit {len(cached)} samples")
        return cached

    # 3. soft cap — very large series: representative sample instead of efetch-all.
    n = ds.get("sample_count")
    try:
        n_int = int(n) if n is not None else 0
    except (TypeError, ValueError):
        n_int = 0
    if max_all_fetch and n_int and n_int > max_all_fetch:
        rep = geo_client.get_representative_gsm_details(accession, wanted_sample_type=wanted_sample_type)
        logger.info(
            f"resolve_gsm_details({accession}): {n_int} samples > cap {max_all_fetch} "
            f"→ representative fallback ({len(rep)} GSMs)"
        )
        return rep

    # 4. full efetch — every GSM, then cache.
    gsm = geo_client.get_all_gsm_metadata(accession)
    logger.info(f"resolve_gsm_details({accession}): efetch-all {len(gsm)} samples")
    if gsm:
        write_gsm_cache(cache_path, gsm)
    return gsm
