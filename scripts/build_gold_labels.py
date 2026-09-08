#!/usr/bin/env python3
"""
Label every sample combo of each pool GSE with FACTUAL tags (gold standard).

Key design points (docs/gold_standard_design.md):
  * Labels are facts, never verdicts — a combo is "plasma cfDNA, CRC, case",
    NOT "download". Verdict semantics live only in gold/queries.yaml +
    scripts/derive_truth.py, so SPEC.md edits never invalidate labels.
  * Evidence = exactly what the filter sees: series metadata + full-sample GSM
    details (series_matrix first) + optional abstract. Each combo records
    `decidable_from_metadata` so filter scoring can split fair / reference tiers.
  * One LLM call per GSE (all combos in one shot, cross-combo context);
    combos are chunked at 25/call to stay under the output cap.
  * SQLite cache keyed (accession, model) → resume-safe, and the deepseek
    cross-validation run is just `--model deepseek-chat --backend deepseek`.

Usage (proxy + .env as in build_gold_pool.py):
    python scripts/build_gold_labels.py --pool gold/pool/pool_v1.jsonl
    python scripts/build_gold_labels.py --limit 3          # smoke
    python scripts/build_gold_labels.py --export           # → labels_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skills.geo_filter.gsm_resolve import resolve_gsm_details
from skills.geo_filter.skill import _dedup_gsm_combos, _extract_usage, _safe_json
from tools.geo_tools import GEOClient
from utils.llm_factory import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)

LABELS_DIR = _ROOT / "gold" / "labels" / "v1"
EVIDENCE_DIR = _ROOT / "gold" / "evidence_cache"
CHUNK_SIZE = 25            # max combos per LLM call
MAX_ALL_FETCH = 1500       # resolve_gsm_details soft cap (above → partial coverage)
LABELER_MAX_TOKENS = 8192  # > settings llm.max_tokens (4096) — JSON with many combos
LLM_TIMEOUT = 300          # seconds per LLM call; glm occasionally hangs on large
                           # prompts (GSE85356: 31min no-response) — timeout+retry
                           # instead of waiting indefinitely.

# Daemon thread pool for timeout-wrapped LLM calls. Module-level singleton:
# worker threads are daemons so a timed-out (still-running) call never blocks
# interpreter exit.
_LLM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-call")


# ---------------------------------------------------------------------- #
#  Labeler prompt                                                         #
# ---------------------------------------------------------------------- #

LABELER_SYSTEM_PROMPT = """\
You are a meticulous biocurator annotating GEO datasets for a cancer-early-detection
methylation data benchmark. You label FACTS about samples — you NEVER judge whether
a dataset should be downloaded or is "usable". Another process owns that decision.

For each dataset you receive: GEO series metadata, the deduplicated sample combos
(groups of biologically identical GSM samples), and optionally a PubMed abstract.

Return per-dataset labels and per-combo labels:

DATASET-LEVEL:
  data_type   : "array_methylation" | "seq_methylation" | "expression" | "mirna" |
                "rna_modification" | "histone/chromatin" | "other"
                (What was measured? m6A/RNA methylation and histone marks are NOT
                DNA methylation. ChIP/chromatin accessibility is NOT methylation.)
  technology  : "450K" | "EPIC" | "EPICv2" | "27K" | "WGBS" | "RRBS" | "MeDIP" |
                "MIRA" | "BS-seq(other)" | "panel" | "other" | null
  organism    : primary organism, scientific name (e.g. "Homo sapiens")
  multi_organism: true if the series mixes organisms (e.g. human tumors + mouse stroma)
  has_control_like_samples: true if ANY combo is a non-cancer/non-disease reference
                (healthy, normal tissue, adjacent normal, benign)
  evidence    : short verbatim quote supporting data_type

COMBO-LEVEL (one entry per combo, keyed by its representative combo_gsm id):
  organism   : "Homo sapiens" | "Mus musculus" | ... | "unknown"
  sample_kind: "primary" (patient/in vivo tissue or body fluid) | "cell_line" |
               "organoid" | "pdx" | "animal" (in-vivo animal, non-PDX) | "unknown"
  specimen   : where the sample came FROM —
               "plasma" | "serum" | "whole_blood" | "pbmc" | "wbc" |
               "tumor_tissue" | "adjacent_normal" | "normal_tissue" |
               "ffpe_tissue" | "ascites" | "pleural_effusion" | "bal" | "stool" |
               "tissue_other" | "other" | "unknown"
               (tumor vs adjacent vs normal: trust disease/lesion fields over source_name
               when they conflict; use tumor_tissue only when the combo is cancerous)
  analyte    : what molecule was measured —
               "cfDNA" | "genomic_DNA" | "total_DNA" | "rna" | "other" | "unknown"
               (plasma/serum DNA is cfDNA even if GEO says "genomic DNA";
                matched tumor tissue measured as a comparison for cfDNA is genomic_DNA)
  disease    : canonical lowercase disease name for the samples in THIS combo —
               e.g. "colorectal cancer", "lung adenocarcinoma", "healthy",
               "adenoma", "cirrhosis", "asthma", "unknown".
               Use "healthy" for healthy donors; use the benign/precursor lesion
               name (e.g. "colorectal adenoma") when the combo is not malignant.
  disease_role: role of this combo in a case/control design —
               "case" | "control" | "precursor" | "benign" | "adjacent_normal" | "unknown"
               (healthy=control; benign lesion=benign; adenoma=precursor;
                adjacent non-tumor tissue=adjacent_normal)
  treatment  : "naive" (no treatment / pre-treatment baseline) | "treated" |
               "post_treatment" | "mixed" | "in_vitro_treated" | "unknown"
               (drug/radiation/gene-edit treatment of the sample → in_vitro_treated;
                patient cohorts analyzed BEFORE therapy → naive)
  lesion     : "primary" | "metastasis" | "local_recurrence" | "na" | "unknown"
  confidence : "high" (explicit in characteristics/source_name/abstract) |
               "medium" (inferred from title/summary, consistent) | "low" (guess)
  evidence   : short verbatim quote from the provided text supporting the labels
               (may quote several fields, e.g. source_name + characteristics)
  decidable_from_metadata: true if the labels above could be determined from GEO
               metadata alone (title/summary/GSM fields) WITHOUT needing the
               abstract. If the abstract was needed to disambiguate, false.
               When no abstract was provided, judge whether metadata alone sufficed.

Rules:
  - Label what the samples ARE, not what would be convenient for downstream use.
  - A combo groups biologically identical samples — all fields should hold for
    every GSM in the combo. If they genuinely differ, trust the characteristics.
  - "disease" is per-combo: in one series, tumor combos are cases, adjacent
    tissue combos are adjacent_normal role with the SAME disease label as the
    tumor (e.g. "colorectal cancer"), healthy donors are "healthy"/"control".
  - Cell lines: sample_kind=cell_line, specimen=other, disease = the cell
    disease of origin (e.g. "colorectal cancer") — they are still not primary samples.
  - Do not infer methylation platform from study type alone; use the platform
    metadata provided (450K/850K/EPIC mapping) when present.
  - EVERY combo in the input must get exactly one combo_labels entry (same combo_gsm id).

Respond with ONLY a single JSON object (no markdown fences, no prose):
{
  "gse_labels": {
    "data_type": "...", "technology": "...", "organism": "...",
    "multi_organism": false, "has_control_like_samples": false,
    "evidence": "..."
  },
  "combo_labels": [
    {"combo_gsm": "<id>", "organism": "...", "sample_kind": "...",
     "specimen": "...", "analyte": "...", "disease": "...",
     "disease_role": "...", "treatment": "...", "lesion": "...",
     "confidence": "...", "evidence": "...", "decidable_from_metadata": true}
  ]
}
"""


# ---------------------------------------------------------------------- #
#  Evidence assembly (mirrors agent1_pipeline._filter_one)                #
# ---------------------------------------------------------------------- #

def _combo_block(combos: List[Dict[str, Any]]) -> str:
    """Render combos the same way geo_filter._gsm_block does (familiar format,
    and the same info the filter LLM sees — keeps decidable_from_metadata honest)."""
    lines = [f"Sample combos ({len(combos)} unique, deduplicated):"]
    for c in combos:
        ch = c.get("characteristics") or {}
        ch_str = "; ".join(f"{k}: {v}" for k, v in ch.items()) if ch else "(none)"
        lines.append(
            f"  - combo_id {c.get('gsm', '?')} [x{c.get('count', 1)}]: "
            f"source_name={c.get('source_name', '')!r}, "
            f"molecule={c.get('molecule', '')!r}, characteristics={{{ch_str}}}"
        )
    return "\n".join(lines)


def _user_message(ds: Dict[str, Any], combos: List[Dict[str, Any]],
                  abstract: Optional[str]) -> str:
    msg = (
        f"=== GEO SERIES METADATA ===\n"
        f"Accession: {ds.get('accession', '?')}\n"
        f"Title: {ds.get('title', '')[:250]}\n"
        f"Summary: {ds.get('summary', '')[:800]}\n"
        f"Overall Design: {ds.get('overall_design', '')[:400]}\n"
        f"Platform: {ds.get('platform_canonical') or ds.get('platforms', [])}\n"
        f"Sample count (GEO): {ds.get('sample_count')}\n"
        f"\n=== SAMPLE COMBOS ===\n{_combo_block(combos)}\n"
    )
    if abstract:
        msg += f"\n=== PUBMED ABSTRACT ===\n{abstract[:2500]}\n"
    else:
        msg += "\n=== PUBMED ABSTRACT ===\n(none available)\n"
    return msg


# ---------------------------------------------------------------------- #
#  Label cache (SQLite, keyed accession+model — resume + cross-model)     #
# ---------------------------------------------------------------------- #

def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            accession   TEXT NOT NULL,
            model       TEXT NOT NULL,
            raw_json    TEXT NOT NULL,
            usage_json  TEXT,
            coverage    TEXT NOT NULL DEFAULT 'full',
            labeled_at  TEXT NOT NULL,
            PRIMARY KEY (accession, model)
        )
    """)
    return conn


def _cache_get(conn: sqlite3.Connection, accession: str, model: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT raw_json, usage_json, coverage FROM labels WHERE accession=? AND model=?",
        (accession, model),
    ).fetchone()
    if not row:
        return None
    rec = json.loads(row[0])
    rec["_usage"] = json.loads(row[1]) if row[1] else {}
    rec.setdefault("coverage", row[2])
    return rec


def _cache_put(conn: sqlite3.Connection, accession: str, model: str,
               rec: Dict[str, Any]) -> None:
    usage = rec.pop("_usage", None)
    coverage = rec.get("coverage", "full")
    conn.execute(
        "INSERT OR REPLACE INTO labels (accession, model, raw_json, usage_json, coverage, labeled_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (accession, model, json.dumps(rec, ensure_ascii=False),
         json.dumps(usage) if usage else None, coverage,
         time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()


# ---------------------------------------------------------------------- #
#  Core labeling                                                          #
# ---------------------------------------------------------------------- #

def label_gse(llm: Any, ds: Dict[str, Any], combos: List[Dict[str, Any]],
              abstract: Optional[str], coverage: str) -> Dict[str, Any]:
    """Label all combos of one GSE (chunked). Returns the label record.

    Each chunk call is wrapped in a hard timeout (LLM_TIMEOUT): glm occasionally
    hangs on large prompts with no error — timeout + retry keeps one bad call
    from stalling the whole run."""
    merged: Dict[str, Any] = {
        "accession": ds.get("accession"),
        "gse_labels": None,
        "combo_labels": [],
    }
    usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0,
                                   "total_tokens": 0, "cached_tokens": 0}

    for i in range(0, len(combos), CHUNK_SIZE):
        chunk = combos[i:i + CHUNK_SIZE]
        user_msg = _user_message(ds, chunk, abstract)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = _invoke_with_timeout(llm, user_msg)
                raw = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed, _tier = _safe_json(raw)
                if merged["gse_labels"] is None:
                    merged["gse_labels"] = parsed.get("gse_labels") or {}
                merged["combo_labels"].extend(parsed.get("combo_labels") or [])
                usage = _extract_usage(resp)
                for k in usage_total:
                    usage_total[k] += int(usage.get(k, 0) or 0)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001 — retry, then surface
                last_err = e
                logger.warning(f"label chunk {ds.get('accession')}[{i}:{i+len(chunk)}] "
                               f"attempt {attempt+1} failed: {e}")
                time.sleep(min(2 ** attempt, 8))
        if last_err is not None:
            raise RuntimeError(f"label chunk failed: {last_err}")

    merged["coverage"] = coverage
    merged["_usage"] = usage_total
    return merged


def _invoke_with_timeout(llm: Any, user_msg: str):
    """llm.invoke under a hard timeout (thread-based; see LLM_TIMEOUT note).

    The pool is a module-level singleton so a timed-out worker thread never
    blocks the with-block exit (ThreadPoolExecutor.__exit__ waits for its
    workers — a fresh pool per call would hang exactly as long as the call we
    are trying to escape)."""

    fut = _LLM_POOL.submit(
        llm.invoke,
        [("system", LABELER_SYSTEM_PROMPT), ("human", user_msg)],
    )
    try:
        return fut.result(timeout=LLM_TIMEOUT)
    except FuturesTimeoutError:
        fut.cancel()
        raise TimeoutError(f"LLM call exceeded {LLM_TIMEOUT}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Label pool GSE combos with factual tags.")
    ap.add_argument("--pool", default=str(_ROOT / "gold" / "pool" / "pool_v1.jsonl"))
    ap.add_argument("--db", default=str(LABELS_DIR / "labels.db"))
    ap.add_argument("--config", default=str(_ROOT / "config" / "settings.yaml"))
    ap.add_argument("--model", default=None, help="Override model (also sets the "
                                                  "backend env var, e.g. ZHIPU_MODEL).")
    ap.add_argument("--backend", default=None, help="Override LLM backend (zhipu/deepseek/...).")
    ap.add_argument("--limit", type=int, default=None, help="Label only the first N pool GSEs.")
    ap.add_argument("--only", default=None, help="Comma-separated accessions to (re)label.")
    ap.add_argument("--force", action="store_true", help="Relabel even if cached.")
    ap.add_argument("--export", action="store_true",
                    help="Export labels.db → labels_v1.jsonl (no labeling).")
    ap.add_argument("--export-out", default=str(LABELS_DIR / "labels_v1.jsonl"))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    conn = _connect(Path(args.db))

    # --- export-only mode ---
    if args.export:
        n = _export(conn, Path(args.export_out))
        print(f"[labels] exported {n} GSE records → {args.export_out}")
        return 0

    # --- LLM setup (env var wins over cfg in get_llm → sync it) ---
    llm_cfg = dict(config["llm"])
    llm_cfg["max_tokens"] = LABELER_MAX_TOKENS
    if args.backend:
        llm_cfg["backend"] = args.backend
    if args.model:
        llm_cfg["model"] = args.model
        env_map = {"zhipu": "ZHIPU_MODEL", "deepseek": "DEEPSEEK_MODEL",
                   "qwen": "QWEN_MODEL", "kimi": "KIMI_MODEL", "openai": "OPENAI_MODEL"}
        env_var = env_map.get(llm_cfg["backend"])
        if env_var:
            os.environ[env_var] = args.model
    llm = get_llm(llm_cfg, json_mode=True)
    model_name = args.model or os.environ.get("ZHIPU_MODEL", "unknown")

    # --- GEO client (same construction as agent1_pipeline) ---
    geo_cfg = config.get("geo", {})
    api_key = os.environ.get(geo_cfg.get("api_key_env", ""), "") or None
    proxy = os.environ.get("NCBI_PROXY", "") or geo_cfg.get("proxy", "") or None
    geo = GEOClient(api_key=api_key, proxy=proxy)

    # --- pool ---
    with open(args.pool, encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]
    if args.only:
        wanted = {a.strip().upper() for a in args.only.split(",")}
        pool = [p for p in pool if p["accession"].upper() in wanted]
    if args.limit:
        pool = pool[: args.limit]

    n_ok = n_fail = n_cached = 0
    for idx, rec in enumerate(pool, 1):
        acc = rec["accession"]
        if not args.force and _cache_get(conn, acc, model_name) is not None:
            n_cached += 1
            continue

        try:
            ds = geo.get_series_metadata(acc)
            if ds.get("error"):
                raise RuntimeError(f"series metadata: {ds['error']}")
            gsm_details = resolve_gsm_details(
                geo, acc, ds, output_dir=str(EVIDENCE_DIR),
                max_all_fetch=MAX_ALL_FETCH,
            )
            # resolve_gsm_details returns early on the series_matrix path
            # without writing its JSON cache — persist the full sample list
            # ourselves so --export can expand combos → GSM ids offline.
            if gsm_details:
                cache = EVIDENCE_DIR / acc / "gsm_metadata_cache.json"
                if not cache.exists():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(
                        json.dumps(gsm_details, ensure_ascii=False), encoding="utf-8"
                    )
            coverage = "full" if len(gsm_details) >= int(ds.get("sample_count") or 0) else "partial"
            if not gsm_details:
                coverage = "none"
            combos = _dedup_gsm_combos(gsm_details) if gsm_details else []
            if not combos:
                raise RuntimeError("no GSM details resolved (series_matrix, cache, "
                                   "efetch all empty/over-cap)")
            if coverage == "partial":
                logger.warning(f"[labels] {acc}: partial coverage "
                               f"({len(gsm_details)}/{ds.get('sample_count')} samples)")
            abstract = None
            if ds.get("pubmed_ids"):
                abstract = geo.fetch_pubmed_abstract(str(ds["pubmed_ids"][0])) or None

            label_rec = label_gse(llm, ds, combos, abstract, coverage)
            _cache_put(conn, acc, model_name, label_rec)
            n_ok += 1
            gl = label_rec.get("gse_labels") or {}
            logger.info(
                f"[labels] {idx}/{len(pool)} {acc}: {len(label_rec['combo_labels'])} combos, "
                f"data_type={gl.get('data_type')}, coverage={label_rec['coverage']}"
            )
        except Exception as e:  # noqa: BLE001 — record failure, keep going
            n_fail += 1
            logger.warning(f"[labels] {idx}/{len(pool)} {acc}: FAILED — {e}")

    print(f"[labels] done: {n_ok} labeled, {n_cached} cached-skip, {n_fail} failed "
          f"(model={model_name})")
    return 0 if n_fail == 0 else 2


def _export(conn: sqlite3.Connection, out_path: Path,
            evidence_dir: Path = EVIDENCE_DIR) -> int:
    """Export cached labels → flat JSONL (one line per GSE), joined with the
    combo→GSM expansion (gsm_ids per combo label) so truth derivation can
    expand combo labels to individual GSM ids.

    The gsm_ids come from re-running _dedup_gsm_combos over the cached GSM
    metadata (gold/evidence_cache/{acc}/gsm_metadata_cache.json / series
    re-fetch). No LLM calls — pure join."""
    # Reconstruct GEOClient-free: read the gsm cache files the labeling run wrote.
    rows = conn.execute(
        "SELECT accession, model, raw_json, coverage FROM labels ORDER BY accession"
    ).fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for acc, model, raw, coverage in rows:
            rec = json.loads(raw)
            rec["model"] = model
            rec.setdefault("coverage", coverage)

            combo_index = _combo_gsm_index(acc, evidence_dir)
            if combo_index:
                for c in rec.get("combo_labels", []):
                    ids = combo_index.get(c.get("combo_gsm"))
                    if ids:
                        c["gsm_ids"] = ids
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _combo_gsm_index(accession: str, evidence_dir: Path) -> Dict[str, List[str]]:
    """combo representative gsm → all gsm ids in that combo, from the cached
    GSM metadata (written by resolve_gsm_details during labeling). Empty dict
    if the cache is gone (labels still usable for GSE-level truth)."""
    cache = evidence_dir / accession / "gsm_metadata_cache.json"
    if not cache.exists():
        return {}
    try:
        gsm_details = json.loads(cache.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — malformed cache → skip expansion
        return {}
    if not isinstance(gsm_details, list):
        return {}
    index: Dict[str, List[str]] = {}
    for combo in _dedup_gsm_combos(gsm_details):
        index[str(combo.get("gsm"))] = [str(g) for g in combo.get("gsm_ids", [])]
    return index


if __name__ == "__main__":
    sys.exit(main())
