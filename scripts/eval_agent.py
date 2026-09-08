#!/usr/bin/env python3
"""
Evaluate agent runs (query_log CSVs [+ registry]) against derived gold truth.

Inputs:
  --truth-dir gold/truth/v1      (scripts/derive_truth.py output)
  --pool      gold/pool/pool_v1.jsonl
  --logs      data/benchmark/query_logs  (dir) or individual CSV paths
  --registry  registry/methyagent.db --task-id <id>   (optional, GSM-level)

query_log ↔ query matching: the log's "# query:" preamble must equal a
gold/queries.yaml `text` (whitespace-normalized).

Per-query metrics (docs/benchmark.md §1):
  1. search recall   — |truth GSE ∩ judged accessions| / |truth GSE| (+ misses)
  2. filter confusion — 3-state outcome vs binary truth; manual_review counts
     as ABSTENTION (reported separately, not as error) + risk-coverage point;
     split into fair tier (all relevant combos decidable_from_metadata) vs
     reference tier (all GSEs).
  3. GSM P/R/F1      — registry verdict='download' vs truth gsm_positive
                       (restricted to GSEs the agent judged)
  4. e2e decomposition — search_recall × filter_recall_on_found × gsm_selection

Output: data/eval/eval_{ts}.md + per-query CSV.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

QUERIES_FILE = _ROOT / "gold" / "queries.yaml"


# ---------------------------------------------------------------------- #
#  query_log parsing (same pattern as compare_benchmarks.py)              #
# ---------------------------------------------------------------------- #

def parse_query_log(path: Path) -> Optional[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as e:  # noqa: BLE001
        print(f"  skip {path}: {e}")
        return None
    meta: Dict[str, str] = {}
    data_lines: List[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("#") and ":" in s:
            k, v = s[1:].split(":", 1)
            meta[k.strip()] = v.strip()
        elif s:
            data_lines.append(line)
    if not data_lines:
        return None
    rows = list(csv.DictReader(io.StringIO("\n".join(data_lines))))
    return {
        "query_text": meta.get("query", ""),
        "llm_model": meta.get("llm_model", "unknown"),
        "spec": meta.get("注意事项", "unknown"),
        "rows": rows,
        "path": str(path),
    }


def _norm_query(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


# ---------------------------------------------------------------------- #
#  Fairness tier: which GSEs are metadata-decidable                       #
# ---------------------------------------------------------------------- #

def decidable_accessions(labels_file: Path, truth: Dict[str, Any]) -> Dict[str, bool]:
    """accession → all combo labels decidable_from_metadata (GSEs not in the
    label file count as NOT decidable → reference tier only)."""
    dec: Dict[str, bool] = {}
    if not labels_file.exists():
        return dec
    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            combos = rec.get("combo_labels") or []
            rec_dec = all(c.get("decidable_from_metadata") for c in combos) if combos else True
            dec[rec["accession"]] = rec_dec
    return dec


# ---------------------------------------------------------------------- #
#  Metrics                                                                #
# ---------------------------------------------------------------------- #

def search_metrics(truth: Dict[str, Any], judged: set) -> Dict[str, Any]:
    truth_gse = set(truth["gse_positive"])
    found = truth_gse & judged
    return {
        "n_truth": len(truth_gse),
        "n_found": len(found),
        "recall": len(found) / len(truth_gse) if truth_gse else None,
        "missed": sorted(truth_gse - judged),
    }


def filter_metrics(truth: Dict[str, Any], rows: List[Dict], decidable: Dict[str, bool],
                   pool: set) -> Dict[str, Any]:
    """3-state outcome vs binary truth. Returns per-tier confusion + abstentions.

    Scoring is restricted to pool GSEs (the universe the truth was derived
    over): a GSE the agent judged that is NOT in the pool is unscorable
    (counted separately as n_out_of_pool)."""
    truth_gse = set(truth["gse_positive"])
    tiers = {"fair": Counter(), "reference": Counter()}
    abstain = Counter()
    out_of_pool = 0
    for r in rows:
        acc = (r.get("accession") or "").upper()
        if acc not in pool:
            out_of_pool += 1
            continue
        outcome = (r.get("outcome") or r.get("recommended_action") or "").strip()
        is_pos = acc in truth_gse
        if outcome == "manual_review":
            abstain["reference"] += 1
            if decidable.get(acc, False):
                abstain["fair"] += 1
            continue
        pred_pos = outcome == "download"
        cell = ("TP" if is_pos else "FP") if pred_pos else ("FN" if is_pos else "TN")
        tiers["reference"][cell] += 1
        if decidable.get(acc, False):
            tiers["fair"][cell] += 1
    return {"tiers": tiers, "abstain": abstain, "n_out_of_pool": out_of_pool}


def gsm_metrics(truth: Dict[str, Any], registry_path: Path, task_id: str,
                judged: set) -> Optional[Dict[str, Any]]:
    if not registry_path.exists() or not task_id:
        return None
    sys.path.insert(0, str(_ROOT))
    from registry.registry import Registry

    reg = Registry(str(registry_path))
    rows = reg.get_samples_by_task_id(task_id)
    pred_by_acc: Dict[str, set] = {}
    for r in rows:
        if r.get("verdict") == "download":
            pred_by_acc.setdefault(r["accession"], set()).add(r["gsm"])

    tp = fp = fn = 0
    per_gse: Dict[str, Dict[str, int]] = {}
    scored_accs = set(truth.get("gsm_positive", {})) & judged
    for acc in scored_accs:
        truth_gsm = set(truth["gsm_positive"][acc])
        pred_gsm = pred_by_acc.get(acc, set())
        tp += len(truth_gsm & pred_gsm)
        fp += len(pred_gsm - truth_gsm)
        fn += len(truth_gsm - pred_gsm)
        per_gse[acc] = {"tp": len(truth_gsm & pred_gsm),
                        "fp": len(pred_gsm - truth_gsm),
                        "fn": len(truth_gsm - pred_gsm)}
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and (precision + recall) > 0 else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "per_gse": per_gse}


def _fmt(x: Optional[float], pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


# ---------------------------------------------------------------------- #
#  Main                                                                   #
# ---------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate agent query_logs against gold truth.")
    ap.add_argument("--truth-dir", default=str(_ROOT / "gold" / "truth" / "v1"))
    ap.add_argument("--pool", default=str(_ROOT / "gold" / "pool" / "pool_v1.jsonl"))
    ap.add_argument("--labels", default=str(_ROOT / "gold" / "labels" / "v1" / "labels_v1.jsonl"),
                    help="Label export (for the fair/reference tier split).")
    ap.add_argument("--queries", default=str(QUERIES_FILE))
    ap.add_argument("--logs", nargs="+", required=True,
                    help="query_log CSV path(s) or a directory of them.")
    ap.add_argument("--registry", default=str(_ROOT / "registry" / "methyagent.db"))
    ap.add_argument("--task-id", default=None,
                    help="Task id for GSM-level scoring (registry query_sample_map).")
    ap.add_argument("--out-dir", default=str(_ROOT / "data" / "eval"))
    args = ap.parse_args()

    # --- gather logs ---
    log_paths: List[Path] = []
    for p in args.logs:
        path = Path(p)
        if path.is_dir():
            log_paths.extend(sorted(path.glob("*.csv")))
        else:
            log_paths.append(path)
    logs = [l for l in (parse_query_log(p) for p in log_paths) if l]
    if not logs:
        print("[eval] no parsable query_logs")
        return 1

    # --- gold lookups ---
    pool_accs = set()
    with open(args.pool, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pool_accs.add(json.loads(line)["accession"].upper())

    with open(args.queries, encoding="utf-8") as f:
        qcfg = yaml.safe_load(f)
    text_to_id = {_norm_query(q["text"]): q["id"] for q in qcfg.get("queries", [])}

    truth_by_id: Dict[str, Dict] = {}
    for tf in Path(args.truth_dir).glob("*.json"):
        t = json.loads(tf.read_text())
        truth_by_id[t["query_id"]] = t

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_lines: List[str] = [f"# Agent eval vs gold truth — {ts}",
                           f"- pool: {len(pool_accs)} GSE | logs: {len(logs)} | "
                           f"truth dir: {args.truth_dir}"]
    csv_rows: List[Dict[str, Any]] = []

    for log in logs:
        qid = text_to_id.get(_norm_query(log["query_text"]))
        if qid is None or qid not in truth_by_id:
            md_lines.append(f"\n## ⚠ unmatched query: `{log['query_text']}` ({log['path']})\n"
                            f"No gold/queries.yaml entry with this exact text — skipped.")
            continue
        truth = truth_by_id[qid]
        decidable = decidable_accessions(Path(args.labels), truth)

        rows = log["rows"]
        judged = {(r.get("accession") or "").upper() for r in rows}
        sm = search_metrics(truth, judged)
        fm = filter_metrics(truth, rows, decidable, pool_accs)
        gm = gsm_metrics(truth, Path(args.registry), args.task_id or "", judged)

        # e2e decomposition (GSM-level, judged GSEs only)
        found_pos = set(truth["gse_positive"]) & judged
        dl_pos = {acc for acc in found_pos
                  if any((r.get("accession") or "").upper() == acc
                         and (r.get("outcome") or r.get("recommended_action")) == "download"
                         for r in rows)}
        e2e_parts = {
            "search_recall": sm["recall"],
            "filter_recall_on_found": (len(dl_pos) / len(found_pos)) if found_pos else None,
            "gsm_selection": gm["recall"] if gm else None,
        }

        md_lines.append(f"\n## {qid}")
        md_lines.append(f"`{truth['text']}` — model={log['llm_model']} spec={log['spec']}"
                        + (" — **expect_empty probe**" if truth["expect_empty"] else ""))
        md_lines.append(f"\n**Search recall**: {sm['n_found']}/{sm['n_truth']} "
                        f"= {_fmt(sm['recall'], pct=True)}"
                        + (f" — missed: {', '.join(sm['missed'][:10])}"
                           + (" …" if len(sm['missed']) > 10 else "") if sm["missed"] else ""))
        if fm["n_out_of_pool"]:
            md_lines.append(f"*(scoring universe: pool GSEs; "
                            f"{fm['n_out_of_pool']} judged GSEs outside pool not scored)*")
        md_lines.append("\n| tier | TP | FP | TN | FN | abstain | precision | recall |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for tier in ("fair", "reference"):
            c = fm["tiers"][tier]
            tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
            prec = tp / (tp + fp) if (tp + fp) else None
            rec = tp / (tp + fn) if (tp + fn) else None
            md_lines.append(f"| {tier} | {tp} | {fp} | {tn} | {fn} | "
                            f"{fm['abstain'][tier]} | {_fmt(prec, pct=True)} | {_fmt(rec, pct=True)} |")
        if gm:
            md_lines.append(f"\n**GSM-level** (registry, {args.task_id or '?'}): "
                            f"TP={gm['tp']} FP={gm['fp']} FN={gm['fn']} → "
                            f"P={_fmt(gm['precision'], pct=True)} "
                            f"R={_fmt(gm['recall'], pct=True)} "
                            f"F1={_fmt(gm['f1'])}")
        else:
            md_lines.append("\n*(GSM-level scoring skipped — pass --registry + --task-id)*")
        md_lines.append("\n**e2e (GSM recall decomposition)**: "
                        + " × ".join(f"{k}={_fmt(v, pct=True)}" for k, v in e2e_parts.items()))
        if truth["expect_empty"]:
            # Probe queries (cell line / miRNA / mouse) are things the agent's
            # SPEC hard-excludes: any download verdict is a false positive.
            # FP was already accumulated into the confusion cells above (their
            # truth sets are tiny by construction) — report it prominently.
            fp_ref = fm["tiers"]["reference"]["FP"]
            fp_fair = fm["tiers"]["fair"]["FP"]
            md_lines.append(f"\n⚠ false-positive probe: {fp_ref} download verdicts on a "
                            f"should-reject query (fair tier: {fp_fair})")

        csv_rows.append({
            "query_id": qid, "n_truth_gse": sm["n_truth"], "n_found": sm["n_found"],
            "search_recall": sm["recall"], "missed": ";".join(sm["missed"]),
            "fair_TP": fm["tiers"]["fair"]["TP"], "fair_FP": fm["tiers"]["fair"]["FP"],
            "fair_FN": fm["tiers"]["fair"]["FN"], "fair_TN": fm["tiers"]["fair"]["TN"],
            "fair_abstain": fm["abstain"]["fair"],
            "ref_TP": fm["tiers"]["reference"]["TP"], "ref_FP": fm["tiers"]["reference"]["FP"],
            "ref_FN": fm["tiers"]["reference"]["FN"], "ref_TN": fm["tiers"]["reference"]["TN"],
            "ref_abstain": fm["abstain"]["reference"],
            "gsm_tp": gm["tp"] if gm else "", "gsm_fp": gm["fp"] if gm else "",
            "gsm_fn": gm["fn"] if gm else "", "gsm_f1": gm["f1"] if gm else "",
            "out_of_pool": fm["n_out_of_pool"],
        })

    md_path = out_dir / f"eval_{ts}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    csv_path = out_dir / f"eval_{ts}_perquery.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"[eval] wrote {md_path}" + (f" and {csv_path}" if csv_rows else ""))
    print("\n".join(md_lines[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
