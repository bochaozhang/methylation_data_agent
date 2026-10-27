"""
Submitter-coded sample-title decoding (backlog §6-8, 2026-10-26).

Problem: some submitters encode the disease group ONLY in sample titles
(GSE97923: "COAD.ct-511") while GSM characteristics carry nothing useful —
the deterministic _gsm_cancer_map then marks every sample cancer=unclear.

Approach (mirrors Tier-2 file selection): ONE LLM call per dataset sees the
distinct title patterns (digits normalised) + dataset context and returns a
pattern→class mapping; the rule is applied to ALL GSMs deterministically.

Classes: query_cancer | normal | other  (+confidence +note per pattern).

Apply path: registry samples.cancer + the cancer column of
sample_metadata.csv + datasets.disease_groups (decoded breakdown text).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_CLASSES = {"query_cancer", "normal", "other"}

_DECODE_SYSTEM_PROMPT = """You decode submitter-coded GEO sample titles.

Given a dataset's context (title/summary/cancer type) and the DISTINCT
sample-title patterns (digits replaced by N, with counts), infer what each
pattern means clinically: cancer case ("query_cancer"), control/healthy/
normal tissue or blood ("normal"), or something else ("other" — benign
disease, premalignant, different cancer, QC, etc.).

Respond with ONLY a JSON object:
{
  "confidence": "high" | "medium" | "low",
  "patterns": [
    {"pattern": "<exact pattern string from the input>",
     "cancer": "query_cancer" | "normal" | "other",
     "note": "<short justification>"}
  ],
  "disease_groups": "<one-line breakdown like 'Primary Tumor 293 / Solid Tissue Normal 26' built from the counts>",
  "caveats": "<anything ambiguous, or empty string>"
}

Rules: cover EVERY input pattern exactly once (copy pattern strings
verbatim). Base inference on the naming scheme and dataset context; when a
pattern is genuinely ambiguous use "other" and say so in caveats."""


def _normalise_title(t: str) -> str:
    return re.sub(r"\d+", "N", (t or "").strip())


def load_titles(accession: str, data_dir: str) -> List[Dict[str, str]]:
    """[{gsm, title, group?}] from the gsm cache; [] when absent/empty."""
    cache = Path(data_dir) / accession / "gsm_metadata_cache.json"
    if not cache.exists():
        return []
    entries = json.load(open(cache, encoding="utf-8"))
    if isinstance(entries, dict):
        entries = entries.get("samples", [])
    out = []
    for e in entries:
        if isinstance(e, dict) and e.get("gsm") and e.get("title"):
            out.append({"gsm": e["gsm"], "title": str(e["title"]),
                        "group": e.get("group", "")})
    return out


def merge_titles_into_cache(accession: str, data_dir: str,
                            titles: Dict[str, str]) -> int:
    """Backfill title onto cached gsm entries (cache-first decoding enabler).
    Returns the number of entries that gained a title."""
    cache = Path(data_dir) / accession / "gsm_metadata_cache.json"
    if not cache.exists() or not titles:
        return 0
    entries = json.load(open(cache, encoding="utf-8"))
    wrapped = isinstance(entries, dict)
    lst = entries.get("samples", []) if wrapped else entries
    added = 0
    for e in lst:
        if isinstance(e, dict) and e.get("gsm") and not e.get("title"):
            t = titles.get(e["gsm"])
            if t:
                e["title"] = t
                added += 1
    if added:
        json.dump(entries, open(cache, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    return added


def decode_titles(
    llm: Any,
    dataset_row: Dict[str, Any],
    samples: List[Dict[str, str]],
) -> Dict[str, Any]:
    """One LLM call → {pattern→class map, confidence, disease_groups, caveats}."""
    pattern_counts: Dict[str, int] = {}
    gsm_pattern: Dict[str, str] = {}
    for s in samples:
        p = _normalise_title(s["title"])
        if not p:
            continue
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
        gsm_pattern[s["gsm"]] = p

    if not pattern_counts:
        return {"error": "no sample titles available (cache predates title "
                         "persistence — re-resolve the dataset first)"}

    ctx = (
        f"Dataset: {dataset_row.get('accession')} — "
        f"{(dataset_row.get('title') or '')[:150]}\n"
        f"Summary: {(dataset_row.get('summary') or '')[:300]}\n"
        f"Registry cancer_type: {dataset_row.get('cancer_type')}\n"
        f"Sample groups (keyword-classified): see patterns below\n\n"
        f"DISTINCT TITLE PATTERNS (digits→N, with counts):\n"
        + "\n".join(f"- {p}  (n={n})" for p, n in
                    sorted(pattern_counts.items(), key=lambda kv: -kv[1]))
    )
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = llm.invoke([
        SystemMessage(content=_DECODE_SYSTEM_PROMPT),
        HumanMessage(content=ctx),
    ])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    out = json.loads(raw)

    pat_map = {}
    for p in out.get("patterns", []):
        pat = p.get("pattern")
        cancer = (p.get("cancer") or "").strip()
        if pat and cancer in _VALID_CLASSES:
            pat_map[pat] = {"cancer": cancer, "note": p.get("note", "")}

    gsm_class = {g: pat_map[p]["cancer"]
                 for g, p in gsm_pattern.items() if p in pat_map}
    out["gsm_class"] = gsm_class
    out["n_classified"] = len(gsm_class)
    out["n_total"] = len(gsm_pattern)
    return out


def apply_decoded(registry: Any, accession: str, data_dir: str,
                  gsm_class: Dict[str, str]) -> Dict[str, int]:
    """Persist the decoded classes: samples table, CSV cancer column."""
    counts = {"registry": 0, "csv": 0}
    if registry is not None and gsm_class:
        counts["registry"] = registry.update_samples_cancer(accession, gsm_class)
    counts["csv"] = _rewrite_csv_cancer(accession, data_dir, gsm_class)
    return counts


def _rewrite_csv_cancer(accession: str, data_dir: str,
                        gsm_class: Dict[str, str]) -> int:
    """Update ONLY the cancer column of sample_metadata.csv in place."""
    import csv
    path = Path(data_dir) / accession / "sample_metadata.csv"
    if not path.exists():
        return 0
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return 0
    fields = list(rows[0].keys())
    changed = 0
    for r in rows:
        new = gsm_class.get(r.get("gsm", ""))
        if new and new != r.get("cancer"):
            r["cancer"] = new
            changed += 1
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    return changed


def decoded_disease_groups(gsm_class: Dict[str, str],
                           samples: List[Dict[str, str]]) -> Optional[str]:
    """Counts per decoded class, e.g. 'query_cancer 293 / normal 26 / other 3'."""
    if not gsm_class:
        return None
    title_by_gsm = {s["gsm"]: s["title"] for s in samples}
    from collections import Counter
    c = Counter(gsm_class.values())
    parts = []
    label = {"query_cancer": "cancer", "normal": "normal", "other": "other"}
    for k in ("query_cancer", "normal", "other"):
        if c.get(k):
            parts.append(f"{label[k]} {c[k]}")
    del title_by_gsm
    return " / ".join(parts) if parts else None
