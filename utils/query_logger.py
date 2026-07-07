"""
Per-query CSV logger for the geo_filter skill.

For every query, writes ONE CSV file capturing:
  - metadata preamble: timestamp, query, llm_model, api_model, 注意事项 (SPEC
    document name), datasets judged, kept count, total tokens
  - one row per dataset judged: accession, keep/exclude verdict, sample type,
    reason, notes, and the token cost of that judgment's LLM call

The preamble is written as `#`-comment lines (human-scannable); the data table
below the blank line is clean CSV. Load with pandas via:
    pd.read_csv(path, comment="#")

Thread-safe: rows are appended under a lock (the skill judges datasets
concurrently). The file is written once at finalize(), after all judgments land.

The logger NEVER raises into the pipeline — a logging failure is logged and
swallowed (logging is best-effort, not on the critical path).
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# Actions that count as "kept" (queued for approval / review).
_KEEP_ACTIONS = {"keep", "manual_review"}

# One kept == queued for approval (keep) OR flagged for review (manual_review).
COLUMNS: List[str] = [
    "accession",
    "source",
    "title",
    "recommended_action",
    "kept",
    "usable",
    "sample_type",
    "cancer_type",
    "platform",
    "sample_count",
    "reasoning",
    "reason",
    "notes",
    "gsm_groups",
    "had_abstract",
    "n_representative_gsm",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "api_model",
]


class QueryLogger:
    """Collect per-dataset geo_filter verdicts for one query, write one CSV."""

    def __init__(
        self,
        query: str,
        model_name: str,
        spec_name: str,
        output_dir: str,
        query_id: Optional[str] = None,
    ) -> None:
        self.query = query or ""
        self.model_name = model_name or "unknown"
        self.spec_name = spec_name or "unknown"
        self.output_dir = output_dir

        self._started = datetime.datetime.now()
        ts = self._started.strftime("%Y%m%d_%H%M%S")
        qh = hashlib.md5(self.query.encode("utf-8")).hexdigest()[:6]
        self.query_id = query_id or f"{ts}_{qh}"

        self.path = Path(output_dir) / "query_logs" / f"query_{self.query_id}.csv"

        self._rows: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_tokens = 0
        self._api_model = ""

    # ------------------------------------------------------------------ #
    #  Collect                                                            #
    # ------------------------------------------------------------------ #

    def log_dataset(self, ds: Dict[str, Any], verdict: Dict[str, Any]) -> None:
        """Append one judged dataset. Safe to call from worker threads."""
        usage = verdict.get("_usage") or {}
        evidence = verdict.get("_evidence") or {}
        action = verdict.get("recommended_action")
        gsm_groups = evidence.get("gsm_groups") or {}
        groups_str = ", ".join(f"{g}={n}" for g, n in gsm_groups.items() if n) if gsm_groups else ""
        with self._lock:
            self._total_tokens += int(usage.get("total_tokens") or 0)
            if not self._api_model and usage.get("api_model"):
                self._api_model = str(usage["api_model"])
            self._rows.append(
                {
                    "accession": ds.get("accession", ""),
                    "source": ds.get("source", "GEO"),
                    "title": (ds.get("title") or "")[:120],
                    "recommended_action": action or "",
                    "kept": "yes" if action in _KEEP_ACTIONS else "no",
                    "usable": verdict.get("usable", ""),
                    "sample_type": verdict.get("confirmed_sample_type")
                    or ds.get("sample_type", ""),
                    "cancer_type": verdict.get("confirmed_cancer_type")
                    or ds.get("cancer_type", ""),
                    "platform": ds.get("platform_canonical") or ds.get("platform", ""),
                    "sample_count": ds.get("sample_count", ""),
                    # The full step-by-step logic chain (the model's reasoning),
                    # so a human can audit WHY a dataset was kept/excluded and
                    # spot self-contradictions without re-running anything.
                    "reasoning": verdict.get("reasoning", ""),
                    "reason": verdict.get("reason", ""),
                    "notes": verdict.get("notes", ""),
                    # What evidence the model was given (weak evidence often
                    # explains a bad verdict): GSM group breakdown + whether a
                    # PubMed abstract was available.
                    "gsm_groups": groups_str,
                    "had_abstract": "yes" if evidence.get("had_abstract") else "no",
                    "n_representative_gsm": evidence.get("n_representative_gsm", ""),
                    "prompt_tokens": usage.get("prompt_tokens", ""),
                    "completion_tokens": usage.get("completion_tokens", ""),
                    "total_tokens": usage.get("total_tokens", ""),
                    "cached_tokens": usage.get("cached_tokens", ""),
                    "api_model": usage.get("api_model", ""),
                }
            )

    # ------------------------------------------------------------------ #
    #  Write                                                              #
    # ------------------------------------------------------------------ #

    def finalize(self) -> Optional[str]:
        """Write the CSV. Returns the path, or None on failure."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            kept = sum(1 for r in self._rows if r["kept"] == "yes")
            with open(self.path, "w", newline="", encoding="utf-8-sig") as f:
                w = f.write
                w("# MethyAgent — geo_filter per-query log\n")
                w(f"# timestamp: {self._started.strftime('%Y-%m-%d %H:%M:%S')}\n")
                w(f"# query: {self.query}\n")
                w(f"# llm_model: {self.model_name}\n")
                w(f"# api_model: {self._api_model or self.model_name}\n")
                w(f"# 注意事项: {self.spec_name}\n")
                w(f"# datasets_judged: {len(self._rows)}\n")
                w(f"# kept: {kept}\n")
                w(f"# total_tokens: {self._total_tokens}\n")
                w("\n")
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                for row in self._rows:
                    writer.writerow(row)
            logger.info(f"Query log written: {self.path} ({len(self._rows)} rows, kept={kept})")
            return str(self.path)
        except Exception as e:
            logger.warning(f"QueryLogger.finalize failed: {e}")
            return None
