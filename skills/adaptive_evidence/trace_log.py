"""
Persist adaptive-evidence traces for the §2 guardrail benchmark.

Each adaptive run (a manual_review dataset that triggered the bounded ReAct
evidence loop) is summarized to one JSON line in
`{output_dir}/adaptive_traces.jsonl`, so `scripts/aggregate_adaptive_traces.py`
can compute resolution rate, guard-trigger histogram, step/fetch distribution,
and outcome transitions offline.

Best-effort: a write failure is logged and swallowed — never breaks the pipeline.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_RESOLVED = {"download", "exclude"}  # decisive outcomes (not manual_review)
_FALLBACK_EVENTS = {"no_tool_call", "max_steps_exhausted", "agent_error"}


def _summarize(accession: str, outcome_before: Optional[str],
               outcome_after: Optional[str], trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    events = [t.get("event") for t in (trace or [])]
    counts: Dict[str, int] = {}
    for e in events:
        counts[e] = counts.get(e, 0) + 1
    steps = [t.get("step") for t in (trace or []) if isinstance(t.get("step"), int)]
    return {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "accession": accession,
        "outcome_before": outcome_before,
        "outcome_after": outcome_after,
        "resolved": outcome_after in _RESOLVED,
        "n_steps": max(steps) if steps else len(trace or []),
        "n_fetches": counts.get("fetch", 0),
        "fallback": any(e in _FALLBACK_EVENTS for e in events),
        "events": events,
        "event_counts": counts,
    }


def append_trace(
    output_dir: str,
    accession: str,
    outcome_before: Optional[str],
    outcome_after: Optional[str],
    trace: Optional[List[Dict[str, Any]]],
) -> None:
    """Append one adaptive run's summary to {output_dir}/adaptive_traces.jsonl."""
    try:
        record = _summarize(accession, outcome_before, outcome_after, trace or [])
        path = Path(output_dir) / "adaptive_traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 — never break the pipeline over telemetry
        logger.warning(f"adaptive trace log failed for {accession}: {e}")
