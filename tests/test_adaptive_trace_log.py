"""Tests for skills.adaptive_evidence.trace_log — JSONL trace persistence (§2 metric)."""
import json
import tempfile
import unittest
from pathlib import Path

from skills.adaptive_evidence.trace_log import append_trace


class TestAppendTrace(unittest.TestCase):
    def test_writes_summary_record(self):
        with tempfile.TemporaryDirectory() as d:
            trace = [
                {"step": 1, "event": "fetch", "name": "fetch_abstract", "target": "pmid:123"},
                {"step": 2, "event": "conclude", "outcome": "download"},
            ]
            append_trace(d, "GSE1", "manual_review", "download", trace)
            p = Path(d) / "adaptive_traces.jsonl"
            self.assertTrue(p.exists())
            rec = json.loads(p.read_text().strip())
            self.assertEqual(rec["accession"], "GSE1")
            self.assertEqual(rec["outcome_before"], "manual_review")
            self.assertEqual(rec["outcome_after"], "download")
            self.assertTrue(rec["resolved"])
            self.assertFalse(rec["fallback"])
            self.assertEqual(rec["n_fetches"], 1)
            self.assertEqual(rec["n_steps"], 2)
            self.assertEqual(rec["event_counts"], {"fetch": 1, "conclude": 1})

    def test_unresolved_fallback_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            trace = [
                {"step": 1, "event": "fetch", "name": "x", "target": "t"},
                {"event": "max_steps_exhausted", "fallback": True},
            ]
            append_trace(d, "GSE2", "manual_review", "manual_review", trace)
            rec = json.loads((Path(d) / "adaptive_traces.jsonl").read_text().strip())
            self.assertFalse(rec["resolved"])   # stayed manual_review
            self.assertTrue(rec["fallback"])    # hit a fallback event

    def test_appends_multiple_records(self):
        with tempfile.TemporaryDirectory() as d:
            append_trace(d, "GSE1", "manual_review", "download", [{"event": "conclude"}])
            append_trace(d, "GSE2", "manual_review", "exclude", [{"event": "conclude"}])
            lines = (Path(d) / "adaptive_traces.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)

    def test_empty_trace_and_bad_path_never_raise(self):
        with tempfile.TemporaryDirectory() as d:
            append_trace(d, "GSE3", "a", "b", [])  # empty trace is fine
            rec = json.loads((Path(d) / "adaptive_traces.jsonl").read_text().strip())
            self.assertEqual(rec["n_fetches"], 0)
            self.assertEqual(rec["event_counts"], {})
        # Unwritable path is swallowed (best-effort telemetry, never breaks pipeline).
        append_trace("/nonexistent/protected/x", "GSE4", "a", "b", [{"event": "conclude"}])


if __name__ == "__main__":
    unittest.main()
