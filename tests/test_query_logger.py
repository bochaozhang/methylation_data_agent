"""Tests for utils/query_logger.py — per-query CSV logging of geo_filter verdicts."""
import csv
import io
import os
import tempfile
import unittest

from utils.query_logger import QueryLogger


def _read_data_rows(path):
    """Read the CSV data table, skipping the '#'-comment preamble."""
    lines = [ln for ln in open(path, encoding="utf-8-sig")
             if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


def _row(verdict_extra=None, usage=None, ds=None):
    ds = ds or {"accession": "GSE1", "source": "GEO", "title": "t", "sample_count": 10}
    v = {"recommended_action": "keep", "usable": "yes", "confirmed_sample_type": "cfdna",
         "confirmed_cancer_type": "colorectal cancer", "reason": "ok", "notes": ""}
    if verdict_extra:
        v.update(verdict_extra)
    v["_usage"] = usage or {"prompt_tokens": 100, "completion_tokens": 10,
                            "total_tokens": 110, "cached_tokens": 0, "api_model": "deepseek-chat"}
    return ds, v


class TestQueryLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_qlog_")

    def test_metadata_and_rows_written(self):
        qlog = QueryLogger(query="crc cfDNA", model_name="deepseek-chat",
                           spec_name="GEO 数据检索注意事项 v3", output_dir=self.tmp)
        ds, v = _row()
        qlog.log_dataset(ds, v)
        path = qlog.finalize()
        self.assertTrue(os.path.exists(path))
        text = open(path, encoding="utf-8-sig").read()
        # Metadata preamble
        self.assertIn("# llm_model: deepseek-chat", text)
        self.assertIn("# 注意事项: GEO 数据检索注意事项 v3", text)
        self.assertIn("# query: crc cfDNA", text)
        self.assertIn("# total_tokens: 110", text)
        # Data table below the blank line
        rows = _read_data_rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["accession"], "GSE1")
        self.assertEqual(rows[0]["kept"], "yes")
        self.assertEqual(rows[0]["total_tokens"], "110")

    def test_kept_flag_excludes(self):
        qlog = QueryLogger(query="q", model_name="m", spec_name="s", output_dir=self.tmp)
        qlog.log_dataset({"accession": "A"}, {"recommended_action": "exclude", "usable": "no",
                                              "_usage": {"total_tokens": 50}})
        qlog.log_dataset({"accession": "B"}, {"recommended_action": "manual_review",
                                              "usable": "unclear", "_usage": {"total_tokens": 50}})
        qlog.log_dataset({"accession": "C"}, {"recommended_action": "keep", "usable": "yes",
                                              "_usage": {"total_tokens": 50}})
        path = qlog.finalize()
        rows = _read_data_rows(path)
        kept = {r["accession"]: r["kept"] for r in rows}
        self.assertEqual(kept, {"A": "no", "B": "yes", "C": "yes"})  # manual_review counts as kept
        self.assertIn("# kept: 2", open(path, encoding="utf-8-sig").read())

    def test_finalize_swallows_errors(self):
        # Unwritable dir must NOT raise.
        qlog = QueryLogger(query="q", model_name="m", spec_name="s", output_dir="/proc/cannot")
        self.assertIsNone(qlog.finalize())


if __name__ == "__main__":
    unittest.main()
