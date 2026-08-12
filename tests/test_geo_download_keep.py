"""Tests for geo-download Tier-2 file selection.

Covers:
  - _junk_filter (demoted inspect_matrix_head): drop only README/empty/p-value tables;
    keep MCTA-Seq count matrices, beta matrices, series_matrix.
  - _llm_select_files (LLM sample-type selection) with a FakeLLM.
  - _select_relevant_files conservative fallback (no LLM / parse error / keep-none).
  - _target_sample_summary.
"""
import gzip
import json
import os
import tempfile
import unittest

import pandas as pd

from skills.geo_download.skill import DownloadSkill


def _write_gz(path: str, text: str) -> str:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)
    return path


def _done_result(path: str, url: str = "https://example/x") -> dict:
    return {
        "accession": "GSETEST",
        "status": "done",
        "local_path": path,
        "file_size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        "checksum_md5": "abc",
        "url": url,
    }


class _Resp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Mimics langchain BaseChatModel.invoke — returns a fixed JSON string."""
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, messages):
        return _Resp(json.dumps(self._payload))


def _make_skill(llm=None):
    """Bypass DownloadSkill.__init__ (no engine/geo_client/real LLM)."""
    skill = DownloadSkill.__new__(DownloadSkill)
    skill.llm = llm
    return skill


class TestJunkFilter(unittest.TestCase):
    def setUp(self):
        self.skill = _make_skill()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _filter(self, files):
        return self.skill._junk_filter("GSETEST", [_done_result(f) for f in files])

    def test_pvalue_table_is_junk(self):
        m = "gene\tlogFC\tpvalue\nA\t2.3\t0.001\nB\t-1.2\t0.04\n"
        p = _write_gz(os.path.join(self.dir, "deg.txt.gz"), m)
        kept, junk = self._filter([p])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(junk), 1)
        self.assertFalse(os.path.exists(p))

    def test_empty_file_is_junk(self):
        p = _write_gz(os.path.join(self.dir, "readme.txt.gz"), "")
        kept, junk = self._filter([p])
        self.assertEqual(len(junk), 1)
        self.assertEqual(len(kept), 0)

    def test_mctaseq_count_matrix_NOT_junk(self):
        # MCTA-Seq: coordinate cols + sample cols, integer/float counts.
        m = ("#chr\tstr\tend\tCGI_num\tstrand\tcg_num\tPcrc90\tPcrc88\n"
             "chr10\t100\t101\t4\t+\t1\t5\t7\n"
             "chr10\t200\t201\t4\t+\t2\t0\t2\n")
        p = _write_gz(os.path.join(self.dir, "sumP_umepm.txt.gz"), m)
        kept, junk = self._filter([p])
        self.assertEqual(len(kept), 1, "MCTA-Seq count matrix must pass junk filter")
        self.assertEqual(len(junk), 0)
        self.assertTrue(os.path.exists(p))

    def test_beta_matrix_kept(self):
        m = "ID_REF\tGSM3690001\tGSM3690002\ncg1\t0.1\t0.9\ncg2\t0.2\t0.8\n"
        p = _write_gz(os.path.join(self.dir, "beta.txt.gz"), m)
        kept, junk = self._filter([p])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(junk), 0)

    def test_series_matrix_trusted_kept(self):
        p = _write_gz(os.path.join(self.dir, "GSETEST_series_matrix.txt.gz"),
                      "junk content\n")
        kept, junk = self._filter([p])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["data_form"], "series_matrix")
        self.assertEqual(len(junk), 0)


class TestLLMSelectFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        # Two beta-style candidate files (pass junk filter).
        self.plasma = _write_gz(
            os.path.join(self.dir, "sumP_plasma.txt.gz"),
            "ID_REF\tPcrc90\tPn43\nr1\t0.2\t0.1\nr2\t0.3\t0.2\n")
        self.tissue = _write_gz(
            os.path.join(self.dir, "MePM_tissue.txt.gz"),
            "ID_REF\tTcrc1\tTnm1\nr1\t12\t5\nr2\t8\t3\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_llm_keeps_plasma_drops_tissue(self):
        payload = {"reasoning": "plasma file matches cfDNA query; tissue does not",
                   "files": [
                       {"name": "sumP_plasma.txt.gz", "keep": True,
                        "sample_type": "plasma", "reason": "plasma cfDNA"},
                       {"name": "MePM_tissue.txt.gz", "keep": False,
                        "sample_type": "tissue", "reason": "tissue, not cfDNA"},
                   ]}
        skill = _make_skill(FakeLLM(payload))
        rec = {"raw_query": "CRC cfDNA methylation", "sample_type": "cfdna",
               "cancer_type": "colorectal cancer", "title": "CRC cfDNA"}
        # Use the real entry point _select_relevant_files (junk-filter + LLM + commit).
        kept, discarded, note, forced = skill._select_relevant_files(
            "GSETEST",
            [_done_result(self.plasma), _done_result(self.tissue)],
            rec, None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(discarded), 1)
        self.assertIsNone(forced)  # normal path, not the fallback
        self.assertTrue(os.path.exists(self.plasma))
        self.assertFalse(os.path.exists(self.tissue))  # deleted on commit
        self.assertEqual(kept[0]["data_form"], "plasma")
        self.assertIn("selected 1/2", note)
        self.assertIn("cfDNA", note)  # reasoning surfaced in the note


class TestSelectRelevantFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.beta = _write_gz(
            os.path.join(self.dir, "beta.txt.gz"),
            "ID_REF\tGSM1\tGSM2\ncg1\t0.1\t0.2\ncg2\t0.3\t0.4\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_llm_keeps_all_nonjunk_and_flags_review(self):
        skill = _make_skill(llm=None)
        kept, discarded, note, forced = skill._select_relevant_files(
            "GSETEST", [_done_result(self.beta)], {}, None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(discarded), 0)
        self.assertEqual(forced, "manual_review_file_selection")
        self.assertIn("manual_review", note)
        self.assertTrue(os.path.exists(self.beta))

    def test_llm_keeps_none_reverts_to_all_and_flags_review(self):
        payload = {"reasoning": "unsure", "files": [
            {"name": "beta.txt.gz", "keep": False, "sample_type": "unknown",
             "reason": "unsure"}]}
        skill = _make_skill(FakeLLM(payload))
        kept, discarded, note, forced = skill._select_relevant_files(
            "GSETEST", [_done_result(self.beta)], {}, None)
        self.assertEqual(len(kept), 1)            # reverted: keep all non-junk
        self.assertEqual(forced, "manual_review_file_selection")
        self.assertTrue(os.path.exists(self.beta))

    def test_llm_parse_error_falls_back(self):
        class BadLLM:
            def invoke(self, msgs):
                return _Resp("not json at all {{{")
        skill = _make_skill(BadLLM())
        kept, discarded, note, forced = skill._select_relevant_files(
            "GSETEST", [_done_result(self.beta)], {}, None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(forced, "manual_review_file_selection")


class TestTargetSampleSummary(unittest.TestCase):
    def test_summary_from_metadata(self):
        df = pd.DataFrame({
            "gsm": ["GSM1", "GSM2", "GSM3"],
            "source_name": ["Plasma", "Plasma", "Colorectal cancer tissue"],
            "molecule": ["genomic DNA"] * 3,
            "group": ["plasma_cfdna", "plasma_cfdna", "tissue"],
            "cancer": ["query_cancer", "control", "query_cancer"],
            "abcd1234": ["download", "download", "not download"],
        })
        s = DownloadSkill._target_sample_summary(df)
        self.assertIn("2 sample(s)", s)
        self.assertIn("Plasma=2", s)
        self.assertIn("plasma_cfdna=2", s)

    def test_no_metadata(self):
        self.assertIn("unavailable", DownloadSkill._target_sample_summary(None))


if __name__ == "__main__":
    unittest.main()
