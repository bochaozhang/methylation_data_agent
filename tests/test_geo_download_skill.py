"""Tests for the geo-download skill (Phase 1 download + Phase 2 cancer subset)."""
import gzip
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from skills.geo_download import DownloadSkill


class TestDownloadSkill(unittest.TestCase):
    def setUp(self):
        cfg = yaml.safe_load(open("config/settings.yaml"))
        self.sk = DownloadSkill(cfg)
        self.sk.downloader = MagicMock()
        self.sk.geo_client = MagicMock()
        # Phase 1 tests only exercise download aggregation, not the Phase 2
        # cancer-labeling (which would hit the network via get_all_gsm_metadata).
        # Patch build_sample_metadata_with_cancer to skip subsetting (sm=None).
        patcher = patch("skills.geo_download.skill.build_sample_metadata_with_cancer",
                        return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_aggregates_success(self):
        self.sk.downloader.download_many_sync.return_value = [
            {"accession": "GSE1", "status": "done",
             "local_path": "/x/GSE1/m.txt.gz", "file_size_bytes": 100, "checksum_md5": "abc"},
        ]
        out = self.sk.run({"download_list": [{"accession": "GSE1", "source": "GEO",
                                              "flags": "case_only"}],
                           "output_dir": "./data"})
        r = out["download_results"][0]
        self.assertEqual(r["outcome_final"], "download_success")
        self.assertEqual(r["files_downloaded"][0]["size_bytes"], 100)
        self.assertEqual(r["files_downloaded"][0]["provenance"]["checksum_md5"], "abc")
        # flags inherited from filter record
        self.assertEqual(r["flags"], "case_only")

    def test_failed_outcome(self):
        self.sk.downloader.download_many_sync.return_value = [
            {"accession": "GSE2", "status": "failed", "error": "404"},
        ]
        out = self.sk.run({"download_list": [{"accession": "GSE2", "source": "GEO"}],
                           "output_dir": "./data"})
        self.assertEqual(out["download_results"][0]["outcome_final"], "failed")
        self.assertIn("404", out["download_results"][0]["notes"])

    def test_empty_download_list(self):
        out = self.sk.run({"download_list": [], "output_dir": "./data"})
        self.assertEqual(out["download_results"], [])
        self.sk.downloader.download_many_sync.assert_not_called()


class TestDownloadCancerSubset(unittest.TestCase):
    """Phase 2c: cancer subset + manual_review on unclear labels."""

    def setUp(self):
        cfg = yaml.safe_load(open("config/settings.yaml"))
        self.sk = DownloadSkill(cfg)
        self.sk.downloader = MagicMock()
        self.sk.geo_client = MagicMock()  # avoid real network in __init__-built client
        self.tmp = tempfile.mkdtemp()
        # synthetic gzip matrix with 4 GSM columns
        self.mtx_path = os.path.join(self.tmp, "GSE1_beta.txt.gz")
        with gzip.open(self.mtx_path, "wt") as f:
            f.write("cg\tGSM1\tGSM2\tGSM3\tGSM4\n")
            f.write("cg1\t0.1\t0.2\t0.3\t0.4\ncg2\t0.5\t0.6\t0.7\t0.8\n")

    def _state(self):
        return {
            "download_list": [{"accession": "GSE1", "source": "GEO", "flags": "",
                               "available_file_type": "beta"}],
            "parsed_intent": {"cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"}},
            "output_dir": self.tmp,
        }

    @patch("skills.geo_download.skill.build_sample_metadata_with_cancer")
    def test_multicancer_subset(self, mock_sm):
        # multi-cancer labels (2 query, 1 unclear, 1 control) → subset to query GSMs
        import pandas as pd
        mock_sm.return_value = pd.DataFrame([
            {"gsm": "GSM1", "cancer": "query_cancer"},
            {"gsm": "GSM2", "cancer": "unclear"},
            {"gsm": "GSM3", "cancer": "query_cancer"},
            {"gsm": "GSM4", "cancer": "control"},
        ])
        self.sk.downloader.download_many_sync.return_value = [
            {"accession": "GSE1", "status": "done", "local_path": self.mtx_path,
             "file_size_bytes": 100, "url": "https://x/GSE1_beta.txt.gz"}]

        out = self.sk.run(self._state())
        r = out["download_results"][0]
        self.assertEqual(r["outcome_final"], "download_success")
        self.assertIsNotNone(r["subset_path"])
        # subset file keeps feature col + query GSM columns (GSM1, GSM3)
        kept = gzip.open(r["subset_path"], "rt").readline().strip().split("\t")
        self.assertIn("GSM1", kept)
        self.assertIn("GSM3", kept)
        self.assertNotIn("GSM2", kept)

    @patch("skills.geo_download.skill.build_sample_metadata_with_cancer")
    def test_unclear_majority_goes_to_manual_review(self, mock_sm):
        import pandas as pd
        # 4 unclear of 5 → 80% unclear > threshold → manual_review
        mock_sm.return_value = pd.DataFrame([
            {"gsm": "GSM1", "cancer": "query_cancer"},
            {"gsm": "GSM2", "cancer": "unclear"},
            {"gsm": "GSM3", "cancer": "unclear"},
            {"gsm": "GSM4", "cancer": "unclear"},
            {"gsm": "GSM5", "cancer": "unclear"},
        ])
        self.sk.downloader.download_many_sync.return_value = [
            {"accession": "GSE1", "status": "done", "local_path": self.mtx_path,
             "file_size_bytes": 100, "url": "https://x/GSE1_beta.txt.gz"}]

        out = self.sk.run(self._state())
        r = out["download_results"][0]
        self.assertEqual(r["outcome_final"], "qc_failed_reverted_manual_review")
        self.assertIsNone(r["subset_path"])
        self.assertIn("unclear", r["notes"])

    @patch("skills.geo_download.skill.build_sample_metadata_with_cancer")
    def test_unclear_single_cancer_fallback(self, mock_sm):
        import pandas as pd
        # all unclear, but dataset cancer_type matches query cancer → assume
        # single-cancer (no subset, success — NOT manual_review).
        mock_sm.return_value = pd.DataFrame([
            {"gsm": "GSM1", "cancer": "unclear"},
            {"gsm": "GSM2", "cancer": "unclear"},
            {"gsm": "GSM3", "cancer": "unclear"},
        ])
        self.sk.downloader.download_many_sync.return_value = [
            {"accession": "GSE1", "status": "done", "local_path": self.mtx_path,
             "file_size_bytes": 100, "url": "https://x/GSE1_beta.txt.gz"}]
        state = self._state()
        state["download_list"][0]["cancer_type"] = "colorectal cancer"  # matches query COAD
        out = self.sk.run(state)
        r = out["download_results"][0]
        self.assertEqual(r["outcome_final"], "download_success")  # not manual_review
        self.assertIsNone(r["subset_path"])  # no subset (single-cancer assumed)
        self.assertIn("single-cancer assumed", r["notes"])


if __name__ == "__main__":
    unittest.main()
