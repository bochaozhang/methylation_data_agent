"""Tests for the geo-download skill (Phase 1: download + md5 aggregation)."""
import unittest
from unittest.mock import MagicMock

import yaml

from skills.geo_download import DownloadSkill


class TestDownloadSkill(unittest.TestCase):
    def setUp(self):
        cfg = yaml.safe_load(open("config/settings.yaml"))
        self.sk = DownloadSkill(cfg)
        self.sk.downloader = MagicMock()

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


if __name__ == "__main__":
    unittest.main()
