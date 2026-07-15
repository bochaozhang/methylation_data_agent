"""Tests for the TCGA recall module (keyword → GDC → candidates, search-only)."""
import unittest
from unittest.mock import MagicMock, patch

import yaml

from agents.tcga_direct import run_tcga_direct


class TestTcgaDirect(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(open("config/settings.yaml"))

    def test_skip_when_no_cancer_code(self):
        out = run_tcga_direct({"parsed_intent": {"cancer_type": None}}, self.cfg)
        self.assertEqual(out["tcga_candidates"], [])
        self.assertIn("no cancer_type", out["tcga_log"])

    def test_skip_liquid_biopsy_request(self):
        out = run_tcga_direct({"parsed_intent": {
            "cancer_type": {"tcga_code": "COAD"}, "sample_type": "cfdna",
        }}, self.cfg)
        self.assertEqual(out["tcga_candidates"], [])
        self.assertIn("skipped", out["tcga_log"])

    @patch("agents.tcga_direct.GDCClient")
    def test_search_returns_candidates(self, MockGDC):
        gdc = MockGDC.return_value
        gdc.search_methylation_files.return_value = [{"file_id": "f1"}]
        gdc.files_to_dataset_records.return_value = [
            {"accession": "TCGA-COAD", "file_ids": ["f1"]}]

        out = run_tcga_direct({"parsed_intent": {
            "cancer_type": {"tcga_code": "COAD"}, "sample_type": "tumor",
        }}, self.cfg)

        gdc.search_methylation_files.assert_called_once()
        self.assertEqual(len(out["tcga_candidates"]), 1)
        self.assertEqual(out["tcga_candidates"][0]["accession"], "TCGA-COAD")
        # source stamped as TCGA
        self.assertEqual(out["tcga_candidates"][0]["source"], "TCGA")

    @patch("agents.tcga_direct.GDCClient")
    def test_no_gdc_files(self, MockGDC):
        gdc = MockGDC.return_value
        gdc.search_methylation_files.return_value = []
        out = run_tcga_direct({"parsed_intent": {
            "cancer_type": {"tcga_code": "COAD"}, "sample_type": "tumor",
        }}, self.cfg)
        self.assertEqual(out["tcga_candidates"], [])


if __name__ == "__main__":
    unittest.main()
