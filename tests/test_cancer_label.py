"""Tests for skills/geo_download/cancer_label.py (Phase 2b per-GSM cancer labeling)."""
import unittest

from skills.geo_download.cancer_label import label_gsm_cancer, query_cancer_terms


class TestCancerLabel(unittest.TestCase):
    def setUp(self):
        self.qt = query_cancer_terms(
            {"cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"}})

    def test_query_terms_include_synonyms(self):
        # display + synonyms from synonyms.yaml (CRC, colon cancer, ...)
        self.assertIn("colorectal cancer", self.qt)
        self.assertIn("crc", self.qt)

    def test_label_query_match(self):
        self.assertEqual(label_gsm_cancer({"disease state": "colorectal cancer"}, self.qt), "query_cancer")
        self.assertEqual(label_gsm_cancer({"disease": "CRC tumor"}, self.qt), "query_cancer")

    def test_label_control(self):
        self.assertEqual(label_gsm_cancer({"disease state": "healthy control"}, self.qt), "control")
        self.assertEqual(label_gsm_cancer({"status": "normal"}, self.qt), "control")

    def test_label_unclear(self):
        self.assertEqual(label_gsm_cancer({"source": "sample A"}, self.qt), "unclear")
        self.assertEqual(label_gsm_cancer({}, self.qt), "unclear")


if __name__ == "__main__":
    unittest.main()
