"""Tests for the geo-search skill (deterministic GEO recall)."""
import unittest
from unittest.mock import MagicMock

import yaml

from skills.geo_search import SearchSkill


class TestSearchSkill(unittest.TestCase):
    def setUp(self):
        cfg = yaml.safe_load(open("config/settings.yaml"))
        self.sk = SearchSkill(cfg)
        self.sk.geo_client = MagicMock()
        self.sk.geo_client.search_gse.return_value = ["GSE1", "GSE2"]
        self.sk.geo_client.filter_methylation_datasets.return_value = [
            {"accession": "GSE1", "title": "t1", "summary": "s", "overall_design": "d",
             "platform_canonical": "450K", "sample_count": 10, "pubmed_ids": [], "data_type": "array"},
            {"accession": "GSE2", "title": "t2", "summary": "s", "overall_design": "d",
             "platform_canonical": "EPIC", "sample_count": 20, "pubmed_ids": ["123"], "data_type": "array"},
        ]

    def test_semantic_returns_full_dicts_and_injects_cancer(self):
        out = self.sk.run({"parsed_intent": {
            "mode": "semantic",
            "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
            "sample_type": "cfdna", "sample_types": ["cfdna"],
        }})
        cands = out["candidate_gse_list"]
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0]["accession"], "GSE1")
        # full dict preserved (overall_design, sample_count, pubmed_ids)
        self.assertEqual(cands[1]["pubmed_ids"], ["123"])
        # cancer_type injected from intent
        self.assertEqual(cands[0]["cancer_type"], "colorectal cancer")
        # search query captured for logging
        self.assertEqual(len(out["search_queries"]), 1)
        self.assertIn("search_log", out)

    def test_accession_mode_fetches_each(self):
        self.sk.geo_client.get_series_metadata.side_effect = lambda acc: {"accession": acc, "title": acc}
        out = self.sk.run({"parsed_intent": {
            "mode": "accession", "accessions": {"geo": ["GSE9"]},
        }})
        self.assertEqual(len(out["candidate_gse_list"]), 1)
        self.assertEqual(out["candidate_gse_list"][0]["accession"], "GSE9")
        # semantic search not used in accession mode
        self.sk.geo_client.search_gse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
