"""Tests for the agent1 skill pipeline: graph compilation + registry bridge.

Pipeline no longer downloads inline — register puts download/lead/tcga into the
bulk "待下载" bucket (awaiting_approval, needs_review=0) and manual_review into
the Review Queue (awaiting_approval, needs_review=1); exclude is counted only.
"""
import unittest
from unittest.mock import MagicMock, patch

import yaml


class TestAgent1PipelineBuild(unittest.TestCase):
    def test_pipeline_compiles_no_inline_download(self):
        with patch("agents.agent1_pipeline.get_llm", return_value=MagicMock()):
            from agents.agent1_pipeline import build_agent1_pipeline
            cfg = yaml.safe_load(open("config/settings.yaml"))
            app = build_agent1_pipeline(cfg, registry=None)
            nodes = set(app.get_graph().nodes)
            for n in ("parse", "search", "filter", "tcga", "register"):
                self.assertIn(n, nodes, f"missing node {n}")
            # download node removed (no inline download)
            self.assertNotIn("download", nodes)


class TestRegisterBridge(unittest.TestCase):
    def _state(self):
        return {
            "download_list": [{"accession": "GSE1", "source": "GEO", "pubmed_ids": []}],
            "lead_list": [{"accession": "GSE2", "source": "GEO"}],
            "manual_review_list": [{"accession": "GSE3", "source": "GEO"}],
            "exclude_list": [{"accession": "GSE4"}],
            "tcga_candidates": [{"accession": "TCGA-COAD", "source": "TCGA"}],
        }

    def test_counts(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        n = register_state_to_registry(self._state(), reg)
        # bucket = download(GSE1) + lead(GSE2) + tcga(TCGA-COAD) = 3
        # review = manual_review(GSE3) = 1 ; excluded = 1
        self.assertEqual(n, {"bucket": 3, "review": 1, "excluded": 1})

    def test_all_upserted_to_awaiting_approval(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        register_state_to_registry(self._state(), reg)
        statuses = [c.kwargs.get("download_status")
                    for c in reg.upsert_dataset.call_args_list]
        # everything registered goes to awaiting_approval (no done/failed)
        self.assertEqual(statuses.count("awaiting_approval"), 4)  # GSE1,GSE2,GSE3,TCGA
        self.assertNotIn("done", statuses)
        self.assertNotIn("failed", statuses)

    def test_needs_review_split(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        register_state_to_registry(self._state(), reg)
        needs = [c.kwargs.get("needs_review") for c in reg.upsert_dataset.call_args_list]
        # download/lead/tcga → needs_review=0 (bulk bucket); manual_review → needs_review=1
        self.assertEqual(needs.count(False), 3)   # GSE1, GSE2, TCGA
        self.assertEqual(needs.count(True), 1)    # GSE3 manual_review

    def test_exclude_not_upserted(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        n = register_state_to_registry(self._state(), reg)
        # exclude (GSE4) counted but not written
        accessions = [c.kwargs.get("accession") for c in reg.upsert_dataset.call_args_list]
        self.assertNotIn("GSE4", accessions)
        self.assertEqual(n["excluded"], 1)


if __name__ == "__main__":
    unittest.main()
