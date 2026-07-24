"""Tests for the agent1 skill pipeline: graph compilation + registry bridge.

Pipeline registers: download/tcga → pending (auto-download), lead/manual_review →
awaiting_approval (Review Queue), exclude → skipped.
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
        # auto_download = download(GSE1) + tcga(TCGA-COAD) = 2
        # review = lead(GSE2) + manual_review(GSE3) = 2 ; excluded = 1
        self.assertEqual(n, {"auto_download": 2, "review": 2, "excluded": 1})

    def test_download_and_tcga_go_to_pending(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        register_state_to_registry(self._state(), reg)
        statuses = [c.kwargs.get("download_status") for c in reg.upsert_dataset.call_args_list]
        # GSE1 + TCGA-COAD → pending
        self.assertEqual(statuses.count("pending"), 2)
        # GSE2 + GSE3 → awaiting_approval
        self.assertEqual(statuses.count("awaiting_approval"), 2)

    def test_lead_and_manual_review_both_needs_review_1(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        register_state_to_registry(self._state(), reg)
        statuses = {c.kwargs.get("accession"): c.kwargs for c in reg.upsert_dataset.call_args_list}
        # lead (GSE2) and manual_review (GSE3) both needs_review=True
        self.assertTrue(statuses["GSE2"]["needs_review"])
        self.assertTrue(statuses["GSE3"]["needs_review"])
        # download (GSE1) and tcga → needs_review=False
        self.assertFalse(statuses["GSE1"]["needs_review"])

    def test_exclude_not_upserted(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        n = register_state_to_registry(self._state(), reg)
        accessions = [c.kwargs.get("accession") for c in reg.upsert_dataset.call_args_list]
        self.assertNotIn("GSE4", accessions)
        self.assertEqual(n["excluded"], 1)


if __name__ == "__main__":
    unittest.main()
