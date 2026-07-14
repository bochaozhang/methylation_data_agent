"""Tests for the agent1 skill pipeline: graph compilation + registry bridge."""
import unittest
from unittest.mock import MagicMock, patch

import yaml


class TestAgent1PipelineBuild(unittest.TestCase):
    def test_pipeline_compiles_with_all_nodes(self):
        with patch("agents.agent1_pipeline.get_llm", return_value=MagicMock()):
            from agents.agent1_pipeline import build_agent1_pipeline
            cfg = yaml.safe_load(open("config/settings.yaml"))
            app = build_agent1_pipeline(cfg, registry=None)
            nodes = set(app.get_graph().nodes)
            for n in ("parse", "search", "filter", "download", "tcga", "register"):
                self.assertIn(n, nodes, f"missing node {n}")


class TestRegisterBridge(unittest.TestCase):
    def _state(self):
        return {
            "download_list": [
                {"accession": "GSE1", "source": "GEO", "pubmed_ids": [], "sample_count": 10},
                {"accession": "GSE_FAIL", "source": "GEO", "pubmed_ids": [], "sample_count": 5},
            ],
            "download_results": [
                {"accession": "GSE1", "outcome_final": "download_success",
                 "files_downloaded": [{"local_path": "/x/GSE1/m.txt", "size_bytes": 100}]},
                {"accession": "GSE_FAIL", "outcome_final": "failed",
                 "files_downloaded": []},
            ],
            "lead_list": [{"accession": "GSE2", "source": "GEO"}],
            "manual_review_list": [{"accession": "GSE3", "source": "GEO"}],
            "exclude_list": [{"accession": "GSE4"}],
            "tcga_results": [
                {"accession": "TCGA-COAD", "outcome_final": "download_success",
                 "files_downloaded": [{"local_path": "/x/t.txt", "size_bytes": 50}]},
            ],
        }

    def test_counts(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        n = register_state_to_registry(self._state(), reg)
        self.assertEqual(n, {"downloaded": 1, "failed": 1, "review": 2,
                             "excluded": 1, "tcga": 1})

    def test_status_mapping(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        register_state_to_registry(self._state(), reg)
        statuses = [c.kwargs.get("download_status")
                    for c in reg.upsert_dataset.call_args_list]
        # GSE1 → done, GSE_FAIL → failed, GSE2/GSE3 → awaiting_approval, TCGA → done
        self.assertEqual(statuses.count("done"), 2)        # GSE1 + TCGA
        self.assertEqual(statuses.count("failed"), 1)      # GSE_FAIL
        self.assertEqual(statuses.count("awaiting_approval"), 2)  # lead + manual_review

    def test_done_records_get_local_path(self):
        from agents.agent1_pipeline import register_state_to_registry
        reg = MagicMock()
        register_state_to_registry(self._state(), reg)
        # update_status called for done datasets with local_path/size
        upd = [c.kwargs for c in reg.update_status.call_args_list]
        paths = [u.get("local_path") for u in upd]
        self.assertIn("/x/GSE1/m.txt", paths)
        self.assertIn("/x/t.txt", paths)


if __name__ == "__main__":
    unittest.main()
