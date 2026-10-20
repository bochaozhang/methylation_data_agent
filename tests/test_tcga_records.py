"""
Tests for TCGA record verdict semantics.

Upstream feedback (2026-10-20): the registry's single TCGA-COAD row had
recommended_action=NULL (tcga-direct bypasses geo-filter by design, and the
register path never stamped a verdict) → get_task_result dropped it into a
semantically-unclear "unknown" bucket that consumers conservatively skip.
Fix: files_to_dataset_records stamps a design-trust download verdict plus
disease_groups (real GDC sample_type counts — the field platforms parse for
control counts) and sample_level_annotation.
"""
import tempfile
import unittest
from pathlib import Path

from agents.agent1_pipeline import register_state_to_registry
from registry.registry import Registry
from tools.tcga_tools import GDCClient, _count_sample_types


def _hit(sample_types, project="TCGA-COAD"):
    return {
        "file_id": f"uuid-{'-'.join(sample_types)}"[:40],
        "file_size": 1000,
        "platform": "Illumina Human Methylation 450",
        "cases": [{
            "project": {"project_id": project},
            "case_id": "c1",
            "samples": [{"sample_type": st} for st in sample_types],
        }],
    }


class TestCountSampleTypes(unittest.TestCase):
    def test_counts_per_type(self):
        out = _count_sample_types([_hit(["Primary Tumor"]), _hit(["Primary Tumor"]),
                                   _hit(["Solid Tissue Normal"])])
        self.assertEqual(out, "Primary Tumor 2 / Solid Tissue Normal 1")

    def test_no_sample_info_returns_none(self):
        self.assertIsNone(_count_sample_types([{"cases": []}]))
        self.assertIsNone(_count_sample_types([{"cases": [{"samples": []}]}]))


class TestFilesToDatasetRecords(unittest.TestCase):
    def test_verdict_fields_stamped(self):
        recs = GDCClient().files_to_dataset_records(
            [_hit(["Primary Tumor"]), _hit(["Primary Tumor"]),
             _hit(["Solid Tissue Normal"]), _hit(["Solid Tissue Normal"])],
            "COAD",
        )
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["recommended_action"], "download")
        self.assertIn("trusted by design", r["reason"])
        self.assertEqual(r["disease_groups"], "Primary Tumor 2 / Solid Tissue Normal 2")
        self.assertEqual(r["sample_level_annotation"], "yes")


class TestRegisterPath(unittest.TestCase):
    def test_tcga_candidate_registered_with_verdict(self):
        """End-to-end: register a tcga_candidate → row carries recommended_action."""
        tmp = tempfile.TemporaryDirectory()
        try:
            reg = Registry(db_path=str(Path(tmp.name) / "reg.db"))
            state = {
                "task_id": "t-tcga",
                "raw_query": "structured test",
                "output_dir": tmp.name,
                "tcga_candidates": [{
                    "accession": "TCGA-READ",
                    "source": "TCGA",
                    "data_type": "array",
                    "cancer_type": "READ",
                    "platform": "450K",
                    "sample_count": 41,
                    "title": "Rectum adenocarcinoma",
                    "recommended_action": "download",
                    "reason": "TCGA TCGA-READ Level-3 methylation (GDC) — trusted by design",
                    "disease_groups": "Primary Tumor 30 / Solid Tissue Normal 11",
                    "sample_level_annotation": "yes",
                }],
            }
            n = register_state_to_registry(state, reg)
            self.assertEqual(n["auto_download"], 1)
            row = reg.get("TCGA-READ")
            self.assertEqual(row["recommended_action"], "download")
            self.assertIn("Primary Tumor 30", row["disease_groups"])
            self.assertEqual(row["download_status"], "pending")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
