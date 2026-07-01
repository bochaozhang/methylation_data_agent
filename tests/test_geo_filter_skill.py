"""
Tests for the spec-driven geo_filter skill.

Covers: SYSTEM_PROMPT loading, structured-verdict parsing (plain + fenced),
error fallback (conservative manual_review), and apply_verdict field mapping
including sample-count correction and notes accumulation.

LLM is mocked with the same MagicMock pattern used in test_llm_extractor.py.
The 4-GSE regression (GSE122126 / GSE110185 / GSE79277 / GSE97932) requires
live NCBI + LLM access and is documented in the plan as a manual/CI eval;
these unit tests pin the parsing/mapping logic that the regression relies on.
"""
import json
import unittest
from unittest.mock import MagicMock

from skills.geo_filter import (
    SPEC,
    SYSTEM_PROMPT,
    apply_verdict,
    filter_dataset,
)
from skills.geo_filter.grouping import classify_group, group_summary


def make_mock_llm(response_text: str):
    """Mock LLM whose .invoke() returns an object with .content = response_text."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock.invoke.return_value = mock_response
    return mock


def make_mock_llm_json(payload: dict):
    return make_mock_llm(json.dumps(payload))


INTENT = {
    "raw_query": "结直肠癌cfDNA甲基化血浆数据",
    "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
    "sample_type": "cfdna",
    "sample_types": ["cfdna", "non_cancer"],
    "platform": "450K",
}

DS = {
    "accession": "GSE999999",
    "title": "Plasma cfDNA methylation in CRC",
    "summary": "Cell-free DNA from colorectal cancer patients and healthy controls.",
    "overall_design": "450K array, plasma cfDNA",
    "platform_canonical": "450K",
    "sample_count": 100,
    "sample_type": "plasma",
    "cancer_type": "colorectal cancer",
    "pubmed_ids": ["12345678"],
}

GSM_DETAILS = [
    {"gsm": "GSM1", "source_name": "plasma", "molecule": "genomic DNA",
     "characteristics": {"disease state": "colorectal cancer"}, "group": "plasma_cfdna"},
    {"gsm": "GSM2", "source_name": "plasma", "molecule": "genomic DNA",
     "characteristics": {"disease state": "healthy"}, "group": "plasma_cfdna"},
]


class TestSpecLoading(unittest.TestCase):
    def test_spec_and_contract_loaded(self):
        self.assertTrue(SPEC.strip().startswith("# GEO"))
        self.assertIn("OUTPUT CONTRACT", SYSTEM_PROMPT)
        # SPEC is the canonical procedure; contract is appended after it.
        self.assertGreater(len(SYSTEM_PROMPT), len(SPEC))

    def test_grouping_helpers(self):
        self.assertEqual(classify_group("plasma cfDNA"), "plasma_cfdna")
        self.assertEqual(classify_group("HCT116 cell line"), "cell_line")
        self.assertEqual(classify_group("Sample 1"), "unknown")
        summary = group_summary(GSM_DETAILS)
        self.assertEqual(summary["plasma_cfdna"], 2)


class TestFilterDatasetParsing(unittest.TestCase):
    def test_parses_plain_json_keep(self):
        payload = {
            "usable": "yes",
            "recommended_action": "keep",
            "confirmed_sample_type": "cfdna",
            "confirmed_cancer_type": "colorectal cancer",
            "sample_count_in_paper": 100,
            "stage_treatment": "treatment-naive",
            "consistency": "consistent",
            "sample_level_annotation": "yes",
            "available_file_type": "normalized beta matrix",
            "disease_groups": "50 CRC vs 50 healthy",
            "reason": "plasma cfDNA, CRC vs healthy",
            "notes": "",
            "gsm_includes": [
                {"gsm": "GSM1", "include": True, "reason": None},
                {"gsm": "GSM2", "include": True, "reason": None},
            ],
        }
        verdict = filter_dataset(make_mock_llm_json(payload), DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["recommended_action"], "keep")
        self.assertEqual(verdict["usable"], "yes")
        self.assertEqual(len(verdict["gsm_includes"]), 2)

    def test_strips_markdown_fences(self):
        payload = {"recommended_action": "exclude", "usable": "no",
                   "reason": "cell lines", "gsm_includes": []}
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        verdict = filter_dataset(make_mock_llm(fenced), DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["recommended_action"], "exclude")

    def test_normalises_bad_enum(self):
        payload = {"recommended_action": "YES", "usable": "definitely", "gsm_includes": []}
        verdict = filter_dataset(make_mock_llm_json(payload), DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["recommended_action"], "manual_review")
        self.assertEqual(verdict["usable"], "unclear")

    def test_error_falls_back_to_manual_review(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("boom")
        verdict = filter_dataset(llm, DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["recommended_action"], "manual_review")
        self.assertIn("filter_error", verdict["notes"])


class TestApplyVerdict(unittest.TestCase):
    def test_keep_maps_usable_and_fields(self):
        verdict = {
            "recommended_action": "keep",
            "usable": "yes",
            "confirmed_sample_type": "cfdna",
            "confirmed_cancer_type": "colorectal cancer",
            "stage_treatment": "treatment-naive",
            "available_file_type": "beta matrix",
            "sample_level_annotation": "yes",
            "disease_groups": "CRC vs healthy",
            "consistency": "consistent",
            "reason": "ok",
            "notes": "",
            "gsm_includes": [],
        }
        out = apply_verdict(DS, verdict)
        self.assertEqual(out["usable"], 1)
        self.assertEqual(out["sample_type"], "cfdna")
        self.assertEqual(out["stage_treatment"], "treatment-naive")
        self.assertEqual(out["_verdict"], verdict)

    def test_exclude_maps_usable_zero(self):
        verdict = {"recommended_action": "exclude", "usable": "no",
                   "reason": "cell line", "gsm_includes": []}
        out = apply_verdict(DS, verdict)
        self.assertEqual(out["usable"], 0)

    def test_sample_count_corrected_when_drift_large(self):
        # GEO says 100, paper says 150 → 50% drift → corrected, noted.
        verdict = {"recommended_action": "keep", "usable": "yes",
                   "sample_count_in_paper": 150, "gsm_includes": []}
        out = apply_verdict(dict(DS), verdict)
        self.assertEqual(out["sample_count"], 150)
        self.assertIn("sample_count GEO=100 paper=150", out["notes"])

    def test_sample_count_kept_when_drift_small(self):
        verdict = {"recommended_action": "keep", "usable": "yes",
                   "sample_count_in_paper": 105, "gsm_includes": []}
        out = apply_verdict(dict(DS), verdict)
        self.assertEqual(out["sample_count"], 100)  # unchanged (5% drift)

    def test_notes_appended_not_overwritten(self):
        ds = dict(DS)
        ds["notes"] = "no_pubmed_link"
        verdict = {"recommended_action": "manual_review", "usable": "unclear",
                   "notes": "conflicting sample type", "gsm_includes": []}
        out = apply_verdict(ds, verdict)
        self.assertIn("no_pubmed_link", out["notes"])
        self.assertIn("conflicting sample type", out["notes"])


if __name__ == "__main__":
    unittest.main()
