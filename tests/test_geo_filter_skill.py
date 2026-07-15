"""
Tests for the spec-driven geo_filter skill (four-state outcome contract).

Covers: SYSTEM_PROMPT loading, structured-verdict parsing (plain + fenced),
outcome normalisation + legacy-field derivation, error fallback (manual_review),
apply_verdict field mapping (incl. sample-count correction, notes accumulation,
available_file_type derived from files[]), and split_by_outcome.

LLM is mocked with the same MagicMock pattern used in test_llm_extractor.py.
"""
import json
import unittest
from unittest.mock import MagicMock

from skills.geo_filter import (
    SPEC,
    SYSTEM_PROMPT,
    apply_verdict,
    filter_dataset,
    split_by_outcome,
)
from skills.geo_filter.grouping import classify_group, group_summary


def make_mock_llm(response_text: str):
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
        self.assertGreater(len(SYSTEM_PROMPT), len(SPEC))

    def test_grouping_helpers(self):
        self.assertEqual(classify_group("plasma cfDNA"), "plasma_cfdna")
        self.assertEqual(classify_group("HCT116 cell line"), "cell_line")
        self.assertEqual(classify_group("Sample 1"), "unknown")
        summary = group_summary(GSM_DETAILS)
        self.assertEqual(summary["plasma_cfdna"], 2)


class TestFilterDatasetParsing(unittest.TestCase):
    def test_parses_plain_json_download(self):
        payload = {
            "outcome": "download",
            "confirmed_sample_type": "cfdna",
            "confirmed_cancer_type": "colorectal cancer",
            "sample_count_in_paper": 100,
            "stage_or_treatment_status": "treatment-naive",
            "consistency": "consistent",
            "sample_level_annotation": "yes",
            "technology": "450K",
            "files": [
                {"name": "GSE999999_beta_matrix.txt.gz", "is_A_level": True,
                 "download": True, "data_form": "merged_beta_matrix", "reason": "beta matrix"}
            ],
            "flags": "",
            "disease_groups": "50 CRC vs 50 healthy",
            "reason": "plasma cfDNA, CRC vs healthy",
            "notes": "",
            "gsm_includes": [
                {"gsm": "GSM1", "include": True, "reason": None},
                {"gsm": "GSM2", "include": True, "reason": None},
            ],
        }
        verdict = filter_dataset(make_mock_llm_json(payload), DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["outcome"], "download")
        # recommended_action now mirrors the true outcome (download/lead/exclude/manual_review)
        self.assertEqual(verdict["recommended_action"], "download")
        self.assertEqual(verdict["usable"], "yes")
        self.assertEqual(len(verdict["gsm_includes"]), 2)
        self.assertEqual(len(verdict["files"]), 1)

    def test_strips_markdown_fences(self):
        payload = {"outcome": "exclude", "exclude_reason": "cell_line",
                   "reason": "cell lines", "gsm_includes": []}
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        verdict = filter_dataset(make_mock_llm(fenced), DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["outcome"], "exclude")

    def test_normalises_bad_outcome(self):
        payload = {"outcome": "YES", "gsm_includes": []}
        verdict = filter_dataset(make_mock_llm_json(payload), DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["outcome"], "manual_review")
        self.assertEqual(verdict["usable"], "unclear")

    def test_error_falls_back_to_manual_review(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("boom")
        verdict = filter_dataset(llm, DS, INTENT, GSM_DETAILS)
        self.assertEqual(verdict["outcome"], "manual_review")
        self.assertIn("filter_error", verdict["notes"])


class TestApplyVerdict(unittest.TestCase):
    def test_download_maps_usable_and_fields(self):
        verdict = {
            "outcome": "download",
            "recommended_action": "keep",
            "usable": "yes",
            "confirmed_sample_type": "cfdna",
            "confirmed_cancer_type": "colorectal cancer",
            "stage_or_treatment_status": "treatment-naive",
            "sample_level_annotation": "yes",
            "disease_groups": "CRC vs healthy",
            "consistency": "consistent",
            "files": [{"name": "beta.txt.gz", "is_A_level": True, "download": True,
                       "data_form": "merged_beta_matrix", "reason": ""}],
            "flags": "",
            "reason": "ok",
            "notes": "",
            "gsm_includes": [],
        }
        out = apply_verdict(DS, verdict)
        self.assertEqual(out["outcome"], "download")
        self.assertEqual(out["usable"], 1)
        self.assertEqual(out["sample_type"], "cfdna")
        self.assertEqual(out["stage_treatment"], "treatment-naive")
        self.assertEqual(out["available_file_type"], "merged_beta_matrix")
        self.assertEqual(out["_verdict"], verdict)

    def test_exclude_maps_usable_zero(self):
        verdict = {"outcome": "exclude", "recommended_action": "exclude", "usable": "no",
                   "exclude_reason": "cell_line", "reason": "cell line",
                   "files": [], "gsm_includes": []}
        out = apply_verdict(DS, verdict)
        self.assertEqual(out["usable"], 0)

    def test_sample_count_corrected_when_drift_large(self):
        verdict = {"outcome": "download", "recommended_action": "keep", "usable": "yes",
                   "sample_count_in_paper": 150, "files": [], "gsm_includes": []}
        out = apply_verdict(dict(DS), verdict)
        self.assertEqual(out["sample_count"], 150)
        self.assertIn("sample_count GEO=100 paper=150", out["notes"])

    def test_sample_count_kept_when_drift_small(self):
        verdict = {"outcome": "download", "recommended_action": "keep", "usable": "yes",
                   "sample_count_in_paper": 105, "files": [], "gsm_includes": []}
        out = apply_verdict(dict(DS), verdict)
        self.assertEqual(out["sample_count"], 100)  # unchanged (5% drift)

    def test_notes_appended_not_overwritten(self):
        ds = dict(DS)
        ds["notes"] = "no_pubmed_link"
        verdict = {"outcome": "manual_review", "recommended_action": "manual_review",
                   "usable": "unclear", "notes": "conflicting sample type",
                   "files": [], "gsm_includes": []}
        out = apply_verdict(ds, verdict)
        self.assertIn("no_pubmed_link", out["notes"])
        self.assertIn("conflicting sample type", out["notes"])


class TestSplitByOutcome(unittest.TestCase):
    def test_split_into_four_lists(self):
        records = [
            {"accession": "A", "outcome": "download"},
            {"accession": "B", "outcome": "lead"},
            {"accession": "C", "outcome": "exclude"},
            {"accession": "D", "outcome": "manual_review"},
            {"accession": "E", "outcome": "download"},
            {"accession": "F"},  # missing outcome → manual_review
        ]
        buckets = split_by_outcome(records)
        self.assertEqual([r["accession"] for r in buckets["download_list"]], ["A", "E"])
        self.assertEqual([r["accession"] for r in buckets["lead_list"]], ["B"])
        self.assertEqual([r["accession"] for r in buckets["exclude_list"]], ["C"])
        self.assertEqual([r["accession"] for r in buckets["manual_review_list"]], ["D", "F"])


if __name__ == "__main__":
    unittest.main()
