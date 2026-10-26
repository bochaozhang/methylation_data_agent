"""
Tests for the geo_filter truncation retry (backlog §6-7).

2026-10-22 finding: 3/17 review-queue items had reason "filter_error:
Could not parse response content as the length limit was reached" — the
endpoint's total (prompt+completion) budget truncated the output, the
JSON parse failed, and the verdict silently degraded to a bogus
manual_review. Fix: on truncation, retry ONCE with a compact prompt
(fewer GSMs / trimmed evidence / brevity directive); persisting
truncation degrades to manual_review with a DISTINGUISHED reason
(filter_error_truncated, not filter_error).
"""
import json
import unittest

from skills.geo_filter.skill import filter_dataset

_TRUNC_ERR = RuntimeError(
    "Could not parse response content as the length limit was reached - "
    "CompletionUsage(completion_tokens=1024, prompt_tokens=10285)")

_VALID = {
    "reasoning": "human CRC 450K tumor/adjacent; matches request",
    "outcome": "download",
    "confirmed_sample_type": "tumor",
    "confirmed_cancer_type": "colorectal cancer",
    "technology": "450K",
    "platform": "GPL13534",
    "sample_size": "22 tumor, 22 adjacent",
    "stage_or_treatment_status": None,
    "disease_groups": "tumor 22 / adjacent 22",
    "sample_level_annotation": "yes",
    "annotation_source": "GSM_characteristics",
    "download_tier": 1,
    "exclude_reason": None,
    "flags": "",
    "sample_count_in_paper": 44,
    "consistency": "consistent",
    "reason": "paired tumor/adjacent 450K",
    "notes": "",
    "gsm_includes": [],
}

_DS = {
    "accession": "GSE42752",
    "title": "Colon adenocarcinoma paired 450K",
    "summary": "x" * 800,
    "overall_design": "paired",
    "platform_canonical": "450K",
    "sample_count": 44,
    "sample_type": "tumor",
    "cancer_type": "colorectal cancer",
    "pubmed_ids": ["12345"],
}
_INTENT = {"mode": "semantic", "cancer_type": {"display": "colorectal cancer"},
           "sample_types": ["tumor", "adjacent"]}
_GSM = [{"gsm": f"GSM{i}", "source_name": f"tumor block {i}", "molecule": "genomic DNA",
         "characteristics": {"disease state": "cancer"}, "group": "tumor"}
        for i in range(20)]


class _Resp:
    def __init__(self, payload):
        self.content = json.dumps(payload)


class _ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return _Resp(step)


class TestTruncationRetry(unittest.TestCase):
    def test_retry_recovers_verdict(self):
        llm = _ScriptedLLM([_TRUNC_ERR, dict(_VALID)])
        v = filter_dataset(llm, _DS, _INTENT, _GSM, abstract="abstract " * 200)
        self.assertEqual(v["outcome"], "download")
        self.assertTrue(v.get("_truncation_retried"))
        self.assertEqual(len(llm.calls), 2)
        # Retry prompt is compact + carries the directive.
        retry_msg = llm.calls[1][1].content
        self.assertIn("CONCISE RETRY", retry_msg)
        self.assertLess(len(retry_msg), len(llm.calls[0][1].content))
        self.assertIn("GSM7", retry_msg)   # 8-GSM subset
        self.assertNotIn("GSM8", retry_msg)

    def test_persistent_truncation_distinguished(self):
        llm = _ScriptedLLM([_TRUNC_ERR, _TRUNC_ERR])
        v = filter_dataset(llm, _DS, _INTENT, _GSM)
        self.assertEqual(v["outcome"], "manual_review")
        self.assertTrue(v["reason"].startswith("filter_error_truncated"))
        self.assertEqual(len(llm.calls), 2)

    def test_non_truncation_error_single_call(self):
        llm = _ScriptedLLM([RuntimeError("api down")])
        v = filter_dataset(llm, _DS, _INTENT, _GSM)
        self.assertEqual(v["outcome"], "manual_review")
        self.assertTrue(v["reason"].startswith("filter_error:"))
        self.assertEqual(len(llm.calls), 1)

    def test_normal_path_unaffected(self):
        llm = _ScriptedLLM([dict(_VALID)])
        v = filter_dataset(llm, _DS, _INTENT, _GSM, abstract="abs")
        self.assertEqual(v["outcome"], "download")
        self.assertFalse(v.get("_truncation_retried"))
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(v["_evidence"]["had_abstract"])


if __name__ == "__main__":
    unittest.main()
