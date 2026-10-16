"""
Tests for _parse_query_intent — the parse priority chain of agent1:

    intent_json (structured submission) → LLM parse → rules fallback

The structured path must be deterministic and must NEVER touch the LLM.
The LLM path is exercised via a fake chat model; failure falls back to rules.
"""
import json
import unittest

from agents.agent1_pipeline import _parse_query_intent


class _FakeLLM:
    """Minimal LangChain-ish chat model: returns canned JSON content."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        content = json.dumps(self.payload)

        class _R:
            pass

        r = _R()
        r.content = content
        return r


class _BoomLLM:
    def invoke(self, messages):
        raise RuntimeError("api down")


class TestParseQueryIntent(unittest.TestCase):
    _STRUCTURED = json.dumps({
        "mode": "semantic",
        "parse_method": "structured",
        "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
        "sample_types": ["cfdna", "plasma"],
        "platform": "450K",
    })

    def test_structured_no_llm_call(self):
        llm = _FakeLLM({"mode": "semantic"})
        intent = _parse_query_intent("structured: COAD cfDNA", llm, self._STRUCTURED)
        self.assertEqual(llm.calls, 0)  # deterministic path — LLM untouched
        self.assertEqual(intent["parse_method"], "structured")
        self.assertEqual(intent["cancer_type"]["tcga_code"], "COAD")
        self.assertEqual(intent["platform"], "450K")

    def test_structured_missing_parse_method_gets_it(self):
        payload = json.loads(self._STRUCTURED)
        payload.pop("parse_method")
        intent = _parse_query_intent("q", _FakeLLM({}), json.dumps(payload))
        self.assertEqual(intent["parse_method"], "structured")

    def test_llm_path_used_without_intent_json(self):
        llm = _FakeLLM({
            "mode": "semantic",
            "cancer_type": {"display": "LUAD", "tcga_code": "LUAD"},
        })
        intent = _parse_query_intent("lung cancer methylation", llm)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(intent["cancer_type"]["tcga_code"], "LUAD")
        self.assertEqual(intent["parse_method"], "llm")

    def test_llm_failure_falls_back_to_rules(self):
        intent = _parse_query_intent("GSE124600 cfDNA methylation", _BoomLLM())
        self.assertEqual(intent["parse_method"], "rules_fallback")
        self.assertEqual(intent["mode"], "accession")
        self.assertEqual(intent["accessions"]["geo"], ["GSE124600"])

    def test_intent_json_roundtrip_is_idempotent(self):
        """normalize twice (API stores normalized; parse_node re-normalizes)."""
        once = _parse_query_intent("q", None, self._STRUCTURED)
        twice = _parse_query_intent("q", None, json.dumps(once))
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
