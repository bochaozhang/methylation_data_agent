"""
Tests for search-failure semantics (backlog §6-6).

2026-10-19 incident: a dead proxy made esearch fail; the exception was
swallowed all the way down and the task "succeeded" with 0 candidates.
Now: esearch failure raises GEOSearchError → SearchSkill propagates →
run_database_agent re-raises → daemon fail_task. A SUCCESSFUL esearch
matching nothing remains a legitimate empty recall (returns []).
"""
import unittest
from unittest.mock import MagicMock

from skills.geo_search.skill import SearchSkill
from tools.geo_tools import GEOSearchError, GEOClient


def _intent():
    return {"mode": "semantic", "cancer_type": {"display": "crc"},
            "sample_types": ["cfdna"]}


class _FakeResp:
    def __init__(self, ids):
        self._ids = ids

    def json(self):
        return {"esearchresult": {"idlist": self._ids}}


class TestSearchGSEFailureSemantics(unittest.TestCase):
    def _client(self):
        return GEOClient(api_key=None)

    def test_both_attempts_fail_raises(self):
        c = self._client()
        c._get = MagicMock(side_effect=ConnectionError("proxy dead"))
        with self.assertRaises(GEOSearchError) as cm:
            c.search_gse("some query")
        self.assertIn("primary and fallback", str(cm.exception))

    def test_abuse_redirect_raises_immediately(self):
        c = self._client()
        c._get = MagicMock(
            side_effect=RuntimeError("302 to misuse.ncbi.nlm.nih.gov"))
        with self.assertRaises(GEOSearchError) as cm:
            c.search_gse("some query")
        self.assertIn("abuse", str(cm.exception))
        self.assertEqual(c._get.call_count, 1)  # no pointless retry

    def test_transient_primary_fallback_recovers(self):
        c = self._client()
        c._get = MagicMock(side_effect=[ConnectionError("blip"),
                                        _FakeResp(["2000001"])])
        c._uids_to_accessions = MagicMock(return_value=["GSE1"])
        self.assertEqual(c.search_gse("q"), ["GSE1"])

    def test_successful_empty_is_legitimate_zero(self):
        c = self._client()
        c._get = MagicMock(side_effect=[_FakeResp([]), _FakeResp([])])
        self.assertEqual(c.search_gse("matches nothing"), [])


class TestSearchSkillPropagates(unittest.TestCase):
    def test_skill_run_raises_on_search_failure(self):
        skill = SearchSkill({"geo": {}})
        skill.geo_client = MagicMock()
        skill.geo_client.search_gse.side_effect = GEOSearchError("proxy dead")
        with self.assertRaises(GEOSearchError):
            skill.run({"parsed_intent": _intent()})

    def test_skill_run_empty_recall_ok(self):
        skill = SearchSkill({"geo": {}})
        skill.geo_client = MagicMock()
        skill.geo_client.search_gse.return_value = []
        out = skill.run({"parsed_intent": _intent()})
        self.assertEqual(out["candidate_gse_list"], [])


if __name__ == "__main__":
    unittest.main()
