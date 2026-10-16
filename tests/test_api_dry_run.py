"""
Tests for POST /search/dry-run (recall preview: parse + geo-search only).

The structured-intent path is fully deterministic (no LLM, no network): the
SearchSkill singleton is stubbed with canned candidates, so these tests pin
request validation, intent normalization passthrough, candidate mapping and
truncation without touching NCBI or ZhipuAI. The free-text path (LLM parse)
is covered by the live smoke test against the running webui container.
"""
import unittest

from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app

client = TestClient(app)


def _fake_candidates(n: int = 3):
    return [
        {
            "accession": f"GSE{i}",
            "title": f"series {i}",
            "summary": "s" * 100,
            "sample_titles": [f"sample {j}" for j in range(3)],
            "pubmed_ids": ["12345"] if i == 0 else [],
            "platforms": ["Illumina HumanMethylation450"],
            "platform_canonical": "450K",
            "sample_count": 100 + i,
            "data_type": "methylation",
            "year": 2018 + i,
            "source": "GEO",
        }
        for i in range(n)
    ]


class FakeSearchSkill:
    def __init__(self, candidates):
        self.candidates = candidates
        self.seen_states = []

    def run(self, state):
        self.seen_states.append(state)
        return {
            "candidate_gse_list": self.candidates,
            "search_queries": ["(colorectal[...]) AND (plasma[...])"],
        }


class TestDryRun(unittest.TestCase):
    def setUp(self):
        self.fake = FakeSearchSkill(_fake_candidates())
        # Stub the lazy singleton before the first dry-run request builds it.
        self._orig_skill = api_main._search_skill
        api_main._search_skill = self.fake

    def tearDown(self):
        api_main._search_skill = self._orig_skill

    def test_neither_query_nor_intent_rejected(self):
        r = client.post("/search/dry-run", json={})
        self.assertEqual(r.status_code, 422)
        self.assertIn("query", r.json()["detail"])

    def test_structured_intent_no_llm(self):
        """intent passthrough: deterministic parse, intent normalized, no LLM."""
        r = client.post("/search/dry-run", json={
            "intent": {
                "mode": "semantic",
                "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
                "sample_types": ["cfdna", "plasma"],
                "platform": "450K",
            },
            "max_candidates": 2,
        })
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        # Deterministic path — must NOT be llm / rules_fallback.
        self.assertEqual(out["parse_method"], "structured")
        self.assertEqual(out["intent"]["cancer_type"]["tcga_code"], "COAD")
        self.assertEqual(out["intent"]["platform"], "450K")
        # Search skill got exactly the normalized intent.
        self.assertEqual(len(self.fake.seen_states), 1)
        self.assertEqual(self.fake.seen_states[0]["parsed_intent"], out["intent"])
        # Truncation: 3 candidates exist, 2 requested.
        self.assertEqual(out["total_candidates"], 3)
        self.assertEqual(out["returned"], 2)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["candidates"][0]["accession"], "GSE0")
        self.assertEqual(out["candidates"][0]["platform"], "450K")
        self.assertEqual(out["candidates"][0]["pubmed_ids"], ["12345"])

    def test_no_truncation_flag_when_all_fit(self):
        r = client.post("/search/dry-run", json={
            "intent": {"mode": "semantic", "cancer_type": {"display": "LUAD"}},
            "max_candidates": 10,
        })
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual(out["returned"], 3)
        self.assertFalse(out["truncated"])

    def test_search_queries_echoed(self):
        r = client.post("/search/dry-run", json={
            "intent": {"mode": "semantic"},
        })
        out = r.json()
        self.assertEqual(len(out["search_queries"]), 1)
        self.assertIn("colorectal", out["search_queries"][0])


if __name__ == "__main__":
    unittest.main()
