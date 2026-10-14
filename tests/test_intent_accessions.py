"""
Tests for the removal of accessions.tcga from the query-parser intent JSON.

accessions.tcga had no consumer in the active pipeline (SearchSkill reads only
.geo; tcga-direct keys off cancer_type.tcga_code) and its stochastic emission
by the LLM caused confusion. These tests pin: the rules parser and
normalize_intent never emit it, inbound tcga (old prompts / saved intents) is
tolerated and dropped, and geo accessions pass through untouched.
extract_accessions() itself still extracts tcga — the literature agent
consumes that shared utility.
"""
import unittest

from tools.parser_tools import extract_accessions, parse_query_rules
from state.agent1_state import normalize_intent


class TestNoTcgaInIntent(unittest.TestCase):
    def test_rules_parser_omits_tcga(self):
        """A query naming TCGA-COAD must not put it into intent.accessions."""
        intent = parse_query_rules("下载 TCGA-COAD 的甲基化数据")
        self.assertNotIn("tcga", intent["accessions"])
        # mode detection is unchanged: an explicit TCGA accession still counts.
        self.assertEqual(intent["mode"], "accession")

    def test_rules_parser_keeps_geo(self):
        intent = parse_query_rules("GSE124600 和 GSE79277 cfDNA")
        self.assertEqual(intent["accessions"]["geo"], ["GSE124600", "GSE79277"])
        self.assertNotIn("tcga", intent["accessions"])

    def test_normalize_drops_inbound_tcga(self):
        """Old-prompt LLM output / saved intents may still carry tcga — dropped."""
        parsed = {
            "mode": "semantic",
            "accessions": {"geo": ["GSE124600"], "tcga": ["TCGA-COAD"]},
            "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
        }
        intent = normalize_intent("query", parsed)
        self.assertEqual(intent["accessions"], {"geo": ["GSE124600"]})

    def test_normalize_geo_passthrough(self):
        parsed = {"mode": "accession", "accessions": {"geo": ["GSE11173"]}}
        intent = normalize_intent("query", parsed)
        self.assertEqual(intent["accessions"], {"geo": ["GSE11173"]})

    def test_extract_accessions_still_extracts_tcga(self):
        """The shared literature-agent utility is NOT affected by this change."""
        out = extract_accessions("data from TCGA-READ and GSE86078")
        self.assertEqual(out["tcga"], ["TCGA-READ"])
        self.assertEqual(out["geo"], ["GSE86078"])


if __name__ == "__main__":
    unittest.main()
