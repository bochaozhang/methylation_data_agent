"""
Tests for submitter-coded sample-title decoding (backlog §6-8).

GSE97923 hit the known gap at scale: disease state only in titles
("COAD.ct-N"), characteristics nearly empty → 773 samples cancer=unclear.
Covers: title persistence in the efetch path, cache backfill, the decode
core (one scripted LLM call → pattern rule → per-GSM classes), and the
apply path (registry + CSV + disease_groups).
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from registry.registry import Registry
from tools.title_decode import (
    apply_decoded, decode_titles, load_titles, merge_titles_into_cache,
    _normalise_title,
)


def _samples():
    return [
        {"gsm": f"GSM1{i:03d}", "title": f"COAD.ct-{i}", "group": "plasma_cfdna"}
        for i in range(30)
    ] + [
        {"gsm": f"GSM2{i:03d}", "title": f"Health.ht-{i}", "group": "plasma_cfdna"}
        for i in range(10)
    ]


class _Resp:
    def __init__(self, payload):
        self.content = json.dumps(payload)


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Resp(self.payload)


_LLM_PAYLOAD = {
    "confidence": "high",
    "patterns": [
        {"pattern": "COAD.ct-N", "cancer": "query_cancer", "note": "ctDNA of COAD cases"},
        {"pattern": "Health.ht-N", "cancer": "normal", "note": "healthy control ctDNA"},
    ],
    "disease_groups": "cancer 30 / normal 10",
    "caveats": "",
}


class TestDecodeCore(unittest.TestCase):
    def test_pattern_rule_applied_to_all(self):
        llm = _FakeLLM(_LLM_PAYLOAD)
        out = decode_titles(llm, {"accession": "GSE97923",
                                  "title": "cfDNA common cancers",
                                  "cancer_type": "colorectal cancer"},
                            _samples())
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(out["n_classified"], 40)
        self.assertEqual(out["n_total"], 40)
        self.assertEqual(out["gsm_class"]["GSM1000"], "query_cancer")
        self.assertEqual(out["gsm_class"]["GSM2005"], "normal")

    def test_no_titles_is_error(self):
        out = decode_titles(_FakeLLM(_LLM_PAYLOAD), {"accession": "GSE1"},
                            [{"gsm": "GSM1", "title": "", "group": ""}])
        self.assertIn("error", out)

    def test_invalid_class_dropped(self):
        payload = {"confidence": "low",
                   "patterns": [{"pattern": "COAD.ct-N", "cancer": "maybe", "note": ""}],
                   "disease_groups": "", "caveats": "ambiguous"}
        out = decode_titles(_FakeLLM(payload), {}, _samples()[:3])
        self.assertEqual(out["n_classified"], 0)


class TestCacheBackfill(unittest.TestCase):
    def test_merge_and_load(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            acc_dir = Path(tmp.name) / "GSE1"
            acc_dir.mkdir()
            entries = [{"gsm": "GSM1", "source_name": "serum",
                        "molecule": "DNA", "characteristics": {}, "group": "x"}]
            cache = acc_dir / "gsm_metadata_cache.json"
            cache.write_text(json.dumps(entries), encoding="utf-8")

            self.assertEqual(load_titles("GSE1", tmp.name), [])  # no titles yet
            added = merge_titles_into_cache("GSE1", tmp.name, {"GSM1": "COAD.ct-1"})
            self.assertEqual(added, 1)
            got = load_titles("GSE1", tmp.name)
            self.assertEqual(got, [{"gsm": "GSM1", "title": "COAD.ct-1", "group": "x"}])
            # Second merge does not duplicate.
            self.assertEqual(merge_titles_into_cache("GSE1", tmp.name, {"GSM1": "other"}), 0)
        finally:
            tmp.cleanup()


class TestApplyPath(unittest.TestCase):
    def test_registry_and_csv_updated(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            reg = Registry(db_path=str(Path(tmp.name) / "reg.db"))
            reg.upsert_samples("GSE1", [
                {"gsm": "GSM1", "source_name": "s", "molecule": "DNA",
                 "characteristics": {}, "group": "x"},
                {"gsm": "GSM2", "source_name": "s", "molecule": "DNA",
                 "characteristics": {}, "group": "x"},
            ], cancer_map={})
            acc_dir = Path(tmp.name) / "GSE1"
            acc_dir.mkdir()
            with open(acc_dir / "sample_metadata.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["gsm", "source_name", "molecule", "group", "cancer"])
                w.writerow(["GSM1", "s", "DNA", "x", "unclear"])
                w.writerow(["GSM2", "s", "DNA", "x", "unclear"])

            counts = apply_decoded(reg, "GSE1", tmp.name,
                                   {"GSM1": "query_cancer", "GSM2": "normal"})
            self.assertEqual(counts["registry"], 2)
            self.assertEqual(counts["csv"], 2)
            with reg._get_conn() as conn:
                cancers = dict(conn.execute(
                    "SELECT gsm, cancer FROM samples WHERE accession='GSE1'"))
            self.assertEqual(cancers, {"GSM1": "query_cancer", "GSM2": "normal"})
            rows = list(csv.DictReader(open(acc_dir / "sample_metadata.csv")))
            self.assertEqual([r["cancer"] for r in rows],
                             ["query_cancer", "normal"])
        finally:
            tmp.cleanup()


class TestNormalise(unittest.TestCase):
    def test_digits_normalised(self):
        self.assertEqual(_normalise_title("COAD.ct-511"), "COAD.ct-N")
        self.assertEqual(_normalise_title("  Health 12 "), "Health N")


if __name__ == "__main__":
    unittest.main()
