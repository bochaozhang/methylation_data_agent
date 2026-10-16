"""
Tests for POST /query Phase-2 semantics: structured intent, label, dedup.

api.main reads REGISTRY_PATH at import time. The isolation MUST happen at
module-import time too — unittest calls setUpModule only AFTER the module
body has run, so putting the env swap there silently binds the app to the
PRODUCTION registry (a real incident on 2026-10-16: leaked test tasks were
claimed by the live daemon). Everything therefore runs at module level here;
tearDownModule restores the original binding for later test modules.
"""
import importlib
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

_ORIG_REGISTRY_PATH = os.environ.get("REGISTRY_PATH")

# --- module-import-time isolation (see docstring) -------------------------
_TMP = tempfile.TemporaryDirectory()
_TMP_DB = str(Path(_TMP.name) / "reg.db")
os.environ["REGISTRY_PATH"] = _TMP_DB

import api.main as api_main  # noqa: E402

api_main = importlib.reload(api_main)   # re-read REGISTRY_PATH → temp DB
api_main.REGISTRY_PATH = _TMP_DB        # belt & suspenders
api_main._registry = api_main.Registry(db_path=_TMP_DB)
client = TestClient(api_main.app)


def tearDownModule():
    if _ORIG_REGISTRY_PATH is None:
        os.environ.pop("REGISTRY_PATH", None)
    else:
        os.environ["REGISTRY_PATH"] = _ORIG_REGISTRY_PATH
    importlib.reload(api_main)  # rebind later imports to the original path
    _TMP.cleanup()

_INTENT = {
    "mode": "semantic",
    "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
    "sample_types": ["cfdna", "plasma"],
    "platform": "450K",
}


class TestSubmitQuery(unittest.TestCase):
    def setUp(self):
        # Each test starts from an empty queue (methods run alphabetically and
        # would otherwise dedup against tasks created by earlier tests).
        import sqlite3
        conn = sqlite3.connect(_TMP_DB)
        conn.execute("DELETE FROM task_queue")
        conn.commit()
        conn.close()

    def test_neither_query_nor_intent_rejected(self):
        r = client.post("/query", json={})
        self.assertEqual(r.status_code, 422)

    def test_structured_submission(self):
        r = client.post("/query", json={
            "intent": _INTENT,
            "agent_type": "database",
            "label": "main",
        })
        self.assertEqual(r.status_code, 201, r.text)
        out = r.json()
        self.assertEqual(out["label"], "main")
        self.assertFalse(out["deduplicated"])
        self.assertTrue(out["query"].startswith("structured: colorectal cancer"))
        # intent_json persisted and normalized (parse_method=structured inside)
        task = api_main.get_registry().get_task(out["task_id"])
        self.assertIsNotNone(task["intent_json"])
        self.assertIn("COAD", task["intent_json"])

    def test_structured_dedup_same_intent(self):
        r1 = client.post("/query", json={"intent": _INTENT, "agent_type": "database"})
        self.assertEqual(r1.status_code, 201)
        # Different input key order / extra defaults — same canonical intent.
        r2 = client.post("/query", json={
            "intent": {**_INTENT, "notes": ""},
            "agent_type": "database",
        })
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.json()["deduplicated"])
        self.assertEqual(r2.json()["task_id"], r1.json()["task_id"])

    def test_different_intent_not_deduped(self):
        client.post("/query", json={"intent": _INTENT, "agent_type": "database"})
        other = {**_INTENT, "sample_types": ["tumor", "adjacent"]}
        r = client.post("/query", json={"intent": other, "agent_type": "database"})
        self.assertEqual(r.status_code, 201)
        self.assertFalse(r.json()["deduplicated"])

    def test_free_text_dedup_and_force(self):
        r1 = client.post("/query", json={"query": "LUAD cfDNA methylation"})
        self.assertEqual(r1.status_code, 201)
        r2 = client.post("/query", json={"query": "LUAD cfDNA methylation"})
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.json()["deduplicated"])
        r3 = client.post("/query", json={
            "query": "LUAD cfDNA methylation", "force": True})
        self.assertEqual(r3.status_code, 201)
        self.assertNotEqual(r3.json()["task_id"], r1.json()["task_id"])

    def test_done_task_not_dedup_target(self):
        """Dedup only considers pending/running — after completion, resubmit works."""
        r1 = client.post("/query", json={"query": "breast cancer methylation EPIC"})
        tid = r1.json()["task_id"]
        api_main.get_registry().complete_task(tid, result={"done": True})
        r2 = client.post("/query", json={"query": "breast cancer methylation EPIC"})
        self.assertEqual(r2.status_code, 201)
        self.assertNotEqual(r2.json()["task_id"], tid)


if __name__ == "__main__":
    unittest.main()
