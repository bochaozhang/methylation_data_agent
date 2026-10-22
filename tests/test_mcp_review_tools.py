"""
Tests for the review-queue MCP tools (approve/reject downloads).

Isolation note: unlike test_api_query_submit (which must bind its temp
registry at module-import time because its client is module-level), this
module binds in setUpClass/tearDownClass — rebinding the shared api.main at
IMPORT time poisons modules imported earlier during unittest's all-at-once
discovery (found live 2026-10-22: it broke test_api_query_submit's dedup
tests). Alphabetical order guarantees query_submit's tests (and its
tearDownModule restore) complete before this class sets up.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import api.main as api_main


def _seed_needs_review(accession="GSE199999"):
    reg = api_main.get_registry()
    reg.upsert_dataset(
        accession=accession, source="GEO", discovered_by="test",
        data_type="array", download_status="awaiting_approval",
        needs_review=True, title="test review candidate",
    )
    return accession


class TestReviewRoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = str(Path(cls.tmp.name) / "reg.db")
        cls._orig = (api_main.REGISTRY_PATH, api_main._registry)
        api_main.REGISTRY_PATH = cls.db
        api_main._registry = api_main.Registry(db_path=cls.db)
        cls.client = TestClient(api_main.app)

    @classmethod
    def tearDownClass(cls):
        api_main.REGISTRY_PATH, api_main._registry = cls._orig
        cls.tmp.cleanup()

    def test_approve_queues_download(self):
        acc = _seed_needs_review()
        r = self.client.post(f"/review/{acc}/approve", json={"note": "ok via test"})
        self.assertEqual(r.status_code, 200, r.text)
        d = api_main.get_registry().get(acc)
        self.assertFalse(d["needs_review"])
        self.assertEqual(d["download_status"], "pending")

    def test_reject_skips(self):
        acc = _seed_needs_review("GSE188888")
        r = self.client.post(f"/review/{acc}/reject", json={"note": "pooled"})
        self.assertEqual(r.status_code, 200, r.text)
        d = api_main.get_registry().get(acc)
        self.assertEqual(d["download_status"], "skipped")

    def test_not_pending_review_rejected_400(self):
        acc = _seed_needs_review("GSE177777")
        self.client.post(f"/review/{acc}/approve")
        r = self.client.post(f"/review/{acc}/approve")   # second time
        self.assertEqual(r.status_code, 400)

    def test_unknown_accession_404(self):
        self.assertEqual(
            self.client.post("/review/GSE000001/reject").status_code, 404)


class TestMcpGateAndTools(unittest.TestCase):
    def test_gate_blocks_when_disabled(self):
        from mcp_server import server as m
        os.environ["MCP_ALLOW_WRITES"] = "0"
        try:
            with self.assertRaises(PermissionError):
                m.approve_dataset("GSE199999")
            with self.assertRaises(PermissionError):
                m.reject_dataset("GSE199999")
        finally:
            os.environ.pop("MCP_ALLOW_WRITES", None)

    def test_tools_proxy_to_api(self):
        from mcp_server import server as m
        fake_api = MagicMock()
        fake_api.post.return_value = {"accession": "GSE166666", "action": "approved"}
        orig_client = m._client
        m._client = fake_api
        try:
            out = m.approve_dataset("GSE166666", note="via mcp")
            self.assertEqual(out["action"], "approved")
            fake_api.post.assert_called_once_with(
                "/review/GSE166666/approve", json={"note": "via mcp"})
            m.reject_dataset("GSE166666")
            fake_api.post.assert_called_with("/review/GSE166666/reject", json={})
        finally:
            m._client = orig_client


if __name__ == "__main__":
    unittest.main()
