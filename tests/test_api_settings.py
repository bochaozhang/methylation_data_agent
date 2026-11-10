"""Tests for GET/PUT /settings/llm-file-selection (runtime selection toggle).

Isolation: swap api.main's registry singleton per test (setUp/tearDown).
Deliberately NOT the module-import-time reload dance from
test_api_query_submit — discover imports ALL modules before running ANY
tests, so two modules reloading api.main clobber each other's _registry
binding (observed: query-submit dedup tests hit this module's temp DB).
"""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main  # noqa: E402

client = TestClient(api_main.app)


class TestFileSelectionSetting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "reg.db")
        self._orig_registry = getattr(api_main, "_registry", None)
        api_main._registry = api_main.Registry(db_path=self.db)

    def tearDown(self):
        api_main._registry = self._orig_registry
        self.tmp.cleanup()

    def test_get_defaults_to_config_source(self):
        r = client.get("/settings/llm-file-selection")
        self.assertEqual(r.status_code, 200)
        # settings.yaml ships llm_file_selection: false
        self.assertEqual(r.json(), {"enabled": False, "source": "config"})

    def test_put_then_get_roundtrip(self):
        r = client.put("/settings/llm-file-selection", json={"enabled": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"enabled": True, "source": "registry"})
        r2 = client.get("/settings/llm-file-selection")
        self.assertEqual(r2.json(), {"enabled": True, "source": "registry"})

        r3 = client.put("/settings/llm-file-selection", json={"enabled": False})
        self.assertEqual(r3.json(), {"enabled": False, "source": "registry"})
        r4 = client.get("/settings/llm-file-selection")
        self.assertEqual(r4.json(), {"enabled": False, "source": "registry"})

    def test_put_rejects_missing_field(self):
        r = client.put("/settings/llm-file-selection", json={})
        self.assertEqual(r.status_code, 422)
