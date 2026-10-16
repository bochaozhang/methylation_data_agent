"""
Tests for mcp_server.paths — container→host path translation.

The registry stores container-side paths (/app/data/..., written by the
in-container download worker). MCP responses must translate them to the
host side (compose mount ./data:/app/data) so consumers can open the files.
Unknown/foreign prefixes must pass through untouched.
"""
import os
import unittest
from pathlib import Path

from mcp_server.paths import (
    REPO_ROOT,
    host_data_root,
    translate_path,
    translate_record,
)


class TestTranslatePath(unittest.TestCase):
    def setUp(self):
        # Isolate from the developer's environment in every test.
        self._orig = os.environ.pop("MCP_DATA_HOST_PATH", None)

    def tearDown(self):
        if self._orig is not None:
            os.environ["MCP_DATA_HOST_PATH"] = self._orig
        else:
            os.environ.pop("MCP_DATA_HOST_PATH", None)

    def test_container_prefix_translated(self):
        out = translate_path("/app/data/GSE124600/filelist.txt")
        self.assertEqual(
            out,
            os.path.join(str(REPO_ROOT / "data"), "GSE124600/filelist.txt"),
        )

    def test_env_override_root(self):
        os.environ["MCP_DATA_HOST_PATH"] = "/srv/methy"
        self.assertEqual(host_data_root(), "/srv/methy")
        out = translate_path("/app/data/GSE1/a.csv")
        self.assertEqual(out, os.path.join("/srv/methy", "GSE1/a.csv"))

    def test_default_root_is_repo_data(self):
        self.assertEqual(host_data_root(), str(REPO_ROOT / "data"))

    def test_unknown_prefix_passthrough(self):
        for p in (
            "/home/ubuntu/elsewhere/x.txt",
            "relative/path.txt",
            "https://example.edu/geofile.gz",
        ):
            self.assertEqual(translate_path(p), p)

    def test_empty_and_none_passthrough(self):
        self.assertIsNone(translate_path(None))
        self.assertEqual(translate_path(""), "")


class TestTranslateRecord(unittest.TestCase):
    def test_path_fields_translated(self):
        rec = {
            "accession": "GSE124600",
            "local_path": "/app/data/GSE124600/filelist.txt",
            "sample_metadata_path": "/app/data/GSE124600/sample_metadata.csv",
            "title": "untouched",
        }
        out = translate_record(rec)
        self.assertTrue(out["local_path"].endswith("data/GSE124600/filelist.txt"))
        self.assertNotIn("/app/data", out["local_path"])
        self.assertTrue(out["sample_metadata_path"].endswith(
            "GSE124600/sample_metadata.csv"))
        self.assertEqual(out["title"], "untouched")

    def test_record_without_paths_untouched(self):
        rec = {"accession": "GSE1", "local_path": None, "title": "t"}
        self.assertEqual(translate_record(rec), rec)


if __name__ == "__main__":
    unittest.main()
