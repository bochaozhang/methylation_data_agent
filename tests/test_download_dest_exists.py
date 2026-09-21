"""
Tests for the dest-exists skip in DownloadEngine._download_one.

A file already on disk (dataset re-queued by a repeat query or a manual reset)
must not be re-fetched — the engine returns the cached success result with
already_present=True (docs/caching_audit.md §3, fix #2). No network: the
attempt path is patched, so the tests assert it is (not) reached.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.download_tools import DownloadEngine


def _task(tmp, filename="file.txt.gz"):
    return {
        "accession": "GSE_TEST",
        "url": f"https://example.gov/{filename}",
        "filename": filename,
        "subdir": "GSE_TEST",
    }


class TestDestExistsSkip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_dl_dest_")
        self.engine = DownloadEngine(output_dir=self.tmp, max_concurrent=1)

    def _dest(self, filename="file.txt.gz"):
        return Path(self.tmp) / "GSE_TEST" / filename

    def test_existing_file_skips_fetch(self):
        dest = self._dest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"already here")

        with patch.object(self.engine, "_attempt_download") as attempt:
            result = asyncio.run(self.engine._download_one(None, _task(self.tmp)))

        attempt.assert_not_called()
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["local_path"], str(dest))
        self.assertEqual(result["file_size_bytes"], dest.stat().st_size)
        self.assertTrue(result["already_present"])

    def test_zero_byte_file_is_refetched(self):
        """A 0-byte destination means a previous attempt died — re-fetch."""
        dest = self._dest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"")

        sentinel = {"accession": "GSE_TEST", "status": "done",
                    "local_path": None, "error": None}
        with patch.object(self.engine, "_attempt_download", return_value=sentinel) as attempt:
            result = asyncio.run(self.engine._download_one(None, _task(self.tmp)))

        attempt.assert_called_once()
        self.assertEqual(result, sentinel)

    def test_missing_file_fetches(self):
        sentinel = {"accession": "GSE_TEST", "status": "done",
                    "local_path": None, "error": None}
        with patch.object(self.engine, "_attempt_download", return_value=sentinel) as attempt:
            result = asyncio.run(self.engine._download_one(None, _task(self.tmp)))

        attempt.assert_called_once()
        self.assertEqual(result, sentinel)

    def test_part_file_does_not_skip(self):
        """Only the final destination short-circuits; a .part means resume."""
        dest = self._dest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.with_suffix(dest.suffix + ".part").write_bytes(b"partial")

        sentinel = {"accession": "GSE_TEST", "status": "done",
                    "local_path": None, "error": None}
        with patch.object(self.engine, "_attempt_download", return_value=sentinel) as attempt:
            result = asyncio.run(self.engine._download_one(None, _task(self.tmp)))

        attempt.assert_called_once()
        self.assertEqual(result, sentinel)


if __name__ == "__main__":
    unittest.main()
