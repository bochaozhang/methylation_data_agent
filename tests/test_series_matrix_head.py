"""
Tests for the streaming header-only series_matrix fetch.

Incident 2026-10-19 (GSE51032): fetch_series_matrix_sample_info used to
materialize the WHOLE series_matrix (gzip.decompress(resp.content) +
.decode + splitlines) — array matrices reach 3+ GB compressed / 10+ GB raw
and OOM-killed the worker at 13.4 GB RSS. The fix streams, decompresses
incrementally, and stops at !series_matrix_table_begin (metadata precedes
the matrix) with a decompressed-size cap as backstop.

These tests use a fake session: no network, byte-exact accounting.
"""
import gzip
import unittest
from unittest.mock import MagicMock

from tools.geo_tools import GEOClient


def _make_matrix_text(n_samples=3, table_lines=50):
    header_lines = [
        "!Series_title\t\"test series\"",
        "!Series_platform_id\t\"GPL13534\"",
        "!Sample_geo_accession\t" + "\t".join(f'"GSM{i}"' for i in range(n_samples)),
        '!Sample_title\t' + "\t".join(f'"sample {i}"' for i in range(n_samples)),
        '!Sample_source_name_ch1\t' + "\t".join(
            f'"{"tumor" if i % 2 else "adjacent"}"' for i in range(n_samples)),
        '!Sample_molecule_ch1\t' + "\t".join('"genomic DNA"' for _ in range(n_samples)),
        '!Sample_characteristics_ch1\t' + "\t".join(
            f'"disease state: {"cancer" if i % 2 else "normal"}"' for i in range(n_samples)),
    ]
    table = ["!series_matrix_table_begin", '"ID_REF"\t"value1"\t"value2"']
    table += [f'"cg000000{i}"\t0.1\t0.2' for i in range(table_lines)]
    table.append("!series_matrix_table_end")
    return ("\n".join(header_lines) + "\n" + "\n".join(table) + "\n").encode()


class _FakeResponse:
    def __init__(self, compressed: bytes, chunk_size: int = 65536):
        self._chunks = [compressed[i:i + chunk_size]
                        for i in range(0, len(compressed), chunk_size)]
        self.consumed_chunks = 0

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        for c in self._chunks:
            self.consumed_chunks += 1
            yield c

    def close(self):
        pass


class TestStreamingSeriesMatrix(unittest.TestCase):
    def setUp(self):
        self.client = GEOClient(api_key=None, proxy=None)
        self.client._rate_limit_delay = 0.0

    def _patch_session(self, compressed: bytes):
        fake_resp = _FakeResponse(compressed)
        self.client.session = MagicMock()
        head = MagicMock()
        head.status_code = 200
        self.client.session.head.return_value = head
        self.client.session.get.return_value = fake_resp
        return fake_resp

    def test_metadata_extracted_and_stream_stopped_at_marker(self):
        matrix = _make_matrix_text(n_samples=3, table_lines=50000)  # >1 chunk
        compressed = gzip.compress(matrix)
        fake = self._patch_session(compressed)

        samples = self.client.fetch_series_matrix_sample_info("GSETEST")

        # All 3 samples parsed with the right fields.
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["gsm"], "GSM0")
        self.assertEqual(samples[1]["source_name"], "tumor")
        self.assertEqual(samples[0]["characteristics"].get("disease state"), "normal")

        # The stream stopped BEFORE consuming everything: the marker sits in
        # the first chunk of the table, so later chunks were never pulled.
        self.assertLess(fake.consumed_chunks, len(fake._chunks))

    def test_cap_without_marker_returns_none(self):
        # File with metadata but NO table marker → cap must abort, not hang.
        no_marker = (_make_matrix_text(3, table_lines=0)
                     .split(b"!series_matrix_table_begin")[0] * 200)
        self._patch_session(gzip.compress(no_marker))
        self.client.MAX_SERIES_MATRIX_HEADER_BYTES = 1024  # tiny for the test

        out = self.client.fetch_series_matrix_sample_info("GSETEST")
        self.assertIsNone(out)

    def test_head_404_returns_none(self):
        self.client.session = MagicMock()
        head = MagicMock()
        head.status_code = 404
        self.client.session.head.return_value = head
        self.assertIsNone(self.client.fetch_series_matrix_sample_info("GSETEST"))
        self.client.session.get.assert_not_called()

    def test_marker_split_across_chunks_detected(self):
        """The table marker straddling a chunk boundary must still stop the scan."""
        matrix = _make_matrix_text(n_samples=2, table_lines=2000)
        compressed = gzip.compress(matrix)
        # Force tiny chunks so the marker almost certainly splits.
        fake = self._patch_session(compressed)
        fake._chunks = [compressed[i:i + 7] for i in range(0, len(compressed), 7)]

        samples = self.client.fetch_series_matrix_sample_info("GSETEST")
        self.assertEqual(len(samples), 2)
        self.assertLess(fake.consumed_chunks, len(fake._chunks))


if __name__ == "__main__":
    unittest.main()
