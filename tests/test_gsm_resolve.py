"""
Tests for skills.geo_filter.gsm_resolve — the GSM-evidence resolver.

Covers the four resolution branches (series_matrix / cache / over-cap fallback /
efetch-all) and the JSON cache round-trip + robustness. GEOClient is mocked; no
network.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from skills.geo_filter.gsm_resolve import (
    read_gsm_cache,
    resolve_gsm_details,
    write_gsm_cache,
)


def _gsm(gsm_id, disease="colorectal cancer"):
    return {
        "gsm": gsm_id,
        "source_name": "plasma",
        "molecule": "genomic DNA",
        "characteristics": {"disease state": disease},
        "group": "plasma_cfdna",
    }


def _mock_geo(sm=None, all_gsm=None, rep=None):
    """GEOClient mock with the three methods the resolver calls."""
    m = MagicMock()
    m.fetch_series_matrix_sample_info.return_value = sm
    m.get_all_gsm_metadata.return_value = all_gsm if all_gsm is not None else []
    m.get_representative_gsm_details.return_value = rep if rep is not None else []
    return m


class TestResolveBranches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_series_matrix_hit_short_circuits(self):
        """Path A: series_matrix present → returned, no cache/efetch touched."""
        sm = [_gsm("GSM1"), _gsm("GSM2", "healthy")]
        geo = _mock_geo(sm=sm, all_gsm=[_gsm("X")], rep=[_gsm("Y")])
        ds = {"sample_count": 2}

        out = resolve_gsm_details(geo, "GSE1", ds, self.output_dir)

        self.assertEqual(out, sm)
        geo.fetch_series_matrix_sample_info.assert_called_once_with("GSE1")
        geo.get_all_gsm_metadata.assert_not_called()
        geo.get_representative_gsm_details.assert_not_called()
        # Nothing cached (series_matrix is cheap; not persisted).
        self.assertFalse(
            (Path(self.output_dir) / "GSE1" / "gsm_metadata_cache.json").exists()
        )

    def test_cache_hit_skips_efetch(self):
        """Cached efetch-all result is reused; no efetch on a re-run."""
        cached = [_gsm("GSM1"), _gsm("GSM2")]
        cache_path = Path(self.output_dir) / "GSE2" / "gsm_metadata_cache.json"
        write_gsm_cache(cache_path, cached)

        geo = _mock_geo(sm=None, all_gsm=[_gsm("SHOULD_NOT")], rep=[_gsm("NEITHER")])
        ds = {"sample_count": 2}

        out = resolve_gsm_details(geo, "GSE2", ds, self.output_dir)

        self.assertEqual(out, cached)
        geo.get_all_gsm_metadata.assert_not_called()
        geo.get_representative_gsm_details.assert_not_called()

    def test_over_cap_uses_representative_fallback(self):
        """sample_count > cap → representative sampling; no efetch-all, no cache write."""
        rep = [_gsm("GSM1"), _gsm("GSM2")]
        geo = _mock_geo(sm=None, all_gsm=[_gsm("SHOULD_NOT")], rep=rep)
        ds = {"sample_count": 1000}

        out = resolve_gsm_details(
            geo, "GSE3", ds, self.output_dir, wanted_sample_type="cfdna",
            max_all_fetch=600,
        )

        self.assertEqual(out, rep)
        geo.get_representative_gsm_details.assert_called_once_with(
            "GSE3", wanted_sample_type="cfdna"
        )
        geo.get_all_gsm_metadata.assert_not_called()
        self.assertFalse(
            (Path(self.output_dir) / "GSE3" / "gsm_metadata_cache.json").exists()
        )

    def test_under_cap_efetch_all_and_caches(self):
        """sample_count ≤ cap → efetch every GSM, then persist the cache."""
        all_gsm = [_gsm("GSM1"), _gsm("GSM2"), _gsm("GSM3")]
        geo = _mock_geo(sm=None, all_gsm=all_gsm, rep=[_gsm("NOPE")])
        ds = {"sample_count": 3}

        out = resolve_gsm_details(geo, "GSE4", ds, self.output_dir, max_all_fetch=600)

        self.assertEqual(out, all_gsm)
        geo.get_all_gsm_metadata.assert_called_once_with("GSE4")
        geo.get_representative_gsm_details.assert_not_called()

        cache_path = Path(self.output_dir) / "GSE4" / "gsm_metadata_cache.json"
        self.assertTrue(cache_path.exists())
        with cache_path.open() as f:
            self.assertEqual(json.load(f), all_gsm)

    def test_missing_sample_count_skips_cap_and_efetches_all(self):
        """No sample_count → cap can't apply → efetch-all (the cap is best-effort)."""
        geo = _mock_geo(sm=None, all_gsm=[_gsm("GSM1")])
        out = resolve_gsm_details(geo, "GSE5", {}, self.output_dir, max_all_fetch=600)
        self.assertEqual(len(out), 1)
        geo.get_all_gsm_metadata.assert_called_once_with("GSE5")

    def test_series_matrix_exception_falls_through(self):
        """A network error on series_matrix must not abort — fall through to cache/efetch."""
        geo = _mock_geo(sm=None, all_gsm=[_gsm("GSM1")])
        geo.fetch_series_matrix_sample_info.side_effect = RuntimeError("timeout")
        ds = {"sample_count": 1}

        out = resolve_gsm_details(geo, "GSE6", ds, self.output_dir)

        self.assertEqual(len(out), 1)
        geo.get_all_gsm_metadata.assert_called_once_with("GSE6")


class TestCacheHelpers(unittest.TestCase):
    def test_round_trip_preserves_characteristics(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gsm_metadata_cache.json"
            payload = [_gsm("GSM1", "colorectal cancer"), _gsm("GSM2", "healthy")]
            write_gsm_cache(p, payload)
            back = read_gsm_cache(p)
            self.assertEqual(back, payload)
            # characteristics dict survives intact
            self.assertEqual(back[0]["characteristics"], {"disease state": "colorectal cancer"})

    def test_read_missing_returns_none(self):
        self.assertIsNone(read_gsm_cache(Path("/nonexistent/does-not-exist.json")))

    def test_read_corrupt_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "corrupt.json"
            p.write_text("{ not valid json ")
            self.assertIsNone(read_gsm_cache(p))

    def test_read_non_list_payload_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wrong.json"
            p.write_text(json.dumps({"not": "a list"}))
            self.assertIsNone(read_gsm_cache(p))


if __name__ == "__main__":
    unittest.main()
