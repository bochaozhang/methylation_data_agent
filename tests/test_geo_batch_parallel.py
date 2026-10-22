"""
Tests for the global rate limiter + parallel batch metadata fetch.

Motivation (2026-10-22): search_datasets (dry-run) took ~50-60s — longer
than Claude Code's default MCP tool timeout (30s). Two fixes in
GEOClient.batch_get_series_metadata / _get:
  - _throttle(): ONE shared pacing clock across threads. Per-call sleeps
    multiplied the actual rate by the thread count (the filter loop already
    ran 3 concurrent _get callers → latent NCBI rate violation).
  - batches fetched in parallel (4 workers, order-preserving executor.map).
"""
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from tools.geo_tools import GEOClient


class TestGlobalThrottle(unittest.TestCase):
    def test_aggregate_rate_is_paced_not_multiplied(self):
        """8 calls from 4 threads must take ≥ (8-1)*delay total — per-thread
        sleeps would allow ~8*delay/4 = 2*delay."""
        c = GEOClient(api_key=None)
        c._rate_limit_delay = 0.05
        c._last_req_at = 0.0  # reset so the first call isn't free-running

        def _hit(_):
            c._throttle()

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(_hit, range(8)))
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 7 * 0.05 * 0.9)  # ~0.35s floor
        self.assertLess(elapsed, 0.8)                     # but not serialized-slow

    def test_single_thread_still_paced(self):
        c = GEOClient(api_key=None)
        c._rate_limit_delay = 0.03
        c._last_req_at = 0.0
        t0 = time.monotonic()
        for _ in range(5):
            c._throttle()
        self.assertGreaterEqual(time.monotonic() - t0, 4 * 0.03 * 0.9)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.url = "https://eutils.ncbi.nlm.nih.gov/ok"

    def json(self):
        return self._payload


class TestParallelBatchOrder(unittest.TestCase):
    def test_output_order_preserved_under_out_of_order_completion(self):
        c = GEOClient(api_key=None)

        def fake_get(endpoint, params):
            if endpoint == "esearch.fcgi":
                accs = [t.split("[")[0] for t in params["term"].split(" OR ")]
                # Later batches complete FASTER (sleep inverted) to prove the
                # output order follows submission, not completion.
                batch_idx = int(accs[0].replace("GSE", "")) // 50
                time.sleep(max(0, 0.25 - batch_idx * 0.08))
                uids = [str(1000 + int(a.replace("GSE", ""))) for a in accs]
                return _FakeResp({"esearchresult": {"idlist": uids}})
            if endpoint == "esummary.fcgi":
                result = {"uids": []}
                for uid in params["id"].split(","):
                    n = int(uid) - 1000
                    result["uids"].append(uid)
                    result[uid] = {
                        "accession": f"GSE{n}", "title": f"series {n}",
                        "summary": "methylation", "gpl": "GPL13534",
                        "samples": [], "pubmedids": [], "pdat": "2020/01",
                        "n_samples": 5,
                    }
                return _FakeResp({"result": result})
            raise AssertionError(endpoint)

        accessions = [f"GSE{i}" for i in range(1, 151)]  # 3 batches of 50
        with patch.object(c, "_get", side_effect=fake_get), \
                patch.object(c, "_get_supplementary_urls", return_value=[]):
            out = c.batch_get_series_metadata(accessions, batch_size=50)

        got = [m["accession"] for m in out]
        self.assertEqual(got, [f"GSE{i}" for i in range(1, 151)])
        self.assertTrue(all(m.get("title") for m in out))


if __name__ == "__main__":
    unittest.main()
