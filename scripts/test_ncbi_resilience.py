"""
Tests for the NCBI rate-limit / PMC full-text resilience fixes (Phase 3).

Reproduces the failure seen in a real orchestrator_v2 run — every PMC full-text
fetch came back as NCBI's HTML abuse-redirect page, which the JSON parser choked
on — and asserts the new handling:

  1. _is_abuse_redirect() recognises the misuse redirect (which returns HTTP 200).
  2. _ncbi_get() retries with exponential backoff, then raises NCBIRateLimitError
     instead of letting HTML reach a JSON parser.
  3. _ncbi_get_json() converts a non-JSON body into NCBIRateLimitError.
  4. get_pmc_fulltext() degrades to None rather than raising.
  5. _fetch_fulltext_safe() memoises per PMID — a repeated search cannot re-hit PMC.
  6. LiteratureClient routes through NCBI_PROXY (the bug that made PMC calls go
     out unproxied while search traffic used the tunnel).

All network calls are stubbed; no NCBI access or API key required.

Run: python3 -m scripts.test_ncbi_resilience
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools.pubmed_tools as pt
from tools.pubmed_tools import LiteratureClient, NCBIRateLimitError, _is_abuse_redirect

ABUSE_URL = "https://www.ncbi.nlm.nih.gov/corehtml/query/static/abuse.shtml"
ABUSE_HTML = "<html><body>Your access has been blocked.</body></html>"


class FakeResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, url, text='{"ok": true}', content_type="application/json", status=200):
        self.url = url
        self.text = text
        self.content = text.encode()
        self.headers = {"Content-Type": content_type}
        self.status_code = status
        self.request = None

    def json(self):
        import json as _json
        return _json.loads(self.text)  # raises ValueError on HTML

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_abuse_redirect_detection():
    assert _is_abuse_redirect(FakeResponse(ABUSE_URL, ABUSE_HTML, "text/html"))
    assert _is_abuse_redirect(FakeResponse("https://misuse.ncbi.nlm.nih.gov/error/abuse.shtml"))
    assert not _is_abuse_redirect(FakeResponse("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"))
    print("  [1] abuse-redirect detection (incl. HTTP-200 HTML page)      PASS")


def test_ncbi_get_retries_then_raises():
    client = LiteratureClient(ncbi_api_key="fake-key", proxy=None)
    calls = {"n": 0, "had_key": []}

    def always_blocked(url, params=None, timeout=None, allow_redirects=None):
        calls["n"] += 1
        calls["had_key"].append("api_key" in (params or {}))
        return FakeResponse(ABUSE_URL, ABUSE_HTML, "text/html")

    with patch.object(client.session, "get", side_effect=always_blocked), \
         patch.object(pt, "_NCBI_BACKOFF_BASE", 0.01), \
         patch.object(client, "_rate_delay", 0.0):
        try:
            client._ncbi_get("elink.fcgi", {"db": "pmc", "retmode": "json"})
            raise AssertionError("expected NCBIRateLimitError")
        except NCBIRateLimitError as exc:
            assert "abuse redirect" in str(exc).lower()

    assert calls["n"] == pt._NCBI_MAX_ATTEMPTS, f"expected {pt._NCBI_MAX_ATTEMPTS} attempts, got {calls['n']}"
    assert calls["had_key"] == [True, False, False], f"key should be dropped after attempt 1: {calls['had_key']}"
    print(f"  [2] _ncbi_get retries {calls['n']}x w/ backoff, drops key, raises  PASS")


def test_ncbi_get_recovers_after_transient_block():
    """A burst that clears on retry should succeed, not fail the run."""
    client = LiteratureClient(proxy=None)
    seq = [FakeResponse(ABUSE_URL, ABUSE_HTML, "text/html"),
           FakeResponse("https://eutils.ncbi.nlm.nih.gov/x", '{"linksets": []}')]
    calls = {"n": 0}

    def flaky(url, params=None, timeout=None, allow_redirects=None):
        resp = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return resp

    with patch.object(client.session, "get", side_effect=flaky), \
         patch.object(pt, "_NCBI_BACKOFF_BASE", 0.01), \
         patch.object(client, "_rate_delay", 0.0):
        data = client._ncbi_get_json("elink.fcgi", {"retmode": "json"})
    assert data == {"linksets": []}
    print("  [3] transient block recovers on retry (no run failure)       PASS")


def test_html_body_becomes_rate_limit_error():
    """HTML that slips past the URL check must not surface as JSONDecodeError."""
    client = LiteratureClient(proxy=None)
    good_url_bad_body = FakeResponse(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi", ABUSE_HTML, "text/plain"
    )
    with patch.object(client, "_ncbi_get", return_value=good_url_bad_body):
        try:
            client._ncbi_get_json("elink.fcgi", {})
            raise AssertionError("expected NCBIRateLimitError")
        except NCBIRateLimitError as exc:
            assert "non-JSON" in str(exc)
        except ValueError:
            raise AssertionError("raw JSONDecodeError escaped — this was the original bug")
    print("  [4] non-JSON body -> NCBIRateLimitError (not JSONDecodeError)  PASS")


def test_get_pmc_fulltext_degrades_to_none():
    client = LiteratureClient(proxy=None)
    with patch.object(client, "_ncbi_get_json", side_effect=NCBIRateLimitError("blocked")):
        assert client.get_pmc_fulltext("40860669") is None
    print("  [5] get_pmc_fulltext degrades to None on rate limit          PASS")


def test_fulltext_cached_per_pmid():
    from tools import ncbi_search
    ncbi_search.reset_fulltext_cache()
    calls = {"n": 0}

    class FakeClient:
        def get_pmc_fulltext(self, pmid):
            calls["n"] += 1
            return f"FULL TEXT for {pmid}"

    with patch.object(ncbi_search, "_get_lit_client", return_value=FakeClient()):
        for _ in range(10):
            assert ncbi_search._fetch_fulltext_safe("40860669") == "FULL TEXT for 40860669"
    assert calls["n"] == 1, f"expected 1 PMC fetch for 10 lookups, got {calls['n']}"

    # Failures must be cached too, or a blocked PMID gets retried all run.
    ncbi_search.reset_fulltext_cache()
    fail_calls = {"n": 0}

    class FailingClient:
        def get_pmc_fulltext(self, pmid):
            fail_calls["n"] += 1
            raise NCBIRateLimitError("blocked")

    with patch.object(ncbi_search, "_get_lit_client", return_value=FailingClient()):
        for _ in range(5):
            assert ncbi_search._fetch_fulltext_safe("12345678") is None
    assert fail_calls["n"] == 1, f"failure should be cached, got {fail_calls['n']} attempts"
    print("  [6] full text fetched once per PMID (hits AND misses cached) PASS")


def test_proxy_wired_from_env():
    """The root cause: PMC calls went out unproxied when only NCBI_PROXY was set."""
    with patch.dict(os.environ, {"NCBI_PROXY": "socks5h://127.0.0.1:1080"}, clear=False):
        client = LiteratureClient()
    assert client.session.proxies.get("https") == "socks5h://127.0.0.1:1080", client.session.proxies

    explicit = LiteratureClient(proxy="http://proxy.example:3128")
    assert explicit.session.proxies.get("https") == "http://proxy.example:3128"
    print("  [7] LiteratureClient honours NCBI_PROXY + explicit proxy     PASS")


if __name__ == "__main__":
    print("=== NCBI rate-limit / PMC resilience tests (all network stubbed) ===")
    test_abuse_redirect_detection()
    test_ncbi_get_retries_then_raises()
    test_ncbi_get_recovers_after_transient_block()
    test_html_body_becomes_rate_limit_error()
    test_get_pmc_fulltext_degrades_to_none()
    test_fulltext_cached_per_pmid()
    test_proxy_wired_from_env()
    print("\nAll 7 resilience tests passed.")
