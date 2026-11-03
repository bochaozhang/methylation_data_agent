"""
Tests for NCBI abuse-block handling in the Tier-3 GSM page scrape
(fetch_gsm_supplementary_file) — the GSE186007 incident class.

Before the fix, an abuse redirect was followed by requests into a 200 misuse
page; BeautifulSoup found no /suppl/ link and the scraper returned None —
indistinguishable from "this GSM has no file" — silently degrading large
Tier-3 downloads to partial sets (caught only downstream by the Phase-2
column-count verify). Expected behaviour now:

  * abuse redirect → backoff retries → GEOPageBlockedError (never None)
  * a block starts a cooldown: subsequent scrapes fast-fail without HTTP
  * recovery: a retry that gets a real page clears the cooldown
  * a genuinely link-less page still returns None (no false positives)
  * DownloadSkill.process_dataset maps the error to outcome "failed" with an
    ncbi_blocked note instead of quietly recording no files / no_file

No network: GEOClient.session is replaced with a MagicMock.
"""
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools import geo_tools
from tools.geo_tools import GEOClient, GEOPageBlockedError


def _ok_resp(url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM_TEST"):
    r = MagicMock()
    r.url = url
    r.text = ('<html><a href="https://ftp.ncbi.nlm.nih.gov/geo/samples/'
              'GSM7426nnnn/suppl/GSM7426116_CpG.cov.gz">supplementary file</a></html>')
    r.raise_for_status = lambda: None
    return r


def _blocked_resp():
    r = MagicMock()
    r.url = "https://misuse.ncbi.nlm.nih.gov/index.html"
    r.text = "<html>possible abuse</html>"
    return r


class TestGSMScrapeBlocked(unittest.TestCase):
    def setUp(self):
        self.client = GEOClient()
        self.client._rate_limit_delay = 0  # no pacing in tests
        self.session = MagicMock()
        self.client.session = self.session

    def test_blocked_after_retries_raises(self):
        self.session.get.return_value = _blocked_resp()
        with patch.object(geo_tools, "_GSM_SCRAPE_BACKOFFS", (0.0, 0.0)):
            with self.assertRaises(GEOPageBlockedError):
                self.client.fetch_gsm_supplementary_file("GSM_TEST")
        # 1 initial attempt + 2 backoff retries
        self.assertEqual(self.session.get.call_count, 3)

    def test_cooldown_fast_fail_no_http(self):
        self.session.get.return_value = _blocked_resp()
        with patch.object(geo_tools, "_GSM_SCRAPE_BACKOFFS", (0.0, 0.0)):
            with self.assertRaises(GEOPageBlockedError):
                self.client.fetch_gsm_supplementary_file("GSM_A")
            # Second GSM inside the cooldown window fast-fails: no backoff
            # ladder, and crucially no further HTTP hits on the blocked host.
            with self.assertRaises(GEOPageBlockedError):
                self.client.fetch_gsm_supplementary_file("GSM_B")
        self.assertEqual(self.session.get.call_count, 3)

    def test_recovery_clears_cooldown(self):
        self.session.get.side_effect = [_blocked_resp(), _ok_resp(), _ok_resp()]
        with patch.object(geo_tools, "_GSM_SCRAPE_BACKOFFS", (0.0, 0.0)):
            info = self.client.fetch_gsm_supplementary_file("GSM_A")
            self.assertEqual(info["gsm"], "GSM_A")
            self.assertIn("/suppl/", info["url"])
            # NCBI served a real page → cooldown must be lifted...
            self.assertEqual(self.client._gsm_blocked_until, 0.0)
            # ...so the next GSM scrapes normally instead of fast-failing.
            info2 = self.client.fetch_gsm_supplementary_file("GSM_B")
            self.assertEqual(info2["gsm"], "GSM_B")
        self.assertEqual(self.session.get.call_count, 3)

    def test_linkless_page_still_returns_none(self):
        """No abuse redirect + no supp links = a genuine no-file GSM → None."""
        r = _ok_resp()
        r.text = "<html>sample page without supplementary sections</html>"
        self.session.get.return_value = r
        self.assertIsNone(self.client.fetch_gsm_supplementary_file("GSM_TEST"))


class TestSkillBlockedSemantics(unittest.TestCase):
    def test_tier3_blocked_fails_dataset_not_no_file(self):
        """Tier-3 scrape blocked → outcome failed + ncbi_blocked note.

        The regression this guards: the same situation used to fall through
        as "Tier 3 found no GSM supp files" → empty download → no_file,
        silently losing a dataset that demonstrably has files.
        """
        from skills.geo_download.skill import DownloadSkill

        config = {"download": {"output_dir": tempfile.mkdtemp(prefix="gsm_blocked_"),
                               "max_concurrent": 1, "retry_attempts": 1,
                               "retry_delay": 0, "chunk_size_mb": 1, "timeout": 5},
                  "geo": {}, "llm": {}}
        skill = DownloadSkill(config, registry=None)
        skill.downloader = MagicMock()  # must never be reached

        def _raise(gsm):
            raise GEOPageBlockedError(
                f"NCBI abuse redirect for {gsm} page scrape persisted after 2 backoff retries")

        with patch.object(skill, "geo_client") as client:
            client.series_matrix_has_data.return_value = False
            client.fetch_gsm_supplementary_file.side_effect = _raise
            with patch("skills.geo_download.skill._read_downloadable_gsms",
                       return_value=["GSM7426116"]):
                result = skill.process_dataset(
                    {"accession": "GSE186007", "flags": ""},
                    query_terms=["colorectal"],
                    output_dir=config["download"]["output_dir"])

        self.assertEqual(result["outcome_final"], "failed")
        self.assertIn("ncbi_blocked", result["notes"])
        self.assertEqual(result["files_downloaded"], [])
        skill.downloader.download_many_sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
