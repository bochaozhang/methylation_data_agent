"""
Environment + live tests for the geo_filter debug harness.

Two tiers, by cost:

1. TestEnvironment (runs on EVERY `unittest discover` — cheap):
   Verifies the live environment is reachable via `./debug.sh --check`, which
   loads .env and probes (auto-starting if needed) the NCBI SOCKS tunnel with
   a single curl to eutils. No LLM call, no query — just connectivity.
   SKIPs if the environment can't be provisioned (no API key / tunnel).

2. TestDebugGeoFilterLive (OPT-IN — costs an LLM call):
   Runs the full debug harness on one GSE and asserts a non-empty reasoning
   chain. Only runs when RUN_LIVE=1 is set, so routine test runs stay cheap.
"""
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEBUG_SH = ROOT / "debug.sh"

GSE = os.environ.get("DEBUG_GSE", "GSE124600")
QUERY = os.environ.get("DEBUG_QUERY", "colorectal cancer和非癌对照的cfDNA甲基化数据")


def _load_dotenv() -> None:
    """Load .env into os.environ (without overriding already-set vars)."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env_ready():
    """
    Check the live environment. Returns (ready: bool, reason: str).

    `debug.sh --check` loads .env and probes/starts the NCBI tunnel with one
    curl, so this CHECKS and AUTO-PROVISIONS connectivity in one step — no
    LLM call, no query.
    """
    _load_dotenv()
    key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not key:
        return False, "no LLM API key in env or .env"
    try:
        r = subprocess.run(
            ["bash", str(DEBUG_SH), "--check"],
            capture_output=True, text=True, timeout=45,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, f"env check failed: {e}"
    reason = (r.stdout or r.stderr).strip()
    return r.returncode == 0, reason


class TestEnvironment(unittest.TestCase):
    """Cheap connectivity check — runs on every test invocation (no LLM call)."""

    def test_live_environment_ready(self):
        """.env key present AND NCBI reachable through the (auto-provisioned) tunnel."""
        ready, reason = _env_ready()
        if not ready:
            self.skipTest(f"environment not provisionable: {reason}")
        # ready == True here → tunnel is up (curl succeeded)
        self.assertTrue(ready)


@unittest.skipUnless(os.environ.get("RUN_LIVE"), "set RUN_LIVE=1 to run the full LLM query test")
class TestDebugGeoFilterLive(unittest.TestCase):
    """Full harness run — OPT-IN (costs an LLM call)."""

    def setUp(self):
        ready, reason = _env_ready()
        if not ready:
            self.skipTest(f"environment not provisionable: {reason}")

    def test_harness_emits_reasoning_chain(self):
        """The harness must produce a non-empty reasoning chain + verdict for a known GSE."""
        try:
            r = subprocess.run(
                ["bash", str(DEBUG_SH),
                 "--query", QUERY, "--accession", GSE, "--no-abstract"],
                capture_output=True, text=True, timeout=120, cwd=str(ROOT),
            )
        except subprocess.TimeoutExpired:
            self.skipTest("harness timed out (network/LLM slow)")
            return

        self.assertEqual(
            r.returncode, 0,
            f"harness exited {r.returncode}\n--- stderr ---\n{r.stderr[-1500:]}",
        )

        out = r.stdout
        self.assertIn("REASONING CHAIN", out)
        self.assertIn("VERDICT:", out)

        m = re.search(r"REASONING CHAIN:\n-+\n(.*?)\n-+", out, re.S)
        self.assertIsNotNone(m, "reasoning chain block not found in output")
        chain = m.group(1).strip()
        self.assertGreater(
            len(chain), 30,
            f"reasoning chain is suspiciously short:\n{chain}",
        )


if __name__ == "__main__":
    unittest.main()
