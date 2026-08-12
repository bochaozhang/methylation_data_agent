"""
End-to-end smoke / replay harness for the adaptive-evidence agent (path B).

Replays historical manual_review datasets through the REAL agent: real LLM
(deepseek-chat) + real GEO/Literature clients (real NCBI fetches, no mocks).
For each accession we synthesize the manual_review first-pass verdict (using the
reason logged in query_logs) and let the bounded ReAct agent gather evidence and
re-judge. Prints before -> after outcome, the tool-call trace, and whether guards
held (steps within budget, no fallback).

This is off the production path (geo.adaptive_evidence stays false in settings) —
it exercises the agent directly.

Run: .venv/bin/python -m scripts.replay_manual_review [GSE1 GSE2 ...]
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_dotenv() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

from skills.adaptive_evidence.agent import run_evidence_agent  # noqa: E402
from skills.base import SkillContext  # noqa: E402
from tools.geo_tools import GEOClient  # noqa: E402
from tools.pubmed_tools import LiteratureClient  # noqa: E402
from utils.llm_factory import get_llm  # noqa: E402

INTENT = {"cancer_type": "colorectal cancer", "sample_type": "plasma",
          "raw_query": "colorectal cancer和非癌对照的cfDNA甲基化数据"}


def logged_reason(accession: str) -> str:
    """Pull the manual_review reason for an accession from the latest query_log."""
    files = sorted(glob.glob(str(ROOT / "data" / "query_logs" / "query_*.csv")))
    if not files:
        return "unclear sample type / controls"
    raw = open(files[-1], encoding="utf-8-sig").read()
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    for r in csv.DictReader(__import__("io").StringIO("\n".join(lines))):
        if (r.get("accession") or "").strip() == accession:
            return (r.get("reason") or "unclear")[:200]
    return "unclear sample type / controls"


def main():
    accessions = sys.argv[1:] or ["GSE208596", "GSE97923"]

    llm = get_llm({"backend": os.environ.get("REPLAY_BACKEND", "deepseek")})
    ncbi_key = os.environ.get("NCBI_API_KEY") or None
    proxy = os.environ.get("NCBI_PROXY") or None
    geo = GEOClient(api_key=ncbi_key, proxy=proxy)
    lit = LiteratureClient(ncbi_api_key=ncbi_key, geo_email=os.environ.get("GEO_EMAIL") or None)
    ctx = SkillContext(config={}, geo_client=geo, lit_client=lit, llm=llm)

    for acc in accessions:
        print(f"\n{'='*72}\n{acc}\n{'='*72}")
        try:
            ds = geo.get_series_metadata(acc)
        except Exception as e:  # noqa: BLE001
            print(f"  metadata fetch failed: {e}")
            continue
        if ds.get("error"):
            print(f"  GEO error: {ds.get('error')}")
            continue

        first = {
            "outcome": "manual_review",
            "reason": logged_reason(acc),
            "notes": "",
            "files": [],
            "gsm_includes": [],
        }
        print(f"  title: {(ds.get('title') or '')[:120]}")
        print(f"  pmids: {ds.get('pubmed_ids')}")
        print(f"  BEFORE: outcome={first['outcome']}  reason={first['reason'][:100]}")

        t0 = time.time()
        verdict, trace = run_evidence_agent(
            llm, ctx, ds, INTENT, first, max_steps=4, max_fetches=6)
        dt = time.time() - t0

        print(f"  AFTER : outcome={verdict.get('outcome')}  "
              f"sample={verdict.get('confirmed_sample_type')}  "
              f"reason={(verdict.get('reason') or '')[:120]}")
        print(f"  trace ({len(trace)} steps, {dt:.1f}s):")
        for t in trace:
            extra = ""
            for k in ("name", "outcome", "target", "fallback"):
                if k in t:
                    extra += f"  {k}={t[k]}"
            print(f"     - {t.get('event')}{extra}")
        fell_back = any(t.get("fallback") for t in trace)
        print(f"  guards: {'FELL BACK (today behaviour)' if fell_back else 'concluded within budget'}")


if __name__ == "__main__":
    main()
