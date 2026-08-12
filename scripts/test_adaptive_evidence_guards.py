"""
Unit tests for the adaptive-evidence agent GUARDS (no LLM, no network).

A stub LLM emits a scripted sequence of tool_calls per step; fake GEO/Literature
clients return canned evidence. These tests assert the bounded-loop invariants:

  1. conclude -> merged verdict (carry-over + override).
  2. one fetch then conclude -> normal path.
  3. repeat fetch -> blocked, then conclude (ledger dedup).
  4. budget exhausted (max_fetches) -> nudged to conclude.
  5. max_steps exhausted without conclude -> fallback to first_verdict.
  6. no tool_call -> fallback.
  7. hallucinated tool name -> nudged, then conclude.
  8. agent exception -> fallback.

Run: .venv/bin/python -m scripts.test_adaptive_evidence_guards
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.adaptive_evidence.agent import run_evidence_agent  # noqa: E402
from skills.base import SkillContext  # noqa: E402

DS = {"accession": "GSE999", "title": "Test dataset", "summary": "s",
      "overall_design": "d", "sample_count": 10, "pubmed_ids": ["111"]}
INTENT = {"cancer_type": "colorectal cancer", "sample_type": "plasma",
          "raw_query": "crc cfDNA"}
FIRST = {"outcome": "manual_review", "reason": "unclear sample type",
         "notes": "", "files": [{"name": "matrix.csv"}], "gsm_includes": [{"gsm": "GSM1"}]}


# --------------------------------------------------------------------------- #
#  Stubs                                                                       #
# --------------------------------------------------------------------------- #
class StubResp:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.content = ""


class StubLLM:
    """bind_tools()->self; invoke()-> next scripted response (or [] when exhausted)."""
    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        if not self.script:
            return StubResp([])
        return StubResp(self.script.pop(0))


class RaisingLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        raise RuntimeError("API down")


def _tc(name, **args):
    return {"name": name, "args": args, "id": name + "_id"}


class FakeGeo:
    def fetch_pubmed_abstract(self, pmid): return f"ABSTRACT[{pmid}]: cfDNA CRC patients n=60."
    def _get_supplementary_urls(self, acc): return ["https://x/%s_matrix.csv" % acc]
    def fetch_file_head(self, url, max_bytes=65536): return b"sample_id,disease_status,beta\n"
    def get_representative_gsm_details(self, acc, wanted_sample_type=""):
        return [{"gsm": "GSM1", "source_name": "plasma", "molecule": "genomic DNA",
                 "group": "plasma_cfdna", "characteristics": {"disease": "CRC"}}]
    def get_all_gsm_metadata(self, acc): return self.get_representative_gsm_details(acc)


class FakeLit:
    def search_pubmed(self, q, max_results=5):
        return [{"pmid": "111", "title": "CRC cfDNA study", "abstract": "abstract text"}]
    def get_pmc_fulltext(self, pmid): return "METHODS: plasma cfDNA from CRC patients."
    def get_pmc_data_availability(self, pmid): return "DATA AVAILABILITY: matrix on GEO."


def _ctx():
    return SkillContext(config={}, geo_client=FakeGeo(), lit_client=FakeLit(), llm=None)


def _run(stub, **kw):
    return run_evidence_agent(stub, _ctx(), DS, INTENT, FIRST, **kw)


def _events(trace):
    return [t.get("event") for t in trace]


# --------------------------------------------------------------------------- #
#  Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_conclude_immediate():
    stub = StubLLM([[_tc("conclude", outcome="exclude", reason="cell line",
                         confirmed_sample_type="cell_line")]])
    v, trace = _run(stub)
    assert v["outcome"] == "exclude", v
    assert v["reason"] == "cell line"
    assert v["confirmed_sample_type"] == "cell_line"
    assert v["recommended_action"] == "exclude" and v["usable"] == "no"
    # carry-over preserved
    assert v["files"] == [{"name": "matrix.csv"}] and v["gsm_includes"] == [{"gsm": "GSM1"}]
    assert v.get("_adaptive") is True
    assert _events(trace) == ["conclude"]
    print("  [1] conclude immediate -> merged verdict (carry-over + override)   PASS")


def test_fetch_then_conclude():
    stub = StubLLM([[_tc("fetch_abstract", pmid="111")],
                    [_tc("conclude", outcome="download", reason="plasma cfDNA confirmed",
                         confirmed_sample_type="plasma")]])
    v, trace = _run(stub)
    assert v["outcome"] == "download"
    assert _events(trace) == ["fetch", "conclude"]
    assert trace[0]["name"] == "fetch_abstract"
    print("  [2] one fetch then conclude -> normal path                         PASS")


def test_repeat_blocked():
    stub = StubLLM([[_tc("fetch_abstract", pmid="111")],
                    [_tc("fetch_abstract", pmid="111")],   # same target -> blocked
                    [_tc("conclude", outcome="manual_review", reason="still unclear")]])
    v, trace = _run(stub)
    assert _events(trace) == ["fetch", "repeat_blocked", "conclude"]
    print("  [3] repeat fetch blocked by ledger, then conclude                 PASS")


def test_budget_exhausted():
    stub = StubLLM([[_tc("fetch_abstract", pmid="111")],
                    [_tc("fetch_more_gsm", accession="GSE999")],  # 2nd fetch -> budget
                    [_tc("conclude", outcome="lead", reason="limited")]])
    v, trace = _run(stub, max_fetches=1, max_steps=4)
    assert _events(trace) == ["fetch", "budget_exhausted", "conclude"]
    assert v["outcome"] == "lead"
    print("  [4] max_fetches budget -> nudged to conclude                       PASS")


def test_max_steps_exhausted_fallback():
    stub = StubLLM([[_tc("fetch_abstract", pmid="111")],
                    [_tc("fetch_more_gsm", accession="GSE999")]])  # no conclude
    v, trace = _run(stub, max_steps=2)
    assert v is FIRST, "must fall back to first_verdict (manual_review)"
    assert v["outcome"] == "manual_review"
    assert trace[-1].get("event") == "max_steps_exhausted"
    print("  [5] max_steps exhausted -> fallback to first_verdict (today)      PASS")


def test_no_tool_call_fallback():
    stub = StubLLM([[]])  # model emits nothing
    v, trace = _run(stub)
    assert v is FIRST
    assert trace[-1].get("event") == "no_tool_call"
    print("  [6] no tool_call -> fallback                                        PASS")


def test_hallucinated_tool_nudged():
    stub = StubLLM([[_tc("bogus_tool", x="y")],
                    [_tc("conclude", outcome="exclude", reason="non-methylation")]])
    v, trace = _run(stub)
    assert _events(trace) == ["hallucinated_tool", "conclude"]
    assert v["outcome"] == "exclude"
    print("  [7] hallucinated tool name -> nudged, then conclude                PASS")


def test_exception_fallback():
    v, trace = _run(RaisingLLM())
    assert v is FIRST
    assert trace[-1].get("event") == "agent_error"
    print("  [8] agent exception -> fallback, pipeline not broken               PASS")


if __name__ == "__main__":
    print("=== adaptive_evidence guard tests (stub LLM, no network) ===")
    test_conclude_immediate()
    test_fetch_then_conclude()
    test_repeat_blocked()
    test_budget_exhausted()
    test_max_steps_exhausted_fallback()
    test_no_tool_call_fallback()
    test_hallucinated_tool_nudged()
    test_exception_fallback()
    print("\nAll 8 guard tests passed.")
