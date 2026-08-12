"""
Phase 0 benchmark — DeepSeek/GLM tool-calling reliability (A vs B decision).

QUESTION: is bind_tools/ReAct (path B) reliable enough on deepseek-chat / glm-5.2
to drive an adaptive evidence-gathering loop, or should we use structured-output
plan-then-dispatch (path A)?

METHOD: replay the 133 real geo_filter judgments logged in data/query_logs/ as
"next-action" decision scenarios. Each scenario = (dataset metadata + evidence
snapshot + logged verdict + user intent). Two modes, SAME input:

  Mode A (structured plan): one llm.invoke -> JSON {next_action, evidence_gaps}
                            parsed with skills.geo_filter.skill._safe_json.
  Mode B (bind_tools/ReAct): llm.bind_tools(6 mock tools via the REAL
                             skills.base.to_tool()) + a hand-rolled ReAct loop
                             (<=4 steps, mock ToolMessage observations).

Tools are MOCKED (canned evidence) -> deterministic, zero NCBI calls. Only the
LLM calls are real.

Gate 1 (reliability, no gold needed): well-formed >=95%, termination >=90%,
loop <=5%, arg-valid >=95%, hallucination <=3%. B fails -> A locked.
Gate 2 (correct-action, only if gate 1 passes): correct >=85%.

Run:
    .venv/bin/python -m scripts.bench_toolcall_reliability                 # full
    .venv/bin/python -m scripts.bench_toolcall_reliability --limit 12      # smoke
    .venv/bin/python -m scripts.bench_toolcall_reliability --models deepseek
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
#  .env loader (copied from tests/test_debug_geo_filter.py — no new dep)       #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Scenario corpus from query_logs                                             #
# --------------------------------------------------------------------------- #
USER_INTENT = (
    "User query: colorectal cancer 和非癌对照的 cfDNA 甲基化数据 "
    "(colorectal cancer cfDNA methylation, with non-cancer controls)"
)

ACTION_ENUM = {
    "finish", "abstract", "pubmed_reverse_lookup",
    "full_text", "supplementary_table", "more_gsm",
}


def load_pmid_map() -> Dict[str, str]:
    """accession -> paper_pmid from the registry (so fetch_abstract(pmid) is fair)."""
    import sqlite3
    db = ROOT / "registry" / "methyagent.db"
    out: Dict[str, str] = {}
    if not db.exists():
        return out
    con = sqlite3.connect(str(db))
    try:
        for acc, pmid in con.execute("SELECT accession, paper_pmid FROM datasets"):
            if pmid:
                out[str(acc)] = str(pmid)
    finally:
        con.close()
    return out


def load_scenarios(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load judged rows from the latest query_log CSV as next-action scenarios.

    Each scenario carries the dataset context + an `acceptable` action set:
      resolved (download/lead/exclude) -> {finish}        (must STOP)
      manual_review                    -> {finish} ∪ sensible fetches
    (concluding an unresolvable manual_review is legitimate, so finish is always
    acceptable; a fetch is acceptable only if it matches the evidence gap.)
    """
    files = sorted(glob.glob(str(ROOT / "data" / "query_logs" / "query_*.csv")))
    if not files:
        raise SystemExit("No data/query_logs/query_*.csv found.")
    raw = open(files[-1], encoding="utf-8-sig").read()
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    pmid_map = load_pmid_map()

    scenarios: List[Dict[str, Any]] = []
    for r in rows:
        outcome = (r.get("recommended_action") or "").strip()
        if outcome not in {"download", "lead", "exclude", "manual_review"}:
            continue
        pmid = pmid_map.get((r.get("accession") or "").strip(), "")
        scenarios.append({**r, "_outcome": outcome, "_pmid": pmid,
                          "_acceptable": _acceptable(r, outcome, pmid)})

    # Always keep ALL manual_review; sample the rest if limit is set.
    mr = [s for s in scenarios if s["_outcome"] == "manual_review"]
    rest = [s for s in scenarios if s["_outcome"] != "manual_review"]
    if limit is not None:
        # stratified small sample: all non-exclude + some exclude
        nonexc = [s for s in rest if s["_outcome"] != "exclude"]
        exc = [s for s in rest if s["_outcome"] == "exclude"]
        keep = min(8, len(nonexc))
        rest = nonexc[:keep] + exc[: max(0, limit - len(mr) - keep)]
    scenarios = mr + rest
    return scenarios


def _acceptable(row: Dict[str, Any], outcome: str, pmid: str) -> Set[str]:
    if outcome in {"download", "lead", "exclude"}:
        return {"finish"}
    # manual_review: concluding is always legit; fetches must match the gap
    acc: Set[str] = {"finish"}
    had = (row.get("had_abstract") or "").strip().lower()
    if had == "no":
        # no abstract evidence yet -> fetch it (if a PMID exists) or reverse-lookup
        acc.add("abstract" if pmid else "pubmed_reverse_lookup")
    elif had == "yes":
        # abstract present but still unclear -> need deeper sources
        acc |= {"full_text", "supplementary_table"}
    if "unknown" in (row.get("gsm_groups") or ""):
        acc.add("more_gsm")
    if len(acc) == 1:  # only finish -> permissive fallback
        acc |= {"abstract", "pubmed_reverse_lookup", "full_text", "supplementary_table", "more_gsm"}
    return acc


# --------------------------------------------------------------------------- #
#  Mock evidence-fetch skills (wrapped with the REAL skills.base.to_tool)      #
# --------------------------------------------------------------------------- #
from pydantic import BaseModel, Field  # noqa: E402

from skills.base import Skill, SkillContext, to_tool  # noqa: E402

# Canned evidence returned by every mock tool — plausible enough that the model
# can then decide to finish (tests multi-step convergence in mode B).
_CANNED = {
    "abstract": "[MOCK PUBMED ABSTRACT] We profiled cfDNA methylation in colorectal "
                "cancer patients (n=60) and healthy controls (n=30) using EPIC arrays. "
                "Plasma samples were collected before treatment.",
    "full_text": "[MOCK PMC METHODS] Sample types: plasma cfDNA (cases+controls), "
                 "treatment-naive. Data availability: beta-value matrix on GEO.",
    "supplementary_table": "[MOCK SUPP TABLE] sample_id, disease_status, sample_type "
                           "(plasma/tissue), treatment (none).",
    "more_gsm": "[MOCK GSM DETAILS] additional representative samples confirm plasma cfDNA.",
}


class _Ctx:
    """Build a SkillContext once for to_tool() factories."""
    @staticmethod
    def make() -> SkillContext:
        return SkillContext(config={})


def _mock_tools() -> List[Any]:
    """Return 6 StructuredTools built through the production to_tool() adapter."""
    ctx = _Ctx.make()

    class FinishArgs(BaseModel):
        outcome: str = Field(..., description="final outcome: download|lead|exclude|manual_review")
        reason: str = Field("", description="one-sentence final reason")

    class PmidArgs(BaseModel):
        pmid: str = Field(..., description="PubMed ID")

    class QueryArgs(BaseModel):
        query: str = Field(..., description="search query (e.g. GSE title or accession)")

    class AccArgs(BaseModel):
        accession: str = Field(..., description="GSE accession")
        group: str = Field("", description="optional sample group to expand")

    class FinishSkill(Skill):
        name = "finish"; description = ("Conclude: stop gathering evidence and emit the final "
                                        "verdict. Call this when the dataset is resolved "
                                        "(clear download/lead/exclude) or after enough evidence.")
        args_schema = FinishArgs
        def run(self, ctx, outcome: str = "manual_review", reason: str = "") -> str:
            return f"FINISHED: outcome={outcome}; {reason}"

    class AbstractSkill(Skill):
        name = "fetch_abstract"; description = ("Fetch the PubMed abstract for a PMID. Use when a "
                                                "dataset has a PMID but the abstract was not fetched "
                                                "or was empty (had_abstract=no).")
        args_schema = PmidArgs
        def run(self, ctx, pmid: str = "") -> str:
            return _CANNED["abstract"]

    class ReverseSkill(Skill):
        name = "pubmed_reverse_lookup"; description = ("Search PubMed by GSE title/accession to find "
                                                       "a linked paper, then fetch its abstract. Use "
                                                       "when the dataset has NO PMID.")
        args_schema = QueryArgs
        def run(self, ctx, query: str = "") -> str:
            return _CANNED["abstract"]

    class FullTextSkill(Skill):
        name = "fetch_full_text"; description = ("Fetch PMC open-access full text (Methods / Data "
                                                 "availability). Use when the abstract is available "
                                                 "but insufficient (e.g. treatment status unclear).")
        args_schema = PmidArgs
        def run(self, ctx, pmid: str = "") -> str:
            return _CANNED["full_text"]

    class SuppSkill(Skill):
        name = "fetch_supplementary_table"; description = ("Fetch/preview a GEO supplementary table "
                                                           "or per-GSM file. Use to confirm sample "
                                                           "types/counts from metadata files.")
        args_schema = AccArgs
        def run(self, ctx, accession: str = "", group: str = "") -> str:
            return _CANNED["supplementary_table"]

    class MoreGsmSkill(Skill):
        name = "fetch_more_gsm"; description = ("Fetch additional representative GSM details, "
                                                "especially for 'unknown' sample groups. Use when "
                                                "sample types are unclear from current GSMs.")
        args_schema = AccArgs
        def run(self, ctx, accession: str = "", group: str = "") -> str:
            return _CANNED["more_gsm"]

    skills = [FinishSkill(), AbstractSkill(), ReverseSkill(), FullTextSkill(),
              SuppSkill(), MoreGsmSkill()]
    return [to_tool(s)(ctx) for s in skills]


# --------------------------------------------------------------------------- #
#  Clients                                                                     #
# --------------------------------------------------------------------------- #
def get_clients(models: List[str]) -> Dict[str, Any]:
    from utils.llm_factory import get_llm
    out: Dict[str, Any] = {}
    for m in models:
        if m == "deepseek":
            out[m] = get_llm({"backend": "deepseek"})
        elif m in ("zhipu", "glm"):
            out[m] = get_llm({"backend": "zhipu"})
        else:
            raise SystemExit(f"unknown model: {m}")
    return out


# --------------------------------------------------------------------------- #
#  Prompt                                                                      #
# --------------------------------------------------------------------------- #
SYS_COMMON = (
    "You are deciding the NEXT ACTION for one GEO DNA-methylation dataset that has "
    "ALREADY been judged. You are given: the user's request, the dataset metadata, "
    "the evidence gathered so far, and the current verdict.\n\n"
    "Decide either:\n"
    "  (a) CONCLUDE — the verdict is final. This is correct when the outcome is a clear "
    "download / lead / exclude (the dataset is resolved on the current evidence).\n"
    "  (b) FETCH MORE EVIDENCE — only when the outcome is manual_review and a specific "
    "piece of evidence is missing. Available evidence types:\n"
    "     - abstract: the linked PubMed abstract (use when had_abstract=no and a PMID exists)\n"
    "     - pubmed_reverse_lookup: search PubMed by GSE title/accession (use when there is NO PMID)\n"
    "     - full_text: PMC open-access full text / Methods (use when abstract is had but insufficient)\n"
    "     - supplementary_table: GEO supplementary / per-GSM files (confirm sample types/counts)\n"
    "     - more_gsm: more representative GSM details (clarify 'unknown' sample groups)\n"
)


def human_block(s: Dict[str, Any]) -> str:
    return (
        f"=== USER REQUEST ===\n{USER_INTENT}\n\n"
        f"=== DATASET ===\n"
        f"Accession: {s.get('accession','?')}\n"
        f"Title: {(s.get('title') or '')[:200]}\n"
        f"Sample type (GEO): {s.get('sample_type')}\n"
        f"Cancer type: {s.get('cancer_type')}\n"
        f"Platform: {s.get('platform')}\n"
        f"Sample count: {s.get('sample_count')}\n\n"
        f"=== EVIDENCE GATHERED SO FAR ===\n"
        f"PubMed ID: {s.get('_pmid') or 'none'}\n"
        f"had_abstract: {s.get('had_abstract')}\n"
        f"gsm_groups: {s.get('gsm_groups')}\n"
        f"n_representative_gsm: {s.get('n_representative_gsm')}\n\n"
        f"=== CURRENT VERDICT ===\n"
        f"outcome: {s.get('recommended_action')}\n"
        f"reason: {(s.get('reason') or '')[:300]}\n"
        f"notes: {(s.get('notes') or '')[:200]}\n"
    )


SYS_A = SYS_COMMON + (
    "\nRespond with ONLY a JSON object (no markdown, no prose outside JSON):\n"
    '{"next_action": "finish|abstract|pubmed_reverse_lookup|full_text|supplementary_table|more_gsm", '
    '"evidence_gaps": [{"dimension": "...", "note": "..."}], "reason": "one sentence"}\n'
    "If the outcome is download/lead/exclude, next_action MUST be \"finish\"."
)

SYS_B = SYS_COMMON + (
    "\nExpress your decision by calling EXACTLY ONE tool now:\n"
    "- If the outcome is download/lead/exclude, call `finish`.\n"
    "- Otherwise call the single most appropriate evidence-fetch tool.\n"
    "After you receive evidence, call `finish` with your final outcome. Do not repeat a tool call."
)


# --------------------------------------------------------------------------- #
#  Mode A: structured-output plan (single shot)                                #
# --------------------------------------------------------------------------- #
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: E402

# reuse the production tolerant JSON parser
from skills.geo_filter.skill import _safe_json  # noqa: E402


def run_mode_a(llm: Any, s: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    res = {"mode": "A", "well_formed": False, "correct": False,
           "next_action": None, "tokens": 0, "latency": 0.0, "error": None}
    try:
        resp = llm.invoke([SystemMessage(content=SYS_A), HumanMessage(content=human_block(s))])
        res["tokens"] = _tokens(resp)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        verdict = _safe_json(raw)
        act = (verdict.get("next_action") or "").strip()
        res["next_action"] = act
        res["well_formed"] = act in ACTION_ENUM
        res["correct"] = act in s["_acceptable"]
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    res["latency"] = time.time() - t0
    return res


# --------------------------------------------------------------------------- #
#  Mode B: bind_tools + hand-rolled ReAct loop                                 #
# --------------------------------------------------------------------------- #
MAX_STEPS = 4


def run_mode_b(llm: Any, tools: List[Any], s: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    res = {"mode": "B", "well_formed": True, "correct": False, "terminated": False,
           "looped": False, "arg_valid": True, "hallucinated": False,
           "n_steps": 0, "tokens": 0, "latency": 0.0,
           "tool_calls": [], "terminal": None, "error": None}
    valid_names = {t.name for t in tools}
    seen: Set[Tuple[str, str]] = set()
    called_fetch_acceptable = False
    try:
        bound = llm.bind_tools(tools)
        msgs: List[Any] = [SystemMessage(content=SYS_B), HumanMessage(content=human_block(s))]
        for step in range(MAX_STEPS):
            res["n_steps"] = step + 1
            resp = bound.invoke(msgs)
            res["tokens"] += _tokens(resp)
            msgs.append(resp)
            tcs = list(getattr(resp, "tool_calls", None) or [])
            if not tcs:
                res["terminated"] = True  # model chose to stop
                break
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                res["tool_calls"].append(name)
                if name not in valid_names:
                    res["hallucinated"] = True
                    res["well_formed"] = False
                    continue
                # arg validity: required fields present + non-empty
                if not _args_valid(name, args):
                    res["arg_valid"] = False
                    res["well_formed"] = False
                key = (name, json.dumps(args, sort_keys=True, default=str))
                if key in seen:
                    res["looped"] = True
                seen.add(key)
                act = _ACTION_OF.get(name, name)  # canonical short action name
                if act == "finish":
                    res["terminal"] = "finish"
                elif act in s["_acceptable"]:
                    called_fetch_acceptable = True
                # execute the mock tool, feed observation back
                tool = next((t for t in tools if t.name == name), None)
                obs = tool.invoke(args) if tool is not None else "(tool not found)"
                msgs.append(ToolMessage(content=str(obs), tool_call_id=tcid or name))
                res["terminal"] = act  # last-called action
            if any((tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) == "finish"
                   for tc in tcs):
                res["terminated"] = True
                break
        else:
            # loop exhausted without break -> did not terminate cleanly
            res["terminated"] = False
        # correctness
        if s["_outcome"] in {"download", "lead", "exclude"}:
            res["correct"] = (res["terminal"] == "finish" and res["terminated"])
        else:  # manual_review
            res["correct"] = called_fetch_acceptable
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        res["well_formed"] = False
    res["latency"] = time.time() - t0
    return res


_REQ = {  # required, non-empty fields per tool
    "finish": ("outcome",),
    "fetch_abstract": ("pmid",),
    "pubmed_reverse_lookup": ("query",),
    "fetch_full_text": ("pmid",),
    "fetch_supplementary_table": ("accession",),
    "fetch_more_gsm": ("accession",),
}

# canonical short action name for each (descriptively-named) tool, so mode B's
# tool_calls score against the SAME namespace as mode A / the acceptable sets.
_ACTION_OF = {
    "finish": "finish",
    "fetch_abstract": "abstract",
    "pubmed_reverse_lookup": "pubmed_reverse_lookup",
    "fetch_full_text": "full_text",
    "fetch_supplementary_table": "supplementary_table",
    "fetch_more_gsm": "more_gsm",
}


def _args_valid(name: str, args: Any) -> bool:
    if not isinstance(args, dict):
        return False
    for f in _REQ.get(name, ()):
        v = args.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False
    return True


def _tokens(resp: Any) -> int:
    um = getattr(resp, "usage_metadata", None) or {}
    if isinstance(um, dict):
        return int(um.get("total_tokens") or um.get("input_tokens") or 0)
    return 0


# --------------------------------------------------------------------------- #
#  Driver                                                                      #
# --------------------------------------------------------------------------- #
def run_grid(models: List[str], scenarios: List[Dict[str, Any]],
             concurrency: int) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    tools = _mock_tools()
    clients = get_clients(models)
    jobs: List[Tuple[str, str, Dict[str, Any]]] = []
    for m in models:
        for s in scenarios:
            jobs.append((m, "A", s))
            jobs.append((m, "B", s))

    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {m: {"A": [], "B": []} for m in models}

    def _do(job):
        m, mode, s = job
        try:
            if mode == "A":
                r = run_mode_a(clients[m], s)
            else:
                r = run_mode_b(clients[m], tools, s)
        except Exception as e:  # noqa: BLE001
            r = {"mode": mode, "error": f"DRIVER:{type(e).__name__}:{e}", "well_formed": False,
                 "correct": False, "terminated": False, "looped": False, "tokens": 0, "latency": 0.0}
        r["accession"] = s.get("accession")
        r["outcome"] = s["_outcome"]
        return (m, mode, r)

    n_done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(_do, j): j for j in jobs}
        for fut in as_completed(futs):
            m, mode, r = fut.result()
            results[m][mode].append(r)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  ...{n_done}/{len(jobs)} calls done", flush=True)
    return results


# --------------------------------------------------------------------------- #
#  Scorecard                                                                   #
# --------------------------------------------------------------------------- #
GATE1 = {"well_formed": 0.95, "termination": 0.90, "loop": 0.05,
         "arg_valid": 0.95, "halluc": 0.03}
GATE2_CORRECT = 0.85


def agg(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "well_formed": sum(r.get("well_formed") for r in rows) / n,
        "correct": sum(r.get("correct") for r in rows) / n,
        "termination": sum(r.get("terminated", True) for r in rows) / n,  # A always True
        "loop": sum(r.get("looped", False) for r in rows) / n,
        "arg_valid": sum(r.get("arg_valid", True) for r in rows) / n,
        "halluc": sum(r.get("hallucinated", False) for r in rows) / n,
        "tokens_avg": sum(r.get("tokens", 0) for r in rows) / n,
        "latency_avg": sum(r.get("latency", 0) for r in rows) / n,
        "steps_avg": sum(r.get("n_steps", 1) for r in rows) / n,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def print_scorecard(results: Dict[str, Dict[str, List[Any]]]) -> None:
    for m in results:
        print(f"\n{'='*72}\nMODEL: {m}  (gate1: wf≥95% term≥90% loop≤5% argv≥95% halluc≤3%)\n{'='*72}")
        for mode in ("A", "B"):
            a = agg(results[m][mode])
            tag = "A structured-JSON" if mode == "A" else "B bind_tools/ReAct"
            print(f"\n  [{tag}]  n={a['n']}")
            print(f"     well-formed   : {fmt_pct(a['well_formed'])}"
                  + (f"   (gate ≥95%) {'PASS' if a['well_formed']>=GATE1['well_formed'] else 'FAIL'}" if mode=='B' else ''))
            print(f"     correct-action: {fmt_pct(a['correct'])}")
            if mode == "B":
                print(f"     termination   : {fmt_pct(a['termination'])}"
                      f"   (gate ≥90%) {'PASS' if a['termination']>=GATE1['termination'] else 'FAIL'}")
                print(f"     loop rate     : {fmt_pct(a['loop'])}"
                      f"   (gate ≤5%)  {'PASS' if a['loop']<=GATE1['loop'] else 'FAIL'}")
                print(f"     arg-valid     : {fmt_pct(a['arg_valid'])}"
                      f"   (gate ≥95%) {'PASS' if a['arg_valid']>=GATE1['arg_valid'] else 'FAIL'}")
                print(f"     hallucination : {fmt_pct(a['halluc'])}"
                      f"   (gate ≤3%)  {'PASS' if a['halluc']<=GATE1['halluc'] else 'FAIL'}")
                print(f"     avg steps     : {a['steps_avg']:.2f}")
            print(f"     avg tokens    : {a['tokens_avg']:.0f}")
            print(f"     avg latency(s): {a['latency_avg']:.2f}")
            print(f"     errors        : {a['errors']}")
            # sample errors
            errs = [r for r in results[m][mode] if r.get("error")][:2]
            for e in errs:
                print(f"       err[{e.get('accession')}]: {e['error']}")


def write_csv(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "mode", "accession", "outcome", "well_formed", "correct",
            "terminated", "looped", "arg_valid", "hallucinated", "n_steps",
            "next_action", "terminal", "tool_calls", "tokens", "latency", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in results:
            for mode in ("A", "B"):
                for r in results[m][mode]:
                    w.writerow({
                        "model": m, "mode": mode,
                        "accession": r.get("accession"), "outcome": r.get("outcome"),
                        "well_formed": r.get("well_formed"), "correct": r.get("correct"),
                        "terminated": r.get("terminated", ""), "looped": r.get("looped", ""),
                        "arg_valid": r.get("arg_valid", ""), "hallucinated": r.get("hallucinated", ""),
                        "n_steps": r.get("n_steps", 1),
                        "next_action": r.get("next_action"), "terminal": r.get("terminal"),
                        "tool_calls": "|".join(r.get("tool_calls") or []),
                        "tokens": r.get("tokens", 0), "latency": f"{r.get('latency',0):.2f}",
                        "error": r.get("error"),
                    })


def verdict(results) -> str:
    """Apply gates -> recommendation text."""
    lines = ["\n" + "="*72, "VERDICT", "="*72]
    for m in results:
        a = agg(results[m]["B"])
        g1 = (a["well_formed"] >= GATE1["well_formed"] and a["termination"] >= GATE1["termination"]
              and a["loop"] <= GATE1["loop"] and a["arg_valid"] >= GATE1["arg_valid"]
              and a["halluc"] <= GATE1["halluc"])
        lines.append(f"\n{m}:")
        lines.append(f"  Gate1 (reliability)  : {'PASS' if g1 else 'FAIL'}")
        if g1:
            g2 = a["correct"] >= GATE2_CORRECT
            lines.append(f"  Gate2 (correct≥85%)  : {'PASS' if g2 else 'FAIL'}  (correct={fmt_pct(a['correct'])})")
            lines.append("  -> B is a VIABLE option; weigh vs A on simplicity/cost.")
        else:
            lines.append("  -> B NOT reliable enough; A LOCKED IN for this model.")
    lines.append("\nNote: gate1 reliability is the real A/B decider (no gold needed).")
    lines.append("      correct-action is dominated by 'should-stop' resolved cases; "
                 "manual_review fetch-selection has small N.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap non-manual_review scenarios (smoke)")
    ap.add_argument("--models", default="deepseek,zhipu", help="comma list: deepseek,zhipu")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--csv", default="data/query_logs/bench_toolcall_reliability.csv")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    scenarios = load_scenarios(limit=args.limit)
    nmr = sum(1 for s in scenarios if s["_outcome"] == "manual_review")
    print(f"Loaded {len(scenarios)} scenarios ({nmr} manual_review, "
          f"{len(scenarios)-nmr} resolved). Models: {models}. Concurrency: {args.concurrency}")

    t0 = time.time()
    results = run_grid(models, scenarios, args.concurrency)
    print(f"\nDone in {time.time()-t0:.0f}s.")

    print_scorecard(results)
    print(verdict(results))

    out = ROOT / args.csv
    write_csv(results, out)
    print(f"\nPer-call CSV -> {out}")


if __name__ == "__main__":
    main()
