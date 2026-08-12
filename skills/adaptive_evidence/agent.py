"""
Bounded ReAct agent for adaptive evidence gathering (path B).

Fires when geo_filter returns manual_review. The model, with SPEC as its system
prompt and real fetch tools bound via to_tool()/bind_tools, autonomously gathers
evidence (PMC full text / supplementary tables / GSE->PubMed reverse lookup / more
GSMs) and re-judges by calling `conclude`.

Guarantees (the guards — required regardless of path A/B):
  - max_steps          : hard loop bound.
  - max_fetches        : hard budget on evidence fetches.
  - ledger             : (tool, target) dedup — never re-fetch the same thing.
  - conclude           : returns a verdict merged over the first-pass verdict.
  - fallback           : budget exhausted / no conclude / hallucination storm /
                         exception / max_steps -> return the first-pass verdict
                         (manual_review) = TODAY's behaviour. Worst case = today.

Public surface: run_evidence_agent(llm, ctx, ds, intent, first_verdict) -> (verdict, trace)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from skills.adaptive_evidence.skills import build_tools
from skills.geo_filter.skill import SPEC  # the filtering rules — single source of truth

_AGENT_PROTOCOL = """
========================================
ADAPTIVE EVIDENCE-GATHERING PROTOCOL
========================================
You are re-judging ONE GEO dataset that was initially marked manual_review (the
evidence was insufficient). You have tools to gather MORE evidence. Apply the
filtering SPEC above to decide the final outcome.

Procedure:
1. Read the dataset metadata, the initial verdict, and what evidence is missing.
2. Pick the ONE piece of evidence most likely to resolve the uncertainty:
     - sample type / controls unclear         -> fetch_more_gsm
     - has PMID, abstract missing              -> fetch_abstract
     - no PMID                                 -> pubmed_reverse_lookup
     - abstract present but insufficient       -> fetch_full_text
     - file form / sample counts from files    -> fetch_supplementary_table
3. Call that tool ONCE. Re-assess with the result.
4. If still uncertain AND a different evidence type could help, fetch ONE more
   (never repeat a fetch you already made).
5. As soon as you can decide, call `conclude` with the final outcome + reason.
   If the dataset genuinely cannot be resolved from available evidence, call
   conclude with outcome=manual_review.

Hard rules:
- Never repeat the same fetch. Call at most a few tools, then conclude.
- Four outcomes (see SPEC): download / lead / exclude / manual_review.
- Apply the SPEC hard gates: cell line / organoid / animal / in-vitro / treated /
  metastasis-only / non-target-unsplittable / non-methylation -> exclude.
- plasma / serum = cfDNA.
"""

_VALID_OUTCOMES = {"download", "lead", "exclude", "manual_review"}
_OUTCOME_TO_LEGACY = {
    "download": ("download", "yes"),
    "lead": ("lead", "partial"),
    "exclude": ("exclude", "no"),
    "manual_review": ("manual_review", "unclear"),
}


def _intent_cancer(intent: Dict[str, Any]) -> str:
    ct = intent.get("cancer_type")
    if isinstance(ct, dict):
        return ct.get("display") or "not specified"
    return ct or intent.get("cancer_type_display") or "not specified"


def _scenario_block(ds: Dict[str, Any], intent: Dict[str, Any],
                    first_verdict: Dict[str, Any]) -> str:
    pmids = ds.get("pubmed_ids") or []
    return (
        f"=== USER REQUEST ===\n"
        f"Cancer type: {_intent_cancer(intent)}\n"
        f"Sample type: {intent.get('sample_type') or 'not specified'}\n"
        f"Original query: {(intent.get('raw_query') or '')[:200]}\n\n"
        f"=== DATASET ===\n"
        f"Accession: {ds.get('accession', '?')}\n"
        f"Title: {(ds.get('title') or '')[:200]}\n"
        f"Summary: {(ds.get('summary') or '')[:500]}\n"
        f"Overall Design: {(ds.get('overall_design') or '')[:300]}\n"
        f"Platform: {ds.get('platform_canonical') or ds.get('platforms', [])}\n"
        f"Sample count (GEO): {ds.get('sample_count')}\n"
        f"PubMed IDs: {pmids}\n\n"
        f"=== INITIAL VERDICT (manual_review -- needs more evidence) ===\n"
        f"reason: {(first_verdict.get('reason') or '')[:300]}\n"
        f"notes: {(first_verdict.get('notes') or '')[:200]}\n"
    )


def _merge_verdict(first_verdict: Dict[str, Any], conclude_args: Dict[str, Any]) -> Dict[str, Any]:
    """Carry over first-pass fields (files, gsm_includes, ...); override the ones conclude set."""
    outcome = conclude_args.get("outcome")
    if outcome not in _VALID_OUTCOMES:
        outcome = "manual_review"
    merged = dict(first_verdict)
    merged["outcome"] = outcome
    for k in ("reason", "notes", "reasoning", "confirmed_sample_type", "confirmed_cancer_type"):
        v = conclude_args.get(k)
        if v not in (None, ""):
            merged[k] = v
    rec_action, usable = _OUTCOME_TO_LEGACY.get(outcome, ("manual_review", "unclear"))
    merged["recommended_action"] = rec_action
    merged["usable"] = usable
    merged["_adaptive"] = True
    return merged


def _target_key(name: str, args: Dict[str, Any]) -> str:
    """Per-tool dedup key: which target a fetch hits."""
    if name in ("fetch_abstract", "fetch_full_text"):
        return f"pmid:{(args.get('pmid') or '').strip()}"
    if name == "pubmed_reverse_lookup":
        return f"q:{(args.get('query') or '').lower()[:80]}"
    if name in ("fetch_supplementary_table", "fetch_more_gsm"):
        return f"acc:{(args.get('accession') or '').strip()}:{(args.get('group') or '').strip()}"
    return ""


def _tc_field(tc: Any, key: str, default: Any = "") -> Any:
    if isinstance(tc, dict):
        return tc.get(key, default)
    return getattr(tc, key, default)


def run_evidence_agent(
    llm: Any,
    ctx: Any,
    ds: Dict[str, Any],
    intent: Dict[str, Any],
    first_verdict: Dict[str, Any],
    *,
    max_steps: int = 4,
    max_fetches: int = 6,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run the bounded adaptive evidence loop. Returns (verdict, trace).

    verdict = the conclude-merged verdict, OR first_verdict on any fallback.
    trace   = a list of {step, event, ...} dicts for auditing (logged to query_log).
    """
    tools = build_tools(ctx)
    valid_names = {t.name for t in tools}
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)

    sys_prompt = SPEC + "\n" + _AGENT_PROTOCOL
    msgs: List[Any] = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=_scenario_block(ds, intent, first_verdict)),
    ]
    ledger: set = set()
    trace: List[Dict[str, Any]] = []
    n_fetches = 0

    try:
        for step in range(max_steps):
            resp = bound.invoke(msgs)
            msgs.append(resp)
            tcs = list(getattr(resp, "tool_calls", None) or [])

            if not tcs:
                trace.append({"step": step + 1, "event": "no_tool_call", "fallback": True})
                return first_verdict, trace  # model stopped without conclude -> fallback

            for tc in tcs:
                name = _tc_field(tc, "name", "")
                args = _tc_field(tc, "args", {})
                tcid = _tc_field(tc, "id", "")
                if not isinstance(args, dict):
                    args = {}

                # 1) conclude -> merge + return
                if name == "conclude":
                    verdict = _merge_verdict(first_verdict, args)
                    trace.append({"step": step + 1, "event": "conclude",
                                  "outcome": verdict.get("outcome")})
                    return verdict, trace

                # 2) hallucinated tool name -> nudge, continue
                if name not in valid_names:
                    trace.append({"step": step + 1, "event": "hallucinated_tool", "name": name})
                    msgs.append(SystemMessage(
                        content=f"'{name}' is not a valid tool. Use one of: {sorted(valid_names)}."))
                    continue

                # 3) repeat fetch -> block, nudge conclude
                tk = _target_key(name, args)
                if (name, tk) in ledger:
                    trace.append({"step": step + 1, "event": "repeat_blocked", "name": name})
                    msgs.append(SystemMessage(
                        content=f"You already called {name} for that target — do not repeat. "
                                f"Call a different tool or conclude."))
                    continue

                # 4) budget exhausted -> nudge conclude
                if n_fetches >= max_fetches:
                    trace.append({"step": step + 1, "event": "budget_exhausted"})
                    msgs.append(SystemMessage(
                        content="Fetch budget exhausted — call conclude now with your best verdict."))
                    continue

                # 5) execute the real fetch tool
                tool = by_name[name]
                try:
                    obs = tool.invoke(args)
                except Exception as e:  # noqa: BLE001  -- isolate fetch failures
                    obs = f"(tool error: {e})"
                ledger.add((name, tk))
                n_fetches += 1
                trace.append({"step": step + 1, "event": "fetch", "name": name, "target": tk})
                msgs.append(ToolMessage(content=str(obs), tool_call_id=tcid or name))

        # loop exhausted without conclude -> fallback
        trace.append({"event": "max_steps_exhausted", "fallback": True})
        return first_verdict, trace

    except Exception as e:  # noqa: BLE001  -- any unexpected error -> fallback
        trace.append({"event": "agent_error",
                      "error": f"{type(e).__name__}: {e}", "fallback": True})
        return first_verdict, trace
