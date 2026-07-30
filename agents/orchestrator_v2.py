"""
Agentic orchestrator for MethyAgent — v2.

Does NOT modify or replace agents/orchestrator.py, which stays the fixed
sequential pipeline used by the working DatabaseAgent/LiteratureAgent flow.
This is a separate, additive path.

Where orchestrator.py hardcodes node order (parse_query -> run_database_agent
-> run_literature_agent -> generate_report), this module exposes each
capability as a tool and lets a single LLM node (LangGraph's prebuilt
create_react_agent) decide which tools to call, in what order, and how many
times, based on the user's query. It is a first, minimal proof of the
"agentic" pattern the mentor asked for — not the full skill-based rewrite
planned for later.

Tools exposed to the agent:
    search_papers            -> tools/ncbi_search.py:search_and_extract()
                                 (Stage 1/2 extraction + LLM reviewer, see
                                 tools/extraction_reviewer.py)
    evaluate_geo_dataset_tool -> tools/query_clarifier.py:evaluate_geo_dataset()
                                 followed by tools/extraction_reviewer.py:
                                 review_geo_verdict() (second-pass QC on the
                                 verdict itself: species / methylation-data-type
                                 / tissue-vs-cfDNA sample type)
    write_to_registry         -> registry/registry.py:Registry.upsert_dataset()

Both reviewer passes (review_extraction inside search_papers, review_geo_verdict
inside evaluate_geo_dataset_tool) are toggled together by config["review"]["enabled"]
(default True) — see config/settings.yaml's `review:` section.

This chain mirrors scripts/pipeline_prototype.py (search -> evaluate -> write),
but there the order/branching is hardcoded by the script; here the LLM decides.

Public API:
    build_orchestrator_v2(config, registry) -> compiled LangGraph app
    run_methyagent_v2(query, config_path=...) -> dict (final structured report)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from registry.registry import Registry
from tools.extraction_reviewer import review_geo_verdict
from tools.geo_tools import GEOClient
from tools.ncbi_search import search_and_extract
from tools.parser_tools import parse_query_rules
from tools.query_clarifier import evaluate_geo_dataset
from utils.llm_factory import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_ORCHESTRATOR_SYSTEM = """You are MethyAgent's autonomous literature and dataset \
acquisition orchestrator. Given a researcher's query about DNA methylation cancer \
datasets, decide which of the following tools to call, in what order, and whether to \
call each one at all, to satisfy the request. You are not following a fixed script — \
use your judgment based on what each tool returns.

Tools available:
  - search_papers(query): search PubMed for matching papers and extract structured
    fields from each (cancer type, sample type, AUC, dataset accessions, markers,
    etc). Each extraction is independently reviewed by a second LLM pass for
    AUC/sample_type consistency and reference-dataset misattribution before it is
    returned to you. Usually the right first step for a natural-language query.
  - evaluate_geo_dataset_tool(accession, cancer_type, sample_types): judge whether a
    GEO series accession (must start with "GSE") found via search_papers is usable
    for this project (checks species, sample type match, metadata completeness).
    sample_types is a comma-separated string, e.g. "plasma,cfdna".
  - write_to_registry(accession, cancer_type, sample_type, recommended_action, reason,
    pmid): persist an evaluated dataset's verdict to the registry. Only call this
    AFTER evaluate_geo_dataset_tool has judged that accession.

Guidance:
  - Only evaluate accessions that look like GEO series IDs (start with "GSE"); other
    accession types (TCGA, dbGaP, NCT trial IDs, etc.) cannot be evaluated by
    evaluate_geo_dataset_tool — skip them.
  - Deduplicate: don't evaluate or write the same accession twice.
  - If search_papers finds no papers or no GSE accessions, do not call the other
    tools — say so in your final summary instead.
  - When you have nothing more useful to do, stop calling tools and reply with a
    concise natural-language summary of what you found, evaluated, and wrote."""


# Convergence guards, tuned from an observed 12-round runaway. A ReAct round costs
# 2 graph steps, so recursion_limit=25 permits ~12 rounds; max_tool_calls must sit
# well below that or the graph dies of recursion before the budget ever trips (which
# is exactly what happened at 12/25 — refused_calls stayed 0). max_searches is the
# real workhorse: search_papers is the expensive tool (~6 LLM calls per invocation).
_DEFAULT_MAX_TOOL_CALLS = 8
_DEFAULT_MAX_SEARCHES = 2
_DEFAULT_RECURSION_LIMIT = 25

_BUDGET_MSG = (
    "Tool-call budget for this run is exhausted. Do not call any more tools — "
    "reply now with your final summary based on what you already have."
)

# Fields the orchestrator LLM actually needs to decide what to do next. Long
# free-text fields (reason, review_report, normal_control_description) are
# deliberately omitted — they are preserved in full in run_trace["papers"].
_PAPER_DIGEST_FIELDS = (
    "pmid", "title", "cancer_type", "sample_type", "technology",
    "dataset_ids", "excluded_reference_datasets", "sample_size_case",
    "sample_size_control", "data_availability", "confidence_level",
    "needs_human_review", "source_basis",
)


def _summarize_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-paper digest for the agent's context (see search_papers)."""
    digest = {k: paper.get(k) for k in _PAPER_DIGEST_FIELDS if paper.get(k) is not None}
    metrics = paper.get("performance_metrics") or {}
    aucs = {k: v for k, v in metrics.items() if v is not None}
    if aucs:
        digest["performance_metrics"] = aucs
    if paper.get("title"):
        digest["title"] = str(paper["title"])[:160]
    return digest


def _normalize_query(query: str) -> str:
    """Collapse whitespace/case so trivially-reworded repeat searches share a cache key."""
    return " ".join((query or "").lower().split())


def _build_dataset_info(accession: str, meta: Dict[str, Any]) -> str:
    """Format GEOClient.get_series_metadata() output as free text for evaluate_geo_dataset()."""
    lines = [f"{accession} — Title: {meta.get('title', '(no title)')}"]
    if meta.get("summary"):
        lines.append(f"Summary: {meta['summary']}")
    if meta.get("sample_titles"):
        lines.append("Sample titles: " + "; ".join(meta["sample_titles"]))
    if meta.get("platform_canonical") or meta.get("platforms"):
        lines.append(f"Platform: {meta.get('platform_canonical') or meta.get('platforms')}")
    if meta.get("sample_count"):
        lines.append(f"Sample count: {meta['sample_count']}")
    if meta.get("data_type"):
        lines.append(f"Data type: {meta['data_type']}")
    return "\n".join(lines)


def build_tools(config: Dict[str, Any], registry: Registry, llm: BaseChatModel):
    """
    Build the tool list the agent can call, plus a run_trace dict that
    accumulates everything the tools did — used to build a deterministic
    final structured report independent of the LLM's own text summary.
    """
    ncbi_key = os.environ.get(config.get("geo", {}).get("api_key_env", ""), "")
    ncbi_proxy = os.environ.get("NCBI_PROXY", "") or config.get("geo", {}).get("proxy", "")
    geo_client = GEOClient(api_key=ncbi_key or None, proxy=ncbi_proxy or None)

    review_enabled = config.get("review", {}).get("enabled", True)
    orch_cfg = config.get("orchestrator", {})
    max_tool_calls = int(orch_cfg.get("max_tool_calls", _DEFAULT_MAX_TOOL_CALLS))
    max_searches = int(orch_cfg.get("max_searches", _DEFAULT_MAX_SEARCHES))

    run_trace: Dict[str, Any] = {
        "papers": [],
        "evaluations": [],
        "registry_writes": [],
        "_geo_meta_cache": {},
        # Convergence bookkeeping. A real run looped search_papers 12x, re-doing
        # ~72 LLM calls of identical work: query-string caching alone did not help
        # because the model rewords its query on every iteration, so every lookup
        # missed. search_calls is therefore capped by COUNT, not by query text.
        "tool_calls": 0,
        "refused_calls": 0,
        "search_calls": 0,
        "_search_cache": {},
        "_eval_cache": {},
        "_last_search_payload": None,
        "skipped_duplicate_calls": [],
    }

    def _budget_exhausted(tool_name: str) -> bool:
        """
        Count this attempt; True once the per-run cap is hit, after which tools
        refuse to do real work (no further NCBI/LLM/registry calls). A cooperative
        model stops here; an uncooperative one is stopped by recursion_limit.
        """
        run_trace["tool_calls"] += 1
        if run_trace["tool_calls"] > max_tool_calls:
            run_trace["refused_calls"] += 1
            logger.warning(
                f"[orchestrator_v2] tool-call budget exhausted "
                f"({max_tool_calls}) — refusing {tool_name}"
            )
            return True
        return False

    @tool
    def search_papers(query: str) -> str:
        """Search PubMed for papers matching the query and extract structured fields
        (cancer type, sample type, AUC, dataset accessions, markers, etc) from each,
        with a second-pass LLM review for AUC/sample_type and dataset_id consistency.
        Returns a JSON array of structured paper records."""
        if _budget_exhausted("search_papers"):
            return _BUDGET_MSG

        key = _normalize_query(query)
        if key in run_trace["_search_cache"]:
            logger.info(f"[orchestrator_v2] search_papers cache hit: {key!r}")
            run_trace["skipped_duplicate_calls"].append(f"search_papers({key!r})")
            return run_trace["_search_cache"][key]

        # Count-based cap. Text-keyed caching is not enough on its own: the model
        # rewords the query every iteration, so each lookup misses and the whole
        # Stage 1/2 pipeline (~6 LLM calls) re-runs for no new information.
        if run_trace["search_calls"] >= max_searches:
            logger.warning(
                f"[orchestrator_v2] search budget spent ({max_searches}) — "
                f"refusing a {run_trace['search_calls'] + 1}th search"
            )
            run_trace["skipped_duplicate_calls"].append(f"search_papers[over-limit]({key!r})")
            return (
                f"Search limit reached ({max_searches} searches per run). You already have "
                f"{len(run_trace['papers'])} paper record(s) from earlier search_papers calls — "
                f"do NOT search again. Use what you have: evaluate any GSE accessions with "
                f"evaluate_geo_dataset_tool, or write your final summary now."
            )

        run_trace["search_calls"] += 1
        intent = parse_query_rules(query)
        papers = search_and_extract(intent, llm, top_n=5, review=review_enabled)

        # Accumulate by PMID. A reworded second search often re-returns the same
        # PMIDs (observed: 2 searches -> 6 records but only 3 distinct papers),
        # which double-counted papers_found and duplicated the saved report.
        # Deduping also makes a zero-yield extra search visible: search_calls
        # rises while papers_found does not.
        # A record with no pmid is kept rather than dropped — it cannot be
        # deduped, but losing it outright would be worse than a rare duplicate.
        seen = {p.get("pmid") for p in run_trace["papers"] if p.get("pmid")}
        new_papers = [p for p in papers if not p.get("pmid") or p["pmid"] not in seen]
        run_trace["papers"].extend(new_papers)
        if len(new_papers) < len(papers):
            logger.info(
                f"[orchestrator_v2] search {run_trace['search_calls']}: "
                f"{len(new_papers)}/{len(papers)} papers were new"
            )

        # Hand the model a compact digest instead of the full extraction blob. The
        # verbose payload (multi-paragraph `reason` + `review_report` per paper) ran
        # to tens of KB and, repeated across iterations, crowded the context enough
        # that the model never progressed to evaluate_geo_dataset_tool. Full records
        # still go to run_trace, so the saved report and registry writes are unchanged.
        payload = json.dumps(
            [_summarize_paper(p) for p in papers], ensure_ascii=False, default=str
        )
        run_trace["_search_cache"][key] = payload
        run_trace["_last_search_payload"] = payload
        return payload

    @tool
    def evaluate_geo_dataset_tool(accession: str, cancer_type: str, sample_types: str) -> str:
        """Fetch a GEO series' metadata (by accession, e.g. GSE50132) and judge whether
        it is usable for this project given the target cancer type and desired sample
        types (comma-separated, e.g. "plasma,cfdna"). Returns a JSON verdict with
        usable/recommended_action/reason."""
        if _budget_exhausted("evaluate_geo_dataset_tool"):
            return _BUDGET_MSG
        eval_key = (accession or "").strip().upper()
        if eval_key in run_trace["_eval_cache"]:
            logger.info(f"[orchestrator_v2] evaluate cache hit: {eval_key}")
            run_trace["skipped_duplicate_calls"].append(
                f"evaluate_geo_dataset_tool({eval_key})"
            )
            return run_trace["_eval_cache"][eval_key]

        meta = geo_client.get_series_metadata(accession)
        if meta.get("error"):
            return json.dumps({"accession": accession, "error": meta["error"]})

        run_trace["_geo_meta_cache"][accession] = meta
        dataset_info = _build_dataset_info(accession, meta)
        types = [t.strip() for t in sample_types.split(",") if t.strip()]
        judgment = evaluate_geo_dataset(dataset_info, cancer_type, types, llm)

        if review_enabled:
            review_report = review_geo_verdict(judgment, meta, llm)
            judgment["review_report"] = review_report
            if review_report.get("needs_human_review"):
                judgment["recommended_action"] = review_report["corrected_fields"].get(
                    "recommended_action", "manual_review"
                )

        run_trace["evaluations"].append({"accession": accession, **judgment})
        payload = json.dumps(judgment, ensure_ascii=False)
        run_trace["_eval_cache"][eval_key] = payload
        return payload

    @tool
    def write_to_registry(
        accession: str,
        cancer_type: str = "",
        sample_type: str = "",
        recommended_action: str = "manual_review",
        reason: str = "",
        pmid: str = "",
    ) -> str:
        """Persist an evaluated dataset's verdict to the registry. Call only after
        evaluate_geo_dataset_tool has judged this accession."""
        if _budget_exhausted("write_to_registry"):
            return _BUDGET_MSG
        if accession in run_trace["registry_writes"]:
            run_trace["skipped_duplicate_calls"].append(f"write_to_registry({accession})")
            return f"{accession} was already written to the registry this session."

        meta = run_trace["_geo_meta_cache"].get(accession, {})
        registry.upsert_dataset(
            accession=accession,
            source="GEO",
            discovered_by="orchestrator_v2",
            data_type=meta.get("data_type"),
            cancer_type=cancer_type or None,
            platform=meta.get("platform_canonical"),
            sample_count=meta.get("sample_count"),
            year=meta.get("year"),
            title=meta.get("title"),
            paper_pmid=pmid or None,
            download_status="pending",
            needs_review=(recommended_action == "manual_review"),
            llm_evidence=reason,
            sample_type=sample_type or None,
            usable=1 if recommended_action in ("keep", "manual_review") else 0,
            recommended_action=recommended_action,
            reason=reason,
        )
        run_trace["registry_writes"].append(accession)
        return f"Wrote {accession} to registry (action={recommended_action})"

    return [search_papers, evaluate_geo_dataset_tool, write_to_registry], run_trace


def build_orchestrator_v2(config: Dict[str, Any], registry: Registry, llm: Optional[BaseChatModel] = None):
    """
    Build the agentic graph: a single create_react_agent tool-calling loop.

    Args:
        config:   Settings dict from settings.yaml.
        registry: Shared Registry instance.
        llm:      Optional pre-built chat model (mainly for tests, so a stub
                  can be injected without touching config). Defaults to
                  get_llm(config["llm"]).

    Returns:
        (compiled LangGraph app, run_trace dict) — the run_trace is mutated
        in place by tool calls during app.invoke(), so read it afterward.
    """
    from langgraph.prebuilt import create_react_agent

    llm = llm or get_llm(config["llm"])
    tools, run_trace = build_tools(config, registry, llm)

    app = create_react_agent(llm, tools, prompt=_ORCHESTRATOR_SYSTEM)
    return app, run_trace


def _final_ai_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (msg.content or "").strip():
            return msg.content
    return ""


def run_methyagent_v2(
    query: str,
    config_path: str = "config/settings.yaml",
    llm: Optional[BaseChatModel] = None,
    save_log: bool = True,
) -> Dict[str, Any]:
    """
    Run the agentic orchestrator end-to-end for a query.

    Returns a structured report dict:
        {
          "query", "timestamp", "papers_found", "gse_evaluated",
          "registry_writes", "agent_summary", "messages" (role/content trace),
        }
    and (if save_log) writes it as JSON under config["download"]["output_dir"].
    """
    config = load_config(config_path)
    registry = Registry(config["registry"]["db_path"])

    app, run_trace = build_orchestrator_v2(config, registry, llm=llm)

    recursion_limit = int(
        config.get("orchestrator", {}).get("recursion_limit", _DEFAULT_RECURSION_LIMIT)
    )

    logger.info(
        f"[orchestrator_v2] Starting agentic run | query='{query}' "
        f"| recursion_limit={recursion_limit}"
    )
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": recursion_limit},
        )
        messages: List[BaseMessage] = result.get("messages", [])
    except GraphRecursionError:
        # Hard backstop: the per-tool budget should normally stop the loop first.
        # Preserve whatever the tools already accomplished rather than losing the run.
        logger.error(
            f"[orchestrator_v2] Hit recursion_limit={recursion_limit} without converging; "
            f"reporting partial results ({run_trace['tool_calls']} tool calls made)."
        )
        messages = []

    logger.info(
        f"[orchestrator_v2] Completed | {len(messages)} messages | "
        f"{run_trace['tool_calls']} tool calls | "
        f"{len(run_trace['skipped_duplicate_calls'])} duplicates skipped"
    )

    report = {
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "papers_found": len(run_trace["papers"]),
        "papers": run_trace["papers"],
        "gse_evaluated": run_trace["evaluations"],
        "registry_writes": run_trace["registry_writes"],
        "tool_calls": run_trace["tool_calls"],
        "refused_calls": run_trace["refused_calls"],
        "search_calls": run_trace["search_calls"],
        "skipped_duplicate_calls": run_trace["skipped_duplicate_calls"],
        "agent_summary": _final_ai_text(messages),
        "messages": [_message_to_dict(m) for m in messages],
    }

    if save_log:
        output_dir = Path(config["download"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_dir / f"orchestrator_v2_run_{ts}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        report["log_path"] = str(log_path)
        logger.info(f"[orchestrator_v2] Run log saved: {log_path}")

    return report


def _message_to_dict(msg: BaseMessage) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": msg.__class__.__name__, "content": msg.content}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls]
    if isinstance(msg, ToolMessage):
        d["tool_call_id"] = msg.tool_call_id
        d["name"] = getattr(msg, "name", None)
    return d


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "breast cancer plasma cfDNA methylation EPIC early detection"
    report = run_methyagent_v2(query)
    print(json.dumps(
        {k: v for k, v in report.items() if k != "messages"},
        ensure_ascii=False, indent=2, default=str,
    ))
