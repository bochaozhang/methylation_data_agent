"""
agent1 deterministic skill pipeline (the "接线").

A LangGraph StateGraph that chains the GEO skills and the TCGA module:

  parse → ┬ geo-search → geo-filter → geo-download ┬→ register → END
          └ tcga-direct ────────────────────────────┘

Deterministic — NO LLM routing. The LLM lives only inside geo-filter (per-dataset
judgment). This is distinct from orchestrator_v2 (ReAct, separate owner).

Each node wraps a skill / module and returns a state-dict update. The register
node is the registry bridge: maps the skill-world State (download_results /
lead_list / manual_review_list / tcga_results) onto the existing SQLite registry
so the Web UI / approval / daemon keep working.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List

from langgraph.graph import END, START, StateGraph

from agents.tcga_direct import run_tcga_direct
from skills.geo_filter import apply_verdict, filter_dataset, split_by_outcome
from skills.geo_filter.file_inspect import verify_a_level_files
from skills.geo_filter.skill import _OUTCOME_TO_LEGACY
from skills.geo_search import SearchSkill
from state.agent1_state import Agent1State, normalize_intent
from tools.geo_tools import GEOClient
from tools.parser_tools import parse_query_rules, parse_query_with_llm
from utils.llm_factory import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------- #
#  Concurrency helper (same asyncio pattern as DatabaseAgent)            #
# ---------------------------------------------------------------------- #

def _run_concurrent(fn: Callable, items: List[Any], *args, max_concurrent: int = 3) -> List[Any]:
    """Run fn(item, *args) over items concurrently (asyncio + Semaphore)."""
    if not items:
        return []
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(item):
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, fn, item, *args)

    async def _all():
        return await asyncio.gather(*[_one(i) for i in items])

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _all()).result()
        return loop.run_until_complete(_all())
    except RuntimeError:
        return asyncio.run(_all())


# ---------------------------------------------------------------------- #
#  Pipeline builder                                                       #
# ---------------------------------------------------------------------- #

def build_agent1_pipeline(config: Dict[str, Any], registry: Any = None):
    """
    Build and compile the agent1 skill pipeline graph.

    Args:
        config: settings dict.
        registry: shared Registry (may also be supplied via state["registry"]).
    """
    llm = get_llm(config["llm"])

    ncbi_key = os.environ.get(config.get("geo", {}).get("api_key_env", ""), "") or None
    ncbi_proxy = (
        os.environ.get("NCBI_PROXY", "")
        or config.get("geo", {}).get("proxy", "")
        or None
    )
    geo_client = GEOClient(api_key=ncbi_key or None, proxy=ncbi_proxy or None)

    search_skill = SearchSkill(config)

    # ---- parse ----
    def parse_node(state: Agent1State) -> Dict[str, Any]:
        raw = state.get("raw_query", "")
        try:
            parsed = parse_query_with_llm(raw, llm)
        except Exception as e:
            logger.warning(f"agent1 parse: LLM failed ({e}), rules fallback")
            parsed = parse_query_rules(raw)
        intent = normalize_intent(raw, parsed)
        logger.info(
            f"agent1 parse: mode={intent.get('mode')} "
            f"cancer={(intent.get('cancer_type') or {}).get('tcga_code')} "
            f"sample={intent.get('sample_type')}"
        )
        return {"parsed_intent": intent}

    # ---- geo-search ----
    def search_node(state: Agent1State) -> Dict[str, Any]:
        return search_skill.run(dict(state))

    # ---- geo-filter ----
    def _filter_one(ds: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        acc = ds.get("accession", "?")
        wanted = intent.get("sample_type", "") or ""
        gsm = geo_client.get_representative_gsm_details(acc, wanted_sample_type=wanted)
        abstract = None
        pmids = ds.get("pubmed_ids") or []
        if pmids:
            try:
                abstract = geo_client.fetch_pubmed_abstract(str(pmids[0]))
            except Exception as e:
                logger.debug(f"agent1 filter abstract {acc}: {e}")
        verdict = filter_dataset(llm, ds, intent, gsm, abstract=abstract)

        # Phase 2a: file-form A-level preview (核验前置). For download/lead
        # candidates, inspect the REAL supplementary file heads and override the
        # LLM's metadata-predicted files[] with file-head evidence. If a
        # download candidate has no A-level file on inspection → downgrade to lead.
        if verdict.get("outcome") in ("download", "lead"):
            supp = ds.get("supplementary_files") or []
            if supp:
                try:
                    has_A, inspected, a_form = verify_a_level_files(supp, geo_client)
                except Exception as e:
                    logger.debug(f"agent1 file verify {acc} failed: {e}")
                    has_A, inspected, a_form = None, None, None
                if inspected is not None:
                    verdict["files"] = inspected
                    if a_form:
                        verdict["available_file_type"] = a_form
                    if verdict["outcome"] == "download" and has_A is False:
                        verdict["outcome"] = "lead"
                        verdict["lead_type"] = "no_A_file"
                        verdict["reason"] = (
                            (verdict.get("reason") or "")
                            + "; downgraded: no A-level file on head inspection"
                        ).strip()
                        rec, usable = _OUTCOME_TO_LEGACY["lead"]
                        verdict["recommended_action"] = rec
                        verdict["usable"] = usable
                        logger.info(f"agent1 file verify {acc}: downgraded download→lead (no A-level file)")

        return apply_verdict(ds, verdict)

    def filter_node(state: Agent1State) -> Dict[str, Any]:
        intent = state.get("parsed_intent") or {}
        candidates = state.get("candidate_gse_list") or []
        if not candidates:
            empty = {"download_list": [], "lead_list": [], "exclude_list": [],
                     "manual_review_list": []}
            return {**empty, "filter_log": "geo-filter: no candidates"}
        judged = _run_concurrent(_filter_one, candidates, intent, max_concurrent=3)
        buckets = split_by_outcome(judged)
        log = (
            f"geo-filter: {len(candidates)} candidates → "
            f"download={len(buckets['download_list'])} "
            f"lead={len(buckets['lead_list'])} "
            f"exclude={len(buckets['exclude_list'])} "
            f"manual_review={len(buckets['manual_review_list'])}"
        )
        logger.info(log)
        return {**buckets, "filter_log": log}

    # ---- tcga-direct (search-only; no download here) ----
    def tcga_node(state: Agent1State) -> Dict[str, Any]:
        return run_tcga_direct(dict(state), config)

    # ---- register (registry bridge) ----
    def register_node(state: Agent1State) -> Dict[str, Any]:
        reg = state.get("registry") or registry
        if reg is None:
            logger.warning("agent1 register: no registry, skipping registry writes")
            return {}
        n = register_state_to_registry(state, reg)
        logger.info(
            f"agent1 register: bucket(review0)={n['bucket']} review_queue(review1)={n['review']} "
            f"excluded={n['excluded']}"
        )
        return {"register_log": str(n)}

    # ---- graph ----
    # No inline download: pipeline only registers. Downloads happen later in the
    # daemon after the user's bulk "待下载" confirm.
    graph = StateGraph(Agent1State)
    graph.add_node("parse", parse_node)
    graph.add_node("search", search_node)
    graph.add_node("filter", filter_node)
    graph.add_node("tcga", tcga_node)
    graph.add_node("register", register_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "search")
    graph.add_edge("search", "filter")
    graph.add_edge("filter", "tcga")
    graph.add_edge("tcga", "register")
    graph.add_edge("register", END)

    return graph.compile()


# ---------------------------------------------------------------------- #
#  Registry bridge helper                                                 #
# ---------------------------------------------------------------------- #

def register_state_to_registry(state: Dict[str, Any], reg: Any) -> Dict[str, int]:
    """
    Register filter/tcga outcomes into the registry. No downloads here.

    - download_list + lead_list + tcga_candidates → awaiting_approval (needs_review=0,
      the bulk "待下载" bucket).
    - manual_review_list → awaiting_approval (needs_review=1, the Review Queue).
    - exclude_list → skipped.
    Returns counts.
    """
    n = {"bucket": 0, "review": 0, "excluded": 0}

    # Bulk "待下载" bucket (needs_review=0): download + lead + TCGA.
    for rec in (state.get("download_list") or []) + (state.get("lead_list") or []):
        _upsert(reg, rec, "awaiting_approval", needs_review=False)
        n["bucket"] += 1
    for rec in state.get("tcga_candidates") or []:
        _upsert(reg, {**rec, "source": "TCGA"}, "awaiting_approval", needs_review=False)
        n["bucket"] += 1

    # Review Queue (needs_review=1): manual_review.
    for rec in state.get("manual_review_list") or []:
        _upsert(reg, rec, "awaiting_approval", needs_review=True)
        n["review"] += 1

    n["excluded"] = len(state.get("exclude_list") or [])

    return n


def _upsert(reg: Any, rec: Dict[str, Any], status: str,
            local_path: str = None, file_size_bytes: int = None,
            needs_review: bool = False) -> None:
    """Map a skill record onto Registry.upsert_dataset(...) + status update."""
    acc = rec.get("accession")
    if not acc:
        return
    notes = rec.get("notes") or ""
    no_pubmed = "no_pubmed_link" in notes
    try:
        reg.upsert_dataset(
            accession=acc,
            source=rec.get("source", "GEO"),
            discovered_by="agent1_pipeline",
            data_type=rec.get("data_type"),
            cancer_type=rec.get("cancer_type"),
            platform=rec.get("platform_canonical") or rec.get("platform"),
            sample_count=rec.get("sample_count"),
            year=rec.get("year"),
            title=rec.get("title"),
            sample_type=rec.get("sample_type"),
            paper_pmid=str((rec.get("pubmed_ids") or [None])[0]) if rec.get("pubmed_ids") else None,
            notes=notes,
            no_pubmed_link=no_pubmed,
            sample_metadata_path=rec.get("sample_metadata_path"),
            usable=rec.get("usable", 1),
            recommended_action=rec.get("recommended_action"),
            reason=rec.get("reason"),
            stage_treatment=rec.get("stage_treatment"),
            available_file_type=rec.get("available_file_type"),
            sample_level_annotation=rec.get("sample_level_annotation"),
            disease_groups=rec.get("disease_groups"),
            needs_review=needs_review,
            download_status=status,
        )
        # For completed downloads, also set local_path/size.
        if local_path and status in ("done", "failed"):
            reg.update_status(acc, status, local_path=str(local_path),
                              file_size_bytes=file_size_bytes)
        reg.log_event(acc, status, f"agent1_pipeline → {status}")
    except Exception as e:
        logger.warning(f"agent1 register: upsert {acc} failed: {e}")


# ---------------------------------------------------------------------- #
#  Runner                                                                 #
# ---------------------------------------------------------------------- #

def run_agent1_pipeline(query: str, config: Dict[str, Any], registry: Any,
                        output_dir: str = None) -> Dict[str, Any]:
    """Compile + invoke the pipeline for one query. Returns the final state."""
    app = build_agent1_pipeline(config, registry)
    initial: Agent1State = {
        "raw_query": query,
        "config": config,
        "registry": registry,
        "output_dir": output_dir or config.get("download", {}).get("output_dir", "./data/methylation"),
        "error_log": [],
    }
    logger.info(f"agent1 pipeline start | query='{query}'")
    final = app.invoke(initial)
    logger.info("agent1 pipeline done.")
    return final
