"""
MethyAgent Daemon — polls the shared SQLite task_queue and runs agents on demand.

Usage:
    # agent1 (DatabaseAgent, always-on):
    python scripts/agent_daemon.py --agent database

    # agent2 (LiteratureAgent, on-demand):
    python scripts/agent_daemon.py --agent literature

Environment variables (from .env / docker-compose):
    REGISTRY_PATH       Path to the SQLite registry file (default: ./registry/methyagent.db)
    DATA_DIR            Directory for downloaded data (default: ./data)
    POLL_INTERVAL       Seconds between task queue polls (default: 5)
    HEARTBEAT_INTERVAL  Seconds between heartbeat writes (default: 30)
    LOG_LEVEL           Logging level: DEBUG | INFO | WARNING (default: INFO)
    OPENAI_API_KEY      Required for LLM extraction (agent2)
    ANTHROPIC_API_KEY   Alternative LLM key (agent2)
    NCBI_API_KEY        NCBI Entrez API key (both agents)
    GEO_EMAIL           Email for NCBI Entrez (both agents)
"""
import argparse
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or inside container (/app)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registry.registry import Registry  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("methyagent.daemon")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
REGISTRY_PATH = os.environ.get("REGISTRY_PATH", "./registry/methyagent.db")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "30"))

# ---------------------------------------------------------------------------
# Load settings.yaml
# ---------------------------------------------------------------------------
_CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    """Load settings.yaml and override download.output_dir with DATA_DIR env var."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    # Let DATA_DIR env var override the download output dir
    if DATA_DIR:
        cfg["download"]["output_dir"] = DATA_DIR
    # Let REGISTRY_PATH env var override the registry db path
    if REGISTRY_PATH:
        cfg["registry"]["db_path"] = REGISTRY_PATH
    return cfg

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _handle_sigterm(signum, frame):
    global _shutdown_requested
    logger.info("SIGTERM received — finishing current task then shutting down.")
    _shutdown_requested = True


def _handle_sigint(signum, frame):
    global _shutdown_requested
    logger.info("SIGINT received — shutting down.")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigint)


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------

def _build_initial_state(query: str, config: dict) -> dict:
    """
    Build a minimal MethyAgentState dict from a raw query string.
    Uses LLM parser first (for accurate cancer_type/platform extraction);
    falls back to rule-based parser if LLM is unavailable.
    """
    # Try LLM parser first — it returns the richer format DatabaseAgent expects
    try:
        from tools.parser_tools import parse_query_with_llm
        from utils.llm_factory import get_llm
        llm = get_llm(config["llm"])
        intent = parse_query_with_llm(query, llm)
        logger.debug(f"LLM parsed intent: {intent}")
    except Exception as e:
        logger.warning(f"LLM query parsing failed ({e}), falling back to rule-based parser")
        from tools.parser_tools import parse_query_rules
        raw_intent = parse_query_rules(query)
        # Normalise rule-based output to match LLM output format
        cancer_code = raw_intent.get("cancer_type_code")
        cancer_display = raw_intent.get("cancer_type_display")
        intent = {
            **raw_intent,
            "cancer_type": {
                "display": cancer_display or "",
                "tcga_code": cancer_code or "",
                "mesh_term": cancer_display or "",
            } if cancer_code else None,
        }

    return {
        "raw_query": query,
        "parsed_intent": intent,
        "db_candidates": [],
        "db_downloaded": [],
        "db_failed": [],
        "db_skipped": [],
        "papers_found": [],
        "lit_candidates": [],
        "lit_downloaded": [],
        "lit_failed": [],
        "lit_skipped": [],
        "messages": [],
        "error_log": [],
        "final_report": {},
        "config": config,
    }


def run_database_agent(query: str, registry: Registry) -> dict:
    """
    Run the database (agent1) path for the given query.
    Returns a summary dict written to task_queue.result_json.

    config.agent1.pipeline selects the implementation:
      "skills" (default) → deterministic skill pipeline (geo-search → geo-filter
                            → geo-download // tcga-direct → register)
      "legacy"            → original DatabaseAgent fixed pipeline (rollback)
    """
    try:
        config = load_config()
        pipeline_mode = (config.get("agent1") or {}).get("pipeline", "skills")

        if pipeline_mode == "skills":
            from agents.agent1_pipeline import run_agent1_pipeline
            logger.info(f"[agent1] Running skill pipeline for query: {query!r}")
            state = run_agent1_pipeline(query, config, registry)

            dl = state.get("download_results") or []
            tcga = state.get("tcga_results") or []
            review = (state.get("lead_list") or []) + (state.get("manual_review_list") or [])
            geo_ok = [r.get("accession") for r in dl if r.get("outcome_final") == "download_success"]
            geo_fail = [r.get("accession") for r in dl if r.get("outcome_final") != "download_success"]
            tcga_ok = [r.get("accession") for r in tcga if r.get("outcome_final") == "download_success"]
            summary = {
                "agent": "database",
                "pipeline": "skills",
                "query": query,
                "datasets_downloaded": geo_ok + tcga_ok,
                "datasets_failed": geo_fail,
                "datasets_review": [r.get("accession") for r in review],
                "datasets_excluded": len(state.get("exclude_list") or []),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            logger.info(
                f"[agent1 pipeline] Done — GEO {len(geo_ok)} ok / {len(geo_fail)} fail, "
                f"TCGA {len(tcga_ok)} ok, {len(review)} review, "
                f"{summary['datasets_excluded']} excluded."
            )
            return summary

        # ---- legacy DatabaseAgent path (rollback) ----
        from agents.database_agent import DatabaseAgent
        agent = DatabaseAgent(config=config, registry=registry)
        state = _build_initial_state(query, config)
        logger.info(f"[agent1] Running legacy DatabaseAgent for query: {query!r}")
        result_state = agent.run(state)
        downloaded = result_state.get("db_downloaded", [])
        failed = result_state.get("db_failed", [])
        skipped = result_state.get("db_skipped", [])
        summary = {
            "agent": "database",
            "pipeline": "legacy",
            "query": query,
            "datasets_found": len(downloaded),
            "datasets_downloaded": downloaded,
            "datasets_failed": failed,
            "datasets_skipped": skipped,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"[agent1] Done — {len(downloaded)} downloaded, "
                    f"{len(failed)} failed, {len(skipped)} skipped.")
        return summary
    except Exception as exc:
        logger.error(f"[agent1] DatabaseAgent failed: {exc}")
        raise


def run_literature_agent(query: str, registry: Registry) -> dict:
    """
    Run LiteratureAgent for the given query.
    Returns a summary dict written to task_queue.result_json.
    """
    try:
        from agents.literature_agent import LiteratureAgent
        config = load_config()
        agent = LiteratureAgent(config=config, registry=registry)
        state = _build_initial_state(query, config)
        logger.info(f"[agent2] Running LiteratureAgent for query: {query!r}")
        result_state = agent.run(state)
        downloaded = result_state.get("lit_downloaded", [])
        failed = result_state.get("lit_failed", [])
        skipped = result_state.get("lit_skipped", [])
        pending = [
            d for d in registry.get_pending_review()
        ]
        summary = {
            "agent": "literature",
            "query": query,
            "papers_searched": len(result_state.get("papers_found", [])),
            "datasets_found": len(downloaded),
            "datasets_downloaded": downloaded,
            "datasets_failed": failed,
            "datasets_skipped": skipped,
            "pending_review": len(pending),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"[agent2] Done — {len(downloaded)} downloaded, "
            f"{len(pending)} pending review."
        )
        return summary
    except Exception as exc:
        logger.error(f"[agent2] LiteratureAgent failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Pending-download execution loop (database agent only)
# ---------------------------------------------------------------------------

def run_pending_downloads(registry: Registry) -> None:
    """
    Download datasets with download_status='pending' (human-approved via Web UI,
    or reset via retry-failed). Called every daemon poll cycle (database agent).

    GEO pending → DownloadSkill (cancer labeling + subset, unified with the
    pipeline path so human-approved downloads get the same per-GSM cancer
    treatment as auto-downloads). TCGA pending → direct download (no subset).
    """
    pending = registry.get_all(status="pending")
    if not pending:
        return

    logger.info(f"[download] Found {len(pending)} pending dataset(s) — starting downloads")
    try:
        config = load_config()
    except Exception as exc:
        logger.error(f"[download] load_config failed: {exc}")
        return

    output_dir = config["download"]["output_dir"]
    geo_pending = [d for d in pending if d.get("source") != "TCGA"]
    tcga_pending = [d for d in pending if d.get("source") == "TCGA"]

    # ---- GEO pending -> DownloadSkill (cancer labeling + subset), dataset-level
    #      concurrency (so back-to-back datasets don't block each other). ----
    if geo_pending:
        try:
            from skills.geo_download import DownloadSkill
            from skills.geo_download.cancer_label import query_terms_from_label
            from agents.agent1_pipeline import _run_concurrent
            skill = DownloadSkill(config)
        except Exception as exc:
            logger.error(f"[download] DownloadSkill init failed: {exc}")
            geo_pending = []

        def _download_one(ds):
            acc = ds["accession"]
            registry.update_status(acc, "downloading")
            # Re-fetch metadata to recover supplementary_files (best-effort).
            try:
                meta = skill.geo_client.get_series_metadata(acc) or {}
            except Exception as exc:
                logger.debug(f"[download] metadata re-fetch {acc} failed: {exc}")
                meta = {}
            rec = {
                "accession": acc,
                "source": "GEO",
                "supplementary_files": meta.get("supplementary_files")
                    or ds.get("supplementary_files") or [],
                "title": ds.get("title") or meta.get("title"),
                "data_type": ds.get("data_type") or meta.get("data_type"),
                "cancer_type": ds.get("cancer_type") or meta.get("cancer_type"),
                "flags": ds.get("notes") or "",
                "available_file_type": ds.get("available_file_type"),
            }
            query_terms = query_terms_from_label(ds.get("cancer_type") or "")
            try:
                result = skill.process_dataset(rec, query_terms, output_dir)
            except Exception as exc:
                logger.error(f"[download] {acc} DownloadSkill failed: {exc}")
                result = {"outcome_final": "failed", "files_downloaded": [],
                          "notes": f"DownloadSkill error: {exc}"}
            _apply_skill_download_result(registry, acc, result)

        _run_concurrent(_download_one, geo_pending, max_concurrent=3)

    # ---- TCGA pending -> direct download (no filter / no subset). ----
    if tcga_pending:
        _download_tcga_pending(registry, tcga_pending, config)


def _apply_skill_download_result(registry: Registry, acc: str, result: dict) -> None:
    """Map a DownloadSkill.process_dataset() result onto the registry."""
    outcome = result.get("outcome_final")
    files = result.get("files_downloaded") or []
    local_path = files[0].get("local_path") if files else None
    size = sum(f.get("size_bytes") or 0 for f in files) or None
    subset = result.get("subset_path")
    extra = (f"; subset={subset}" if subset else "") + (
        f"; {result['notes']}" if result.get("notes") else "")
    if outcome == "download_success":
        registry.update_status(acc, "done", local_path=local_path, file_size_bytes=size)
        registry.log_event(acc, "done", f"DownloadSkill ok{extra}")
        logger.info(f"[download] {acc} done{extra}")
    elif outcome and "manual_review" in outcome:
        # File downloaded but cancer labels unclear - keep the file, flag for review.
        registry.update_status(acc, "done", local_path=local_path, file_size_bytes=size)
        registry.log_event(acc, "review", f"cancer unclear (file kept){extra}")
        logger.info(f"[download] {acc} downloaded, cancer unclear -> review{extra}")
    else:
        registry.update_status(acc, "failed")
        registry.log_event(acc, "error", result.get("notes", "download failed"))
        logger.warning(f"[download] {acc} failed: {result.get('notes', '')}")


def _download_tcga_pending(registry: Registry, tcga_pending: list, config: dict) -> None:
    """TCGA direct download (no subset) for human-approved TCGA datasets."""
    from tools.download_tools import DownloadEngine, build_tcga_download_tasks
    dl_cfg = config["download"]
    try:
        downloader = DownloadEngine(
            output_dir=dl_cfg["output_dir"], max_concurrent=dl_cfg["max_concurrent"],
            retry_attempts=dl_cfg["retry_attempts"], retry_delay=dl_cfg["retry_delay"],
            chunk_size_mb=dl_cfg["chunk_size_mb"], timeout=dl_cfg["timeout"],
        )
    except Exception as exc:
        logger.error(f"[download] TCGA DownloadEngine init failed: {exc}")
        return
    tasks = []
    for ds in tcga_pending:
        acc = ds["accession"]
        registry.update_status(acc, "downloading")
        try:
            tasks.extend(build_tcga_download_tasks(
                ds, dl_cfg["output_dir"], config["tcga"]["gdc_api_base"]))
        except Exception as exc:
            logger.error(f"[download] TCGA build tasks {acc}: {exc}")
            registry.update_status(acc, "failed")
    if not tasks:
        return
    try:
        results = downloader.download_many_sync(tasks)
    except Exception as exc:
        logger.error(f"[download] TCGA download_many_sync: {exc}")
        for ds in tcga_pending:
            registry.update_status(ds["accession"], "failed")
        return
    acc_results: dict = {}
    for r in results:
        acc_results.setdefault(r["accession"], []).append(r)
    for acc, acc_res in acc_results.items():
        if all(r["status"] == "done" for r in acc_res):
            size = sum(r.get("file_size_bytes", 0) for r in acc_res)
            registry.update_status(acc, "done",
                                    local_path=str(acc_res[0]["local_path"]), file_size_bytes=size)
            registry.log_event(acc, "done", f"TCGA downloaded {len(acc_res)} file(s)")
        else:
            registry.update_status(acc, "failed")
            registry.log_event(acc, "error", "; ".join(
                r.get("error", "") for r in acc_res if r["status"] != "done"))


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------

def daemon_loop(agent_type: str, registry: Registry):
    """
    Poll the task_queue for pending tasks matching agent_type.
    Runs indefinitely until SIGTERM/SIGINT.
    """
    runner = run_database_agent if agent_type == "database" else run_literature_agent
    last_heartbeat = 0.0

    logger.info(
        f"MethyAgent daemon started — agent_type={agent_type!r}, "
        f"poll_interval={POLL_INTERVAL}s, heartbeat_interval={HEARTBEAT_INTERVAL}s"
    )

    while not _shutdown_requested:
        # ---- Heartbeat ----
        now = time.monotonic()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            try:
                registry.update_heartbeat(agent_type)
                logger.debug(f"Heartbeat written for {agent_type!r}")
            except Exception as hb_exc:
                logger.warning(f"Heartbeat write failed: {hb_exc}")
            last_heartbeat = now

        # ---- Claim a task ----
        try:
            task = registry.claim_task(agent_type)
        except Exception as claim_exc:
            logger.error(f"Error claiming task: {claim_exc}")
            time.sleep(POLL_INTERVAL)
            continue

        if task is None:
            # No pending tasks — check for approved downloads, then sleep
            if agent_type == "database":
                try:
                    run_pending_downloads(registry)
                except Exception as dl_exc:
                    logger.error(f"[download] Error in idle download poll: {dl_exc}")
            time.sleep(POLL_INTERVAL)
            continue

        task_id = task["task_id"]
        query = task["query"]
        logger.info(f"Claimed task {task_id} — query={query!r}")

        # ---- Execute ----
        try:
            result = runner(query, registry)
            registry.complete_task(task_id, result=result)
            logger.info(f"Task {task_id} completed successfully.")
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            registry.fail_task(task_id, error=error_msg)
            logger.error(f"Task {task_id} failed: {exc}")

        # ---- Download pending datasets (database agent only) ----
        # Picks up datasets approved via Web UI or reset via retry-failed.
        if agent_type == "database":
            try:
                run_pending_downloads(registry)
            except Exception as dl_exc:
                logger.error(f"[download] Unexpected error in run_pending_downloads: {dl_exc}")

    logger.info(f"Daemon {agent_type!r} shut down cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Declare globals first, before any reference to them
    global DATA_DIR, POLL_INTERVAL

    parser = argparse.ArgumentParser(
        description="MethyAgent daemon — polls task_queue and runs agents."
    )
    parser.add_argument(
        "--agent",
        choices=["database", "literature"],
        required=True,
        help="Which agent to run: 'database' (agent1) or 'literature' (agent2).",
    )
    parser.add_argument(
        "--registry",
        default=REGISTRY_PATH,
        help=f"Path to SQLite registry file (default: {REGISTRY_PATH}).",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=f"Directory for downloaded data (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL,
        help=f"Seconds between task queue polls (default: {POLL_INTERVAL}).",
    )
    args = parser.parse_args()

    # Override globals from CLI args
    DATA_DIR = args.data_dir
    POLL_INTERVAL = args.poll_interval

    # Ensure data directory exists
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    registry = Registry(db_path=args.registry)
    daemon_loop(agent_type=args.agent, registry=registry)


if __name__ == "__main__":
    main()
