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
    Scan the registry for datasets with download_status='pending' and
    execute their downloads.  Called at the end of every daemon poll cycle
    (database agent only).

    This handles two cases:
      1. Datasets approved by the user via POST /datasets/approve
      2. Failed datasets reset to 'pending' via POST /datasets/retry-failed
    """
    pending = registry.get_all(status="pending")
    if not pending:
        return

    logger.info(f"[download] Found {len(pending)} pending dataset(s) — starting downloads")

    try:
        from tools.download_tools import DownloadEngine, build_geo_download_tasks, build_tcga_download_tasks
        config = load_config()
        dl_cfg = config["download"]
        downloader = DownloadEngine(
            output_dir=dl_cfg["output_dir"],
            max_concurrent=dl_cfg["max_concurrent"],
            retry_attempts=dl_cfg["retry_attempts"],
            retry_delay=dl_cfg["retry_delay"],
            chunk_size_mb=dl_cfg["chunk_size_mb"],
            timeout=dl_cfg["timeout"],
        )
    except Exception as exc:
        logger.error(f"[download] Failed to initialise DownloadEngine: {exc}")
        return

    # Build download tasks for all pending datasets
    download_tasks = []
    for ds in pending:
        acc = ds["accession"]
        registry.update_status(acc, "downloading")
        try:
            if ds.get("source") == "TCGA":
                tasks = build_tcga_download_tasks(
                    ds,
                    config["download"]["output_dir"],
                    config["tcga"]["gdc_api_base"],
                )
            else:
                tasks = build_geo_download_tasks(ds, config["download"]["output_dir"])
            download_tasks.extend(tasks)
        except Exception as exc:
            logger.error(f"[download] Failed to build tasks for {acc}: {exc}")
            registry.update_status(acc, "failed")
            registry.log_event(acc, "error", f"Task build failed: {exc}")

    if not download_tasks:
        return

    # Execute downloads
    try:
        results = downloader.download_many_sync(download_tasks)
    except Exception as exc:
        logger.error(f"[download] download_many_sync raised: {exc}")
        for ds in pending:
            registry.update_status(ds["accession"], "failed")
            registry.log_event(ds["accession"], "error", f"Download engine error: {exc}")
        return

    # Aggregate results by accession
    acc_results: dict = {}
    for r in results:
        acc_results.setdefault(r["accession"], []).append(r)

    for acc, acc_res in acc_results.items():
        all_done = all(r["status"] == "done" for r in acc_res)
        if all_done:
            local_path = acc_res[0]["local_path"]
            file_size = sum(r.get("file_size_bytes", 0) for r in acc_res)
            registry.update_status(acc, "done", local_path=str(local_path), file_size_bytes=file_size)
            registry.log_event(acc, "done", f"Downloaded {len(acc_res)} file(s)")
            logger.info(f"[download] {acc} — done ({file_size} bytes)")
        else:
            error_msgs = [r.get("error", "") for r in acc_res if r["status"] == "failed"]
            registry.update_status(acc, "failed")
            registry.log_event(acc, "error", "; ".join(error_msgs))
            logger.warning(f"[download] {acc} — failed: {error_msgs}")


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
