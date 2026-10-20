"""
MethyAgent Web API — FastAPI application.

Endpoints:
    GET  /                              HTML dashboard
    POST /query                         Submit a new query → task_id
    POST /search/dry-run                Recall preview: parse + geo-search only
    GET  /tasks                         List all tasks
    GET  /tasks/{task_id}               Single task status + result
    GET  /datasets                      All datasets in registry
    GET  /datasets/{accession}          Single dataset details
    GET  /review                        List pending_review datasets
    POST /review/{accession}/approve    Approve a pending dataset
    POST /review/{accession}/reject     Reject a pending dataset
    GET  /health                        Agent heartbeat + registry stats

Environment variables:
    REGISTRY_PATH   Path to SQLite registry (default: ./registry/methyagent.db)
    HEARTBEAT_STALE_SECS  Seconds after which a heartbeat is considered stale (default: 120)
"""
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request, Query, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registry.registry import Registry  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from api.models import (  # noqa: E402
    QueryRequest,
    DryRunRequest,
    DryRunResponse,
    DryRunCandidate,
    ReviewDecision,
    TaskResponse,
    TaskListResponse,
    DatasetResponse,
    DatasetListResponse,
    ReviewItemResponse,
    ReviewListResponse,
    ReviewActionResponse,
    HealthResponse,
    CancelTaskResponse,
    CancelAllResponse,
    RetryFailedResponse,
    ApprovalItem,
    ApprovalListResponse,
    ApprovalRequest,
    ApprovalResponse,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGISTRY_PATH = os.environ.get("REGISTRY_PATH", "./registry/methyagent.db")
HEARTBEAT_STALE_SECS = int(os.environ.get("HEARTBEAT_STALE_SECS", "120"))

# ---------------------------------------------------------------------------
# App + templates
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

app = FastAPI(
    title="MethyAgent API",
    description="Dual-agent methylation dataset acquisition system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Registry singleton (shared across requests)
# ---------------------------------------------------------------------------
_registry: Optional[Registry] = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry(db_path=REGISTRY_PATH)
    return _registry


# ---------------------------------------------------------------------------
# Dry-run singletons (settings + json LLM + SearchSkill, lazily built)
# ---------------------------------------------------------------------------
_settings: Optional[Dict[str, Any]] = None
_json_llm: Any = None
_search_skill: Any = None


def _get_settings() -> Dict[str, Any]:
    """Load config/settings.yaml once (SETTINGS_PATH env overrides)."""
    global _settings
    if _settings is None:
        path = os.environ.get("SETTINGS_PATH", str(REPO_ROOT / "config" / "settings.yaml"))
        with open(path, "r", encoding="utf-8") as f:
            _settings = yaml.safe_load(f) or {}
    return _settings


def _get_search_skill() -> Any:
    """Build the SearchSkill (and its json-mode LLM) once, on first dry-run."""
    global _json_llm, _search_skill
    if _search_skill is None:
        from skills.geo_search import SearchSkill
        from utils.llm_factory import get_llm

        cfg = _get_settings()
        _json_llm = get_llm(cfg["llm"], json_mode=True)
        _search_skill = SearchSkill(cfg)
    return _search_skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_heartbeat_alive(ts: Optional[str]) -> bool:
    """Return True if the heartbeat timestamp is within HEARTBEAT_STALE_SECS."""
    if ts is None:
        return False
    try:
        hb_time = datetime.fromisoformat(ts)
        if hb_time.tzinfo is None:
            hb_time = hb_time.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - hb_time
        return age < timedelta(seconds=HEARTBEAT_STALE_SECS)
    except Exception:
        return False


def _task_to_response(task: dict, deduplicated: bool = False) -> TaskResponse:
    intent = None
    if task.get("intent_json"):
        try:
            intent = json.loads(task["intent_json"])
        except Exception:
            intent = None
    return TaskResponse(
        task_id=task["task_id"],
        agent_type=task["agent_type"],
        query=task["query"],
        status=task["status"],
        label=task.get("label"),
        deduplicated=deduplicated,
        intent=intent,
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        finished_at=task.get("finished_at"),
        result=task.get("result"),
        error=task.get("error"),
    )


def _dataset_to_response(d: dict) -> DatasetResponse:
    return DatasetResponse(
        accession=d["accession"],
        source=d["source"],
        data_type=d.get("data_type"),
        cancer_type=d.get("cancer_type"),
        platform=d.get("platform"),
        sample_count=d.get("sample_count"),
        year=d.get("year"),
        title=d.get("title"),
        download_status=d["download_status"],
        local_path=d.get("local_path"),
        sample_metadata_path=d.get("sample_metadata_path"),
        paper_pmid=d.get("paper_pmid"),
        paper_doi=d.get("paper_doi"),
        discovered_by=d["discovered_by"],
        file_size_bytes=d.get("file_size_bytes"),
        needs_review=bool(d.get("needs_review", 0)),
        llm_evidence=d.get("llm_evidence"),
        sample_type=d.get("sample_type"),
        disease_groups=d.get("disease_groups"),
        sample_level_annotation=d.get("sample_level_annotation"),
        stage_treatment=d.get("stage_treatment"),
        recommended_action=d.get("recommended_action"),
        reason=d.get("reason"),
        notes=d.get("notes"),
        available_file_type=d.get("available_file_type"),
        no_pubmed_link=bool(d.get("no_pubmed_link", 0)),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    """Serve the single-page HTML dashboard."""
    try:
        # Starlette >= 0.36 / Jinja2 >= 3.1.4: pass request as keyword arg
        return templates.TemplateResponse(request=request, name="index.html")
    except TypeError:
        # Fallback for older versions
        return templates.TemplateResponse("index.html", {"request": request})


def _intent_summary(intent: Dict[str, Any]) -> str:
    """Human-readable one-liner for a structured intent (stored as task.query)."""
    parts = []
    ct = intent.get("cancer_type") or {}
    if ct.get("display"):
        parts.append(str(ct["display"]))
    sample_types = intent.get("sample_types") or []
    if sample_types:
        parts.append("/".join(sample_types))
    if intent.get("platform"):
        parts.append(str(intent["platform"]))
    if intent.get("mode") == "accession":
        geo = (intent.get("accessions") or {}).get("geo") or []
        if geo:
            parts.append(",".join(geo[:5]))
    return "structured: " + (" ".join(parts) if parts else "unspecified")


def _intent_key(intent: Dict[str, Any]) -> str:
    """Canonical serialization for intent equivalence (dedup)."""
    import json as _json
    return _json.dumps(intent, sort_keys=True, ensure_ascii=False, default=str)


@app.post("/query", response_model=TaskResponse, status_code=201)
async def submit_query(body: QueryRequest, response: Response):
    """
    Submit a new methylation search query.

    Free text (`query`, LLM-parsed) or structured (`intent`, deterministic —
    no LLM parse; `query` optional then, a readable summary is generated).
    `label` tags sub-task roles for compound queries ('main'/'auxiliary').

    Dedup: unless `force`, an equivalent pending/running task (same agent_type;
    identical canonical intent, or identical free-text query) is returned
    instead of creating a duplicate (deduplicated=True, HTTP 200).
    """
    from state.agent1_state import normalize_intent

    registry = get_registry()

    # ---- resolve query + optional structured intent -----------------------
    intent_json: Optional[str] = None
    intent: Optional[Dict[str, Any]] = None
    if body.intent is not None:
        parsed = dict(body.intent)
        parsed.setdefault("parse_method", "structured")
        intent = normalize_intent(body.query or "", parsed)
        intent_json = json.dumps(intent, ensure_ascii=False)
        query = body.query or _intent_summary(intent)
    elif body.query:
        query = body.query.strip()
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide `query` (free text) or `intent` (structured), not neither.",
        )

    # ---- submit-side dedup: reuse an equivalent active task ----------------
    if not body.force:
        for t in registry.list_active_tasks(agent_type=body.agent_type):
            if intent_json is not None:
                if not t.get("intent_json"):
                    continue
                try:
                    other = json.loads(t["intent_json"])
                except Exception:
                    continue
                if _intent_key(other) == _intent_key(intent):
                    existing = registry.get_task(t["task_id"])
                    response.status_code = 200
                    return _task_to_response(existing, deduplicated=True)
            elif not t.get("intent_json") and t.get("query", "").strip() == query:
                existing = registry.get_task(t["task_id"])
                response.status_code = 200
                return _task_to_response(existing, deduplicated=True)

    task_id = registry.create_task(
        query=query,
        agent_type=body.agent_type,
        label=body.label,
        intent_json=intent_json,
    )
    task = registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=500, detail="Task creation failed.")
    return _task_to_response(task)


@app.post("/search/dry-run", response_model=DryRunResponse)
def search_dry_run(body: DryRunRequest):
    """
    Recall preview: run the parse + geo-search stages only.

    No LLM filter verdicts, no registry writes, no downloads — a cheap, safe
    way for callers (MCP / external agents) to see what a full query would
    recall before committing to a ~55 min task. Give `query` (free text,
    LLM-parsed) or `intent` (structured, deterministic — no LLM parse).

    Sync handler on purpose: runs in the FastAPI threadpool so a slow NCBI
    search blocks one worker thread, not the event loop.
    """
    if body.intent is None and not body.query:
        raise HTTPException(
            status_code=422,
            detail="Provide `query` (free text) or `intent` (structured), not neither.",
        )

    started = time.time()
    from state.agent1_state import normalize_intent

    # ---- parse (mirrors agent1 parse_node; QueryLogger omitted — no writes) --
    if body.intent is not None:
        parsed = dict(body.intent)
        parsed.setdefault("parse_method", "structured")
        intent = normalize_intent(body.query or "", parsed)
    else:
        from tools.parser_tools import parse_query_rules, parse_query_with_llm

        _get_search_skill()  # initializes _json_llm
        try:
            parsed = parse_query_with_llm(body.query, _json_llm)
        except Exception as e:
            logger.warning(f"dry-run parse: LLM failed ({e}), rules fallback")
            parsed = parse_query_rules(body.query)
        intent = normalize_intent(body.query, parsed)

    # ---- search (SearchSkill.run is deterministic and read-only) ------------
    out = _get_search_skill().run({"parsed_intent": intent})
    candidates = out.get("candidate_gse_list") or []
    keep = candidates[: body.max_candidates]

    return DryRunResponse(
        parse_method=intent.get("parse_method", "rules_fallback"),
        intent=intent,
        search_queries=out.get("search_queries") or [],
        total_candidates=len(candidates),
        returned=len(keep),
        truncated=len(candidates) > len(keep),
        candidates=[
            DryRunCandidate(
                accession=c.get("accession", "?"),
                title=c.get("title"),
                cancer_type=c.get("cancer_type"),
                platform=c.get("platform") or c.get("platform_canonical"),
                data_type=c.get("data_type"),
                sample_count=c.get("sample_count"),
                year=c.get("year"),
                pubmed_ids=list(c.get("pubmed_ids") or []),
            )
            for c in keep
        ],
        elapsed_seconds=round(time.time() - started, 2),
    )


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: pending | running | done | failed | cancelled",
    ),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List tasks from the queue, newest first."""
    registry = get_registry()
    tasks = registry.list_tasks(status=status, limit=limit)
    return TaskListResponse(
        tasks=[_task_to_response(t) for t in tasks],
        total=len(tasks),
    )


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get a single task by ID, including its result when done."""
    registry = get_registry()
    task = registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return _task_to_response(task)


@app.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    status: Optional[str] = Query(
        default=None,
        description="Filter by download_status: pending | downloading | done | failed | skipped",
    ),
    limit: int = Query(default=200, ge=1, le=2000),
):
    """List all datasets in the registry."""
    registry = get_registry()
    datasets = registry.get_all(status=status)[:limit]
    return DatasetListResponse(
        datasets=[_dataset_to_response(d) for d in datasets],
        total=len(datasets),
    )


@app.get("/datasets/awaiting-approval", response_model=ApprovalListResponse)
async def list_awaiting_approval():
    """
    Return all datasets discovered by the agent that are waiting for
    human confirmation before downloading.
    """
    registry = get_registry()
    items = registry.get_awaiting_approval()
    return ApprovalListResponse(
        items=[
            ApprovalItem(
                accession=d["accession"],
                source=d["source"],
                title=d.get("title"),
                cancer_type=d.get("cancer_type"),
                platform=d.get("platform"),
                sample_count=d.get("sample_count"),
                year=d.get("year"),
                sample_type=d.get("sample_type"),
                paper_pmid=d.get("paper_pmid"),
                no_pubmed_link=bool(d.get("no_pubmed_link", 0)),
                notes=d.get("notes"),
                sample_metadata_path=d.get("sample_metadata_path"),
                needs_review=bool(d.get("needs_review", 0)),
                recommended_action=d.get("recommended_action"),
                reason=d.get("reason"),
                created_at=d["created_at"],
            )
            for d in items
        ],
        total=len(items),
    )


@app.post("/datasets/retry-failed", response_model=RetryFailedResponse)
async def retry_failed_datasets():
    """
    Reset all failed dataset downloads back to pending.

    The agent daemon will pick them up on the next poll cycle
    (default every 5 seconds) and attempt to re-download.
    """
    registry = get_registry()
    n = registry.retry_failed_datasets()
    return RetryFailedResponse(
        datasets_reset=n,
        message=f"Reset {n} failed dataset(s) to pending. Agent will retry on next poll.",
    )


@app.get("/datasets/{accession}", response_model=DatasetResponse)
async def get_dataset(accession: str):
    """Get a single dataset record by accession ID."""
    registry = get_registry()
    d = registry.get(accession)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{accession}' not found.")
    return _dataset_to_response(d)


@app.get("/query-datasets")
async def list_query_datasets():
    """List distinct queries with dataset counts (master view)."""
    registry = get_registry()
    items = registry.get_query_list()
    return {"items": items, "total": len(items)}


@app.get("/query-datasets/{task_id}")
async def get_query_detail(task_id: str):
    """Get all datasets for a specific task_id (detail view)."""
    registry = get_registry()
    items = registry.get_datasets_by_task_id(task_id)
    return {"task_id": task_id, "items": items, "total": len(items)}


@app.get("/query-datasets/{task_id}/samples")
async def get_query_samples(task_id: str):
    """
    GSM-level view for a task: one row per (accession, gsm) with its per-query
    verdict ("download" / "not download") and GSM metadata. This is the sample-grain
    companion to /query-datasets/{task_id} (which is GSE-level).
    """
    registry = get_registry()
    items = registry.get_samples_by_task_id(task_id)
    counts = registry.get_gsm_counts_by_task(task_id)
    return {"task_id": task_id, "items": items, "counts": counts, "total": len(items)}


@app.get("/review", response_model=ReviewListResponse)
async def list_pending_review():
    """List all datasets flagged for human review (medium-confidence LLM extractions)."""
    registry = get_registry()
    items = registry.get_pending_review()
    return ReviewListResponse(
        items=[
            ReviewItemResponse(
                accession=d["accession"],
                source=d.get("source", ""),
                title=d.get("title"),
                cancer_type=d.get("cancer_type"),
                platform=d.get("platform"),
                sample_count=d.get("sample_count"),
                sample_type=d.get("sample_type"),
                outcome=d.get("recommended_action"),
                reason=d.get("reason"),
                paper_doi=d.get("paper_doi"),
                paper_pmid=d.get("paper_pmid"),
                llm_evidence=d.get("llm_evidence"),
                created_at=d["created_at"],
            )
            for d in items
        ],
        total=len(items),
    )


@app.post("/review/{accession}/approve", response_model=ReviewActionResponse)
async def approve_dataset(accession: str, body: ReviewDecision = ReviewDecision()):
    """
    Approve a pending_review dataset.
    Clears the needs_review flag and sets download_status back to 'pending'
    so the database agent will pick it up on the next poll.
    """
    registry = get_registry()
    d = registry.get(accession)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{accession}' not found.")
    if not d.get("needs_review"):
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{accession}' is not pending review (needs_review=0).",
        )
    registry.approve_review(accession)
    if body.note:
        registry.log_event(accession, "review_approved", message=body.note)
    else:
        registry.log_event(accession, "review_approved", message="Approved via Web UI")
    return ReviewActionResponse(
        accession=accession,
        action="approved",
        message=f"Dataset '{accession}' approved and queued for download.",
    )


@app.post("/review/{accession}/reject", response_model=ReviewActionResponse)
async def reject_dataset(accession: str, body: ReviewDecision = ReviewDecision()):
    """
    Reject a pending_review dataset.
    Sets download_status to 'skipped' so it won't be downloaded.
    """
    registry = get_registry()
    d = registry.get(accession)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{accession}' not found.")
    if not d.get("needs_review"):
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{accession}' is not pending review (needs_review=0).",
        )
    registry.reject_review(accession)
    if body.note:
        registry.log_event(accession, "review_rejected", message=body.note)
    else:
        registry.log_event(accession, "review_rejected", message="Rejected via Web UI")
    return ReviewActionResponse(
        accession=accession,
        action="rejected",
        message=f"Dataset '{accession}' rejected and marked as skipped.",
    )


# ---------------------------------------------------------------------------
# Task cancellation
# ---------------------------------------------------------------------------

@app.post("/tasks/cancel-all", response_model=CancelAllResponse)
async def cancel_all_tasks():
    """
    Cancel all pending/running tasks and skip all pending dataset downloads.

    Useful for stopping a runaway search without touching the database manually.
    Note: tasks already in progress (status=running) are marked cancelled in the
    registry; the agent daemon will stop picking up new work immediately, but any
    in-flight HTTP request may still complete before the agent checks the flag.
    """
    registry = get_registry()
    result = registry.cancel_all_tasks()
    return CancelAllResponse(
        tasks_cancelled=result["tasks_cancelled"],
        datasets_skipped=result["datasets_skipped"],
        message=(
            f"Cancelled {result['tasks_cancelled']} task(s) and "
            f"skipped {result['datasets_skipped']} pending download(s)."
        ),
    )


@app.post("/tasks/{task_id}/cancel", response_model=CancelTaskResponse)
async def cancel_task(task_id: str):
    """
    Cancel a single pending task by ID.

    Only tasks with status='pending' can be cancelled this way.
    Running tasks are not interrupted mid-execution.
    """
    registry = get_registry()
    task = registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    if task["status"] not in ("pending", "running"):
        return CancelTaskResponse(
            task_id=task_id,
            cancelled=False,
            message=f"Task is already in status '{task['status']}', cannot cancel.",
        )
    cancelled = registry.cancel_task(task_id)
    return CancelTaskResponse(
        task_id=task_id,
        cancelled=cancelled,
        message="Task cancelled." if cancelled else "Task could not be cancelled (may have just started running).",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Return agent heartbeat status and registry statistics.

    agent1/agent2 are considered 'alive' if their last heartbeat is within
    HEARTBEAT_STALE_SECS (default 120s).
    """
    registry = get_registry()

    hb1 = registry.get_heartbeat("database")
    hb2 = registry.get_heartbeat("literature")
    alive1 = _is_heartbeat_alive(hb1)
    alive2 = _is_heartbeat_alive(hb2)

    summary = registry.get_summary()
    pending_tasks = len(registry.list_tasks(status="pending"))

    overall = "ok" if (alive1 and alive2) else "degraded"

    return HealthResponse(
        status=overall,
        agent1_heartbeat=hb1,
        agent2_heartbeat=hb2,
        agent1_alive=alive1,
        agent2_alive=alive2,
        registry_datasets=summary.get("total", 0),
        registry_pending_tasks=pending_tasks,
    )
