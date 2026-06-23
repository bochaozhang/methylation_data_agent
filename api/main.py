"""
MethyAgent Web API — FastAPI application.

Endpoints:
    GET  /                              HTML dashboard
    POST /query                         Submit a new query → task_id
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
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registry.registry import Registry  # noqa: E402
from api.models import (  # noqa: E402
    QueryRequest,
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
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGISTRY_PATH = os.environ.get("REGISTRY_PATH", "./registry/methyagent.db")
HEARTBEAT_STALE_SECS = int(os.environ.get("HEARTBEAT_STALE_SECS", "120"))

# ---------------------------------------------------------------------------
# App + templates
# ---------------------------------------------------------------------------
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


def _task_to_response(task: dict) -> TaskResponse:
    return TaskResponse(
        task_id=task["task_id"],
        agent_type=task["agent_type"],
        query=task["query"],
        status=task["status"],
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
        paper_pmid=d.get("paper_pmid"),
        paper_doi=d.get("paper_doi"),
        discovered_by=d["discovered_by"],
        file_size_bytes=d.get("file_size_bytes"),
        needs_review=bool(d.get("needs_review", 0)),
        llm_evidence=d.get("llm_evidence"),
        sample_type=d.get("sample_type"),
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


@app.post("/query", response_model=TaskResponse, status_code=201)
async def submit_query(body: QueryRequest):
    """
    Submit a new methylation search query.

    Creates one or two tasks in the queue (one per agent if agent_type='both').
    Returns the primary task record.
    """
    registry = get_registry()
    task_id = registry.create_task(query=body.query, agent_type=body.agent_type)
    task = registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=500, detail="Task creation failed.")
    return _task_to_response(task)


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


@app.get("/datasets/{accession}", response_model=DatasetResponse)
async def get_dataset(accession: str):
    """Get a single dataset record by accession ID."""
    registry = get_registry()
    d = registry.get(accession)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{accession}' not found.")
    return _dataset_to_response(d)


@app.get("/review", response_model=ReviewListResponse)
async def list_pending_review():
    """List all datasets flagged for human review (medium-confidence LLM extractions)."""
    registry = get_registry()
    items = registry.get_pending_review()
    return ReviewListResponse(
        items=[
            ReviewItemResponse(
                accession=d["accession"],
                source=d["source"],
                title=d.get("title"),
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
