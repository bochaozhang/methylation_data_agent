"""
Pydantic models for MethyAgent Web API request/response validation.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """POST /query — submit a new methylation search query."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Search query, e.g. 'LUAD DNA methylation 450k'",
        examples=["lung adenocarcinoma methylation TCGA"],
    )
    agent_type: str = Field(
        default="both",
        description="Which agent(s) to run: 'database', 'literature', or 'both'",
        pattern="^(database|literature|both)$",
    )


class ReviewDecision(BaseModel):
    """POST /review/{accession}/approve or /reject — optional note."""
    note: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional reviewer note stored in the log.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class TaskResponse(BaseModel):
    """Returned when a task is created or queried."""
    task_id: str
    agent_type: str
    query: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """GET /tasks — list of tasks."""
    tasks: List[TaskResponse]
    total: int


class DatasetResponse(BaseModel):
    """Single dataset record from the registry."""
    accession: str
    source: str
    data_type: Optional[str] = None
    cancer_type: Optional[str] = None
    platform: Optional[str] = None
    sample_count: Optional[int] = None
    year: Optional[int] = None
    title: Optional[str] = None
    download_status: str
    local_path: Optional[str] = None
    paper_pmid: Optional[str] = None
    paper_doi: Optional[str] = None
    discovered_by: str
    file_size_bytes: Optional[int] = None
    needs_review: bool = False
    llm_evidence: Optional[str] = None
    sample_type: Optional[str] = None
    no_pubmed_link: bool = False
    created_at: str
    updated_at: str


class DatasetListResponse(BaseModel):
    """GET /datasets — list of datasets."""
    datasets: List[DatasetResponse]
    total: int


class ReviewItemResponse(BaseModel):
    """Single pending-review item with LLM evidence."""
    accession: str
    source: str
    title: Optional[str] = None
    paper_doi: Optional[str] = None
    paper_pmid: Optional[str] = None
    llm_evidence: Optional[str] = None
    created_at: str


class ReviewListResponse(BaseModel):
    """GET /review — list of pending-review datasets."""
    items: List[ReviewItemResponse]
    total: int


class ReviewActionResponse(BaseModel):
    """POST /review/{accession}/approve or /reject."""
    accession: str
    action: str   # 'approved' | 'rejected'
    message: str


class HealthResponse(BaseModel):
    """GET /health — agent heartbeat status."""
    status: str   # 'ok' | 'degraded'
    agent1_heartbeat: Optional[str] = None   # ISO timestamp or None
    agent2_heartbeat: Optional[str] = None
    agent1_alive: bool = False
    agent2_alive: bool = False
    registry_datasets: int = 0
    registry_pending_tasks: int = 0


class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str


class CancelTaskResponse(BaseModel):
    """POST /tasks/{task_id}/cancel — cancel a single task."""
    task_id: str
    cancelled: bool
    message: str


class CancelAllResponse(BaseModel):
    """POST /tasks/cancel-all — cancel all pending tasks and downloads."""
    tasks_cancelled: int
    datasets_skipped: int
    message: str


class RetryFailedResponse(BaseModel):
    """POST /datasets/retry-failed — reset failed downloads to pending."""
    datasets_reset: int
    message: str


class ApprovalItem(BaseModel):
    """Single dataset awaiting human download approval."""
    accession: str
    source: str
    title: Optional[str] = None
    cancer_type: Optional[str] = None
    platform: Optional[str] = None
    sample_count: Optional[int] = None
    year: Optional[int] = None
    sample_type: Optional[str] = None
    paper_pmid: Optional[str] = None
    no_pubmed_link: bool = False
    notes: Optional[str] = None
    sample_metadata_path: Optional[str] = None
    created_at: str


class ApprovalListResponse(BaseModel):
    """GET /datasets/awaiting-approval — list of datasets pending human confirmation."""
    items: List[ApprovalItem]
    total: int


class ApprovalRequest(BaseModel):
    """POST /datasets/approve — accessions the user has chosen to download."""
    approved: List[str] = Field(
        default=[],
        description="List of accession IDs to approve for download. "
                    "All other awaiting_approval datasets will be skipped.",
    )


class ApprovalResponse(BaseModel):
    """POST /datasets/approve — result of the approval action."""
    approved_count: int
    skipped_count: int
    message: str
