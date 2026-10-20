"""
MethyAgent MCP server — read-only + submission tools over stdio.

Every tool proxies the MethyAgent REST API (the webui container) and
translates container-side file paths (/app/data/**) to host-side paths before
returning, so consumers can read the files directly.

Tools:
    search_datasets         dry-run recall preview (parse + GEO search, no
                            filter, no writes — seconds, near-zero cost)
    submit_query            submit a pipeline task (free text or structured
                            intent; label for compound-query roles)
    submit_compound_query   main + auxiliary sub-tasks in one call (e.g.
                            training vs validation data) — two labeled tasks
    get_task_status         one pipeline task by ID
    get_task_result         finished-task verdict buckets + file paths
    cancel_task             cancel a pending task
    list_datasets           browse the registry (optionally by download_status)
    get_dataset             full record for one accession, incl. file paths
    health                  backend heartbeat + registry stats

Compound-query semantics: the task boundary IS the sub-task boundary. The two
sub-tasks get independent search strings and filter judgments; the registry
tracks per-task dataset/sample membership (a GSE wanted by both is downloaded
once and attributed per sample). Role semantics (training/validation) stay
with the caller — labels are just tags.

Plan: docs/mcp_interface_plan.md.

Run (stdio, any cwd):
    .venv/bin/python /abs/path/to/mcp_server/__main__.py
Register with Claude Code:
    claude mcp add methyagent -- <venv>/bin/python <repo>/mcp_server/__main__.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Bootstrap: allow running as a plain script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from mcp_server.api_client import ApiClient  # noqa: E402
from mcp_server.paths import translate_record, translate_path  # noqa: E402

server = MCPServer(
    name="methyagent",
    title="MethyAgent",
    description=(
        "DNA methylation dataset retrieval agent: GEO/TCGA search, LLM-based "
        "filtering against SPEC criteria, download registry."
    ),
    instructions=(
        "Typical flow: (1) search_datasets to preview what a query recalls "
        "(cheap, no side effects); (2) submit_query or submit_compound_query "
        "— a full pipeline task takes ~1 hour (search → per-dataset LLM "
        "filter → register → async download); (3) poll get_task_status; "
        "(4) get_task_result for the verdict buckets and file paths. "
        "Duplicate submissions of an equivalent active task are deduplicated "
        "server-side (deduplicated=true means an existing task was returned). "
        "Paths returned are host-side; read data files directly from disk. "
        "download_status: pending | downloading | done | failed | skipped."
    ),
)

_client: Optional[ApiClient] = None


def _api() -> ApiClient:
    global _client
    if _client is None:
        _client = ApiClient()
    return _client


# Registry fields worth forwarding in LIST views (drop bulky ones such as
# llm_evidence — get_dataset returns the full record instead).
_LIST_FIELDS = (
    "accession", "source", "title", "cancer_type", "platform", "data_type",
    "sample_count", "year", "download_status", "local_path",
    "sample_metadata_path", "file_size_bytes", "sample_type",
    "paper_pmid", "paper_doi", "updated_at",
    # eligibility inputs — downstream platforms parse control counts from
    # disease_groups (COMPARATOR-NOT-EVIDENCED gates without it)
    "disease_groups", "sample_level_annotation",
)


def _compact_dataset(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the list-worthy fields of a dataset record, translate paths."""
    return translate_record({k: d.get(k) for k in _LIST_FIELDS if d.get(k) is not None})


# --------------------------------------------------------------------------- #
#  Tools                                                                      #
# --------------------------------------------------------------------------- #

@server.tool()
def search_datasets(query: str = "", intent: Optional[Dict[str, Any]] = None,
                    max_candidates: int = 50) -> Dict[str, Any]:
    """Dry-run recall preview: parse the query + search GEO. Read-only.

    No LLM filter verdicts, no registry writes, no downloads — takes seconds.
    Use this to explore what a full task would recall before committing to a
    ~1 hour pipeline run. Returns the normalized intent, the NCBI esearch
    string, and candidate GSE datasets.

    Args:
        query: Free-text query, e.g. "colorectal cancer plasma cfDNA methylation".
        intent: Structured intent instead of free text (deterministic, no LLM
            parse). Fields: mode, accessions{geo[]}, cancer_type{display,
            tcga_code}, platform ("450K"|"EPIC"|"WGBS"|"RRBS"), data_type,
            sample_types[] ("cfdna"|"plasma"|"serum"|"tumor"|"adjacent"|
            "normal"|...), year_start, year_end, extra_keywords[], notes.
        max_candidates: Max candidates in the response (1-200; the total
            recall count is always included).
    """
    body: Dict[str, Any] = {"max_candidates": max(1, min(int(max_candidates), 200))}
    if query:
        body["query"] = query
    if intent:
        body["intent"] = intent
    if "query" not in body and "intent" not in body:
        raise ValueError("Provide `query` (free text) or `intent` (structured).")
    return _api().post("/search/dry-run", json=body)


@server.tool()
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get one pipeline task by ID: status (pending|running|done|failed|
    cancelled), timestamps, and the result summary when done."""
    return _api().get(f"/tasks/{task_id}")


# --------------------------------------------------------------------------- #
#  Submission (Phase 2)                                                       #
# --------------------------------------------------------------------------- #

def _submit_one(spec: Dict[str, Any], agent_type: str = "database",
                force: bool = False) -> Dict[str, Any]:
    """POST /query for one spec ({query?|intent?, label?}). Shared by
    submit_query and submit_compound_query."""
    body: Dict[str, Any] = {"agent_type": agent_type, "force": force}
    if spec.get("query"):
        body["query"] = spec["query"]
    if spec.get("intent"):
        body["intent"] = spec["intent"]
    if spec.get("label"):
        body["label"] = spec["label"]
    if "query" not in body and "intent" not in body:
        raise ValueError("Each sub-task needs `query` (free text) or `intent` (structured).")
    return _api().post("/query", json=body)


@server.tool()
def submit_query(query: str = "", intent: Optional[Dict[str, Any]] = None,
                 label: str = "", agent_type: str = "database",
                 force: bool = False) -> Dict[str, Any]:
    """Submit a full pipeline task (search → per-dataset LLM filter → register
    → async download). Takes ~1 hour; returns a task_id immediately — poll
    with get_task_status / get_task_result.

    Prefer search_datasets first to preview the recall. Re-submitting an
    equivalent active task is deduplicated server-side (deduplicated=true →
    the existing task_id was returned).

    Args:
        query: Free-text query (LLM-parsed), e.g. "colorectal cancer plasma
            cfDNA methylation".
        intent: Structured intent instead of free text (deterministic, no LLM
            parse). Fields: mode, cancer_type{display, tcga_code}, platform,
            data_type, sample_types[], year_start, year_end, extra_keywords[],
            notes. Provide query or intent.
        label: Optional sub-task role tag, e.g. "main" / "auxiliary".
        agent_type: "database" (default; dataset retrieval) or "both" (also
            runs the literature agent).
        force: Skip dedup and always create a new task.
    """
    return _submit_one({"query": query, "intent": intent, "label": label},
                       agent_type=agent_type, force=force)


@server.tool()
def submit_compound_query(main: Dict[str, Any],
                          auxiliary: Optional[Dict[str, Any]] = None,
                          agent_type: str = "database") -> Dict[str, Any]:
    """Submit a compound request as TWO independent, labeled tasks — the task
    boundary is the sub-task boundary, so each gets its own search string and
    filter criteria (a merged single query demonstrably drops the auxiliary
    clause). Typical use: main = training data (e.g. plasma cfDNA, case vs
    healthy controls), auxiliary = validation/marker-space data (e.g. tumor vs
    adjacent normal tissue, TCGA COAD/READ).

    Each of main/auxiliary is an object: {query: "..." } or {intent: {...}}
    (see submit_query), optionally {label: "..."} (defaults: "main" /
    "auxiliary").

    Returns {main: <task>, auxiliary: <task|null>}. Poll each task_id with
    get_task_result; the label distinguishes the roles. A GSE recalled by both
    tasks is downloaded once and attributed per sample.
    """
    m = dict(main)
    m.setdefault("label", "main")
    out: Dict[str, Any] = {"main": _submit_one(m, agent_type=agent_type)}
    if auxiliary:
        a = dict(auxiliary)
        a.setdefault("label", "auxiliary")
        out["auxiliary"] = _submit_one(a, agent_type=agent_type)
    else:
        out["auxiliary"] = None
    return out


@server.tool()
def get_task_result(task_id: str, exclude_limit: int = 0) -> Dict[str, Any]:
    """Result of a finished (or running) task: per-verdict dataset buckets
    with host-side file paths.

    The `download` and `manual_review` buckets list full records; excluded
    datasets are summarized as a count (set exclude_limit > 0 to list up to
    that many accessions). While the task is pending/running the buckets are
    empty — check `status`.

    Bucket items carry local_path (the data file) once downloaded; the
    per-sample metadata table lives next to it as sample_metadata.csv in the
    same directory.
    """
    task = _api().get(f"/tasks/{task_id}")
    items = _api().get(f"/query-datasets/{task_id}").get("items", [])

    buckets: Dict[str, list] = {"download": [], "manual_review": [], "exclude": []}
    for it in items:
        action = it.get("recommended_action") or "unknown"
        if action not in buckets:
            buckets[action] = []
        buckets[action].append(it)

    def _entry(it: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "accession": it.get("accession"),
            "title": it.get("title"),
            "platform": it.get("platform"),
            "cancer_type": it.get("cancer_type"),
            "sample_count": it.get("sample_count"),
            "download_status": it.get("download_status"),
            "local_path": translate_path(it.get("local_path")),
            # eligibility inputs (see _LIST_FIELDS note)
            "disease_groups": it.get("disease_groups"),
            "sample_level_annotation": it.get("sample_level_annotation"),
        }

    excluded = buckets.get("exclude", [])
    return {
        "task_id": task_id,
        "label": task.get("label"),
        "status": task.get("status"),
        "error": task.get("error"),
        "summary": task.get("result"),
        "counts": {k: len(v) for k, v in buckets.items()},
        "download": [_entry(i) for i in buckets.get("download", [])],
        "manual_review": [_entry(i) for i in buckets.get("manual_review", [])],
        "exclude": (
            [i.get("accession") for i in excluded[: max(0, int(exclude_limit))]]
            if exclude_limit else []
        ),
    }


@server.tool()
def cancel_task(task_id: str) -> Dict[str, Any]:
    """Cancel a PENDING task (running tasks cannot be cancelled — the worker
    is already executing them)."""
    return _api().post(f"/tasks/{task_id}/cancel")


@server.tool()
def list_datasets(status: str = "", limit: int = 50) -> Dict[str, Any]:
    """List datasets in the registry, newest-updated first.

    Args:
        status: Optional download_status filter
            (pending|downloading|done|failed|skipped).
        limit: Max datasets to return (1-500).
    """
    params: Dict[str, Any] = {"limit": max(1, min(int(limit), 500))}
    if status:
        params["status"] = status
    out = _api().get("/datasets", params=params)
    return {
        "total": out.get("total"),
        "datasets": [_compact_dataset(d) for d in out.get("datasets", [])],
    }


@server.tool()
def get_dataset(accession: str) -> Dict[str, Any]:
    """Full registry record for one dataset (e.g. "GSE124600"), including
    host-side local_path / sample_metadata_path and the LLM filter evidence."""
    return translate_record(_api().get(f"/datasets/{accession}"))


@server.tool()
def health() -> Dict[str, Any]:
    """Backend health: worker heartbeats + registry dataset counts."""
    return _api().get("/health")
