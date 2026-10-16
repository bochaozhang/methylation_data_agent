"""
MethyAgent MCP server — Phase 1 read-only tools over stdio.

Every tool proxies the MethyAgent REST API (the webui container) and
translates container-side file paths (/app/data/**) to host-side paths before
returning, so consumers can read the files directly.

Tools:
    search_datasets    dry-run recall preview (parse + GEO search, no filter,
                       no writes, no downloads — seconds, near-zero cost)
    get_task_status    one pipeline task by ID
    list_datasets      browse the registry (optionally by download_status)
    get_dataset        full record for one accession, incl. file paths
    health             backend heartbeat + registry stats

Phase 2 (planned): submit_query / submit_compound_query / get_task_result /
cancel_task — see docs/mcp_interface_plan.md.

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
from mcp_server.paths import translate_record  # noqa: E402

server = MCPServer(
    name="methyagent",
    title="MethyAgent",
    description=(
        "DNA methylation dataset retrieval agent: GEO/TCGA search, LLM-based "
        "filtering against SPEC criteria, download registry."
    ),
    instructions=(
        "Typical flow: (1) search_datasets to preview what a query recalls "
        "(cheap, no side effects); (2) a full pipeline task takes ~1 hour — "
        "submit it via the REST API POST /query (MCP submission is Phase 2); "
        "(3) poll get_task_status; (4) list_datasets/get_dataset to locate "
        "downloaded files. Paths returned are host-side; read data files "
        "directly from disk. download_status: pending | downloading | done | "
        "failed | skipped."
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
