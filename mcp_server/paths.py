"""
Container → host path translation.

The registry stores container-side paths (/app/data/...) because the download
worker runs inside Docker (compose mount: ./data:/app/data). MCP consumers
usually run on the host, so every path crossing the MCP boundary is translated
to its host-side equivalent before being returned.

Override the host root with MCP_DATA_HOST_PATH (default: <repo>/data).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

CONTAINER_DATA_PREFIX = "/app/data/"

# Registry record fields that hold filesystem paths and need translation.
PATH_FIELDS = ("local_path", "sample_metadata_path")


def host_data_root() -> str:
    """Host-side directory corresponding to the container's /app/data."""
    return os.environ.get("MCP_DATA_HOST_PATH") or str(REPO_ROOT / "data")


def translate_path(path: Optional[str]) -> Optional[str]:
    """Map a container-side /app/data/** path to the host side.

    Anything else (already-host paths, URLs, None, "") passes through unchanged.
    """
    if not path or not isinstance(path, str):
        return path
    if path.startswith(CONTAINER_DATA_PREFIX):
        return os.path.join(host_data_root(), path[len(CONTAINER_DATA_PREFIX):])
    return path


def translate_record(record: Dict[str, Any],
                     fields: Iterable[str] = PATH_FIELDS) -> Dict[str, Any]:
    """Translate every path field of a registry record dict, in place."""
    for f in fields:
        if record.get(f):
            record[f] = translate_path(record[f])
    return record
