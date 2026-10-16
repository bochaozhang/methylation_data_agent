"""
Thin HTTP client for the MethyAgent REST API (the webui container).

Configuration (env):
    MCP_API_BASE     Base URL of the API   (default http://localhost:80)
    MCP_API_TIMEOUT  Per-request timeout   (default 300 s — dry-run search
                     can legitimately take a minute over the NCBI proxy)
"""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx


class ApiError(RuntimeError):
    """Non-2xx response from the REST API; message is caller-facing."""

    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"MethyAgent API error {status_code}: {detail}")


class ApiClient:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or os.environ.get("MCP_API_BASE") or "http://localhost:80").rstrip("/")
        effective_timeout = float(
            timeout or os.environ.get("MCP_API_TIMEOUT") or 300
        )
        self._client = httpx.Client(base_url=self.base_url, timeout=effective_timeout)

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._unwrap(self._client.get(path, params=params))

    def post(self, path: str, json: Optional[Any] = None,
             params: Optional[dict] = None) -> Any:
        return self._unwrap(self._client.post(path, json=json, params=params))

    @staticmethod
    def _unwrap(resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(resp.status_code, detail)
        return resp.json()

    def close(self) -> None:
        self._client.close()
