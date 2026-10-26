"""
MCP over streamable HTTP (remote consumers) + bearer-token auth.

Run:
    python -m mcp_server --http --host 0.0.0.0 --port 8081

Auth: set MCP_TOKEN to require `Authorization: Bearer <token>` on every
request. Without MCP_TOKEN the server is open — fine for local dev, NOT
for anything reachable by strangers. The stdio transport (__main__ default)
never requires a token (local process trust).

The streamable-http app runs STATELESS: no session affinity, so any number
of replicas can sit behind a plain load balancer.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from mcp_server.server import server


class BearerAuthMiddleware:
    """Minimal ASGI middleware: constant-time bearer check on http scopes."""

    def __init__(self, app: Any, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http":
            import hmac
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode("latin-1")
            expected = f"Bearer {self.token}"
            if not hmac.compare_digest(auth, expected):
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"error": "unauthorized: missing/invalid bearer token"}',
                })
                return
        await self.app(scope, receive, send)


def build_http_app(token: Optional[str] = None):
    """Streamable-HTTP ASGI app, optionally wrapped in bearer auth."""
    app = server.streamable_http_app(stateless_http=True)
    if token:
        app = BearerAuthMiddleware(app, token)
    return app


def run_http(host: str = "0.0.0.0", port: int = 8081) -> None:
    import uvicorn

    token = os.environ.get("MCP_TOKEN") or None
    if not token:
        import warnings
        warnings.warn(
            "MCP_TOKEN is not set — the HTTP MCP endpoint is UNAUTHENTICATED. "
            "Set MCP_TOKEN before exposing this to anything but localhost."
        )
    uvicorn.run(build_http_app(token), host=host, port=port)
