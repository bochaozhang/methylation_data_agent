"""
Tests for the Phase-3 remote pieces: bearer-auth ASGI middleware (for the
streamable-HTTP MCP endpoint) — the submit-side quota guard is covered in
test_api_query_submit.py.
"""
import unittest

from mcp_server.http_server import BearerAuthMiddleware, build_http_app


class _Recorder:
    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1


class _SendCollector:
    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)


def _scope(headers=None):
    return {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or [])],
    }


async def _noop_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


class TestBearerAuthMiddleware(unittest.TestCase):
    def _run(self, mw, scope):
        import asyncio
        collector = _SendCollector()
        asyncio.run(mw(scope, _noop_receive, collector))
        return collector

    def test_missing_token_401(self):
        inner = _Recorder()
        mw = BearerAuthMiddleware(inner, "secret")
        out = self._run(mw, _scope())
        self.assertEqual(out.events[0]["status"], 401)
        self.assertEqual(inner.calls, 0)

    def test_wrong_token_401(self):
        inner = _Recorder()
        mw = BearerAuthMiddleware(inner, "secret")
        out = self._run(mw, _scope([("Authorization", "Bearer wrong")]))
        self.assertEqual(out.events[0]["status"], 401)
        self.assertEqual(inner.calls, 0)

    def test_correct_token_passthrough(self):
        inner = _Recorder()
        mw = BearerAuthMiddleware(inner, "secret")
        self._run(mw, _scope([("authorization", "Bearer secret")]))
        self.assertEqual(inner.calls, 1)

    def test_non_http_scope_passthrough(self):
        inner = _Recorder()
        mw = BearerAuthMiddleware(inner, "secret")
        import asyncio
        asyncio.run(mw({"type": "lifespan"}, _noop_receive, _SendCollector()))
        self.assertEqual(inner.calls, 1)


class TestBuildApp(unittest.TestCase):
    def test_app_builds_with_and_without_token(self):
        self.assertTrue(callable(build_http_app()))
        self.assertTrue(callable(build_http_app("t")))


if __name__ == "__main__":
    unittest.main()
