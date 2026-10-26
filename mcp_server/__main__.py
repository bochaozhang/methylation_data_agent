"""
Entry point: run the MethyAgent MCP server.

Default: stdio (local Claude Code, any cwd):
    <venv>/bin/python /abs/path/to/mcp_server/__main__.py

Remote consumers: streamable HTTP + optional bearer auth (MCP_TOKEN):
    python -m mcp_server --http --host 0.0.0.0 --port 8081
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp_server.server import server  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MethyAgent MCP server")
    parser.add_argument("--http", action="store_true",
                        help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    if args.http:
        from mcp_server.http_server import run_http
        run_http(host=args.host, port=args.port)
    else:
        server.run()  # stdio transport by default


if __name__ == "__main__":
    main()
