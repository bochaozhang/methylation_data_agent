"""
Entry point: run the MethyAgent MCP server over stdio.

Works as a plain script from any cwd (Claude Code launches it this way):
    <venv>/bin/python /abs/path/to/mcp_server/__main__.py
or as a module from the repo root:
    python -m mcp_server
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp_server.server import server  # noqa: E402

if __name__ == "__main__":
    server.run()  # stdio transport by default
