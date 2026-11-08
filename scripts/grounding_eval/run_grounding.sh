#!/usr/bin/env bash
# Run scripts/eval_grounding.py from the repo root with API keys loaded from an env file.
# Env file: $METHY_ENV_FILE if set, else <repo>/.env. Keys are exported, never printed.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
ENVF="${METHY_ENV_FILE:-$REPO/.env}"
[ -f "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
exec ./.venv/bin/python3 scripts/eval_grounding.py "$@"
