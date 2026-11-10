#!/usr/bin/env bash
# Planted-error test of both reviewers (scripts/eval_reviewer.py) on the real papers from the
# grounding run. Build the base first: python scripts/grounding_eval/build_reviewer_base.py
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
ENVF="${METHY_ENV_FILE:-$REPO/.env}"
[ -f "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
exec ./.venv/bin/python3 scripts/eval_reviewer.py --base data/eval/reviewer_base.json "$@"
