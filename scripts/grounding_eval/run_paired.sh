#!/usr/bin/env bash
# Paired (controlled) run: one extraction per paper, graded before and after the reviewer.
# Output: data/eval/clean/q{1,2,3}.json  ->  analyze with analyze_paired.py
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
ENVF="${METHY_ENV_FILE:-$REPO/.env}"
[ -f "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
exec ./.venv/bin/python3 scripts/grounding_eval/paired_review_eval.py "$@"
