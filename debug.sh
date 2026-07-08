#!/usr/bin/env bash
#
# debug.sh — wrapper for scripts/debug_geo_filter.py
#
# Loads .env, ensures the NCBI SOCKS tunnel (1080) is up (starts it if down),
# then runs the geo_filter debug harness with any args you pass.
#
# Usage:
#   ./debug.sh --query "colorectal cancer和非癌对照的cfDNA甲基化数据" --accession GSE124600
#   ./debug.sh --query "..." --accession GSE124600 GSE110185   # 多个对比
#   ./debug.sh --check            # provision env+tunnel only; print env-ok/env-bad; exit
#
# Override the tunnel-start command with GEO_TUNNEL_SSH=... if your setup differs.
set -euo pipefail

# debug.sh lives at the project root (not in a subdir), so ROOT is its own dir.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 1) load .env without overriding already-exported vars
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
export NCBI_PROXY="${NCBI_PROXY:-socks5h://127.0.0.1:1080}"

# 2) default tunnel-start command (BatchMode = fail fast, never prompt)
GEO_TUNNEL_SSH="${GEO_TUNNEL_SSH:-ssh -D 0.0.0.0:1080 -N -f -o BatchMode=yes -o ConnectTimeout=10 -i ${HOME}/.ssh/id_ed25519 -o ServerAliveInterval=60 zhangbochao1222@35.237.123.242}"

probe_tunnel() {
  curl -fsS --max-time 6 --socks5-hostname 127.0.0.1:1080 \
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json" >/dev/null 2>&1
}

ensure_tunnel() {
  if probe_tunnel; then return 0; fi
  echo "[debug.sh] tunnel down — starting: ${GEO_TUNNEL_SSH}" >&2
  # shellcheck disable=SC2086
  ${GEO_TUNNEL_SSH} >/dev/null 2>&1 || true
  sleep 2
  probe_tunnel
}

if [[ "${1:-}" == "--check" ]]; then
  if ensure_tunnel; then echo "env-ok"; exit 0; else echo "env-bad (NCBI tunnel unavailable)"; exit 1; fi
fi

ensure_tunnel || {
  echo "[debug.sh] NCBI tunnel unavailable and could not auto-start." >&2
  echo "[debug.sh] Start it manually, e.g.: ssh -D 0.0.0.0:1080 -N -f -i ~/.ssh/id_ed25519 ..." >&2
  exit 1
}

PY="${ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" "${ROOT}/scripts/debug_geo_filter.py" "$@"
