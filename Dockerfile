# ============================================================
# MethyAgent — single image used by all three services
#   agent1:  python scripts/agent_daemon.py --agent database
#   agent2:  python scripts/agent_daemon.py --agent literature
#   webui:   uvicorn api.main:app --host 0.0.0.0 --port 8080
# ============================================================
FROM python:3.11-slim

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root user ----
RUN groupadd --gid 1000 methyagent \
 && useradd  --uid 1000 --gid methyagent --shell /bin/bash --create-home methyagent

# ---- Working directory ----
WORKDIR /app

# ---- Python dependencies ----
# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY --chown=methyagent:methyagent . .

# ---- Runtime directories (will be overridden by volume mounts) ----
RUN mkdir -p /app/registry /app/data \
 && chown -R methyagent:methyagent /app/registry /app/data

# ---- Switch to non-root ----
USER methyagent

# ---- Default command (overridden per service in docker-compose.yml) ----
CMD ["python", "scripts/agent_daemon.py", "--agent", "database"]

# ---- Health check ----
# Default probe: is the registry writable by this uid? The worker daemons
# serve no HTTP port, so an endpoint probe is meaningless for them; the real
# failure mode this catches is WAL/-shm re-owned by a host-side DB opener
# (2026-11-02 incident: daemons ran ~30h unable to write anything).
# webui/mcp override this with their own endpoint probes in docker-compose.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python scripts/healthcheck_registry.py
