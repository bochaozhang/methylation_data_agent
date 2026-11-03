#!/usr/bin/env python
"""Container healthcheck: is the production registry WRITABLE by this uid?

Guards the 2026-11-02 incident class: a collaborator opening the SQLite DB
directly on the host (as a different uid) re-owns methyagent.db-wal/-shm;
SQLite WAL mode requires write access to those files, so the container
daemons kept running but every write — heartbeat, claim, complete — failed
with "attempt to write a readonly database" for ~30h while `docker ps`
showed a meaningless badge (the old probe curl'd :8080/health, a port the
worker containers never serve).

Exit 0 = healthy (DB + existing WAL/SHM + parent dir writable),
exit 1 = the daemon will not be able to persist anything.

Note: this is deliberately NOT a heartbeat-freshness probe — tasks run
inline in the daemon claim loop, so heartbeats legitimately freeze for the
~1h duration of a running task (a freshness check would flap unhealthy on
every task).
"""
import os
import sys

DB_DIR = os.environ.get("REGISTRY_DIR", "/app/registry")
DB = os.path.join(DB_DIR, "methyagent.db")


def main() -> int:
    if not os.path.exists(DB):
        print(f"healthcheck: registry db missing: {DB}")
        return 1
    paths = [DB] + [
        side for side in (DB + "-wal", DB + "-shm")
        if os.path.exists(side)  # absent side files are fine (created on demand)
    ]
    for path in [DB_DIR] + paths:
        if not (os.access(path, os.R_OK) and os.access(path, os.W_OK)):
            print(f"healthcheck: not writable by uid {os.getuid()}: {path}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
