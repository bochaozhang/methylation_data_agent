"""
Shared SQLite registry for MethyAgent.

Both DatabaseAgent and LiteratureAgent read/write this registry to:
  - Record discovered datasets and their download status
  - Prevent duplicate downloads across agents
  - Track download progress and errors
  - Manage the task queue for daemon-based execution

Tables:
  datasets              - One row per unique accession (primary dedup key)
  download_log          - Append-only event log for each accession
  llm_extraction_cache  - DOI-keyed cache for LLM extraction results
  task_queue            - Task queue for agent daemon polling
"""
import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Registry:
    """
    Thread-safe SQLite registry for methylation dataset tracking.

    Usage:
        registry = Registry("./registry/methyagent.db")
        registry.upsert_dataset(accession="GSE124600", source="GEO", ...)
        exists = registry.exists("GSE124600")
        registry.update_status("GSE124600", "done", local_path="/data/GSE124600")

        # LLM extraction cache
        registry.cache_llm_result(doi="10.1038/...", accessions=["GSE124600"], ...)
        cached = registry.get_llm_cache(doi="10.1038/...")

        # Task queue (daemon mode)
        task_id = registry.create_task(agent_type="database", query="LUAD methylation")
        task = registry.claim_task(agent_type="database")
        registry.complete_task(task_id, result={"datasets": [...]})
    """

    # Valid download status values
    STATUS_AWAITING_APPROVAL = "awaiting_approval"  # found by agent, waiting for human confirmation
    STATUS_PENDING = "pending"       # approved, queued for download
    STATUS_DOWNLOADING = "downloading"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_SKIPPED = "skipped"
    STATUS_PENDING_REVIEW = "pending_review"   # LLM medium-confidence, awaiting human review

    # Valid task status values
    TASK_PENDING = "pending"
    TASK_RUNNING = "running"
    TASK_DONE = "done"
    TASK_FAILED = "failed"
    TASK_CANCELLED = "cancelled"

    def __init__(self, db_path: str = "./registry/methyagent.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context manager for thread-safe database connections."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read/write
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create tables and add new columns if they don't exist."""
        # Step 1: Create tables (no-op if they already exist)
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    accession       TEXT PRIMARY KEY,
                    source          TEXT NOT NULL,
                    data_type       TEXT,
                    cancer_type     TEXT,
                    platform        TEXT,
                    sample_count    INTEGER,
                    year            INTEGER,
                    title           TEXT,
                    download_status TEXT DEFAULT 'pending',
                    local_path      TEXT,
                    paper_pmid      TEXT,
                    paper_doi       TEXT,
                    discovered_by   TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    checksum_md5    TEXT,
                    needs_review    INTEGER DEFAULT 0,
                    llm_evidence    TEXT,
                    sample_type     TEXT,
                    disease_groups          TEXT,
                    stage_treatment         TEXT,
                    available_file_type     TEXT,
                    sample_level_annotation TEXT,
                    usable                  INTEGER DEFAULT 1,
                    recommended_action      TEXT,
                    reason                  TEXT,
                    notes                   TEXT,
                    no_pubmed_link          INTEGER DEFAULT 0,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS download_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    accession   TEXT NOT NULL,
                    event       TEXT NOT NULL,
                    message     TEXT,
                    timestamp   TEXT NOT NULL,
                    FOREIGN KEY (accession) REFERENCES datasets(accession)
                );

                CREATE TABLE IF NOT EXISTS llm_extraction_cache (
                    doi             TEXT PRIMARY KEY,
                    pdf_url         TEXT,
                    extracted_json  TEXT,
                    accessions      TEXT,
                    model_used      TEXT,
                    created_at      TEXT,
                    hit_count       INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS task_queue (
                    task_id     TEXT PRIMARY KEY,
                    agent_type  TEXT NOT NULL,
                    query       TEXT NOT NULL,
                    status      TEXT DEFAULT 'pending',
                    created_at  TEXT NOT NULL,
                    started_at  TEXT,
                    finished_at TEXT,
                    result_json TEXT,
                    error       TEXT
                );
            """)

        # Step 2: Migrate existing databases BEFORE creating indexes
        # (indexes on new columns will fail if the column doesn't exist yet)
        self._migrate_schema()

        # Step 3: Create indexes (safe after migration ensures columns exist)
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_datasets_status
                    ON datasets(download_status);
                CREATE INDEX IF NOT EXISTS idx_datasets_source
                    ON datasets(source);
                CREATE INDEX IF NOT EXISTS idx_datasets_needs_review
                    ON datasets(needs_review);
                CREATE INDEX IF NOT EXISTS idx_log_accession
                    ON download_log(accession);
                CREATE INDEX IF NOT EXISTS idx_task_status
                    ON task_queue(status, agent_type);
            """)

    def _migrate_schema(self):
        """Add new columns to existing databases (safe no-op if already present)."""
        migrations = [
            "ALTER TABLE datasets ADD COLUMN needs_review INTEGER DEFAULT 0",
            "ALTER TABLE datasets ADD COLUMN llm_evidence TEXT",
            "ALTER TABLE datasets ADD COLUMN sample_type TEXT",
            # v2 columns
            "ALTER TABLE datasets ADD COLUMN disease_groups TEXT",
            "ALTER TABLE datasets ADD COLUMN stage_treatment TEXT",
            "ALTER TABLE datasets ADD COLUMN available_file_type TEXT",
            "ALTER TABLE datasets ADD COLUMN sample_level_annotation TEXT",
            "ALTER TABLE datasets ADD COLUMN usable INTEGER DEFAULT 1",
            "ALTER TABLE datasets ADD COLUMN recommended_action TEXT",
            "ALTER TABLE datasets ADD COLUMN reason TEXT",
            "ALTER TABLE datasets ADD COLUMN notes TEXT",
            # v5 columns
            "ALTER TABLE datasets ADD COLUMN no_pubmed_link INTEGER DEFAULT 0",
            # v6 columns
            "ALTER TABLE datasets ADD COLUMN sample_metadata_path TEXT",
        ]
        with self._get_conn() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # Column already exists

    # ------------------------------------------------------------------ #
    #  Core CRUD                                                           #
    # ------------------------------------------------------------------ #

    def exists(self, accession: str) -> bool:
        """Return True if the accession is already registered."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM datasets WHERE accession = ?", (accession,)
            ).fetchone()
            return row is not None

    def get(self, accession: str) -> Optional[Dict]:
        """Return the dataset record as a dict, or None if not found."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE accession = ?", (accession,)
            ).fetchone()
            return dict(row) if row else None

    def upsert_dataset(
        self,
        accession: str,
        source: str,
        discovered_by: str,
        data_type: Optional[str] = None,
        cancer_type: Optional[str] = None,
        platform: Optional[str] = None,
        sample_count: Optional[int] = None,
        year: Optional[int] = None,
        title: Optional[str] = None,
        paper_pmid: Optional[str] = None,
        paper_doi: Optional[str] = None,
        download_status: str = "pending",
        needs_review: bool = False,
        llm_evidence: Optional[str] = None,
        sample_type: Optional[str] = None,
        # v2 columns
        disease_groups: Optional[str] = None,
        stage_treatment: Optional[str] = None,
        available_file_type: Optional[str] = None,
        sample_level_annotation: Optional[str] = None,
        usable: int = 1,
        recommended_action: Optional[str] = None,
        reason: Optional[str] = None,
        notes: Optional[str] = None,
        no_pubmed_link: bool = False,
        sample_metadata_path: Optional[str] = None,
    ) -> bool:
        """
        Insert a new dataset or update metadata if it already exists.

        Returns:
            True if a new record was inserted, False if updated.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                existing = conn.execute(
                    "SELECT accession FROM datasets WHERE accession = ?",
                    (accession,),
                ).fetchone()

                if existing:
                    conn.execute(
                        """
                        UPDATE datasets SET
                            data_type               = COALESCE(?, data_type),
                            cancer_type             = COALESCE(?, cancer_type),
                            platform                = COALESCE(?, platform),
                            sample_count            = COALESCE(?, sample_count),
                            year                    = COALESCE(?, year),
                            title                   = COALESCE(?, title),
                            paper_pmid              = COALESCE(?, paper_pmid),
                            paper_doi               = COALESCE(?, paper_doi),
                            needs_review            = COALESCE(?, needs_review),
                            llm_evidence            = COALESCE(?, llm_evidence),
                            sample_type             = COALESCE(?, sample_type),
                            disease_groups          = COALESCE(?, disease_groups),
                            stage_treatment         = COALESCE(?, stage_treatment),
                            available_file_type     = COALESCE(?, available_file_type),
                            sample_level_annotation = COALESCE(?, sample_level_annotation),
                            usable                  = COALESCE(?, usable),
                            recommended_action      = COALESCE(?, recommended_action),
                            reason                  = COALESCE(?, reason),
                            notes                   = COALESCE(?, notes),
                            no_pubmed_link          = COALESCE(?, no_pubmed_link),
                            sample_metadata_path    = COALESCE(?, sample_metadata_path),
                            download_status         = ?,
                            updated_at              = ?
                        WHERE accession = ?
                        """,
                        (
                            data_type, cancer_type, platform, sample_count,
                            year, title, paper_pmid, paper_doi,
                            int(needs_review), llm_evidence, sample_type,
                            disease_groups, stage_treatment, available_file_type,
                            sample_level_annotation, usable, recommended_action,
                            reason, notes,
                            int(no_pubmed_link),
                            sample_metadata_path,
                            download_status,
                            now, accession,
                        ),
                    )
                    return False
                else:
                    conn.execute(
                        """
                        INSERT INTO datasets (
                            accession, source, data_type, cancer_type, platform,
                            sample_count, year, title, download_status,
                            paper_pmid, paper_doi, discovered_by,
                            needs_review, llm_evidence, sample_type,
                            disease_groups, stage_treatment, available_file_type,
                            sample_level_annotation, usable, recommended_action,
                            reason, notes, no_pubmed_link,
                            sample_metadata_path,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            accession, source, data_type, cancer_type, platform,
                            sample_count, year, title, download_status,
                            paper_pmid, paper_doi, discovered_by,
                            int(needs_review), llm_evidence, sample_type,
                            disease_groups, stage_treatment, available_file_type,
                            sample_level_annotation, usable, recommended_action,
                            reason, notes,
                            int(no_pubmed_link),
                            sample_metadata_path,
                            now, now,
                        ),
                    )
                    return True

    def update_status(
        self,
        accession: str,
        status: str,
        local_path: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        checksum_md5: Optional[str] = None,
    ):
        """Update the download status and optional file metadata."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    UPDATE datasets SET
                        download_status = ?,
                        local_path      = COALESCE(?, local_path),
                        file_size_bytes = COALESCE(?, file_size_bytes),
                        checksum_md5    = COALESCE(?, checksum_md5),
                        updated_at      = ?
                    WHERE accession = ?
                    """,
                    (status, local_path, file_size_bytes, checksum_md5, now, accession),
                )

    def approve_review(self, accession: str):
        """Approve a pending_review dataset for download."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    UPDATE datasets SET
                        needs_review    = 0,
                        download_status = 'pending',
                        updated_at      = ?
                    WHERE accession = ? AND needs_review = 1
                    """,
                    (now, accession),
                )

    def reject_review(self, accession: str):
        """Reject a pending_review dataset (mark as skipped)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    UPDATE datasets SET
                        needs_review    = 0,
                        download_status = 'skipped',
                        updated_at      = ?
                    WHERE accession = ? AND needs_review = 1
                    """,
                    (now, accession),
                )

    def log_event(self, accession: str, event: str, message: str = ""):
        """Append an event to the download log."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO download_log (accession, event, message, timestamp) VALUES (?, ?, ?, ?)",
                (accession, event, message, now),
            )

    # ------------------------------------------------------------------ #
    #  Query helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_all(self, status: Optional[str] = None) -> List[Dict]:
        """Return all datasets, optionally filtered by status."""
        with self._get_conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM datasets WHERE download_status = ? ORDER BY created_at",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM datasets ORDER BY created_at"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_pending_review(self) -> List[Dict]:
        """Return all datasets flagged for human review."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets WHERE needs_review = 1 ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_awaiting_approval(self) -> List[Dict]:
        """Return all datasets awaiting human download approval."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets WHERE download_status = 'awaiting_approval' ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def approve_downloads(self, approved_accessions: List[str]) -> Dict[str, int]:
        """
        Confirm download for the given accessions.

        - approved_accessions → download_status = 'pending'
        - all other 'awaiting_approval' records NOT in the list → 'skipped'

        Returns dict with keys "approved" and "skipped".
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                approved_count = 0
                skipped_count = 0
                if approved_accessions:
                    placeholders = ",".join("?" * len(approved_accessions))
                    cur = conn.execute(
                        f"UPDATE datasets SET download_status = 'pending', updated_at = ? "
                        f"WHERE accession IN ({placeholders}) AND download_status = 'awaiting_approval'",
                        [now] + list(approved_accessions),
                    )
                    approved_count = cur.rowcount
                # Skip all remaining awaiting_approval records not in the approved list
                if approved_accessions:
                    placeholders = ",".join("?" * len(approved_accessions))
                    cur2 = conn.execute(
                        f"UPDATE datasets SET download_status = 'skipped', updated_at = ? "
                        f"WHERE download_status = 'awaiting_approval' AND accession NOT IN ({placeholders})",
                        [now] + list(approved_accessions),
                    )
                else:
                    # No accessions approved → skip all awaiting
                    cur2 = conn.execute(
                        "UPDATE datasets SET download_status = 'skipped', updated_at = ? "
                        "WHERE download_status = 'awaiting_approval'",
                        (now,),
                    )
                skipped_count = cur2.rowcount
                return {"approved": approved_count, "skipped": skipped_count}

    def get_accession_set(self) -> set:
        """Return the set of all registered accessions (fast dedup check)."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT accession FROM datasets").fetchall()
            return {r["accession"] for r in rows}

    def get_summary(self) -> Dict:
        """Return a summary dict for the final report."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            by_status = {
                row["download_status"]: row["cnt"]
                for row in conn.execute(
                    "SELECT download_status, COUNT(*) as cnt FROM datasets GROUP BY download_status"
                ).fetchall()
            }
            by_source = {
                row["source"]: row["cnt"]
                for row in conn.execute(
                    "SELECT source, COUNT(*) as cnt FROM datasets GROUP BY source"
                ).fetchall()
            }
            by_type = {
                row["data_type"]: row["cnt"]
                for row in conn.execute(
                    "SELECT data_type, COUNT(*) as cnt FROM datasets WHERE data_type IS NOT NULL GROUP BY data_type"
                ).fetchall()
            }
            by_sample_type = {
                row["sample_type"]: row["cnt"]
                for row in conn.execute(
                    "SELECT sample_type, COUNT(*) as cnt FROM datasets WHERE sample_type IS NOT NULL GROUP BY sample_type"
                ).fetchall()
            }
            agent1_count = conn.execute(
                "SELECT COUNT(*) FROM datasets WHERE discovered_by = 'agent1'"
            ).fetchone()[0]
            agent2_count = conn.execute(
                "SELECT COUNT(*) FROM datasets WHERE discovered_by = 'agent2'"
            ).fetchone()[0]
            pending_review_count = conn.execute(
                "SELECT COUNT(*) FROM datasets WHERE needs_review = 1"
            ).fetchone()[0]
            llm_cache_count = conn.execute(
                "SELECT COUNT(*) FROM llm_extraction_cache"
            ).fetchone()[0]

        return {
            "total": total,
            "by_status": by_status,
            "by_source": by_source,
            "by_data_type": by_type,
            "by_sample_type": by_sample_type,
            "agent1_discovered": agent1_count,
            "agent2_discovered": agent2_count,
            "pending_review": pending_review_count,
            "llm_cache_entries": llm_cache_count,
        }

    # ------------------------------------------------------------------ #
    #  LLM extraction cache                                               #
    # ------------------------------------------------------------------ #

    def cache_llm_result(
        self,
        doi: str,
        accessions: List[str],
        extracted_json: str = "",
        pdf_url: str = "",
        model_used: str = "",
    ) -> None:
        """Store LLM extraction result in the cache table."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_extraction_cache
                   (doi, pdf_url, extracted_json, accessions, model_used, created_at, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (doi, pdf_url, extracted_json, json.dumps(accessions), model_used, now),
            )

    def get_llm_cache(self, doi: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached LLM extraction result for a DOI."""
        if not doi:
            return None
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT extracted_json, accessions, model_used, hit_count FROM llm_extraction_cache WHERE doi = ?",
                (doi,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE llm_extraction_cache SET hit_count = hit_count + 1 WHERE doi = ?",
                    (doi,),
                )
                return {
                    "extracted_json": row[0],
                    "accessions": json.loads(row[1]) if row[1] else [],
                    "model_used": row[2],
                    "hit_count": row[3] + 1,
                }
        return None

    def clear_llm_cache(self, doi: Optional[str] = None) -> int:
        """Clear LLM cache. If doi given, clear only that entry. Returns rows deleted."""
        with self._get_conn() as conn:
            if doi:
                cur = conn.execute(
                    "DELETE FROM llm_extraction_cache WHERE doi = ?", (doi,)
                )
            else:
                cur = conn.execute("DELETE FROM llm_extraction_cache")
            return cur.rowcount

    # ------------------------------------------------------------------ #
    #  Task queue (daemon mode)                                           #
    # ------------------------------------------------------------------ #

    def create_task(
        self,
        query: str,
        agent_type: str = "both",
        task_id: Optional[str] = None,
    ) -> str:
        """
        Create a new task in the queue.

        Args:
            query: The search query string.
            agent_type: 'database', 'literature', or 'both'.
            task_id: Optional explicit task ID; auto-generated if None.

        Returns:
            The task_id string.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO task_queue (task_id, agent_type, query, status, created_at)
                       VALUES (?, ?, ?, 'pending', ?)""",
                    (task_id, agent_type, query, now),
                )
        return task_id

    def claim_task(self, agent_type: str) -> Optional[Dict]:
        """
        Atomically claim the oldest pending task for the given agent type.

        For 'database': claims tasks with agent_type IN ('database', 'both').
        For 'literature': claims tasks with agent_type IN ('literature', 'both').

        Returns:
            Task dict if claimed, None if no pending tasks.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                if agent_type == "database":
                    row = conn.execute(
                        """SELECT task_id, query, agent_type FROM task_queue
                           WHERE status = 'pending'
                             AND agent_type IN ('database', 'both')
                           ORDER BY created_at ASC LIMIT 1"""
                    ).fetchone()
                elif agent_type == "literature":
                    row = conn.execute(
                        """SELECT task_id, query, agent_type FROM task_queue
                           WHERE status = 'pending'
                             AND agent_type IN ('literature', 'both')
                           ORDER BY created_at ASC LIMIT 1"""
                    ).fetchone()
                else:
                    row = conn.execute(
                        """SELECT task_id, query, agent_type FROM task_queue
                           WHERE status = 'pending'
                           ORDER BY created_at ASC LIMIT 1"""
                    ).fetchone()

                if row is None:
                    return None

                task_id = row["task_id"]
                conn.execute(
                    "UPDATE task_queue SET status = 'running', started_at = ? WHERE task_id = ?",
                    (now, task_id),
                )
                return {
                    "task_id": task_id,
                    "query": row["query"],
                    "agent_type": row["agent_type"],
                }

    def complete_task(self, task_id: str, result: Any = None) -> None:
        """Mark a task as done and store the result JSON."""
        now = datetime.now(timezone.utc).isoformat()
        result_json = json.dumps(result) if result is not None else None
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """UPDATE task_queue SET status = 'done', finished_at = ?, result_json = ?
                       WHERE task_id = ?""",
                    (now, result_json, task_id),
                )

    def fail_task(self, task_id: str, error: str = "") -> None:
        """Mark a task as failed with an error message."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """UPDATE task_queue SET status = 'failed', finished_at = ?, error = ?
                       WHERE task_id = ?""",
                    (now, error, task_id),
                )

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task. Returns True if cancelled."""
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "UPDATE task_queue SET status = 'cancelled' WHERE task_id = ? AND status = 'pending'",
                    (task_id,),
                )
                return cur.rowcount > 0

    def cancel_all_tasks(self) -> Dict[str, int]:
        """
        Cancel all pending tasks and skip all pending dataset downloads.

        Returns:
            dict with keys "tasks_cancelled" and "datasets_skipped".
        """
        with self._lock:
            with self._get_conn() as conn:
                cur_tasks = conn.execute(
                    "UPDATE task_queue SET status = 'cancelled' WHERE status IN ('pending', 'running')"
                )
                cur_datasets = conn.execute(
                    "UPDATE datasets SET download_status = 'skipped' WHERE download_status = 'pending'"
                )
                return {
                    "tasks_cancelled": cur_tasks.rowcount,
                    "datasets_skipped": cur_datasets.rowcount,
                }

    def retry_failed_datasets(self) -> int:
        """
        Reset all failed dataset downloads back to 'pending' so the
        agent daemon will pick them up on the next poll.

        Returns:
            Number of datasets reset to pending.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    """
                    UPDATE datasets
                    SET download_status = 'pending', updated_at = ?
                    WHERE download_status = 'failed'
                    """,
                    (now,),
                )
                return cur.rowcount

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Return a single task record as a dict, or None."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM task_queue WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            if d.get("result_json"):
                try:
                    d["result"] = json.loads(d["result_json"])
                except Exception:
                    d["result"] = None
            else:
                d["result"] = None
            return d

    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        List tasks from the queue, ordered by created_at DESC.

        Args:
            status: Filter by status ('pending', 'running', 'done', 'failed').
            limit: Maximum number of tasks to return.
        """
        with self._get_conn() as conn:
            if status:
                rows = conn.execute(
                    """SELECT * FROM task_queue
                       WHERE status = ? AND task_id NOT LIKE 'heartbeat_%'
                       ORDER BY created_at DESC LIMIT ?""",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM task_queue
                       WHERE task_id NOT LIKE 'heartbeat_%'
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()

            result = []
            for row in rows:
                d = dict(row)
                if d.get("result_json"):
                    try:
                        d["result"] = json.loads(d["result_json"])
                    except Exception:
                        d["result"] = None
                else:
                    d["result"] = None
                result.append(d)
            return result

    def update_heartbeat(self, agent_type: str) -> None:
        """Record a heartbeat timestamp for the given agent daemon."""
        now = datetime.now(timezone.utc).isoformat()
        heartbeat_id = f"heartbeat_{agent_type}"
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO task_queue
                   (task_id, agent_type, query, status, created_at, started_at)
                   VALUES (?, ?, '__heartbeat__', 'running', ?, ?)""",
                (heartbeat_id, agent_type, now, now),
            )

    def get_heartbeat(self, agent_type: str) -> Optional[str]:
        """Return the last heartbeat ISO timestamp for the given agent, or None."""
        heartbeat_id = f"heartbeat_{agent_type}"
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT started_at FROM task_queue WHERE task_id = ?",
                (heartbeat_id,),
            ).fetchone()
            return row["started_at"] if row else None

    # ------------------------------------------------------------------ #
    #  Dedup utility for URL-based sources (supplementary files)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def url_to_accession(url: str) -> str:
        """Generate a stable pseudo-accession from a URL (MD5 hash prefix)."""
        return "URL_" + hashlib.md5(url.encode()).hexdigest()[:12].upper()
