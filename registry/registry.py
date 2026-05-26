"""
Shared SQLite registry for MethyAgent.

Both DatabaseAgent and LiteratureAgent read/write this registry to:
  - Record discovered datasets and their download status
  - Prevent duplicate downloads across agents
  - Track download progress and errors

Tables:
  datasets              - One row per unique accession (primary dedup key)
  download_log          - Append-only event log for each accession
  llm_extraction_cache  - DOI-keyed cache for LLM extraction results
"""
import hashlib
import sqlite3
import threading
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
    """

    # Valid download status values
    STATUS_PENDING = "pending"
    STATUS_DOWNLOADING = "downloading"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_SKIPPED = "skipped"
    STATUS_PENDING_REVIEW = "pending_review"   # LLM medium-confidence, awaiting human review

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

                CREATE INDEX IF NOT EXISTS idx_datasets_status
                    ON datasets(download_status);
                CREATE INDEX IF NOT EXISTS idx_datasets_source
                    ON datasets(source);
                CREATE INDEX IF NOT EXISTS idx_datasets_needs_review
                    ON datasets(needs_review);
                CREATE INDEX IF NOT EXISTS idx_log_accession
                    ON download_log(accession);
            """)

        # Migrate existing databases: add new columns if missing
        self._migrate_schema()

    def _migrate_schema(self):
        """Add new columns to existing databases (safe no-op if already present)."""
        migrations = [
            "ALTER TABLE datasets ADD COLUMN needs_review INTEGER DEFAULT 0",
            "ALTER TABLE datasets ADD COLUMN llm_evidence TEXT",
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
    ) -> bool:
        """
        Insert a new dataset or update metadata if it already exists.

        Args:
            needs_review: If True, marks this as a medium-confidence LLM extraction
                          requiring human confirmation before download.
            llm_evidence: The evidence quote from the LLM extraction (for review UI).

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
                    # Update metadata but preserve download_status if already done
                    conn.execute(
                        """
                        UPDATE datasets SET
                            data_type    = COALESCE(?, data_type),
                            cancer_type  = COALESCE(?, cancer_type),
                            platform     = COALESCE(?, platform),
                            sample_count = COALESCE(?, sample_count),
                            year         = COALESCE(?, year),
                            title        = COALESCE(?, title),
                            paper_pmid   = COALESCE(?, paper_pmid),
                            paper_doi    = COALESCE(?, paper_doi),
                            needs_review = COALESCE(?, needs_review),
                            llm_evidence = COALESCE(?, llm_evidence),
                            updated_at   = ?
                        WHERE accession = ?
                        """,
                        (
                            data_type, cancer_type, platform, sample_count,
                            year, title, paper_pmid, paper_doi,
                            int(needs_review), llm_evidence,
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
                            needs_review, llm_evidence,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            accession, source, data_type, cancer_type, platform,
                            sample_count, year, title, download_status,
                            paper_pmid, paper_doi, discovered_by,
                            int(needs_review), llm_evidence,
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
        """
        Approve a pending_review dataset for download.
        Clears needs_review flag and sets status to 'pending'.
        """
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
        """
        Reject a pending_review dataset (mark as skipped).
        """
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
        """Return all datasets flagged for human review (medium-confidence LLM extractions)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets WHERE needs_review = 1 ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

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
        """
        Store LLM extraction result in the cache table.

        Args:
            doi: Paper DOI (cache key).
            accessions: List of extracted accession strings.
            extracted_json: Raw LLM JSON response.
            pdf_url: Source PDF URL.
            model_used: LLM model identifier.
        """
        import json
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_extraction_cache
                   (doi, pdf_url, extracted_json, accessions, model_used, created_at, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (doi, pdf_url, extracted_json, json.dumps(accessions), model_used, now),
            )

    def get_llm_cache(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached LLM extraction result for a DOI.

        Returns dict with 'accessions', 'model_used', 'hit_count', or None.
        """
        import json
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
    #  Dedup utility for URL-based sources (supplementary files)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def url_to_accession(url: str) -> str:
        """Generate a stable pseudo-accession from a URL (MD5 hash prefix)."""
        return "URL_" + hashlib.md5(url.encode()).hexdigest()[:12].upper()
