"""
Tests for task_queue v2 columns: label + intent_json.

Covers: fresh-DB creation, migration of a pre-v2 database (old CREATE TABLE
without the columns, then Registry() adds them), create→claim→get roundtrip,
and list_active_tasks (the submit-side dedup helper).
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from registry.registry import Registry


class TestTaskColumns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "reg.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _intent_json(self):
        import json
        return json.dumps({
            "mode": "semantic",
            "cancer_type": {"display": "colorectal cancer", "tcga_code": "COAD"},
            "sample_types": ["cfdna"],
        })

    def test_fresh_db_roundtrip(self):
        reg = Registry(db_path=self.db)
        tid = reg.create_task(
            query="structured: COAD cfDNA",
            agent_type="database",
            label="main",
            intent_json=self._intent_json(),
        )
        claimed = reg.claim_task("database")
        self.assertEqual(claimed["task_id"], tid)
        self.assertEqual(claimed["label"], "main")
        self.assertEqual(claimed["intent_json"], self._intent_json())
        # get_task picks the new columns up via SELECT *
        task = reg.get_task(tid)
        self.assertEqual(task["label"], "main")
        self.assertEqual(task["status"], "running")

    def test_migration_from_pre_v2_db(self):
        """A DB created with the old schema gains label/intent_json on init."""
        conn = sqlite3.connect(self.db)
        conn.execute(
            """CREATE TABLE task_queue (
                   task_id TEXT PRIMARY KEY, agent_type TEXT NOT NULL,
                   query TEXT NOT NULL, status TEXT DEFAULT 'pending',
                   created_at TEXT NOT NULL, started_at TEXT,
                   finished_at TEXT, result_json TEXT, error TEXT)"""
        )
        conn.execute(
            "INSERT INTO task_queue (task_id, agent_type, query, status, created_at)"
            " VALUES ('t1', 'database', 'old query', 'pending', '2026-01-01')"
        )
        conn.commit()
        conn.close()

        reg = Registry(db_path=self.db)  # runs migrations
        cols = {r[1] for r in sqlite3.connect(self.db).execute(
            "PRAGMA table_info(task_queue)")}
        self.assertIn("label", cols)
        self.assertIn("intent_json", cols)
        # Pre-existing row survived, claims fine, new columns default NULL.
        claimed = reg.claim_task("database")
        self.assertEqual(claimed["task_id"], "t1")
        self.assertIsNone(claimed["label"])
        self.assertIsNone(claimed["intent_json"])

    def test_create_defaults_none(self):
        reg = Registry(db_path=self.db)
        tid = reg.create_task(query="plain free-text query", agent_type="both")
        task = reg.get_task(tid)
        self.assertIsNone(task["label"])
        self.assertIsNone(task["intent_json"])

    def test_list_active_tasks(self):
        reg = Registry(db_path=self.db)
        t_pending = reg.create_task(query="q1", agent_type="database", label="main")
        t_done = reg.create_task(query="q2", agent_type="database")
        reg.complete_task(t_done, result={"ok": True})

        active = reg.list_active_tasks(agent_type="database")
        self.assertEqual([t["task_id"] for t in active], [t_pending])
        self.assertEqual(active[0]["label"], "main")

        # No agent_type filter still excludes done tasks.
        active_all = reg.list_active_tasks()
        self.assertEqual([t["task_id"] for t in active_all], [t_pending])


class TestRequeueOrphanedTasks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "reg.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_orphaned_running_requeued_to_pending(self):
        reg = Registry(db_path=self.db)
        tid = reg.create_task(query="q", agent_type="database", label="auxiliary")
        reg.claim_task("database")  # → running, started_at set

        requeued = reg.requeue_orphaned_tasks("database")
        self.assertEqual(requeued, [tid])
        task = reg.get_task(tid)
        self.assertEqual(task["status"], "pending")
        self.assertIsNone(task["started_at"])
        # And it is claimable again.
        again = reg.claim_task("database")
        self.assertEqual(again["task_id"], tid)
        self.assertEqual(again["label"], "auxiliary")

    def test_done_and_other_type_untouched(self):
        reg = Registry(db_path=self.db)
        t_done = reg.create_task(query="done q", agent_type="database")
        reg.claim_task("database")
        reg.complete_task(t_done, result={})
        t_lit = reg.create_task(query="lit q", agent_type="literature")
        reg.claim_task("literature")

        # Database worker startup: its own done task and the literature task
        # (not claimable by database) must be left alone.
        requeued = reg.requeue_orphaned_tasks("database")
        self.assertEqual(requeued, [])
        self.assertEqual(reg.get_task(t_done)["status"], "done")
        self.assertEqual(reg.get_task(t_lit)["status"], "running")

    def test_both_agent_type_is_claimable(self):
        """'both' tasks are claimed by database workers → requeued by them."""
        reg = Registry(db_path=self.db)
        t_both = reg.create_task(query="both q", agent_type="both")
        reg.claim_task("database")
        self.assertEqual(reg.get_task(t_both)["status"], "running")

        self.assertEqual(reg.requeue_orphaned_tasks("database"), [t_both])
        self.assertEqual(reg.get_task(t_both)["status"], "pending")

    def test_heartbeat_rows_never_requeued_never_claimed(self):
        """Incident 2026-10-19: a requeued heartbeat row briefly became a
        claimable 'pending' task. Heartbeat rows must be excluded from both
        requeue and claim."""
        reg = Registry(db_path=self.db)
        t_real = reg.create_task(query="real q", agent_type="database")
        reg.update_heartbeat("database")  # heartbeat_database row, status running

        # Requeue skips the heartbeat row even if it sits in 'running'.
        self.assertEqual(reg.requeue_orphaned_tasks("database"), [])

        # Even a pending heartbeat row (the race window) is never claimed.
        import sqlite3
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE task_queue SET status='pending' WHERE task_id='heartbeat_database'")
        conn.commit()
        conn.close()
        claimed = reg.claim_task("database")
        self.assertEqual(claimed["task_id"], t_real)


if __name__ == "__main__":
    unittest.main()
