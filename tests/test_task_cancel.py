"""
Tests for cooperative task cancellation.

Upstream feedback (2026-10-20): running tasks (~1h + ~136 LLM verdicts)
could not be stopped — cancel_task only accepted pending. Now cancelling
a running task flips it to 'cancel_requested'; the pipeline's filter loop
checks the flag at every candidate boundary (_raise_if_cancelled) and
raises TaskCancelledError, which the daemon finalizes as 'cancelled'.
The startup self-heal honors the request instead of resurrecting such tasks.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agents.agent1_pipeline import TaskCancelledError, _raise_if_cancelled
from registry.registry import Registry


class TestRegistryCancel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = Registry(db_path=str(Path(self.tmp.name) / "reg.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def _running_task(self):
        tid = self.reg.create_task(query="q", agent_type="database")
        self.reg.claim_task("database")  # → running
        return tid

    def test_pending_cancels_immediately(self):
        tid = self.reg.create_task(query="q", agent_type="database")
        self.assertTrue(self.reg.cancel_task(tid))
        self.assertEqual(self.reg.get_task(tid)["status"], "cancelled")

    def test_running_goes_cancel_requested(self):
        tid = self._running_task()
        self.assertTrue(self.reg.cancel_task(tid))
        task = self.reg.get_task(tid)
        self.assertEqual(task["status"], "cancel_requested")
        self.assertTrue(self.reg.is_cancel_requested(tid))

    def test_finalize_cancelled_task(self):
        tid = self._running_task()
        self.reg.cancel_task(tid)
        self.reg.finalize_cancelled_task(tid, result={"partial": True})
        task = self.reg.get_task(tid)
        self.assertEqual(task["status"], "cancelled")
        self.assertIsNotNone(task["finished_at"])
        self.assertEqual(task["result"]["partial"], True)

    def test_finalize_ignores_non_requested(self):
        """A task that completed normally must not be overwritten by finalize."""
        tid = self._running_task()
        self.reg.complete_task(tid, result={"ok": 1})
        self.reg.finalize_cancelled_task(tid, result={"partial": True})
        self.assertEqual(self.reg.get_task(tid)["status"], "done")
        self.assertEqual(self.reg.get_task(tid)["result"], {"ok": 1})

    def test_done_and_cancelled_rejected(self):
        tid = self._running_task()
        self.reg.complete_task(tid, result={})
        self.assertFalse(self.reg.cancel_task(tid))
        t2 = self.reg.create_task(query="q2", agent_type="database")
        self.reg.cancel_task(t2)
        self.assertFalse(self.reg.cancel_task(t2))

    def test_cancel_requested_not_claimable(self):
        tid = self._running_task()
        self.reg.cancel_task(tid)
        # No pending tasks → nothing to claim.
        self.assertIsNone(self.reg.claim_task("database"))

    def test_selfheal_honors_cancel_request(self):
        """A cancel_requested orphan is finalized cancelled, NOT resurrected."""
        tid = self._running_task()
        self.reg.cancel_task(tid)
        t2 = self.reg.create_task(query="q2", agent_type="database")
        self.reg.claim_task("database")  # t2 → running (a real orphan)

        requeued = self.reg.requeue_orphaned_tasks("database")
        self.assertEqual(requeued, [t2])          # only the true orphan requeued
        self.assertEqual(self.reg.get_task(tid)["status"], "cancelled")
        self.assertEqual(self.reg.get_task(t2)["status"], "pending")


class TestCancelCheckpoint(unittest.TestCase):
    def test_raises_when_requested(self):
        reg = MagicMock()
        reg.is_cancel_requested.return_value = True
        with self.assertRaises(TaskCancelledError):
            _raise_if_cancelled(reg, "task-1")

    def test_noop_when_not_requested(self):
        reg = MagicMock()
        reg.is_cancel_requested.return_value = False
        _raise_if_cancelled(reg, "task-1")  # must not raise
        reg.is_cancel_requested.assert_called_once_with("task-1")

    def test_noop_without_task_id_or_registry(self):
        _raise_if_cancelled(None, None)
        _raise_if_cancelled(None, "task-1")
        reg = MagicMock()
        _raise_if_cancelled(reg, None)
        reg.is_cancel_requested.assert_not_called()


if __name__ == "__main__":
    unittest.main()
