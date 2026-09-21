"""
Tests for the non-downgrading download_status guard in Registry.upsert_dataset.

A repeat query re-registers a dataset with download_status='pending'; before the
guard this reset already-done datasets and the daemon re-downloaded them
(docs/caching_audit.md §2.3, GSE149438 ×3). 'done'/'downloading' must survive;
softer transitions ('failed'/'awaiting_approval' → 'pending') stay allowed, and
update_status keeps its explicit hard-override semantics.
"""
import tempfile
import unittest

from registry.registry import Registry


def _status(reg, acc):
    with reg._get_conn() as conn:
        row = conn.execute(
            "SELECT download_status FROM datasets WHERE accession = ?", (acc,)
        ).fetchone()
    return row["download_status"] if row else None


class TestUpsertNoDowngrade(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_reg_status_")
        self.reg = Registry(db_path=f"{self.tmp}/test.db")

    def _upsert(self, acc, status):
        self.reg.upsert_dataset(
            accession=acc, source="GEO", discovered_by="test",
            download_status=status,
        )

    # -- protected statuses -------------------------------------------------- #

    def test_done_survives_pending_reregister(self):
        """Repeat query re-registering a done dataset must not reset it."""
        self._upsert("GSE_A", "pending")
        self.reg.update_status("GSE_A", "done", local_path="/data/GSE_A")
        self._upsert("GSE_A", "pending")          # repeat query arrives
        self.assertEqual(_status(self.reg, "GSE_A"), "done")

    def test_done_survives_awaiting_approval(self):
        self._upsert("GSE_B", "pending")
        self.reg.update_status("GSE_B", "done")
        self._upsert("GSE_B", "awaiting_approval")  # re-judged as manual_review
        self.assertEqual(_status(self.reg, "GSE_B"), "done")

    def test_downloading_survives_pending(self):
        """Mid-download re-queue must not make the next poll double-process it."""
        self._upsert("GSE_C", "pending")
        self.reg.update_status("GSE_C", "downloading")
        self._upsert("GSE_C", "pending")
        self.assertEqual(_status(self.reg, "GSE_C"), "downloading")

    # -- allowed transitions -------------------------------------------------- #

    def test_failed_to_pending_still_allowed(self):
        """Repeat query re-queues a failed download (retry path unchanged)."""
        self._upsert("GSE_D", "pending")
        self.reg.update_status("GSE_D", "failed")
        self._upsert("GSE_D", "pending")
        self.assertEqual(_status(self.reg, "GSE_D"), "pending")

    def test_awaiting_approval_to_pending_allowed(self):
        """New verdict (manual_review → download) still re-queues."""
        self._upsert("GSE_E", "awaiting_approval")
        self._upsert("GSE_E", "pending")
        self.assertEqual(_status(self.reg, "GSE_E"), "pending")

    def test_insert_stores_status_verbatim(self):
        self._upsert("GSE_F", "awaiting_approval")
        self.assertEqual(_status(self.reg, "GSE_F"), "awaiting_approval")

    # -- explicit paths bypass the guard -------------------------------------- #

    def test_update_status_still_hard_overrides(self):
        """Worker/approve/retry paths use update_status — must keep authority."""
        self._upsert("GSE_G", "pending")
        self.reg.update_status("GSE_G", "done")
        self.reg.update_status("GSE_G", "downloading")
        self.assertEqual(_status(self.reg, "GSE_G"), "downloading")
        self.reg.update_status("GSE_G", "done")
        self.assertEqual(_status(self.reg, "GSE_G"), "done")


if __name__ == "__main__":
    unittest.main()
