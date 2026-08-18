"""
Tests for the GSM-grain registry tables (samples + query_sample_map).

Covers the new sample-level many-to-many model that mirrors
data/{GSE}/sample_metadata.csv: (accession, gsm) as joint primary key and
task_id ↔ gsm verdict mapping, plus the task-routed lookup the download worker
uses instead of guessing from a CSV column.
"""
import tempfile
import unittest

from registry.registry import Registry


def _gsm_details():
    return [
        {"gsm": "GSM1", "source_name": "Brain", "molecule": "genomic DNA",
         "group": "tumor", "characteristics": {"tissue": "brain"}},
        {"gsm": "GSM2", "source_name": "Brain", "molecule": "genomic DNA",
         "group": "control", "characteristics": {}},
    ]


class TestGsmRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_reg_gsm_")
        self.reg = Registry(db_path=f"{self.tmp}/test.db")

    # -- schema / migration ------------------------------------------------- #

    def test_tables_created_and_migration_idempotent(self):
        """Constructing Registry twice must not error; both new tables exist."""
        with self.reg._get_conn() as conn:
            for table in ("samples", "query_sample_map"):
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                self.assertTrue(cols, f"{table} missing")
        # Second construction (re-running _init_db) is a no-op.
        reg2 = Registry(db_path=str(self.reg.db_path))
        self.assertIsNotNone(reg2)

    # -- samples ------------------------------------------------------------ #

    def test_upsert_samples_idempotent_and_merges(self):
        cancer_map = {"GSM1": "query_cancer", "GSM2": "control"}
        n1 = self.reg.upsert_samples("GSE_X", _gsm_details(), cancer_map=cancer_map)
        n2 = self.reg.upsert_samples("GSE_X", _gsm_details())  # no cancer_map
        self.assertEqual(n1, 2)
        self.assertEqual(n2, 2)

        with self.reg._get_conn() as conn:
            rows = conn.execute(
                "SELECT gsm, source_name, sample_group, cancer, characteristics_json "
                "FROM samples WHERE accession='GSE_X' ORDER BY gsm"
            ).fetchall()
        self.assertEqual([r["gsm"] for r in rows], ["GSM1", "GSM2"])
        # cancer set on first write, preserved (COALESCE) on the cancer-less write.
        self.assertEqual(rows[0]["cancer"], "query_cancer")
        self.assertEqual(rows[1]["cancer"], "control")
        # characteristics round-trips as JSON.
        self.assertIn('"tissue"', rows[0]["characteristics_json"] or "")

    def test_upsert_samples_empty_is_noop(self):
        self.assertEqual(self.reg.upsert_samples("GSE_X", []), 0)

    # -- query_sample_map (many-to-many verdicts) --------------------------- #

    def test_record_gsm_verdicts_many_to_many(self):
        """Two tasks hitting the same GSE each store their own per-GSM verdicts."""
        self.reg.upsert_samples("GSE_X", _gsm_details())
        self.reg.record_gsm_verdicts("t1", "GSE_X",
                                     [{"gsm": "GSM1", "include": True},
                                      {"gsm": "GSM2", "include": False}],
                                     raw_query="q1")
        self.reg.record_gsm_verdicts("t2", "GSE_X",
                                     [{"gsm": "GSM1", "include": False, "reason": "wrong tissue"},
                                      {"gsm": "GSM2", "include": True}],
                                     raw_query="q2")

        with self.reg._get_conn() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM query_sample_map WHERE accession='GSE_X'"
            ).fetchone()[0]
        self.assertEqual(n, 4)  # 2 tasks × 2 GSMs

    def test_get_downloadable_gsms_isolated_per_task(self):
        """The core routing fix: each task gets only its own download GSMs."""
        self.reg.upsert_samples("GSE_X", _gsm_details())
        self.reg.record_gsm_verdicts("t1", "GSE_X",
                                     [{"gsm": "GSM1", "include": True},
                                      {"gsm": "GSM2", "include": False}])
        self.reg.record_gsm_verdicts("t2", "GSE_X",
                                     [{"gsm": "GSM1", "include": False},
                                      {"gsm": "GSM2", "include": True}])

        self.assertEqual(self.reg.get_downloadable_gsms("t1", "GSE_X"), ["GSM1"])
        self.assertEqual(self.reg.get_downloadable_gsms("t2", "GSE_X"), ["GSM2"])
        # No/unknown task -> empty (caller falls back to whole-file / CSV).
        self.assertEqual(self.reg.get_downloadable_gsms("", "GSE_X"), [])
        self.assertEqual(self.reg.get_downloadable_gsms("tX", "GSE_X"), [])

    def test_record_gsm_verdicts_rerun_updates_cells(self):
        """Re-judging the same task updates its rows in place (INSERT OR REPLACE)."""
        self.reg.upsert_samples("GSE_X", _gsm_details())
        self.reg.record_gsm_verdicts("t1", "GSE_X",
                                     [{"gsm": "GSM1", "include": True},
                                      {"gsm": "GSM2", "include": False}])
        self.reg.record_gsm_verdicts("t1", "GSE_X",
                                     [{"gsm": "GSM1", "include": False},
                                      {"gsm": "GSM2", "include": True}])

        self.assertEqual(self.reg.get_downloadable_gsms("t1", "GSE_X"), ["GSM2"])
        with self.reg._get_conn() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM query_sample_map "
                "WHERE task_id='t1' AND accession='GSE_X'"
            ).fetchone()[0]
        self.assertEqual(n, 2)  # not duplicated

    def test_record_gsm_verdicts_empty_inputs(self):
        self.assertEqual(
            self.reg.record_gsm_verdicts("", "GSE_X", [{"gsm": "GSM1", "include": True}]), 0)
        self.assertEqual(
            self.reg.record_gsm_verdicts("t1", "GSE_X", []), 0)

    # -- joined views ------------------------------------------------------- #

    def test_get_samples_by_task_id_and_counts(self):
        # GSE-level row so the join has a title/status to pick up.
        self.reg.upsert_dataset(accession="GSE_X", source="GEO", discovered_by="test",
                                title="Some series", download_status="pending")
        self.reg.upsert_samples("GSE_X", _gsm_details())
        self.reg.record_gsm_verdicts("t1", "GSE_X",
                                     [{"gsm": "GSM1", "include": True},
                                      {"gsm": "GSM2", "include": False}],
                                     raw_query="q1")

        rows = self.reg.get_samples_by_task_id("t1")
        self.assertEqual(len(rows), 2)
        by_gsm = {r["gsm"]: r for r in rows}
        self.assertEqual(by_gsm["GSM1"]["verdict"], "download")
        self.assertEqual(by_gsm["GSM2"]["verdict"], "not download")
        self.assertEqual(by_gsm["GSM1"]["gse_title"], "Some series")
        self.assertIsNone(by_gsm["GSM1"]["cancer"])  # no cancer_map passed here

        counts = self.reg.get_gsm_counts_by_task("t1")
        self.assertEqual(counts, [{"accession": "GSE_X", "total_gsm": 2,
                                   "download_gsm": 1, "not_download_gsm": 1}])


if __name__ == "__main__":
    unittest.main()
