"""Tests for skills/geo_filter/file_inspect.py (Phase 2a file-form preview)."""
import gzip
import unittest

from skills.geo_filter.file_inspect import inspect_matrix_head, verify_a_level_files


class TestInspectMatrixHead(unittest.TestCase):
    def test_beta_matrix(self):
        m = "cg\tGSM1\tGSM2\tGSM3\ncg1\t0.1\t0.9\t0.5\ncg2\t0.2\t0.8\t0.4\n"
        r = inspect_matrix_head(m.encode())
        self.assertEqual(r["value_type"], "beta")
        self.assertTrue(r["is_A_level"])
        self.assertTrue(r["gsm_in_columns"])

    def test_m_value_matrix(self):
        m = "cg\tGSM1\tGSM2\ncg1\t-2.3\t1.5\ncg2\t-0.5\t3.1\n"
        r = inspect_matrix_head(m.encode())
        self.assertEqual(r["value_type"], "m_value")
        self.assertTrue(r["is_A_level"])

    def test_paired_counts(self):
        m = "cg\tmeth_count\tunmeth_count\ncg1\t10\t90\ncg2\t40\t60\n"
        r = inspect_matrix_head(m.encode())
        self.assertEqual(r["value_type"], "paired_counts")
        self.assertTrue(r["is_A_level"])

    def test_count_table_is_not_A_level(self):
        m = "gene\ts1\ts2\nA\t1200\t1500\nB\t800\t950\n"
        r = inspect_matrix_head(m.encode())
        self.assertFalse(r["is_A_level"])

    def test_pvalue_table_is_non_methylation(self):
        m = "gene\tlogFC\tpvalue\nA\t2.3\t0.001\nB\t-1.2\t0.04\n"
        r = inspect_matrix_head(m.encode())
        self.assertEqual(r["value_type"], "non_methylation")
        self.assertFalse(r["is_A_level"])

    def test_gzip_truncated_head(self):
        m = "cg\tGSM1\tGSM2\ncg1\t0.1\t0.9\ncg2\t0.2\t0.8\n"
        gz = gzip.compress(m.encode())[:60]  # truncated
        r = inspect_matrix_head(gz)
        self.assertTrue(r["is_A_level"])

    def test_empty_head(self):
        r = inspect_matrix_head(b"")
        self.assertEqual(r["value_type"], "unknown")
        self.assertFalse(r["is_A_level"])


class TestVerifyALevelFiles(unittest.TestCase):
    class _Geo:
        def __init__(self, head=b""):
            self._head = head
        def fetch_file_head(self, url):
            return self._head

    def test_series_matrix_trusted(self):
        has_A, files, form = verify_a_level_files(
            ["https://x/GSE1_series_matrix.txt.gz"], self._Geo())
        self.assertTrue(has_A)
        self.assertTrue(files[0]["is_A_level"])
        self.assertEqual(files[0]["data_form"], "series_matrix")
        # did not fetch (trusted)
        self.assertEqual(form, "series_matrix")

    def test_inspected_beta_file(self):
        head = "cg\tGSM1\tGSM2\ncg1\t0.1\t0.9\n".encode()
        has_A, files, form = verify_a_level_files(
            ["https://x/GSE1_beta.txt.gz"], self._Geo(head))
        self.assertTrue(has_A)
        self.assertTrue(files[0]["is_A_level"])
        self.assertEqual(form, "beta")

    def test_inspected_non_methylation(self):
        head = "gene\ts1\ts2\nA\t1200\t1500\n".encode()
        has_A, files, form = verify_a_level_files(
            ["https://x/GSE1_counts.txt.gz"], self._Geo(head))
        self.assertFalse(has_A)
        self.assertFalse(files[0]["is_A_level"])

    def test_unfetchable_file(self):
        has_A, files, form = verify_a_level_files(
            ["https://x/GSE1_x.txt.gz"], self._Geo(b""))
        self.assertFalse(has_A)
        self.assertEqual(files[0]["reason"], "could not fetch/inspect head")


if __name__ == "__main__":
    unittest.main()
