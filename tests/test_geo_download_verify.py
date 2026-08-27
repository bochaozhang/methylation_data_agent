"""Tests for geo-download Phase-2 verification (①②③④⑤).

Covers verify.py (column counting with mixed delimiters, aggregate column
check, GSM→column mapping cascade, group completeness, quarantine) and
archive_extract.py (member extraction, per-sample detection), plus the
subset-through-column-map path in skill.py.
"""
import gzip
import io
import os
import tarfile
import tempfile
import unittest
import zipfile

import pandas as pd

from skills.geo_download.archive_extract import (
    detect_per_sample_set,
    extract_archive_members,
    is_archive,
)
from skills.geo_download.verify import (
    build_gsm_column_map,
    check_column_counts,
    check_group_completeness,
    parse_sample_columns,
    parse_series_matrix_samples_local,
    quarantine_files,
)


def _write_gz(path: str, text: str) -> str:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)
    return path


# --------------------------------------------------------------------- #
#  ① Column counting                                                    #
# --------------------------------------------------------------------- #

class TestParseSampleColumns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_mixed_delimiter_header_gse124600_form(self):
        # Header: 7 coord cols tab-separated + sample names SPACE-separated
        # inside the last tab field; data rows pure tab. The real-world trap.
        header = "#chr\tstr\tend\tCGI_num\tcgcgcgg_num\tstrand\tcg_num\tPcrc90 Pcrc88 Pn43"
        rows = "\n".join(
            f"chr10\t100\t101\t4\t1\t+\t{i}\t0.0\t0.1\t0.2" for i in range(1, 4))
        p = _write_gz(os.path.join(self.dir, "m.txt.gz"), header + "\n" + rows)
        res = parse_sample_columns(p)
        self.assertEqual(res["n_sample_cols"], 3)
        self.assertEqual(res["col_names"], ["Pcrc90", "Pcrc88", "Pn43"])

    def test_plain_beta_matrix(self):
        m = "ID_REF\tGSM1\tGSM2\ncg1\t0.1\t0.2\ncg2\t0.3\t0.4\n"
        p = _write_gz(os.path.join(self.dir, "beta.txt.gz"), m)
        res = parse_sample_columns(p)
        self.assertEqual(res["n_sample_cols"], 2)
        self.assertEqual(res["n_gsm_cols"], 2)

    def test_header_only_file(self):
        p = _write_gz(os.path.join(self.dir, "h.txt.gz"),
                      "ID_REF\tGSM1\tGSM2\n")
        res = parse_sample_columns(p)
        self.assertEqual(res["basis"], "header_only")
        self.assertEqual(res["n_sample_cols"], 2)

    def test_annotation_columns_excluded(self):
        m = ("chr\tstart\tend\tGSM1\tGSM2\n"
             "chr1\t100\t200\t0.1\t0.2\nchr1\t300\t400\t0.3\t0.4\n")
        p = _write_gz(os.path.join(self.dir, "coord.txt.gz"), m)
        res = parse_sample_columns(p)
        self.assertEqual(res["n_sample_cols"], 2)

    def test_unparseable_returns_none(self):
        p = _write_gz(os.path.join(self.dir, "empty.txt.gz"), "\n\n")
        self.assertIsNone(parse_sample_columns(p))


class TestCheckColumnCounts(unittest.TestCase):
    def _parsed(self, names):
        return {"n_sample_cols": len(names), "col_names": names,
                "n_gsm_cols": 0, "n_annotation": 1, "n_total_fields": len(names) + 1}

    def test_aggregate_union_across_files(self):
        # GSE124600 shape: 6 + 268 disjoint name sets = 274 union.
        a = self._parsed([f"Pcrc{i}" for i in range(6)])
        b = self._parsed([f"Pn{i}" for i in range(268)])
        chk = check_column_counts([a, b], 274)
        self.assertTrue(chk["passed"])
        self.assertEqual(chk["n_cols"], 274)

    def test_overlap_deduped(self):
        # Two files sharing the same 10 samples → union 10, not 20.
        a = self._parsed([f"GSM{i}" for i in range(10)])
        chk = check_column_counts([a, a], 10)
        self.assertTrue(chk["passed"])
        self.assertEqual(chk["n_cols"], 10)

    def test_fails_beyond_tolerance(self):
        a = self._parsed([f"GSM{i}" for i in range(100)])
        chk = check_column_counts([a], 274)  # |100-274| > max(5, 27)
        self.assertFalse(chk["passed"])

    def test_tolerance_is_max5_10pct(self):
        # 274 expected → tol = max(5, 27) = 27: 300 (diff 26) passes,
        # 310 (diff 36) fails.
        a = self._parsed([f"GSM{i}" for i in range(300)])
        self.assertTrue(check_column_counts([a], 274)["passed"])
        b = self._parsed([f"GSM{i}" for i in range(310)])
        self.assertFalse(check_column_counts([b], 274)["passed"])

    def test_per_sample_form_counts_files(self):
        paths = [f"/tmp/f{i}.txt" for i in range(274)]
        chk = check_column_counts([], 274, per_sample_files=paths)
        self.assertTrue(chk["passed"])
        self.assertEqual(chk["form"], "per_sample")

    def test_no_gsm_set_skips(self):
        chk = check_column_counts([self._parsed(["a"])], 0)
        self.assertTrue(chk["passed"])
        self.assertEqual(chk["form"], "skip")

    def test_no_names_falls_back_to_sum(self):
        a = {"n_sample_cols": 6, "col_names": [], "n_gsm_cols": 0}
        b = {"n_sample_cols": 268, "col_names": [], "n_gsm_cols": 0}
        chk = check_column_counts([a, b], 274)
        self.assertTrue(chk["passed"])
        self.assertEqual(chk["form"], "merged_sum")


# --------------------------------------------------------------------- #
#  ② GSM→column mapping                                                 #
# --------------------------------------------------------------------- #

class TestBuildGsmColumnMap(unittest.TestCase):
    SERIES = [
        {"gsm": "GSM1", "title": "Pcrc90", "source_name": "Plasma"},
        {"gsm": "GSM2", "title": "Pn43_m", "source_name": "Plasma"},
        {"gsm": "GSM3", "title": "Pcrc1_m_dup1.5", "source_name": "Plasma"},
        {"gsm": "GSM4", "title": "Tcrc1", "source_name": "Tissue"},
    ]

    def test_l1_gsm_id_in_column(self):
        # col_names here are SAMPLE columns only (parse_sample_columns already
        # excluded annotation cols like ID_REF).
        m = build_gsm_column_map({"f": ["GSM123", "GSM456"]}, [])
        self.assertEqual(m["level"], "gsm_id")
        self.assertEqual(m["column_map"]["GSM123"]["gsm"], "GSM123")
        self.assertEqual(m["coverage"], 1.0)

    def test_l2_exact_title(self):
        m = build_gsm_column_map({"f": ["Pcrc90", "Pn43_m"]}, self.SERIES)
        self.assertEqual(m["level"], "title")
        self.assertEqual(m["column_map"]["Pcrc90"]["gsm"], "GSM1")
        self.assertEqual(m["column_map"]["Pn43_m"]["gsm"], "GSM2")

    def test_l2_title_prefix_suffix_variant(self):
        # Column Pcrc1_m_dup1.5_extra is title + suffix → prefix match.
        m = build_gsm_column_map({"f": ["Pcrc90", "Pcrc1_m_dup1.5_extra"]},
                                 self.SERIES)
        self.assertEqual(m["column_map"]["Pcrc1_m_dup1.5_extra"]["gsm"], "GSM3")

    def test_duplicate_title_not_mapped(self):
        # Two GSMs share one title → a column matching it is ambiguous and
        # must NOT be mapped to either.
        series = [{"gsm": "GSM1", "title": "P1", "source_name": ""},
                  {"gsm": "GSM2", "title": "P1", "source_name": ""}]
        m = build_gsm_column_map({"f": ["P1"]}, series)
        self.assertNotIn("P1", m["column_map"])
        self.assertEqual(m["coverage"], 0.0)

    def test_no_series_no_mapping(self):
        m = build_gsm_column_map({"f": ["Pcrc90"]}, [])
        self.assertEqual(m["level"], "none")
        self.assertEqual(m["coverage"], 0.0)


class TestParseSeriesMatrixSamplesLocal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_sample_rows(self):
        text = ("!Series_title\t\"test\"\n"
                "!Sample_geo_accession\t\"GSM1\"\t\"GSM2\"\n"
                "!Sample_title\t\"Pcrc1\"\t\"Pn1\"\n"
                "!Sample_source_name_ch1\t\"Plasma\"\t\"Plasma\"\n"
                "!series_matrix_table_begin\nID_REF\tGSM1\tGSM2\n")
        p = _write_gz(os.path.join(self.dir, "sm.txt.gz"), text)
        rows = parse_series_matrix_samples_local(p)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"gsm": "GSM1", "title": "Pcrc1",
                                   "source_name": "Plasma"})

    def test_missing_file_returns_empty(self):
        self.assertEqual(parse_series_matrix_samples_local("/nonexistent"), [])


# --------------------------------------------------------------------- #
#  ③ Group completeness (report-only)                                   #
# --------------------------------------------------------------------- #

class TestGroupCompleteness(unittest.TestCase):
    def setUp(self):
        self.sm = pd.DataFrame({
            "gsm": ["GSM1", "GSM2", "GSM3", "GSM4"],
            "cancer": ["query_cancer", "query_cancer", "control", "control"],
        })
        self.dl = ["GSM1", "GSM2", "GSM3", "GSM4"]

    def test_full_coverage(self):
        cmap = {c: {"gsm": g, "source": "title"} for c, g in
                zip(["a", "b", "c", "d"], ["GSM1", "GSM2", "GSM3", "GSM4"])}
        res = check_group_completeness(cmap, self.sm, self.dl)
        self.assertEqual(res["groups"]["query_cancer"],
                         {"expected": 2, "mapped": 2})
        self.assertEqual(res["groups"]["control"], {"expected": 2, "mapped": 2})
        self.assertEqual(res["gaps"], [])

    def test_gap_reported_not_raised(self):
        cmap = {"a": {"gsm": "GSM1", "source": "title"}}  # GSM2-4 unmapped
        res = check_group_completeness(cmap, self.sm, self.dl)
        # report-only: gaps listed, no exception, no verdict field to fail
        self.assertEqual(len(res["gaps"]), 3)
        self.assertIn("GSM2", res["gaps"])

    def test_no_metadata(self):
        cmap = {"a": {"gsm": "GSM1", "source": "title"}}
        res = check_group_completeness(cmap, None, ["GSM1"])
        self.assertIn("1/1", res["summary"])


# --------------------------------------------------------------------- #
#  ④ Quarantine                                                         #
# --------------------------------------------------------------------- #

class TestQuarantine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.p = _write_gz(os.path.join(self.dir, "beta.txt.gz"),
                           "ID_REF\tGSM1\n cg1\t0.1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_moves_file_and_updates_path(self):
        results = [{"local_path": self.p, "url": "https://x"}]
        recs = quarantine_files("GSETEST", results, self.dir, "col count fail")
        dest = os.path.join(self.dir, "quarantine", "GSETEST", "beta.txt.gz")
        self.assertTrue(os.path.exists(dest))
        self.assertFalse(os.path.exists(self.p))
        self.assertEqual(results[0]["local_path"], dest)  # honest downstream
        self.assertEqual(recs[0]["reason"], "col count fail")
        self.assertEqual(recs[0]["quarantine_path"], dest)
        self.assertTrue(recs[0]["md5"])

    def test_missing_file_skipped(self):
        results = [{"local_path": None}]
        self.assertEqual(quarantine_files("GSETEST", results, self.dir, "r"), [])


# --------------------------------------------------------------------- #
#  ⑤ Archive extraction                                                 #
# --------------------------------------------------------------------- #

class TestArchiveExtract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _make_tar_gz(self, path, members):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data in members:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        with open(path, "wb") as f:
            f.write(buf.getvalue())
        return path

    def test_tar_gz_detected_and_extracted(self):
        p = self._make_tar_gz(os.path.join(self.dir, "supp.tar.gz"), [
            ("sample1.txt", b"ID_REF\tGSM1\ncg1\t0.1\n"),
            ("sample2.txt", b"ID_REF\tGSM2\ncg1\t0.2\n"),
        ])
        self.assertTrue(is_archive(p))
        out = extract_archive_members(p, os.path.join(self.dir, "members"))
        self.assertEqual(len(out), 2)
        with open(out[0]) as f:
            self.assertIn("GSM1", f.read())

    def test_plain_matrix_gz_is_not_archive(self):
        p = _write_gz(os.path.join(self.dir, "matrix.txt.gz"),
                      "ID_REF\tGSM1\ncg1\t0.1\n")
        self.assertFalse(is_archive(p))

    def test_zip_extracted(self):
        p = os.path.join(self.dir, "supp.zip")
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("s1.txt", "ID_REF\tGSM1\ncg1\t0.1\n")
        out = extract_archive_members(p, os.path.join(self.dir, "members"))
        self.assertEqual(len(out), 1)

    def test_corrupt_archive_returns_none(self):
        p = os.path.join(self.dir, "bad.tar.gz")
        with open(p, "wb") as f:
            f.write(b"\x1f\x8bnot-really-gzip")
        self.assertIsNone(extract_archive_members(p, self.dir))

    def test_path_traversal_member_rejected(self):
        p = self._make_tar_gz(os.path.join(self.dir, "evil.tar.gz"), [
            ("../escape.txt", b"x"),
            ("ok.txt", b"y"),
        ])
        out = extract_archive_members(p, os.path.join(self.dir, "members"))
        names = [os.path.basename(x) for x in out]
        self.assertIn("ok.txt", names)
        self.assertNotIn("escape.txt", names)

    def test_detect_per_sample_set(self):
        self.assertTrue(detect_per_sample_set(
            [f"/tmp/GSM{i}.txt" for i in range(5)], 5))
        self.assertTrue(detect_per_sample_set(
            [f"/tmp/f{i}" for i in range(5)], 5))
        self.assertFalse(detect_per_sample_set(["/tmp/one.txt"], 274))


# --------------------------------------------------------------------- #
#  Subset through the column map (skill integration)                    #
# --------------------------------------------------------------------- #

class TestSubsetViaColumnMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        from skills.geo_download.skill import DownloadSkill
        self.skill = DownloadSkill.__new__(DownloadSkill)

    def tearDown(self):
        self.tmp.cleanup()

    def test_submitter_coded_columns_subset_via_map(self):
        # The GSE124600 acceptance scenario, minimised: header columns are
        # submitter codes; only the map knows Pcrc* = query cancer.
        header = "#chr\tstr\tend\tCGI_num\tcgcgcgg_num\tstrand\tcg_num\tPcrc90 Pn43"
        rows = "\n".join(
            f"chr10\t100\t101\t4\t1\t+\t{i}\t0.{i}\t0.0" for i in range(1, 4))
        mtx = _write_gz(os.path.join(self.dir, "GSET_m.txt.gz"),
                        header + "\n" + rows)
        sm = pd.DataFrame({
            "gsm": ["GSM1", "GSM2"],
            "source_name": ["Plasma", "Plasma"],
            "molecule": ["genomic DNA"] * 2,
            "group": ["plasma_cfdna"] * 2,
            "cancer": ["query_cancer", "control"],
        })
        cmap = {"Pcrc90": {"gsm": "GSM1", "source": "title"},
                "Pn43": {"gsm": "GSM2", "source": "title"}}
        results = [{"local_path": mtx, "file_size_bytes": os.path.getsize(mtx)}]
        subset_path, note, forced = self.skill._subset_by_cancer(
            "GSET", results, sm, self.dir,
            query_terms=["colorectal"], cancer_type="colorectal cancer",
            column_map=cmap)
        self.assertIsNone(forced)
        self.assertIsNotNone(subset_path, f"subset failed: {note}")
        self.assertIn("column map", note)
        with gzip.open(subset_path, "rt") as f:
            out_header = f.readline().strip()
        self.assertIn("Pcrc90", out_header)
        self.assertNotIn("Pn43", out_header)  # control excluded


if __name__ == "__main__":
    unittest.main()
