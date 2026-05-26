"""
Unit tests for LLM-assisted accession extraction pipeline.

Tests:
  1. Standard English accession extraction
  2. Chinese natural language description
  3. Implicit reference ("series GSE98765")
  4. Multiple accessions in one text
  5. No accession found → empty result
  6. Hallucination filtering (GEO API verification)
  7. Cache hit on second call with same DOI
  8. JSON parse failure → regex fallback
  9. PDFSectionExtractor: English section detection
  10. PDFSectionExtractor: Chinese section detection
  11. PDFSectionExtractor: fallback when no sections found
  12. Registry: needs_review column and pending_review status
  13. Registry: llm_extraction_cache table CRUD
  14. Registry: schema migration (safe no-op)
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_mock_llm(response_json: dict):
    """Create a mock LLM that returns a fixed JSON response."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(response_json)
    mock.invoke.return_value = mock_response
    return mock


def make_extraction_response(accessions: list, confidence: str = "high") -> dict:
    """Build a valid LLM extraction JSON response."""
    return {
        "extractions": [
            {
                "accession": acc,
                "database": "GEO" if acc.startswith("GSE") else "TCGA",
                "confidence": confidence,
                "evidence": f"test evidence for {acc}",
                "context": "methylation dataset",
            }
            for acc in accessions
        ],
        "summary": f"Found {len(accessions)} accession(s)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test: LLMAccessionExtractor
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMAccessionExtractor(unittest.TestCase):

    def setUp(self):
        from tools.llm_accession_extractor import LLMAccessionExtractor
        self.tmp = "/workspace/test_methyagent_1.db"
        self.ExtractorClass = LLMAccessionExtractor

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)

    def _make_extractor(self, llm):
        return self.ExtractorClass(
            llm=llm,
            cache_db_path=self.tmp,
            model_name="test-model",
        )

    # Test 1: Standard English
    def test_standard_english_extraction(self):
        """LLM correctly extracts explicitly stated GSE accession."""
        llm = make_mock_llm(make_extraction_response(["GSE124600"], "high"))
        extractor = self._make_extractor(llm)
        result = extractor.extract(
            "Data deposited in GEO under accession number GSE124600.",
            doi="10.1038/test1",
        )
        self.assertIn("GSE124600", result.high_confidence)
        self.assertEqual(result.error, None)
        self.assertFalse(result.cache_hit)

    # Test 2: Chinese natural language
    def test_chinese_natural_language(self):
        """LLM extracts accession from Chinese description."""
        llm = make_mock_llm(make_extraction_response(["GSE200234"], "high"))
        extractor = self._make_extractor(llm)
        result = extractor.extract(
            "数据存储于GEO数据库，编号为GSE200234，可公开获取。",
            doi="10.1016/test2",
        )
        self.assertIn("GSE200234", result.high_confidence)

    # Test 3: Implicit reference
    def test_implicit_reference(self):
        """LLM extracts accession from 'series GSE98765' phrasing."""
        llm = make_mock_llm(make_extraction_response(["GSE98765"], "high"))
        extractor = self._make_extractor(llm)
        result = extractor.extract(
            "Methylation data are available at NCBI GEO (series GSE98765).",
            doi="10.1093/test3",
        )
        self.assertIn("GSE98765", result.high_confidence)

    # Test 4: Multiple accessions
    def test_multiple_accessions(self):
        """LLM extracts multiple accessions from one text."""
        llm = make_mock_llm(make_extraction_response(["GSE100", "GSE200"], "high"))
        extractor = self._make_extractor(llm)
        result = extractor.extract(
            "GSE100 and GSE200 were deposited in GEO.",
            doi="10.1038/test4",
        )
        self.assertIn("GSE100", result.high_confidence)
        self.assertIn("GSE200", result.high_confidence)
        self.assertEqual(len(result.high_confidence), 2)

    # Test 5: No accession found
    def test_no_accession_found(self):
        """LLM returns empty list when no accession present."""
        llm = make_mock_llm({"extractions": [], "summary": "No accession numbers found"})
        extractor = self._make_extractor(llm)
        result = extractor.extract(
            "Data available upon reasonable request to the corresponding author.",
            doi="10.1038/test5",
        )
        self.assertEqual(result.high_confidence, [])
        self.assertEqual(result.medium_confidence, [])
        self.assertEqual(result.error, None)

    # Test 6: Hallucination filtering (medium confidence → pending_review)
    def test_medium_confidence_goes_to_pending_review(self):
        """Medium-confidence accessions go to pending_review, not auto_download."""
        llm = make_mock_llm(make_extraction_response(["GSE999999"], "medium"))
        extractor = self._make_extractor(llm)
        result = extractor.extract(
            "Methylation data may be available at GEO.",
            doi="10.1038/test6",
        )
        self.assertNotIn("GSE999999", result.high_confidence)
        self.assertIn("GSE999999", result.pending_review)

    # Test 7: Cache hit on second call
    def test_cache_hit_on_second_call(self):
        """Second call with same DOI returns cached result without calling LLM."""
        llm = make_mock_llm(make_extraction_response(["GSE111111"], "high"))
        extractor = self._make_extractor(llm)

        # First call — should call LLM
        result1 = extractor.extract("GSE111111 deposited in GEO.", doi="10.1038/cache-test")
        self.assertFalse(result1.cache_hit)
        self.assertEqual(llm.invoke.call_count, 1)

        # Second call — should hit cache
        result2 = extractor.extract("different text", doi="10.1038/cache-test")
        self.assertTrue(result2.cache_hit)
        self.assertEqual(llm.invoke.call_count, 1)  # LLM not called again

    # Test 8: JSON parse failure → regex fallback
    def test_json_parse_failure_regex_fallback(self):
        """When LLM returns non-JSON, regex fallback extracts accessions."""
        llm = MagicMock()
        mock_response = MagicMock()
        # Return malformed JSON but with a GSE accession in the text
        mock_response.content = "I found GSE123456 in the text. Here is my analysis..."
        llm.invoke.return_value = mock_response

        extractor = self._make_extractor(llm)
        result = extractor.extract(
            "Data deposited as GSE123456.",
            doi="10.1038/test8",
        )
        # Should not raise; may find GSE123456 via regex fallback
        self.assertIsNone(result.error)
        # GSE123456 may appear in medium_confidence (downgraded from JSON failure)
        all_found = result.high_confidence + result.medium_confidence
        self.assertIn("GSE123456", all_found)


# ─────────────────────────────────────────────────────────────────────────────
# Test: PDFSectionExtractor
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFSectionExtractor(unittest.TestCase):

    def setUp(self):
        from tools.pdf_section_extractor import PDFSectionExtractor
        self.extractor = PDFSectionExtractor()

    # Test 9: English section detection
    def test_english_data_availability_section(self):
        """Correctly extracts Data Availability section from English text."""
        text = (
            "Introduction\nThis study examines breast cancer methylation.\n\n"
            "Data Availability\n"
            "All methylation data are deposited in GEO under accession GSE124600.\n\n"
            "Methods\nSamples were processed using the EPIC array.\n"
        )
        result = self.extractor.extract(text)
        self.assertGreater(len(result.sections), 0)
        # Should find the data availability section
        section_names = [s.section_name for s in result.sections]
        self.assertTrue(
            any("data availability" in n for n in section_names),
            f"Expected 'data availability' in {section_names}"
        )
        # Should have trigger keywords (GEO, accession)
        triggered = result.trigger_sections
        self.assertGreater(len(triggered), 0)

    # Test 10: Chinese section detection
    def test_chinese_section_detection(self):
        """Correctly extracts Chinese 数据可用性 section."""
        text = (
            "引言\n本研究分析乳腺癌甲基化数据。\n\n"
            "数据可用性\n"
            "所有甲基化数据已存储于GEO数据库，编号为GSE200234。\n\n"
            "材料与方法\n使用EPIC芯片进行检测。\n"
        )
        result = self.extractor.extract(text)
        self.assertGreater(len(result.sections), 0)
        section_names = [s.section_name for s in result.sections]
        self.assertTrue(
            any("数据可用性" in n for n in section_names),
            f"Expected '数据可用性' in {section_names}"
        )

    # Test 11: Fallback when no sections found
    def test_fallback_when_no_sections(self):
        """Uses document start as fallback when no section headers found."""
        text = "This is a paper about methylation. GSE124600 was used. " * 50
        result = self.extractor.extract(text)
        self.assertTrue(result.fallback_used)
        self.assertGreater(len(result.sections), 0)
        self.assertGreater(len(result.best_text), 0)

    def test_empty_text_returns_empty_result(self):
        """Empty text returns empty result without error."""
        result = self.extractor.extract("")
        self.assertEqual(result.sections, [])
        self.assertEqual(result.total_chars_extracted, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Registry extensions
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryExtensions(unittest.TestCase):

    def setUp(self):
        from registry.registry import Registry
        self.tmp = "/workspace/test_methyagent_2.db"
        self.registry = Registry(self.tmp)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)

    # Test 12: needs_review column and pending_review status
    def test_needs_review_column(self):
        """Datasets can be registered with needs_review=True."""
        from registry.registry import Registry
        self.registry.upsert_dataset(
            accession="GSE999001",
            source="GEO",
            discovered_by="agent2_llm",
            download_status=Registry.STATUS_PENDING_REVIEW,
            needs_review=True,
            llm_evidence="LLM medium confidence: 'data may be at GEO'",
        )
        record = self.registry.get("GSE999001")
        self.assertIsNotNone(record)
        self.assertEqual(record["needs_review"], 1)
        self.assertEqual(record["download_status"], "pending_review")
        self.assertIn("LLM medium confidence", record["llm_evidence"])

        # get_pending_review should return it
        pending = self.registry.get_pending_review()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["accession"], "GSE999001")

    def test_approve_review(self):
        """approve_review clears needs_review and sets status to pending."""
        from registry.registry import Registry
        self.registry.upsert_dataset(
            accession="GSE999002",
            source="GEO",
            discovered_by="agent2_llm",
            download_status=Registry.STATUS_PENDING_REVIEW,
            needs_review=True,
        )
        self.registry.approve_review("GSE999002")
        record = self.registry.get("GSE999002")
        self.assertEqual(record["needs_review"], 0)
        self.assertEqual(record["download_status"], "pending")

    def test_reject_review(self):
        """reject_review clears needs_review and sets status to skipped."""
        from registry.registry import Registry
        self.registry.upsert_dataset(
            accession="GSE999003",
            source="GEO",
            discovered_by="agent2_llm",
            download_status=Registry.STATUS_PENDING_REVIEW,
            needs_review=True,
        )
        self.registry.reject_review("GSE999003")
        record = self.registry.get("GSE999003")
        self.assertEqual(record["needs_review"], 0)
        self.assertEqual(record["download_status"], "skipped")

    # Test 13: llm_extraction_cache table CRUD
    def test_llm_cache_crud(self):
        """LLM cache stores and retrieves results by DOI."""
        doi = "10.1038/test-cache"
        self.registry.cache_llm_result(
            doi=doi,
            accessions=["GSE111", "GSE222"],
            extracted_json='{"extractions": []}',
            pdf_url="https://example.com/paper.pdf",
            model_used="gpt-4o",
        )
        cached = self.registry.get_llm_cache(doi)
        self.assertIsNotNone(cached)
        self.assertIn("GSE111", cached["accessions"])
        self.assertIn("GSE222", cached["accessions"])
        self.assertEqual(cached["model_used"], "gpt-4o")
        self.assertEqual(cached["hit_count"], 1)

        # Second retrieval increments hit_count
        cached2 = self.registry.get_llm_cache(doi)
        self.assertEqual(cached2["hit_count"], 2)

    def test_llm_cache_miss_returns_none(self):
        """Cache miss returns None."""
        result = self.registry.get_llm_cache("10.1038/nonexistent")
        self.assertIsNone(result)

    def test_clear_llm_cache(self):
        """clear_llm_cache removes entries."""
        self.registry.cache_llm_result(
            doi="10.1038/to-clear",
            accessions=["GSE333"],
        )
        deleted = self.registry.clear_llm_cache("10.1038/to-clear")
        self.assertEqual(deleted, 1)
        self.assertIsNone(self.registry.get_llm_cache("10.1038/to-clear"))

    # Test 14: Schema migration (safe no-op)
    def test_schema_migration_safe_noop(self):
        """Running _migrate_schema() twice does not raise errors."""
        # Should not raise even if columns already exist
        try:
            self.registry._migrate_schema()
            self.registry._migrate_schema()
        except Exception as e:
            self.fail(f"Schema migration raised: {e}")

    def test_summary_includes_pending_review(self):
        """get_summary() includes pending_review count."""
        from registry.registry import Registry
        self.registry.upsert_dataset(
            accession="GSE888001",
            source="GEO",
            discovered_by="agent2_llm",
            download_status=Registry.STATUS_PENDING_REVIEW,
            needs_review=True,
        )
        summary = self.registry.get_summary()
        self.assertIn("pending_review", summary)
        self.assertEqual(summary["pending_review"], 1)
        self.assertIn("llm_cache_entries", summary)


# ─────────────────────────────────────────────────────────────────────────────
# Test: LLMExtractionCache (standalone)
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMExtractionCache(unittest.TestCase):

    def setUp(self):
        from tools.llm_accession_extractor import LLMExtractionCache
        self.tmp = "/workspace/test_methyagent_3.db"
        self.cache = LLMExtractionCache(self.tmp)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)

    def test_put_and_get(self):
        self.cache.put(
            doi="10.1038/x",
            pdf_url="https://example.com/x.pdf",
            extracted_json='{"extractions": []}',
            accessions=["GSE100", "GSE200"],
            model_used="gpt-4o",
        )
        result = self.cache.get("10.1038/x")
        self.assertIsNotNone(result)
        self.assertIn("GSE100", result["accessions"])

    def test_get_miss(self):
        self.assertIsNone(self.cache.get("10.1038/missing"))

    def test_stats(self):
        self.cache.put("10.1038/a", "", "", ["GSE1"], "gpt-4o")
        self.cache.put("10.1038/b", "", "", ["GSE2"], "gpt-4o")
        stats = self.cache.stats()
        self.assertEqual(stats["entries"], 2)

    def test_clear_all(self):
        self.cache.put("10.1038/c", "", "", [], "gpt-4o")
        deleted = self.cache.clear()
        self.assertEqual(deleted, 1)
        self.assertIsNone(self.cache.get("10.1038/c"))


# ─────────────────────────────────────────────────────────────────────────────
# Test: GEOClient verify methods (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestGEOClientVerify(unittest.TestCase):

    def setUp(self):
        from tools.geo_tools import GEOClient
        self.client = GEOClient()

    def _mock_esearch_response(self, count: int, idlist: list = None):
        """Build a mock esearch JSON response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "esearchresult": {
                "count": str(count),
                "idlist": idlist or (["12345"] if count > 0 else []),
            }
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_verify_existing_accession(self):
        """verify_accession returns True for existing GSE."""
        with patch.object(self.client, "_get", return_value=self._mock_esearch_response(1)):
            result = self.client.verify_accession("GSE124600")
        self.assertTrue(result)

    def test_verify_nonexistent_accession(self):
        """verify_accession returns False for non-existent GSE."""
        with patch.object(self.client, "_get", return_value=self._mock_esearch_response(0)):
            result = self.client.verify_accession("GSE999999999")
        self.assertFalse(result)

    def test_verify_unknown_prefix_returns_false(self):
        """verify_accession returns False for unknown prefix (cannot verify = do not auto-download)."""
        result = self.client.verify_accession("UNKNOWN123")
        self.assertFalse(result)

    def test_verify_empty_string(self):
        """verify_accession returns False for empty string."""
        result = self.client.verify_accession("")
        self.assertFalse(result)

    def test_batch_verify_empty_list(self):
        """batch_verify_accessions returns empty dict for empty input."""
        result = self.client.batch_verify_accessions([])
        self.assertEqual(result, {})


# ─────────────────────────────────────────────────────────────────────────────
# Test: AST syntax check for all new files
# ─────────────────────────────────────────────────────────────────────────────

class TestASTSyntaxNewFiles(unittest.TestCase):

    NEW_FILES = [
        "tools/pdf_section_extractor.py",
        "tools/llm_accession_extractor.py",
    ]

    UPDATED_FILES = [
        "tools/geo_tools.py",
        "tools/pubmed_tools.py",
        "registry/registry.py",
        "agents/literature_agent.py",
        "config/settings.yaml",  # YAML, not Python — skip AST
    ]

    def _get_project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_new_files_syntax(self):
        """All new Python files pass AST syntax check."""
        import ast
        root = self._get_project_root()
        for rel_path in self.NEW_FILES:
            full_path = os.path.join(root, rel_path)
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            try:
                ast.parse(source)
            except SyntaxError as e:
                self.fail(f"Syntax error in {rel_path}: {e}")

    def test_updated_files_syntax(self):
        """All updated Python files pass AST syntax check."""
        import ast
        root = self._get_project_root()
        for rel_path in self.UPDATED_FILES:
            if rel_path.endswith(".yaml"):
                continue  # Skip YAML files
            full_path = os.path.join(root, rel_path)
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            try:
                ast.parse(source)
            except SyntaxError as e:
                self.fail(f"Syntax error in {rel_path}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
