"""
llm_accession_extractor.py
--------------------------
Layer 2 of the three-layer PDF accession extraction pipeline.

Uses an LLM to extract dataset accession numbers from scientific text,
handling non-standard natural language descriptions such as:
  - "data deposited in GEO under accession number GSE124600"
  - "数据存储于GEO数据库，编号为GSE200234"
  - "methylation data are available at NCBI GEO (series GSE98765)"

Features:
  - Bilingual prompt (English + Chinese)
  - Three confidence levels: high / medium / low
  - DOI-keyed SQLite cache to avoid redundant LLM calls
  - Robust JSON parsing with regex fallback
  - Graceful degradation on LLM timeout / parse failure
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_LEVELS = ("high", "medium", "low")

# Actions keyed by confidence level
CONFIDENCE_ACTIONS: Dict[str, str] = {
    "high":   "auto_download",
    "medium": "pending_review",
    "low":    "discard",
}

# Databases we care about
SUPPORTED_DATABASES = [
    "GEO", "TCGA", "ArrayExpress", "SRA", "dbGaP",
    "Figshare", "Zenodo", "Dryad", "GitHub",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a biomedical data extraction specialist. Your task is to extract \
dataset accession numbers from scientific paper text.

Extract ALL database accession numbers mentioned, including:
- GEO: GSE/GSM/GPL/GDS numbers (e.g., GSE124600, GSM1234567)
- TCGA: project codes (e.g., TCGA-BRCA, TCGA-LUAD)
- ArrayExpress: E-MTAB-XXXX, E-GEOD-XXXX
- SRA: SRP/SRR/SRX numbers
- dbGaP: phs numbers (e.g., phs000178)
- Figshare/Zenodo: DOIs (e.g., 10.6084/..., 10.5281/...)

IMPORTANT: Also extract accessions described in natural language, such as:
- "data deposited in GEO under accession number GSE..."
- "raw data available at NCBI GEO (GSE...)"
- "数据存储于GEO数据库，编号为GSE..."
- "the dataset is accessible at GEO with the identifier GSE..."
- "methylation data were submitted to GEO (accession: GSE...)"
- Cases where the accession appears NEAR words like "GEO", "deposited", \
"accession", "available", "submitted", "repository"

Output ONLY valid JSON. No explanation text outside the JSON."""

USER_PROMPT_TEMPLATE = """Extract all dataset accession numbers from this text.
For each accession found, assess your confidence.

Text:
{section_text}

Output JSON format:
{{
  "extractions": [
    {{
      "accession": "GSE124600",
      "database": "GEO",
      "confidence": "high",
      "evidence": "exact quote from text containing the accession",
      "context": "brief description of what data this accession refers to"
    }}
  ],
  "summary": "one sentence describing what datasets were found"
}}

Confidence levels:
- "high": accession string is explicitly present in text (e.g., "GSE124600")
- "medium": accession is strongly implied but may have OCR errors or formatting issues
- "low": database is mentioned but no specific accession number found

If no accessions found, return: {{"extractions": [], "summary": "No accession numbers found"}}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AccessionExtraction:
    """A single accession extracted by the LLM."""
    accession: str
    database: str
    confidence: str          # "high" | "medium" | "low"
    evidence: str            # Quote from source text
    context: str             # Brief description
    action: str = field(init=False)

    def __post_init__(self) -> None:
        self.confidence = self.confidence.lower()
        if self.confidence not in CONFIDENCE_LEVELS:
            self.confidence = "low"
        self.action = CONFIDENCE_ACTIONS[self.confidence]


@dataclass
class LLMExtractionResult:
    """Full result from one LLM extraction call."""
    extractions: List[AccessionExtraction]
    summary: str
    model_used: str
    cache_hit: bool
    doi: str
    elapsed_seconds: float
    raw_response: str = ""
    error: Optional[str] = None

    @property
    def high_confidence(self) -> List[str]:
        return [e.accession for e in self.extractions if e.confidence == "high"]

    @property
    def medium_confidence(self) -> List[str]:
        return [e.accession for e in self.extractions if e.confidence == "medium"]

    @property
    def low_confidence(self) -> List[str]:
        return [e.accession for e in self.extractions if e.confidence == "low"]

    @property
    def auto_download(self) -> List[str]:
        return [e.accession for e in self.extractions if e.action == "auto_download"]

    @property
    def pending_review(self) -> List[str]:
        return [e.accession for e in self.extractions if e.action == "pending_review"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_confidence": self.high_confidence,
            "pending_review": self.pending_review,
            "low_confidence": self.low_confidence,
            "summary": self.summary,
            "model_used": self.model_used,
            "cache_hit": self.cache_hit,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class LLMExtractionCache:
    """
    SQLite-backed DOI-keyed cache for LLM extraction results.

    Prevents redundant LLM calls for the same paper across sessions.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS llm_extraction_cache (
        doi             TEXT PRIMARY KEY,
        pdf_url         TEXT,
        extracted_json  TEXT,
        accessions      TEXT,
        model_used      TEXT,
        created_at      TEXT,
        hit_count       INTEGER DEFAULT 0
    )
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            conn.commit()

    def get(self, doi: str) -> Optional[Dict[str, Any]]:
        """Return cached result for DOI, or None if not cached."""
        if not doi:
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT extracted_json, accessions, model_used, hit_count FROM llm_extraction_cache WHERE doi = ?",
                (doi,),
            ).fetchone()
            if row:
                # Increment hit count
                conn.execute(
                    "UPDATE llm_extraction_cache SET hit_count = hit_count + 1 WHERE doi = ?",
                    (doi,),
                )
                conn.commit()
                return {
                    "extracted_json": row[0],
                    "accessions": json.loads(row[1]) if row[1] else [],
                    "model_used": row[2],
                    "hit_count": row[3] + 1,
                }
        return None

    def put(
        self,
        doi: str,
        pdf_url: str,
        extracted_json: str,
        accessions: List[str],
        model_used: str,
    ) -> None:
        """Store extraction result in cache."""
        if not doi:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_extraction_cache
                   (doi, pdf_url, extracted_json, accessions, model_used, created_at, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (
                    doi,
                    pdf_url,
                    extracted_json,
                    json.dumps(accessions),
                    model_used,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def clear(self, doi: Optional[str] = None) -> int:
        """Clear cache. If doi given, clear only that entry. Returns rows deleted."""
        with sqlite3.connect(self.db_path) as conn:
            if doi:
                cur = conn.execute(
                    "DELETE FROM llm_extraction_cache WHERE doi = ?", (doi,)
                )
            else:
                cur = conn.execute("DELETE FROM llm_extraction_cache")
            conn.commit()
            return cur.rowcount

    def stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(hit_count) FROM llm_extraction_cache"
            ).fetchone()
            return {"entries": row[0] or 0, "total_hits": row[1] or 0}


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class LLMAccessionExtractor:
    """
    LLM-powered accession number extractor (Layer 2 of the pipeline).

    Parameters
    ----------
    llm : langchain BaseChatModel
        Any LangChain-compatible chat model (OpenAI, Anthropic, Ollama, etc.)
    cache_db_path : str
        Path to SQLite database for caching results.
    model_name : str
        Human-readable model name for logging/cache metadata.
    max_retries : int
        Number of LLM call retries on timeout/error.
    timeout_seconds : int
        Per-call timeout in seconds.
    """

    def __init__(
        self,
        llm: Any,
        cache_db_path: str = "/workspace/methyagent_llm_cache.db",
        model_name: str = "unknown",
        max_retries: int = 1,
        timeout_seconds: int = 30,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.cache = LLMExtractionCache(cache_db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        doi: str = "",
        pdf_url: str = "",
    ) -> LLMExtractionResult:
        """
        Extract accession numbers from text using LLM.

        Parameters
        ----------
        text : str
            Section text to analyze (ideally from PDFSectionExtractor).
        doi : str
            Paper DOI used as cache key.
        pdf_url : str
            Source URL for cache metadata.

        Returns
        -------
        LLMExtractionResult
        """
        start = time.time()

        # --- Cache check ---
        if doi:
            cached = self.cache.get(doi)
            if cached:
                logger.info("LLM cache hit for DOI: %s", doi)
                return self._result_from_cache(cached, doi, start)

        # --- LLM call ---
        raw_response = ""
        error = None
        extractions: List[AccessionExtraction] = []
        summary = ""

        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self._call_llm(text)
                parsed = self._parse_response(raw_response)
                extractions = parsed["extractions"]
                summary = parsed["summary"]
                break
            except Exception as exc:
                error = str(exc)
                logger.warning(
                    "LLM extraction attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2)

        elapsed = time.time() - start

        # --- Cache store ---
        if doi and not error:
            all_accessions = [e.accession for e in extractions]
            self.cache.put(
                doi=doi,
                pdf_url=pdf_url,
                extracted_json=raw_response,
                accessions=all_accessions,
                model_used=self.model_name,
            )

        result = LLMExtractionResult(
            extractions=extractions,
            summary=summary,
            model_used=self.model_name,
            cache_hit=False,
            doi=doi,
            elapsed_seconds=elapsed,
            raw_response=raw_response,
            error=error,
        )

        logger.info(
            "LLM extraction: doi=%s high=%d medium=%d low=%d elapsed=%.1fs",
            doi or "N/A",
            len(result.high_confidence),
            len(result.medium_confidence),
            len(result.low_confidence),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, text: str) -> str:
        """Call the LLM and return raw string response."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT_TEMPLATE.format(section_text=text)),
        ]
        response = self.llm.invoke(messages)
        # Handle both string and AIMessage responses
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured data.

        Tries direct JSON parse first, then regex extraction of JSON block,
        then returns empty result on failure.
        """
        if not raw or not raw.strip():
            return {"extractions": [], "summary": "Empty LLM response"}

        # Attempt 1: direct JSON parse
        try:
            data = json.loads(raw.strip())
            return self._validate_parsed(data)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract JSON block from response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_parsed(data)
            except json.JSONDecodeError:
                pass

        # Attempt 3: extract accession-like strings directly from raw text
        logger.warning("JSON parse failed; falling back to regex on LLM output")
        accessions = self._regex_fallback(raw)
        extractions = [
            AccessionExtraction(
                accession=acc,
                database=self._guess_database(acc),
                confidence="medium",  # downgrade since JSON failed
                evidence="[extracted from malformed LLM response]",
                context="",
            )
            for acc in accessions
        ]
        return {
            "extractions": extractions,
            "summary": f"Extracted {len(accessions)} accessions via regex fallback",
        }

    def _validate_parsed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parsed JSON data."""
        raw_extractions = data.get("extractions", [])
        extractions: List[AccessionExtraction] = []

        for item in raw_extractions:
            if not isinstance(item, dict):
                continue
            accession = str(item.get("accession", "")).strip()
            if not accession:
                continue
            extractions.append(
                AccessionExtraction(
                    accession=accession,
                    database=str(item.get("database", "Unknown")),
                    confidence=str(item.get("confidence", "low")),
                    evidence=str(item.get("evidence", "")),
                    context=str(item.get("context", "")),
                )
            )

        return {
            "extractions": extractions,
            "summary": str(data.get("summary", "")),
        }

    @staticmethod
    def _regex_fallback(text: str) -> List[str]:
        """Extract accession-like strings from raw text using regex."""
        patterns = [
            r"(?<![A-Za-z0-9])(GSE\d{4,8})(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])(GSM\d{4,8})(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])(GPL\d{4,8})(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])(GDS\d{4,8})(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])(TCGA-[A-Z]{2,4})(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])(E-[A-Z]{4}-\d{4,6})(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])(SRP\d{6,9})(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])(phs\d{6,9})(?![A-Za-z0-9])",
        ]
        found = []
        for pat in patterns:
            found.extend(re.findall(pat, text, re.IGNORECASE))
        return list(dict.fromkeys(found))  # deduplicate, preserve order

    @staticmethod
    def _guess_database(accession: str) -> str:
        """Guess database from accession prefix."""
        acc_upper = accession.upper()
        if acc_upper.startswith(("GSE", "GSM", "GPL", "GDS")):
            return "GEO"
        if acc_upper.startswith("TCGA-"):
            return "TCGA"
        if acc_upper.startswith("E-"):
            return "ArrayExpress"
        if acc_upper.startswith(("SRP", "SRR", "SRX")):
            return "SRA"
        if acc_upper.startswith("PHS"):
            return "dbGaP"
        return "Unknown"

    def _result_from_cache(
        self, cached: Dict[str, Any], doi: str, start: float
    ) -> LLMExtractionResult:
        """Reconstruct LLMExtractionResult from cache entry."""
        # Re-parse the stored JSON to get full extraction objects
        raw_json = cached.get("extracted_json", "")
        extractions: List[AccessionExtraction] = []
        summary = ""

        if raw_json:
            try:
                parsed = self._validate_parsed(json.loads(raw_json))
                extractions = parsed["extractions"]
                summary = parsed["summary"]
            except Exception:
                # Fallback: reconstruct minimal extractions from accession list
                for acc in cached.get("accessions", []):
                    extractions.append(
                        AccessionExtraction(
                            accession=acc,
                            database=self._guess_database(acc),
                            confidence="high",
                            evidence="[from cache]",
                            context="",
                        )
                    )

        return LLMExtractionResult(
            extractions=extractions,
            summary=summary or "Loaded from cache",
            model_used=cached.get("model_used", "unknown"),
            cache_hit=True,
            doi=doi,
            elapsed_seconds=time.time() - start,
            raw_response=raw_json,
            error=None,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_accessions_with_llm(
    text: str,
    llm: Any,
    doi: str = "",
    pdf_url: str = "",
    cache_db_path: str = "/workspace/methyagent_llm_cache.db",
    model_name: str = "unknown",
) -> LLMExtractionResult:
    """
    Module-level convenience wrapper.

    Parameters
    ----------
    text : str
        Section text to analyze.
    llm : BaseChatModel
        LangChain-compatible chat model.
    doi : str
        Paper DOI for caching.
    pdf_url : str
        Source URL for cache metadata.
    cache_db_path : str
        Path to SQLite cache database.
    model_name : str
        Model identifier for logging.

    Returns
    -------
    LLMExtractionResult
    """
    extractor = LLMAccessionExtractor(
        llm=llm,
        cache_db_path=cache_db_path,
        model_name=model_name,
    )
    return extractor.extract(text, doi=doi, pdf_url=pdf_url)
