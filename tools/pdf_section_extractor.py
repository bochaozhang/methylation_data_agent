"""
pdf_section_extractor.py
------------------------
Chapter locator for PDF supplementary material text.

Extracts target sections (Data Availability, Methods, Accession Numbers, etc.)
from raw PDF text to minimize tokens sent to the LLM.

Supports both English and Chinese section headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Target section definitions (ordered by priority)
# ---------------------------------------------------------------------------

TARGET_SECTIONS: List[str] = [
    # English – highest priority
    "data availability statement",
    "data availability",
    "data access",
    "availability of data and materials",
    "data and code availability",
    "accession numbers",
    "accession codes",
    "data deposition",
    "data deposit",
    "database accession",
    # Nature / Cell specific
    "key resources table",
    "reporting summary",
    "extended data",
    # Methods sections
    "materials and methods",
    "methods and materials",
    "supplementary methods",
    "experimental procedures",
    "methods",
    # Chinese equivalents
    "数据可用性声明",
    "数据可用性",
    "数据获取",
    "数据存储",
    "数据访问",
    "登录号",
    "登录编号",
    "数据库编号",
    "材料与方法",
    "实验方法",
    "补充方法",
]

# Keywords that must appear in a section for it to be worth sending to LLM
TRIGGER_KEYWORDS: List[str] = [
    # English
    "geo", "gse", "gsm", "gpl", "gds",
    "accession", "deposited", "available at", "available from",
    "repository", "database", "ncbi", "tcga",
    "arrayexpress", "e-mtab", "e-geod",
    "sra", "srp", "srr", "srx",
    "dbgap", "phs",
    "figshare", "zenodo", "dryad", "github",
    "methylation data", "methylation array",
    # Chinese
    "数据库", "编号", "存储", "获取", "登录",
    "基因表达综合数据库", "甲基化数据",
]

# Maximum characters to extract per section
MAX_SECTION_CHARS: int = 3000

# Fallback: first N characters of full text when no section found
FALLBACK_CHARS: int = 3000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSection:
    """A single extracted section from a PDF."""
    section_name: str          # Matched section header
    text: str                  # Section body text
    page_hint: Optional[int]   # Approximate page number (0-indexed), if known
    has_trigger: bool          # Whether trigger keywords were found
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class SectionExtractionResult:
    """Result of section extraction from a full PDF text."""
    sections: List[ExtractedSection]
    fallback_used: bool
    total_chars_extracted: int
    source_text_chars: int

    @property
    def trigger_sections(self) -> List[ExtractedSection]:
        """Return only sections that contain trigger keywords."""
        return [s for s in self.sections if s.has_trigger]

    @property
    def best_text(self) -> str:
        """
        Return the best text to send to LLM:
        - Prefer trigger sections (joined)
        - Fall back to all sections
        - Fall back to whatever was extracted
        """
        triggered = self.trigger_sections
        if triggered:
            return "\n\n---\n\n".join(s.text for s in triggered)[:MAX_SECTION_CHARS * 2]
        if self.sections:
            return "\n\n---\n\n".join(s.text for s in self.sections)[:MAX_SECTION_CHARS * 2]
        return ""


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class PDFSectionExtractor:
    """
    Locates and extracts target sections from raw PDF text.

    Usage::

        extractor = PDFSectionExtractor()
        result = extractor.extract(full_text)
        llm_input = result.best_text
    """

    def __init__(
        self,
        target_sections: Optional[List[str]] = None,
        trigger_keywords: Optional[List[str]] = None,
        max_section_chars: int = MAX_SECTION_CHARS,
        fallback_chars: int = FALLBACK_CHARS,
    ) -> None:
        self.target_sections = [s.lower() for s in (target_sections or TARGET_SECTIONS)]
        self.trigger_keywords = [k.lower() for k in (trigger_keywords or TRIGGER_KEYWORDS)]
        self.max_section_chars = max_section_chars
        self.fallback_chars = fallback_chars

        # Pre-compile section header patterns
        self._section_patterns: List[Tuple[str, re.Pattern]] = []
        for sec in self.target_sections:
            # Match section header at start of line, optionally followed by
            # numbering (e.g., "3. Methods", "Methods:", "METHODS")
            escaped = re.escape(sec)
            pattern = re.compile(
                r"(?im)"                          # ignore case, multiline
                r"^"                              # start of line
                r"(?:\d+[\.\s]+)?"               # optional numbering
                r"(?:" + escaped + r")"
                r"[\s:：\.\-]*$",                 # optional trailing punctuation
            )
            self._section_patterns.append((sec, pattern))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str, pages: Optional[List[str]] = None) -> SectionExtractionResult:
        """
        Extract target sections from PDF text.

        Parameters
        ----------
        text : str
            Full concatenated PDF text.
        pages : list of str, optional
            Per-page text list. If provided, page hints are included in results.

        Returns
        -------
        SectionExtractionResult
        """
        if not text or not text.strip():
            return SectionExtractionResult(
                sections=[],
                fallback_used=False,
                total_chars_extracted=0,
                source_text_chars=0,
            )

        sections = self._find_sections(text, pages)

        fallback_used = False
        if not sections:
            # No section headers found — use beginning of document
            fallback_text = text[: self.fallback_chars].strip()
            if fallback_text:
                has_trigger = self._has_trigger_keywords(fallback_text)
                sections = [
                    ExtractedSection(
                        section_name="[fallback: document start]",
                        text=fallback_text,
                        page_hint=None,
                        has_trigger=has_trigger,
                    )
                ]
                fallback_used = True

        total_chars = sum(s.char_count for s in sections)
        return SectionExtractionResult(
            sections=sections,
            fallback_used=fallback_used,
            total_chars_extracted=total_chars,
            source_text_chars=len(text),
        )

    def has_data_availability_content(self, text: str) -> bool:
        """
        Quick check: does this text likely contain data availability info?
        Returns True if any trigger keyword is present.
        """
        return self._has_trigger_keywords(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_sections(
        self, text: str, pages: Optional[List[str]]
    ) -> List[ExtractedSection]:
        """
        Find all target sections in text.
        Returns list of ExtractedSection ordered by appearance in document.
        """
        # Build a list of (position, section_name, match) for all hits
        hits: List[Tuple[int, str, re.Match]] = []
        for sec_name, pattern in self._section_patterns:
            for m in pattern.finditer(text):
                hits.append((m.start(), sec_name, m))

        if not hits:
            return []

        # Sort by position
        hits.sort(key=lambda x: x[0])

        # Deduplicate: keep first hit per section name
        seen_names: set = set()
        unique_hits: List[Tuple[int, str, re.Match]] = []
        for pos, name, m in hits:
            if name not in seen_names:
                seen_names.add(name)
                unique_hits.append((pos, name, m))

        # Extract text between consecutive section headers
        sections: List[ExtractedSection] = []
        for i, (pos, name, m) in enumerate(unique_hits):
            # Section body starts after the header line
            body_start = m.end()

            # Section body ends at the next section header (or end of text)
            if i + 1 < len(unique_hits):
                next_pos = unique_hits[i + 1][0]
                body_end = min(body_start + self.max_section_chars, next_pos)
            else:
                body_end = body_start + self.max_section_chars

            body = text[body_start:body_end].strip()
            if not body:
                continue

            # Estimate page number
            page_hint = self._estimate_page(pos, text, pages)

            has_trigger = self._has_trigger_keywords(body)
            sections.append(
                ExtractedSection(
                    section_name=name,
                    text=body,
                    page_hint=page_hint,
                    has_trigger=has_trigger,
                )
            )

        return sections

    def _has_trigger_keywords(self, text: str) -> bool:
        """Return True if any trigger keyword appears in text (case-insensitive)."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.trigger_keywords)

    @staticmethod
    def _estimate_page(
        char_pos: int, full_text: str, pages: Optional[List[str]]
    ) -> Optional[int]:
        """Estimate which page a character position falls on."""
        if pages is None:
            return None
        cumulative = 0
        for i, page_text in enumerate(pages):
            cumulative += len(page_text)
            if char_pos < cumulative:
                return i
        return len(pages) - 1


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_relevant_sections(
    text: str,
    pages: Optional[List[str]] = None,
    max_section_chars: int = MAX_SECTION_CHARS,
) -> SectionExtractionResult:
    """
    Module-level convenience wrapper around PDFSectionExtractor.

    Parameters
    ----------
    text : str
        Full PDF text.
    pages : list of str, optional
        Per-page text for page hint calculation.
    max_section_chars : int
        Maximum characters per section.

    Returns
    -------
    SectionExtractionResult
    """
    extractor = PDFSectionExtractor(max_section_chars=max_section_chars)
    return extractor.extract(text, pages)
