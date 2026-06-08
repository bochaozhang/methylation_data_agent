"""
Keyword parser and accession extractor for MethyAgent.

Handles two input modes:
  1. Semantic search: "EPIC平台在2024年的乳腺癌相关数据"
  2. Exact accession:  "下载GEO编号GSE124600的所有数据"

The LLM extracts structured intent; regex handles accession extraction
from free text (paper abstracts, supplementary materials, etc.).
"""
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# ------------------------------------------------------------------ #
#  Accession regex patterns                                            #
# ------------------------------------------------------------------ #

# GEO series (GSE), samples (GSM), platforms (GPL), datasets (GDS)
# Use Unicode-safe boundaries ((?<![A-Za-z0-9]) / (?![A-Za-z0-9])) instead of \b
# so that accessions embedded in Chinese text are correctly detected.
GEO_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(GSE\d{4,8}|GSM\d{4,8}|GPL\d{3,7}|GDS\d{3,7})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# TCGA project codes (e.g. TCGA-BRCA, TCGA-LUAD)
TCGA_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(TCGA-[A-Z]{2,4})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# ArrayExpress accessions (E-MTAB-XXXX, E-GEOD-XXXX)
AE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(E-[A-Z]{4}-\d{3,7})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# DOI pattern
DOI_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(10\.\d{4,9}/[^\s\"\'<>]+)",
    re.IGNORECASE,
)

# Platform keywords → canonical names
PLATFORM_MAP = {
    "450k": "450K",
    "450": "450K",
    "hm450": "450K",
    "humanmethylation450": "450K",
    "epic": "EPIC",
    "850k": "EPIC",
    "hm850": "EPIC",
    "humanmethylationepic": "EPIC",
    "wgbs": "WGBS",
    "全基因组亚硫酸盐": "WGBS",
    "rrbs": "RRBS",
    "简化亚硫酸盐": "RRBS",
    "bisulfite": "WGBS",
}

# Cancer type keywords → canonical TCGA project codes
CANCER_MAP = {
    "乳腺癌": "BRCA",
    "breast": "BRCA",
    "brca": "BRCA",
    "肺癌": "LUAD",
    "lung": "LUAD",
    "luad": "LUAD",
    "lusc": "LUSC",
    "结直肠癌": "COAD",
    "colorectal": "COAD",
    "colon": "COAD",
    "coad": "COAD",
    "肝癌": "LIHC",
    "liver": "LIHC",
    "lihc": "LIHC",
    "胃癌": "STAD",
    "gastric": "STAD",
    "stomach": "STAD",
    "stad": "STAD",
    "前列腺癌": "PRAD",
    "prostate": "PRAD",
    "prad": "PRAD",
    "卵巢癌": "OV",
    "ovarian": "OV",
    "ov": "OV",
    "宫颈癌": "CESC",
    "cervical": "CESC",
    "cesc": "CESC",
    "胰腺癌": "PAAD",
    "pancreatic": "PAAD",
    "paad": "PAAD",
    "膀胱癌": "BLCA",
    "bladder": "BLCA",
    "blca": "BLCA",
    "肾癌": "KIRC",
    "renal": "KIRC",
    "kidney": "KIRC",
    "kirc": "KIRC",
    "甲状腺癌": "THCA",
    "thyroid": "THCA",
    "thca": "THCA",
    "黑色素瘤": "SKCM",
    "melanoma": "SKCM",
    "skcm": "SKCM",
    "胶质瘤": "GBM",
    "glioma": "GBM",
    "gbm": "GBM",
    "白血病": "LAML",
    "leukemia": "LAML",
    "laml": "LAML",
}


# ------------------------------------------------------------------ #
#  Regex-based accession extraction (no LLM needed)                   #
# ------------------------------------------------------------------ #

def extract_accessions(text: str) -> Dict[str, List[str]]:
    """
    Extract all database accession numbers from free text.

    Args:
        text: Any text (paper abstract, supplementary material, user query).

    Returns:
        Dict with keys 'geo', 'tcga', 'arrayexpress', 'dois'.
    """
    geo = list({m.upper() for m in GEO_PATTERN.findall(text)})
    tcga = list({m.upper() for m in TCGA_PATTERN.findall(text)})
    ae = list({m.upper() for m in AE_PATTERN.findall(text)})
    dois = list({m.lower() for m in DOI_PATTERN.findall(text)})

    return {
        "geo": sorted(geo),
        "tcga": sorted(tcga),
        "arrayexpress": sorted(ae),
        "dois": sorted(dois),
    }


def has_explicit_accession(query: str) -> bool:
    """Return True if the query contains at least one explicit accession number."""
    result = extract_accessions(query)
    return any(result[k] for k in ("geo", "tcga", "arrayexpress"))


# ------------------------------------------------------------------ #
#  Rule-based intent parsing (fast path, no LLM)                      #
# ------------------------------------------------------------------ #

def _extract_year_range(text: str):
    """
    Extract year or year range from text. Returns (start, end) or None.
    Uses Unicode-safe boundaries (no \\b) to handle Chinese text like '2024年'.
    """
    # Range: 2020-2023 or 2020~2023 or 2020至2023 or 2020到2023
    range_match = re.search(r"(?<!\d)(20\d{2})\s*[-~至到]\s*(20\d{2})(?!\d)", text)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))
    # Single year (e.g. "2024年" or "in 2024" or "2024")
    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", text)
    if year_match:
        y = int(year_match.group(1))
        return y, y
    return None


def _extract_platform(text: str) -> Optional[str]:
    """Extract methylation platform from text."""
    text_lower = text.lower()
    for keyword, canonical in PLATFORM_MAP.items():
        if keyword in text_lower:
            return canonical
    return None


def _extract_cancer_type(text: str):
    """Extract cancer type and return (display_name, tcga_code)."""
    text_lower = text.lower()
    for keyword, code in CANCER_MAP.items():
        if keyword in text_lower:
            # Return the original keyword as display name
            return keyword, code
    return None, None


def parse_query_rules(query: str) -> Dict[str, Any]:
    """
    Fast rule-based query parser (no LLM).
    Used as fallback or pre-processing before LLM parsing.

    Returns:
        Structured intent dict.
    """
    accessions = extract_accessions(query)
    year_range = _extract_year_range(query)
    platform = _extract_platform(query)
    cancer_display, cancer_code = _extract_cancer_type(query)

    return {
        "raw_query": query,
        "mode": "accession" if has_explicit_accession(query) else "semantic",
        "accessions": accessions,
        "platform": platform,
        "cancer_type_display": cancer_display,
        "cancer_type_code": cancer_code,
        "year_start": year_range[0] if year_range else None,
        "year_end": year_range[1] if year_range else None,
    }


# ------------------------------------------------------------------ #
#  LLM-enhanced intent parsing                                         #
# ------------------------------------------------------------------ #

PARSE_SYSTEM_PROMPT = """You are a biomedical data retrieval assistant specializing in DNA methylation datasets.

Your task is to parse a user query into a structured JSON object for searching TCGA and GEO databases.

Output ONLY valid JSON with these fields (use null for missing values):
{
  "mode": "accession" | "semantic",
  "accessions": {
    "geo": ["GSE..."],
    "tcga": ["TCGA-..."],
    "arrayexpress": []
  },
  "cancer_type": {
    "display": "breast cancer",
    "tcga_code": "BRCA",
    "mesh_term": "Breast Neoplasms"
  },
  "platform": "450K" | "EPIC" | "WGBS" | "RRBS" | null,
  "data_type": "array" | "sequencing" | "both" | null,
  "year_start": 2024,
  "year_end": 2024,
  "sample_type": "tumor" | "normal" | "both" | null,
  "geo_search_query": "breast cancer EPIC methylation[GEO]",
  "pubmed_search_query": "breast cancer DNA methylation EPIC array 2024",
  "notes": "any special instructions"
}

Rules:
- If the query contains explicit accession numbers (GSE..., TCGA-...), set mode="accession"
- For Chinese cancer names: 乳腺癌=BRCA, 肺癌=LUAD, 肝癌=LIHC, 胃癌=STAD, 结直肠癌=COAD
- For platform: EPIC/850K → "EPIC", 450K/HM450 → "450K", WGBS/全基因组亚硫酸盐 → "WGBS"
- geo_search_query: construct an NCBI GEO-compatible search string
- pubmed_search_query: construct a PubMed search string with MeSH terms where possible
"""


def parse_query_with_llm(query: str, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Use LLM to parse user query into structured search intent.

    Args:
        query: Raw user query string.
        llm: LangChain chat model instance.

    Returns:
        Structured intent dict.
    """
    import json

    messages = [
        SystemMessage(content=PARSE_SYSTEM_PROMPT),
        HumanMessage(content=f"Parse this query: {query}"),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback to rule-based parsing
        parsed = parse_query_rules(query)
        parsed["parse_method"] = "rules_fallback"
        return parsed

    parsed["parse_method"] = "llm"
    return parsed


def build_geo_search_string(intent: Dict[str, Any]) -> str:
    """
    Build a GEO NCBI E-utilities search string from parsed intent.

    Always includes a methylation platform filter so results are restricted
    to actual methylation datasets (450K, EPIC, WGBS, RRBS), not RNA-seq etc.

    Examples:
        "lung cancer" →
        'lung cancer[Title/Abstract] AND (GPL13534 OR GPL21145 OR GPL23976 OR
         bisulfite OR methylation profiling) AND GSE[Entry Type]'
    """
    parts = []

    # Cancer type — use plain keyword (no field tag) for broadest GEO match
    if intent.get("cancer_type"):
        ct = intent["cancer_type"]
        if isinstance(ct, dict):
            # Prefer display name over MeSH term for GEO title matching
            term = ct.get("display", "") or ct.get("mesh_term", "")
            if term:
                parts.append(term)
        elif isinstance(ct, str):
            parts.append(ct)

    # Platform — use GPL accessions for precision, plus text fallback
    platform = intent.get("platform")
    if platform:
        platform_terms = {
            "EPIC":  "(GPL21145 OR GPL23976 OR HumanMethylationEPIC OR 850K)",
            "450K":  "(GPL13534 OR HumanMethylation450 OR 450K OR HM450)",
            "WGBS":  "(WGBS OR whole genome bisulfite OR bisulfite sequencing)",
            "RRBS":  "(RRBS OR reduced representation bisulfite)",
        }
        parts.append(platform_terms.get(platform, platform))
    else:
        # No platform specified — restrict to known methylation platforms
        # GPL13534=450K, GPL21145=EPIC v1, GPL23976=EPIC v2, GPL8490=27K
        parts.append(
            "(GPL13534 OR GPL21145 OR GPL23976 OR GPL8490"
            " OR bisulfite OR methylation profiling)"
        )

    # Year range
    if intent.get("year_start") and intent.get("year_end"):
        y1, y2 = intent["year_start"], intent["year_end"]
        parts.append(f'("{y1}/01/01"[PDAT] : "{y2}/12/31"[PDAT])')

    # Always restrict to GSE entry type
    parts.append("GSE[Entry Type]")

    return " AND ".join(parts) if parts else \
        "(GPL13534 OR GPL21145 OR GPL23976) AND GSE[Entry Type]"
