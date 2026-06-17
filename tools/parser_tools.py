"""
Keyword parser and accession extractor for MethyAgent.

Handles two input modes:
  1. Semantic search: "EPIC平台在2024年的乳腺癌相关数据"
  2. Exact accession:  "下载GEO编号GSE124600的所有数据"

The LLM extracts structured intent; regex handles accession extraction
from free text (paper abstracts, supplementary materials, etc.).
"""
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from langchain_core.language_models import BaseChatModel
from utils.logger import get_logger

logger = get_logger(__name__)
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

# Sample type keywords → canonical sample type codes
# Each entry maps a keyword (English or Chinese) to a canonical sample type.
# The canonical types are:
#   tumor        — 癌症组织 / tumor tissue
#   adjacent     — 癌旁组织 / adjacent normal tissue
#   normal       — 正常组织 / normal tissue (from healthy individuals)
#   non_cancer   — 非癌对照组织 / non-cancer control tissue (e.g. benign, inflammation)
#   wbc          — 白细胞 / white blood cells / buffy coat / PBMC
#   cfdna        — cfDNA / cell-free DNA / circulating DNA / 血浆cfDNA
#   plasma       — 血浆 / plasma (broader than cfDNA, often used interchangeably)
#   serum        — 血清 / serum
#   whole_blood  — 全血 / whole blood
SAMPLE_TYPE_MAP = {
    # --- Tumor tissue ---
    "tumor tissue": "tumor",
    "tumour tissue": "tumor",
    "tumor": "tumor",
    "tumour": "tumor",
    "cancer tissue": "tumor",
    "primary tumor": "tumor",
    "primary tumour": "tumor",
    "malignant": "tumor",
    "neoplasm": "tumor",
    "癌症组织": "tumor",
    "癌组织": "tumor",
    "肿瘤组织": "tumor",
    "原发灶": "tumor",
    "原发肿瘤": "tumor",
    # --- Adjacent normal tissue ---
    "adjacent normal": "adjacent",
    "adjacent tissue": "adjacent",
    "adjacent": "adjacent",
    "paratumor": "adjacent",
    "paratumour": "adjacent",
    "peritumoral": "adjacent",
    "margin": "adjacent",
    "癌旁": "adjacent",
    "癌旁组织": "adjacent",
    "旁组织": "adjacent",
    "癌旁正常": "adjacent",
    # --- Normal tissue (healthy individuals) ---
    "normal tissue": "normal",
    "normal": "normal",
    "healthy tissue": "normal",
    "healthy control": "normal",
    "正常组织": "normal",
    "正常": "normal",
    "健康组织": "normal",
    # --- Non-cancer control ---
    "non-cancer": "non_cancer",
    "non-cancer control": "non_cancer",
    "noncancer": "non_cancer",
    "non-cancerous": "non_cancer",
    "benign": "non_cancer",
    "benign control": "non_cancer",
    "control tissue": "non_cancer",
    "非癌对照": "non_cancer",
    "非癌": "non_cancer",
    "非癌症": "non_cancer",
    "良性": "non_cancer",
    "良性对照": "non_cancer",
    # --- White blood cells ---
    "wbc": "wbc",
    "white blood cell": "wbc",
    "white blood cells": "wbc",
    "leukocyte": "wbc",
    "leukocytes": "wbc",
    "buffy coat": "wbc",
    "pbmc": "wbc",
    "peripheral blood mononuclear": "wbc",
    "peripheral blood": "wbc",
    "白细胞": "wbc",
    "外周血单个核细胞": "wbc",
    "外周血白细胞": "wbc",
    "血细胞": "wbc",
    # --- cfDNA ---
    "cfdna": "cfdna",
    "cfDNA": "cfdna",
    "cell-free dna": "cfdna",
    "cell free dna": "cfdna",
    "cell-free DNA": "cfdna",
    "circulating dna": "cfdna",
    "circulating tumor dna": "cfdna",
    "ctdna": "cfdna",
    "ctDNA": "cfdna",
    "cfDNA甲基化": "cfdna",
    "血浆cfDNA": "cfdna",
    "血浆cfDNA甲基化": "cfdna",
    "游离DNA": "cfdna",
    "循环DNA": "cfdna",
    "循环肿瘤DNA": "cfdna",
    # --- Plasma ---
    "plasma": "plasma",
    "blood plasma": "plasma",
    "血浆": "plasma",
    "血浆样本": "plasma",
    # --- Serum ---
    "serum": "serum",
    "blood serum": "serum",
    "血清": "serum",
    # --- Whole blood ---
    "whole blood": "whole_blood",
    "全血": "whole_blood",
    "血液": "whole_blood",
}

# Canonical sample type → GEO search terms (for building search strings)
SAMPLE_TYPE_GEO_TERMS = {
    "cfdna": "(cfDNA OR cell-free DNA OR circulating DNA OR ctdna OR plasma DNA)",
    "plasma": "(plasma OR blood plasma)",
    "serum": "(serum OR blood serum)",
    "wbc": "(WBC OR leukocyte OR buffy coat OR PBMC OR peripheral blood mononuclear)",
    "whole_blood": "(whole blood)",
    "tumor": "(tumor OR tumour OR cancer tissue OR primary tumor OR malignant)",
    "adjacent": "(adjacent normal OR paratumor OR peritumoral OR margin)",
    "normal": "(normal tissue OR healthy tissue OR healthy control)",
    "non_cancer": "(non-cancer OR benign OR control tissue OR noncancerous)",
}

# Canonical sample type → PubMed search terms
SAMPLE_TYPE_PUBMED_TERMS = {
    "cfdna": '("cell-free DNA" OR cfDNA OR "circulating DNA" OR "circulating tumor DNA" OR ctDNA)',
    "plasma": '(plasma OR "blood plasma")',
    "serum": '(serum OR "blood serum")',
    "wbc": '(leukocyte OR WBC OR "buffy coat" OR PBMC OR "peripheral blood mononuclear cell")',
    "whole_blood": '"whole blood"',
    "tumor": '(tumor OR tumour OR "cancer tissue" OR "primary tumor" OR malignant)',
    "adjacent": '("adjacent normal" OR paratumor OR peritumoral)',
    "normal": '("normal tissue" OR "healthy tissue" OR "healthy control")',
    "non_cancer": '("non-cancer" OR benign OR noncancerous OR "control tissue")',
}

# Sample type hierarchy: which types are "related" for filtering purposes
# When a user asks for cfDNA, we should also consider plasma (superset)
# When a user asks for non_cancer, we should also consider normal and adjacent
SAMPLE_TYPE_RELATED = {
    "cfdna": {"cfdna", "plasma"},           # cfDNA is from plasma
    "plasma": {"plasma", "cfdna"},           # plasma may contain cfDNA
    "serum": {"serum"},
    "wbc": {"wbc", "whole_blood"},
    "whole_blood": {"whole_blood", "wbc"},
    "tumor": {"tumor"},
    "adjacent": {"adjacent", "normal"},      # adjacent is a type of normal
    "normal": {"normal", "non_cancer"},      # normal is a type of non_cancer
    "non_cancer": {"non_cancer", "normal", "adjacent"},  # non_cancer includes normal + adjacent
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

# TCGA code → English display name (for GEO/PubMed queries which require English)
TCGA_CODE_TO_ENGLISH = {
    "BRCA": "breast cancer",
    "LUAD": "lung adenocarcinoma",
    "LUSC": "lung squamous cell carcinoma",
    "COAD": "colorectal cancer",
    "LIHC": "liver cancer",
    "STAD": "stomach cancer",
    "PRAD": "prostate cancer",
    "OV": "ovarian cancer",
    "CESC": "cervical cancer",
    "PAAD": "pancreatic cancer",
    "BLCA": "bladder cancer",
    "KIRC": "kidney cancer",
    "THCA": "thyroid cancer",
    "SKCM": "melanoma",
    "GBM": "glioblastoma",
    "LAML": "leukemia",
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


def _extract_sample_type(text: str) -> Optional[str]:
    """
    Extract sample type from text and return canonical code.
    
    Priority: longer/more specific matches first.
    e.g. "cfDNA" should match before "plasma", "癌旁组织" before "癌旁"
    
    Returns canonical sample type code (e.g. 'cfdna', 'tumor', 'wbc')
    or None if no sample type keyword found.
    """
    text_lower = text.lower()
    # Sort by keyword length (longest first) to prioritize specific matches
    # e.g. "血浆cfDNA甲基化" (7 chars) > "血浆cfDNA" (5 chars) > "cfDNA" (4 chars)
    sorted_keywords = sorted(SAMPLE_TYPE_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        # Use keyword.lower() for comparison since text is lowered
        # This handles mixed-case keys like "cfDNA", "ctDNA", "血浆cfDNA"
        if keyword.lower() in text_lower:
            return SAMPLE_TYPE_MAP[keyword]
    return None


def _extract_sample_types(text: str) -> List[str]:
    """
    Extract ALL sample types mentioned in text.
    Returns list of canonical sample type codes (deduplicated, order of first appearance).
    
    Example: "colorectal cancer和非癌对照的cfDNA甲基化数据"
    → ['non_cancer', 'cfdna']
    """
    text_lower = text.lower()
    found = []
    seen = set()
    # Sort by keyword length (longest first) for priority
    sorted_keywords = sorted(SAMPLE_TYPE_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        # Use keyword.lower() for comparison since text is lowered
        if keyword.lower() in text_lower:
            canonical = SAMPLE_TYPE_MAP[keyword]
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)
    return found


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
    sample_types = _extract_sample_types(query)
    # Primary sample type: the first (most specific) one found
    primary_sample_type = sample_types[0] if sample_types else None

    return {
        "raw_query": query,
        "mode": "accession" if has_explicit_accession(query) else "semantic",
        "accessions": accessions,
        "platform": platform,
        "cancer_type_display": cancer_display,
        "cancer_type_code": cancer_code,
        "sample_type": primary_sample_type,
        "sample_types": sample_types,
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
  "sample_type": "cfdna" | "plasma" | "serum" | "wbc" | "whole_blood" | "tumor" | "adjacent" | "normal" | "non_cancer" | null,
  "sample_types": ["cfdna", "non_cancer"],
  "year_start": 2024,
  "year_end": 2024,
  "sample_type_detail": "Brief description of what sample types the user wants",
  "geo_search_query": "colorectal cancer cfDNA methylation[GEO]",
  "pubmed_search_query": "colorectal cancer DNA methylation cfDNA cell-free DNA 2024",
  "notes": "any special instructions"
}

Rules:
- If the query contains explicit accession numbers (GSE..., TCGA-...), set mode="accession"
- For Chinese cancer names: 乳腺癌=BRCA, 肺癌=LUAD, 肝癌=LIHC, 胃癌=STAD, 结直肠癌=COAD
- For platform: EPIC/850K → "EPIC", 450K/HM450 → "450K", WGBS/全基因组亚硫酸盐 → "WGBS"

SAMPLE TYPE PARSING (CRITICAL for cfDNA/liquid biopsy queries):
- cfDNA / cell-free DNA / circulating DNA / ctDNA / 游离DNA / 循环DNA → "cfdna"
- plasma / 血浆 → "plasma"
- serum / 血清 → "serum"
- WBC / leukocyte / buffy coat / PBMC / 白细胞 / 血细胞 → "wbc"
- whole blood / 全血 → "whole_blood"
- tumor tissue / cancer tissue / 癌症组织 / 肿瘤组织 / 原发灶 → "tumor"
- adjacent normal / paratumor / 癌旁 / 癌旁组织 → "adjacent"
- normal tissue / healthy control / 正常组织 / 健康 → "normal"
- non-cancer control / benign / 非癌对照 / 非癌 / 良性 → "non_cancer"

- sample_type: the PRIMARY sample type the user wants (single value)
- sample_types: ALL sample types mentioned in the query (list, may include multiple)
- When user asks for "非癌对照" (non-cancer control), include both "non_cancer" and "normal" in sample_types
- When user asks for cfDNA, also include "plasma" in sample_types (cfDNA comes from plasma)
- geo_search_query: construct an NCBI GEO-compatible search string INCLUDING sample type terms
- pubmed_search_query: construct a PubMed search string with MeSH terms AND sample type terms
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




# ------------------------------------------------------------------ #
#  Cancer synonyms loader (config/cancer_synonyms.yaml)               #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=1)
def _load_cancer_synonyms() -> dict:
    """
    Load cancer synonyms, methylation tech terms, and liquid biopsy terms
    from config/cancer_synonyms.yaml.

    Returns dict with keys:
      cancer_synonyms        : {TCGA_CODE: [synonym, ...]}
      methylation_tech_terms : [term, ...]
      liquid_biopsy_terms    : [term, ...]

    Falls back to empty structures if the file is not found.
    """
    candidates = [
        Path(__file__).parent.parent / "config" / "cancer_synonyms.yaml",
        Path("config") / "cancer_synonyms.yaml",
        Path("cancer_synonyms.yaml"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                "cancer_synonyms":        data.get("cancer_synonyms", {}),
                "methylation_tech_terms": data.get("methylation_tech_terms", []),
                "liquid_biopsy_terms":    data.get("liquid_biopsy_terms", []),
            }
    return {"cancer_synonyms": {}, "methylation_tech_terms": [], "liquid_biopsy_terms": []}


def _get_cancer_synonyms(tcga_code: str) -> List[str]:
    """Return GEO search synonyms for a TCGA cancer code."""
    data = _load_cancer_synonyms()
    syns = data["cancer_synonyms"].get(tcga_code, [])
    if not syns:
        english = TCGA_CODE_TO_ENGLISH.get(tcga_code, "")
        return [english] if english else []
    return syns

def build_geo_search_string(intent: Dict[str, Any]) -> str:
    """
    Build a GEO NCBI E-utilities search string from parsed intent.

    v3 improvements:
    - Cancer type: expanded with synonyms from cancer_synonyms.yaml
      (e.g. COAD → CRC OR "colorectal cancer" OR "colon cancer" OR adenoma OR ...)
    - Methylation tech: expanded beyond GPL numbers to include technique keywords
      (MCTA, MeDIP, methylome, bisulfite sequencing, etc.)
    - Liquid biopsy: when user asks for cfDNA/plasma, adds full liquid biopsy
      vocabulary (cfDNA, ctDNA, plasma, serum, "liquid biopsy", etc.)
    - Species: always adds (human OR "Homo sapiens") filter
    """
    synonyms_data = _load_cancer_synonyms()
    parts = []

    # ---- Cancer type: expand with synonyms ----
    tcga_code = None
    display = ""
    if intent.get("cancer_type"):
        ct = intent["cancer_type"]
        if isinstance(ct, dict):
            tcga_code = ct.get("tcga_code") or ct.get("code")
            display = ct.get("display", "") or ct.get("mesh_term", "")
        elif isinstance(ct, str):
            display = ct
    elif intent.get("cancer_type_display"):
        tcga_code = intent.get("cancer_type_code")
        display = intent.get("cancer_type_display", "")

    if tcga_code and tcga_code in synonyms_data["cancer_synonyms"]:
        synonyms = synonyms_data["cancer_synonyms"][tcga_code]
        quoted = []
        for s in synonyms:
            s = str(s)
            quoted.append(f'"{s}"' if " " in s else s)
        parts.append("(" + " OR ".join(quoted) + ")")
    elif display:
        # No synonyms — use canonical English name
        if tcga_code and tcga_code in TCGA_CODE_TO_ENGLISH:
            english = TCGA_CODE_TO_ENGLISH[tcga_code]
            parts.append(f'"{english}"' if " " in english else english)
        else:
            parts.append(f'"{display}"' if " " in display else display)

    # ---- Sample type: add sample type search terms ----
    sample_type = intent.get("sample_type")
    sample_types = intent.get("sample_types", [])
    is_liquid_biopsy = (
        sample_type in ("cfdna", "plasma", "serum")
        or any(st in ("cfdna", "plasma", "serum") for st in sample_types)
    )

    if is_liquid_biopsy and synonyms_data["liquid_biopsy_terms"]:
        lb_terms = synonyms_data["liquid_biopsy_terms"]
        quoted_lb = []
        for t in lb_terms:
            t = str(t)
            quoted_lb.append(f'"{t}"' if " " in t else t)
        parts.append("(" + " OR ".join(quoted_lb) + ")")
    elif sample_type and sample_type in SAMPLE_TYPE_GEO_TERMS:
        parts.append(SAMPLE_TYPE_GEO_TERMS[sample_type])
    elif sample_types:
        geo_terms = []
        seen: set = set()
        for st in sample_types:
            if st in SAMPLE_TYPE_GEO_TERMS and st not in seen:
                geo_terms.append(SAMPLE_TYPE_GEO_TERMS[st])
                seen.add(st)
        for st in sample_types:
            for related in SAMPLE_TYPE_RELATED.get(st, set()):
                if related in SAMPLE_TYPE_GEO_TERMS and related not in seen:
                    geo_terms.append(SAMPLE_TYPE_GEO_TERMS[related])
                    seen.add(related)
        if geo_terms:
            parts.append(
                "(" + " OR ".join(geo_terms) + ")" if len(geo_terms) > 1 else geo_terms[0]
            )

    # ---- Methylation technology: expanded keyword list ----
    platform = intent.get("platform")
    if platform:
        platform_terms = {
            "EPIC": "(GPL21145 OR GPL23976 OR HumanMethylationEPIC OR 850K)",
            "450K": "(GPL13534 OR HumanMethylation450 OR 450K OR HM450)",
            "WGBS": '(WGBS OR "whole genome bisulfite" OR "bisulfite sequencing")',
            "RRBS": '(RRBS OR "reduced representation bisulfite")',
        }
        parts.append(platform_terms.get(platform, platform))
    else:
        # No platform specified — use expanded tech vocabulary from YAML
        tech_terms = synonyms_data.get("methylation_tech_terms", [])
        if tech_terms:
            quoted_tech = []
            for t in tech_terms:
                t = str(t)
                quoted_tech.append(f'"{t}"' if " " in t else t)
            parts.append("(" + " OR ".join(quoted_tech) + ")")
        else:
            # Fallback: GPL numbers only
            parts.append(
                "(GPL13534 OR GPL21145 OR GPL23976 OR GPL8490"
                " OR bisulfite OR methylation profiling)"
            )

    # ---- Species filter ----
    parts.append('(human OR "Homo sapiens")')

    # ---- Year range ----
    if intent.get("year_start") and intent.get("year_end"):
        y1, y2 = intent["year_start"], intent["year_end"]
        parts.append(f'("{y1}/01/01"[PDAT] : "{y2}/12/31"[PDAT])')

    # Always restrict to GSE entry type
    parts.append("GSE[Entry Type]")

    query = " AND ".join(parts) if parts else \
        "(GPL13534 OR GPL21145 OR GPL23976) AND GSE[Entry Type]"

    # ---- Safety: cap query length to avoid NCBI abuse detection ----
    # NCBI redirects to abuse.shtml for very long queries.
    # 400 chars is a safe upper bound; most well-formed queries are 200-350 chars.
    MAX_QUERY_LENGTH = 400
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(
            f"GEO query too long ({len(query)} chars, max {MAX_QUERY_LENGTH}) — "
            f"trimming cancer synonyms"
        )
        # Strategy: rebuild with progressively fewer cancer synonyms
        # until the query fits. Other parts (tech, sample type, species) are kept intact.
        if tcga_code and tcga_code in synonyms_data["cancer_synonyms"]:
            synonyms = synonyms_data["cancer_synonyms"][tcga_code]
            # Try with fewer synonyms: keep first N, reduce until fits
            for n in range(len(synonyms), 0, -1):
                subset = synonyms[:n]
                quoted = []
                for s in subset:
                    s = str(s)
                    quoted.append(f'"{s}"' if " " in s else s)
                trimmed_parts = ["(" + " OR ".join(quoted) + ")"] + parts[1:]
                query = " AND ".join(trimmed_parts)
                if len(query) <= MAX_QUERY_LENGTH:
                    logger.info(f"Trimmed cancer synonyms to {n}/{len(synonyms)} terms, query={len(query)} chars")
                    break
            else:
                # Even 1 synonym is too long — use just the TCGA code
                trimmed_parts = [tcga_code] + parts[1:]
                query = " AND ".join(trimmed_parts)
                logger.warning(f"Query still long with 1 synonym, using TCGA code only: {len(query)} chars")

    return query
