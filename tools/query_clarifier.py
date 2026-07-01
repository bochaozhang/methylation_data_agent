
"""
Query clarifier, keyword builder, and structured extractor for MethyAgent.

Five components:
  1. ask_clarifying_questions()     — 追问: detect missing dimensions, ask follow-ups
  2. build_ncbi_safe_pubmed_query() — ranked keyword assembly within NCBI limits
  3. build_pubmed_query_with_controls() — paired cancer + normal/control queries
  4. evaluate_geo_dataset()         — LLM judge: is this GEO dataset usable?
  5. extract_paper_structured()     — LLM extractor: pull structured fields from abstract

Imports from parser_tools.py — does NOT duplicate existing maps or functions.
Standalone importable (no LangGrqaph state dependency).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from tools.parser_tools import (
    SAMPLE_TYPE_PUBMED_TERMS,
    TCGA_CODE_TO_ENGLISH,
    _load_cancer_synonyms,
    parse_query_rules,
)


# ============================================================
# Helpers
# ============================================================

def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ============================================================
# Component 1 — Clarifying Questions (追问)
# ============================================================

_CLARIFIER_SYSTEM = """You are a biomedical literature search assistant helping a researcher \
find DNA methylation cfDNA cancer early-screening datasets and biomarkers.

You understand both English and Chinese queries equally. Chinese terms FULLY satisfy dimensions:
  cancer_type : 结直肠癌=colorectal, 乳腺癌=breast, 肺癌=lung, 肝癌=liver, 胃癌=gastric
  sample_type : 血浆cfDNA / cfDNA甲基化 / 循环DNA / 游离DNA = cfDNA/plasma (SATISFIED)
  controls    : 需要健康对照 / 正常对照 / 配对对照 = controls required (SATISFIED)
  platform    : 450K / 850K / EPIC / WGBS / RRBS = platform specified (SATISFIED)
  year_range  : any "YYYY" or "YYYY-YYYY" or "YYYY~YYYY" pattern = year specified (SATISFIED)

Given a user query, decide which of the following dimensions are ambiguous or TRULY missing:
  A. cancer_type  — which cancer? (colorectal, lung, breast, liver, multi-cancer, unknown)
  B. sample_type  — cfDNA/plasma vs tumor tissue vs both vs unknown
  C. controls     — are matched healthy/normal controls required? (yes / no / no preference)
  D. goal         — find datasets / find biomarkers / find performance metrics / all of the above
  E. platform     — 450K, EPIC/850K, WGBS, RRBS, MCTA-Seq, targeted, any
  F. year_range   — specific year range or no preference

A query is SPECIFIC ENOUGH (is_specific_enough=true, questions=[]) when ALL THREE core
dimensions are answered: cancer_type + sample_type + controls. Platform and year_range
are optional — their absence alone does NOT make a query vague.

Rules:
- A dimension is satisfied if the query contains any signal for it, in any language.
- DO NOT ask about a dimension that is already answered in the query.
- DO NOT ask about year_range or platform if they are not critical to narrowing the search.
- Only ask about dimensions that are TRULY unclear from the query.
- Ask at most 3 questions. Each question must have 3-4 concrete options.
- If the query is already specific enough, set is_specific_enough=true and questions=[].
- Output ONLY valid JSON, no extra text:

{
  "is_specific_enough": false,
  "missing_dimensions": ["cancer_type", "sample_type"],
  "questions": [
    {
      "dimension": "cancer_type",
      "question": "Which cancer type are you studying?",
      "options": ["Colorectal (CRC)", "Lung", "Breast", "Multiple cancers / pan-cancer"]
    }
  ],
  "clarified_intent_hint": "User wants methylation data but cancer type and sample type unclear"
}"""


@dataclass
class ClarifyingQuestion:
    dimension: str
    question: str
    options: List[str]


@dataclass
class ClarificationResult:
    is_specific_enough: bool
    questions: List[ClarifyingQuestion] = field(default_factory=list)
    missing_dimensions: List[str] = field(default_factory=list)
    clarified_intent_hint: str = ""


def ask_clarifying_questions(
    query: str,
    llm: BaseChatModel,
) -> ClarificationResult:
    """
    Analyse the user query and return clarifying questions for missing dimensions.
    Returns ClarificationResult with up to 3 questions.
    If the query is specific enough, returns is_specific_enough=True with no questions.
    """
    messages = [
        SystemMessage(content=_CLARIFIER_SYSTEM),
        HumanMessage(content=f"User query: {query}"),
    ]
    response = llm.invoke(messages)
    content = _strip_json_fences(response.content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # If LLM fails to return valid JSON, assume query is specific enough
        return ClarificationResult(is_specific_enough=True)

    questions = [
        ClarifyingQuestion(
            dimension=q.get("dimension", ""),
            question=q.get("question", ""),
            options=q.get("options", []),
        )
        for q in data.get("questions", [])
    ]
    return ClarificationResult(
        is_specific_enough=data.get("is_specific_enough", True),
        questions=questions,
        missing_dimensions=data.get("missing_dimensions", []),
        clarified_intent_hint=data.get("clarified_intent_hint", ""),
    )


def format_clarifying_questions(result: ClarificationResult) -> str:
    """
    Format clarifying questions as a readable string to show the user.
    Returns empty string if no questions needed.
    """
    if result.is_specific_enough or not result.questions:
        return ""
    lines = ["To search more precisely, please answer:\n"]
    for i, q in enumerate(result.questions, 1):
        lines.append(f"{i}. {q.question}")
        for j, opt in enumerate(q.options, 1):
            lines.append(f"   {chr(96 + j)}) {opt}")
        lines.append("")
    return "\n".join(lines)


def apply_clarification_answers(
    original_query: str,
    questions: List[ClarifyingQuestion],
    answers: Dict[str, str],
) -> str:
    """
    Merge the user's clarification answers back into the original query string.

    Args:
        original_query: The original vague query.
        questions: The clarifying questions that were asked.
        answers: Dict mapping dimension → chosen option text.
                 e.g. {"cancer_type": "Colorectal (CRC)", "sample_type": "cfDNA/plasma"}

    Returns:
        Augmented query string for re-parsing with parse_query_with_llm().
    """
    additions = [answers[q.dimension] for q in questions if q.dimension in answers]
    if not additions:
        return original_query
    return original_query + ". " + ". ".join(additions)


# ============================================================
# Component 2 — NCBI-Safe PubMed Keyword Builder
# ============================================================

# NCBI PubMed limits: too many boolean clauses cause query rejection.
_MAX_OR_TERMS = 5
_MAX_AND_GROUPS = 4

# PubMed cancer synonyms — max 5 per cancer to stay within NCBI limits.
# Different from GEO synonyms (which can be longer).
CANCER_PUBMED_SYNONYMS: Dict[str, List[str]] = {
    "COAD": [
        '"colorectal cancer"', '"colorectal carcinoma"',
        '"colon cancer"', '"rectal cancer"', "CRC",
    ],
    "LUAD": [
        '"lung cancer"', '"lung adenocarcinoma"',
        "NSCLC", '"non-small cell lung cancer"', '"pulmonary nodule"',
    ],
    "LUSC": [
        '"lung squamous cell carcinoma"', "LUSC",
        "NSCLC", '"non-small cell lung cancer"',
    ],
    "BRCA": [
        '"breast cancer"', '"breast carcinoma"',
        '"invasive breast cancer"', '"ductal carcinoma"',
    ],
    "LIHC": [
        '"hepatocellular carcinoma"', "HCC",
        '"liver cancer"', '"primary liver cancer"',
    ],
    "STAD": [
        '"gastric cancer"', '"stomach cancer"',
        '"gastric carcinoma"', '"gastric adenocarcinoma"',
    ],
    "PAAD": [
        '"pancreatic cancer"', "PDAC",
        '"pancreatic ductal adenocarcinoma"', '"pancreatic carcinoma"',
    ],
    "PRAD": [
        '"prostate cancer"', '"prostate carcinoma"',
        '"prostate adenocarcinoma"',
    ],
    "OV": [
        '"ovarian cancer"', '"ovarian carcinoma"',
        '"high-grade serous ovarian"', "HGSC",
    ],
    "KIRC": [
        '"renal cell carcinoma"', '"kidney cancer"',
        "RCC", '"clear cell renal cell carcinoma"',
    ],
    "BLCA": [
        '"bladder cancer"', '"urothelial carcinoma"',
        '"transitional cell carcinoma"',
    ],
    "ESCA": [
        '"esophageal cancer"', "ESCC",
        '"esophageal squamous cell carcinoma"',
    ],
    "THCA": [
        '"thyroid cancer"', '"papillary thyroid carcinoma"', "THCA",
    ],
    "SKCM": [
        "melanoma", '"cutaneous melanoma"', '"malignant melanoma"',
    ],
    "GBM": [
        '"glioblastoma"', "GBM", "glioma", '"high-grade glioma"',
    ],
    "LAML": [
        '"acute myeloid leukemia"', "AML",
        '"chronic lymphocytic leukemia"', "CLL",
    ],
    "CESC": [
        '"cervical cancer"', '"cervical carcinoma"', "CESC",
    ],
}

# Control/normal terms added when require_controls=True
_CONTROL_TERMS = [
    '"healthy control"',
    '"healthy donor"',
    '"non-cancer control"',
    "benign",
    '"adjacent normal"',
]

_METHYLATION_MESH = '"DNA methylation"[MeSH Terms]'


def build_ncbi_safe_pubmed_query(
    intent: Dict[str, Any],
    require_controls: bool = False,
) -> str:
    """
    Build an NCBI-safe PubMed boolean query from parsed intent.

    Respects NCBI limits:
      - Max 5 OR-terms per AND-group
      - Max 4 AND-groups total

    Priority order (drops from lowest priority if over limit):
      1. cancer_type  — required
      2. DNA methylation[MeSH Terms]  — required
      3. sample_type  — important
      4. controls     — only if require_controls=True
      5. platform     — optional
      6. year         — optional

    Args:
        intent: Parsed intent dict from parse_query_with_llm() or parse_query_rules().
        require_controls: If True, add healthy/normal control terms.

    Returns:
        PubMed-compatible boolean query string.
    """
    groups: List[str] = []   # each element is one AND-group (already formatted)
    required_count = 0

    # --- Group 1: Cancer type (required) ---
    cancer_group = _build_cancer_group(intent)
    if cancer_group:
        groups.append(cancer_group)
        required_count += 1

    # --- Group 2: DNA methylation MeSH (required) ---
    groups.append(_METHYLATION_MESH)
    required_count += 1

    # --- Group 3: Sample type ---
    sample_group = _build_sample_group(intent)
    if sample_group:
        groups.append(sample_group)

    # --- Group 4: Controls (only if requested) ---
    if require_controls or _intent_mentions_controls(intent):
        ctrl = "(" + " OR ".join(_CONTROL_TERMS[:_MAX_OR_TERMS]) + ")"
        groups.append(ctrl)

    # --- Group 5: Platform (optional) ---
    platform_group = _build_platform_group(intent)
    if platform_group:
        groups.append(platform_group)

    # --- Group 6: Year range (optional) ---
    year_group = _build_year_group(intent)
    if year_group:
        groups.append(year_group)

    # Trim to _MAX_AND_GROUPS, keeping required groups first
    if len(groups) > _MAX_AND_GROUPS:
        required = groups[:required_count]
        optional = groups[required_count:]
        optional = optional[:(_MAX_AND_GROUPS - required_count)]
        groups = required + optional

    query = " AND ".join(groups) if groups else _METHYLATION_MESH

    # Character-length safety cap — drop lowest-priority groups until under limit
    _MAX_QUERY_CHARS = 400
    while len(query) > _MAX_QUERY_CHARS and len(groups) > required_count:
        groups.pop()   # drop the last (lowest-priority) optional group
        query = " AND ".join(groups)

    return query


def _build_cancer_group(intent: Dict[str, Any]) -> str:
    """Build the cancer type OR-group."""
    ct = intent.get("cancer_type", {})
    tcga_code = ""
    display = ""

    if isinstance(ct, dict):
        tcga_code = ct.get("tcga_code") or ct.get("code", "")
        display = ct.get("display", "") or ct.get("mesh_term", "")
    elif isinstance(ct, str):
        display = ct

    # Also check flat fields from rule-based parser
    if not tcga_code:
        tcga_code = intent.get("cancer_type_code", "")
    if not display:
        display = intent.get("cancer_type_display", "")

    if tcga_code and tcga_code in CANCER_PUBMED_SYNONYMS:
        terms = CANCER_PUBMED_SYNONYMS[tcga_code][:_MAX_OR_TERMS]
        return "(" + " OR ".join(terms) + ")"

    if display:
        english = TCGA_CODE_TO_ENGLISH.get(tcga_code, "") if tcga_code else ""
        name = english or display
        return f'"{name}"' if " " in name else name

    return ""


def _build_sample_group(intent: Dict[str, Any]) -> str:
    """Build the sample type OR-group using SAMPLE_TYPE_PUBMED_TERMS."""
    sample_types: List[str] = intent.get("sample_types") or []
    primary = intent.get("sample_type")
    if primary and primary not in sample_types:
        sample_types = [primary] + sample_types

    terms: List[str] = []
    seen: set = set()
    for st in sample_types:
        if st in SAMPLE_TYPE_PUBMED_TERMS and st not in seen:
            terms.append(SAMPLE_TYPE_PUBMED_TERMS[st])
            seen.add(st)
        if len(terms) >= _MAX_OR_TERMS:
            break

    if not terms:
        return ""
    return "(" + " OR ".join(terms) + ")" if len(terms) > 1 else terms[0]


def _build_platform_group(intent: Dict[str, Any]) -> str:
    """Build platform search terms."""
    platform = intent.get("platform")
    if not platform:
        return ""
    mapping = {
        "EPIC":  '(EPIC OR HumanMethylationEPIC OR "850K")',
        "450K":  '(HumanMethylation450 OR "450K")',
        "WGBS":  '("whole genome bisulfite sequencing" OR WGBS)',
        "RRBS":  '("reduced representation bisulfite" OR RRBS)',
        "MCTA":  '"MCTA-Seq"',
    }
    return mapping.get(platform, f'"{platform}"')


def _build_year_group(intent: Dict[str, Any]) -> str:
    """Build year range filter."""
    y1 = intent.get("year_start")
    y2 = intent.get("year_end") or y1
    if not y1:
        return ""
    return f'("{y1}/01/01"[PDAT]:"{y2}/12/31"[PDAT])'


def _intent_mentions_controls(intent: Dict[str, Any]) -> bool:
    """Detect if the intent mentions needing matched controls."""
    signals = ["control", "normal", "healthy", "non-cancer", "对照", "正常", "配对", "健康"]
    text = " ".join([
        str(intent.get("notes", "")),
        str(intent.get("raw_query", "")),
        str(intent.get("sample_type_detail", "")),
    ]).lower()
    return any(s in text for s in signals)


# ============================================================
# Component 3 — Normal/Cancer Control Query Builder (对照试验)
# ============================================================

def build_pubmed_query_with_controls(intent: Dict[str, Any]) -> Dict[str, str]:
    """
    Build three complementary PubMed queries to find papers with
    both cancer cases AND normal/healthy controls.

    Many papers only have cancer samples — this helps find studies
    that tested BOTH groups (required for sensitivity/specificity calculation).

    Returns dict with keys:
      "main"         — base query + control terms (AND): finds papers with both groups
      "cancer_only"  — base + early detection / biomarker terms
      "control_only" — base + healthy donor / normal plasma terms
    """
    # Build base without controls — each variant appends its own clause explicitly.
    # Strip raw_query/notes so _intent_mentions_controls doesn't fire on the base.
    base_intent = {**intent, "raw_query": "", "notes": "", "sample_type_detail": ""}
    base = build_ncbi_safe_pubmed_query(base_intent, require_controls=False)

    # Main: explicitly require control samples
    control_clause = (
        '("healthy control" OR "healthy donor" OR "non-cancer control" OR benign OR "adjacent normal")'
    )
    main = f"{base} AND {control_clause}"

    # Cancer-only: focused on detection/screening papers
    cancer_clause = (
        '("early detection" OR "early screening" OR biomarker OR "cancer screening" OR "early diagnosis")'
    )
    cancer_only = f"{base} AND {cancer_clause}"

    # Control-only: specifically for matched normal/healthy samples
    normal_clause = (
        '("healthy donor" OR "healthy volunteer" OR "normal plasma" OR "non-cancer" OR "control group")'
    )
    control_only = f"{base} AND {normal_clause}"

    return {
        "main": main,
        "cancer_only": cancer_only,
        "control_only": control_only,
    }


# ============================================================
# Component 4 — GEO Dataset Usability Evaluator
# ============================================================

_GEO_EVALUATOR_SYSTEM = """You are an AI that evaluates GEO methylation dataset usability \
for cancer early screening liquid biopsy method development.

Given dataset information, judge whether it is suitable for the project.

PRIORITY KEEP:
- Plasma/serum cfDNA or ctDNA with BOTH cancer cases AND healthy/benign/pre-cancer controls
- Early-stage, pre-treatment samples
- Tumor vs adjacent/normal tissue (useful for marker discovery)
- Healthy donor cfDNA, whole blood, WBC, PBMC (useful for background estimation)
- 450K/850K/EPIC array data, especially if normalized beta matrix is available
- Large sample size, clear annotation, associated publication

DEFAULT EXCLUDE:
- Cell lines (HCT-116, A549, etc.), organoids, PDX, xenograft, animal models
- In vitro treated / drug treatment / DNMTi treatment / radiation / gene-edited samples
- Post-treatment, relapse, or drug-resistance model samples
  (unless pre-treatment baseline can be separately extracted)
- Metaplasia samples (not relevant for early screening)
- Ascites, pleural fluid, fecal methylation (unless task explicitly requires them)
- Non-target cancer type with no way to separate by sample ID
- Only a marker list / differential CpG table — no full matrix or sample-level data
- Data locked or unavailable for download

FLAG AS manual_review:
- Pooled cfDNA samples (cannot treat as individual patient samples)
- Only cancer cases present, no healthy control cfDNA
- Only whole blood / WBC (not cfDNA, but usable for background signal exclusion)
- 27K array data (very few probes, low priority)
- Pan-cancer or mixed cancer type data (check if separable by sample ID)
- GEO metadata label conflicts with the associated paper's label
- Paper only reports marker results, no full methylation matrix available

FILE TYPE PRIORITY for 450K/850K (best → worst):
1. Multi-sample normalized beta matrix (BEST — ready to use)
2. IDAT raw files + complete sample annotation
3. Raw intensity / detection p-value tables
4. Per-GSM individual average beta (usable but high processing cost)
5. Marker list only (NOT usable as full dataset)

SPECIES: Only keep human / Homo sapiens. Exclude mouse and other animal models.

Output ONLY valid JSON, no extra text:
{
  "cancer_type": "",
  "sample_type": "",
  "disease_groups": "",
  "sample_size": "",
  "stage_or_treatment_status": "",
  "technology": "",
  "available_file_type": "",
  "sample_level_annotation": "yes/no/unclear",
  "usable": "yes/no/partial/unclear",
  "recommended_action": "keep/exclude/manual_review/article_only",
  "reason": "",
  "notes": ""
}

Field definitions:
  sample_type            — actual sample type: plasma cfDNA / tumor tissue / cell line / WBC / ...
  disease_groups         — describe case and control groups, e.g. "CRC n=50, healthy n=40"
  stage_or_treatment_status — early / late / pre-treatment / post-treatment / unknown
  technology             — 450K / 850K / WGBS / RRBS / MCTA / targeted panel / other
  available_file_type    — beta matrix / IDAT / marker list / raw counts / unknown
  recommended_action     — keep: use directly; exclude: skip; manual_review: human check needed;
                           article_only: useful as reference but data not usable
  reason                 — 1-3 sentences explaining the judgment
  notes                  — special cases: pooled samples, post-treatment mixing, label conflicts, etc."""


def evaluate_geo_dataset(
    dataset_info: str,
    cancer_type: str,
    sample_types: List[str],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Use LLM to judge whether a GEO dataset is usable for the project.

    Args:
        dataset_info: GEO series description, title, sample counts, platform info, etc.
        cancer_type:  The target cancer type for this search (e.g. "colorectal cancer").
        sample_types: List of desired sample types (e.g. ["cfdna", "plasma"]).
        llm:          LangChain chat model.

    Returns:
        Structured judgment dict with keys: usable, recommended_action, reason, etc.
    """
    context = (
        f"Target cancer type: {cancer_type}\n"
        f"Desired sample types: {', '.join(sample_types)}\n\n"
        f"Dataset information:\n{dataset_info}"
    )
    messages = [
        SystemMessage(content=_GEO_EVALUATOR_SYSTEM),
        HumanMessage(content=context),
    ]
    response = llm.invoke(messages)
    content = _strip_json_fences(response.content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "usable": "unclear",
            "recommended_action": "manual_review",
            "reason": "LLM returned unparseable response.",
            "notes": content[:300],
        }


# ============================================================
# Component 5 — PubMed Abstract Structured Extractor
# ============================================================

_EXTRACTOR_SYSTEM = """You are a biomedical literature extraction assistant specializing in \
DNA methylation cancer early screening research.

Extract structured information from the paper abstract provided.
Output ONLY valid JSON. Use null for missing fields.

CRITICAL RULES:
1. sample_type MUST distinguish tissue vs cfDNA — they are NOT interchangeable.
   Tissue methylation data CANNOT substitute for cfDNA early screening validation.
   Use one of: tissue / plasma_cfdna / serum_cfdna / wbc / whole_blood / mixed / unknown

2. data_availability must be one of:
   public_download / controlled_access / request_required /
   supplementary_only / not_available / unknown

3. If AUC or sensitivity is reported, always record the value AND which cohort
   it applies to (training / validation / external).

4. For markers: extract CpG IDs (cg...) or gene names if mentioned.

5. confidence_level:
   high   = abstract clearly states sample type + dataset ID + at least one performance metric
   medium = mentions methylation + cancer + controls but key fields are ambiguous
   low    = vaguely relevant, most fields are null

6. Only extract auc_validation/auc_external if the abstract explicitly contains the term
   "AUC" or "area under the curve" or "ROC". Do NOT infer AUC from sensitivity, specificity,
   or accuracy values.

7. When a paper reports metrics for both tissue and cfDNA samples, only report the metrics
   matching the PRIMARY sample_type you assigned. If you set sample_type to plasma_cfdna,
   only report cfDNA metrics, not tissue metrics.

8. Only extract dataset_ids that contain the study's PRIMARY experimental data. Exclude
   datasets used only as reference panels, background noise filters, normalization controls,
   or annotation sources.

9. early_stage_count is an integer count of early-stage (I/II) samples, only if the abstract
   explicitly states it — otherwise null. has_external_validation is a boolean: true only if
   the abstract describes validation on an independent/external cohort, otherwise false.
   supplementary_links is a list of URLs to supplementary data if mentioned in the abstract,
   otherwise null.

10. The schema below is a FORMAT TEMPLATE ONLY. Every value in it (numbers, accession IDs,
    gene names, URLs) is a fake placeholder to illustrate the expected type and shape.
    Do NOT copy any value from the schema into your output under any circumstance. Every
    field you output must come from the abstract text itself, or be null if the abstract
    does not state it.

Output schema (values below are placeholders — do not copy them):
{
  "cancer_type": "CRC | lung | breast | liver | gastric | pancreatic | multi | other | null",
  "sample_type": "tissue | plasma_cfdna | serum_cfdna | wbc | whole_blood | mixed | unknown",
  "has_normal_control": true,
  "has_cancer_samples": true,
  "normal_control_description": "e.g. healthy donors n=40, benign polyps n=20",
  "technology": "WGBS | RRBS | EPIC | 450K | MCTA-Seq | targeted | other | null",
  "dataset_ids": ["<real accession from abstract, e.g. GSE203944>"],
  "sample_size_case": 143,
  "sample_size_control": 59,
  "early_stage_count": 22,
  "has_external_validation": false,
  "supplementary_links": ["https://..."],
  "markers_or_panel": [
    {"id": "cg08122047", "gene": "VIM", "type": "CpG | gene | DMR | panel"}
  ],
  "performance_metrics": {
    "auc_training": null,
    "auc_validation": 0.76,
    "auc_external": null,
    "sensitivity_at_specificity": "80% sensitivity at 95% specificity"
  },
  "data_availability": "public_download | controlled_access | request_required | supplementary_only | not_available | unknown",
  "confidence_level": "high | medium | low",
  "needs_human_review": false,
  "reason": "brief explanation of confidence level and any flags"
}

Reminder: the JSON above is a shape example, not real data. Re-derive every value from the
abstract you were given."""

# Keywords that must appear in the abstract for an AUC value to be considered real.
_AUC_KEYWORDS = ("auc", "area under the curve", "roc")


def _sanitize_extracted(result: Dict[str, Any], abstract: str) -> Dict[str, Any]:
    """
    Post-hoc guard against the LLM anchoring on the schema's example placeholder
    values instead of the abstract (observed on 2026-06-30: glm-4-flash echoed
    the schema's example AUC/dataset_ids verbatim, and even copied the literal
    "<placeholder>" template text, regardless of prompt wording). Enforces
    CRITICAL RULE 6 in code rather than relying on the model to follow it.
    """
    abstract_lower = (abstract or "").lower()
    has_auc_mention = any(kw in abstract_lower for kw in _AUC_KEYWORDS)

    metrics = result.get("performance_metrics")
    if isinstance(metrics, dict) and not has_auc_mention:
        for key in ("auc_training", "auc_validation", "auc_external"):
            if metrics.get(key) is not None:
                metrics[key] = None

    dataset_ids = result.get("dataset_ids")
    if isinstance(dataset_ids, list):
        cleaned = [d for d in dataset_ids if isinstance(d, str) and "<" not in d and ">" not in d]
        result["dataset_ids"] = cleaned or None

    return result


def extract_paper_structured(
    abstract: str,
    llm: BaseChatModel,
    pmid: str = "",
    title: str = "",
) -> Dict[str, Any]:
    """
    Extract structured fields from a PubMed paper abstract.

    Args:
        abstract: The paper abstract text.
        llm:      LangChain chat model.
        pmid:     PubMed ID (optional, included in context for traceability).
        title:    Paper title (optional).

    Returns:
        Structured dict with cancer type, sample type, dataset IDs,
        performance metrics, markers, etc.
        Falls back to a minimal dict with needs_human_review=True on parse error.
    """
    context_lines = []
    if pmid:
        context_lines.append(f"PMID: {pmid}")
    if title:
        context_lines.append(f"Title: {title}")
    context_lines.append(f"\nAbstract:\n{abstract}")
    context = "\n".join(context_lines)

    messages = [
        SystemMessage(content=_EXTRACTOR_SYSTEM),
        HumanMessage(content=context),
    ]
    response = llm.invoke(messages)
    content = _strip_json_fences(response.content)

    try:
        result = json.loads(content)
        result = _sanitize_extracted(result, abstract)
        # Ensure traceability fields are set
        if pmid:
            result.setdefault("pmid", pmid)
        if title:
            result.setdefault("title", title)
        return result
    except json.JSONDecodeError:
        return {
            "pmid": pmid,
            "title": title,
            "sample_type": "unknown",
            "confidence_level": "low",
            "needs_human_review": True,
            "reason": "JSON parse failed — LLM response was not valid JSON.",
            "_raw_llm_output": content[:500],
        }


# ============================================================
# Quick test (run directly: python tools/query_clarifier.py)
# ============================================================

if __name__ == "__main__":
    from tools.parser_tools import parse_query_rules

    print("=== Test: build_ncbi_safe_pubmed_query ===\n")

    test_cases = [
        "find methylation data for cancer",
        "find CRC plasma cfDNA methylation datasets with healthy controls, 450k or WGBS, 2018-2024",
        "结直肠癌血浆cfDNA甲基化，需要健康对照，450K平台",
    ]

    for query in test_cases:
        intent = parse_query_rules(query)
        require_controls = _intent_mentions_controls(intent)
        pubmed_q = build_ncbi_safe_pubmed_query(intent, require_controls=require_controls)
        control_qs = build_pubmed_query_with_controls(intent)

        print(f"Query: {query}")
        print(f"  Intent cancer_type : {intent.get('cancer_type_code')} / {intent.get('cancer_type_display')}")
        print(f"  Intent sample_types: {intent.get('sample_types')}")
        print(f"  Require controls   : {require_controls}")
        print(f"  PubMed query       : {pubmed_q}")
        print(f"  Query length       : {len(pubmed_q)} chars {'✓' if len(pubmed_q) < 400 else '✗ TOO LONG'}")
        print(f"  Control main query : {control_qs['main'][:120]}...")
        print()