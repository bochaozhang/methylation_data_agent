"""
NCBI PubMed search runner and two-stage paper filter for MethyAgent.

Pipeline:
    intent (from parse_query_with_llm)
        → build_pubmed_query_with_controls()  [3 query variants]
        → esearch  (PMIDs)
        → efetch   (abstracts + titles)
        → Stage 1  rule-based keyword filter  (fast, no LLM)
        → Stage 2  LLM extraction via extract_paper_structured()
        → sorted structured records

Public API:
    fetch_pubmed_records(query, max_results) → list[{pmid, title, abstract}]
    stage1_filter(records)                  → list[{pmid, title, abstract}]
    search_and_extract(intent, llm, top_n)  → list[dict]  (fully structured)
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests
from langchain_core.language_models import BaseChatModel

from tools.parser_tools import parse_query_rules
from tools.query_clarifier import (
    build_pubmed_query_with_controls,
    extract_paper_structured,
)

# ------------------------------------------------------------------ #
#  NCBI E-utilities configuration                                     #
# ------------------------------------------------------------------ #

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ESEARCH_URL = f"{_EUTILS_BASE}/esearch.fcgi"
_EFETCH_URL  = f"{_EUTILS_BASE}/efetch.fcgi"

# NCBI allows 3 req/s without a key, 10/s with one.
_REQUEST_DELAY = 0.34   # seconds between requests (no-key safe)

_DEFAULT_MAX_RESULTS = 20
_FETCH_BATCH_SIZE    = 100   # efetch supports up to 500 per call

# ------------------------------------------------------------------ #
#  Shared requests session (proxy-aware, same pattern as GEOClient)  #
# ------------------------------------------------------------------ #

def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "MethyAgent/1.0 (methylation data collector)"})
    return session

_SESSION: requests.Session = _build_session()


def _resolve_proxy() -> str:
    """
    Resolve the NCBI proxy URL using the same priority as database_agent.py:
      1. NCBI_PROXY env var
      2. geo.proxy in config/settings.yaml
      3. Standard HTTP_PROXY / HTTPS_PROXY env vars
    Result is cached after the first call.
    """
    if hasattr(_resolve_proxy, "_cache"):
        return _resolve_proxy._cache  # type: ignore[attr-defined]

    proxy = os.environ.get("NCBI_PROXY", "")

    if not proxy:
        try:
            import yaml
            from pathlib import Path
            cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            proxy = cfg.get("geo", {}).get("proxy", "")
        except Exception:
            pass

    if not proxy:
        proxy = (
            os.environ.get("HTTPS_PROXY", "")
            or os.environ.get("HTTP_PROXY", "")
            or os.environ.get("ALL_PROXY", "")
        )

    _resolve_proxy._cache = proxy  # type: ignore[attr-defined]
    return proxy


def _apply_proxy(session: requests.Session) -> None:
    """Apply proxy to session. Mirrors database_agent.py GEOClient setup."""
    proxy = _resolve_proxy()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    else:
        session.proxies = {}

# ------------------------------------------------------------------ #
#  Stage-1 filter keyword sets                                        #
# ------------------------------------------------------------------ #

_METHYLATION_KEEP = {
    "methylation", "methylated", "methylome", "cpg", "dnmt",
    "bisulfite", "epic", "450k", "850k", "wgbs", "rrbs", "mcta",
}

_LIQUID_BIOPSY_KEEP = {
    "cfdna", "cell-free dna", "cell free dna", "ctdna", "circulating dna",
    "circulating tumor dna", "plasma", "liquid biopsy", "blood plasma",
    "serum", "circulating methylation",
}

_TISSUE_EXCLUDE = {
    "cell line", "cell lines", "in vitro", "xenograft", "pdx",
    "patient-derived xenograft", "organoid", "organoids",
    "mouse model", "murine", "rat model",
    "dnmti", "5-azacytidine", "decitabine", "drug resistance",
    "drug treatment", "radiation treatment",
}


# ------------------------------------------------------------------ #
#  NCBI HTTP helpers                                                  #
# ------------------------------------------------------------------ #

def _ncbi_params() -> Dict[str, str]:
    """
    Return common NCBI E-utilities params.
    tool + email are required to avoid IP bans (misuse.ncbi.nlm.nih.gov redirect).
    See: https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requiremen
    """
    params: Dict[str, str] = {
        "tool":  "MethyAgent",
        "email": os.environ.get("GEO_EMAIL", "methyagent@research.local"),
    }
    key = os.environ.get("NCBI_API_KEY", "")
    if key:
        params["api_key"] = key
    return params


def _get(url: str, params: Dict[str, Any], timeout: int = 30) -> requests.Response:
    """GET with retry (2 attempts, 1-second back-off) through shared proxy session."""
    _apply_proxy(_SESSION)
    for attempt in range(2):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()

            # Detect NCBI abuse/block redirect (same check as GEOClient)
            if "misuse.ncbi.nlm.nih.gov" in resp.url or "abuse.shtml" in resp.url:
                raise RuntimeError(
                    "NCBI blocked this request (abuse redirect). "
                    "Check that the proxy tunnel is running: "
                    "bash /home/ubuntu/bochaozhang/proxy.sh"
                )

            # Detect empty body — proxy not routing or server IP blocked
            if not resp.text.strip():
                proxy = os.environ.get("NCBI_PROXY", "(not set)")
                raise RuntimeError(
                    f"NCBI returned an empty response. "
                    f"NCBI_PROXY={proxy}. "
                    f"Ensure the proxy tunnel is running and NCBI_PROXY is set in .env."
                )

            return resp
        except RuntimeError:
            raise
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(1)
            else:
                raise RuntimeError(f"NCBI request failed: {exc}") from exc
    raise RuntimeError("NCBI request failed after retry")  # unreachable


# ------------------------------------------------------------------ #
#  esearch — get PMIDs                                               #
# ------------------------------------------------------------------ #

def esearch_pmids(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> List[str]:
    """
    Call NCBI esearch and return a list of PMIDs matching *query*.

    Args:
        query:       PubMed boolean query string.
        max_results: Maximum number of PMIDs to retrieve (default 20).

    Returns:
        List of PMID strings, empty list on failure.
    """
    params = {
        **_ncbi_params(),
        "db":      "pubmed",
        "term":    query,
        "retmax":  str(max_results),
        "retmode": "json",
    }
    time.sleep(_REQUEST_DELAY)
    resp = _get(_ESEARCH_URL, params)
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(
            f"NCBI esearch returned non-JSON (status={resp.status_code}). "
            f"First 200 chars: {resp.text[:200]!r}"
        )
    return data.get("esearchresult", {}).get("idlist", [])


# ------------------------------------------------------------------ #
#  efetch — get abstracts                                            #
# ------------------------------------------------------------------ #

def _parse_efetch_xml(xml_text: str) -> List[Dict[str, str]]:
    """
    Parse PubMed efetch XML and return list of {pmid, title, abstract}.
    Handles missing or multi-paragraph abstracts gracefully.
    """
    records: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return records

    for article in root.findall(".//PubmedArticle"):
        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

        # Title
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract — may have multiple <AbstractText> nodes (labeled sections)
        abstract_parts: List[str] = []
        for ab in article.findall(".//AbstractText"):
            label = ab.get("Label")
            text  = "".join(ab.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)

        if pmid:
            records.append({"pmid": pmid, "title": title, "abstract": abstract})

    return records


def efetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    """
    Fetch title + abstract for a list of PMIDs.

    Batches into groups of _FETCH_BATCH_SIZE to avoid URL length limits.
    Returns list of {pmid, title, abstract}.
    """
    if not pmids:
        return []

    results: List[Dict[str, str]] = []
    for i in range(0, len(pmids), _FETCH_BATCH_SIZE):
        batch = pmids[i : i + _FETCH_BATCH_SIZE]
        params = {
            **_ncbi_params(),
            "db":      "pubmed",
            "id":      ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        time.sleep(_REQUEST_DELAY)
        resp = _get(_EFETCH_URL, params)
        results.extend(_parse_efetch_xml(resp.text))

    return results


def fetch_pubmed_records(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
) -> List[Dict[str, str]]:
    """
    End-to-end: search PubMed and fetch abstracts.

    Args:
        query:       PubMed boolean query string.
        max_results: Maximum papers to return.

    Returns:
        List of {pmid, title, abstract}.
    """
    pmids = esearch_pmids(query, max_results=max_results)
    if not pmids:
        return []
    return efetch_abstracts(pmids)


# ------------------------------------------------------------------ #
#  Stage 1 — rule-based keyword filter (fast, no LLM)               #
# ------------------------------------------------------------------ #

def stage1_filter(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Fast rule-based filter. Keep a record if:
      - Contains at least one methylation keyword AND
      - Contains at least one liquid-biopsy / plasma keyword

    Exclude if:
      - Contains any cell-line / animal-model keyword.

    Args:
        records: List of {pmid, title, abstract} dicts.

    Returns:
        Filtered list (subset of input, order preserved).
    """
    kept: List[Dict[str, str]] = []
    for rec in records:
        text = (rec.get("title", "") + " " + rec.get("abstract", "")).lower()

        # Hard exclude
        if any(kw in text for kw in _TISSUE_EXCLUDE):
            continue

        # Must have methylation signal
        has_meth = any(kw in text for kw in _METHYLATION_KEEP)
        if not has_meth:
            continue

        # Must have liquid biopsy / plasma signal
        has_liquid = any(kw in text for kw in _LIQUID_BIOPSY_KEEP)
        if not has_liquid:
            continue

        kept.append(rec)

    return kept


# ------------------------------------------------------------------ #
#  Stage 2 — LLM structured extraction                              #
# ------------------------------------------------------------------ #

_STAGE2_MAX = 10  # cap LLM calls per search_and_extract invocation


def stage2_extract(
    records: List[Dict[str, str]],
    llm: BaseChatModel,
    max_records: int = _STAGE2_MAX,
) -> List[Dict[str, Any]]:
    """
    Run extract_paper_structured() on each record passing Stage 1.
    Capped at max_records to limit LLM calls (each call ~3-8s).
    Prints progress so callers can see it's working.
    """
    capped = records[:max_records]
    if len(records) > max_records:
        print(f"  [stage2] {len(records)} records passed Stage 1, capping at {max_records}")

    structured: List[Dict[str, Any]] = []
    for i, rec in enumerate(capped, 1):
        pmid = rec.get("pmid", "")
        print(f"  [stage2] {i}/{len(capped)} LLM extract — PMID={pmid} ...", flush=True)
        result = extract_paper_structured(
            abstract=rec.get("abstract", ""),
            llm=llm,
            pmid=pmid,
            title=rec.get("title", ""),
        )
        result.setdefault("pmid", pmid)
        result.setdefault("title", rec.get("title", ""))
        structured.append(result)

    return structured


# ------------------------------------------------------------------ #
#  Confidence sort helper                                             #
# ------------------------------------------------------------------ #

_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def _sort_by_confidence(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: _CONFIDENCE_ORDER.get(r.get("confidence_level", "low"), 2),
    )


# ------------------------------------------------------------------ #
#  Pass-rate / completeness summary                                   #
# ------------------------------------------------------------------ #

# field -> predicate that returns True if the structured record has a
# non-null/non-empty value for that field
_COMPLETENESS_CHECKS = {
    "auc (any)": lambda r: any(
        (r.get("performance_metrics") or {}).get(k) is not None
        for k in ("auc_training", "auc_validation", "auc_external")
    ),
    "dataset_ids": lambda r: bool(r.get("dataset_ids")),
    "has_normal_control": lambda r: r.get("has_normal_control") is not None,
    "sample_type": lambda r: r.get("sample_type") not in (None, "unknown"),
    "markers_or_panel": lambda r: bool(r.get("markers_or_panel")),
    "data_availability": lambda r: r.get("data_availability") not in (None, "unknown"),
}


def _print_pipeline_summary(
    total_fetched: int,
    passed_s1: int,
    structured: List[Dict[str, Any]],
) -> None:
    extracted = len(structured)
    rejected_s1 = total_fetched - passed_s1

    print(f"\n{'=' * 60}")
    print("Pipeline summary")
    print('=' * 60)
    print(f"  Total fetched:     {total_fetched}")
    print(f"  Stage 1 passed:    {passed_s1}")
    print(f"  Stage 1 rejected:  {rejected_s1}")
    print(f"  Stage 2 extracted: {extracted}")

    if extracted:
        print("\n  Per-field completeness (Stage 2 output):")
        for label, check in _COMPLETENESS_CHECKS.items():
            n_complete = sum(1 for r in structured if check(r))
            pct = n_complete / extracted * 100
            print(f"    {label:20s} {n_complete}/{extracted}  ({pct:.0f}%)")
    print('=' * 60)


# ------------------------------------------------------------------ #
#  Main entry point                                                   #
# ------------------------------------------------------------------ #

def search_and_extract(
    intent: Dict[str, Any],
    llm: BaseChatModel,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Full pipeline: intent → PubMed queries → fetch → filter → LLM extract.

    Args:
        intent: Parsed intent dict from parse_query_with_llm() or parse_query_rules().
        llm:    LangChain chat model (used in Stage 2 and query building).
        top_n:  Maximum structured records to return per variant query
                before deduplication (total results may be up to 3×top_n).

    Returns:
        Deduplicated list of structured paper records sorted by confidence_level
        (high → medium → low).
    """
    # Build 3 query variants
    queries = build_pubmed_query_with_controls(intent)

    # Fetch records for each variant, deduplicate by PMID
    seen_pmids: set = set()
    all_raw: List[Dict[str, str]] = []

    for variant_name, query in queries.items():
        raw = fetch_pubmed_records(query, max_results=top_n)
        for rec in raw:
            pmid = rec.get("pmid", "")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_raw.append(rec)

    # Stage 1: rule-based filter
    passed_s1 = stage1_filter(all_raw)

    # Stage 2: LLM extraction
    structured = stage2_extract(passed_s1, llm)

    _print_pipeline_summary(len(all_raw), len(passed_s1), structured)

    # Sort by confidence
    return _sort_by_confidence(structured)


# ------------------------------------------------------------------ #
#  Quick smoke test (python tools/ncbi_search.py)                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import json
    import sys
    import yaml
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from tools.parser_tools import parse_query_rules
    from utils.llm_factory import get_llm

    cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    llm = get_llm(cfg["llm"])

    TEST_QUERIES = [
        "结直肠癌血浆cfDNA甲基化，需要健康对照",
        "breast cancer plasma cfDNA methylation EPIC early detection",
    ]

    for q in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        intent = parse_query_rules(q)
        print(f"  cancer: {intent.get('cancer_type_code')} | "
              f"sample: {intent.get('sample_types')}")

        results = search_and_extract(intent, llm, top_n=5)
        print(f"  Returned {len(results)} structured records")
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] PMID={r.get('pmid')}  confidence={r.get('confidence_level')}")
            print(f"      title:   {r.get('title', '')[:80]}")
            print(f"      cancer:  {r.get('cancer_type')} | sample: {r.get('sample_type')}")
            print(f"      control: {r.get('has_normal_control')} | "
                  f"n_case={r.get('sample_size_case')} n_ctrl={r.get('sample_size_control')}")
            metrics = r.get("performance_metrics") or {}
            print(f"      AUC val: {metrics.get('auc_validation')} | "
                  f"AUC ext: {metrics.get('auc_external')}")
            print(f"      dataset: {r.get('dataset_ids')}")
