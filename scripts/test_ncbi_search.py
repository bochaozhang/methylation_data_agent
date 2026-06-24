"""
Live integration tests for tools/ncbi_search.py.

Usage (on SSH server):
    set -a && source .env && set +a
    # proxy must already be running: bash /home/ubuntu/bochaozhang/proxy.sh
    source .venv/bin/activate
    python scripts/test_ncbi_search.py

Tests:
    1. esearch   — NCBI esearch returns >0 PMIDs for a known query
    2. efetch    — abstract + title present for first 3 PMIDs
    3. stage1    — rule-based filter: cell line and mouse records excluded
    4. end-to-end search_and_extract() — CRC plasma cfDNA intent → structured records
    5. Chinese query end-to-end — 结直肠癌血浆cfDNA甲基化 → same output shape
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ncbi_search import (
    efetch_abstracts,
    esearch_pmids,
    search_and_extract,
    stage1_filter,
)
from tools.parser_tools import parse_query_rules
from utils.llm_factory import get_llm


# ------------------------------------------------------------------ #
#  Setup                                                              #
# ------------------------------------------------------------------ #

def load_llm():
    cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # NOTE: pass config["llm"], NOT the full config dict
    llm = get_llm(cfg["llm"])
    proxy = os.environ.get("NCBI_PROXY", "(not set)")
    print(f"[setup] backend={cfg['llm']['backend']}  model={cfg['llm'].get('model','?')}")
    print(f"[setup] NCBI_PROXY={proxy}\n")
    return llm


def _separator(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)


# ------------------------------------------------------------------ #
#  Test 1 — esearch                                                   #
# ------------------------------------------------------------------ #

def test_esearch() -> bool:
    _separator("Test 1: esearch — PMIDs for known PubMed query")
    query = (
        '("colorectal cancer" OR CRC) AND "DNA methylation"[MeSH Terms] '
        'AND ("cell-free DNA" OR cfDNA OR plasma)'
    )
    print(f"  Query: {query[:90]}...")
    pmids = esearch_pmids(query, max_results=10)
    print(f"  PMIDs returned: {len(pmids)}")
    if pmids:
        print(f"  First 5: {pmids[:5]}")
    ok = len(pmids) > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] expected >0 PMIDs, got {len(pmids)}")
    return ok


# ------------------------------------------------------------------ #
#  Test 2 — efetch abstracts                                          #
# ------------------------------------------------------------------ #

def test_efetch() -> bool:
    _separator("Test 2: efetch — title + abstract for 3 PMIDs")
    # Well-known PMIDs for cfDNA methylation CRC papers
    test_pmids = ["28678075", "31327558", "33268327"]
    print(f"  PMIDs: {test_pmids}")
    records = efetch_abstracts(test_pmids)
    print(f"  Records fetched: {len(records)}")
    all_ok = True
    for r in records:
        has_title    = bool(r.get("title", "").strip())
        has_abstract = bool(r.get("abstract", "").strip())
        ok = has_title and has_abstract
        if not ok:
            all_ok = False
        print(f"  PMID {r.get('pmid')}: title={'OK' if has_title else 'MISSING'}  "
              f"abstract={'OK' if has_abstract else 'MISSING'}  "
              f"[{'PASS' if ok else 'FAIL'}]")
        if has_title:
            print(f"    title: {r['title'][:80]}")
    return all_ok


# ------------------------------------------------------------------ #
#  Test 3 — Stage 1 rule-based filter                                 #
# ------------------------------------------------------------------ #

FAKE_RECORDS = [
    {
        "pmid": "A001",
        "title": "Plasma cfDNA methylation markers for colorectal cancer detection",
        "abstract": (
            "We analyzed plasma cfDNA methylation in 100 CRC patients and 80 healthy "
            "controls using EPIC array. The panel achieved AUC=0.92."
        ),
        "_expect": "keep",
    },
    {
        "pmid": "A002",
        "title": "DNA methylation in HCT-116 cell line after drug treatment",
        "abstract": (
            "We treated HCT-116 cell lines with 5-azacytidine and analyzed methylation "
            "changes by 450K array. In vitro results showed promoter hypomethylation."
        ),
        "_expect": "exclude",  # cell line + in vitro
    },
    {
        "pmid": "A003",
        "title": "Methylation profiling in a murine model of colorectal cancer",
        "abstract": (
            "Using a mouse model of CRC, we studied DNA methylation at SEPT9 and VIM "
            "promoters by bisulfite sequencing."
        ),
        "_expect": "exclude",  # mouse model
    },
    {
        "pmid": "A004",
        "title": "Tumor tissue methylation landscape in breast cancer",
        "abstract": (
            "We performed WGBS on breast tumor tissue from 50 patients. "
            "Differential methylation was identified at BRCA1 and RASSF1A promoters."
        ),
        "_expect": "exclude",  # no liquid biopsy / plasma signal
    },
    {
        "pmid": "A005",
        "title": "Serum cfDNA methylation for early lung cancer detection",
        "abstract": (
            "Cell-free DNA from serum was analyzed by targeted bisulfite sequencing "
            "in 60 NSCLC patients and 45 healthy donors."
        ),
        "_expect": "keep",
    },
]


def test_stage1_filter() -> bool:
    _separator("Test 3: stage1_filter — cell line / mouse excluded")
    records = [{k: v for k, v in r.items() if k != "_expect"} for r in FAKE_RECORDS]
    expected_keep = {r["pmid"] for r in FAKE_RECORDS if r["_expect"] == "keep"}
    expected_excl = {r["pmid"] for r in FAKE_RECORDS if r["_expect"] == "exclude"}

    kept = stage1_filter(records)
    kept_pmids = {r["pmid"] for r in kept}

    all_ok = True
    for r in FAKE_RECORDS:
        pmid = r["pmid"]
        exp  = r["_expect"]
        got  = "keep" if pmid in kept_pmids else "exclude"
        ok   = got == exp
        if not ok:
            all_ok = False
        print(f"  PMID {pmid}: expected={exp}  got={got}  [{'PASS' if ok else 'FAIL'}]")
        print(f"    title: {r['title'][:70]}")

    print(f"\n  Kept {len(kept)}/{len(records)} records "
          f"(expected {len(expected_keep)})")
    return all_ok


# ------------------------------------------------------------------ #
#  Test 4 — search_and_extract end-to-end (English)                  #
# ------------------------------------------------------------------ #

REQUIRED_FIELDS = {"pmid", "cancer_type", "sample_type", "confidence_level"}


def _check_result_shape(results: list, label: str) -> bool:
    if not results:
        print(f"  [FAIL] {label}: got 0 results")
        return False

    all_ok = True
    for i, r in enumerate(results, 1):
        missing = REQUIRED_FIELDS - set(r.keys())
        ok = len(missing) == 0
        if not ok:
            all_ok = False
        print(f"\n  [{i}] PMID={r.get('pmid')}  confidence={r.get('confidence_level')}")
        print(f"      title:      {str(r.get('title',''))[:70]}")
        print(f"      cancer:     {r.get('cancer_type')} | sample: {r.get('sample_type')}")
        print(f"      control:    {r.get('has_normal_control')} | "
              f"n_case={r.get('sample_size_case')} n_ctrl={r.get('sample_size_control')}")
        metrics = r.get("performance_metrics") or {}
        print(f"      AUC_val:    {metrics.get('auc_validation')} | "
              f"AUC_ext: {metrics.get('auc_external')}")
        print(f"      datasets:   {r.get('dataset_ids')}")
        if missing:
            print(f"      [FAIL] missing fields: {missing}")
        else:
            print(f"      [PASS] all required fields present")
    return all_ok


def test_search_and_extract_english(llm) -> bool:
    _separator("Test 4: search_and_extract — CRC plasma cfDNA (English)")
    print("  (makes real NCBI + LLM calls — may take 30-60 s)")
    intent = parse_query_rules("CRC plasma cfDNA methylation healthy controls")
    print(f"  Intent: cancer={intent.get('cancer_type_code')}  "
          f"sample={intent.get('sample_types')}")
    results = search_and_extract(intent, llm, top_n=3)
    print(f"  Results: {len(results)} structured records")
    ok = _check_result_shape(results, "English CRC query")
    print(f"\n  [{'PASS' if ok else 'FAIL'}] English query")
    return ok


# ------------------------------------------------------------------ #
#  Test 5 — search_and_extract end-to-end (Chinese query)            #
# ------------------------------------------------------------------ #

def test_search_and_extract_chinese(llm) -> bool:
    _separator("Test 5: search_and_extract — 结直肠癌血浆cfDNA甲基化 (Chinese)")
    print("  (makes real NCBI + LLM calls — may take 30-60 s)")
    intent = parse_query_rules("结直肠癌血浆cfDNA甲基化，需要健康对照")
    print(f"  Intent: cancer={intent.get('cancer_type_code')}  "
          f"sample={intent.get('sample_types')}")
    results = search_and_extract(intent, llm, top_n=3)
    print(f"  Results: {len(results)} structured records")
    ok = _check_result_shape(results, "Chinese CRC query")
    print(f"\n  [{'PASS' if ok else 'FAIL'}] Chinese query")
    return ok


# ------------------------------------------------------------------ #
#  Runner                                                             #
# ------------------------------------------------------------------ #

def main():
    print("MethyAgent — ncbi_search.py integration tests")
    print("─" * 60)

    llm = load_llm()

    results = {}

    # Tests 1-3: no LLM needed
    results["test1_esearch"]       = test_esearch()
    results["test2_efetch"]        = test_efetch()
    results["test3_stage1_filter"] = test_stage1_filter()

    # Tests 4-5: LLM + NCBI (slow)
    results["test4_end_to_end_en"] = test_search_and_extract_english(llm)
    results["test5_end_to_end_zh"] = test_search_and_extract_chinese(llm)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    overall = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            overall = False

    if overall:
        print("\nAll tests passed.")
    else:
        print("\nSome tests failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
