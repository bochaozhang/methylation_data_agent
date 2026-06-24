"""
Live integration test for query_clarifier.py Components 1, 4, 5.

Usage (on SSH server):
    set -a && source .env && set +a
    source .venv/bin/activate
    python scripts/test_components_145.py

Backend: reads config/settings.yaml — defaults to zhipu (GLM-4-flash).
To switch to Anthropic, edit settings.yaml: backend: anthropic

Components tested:
    1. ask_clarifying_questions()    — vague query → structured follow-up Qs
    4. evaluate_geo_dataset()        — GEO metadata string → keep/exclude/manual_review
    5. extract_paper_structured()    — PubMed abstract → structured dict
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.query_clarifier import (
    ClarificationResult,
    ask_clarifying_questions,
    evaluate_geo_dataset,
    extract_paper_structured,
    format_clarifying_questions,
)
from utils.llm_factory import get_llm


# ------------------------------------------------------------------ #
#  Shared setup                                                       #
# ------------------------------------------------------------------ #

def load_llm():
    cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    llm = get_llm(cfg["llm"])
    print(f"[setup] backend={cfg['llm']['backend']}  model={cfg['llm'].get('model','?')}\n")
    return llm


def _separator(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)


# ------------------------------------------------------------------ #
#  Component 1 — ask_clarifying_questions                            #
# ------------------------------------------------------------------ #

CLARIFIER_TESTS = [
    # (label, query, expect_specific)
    (
        "vague query (should ask questions)",
        "I need methylation data for cancer",
        False,
    ),
    (
        "specific query (should NOT ask questions)",
        "结直肠癌血浆cfDNA甲基化 450K 2018-2023 需要健康对照",
        True,
    ),
    (
        "partial query — cancer known, sample type missing",
        "breast cancer methylation datasets, need healthy controls",
        None,   # either outcome is valid — just show result
    ),
]


def test_component1(llm):
    _separator("Component 1: ask_clarifying_questions")
    all_pass = True
    for label, query, expect_specific in CLARIFIER_TESTS:
        print(f"\n--- Test: {label}")
        print(f"    Input: {query!r}")
        result: ClarificationResult = ask_clarifying_questions(query, llm)
        print(f"    is_specific_enough : {result.is_specific_enough}")
        print(f"    missing_dimensions : {result.missing_dimensions}")
        print(f"    num questions      : {len(result.questions)}")
        if result.questions:
            formatted = format_clarifying_questions(result)
            print(f"    formatted output:\n{formatted}")
        if expect_specific is not None:
            ok = result.is_specific_enough == expect_specific
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{status}] expected is_specific_enough={expect_specific}, "
                  f"got {result.is_specific_enough}")
    return all_pass


# ------------------------------------------------------------------ #
#  Component 4 — evaluate_geo_dataset                                #
# ------------------------------------------------------------------ #

GEO_TESTS = [
    (
        "KEEP — CRC plasma cfDNA 450K with healthy controls",
        """GSE149212 — Title: DNA methylation profiling of plasma cfDNA from colorectal cancer patients
Series: 70 CRC patients + 50 healthy donors
Platform: GPL13534 (Illumina HumanMethylation450)
Organism: Homo sapiens
Sample types: EDTA plasma, cell-free DNA
Data files: GSE149212_beta_matrix.txt.gz (470K probes × 120 samples)
Sample annotation: detailed (stage I-IV, age, sex, treatment-naive)
Citation: Smith et al. 2021, Clinical Cancer Research""",
        "colorectal cancer",
        ["cfdna", "plasma"],
        "keep",
    ),
    (
        "EXCLUDE — cell line data",
        """GSE88836 — Title: Methylation changes in CRC cell lines after DNMT inhibitor treatment
Series: 20 HCT-116 cell line samples treated with 5-azacytidine
Platform: GPL13534 (HumanMethylation450)
Organism: Homo sapiens
Data: processed beta matrix available
Notes: in vitro experiment, drug treatment""",
        "colorectal cancer",
        ["cfdna", "plasma"],
        "exclude",
    ),
    (
        "MANUAL_REVIEW — only cancer cases, no healthy controls",
        """GSE131013 — Title: cfDNA methylation in early-stage NSCLC
Series: 45 lung cancer patients (stage I-II), no healthy donor controls included
Platform: targeted bisulfite sequencing panel (100 CpG sites)
Sample type: plasma cfDNA
Data: supplementary Table S2 (average beta per patient, 100 CpGs)""",
        "lung cancer",
        ["cfdna", "plasma"],
        "manual_review",
    ),
]


def test_component4(llm):
    _separator("Component 4: evaluate_geo_dataset")
    all_pass = True
    for label, dataset_info, cancer_type, sample_types, expected_action in GEO_TESTS:
        print(f"\n--- Test: {label}")
        result = evaluate_geo_dataset(dataset_info, cancer_type, sample_types, llm)
        action = result.get("recommended_action", "?")
        usable = result.get("usable", "?")
        reason = result.get("reason", "")
        print(f"    recommended_action : {action}")
        print(f"    usable             : {usable}")
        print(f"    reason             : {reason[:120]}")
        ok = action == expected_action
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{status}] expected={expected_action}, got={action}")
    return all_pass


# ------------------------------------------------------------------ #
#  Component 5 — extract_paper_structured                            #
# ------------------------------------------------------------------ #

ABSTRACT_TESTS = [
    (
        "High-confidence — AUC + dataset IDs + healthy controls",
        "39123456",
        "Plasma cfDNA methylation panel for early colorectal cancer detection",
        """We performed EPIC array profiling (GSE198765) on plasma cfDNA from
120 colorectal cancer patients and 80 healthy donors. A 12-CpG panel
(including cg07779434/SEPT9, cg19859270/VIM, cg11938475/SFRP2) achieved
AUC=0.94 (validation cohort, n=150; AUC_external=0.89 in an independent
dataset GSE200123). Sensitivity was 82% at 97% specificity for stage I-II CRC.
Data available at GEO under accession GSE198765.""",
        {
            "check_has_normal_control": True,
            "check_confidence": "high",
            "check_dataset_ids": ["GSE198765"],
        },
    ),
    (
        "Medium-confidence — methylation + cancer mentioned, no AUC",
        "38000001",
        "Methylation alterations in breast cancer tissue",
        """We analyzed DNA methylation in breast tumor tissue vs adjacent normal
tissue from 45 patients using the 450K array. Differential methylation was
observed at BRCA1 and RASSF1A promoters. Data will be available upon request.""",
        {
            "check_has_normal_control": True,
            "check_confidence": "medium",
        },
    ),
    (
        "Low-confidence — vague, no dataset",
        "37000001",
        "Review: Epigenetic regulation in cancer",
        """DNA methylation is an important epigenetic mechanism in cancer development.
Aberrant methylation of tumor suppressor genes has been reported in multiple
cancer types. Future studies should focus on liquid biopsy applications.""",
        {
            "check_confidence": "low",
        },
    ),
]


def test_component5(llm):
    _separator("Component 5: extract_paper_structured")
    all_pass = True
    for label, pmid, title, abstract, checks in ABSTRACT_TESTS:
        print(f"\n--- Test: {label}")
        result = extract_paper_structured(abstract, llm, pmid=pmid, title=title)
        print(f"    pmid             : {result.get('pmid')}")
        print(f"    cancer_type      : {result.get('cancer_type')}")
        print(f"    sample_type      : {result.get('sample_type')}")
        print(f"    has_normal_ctrl  : {result.get('has_normal_control')}")
        print(f"    confidence_level : {result.get('confidence_level')}")
        print(f"    dataset_ids      : {result.get('dataset_ids')}")
        metrics = result.get("performance_metrics") or {}
        print(f"    AUC_validation   : {metrics.get('auc_validation')}")
        print(f"    AUC_external     : {metrics.get('auc_external')}")
        print(f"    markers          : {result.get('markers_or_panel')}")
        print(f"    needs_review     : {result.get('needs_human_review')}")
        print(f"    reason           : {result.get('reason', '')[:100]}")

        test_ok = True
        if "check_has_normal_control" in checks:
            exp = checks["check_has_normal_control"]
            got = result.get("has_normal_control")
            if got != exp:
                print(f"    [FAIL] has_normal_control: expected={exp}, got={got}")
                test_ok = False
        if "check_confidence" in checks:
            exp = checks["check_confidence"]
            got = result.get("confidence_level")
            if got != exp:
                print(f"    [FAIL] confidence_level: expected={exp}, got={got}")
                test_ok = False
        if "check_dataset_ids" in checks:
            exp_ids = set(checks["check_dataset_ids"])
            got_ids = set(result.get("dataset_ids") or [])
            if not exp_ids.issubset(got_ids):
                print(f"    [FAIL] dataset_ids: expected subset {exp_ids}, got {got_ids}")
                test_ok = False

        if not test_ok:
            all_pass = False
        if test_ok:
            print(f"    [PASS]")

    return all_pass


# ------------------------------------------------------------------ #
#  Runner                                                             #
# ------------------------------------------------------------------ #

def main():
    print("MethyAgent — Live test: Components 1, 4, 5")
    print("─" * 60)

    llm = load_llm()

    results = {
        "component1": test_component1(llm),
        "component4": test_component4(llm),
        "component5": test_component5(llm),
    }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    overall = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL (check output above)"
        print(f"  {name}: {status}")
        if not ok:
            overall = False

    if overall:
        print("\nAll components passed.")
    else:
        print("\nSome tests failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
