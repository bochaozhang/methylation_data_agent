"""
LLM Reviewer Agent for MethyAgent — second-pass consistency check on the
output of extract_paper_structured() (tools/query_clarifier.py).

Why a second LLM call instead of extending the existing regex validators
(_validate_auc_sample_type_cooccurrence, _exclude_reference_datasets in
tools/query_clarifier.py):

  - The regex same-sentence check for AUC/sample_type co-occurrence is
    brittle: a correctly-linked AUC value that happens to be described one
    sentence away from its sample-type mention gets nulled as a false
    positive, and a wrong value that happens to share a sentence with the
    right sample-type keyword slips through as a false negative.
  - The reference-dataset keyword list ("reference", "background",
    "normalization", ...) only catches datasets whose exclusion is signaled
    by one of those literal words nearby; it has no actual understanding of
    what the dataset was used for.

An LLM given the full abstract and the draft extraction can reason about
both questions the way a human reviewer would, at the cost of one extra LLM
call per paper. This directly addresses Bug 2 (tissue/cfDNA AUC confusion,
PMID 40860669) and Bug 3 (reference-dataset misattribution, same paper), and
functions as the reflection/verification step ZhipuAI's 2026-06-30
consultation flagged as missing (no multi-turn reasoning).

This module does NOT replace the existing regex validators — both still run
inside extract_paper_structured(). review_extraction() runs after them, as
an independent second opinion; it can restore/correct fields the regex
pass got wrong in either direction.

Public API:
    review_extraction(abstract, draft_extraction, llm) -> dict
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from utils.logger import get_logger

logger = get_logger(__name__)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


_REVIEWER_SYSTEM = """You are a biomedical literature review assistant. Another model has \
already extracted structured fields from a paper abstract; your job is to independently \
verify two specific things by re-reading the abstract, and to correct the draft if it is wrong.

CHECK A — AUC / sample_type consistency:
For each non-null value in performance_metrics (auc_training, auc_validation, auc_external),
find where that number is reported in the abstract and determine which sample type it was
actually measured on (tissue/tumor vs plasma cfDNA vs serum cfDNA vs WBC/whole blood, etc).
If that sample type does NOT match the draft's "sample_type" field, the value is wrong for
this record — set it to null in your output. If it matches, keep it. If you cannot locate the
value in the abstract at all, set it to null (do not guess).

CHECK B — dataset_id attribution:
For each accession in dataset_ids, decide whether the abstract describes it as the study's
own primary/analysis data (including cohorts explicitly used to compute a reported
performance metric, e.g. an external validation cohort), OR whether it is cited only as a
reference/background/normalization/noise-filtering panel that the authors used to pre-process
or filter their own data but did not directly analyze as a study cohort. Drop any accession
of the second kind from your output dataset_ids list.

Output ONLY valid JSON, no markdown fences, matching this shape exactly:
{
  "performance_metrics": {
    "auc_training": null,
    "auc_validation": null,
    "auc_external": null
  },
  "dataset_ids": ["<accessions you determined are primary/analysis data>"],
  "needs_human_review": false,
  "reason": "one or two sentences: what you checked, what (if anything) you changed and why"
}

Include every key from performance_metrics that was present in the draft, even if you are
keeping its value unchanged. Set needs_human_review=true if you changed anything, or if you
are not confident about a value even after your check."""


def _build_review_payload(draft_extraction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_type": draft_extraction.get("sample_type"),
        "performance_metrics": draft_extraction.get("performance_metrics") or {},
        "dataset_ids": draft_extraction.get("dataset_ids") or [],
    }


def review_extraction(
    abstract: str,
    draft_extraction: Dict[str, Any],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Independently re-check a draft extraction's AUC/sample_type consistency
    (Bug 2) and dataset_id attribution (Bug 3) against the source abstract.

    Args:
        abstract:         The paper abstract text used to produce draft_extraction.
        draft_extraction: Output dict from extract_paper_structured().
        llm:              LangChain chat model.

    Returns:
        A copy of draft_extraction with performance_metrics/dataset_ids
        corrected where the reviewer found a mismatch, needs_human_review
        set when anything changed, and a "reviewer_verdict" key holding the
        raw reviewer output for traceability. On reviewer failure (bad JSON,
        no metrics/dataset_ids to check), returns draft_extraction unchanged
        except for a needs_human_review flag noting the failure.
    """
    result = dict(draft_extraction)

    metrics = result.get("performance_metrics")
    dataset_ids = result.get("dataset_ids")
    has_metrics_to_check = isinstance(metrics, dict) and any(
        metrics.get(k) is not None for k in ("auc_training", "auc_validation", "auc_external")
    )
    has_datasets_to_check = isinstance(dataset_ids, list) and len(dataset_ids) > 0

    if not abstract or not (has_metrics_to_check or has_datasets_to_check):
        return result  # nothing for the reviewer to usefully check

    payload = _build_review_payload(result)
    context = (
        f"Abstract:\n{abstract}\n\n"
        f"Draft extraction (fields under review only):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(content=_REVIEWER_SYSTEM),
        HumanMessage(content=context),
    ]

    try:
        response = llm.invoke(messages)
        verdict = json.loads(_strip_json_fences(response.content))
    except json.JSONDecodeError:
        result["needs_human_review"] = True
        note = "Reviewer response was not valid JSON; corrections skipped."
        result["reason"] = f"{result.get('reason') or ''} {note}".strip()
        logger.warning(f"extraction_reviewer: unparseable response (PMID {result.get('pmid', '')})")
        return result
    except Exception as e:
        result["needs_human_review"] = True
        note = f"Reviewer call failed ({e}); corrections skipped."
        result["reason"] = f"{result.get('reason') or ''} {note}".strip()
        logger.warning(f"extraction_reviewer: call failed for PMID {result.get('pmid', '')}: {e}")
        return result

    changes: List[str] = []

    reviewed_metrics = verdict.get("performance_metrics")
    if isinstance(reviewed_metrics, dict) and isinstance(result.get("performance_metrics"), dict):
        orig_metrics = result["performance_metrics"]
        for key in ("auc_training", "auc_validation", "auc_external"):
            if key not in reviewed_metrics:
                continue
            new_val = reviewed_metrics[key]
            if new_val != orig_metrics.get(key):
                changes.append(f"{key}: {orig_metrics.get(key)!r} -> {new_val!r}")
                orig_metrics[key] = new_val

    reviewed_dataset_ids = verdict.get("dataset_ids")
    if isinstance(reviewed_dataset_ids, list):
        orig_ids = result.get("dataset_ids") or []
        removed = [d for d in orig_ids if d not in reviewed_dataset_ids]
        if removed:
            result["dataset_ids"] = reviewed_dataset_ids or None
            existing_excluded = result.get("excluded_reference_datasets") or []
            result["excluded_reference_datasets"] = existing_excluded + [
                d for d in removed if d not in existing_excluded
            ]
            changes.append(f"dataset_ids: removed reference-only {removed}")

    if changes:
        logger.info(f"extraction_reviewer corrected PMID {result.get('pmid', '')}: {'; '.join(changes)}")

    if changes or verdict.get("needs_human_review"):
        result["needs_human_review"] = True
        reviewer_reason = verdict.get("reason") or "; ".join(changes)
        result["reason"] = f"{result.get('reason') or ''} [reviewer] {reviewer_reason}".strip()

    result["reviewer_verdict"] = verdict
    return result


# ============================================================
# Quick test. Two modes:
#   python -m tools.extraction_reviewer        -> stubbed LLM, verifies
#       merge/parsing logic only (works with no API key / no network).
#   python -m tools.extraction_reviewer --live -> real get_llm(config["llm"])
#       from config/settings.yaml, verifies actual reviewer reasoning
#       quality. Requires a configured API key (.env) and network access.
# ============================================================

if __name__ == "__main__":
    # PMID 40860669 real abstract (Gao et al., PeerJ 2025) and the known-bad
    # draft extraction observed from the live pipeline (docs/6_30/NCBI_Test_out.rtf).
    _ABSTRACT = (
        "BACKGROUND: Preoperative identification of breast cancer (BC) subtypes is "
        "essential for optimizing treatment strategies and improving patient outcomes. "
        "This study aimed to identify circulating cell-free DNA (cfDNA) methylation "
        "signatures to differentiate triple-negative breast cancer (TNBC) from other BC "
        "subtypes (non-TNBC). METHODS: We initially performed a genome-wide analysis to "
        "identify differentially methylated CpG sites (DMCs; |Δβ| > 0.10 and P < 0.05) between "
        "five TNBC and nine non-TNBC tissues using the Infinium HumanMethylationEPIC "
        "BeadChip. These DMCs were further validated using large-scale data from the "
        "Cancer Genome Atlas (TCGA, n = 774; |Δβ| > 0.25 and P < 0.05), and only CpG "
        "sites with average β values > 0.90 or < 0.10 in white blood cells (GSE50132, n = "
        "233) were retained to minimize potential background methylation interference. "
        "Least absolute shrinkage and selection operator (LASSO) regression was applied "
        "to select optimal markers. Diagnostic performance was assessed by the area under "
        "the receiver operating characteristic curve (AUC), and prognostic value was "
        "evaluated using Cox regression analysis. A multiplex digital droplet PCR "
        "(mddPCR) assay was developed to simultaneously detect cg06268921 and cg23247845 "
        "in cfDNA from TNBC (n = 33) and non-TNBC (n = 80) patients. "
        "RESULTS: We identified 113 DMCs, of which eight were selected as optimal "
        "markers. They effectively discriminated TNBC from non-TNBC tissues. Then an "
        "eight-marker diagnostic panel was developed with an AUC of 0.922 in TCGA and "
        "0.875 in GSE69914. Among them, cg06268921 was significantly associated with "
        "overall survival (hazard ratio = 0.249, P = 0.044) and disease-free survival "
        "(hazard ratio = 0.194, P = 0.015) in the TCGA-TNBC cohort. In the cfDNA cohort, "
        "cg06268921 significantly differentiated TNBC from non-TNBC (P < 0.001), and the "
        "combination of both markers yielded an AUC of 0.728. The findings demonstrated "
        "the potential of methylation signatures as non-invasive diagnostic tools for "
        "TNBC. Future research with larger cohorts is warranted."
    )

    _DRAFT = {
        "pmid": "40860669",
        "sample_type": "plasma_cfdna",
        "performance_metrics": {
            "auc_training": None,
            "auc_validation": 0.922,
            "auc_external": None,
        },
        "dataset_ids": ["GSE50132", "TCGA", "GSE69914"],
        "confidence_level": "medium",
        "needs_human_review": False,
        "reason": "",
    }

    class _FakeLLM:
        """Stub chat model — no network/API key available in this sandbox.

        Returns the response a correct reviewer SHOULD produce for the fixture
        above, so this test exercises review_extraction()'s parsing/merge
        logic end-to-end. It does NOT validate the reviewer prompt's actual
        reasoning quality — that requires a live LLM call against a real
        backend (see README note printed below).
        """

        def invoke(self, messages):
            class _Resp:
                content = json.dumps({
                    "performance_metrics": {
                        "auc_training": None,
                        "auc_validation": None,  # 0.922 is the TCGA/tissue AUC, not cfDNA
                        "auc_external": None,
                    },
                    "dataset_ids": ["TCGA", "GSE69914"],  # GSE50132 is a WBC background filter panel
                    "needs_human_review": True,
                    "reason": (
                        "auc_validation=0.922 is reported for the TCGA tissue cohort, not the "
                        "plasma_cfdna sample_type assigned to this record (real cfDNA AUC is "
                        "0.728, in a different sentence); nulled. GSE50132 is cited only as a "
                        "background/noise-filtering WBC panel, not analyzed data; removed."
                    ),
                })
            return _Resp()

    if "--live" in sys.argv:
        import yaml
        from utils.llm_factory import get_llm

        cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        llm = get_llm(cfg["llm"])
        print(f"=== extraction_reviewer LIVE test (PMID 40860669) — backend={cfg['llm']['backend']} ===\n")
    else:
        llm = _FakeLLM()
        print("=== extraction_reviewer mock test (PMID 40860669) ===")
        print("NOTE: no LLM API key is available in this sandbox — using a stubbed LLM that")
        print("returns the expected-correct reviewer verdict, to exercise the merge/parsing")
        print("logic only. Re-run with `--live` on a machine with API access to validate")
        print("actual reviewer reasoning quality.\n")

    corrected = review_extraction(_ABSTRACT, _DRAFT, llm)
    print(json.dumps(corrected, ensure_ascii=False, indent=2))

    if "--live" not in sys.argv:
        assert corrected["performance_metrics"]["auc_validation"] is None, "auc_validation should be nulled"
        assert "GSE50132" not in (corrected["dataset_ids"] or []), "GSE50132 should be removed"
        assert "GSE50132" in corrected.get("excluded_reference_datasets", []), "GSE50132 should be logged as excluded"
        assert corrected["needs_human_review"] is True
        print("\nAll assertions passed (merge/parsing logic verified against stubbed response).")
    else:
        print("\n--- Manually check above: did auc_validation get nulled? Was GSE50132 removed? ---")
