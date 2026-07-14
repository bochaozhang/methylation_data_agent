"""
Mock end-to-end test for agents/orchestrator_v2.py (P0-5).

No LLM API key or network/NCBI proxy is available in this environment
(pipeline_prototype.py's docstring confirms real runs normally happen over
SSH on a remote server), so this script proves the AGENTIC WIRING works —
that the top-level LLM genuinely drives which of the three tools get called,
in what order, based on tool results — using:

  - A scripted fake tool-calling chat model for the top-level ReAct loop
    (decides: call search_papers -> call evaluate_geo_dataset_tool for the
    GSE accession found -> call write_to_registry -> final summary).
  - Monkeypatched search_and_extract() / evaluate_geo_dataset() /
    GEOClient.get_series_metadata() so no real PubMed/GEO/LLM network calls
    happen inside the tools themselves. The canned search_and_extract
    result is the real PMID 40860669 abstract's extraction after both bug
    fixes (regex validators + LLM reviewer) — see tools/extraction_reviewer.py.
  - A real Registry pointed at a temp sqlite file, so write_to_registry
    exercises the actual upsert_dataset() write path (not mocked).

Run: python -m scripts.test_orchestrator_v2

Re-run against a real backend (real get_llm(config["llm"]), real NCBI/GEO
network access) on the SSH server to validate actual tool-selection
reasoning quality — this script only validates the plumbing.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator_v2 import run_methyagent_v2
from registry.registry import Registry


class ScriptedToolCallingLLM(FakeMessagesListChatModel):
    """FakeMessagesListChatModel doesn't implement bind_tools(); create_react_agent
    calls model.bind_tools(tools) once at graph-build time, so provide a no-op that
    returns self (the scripted response list is unaffected by tool schemas)."""

    def bind_tools(self, tools, **kwargs):
        return self


# ------------------------------------------------------------------ #
#  Canned tool outputs (real PMID 40860669 abstract, post bug-fix)    #
# ------------------------------------------------------------------ #

_CANNED_PAPER = {
    "pmid": "40860669",
    "title": "Genome-wide discovery of circulating cell-free DNA methylation signatures "
             "for the differential diagnosis of triple-negative breast cancer.",
    "cancer_type": "breast",
    "sample_type": "plasma_cfdna",
    "performance_metrics": {"auc_training": None, "auc_validation": None, "auc_external": None},
    "dataset_ids": ["TCGA", "GSE69914"],
    "excluded_reference_datasets": ["GSE50132"],
    "needs_human_review": True,
    "confidence_level": "medium",
    "reason": "[reviewer] auc_validation nulled (was TCGA tissue AUC, not cfDNA); "
              "GSE50132 removed (WBC background-filter panel, not analyzed data).",
}

_CANNED_GEO_META = {
    "title": "Whole blood DNA methylation reference panel",
    "summary": "Background methylation reference panel used for filtering.",
    "platform_canonical": "EPIC",
    "sample_count": 233,
    "year": 2018,
    "data_type": "methylation array",
}

_CANNED_JUDGMENT = {
    "usable": "no",
    "recommended_action": "exclude",
    "reason": "Reference/background WBC methylation panel cited only for noise "
              "filtering in the source paper, not the study's own case/control cohort.",
}


def _fake_search_and_extract(intent, llm, top_n=5, review=True):
    return [_CANNED_PAPER]


def _fake_evaluate_geo_dataset(dataset_info, cancer_type, sample_types, llm):
    return _CANNED_JUDGMENT


def _fake_get_series_metadata(self, accession):
    return {**_CANNED_GEO_META, "accession": accession}


def main() -> None:
    query = "breast cancer plasma cfDNA methylation EPIC early detection"

    scripted_responses = [
        AIMessage(content="", tool_calls=[{
            "name": "search_papers", "args": {"query": query}, "id": "call_1",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "evaluate_geo_dataset_tool",
            "args": {"accession": "GSE50132", "cancer_type": "breast", "sample_types": "plasma,cfdna"},
            "id": "call_2",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "write_to_registry",
            "args": {
                "accession": "GSE50132", "cancer_type": "breast", "sample_type": "plasma_cfdna",
                "recommended_action": "exclude", "reason": _CANNED_JUDGMENT["reason"], "pmid": "40860669",
            },
            "id": "call_3",
        }]),
        AIMessage(content=(
            "Found 1 paper (PMID 40860669, breast cancer plasma cfDNA). It cites GSE50132, "
            "but evaluation found it's a WBC background-filtering reference panel, not usable "
            "case/control data — excluded and logged to the registry. No other GEO accessions "
            "in this paper's dataset_ids were evaluable GSE series."
        )),
    ]
    llm = ScriptedToolCallingLLM(responses=scripted_responses)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = str(Path(tmpdir) / "test_registry.db")
        tmp_output = str(Path(tmpdir) / "output")

        base_config = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "settings.yaml"))
        base_config["registry"]["db_path"] = tmp_db
        base_config["download"]["output_dir"] = tmp_output
        tmp_config_path = str(Path(tmpdir) / "settings.yaml")
        with open(tmp_config_path, "w") as f:
            yaml.safe_dump(base_config, f)

        with patch("agents.orchestrator_v2.search_and_extract", _fake_search_and_extract), \
             patch("agents.orchestrator_v2.evaluate_geo_dataset", _fake_evaluate_geo_dataset), \
             patch("tools.geo_tools.GEOClient.get_series_metadata", _fake_get_series_metadata):
            report = run_methyagent_v2(query, config_path=tmp_config_path, llm=llm, save_log=True)

        # ---- Verify the registry write actually landed (real Registry, tmp db) ----
        registry = Registry(tmp_db)
        rows = registry.get_all()

    print("=== orchestrator_v2 mock end-to-end test ===\n")
    print(json.dumps({k: v for k, v in report.items() if k != "messages"}, ensure_ascii=False, indent=2))

    print("\n--- message trace ---")
    for m in report["messages"]:
        print(f"  {m['type']}: {m.get('tool_calls') or (m['content'][:120] if m['content'] else '')}")

    print(f"\n--- registry rows written (tmp db) ---")
    for row in rows:
        print(f"  {row.get('accession')} action={row.get('recommended_action')} needs_review={row.get('needs_review')}")

    assert report["papers_found"] == 1
    assert len(report["gse_evaluated"]) == 1 and report["gse_evaluated"][0]["accession"] == "GSE50132"
    assert report["registry_writes"] == ["GSE50132"]
    assert report["agent_summary"], "agent should produce a final natural-language summary"
    assert len(rows) == 1 and rows[0]["accession"] == "GSE50132"
    assert rows[0]["recommended_action"] == "exclude"
    print("\nAll assertions passed: agent called search_papers -> evaluate_geo_dataset_tool "
          "-> write_to_registry in order, and the registry write is real (tmp db).")
    print(f"Run log saved at: {report.get('log_path')} (inside tmpdir, printed above; see also copy under docs/07_03/)")


if __name__ == "__main__":
    main()
