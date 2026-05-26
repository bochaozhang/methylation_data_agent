"""
LangGraph Orchestrator for MethyAgent.

Defines the StateGraph with:
  - parse_query node: LLM-powered intent parsing
  - run_database_agent node: Agent 1 (GEO + TCGA)
  - run_literature_agent node: Agent 2 (PubMed + PMC + bioRxiv)
  - generate_report node: Final summary

Graph flow:
  parse_query → run_database_agent → run_literature_agent → generate_report

Conditional edges handle:
  - Empty results from Agent 1 → skip to report
  - Agent-only mode (--agent db-only or lit-only)
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from agents.database_agent import DatabaseAgent
from agents.literature_agent import LiteratureAgent
from registry.registry import Registry
from state.graph_state import MethyAgentState
from tools.parser_tools import parse_query_with_llm, parse_query_rules
from utils.llm_factory import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load and return the settings.yaml configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Node functions                                                      #
# ------------------------------------------------------------------ #

def make_parse_query_node(config: Dict[str, Any]):
    """Factory: returns the parse_query node function."""
    llm = get_llm(config["llm"])

    def parse_query(state: MethyAgentState) -> MethyAgentState:
        """
        Parse the raw user query into structured intent.
        Uses LLM for semantic queries; falls back to rule-based for explicit accessions.
        """
        raw_query = state["raw_query"]
        logger.info(f"Parsing query: {raw_query}")

        try:
            intent = parse_query_with_llm(raw_query, llm)
        except Exception as e:
            logger.warning(f"LLM parsing failed ({e}), falling back to rule-based parser")
            intent = parse_query_rules(raw_query)

        logger.info(f"Parsed intent: {json.dumps(intent, ensure_ascii=False, indent=2)}")

        return {
            **state,
            "parsed_intent": intent,
            "db_candidates": [],
            "db_downloaded": [],
            "db_failed": [],
            "db_skipped": [],
            "papers_found": [],
            "lit_candidates": [],
            "lit_downloaded": [],
            "lit_failed": [],
            "lit_skipped": [],
            "error_log": [],
            "final_report": {},
            "messages": [
                HumanMessage(content=raw_query),
            ],
        }

    return parse_query


def make_database_agent_node(config: Dict[str, Any], registry: Registry):
    """Factory: returns the run_database_agent node function."""
    agent = DatabaseAgent(config, registry)

    def run_database_agent(state: MethyAgentState) -> MethyAgentState:
        logger.info("=== Running DatabaseAgent (Agent 1) ===")
        return agent.run(state)

    return run_database_agent


def make_literature_agent_node(config: Dict[str, Any], registry: Registry):
    """Factory: returns the run_literature_agent node function."""
    agent = LiteratureAgent(config, registry)

    def run_literature_agent(state: MethyAgentState) -> MethyAgentState:
        logger.info("=== Running LiteratureAgent (Agent 2) ===")
        return agent.run(state)

    return run_literature_agent


def make_report_node(config: Dict[str, Any], registry: Registry):
    """Factory: returns the generate_report node function."""

    def generate_report(state: MethyAgentState) -> MethyAgentState:
        """Generate the final summary report from registry data."""
        logger.info("=== Generating final report ===")

        summary = registry.get_summary()
        all_datasets = registry.get_all()

        # Build report
        report = {
            "query": state.get("raw_query", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "datasets": all_datasets,
            "agent1": {
                "candidates_found": len(state.get("db_candidates", [])),
                "downloaded": state.get("db_downloaded", []),
                "failed": state.get("db_failed", []),
                "skipped": state.get("db_skipped", []),
            },
            "agent2": {
                "papers_searched": len(state.get("papers_found", [])),
                "new_accessions": len(state.get("lit_candidates", [])),
                "downloaded": state.get("lit_downloaded", []),
                "failed": state.get("lit_failed", []),
                "skipped": state.get("lit_skipped", []),
            },
            "errors": state.get("error_log", []),
        }

        # Save report files
        output_dir = Path(config["download"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = output_dir / f"report_{ts}.json"
        md_path = output_dir / f"report_{ts}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        md_content = _render_markdown_report(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Report saved: {json_path}, {md_path}")

        return {**state, "final_report": report}

    return generate_report


# ------------------------------------------------------------------ #
#  Conditional edge functions                                          #
# ------------------------------------------------------------------ #

def should_run_literature_agent(
    state: MethyAgentState,
) -> Literal["run_literature_agent", "generate_report"]:
    """
    Decide whether to run LiteratureAgent after DatabaseAgent.
    Skip if agent_mode is 'db-only'.
    """
    agent_mode = state.get("config", {}).get("_agent_mode", "both")
    if agent_mode == "db-only":
        logger.info("Agent mode is db-only, skipping LiteratureAgent")
        return "generate_report"
    return "run_literature_agent"


def should_run_database_agent(
    state: MethyAgentState,
) -> Literal["run_database_agent", "run_literature_agent"]:
    """
    Decide whether to run DatabaseAgent first.
    Skip if agent_mode is 'lit-only'.
    """
    agent_mode = state.get("config", {}).get("_agent_mode", "both")
    if agent_mode == "lit-only":
        logger.info("Agent mode is lit-only, skipping DatabaseAgent")
        return "run_literature_agent"
    return "run_database_agent"


# ------------------------------------------------------------------ #
#  Graph builder                                                       #
# ------------------------------------------------------------------ #

def build_graph(config: Dict[str, Any], registry: Registry):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        config: Settings dict from settings.yaml.
        registry: Shared Registry instance.

    Returns:
        Compiled LangGraph app (callable with initial state).
    """
    from langgraph.graph import StateGraph, END

    # Create node functions
    parse_query_node = make_parse_query_node(config)
    db_agent_node = make_database_agent_node(config, registry)
    lit_agent_node = make_literature_agent_node(config, registry)
    report_node = make_report_node(config, registry)

    # Build graph
    graph = StateGraph(MethyAgentState)

    # Add nodes
    graph.add_node("parse_query", parse_query_node)
    graph.add_node("run_database_agent", db_agent_node)
    graph.add_node("run_literature_agent", lit_agent_node)
    graph.add_node("generate_report", report_node)

    # Set entry point
    graph.set_entry_point("parse_query")

    # Add conditional edge from parse_query
    graph.add_conditional_edges(
        "parse_query",
        should_run_database_agent,
        {
            "run_database_agent": "run_database_agent",
            "run_literature_agent": "run_literature_agent",
        },
    )

    # Add conditional edge from database agent
    graph.add_conditional_edges(
        "run_database_agent",
        should_run_literature_agent,
        {
            "run_literature_agent": "run_literature_agent",
            "generate_report": "generate_report",
        },
    )

    # Literature agent always goes to report
    graph.add_edge("run_literature_agent", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ------------------------------------------------------------------ #
#  Main runner                                                         #
# ------------------------------------------------------------------ #

def run_methyagent(
    query: str,
    config_path: str = "config/settings.yaml",
    agent_mode: str = "both",
) -> Dict[str, Any]:
    """
    Run the full MethyAgent pipeline for a given query.

    Args:
        query: User query string (natural language or accession number).
        config_path: Path to settings.yaml.
        agent_mode: 'both' | 'db-only' | 'lit-only'

    Returns:
        Final state dict including the report.
    """
    config = load_config(config_path)
    config["_agent_mode"] = agent_mode  # Pass mode through state

    registry = Registry(config["registry"]["db_path"])

    app = build_graph(config, registry)

    initial_state: MethyAgentState = {
        "raw_query": query,
        "parsed_intent": {},
        "db_candidates": [],
        "db_downloaded": [],
        "db_failed": [],
        "db_skipped": [],
        "papers_found": [],
        "lit_candidates": [],
        "lit_downloaded": [],
        "lit_failed": [],
        "lit_skipped": [],
        "messages": [],
        "error_log": [],
        "final_report": {},
        "config": config,
    }

    logger.info(f"Starting MethyAgent | query='{query}' | mode={agent_mode}")
    final_state = app.invoke(initial_state)
    logger.info("MethyAgent completed.")

    return final_state


# ------------------------------------------------------------------ #
#  Markdown report renderer                                            #
# ------------------------------------------------------------------ #

def _render_markdown_report(report: Dict[str, Any]) -> str:
    """Render the report dict as a Markdown document."""
    s = report.get("summary", {})
    a1 = report.get("agent1", {})
    a2 = report.get("agent2", {})
    ts = report.get("timestamp", "")[:19].replace("T", " ")

    lines = [
        "# MethyAgent 采集报告",
        "",
        f"**查询**: {report.get('query', '')}",
        f"**时间**: {ts}",
        "",
        "---",
        "",
        "## 采集摘要",
        "",
        f"| 指标 | 数量 |",
        f"|------|------|",
        f"| 总数据集数 | {s.get('total', 0)} |",
        f"| Agent 1 发现 | {s.get('agent1_discovered', 0)} |",
        f"| Agent 2 新增 | {s.get('agent2_discovered', 0)} |",
        f"| 下载成功 | {s.get('by_status', {}).get('done', 0)} |",
        f"| 下载失败 | {s.get('by_status', {}).get('failed', 0)} |",
        f"| 已跳过（去重）| {s.get('by_status', {}).get('skipped', 0)} |",
        "",
        "### 数据来源分布",
        "",
    ]

    for src, cnt in s.get("by_source", {}).items():
        lines.append(f"- **{src}**: {cnt} 个数据集")

    lines += [
        "",
        "### 数据类型分布",
        "",
    ]
    for dtype, cnt in s.get("by_data_type", {}).items():
        lines.append(f"- **{dtype}**: {cnt} 个数据集")

    lines += [
        "",
        "---",
        "",
        "## Agent 1 (DatabaseAgent) 详情",
        "",
        f"- 候选数据集: {a1.get('candidates_found', 0)}",
        f"- 成功下载: {len(a1.get('downloaded', []))}",
        f"- 下载失败: {len(a1.get('failed', []))}",
        f"- 已跳过: {len(a1.get('skipped', []))}",
        "",
        "## Agent 2 (LiteratureAgent) 详情",
        "",
        f"- 检索文献数: {a2.get('papers_searched', 0)}",
        f"- 文献中新发现 accession: {a2.get('new_accessions', 0)}",
        f"- 成功下载: {len(a2.get('downloaded', []))}",
        f"- 下载失败: {len(a2.get('failed', []))}",
        f"- 已跳过（Agent 1 已覆盖）: {len(a2.get('skipped', []))}",
        "",
        "---",
        "",
        "## 数据集列表",
        "",
        "| Accession | 来源 | 平台 | 癌种 | 样本数 | 年份 | 状态 |",
        "|-----------|------|------|------|--------|------|------|",
    ]

    for ds in report.get("datasets", []):
        lines.append(
            f"| {ds.get('accession','')} "
            f"| {ds.get('source','')} "
            f"| {ds.get('platform') or '-'} "
            f"| {ds.get('cancer_type') or '-'} "
            f"| {ds.get('sample_count') or '-'} "
            f"| {ds.get('year') or '-'} "
            f"| {ds.get('download_status','')} |"
        )

    if report.get("errors"):
        lines += [
            "",
            "---",
            "",
            "## 错误日志",
            "",
        ]
        for err in report["errors"]:
            lines.append(f"- {err}")

    return "\n".join(lines)
