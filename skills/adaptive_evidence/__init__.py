"""Adaptive evidence-gathering skill package (path B).

A bounded ReAct agent that fires when geo_filter returns manual_review: it calls
real evidence-fetch tools (PMC full text / supplementary tables / GSE->PubMed
reverse lookup / more GSMs) via the production to_tool() + bind_tools path, then
re-judges. See skills/adaptive_evidence/agent.py:run_evidence_agent.
"""
from skills.adaptive_evidence import agent, skills  # noqa: F401
