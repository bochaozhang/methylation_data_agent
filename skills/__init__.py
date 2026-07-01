"""
Skill-based architecture for MethyAgent.

A *skill* bundles a capability's domain knowledge (a canonical SPEC) together
with the code that executes it. Skills are the unit the orchestrator selects.

See skills/base.py for the Skill protocol and registry.
"""
from skills.base import Skill, skill_registry, register_skill, get_skill, all_skills

__all__ = ["Skill", "skill_registry", "register_skill", "get_skill", "all_skills"]
