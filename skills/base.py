"""
Skill base class, registry, and LangChain tool adapter.

A Skill is a self-describing capability:
  - name:        short identifier (kebab/snake)
  - description: what it does, used by the orchestrator LLM to decide when to call it
  - args_schema: a pydantic model describing the callable's arguments
  - run(ctx, **kwargs): execute and return a JSON-serialisable result

`ctx` is a lightweight context dict carrying shared resources (config, registry,
geo_client, llm, ...) so skills don't reach for globals.

The registry lets skills self-register on import; `load_all_skills()` imports
every skill subpackage so they are available to the orchestrator.

`to_tool(skill)` adapts a Skill to a LangChain StructuredTool so the dynamic
orchestrator can `bind_tools(...)` it.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, create_model


# ---------------------------------------------------------------------- #
#  Execution context                                                     #
# ---------------------------------------------------------------------- #

class SkillContext:
    """
    Shared resources handed to every skill.run() call.

    Populated by the orchestrator. Skills should treat this as read-only
    except for the explicit accumulators they own.
    """

    def __init__(self, config: Dict[str, Any], registry: Any = None,
                 geo_client: Any = None, gdc_client: Any = None,
                 lit_client: Any = None, llm: Any = None,
                 state: Optional[Dict[str, Any]] = None):
        self.config = config
        self.registry = registry
        self.geo_client = geo_client
        self.gdc_client = gdc_client
        self.lit_client = lit_client
        self.llm = llm
        # LangGraph state dict (for back-compat fields the CLI/daemon read)
        self.state: Dict[str, Any] = state or {}


# ---------------------------------------------------------------------- #
#  Skill base class                                                      #
# ---------------------------------------------------------------------- #

class Skill:
    """Base class for all skills. Subclasses set the class attributes."""

    name: str = ""
    description: str = ""
    # Pydantic model describing run() kwargs. Subclasses may override.
    args_schema: Type[BaseModel] = BaseModel

    def run(self, ctx: SkillContext, **kwargs) -> Any:
        raise NotImplementedError

    # convenience for ad-hoc direct calls (not via orchestrator)
    def __call__(self, ctx: SkillContext, **kwargs) -> Any:
        return self.run(ctx, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Skill {self.name}>"


# ---------------------------------------------------------------------- #
#  Registry                                                              #
# ---------------------------------------------------------------------- #

class _SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> Skill:
        if not skill.name:
            raise ValueError(f"Skill {skill!r} has no name")
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def all(self) -> List[Skill]:
        return list(self._skills.values())

    def names(self) -> List[str]:
        return list(self._skills.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._skills


skill_registry = _SkillRegistry()


def register_skill(skill: Skill) -> Skill:
    """Register a skill instance in the global registry."""
    return skill_registry.register(skill)


def get_skill(name: str) -> Skill:
    return skill_registry.get(name)


def all_skills() -> List[Skill]:
    return skill_registry.all()


# ---------------------------------------------------------------------- #
#  Loading                                                               #
# ---------------------------------------------------------------------- #

_SKILL_PACKAGES = [
    "skills.geo_filter",
    "skills.parse_query",
    "skills.literature",
    "skills.registry_ops",
    "skills.report",
]


def load_all_skills() -> None:
    """Import every skill subpackage so they self-register.

    Missing packages are tolerated (skills land in the tree incrementally
    across the phased rollout).
    """
    for pkg in _SKILL_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            # Skill not implemented yet in the current phase — skip.
            pass


# ---------------------------------------------------------------------- #
#  LangChain tool adapter (used by the dynamic orchestrator, Phase 3)    #
# ---------------------------------------------------------------------- #

def to_tool(skill: Skill):
    """
    Adapt a Skill into a LangChain StructuredTool bound to a SkillContext.

    Returns a factory: `to_tool(skill)(ctx)` -> StructuredTool, so each
    orchestrator session can bake in its own context.
    """
    from langchain_core.tools import StructuredTool

    def make(ctx: SkillContext) -> Any:
        def _run(**kwargs) -> Any:
            return skill.run(ctx, **kwargs)

        return StructuredTool.from_function(
            func=_run,
            name=skill.name,
            description=skill.description,
            args_schema=skill.args_schema,
        )

    return make
