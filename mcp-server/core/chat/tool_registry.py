"""Phase A.2 tool registry — env-driven whitelist of chat tools.

``CHAT_TOOLS_ENABLED`` is a comma-separated list of tool group names
that mirrors the Phase column of ``docs/chat_tool_matrix.md``. Default
``"inspect,summary"`` matches the Phase A.2 row set.

A tool is a :class:`ToolSpec`:

    - ``name``: stable string the LLM uses in ``tool_call.name``
    - ``group``: phase tag (``inspect`` / ``summary`` / ``recs`` / ``edit`` / ``kds``)
    - ``description``: passed to the LLM as the function description
    - ``parameters``: JSON Schema, also passed to the LLM
    - ``func``: ``(arguments: dict, *, session: dict) -> dict``

Disabled tools are hidden from :meth:`ToolRegistry.llm_schemas` so the
LLM never sees them, and :meth:`ToolRegistry.call_tool` raises before
executing one whose group is not enabled — defence in depth against an
LLM hallucinating a disabled tool name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class ToolSpec:
    """A registered chat tool.

    ``creativity_hint`` tells the orchestrator how much sampling
    variance the LLM needs when narrating *this* tool's result:

      * ``"factual"`` (default) — answer is a faithful translation of
        ``tool_result`` fields into Korean. Want deterministic output so
        the model can't drift away from numbers/IDs it just read.
        Examples: inspect_selection, get_analysis_summary, edit tools
        that echo a confirmation.
      * ``"narrative"`` — answer is a free-form prose summary built
        from the result (e.g. a future ``generate_report`` that drafts
        a few paragraphs of design-review text). Varied phrasing
        improves readability; some sampling temperature is desirable.

    The orchestrator maps these hints to per-call ``temperature`` values
    on the Ollama provider (see :data:`_TEMPERATURE_BY_HINT`). A future
    ``"creative"`` tier (for marketing-style copy etc.) can be added
    without breaking the existing call sites.
    """
    name: str
    group: str
    description: str
    parameters: dict
    func: Callable[..., dict]
    creativity_hint: str = "factual"


class ToolNotFoundError(KeyError):
    """No tool by that name is registered."""


class ToolDisabledError(PermissionError):
    """The tool exists but its group is not in ``CHAT_TOOLS_ENABLED``."""


def parse_enabled_groups(raw: Optional[str] = None) -> frozenset[str]:
    raw = raw if raw is not None else os.environ.get(
        "CHAT_TOOLS_ENABLED", "inspect,summary,edit,kds",
    )
    return frozenset(g.strip() for g in raw.split(",") if g.strip())


class ToolRegistry:
    """Holds a fixed catalog of tools plus an enabled-group filter."""

    def __init__(
        self,
        tools: Iterable[ToolSpec],
        *,
        enabled_groups: Optional[frozenset[str]] = None,
    ):
        self._by_name: dict[str, ToolSpec] = {}
        for t in tools:
            if t.name in self._by_name:
                raise ValueError(f"duplicate tool name in registry: {t.name!r}")
            self._by_name[t.name] = t
        self._enabled = (
            enabled_groups if enabled_groups is not None else parse_enabled_groups()
        )

    @property
    def enabled_groups(self) -> frozenset[str]:
        return self._enabled

    def enabled_tools(self) -> list[ToolSpec]:
        return [t for t in self._by_name.values() if t.group in self._enabled]

    def llm_schemas(self) -> list[dict]:
        """OpenAI/Ollama-compatible function schemas. Only enabled tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.enabled_tools()
        ]

    def call_tool(self, name: str, arguments: dict, *, session: dict) -> dict:
        spec = self._by_name.get(name)
        if spec is None:
            raise ToolNotFoundError(f"unknown tool: {name!r}")
        if spec.group not in self._enabled:
            raise ToolDisabledError(
                f"tool {name!r} is in group {spec.group!r}, "
                f"not in CHAT_TOOLS_ENABLED ({sorted(self._enabled)})"
            )
        if not isinstance(arguments, dict):
            raise TypeError(
                f"tool {name!r} arguments must be a dict, got {type(arguments).__name__}"
            )
        return spec.func(arguments, session=session)


def default_registry() -> ToolRegistry:
    """Build the default registry.

    Phase A.2 inspect_selection (group: inspect) + get_analysis_summary
    (group: summary). Phase B adds propose_section_change (group: edit).
    Phase D will add KDS narrative tools (group: kds). Each phase
    appends; the env flag controls which are visible.
    """
    # Lazy import so this module doesn't pull the tools — and their
    # webapp-side cache imports — at module-load time.
    from .tools.inspect import INSPECT_SELECTION_TOOL, GET_ANALYSIS_SUMMARY_TOOL
    from .tools.kds_compliance import EXPLAIN_MEMBER_COMPLIANCE_TOOL
    from .tools.section_change import PROPOSE_SECTION_CHANGE_TOOL

    return ToolRegistry([
        INSPECT_SELECTION_TOOL,
        GET_ANALYSIS_SUMMARY_TOOL,
        PROPOSE_SECTION_CHANGE_TOOL,
        EXPLAIN_MEMBER_COMPLIANCE_TOOL,
    ])
