"""Rule registry for retrofit-candidate generators.

Replaces the in-line ``if/elif`` chain in
:mod:`core.recommendation.candidate_generator` with a small dispatch
table keyed by :class:`IssueType`. The handlers themselves still live
in ``candidate_generator`` — this module only fixes how they are
selected. Goals:

    * Keep candidate generation deterministic (no random / no LLM).
    * Make it trivial to plug in a new ``issue_type → handler`` rule
      without touching the dispatch code.
    * Surface the registry contents for tests / debugging.

A handler is just::

    Callable[[StructuralIssue], Optional[RetrofitCandidate]]

Returning ``None`` is a signal "I cannot produce a typed candidate for
this issue — fall back to engineer-review". The default registry is
populated lazily by :mod:`core.recommendation.candidate_generator` to
avoid a circular import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

from .schemas import RetrofitCandidate, StructuralIssue


# Handler signature: (issue) -> one of
#   * None                             — handler opts out for this issue
#   * RetrofitCandidate                — single-candidate handler (legacy)
#   * Iterable[RetrofitCandidate]      — multi-candidate handler
#
# ``generate_candidates`` is responsible for normalizing the return value
# into a list. Multi-candidate handlers let one issue produce competing
# alternatives that the scoring layer then ranks against each other.
RuleHandlerResult = Union[
    None,
    RetrofitCandidate,
    Iterable[RetrofitCandidate],
]
RuleHandler = Callable[[StructuralIssue], RuleHandlerResult]


def normalize_handler_result(result: RuleHandlerResult) -> list[RetrofitCandidate]:
    """Coerce a handler return value into a list.

    Used by the dispatcher so individual handlers can stay terse:
    return a single candidate, a list, a generator, or None.
    """
    if result is None:
        return []
    if isinstance(result, RetrofitCandidate):
        return [result]
    # Defensive: tolerate any iterable but skip non-RetrofitCandidate items.
    out: list[RetrofitCandidate] = []
    for item in result:
        if isinstance(item, RetrofitCandidate):
            out.append(item)
    return out


@dataclass
class RuleEntry:
    """A single rule entry — handler + short label for introspection."""
    issue_type: str
    handler: RuleHandler
    label: str = ""        # human-readable, used in summaries / debug
    priority: int = 0       # tie-breaker if multiple handlers ever match

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"RuleEntry(issue_type={self.issue_type!r}, label={self.label!r})"


class RuleRegistry:
    """Single-handler-per-issue-type dispatch table.

    Last registration wins (so callers/tests can override). The reason
    we don't allow multiple handlers per issue type today is that
    candidate generation must be deterministic per issue — multiple
    overlapping handlers would force a ranking step here, which we
    already do in :mod:`core.recommendation.scoring`.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RuleEntry] = {}

    def register(
        self,
        issue_type: str,
        handler: RuleHandler,
        *,
        label: str = "",
        priority: int = 0,
    ) -> None:
        self._entries[issue_type] = RuleEntry(
            issue_type=issue_type,
            handler=handler,
            label=label or issue_type,
            priority=priority,
        )

    def get(self, issue_type: str) -> Optional[RuleEntry]:
        return self._entries.get(issue_type)

    def has(self, issue_type: str) -> bool:
        return issue_type in self._entries

    def unregister(self, issue_type: str) -> None:
        self._entries.pop(issue_type, None)

    def clear(self) -> None:
        self._entries.clear()

    def items(self) -> list[RuleEntry]:
        return list(self._entries.values())

    def labels(self) -> dict[str, str]:
        """Map issue_type → label, useful for summaries / docs / tests."""
        return {k: v.label for k, v in self._entries.items()}


# Module-level singleton. ``candidate_generator`` registers the default
# strength / drift handlers into this instance at import time.
_DEFAULT_REGISTRY = RuleRegistry()


def default_registry() -> RuleRegistry:
    """Return the module-level default registry singleton."""
    return _DEFAULT_REGISTRY


def register_rule(
    issue_type: str,
    handler: RuleHandler,
    *,
    label: str = "",
    priority: int = 0,
) -> None:
    """Convenience: register on the default registry."""
    _DEFAULT_REGISTRY.register(
        issue_type, handler, label=label, priority=priority,
    )


def get_rule_for_issue(issue: StructuralIssue) -> Optional[RuleEntry]:
    """Return the rule for ``issue.issue_type`` on the default registry."""
    return _DEFAULT_REGISTRY.get(issue.issue_type)


def list_registered_handlers() -> dict[str, str]:
    """Return ``{issue_type: label}`` map for debug / tests."""
    return _DEFAULT_REGISTRY.labels()
