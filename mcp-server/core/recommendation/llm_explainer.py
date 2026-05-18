"""Optional LLM provider interface for retrofit recommendation explanations.

The Phase 3 design intentionally separates the deterministic explainer
from any actual LLM call. This module defines the *interface* an LLM
provider must satisfy if it wants to refine the explanation text, and
ships a Noop default that simply refuses to run.

A real provider (Anthropic Claude, Voyage rerank-as-prompt, local model)
would subclass :class:`BaseExplanationLLMProvider`, implement
:meth:`generate`, and be wired into the API layer or the
``explain_candidate`` callsite. The provider receives the deterministic
explanation context plus the already-retrieved (and validated) KDS
evidence — it MUST NOT perform its own retrieval, MUST NOT cite clauses
that aren't in the supplied evidence list, and MUST NOT mutate model
state.

Until a provider is wired, the explanation endpoint always falls back to
:func:`core.recommendation.explainer.deterministic_explanation`.
"""
from __future__ import annotations

from typing import Any


class BaseExplanationLLMProvider:
    """Abstract LLM provider for explanation refinement.

    Implementations must return a dict whose keys are a subset of the
    explanation section keys (summary, issue_interpretation,
    recommended_change, expected_structural_effect, verified_result,
    tradeoffs, limitations, next_user_decision). Sections not provided
    fall back to the deterministic text.
    """

    def generate(
        self,
        *,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
        language: str,
        style: str,
    ) -> dict[str, str]:
        raise NotImplementedError(
            "BaseExplanationLLMProvider.generate must be implemented by "
            "a concrete subclass"
        )


class NoopExplanationLLMProvider(BaseExplanationLLMProvider):
    """The default 'no LLM wired' provider — always raises.

    The orchestrator in :mod:`core.recommendation.explainer` catches the
    raised RuntimeError and falls back to the deterministic explanation.
    The Noop is therefore not an error condition — it is the explicit
    statement that no LLM is configured for this deployment.
    """

    def generate(
        self,
        *,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
        language: str,
        style: str,
    ) -> dict[str, str]:
        raise RuntimeError(
            "LLM explanation provider not configured; "
            "deterministic fallback will be used"
        )


__all__ = [
    "BaseExplanationLLMProvider",
    "NoopExplanationLLMProvider",
]
