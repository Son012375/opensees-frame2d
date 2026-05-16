"""Data contracts for the recommendation pipeline.

These dataclasses are the *stable* shape that the recommendation engine
(deterministic now, RAG/LLM-enriched later) reads and produces.

Design notes
------------
* Pure ``@dataclass`` (no Pydantic) so the core layer keeps zero web
  dependencies. The FastAPI layer simply ``asdict()``s these.
* All fields are additive vs. the existing ``/api/v2/analyze`` payload.
  Existing string-based ``warnings: list[str]`` is preserved as
  ``warning_messages`` for frontend back-compat.
* String enum values (``Severity``, ``IssueType``, ...) are used instead
  of Python ``Enum`` so JSON serialization stays trivial and the values
  remain greppable in logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# String enums (kept as bare strings — see module docstring)
# ---------------------------------------------------------------------------

class Severity:
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueSource:
    ANALYSIS = "analysis"
    DESIGN_CHECK = "design_check"
    WARNING = "warning"


class IssueType:
    STRENGTH_EXCEEDED = "strength_exceeded"
    DRIFT_EXCEEDED = "drift_exceeded"
    MISSING_DESIGN_CHECK = "missing_design_check"
    ANALYSIS_WARNING = "analysis_warning"
    SHEAR_EXCEEDED = "shear_exceeded"


class ActionType:
    INCREASE_SECTION = "increase_section"
    CHANGE_MATERIAL = "change_material"
    ADD_MEMBER = "add_member"
    ADD_LATERAL_RESISTANCE = "add_lateral_resistance"
    CHANGE_SUPPORT = "change_support"
    REQUIRES_ENGINEER_REVIEW = "requires_engineer_review"


class Confidence:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Analysis-side contracts
# ---------------------------------------------------------------------------

@dataclass
class AnalysisEnvelope:
    """Worst-case quantities across all load cases / combos."""
    max_dx_mm: float = 0.0
    max_dy_mm: float = 0.0
    max_dz_mm: float = 0.0
    max_drift_x: float = 0.0
    max_drift_y: float = 0.0
    max_moment_kNm: float = 0.0
    max_axial_kN: float = 0.0
    max_shear_kN: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisCaseSummary:
    """Per-case (or per-combo) summary of the analysis result."""
    case_name: str
    is_combination: bool = False
    max_dx_mm: float = 0.0
    max_dy_mm: float = 0.0
    max_dz_mm: float = 0.0
    max_drift_x: float = 0.0
    max_drift_y: float = 0.0
    max_moment_kNm: float = 0.0
    max_axial_kN: float = 0.0
    max_shear_kN: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemberForceSummary:
    """Envelope member forces for a single structural member."""
    member_id: int
    element_id: Optional[int] = None
    member_type: str = ""            # column / beam_x / beam_y / ...
    section: str = ""
    governing_combo: str = ""
    Pu_kN: float = 0.0
    Mux_kNm: float = 0.0
    Muy_kNm: float = 0.0
    Vu_kN: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemberDesignCheck:
    """Single-member design-check result (KDS / AISC)."""
    member_id: int
    element_id: Optional[int] = None
    member_type: str = ""
    section: str = ""
    material: str = ""
    story: Optional[int] = None
    connected_node_ids: list[int] = field(default_factory=list)
    governing_combo: str = ""
    demand_capacity_ratio: float = 0.0   # max(interaction, shear)
    interaction_ratio: float = 0.0
    shear_ratio: float = 0.0
    status: str = "OK"                   # OK / NG / UNKNOWN
    code_refs: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)  # original design_check dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# AnalysisWarning — structured replacement for `warnings: list[str]`
# ---------------------------------------------------------------------------

@dataclass
class AnalysisWarning:
    """Structured warning emitted from the analysis pipeline.

    Replaces the legacy ``warnings: list[str]`` shape with something the
    LLM / RAG layer can actually reason about (severity, stage, recoverable
    flag). The plain-string view is exposed alongside via
    ``warning_messages`` for frontend back-compat.
    """
    code: str
    message: str
    severity: str = Severity.WARNING        # info | warning | error
    stage: str = ""                          # e.g. "design_check", "report", "rsa"
    recoverable: bool = True
    detail: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("detail") is None:
            d.pop("detail")
        return d

    @classmethod
    def from_legacy_string(cls, msg: str, stage: str = "") -> "AnalysisWarning":
        """Best-effort parse of ``"<code>: <message>"`` legacy strings."""
        if ":" in msg:
            code, _, body = msg.partition(":")
            code = code.strip() or "legacy_warning"
            body = body.strip() or msg
        else:
            code = "legacy_warning"
            body = msg
        return cls(
            code=code,
            message=body,
            severity=Severity.WARNING,
            stage=stage,
            recoverable=True,
        )


# ---------------------------------------------------------------------------
# KDS code-reference placeholder (RAG will fill this in later)
# ---------------------------------------------------------------------------

@dataclass
class CodeReference:
    """Pointer to a clause in a design code.

    Populated by the future KDS-RAG layer. The deterministic layer creates
    *empty* or hand-coded references only; ``quote`` / ``relevance_reason``
    stay ``None`` until RAG runs.
    """
    standard_id: str                         # e.g. "KDS 41 31 00"
    version: Optional[str] = None
    clause_id: Optional[str] = None          # e.g. "§8.2.3" / "H1"
    title: Optional[str] = None
    source_url: Optional[str] = None
    quote: Optional[str] = None              # TODO(rag): filled by RAG
    relevance_reason: Optional[str] = None   # TODO(rag): filled by RAG

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# Issues (deterministic extraction layer)
# ---------------------------------------------------------------------------

@dataclass
class StructuralIssue:
    """A single design issue that the recommendation engine should react to."""
    issue_id: str
    issue_type: str                          # see IssueType.*
    severity: str                            # see Severity.*
    source: str                              # see IssueSource.*
    description: str
    member_id: Optional[int] = None
    element_id: Optional[int] = None
    governing_combo: Optional[str] = None
    demand_capacity_ratio: Optional[float] = None
    status: Optional[str] = None             # OK / NG / UNKNOWN
    evidence: dict[str, Any] = field(default_factory=dict)
    code_refs: list[CodeReference] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["code_refs"] = [c.to_dict() if hasattr(c, "to_dict") else c
                          for c in self.code_refs]
        return d


@dataclass
class IssueExtractionResult:
    issues: list[StructuralIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Retrofit candidates (placeholder — real optimizer / RAG comes later)
# ---------------------------------------------------------------------------

@dataclass
class RetrofitCandidate:
    """A *proposed* modification to a member.

    IMPORTANT semantics (intentionally encoded in the data model):
        * ``requires_reanalysis`` defaults to True — candidates are NOT
          a final design decision.
        * ``confidence`` is low/medium/high; the deterministic layer
          emits low/medium only. High is reserved for RAG-validated picks.
        * ``code_refs`` is empty for now; RAG will fill it.
    """
    candidate_id: str
    issue_id: str
    action_type: str                         # see ActionType.*
    description: str
    member_id: Optional[int] = None
    element_id: Optional[int] = None
    expected_effect: str = ""
    tradeoffs: str = ""
    requires_reanalysis: bool = True
    confidence: str = Confidence.LOW
    code_refs: list[CodeReference] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["code_refs"] = [c.to_dict() if hasattr(c, "to_dict") else c
                          for c in self.code_refs]
        return d
