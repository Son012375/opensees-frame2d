"""Executor that applies a :class:`RetrofitCandidate` to a V2 model JSON.

Phase 1 supports two operations:

    * ``replace_section``           — change a single element's section
    * ``replace_sections_by_story`` — bump every element of a given
                                       member_type at one story by
                                       ``upgrade_step`` rungs of its
                                       family ladder

Everything else (``add_lateral_resistance``, ``manual_review``,
``change_material``, ...) raises :class:`InapplicableCandidateError`
so the evaluator can mark such candidates as ``skipped_not_applicable``.

The executor is pure: it never mutates the input ``model_json``. The
caller receives a deep-copied dict with the change applied plus a
:class:`ChangeDiff` describing exactly which elements moved and to what
section.

Element ↔ candidate matching
----------------------------
A V2 ``StructuralModel.to_json()`` payload exposes elements as

    {"id": int, "node_i": ..., "node_j": ...,
     "elem_type": "column"|"beam"|"brace",
     "section": "H-300x300", ...}

When :mod:`frame_3d` later builds the OpenSees model it re-indexes
elements as 1-based ``member_id`` by sorting elements by their ``id``
field (see ``analyze_from_model``). For candidates that carry a
``target.element_id``, we match directly on ``elements[].id``. For
those that only carry a ``target.member_id`` we fall back to the same
sort-and-enumerate scheme so the result is consistent with the
analyzer's perspective.

Story derivation
----------------
For ``replace_sections_by_story`` we need to know which elements live
at which story. We use this rule, mirroring how the analyzer/design
check interpret "story N":

    * column  → element belongs to ``story(node_j) if node_j.story > node_i.story else story(node_i)``
                i.e. the upper node's story (the column "supports floor N")
    * beam    → element belongs to ``node_i.story`` (= ``node_j.story``)
    * brace   → element belongs to ``max(node_i.story, node_j.story)``
                (treated like a column for story attribution)

Stories outside ``[1, num_stories]`` are silently skipped — they are
ground/foundation nodes and not user-addressable as a "story" target.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import ChangeOperation, RetrofitCandidate
from .section_catalog import next_sections


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InapplicableCandidateError(Exception):
    """Raised when a candidate's operation is not auto-applicable.

    Carries the candidate id and a short reason string; the evaluator
    surfaces these as ``skipped_not_applicable`` entries in the
    recommendation summary.
    """

    def __init__(self, candidate_id: str, reason: str) -> None:
        super().__init__(f"candidate {candidate_id} inapplicable: {reason}")
        self.candidate_id = candidate_id
        self.reason = reason


# ---------------------------------------------------------------------------
# Diff record
# ---------------------------------------------------------------------------

@dataclass
class ChangeDiff:
    """What a single candidate actually changed in the model."""
    candidate_id: str
    operation: str
    reason: str = ""
    changed_members: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed_member_count(self) -> int:
        return len(self.changed_members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "operation": self.operation,
            "reason": self.reason,
            "changed_member_count": self.changed_member_count,
            "changed_members": list(self.changed_members),
        }


# ---------------------------------------------------------------------------
# Element lookup helpers
# ---------------------------------------------------------------------------

def _find_element_by_id(elements: list[dict], element_id: int) -> Optional[dict]:
    for e in elements:
        if e.get("id") == element_id:
            return e
    return None


def _find_element_by_member_id(elements: list[dict],
                               member_id: int) -> Optional[dict]:
    """Return the element that ``analyze_from_model`` would index as ``member_id``.

    ``member_id`` is the 1-based position after sorting by ``id``.
    """
    if member_id is None or member_id < 1:
        return None
    sorted_elems = sorted(
        elements, key=lambda e: (e.get("id") is None, e.get("id", 0))
    )
    idx = member_id - 1
    if 0 <= idx < len(sorted_elems):
        return sorted_elems[idx]
    return None


def _resolve_target_element(
    elements: list[dict],
    target: dict,
    candidate_id: str,
) -> dict:
    """Locate the element a member-scope candidate refers to. Raises if not found."""
    element_id = target.get("element_id")
    member_id = target.get("member_id")

    if element_id is not None:
        elem = _find_element_by_id(elements, int(element_id))
        if elem is not None:
            return elem
    if member_id is not None:
        elem = _find_element_by_member_id(elements, int(member_id))
        if elem is not None:
            return elem

    raise InapplicableCandidateError(
        candidate_id,
        f"target element not found (element_id={element_id}, "
        f"member_id={member_id})",
    )


def _node_story(nodes_by_id: dict[int, dict], node_id: int) -> Optional[int]:
    n = nodes_by_id.get(node_id)
    if n is None:
        return None
    s = n.get("story")
    if s is None:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _element_story(
    elem: dict,
    nodes_by_id: dict[int, dict],
) -> Optional[int]:
    """Story attribution for a V2 element. See module docstring."""
    si = _node_story(nodes_by_id, elem.get("node_i"))
    sj = _node_story(nodes_by_id, elem.get("node_j"))
    etype = elem.get("elem_type", "")

    if etype == "beam":
        if si is None and sj is None:
            return None
        if si is None:
            return sj
        if sj is None:
            return si
        return si if si == sj else max(si, sj)

    # column / brace: upper node's story
    candidates = [s for s in (si, sj) if s is not None]
    if not candidates:
        return None
    return max(candidates)


def _matches_member_type(elem: dict, member_type: str) -> bool:
    """Best-effort match between candidate ``target.member_type`` and the
    V2 element's ``elem_type``.

    V2 elem_type is one of {"column", "beam", "brace"}. The candidate
    target may say "column" / "beam" / "beam_x" / "beam_y" (the latter
    two come from frame_3d which splits beams by direction). We treat
    any "beam*" target as matching V2 ``beam`` elements; "column"
    matches columns; nothing else matches.
    """
    if not member_type:
        return False
    etype = elem.get("elem_type", "")
    mt = member_type.lower()
    if mt == "column":
        return etype == "column"
    if mt.startswith("beam"):
        return etype == "beam"
    return mt == etype


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_candidate_to_model(
    model_json: dict,
    candidate: RetrofitCandidate,
) -> tuple[dict, ChangeDiff]:
    """Return ``(new_model_json, diff)`` with ``candidate`` applied.

    The input is never mutated. Raises :class:`InapplicableCandidateError`
    if the operation is unsupported, the target can't be found, or the
    section catalog has no further rungs to suggest.
    """
    if not isinstance(model_json, dict):
        raise TypeError(
            f"apply_candidate_to_model expects a dict, got "
            f"{type(model_json).__name__}",
        )

    op = candidate.proposed_change.get("operation") if candidate.proposed_change else None
    if not op:
        raise InapplicableCandidateError(
            candidate.candidate_id, "no operation in proposed_change",
        )

    applicable = bool(candidate.proposed_change.get("applicable", True))
    if not applicable:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            f"candidate is abstract (applicable=False, op={op})",
        )

    new_model = copy.deepcopy(model_json)
    elements = new_model.setdefault("elements", [])
    nodes_by_id = {n["id"]: n for n in new_model.get("nodes", [])
                   if isinstance(n, dict) and "id" in n}
    # Build the element_id → member_id index ONCE per apply so
    # replace_sections_by_story stays O(N), not O(N²).
    member_id_by_element_id = _build_member_id_index(elements)

    reason = candidate.proposed_change.get("reason", "")
    diff = ChangeDiff(
        candidate_id=candidate.candidate_id,
        operation=op,
        reason=reason,
    )

    if op == ChangeOperation.REPLACE_SECTION:
        _apply_replace_section(
            candidate, elements, nodes_by_id, diff,
            member_id_by_element_id,
        )
    elif op == ChangeOperation.REPLACE_SECTIONS_BY_STORY:
        _apply_replace_sections_by_story(
            candidate, elements, nodes_by_id, diff,
            member_id_by_element_id,
        )
    else:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            f"operation '{op}' is not in the Phase 1 whitelist "
            "(replace_section, replace_sections_by_story)",
        )

    if not diff.changed_members:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            f"operation '{op}' resolved to zero changes — refusing to "
            "produce a no-op diff",
        )

    return new_model, diff


def apply_candidates_batch(
    model_json: dict,
    candidates: list[RetrofitCandidate],
) -> list[tuple[dict, ChangeDiff]]:
    """Apply each candidate independently against the *original* model.

    Each candidate gets its own fresh deep copy. **The first
    ``InapplicableCandidateError`` aborts the batch and is re-raised**
    — successes accumulated up to that point are NOT returned. This
    matches Phase 1's actual usage: the evaluator calls
    :func:`apply_candidate_to_model` one-by-one and wraps its own
    try/except, so resilience lives there, not here. If a future
    caller needs partial-success / per-item error reporting (e.g. a
    Phase 2 diff-preview batch), don't change this function — add a
    sibling like ``try_apply_candidates`` that catches and returns
    ``list[Result[diff, error]]``.
    """
    out: list[tuple[dict, ChangeDiff]] = []
    for cand in candidates:
        out.append(apply_candidate_to_model(model_json, cand))
    return out


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------

def _build_member_id_index(elements: list[dict]) -> dict[int, int]:
    """Map ``element_id → member_id`` using ``analyze_from_model``'s
    sort-position rule (1-based enumerate over elements sorted by
    ``id``). Computing this once per apply() avoids O(N²) walks when
    a story-wide candidate hits many elements.
    """
    sorted_elems = sorted(
        elements, key=lambda e: (e.get("id") is None, e.get("id", 0))
    )
    return {
        e.get("id"): idx
        for idx, e in enumerate(sorted_elems, start=1)
        if e.get("id") is not None
    }


def _member_label(elem: dict, story: Optional[int]) -> str:
    """UI-friendly stable label for diff display: e.g.
    ``"column @ story 2 (element 7)"``. Not for programmatic use —
    Phase 2 UI shows this as the human-readable hook."""
    etype = elem.get("elem_type") or "member"
    eid = elem.get("id")
    if story is None:
        return f"{etype} (element {eid})"
    return f"{etype} @ story {story} (element {eid})"


def _record_change(
    diff: ChangeDiff,
    elem: dict,
    new_section: str,
    member_type: str,
    story: Optional[int],
    reason: str,
    member_id: Optional[int] = None,
) -> None:
    diff.changed_members.append({
        "element_id": elem.get("id"),
        # 1-based sort-position member_id matches the index
        # ``analyze_from_model`` assigns, so Phase 2 UI can refer to
        # "member N" the same way the analyzer/design-check do.
        "member_id": member_id,
        "member_label": _member_label(elem, story),
        "member_type": elem.get("elem_type"),
        "section_from": elem.get("section"),
        "section_to": new_section,
        "story": story,
        "reason": reason,
    })
    elem["section"] = new_section
    _ = member_type  # currently unused; reserved for richer diffs


def _apply_replace_section(
    candidate: RetrofitCandidate,
    elements: list[dict],
    nodes_by_id: dict[int, dict],
    diff: ChangeDiff,
    member_id_by_element_id: dict[int, int],
) -> None:
    target = candidate.target or {}
    elem = _resolve_target_element(elements, target, candidate.candidate_id)

    to_section = candidate.proposed_change.get("to")
    if not to_section:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            "replace_section requires proposed_change.to",
        )
    if elem.get("section") == to_section:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            f"target already has section '{to_section}'",
        )

    story = _element_story(elem, nodes_by_id)
    _record_change(
        diff, elem,
        new_section=str(to_section),
        member_type=target.get("member_type", elem.get("elem_type", "")),
        story=story,
        reason=candidate.proposed_change.get("reason", "replace_section"),
        member_id=member_id_by_element_id.get(elem.get("id")),
    )


def _apply_replace_sections_by_story(
    candidate: RetrofitCandidate,
    elements: list[dict],
    nodes_by_id: dict[int, dict],
    diff: ChangeDiff,
    member_id_by_element_id: dict[int, int],
) -> None:
    target = candidate.target or {}
    story = target.get("story")
    member_type = target.get("member_type")
    upgrade_step = int(
        candidate.proposed_change.get("upgrade_step", 1)
    )

    if story is None:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            "replace_sections_by_story requires target.story",
        )
    if not member_type:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            "replace_sections_by_story requires target.member_type",
        )
    if upgrade_step <= 0:
        raise InapplicableCandidateError(
            candidate.candidate_id,
            f"upgrade_step must be ≥ 1 (got {upgrade_step})",
        )

    story_int = int(story)
    reason = candidate.proposed_change.get("reason", "replace_sections_by_story")

    # Walk every element at this story that matches the member_type. For
    # each, walk its own family ladder up by ``upgrade_step`` rungs.
    # If a particular element has no ladder progression (top of catalog
    # or unknown family) we simply skip it; the diff still records the
    # other successful changes.
    for elem in elements:
        if not _matches_member_type(elem, member_type):
            continue
        es = _element_story(elem, nodes_by_id)
        if es != story_int:
            continue

        current = elem.get("section")
        if not current:
            continue
        # member-type used for ladder filter: V2 stores plain "beam" but
        # section_catalog wants "beam" (any) — we pass elem_type directly.
        ladder_member_type = (
            "column" if elem.get("elem_type") == "column" else "beam"
        )
        steps = next_sections(current, ladder_member_type, steps=upgrade_step)
        if len(steps) < upgrade_step:
            # Not enough rungs left — skip this element.
            continue
        target_entry = steps[upgrade_step - 1]
        if target_entry.name == current:
            continue

        _record_change(
            diff, elem,
            new_section=target_entry.name,
            member_type=ladder_member_type,
            story=story_int,
            reason=reason,
            member_id=member_id_by_element_id.get(elem.get("id")),
        )
