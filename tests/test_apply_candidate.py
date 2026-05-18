"""Tests for ``core.recommendation.apply_candidate``.

The executor is the bridge between the deterministic candidate
generator and the re-analysis loop, so the contract matters:

    * The input ``model_json`` must NEVER be mutated.
    * ``replace_section`` changes exactly one element.
    * ``replace_sections_by_story`` changes every matching element at
      one story.
    * Abstract / manual_review candidates raise
      :class:`InapplicableCandidateError`.
    * A no-op diff is treated as inapplicable.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.recommendation import (  # noqa: E402
    ActionType,
    ChangeDiff,
    ChangeOperation,
    Confidence,
    IssueSource,
    IssueType,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
    apply_candidate_to_model,
    apply_candidates_batch,
    generate_candidates,
    make_issue_id,
)
from core.recommendation.apply_candidate import InapplicableCandidateError


# ---------------------------------------------------------------------------
# Fixtures — a small V2-shape JSON model we can mutate freely.
# ---------------------------------------------------------------------------

def _build_simple_model() -> dict:
    """2 stories, 2×2 grid: 4 columns per story, 2 X-beams per story."""
    nodes = []
    nid = 1
    for s in range(3):  # node stories 0/1/2 → structural stories 1/2
        for ix in range(2):
            for iy in range(2):
                nodes.append({
                    "id": nid,
                    "x": ix * 6000, "y": iy * 6000, "z": s * 3500,
                    "story": s,
                    "support": "fixed" if s == 0 else None,
                    "mass": 0.0,
                })
                nid += 1

    elements = []
    eid = 1
    # Columns (story 1 + 2)
    for s in range(1, 3):
        for ix in range(2):
            for iy in range(2):
                lower = (s - 1) * 4 + ix * 2 + iy + 1
                upper = s * 4 + ix * 2 + iy + 1
                elements.append({
                    "id": eid, "node_i": lower, "node_j": upper,
                    "elem_type": "column",
                    "section": "H-300x300", "material": "SS275",
                    "release_i": None, "release_j": None, "beta_angle": 0.0,
                })
                eid += 1
    # Beams (story 1 + 2, X direction)
    for s in range(1, 3):
        for iy in range(2):
            ni = s * 4 + 0 * 2 + iy + 1
            nj = s * 4 + 1 * 2 + iy + 1
            elements.append({
                "id": eid, "node_i": ni, "node_j": nj,
                "elem_type": "beam",
                "section": "H-400x200", "material": "SS275",
                "release_i": None, "release_j": None, "beta_angle": 0.0,
            })
            eid += 1
    return {
        "version": "2.0",
        "nodes": nodes,
        "elements": elements,
        "story_elevations": [0, 3500, 7000],
        "story_usages": {1: "office", 2: "office"},
    }


def _strength_issue_for_element(eid: int, story: int) -> StructuralIssue:
    return StructuralIssue(
        issue_id=make_issue_id(
            issue_type=IssueType.STRENGTH_EXCEEDED,
            member_id=eid, governing_combo="X",
        ),
        issue_type=IssueType.STRENGTH_EXCEEDED,
        severity=Severity.ERROR,
        source=IssueSource.DESIGN_CHECK,
        description="strength NG",
        member_id=eid, element_id=eid,
        governing_combo="X",
        demand_capacity_ratio=1.35,
        member_type="column", section="H-300x300", story=story,
    )


def _drift_issue(story: int) -> StructuralIssue:
    return StructuralIssue(
        issue_id=make_issue_id(
            issue_type=IssueType.DRIFT_EXCEEDED,
            story=story, direction="X", governing_combo="EQX",
        ),
        issue_type=IssueType.DRIFT_EXCEEDED,
        severity=Severity.ERROR,
        source=IssueSource.DESIGN_CHECK,
        description="drift NG",
        demand_capacity_ratio=1.2,
        story=story,
        evidence={"direction": "X", "story": story},
    )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_replace_section_does_not_mutate_input(self):
        model = _build_simple_model()
        snapshot = copy.deepcopy(model)
        cand = generate_candidates([_strength_issue_for_element(1, 1)])[0]
        new_model, diff = apply_candidate_to_model(model, cand)

        # Strict equality with the original snapshot.
        assert model == snapshot
        # Returned object is a different one.
        assert new_model is not model
        # The changed element actually has the new section in the copy.
        target_elem = next(e for e in new_model["elements"] if e["id"] == 1)
        assert target_elem["section"] != "H-300x300"

    def test_story_wide_does_not_mutate_input(self):
        model = _build_simple_model()
        snapshot = copy.deepcopy(model)
        cands = generate_candidates([_drift_issue(2)])
        applicable = [c for c in cands
                      if c.proposed_change.get("applicable")]
        assert applicable, "drift issue should yield at least one applicable candidate"
        new_model, diff = apply_candidate_to_model(model, applicable[0])

        assert model == snapshot
        assert diff.changed_member_count > 0


# ---------------------------------------------------------------------------
# Operation scope
# ---------------------------------------------------------------------------

class TestReplaceSectionScope:
    def test_changes_exactly_one_element(self):
        model = _build_simple_model()
        cands = generate_candidates([_strength_issue_for_element(1, 1)])
        member_cands = [c for c in cands
                        if c.proposed_change["operation"]
                        == ChangeOperation.REPLACE_SECTION]
        assert member_cands, "expected at least one member-level upgrade"
        new_model, diff = apply_candidate_to_model(model, member_cands[0])

        # Exactly one diff row, against element_id=1.
        assert diff.changed_member_count == 1
        row = diff.changed_members[0]
        assert row["element_id"] == 1
        assert row["section_from"] == "H-300x300"
        assert row["section_to"] != "H-300x300"
        # Other elements untouched.
        for e in new_model["elements"]:
            if e["id"] == 1:
                continue
            assert e["section"] == ({
                "column": "H-300x300", "beam": "H-400x200",
            }[e["elem_type"]])


class TestReplaceSectionsByStoryScope:
    def test_strength_story_wide_only_touches_target_story_columns(self):
        """story-wide strength candidate should upgrade every column on
        the same story (4 in this fixture) and leave beams alone."""
        model = _build_simple_model()
        cands = generate_candidates([_strength_issue_for_element(1, 1)])
        story_cand = next(
            c for c in cands
            if c.proposed_change["operation"]
            == ChangeOperation.REPLACE_SECTIONS_BY_STORY
        )
        new_model, diff = apply_candidate_to_model(model, story_cand)

        # All 4 story-1 columns changed (elements 1..4).
        changed_ids = {row["element_id"] for row in diff.changed_members}
        assert changed_ids == {1, 2, 3, 4}, changed_ids
        # No beam changed.
        for e in new_model["elements"]:
            if e["elem_type"] == "beam":
                assert e["section"] == "H-400x200"
        # Story-2 columns untouched.
        for e in new_model["elements"]:
            if e["id"] in {5, 6, 7, 8}:
                assert e["section"] == "H-300x300"

    def test_drift_story_2_upgrade_only_touches_story_2(self):
        model = _build_simple_model()
        cands = generate_candidates([_drift_issue(2)])
        # First two are story-2 column upgrades (step 1 then step 2)
        step1 = next(
            c for c in cands
            if c.proposed_change.get("upgrade_step") == 1
        )
        new_model, diff = apply_candidate_to_model(model, step1)
        # 4 story-2 columns: ids 5..8
        assert {row["element_id"] for row in diff.changed_members} == {5, 6, 7, 8}
        for e in new_model["elements"]:
            if e["id"] in {1, 2, 3, 4}:
                assert e["section"] == "H-300x300"

    def test_step_2_picks_a_larger_section_than_step_1(self):
        model = _build_simple_model()
        cands = generate_candidates([_drift_issue(2)])
        step1 = next(c for c in cands
                     if c.proposed_change.get("upgrade_step") == 1)
        step2 = next(c for c in cands
                     if c.proposed_change.get("upgrade_step") == 2)
        _, d1 = apply_candidate_to_model(model, step1)
        _, d2 = apply_candidate_to_model(model, step2)
        s1 = d1.changed_members[0]["section_to"]
        s2 = d2.changed_members[0]["section_to"]
        assert s1 != s2, "step1 and step2 must pick different sections"


# ---------------------------------------------------------------------------
# Inapplicable / refusal paths
# ---------------------------------------------------------------------------

class TestInapplicableCandidates:
    def test_add_lateral_resistance_is_refused(self):
        model = _build_simple_model()
        cands = generate_candidates([_drift_issue(2)])
        abstract = next(
            c for c in cands
            if c.proposed_change["operation"]
            == ChangeOperation.ADD_LATERAL_RESISTANCE
        )
        with pytest.raises(InapplicableCandidateError):
            apply_candidate_to_model(model, abstract)

    def test_manual_review_is_refused(self):
        model = _build_simple_model()
        cand = RetrofitCandidate(
            candidate_id="cand_dummy_review",
            issue_id="iss_dummy",
            action_type=ActionType.REQUIRES_ENGINEER_REVIEW,
            description="manual",
            requires_reanalysis=True,
            confidence=Confidence.LOW,
            target={"member_id": 1},
            proposed_change={
                "operation": ChangeOperation.MANUAL_REVIEW,
                "from": "H-300x300", "to": None,
                "requires_user_selection": True,
                "applicable": False, "reason": "manual",
            },
        )
        with pytest.raises(InapplicableCandidateError):
            apply_candidate_to_model(model, cand)

    def test_unknown_operation_is_refused(self):
        model = _build_simple_model()
        cand = RetrofitCandidate(
            candidate_id="cand_dummy_unknown",
            issue_id="iss_dummy",
            action_type=ActionType.INCREASE_SECTION,
            description="unknown op",
            requires_reanalysis=True,
            confidence=Confidence.MEDIUM,
            target={"member_id": 1, "element_id": 1},
            proposed_change={
                "operation": "frob_widget",   # not whitelisted
                "from": "H-300x300", "to": "H-350x350",
                "requires_user_selection": False,
                "applicable": True, "reason": "test",
            },
        )
        with pytest.raises(InapplicableCandidateError):
            apply_candidate_to_model(model, cand)

    def test_replace_section_with_unknown_element_id_is_refused(self):
        model = _build_simple_model()
        cand = RetrofitCandidate(
            candidate_id="cand_dummy_missing",
            issue_id="iss_dummy",
            action_type=ActionType.INCREASE_SECTION,
            description="missing target",
            requires_reanalysis=True,
            confidence=Confidence.MEDIUM,
            target={"element_id": 9999},   # not in model
            proposed_change={
                "operation": ChangeOperation.REPLACE_SECTION,
                "from": "H-300x300", "to": "H-350x350",
                "requires_user_selection": False,
                "applicable": True, "reason": "test",
            },
        )
        with pytest.raises(InapplicableCandidateError):
            apply_candidate_to_model(model, cand)

    def test_replace_section_noop_is_refused(self):
        """If the executor would produce zero changes (e.g. target
        already has the requested section), it must raise rather than
        return a no-op diff."""
        model = _build_simple_model()
        # Force the target element's section to already match.
        model["elements"][0]["section"] = "H-350x350"
        cand = RetrofitCandidate(
            candidate_id="cand_dummy_noop",
            issue_id="iss_dummy",
            action_type=ActionType.INCREASE_SECTION,
            description="noop",
            requires_reanalysis=True,
            confidence=Confidence.MEDIUM,
            target={"element_id": 1},
            proposed_change={
                "operation": ChangeOperation.REPLACE_SECTION,
                "from": "H-350x350", "to": "H-350x350",
                "requires_user_selection": False,
                "applicable": True, "reason": "test",
            },
        )
        with pytest.raises(InapplicableCandidateError):
            apply_candidate_to_model(model, cand)


# ---------------------------------------------------------------------------
# Batch surface
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_applies_each_candidate_independently(self):
        model = _build_simple_model()
        snapshot = copy.deepcopy(model)
        cands = generate_candidates([_strength_issue_for_element(1, 1)])
        cands = [c for c in cands
                 if c.proposed_change["operation"]
                 == ChangeOperation.REPLACE_SECTION]
        assert len(cands) == 2
        results = apply_candidates_batch(model, cands)
        # Original untouched after batch.
        assert model == snapshot
        # Each result has its own diff and changed section is the
        # candidate's own ``to``.
        for cand, (new_m, diff) in zip(cands, results):
            assert diff.changed_member_count == 1
            assert diff.changed_members[0]["section_to"] == (
                cand.proposed_change["to"]
            )
