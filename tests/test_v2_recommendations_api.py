"""Smoke tests for ``/api/v2/recommendations/evaluate``.

These tests exercise the API layer without actually running OpenSees:
we pre-seed ``analysis_context_cache`` with a fake baseline and patch
``evaluate_candidate`` to a deterministic stub. End-to-end OpenSees
verification is done manually with a running uvicorn server (see
plan §verification).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

from app.main_simple import (  # noqa: E402
    analysis_context_cache,
    app,
    eval_jobs_db,
    _ANALYSIS_CONTEXT_LOCK,
    _EVAL_JOBS_LOCK,
)
from core.recommendation import (  # noqa: E402
    CandidateEvaluation,
    STATUS_EVALUATED,
    STATUS_REJECTED_NEW_NG,
    STATUS_SKIPPED_INAPPLICABLE,
    ScoreBreakdown,
)
from core.recommendation.evaluator import VERIFIED_SCORE_METHOD  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ANALYSIS_ID = "analysis_test_001"

CANDIDATE_A = {
    "candidate_id": "cand_test_A",
    "issue_id": "iss_x",
    "action_type": "increase_section",
    "description": "1-step",
    "member_id": 1, "element_id": 1,
    "expected_effect": "", "tradeoffs": "",
    "requires_reanalysis": True, "confidence": "medium",
    "code_refs": [],
    "target": {"scope": "member", "member_id": 1, "element_id": 1,
               "member_type": "column", "story": 1},
    "proposed_change": {
        "operation": "replace_section",
        "from": "H-300x300", "to": "H-310x310",
        "requires_user_selection": False, "applicable": True,
        "reason": "strength_exceeded",
    },
    "metadata": {},
}
CANDIDATE_B = {
    "candidate_id": "cand_test_B",
    "issue_id": "iss_x",
    "action_type": "increase_section",
    "description": "2-step",
    "member_id": 1, "element_id": 1,
    "expected_effect": "", "tradeoffs": "",
    "requires_reanalysis": True, "confidence": "medium",
    "code_refs": [],
    "target": {"scope": "member", "member_id": 1, "element_id": 1,
               "member_type": "column", "story": 1},
    "proposed_change": {
        "operation": "replace_section",
        "from": "H-300x300", "to": "H-350x350",
        "requires_user_selection": False, "applicable": True,
        "reason": "strength_exceeded",
    },
    "metadata": {},
}
CANDIDATE_ABSTRACT = {
    "candidate_id": "cand_test_abstract",
    "issue_id": "iss_x",
    "action_type": "add_lateral_resistance",
    "description": "abstract",
    "expected_effect": "", "tradeoffs": "",
    "requires_reanalysis": True, "confidence": "low",
    "code_refs": [],
    "target": {"scope": "story", "story": 1, "direction": "X"},
    "proposed_change": {
        "operation": "add_lateral_resistance",
        "from": None, "to": None,
        "requires_user_selection": True, "applicable": False,
        "reason": "drift_exceeded",
    },
    "metadata": {"abstract": True},
}


def _seed_analysis_context():
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[ANALYSIS_ID] = {
            "model_json": {"nodes": [], "elements": []},
            "load_cases": {}, "load_combinations": {},
            "building_model": None, "seismic_report": None,
            "design_check": {
                "overall_status": "NG",
                "summary": {"max_drift_ratio": 0.005,
                            "max_interaction_ratio": 1.4,
                            "ng_members": 1, "ng_stories": 0},
                "drift_check": {"status": "OK", "checks": []},
                "member_check": {
                    "status": "NG",
                    "members": [
                        {"member_id": 1, "status": "NG"},
                        {"member_id": 2, "status": "OK"},
                    ],
                    "summary": {"max_interaction_ratio": 1.4, "ng": 1},
                },
            },
            "candidates_by_id": {
                CANDIDATE_A["candidate_id"]: CANDIDATE_A,
                CANDIDATE_B["candidate_id"]: CANDIDATE_B,
                CANDIDATE_ABSTRACT["candidate_id"]: CANDIDATE_ABSTRACT,
            },
            "expires_at": time.time() + 600,
        }


def _clear():
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache.clear()
    with _EVAL_JOBS_LOCK:
        eval_jobs_db.clear()


def _wait_for_done(client, eval_job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v2/recommendations/evaluate/{eval_job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in {"done", "failed"}:
            return data
        time.sleep(0.05)
    raise AssertionError(f"eval job {eval_job_id} did not finish in {timeout}s")


# ---------------------------------------------------------------------------
# Stub for evaluate_candidate (so we don't run OpenSees)
# ---------------------------------------------------------------------------

def _stub_evaluate_candidate(*, candidate, baseline, **_):
    """Return canned evaluations keyed by candidate_id."""
    cid = candidate.candidate_id
    if cid == CANDIDATE_A["candidate_id"]:
        # Best result — DCR drops the most.
        return CandidateEvaluation(
            candidate_id=cid, status=STATUS_EVALUATED,
            metrics={
                "max_interaction_ratio": 0.85, "max_drift_ratio": 0.005,
                "ng_member_count": 0, "ng_drift_count": 0,
                "weight_proxy": 5e5, "changed_member_count": 1,
                "analysis_succeeded": True,
            },
            improvement={"dcr_delta": -0.55, "drift_delta": 0.0,
                         "ng_member_delta": -1, "ng_drift_delta": 0},
            score=ScoreBreakdown(
                safety_gain=1.0, code_compliance=1.0, relative_cost=0.99,
                disruption=0.5, side_effect_risk=1.0, total=0.85,
                method=VERIFIED_SCORE_METHOD, verified=True, status="verified",
            ),
        )
    if cid == CANDIDATE_B["candidate_id"]:
        # Still applicable but more expensive (smaller dcr_delta).
        return CandidateEvaluation(
            candidate_id=cid, status=STATUS_EVALUATED,
            metrics={
                "max_interaction_ratio": 0.95, "max_drift_ratio": 0.005,
                "ng_member_count": 0, "ng_drift_count": 0,
                "weight_proxy": 8e5, "changed_member_count": 1,
                "analysis_succeeded": True,
            },
            improvement={"dcr_delta": -0.45, "drift_delta": 0.0,
                         "ng_member_delta": -1, "ng_drift_delta": 0},
            score=ScoreBreakdown(
                safety_gain=1.0, code_compliance=1.0, relative_cost=0.98,
                disruption=0.5, side_effect_risk=1.0, total=0.80,
                method=VERIFIED_SCORE_METHOD, verified=True, status="verified",
            ),
        )
    # Abstract would be filtered out before reaching here, but cover the
    # branch defensively.
    return CandidateEvaluation(
        candidate_id=cid, status=STATUS_SKIPPED_INAPPLICABLE,
        error="abstract candidate is not applicable",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup():
    _clear()
    yield
    _clear()


class TestEvaluateEndpoints:
    def test_missing_analysis_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post("/api/v2/recommendations/evaluate",
                               json={"candidate_ids": ["x"]})
            assert resp.status_code == 400
            assert "analysis_id" in resp.text

    def test_unknown_analysis_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post("/api/v2/recommendations/evaluate",
                               json={"analysis_id": "ghost",
                                     "candidate_ids": ["x"]})
            assert resp.status_code == 400
            assert "not found" in resp.text or "expired" in resp.text

    def test_unknown_eval_job_id_returns_404(self):
        with TestClient(app) as client:
            resp = client.get("/api/v2/recommendations/evaluate/missing")
            assert resp.status_code == 404

    def test_empty_candidate_ids_returns_400(self):
        _seed_analysis_context()
        with TestClient(app) as client:
            resp = client.post("/api/v2/recommendations/evaluate",
                               json={"analysis_id": ANALYSIS_ID,
                                     "candidate_ids": []})
            assert resp.status_code == 400


class TestEvaluateLifecycle:
    def test_post_returns_queued_then_poll_to_done(self):
        _seed_analysis_context()

        # The worker imports ``evaluate_candidate`` via the
        # ``core.recommendation`` re-export, so we patch that name —
        # patching ``core.recommendation.evaluator.evaluate_candidate``
        # alone leaves the re-export pointing at the original.
        with patch("core.recommendation.evaluate_candidate",
                   side_effect=_stub_evaluate_candidate):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v2/recommendations/evaluate",
                    json={
                        "analysis_id": ANALYSIS_ID,
                        "candidate_ids": [
                            CANDIDATE_A["candidate_id"],
                            CANDIDATE_B["candidate_id"],
                        ],
                    },
                )
                assert resp.status_code == 200
                start = resp.json()
                assert start["status"] == "queued"
                assert start["job_id"].startswith("rec_eval_")
                assert start["num_total"] == 2
                assert start["num_applicable_requested"] == 2

                final = _wait_for_done(client, start["job_id"])
                assert final["status"] == "done", final
                # Better candidate (A, dcr_delta -0.55) ranks first.
                assert final["ranked_order"] == [
                    CANDIDATE_A["candidate_id"],
                    CANDIDATE_B["candidate_id"],
                ]
                # Verified summary markers
                s = final["recommendation_summary"]
                assert s["ranking_verified"] is True
                assert s["score_method"] == VERIFIED_SCORE_METHOD
                assert s["num_evaluated"] == 2
                assert s["num_rejected_new_ng"] == 0
                assert s["num_skipped_inapplicable"] == 0

    def test_abstract_candidate_is_reported_as_skipped(self):
        _seed_analysis_context()
        with patch("core.recommendation.evaluate_candidate",
                   side_effect=_stub_evaluate_candidate):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v2/recommendations/evaluate",
                    json={
                        "analysis_id": ANALYSIS_ID,
                        "candidate_ids": [
                            CANDIDATE_A["candidate_id"],
                            CANDIDATE_ABSTRACT["candidate_id"],
                        ],
                    },
                )
                assert resp.status_code == 200
                start = resp.json()
                assert start["num_applicable_requested"] == 1

                final = _wait_for_done(client, start["job_id"])
                # Abstract candidate gets filtered out via applicable=False.
                # The evaluator's own short-circuit (we don't stub that
                # branch) handles it deterministically inside the worker.
                assert final["status"] == "done"
                rejected_ids = {
                    r["candidate_id"] for r in final["rejected_candidates"]
                }
                assert CANDIDATE_ABSTRACT["candidate_id"] in rejected_ids
                ranked_ids = final["ranked_order"]
                assert CANDIDATE_A["candidate_id"] in ranked_ids
                assert CANDIDATE_ABSTRACT["candidate_id"] not in ranked_ids
