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
            # Mirror /api/v2/analyze's marker so the recommendation
            # endpoints accept this fixture. /api/building/analyze caches
            # mark themselves "compact" and are explicitly rejected by
            # ``_assert_full_context_for_recommendations``.
            "context_kind": "full",
            "expires_at": time.time() + 600,
        }


def _seed_compact_analysis_context():
    """Mimic what /api/building/analyze writes — only the compact chat-
    tool subset, no model_json / candidates_by_id. The recommendation
    endpoints must reject this with 422 instead of crashing on the
    missing keys."""
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[ANALYSIS_ID] = {
            # Compact subset fields only — what build_compact_subset emits.
            "analysis_summary": {"ng_count": 1, "num_stories": 3},
            "envelope": {},
            "member_info_by_elem_id": {},
            "member_info_by_member_id": {},
            "member_ratios_by_elem_id": {},
            "member_ratios_by_member_id": {},
            "modal_summary": None,
            "context_kind": "compact",
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
        """An analysis_id that was never created is a client mistake
        (400 Bad Request), distinct from one that existed but expired."""
        with TestClient(app) as client:
            resp = client.post("/api/v2/recommendations/evaluate",
                               json={"analysis_id": "ghost",
                                     "candidate_ids": ["x"]})
            assert resp.status_code == 400
            assert "not found" in resp.text

    def test_expired_analysis_context_returns_410_and_evicts(self):
        """An entry whose ``expires_at`` is in the past must be treated
        as gone (HTTP 410), evicted from the cache on the read attempt,
        and never seen by the worker."""
        _seed_analysis_context()
        with _ANALYSIS_CONTEXT_LOCK:
            analysis_context_cache[ANALYSIS_ID]["expires_at"] = (
                time.time() - 1
            )
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/evaluate",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_ids": [CANDIDATE_A["candidate_id"]]},
            )
        assert resp.status_code == 410, resp.text
        assert "expired" in resp.text
        with _ANALYSIS_CONTEXT_LOCK:
            assert ANALYSIS_ID not in analysis_context_cache, (
                "expired context should be evicted on the failed read"
            )

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


class TestCompactContextRejected:
    """Codex P2 regression: /api/building/analyze caches only the compact
    chat-tool subset (no model_json / candidates_by_id), so the recom-
    mendation endpoints must surface a clear 422 rather than crash on
    ``ctx['model_json']`` or silently process zero candidates."""

    def test_evaluate_rejects_compact_context_with_422(self):
        _seed_compact_analysis_context()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/evaluate",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_ids": [CANDIDATE_A["candidate_id"]]},
            )
        assert resp.status_code == 422, resp.text
        # Message names the actual recovery path
        assert "/api/building/analyze" in resp.text
        assert "/api/v2/analyze" in resp.text

    def test_preview_apply_rejects_compact_context_with_422(self):
        _seed_compact_analysis_context()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 422, resp.text
        # Must surface BEFORE the candidate_id lookup so the message
        # tells the user about the baseline, not about a missing candidate.
        assert "compact" in resp.text

    def test_explain_rejects_compact_context_with_422(self):
        _seed_compact_analysis_context()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 422, resp.text
        assert "compact" in resp.text


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


# ---------------------------------------------------------------------------
# Preview-apply endpoint (Phase 2)
# ---------------------------------------------------------------------------

def _real_model() -> dict:
    """One column at story 1 — the smallest model that ``apply_candidate``
    can actually mutate. Element id=1 carries section ``H-300x300`` so
    ``CANDIDATE_A`` (replace_section from H-300x300 → H-310x310, targeting
    element_id=1) succeeds."""
    return {
        "version": "2.0",
        "nodes": [
            {"id": 1, "x": 0, "y": 0, "z": 0,
             "story": 0, "support": "fixed", "mass": 0.0},
            {"id": 2, "x": 0, "y": 0, "z": 3500,
             "story": 1, "support": None, "mass": 0.0},
        ],
        "elements": [
            {"id": 1, "node_i": 1, "node_j": 2,
             "elem_type": "column", "section": "H-300x300",
             "material": "SS275",
             "release_i": None, "release_j": None, "beta_angle": 0.0},
        ],
        "story_elevations": [0, 3500],
        "story_usages": {1: "office"},
    }


def _seed_with_real_model(*candidates: dict) -> None:
    """Seed analysis_context_cache with a model that has real elements
    plus the supplied candidate dicts (already in to_dict-ish shape)."""
    with _ANALYSIS_CONTEXT_LOCK:
        analysis_context_cache[ANALYSIS_ID] = {
            "model_json": _real_model(),
            "load_cases": {}, "load_combinations": {},
            "building_model": None, "seismic_report": None,
            "design_check": {"overall_status": "OK"},
            "candidates_by_id": {c["candidate_id"]: c for c in candidates},
            # Full /api/v2/analyze marker — recommendation endpoints accept.
            "context_kind": "full",
            "expires_at": time.time() + 600,
        }


class TestPreviewApplyEndpoint:
    def test_preview_apply_returns_diff_and_updated_model(self):
        _seed_with_real_model(CANDIDATE_A)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["candidate_id"] == CANDIDATE_A["candidate_id"]
        assert data["applicable"] is True

        diff = data["diff"]
        assert diff["candidate_id"] == CANDIDATE_A["candidate_id"]
        assert diff["operation"] == "replace_section"
        assert diff["changed_member_count"] >= 1
        assert len(diff["changed_members"]) == diff["changed_member_count"]
        row = diff["changed_members"][0]
        # All UI-required keys present.
        for key in ("element_id", "member_id", "member_label",
                    "member_type", "section_from", "section_to",
                    "story", "reason"):
            assert key in row, f"missing diff key: {key}"
        assert row["section_from"] == "H-300x300"
        assert row["section_to"] == "H-310x310"
        assert row["member_label"]  # non-empty

        # updated_model carries the new section so the frontend can
        # swap window._v2Model without further work.
        assert isinstance(data["updated_model"], dict)
        new_elems = data["updated_model"]["elements"]
        assert any(e["id"] == 1 and e["section"] == "H-310x310"
                   for e in new_elems)

    def test_preview_apply_does_not_mutate_cached_model(self):
        """Two preview-apply calls in a row must both succeed and return
        identical diffs — the cached model_json must NOT drift toward
        the proposed section."""
        _seed_with_real_model(CANDIDATE_A)
        import copy
        with _ANALYSIS_CONTEXT_LOCK:
            snapshot = copy.deepcopy(
                analysis_context_cache[ANALYSIS_ID]["model_json"]
            )

        with TestClient(app) as client:
            r1 = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
            assert r1.status_code == 200
            with _ANALYSIS_CONTEXT_LOCK:
                assert (
                    analysis_context_cache[ANALYSIS_ID]["model_json"]
                    == snapshot
                ), "cached model_json was mutated by the first call"

            r2 = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
            assert r2.status_code == 200
            assert r1.json()["diff"] == r2.json()["diff"]

    def test_preview_apply_missing_analysis_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"candidate_id": "x"},
            )
            assert resp.status_code == 400
            assert "analysis_id" in resp.text

    def test_preview_apply_missing_candidate_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": "x"},
            )
            assert resp.status_code == 400
            assert "candidate_id" in resp.text

    def test_preview_apply_unknown_analysis_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": "ghost",
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
            assert resp.status_code == 400
            assert "not found" in resp.text

    def test_preview_apply_expired_analysis_id_returns_410_and_evicts(self):
        _seed_with_real_model(CANDIDATE_A)
        with _ANALYSIS_CONTEXT_LOCK:
            analysis_context_cache[ANALYSIS_ID]["expires_at"] = (
                time.time() - 1
            )
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 410, resp.text
        assert "expired" in resp.text
        with _ANALYSIS_CONTEXT_LOCK:
            assert ANALYSIS_ID not in analysis_context_cache

    def test_preview_apply_unknown_candidate_id_returns_404(self):
        _seed_with_real_model(CANDIDATE_A)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": "cand_does_not_exist"},
            )
        assert resp.status_code == 404
        assert "not found" in resp.text

    def test_preview_apply_abstract_candidate_returns_422(self):
        _seed_with_real_model(CANDIDATE_ABSTRACT)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_ABSTRACT["candidate_id"]},
            )
        assert resp.status_code == 422, resp.text
        assert "not applicable" in resp.text
        # The 422 mentions the operation so the UI can label the failure.
        assert "add_lateral_resistance" in resp.text

    def test_preview_apply_inapplicable_error_returns_422(self):
        """A candidate that's flagged applicable=True but turns out to
        be a no-op (section already equals proposed_change.to) must
        surface the InapplicableCandidateError as a 422."""
        noop_candidate = {
            **CANDIDATE_A,
            "candidate_id": "cand_test_noop",
            "proposed_change": {
                **CANDIDATE_A["proposed_change"],
                "to": "H-300x300",   # same as the element's current section
            },
        }
        _seed_with_real_model(noop_candidate)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/preview-apply",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": "cand_test_noop"},
            )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"].startswith("apply failed:")


class TestExplainEndpoint:
    """``/api/v2/recommendations/explain`` — Phase 3A.

    The endpoint must succeed even when no KDS retriever is configured
    (the Noop default is the supported fallback) and must never mutate
    the cached model_json.
    """

    @pytest.fixture(autouse=True)
    def _force_noop_retriever(self, monkeypatch):
        """Clear Voyage env vars so ``get_default_kds_retriever()`` always
        returns ``NoopKDSRetriever`` during these tests, regardless of
        the operator's local ``.env``.

        Without this fixture, the moment a developer activates Voyage
        locally (sets VOYAGE_API_KEY + KDS_RAG_INDEX_PATH in .env), the
        ``rag_used is False`` assertion below starts failing because the
        factory returns a real retriever and the sample corpus actually
        produces evidence. The Explain endpoint contract under test here
        is the Noop fallback path — keep it deterministic.
        """
        for key in ("VOYAGE_API_KEY", "VOYAGEAI_API_KEY",
                    "KDS_RAG_INDEX_PATH"):
            monkeypatch.delenv(key, raising=False)
        yield

    def test_explain_happy_path_applicable_candidate_without_evaluation(self):
        _seed_with_real_model(CANDIDATE_A)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["candidate_id"] == CANDIDATE_A["candidate_id"]
        # 8 required explanation sections
        for k in ("summary", "issue_interpretation", "recommended_change",
                  "expected_structural_effect", "verified_result",
                  "tradeoffs", "limitations", "next_user_decision"):
            assert k in data["explanation"]
            assert isinstance(data["explanation"][k], str)
            assert data["explanation"][k].strip()
        # Without a configured retriever, rag_used=False and a warning says so.
        assert data["source"]["rag_used"] is False
        assert data["source"]["llm_used"] is False
        joined = " | ".join(data["warnings"])
        assert "kds_rag_unavailable" in joined or "kds_evidence_missing" in joined

    def test_explain_with_supplied_evaluation_marks_verified(self):
        _seed_with_real_model(CANDIDATE_A)
        verified_eval = {
            "candidate_id": CANDIDATE_A["candidate_id"],
            "status": STATUS_EVALUATED,
            "metrics": {"max_interaction_ratio": 0.85,
                        "max_drift_ratio": 0.005,
                        "ng_member_count": 0,
                        "ng_drift_count": 0,
                        "weight_proxy": 5e5,
                        "changed_member_count": 1,
                        "analysis_succeeded": True},
            "improvement": {"dcr_delta": -0.55, "drift_delta": 0.0,
                            "ng_member_delta": -1, "ng_drift_delta": 0},
            "score": {"safety_gain": 1.0, "code_compliance": 1.0,
                      "relative_cost": 0.99, "disruption": 0.5,
                      "side_effect_risk": 1.0, "total": 0.85,
                      "method": VERIFIED_SCORE_METHOD,
                      "verified": True, "status": "verified"},
            "error": None,
        }
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"],
                      "evaluation": verified_eval},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["source"]["score_method"] == VERIFIED_SCORE_METHOD
        assert "재해석" in data["explanation"]["verified_result"]

    def test_explain_missing_analysis_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"candidate_id": "x"},
            )
        assert resp.status_code == 400
        assert "analysis_id" in resp.text

    def test_explain_missing_candidate_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": "x"},
            )
        assert resp.status_code == 400
        assert "candidate_id" in resp.text

    def test_explain_unknown_analysis_id_returns_400(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": "ghost",
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 400
        assert "not found" in resp.text

    def test_explain_expired_analysis_id_returns_410(self):
        _seed_with_real_model(CANDIDATE_A)
        with _ANALYSIS_CONTEXT_LOCK:
            analysis_context_cache[ANALYSIS_ID]["expires_at"] = (
                time.time() - 1
            )
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 410, resp.text
        assert "expired" in resp.text

    def test_explain_unknown_candidate_id_returns_404(self):
        _seed_with_real_model(CANDIDATE_A)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": "cand_does_not_exist"},
            )
        assert resp.status_code == 404
        assert "not found" in resp.text

    def test_explain_abstract_candidate_returns_200_with_warning(self):
        """Unlike /preview-apply (which 422s), /explain succeeds for an
        abstract candidate and emits a manual-review warning."""
        _seed_with_real_model(CANDIDATE_ABSTRACT)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_ABSTRACT["candidate_id"]},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        joined = " | ".join(data["warnings"])
        assert "abstract_candidate" in joined
        # The next-step instruction must steer toward manual review.
        nxt = data["explanation"]["next_user_decision"]
        assert "엔지니어" in nxt or "수동" in nxt

    def test_explain_does_not_mutate_cached_model(self):
        import copy
        _seed_with_real_model(CANDIDATE_A)
        with _ANALYSIS_CONTEXT_LOCK:
            snapshot = copy.deepcopy(
                analysis_context_cache[ANALYSIS_ID]["model_json"]
            )
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 200
        with _ANALYSIS_CONTEXT_LOCK:
            assert (
                analysis_context_cache[ANALYSIS_ID]["model_json"]
                == snapshot
            ), "cached model_json was mutated by /explain"

    def test_explain_response_has_no_invented_kds_clause(self):
        """Regression: deterministic fallback must NOT print a fabricated
        KDS clause id when no retriever is configured."""
        import re as _re
        _seed_with_real_model(CANDIDATE_A)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v2/recommendations/explain",
                json={"analysis_id": ANALYSIS_ID,
                      "candidate_id": CANDIDATE_A["candidate_id"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        clause_re = _re.compile(r"KDS\s+\d{2}\s+\d{2}\s+\d{2}")
        for k, v in data["explanation"].items():
            assert not clause_re.search(v), (
                f"section {k!r} contains an invented KDS clause: {v!r}"
            )
        # And no evidence was returned, since the Noop retriever is the default.
        assert data["kds_evidence"] == []
