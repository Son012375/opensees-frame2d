"""Async job state + worker pool for ``/api/v2/recommendations/evaluate``.

``eval_jobs_db[eval_job_id]`` tracks each batch reanalysis (status,
progress, results, errors). OpenSees keeps process-global state, so
reanalysis runs are serialised through a single-worker thread pool
(:data:`MAX_PARALLEL_EVALS` = 1).

Today the executor is exposed as :data:`_eval_executor` (underscore =
internal) and ``main_simple.post_recommendations_evaluate`` is its only
caller. Before Phase B (chat-router recommendations tooling) lands we
plan to add public ``submit_eval`` / ``poll_eval`` wrappers here so new
callers go through a documented service API instead of touching the
private executor directly.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

eval_jobs_db: dict[str, dict] = {}
_EVAL_JOBS_LOCK = threading.Lock()

MAX_PARALLEL_EVALS = 1
_eval_executor = ThreadPoolExecutor(
    max_workers=MAX_PARALLEL_EVALS,
    thread_name_prefix="rec-eval",
)


def _rehydrate_candidate(cand_dict: dict):
    """Reconstruct a :class:`RetrofitCandidate` from its ``to_dict`` form."""
    # Imported lazily so this module does not pull mcp-server's core
    # package at import time (keeps the chat-router import path light).
    from core.recommendation import RetrofitCandidate

    return RetrofitCandidate(
        candidate_id=cand_dict["candidate_id"],
        issue_id=cand_dict["issue_id"],
        action_type=cand_dict["action_type"],
        description=cand_dict.get("description", ""),
        member_id=cand_dict.get("member_id"),
        element_id=cand_dict.get("element_id"),
        expected_effect=cand_dict.get("expected_effect", ""),
        tradeoffs=cand_dict.get("tradeoffs", ""),
        requires_reanalysis=cand_dict.get("requires_reanalysis", True),
        confidence=cand_dict.get("confidence", "low"),
        code_refs=[],   # RAG references not needed for execution
        target=dict(cand_dict.get("target") or {}),
        proposed_change=dict(cand_dict.get("proposed_change") or {}),
        metadata=dict(cand_dict.get("metadata") or {}),
    )


rehydrate_candidate = _rehydrate_candidate
