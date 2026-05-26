"""FastAPI router for ``/api/v2/chat/*`` — Phase A.2.

Mounts two POST endpoints plus a debug GET:

    POST /api/v2/chat/sessions       create a session (optional analysis bind)
    POST /api/v2/chat/messages       send a turn, receive NDJSON stream
    GET  /api/v2/chat/sessions/{id}  inspect history (debug only)

Phase A.2 wires the tool registry (``inspect_selection`` +
``get_analysis_summary``) and a per-session ``asyncio.Lock`` so two
concurrent ``/messages`` calls on the same session_id serialise through
the orchestrator instead of interleaving ``history`` (Codex P2 on
6112155). The OllamaProvider lands in A.3; today only scripted test
providers exercise the tool loop, but the route surface is already the
final A.2 one.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# Inject the mcp-server root so we can ``import core.chat.*`` at module
# load. main_simple uses the same MCP_SERVER_PATH constant lazily inside
# its endpoints; we reuse it here so a future relocation only has to be
# changed in one place.
from app.core.config import MCP_SERVER_PATH

if str(MCP_SERVER_PATH) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_PATH))

from core.chat.llm.base import BaseLLMProvider  # noqa: E402
from core.chat.llm.noop_provider import NoopProvider  # noqa: E402
from core.chat.orchestrator import ChatOrchestrator  # noqa: E402
from core.chat.tool_registry import ToolRegistry, default_registry  # noqa: E402

from app.services.chat_session import (  # noqa: E402
    _CHAT_SESSION_LOCK,
    _CHAT_SESSION_TTL_SEC,
    chat_session_cache,
)


# Module-level singleton — the registry is stateless and reading
# ``CHAT_TOOLS_ENABLED`` once at import time matches how the rest of
# main_simple treats env-driven config (no per-request re-parsing).
_TOOL_REGISTRY: ToolRegistry = default_registry()


router = APIRouter(prefix="/api/v2/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Session cache helpers (mirror analysis_context patterns)
# ---------------------------------------------------------------------------

def _purge_expired_sessions() -> None:
    now = time.time()
    with _CHAT_SESSION_LOCK:
        expired = [
            sid for sid, sess in chat_session_cache.items()
            if sess.get("expires_at", 0) < now
        ]
        for sid in expired:
            chat_session_cache.pop(sid, None)


def _get_session(session_id: str) -> Optional[dict]:
    """TTL-aware read. Returns ``None`` for unknown or expired ids."""
    now = time.time()
    with _CHAT_SESSION_LOCK:
        sess = chat_session_cache.get(session_id)
        if sess is None:
            return None
        if sess.get("expires_at", 0) < now:
            chat_session_cache.pop(session_id, None)
            return None
        return sess


# ---------------------------------------------------------------------------
# LLM provider resolution (env-driven, NoopProvider fallback)
# ---------------------------------------------------------------------------

def _resolve_llm_provider() -> BaseLLMProvider:
    """Pick a provider based on ``CHAT_LLM_PROVIDER``.

    Phase A.3 wires Ollama. If the daemon is down or the env is
    misconfigured we fall back to :class:`NoopProvider` with a logger
    warning rather than 500-ing — the route's streaming contract
    matters more than the LLM being live, and the Noop placeholder
    text tells the user what to fix.
    """
    name = (os.environ.get("CHAT_LLM_PROVIDER") or "noop").strip().lower()

    if name == "ollama":
        try:
            from core.chat.llm.ollama_provider import OllamaProvider
            return OllamaProvider()
        except Exception as exc:  # noqa: BLE001 — keep route up
            logger.warning(
                "CHAT_LLM_PROVIDER=ollama but provider init failed (%s) — "
                "falling back to NoopProvider", exc,
            )
            return NoopProvider(message=(
                f"[Ollama 초기화 실패] {exc} — "
                "OLLAMA_BASE_URL / OLLAMA_MODEL 환경 변수를 확인하거나 "
                "`ollama serve`가 실행 중인지 확인하세요."
            ))

    if name and name != "noop":
        logger.warning(
            "CHAT_LLM_PROVIDER=%r is not a recognised provider — using NoopProvider",
            name,
        )
    return NoopProvider()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    analysis_id: Optional[str] = Field(
        default=None,
        description="Optional /api/v2/analyze job_id to bind this chat to.",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    provider: str
    expires_at: float
    analysis_id: Optional[str] = None


class PostMessageRequest(BaseModel):
    session_id: str
    message: str
    ui_context: Optional[dict] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(payload: CreateSessionRequest) -> CreateSessionResponse:
    _purge_expired_sessions()
    provider = _resolve_llm_provider()
    session_id = "chat_" + uuid.uuid4().hex[:16]
    now = time.time()
    expires_at = now + _CHAT_SESSION_TTL_SEC
    with _CHAT_SESSION_LOCK:
        chat_session_cache[session_id] = {
            "session_id": session_id,
            "analysis_id": payload.analysis_id,
            "history": [],
            "provider": provider.name,
            "created_at": now,
            "expires_at": expires_at,
        }
    return CreateSessionResponse(
        session_id=session_id,
        provider=provider.name,
        expires_at=expires_at,
        analysis_id=payload.analysis_id,
    )


@router.post("/messages")
async def post_message(payload: PostMessageRequest):
    """Stream a one-turn chat response as NDJSON.

    Returns 404 for unknown / expired sessions and 400 for blank
    messages. On success the response is ``application/x-ndjson`` with
    one JSON object per line, terminated by ``done``.

    Concurrent POSTs to the same session_id serialise through a
    per-session ``asyncio.Lock`` (Codex P2 from the 6112155 review):
    history mutation while another turn streams would otherwise
    interleave user/assistant entries. The lock is created lazily on
    first use and lives on the session dict — disposed when the session
    expires.
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    sess = _get_session(payload.session_id)
    if sess is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown or expired chat session: {payload.session_id}",
        )

    # Refresh the session's TTL on activity so an active conversation
    # doesn't expire mid-turn.
    sess["expires_at"] = time.time() + _CHAT_SESSION_TTL_SEC

    # Per-session busy lock — see docstring above. dict.setdefault is
    # atomic under the GIL, and asyncio.Lock() construction is cheap +
    # lazy so a rare double-create is harmless.
    session_lock: asyncio.Lock = sess.setdefault("_busy_lock", asyncio.Lock())

    provider = _resolve_llm_provider()
    orchestrator = ChatOrchestrator(llm=provider, registry=_TOOL_REGISTRY)

    async def _stream():
        async with session_lock:
            async for line in orchestrator.run_turn(
                session=sess,
                user_message=payload.message,
                ui_context=payload.ui_context or {},
            ):
                yield line

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Debug-only history inspector. Strips per-turn ``ui_context`` snapshots
    so a noisy frontend doesn't dominate the response payload."""
    sess = _get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": sess["session_id"],
        "analysis_id": sess.get("analysis_id"),
        "provider": sess.get("provider"),
        "created_at": sess["created_at"],
        "expires_at": sess["expires_at"],
        "history": [
            {"role": h["role"], "content": h["content"]}
            for h in sess.get("history", [])
        ],
    }
