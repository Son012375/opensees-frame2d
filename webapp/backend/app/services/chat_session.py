"""Skeleton for chat-session state (Phase A wires it, Phase C extends).

Defined here in Phase 0 so the chat router has a single import target
when it lands. Nothing reads from ``chat_session_cache`` or
``virtual_candidates_by_id`` yet — wiring happens once the chat
orchestrator is added.
"""
from __future__ import annotations

import threading

chat_session_cache: dict[str, dict] = {}
_CHAT_SESSION_LOCK = threading.Lock()
_CHAT_SESSION_TTL_SEC = 30 * 60

# Phase C: per-session virtual candidate store keyed by preview_id so a
# user-confirmed Apply can resolve back to the exact diff the chatbot
# offered. Not used yet.
virtual_candidates_by_id: dict[str, dict[str, dict]] = {}
