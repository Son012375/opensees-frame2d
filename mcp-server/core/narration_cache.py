"""Facts-hash cache for §10 종합검토의견 (역할 #3 narrator) — C2′ 결정성.

KHU/Mindlogic 게이트웨이는 temperature를 지원하지 않아 동일 입력에도 생성문이
매번 달라진다. 이 모듈은 (prose-safe) facts + 시스템 프롬프트 지문(prompt_hash)을
키로 **검증을 통과한 LLM 후보를 디스크에 저장**한다. 동일 facts·동일 프롬프트면
같은 후보를 재사용하므로 §10이 결정론적이 되고 Opus 호출(토큰)도 절감된다.

prompt_hash를 키에 포함하므로 few-shot/시스템 프롬프트가 바뀌면 캐시는 자동 무효화된다.

원자적·best-effort: 디스크 실패는 warning만 남기고 절대 해석을 깨지 않는다.
재사용 후보는 narrate_interpretation에서 다시 apply_narration으로 **재검증**되므로
(동일 facts → 동일 allowlist) anti-hallucination 안전성은 캐시 경로에서도 유지된다.

env:
    NARRATION_CACHE_DISABLE=1   캐시 비활성(항상 라이브 생성)
    NARRATION_CACHE_DIR=<path>  저장 위치 (기본 repo-root/data/narration_cache)
    NARRATION_CACHE_MAX=<n>     최대 항목 수 (기본 2000, 초과 시 오래된 것부터 제거)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MAX_ENTRIES = int(os.environ.get("NARRATION_CACHE_MAX", "2000"))


def _enabled() -> bool:
    return os.environ.get("NARRATION_CACHE_DISABLE", "") not in ("1", "true", "True")


def _cache_dir() -> Path:
    raw = os.environ.get("NARRATION_CACHE_DIR")
    if raw:
        return Path(raw)
    # core/ -> mcp-server/ -> repo root
    return Path(__file__).resolve().parents[2] / "data" / "narration_cache"


def cache_key(facts: dict, prompt_hash: str | None) -> str | None:
    """facts(정렬 직렬화) + prompt_hash 의 sha256(32hex). prompt_hash 없으면 None."""
    if not prompt_hash:
        return None
    try:
        blob = json.dumps(facts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    h = hashlib.sha256()
    h.update(blob.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(prompt_hash).encode("utf-8"))
    return h.hexdigest()[:32]


def get(key: str | None) -> dict | None:
    """캐시된 후보(dict) 반환. 미스/비활성/오류 시 None."""
    if not key or not _enabled():
        return None
    try:
        path = _cache_dir() / f"{key}.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception as exc:  # noqa: BLE001 — 캐시 실패는 라이브 생성으로 폴백
        logger.warning("narration cache read failed: %s", exc)
        return None


def put(key: str | None, value: dict) -> None:
    """검증 통과 후보를 원자적으로 저장(best-effort). 용량 초과 시 오래된 항목 제거."""
    if not key or not _enabled() or not isinstance(value, dict):
        return
    try:
        d = _cache_dir()
        with _LOCK:
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{key}.json"
            tmp = d / f"{key}.json.tmp"
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
            _evict_if_needed_locked(d)
    except Exception as exc:  # noqa: BLE001 — 캐시 실패는 무시
        logger.warning("narration cache write failed: %s", exc)


def _evict_if_needed_locked(d: Path) -> None:
    """항목 수가 _MAX_ENTRIES를 넘으면 mtime 오래된 것부터 제거 (caller holds _LOCK)."""
    if _MAX_ENTRIES <= 0:
        return
    entries = [p for p in d.iterdir() if p.suffix == ".json"]
    if len(entries) <= _MAX_ENTRIES:
        return
    entries.sort(key=lambda p: p.stat().st_mtime)
    for p in entries[: len(entries) - _MAX_ENTRIES]:
        try:
            p.unlink()
        except OSError:
            continue
