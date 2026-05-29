"""Measure qwen2.5:14b hybrid-prefix consistency for explain_member_compliance.

Phase D's anti-hallucination is by-construction: the KDS/AISC citation block
is rendered deterministically and the LLM only writes a short Korean PREFIX
(system-prompt rule 7), which the orchestrator sanitizes
(``_sanitize_llm_prefix`` — discard fabricated-clause or over-long output).
Observed informally: 14b often returns an EMPTY prefix, so the user just
sees the deterministic summary with no prose intro. This script quantifies
that: across diverse question phrasings (member_summary held fixed), how
often does 14b produce a *kept* prefix vs empty / fabricated / too-long?

It reproduces the orchestrator's prefix-step inputs directly (system prompt
+ a stripped explain tool result, exactly what ``_pop_mandatory_response``
leaves — kds_evidence + mandatory_response_* removed) and calls the live
OllamaProvider. Pure Ollama: no Voyage, no force-routing.

Needs the live 14b (e.g. SSH tunnel: ssh -L 18080:localhost:11500 ...) and
``OLLAMA_BASE_URL`` / ``OLLAMA_MODEL`` in ``.env``.

Usage:
    python scripts/kds_prefix_consistency.py [--repeats 1] [--temperature 0.1]

Exit codes: 0 ran; 2 Ollama unreachable for every call.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from core.chat.llm.ollama_provider import OllamaProvider  # noqa: E402
from core.chat.orchestrator import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    _FABRICATED_CLAUSE_RE,
    _MAX_PREFIX_CHARS,
    _sanitize_llm_prefix,
)

# Fixed member result (shear-governed NG) — isolates the phrasing effect.
MEMBER_SUMMARY = {
    "member_id": 5, "type": "column", "section": "H-300x300",
    "material": "SS275", "story": 2, "status": "NG",
    "governing_ratio": "shear",
    "ratios": {"interaction": 0.42, "shear": 1.28, "axial": 0.20, "bending": 0.55},
}
# Mirrors the tool entry AFTER _pop_mandatory_response strips kds_evidence
# and mandatory_response_* — i.e. what the prefix-step LLM actually sees.
STRIPPED_TOOL_RESULT = {
    "analysis_id": "prefix_bench",
    "member_summary": MEMBER_SUMMARY,
    "rag_used": True,
    "warnings": [],
    "answer_hint": (
        "mandatory_response_summary + mandatory_response_collapsible를 그대로 "
        "사용자에게 전달합니다 (LLM 합성 우회 — 환각 방지)."
    ),
}

PHRASINGS = [
    "5번 부재 왜 NG야?",
    "이 부재 왜 불합격이야?",
    "왜 안전하지 않아?",
    "이 기둥 뭐가 문제야?",
    "설계기준 근거 알려줘",
    "전단 때문에 NG인 거야?",
    "이 부재 안전한가요?",
    "왜 이게 통과를 못 했어?",
    "근거 좀 보여줘",
    "이 부재 검토 결과 설명해줘",
    "이거 왜 NG 떴어?",
    "5번 기둥 판정 이유가 뭐야?",
]


def _messages_for(question: str) -> list[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "tool", "name": "explain_member_compliance",
         "content": json.dumps(STRIPPED_TOOL_RESULT, ensure_ascii=False)},
    ]


def _classify(raw: str) -> tuple[str, str]:
    """Return (label, cleaned_or_raw). Mirrors _sanitize_llm_prefix order."""
    stripped = (raw or "").strip()
    if not stripped:
        return "empty", ""
    if len(stripped) > _MAX_PREFIX_CHARS:
        return "toolong", stripped
    if _FABRICATED_CLAUSE_RE.search(stripped):
        return "fabricated", stripped
    return "kept", _sanitize_llm_prefix(stripped) or stripped


async def _run(provider: OllamaProvider, temperature, repeats: int) -> int:
    counts = {"kept": 0, "empty": 0, "fabricated": 0, "toolong": 0, "error": 0}
    total = 0
    print(f"model={provider.model}  base={provider.base_url}  "
          f"temp={'default' if temperature is None else temperature}  "
          f"phrasings={len(PHRASINGS)}  repeats={repeats}\n")
    for q in PHRASINGS:
        for _ in range(repeats):
            total += 1
            raw = ""
            try:
                async for tok in provider.stream_tokens(
                    messages=_messages_for(q), temperature=temperature,
                ):
                    if tok:
                        raw += tok
            except Exception as exc:  # noqa: BLE001
                counts["error"] += 1
                print(f"  [ERROR ] {q!r}: {type(exc).__name__}: {exc}")
                continue
            label, text = _classify(raw)
            counts[label] += 1
            snippet = text.replace("\n", " ")[:80]
            print(f"  [{label:10}] {q!r} -> {snippet!r}")

    if counts["error"] == total:
        print("\nERROR: every call failed — is the Ollama 14b reachable "
              "(SSH tunnel up, OLLAMA_BASE_URL correct)?", file=sys.stderr)
        return 2

    scored = total - counts["error"]
    kept_rate = counts["kept"] / scored if scored else 0.0
    print(f"\ntotal={total} scored={scored}  "
          f"kept={counts['kept']} empty={counts['empty']} "
          f"fabricated={counts['fabricated']} toolong={counts['toolong']} "
          f"error={counts['error']}")
    print(f"kept-rate = {counts['kept']}/{scored} ({kept_rate:.0%})  "
          f"(fabricated reaching the user = {counts['fabricated']}; sanitizer "
          f"discards these, deterministic block still shown)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="14b hybrid-prefix consistency probe.")
    p.add_argument("--repeats", type=int, default=1,
                   help="Runs per phrasing (default 1).")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override temp (default = provider's, ~0.1 factual).")
    args = p.parse_args()
    provider = OllamaProvider()
    return asyncio.run(_run(provider, args.temperature, args.repeats))


if __name__ == "__main__":
    sys.exit(main())
