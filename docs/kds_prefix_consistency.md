# 14b Hybrid-Prefix Consistency Probe

Phase D's hybrid anti-hallucination lets the chat LLM (qwen2.5:14b) write a
short Korean **prefix** before the deterministic KDS/AISC citation block.
The orchestrator sanitizes that prefix (`_sanitize_llm_prefix`): empty,
fabricated-clause (`KDS NN NN NN` / `§N.N` / `AISC NNN`), or over-long
output is discarded. An informal earlier observation was that 14b *often
returns an empty prefix*, leaving the user with just the deterministic
summary. This probe quantifies that.

## Method

`scripts/kds_prefix_consistency.py` reproduces the orchestrator's prefix
step directly (no Voyage, no force-routing):

- system prompt = `DEFAULT_SYSTEM_PROMPT` (rule 7: 1–2 sentence prefix,
  numbers/status only, **no clause numbers**).
- tool message = the explain result AFTER `_pop_mandatory_response` strips
  `kds_evidence` + `mandatory_response_*` — i.e. `member_summary` (fixed:
  member #5, column, NG, governing=shear, ratios shear 1.28 / interaction
  0.42) + `warnings: []` + `answer_hint`.
- user message = each of 12 diverse phrasings ("왜 NG?", "이 부재 왜
  불합격이야?", "근거 좀 보여줘", "이 기둥 뭐가 문제야?", …).

Each raw output is classified exactly as the orchestrator would:
**kept** (passes sanitizer) / **empty** / **fabricated** / **toolong**.

Run (needs the live 14b — SSH tunnel up, `OLLAMA_BASE_URL`/`OLLAMA_MODEL`
in `.env`):

```bash
python scripts/kds_prefix_consistency.py --repeats 3
```

Ollama is the user's own server (no Voyage-style RPM cap), so phrasings run
back-to-back.

## Baseline (2026-05-29, qwen2.5:14b, temp ≈ 0.1, fixed shear-NG member)

**kept = 36/36 (100%)** over 12 phrasings × 3 repeats. empty = 0,
fabricated = 0, toolong = 0, error = 0.

Every prefix was a clean 1-sentence intro built from `member_summary`, e.g.:

> 5번 부재가 전단력 과다로 NG 판정되었습니다. 주요 근거는 다음과 같습니다:

with **no clause numbers** (rule 7 honored). Some variants embed the actual
ratios ("주요 균형 비율은 다음과 같습니다: 전단력 1.28, 인터랙션 0.42") —
allowed (numbers from `member_summary`), not a fabrication. Near-deterministic
across repeats at temp 0.1.

## Interpretation

The "frequent empty prefix" concern **does not reproduce** in the current
state (AISC proxy removed, `warnings: []`, clear `member_summary`). No code
change needed; rule 7 + low temperature are sufficient for consistent,
clause-free prefixes. The sanitizer remains the hard guard (locked by
`tests/test_chat_kds_tool.py`) for the rare fabrication.

### Scope / future

- `member_summary` is held fixed (shear-NG) to isolate the *phrasing* axis.
  Varying the governing ratio (axial / flexure / interaction) is a cheap
  extension if prefix behavior per limit-state ever matters.
- Minor cosmetic: "균형 비율" is an awkward gloss for "interaction ratio" —
  not wrong (uses real numbers), not worth a prompt change.
