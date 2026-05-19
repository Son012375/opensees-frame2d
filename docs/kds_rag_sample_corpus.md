# KDS RAG Sample Corpus (Phase 3B Smoke)

This directory holds a **minimal** retrieval corpus used to smoke-test the
end-to-end Voyage KDS RAG path. The full standards live under
`C:\Users\youm\Desktop\RAG용\` as PDFs and are NOT ingested here — PDF
parsing is Phase 4 work.

Each file is a paraphrased clause summary (never the verbatim standard
text) carrying only the metadata fields needed by
`core.kds_rag.ingest`:

| File | Topic | Limit state | Issue type covered | Reference status |
|------|-------|-------------|--------------------|-------------------|
| `kds_41_17_00_drift.json` | `story_drift` | `drift_limit` | `drift_exceeded` | KDS 원문 (`temporary_reference: false`) |
| `aisc_360_22_ch_f_flexure.json` | `member_strength` | `strength` | `strength_exceeded` | **AISC 임시 참조** (`temporary_reference: true`) |
| `aisc_360_22_ch_g_shear.json` | `member_shear` | `shear_strength` | `shear_exceeded` | **AISC 임시 참조** (`temporary_reference: true`) |

## Why these three

The deterministic explainer in `core.recommendation.explainer` builds a
KDS query per issue. Three of our five `issue_type` values dominate
production reports:

- `drift_exceeded` → KDS 41 17 00 §8.2 (Korean limit on inter-story drift)
- `strength_exceeded` → AISC 360-22 Chapter F (flexural strength)
- `shear_exceeded` → AISC 360-22 Chapter G (shear strength)

This smoke corpus exercises one chunk per query path. It is intentionally
tiny so the build is cheap and a manual eyeball check of the top-1 result
is easy.

## How to build the index

See `docs/kds_rag_smoke.md` for the full procedure. Short form:

```powershell
$env:VOYAGE_API_KEY = "vo-..."
python scripts/build_kds_rag_index.py `
  --source-dir data/kds_sample `
  --index-path data/kds_sample_index.jsonl
```

## AISC chunks are a TEMPORARY reference (not "KDS-equivalent")

The Korean steel-structure standards (KDS 14 31 00 / KDS 41 31 00) are
**not yet** in `C:\Users\youm\Desktop\RAG용\`. For Phase 3B the
flexure / shear chunks cite AISC 360-22 only as a stand-in. Three
guard-rails enforce that we never claim KDS-equivalence by accident:

1. The chunk **text body** contains zero "KDS와 동등하다" sentences. The
   text only describes AISC behavior. (Previous revisions had a closing
   "KDS 14 31 00 §6/§7 동등" line — removed after GPT cross-review.)
2. Each AISC JSON carries machine-readable metadata flags:

   ```json
   "temporary_reference": true,
   "replacement_target": "KDS 14 31 00 / KDS 41 31 00",
   "replacement_note": "강구조 KDS 원문 확보 전까지 AISC 360-22를 임시 참조로 사용한다. KDS와의 동등성은 검증되지 않음."
   ```

   The current ingester (`core.kds_rag.ingest._doc_from_json`) silently
   drops unknown fields so these flags do **not** leak into the JSONL
   index — they exist for a future UI surface to read directly from the
   source JSON.
3. When `temporary_reference: true` chunks surface in the Explain modal,
   the user-facing copy MUST say:

   > 현재 강구조 근거는 KDS 원문이 아닌 AISC 360-22 임시 참조입니다.
   > KDS 14 31 00 / KDS 41 31 00 원문 확보 후 교체 검증이 필요합니다.

   Wiring this UI string is a follow-up (Phase 4 — alongside LLM
   provider activation), but the data side is already truthful.

Once a KDS-side PDF is acquired the operator can drop a parallel `.json`
file in this directory, flip `temporary_reference` to `false` on the
KDS version, and let dense retrieval rank the KDS chunk above the AISC
one. No pipeline change needed.

## Scope guarantees

- All text is paraphrased, never the verbatim clause.
- Each file ≤ ~2 KB so the smoke build hits Voyage with ≤ 10 chunks.
- No PDF, no binary, no third-party reproductions.
- "KDS-equivalent" is never claimed in chunk text. AISC chunks are
  labelled `temporary_reference: true` in their JSON metadata.
