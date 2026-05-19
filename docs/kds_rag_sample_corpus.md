# KDS RAG Sample Corpus (Phase 3B Smoke)

This directory holds a **minimal** retrieval corpus used to smoke-test the
end-to-end Voyage KDS RAG path. The full standards live under
`C:\Users\youm\Desktop\RAG용\` as PDFs and are NOT ingested here — PDF
parsing is Phase 4 work.

Each file is a paraphrased clause summary (never the verbatim standard
text) carrying only the metadata fields needed by
`core.kds_rag.ingest`:

| File | Topic | Limit state | Issue type covered |
|------|-------|-------------|--------------------|
| `kds_41_17_00_drift.json` | `story_drift` | `drift_limit` | `drift_exceeded` |
| `aisc_360_22_ch_f_flexure.md` | `member_strength` | `strength` | `strength_exceeded` |
| `aisc_360_22_ch_g_shear.md` | `member_shear` | `shear_strength` | `shear_exceeded` |

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

## Why we do NOT cite KDS 14 31 00 / KDS 41 31 00

The Korean steel-structure standards are not yet in
`C:\Users\youm\Desktop\RAG용\`. For Phase 3B we cite AISC 360-22 as the
source for strength/shear; once a KDS-side PDF is acquired the operator
can drop a parallel `.md` / `.json` file in this directory and rebuild
the index. Nothing else has to change.

## Scope guarantees

- All text is paraphrased, never the verbatim clause.
- Each file ≤ ~2 KB so the smoke build hits Voyage with ≤ 10 chunks.
- No PDF, no binary, no third-party reproductions.
