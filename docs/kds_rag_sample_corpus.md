# KDS RAG Sample Corpus

This directory holds the retrieval corpus used by the Voyage KDS RAG path
(`/explain` + the chat `explain_member_compliance` tool). It started as a
3-file Phase 3B smoke corpus; the steel-member design clauses are now
ingested from **KDS 14 31 10 강구조 부재 설계기준 (하중저항계수설계법, LRFD)**,
so the earlier AISC 360-22 stand-ins have been retired.

Each file is a **curated clause summary** (never the verbatim standard
text — copyright + UI length) carrying only the metadata fields read by
`core.kds_rag.ingest`. Equations are transcribed from the official PDF
(rendered to images, since the source PDF text is mojibake); fabrication
is forbidden.

| File | Topic | Limit state | Issue type covered | Standard / clause |
|------|-------|-------------|--------------------|-------------------|
| `kds_14_31_10_tension.json` | `member_axial` | `axial_strength` | `axial_exceeded` | KDS 14 31 10 §4.1.3 인장강도 |
| `kds_14_31_10_compression.json` | `member_axial` | `axial_strength` | `axial_exceeded` | KDS 14 31 10 §4.2.3 압축강도(휨좌굴) |
| `kds_14_31_10_flexure.json` | `member_flexure` | `flexural_strength` | `flexure_exceeded` | KDS 14 31 10 §4.3.2.1.1 휨강도 |
| `kds_14_31_10_shear.json` | `member_shear` | `shear_strength` | `shear_exceeded` | KDS 14 31 10 §4.3.2.1.2 전단강도 |
| `kds_14_31_10_interaction.json` | `member_strength` | `strength` | `strength_exceeded` | KDS 14 31 10 §4.4.1.1 조합력 상관식 |
| `kds_41_17_00_drift.json` | `story_drift` | `drift_limit` | `drift_exceeded` | KDS 41 17 00 §8.2.3 허용 층간변위비 |

All six chunks are **real KDS** (`jurisdiction: "KDS"`, `temporary_reference:
false`), so the `aisc_temporary_reference` proxy warning no longer fires
for any member design-check query.

## Coverage vs. issue_type buckets

The chat tool's `_governing_issue_type` (and the recommendation explainer)
route a member's governing ratio to one of these `issue_type` buckets;
each now has a KDS chunk:

- `axial_exceeded` → §4.1 인장 / §4.2 압축
- `flexure_exceeded` → §4.3.2.1.1 휨
- `shear_exceeded` → §4.3.2.1.2 전단
- `strength_exceeded` (조합 P+M) → §4.4.1.1 상관식
- `drift_exceeded` → KDS 41 17 00 §8.2.3

Retrieval smoke (top-1, voyage-4-large): shear→§4.3.2.1.2, 조합→§4.4.1.1,
휨→§4.3.2.1.1, drift→§8.2.3, 압축→§4.2.3 — all KDS, zero AISC proxy.

## How to build the index

```powershell
$env:VOYAGE_API_KEY = "vo-..."
python scripts/build_kds_rag_index.py `
  --source-dir data/kds_sample `
  --index-path data/kds_sample_index.jsonl
```

The index file (`data/kds_sample_index.jsonl`) is gitignored — it is
regenerable from the source JSONs at any time. Note the free-tier Voyage
rate limit (3 RPM without a payment method): the batched build is one
request, but per-query retrieval smokes must be spaced ~20 s apart.

## AISC 360-22 stand-ins — retired (history)

Before the KDS PDFs were acquired, `aisc_360_22_ch_f_flexure.json` and
`aisc_360_22_ch_g_shear.json` cited AISC 360-22 as **temporary references**
(`temporary_reference: true`, `replacement_target: "KDS 14 31 00 / KDS 41
31 00"`). They were retired once KDS 14 31 10 was ingested. Notes kept for
provenance:

- Steel design provisions live in **KDS 14 31** (LRFD); the building code
  KDS 41 directs steel design to KDS 14 31, and KCS 41 31 00 is a
  *construction* spec (not design) — so KDS 14 31 10 is the authoritative
  source, not KDS 41 31 00.
- The ingester (`_doc_from_json`) drops unknown fields, so `temporary_reference`
  / `replacement_target` never entered the JSONL index — proxy detection
  keys off `jurisdiction` / `standard_id` (`AISC*`) in
  `core.recommendation.explainer._retrieve_evidence`.
- The chat surfaces (summary note + audit `evidence_provenance` flag +
  collapsible disclaimer) report the AISC-proxy state from one derivation
  (`kds_compliance._aisc_proxy_standards`); with no AISC chunks left, that
  derivation is now always empty.

## Scope guarantees

- All text is a curated summary, never the verbatim clause.
- Each file ≤ ~2 KB so the smoke build hits Voyage with ≤ 10 chunks.
- No PDF, no binary, no third-party reproductions.
- "KDS-equivalent" claims are unnecessary now — the chunks ARE KDS.
