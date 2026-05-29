# KDS-RAG Retrieval Benchmark (R1)

A lightweight golden-set evaluation of the KDS-RAG retrieval path, added
after the KDS 14 31 10 corpus was ingested (the earlier check was a top-1
eyeball wiring smoke). Two parts:

| Part | File | Retriever | Runs in CI? | Purpose |
|------|------|-----------|-------------|---------|
| Deterministic regression | `tests/test_kds_rag_golden.py` | InMemory over real `data/kds_sample` | ✅ yes | Lock corpus tags + issue_type→topic routing; no API |
| Live quality benchmark | `scripts/kds_rag_benchmark.py` | VoyageKDSRetriever (real index) | ❌ manual | Real recall@k / precision@1 / MRR numbers |

Golden set: `data/kds_eval/golden_set.json` — one case per member
design-check `issue_type`, mapping a scenario to its expected KDS clause(s).
Both parts read it, so adding a case updates both. `data/kds_eval/` is NOT
under the index source dir (`data/kds_sample`), so the golden set never
pollutes the corpus.

## Metrics

- **recall@1 / recall@3** — expected clause appears at rank 1 / within top-3.
- **precision@1** — top-1 is an acceptable clause.
- **MRR** — mean reciprocal rank of the first acceptable hit.
- A case lists `expected_clauses` (acceptable set); a hit = any of them
  from the `expected_standard`.

## How to run

Deterministic (CI-safe):

```powershell
python -m pytest tests/test_kds_rag_golden.py -q
```

Live (needs VOYAGE_API_KEY + KDS_RAG_INDEX_PATH + a built index):

```powershell
python scripts/kds_rag_benchmark.py --top-k 3 --sleep 42
```

> Voyage free tier (no payment method) = ~3 RPM, and each retrieval is **2
> calls** (embed_query + rerank-2.5), so use `--sleep 42` (≈1 retrieval per
> 42 s) to avoid 429s. At 22 s some queries get rate-limited and show as
> `RETRIEVE FAILED` (counted as a miss — re-run that case).

## Baseline (2026-05-29, voyage-4-large + rerank-2.5, 6-chunk corpus)

Deterministic (InMemory): **recall@1 = 5/5, recall@3 = 5/5, precision@1 = 5/5.**

Live (Voyage), after the axial fix below:

| case | issue_type | top-1 | rank of expected | hit@1 | hit@3 |
|------|------------|-------|------------------|-------|-------|
| shear | shear_exceeded | 4.3.2.1.2 | 1 | ✅ | ✅ |
| flexure | flexure_exceeded | 4.3.2.1.1 | 1 | ✅ | ✅ |
| interaction | strength_exceeded | 4.4.1.1 | 1 | ✅ | ✅ |
| axial | axial_exceeded | 4.2.3 | 1 | ✅ | ✅ |
| drift | drift_exceeded | 8.2.3 (KDS 41 17 00) | 1 | ✅ | ✅ |

**recall@1 = 5/5 (100%), recall@3 = 5/5 (100%), precision@1 = 5/5 (100%),
MRR = 1.000. AISC temporary-reference proxy: 0/5 (fully eliminated).**

### Finding + fix: axial precision@1 (resolved)

*Initial run:* the axial (compression) scenario ranked the **interaction
chunk §4.4.1.1 at #1**, above compression §4.2.3 (#2). The interaction
clause text is axial-heavy (압축력+휨 상관식), so Voyage embedded it close
to an axial query. (The deterministic InMemory test always ranked §4.2.3
first via topic=member_axial exact match — so this was a Voyage-embedding
effect, not a tagging error.)

*Fix:* added compression-buckling discriminators **세장비 / 유효좌굴길이**
to the `axial_exceeded` query seed (`ISSUE_TYPE_KEYWORDS`, pipeline.py).
These appear in the §4.2 compression chunk but NOT the §4.4.1.1 interaction
chunk, so they pull the axial query embedding toward the pure-compression
clause. Query-side only — no chunk edit, no index rebuild. Result: axial
§4.2.3 → rank 1, interaction → rank 3. Tradeoff: the axial seed now leans
compression (the dominant axial-NG case for columns); a tension-only query
still finds §4.1.3 within top-3. Splitting compression/tension into
separate buckets is deferred (the chat tool only has |ratio_axial|, no
sign) until corpus expansion warrants it.
