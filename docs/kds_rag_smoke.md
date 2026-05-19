# Phase 3B — KDS-RAG Voyage Smoke Runner

> **목적:** `data/kds_sample/` 의 3-파일 코퍼스로 Voyage 임베딩 → JSONL 인덱스
> → `/api/v2/recommendations/explain` end-to-end 경로가 실제 동작하는지
> 검증한다. LLM 후보 생성·정렬·model_json 변경은 절대 없다 — 본 절차는
> **retrieval-only** 스모크다.
>
> 작성: 2026-05-19 (커밋 `aa99a14` 이후)

## 사전 준비

### 1) 의존성 설치 (한 번)

```powershell
# 프로젝트 venv
& "D:\son\opensees-MCP\opensees-mcp\Scripts\python.exe" -m pip install voyageai numpy
```

`numpy` 는 코사인 계산 fast path 용. 없어도 동작하지만 큰 인덱스에서는
순수 Python loop 가 느려진다 (`_try_numpy_cosine` 의 silent fallback).

### 2) Voyage API 키 발급

[voyageai.com](https://www.voyageai.com/) 에서 키 발급 후 환경변수로
주입한다. **세션 단위 환경변수만 사용** — 키를 settings.json 이나
git-tracked 파일에 절대 적지 않는다.

```powershell
$env:VOYAGE_API_KEY = "vo-..."
```

### 3) 샘플 코퍼스 확인

```powershell
Get-ChildItem D:\son\opensees-MCP\data\kds_sample\
```

기대 출력 (3-파일):
```
aisc_360_22_ch_f_flexure.json
aisc_360_22_ch_g_shear.json
kds_41_17_00_drift.json
```

코퍼스 맵 (메타 정보, KDS 본문이 아니므로 인덱싱 경로 밖) →
`docs/kds_rag_sample_corpus.md` 참고.

## 인덱스 빌드

```powershell
$env:VOYAGE_API_KEY = "vo-..."   # 안 깔려있으면 위 3) 참고
python D:\son\opensees-MCP\scripts\build_kds_rag_index.py `
  --source-dir D:\son\opensees-MCP\data\kds_sample `
  --index-path D:\son\opensees-MCP\data\kds_sample_index.jsonl `
  --model voyage-4-large `
  -v
```

기대 출력 (한 줄 요약):
```
OK: indexed 3 chunks from 3 docs (model=voyage-4-large, dim=1024) -> ...
```

- `n_chunks` 가 3 이 아니면 chunker 가 한 문서를 쪼갠 것 (정상). drift JSON 은
  단일 paragraph 가 크므로 1 청크, AISC 둘은 각각 1~2 청크 정도 예상.
- exit 코드:
  - `0` — 정상
  - `2` — API 키 누락 또는 voyageai 미설치
  - `3` — source-dir 비었거나 .txt/.md/.json 0 개

## 서버 활성화

```powershell
$env:VOYAGE_API_KEY      = "vo-..."
$env:KDS_RAG_INDEX_PATH  = "D:\son\opensees-MCP\data\kds_sample_index.jsonl"

cd D:\son\opensees-MCP\webapp\backend
python -m uvicorn app.main_simple:app --port 8001
```

기동 후 두 환경변수가 모두 설정되어 있고 인덱스 파일이 존재하면
`get_default_kds_retriever()` 가 `VoyageKDSRetriever` 를 반환한다.
하나라도 빠지면 silent fallback → `NoopKDSRetriever` (각 응답 warning 에
`kds_rag_unavailable` 노출).

## Explain 스모크

웹 UI 의 V2 Editor 에서:

1. 해석 수행 (`/api/v2/analyze`) → 추천 후보 카드가 뜨는지 확인.
2. drift_exceeded · strength_exceeded · shear_exceeded 가 섞인 모델에서
   하나의 후보의 "왜 추천됐는지" 모달을 연다.
3. 모달 상단 RAG 배지가 **"RAG 사용"** 으로 표시되어야 한다 (Noop 일 때는
   "RAG 미사용").
4. "KDS 근거" 섹션에 1개 이상의 chunk 가 인용되어야 하며, quote 는
   `data/kds_sample/*.json` 의 text 중 일부와 일치해야 한다.

### 직접 curl 로 검증 (UI 없이)

```powershell
# 1) /api/v2/analyze 호출 후 analysis_id, candidate_id 확보
$body = @{
  analysis_id  = "<analysis_id from /analyze response>"
  candidate_id = "<candidate_id from recommendations[]>"
  language     = "ko"
  style        = "engineer_brief"
} | ConvertTo-Json

curl.exe -X POST http://localhost:8001/api/v2/recommendations/explain `
  -H "Content-Type: application/json" `
  -d $body
```

검사 포인트:
- `source.rag_used == true`
- `kds_evidence` 배열이 비어 있지 않다
- `kds_evidence[].quote` 의 일부 글자가 `data/kds_sample/*.json` 의
  text 에 실제로 들어 있다 (citation guardrail 통과 증거)
- `warnings` 에 `kds_rag_unavailable` 이 없다

## 수동 retrieval 품질 평가

세 가지 issue_type 에 대해 retrieval 이 의도된 chunk 를 top-1 으로 잡는지
**눈으로** 확인한다. `tests/test_kds_rag_retrieval_routing.py` 의 결정론적
fake embedder 회귀는 *라우팅* (make_kds_query → cosine 경로의 wiring)
까지만 보장하며, voyage-4-large 의 실제 한국어 임베딩 품질은
**여기서 수동으로** 측정해야 한다.

| 쿼리 시나리오 (issue_type / action_type) | 기대 top-1 chunk | 기대 quote 키워드 |
|------------------------------------------|------------------|--------------------|
| `drift_exceeded` + `add_lateral_resistance` | `kds_41_17_00_drift` | "허용 층간변위비", "Δ = Cd × δ_xe / IE" |
| `strength_exceeded` + `replace_section` | `aisc_360_22_ch_f_flexure` | "Mp", "LTB", "Pr/Pc + 8/9·(Mrx/..." |
| `shear_exceeded` + `replace_section` | `aisc_360_22_ch_g_shear` | "Vn = 0.6·Fy·Aw·Cv1" |

상이한 결과가 나오면 다음 의심:

1. `make_kds_query` 가 query_text 에 단면명·키워드를 안 넣음
   → `tests/test_kds_voyage_rag.py::TestMakeKdsQuery` 와 핵심
   `core.kds_rag.pipeline.make_kds_query` 를 비교.
2. Voyage-4-large 가 짧은 한국어 키워드만으로는 LTB / 전단을 강하게
   분리하지 못함 → rerank-2.5 가 자동으로 붙으므로 보통 회복되지만,
   안 되면 `top_n_dense` 를 20 → 50 으로 올려본다.
3. AISC 본문 내부에 "휨" / "전단" 키워드가 너무 가까이 공존
   → corpus 분리 (한 파일에 한 limit_state 만 두기) — 본 코퍼스는 이미
   그렇게 분리되어 있다.

## 회귀 테스트 (네트워크 없이)

```powershell
& "D:\son\opensees-MCP\opensees-mcp\Scripts\python.exe" -m pytest `
  tests\test_kds_rag.py `
  tests\test_kds_voyage_rag.py `
  tests\test_kds_rag_retrieval_routing.py `
  tests\test_recommendation_explainer.py `
  tests\test_v2_recommendations_api.py::TestExplainEndpoint `
  -q
```

- 전체 통과해야 한다 — Voyage SDK / 네트워크 호출 0 건.
- `test_kds_rag_retrieval_routing.py` 가 추가된 deterministic top-1
  *라우팅* 회귀 (drift / shear / strength). 실제 임베딩 품질은 위
  "수동 retrieval 품질 평가" 절차로 별도 측정.

## 안전 가드 (절대 깨지면 안 됨)

- Voyage 가 응답 실패해도 `/explain` 은 **deterministic explanation 만으로**
  200 을 반환한다 (`source.rag_used = false`, warnings 에 사유 기록).
- `validate_code_reference` 를 통과하지 못한 chunk 는 evidence 로 노출되지
  않는다 — quote 길이 cap 400 자, topic/limit_state 미스매치는 reject.
- `model_json` 캐시는 deepcopy 후에만 사용되며 explain endpoint 는 캐시를
  변경하지 않는다.
- LLM provider 는 아직 `NoopExplanationLLMProvider` 만 — 본 스모크에서
  실제 LLM 호출은 없다. 후속 Phase 4 에서 Anthropic / OpenAI 연결.

## AISC 임시 참조 — UI 표시 의무

`data/kds_sample/aisc_360_22_ch_*.json` 두 파일은 `temporary_reference:
true` 로 표기된 임시 참조다. 이 청크가 Explain 모달에 인용될 때, 운영
측은 다음 한국어 디스클레이머를 반드시 함께 노출해야 한다:

> 현재 강구조 근거는 KDS 원문이 아닌 AISC 360-22 임시 참조입니다.
> KDS 14 31 00 / KDS 41 31 00 원문 확보 후 교체 검증이 필요합니다.

ingester (`core.kds_rag.ingest._doc_from_json`) 가 현재 시점에는
`temporary_reference` / `replacement_*` 필드를 화이트리스트에서 제외해
JSONL 인덱스에 싣지 않으므로, UI 측은 chunk metadata 가 아닌 **소스
JSON 을 직접 읽어** 플래그를 확인해야 한다. UI 와이어링은 Phase 4
작업 (LLM provider 활성화와 함께) 으로 남는다.

`data/kds_sample/kds_41_17_00_drift.json` 은 `temporary_reference:
false` 이므로 위 디스클레이머가 필요 없다.

## 다음 작업 후보 (Phase 4)

- KDS 14 31 00 PDF 입수 후 `data/kds_sample/` 에 추가, 인덱스 재빌드 →
  AISC 인용이 KDS 인용으로 전환되는지 확인.
- PDF → 텍스트 추출 파이프라인 (PyMuPDF/pdfplumber) → `data/kds_full/` 로
  확장.
- LLM provider 연결 — prompt caching + "evidence-only citation" 가드 +
  `validate_code_reference()` post-check.
- 벡터 백엔드 마이그레이션 (sqlite-vec / Supabase pgvector) — 청크 1000+
  시 JSONL+numpy 가 비효율.

연관 메모리: `memory/recommendation_phase3.md`
