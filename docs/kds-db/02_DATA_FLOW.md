# KDS 파라미터 DB - Agent 간 데이터 흐름

> **버전:** 1.0.0
> **작성일:** 2026-02-10

---

## 1. 전체 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              KDS Parameter DB Pipeline                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐
   │   PDF 입력    │  C:\Users\youm\Downloads\KDS 41 12 00_건축물 설계하중.pdf
   └──────┬───────┘
          │
          ▼
┌─────────────────────┐
│  1. Document Agent  │──────────────────────────────────────────────┐
│                     │                                              │
│  - PDF 파싱         │     DocumentOutput                           │
│  - 메타데이터 추출   │     ├── code_id: "KDS 41 12 00"              │
│  - 목차/표 구조화    │     ├── version_date: "2022-10-11"           │
│                     │     ├── sections: [{id, title, level}, ...]  │
└────────┬────────────┘     └── tables: [{table_id, title, page}, ...]
         │
         │ DocumentOutput + TableMeta[]
         ▼
┌─────────────────────┐
│   2. Table Agent    │──────────────────────────────────────────────┐
│                     │                                              │
│  - 표 추출          │     ExtractedTable[]                         │
│  - 셀 파싱          │     ├── table_id: "표 3.1-1"                  │
│  - 단위 감지        │     ├── headers: ["용도", "활하중", ...]      │
│  - 확신도 계산      │     ├── rows: [{cells: [...]}]               │
└────────┬────────────┘     └── confidence: 0.95                     │
         │
         │ ExtractedTable[]
         ▼
┌─────────────────────┐
│  3. Normalize Agent │──────────────────────────────────────────────┐
│                     │                                              │
│  - 키 매핑          │     NormalizedRecord[]                       │
│  - 단위 변환        │     ├── primary_key: "office"                │
│  - 다국어 표기      │     ├── value: 2.5                           │
│                     │     ├── unit: "kN/m²"                        │
└────────┬────────────┘     └── source: {code_id, clause_id, ...}    │
         │
         │ NormalizedRecord[]
         ▼
┌─────────────────────┐
│  4. Validate Agent  │──────────────────────────────────────────────┐
│                     │                                              │
│  - 범위 검증        │     ValidationResult                         │
│  - 중복 탐지        │     ├── passed: 45                           │
│  - 충돌 감지        │     ├── failed: 2                            │
│  - 리뷰 리스트 생성  │     ├── needs_review: 5                      │
└────────┬────────────┘     └── review_list: [...]                   │
         │
         │ Validated Records + ValidationResult
         ▼
┌─────────────────────┐
│   5. Loader Agent   │──────────────────────────────────────────────┐
│                     │                                              │
│  - Dry-run 미리보기 │     LoadResult                               │
│  - Upsert 실행      │     ├── inserted: 40                         │
│  - 이력 추적        │     ├── updated: 5                           │
│                     │     └── skipped: 7                           │
└─────────────────────┘
         │
         ▼
   ┌──────────────┐
   │   Supabase   │
   │   Database   │
   └──────────────┘
```

---

## 2. 단계별 데이터 변환

### 2.1 Stage 1: PDF → DocumentOutput

```
입력: PDF 바이너리
      ↓
처리: PyMuPDF/pdfplumber 텍스트 추출
      정규표현식 메타데이터 파싱
      목차 구조 분석
      ↓
출력: {
        document_id: "uuid-xxx",
        code_id: "KDS 41 12 00",
        version_date: "2022-10-11",
        structure: {
          sections: [...],
          tables: [
            { table_id: "표 3.1-1", title: "건축물 바닥의 용도별 활하중", page: 12 },
            { table_id: "표 3.1-2", title: "특수용도 활하중", page: 14 },
            ...
          ]
        }
      }
```

### 2.2 Stage 2: TableMeta → ExtractedTable

```
입력: { table_id: "표 3.1-1", page: 12 }
      ↓
처리: 표 영역 추출
      셀 분리 및 파싱
      단위 감지 (kN/m², kPa)
      조건/예외 텍스트 분리
      ↓
출력: {
        table_id: "표 3.1-1",
        headers: ["용도", "등분포활하중 (kN/m²)", "집중활하중 (kN)"],
        rows: [
          {
            cells: [
              { column: "용도", raw_value: "사무실" },
              { column: "등분포활하중 (kN/m²)", raw_value: "2.5", parsed_value: 2.5, unit: "kN/m²" },
              { column: "집중활하중 (kN)", raw_value: "4.5", parsed_value: 4.5, unit: "kN" }
            ]
          },
          ...
        ],
        confidence: 0.95
      }
```

### 2.3 Stage 3: ExtractedRow → NormalizedRecord

```
입력: { column: "용도", raw_value: "사무실" }
       { column: "등분포활하중", raw_value: "2.5", unit: "kN/m²" }
      ↓
처리: 용도 키 매핑 ("사무실" → "office")
      단위 표준화
      출처 메타데이터 첨부
      ↓
출력: {
        record_id: "uuid-yyy",
        record_type: "live_load",
        primary_key: "office",
        display_name_ko: "사무실",
        display_name_en: "Office",
        value: 2.5,
        unit: "kN/m²",
        load_type: "distributed",
        source: {
          code_id: "KDS 41 12 00",
          code_version: "2022-10-11",
          clause_id: "3.1.1",
          table_id: "표 3.1-1"
        },
        confidence: 0.95,
        mapping_method: "alias"
      }
```

### 2.4 Stage 4: NormalizedRecord → ValidationResult

```
입력: NormalizedRecord[]
      ↓
처리: 범위 검증 (0.5 ≤ live_load ≤ 20.0)
      중복 검사 (같은 키 + 같은 버전)
      Cross-reference (KDS 간 정합성)
      ↓
출력: {
        passed: 45,
        failed: 2,
        needs_review: 5,
        issues: [
          { severity: "warning", issue_type: "low_confidence", record_ids: ["uuid-1", "uuid-2"] },
          { severity: "error", issue_type: "out_of_range", description: "값 25.0은 범위 초과" }
        ],
        review_list: [
          { record_id: "uuid-3", reason: "매핑 확신도 0.75", priority: "medium" }
        ]
      }
```

### 2.5 Stage 5: ValidatedRecords → Supabase

```
입력: Validated NormalizedRecord[]
      mode: "dry_run" | "upsert"
      ↓
처리: (dry_run) 미리보기 생성
      (upsert) 기존 레코드와 비교 → INSERT/UPDATE
      이력 테이블 기록
      ↓
출력: {
        mode: "upsert",
        summary: { inserted: 40, updated: 5, skipped: 7 },
        affected_tables: ["load_params", "code_versions", "source_clauses"]
      }
```

---

## 3. 에러 흐름

### 3.1 에러 전파 경로

```
Agent 1 에러 → 파이프라인 중단 (문서 메타 없이 진행 불가)
Agent 2 에러 → 해당 표만 스킵, 다른 표는 계속 처리
Agent 3 에러 → 해당 레코드 리뷰 리스트로 분류
Agent 4 에러 → 검증 실패 레코드만 분류
Agent 5 에러 → 롤백 후 재시도 or 부분 커밋
```

### 3.2 복구 전략

| 에러 유형 | 복구 전략 |
|-----------|-----------|
| PDF 파싱 실패 | 다른 라이브러리 시도 (PyMuPDF → pdfplumber) |
| 표 추출 실패 | OCR fallback 또는 수동 입력 요청 |
| 매핑 실패 | 리뷰 리스트에 추가, 수동 매핑 요청 |
| 검증 실패 | 실패 레코드 분리, 통과 레코드만 적재 |
| DB 적재 실패 | 트랜잭션 롤백, 에러 로깅 |

---

## 4. 병렬 처리 흐름

### 4.1 문서 병렬 처리

```
┌────────────┐    ┌────────────┐    ┌────────────┐
│ KDS 41 12  │    │ KDS 41 17  │    │ KDS 41 10  │
│    00      │    │    00      │    │    15      │
└─────┬──────┘    └─────┬──────┘    └─────┬──────┘
      │                 │                 │
      ▼                 ▼                 ▼
┌─────────────────────────────────────────────────┐
│            Parallel Document Processing          │
│  Worker 1    │    Worker 2    │    Worker 3     │
└─────────────────────────────────────────────────┘
      │                 │                 │
      └────────────────┬┘─────────────────┘
                       ▼
              ┌────────────────┐
              │  Merge Results │
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │   Validation   │  ← 전체 Cross-reference 검증
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │     Loader     │
              └────────────────┘
```

### 4.2 표 병렬 추출

한 문서 내에서 여러 표를 병렬 추출:

```python
async def extract_tables_parallel(document: DocumentOutput) -> list[ExtractedTable]:
    tasks = [
        extract_table(document.document_id, table_meta)
        for table_meta in document.structure.tables
    ]
    return await asyncio.gather(*tasks)
```

---

## 5. 상태 관리

### 5.1 파이프라인 상태

```typescript
interface PipelineState {
  pipeline_id: string;
  status: "pending" | "running" | "paused" | "completed" | "failed";

  current_stage: 1 | 2 | 3 | 4 | 5;
  current_agent: AgentId;

  progress: {
    documents: { total: number; processed: number };
    tables: { total: number; processed: number };
    records: { total: number; processed: number };
  };

  checkpoints: Checkpoint[];
  errors: PipelineError[];
}

interface Checkpoint {
  stage: number;
  timestamp: string;
  data_snapshot_id: string;   // 중간 결과 저장
}
```

### 5.2 재시작 (Resume)

```
파이프라인 중단 시:
1. 마지막 Checkpoint 로드
2. 해당 Stage부터 재시작
3. 이미 처리된 데이터 스킵
```

---

## 6. 로깅 및 추적

### 6.1 로그 구조

```typescript
interface PipelineLog {
  timestamp: string;
  pipeline_id: string;
  agent: AgentId;
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  message: string;
  context: {
    document_id?: string;
    table_id?: string;
    record_id?: string;
  };
}
```

### 6.2 추적 ID

모든 데이터에는 Lineage 추적 가능:

```
Record UUID → 추출된 Table → 원본 Document → PDF 파일
```

---

## 7. 외부 연동

### 7.1 입력 소스

| 소스 | 지원 | 비고 |
|------|------|------|
| 로컬 PDF | ✅ | 기본 |
| URL (KCSC 뷰어) | ⚠️ | 동적 페이지, 제한적 |
| Google Drive | 🔜 | 향후 |

### 7.2 출력 대상

| 대상 | 용도 |
|------|------|
| Supabase (정형) | 하중 파라미터 DB |
| Vector DB (선택) | RAG용 조항 임베딩 |
| Export (JSON/CSV) | 검토용 |

---

## 8. 관련 문서

- [01_AGENT_ARCHITECTURE.md](./01_AGENT_ARCHITECTURE.md) - Agent 상세 스펙
- [03_SUPABASE_SCHEMA.md](./03_SUPABASE_SCHEMA.md) - DB 스키마
