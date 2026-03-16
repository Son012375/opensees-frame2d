# KDS 파라미터 DB 구축 Agent Team 설계

> **버전:** 1.0.0
> **작성일:** 2026-02-10
> **목적:** KDS 기반 하중/계수 파라미터 DB 자동 구축 및 버전 동기화

---

## 1. 시스템 개요

### 1.1 목표

KDS(한국건설기준)에서 구조해석에 필요한 정형 데이터를 추출하여 Supabase DB로 구축:

| 데이터 유형 | 예시 | 출처 |
|-------------|------|------|
| 용도별 활하중 | 사무실 2.5 kN/m² | KDS 41 12 00 표 3.1-1 |
| 고정하중 | 콘크리트 슬래브 4.0 kN/m² | KDS 41 12 00 |
| 지역별 풍속 | 서울 26 m/s | KDS 41 12 10 |
| 지역별 지면조도 | I, II, III, IV | KDS 41 12 10 |
| 적설하중 | 지역별 기본값 | KDS 41 12 20 |
| 지진구역계수 | I, II 구역 | KDS 41 17 00 |
| 하중조합 계수 | 1.2D + 1.6L | KDS 41 10 15 |

### 1.2 핵심 요구사항

1. **출처 추적성**: 모든 수치에 `(KDS 번호, 조항, 표 번호)` 메타데이터 저장
2. **버전 관리**: KDS 개정판(예: 2022-10-11) 별 데이터 분리
3. **정규화**: 다양한 표현(사무실/업무시설/office) → 통일된 키
4. **검증**: 확신도(confidence) 기반 리뷰 리스트 분리

### 1.3 시스템 파이프라인

```
[PDF/텍스트] → [문서 수집 Agent] → [표 추출 Agent] → [정규화 Agent] → [검증 Agent] → [Supabase 적재 Agent]
                    ↓                    ↓                 ↓                ↓                  ↓
              버전 메타데이터        구조화된 표       통일된 키        리뷰 리스트         DB 저장
```

---

## 2. Agent 상세 설계

### 2.1 Agent 1: 문서 수집/버전 관리 Agent

#### 역할
PDF/텍스트로부터 KDS 문서 메타데이터 추출 및 구조화

#### 입력
```typescript
interface DocumentInput {
  file_path: string;           // PDF 또는 텍스트 파일 경로
  file_type: "pdf" | "txt";
  manual_metadata?: {          // (선택) 수동 입력 메타데이터
    code_id: string;           // 예: "KDS 41 12 00"
    version_date: string;      // 예: "2022-10-11"
  };
}
```

#### 출력
```typescript
interface DocumentOutput {
  document_id: string;         // UUID
  code_id: string;             // "KDS 41 12 00"
  code_title: string;          // "건축구조기준 하중"
  version_date: string;        // "2022-10-11"
  effective_date: string;      // 시행일
  source: "KCSC" | "PDF" | "OTHER";

  structure: {
    sections: Section[];       // 절/항 구조
    tables: TableMeta[];       // 표 목록 (표 번호, 제목, 페이지)
    figures: FigureMeta[];     // 그림 목록
  };

  raw_text: string;            // 전문 텍스트 (RAG용)
  extraction_timestamp: string;
}

interface Section {
  id: string;                  // "3.1.1"
  title: string;               // "바닥활하중"
  level: number;               // 절 깊이 (1, 2, 3)
  page: number;
}

interface TableMeta {
  table_id: string;            // "표 3.1-1"
  title: string;               // "건축물 바닥의 용도별 활하중"
  page: number;
  section_id: string;          // 소속 절 ID
}
```

#### 처리 로직
1. PDF → 텍스트 추출 (PyMuPDF, pdfplumber)
2. 정규표현식으로 KDS 번호, 버전 추출
3. 목차 구조 파싱 (절/항/표 목록)
4. 표 위치 감지 (페이지, 바운딩 박스)

#### 에러 케이스
| 에러 | 처리 |
|------|------|
| KDS 번호 추출 실패 | 수동 메타데이터 요청 |
| 버전 정보 없음 | 파일명/경로에서 추론 시도 |
| 표 감지 실패 | OCR fallback 또는 수동 표시 요청 |

---

### 2.2 Agent 2: 표/수치 추출 Agent

#### 역할
KDS 문서 내 표에서 하중/계수 값을 구조화된 형태로 추출

#### 입력
```typescript
interface TableExtractionInput {
  document_id: string;
  table_meta: TableMeta;
  extraction_mode: "auto" | "ocr" | "manual_assist";
}
```

#### 출력
```typescript
interface ExtractedTable {
  table_id: string;
  document_id: string;
  source_clause: string;       // "3.1.1"

  headers: string[];           // 열 헤더
  rows: ExtractedRow[];

  extraction_method: "pdfplumber" | "camelot" | "ocr" | "manual";
  confidence: number;          // 0.0 ~ 1.0
  extraction_issues: string[]; // 문제점 기록
}

interface ExtractedRow {
  row_index: number;
  cells: ExtractedCell[];
}

interface ExtractedCell {
  column: string;              // 헤더명
  raw_value: string;           // 원본 텍스트
  parsed_value?: number;       // 파싱된 숫자
  unit?: string;               // 단위 (kN/m², kPa 등)
  conditions?: string[];       // 조건/예외 (예: "창고는 적재물에 따라")
}
```

#### 표 유형별 처리

| 표 유형 | 추출 전략 | 예시 |
|---------|-----------|------|
| 단순 테이블 | pdfplumber 직접 추출 | 용도별 활하중표 |
| 병합 셀 테이블 | camelot + 후처리 | 풍속 지역표 |
| 이미지 테이블 | OCR (Tesseract/Vision API) | 스캔 문서 |
| 복잡한 중첩 | 수동 보조 입력 요청 | 하중조합표 |

#### 추출 품질 지표

```typescript
interface QualityMetrics {
  cell_fill_rate: number;      // 빈 셀 비율
  numeric_parse_rate: number;  // 숫자 파싱 성공률
  unit_detection_rate: number; // 단위 감지율
  overall_confidence: number;  // 종합 확신도
}
```

---

### 2.3 Agent 3: 정규화/키 매핑 Agent

#### 역할
추출된 원본 텍스트를 시스템 표준 키로 통일, 단위 변환

#### 입력
```typescript
interface NormalizationInput {
  extracted_tables: ExtractedTable[];
  mapping_config: MappingConfig;
}

interface MappingConfig {
  occupancy_aliases: Record<string, string[]>;  // 용도 매핑
  region_aliases: Record<string, string[]>;     // 지역 매핑
  unit_conversions: UnitConversion[];           // 단위 변환 규칙
}
```

#### 출력
```typescript
interface NormalizedRecord {
  record_id: string;
  record_type: RecordType;

  // 정규화된 키
  primary_key: string;         // 예: "office", "seoul", "seismic_zone_1"
  display_name_ko: string;     // "사무실"
  display_name_en: string;     // "Office"

  // 값
  value: number;
  unit: string;                // 표준 단위 (kN/m², m/s, kPa)
  original_unit: string;       // 원본 단위

  // 출처
  source: {
    document_id: string;
    code_id: string;           // "KDS 41 12 00"
    code_version: string;      // "2022-10-11"
    clause_id: string;         // "3.1.1"
    table_id: string;          // "표 3.1-1"
  };

  // 조건
  conditions: Condition[];
  notes: string[];

  // 메타
  confidence: number;
  mapping_method: "exact" | "alias" | "fuzzy" | "manual";
  needs_review: boolean;
}

type RecordType =
  | "live_load"
  | "dead_load"
  | "wind_speed"
  | "snow_load"
  | "seismic_coefficient"
  | "load_combination_factor";

interface Condition {
  type: "range" | "exception" | "override";
  description: string;
  applies_when: string;
}
```

#### 매핑 테이블 (예시)

```typescript
const occupancy_aliases = {
  "office": ["사무실", "업무시설", "사무소", "오피스", "Office", "업무용"],
  "residential": ["주거", "주택", "아파트", "공동주택", "Residential"],
  "retail": ["소매점", "상점", "판매시설", "매장", "Retail"],
  "assembly": ["집회", "집회시설", "공연장", "Assembly"],
  "storage": ["창고", "창고시설", "Storage", "저장"],
  "parking": ["주차장", "주차시설", "Parking"],
};

const unit_conversions = [
  { from: "kgf/m²", to: "kN/m²", factor: 0.00981 },
  { from: "kPa", to: "kN/m²", factor: 1.0 },
  { from: "tf/m²", to: "kN/m²", factor: 9.81 },
];
```

#### Fuzzy Matching 전략

```typescript
interface FuzzyMatchResult {
  input: string;
  matched_key: string;
  similarity: number;          // 0.0 ~ 1.0
  method: "levenshtein" | "jaro_winkler" | "embedding";
}
```

- 유사도 > 0.9: 자동 매핑
- 유사도 0.7~0.9: 후보 제시 + 확인 요청
- 유사도 < 0.7: 리뷰 리스트로 분류

---

### 2.4 Agent 4: 검증 Agent

#### 역할
추출/정규화된 데이터의 품질 검증, 충돌 감지, 리뷰 리스트 생성

#### 입력
```typescript
interface ValidationInput {
  records: NormalizedRecord[];
  existing_db?: ExistingDatabase;  // 기존 DB (있으면)
  validation_rules: ValidationRule[];
}
```

#### 출력
```typescript
interface ValidationResult {
  total_records: number;
  passed: number;
  failed: number;
  needs_review: number;

  issues: ValidationIssue[];
  review_list: ReviewItem[];
  statistics: ValidationStats;
}

interface ValidationIssue {
  issue_id: string;
  severity: "error" | "warning" | "info";
  issue_type: IssueType;
  record_ids: string[];
  description: string;
  suggested_action: string;
}

type IssueType =
  | "missing_value"          // 필수 값 누락
  | "out_of_range"           // 값 범위 초과
  | "duplicate"              // 중복 레코드
  | "version_conflict"       // 버전 간 충돌
  | "cross_reference_mismatch" // 다른 KDS 문서와 불일치
  | "low_confidence"         // 낮은 확신도
  | "unit_mismatch";         // 단위 불일치

interface ReviewItem {
  record_id: string;
  record: NormalizedRecord;
  reason: string;
  priority: "high" | "medium" | "low";
  suggested_values?: any[];
}
```

#### 검증 규칙

```typescript
const validation_rules: ValidationRule[] = [
  // 범위 검증
  { type: "live_load", min: 0.5, max: 20.0, unit: "kN/m²" },
  { type: "wind_speed", min: 20.0, max: 50.0, unit: "m/s" },
  { type: "seismic_coefficient", min: 0.05, max: 0.20, unit: null },

  // 필수 필드
  { required: ["primary_key", "value", "unit", "source.code_id", "source.clause_id"] },

  // Cross-reference (KDS 간 정합성)
  { cross_ref: "KDS 41 12 00 vs KDS 41 17 00", check: "seismic_zone_mapping" },
];
```

#### 검증 통계

```typescript
interface ValidationStats {
  by_type: Record<RecordType, { total: number; passed: number; failed: number }>;
  by_confidence: {
    high: number;    // > 0.9
    medium: number;  // 0.7 ~ 0.9
    low: number;     // < 0.7
  };
  by_source: Record<string, number>;
}
```

---

### 2.5 Agent 5: Supabase 적재 Agent

#### 역할
검증된 데이터를 Supabase 정형 테이블에 upsert, 이력 관리

#### 입력
```typescript
interface LoadInput {
  records: NormalizedRecord[];
  validation_result: ValidationResult;
  load_mode: "dry_run" | "upsert" | "insert_only";
  options: LoadOptions;
}

interface LoadOptions {
  skip_failed: boolean;        // 실패 레코드 스킵
  skip_needs_review: boolean;  // 리뷰 필요 레코드 스킵
  create_backup: boolean;      // 기존 데이터 백업
}
```

#### 출력
```typescript
interface LoadResult {
  mode: "dry_run" | "upsert" | "insert_only";
  summary: {
    total: number;
    inserted: number;
    updated: number;
    skipped: number;
    failed: number;
  };

  affected_tables: string[];
  backup_id?: string;          // 백업 생성 시

  errors: LoadError[];
  dry_run_preview?: DryRunPreview;
}

interface DryRunPreview {
  inserts: { table: string; count: number; sample: any[] }[];
  updates: { table: string; count: number; changes: any[] }[];
}
```

#### 적재 전략

```typescript
// Upsert 키 정의
const upsert_keys = {
  load_params: ["param_type", "primary_key", "code_version"],
  code_versions: ["code_id", "version_date"],
  source_clauses: ["code_id", "clause_id"],
};

// 이력 관리
const history_tracking = {
  enabled: true,
  table_suffix: "_history",
  track_fields: ["value", "unit", "conditions"],
};
```

---

## 3. Agent 간 통신

### 3.1 메시지 구조

```typescript
interface AgentMessage {
  from: AgentId;
  to: AgentId;
  timestamp: string;
  message_type: "request" | "response" | "error" | "progress";
  payload: any;
  correlation_id: string;      // 요청-응답 매칭
}

type AgentId =
  | "document_collector"
  | "table_extractor"
  | "normalizer"
  | "validator"
  | "loader";
```

### 3.2 실행 모드

| 모드 | 설명 |
|------|------|
| **Sequential** | Agent 1 → 2 → 3 → 4 → 5 순차 실행 |
| **Parallel** | 여러 문서를 병렬로 처리, 각 문서 내에서는 순차 |
| **Interactive** | 리뷰 필요 시 사용자 개입 요청 |

---

## 4. 구성 옵션

```typescript
interface PipelineConfig {
  // 실행 모드
  execution_mode: "sequential" | "parallel" | "interactive";

  // 병렬 처리
  max_parallel_documents: number;

  // 확신도 임계값
  confidence_threshold: {
    auto_accept: number;       // 이상이면 자동 승인 (기본 0.9)
    needs_review: number;      // 이하면 리뷰 필요 (기본 0.7)
  };

  // 적재 옵션
  load_options: LoadOptions;

  // 알림
  notifications: {
    on_complete: boolean;
    on_error: boolean;
    on_review_needed: boolean;
  };
}
```

---

## 5. 모니터링 및 로깅

### 5.1 로그 레벨

| 레벨 | 용도 |
|------|------|
| DEBUG | 상세 추출 과정 |
| INFO | Agent 시작/종료, 처리 건수 |
| WARNING | 낮은 확신도, 매핑 실패 |
| ERROR | 추출 실패, DB 오류 |

### 5.2 메트릭

```typescript
interface PipelineMetrics {
  documents_processed: number;
  tables_extracted: number;
  records_normalized: number;
  records_loaded: number;

  avg_confidence: number;
  avg_extraction_time_ms: number;

  error_rate: number;
  review_rate: number;
}
```

---

## 6. 다음 문서

- [02_DATA_FLOW.md](./02_DATA_FLOW.md) - Agent 간 데이터 흐름 상세
- [03_SUPABASE_SCHEMA.md](./03_SUPABASE_SCHEMA.md) - DB 스키마 설계
- [04_PROMPT_TEMPLATES.md](./04_PROMPT_TEMPLATES.md) - 각 Agent 프롬프트 템플릿
- [05_IMPLEMENTATION_PLAN.md](./05_IMPLEMENTATION_PLAN.md) - 구현 계획
