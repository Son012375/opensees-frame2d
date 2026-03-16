# KDS 파라미터 DB - Supabase 스키마 설계

> **버전:** 1.0.0
> **작성일:** 2026-02-10
> **DB:** Supabase (PostgreSQL)

---

## 1. 스키마 개요

### 1.1 테이블 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KDS Parameter DB Schema                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  code_documents  │──┬──▶│  code_versions   │◀──┬──│  source_clauses  │
│                  │  │   │                  │   │  │                  │
│  - PDF 문서 정보  │  │   │  - 버전별 메타    │   │  │  - 조항 정보      │
└──────────────────┘  │   └──────────────────┘   │  └──────────────────┘
                      │            │             │            │
                      │            ▼             │            ▼
                      │   ┌──────────────────┐   │   ┌──────────────────┐
                      │   │  source_tables   │───┘   │  clause_text     │
                      │   │                  │       │  (RAG용 청킹)     │
                      │   │  - 표 메타데이터  │       └──────────────────┘
                      │   └────────┬─────────┘
                      │            │
                      ▼            ▼
              ┌───────────────────────────────┐
              │         load_params           │
              │                               │
              │  - 하중 파라미터 (핵심 테이블)  │
              │  - 활하중, 고정하중, 풍속 등    │
              └───────────────────────────────┘
                               │
                               ▼
              ┌───────────────────────────────┐
              │      load_params_history      │
              │                               │
              │  - 변경 이력 추적              │
              └───────────────────────────────┘

              ┌───────────────────────────────┐
              │       occupancy_mapping       │
              │                               │
              │  - 용도 키 매핑 (alias)        │
              └───────────────────────────────┘

              ┌───────────────────────────────┐
              │       region_mapping          │
              │                               │
              │  - 지역 키 매핑                │
              └───────────────────────────────┘

              ┌───────────────────────────────┐
              │     load_combinations         │
              │                               │
              │  - 하중조합 계수               │
              └───────────────────────────────┘
```

### 1.2 네이밍 규칙

| 규칙 | 예시 |
|------|------|
| 테이블명: snake_case | `load_params`, `code_versions` |
| 컬럼명: snake_case | `primary_key`, `code_version` |
| PK: `id` (UUID) | `id uuid primary key default gen_random_uuid()` |
| FK: `{table}_id` | `document_id`, `version_id` |
| 타임스탬프: `_at` 접미사 | `created_at`, `updated_at` |

---

## 2. 테이블 정의

### 2.1 code_documents (문서 정보)

```sql
CREATE TABLE code_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 문서 식별
    code_id VARCHAR(20) NOT NULL,          -- "KDS 41 12 00"
    code_title VARCHAR(200) NOT NULL,      -- "건축구조기준 하중"
    code_category VARCHAR(50),             -- "structural", "architectural", "mep"

    -- 파일 정보
    file_path VARCHAR(500),                -- 원본 PDF 경로
    file_hash VARCHAR(64),                 -- SHA-256 해시 (중복 방지)

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(code_id)
);

-- 인덱스
CREATE INDEX idx_code_documents_code_id ON code_documents(code_id);

COMMENT ON TABLE code_documents IS 'KDS 문서 기본 정보';
COMMENT ON COLUMN code_documents.code_id IS 'KDS 번호 (예: KDS 41 12 00)';
```

### 2.2 code_versions (버전 관리)

```sql
CREATE TABLE code_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES code_documents(id) ON DELETE CASCADE,

    -- 버전 정보
    version_date DATE NOT NULL,            -- 개정일 (2022-10-11)
    effective_date DATE,                   -- 시행일
    version_label VARCHAR(50),             -- "2022년판", "제1차 개정"

    -- 상태
    is_current BOOLEAN DEFAULT FALSE,      -- 현행 버전 여부
    is_deprecated BOOLEAN DEFAULT FALSE,   -- 폐지 여부

    -- 출처
    source_type VARCHAR(20) DEFAULT 'pdf', -- "pdf", "kcsc", "manual"
    source_url VARCHAR(500),               -- KCSC 뷰어 URL (있으면)

    -- 추출 메타
    extraction_date TIMESTAMPTZ,
    extraction_method VARCHAR(50),         -- "pymupdf", "pdfplumber", "manual"
    extraction_confidence DECIMAL(3,2),    -- 0.00 ~ 1.00

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(document_id, version_date)
);

-- 인덱스
CREATE INDEX idx_code_versions_document ON code_versions(document_id);
CREATE INDEX idx_code_versions_current ON code_versions(is_current) WHERE is_current = TRUE;

COMMENT ON TABLE code_versions IS 'KDS 문서 버전별 정보';
COMMENT ON COLUMN code_versions.is_current IS '현행 버전 플래그 (문서당 1개만 TRUE)';
```

### 2.3 source_clauses (조항 정보)

```sql
CREATE TABLE source_clauses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID REFERENCES code_versions(id) ON DELETE CASCADE,

    -- 조항 식별
    clause_id VARCHAR(20) NOT NULL,        -- "3.1.1"
    clause_title VARCHAR(200),             -- "바닥활하중"
    clause_level INTEGER DEFAULT 1,        -- 절 깊이 (1, 2, 3)

    -- 위치
    page_start INTEGER,
    page_end INTEGER,

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(version_id, clause_id)
);

-- 인덱스
CREATE INDEX idx_source_clauses_version ON source_clauses(version_id);

COMMENT ON TABLE source_clauses IS 'KDS 문서 내 조항(절/항) 정보';
```

### 2.4 source_tables (표 정보)

```sql
CREATE TABLE source_tables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clause_id UUID REFERENCES source_clauses(id) ON DELETE CASCADE,

    -- 표 식별
    table_id VARCHAR(30) NOT NULL,         -- "표 3.1-1"
    table_title VARCHAR(200),              -- "건축물 바닥의 용도별 활하중"

    -- 위치
    page_number INTEGER,

    -- 추출 메타
    extraction_method VARCHAR(30),         -- "pdfplumber", "camelot", "ocr"
    extraction_confidence DECIMAL(3,2),
    row_count INTEGER,
    column_count INTEGER,

    -- 원본 데이터 (JSON)
    raw_headers JSONB,                     -- ["용도", "등분포활하중", ...]
    raw_data JSONB,                        -- [[row1], [row2], ...]

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(clause_id, table_id)
);

-- 인덱스
CREATE INDEX idx_source_tables_clause ON source_tables(clause_id);

COMMENT ON TABLE source_tables IS 'KDS 문서 내 표 메타데이터 및 원본 데이터';
```

### 2.5 load_params (하중 파라미터 - 핵심)

```sql
CREATE TABLE load_params (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 분류
    param_type VARCHAR(30) NOT NULL,       -- 'live_load', 'dead_load', 'wind_speed', 'snow_load', 'seismic'
    param_subtype VARCHAR(30),             -- 'distributed', 'concentrated', 'basic', 'design'

    -- 키
    primary_key VARCHAR(50) NOT NULL,      -- 'office', 'seoul', 'zone_1'
    secondary_key VARCHAR(50),             -- 추가 조건 키 (예: 'heavy_storage')

    -- 표시명
    display_name_ko VARCHAR(100) NOT NULL, -- "사무실"
    display_name_en VARCHAR(100),          -- "Office"

    -- 값
    value DECIMAL(10,4) NOT NULL,          -- 2.5
    value_min DECIMAL(10,4),               -- 범위가 있는 경우 최소값
    value_max DECIMAL(10,4),               -- 범위가 있는 경우 최대값
    unit VARCHAR(20) NOT NULL,             -- "kN/m²"

    -- 조건
    conditions JSONB DEFAULT '[]',         -- [{"type": "range", "description": "..."}]
    notes TEXT,                            -- 추가 설명

    -- 출처 (FK)
    source_table_id UUID REFERENCES source_tables(id),

    -- 출처 (직접 참조용 - 조회 성능)
    code_id VARCHAR(20) NOT NULL,          -- "KDS 41 12 00"
    code_version DATE NOT NULL,            -- 2022-10-11
    clause_id VARCHAR(20),                 -- "3.1.1"
    table_id VARCHAR(30),                  -- "표 3.1-1"

    -- 품질
    confidence DECIMAL(3,2) DEFAULT 1.00,  -- 추출 확신도
    needs_review BOOLEAN DEFAULT FALSE,    -- 검토 필요 플래그
    review_note TEXT,                      -- 검토 메모

    -- 상태
    is_active BOOLEAN DEFAULT TRUE,        -- 활성화 여부

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),               -- 'agent_pipeline', 'manual'

    -- 복합 유니크 (같은 버전에서 같은 파라미터는 하나)
    UNIQUE(param_type, primary_key, secondary_key, code_id, code_version)
);

-- 인덱스
CREATE INDEX idx_load_params_type ON load_params(param_type);
CREATE INDEX idx_load_params_key ON load_params(primary_key);
CREATE INDEX idx_load_params_code ON load_params(code_id, code_version);
CREATE INDEX idx_load_params_review ON load_params(needs_review) WHERE needs_review = TRUE;
CREATE INDEX idx_load_params_active ON load_params(is_active) WHERE is_active = TRUE;

-- GIN 인덱스 (조건 검색용)
CREATE INDEX idx_load_params_conditions ON load_params USING GIN (conditions);

COMMENT ON TABLE load_params IS '하중 파라미터 메인 테이블';
COMMENT ON COLUMN load_params.param_type IS 'live_load, dead_load, wind_speed, snow_load, seismic, combination_factor';
COMMENT ON COLUMN load_params.primary_key IS '정규화된 키 (예: office, residential, seoul)';
COMMENT ON COLUMN load_params.confidence IS '추출 확신도 (0.00 ~ 1.00)';
```

### 2.6 load_params_history (변경 이력)

```sql
CREATE TABLE load_params_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    param_id UUID NOT NULL,                -- load_params.id (FK 아님, 삭제된 것도 추적)

    -- 변경 전 값
    old_value DECIMAL(10,4),
    old_unit VARCHAR(20),
    old_conditions JSONB,

    -- 변경 후 값
    new_value DECIMAL(10,4),
    new_unit VARCHAR(20),
    new_conditions JSONB,

    -- 변경 유형
    change_type VARCHAR(20) NOT NULL,      -- 'insert', 'update', 'delete'
    change_reason TEXT,                    -- 변경 사유

    -- 메타
    changed_at TIMESTAMPTZ DEFAULT NOW(),
    changed_by VARCHAR(100)
);

-- 인덱스
CREATE INDEX idx_history_param ON load_params_history(param_id);
CREATE INDEX idx_history_date ON load_params_history(changed_at);

COMMENT ON TABLE load_params_history IS '하중 파라미터 변경 이력';
```

### 2.7 occupancy_mapping (용도 매핑)

```sql
CREATE TABLE occupancy_mapping (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 정규화된 키
    canonical_key VARCHAR(50) NOT NULL,    -- "office"
    display_name_ko VARCHAR(100) NOT NULL, -- "사무실"
    display_name_en VARCHAR(100),          -- "Office"

    -- 별칭 (여러 표현)
    aliases JSONB DEFAULT '[]',            -- ["업무시설", "사무소", "오피스", "업무용"]

    -- 분류
    category VARCHAR(30),                  -- "commercial", "residential", "industrial"

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(canonical_key)
);

-- 인덱스
CREATE INDEX idx_occupancy_aliases ON occupancy_mapping USING GIN (aliases);

-- 초기 데이터
INSERT INTO occupancy_mapping (canonical_key, display_name_ko, display_name_en, aliases, category) VALUES
('office', '사무실', 'Office', '["업무시설", "사무소", "오피스", "업무용"]', 'commercial'),
('residential', '주거', 'Residential', '["주택", "아파트", "공동주택", "단독주택"]', 'residential'),
('retail', '소매점', 'Retail', '["상점", "판매시설", "매장", "점포"]', 'commercial'),
('assembly', '집회', 'Assembly', '["집회시설", "공연장", "극장", "강당"]', 'public'),
('storage', '창고', 'Storage', '["창고시설", "저장", "물류"]', 'industrial'),
('parking', '주차장', 'Parking', '["주차시설", "주차", "차고"]', 'utility'),
('hospital', '병원', 'Hospital', '["의료시설", "병실", "진료"]', 'healthcare'),
('school', '학교', 'School', '["교육시설", "학원", "강의실"]', 'education'),
('restaurant', '식당', 'Restaurant', '["음식점", "식음료", "주방"]', 'commercial'),
('library', '도서관', 'Library', '["열람실", "서고"]', 'education');

COMMENT ON TABLE occupancy_mapping IS '용도 정규화 매핑 테이블';
```

### 2.8 region_mapping (지역 매핑)

```sql
CREATE TABLE region_mapping (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 정규화된 키
    canonical_key VARCHAR(50) NOT NULL,    -- "seoul"
    display_name_ko VARCHAR(100) NOT NULL, -- "서울특별시"
    display_name_en VARCHAR(100),          -- "Seoul"

    -- 별칭
    aliases JSONB DEFAULT '[]',            -- ["서울", "서울시"]

    -- 지역 정보
    region_type VARCHAR(30),               -- "metropolitan", "province", "city"
    parent_region VARCHAR(50),             -- 상위 지역

    -- 구역 정보 (풍속, 적설, 지진)
    wind_zone VARCHAR(10),                 -- "I", "II", "III"
    snow_zone VARCHAR(10),                 -- "I", "II", "III"
    seismic_zone VARCHAR(10),              -- "I", "II"

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(canonical_key)
);

-- 인덱스
CREATE INDEX idx_region_aliases ON region_mapping USING GIN (aliases);

COMMENT ON TABLE region_mapping IS '지역 정규화 매핑 테이블';
```

### 2.9 load_combinations (하중조합)

```sql
CREATE TABLE load_combinations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 조합 식별
    combination_id VARCHAR(50) NOT NULL,   -- "LC1", "1.2D+1.6L"
    combination_name VARCHAR(100),         -- "강도설계조합 1"
    design_method VARCHAR(30),             -- "USD", "ASD", "LSD"

    -- 계수
    factors JSONB NOT NULL,                -- {"D": 1.2, "L": 1.6, "W": 0.5, ...}

    -- 용도
    load_case_type VARCHAR(30),            -- "strength", "service", "extreme"

    -- 출처
    code_id VARCHAR(20) NOT NULL,
    code_version DATE NOT NULL,
    clause_id VARCHAR(20),

    -- 메타
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(combination_id, code_id, code_version)
);

-- 인덱스
CREATE INDEX idx_combinations_code ON load_combinations(code_id, code_version);
CREATE INDEX idx_combinations_method ON load_combinations(design_method);

COMMENT ON TABLE load_combinations IS '하중조합 계수 테이블';
COMMENT ON COLUMN load_combinations.factors IS 'JSON 형식 계수 (예: {"D": 1.2, "L": 1.6})';
```

### 2.10 clause_text (RAG용 - 선택)

```sql
CREATE TABLE clause_text (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clause_id UUID REFERENCES source_clauses(id) ON DELETE CASCADE,

    -- 청크
    chunk_index INTEGER NOT NULL,          -- 청크 순서
    chunk_text TEXT NOT NULL,              -- 조항 텍스트
    chunk_size INTEGER,                    -- 토큰 수

    -- 임베딩
    embedding VECTOR(1536),                -- OpenAI ada-002 / 다른 모델

    -- 메타
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(clause_id, chunk_index)
);

-- 벡터 인덱스 (pgvector)
CREATE INDEX idx_clause_embedding ON clause_text USING ivfflat (embedding vector_cosine_ops);

COMMENT ON TABLE clause_text IS 'RAG용 조항 텍스트 청킹 및 임베딩';
```

---

## 3. 뷰 (Views)

### 3.1 현행 버전 하중 파라미터

```sql
CREATE VIEW v_current_load_params AS
SELECT
    lp.*,
    cv.version_date AS current_version_date,
    cd.code_title
FROM load_params lp
JOIN code_versions cv ON lp.code_id = (
    SELECT code_id FROM code_documents WHERE id = cv.document_id
) AND lp.code_version = cv.version_date
JOIN code_documents cd ON cv.document_id = cd.id
WHERE cv.is_current = TRUE
  AND lp.is_active = TRUE;

COMMENT ON VIEW v_current_load_params IS '현행 버전 하중 파라미터만 조회';
```

### 3.2 리뷰 필요 항목

```sql
CREATE VIEW v_review_needed AS
SELECT
    lp.id,
    lp.param_type,
    lp.primary_key,
    lp.display_name_ko,
    lp.value,
    lp.unit,
    lp.confidence,
    lp.review_note,
    lp.code_id,
    lp.clause_id,
    lp.table_id
FROM load_params lp
WHERE lp.needs_review = TRUE
ORDER BY lp.confidence ASC;

COMMENT ON VIEW v_review_needed IS '검토 필요 항목 (confidence 오름차순)';
```

---

## 4. 함수 (Functions)

### 4.1 파라미터 Upsert

```sql
CREATE OR REPLACE FUNCTION upsert_load_param(
    p_param_type VARCHAR,
    p_primary_key VARCHAR,
    p_secondary_key VARCHAR,
    p_display_name_ko VARCHAR,
    p_value DECIMAL,
    p_unit VARCHAR,
    p_code_id VARCHAR,
    p_code_version DATE,
    p_clause_id VARCHAR,
    p_table_id VARCHAR,
    p_confidence DECIMAL DEFAULT 1.00
)
RETURNS UUID AS $$
DECLARE
    v_id UUID;
    v_old_value DECIMAL;
BEGIN
    -- 기존 레코드 확인
    SELECT id, value INTO v_id, v_old_value
    FROM load_params
    WHERE param_type = p_param_type
      AND primary_key = p_primary_key
      AND COALESCE(secondary_key, '') = COALESCE(p_secondary_key, '')
      AND code_id = p_code_id
      AND code_version = p_code_version;

    IF v_id IS NOT NULL THEN
        -- 값이 변경된 경우에만 업데이트
        IF v_old_value != p_value THEN
            -- 이력 기록
            INSERT INTO load_params_history (param_id, old_value, new_value, change_type, changed_by)
            VALUES (v_id, v_old_value, p_value, 'update', 'agent_pipeline');

            -- 업데이트
            UPDATE load_params SET
                value = p_value,
                unit = p_unit,
                confidence = p_confidence,
                updated_at = NOW()
            WHERE id = v_id;
        END IF;
    ELSE
        -- 신규 삽입
        INSERT INTO load_params (
            param_type, primary_key, secondary_key, display_name_ko,
            value, unit, code_id, code_version, clause_id, table_id, confidence
        ) VALUES (
            p_param_type, p_primary_key, p_secondary_key, p_display_name_ko,
            p_value, p_unit, p_code_id, p_code_version, p_clause_id, p_table_id, p_confidence
        ) RETURNING id INTO v_id;

        -- 이력 기록
        INSERT INTO load_params_history (param_id, new_value, change_type, changed_by)
        VALUES (v_id, p_value, 'insert', 'agent_pipeline');
    END IF;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;
```

### 4.2 용도 키 검색

```sql
CREATE OR REPLACE FUNCTION find_occupancy_key(p_input VARCHAR)
RETURNS VARCHAR AS $$
DECLARE
    v_key VARCHAR;
BEGIN
    -- 정확히 일치
    SELECT canonical_key INTO v_key
    FROM occupancy_mapping
    WHERE display_name_ko = p_input
       OR display_name_en = p_input
       OR canonical_key = p_input;

    IF v_key IS NOT NULL THEN
        RETURN v_key;
    END IF;

    -- 별칭에서 검색
    SELECT canonical_key INTO v_key
    FROM occupancy_mapping
    WHERE aliases ? p_input;

    RETURN v_key; -- NULL이면 매핑 실패
END;
$$ LANGUAGE plpgsql;
```

---

## 5. Row Level Security (RLS)

```sql
-- RLS 활성화
ALTER TABLE load_params ENABLE ROW LEVEL SECURITY;
ALTER TABLE load_params_history ENABLE ROW LEVEL SECURITY;

-- 읽기 정책 (모든 사용자)
CREATE POLICY "Anyone can read load_params"
ON load_params FOR SELECT
USING (true);

-- 쓰기 정책 (인증된 사용자만)
CREATE POLICY "Authenticated users can insert"
ON load_params FOR INSERT
WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can update"
ON load_params FOR UPDATE
USING (auth.role() = 'authenticated');
```

---

## 6. 초기 데이터 예시

### 6.1 활하중 (KDS 41 12 00)

```sql
-- KDS 41 12 00 문서 등록
INSERT INTO code_documents (code_id, code_title, code_category)
VALUES ('KDS 41 12 00', '건축구조기준 하중', 'structural');

-- 버전 등록
INSERT INTO code_versions (document_id, version_date, is_current, source_type)
SELECT id, '2022-10-11', TRUE, 'pdf'
FROM code_documents WHERE code_id = 'KDS 41 12 00';

-- 활하중 데이터
INSERT INTO load_params (param_type, primary_key, display_name_ko, display_name_en, value, unit, code_id, code_version, clause_id, table_id)
VALUES
('live_load', 'office', '사무실', 'Office', 2.5, 'kN/m²', 'KDS 41 12 00', '2022-10-11', '3.1.1', '표 3.1-1'),
('live_load', 'residential', '주거', 'Residential', 2.0, 'kN/m²', 'KDS 41 12 00', '2022-10-11', '3.1.1', '표 3.1-1'),
('live_load', 'retail', '소매점', 'Retail', 4.0, 'kN/m²', 'KDS 41 12 00', '2022-10-11', '3.1.1', '표 3.1-1'),
('live_load', 'assembly', '집회', 'Assembly', 5.0, 'kN/m²', 'KDS 41 12 00', '2022-10-11', '3.1.1', '표 3.1-1'),
('live_load', 'storage', '창고', 'Storage', 6.0, 'kN/m²', 'KDS 41 12 00', '2022-10-11', '3.1.1', '표 3.1-1'),
('live_load', 'parking', '주차장', 'Parking', 2.5, 'kN/m²', 'KDS 41 12 00', '2022-10-11', '3.1.1', '표 3.1-1');
```

---

## 7. 마이그레이션 스크립트

```sql
-- migrations/001_initial_schema.sql
-- 위의 모든 CREATE 문 포함

-- migrations/002_seed_data.sql
-- occupancy_mapping, region_mapping 초기 데이터

-- migrations/003_add_indexes.sql
-- 성능 최적화 인덱스 추가
```

---

## 8. 조회 예시

### 8.1 용도별 활하중 조회

```sql
SELECT primary_key, display_name_ko, value, unit
FROM load_params
WHERE param_type = 'live_load'
  AND code_version = '2022-10-11'
  AND is_active = TRUE
ORDER BY value DESC;
```

### 8.2 특정 버전 비교

```sql
SELECT
    p1.display_name_ko,
    p1.value AS "2022_value",
    p2.value AS "2019_value",
    p1.value - p2.value AS diff
FROM load_params p1
JOIN load_params p2
    ON p1.primary_key = p2.primary_key
    AND p1.param_type = p2.param_type
WHERE p1.code_version = '2022-10-11'
  AND p2.code_version = '2019-08-01';
```

---

## 9. 관련 문서

- [01_AGENT_ARCHITECTURE.md](./01_AGENT_ARCHITECTURE.md) - Agent 스펙
- [02_DATA_FLOW.md](./02_DATA_FLOW.md) - 데이터 흐름
- [04_PROMPT_TEMPLATES.md](./04_PROMPT_TEMPLATES.md) - 프롬프트 템플릿
