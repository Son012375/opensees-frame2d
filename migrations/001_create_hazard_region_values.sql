-- ============================================================
-- hazard_region_values: 시/군/구별 위험도 기반값 (Sg, V0 등)
-- KDS 41 12 00 등고선 지도 → 행정구역 매핑 테이블
-- ============================================================

CREATE TABLE IF NOT EXISTS hazard_region_values (
    id              uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    hazard_type     text NOT NULL,           -- 'snow_sg' | 'wind_v0'
    region_sido     text NOT NULL,           -- 시/도명 (예: '서울특별시')
    region_sigungu  text NOT NULL,           -- 시/군/구명 (예: '강남구')
    region_code     text,                    -- 법정동코드 앞 5자리 (선택)
    value           numeric NOT NULL,        -- Sg(kN/m²) 또는 V₀(m/s)
    unit            text NOT NULL,           -- 'kN/m2' | 'm/s'
    lat             double precision,        -- 대표 좌표 (위도)
    lon             double precision,        -- 대표 좌표 (경도)
    source_code     text NOT NULL,           -- 'KDS 41 12 00'
    source_ref      text,                    -- '그림 4.2-1' 또는 '그림 5.5-1'
    source_version  text,                    -- '2022-10-11'
    verified_by     text,                    -- 검증 출처 (예: 'midas_gen')
    notes           text,
    created_at      timestamptz DEFAULT now(),
    updated_at      timestamptz DEFAULT now(),

    -- 동일 hazard_type + 행정구역 조합은 유일해야 함
    CONSTRAINT uq_hazard_region UNIQUE (hazard_type, region_sido, region_sigungu)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_hazard_region_type ON hazard_region_values (hazard_type);
CREATE INDEX IF NOT EXISTS idx_hazard_region_sido ON hazard_region_values (region_sido);
CREATE INDEX IF NOT EXISTS idx_hazard_region_code ON hazard_region_values (region_code) WHERE region_code IS NOT NULL;

-- updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_hazard_region_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_hazard_region_updated ON hazard_region_values;
CREATE TRIGGER trg_hazard_region_updated
    BEFORE UPDATE ON hazard_region_values
    FOR EACH ROW
    EXECUTE FUNCTION update_hazard_region_timestamp();

-- RLS (기존 프로젝트 스타일)
ALTER TABLE hazard_region_values ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow read access" ON hazard_region_values
    FOR SELECT USING (true);

CREATE POLICY "Allow insert for authenticated" ON hazard_region_values
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow update for authenticated" ON hazard_region_values
    FOR UPDATE USING (true);
