# LOAD_DB_CONTRACT — KDS 하중 파라미터 DB 명세

> **Version:** 1.0.0
> **Date:** 2026-02-10
> **Status:** Active
> **Machine-readable:** [`load_contract.yaml`](./load_contract.yaml)

---

## 1. 목적

`public.load_params` 테이블은 **결정론적 계산의 단일 진실원(Single Source of Truth)**이다.

- DB에 저장된 수치·계수·조건은 KDS 원문에서 추출·검증된 확정값이다.
- LLM(Claude)은 사용자 입력을 정규화하고, DB 값을 조회·조합하여 설명을 생성하는 **보조 역할**이다.
- LLM이 DB 값을 "추정"하거나 "보정"하는 것은 금지한다. 값이 없으면 `needs_review=true`로 플래그하고 사용자에게 확인을 요청한다.

---

## 2. 데이터 모델

### 2.1 테이블: `public.load_params`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | uuid | PK (auto) |
| `param_type` | varchar | 파라미터 대분류 |
| `param_subtype` | varchar | 세부 분류 |
| `primary_key` | varchar | 주요 식별키 (하중 유형, 변수명 등) |
| `secondary_key` | varchar | 보조 식별키 (세부 용도, 구간, 조합명 등) |
| `display_name_ko` | varchar(200) | 한국어 표시명 |
| `display_name_en` | varchar(200) | 영어 표시명 |
| `value` | numeric | 대표값 (계수, 하중값 등). 수식만 있는 경우 null |
| `value_min` | numeric | 최솟값 또는 범위 하한 |
| `value_max` | numeric | 최댓값 또는 범위 상한 |
| `unit` | varchar | 단위 (`kN/m²`, `kN`, `kN/m`, `m²`, `-`) |
| `conditions` | jsonb | 구조화된 적용 조건 (§4 참조) |
| `notes` | text | 원문 수식, 비고, 자유 텍스트 |
| `code_id` | varchar | 설계기준 ID (예: `KDS 41 12 00`) |
| `code_version` | varchar | 기준 버전 (예: `2022-10-11`) |
| `clause_id` | varchar | 절/항 번호 (예: `3.5.1`) |
| `table_id` | varchar | 표 번호 (예: `표 3.2-1`) |
| `confidence` | numeric | 추출 신뢰도 (0.0~1.0) |
| `needs_review` | boolean | 사람 검토 필요 여부 |
| `review_note` | text | 검토 사유 |
| `is_active` | boolean | 활성 여부 (soft delete) |
| `created_by` | varchar | 생성 주체 |
| `created_at` | timestamptz | 생성 시각 |
| `updated_at` | timestamptz | 수정 시각 |

### 2.2 UNIQUE 제약

```
UNIQUE(param_type, param_subtype, primary_key, secondary_key, code_id, code_version)
```

동일 기준·동일 키 조합에 대해 하나의 레코드만 존재한다. `secondary_key`가 NULL인 경우 PostgreSQL은 각 NULL을 고유하게 취급한다.

### 2.3 Views

| View 이름 | 필터 조건 | 용도 |
|-----------|----------|------|
| `live_load_distributed` | `param_type='live_load' AND param_subtype='distributed'` | 표 3.2-1 등분포활하중 |
| `live_load_concentrated` | `param_type='live_load' AND param_subtype='concentrated'` | 표 3.3-1 집중활하중 |
| `live_reduction` | `param_type='live_reduction'` | 3.5절 활하중 저감 |
| `roof_live_reduction` | `param_type='roof_live_reduction'` | 3.6절 지붕활하중 저감 |
| `similar_live_load` | `param_type='similar_live_load'` | 3.7~3.8절 유사활하중 |
| `load_combo` | `param_type='load_combo'` | 1.7절 하중조합 계수 |

---

## 3. param_type별 역할 정의

### 3.1 `live_load` — 기본 하중값

활하중의 **원시값(raw value)**을 저장한다. 저감 전 기본값이다.

| param_subtype | primary_key 예시 | secondary_key 예시 | value 의미 |
|---|---|---|---|
| `distributed` | `office`, `residential` | `null`, `hospital_ward` | 등분포활하중 (kN/m²) |
| `concentrated` | `office`, `parking` | `null`, `parking_truck` | 집중활하중 (kN) |

### 3.2 `live_reduction` — 활하중 저감 규칙

영향면적 기반 저감계수, 적용 한계, 예외 용도를 저장한다.

| param_subtype | primary_key 예시 | value 의미 |
|---|---|---|
| `general` | `reduction_formula`, `min_influence_area`, `min_factor_1floor` | 저감식, 최소면적(36m²), 하한(0.5) |
| `influence_area` | `column_foundation`, `beam_wall`, `slab` | 영향면적 배수 (4, 2, 1) |
| `exceptions` | `heavy_load_threshold`, `assembly_no_reduction` | 저감 불가 조건 |

### 3.3 `roof_live_reduction` — 지붕활하중 저감

부하면적(R1)과 물매(R2)에 따른 저감계수를 저장한다.

| param_subtype | primary_key | secondary_key 패턴 | value 의미 |
|---|---|---|---|
| `formula` | `L` | null | value_min/max = 저감 범위 |
| `general` | `Lo`, `At`, `F` | null | 변수 정의 |
| `by_area` | `R1` | `At_0_20`, `At_20_60`, `At_ge_60` | R1 계수 또는 null(수식) |
| `by_slope` | `R2` | `F_le_1_3`, `F_1_3_1`, `F_ge_1` | R2 계수 또는 null(수식) |
| `exceptions` | `occupancy_assembly` | null | 예외 규칙 |

### 3.4 `similar_live_load` — 유사활하중

난간, 내벽 횡하중, 차량 방호하중 등 특수 하중을 저장한다.

| param_subtype | primary_key 예시 | unit |
|---|---|---|
| `railing` | `railing_point_load`, `railing_line_load`, `fixed_ladder_load` | kN, kN/m |
| `partition_wall` | `partition_lateral_load` | kN/m² |
| `vehicle_barrier` | `vehicle_barrier_load` | kN |

### 3.5 `load_combo` — 하중조합 계수

각 조합 내 **하중 항(load case)의 계수**를 개별 레코드로 저장한다.

| param_subtype | secondary_key | primary_key | value |
|---|---|---|---|
| `uls` | `ULS1`~`ULS7` | `DL`, `LL`, `RLL`, `SNOW`, `WIND`, `EQ` | 계수 (1.4, 1.2, 0.5 등) |
| `sls` | `SLS1`~`SLS8` | `DL`, `LL`, `RLL`, `SNOW`, `WIND`, `EQ` | 계수 (1.0, 0.75, 0.6 등) |

---

## 4. conditions 해석 규칙 (Interpreter Spec)

`conditions`는 jsonb 배열이다. 각 원소는 `kind` 필드로 분류한다.

### 4.1 `kind: "alt"` — 대안 선택

```json
{
  "kind": "alt",
  "group": "roof_or_snow",
  "candidates": ["RLL", "SNOW"],
  "choose": 1,
  "policy": "max_effect"
}
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `group` | Y | 대안 그룹 ID. 같은 group 내 레코드들은 상호 배타 |
| `candidates` | Y | 대안 후보 primary_key 목록 |
| `choose` | N | 선택 개수 (기본 1) |
| `policy` | N | 선택 정책: `max_effect`(기본, 최대 효과), `envelope`(모두 검토) |

**표준 그룹:**

| group | candidates | 출처 |
|-------|-----------|------|
| `roof_or_snow` | `["RLL", "SNOW"]` | Lr 또는 S 또는 R |
| `wind_or_eq` | `["WIND", "EQ"]` | 0.6W 또는 0.7E |
| `live_or_wind` | `["LL", "WIND"]` | 1.0L 또는 0.5W (ULS3) |

**해석 규칙:**
1. 같은 `secondary_key`(조합) 내에서 동일 `group`을 가진 레코드들을 수집한다.
2. `policy=max_effect`: 각 후보를 적용해보고, 구조물에 가장 불리한 결과를 채택한다.
3. `policy=envelope`: 모든 후보 조합을 생성하여 포락선 검토한다.
4. 선택되지 않은 후보의 계수는 **0**으로 처리한다.

**`max_effect` 알고리즘 (수식 정의):**

```
for candidate_key in candidates:
    factor = combo_record[candidate_key].value   # DB 저장 계수
    load   = load_cases[candidate_key]           # Step 5 하중값
    effect = factor × load

selected = argmax(effect over all candidates)
```

- `effect = factor × load_case_value` — 계수와 하중값의 곱이다.
- `effect`가 가장 큰 후보를 채택한다. 나머지 후보의 계수는 0이 된다.
- 모든 후보의 effect가 0이면 (하중 미입력), 첫 번째 후보를 기본 선택한다.
- 부호가 의미 있는 경우(양/음 효과), **절대값이 아닌 구조 효과의 크기(magnitude)**로 판단한다.
- 예: `live_or_wind` 그룹에서 LL factor=1.0, load=2.5 → effect=2.5 vs WIND factor=0.5, load=1.2 → effect=0.6 → LL 채택

### 4.2 `kind: "reduction"` — 하중 저감

```json
{
  "kind": "reduction",
  "target": "LL",
  "allowed": true,
  "factor": 0.5,
  "ref": "live_reduction",
  "condition_ko": "기본등분포활하중 ≤5.0kN/m² (주차장·공공집회 제외)"
}
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `target` | Y | 저감 대상 load case 키 |
| `allowed` | Y | 저감 허용 여부 |
| `factor` | Y | 저감 시 적용 계수 (원래 value를 이 값으로 대체) |
| `ref` | N | 참조 param_type (상세 조건 조회용) |
| `condition_ko` | N | 적용 조건 한국어 설명 |

**해석 규칙:**
1. 해당 조합의 `target` 레코드를 찾는다.
2. 적용 조건(`condition_ko`)을 사용자 입력과 대조한다.
3. 조건 충족 시 `value`를 `factor`로 교체한다 (예: LL 1.0 → 0.5).
4. `ref`가 있으면 해당 param_type의 상세 규칙을 추가 조회한다.

### 4.3 `kind: "derived"` — 파생 계수

```json
{
  "kind": "derived",
  "expression": "0.75*0.6",
  "base_factors": [0.75, 0.6]
}
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `expression` | Y | 계수 도출 산식 문자열 |
| `base_factors` | Y | 기저 계수 배열 |

**해석 규칙:**
- `derived`는 **정보 제공용**이다. 실제 계산에는 레코드의 `value`를 사용한다.
- 리포트 생성 시 계수의 근거를 추적하는 데 활용한다.
- 예: SLS6 WIND의 value=0.45는 0.75×0.6에서 도출됨.

### 4.4 `kind: "includes"` — 동반 하중 (F/T/H 처리 정책)

```json
{
  "includes": ["F", "T"],
  "note": "F(유체압)·T(온도하중) 동일 계수 적용"
}
```

- `kind` 필드 없이 `includes` 배열이 존재하면 동반 하중을 의미한다.
- D(고정하중)에 F(유체압), T(온도하중), H(토압)가 동일 계수로 포함됨을 명시한다.

**F/T/H 처리 정책:**

| 기호 | 의미 | 적용 | DB 처리 |
|------|------|------|---------|
| F | 유체하중 (Fluid) | DL에 포함 | 별도 load case 미분리 |
| T | 온도하중 (Temperature) | DL에 포함 | 별도 load case 미분리 |
| H | 토압/수압 (Hydrostatic) | DL에 포함 | 별도 load case 미분리 |

1. **현재 정책 (Phase 1):** F/T/H는 DL 또는 SDL에 **흡수(absorbed)**하여 처리한다.
   - 사용자가 입력하는 DL 값에 F/T/H가 이미 포함되어 있다고 가정한다.
   - 조합식의 `1.2(D+F+T)` → 실질적으로 `1.2×DL`로 계산한다.
2. **리포트 표시:** `includes` 조건이 있는 레코드는 리포트에서 "(F 포함)" 등을 주석으로 표시한다.
3. **향후 확장 (Phase 2+):** F/T/H를 별도 load case로 분리할 경우, `includes` 조건을 해석하여 동일 계수를 자동 적용한다.

### 4.5 `needs_review` 처리

- `needs_review=true`인 레코드는 **자동 적용하지 않는다**.
- 사용자에게 해당 값의 출처와 불확실 사유(`review_note`)를 표시하고 확인을 받는다.
- `confidence < 0.7`이면 `needs_review=true`로 설정한다.

---

## 5. 키 사전

### 5.1 Load Case 키

| 키 | 의미 | KDS 기호 | 비고 |
|----|------|----------|------|
| `DL` | 고정하중 (자중) | D | F(유체압) 포함 가능 |
| `SDL` | 추가 고정하중 | D (일부) | 마감재, 설비 등. 현재 미분리 |
| `LL` | 바닥 활하중 | L | 용도별 표 3.2-1 값 |
| `RLL` | 지붕 활하중 | Lr | 용도별 표 3.2-1 지붕 항목 |
| `SNOW` | 설하중 | S | P2 확장 예정 |
| `WIND` | 풍하중 | W | 방향별 확장: WIND_X_POS/NEG, WIND_Y_POS/NEG |
| `EQ` | 지진하중 | E | 방향별 확장: EQ_X, EQ_Y |

**방향 확장 규칙 (WIND / EQ):**

DB에는 `WIND`, `EQ` 단일 키만 저장한다. **해석 단계(Step 6)**에서 내부적으로 방향별 케이스로 확장한다.

| DB 키 | 확장 케이스 | 의미 |
|--------|-----------|------|
| `WIND` | `WIND_X_POS` | +X 방향 풍하중 |
| | `WIND_X_NEG` | -X 방향 풍하중 |
| | `WIND_Y_POS` | +Y 방향 풍하중 |
| | `WIND_Y_NEG` | -Y 방향 풍하중 |
| `EQ` | `EQ_X` | X 방향 지진하중 |
| | `EQ_Y` | Y 방향 지진하중 |

- 확장은 조합 적용 **직전**(Step 5→6 사이)에 수행한다.
- DB의 `load_combo` 레코드에서 `primary_key='WIND'`인 계수는 4개 방향 케이스 모두에 동일하게 적용한다.
- 각 방향 케이스의 하중값은 사용자가 별도로 입력한다 (풍압 계산 결과 등).
- 결과적으로 하나의 ULS 조합이 WIND 방향 수만큼 복제된다. 이 중 가장 불리한 것을 `governing`으로 선택한다.

### 5.2 조합 키

| 키 | 식 번호 | 수식 요약 |
|----|--------|----------|
| `ULS1` | 1.7-1 | 1.4(D+F) |
| `ULS2` | 1.7-2 | 1.2(D+F+T)+1.6(L+H)+0.5(Lr/S/R) |
| `ULS3` | 1.7-3 | 1.2D+1.6(Lr/S/R)+(1.0L or 0.5W) |
| `ULS4` | 1.7-4 | 1.2D+1.0W+1.0L+0.5(Lr/S/R) |
| `ULS5` | 1.7-5 | 1.2D+1.0E+1.0L+0.2S |
| `ULS6` | 1.7-6 | 0.9D+1.0W |
| `ULS7` | 1.7-7 | 0.9D+1.0E |
| `SLS1` | 1.7-8 | D+F |
| `SLS2` | 1.7-9 | D+F+L+T |
| `SLS3` | 1.7-10 | D+F+(Lr/S/R) |
| `SLS4` | 1.7-11 | D+F+0.75(L+T)+0.75(Lr/S/R) |
| `SLS5` | 1.7-12 | D+F+(0.6W or 0.7E) |
| `SLS6` | 1.7-13 | D+F+0.75(0.6W/0.7E)+0.75L+0.75(Lr/S/R) |
| `SLS7` | 1.7-14 | 0.6D+0.6W |
| `SLS8` | 1.7-15 | 0.6D+0.7E |

### 5.3 Occupancy 키 (primary_key for live_load)

전체 매핑: `data/mapping/occupancy.json`

주요 키: `office`, `residential`, `hospital`, `school`, `library`, `retail`, `assembly`, `storage`, `parking`, `factory`, `roof`

### 5.4 하중 적용 규칙 (Load Application Spec)

하중의 `unit` 필드에 따라 구조 모델에 적용하는 방식이 결정된다.

| unit | 하중 유형 | 적용 대상 | 적용 방식 |
|------|----------|----------|----------|
| `kN/m²` | 면하중 (area load) | 슬래브, 다이어프램 요소 | 면적 비례 분배 |
| `kN/m` | 선하중 (line load) | 보, 난간, 벽체 상단 | 길이 비례 분배 |
| `kN` | 집중하중 (point load) | 특정 절점 | 단일 점 적용 |

**적용 규칙:**

1. **면하중 (kN/m²):** 해당 부재의 부하면적(tributary area)에 곱하여 등가 절점하중으로 변환한다.
   - 예: 사무실 2.5 kN/m² × 부하면적 18m² = 45 kN
2. **선하중 (kN/m):** 해당 부재의 길이에 곱하여 등가 절점하중으로 변환한다.
   - 예: 난간 0.8 kN/m × 난간 길이 5m = 4.0 kN
3. **집중하중 (kN):** 지정된 위치의 절점에 직접 적용한다.
   - 예: 차량 방호하중 30 kN → 접촉면 위치 절점에 적용
4. **동시 적용 불가:** 같은 부재에 면하중과 집중하중이 모두 정의된 경우, **불리한 쪽** 하나만 적용한다 (동시 재하 아님).

---

## 6. 조회/적용 순서 (Evaluation Order)

```
사용자 입력 → [Step 1~6] → 설계하중 조합 결과
```

### Step 1: 설계기준 선택

```sql
WHERE code_id = 'KDS 41 12 00' AND code_version = '2022-10-11'
```

향후 복수 기준 지원 시, 사용자가 기준을 명시적으로 선택한다.

### Step 2: 입력 정규화

사용자의 자연어 입력을 도메인 키로 변환한다.

| 사용자 입력 | 정규화 키 |
|------------|----------|
| "사무실", "업무시설" | `occupancy=office` |
| "3층 기둥" | `member_type=column`, `floors_supported=3` |
| "지붕 경사 1:4" | `F=0.25` |

이 단계는 **LLM이 수행**하되, `data/mapping/occupancy.json`을 참조한다.

### Step 3: 기본 하중 조회

```sql
SELECT value, unit FROM live_load_distributed
WHERE primary_key = :occupancy AND (secondary_key = :sub_use OR secondary_key IS NULL);
```

- 결과: `base_LL = 2.5 kN/m²` (예: office)
- 지붕: `roof_live_reduction` view에서 Lo 참조

### Step 4: 저감/예외 적용

#### 4a. 바닥 활하중 저감 (live_reduction)

```python
A = tributary_area * influence_multiplier  # Step: 영향면적
if A >= 36:
    C = 0.3 + 4.2 / sqrt(A)
    C = max(C, C_min)  # 층수에 따른 하한
    # 예외 확인: assembly, parking, heavy_load
    reduced_LL = base_LL * C
```

#### 4b. 지붕활하중 저감 (roof_live_reduction)

```python
R1 = lookup_R1(At)  # by_area 구간별 조회
R2 = lookup_R2(F)   # by_slope 구간별 조회
L = Lo * R1 * R2
L = clamp(L, 0.6, 1.0)  # 범위 제한
```

### Step 5: Load Case 구성

저감 적용된 최종 하중값으로 load case 딕셔너리를 구성한다.

```json
{
  "DL": 5.0,
  "LL": 2.5,
  "RLL": 1.0,
  "SNOW": 0.5,
  "WIND": 1.2,
  "EQ": 0.0
}
```

단위는 모두 kN/m²로 통일한다 (집중하중은 kN).

### Step 6: 하중조합 적용

```sql
SELECT primary_key, value, conditions
FROM load_combo
WHERE param_subtype = 'uls' AND secondary_key = :combo_id;
```

각 조합에 대해:

1. 해당 조합의 모든 레코드를 가져온다.
2. `kind=alt` 조건 처리: 같은 group 내 후보 중 하나를 선택한다.
3. `kind=reduction` 조건 처리: 적용 가능하면 계수를 교체한다.
4. 최종 조합하중 = Σ(load_case[key] × factor)

```python
combo_load = sum(
    load_cases.get(rec.primary_key, 0) * rec.value
    for rec in selected_records
)
```

---

## 7. 향후 확장 가이드

### P2: 설하중 / 풍하중

| 항목 | param_type | param_subtype 예시 |
|------|-----------|-------------------|
| 지상설하중 | `snow_load` | `ground`, `flat_roof`, `unbalanced` |
| 기본풍속 | `wind_speed` | `region`, `exposure`, `topography` |
| 풍압계수 | `wind_pressure` | `external`, `internal` |

동일한 키 구조(`primary_key`/`secondary_key`)와 conditions 패턴을 그대로 사용한다.

### P3: 내진설계

| 항목 | param_type | param_subtype 예시 |
|------|-----------|-------------------|
| 지반분류 | `seismic` | `site_class` |
| 응답스펙트럼 | `seismic` | `response_spectrum` |

---

## 8. 참조 파일

| 파일 | 설명 |
|------|------|
| `docs/specs/load_contract.yaml` | 머신 리더블 스키마 |
| `docs/specs/examples/combo_apply_example.json` | 조합 적용 예시 |
| `docs/specs/examples/live_reduction_example.json` | 활하중 저감 예시 |
| `docs/specs/examples/roof_live_reduction_example.json` | 지붕 저감 예시 |
| `docs/specs/examples/similar_live_load_example.json` | 유사활하중 예시 |
| `data/mapping/occupancy.json` | 용도 키 매핑 |
| `scripts/supabase_loader.py` | DB 적재 스크립트 |
