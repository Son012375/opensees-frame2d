# Case 6 — Midas Gen 2019 단계별 모델링 가이드 (한국어)

본 문서는 [Case 6 L-shape 검증 스펙](_case6_lshape_spec.md)과 [영문 handoff](_case6_lshape_midas_handoff.md)의 **Midas Gen 2019 UI 기준 실행 가이드**입니다. Midas 2019의 메뉴 구조를 따라 순서대로 정리했습니다.

## 0. 사전 준비

작업 시작 전 다음 두 파일을 열어두세요:

| 파일 | 용도 |
|---|---|
| [tests/benchmark/case6_lshape_loadtables.json](../tests/benchmark/case6_lshape_loadtables.json) | 보 선하중 + 절점 횡하중 + master 노드 (OpenSees runner가 생성) |
| [tests/benchmark/midas_results/case6_lshape.json](../tests/benchmark/midas_results/case6_lshape.json) | 결과 입력용 24개 키 템플릿 (현재 전부 `null`) |

JSON 뷰어 추천: VS Code 또는 Notepad++. Excel로도 열 수 있지만 들여쓰기 깨질 수 있음.

**전체 모델 규모 사전 인지** (입력량 가늠용):
- 노드 70개, 부재 135개 (기둥 57 + 보 78)
- 보 선하중 156건 (DL 78 + LL 78)
- 절점 횡하중 110건 (EQX 55 + EQY 55) → 1F~3F 13개 × 3 + 4F~5F 9개 × 2 = 57개 노드 × 2 방향
- 5개 층 강막 정의

UI로 일일이 입력하면 약 2~3시간 소요. **MCT 명령어 일괄 입력**(아래 §15)을 추천합니다.

---

## 1. 새 프로젝트 + 단위계 설정

**메뉴**: `File → New Project`

단위 설정: `Tools → Unit System`

| 항목 | 값 |
|---|---|
| Force | **kN** |
| Length | **mm** |
| Heat | kJ |
| Temperature | Celsius |

> **주의**: 본 가이드는 길이를 mm로 고정합니다. m로 작업할 경우 Section 입력 시 단위 환산을 다시 해야 하므로 mm 권장.

구조 형식: `Tools → Project Settings`

- Structure Type: **3-D**
- Mass Control: Lumped Mass (옵션, 이 검증에선 정적해석이라 무관)
- Self-weight: **사용 안 함** (DL을 직접 적용하므로 자중 OFF)

---

## 2. 재료 정의 — SS275 (E = 205,000 MPa)

**메뉴**: `Property → Material Properties` → `Add`

| 항목 | 값 |
|---|---|
| Material Number | 1 |
| Name | SS275 |
| Type of Design | **User Defined** (DB 사용 안 함 — E 값 직접 통제) |
| Modulus of Elasticity (E) | **205000** N/mm² |
| Poisson's Ratio (ν) | 0.3 |
| Thermal Coefficient (α) | 1.2e-05 |
| Weight Density (γ) | **0** (자중 적용 안 함) |
| Mass Density (ρ) | 0 |
| Plastic Material Property | None |

> **중요**: DB → KS-SS275를 선택하면 E가 200 GPa 또는 210 GPa로 잡힐 수 있습니다. OpenSees runner는 **205 GPa**를 쓰므로 반드시 User Defined로 직접 입력해주세요. ([_case6_lshape_spec.md §4](_case6_lshape_spec.md) 참고)

---

## 3. 단면 정의 (H-300x300, H-400x200)

**메뉴**: `Property → Section Properties` → `Add`

### 3.1 H-300×300 (기둥)

| 항목 | 값 |
|---|---|
| Section Number | 1 |
| Name | H-300x300x10x15 |
| Section Type | **DB/User** |
| DB | KS 또는 None |
| Section Shape | **H-Section** |
| H (overall depth) | 300 mm |
| B (flange width) | 300 mm |
| tw (web) | 10 mm |
| tf (flange) | 15 mm |

**확인** (Properties 탭 또는 Detail):
- A = 11,980 mm² (Midas 자동 계산값 약 ±0.5% 차이 허용)
- Iy = 2.04e8 mm⁴ (Midas 표기로는 "강축 Ix" — 단면축 정의가 Midas에선 y-y가 강축)
- Iz = 6.75e7 mm⁴

> **주의 — Midas vs OpenSees 단면축 정의**:
> - Midas: y-y축이 강축, z-z축이 약축
> - OpenSees: x축이 부재축, y/z가 단면 주축 (vecxz로 방향 정의)
> 본 검증에선 기둥은 강축이 글로벌 X에 정렬됩니다. Midas에서 부재 생성 시 `Beta Angle = 0°` 두면 강축이 글로벌 X에 자동 정렬됩니다 (Midas 기본 회전).

### 3.2 H-400×200 (보)

| 항목 | 값 |
|---|---|
| Section Number | 2 |
| Name | H-400x200x8x13 |
| Section Type | DB/User |
| Section Shape | **H-Section** |
| H | 400 mm |
| B | 200 mm |
| tw | 8 mm |
| tf | 13 mm |

확인:
- A = 8,412 mm²
- Iy(강축) = 2.37e8 mm⁴
- Iz(약축) = 1.74e7 mm⁴

---

## 4. 노드 입력 (70개)

**메뉴**: `Model → Nodes → Create Nodes` (또는 Tree Menu에서 Right-click → Create Nodes)

가장 빠른 방법: **Excel-MCT 복사** (아래 §15에서 자세히 설명).

수동 입력 시 좌표 (단위 m → mm 변환 주의):

| 노드 ID | x (mm) | y (mm) | z (mm) | 위치 |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | L1 base |
| 2 | 0 | 0 | 3500 | L1 1F |
| 3 | 0 | 0 | 7000 | L1 2F |
| 4 | 0 | 0 | 10500 | L1 3F |
| 5 | 0 | 0 | 14000 | L1 4F |
| 6 | 0 | 0 | 17500 | L1 5F |
| 7~12 | 6000 | 0 | 0~17500 | L2 (x=6) |
| 13~18 | 12000 | 0 | 0~17500 | L3 (x=12) |
| 19~24 | 0 | 4000 | 0~17500 | L4 (y=4) |
| 25~30 | 6000 | 4000 | 0~17500 | L5 |
| 31~36 | 12000 | 4000 | 0~17500 | L6 (shared boundary) |
| 37~42 | 0 | 8000 | 0~17500 | L7 (y=8) |
| 43~48 | 6000 | 8000 | 0~17500 | L8 |
| 49~54 | 12000 | 8000 | 0~17500 | L9 |
| 55~58 | 18000 | 0 | 0~10500 | R1 (Zone B, x=18) |
| 59~62 | 24000 | 0 | 0~10500 | R2 (Zone B, x=24) |
| 63~66 | 18000 | 4000 | 0~10500 | R3 |
| 67~70 | 24000 | 4000 | 0~10500 | R4 |

> **노드 ID 순서**: 본 검증은 **노드 ID = OpenSees ID와 1:1 일치**해야 결과 비교가 명확합니다. Midas에선 Excel-MCT 또는 위 표 순서대로 입력하면 ID가 1~70까지 동일하게 매겨집니다.

전체 좌표 spec: [_case6_lshape_spec.md §1.1](_case6_lshape_spec.md) (column line별 stack)

---

## 5. 부재 (Element) 생성 — 기둥 57 + 보 78 = 135개

**메뉴**: `Model → Elements → Create Elements`

| 공통 설정 | 값 |
|---|---|
| Element Type | **General Beam/Tapered Beam** |
| Material | 1 (SS275) |
| Section | 1 (기둥) 또는 2 (보) |
| Beta Angle | **0°** |
| Sub-Type | Beam |

### 5.1 기둥 (Section = 1, H-300×300)

각 column line별로 base node → 위로 5개(Zone A) 또는 3개(Zone B) 기둥 생성.

빠른 입력: `Extrude Elements` 사용
- Node 1 선택 → Extrude Direction: +Z, Distance: 3500 mm, 5회 (Zone A)
- Zone B node (55, 59, 63, 67)는 3회만 (3층까지)

생성 결과: 57개 기둥 부재 (ID 1~57 권장)

### 5.2 보 (Section = 2, H-400×200)

층별로 2-노드 직접 연결:

**1F (z=3500), 2F (z=7000), 3F (z=10500)** — 전체 L자 평면:

X방향 보:
- (2↔8), (8↔14), (20↔26), (26↔32), (38↔44), (44↔50) — Zone A 6개
- (14↔56), (56↔60), (32↔64), (64↔68) — Zone B 4개 (1F-3F 한정)

Y방향 보:
- (2↔20), (20↔38), (8↔26), (26↔44), (14↔32), (32↔50) — Zone A 6개
- (56↔64), (60↔68) — Zone B 2개

각 층마다 18개 보 → 1F~3F 54개

**4F (z=14000), 5F (z=17500)** — Zone A만:
- X방향 6개 + Y방향 6개 = 12개 × 2층 = 24개 보

> 노드 ID는 §4 표 참고. 예: 1F의 (x=0,y=0)→(x=6,y=0) 보는 노드 2와 8을 연결.

생성 결과: 78개 보 부재 (ID 58~135 권장)

---

## 6. 경계조건 (지점) — 13개 base FIXED

**메뉴**: `Boundary → Define Supports`

대상 노드: 1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 59, 63, 67 (z=0 노드)

설정:
- Dx, Dy, Dz, Rx, Ry, Rz **전부 체크** (Fixed 6-DOF)

선택 방법: `Activate Group` 또는 좌표 z=0인 노드 박스 선택 후 한꺼번에 지점 적용.

---

## 7. 강막 (Rigid Diaphragm) 설정 — 5층

**메뉴**: `Boundary → Rigid Link` (Midas 2019에서 가장 안전한 방법)

또는 `Boundary → Story Data`에서 Floor Diaphragm 자동 설정도 가능하지만, **부분 층(4F, 5F)에서 Zone B 노드가 포함되지 않도록 수동 검증 필수**.

### 권장: Rigid Link 직접 정의

각 층마다 1개의 Rigid Link 생성:

| 층 | Master Node | Slaves (Master 제외) | DOF |
|---|---|---|---|
| 1F | 38 | 2, 8, 14, 20, 26, 32, 44, 50, 56, 60, 64, 68 | Dx, Dy, Rz |
| 2F | 39 | 3, 9, 15, 21, 27, 33, 45, 51, 57, 61, 65, 69 | Dx, Dy, Rz |
| 3F | 40 | 4, 10, 16, 22, 28, 34, 46, 52, 58, 62, 66, 70 | Dx, Dy, Rz |
| 4F | 29 | 5, 11, 17, 23, 35, 41, 47, 53 | Dx, Dy, Rz |
| 5F | 30 | 6, 12, 18, 24, 36, 42, 48, 54 | Dx, Dy, Rz |

> **Master 노드 위치 주의**:
> - 1F~3F의 master는 Zone A 북서 모서리 (0, 8) — 일반적 관행과 다르나 OpenSees runner 기본 선택이라 그대로 사용 (강막은 master 선택에 관계없이 같은 응답 산출)
> - 4F~5F는 (6, 4) — Zone A 내부
> - **다른 노드를 master로 잡아도 응답은 동일**합니다. 위 표는 OpenSees와 정확히 같은 master를 쓸 경우입니다.

**Rigid Link 입력 화면**:
1. Master Node ID 입력 (예: 38)
2. Slaves: Slave 노드들을 차례로 선택
3. Constraint Type: **Plate (XY)** — Dx, Dy, Rz만 강결 (수직 방향 Dz와 Rx, Ry는 풀림)

> **4F, 5F에서 Zone B 노드 포함 금지**: 4F는 노드 5, 11, 17, 23, 29, 35, 41, 47, 53 (Zone A 9개만). 노드 11(x=6, y=0, z=14)이 아니라 노드 23(x=0, y=4, z=14)이 master가 아니므로 명확히 분리.

---

## 8. 하중 케이스 정의 — DL, LL, EQX, EQY

**메뉴**: `Load → Static Loads → Static Load Cases`

다음 4개 케이스 추가:

| Case Number | Name | Type | Description |
|---|---|---|---|
| 1 | DL | **Dead Load (D)** | Floor area 5.1 kN/m² → 보 선하중 |
| 2 | LL | **Live Load (L)** | Floor area 2.5 kN/m² → 보 선하중 |
| 3 | EQX | **User Defined (UL)** 또는 Earthquake X | 등가정적 X방향 |
| 4 | EQY | **User Defined (UL)** 또는 Earthquake Y | 등가정적 Y방향 |

> **하중 조합 없음**: 4개 케이스 모두 독립적으로 선형정적 해석. 1.2D+1.6L 등 조합 사용 안 함.

---

## 9. DL/LL 보 선하중 입력 (각 78건)

**메뉴**: `Load → Static Loads → Element Beam Loads`

### 9.1 입력 형식

| 항목 | 값 |
|---|---|
| Load Case Name | **DL** (또는 LL) |
| Element List | 해당 보 부재 ID |
| Load Type | **Uniform Loads (UNILOAD)** |
| Direction | **Global Z** |
| Projection | NO |
| Value | **−w_line** (kN/m, 하향이므로 음수) |
| x1, x2 | 0, 1 (보 전체 등분포) |

### 9.2 데이터 위치

`tests/benchmark/case6_lshape_loadtables.json`의 `DL_line_loads_kNm` 배열 (DL 78건) 및 `LL_line_loads_kNm` 배열 (LL 78건).

각 항목 예시:
```json
{
  "elem_id": 58,
  "ni": 2, "nj": 8,
  "type": "beam_x",
  "story": 1,
  "x_i": 0, "y_i": 0, "z_i": 3.5,
  "x_j": 6, "y_j": 0, "z_j": 3.5,
  "length_m": 6.0,
  "trib_w_m": 2.0,
  "w_line_kNm": 5.1,
  "total_kN": 30.6
}
```
→ Midas에선 **elem_id=58 부재**에 **Global Z 방향 −5.1 kN/m 등분포** 적용.

### 9.3 빠른 입력 방법

156건을 일일이 클릭하면 매우 비효율적입니다. 두 가지 선택지:

**(A) Excel-link 사용**
1. JSON의 `DL_line_loads_kNm`을 Excel로 변환 (Python 한 줄로 가능: 아래 §15 참고)
2. Midas의 Beam Loads 테이블에서 Excel 셀 붙여넣기 (Midas 2019의 테이블 입력 모드)

**(B) MCT 명령어** (가장 빠름)
- 아래 §15에서 자세히 설명
- 156건 입력을 약 1초에 완료

### 9.4 검증

입력 후 다음 화면에서 합계 확인:
- `Load → Static Loads → Element Beam Loads` 화면에서 DL 케이스 필터링 → 합계 표시
- 예상 합계 (DL): −3,457.8 kN (수직 합계)
- 예상 합계 (LL): −1,695.0 kN

---

## 10. EQX / EQY 절점 횡하중 입력 (각 55건)

**메뉴**: `Load → Static Loads → Nodal Loads`

### 10.1 EQX 입력

`lateral_force_table_kN`에서 각 층별 per_node_kN을 모든 node_ids에 적용:

| 층 | per-node Fx (kN) | 대상 노드 IDs |
|---|---|---|
| 1F | 2.5641 | 2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 60, 64, 68 |
| 2F | 5.1282 | 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 61, 65, 69 |
| 3F | 7.6923 | 4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 62, 66, 70 |
| 4F | 14.8148 | 5, 11, 17, 23, 29, 35, 41, 47, 53 |
| 5F | 18.5185 | 6, 12, 18, 24, 30, 36, 42, 48, 54 |

입력 화면:
- Load Case: **EQX**
- Node List: 위 표의 노드 IDs (층별로 선택 후 일괄 적용)
- FX: per-node 값
- FY, FZ, MX, MY, MZ: 0

### 10.2 EQY 입력

동일한 노드 + 동일한 per-node 값이지만 **FY**에 입력 (FX=0).

| 층 | per-node Fy (kN) |
|---|---|
| 1F | 2.5641 |
| 2F | 5.1282 |
| 3F | 7.6923 |
| 4F | 14.8148 |
| 5F | 18.5185 |

### 10.3 BINDING 룰

**횡하중은 반드시 §7의 slave 노드(물리적 격자 노드)에 적용**해야 합니다. Rigid Diaphragm master 노드(38, 39, 40, 29, 30)에 단일 합력으로 적용하지 마세요. ([spec §6 binding rule](_case6_lshape_spec.md#6-rigid-diaphragm-configuration))

이유: 횡하중이 master 노드 1개에 집중되면 비틀림 분포가 달라집니다. OpenSees runner는 모든 slave 노드에 분배 적용했으므로 Midas도 동일하게 해야 합니다.

### 10.4 검증

EQX 합계: 33.33 + 66.67 + 100 + 133.33 + 166.67 = **500.0 kN** (FX 합)
EQY 동일.

---

## 11. 해석 설정 + 실행

### 11.1 해석 옵션

**메뉴**: `Analysis → Analysis Control` → `Main Control`

| 항목 | 값 |
|---|---|
| Analysis Type | **Static** |
| P-Delta | **OFF** (선형 탄성 해석만) |
| Geometric Nonlinearity | OFF |
| Nonlinear Analysis | OFF |

### 11.2 해석 실행

**메뉴**: `Analysis → Perform Analysis` (또는 F5)

해석 시간: 모델 규모 작아 보통 1~3초 이내. 메시지창에 "Analysis completed" 확인.

해석 실패 시 체크:
- 강막 master 노드가 slave 리스트에 들어가 있지는 않은가
- 부재 끝 노드가 잘못 연결되어 분리된 부재가 없는가
- 지점 13개가 모두 FIXED 6-DOF로 잡혀 있는가

---

## 12. 결과 추출 — 24개 metric

채워야 할 파일: [tests/benchmark/midas_results/case6_lshape.json](../tests/benchmark/midas_results/case6_lshape.json)

### 12.1 반력 합계 / 절점 반력 (DL, LL)

**메뉴**: `Results → Reactions → Reaction Forces/Moments`

| JSON 키 | 어떻게 읽나 |
|---|---|
| `DL Base SumFz (kN)` | DL Case 선택 → Reaction Table에서 FZ 열 합계. 약 **3,457.81 kN** 예상 |
| `DL Reaction ZoneA_corner Fz (kN)` | DL Case → Node 1(0,0,0)의 FZ. 약 **195.51 kN** |
| `DL Reaction SharedBoundary Fz (kN)` | DL Case → Node 13(12,0,0)의 FZ. 약 **269.69 kN** |
| `DL Reaction ZoneB_far Fz (kN)` | DL Case → Node 59(24,0,0)의 FZ. 약 **90.43 kN** |
| `LL Base SumFz (kN)` | LL Case 동일 절차. 약 **1,695.01 kN** |
| `LL Reaction ZoneA_corner Fz (kN)` | LL Case → Node 1 FZ. 약 **95.84 kN** |
| `LL Reaction SharedBoundary Fz (kN)` | LL Case → Node 13 FZ. 약 **132.20 kN** |
| `LL Reaction ZoneB_far Fz (kN)` | LL Case → Node 59 FZ. 약 **44.33 kN** |

> Midas의 반력은 **지점 반력이 하중 방향과 반대**로 잡힙니다. 수직 하중이 −Z 방향이면 반력 FZ는 +. OpenSees runner는 양수 RZ로 보고하므로 그대로 입력하면 됩니다.

### 12.2 수직 처짐 (DL, LL)

**메뉴**: `Results → Deformations → Displacements`

| JSON 키 | 어떻게 읽나 |
|---|---|
| `DL ZoneB 3F Far Corner dz (mm)` | DL Case → Node 62(24,0,10.5)의 DZ. 약 **−0.26 mm** |
| `LL ZoneB 3F Far Corner dz (mm)` | LL Case → Node 62 DZ. 약 **−0.13 mm** |

### 12.3 횡변위 (EQX, EQY)

| JSON 키 | 어떻게 읽나 |
|---|---|
| `EQX ZoneA Far 5F dx (mm)` | EQX Case → Node 42(0,8,17.5)의 DX. 약 **37.4 mm** |
| `EQX ZoneA Far 5F dy (mm)` | EQX Case → Node 42 DY. 약 **3.1 mm** |
| `EQY ZoneA Far 5F dx (mm)` | EQY Case → Node 42 DX. 약 **7.2 mm** |
| `EQY ZoneA Far 5F dy (mm)` | EQY Case → Node 42 DY. 약 **77.2 mm** |

### 12.4 층간변위각 envelope

**메뉴**: `Results → Story → Story Drift` (또는 Results → Results-Story → Drift)

| JSON 키 | 어떻게 읽나 |
|---|---|
| `EQX Max StoryDrift X (ratio)` | EQX Case의 모든 층에 대해 dx_top - dx_bot / h, 최대값. 약 **0.00248** |
| `EQY Max StoryDrift Y (ratio)` | EQY Case의 모든 층에 대해 dy_top - dy_bot / h, 최대값. 약 **0.00571** |

> Midas의 Story Drift는 보통 `drift = (top - bottom) / h` 단위로 계산됩니다. 부호 제거(절댓값)된 envelope를 사용하세요.

### 12.5 베이스 전단력

| JSON 키 | 어떻게 읽나 |
|---|---|
| `EQX Base Shear Fx (kN)` | EQX Case → Reaction Table의 FX 합계. 약 **−500.0 kN** (반력이라 부호 −) |
| `EQY Base Shear Fy (kN)` | EQY Case → FY 합계. 약 **−500.0 kN** |

### 12.6 비틀림 진단

| JSON 키 | 어떻게 읽나 |
|---|---|
| `EQX Torsion (A_5F dx - B_3F dx) (mm)` | EQX: Node 42 DX − Node 70(24,4,10.5) DX. 약 **14.6 mm** |
| `EQY Torsion (A_5F dy - B_3F dy) (mm)` | EQY: Node 42 DY − Node 70 DY. 약 **55.6 mm** |

### 12.7 부재력 (기둥 모멘트)

**메뉴**: `Results → Forces → Beam Forces/Moments`

#### 모서리 기둥 1F (위치: x=0, y=0, z=0→3500)

해당 기둥의 i-단(z=0, base) 모멘트 읽기:

| JSON 키 | 어떻게 읽나 |
|---|---|
| `EQX CornerCol1F Base My (kNm)` | EQX Case → 기둥 부재(1번 또는 해당 ID)의 My (i-end). 약 **76.83 kNm** |
| `EQX CornerCol1F Base Mz (kNm)` | EQX Case → Mz (i-end). 약 **3.78 kNm** |

#### Setback 경계 기둥 3F→4F (위치: x=12, y=4, z=10500→14000)

해당 기둥의 j-단(z=14000, upper end) 모멘트:

| JSON 키 | 어떻게 읽나 |
|---|---|
| `EQX SetbackCol 3F->4F UpperEnd My (kNm)` | EQX → 해당 기둥의 My (j-end). 약 **−29.09 kNm** |
| `EQX SetbackCol 3F->4F UpperEnd Mz (kNm)` | EQX → Mz (j-end). 약 **−0.88 kNm** |

> **부호 주의**: Midas의 부재 로컬축이 OpenSees와 다를 수 있습니다. My, Mz의 부호가 반대로 나오면 입력 시 부호를 OpenSees 기준으로 맞춰주세요. 단, 비교에선 절댓값 차이가 중심이라 부호 일치는 보조 확인.

### 12.8 OpenSees 측 예상값 (참고)

`tests/benchmark/opensees_results/case6_lshape.json`에 이미 저장되어 있습니다. Midas 값과 비교하기 전에 OpenSees 값을 미리 봐두면 어느 정도 차이가 예상되는지 가늠할 수 있습니다 — 단, **이걸 보고 Midas 입력을 맞추지 마세요** (decision-gate 의미 사라짐).

---

## 13. JSON 채우기 및 비교 실행

### 13.1 JSON 편집

`tests/benchmark/midas_results/case6_lshape.json`의 각 키에 §12에서 읽은 값을 입력. 예시:

```json
{
  "DL Base SumFz (kN)": 3457.8,
  "DL Reaction ZoneA_corner Fz (kN)": 195.5,
  ...
}
```

`null`을 실수치로 교체. 모든 24개 키를 채우거나, 일부만 채워도 됨 (남은 건 PENDING으로 표시).

### 13.2 비교 실행

```powershell
cd D:\son\opensees-MCP
python tests/benchmark/run_benchmarks.py case6_lshape
```

출력 예시:
```
Metric                                       Unit    OpenSees     Midas    Diff%   Status
DL Base SumFz (kN)                            kN     3457.81     3457.8    0.000%      OK
...
Total: 24 | OK: 20 | CHECK: 3 | FAIL: 1 | PENDING: 0
```

### 13.3 Decision Gate 판정 (Step 5)

| Scenario | 조건 | 결과 |
|---|---|---|
| **A (긍정)** | OK ≥ 80% AND FAIL = 0 | 논문 §3에 Case 6 통합 가능 (별도 plan 작성) |
| **B (부정)** | FAIL ≥ 3 OR systemic CHECK 패턴 | 논문 변경 없음, internal record만 |
| **C (부분)** | FAIL 1~2 OR borderline | Ablation 추가 후 재평가 (강막 ON/OFF, setback 효과 등) |

---

## 14. 자주 묻는 문제

### Q1. Midas Section DB의 단면 물성이 spec과 다릅니다.
A. User-Defined로 H, B, tw, tf만 입력하면 A, Iy, Iz는 자동 계산되며 spec과 거의 일치합니다 (Midas는 정확한 적분식 사용). J는 ±5% 정도 차이날 수 있으나 본 검증에선 무시 가능.

### Q2. Story Diaphragm 자동 인식이 4F에서 Zone B 노드를 포함했습니다.
A. `Boundary → Story Data`에서 4F의 강막 정의를 Modify → Zone B 노드 제외. 더 안전한 방법은 **Story Data 사용 안 하고 Rigid Link로 직접 정의** (§7).

### Q3. 반력 합계가 −3458 kN인데 spec엔 +3458이라 합니다.
A. Midas 반력은 지점 반응이라 하중과 반대 부호. DL은 −Z 방향이므로 반력 FZ는 **+**. spec과 OpenSees 모두 +3458 보고. Midas에서 **−**로 나오면 부호 반전해서 JSON에 입력.

### Q4. 베이스 전단력이 +500인데 spec에선 −500이라 합니다.
A. OpenSees는 횡하중 +500 입력 시 반력을 −500으로 보고 (반대 부호 협약). Midas가 +500으로 나오면 부호 반전해서 입력. **diff_pct 계산은 절댓값 기준**이라 부호와 무관하게 0%로 잡힙니다.

### Q5. 강막 master 위치가 Zone A 모서리(0,8)인 게 어색합니다.
A. OpenSees runner의 자동 선택입니다 (정렬된 노드 ID의 중앙 인덱스). 강막은 master 위치에 관계없이 동일 응답이라 어느 노드를 master로 잡아도 됩니다. 검증 일관성을 위해 같은 위치 권장.

### Q6. EQX/EQY 입력 시 Midas Eigenvalue를 거쳐야 하나요?
A. 아니요. 본 검증은 **등가정적 명시 nodal 하중**입니다. Eigenvalue → 응답스펙트럼 → 자동 분배가 아닌 직접 입력. Static Load Case로 직접 절점하중 적용.

---

## 15. (선택) MCT 명령어로 일괄 입력

UI로 156건의 보 선하중 + 110건의 절점하중을 입력하는 게 부담스럽다면, MCT(Midas Command Text) 일괄 입력을 추천합니다.

### 15.1 MCT란?

Midas Gen이 지원하는 텍스트 기반 입력 형식. 노드/요소/하중/지점 등 모든 입력을 한 텍스트 파일로 작성한 뒤 `File → Import → Midas MCT File`로 일괄 import.

### 15.2 자동 생성 스크립트

원하시면 `tests/benchmark/case6_lshape_loadtables.json`을 읽어 Midas 2019용 MCT 파일을 자동 생성하는 Python 스크립트를 만들어 드릴 수 있습니다:

```
scripts/generate_case6_mct.py
  → scripts/case6_lshape_input.mct
```

생성될 MCT 내용:
- `*MATERIAL` SS275 (E=205000)
- `*SECTION` H-300×300, H-400×200
- `*NODE` 70개 좌표
- `*ELEMENT` 135개 부재
- `*CONSTRAINT` 13개 base FIXED
- `*RIGIDLINK` 5개 강막
- `*LDCASE` DL/LL/EQX/EQY
- `*BEAMLOAD` DL+LL 156건
- `*CONLOAD` EQX+EQY 110건

이걸 Midas에서 `File → Import → MIDAS GEN MCT File`로 import하면 약 5분 안에 모델 + 모든 하중이 자동 입력됩니다.

> **요청 시 작성해드리겠습니다.** "MCT 자동 생성 해줘" 또는 비슷한 요청 주시면 됩니다.

### 15.3 MCT 미니 예시 (참고용)

```
*VERSION
   2.1.0
*UNIT
   KN, MM, KJ, C
*MATERIAL
   1, USER, SS275, , , , C, 0, , NO
       2.0500e+05, 0.3, 1.2000e-05, 0, 0
*SECTION
   1, DBUSER, H-300x300, CC, 0, 0, 0, 0, 0, 0, 0, YES, NO, H, 0, , 300, 300, 10, 15
   2, DBUSER, H-400x200, CC, 0, 0, 0, 0, 0, 0, 0, YES, NO, H, 0, , 400, 200, 8, 13
*NODE
   1, 0, 0, 0
   2, 0, 0, 3500
   ...
*ELEMENT
   1, BEAM, 1, 1, 1, 2, 0, 0
   ...
*CONSTRAINT
   1, 111111
   ...
*RIGIDLINK
   38, "PLATE", 2, 8, 14, 20, 26, 32, 44, 50, 56, 60, 64, 68
   ...
*LOADCASE
   1, DL, USER
   2, LL, USER
   3, EQX, USER
   4, EQY, USER
*USE-STLD, DL
*BEAMLOAD
   58, BEAM, UNILOAD, GZ, NO, 0, 1, 1, 0, -5.1, -5.1, 0, ""
   ...
*USE-STLD, EQX
*CONLOAD
   2, 2.5641, 0, 0, 0, 0, 0
   ...
```

---

## 16. 작업 체크리스트

작업 진행 시 단계별 완료 표시용:

- [ ] 1. 새 프로젝트 + 단위 (kN, mm) + 3D
- [ ] 2. 재료 SS275 E=205000 User-Defined
- [ ] 3. 단면 H-300×300, H-400×200
- [ ] 4. 노드 70개 (ID 1~70)
- [ ] 5. 부재 135개 (기둥 57 + 보 78)
- [ ] 6. 지점 13개 base FIXED
- [ ] 7. 강막 5개 층 (4F/5F Zone B 제외 확인)
- [ ] 8. 하중 케이스 DL, LL, EQX, EQY
- [ ] 9. DL/LL 보 선하중 156건 입력 (합계 검증: DL −3,458 / LL −1,695)
- [ ] 10. EQX/EQY 절점하중 110건 입력 (합계 검증: 500 / 500)
- [ ] 11. Analysis Control: Static, P-Delta OFF
- [ ] 12. Perform Analysis 성공
- [ ] 13. 24개 metric을 `midas_results/case6_lshape.json`에 입력
- [ ] 14. `python tests/benchmark/run_benchmarks.py case6_lshape` 실행 → OK/CHECK/FAIL 분포 확인
- [ ] 15. Decision Gate 판정 (A/B/C)

---

**문서 끝.** 입력 중 막히는 부분이나 Midas 화면에서 못 찾는 항목 있으면 알려주세요. MCT 자동 생성도 원하시면 바로 만들어드립니다.
