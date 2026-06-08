# 03. 단위와 부호규약 — 결과를 올바르게 읽기

> [← 02. comtypes 호출 패턴](02_comtypes_patterns.md) | [다음: 04. ETABSClient API 워크스루 →](04_etabs_api_walkthrough.md)
>
> 코드: [mcp-server/core/etabs_api.py](../../mcp-server/core/etabs_api.py), [tests/benchmark/etabs_benchmark_case1_2.py](../../tests/benchmark/etabs_benchmark_case1_2.py)

ETABS 결과의 숫자는 **단위 설정**과 **로컬 좌표축**을 모르면 절반의 정보입니다. 이 장에서 부호 헷갈리는 일을 끝냅니다.

---

## 1. eUnits — 단위 설정

ETABS는 10가지 단위 조합을 지원하고, `SapModel.SetPresentUnits(eUnits)`로 설정합니다. `etabs_api.py`는 이를 dict로 매핑합니다.

[etabs_api.py:28-40](../../mcp-server/core/etabs_api.py#L28-L40)

```python
UNITS = {
    "lb_in_F":  1,
    "lb_ft_F":  2,
    "kip_in_F": 3,
    "kip_ft_F": 4,
    "kgf_m_C":  5,
    "kN_m_C":   6,   # SI 권장
    "tf_m_C":   7,
    "kN_mm_C":  8,
    "kgf_mm_C": 9,
    "N_mm_C":  10,
}
```

### 어떤 단위를 언제 쓰는가

| 상황 | 추천 단위 | 이유 |
|------|-----------|------|
| 일반 구조해석 (건물, 횡력) | `kN_m_C` (6) | SI, 수치가 인간 친화적 (변위 mm 대신 m, 모멘트 kN·m) |
| 강구조 부재 설계 | `N_mm_C` (10) | KDS·AISC가 N·mm² (= MPa) 응력 단위 |
| 벤치마크 (Midas와 정밀 비교) | `N_mm_C` (10) | Midas와 동일한 단위로 부호 확인 |
| 미국 기준 비교 | `kip_ft_F` (4) | AISC 영문 사례 |

### 단위가 결과에 미치는 영향

`set_units()`를 **결과 추출 직전에** 호출해야 단위가 적용됩니다. 호출 시점이 바뀌면 같은 모델이라도 숫자가 다르게 나옵니다.

[etabs_api.py:202-207](../../mcp-server/core/etabs_api.py#L202-L207)

```python
def set_units(self, unit_key: str = "kN_m_C") -> None:
    val = UNITS.get(unit_key)
    if val is None:
        raise ValueError(f"지원 단위 키: {list(UNITS)}")
    self.model.SetPresentUnits(val)
```

> **벤치마크 코드에서의 단위 처리**: `etabs_benchmark_case1_2.py`는 `_init`에서 `SetPresentUnits(10)` (N·mm)로 빌드/해석을 모두 통일했습니다.

[etabs_benchmark_case1_2.py:50-53](../../tests/benchmark/etabs_benchmark_case1_2.py#L50-L53)

```python
def _init(model) -> None:
    """Blank model, set N-mm units."""
    model.InitializeNewModel()
    model.SetPresentUnits(_N_MM_C)   # _N_MM_C = 10
```

그 다음 결과 변환은 명시적으로 수행:
- 변위: `mm` 그대로
- 모멘트: `m3_mid / 1e6` → `N·mm` → `kN·m`
- 반력: `f3_N1 / 1000.0` → `N` → `kN`

---

## 2. 좌표계 — 전역 vs 로컬

### 전역 좌표계 (Global)

ETABS의 전역 좌표는 일반적으로:
- **X, Y**: 수평면
- **Z**: 연직 상방향 (up)

벤치마크 케이스는 **2D 골조를 X-Z 평면**에 배치합니다.

[etabs_benchmark_case1_2.py:13](../../tests/benchmark/etabs_benchmark_case1_2.py#L13)

```python
# Coordinate system: X = span/bay, Y = out-of-plane, Z = vertical (up)
```

3D 골조는 X·Y·Z를 일반적으로 모두 사용.

### 로컬 좌표계 (Local) — 프레임 요소

프레임 요소(보·기둥)는 **자기 자신의 로컬 좌표 1·2·3축**을 가집니다.

- **local-1**: 부재 축방향 (i-end → j-end)
- **local-2, local-3**: 단면 평면 내 (휨축)

부재력은 로컬 좌표 기준으로 출력됩니다.

| 출력 키 | 의미 |
|---------|------|
| `P` | local-1 방향 축력 |
| `V2` | local-2 방향 전단 |
| `V3` | local-3 방향 전단 |
| `T` | local-1 축 비틀림 |
| `M2` | local-2축 회전 모멘트 (V3와 짝) |
| `M3` | local-3축 회전 모멘트 (V2와 짝) |

`M3`이 일반적인 **강축 휨 모멘트**입니다 (단면이 H/I형강일 때).

### 보·기둥의 local-2/3 정의

ETABS 기본값으로:

| 부재 방향 | local-1 | local-2 | local-3 |
|-----------|---------|---------|---------|
| 보 (+X 방향) | +X | **+Z** | **−Y** |
| 기둥 (+Z 방향) | +Z | **+X** | **+Y** |

[etabs_benchmark_case1_2.py:16-21](../../tests/benchmark/etabs_benchmark_case1_2.py#L16-L21)

```python
# Sign convention — ETABS local axes vs Midas 2D convention:
#   Horizontal beam along +X:  local2 = +Z, local3 = -Y
#     → M3 at j-end is NEGATIVE for sagging  ✓  matches Midas
#   Vertical column along +Z:  local2 = +X, local3 = +Y
#     → M3 at base is NEGATIVE for +X lateral  ✓  matches Midas
#   JointReact M2 (about global Y) = in-plane restraint moment  ≡  Midas Mz
```

### M3 부호의 의미

**보 (+X 방향, local-3 = −Y)**에서:
- 단순보 + 중앙 하향 하중 → 처짐 아래로 → **하부 인장 (sagging)** → `M3 < 0` ✓

**기둥 (+Z 방향, local-3 = +Y)**에서:
- 베이스에서 수평 +X 외력 → 기둥이 휘는 방향 → **베이스에서 M3 < 0** ✓

> **부호가 헷갈릴 때 가장 확실한 검증**: 단순보 케이스(Case 1)로 분석해서 M3 부호와 처짐 부호를 확인. 그 다음 더 복잡한 모델로 확장.

---

## 3. JointReact 좌표 — 전역 기준

`JointReact`(절점 반력)는 **전역 좌표 기준**으로 출력됩니다.

| 키 | 의미 |
|----|------|
| `F1` | 전역 +X 방향 반력 |
| `F2` | 전역 +Y 방향 반력 |
| `F3` | 전역 +Z 방향 반력 (vertical) |
| `M1` | 전역 X축 회전 |
| `M2` | 전역 Y축 회전 |
| `M3` | 전역 Z축 회전 |

벤치마크에서 2D 골조의 평면내 회전(반력 모멘트)은 **전역 Y축 회전 = M2** 입니다.

[etabs_benchmark_case1_2.py:21](../../tests/benchmark/etabs_benchmark_case1_2.py#L21)

```python
# JointReact M2 (about global Y) = in-plane restraint moment  ≡  Midas Mz
```

| 의미 | ETABS 키 | Midas 키 |
|------|----------|----------|
| 전역 X 반력 | F1 | Fx |
| 전역 Z 반력 (수직) | F3 | **Fy** |
| 평면내 모멘트 | M2 | **Mz** |

> Midas 2D는 X·Y 평면을 사용하므로 ETABS의 X·Z를 X·Y로 매핑해서 비교합니다.

---

## 4. JointDispl 좌표 — 전역 기준

`JointDispl`도 전역 좌표 기준입니다.

| 키 | 의미 |
|----|------|
| `U1` | 전역 X 변위 |
| `U2` | 전역 Y 변위 |
| `U3` | 전역 Z 변위 (vertical) |
| `R1` | X축 회전 |
| `R2` | Y축 회전 |
| `R3` | Z축 회전 |

벤치마크 케이스에서 단순보의 수직 처짐은 `U3` (전역 −Z 방향). Midas의 `dy`와 비교합니다.

[etabs_benchmark_case1_2.py:228-230](../../tests/benchmark/etabs_benchmark_case1_2.py#L228-L230)

```python
return {
    "midspan_disp_mm":    u3_N2,             # U3 (Uz) ≡ Midas dy
    ...
}
```

---

## 5. 부호규약 매트릭스 (요약)

| 항목 | 2D Beam (+X) | 2D Column (+Z) | 3D 일반 |
|------|--------------|----------------|---------|
| local-1 | +X | +Z | i→j 방향 |
| local-2 | +Z | +X | 단면 평면 내 |
| local-3 | −Y | +Y | local-1 × local-2 |
| M3<0 의미 | sagging (하부 인장) | base에 +X 외력 | 단면별 회전 방향 |
| `JointReact` 좌표 | 전역 | 전역 | 전역 |
| `JointDispl` 좌표 | 전역 | 전역 | 전역 |

---

## 6. 단위 변환 치트시트

벤치마크 코드는 `N·mm`로 해석한 뒤 `kN·m`로 변환합니다.

| 변환 | 식 |
|------|-----|
| N → kN | `/ 1000.0` |
| mm → m | `/ 1000.0` |
| N·mm → kN·m | `/ 1e6` (= 1000 × 1000) |
| 모멘트 단위 (N·mm → MPa·mm³) | × 1 (numerically same) |
| 응력 (N/mm² = MPa) | 그대로 |

[etabs_benchmark_case1_2.py:226-232](../../tests/benchmark/etabs_benchmark_case1_2.py#L226-L232)

```python
return {
    "midspan_disp_mm":    u3_N2,
    "midspan_moment_kNm": m3_mid / 1e6,      # N·mm → kN·m
    "reaction_N1_Fy_kN":  f3_N1 / 1000.0,   # N → kN
    "reaction_N3_Fy_kN":  f3_N3 / 1000.0,
    "reaction_N1_Fx_kN":  f1_N1 / 1000.0,
}
```

---

## 7. 흔히 헷갈리는 함정

1. **M3 vs M2 혼동**: 강축 휨은 H형강에서 **M3**. M2는 약축. 단면이 회전돼 있으면 (`PropFrame.SetLocalAxes`) 바뀔 수 있음.
2. **JointReact M2 vs FrameForce M3**: 같은 단어 "모멘트"지만 전자는 전역 Y축, 후자는 local-3축. 2D에서는 우연히 양이 같지만 의미는 다름.
3. **단위 다른 채로 계산**: `set_units("N_mm_C")`로 빌드해놓고 결과를 kN으로 가정하면 1000배 오차.
4. **회전각 단위**: ETABS는 항상 **라디안**. 도(degree)가 아님. 층간변위비는 비율(무차원).
5. **station 좌표**: `FrameForce`의 station은 **i-end에서 거리** (mm 또는 m, set_units에 따름). 0 = i-end, L = j-end.

---

## 8. 다음 단계

단위와 부호규약이 잡혔습니다. 04장에서 11개 메서드를 한 줄씩 보면서 02·03의 패턴을 모두 적용해 봅니다.

> [← 02. comtypes 호출 패턴](02_comtypes_patterns.md) | [다음: 04. ETABSClient API 워크스루 →](04_etabs_api_walkthrough.md)
