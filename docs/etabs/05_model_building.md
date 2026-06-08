# 05. 모델 빌드 헬퍼 — 코드로 ETABS 모델 만들기

> [← 04. ETABSClient API 워크스루](04_etabs_api_walkthrough.md) | [다음: 06. 확장 레시피 →](06_extending_recipes.md)
>
> 코드: [tests/benchmark/etabs_benchmark_case1_2.py](../../tests/benchmark/etabs_benchmark_case1_2.py)

지금까지 본 `etabs_api.py`는 **결과 추출** 위주였습니다. 이 장에서는 ETABS COM API로 **노드/재료/단면/지점/하중**을 직접 생성하는 방법을 봅니다. 벤치마크 케이스 1·2의 helper 함수를 해부합니다.

---

## 1. 왜 helper 함수로 분리했나

ETABS COM API 호출은 시그니처가 복잡하고 ret 코드 체크가 반복됩니다. 매번 풀어서 쓰면 케이스 함수가 길어지므로, **재사용 가능한 helper로 추출**한 패턴입니다.

```
케이스 함수 (run_case1_etabs, run_case2_etabs)
   │
   ▼
helper 함수들 (_init, _material, _isection, _pt, _frame, _restrain, _load_pattern, _joint_load, _run, _select_case)
   │
   ▼
ETABS COM API (model.PropMaterial.SetMaterial 등)
```

helper는 모두 module-level 함수(밑줄 prefix)로, `etabs_api.py`의 클래스 메서드가 **아님**에 주의. 필요하면 별도 모듈(`etabs_model_builder.py`)로 승격할 수 있습니다 (06장 레시피).

---

## 2. helper 카탈로그

| # | helper | 호출하는 ETABS API | 역할 |
|---|--------|---------------------|------|
| 1 | `_init(model)` | `InitializeNewModel`, `SetPresentUnits` | 빈 모델 + 단위 |
| 2 | `_material(model)` | `PropMaterial.SetMaterial`, `PropMaterial.SetMPIsotropic` | SS275 재료 정의 |
| 3 | `_isection(model, name, h, bf, tf, tw)` | `PropFrame.SetISection` | H/I 단면 등록 |
| 4 | `_pt(model, name, x, y, z)` | `PointObj.AddCartesian` | 노드 생성 |
| 5 | `_frame(model, name, i, j, sec)` | `FrameObj.AddByPoint` | 프레임 요소 생성 |
| 6 | `_restrain(model, node, dofs)` | `PointObj.SetRestraint` | 지점 조건 |
| 7 | `_load_pattern(model, lp)` | `LoadPatterns.Add` | 하중 패턴 정의 |
| 8 | `_joint_load(model, node, lp, forces)` | `PointObj.SetLoadForce` | 절점 하중 |
| 9 | `_run(model)` | `Analyze.RunAnalysis` | 해석 실행 |
| 10 | `_select_case(model, case)` | `Setup.DeselectAll...`, `Setup.SetCaseSelectedForOutput` | 결과 출력 케이스 선택 |

---

## 3. 빌드 순서 (꼭 이 순서)

ETABS는 의존성 있는 객체부터 정의해야 합니다.

```
1) _init           ← 항상 첫 호출
2) _material       ← 단면이 재료를 참조
3) _isection       ← 프레임이 단면을 참조
4) _pt             ← 프레임이 노드를 참조
5) _frame          ← 노드와 단면이 있어야 함
6) _restrain       ← 노드가 있어야 함
7) _load_pattern   ← 하중이 패턴을 참조
8) _joint_load     ← 노드와 패턴이 있어야 함
9) _run            ← 모델이 완성된 후
10) _select_case   ← 결과 추출 직전
```

이 순서를 거스르면 ret≠0이 나거나 모델이 비정상.

---

## 4. helper 코드 해부

### 4.1. `_init` — 모델 초기화

[etabs_benchmark_case1_2.py:50-53](../../tests/benchmark/etabs_benchmark_case1_2.py#L50-L53)

```python
def _init(model) -> None:
    """Blank model, set N-mm units."""
    model.InitializeNewModel()
    model.SetPresentUnits(_N_MM_C)   # _N_MM_C = 10
```

- `InitializeNewModel()`: 기존 모델을 지우고 빈 상태로. 항상 첫 호출.
- `SetPresentUnits(10)`: N·mm·°C 단위로. (`etabs_api.py`의 `set_units("N_mm_C")`와 동일)

> 단위는 빌드 중에 바꿔도 되지만, 보통 시작 시 한 번 설정 후 끝까지 같게 유지.

### 4.2. `_material` — 재료 정의

[etabs_benchmark_case1_2.py:56-61](../../tests/benchmark/etabs_benchmark_case1_2.py#L56-L61)

```python
def _material(model) -> None:
    """SS275: E = 210 000 N/mm², ν = 0.3, α = 1.2e-5 /°C."""
    if model.PropMaterial.SetMaterial(_MAT, _STEEL) != 0:
        raise RuntimeError("SetMaterial SS275 failed")
    if model.PropMaterial.SetMPIsotropic(_MAT, 210000.0, 0.3, 1.2e-5) != 0:
        raise RuntimeError("SetMPIsotropic SS275 failed")
```

- `SetMaterial(name, mat_type)`: 재료 이름 + 타입 (`_STEEL = 1` = eMatType_Steel).
- `SetMPIsotropic(name, E, nu, alpha)`: 등방성 재료의 탄성계수·푸아송비·열팽창계수.
- 비등방성, 비선형 재료는 다른 메서드 (`SetMPOrthotropic`, `SetOConcrete`).

> SS275의 E=210 GPa = 210000 N/mm² (현재 단위가 N·mm이므로). 단위가 kN·m이었으면 210000000 입력해야 함.

### 4.3. `_isection` — H/I 단면 등록

[etabs_benchmark_case1_2.py:64-70](../../tests/benchmark/etabs_benchmark_case1_2.py#L64-L70)

```python
def _isection(model, name: str, h: float, bf: float,
              tf: float, tw: float) -> None:
    """Symmetric I-section in mm.  h=depth, bf=flange width,
    tf=flange thickness, tw=web thickness."""
    ret = model.PropFrame.SetISection(name, _MAT, h, bf, tf, tw, bf, tf)
    if ret != 0:
        raise RuntimeError(f"SetISection '{name}' failed (ret={ret})")
```

- `SetISection(name, mat, h, bf_top, tf_top, tw, bf_bot, tf_bot)`: 단면 이름 + 재료 + 치수.
- 본 케이스는 상하 플랜지 동일이므로 `bf, tf` 두 번 반복 (대칭 단면).
- 비대칭 단면은 상·하 플랜지 따로 지정.

**다른 단면 타입**:
| 단면 타입 | ETABS 메서드 |
|-----------|--------------|
| 사각관 | `PropFrame.SetTube` |
| 원형관 | `PropFrame.SetPipe` |
| Channel | `PropFrame.SetChannel` |
| 직사각형 콘크리트 | `PropFrame.SetRectangle` |
| 원형 콘크리트 | `PropFrame.SetCircle` |
| DB 단면 | `PropFrame.ImportProp` |

### 4.4. `_pt` — 노드 생성

[etabs_benchmark_case1_2.py:73-78](../../tests/benchmark/etabs_benchmark_case1_2.py#L73-L78)

```python
def _pt(model, name: str, x: float, y: float, z: float) -> str:
    """Add a joint and return its name."""
    n, ret = model.PointObj.AddCartesian(x, y, z, name)
    if ret != 0:
        raise RuntimeError(f"AddCartesian '{name}' failed (ret={ret})")
    return n
```

- `AddCartesian(x, y, z, name)`: 카테시안 좌표로 노드 생성.
- 반환 (n, ret) — n은 ETABS가 실제로 할당한 이름. 입력 name과 동일하면 그대로 쓰지만 충돌 시 다를 수 있어서 받아서 반환.
- 좌표 단위는 `_init`에서 설정한 단위 (현재 mm).

### 4.5. `_frame` — 프레임 요소 생성

[etabs_benchmark_case1_2.py:81-86](../../tests/benchmark/etabs_benchmark_case1_2.py#L81-L86)

```python
def _frame(model, name: str, i_node: str, j_node: str, sec: str) -> str:
    """Add a frame element, return its assigned name."""
    n, ret = model.FrameObj.AddByPoint(i_node, j_node, name, False, sec)
    if ret != 0:
        raise RuntimeError(f"AddByPoint '{name}' failed (ret={ret})")
    return n
```

- `AddByPoint(i, j, name, propIsAdvanced, sec)`:
  - `i_node`, `j_node`: 노드 이름
  - `name`: 요소 이름 (ETABS가 자동 부여하기를 원하면 빈 문자열)
  - `propIsAdvanced=False`: section property가 단순 단면임
  - `sec`: 단면 이름

> **로컬 좌표 회전이 필요하면**: `model.FrameObj.SetLocalAxes(name, angle)` 호출. 기본값은 ETABS가 자동 결정 (수평 보면 local-2 = +Z 등).

### 4.6. `_restrain` — 지점 조건

[etabs_benchmark_case1_2.py:89-93](../../tests/benchmark/etabs_benchmark_case1_2.py#L89-L93)

```python
def _restrain(model, node: str, dofs: list) -> None:
    """Apply restraints.  dofs = [U1, U2, U3, R1, R2, R3], True = fixed."""
    _, ret = model.PointObj.SetRestraint(node, dofs)
    if ret != 0:
        raise RuntimeError(f"SetRestraint '{node}' failed (ret={ret})")
```

- `SetRestraint(node, dofs)`: 6개 DOF (U1, U2, U3, R1, R2, R3) 구속 여부.
- True = 고정, False = 자유.

**전형적인 패턴**:
| 지점 | dofs |
|------|------|
| Pin (전 변위 구속, 회전 자유) | `[True, True, True, False, False, False]` |
| Roller (한 방향 변위만 자유) | `[False, True, True, False, False, False]` |
| Fixed (전부 구속) | `[True] * 6` |
| Free | `[False] * 6` |

[Case 1 단순보 예](../../tests/benchmark/etabs_benchmark_case1_2.py#L209-L210)
```python
_restrain(m, "N1", [True,  True,  True,  False, False, False])  # pin
_restrain(m, "N3", [False, True,  True,  False, False, False])  # roller
```

### 4.7. `_load_pattern` — 하중 패턴

[etabs_benchmark_case1_2.py:96-100](../../tests/benchmark/etabs_benchmark_case1_2.py#L96-L100)

```python
def _load_pattern(model, lp: str) -> None:
    """Add a load pattern (type = Other, no self-weight)."""
    ret = model.LoadPatterns.Add(lp, _LP_OTHER, 0.0, True)
    if ret != 0:
        raise RuntimeError(f"LoadPatterns.Add '{lp}' failed (ret={ret})")
```

- `Add(name, type, self_weight_multiplier, create_load_case)`:
  - `type = 8` (eLoadPatternType_Other) — 기타 하중
  - `self_weight_multiplier = 0.0` — 자중 미포함
  - `create_load_case = True` — 동일 이름의 하중 케이스 자동 생성

**ETABS 하중 패턴 타입** (자주 쓰는 것만):
| 타입 | enum | 의미 |
|------|------|------|
| Dead | 1 | 고정하중 |
| SuperDead | 2 | 후고정하중 |
| Live | 3 | 활하중 |
| Quake | 5 | 지진 |
| Wind | 6 | 풍 |
| Snow | 7 | 적설 |
| Other | 8 | 기타 |

### 4.8. `_joint_load` — 절점 하중

[etabs_benchmark_case1_2.py:103-107](../../tests/benchmark/etabs_benchmark_case1_2.py#L103-L107)

```python
def _joint_load(model, node: str, lp: str, forces: list) -> None:
    """Apply joint load.  forces = [F1, F2, F3, M1, M2, M3] in N / N·mm."""
    _, ret = model.PointObj.SetLoadForce(node, lp, forces)
    if ret != 0:
        raise RuntimeError(f"SetLoadForce '{node}' failed (ret={ret})")
```

- `SetLoadForce(node, lp, forces)`: 절점에 6성분 하중.
- `forces = [F1, F2, F3, M1, M2, M3]` 전역 좌표 기준.
- 단위는 현재 N·mm 단위 → 힘은 N, 모멘트는 N·mm.

[Case 1 예](../../tests/benchmark/etabs_benchmark_case1_2.py#L213)
```python
_joint_load(m, "N2", "CASE1", [0.0, 0.0, -60000.0, 0.0, 0.0, 0.0])
# N2에 -60000 N (=−60 kN) 수직 하향
```

**프레임에 분포 하중을 주려면**:
`model.FrameObj.SetLoadDistributed(frame, lp, type, dir, dist1, dist2, val1, val2, ...)` 사용. helper에는 없지만 ETABS API에 존재.

### 4.9. `_run` — 해석 실행

[etabs_benchmark_case1_2.py:110-112](../../tests/benchmark/etabs_benchmark_case1_2.py#L110-L112)

```python
def _run(model) -> None:
    if model.Analyze.RunAnalysis() != 0:
        raise RuntimeError("RunAnalysis failed")
```

`etabs_api.py`의 `run_analysis()`와 동일.

### 4.10. `_select_case` — 출력 케이스 선택

[etabs_benchmark_case1_2.py:115-119](../../tests/benchmark/etabs_benchmark_case1_2.py#L115-L119)

```python
def _select_case(model, case: str) -> None:
    """Select a single load case for results output."""
    setup = model.Results.Setup
    setup.DeselectAllCasesAndCombosForOutput()
    setup.SetCaseSelectedForOutput(case, True)
```

02장 6절에서 본 그 패턴. 결과 추출 전 필수.

---

## 5. 완성된 케이스 — Case 1 단순보

helper를 어떻게 조합하는지 보세요.

[etabs_benchmark_case1_2.py:183-232](../../tests/benchmark/etabs_benchmark_case1_2.py#L183-L232)

```python
def run_case1_etabs(client) -> dict:
    """3-node simple beam, 6 m span, 60 kN point load at midspan.

    Layout (X-axis, Z=0):
        N1(0,0,0) —E1— N2(3000,0,0) —E2— N3(6000,0,0)
    BC:   pin at N1  [U1/U2/U3 fixed, R free]
          roller at N3 [U2/U3 fixed, U1/R free]
    Load: Fz = −60 000 N at N2
    """
    m = client.model
    _init(m)
    _material(m)
    _isection(m, "H400x200", 400.0, 200.0, 13.0, 8.0)

    _pt(m, "N1", 0.0,    0.0, 0.0)
    _pt(m, "N2", 3000.0, 0.0, 0.0)
    _pt(m, "N3", 6000.0, 0.0, 0.0)

    _frame(m, "E1", "N1", "N2", "H400x200")
    _frame(m, "E2", "N2", "N3", "H400x200")

    _restrain(m, "N1", [True,  True,  True,  False, False, False])  # pin
    _restrain(m, "N3", [False, True,  True,  False, False, False])  # roller

    _load_pattern(m, "CASE1")
    _joint_load(m, "N2", "CASE1", [0.0, 0.0, -60000.0, 0.0, 0.0, 0.0])

    _run(m)
    _select_case(m, "CASE1")

    _, _, u3_N2, _, _, _ = _displ(m, "N2")
    f1_N1, _, f3_N1, _, _, _ = _react(m, "N1")
    _, _, f3_N3, _, _, _ = _react(m, "N3")

    m3_mid = _m3_at(m, "E1", 3000.0)

    return {
        "midspan_disp_mm":    u3_N2,
        "midspan_moment_kNm": m3_mid / 1e6,
        "reaction_N1_Fy_kN":  f3_N1 / 1000.0,
        "reaction_N3_Fy_kN":  f3_N3 / 1000.0,
        "reaction_N1_Fx_kN":  f1_N1 / 1000.0,
    }
```

이 함수 하나에 모든 단계가 들어있습니다.
1. 모델 초기화 → 재료/단면 정의
2. 노드 3개, 프레임 2개 생성
3. 핀/롤러 지점
4. 하중 패턴 + 절점 하중
5. 해석 + 결과 출력 케이스 선택
6. 결과 추출 (변위, 반력, 모멘트) + 단위 변환

---

## 6. 결과 추출 보조 함수 — `_displ`, `_react`, `_m3_at`

벤치마크는 `etabs_api.py`의 `get_joint_displacements` 대신 더 가벼운 보조 함수를 씁니다.

### `_displ` — JointDispl 직접 호출

[etabs_benchmark_case1_2.py:126-136](../../tests/benchmark/etabs_benchmark_case1_2.py#L126-L136)

```python
def _displ(model, node: str) -> tuple:
    """Return (U1, U2, U3, R1, R2, R3) for the first result row (mm / rad)."""
    (n, obj, elm, lc, st, sn,
     u1, u2, u3, r1, r2, r3, ret) = model.Results.JointDispl(
        node, 0, 0, [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"JointDispl '{node}' failed (ret={ret})")
    def _f(arr):
        return float(list(arr)[0]) if arr else 0.0
    return _f(u1), _f(u2), _f(u3), _f(r1), _f(r2), _f(r3)
```

- 정적 해석 결과는 1행만 → `[0]` 꺼냄.
- 시간이력은 여러 행이지만 본 벤치마크에서는 정적만.

### `_react` — JointReact

[etabs_benchmark_case1_2.py:139-149](../../tests/benchmark/etabs_benchmark_case1_2.py#L139-L149)

`_displ`과 같은 패턴, 메서드만 `JointReact`.

### `_m3_at` — FrameForce + station 매칭

[etabs_benchmark_case1_2.py:152-176](../../tests/benchmark/etabs_benchmark_case1_2.py#L152-L176)

```python
def _m3_at(model, elem: str, target_sta: float) -> float:
    """M3 (N·mm) at the nearest output station to target_sta (mm from i-end)."""
    (n, obj, obj_sta, elm, elm_sta, lc, st, sn,
     p, v2, v3, t, m2, m3, ret) = model.Results.FrameForce(
        elem, 0, 0, [], [], [], [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"FrameForce '{elem}' failed (ret={ret})")

    sta  = list(obj_sta) if obj_sta else []
    m3l  = list(m3)      if m3      else []
    if not sta:
        return 0.0

    idx  = min(range(len(sta)), key=lambda i: abs(sta[i] - target_sta))
    dist = abs(sta[idx] - target_sta)
    if dist > 100.0:
        print(f"  WARNING: '{elem}' nearest station {sta[idx]:.1f} mm is "
              f"{dist:.1f} mm from target {target_sta:.1f} mm")
    return float(m3l[idx])
```

- ETABS는 부재마다 여러 station에서 결과를 줌 (기본 5개).
- 우리가 원하는 위치 (`target_sta`)와 가장 가까운 station을 선택.
- 100mm 이상 차이나면 경고. 정밀 비교가 필요하면 `PropFrame.SetOutputStations`로 station 수 증가.

---

## 7. Case 2 — 1층 1경간 포탈프레임

좀 더 복잡한 케이스. 두 개 단면 (보·기둥), 4 노드, 3 부재, 횡력 + 수직력.

[etabs_benchmark_case1_2.py:239-320](../../tests/benchmark/etabs_benchmark_case1_2.py#L239-L320) 참조.

```python
def run_case2_etabs(client) -> dict:
    m = client.model
    _init(m)
    _material(m)
    _isection(m, "H400x200", 400.0, 200.0, 13.0,  8.0)  # beam
    _isection(m, "H350x350", 350.0, 350.0, 19.0, 12.0)  # column

    _pt(m, "N1", 0.0,    0.0, 0.0)
    _pt(m, "N2", 6000.0, 0.0, 0.0)
    _pt(m, "N3", 0.0,    0.0, 3000.0)
    _pt(m, "N4", 6000.0, 0.0, 3000.0)

    _frame(m, "C1", "N1", "N3", "H350x350")  # column 1
    _frame(m, "C2", "N2", "N4", "H350x350")  # column 2
    _frame(m, "B1", "N3", "N4", "H400x200")  # beam

    _restrain(m, "N1", [True] * 6)
    _restrain(m, "N2", [True] * 6)

    _load_pattern(m, "CASE2")
    _joint_load(m, "N3", "CASE2", [25000.0, 0.0, -100000.0, 0.0, 0.0, 0.0])
    _joint_load(m, "N4", "CASE2", [25000.0, 0.0, -100000.0, 0.0, 0.0, 0.0])

    _run(m)
    _select_case(m, "CASE2")
    # ... 결과 추출
```

helper의 재사용성이 보입니다. 더 큰 모델도 같은 패턴으로 확장.

---

## 8. helper를 클래스 메서드로 승격하기

벤치마크 helper들이 유용하다 싶으면 `etabs_api.py`의 `ETABSClient`에 메서드로 추가할 수 있습니다.

```python
class ETABSClient:
    # ...

    def add_material(self, name: str, E: float, nu: float, alpha: float,
                     mat_type: int = 1):  # 1 = Steel
        if self.model.PropMaterial.SetMaterial(name, mat_type) != 0:
            raise RuntimeError(f"SetMaterial '{name}' failed")
        if self.model.PropMaterial.SetMPIsotropic(name, E, nu, alpha) != 0:
            raise RuntimeError(f"SetMPIsotropic '{name}' failed")

    def add_i_section(self, name: str, mat: str,
                       h: float, bf: float, tf: float, tw: float):
        ret = self.model.PropFrame.SetISection(name, mat, h, bf, tf, tw, bf, tf)
        if ret != 0:
            raise RuntimeError(f"SetISection '{name}' failed (ret={ret})")

    # ... (add_point, add_frame, set_restraint, add_load_pattern, add_joint_load)
```

이렇게 하면 `client.add_material(...)`처럼 객체지향적으로 호출 가능. 06장 레시피 C에서 자세히 다룹니다.

---

## 9. 다음 단계

이제 결과 추출과 모델 빌드를 모두 이해했습니다. 06장에서는 **여러분이 직접 새 기능을 추가**하는 레시피를 봅니다.

> [← 04. ETABSClient API 워크스루](04_etabs_api_walkthrough.md) | [다음: 06. 확장 레시피 →](06_extending_recipes.md)
