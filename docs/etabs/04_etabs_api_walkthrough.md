# 04. ETABSClient API 워크스루

> [← 03. 단위와 부호규약](03_units_and_conventions.md) | [다음: 05. 모델 빌드 헬퍼 →](05_model_building.md)
>
> 코드: [mcp-server/core/etabs_api.py](../../mcp-server/core/etabs_api.py)

02·03장에서 익힌 패턴을 적용해서 `etabs_api.py`의 11개 메서드를 모두 분해합니다. 각 메서드마다 **시그니처 → 호출하는 ETABS API → 코드 → 사용 예** 4단계로 정리합니다.

---

## 1. `attach()` — 팩토리 ①

**시그니처**
```python
@classmethod
def attach(cls) -> "ETABSClient":
```

**호출하는 ETABS API**: `cHelper.GetObject("CSI.ETABS.API.ETABSObject")` → `cOAPI.SapModel`

**코드** [etabs_api.py:81-102](../../mcp-server/core/etabs_api.py#L81-L102)

```python
@classmethod
def attach(cls) -> "ETABSClient":
    helper = _make_helper()
    try:
        etabs_obj = helper.GetObject("CSI.ETABS.API.ETABSObject")
    except Exception as e:
        raise RuntimeError(f"실행 중인 ETABS를 찾을 수 없습니다: {e}\n...") from e
    if etabs_obj is None:
        raise RuntimeError("실행 중인 ETABS를 찾을 수 없습니다.\n...")
    sap_model = etabs_obj.SapModel
    return cls(etabs_obj, sap_model)
```

**사용 예**
```python
# ETABS GUI에서 이미 모델을 열어둔 상태
client = ETABSClient.attach()
client.set_units("kN_m_C")
info = client.get_model_info()
```

**언제 쓰나**: 결과만 뽑을 때 (사용자가 GUI에서 모델 준비 → 스크립트가 결과 추출). 자동 배치보다 인터랙티브 작업에 적합.

---

## 2. `launch(model_path, visible)` — 팩토리 ②

**시그니처**
```python
@classmethod
def launch(cls, model_path: Optional[str] = None, visible: bool = True) -> "ETABSClient":
```

**호출하는 ETABS API**: `cHelper.CreateObject(ETABS_EXE)` → `cOAPI.ApplicationStart()` → `SapModel.InitializeNewModel()` → (`SapModel.File.OpenFile(model_path)` 옵션)

**코드** [etabs_api.py:104-137](../../mcp-server/core/etabs_api.py#L104-L137)

```python
@classmethod
def launch(cls, model_path=None, visible=True):
    if not ETABS_EXE.exists():
        raise FileNotFoundError(f"ETABS 실행 파일을 찾을 수 없습니다: {ETABS_EXE}")

    helper = _make_helper()
    try:
        etabs_obj = helper.CreateObject(str(ETABS_EXE))
    except Exception as e:
        raise RuntimeError(f"ETABS 실행 실패: {e}") from e

    etabs_obj.ApplicationStart()
    sap_model = etabs_obj.SapModel
    sap_model.InitializeNewModel()

    if not visible:
        sap_model.SetModelIsLocked(False)

    if model_path:
        ret = sap_model.File.OpenFile(model_path)
        if ret != 0:
            raise RuntimeError(f"모델 파일 열기 실패 (ret={ret}): {model_path}")

    return cls(etabs_obj, sap_model)
```

**사용 예**
```python
# 새 빈 모델
client = ETABSClient.launch()

# 기존 .edb 파일 열기
client = ETABSClient.launch(model_path=r"D:\models\my_model.edb")

# GUI 숨기고 실행 (배치 자동화)
client = ETABSClient.launch(visible=False)
```

**언제 쓰나**: 배치 자동화, 벤치마크 케이스 자동 빌드. `launch()`는 항상 깨끗한 새 인스턴스이므로 재현성이 보장됩니다.

> **`visible=False`의 동작**: ETABS는 일반적으로 항상 GUI를 띄웁니다. `SetModelIsLocked(False)`는 모델 잠금만 해제하는 거고, 완전히 백그라운드 실행은 ETABS API 한계로 어렵습니다. 화면에서 안 보이게 하려면 `MinimizeWindow()`나 ETABS 설정을 별도 조정.

---

## 3. `__enter__` / `__exit__` / `close(save, save_path)` — 컨텍스트 매니저

**코드** [etabs_api.py:143-156](../../mcp-server/core/etabs_api.py#L143-L156)

```python
def __enter__(self):
    return self

def __exit__(self, *_):
    self.close()

def close(self, save: bool = False, save_path: str = ""):
    try:
        if save:
            self.model.File.Save(save_path)
        self._etabs_object.ApplicationExit(False)
    except Exception:
        pass
```

**사용 예**
```python
with ETABSClient.launch() as client:
    client.set_units("kN_m_C")
    client.run_analysis()
    drifts = client.get_story_drifts()
# 블록 종료 시 자동으로 ETABS 종료

# 명시적 저장 후 종료
client.close(save=True, save_path=r"D:\out\model.edb")
```

**왜 `close()`가 예외를 swallow하나**: ETABS가 이미 죽었거나 라이선스 해제됐을 때 `ApplicationExit` 실패가 잦은데, 종료 시점에서는 의미 없음 → `except Exception: pass`.

---

## 4. `set_units(unit_key)` — 단위 설정

**시그니처**
```python
def set_units(self, unit_key: str = "kN_m_C") -> None:
```

**호출하는 ETABS API**: `SapModel.SetPresentUnits(eUnits)`

**코드** [etabs_api.py:202-207](../../mcp-server/core/etabs_api.py#L202-L207)

```python
def set_units(self, unit_key: str = "kN_m_C") -> None:
    val = UNITS.get(unit_key)
    if val is None:
        raise ValueError(f"지원 단위 키: {list(UNITS)}")
    self.model.SetPresentUnits(val)
```

**사용 예**
```python
client.set_units("kN_m_C")   # SI 권장
client.set_units("N_mm_C")   # 벤치마크/응력 단위
```

전체 단위 키: [03장 단위와 부호규약](03_units_and_conventions.md#1-eunits--단위-설정)

---

## 5. `run_analysis()` — 해석 실행

**시그니처**
```python
def run_analysis(self) -> None:
```

**호출하는 ETABS API**: `Analyze.RunAnalysis()`

**코드** [etabs_api.py:213-217](../../mcp-server/core/etabs_api.py#L213-L217)

```python
def run_analysis(self) -> None:
    ret = self.model.Analyze.RunAnalysis()
    if ret != 0:
        raise RuntimeError(f"해석 실행 실패 (ret={ret})")
```

**사용 예**
```python
client.run_analysis()
# 해석이 끝날 때까지 블로킹 (큰 모델은 수분 ~ 수십분)
```

**언제 ret≠0이 나오나**:
- 모델 빌드가 불완전 (지점 없음, 하중 없음)
- 단면/재료 미지정
- 라이선스 만료
- 디스크 공간 부족

---

## 6. `get_base_reactions(load_cases)` — 기저 반력

**시그니처**
```python
def get_base_reactions(self, load_cases: Optional[list] = None) -> dict:
    # → {"load_case": [...], "Fx": [...], ..., "Mz": [...]}
```

**호출하는 ETABS API**:
1. `Setup.DeselectAllCasesAndCombosForOutput()`
2. `LoadCases.GetNameList()` (load_cases=None일 때)
3. `Setup.SetCaseSelectedForOutput(name, True)` (각 케이스)
4. `Results.BaseReact(...)` (13개 in/out)

**코드** [etabs_api.py:223-255](../../mcp-server/core/etabs_api.py#L223-L255)

```python
def get_base_reactions(self, load_cases: Optional[list] = None) -> dict:
    setup = self.model.Results.Setup
    setup.DeselectAllCasesAndCombosForOutput()

    if load_cases is None:
        n, names, _ = self.model.LoadCases.GetNameList(0, [])
        load_cases = list(names) if names else []

    for name in load_cases:
        setup.SetCaseSelectedForOutput(name, True)

    (n, lc, step_type, step_num,
     fx, fy, fz, mx, my, mz,
     gx, gy, gz, ret) = self.model.Results.BaseReact(
        0, [], [], [], [], [], [], [], [], [], 0.0, 0.0, 0.0
    )
    if ret != 0:
        raise RuntimeError(f"기저 반력 추출 실패 (ret={ret})")
    return {
        "load_case": list(lc) if lc else [],
        "Fx": list(fx) if fx else [], "Fy": list(fy) if fy else [], "Fz": list(fz) if fz else [],
        "Mx": list(mx) if mx else [], "My": list(my) if my else [], "Mz": list(mz) if mz else [],
    }
```

**사용 예**
```python
# 전체 케이스
reactions = client.get_base_reactions()

# 특정 케이스만
reactions = client.get_base_reactions(load_cases=["DL", "EQX"])

# 출력 구조
# {
#   "load_case": ["DL", "DL", "EQX", "EQX"],   # 케이스가 여러 번 반복될 수 있음
#   "Fx": [0.0, 0.0, 50.0, 50.0],
#   "Fy": [...], ...
# }
```

**길이가 케이스 수와 다를 수 있는 이유**: 모달/시간이력 케이스는 step별로 한 행씩 나옴. `step_type`과 `step_num`을 함께 보면 어떤 행이 어떤 step인지 파악 가능.

---

## 7. `get_modal_periods()` — 모달 주기

**시그니처**
```python
def get_modal_periods(self) -> list:
    # → [T1, T2, T3, ...] (초)
```

**호출하는 ETABS API**: `Results.ModalPeriod(...)` (8개 in/out)

**코드** [etabs_api.py:257-266](../../mcp-server/core/etabs_api.py#L257-L266)

```python
def get_modal_periods(self) -> list:
    (n, lc, step_type, step_num,
     period, freq, circ_freq, eig_val, ret) = self.model.Results.ModalPeriod(
        0, [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"모달 주기 추출 실패 (ret={ret})")
    return list(period) if period else []
```

**사용 예**
```python
periods = client.get_modal_periods()
# [1.234, 0.987, 0.654, ...]
print(f"1차 주기: {periods[0]:.3f}초")
```

**참고**: `Setup`을 안 호출해도 ETABS가 자동으로 modal case를 사용. 단, modal 해석을 사전에 정의해놔야 함 (`LoadCases.ModalEigen.SetCase`).

---

## 8. `get_story_drifts()` — 층간변위비

**시그니처**
```python
def get_story_drifts(self) -> list:
    # → [{"story": str, "load_case": str, "direction": str, "drift": float}, ...]
```

**호출하는 ETABS API**: `Results.StoryDrifts(...)` (11개 in/out)

**코드** [etabs_api.py:268-291](../../mcp-server/core/etabs_api.py#L268-L291)

```python
def get_story_drifts(self) -> list:
    (n, story, lc, step_type, step_num,
     direction, drift, label, x, y, z, ret) = self.model.Results.StoryDrifts(
        0, [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"층간변위 추출 실패 (ret={ret})")
    if not story:
        return []
    return [
        {
            "story": story[i],
            "load_case": lc[i],
            "direction": direction[i],
            "drift": drift[i],
        }
        for i in range(n)
    ]
```

**사용 예**
```python
drifts = client.get_story_drifts()
# [{"story": "Story1", "load_case": "EQX", "direction": "X", "drift": 0.0023}, ...]

# X방향 EQX 케이스 최대 drift
max_x = max(d["drift"] for d in drifts if d["direction"] == "X" and d["load_case"] == "EQX")
```

**direction 값**: `"X"`, `"Y"`. ETABS가 자동으로 두 방향을 모두 출력.

---

## 9. `get_joint_displacements(joint_name, load_case)` — 절점 6-DOF 변위

**시그니처**
```python
def get_joint_displacements(self, joint_name: str, load_case: str) -> dict:
    # → {"U1": [...], "U2": [...], "U3": [...], "R1": [...], "R2": [...], "R3": [...]}
```

**호출하는 ETABS API**:
1. `Setup.DeselectAllCasesAndCombosForOutput()`
2. `Setup.SetCaseSelectedForOutput(load_case, True)`
3. `Results.JointDispl(joint_name, 0, ...)` (input 2 + in/out 11)

**코드** [etabs_api.py:293-309](../../mcp-server/core/etabs_api.py#L293-L309)

```python
def get_joint_displacements(self, joint_name: str, load_case: str) -> dict:
    setup = self.model.Results.Setup
    setup.DeselectAllCasesAndCombosForOutput()
    setup.SetCaseSelectedForOutput(load_case, True)
    (n, obj, elm, lc_out, step_type, step_num,
     u1, u2, u3, r1, r2, r3, ret) = self.model.Results.JointDispl(
        joint_name, 0,
        0, [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"절점 변위 추출 실패 (ret={ret})")
    return {
        "U1": list(u1) if u1 else [], "U2": list(u2) if u2 else [], "U3": list(u3) if u3 else [],
        "R1": list(r1) if r1 else [], "R2": list(r2) if r2 else [], "R3": list(r3) if r3 else [],
    }
```

**사용 예**
```python
disp = client.get_joint_displacements("N2", "CASE1")
print(f"N2의 수직 변위: {disp['U3'][0]:.3f} mm")
# 정적 해석은 결과가 한 행 ([0]), 시간이력은 여러 행
```

**`ItemTypeElm = 0`** (두 번째 input): 객체 자체에 대한 결과. 1은 그룹, 2는 선택된 객체.

좌표 의미는 [03장 4절](03_units_and_conventions.md#4-jointdispl-좌표--전역-기준) 참조.

---

## 10. `get_frame_forces(frame_name, load_case)` — 부재력

**시그니처**
```python
def get_frame_forces(self, frame_name: str, load_case: str) -> dict:
    # → {"station": [...], "P": [...], "V2": [...], "V3": [...], "T": [...], "M2": [...], "M3": [...]}
```

**호출하는 ETABS API**:
1. `Setup.DeselectAllCasesAndCombosForOutput()`
2. `Setup.SetCaseSelectedForOutput(load_case, True)`
3. `Results.FrameForce(frame_name, 0, ...)` (input 2 + in/out 14)

**코드** [etabs_api.py:311-330](../../mcp-server/core/etabs_api.py#L311-L330)

```python
def get_frame_forces(self, frame_name: str, load_case: str) -> dict:
    setup = self.model.Results.Setup
    setup.DeselectAllCasesAndCombosForOutput()
    setup.SetCaseSelectedForOutput(load_case, True)
    (n, obj, obj_sta, elm, elm_sta, lc_out, step_type, step_num,
     p, v2, v3, t, m2, m3, ret) = self.model.Results.FrameForce(
        frame_name, 0,
        0, [], [], [], [], [], [], [],
        [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"부재력 추출 실패 (ret={ret})")
    return {
        "station":  list(obj_sta) if obj_sta else [],
        "P":  list(p)  if p  else [], "V2": list(v2) if v2 else [],
        "V3": list(v3) if v3 else [], "T":  list(t)  if t  else [],
        "M2": list(m2) if m2 else [], "M3": list(m3) if m3 else [],
    }
```

**사용 예**
```python
forces = client.get_frame_forces("E1", "CASE1")
# 출력 (단순보 6m, 2개 요소, 5 stations per element 가정)
# forces["station"] = [0, 750, 1500, 2250, 3000]   # mm from i-end
# forces["M3"]      = [0, -22.5e6, -45e6, -67.5e6, -90e6]   # N·mm
```

**station이란?**: ETABS가 부재 내부 여러 지점에서 결과를 평가. 기본은 5 station (i-end, 1/4, mid, 3/4, j-end). `PropFrame.SetOutputStations`로 조정 가능.

**M3 부호 의미**: [03장 2절](03_units_and_conventions.md#m3-부호의-의미) 참조.

---

## 11. `get_model_info()` — 모델 요약

**시그니처**
```python
def get_model_info(self) -> dict:
    # → {"n_joints": int, "n_frames": int, "n_areas": int,
    #    "n_load_patterns": int, "load_patterns": list,
    #    "n_stories": int, "stories": list}
```

**호출하는 ETABS API**: 4번의 `GetNameList` + 1번의 `GetStories`.

**코드** [etabs_api.py:336-357](../../mcp-server/core/etabs_api.py#L336-L357)

```python
def get_model_info(self) -> dict:
    n_pts, pts, _ = self.model.PointObj.GetNameList(0, [])
    n_fr,  fr,  _ = self.model.FrameObj.GetNameList(0, [])
    n_ar,  ar,  _ = self.model.AreaObj.GetNameList(0, [])
    n_lp,  lp,  _ = self.model.LoadPatterns.GetNameList(0, [])

    (n_st, st, base_e, heights,
     is_m, sim, sp_a, sp_h, _) = self.model.Story.GetStories(
        0, [], [], [], [], [], [], []
    )

    return {
        "n_joints":        n_pts,
        "n_frames":        n_fr,
        "n_areas":         n_ar,
        "n_load_patterns": n_lp,
        "load_patterns":   list(lp) if lp else [],
        "n_stories":       n_st,
        "stories":         list(st) if st else [],
    }
```

**사용 예**
```python
info = client.get_model_info()
print(f"노드 {info['n_joints']}, 프레임 {info['n_frames']}, 층 {info['n_stories']}")
# 노드 4, 프레임 3, 층 1
```

**디버깅에 유용**: 모델이 제대로 빌드됐는지 빠른 확인. 노드/프레임 수가 0이면 빌드 헬퍼가 실패한 것.

---

## 12. `get_table(table_key)` — 범용 결과 추출

**시그니처**
```python
def get_table(self, table_key: str) -> list[dict]:
    # → [{"field1": value, "field2": value, ...}, ...]
```

**호출하는 ETABS API**: `DatabaseTables.GetTableForDisplayArray(...)` (input 3 + in/out 6)

**코드** [etabs_api.py:363-390](../../mcp-server/core/etabs_api.py#L363-L390)

```python
def get_table(self, table_key: str) -> list[dict]:
    (field_key_list, table_version, fields_included,
     n_records, table_data, ret) = self.model.DatabaseTables.GetTableForDisplayArray(
        table_key, [], "", 0, [], 0, []
    )
    if ret != 0:
        raise RuntimeError(f"테이블 '{table_key}' 추출 실패 (ret={ret})")

    keys = list(fields_included) if fields_included else []
    vals = list(table_data) if table_data else []
    n_f = len(keys)
    if n_f == 0 or n_records == 0:
        return []

    return [
        {keys[j]: vals[i * n_f + j] for j in range(n_f)}
        for i in range(n_records)
    ]
```

**사용 예**
```python
# 층 강성
rows = client.get_table("Story Stiffness")
# [{"Story": "Story1", "OutputCase": "EQX", "StiffX": 1234.5, ...}, ...]

# 모달 질량 참여율
mass_ratios = client.get_table("Modal Participating Mass Ratios")
```

**table_key 어디서 찾나**: `C:\Program Files\Computers and Structures\ETABS 23\NativeAPI\Table and Field Keys.xml`

또는 GUI에서 `Display → Show Tables → ...`에서 보이는 이름이 대부분 그대로 키.

**`table_data`가 1D 배열인 이유**: ETABS는 (n_records × n_fields)를 row-major 1D 배열로 압축해서 보냅니다. 코드에서 `vals[i * n_f + j]`로 인덱싱하여 dict로 복원.

> **`get_table`이 강력한 이유**: ETABS GUI의 모든 결과 테이블에 접근할 수 있어서, 새 전용 메서드를 만들기 전에 임시로 쓰기 좋습니다. 06장 레시피 B에서 자세히 다룸.

---

## 13. 메서드 간 의존 관계

전형적인 호출 순서:

```
ETABSClient.launch() / attach()
       │
       ▼
set_units("kN_m_C")
       │
       ▼
[모델 빌드 — 05장]
       │
       ▼
run_analysis()
       │
       ├──► get_base_reactions()
       ├──► get_modal_periods()
       ├──► get_story_drifts()
       ├──► get_joint_displacements(node, lc)   ── 여러 노드 반복
       ├──► get_frame_forces(elem, lc)          ── 여러 요소 반복
       └──► get_table(custom_key)               ── 임시 결과
       │
       ▼
close()
```

`get_*` 메서드는 모두 `run_analysis()` 이후에만 의미가 있습니다 (해석 결과를 읽기 때문).

---

## 14. 다음 단계

11개 메서드가 모두 02·03장의 패턴을 따른다는 게 보이셨을 겁니다. 다음 장에서는 **결과 추출이 아닌 모델 빌드** 쪽을 다룹니다. Case 1·2 벤치마크 코드를 분해해서 ETABS COM API로 노드·재료·단면·하중을 어떻게 만드는지 봅니다.

> [← 03. 단위와 부호규약](03_units_and_conventions.md) | [다음: 05. 모델 빌드 헬퍼 →](05_model_building.md)
