# 02. comtypes 호출 패턴 — 가장 중요한 한 장

> [← 01. Architecture](01_architecture.md) | [다음: 03. 단위와 부호규약 →](03_units_and_conventions.md)
>
> 코드: [mcp-server/core/etabs_api.py](../../mcp-server/core/etabs_api.py)

이 장의 패턴을 이해하면 04장의 11개 메서드는 전부 같은 모양임이 보입니다. 매뉴얼에서 가장 중요한 한 장입니다.

---

## 1. TLB(Type Library)란 무엇인가

COM(Component Object Model)은 Microsoft가 만든 언어 독립적 컴포넌트 규격입니다. ETABS는 COM 객체로 자기 자신을 노출합니다. 외부 프로그램이 "이 COM 객체에 어떤 메서드가 있고, 어떤 매개변수를 받는지" 알려면 **TLB(Type Library)** 파일을 읽어야 합니다.

ETABS 23은 다음 위치에 TLB를 둡니다.

```
C:\Program Files\Computers and Structures\ETABS 23\NativeAPI\x64\ETABSv1.tlb
```

이 파일에는 `cHelper`, `cOAPI`, `cSapModel`, `cBaseReact` 같은 인터페이스의 정의가 들어있습니다.

[etabs_api.py:24-25](../../mcp-server/core/etabs_api.py#L24-L25)

```python
ETABS_INSTALL_DIR = Path(r"C:\Program Files\Computers and Structures\ETABS 23")
_TLB_PATH = ETABS_INSTALL_DIR / "NativeAPI" / "x64" / "ETABSv1.tlb"
```

---

## 2. comtypes의 역할

Python은 COM을 직접 모릅니다. `comtypes`는 두 가지를 해줍니다.

1. **TLB → Python 모듈 변환** (`GetModule`): TLB를 읽어서 `comtypes.gen.ETABSv1` 모듈을 생성. 이 모듈에 `cSapModel` 같은 인터페이스 클래스가 등록됨.
2. **CoClass → 인스턴스 생성** (`CreateObject`): ETABS COM 객체 인스턴스를 만들고 지정한 인터페이스로 반환.

[etabs_api.py:43-63](../../mcp-server/core/etabs_api.py#L43-L63)

```python
def _get_etabs_lib():
    """comtypes로 ETABSv1 TLB 모듈 로드 (최초 호출 시 gen 캐시 생성)."""
    try:
        import comtypes.client  # noqa: F401
    except ImportError:
        raise ImportError(
            "comtypes가 설치되지 않았습니다. pip install comtypes 를 실행하세요."
        )
    if not _TLB_PATH.exists():
        raise FileNotFoundError(
            f"ETABS TLB를 찾을 수 없습니다: {_TLB_PATH}\n"
            "ETABS 23이 설치되어 있는지 확인하세요."
        )
    return comtypes.client.GetModule(str(_TLB_PATH))


def _make_helper():
    """ETABSv1.Helper CoClass 인스턴스를 cHelper 인터페이스로 반환."""
    import comtypes.client
    lib = _get_etabs_lib()
    return comtypes.client.CreateObject(lib.Helper, interface=lib.cHelper)
```

### gen 캐시

`GetModule()`을 처음 호출하면 `comtypes/gen/ETABSv1_*.py` 같은 파이썬 파일이 자동 생성됩니다. 두 번째 호출부터는 캐시를 재사용해서 빠릅니다.

> **gen 캐시 위치**: `<python_path>\Lib\site-packages\comtypes\gen\`
>
> ETABS를 업데이트하면 gen 캐시를 지우는 게 안전합니다 (TLB 시그니처가 바뀔 수 있음).

### CoClass vs Interface

`lib.Helper`는 **CoClass** (인스턴스를 만들 수 있는 클래스), `lib.cHelper`는 **Interface** (메서드 시그니처만 정의). `CreateObject(coclass, interface=...)`가 둘을 묶어서 "이 CoClass의 인스턴스를 이 인터페이스로 보여줘"라는 의미.

---

## 3. ETABS 객체 진입점 확보 — `attach` vs `launch`

### attach (실행 중인 ETABS에 연결)

[etabs_api.py:81-102](../../mcp-server/core/etabs_api.py#L81-L102)

```python
@classmethod
def attach(cls) -> "ETABSClient":
    helper = _make_helper()
    try:
        etabs_obj = helper.GetObject("CSI.ETABS.API.ETABSObject")
    except Exception as e:
        raise RuntimeError(
            f"실행 중인 ETABS를 찾을 수 없습니다: {e}\n"
            "ETABS를 먼저 열거나 ETABSClient.launch()를 사용하세요."
        ) from e
    if etabs_obj is None:
        raise RuntimeError(...)
    sap_model = etabs_obj.SapModel
    return cls(etabs_obj, sap_model)
```

`helper.GetObject("CSI.ETABS.API.ETABSObject")`는 **이미 실행 중인** ETABS 인스턴스를 찾아서 반환합니다. ETABS GUI에서 모델을 열어두고 외부 스크립트로 결과만 뽑을 때 유용.

### launch (ETABS를 새로 실행)

[etabs_api.py:104-137](../../mcp-server/core/etabs_api.py#L104-L137)

```python
@classmethod
def launch(cls, model_path=None, visible=True) -> "ETABSClient":
    if not ETABS_EXE.exists():
        raise FileNotFoundError(...)

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

`helper.CreateObject(ETABS_EXE)`는 새 ETABS 프로세스를 띄웁니다. 자동화 배치에서는 이 방식이 깔끔.

---

## 4. ★ in/out 매개변수 언패킹 — 핵심 패턴 ★

ETABS COM API의 거의 모든 결과 메서드는 **in/out 매개변수 패턴**을 씁니다. C++에서는 `int* count`처럼 포인터로 받지만, Python comtypes에서는 다음 규칙으로 동작합니다.

### 규칙

1. **input 자리에는 그냥 값을 넣는다.** 예: 노드 이름 `"N1"`, 케이스 인덱스 `0`.
2. **in/out 자리에는 "타입이 맞는 placeholder"를 넣는다.**
   - 정수가 출력될 자리 → `0`
   - 실수가 출력될 자리 → `0.0`
   - 배열(리스트)이 출력될 자리 → `[]` (빈 리스트)
3. **메서드는 모든 in/out 값을 튜플로 반환한다.** 마지막에 정수 `ret` (성공 = 0).
4. **placeholder의 개수와 순서가 TLB 시그니처와 정확히 일치해야 한다.** 하나라도 빠지면 `TypeError`.

### 예제: `BaseReact`

ETABS C# 시그니처:
```csharp
int BaseReact(
    ref int NumberResults,
    ref string[] LoadCase, ref string[] StepType, ref double[] StepNum,
    ref double[] FX, ref double[] FY, ref double[] FZ,
    ref double[] MX, ref double[] MY, ref double[] MZ,
    ref double GX, ref double GY, ref double GZ
);
```

총 13개의 in/out 매개변수. Python에서는 다음과 같이 부릅니다.

[etabs_api.py:243-248](../../mcp-server/core/etabs_api.py#L243-L248)

```python
(n, lc, step_type, step_num,
 fx, fy, fz, mx, my, mz,
 gx, gy, gz, ret) = self.model.Results.BaseReact(
    0, [], [], [], [], [], [], [], [], [], 0.0, 0.0, 0.0
)
```

좌변은 **13개 결과 + 1개 ret = 14개 튜플 언패킹**.
우변은 **13개 placeholder**.

placeholder 순서를 잘 보세요.

| 자리 | C# 매개변수 | placeholder | 반환 변수 |
|------|-------------|-------------|-----------|
| 1 | int NumberResults | `0` | `n` |
| 2 | string[] LoadCase | `[]` | `lc` |
| 3 | string[] StepType | `[]` | `step_type` |
| 4 | double[] StepNum | `[]` | `step_num` |
| 5 | double[] FX | `[]` | `fx` |
| 6 | double[] FY | `[]` | `fy` |
| 7 | double[] FZ | `[]` | `fz` |
| 8 | double[] MX | `[]` | `mx` |
| 9 | double[] MY | `[]` | `my` |
| 10 | double[] MZ | `[]` | `mz` |
| 11 | double GX | `0.0` | `gx` |
| 12 | double GY | `0.0` | `gy` |
| 13 | double GZ | `0.0` | `gz` |
| — | int (return value) | (없음) | `ret` |

### 또 다른 예: `JointDispl` — input + in/out 혼합

`JointDispl`은 **input 두 개** (노드 이름, ItemTypeElm) + **in/out 11개**입니다.

[etabs_api.py:298-303](../../mcp-server/core/etabs_api.py#L298-L303)

```python
(n, obj, elm, lc_out, step_type, step_num,
 u1, u2, u3, r1, r2, r3, ret) = self.model.Results.JointDispl(
    joint_name, 0,            # ← input
    0, [], [], [], [], [], [], [], [], [], []   # ← in/out (placeholder 11개)
)
```

좌변 12개 + ret = 13개 튜플. 우변은 input 2 + placeholder 11 = 13개.

> **꼭 기억할 것**: 좌변 변수의 첫 개수는 (placeholder 개수 - 0 또는 그 외), 마지막은 항상 `ret`. input은 placeholder가 아니므로 카운팅에서 제외.

---

## 5. ret 코드의 의미

ETABS COM API는 **모든 메서드가 정수 `ret`을 반환**합니다.

| ret | 의미 |
|-----|------|
| `0` | 성공 |
| `1` 이상 | 실패 (원인은 메서드별로 다름. 일부는 매뉴얼 명시 안 됨) |

코드에서는 다음 패턴으로 항상 체크합니다.

[etabs_api.py:249-250](../../mcp-server/core/etabs_api.py#L249-L250)

```python
if ret != 0:
    raise RuntimeError(f"기저 반력 추출 실패 (ret={ret})")
```

> **ret != 0의 가장 흔한 원인**: (a) 해석을 안 돌렸다, (b) `Setup.SetCaseSelectedForOutput()`을 안 했다, (c) 노드/요소 이름이 틀렸다, (d) 단위 키가 잘못됐다. 자세한 매핑은 [07장 트러블슈팅](07_troubleshooting.md).

---

## 6. ★ 결과 추출 전 Setup 선택 — 잊으면 빈 배열 ★

`Results.BaseReact`, `JointDispl`, `FrameForce` 같은 결과 메서드는 **출력할 케이스를 먼저 선택**해야 합니다. 안 하면 ret=0이지만 모든 배열이 비어서 돌아옵니다.

[etabs_api.py:232-240](../../mcp-server/core/etabs_api.py#L232-L240)

```python
def get_base_reactions(self, load_cases=None) -> dict:
    setup = self.model.Results.Setup
    setup.DeselectAllCasesAndCombosForOutput()  # ① 전부 해제

    if load_cases is None:
        n, names, _ = self.model.LoadCases.GetNameList(0, [])
        load_cases = list(names) if names else []

    for name in load_cases:
        setup.SetCaseSelectedForOutput(name, True)  # ② 원하는 케이스만 켬
```

이 두 호출(`DeselectAllCasesAndCombosForOutput` → `SetCaseSelectedForOutput`)이 **결과 추출 메서드 6개 모두에 공통**입니다. 빠진 메서드(`get_modal_periods`, `get_story_drifts`)는 ETABS가 자동으로 사용 가능한 케이스를 사용합니다.

> **잊었을 때 증상**: 예외는 안 나는데 결과 dict의 모든 리스트가 `[]`. → 디버깅 1순위: Setup 선택했는지 확인.

---

## 7. 배열 결과 안전 처리

ETABS가 placeholder 자리에 `None`을 돌려주는 경우가 있습니다. `list(None)`은 `TypeError`이므로 다음 패턴으로 방어합니다.

[etabs_api.py:251-255](../../mcp-server/core/etabs_api.py#L251-L255)

```python
return {
    "load_case": list(lc) if lc else [],
    "Fx": list(fx) if fx else [],
    ...
}
```

`if lc else []` — None이거나 빈 튜플이면 `[]`로 대체.

---

## 8. GetNameList — 또 다른 in/out 패턴

`GetNameList`는 객체 이름 리스트를 받는 헬퍼로, **input이 두 개의 placeholder**입니다.

```python
n_pts, pts, _ = self.model.PointObj.GetNameList(0, [])
#                                                 │  └─ 이름 배열 placeholder
#                                                 └──── 개수 placeholder
```

반환:
- `n_pts` (int): 노드 개수
- `pts` (tuple[str]): 노드 이름 튜플
- `_` (int): ret 코드 (성공 = 0)

대부분의 `GetNameList`는 `(0, [])` placeholder를 받고 `(개수, 이름튜플, ret)`을 반환합니다.

`GetStories`는 더 복잡합니다 (8개 in/out).

[etabs_api.py:343-346](../../mcp-server/core/etabs_api.py#L343-L346)

```python
(n_st, st, base_e, heights,
 is_m, sim, sp_a, sp_h, _) = self.model.Story.GetStories(
    0, [], [], [], [], [], [], []
)
```

---

## 9. 정리 — 결과 추출 메서드 작성 4단계

새 결과 추출 메서드를 만든다면 항상 이 순서:

1. **TLB에서 시그니처 확인**: 인자 개수, 타입, in/out 여부.
2. **Setup 선택**: 결과 메서드라면 `DeselectAllCasesAndCombosForOutput` → `SetCaseSelectedForOutput`.
3. **placeholder로 호출**: input은 그대로, in/out은 `0`/`0.0`/`[]`.
4. **튜플 언패킹 + ret 체크 + None 방어**: `list(x) if x else []`로 dict 구성.

이 패턴이 [04장 ETABSClient API 워크스루](04_etabs_api_walkthrough.md)의 11개 메서드를 관통합니다.

---

> [← 01. Architecture](01_architecture.md) | [다음: 03. 단위와 부호규약 →](03_units_and_conventions.md)
