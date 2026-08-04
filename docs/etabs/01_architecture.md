# 01. Architecture — 전체 구조와 메서드 카탈로그

> [← README](README.md) | [다음: 02. comtypes 호출 패턴 →](02_comtypes_patterns.md)
>
> 코드: [mcp-server/core/etabs_api.py](../../mcp-server/core/etabs_api.py)

이 장에서는 "사용자가 자연어를 입력하면 ETABS에서 해석이 돌고 결과가 돌아온다"는 큰 그림이 **어느 계층을 거치는지** 한눈에 봅니다. 코드 한 줄씩의 설명은 04장에서 합니다.

---

## 1. 데이터 흐름도

```
[사용자 자연어]
      │
      ▼
[Claude / LLM]
      │  (도메인 의도 추출)
      ▼
[MCP Tool 호출]   ← 현재 미구현 (06장에서 추가 예정)
      │
      ▼
[ETABSClient (Python)]   ← etabs_api.py
      │  (메서드 호출)
      ▼
[comtypes]   ← Python ↔ COM 바인딩 라이브러리
      │  (TLB 시그니처 매칭)
      ▼
[ETABSv1.tlb]   ← C:\Program Files\...\NativeAPI\x64\ETABSv1.tlb
      │  (COM 인터페이스 정의)
      ▼
[ETABS 23 (ETABS.exe)]   ← 실제 FEA 엔진
      │  (해석/결과 생성)
      ▼
(역방향으로 결과가 dict / list로 반환)
```

**핵심 관찰**:
- ETABS는 GUI/엔진이 하나의 `.exe`에 있고, 외부에서 제어할 때는 `ETABSv1.tlb`(타입 라이브러리)가 정의한 COM 인터페이스를 통과합니다.
- Python은 COM을 직접 모르므로, `comtypes`가 TLB를 읽어서 Python 호출 → COM 호출 변환을 처리합니다.
- `ETABSClient`는 comtypes의 raw 인터페이스를 도메인 친화적인 메서드(`get_base_reactions()` 등)로 한 번 더 감싼 얇은 래퍼입니다.

---

## 2. ETABSClient 클래스 구조

[etabs_api.py:66-390](../../mcp-server/core/etabs_api.py#L66-L390)

```python
class ETABSClient:
    def __init__(self, etabs_object, sap_model):
        self._etabs_object = etabs_object  # cOAPI / ETABSObject
        self.model = sap_model             # cSapModel ← 모든 API의 진입점
```

ETABS COM API의 구조는 다음과 같습니다.

```
cHelper          ← TLB의 helper. 인스턴스 생성용
  └─ CreateObject("...ETABS.exe") → cOAPI
                                       └─ SapModel (= cSapModel)
                                              ├─ PointObj           (절점)
                                              ├─ FrameObj           (프레임)
                                              ├─ AreaObj            (면 요소)
                                              ├─ LoadPatterns       (하중 패턴)
                                              ├─ LoadCases          (하중 케이스)
                                              ├─ Story              (층)
                                              ├─ PropMaterial       (재료)
                                              ├─ PropFrame          (프레임 단면)
                                              ├─ Analyze            (해석 제어)
                                              ├─ Results            (결과 추출)
                                              │     ├─ Setup        (출력 케이스 선택)
                                              │     ├─ BaseReact
                                              │     ├─ ModalPeriod
                                              │     ├─ StoryDrifts
                                              │     ├─ JointDispl
                                              │     ├─ JointReact
                                              │     └─ FrameForce
                                              ├─ DatabaseTables     (범용 테이블 쿼리)
                                              └─ File               (저장/열기)
```

`ETABSClient`는 위 트리에서 `self.model` 한 점만 들고 있고, 자주 쓰는 서브객체는 `@property`로 단축키를 제공합니다.

[etabs_api.py:162-196](../../mcp-server/core/etabs_api.py#L162-L196)

```python
@property
def frame(self):         return self.model.FrameObj
@property
def point(self):         return self.model.PointObj
@property
def area(self):          return self.model.AreaObj
@property
def load_patterns(self): return self.model.LoadPatterns
@property
def load_cases(self):    return self.model.LoadCases
@property
def story(self):         return self.model.Story
@property
def analyze(self):       return self.model.Analyze
@property
def results(self):       return self.model.Results
@property
def database(self):      return self.model.DatabaseTables
```

> **왜 단축키만 두고 메서드를 안 만들었나?**  단축키는 ETABS API를 "직접" 쓸 수 있는 탈출구입니다. 04장의 6개 결과 메서드가 커버하지 못하는 호출은 `client.frame.SetSection(...)`처럼 ETABS 메서드를 그대로 부를 수 있습니다.

---

## 3. 11개 메서드 카탈로그

| # | 메서드 | 분류 | 호출하는 ETABS API | 한 줄 역할 |
|---|--------|------|---------------------|------------|
| 1 | `attach()` (classmethod) | 팩토리 | `helper.GetObject("CSI.ETABS.API.ETABSObject")` | 실행 중인 ETABS에 연결 |
| 2 | `launch(model_path, visible)` (classmethod) | 팩토리 | `helper.CreateObject(ETABS_EXE)` → `ApplicationStart` → `SapModel.InitializeNewModel` → (`File.OpenFile`) | ETABS를 새로 실행 |
| 3 | `close(save, save_path)` | 컨텍스트 | (`File.Save`) → `ApplicationExit(False)` | 종료 |
| 4 | `set_units(unit_key)` | 단위 | `SapModel.SetPresentUnits(eUnits)` | 단위 변경 |
| 5 | `run_analysis()` | 해석 | `Analyze.RunAnalysis()` | 해석 실행 |
| 6 | `get_base_reactions(load_cases)` | 결과 | `Setup.DeselectAllCasesAndCombosForOutput` → `Setup.SetCaseSelectedForOutput` → `Results.BaseReact` | 기저 반력 추출 |
| 7 | `get_modal_periods()` | 결과 | `Results.ModalPeriod` | 모달 주기 리스트 |
| 8 | `get_story_drifts()` | 결과 | `Results.StoryDrifts` | 층간변위비 |
| 9 | `get_joint_displacements(node, lc)` | 결과 | `Setup` 선택 → `Results.JointDispl` | 절점 6-DOF 변위 |
| 10 | `get_frame_forces(elem, lc)` | 결과 | `Setup` 선택 → `Results.FrameForce` | 부재력 (station별 P/V2/V3/T/M2/M3) |
| 11 | `get_model_info()` | 메타 | `PointObj.GetNameList`, `FrameObj.GetNameList`, `AreaObj.GetNameList`, `LoadPatterns.GetNameList`, `Story.GetStories` | 모델 요약 |
| 12 | `get_table(table_key)` | 범용 | `DatabaseTables.GetTableForDisplayArray` | 모든 결과 테이블을 dict 리스트로 |

(11+1 = 총 12개. 클래스 메서드 두 개를 1로 셀지에 따라 11~12개. 본 매뉴얼은 11개로 표기합니다.)

### 분류별 호출 빈도

전형적인 해석 워크플로우에서 호출 빈도는 대략:

```
attach() / launch()  ← 1회
set_units()          ← 1회
[모델 빌드: 05장 helper 호출들]  ← 다수
run_analysis()       ← 1회
get_base_reactions() / get_modal_periods() / get_story_drifts()  ← 각 1회
get_joint_displacements() / get_frame_forces()  ← 노드/요소별 다수
close()              ← 1회
```

---

## 4. 의존성

### Python 패키지

```bash
pip install comtypes
```

`comtypes`만 있으면 됩니다. `pywin32`나 `win32com.client`는 사용하지 않습니다 (둘 다 COM 바인딩이지만 `comtypes`가 in/out 매개변수를 더 깔끔하게 처리).

### ETABS 설치 경로 (etabs_api.py에서 하드코딩)

[etabs_api.py:24-26](../../mcp-server/core/etabs_api.py#L24-L26)

```python
ETABS_INSTALL_DIR = Path(r"C:\Program Files\Computers and Structures\ETABS 23")
_TLB_PATH = ETABS_INSTALL_DIR / "NativeAPI" / "x64" / "ETABSv1.tlb"
ETABS_EXE = ETABS_INSTALL_DIR / "ETABS.exe"
```

> **다른 버전이나 경로를 쓴다면?** `ETABS_INSTALL_DIR`을 수정하거나, 코드를 환경변수에서 읽도록 리팩터링하면 됩니다 (06장 확장 레시피).

### 실행 권한

- ETABS는 라이선스(USB/네트워크 동글) 필수. 라이선스 없으면 `ApplicationStart()`에서 실패.
- 첫 `comtypes.client.GetModule(tlb)` 호출 시 `comtypes\gen\` 디렉터리에 캐시 파일이 자동 생성됨 (몇 초 소요).

---

## 5. 다음 단계

이제 큰 그림이 잡혔으니, 02장에서 **comtypes가 Python ↔ COM 호출을 어떻게 매개하는지** 깊이 파봅니다. 그 패턴 하나만 이해하면 04장의 11개 메서드가 전부 같은 모양임을 알게 됩니다.

> [← README](README.md) | [다음: 02. comtypes 호출 패턴 →](02_comtypes_patterns.md)
