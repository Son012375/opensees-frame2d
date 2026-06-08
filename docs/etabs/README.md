# ETABS API MCP 학습 매뉴얼

> OpenSees-MCP 프로젝트의 ETABS 23 COM API 연동 코드를 직접 이해하고 확장하기 위한 학습 가이드.

## 학습 목표

이 매뉴얼을 끝까지 학습하면 다음을 할 수 있습니다.

1. `mcp-server/core/etabs_api.py`의 11개 메서드가 **어떤 ETABS COM 호출을 어떻게 래핑하는지** 설명할 수 있다.
2. 새로운 결과 추출 메서드(예: `ModalParticipatingMassRatios`)를 직접 추가할 수 있다.
3. Python 코드만으로 ETABS 모델(노드/부재/하중/지점)을 빌드하고 해석을 실행할 수 있다.
4. `mcp-server/server.py`에 ETABS 기능을 새 MCP tool로 등록할 수 있다.
5. `ret ≠ 0`이나 빈 결과 배열 같은 오류를 진단하고 해결할 수 있다.

## 전제 조건

| 항목 | 필요 수준 |
|------|-----------|
| Python | 3.10+ (typing, dict literal 사용) |
| ETABS | v23 설치 (`C:\Program Files\Computers and Structures\ETABS 23`) |
| OpenSees-MCP 프로젝트 구조 | 기본 이해 (`mcp-server/`, `tests/benchmark/`, `scripts/`) |
| 구조해석 도메인 | 보·기둥·골조 모델링, 변위/모멘트/반력 의미 |
| COM API | 사전 지식 없어도 됨 — 02장에서 다룸 |

## 권장 학습 순서

```
README → 01 → 02 → 03 → 04 → 05 → 06 / 07 (필요할 때 참조)
```

| 순서 | 문서 | 학습 시간 (예상) | 비고 |
|------|------|------------------|------|
| 1 | [01. Architecture](01_architecture.md) | 15분 | 전체 그림 + 11개 메서드 카탈로그 |
| 2 | [02. comtypes 호출 패턴](02_comtypes_patterns.md) | 30분 | **핵심** — 모든 메서드의 공통 패턴 |
| 3 | [03. 단위와 부호규약](03_units_and_conventions.md) | 30분 | **핵심** — 결과 해석에 필수 |
| 4 | [04. ETABSClient API 워크스루](04_etabs_api_walkthrough.md) | 40분 | 02·03의 적용 예 11개 |
| 5 | [05. 모델 빌드 헬퍼](05_model_building.md) | 30분 | 벤치마크 케이스 분해 |
| 6 | [06. 확장 레시피](06_extending_recipes.md) | 60분 | 직접 새 기능 추가하기 |
| 7 | [07. 트러블슈팅](07_troubleshooting.md) | 참조용 | FAQ |
| 8 | [08. 벤치마크 결과](08_benchmark_results.md) | 25분 | OpenSees/ETABS/Midas 3-way 비교 (72개 메트릭 FAIL 0건, OS↔ET 6자리 일치) |

## 빠른 참조 표

### 핵심 코드 파일

| 파일 | 줄 수 | 용도 |
|------|------|------|
| [mcp-server/core/etabs_api.py](../../mcp-server/core/etabs_api.py) | 390 | ETABSClient 클래스 본체 |
| [scripts/test_etabs_connection.py](../../scripts/test_etabs_connection.py) | 75 | 첫 연결 테스트 |
| [tests/benchmark/etabs_benchmark_case1_2.py](../../tests/benchmark/etabs_benchmark_case1_2.py) | 529 | Case 1·2 벤치마크 + 모델 빌드 헬퍼 |
| [data/etabs_summary/SUMMARY.md](../../data/etabs_summary/SUMMARY.md) | 194 | ETABS v23 기능 배경 |

### ETABS 설치 경로 (Windows 기본값)

```
C:\Program Files\Computers and Structures\ETABS 23\
├── ETABS.exe
└── NativeAPI\x64\
    ├── ETABSv1.tlb            ← 핵심 TLB
    └── Table and Field Keys.xml  ← get_table()용 키 목록
```

### 단위 키 (자주 쓰는 것만)

| 키 | enum 값 | 의미 |
|----|---------|------|
| `kN_m_C` | 6 | **권장 (SI)** |
| `N_mm_C` | 10 | 벤치마크 N·mm |

전체 목록은 [03 단위와 부호규약](03_units_and_conventions.md) 참조.

### ETABSClient 메서드 한눈에 보기

| 분류 | 메서드 | 역할 |
|------|--------|------|
| 팩토리 | `attach()`, `launch(model_path, visible)` | ETABS 인스턴스 확보 |
| 컨텍스트 | `__enter__`, `__exit__`, `close()` | `with` 구문 지원 |
| 단위 | `set_units(unit_key)` | eUnits 설정 |
| 해석 | `run_analysis()` | `Analyze.RunAnalysis()` 호출 |
| 결과 | `get_base_reactions()` | 기저 반력 (Fx/Fy/Fz/Mx/My/Mz) |
| 결과 | `get_modal_periods()` | 모달 주기 리스트 |
| 결과 | `get_story_drifts()` | 층간변위비 |
| 결과 | `get_joint_displacements(node, lc)` | 절점 6-DOF 변위 |
| 결과 | `get_frame_forces(elem, lc)` | 부재력 (P, V2, V3, T, M2, M3) |
| 결과 | `get_model_info()` | 모델 요약 (노드/프레임/층 수) |
| 범용 | `get_table(table_key)` | 모든 데이터베이스 테이블 |

## 학습 핵심 요약 (한 페이지 치트시트)

세 가지만 기억하세요.

1. **comtypes 메서드는 in/out 매개변수를 placeholder로 받고 튜플로 돌려준다.**
   ```python
   (n, lc, ..., ret) = model.Results.BaseReact(0, [], [], [], [], [], [], [], [], [], 0.0, 0.0, 0.0)
   ```
   리스트 개수와 순서가 TLB 시그니처와 정확히 일치해야 함.

2. **모든 ETABS API는 정수 `ret`을 마지막에 반환한다. `ret != 0` 이면 실패.**

3. **결과를 뽑기 전에 `Setup.SetCaseSelectedForOutput()`을 호출해야 한다.** 누락하면 빈 배열만 돌아온다.

## 매뉴얼 외 보조 자료

- ETABS API 공식 도움말: `C:\Program Files\Computers and Structures\ETABS 23\Help\` (CSI Reference Guide)
- TLB 시그니처 직접 보기: PowerShell에서 `oleview /TLB:"C:\...\ETABSv1.tlb"` 또는 Python `comtypes.client.GetModule()` 후 `dir(lib.cSapModel)` 등
- `get_table()` 키 전체: `C:\Program Files\Computers and Structures\ETABS 23\NativeAPI\Table and Field Keys.xml`
