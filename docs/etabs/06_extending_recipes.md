# 06. 확장 레시피 — 새 기능 직접 추가하기

> [← 05. 모델 빌드 헬퍼](05_model_building.md) | [다음: 07. 트러블슈팅 →](07_troubleshooting.md)
>
> 코드: [mcp-server/core/etabs_api.py](../../mcp-server/core/etabs_api.py), [mcp-server/server.py](../../mcp-server/server.py)

지금까지의 학습이 진짜 학습이 되려면 직접 손으로 새 기능을 만들어봐야 합니다. 이 장에는 자주 발생할 확장 시나리오 세 가지를 레시피로 정리했습니다.

- **레시피 A**: 새 결과 추출 메서드 추가 (`get_modal_mass_ratios()`)
- **레시피 B**: `get_table()` 활용 — 메서드 추가 없이 즉석 결과
- **레시피 C**: MCP tool로 등록 (`server.py`)

---

## 레시피 A: 새 결과 추출 메서드 추가

목표: ETABS의 **모달 질량 참여율** (`ModalParticipatingMassRatios`)을 `get_modal_mass_ratios()` 메서드로 추가.

### A-1. TLB에서 시그니처 확인

ETABS API의 메서드 시그니처는 두 가지 방법으로 확인할 수 있습니다.

**방법 1**: 공식 문서  
`C:\Program Files\Computers and Structures\ETABS 23\Help\CSi_OAPI_Documentation.chm`을 열고 `ModalParticipatingMassRatios`를 검색.

**방법 2**: Python REPL에서 직접 검사

```python
from mcp_server.core.etabs_api import _get_etabs_lib
lib = _get_etabs_lib()

# Results 인터페이스의 메서드 목록
import inspect
print([m for m in dir(lib.cResults) if "Modal" in m])
# ['ModalLoadParticipationRatios', 'ModalParticipatingMassRatios', 'ModalParticipationFactors', 'ModalPeriod']

# 시그니처 (comtypes는 _methods_ 속성에 IDL 메타데이터 보관)
import comtypes
print(lib.cResults._methods_)
```

ETABS 23의 `ModalParticipatingMassRatios`는 13개 in/out 매개변수입니다.

| 자리 | 매개변수 | 타입 | 의미 |
|------|----------|------|------|
| 1 | NumberResults | int | 결과 행 수 |
| 2 | LoadCase | string[] | 케이스 이름 |
| 3 | StepType | string[] | 모드 타입 |
| 4 | StepNum | double[] | 모드 번호 |
| 5 | Period | double[] | 주기 (초) |
| 6 | Ux | double[] | X 질량 참여율 |
| 7 | Uy | double[] | Y |
| 8 | Uz | double[] | Z |
| 9 | SumUx | double[] | X 누적 |
| 10 | SumUy | double[] | Y 누적 |
| 11 | SumUz | double[] | Z 누적 |
| 12 | Rx | double[] | X 회전 |
| 13 | Ry | double[] | Y 회전 |
| 14 | Rz | double[] | Z 회전 |
| 15 | SumRx | double[] | X 회전 누적 |
| 16 | SumRy | double[] | Y 회전 누적 |
| 17 | SumRz | double[] | Z 회전 누적 |

(버전마다 약간 다를 수 있음. TLB로 직접 확인 권장.)

### A-2. 메서드 작성

[etabs_api.py:257-266](../../mcp-server/core/etabs_api.py#L257-L266)의 `get_modal_periods()` 패턴을 그대로 본떠 새 메서드를 만듭니다.

`mcp-server/core/etabs_api.py`의 `# Result extraction helpers` 섹션에 추가:

```python
def get_modal_mass_ratios(self) -> list:
    """모달 질량 참여율 리스트.

    Returns:
        [{"mode": int, "period": float,
          "Ux": float, "Uy": float, "Uz": float,
          "SumUx": float, "SumUy": float, "SumUz": float,
          "Rx": float, "Ry": float, "Rz": float,
          "SumRx": float, "SumRy": float, "SumRz": float}, ...]
    """
    (n, lc, step_type, step_num, period,
     ux, uy, uz, sum_ux, sum_uy, sum_uz,
     rx, ry, rz, sum_rx, sum_ry, sum_rz, ret) = self.model.Results.ModalParticipatingMassRatios(
        0, [], [], [], [],
        [], [], [], [], [], [],
        [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"모달 질량 참여율 추출 실패 (ret={ret})")
    if not period:
        return []
    return [
        {
            "mode":   int(step_num[i]),
            "period": float(period[i]),
            "Ux":     float(ux[i]),     "Uy":     float(uy[i]),     "Uz":     float(uz[i]),
            "SumUx":  float(sum_ux[i]), "SumUy":  float(sum_uy[i]), "SumUz":  float(sum_uz[i]),
            "Rx":     float(rx[i]),     "Ry":     float(ry[i]),     "Rz":     float(rz[i]),
            "SumRx":  float(sum_rx[i]), "SumRy":  float(sum_ry[i]), "SumRz":  float(sum_rz[i]),
        }
        for i in range(n)
    ]
```

### A-3. 동작 확인

```python
from mcp_server.core.etabs_api import ETABSClient

client = ETABSClient.attach()   # ETABS에 모달 해석 완료 모델 열려있어야 함
ratios = client.get_modal_mass_ratios()

print(f"전체 모드: {len(ratios)}")
print(f"1차 모드: 주기 {ratios[0]['period']:.3f}초, "
      f"Ux 참여율 {ratios[0]['Ux']*100:.1f}%")
print(f"90% 도달 모드: "
      f"X = {next(r['mode'] for r in ratios if r['SumUx'] >= 0.9)}")
```

### A-4. 체크리스트

- [ ] TLB 시그니처 확인 (매개변수 개수와 순서)
- [ ] placeholder 타입이 맞는지 (int = 0, double = 0.0, array = [])
- [ ] 좌변 변수 개수 = placeholder 개수 + 1 (ret)
- [ ] ret != 0 체크
- [ ] None 방어 (`if x else []`)
- [ ] 결과 dict의 키 이름이 일관적인지 (다른 메서드와 통일)

---

## 레시피 B: `get_table()` 활용 — 즉석 결과

매번 메서드를 만들 필요 없이 `get_table()`로 ETABS의 모든 데이터베이스 테이블에 접근할 수 있습니다. **프로토타이핑**이나 **자주 안 쓰는 결과**에 적합.

### B-1. 사용 가능한 table_key 찾기

세 가지 방법:

**방법 1**: ETABS GUI  
`Display → Show Tables → ...`에서 보이는 카테고리/테이블 이름. 대부분 그 이름 그대로 키로 동작.

**방법 2**: 키 목록 XML  
`C:\Program Files\Computers and Structures\ETABS 23\NativeAPI\Table and Field Keys.xml` 파일 열어서 `<TableKey>` 태그 찾기.

**방법 3**: API로 동적 조회

```python
from mcp_server.core.etabs_api import ETABSClient

client = ETABSClient.attach()
db = client.database   # 단축키

# 사용 가능한 모든 테이블 키 (n_tables 개)
(n_tables, table_keys, ret) = db.GetAvailableTables(0, [])
for key in list(table_keys)[:20]:
    print(key)
```

자주 쓰는 키 예시:
- `"Modal Participating Mass Ratios"`
- `"Story Stiffness"`
- `"Story Drifts"`
- `"Joint Reactions"`
- `"Element Forces - Beams"`
- `"Material Properties - Summary"`

### B-2. 호출

```python
client.set_units("kN_m_C")
client.run_analysis()  # 해석 안 했으면 일부 테이블 비어있음

rows = client.get_table("Story Stiffness")
# [{"Story": "Story1", "OutputCase": "EQX", "StiffX": 1234.5, "StiffY": 1100.2, ...},
#  {"Story": "Story1", "OutputCase": "EQY", "StiffX": 1230.0, ...}, ...]
```

### B-3. 필터링과 가공

```python
# X방향 EQX 케이스만
eqx_x = [r for r in rows if r["OutputCase"] == "EQX"]

# Pandas로 더 강력하게
import pandas as pd
df = pd.DataFrame(rows)
df = df[df["OutputCase"] == "EQX"]
df["StiffRatio"] = df["StiffX"] / df["StiffY"]
print(df)
```

### B-4. 언제 메서드로 승격하나

같은 `table_key`를 3번 이상 호출하게 되면 → 전용 메서드 작성 권장 (레시피 A 적용).

장점:
- 결과 타입 힌트
- 단위 변환 코드를 메서드 내부로
- 에러 메시지가 도메인 친화적

단점:
- 코드 길이 증가, 유지보수 부담

---

## 레시피 C: MCP tool로 등록

지금까지 `ETABSClient`는 Python 스크립트로만 호출했습니다. **Claude에서 자연어로 호출**하려면 MCP tool로 등록해야 합니다. 이 레시피는 `etabs_run_analysis` tool 하나를 추가하는 예제.

### C-1. 기존 패턴 파악

`mcp-server/server.py`는 다음 패턴을 따릅니다.

[server.py:14-15](../../mcp-server/server.py#L14-L15)
```python
from mcp.types import Tool, TextContent, ImageContent
from pydantic import BaseModel, Field
```

각 tool은 세 부분:
1. **Pydantic 입력 모델** (파일 상단에 클래스 정의)
2. **Tool 등록** (`@server.list_tools()` 데코레이터 함수 안에 `Tool(...)` 추가)
3. **Handler** (`@server.call_tool()` 데코레이터 함수 안에 `if name == "..."` 분기 추가)

### C-2. Pydantic 입력 모델 작성

`server.py`의 입력 모델 정의 영역에 추가:

```python
class ETABSRunAnalysisInput(BaseModel):
    """ETABS 23 해석 실행 입력"""
    model_path: str | None = Field(
        default=None,
        description=".edb 파일 경로. None이면 실행 중인 ETABS에 attach"
    )
    extract_targets: list[Literal[
        "base_reactions", "modal_periods", "story_drifts", "model_info"
    ]] = Field(
        default=["base_reactions", "model_info"],
        description="추출할 결과 목록"
    )
    load_cases: list[str] | None = Field(
        default=None,
        description="base_reactions에서 출력할 케이스. None이면 전체"
    )
```

### C-3. Tool 등록 (list_tools 안)

[server.py:429](../../mcp-server/server.py#L429)의 `return [` 리스트에 추가:

```python
Tool(
    name="etabs_run_analysis",
    description="""ETABS 23 모델을 해석하고 결과를 추출합니다.

전제: ETABS 23 설치 + 라이선스 활성 + 모델 경로 또는 실행 중인 ETABS.

입력:
- model_path: .edb 파일 경로 (없으면 실행 중인 ETABS에 attach)
- extract_targets: 추출할 결과 (base_reactions, modal_periods, story_drifts, model_info)
- load_cases: base_reactions의 케이스 (전체면 생략)

출력:
- 단위 kN, m, °C로 통일
- 각 target에 해당하는 dict (예: base_reactions = {Fx, Fy, Fz, Mx, My, Mz})""",
    inputSchema=ETABSRunAnalysisInput.model_json_schema(),
),
```

### C-4. Handler 작성 (call_tool 안)

[server.py:762](../../mcp-server/server.py#L762)의 `call_tool` 안에 `elif` 추가:

```python
elif name == "etabs_run_analysis":
    from core.etabs_api import ETABSClient   # 지연 import (comtypes 없는 환경 보호)

    input_data = ETABSRunAnalysisInput(**arguments)

    if input_data.model_path:
        client = ETABSClient.launch(model_path=input_data.model_path)
    else:
        client = ETABSClient.attach()

    try:
        client.set_units("kN_m_C")
        client.run_analysis()

        response = {"status": "success", "extracted": {}}
        for target in input_data.extract_targets:
            if target == "base_reactions":
                response["extracted"]["base_reactions"] = client.get_base_reactions(
                    load_cases=input_data.load_cases
                )
            elif target == "modal_periods":
                response["extracted"]["modal_periods"] = client.get_modal_periods()
            elif target == "story_drifts":
                response["extracted"]["story_drifts"] = client.get_story_drifts()
            elif target == "model_info":
                response["extracted"]["model_info"] = client.get_model_info()
    finally:
        if input_data.model_path:   # launch한 경우만 닫음
            client.close()

    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2, ensure_ascii=False)
    )]
```

### C-5. 동작 확인

```bash
# MCP 서버 재시작 후 Claude에서:
"ETABS에 열려있는 모델로 해석 실행하고 base reactions 가져와줘"
```

Claude가 `etabs_run_analysis` tool을 자동 호출.

### C-6. 추가 고려 사항

- **라이선스**: ETABS는 USB/네트워크 동글 필수. MCP 서버를 다른 머신에서 띄우면 라이선스가 없어서 실패.
- **에러 메시지**: ETABS API의 ret 코드를 사람 친화적 메시지로 변환 (07장 트러블슈팅 참조).
- **시각화**: ETABS 결과를 OpenSees처럼 HTML 리포트로 만들 수도 있음 (`visualization_3d.py` 패턴 모방).
- **빌드 tool**: 해석 전 모델 빌드도 별도 tool로 (`etabs_build_simple_beam` 등). 05장 helper를 재사용.

---

## 종합 — 어떤 레시피를 언제 쓰나

| 시나리오 | 추천 레시피 |
|----------|-------------|
| 한 번만 결과 확인 | B (get_table) |
| 정기적으로 같은 결과 추출 | A (전용 메서드) |
| Claude가 자연어로 트리거 | C (MCP tool) |
| 자동화 배치 스크립트 | A + B 조합 (메서드 + table fallback) |

---

## 더 나아가기

확장 아이디어:

1. **자동 단면 설계**: `get_frame_forces()`로 부재력 추출 → KDS 14 31 강도 검토 → 결과 dict에 OK/NG 추가
2. **자동 비교 리포트**: OpenSees + ETABS + Midas를 한 번에 돌리고 [etabs_benchmark_case1_2.py:397-441](../../tests/benchmark/etabs_benchmark_case1_2.py#L397-L441)의 `format_3way`를 재사용
3. **모달 비교**: `get_modal_mass_ratios()` (레시피 A) + OpenSees 고유치 → 주기/방향 자동 매칭
4. **ETABS 파일 일괄 처리**: 디렉터리 내 모든 .edb 순회, 각각 `launch(visible=False)`로 백그라운드 해석 → 결과 CSV 집계
5. **ETABS → IFC 역변환**: `get_model_info()` + `PointObj.GetCoordCartesian()` + `FrameObj.GetPoints()` 등으로 모델을 재구성 → ifcopenshell로 IFC 생성

---

> [← 05. 모델 빌드 헬퍼](05_model_building.md) | [다음: 07. 트러블슈팅 →](07_troubleshooting.md)
