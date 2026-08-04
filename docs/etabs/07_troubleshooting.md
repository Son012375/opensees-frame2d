# 07. 트러블슈팅 — 자주 만나는 오류와 해결법

> [← 06. 확장 레시피](06_extending_recipes.md) | [README로 →](README.md)

ETABS API를 다루다 보면 만나는 전형적인 오류들과 해결책을 모아둡니다. **증상 → 원인 → 해결** 순서로 정리했습니다.

---

## 1. 연결/실행 단계

### 1.1. `FileNotFoundError: ETABS TLB를 찾을 수 없습니다`

**증상**
```
FileNotFoundError: ETABS TLB를 찾을 수 없습니다:
C:\Program Files\Computers and Structures\ETABS 23\NativeAPI\x64\ETABSv1.tlb
```

**원인**
- ETABS 23이 설치 안 됨
- 다른 버전(예: ETABS 22, 21)이 설치됨
- 사용자 지정 경로에 설치됨

**해결**
1. ETABS 23 설치 확인 (`Programs and Features`)
2. 경로 확인: `Get-ChildItem "C:\Program Files\Computers and Structures"`
3. 다른 경로/버전이면 `etabs_api.py` 수정:
   ```python
   ETABS_INSTALL_DIR = Path(r"C:\Program Files\Computers and Structures\ETABS 21")
   ```
4. 또는 환경변수에서 읽도록 리팩터링 (06장 레시피).

### 1.2. `ImportError: comtypes가 설치되지 않았습니다`

**증상**
```
ImportError: comtypes가 설치되지 않았습니다. pip install comtypes 를 실행하세요.
```

**해결**
```powershell
.\opensees-mcp\Scripts\python.exe -m pip install comtypes
```

또는 프로젝트 venv 활성화 후:
```powershell
pip install comtypes
```

### 1.3. `RuntimeError: 실행 중인 ETABS를 찾을 수 없습니다`

**증상**
```
RuntimeError: 실행 중인 ETABS를 찾을 수 없습니다: ...
ETABS를 먼저 열거나 ETABSClient.launch()를 사용하세요.
```

**원인**
- `attach()` 호출 시 ETABS GUI가 실행되어 있지 않음
- ETABS는 떠 있지만 COM 등록이 안 됨 (드물게 ETABS 첫 실행 시)

**해결**
1. ETABS 23을 GUI로 한 번 실행해서 모델 띄움
2. `attach()` 대신 `launch()` 사용 (자동으로 ETABS 띄움)
3. ETABS를 관리자 권한으로 실행 (COM 등록 문제)

### 1.4. `RuntimeError: ETABS 실행 실패` (launch)

**증상**
```
RuntimeError: ETABS 실행 실패: <COMError ...>
```

**원인**
- 라이선스 동글 (USB/네트워크) 미인식
- ETABS 라이선스 만료
- 이전 ETABS 프로세스가 메모리에 남아있음 (충돌)

**해결**
1. ETABS GUI로 한 번 띄워서 라이선스 확인 (라이선스 매니저)
2. `taskkill /F /IM ETABS.exe` (남은 프로세스 종료)
3. Windows 작업 관리자에서 `ETABS.exe`와 `Csi.SapModel.dll` 관련 프로세스 모두 종료
4. 컴퓨터 재부팅 (COM 가비지 정리)

### 1.5. `RuntimeError: 모델 파일 열기 실패 (ret=N)`

**원인 (ret 값별)**

| ret | 의미 |
|-----|------|
| 1 | 파일 경로 오류 또는 권한 없음 |
| 2 | 파일 포맷 오류 (다른 버전 ETABS로 저장됨) |
| 3+ | 파일 손상 또는 라이선스 문제 |

**해결**
- 절대 경로 사용 (`Path(p).resolve()`)
- 파일 확장자 확인 (`.edb`)
- 다른 버전이면 GUI에서 한 번 열어서 새 버전으로 저장 후 재시도

---

## 2. 모델 빌드 단계

### 2.1. `SetMaterial 'XXX' failed`

**원인**
- 재료 타입 enum 값이 잘못됨
- 같은 이름이 이미 존재

**해결**
- 재료 타입 확인 (Steel=1, Concrete=2, NoDesign=3, ...)
- 다른 이름 사용 또는 `PropMaterial.Delete(name)` 후 재생성

### 2.2. `SetISection 'XXX' failed`

**원인**
- 단면명에 사용된 재료가 정의되지 않음 (`_material` 호출 전 `_isection` 호출)
- 치수가 0 또는 음수
- 단위 불일치 (예: kN_m 단위인데 mm 값 입력)

**해결**
1. 빌드 순서 확인: `_init` → `_material` → `_isection` → ...
2. 치수가 모두 양수인지
3. `set_units()` 호출 후 단위 일관성 유지

### 2.3. `AddCartesian 'N1' failed`

**원인**
- 같은 이름의 노드가 이미 존재 (단, 일반적으로 ETABS는 자동으로 이름 충돌을 처리)
- 좌표가 비현실적 (극단적 값)

**해결**
- 노드 이름을 명시적으로 다르게
- 또는 빈 문자열 `""`로 호출해서 ETABS가 자동 부여하도록

### 2.4. `AddByPoint 'E1' failed (ret=N)`

**원인 (자주)**
- i_node 또는 j_node 이름이 존재하지 않음
- i_node == j_node (zero-length element)
- 단면 이름이 존재하지 않음

**해결**
1. `_pt()` 호출 결과의 반환 이름을 다시 사용 (입력 이름과 다를 수 있음)
2. 두 노드가 다른 위치인지
3. 단면을 먼저 `_isection()`으로 정의했는지

### 2.5. 해석은 돌지만 결과가 이상함

**의심 패턴**
- 모든 변위가 0 → 지점이 over-constrained
- 변위가 비정상적으로 큼 → 지점 부족 또는 기구 (mechanism)
- 모멘트가 0 → 핀 연결이 모든 곳에 (회전 해제)

**해결**
- `get_model_info()`로 노드/프레임 수 확인
- ETABS GUI에서 시각적으로 모델 점검 (`Display → Show Restraints`, `Show Frame Releases`)
- 지점 dofs를 [03장 부호규약](03_units_and_conventions.md) 참조

---

## 3. 해석 단계

### 3.1. `RunAnalysis failed` (ret≠0)

**원인 (자주)**
- 모델 미완성 (지점 0, 하중 0, 또는 단면 미정의 요소)
- 자유도 부족 (기구)
- 디스크 공간 부족 (큰 모델의 경우)
- 라이선스 만료 (해석 도중)

**해결**
1. ETABS GUI를 열고 `Analyze → Check Model` 실행 → 오류 메시지 확인
2. 모델을 일단 저장하고 GUI에서 수동 해석 시도
3. 디스크 공간 확인 (`Get-PSDrive`)

### 3.2. 해석은 끝났지만 너무 오래 걸림

**원인**
- 큰 모델 (수만 노드)
- 복잡한 비선형 해석
- 모달 모드 수 과다

**해결**
- ETABS GUI에서 `Analysis Options` 조정 (모드 수, 솔버 옵션)
- 작은 서브모델로 검증 후 확장

---

## 4. 결과 추출 단계

### 4.1. ★ 결과 배열이 모두 비어있음 ★

**증상**
```python
reactions = client.get_base_reactions()
print(reactions["Fx"])   # [] (빈 배열, 예외도 안 남)
```

**원인 (1순위)**
- `Setup.DeselectAllCasesAndCombosForOutput` → `SetCaseSelectedForOutput` 누락
- 이 패턴이 빠지면 ret=0이지만 모든 배열이 비어서 옴

**원인 (그 외)**
- 해석을 안 돌림
- 케이스 이름이 잘못됨 (대소문자 등)

**해결**
1. `get_base_reactions()` 코드를 읽고 Setup 호출 확인
2. 직접 호출하는 경우 반드시 두 줄 추가:
   ```python
   setup = client.model.Results.Setup
   setup.DeselectAllCasesAndCombosForOutput()
   setup.SetCaseSelectedForOutput("CASE1", True)
   ```
3. 케이스 이름 확인: `client.model.LoadCases.GetNameList(0, [])`

### 4.2. `JointDispl '...' failed (ret=N)`

**원인**
- 노드 이름이 존재하지 않음
- 해석 결과가 없음 (모달만 돌리고 정적 안 돌림)
- Setup에서 해당 케이스가 선택 안 됨

**해결**
1. 노드 이름 확인 (`PointObj.GetNameList(0, [])`)
2. `get_model_info()`로 노드 수 확인
3. Setup 호출 확인

### 4.3. `FrameForce '...' failed (ret=N)`

**원인**
- 요소 이름이 존재하지 않음
- 해당 케이스가 정적 해석을 안 돌림
- 보 단부(i/j-end)가 ETABS의 actual node와 다름 (변환 후)

**해결**
- 요소 이름 확인: `FrameObj.GetNameList(0, [])`
- 케이스 종류 확인 (`LoadCases.GetTypeOAPI(name)`)

### 4.4. 결과는 나오지만 부호가 반대

**증상**
- M3가 양수일 줄 알았는데 음수, 또는 그 반대
- Midas/OpenSees와 부호 다름

**원인**
- ETABS local2/local3 정의와 다른 가정
- 좌표축 매핑 오류 (X-Z 평면 모델에서 vertical을 Y로 가정)

**해결**
[03장 부호규약](03_units_and_conventions.md) 참조. 핵심:
- 보 (+X): local2=+Z, local3=−Y → M3<0 = sagging
- 기둥 (+Z): local2=+X, local3=+Y → M3<0 = base에 +X 외력

벤치마크 코드의 [`Sign convention` 주석](../../tests/benchmark/etabs_benchmark_case1_2.py#L16-L21)을 참조.

### 4.5. 단위가 예상과 다름

**증상**
- 변위가 5 mm일 줄 알았는데 0.005 (단위가 m로 나옴)
- 모멘트가 90 kN·m 일 줄 알았는데 90000000 (N·mm)

**원인**
- `set_units()` 누락 또는 다른 단위 사용 중

**해결**
1. 결과 추출 직전 `set_units("kN_m_C")` 호출
2. 코드에서 단위 변환 직접 처리:
   ```python
   m3_kNm = m3_Nmm / 1e6
   f_kN = f_N / 1000.0
   ```

### 4.6. `get_table()` 결과가 빈 리스트

**원인 (자주)**
- 해석을 안 돌림 (결과 테이블)
- table_key 이름이 잘못됨 (오타, 대소문자)
- 모델 정의가 부족해서 ETABS가 테이블을 안 만듦 (예: 모달 해석이 없는데 Modal Mass Ratios 요청)

**해결**
1. `db.GetAvailableTables(0, [])`로 실제 사용 가능한 키 목록 확인
2. 정확한 이름 사용 (`Table and Field Keys.xml` 참고)
3. 필요한 해석이 모델에 정의돼 있는지

---

## 5. comtypes 단계 (드물지만 헷갈리는)

### 5.1. `TypeError: BaseReact() takes exactly 13 arguments`

**원인**
- placeholder 개수가 TLB 시그니처와 안 맞음

**해결**
- TLB에서 정확한 매개변수 개수 확인
- 메서드 시그니처가 ETABS 버전마다 다를 수 있으니, 본인 ETABS 버전의 TLB 기준

### 5.2. `comtypes.COMError: Object reference not set`

**원인**
- ETABS 인스턴스가 이미 닫혔는데 메서드 호출
- `attach()` 후 ETABS GUI를 사용자가 직접 닫음

**해결**
- ETABSClient를 컨텍스트 매니저로 사용 (`with ETABSClient.launch() as client:`)
- 사용 후 즉시 `client.close()`

### 5.3. comtypes gen 캐시가 stale

**증상**
- TLB 변경(ETABS 업데이트) 후 새 메서드를 인식 못 함
- 또는 매개변수 시그니처가 옛 버전대로

**해결**
1. `comtypes/gen/` 디렉터리 위치 확인:
   ```python
   import comtypes.client
   print(comtypes.client.gen_dir)
   ```
2. 해당 디렉터리에서 `ETABSv1_*` 파일 모두 삭제
3. 코드 재실행 시 자동 재생성

---

## 6. 기타

### 6.1. ETABS 한국어 GUI에서 단위 표시가 다름

ETABS는 GUI 표시 단위와 API의 `SetPresentUnits` 단위가 **별도**. API 단위만 신경 쓰면 됨.

### 6.2. ETABS가 백그라운드에서 안 닫힘

```python
client = ETABSClient.launch()
# ... 코드 ...
# client.close() 안 부르고 스크립트 종료
```

ETABS 프로세스가 메모리에 남음. **해결**:
- 항상 `with ETABSClient.launch() as client:` 또는 try/finally
- 강제 종료: `taskkill /F /IM ETABS.exe`

### 6.3. WSL/Mac/Linux에서 실행

ETABS COM API는 **Windows 전용**. WSL이나 macOS에서는 동작 안 함.

**대안**:
- Windows 머신에서 MCP 서버 실행
- 결과 JSON만 다른 OS로 복사
- 또는 OpenSees로 동등 해석 후 비교

### 6.4. 디버깅 팁

문제 원인을 좁히는 순서:
1. `get_model_info()` — 모델이 빌드됐는지
2. `client.model.LoadCases.GetNameList(0, [])` — 케이스가 있는지
3. ETABS GUI를 열어서 시각적 점검 (`launch(visible=True)`)
4. `Analyze → Check Model` GUI 메뉴로 오류 메시지 확인

### 6.5. 도움말 자료

- ETABS API 공식: `C:\Program Files\Computers and Structures\ETABS 23\Help\CSi_OAPI_Documentation.chm`
- CSI 공식 포럼: https://wiki.csiamerica.com/
- comtypes 문서: https://pythonhosted.org/comtypes/

---

## 자주 보는 ret 코드 (요약 표)

| 메서드 | ret | 가장 가능성 높은 원인 |
|--------|-----|-----------------------|
| `File.OpenFile` | 1 | 파일 경로 오류 |
| `File.OpenFile` | 2 | 파일 포맷 오류 |
| `SetMaterial` | 1+ | 동명 재료 존재 |
| `SetISection` | 1+ | 재료 미정의 |
| `AddCartesian` | 1+ | 동명 노드 존재 |
| `AddByPoint` | 1+ | 노드/단면 미존재 |
| `SetRestraint` | 1+ | 노드 미존재 |
| `RunAnalysis` | 1+ | 모델 불완전 |
| `BaseReact` | 1+ | 케이스 미선택, 해석 안 됨 |
| `JointDispl` | 1+ | 노드 미존재, 케이스 미선택 |
| `FrameForce` | 1+ | 요소 미존재, 케이스 미선택 |
| `GetTableForDisplayArray` | 1+ | 잘못된 table_key |

> ret 값의 구체적 의미는 ETABS API 매뉴얼에 항상 명시돼 있는 것은 아닙니다. ret≠0이면 위 원인을 순차 점검.

---

> [← 06. 확장 레시피](06_extending_recipes.md) | [README로 →](README.md)
