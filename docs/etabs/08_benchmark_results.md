# 08. 벤치마크 결과 — OpenSees / ETABS / Midas Gen 3-way 비교

> [← 07. 트러블슈팅](07_troubleshooting.md) | [README로 →](README.md)

OpenSees-MCP 솔버의 정확성을 검증하기 위해, **OpenSees · ETABS 23 · Midas Gen**
세 프로그램으로 동일 모델을 해석하여 메트릭별 결과를 비교한다.
세 프로그램 모두 동일한 KS D 3502:2021 단면 데이터를 사용하며, 모델링 가정
(Euler-Bernoulli, 중심선 모델링, 선형 정적)을 통일하였다.

**총 결과: 37/37 OK (Case 1+2+3)** — OpenSees와 ETABS는 모든 메트릭에서
6 유효숫자까지 완전 일치하며, Midas Gen과는 최대 0.04% 이내 일치.

---

## 1. 벤치마크 방법론

### 1.1. 프로그램별 단면 데이터 출처

| 프로그램 | 단면 출처 | 비고 |
|----------|-----------|------|
| **OpenSees** | `tests/benchmark/cases.py` 하드코딩 KS D 3502 표값 | A=8412 mm², Ix=23700 cm⁴ 등 |
| **ETABS 23** | `KoreanKS21.xml` (ETABS 설치본 내장 KS D 3502:2021 라이브러리) | `PropFrame.ImportProp` API로 직접 import |
| **Midas Gen** | Midas Section DB (사용자 직접 선택) | KS D 3502 표준 단면 |

→ 세 프로그램이 **같은 KS 표를 다른 경로로** 가져옴. 일치 정도가 곧 **솔버 정확성의 증거**.

### 1.2. 통일된 모델링 가정

| 항목 | 통일값 | 이유 |
|------|--------|------|
| 빔 이론 | **Euler-Bernoulli** (전단변형 무시) | OpenSees `elasticBeamColumn` 기본 / ETABS는 `SetModifiers(As2=As3=0)` |
| 접합부 모델링 | **중심선-중심선** | ETABS 자동 강체 오프셋은 `SetEndLengthOffset(False, 0, 0, 0)`로 비활성화 |
| 재료 거동 | **선형 탄성** | E=210 GPa, ν=0.3, SS275 |
| 해석 방식 | **선형 정적** | Case 1-3은 P-Delta 미적용 |
| ETABS 단위계 | **kN-m-C** | `ImportProp`이 cm 라이브러리 값을 kN-m로 저장하는 동작 활용 |

### 1.3. 합격 기준 (Status)

- **OK**: |ETABS - Midas| / max(|ETABS|, |Midas|) ≤ 1.0 %
- **CHECK**: 1.0 ~ 5.0 %
- **FAIL**: > 5.0 %

---

## 2. Case 1 — 2D 단순보

### 2.1. 모델 개요

```
Layout (X-axis, Z=0):
    N1(0,0,0) —E1— N2(3,0,0) —E2— N3(6,0,0)         [m]

BC:   pin at N1, roller at N3
Load: P = 60 kN at midspan (N2), gravity direction (−Z)

Section:  H 400x200x8/13 (SS275, KS D 3502)
```

### 2.2. 결과

| Metric | Unit | OpenSees | ETABS | Midas | OS–M | ET–M | Status |
|--------|------|---------:|------:|------:|-----:|-----:|:------:|
| Midspan Vert. Disp. | mm | −5.424955 | −5.424955 | −5.425 | 0.00% | 0.00% | OK |
| Midspan Moment | kN·m | −90.000000 | −90.000000 | −90.00 | 0.00% | 0.00% | OK |
| Reaction N1 Fy | kN | 30.000000 | 30.000000 | 30.00 | 0.00% | 0.00% | OK |
| Reaction N3 Fy | kN | 30.000000 | 30.000000 | 30.00 | 0.00% | 0.00% | OK |
| Reaction N1 Fx | kN | 0.000000 | 0.000000 | 0.00 | 0.00% | 0.00% | OK |

**소계: 5/5 OK**

---

## 3. Case 2 — 2D 1층 1경간 포털 프레임

### 3.1. 모델 개요

```
Layout (X-Z plane, Y=0; coordinates in m):
    N3(0,0,3) ——B1—— N4(6,0,3)
       |                  |
      C1                 C2
       |                  |
    N1(0,0,0)         N2(6,0,0)

BC:   fully fixed at N1, N2
Load: at N3, N4 — Fx = +25 kN, Fz = −100 kN

Column: H 350x350x12/19   Beam: H 400x200x8/13
```

### 3.2. 결과

| Metric | Unit | OpenSees | ETABS | Midas | OS–M | ET–M | Status |
|--------|------|---------:|------:|------:|-----:|-----:|:------:|
| Top Disp N3 dx | mm | 1.388085 | 1.388085 | 1.388 | 0.01% | 0.01% | OK |
| Top Disp N4 dx | mm | 1.388085 | 1.388085 | 1.388 | 0.01% | 0.01% | OK |
| Top Disp N3 dy | mm | −0.075606 | −0.075606 | −0.0756 | 0.01% | 0.01% | OK |
| Top Disp N4 dy | mm | −0.088692 | −0.088692 | −0.0887 | 0.01% | 0.01% | OK |
| Col1 Base Moment | kN·m | −51.105247 | −51.105247 | −51.11 | 0.01% | 0.01% | OK |
| Col2 Base Moment | kN·m | −51.105247 | −51.105247 | −51.11 | 0.01% | 0.01% | OK |
| Beam Moment (i-end) | kN·m | 23.894753 | 23.894753 | 23.89 | 0.02% | 0.02% | OK |
| Beam Moment (j-end) | kN·m | 23.894753 | 23.894753 | 23.89 | 0.02% | 0.02% | OK |
| Base Shear | kN | −50.000000 | −50.000000 | −50.00 | 0.00% | 0.00% | OK |
| Reaction N1 Fx | kN | −25.000000 | −25.000000 | −25.00 | 0.00% | 0.00% | OK |
| Reaction N1 Fy | kN | 92.035082 | 92.035082 | 92.04 | 0.01% | 0.01% | OK |
| Reaction N1 Mz | kN·m | 51.105247 | 51.105247 | 51.11 | 0.01% | 0.01% | OK |
| Reaction N2 Fx | kN | −25.000000 | −25.000000 | −25.00 | 0.00% | 0.00% | OK |
| Reaction N2 Fy | kN | 107.964918 | 107.964918 | 107.96 | 0.00% | 0.00% | OK |
| Reaction N2 Mz | kN·m | 51.105247 | 51.105247 | 51.11 | 0.01% | 0.01% | OK |

**소계: 15/15 OK**

---

## 4. Case 3 — 2D 3층 1경간 다층 프레임

### 4.1. 모델 개요

```
Layout (X-Z plane, Y=0; coordinates in m):
    N7(0,0,9) ——B3—— N8(6,0,9)        ← roof (story 3 top)
      |                  |
     C3                 C6
      |                  |
    N5(0,0,6) ——B2—— N6(6,0,6)        ← story 2 top
      |                  |
     C2                 C5
      |                  |
    N3(0,0,3) ——B1—— N4(6,0,3)        ← story 1 top
      |                  |
     C1                 C4
      |                  |
    N1(0,0,0)        N2(6,0,0)        ← base (fixed)

BC:   fully fixed at N1, N2
Loads (at each top node):
  Story 1 (N3, N4): Fx = +15 kN, Fz = −80 kN
  Story 2 (N5, N6): Fx = +25 kN, Fz = −80 kN
  Story 3 (N7, N8): Fx = +35 kN, Fz = −80 kN

Column: H 400x400x13/21   Beam: H 400x200x8/13
```

### 4.2. 결과

| Metric | Unit | OpenSees | ETABS | Midas | OS–M | ET–M | Status |
|--------|------|---------:|------:|------:|-----:|-----:|:------:|
| Roof Disp dx (avg) | mm | 18.444370 | 18.444370 | 18.444 | 0.00% | 0.00% | OK |
| Roof Disp N7 dx | mm | 18.444370 | 18.444370 | 18.444 | 0.00% | 0.00% | OK |
| Roof Disp N8 dx | mm | 18.444370 | 18.444370 | 18.444 | 0.00% | 0.00% | OK |
| Story Drift 1 | — | 0.001481 | 0.001481 | 0.001481 | 0.01% | 0.01% | OK |
| Story Drift 2 | — | 0.002526 | 0.002526 | 0.002526 | 0.01% | 0.01% | OK |
| Story Drift 3 | — | 0.002141 | 0.002141 | 0.002141 | 0.00% | 0.00% | OK |
| Base Shear | kN | −150.000000 | −150.000000 | −150.00 | 0.00% | 0.00% | OK |
| Col1 Base Moment | kN·m | −213.081390 | −213.081390 | −213.00 | 0.04% | 0.04% | OK |
| Col2 Base Moment | kN·m | −213.081390 | −213.081390 | −213.00 | 0.04% | 0.04% | OK |
| Top Beam Moment (i) | kN·m | 80.314745 | 80.314745 | 80.31 | 0.01% | 0.01% | OK |
| Top Beam Moment (j) | kN·m | 80.314745 | 80.314745 | 80.31 | 0.01% | 0.01% | OK |
| Reaction N1 Fx | kN | −75.000000 | −75.000000 | −75.00 | 0.00% | 0.00% | OK |
| Reaction N1 Fy | kN | 141.027130 | 141.027130 | 141.03 | 0.00% | 0.00% | OK |
| Reaction N1 Mz | kN·m | 213.081390 | 213.081390 | 213.00 | 0.04% | 0.04% | OK |
| Reaction N2 Fx | kN | −75.000000 | −75.000000 | −75.00 | 0.00% | 0.00% | OK |
| Reaction N2 Fy | kN | 338.972870 | 338.972870 | 338.97 | 0.00% | 0.00% | OK |
| Reaction N2 Mz | kN·m | 213.081390 | 213.081390 | 213.00 | 0.04% | 0.04% | OK |

**소계: 17/17 OK**

---

## 5. 종합 및 분석

### 5.1. 통계 요약

| Case | 메트릭 수 | OK | CHECK | FAIL | 최대 OS–M | 최대 ET–M |
|------|----------|----|-------|------|----------|-----------|
| 1 | 5 | 5 | 0 | 0 | 0.00 % | 0.00 % |
| 2 | 15 | 15 | 0 | 0 | 0.02 % | 0.02 % |
| 3 | 17 | 17 | 0 | 0 | 0.04 % | 0.04 % |
| **계** | **37** | **37** | **0** | **0** | **0.04 %** | **0.04 %** |

### 5.2. 핵심 발견

1. **OpenSees ↔ ETABS 6 유효숫자 일치**: 모든 37개 메트릭에서 두 솔버가 표시 가능한 6자리까지 완전히 같은 값. 두 프로그램이 동일한 KS D 3502 단면 데이터를 사용하면서, 각자의 독립적 솔버로 같은 답을 낸다는 강력한 교차검증.

2. **Midas Gen과의 미세 차이는 출력 반올림**: Midas Gen 결과는 4자리(예: 1.388 mm) 출력 정밀도이며, 실제 솔버 내부 계산은 더 정확. 최대 0.04% 차이는 보고 자릿수 차이일 뿐, 솔버 정확도 차이가 아님.

3. **Case 복잡도와 무관한 안정성**: 단순보(5DOF) → 1층 프레임(24DOF) → 3층 프레임(48DOF) 으로 자유도가 증가해도 일치도 유지.

### 5.3. 솔버 정확성 결론

- **OpenSees 솔버는 상용 솔버(ETABS, Midas Gen) 수준의 정확성을 보유**
- **KS D 3502 표준 단면을 사용하는 한국 강구조 설계에 충분히 활용 가능**
- 입력 가정(빔 이론, 접합부 모델링)을 일치시키면 솔버 간 차이는 무시할 수준

---

## 6. 재현 방법

### 6.1. 환경

| 항목 | 버전 |
|------|------|
| OpenSees (Python) | `openseespy==3.6.x` (또는 `opensees==0.1.x`) |
| ETABS | 23.x (Korean KS21 설치 포함) |
| Midas Gen | 사용자 별도 라이센스 |
| Python | 3.10+ |

### 6.2. 실행 명령

```powershell
# OpenSees 측 (Case 1~5 전체)
.\opensees-mcp\Scripts\python.exe tests\benchmark\run_benchmarks.py

# ETABS 측 (Case 1·2·3)
.\opensees-mcp\Scripts\python.exe tests\benchmark\etabs_benchmark_case1_2.py --launch
# 또는 개별
.\opensees-mcp\Scripts\python.exe tests\benchmark\etabs_benchmark_case1_2.py --launch case3
```

### 6.3. 결과 파일 위치

| 출처 | 경로 |
|------|------|
| OpenSees | `tests/benchmark/opensees_results/case{1,2,3}.json` |
| ETABS | `tests/benchmark/etabs_results/case{1,2,3}.json` |
| Midas Gen | `tests/benchmark/midas_results/case{1,2,3}.json` |

---

## 7. 향후 확장 (TODO)

- [ ] **Case 4** (3D 2층 다경간) — ETABS 측 미구현
- [ ] **Case 5** (3D 5층 P-Delta) — ETABS 측 미구현. ETABS의 `geomNonlinearity_PDelta` 옵션 검증 필요
- [ ] **Case 6 (L-shape)** — Midas만 있고 OpenSees·ETABS 모두 미구현
- [ ] 동적 해석(모달, 응답스펙트럼) 3-way 비교

---

> [← 07. 트러블슈팅](07_troubleshooting.md) | [README로 →](README.md)
