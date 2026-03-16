# OpenSeesPy 전체 명령어 레퍼런스

> **버전:** openseespy 3.7.1.2 (2025.02.21 기준 최신)
> **Python 요구사항:** Python >= 3.9
> **공식 문서:** https://openseespydoc.readthedocs.io/

---

## 📋 목차

### OpenSeesPy 명령어 레퍼런스
1. [Model Commands (모델 정의)](#1-model-commands-모델-정의)
2. [Elements (요소)](#2-elements-요소)
3. [Materials (재료)](#3-materials-재료)
4. [Sections (단면)](#4-sections-단면)
5. [Constraints (구속조건)](#5-constraints-구속조건)
6. [Loading & Patterns (하중)](#6-loading--patterns-하중)
7. [Analysis Commands (해석)](#7-analysis-commands-해석)
8. [Output Commands (출력)](#8-output-commands-출력)
9. [Utility Commands (유틸리티)](#9-utility-commands-유틸리티)
10. [Advanced Features (고급 기능)](#10-advanced-features-고급-기능)

### MCP Tool 설계
11. [MCP Tool 설계 근거 조사](#-mcp-tool-설계-근거-조사)
12. [MCP Tool 분류 (High-Level Workflow)](#-mcp-tool-분류-high-level-workflow)

---

## 1. Model Commands (모델 정의)

### 1.1 기본 모델 명령어

| 명령어 | 설명 | Python 문법 |
|--------|------|-------------|
| `model()` | 모델 도메인 생성 | `ops.model('basic', '-ndm', 2, '-ndf', 3)` |
| `node()` | 절점 정의 | `ops.node(nodeTag, *coords)` |
| `mass()` | 질량 할당 | `ops.mass(nodeTag, *massValues)` |
| `region()` | 영역 정의 | `ops.region(regTag, '-eleOnly', *eleTags)` |
| `rayleigh()` | Rayleigh 감쇠 | `ops.rayleigh(alphaM, betaK, betaK0, betaKc)` |

### 1.2 기하학적 변환 (geomTransf)

| 변환 유형 | 설명 | 적용 |
|-----------|------|------|
| `Linear` | 선형 변환 | 소변형 해석 |
| `PDelta` | P-Delta 효과 | 2차 효과 고려 |
| `Corotational` | 공회전 변환 | 대변형 해석 |

```python
ops.geomTransf('Linear', transfTag)
ops.geomTransf('PDelta', transfTag, *vecxz)
ops.geomTransf('Corotational', transfTag, *vecxz)
```

### 1.3 보 적분 (beamIntegration)

| 적분 방식 | 설명 |
|-----------|------|
| `Lobatto` | Gauss-Lobatto 적분 |
| `Legendre` | Gauss-Legendre 적분 |
| `Radau` | Gauss-Radau 적분 |
| `NewtonCotes` | Newton-Cotes 적분 |
| `Trapezoidal` | 사다리꼴 적분 |
| `CompositeSimpson` | Simpson 적분 |
| `HingeRadau` | 힌지 요소용 |
| `HingeMidpoint` | 힌지 요소용 |
| `HingeEndpoint` | 힌지 요소용 |

---

## 2. Elements (요소)

### 2.1 Zero-Length Elements (영길이 요소)

| 요소 | 설명 | 용도 |
|------|------|------|
| `zeroLength` | 영길이 요소 | 스프링, 댐퍼 |
| `zeroLengthND` | nD 재료용 | 다축 거동 |
| `zeroLengthSection` | 단면용 | 소성힌지 |
| `CoupledZeroLength` | 연성 영길이 | 연성 거동 |
| `zeroLengthContact` | 접촉 요소 | 접촉 해석 |
| `zeroLengthContactNTS2D` | 2D NTS 접촉 | |
| `zeroLengthInterface2D` | 2D 인터페이스 | |
| `zeroLengthImpact3D` | 3D 충격 | 충격 해석 |

### 2.2 Truss Elements (트러스 요소)

| 요소 | 설명 |
|------|------|
| `Truss` | 기본 트러스 |
| `corotTruss` | 공회전 트러스 (대변형) |

```python
ops.element('Truss', eleTag, iNode, jNode, A, matTag)
ops.element('corotTruss', eleTag, iNode, jNode, A, matTag)
```

### 2.3 Beam-Column Elements (보-기둥 요소)

| 요소 | 설명 | 특징 |
|------|------|------|
| `elasticBeamColumn` | 탄성 보-기둥 | 선형 해석 |
| `ModElasticBeam2d` | 수정 탄성 보 | 강성 수정 |
| `ElasticTimoshenkoBeam` | 티모센코 보 | 전단변형 고려 |
| `beamWithHinges` | 힌지 보 | 집중소성 |
| `dispBeamColumn` | 변위기반 보 | 분산소성 |
| `forceBeamColumn` | 힘기반 보 | 분산소성 |
| `nonlinearBeamColumn` | 비선형 보 | 일반 비선형 |
| `dispBeamColumnInt` | 내부 힌지 보 | |
| `MVLEM` | 다중수직선 | 벽체 해석 |
| `SFI_MVLEM` | 전단-휨 상호작용 | 벽체 해석 |

### 2.4 Joint Elements (절점 요소)

| 요소 | 설명 |
|------|------|
| `beamColumnJoint` | 보-기둥 절점 |
| `ElasticTubularJoint` | 탄성 관형 절점 |
| `Joint2D` | 2D 절점 |

### 2.5 Link & Bearing Elements (링크 & 베어링)

| 요소 | 설명 | 용도 |
|------|------|------|
| `twoNodeLink` | 두절점 링크 | 일반 연결 |
| `elastomericBearingPlasticity` | 탄성체 베어링 | 교량 받침 |
| `elastomericBearingBoucWen` | Bouc-Wen 베어링 | 이력 거동 |
| `flatSliderBearing` | 평판 슬라이더 | 면진장치 |
| `singleFPBearing` | 단일 마찰진자 | 면진장치 |
| `TFP` | 삼중 마찰진자 | 면진장치 |
| `TripleFrictionPendulum` | 삼중 마찰진자 | 면진장치 |
| `multipleShearSpring` | 다중전단스프링 | |
| `KikuchiBearing` | Kikuchi 베어링 | |
| `YamamotoBiaxialHDR` | 고감쇠고무 | |
| `ElastomericX` | X형 탄성체 | |
| `LeadRubberX` | 납-고무 베어링 | 면진장치 |
| `HDR` | 고감쇠고무 | |
| `RJWatsonEqsBearing` | RJ Watson 베어링 | |
| `FPBearingPTV` | 마찰진자 (PTV) | |

### 2.6 Quadrilateral Elements (사각형 요소)

| 요소 | 설명 | 용도 |
|------|------|------|
| `quad` | 기본 사각형 | 평면응력/변형 |
| `ShellMITC4` | MITC4 쉘 | 판/쉘 해석 |
| `ShellDKGQ` | DKG 사각 쉘 | |
| `ShellDKGT` | DKG 삼각 쉘 | |
| `ShellNLDKGQ` | 비선형 DKG | 대변형 |
| `ShellNLDKGT` | 비선형 DKG 삼각 | |
| `ShellNL` | 비선형 쉘 | |
| `bbarQuad` | B-bar 사각형 | 체적잠김 방지 |
| `enhancedQuad` | 향상 변형률 | |
| `SSPquad` | SSP 사각형 | 안정화 |

### 2.7 Triangular Elements (삼각형 요소)

| 요소 | 설명 |
|------|------|
| `tri31` | 3절점 삼각형 |

### 2.8 Brick Elements (벽돌 요소)

| 요소 | 설명 | 절점 수 |
|------|------|---------|
| `stdBrick` | 표준 벽돌 | 8절점 |
| `bbarBrick` | B-bar 벽돌 | 8절점 |
| `20NodeBrick` | 20절점 벽돌 | 20절점 |
| `SSPbrick` | SSP 벽돌 | 8절점 |

### 2.9 Tetrahedron Elements (사면체 요소)

| 요소 | 설명 |
|------|------|
| `FourNodeTetrahedron` | 4절점 사면체 |

### 2.10 u-p Elements (포화토 요소)

| 요소 | 설명 |
|------|------|
| `quadUP` | 사각 u-p |
| `brickUP` | 벽돌 u-p |
| `bbarQuadUP` | B-bar 사각 u-p |
| `bbarBrickUP` | B-bar 벽돌 u-p |
| `NineFourNodeQuadUP` | 9-4절점 사각 |
| `TwentyEightNodeBrickUP` | 28-8절점 벽돌 |
| `SSPquadUP` | SSP 사각 u-p |
| `SSPbrickUP` | SSP 벽돌 u-p |

### 2.11 Contact Elements (접촉 요소)

| 요소 | 설명 |
|------|------|
| `SimpleContact2D` | 2D 단순 접촉 |
| `SimpleContact3D` | 3D 단순 접촉 |
| `BeamContact2D` | 2D 보 접촉 |
| `BeamContact3D` | 3D 보 접촉 |
| `BeamEndContact3D` | 3D 보 단부 접촉 |

### 2.12 Cable Elements (케이블 요소)

| 요소 | 설명 |
|------|------|
| `CatenaryCableElement` | 현수선 케이블 |

### 2.13 PFEM Elements (입자유한요소)

| 요소 | 설명 |
|------|------|
| `PFEMElementBubble` | PFEM 버블 요소 |
| `PFEMElementCompressible` | 압축성 PFEM |

### 2.14 Miscellaneous Elements (기타 요소)

| 요소 | 설명 | 용도 |
|------|------|------|
| `SurfaceLoad` | 표면하중 | 분포하중 |
| `VS3D4` | 점성 경계 | 동적 해석 |
| `AC3D8` | 음향 요소 | 음향 해석 |
| `ASI3D8` | 음향-구조 | 연성 해석 |
| `AV3D4` | 흡수 경계 | |
| `MasonPan12` | 조적패널 | 조적벽 |

---

## 3. Materials (재료)

### 3.1 Uniaxial Materials - Steel (강재)

| 재료 | 설명 | 특징 |
|------|------|------|
| `Steel01` | 이선형 강재 | 등방경화 |
| `Steel02` | Giuffre-Menegotto-Pinto | 바우싱거 효과 |
| `Steel4` | Steel02 확장 | 비대칭, 극한변형 |
| `ReinforcingSteel` | 철근 모델 | 좌굴, 피로 |
| `Dodd_Restrepo` | Dodd-Restrepo | 철근 이력 |
| `RambergOsgoodSteel` | Ramberg-Osgood | 곡선 항복 |
| `SteelMPF` | Menegotto-Pinto-Filippou | |
| `Steel01Thermal` | 열-강재 | 온도 의존 |

```python
# Steel01 예시
ops.uniaxialMaterial('Steel01', matTag, Fy, E0, b)

# Steel02 예시
ops.uniaxialMaterial('Steel02', matTag, Fy, E0, b, R0, cR1, cR2)
```

### 3.2 Uniaxial Materials - Concrete (콘크리트)

| 재료 | 설명 | 특징 |
|------|------|------|
| `Concrete01` | Kent-Scott-Park | 무인장강도 |
| `Concrete02` | 인장강화 | 인장강도 포함 |
| `Concrete04` | Popovics | 다양한 곡선 |
| `Concrete06` | Thorenfeldt | |
| `Concrete07` | Chang & Mander | |
| `Concrete01WithSITC` | SITC 포함 | |
| `ConfinedConcrete01` | 구속 콘크리트 | 자동 구속효과 |
| `ConcreteD` | 손상 콘크리트 | 손상역학 |
| `FRPConfinedConcrete` | FRP 구속 | FRP 보강 |
| `FRPConfinedConcrete02` | FRP 구속 v2 | |
| `ConcreteCM` | Chang-Mander | |
| `TDConcrete` | 시간의존 | 크리프, 건조수축 |
| `TDConcreteEXP` | 시간의존 (지수) | |
| `TDConcreteMC10` | MC2010 기반 | |
| `TDConcreteMC10NL` | MC2010 비선형 | |

```python
# Concrete01 예시
ops.uniaxialMaterial('Concrete01', matTag, fpc, epsc0, fpcu, epsU)

# Concrete02 예시
ops.uniaxialMaterial('Concrete02', matTag, fpc, epsc0, fpcu, epsU, lambda_, ft, Ets)
```

### 3.3 Uniaxial Materials - Standard (기본)

| 재료 | 설명 |
|------|------|
| `Elastic` | 선형 탄성 |
| `ElasticPP` | 탄소성 |
| `ElasticPPGap` | 갭 탄소성 |
| `ENT` | 인장무시 탄성 |
| `Hysteretic` | 이력 모델 |
| `Parallel` | 병렬 조합 |
| `Series` | 직렬 조합 |

### 3.4 Uniaxial Materials - Soil (지반)

| 재료 | 설명 | 용도 |
|------|------|------|
| `PySimple1` | p-y 스프링 | 수평 지반반력 |
| `TzSimple1` | t-z 스프링 | 축방향 마찰 |
| `QzSimple1` | q-z 스프링 | 선단지지 |
| `PyLiq1` | 액상화 p-y | 액상화 고려 |
| `TzLiq1` | 액상화 t-z | |
| `QzLiq1` | 액상화 q-z | |

### 3.5 Uniaxial Materials - Specialized (특수)

| 재료 | 설명 | 용도 |
|------|------|------|
| `Hardening` | 경화 모델 | 등방/이동 경화 |
| `CastFuse` | 주조 퓨즈 | 에너지 소산 |
| `ViscousDamper` | 점성 댐퍼 | 제진장치 |
| `BilinearOilDamper` | 오일 댐퍼 | |
| `Bilin` | 이선형 이력 | 보-기둥 힌지 |
| `ModIMKPeakOriented` | IMK 모델 | 구조물 붕괴 |
| `ModIMKPinching` | IMK 핀칭 | |
| `SAWS` | 목재 이력 | 목구조 |
| `BarSlip` | 철근 슬립 | 부착 파괴 |
| `Bond_SP01` | 부착-슬립 | |
| `Fatigue` | 피로 래퍼 | 저주기 피로 |
| `Impact` | 충격 | |
| `HyperbolicGap` | 쌍곡 갭 | |
| `LimitState` | 한계상태 | |
| `MinMax` | 최소최대 래퍼 | 파단 모사 |
| `ElasticBilin` | 탄성 이선형 | |
| `ElasticMultiLinear` | 탄성 다선형 | |
| `MultiLinear` | 다선형 이력 | |
| `InitStrain` | 초기변형률 래퍼 | 프리스트레스 |
| `InitStress` | 초기응력 래퍼 | |
| `PathIndependent` | 경로독립 래퍼 | |
| `Pinching4` | 4점 핀칭 | RC 부재 |
| `ECC01` | ECC 재료 | |
| `SelfCentering` | 자기복원 | 면진/제진 |
| `Viscous` | 점성 | |
| `BoucWen` | Bouc-Wen 이력 | 비선형 이력 |
| `BWBN` | Bouc-Wen-Baber-Noori | |
| `KikuchiAikenHDR` | 고감쇠고무 | 면진장치 |
| `KikuchiAikenLRB` | 납고무 | 면진장치 |
| `AxialSp` | 축방향 스프링 | |
| `AxialSpHD` | 축방향 스프링 HD | |
| `PinchingLimitStateMaterial` | 핀칭 한계상태 | |
| `CFSWSWP` | 냉간성형강 벽체 | |
| `CFSSSWP` | 냉간성형강 전단벽 | |

### 3.6 nDMaterial (다축 재료)

| 재료 | 설명 | 용도 |
|------|------|------|
| `ElasticIsotropic` | 등방 탄성 | 기본 3D |
| `ElasticOrthotropic` | 직교 이방성 | 복합재 |
| `J2Plasticity` | J2 소성 | 금속 |
| `DruckerPrager` | Drucker-Prager | 지반, 콘크리트 |
| `PlaneStress` | 평면응력 래퍼 | 2D 해석 |
| `PlaneStrain` | 평면변형률 래퍼 | 2D 해석 |
| `MultiaxialCyclicPlasticity` | 다축 반복소성 | |
| `BoundingCamClay` | Cam-Clay | 점성토 |
| `PlateFiber` | 판 섬유 | 쉘 요소 |
| `FSAM` | 고정균열 | RC 벽체 |
| `ManzariDafalias` | Manzari-Dafalias | 사질토 |
| `PM4Sand` | PM4Sand | 사질토 |
| `PM4Silt` | PM4Silt | 실트 |
| `StressDensityModel` | 응력밀도 | 사질토 |
| `AcousticMedium` | 음향 매질 | 음향 해석 |

### 3.7 nDMaterial - Contact (접촉)

| 재료 | 설명 |
|------|------|
| `ContactMaterial2D` | 2D 접촉 재료 |
| `ContactMaterial3D` | 3D 접촉 재료 |

### 3.8 nDMaterial - Soil (지반)

| 재료 | 설명 |
|------|------|
| `PressureIndependMultiYield` | 압력독립 다항복 |
| `PressureDependMultiYield` | 압력의존 다항복 |
| `PressureDependMultiYield02` | 압력의존 v2 |
| `PressureDependMultiYield03` | 압력의존 v3 |
| `FluidSolidPorousMaterial` | 유체-고체 다공 |

---

## 4. Sections (단면)

### 4.1 Section Types

| 단면 | 설명 | 용도 |
|------|------|------|
| `Elastic` | 탄성 단면 | 선형 해석 |
| `Fiber` | 섬유 단면 | 비선형 해석 |
| `FiberThermal` | 열-섬유 | 화재 해석 |
| `NDFiber` | nD 섬유 | 전단 고려 |
| `WFSection2d` | 광폭 플랜지 | 철골 |
| `RCSection2d` | RC 단면 | 철근콘크리트 |
| `RCCircularSection` | 원형 RC | 철근콘크리트 |
| `Parallel` | 병렬 단면 | 조합 단면 |
| `Aggregator` | 집합체 | 추가 자유도 |
| `Uniaxial` | 1축 단면 | 스프링 등 |
| `ElasticMembranePlateSection` | 탄성 판 | 쉘 해석 |
| `PlateFiber` | 섬유 판 | 비선형 쉘 |
| `Bidirectional` | 양방향 | |
| `Isolator2spring` | 면진 스프링 | 면진장치 |
| `LayeredShell` | 층상 쉘 | 복합 쉘 |

### 4.2 Fiber Section 구성

```python
# 섬유 단면 예시
ops.section('Fiber', secTag)
ops.patch('rect', matTag, numSubdivY, numSubdivZ, y1, z1, y2, z2)
ops.layer('straight', matTag, numFiber, areaFiber, y1, z1, y2, z2)
```

| 패치/층 명령 | 설명 |
|--------------|------|
| `patch('rect', ...)` | 사각형 패치 |
| `patch('quad', ...)` | 사변형 패치 |
| `patch('circ', ...)` | 원형 패치 |
| `layer('straight', ...)` | 직선 철근층 |
| `layer('circ', ...)` | 원형 철근층 |

---

## 5. Constraints (구속조건)

### 5.1 Single-Point Constraints (단일점 구속)

| 명령어 | 설명 |
|--------|------|
| `fix(nodeTag, *constrValues)` | 자유도별 구속 |
| `fixX(x, *constrValues, tol=)` | X좌표 기준 구속 |
| `fixY(y, *constrValues, tol=)` | Y좌표 기준 구속 |
| `fixZ(z, *constrValues, tol=)` | Z좌표 기준 구속 |

```python
# 예시: 2D 고정단
ops.fix(1, 1, 1, 1)  # Ux, Uy, Rz 모두 고정
```

### 5.2 Multi-Point Constraints (다중점 구속)

| 명령어 | 설명 |
|--------|------|
| `equalDOF(rNode, cNode, *dofs)` | 동일 자유도 |
| `equalDOF_Mixed(rNode, cNode, ...)` | 혼합 동일 자유도 |
| `rigidDiaphragm(perpDirn, rNode, *cNodes)` | 강체 다이어프램 |
| `rigidLink(type, rNode, cNode)` | 강체 링크 |

---

## 6. Loading & Patterns (하중)

### 6.1 Time Series

| 유형 | 설명 | 용도 |
|------|------|------|
| `Constant` | 상수 | 정적 해석 |
| `Linear` | 선형 | 하중 증분 |
| `Trig` | 삼각함수 | 조화 하중 |
| `Triangle` | 삼각파 | |
| `Rectangular` | 구형파 | 충격 하중 |
| `Pulse` | 펄스 | |
| `Path` | 경로 | 지진 기록 |

```python
ops.timeSeries('Constant', tsTag)
ops.timeSeries('Linear', tsTag)
ops.timeSeries('Path', tsTag, '-dt', dt, '-values', *values)
```

### 6.2 Load Patterns

| 유형 | 설명 | 용도 |
|------|------|------|
| `Plain` | 일반 하중 | 정적 하중 |
| `UniformExcitation` | 균일 가진 | 지진 해석 |
| `MultipleSupport` | 다점 지지 | 비동시 가진 |

```python
ops.pattern('Plain', patternTag, tsTag)
ops.load(nodeTag, *loadValues)
```

### 6.3 하중 명령어

| 명령어 | 설명 |
|--------|------|
| `load(nodeTag, *loadValues)` | 절점 하중 |
| `eleLoad('-ele', eleTag, '-type', ...)` | 요소 하중 |
| `sp(nodeTag, dof, value)` | 단일점 지정 |

### 6.4 Ground Motion

| 유형 | 설명 |
|------|------|
| `Plain` | 일반 지반운동 |
| `Interpolated` | 보간 지반운동 |

---

## 7. Analysis Commands (해석)

### 7.1 Constraint Handlers

| 핸들러 | 설명 | 용도 |
|--------|------|------|
| `Plain` | 기본 | 단순 구속 |
| `Lagrange` | 라그랑주 승수 | 정확한 구속 |
| `Penalty` | 페널티 방법 | 대규모 시스템 |
| `Transformation` | 변환 방법 | 추천 |

```python
ops.constraints('Transformation')
```

### 7.2 Numberers

| 넘버러 | 설명 |
|--------|------|
| `Plain` | 기본 순서 |
| `RCM` | Reverse Cuthill-McKee |
| `AMD` | Approximate Minimum Degree |
| `ParallelPlain` | 병렬 기본 |
| `ParallelRCM` | 병렬 RCM |

```python
ops.numberer('RCM')
```

### 7.3 System Solvers

| 솔버 | 설명 | 특징 |
|------|------|------|
| `BandGeneral` | 밴드 일반 | 비대칭 |
| `BandSPD` | 밴드 SPD | 대칭 양정치 |
| `ProfileSPD` | 프로파일 SPD | 대칭 양정치 |
| `SuperLU` | SuperLU | 희소 직접 |
| `UmfPack` | UmfPack | 희소 직접 |
| `FullGeneral` | 전체 일반 | 소규모 |
| `SparseSYM` | 희소 대칭 | 대칭 |
| `Mumps` | MUMPS | 병렬 직접 |

```python
ops.system('BandGeneral')
ops.system('UmfPack')
```

### 7.4 Convergence Tests

| 테스트 | 설명 |
|--------|------|
| `NormUnbalance` | 불균형력 노름 |
| `NormDispIncr` | 변위증분 노름 |
| `EnergyIncr` | 에너지 증분 |
| `RelativeNormUnbalance` | 상대 불균형력 |
| `RelativeNormDispIncr` | 상대 변위증분 |
| `RelativeTotalNormDispIncr` | 상대 총 변위 |
| `RelativeEnergyIncr` | 상대 에너지 |
| `FixedNumIter` | 고정 반복수 |
| `NormDispAndUnbalance` | 변위 AND 불균형 |
| `NormDispOrUnbalance` | 변위 OR 불균형 |

```python
ops.test('NormDispIncr', tol, maxIter)
ops.test('EnergyIncr', tol, maxIter, printFlag)
```

### 7.5 Algorithms

| 알고리즘 | 설명 | 용도 |
|----------|------|------|
| `Linear` | 선형 | 탄성 해석 |
| `Newton` | 뉴턴-랩슨 | 일반 비선형 |
| `NewtonLineSearch` | 선탐색 뉴턴 | 수렴 개선 |
| `ModifiedNewton` | 수정 뉴턴 | 강성 고정 |
| `KrylovNewton` | 크릴로프 뉴턴 | 대규모 |
| `SecantNewton` | 시컨트 뉴턴 | |
| `RaphsonNewton` | 랩슨 뉴턴 | |
| `PeriodicNewton` | 주기적 뉴턴 | |
| `BFGS` | BFGS | 준뉴턴 |
| `Broyden` | 브로이든 | 준뉴턴 |

```python
ops.algorithm('Newton')
ops.algorithm('NewtonLineSearch', '-type', 'Bisection')
```

### 7.6 Integrators - Static

| 적분기 | 설명 | 용도 |
|--------|------|------|
| `LoadControl` | 하중 제어 | 하중 증분 |
| `DisplacementControl` | 변위 제어 | 변위 증분 |
| `ParallelDisplacementControl` | 병렬 변위 제어 | |
| `MinUnbalDispNorm` | 최소 불균형 | 스냅스루 |
| `ArcLength` | 호장법 | 스냅백 |

```python
ops.integrator('LoadControl', incr)
ops.integrator('DisplacementControl', nodeTag, dof, incr)
ops.integrator('ArcLength', s, alpha)
```

### 7.7 Integrators - Transient

| 적분기 | 설명 | 안정성 |
|--------|------|--------|
| `CentralDifference` | 중앙 차분 | 조건부 |
| `Newmark` | 뉴마크 | 무조건 (γ≥0.5, β≥0.25) |
| `HHT` | Hilber-Hughes-Taylor | 무조건 |
| `GeneralizedAlpha` | 일반화 알파 | 무조건 |
| `TRBDF2` | TRBDF2 | 무조건 |
| `ExplicitDifference` | 명시적 차분 | 조건부 |

```python
ops.integrator('Newmark', gamma, beta)
ops.integrator('HHT', alpha)
```

### 7.8 Analysis Types

| 유형 | 명령어 | 설명 |
|------|--------|------|
| 정적 | `analysis('Static')` | 정적 해석 |
| 동적 | `analysis('Transient')` | 시간이력 해석 |
| 가변동적 | `analysis('VariableTransient')` | 가변 시간증분 |

### 7.9 Analysis Execution

| 명령어 | 설명 |
|--------|------|
| `analyze(numSteps)` | 해석 수행 (정적) |
| `analyze(numSteps, dt)` | 해석 수행 (동적) |
| `eigen(numModes)` | 고유치 해석 |
| `modalProperties('-print')` | 모달 특성 |
| `responseSpectrumAnalysis(...)` | 응답스펙트럼 해석 |

---

## 8. Output Commands (출력)

### 8.1 Recorder Types

| 레코더 | 설명 | 용도 |
|--------|------|------|
| `Node` | 절점 레코더 | 변위, 속도, 가속도, 반력 |
| `Element` | 요소 레코더 | 응력, 변형률, 힘 |
| `EnvelopeNode` | 절점 포락 | 최대/최소값 |
| `EnvelopeElement` | 요소 포락 | |
| `PVD` | ParaView 출력 | 시각화 |
| `Drift` | 층간변위 | 건물 해석 |
| `Collapse` | 붕괴 레코더 | 붕괴 해석 |

```python
ops.recorder('Node', '-file', 'disp.out', '-time', '-node', 2, '-dof', 1, 2, 'disp')
ops.recorder('Element', '-file', 'ele.out', '-time', '-ele', 1, 'force')
```

### 8.2 Node Output

| 응답 유형 | 설명 |
|-----------|------|
| `disp` | 변위 |
| `vel` | 속도 |
| `accel` | 가속도 |
| `incrDisp` | 증분 변위 |
| `reaction` | 반력 |
| `rayleighForces` | 레일리 감쇠력 |
| `pressure` | 간극수압 |
| `eigen` | 고유벡터 |

### 8.3 Element Output

요소 유형별 출력 항목이 다름:

**Truss/Beam-Column:**
- `force`, `localForce`, `globalForce`
- `deformation`, `deformations`
- `stiff`, `stiffness`

**Section:**
- `section`, `fiber`
- `deformation`, `force`

### 8.4 즉시 출력 명령

| 명령어 | 설명 |
|--------|------|
| `nodeDisp(nodeTag, dof)` | 절점 변위 |
| `nodeVel(nodeTag, dof)` | 절점 속도 |
| `nodeAccel(nodeTag, dof)` | 절점 가속도 |
| `nodeReaction(nodeTag, dof)` | 절점 반력 |
| `nodeCoord(nodeTag, dim)` | 절점 좌표 |
| `nodeEigenvector(nodeTag, mode, dof)` | 고유벡터 |
| `eleForce(eleTag, dof)` | 요소 힘 |
| `eleResponse(eleTag, *args)` | 요소 응답 |

---

## 9. Utility Commands (유틸리티)

### 9.1 Model Management

| 명령어 | 설명 |
|--------|------|
| `wipe()` | 모델 초기화 |
| `wipeAnalysis()` | 해석만 초기화 |
| `reset()` | 상태 리셋 |
| `remove('node', nodeTag)` | 절점 제거 |
| `remove('element', eleTag)` | 요소 제거 |
| `remove('loadPattern', patternTag)` | 하중패턴 제거 |
| `remove('sp', nodeTag, dof, patternTag)` | SP 제거 |

### 9.2 State Management

| 명령어 | 설명 |
|--------|------|
| `loadConst('-time', pseudoTime)` | 하중 상수화 |
| `setTime(newTime)` | 시간 설정 |
| `getTime()` | 현재 시간 |
| `setCreep(creepFlag)` | 크리프 설정 |

### 9.3 Node Property Setters

| 명령어 | 설명 |
|--------|------|
| `setNodeCoord(nodeTag, dim, value)` | 좌표 설정 |
| `setNodeDisp(nodeTag, dof, value)` | 변위 설정 |
| `setNodeVel(nodeTag, dof, value)` | 속도 설정 |
| `setNodeAccel(nodeTag, dof, value)` | 가속도 설정 |

### 9.4 Information Retrieval

| 명령어 | 설명 |
|--------|------|
| `getNP()` | 프로세서 수 |
| `getPID()` | 프로세서 ID |
| `getNumElements()` | 요소 수 |
| `getNumNodes()` | 절점 수 |
| `getNodeTags()` | 절점 태그 목록 |
| `getEleTags()` | 요소 태그 목록 |
| `getNodeDOFs(nodeTag)` | 절점 자유도 |
| `getLoadFactor(patternTag)` | 하중계수 |

### 9.5 Print Commands

| 명령어 | 설명 |
|--------|------|
| `printModel()` | 모델 출력 |
| `printModel('node', nodeTag)` | 절점 정보 |
| `printModel('ele', eleTag)` | 요소 정보 |

---

## 10. Advanced Features (고급 기능)

### 10.1 Sensitivity Analysis

| 명령어 | 설명 |
|--------|------|
| `parameter(tag, *args)` | 파라미터 정의 |
| `addToParameter(tag, *args)` | 파라미터 추가 |
| `updateParameter(tag, value)` | 파라미터 업데이트 |
| `setParameter('-val', value, '-ele', *tags)` | 파라미터 설정 |
| `getParamValue(tag)` | 파라미터 값 |
| `getParamTags()` | 파라미터 태그 |
| `computeGradients()` | 기울기 계산 |
| `sensitivityAlgorithm('-computeAtEachStep')` | 민감도 알고리즘 |

### 10.2 Mesh Generation

| 명령어 | 설명 |
|--------|------|
| `mesh('line', ...)` | 선 메쉬 |
| `mesh('tri', ...)` | 삼각 메쉬 |
| `mesh('quad', ...)` | 사각 메쉬 |
| `mesh('tet', ...)` | 사면체 메쉬 |
| `mesh('part', ...)` | 입자 메쉬 |
| `remesh()` | 리메쉬 |

### 10.3 PFEM (Particle Finite Element Method)

| 명령어 | 설명 |
|--------|------|
| `integrator('PFEM')` | PFEM 적분기 |
| `system('PFEM')` | PFEM 시스템 |
| `mesh('part', ...)` | 입자 생성 |
| `mesh('bg', ...)` | 배경 메쉬 |

### 10.4 Parallel Computing

| 명령어 | 설명 |
|--------|------|
| `getNP()` | 프로세서 수 |
| `getPID()` | 프로세서 ID |
| `send('-pid', pid, '-data', data)` | 데이터 전송 |
| `recv('-pid', pid)` | 데이터 수신 |
| `barrier()` | 동기화 |
| `Bcast(*data)` | 브로드캐스트 |
| `partition('-ncuts', ncuts)` | 도메인 분할 |

### 10.5 Friction Models

| 모델 | 설명 |
|------|------|
| `Coulomb` | 쿨롱 마찰 |
| `VelDependent` | 속도 의존 |
| `VelNormalFrcDep` | 속도-수직력 의존 |
| `VelPressureDep` | 속도-압력 의존 |
| `VelDepMultiLinear` | 다선형 속도 의존 |

```python
ops.frictionModel('Coulomb', frnTag, mu)
```

---

## 📊 MCP Tool 설계 근거 조사

기존 MCP 구현체들의 Tool 분류 방식을 조사하여 OpenSeesPy MCP 설계 방향을 결정했습니다.

### 조사한 MCP 구현체들

#### 1. FreeCAD MCP (High-Level 방식)
- **GitHub:** https://github.com/kitsunehunter/freecad-mcp
- **Tool 수:** 10개
- **설계 방식:** High-Level Workflow
- **주요 Tool:**
  - `create_parametric_shape()` - 파라메트릭 형상 생성
  - `create_assembly()` - 조립체 생성
  - `export_to_step()` - STEP 파일 내보내기
  - `create_sketch()` - 스케치 생성
- **특징:** 하나의 Tool 호출로 전체 워크플로우 처리

#### 2. Modelica Simulation MCP (High-Level Workflow)
- **GitHub:** https://github.com/modelica/modelica-mcp
- **Tool 수:** 2개
- **설계 방식:** High-Level Workflow
- **주요 Tool:**
  - `simulate_model()` - 모델 시뮬레이션 (전체 과정 자동화)
  - `get_simulation_results()` - 결과 조회
- **특징:** 극도로 단순화된 인터페이스, LLM이 복잡한 시퀀스를 알 필요 없음

#### 3. AutoCAD MCP (Low-Level Primitives)
- **GitHub:** https://github.com/autodesk/autocad-mcp
- **Tool 수:** 20개+
- **설계 방식:** Low-Level Primitives
- **주요 Tool:**
  - `draw_line()`, `draw_circle()`, `draw_arc()`
  - `set_layer()`, `set_color()`
  - `create_block()`, `insert_block()`
- **특징:** CAD 명령어를 1:1로 매핑, LLM이 순서대로 호출해야 함

#### 4. GitHub MCP Server (Toolset 기반)
- **GitHub:** https://github.com/modelcontextprotocol/servers
- **Tool 수:** 100개+
- **설계 방식:** Toolset 기반 그룹화
- **그룹:**
  - `repos` - 저장소 관련 도구
  - `issues` - 이슈 관련 도구
  - `pull_requests` - PR 관련 도구
- **특징:** 도메인별로 Tool을 그룹화, 필요한 Toolset만 활성화

### MCP 설계 패턴 (Klavis.ai 참조)

Klavis.ai의 MCP 설계 패턴 문서에서 권장하는 방식:

| 패턴 | 설명 | 적용 사례 |
|------|------|-----------|
| **Workflow-Based** | 하나의 Tool이 전체 작업 수행 | 시뮬레이션, 해석 |
| **Primitive-Based** | 기본 단위 명령어 제공 | CAD, 그래픽 |
| **Toolset-Based** | 도메인별 Tool 그룹화 | API 연동, 데이터베이스 |

### 비교 요약

| 접근 방식 | Tool 수 | LLM 부담 | 사용성 | 적용 |
|-----------|---------|----------|--------|------|
| High-Level Workflow | 적음 (5-10개) | 낮음 | 높음 | **시뮬레이션, 해석** |
| Low-Level Primitives | 많음 (20+) | 높음 | 낮음 | CAD, 그래픽 |
| Toolset-Based | 중간 | 중간 | 중간 | API, 데이터 |

### 결론

**OpenSeesPy MCP는 High-Level Workflow 방식을 채택합니다.**

- **근거 1:** FreeCAD, Modelica 등 엔지니어링 시뮬레이션 MCP들이 모두 High-Level 방식 사용
- **근거 2:** 구조 해석은 정해진 워크플로우가 있음 (모델링 → 하중 → 해석 → 결과)
- **근거 3:** LLM이 OpenSees 내부 명령어 순서를 알 필요 없이 "단순보 해석해줘"로 가능
- **근거 4:** Midas MCP도 동일한 High-Level 방식으로 설계됨

---

## 🎯 MCP Tool 분류 (High-Level Workflow)

FreeCAD, Modelica 등 엔지니어링 시뮬레이션 MCP 사례를 참고하여 High-Level Workflow 방식으로 설계합니다.
하나의 Tool이 전체 해석 워크플로우를 처리합니다.

### 구조물 형태별 해석 Tool

#### `analyze_simple_beam()` - 단순보 해석
양단 지지 보의 정적 해석을 수행합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 스팬 길이, 단면(폭/높이), 재료(E, fy), 하중(집중/분포) |
| **출력** | 최대 처짐, 최대 모멘트, 최대 전단력, 응력 |
| **내부** | node → element → material → section → load → analysis 자동 구성 |

#### `analyze_cantilever()` - 캔틸레버 해석
일단 고정 보의 정적 해석을 수행합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 캔틸레버 길이, 단면, 재료, 하중(끝단 집중/분포) |
| **출력** | 끝단 처짐, 고정단 모멘트, 전단력 분포 |
| **내부** | 고정단 경계조건 자동 설정 |

#### `analyze_frame()` - 라멘 프레임 해석
다층 라멘 구조물의 해석을 수행합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 층수, 스팬수, 층고, 스팬길이, 기둥/보 단면, 하중 |
| **출력** | 층별 변위, 부재력, 지점 반력 |
| **내부** | 절점/요소 자동 생성, 강접합 조건 적용 |

#### `analyze_truss()` - 트러스 해석
트러스 구조물의 해석을 수행합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 트러스 형태(프랫/하우/워렌), 스팬, 높이, 부재 단면 |
| **출력** | 부재별 축력, 절점 변위 |
| **내부** | 핀접합 조건, Truss 요소 사용 |

### 해석 유형별 Tool

#### `run_pushover()` - 푸시오버 해석
비선형 정적 해석(푸시오버)을 수행합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 구조물 정보, 하중 패턴(역삼각형/균일), 목표 변위 |
| **출력** | Base Shear-Displacement 곡선, 소성 힌지 형성 순서 |
| **내부** | DisplacementControl integrator, 비선형 재료 적용 |

#### `run_time_history()` - 시간이력 해석
동적 시간이력 해석을 수행합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 구조물 정보, 지진파 데이터(acc/time), 감쇠비 |
| **출력** | 시간별 변위/속도/가속도 이력, 최대 응답 |
| **내부** | Transient analysis, Newmark integrator |

#### `run_eigen_analysis()` - 고유치 해석
고유치 해석을 수행하여 모드 형상과 고유주기를 산출합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 구조물 정보, 모드 수 |
| **출력** | 고유주기, 고유진동수, 모드 형상 |
| **내부** | `eigen()` 명령 활용 |

### 보조 Tool

#### `get_section_properties()` - 단면 정보 조회
표준 단면(H형강, 각형강관 등)의 단면 특성을 조회합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 단면 이름 (e.g., 'H-400x200x8x13') |
| **출력** | A, Ix, Iy, Sx, Sy, rx, ry |

#### `get_material_properties()` - 재료 정보 조회
표준 재료(SS400, SM490 등)의 물성치를 조회합니다.

| 구분 | 내용 |
|------|------|
| **입력** | 재료 이름 (e.g., 'SS400') |
| **출력** | E, fy, fu, 포아송비, 밀도 |

---

## 참고 자료

### OpenSeesPy
- [OpenSeesPy 공식 문서](https://openseespydoc.readthedocs.io/)
- [OpenSeesPy GitHub](https://github.com/zhuminjie/OpenSeesPy)
- [OpenSees Wiki](https://opensees.berkeley.edu/wiki/)
- [PyPI - openseespy](https://pypi.org/project/openseespy/)

### MCP 설계 참고
- [FreeCAD MCP](https://github.com/kitsunehunter/freecad-mcp) - High-Level Workflow 방식
- [Modelica MCP](https://github.com/modelica/modelica-mcp) - 시뮬레이션 MCP 사례
- [GitHub MCP Server](https://github.com/modelcontextprotocol/servers) - Toolset 기반 설계
- [Klavis.ai MCP Design Patterns](https://klavis.ai/blog/mcp-design-patterns) - MCP 설계 패턴 가이드
