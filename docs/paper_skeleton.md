# Paper Skeleton v3.1 — Paragraph-Level Detail (Revised)

# IFC 기반 구조해석 자동화 프레임워크 + KDS 하중 DB + OpenSeesPy 검증

> SCI 스타일 논문 골격 (v3.1). 각 문단의 역할·핵심 주장·근거(문헌 + 로컬 파일)를 구체화.
> 참고 논문: Leonardi et al. (2024) Methods 구조, Wang et al. (2025) Stage 구성, Hasan et al. (2019) 문제 정의, Fernández-Mora et al. (2022) 문헌 리뷰.
> 2026-03-17 v3.1 개정.
>
> **v3 → v3.1 주요 변경 (6건):**
> 1. **A4 경량화**: DB 건수·기준명 제거, 파이프라인 뼈대만 유지 (초록 ≠ 기능 카탈로그)
> 2. **LLM Abstract 후퇴**: A3에서 LLM 제거 → A6 마지막에 한 줄 배치 (심사자 오분류 방지)
> 3. **I3 gap 요약 prose**: 표 앞에 한 문장 요약 추가 (Leonardi 키워드 압축 패턴)
> 4. **I6 scope 이유**: 왜 정형 철골인지 3가지 구체적 이유 추가 (심사자 방어)
> 5. **Design Check 비중 유지**: AISC 공식 상세를 Appendix로 이동 안내 (현재 수준 유지, 확장 방지)
> 6. **Visualization 연구 역할**: "verification aid" / "post-processing transparency" 명시

---

# 0. TITLE STRATEGY

## 추천 제목

**T2: "Integrating IFC Parsing and KDS-Based Load Database for Automated Structural Analysis: Framework and Benchmark Validation"**

**이유:**
- 핵심 기여(IFC 파싱 + KDS 하중 DB + 벤치마크 검증)를 정확히 반영
- LLM은 §2.3에서 보조 인터페이스로 소개하면 충분
- "Framework and Benchmark Validation"이 방법론 + 검증을 모두 포함

**대안 (LLM 포함 시):** T8 — "BIM-to-Analysis Automation via IFC and KDS Database: Framework, Validation, and Preliminary LLM Interface"

---

# 1. ABSTRACT (~250 words, 6 segments)

> **참고:** Wang et al. (2025) 초록 구조 — 배경(BIM 가치) → gap(bridge 부재) → 제안(framework) → 파이프라인 압축 → feasibility 결과

---

### [A1] Background (2문장)

**목표:** BIM/IFC 기반 구조해석 자동화의 산업적 필요성 제시
**핵심 주장:**
- BIM과 IFC 표준이 건설 산업의 상호운용 데이터 교환에 핵심이 되었으나, BIM 데이터를 해석 가능한 구조모델로 변환하는 과정은 여전히 수작업에 의존한다.
- IFC 기반 기하/물성 정보를 기준 적합 하중 정의 및 유한요소해석(FEA)과 일관된 워크플로우로 연결하는 것은 미해결 연구 과제이다.

**모델 문장 (Wang, 2025 Abstract):**
> *"Despite the rich structural information encapsulated within BIM models, the lack of effective data exchange bridges between architectural design and structural analysis stages presents challenges."*

**근거:** Fernández-Mora et al. (2022) — BIM adoption 리뷰, buildingSMART IFC schema (ISO 16739-1:2018)

---

### [A2] Topic Declaration (1문장)

**목표:** 논문 identity를 한 문장으로 선언
**핵심 주장:**
- 본 논문은 정형 철골 프레임 건물의 자동 구조해석을 위한 프레임워크를 제시한다: IFC 기반 기하 추출, 한국건설기준(KDS) 파라미터 DB 기반 하중 생성, OpenSeesPy 기반 3D FEA를 통합한다.

**모델 문장 (Leonardi, 2024 Abstract):**
> *"This paper introduces an open-source automated solid finite element analysis method using OpenBIM data (IFC)."*

**근거 파일:**
- [mcp-server/server.py](mcp-server/server.py) — 14 MCP tools, 전체 파이프라인 조율
- [mcp-server/core/building_model.py](mcp-server/core/building_model.py) — BuildingModel IR 정의

---

### [A3] Research Objective (1문장)

**목표:** 연구 목적을 한 문장으로 — LLM 언급 없이 핵심 파이프라인에 집중
**핵심 주장:**
- 본 연구의 목적은 IFC 파일과 설계기준 DB로부터 기하, 하중, 해석 파라미터를 자동 도출하여, 정형 프레임 구조에 대한 수동 모델링 노력을 줄이는 것의 실현가능성을 보이는 것이다.

> **v3→v3.1 변경:** LLM 문장을 A3에서 제거. 초록 앞쪽에 LLM이 등장하면 심사자가 논문을 "LLM interface paper"로 오분류할 위험이 있음. LLM은 A6 마지막에 한 줄로 이동.

**근거 파일:**
- [mcp-server/core/ifc_parser.py](mcp-server/core/ifc_parser.py) — IFC→구조 데이터
- [mcp-server/core/building_model.py](mcp-server/core/building_model.py) — BuildingModel IR

---

### [A4] Methodology Summary (2–3문장)

**목표:** 방법론의 순차적 절차를 파이프라인처럼 압축

**모델 문장 (Wang, 2025 Abstract):**
> *"Leveraging the Revit-Dynamo platform, we extract geometric and material information... A custom program named BTO is developed using Python to comprehensively configure element types, material constitutive models, load information, and analysis types for OpenSees simulations."*

**핵심 주장:**
- 제안 워크플로우는 세 단계로 구성된다: (1) IFC 파싱을 통한 기하·단면·재료 추출; (2) 한국건설기준(KDS) 파라미터 DB에 기반한 고정·활·지진·풍 하중 및 하중조합의 자동 생성; (3) OpenSeesPy를 이용한 3D 유한요소해석 및 초보적 설계검토.
- 해석 엔진은 Corotational 기하비선형 정식화를 지원하며, 웹 기반 3D 에디터가 결과 검토를 위해 제공된다.

> **v3→v3.1 변경:** A4에서 구현 스펙 숫자(712, 2,290, 18, 6-DOF, AISC 360)를 제거하고 파이프라인 뼈대만 유지. 초록의 목적은 기능 목록이 아니라 문제–제안–방법–결과–의의를 압축하는 것. 상세 숫자는 본문 Methods와 Table로 이동.

**근거 파일:**
- [mcp-server/core/load_generator.py](mcp-server/core/load_generator.py) — 4종 하중
- [mcp-server/core/frame_3d.py](mcp-server/core/frame_3d.py) — 3D FEA
- [mcp-server/core/design_check.py](mcp-server/core/design_check.py) — 설계검토

---

### [A5] Key Results (2문장)

**목표:** 정량적 검증 결과 요약

**핵심 주장:**
- Midas Gen 대비 5개 케이스 벤치마크 비교(112개 응답 메트릭)에서 양호한 일치를 보임: 100개 메트릭 1% 미만 차이, 12개 메트릭 3% 미만 차이(후자는 3D 보-기둥 요소 강성 정식화 차이에 기인).
- 대표적 다층 건물 예제에 워크플로우를 적용하여, IFC 유래 모델과 DB 생성 하중이 검증 범위 내에서 구조적으로 일관된 결과를 산출함을 보임.

**톤 조정:** ~~"commercial-grade accuracy"~~ → "close agreement" / "structurally consistent results within the validated scope"

**근거 파일:**
- [tests/benchmark/cases.py](tests/benchmark/cases.py) — 5개 케이스
- [tests/benchmark/compare.py](tests/benchmark/compare.py) — 비교 유틸리티

---

### [A6] Significance and Limitations (1–2문장)

**목표:** 기여의 의미 + 향후 과제

**모델 문장 (Leonardi, 2024 Abstract):**
> *"The methodology is applied to an actual masonry building aggregate to validate its capability of working with complex geometries and scalability."*

**핵심 주장:**
- 결과는 IFC-to-analysis 파이프라인이 기준 기반 하중 DB와 결합될 때, 확립된 상용 소프트웨어와 양호한 대응을 유지하면서 정형 프레임 해석의 수동 노력을 상당히 줄일 수 있음을 시사한다.
- 이 프레임워크는 보조적 자연어 입력 계층도 수용하며, 그 체계적 평가는 향후 연구로 남긴다.
- 향후 연구: 비정형 기하, 비선형 재료, 더 넓은 설계기준 적용.

> **v3→v3.1 변경:** LLM을 A6 마지막에 "supplementary natural-language input layer"로 한 줄만 배치. A3에서 제거된 내용의 최소한 언급. "whose systematic evaluation is deferred to future work"로 범위 한정.

---

# 2. INTRODUCTION (~1500 words, 7 paragraphs)

> **참고 전략:**
> - **¶1–¶2**: Hasan (2019) 스타일 — 문제를 직접적으로 제시, "BIM model ≠ analysis-ready model"
> - **¶3**: Leonardi (2024) 스타일 — 기존 연구를 "laborious, semi-automatic, proprietary-dependent"로 압축
> - **¶3**: Fernández-Mora (2022) — "tendency towards design tools and new buildings" 연구 편중 지적
> - **¶5**: Wang (2025) 스타일 — 범위 선언 + 구성 안내
> - **¶7**: Leonardi (2024) 스타일 — 번호 기여점 + 이번 논문 범위 한정

---

### [I1] Digital Transformation and the BIM–Analysis Gap (¶1, ~200 words)

**목표:** BIM의 확산 배경을 제시하되, 곧바로 구조해석과의 단절 문제를 연결

**문단 구조:**
1. BIM이 AEC 산업에서 3D 모델링, 정보 통합, 협업 향상을 가져왔다는 배경 (2문장)
2. 그러나 BIM 모델이 곧 구조해석 모델이 되지는 않는다는 핵심 문제 (1문장)
3. 상용 FEA 소프트웨어가 존재함에도, BIM 모델→해석 모델 변환은 여전히 수작업 병목이다 (1문장)

**핵심 주장:**
- BIM은 건물의 3D 기하, 재료, 부재 관계를 풍부하게 기술하지만, 이 정보가 곧바로 구조해석에 사용 가능한 형태(analysis-ready)는 아니다.
- 기존 BIM 소프트웨어의 AMV(Analytical Model View)는 부재를 중심선/와이어프레임으로 축소하며, 이 과정에서 구조적 세부정보가 손실될 수 있다.

**참고 문장 (Hasan, 2019 Abstract):**
> *"Current BIM models impose restrictions on the geometry of building members in their analytical models, where components are fitted to wireframe representations. This unnecessary reduction in geometrical representation drives the loss of structural details and may lead to defective structural analysis."*

**참고 문장 (Wang, 2025 Introduction):**
> *"Despite the rich structural information encapsulated within BIM models, the lack of effective data exchange bridges between architectural design and structural analysis stages presents challenges."*

**문헌 근거:**
- **Hasan et al. (2019)** — BIM AMV의 기하학적 제약과 구조적 세부 손실
- **Wang et al. (2025)** — BIM 정보의 풍부성과 해석 단계 교환 브릿지 부재
- **buildingSMART IFC schema (ISO 16739-1:2018)** — vendor-neutral openBIM 표준
- 일반 BIM 채택 동향: Fernández-Mora et al. (2022)

**본 연구와의 연결:** 이 배경이 IFC 기반 자동 변환 프레임워크의 동기를 제공

---

### [I2] IFC as a Structural Data Bridge — and Its Limitations (¶2, ~200 words)

**목표:** IFC를 설계-해석 연결의 핵심 데이터 브릿지로 위치시키되, 목적 맞춤 파싱이 필요함을 보임

**문단 구조:**
1. IFC가 건물 기하, 공간 구조, 재료 물성, 부재 관계를 인코딩하는 개방형 표준이라는 정의 (1문장)
2. IfcColumn, IfcBeam, IfcSlab, IfcBuildingStorey 등이 해석 모델 구축에 필요한 기하·물성 데이터를 제공한다 (1문장)
3. **그러나** IFC 파일은 주로 건축 목적으로 작성되므로, 구조 등급 정보(그리드 좌표, 단면 프로파일, 경계 조건)를 추출하려면 목적 맞춤형 파싱 로직이 필요하다 (1–2문장)
4. 특히 IFC의 기하 표현(SweptSolid, BRep 등)이 해석용 wireframe과 다르며, 자동 매핑이 간단하지 않다 (1문장)

**핵심 주장:**
- IFC는 해석 모델에 필요한 원재료 정보를 포함하지만, 건축 모델링 목적의 표현 체계와 구조해석의 요구 사이에 의미론적 간극(semantic gap)이 존재한다.
- 이 간극을 메우려면 단순 포맷 변환이 아닌, 구조적 의미를 해석하는 파싱 로직이 필요하다.

**참고 문장 (Hasan, 2019 §2):**
> *"Recent BIM software cannot interpret specific geometric structural essentials and other structural semantics of models that are essential to represent the model realistically."*

**참고 (Leonardi, 2024 §2.3):**
> *"The passage from an architectural model to a structural model consists of only exporting the structural elements and neglecting the non-structural ones."*
> *"In the context of masonry aggregates, exporting a model using the 'Reference View' reveals that the complex geometries lead to the creation of a Tessellation."*

**문헌 근거:**
- **ISO 16739-1:2018** — IFC 스키마 공식 정의
- **Ramaji & Memari (2018)** — IFC-to-analytical model 매핑 및 해석
- **Leonardi et al. (2024)** — IFC 기하 표현 유형(SweptSolid, BRep, Tessellation)과 해석 호환성

**로컬 파일:**
- [mcp-server/core/ifc_parser.py](mcp-server/core/ifc_parser.py) — IfcColumn, IfcBeam, IfcSlab, IfcBuildingStorey, IfcIShapeProfileDef 파싱; 200mm 허용오차 그리드 클러스터링

---

### [I3] Prior Work on BIM-to-FEA Automation and Remaining Gaps (¶3, ~250 words)

**목표:** 기존 연구를 분류하고, 아직 해결되지 않은 구체적 갭을 식별

**문단 구조:**
1. 기존 BIM→구조해석 연구를 세 흐름으로 분류 (2문장):
   - (a) 직접 IFC→FEA 모델 변환 (Leonardi 2024, Crespi 2015)
   - (b) BIM 저작 도구 내 해석 통합 (Wang 2025 — Revit-Dynamo-OpenSees)
   - (c) 미들웨어/add-in 접근 (Hasan 2019 — Revit add-in for hybrid BEM-FEM)
2. Fernández-Mora의 리뷰 결과를 인용하여 연구 편중 지적 (1문장)
3. **구체적 gap 4개** 제시 (Leonardi 스타일 키워드 압축):

**핵심 주장 — 4개 갭:**

> **Gap 요약 prose (표 앞에 배치):**
> *"Existing BIM-to-analysis studies remain limited by insufficient integration of code-based load automation, limited support for regional design standards, dependence on proprietary or semi-automatic workflows, and a lack of systematic quantitative validation against established commercial software."*
> → 심사자가 표를 읽기 전에 gap 전체를 한 문장으로 파악할 수 있도록 배치. Leonardi의 "laborious, semi-automatic, proprietary" 압축 패턴과 동일한 방식.

| # | Gap | 설명 | 근거 |
|---|-----|------|------|
| G1 | **하중 자동화 부재** | 대부분 IFC→해석 워크플로우는 하중을 수동 입력하며, 기준 기반 하중 DB 연계가 통합되지 않았다 | Wang (2025): BTO에서도 하중은 수동 설정; Leonardi (2024): self-weight만 자동 |
| G2 | **지역 설계기준 연동 부족** | KDS 등 지역 기준에 따른 자동 하중 생성을 해석과 통합한 연구가 드물다 | 기존 연구 대부분 Chinese/European code; **to the authors' knowledge,** 한국 KDS 통합 사례 없음 |
| G3 | **재현성·개방성 한계** | 기존 워크플로우가 상용 SW에 종속적이거나 반자동적 | Leonardi (2024 Abstract): "laborious, semi-automatic, and based on proprietary software" |
| G4 | **메트릭 수준 정량 검증 부족** | 상용 소프트웨어 대비 다수 응답 물리량에 걸친 메트릭 수준 정량 비교를 제공하는 연구가 적다 | Wang (2025): conversion 전후 비교만, 다수 응답 메트릭에 대한 체계적 비교 없음 |

**참고 문장 (Leonardi, 2024 Abstract):**
> *"Thus far, BIM to finite element analysis procedures applied to historic constructions have remained laborious, semi-automatic, and based on proprietary software."*

**참고 문장 (Fernández-Mora, 2022 §5):**
> *"This set of studies shows a tendency towards design tools and new buildings."*
> *"80.25% of the analysed papers are destined to be used during the design process of the structure."*

**참고 문장 (Wang, 2025 §2.1):**
> *"However, there remains a gap in fully integrating BIM with seismic analysis, particularly for RC structures. OpenSees... effective methods to convert BIM models into OpenSees-compatible formats are still lacking."*

**참고 문장 (Hasan, 2019 Introduction):**
> *"Despite the previous attempts, current BIM enrichments contain insufficient semantic definition to allow for hybrid boundary element-finite element numerical modeling."*

**문헌 근거:**
- **Fernández-Mora et al. (2022)** — 81개 논문 분석, design tools/new buildings 편중 확인
- **Leonardi et al. (2024)** — "laborious, semi-automatic, proprietary-dependent" 3-키워드 갭
- **Wang et al. (2025)** — BIM-OpenSees 연계 최신 사례, 남은 갭
- **Hasan et al. (2019)** — AMV 의미론적 한계

**본 연구와의 연결:** 본 논문은 위 네 가지 갭을 모두 다루는 통합 프레임워크를 제안

---

### [I4] The Choice of OpenSees and Its Preprocessing Challenge (¶4, ~150 words)

**목표:** OpenSees를 해석 엔진으로 선택한 이유와 그에 따른 전처리 과제를 동시에 제시

**문단 구조:**
1. OpenSees(Py)가 비선형 해석 능력으로 구조/지진공학에서 널리 사용된다 (1문장)
2. 그러나 공식 GUI/전처리기 부재로 모델링이 번거롭다 (1문장)
3. 따라서 BIM에서 추출한 정보를 자동으로 OpenSees 입력으로 변환하는 파이프라인의 가치가 있다 (1문장)

**핵심 주장:**
- OpenSees는 강력한 비선형 해석 능력을 가진 개방형 엔진이지만, 전처리 인터페이스의 부재가 실무 활용의 병목이다.
- 이 병목은 BIM/IFC 기반 자동 전처리로 해소할 수 있다.

**참고 문장 (Wang, 2025 Introduction):**
> *"OpenSees, known for its robust nonlinear analysis capabilities and user-friendly open-source ecosystem, is widely employed in the fields of structural and earthquake engineering... Nonetheless, the preprocessing of OpenSees models, including geometry, material parameters, and loads, remains cumbersome."*

**참고 (Leonardi, 2024 §2.2):**
> *"Since OpenSees does not have an official open graphical interface, the results are displayed using the open FEM meshing software Gmsh."*

**문헌 근거:**
- **McKenna et al. (2000/2011)** / **Zhu et al. (2018)** — OpenSees/OpenSeesPy
- **Wang et al. (2025)** — OpenSees 전처리 부담 지적

**로컬 파일:**
- [mcp-server/core/ops_compat.py](mcp-server/core/ops_compat.py) — OpenSeesPy 3.8/3.12+ 호환 레이어

---

### [I5] AI/LLM in Structural Design — Brief Context (¶5, ~80 words)

**목표:** AI/LLM 맥락을 최소한으로 언급하되, 핵심 논점에서 벗어나지 않도록 짧게 제한

**문단 구조:**
1. AI/ML 기반 구조설계 자동화가 부상하고 있으며, LLM 기반 입력 처리도 탐구되고 있다 (1문장)
2. 그러나 이러한 접근의 재현성과 일반화는 아직 체계적으로 검증되지 않았다 (1문장)
3. 본 연구에서는 자연어 입력을 보조적 채널로만 다루며, 핵심 검증 대상은 IFC-KDS-OpenSees 파이프라인이다 (1문장)

> **v3→v3.1 변경:** 120 → ~80 words로 축소. 이 문단이 길어지면 논문 초점이 LLM으로 이동하는 것처럼 보임. AI 맥락은 2–3문장이면 충분.

**문헌 근거:**
- **Xie et al. (2025)** — AI in structural design automation
- **Liao et al. (2024)** — LLM possibilities and limitations

**톤 주의:** LLM을 "core validated contribution"이 아닌 "exploratory extension" / "supplementary input channel"로 표현

---

### [I6] Scope of This Study (¶6, ~180 words)

**목표:** 본 연구의 정확한 범위를 한정 — 할 수 있는 것과 이번 논문에서 다루는 것을 구분하며, **왜 이 범위를 택했는지**를 명시

**문단 구조:**
1. 본 연구의 대상: 정형(직교 그리드) 다층 철골조 건물 (1문장)
2. **범위 선택 이유 3가지** (1–2문장):
   - **(i)** 직교 그리드 기하는 IFC-to-grid 매핑을 모호성 없이 결정론적으로 수행할 수 있어, 파이프라인 자체의 실현가능성 검증에 적합하다.
   - **(ii)** 탄성 철골 재료 가정은 재료 모델 복잡성을 배제하여, 자동화 파이프라인 검증을 재료 거동 검증과 분리한다.
   - **(iii)** 정형 철골 프레임은 Midas Gen에서 동일 조건 벤치마크가 용이하여, 정량적 비교 기반을 제공한다.
3. 포함 범위: IFC 기하 추출 → KDS 기반 DL/LL/EQ/Wind → 3D FEA → 초보적 설계 검증 (1문장)
4. 제외 범위: 비정형 기하, RC/합성 부재, 비선형 재료, 동적 시간이력 (1문장)
5. 논문 구성 안내 (1문장)

> **v3→v3.1 변경:** scope 선택 이유를 3가지 구체적으로 제시. 심사자가 "왜 RC 안 하나, 왜 비정형 안 되나"를 물을 때 방어할 수 있도록. 단순히 "before extending to more heterogeneous systems"만 쓰면 "못 해서 안 한 거"로 읽힐 위험이 있음.

**참고 (Leonardi, 2024 §1 마지막):**
> *"The model holds the potential for both static and dynamic nonlinear analyses... However, nonlinear analysis would add extra complexity that would deviate from the main purpose of this paper, which is to present the developed algorithm in detail."*

**참고 (Hasan, 2019 §1 마지막):**
> *"In the subsequent section, the deficiencies... are presented... The last section presents various tests and practical examples to illustrate the usefulness."*

**로컬 파일:**
- [mcp-server/core/building_model.py](mcp-server/core/building_model.py) — `bays_x`, `bays_y` → regular grid 가정
- [mcp-server/core/frame_3d.py](mcp-server/core/frame_3d.py) — `elasticBeamColumn` (선형 재료)

---

### [I7] Research Questions and Contributions (¶7, ~180 words)

**목표:** 명시적 연구 질문 + 번호 기여점으로 Introduction 마무리

**문단 구조:**
1. RQ1–RQ3 제시 (3문장)
2. 5개 기여점을 위계순으로 나열 (번호 목록)

**연구 질문:**
- **RQ1**: IFC 기반 건물 정보를 기준 적합 하중과 함께 자동으로 구조해석 모델로 변환할 수 있는가?
- **RQ2**: 제안된 자동화 워크플로우가 확립된 상용 소프트웨어(Midas Gen)와 일관된 결과를 산출하는가?
- **RQ3** *(탐색적)*: LLM 기반 자연어 인터페이스가 구조해석 입력의 보조 수단으로 기능할 수 있는가?

**기여점** (위계순, Leonardi conclusion 스타일 — 번호 나열):
1. **IFC→해석모델 자동 변환**: 정형 철골 프레임의 기하·단면·재료 자동 추출
2. **KDS 기반 하중 DB 연계**: 고정/활/지진/풍 하중 + 18개 조합의 자동 생성
3. **OpenSeesPy 기반 3D FEA 통합**: 6-DOF, Corotational 비선형, rigid diaphragm
4. **벤치마크 검증**: Midas Gen 대비 5개 케이스, 112개 메트릭
5. **LLM 보조 인터페이스** *(확장)*: 한국어 자연어 → Config 변환

**참고 (Leonardi, 2024 Conclusion):**
> *"This paper provided three primary results: (1) the Level of Information Need for structural analysis; (2) the automated implementation of solid finite element models from IFC models; (3) the results of static and dynamic analysis of an entire aggregate."*

---

# 2. PROPOSED METHODOLOGY (~1700 words)

> **전체 참고 전략:**
> - **§2.1**: Wang (2025) 스타일 — 3-stage overview + Figure (workflow diagram)
> - **§2.2–§2.3**: Leonardi (2024) §2.3 스타일 — modelling requirements를 Methods 전에 배치
> - **§2.4–§2.5**: Leonardi (2024) §2.4 스타일 — 알고리즘을 입력–처리–출력 구조로 설명
> - **scope control**: Leonardi (2024) — 각 섹션에서 "할 수 있는 것"과 "이번 논문에서 한 것"을 구분

---

## §2.1 System Architecture Overview

**역할:** 전체 시스템 파이프라인을 Figure와 함께 제시

**문단 구조 (Wang 2025 §4 참고):**
1. 전체 프레임워크를 3단계(stage)로 요약하는 선언문 (1문장)
2. 각 stage를 한 줄씩 설명 (3–4문장)
3. Figure 1 참조 안내 (1문장)

**3-Stage Overview:**

| Stage | 내용 | 핵심 모듈 |
|-------|------|-----------|
| **Stage 1: Input Processing** | IFC 파싱 / NL 파싱(보조) / Manual 입력 → BuildingModel IR | `ifc_parser.py`, `nl_resolver.py` |
| **Stage 2: Load & Analysis** | KDS DB → 4종 하중 + 18조합 → OpenSeesPy 3D FEA | `load_generator.py`, `frame_3d.py` |
| **Stage 3: Post-Processing** | 설계검토(KDS drift + AISC member) + 시각화 + 리포트 | `design_check.py`, `visualization_3d.py` |

**참고 (Wang, 2025 §4):**
> *"The comprehensive framework is divided into three distinct stages: (1) conversion of BIM models to OpenSees; (2) seismic reliability assessment; (3) visualization of seismic reliability analysis results using Dynamo."*

**핵심 주장:**
- 모든 입력 경로(IFC, NL, Manual)가 공통 중간 표현(BuildingModel IR)으로 수렴한다.
- 이 구조는 입력 채널 간 공유 중간 표현(shared intermediate representation)을 제공한다.

**Figure 1**: System architecture block diagram — 3 stages + data flow arrows

**근거 파일:**
- [mcp-server/server.py](mcp-server/server.py) — 14 MCP tools, 전체 조율
- [mcp-server/core/building_model.py](mcp-server/core/building_model.py) — IR 정의

---

## §2.2 Modelling Requirements and Input Specification

**역할:** 알고리즘 설명 전에, 시스템이 요구하는 입력 사양을 먼저 정의

> **이 절의 위치 근거 (Leonardi, 2024 §2.3):**
> *"The modeling requirements were established in accordance with the 'Level Of Information Need' schema outlined in EN17412-1."*
> → Leonardi는 algorithm (§2.4) 전에 modelling requirements (§2.3)를 별도 절로 두어, 독자가 "이 시스템에 무엇을 넣어야 하는가"를 먼저 파악하도록 했다.

**문단 구조:**
1. 입력 데이터의 세 범주 정의 (기하, 물성, 하중 관련) (1문장)
2. IFC 입력 시 필요한 엔티티 목록과 속성 (Table 1 참조) (2문장)
3. KDS DB 입력 시 필요한 파라미터: 용도(occupancy), 지역(region), 중요도(importance) (1문장)
4. 시스템이 처리하는 구조 유형의 범위와 제한 (1문장)

**핵심 주장:**
- IFC 모델에서 추출해야 하는 정보: IfcBuildingStorey(층 표고), IfcColumn(위치, 단면), IfcBeam(단면), IfcSlab(두께)
- 해석 모델 생성에 추가로 필요한 비기하학적 정보: 용도별 활하중, 지역별 지진/풍 위험도, 중요도 계수
- 현재 범위: 정형 직교 그리드 철골 프레임 (irregular geometry, RC, 합성 부재 미지원)

**Table 1**: IFC entity → extracted structural parameter mapping

| IFC Entity | Extracted Parameter | Usage |
|------------|-------------------|-------|
| `IfcBuildingStorey` | 층 표고, 층고 | 층 정의 |
| `IfcColumn` | X/Y 위치, 단면 프로파일 | 그리드 감지, 기둥 모델링 |
| `IfcBeam` | 단면 프로파일 | 보 모델링 |
| `IfcSlab` | 두께 | 슬래브 자중, 하중 면적 |
| `IfcIShapeProfileDef` | h, b, tw, tf | H형강 단면 물성 |

**참고 (Leonardi, 2024 §2.3):**
> *"In the definition of the modelling requirements, BIM is leveraged as a collaborative platform for integrating diverse disciplines..."*
> *"Each IFC object belongs to a specific 'IfcBuildingElementType'."*

**근거 파일:**
- [mcp-server/core/ifc_parser.py](mcp-server/core/ifc_parser.py) — 21개 함수, 파싱 대상 엔티티
- [data/mapping/occupancy.json](data/mapping/occupancy.json) — 30개 용도 매핑

---

## §2.3 Stage 1: Input Processing

### §2.3.1 IFC Parsing and Structural Data Extraction [검증 완료]

**역할:** IFC → BuildingModel IR 변환 상세

**문단 구조 (Leonardi §2.4 "developed algorithm" 스타일 — 입력→처리→출력):**
1. **입력**: IFC 파일 (IfcOpenShell 라이브러리로 파싱)
2. **처리**:
   - 층 추출: IfcBuildingStorey 반복 → 표고 정렬 → 층고 계산, 지상층 필터링, 옥상층 감지
   - 그리드 감지: 기둥 X/Y 좌표 수집 → 200mm 허용오차 클러스터링 → 정형 그리드 라인
   - 단면 추출: IfcIShapeProfileDef → H형강 (h, b, tw, tf) → KS D 3502 표준 단면 매칭
   - 재료 정규화: Revit 명명 규칙 → 한국 표준 강종 (SS275, SS400 등)
3. **출력**: BuildingModel IR (stories, bays_x, bays_y, sections, materials)

**핵심 주장:**
- IFC 파일의 건축 기하학적 표현에서 구조해석에 필요한 그리드 라인, 층 정보, 단면 물성을 자동 추출하는 파싱 파이프라인을 구현하였다.
- 그리드 감지 알고리즘은 기둥 위치의 좌표 클러스터링에 기반하며, 200mm 허용오차로 시공 오차를 수용한다.

**참고 (Leonardi, 2024 §2.4):**
> *"The first algorithm can be divided into three parts: 1) Input; 2) Material collection; 3) Material data collection."*
> → 알고리즘을 입력/처리/출력으로 나누어 설명하는 패턴

**참고 (Wang, 2025 §2.2):**
> *"In Revit, structural elements of RC structures are usually categorized into classes such as beams, columns, walls, and slabs. Dynamo facilitates programmatic access to these elements and their associated properties."*

**검증 상태:** IFC 파싱은 벽 기반(3층) + 기둥 기반(10층) 건물로 검증 완료

**근거 파일:**
- [mcp-server/core/ifc_parser.py](mcp-server/core/ifc_parser.py) — 813줄, 21개 함수

---

### §2.3.2 Natural Language Input Resolution (Supplementary Interface) [탐색적 구현]

**역할:** LLM 기반 보조 입력 채널 설명

> **논문 내 위치:** Hasan (2019)이 add-in을 "intermediate solution"으로 제시했듯, 본 절도 NL 인터페이스를 "supplementary channel"로 명시적 경계를 둔다.

**문단 구조:**
1. 파이프라인 설명: 사용자 한국어 텍스트 → Claude API → BuildingIntent JSON → `resolve_building_config()` (1문장)
2. 용도 매핑 전략 (5단계 fallback): exact alias → composite → normalized → substring → unresolved (1문장)
3. 지역 매핑: 229개 시군구 fuzzy matching (1문장)
4. 검증 상태와 한계: 38개 단위 테스트, LLM 재현성 한계 (1문장)

**참고 (Hasan, 2019 Abstract):**
> *"To demonstrate the need and provide an intermediate solution, an add-in software is developed..."*
> → "intermediate solution" framing을 차용

**검증 상태:** 38개 단위 테스트 통과. LLM 응답 재현성 보장 불가 → limitation 명시.

**근거 파일:**
- [mcp-server/core/nl_resolver.py](mcp-server/core/nl_resolver.py) — 675줄
- [data/mapping/occupancy.json](data/mapping/occupancy.json) — 30개 유형, 60+ 별칭
- [tests/test_nl_resolver.py](tests/test_nl_resolver.py) — 38개 테스트

---

## §2.4 Stage 2: Load Generation and Structural Analysis

### §2.4.1 Code-Based Load Generation [검증 완료]

**역할:** KDS 기준이 자동 하중 산정을 구동하는 방식 상세

**문단 구조 (입력→처리→출력):**
1. **입력**: BuildingModel IR의 기하 + 용도/지역/중요도 파라미터
2. **처리**: 4종 하중 생성
   - **DL (고정하중)** — KDS 41 12 00 §2: 슬래브 자중(γ×t), 마감, 설비/칸막이
   - **LL (활하중)** — KDS 41 12 00 §3: 용도→DB 조회 (712건 load_params, Supabase)
   - **EQ (지진하중)** — KDS 41 17 00: Sa(T) 설계응답스펙트럼, 등가정적 V=Cs×W, 지반계수 Fa/Fv, 내진시스템 R
   - **Wind (풍하중)** — KDS 41 12 00 §5: qz=0.613×Kz×Kd×V₀², F=qz×Gf×Cp×수압면적
3. **하중조합** — KDS 41 17 00 §1.7: 18개 자동 생성 (ULS + SLS)
4. **출력**: LoadCase 객체 리스트 → OpenSeesPy 입력

**핵심 주장:**
- 기존 BIM-to-FEA 연구에서 대부분 수동으로 남겨두었던 하중 생성을 설계기준 DB에서 직접 생성하여, 입력 일관성과 재현성을 확보한다.
- 지역별 위험도(229개 시군구, 2,290건)를 DB에서 조회하여 지진/풍 하중에 반영한다.

**Table 2**: 하중 유형별 KDS 기준 참조 매핑

| Load Type | KDS Reference | DB Records | Key Parameters |
|-----------|--------------|------------|----------------|
| DL | KDS 41 12 00 §2 | — | γ_concrete, t_slab, finishes |
| LL | KDS 41 12 00 §3 | 712 | occupancy → live load (kN/m²) |
| EQ | KDS 41 17 00 | 2,290 | SDS, SD1, Fa, Fv, R, IE |
| Wind | KDS 41 12 00 §5 | 49 | V₀, Kz, Gf, Cp |
| Combinations | KDS 41 17 00 §1.7 | — | 18 combinations (ULS+SLS) |

**근거 파일:**
- [mcp-server/core/load_generator.py](mcp-server/core/load_generator.py) — 738줄
- [mcp-server/core/design_spectrum.py](mcp-server/core/design_spectrum.py) — Sa(T) 곡선, 320줄
- [mcp-server/core/kds_loads.py](mcp-server/core/kds_loads.py) — KDS 파라미터 조회, 213줄

---

### §2.4.2 3D Finite Element Analysis [검증 완료 — 벤치마크]

**역할:** OpenSeesPy 기반 해석 엔진 설명

**문단 구조:**
1. 요소 정식화: elasticBeamColumn (Euler-Bernoulli), 절점당 6-DOF (1문장)
2. 단면 물성: A, Ix, Iy, J — KS D 3502/3568 DB (738개 단면) (1문장)
3. 기하비선형: Linear / Corotational geomTransf + Newton solver (1문장)
4. 부재 릴리즈: 절점 분할 + equalDOF 구속 (1문장)
5. Rigid Diaphragm: 층별 rigidDiaphragm + 기여면적 기반 질량 분배 (1문장)
6. 다중 하중케이스/조합 → envelope 추출 (1문장)

**핵심 주장:**
- OpenSeesPy의 elasticBeamColumn 요소와 Corotational geomTransf를 사용하여 대변위 기하비선형 효과를 포함하는 3D 프레임 해석을 수행한다.
- 실무 조건(rigid diaphragm, 부재 릴리즈, 다중 하중 조합)을 지원한다.

**참고 (Wang, 2025 §2.3.3):**
> *"For beam and column elements, based on their vertices array, the algorithm calculates their dimensions along the global coordinate axes X, Y, and Z, cross-sectional area (A), inertia moments along the local coordinate axes Y and Z (Iy, Iz), and volume (v)."*

**Table 3**: 솔버 구성 파라미터

| Parameter | Value | Note |
|-----------|-------|------|
| Element | `elasticBeamColumn` | Euler-Bernoulli |
| DOF/node | 6 | Ux, Uy, Uz, Rx, Ry, Rz |
| geomTransf | Linear / Corotational | 선형 / 대변위 비선형 |
| Solver | Newton (10 steps) | Fallback: ModifiedNewton (50) |
| Diaphragm | `rigidDiaphragm(3, master, *slaves)` | 층별 |

**근거 파일:**
- [mcp-server/core/frame_3d.py](mcp-server/core/frame_3d.py) — 1605줄
- [mcp-server/core/section_3d.py](mcp-server/core/section_3d.py) — 317줄

---

## §2.5 Stage 3: Post-Processing

### §2.5.1 Design Checking (Preliminary Integration) [구현 완료, 독립 벤치마크 미실시]

**역할:** 해석 후 설계 검증 설명

> **톤 (Hasan "intermediate solution" 스타일):**
> 이 기능은 구현되어 워크플로우에 통합되었으나, 설계검토 결과 자체의 독립적 검증은 본 논문의 범위에 포함하지 않는다.
> 따라서 "preliminary support" / "integrated design checking capability"로 표현.

**문단 구조:**
1. 층간변위 검토: KDS 41 17 00 §8.2.3, Δ_inelastic = Cd×δ_elastic/IE, 허용치 (1문장)
2. 부재강도 검토: AISC 360 기준에 따라 압축, 휨, 전단, 상관 검토를 수행한다고 **참조만 명시** (1문장)
3. 가정과 제한: K=1.0, compact 가정, φ=0.9 (1문장)
4. 향후 독립 검증 필요성 명시 (1문장)

> **v3→v3.1 변경:** 본문에서는 drift check 공식 1개와 AISC 절 참조만 유지. 세부 공식 전개(Fcr, φMn, φVn, H1-1a/b)는 Appendix 또는 Supplementary Material로 이동. 본문 비중이 과도하면 핵심이 "analysis automation"이 아니라 "design checking system"으로 보일 수 있음.

**Table 4**: 설계검토 기준 요약 (본문용, 간략)
*Appendix A*: AISC 360 부재강도 검토 공식 상세 (§E3, §F2, §G2, §H1.1)

**근거 파일:**
- [mcp-server/core/design_check.py](mcp-server/core/design_check.py) — 610줄
- [tests/test_design_check.py](tests/test_design_check.py) — 16개 단위 테스트

---

### §2.5.2 Visualization and Analysis Review Support [구현 완료]

**역할:** 출력 계층의 간략 설명 + **자동 생성 모델의 검토를 지원하는 연구적 역할** 명시

**참고 (Wang, 2025 §4.3):**
> *"By developing custom Dynamo scripts, different colors are employed to represent the reliability indices of each story, which are directly mapped onto the BIM model."*
> → 결과 시각화를 별도 stage로 분리하는 패턴

**문단 구조:**
1. **연구적 역할 선언** (1문장): 시각화 계층은 자동 생성된 해석 모델과 결과를 엔지니어가 검토·확인할 수 있도록 하는 검증 보조 도구(verification aid)로 기능한다.
   > *"The visualization layer serves as a verification aid, enabling engineers to review the automatically generated model, inspect load distributions, and confirm that analysis assumptions are consistent with design intent before accepting results."*
2. 독립 HTML 리포트: 하중 요약, 변형 형상, 부재력, 층별 drift 검토를 포함하는 인터랙티브 차트 (1문장)
3. 웹 기반 3D 에디터: Three.js, 설계검토 색상 오버레이 (OK/NG 즉시 확인) (1문장)

> **v3→v3.1 변경:** 단순 "3D editor가 있다"에서 "자동 생성 모델의 검토를 지원한다 / post-processing transparency를 높인다"로 연구적 역할 명시. 실제 코드상 assumption_tracker, equilibrium_residual, critical_member 기능이 이 역할을 수행.

**근거 파일:**
- [mcp-server/core/visualization_3d.py](mcp-server/core/visualization_3d.py) — 2597줄 (assumption confirmation 탭, story drift 요약, critical member 하이라이팅, equilibrium residual)
- [webapp/backend/static/js/editor3d.js](webapp/backend/static/js/editor3d.js) — ~1650줄 (DC 색상 오버레이, 부재 선택, 결과 패널)

---

### §2 Validation Handoff (§2.5.2 끝 또는 §2 마지막 독립 문단)

> "The validation of this workflow through benchmark comparison with Midas Gen is presented in Section 3."

---

# 3. BENCHMARK VALIDATION (~1500 words)

> **참고 전략:**
> - Wang (2025) §5 — conversion 전후 비교를 결과의 일부로 제시
> - Leonardi (2024) §3.3 — 결과를 숫자 나열이 아니라 구조적 해석과 함께 제시
> - Hasan (2019) §Verification — "proof of need" 차원에서 검증

---

## §3.1 Benchmark Case Design [검증 완료]

**문단 구조:**
1. 벤치마크 목적 선언: 제안 프레임워크의 해석 결과가 확립된 상용 소프트웨어와 일관되는지 검증 (1문장)
2. 5개 케이스 설계 근거: 단순→복잡 순서, 2D→3D, 선형→비선형 (2문장)
3. 비교 소프트웨어: Midas Gen (동일 모델, 동일 하중, 동일 경계조건) (1문장)

**Table 5**: 벤치마크 케이스 사양

| Case | Description | Nodes | Elements | Key Feature |
|------|-------------|-------|----------|-------------|
| 1 | 단순보 (집중하중) | 3 | 2 | 기본 평형 |
| 2 | 2D 문형 라멘 (고정지지) | 4 | 3 | 횡+중력 |
| 3 | 3층 2D 프레임 (단계적 횡력) | 8 | 9 | 다층 상호작용 |
| 4 | 3D 2층 프레임 (1×1 bay) | 12 | 12 | 3D 선형 |
| 5† | 3D 5층 프레임 (P-Delta) | 24 | 40 | 비선형 기하 (supplementary) |

> † Case 5 = supplementary advanced configuration check. §2.4.2에서 Corotational은 "outside the primary validation scope"로 명시되어 있으므로, 보충 검증으로 위치.

**참고 (Leonardi, 2024 §3):**
> *"Special attention was given to the application in a meaningful study case, to prove the potential of the proposed methodology."*

**근거 파일:**
- [tests/benchmark/cases.py](tests/benchmark/cases.py) — 1069줄
- [tests/benchmark/midas_results/](tests/benchmark/midas_results/) — Midas Gen 참조값

---

## §3.2 Comparison Metrics and Criteria

**문단 구조:**
1. 비교 대상: 절점 변위, 반력, 요소 단부력 (N, V, M), 층간변위 (1문장)
2. 판정 기준: OK (<1%), CHECK (1–5%), FAIL (>5%) (1문장)
3. 부호규약 정렬: OpenSees 로컬 → 교과서 규약 변환 후 비교 (1문장)

**근거 파일:**
- [tests/benchmark/compare.py](tests/benchmark/compare.py)
- [mcp-server/core/sign_convention.py](mcp-server/core/sign_convention.py)

---

## §3.3 Results

**문단 구조 (Leonardi §3.3 결과 해석 스타일):**
1. 전체 요약: 112개 메트릭 — 100 OK, 12 CHECK, 0 FAIL (1문장)
2. Cases 1–3 (2D): 전 메트릭 <0.1% → 기본 FEM 정식화 일치 확인 (1문장)
3. Case 4 (3D): 12 CHECK → 원인 분석(3D 요소 강성 정식화 차이) (2문장)
4. Case 5† (3D P-Delta, supplementary): 40/40 OK, 최대 0.05% → Corotational vs P-Delta 일치, Case 4와 대비하여 차이가 요소 정식화에 기인함을 시사 (1문장)
5. 결과의 구조적 의미 해석 (1문장)

**핵심 주장:**
- 2D 해석(Cases 1–3)에서는 두 소프트웨어 간 사실상 완전 일치가 확인되며, 이는 기본 FEM 정식화가 동일함을 보여준다.
- 3D에서의 소규모 차이(~3%)는 elasticBeamColumn 요소의 전단-비틀림 결합 처리 방식 차이에 기인하며, 공학적 허용 범위 내이다.

**참고 (Leonardi, 2024 §3.3):**
> *"Stresses distributions... were computed. Higher concentrations of compression stresses are observed in the horizontal wall portions between the openings."*
> → 결과를 단순 수치가 아닌 구조적 의미와 함께 설명

**참고 (Wang, 2025 §5.1):**
> *"After converting the BIM model, nonlinear dynamic time-history analyses are performed on the structure using the OpenSees platform."*
> → conversion 성공 자체도 결과의 일부

**Table 6**: 케이스별 결과 요약

| Case | Total Metrics | OK (<1%) | CHECK (1–5%) | Max Diff | Critical Metric |
|------|--------------|----------|--------------|----------|-----------------|
| 1 | 5 | 5 | 0 | <0.01% | — |
| 2 | 15 | 15 | 0 | <0.1% | — |
| 3 | 17 | 17 | 0 | <0.1% | — |
| 4 | 35 | 23 | 12 | 3.90% | 3D element stiffness formulation |
| 5† | 40 | 40 | 0 | 0.05% | — |
| **Total** | **112** | **100** | **12** | — | — |

> † Case 5 = supplementary advanced configuration check

**Figure 5**: OpenSees vs. Midas 변위 비교 산점도
**Figure 6**: 케이스별 OK/CHECK 분포 바 차트

**근거 파일:**
- [tests/benchmark/opensees_results/](tests/benchmark/opensees_results/)
- [docs/benchmark_charts/](docs/benchmark_charts/)

---

## §3.4 Discussion of Discrepancies

**문단 구조:**
1. Case 4의 ~3% 차이 원인: elasticBeamColumn vs Midas의 3D 요소 강성 항 처리 차이 (1문장)
2. Corotational vs P-Delta: 낮은 drift (<0.25%)에서 ~0.03% 차이 → 실용적 동등 (1문장)
3. 이 차이들이 공학적 허용 범위 내이며 기존 FEA 비교 연구와 일관됨 (1문장)

**톤:** ~~"commercial-grade accuracy"~~ → "The results show close agreement with the reference software across the tested configurations."

---

# 4. APPLICATION EXAMPLE (~800 words)

> **참고:** Wang (2025) §5 — case description을 매우 구체적으로 (층고, 하중, 재료, 단면, 해석 설정)

---

## §4.1 Example Building Description

**문단 구조 (Wang §5.1 스타일):**
1. 건물 개요: 5층, 3×2 bay, 서울 강남 (1문장)
2. 입력 경로: IFC 업로드 또는 NL 텍스트 "서울 강남, 1층 근생, 2~5층 오피스" (1문장)
3. 구조 사양: H-300×300 기둥, H-400×200 보, SS275 강재 (1문장)
4. 하중 설정: DL + LL(용도별) + EQ X/Y + Wind X/Y, 18개 조합 (1문장)
5. 해석 설정: 3D 6-DOF, rigid diaphragm, 260개 요소 (1문장)

**참고 (Wang, 2025 §5.1):**
> *"This section examines a six-story RC frame structure designed according to current Chinese standards. The structural dimensions and layout are shown in Fig. 11(a)-(b), with each story measuring 3.3 m in height."*
> → 입력 조건의 구체성

**참고 (Wang, 2025 §5.1):**
> *"The structure is subjected to a uniform floor load of 4.25 kN/m² and a live load of 2.0 kN/m²..."*
> → 하중값 명시

## §4.2 Conversion and Analysis Results

**문단 구조:**
1. IFC 파싱 → BuildingModel IR 변환 성공 확인 (conversion 성공 = 결과의 일부) (1문장)
2. KDS DB 조회 → 하중 자동 생성 성공 (1문장)
3. 해석 결과: 최대 drift X/Y, 최대 변위, 설계검토 OK/NG (2문장)
4. NL 입력 경로를 통한 동일 결과 재현 (1문장)

**참고 (Wang, 2025 §5.1, Fig. 14):**
> *"Fig. 14. Comparison of models before and after conversion for Case 1."*
> → 변환 전후 비교가 결과의 일부

**Figure 7**: 3D 에디터 해석 결과 스크린샷 (변환 전 IFC vs 해석 결과)
**Figure 8**: 설계검토 색상 오버레이 (OK=녹색, NG=적색)

---

# 5. DISCUSSION (~800 words)

---

## §5.1 Demonstrated Capabilities

**문단 구조:**
1. IFC→해석 모델 자동 변환이 검증 범위에서 동작함을 확인 (1문장)
2. KDS 하중 자동화가 4종 하중에 대해 작동 (1문장)
3. 벤치마크에서 112개 메트릭 중 100개가 <1% 일치 (1문장)

**톤:** "Good correspondence with the reference software was observed within the tested configurations."

---

## §5.2 Current Limitations

**문단 구조 (Hasan 2019의 deficiency 목록 스타일 — 구체 항목 나열):**

| # | Limitation | Impact |
|---|-----------|--------|
| L1 | 정형 직교 그리드만 지원 | 비정형, 곡선 부재, 세트백 미지원 |
| L2 | 선형 탄성 재료만 | RC, 합성, 비선형 재료 미지원 |
| L3 | IFC 파싱은 벽/기둥 기반 프레임만 가정 | 혼합 시스템 미지원 |
| L4 | K=1.0, compact 가정 | LTB, 국부좌굴, 스웨이 증폭 미적용 |
| L5 | LLM 입력은 Claude API 종속 | 재현성·비용 한계 |
| L6 | 설계검토 독립 벤치마크 미실시 | 별도 검증 필요 |

**참고 (Hasan, 2019 §2):**
> *"The structural components mentioned here are currently deficient in all popular BIM software."*
> → 한계를 구체적 항목으로 나열하면 독자가 정확히 어디까지 신뢰할 수 있는지 판단 가능

**참고 (Wang, 2025 Conclusion):**
> *"(1) The framework is currently limited to the conversion of BIM models for RC structures, lacking adaptation for other structural types... (2) The conversion process is restricted to BIM models developed on the Revit platform..."*
> → 한계를 번호로 나열하는 패턴

---

## §5.3 Comparison with Existing Approaches

**문단 구조:**
1. vs 상용 BIM→해석 플러그인 (Revit→ETABS): vendor-neutral (IFC), code-transparent (KDS DB), open-source (OpenSeesPy) (1문장)
2. vs 기존 IFC→FEA 연구 (Leonardi, Wang): 하중 자동화 + 설계검토 통합이 대부분 없음 (1문장)
3. vs 수동 워크플로우: 정형 건물에서 입력 노력 감소 (1문장)

**톤:** ~~"Reduces from ~30 min to ~2 min"~~ → "The workflow is expected to reduce manual input effort substantially for regular buildings, though a formal time comparison was not conducted."

---

# 6. CONCLUSION (~500 words)

> **참고:** Leonardi (2024) Conclusion 스타일 — 번호 기여점 + 의미 해석 + 구체적 future work

---

### [C1] Summary (1문단)

**문단 구조:**
1. 본 연구가 무엇을 제시했는지 한 문장 요약 (1문장)
2. IFC, KDS DB, OpenSeesPy 3개 키워드로 파이프라인 재압축 (1문장)
3. LLM 보조 인터페이스 언급 (1문장)

---

### [C2] Key Findings (번호 목록, 위계순)

**참고 (Leonardi, 2024 Conclusion):**
> *"This paper provided three primary results: (1)...; (2)...; (3)..."*

1. **IFC→해석 모델 자동 변환의 실현가능성 확인**: 정형 철골 프레임에서 기하, 단면, 재료를 신뢰성 있게 추출·변환 가능
2. **KDS 기준 적합 하중이 워크플로우에 내장 가능**: 712건 하중 파라미터 + 2,290건 지역 위험도에서 4종 하중과 18개 조합 자동 생성
3. **벤치마크에서 양호한 일치 확인**: 112개 메트릭 중 100개가 Midas Gen 대비 1% 이내, 12개도 3% 이내
4. **설계검토 지원이 해석 워크플로우에 초보적으로 통합**: KDS 층간변위 + AISC 360 부재강도
5. **LLM 보조 인터페이스의 가능성 확인** *(탐색적)*: 30개 용도, 229개 지역

---

### [C3] Significance

**참고 (Leonardi, 2024 Conclusion):**
> *"The definition of the Level of Information Need is a contribution in the field of standardisation of information management... The choice of continuum finite element analysis as a modelling strategy shortened the distance between BIM model and structural model."*

**핵심 주장:**
- 기하 추출, 하중 생성, 해석 실행을 하나의 개방형 파이프라인으로 통합함으로써, BIM 모델과 구조해석 모델 사이의 거리를 줄이는 한 단계(a step toward)를 제시하였다.
- KDS 하중 DB의 구조화는 한국 기준 기반 자동화의 기반(basis for)이 될 수 있다.

**톤:** "a new step" / "a step toward" / "basis for" — 과장 회피

---

### [C4] Future Research Directions

**문단 구조 (Leonardi 스타일 — 구체적 항목):**
1. **기하 확장**: 비정형 평면, 세트백, 비직교 그리드
2. **재료 비선형**: 파이버 단면 요소 (RC, 합성 부재)
3. **동적 해석**: 응답스펙트럼, 시간이력 해석
4. **기준 DB 확장**: 추가 KDS 기준 (기초, 접합부), 국제 기준 (Eurocode, ASCE 7)
5. **설계검토 검증 강화**: 독립 벤치마크 (수동 계산 또는 상용 SW 대비)
6. **AI 기반 설계 피드백 강화**: RAG 기반 기준 조문 조회, 자동 모델 검토

**참고 (Wang, 2025 Conclusion):**
> *"Future investigations should prioritize systematic quantification and integration of these epistemic uncertainties."*

---

# 8. TONE CORRECTION TABLE

| # | 원래 표현 (과장) | 수정 표현 (학술적) | 이유 |
|---|-----------------|-------------------|------|
| 1 | "commercial-grade accuracy" | "close agreement with reference software" | 특정 범위의 벤치마크 |
| 2 | "end-to-end automated" | "framework-level automation within the current scope" | 모든 건물 유형 아님 |
| 3 | "minimal manual intervention" | "reducing manual modeling effort for regular frame structures" | 범위 한정 |
| 4 | "automated design checking" | "preliminary design checking support" | 독립 벤치마크 미실시 |
| 5 | "seamless path" | "a coherent workflow linking BIM data to structural analysis" | 미검증 |
| 6 | "LLM-based NL input is viable" | "shows potential as a supplementary interface" | 재현성 미검증 |
| 7 | "real-time design checking" | "integrated design checking capability" | 성능 벤치마크 없음 |
| 8 | "~30 min to ~2 min" | "substantially reduce manual input effort" | 정량적 비교 없음 |
| 9 | "14 MCP tools for LLM integration" | "14 analysis tools accessible via MCP interface" | 구현 수단 |
| 10 | "correctly maps 38 scenarios" | "tested against 38 predefined scenarios" | 테스트 결과 |

---

# 9. VALIDATED vs FUTURE SCOPE

| 항목 | 검증 완료 (Validated) | 향후 과제 (Future) |
|------|----------------------|-------------------|
| **기하** | 정형 직교 그리드 (벽/기둥 기반 IFC) | 비정형 평면, 세트백, 곡선 부재 |
| **재료** | 선형 탄성 철골 (SS275, SS400 등) | RC, 합성, 비선형 재료 |
| **단면** | H형강 (KS D 3502/3568, 738개) | 비표준 단면 |
| **하중** | DL/LL/EQ(등가정적)/Wind — KDS DB (712건) | 동적 해석, 국제 기준 |
| **지역** | 229개 시군구 위험도 (2,290건) | 국제 지역 DB |
| **해석** | 3D 6-DOF, Corotational, rigid diaphragm | 시간이력, 비선형 재료 |
| **벤치마크** | Midas Gen 5케이스, 112메트릭 | ETABS/SAP2000 |
| **설계검토** | KDS drift + AISC 360 **(독립 벤치마크 미실시)** | 독립 검증, 접합부 |
| **NL 입력** | 38개 시나리오, 30개 용도 | LLM 재현성, 다국어 |

---

# 10. SOURCE FILE REFERENCE MAP

| Paper Section | Primary Source Files | Lines | Evidence Type |
|---------------|---------------------|-------|---------------|
| §2.1 Architecture | `server.py`, `building_model.py` | 1,539 | Implementation |
| §2.2 Requirements | `ifc_parser.py`, `occupancy.json` | 967 | Implementation |
| §2.3.1 IFC Parsing | `ifc_parser.py` | 813 | Implementation + IFC tests |
| §2.3.2 NL Resolution | `nl_resolver.py`, `occupancy.json` | 829 | Implementation + 38 tests |
| §2.4.1 Load Generation | `load_generator.py`, `design_spectrum.py`, `kds_loads.py` | 1,271 | Implementation + DB |
| §2.4.2 3D FEA | `frame_3d.py`, `section_3d.py`, `ops_compat.py` | 2,005 | Implementation + Benchmark |
| §2.5.1 Design Check | `design_check.py` | 610 | Implementation + 16 tests |
| §2.5.2 Visualization | `visualization_3d.py`, `editor3d.js` | 4,247 | Implementation |
| §3 Benchmark | `benchmark/cases.py`, `compare.py`, Midas JSON | 1,206 | Validation |
| §4 Application | `main_simple.py`, `test_nl_resolver.py` | 1,416 | Demonstration |
| **Total core** | | **~14,900** | |

---

# 11. SUGGESTED FIGURES AND TABLES

## Figures

| Fig # | Description | Source | Section |
|-------|-------------|--------|---------|
| 1 | System architecture (3-stage block diagram) | To be drawn | §2.1 |
| 2 | IFC parsing pipeline (entity → IR mapping) | `ifc_parser.py` | §2.3.1 |
| 3 | Load generation pipeline (4 types + combinations) | `load_generator.py` | §2.4.1 |
| 4 | Benchmark Case 5 geometry (5-story 3D frame) | `benchmark/cases.py` | §3.1 |
| 5 | Displacement comparison scatter (OpenSees vs Midas) | `benchmark_charts/` | §3.3 |
| 6 | Metric distribution bar chart (OK/CHECK per case) | `benchmark_charts/` | §3.3 |
| 7 | Application: IFC vs analysis result 3D view | `editor3d.js` | §4 |
| 8 | Design check output: drift + member summary | `design_check.py` | §2.5.1/§4 |

## Tables

| Table # | Description | Source | Section |
|---------|-------------|--------|---------|
| 1 | IFC entity → structural parameter mapping | `ifc_parser.py` | §2.2 |
| 2 | KDS code references per load type | `load_generator.py` | §2.4.1 |
| 3 | OpenSeesPy solver configuration | `frame_3d.py` | §2.4.2 |
| 4 | Design check criteria (KDS + AISC) | `design_check.py` | §2.5.1 |
| 5 | Benchmark case specifications | `benchmark/cases.py` | §3.1 |
| 6 | Benchmark results summary (per case) | `benchmark/compare.py` | §3.3 |
| 7 | Validated scope vs future extensions | This skeleton §9 | §5/§7 |

---

# 12. LITERATURE REFERENCE PLAN

| # | 인용 역할 | 문헌 | 사용 위치 | 핵심 인용 문장/주장 |
|---|----------|------|----------|-------------------|
| 1 | IFC/openBIM 표준 | ISO 16739-1:2018 (buildingSMART) | §I1, §I2 | IFC가 vendor-neutral 표준 |
| 2 | BIM-structural 리뷰 | Fernández-Mora et al. (2022) | §I1, §I3 | "tendency towards design tools and new buildings" (§5, p.16) |
| 3 | BIM AMV 한계 | Hasan et al. (2019) | §I1, §I2, §I3 | "unnecessary reduction in geometrical representation drives loss of structural details" (Abstract) |
| 4 | BIM-OpenSees 연계 | Wang et al. (2025) | §I1, §I3, §I4 | "lack of effective data exchange bridges" (Abstract); BTO 3-stage framework |
| 5 | BIM-FEA 개방형 워크플로우 | Leonardi et al. (2024) | §I3, §2.2, §2.3.1 | "laborious, semi-automatic, and based on proprietary software" (Abstract); modelling requirements 패턴 |
| 6 | IFC→분석 모델 매핑/해석 | Ramaji & Memari (2018) | §I2, §2.2, §2.3 | IFC-to-analytical model interpretation |
| 7 | AI 구조설계 | Xie et al. (2025) | §I5 | AI/ML structural design context |
| 8 | LLM 설계 자동화 | Liao et al. (2024) | §I5 | LLM possibilities and limitations |
| 9 | OpenSees 프레임워크 | McKenna (2011) | §2.4.2 | OpenSees framework architecture |
| 10 | OpenSeesPy | Zhu et al. (2018) | §2.4.2 | Python binding for OpenSees |
| 11 | KDS 기준 | KDS 41 12 00, 41 17 00, 17 10 00 | §2.4.1, §2.5.1 | Korean Design Standards |
| 12 | AISC 360 | AISC 360-22 | §2.5.1 | Member strength criteria |
| 13 | FEA 비교 연구 | (추가 필요) | §3.4 | Element formulation differences |

---

# 13. PARAGRAPH-LEVEL REFERENCE MAPPING

> 각 논문이 어떤 섹션에서 어떤 역할로 사용되는지 한눈에 보기

| 참고 논문 | Introduction | Methods | Validation | Discussion | Conclusion |
|----------|-------------|---------|------------|------------|------------|
| **Hasan (2019)** | ¶1: AMV 제약 문제 정의, ¶2: semantic gap, ¶3: "insufficient definition" | §2.3.2: "intermediate solution" 프레이밍 | — | §5.2: deficiency 목록 스타일 | — |
| **Leonardi (2024)** | ¶3: "laborious, semi-automatic, proprietary" 갭 압축 | §2.2: modelling requirements 패턴, §2.3.1: 입력-처리-출력 알고리즘, scope control | §3.1: case study 목적 | — | [C2]: 번호 기여점, [C4]: 구체적 future work |
| **Wang (2025)** | ¶1: "rich info but bridge missing", ¶4: OpenSees 선택+한계 | §2.1: 3-stage 구조, §2.4.2: 좌표변환/단면계산 | §3.3: conversion 전후 비교 | §5.3: 기존 접근 비교 | [C1]: 파이프라인 재압축 |
| **Fernández-Mora (2022)** | ¶1: BIM 확산 배경, ¶3: "design tools/new buildings 편중" | — | — | §5.3: 연구 동향 위치 | — |
