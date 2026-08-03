# 데모 기능 전수조사 + 랜딩 재설계 (2026-08-03)

> 5-에이전트 워크플로우 산출물. 모든 항목은 file:line 근거 또는 라이브 서버 프로브로 검증됨.
> 브랜치 `demo/public-landing`, 서버 `127.0.0.1:8099` (KDS_CACHE_MODE=prefer, DEMO_MAX_MEMBERS=400) 기준.

---

## Part 1 — 재설계 제안

# 데모 루트 · 랜딩 재설계 제안

## 0. 설계 전제 (제약에서 도출)

| 제약 | 데모 설계에 미치는 결론 |
|---|---|
| 솔버 1스레드 · 1회 12s · 오버레이가 화면을 막음 (`figma_menu.js:441-453`) | **해석은 방문자당 1회**로 설계. 대기 중 탐색 불가 → 탐색은 Run **이전**에 배치 |
| 백엔드 수면 가능 · 랜딩은 정적 | 헤드라인 산출물(구조계산서·Excel·DXF)은 **랜딩에 baked 정적 파일로 선탑재**. 앱은 "재현 확인"용 |
| API key 필요한 것은 경로에서 제외 | NL(Claude), 챗봇(Ollama), RAG explain(Voyage) 전부 제외 |
| 1인 개발 | 신규 엔진 코드 0. **이미 동작하는 표면을 노출**시키는 작업만 |

**핵심 발견 1 — 인상적인 표면 대부분은 솔버가 필요 없다.** Section picker 738건, Material 18강종, Load 팝업(KDS 활하중), Model Explorer 트리는 모두 모델 없이 즉시 열린다. 실측: `GET /api/sections/list` 200/11,956B, `/api/materials/list` 200, `/api/loads/live-preview` 200/1.3ms. → **Run 전 3분이 데모의 알맹이**다.

**핵심 발견 2 — 랜딩이 파는 모델과 앱 데모가 다른 모델이다.** 랜딩 hero는 "이 3층 골조는 지금 NG"(H-300x300, ratio 1.4933, NG 10/63)인데 CTA는 `?demo=ifc`(87부재, **전 부재 OK, max 0.4329**)로 간다. 방문자는 NG를 보러 와서 초록색 전부-OK를 만난다. DC Colors를 켜도 87개 전부 초록이라 "아무 일도 안 일어난 것"으로 보인다.
그런데 **baked bundle의 base_config는 에디터 기본 프리셋과 완전히 동일**하다 — `data/demo_bundle/manifest.json` base_config(3층 4.0/3.5/3.5 office, 8×8 2×2, H-300x300, H-400x200, SS275, 서울, II, ELF) ↔ 라이브 프로브 결과(36 nodes / 63 elements / total_weight **3916.8 kN**) ↔ bundle `load_chain.seismic.W_kN` = **3916.8**. 숫자가 일치한다.
→ **주 CTA를 `?demo=bench`(직접입력 기본 프리셋)로 바꾸면, 방문자의 실행 결과가 랜딩에 인쇄된 숫자를 그대로 재현한다.** 이건 만들 필요가 없는, 이미 참인 신뢰 장치다.

**핵심 발견 3 — 랜딩은 이미 구운 데이터의 15%만 쓰고 있다.** `landing/assets/bench.js` grep 결과 `load_chain`은 `W_kN`/`V_kN` 2개 스칼라만, `combos`(36개)·`modal`(9모드)·wind/snow 체인·members 63행 전체는 **전혀 렌더하지 않는다**. 아래 신규 섹션 대부분은 **재베이크 0, 템플릿+bench.js 작업만**으로 나온다.

---

## 1. 랭크된 데모 루트

각 단계는 딥링크 1개 또는 명백한 클릭 1회. ⏱은 누적 시간, 💾는 솔버 사용.

### 【Tier A — 서버 없이, 랜딩에서】 ⏱0:00–2:00 · 💾0

**A1. Hero bench에서 판정을 뒤집는다** *(기존, 유지)*
- 한다: 기둥 단면 드롭다운을 H-250x250 → H-400x400로 움직인다
- 본다: 종합 NG↔OK, 상관비 2.1809 → 1.4933 → 0.9658 → 0.9295, 프레임 SVG의 NG 부재가 빨강→회색으로 바뀜
- 이유: 첫 5초에 "숫자가 살아있다"를 증명. 재계산 없이 4개 실해석 결과를 왕복하므로 서버 무관
- 링크: 없음 (인라인 baked)

**A2. 구조계산서 원본을 연다** ★ *(신규 — 최우선)*
- 한다: "구조계산서 원본 보기" 클릭
- 본다: 540KB 한국어 문서형 계산서. 10개 번호 섹션 + 부록 3, 표 28개, §4에 z=0.11 → Fa 1.68 → SDS 0.462 → Ta 0.493 → Cs 0.1077 전개, §8 Pu/Mux/Muy, 부록 A.1 전 부재 일람, 작성자/검토자/승인자 도장란
- 이유: **이 도구의 최종 산출물이 무엇인지 1클릭으로 끝낸다.** "웹 데모"가 아니라 "제출 문서"라는 프레이밍 전환
- 링크: `landing/files/report_example.html` — **신규(베이크 필요)**. 실물은 이미 존재: `webapp/backend/jobs/73a55f52-.../report.html` 539,815B, 외부 참조는 plotly CDN 1개뿐 → 정적 복사로 성립

**A3. 하중 산정 영수증을 읽는다** ★ *(신규)*
- 한다: 스크롤
- 본다: baked `load_chain` 전체 — 지진 `z=0.11 / Fa=1.68 / Fv=1.69 / SDS=0.462 / SD1=0.1859 / R=3.5 / Ct=0.0724 / Ta=0.493s / Cs=0.1077 (Cs_governed_by: Cs_max) / W=3916.8kN / V=421.98kN`, 풍 `V0=30m/s, exposure B, Gf=2.2, Cp=1.3 → Fx=182.06kN`, 눈 `Sg=0.5, Cb=0.7 → S=0.5 (governed_by: minimum)`
- 이유: **반환각독(anti-hallucination)의 유일한 결정적 증거.** "LLM이 지어낸 게 아니라 표에서 끌어온 값"을 조항 코드와 함께 보여줌
- 링크: 없음 (bundle에 이미 있음, 렌더만 추가)

**A4. 산출물 3종을 내려받는다** ★ *(신규)*
- 한다: Excel / DXF / IFC 다운로드
- 본다: `.xlsx` 167KB(5시트, Member_Forces 89×263, 43케이스×87부재 22,446셀), `.dxf` 317KB(평면 4 + 입면 7, 254 LINE / 80 DIMENSION / 68 레이어), `.ifc` 47KB
- 이유: 실무자는 "화면"이 아니라 "파일"을 신뢰한다. 로그인 없이 열어볼 수 있는 실물
- 링크: `landing/files/{tables_example.xlsx, drawing_example.dxf, ifc_example.ifc}` — 앞 2개 **신규(베이크 필요)**

### 【Tier B — 앱, 솔버 0】 ⏱2:00–5:00 · 💾0

**B1. 벤치와 같은 모델로 에디터를 연다** ★
- 한다: "이 모델을 직접 돌려보기" 클릭
- 본다: 3D 뷰에 3층 골조 와이어프레임이 이미 그려져 있고, 좌측 Model Explorer 트리가 채워지고, ▶ Run에 코치마크가 붙음. 상단에 "랜딩의 숫자와 같은 모델입니다"
- 이유: 빈 화면 없음. 그리고 **곧 나올 숫자를 이미 알고 있다**는 긴장
- 링크: `/editor-figma?demo=bench` — **신규**. `applyPreset()`이 이미 DOMContentLoaded에서 3story로 폴백하므로(`editor3d_figma.js:108-112`) 실질 작업은 "프리셋 확정 + 코치마크"뿐

**B2. 단면 카탈로그를 연다** ★
- 한다: 리본 **S** 클릭 (또는 딥링크로 자동 오픈)
- 본다: KS D 3502/3568 **738건**, 12분류(H형강 94, 원형강관 235, 정사각중공 132…), Type 필터 + Find, 행 더블클릭 → 치수 + 캔버스 단면 스케치 + A/Ix/Iy/J. `H-400x400` → A=218.7cm², Ix=66,600cm⁴, J=273.2cm⁴
- 이유: **"목업이 아니다"를 가장 싸게 증명하는 화면.** 모델도 로그인도 해석도 필요 없다
- 링크: `/editor-figma?demo=bench&panel=sections` — **신규** (`window.figmaOpenSections` 이미 존재, `figma_section.js:149`)

**B3. J 근사 고지를 본다** *(B2 안에서 1클릭)*
- 한다: "Show Section Properties…" 
- 본다: J 값 옆 별표 + "개방단면 근사" 경고. 실측 분포 `j_source`: db 499 / approximate 219 / fallback 20
- 이유: 스스로 약점을 먼저 말하는 유일한 화면. 심사자·교수에게 가장 잘 먹히는 1초
- 링크: 없음

**B4. 활하중이 KDS 표에서 온다는 걸 본다** ★
- 한다: 리본 **L** → 층별 용도를 사무실 → **창고**로
- 본다: 2.5 → **6.0 kN/m²**로 즉시 변경, 힌트에 `KDS 41 12 00 표 3.2-1` + db_primary_key. 실측: `/api/loads/live-preview` 11개 용도 전부 `source=db_lookup`
- 이유: **화면에 표시된 값이 해석에 들어가는 값과 같은 코드 경로**라는 점이 유일하게 눈으로 검증되는 지점 (검증됨: 창고 전환 시 생성된 LL floor_area가 정확히 2.5→6.0)
- 링크: `/editor-figma?demo=bench&panel=load-live` — **신규** (`figma_load.js:550` 존재)
- ⚠ 데모 스크립트에서는 **되돌린 뒤** Run할 것. 안 그러면 랜딩 숫자와 안 맞는다. → 안내 문구 필요

**B5. 지진하중 입력을 본다** *(선택)*
- 한다: 리본 **E**
- 본다: 지역 + 지반 S1~S5 + 중요도 + 횡력저항시스템 10종 + ELF/RSA 전환
- 이유: 응답스펙트럼까지 있다는 사실을 알림. 다만 값을 바꾸면 Tier C의 재현이 깨지므로 **보기만**
- 링크: `?panel=load-seismic`

### 【Tier C — 앱, 해석 1회】 ⏱5:00–6:00 · 💾1회 12s

**C1. ▶ Run** ★★
- 한다: 리본 ▶ Run 1회
- 본다: 모달 스피너 + 경과 초 카운터 → 12초 후 결과. **Analysis Control에 `36 combinations` / `7 load cases`**, 종합 NG
- 이유: **랜딩에 인쇄된 W=3916.8kN / V=421.98kN / max ratio 1.4933 / 지배조합 `1.2924DL+1.0LL+1.0EQY+0.3EQX+0.2S`가 그대로 재현된다.** "우리가 미리 계산해 둔 값 = 방금 당신이 돌린 값" — 다른 어떤 문구보다 강력
- 링크: `?demo=bench&run=1`(자동 실행)도 가능하나 **비권장** — 클릭의 소유감을 없애고, 대기 12초가 첫 인상이 됨

**C2. DC Colors 체크** ★
- 한다: 3D 뷰 상단 `DC Colors` 체크박스
- 본다: **NG 10개 부재가 빨강**, 나머지 초록/황색
- 이유: bench 모델이라야 성립. (IFC 모델은 87개 전부 초록이라 무의미)
- 링크: 없음 (화면에 노출된 2개 토글 중 하나)

**C3. Design Check 표** ★
- 한다: 우측 Result Tables → Design Check
- 본다: 63행, 상관비 내림차순, 1행이 1.4933
- 이유: 한 화면에서 "무엇이 왜 NG인지"
- 링크: `?result=design` — **신규** (`figma_explorer.js:335` 존재). ⚠ 정렬·governing 채우기 선행 필요(§4 S8)

**C4. Generate Report** ★★
- 한다: 우측 Analysis Control → Generate Report
- 본다: A2에서 본 것과 **같은 형식의, 방금 자기 해석으로 만들어진** 계산서
- 이유: 루프 종결. "샘플이 아니라 내 실행의 산출물"
- 링크: 없음

**C5. Export Tables (Excel)**
- 한다: Export Tables
- 본다: 즉시 다운로드
- 이유: 마지막 실무 신뢰 도장
- 링크: 없음

### 【Tier D — 2차 방문자 / 관심 있는 사람만】

**D1. IFC 경로** — `/editor-figma?demo=ifc` (기존). 48절점/87부재, 파싱 0.38s, 접합 스냅 102건. 랜딩에서 **별도 섹션**으로 분리하고 "논문에서 ETABS와 31/31 일치 확인" 문구를 여기로 이전. (현재 이 문구가 run 섹션에 있는데, 주 CTA를 bench로 바꾸면 위치가 틀려진다)

**D2. OK로 뒤집기 (해석 2회차)** — `?demo=bench&col=H-350x350`. 랜딩 knob에서 H-350x350를 고른 뒤 나타나는 "이 판정을 앱에서 확인" 링크. 1클릭 = 1런.

**D3. 자기 IFC 업로드** — 400부재 상한, 업로드 경고 유지.

**루트에서 제외한 것**: 자연어 입력(빌링 400), 챗봇(ConnectError), 평면 그리기(Run 라우팅 시 무고지 소실 버그), 비정형(96초, 단일 솔버 독점), 10층 프리셋(400부재 = 약 6GB), 모드 애니메이션(CSS 2줄에 막혀 도달 불가), 추천 레이어(DOM 부재).

---

## 2. 개정 랜딩 섹션 구성

원칙: **산문 대신 baked 실물**. 순서는 "증거 → 산출물 → 한계 → 실행".

| # | 섹션 | 보여주는 것 | 데이터 출처 | 신규? |
|---|---|---|---|---|
| **00** | Hero — the bench | 기존 유지. 문구만 1줄 추가: "아래 버튼을 누르면 이 숫자가 **같은 값으로** 재현됩니다" | 기존 인라인 | 문구만 |
| **01** | **최종 산출물** ★ | 계산서 첫 페이지 스크린샷 + "구조계산서 원본 열기(540KB)" + 섹션 목차 10개 나열 | `landing/files/report_example.html` | **신규(베이크)** |
| **02** | **하중은 어디서 왔나** ★ | 지진/풍/눈 3열 영수증 카드. 각 값에 KDS 조항 라벨. `Cs_governed_by: Cs_max`, `governed_by: minimum` 같은 필드명 그대로 노출 | bundle `load_chain` (**이미 존재**) | **신규(렌더만)** |
| **03** | 검토 10항목 | 기존 유지 | bundle `checks` | — |
| **04** | **하중조합 36개** | 접이식 `<details>`. 36개 전부 나열, 지배조합 하이라이트 | bundle `combos` (**이미 존재**) | **신규(렌더만)** |
| **05** | **부재 63개 일람** | 정렬 가능한 표. NG 10행 빨강, 상관비/지배조합/단면 | bundle `members` (**이미 존재**) | **신규(렌더만)** |
| **06** | **고유치** | T1x/T1y/T1rz + 9모드 주기·질량참여, 누적 100% | bundle `modal` (**이미 존재**) | **신규(렌더만)** |
| **07** | **파일로 나갑니다** ★ | Excel/DXF/IFC 다운로드 3카드 + 각 내용 요약(5시트 / 평면4·입면7 / 47KB) | `landing/files/` | **신규(베이크)** |
| **08** | **입력은 3가지 경로** | 3카드 각각 스크린샷 1장 + 딥링크: ①직접입력(738단면 picker 캡처) ②IFC(파싱 결과 캡처, ETABS 31/31 문구 여기로) ③평면 그리기(캡처만, 링크는 "준비 중") | 캡처 3장 | **신규** |
| **09** | 안 되는 것들 | 기존 유지 + 2줄 추가: "RSA 부재력 미생성", "재료는 모델당 1강종으로 축약됨(`frame_3d.py:2792`)" | 기존 | 문구 추가 |
| **10** | 직접 돌려보세요 | **주 CTA = `?demo=bench`**, 부 CTA = `?demo=ifc`. `/api/health/demo` 프로브 결과에 따라 버튼 라벨 변경("지금 실행 가능" / "N건 대기 중") | 라이브 | **CTA 교체 + health** |
| **11** | 피드백 | 기존 | — | — |

01~07은 **한 번도 서버를 부르지 않는다.** 백엔드가 죽어 있어도 랜딩만으로 이 도구의 전모가 전달된다 — 이것이 "얇다"는 문제의 실제 해법이다.

02·04·05·06은 **재베이크 불필요**. `data/demo_bundle/variant_h300x300.json`에 이미 들어있고 `bench.js`가 안 읽고 있을 뿐이다.

---

## 3. 공개 데모에서 숨기거나 게이트할 것

### 즉시 차단 (보안/평판)

| 대상 | 현 상태(실측) | 조치 | 이유 |
|---|---|---|---|
| `/docs`, `/openapi.json` | 200, 34 path 전체 노출 | `FastAPI(docs_url=None, openapi_url=None)` — `main_simple.py:205` | 아래 모든 깨진 표면으로 가는 발견 경로 |
| `/editor-v2`, `/editor-lab` | 둘 다 200, 무인증, 챗봇 FAB 탑재 | 라우트 제거 또는 `DEMO_AUTH_TOKEN` 게이트 — `main_simple.py:406-416` | URL을 만져본 방문자가 `ConnectError: All connection attempts failed`(영문 스택 냄새)를 만남 |
| `chat_router` 전체 | `/api/v2/chat/audit/{id}`가 **무인증 operator 권한**, `include_quotes=true` 허용 | 데모 배포에서 include 하지 않음, 또는 `DEMO_AUTH_TOKEN` 설정 | 저작권 있는 KDS 인용 원문이 익명으로 덤프 가능 |
| `/api/v2/recommendations/evaluate` | 후보 개수 무제한, 솔버 스레드 1개 공유, `_enforce_demo_size` 미적용 | 배포에서 미노출 (UI 없으므로 라우트만 제거) | 24후보 제출 1건이 데모 전체를 수 분 점유 |

### UI에서 감추기 (깨져 보임)

| 대상 | 조치 | 위치 |
|---|---|---|
| 자연어 입력 | File/Load 메뉴 항목 제거 (또는 버튼 disable + 한국어 안내) | `figma_menu.js:174, 225` |
| 리본 Plate / Spring / Constraint | `disabled` 속성 부여 또는 메뉴와 동일한 `soon()` 토스트 | `editor_figma.html:61, 73, 74` |
| 리본 "Model" 탭 · 좌측 "Model" 탭 | 드로어 오프너로 재배선하거나 제거 (현재 클릭해도 아무 변화 없음) | `editor_figma.html:50, 117` |
| Model 메뉴 → '보 단면 지정' | 제거 ('beam' 분기 부재로 무동작) | `figma_menu.js:210` |
| 10층 프리셋 | 옵션에서 제거 **또는** `DEMO_MAX_MEMBERS=250` | `editor3d_figma.js` applyPreset / 서버 env |
| 비정형 체크박스 | 유지하되 "약 90초 소요" 라벨 부착 (135부재에 96.6s 실측) | `editor_figma.html:169-217` |
| 상단 `Project: academy_tower_v1.opf` | 문자열 교체 (File에 Open/Save 없음 = 거짓 고지) | `editor_figma.html:42` |

### 남겨도 되는 미구현

Section 팝업의 disabled 7버튼(Import/Add New/Delete/Export XML)은 **툴팁으로 사유를 말하고 있으므로 정직하다**. 그대로 두되, 랜딩 스크린샷은 목록+필터 영역만 크롭할 것.

---

## 4. 빌드 리스트 (영향/노력 순)

### 【링크 보내기 전 — 필수】 총 예상 1~1.5일

| # | 작업 | 파일 | 노력 | 영향 |
|---|---|---|---|---|
| **S1** | 공개 표면 차단: docs/openapi 끄기, `/editor-v2`·`/editor-lab` 라우트 제거, chat_router 미포함, recommendations/evaluate 미노출 | `webapp/backend/app/main_simple.py:205, 406-416` + router include | 30분 | 최악의 인상(ConnectError·무인증 감사로그) 제거 |
| **S2** | **정적 산출물 베이크**: `report.html` / `.xlsx` / `.dxf`를 `landing/files/`로 | `scripts/bake_demo_bundle.py` (기존에 `plot_frame_3d_calc_report` 호출 추가 — `visualization_calc_report.py:30`), 또는 라이브 서버 1회 실행 후 `curl /api/jobs/{id}/report`·`/api/export/excel/{id}`·`/api/export/dxf` 결과 복사 | 2시간 | **백엔드 수면 무관하게 헤드라인 산출물 노출.** 실물은 `webapp/backend/jobs/73a55f52-.../report.html`로 이미 존재 |
| **S3** | **`?demo=bench` 딥링크 + 랜딩 주 CTA 교체** | 신규 `webapp/backend/static/js/figma_deeplink.js?v=1`, `editor_figma.html:873-881`에 script 추가, `landing/index.template.html:191` `data-app-path` | 3시간 | 랜딩 서사(NG)와 앱 데모(전부 OK) 불일치 해소 + 숫자 재현 신뢰장치 확보 + DC Colors가 의미를 가짐 |
| **S4** | **랜딩 신규 섹션 02/04/05/06** (하중 영수증·36조합·부재 63행·고유치) | `landing/index.template.html`, `landing/assets/bench.js`, `bench.css` | 4시간 | **재베이크 0.** 랜딩 정보량 3~4배. "얇다"의 직접 해결 |
| **S5** | 랜딩 신규 섹션 01/07 (계산서·다운로드 카드) | 위 + S2 산출물 | 2시간 | 최종 산출물이 무엇인지 1클릭 |
| **S6** | 하단 상태바 라벨 4개 교체 | `editor_figma.html:839, 843, 847, 857` (→ `Max Drift X`, `Max Drift Y`, `Max Disp X`, `Design Check`) | **5분** | "Last run: 0 warnings, 0 errors — 23.56mm" 라는 자해적 표시 제거. 값은 이미 정확함 |
| **S7** | 죽은 버튼 정리 (Plate/Spring/Constraint/Model탭/보단면지정/NL) | `editor_figma.html:50, 61, 73, 74, 117`, `figma_menu.js:174, 210, 225` | 1시간 | 리본 탐색 시 무반응 클릭 확률 대폭 감소 |
| **S8** | Design Check 표: ratio 내림차순 정렬 + `governing_combo` 채우기 | `figma_explorer.js:300-307` (정렬), `main_simple.py:1441-1446` (필드 4개 추가) | 1시간 | 결과 팝업 5개 중 가장 많이 눌리는 것이 자립함. 빈 governing 열 제거 |
| **S9** | Result Table 기본 케이스를 `case_names[0]`(=DL) → 지배조합으로 | `figma_explorer.js:286` | **15분** | Story Drift 첫 화면이 전부 0.0인 문제 제거 |
| **S10** | 10층 프리셋 제거 또는 `DEMO_MAX_MEMBERS=250` | `editor3d_figma.js` applyPreset 목록 / 배포 env | 15분 | 400부재 ≈ 6GB RSS 폭발 방지 |
| **S11** | `?panel=sections` / `?panel=load-live` 딥링크 | S3의 `figma_deeplink.js`에 포함 | (S3에 포함) | Tier B 스텝을 랜딩에서 1클릭으로 |
| **S12** | 랜딩이 `/api/health/demo` 프로브 후 CTA 라벨 조정 | `landing/assets/bench.js` | 30분 | 엔드포인트는 이미 존재(`main_simple.py:3175`)하고 docstring이 "랜딩이 호출한다"고 쓰여 있으나 **실제로는 아무도 안 부름** |
| **S13** | 비정형 "약 90초" 라벨 | `editor_figma.html:169-217` | 10분 | 96.6s 무설명 블로킹 방지 |

### 【나중에 — 가치순】

| # | 작업 | 파일 | 노력 | 비고 |
|---|---|---|---|---|
| L1 | **모드 형상 애니메이션 부활** | `editor_figma.html:799` (`display:none` 제거) + `editor_figma.css:2385-2387` 밖으로 재배치 | 반나절 | 이 영역에서 가장 시각적으로 강한 기능이 **CSS 2줄에 막혀** 도달 불가. 엔진(`startModeAnimation`/jet colormap 범례)은 완성돼 있음 |
| L2 | 추천 레이어 **preview-apply만** 이식 | `editor_v2.html:1044, 1052, 1122`의 `rec-issues-list`/`rec-candidates-list`/`rec-diff-modal` 마크업 → `editor_figma.html`. 렌더러는 `editor3d_figma.js:4086, 4328`에 이미 존재 | 1일 | "H-300x300 → H-310x310, 사유 strength_exceeded" 디프 = 솔버 0·키 0. **`evaluate`(솔버 독점)와 `explain`(Voyage 키 + 환경변수 문자열 노출)은 이식 금지** |
| L3 | Run 라우팅 버그: Manual 모델에서 편집이 무고지 소실 | `figma_menu.js:403-405` | 반나절 | **평면 그리기를 공개하기 전 필수** |
| L4 | Member Force 팝업 5-station 전개 (현재 12열 중 7열이 JSON 배열) | `figma_explorer.js:297-321` | 반나절 | 엔지니어가 가장 먼저 누르는데 가장 못 읽힘. 같은 숫자가 Excel에선 완벽 |
| L5 | 변형형상/다이어그램 컨트롤 스트립 이식 (케이스 선택 + 스케일 슬라이더) | `editor_figma.html` (v2에서 이식) | 1일 | 현재 메뉴로만 도달, `1.4DL` 고정 50배. 아이러니하게 **계산서 §6.1에는 둘 다 있다** |
| L6 | LLM 서술을 결과 패널에 노출 | `.legacy-property-ui` 밖으로 `#interp-summary` 재배치 | 반나절 | 실제 생성 확인됨(`claude-opus-4-8`, V=Cs·W=211kN, H-300x300→H-350x350 권고). 지금은 리포트를 열어야만 보임 |
| L7 | `saveProject`/`loadProject`를 File 메뉴에 배선 | `figma_menu.js:172-181` (함수는 `editor3d_figma.js:7507, 7549`에 완성) | 1시간 | 상단 `.opf` 표시의 거짓말 해소 |
| L8 | 표지/도장란 — 서버측 데모 표지 프리시드 | `bake` 또는 `openCoverModal` DOM 이식 | 반나절 | 계산서 첫 페이지가 `[프로젝트명]`·`[작성자]` 대괄호. 랜딩 캡처 1장이면 티가 남 |
| L9 | 릴리즈 표시 버그 (`fmtEnd`가 문자열 `'all'`을 배열로 검사) | `figma_plan.js:576` | **1줄** | 모델·해석은 정상, 표시만 거짓 |
| L10 | 릴리즈 적용 시 T1 붕괴(0.275s → 0.0123s) 원인 규명 | `frame_3d.py` equalDOF 힌지 절점 질량 | 조사 필요 | 원인 미확인. **그때까지 릴리즈 모델의 모달 결과를 데모하지 말 것** |

---

## 5. 신규 딥링크 · 파라미터 명세

전부 **1개 신규 파일** `webapp/backend/static/js/figma_deeplink.js`에 URL 파서 하나로 구현. `editor_figma.html:873-881` 스크립트 블록 **마지막**(explorer 뒤)에 추가하고, `window.addEventListener('load', ...)`에서 실행 — 기존 `?demo=ifc` 블록(`editor3d_figma.js:118-127`)과 동일한 타이밍 규약을 따라야 `figma_menu.js`의 `uploadIFC` 래핑이 완료된 뒤 동작한다.

| 파라미터 | 값 | 동작 | 필요한 배선 | 상태 |
|---|---|---|---|---|
| `demo` | `ifc` | 번들 예제 IFC 자동 로드 | — | **존재** (`editor3d_figma.js:118-127`) |
| `demo` | `bench` | 3story 프리셋 확정 → `applyPreset('3story')` + `updateManualPreview()` + ▶ Run 코치마크 | `applyPreset`은 이미 window 전역 | **신규 · 최우선** |
| `col` | KS 단면명 (`H-350x350`) | 기둥 단면 select 프리필 후 프리뷰 갱신 | `#column-section` 값 설정 + `updateManualPreview()` | **신규** |
| `beam` | KS 단면명 | 보 단면 프리필 (X/Y 동시) | 동상 | 신규(선택) |
| `run` | `1` | 로드 완료 후 `runAnalysis()` 자동 호출 | `/api/health/demo`가 `ready:true`일 때만 실행하도록 가드 | 신규(선택, **주 경로엔 비권장**) |
| `panel` | `sections` | `window.figmaOpenSections()` | 이미 전역 (`figma_section.js:149`) | **신규(파서만)** |
| `panel` | `materials` | `window.figmaOpenMaterials()` | 이미 전역 (`figma_material.js:177`) | 신규(파서만) |
| `panel` | `load-dead`\|`load-live`\|`load-seismic`\|`load-wind` | `window.figmaOpenLoads(kind)` | 이미 전역 (`figma_load.js:550`) | **신규(파서만)** |
| `panel` | `input`\|`ifc` | `openInputDrawer(tab)` | ⚠ **현재 module-private** (`figma_menu.js:133`, IIFE 내부, window 미노출) → `window.figmaOpenInputDrawer = openInputDrawer;` 1줄 추가 필요 | 신규(1줄 + 파서) |
| `result` | `drift`\|`reaction`\|`member`\|`design`\|`modal` | `window.figmaOpenResultTable(kind)` | 이미 전역 (`figma_explorer.js:335`). 결과 없으면 자체 경고 표시하므로 `run=1`과 조합하거나 해석 완료 후 지연 실행 | 신규(파서만) |
| `report` | `1` | 해석 완료 시 `figmaOpenReport()` 자동 | 이미 전역 (`figma_explorer.js:273`) | 신규(선택) |
| `tour` | `1` | 코치마크 시퀀스(B1→B2→B4→C1→C2→C4) | 신규 `figma_tour.js` — 스텝별 대상 셀렉터 + 다음 버튼. 각 스텝은 위 전역 함수 호출 | 신규(중간 노력, 나중에) |

**랜딩에서 쓰이는 최종 URL 5개**

```
/editor-figma?demo=bench                       ← 주 CTA (섹션 10)
/editor-figma?demo=bench&panel=sections        ← 섹션 08 "직접입력" 카드
/editor-figma?demo=bench&panel=load-live       ← 섹션 02 "하중 영수증" 하단 링크
/editor-figma?demo=bench&col=H-350x350         ← 섹션 00 knob에서 OK 선택 시 노출
/editor-figma?demo=ifc                         ← 섹션 08 "IFC" 카드 (기존)
```

전부 `landing/index.template.html`의 `data-app-path` 속성으로만 쓰고, origin은 `<meta name="app-base">` 한 곳에서 해결한다(`bench.js:47-51`, 기존 규약 유지). 현재 랜딩은 앱과 동일 오리진에서 `/`로 서빙되므로(`main_simple.py:226-232`, `/landing` → 307 → `/`) app-base는 빈 값 그대로면 된다.

---

## 요약 — 가장 중요한 3가지

1. **주 CTA를 `?demo=ifc` → `?demo=bench`로 바꿔라.** 랜딩이 파는 NG 서사와 앱이 보여주는 전부-OK 모델이 다른 것이 현재 "얇음"의 절반이다. 그리고 bench 프리셋의 baked 값(W=3916.8kN)과 라이브 실행 값이 **이미 일치**하므로, 코드 없이 "재현 가능"이라는 최상급 신뢰 장치를 얻는다.
2. **랜딩에 baked 산출물 4종(계산서·Excel·DXF·하중 영수증)을 얹어라.** 계산서 실물은 `webapp/backend/jobs/73a55f52-.../report.html`(540KB, 외부 참조 plotly 1개)로 이미 디스크에 있고, 하중 체인·36조합·9모드·63부재는 `data/demo_bundle/`에 **이미 구워져 있는데 렌더만 안 되고 있다**. 재해석 0, 재베이크 거의 0.
3. **해석 전 3분(Section 738 · Load KDS 표 · Material 18강종)이 데모의 알맹이다.** 솔버 0, 키 0, 모델 0으로 열리는 화면들이고, 동시에 12초 대기 이전에 배치할 수 있는 유일한 구간이다(Run 오버레이가 화면을 막으므로 실행 중 탐색은 불가능).

---

## Part 2 — 기능 인벤토리 (64건)

| 데모가치 | 상태 | 기능 | 진입점 | 근거 |
|---|---|---|---|---|
| headline | works | POST /api/v2/recommendations/explain (KDS-RAG evidence) | Backend only on /editor-figma. On /editor-v2: '추천 근거 설명' button on a candidate -> rec-explain-modal (editor_v2.html:1148). | LIVE PROBE -> 200 with a real Voyage call. source={"deterministic":true,"rag_used":true,"llm_used":false,"score_method":"heuristic_v1"}, confidence=medium, warnings=["evaluation_missing: 재해석 검증이 아직 수행되지 않았습니다."], and 5 kds_evidence items: KDS 14 31 10 §4.4.1.1 |
| headline | works | Direct/manual config input (직접 입력 drawer) | Top menu bar → File → '직접 입력 (모델 정의)'. This is the ONLY working entry: it calls openInputDrawer('manual') which physically moves the hidden .legacy-co | UI: editor_figma.html:144-309 (#tab-manual); hidden by editor_figma.css:2385-2387 `.legacy-config-ui{display:none!important}` until the drawer is built. Menu wiring figma_menu.js:173. Config assembly editor3d_figma.js:2260-2314. HTTP probe: POST /api/v2/analyz |
| headline | works | IFC import — bundled sample / landing deep link | Two ways: (1) landing button → /editor-figma?demo=ifc auto-fires loadSampleIFC() on window load (editor3d_figma.js:118-127; landing/index.html:269 dat | loadSampleIFC() editor3d_figma.js:1036-1058 fetches /static/files/ifc_example.ifc (file present, 47,608 bytes) and reuses uploadIFC(). LIVE PROBE: POST /api/v2/parse-ifc with that file → HTTP 200 in 0.38 s, success=true, 150 nodes / 87 elements, 36 columns + 5 |
| headline | works | Section picker — Frame Properties list (738 KS sections, Type filter + | Ribbon → Properties group → **S / Section** button (editor_figma.html:67), or menu bar Model → '단면 정의 (Sections)…' (figma_menu.js:206). Both call wind | figma_section.js:149-157 figmaOpenSections; skeleton+filters at :160-230; renderList :237-279. Data source is the engine global `sectionsList`, fetched at page load independent of any model (editor3d_figma.js:249-268 loadSectionsAndMaterials, called from the D |
| headline | works | Section picker — Frame Section Property Data (dimensions + canvas cros | Inside the Frame Properties window: double-click a row, or select a row → 'Modify/Show Property…' (figma_section.js:206-209 / :263). | figma_section.js:288-367 openDetail; dimension rows per shape type at :368-377; canvas sketch renderer :380-472 (draws H/I, C, T, L, FB, CHS, SHS/RHS outlines plus ETABS-style green '2' / blue '3' axis arrows); material pairing select at :302-320 with a '…' bu |
| headline | works | Load popup — L / Live (per-story usage → KDS auto live load) | Ribbon → Load Case group → **L / Live** (editor_figma.html:79), or menu bar Load → '활하중 정의 (Live)…' (figma_menu.js:224). Both call figmaOpenLoads('liv | figma_load.js:396-442 openLive. Per-story usage dropdown (11 options), with the live kN/m² value re-read on every change (:414). **Live analysis probe** (2-story model, POST /api/v2/analyze): office → storage changed LL 2.5 → 6.0 kN/m², total_weight_kN 367.2 → |
| headline | works | Load popup — E / Seismic (region, site class, importance, lateral syst | Ribbon → Load Case group → **E / Seismic** (editor_figma.html:80) or menu Load → '지진하중 정의 (Seismic)…' (figma_menu.js:225). | figma_load.js:466-513 openSeismic. 10 lateral systems (:445-456), 5 site classes (:457-463), 3 importance levels (:464), region as a free-text input with a 19-city datalist (:314-327, deliberately free-form because the hazard DB holds 229 시·군·구). RSA sub-rows  |
| headline | works | Korean structural calculation report (문서형 구조계산서) | Right panel 'Analysis Control' -> Generate Report button (figma_explorer.js:273-277 window.figmaOpenReport, editor_figma.html:721), or menu File -> 해석 | HTTP probe: GET /api/jobs/73a55f52-9eac-4688-bbc0-bd0162e5c872/report -> 200, 539,815 bytes (demo IFC job); GET /api/jobs/b7077a45-.../report -> 200, 400,224 bytes (3-story config job). 14 <h2>: 10 numbered sections (구조설계 개요 / 적용 기준 / 사용 재료 / 하중 산정 / 하중 조합 / 해 |
| headline | works | Excel export (/api/export/excel/{job_id}) | Right panel 'Analysis Control' -> Export Tables button (editor_figma.html:722 -> figma_explorer.js:278-283 figmaExportTables -> exportToExcel, editor3 | HTTP probe (demo IFC job): GET /api/export/excel/73a55f52-... -> 200, 166,957 bytes, content-type application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, content-disposition attachment. openpyxl read-back: 5 sheets — Displacements 50x262, Member_For |
| headline | works | Design-check depth behind the results (10 check groups, AISC H1 per me | Surfaced three ways: Result Tables -> Design Check popup (ratios only), report §8 + Appendix A.1 (full), and the DC Colors toggle on the 3D view. The  | Live response inspection (3-story config job, resp keys design_check.*): 10 groups all present and all carrying a status — drift_check OK, member_check NG, deflection_check OK, system_check OK, stability_check OK, pdelta_check OK, torsion_check OK, wind_check  |
| strong | partial | Plan drawing — beams / columns / nodes with snapping and auto-split | Ribbon Geometry group: 'N' (Node) and 'B' (Beam) buttons (editor_figma.html:59-60), or menu Edit / Node-Element → 절점 추가 · 부재 추가. Entering the mode rev | figma_plan.js: draw panel 472-506, click handling 455-469 + canvas listeners 766-781 (left-click draw, right-click break chain / exit), snapping snapPoint/snapAxis 329-355 (existing node > grid axis > 0.5 m), creation createPlanNode/createPlanBeam/createPlanCo |
| strong | partial | Selection edit panel — per-member 단면 / 재료 / 릴리즈 dropdowns, per-node 지점 | Click any member (3D view or Plan view) → the '편집' panel appears beside the Properties-of-Object readout (figma_plan.js:699-717 buildEditPanel). | figma_plan.js:699-717. For an element it builds three selects — 단면 (options from the 738-item sectionsList, :670-678) → bulkApplySection; 재료 (options from window._figmaMaterialNames(), figma_material.js:75-81) → bulkApplyMaterial; 릴리즈 (Fixed–Fixed / Pin i / Pi |
| strong | partial | Result Table — Design Check | Right panel 'Result Tables' -> Design Check row (editor_figma.html:732). | Rows built from member_checks (figma_explorer.js:300-307) as {member, status, ratio, governing}. Live demo: 87 rows, e.g. {member:'#1', status:'OK', ratio:0.2636, governing:''}. The 'governing' column is empty for EVERY row because /api/v2/analyze emits only { |
| strong | partial | Modal / eigenvalue output | Three surfaces: report §7 고유치/동적해석 (works fully), Result Tables -> Modal Period popup (works but dumps mode-shape JSON), and the 3D animated mode-shap | Solver output is strong and verified live: 9 modes, fundamental_periods {T1_x_s 0.337, T1_y_s 0.5076, T1_rz_s 0.4024} for the demo model, cumulative_participation {x 100.0%, y 100.0%, rz 100.0%, sufficient_90pct true}, ETABS-style direction labels (TRAN-X / TR |
| strong | works | POST /api/v2/recommendations/evaluate (+ GET .../evaluate/{job_id} pol | Backend only on /editor-figma. On /editor-v2: Results panel -> 'Recs' tab -> 'Evaluate Candidates' button (editor_v2.html:1056, onclick=runRecommendat | LIVE END-TO-END. I first ran POST /api/v2/analyze with a 3-story 2x2-bay SS275 config -> 200 in 75.8s, analysis_id 6a9c5f82-8e59-47b4-973b-03c5cc28c329, 8 issues, 24 candidates, summary {"ranking_method":"heuristic_v1","ranking_verified":false,"rag_enabled":fa |
| strong | works | POST /api/v2/recommendations/preview-apply | Backend only on /editor-figma. On /editor-v2: click a candidate card in the Recs tab -> rec-diff-modal (editor_v2.html:1122) showing a before/after ta | LIVE PROBE -> 200. Returned {"applicable":true, "diff":{"operation":"replace_section","changed_member_count":1,"changed_members":[{"element_id":5,"member_label":"column @ story 1 (element 5)","section_from":"H-300x300","section_to":"H-310x310","story":1,"reaso |
| strong | works | Preset dropdown (3-Story / 5-Story / 5-Story Mixed / 10-Story) | Inside the 직접 입력 drawer, top field 'Preset' (select#preset-select, onchange=applyPreset()). | editor_figma.html:148-154; applyPreset() editor3d_figma.js:2191-2257 (fills stories/bays/sections then calls updateManualPreview). Element counts computed offline with the real builder: `StructuralModel.from_building_config` → 10-story preset = 176 nodes / 400 |
| strong | works | Apply section to selected members (section + paired material in one un | Frame Properties window → '선택 부재에 적용 (N)' button (figma_section.js:214-219). Button label carries the live selection count and is disabled at 0 (:273- | figma_section.js:53-65 figmaApplySectionToSelection — pushUndo(), writes e.section (and the paired e.material when set) on every element in getSelectedElementIds(), then refreshEditPreview(). Selection source: editor3d_figma.js:3549-3556 getSelectedElementIds( |
| strong | works | Material picker — Add New Material Property (Region / Type / Standard  | Define Materials window → 'Add New Material…' (figma_material.js:213). | figma_material.js:261-306 openAddNew. Standard dropdown is derived from the live catalog (:265-269), Grade refills on Standard change (:280-289). Region is fixed to 'Korea' and disabled (:271); Concrete / Rebar / Other are present but `option.disabled = true`  |
| strong | works | Material picker — Material Property Data (E / ν / auto-G, weight↔mass  | Define Materials → double-click a row or 'Modify/Show Material…' (figma_material.js:205, :221-224); also the OK path out of Add New (:298-303). | figma_material.js:309-408 openDetail. Live interlocks that actually compute: weight↔mass at :329-336 (G0=9.80665), G=E/2(1+ν) recalculated on E or ν input at :345-350. The Fy table is built from the real per-thickness bands (:363-374). Backend shape verified — |
| strong | works | GET /api/loads/live-preview (KDS 41 12 00 표 3.2-1 live loads, display  | Fetched lazily by the Live load popup (figma_load.js:208-215 ensureLivePreview). | Probe → 200, 11 usages, **every one source="db_lookup"** (no fallback): office 2.5 / retail 5.0 / residential 2.0 / parking 3.0 / storage 6.0 / hospital 2.0 / school 3.0 / assembly 7.0 / corridor 5.0 / mechanical_room 5.0 / roof 1.0 kN/m², each with db_primary |
| strong | works | Load popup — D / Dead (per-story slab thickness + finish DL, auto slab | Ribbon → Load Case group → **D / Dead** (editor_figma.html:78) or menu Load → '고정하중 정의 (Dead)…' (figma_menu.js:223). | figma_load.js:330-393 openDead. Slab self-weight column recomputes live as 24.0 × t on input (:348-351); there is a working '전층 일괄' bulk-apply row (:363-374). **Live probe**: t 0.15→0.30 m and finish 1.0→3.0 kN/m² changed the generated DL floor_area 5.1 → 10.7 |
| strong | works | Load popup — W / Wind (region → V₀, exposure category, importance) | Ribbon → Load Case group → **W / Wind** (editor_figma.html:81) or menu Load → '풍하중 정의 (Wind)…' (figma_menu.js:226). | figma_load.js:522-547 openWind, 4 exposure categories at :516-521. **Live probe**: exposure B → D changed WX top-story force 24.83 → 39.0 kN with EQX unchanged at 32.31 — clean, attributable, single-variable effect. |
| strong | works | DXF export (/api/export/dxf) | Menu File -> 도면 내보내기 (DXF) or Tools -> 도면 내보내기 (figma_menu.js:180, 252 -> editor3d_figma.js:7462-7505 exportToDXF). There is NO button for it in the r | HTTP probe: POST /api/export/dxf with the demo IFC updated_model -> 200, 317,276 bytes, content-type application/dxf. ezdxf read-back: 254 LINE, 91 TEXT, 81 LWPOLYLINE, 80 DIMENSION, 68 layers. View titles present in the file: 'GL PLAN EL. 0.0m', '1F PLAN EL.  |
| strong | works | Model Explorer live tree | Left panel, always visible (editor_figma.html:113 header, :123 #fig-mx-tree). Built by figma_explorer.js:107-220. | Tree source verified against the live demo model: 48 nodes with story ∈ {0,1,2,3} (12 each), 87 elements (36 column / 51 beam), sections H-300x300 x36 and H-400x200 x51, material SS275 x87 — so every branch populates. Structure: Project > Structure > Stories ( |
| strong | works | Analysis Control panel | Right panel, top (editor_figma.html:711-723). Populated by figma_explorer.js:249-272. | Five rows fed from live data: Model Check (overallCheck() — NG if member_check.summary.ng>0 or drift NG, figma_explorer.js:230-248), Model Size ('48 nodes · 87 elems'), Load Cases (7), Combinations (36), Solver ('Linear Static (ELF)' / '+ RSA'). Verified again |
| strong | works | Design-check colour mode on the 3D view (DC Colors) | Visible checkbox in the 3D viewport title bar, 'DC Colors' (editor_figma.html:626-630, id toggle-dc-colors), and menu View / Design -> 설계검토 색상 (DC) (f | toggleDesignCheckColors -> applyDesignCheckColors (editor3d_figma.js:5429-5471) recolours both wireframe cylinder meshes and solid-section meshes from currentResult.member_checks. Banding at editor3d_figma.js:5441-5446: NG -> dc_ng 0xea4335 red, OK with ratio  |
| strong | works | Result-table popup shell (Case/Combo selector, 400-row cap, Esc-to-clo | Shared by all five Result Tables (figma_explorer.js:325-394). | Overlay created lazily, Esc closes (330-333), ✕ button, per-table title with row count, and for case-scoped tables a <select> listing all 7 cases + 36 combos that re-renders on change (357-369). Rows capped at 400 with an explicit '표시 400행 / 전체 N행 — 전체는 Export |
| supporting | needs-credentials | KDS-RAG retrieval index (Voyage embeddings) | Not user-facing directly. Consumed by the chat tool `explain_member_compliance` (mcp-server/core/chat/tools/kds_compliance.py:565-571) and by POST /ap | Index file EXISTS: data/kds_sample_index.jsonl, 147,759 bytes, dated May 28. I parsed it: **6 chunks total**, 1024-dim embeddings — kds_14_31_10 compression/flexure/interaction/shear/tension (5) + kds_41_17_00_drift (1). Selection logic in mcp-server/core/kds_ |
| supporting | partial | Material picker — Define Materials (list window) | Ribbon → Properties group → **M / Material** (editor_figma.html:66), or menu bar Model → '재료 정의 (Materials)…' (figma_menu.js:205). Both call window.fi | figma_material.js:177-185 figmaOpenMaterials, renderList :186-253. The list content is definedNames() = materials used by the model ∪ session-custom ∪ session-'defined' (:67-73) — it is NOT the catalog. With no model loaded, usageCounts() returns {} and the bo |
| supporting | partial | Result Table — Story Drift | Right panel 'Result Tables' -> Story Drift row (editor_figma.html:729) -> figma_explorer.js:335-394 modal popup with a Case/Combo dropdown. | Rows come from case_data[case].story_drifts (figma_explorer.js:289-292). Live demo data: 3 rows for the demo IFC model, columns story / height_m / drift_x / drift_y / drift_resultant / disp_x_max / disp_x_min / disp_y_max / disp_y_min — all scalars, so fmtCell |
| supporting | partial | Result Table — Member Force | Right panel 'Result Tables' -> Member Force row (editor_figma.html:731). | Rows from currentResult.member_forces[case] (figma_explorer.js:297-299), passed through unchanged from the V2 response (editor3d_figma.js:2158). The underlying data is excellent — 43 cases x 87 members, each record {member_id, type, ni, nj, length_m, s:[5 stat |
| supporting | partial | Result Table — Modal Period | Right panel 'Result Tables' -> Modal Period row (editor_figma.html:733). | Rows are modal_analysis.modes (figma_explorer.js:308-311) AFTER _convertModalV2toV1 (editor3d_figma.js:2050-2069), so each row is {mode, period_s, frequency_hz, direction, dominance_pct, mass_participation:{...}, shape:{...}}. fmtCell JSON.stringifies the last |
| supporting | partial | Deformed shape display | Menu ONLY: View -> 변형 형상 or Results -> 변형 형상 (figma_menu.js:202, 244). There is no visible checkbox — verified the id 'toggle-deformed' appears 0 time | The menu path works because figma_menu.js:37-47 ensureCheckbox() creates the missing hidden <input id=toggle-deformed> on demand, flips it, then calls toggleDeformedShape(). That reaches _applyDeformedIfEnabled (editor3d_figma.js:4896-4921) -> applyDeformedSha |
| supporting | partial | Force diagrams on the 3D view (N / V / M) | Menu Results -> 축력도 (N) / 전단력도 (V) / 모멘트도 (M) / 다이어그램 끄기 (figma_menu.js:238-241). No visible buttons — 'diagram-btn-wrap', 'dgm-btn', 'diagram-scale-s | setDiagramMode (editor3d_figma.js:6590-6603) guards its DOM lookups (querySelectorAll('.dgm-btn') returns empty; 'diagram-scale-wrap' behind an if), so the menu path runs without error and calls _buildDiagrams (6657+). That reads currentResult.member_forces[ca |
| supporting | partial | Bottom status bar (quick results) | Always visible strip at the bottom (editor_figma.html:838-861), filled by updateBottomBar / updateBottomBarValues (editor3d_figma.js:4788-4812, 5012-5 | The VALUES are real and correct: #bot-drift-x/#bot-drift-y = envelope max_drift_x/y (5 dp), #bot-disp-x/#bot-disp-y = max_dx_mm/max_dy_mm, #bot-moment = max_moment_kNm, #bot-dc = 'ALL OK' / 'NG' from design_check. The LABELS are leftover Figma mockup text: edi |
| supporting | works | Irregular plan input (비정형 zones: L-shape / T-shape / Setback) | 직접 입력 drawer → Bays section → '비정형' checkbox → 형태 프리셋 select (L자형/T자형/Setback) + per-zone rows + 2D zone plan canvas. | editor_figma.html:169-217; toggleIrregular() editor3d_figma.js:5604, applyZonePreset() 5620-5647, getZonesFromEditor() 5717-5727, config.zones attached at 2311-2313. HTTP probe: POST /api/v2/analyze with the exact L-shape zone payload (zone A 3×2 bays, zone B  |
| supporting | works | IFC import — visitor's own .ifc file | File → 'IFC 가져오기…' → dropzone (click or drag-drop, accept=.ifc) → '업로드 & 파싱'. | editor_figma.html:373-404; initIFCDropzone() editor3d_figma.js:855+, uploadIFC() 1061-1140 (parse → auto-snap → origin normalize → merge nearby nodes → beam intersections → split → connectivity validate). Backend POST /api/v2/parse-ifc main_simple.py:953+ (500 |
| supporting | works | Load-definition popups as input enrichment (D/L/E/W) | Ribbon Load Case group buttons D/L/E/W, or menu Load → 각 하중 정의 (editor_figma.html:78-81; figma_menu.js:227-230). | figma_load.js:550 defines figmaOpenLoads; the popups merge their values into every /api/v2/analyze request by wrapping window.fetch (figma_load.js:12-17, 192), so they apply to the manual, NL, IFC and draw routes alike. Live-load display values come from GET / |
| supporting | works | Section picker — Show Section Properties (A / Ix / Iy / J with J-appro | Frame Section Property Data window → 'Show Section Properties…' button (figma_section.js:350-352). | figma_section.js:475-502 openProps. Renders A_cm2 / Ix_cm4 (I33) / Iy_cm4 (I22) / J_cm4, and when j_source=='approximate' appends an asterisk plus the warning at :495-498. Endpoint field verified across all 738: j_source distribution = **db 499, approximate 21 |
| supporting | works | Member release tool — 6-DOF hexagon dialog (ribbon R) | Ribbon → Properties group → **R / Release** (editor_figma.html:68 setEditMode('release')), or menu Edit → '릴리즈 지정' (figma_menu.js:187) / Node·Element  | figma_edit.js:592 routes the canvas click; figma_tools.js:45-55 handleRelease raycasts to the member, :57-94 showReleaseDialog renders two 6-cell DOF hexagons (i-end / j-end) plus Pin i / Pin j / Pin Both / All Fixed presets (:116-132), :164-180 confirmRelease |
| supporting | works | Support tool — node boundary-condition dialog (ribbon SP) | Ribbon → Boundary group → **SP / Support** (editor_figma.html:72 setEditMode('support')), or menu Edit → '지점 지정' (figma_menu.js:188) / Boundary → '지점  | figma_edit.js:593 routes the click; figma_tools.js:187-197 handleSupport raycasts only SphereGeometry meshes (node spheres exist — editor3d_figma.js:1434/2795/5877), :199-232 showSupportDialog offers Free / Fixed(6-DOF) / Pinned(3-DOF) / Roller X / Roller Y wi |
| supporting | works | Result Table — Base Reaction | Right panel 'Result Tables' -> Base Reaction row (editor_figma.html:730). | Rows from case_data[case].reactions (figma_explorer.js:293-296). Live demo: 12 rows (one per fixed support). Columns node / x_m / y_m / RX_kN / RY_kN / RZ_kN / MX_kNm / MY_kNm / MZ_kNm, all scalars. Sample DL row: {node 1, x 0.0, y 0.0, RX 7.1, RY 5.32, RZ 240 |
| skip | broken | KDS-RAG chatbot — presence on the main demo UI (/editor-figma) | None. There is no FAB, no menu item, no ribbon button. The chat widget is simply not loaded on the page the landing sends visitors to. | webapp/backend/templates/editor_figma.html lines 873-881 list all 9 script tags (editor3d_figma, figma_edit, figma_tools, figma_menu, figma_plan, figma_material, figma_section, figma_load, figma_explorer) — chat_widget.js is absent. Live probe: `curl -s http:/ |
| skip | broken | KDS-RAG chatbot — actual behaviour when reachable (/editor-v2 FAB) | Navigate directly to /editor-v2 (returns HTTP 200, publicly reachable, not linked from the landing) -> purple 💬 FAB bottom-right -> type a message. | LIVE PROBE against 127.0.0.1:8099. POST /api/v2/chat/sessions -> {"session_id":"chat_4fcf58392a96405e","provider":"ollama"}. POST /api/v2/chat/messages with a Korean question returned this exact NDJSON in 4.8s:\n  {"type":"status","message":"thinking","provide |
| skip | broken | Retrofit recommendations — presence on the main demo UI (/editor-figma | None. editor_figma.html has no result-tab strip at all and no recommendation markup. | grep 'rec-\|recPanel\|recommend' webapp/backend/templates/editor_figma.html -> ZERO hits; grep 'rtab' -> ZERO hits. The v2 template has all of it: editor_v2.html:928 (`data-rtab="recommend"` tab button), :1044 rec-issues-list, :1052 rec-candidates-list, :1056  |
| skip | broken | GET /api/v2/recommendations/chat-preview/{preview_id} | Only reachable from the chatbot: the `propose_section_change` tool returns ui_action=open_diff_preview, chat_widget.js:238-259 dispatches it to Editor | LIVE PROBE with a bogus id -> 404 {"detail":"chat preview 'zzz' not found — it may have expired (30 min TTL) or never existed. Ask the chatbot to repeat the section-change request."} — a well-worded error. But the ONLY producer of a valid preview_id is the cha |
| skip | broken | Natural language input (자연어 → 모델·하중) | File → '자연어 입력', or Load → '자연어 입력 (모델·하중 자동 생성)' (figma_menu.js:174, 225). Drawer shows a textarea, three example chips (fillNLExample 0/1/2), and th | UI: editor_figma.html:317-354. Frontend calls POST /api/claude/parse-building (editor3d_figma.js:731). LIVE PROBE against the running server: `POST /api/claude/parse-building {"text":"서울, 5층 오피스, 3×2 경간, 8m"}` → HTTP 200 body `{"success":false,"error":"파싱 오류:  |
| skip | broken | Ribbon 'Model' tab / Model Explorer dock 'Model' tab (as an input entr | Ribbon tab strip 'Model' and the left dock tab 'Model', both onclick=switchInputTab('manual'). | editor_figma.html:50 and 117. switchInputTab (editor3d_figma.js:406-419) only toggles `.active` classes on `.tab-content` divs — which live inside `.legacy-config-ui`, hidden by editor_figma.css:2385-2387, and after the drawer has been built once they live ins |
| skip | broken | Model menu → '보 단면 지정' (global beam section) | Menu bar → Model → '기둥 단면 지정' / '보 단면 지정' (figma_menu.js:209-210). | The menu passes 'beam', but applyGlobalSection (editor3d_figma.js:2443-2454) only branches on 'column' \| 'beam_x' \| 'beam_y', so 'beam' yields an empty `modifications` object that is then handed to _applyV2SectionAndReanalyze — a no-op re-analysis. Both item |
| skip | broken | Dead ribbon buttons adjacent to this area: Plate, Spring, Constraint | Ribbon → Geometry group 'P / Plate' (editor_figma.html:61); Ribbon → Boundary group 'K / Spring' (:73) and 'C / Constraint' (:74). | All three are `<button class="ribbon-command">` with **no onclick attribute** — verified by reading editor_figma.html:61, :73, :74. They are styled identically to the live Material/Section/Support buttons, so they look clickable and do nothing at all (no toast |
| skip | broken | Right-panel results block (envelope table, case selector, DC summary,  | Intended: right panel below Result Tables (editor_figma.html:736-834). Actually: unreachable. | editor_figma.css:2385-2387 sets '.legacy-config-ui, .legacy-property-ui { display: none !important; }' and editor_figma.html:736 is '<div class="panel-body legacy-property-ui">'. Everything inside is therefore invisible: #prop-results (:784), #case-selector-wr |
| skip | needs-credentials | Chat tool registry (inspect / summary / edit / kds groups) | Invoked by the LLM mid-conversation; the widget renders each call as a `[round N] tool(args)` line (chat_widget.js:204-229). | mcp-server/core/chat/tool_registry.py:70-73 — CHAT_TOOLS_ENABLED defaults to "inspect,summary,edit,kds"; .env does not set it, so all four groups are live. Tool implementations exist: mcp-server/core/chat/tools/{inspect.py, kds_compliance.py, section_change.py |
| skip | partial | Chat LLM provider resolution — what a keyless deployment actually reso | Server-side, per chat request. Two distinct degraded modes depending on whether CHAT_LLM_PROVIDER is set. | app/chat_router.py:99-130 `_resolve_llm_provider()`. Path A: CHAT_LLM_PROVIDER=ollama (current .env value) -> line 110-113 constructs OllamaProvider and returns it. mcp-server/core/chat/llm/ollama_provider.py:83-110 __init__ only reads env vars — it NEVER prob |
| skip | partial | Node coordinate input (XYZ dialog) | 3D edit toolbar → 'XYZ' button (showCoordInputPanel) → #coord-input-panel with X/Y/Z + Support fields → 'Create (Enter)'. | editor_figma.html:640-680 (panel markup, and the XYZ button lives inside #addnode-options which is inside #edit-toolbar, `style="display:none"` at line 636). Handlers exist: showCoordInputPanel figma_edit.js:808, createNodeFromCoords figma_edit.js:826. The too |
| skip | partial | Apply material to selected members (bulkApplyMaterial) | Define Materials window → '선택 부재에 적용 (N)' (figma_material.js:233-242, disabled at 0 selection); also the 재료 dropdown in the selection edit panel (figm | figma_material.js:84-98 overrides the engine's own bulkApplyMaterial (editor3d_figma.js:3581) because figma_material.js loads later (editor_figma.html:873 vs :878). It does mutate e.material with pushUndo/refreshEditPreview, and the mutation is serialized (str |
| skip | stub | Project file open/save (.opf-style JSON) | None in /editor-figma. | saveProject() editor3d_figma.js:7507 and loadProject(event) 7549-7600 exist and are complete (restores _v2Model, config and a prior analysis, sets modelSource='Loaded'), but grep over webapp/backend/templates/editor_figma.html and all figma_*.js finds zero ref |
| skip | stub | Section picker — disabled catalog-mutation buttons (Import / Add New / | Right-hand 'Click to:' column of the Frame Properties window. | figma_section.js:203-205, :211-212, :221-222 — all created via disabledBtn() (:141-146), which sets `b.disabled = true` and a title tooltip explaining why (e.g. '사용자 정의 단면 미지원 (해석이 KS 이름으로 DB 조회) — 백로그'). They are visibly greyed, not dead-clickable. |
| skip | stub | Report cover / stamp form (표지·도장란, /api/jobs/{id}/report-cover) | None in /editor-figma. openCoverModal and its inputs (cv-project_name, cv-firm, cv-submit) appear 0 times in the served HTML; all present in editor_v2 | Backend route exists and is documented to re-render from the calc_data.json sidecar without re-analysis (main_simple.py:2147-2160). Client code is fully implemented in editor3d_figma.js:3788-3960 (localStorage persistence, 300 KB logo/seal data-URI upload, quo |
| skip | works | Chat audit endpoint GET /api/v2/chat/audit/{analysis_id} | No UI. Direct HTTP; also fully documented at the open /docs page. | LIVE PROBE, no headers, no token: GET /api/v2/chat/audit/6a9c5f82-... -> 200 {"access_mode":"operator","records":[]}. GET /api/v2/chat/audit/anything?include_quotes=true -> 200, also access_mode "operator". Cause: app/core/auth.py `is_operator_token()` returns |
| skip | works | Open /docs and /openapi.json (exposes both layers) | http://<host>/docs — Swagger UI, no auth. | LIVE: GET /docs -> 200, GET /openapi.json -> 200. The spec enumerates 34 paths, 9 of them in my area: /api/v2/chat/{audit/{analysis_id}, messages, sessions, sessions/{session_id}} and /api/v2/recommendations/{chat-preview/{preview_id}, evaluate, evaluate/{eval |
| skip | works | GET /api/materials/list | Called at page load by editor3d_figma.js:249-268 to fill the manual-input material dropdown (populateMaterialDropdown). | Probe → 200, {"materials":["HSB380","HSB460","HSB690","SM275","SM355","SM420","SM460","SMA275","SMA275C","SMA355","SMA355C","SMA460","SS235","SS275","SS315","SS410","SS450","SS550"]} — 18 names, no fallback flag. Route main_simple.py:414-425. |
| skip | works | Load-definition → analysis wiring (window.fetch wrapper merging popup  | No UI — installed at load time (figma_load.js:566 wrapFetch()). | figma_load.js:187-205 wraps window.fetch and, for any POST to /api/v2/analyze carrying a JSON body with `config`, merges the popup override state (:172-186 mergeConfig) before the request leaves. Because all four routes (manual form, NL, IFC, DRAW) funnel thro |
| skip | works | Deployment credential requirement for this area (materials, sections,  | N/A — infrastructure fact governing all of the above. | No environment credentials are needed. The Supabase URL and anon key are hard-coded at mcp-server/core/simple_beam.py:34-35, and get_supabase() wraps the client in the offline mirror (:44-48). mcp-server/core/kds_cache.py:41-58 mirrors all 15 tables this area  |


### 상세 (broken / needs-credentials 만)

#### KDS-RAG retrieval index (Voyage embeddings)

- **상태**: needs-credentials · 데모가치 supporting
- **진입점**: Not user-facing directly. Consumed by the chat tool `explain_member_compliance` (mcp-server/core/chat/tools/kds_compliance.py:565-571) and by POST /api/v2/recommendations/explain.
- **근거**: Index file EXISTS: data/kds_sample_index.jsonl, 147,759 bytes, dated May 28. I parsed it: **6 chunks total**, 1024-dim embeddings — kds_14_31_10 compression/flexure/interaction/shear/tension (5) + kds_41_17_00_drift (1). Selection logic in mcp-server/core/kds_rag/factory.py:36-55: VoyageKDSRetriever is used only if KDS_RAG_INDEX_PATH exists AND VOYAGE_API_KEY (or VOYAGEAI_API_KEY) is set; otherwise it silently returns NoopKDSRetriever. NoopKDSRetriever (mcp-server/core/kds_rag/retriever.py:190-209) returns zero chunks plus NOOP_WARNING (retriever.py:184-187): 'kds_rag_unavailable: KDS 검색 인덱스가 설정되지 않았습니다. VOYAGE_API_KEY와 KDS_RAG_INDEX_PATH 환경변수를 설정하면 활성화됩니다.' This machine's .env has a real 46-char VOYAGE_API_KEY, which is why retrieval works here.
- **비고**: BLUNT: calling 6 chunks a 'RAG index' oversells it. My live /explain probe returned 5 evidence items — i.e. 5 of the 6 chunks in the entire corpus. The semantic ranking is real (scores 0.664 down to 0.453) but with a corpus this small it will look plausible for any steel-member query and return nothing useful for anything else. Do not let the landing imply a searchable KDS library.

#### KDS-RAG chatbot — presence on the main demo UI (/editor-figma)

- **상태**: broken · 데모가치 skip
- **진입점**: None. There is no FAB, no menu item, no ribbon button. The chat widget is simply not loaded on the page the landing sends visitors to.
- **근거**: webapp/backend/templates/editor_figma.html lines 873-881 list all 9 script tags (editor3d_figma, figma_edit, figma_tools, figma_menu, figma_plan, figma_material, figma_section, figma_load, figma_explorer) — chat_widget.js is absent. Live probe: `curl -s http://127.0.0.1:8099/editor-figma \| grep -c chat_widget` -> 0. Grep over webapp/backend/static/js/figma_*.js and editor_figma.html for ChatWidget\|chat-fab\|EditorV2ChatBridge\|api/v2/chat -> zero hits. By contrast webapp/backend/templates/editor_v2.html:1326-1327 loads chat_widget.css?v=2 + chat_widget.js?v=2.
- **비고**: editor3d_figma.js:7730 DOES define window.EditorV2ChatBridge (the context/diff bridge the widget calls), so the host side was ported but the widget itself never was. Leave it that way for the public demo unless a real LLM is deployed.

#### KDS-RAG chatbot — actual behaviour when reachable (/editor-v2 FAB)

- **상태**: broken · 데모가치 skip
- **진입점**: Navigate directly to /editor-v2 (returns HTTP 200, publicly reachable, not linked from the landing) -> purple 💬 FAB bottom-right -> type a message.
- **근거**: LIVE PROBE against 127.0.0.1:8099. POST /api/v2/chat/sessions -> {"session_id":"chat_4fcf58392a96405e","provider":"ollama"}. POST /api/v2/chat/messages with a Korean question returned this exact NDJSON in 4.8s:\n  {"type":"status","message":"thinking","provider":"ollama"}\n  {"type":"error","message":"Ollama tool-round request failed: ConnectError: All connection attempts failed","code":"tool_request_failure"}\n  {"type":"error","message":"Ollama stream request failed: ConnectError: All connection attempts failed","code":"llm_failure"}\n  {"type":"done","rounds":0,"total_tokens":0,"ms_total":4820}\nchat_widget.js:419-422 renders each error verbatim as `[${event.code}] ${event.message}` in a red bubble. Zero assistant text is produced.
- **비고**: BLUNT: /editor-v2 and /editor-lab both return 200 and are not behind any auth. If the landing ships as-is, a curious visitor who edits the URL lands on a UI whose headline AI feature emits ConnectError. Either deploy an LLM, or make chat_widget.js render a friendly Korean 'AI 도우미는 데모에서 비활성화되어 있습니다' when the first turn errors, or drop /editor-v2 from the deployment.

#### Retrofit recommendations — presence on the main demo UI (/editor-figma)

- **상태**: broken · 데모가치 skip
- **진입점**: None. editor_figma.html has no result-tab strip at all and no recommendation markup.
- **근거**: grep 'rec-\|recPanel\|recommend' webapp/backend/templates/editor_figma.html -> ZERO hits; grep 'rtab' -> ZERO hits. The v2 template has all of it: editor_v2.html:928 (`data-rtab="recommend"` tab button), :1044 rec-issues-list, :1052 rec-candidates-list, :1056 rec-eval-btn, :1122 rec-diff-modal, :1148 rec-explain-modal, :1168 rec-toast-container. The engine code WAS ported — editor3d_figma.js:4248 /evaluate, :4328 /preview-apply, :4748 /explain, :7840 /chat-preview — but every renderer targets DOM ids that do not exist and bails: editor3d_figma.js:4124-4127 `if (!list) return;`, :4142-4144 `if (!list) return;`, :4437-4439 `if (m)` guard in _showRecModal.
- **비고**: This is the biggest missed asset in my area: the backend works (proven below) and the frontend JS is already in editor3d_figma.js. Only the ~40 lines of panel/modal markup were left behind in editor_v2.html. Porting that DOM is the cheapest large win available for the 'landing feels thin' complaint — but it is a build task, and I am not claiming it works until it exists.

#### GET /api/v2/recommendations/chat-preview/{preview_id}

- **상태**: broken · 데모가치 skip
- **진입점**: Only reachable from the chatbot: the `propose_section_change` tool returns ui_action=open_diff_preview, chat_widget.js:238-259 dispatches it to EditorV2ChatBridge.openDiffPreview (editor3d_figma.js:7813+), which fetches this endpoint and opens rec-diff-modal.
- **근거**: LIVE PROBE with a bogus id -> 404 {"detail":"chat preview 'zzz' not found — it may have expired (30 min TTL) or never existed. Ask the chatbot to repeat the section-change request."} — a well-worded error. But the ONLY producer of a valid preview_id is the chat tool, and the chat cannot reach an LLM (probe above), so no preview_id can be minted. Doubly dead on /editor-figma: chat_widget.js is not loaded there, and even if it were, editor3d_figma.js:4437-4439 `_showRecModal` no-ops because rec-diff-modal is absent from editor_figma.html.
- **비고**: This is the single most-coupled feature in the codebase (chat LLM x chat tools x rec modal DOM x preview cache TTL). Every one of those four links is currently broken or missing on the demo path. Do not attempt to demo it.

#### Natural language input (자연어 → 모델·하중)

- **상태**: broken · 데모가치 skip
- **진입점**: File → '자연어 입력', or Load → '자연어 입력 (모델·하중 자동 생성)' (figma_menu.js:174, 225). Drawer shows a textarea, three example chips (fillNLExample 0/1/2), and the 'Claude로 변환' button.
- **근거**: UI: editor_figma.html:317-354. Frontend calls POST /api/claude/parse-building (editor3d_figma.js:731). LIVE PROBE against the running server: `POST /api/claude/parse-building {"text":"서울, 5층 오피스, 3×2 경간, 8m"}` → HTTP 200 body `{"success":false,"error":"파싱 오류: Error code: 400 - {'type':'error','error':{'type':'invalid_request_error','message':'Your credit balance is too low to access the Anthropic API...'}}"}`. The UI's only availability gate is GET /api/claude/status, which probes merely `bool(ANTHROPIC_API_KEY)` (claude_service.py:14-16) — it returned `{"available":true}`, so the warning banner stays hidden and the button stays ENABLED (editor3d_figma.js:699-707). parseBuilding()'s catch then shows `alert('변환 실패: ' + …)` (editor3d_figma.js:745-747), i.e. the visitor gets the raw English Anthropic billing string in a browser alert.
- **비고**: Two cheap fixes if this route matters for the launch: (a) make /api/claude/status do a real 1-token probe (or catch the 400 and show the existing Korean 'API 키 미설정' banner) so the button disables gracefully instead of surfacing billing text; (b) change example chips 0 and 2 to 시군구-qualified strings, or wire the ambiguous_region clarification list into the UI as a picker. Note that /api/building/resolve-config — the deterministic, no-Claude half of the pipeline — is fully working but has ZERO frontend callers (grep over webapp/backend/static/js: only /api/claude/parse-building is called). A dropdown-driven 'NL-lite' demo could be built on it with no API key at all.

#### Ribbon 'Model' tab / Model Explorer dock 'Model' tab (as an input entry point)

- **상태**: broken · 데모가치 skip
- **진입점**: Ribbon tab strip 'Model' and the left dock tab 'Model', both onclick=switchInputTab('manual').
- **근거**: editor_figma.html:50 and 117. switchInputTab (editor3d_figma.js:406-419) only toggles `.active` classes on `.tab-content` divs — which live inside `.legacy-config-ui`, hidden by editor_figma.css:2385-2387, and after the drawer has been built once they live inside the drawer overlay, which is revealed only by openInputDrawer()'s `.open` class (figma_menu.js:133-141). Neither click ever opens the drawer.
- **비고**: Cheapest high-value fix in my area: point these two onclicks at the same drawer opener the File menu uses. Code-path evidence, not browser-verified.

#### Model menu → '보 단면 지정' (global beam section)

- **상태**: broken · 데모가치 skip
- **진입점**: Menu bar → Model → '기둥 단면 지정' / '보 단면 지정' (figma_menu.js:209-210).
- **근거**: The menu passes 'beam', but applyGlobalSection (editor3d_figma.js:2443-2454) only branches on 'column' \| 'beam_x' \| 'beam_y', so 'beam' yields an empty `modifications` object that is then handed to _applyV2SectionAndReanalyze — a no-op re-analysis. Both items also early-return with alert('먼저 해석을 실행해주세요.') when currentJobId is null (2444).
- **비고**: Latent dead path nearby: the non-V2 fallback in applyGlobalSection PATCHes /api/building/{job_id}, but the live OpenAPI (probed) registers only GET for that path, so that branch would 405. It is currently unreachable because /api/v2/analyze always returns updated_model and sets _v2Model.

#### Dead ribbon buttons adjacent to this area: Plate, Spring, Constraint

- **상태**: broken · 데모가치 skip
- **진입점**: Ribbon → Geometry group 'P / Plate' (editor_figma.html:61); Ribbon → Boundary group 'K / Spring' (:73) and 'C / Constraint' (:74).
- **근거**: All three are `<button class="ribbon-command">` with **no onclick attribute** — verified by reading editor_figma.html:61, :73, :74. They are styled identically to the live Material/Section/Support buttons, so they look clickable and do nothing at all (no toast, no status message). By contrast the corresponding menu-bar entries are honest: figma_menu.js:221-222 registers 스프링 and 구속조건 via soon() with the reason '엔진 훅 없음 (백로그)'.
- **비고**: Highest-value cheap fix in my area for a public landing: Spring and Constraint sit in the same 3-button Boundary group as the working Support button, so a visitor exploring boundary conditions has a 2-in-3 chance of hitting a dead button first. Either wire them to the same soon() toast the menu uses, or grey them out.

#### Right-panel results block (envelope table, case selector, DC summary, LLM narrative, ELF vs RSA, report link, per-member design check)

- **상태**: broken · 데모가치 skip
- **진입점**: Intended: right panel below Result Tables (editor_figma.html:736-834). Actually: unreachable.
- **근거**: editor_figma.css:2385-2387 sets '.legacy-config-ui, .legacy-property-ui { display: none !important; }' and editor_figma.html:736 is '<div class="panel-body legacy-property-ui">'. Everything inside is therefore invisible: #prop-results (:784), #case-selector-wrap (:788), #results-table (:793), #elf-rsa-comparison (:796), #modal-section (:799), #dc-summary (:812), #interp-summary (:820), #report-link (:826), plus #prop-member / #prop-dc (per-member section, material, length and design-check banner). updateResultsPanel (editor3d_figma.js:3699-3779) faithfully fills all of them — it sets panel.style.display='block' (3701), writes the DC banner + Members OK/NG + Max Ratio (3731-3753), writes interpretation.summary_ko into #interp-text (3757-3760) and sets reportDiv.style.display='block' (3765) — but the ancestor's !important wins in every case. Additionally switchResultTab('dc') at 3778 is a no-op: 'rtab', 'rtab-btn', 'rtab-content' all appear 0 times in the served HTML. And renderRecommendationsPanel (4086) targets #rec-eval-wrap / #rec-issues-list / #rec-issues-count (4116, 4124, 4125), none of which exist in editor_figma.html (all 3 exist in editor_v2.html).
- **비고**: This is the biggest single loss in the post-analysis surface, and it is what makes the demo 'feel thin'. Casualties include the Korean LLM narrative — verified genuinely generated (narration_meta {llm_used:true, model:'claude-opus-4-8', applied_fields:[summary_ko, summary_en, diagnosis_narrative_ko, diagnosis_narrative_en]}) and genuinely good prose that names V=Cs·W=0.1077x1958=211 kN, SDC C, T1=1.12 s vs Ta=0.51 s, 4 NG columns at max 1.18, and recommends H-300x300 -> H-350x350. That text is currently visible ONLY if the visitor opens the full report. The Figma shell does replace parts of this block (Analysis Control + Result Tables cover the envelope, DC counts and report link), so the real gaps are: the LLM narrative, the case/combo selector (which also silently pins the deformed shape and diagrams to 1.4DL), the per-member design-check readout on click, and the whole retrofit-recommendation UI.

#### Chat tool registry (inspect / summary / edit / kds groups)

- **상태**: needs-credentials · 데모가치 skip
- **진입점**: Invoked by the LLM mid-conversation; the widget renders each call as a `[round N] tool(args)` line (chat_widget.js:204-229).
- **근거**: mcp-server/core/chat/tool_registry.py:70-73 — CHAT_TOOLS_ENABLED defaults to "inspect,summary,edit,kds"; .env does not set it, so all four groups are live. Tool implementations exist: mcp-server/core/chat/tools/{inspect.py, kds_compliance.py, section_change.py}. I could NOT verify any tool end-to-end because the orchestrator only reaches them via OllamaProvider.request_tool_call, and the daemon is unreachable (probe above returned rounds=0, so zero tools were ever invoked).
- **비고**: The `edit` group includes propose_section_change, which stages a model mutation. That is a lot of authority to hand an unauthenticated public visitor's LLM turn. If the chat is ever enabled publicly, set CHAT_TOOLS_ENABLED=inspect,summary,kds and drop `edit`.

