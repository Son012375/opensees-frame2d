# 3D Building Editor UI - Review Document

> **목적:** GPT에게 UI/UX 검토를 요청하기 위한 통합 문서
> **프로젝트:** OpenSees 구조해석 웹 플랫폼
> **페이지:** `http://localhost:8001/editor`

---

## 1. 시스템 개요

사용자가 건물 구조물을 정의하고 3D로 시각화하며, OpenSees 기반 구조해석을 수행하는 웹 에디터.

### 입력 방식 (3개 탭)
| 탭 | 설명 |
|----|------|
| **직접 입력 (Manual)** | 폼 기반: 층수/높이/용도, 경간, 단면, 재료, 지역, 중요도 |
| **자연어 (NL)** | Claude API로 한국어 자연어 → 구조해석 Config 변환 |
| **IFC** | 3단계 위자드: Upload → Geometry Preview → Config → 해석 |

### 출력
- 3D 뷰어 (Three.js): 기둥(파랑), X보(초록), Y보(노랑) 렌더링
- 우측 패널: 부재 클릭 시 속성 표시, 해석 결과 요약, Design Check (OK/NG)
- 하단 바: Max Drift, Max Displacement, Design Check 요약
- HTML Report: 새 탭에서 상세 리포트 열기

### 테마
- **라이트 모드 (기본)**: 흰색 배경, Midas Gen 스타일
- **다크 모드**: 토글 버튼으로 전환, localStorage 저장

---

## 2. 레이아웃 구조

```
┌─────────────── Top Bar (42px) ──────────────────────────┐
│ [Logo] | 3D Building Editor    [Status]    [Theme] [Home]│
├──────────┬──────────────────────────┬───────────────────┤
│          │                          │                   │
│  Left    │       Center             │     Right         │
│  Panel   │       3D Viewer          │     Properties    │
│  280px   │       (Three.js)         │     280px         │
│          │                          │                   │
│ [Manual] │                          │ - Member Props    │
│ [NL]     │                          │ - Design Check    │
│ [IFC]    │                          │ - Results Summary │
│          │                          │ - Interpretation  │
│          │                          │ - [HTML Report]   │
├──────────┴──────────────────────────┴───────────────────┤
│ Bottom Bar: Drift X | Drift Y | Disp X | Disp Y | DC   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. HTML 구조 (editor.html, 449줄)

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>3D Building Editor - OpenSees</title>
    <script src="three.js r128 + OrbitControls"></script>
    <link rel="stylesheet" href="/static/css/editor.css">
</head>
<body>
    <!-- Top Bar -->
    <div class="editor-topbar">
        <div class="topbar-left">
            <a href="/" class="logo">OpenSees</a>
            <span class="separator">|</span>
            <span class="page-title">3D Building Editor</span>
        </div>
        <div class="topbar-center" id="status-bar">
            <span class="status-text">Ready</span>
        </div>
        <div class="topbar-right">
            <button class="theme-toggle" id="theme-toggle"
                    onclick="toggleTheme()" title="테마 전환">&#x263E;</button>
            <a href="/" class="nav-link">Home</a>
        </div>
    </div>

    <div class="editor-layout">
        <!-- ═══ Left Panel: Building Config ═══ -->
        <div class="panel panel-left">
            <!-- 3 Tabs: Manual / NL / IFC -->
            <div class="input-tabs">
                <button class="input-tab active" onclick="switchInputTab('manual')">직접 입력</button>
                <button class="input-tab" onclick="switchInputTab('nl')">자연어</button>
                <button class="input-tab" onclick="switchInputTab('ifc')">IFC</button>
            </div>

            <!-- ── Manual Tab ── -->
            <div id="tab-manual" class="tab-content active">
                <!-- Preset selector (3/5/10-story) -->
                <!-- Per-story editor: height(m) + usage dropdown per floor -->
                <!-- Bays X/Y (comma-separated widths in m) -->
                <!-- Section dropdowns: Column, Beam X, Beam Y (+ 전체 적용 buttons) -->
                <!-- Material, Supports, Region, Importance -->
                <!-- [Analyze] button → runAnalysis() -->
            </div>

            <!-- ── NL Tab ── -->
            <div id="tab-nl" class="tab-content">
                <!-- Claude API warning (shown if key not set) -->
                <!-- Textarea: "서울 강남, 1층 근생, 2~5층 오피스" -->
                <!-- Example prompts -->
                <!-- [Claude로 변환] → parseBuilding() -->
                <!-- Resolution report: warnings, usage mapping -->
                <!-- [직접 입력으로 수정] / [바로 해석] buttons -->
            </div>

            <!-- ── IFC Tab (3-Step Wizard) ── -->
            <div id="tab-ifc" class="tab-content">
                <!-- Step Indicator: ①─②─③ circles with labels -->

                <!-- Step 1: Upload -->
                <!--   Dropzone (drag & drop .ifc) -->
                <!--   [업로드 & 파싱] → uploadIFC() → goToIFCStep(2) -->

                <!-- Step 2: Geometry Preview -->
                <!--   건물 개요 (파일명, 기둥수, 벽수) -->
                <!--   층별 높이 편집 (number inputs) -->
                <!--   X/Y 경간 편집 (text inputs, comma-separated) -->
                <!--   감지 단면/재료 (read-only) -->
                <!--   IFC warnings list -->
                <!--   3D wireframe preview in viewer -->
                <!--   [← 이전] [다음: 부가정보 →] -->

                <!-- Step 3: Supplementary Config -->
                <!--   NL 보조 입력 (optional textarea + [NL 적용]) -->
                <!--   층별 용도 dropdowns (IFC에서 층 수 결정) -->
                <!--   지역, 중요도, 단면, 재료, 지지조건 -->
                <!--   [← 이전] [해석 실행] → runAnalysisFromIFCWizard() -->
            </div>
        </div>

        <!-- ═══ Center Panel: 3D Viewer ═══ -->
        <div class="panel panel-center">
            <div class="panel-header">
                <h3>3D Viewer</h3>
                <div class="viewer-controls">
                    [Reset View] [Wireframe] [Axes] [DC Colors checkbox]
                </div>
            </div>
            <div id="viewer-container">
                <canvas id="three-canvas"></canvas>
                <div id="preview-badge" style="display:none;">PREVIEW</div>
                <div id="loading-overlay" style="display:none;">...</div>
            </div>
        </div>

        <!-- ═══ Right Panel: Properties ═══ -->
        <div class="panel panel-right">
            <!-- Empty state: "Click a member..." -->

            <!-- Selected member: type badge, section, material, nodes, length -->
            <!-- Design Check per-member: ratio, governing combo -->
            <!-- [Modify Section] dropdown + [Apply & Re-analyze] -->

            <!-- Results Summary: envelope values -->
            <!--   Max Drift X/Y, Max Disp X/Y/Z, Max Moment, Max Axial, Max Shear -->
            <!--   Modal Analysis: T1, T2, T3 -->

            <!-- Design Check Summary: OK/NG banner, drift/member status -->
            <!-- Interpretation: 한국어 요약 텍스트 -->
            <!-- [HTML Report] button → opens in new tab -->
        </div>
    </div>

    <!-- Bottom Bar -->
    <div class="editor-bottombar" id="bottom-bar" style="display:none;">
        Max Drift X | Max Drift Y | Max Disp X | Max Disp Y | Max Moment | Design Check
    </div>
</body>
</html>
```

---

## 4. CSS 테마 시스템 (editor.css, 923줄)

CSS Custom Properties 기반 라이트/다크 테마 시스템.

### 변수 구조 (60개+)
```css
:root {
    /* ═══ Light Theme (Default) ═══ */
    --bg-body: #f0f2f5;          /* 전체 배경 */
    --bg-panel: #ffffff;          /* 패널 배경 */
    --bg-panel-header: #e8ecf0;   /* 패널 헤더 */
    --bg-topbar: #ffffff;
    --bg-input: #f7f8fa;          /* 입력 필드 배경 */
    --bg-viewer: #dfe3e8;         /* 3D 뷰어 배경 */

    --border-main: #d0d4da;
    --border-input: #c0c4ca;
    --border-focus: #1a73e8;      /* 포커스 시 파란 테두리 */

    --text-primary: #202124;
    --text-secondary: #5f6368;
    --text-tertiary: #80868b;

    --accent: #1a73e8;            /* 주 강조색 (파랑) */
    --accent-hover: #1557b0;
    --accent-text: #1a73e8;

    --color-success: #188038;     /* 초록 */
    --color-warning: #e37400;     /* 주황 */
    --color-error: #d93025;       /* 빨강 */

    /* Design Check 배너 */
    --dc-ok-bg: #e6f4ea;   --dc-ok-text: #188038;
    --dc-ng-bg: #fce8e6;   --dc-ng-text: #d93025;

    /* Member type badges */
    --badge-column-bg: #e8f0fe;   --badge-column-text: #1a73e8;
    --badge-beam-x-bg: #e6f4ea;   --badge-beam-x-text: #188038;
    --badge-beam-y-bg: #fef7e0;   --badge-beam-y-text: #e37400;
}

[data-theme="dark"] {
    /* ═══ Dark Theme ═══ */
    --bg-body: #1a1a2e;
    --bg-panel: #16213e;
    --bg-panel-header: #0f3460;
    --bg-input: #1a1a2e;
    --bg-viewer: #0d1117;

    --border-main: #0f3460;
    --border-focus: #4fc3f7;

    --text-primary: #e0e0e0;
    --accent: #4fc3f7;
    --color-success: #69f0ae;
    --color-warning: #ffab40;
    --color-error: #ff5252;
    /* ... (모든 변수 다크 오버라이드) */
}
```

### 주요 컴포넌트 스타일
| 컴포넌트 | 클래스 | 설명 |
|----------|--------|------|
| 상단바 | `.editor-topbar` | 42px 높이, 로고+상태+테마토글 |
| 패널 | `.panel`, `.panel-left/center/right` | 좌 280px, 중앙 flex, 우 280px |
| 입력 탭 | `.input-tabs`, `.input-tab` | 3개 탭 (active시 하단 파란 테두리) |
| 스토리 편집기 | `.story-row` | 층고(number) + 용도(select) + 삭제 |
| IFC 위자드 | `.ifc-wizard-steps`, `.ifc-step-circle` | 3단계 인디케이터 (active/completed) |
| IFC 형상편집 | `.ifc-geo-row`, `.ifc-geo-section` | 층고/경간 편집 행 |
| 버튼 | `.btn-primary`, `.btn-secondary` | 파란 주 버튼 / 회색 보조 버튼 |
| Design Check | `.dc-banner.ok/.ng` | 초록 OK / 빨강 NG 배너 |
| 테마 토글 | `.theme-toggle` | 32px 둥근 버튼 (달/해 아이콘) |

---

## 5. JavaScript 로직 (editor3d.js, 1641줄)

### 5.1 상태 변수

```javascript
// 3D Scene
let scene, camera, renderer, controls;
let memberMeshes = [];      // 해석 결과 부재 메시
let nodeMeshes = [];
let selectedMesh = null;

// Analysis
let currentJobId = null;
let currentResult = null;
let sectionsList = {};      // API에서 로드한 단면 목록
let materialsList = [];

// NL Tab
let resolvedConfig = null;
let claudeAvailable = false;

// IFC Wizard
let ifcParsedData = null;   // 서버에서 파싱된 IFC 데이터
let ifcSelectedFile = null;
let ifcWizardStep = 1;
let previewMeshes = [];     // 미리보기 와이어프레임
let ifcEditedData = null;   // 사용자 편집 반영 데이터

// Theme
const SCENE_BG = { light: 0xdfe3e8, dark: 0x0d1117 };
```

### 5.2 핵심 함수 목록

#### 초기화
| 함수 | 설명 |
|------|------|
| `initTheme()` | localStorage에서 테마 복원 |
| `initThreeJS()` | Scene, Camera, Renderer, Controls, Axes, Grid 초기화 |
| `loadSectionsAndMaterials()` | `/api/sections/list`, `/api/materials/list` fetch |
| `initIFCDropzone()` | 드래그&드롭 이벤트 바인딩 |

#### 테마
| 함수 | 설명 |
|------|------|
| `toggleTheme()` | 라이트↔다크 전환, localStorage 저장 |
| `updateSceneBg()` | Three.js 배경색 + 그리드 색상 갱신 |

#### Manual 탭
| 함수 | 설명 |
|------|------|
| `buildStoryEditorUI(stories)` | 층별 높이+용도 편집 UI 동적 생성 |
| `addStory()` / `removeStory(idx)` | 층 추가/제거 |
| `applyPreset()` | 3/5/10층 프리셋 적용 |
| `runAnalysis(configOverride=null)` | POST `/api/building/analyze` → buildScene + updateResults |

#### NL 탭
| 함수 | 설명 |
|------|------|
| `parseBuilding()` | POST `/api/claude/parse-building` → 변환 리포트 표시 |
| `applyResolvedConfig()` | 변환 결과 → Manual 폼에 반영 |
| `runAnalysisFromNL()` | Manual 폼 채운 후 runAnalysis() |

#### IFC 위자드
| 함수 | 설명 |
|------|------|
| `handleIFCFile(file)` | 파일 선택 시 UI 업데이트 |
| `uploadIFC()` | POST `/api/building/parse-ifc` → goToIFCStep(2) |
| `goToIFCStep(step)` | 위자드 스텝 전환 + 스텝별 초기화 |
| `buildIFCGeometrySummary(data)` | Step 2: 층고/경간 편집 UI 생성 |
| `buildPreviewScene(data)` | Step 2: 3D 와이어프레임 렌더링 (OpenSees 없이) |
| `updatePreviewFromEdits()` | 편집 시 와이어프레임 실시간 갱신 |
| `buildIFCSupplementaryForm()` | Step 3: 용도/지역/단면 폼 생성 |
| `applyNLToIFCForm()` | Step 3: NL 보조 입력 → 용도/지역/재료 자동 채움 |
| `runAnalysisFromIFCWizard()` | Step 2 형상 + Step 3 설정 → runAnalysis(config) |
| `clearPreviewScene()` | 와이어프레임 제거 (해석 전 호출) |

#### 3D 렌더링
| 함수 | 설명 |
|------|------|
| `buildScene(result)` | 해석 결과 → CylinderGeometry 기둥/보 렌더링 |
| `onCanvasClick(e)` | Raycaster로 부재 선택 → 속성 패널 업데이트 |
| `resetCamera()` | 카메라 위치 초기화 |
| `toggleWireframe()` / `toggleAxes()` | 와이어프레임/축 토글 |
| `toggleDesignCheckColors()` | OK=초록, NG=빨강 부재 색상 |

#### 결과 표시
| 함수 | 설명 |
|------|------|
| `updateResultsPanel(result)` | 우측 패널: envelope, modal, DC, interpretation |
| `updateBottomBar(result)` | 하단 바: 주요 수치 표시 |
| `showMemberProperties(meshData)` | 선택 부재 속성 + DC ratio |

### 5.3 IFC 위자드 데이터 흐름

```
Step 1: upload .ifc
  → POST /api/building/parse-ifc
  → ifcParsedData = {stories, bays_x, bays_y, grid_x, grid_y,
                      detected_sections, detected_material, warnings, summary}
  → ifcEditedData = deep copy of geometry

Step 2: review + edit geometry
  → buildPreviewScene(ifcParsedData) → THREE.Line 와이어프레임
    - 기둥: 파랑 | 보X: 초록 | 보Y: 노랑 | 절점: 회색 구 | 지지: 주황 콘
  → user edits heights/bays → updatePreviewFromEdits() → 와이어프레임 갱신
  → "PREVIEW" 뱃지 표시

Step 3: supplementary config
  → (선택) NL 보조: "1층 근생, 재료 SS275, 해운대구"
    → POST /api/claude/parse-building (IFC 형상 컨텍스트 자동 추가)
    → intent.material → 재료 드롭다운 자동 설정
    → resolved.config.stories → 용도 드롭다운 자동 설정
  → [해석 실행] → merge geometry + config → runAnalysis(config)
  → clearPreviewScene() → buildScene(result) → 해석 결과 3D 렌더링
```

### 5.4 runAnalysis(configOverride) 패턴

```javascript
async function runAnalysis(configOverride = null) {
    // configOverride 있으면 직접 사용, 없으면 Manual 폼에서 읽기
    let config = configOverride || buildConfigFromForm();

    const response = await fetch('/api/building/analyze', {
        method: 'POST',
        body: JSON.stringify({ config }),
    });
    const result = await response.json();

    buildScene(result);          // 3D 렌더링
    updateResultsPanel(result);  // 결과 패널
    updateBottomBar(result);     // 하단 바
    // result.report_url → [HTML Report] 버튼 표시
}
```

---

## 6. Backend API (main_simple.py, 956줄)

### 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/editor` | Editor HTML 페이지 |
| GET | `/api/sections/list` | 단면 목록 (H형강 등) |
| GET | `/api/materials/list` | 재료 목록 (SS275, SS400, ...) |
| GET | `/api/claude/status` | Claude API 키 설정 여부 |
| POST | `/api/claude/parse-building` | 자연어 → BuildingIntent → Config |
| POST | `/api/building/analyze` | 건물 3D 해석 (config → OpenSees) |
| POST | `/api/building/parse-ifc` | IFC 파일 업로드 → 형상 파싱 |
| PATCH | `/api/building/{job_id}` | 부재 단면 변경 후 재해석 |
| GET | `/api/jobs/{job_id}/report` | HTML 리포트 파일 서빙 |

### analyze 응답 구조

```json
{
    "job_id": "uuid",
    "status": "success",
    "viewer": {
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, ...],
        "elements": [{"id": 1, "ni": 1, "nj": 2, "type": "column", "section": "H-300x300"}, ...],
        "stories": [4.0, 4.0],
        "bays_x": [4.5, 6.5, 6.0],
        "bays_y": [4.0, 6.0, 5.0]
    },
    "envelope": {
        "max_drift_x": 0.00123,
        "max_drift_y": 0.00098,
        "max_dx_mm": 5.23,
        "max_dy_mm": 4.12,
        "max_moment_kNm": 234.5,
        "max_axial_kN": 1523.4,
        "max_shear_kN": 89.3
    },
    "design_check": {
        "drift_check": {"status": "OK"},
        "member_check": {"status": "OK", "summary": {"ok": 78, "ng": 2, "max_interaction_ratio": 0.87}}
    },
    "interpretation": {
        "summary_ko": "구조 안전성 양호. 최대 층간변위 0.12% (허용치 2.0% 이내)..."
    },
    "modal_analysis": {
        "periods": [0.523, 0.498, 0.312],
        "frequencies": [1.91, 2.01, 3.21]
    },
    "member_checks": {"1": {"status": "OK", "interaction_ratio": 0.45}, ...},
    "report_url": "/api/jobs/{job_id}/report"
}
```

### parse-ifc 응답 구조

```json
{
    "success": true,
    "stories": [
        {"name": "1F", "height": 4.0, "usage": "office", "slab_thickness_mm": null}
    ],
    "bays_x": [4.5, 6.5, 6.0],
    "bays_y": [4.0, 6.0, 5.0],
    "grid_x": [0.0, 4.5, 11.0, 17.0],
    "grid_y": [0.0, 4.0, 10.0, 15.0],
    "detected_sections": {"column": "H-400x408", "beam": "H-250x250"},
    "detected_material": null,
    "grid_source": "column",
    "num_columns": 32,
    "warnings": ["..."],
    "summary": {"num_stories": 2, "num_bays_x": 3, "num_bays_y": 3, "total_height": 8.0}
}
```

---

## 7. 주요 UX 흐름

### Flow A: Manual 입력 → 해석
```
[프리셋 선택] → [층별 높이/용도 편집] → [경간 설정] → [단면/재료 선택]
→ [Analyze] → 3D 렌더링 + 결과 표시 + [HTML Report]
```

### Flow B: 자연어 입력 → 해석
```
[한국어 텍스트 입력] → [Claude로 변환] → [변환 결과 확인]
→ [바로 해석] 또는 [직접 입력으로 수정 → Analyze]
```

### Flow C: IFC 업로드 → 해석
```
[Step 1: IFC 파일 드래그&드롭] → [업로드 & 파싱]
→ [Step 2: 3D 와이어프레임 확인 + 층고/경간 수정] → [다음]
→ [Step 3: (NL 보조) 용도/지역/중요도/재료 설정] → [해석 실행]
→ 3D 결과 렌더링 + 결과 표시 + [HTML Report]
```

### Flow D: 부재 수정 → 재해석
```
[3D 뷰에서 부재 클릭] → [우측 패널에서 단면 변경]
→ [Apply & Re-analyze] → 결과 갱신
```

---

## 8. 검토 요청 사항

다음 관점에서 UI/UX를 검토해 주세요:

1. **레이아웃/구조**: 3패널 레이아웃의 적절성, 패널 크기 비율
2. **입력 흐름**: Manual/NL/IFC 3개 탭의 사용자 경험, 전환 시 상태 유지
3. **IFC 위자드**: 3단계 워크플로우의 직관성, 스텝 인디케이터 디자인
4. **테마 시스템**: 라이트/다크 전환의 일관성, 색상 선택
5. **결과 표시**: 해석 결과의 가독성, 정보 우선순위
6. **접근성**: 폰트 크기, 대비, 클릭 영역
7. **에러 처리**: 오류 메시지의 명확성, 사용자 가이드
8. **개선 제안**: 추가 기능, UX 개선점

---

## 9. 기술 스택

- **Frontend**: Vanilla JS, Three.js r128, CSS Custom Properties
- **Backend**: FastAPI (Python), OpenSeesPy (구조해석 엔진)
- **3D**: Three.js OrbitControls, Raycaster, CylinderGeometry
- **외부 API**: Claude API (자연어 파싱), Supabase (KDS 하중 DB)
- **설계 기준**: KDS 41 (한국건설기준), AISC 360 (부재강도)
