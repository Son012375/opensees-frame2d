/**
 * 3D Building Editor — Phase A + UI Improvements
 * Three.js viewer + Raycaster selection + Property panel + Re-analysis
 * Input Mode Tabs (Manual / NL / IFC) + Per-story usage editing
 */

// ─── State ────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let raycaster, mouse;
let memberMeshes = [];      // { mesh, elementData }
let nodeMeshes = [];
let selectedMesh = null;
let currentJobId = null;
let currentResult = null;
let sectionsList = {};      // {type: [names]}
let materialsList = [];
let showDCColors = false;
let axesHelper = null;
let gridHelper = null;
let modelSource = '';  // 'Manual' | 'NL' | 'IFC'

// Colors
const COLORS = {
    column:   0x4285f4,  // blue
    beam_x:   0x34a853,  // green
    beam_y:   0xfbbc04,  // orange/yellow
    selected: 0xff4081,  // pink
    node:     0x888888,
    dc_ok:    0x34a853,
    dc_ng:    0xea4335,
    dc_marginal: 0xfbbc04,
    ground:   0x1a1a2e,
};

// Usage options for per-story dropdown
const USAGE_OPTIONS = [
    { value: 'office',          label: '사무실 (Office)' },
    { value: 'retail',          label: '소매점 (Retail)' },
    { value: 'residential',     label: '주거 (Residential)' },
    { value: 'parking',         label: '주차장 (Parking)' },
    { value: 'storage',         label: '창고 (Storage)' },
    { value: 'hospital',        label: '병원 (Hospital)' },
    { value: 'school',          label: '학교 (School)' },
    { value: 'assembly',        label: '집회 (Assembly)' },
    { value: 'corridor',        label: '복도 (Corridor)' },
    { value: 'mechanical_room', label: '기계실 (Mech.)' },
    { value: 'roof',            label: '옥상 (Roof)' },
];

let resolvedConfig = null;  // NL resolved config (for "바로 해석")
let claudeAvailable = false;

// ─── Theme ────────────────────────────────────────────────────────────────
const SCENE_BG = { light: 0xdfe3e8, dark: 0x0d1117 };

function initTheme() {
    const saved = localStorage.getItem('editor-theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    updateThemeIcon();
}

function toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('editor-theme', 'light');
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('editor-theme', 'dark');
    }
    updateThemeIcon();
    updateSceneBg();
}

function updateThemeIcon() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.innerHTML = isDark ? '&#x2600;' : '&#x263E;';  // sun / moon
    btn.title = isDark ? '라이트 모드로 전환' : '다크 모드로 전환';
}

function updateSceneBg() {
    if (!scene) return;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    scene.background = new THREE.Color(isDark ? SCENE_BG.dark : SCENE_BG.light);
    // Update grid colors
    if (gridHelper) {
        scene.remove(gridHelper);
        gridHelper.geometry.dispose();
        gridHelper.material.forEach ? gridHelper.material.forEach(m => m.dispose()) : gridHelper.material.dispose();
        const g1 = isDark ? 0x334455 : 0xb0b8c0;
        const g2 = isDark ? 0x222233 : 0xc8d0d8;
        const pos = gridHelper.position.clone();
        gridHelper = new THREE.GridHelper(60, 30, g1, g2);
        gridHelper.position.copy(pos);
        scene.add(gridHelper);
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initThreeJS();
    loadSectionsAndMaterials();
    applyPreset();  // load default preset
    checkClaudeStatus();
    initIFCDropzone();
    animate();
});

function initThreeJS() {
    const container = document.getElementById('viewer-container');
    const canvas = document.getElementById('three-canvas');
    const w = container.clientWidth;
    const h = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    scene.background = new THREE.Color(isDark ? SCENE_BG.dark : SCENE_BG.light);

    // Camera
    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    camera.position.set(30, 30, 25);
    camera.lookAt(0, 0, 5);

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 5);
    controls.update();

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(20, 30, 25);
    scene.add(directional);

    // Axes — thick lines + labels
    axesHelper = new THREE.Group();
    const axisLen = 8;
    const axisDefs = [
        { dir: [1,0,0], color: 0xff3333, label: 'X' },
        { dir: [0,1,0], color: 0x33cc33, label: 'Z' },  // Three Y-up = structural Z
        { dir: [0,0,1], color: 0x3377ff, label: 'Y' },   // Three Z = structural Y (neg)
    ];
    axisDefs.forEach(({ dir, color, label }) => {
        // Shaft (cylinder)
        const shaft = new THREE.Mesh(
            new THREE.CylinderGeometry(0.06, 0.06, axisLen, 8),
            new THREE.MeshBasicMaterial({ color })
        );
        // Orient cylinder along axis
        if (dir[0]) { shaft.rotation.z = -Math.PI / 2; shaft.position.x = axisLen / 2; }
        else if (dir[2]) { shaft.rotation.x = Math.PI / 2; shaft.position.z = axisLen / 2; }
        else { shaft.position.y = axisLen / 2; }
        axesHelper.add(shaft);

        // Cone head
        const cone = new THREE.Mesh(
            new THREE.ConeGeometry(0.18, 0.5, 8),
            new THREE.MeshBasicMaterial({ color })
        );
        if (dir[0]) { cone.rotation.z = -Math.PI / 2; cone.position.x = axisLen + 0.25; }
        else if (dir[2]) { cone.rotation.x = Math.PI / 2; cone.position.z = axisLen + 0.25; }
        else { cone.position.y = axisLen + 0.25; }
        axesHelper.add(cone);

        // Label (sprite)
        const canvas = document.createElement('canvas');
        canvas.width = 64; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
        ctx.font = 'bold 48px sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(label, 32, 32);
        const tex = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex }));
        sprite.scale.set(1.2, 1.2, 1);
        if (dir[0]) sprite.position.set(axisLen + 1.2, 0, 0);
        else if (dir[2]) sprite.position.set(0, 0, axisLen + 1.2);
        else sprite.position.set(0, axisLen + 1.2, 0);
        axesHelper.add(sprite);
    });
    scene.add(axesHelper);

    // Ground grid
    const gridColor1 = isDark ? 0x334455 : 0xb0b8c0;
    const gridColor2 = isDark ? 0x222233 : 0xc8d0d8;
    gridHelper = new THREE.GridHelper(60, 30, gridColor1, gridColor2);
    gridHelper.rotation.x = 0; // Y-up in Three.js
    scene.add(gridHelper);

    // Raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Events
    canvas.addEventListener('click', onCanvasClick, false);
    window.addEventListener('resize', onResize, false);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onResize() {
    const container = document.getElementById('viewer-container');
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

// ─── Sections & Materials Loading ─────────────────────────────────────────
async function loadSectionsAndMaterials() {
    try {
        const [secRes, matRes] = await Promise.all([
            fetch('/api/sections/list').then(r => r.json()),
            fetch('/api/materials/list').then(r => r.json()),
        ]);

        sectionsList = secRes.sections || {};
        materialsList = matRes.materials || [];

        // Populate dropdowns
        populateSectionDropdowns();
        populateMaterialDropdown();
    } catch (e) {
        console.error('Failed to load sections/materials:', e);
    }
}

function populateSectionDropdowns() {
    const colSelect = document.getElementById('input-col-section');
    const beamXSelect = document.getElementById('input-beamx-section');
    const beamYSelect = document.getElementById('input-beamy-section');
    const propSelect = document.getElementById('prop-new-section');

    [colSelect, beamXSelect, beamYSelect, propSelect].forEach(sel => {
        sel.innerHTML = '';
    });

    // H-beam sections are most relevant for frames
    const allSections = [];
    for (const [type, names] of Object.entries(sectionsList)) {
        const group = document.createElement('optgroup');
        group.label = type;

        for (const name of names) {
            allSections.push(name);
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            group.appendChild(opt);
        }

        [colSelect, beamXSelect, beamYSelect, propSelect].forEach(sel => {
            sel.appendChild(group.cloneNode(true));
        });
    }

    // Set defaults
    setSelectValue(colSelect, 'H-300x300');
    setSelectValue(beamXSelect, 'H-400x200');
    setSelectValue(beamYSelect, 'H-400x200');
}

function populateMaterialDropdown() {
    const matSelect = document.getElementById('input-material');
    matSelect.innerHTML = '';
    for (const name of materialsList) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        matSelect.appendChild(opt);
    }
    setSelectValue(matSelect, 'SS275');
}

// ─── Input Tab Switching ──────────────────────────────────────────────────
function switchInputTab(tabName) {
    document.querySelectorAll('.input-tab').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.trim() === {
            manual: '직접 입력', nl: '자연어', ifc: 'IFC'
        }[tabName]);
    });
    document.querySelectorAll('.tab-content').forEach(div => {
        div.classList.toggle('active', div.id === 'tab-' + tabName);
    });
}

// ─── Story Editor ────────────────────────────────────────────────────────
function buildStoryEditorUI(stories) {
    // stories: [{height, usage}, ...]
    const container = document.getElementById('story-list-container');
    container.innerHTML = '';
    stories.forEach((s, i) => {
        container.appendChild(createStoryRow(i, s.height, s.usage));
    });
}

function createStoryRow(index, height, usage) {
    const row = document.createElement('div');
    row.className = 'story-row';
    row.dataset.index = index;

    // Floor label
    const label = document.createElement('span');
    label.className = 'story-label';
    label.textContent = (index + 1) + 'F';

    // Height input
    const heightInput = document.createElement('input');
    heightInput.type = 'number';
    heightInput.className = 'story-height';
    heightInput.value = height;
    heightInput.step = '0.5';
    heightInput.min = '2.5';
    heightInput.max = '10';

    // Unit label
    const unit = document.createElement('span');
    unit.className = 'unit-label';
    unit.textContent = 'm';

    // Usage select
    const usageSelect = document.createElement('select');
    usageSelect.className = 'usage-select';
    USAGE_OPTIONS.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.label;
        if (opt.value === usage) option.selected = true;
        usageSelect.appendChild(option);
    });

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-remove-story';
    removeBtn.textContent = '\u00D7';
    removeBtn.title = '이 층 삭제';
    removeBtn.onclick = () => {
        row.remove();
        renumberStoryRows();
    };

    row.appendChild(label);
    row.appendChild(heightInput);
    row.appendChild(unit);
    row.appendChild(usageSelect);
    row.appendChild(removeBtn);
    return row;
}

function addStory() {
    const container = document.getElementById('story-list-container');
    const count = container.children.length;
    container.appendChild(createStoryRow(count, 3.5, 'office'));
}

function renumberStoryRows() {
    const container = document.getElementById('story-list-container');
    Array.from(container.children).forEach((row, i) => {
        row.dataset.index = i;
        row.querySelector('.story-label').textContent = (i + 1) + 'F';
    });
}

function getStoriesFromEditor() {
    const rows = document.querySelectorAll('#story-list-container .story-row');
    const stories = [];
    rows.forEach(row => {
        const height = parseFloat(row.querySelector('.story-height').value) || 3.5;
        const usage = row.querySelector('.usage-select').value || 'office';
        stories.push({ height, usage });
    });
    return stories;
}

// ─── Natural Language Input ───────────────────────────────────────────────
async function checkClaudeStatus() {
    try {
        const res = await fetch('/api/claude/status');
        const data = await res.json();
        claudeAvailable = data.available === true;
    } catch { claudeAvailable = false; }

    const warning = document.getElementById('nl-api-warning');
    const btn = document.getElementById('btn-nl-parse');
    if (!claudeAvailable) {
        if (warning) warning.style.display = 'block';
        if (btn) btn.disabled = true;
    } else {
        if (warning) warning.style.display = 'none';
        if (btn) btn.disabled = false;
    }
}

const NL_EXAMPLES = [
    '서울, 5층 오피스, 3×2 경간, 8m',
    '서울 강남, 1층 근생, 2~5층 오피스, 3경간',
    '대전, 1층 주차장 4.5m, 2~3층 사무실',
];

function fillNLExample(n) {
    const textarea = document.getElementById('nl-input');
    if (textarea && NL_EXAMPLES[n]) textarea.value = NL_EXAMPLES[n];
}

async function parseBuilding() {
    const text = document.getElementById('nl-input').value.trim();
    if (!text) { alert('건물 설명을 입력해주세요.'); return; }

    const btn = document.getElementById('btn-nl-parse');
    btn.disabled = true;
    btn.textContent = '변환 중...';
    setStatus('NL 변환 중...', 'running');

    try {
        const res = await fetch('/api/claude/parse-building', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await res.json();

        if (!data.success) {
            throw new Error(data.error || '변환 실패');
        }

        resolvedConfig = data.resolved;
        showResolutionReport(data.intent, data.resolved);
        setStatus('변환 완료', 'success');
    } catch (e) {
        setStatus('변환 실패: ' + e.message, 'error');
        alert('변환 실패: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Claude로 변환';
    }
}

function showResolutionReport(intent, resolved) {
    const reportDiv = document.getElementById('nl-resolution-report');
    const content = document.getElementById('nl-report-content');
    reportDiv.style.display = 'block';

    let html = '';

    // Status badge
    const status = resolved?.status || 'unknown';
    const warnings = resolved?.warnings || [];
    const hasWarnings = warnings.length > 0;
    let badgeClass, badgeText;
    if (status === 'resolved' && !hasWarnings) {
        badgeClass = 'resolved'; badgeText = 'RESOLVED';
    } else if (status === 'resolved' && hasWarnings) {
        badgeClass = 'warning'; badgeText = 'NEEDS REVIEW';
    } else {
        badgeClass = 'error'; badgeText = 'ERROR';
    }
    html += `<span class="nl-status-badge ${badgeClass}">${badgeText}</span>`;

    // Stories
    if (resolved?.config?.stories) {
        html += '<div class="nl-report-section"><h5>Stories</h5>';
        resolved.config.stories.forEach((s, i) => {
            html += `<div class="nl-report-item">${i+1}F: ${s.height}m — ${s.usage}</div>`;
        });
        html += '</div>';
    }

    // Region
    const rm = resolved?.resolution_report?.region_match;
    if (rm) {
        const regionLabel = rm.match_type === 'exact' || rm.match_type === 'partial'
            ? `${rm.region_sido || ''} ${rm.region_sigungu || ''}`.trim()
            : intent?.region_raw || '-';
        html += `<div class="nl-report-section"><h5>Region</h5>`;
        html += `<div class="nl-report-item ${rm.match_type === 'not_found' ? 'warning' : 'ok'}">${regionLabel} (${rm.match_type})</div>`;
        html += '</div>';
    }

    // Warnings list
    if (warnings.length > 0) {
        html += '<div class="nl-report-section"><h5>Warnings</h5>';
        warnings.forEach(w => {
            html += `<div class="nl-report-item warning">[${w.code}] ${w.message}</div>`;
        });
        html += '</div>';
    }

    content.innerHTML = html;
}

function applyResolvedConfig() {
    if (!resolvedConfig?.config) { alert('변환된 config가 없습니다.'); return; }
    const cfg = resolvedConfig.config;

    // Populate story editor
    if (cfg.stories) {
        buildStoryEditorUI(cfg.stories);
    }

    // Bays
    if (cfg.bays_x) document.getElementById('input-bays-x').value = cfg.bays_x.join(', ');
    if (cfg.bays_y) document.getElementById('input-bays-y').value = cfg.bays_y.join(', ');

    // Sections
    if (cfg.column_section) setSelectValue(document.getElementById('input-col-section'), cfg.column_section);
    if (cfg.beam_x_section) setSelectValue(document.getElementById('input-beamx-section'), cfg.beam_x_section);
    if (cfg.beam_y_section) setSelectValue(document.getElementById('input-beamy-section'), cfg.beam_y_section);

    // Material
    if (cfg.material_name) setSelectValue(document.getElementById('input-material'), cfg.material_name);

    // Region
    if (cfg.region) document.getElementById('input-region').value = cfg.region;

    // Supports / Importance
    if (cfg.supports) setSelectValue(document.getElementById('input-supports'), cfg.supports);
    if (cfg.importance) setSelectValue(document.getElementById('input-importance'), cfg.importance);

    // Switch to manual tab
    switchInputTab('manual');
}

async function runAnalysisFromNL() {
    if (!resolvedConfig?.config) { alert('먼저 변환을 실행해주세요.'); return; }

    // Apply resolved config to form, then run analysis
    applyResolvedConfig();
    modelSource = 'NL';
    await runAnalysis();
}

// ─── IFC Wizard ──────────────────────────────────────────────────────────
let ifcParsedData = null;
let ifcSelectedFile = null;
let ifcWizardStep = 1;
let previewMeshes = [];
let ifcEditedData = null;

function initIFCDropzone() {
    const dropzone = document.getElementById('ifc-dropzone');
    const fileInput = document.getElementById('ifc-file-input');
    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].name.toLowerCase().endsWith('.ifc')) {
            handleIFCFile(files[0]);
        } else {
            alert('.ifc 파일만 업로드할 수 있습니다.');
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleIFCFile(e.target.files[0]);
    });
}

function handleIFCFile(file) {
    ifcSelectedFile = file;
    const fnEl = document.getElementById('ifc-filename');
    fnEl.innerHTML = `<span>${file.name} (${(file.size / 1024).toFixed(0)} KB)</span><span class="ifc-remove" onclick="clearIFCFile()">&times;</span>`;
    fnEl.style.display = 'flex';
    document.getElementById('btn-ifc-upload').disabled = false;
    ifcParsedData = null;
    ifcEditedData = null;
    goToIFCStep(1);
}

function clearIFCFile() {
    ifcSelectedFile = null;
    ifcParsedData = null;
    ifcEditedData = null;
    document.getElementById('ifc-filename').style.display = 'none';
    document.getElementById('ifc-file-input').value = '';
    document.getElementById('btn-ifc-upload').disabled = true;
    clearPreviewScene();
    goToIFCStep(1);
}

// ─── Wizard Navigation ──────────────────────────────────────────────────
function goToIFCStep(step) {
    ifcWizardStep = step;
    // Update step containers
    for (let i = 1; i <= 3; i++) {
        const el = document.getElementById('ifc-step-' + i);
        if (el) el.classList.toggle('active', i === step);
    }
    // Update step indicator circles + lines
    document.querySelectorAll('.ifc-step-circle').forEach(c => {
        const s = parseInt(c.dataset.step);
        c.classList.remove('active', 'completed');
        if (s === step) c.classList.add('active');
        else if (s < step) c.classList.add('completed');
    });
    document.querySelectorAll('.ifc-step-line').forEach((line, i) => {
        line.classList.toggle('completed', i < step - 1);
    });
    // Update labels
    const labels = document.querySelectorAll('.ifc-step-labels span');
    labels.forEach((lbl, i) => {
        lbl.classList.remove('active', 'completed');
        if (i === step - 1) lbl.classList.add('active');
        else if (i < step - 1) lbl.classList.add('completed');
    });

    // Step-specific init
    if (step === 2 && ifcParsedData) {
        buildIFCGeometrySummary(ifcParsedData);
        buildPreviewScene(ifcParsedData);
        document.getElementById('preview-badge').style.display = 'block';
    }
    if (step === 3 && ifcParsedData) {
        buildIFCSupplementaryForm();
    }
    if (step === 1) {
        clearPreviewScene();
        document.getElementById('preview-badge').style.display = 'none';
    }
}

// ─── IFC Upload ─────────────────────────────────────────────────────────
async function uploadIFC() {
    if (!ifcSelectedFile) { alert('IFC 파일을 선택해주세요.'); return; }

    const btn = document.getElementById('btn-ifc-upload');
    btn.disabled = true;
    btn.textContent = '파싱 중...';
    setStatus('IFC 파일 분석 중...', 'running');

    try {
        const formData = new FormData();
        formData.append('file', ifcSelectedFile);

        const resp = await fetch('/api/building/parse-ifc', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'IFC 파싱 실패');
        }

        const data = await resp.json();
        if (!data.success) throw new Error(data.error || 'IFC 파싱 실패');

        ifcParsedData = data;
        ifcEditedData = {
            stories: data.stories.map(s => ({ ...s })),
            bays_x: [...(data.bays_x || [])],
            bays_y: [...(data.bays_y || [])],
        };
        setStatus('IFC 파싱 완료', 'success');
        goToIFCStep(2);  // Advance to geometry preview
    } catch (e) {
        alert('IFC 파싱 오류: ' + e.message);
        setStatus('IFC 파싱 실패', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '업로드 & 파싱';
    }
}

// ─── Step 2: Geometry Preview ───────────────────────────────────────────
function buildIFCGeometrySummary(data) {
    const container = document.getElementById('ifc-geometry-summary');
    const ed = ifcEditedData || data;
    let html = '';

    // Overview
    const s = data.summary || {};
    html += `<div class="ifc-geo-section"><h5>건물 개요</h5>`;
    html += `<div class="ifc-geo-info">${s.filename || '-'} | ${data.grid_source || '-'} 기반 (기둥 ${data.num_columns || 0}, 벽 ${data.num_walls || 0})</div>`;
    html += `</div>`;

    // Editable story heights
    html += `<div class="ifc-geo-section"><h5>층별 높이</h5>`;
    ed.stories.forEach((st, i) => {
        html += `<div class="ifc-geo-row">`;
        html += `<span class="ifc-geo-label">${st.name || (i + 1) + 'F'}</span>`;
        html += `<input type="number" class="ifc-story-h" data-index="${i}" value="${st.height}" step="0.5" min="2.0" max="10" onchange="updatePreviewFromEdits()">`;
        html += `<span class="ifc-geo-value">m</span>`;
        html += `</div>`;
    });
    html += `</div>`;

    // Editable bays
    html += `<div class="ifc-geo-section"><h5>경간</h5>`;
    html += `<div class="ifc-geo-row"><span class="ifc-geo-label">X</span>`;
    html += `<input type="text" id="ifc-edit-bays-x" value="${ed.bays_x.map(b => b.toFixed(1)).join(', ')}" onchange="updatePreviewFromEdits()">`;
    html += `<span class="ifc-geo-value">m</span></div>`;
    html += `<div class="ifc-geo-row"><span class="ifc-geo-label">Y</span>`;
    html += `<input type="text" id="ifc-edit-bays-y" value="${ed.bays_y.map(b => b.toFixed(1)).join(', ')}" onchange="updatePreviewFromEdits()">`;
    html += `<span class="ifc-geo-value">m</span></div>`;
    html += `</div>`;

    // Detected sections/material (read-only)
    const ds = data.detected_sections || {};
    html += `<div class="ifc-geo-section"><h5>감지 정보</h5>`;
    if (ds.column) html += `<div class="ifc-geo-info">기둥: ${ds.column}</div>`;
    if (ds.beam) html += `<div class="ifc-geo-info">보: ${ds.beam}</div>`;
    if (data.detected_material) html += `<div class="ifc-geo-info">재료: ${data.detected_material}</div>`;
    html += `</div>`;

    // Warnings
    const warnEl = document.getElementById('ifc-warnings-list');
    if (data.warnings?.length) {
        warnEl.innerHTML = data.warnings.map(w => `<div class="ifc-warning-item">${w}</div>`).join('');
    } else {
        warnEl.innerHTML = '';
    }

    container.innerHTML = html;
}

function updatePreviewFromEdits() {
    if (!ifcEditedData) return;
    // Read edited story heights
    document.querySelectorAll('.ifc-story-h').forEach(inp => {
        const idx = parseInt(inp.dataset.index);
        ifcEditedData.stories[idx].height = parseFloat(inp.value) || 3.5;
    });
    // Read edited bays
    const bxVal = document.getElementById('ifc-edit-bays-x')?.value || '';
    const byVal = document.getElementById('ifc-edit-bays-y')?.value || '';
    ifcEditedData.bays_x = bxVal.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v) && v > 0);
    ifcEditedData.bays_y = byVal.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v) && v > 0);
    // Rebuild wireframe
    buildPreviewScene({ ...ifcParsedData, ...ifcEditedData });
}

// ─── 3D Wireframe Preview (client-side, no OpenSees) ────────────────────
function buildPreviewScene(data) {
    clearPreviewScene();

    const stories = data.stories || [];
    const bays_x = data.bays_x || [];
    const bays_y = data.bays_y || [];
    if (stories.length === 0 || bays_x.length === 0 || bays_y.length === 0) return;

    // Compute grid coordinates
    const xCoords = [0]; bays_x.forEach(b => xCoords.push(xCoords[xCoords.length - 1] + b));
    const yCoords = [0]; bays_y.forEach(b => yCoords.push(yCoords[yCoords.length - 1] + b));
    const zCoords = [0]; stories.forEach(s => zCoords.push(zCoords[zCoords.length - 1] + (s.height || 3.5)));

    const nx = xCoords.length, ny = yCoords.length, ns = stories.length;

    // Materials
    const colMat = new THREE.LineBasicMaterial({ color: 0x4285f4, linewidth: 2 });
    const beamXMat = new THREE.LineBasicMaterial({ color: 0x34a853, linewidth: 2 });
    const beamYMat = new THREE.LineBasicMaterial({ color: 0xfbbc04, linewidth: 2 });
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    const nodeGeo = new THREE.SphereGeometry(0.12, 6, 6);

    // Create line segments
    function addLine(p1, p2, mat) {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(p1[0], p1[2], -p1[1]),  // swap Y/Z for Three.js
            new THREE.Vector3(p2[0], p2[2], -p2[1]),
        ]);
        const line = new THREE.Line(geo, mat);
        scene.add(line);
        previewMeshes.push(line);
    }

    // Columns (vertical)
    for (let s = 0; s < ns; s++) {
        for (let iy = 0; iy < ny; iy++) {
            for (let ix = 0; ix < nx; ix++) {
                addLine(
                    [xCoords[ix], yCoords[iy], zCoords[s]],
                    [xCoords[ix], yCoords[iy], zCoords[s + 1]],
                    colMat
                );
            }
        }
    }

    // Beams X
    for (let s = 1; s <= ns; s++) {
        for (let iy = 0; iy < ny; iy++) {
            for (let ix = 0; ix < nx - 1; ix++) {
                addLine(
                    [xCoords[ix], yCoords[iy], zCoords[s]],
                    [xCoords[ix + 1], yCoords[iy], zCoords[s]],
                    beamXMat
                );
            }
        }
    }

    // Beams Y
    for (let s = 1; s <= ns; s++) {
        for (let iy = 0; iy < ny - 1; iy++) {
            for (let ix = 0; ix < nx; ix++) {
                addLine(
                    [xCoords[ix], yCoords[iy], zCoords[s]],
                    [xCoords[ix], yCoords[iy + 1], zCoords[s]],
                    beamYMat
                );
            }
        }
    }

    // Nodes at grid intersections
    for (let s = 0; s <= ns; s++) {
        for (let iy = 0; iy < ny; iy++) {
            for (let ix = 0; ix < nx; ix++) {
                const sphere = new THREE.Mesh(nodeGeo, nodeMat);
                sphere.position.set(xCoords[ix], zCoords[s], -yCoords[iy]);
                scene.add(sphere);
                previewMeshes.push(sphere);
            }
        }
    }

    // Support triangles at base
    const triGeo = new THREE.ConeGeometry(0.3, 0.4, 4);
    const triMat = new THREE.MeshBasicMaterial({ color: 0xff6600 });
    for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
            const tri = new THREE.Mesh(triGeo, triMat);
            tri.position.set(xCoords[ix], -0.2, -yCoords[iy]);
            scene.add(tri);
            previewMeshes.push(tri);
        }
    }

    // Fit camera
    const cx = xCoords[xCoords.length - 1] / 2;
    const cy = zCoords[zCoords.length - 1] / 2;
    const cz = -yCoords[yCoords.length - 1] / 2;
    const maxDim = Math.max(xCoords[xCoords.length - 1], yCoords[yCoords.length - 1], zCoords[zCoords.length - 1]);
    const dist = maxDim * 1.8;
    camera.position.set(cx + dist * 0.7, cy + dist * 0.5, cz + dist * 0.7);
    controls.target.set(cx, cy, cz);
    controls.update();
}

function clearPreviewScene() {
    previewMeshes.forEach(m => {
        scene.remove(m);
        if (m.geometry) m.geometry.dispose();
        if (m.material) m.material.dispose();
    });
    previewMeshes = [];
    document.getElementById('preview-badge').style.display = 'none';
}

// ─── Step 3: Supplementary Config Form ──────────────────────────────────
function buildIFCSupplementaryForm() {
    if (!ifcParsedData) return;
    const ed = ifcEditedData || ifcParsedData;

    // Build per-story usage rows
    const container = document.getElementById('ifc-story-usage-container');
    let html = '';
    ed.stories.forEach((s, i) => {
        html += `<div class="ifc-usage-row">`;
        html += `<span class="story-label">${s.name || (i + 1) + 'F'}</span>`;
        html += `<select class="usage-select ifc-usage-sel" data-index="${i}">`;
        USAGE_OPTIONS.forEach(opt => {
            const selected = (opt.value === (s.usage || 'office')) ? ' selected' : '';
            html += `<option value="${opt.value}"${selected}>${opt.label}</option>`;
        });
        html += `</select></div>`;
    });
    container.innerHTML = html;

    // Pre-fill sections from IFC detection
    const ds = ifcParsedData.detected_sections || {};
    populateIFCSectionDropdowns();
    if (ds.column) setSelectValue('ifc-col-section', ds.column);
    if (ds.beam) {
        setSelectValue('ifc-beamx-section', ds.beam);
        setSelectValue('ifc-beamy-section', ds.beam);
    }
    if (ifcParsedData.detected_material) {
        setSelectValue('ifc-material', ifcParsedData.detected_material);
    }
}

function populateIFCSectionDropdowns() {
    // Clone options from Manual tab section dropdowns
    const sectionIds = [
        ['input-col-section', 'ifc-col-section'],
        ['input-beamx-section', 'ifc-beamx-section'],
        ['input-beamy-section', 'ifc-beamy-section'],
    ];
    sectionIds.forEach(([srcId, dstId]) => {
        const src = document.getElementById(srcId);
        const dst = document.getElementById(dstId);
        if (!src || !dst) return;
        dst.innerHTML = src.innerHTML;
    });
    // Clone material
    const matSrc = document.getElementById('input-material');
    const matDst = document.getElementById('ifc-material');
    if (matSrc && matDst) matDst.innerHTML = matSrc.innerHTML;
}

// ─── NL Helper for Step 3 ───────────────────────────────────────────────
async function applyNLToIFCForm() {
    const userText = document.getElementById('ifc-nl-input')?.value?.trim();
    if (!userText) { alert('자연어 입력을 작성해주세요.'); return; }
    if (!claudeAvailable) { alert('Claude API가 설정되지 않았습니다.'); return; }

    // Build context from IFC geometry so Claude returns valid BuildingIntent
    const ed = ifcEditedData || ifcParsedData;
    const storyDescs = ed.stories.map((s, i) => `${i+1}층(${s.height}m)`).join(', ');
    const augmented = `${ed.stories.length}층 건물 (${storyDescs}), ${userText}`;

    setStatus('NL 변환 중...', 'running');
    try {
        const resp = await fetch('/api/claude/parse-building', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: augmented }),
        });
        const data = await resp.json();
        if (!data.success) throw new Error(data.error || 'NL 변환 실패');

        const cfg = data.resolved?.config || {};
        const intent = data.intent || {};

        // Apply usage from NL to IFC form (only usage, not geometry)
        if (cfg.stories?.length) {
            const usageRows = document.querySelectorAll('.ifc-usage-sel');
            cfg.stories.forEach((s, i) => {
                if (i < usageRows.length && s.usage) usageRows[i].value = s.usage;
            });
        }
        // Apply region, importance (resolved config first, then intent fallback)
        const region = cfg.region || intent.region_raw;
        if (region) document.getElementById('ifc-region').value = region;
        const importance = cfg.importance || intent.importance;
        if (importance) setSelectValue('ifc-importance', importance);
        // Apply material (intent.material — resolver doesn't handle material)
        const material = cfg.material_name || intent.material || intent.material_name;
        if (material) setSelectValue('ifc-material', material);
        // Apply sections (intent fields)
        const colSec = cfg.column_section || intent.column_section;
        if (colSec) setSelectValue('ifc-col-section', colSec);
        const beamSec = cfg.beam_x_section || intent.beam_section || intent.beam_x_section;
        if (beamSec) {
            setSelectValue('ifc-beamx-section', beamSec);
            setSelectValue('ifc-beamy-section', cfg.beam_y_section || intent.beam_y_section || beamSec);
        }
        // Apply supports
        const supports = cfg.supports || intent.supports;
        if (supports) setSelectValue('ifc-supports', supports);

        setStatus('NL 적용 완료', 'success');
    } catch (e) {
        alert('NL 변환 오류: ' + e.message);
        setStatus('NL 변환 실패', 'error');
    }
}

// ─── Run Analysis from IFC Wizard ───────────────────────────────────────
async function runAnalysisFromIFCWizard() {
    if (!ifcEditedData) { alert('형상 정보가 없습니다.'); return; }

    // Collect stories with usage from Step 3
    const stories = [];
    const usageRows = document.querySelectorAll('.ifc-usage-sel');
    ifcEditedData.stories.forEach((s, i) => {
        stories.push({
            height: s.height,
            usage: (i < usageRows.length) ? usageRows[i].value : 'office',
        });
    });

    const config = {
        stories,
        bays_x: ifcEditedData.bays_x,
        bays_y: ifcEditedData.bays_y,
        column_section: document.getElementById('ifc-col-section')?.value || 'H-300x300',
        beam_x_section: document.getElementById('ifc-beamx-section')?.value || 'H-400x200',
        beam_y_section: document.getElementById('ifc-beamy-section')?.value || 'H-400x200',
        material_name: document.getElementById('ifc-material')?.value || 'SS275',
        supports: document.getElementById('ifc-supports')?.value || 'fixed',
        region: document.getElementById('ifc-region')?.value || '서울',
        importance: document.getElementById('ifc-importance')?.value || 'II',
        auto_combinations: true,
        geometric_nonlinearity: 'linear',
    };

    clearPreviewScene();
    modelSource = 'IFC';
    await runAnalysis(config);
}

// ─── Utility ────────────────────────────────────────────────────────────
function setSelectValue(selectId, value) {
    const sel = (typeof selectId === 'string') ? document.getElementById(selectId) : selectId;
    if (!sel || !value) return;
    const v = value.toUpperCase();
    // Exact match
    for (const opt of sel.options) {
        if (opt.value === value) { sel.value = value; return; }
    }
    // Case-insensitive match
    for (const opt of sel.options) {
        if (opt.value.toUpperCase() === v) { sel.value = opt.value; return; }
    }
    // Substring match (case-insensitive)
    for (const opt of sel.options) {
        if (opt.value.toUpperCase().includes(v) || v.includes(opt.value.toUpperCase())) {
            sel.value = opt.value; return;
        }
    }
}

// ─── Presets ──────────────────────────────────────────────────────────────
function applyPreset() {
    const preset = document.getElementById('preset-select').value || '3story';

    const presets = {
        '3story': {
            stories: [
                { height: 4.0, usage: 'office' },
                { height: 3.5, usage: 'office' },
                { height: 3.5, usage: 'office' },
            ],
            bays_x: '8.0, 8.0',
            bays_y: '8.0, 8.0',
            col: 'H-300x300', beamx: 'H-400x200', beamy: 'H-400x200',
        },
        '5story': {
            stories: [
                { height: 4.0, usage: 'office' },
                { height: 3.5, usage: 'office' },
                { height: 3.5, usage: 'office' },
                { height: 3.5, usage: 'office' },
                { height: 3.5, usage: 'office' },
            ],
            bays_x: '8.0, 8.0, 8.0',
            bays_y: '8.0, 8.0',
            col: 'H-350x350', beamx: 'H-400x200', beamy: 'H-400x200',
        },
        '5story_mixed': {
            stories: [
                { height: 4.5, usage: 'retail' },
                { height: 3.5, usage: 'office' },
                { height: 3.5, usage: 'office' },
                { height: 3.5, usage: 'office' },
                { height: 3.5, usage: 'office' },
            ],
            bays_x: '8.0, 8.0',
            bays_y: '8.0, 8.0',
            col: 'H-350x350', beamx: 'H-400x200', beamy: 'H-400x200',
        },
        '10story': {
            stories: [
                { height: 4.5, usage: 'office' },
                ...Array(9).fill(null).map(() => ({ height: 3.6, usage: 'office' })),
            ],
            bays_x: '8.0, 8.0, 8.0',
            bays_y: '8.0, 8.0, 8.0',
            col: 'H-400x400', beamx: 'H-500x200', beamy: 'H-500x200',
        },
    };

    const p = presets[preset] || presets['3story'];

    // Build story editor rows
    buildStoryEditorUI(p.stories);

    document.getElementById('input-bays-x').value = p.bays_x;
    document.getElementById('input-bays-y').value = p.bays_y;
    setSelectValue(document.getElementById('input-col-section'), p.col);
    setSelectValue(document.getElementById('input-beamx-section'), p.beamx);
    setSelectValue(document.getElementById('input-beamy-section'), p.beamy);
}

// ─── Run Analysis ─────────────────────────────────────────────────────────
async function runAnalysis(configOverride = null) {
    let config;
    if (configOverride) {
        config = configOverride;
    } else {
        if (!modelSource) modelSource = 'Manual';
        // Parse inputs from story editor
        const stories = getStoriesFromEditor();
        const bays_x = document.getElementById('input-bays-x').value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
        const bays_y = document.getElementById('input-bays-y').value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));

        if (stories.length === 0 || bays_x.length === 0 || bays_y.length === 0) {
            alert('층, Bays X, Bays Y 값을 입력해주세요.');
            return;
        }

        config = {
            stories,
            bays_x,
            bays_y,
            column_section: document.getElementById('input-col-section').value,
            beam_x_section: document.getElementById('input-beamx-section').value,
            beam_y_section: document.getElementById('input-beamy-section').value,
            material_name: document.getElementById('input-material').value,
            supports: document.getElementById('input-supports').value,
            region: document.getElementById('input-region').value || '서울',
            importance: document.getElementById('input-importance').value,
            auto_combinations: true,
            geometric_nonlinearity: 'linear',
        };
    }

    showLoading('Analyzing...');
    setStatus('Analyzing...', 'running');

    try {
        const response = await fetch('/api/building/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Analysis failed');
        }

        const result = await response.json();
        currentJobId = result.job_id;
        currentResult = result;

        buildScene(result);
        updateResultsPanel(result);
        updateBottomBar(result);

        setStatus('Analysis Complete', 'success');
    } catch (e) {
        setStatus('Error: ' + e.message, 'error');
        alert('Analysis failed: ' + e.message);
    } finally {
        hideLoading();
    }
}

// ─── Re-analysis ──────────────────────────────────────────────────────────
async function applyMemberChange() {
    if (!currentJobId) {
        alert('Run an analysis first.');
        return;
    }

    const newSection = document.getElementById('prop-new-section').value;
    if (!newSection) return;

    // Determine what to modify based on selected element type
    const modifications = {};
    if (selectedMesh && selectedMesh.userData.elementData) {
        const elemType = selectedMesh.userData.elementData.type;
        if (elemType === 'column') modifications.column_section = newSection;
        else if (elemType === 'beam_x') modifications.beam_x_section = newSection;
        else if (elemType === 'beam_y') modifications.beam_y_section = newSection;
    }

    if (Object.keys(modifications).length === 0) {
        alert('Select a member first.');
        return;
    }

    showLoading('Re-analyzing with modified sections...');
    setStatus('Re-analyzing...', 'running');

    try {
        const response = await fetch(`/api/building/${currentJobId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(modifications),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Re-analysis failed');
        }

        const result = await response.json();
        currentJobId = result.job_id;
        currentResult = result;

        buildScene(result);
        updateResultsPanel(result);
        updateBottomBar(result);

        // Update config panel to reflect new sections
        if (modifications.column_section) setSelectValue(document.getElementById('input-col-section'), modifications.column_section);
        if (modifications.beam_x_section) setSelectValue(document.getElementById('input-beamx-section'), modifications.beam_x_section);
        if (modifications.beam_y_section) setSelectValue(document.getElementById('input-beamy-section'), modifications.beam_y_section);

        setStatus('Re-analysis Complete', 'success');
    } catch (e) {
        setStatus('Error: ' + e.message, 'error');
        alert('Re-analysis failed: ' + e.message);
    } finally {
        hideLoading();
    }
}

// ─── Global Section Apply ─────────────────────────────────────────────────
async function applyGlobalSection(memberType) {
    if (!currentJobId) { alert('먼저 해석을 실행해주세요.'); return; }

    const modifications = {};
    if (memberType === 'column') modifications.column_section = document.getElementById('input-col-section').value;
    else if (memberType === 'beam_x') modifications.beam_x_section = document.getElementById('input-beamx-section').value;
    else if (memberType === 'beam_y') modifications.beam_y_section = document.getElementById('input-beamy-section').value;

    const typeLabel = { column: '기둥', beam_x: 'X보', beam_y: 'Y보' }[memberType];
    showLoading(`전체 ${typeLabel} 단면 변경 중...`);
    setStatus('Re-analyzing...', 'running');

    try {
        const response = await fetch(`/api/building/${currentJobId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(modifications),
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Re-analysis failed');
        }

        const result = await response.json();
        currentJobId = result.job_id;
        currentResult = result;

        buildScene(result);
        updateResultsPanel(result);
        updateBottomBar(result);

        setStatus(`${typeLabel} 전체 적용 완료`, 'success');
    } catch (e) {
        setStatus('Error: ' + e.message, 'error');
        alert('재해석 실패: ' + e.message);
    } finally {
        hideLoading();
    }
}

// ─── Build 3D Scene ───────────────────────────────────────────────────────
function buildScene(result) {
    // Clear preview wireframe (IFC wizard) if present
    clearPreviewScene();

    // Clear existing members
    memberMeshes.forEach(m => scene.remove(m.mesh));
    nodeMeshes.forEach(m => scene.remove(m));
    memberMeshes = [];
    nodeMeshes = [];
    selectedMesh = null;

    const viewer = result.viewer;
    if (!viewer) return;

    const nodes = viewer.nodes;
    const elements = viewer.elements;

    // Build node lookup
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    // Draw elements as cylinders
    elements.forEach(elem => {
        const ni = nodeMap[elem.ni];
        const nj = nodeMap[elem.nj];
        if (!ni || !nj) return;

        // Three.js: Y=up, so map Z→Y
        const start = new THREE.Vector3(ni.x, ni.z, ni.y);
        const end = new THREE.Vector3(nj.x, nj.z, nj.y);

        const dir = new THREE.Vector3().subVectors(end, start);
        const length = dir.length();
        const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

        // Cylinder radius based on type
        const radius = elem.type === 'column' ? 0.15 : 0.10;
        const color = getElementColor(elem);

        const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);
        const material = new THREE.MeshPhongMaterial({
            color,
            transparent: true,
            opacity: 0.85,
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(mid);

        // Orient cylinder to element direction
        const axis = new THREE.Vector3(0, 1, 0);
        const direction = dir.clone().normalize();
        const quaternion = new THREE.Quaternion().setFromUnitVectors(axis, direction);
        mesh.setRotationFromQuaternion(quaternion);

        mesh.userData.elementData = elem;
        scene.add(mesh);
        memberMeshes.push({ mesh, elementData: elem });
    });

    // Draw support nodes (ground level)
    nodes.forEach(n => {
        if (Math.abs(n.z) < 0.01) {
            const geo = new THREE.SphereGeometry(0.2, 8, 8);
            const mat = new THREE.MeshPhongMaterial({ color: 0xff6600 });
            const sphere = new THREE.Mesh(geo, mat);
            sphere.position.set(n.x, n.z, n.y);
            scene.add(sphere);
            nodeMeshes.push(sphere);
        }
    });

    // Fit camera
    fitCameraToModel(viewer);

    // Show DC colors if toggle is on
    if (showDCColors && result.member_checks) {
        applyDesignCheckColors(result.member_checks);
    }
}

function getElementColor(elem) {
    if (elem.type === 'column') return COLORS.column;
    if (elem.type === 'beam_x') return COLORS.beam_x;
    if (elem.type === 'beam_y') return COLORS.beam_y;
    return 0xaaaaaa;
}

function fitCameraToModel(viewer) {
    const cx = viewer.total_width_x / 2;
    const cy = viewer.total_width_y / 2;
    const cz = viewer.total_height / 2;
    const maxDim = Math.max(viewer.total_width_x, viewer.total_width_y, viewer.total_height);
    const dist = maxDim * 1.8;

    camera.position.set(cx + dist * 0.6, cz + dist * 0.4, cy + dist * 0.6);
    controls.target.set(cx, cz, cy);
    controls.update();

    // Update grid
    if (gridHelper) {
        gridHelper.position.set(cx, 0, cy);
    }
}

// ─── Raycaster Click ──────────────────────────────────────────────────────
function onCanvasClick(event) {
    const container = document.getElementById('viewer-container');
    const rect = container.getBoundingClientRect();

    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    const meshes = memberMeshes.map(m => m.mesh);
    const intersects = raycaster.intersectObjects(meshes);

    // Reset previous selection to its original color
    if (selectedMesh) {
        const origColor = selectedMesh.userData.originalColor;
        if (origColor !== undefined) {
            selectedMesh.material.color.setHex(origColor);
        } else {
            selectedMesh.material.color.setHex(getElementColor(selectedMesh.userData.elementData));
        }
        selectedMesh.material.opacity = 0.85;
        delete selectedMesh.userData.originalColor;
    }

    if (intersects.length > 0) {
        selectedMesh = intersects[0].object;
        // Save current color before overwriting
        selectedMesh.userData.originalColor = selectedMesh.material.color.getHex();
        selectedMesh.material.color.setHex(COLORS.selected);
        selectedMesh.material.opacity = 1.0;

        showMemberProperties(selectedMesh.userData.elementData);
    } else {
        selectedMesh = null;
        hideMemberProperties();
    }
}

// ─── Property Panel ───────────────────────────────────────────────────────
function showMemberProperties(elem) {
    document.getElementById('prop-empty').style.display = 'none';
    document.getElementById('prop-member').style.display = 'block';

    // Type badge
    const typeEl = document.getElementById('prop-elem-type');
    typeEl.textContent = elem.type.replace('_', ' ');
    typeEl.className = 'prop-type ' + elem.type;

    document.getElementById('prop-elem-id').textContent = '#' + elem.id;
    document.getElementById('prop-section').textContent = elem.section;
    document.getElementById('prop-material').textContent = currentResult?.viewer?.material_name || '-';

    document.getElementById('prop-node-i').textContent = elem.ni;
    document.getElementById('prop-node-j').textContent = elem.nj;

    // Calculate length
    if (currentResult?.viewer?.nodes) {
        const nodes = currentResult.viewer.nodes;
        const ni = nodes.find(n => n.id === elem.ni);
        const nj = nodes.find(n => n.id === elem.nj);
        if (ni && nj) {
            const dx = nj.x - ni.x, dy = nj.y - ni.y, dz = nj.z - ni.z;
            const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
            document.getElementById('prop-length').textContent = len.toFixed(2) + ' m';
        }
    }

    // Design check for this member
    const dcDiv = document.getElementById('prop-dc');
    if (currentResult?.member_checks && currentResult.member_checks[String(elem.id)]) {
        const mc = currentResult.member_checks[String(elem.id)];
        dcDiv.style.display = 'block';

        const banner = document.getElementById('prop-dc-banner');
        banner.textContent = mc.status;
        banner.className = 'dc-banner ' + (mc.status === 'OK' ? 'ok' : 'ng');

        document.getElementById('prop-dc-ratio').textContent = mc.interaction_ratio.toFixed(3);
        document.getElementById('prop-dc-governing').textContent = mc.governing || '-';
    } else {
        dcDiv.style.display = 'none';
    }

    // Set section dropdown to match current type
    const propSectionSel = document.getElementById('prop-new-section');
    setSelectValue(propSectionSel, elem.section);

    // Edit scope hint
    const typeLabels = { column: '전체 기둥 (Column)', beam_x: '전체 X보 (Beam X)', beam_y: '전체 Y보 (Beam Y)' };
    const scopeLabel = typeLabels[elem.type] || elem.type;
    document.getElementById('prop-edit-scope').textContent = `적용 범위: ${scopeLabel}`;
    const btnLabels = { column: 'Apply to All Columns', beam_x: 'Apply to All Beam X', beam_y: 'Apply to All Beam Y' };
    document.getElementById('btn-apply-member').textContent = btnLabels[elem.type] || 'Apply & Re-analyze';
}

function hideMemberProperties() {
    document.getElementById('prop-empty').style.display = 'block';
    document.getElementById('prop-member').style.display = 'none';
}

// ─── Results Panel ────────────────────────────────────────────────────────
function updateResultsPanel(result) {
    const panel = document.getElementById('prop-results');
    panel.style.display = 'block';

    // Model source tag
    const srcTag = document.getElementById('model-source-tag');
    if (srcTag && modelSource) {
        const labels = { Manual: 'Manual', NL: 'NL (자연어)', IFC: 'IFC + Supplement' };
        srcTag.textContent = 'Source: ' + (labels[modelSource] || modelSource);
    }

    const env = result.envelope || {};
    const table = document.getElementById('results-table');
    table.innerHTML = `
        <tr><td>Max Drift X</td><td>${(env.max_drift_x || 0).toFixed(5)}</td></tr>
        <tr><td>Max Drift Y</td><td>${(env.max_drift_y || 0).toFixed(5)}</td></tr>
        <tr><td>Max Disp X</td><td>${(env.max_dx_mm || 0).toFixed(2)} mm</td></tr>
        <tr><td>Max Disp Y</td><td>${(env.max_dy_mm || 0).toFixed(2)} mm</td></tr>
        <tr><td>Max Moment</td><td>${(env.max_moment_kNm || 0).toFixed(1)} kN·m</td></tr>
        <tr><td>Max Axial</td><td>${(env.max_axial_kN || 0).toFixed(1)} kN</td></tr>
        <tr><td>Max Shear</td><td>${(env.max_shear_kN || 0).toFixed(1)} kN</td></tr>
    `;

    // Modal analysis
    if (result.modal_analysis) {
        const modal = result.modal_analysis;
        if (modal.periods && modal.periods.length > 0) {
            table.innerHTML += `
                <tr><td colspan="2" style="color:#4fc3f7; padding-top:8px;"><strong>Modal Analysis</strong></td></tr>
                <tr><td>T1</td><td>${modal.periods[0].toFixed(3)} s</td></tr>
                ${modal.periods.length > 1 ? `<tr><td>T2</td><td>${modal.periods[1].toFixed(3)} s</td></tr>` : ''}
                ${modal.periods.length > 2 ? `<tr><td>T3</td><td>${modal.periods[2].toFixed(3)} s</td></tr>` : ''}
            `;
        }
    }

    // Design Check summary
    const dcSummary = document.getElementById('dc-summary');
    if (result.design_check) {
        dcSummary.style.display = 'block';
        const dc = result.design_check;

        const banner = document.getElementById('dc-overall-banner');
        const driftOK = (dc.drift_check?.status || dc.story_drift?.status) === 'OK';
        const memberOK = (dc.member_check?.status || dc.member_strength?.status) === 'OK';
        const allOK = driftOK && memberOK;

        banner.textContent = allOK ? 'ALL OK' : 'NG';
        banner.className = 'dc-banner ' + (allOK ? 'ok' : 'ng');

        const dcTable = document.getElementById('dc-summary-table');
        dcTable.innerHTML = `
            <tr><td>Drift Check</td><td class="${driftOK ? '' : 'ng'}">${dc.drift_check?.status || dc.story_drift?.status || '-'}</td></tr>
            <tr><td>Member Check</td><td class="${memberOK ? '' : 'ng'}">${dc.member_check?.status || dc.member_strength?.status || '-'}</td></tr>
        `;

        const mcData = dc.member_check || dc.member_strength;
        if (mcData?.summary) {
            const ms = mcData.summary;
            dcTable.innerHTML += `
                <tr><td>Members OK</td><td>${ms.ok || 0}</td></tr>
                <tr><td>Members NG</td><td class="${(ms.ng || 0) > 0 ? 'ng' : ''}">${ms.ng || 0}</td></tr>
                <tr><td>Max Ratio</td><td>${(ms.max_interaction_ratio || 0).toFixed(3)}</td></tr>
            `;
        }
    } else {
        dcSummary.style.display = 'none';
    }

    // Interpretation
    const interpDiv = document.getElementById('interp-summary');
    if (result.interpretation) {
        interpDiv.style.display = 'block';
        document.getElementById('interp-text').textContent =
            result.interpretation.summary_ko || result.interpretation.summary_en || '';
    } else {
        interpDiv.style.display = 'none';
    }

    // Report link
    const reportDiv = document.getElementById('report-link');
    if (result.report_url) {
        reportDiv.style.display = 'block';
        document.getElementById('report-url').href = result.report_url;
    } else {
        reportDiv.style.display = 'none';
    }
}

function updateBottomBar(result) {
    const bar = document.getElementById('bottom-bar');
    bar.style.display = 'flex';

    const env = result.envelope || {};
    document.getElementById('bot-drift-x').textContent = (env.max_drift_x || 0).toFixed(5);
    document.getElementById('bot-drift-y').textContent = (env.max_drift_y || 0).toFixed(5);
    document.getElementById('bot-disp-x').textContent = (env.max_dx_mm || 0).toFixed(2) + ' mm';
    document.getElementById('bot-disp-y').textContent = (env.max_dy_mm || 0).toFixed(2) + ' mm';
    document.getElementById('bot-moment').textContent = (env.max_moment_kNm || 0).toFixed(1) + ' kN·m';

    // Design check status
    const botDC = document.getElementById('bot-dc');
    if (result.design_check) {
        const dc = result.design_check;
        const allOK = (dc.drift_check?.status || dc.story_drift?.status) === 'OK' && (dc.member_check?.status || dc.member_strength?.status) === 'OK';
        botDC.textContent = allOK ? 'ALL OK' : 'NG';
        botDC.className = 'bottom-value ' + (allOK ? 'ok' : 'ng');
    } else {
        botDC.textContent = '-';
        botDC.className = 'bottom-value';
    }
}

// ─── Design Check Colors ─────────────────────────────────────────────────
function toggleDesignCheckColors() {
    showDCColors = document.getElementById('toggle-dc-colors').checked;
    if (showDCColors && currentResult?.member_checks) {
        applyDesignCheckColors(currentResult.member_checks);
    } else {
        resetElementColors();
    }
}

function applyDesignCheckColors(memberChecks) {
    memberMeshes.forEach(({ mesh, elementData }) => {
        if (mesh === selectedMesh) return;  // don't override selection highlight
        const mc = memberChecks[String(elementData.id)];
        if (mc) {
            if (mc.status === 'OK') {
                mesh.material.color.setHex(mc.interaction_ratio > 0.7 ? COLORS.dc_marginal : COLORS.dc_ok);
            } else {
                mesh.material.color.setHex(COLORS.dc_ng);
            }
        }
    });
    // Update saved originalColor for selected mesh so deselection restores DC color
    if (selectedMesh) {
        const mc = memberChecks[String(selectedMesh.userData.elementData?.id)];
        if (mc) {
            if (mc.status === 'OK') {
                selectedMesh.userData.originalColor = mc.interaction_ratio > 0.7 ? COLORS.dc_marginal : COLORS.dc_ok;
            } else {
                selectedMesh.userData.originalColor = COLORS.dc_ng;
            }
        }
    }
}

function resetElementColors() {
    memberMeshes.forEach(({ mesh, elementData }) => {
        if (mesh === selectedMesh) return;  // don't override selection highlight
        mesh.material.color.setHex(getElementColor(elementData));
    });
    // Update saved originalColor for selected mesh
    if (selectedMesh) {
        selectedMesh.userData.originalColor = getElementColor(selectedMesh.userData.elementData);
    }
}

// ─── Viewer Controls ──────────────────────────────────────────────────────
function resetCamera() {
    if (currentResult?.viewer) {
        fitCameraToModel(currentResult.viewer);
    } else {
        camera.position.set(30, 30, 25);
        controls.target.set(0, 0, 5);
        controls.update();
    }
}

function toggleWireframe() {
    memberMeshes.forEach(({ mesh }) => {
        mesh.material.wireframe = !mesh.material.wireframe;
    });
}

function toggleAxes() {
    if (axesHelper) axesHelper.visible = !axesHelper.visible;
    if (gridHelper) gridHelper.visible = !gridHelper.visible;
}

// ─── UI Helpers ───────────────────────────────────────────────────────────
function showLoading(text) {
    document.getElementById('loading-overlay').style.display = 'flex';
    document.getElementById('loading-text').textContent = text || 'Loading...';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function setStatus(text, type) {
    const el = document.querySelector('.status-text');
    el.textContent = text;
    el.className = 'status-text ' + (type || '');
}
