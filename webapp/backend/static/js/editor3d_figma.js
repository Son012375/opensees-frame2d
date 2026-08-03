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
let selectedMesh = null;       // primary selection (last clicked) for property panel
let selectedMeshSet = new Set(); // multi-selection set
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

// Section property cache
const sectionPropsCache = {};

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

// Deep link: /editor-figma?demo=ifc loads the bundled example straight away, so
// a visitor arriving from the landing page never meets an empty file dialog.
// Runs on `load` (not DOMContentLoaded) so figma_menu.js has already wrapped
// window.uploadIFC — otherwise the parsed model never reaches the 3D view.
window.addEventListener('load', () => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('demo') === 'ifc' && typeof loadSampleIFC === 'function') {
        loadSampleIFC();
    }
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

    // Events — selection on canvas only (not container, to avoid blocking UI buttons)
    canvas.addEventListener('pointerdown', onCanvasMouseDown, false);
    canvas.addEventListener('pointermove', onCanvasMouseMove, false);
    canvas.addEventListener('pointerup', onCanvasMouseUp, false);
    window.addEventListener('pointerup', onCanvasMouseUpWindow, false); // finish box drag outside canvas
    canvas.addEventListener('mouseleave', () => { hideHoverTooltip(); }, false);
    window.addEventListener('resize', onResize, false);
    window.addEventListener('keydown', onSelectionKeyDown, false);
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

        // Bind section property preview popups
        bindAllSectionPreviews();
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

// ─── Section Property Preview ─────────────────────────────────────────────
let sectionPopupEl = null;
let sectionPopupTarget = null;

function getSectionPopup() {
    if (!sectionPopupEl) {
        sectionPopupEl = document.createElement('div');
        sectionPopupEl.id = 'section-props-popup';
        document.body.appendChild(sectionPopupEl);
    }
    return sectionPopupEl;
}

async function fetchSectionProps(name) {
    if (sectionPropsCache[name]) return sectionPropsCache[name];
    try {
        const res = await fetch('/api/sections/properties/' + encodeURIComponent(name));
        const data = await res.json();
        if (!data.error) sectionPropsCache[name] = data;
        return data;
    } catch { return null; }
}

function renderSectionPopup(popup, data) {
    if (!data || data.error) {
        popup.innerHTML = `<div class="sp-loading">${data?.error || '데이터 없음'}</div>`;
        return;
    }
    const rows = [
        ['A',  data.A_cm2,  'cm\u00B2'],
        ['Ix', data.Ix_cm4, 'cm\u2074'],
        ['Iy', data.Iy_cm4, 'cm\u2074'],
        ['J',  data.J_cm4,  'cm\u2074'],
        ['H',  data.h_mm,   'mm'],
        ['B',  data.b_mm,   'mm'],
        ['tw', data.tw_mm,  'mm'],
        ['tf', data.tf_mm,  'mm'],
    ];
    let grid = rows
        .filter(r => r[1] > 0)
        .map(([k, v, u]) => `<span class="sp-key">${k}</span><span class="sp-val">${v.toLocaleString()}</span><span class="sp-unit">${u}</span>`)
        .join('');
    popup.innerHTML = `<div class="sp-title">${data.name}</div><div class="sp-grid">${grid}</div>`;
}

async function showSectionPopup(selectEl) {
    const popup = getSectionPopup();
    const name = selectEl.value;
    if (!name) return;
    sectionPopupTarget = selectEl;

    // Position fixed below the select element
    const rect = selectEl.getBoundingClientRect();
    popup.style.left = rect.left + 'px';
    popup.style.top = (rect.bottom + 4) + 'px';

    popup.innerHTML = '<div class="sp-loading">Loading...</div>';
    popup.classList.add('visible');

    const data = await fetchSectionProps(name);
    if (sectionPopupTarget === selectEl && popup.classList.contains('visible')) {
        renderSectionPopup(popup, data);
    }
}

function hideSectionPopup() {
    const popup = getSectionPopup();
    popup.classList.remove('visible');
    sectionPopupTarget = null;
}

function bindSectionPreview(selectEl) {
    if (!selectEl || selectEl.dataset.spBound) return;
    selectEl.dataset.spBound = '1';
    selectEl.addEventListener('focus', () => showSectionPopup(selectEl));
    selectEl.addEventListener('change', () => showSectionPopup(selectEl));
    selectEl.addEventListener('blur', () => setTimeout(hideSectionPopup, 200));
}

function bindAllSectionPreviews() {
    // Manual tab dropdowns
    ['input-col-section', 'input-beamx-section', 'input-beamy-section'].forEach(id => {
        bindSectionPreview(document.getElementById(id));
    });
    // Properties panel dropdown
    bindSectionPreview(document.getElementById('prop-new-section'));
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
    // Show live wireframe when switching to Manual tab (if no analysis result)
    if (tabName === 'manual' && !currentResult) {
        updateManualPreview();
    }
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

    // Height slider
    const heightSlider = document.createElement('input');
    heightSlider.type = 'range';
    heightSlider.className = 'story-slider';
    heightSlider.value = height;
    heightSlider.step = '0.1';
    heightSlider.min = '2.5';
    heightSlider.max = '10';

    // Hidden number input (keeps existing getter working)
    const heightInput = document.createElement('input');
    heightInput.type = 'number';
    heightInput.className = 'story-height';
    heightInput.value = height;
    heightInput.step = '0.5';
    heightInput.min = '2.5';
    heightInput.max = '10';
    heightInput.style.display = 'none';

    // Slider value display
    const valDisplay = document.createElement('span');
    valDisplay.className = 'slider-value';
    valDisplay.textContent = parseFloat(height).toFixed(1);

    // Unit label
    const unit = document.createElement('span');
    unit.className = 'unit-label';
    unit.textContent = 'm';

    // Bidirectional binding: slider ↔ hidden input + display
    heightSlider.oninput = () => {
        const v = parseFloat(heightSlider.value);
        heightInput.value = v;
        valDisplay.textContent = v.toFixed(1);
        updateManualPreview();
    };

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
        updateManualPreview();
    };

    row.appendChild(label);
    row.appendChild(heightSlider);
    row.appendChild(valDisplay);
    row.appendChild(unit);
    row.appendChild(heightInput);
    row.appendChild(usageSelect);
    row.appendChild(removeBtn);
    return row;
}

function addStory() {
    const container = document.getElementById('story-list-container');
    const count = container.children.length;
    container.appendChild(createStoryRow(count, 3.5, 'office'));
    updateManualPreview();
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

// ─── Bay Slider Editor ───────────────────────────────────────────────────
function buildBaySliders(axis) {
    // axis: 'x' or 'y'
    const textInput = document.getElementById('input-bays-' + axis);
    const container = document.getElementById('bays-' + axis + '-slider-container');
    if (!container) return;

    const bays = textInput.value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v) && v > 0);
    container.innerHTML = '';

    bays.forEach((w, i) => {
        container.appendChild(createBayRow(i, w, axis));
    });

    // Add bay button
    const addBtn = document.createElement('button');
    addBtn.className = 'btn-add-bay';
    addBtn.textContent = '+ 경간 추가';
    addBtn.onclick = () => addBay(axis);
    container.appendChild(addBtn);
}

function createBayRow(index, width, axis) {
    const row = document.createElement('div');
    row.className = 'bay-row';
    row.dataset.axis = axis;
    row.dataset.index = index;

    const label = document.createElement('span');
    label.className = 'bay-label';
    label.textContent = (index + 1);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'bay-slider';
    slider.value = width;
    slider.step = '0.5';
    slider.min = '3.0';
    slider.max = '15.0';

    const valDisplay = document.createElement('span');
    valDisplay.className = 'slider-value';
    valDisplay.textContent = parseFloat(width).toFixed(1) + 'm';

    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-remove-bay';
    removeBtn.textContent = '\u00D7';
    removeBtn.title = '이 경간 삭제';
    removeBtn.onclick = () => {
        row.remove();
        syncBaySlidersToText(axis);
        renumberBayRows(axis);
        updateManualPreview();
    };

    slider.oninput = () => {
        valDisplay.textContent = parseFloat(slider.value).toFixed(1) + 'm';
        syncBaySlidersToText(axis);
        updateManualPreview();
    };

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(valDisplay);
    row.appendChild(removeBtn);
    return row;
}

function addBay(axis) {
    const container = document.getElementById('bays-' + axis + '-slider-container');
    const addBtn = container.querySelector('.btn-add-bay');
    const count = container.querySelectorAll('.bay-row').length;
    container.insertBefore(createBayRow(count, 8.0, axis), addBtn);
    syncBaySlidersToText(axis);
    updateManualPreview();
}

function renumberBayRows(axis) {
    const container = document.getElementById('bays-' + axis + '-slider-container');
    container.querySelectorAll('.bay-row').forEach((row, i) => {
        row.dataset.index = i;
        row.querySelector('.bay-label').textContent = (i + 1);
    });
}

function syncBaySlidersToText(axis) {
    const container = document.getElementById('bays-' + axis + '-slider-container');
    const values = [];
    container.querySelectorAll('.bay-slider').forEach(s => {
        values.push(parseFloat(s.value).toFixed(1));
    });
    const textInput = document.getElementById('input-bays-' + axis);
    textInput.value = values.join(', ');
}

function syncTextToBaySliders(axis) {
    buildBaySliders(axis);
    updateManualPreview();
}

function getBaysFromSliders(axis) {
    const container = document.getElementById('bays-' + axis + '-slider-container');
    if (!container) return [];
    const values = [];
    container.querySelectorAll('.bay-slider').forEach(s => {
        const v = parseFloat(s.value);
        if (!isNaN(v) && v > 0) values.push(v);
    });
    return values;
}

// ─── Real-time Manual Preview ────────────────────────────────────────────
function parseBaysFromText(axis) {
    const el = document.getElementById('input-bays-' + axis);
    if (!el) return [];
    return el.value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v) && v > 0);
}

function updateManualPreview() {
    // Only update when Manual tab is active
    const manualTab = document.getElementById('tab-manual');
    if (!manualTab || !manualTab.classList.contains('active')) return;

    const stories = getStoriesFromEditor();
    const isIrregular = document.getElementById('irregular-toggle')?.checked;

    // Clear previous analysis scene to show wireframe preview
    if (currentResult) {
        memberMeshes.forEach(m => scene.remove(m.mesh));
        nodeMeshes.forEach(m => scene.remove(m));
        memberMeshes = [];
        nodeMeshes = [];
        selectedMesh = null;
        selectedMeshSet.clear();
        currentResult = null;
        currentJobId = null;
        _setExportBtnEnabled(false);
    }

    if (isIrregular) {
        const zones = getZonesFromEditor();
        if (stories.length > 0 && zones.length > 0) {
            buildIrregularPreview({ stories, zones });
        }
    } else {
        let bays_x = getBaysFromSliders('x');
        let bays_y = getBaysFromSliders('y');
        if (bays_x.length === 0) bays_x = parseBaysFromText('x');
        if (bays_y.length === 0) bays_y = parseBaysFromText('y');
        if (stories.length > 0 && bays_x.length > 0 && bays_y.length > 0) {
            buildPreviewScene({ stories, bays_x, bays_y });
        }
    }
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
function goToIFCStepSafely(step) {
    // 해석 결과가 있는 상태에서 이전 단계로 돌아갈 때: 확인 + 정리
    if (currentResult && step < ifcWizardStep) {
        const ok = confirm('편집 모드로 돌아가면 해석 결과가 초기화됩니다.\n계속하시겠습니까?');
        if (!ok) return;
        _clearAnalysisResults();
    }
    goToIFCStep(step);
}

function _clearAnalysisResults() {
    // 해석 결과 완전 정리
    currentResult = null;
    currentJobId = null;
    // 결과 씬의 부재/노드 메쉬 완전 정리 (겹침 방지)
    memberMeshes.forEach(({ mesh }) => {
        if (!mesh) return;
        if (mesh.parent) scene.remove(mesh);
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
            if (Array.isArray(mesh.material)) mesh.material.forEach(m => m.dispose());
            else mesh.material.dispose();
        }
    });
    memberMeshes = [];
    nodeMeshes.forEach(m => {
        if (!m) return;
        if (m.parent) scene.remove(m);
        if (m.geometry) m.geometry.dispose();
        if (m.material) m.material.dispose();
    });
    nodeMeshes = [];
    // 결과 관련 3D 요소 정리
    if (typeof _clearLoadArrows === 'function') _clearLoadArrows();
    if (typeof _clearDiagrams === 'function') _clearDiagrams();
    if (typeof clearAllSelection === 'function') clearAllSelection();
    // Solid Section 리셋
    if (typeof removeSolidMeshes === 'function') removeSolidMeshes();
    window.solidMode = false;
    const chkSolid = document.getElementById('chk-solid-section');
    if (chkSolid) chkSolid.checked = false;
    const chkSolidTop = document.getElementById('chk-solid-section-top');
    if (chkSolidTop) chkSolidTop.checked = false;
    // Deformed shape 리셋
    const chkDeformed = document.getElementById('toggle-deformed');
    if (chkDeformed) chkDeformed.checked = false;
    _lastDisplacements = null;
    originalMemberState = null;
    originalNodeState = null;
    // 결과 패널 숨기기
    const propResults = document.getElementById('prop-results');
    if (propResults) propResults.style.display = 'none';
    const propEmpty = document.getElementById('prop-empty');
    if (propEmpty) propEmpty.style.display = 'block';
    // Export 버튼 비활성화
    if (typeof _setExportBtnEnabled === 'function') _setExportBtnEnabled(false);
    // 3D Model 탭으로 전환
    if (typeof switchViewerTab === 'function') switchViewerTab('model');
    // 바닥 상태바 숨기기
    const botBar = document.getElementById('bottom-bar');
    if (botBar) botBar.style.display = 'none';
}

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
        // V2: 실제 노드-요소 기반 프리뷰 (비정형 포함)
        if (window._v2Model) {
            buildV2PreviewScene(window._v2Model);
        } else if (ifcParsedData.detected_zones && ifcParsedData.detected_zones.length > 0) {
            buildIrregularPreview({
                stories: ifcEditedData.stories,
                zones: ifcParsedData.detected_zones,
            });
        } else {
            buildPreviewScene(ifcParsedData);
        }
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

// ─── Sample IFC ─────────────────────────────────────────────────────────
// A visitor arriving with no file of their own is the common case for a link
// sent cold. This fetches the bundled example and hands it to the *existing*
// upload path — no duplicated parsing, snapping or analysis logic.
const SAMPLE_IFC_URL = '/static/files/ifc_example.ifc';

// While a model is being fetched/parsed/snapped, Run must not fire: the IFC
// model is not the active one yet, so an early click silently analyses the
// manual-input default preset instead — a wrong answer that looks right.
// `?demo=ifc` makes this window reachable, because the load starts by itself
// the moment the page appears.
window._modelLoading = false;

function setModelLoading(on) {
    window._modelLoading = !!on;
    document.querySelectorAll('.ribbon-command.run, .analysis-btn.primary')
        .forEach(function (b) {
            b.disabled = !!on;
            b.title = on ? '모델을 불러오는 중입니다…' : '';
        });
}

async function loadSampleIFC() {
    const btn = document.getElementById('btn-ifc-sample');
    const original = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = '예제 불러오는 중...'; }
    setModelLoading(true);

    try {
        const resp = await fetch(SAMPLE_IFC_URL, { cache: 'force-cache' });
        if (!resp.ok) throw new Error('예제 파일을 불러오지 못했습니다 (' + resp.status + ')');
        const blob = await resp.blob();
        const file = new File([blob], 'ifc_example.ifc', { type: 'application/octet-stream' });

        handleIFCFile(file);          // sets ifcSelectedFile + enables the button
        await uploadIFC();            // parse -> auto-snap -> merge -> split
    } catch (e) {
        console.error('[sample IFC]', e);
        setStatus('예제 IFC 로드 실패: ' + e.message, 'error');
        alert('예제 IFC를 불러오지 못했습니다.\n' + e.message);
    } finally {
        setModelLoading(false);
        if (btn) { btn.disabled = false; btn.textContent = original; }
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

        // ── V2 IFC Parser (node-element based) ──
        const resp = await fetch('/api/v2/parse-ifc', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'IFC V2 파싱 실패');
        }

        const v2data = await resp.json();
        if (!v2data.success) throw new Error('IFC V2 파싱 실패');

        // V2 model을 전역에 저장 (해석 시 사용)
        window._v2Model = v2data.model;
        window._v2Validation = v2data.validation;
        window._v2ViewerUrl = v2data.viewer_url;

        // 자동 스냅 시도
        if (v2data.validation.needs_user_input.some(i => i.code === 'IFC_DISCONNECTED_JOINTS')) {
            try {
                const snapResp = await fetch('/api/v2/snap-joints', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ model: v2data.model, snap_tolerance: 0.5 }),
                });
                const snapData = await snapResp.json();
                if (snapData.success) {
                    window._v2Model = snapData.model;
                    console.log(`Snapped ${snapData.snapped_count} nodes`);
                }
            } catch (e) { console.warn('Snap failed:', e); }
        }

        // ── 좌표 원점 보정: 최소 좌표 → (0,0,0) ──
        normalizeV2ModelOrigin(window._v2Model);

        // ── A-1: 근접 노드 병합 (보-기둥 접합 보장, 적응형 tolerance) ──
        if (typeof mergeNearbyNodes === 'function') {
            console.log('[Merge] before: ' + window._v2Model.nodes.length + ' nodes, ' + window._v2Model.elements.length + ' elems');
            var mergeCount = mergeNearbyNodes(window._v2Model);
            console.log('[Merge] result: ' + mergeCount + ' merged → ' + window._v2Model.nodes.length + ' nodes, ' + window._v2Model.elements.length + ' elems');
        } else {
            console.warn('[Merge] mergeNearbyNodes not found');
        }

        // ── 보-보 교차점 노드 생성 ──
        if (typeof createBeamIntersectionNodes === 'function') {
            var intCount = createBeamIntersectionNodes(window._v2Model);
            if (intCount > 0) {
                console.log('[Intersection] ' + intCount + ' intersection nodes created');
            }
        }

        // ── A-4: Element 자동 분할: 중간 노드가 있으면 분할 (적응형 tolerance) ──
        if (typeof splitElementsAtNodes === 'function') {
            var splitCount = splitElementsAtNodes(window._v2Model);
            if (splitCount > 0) {
                console.log('Auto-split: ' + splitCount + ' elements divided at intermediate nodes');
            }
        }

        // ── A-2: 연결성 검증 ──
        if (typeof validateConnectivity === 'function') {
            var connWarnings = validateConnectivity(window._v2Model);
            connWarnings.forEach(function(w) {
                console.warn('[Connectivity] ' + w.severity + ': ' + w.message);
            });
            if (window._v2Model) window._v2Model._connectivityWarnings = connWarnings;
        }

        // V2 → V1 호환 형식 변환 (스냅 후 모델 기준)
        const currentV2Model = window._v2Model;
        const sm = v2data.summary;
        const elevations = currentV2Model.story_elevations || v2data.model.story_elevations || [];
        const v1Stories = [];
        for (let i = 1; i < elevations.length; i++) {
            v1Stories.push({
                name: `${i}F`,
                height: Math.round((elevations[i] - elevations[i-1]) * 100) / 100,
                usage: 'office',
            });
        }

        // bays 추정 (스냅 후 V2 모델 노드에서 — 기둥 노드만 사용)
        const v2Nodes = currentV2Model.nodes || v2data.model.nodes || [];
        // 기둥의 상단 노드만 (story >= 1) 사용하여 정확한 그리드 추정
        const columnNodeIds = new Set();
        (currentV2Model.elements || []).forEach(e => {
            if (e.elem_type === 'column') { columnNodeIds.add(e.node_i); columnNodeIds.add(e.node_j); }
        });
        const colNodes = v2Nodes.filter(n => columnNodeIds.has(n.id));
        const nodesForBays = colNodes.length > 4 ? colNodes : v2Nodes;

        const xSet = [...new Set(nodesForBays.map(n => Math.round(n.x * 10) / 10))].sort((a,b) => a-b);
        const ySet = [...new Set(nodesForBays.map(n => Math.round(n.y * 10) / 10))].sort((a,b) => a-b);
        const bays_x = xSet.length >= 2 ? xSet.slice(1).map((x, i) => Math.round((x - xSet[i]) * 10) / 10).filter(b => b > 0.5) : [8.0];
        const bays_y = ySet.length >= 2 ? ySet.slice(1).map((y, i) => Math.round((y - ySet[i]) * 10) / 10).filter(b => b > 0.5) : [8.0];

        const sectionsUsed = sm.sections_used || [];
        const data = {
            success: true,
            stories: v1Stories,
            bays_x: bays_x,
            bays_y: bays_y,
            grid_x: xSet,
            grid_y: ySet,
            detected_sections: {
                column: sectionsUsed.find(s => s.includes('300') || s.includes('350') || s.includes('400x4')) || sectionsUsed[0] || 'H-300x300',
                beam: sectionsUsed.find(s => s.includes('200') || s.includes('250x2')) || sectionsUsed[1] || sectionsUsed[0] || 'H-400x200',
            },
            detected_material: sm.materials_used?.[0] || 'SS275',
            grid_source: 'V2 node-element',
            num_columns: sm.num_columns || 0,
            num_walls: 0,
            warnings: (v2data.validation.issues || []).map(i => `[${i.severity}] ${i.message}`),
            detected_zones: null,
            is_irregular: false,
            summary: {
                num_stories: sm.num_stories,
                num_bays_x: bays_x.length,
                num_bays_y: bays_y.length,
                total_height: sm.total_height_m,
                filename: ifcSelectedFile.name,
                is_irregular: false,
                num_zones: 0,
                // V2 specific
                num_nodes: sm.num_nodes,
                num_elements: sm.num_elements,
                num_columns: sm.num_columns,
                num_beams: sm.num_beams,
                num_braces: sm.num_braces,
                pipeline: 'v2_node_element',
            },
        };

        ifcParsedData = data;
        ifcEditedData = {
            stories: data.stories.map(s => ({ ...s })),
            bays_x: [...(data.bays_x || [])],
            bays_y: [...(data.bays_y || [])],
            detected_zones: null,
            is_irregular: false,
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
    if (!container) {
        // Step 2가 도구 팔레트 모드일 때 — tool model info로 대체
        if (typeof updateToolModelInfo === 'function') updateToolModelInfo();
        return;
    }
    const ed = ifcEditedData || data;
    let html = '';

    // Overview
    const s = data.summary || {};
    html += `<div class="ifc-geo-section"><h5>건물 개요</h5>`;
    html += `<div class="ifc-geo-info">${s.filename || '-'} | ${data.grid_source || '-'} 기반 (기둥 ${data.num_columns || 0}, 벽 ${data.num_walls || 0})</div>`;
    // Zone detection banner
    if (data.detected_zones && data.detected_zones.length > 0) {
        const zoneNames = data.detected_zones.map(z => z.id).join(', ');
        html += `<div class="ifc-zone-banner">
            <strong>비정형 평면 감지</strong> — ${data.detected_zones.length}개 존 (${zoneNames})
        </div>`;
        data.detected_zones.forEach((z, zi) => {
            const color = ['#4285f4','#34a853','#fbbc04','#ea4335'][zi % 4];
            const wx = z.bays_x.reduce((a,b) => a+b, 0);
            const wy = z.bays_y.reduce((a,b) => a+b, 0);
            html += `<div class="ifc-zone-info" style="border-left:3px solid ${color}">
                <strong style="color:${color}">Zone ${z.id}</strong>:
                ${z.bays_x.length}×${z.bays_y.length} 경간,
                ${wx}×${wy}m,
                원점 (${z.origin_x}, ${z.origin_y})
            </div>`;
        });
    }
    html += `</div>`;

    // Editable story heights (slider + value + delete btn)
    html += `<div class="ifc-geo-section"><h5>층별 높이</h5>`;
    ed.stories.forEach((st, i) => {
        const h = parseFloat(st.height) || 3.5;
        html += `<div class="ifc-geo-row">`;
        html += `<span class="ifc-geo-label">${st.name || (i + 1) + 'F'}</span>`;
        html += `<input type="range" class="ifc-story-slider" data-index="${i}" value="${h}" step="0.1" min="2.0" max="10">`;
        html += `<span class="slider-value ifc-story-val" data-index="${i}">${h.toFixed(1)}</span>`;
        html += `<span class="ifc-geo-value">m</span>`;
        html += `<button class="btn-ifc-remove-story" onclick="removeIFCStory(${i})" title="이 층 제거 (옥상 등)">&times;</button>`;
        html += `<input type="hidden" class="ifc-story-h" data-index="${i}" value="${h}">`;
        html += `</div>`;
    });
    html += `</div>`;

    // Editable bays (text input + per-bay sliders)
    html += `<div class="ifc-geo-section"><h5>경간</h5>`;
    html += `<div class="ifc-geo-row"><span class="ifc-geo-label">X</span>`;
    html += `<input type="text" id="ifc-edit-bays-x" value="${ed.bays_x.map(b => b.toFixed(1)).join(', ')}" onchange="syncIFCBayText('x')">`;
    html += `<span class="ifc-geo-value">m</span></div>`;
    html += `<div id="ifc-bays-x-sliders" class="ifc-bay-sliders"></div>`;
    html += `<div class="ifc-geo-row"><span class="ifc-geo-label">Y</span>`;
    html += `<input type="text" id="ifc-edit-bays-y" value="${ed.bays_y.map(b => b.toFixed(1)).join(', ')}" onchange="syncIFCBayText('y')">`;
    html += `<span class="ifc-geo-value">m</span></div>`;
    html += `<div id="ifc-bays-y-sliders" class="ifc-bay-sliders"></div>`;
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
    if (!warnEl) return;
    if (data.warnings?.length) {
        warnEl.innerHTML = data.warnings.map(w => `<div class="ifc-warning-item">${w}</div>`).join('');
    } else {
        warnEl.innerHTML = '';
    }

    container.innerHTML = html;

    // Bind story slider events (after innerHTML)
    container.querySelectorAll('.ifc-story-slider').forEach(slider => {
        slider.addEventListener('input', () => {
            const idx = slider.dataset.index;
            const v = parseFloat(slider.value);
            const valEl = container.querySelector(`.ifc-story-val[data-index="${idx}"]`);
            const hiddenEl = container.querySelector(`.ifc-story-h[data-index="${idx}"]`);
            if (valEl) valEl.textContent = v.toFixed(1);
            if (hiddenEl) hiddenEl.value = v;
            updatePreviewFromEdits();
        });
    });

    // Build IFC bay sliders
    buildIFCBaySliders('x', ed.bays_x);
    buildIFCBaySliders('y', ed.bays_y);
}

function buildIFCBaySliders(axis, bays) {
    const container = document.getElementById('ifc-bays-' + axis + '-sliders');
    if (!container) return;
    container.innerHTML = '';
    bays.forEach((w, i) => {
        const row = document.createElement('div');
        row.className = 'ifc-geo-row';
        row.innerHTML = `
            <span class="ifc-geo-label" style="font-size:10px">${i + 1}</span>
            <input type="range" class="ifc-bay-slider" data-axis="${axis}" data-index="${i}"
                   value="${w}" step="0.5" min="3.0" max="15.0">
            <span class="slider-value">${parseFloat(w).toFixed(1)}m</span>`;
        const slider = row.querySelector('input[type="range"]');
        const valSpan = row.querySelector('.slider-value');
        slider.addEventListener('input', () => {
            valSpan.textContent = parseFloat(slider.value).toFixed(1) + 'm';
            syncIFCBaySliders(axis);
            updatePreviewFromEdits();
        });
        container.appendChild(row);
    });
}

function syncIFCBaySliders(axis) {
    const container = document.getElementById('ifc-bays-' + axis + '-sliders');
    if (!container) return;
    const values = [];
    container.querySelectorAll('.ifc-bay-slider').forEach(s => {
        values.push(parseFloat(s.value).toFixed(1));
    });
    const textInput = document.getElementById('ifc-edit-bays-' + axis);
    if (textInput) textInput.value = values.join(', ');
}

function syncIFCBayText(axis) {
    const textInput = document.getElementById('ifc-edit-bays-' + axis);
    if (!textInput) return;
    const bays = textInput.value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v) && v > 0);
    buildIFCBaySliders(axis, bays);
    updatePreviewFromEdits();
}

function removeIFCStory(index) {
    if (!ifcEditedData || ifcEditedData.stories.length <= 1) return;
    ifcEditedData.stories.splice(index, 1);
    // Rebuild the geometry summary with updated stories
    buildIFCGeometrySummary(ifcParsedData);
    updatePreviewFromEdits();
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
// ─── V2 좌표 원점 보정 ─────────────────────────────────────────────────
function normalizeV2ModelOrigin(model) {
    if (!model || !model.nodes || model.nodes.length === 0) return;
    var xs = model.nodes.map(function(n){return n.x;});
    var ys = model.nodes.map(function(n){return n.y;});
    var zs = model.nodes.map(function(n){return n.z;});
    var dx = Math.min.apply(null, xs);
    var dy = Math.min.apply(null, ys);
    var dz = Math.min.apply(null, zs);
    if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01 && Math.abs(dz) < 0.01) return;
    // 모든 노드 이동
    model.nodes.forEach(function(n) {
        n.x = Math.round((n.x - dx) * 1000) / 1000;
        n.y = Math.round((n.y - dy) * 1000) / 1000;
        n.z = Math.round((n.z - dz) * 1000) / 1000;
    });
    // 층 표고도 보정
    if (model.story_elevations) {
        model.story_elevations = model.story_elevations.map(function(e) {
            return Math.round((e - dz) * 1000) / 1000;
        });
    }
    console.log('Origin normalized: dx=' + dx.toFixed(2) + ', dy=' + dy.toFixed(2) + ', dz=' + dz.toFixed(2));
}

// ─── V2 Preview: 실제 노드-요소 기반 프리뷰 ─────────────────────────────
function buildV2PreviewScene(v2Model, skipCameraFit) {
    clearPreviewScene();

    const nodes = v2Model.nodes || [];
    const elements = v2Model.elements || [];
    if (nodes.length === 0) return;

    // 노드 맵
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    // Materials — MeshBasicMaterial for selectable tube geometry
    const colMat = new THREE.MeshBasicMaterial({ color: 0x4285f4, transparent: true, opacity: 0.85 });
    const beamMat = new THREE.MeshBasicMaterial({ color: 0x34a853, transparent: true, opacity: 0.85 });
    const braceMat = new THREE.MeshBasicMaterial({ color: 0xfbbc04, transparent: true, opacity: 0.85 });
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    const supportMat = new THREE.MeshBasicMaterial({ color: 0xff6600 });
    const nodeGeo = new THREE.SphereGeometry(0.25, 8, 8);
    const supportGeo = new THREE.ConeGeometry(0.3, 0.4, 4);
    const TUBE_RADIUS = 0.12;
    const TUBE_SEGMENTS = 6;

    function addTube(n1, n2, mat, elemData) {
        const p1 = new THREE.Vector3(n1.x, n1.z, -n1.y);
        const p2 = new THREE.Vector3(n2.x, n2.z, -n2.y);
        const dir = new THREE.Vector3().subVectors(p2, p1);
        const len = dir.length();
        if (len < 0.001) return;
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        const geo = new THREE.CylinderGeometry(TUBE_RADIUS, TUBE_RADIUS, len, TUBE_SEGMENTS);
        const mesh = new THREE.Mesh(geo, mat.clone());
        mesh.position.copy(mid);
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
        mesh.userData.elementData = elemData;
        mesh.userData._isPreviewElement = true;
        scene.add(mesh);
        previewMeshes.push(mesh);
        // Also register in memberMeshes for selection
        memberMeshes.push({ mesh, elementData: elemData });
    }

    // Draw elements
    elements.forEach(e => {
        const ni = nodeMap[e.node_i];
        const nj = nodeMap[e.node_j];
        if (!ni || !nj) return;

        let mat = beamMat;
        if (e.elem_type === 'column') mat = colMat;
        else if (e.elem_type === 'brace') mat = braceMat;

        // Build elementData compatible with selection system
        const elemData = {
            id: e.id,
            type: e.elem_type || 'beam_x',
            section: e.section || '-',
            ni: e.node_i,
            nj: e.node_j,
        };
        addTube(ni, nj, mat, elemData);
    });

    // Draw nodes
    nodes.forEach(n => {
        const isSupport = n.support === 'fixed' || n.support === 'pinned';
        const mat = isSupport ? supportMat.clone() : nodeMat.clone();
        mat.transparent = true;
        mat.opacity = 0.85;
        const sphere = new THREE.Mesh(nodeGeo, mat);
        sphere.position.set(n.x, n.z, -n.y);
        const nodeData = {
            id: n.id, type: 'node',
            support: n.support || 'free',
            x: n.x, y: n.y, z: n.z,
        };
        sphere.userData.elementData = nodeData;
        sphere.userData._isPreviewElement = true;
        scene.add(sphere);
        previewMeshes.push(sphere);
        memberMeshes.push({ mesh: sphere, elementData: nodeData });

        // Support triangle
        if (isSupport) {
            const cone = new THREE.Mesh(supportGeo, supportMat);
            cone.position.set(n.x, n.z - 0.2, -n.y);
            cone.rotation.x = Math.PI;
            scene.add(cone);
            previewMeshes.push(cone);
        }
    });

    // Node labels
    if (window._labelMode === 'id' || window._labelMode === 'full') {
        nodes.forEach(n => {
            let text;
            if (window._labelMode === 'id') {
                text = 'N' + n.id;
            } else {
                text = 'N' + n.id + '  (' + n.x.toFixed(1) + ', ' + n.y.toFixed(1) + ', ' + n.z.toFixed(1) + ')';
            }
            const sprite = makeTextSprite(text, window._labelMode === 'full');
            sprite.position.set(n.x, n.z + 0.4, -n.y);
            scene.add(sprite);
            previewMeshes.push(sprite);
        });
    }

    // Fit camera (skip during editing to maintain view)
    if (nodes.length > 0 && !skipCameraFit) {
        const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y), zs = nodes.map(n => n.z);
        const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
        const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
        const span = Math.max(
            Math.max(...xs) - Math.min(...xs),
            Math.max(...ys) - Math.min(...ys),
            Math.max(...zs) - Math.min(...zs),
            1
        );
        controls.target.set(cx, cz, -cy);
        camera.position.set(cx + span * 1.5, cz + span * 0.8, -cy + span * 1.5);
        controls.update();
    }

    // Numbers 토글 활성 시 라벨 재빌드
    if (typeof _refreshNumberLabels === 'function') _refreshNumberLabels();
    if (typeof _rebuildStoryCheckboxes === 'function') _rebuildStoryCheckboxes();
}

// ─── Text Sprite Helper (high-res, proper aspect ratio) ─────────────────
function makeTextSprite(text, wide) {
    const dpr = 3;  // high-res
    const fontSize = 14 * dpr;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Measure text width first
    ctx.font = 'bold ' + fontSize + 'px "Segoe UI", Arial, sans-serif';
    const metrics = ctx.measureText(text);
    const textW = metrics.width;
    const pad = 8 * dpr;

    const w = Math.ceil(textW + pad * 2);
    const h = Math.ceil(fontSize + pad * 1.4);
    canvas.width = w;
    canvas.height = h;

    // Re-set font after canvas resize
    ctx.font = 'bold ' + fontSize + 'px "Segoe UI", Arial, sans-serif';

    // Rounded rect background
    const r = 4 * dpr;
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.beginPath();
    ctx.moveTo(r, 0); ctx.lineTo(w-r, 0); ctx.quadraticCurveTo(w,0,w,r);
    ctx.lineTo(w,h-r); ctx.quadraticCurveTo(w,h,w-r,h);
    ctx.lineTo(r,h); ctx.quadraticCurveTo(0,h,0,h-r);
    ctx.lineTo(0,r); ctx.quadraticCurveTo(0,0,r,0);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#90a4ae';
    ctx.lineWidth = 1 * dpr;
    ctx.stroke();

    // Text
    ctx.fillStyle = '#1a237e';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, h / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(spriteMat);

    // Set scale with correct aspect ratio (world units)
    const aspect = w / h;
    const worldH = 0.6;  // label height in meters
    sprite.scale.set(worldH * aspect, worldH, 1);

    return sprite;
}

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
    // Remove preview elements from memberMeshes
    memberMeshes = memberMeshes.filter(({ mesh }) => !mesh.userData._isPreviewElement);
    selectedMeshSet.clear();
    selectedMesh = null;
    // Solid meshes도 함께 정리 (Solid Section이 켜져 있던 경우)
    if (typeof removeSolidMeshes === 'function') removeSolidMeshes();
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

    // Bind section property previews to IFC dropdowns
    ['ifc-col-section', 'ifc-beamx-section', 'ifc-beamy-section'].forEach(id => {
        bindSectionPreview(document.getElementById(id));
    });
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

    // V2 분기로 가기 전에 source marker 확정. 이전엔 _v2Model 체크 후
    // 바로 runAnalysisV2()로 빠져나가서 아래 V1 fallback의 modelSource='IFC'
    // 라인에 도달하지 못해, 첫 IFC V2 분석에서 isIFCSource가 false가 되고
    // IFC 폼의 region/importance가 누락되던 버그.
    modelSource = 'IFC';

    // V2 모델이 있으면 V2 파이프라인 사용
    if (window._v2Model) {
        return await runAnalysisV2();
    }

    // Fallback: V1 방식
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

    if (ifcEditedData.detected_zones && ifcEditedData.detected_zones.length > 0) {
        config.zones = ifcEditedData.detected_zones;
    }

    clearPreviewScene();
    modelSource = 'IFC';
    await runAnalysis(config);
}

// ─── V2 Analysis (Node-Element Pipeline) ─────────────────────────────────
// Pass { rethrow: true } to make the returned Promise REJECT on failure.
// The default behavior (used by existing user-triggered flows) swallows
// the error after showing alert + status — that keeps fire-and-forget
// callers from producing unhandled-rejection noise. The recommendations
// Apply flow opts into rethrow so it can auto-rollback _v2Model.
//
// Pass { skipUndo: true } when the caller has ALREADY captured the
// pre-change undo point (e.g. applyRecDiff snapshots BEFORE swapping
// _v2Model). Without this, the analysis-time pushUndo() would save the
// post-Apply model and Ctrl+Z could not return to the pre-Apply state.
async function runAnalysisV2({ rethrow = false, skipUndo = false } = {}) {
    if (!window._v2Model) {
        if (rethrow) throw new Error('V2 모델이 없습니다.');
        alert('V2 모델이 없습니다.');
        return;
    }

    // 해석 전 undo 포인트 저장 (Ctrl+Z로 해석 전 상태 복원 가능)
    if (!skipUndo && typeof pushUndo === 'function') pushUndo();

    setStatus('V2 해석 중 (KDS Load Gen + Analysis + Design Check)...', 'running');

    // 층별 용도 수집 — IFC 경로는 ifcEditedData + IFC DOM에서, 직접입력/NL은
    // _v2Model 메타에서 derive (applyRecDiff → runAnalysisV2 재해석 경로에서
    // ifcEditedData가 null이라 NPE가 났던 버그 fix).
    const usageRows = document.querySelectorAll('.ifc-usage-sel');
    const storyConfigs = [];
    if (ifcEditedData && Array.isArray(ifcEditedData.stories)) {
        ifcEditedData.stories.forEach((s, i) => {
            storyConfigs.push({
                story: i + 1,
                usage: (i < usageRows.length) ? usageRows[i].value : 'office',
                slab_thickness: 0.15,
                dead_load_finish: 1.0,
            });
        });
    } else {
        const m = window._v2Model;
        const numStories = (m.story_elevations?.length || 1) - 1;
        const su = m.story_usages || {};
        const st = m.story_slab_thickness || {};
        const sdl = m.story_dead_load_finish || {};
        for (let i = 0; i < numStories; i++) {
            const sn = i + 1;
            storyConfigs.push({
                story: sn,
                usage: su[sn] ?? su[String(sn)] ?? 'office',
                slab_thickness: st[sn] ?? st[String(sn)] ?? 0.15,
                dead_load_finish: sdl[sn] ?? sdl[String(sn)] ?? 1.0,
            });
        }
    }

    // env DOM — IFC와 직접입력 탭의 input은 DOM에 둘 다 상존하므로,
    // 모델 소스에 따라 우선 prefix를 정확히 골라야 한다. 잘못 고르면
    // 숨은 다른 탭의 기본값(예: ifc-region=서울)이 적용되어 분석 환경이
    // 조용히 바뀌는 위험이 있음. _v2Model.environment를 1차 source로 쓰고
    // (분석 시점에 확정된 값), DOM은 IFC 탭에서만 우선 적용.
    //
    // isIFCSource: modelSource AND ifcEditedData 두 신호가 모두 IFC를
    // 가리킬 때만 true. ifcEditedData 단독 체크는 "IFC 로드 → 직접입력
    // 분석 → Apply 재해석" 순서에서 stale ifcEditedData가 IFC 분기를
    // 잘못 트리거하던 버그를 막기 위함. modelSource 단독은 분석 직후
    // 재할당 누락(예: runAnalysisV2 성공시) 같은 휴먼 에러에 약함 →
    // AND 조합으로 defense in depth.
    const isIFCSource = (modelSource || '').startsWith('IFC')
        && !!(ifcEditedData && Array.isArray(ifcEditedData.stories));
    const envFromModel = (window._v2Model && window._v2Model.environment) || {};
    const envVal = (ifcId, inputId, modelKey, fallback) => {
        if (isIFCSource) {
            return document.getElementById(ifcId)?.value
                || envFromModel[modelKey]
                || fallback;
        }
        // 직접입력/NL — model env가 진실의 원천. DOM은 마지막 폴백.
        return envFromModel[modelKey]
            || document.getElementById(inputId)?.value
            || fallback;
    };

    const config = {
        region: envVal('ifc-region', 'input-region', 'region', '서울'),
        importance: envVal('ifc-importance', 'input-importance', 'importance', 'II'),
        site_class: envFromModel.site_class || 'S3',
        seismic_system: envFromModel.seismic_system || 'ordinary_moment_frame',
        exposure_category: envFromModel.exposure_category || 'B',
        geometric_nonlinearity: 'linear',
        stories: storyConfigs,
        // Seismic method (ELF / RSA) — DOM only (model meta에 없음)
        seismic_method: isIFCSource
            ? (document.getElementById('ifc-seismic-method')?.value || 'ELF')
            : (document.getElementById('input-seismic-method')?.value || 'ELF'),
        rsa_combination: isIFCSource
            ? (document.getElementById('ifc-rsa-combination')?.value || 'CQC')
            : (document.getElementById('input-rsa-combination')?.value || 'CQC'),
        rsa_direction: isIFCSource
            ? (document.getElementById('ifc-rsa-direction')?.value || '30pct')
            : (document.getElementById('input-rsa-direction')?.value || '30pct'),
    };

    try {
        const resp = await fetch('/api/v2/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: window._v2Model, config, cover_info: window._coverInfo || null }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'V2 해석 실패');
        }

        const result = await resp.json();
        if (result.status !== 'success') throw new Error('V2 해석 실패');

        // 서버에서 split 후 모델로 동기화 (교차점 노드 포함)
        if (result.updated_model) {
            window._v2Model = result.updated_model;
            console.log('[V2] Model synced after split:', window._v2Model.nodes.length, 'nodes,', window._v2Model.elements.length, 'elems');
        }

        // V2 결과를 V1 형식으로 변환하여 기존 UI에 표시
        currentJobId = result.job_id;
        _setExportBtnEnabled(true);
        // modelSource는 호출자가 정한 값을 보존 — 직접입력/NL이 V2 경로를
        // 거치는 경우(Apply, Ctrl+Z 자동 재해석)도 'IFC (V2)'로 덮어쓰면
        // 이후 isIFCSource 판정이 오염됨. IFC 출발일 때만 V2 마커 부착.
        if ((modelSource || '').startsWith('IFC')) {
            modelSource = 'IFC (V2)';
        }

        // V1 buildScene/updateResultsPanel이 기대하는 형식으로 변환
        const v1Result = convertV2ResultToV1(result);
        currentResult = v1Result;

        buildScene(v1Result);
        updateResultsPanel(v1Result);
        updateBottomBar(v1Result);
        if (typeof _refreshNumberLabels === 'function') _refreshNumberLabels();
    if (typeof _rebuildStoryCheckboxes === 'function') _rebuildStoryCheckboxes();

        // 해석 후 편집 비활성화 + 결과 Selection toolbar 표시
        if (typeof showResultSelectionToolbar === 'function') {
            showResultSelectionToolbar();
        } else if (typeof disableEditing === 'function') {
            disableEditing();
        }
        if (typeof removeSolidMeshes === 'function') removeSolidMeshes();
        window.solidMode = false;
        var chkSolid = document.getElementById('chk-solid-section');
        if (chkSolid) chkSolid.checked = false;
        var chkSolidTop = document.getElementById('chk-solid-section-top');
        if (chkSolidTop) chkSolidTop.checked = false;
        setStatus('V2 해석 완료 (KDS + Design Check)', 'success');

    } catch (e) {
        setStatus('V2 해석 실패', 'error');
        if (rethrow) throw e;       // caller owns user-facing feedback
        alert('V2 해석 오류: ' + e.message);
    }
}

function _convertModalV2toV1(modal) {
    if (!modal || !modal.modes) return null;
    return {
        num_modes: modal.num_modes,
        fundamental_periods: modal.fundamental_periods,
        modes: modal.modes.map(m => ({
            mode: m.mode_num || m.mode,
            period_s: m.period_s,
            frequency_hz: m.frequency_Hz || m.frequency_hz,
            direction: m.direction,
            dominance_pct: Math.max(m.mass_participation_x_pct || 0, m.mass_participation_y_pct || 0),
            mass_participation: {
                x_pct: m.mass_participation_x_pct || (m.mass_participation?.x_pct) || 0,
                y_pct: m.mass_participation_y_pct || (m.mass_participation?.y_pct) || 0,
                rz_pct: m.mass_participation_rz_pct || (m.mass_participation?.rz_pct) || 0,
            },
            shape: m.shape,
        })),
    };
}

function convertV2ResultToV1(v2Result) {
    // V2 API 응답 → V1 editor3d.js가 기대하는 형식
    const b = v2Result.building || {};
    const env = v2Result.envelope || {};

    // viewer 노드/요소 (V2 model에서)
    const model = window._v2Model;
    const viewerNodes = (model.nodes || []).map(n => ({
        id: n.id, x: n.x, y: n.y, z: n.z
    }));

    // 요소에서 member_info 생성
    const viewerElements = (model.elements || []).map((e, idx) => ({
        id: e.id, ni: e.node_i, nj: e.node_j,
        type: e.elem_type === 'beam' ? 'beam_x' : e.elem_type,
        section: e.section
    }));

    const memberInfo = viewerElements.map((e, idx) => ({
        member_id: idx + 1, ni: e.ni, nj: e.nj,
        type: e.type, section: e.section,
        length_m: 0, element_ids: [e.id],
    }));

    // case/combo 데이터 변환
    const caseData = {};
    for (const [name, cd] of Object.entries(v2Result.case_data || {})) {
        caseData[name] = {
            summary: cd.summary || {},
            displacements: cd.displacements || {},
            story_drifts: cd.story_drifts || [],
            reactions: cd.reactions || [],
        };
    }

    return {
        job_id: v2Result.job_id,
        status: 'success',
        building: {
            num_stories: b.num_stories || 0,
            total_height_m: b.total_height_m || 0,
            column_section: b.sections_used?.[0] || 'H-300x300',
            beam_x_section: b.sections_used?.[1] || b.sections_used?.[0] || 'H-400x200',
            beam_y_section: b.sections_used?.[1] || b.sections_used?.[0] || 'H-400x200',
            material: b.materials_used?.[0] || 'SS275',
            region: v2Result.config?.region || '',
            is_irregular: false,
        },
        viewer: {
            nodes: viewerNodes,
            elements: viewerElements,
            stories: model.story_elevations?.slice(1)?.map((e, i) =>
                Math.round((e - model.story_elevations[i]) * 100) / 100
            ) || [],
            bays_x: ifcEditedData?.bays_x || [],
            bays_y: ifcEditedData?.bays_y || [],
            total_height: b.total_height_m || 0,
            total_width_x: 0, total_width_y: 0,
            column_section: b.sections_used?.[0] || '',
            beam_x_section: b.sections_used?.[1] || '',
            beam_y_section: b.sections_used?.[1] || '',
            material_name: b.materials_used?.[0] || 'SS275',
            is_irregular: false,
        },
        envelope: {
            max_dx_mm: env.max_dx_mm || 0,
            max_dy_mm: env.max_dy_mm || 0,
            max_dz_mm: env.max_dz_mm || 0,
            max_drift_x: env.max_drift_x || 0,
            max_drift_y: env.max_drift_y || 0,
            max_moment_kNm: env.max_moment_kNm || 0,
            max_axial_kN: env.max_axial_kN || 0,
            max_shear_kN: env.max_shear_kN || 0,
        },
        case_names: v2Result.case_names || [],
        combo_names: v2Result.combo_names || [],
        case_data: caseData,
        member_info: memberInfo,
        design_check: v2Result.design_check,
        interpretation: v2Result.interpretation,
        member_checks: v2Result.member_checks || {},
        report_url: v2Result.report_url,
        modal_analysis: _convertModalV2toV1(v2Result.modal_analysis),
        rsa: v2Result.rsa || null,
        seismic_method: v2Result.seismic_method || 'ELF',
        load_summary: v2Result.load_summary,
        load_cases_raw: v2Result.load_cases_raw || {},
        member_forces: v2Result.member_forces || {},
        member_info_raw: v2Result.member_info || [],
        // Pass through the recommendation block + analysis_id so the
        // Issues & Candidates tab can read them without re-fetching.
        analysis_id: v2Result.analysis_id || v2Result.job_id,
        issues: v2Result.issues || [],
        recommendation_candidates: v2Result.recommendation_candidates || [],
        recommendation_summary: v2Result.recommendation_summary || null,
    };
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

    // Build bay sliders from text values
    buildBaySliders('x');
    buildBaySliders('y');

    // Show live wireframe preview
    updateManualPreview();
}

// ─── Run Analysis ─────────────────────────────────────────────────────────
async function runAnalysis(configOverride = null) {
    let config;
    if (configOverride) {
        config = configOverride;
    } else {
        // 직접입력 진입: 이전 IFC/V2 마커가 남아있으면 강제로 'Manual'로
        // 리셋해서 후속 재해석의 isIFCSource 판정이 오염되지 않도록 한다.
        // runAnalysisFromNL이 미리 'NL'로 세팅한 경우는 보존.
        if (!modelSource || (modelSource || '').startsWith('IFC')) {
            modelSource = 'Manual';
        }
        const stories = getStoriesFromEditor();
        const isIrregular = document.getElementById('irregular-toggle')?.checked;

        let bays_x, bays_y, zones;
        if (isIrregular) {
            zones = getZonesFromEditor();
            if (zones.length === 0) {
                alert('비정형: 최소 1개 존을 정의해주세요.');
                return;
            }
            // bays_x/bays_y = first zone's (for fallback)
            bays_x = zones[0].bays_x;
            bays_y = zones[0].bays_y;
        } else {
            bays_x = document.getElementById('input-bays-x').value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
            bays_y = document.getElementById('input-bays-y').value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
        }

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
            seismic_method: document.getElementById('input-seismic-method')?.value || 'ELF',
            rsa_combination: document.getElementById('input-rsa-combination')?.value || 'CQC',
            rsa_direction: document.getElementById('input-rsa-direction')?.value || '30pct',
        };
        if (isIrregular && zones) {
            config.zones = zones;
        }
    }

    // 해석 전 undo 포인트 저장
    if (window._v2Model && typeof pushUndo === 'function') pushUndo();

    showLoading('Analyzing...');
    setStatus('Analyzing...', 'running');

    try {
        // V2 경로 사용: 직접입력/NL도 IFC와 동일하게 full context(model_json +
        // candidates_by_id)를 캐시해서 Phase B 단면 변경 명령과 추천 탭이
        // 메인 탭에서도 동작하도록 한다. 백엔드는 body에 model이 없으면
        // config로부터 StructuralModel을 자동 생성한다.
        const response = await fetch('/api/v2/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config, cover_info: window._coverInfo || null }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || 'Analysis failed');
        }

        const v2Result = await response.json();
        if (v2Result.status && v2Result.status !== 'success') {
            throw new Error('V2 해석 실패');
        }

        // updated_model을 window._v2Model에 저장해서 후속 채팅 명령
        // (propose_section_change → diff modal Apply)이 동작하도록 한다.
        if (v2Result.updated_model) {
            window._v2Model = v2Result.updated_model;
        }

        // V2 응답을 V1 editor가 기대하는 형식으로 변환 (IFC 경로와 동일)
        const result = convertV2ResultToV1(v2Result);
        currentJobId = result.job_id;
        _setExportBtnEnabled(true);
        currentResult = result;

        buildScene(result);
        updateResultsPanel(result);
        updateBottomBar(result);
        if (typeof _refreshNumberLabels === 'function') _refreshNumberLabels();
        if (typeof _rebuildStoryCheckboxes === 'function') _rebuildStoryCheckboxes();

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
    let memberType = null;
    if (selectedMesh && selectedMesh.userData.elementData) {
        const elemType = selectedMesh.userData.elementData.type;
        if (elemType === 'column') { modifications.column_section = newSection; memberType = 'column'; }
        else if (elemType === 'beam_x') { modifications.beam_x_section = newSection; memberType = 'beam_x'; }
        else if (elemType === 'beam_y') { modifications.beam_y_section = newSection; memberType = 'beam_y'; }
    }

    if (Object.keys(modifications).length === 0) {
        alert('Select a member first.');
        return;
    }

    // V2 경로
    if (window._v2Model) {
        // Update config panel
        if (modifications.column_section) setSelectValue(document.getElementById('input-col-section'), modifications.column_section);
        if (modifications.beam_x_section) setSelectValue(document.getElementById('input-beamx-section'), modifications.beam_x_section);
        if (modifications.beam_y_section) setSelectValue(document.getElementById('input-beamy-section'), modifications.beam_y_section);
        return await _applyV2SectionAndReanalyze(modifications, memberType);
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
        _setExportBtnEnabled(true);
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

    // V2 모델이 있으면 V2 경로로 단면 변경 + 재해석
    if (window._v2Model) {
        return await _applyV2SectionAndReanalyze(modifications, memberType);
    }

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
        _setExportBtnEnabled(true);
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

async function _applyV2SectionAndReanalyze(modifications, memberType) {
    const typeLabel = { column: '기둥', beam_x: 'X보', beam_y: 'Y보' }[memberType] || '부재';
    const model = window._v2Model;
    if (!model || !model.elements) return;

    // V2 모델 요소 단면 일괄 변경
    let changed = 0;
    model.elements.forEach(e => {
        if (modifications.column_section && e.elem_type === 'column') {
            e.section = modifications.column_section; changed++;
        }
        if (modifications.beam_x_section && (e.elem_type === 'beam' || e.elem_type === 'beam_x')) {
            e.section = modifications.beam_x_section; changed++;
        }
        if (modifications.beam_y_section && e.elem_type === 'beam_y') {
            e.section = modifications.beam_y_section; changed++;
        }
    });

    if (changed === 0) {
        alert('해당 타입 요소가 없습니다.');
        return;
    }

    console.log(`[V2] Section change: ${typeLabel} → ${changed} elements updated`);
    setStatus(`${typeLabel} ${changed}개 변경 → 재해석 중...`, 'running');

    // V2 재해석
    await runAnalysisV2();
}

async function applySingleMemberChange() {
    const newSection = document.getElementById('prop-new-section')?.value;
    if (!newSection) return;
    if (!window._v2Model || !selectedMesh?.userData?.elementData) {
        alert('부재를 선택해주세요.');
        return;
    }

    const elemId = selectedMesh.userData.elementData.id;
    const model = window._v2Model;

    // V2 모델에서 해당 요소 단면 변경
    const elem = model.elements.find(e => e.id === elemId);
    if (elem) {
        elem.section = newSection;
        console.log(`[V2] Element E${elemId} section → ${newSection}`);
        setStatus(`E${elemId} → ${newSection}, 재해석 중...`, 'running');
        await runAnalysisV2();
    }
}

// ─── 층별 단면 일괄 변경 ──────────────────────────────────────────────────

const _PREVIEW_COLOR = 0xff9800;  // 층 체크 프리뷰 색 (주황)
let _storyPreviewMeshes = new Set();  // 프리뷰 하이라이트 적용된 mesh

/** 부재의 대표 층 (하위 층 기준: 기둥은 하단 노드 story, 보는 그 층) */
function _elemStoryOf(e, nodeMap) {
    const ni = nodeMap[e.node_i], nj = nodeMap[e.node_j];
    const si = ni?.story, sj = nj?.story;
    if (si == null && sj == null) return null;
    if (si == null) return sj;
    if (sj == null) return si;
    return Math.min(si, sj);
}

function _clearStoryPreview() {
    _storyPreviewMeshes.forEach(mesh => {
        const orig = mesh.userData._previewOrigColor;
        if (orig !== undefined) {
            mesh.material.color.setHex(orig);
            delete mesh.userData._previewOrigColor;
        }
    });
    _storyPreviewMeshes.clear();
}

function _updateStoryPreview() {
    // 기존 프리뷰 해제
    _clearStoryPreview();

    const sel = selectedMesh?.userData?.elementData;
    if (!sel) return;
    let elemType = sel.type;
    if (elemType === 'beam') elemType = 'beam_x';

    const checkedStories = Array.from(
        document.querySelectorAll('#modify-story-checks input[data-story-sel]:checked')
    ).map(c => parseInt(c.dataset.storySel, 10));
    if (checkedStories.length === 0) return;

    const model = window._v2Model;
    if (!model) return;
    const nodeMap = {};
    model.nodes.forEach(n => { nodeMap[n.id] = n; });

    // 매칭되는 element id 수집
    const matchIds = new Set();
    model.elements.forEach(e => {
        let et = e.elem_type;
        if (et === 'beam') et = 'beam_x';
        if (et !== elemType) return;
        const story = _elemStoryOf(e, nodeMap);
        if (story != null && checkedStories.includes(story)) {
            matchIds.add(e.id);
        }
    });

    // 해당 mesh에 프리뷰 색 적용 (선택된 mesh는 제외)
    memberMeshes.forEach(({ mesh, elementData }) => {
        if (!elementData || elementData.type === 'node') return;
        if (!matchIds.has(elementData.id)) return;
        if (selectedMeshSet.has(mesh)) return;  // 선택 색 유지
        if (mesh.userData._previewOrigColor === undefined) {
            mesh.userData._previewOrigColor = mesh.material.color.getHex();
        }
        mesh.material.color.setHex(_PREVIEW_COLOR);
        _storyPreviewMeshes.add(mesh);
    });

    // Solid mesh에도 프리뷰 적용
    if (window._solidMeshMap) {
        matchIds.forEach(eid => {
            const sm = window._solidMeshMap[eid];
            if (!sm) return;
            if (sm.userData._previewOrigColor === undefined) {
                sm.userData._previewOrigColor = sm.material.color.getHex();
            }
            sm.material.color.setHex(_PREVIEW_COLOR);
            _storyPreviewMeshes.add(sm);
        });
    }
}

function _rebuildStoryCheckboxes() {
    const wrap = document.getElementById('modify-story-checks');
    if (!wrap) return;
    const model = window._v2Model;
    if (!model || !model.nodes) {
        wrap.innerHTML = '<span style="color:#999;">(모델 없음)</span>';
        _clearStoryPreview();
        return;
    }
    const stories = new Set();
    model.nodes.forEach(n => { if (n.story != null) stories.add(n.story); });
    const sorted = Array.from(stories).sort((a, b) => a - b);
    if (sorted.length === 0) {
        wrap.innerHTML = '<span style="color:#999;">(층 정보 없음)</span>';
        _clearStoryPreview();
        return;
    }
    // 기존 체크 상태 보존
    const prevChecked = new Set(
        Array.from(wrap.querySelectorAll('input[data-story-sel]:checked'))
            .map(c => parseInt(c.dataset.storySel, 10))
    );
    wrap.innerHTML = sorted.map(s => {
        const c = prevChecked.has(s) ? ' checked' : '';
        return `<label style="display:flex; align-items:center; gap:2px; cursor:pointer;"><input type="checkbox" data-story-sel="${s}" onchange="_updateStoryPreview()"${c}> ${s}F</label>`;
    }).join('');
    _updateStoryPreview();
}

function storySelAll(on) {
    document.querySelectorAll('#modify-story-checks input[data-story-sel]').forEach(c => c.checked = on);
    _updateStoryPreview();
}

function storySelCurrent() {
    const elem = selectedMesh?.userData?.elementData;
    if (!elem) { alert('부재를 먼저 선택하세요.'); return; }
    const model = window._v2Model;
    if (!model) return;
    const nodeMap = {};
    model.nodes.forEach(n => { nodeMap[n.id] = n; });
    // elem.ni / elem.nj 는 buildScene/preview에서 저장된 노드 ID
    const fakeEl = { node_i: elem.ni, node_j: elem.nj };
    const story = _elemStoryOf(fakeEl, nodeMap);
    storySelAll(false);
    if (story != null) {
        const cb = document.querySelector(`#modify-story-checks input[data-story-sel="${story}"]`);
        if (cb) cb.checked = true;
    }
    _updateStoryPreview();
}

async function applyStorySection() {
    if (!currentJobId) { alert('먼저 해석을 실행해주세요.'); return; }
    const newSection = document.getElementById('prop-new-section')?.value;
    if (!newSection) return;
    if (!selectedMesh?.userData?.elementData) {
        alert('부재를 먼저 선택하세요.');
        return;
    }

    const sel = selectedMesh.userData.elementData;
    let elemType = sel.type;
    if (elemType === 'beam') elemType = 'beam_x';

    const checkedStories = Array.from(
        document.querySelectorAll('#modify-story-checks input[data-story-sel]:checked')
    ).map(c => parseInt(c.dataset.storySel, 10));

    if (checkedStories.length === 0) {
        alert('적용할 층을 하나 이상 체크하세요.');
        return;
    }

    const model = window._v2Model;
    if (!model) return;
    const nodeMap = {};
    model.nodes.forEach(n => { nodeMap[n.id] = n; });

    let changed = 0;
    model.elements.forEach(e => {
        let et = e.elem_type;
        if (et === 'beam') et = 'beam_x';
        if (et !== elemType) return;
        const story = _elemStoryOf(e, nodeMap);
        if (story == null) return;
        if (checkedStories.includes(story)) {
            e.section = newSection;
            changed++;
        }
    });

    if (changed === 0) {
        alert('해당 조건(타입+층)에 맞는 부재가 없습니다.');
        return;
    }

    const typeLabel = { column: '기둥', beam_x: 'X보', beam_y: 'Y보' }[elemType] || elemType;
    console.log(`[V2] Story-filtered section change: ${typeLabel} @ ${checkedStories.join(',')}F → ${changed} elements`);
    setStatus(`${typeLabel} ${changed}개 변경 (층 ${checkedStories.join(',')}) → 재해석 중...`, 'running');
    _clearStoryPreview();  // 재해석 전 프리뷰 해제 (씬 재빌드됨)
    await runAnalysisV2();
}

// ─── Build 3D Scene ───────────────────────────────────────────────────────
function buildScene(result) {
    // Clear preview wireframe (IFC wizard) if present
    clearPreviewScene();
    // Clear V2 solid meshes if present
    if (typeof removeSolidMeshes === 'function') removeSolidMeshes();

    // Clear existing members
    memberMeshes.forEach(m => scene.remove(m.mesh));
    nodeMeshes.forEach(m => scene.remove(m));
    memberMeshes = [];
    nodeMeshes = [];
    selectedMesh = null;
    selectedMeshSet.clear();

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

        // Three.js: Y=up, so map (x, z, -y) — consistent with V2 preview
        const start = new THREE.Vector3(ni.x, ni.z, -ni.y);
        const end = new THREE.Vector3(nj.x, nj.z, -nj.y);

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

    // Draw all nodes (supports highlighted, others more visible for deformed shape)
    nodes.forEach(n => {
        const isSupport = Math.abs(n.z) < 0.01;
        const geo = new THREE.SphereGeometry(isSupport ? 0.2 : 0.15, 10, 10);
        const mat = new THREE.MeshPhongMaterial({
            color: isSupport ? 0xff6600 : 0x555555,
            transparent: !isSupport,
            opacity: isSupport ? 1.0 : 0.85,
        });
        const sphere = new THREE.Mesh(geo, mat);
        sphere.position.set(n.x, n.z, -n.y);
        sphere.userData.nodeId = n.id;
        sphere.userData.elementData = { id: n.id, type: 'node', x: n.x, y: n.y, z: n.z };
        scene.add(sphere);
        nodeMeshes.push(sphere);
        memberMeshes.push({ mesh: sphere, elementData: sphere.userData.elementData });
    });

    // Reset deformed shape state
    originalNodePositions = null;

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
    if (elem.type === 'node') return elem.support && elem.support !== 'free' ? 0xff6600 : COLORS.node;
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

// ─── 3D Hover Tooltip ─────────────────────────────────────────────────────
let hoverTooltipEl = null;
let hoveredMesh = null;

function getHoverTooltip() {
    if (!hoverTooltipEl) {
        hoverTooltipEl = document.createElement('div');
        hoverTooltipEl.id = 'member-hover-tooltip';
        document.body.appendChild(hoverTooltipEl);
    }
    return hoverTooltipEl;
}

function onCanvasHover(event) {
    if (memberMeshes.length === 0) return;

    const container = document.getElementById('viewer-container');
    const rect = container.getBoundingClientRect();
    const mx = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const my = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);
    const meshes = memberMeshes.map(m => m.mesh);
    const intersects = raycaster.intersectObjects(meshes);

    if (intersects.length > 0) {
        const mesh = intersects[0].object;
        const elem = mesh.userData.elementData;
        if (!elem) { hideHoverTooltip(); return; }

        // Change cursor
        container.style.cursor = 'pointer';

        // Only update content if hovered member changed
        if (hoveredMesh !== mesh) {
            hoveredMesh = mesh;
            showHoverTooltip(elem, event.clientX, event.clientY);
        } else {
            // Just update position
            positionHoverTooltip(event.clientX, event.clientY);
        }
    } else {
        container.style.cursor = '';
        if (hoveredMesh) {
            hoveredMesh = null;
            hideHoverTooltip();
        }
    }
}

async function showHoverTooltip(elem, cx, cy) {
    const tooltip = getHoverTooltip();
    const typeLabel = elem.type === 'column' ? 'COLUMN' : elem.type === 'beam_x' ? 'BEAM X' : 'BEAM Y';

    // Show immediately with basic info
    tooltip.innerHTML = `<div class="sp-member-type ${elem.type}">${typeLabel} #${elem.id}</div>`
        + `<div class="sp-title">${elem.section}</div>`
        + `<div class="sp-loading">Loading...</div>`;
    positionHoverTooltip(cx, cy);
    tooltip.classList.add('visible');

    // Fetch section properties
    const data = await fetchSectionProps(elem.section);
    if (hoveredMesh && data && !data.error) {
        // 단면 타입별 표시 필드 선택
        const sType = data.section_type || (typeof _detectSectionType === 'function' ? _detectSectionType(elem.section) : 'H');
        let rows;
        if (sType === 'SHS' || sType === 'RHS') {
            rows = [
                ['A',  data.A_cm2,  'cm\u00B2'],
                ['Ix', data.Ix_cm4, 'cm\u2074'],
                ['Iy', data.Iy_cm4, 'cm\u2074'],
                ['J',  data.J_cm4,  'cm\u2074'],
                ['B',  data.b_mm || data.h_mm, 'mm'],
                ['t',  data.t_mm || data.tw_mm, 'mm'],
            ];
            if (sType === 'RHS') {
                rows.splice(4, 0, ['H', data.h_mm, 'mm']);
            }
        } else if (sType === 'CHS') {
            rows = [
                ['A',  data.A_cm2,  'cm\u00B2'],
                ['I',  data.Ix_cm4, 'cm\u2074'],
                ['J',  data.J_cm4,  'cm\u2074'],
                ['D',  data.h_mm,   'mm'],
                ['t',  data.t_mm || data.tw_mm, 'mm'],
            ];
        } else {
            rows = [
                ['A',  data.A_cm2,  'cm\u00B2'],
                ['Ix', data.Ix_cm4, 'cm\u2074'],
                ['Iy', data.Iy_cm4, 'cm\u2074'],
                ['J',  data.J_cm4,  'cm\u2074'],
                ['H',  data.h_mm,   'mm'],
                ['B',  data.b_mm,   'mm'],
            ];
        }
        let grid = rows
            .filter(r => r[1] > 0)
            .map(([k, v, u]) => `<span class="sp-key">${k}</span><span class="sp-val">${v.toLocaleString()}</span><span class="sp-unit">${u}</span>`)
            .join('');
        // 부재력 정보 추가
        let forceHtml = '';
        if (currentResult?.member_forces) {
            const caseName = _getCurrentCaseName();
            const mfCase = currentResult.member_forces[caseName];
            if (mfCase) {
                const mf = mfCase.find(m => m.member_id === elem.id || m.ni === elem.ni && m.nj === elem.nj);
                if (mf) {
                    const maxN = Math.max(...(mf.N_kN||[0]).map(Math.abs));
                    const maxV = Math.max(...(mf.Vy_kN||[0]).map(Math.abs), ...(mf.Vz_kN||[0]).map(Math.abs));
                    const maxM = Math.max(...(mf.My_kNm||[0]).map(Math.abs), ...(mf.Mz_kNm||[0]).map(Math.abs), ...(mf.T_kNm||[0]).map(Math.abs));
                    forceHtml = `<div style="border-top:1px solid #555;margin-top:4px;padding-top:3px;font-size:10px;">`
                        + `<div style="color:#aaa;margin-bottom:2px;">${caseName}</div>`
                        + `<div class="sp-grid">`
                        + `<span class="sp-key">N</span><span class="sp-val">${maxN.toFixed(1)}</span><span class="sp-unit">kN</span>`
                        + `<span class="sp-key">V</span><span class="sp-val">${maxV.toFixed(1)}</span><span class="sp-unit">kN</span>`
                        + `<span class="sp-key">M</span><span class="sp-val">${maxM.toFixed(1)}</span><span class="sp-unit">kN·m</span>`
                        + `</div></div>`;
                }
            }
        }

        tooltip.innerHTML = `<div class="sp-member-type ${elem.type}">${typeLabel} #${elem.id}</div>`
            + `<div class="sp-title">${data.name}</div>`
            + `<div class="sp-grid">${grid}</div>`
            + forceHtml;
    }
}

function positionHoverTooltip(cx, cy) {
    const tooltip = getHoverTooltip();
    tooltip.style.left = (cx + 16) + 'px';
    tooltip.style.top = (cy - 10) + 'px';
}

function hideHoverTooltip() {
    const tooltip = getHoverTooltip();
    tooltip.classList.remove('visible');
}

// ─── Multi-Selection + Box Selection ─────────────────────────────────────
let _boxSelect = { active: false, startX: 0, startY: 0, el: null, didDrag: false };
const BOX_DRAG_THRESHOLD = 5; // px — distinguish click from drag

function highlightMesh(mesh) {
    if (!mesh.userData._origColor) {
        mesh.userData._origColor = mesh.material.color.getHex();
    }
    mesh.material.color.setHex(COLORS.selected);
    mesh.material.opacity = 1.0;
    selectedMeshSet.add(mesh);
    // Solid mode: 해당 부재의 solid mesh도 하이라이트
    _highlightSolidMesh(mesh.userData.elementData?.id, true);
}

function unhighlightMesh(mesh) {
    const orig = mesh.userData._origColor;
    if (orig !== undefined) {
        mesh.material.color.setHex(orig);
    } else {
        mesh.material.color.setHex(getElementColor(mesh.userData.elementData));
    }
    mesh.material.opacity = 0.85;
    delete mesh.userData._origColor;
    selectedMeshSet.delete(mesh);
    // Solid mode: 해당 부재의 solid mesh 원복
    _highlightSolidMesh(mesh.userData.elementData?.id, false);
}

function _highlightSolidMesh(elemId, on) {
    if (!elemId || !window._solidMeshMap) return;
    const sm = window._solidMeshMap[elemId];
    if (!sm) return;
    if (on) {
        sm.material.color.setHex(COLORS.selected);
        sm.material.emissive = new THREE.Color(0x330011);
    } else {
        const orig = sm.userData._solidOrigColor;
        if (orig !== undefined) sm.material.color.setHex(orig);
        sm.material.emissive = new THREE.Color(0x000000);
    }
}

function clearAllSelection() {
    if (typeof _clearStoryPreview === 'function') _clearStoryPreview();
    selectedMeshSet.forEach(m => unhighlightMesh(m));
    selectedMeshSet.clear();
    selectedMesh = null;
    hideMemberProperties();
    updateSelectionCount();
}

function updateSelectionCount() {
    // Bottom badge on viewer
    let el = document.getElementById('selection-count');
    if (!el) {
        el = document.createElement('div');
        el.id = 'selection-count';
        el.className = 'selection-count';
        document.getElementById('viewer-container').appendChild(el);
    }
    const n = selectedMeshSet.size;
    // Count by type
    const counts = {};
    selectedMeshSet.forEach(m => {
        const t = m.userData.elementData?.type || 'unknown';
        counts[t] = (counts[t] || 0) + 1;
    });
    const parts = Object.entries(counts).map(([t, c]) => `${t.replace('_',' ')}: ${c}`);

    if (n === 0) {
        el.style.display = 'none';
    } else {
        el.textContent = `${n} selected (${parts.join(', ')})`;
        el.style.display = 'block';
    }

    // Left panel "Selected" section
    const toolSel = document.getElementById('tool-selected-info');
    if (toolSel) {
        if (n === 0) {
            toolSel.className = 'tool-selected-empty';
            toolSel.innerHTML = 'Use <b>Select</b> mode — click or drag to select members';
        } else {
            toolSel.className = 'tool-selected-active';
            toolSel.innerHTML = `<b>${n}</b> members selected<br><small>${parts.join(', ')}</small>`;
        }
    }

    // Right panel: switch between single / multi / empty
    // But preserve prop-results visibility (analysis results always stay visible)
    if (n === 0) {
        document.getElementById('prop-member').style.display = 'none';
        document.getElementById('prop-node').style.display = 'none';
        document.getElementById('prop-multi').style.display = 'none';
        // Only show prop-empty if no analysis results are displayed
        const hasResults = document.getElementById('prop-results')?.style.display === 'block';
        document.getElementById('prop-empty').style.display = hasResults ? 'none' : 'block';
    } else if (n === 1) {
        const mesh = [...selectedMeshSet][0];
        if (mesh?.userData?.elementData) {
            document.getElementById('prop-multi').style.display = 'none';
            showMemberProperties(mesh.userData.elementData);
        }
    } else {
        showMultiSelectionPanel();
    }
}

function onSelectionKeyDown(e) {
    if (e.key === 'Delete' && selectedMeshSet.size > 0 && isSelectMode()) {
        e.preventDefault();
        bulkDeleteSelected();
    }
}

// ─── Selection Filter ────────────────────────────────────────────────────
function getSelectionFilter() {
    const el = document.getElementById('select-filter');
    return el ? el.value : 'all';
}

function meshPassesFilter(mesh) {
    const filter = getSelectionFilter();
    if (filter === 'all') return true;
    const d = mesh.userData.elementData;
    if (!d) return false;
    if (filter === 'nodes') return d.type === 'node';
    if (filter === 'elements') return d.type !== 'node';
    if (filter === 'column') return d.type === 'column';
    if (filter === 'beam') return d.type === 'beam' || d.type === 'beam_x' || d.type === 'beam_y';
    return true;
}

// ─── Story Selection ─────────────────────────────────────────────────────
function populateStorySelector() {
    const sel = document.getElementById('select-story');
    if (!sel || !window._v2Model) return;
    sel.innerHTML = '<option value="">Story...</option>';
    const elevations = window._v2Model.story_elevations || [];
    elevations.forEach((z, i) => {
        const label = i === 0 ? `Base (${z}m)` : `${i}F (${z}m)`;
        sel.innerHTML += `<option value="${i}">${label}</option>`;
    });
    // Add "All" option
    sel.innerHTML += '<option value="all">All Stories</option>';
}

function selectByStory(storyIdx) {
    if (storyIdx === '' || !window._v2Model) return;

    const model = window._v2Model;
    const nodeMap = {};
    model.nodes.forEach(n => { nodeMap[n.id] = n; });
    const targetStory = storyIdx === 'all' ? null : parseInt(storyIdx);

    // Select matching meshes
    memberMeshes.forEach(({ mesh }) => {
        if (!meshPassesFilter(mesh)) return;
        const d = mesh.userData.elementData;
        if (!d) return;

        let match = false;

        if (targetStory === null) {
            // All stories
            match = true;
        } else if (d.type === 'node') {
            // 노드: story 속성으로 직접 판별
            const node = nodeMap[d.id];
            match = node && node.story === targetStory;
        } else {
            // 요소: 상단 노드(max story)로 소속 층 판별
            const ni = nodeMap[d.ni], nj = nodeMap[d.nj];
            if (ni && nj) {
                const memberStory = Math.max(ni.story || 0, nj.story || 0);
                if (targetStory === 0) {
                    // Base: story=0인 노드에 연결된 기둥의 하단만 (지점 노드)
                    // → 실제로는 story=0 노드만 선택 (요소는 1F에 속함)
                    match = false;
                } else {
                    match = memberStory === targetStory;
                }
            }
        }
        if (match && !selectedMeshSet.has(mesh)) highlightMesh(mesh);
    });

    // Base(0) 선택 시 노드 mesh도 검사
    if (targetStory === 0) {
        nodeMeshes.forEach(m => {
            const nid = m.userData?.nodeId;
            if (!nid) return;
            const node = nodeMap[nid];
            if (node && node.story === 0 && !selectedMeshSet.has(m)) {
                highlightMesh(m);
            }
        });
    }

    updateSelectionCount();
    document.getElementById('select-story').value = '';
}

// --- Mouse events for click + box drag ---
function isSelectionAllowed() {
    // Selection works in: view mode, select mode, or when editing is disabled
    if (!window._editingEnabled) return true;
    if (typeof editMode === 'undefined') return true;
    return editMode === 'view' || editMode === 'select';
}

function isSelectMode() {
    return typeof editMode !== 'undefined' && editMode === 'select';
}

function onCanvasMouseDown(event) {
    if (event.button !== 0) return;
    if (!isSelectionAllowed()) return;
    const container = document.getElementById('viewer-container');
    const rect = container.getBoundingClientRect();
    _boxSelect.startX = event.clientX;
    _boxSelect.startY = event.clientY;
    _boxSelect.didDrag = false;
    _boxSelect.rect = rect;
    _boxSelect.allowBox = isSelectMode();
    // Capture pointer so we get move/up even if cursor leaves canvas
    if (_boxSelect.allowBox) {
        event.target.setPointerCapture(event.pointerId);
    }
}

function onCanvasMouseMove(event) {
    // Hover tooltip
    if (!_boxSelect.active) onCanvasHover(event);

    // Box selection drag detection
    if (event.buttons !== 1 || !isSelectionAllowed()) return;
    const dx = event.clientX - _boxSelect.startX;
    const dy = event.clientY - _boxSelect.startY;
    if (!_boxSelect.active && _boxSelect.allowBox && (Math.abs(dx) > BOX_DRAG_THRESHOLD || Math.abs(dy) > BOX_DRAG_THRESHOLD)) {
        // Disable orbit controls during box select
        controls.enabled = false;
        _boxSelect.active = true;
        if (!_boxSelect.el) {
            _boxSelect.el = document.createElement('div');
            _boxSelect.el.id = 'box-select-rect';
            _boxSelect.el.className = 'box-select-rect';
            document.getElementById('viewer-container').appendChild(_boxSelect.el);
        }
        _boxSelect.el.style.display = 'block';
    }
    if (_boxSelect.active) {
        _boxSelect.didDrag = true;
        const r = _boxSelect.rect;
        const x1 = Math.min(_boxSelect.startX, event.clientX) - r.left;
        const y1 = Math.min(_boxSelect.startY, event.clientY) - r.top;
        const w = Math.abs(dx);
        const h = Math.abs(dy);
        const el = _boxSelect.el;
        el.style.left = x1 + 'px';
        el.style.top = y1 + 'px';
        el.style.width = w + 'px';
        el.style.height = h + 'px';
        // Left→Right = Window (blue), Right→Left = Crossing (green dashed)
        if (dx >= 0) {
            el.className = 'box-select-rect box-window';
        } else {
            el.className = 'box-select-rect box-crossing';
        }
    }
}

function onCanvasMouseUpWindow(event) {
    // Only handles finishing box select when pointer released outside canvas
    if (event.button !== 0) return;
    if (_boxSelect.active && _boxSelect.didDrag) {
        controls.enabled = true;
        finishBoxSelect(event);
        cancelBoxSelect();
    }
}

function onCanvasMouseUp(event) {
    if (event.button !== 0) return;

    if (_boxSelect.active && _boxSelect.didDrag) {
        controls.enabled = true;
        finishBoxSelect(event);
        cancelBoxSelect();
        return;
    }
    cancelBoxSelect();

    if (!isSelectionAllowed()) return;

    // Normal click — raycaster pick (use canvas rect, not container)
    const canvasEl = renderer.domElement;
    const rect = canvasEl.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const meshes = memberMeshes.map(m => m.mesh);
    const intersects = raycaster.intersectObjects(meshes);

    if (event.ctrlKey) {
        // Ctrl+click: toggle in multi-selection
        if (intersects.length > 0) {
            const mesh = intersects[0].object;
            if (selectedMeshSet.has(mesh)) {
                unhighlightMesh(mesh);
                if (selectedMesh === mesh) selectedMesh = null;
            } else if (meshPassesFilter(mesh)) {
                highlightMesh(mesh);
                selectedMesh = mesh;
            }
        }
    } else {
        // Normal click: clear all, select one
        clearAllSelection();
        if (intersects.length > 0) {
            const mesh = intersects[0].object;
            if (meshPassesFilter(mesh)) {
                highlightMesh(mesh);
                selectedMesh = mesh;
                showMemberProperties(mesh.userData.elementData);
            }
        }
    }
    // Show properties for primary selection
    if (selectedMesh) {
        showMemberProperties(selectedMesh.userData.elementData);
    }
    updateSelectionCount();
}

function cancelBoxSelect() {
    _boxSelect.active = false;
    _boxSelect.didDrag = false;
    if (_boxSelect.el) _boxSelect.el.style.display = 'none';
}

function finishBoxSelect(event) {
    // Use canvas (renderer) rect for accurate screen projection matching
    const canvasRect = renderer.domElement.getBoundingClientRect();
    const x1 = (Math.min(_boxSelect.startX, event.clientX) - canvasRect.left) / canvasRect.width;
    const y1 = (Math.min(_boxSelect.startY, event.clientY) - canvasRect.top) / canvasRect.height;
    const x2 = (Math.max(_boxSelect.startX, event.clientX) - canvasRect.left) / canvasRect.width;
    const y2 = (Math.max(_boxSelect.startY, event.clientY) - canvasRect.top) / canvasRect.height;
    const isWindow = (event.clientX - _boxSelect.startX) >= 0; // L→R = window

    if (!event.ctrlKey) clearAllSelection();

    memberMeshes.forEach(({ mesh }) => {
        const geo = mesh.geometry;
        let points;

        if (geo.parameters && geo.parameters.height) {
            // Element (CylinderGeometry): project endpoints + midpoint
            const pos = mesh.position.clone();
            const halfH = geo.parameters.height / 2;
            const up = new THREE.Vector3(0, 1, 0).applyQuaternion(mesh.quaternion);
            points = [
                pos.clone().add(up.clone().multiplyScalar(halfH)),
                pos.clone().add(up.clone().multiplyScalar(-halfH)),
                pos.clone()
            ];
        } else {
            // Node (SphereGeometry): project center point only
            points = [mesh.position.clone()];
        }

        let allInside = true, anyInside = false;
        points.forEach(p => {
            p.project(camera);
            const sx = (p.x + 1) / 2;
            const sy = (1 - p.y) / 2;
            const inside = sx >= x1 && sx <= x2 && sy >= y1 && sy <= y2;
            if (inside) anyInside = true; else allInside = false;
        });

        const hit = isWindow ? allInside : anyInside;
        if (hit && meshPassesFilter(mesh)) highlightMesh(mesh);
    });
    updateSelectionCount();
}

// ─── Property Panel ───────────────────────────────────────────────────────
function showMemberProperties(elem) {
    // 부재 변경 시 이전 층 프리뷰 해제 (타입이 바뀔 수 있음)
    if (typeof _clearStoryPreview === 'function') _clearStoryPreview();
    document.getElementById('prop-empty').style.display = 'none';
    document.getElementById('prop-node').style.display = 'none';
    document.getElementById('prop-member').style.display = 'none';
    document.getElementById('prop-multi').style.display = 'none';

    // Node: show node-specific panel
    if (elem.type === 'node') {
        document.getElementById('prop-node').style.display = 'block';
        document.getElementById('prop-node-id').textContent = '#' + elem.id;
        document.getElementById('prop-node-x').textContent = (elem.x ?? 0).toFixed(2) + ' m';
        document.getElementById('prop-node-y').textContent = (elem.y ?? 0).toFixed(2) + ' m';
        document.getElementById('prop-node-z').textContent = (elem.z ?? 0).toFixed(2) + ' m';
        document.getElementById('prop-node-support').textContent = elem.support || 'free';
        return;
    }

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

    // 개별 부재 SFD/BMD 다이어그램
    if (typeof drawMemberDiagrams === 'function') {
        drawMemberDiagrams(elem.member_id || elem.id);
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

    // 층 체크박스 빌드/갱신
    if (typeof _rebuildStoryCheckboxes === 'function') _rebuildStoryCheckboxes();
}

function hideMemberProperties() {
    document.getElementById('prop-node').style.display = 'none';
    document.getElementById('prop-member').style.display = 'none';
    document.getElementById('prop-multi').style.display = 'none';
    // Only show empty hint if no analysis results
    const hasResults = document.getElementById('prop-results')?.style.display === 'block';
    document.getElementById('prop-empty').style.display = hasResults ? 'none' : 'block';
}

// ─── Multi-Selection Property Panel ──────────────────────────────────────
function showMultiSelectionPanel() {
    document.getElementById('prop-empty').style.display = 'none';
    document.getElementById('prop-member').style.display = 'none';
    const panel = document.getElementById('prop-multi');
    panel.style.display = 'block';

    // Gather selected element data
    const elems = [];
    const nodes = [];
    selectedMeshSet.forEach(m => {
        const d = m.userData.elementData;
        if (!d) return;
        if (d.type === 'node') nodes.push(d);
        else elems.push(d);
    });

    // Count header
    const parts = [];
    if (elems.length > 0) parts.push(elems.length + ' elements');
    if (nodes.length > 0) parts.push(nodes.length + ' nodes');
    document.getElementById('prop-multi-count').textContent = parts.join(', ');

    // Summary table: type counts, section/material commonality
    const typeCounts = {};
    const sections = new Set();
    const materials = new Set();
    elems.forEach(e => {
        typeCounts[e.type] = (typeCounts[e.type] || 0) + 1;
        if (e.section) sections.add(e.section);
        // Look up material from v2Model
        if (window._v2Model) {
            const me = window._v2Model.elements.find(el => el.id === e.id);
            if (me && me.material) materials.add(me.material);
        }
    });

    let html = '';
    Object.entries(typeCounts).forEach(([t, c]) => {
        html += `<tr><td>${t.replace('_',' ')}</td><td>${c}</td></tr>`;
    });
    if (nodes.length > 0) {
        html += `<tr><td>nodes</td><td>${nodes.length}</td></tr>`;
    }
    html += `<tr><td>Section</td><td>${sections.size === 1 ? [...sections][0] : (sections.size === 0 ? '-' : 'Mixed (' + sections.size + ')')}</td></tr>`;
    html += `<tr><td>Material</td><td>${materials.size === 1 ? [...materials][0] : (materials.size === 0 ? '-' : 'Mixed')}</td></tr>`;
    document.getElementById('prop-multi-summary').innerHTML = html;

    // Populate bulk section dropdown
    populateBulkSectionDropdown();

    // Show/hide Release and Support groups based on selection content
    const releaseGroup = document.getElementById('bulk-release-group');
    const supportGroup = document.getElementById('bulk-support-group');
    if (releaseGroup) releaseGroup.style.display = elems.length > 0 ? 'block' : 'none';
    if (supportGroup) supportGroup.style.display = nodes.length > 0 ? 'block' : 'none';
}

function populateBulkSectionDropdown() {
    const sel = document.getElementById('bulk-section');
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = '<option value="">— keep —</option>';
    // Use sectionsList from editor state
    if (typeof sectionsList === 'object') {
        Object.entries(sectionsList).forEach(([group, names]) => {
            if (!Array.isArray(names)) return;
            const optgroup = document.createElement('optgroup');
            optgroup.label = group;
            names.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                optgroup.appendChild(opt);
            });
            sel.appendChild(optgroup);
        });
    }
    sel.value = current;
}

// ─── Bulk Edit Functions ─────────────────────────────────────────────────
function getSelectedElementIds() {
    const ids = [];
    selectedMeshSet.forEach(m => {
        const d = m.userData.elementData;
        if (d && d.type !== 'node') ids.push(d.id);
    });
    return ids;
}

function getSelectedNodeIds() {
    const ids = [];
    selectedMeshSet.forEach(m => {
        const d = m.userData.elementData;
        if (d && d.type === 'node') ids.push(d.id);
    });
    return ids;
}

function bulkApplySection() {
    const val = document.getElementById('bulk-section').value;
    if (!val || !window._v2Model) return;
    const ids = new Set(getSelectedElementIds());
    if (ids.size === 0) { alert('No elements selected.'); return; }
    if (typeof pushUndo === 'function') pushUndo();
    window._v2Model.elements.forEach(e => {
        if (ids.has(e.id)) e.section = val;
    });
    refreshEditPreview();
    showMultiSelectionPanel(); // refresh summary
    setStatus(`Section → ${val} applied to ${ids.size} elements`, 'success');
}

function bulkApplyMaterial() {
    const val = document.getElementById('bulk-material').value;
    if (!val || !window._v2Model) return;
    const ids = new Set(getSelectedElementIds());
    if (ids.size === 0) { alert('No elements selected.'); return; }
    if (typeof pushUndo === 'function') pushUndo();
    window._v2Model.elements.forEach(e => {
        if (ids.has(e.id)) e.material = val;
    });
    refreshEditPreview();
    showMultiSelectionPanel();
    setStatus(`Material → ${val} applied to ${ids.size} elements`, 'success');
}

function bulkApplyType() {
    const val = document.getElementById('bulk-type').value;
    if (!val || !window._v2Model) return;
    const ids = new Set(getSelectedElementIds());
    if (ids.size === 0) { alert('No elements selected.'); return; }
    if (typeof pushUndo === 'function') pushUndo();
    window._v2Model.elements.forEach(e => {
        if (ids.has(e.id)) e.elem_type = val;
    });
    refreshEditPreview();
    showMultiSelectionPanel();
    setStatus(`Type → ${val} applied to ${ids.size} elements`, 'success');
}

function bulkApplyRelease() {
    const preset = document.getElementById('bulk-release').value;
    if (!preset || !window._v2Model) return;
    const ids = new Set(getSelectedElementIds());
    if (ids.size === 0) { alert('No elements selected.'); return; }
    if (typeof pushUndo === 'function') pushUndo();

    // Map preset to release values
    const allFixed = null;
    const pinned = 'all'; // Rx,Ry,Rz free
    let ri, rj;
    switch (preset) {
        case 'fixed':   ri = allFixed; rj = allFixed; break;
        case 'pin_i':   ri = pinned;   rj = allFixed; break;
        case 'pin_j':   ri = allFixed; rj = pinned;   break;
        case 'pin_both': ri = pinned;  rj = pinned;   break;
        default: return;
    }

    window._v2Model.elements.forEach(e => {
        if (ids.has(e.id)) {
            e.release_i = ri;
            e.release_j = rj;
        }
    });
    refreshEditPreview();
    showMultiSelectionPanel();
    setStatus(`Release → ${preset} applied to ${ids.size} elements`, 'success');
}

function bulkApplySupport() {
    const val = document.getElementById('bulk-support').value;
    if (!val || !window._v2Model) return;
    const ids = new Set(getSelectedNodeIds());
    if (ids.size === 0) { alert('No nodes selected.'); return; }
    if (typeof pushUndo === 'function') pushUndo();

    const supportVal = val === 'free' ? null : val;
    window._v2Model.nodes.forEach(n => {
        if (ids.has(n.id)) n.support = supportVal;
    });
    refreshEditPreview();
    showMultiSelectionPanel();
    setStatus(`Support → ${val} applied to ${ids.size} nodes`, 'success');
}

function bulkDeleteSelected() {
    if (!window._v2Model) return;
    const elemIds = new Set(getSelectedElementIds());
    const nodeIds = new Set(getSelectedNodeIds());
    const total = elemIds.size + nodeIds.size;
    if (total === 0) return;
    if (!confirm(`Delete ${total} selected items?\n(${elemIds.size} elements, ${nodeIds.size} nodes)`)) return;
    if (typeof pushUndo === 'function') pushUndo();

    // Scene에서 선택된 mesh 직접 제거 (해석 결과 화면에서도 동작)
    const removedMeshes = new Set();
    selectedMeshSet.forEach(function(mesh) {
        scene.remove(mesh);
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
            if (Array.isArray(mesh.material)) mesh.material.forEach(m => m.dispose());
            else mesh.material.dispose();
        }
        removedMeshes.add(mesh);
    });
    memberMeshes = memberMeshes.filter(function(m) { return !removedMeshes.has(m.mesh); });
    nodeMeshes = nodeMeshes.filter(function(m) { return !removedMeshes.has(m); });

    // Delete elements from model
    if (elemIds.size > 0) {
        window._v2Model.elements = window._v2Model.elements.filter(e => !elemIds.has(e.id));
    }
    // Delete nodes + connected elements
    if (nodeIds.size > 0) {
        window._v2Model.elements = window._v2Model.elements.filter(e =>
            !nodeIds.has(e.node_i) && !nodeIds.has(e.node_j)
        );
        window._v2Model.nodes = window._v2Model.nodes.filter(n => !nodeIds.has(n.id));
    }

    clearAllSelection();
    // 편집 모드: 프리뷰 갱신 / 결과 모드: scene에서 이미 제거됨
    if (window._editingEnabled) {
        refreshEditPreview();
    }
    setStatus(`Deleted ${total} items`, 'success');
}

// ─── Results Panel ────────────────────────────────────────────────────────
function updateResultsPanel(result) {
    const panel = document.getElementById('prop-results');
    panel.style.display = 'block';
    // Hide empty hint when results are shown
    document.getElementById('prop-empty').style.display = 'none';

    // Display filter: story 체크박스 초기화
    if (typeof initFilterStories === 'function') initFilterStories();

    // Model source tag
    const srcTag = document.getElementById('model-source-tag');
    if (srcTag && modelSource) {
        const labels = { Manual: 'Manual', NL: 'NL (자연어)', IFC: 'IFC + Supplement' };
        srcTag.textContent = 'Source: ' + (labels[modelSource] || modelSource);
    }

    // Build case selector dropdown
    buildCaseSelector(result);

    // Diagram buttons
    showDiagramButtons();

    const env = result.envelope || {};
    renderResultsTable(env);

    // Modal analysis
    const modalData = result.modal_analysis || null;
    buildModalUI(modalData);

    // Design Check summary
    if (result.design_check) {
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
    }

    // Interpretation
    if (result.interpretation) {
        document.getElementById('interp-text').textContent =
            result.interpretation.summary_ko || result.interpretation.summary_en || '';
    }

    // Report link
    const reportDiv = document.getElementById('report-link');
    if (result.report_url) {
        reportDiv.style.display = 'block';
        document.getElementById('report-url').href = result.report_url;
    } else {
        reportDiv.style.display = 'none';
    }

    // Recommendations tab — populated from the V2 response that
    // ``convertV2ResultToV1`` now passes through. Re-rendering here
    // (rather than only on tab activation) lets the count chip update
    // even when the user never visits the tab.
    renderRecommendationsPanel(result);

    // 기본 탭: DC (해석 완료 시)
    switchResultTab('dc');

    // Auto-save
    if (typeof _autoSave === 'function') _autoSave();
}

// ─── 표지·도장란 (구조계산서 cover_info) — step 2 ─────────────────────────
//
// 해석은 cover_info=None(=placeholder)로 리포트를 1차 생성한다. 사용자가
// "표지·도장란 입력" 버튼을 누르면 이 모달이 뜨고, 제출 시 재해석 없이
// /api/jobs/{job_id}/report-cover 가 calc_data.json 사이드카만 다시 렌더한다.
// cover_info 는 localStorage 에 저장 + 다음 해석 요청에도 동봉되어 유지된다.
var _COVER_LS_KEY = 'opensees_cover_info_v1';
try {
    var _cvStored = localStorage.getItem(_COVER_LS_KEY);
    window._coverInfo = _cvStored ? JSON.parse(_cvStored) : null;
} catch (e) { window._coverInfo = null; }

function _cvVal(id) { var e = document.getElementById(id); return e ? e.value.trim() : ''; }
function _cvSet(id, v) { var e = document.getElementById(id); if (e && v != null) e.value = v; }

// 로고/직인 이미지(data URL) 보관소. 파일 input은 프로그램으로 값 복원이 불가하므로
// 읽어들인 data URL을 여기에 들고 있다가 cover_info에 실어 보낸다.
window._cvImages = { logo: null, author_seal: null, reviewer_seal: null, approver_seal: null };
var _CV_IMG_MAX = 300 * 1024;  // 300KB

function _cvImageStatus(key, on) {
    var span = document.getElementById('cv-' + key + '-status');
    if (span) span.innerHTML = on ? '✔ 업로드됨 (지우려면 다시 선택 취소)' : '300KB 이하 이미지';
}

function _cvReadImage(inputId, key) {
    var inp = document.getElementById(inputId);
    var f = inp && inp.files && inp.files[0];
    if (!f) { window._cvImages[key] = null; _cvImageStatus(key, false); return; }
    if (!/^image\//.test(f.type)) { alert('이미지 파일만 업로드할 수 있습니다.'); inp.value = ''; return; }
    if (f.size > _CV_IMG_MAX) { alert('이미지는 300KB 이하만 가능합니다. (현재 ' + Math.round(f.size / 1024) + 'KB)'); inp.value = ''; return; }
    var reader = new FileReader();
    reader.onload = function () { window._cvImages[key] = reader.result; _cvImageStatus(key, true); };
    reader.onerror = function () { alert('이미지 읽기에 실패했습니다.'); };
    reader.readAsDataURL(f);
}

function openCoverModal() {
    var c = window._coverInfo || {};
    var st = c.stamp || {};
    var au = st.author || {}, rv = st.reviewer || {}, ap = st.approver || {};
    _cvSet('cv-project_name', c.project_name || '');
    _cvSet('cv-location', c.location || '');
    _cvSet('cv-client', c.client || '');
    _cvSet('cv-structure_type', c.structure_type || '');
    _cvSet('cv-gross_floor_area', (c.gross_floor_area != null ? c.gross_floor_area : ''));
    // 저장형식 YYYY.MM.DD → <input type=date> 는 YYYY-MM-DD 필요
    if (c.date) _cvSet('cv-date', String(c.date).replace(/\./g, '-'));
    _cvSet('cv-firm', c.firm || '');
    _cvSet('cv-author_name', au.name || '');
    _cvSet('cv-author_qual', au.qualification || '건축구조기술사');
    _cvSet('cv-author_license', au.license_no || '');
    _cvSet('cv-reviewer_name', rv.name || '');
    _cvSet('cv-reviewer_qual', rv.qualification || '');
    _cvSet('cv-reviewer_license', rv.license_no || '');
    _cvSet('cv-approver_name', ap.name || '');
    _cvSet('cv-approver_qual', ap.qualification || '');
    _cvSet('cv-approver_license', ap.license_no || '');
    // 로고/직인 이미지 복원 (파일 input은 복원 불가 → 보관소 + 상태표시로 유지)
    window._cvImages = {
        logo: c.logo || null,
        author_seal: au.seal || null,
        reviewer_seal: rv.seal || null,
        approver_seal: ap.seal || null,
    };
    ['logo', 'author_seal', 'reviewer_seal', 'approver_seal'].forEach(function (k) {
        var inp = document.getElementById('cv-' + k); if (inp) inp.value = '';
        _cvImageStatus(k, !!window._cvImages[k]);
    });
    var m = document.getElementById('cover-modal');
    if (m) {
        // 이전 제출의 빨간 테두리(cv-invalid) 잔상 제거 후 표시
        m.querySelectorAll('.cv-invalid').forEach(function (el) { el.classList.remove('cv-invalid'); });
        m.style.display = 'flex';
    }
}

function closeCoverModal() {
    var m = document.getElementById('cover-modal');
    if (m) m.style.display = 'none';
}

// 폼 → cover_info 객체 (Python _normalize_cover 스키마와 일치). 필수 미입력 시 null.
function collectCoverInfoFromForm() {
    var card = document.getElementById('cover-modal');
    var ok = true;
    card.querySelectorAll('[data-required]').forEach(function (g) {
        var ctrl = g.querySelector('input, select');
        if (ctrl && !ctrl.value.trim()) { g.classList.add('cv-invalid'); ok = false; }
        else { g.classList.remove('cv-invalid'); }
    });
    if (!ok) {
        var first = card.querySelector('.cv-invalid input, .cv-invalid select');
        if (first) first.focus();
        return null;
    }
    var rawDate = _cvVal('cv-date');
    var gfaRaw = _cvVal('cv-gross_floor_area');
    var gfa = gfaRaw ? parseFloat(gfaRaw) : null;
    var imgs = window._cvImages || {};
    function stamp(p) {
        return {
            name: _cvVal('cv-' + p + '_name'),
            qualification: _cvVal('cv-' + p + '_qual'),
            license_no: _cvVal('cv-' + p + '_license'),
            seal: imgs[p + '_seal'] || null,
        };
    }
    return {
        project_name: _cvVal('cv-project_name'),
        location: _cvVal('cv-location'),
        client: _cvVal('cv-client'),
        structure_type: _cvVal('cv-structure_type'),
        date: rawDate ? rawDate.replace(/-/g, '.') : '',
        firm: _cvVal('cv-firm'),
        gross_floor_area: (gfa && gfa > 0) ? gfa : null,
        logo: imgs.logo || null,
        stamp: { author: stamp('author'), reviewer: stamp('reviewer'), approver: stamp('approver') },
    };
}

async function submitCoverInfo() {
    var ci = collectCoverInfoFromForm();
    if (!ci) return;
    window._coverInfo = ci;
    try {
        localStorage.setItem(_COVER_LS_KEY, JSON.stringify(ci));
    } catch (e) {
        // quota 초과(대용량 이미지 등) → 이미지 제외하고 텍스트만이라도 영속
        try {
            var lite = JSON.parse(JSON.stringify(ci));
            lite.logo = null;
            if (lite.stamp) {
                ['author', 'reviewer', 'approver'].forEach(function (r) { if (lite.stamp[r]) lite.stamp[r].seal = null; });
            }
            localStorage.setItem(_COVER_LS_KEY, JSON.stringify(lite));
        } catch (e2) { /* 영속 실패 — 세션 내에서는 window._coverInfo로 유지 */ }
        // 이미지가 커서 영속 못 함을 1회 안내 (리포트 자체는 정상 — 세션 내 유지됨)
        if (!window._coverImgQuotaWarned) {
            window._coverImgQuotaWarned = true;
            alert('로고/직인 이미지가 커서 브라우저에 저장되지 않았습니다.\n이번 세션에서는 정상 적용되지만, 새로고침 시 이미지는 다시 업로드해야 합니다.');
        }
    }
    if (!currentJobId) { alert('해석을 먼저 실행하세요.'); return; }
    var btn = document.getElementById('cv-submit');
    var old = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = '생성 중...'; }
    try {
        var resp = await fetch('/api/jobs/' + currentJobId + '/report-cover', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cover_info: ci }),
        });
        if (!resp.ok) {
            var err = await resp.json().catch(function () { return { detail: resp.statusText }; });
            throw new Error(err.detail || '리포트 재생성 실패');
        }
        var data = await resp.json();
        closeCoverModal();
        // 캐시 무력화 위해 timestamp 부착 후 새 탭 열기 + 링크 갱신
        var url = (data.report_url || ('/api/jobs/' + currentJobId + '/report')) + '?t=' + Date.now();
        var link = document.getElementById('report-url');
        if (link) link.href = url;
        window.open(url, '_blank');
    } catch (e) {
        alert('표지 반영 실패: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = old || '구조계산서 생성'; }
    }
}

// ─── Recommendations Tab (Phase 1 — display + Evaluate only) ──────────────
//
// State captured per /api/v2/analyze response so the Evaluate button
// can talk to /api/v2/recommendations/evaluate without re-fetching.
window._recState = {
    analysisId: null,
    issues: [],
    candidates: [],          // raw candidate dicts from the API
    candidatesById: {},      // candidate_id → candidate dict
    summary: null,
    evalJobId: null,
    evalState: 'idle',       // 'idle' | 'queued' | 'running' | 'done' | 'failed'
    evaluations: {},         // candidate_id → evaluation dict
    rejected: {},            // candidate_id → rejected evaluation dict
    ranked: [],              // candidate_ids in verified-rank order
    verifiedSummary: null,
    // Phase 2 — diff preview / apply / rollback state
    selectedCandidateId: null,
    previewLoading: false,
    applyInFlight: false,
    _pendingApply: null,     // last preview-apply response, pinned for Apply
    lastModelSnapshot: null, // structuredClone(_v2Model) taken before Apply
};

function _severityClass(sev) {
    if (sev === 'error') return 'severity-error';
    if (sev === 'warning') return 'severity-warning';
    return 'severity-info';
}

function _formatIssueTitle(iss) {
    const type = (iss.issue_type || '').replace(/_/g, ' ');
    const mid = iss.member_id != null ? `member ${iss.member_id}` : null;
    const story = iss.story != null ? `story ${iss.story}` : null;
    const dcr = iss.demand_capacity_ratio;
    const parts = [type];
    if (mid) parts.push(mid);
    if (story) parts.push(story);
    if (typeof dcr === 'number' && isFinite(dcr)) {
        parts.push(`D/C=${dcr.toFixed(2)}`);
    }
    return parts.join(' · ');
}

function _formatCandidateChange(cand) {
    const pc = cand.proposed_change || {};
    const op = pc.operation || '';
    if (op === 'replace_section') {
        return `${pc.from || '?'} <span class="rec-change-arrow">→</span> ${pc.to || '?'}`;
    }
    if (op === 'replace_sections_by_story') {
        const story = cand.target?.story ?? '?';
        const mt = cand.target?.member_type || '?';
        const step = pc.upgrade_step || 1;
        return `story ${story} · ${mt} · step ${step} (per-element ladder walk)`;
    }
    if (op === 'add_lateral_resistance') {
        const story = cand.target?.story ?? '?';
        const dir = cand.target?.direction || '';
        return `story ${story} ${dir} — bracing / shear wall (manual review)`;
    }
    if (op === 'manual_review') {
        return 'manual review';
    }
    return op;
}

function _candidateBadges(cand, evaluation) {
    // Badges stack independently:
    //   - applicability badge:  applicable | manual review
    //   - evaluation badge:     verified  | unverified | rejected
    // So a verified-and-applicable candidate now shows BOTH "applicable"
    // and "verified" (previously the if/else swallowed the applicable
    // badge once the candidate was evaluated).
    const pc = cand.proposed_change || {};
    const isVerified = evaluation && evaluation.status === 'evaluated';
    const isRejected = evaluation && (
        evaluation.status === 'rejected_new_ng'
        || evaluation.status === 'rejected_analysis_failed'
    );
    const badges = [];

    // Applicability badge — always meaningful, independent of evaluation.
    if (pc.applicable === false) {
        badges.push('<span class="rec-badge abstract">manual review</span>');
    } else if (pc.applicable === true) {
        badges.push('<span class="rec-badge applicable">applicable</span>');
    }

    // Evaluation badge.
    if (isVerified) {
        badges.push('<span class="rec-badge verified">verified</span>');
    } else if (isRejected) {
        badges.push('<span class="rec-badge rejected">rejected</span>');
    } else if (pc.applicable === true) {
        // Applicable but no evaluation yet — Preview/Apply is still
        // enabled, the hint is informational only.
        badges.push('<span class="rec-badge unverified" title="Evaluate 권장 — verified score 없이 적용됨">unverified</span>');
    }
    return badges.join(' ');
}

function _renderCandidateScore(evaluation) {
    if (!evaluation) return '';
    if (evaluation.status === 'evaluated') {
        const sc = evaluation.score || {};
        const imp = evaluation.improvement || {};
        const m = evaluation.metrics || {};
        const dDcr = (imp.dcr_delta != null) ? imp.dcr_delta.toFixed(3) : '-';
        const dDrift = (imp.drift_delta != null) ? imp.drift_delta.toFixed(4) : '-';
        const ngDelta = imp.ng_member_delta != null ? imp.ng_member_delta : '-';
        return `<div class="rec-score">
            score: <b>${(sc.total || 0).toFixed(2)}</b> (verified) ·
            ΔD/C: ${dDcr} · Δdrift: ${dDrift} · ΔNG: ${ngDelta} ·
            changed: ${m.changed_member_count || 0}
        </div>`;
    }
    if (evaluation.status === 'rejected_new_ng') {
        const newNg = (evaluation.improvement && evaluation.improvement.new_ng_members) || [];
        return `<div class="rec-score" style="color:#c5221f;">
            <b>rejected:</b> introduces new NG members ${JSON.stringify(newNg)}
        </div>`;
    }
    if (evaluation.status === 'rejected_analysis_failed') {
        return `<div class="rec-score" style="color:#c5221f;">
            <b>rejected:</b> analysis failed (${(evaluation.error || '').split('\n')[0]})
        </div>`;
    }
    return '';
}

function renderRecommendationsPanel(result) {
    const issues = result.issues || [];
    const cands = result.recommendation_candidates || [];
    const summary = result.recommendation_summary || null;

    window._recState.analysisId = result.analysis_id || result.job_id || null;
    window._recState.issues = issues;
    window._recState.candidates = cands;
    window._recState.candidatesById = {};
    cands.forEach(c => { window._recState.candidatesById[c.candidate_id] = c; });
    window._recState.summary = summary;
    // Reset any verified state — every fresh analysis starts unranked.
    window._recState.evalJobId = null;
    window._recState.evalState = 'idle';
    window._recState.evaluations = {};
    window._recState.rejected = {};
    window._recState.ranked = [];
    window._recState.verifiedSummary = null;
    // Phase 2 — the new analysis is the new ground truth; drop any
    // pending preview/apply state and the rollback snapshot from a
    // previous Apply (whether successful or not).
    window._recState.selectedCandidateId = null;
    window._recState._pendingApply = null;
    window._recState.lastModelSnapshot = null;
    window._recState.previewLoading = false;
    window._recState.applyInFlight = false;

    _renderIssuesList();
    _renderCandidatesList();

    const evalWrap = document.getElementById('rec-eval-wrap');
    if (evalWrap) {
        const anyApplicable = cands.some(c => c.proposed_change && c.proposed_change.applicable === true);
        evalWrap.style.display = anyApplicable ? '' : 'none';
    }
}

function _renderIssuesList() {
    const list = document.getElementById('rec-issues-list');
    const countEl = document.getElementById('rec-issues-count');
    const issues = window._recState.issues || [];
    if (countEl) countEl.textContent = String(issues.length);
    if (!list) return;
    if (issues.length === 0) {
        list.innerHTML = '<div style="font-size:11px; color:var(--text-secondary,#888);">No issues detected.</div>';
        return;
    }
    list.innerHTML = issues.map(iss => `
        <div class="rec-issue ${_severityClass(iss.severity)}">
            <div class="rec-issue-title">${_escapeHtml(_formatIssueTitle(iss))}</div>
            <div class="rec-issue-meta">${_escapeHtml(iss.description || '')}</div>
        </div>
    `).join('');
}

function _renderCandidatesList() {
    const list = document.getElementById('rec-candidates-list');
    const countEl = document.getElementById('rec-candidates-count');
    if (!list) return;

    // Display order: ranked verified first (if any), then remaining
    // applicable candidates, then abstract/rejected last. Within each
    // bucket we preserve the order coming from the backend (which
    // already deterministically sorts by priority).
    const cands = window._recState.candidates || [];
    const ranked = window._recState.ranked || [];
    const verified = new Set(ranked);
    const verifiedFirst = ranked.map(id => window._recState.candidatesById[id]).filter(Boolean);

    const others = cands.filter(c => !verified.has(c.candidate_id));
    const applicable = others.filter(c => c.proposed_change?.applicable === true);
    const abstract = others.filter(c => c.proposed_change?.applicable === false);

    const ordered = [...verifiedFirst, ...applicable, ...abstract];

    if (countEl) countEl.textContent = String(ordered.length);
    if (ordered.length === 0) {
        list.innerHTML = '<div style="font-size:11px; color:var(--text-secondary,#888);">No candidates.</div>';
        return;
    }

    list.innerHTML = ordered.map(cand => {
        const ev = window._recState.evaluations[cand.candidate_id]
                 || window._recState.rejected[cand.candidate_id];
        const isVerified = ev && ev.status === 'evaluated';
        const isRejected = ev && (ev.status === 'rejected_new_ng'
                                  || ev.status === 'rejected_analysis_failed');
        const isApplicable = cand.proposed_change?.applicable === true;
        // Phase 2 — show Preview/Apply on both verified and (applicable + not-yet-evaluated)
        // cards. Hidden on rejected/abstract per UX rules.
        const canPreview = isVerified || (isApplicable && !ev);

        const klass = [
            'rec-card',
            isVerified ? 'verified'
              : isRejected ? 'rejected'
              : cand.proposed_change?.applicable === false ? 'abstract'
              : 'applicable',
        ].join(' ');

        // Phase 3A — Explain button is shown on every card (applicable,
        // verified, rejected, abstract). Preview/Apply still respects the
        // applicable-or-verified gate from Phase 2.
        const previewBtnHtml = canPreview
            ? `<button type="button" class="rec-card-btn"
                       data-rec-action="preview-apply"
                       data-rec-candidate-id="${_escapeHtml(cand.candidate_id)}">
                 Preview / Apply
               </button>`
            : '';
        const explainBtnHtml = `<button type="button" class="rec-card-btn rec-card-btn-secondary"
                       data-rec-action="explain"
                       data-rec-candidate-id="${_escapeHtml(cand.candidate_id)}">
                 Explain
               </button>`;
        const actionsHtml = `<div class="rec-card-actions">${previewBtnHtml}${explainBtnHtml}</div>`;

        return `
            <div class="${klass}" data-cand-id="${_escapeHtml(cand.candidate_id)}">
                <div class="rec-card-header">
                    <span class="rec-card-title">${_escapeHtml(cand.action_type || '')}</span>
                    ${_candidateBadges(cand, ev)}
                </div>
                <div class="rec-card-body">${_formatCandidateChange(cand)}</div>
                <div class="rec-card-meta">${_escapeHtml(cand.description || '')}</div>
                ${_renderCandidateScore(ev)}
                ${actionsHtml}
            </div>
        `;
    }).join('');
}

function _escapeHtml(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

async function runRecommendationEvaluation() {
    const st = window._recState;
    if (!st.analysisId) {
        alert('Recommendations: missing analysis_id. Re-run analysis first.');
        return;
    }
    const applicableIds = (st.candidates || [])
        .filter(c => c.proposed_change && c.proposed_change.applicable === true)
        .map(c => c.candidate_id);
    if (applicableIds.length === 0) {
        alert('No applicable candidates to evaluate.');
        return;
    }

    const btn = document.getElementById('rec-eval-btn');
    const progressEl = document.getElementById('rec-eval-progress');
    if (btn) btn.disabled = true;
    if (progressEl) {
        progressEl.style.display = '';
        progressEl.textContent = `Queueing ${applicableIds.length} candidate(s)...`;
    }

    try {
        const resp = await fetch('/api/v2/recommendations/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                analysis_id: st.analysisId,
                candidate_ids: applicableIds,
            }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const start = await resp.json();
        st.evalJobId = start.job_id;
        st.evalState = 'queued';

        // Poll every 500ms.
        const deadline = Date.now() + 5 * 60 * 1000;   // 5-minute safety
        while (Date.now() < deadline) {
            await new Promise(r => setTimeout(r, 500));
            const pollResp = await fetch(`/api/v2/recommendations/evaluate/${st.evalJobId}`);
            if (!pollResp.ok) throw new Error(await pollResp.text());
            const data = await pollResp.json();
            st.evalState = data.status;
            const prog = data.progress || {};
            if (progressEl) {
                progressEl.textContent =
                    `Evaluating: ${prog.completed || 0} / ${prog.total || applicableIds.length}`;
            }
            if (data.status === 'done') {
                st.evaluations = {};
                (data.evaluated_candidates || []).forEach(ev => {
                    st.evaluations[ev.candidate_id] = ev;
                });
                st.rejected = {};
                (data.rejected_candidates || []).forEach(ev => {
                    st.rejected[ev.candidate_id] = ev;
                });
                st.ranked = data.ranked_order || [];
                st.verifiedSummary = data.recommendation_summary || null;
                _renderCandidatesList();
                if (progressEl) {
                    const s = st.verifiedSummary || {};
                    progressEl.textContent = `Done — ${s.num_success || 0} verified, ` +
                        `${s.num_rejected_new_ng || 0} rejected (new NG), ` +
                        `${s.num_rejected_analysis_failed || 0} failed, ` +
                        `${s.num_skipped_inapplicable || 0} skipped.`;
                }
                if (btn) btn.disabled = false;
                return;
            }
            if (data.status === 'failed') {
                throw new Error(data.error || 'evaluation job failed');
            }
        }
        throw new Error('evaluation timed out');
    } catch (e) {
        if (progressEl) {
            progressEl.textContent = `Evaluation failed: ${e.message || e}`;
            progressEl.style.color = '#c5221f';
        }
        if (btn) btn.disabled = false;
        st.evalState = 'failed';
    }
}

// ─── Phase 2 — Diff Preview Modal + Apply + Auto-rollback ────────────────

async function openRecDiffModal(candidateId) {
    const st = window._recState;
    if (!st.analysisId) {
        alert('No analysis_id — re-run analysis.');
        return;
    }
    if (st.previewLoading || st.applyInFlight) return;

    st.previewLoading = true;
    st.selectedCandidateId = candidateId;
    _showRecModal(true);
    _renderRecDiffLoading();
    _setRecModalApplyState('disabled');

    try {
        const resp = await fetch('/api/v2/recommendations/preview-apply', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                analysis_id: st.analysisId,
                candidate_id: candidateId,
            }),
        });
        if (!resp.ok) {
            // Read body exactly once, then try to extract `detail`.
            const raw = await resp.text();
            let detail = raw;
            try { detail = JSON.parse(raw).detail || raw; } catch (_) { /* keep raw */ }
            _renderRecDiffError(detail, resp.status);
            return;
        }
        const data = await resp.json();
        st._pendingApply = data;
        _renderRecDiffPreview(data);
        const hasChanges = (data?.diff?.changed_member_count || 0) > 0;
        _setRecModalApplyState(hasChanges ? 'idle' : 'disabled');
    } catch (e) {
        _renderRecDiffError(String(e?.message || e), 0);
    } finally {
        st.previewLoading = false;
    }
}

// `force:true` lets the apply path close the modal even while
// applyInFlight is set — Cancel/backdrop/× still respect the guard.
function closeRecDiffModal({ force = false } = {}) {
    if (window._recState.applyInFlight && !force) return;
    _showRecModal(false);
    window._recState.selectedCandidateId = null;
    window._recState._pendingApply = null;
}

async function applyRecDiff() {
    const st = window._recState;
    const data = st._pendingApply;
    if (!data || !data.updated_model) return;
    if (st.applyInFlight) return;

    // Close BEFORE flipping applyInFlight so we don't trip our own guard.
    // (force:true is defense-in-depth in case future code reorders this.)
    closeRecDiffModal({ force: true });

    st.applyInFlight = true;
    _setRecModalApplyState('applying');

    // Rollback snapshot (for our auto-rollback on reanalysis failure).
    try {
        st.lastModelSnapshot = structuredClone(window._v2Model);
    } catch (_) {
        // structuredClone is in all modern browsers; fall back to JSON.
        st.lastModelSnapshot = JSON.parse(JSON.stringify(window._v2Model));
    }
    // Push the PRE-Apply model onto the editor's undo stack BEFORE we
    // swap _v2Model. If we let runAnalysisV2() push undo on its own, it
    // would save the post-Apply state and Ctrl+Z could not return to
    // the pre-recommendation model. Pass skipUndo:true below so the
    // analysis function doesn't double-save the post-Apply model.
    if (typeof pushUndo === 'function') pushUndo();
    window._v2Model = data.updated_model;

    _showRecToast(
        `Applied "${data.candidate_id}". Re-running analysis…`,
        'info',
    );

    try {
        // Precondition: runAnalysisV2({rethrow:true}) rejects on failure.
        // skipUndo:true because we already pushed the pre-Apply undo
        // point above.
        await runAnalysisV2({ rethrow: true, skipUndo: true });
        st.lastModelSnapshot = null;     // success — drop the snapshot
        _showRecToast('Reanalysis complete.', 'success');
    } catch (e) {
        if (st.lastModelSnapshot) {
            window._v2Model = st.lastModelSnapshot;
            st.lastModelSnapshot = null;
        }
        _refreshAfterRollback();
        _showRecToast(
            `Reanalysis failed — change rolled back: ${e?.message || e}`,
            'error',
        );
    } finally {
        st.applyInFlight = false;
        _setRecModalApplyState('idle');
    }
}

function _refreshAfterRollback() {
    // Clear any in-flight loading overlay so the user sees the restored
    // model rather than the half-applied state.
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'none';
    if (typeof setStatus === 'function') {
        setStatus('Apply rolled back — previous model restored.', 'error');
    }
    // We deliberately do NOT re-render the Recs panel from stale
    // candidates — those came from the analysis we just abandoned. They
    // remain on screen until the user manually re-runs analysis, which
    // matches existing UX after any analysis error.
}

// ─── Modal / toast render helpers ─────────────────────────────────────────

function _showRecModal(open) {
    const m = document.getElementById('rec-diff-modal');
    if (m) m.style.display = open ? 'flex' : 'none';
}

function _renderRecDiffLoading() {
    const summary = document.getElementById('rec-diff-summary');
    const tbl = document.getElementById('rec-diff-table-wrap');
    const err = document.getElementById('rec-diff-error');
    if (summary) summary.textContent = 'Loading preview…';
    if (tbl) tbl.innerHTML = '';
    if (err) { err.style.display = 'none'; err.textContent = ''; }
}

function _renderRecDiffPreview(data) {
    const summary = document.getElementById('rec-diff-summary');
    const tbl = document.getElementById('rec-diff-table-wrap');
    const err = document.getElementById('rec-diff-error');
    if (err) { err.style.display = 'none'; err.textContent = ''; }

    const diff = data?.diff || {};
    const rows = diff.changed_members || [];

    if (summary) {
        const op = _escapeHtml(diff.operation || '?');
        const n = diff.changed_member_count || 0;
        const reason = _escapeHtml(diff.reason || '');
        summary.innerHTML =
            `<b>${op}</b> · ${n} member${n === 1 ? '' : 's'}`
            + (reason ? ` · reason: ${reason}` : '');
    }

    if (!tbl) return;
    if (rows.length === 0) {
        tbl.innerHTML = '<div style="font-size:11px;color:var(--text-secondary,#888);">No members would change.</div>';
        return;
    }
    const body = rows.map(r => `
        <tr>
            <td>${_escapeHtml(r.member_label || '')}</td>
            <td>${r.story != null ? _escapeHtml(String(r.story)) : '-'}</td>
            <td>${_escapeHtml(r.member_type || '')}</td>
            <td>${_escapeHtml(r.section_from || '')}
                <span class="rec-diff-arrow">→</span>
                ${_escapeHtml(r.section_to || '')}</td>
            <td>${_escapeHtml(r.reason || '')}</td>
        </tr>
    `).join('');
    tbl.innerHTML = `
        <table class="rec-diff-table">
            <thead><tr>
                <th>Member</th><th>Story</th><th>Type</th>
                <th>Section</th><th>Reason</th>
            </tr></thead>
            <tbody>${body}</tbody>
        </table>
    `;
}

function _renderRecDiffError(detail, status) {
    const summary = document.getElementById('rec-diff-summary');
    const tbl = document.getElementById('rec-diff-table-wrap');
    const err = document.getElementById('rec-diff-error');
    if (summary) summary.textContent = '';
    if (tbl) tbl.innerHTML = '';
    if (err) {
        const prefix = status ? `HTTP ${status}: ` : '';
        err.textContent = prefix + (detail || 'preview failed');
        err.style.display = '';
    }
    _setRecModalApplyState('disabled');
}

function _setRecModalApplyState(state) {
    const btn = document.getElementById('rec-diff-apply');
    if (!btn) return;
    if (state === 'applying') {
        btn.disabled = true;
        btn.textContent = 'Applying…';
    } else if (state === 'disabled') {
        btn.disabled = true;
        btn.textContent = 'Apply to editor';
    } else {
        btn.disabled = false;
        btn.textContent = 'Apply to editor';
    }
}

function _showRecToast(msg, kind) {
    let container = document.getElementById('rec-toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'rec-toast-container';
        document.body.appendChild(container);
    }
    const t = document.createElement('div');
    t.className = 'rec-toast' + (kind ? ' rec-toast-' + kind : '');
    t.textContent = msg;
    container.appendChild(t);
    setTimeout(() => { t.remove(); }, 4000);
}

// ─── Delegated click handler for Preview/Apply + modal close ──────────────
//
// One listener at document level handles every data-rec-action click.
// We deliberately avoid inline onclick with interpolated candidate_id
// because _escapeHtml() does not make a value safe inside a JS string
// literal context — even though candidate_id is deterministic today.
document.addEventListener('click', (e) => {
    const previewBtn = e.target.closest('[data-rec-action="preview-apply"]');
    if (previewBtn) {
        openRecDiffModal(previewBtn.dataset.recCandidateId);
        return;
    }
    if (e.target.closest('[data-rec-action="apply-confirm"]')) {
        applyRecDiff();
        return;
    }
    if (e.target.closest('[data-rec-action="modal-close"]')) {
        closeRecDiffModal();
        return;
    }
    const explainBtn = e.target.closest('[data-rec-action="explain"]');
    if (explainBtn) {
        openRecExplainModal(explainBtn.dataset.recCandidateId);
        return;
    }
    if (e.target.closest('[data-rec-action="explain-modal-close"]')) {
        closeRecExplainModal();
        return;
    }
});

// Escape key closes whichever rec modal is currently open. Diff modal
// closure still respects the applyInFlight guard; explain modal is
// advisory and always closeable.
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const diff = document.getElementById('rec-diff-modal');
    if (diff && diff.style.display !== 'none') { closeRecDiffModal(); return; }
    const explain = document.getElementById('rec-explain-modal');
    if (explain && explain.style.display !== 'none') { closeRecExplainModal(); return; }
});

// ─── Phase 3A — Recommendation explanation modal ─────────────────────────

const _EXPLAIN_SECTIONS = [
    ['summary', '요약'],
    ['issue_interpretation', '이슈 해석'],
    ['recommended_change', '권장 변경'],
    ['expected_structural_effect', '예상 구조 효과'],
    ['verified_result', '검증 결과'],
    ['tradeoffs', '트레이드오프'],
    ['limitations', '한계 및 미확보 근거'],
    ['next_user_decision', '다음 의사결정'],
];

const _HANGUL_RE = /[가-힯]/;

// Backend emits machine-readable warning codes ("code: detail"). The UI
// translates known codes to Korean labels; unknown codes pass through.
const _WARNING_TRANSLATIONS_KO = {
    'kds_rag_unavailable': 'KDS-RAG 인덱스가 설정되지 않았습니다',
    'kds_evidence_missing': 'KDS 인용이 확보되지 않았습니다 — 결정론적 분석만으로 설명되었습니다',
    'evaluation_missing': '재해석 검증이 아직 수행되지 않았습니다',
    'evaluation_status': '평가 상태 이상',
    'abstract_candidate': '자동 적용 불가능한 추상 후보 — 수동 엔지니어 검토 필요',
    'diff_missing': '구조 변경 내역이 제공되지 않았습니다',
    'llm_provider_failed': 'LLM 설명 생성 실패 — 결정론적 설명으로 대체되었습니다',
    'llm_provider_returned_unexpected_type': 'LLM 응답 형식 이상 — 결정론적 설명으로 대체되었습니다',
    'kds_retriever_exception': 'KDS 검색 호출 실패',
    'kds_query_build_failed': 'KDS 질의 생성 실패',
    'chunk_to_ref_failed': 'KDS 청크 → 인용 변환 실패',
    'diff_derivation_skipped': '변경 내역 산출 건너뜀',
    'diff_derivation_inapplicable': '자동 적용 불가 — 변경 내역 산출 불가',
    'diff_derivation_failed': '변경 내역 산출 중 오류',
    'no_match': 'KDS 검색 결과 없음',
    'weak_match': 'KDS 검색 매칭 약함',
    'voyage_index_empty': 'Voyage KDS 인덱스가 비어 있습니다',
    'voyage_embed_query_failed': 'Voyage 질의 임베딩 실패',
    'voyage_rerank_failed': 'Voyage 리랭크 실패 — 코사인 점수로 대체',
    'voyage_no_match': 'Voyage 검색 결과 없음',
    'aisc_temporary_reference': '현재 강구조 근거는 KDS 원문이 아닌 AISC 360-22 임시 참조입니다. KDS 14 31 00 / KDS 41 31 00 원문 확보 후 교체 검증이 필요합니다.',
};

function _translateWarning(w) {
    const s = String(w || '');
    // Allow newlines in detail (some warnings carry traceback excerpts).
    const m = s.match(/^([a-z_]+)(?::\s*([\s\S]*))?$/);
    if (!m) return s;
    const code = m[1];
    const detail = m[2] ? m[2].trim() : '';
    // Backend-localized detail beats our static label — the explainer
    // and Noop retriever already write user-friendly Korean (often with
    // additional context like env var names). Show the detail verbatim.
    if (detail && _HANGUL_RE.test(detail)) return detail;
    // English (machine) detail → use our short Korean label if known.
    const label = _WARNING_TRANSLATIONS_KO[code];
    if (label) return detail ? `${label} — ${detail}` : label;
    return s;
}

function _showRecExplainModal(open) {
    const m = document.getElementById('rec-explain-modal');
    if (!m) return;
    m.style.display = open ? 'flex' : 'none';
}

function _renderRecExplainLoading() {
    const body = document.getElementById('rec-explain-body');
    if (body) body.innerHTML = '<div class="rec-explain-loading">설명을 불러오는 중...</div>';
}

function _renderRecExplainError(msg, status) {
    const body = document.getElementById('rec-explain-body');
    if (!body) return;
    body.innerHTML = `
      <div class="rec-explain-error">
        <strong>설명을 불러오지 못했습니다.</strong>
        <div class="rec-explain-error-detail">[${_escapeHtml(String(status || ''))}] ${_escapeHtml(String(msg || ''))}</div>
      </div>`;
}

function _renderRecExplainResult(data) {
    const body = document.getElementById('rec-explain-body');
    if (!body) return;
    const exp = data.explanation || {};
    const ev = Array.isArray(data.kds_evidence) ? data.kds_evidence : [];
    const warnings = Array.isArray(data.warnings) ? data.warnings : [];
    const src = data.source || {};
    const conf = data.confidence || 'low';

    const sectionsHtml = _EXPLAIN_SECTIONS.map(([key, label]) => {
        const v = exp[key] || '';
        if (!v) return '';
        return `<section class="rec-explain-section">
                  <h4>${_escapeHtml(label)}</h4>
                  <p>${_escapeHtml(v)}</p>
                </section>`;
    }).join('');

    const evidenceHtml = ev.length
        ? `<section class="rec-explain-section">
             <h4>KDS 근거</h4>
             <ul class="rec-evidence-list">
               ${ev.map(e => `
                 <li class="rec-evidence-card">
                   <div class="rec-evidence-head">
                     <span class="rec-evidence-doc">${_escapeHtml(e.doc_id || '')}</span>
                     ${e.clause ? `<span class="rec-evidence-clause">${_escapeHtml(e.clause)}</span>` : ''}
                     <span class="rec-evidence-score">score ${_escapeHtml(Number(e.score || 0).toFixed(2))}</span>
                   </div>
                   ${e.title ? `<div class="rec-evidence-title">${_escapeHtml(e.title)}</div>` : ''}
                   ${e.quote ? `<blockquote class="rec-evidence-quote">${_escapeHtml(e.quote)}</blockquote>` : ''}
                   ${e.relevance ? `<div class="rec-evidence-rel">${_escapeHtml(e.relevance)}</div>` : ''}
                 </li>`).join('')}
             </ul>
           </section>`
        : `<section class="rec-explain-section">
             <h4>KDS 근거</h4>
             <p class="rec-explain-muted">KDS 인용이 확보되지 않았습니다 (RAG 인덱스 미설정 또는 무매칭).</p>
           </section>`;

    const warningsHtml = warnings.length
        ? `<section class="rec-explain-section">
             <h4>주의</h4>
             <ul class="rec-explain-warning-list">
               ${warnings.map(w => `<li>${_escapeHtml(_translateWarning(w))}</li>`).join('')}
             </ul>
           </section>`
        : '';

    const ragBadge = src.rag_used
        ? '<span class="rec-explain-badge rec-explain-badge-ok">RAG 사용</span>'
        : '<span class="rec-explain-badge rec-explain-badge-warn">RAG 미사용</span>';
    const llmBadge = src.llm_used
        ? '<span class="rec-explain-badge rec-explain-badge-ok">LLM 사용</span>'
        : '<span class="rec-explain-badge">결정론적</span>';
    const methodBadge = src.score_method
        ? `<span class="rec-explain-badge">${_escapeHtml(src.score_method)}</span>`
        : '';
    const confBadge = `<span class="rec-explain-badge rec-explain-confidence-${_escapeHtml(conf)}">신뢰도 ${_escapeHtml(conf)}</span>`;

    body.innerHTML = `
      <div class="rec-explain-meta">
        ${confBadge}${ragBadge}${llmBadge}${methodBadge}
      </div>
      ${sectionsHtml}
      ${evidenceHtml}
      ${warningsHtml}
    `;
}

async function openRecExplainModal(candidateId) {
    const st = window._recState;
    if (!st || !st.analysisId) {
        alert('No analysis_id — re-run analysis.');
        return;
    }
    _showRecExplainModal(true);
    _renderRecExplainLoading();

    const evaluation = (st.evaluations && st.evaluations[candidateId])
        || (st.rejected && st.rejected[candidateId])
        || null;
    const pending = st._pendingApply;
    const diff = (pending && pending.candidate_id === candidateId)
        ? pending.diff
        : undefined;

    try {
        const resp = await fetch('/api/v2/recommendations/explain', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                analysis_id: st.analysisId,
                candidate_id: candidateId,
                evaluation: evaluation || undefined,
                diff: diff,
                language: 'ko',
                style: 'engineer_brief',
            }),
        });
        if (!resp.ok) {
            const raw = await resp.text();
            let detail = raw;
            try { detail = JSON.parse(raw).detail || raw; } catch (_) { /* keep raw */ }
            _renderRecExplainError(detail, resp.status);
            return;
        }
        const data = await resp.json();
        _renderRecExplainResult(data);
    } catch (e) {
        _renderRecExplainError(String(e?.message || e), 0);
    }
}

function closeRecExplainModal() {
    _showRecExplainModal(false);
}

// ─── Result Tab Switching ─────────────────────────────────────────────────
function switchResultTab(tabName) {
    document.querySelectorAll('.rtab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.rtab === tabName);
    });
    document.querySelectorAll('.rtab-content').forEach(div => {
        div.style.display = div.dataset.rtab === tabName ? '' : 'none';
    });
}

function updateBottomBar(result) {
    const bar = document.getElementById('bottom-bar');
    bar.style.display = 'flex';

    updateBottomBarValues(result.envelope || {});

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

// ─── Case/Combo Selector ─────────────────────────────────────────────────
function buildCaseSelector(result) {
    const wrap = document.getElementById('case-selector-wrap');
    const sel = document.getElementById('case-selector');
    if (!sel) return;

    const caseNames = result.case_names || [];
    const comboNames = result.combo_names || [];

    if (caseNames.length === 0 && comboNames.length === 0) {
        wrap.style.display = 'none';
        return;
    }

    sel.innerHTML = '<option value="__envelope__">Envelope (전체 조합)</option>';

    // Split RSA cases from regular combos
    const rsaNames = comboNames.filter(n => n.includes('RSA'));
    const regularCombos = comboNames.filter(n => !n.includes('RSA'));
    const regularCases = caseNames.filter(n => n !== '__RSA__');

    if (regularCombos.length > 0) {
        const grp = document.createElement('optgroup');
        grp.label = 'Load Combinations (ELF)';
        regularCombos.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            grp.appendChild(opt);
        });
        sel.appendChild(grp);
    }

    if (rsaNames.length > 0) {
        const grp = document.createElement('optgroup');
        grp.label = 'RSA (응답스펙트럼)';
        rsaNames.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            grp.appendChild(opt);
        });
        sel.appendChild(grp);
    }

    if (regularCases.length > 0) {
        const grp = document.createElement('optgroup');
        grp.label = 'Load Cases';
        regularCases.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            grp.appendChild(opt);
        });
        sel.appendChild(grp);
    }

    // Build ELF vs RSA comparison table if RSA data exists
    buildElfRsaComparison(result);

    wrap.style.display = 'block';
}

function onCaseSelect() {
    const sel = document.getElementById('case-selector');
    if (!sel || !currentResult) return;
    const caseName = sel.value;

    if (caseName === '__envelope__') {
        renderResultsTable(currentResult.envelope || {});
        updateBottomBarValues(currentResult.envelope || {});
    } else {
        const cd = currentResult.case_data?.[caseName];
        if (!cd) return;
        renderResultsTable(cd.summary, caseName);
        updateBottomBarValues(cd.summary);
    }

    // Deformed shape: 토글 상태에 따라
    _applyDeformedIfEnabled();
}

function toggleDeformedShape() {
    _applyDeformedIfEnabled();
    // 스케일 슬라이더 표시/숨김
    const dsWrap = document.getElementById('deform-scale-wrap');
    const checked = document.getElementById('toggle-deformed')?.checked;
    if (dsWrap) dsWrap.style.display = checked ? '' : 'none';
}

function _applyDeformedIfEnabled() {
    const checked = document.getElementById('toggle-deformed')?.checked;
    if (!checked) {
        restoreOriginalPositions();
        return;
    }
    // 현재 선택된 case의 변위 적용
    const sel = document.getElementById('case-selector');
    const caseName = sel?.value;
    if (!caseName || caseName === '__envelope__') {
        // Envelope: 첫 번째 combo 사용
        const combos = currentResult?.combo_names || [];
        const fallback = combos[0] || currentResult?.case_names?.[0];
        if (fallback) {
            const cd = currentResult?.case_data?.[fallback];
            if (cd?.displacements) { applyDeformedShape(cd.displacements); return; }
        }
        restoreOriginalPositions();
        return;
    }
    const cd = currentResult?.case_data?.[caseName];
    if (cd?.displacements && Object.keys(cd.displacements).length > 0) {
        applyDeformedShape(cd.displacements);
    } else {
        restoreOriginalPositions();
    }
}

function buildElfRsaComparison(result) {
    const section = document.getElementById('elf-rsa-comparison');
    const table = document.getElementById('elf-rsa-table');
    if (!section || !table) return;

    const cd = result.case_data || {};
    const hasRSA = cd['EQX_RSA'] || cd['EQY_RSA'];
    if (!hasRSA) {
        section.style.display = 'none';
        return;
    }

    // Find ELF max (from EQX/EQY combos)
    const env = result.envelope || {};
    const elfDriftX = env.max_drift_x || 0;
    const elfDriftY = env.max_drift_y || 0;
    const elfDispX = Math.abs(env.max_dx_mm || 0);
    const elfDispY = Math.abs(env.max_dy_mm || 0);

    // RSA max (from RSA_100X_30Y / RSA_30X_100Y envelope)
    const rsa1 = cd['RSA_100X_30Y']?.summary || {};
    const rsa2 = cd['RSA_30X_100Y']?.summary || {};
    const rsaDriftX = Math.max(rsa1.max_drift_x || 0, rsa2.max_drift_x || 0);
    const rsaDriftY = Math.max(rsa1.max_drift_y || 0, rsa2.max_drift_y || 0);
    const rsaDispX = Math.max(rsa1.max_dx_mm || 0, rsa2.max_dx_mm || 0);
    const rsaDispY = Math.max(rsa1.max_dy_mm || 0, rsa2.max_dy_mm || 0);

    function ratio(elf, rsa) {
        if (elf === 0) return '-';
        return (rsa / elf * 100).toFixed(0) + '%';
    }

    function diffColor(elf, rsa) {
        if (elf === 0) return '';
        const r = rsa / elf;
        if (r > 1.05) return 'color:#ea4335;';
        if (r < 0.95) return 'color:#34a853;';
        return '';
    }

    table.innerHTML = `
        <tr style="font-size:10px; color:var(--text-tertiary);">
            <th></th><th>ELF</th><th>RSA</th><th>RSA/ELF</th>
        </tr>
        <tr>
            <td>Drift X</td>
            <td>${elfDriftX.toFixed(5)}</td>
            <td>${rsaDriftX.toFixed(5)}</td>
            <td style="${diffColor(elfDriftX, rsaDriftX)}">${ratio(elfDriftX, rsaDriftX)}</td>
        </tr>
        <tr>
            <td>Drift Y</td>
            <td>${elfDriftY.toFixed(5)}</td>
            <td>${rsaDriftY.toFixed(5)}</td>
            <td style="${diffColor(elfDriftY, rsaDriftY)}">${ratio(elfDriftY, rsaDriftY)}</td>
        </tr>
        <tr>
            <td>Disp X</td>
            <td>${elfDispX.toFixed(2)} mm</td>
            <td>${rsaDispX.toFixed(2)} mm</td>
            <td style="${diffColor(elfDispX, rsaDispX)}">${ratio(elfDispX, rsaDispX)}</td>
        </tr>
        <tr>
            <td>Disp Y</td>
            <td>${elfDispY.toFixed(2)} mm</td>
            <td>${rsaDispY.toFixed(2)} mm</td>
            <td style="${diffColor(elfDispY, rsaDispY)}">${ratio(elfDispY, rsaDispY)}</td>
        </tr>
    `;

    section.style.display = 'block';
}

function renderResultsTable(env, caseName) {
    const table = document.getElementById('results-table');
    const label = caseName ? `<tr><td colspan="2" style="color:var(--accent-text); font-weight:600; padding-bottom:4px;">${caseName}</td></tr>` : '';
    table.innerHTML = label + `
        <tr><td>Max Drift X</td><td>${(env.max_drift_x || 0).toFixed(5)}</td></tr>
        <tr><td>Max Drift Y</td><td>${(env.max_drift_y || 0).toFixed(5)}</td></tr>
        <tr><td>Max Disp X</td><td>${(env.max_dx_mm || 0).toFixed(2)} mm</td></tr>
        <tr><td>Max Disp Y</td><td>${(env.max_dy_mm || 0).toFixed(2)} mm</td></tr>
        <tr><td>Max Moment</td><td>${(env.max_moment_kNm || 0).toFixed(1)} kN·m</td></tr>
        <tr><td>Max Axial</td><td>${(env.max_axial_kN || 0).toFixed(1)} kN</td></tr>
        <tr><td>Max Shear</td><td>${(env.max_shear_kN || 0).toFixed(1)} kN</td></tr>
    `;
}

function updateBottomBarValues(env) {
    document.getElementById('bot-drift-x').textContent = (env.max_drift_x || 0).toFixed(5);
    document.getElementById('bot-drift-y').textContent = (env.max_drift_y || 0).toFixed(5);
    document.getElementById('bot-disp-x').textContent = (env.max_dx_mm || 0).toFixed(2) + ' mm';
    document.getElementById('bot-disp-y').textContent = (env.max_dy_mm || 0).toFixed(2) + ' mm';
    document.getElementById('bot-moment').textContent = (env.max_moment_kNm || 0).toFixed(1) + ' kN·m';
}

// ─── Deformed Shape Visualization ────────────────────────────────────────
let originalMemberState = null;  // Map<uuid, {pos, quat, scaleY}>
let originalNodeState = null;    // Map<uuid, {pos}>

function saveOriginalState() {
    if (originalMemberState) return;
    originalMemberState = new Map();
    originalNodeState = new Map();
    memberMeshes.forEach(({ mesh }) => {
        originalMemberState.set(mesh.uuid, {
            pos: mesh.position.clone(),
            quat: mesh.quaternion.clone(),
            scaleY: mesh.scale.y,
            color: mesh.material.color.getHex(),
            opacity: mesh.material.opacity,
        });
    });
    nodeMeshes.forEach(m => {
        originalNodeState.set(m.uuid, {
            pos: m.position.clone(),
            color: m.material.color.getHex(),
            opacity: m.material.opacity,
        });
    });
}

// ─── Deformation Scale ────────────────────────────────────────────────────
let _deformScale = 50;
let _lastDisplacements = null;

function _sliderToScale(v) {
    // slider 0~100 → scale 1~500 (log)
    // 0→1, 50→50, 100→500
    return Math.round(Math.exp(v / 100 * Math.log(500)));
}
function _scaleToSlider(s) {
    if (s <= 1) return 0;
    return Math.round(100 * Math.log(s) / Math.log(500));
}

function onDeformScaleChange(sliderVal) {
    _deformScale = _sliderToScale(parseInt(sliderVal));
    document.getElementById('deform-scale-val').textContent = _deformScale + '×';
    if (_lastDisplacements) {
        applyDeformedShape(_lastDisplacements);
    }
}

function autoDeformScale() {
    // 최대 변위 기반 자동 스케일: 모델 크기의 5% 정도로 변형이 보이도록
    if (!currentResult?.viewer?.nodes || !_lastDisplacements) return;
    const nodes = currentResult.viewer.nodes;
    let maxDim = 1;
    nodes.forEach(n => {
        maxDim = Math.max(maxDim, Math.abs(n.x), Math.abs(n.y), Math.abs(n.z));
    });
    let maxDisp = 0;
    Object.values(_lastDisplacements).forEach(d => {
        if (d) maxDisp = Math.max(maxDisp, Math.abs(d[0]), Math.abs(d[1]), Math.abs(d[2]));
    });
    if (maxDisp < 0.001) return;
    const target = maxDim * 0.05; // 5% of model size
    _deformScale = Math.max(1, Math.min(500, Math.round(target / (maxDisp / 1000))));
    const slider = document.getElementById('deform-scale-slider');
    if (slider) slider.value = _scaleToSlider(_deformScale);
    document.getElementById('deform-scale-val').textContent = _deformScale + '×';
    applyDeformedShape(_lastDisplacements);
}

function showDeformSlider() {
    const wrap = document.getElementById('deform-scale-wrap');
    if (wrap) wrap.style.display = '';
}

function applyDeformedShape(displacements) {
    if (!displacements || !currentResult?.viewer) return;
    _lastDisplacements = displacements;
    saveOriginalState();

    const scale = _deformScale;
    const viewer = currentResult.viewer;
    const nodes = viewer.nodes;

    // Build deformed node positions in Three.js coords (X, Z→Y, -Y→Z)
    const deformedPos = {};  // nodeId -> THREE.Vector3
    nodes.forEach(n => {
        const d = displacements[String(n.id)];
        const dx = d ? d[0] / 1000 * scale : 0;  // mm→m, scaled
        const dy = d ? d[1] / 1000 * scale : 0;
        const dz = d ? d[2] / 1000 * scale : 0;
        // Three.js: (x, z, -y) — consistent with buildScene
        deformedPos[n.id] = new THREE.Vector3(
            n.x + dx,
            n.z + dz,
            -(n.y + dy),
        );
    });

    // Reposition member cylinders to connect deformed nodes
    const yAxis = new THREE.Vector3(0, 1, 0);
    memberMeshes.forEach(({ mesh, elementData }) => {
        const pi = deformedPos[elementData.ni];
        const pj = deformedPos[elementData.nj];
        if (!pi || !pj) return;

        const dir = new THREE.Vector3().subVectors(pj, pi);
        const newLen = dir.length();
        const mid = new THREE.Vector3().addVectors(pi, pj).multiplyScalar(0.5);
        const direction = dir.clone().normalize();

        mesh.position.copy(mid);
        mesh.quaternion.setFromUnitVectors(yAxis, direction);

        // Scale cylinder length to match deformed distance
        const orig = originalMemberState.get(mesh.uuid);
        if (orig && orig.scaleY !== 0) {
            // Original cylinder height = geometry height * scaleY
            // We want new visual length = newLen
            // Original geometry height corresponds to original scaleY=1 length
            mesh.scale.y = newLen / (mesh.geometry.parameters.height || newLen);
        }
    });

    // Move node spheres to deformed positions
    nodeMeshes.forEach(m => {
        const nid = m.userData?.nodeId;
        // nid가 0일 수도 있으므로 undefined/null만 배제
        if (nid === undefined || nid === null) return;
        const dp = deformedPos[nid];
        if (!dp) return;
        m.position.copy(dp);
        m.updateMatrix();
    });

    // Solid section meshes도 변형에 맞게 이동/회전/스케일
    if (window.solidMode && window._solidMeshMap && window._v2Model) {
        const elemMap = {};
        window._v2Model.elements.forEach(e => { elemMap[e.id] = e; });
        const defaultDir = new THREE.Vector3(0, 0, 1);  // ExtrudeGeometry 돌출 방향
        Object.entries(window._solidMeshMap).forEach(([elemId, solidMesh]) => {
            const elem = elemMap[elemId];
            if (!elem) return;
            const pi = deformedPos[elem.node_i];
            const pj = deformedPos[elem.node_j];
            if (!pi || !pj) return;
            const dir = new THREE.Vector3().subVectors(pj, pi);
            const newLen = dir.length();
            if (newLen < 0.001) return;
            const origLen = solidMesh.userData._solidOrigLen || newLen;
            solidMesh.position.copy(pi);
            solidMesh.quaternion.setFromUnitVectors(defaultDir, dir.clone().normalize());
            solidMesh.scale.z = newLen / origLen;
        });
    }
}

function restoreOriginalPositions() {
    if (originalMemberState) {
        memberMeshes.forEach(({ mesh }) => {
            const orig = originalMemberState.get(mesh.uuid);
            if (orig) {
                mesh.position.copy(orig.pos);
                mesh.quaternion.copy(orig.quat);
                mesh.scale.y = orig.scaleY;
                mesh.material.color.setHex(orig.color);
                mesh.material.opacity = orig.opacity;
            }
        });
    }
    if (originalNodeState) {
        nodeMeshes.forEach(m => {
            const orig = originalNodeState.get(m.uuid);
            if (orig) {
                m.position.copy(orig.pos);
                m.material.color.setHex(orig.color);
                m.material.opacity = orig.opacity;
            }
        });
    }
    // Solid meshes 원상 복귀 (solid mode일 때만)
    if (window.solidMode && window._solidMeshMap && window._v2Model) {
        const nodeMap = {};
        window._v2Model.nodes.forEach(n => { nodeMap[n.id] = n; });
        const elemMap = {};
        window._v2Model.elements.forEach(e => { elemMap[e.id] = e; });
        const defaultDir = new THREE.Vector3(0, 0, 1);
        Object.entries(window._solidMeshMap).forEach(([elemId, solidMesh]) => {
            const elem = elemMap[elemId];
            if (!elem) return;
            const ni = nodeMap[elem.node_i];
            const nj = nodeMap[elem.node_j];
            if (!ni || !nj) return;
            const startPos = new THREE.Vector3(ni.x, ni.z, -ni.y);
            const endPos = new THREE.Vector3(nj.x, nj.z, -nj.y);
            const dir = new THREE.Vector3().subVectors(endPos, startPos).normalize();
            solidMesh.position.copy(startPos);
            solidMesh.quaternion.setFromUnitVectors(defaultDir, dir);
            solidMesh.scale.set(1, 1, 1);
        });
    }
}

// ─── Modal Analysis UI + Animation ───────────────────────────────────────
let modeAnimationId = null;
let modeAnimating = false;

function buildModalUI(modal) {
    const section = document.getElementById('modal-section');
    const sel = document.getElementById('mode-selector');
    const table = document.getElementById('modal-table');
    if (!section || !sel || !modal?.modes?.length) {
        return;
    }

    sel.innerHTML = '<option value="">-- 모드 선택 (3D 형상 표시) --</option>';

    let tableHTML = `<tr style="font-size:10px; color:var(--text-tertiary);">
        <th>Mode</th><th>T (s)</th><th>Dir</th><th>Mass%</th></tr>`;

    modal.modes.forEach((m, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `Mode ${m.mode}: T=${m.period_s}s (${m.direction})`;
        sel.appendChild(opt);

        const mp = m.mass_participation || {};
        const dominant = Math.max(mp.x_pct || 0, mp.y_pct || 0, mp.rz_pct || 0);
        const highlight = i < 3 ? 'font-weight:600;' : '';
        tableHTML += `<tr style="${highlight}">
            <td>${m.mode}</td>
            <td>${m.period_s.toFixed(3)}</td>
            <td>${m.direction}</td>
            <td>${dominant.toFixed(1)}%</td>
        </tr>`;
    });

    table.innerHTML = tableHTML;
}

function onModeSelect() {
    stopModeAnimation();
    const sel = document.getElementById('mode-selector');
    const idx = parseInt(sel.value);
    if (isNaN(idx)) {
        restoreOriginalPositions();
        hideModeLegend();
        return;
    }

    const modal = currentResult?.modal_analysis;
    if (!modal?.modes?.[idx]?.shape) return;

    const mode = modal.modes[idx];
    applyModeShape(mode.shape, 1.0);
    showModeLegend(mode);
}

function jetColormap(t) {
    // t: 0.0 (blue/min) → 1.0 (red/max), returns THREE.Color
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.25) {
        r = 0; g = t * 4; b = 1;
    } else if (t < 0.5) {
        r = 0; g = 1; b = 1 - (t - 0.25) * 4;
    } else if (t < 0.75) {
        r = (t - 0.5) * 4; g = 1; b = 0;
    } else {
        r = 1; g = 1 - (t - 0.75) * 4; b = 0;
    }
    return new THREE.Color(r, g, b);
}

function applyModeShape(shape, amplitude) {
    if (!currentResult?.viewer) return;
    saveOriginalState();

    const absAmp = Math.abs(amplitude);
    const scale = 3.0 * amplitude; // Visual scale factor
    const nodes = currentResult.viewer.nodes;

    // Compute per-node displacement magnitude (normalized 0-1)
    const nodeDisp = {};
    nodes.forEach(n => {
        const s = shape[String(n.id)];
        if (s) {
            nodeDisp[n.id] = Math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        } else {
            nodeDisp[n.id] = 0;
        }
    });

    const deformedPos = {};
    nodes.forEach(n => {
        const s = shape[String(n.id)];
        const dx = s ? s[0] * scale : 0;
        const dy = s ? s[1] * scale : 0;
        const dz = s ? s[2] * scale : 0;
        // Three.js: (x, z, -y) — consistent with buildScene
        deformedPos[n.id] = new THREE.Vector3(n.x + dx, n.z + dz, -(n.y + dy));
    });

    const yAxis = new THREE.Vector3(0, 1, 0);
    memberMeshes.forEach(({ mesh, elementData }) => {
        const pi = deformedPos[elementData.ni];
        const pj = deformedPos[elementData.nj];
        if (!pi || !pj) return;

        const dir = new THREE.Vector3().subVectors(pj, pi);
        const newLen = dir.length();
        const mid = new THREE.Vector3().addVectors(pi, pj).multiplyScalar(0.5);

        mesh.position.copy(mid);
        mesh.quaternion.setFromUnitVectors(yAxis, dir.clone().normalize());
        mesh.scale.y = newLen / (mesh.geometry.parameters.height || newLen);

        // Color by displacement magnitude (average of two end nodes)
        const di = nodeDisp[elementData.ni] || 0;
        const dj = nodeDisp[elementData.nj] || 0;
        const avgDisp = (di + dj) / 2 * absAmp;
        mesh.material.color.copy(jetColormap(avgDisp));
        mesh.material.opacity = 1.0;
    });

    nodeMeshes.forEach(m => {
        const nid = m.userData?.nodeId;
        if (nid === undefined || nid === null) return;
        if (deformedPos[nid]) {
            m.position.copy(deformedPos[nid]);
            m.updateMatrix();
        }
        // Color node spheres too
        const d = (nodeDisp[nid] || 0) * absAmp;
        m.material.color.copy(jetColormap(d));
        m.material.opacity = 1.0;
    });
}

function toggleModeAnimation() {
    if (modeAnimating) {
        stopModeAnimation();
    } else {
        startModeAnimation();
    }
}

function startModeAnimation() {
    const sel = document.getElementById('mode-selector');
    const idx = parseInt(sel.value);
    if (isNaN(idx)) return;

    const modal = currentResult?.modal_analysis;
    if (!modal?.modes?.[idx]?.shape) return;

    const shape = modal.modes[idx].shape;
    modeAnimating = true;
    document.getElementById('btn-mode-animate').innerHTML = '&#9724;'; // Stop icon

    let t = 0;
    function animateMode() {
        if (!modeAnimating) return;
        t += 0.04;
        const amplitude = Math.sin(t);
        applyModeShape(shape, amplitude);
        modeAnimationId = requestAnimationFrame(animateMode);
    }
    animateMode();
}

function stopModeAnimation() {
    modeAnimating = false;
    if (modeAnimationId) {
        cancelAnimationFrame(modeAnimationId);
        modeAnimationId = null;
    }
    document.getElementById('btn-mode-animate').innerHTML = '&#9654;'; // Play icon
}

function showModeLegend(mode) {
    const legend = document.getElementById('mode-color-legend');
    if (!legend) return;

    // Build gradient bar (top=red=max, bottom=blue=min)
    const bar = document.getElementById('legend-bar');
    bar.style.background = 'linear-gradient(to bottom, #ff0000, #ff8800, #ffff00, #00ff00, #00ffff, #0000ff)';

    // Build labels (10 steps)
    const labels = document.getElementById('legend-labels');
    const steps = 6;
    let html = '';
    for (let i = 0; i <= steps; i++) {
        const val = (1.0 - i / steps);
        html += `<span>${val.toFixed(2)}</span>`;
    }
    labels.innerHTML = html;

    // Info line
    const info = document.getElementById('legend-info');
    const mp = mode.mass_participation || {};
    info.innerHTML = `MODE ${mode.mode}<br>T = ${mode.period_s} s<br>${mode.direction}`
        + `<br>Mass: X=${(mp.x_pct||0).toFixed(1)}% Y=${(mp.y_pct||0).toFixed(1)}%`;

    legend.style.display = 'block';
}

function hideModeLegend() {
    const legend = document.getElementById('mode-color-legend');
    if (legend) legend.style.display = 'none';
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

function _dcColorForMember(mc) {
    if (!mc) return null;
    if (mc.status === 'OK') {
        return (mc.interaction_ratio > 0.7) ? COLORS.dc_marginal : COLORS.dc_ok;
    }
    return COLORS.dc_ng;
}

function applyDesignCheckColors(memberChecks) {
    // Wireframe (cylinder) meshes
    memberMeshes.forEach(({ mesh, elementData }) => {
        const mc = memberChecks[String(elementData.id)];
        const dcColor = _dcColorForMember(mc);
        if (dcColor === null) return;
        if (selectedMeshSet.has(mesh)) {
            // Update stored orig color so deselection restores DC color
            mesh.userData._origColor = dcColor;
            return; // don't override selection highlight
        }
        mesh.material.color.setHex(dcColor);
    });
    // Solid section meshes (when solid mode is on)
    if (window._solidMeshMap) {
        Object.entries(window._solidMeshMap).forEach(([elemId, solidMesh]) => {
            const mc = memberChecks[String(elemId)];
            const dcColor = _dcColorForMember(mc);
            if (dcColor === null) return;
            solidMesh.material.color.setHex(dcColor);
            solidMesh.userData._dcColor = dcColor;
        });
    }
}

function resetElementColors() {
    memberMeshes.forEach(({ mesh, elementData }) => {
        if (selectedMeshSet.has(mesh)) {
            mesh.userData._origColor = getElementColor(elementData);
            return; // don't override selection highlight
        }
        mesh.material.color.setHex(getElementColor(elementData));
    });
    // Solid meshes: restore original color
    if (window._solidMeshMap) {
        Object.values(window._solidMeshMap).forEach(solidMesh => {
            const origColor = solidMesh.userData._solidOrigColor;
            if (origColor !== undefined) {
                solidMesh.material.color.setHex(origColor);
            }
            delete solidMesh.userData._dcColor;
        });
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

// ─── View Presets (XY/XZ/YZ/ISO) ────────────────────────────────────────
// Returns model center and distance in Three.js coordinates (x, z_up, -y)
function getModelCenterThreeJS() {
    // From V2 model nodes — same transform as buildV2PreviewScene: (n.x, n.z, -n.y)
    if (window._v2Model && window._v2Model.nodes.length > 0) {
        const ns = window._v2Model.nodes;
        let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,minZ=Infinity,maxZ=-Infinity;
        ns.forEach(n => {
            const tx = n.x, ty = n.z, tz = -n.y; // struct → Three.js
            minX=Math.min(minX,tx); maxX=Math.max(maxX,tx);
            minY=Math.min(minY,ty); maxY=Math.max(maxY,ty);
            minZ=Math.min(minZ,tz); maxZ=Math.max(maxZ,tz);
        });
        return {
            x: (minX+maxX)/2, y: (minY+maxY)/2, z: (minZ+maxZ)/2,
            dist: Math.max(maxX-minX, maxY-minY, maxZ-minZ, 1) * 1.8
        };
    }
    // From analysis result viewer
    if (currentResult?.viewer) {
        const v = currentResult.viewer;
        const cx = v.total_width_x / 2;
        const cy = v.total_height / 2;
        const cz = v.total_width_y / 2;
        return {
            x: cx, y: cy, z: cz,
            dist: Math.max(v.total_width_x, v.total_width_y, v.total_height) * 1.8
        };
    }
    return { x: 0, y: 5, z: 0, dist: 40 };
}

function setViewPreset(preset) {
    const c = getModelCenterThreeJS();
    const target = new THREE.Vector3(c.x, c.y, c.z);
    const d = c.dist;
    let pos;
    switch (preset) {
        case 'front':  // 정면: -Y방향(struct)에서 봄 → Three.js +Z에서 봄
            pos = new THREE.Vector3(c.x, c.y, c.z + d);
            break;
        case 'right':  // 우측면: +X방향에서 봄
            pos = new THREE.Vector3(c.x + d, c.y, c.z);
            break;
        case 'top':    // 평면: 위에서 봄
            pos = new THREE.Vector3(c.x, c.y + d, c.z + 0.001);
            break;
        case 'iso':    // Isometric
            pos = new THREE.Vector3(c.x + d*0.6, c.y + d*0.4, c.z + d*0.6);
            break;
        default:
            pos = new THREE.Vector3(c.x + d*0.6, c.y + d*0.4, c.z + d*0.6);
    }
    // Animate camera smoothly
    const startPos = camera.position.clone();
    const startTarget = controls.target.clone();
    const duration = 300;
    const startTime = performance.now();
    function animateView(now) {
        const t = Math.min((now - startTime) / duration, 1);
        const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t; // easeInOut
        camera.position.lerpVectors(startPos, pos, ease);
        controls.target.lerpVectors(startTarget, target, ease);
        controls.update();
        if (t < 1) requestAnimationFrame(animateView);
    }
    requestAnimationFrame(animateView);
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


// ─── Irregular Building (Zones) ───────────────────────────────────────────
let zonesData = [];

function toggleIrregular() {
    const on = document.getElementById('irregular-toggle').checked;
    document.getElementById('regular-bays-panel').style.display = on ? 'none' : '';
    document.getElementById('irregular-zones-panel').style.display = on ? '' : 'none';
    const hint = document.getElementById('irregular-hint');
    if (hint) hint.style.display = on ? '' : 'none';
    if (on && zonesData.length === 0) {
        // Initialize with current bays as Zone A
        const bx = parseBaysFromText('x');
        const by = parseBaysFromText('y');
        zonesData = [{ id: 'A', bays_x: bx.length ? bx : [8.0, 8.0], bays_y: by.length ? by : [8.0, 8.0], origin_x: 0, origin_y: 0, story_from: 1, story_to: null }];
        renderZoneList();
    }
    updateManualPreview();
}

function applyZonePreset() {
    const preset = document.getElementById('zone-preset').value;
    const stories = getStoriesFromEditor();
    const ns = stories.length || 5;

    if (preset === 'L-shape') {
        zonesData = [
            { id: 'A', bays_x: [8, 8, 8], bays_y: [8, 8], origin_x: 0, origin_y: 0, story_from: 1, story_to: null },
            { id: 'B', bays_x: [8, 8], bays_y: [8, 8], origin_x: 0, origin_y: 16, story_from: 1, story_to: null },
        ];
    } else if (preset === 'T-shape') {
        zonesData = [
            { id: 'A', bays_x: [8, 8, 8, 8], bays_y: [8, 8], origin_x: 0, origin_y: 0, story_from: 1, story_to: null },
            { id: 'B', bays_x: [8, 8], bays_y: [8, 8], origin_x: 8, origin_y: 16, story_from: 1, story_to: null },
        ];
    } else if (preset === 'setback') {
        const mid = Math.ceil(ns / 2);
        zonesData = [
            { id: 'Base', bays_x: [8, 8, 8], bays_y: [8, 8, 8], origin_x: 0, origin_y: 0, story_from: 1, story_to: mid },
            { id: 'Tower', bays_x: [8, 8], bays_y: [8, 8], origin_x: 4, origin_y: 4, story_from: 1, story_to: null },
        ];
    } else {
        return;
    }
    renderZoneList();
    drawZonePlan();
    updateManualPreview();
}

function addZone() {
    const nextId = String.fromCharCode(65 + zonesData.length); // A, B, C, ...
    zonesData.push({ id: nextId, bays_x: [8], bays_y: [8], origin_x: 0, origin_y: 0, story_from: 1, story_to: null });
    renderZoneList();
    drawZonePlan();
}

function removeZone(idx) {
    zonesData.splice(idx, 1);
    renderZoneList();
    drawZonePlan();
    updateManualPreview();
}

function renderZoneList() {
    const container = document.getElementById('zone-list-container');
    container.innerHTML = '';
    zonesData.forEach((z, idx) => {
        const color = ZONE_COLORS[idx % ZONE_COLORS.length];
        const row = document.createElement('div');
        row.className = 'zone-row';
        row.style.borderLeftColor = color;
        row.innerHTML = `
            <div class="zone-header">
                <strong style="color:${color}">Zone ${z.id}</strong>
                <button class="btn-remove-zone" onclick="removeZone(${idx})" title="삭제">&times;</button>
            </div>
            <div class="zone-fields-v">
                <div class="zf-row">
                    <label>이름</label>
                    <input type="text" value="${z.id}" style="width:60px"
                           onchange="zonesData[${idx}].id=this.value; renderZoneList()">
                </div>
                <div class="zf-row">
                    <label>X방향 경간 (m)</label>
                    <input type="text" value="${z.bays_x.join(', ')}" placeholder="8, 8, 8"
                           onchange="zonesData[${idx}].bays_x=this.value.split(',').map(Number).filter(v=>v>0); drawZonePlan(); updateManualPreview()">
                </div>
                <div class="zf-row">
                    <label>Y방향 경간 (m)</label>
                    <input type="text" value="${z.bays_y.join(', ')}" placeholder="8, 8"
                           onchange="zonesData[${idx}].bays_y=this.value.split(',').map(Number).filter(v=>v>0); drawZonePlan(); updateManualPreview()">
                </div>
                <div class="zf-row">
                    <label>위치 오프셋 X, Y (m)</label>
                    <div class="zf-pair">
                        <input type="number" value="${z.origin_x}" step="1" placeholder="X"
                               onchange="zonesData[${idx}].origin_x=parseFloat(this.value)||0; drawZonePlan(); updateManualPreview()">
                        <input type="number" value="${z.origin_y}" step="1" placeholder="Y"
                               onchange="zonesData[${idx}].origin_y=parseFloat(this.value)||0; drawZonePlan(); updateManualPreview()">
                    </div>
                </div>
                <div class="zf-row">
                    <label>적용 층 (시작 ~ 끝)</label>
                    <div class="zf-pair">
                        <input type="number" value="${z.story_from}" min="1" placeholder="1"
                               onchange="zonesData[${idx}].story_from=parseInt(this.value)||1; drawZonePlan(); updateManualPreview()">
                        <input type="text" value="${z.story_to || '전체'}" placeholder="전체"
                               onchange="zonesData[${idx}].story_to=this.value==='전체'||this.value==='all'?null:(parseInt(this.value)||null); drawZonePlan(); updateManualPreview()">
                    </div>
                </div>
            </div>
        `;
        container.appendChild(row);
    });
    drawZonePlan();
}

function getZonesFromEditor() {
    return zonesData.map(z => ({
        id: z.id,
        bays_x: z.bays_x,
        bays_y: z.bays_y,
        origin_x: z.origin_x || 0,
        origin_y: z.origin_y || 0,
        story_from: z.story_from || 1,
        story_to: z.story_to || null,
    })).filter(z => z.bays_x.length > 0 && z.bays_y.length > 0);
}

const ZONE_COLORS = ['#4285f4', '#34a853', '#fbbc04', '#ea4335', '#9c27b0', '#00bcd4'];

function drawZonePlan() {
    const canvas = document.getElementById('zone-plan-canvas');
    if (!canvas) return;
    // Handle HiDPI
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    if (zonesData.length === 0) {
        ctx.fillStyle = '#999';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('존을 추가하면 평면도가 표시됩니다', W/2, H/2);
        return;
    }

    // Compute bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    zonesData.forEach(z => {
        const wx = z.bays_x.reduce((a, b) => a + b, 0);
        const wy = z.bays_y.reduce((a, b) => a + b, 0);
        minX = Math.min(minX, z.origin_x || 0);
        maxX = Math.max(maxX, (z.origin_x || 0) + wx);
        minY = Math.min(minY, z.origin_y || 0);
        maxY = Math.max(maxY, (z.origin_y || 0) + wy);
    });

    const pad = 30;
    const scaleX = (W - 2 * pad) / (maxX - minX || 1);
    const scaleY = (H - 2 * pad) / (maxY - minY || 1);
    const scale = Math.min(scaleX, scaleY);

    const offX = pad + ((W - 2 * pad) - (maxX - minX) * scale) / 2;
    const offY = pad + ((H - 2 * pad) - (maxY - minY) * scale) / 2;

    function tx(x) { return offX + (x - minX) * scale; }
    function ty(y) { return H - offY - (y - minY) * scale; }

    // Draw zones
    zonesData.forEach((z, idx) => {
        const wx = z.bays_x.reduce((a, b) => a + b, 0);
        const wy = z.bays_y.reduce((a, b) => a + b, 0);
        const ox = z.origin_x || 0, oy = z.origin_y || 0;
        const x0 = tx(ox);
        const y0 = ty(oy + wy);
        const w = wx * scale;
        const h = wy * scale;
        const color = ZONE_COLORS[idx % ZONE_COLORS.length];

        // Fill + border
        ctx.fillStyle = color + '22';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.fillRect(x0, y0, w, h);
        ctx.strokeRect(x0, y0, w, h);

        // Grid lines
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = color + '88';
        ctx.setLineDash([3, 3]);
        let gx = ox;
        for (let i = 0; i <= z.bays_x.length; i++) {
            const px = tx(gx);
            ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, y0 + h); ctx.stroke();
            if (i < z.bays_x.length) gx += z.bays_x[i];
        }
        let gy = oy;
        for (let j = 0; j <= z.bays_y.length; j++) {
            const py = ty(gy);
            ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x0 + w, py); ctx.stroke();
            if (j < z.bays_y.length) gy += z.bays_y[j];
        }
        ctx.setLineDash([]);

        // Zone label + area
        ctx.fillStyle = color;
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'left';
        const area = wx * wy;
        ctx.fillText(`${z.id}`, x0 + 4, y0 + 13);
        ctx.font = '10px sans-serif';
        ctx.fillText(`${wx}×${wy}m`, x0 + 4, y0 + 25);
        ctx.fillStyle = color + 'aa';
        ctx.fillText(`${area}m²`, x0 + 4, y0 + 36);

        // Dimension arrows (X width at bottom, Y width at left)
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1;
        ctx.font = '9px sans-serif';
        // X dimension below
        if (w > 30) {
            const dy_dim = y0 + h + 10;
            ctx.beginPath(); ctx.moveTo(x0, dy_dim); ctx.lineTo(x0 + w, dy_dim); ctx.stroke();
            ctx.textAlign = 'center';
            ctx.fillText(`${wx}m`, x0 + w/2, dy_dim + 10);
        }
        // Y dimension to the right
        if (h > 30) {
            const dx_dim = x0 + w + 6;
            ctx.beginPath(); ctx.moveTo(dx_dim, y0); ctx.lineTo(dx_dim, y0 + h); ctx.stroke();
            ctx.save();
            ctx.translate(dx_dim + 10, y0 + h/2);
            ctx.rotate(-Math.PI/2);
            ctx.textAlign = 'center';
            ctx.fillText(`${wy}m`, 0, 0);
            ctx.restore();
        }
    });

    // Axes indicator
    ctx.strokeStyle = '#666';
    ctx.fillStyle = '#666';
    ctx.lineWidth = 1;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    // X arrow
    ctx.beginPath(); ctx.moveTo(10, H-10); ctx.lineTo(40, H-10); ctx.stroke();
    ctx.fillText('X', 45, H-7);
    // Y arrow
    ctx.beginPath(); ctx.moveTo(10, H-10); ctx.lineTo(10, H-40); ctx.stroke();
    ctx.fillText('Y', 10, H-44);
}

function buildIrregularPreview(data) {
    clearPreviewScene();
    if (!scene) return;

    const zones = data.zones || [];
    const stories = data.stories || [];
    if (zones.length === 0 || stories.length === 0) return;

    const zCoords = [0];
    stories.forEach(s => zCoords.push(zCoords[zCoords.length - 1] + (s.height || 3.5)));
    const ns = stories.length;

    // Same materials as regular preview
    const colMat = new THREE.LineBasicMaterial({ color: 0x4285f4, linewidth: 2 });
    const beamXMat = new THREE.LineBasicMaterial({ color: 0x34a853, linewidth: 2 });
    const beamYMat = new THREE.LineBasicMaterial({ color: 0xfbbc04, linewidth: 2 });
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    const nodeGeo = new THREE.SphereGeometry(0.12, 6, 6);
    const triGeo = new THREE.ConeGeometry(0.3, 0.4, 4);
    const triMat = new THREE.MeshBasicMaterial({ color: 0xff6600 });

    // Same coordinate swap as regular: Three.js (x, z, -y)
    function addLine(p1, p2, mat) {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(p1[0], p1[2], -p1[1]),
            new THREE.Vector3(p2[0], p2[2], -p2[1]),
        ]);
        const line = new THREE.Line(geo, mat);
        scene.add(line);
        previewMeshes.push(line);
    }

    // Track drawn nodes to avoid duplicates at zone boundaries
    const drawnNodes = new Set();
    const drawnCols = new Set();
    const drawnBeamsX = new Set();
    const drawnBeamsY = new Set();

    function nk(x, y, z) { return `${x.toFixed(3)}_${y.toFixed(3)}_${z.toFixed(3)}`; }
    function ek(x1, y1, z1, x2, y2, z2) { return nk(x1,y1,z1) + '|' + nk(x2,y2,z2); }

    zones.forEach(zone => {
        const ox = zone.origin_x || 0;
        const oy = zone.origin_y || 0;
        const xc = [ox];
        zone.bays_x.forEach(b => xc.push(xc[xc.length - 1] + b));
        const yc = [oy];
        zone.bays_y.forEach(b => yc.push(yc[yc.length - 1] + b));
        const sf = zone.story_from || 1;
        const st = zone.story_to || ns;

        // Columns
        for (let s = Math.max(0, sf - 1); s < Math.min(st, ns); s++) {
            xc.forEach(x => yc.forEach(y => {
                const key = ek(x,y,zCoords[s], x,y,zCoords[s+1]);
                if (!drawnCols.has(key)) {
                    drawnCols.add(key);
                    addLine([x, y, zCoords[s]], [x, y, zCoords[s+1]], colMat);
                }
            }));
        }

        // Beams X
        for (let s = sf; s <= Math.min(st, ns); s++) {
            yc.forEach(y => {
                for (let i = 0; i < xc.length - 1; i++) {
                    const key = ek(xc[i],y,zCoords[s], xc[i+1],y,zCoords[s]);
                    if (!drawnBeamsX.has(key)) {
                        drawnBeamsX.add(key);
                        addLine([xc[i], y, zCoords[s]], [xc[i+1], y, zCoords[s]], beamXMat);
                    }
                }
            });
        }

        // Beams Y
        for (let s = sf; s <= Math.min(st, ns); s++) {
            xc.forEach(x => {
                for (let j = 0; j < yc.length - 1; j++) {
                    const key = ek(x,yc[j],zCoords[s], x,yc[j+1],zCoords[s]);
                    if (!drawnBeamsY.has(key)) {
                        drawnBeamsY.add(key);
                        addLine([x, yc[j], zCoords[s]], [x, yc[j+1], zCoords[s]], beamYMat);
                    }
                }
            });
        }

        // Nodes
        for (let s = Math.max(0, sf - 1); s <= Math.min(st, ns); s++) {
            xc.forEach(x => yc.forEach(y => {
                const key = nk(x, y, zCoords[s]);
                if (!drawnNodes.has(key)) {
                    drawnNodes.add(key);
                    const sphere = new THREE.Mesh(nodeGeo, nodeMat);
                    sphere.position.set(x, zCoords[s], -y);
                    scene.add(sphere);
                    previewMeshes.push(sphere);
                }
            }));
        }

        // Support triangles at base
        if (sf <= 1) {
            xc.forEach(x => yc.forEach(y => {
                const key = nk(x, y, 0);
                if (!drawnNodes.has('tri_' + key)) {
                    drawnNodes.add('tri_' + key);
                    const tri = new THREE.Mesh(triGeo, triMat);
                    tri.position.set(x, -0.2, -y);
                    scene.add(tri);
                    previewMeshes.push(tri);
                }
            }));
        }
    });

    // Show preview badge
    const badge = document.getElementById('preview-badge');
    if (badge) badge.style.display = 'block';

    // Fit camera
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    zones.forEach(z => {
        const wx = z.bays_x.reduce((a, b) => a + b, 0);
        const wy = z.bays_y.reduce((a, b) => a + b, 0);
        minX = Math.min(minX, z.origin_x || 0);
        maxX = Math.max(maxX, (z.origin_x || 0) + wx);
        minY = Math.min(minY, z.origin_y || 0);
        maxY = Math.max(maxY, (z.origin_y || 0) + wy);
    });
    const maxZ = zCoords[zCoords.length - 1];
    const cx = (minX + maxX) / 2;
    const cy = maxZ / 2;
    const cz = -(minY + maxY) / 2;
    const size = Math.max(maxX - minX, maxY - minY, maxZ);
    const dist = size * 1.8;
    camera.position.set(cx + dist * 0.7, cy + dist * 0.5, cz + dist * 0.7);
    controls.target.set(cx, cy, cz);
    controls.update();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Display Filter System + Load Arrow Visualization
// ═══════════════════════════════════════════════════════════════════════════════
let _loadArrowGroup = null;
let _reactionLabelSprites = [];

const _LOAD_COLORS = {
    DL:  0x00bcd4,  // cyan — 기둥(blue)과 구분
    LL:  0xff9800,  // orange — 보(green)와 구분
    EQX: 0xf44336,  // red
    EQY: 0xff5722,  // deep orange
    WX:  0x9c27b0,  // purple
    WY:  0xab47bc,  // light purple
};

// Load category → case name mapping
const _LOAD_CATEGORY = {
    DL: ['DL'], LL: ['LL'], EQ: ['EQX','EQY'], Wind: ['WX','WY'],
};

// ─── Global filter state ──────────────────────────────────────────────────
window._displayFilter = {
    loads: { DL: false, LL: false, EQ: false, Wind: false },
    stories: [],   // populated after analysis
    types: ['column', 'beam', 'brace'],
};

function toggleFilterPanel() {
    const p = document.getElementById('display-filter-panel');
    if (!p) return;
    p.style.display = p.style.display === 'none' ? '' : 'none';
}

function initFilterStories() {
    // 모델 층수에 따라 story 체크박스 동적 생성
    const model = window._v2Model;
    if (!model) return;
    const elevs = model.story_elevations || [];
    const nStories = Math.max(0, elevs.length - 1);
    const container = document.getElementById('filter-story-checks');
    if (!container) return;
    container.innerHTML = '';
    window._displayFilter.stories = [];
    for (let i = 1; i <= nStories; i++) {
        window._displayFilter.stories.push(i);
        const lbl = document.createElement('label');
        lbl.style.marginRight = '8px';
        lbl.innerHTML = `<input type="checkbox" data-filter="story" value="${i}" onchange="onFilterChange()" checked> ${i}F`;
        container.appendChild(lbl);
    }
}

function filterSelectAll(filterType, checked) {
    document.querySelectorAll(`input[data-filter="${filterType}"]`).forEach(cb => {
        cb.checked = checked;
    });
    onFilterChange();
}

function onFilterChange() {
    const f = window._displayFilter;

    // Loads
    f.loads = { DL: false, LL: false, EQ: false, Wind: false };
    document.querySelectorAll('input[data-filter="load"]').forEach(cb => {
        f.loads[cb.value] = cb.checked;
    });

    // Stories
    f.stories = [];
    document.querySelectorAll('input[data-filter="story"]').forEach(cb => {
        if (cb.checked) f.stories.push(parseInt(cb.value));
    });

    // Types
    f.types = [];
    document.querySelectorAll('input[data-filter="type"]').forEach(cb => {
        if (cb.checked) f.types.push(cb.value);
    });

    applyDisplayFilter();
}

function applyDisplayFilter() {
    const f = window._displayFilter;
    _applyMemberVisibility(f);
    _buildLoadArrows();  // 하중 화살표 갱신
}

function _getMemberStory(elemData) {
    // 부재의 층 판별: node_i의 story (상층 노드 기준)
    if (!window._v2Model) return null;
    const nodes = window._v2Model.nodes || [];
    const ni = nodes.find(n => n.id === elemData.ni);
    const nj = nodes.find(n => n.id === elemData.nj);
    if (!ni && !nj) return null;
    // column: 상단 노드 story, beam: 해당 story
    const s1 = ni?.story, s2 = nj?.story;
    return Math.max(s1 || 0, s2 || 0);
}

function _applyMemberVisibility(f) {
    const storySet = new Set(f.stories);
    const typeSet = new Set();
    f.types.forEach(t => {
        typeSet.add(t);
        if (t === 'beam') { typeSet.add('beam_x'); typeSet.add('beam_y'); }
    });

    // 부재별 visibility 맵 (solid mesh 동기화용)
    const _elemVisibility = {};
    memberMeshes.forEach(({ mesh, elementData }) => {
        const mType = elementData?.type || 'beam_x';
        const mStory = _getMemberStory(elementData);

        const typeOk = typeSet.has(mType) || typeSet.has('beam') && (mType === 'beam_x' || mType === 'beam_y');
        const storyOk = mStory === null || mStory === 0 || storySet.has(mStory);
        const vis = typeOk && storyOk;

        mesh.visible = window.solidMode ? false : vis;  // solid mode면 wire는 항상 숨김
        if (elementData?.id != null) _elemVisibility[elementData.id] = vis;
    });

    // Solid mesh에도 같은 필터 적용
    (window.solidMeshes || []).forEach(function(sm) {
        const eid = sm.userData?._solidElementId;
        if (eid != null && _elemVisibility[eid] !== undefined) {
            sm.visible = _elemVisibility[eid];
        }
    });

    // 노드 visibility: 연결된 부재가 하나라도 보이면 표시
    const visibleNodeIds = new Set();
    memberMeshes.forEach(({ mesh, elementData }) => {
        if (mesh.visible) {
            visibleNodeIds.add(elementData.ni);
            visibleNodeIds.add(elementData.nj);
        }
    });
    nodeMeshes.forEach(m => {
        const nid = m.userData?.nodeId;
        m.visible = !nid || visibleNodeIds.has(nid);
    });
}

// Legacy toggle (for backwards compat)
function toggleLoadArrows() { onFilterChange(); }

function _clearLoadArrows() {
    if (_loadArrowGroup) {
        scene.remove(_loadArrowGroup);
        _loadArrowGroup.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
        _loadArrowGroup = null;
    }
}

function _buildLoadArrows() {
    _clearLoadArrows();
    _reactionLabelSprites = [];
    if (!currentResult || !window._v2Model) return;

    const loadCases = currentResult.load_cases_raw;
    if (!loadCases) return;

    // 현재 선택된 케이스: combo면 구성 case 추출, 개별 case면 직접 매칭
    const sel = document.getElementById('case-selector');
    const selectedCase = sel?.value || '__envelope__';

    _loadArrowGroup = new THREE.Group();
    _loadArrowGroup.name = 'loadArrows';

    const model = window._v2Model;
    const nodes = model.nodes || [];
    const elements = model.elements || [];
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    // 모델 크기 기반 화살표 스케일
    let maxDim = 1;
    nodes.forEach(n => { maxDim = Math.max(maxDim, Math.abs(n.x), Math.abs(n.y), Math.abs(n.z)); });

    // 최대 하중값 (정규화용) - gravity / lateral 각각
    let maxGrav = 0, maxLat = 0;
    Object.values(loadCases).forEach(loads => {
        loads.forEach(ld => {
            if (ld.type === 'floor_area') maxGrav = Math.max(maxGrav, Math.abs(ld.value || 0));
            else maxLat = Math.max(maxLat, Math.abs(ld.value || 0));
        });
    });
    if (maxGrav < 0.001) maxGrav = 1;
    if (maxLat < 0.001) maxLat = 1;

    // 필터에서 활성화된 하중 카테고리 → case 이름 목록
    const f = window._displayFilter || {};
    const enabledCases = new Set();
    Object.entries(f.loads || {}).forEach(([cat, on]) => {
        if (on && _LOAD_CATEGORY[cat]) {
            _LOAD_CATEGORY[cat].forEach(cn => enabledCases.add(cn));
        }
    });

    // Case selector와 교차: 선택된 케이스 중 필터 활성인 것만
    let casesToShow;
    if (selectedCase === '__envelope__') {
        casesToShow = Object.keys(loadCases).filter(cn => enabledCases.has(cn));
    } else if (loadCases[selectedCase]) {
        casesToShow = enabledCases.has(selectedCase) ? [selectedCase] : [];
    } else {
        // combo → 구성 case 추출
        casesToShow = [];
        Object.keys(loadCases).forEach(cn => {
            if (selectedCase.includes(cn) && enabledCases.has(cn)) casesToShow.push(cn);
        });
    }
    const showReactions = document.getElementById('toggle-reactions')?.checked;
    if (casesToShow.length === 0 && !showReactions) { scene.add(_loadArrowGroup); return; }

    // Story 필터
    const storyFilter = new Set(f.stories || []);

    const gravArrowLen = maxDim * 0.12;  // 분포하중 화살표 높이
    const latArrowScale = maxDim * 0.3;  // 수평하중 화살표 길이

    // 보 요소 (분포하중 표시용)
    const beamElements = elements.filter(e =>
        e.elem_type === 'beam' || e.elem_type === 'beam_x' || e.elem_type === 'beam_y'
    );

    casesToShow.forEach(caseName => {
        const loads = loadCases[caseName];
        if (!loads) return;
        const color = _LOAD_COLORS[caseName] || 0x888888;

        loads.forEach(ld => {
            const story = ld.story;
            const value = ld.value || 0;
            if (Math.abs(value) < 0.001) return;

            // Story 필터 적용
            if (storyFilter.size > 0 && !storyFilter.has(story)) return;

            const storyNodes = nodes.filter(n => n.story === story && !n.support);
            if (storyNodes.length === 0) return;

            if (ld.type === 'floor_area') {
                // ── 분포하중: 보 위에 화살표 배열 ──
                // Beam이 Type 필터에서 꺼져있으면 floor_area도 숨김
                const typeSet = new Set(f.types || []);
                if (!typeSet.has('beam')) return;

                const norm = Math.abs(value) / maxGrav;
                const aLen = gravArrowLen * Math.max(norm, 0.3);
                const storyNodeIds = new Set(storyNodes.map(n => n.id));
                let labelPlaced = false;

                beamElements.forEach(e => {
                    const ni = nodeMap[e.node_i], nj = nodeMap[e.node_j];
                    if (!ni || !nj) return;
                    if (!storyNodeIds.has(e.node_i) && !storyNodeIds.has(e.node_j)) return;

                    // 보 길이에 따른 화살표 개수 (2m당 1개 — 밀도 절반)
                    const dx = nj.x-ni.x, dy = nj.y-ni.y, dz = nj.z-ni.z;
                    const L = Math.sqrt(dx*dx+dy*dy+dz*dz);
                    const nArrows = Math.max(2, Math.round(L / 2));
                    const dir = new THREE.Vector3(0, -1, 0);

                    for (let k = 0; k <= nArrows; k++) {
                        const t = k / nArrows;
                        const px = ni.x + dx*t;
                        const py = ni.y + dy*t;
                        const pz = ni.z + dz*t;
                        const origin = new THREE.Vector3(px, pz + aLen, -(py));
                        const arrow = new THREE.ArrowHelper(dir, origin, aLen, color, aLen*0.35, aLen*0.15);
                        // 반투명 처리
                        arrow.cone.material.transparent = true;
                        arrow.cone.material.opacity = 0.7;
                        arrow.line.material.transparent = true;
                        arrow.line.material.opacity = 0.7;
                        _loadArrowGroup.add(arrow);
                    }

                    // 상단 연결선
                    const topPts = [];
                    for (let k = 0; k <= nArrows; k++) {
                        const t = k / nArrows;
                        topPts.push(new THREE.Vector3(
                            ni.x + dx*t, (ni.z + dz*t) + aLen, -(ni.y + dy*t)
                        ));
                    }
                    const lineGeo = new THREE.BufferGeometry().setFromPoints(topPts);
                    const lineMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 });
                    _loadArrowGroup.add(new THREE.Line(lineGeo, lineMat));

                    // 수치 라벨: 층당 첫 번째 보의 중앙 위에 1회만 표시
                    if (!labelPlaced) {
                        const mx = (ni.x + nj.x) / 2;
                        const my = (ni.y + nj.y) / 2;
                        const mz = (ni.z + nj.z) / 2;
                        _addLoadLabel(
                            new THREE.Vector3(mx, mz + aLen + 0.4, -(my)),
                            caseName + ': ' + value.toFixed(1) + ' kN/m²', color
                        );
                        labelPlaced = true;
                    }
                });

            } else if (ld.type === 'lateral_x' || ld.type === 'lateral_y') {
                // ── 수평하중: 층 중심 + 개별 노드 ──
                const norm = Math.abs(value) / maxLat;
                const len = latArrowScale * Math.max(norm, 0.2);
                const isX = ld.type === 'lateral_x';

                const cx = storyNodes.reduce((s,n)=>s+n.x,0)/storyNodes.length;
                const cy = storyNodes.reduce((s,n)=>s+n.y,0)/storyNodes.length;
                const cz = storyNodes.reduce((s,n)=>s+n.z,0)/storyNodes.length;

                // 층 중심 대형 화살표
                const dir = isX ? new THREE.Vector3(1,0,0) : new THREE.Vector3(0,0,-1);
                const origin = isX
                    ? new THREE.Vector3(cx - len, cz, -(cy))
                    : new THREE.Vector3(cx, cz, -(cy) + len);
                const bigArrow = new THREE.ArrowHelper(dir, origin, len, color, len*0.2, len*0.1);
                bigArrow.cone.material.transparent = true; bigArrow.cone.material.opacity = 0.8;
                bigArrow.line.material.transparent = true; bigArrow.line.material.opacity = 0.8;
                _loadArrowGroup.add(bigArrow);

                // 개별 노드 소형 화살표
                const sLen = len * 0.4;
                storyNodes.forEach(n => {
                    const o = isX
                        ? new THREE.Vector3(n.x - sLen, n.z, -(n.y))
                        : new THREE.Vector3(n.x, n.z, -(n.y) + sLen);
                    const a = new THREE.ArrowHelper(dir, o, sLen, color, sLen*0.25, sLen*0.1);
                    a.cone.material.transparent = true; a.cone.material.opacity = 0.6;
                    a.line.material.transparent = true; a.line.material.opacity = 0.6;
                    _loadArrowGroup.add(a);
                });

                // 값 라벨 (층 중심)
                _addLoadLabel(
                    isX ? new THREE.Vector3(cx - len*0.3, cz + 0.5, -(cy))
                        : new THREE.Vector3(cx, cz + 0.5, -(cy) + len*0.3),
                    caseName + ': ' + value.toFixed(1) + ' kN', color
                );
            }
        });
    });

    // ── 반력 화살표 ──
    if (showReactions && currentResult?.case_data) {
        const rCaseName = _getCurrentCaseName();
        const rcd = currentResult.case_data[rCaseName];
        const reactions = rcd?.reactions;
        if (reactions && reactions.length > 0) {
            const rxColorV = 0xff6f00;   // 수직 반력 (amber)
            const rxColorH = 0xd84315;   // 수평 반력 (deep orange)
            let maxR = 0;
            reactions.forEach(r => {
                maxR = Math.max(maxR, Math.abs(r.RZ_kN||0));
            });
            if (maxR < 0.01) maxR = 1;
            const rxScale = maxDim * 0.15 / maxR;
            const minLen = maxDim * 0.02;  // 최소 화살표 길이

            reactions.forEach(r => {
                const base = new THREE.Vector3(r.x_m||0, 0, -(r.y_m||0));
                const rz = Math.abs(r.RZ_kN||0);

                // RZ (수직 반력) — 모든 지점에 표시
                if (rz > 0.01) {
                    const len = Math.max(rz * rxScale, minLen);
                    const dir = new THREE.Vector3(0, 1, 0);
                    const origin = base.clone().add(new THREE.Vector3(0, -len, 0));
                    const a = new THREE.ArrowHelper(dir, origin, len, rxColorV, len*0.25, len*0.12);
                    a.cone.material.transparent = true; a.cone.material.opacity = 0.85;
                    _loadArrowGroup.add(a);

                    // 수치 라벨
                    const labelPos = origin.clone().add(new THREE.Vector3(0, -0.3, 0));
                    _addReactionLabel(labelPos, (r.RZ_kN).toFixed(1) + ' kN', rxColorV);
                }

                // RX (수평 X)
                if (Math.abs(r.RX_kN||0) > 0.5) {
                    const len = Math.max(Math.abs(r.RX_kN) * rxScale, minLen);
                    const dir = new THREE.Vector3(r.RX_kN > 0 ? 1 : -1, 0, 0);
                    const a = new THREE.ArrowHelper(dir, base.clone(), len, rxColorH, len*0.2, len*0.1);
                    _loadArrowGroup.add(a);
                }
                // RY (수평 Y)
                if (Math.abs(r.RY_kN||0) > 0.5) {
                    const len = Math.max(Math.abs(r.RY_kN) * rxScale, minLen);
                    const dir = new THREE.Vector3(0, 0, r.RY_kN > 0 ? -1 : 1);
                    const a = new THREE.ArrowHelper(dir, base.clone(), len, rxColorH, len*0.2, len*0.1);
                    _loadArrowGroup.add(a);
                }
            });
        }
    }

    // Values 토글 상태에 따라 반력 라벨 표시/숨김
    const valuesOn = document.getElementById('toggle-dgm-values')?.checked ?? true;
    _reactionLabelSprites.forEach(s => { s.visible = valuesOn; });

    scene.add(_loadArrowGroup);
}

function _addReactionLabel(position, text, color) {
    const s = 3;
    const canvas = document.createElement('canvas');
    canvas.width = 160 * s; canvas.height = 36 * s;
    const ctx = canvas.getContext('2d');
    ctx.scale(s, s);
    ctx.clearRect(0, 0, 160, 36);
    const hex = '#' + color.toString(16).padStart(6, '0');
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.strokeText(text, 80, 18);
    ctx.fillStyle = hex;
    ctx.fillText(text, 80, 18);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.position.copy(position);
    sprite.scale.set(4.0, 0.9, 1);
    _loadArrowGroup.add(sprite);
    _reactionLabelSprites.push(sprite);
}

function _addLoadLabel(position, text, color) {
    const s = 3;
    const canvas = document.createElement('canvas');
    canvas.width = 200 * s; canvas.height = 36 * s;
    const ctx = canvas.getContext('2d');
    ctx.scale(s, s);
    ctx.clearRect(0, 0, 200, 36);
    const hex = '#' + color.toString(16).padStart(6, '0');
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.strokeText(text, 100, 18);
    ctx.fillStyle = hex;
    ctx.fillText(text, 100, 18);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.position.copy(position);
    sprite.scale.set(5.0, 0.9, 1);
    sprite.userData._isLoadLabel = true;
    _loadArrowGroup.add(sprite);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Node / Member 번호 표시 (라벨 토글)
// ═══════════════════════════════════════════════════════════════════════════════
let _nodeNumberGroup = null;
let _memberNumberGroup = null;

function _addNumberLabel(group, position, text, color) {
    const s = 3;
    const canvas = document.createElement('canvas');
    canvas.width = 100 * s; canvas.height = 28 * s;
    const ctx = canvas.getContext('2d');
    ctx.scale(s, s);
    ctx.clearRect(0, 0, 100, 28);
    const hex = '#' + color.toString(16).padStart(6, '0');
    ctx.font = 'bold 13px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.strokeText(text, 50, 14);
    ctx.fillStyle = hex;
    ctx.fillText(text, 50, 14);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.position.copy(position);
    sprite.scale.set(2.0, 0.56, 1);
    sprite.renderOrder = 999;
    group.add(sprite);
}

function _disposeGroup(g) {
    if (!g) return;
    scene.remove(g);
    g.traverse(c => {
        if (c.geometry) c.geometry.dispose();
        if (c.material) { if (c.material.map) c.material.map.dispose(); c.material.dispose(); }
    });
}

function _buildNodeNumbers() {
    _disposeGroup(_nodeNumberGroup);
    _nodeNumberGroup = new THREE.Group();
    _nodeNumberGroup.name = 'nodeNumbers';
    const model = window._v2Model;
    if (!model || !model.nodes) { scene.add(_nodeNumberGroup); return; }
    model.nodes.forEach(n => {
        const pos = new THREE.Vector3(n.x, n.z, -n.y);
        _addNumberLabel(_nodeNumberGroup, pos, String(n.id), 0x2e7d32);  // 녹색
    });
    scene.add(_nodeNumberGroup);
}

function _buildMemberNumbers() {
    _disposeGroup(_memberNumberGroup);
    _memberNumberGroup = new THREE.Group();
    _memberNumberGroup.name = 'memberNumbers';
    const model = window._v2Model;
    if (!model || !model.elements || !model.nodes) { scene.add(_memberNumberGroup); return; }
    const nodeMap = {};
    model.nodes.forEach(n => { nodeMap[n.id] = n; });
    model.elements.forEach(e => {
        const ni = nodeMap[e.node_i], nj = nodeMap[e.node_j];
        if (!ni || !nj) return;
        const mid = new THREE.Vector3((ni.x + nj.x)/2, (ni.z + nj.z)/2, -(ni.y + nj.y)/2);
        _addNumberLabel(_memberNumberGroup, mid, String(e.id), 0x6a1b9a);  // 보라
    });
    scene.add(_memberNumberGroup);
}

function toggleNodeNumbers() {
    const on = document.getElementById('toggle-node-numbers')?.checked;
    if (on) {
        _buildNodeNumbers();
    } else {
        _disposeGroup(_nodeNumberGroup);
        _nodeNumberGroup = null;
    }
}

function toggleMemberNumbers() {
    const on = document.getElementById('toggle-member-numbers')?.checked;
    if (on) {
        _buildMemberNumbers();
    } else {
        _disposeGroup(_memberNumberGroup);
        _memberNumberGroup = null;
    }
}

// 모델이 재생성되면 체크 상태 유지한 채 라벨도 재빌드
function _refreshNumberLabels() {
    if (document.getElementById('toggle-node-numbers')?.checked) _buildNodeNumbers();
    if (document.getElementById('toggle-member-numbers')?.checked) _buildMemberNumbers();
}

// Case 선택 변경 시 화살표도 갱신
const _origOnCaseSelect = typeof onCaseSelect === 'function' ? onCaseSelect : null;
function _hookCaseSelectForLoads() {
    const sel = document.getElementById('case-selector');
    if (sel) {
        sel.addEventListener('change', () => {
            if (_loadArrowsVisible) _buildLoadArrows();
        });
    }
}
// DOM ready 시 hook
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _hookCaseSelectForLoads);
} else {
    setTimeout(_hookCaseSelectForLoads, 500);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3B: SFD / BMD / Axial Force Diagram
// ═══════════════════════════════════════════════════════════════════════════════
let _diagramMode = 'off';  // 'off' | 'axial' | 'shear' | 'moment'
let _diagramGroup = null;
let _showDiagramValues = true;
let _diagramScaleMultiplier = 1.0;  // 사용자 스케일 조절 (기본 1× = 자동 스케일)
let _nodeForceLabels = {};  // nid → {pos, normal, vals: [{val, side}]}

// Force key는 부재 타입별로 동적 선택 (_getDiagramValues에서)
const _DGM_KEY = { axial: 'N_kN', shear: null, moment: null };  // shear/moment는 동적
const _DGM_LABEL = { axial: 'N (kN)', shear: 'V (kN)', moment: 'M (kN·m)' };
const _DGM_POS_COLOR = 0xe53935;  // red
const _DGM_NEG_COLOR = 0x1565c0;  // blue

function setDiagramMode(mode) {
    _diagramMode = mode;
    document.querySelectorAll('.dgm-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.dgm === mode);
    });
    // 스케일 슬라이더 표시/숨김
    const sw = document.getElementById('diagram-scale-wrap');
    if (sw) sw.style.display = (mode === 'off') ? 'none' : '';
    if (mode === 'off') {
        _clearDiagrams();
    } else {
        _buildDiagrams();
    }
}

function onDiagramValuesToggle() {
    if (_diagramMode !== 'off') _buildDiagrams();
    // 반력 수치 라벨도 토글
    const show = document.getElementById('toggle-dgm-values')?.checked ?? true;
    _reactionLabelSprites.forEach(s => { s.visible = show; });
}

function onDiagramScaleChange(v) {
    // slider 0~100 → 0.1×~10× (log)
    const sv = parseInt(v, 10);
    _diagramScaleMultiplier = Math.pow(10, (sv - 50) / 50);
    const label = document.getElementById('diagram-scale-val');
    if (label) label.textContent = _diagramScaleMultiplier.toFixed(2) + '×';
    if (_diagramMode !== 'off') _buildDiagrams();
}

function resetDiagramScale() {
    _diagramScaleMultiplier = 1.0;
    const slider = document.getElementById('diagram-scale-slider');
    if (slider) slider.value = 50;
    const label = document.getElementById('diagram-scale-val');
    if (label) label.textContent = '1.00×';
    if (_diagramMode !== 'off') _buildDiagrams();
}

function showDiagramButtons() {
    const w = document.getElementById('diagram-btn-wrap');
    if (w) w.style.display = '';
}

function _clearDiagrams() {
    if (_diagramGroup) {
        scene.remove(_diagramGroup);
        _diagramGroup.traverse(c => {
            if (c.geometry) c.geometry.dispose();
            if (c.material) { if (c.material.map) c.material.map.dispose(); c.material.dispose(); }
        });
        _diagramGroup = null;
    }
}

function _getCurrentCaseName() {
    const sel = document.getElementById('case-selector');
    const v = sel?.value;
    if (!v || v === '__envelope__') {
        // envelope: 첫 번째 combo 사용
        const combos = currentResult?.combo_names || [];
        return combos[0] || (currentResult?.case_names?.[0]) || 'DL';
    }
    return v;
}

function _buildDiagrams() {
    _clearDiagrams();
    if (_diagramMode === 'off' || !currentResult) return;

    const caseName = _getCurrentCaseName();
    const mfAll = currentResult.member_forces;
    const mfCase = mfAll?.[caseName];
    if (!mfCase || mfCase.length === 0) return;

    const model = window._v2Model;
    if (!model) return;
    const nodes = model.nodes || [];
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    // Display filter
    const f = window._displayFilter || {};
    const storySet = new Set(f.stories || []);
    const typeSet = new Set();
    (f.types || []).forEach(t => {
        typeSet.add(t);
        if (t === 'beam') { typeSet.add('beam_x'); typeSet.add('beam_y'); }
    });

    // 부재 타입별 force 배열 선택 (OpenSees 로컬 좌표 보정)
    // beam_y: geomTransf 3에서 강축 휨이 T_kNm에, 전단이 Vy_kN에 매핑됨
    function _safeMax(arr) { return arr.length > 0 ? Math.max(...arr.map(Math.abs)) : 0; }

    function _getForceArr(mf, mode, mType) {
        if (mode === 'axial') return mf.N_kN;

        // 모든 성분에서 최대값 비교하여 지배 성분 자동 선택
        const vy = mf.Vy_kN || [], vz = mf.Vz_kN || [];
        const my = mf.My_kNm || [], mz = mf.Mz_kNm || [], t = mf.T_kNm || [];

        if (mode === 'shear') {
            // Vy, Vz 중 큰 것
            return _safeMax(vy) >= _safeMax(vz) ? vy : vz;
        }
        if (mode === 'moment') {
            // My, Mz, T 중 가장 큰 것 (beam_y의 T = 실제 강축 휨)
            const maxMy = _safeMax(my), maxMz = _safeMax(mz), maxT = _safeMax(t);
            if (maxT > maxMy && maxT > maxMz) return t;
            return maxMy >= maxMz ? my : mz;
        }
        return null;
    }

    // 전체 최대값 (스케일링용)
    let globalMax = 0;
    mfCase.forEach(mf => {
        const mType = mf.type || 'beam_x';
        const arr = _getForceArr(mf, _diagramMode, mType);
        if (arr) arr.forEach(v => { globalMax = Math.max(globalMax, Math.abs(v)); });
    });
    if (globalMax < 0.001) return;

    // 모델 크기 기반 스케일 (최대값 = 모델크기의 10%)
    let maxDim = 1;
    nodes.forEach(n => { maxDim = Math.max(maxDim, Math.abs(n.x), Math.abs(n.y), Math.abs(n.z)); });
    const diagramScale = maxDim * 0.10 / globalMax * _diagramScaleMultiplier;

    _diagramGroup = new THREE.Group();
    _diagramGroup.name = 'forceDiagrams';
    _showDiagramValues = document.getElementById('toggle-dgm-values')?.checked ?? true;

    mfCase.forEach(mf => {
        const mType = mf.type || 'beam_x';
        const arr = _getForceArr(mf, _diagramMode, mType);
        if (!arr || arr.length < 2) return;

        const ni = nodeMap[mf.ni], nj = nodeMap[mf.nj];
        if (!ni || !nj) return;

        // Filter check
        if (!typeSet.has(mType) && !(typeSet.has('beam') && (mType==='beam_x'||mType==='beam_y'))) return;
        const mStory = Math.max(ni.story||0, nj.story||0);
        if (storySet.size > 0 && mStory > 0 && !storySet.has(mStory)) return;

        // 부재 방향벡터
        const dx = nj.x - ni.x, dy = nj.y - ni.y, dz = nj.z - ni.z;
        const L = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (L < 0.01) return;

        // Three.js 좌표: (x, z, -y)
        const p0 = new THREE.Vector3(ni.x, ni.z, -ni.y);
        const p1 = new THREE.Vector3(nj.x, nj.z, -nj.y);
        const axisDir = new THREE.Vector3().subVectors(p1, p0).normalize();

        // 법선 방향: 부재 로컬 좌표계 기반
        const isColumn = mType === 'column';
        let normal;
        if (isColumn) {
            // 기둥(수직): 강축 방향 = X 또는 부재의 수평 투영 방향에 수직
            // cross(axisDir, globalZ) → 수평 방향
            const globalUp = new THREE.Vector3(0, 1, 0);
            normal = new THREE.Vector3().crossVectors(axisDir, globalUp).normalize();
            if (normal.lengthSq() < 0.01) {
                // axisDir이 Y와 평행하면 X방향
                normal.set(1, 0, 0);
            }
        } else {
            // 보(수평): 항상 Y-up(중력 반대방향) 돌출
            normal = new THREE.Vector3(0, 1, 0);
            // 축에 직교하는 성분만
            normal.sub(axisDir.clone().multiplyScalar(axisDir.dot(normal))).normalize();
            if (normal.lengthSq() < 0.01) normal.set(0, 1, 0);
        }

        // 분할점에서 폴리곤 생성
        const nPts = arr.length;
        const positions = [];
        const colors = [];

        for (let k = 0; k < nPts; k++) {
            const t = k / (nPts - 1);
            const baseP = new THREE.Vector3().lerpVectors(p0, p1, t);
            const val = arr[k];
            const offset = normal.clone().multiplyScalar(val * diagramScale);
            const topP = baseP.clone().add(offset);

            // 기준선 점 + 돌출 점
            positions.push(baseP.x, baseP.y, baseP.z);
            positions.push(topP.x, topP.y, topP.z);

            const c = val >= 0 ? new THREE.Color(_DGM_POS_COLOR) : new THREE.Color(_DGM_NEG_COLOR);
            colors.push(c.r, c.g, c.b);
            colors.push(c.r, c.g, c.b);
        }

        // 삼각형 strip → indexed triangles
        const indices = [];
        for (let k = 0; k < nPts - 1; k++) {
            const i0 = k * 2, i1 = k * 2 + 1, i2 = (k+1) * 2, i3 = (k+1) * 2 + 1;
            indices.push(i0, i1, i2);
            indices.push(i1, i3, i2);
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geo.setIndex(indices);
        geo.computeVertexNormals();

        const mat = new THREE.MeshBasicMaterial({
            vertexColors: true, transparent: true, opacity: 0.5,
            side: THREE.DoubleSide, depthWrite: false,
        });
        _diagramGroup.add(new THREE.Mesh(geo, mat));

        // 외곽선 (돌출된 상단 라인)
        const outlinePts = [];
        for (let k = 0; k < nPts; k++) {
            const t = k / (nPts - 1);
            const baseP = new THREE.Vector3().lerpVectors(p0, p1, t);
            const val = arr[k];
            outlinePts.push(baseP.clone().add(normal.clone().multiplyScalar(val * diagramScale)));
        }
        const lineGeo = new THREE.BufferGeometry().setFromPoints(outlinePts);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x333333, linewidth: 1 });
        _diagramGroup.add(new THREE.Line(lineGeo, lineMat));

        // 부재별 독립 라벨: i-end, j-end, 중앙 최대값
        if (_showDiagramValues) {
            const labelOffset = 0.15;  // 부재 축 안쪽으로 약간 이동 (비율)

            // i-end 값 (부재 시작점 → 약간 안쪽)
            const iVal = arr[0];
            if (Math.abs(iVal) > globalMax * 0.03) {
                const iT = labelOffset;
                const iBase = new THREE.Vector3().lerpVectors(p0, p1, iT);
                const iLabelPos = iBase.clone().add(normal.clone().multiplyScalar(iVal * diagramScale * 1.2));
                _addDiagramLabel(iLabelPos, iVal.toFixed(1), iVal >= 0 ? _DGM_POS_COLOR : _DGM_NEG_COLOR);
            }

            // j-end 값 (부재 끝점 → 약간 안쪽)
            const jVal = arr[nPts - 1];
            if (Math.abs(jVal) > globalMax * 0.03) {
                const jT = 1 - labelOffset;
                const jBase = new THREE.Vector3().lerpVectors(p0, p1, jT);
                const jLabelPos = jBase.clone().add(normal.clone().multiplyScalar(jVal * diagramScale * 1.2));
                _addDiagramLabel(jLabelPos, jVal.toFixed(1), jVal >= 0 ? _DGM_POS_COLOR : _DGM_NEG_COLOR);
            }

            // 중앙 최대값 (내부 고점 — 양 끝과 다른 위치)
            let maxIdx = -1, maxVal = 0;
            for (let k = 1; k < nPts - 1; k++) {
                if (Math.abs(arr[k]) > Math.abs(maxVal)) { maxVal = arr[k]; maxIdx = k; }
            }
            if (maxIdx >= 0 && Math.abs(maxVal) > globalMax * 0.10) {
                const mT = maxIdx / (nPts - 1);
                // 양 끝과 너무 가까우면 skip
                if (mT > 0.2 && mT < 0.8) {
                    const mBase = new THREE.Vector3().lerpVectors(p0, p1, mT);
                    const mLabelPos = mBase.clone().add(normal.clone().multiplyScalar(maxVal * diagramScale * 1.3));
                    _addDiagramLabel(mLabelPos, maxVal.toFixed(1), maxVal >= 0 ? _DGM_POS_COLOR : _DGM_NEG_COLOR);
                }
            }
        }
    });

    scene.add(_diagramGroup);
}

function _addDiagramLabel(pos, text, color) {
    const scale = 4;  // 고해상도
    const canvas = document.createElement('canvas');
    canvas.width = 192 * scale; canvas.height = 48 * scale;
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);
    // 배경 없음 (투명)
    ctx.clearRect(0, 0, 192, 48);
    // 텍스트 외곽선 (가독성)
    const hex = '#' + color.toString(16).padStart(6, '0');
    ctx.font = 'bold 28px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 4;
    ctx.strokeText(text, 96, 24);
    ctx.fillStyle = hex;
    ctx.fillText(text, 96, 24);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.position.copy(pos);
    sprite.scale.set(1.8, 0.45, 1);
    _diagramGroup.add(sprite);
}

// ─── 개별 부재 Canvas 다이어그램 (Properties 패널) ──────────────────────
function drawMemberDiagrams(memberId) {
    const panel = document.getElementById('prop-member-diagrams');
    if (!panel || !currentResult) return;

    const caseName = _getCurrentCaseName();
    const mfCase = currentResult.member_forces?.[caseName];
    if (!mfCase) { panel.style.display = 'none'; return; }

    const mf = mfCase.find(m => m.member_id === memberId);
    if (!mf) { panel.style.display = 'none'; return; }

    panel.style.display = '';
    document.getElementById('prop-dgm-combo').textContent = caseName;

    // 지배 성분 자동 선택 (beam_y: T가 실제 강축 휨)
    const vy = mf.Vy_kN || [], vz = mf.Vz_kN || [];
    const my = mf.My_kNm || [], mz = mf.Mz_kNm || [], t = mf.T_kNm || [];
    function _sm(a) { return a.length > 0 ? Math.max(...a.map(Math.abs)) : 0; }
    const shearArr = _sm(vy) >= _sm(vz) ? vy : vz;
    const maxMy = _sm(my), maxMz = _sm(mz), maxT = _sm(t);
    const momentArr = (maxT > maxMy && maxT > maxMz) ? t : (maxMy >= maxMz ? my : mz);

    _drawSingleDiagram('canvas-axial', mf.N_kN || [], 'N (kN)', '#e53935', '#1565c0');
    _drawSingleDiagram('canvas-shear', shearArr, 'V (kN)', '#e53935', '#1565c0');
    _drawSingleDiagram('canvas-moment', momentArr, 'M (kN·m)', '#e53935', '#1565c0');
}

function _drawSingleDiagram(canvasId, values, label, posColor, negColor) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || values.length < 2) return;

    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const maxAbs = Math.max(...values.map(Math.abs), 0.001);
    const midY = H / 2;
    const scaleY = (H / 2 - 6) / maxAbs;
    const dx = W / (values.length - 1);

    // 기준선
    ctx.strokeStyle = '#999'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();

    // 채워진 영역
    ctx.beginPath();
    ctx.moveTo(0, midY);
    for (let i = 0; i < values.length; i++) {
        ctx.lineTo(i * dx, midY - values[i] * scaleY);
    }
    ctx.lineTo(W, midY);
    ctx.closePath();

    // 그라데이션 채우기 (양=빨강, 음=파랑)
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);
    if (maxVal > 0 && minVal >= 0) {
        ctx.fillStyle = posColor + '30';
    } else if (maxVal <= 0 && minVal < 0) {
        ctx.fillStyle = negColor + '30';
    } else {
        ctx.fillStyle = '#88888830';
    }
    ctx.fill();

    // 외곽선
    ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
        const x = i * dx, y = midY - values[i] * scaleY;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 라벨
    ctx.fillStyle = '#666'; ctx.font = '9px Arial'; ctx.textAlign = 'left';
    ctx.fillText(label, 2, 10);

    // 최대값 표시
    let maxI = 0;
    values.forEach((v, i) => { if (Math.abs(v) > Math.abs(values[maxI])) maxI = i; });
    const mv = values[maxI];
    ctx.fillStyle = mv >= 0 ? posColor : negColor;
    ctx.font = 'bold 9px Arial'; ctx.textAlign = 'right';
    ctx.fillText(mv.toFixed(1), W - 2, 10);

    // 최대값 점
    const px = maxI * dx, py = midY - mv * scaleY;
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = mv >= 0 ? posColor : negColor; ctx.fill();
}

// Case 변경 시 다이어그램 갱신 hook
function _hookCaseSelectForDiagrams() {
    const sel = document.getElementById('case-selector');
    if (sel) {
        sel.addEventListener('change', () => {
            if (_diagramMode !== 'off') _buildDiagrams();
            // 개별 부재 다이어그램도 갱신
            if (selectedMesh?.userData?.elementData) {
                const mid = selectedMesh.userData.elementData.member_id || selectedMesh.userData.elementData.id;
                if (mid) drawMemberDiagrams(mid);
            }
        });
    }
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _hookCaseSelectForDiagrams);
} else {
    setTimeout(_hookCaseSelectForDiagrams, 600);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5: Project Save / Load / Auto-save
// ═══════════════════════════════════════════════════════════════════════════════
const _AUTOSAVE_KEY = 'v2_editor_autosave';

// ═══════════════════════════════════════════════════════════════════════════════
// Story Drift Chart (Viewer Tab)
// ═══════════════════════════════════════════════════════════════════════════════

function switchViewerTab(tab) {
    document.querySelectorAll('.vtab-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.vtab === tab);
    });
    const driftPanel = document.getElementById('drift-panel');
    if (tab === 'drift') {
        driftPanel.style.display = '';
        _populateDriftCaseSelector();
        onDriftCaseChange();  // 자동 전환 로직 + 차트 빌드
    } else {
        driftPanel.style.display = 'none';
    }
    // Three.js canvas resize
    setTimeout(() => { if (typeof onWindowResize === 'function') onWindowResize(); }, 120);
}

function _populateDriftCaseSelector() {
    const sel = document.getElementById('drift-case-selector');
    if (!sel || !currentResult) return;
    const prevVal = sel.value;
    sel.innerHTML = '<option value="__envelope__">Envelope (Max)</option>';
    (currentResult.case_names || []).concat(currentResult.combo_names || []).forEach(cn => {
        sel.innerHTML += `<option value="${cn}">${cn}</option>`;
    });
    sel.value = prevVal || '__envelope__';
}

function onDriftCaseChange() {
    // 케이스에 따라 허용치 자동 전환
    const caseName = document.getElementById('drift-case-selector')?.value || '__envelope__';
    const limitSel = document.getElementById('drift-limit-selector');
    if (limitSel) {
        const cu = caseName.toUpperCase();
        if (cu.includes('EQ') || cu.includes('SEISMIC') || cu === '__ENVELOPE__') {
            // 내진 → design_check 기준 or 기본 1/50
            const dcAllow = currentResult?.design_check?.drift_check?.allowable;
            const val = dcAllow || 0.020;
            // 가장 가까운 옵션 선택
            const opts = Array.from(limitSel.options);
            const best = opts.reduce((a, b) =>
                Math.abs(parseFloat(a.value) - val) < Math.abs(parseFloat(b.value) - val) ? a : b
            );
            limitSel.value = best.value;
        } else {
            // 비지진 (DL, LL, Wind 등) → 1/200
            limitSel.value = '0.005';
        }
    }
    _buildDriftCharts();
}

function onDriftLimitChange() { _buildDriftCharts(); }

function _buildDriftCharts() {
    if (!currentResult?.case_data) return;
    const caseName = document.getElementById('drift-case-selector')?.value || '__envelope__';

    let driftsX = [], driftsY = [], storyLabels = [];

    if (caseName === '__envelope__') {
        const storyMax = {};
        Object.values(currentResult.case_data).forEach(cd => {
            (cd.story_drifts || []).forEach(sd => {
                const s = sd.story;
                if (!storyMax[s]) storyMax[s] = { dx: 0, dy: 0 };
                storyMax[s].dx = Math.max(storyMax[s].dx, Math.abs(sd.drift_x || 0));
                storyMax[s].dy = Math.max(storyMax[s].dy, Math.abs(sd.drift_y || 0));
            });
        });
        const keys = Object.keys(storyMax).sort((a, b) => Number(a) - Number(b));
        keys.forEach(s => {
            storyLabels.push(s + 'F');
            driftsX.push(storyMax[s].dx);
            driftsY.push(storyMax[s].dy);
        });
    } else {
        const cd = currentResult.case_data[caseName];
        if (!cd?.story_drifts) return;
        const sorted = [...cd.story_drifts].sort((a, b) => a.story - b.story);
        sorted.forEach(sd => {
            storyLabels.push(sd.story + 'F');
            driftsX.push(Math.abs(sd.drift_x || 0));
            driftsY.push(Math.abs(sd.drift_y || 0));
        });
    }

    if (storyLabels.length === 0) return;

    // 허용치: 드롭다운 우선, 없으면 design_check fallback
    const limitSel = document.getElementById('drift-limit-selector');
    const allowable = limitSel ? parseFloat(limitSel.value) : (currentResult.design_check?.drift_check?.allowable || 0.020);

    _drawDriftBarChart('canvas-drift-x', storyLabels, driftsX, allowable, 'X');
    _drawDriftBarChart('canvas-drift-y', storyLabels, driftsY, allowable, 'Y');

    // Info
    const invStr = allowable > 0 ? Math.round(1 / allowable) : '-';
    const limitLabel = limitSel ? limitSel.options[limitSel.selectedIndex]?.text : '';
    const infoEl = document.getElementById('drift-info');
    if (infoEl) {
        infoEl.innerHTML = `<b>KDS 41 17 00</b><br>허용: 1/${invStr} (${allowable}) — ${limitLabel}<br>Case: ${caseName === '__envelope__' ? 'Envelope (모든 케이스 중 Max)' : caseName}`;
    }

    // 수직 프로파일 차트 + 상세 테이블
    const storyHeights = currentResult?.viewer?.stories || [];
    _drawDriftProfileChart('canvas-drift-profile', storyLabels, driftsX, driftsY, allowable, storyHeights);
    _buildDriftTable(storyLabels, driftsX, driftsY, allowable, storyHeights);
}

function _drawDriftBarChart(canvasId, labels, values, limit, direction) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || values.length === 0) return;

    const dpr = 2;
    const W_css = canvas.clientWidth || 380;
    const H_css = Math.max(values.length * 36 + 50, 130);
    canvas.style.height = H_css + 'px';
    canvas.width = W_css * dpr;
    canvas.height = H_css * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const marginL = 32, marginR = 70, marginT = 22, marginB = 18;
    const plotW = W_css - marginL - marginR;
    const plotH = H_css - marginT - marginB;
    const gap = plotH / values.length;
    const barH = Math.min(gap * 0.6, 22);

    const maxVal = Math.max(...values, limit) * 1.3;

    // 배경 (어두운 테마 대응: 반투명 어두운 배경)
    ctx.clearRect(0, 0, W_css, H_css);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W_css, H_css);

    // 제목 (밝은 흰색)
    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText('Drift ' + direction, 4, 14);

    // X축 눈금 (밝은 그리드)
    const nTicks = 4;
    ctx.font = '8px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i <= nTicks; i++) {
        const v = (maxVal / nTicks) * i;
        const x = marginL + (v / maxVal) * plotW;
        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(x, marginT); ctx.lineTo(x, marginT + plotH); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        const tickLabel = v > 0.001 ? '1/' + Math.round(1/v) : '0';
        ctx.fillText(tickLabel, x, H_css - 4);
    }

    // 수평 바 (limit 보다 먼저)
    values.forEach((v, i) => {
        const y = marginT + i * gap + (gap - barH) / 2;
        const barW = Math.max((v / maxVal) * plotW, 2);
        const isNG = v > limit;

        // 바 배경 트랙
        ctx.fillStyle = 'rgba(255,255,255,0.05)';
        ctx.fillRect(marginL, y, plotW, barH);

        // 바
        ctx.fillStyle = isNG ? '#ef5350' : '#4caf50';
        ctx.fillRect(marginL, y, barW, barH);

        // 바 외곽
        ctx.strokeStyle = isNG ? '#ff8a80' : '#81c784'; ctx.lineWidth = 0.5;
        ctx.strokeRect(marginL, y, barW, barH);

        // 층 라벨 (왼쪽, 흰색)
        ctx.font = 'bold 10px Arial';
        ctx.fillStyle = '#fff'; ctx.textAlign = 'right';
        ctx.fillText(labels[i], marginL - 5, y + barH / 2 + 4);

        // 값 라벨 (바 오른쪽) — 1/N 형식 위주, 짧게
        ctx.font = 'bold 9px Arial';
        ctx.fillStyle = isNG ? '#ff8a80' : 'rgba(255,255,255,0.8)';
        ctx.textAlign = 'left';
        const inv = v > 1e-8 ? '1/' + Math.round(1 / v) : '0';
        ctx.fillText(inv, marginL + barW + 4, y + barH / 2 + 4);
    });

    // KDS 허용치 라인 — 바 위에 그려서 가리지 않게
    const limitX = marginL + (limit / maxVal) * plotW;
    ctx.strokeStyle = '#ffab00'; ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath(); ctx.moveTo(limitX, marginT - 2); ctx.lineTo(limitX, marginT + plotH + 2); ctx.stroke();
    ctx.setLineDash([]);
    // limit 라벨 (배경 박스 + 텍스트)
    const limitLabel = 'Limit 1/' + Math.round(1 / limit);
    ctx.font = 'bold 9px Arial';
    const lw = ctx.measureText(limitLabel).width;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(limitX + 3, marginT, lw + 6, 14);
    ctx.fillStyle = '#ffab00'; ctx.textAlign = 'left';
    ctx.fillText(limitLabel, limitX + 6, marginT + 10);
}

// ─── Drift Profile Chart (수직 높이 방향, X/Y 겹침) ─────────────────────
function _drawDriftProfileChart(canvasId, labels, driftsX, driftsY, limit, storyHeights) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || labels.length === 0) return;

    const nStories = labels.length;
    // 층 높이에서 누적 높이(elevation) 계산
    const elevations = [0]; // base
    for (let i = 0; i < nStories; i++) {
        const h = (storyHeights[i] || 3.5);
        elevations.push(elevations[elevations.length - 1] + h);
    }
    const totalH = elevations[elevations.length - 1];

    const dpr = 2;
    const W_css = canvas.clientWidth || 380;
    const H_css = Math.max(nStories * 40 + 80, 200);
    canvas.style.height = H_css + 'px';
    canvas.width = W_css * dpr;
    canvas.height = H_css * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const marginL = 42, marginR = 20, marginT = 30, marginB = 28;
    const plotW = W_css - marginL - marginR;
    const plotH = H_css - marginT - marginB;

    const maxDrift = Math.max(...driftsX, ...driftsY, limit) * 1.4;

    // 배경
    ctx.clearRect(0, 0, W_css, H_css);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W_css, H_css);

    // 제목
    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText('Drift Profile (Height vs Drift Ratio)', 4, 14);

    // 범례
    ctx.font = '9px Arial';
    const legX = W_css - 110;
    ctx.fillStyle = '#42a5f5'; ctx.fillRect(legX, 4, 12, 3);
    ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.textAlign = 'left';
    ctx.fillText('Drift X', legX + 16, 10);
    ctx.fillStyle = '#ef5350'; ctx.fillRect(legX + 55, 4, 12, 3);
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillText('Drift Y', legX + 71, 10);

    // 좌표 변환 헬퍼
    function xPos(drift) { return marginL + (drift / maxDrift) * plotW; }
    function yPos(elev) { return marginT + plotH - (elev / totalH) * plotH; }

    // 수평 그리드 (층별)
    ctx.font = '9px Arial';
    ctx.textAlign = 'right';
    for (let i = 0; i <= nStories; i++) {
        const y = yPos(elevations[i]);
        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(marginL, y); ctx.lineTo(marginL + plotW, y); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        const lbl = i === 0 ? 'Base' : labels[i - 1];
        ctx.fillText(lbl, marginL - 4, y + 3);
    }

    // 수직 그리드 (drift ticks)
    const nTicks = 5;
    ctx.font = '8px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i <= nTicks; i++) {
        const v = (maxDrift / nTicks) * i;
        const x = xPos(v);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(x, marginT); ctx.lineTo(x, marginT + plotH); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        if (v > 0.0005) ctx.fillText((v * 100).toFixed(1) + '%', x, H_css - 6);
        else ctx.fillText('0', x, H_css - 6);
    }

    // X축 라벨
    ctx.font = '8px Arial';
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.textAlign = 'center';
    ctx.fillText('Drift Ratio', marginL + plotW / 2, H_css - 0);

    // 허용치 수직선
    const limitXpos = xPos(limit);
    ctx.strokeStyle = '#ffab00'; ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath(); ctx.moveTo(limitXpos, marginT); ctx.lineTo(limitXpos, marginT + plotH); ctx.stroke();
    ctx.setLineDash([]);
    // limit 라벨
    ctx.font = 'bold 8px Arial';
    ctx.fillStyle = '#ffab00'; ctx.textAlign = 'left';
    ctx.fillText('1/' + Math.round(1 / limit), limitXpos + 2, marginT + plotH + 10);

    // Drift X 프로파일 (파란선)
    _drawProfileLine(ctx, driftsX, elevations, '#42a5f5', xPos, yPos);
    // Drift Y 프로파일 (빨간선)
    _drawProfileLine(ctx, driftsY, elevations, '#ef5350', xPos, yPos);
}

function _drawProfileLine(ctx, drifts, elevations, color, xPos, yPos) {
    const n = drifts.length;
    // 꺾은선: base(0) → 1F(drift[0]) → 2F(drift[1]) → ...
    const pts = [{ x: xPos(0), y: yPos(0) }]; // base: drift=0
    for (let i = 0; i < n; i++) {
        pts.push({ x: xPos(drifts[i]), y: yPos(elevations[i + 1]) });
    }

    // 영역 채우기 (반투명)
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(0));
    pts.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(xPos(0), pts[pts.length - 1].y);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // 선
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.stroke();

    // 점 + 값 라벨
    pts.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.5; ctx.stroke();

        if (i > 0) {
            const v = drifts[i - 1];
            const inv = v > 1e-8 ? '1/' + Math.round(1 / v) : '0';
            ctx.font = 'bold 8px Arial';
            ctx.fillStyle = color; ctx.textAlign = 'left';
            ctx.fillText(inv, p.x + 5, p.y + 3);
        }
    });
}

// ─── Drift Detail Table ──────────────────────────────────────────────────
function _buildDriftTable(labels, driftsX, driftsY, allowable, storyHeights) {
    const wrap = document.getElementById('drift-table-wrap');
    if (!wrap) return;

    const n = labels.length;
    if (n === 0) { wrap.innerHTML = ''; return; }

    // 누적 높이
    const elevations = [0];
    for (let i = 0; i < n; i++) elevations.push(elevations[i] + (storyHeights[i] || 3.5));

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const borderC = isDark ? '#444' : '#ddd';
    const bgHead = isDark ? '#2a2a3e' : '#f0f4f8';
    const bgOK = isDark ? 'rgba(76,175,80,0.12)' : 'rgba(76,175,80,0.08)';
    const bgNG = isDark ? 'rgba(239,83,80,0.15)' : 'rgba(239,83,80,0.08)';
    const textC = isDark ? '#e0e0e0' : '#333';

    let html = `<table style="width:100%; border-collapse:collapse; font-size:10px; color:${textC};">`;
    html += `<thead><tr style="background:${bgHead};">`;
    html += `<th style="border:1px solid ${borderC}; padding:4px 6px; text-align:center;">층</th>`;
    html += `<th style="border:1px solid ${borderC}; padding:4px 6px; text-align:center;">높이(m)</th>`;
    html += `<th style="border:1px solid ${borderC}; padding:4px 6px; text-align:center;">EL(m)</th>`;
    html += `<th style="border:1px solid ${borderC}; padding:4px 6px; text-align:center;">Drift X</th>`;
    html += `<th style="border:1px solid ${borderC}; padding:4px 6px; text-align:center;">Drift Y</th>`;
    html += `<th style="border:1px solid ${borderC}; padding:4px 6px; text-align:center;">판정</th>`;
    html += `</tr></thead><tbody>`;

    // 위에서 아래로 (최상층 먼저)
    for (let i = n - 1; i >= 0; i--) {
        const dx = driftsX[i] || 0;
        const dy = driftsY[i] || 0;
        const maxD = Math.max(dx, dy);
        const isNG = maxD > allowable;
        const bg = isNG ? bgNG : bgOK;
        const h = storyHeights[i] || 3.5;
        const el = elevations[i + 1];

        const fmtDrift = (v) => {
            if (v < 1e-8) return '-';
            const inv = Math.round(1 / v);
            return `<span title="${(v * 100).toFixed(3)}%">1/${inv}</span> <span style="opacity:0.5">(${(v * 100).toFixed(2)}%)</span>`;
        };

        html += `<tr style="background:${bg};">`;
        html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:center; font-weight:600;">${labels[i]}</td>`;
        html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:center;">${h.toFixed(1)}</td>`;
        html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:center;">${el.toFixed(1)}</td>`;
        html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:right;">${fmtDrift(dx)}</td>`;
        html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:right;">${fmtDrift(dy)}</td>`;
        html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:center; font-weight:700; color:${isNG ? '#ef5350' : '#4caf50'};">${isNG ? 'NG' : 'OK'}</td>`;
        html += `</tr>`;
    }

    // Max 행
    const maxDX = Math.max(...driftsX);
    const maxDY = Math.max(...driftsY);
    const maxAll = Math.max(maxDX, maxDY);
    html += `<tr style="background:${bgHead}; font-weight:700;">`;
    html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:center;" colspan="3">Max</td>`;
    html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:right; color:${maxDX > allowable ? '#ef5350' : '#4caf50'};">1/${Math.round(1 / maxDX)}</td>`;
    html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:right; color:${maxDY > allowable ? '#ef5350' : '#4caf50'};">1/${Math.round(1 / maxDY)}</td>`;
    html += `<td style="border:1px solid ${borderC}; padding:3px 6px; text-align:center; color:${maxAll > allowable ? '#ef5350' : '#4caf50'};">${maxAll > allowable ? 'NG' : 'OK'}</td>`;
    html += `</tr>`;

    html += '</tbody></table>';
    wrap.innerHTML = html;
}

function _setExportBtnEnabled(on) {
    const b = document.getElementById('btn-export-excel');
    if (!b) return;
    b.disabled = !on;
    b.style.opacity = on ? '1' : '0.5';
}

async function exportToExcel() {
    if (!currentJobId) { alert('해석을 먼저 실행해주세요.'); return; }
    setStatus('Excel 생성 중...', 'running');
    try {
        const resp = await fetch(`/api/export/excel/${currentJobId}`);
        if (!resp.ok) {
            const msg = await resp.text();
            throw new Error(msg || ('HTTP ' + resp.status));
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `analysis_${currentJobId}.xlsx`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        setStatus('Excel 다운로드 완료', 'success');
    } catch (e) {
        console.error('Export failed:', e);
        setStatus('Export 실패: ' + e.message, 'error');
        alert('Export 실패: ' + e.message);
    }
}

async function exportToDXF() {
    if (!window._v2Model) { alert('모델이 없습니다.'); return; }

    // 파일명 입력
    const ts = new Date();
    const dateStr = `${ts.getFullYear()}${String(ts.getMonth()+1).padStart(2,'0')}${String(ts.getDate()).padStart(2,'0')}`;
    const defaultName = `structural_plan_${dateStr}`;
    let filename = prompt('파일명 (확장자 제외):', defaultName);
    if (filename === null) return;  // 취소
    filename = filename.trim();
    if (!filename) filename = defaultName;
    // 확장자 중복 제거
    if (filename.toLowerCase().endsWith('.dxf')) {
        filename = filename.slice(0, -4);
    }
    // 파일명 안전화 (경로/특수문자 제거)
    filename = filename.replace(/[\\/:*?"<>|]/g, '_');

    setStatus('DXF 생성 중...', 'running');
    try {
        const resp = await fetch('/api/export/dxf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: window._v2Model }),
        });
        if (!resp.ok) {
            const msg = await resp.text();
            throw new Error(msg || ('HTTP ' + resp.status));
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename + '.dxf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        setStatus(`DXF 다운로드 완료: ${filename}.dxf`, 'success');
    } catch (e) {
        console.error('DXF Export failed:', e);
        setStatus('DXF Export 실패: ' + e.message, 'error');
    }
}

function saveProject() {
    const model = window._v2Model;
    if (!model) { alert('모델이 없습니다.'); return; }

    const project = {
        version: 3,
        timestamp: new Date().toISOString(),
        model: model,
        config: _gatherConfig(),
    };

    // 해석 결과가 있으면 포함
    if (currentResult) {
        project.analysis = {
            envelope: currentResult.envelope || null,
            case_names: currentResult.case_names || [],
            combo_names: currentResult.combo_names || [],
            case_data: currentResult.case_data || {},
            member_forces: currentResult.member_forces || {},
            member_info: currentResult.member_info || currentResult.member_info_raw || [],
            modal_analysis: currentResult.modal_analysis || null,
            design_check: currentResult.design_check || null,
            interpretation: currentResult.interpretation || null,
            member_checks: currentResult.member_checks || {},
            viewer: currentResult.viewer || null,
            load_cases_raw: currentResult.load_cases_raw || {},
        };
    }

    const json = JSON.stringify(project);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'project_' + new Date().toISOString().slice(0,10) + '.v2proj';
    a.click();
    URL.revokeObjectURL(url);

    const sizeMB = (json.length / 1024 / 1024).toFixed(1);
    setStatus('Project saved (' + sizeMB + ' MB' + (project.analysis ? ', with results' : '') + ')', 'success');
}

function loadProject(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const project = JSON.parse(e.target.result);
            if (!project.model || !project.model.nodes) {
                alert('유효하지 않은 프로젝트 파일입니다.');
                return;
            }

            // 1. 모델 복원
            window._v2Model = project.model;
            if (project.config) _applyConfig(project.config);

            // 2. 3D 뷰어 갱신
            if (typeof refreshEditPreview === 'function') refreshEditPreview();

            // 3. 해석 결과 복원 (있으면)
            if (project.analysis) {
                const a = project.analysis;
                // currentResult 구성 (V1 형식)
                currentResult = {
                    job_id: 'loaded_' + Date.now(),
                    status: 'success',
                    envelope: a.envelope,
                    case_names: a.case_names || [],
                    combo_names: a.combo_names || [],
                    case_data: a.case_data || {},
                    member_forces: a.member_forces || {},
                    member_info: a.member_info || [],
                    member_info_raw: a.member_info || [],
                    modal_analysis: a.modal_analysis,
                    design_check: a.design_check,
                    interpretation: a.interpretation,
                    member_checks: a.member_checks || {},
                    viewer: a.viewer || _buildViewerFromModel(project.model),
                    load_cases_raw: a.load_cases_raw || {},
                    report_url: null,
                };

                modelSource = 'Loaded';

                // 3D 씬 빌드 + 결과 패널 표시
                buildScene(currentResult);
                updateResultsPanel(currentResult);
                updateBottomBar(currentResult);

                setStatus('Project loaded with results: ' + file.name, 'success');
                console.log('[Project] Loaded with analysis:', project.model.nodes.length, 'nodes,',
                    project.model.elements.length, 'elems,',
                    (a.case_names?.length || 0), 'cases');
            } else {
                _fitCameraFromModel(project.model);
                setStatus('Project loaded (model only): ' + file.name, 'success');
                console.log('[Project] Loaded:', project.model.nodes.length, 'nodes,', project.model.elements.length, 'elems');
            }
        } catch (err) {
            alert('파일 읽기 오류: ' + err.message);
            console.error(err);
        }
    };
    reader.readAsText(file);
    event.target.value = '';
}

function _buildViewerFromModel(model) {
    // 모델에서 viewer 데이터 생성 (분석 결과 없을 때 fallback)
    const nodes = (model.nodes || []).map(n => ({
        id: n.id, x: n.x, y: n.y, z: n.z
    }));
    const elements = (model.elements || []).map(e => ({
        id: e.id, ni: e.node_i, nj: e.node_j,
        type: e.elem_type === 'beam' ? 'beam_x' : e.elem_type,
        section: e.section
    }));
    return { nodes, elements };
}

function _fitCameraFromModel(model) {
    // model.nodes의 bbox로 fitCameraToModel(viewer) 안전 호출
    if (typeof fitCameraToModel !== 'function') return;
    const nodes = (model && model.nodes) || [];
    if (!nodes.length) return;
    const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y), zs = nodes.map(n => n.z);
    fitCameraToModel({
        total_width_x: Math.max(...xs) - Math.min(...xs) || 1,
        total_width_y: Math.max(...ys) - Math.min(...ys) || 1,
        total_height:  Math.max(...zs) - Math.min(...zs) || 1,
    });
}

function _gatherConfig() {
    return {
        region: document.getElementById('input-region')?.value || document.getElementById('ifc-region')?.value || '',
        importance: document.getElementById('input-importance')?.value || document.getElementById('ifc-importance')?.value || 'II',
        column_section: document.getElementById('input-col-section')?.value || '',
        beam_x_section: document.getElementById('input-beamx-section')?.value || '',
        beam_y_section: document.getElementById('input-beamy-section')?.value || '',
        material: document.getElementById('input-material')?.value || 'SS275',
        supports: document.getElementById('input-supports')?.value || 'fixed',
    };
}

function _applyConfig(cfg) {
    if (cfg.region) {
        const r1 = document.getElementById('input-region');
        const r2 = document.getElementById('ifc-region');
        if (r1) r1.value = cfg.region;
        if (r2) r2.value = cfg.region;
    }
    if (cfg.column_section) {
        const s = document.getElementById('input-col-section');
        if (s) setSelectValue(s, cfg.column_section);
    }
    if (cfg.beam_x_section) {
        const s = document.getElementById('input-beamx-section');
        if (s) setSelectValue(s, cfg.beam_x_section);
    }
}

// Auto-save: 해석 완료 시 localStorage에 저장
function _autoSave() {
    if (!window._v2Model) return;
    try {
        const saveData = { model: window._v2Model, config: _gatherConfig(), ts: Date.now() };
        // 해석 결과도 포함 (크기 제한으로 member_forces는 제외)
        if (currentResult) {
            saveData.analysis = {
                envelope: currentResult.envelope,
                case_names: currentResult.case_names,
                combo_names: currentResult.combo_names,
                modal_analysis: currentResult.modal_analysis,
                design_check: currentResult.design_check,
                interpretation: currentResult.interpretation,
                viewer: currentResult.viewer,
            };
        }
        const data = JSON.stringify(saveData);
        localStorage.setItem(_AUTOSAVE_KEY, data);
    } catch (e) { /* quota exceeded 등 무시 */ }
}

// Auto-restore: 페이지 로드 시 복원 제안
function _checkAutoRestore() {
    try {
        const saved = localStorage.getItem(_AUTOSAVE_KEY);
        if (!saved) return;
        const data = JSON.parse(saved);
        if (!data.model?.nodes?.length) return;
        const age = (Date.now() - (data.ts || 0)) / 1000 / 60;
        if (age > 60 * 24) return;  // 24시간 이상 지난 건 무시
        const mins = Math.round(age);
        if (confirm('이전 작업이 발견되었습니다 (' + mins + '분 전, ' + data.model.nodes.length + ' nodes). 복원하시겠습니까?')) {
            window._v2Model = data.model;
            if (data.config) _applyConfig(data.config);
            if (typeof refreshEditPreview === 'function') refreshEditPreview();
            _fitCameraFromModel(data.model);
            setStatus('Auto-saved project restored', 'success');
        }
    } catch (e) { /* 무시 */ }
}
setTimeout(_checkAutoRestore, 1000);


// ═══════════════════════════════════════════════════════════════════════════════
// EditorV2ChatBridge — Phase 0 Step 0-3
// ═══════════════════════════════════════════════════════════════════════════════
//
// Stable surface between the floating chat widget (loaded as a separate
// script) and the editor's internal state. The widget MUST NOT reach into
// the top-level `let` variables here — they are not on `window`, so any
// future rename inside this file would silently break the chat tools.
// Everything the chat orchestrator needs flows through this object.
//
// Phase A reads `getContext()` to attach `ui_context` to each chat turn.
// Phase B uses `openRecommendationTab()` / `openCandidate()` to surface
// existing modals from chat citations. Phase C will hydrate
// `openDiffPreview()` to wire virtual (LLM-proposed) candidates into the
// same diff modal that cached candidates use.
window.EditorV2ChatBridge = {
    version: '0.1.0',

    /** UI context attached to every chat message. Returns the freshest
     *  selection + analysis id + visible tab so the LLM can resolve
     *  pronouns like "이 부재" / "현재 결과" without explicit ids.
     *
     *  Element ids and node ids are kept on separate keys — the 3D viewer
     *  treats both as selectable meshes (``userData.elementData``), but a
     *  chat tool asking "이 부재의 ratio" needs the element-only filter
     *  (matches the existing ``getSelectedElementIds()`` semantics).
     */
    getContext() {
        const set = (typeof selectedMeshSet !== 'undefined' && selectedMeshSet instanceof Set)
            ? [...selectedMeshSet]
            : [];
        const elementIds = [];
        const nodeIds = [];
        for (const m of set) {
            const d = m?.userData?.elementData;
            if (!d || d.id == null) continue;
            if (d.type === 'node') nodeIds.push(d.id);
            else elementIds.push(d.id);
        }

        // Single-click selection: ensure the primary mesh's id is present
        // and surfaces first in its respective bucket, even when the
        // multi-select Set is empty.
        const primary = (typeof selectedMesh !== 'undefined')
            ? selectedMesh?.userData?.elementData
            : null;
        if (primary && primary.id != null) {
            if (primary.type === 'node') {
                if (!nodeIds.includes(primary.id)) nodeIds.unshift(primary.id);
            } else {
                if (!elementIds.includes(primary.id)) elementIds.unshift(primary.id);
            }
        }

        const caseSel = document.getElementById('case-selector');
        const activeRtab = document.querySelector('.rtab-btn.active');

        return {
            analysis_id:
                (window._recState && window._recState.analysisId)
                || (typeof currentJobId !== 'undefined' ? currentJobId : null)
                || null,
            selected_element_ids: elementIds,
            selected_node_ids: nodeIds,
            current_load_case: caseSel?.value || null,
            current_result_tab: activeRtab?.dataset?.rtab || null,
            model_source: typeof modelSource === 'string' ? modelSource : null,
        };
    },

    /** Switch the right-panel result tabs to the recommendations view.
     *  The dataset key is ``recommend`` (see editor_v2.html `<button
     *  data-rtab="recommend">`), not ``rec``. */
    openRecommendationTab() {
        if (typeof switchResultTab === 'function') {
            switchResultTab('recommend');
        }
    },

    /** Open the existing diff modal for a cached candidate_id (Phase B). */
    openCandidate(candidateId) {
        if (typeof openRecDiffModal === 'function') {
            return openRecDiffModal(candidateId);
        }
    },

    /**
     * Phase B — open the rec-diff modal for a chat-driven section change.
     *
     * The chat tool ``propose_section_change`` stages the updated_model
     * server-side and returns only a ``preview_id`` (the chat stream
     * guard rejects ``updated_model`` keys — see streaming.py
     * FORBIDDEN_KEYS). We fetch the full payload over plain HTTP, drop
     * it into ``_recState._pendingApply`` matching the shape that
     * ``/preview-apply`` returns, and reuse the existing modal +
     * applyRecDiff path. No new mutation code in chat — the Apply
     * button still goes through ``applyRecDiff → runAnalysisV2``.
     */
    async openDiffPreview(payload) {
        const previewId = payload?.preview_id;
        if (!previewId) {
            console.warn('EditorV2ChatBridge.openDiffPreview: missing preview_id');
            return;
        }
        const st = window._recState;
        if (!st) {
            console.warn('EditorV2ChatBridge.openDiffPreview: _recState not initialised');
            return;
        }
        // Don't trample an in-flight rec-driven modal session — let the
        // user finish that one first. Chat replays the request if needed.
        if (st.previewLoading || st.applyInFlight) {
            return;
        }

        st.previewLoading = true;
        // Chat-driven previews don't correspond to a cached candidate_id.
        // Null it out so any code path that asserts on it can detect.
        st.selectedCandidateId = null;
        if (typeof _showRecModal === 'function') _showRecModal(true);
        if (typeof _renderRecDiffLoading === 'function') _renderRecDiffLoading();
        if (typeof _setRecModalApplyState === 'function') _setRecModalApplyState('disabled');

        try {
            const resp = await fetch(
                '/api/v2/recommendations/chat-preview/'
                + encodeURIComponent(previewId)
            );
            if (!resp.ok) {
                const raw = await resp.text();
                let detail = raw;
                try { detail = JSON.parse(raw).detail || raw; } catch (_) { /* keep raw */ }
                if (typeof _renderRecDiffError === 'function') {
                    _renderRecDiffError(detail, resp.status);
                }
                return;
            }
            const data = await resp.json();
            // Pin the analysis_id from the staged preview so the
            // Apply flow (and any auto-rollback) refers to the right
            // baseline even if the editor state has drifted.
            if (data.analysis_id) {
                st.analysisId = data.analysis_id;
            }
            st._pendingApply = data;
            if (typeof _renderRecDiffPreview === 'function') {
                _renderRecDiffPreview(data);
            }
            const hasChanges = (data?.diff?.changed_member_count || 0) > 0;
            if (typeof _setRecModalApplyState === 'function') {
                _setRecModalApplyState(hasChanges ? 'idle' : 'disabled');
            }
        } catch (e) {
            if (typeof _renderRecDiffError === 'function') {
                _renderRecDiffError(String(e?.message || e), 0);
            }
        } finally {
            st.previewLoading = false;
        }
    },
};
