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
async function runAnalysisV2() {
    if (!window._v2Model) { alert('V2 모델이 없습니다.'); return; }

    setStatus('V2 해석 중 (KDS Load Gen + Analysis + Design Check)...', 'running');

    // 층별 용도 수집
    const usageRows = document.querySelectorAll('.ifc-usage-sel');
    const storyConfigs = [];
    ifcEditedData.stories.forEach((s, i) => {
        storyConfigs.push({
            story: i + 1,
            usage: (i < usageRows.length) ? usageRows[i].value : 'office',
            slab_thickness: 0.15,
            dead_load_finish: 1.0,
        });
    });

    const config = {
        region: document.getElementById('ifc-region')?.value || '서울',
        importance: document.getElementById('ifc-importance')?.value || 'II',
        site_class: 'S3',
        seismic_system: 'ordinary_moment_frame',
        exposure_category: 'B',
        geometric_nonlinearity: 'linear',
        stories: storyConfigs,
        // Seismic method (ELF / RSA)
        seismic_method: document.getElementById('ifc-seismic-method')?.value || 'ELF',
        rsa_combination: document.getElementById('ifc-rsa-combination')?.value || 'CQC',
        rsa_direction: document.getElementById('ifc-rsa-direction')?.value || '30pct',
    };

    try {
        const resp = await fetch('/api/v2/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: window._v2Model, config }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'V2 해석 실패');
        }

        const result = await resp.json();
        if (result.status !== 'success') throw new Error('V2 해석 실패');

        // V2 결과를 V1 형식으로 변환하여 기존 UI에 표시
        currentJobId = result.job_id;
        modelSource = 'IFC (V2)';

        // V1 buildScene/updateResultsPanel이 기대하는 형식으로 변환
        const v1Result = convertV2ResultToV1(result);
        currentResult = v1Result;

        buildScene(v1Result);
        updateResultsPanel(v1Result);
        updateBottomBar(v1Result);

        // 해석 후 편집 비활성화 + Solid Section 끄기
        if (typeof disableEditing === 'function') disableEditing();
        if (typeof removeSolidMeshes === 'function') removeSolidMeshes();
        window.solidMode = false;
        var chkSolid = document.getElementById('chk-solid-section');
        if (chkSolid) chkSolid.checked = false;
        setStatus('V2 해석 완료 (KDS + Design Check)', 'success');

    } catch (e) {
        alert('V2 해석 오류: ' + e.message);
        setStatus('V2 해석 실패', 'error');
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
            displacements: {},
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
        if (!modelSource) modelSource = 'Manual';
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

    // Draw all nodes (supports highlighted, others small)
    nodes.forEach(n => {
        const isSupport = Math.abs(n.z) < 0.01;
        const geo = new THREE.SphereGeometry(isSupport ? 0.2 : 0.1, 8, 8);
        const mat = new THREE.MeshPhongMaterial({
            color: isSupport ? 0xff6600 : 0x888888,
            transparent: !isSupport,
            opacity: isSupport ? 1.0 : 0.4,
        });
        const sphere = new THREE.Mesh(geo, mat);
        sphere.position.set(n.x, n.z, -n.y);
        sphere.userData.nodeId = n.id;
        scene.add(sphere);
        nodeMeshes.push(sphere);
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
        const rows = [
            ['A',  data.A_cm2,  'cm\u00B2'],
            ['Ix', data.Ix_cm4, 'cm\u2074'],
            ['Iy', data.Iy_cm4, 'cm\u2074'],
            ['J',  data.J_cm4,  'cm\u2074'],
            ['H',  data.h_mm,   'mm'],
            ['B',  data.b_mm,   'mm'],
        ];
        let grid = rows
            .filter(r => r[1] > 0)
            .map(([k, v, u]) => `<span class="sp-key">${k}</span><span class="sp-val">${v.toLocaleString()}</span><span class="sp-unit">${u}</span>`)
            .join('');
        tooltip.innerHTML = `<div class="sp-member-type ${elem.type}">${typeLabel} #${elem.id}</div>`
            + `<div class="sp-title">${data.name}</div>`
            + `<div class="sp-grid">${grid}</div>`;
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
}

function clearAllSelection() {
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

    const elevations = window._v2Model.story_elevations || [];
    const filter = getSelectionFilter();

    // Determine Z ranges for the story
    let zMin, zMax;
    if (storyIdx === 'all') {
        zMin = -Infinity;
        zMax = Infinity;
    } else {
        const idx = parseInt(storyIdx);
        zMin = elevations[idx] - 0.01;
        zMax = (idx + 1 < elevations.length) ? elevations[idx + 1] + 0.01 : Infinity;
    }

    // Select matching meshes
    memberMeshes.forEach(({ mesh }) => {
        if (!meshPassesFilter(mesh)) return;
        const d = mesh.userData.elementData;
        if (!d) return;

        let match = false;
        if (d.type === 'node') {
            match = d.z >= zMin && d.z <= zMax;
        } else {
            // Element: check if either endpoint is in this story range
            const model = window._v2Model;
            const ni = model.nodes.find(n => n.id === d.ni);
            const nj = model.nodes.find(n => n.id === d.nj);
            if (ni && nj) {
                const eZmin = Math.min(ni.z, nj.z);
                const eZmax = Math.max(ni.z, nj.z);
                // Element belongs to story if its lower node is at story base
                match = eZmin >= zMin - 0.01 && eZmin < zMax;
            }
        }
        if (match && !selectedMeshSet.has(mesh)) highlightMesh(mesh);
    });

    updateSelectionCount();
    // Reset dropdown
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

    // Delete elements
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
    refreshEditPreview();
    setStatus(`Deleted ${total} items`, 'success');
}

// ─── Results Panel ────────────────────────────────────────────────────────
function updateResultsPanel(result) {
    const panel = document.getElementById('prop-results');
    panel.style.display = 'block';
    // Hide empty hint when results are shown
    document.getElementById('prop-empty').style.display = 'none';

    // Model source tag
    const srcTag = document.getElementById('model-source-tag');
    if (srcTag && modelSource) {
        const labels = { Manual: 'Manual', NL: 'NL (자연어)', IFC: 'IFC + Supplement' };
        srcTag.textContent = 'Source: ' + (labels[modelSource] || modelSource);
    }

    // Build case selector dropdown
    buildCaseSelector(result);

    const env = result.envelope || {};
    renderResultsTable(env);

    // Modal analysis
    const modalData = result.modal_analysis || null;
    buildModalUI(modalData);

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
        // Restore envelope view
        renderResultsTable(currentResult.envelope || {});
        updateBottomBarValues(currentResult.envelope || {});
        restoreOriginalPositions();
    } else {
        const cd = currentResult.case_data?.[caseName];
        if (!cd) return;
        renderResultsTable(cd.summary, caseName);
        updateBottomBarValues(cd.summary);
        applyDeformedShape(cd.displacements);
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

function applyDeformedShape(displacements) {
    if (!displacements || !currentResult?.viewer) return;
    saveOriginalState();

    const scale = 50; // Exaggeration factor
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
        if (!nid || !deformedPos[nid]) return;
        m.position.copy(deformedPos[nid]);
    });
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
}

// ─── Modal Analysis UI + Animation ───────────────────────────────────────
let modeAnimationId = null;
let modeAnimating = false;

function buildModalUI(modal) {
    const section = document.getElementById('modal-section');
    const sel = document.getElementById('mode-selector');
    const table = document.getElementById('modal-table');
    if (!section || !sel || !modal?.modes?.length) {
        if (section) section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
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
        if (!nid) return;
        if (deformedPos[nid]) m.position.copy(deformedPos[nid]);
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

function applyDesignCheckColors(memberChecks) {
    memberMeshes.forEach(({ mesh, elementData }) => {
        if (selectedMeshSet.has(mesh)) {
            // Update stored orig color so deselection restores DC color
            const mc = memberChecks[String(elementData.id)];
            if (mc) {
                mesh.userData._origColor = mc.status === 'OK'
                    ? (mc.interaction_ratio > 0.7 ? COLORS.dc_marginal : COLORS.dc_ok)
                    : COLORS.dc_ng;
            }
            return; // don't override selection highlight
        }
        const mc = memberChecks[String(elementData.id)];
        if (mc) {
            if (mc.status === 'OK') {
                mesh.material.color.setHex(mc.interaction_ratio > 0.7 ? COLORS.dc_marginal : COLORS.dc_ok);
            } else {
                mesh.material.color.setHex(COLORS.dc_ng);
            }
        }
    });
}

function resetElementColors() {
    memberMeshes.forEach(({ mesh, elementData }) => {
        if (selectedMeshSet.has(mesh)) {
            mesh.userData._origColor = getElementColor(elementData);
            return; // don't override selection highlight
        }
        mesh.material.color.setHex(getElementColor(elementData));
    });
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
