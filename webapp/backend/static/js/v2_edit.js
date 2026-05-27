// ═══════════════════════════════════════════════════════════════════════════════
// V2 MODEL EDITING — Append to editor3d_v2.js
// ═══════════════════════════════════════════════════════════════════════════════

let editMode = 'view';
let addElemFirstNode = null;
let undoStack = [];
let redoStack = [];
const MAX_UNDO = 30;

// ═══════════════════════════════════════════════════════════════════════════════
// Element Auto-Split: 중간 노드가 있으면 요소를 자동 분할 (Midas Gen 방식)
// ═══════════════════════════════════════════════════════════════════════════════

function pointToLineDistance(px, py, pz, ax, ay, az, bx, by, bz) {
    // 점 P와 직선 AB 사이의 거리
    var abx = bx-ax, aby = by-ay, abz = bz-az;
    var apx = px-ax, apy = py-ay, apz = pz-az;
    var ab2 = abx*abx + aby*aby + abz*abz;
    if (ab2 < 1e-12) return Math.sqrt(apx*apx + apy*apy + apz*apz);
    var t = (apx*abx + apy*aby + apz*abz) / ab2;
    if (t < 0.001 || t > 0.999) return Infinity;  // 끝점 근처는 무시
    var cx = ax + t*abx, cy = ay + t*aby, cz = az + t*abz;
    var dx = px-cx, dy = py-cy, dz = pz-cz;
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

function distBetween(a, b) {
    return Math.sqrt(Math.pow(a.x-b.x,2) + Math.pow(a.y-b.y,2) + Math.pow(a.z-b.z,2));
}

// ─── Node Merge: 근접 노드 병합 (A-1) ─────────────────────────────────
function mergeNearbyNodes(model, tolerance) {
    if (!model || !model.nodes || model.nodes.length < 2) return 0;

    // 적응형 tolerance: 단면 높이 기반
    if (!tolerance || tolerance <= 0) {
        var maxH = 0;
        (model.elements || []).forEach(function(e) {
            var parts = (e.section || '').replace(/[Hh]-/, '').split('x');
            var h = parseFloat(parts[0]);
            if (!isNaN(h) && h > maxH) maxH = h;
        });
        tolerance = maxH > 0 ? (maxH / 1000 * 0.8) : 0.30;
        // Safety: don't exceed 40% of shortest element
        var minLen = Infinity;
        (model.elements || []).forEach(function(e) {
            var ni = model.nodes.find(function(n){return n.id===e.node_i;});
            var nj = model.nodes.find(function(n){return n.id===e.node_j;});
            if (ni && nj) {
                var L = Math.sqrt(Math.pow(nj.x-ni.x,2)+Math.pow(nj.y-ni.y,2)+Math.pow(nj.z-ni.z,2));
                if (L > 0.01 && L < minLen) minLen = L;
            }
        });
        if (minLen < Infinity) tolerance = Math.min(tolerance, minLen * 0.4);
    }
    console.log('[Merge] tolerance=' + tolerance.toFixed(4) + 'm, maxH=' + maxH + 'mm, minLen=' + (minLen < Infinity ? minLen.toFixed(3) : 'inf'));
    var tolSq = tolerance * tolerance;
    var mergeCount = 0;

    // Build merge map: for each node, find the "canonical" node it should merge into
    // Priority: nodes with support > lower ID
    var nodes = model.nodes.slice().sort(function(a, b) {
        // Support nodes first, then by ID
        var aHas = a.support ? 0 : 1;
        var bHas = b.support ? 0 : 1;
        if (aHas !== bHas) return aHas - bHas;
        return a.id - b.id;
    });

    var mergeMap = {};  // oldId → newId
    var absorbed = {};  // nodes that will be removed

    for (var i = 0; i < nodes.length; i++) {
        if (absorbed[nodes[i].id]) continue;
        for (var j = i + 1; j < nodes.length; j++) {
            if (absorbed[nodes[j].id]) continue;
            var dx = nodes[i].x - nodes[j].x;
            var dy = nodes[i].y - nodes[j].y;
            var dz = nodes[i].z - nodes[j].z;
            if (dx * dx + dy * dy + dz * dz < tolSq) {
                // Merge j into i (i has priority: support or lower ID)
                mergeMap[nodes[j].id] = nodes[i].id;
                absorbed[nodes[j].id] = true;
                mergeCount++;
                // If j has support and i doesn't, copy support
                if (nodes[j].support && !nodes[i].support) {
                    nodes[i].support = nodes[j].support;
                }
                // If j has story and i doesn't, copy story
                if (nodes[j].story != null && nodes[i].story == null) {
                    nodes[i].story = nodes[j].story;
                }
            }
        }
    }

    if (mergeCount === 0) return 0;

    // Update element references
    model.elements.forEach(function(e) {
        if (mergeMap[e.node_i]) e.node_i = mergeMap[e.node_i];
        if (mergeMap[e.node_j]) e.node_j = mergeMap[e.node_j];
    });

    // Remove zero-length elements (node_i === node_j after merge)
    model.elements = model.elements.filter(function(e) { return e.node_i !== e.node_j; });

    // Remove duplicate elements (same node_i, node_j pair)
    var seen = {};
    model.elements = model.elements.filter(function(e) {
        var key = Math.min(e.node_i, e.node_j) + '_' + Math.max(e.node_i, e.node_j);
        if (seen[key]) return false;
        seen[key] = true;
        return true;
    });

    // Remove absorbed nodes
    model.nodes = model.nodes.filter(function(n) { return !absorbed[n.id]; });

    return mergeCount;
}

function splitElementsAtNodes(model, tolerance) {
    // 모든 요소에 대해 직선 위의 중간 노드를 찾아 분할
    if (!model || !model.nodes || !model.elements) return 0;

    // 적응형 tolerance: 단면 높이 기반 (merge와 동일 로직)
    if (!tolerance || tolerance <= 0) {
        var maxH = 0;
        (model.elements || []).forEach(function(e) {
            var parts = (e.section || '').replace(/[Hh]-/, '').split('x');
            var h = parseFloat(parts[0]);
            if (!isNaN(h) && h > maxH) maxH = h;
        });
        tolerance = maxH > 0 ? (maxH / 1000 * 0.8) : 0.30;
        var minLen = Infinity;
        var nodeMap0 = {};
        model.nodes.forEach(function(n) { nodeMap0[n.id] = n; });
        (model.elements || []).forEach(function(e) {
            var ni = nodeMap0[e.node_i], nj = nodeMap0[e.node_j];
            if (ni && nj) {
                var L = distBetween(ni, nj);
                if (L > 0.01 && L < minLen) minLen = L;
            }
        });
        if (minLen < Infinity) tolerance = Math.min(tolerance, minLen * 0.4);
    }
    // 최소 분할 세그먼트 길이: tolerance 이하 세그먼트 방지
    var minSegLen = tolerance * 0.5;

    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });

    var newElements = [];
    var removedIds = {};
    var splitCount = 0;
    var nextId = 1;
    model.elements.forEach(function(e) { if (e.id >= nextId) nextId = e.id + 1; });

    model.elements.forEach(function(elem) {
        var ni = nodeMap[elem.node_i];
        var nj = nodeMap[elem.node_j];
        if (!ni || !nj) return;

        var elemLen = distBetween(ni, nj);
        if (elemLen < 0.01) return;  // 0-길이 요소 무시

        // 이 요소의 직선 위에 있는 중간 노드 찾기
        var midNodes = [];
        model.nodes.forEach(function(n) {
            if (n.id === elem.node_i || n.id === elem.node_j) return;
            var dist = pointToLineDistance(
                n.x, n.y, n.z,
                ni.x, ni.y, ni.z,
                nj.x, nj.y, nj.z
            );
            if (dist < tolerance) {
                var fromI = distBetween(n, ni);
                midNodes.push({ node: n, dist: fromI });
            }
        });

        if (midNodes.length === 0) return;

        // ni에서 가까운 순으로 정렬
        midNodes.sort(function(a, b) { return a.dist - b.dist; });

        // 너무 짧은 세그먼트를 만드는 중간 노드 필터링
        var filtered = [];
        var prev = 0;  // ni에서의 누적 거리
        for (var m = 0; m < midNodes.length; m++) {
            var d = midNodes[m].dist;
            if (d - prev < minSegLen) continue;  // 이전 분할점과 너무 가까움
            if (elemLen - d < minSegLen) continue;  // nj와 너무 가까움
            filtered.push(midNodes[m]);
            prev = d;
        }
        if (filtered.length === 0) return;

        // 기존 요소 삭제 예약
        removedIds[elem.id] = true;
        splitCount++;

        // 분할된 요소 생성: ni → mid1 → mid2 → ... → nj
        var chain = [elem.node_i];
        filtered.forEach(function(fm) { chain.push(fm.node.id); });
        chain.push(elem.node_j);

        for (var k = 0; k < chain.length - 1; k++) {
            newElements.push({
                id: nextId++,
                node_i: chain[k],
                node_j: chain[k + 1],
                elem_type: elem.elem_type,
                section: elem.section,
                material: elem.material,
                release_i: (k === 0) ? elem.release_i : null,
                release_j: (k === chain.length - 2) ? elem.release_j : null,
                beta_angle: elem.beta_angle || 0,
            });
        }
    });

    if (splitCount === 0) return 0;

    // 삭제된 요소 제거 + 새 요소 추가
    model.elements = model.elements.filter(function(e) { return !removedIds[e.id]; });
    model.elements = model.elements.concat(newElements);

    console.log('[Split] tolerance=' + tolerance.toFixed(4) + 'm, ' + splitCount + ' elements → ' + newElements.length + ' new segments');
    return splitCount;
}

// ─── 보-보 교차점 노드 생성 ───────────────────────────────────────
function createBeamIntersectionNodes(model) {
    if (!model || !model.nodes || !model.elements) return 0;
    var tol = 0.01;
    var beams = model.elements.filter(function(e) {
        return e.elem_type === 'beam' || e.elem_type === 'beam_x' || e.elem_type === 'beam_y';
    });
    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });

    var nextId = 1;
    model.nodes.forEach(function(n) { if (n.id >= nextId) nextId = n.id + 1; });

    var created = 0;
    for (var i = 0; i < beams.length; i++) {
        var ei = beams[i];
        var ai = nodeMap[ei.node_i], aj = nodeMap[ei.node_j];
        if (!ai || !aj) continue;
        for (var j = i + 1; j < beams.length; j++) {
            var ej = beams[j];
            var bi = nodeMap[ej.node_i], bj = nodeMap[ej.node_j];
            if (!bi || !bj) continue;
            // 같은 층(Z)에 있는 보만
            var zi = (ai.z + aj.z) / 2, zj = (bi.z + bj.z) / 2;
            if (Math.abs(zi - zj) > tol) continue;
            // 2D 교차 계산
            var pt = _lineLineIntersect2D(ai.x, ai.y, aj.x, aj.y, bi.x, bi.y, bj.x, bj.y);
            if (!pt) continue;
            // 기존 노드와 중복 확인
            var dup = false;
            for (var k = 0; k < model.nodes.length; k++) {
                var n = model.nodes[k];
                if (Math.abs(n.x - pt[0]) < tol && Math.abs(n.y - pt[1]) < tol && Math.abs(n.z - (zi+zj)/2) < tol) {
                    dup = true; break;
                }
            }
            if (dup) continue;
            var story = ai.story != null ? ai.story : (aj.story != null ? aj.story : null);
            model.nodes.push({
                id: nextId++, x: pt[0], y: pt[1], z: Math.round((zi+zj)/2 * 1000)/1000,
                support: null, story: story, mass: null
            });
            nodeMap[model.nodes[model.nodes.length-1].id] = model.nodes[model.nodes.length-1];
            created++;
        }
    }
    return created;
}

function _lineLineIntersect2D(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) {
    var dx1 = ax2-ax1, dy1 = ay2-ay1, dx2 = bx2-bx1, dy2 = by2-by1;
    var denom = dx1*dy2 - dy1*dx2;
    if (Math.abs(denom) < 1e-12) return null;
    var t = ((bx1-ax1)*dy2 - (by1-ay1)*dx2) / denom;
    var u = ((bx1-ax1)*dy1 - (by1-ay1)*dx1) / denom;
    var margin = 0.02;
    if (t < margin || t > 1-margin || u < margin || u > 1-margin) return null;
    return [Math.round((ax1 + t*dx1)*1e6)/1e6, Math.round((ay1 + t*dy1)*1e6)/1e6];
}

// ─── A-2: 연결성 검증 ─────────────────────────────────────────────
function validateConnectivity(model) {
    if (!model || !model.nodes || !model.elements) return [];
    var warnings = [];

    // 노드별 연결 요소 수
    var conn = {};
    model.nodes.forEach(function(n) { conn[n.id] = 0; });
    model.elements.forEach(function(e) {
        if (conn[e.node_i] !== undefined) conn[e.node_i]++;
        if (conn[e.node_j] !== undefined) conn[e.node_j]++;
    });

    // Orphan (0 connections, no support)
    var orphans = model.nodes.filter(function(n) {
        return conn[n.id] === 0 && !n.support;
    }).map(function(n) { return n.id; });
    if (orphans.length > 0) {
        warnings.push({
            type: 'orphan_node', severity: 'warning',
            nodeIds: orphans,
            message: orphans.length + ' orphan nodes (no elements connected): N' + orphans.slice(0, 5).join(',N')
        });
    }

    // Dangling (1 connection, no support) — 자유단 무지지
    var dangling = model.nodes.filter(function(n) {
        return conn[n.id] === 1 && !n.support;
    }).map(function(n) { return n.id; });
    if (dangling.length > 0) {
        warnings.push({
            type: 'dangling_node', severity: 'info',
            nodeIds: dangling,
            message: dangling.length + ' dangling nodes (single element, no support): N' + dangling.slice(0, 5).join(',N')
        });
    }

    // Connected components (Union-Find)
    var parent = {};
    model.nodes.forEach(function(n) { parent[n.id] = n.id; });
    function find(x) {
        while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    model.elements.forEach(function(e) {
        var ra = find(e.node_i), rb = find(e.node_j);
        if (ra !== rb) parent[ra] = rb;
    });
    var comps = {};
    model.nodes.forEach(function(n) {
        var root = find(n.id);
        if (!comps[root]) comps[root] = [];
        comps[root].push(n.id);
    });
    var compList = Object.keys(comps);
    if (compList.length > 1) {
        var sizes = compList.map(function(r) { return comps[r].length; }).sort(function(a,b) { return b-a; });
        warnings.push({
            type: 'disconnected', severity: 'error',
            components: compList.length,
            message: 'Model has ' + compList.length + ' disconnected components: sizes [' + sizes.join(', ') + ']'
        });
    }

    return warnings;
}

const EDIT_HINTS = {
    view: 'View mode — rotate/zoom/click members',
    addNode: 'Add Node — click on 3D plane to place node',
    addElement: 'Add Element — click start node, then end node',
    select: 'Select — click or drag to select members (Ctrl+click to add, Esc to clear)',
    delete: 'Delete — click node or element to remove',
    move: 'Move — drag node or double-click for coordinates',
    release: 'Beam Release — click element to edit 6-DOF releases',
    support: 'Support — click base node to edit boundary conditions',
};

window._editingEnabled = false;
window._currentIFCStep = 0;

function disableEditing() {
    window._editingEnabled = false;
    setEditMode('view');
    var tb = document.getElementById('edit-toolbar');
    if (tb) tb.style.display = 'none';
    removeSnapGrid();
    hideCoordInputPanel();
    closeEditDialog();
}

function showResultSelectionToolbar() {
    // 해석 결과 화면에서 Select/View 전환만 가능한 toolbar 표시
    window._editingEnabled = false;
    var tb = document.getElementById('edit-toolbar');
    if (!tb) return;
    tb.style.display = 'flex';
    // 편집 전용 버튼 숨기기 (copy, addNode, addElement, delete, move, release, support)
    tb.querySelectorAll('.edit-btn[data-mode]').forEach(function(b) {
        var m = b.dataset.mode;
        b.style.display = (m === 'view' || m === 'select') ? '' : 'none';
    });
    tb.querySelectorAll('.tool-btn[data-mode]').forEach(function(b) { b.style.display = 'none'; });
    // addnode-options, edit-hint 등 편집 전용 UI 숨기기
    var addnodeOpts = document.getElementById('addnode-options');
    if (addnodeOpts) addnodeOpts.style.display = 'none';
    var hint = document.getElementById('edit-hint');
    if (hint) hint.textContent = 'Click or drag-select members';
    // "Edit" 복귀 버튼 표시
    var backBtn = document.getElementById('btn-back-to-edit');
    if (backBtn) backBtn.style.display = '';
    // select 모드로 자동 전환
    setEditMode('select');
}

function backToEditMode() {
    // 해석 결과 화면에서 편집 모드로 복귀
    if (typeof editUndo === 'function') editUndo();
}

function showEditToolbar() {
    window._editingEnabled = true;
    const tb = document.getElementById('edit-toolbar');
    if (tb) {
        tb.style.display = 'flex';
        // 결과 모드에서 숨겨진 버튼 복원
        tb.querySelectorAll('.edit-btn[data-mode]').forEach(function(b) { b.style.display = ''; });
        tb.querySelectorAll('.tool-btn[data-mode]').forEach(function(b) { b.style.display = ''; });
        // "Edit" 복귀 버튼 숨기기
        var backBtn = document.getElementById('btn-back-to-edit');
        if (backBtn) backBtn.style.display = 'none';
        populateZLevelSelector();
    }
}

function setEditMode(mode) {
    // copy 모드는 내부적으로 select 동작 + 패널 자동 열기
    const actualMode = mode === 'copy' ? 'select' : mode;
    editMode = actualMode;
    document.querySelectorAll('.edit-btn[data-mode]').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode));
    document.querySelectorAll('.tool-btn[data-mode]').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode));
    const hint = document.getElementById('edit-hint');
    if (hint) hint.textContent = mode === 'copy'
        ? 'Copy — select members, then set offset/mirror in panel'
        : (EDIT_HINTS[actualMode] || '');
    const vc = document.getElementById('viewer-container');
    vc.className = vc.className.replace(/mode-\w+/g, '');
    if (actualMode !== 'view') vc.classList.add('mode-' + actualMode);
    const zSel = document.getElementById('edit-z-level');
    if (zSel) zSel.style.display = actualMode === 'addNode' ? 'inline-block' : 'none';
    // Show/hide selection filter options (select + copy 모드 모두 표시)
    const selOpts = document.getElementById('select-options');
    const showSelectUI = (actualMode === 'select');
    if (selOpts) selOpts.style.display = showSelectUI ? 'inline-flex' : 'none';
    if (showSelectUI) populateStorySelector();
    // Copy 모드: 패널 자동 열기/닫기
    if (mode === 'copy') {
        if (typeof openCopyMirrorPanel === 'function') openCopyMirrorPanel();
    } else {
        if (typeof closeCopyMirrorPanel === 'function') closeCopyMirrorPanel();
    }
    // 카메라 컨트롤: 모드별 마우스 버튼 배정
    if (typeof controls !== 'undefined' && controls) {
        if (mode === 'view') {
            controls.enabled = true;
            controls.enableRotate = true;
            controls.enableZoom = true;
            controls.enablePan = true;
            controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
            controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
            controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
        } else {
            // select / addNode / addElement / delete / move / release / support
            // 좌클릭=편집/선택, 우클릭=회전, 중간=줌
            controls.enabled = true;
            controls.enableZoom = true;
            controls.enablePan = true;
            controls.enableRotate = true;
            controls.mouseButtons.LEFT = -1;  // 좌클릭 비활성 (편집/선택용)
            controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
            controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
        }
    }
    if (mode !== 'addElement') addElemFirstNode = null;
    closeEditDialog();
}

function populateZLevelSelector() {
    const sel = document.getElementById('edit-z-level');
    if (!sel || !window._v2Model) return;
    sel.innerHTML = '';
    (window._v2Model.story_elevations || []).forEach((z, i) => {
        const o = document.createElement('option');
        o.value = z;
        o.textContent = i === 0 ? 'Base (' + z + 'm)' : i + 'F (' + z + 'm)';
        sel.appendChild(o);
    });
    const o = document.createElement('option');
    o.value = 'custom';
    o.textContent = 'Custom Z...';
    sel.appendChild(o);
}

// ─── Undo/Redo ──────────────────────────────────────────────────────────
function pushUndo() {
    if (!window._v2Model) return;
    undoStack.push(JSON.stringify(window._v2Model));
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    redoStack = [];
}
function editUndo() {
    if (!undoStack.length) return;
    redoStack.push(JSON.stringify(window._v2Model));
    window._v2Model = JSON.parse(undoStack.pop());
    _refreshAfterUndoRedo('Undo: 모델 복원됨');
}
function editRedo() {
    if (!redoStack.length) return;
    undoStack.push(JSON.stringify(window._v2Model));
    window._v2Model = JSON.parse(redoStack.pop());
    _refreshAfterUndoRedo('Redo: 모델 복원됨');
}
function _refreshAfterUndoRedo(msg) {
    // 해석 결과 화면에서 undo/redo → 모델이 바뀌었으므로 자동 재해석.
    // (백엔드의 analysis_context_cache는 이전 모델의 design_check/candidates를
    //  들고 있어서, 재해석 없이 챗봇이 답하면 stale answer를 줌. Apply 흐름과
    //  대칭으로 자동 재해석한다.)
    if (!window._editingEnabled && typeof currentResult !== 'undefined' && currentResult) {
        if (typeof setStatus === 'function') {
            setStatus(msg + ' · 자동 재해석 중...', 'running');
        }
        if (typeof runAnalysisV2 === 'function') {
            // 재해석이 in-flight인 동안 채팅이 이전 캐시(post-Apply 상태)로 답하지
            // 않도록 analysis_id를 즉시 무효화. 성공 시 runAnalysisV2 내부에서
            // 새 job_id로 currentResult/currentJobId/_recState가 갱신된다.
            // 실패하면 null로 유지되어 챗봇이 "분석 없음"으로 안전하게 처리.
            if (typeof currentJobId !== 'undefined') currentJobId = null;
            if (typeof currentResult !== 'undefined') currentResult = null;
            if (window._recState) {
                window._recState.analysisId = null;
                window._recState.candidatesById = {};
                window._recState.candidates = [];
            }
            // skipUndo:true — editUndo/editRedo가 이미 스택을 적절히 조정했으므로
            // runAnalysisV2의 기본 pushUndo가 직전 상태를 덮어쓰지 않도록.
            runAnalysisV2({ rethrow: false, skipUndo: true });
        } else {
            // runAnalysisV2 미정의 환경 fallback: 결과를 비우고 편집 모드로
            refreshEditPreview();
            if (typeof setStatus === 'function') setStatus(msg + '. 재해석 필요', 'warning');
        }
    } else {
        refreshEditPreview();
        if (typeof setStatus === 'function') setStatus(msg, 'success');
    }
}
function refreshEditPreview() {
    if (window._v2Model && typeof buildV2PreviewScene === 'function')
        buildV2PreviewScene(window._v2Model, true);
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'z') { e.preventDefault(); editUndo(); }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); editRedo(); }
    if (e.key === 'Escape') {
        // In select mode: first clear selection, second press goes to view
        if (editMode === 'select' && typeof selectedMeshSet !== 'undefined' && selectedMeshSet.size > 0) {
            clearAllSelection();
            return;
        }
        setEditMode('view');
    }
});

// ─── ID generators ──────────────────────────────────────────────────────
function getNextNodeId() {
    var ids = (window._v2Model && window._v2Model.nodes || []).map(function(n){return n.id;});
    return ids.length ? Math.max.apply(null, ids) + 1 : 1;
}
function getNextElemId() {
    var ids = (window._v2Model && window._v2Model.elements || []).map(function(e){return e.id;});
    return ids.length ? Math.max.apply(null, ids) + 1 : 1;
}

// ─── Click dispatcher ───────────────────────────────────────────────────
function handleEditClick(event) {
    if (!window._editingEnabled || !window._v2Model || editMode === 'view') return;
    var rect = renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    if (editMode === 'addNode') handleAddNode(mouse);
    else if (editMode === 'addElement') handleAddElement(mouse);
    else if (editMode === 'delete') handleDelete(mouse);
    else if (editMode === 'release') handleRelease(mouse);
    else if (editMode === 'support') handleSupport(mouse);
    // move는 mousedown/drag로 처리, 더블클릭으로 다이얼로그
}

// ─── Add Node (Grid Snap support) ───────────────────────────────────────
var snapGridHelper = null;
var ghostNode = null;
var ghostNodeMat = new THREE.MeshBasicMaterial({ color: 0xff4081, transparent: true, opacity: 0.5 });
var ghostNodeGeo = new THREE.SphereGeometry(0.2, 8, 8);

function getTargetZ() {
    var zSel = document.getElementById('edit-z-level');
    if (zSel && zSel.value !== 'custom') return parseFloat(zSel.value);
    var v = prompt('Z height (m):', '0');
    return v !== null ? parseFloat(v) : NaN;
}

function getGridSpacing() {
    var sel = document.getElementById('grid-spacing');
    return sel ? parseFloat(sel.value) || 1.0 : 1.0;
}

function isGridSnapOn() {
    var chk = document.getElementById('chk-grid-snap');
    return chk ? chk.checked : true;
}

function snapToGrid(val, spacing) {
    return Math.round(val / spacing) * spacing;
}

function handleAddNode(mouse) {
    var targetZ = getTargetZ();
    if (isNaN(targetZ)) return;

    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -targetZ);
    var pt = new THREE.Vector3();
    if (!rc.ray.intersectPlane(plane, pt)) return;

    var sx = pt.x;
    var sy = -pt.z;

    if (isGridSnapOn()) {
        var spacing = getGridSpacing();
        sx = snapToGrid(sx, spacing);
        sy = snapToGrid(sy, spacing);
    } else {
        sx = Math.round(sx * 10) / 10;
        sy = Math.round(sy * 10) / 10;
    }

    pushUndo();
    var nid = getNextNodeId();
    window._v2Model.nodes.push({
        id: nid, x: sx, y: sy, z: targetZ,
        story: null, support: null, mass: null
    });
    refreshEditPreview();
    updateSnapGrid();
    // 새 노드가 기존 요소 위에 있으면 자동 분할
    var splits = splitElementsAtNodes(window._v2Model);
    setStatus('Node N' + nid + ' added at (' + sx + ', ' + sy + ', ' + targetZ + ')' +
        (splits > 0 ? ' [' + splits + ' elements split]' : ''), 'success');
}

// ─── Grid Helper (Z plane) ──────────────────────────────────────────────
function updateSnapGrid() {
    removeSnapGrid();
    if (editMode !== 'addNode' || !isGridSnapOn()) return;

    var targetZ = getTargetZ();
    if (isNaN(targetZ)) return;
    var spacing = getGridSpacing();

    // Grid size from model bounds
    var nodes = window._v2Model ? window._v2Model.nodes || [] : [];
    var xs = nodes.map(function(n){return n.x;}), ys = nodes.map(function(n){return n.y;});
    var minX = xs.length ? Math.min.apply(null, xs) - 5 : -20;
    var maxX = xs.length ? Math.max.apply(null, xs) + 5 : 20;
    var minY = ys.length ? Math.min.apply(null, ys) - 5 : -20;
    var maxY = ys.length ? Math.max.apply(null, ys) + 5 : 20;

    var size = Math.max(maxX - minX, maxY - minY);
    var divisions = Math.round(size / spacing);

    snapGridHelper = new THREE.GridHelper(size, divisions, 0x4285f4, 0xc0c0c0);
    snapGridHelper.material.transparent = true;
    snapGridHelper.material.opacity = 0.3;
    // Position at Z level (Three.js Y = struct Z)
    snapGridHelper.position.set((minX + maxX) / 2, targetZ, -(minY + maxY) / 2);
    scene.add(snapGridHelper);
}

function removeSnapGrid() {
    if (snapGridHelper) { scene.remove(snapGridHelper); snapGridHelper = null; }
    removeGhostNode();
}

function toggleGridSnap() {
    if (editMode === 'addNode') updateSnapGrid();
}

// ─── Ghost Node + Guide Lines (mousemove preview) ──────────────────────
var guideLineX = null, guideLineY = null, guideLineZ = null;

function makeGuideLine(color, p1, p2) {
    // 매번 새 geometry 생성 (dash 계산 정확도 보장)
    var geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    var mat = new THREE.LineDashedMaterial({
        color: color,
        dashSize: 0.3,
        gapSize: 0.15,
        transparent: true,
        opacity: 0.5,
    });
    var line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    return line;
}

function removeGhostNode() {
    if (ghostNode) { scene.remove(ghostNode); ghostNode = null; }
    if (guideLineX) { scene.remove(guideLineX); guideLineX = null; }
    if (guideLineY) { scene.remove(guideLineY); guideLineY = null; }
    if (guideLineZ) { scene.remove(guideLineZ); guideLineZ = null; }
    var mcDiv = document.getElementById('mouse-coords');
    if (mcDiv) mcDiv.style.display = 'none';
}

function updateGhostNode(event) {
    if (!window._editingEnabled || editMode !== 'addNode' || !window._v2Model) { removeGhostNode(); return; }

    var rect = renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );

    var zSel = document.getElementById('edit-z-level');
    var targetZ = (zSel && zSel.value !== 'custom') ? parseFloat(zSel.value) : 0;
    if (isNaN(targetZ)) return;

    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -targetZ);
    var pt = new THREE.Vector3();
    if (!rc.ray.intersectPlane(plane, pt)) { removeGhostNode(); return; }

    var sx = pt.x, sy = -pt.z;
    if (isGridSnapOn()) {
        var spacing = getGridSpacing();
        sx = snapToGrid(sx, spacing);
        sy = snapToGrid(sy, spacing);
    } else {
        sx = Math.round(sx * 10) / 10;
        sy = Math.round(sy * 10) / 10;
    }

    // Ghost node
    if (!ghostNode) {
        ghostNode = new THREE.Mesh(ghostNodeGeo, ghostNodeMat);
        scene.add(ghostNode);
    }
    ghostNode.position.set(sx, targetZ, -sy);

    // Update coordinate display
    var mcDiv = document.getElementById('mouse-coords');
    if (mcDiv) {
        mcDiv.style.display = 'block';
        document.getElementById('mc-x').textContent = sx.toFixed(1);
        document.getElementById('mc-y').textContent = sy.toFixed(1);
        document.getElementById('mc-z').textContent = targetZ.toFixed(1);
    }

    // Model bounds for guide line length
    var nodes = window._v2Model.nodes || [];
    var allX = nodes.map(function(n){return n.x;}), allY = nodes.map(function(n){return n.y;}), allZ = nodes.map(function(n){return n.z;});
    var ext = 5;
    var xMin = (allX.length ? Math.min.apply(null, allX) : 0) - ext;
    var xMax = (allX.length ? Math.max.apply(null, allX) : 20) + ext;
    var yMin = (allY.length ? Math.min.apply(null, allY) : 0) - ext;
    var yMax = (allY.length ? Math.max.apply(null, allY) : 20) + ext;
    var zMin = (allZ.length ? Math.min.apply(null, allZ) : 0) - ext;
    var zMax = (allZ.length ? Math.max.apply(null, allZ) : 20) + ext;

    // Three.js coords: (structX, structZ, -structY)
    var gx = sx, gy = targetZ, gz = -sy;

    // Remove old guides and recreate (ensures dash computation is correct)
    if (guideLineX) { scene.remove(guideLineX); }
    if (guideLineY) { scene.remove(guideLineY); }
    if (guideLineZ) { scene.remove(guideLineZ); }

    // X guide (red) — along structural X
    guideLineX = makeGuideLine(0xff4444,
        new THREE.Vector3(xMin, gy, gz),
        new THREE.Vector3(xMax, gy, gz));
    scene.add(guideLineX);

    // Y guide (blue) — along structural Y (Three.js -Z)
    guideLineY = makeGuideLine(0x4444ff,
        new THREE.Vector3(gx, gy, -yMax),
        new THREE.Vector3(gx, gy, -yMin));
    scene.add(guideLineY);

    // Z guide (green) — along structural Z (Three.js Y, vertical)
    guideLineZ = makeGuideLine(0x44cc44,
        new THREE.Vector3(gx, zMin, gz),
        new THREE.Vector3(gx, zMax, gz));
    scene.add(guideLineZ);
}

// ─── Coordinate Input Panel ─────────────────────────────────────────────
function showCoordInputPanel() {
    var panel = document.getElementById('coord-input-panel');
    if (!panel) return;
    panel.style.display = 'block';
    // Set Z from current level selector
    var zSel = document.getElementById('edit-z-level');
    if (zSel && zSel.value !== 'custom') {
        document.getElementById('coord-z').value = zSel.value;
    }
    document.getElementById('coord-x').focus();
    document.getElementById('coord-x').select();
}

function hideCoordInputPanel() {
    var panel = document.getElementById('coord-input-panel');
    if (panel) panel.style.display = 'none';
}

function createNodeFromCoords() {
    var x = parseFloat(document.getElementById('coord-x').value);
    var y = parseFloat(document.getElementById('coord-y').value);
    var z = parseFloat(document.getElementById('coord-z').value);
    var support = document.getElementById('coord-support').value || null;
    if (isNaN(x) || isNaN(y) || isNaN(z)) { alert('Invalid coordinates'); return; }

    pushUndo();
    var nid = getNextNodeId();
    window._v2Model.nodes.push({
        id: nid, x: x, y: y, z: z,
        story: null, support: support, mass: null
    });
    // 새 노드가 기존 요소 위에 있으면 자동 분할
    var splits = splitElementsAtNodes(window._v2Model);
    refreshEditPreview();
    updateSnapGrid();
    setStatus('Node N' + nid + ' created at (' + x + ', ' + y + ', ' + z + ')' +
        (splits > 0 ? ' [' + splits + ' elements split]' : '') +
        (support ? ' [' + support + ']' : ''), 'success');

    // Keep panel open for continuous input, clear X/Y
    document.getElementById('coord-x').value = '';
    document.getElementById('coord-y').value = '';
    document.getElementById('coord-x').focus();
}

// ─── Add Element ────────────────────────────────────────────────────────
function handleAddElement(mouse) {
    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var spheres = previewMeshes.filter(function(m) {
        return m.geometry && m.geometry.type === 'SphereGeometry';
    });
    var hits = rc.intersectObjects(spheres);
    if (!hits.length) return;

    var pos = hits[0].object.position;
    var node = findClosestV2Node(pos.x, -pos.z, pos.y);
    if (!node) return;

    if (addElemFirstNode === null) {
        addElemFirstNode = node.id;
        hits[0].object.material.color.setHex(0xff4081);
        hits[0].object.material.opacity = 1.0;
        document.getElementById('edit-hint').textContent =
            'N' + node.id + ' selected — click end node';
    } else {
        if (node.id === addElemFirstNode) return;
        showAddElementDialog(addElemFirstNode, node.id);
    }
}

function showAddElementDialog(ni, nj) {
    closeEditDialog();
    var d = document.createElement('div');
    d.id = 'edit-dialog';
    d.style.cssText = 'left:50%;top:50%;transform:translate(-50%,-50%)';
    d.innerHTML =
        '<h4>Add Element: N' + ni + ' \u2192 N' + nj + '</h4>' +
        '<label>Type</label>' +
        '<select id="dlg-type"><option value="column">Column</option>' +
        '<option value="beam" selected>Beam</option>' +
        '<option value="brace">Brace</option></select>' +
        '<label>Section</label>' +
        '<select id="dlg-section"><option>H-300x300</option>' +
        '<option selected>H-400x200</option><option>H-200x200</option>' +
        '<option>H-250x250</option><option>H-500x200</option></select>' +
        '<label>Material</label>' +
        '<select id="dlg-material"><option selected>SS275</option>' +
        '<option>SM490</option></select>' +
        '<div class="dialog-buttons">' +
        '<button class="btn-cancel" onclick="closeEditDialog();addElemFirstNode=null;">Cancel</button>' +
        '<button class="btn-ok" onclick="confirmAddElement(' + ni + ',' + nj + ')">Create</button>' +
        '</div>';
    document.getElementById('viewer-container').appendChild(d);
    // Populate sections async
    fetch('/api/sections/list').then(function(r){return r.json();}).then(function(data) {
        var sel = document.getElementById('dlg-section');
        if (!sel || !data.sections) return;
        sel.innerHTML = '';
        Object.keys(data.sections).forEach(function(g) {
            var og = document.createElement('optgroup');
            og.label = g;
            data.sections[g].forEach(function(s) {
                var o = document.createElement('option');
                o.value = s; o.textContent = s; og.appendChild(o);
            });
            sel.appendChild(og);
        });
    }).catch(function(){});
}

function confirmAddElement(ni, nj) {
    pushUndo();
    var eid = getNextElemId();
    window._v2Model.elements.push({
        id: eid, node_i: ni, node_j: nj,
        elem_type: document.getElementById('dlg-type').value,
        section: document.getElementById('dlg-section').value,
        material: document.getElementById('dlg-material').value,
        release_i: null, release_j: null, beta_angle: 0
    });
    closeEditDialog();
    addElemFirstNode = null;
    refreshEditPreview();
    setStatus('Element E' + eid + ' added: N' + ni + '\u2192N' + nj, 'success');
}

function closeEditDialog() {
    var d = document.getElementById('edit-dialog');
    if (d) d.remove();
}

// ─── Delete ─────────────────────────────────────────────────────────────
function handleDelete(mouse) {
    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var allObjs = previewMeshes.filter(function(m){return m.geometry;});
    var hits = rc.intersectObjects(allObjs);
    if (!hits.length) return;
    var hitMesh = hits[0].object;
    var pos = hitMesh.position || hits[0].point;

    if (hitMesh.geometry.type === 'SphereGeometry' || hitMesh.geometry.type === 'ConeGeometry') {
        var node = findClosestV2Node(pos.x, -pos.z, pos.y);
        if (!node) return;
        var conn = (window._v2Model.elements || []).filter(function(e) {
            return e.node_i === node.id || e.node_j === node.id;
        });
        if (conn.length > 0 && !confirm('N' + node.id + ': ' + conn.length + ' elements connected. Delete all?')) return;
        pushUndo();
        window._v2Model.elements = window._v2Model.elements.filter(function(e) {
            return e.node_i !== node.id && e.node_j !== node.id;
        });
        window._v2Model.nodes = window._v2Model.nodes.filter(function(n){return n.id !== node.id;});
        refreshEditPreview();
        setStatus('Node N' + node.id + ' deleted', 'success');
    } else {
        var pt = hits[0].point;
        var elem = findClosestV2Element(pt.x, -pt.z, pt.y);
        if (!elem) return;
        if (!confirm('E' + elem.id + ' (' + elem.elem_type + ') delete?')) return;
        pushUndo();
        window._v2Model.elements = window._v2Model.elements.filter(function(e){return e.id !== elem.id;});
        refreshEditPreview();
        setStatus('Element E' + elem.id + ' deleted', 'success');
    }
}

// ─── Move Node (Dialog + Drag) ──────────────────────────────────────────
var dragNode = null;
var isDragging = false;
var lastDragRender = 0;
var DRAG_THROTTLE = 50; // ms

function handleMoveNode(mouse) {
    // Click on node → show move dialog
    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var spheres = previewMeshes.filter(function(m){return m.geometry && m.geometry.type === 'SphereGeometry';});
    var hits = rc.intersectObjects(spheres);
    if (!hits.length) return;
    var pos = hits[0].object.position;
    var node = findClosestV2Node(pos.x, -pos.z, pos.y);
    if (!node) return;
    showMoveDialog(node);
}

function showMoveDialog(node) {
    closeEditDialog();
    var d = document.createElement('div');
    d.id = 'edit-dialog';
    d.style.cssText = 'left:50%;top:50%;transform:translate(-50%,-50%)';
    d.innerHTML =
        '<h4>Move Node N' + node.id + '</h4>' +
        '<div style="display:grid;grid-template-columns:30px 1fr;gap:6px;align-items:center">' +
        '<label style="font-weight:600">X</label><input type="number" id="dlg-move-x" step="0.1" value="' + node.x + '">' +
        '<label style="font-weight:600">Y</label><input type="number" id="dlg-move-y" step="0.1" value="' + node.y + '">' +
        '<label style="font-weight:600">Z</label><input type="number" id="dlg-move-z" step="0.1" value="' + node.z + '">' +
        '</div>' +
        '<div class="dialog-buttons">' +
        '<button class="btn-cancel" onclick="closeEditDialog()">Cancel</button>' +
        '<button class="btn-ok" onclick="confirmMoveNode(' + node.id + ')">Apply</button>' +
        '</div>';
    document.getElementById('viewer-container').appendChild(d);
    document.getElementById('dlg-move-x').focus();
    document.getElementById('dlg-move-x').select();

    // Enter key to apply
    d.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.preventDefault(); confirmMoveNode(node.id); }
        if (e.key === 'Escape') { closeEditDialog(); }
    });
}

function confirmMoveNode(nodeId) {
    var node = window._v2Model.nodes.find(function(n){return n.id === nodeId;});
    if (!node) return;
    var nx = parseFloat(document.getElementById('dlg-move-x').value);
    var ny = parseFloat(document.getElementById('dlg-move-y').value);
    var nz = parseFloat(document.getElementById('dlg-move-z').value);
    if (isNaN(nx) || isNaN(ny) || isNaN(nz)) { alert('Invalid coordinates'); return; }
    pushUndo();
    node.x = nx; node.y = ny; node.z = nz;
    closeEditDialog();
    refreshEditPreview();
    setStatus('Node N' + nodeId + ' moved to (' + nx + ', ' + ny + ', ' + nz + ')', 'success');
}

// ─── Drag Move ──────────────────────────────────────────────────────────
function startDrag(event) {
    if (!window._editingEnabled || editMode !== 'move' || !window._v2Model) return;
    var rect = renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var spheres = previewMeshes.filter(function(m){return m.geometry && m.geometry.type === 'SphereGeometry';});
    var hits = rc.intersectObjects(spheres);
    if (!hits.length) return;
    var pos = hits[0].object.position;
    var node = findClosestV2Node(pos.x, -pos.z, pos.y);
    if (!node) return;

    pushUndo();
    dragNode = node;
    isDragging = true;
    // Highlight
    hits[0].object.material.color.setHex(0xff4081);
    // Disable orbit controls during drag
    if (controls) controls.enabled = false;
}

function duringDrag(event) {
    if (!isDragging || !dragNode) return;
    var now = Date.now();
    if (now - lastDragRender < DRAG_THROTTLE) return;
    lastDragRender = now;

    var rect = renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );

    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    // Move on the same Z plane as the node
    var plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -dragNode.z);
    var pt = new THREE.Vector3();
    if (!rc.ray.intersectPlane(plane, pt)) return;

    var sx = pt.x, sy = -pt.z;
    if (isGridSnapOn()) {
        var spacing = getGridSpacing();
        sx = snapToGrid(sx, spacing);
        sy = snapToGrid(sy, spacing);
    } else {
        sx = Math.round(sx * 10) / 10;
        sy = Math.round(sy * 10) / 10;
    }

    dragNode.x = sx;
    dragNode.y = sy;

    // Update coordinate display
    var mcDiv = document.getElementById('mouse-coords');
    if (mcDiv) {
        mcDiv.style.display = 'block';
        document.getElementById('mc-x').textContent = sx.toFixed(1);
        document.getElementById('mc-y').textContent = sy.toFixed(1);
        document.getElementById('mc-z').textContent = dragNode.z.toFixed(1);
    }

    refreshEditPreview();
}

function endDrag() {
    if (!isDragging) return;
    isDragging = false;
    if (dragNode) {
        setStatus('Node N' + dragNode.id + ' moved to (' + dragNode.x + ', ' + dragNode.y + ', ' + dragNode.z + ')', 'success');
    }
    dragNode = null;
    // Re-enable orbit controls for move mode
    if (controls) {
        controls.enabled = true;
        controls.enableRotate = true;
        controls.mouseButtons.LEFT = -1;
        controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
    }
    var mcDiv = document.getElementById('mouse-coords');
    if (mcDiv) mcDiv.style.display = 'none';
}

// ─── Utility ────────────────────────────────────────────────────────────
function findClosestV2Node(tx, ty, tz) {
    if (!window._v2Model) return null;
    var best = null, bd = Infinity;
    window._v2Model.nodes.forEach(function(n) {
        var d = Math.sqrt(Math.pow(n.x-tx,2)+Math.pow(n.y-ty,2)+Math.pow(n.z-tz,2));
        if (d < bd) { bd = d; best = n; }
    });
    return bd < 1.5 ? best : null;
}

function findClosestV2Element(sx, sy, sz) {
    if (!window._v2Model) return null;
    var nm = {};
    window._v2Model.nodes.forEach(function(n){nm[n.id]=n;});
    var best = null, bd = Infinity;
    window._v2Model.elements.forEach(function(e) {
        var ni = nm[e.node_i], nj = nm[e.node_j];
        if (!ni||!nj) return;
        var cx=(ni.x+nj.x)/2, cy=(ni.y+nj.y)/2, cz=(ni.z+nj.z)/2;
        var d = Math.sqrt(Math.pow(cx-sx,2)+Math.pow(cy-sy,2)+Math.pow(cz-sz,2));
        if (d < bd) { bd = d; best = e; }
    });
    return bd < 3.0 ? best : null;
}

// ─── Node Labels: off / id / full ────────────────────────────────────────
window._labelMode = 'off';  // 'off' | 'id' | 'full'

function setLabelMode(mode) {
    if (window._labelMode === mode) {
        // 같은 버튼 다시 누르면 끄기
        window._labelMode = 'off';
    } else {
        window._labelMode = mode;
    }
    // 버튼 active 상태
    var btnId = document.getElementById('btn-label-id');
    var btnFull = document.getElementById('btn-label-full');
    if (btnId) btnId.classList.toggle('active', window._labelMode === 'id');
    if (btnFull) btnFull.classList.toggle('active', window._labelMode === 'full');
    refreshEditPreview();
}

// ─── setEditMode override: show/hide addNode options + grid ─────────────
(function() {
    var origSetEditMode = setEditMode;
    window.setEditMode = function(mode) {
        origSetEditMode(mode);
        // Show/hide addNode-specific options
        var opts = document.getElementById('addnode-options');
        if (opts) opts.style.display = mode === 'addNode' ? 'inline' : 'none';
        // Grid
        if (mode === 'addNode') updateSnapGrid();
        else removeSnapGrid();
        // Hide coord panel when leaving addNode
        if (mode !== 'addNode') hideCoordInputPanel();
    };
})();

// ─── Hooks ──────────────────────────────────────────────────────────────
(function() {
    // Show toolbar on IFC Step 2
    if (typeof goToIFCStep === 'function') {
        var orig = goToIFCStep;
        window.goToIFCStep = function(step) {
            orig(step);
            window._currentIFCStep = step;
            if (step === 2 && window._v2Model) {
                showEditToolbar();
            } else {
                // Step 1, 3 → 편집 비활성
                disableEditing();
            }
        };
    }
    // Canvas events
    document.addEventListener('DOMContentLoaded', function() {
        var canvas = document.getElementById('three-canvas');
        if (canvas) {
            // Edit click (for addNode, addElement, delete — not move drag)
            canvas.addEventListener('click', function(e) {
                if (isDragging) return;
                if (window._editingEnabled && editMode !== 'view' && window._v2Model) handleEditClick(e);
            });
            // Ghost node on mousemove
            canvas.addEventListener('mousemove', function(e) {
                updateGhostNode(e);
            });
            canvas.addEventListener('mouseleave', function() {
                removeGhostNode();
                if (isDragging) endDrag();
            });
            // Drag: mousedown on node in move mode (must use capture to beat OrbitControls)
            canvas.addEventListener('pointerdown', function(e) {
                if (e.button === 0 && editMode === 'move' && window._v2Model) {
                    startDrag(e);
                    if (isDragging) {
                        e.stopPropagation(); // prevent OrbitControls from grabbing
                    }
                }
            }, true); // capture phase — runs before OrbitControls
            canvas.addEventListener('pointerup', function(e) {
                if (e.button === 0 && isDragging) {
                    endDrag();
                    e.stopPropagation();
                }
            }, true);
            canvas.addEventListener('pointermove', function(e) {
                if (isDragging) {
                    duringDrag(e);
                    e.stopPropagation();
                }
            }, true);
            // Double-click: open move dialog for precise input
            canvas.addEventListener('dblclick', function(e) {
                if (editMode === 'move' && window._v2Model) {
                    var rect = renderer.domElement.getBoundingClientRect();
                    var mouse = new THREE.Vector2(
                        ((e.clientX - rect.left) / rect.width) * 2 - 1,
                        -((e.clientY - rect.top) / rect.height) * 2 + 1
                    );
                    handleMoveNode(mouse);
                }
            });
            // (mouseup replaced by pointerup in capture phase above)
        }
    });
    // Keyboard: N key opens coord panel, Enter creates node from coord panel
    document.addEventListener('keydown', function(e) {
        if (e.key === 'n' || e.key === 'N') {
            if (editMode === 'addNode' && document.activeElement.tagName !== 'INPUT') {
                e.preventDefault();
                showCoordInputPanel();
            }
        }
        if (e.key === 'Enter') {
            var panel = document.getElementById('coord-input-panel');
            if (panel && panel.style.display !== 'none') {
                e.preventDefault();
                createNodeFromCoords();
            }
        }
        if (e.key === 'Escape') {
            hideCoordInputPanel();
        }
    });
})();

// ═══════════════════════════════════════════════════════════════════════════════
// Copy / Mirror / Story Copy
// ═══════════════════════════════════════════════════════════════════════════════

function copySelectedOffset() {
    var model = window._v2Model;
    if (!model || typeof selectedMeshSet === 'undefined') return;
    var ids = Array.from(selectedMeshSet)
        .filter(function(m) { return m.userData?.elementData?.type !== 'node'; })
        .map(function(m) { return m.userData?.elementData?.id; })
        .filter(Boolean);
    if (ids.length === 0) { alert('요소를 먼저 선택하세요.'); return; }

    var dx = parseFloat(prompt('X 오프셋 (m):', '0')) || 0;
    var dy = parseFloat(prompt('Y 오프셋 (m):', '0')) || 0;
    var dz = parseFloat(prompt('Z 오프셋 (m):', '0')) || 0;
    if (dx === 0 && dy === 0 && dz === 0) return;

    if (typeof pushUndo === 'function') pushUndo();

    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });
    var nextNid = 1; model.nodes.forEach(function(n) { if (n.id >= nextNid) nextNid = n.id + 1; });
    var nextEid = 1; model.elements.forEach(function(e) { if (e.id >= nextEid) nextEid = e.id + 1; });

    // 노드 매핑 (원본 → 복사본)
    var copyNodeMap = {};
    var elems = model.elements.filter(function(e) { return ids.indexOf(e.id) >= 0; });

    elems.forEach(function(e) {
        [e.node_i, e.node_j].forEach(function(nid) {
            if (copyNodeMap[nid]) return;
            var orig = nodeMap[nid];
            if (!orig) return;
            // 오프셋 위치에 기존 노드가 있는지 확인
            var existing = model.nodes.find(function(n) {
                return Math.abs(n.x - (orig.x+dx)) < 0.01 &&
                       Math.abs(n.y - (orig.y+dy)) < 0.01 &&
                       Math.abs(n.z - (orig.z+dz)) < 0.01;
            });
            if (existing) {
                copyNodeMap[nid] = existing.id;
            } else {
                var newN = { id: nextNid++, x: orig.x+dx, y: orig.y+dy, z: orig.z+dz,
                             support: orig.support, story: orig.story, mass: null };
                model.nodes.push(newN);
                nodeMap[newN.id] = newN;
                copyNodeMap[nid] = newN.id;
            }
        });
    });

    // 요소 복사
    var copied = 0;
    elems.forEach(function(e) {
        var ni = copyNodeMap[e.node_i], nj = copyNodeMap[e.node_j];
        if (!ni || !nj || ni === nj) return;
        model.elements.push({
            id: nextEid++, node_i: ni, node_j: nj,
            elem_type: e.elem_type, section: e.section, material: e.material,
            release_i: e.release_i, release_j: e.release_j, beta_angle: e.beta_angle || 0,
        });
        copied++;
    });

    if (typeof refreshEditPreview === 'function') refreshEditPreview();
    if (typeof setStatus === 'function') setStatus('Copied ' + copied + ' elements (offset: ' + dx + ',' + dy + ',' + dz + ')', 'success');
}

function mirrorSelected() {
    var model = window._v2Model;
    if (!model || typeof selectedMeshSet === 'undefined') return;
    var ids = Array.from(selectedMeshSet)
        .filter(function(m) { return m.userData?.elementData?.type !== 'node'; })
        .map(function(m) { return m.userData?.elementData?.id; })
        .filter(Boolean);
    if (ids.length === 0) { alert('요소를 먼저 선택하세요.'); return; }

    var axis = prompt('대칭 축 (X 또는 Y):', 'X');
    if (!axis) return;
    axis = axis.toUpperCase();
    var val = parseFloat(prompt(axis + '=' + '? (대칭면 좌표):', '0')) || 0;

    if (typeof pushUndo === 'function') pushUndo();

    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });
    var nextNid = 1; model.nodes.forEach(function(n) { if (n.id >= nextNid) nextNid = n.id + 1; });
    var nextEid = 1; model.elements.forEach(function(e) { if (e.id >= nextEid) nextEid = e.id + 1; });

    var copyNodeMap = {};
    var elems = model.elements.filter(function(e) { return ids.indexOf(e.id) >= 0; });

    elems.forEach(function(e) {
        [e.node_i, e.node_j].forEach(function(nid) {
            if (copyNodeMap[nid]) return;
            var orig = nodeMap[nid];
            if (!orig) return;
            var mx = orig.x, my = orig.y;
            if (axis === 'X') my = 2 * val - orig.y;
            else mx = 2 * val - orig.x;

            var existing = model.nodes.find(function(n) {
                return Math.abs(n.x - mx) < 0.01 && Math.abs(n.y - my) < 0.01 && Math.abs(n.z - orig.z) < 0.01;
            });
            if (existing) {
                copyNodeMap[nid] = existing.id;
            } else {
                var newN = { id: nextNid++, x: Math.round(mx*1e6)/1e6, y: Math.round(my*1e6)/1e6, z: orig.z,
                             support: orig.support, story: orig.story, mass: null };
                model.nodes.push(newN);
                nodeMap[newN.id] = newN;
                copyNodeMap[nid] = newN.id;
            }
        });
    });

    var copied = 0;
    elems.forEach(function(e) {
        var ni = copyNodeMap[e.node_i], nj = copyNodeMap[e.node_j];
        if (!ni || !nj || ni === nj) return;
        model.elements.push({
            id: nextEid++, node_i: ni, node_j: nj,
            elem_type: e.elem_type, section: e.section, material: e.material,
            release_i: e.release_i, release_j: e.release_j, beta_angle: e.beta_angle || 0,
        });
        copied++;
    });

    if (typeof refreshEditPreview === 'function') refreshEditPreview();
    if (typeof setStatus === 'function') setStatus('Mirrored ' + copied + ' elements (' + axis + '=' + val + ')', 'success');
}

function copyStoryPattern() {
    var model = window._v2Model;
    if (!model) return;
    var from = parseInt(prompt('복사 원본 층 (예: 1):', '1'));
    var to = parseInt(prompt('붙여넣기 대상 층 (예: 2):', '2'));
    if (isNaN(from) || isNaN(to) || from === to) return;

    var elevs = model.story_elevations || [];
    if (from >= elevs.length || to >= elevs.length) { alert('층 범위 초과'); return; }

    var dz = elevs[to] - elevs[from];
    if (typeof pushUndo === 'function') pushUndo();

    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });

    // 원본 층의 요소 수집
    var fromElems = model.elements.filter(function(e) {
        var ni = nodeMap[e.node_i], nj = nodeMap[e.node_j];
        if (!ni || !nj) return false;
        return ni.story === from || nj.story === from;
    });

    if (fromElems.length === 0) { alert('원본 층에 요소가 없습니다.'); return; }

    var nextNid = 1; model.nodes.forEach(function(n) { if (n.id >= nextNid) nextNid = n.id + 1; });
    var nextEid = 1; model.elements.forEach(function(e) { if (e.id >= nextEid) nextEid = e.id + 1; });

    var copyNodeMap = {};
    fromElems.forEach(function(e) {
        [e.node_i, e.node_j].forEach(function(nid) {
            if (copyNodeMap[nid]) return;
            var orig = nodeMap[nid];
            if (!orig) return;
            var nz = orig.z + dz;
            var existing = model.nodes.find(function(n) {
                return Math.abs(n.x - orig.x) < 0.01 && Math.abs(n.y - orig.y) < 0.01 && Math.abs(n.z - nz) < 0.01;
            });
            if (existing) {
                copyNodeMap[nid] = existing.id;
            } else {
                var ns = orig.story != null ? orig.story + (to - from) : null;
                var newN = { id: nextNid++, x: orig.x, y: orig.y, z: nz,
                             support: null, story: ns, mass: null };
                model.nodes.push(newN);
                nodeMap[newN.id] = newN;
                copyNodeMap[nid] = newN.id;
            }
        });
    });

    var copied = 0;
    fromElems.forEach(function(e) {
        var ni = copyNodeMap[e.node_i], nj = copyNodeMap[e.node_j];
        if (!ni || !nj || ni === nj) return;
        model.elements.push({
            id: nextEid++, node_i: ni, node_j: nj,
            elem_type: e.elem_type, section: e.section, material: e.material,
            release_i: e.release_i, release_j: e.release_j, beta_angle: e.beta_angle || 0,
        });
        copied++;
    });

    if (typeof refreshEditPreview === 'function') refreshEditPreview();
    if (typeof setStatus === 'function') setStatus('Story ' + from + ' → ' + to + ': ' + copied + ' elements copied', 'success');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Copy/Mirror 플로팅 패널 제어
// ═══════════════════════════════════════════════════════════════════════════════

let _cmCurrentTab = 'offset';

function openCopyMirrorPanel() {
    var panel = document.getElementById('copy-mirror-panel');
    if (!panel) return;
    panel.style.display = '';
    _updateCMSelectionInfo();
    _initCMDrag();
}

function closeCopyMirrorPanel() {
    var panel = document.getElementById('copy-mirror-panel');
    if (panel) panel.style.display = 'none';
}

function switchCMTab(tabName) {
    _cmCurrentTab = tabName;
    document.querySelectorAll('.cm-tab').forEach(function(btn) {
        btn.classList.toggle('active', btn.dataset.cmtab === tabName);
    });
    document.querySelectorAll('.cm-content').forEach(function(div) {
        div.style.display = div.dataset.cmtab === tabName ? '' : 'none';
    });
}

function _updateCMSelectionInfo() {
    var info = document.getElementById('cm-selection-info');
    if (!info || typeof selectedMeshSet === 'undefined') return;
    var n = selectedMeshSet ? selectedMeshSet.size : 0;
    info.textContent = '선택: ' + n + ' elements';
}

function applyCopyMirror() {
    if (_cmCurrentTab === 'offset') {
        var dx = parseFloat(document.getElementById('cm-dx').value) || 0;
        var dy = parseFloat(document.getElementById('cm-dy').value) || 0;
        var dz = parseFloat(document.getElementById('cm-dz').value) || 0;
        if (dx === 0 && dy === 0 && dz === 0) { alert('오프셋을 입력하세요.'); return; }
        _doCopyOffset(dx, dy, dz);
    } else if (_cmCurrentTab === 'mirror') {
        var axis = document.querySelector('input[name="cm-axis"]:checked')?.value || 'X';
        var val = parseFloat(document.getElementById('cm-mirror-val').value) || 0;
        _doMirror(axis, val);
    } else if (_cmCurrentTab === 'story') {
        var from = parseInt(document.getElementById('cm-from-story').value) || 1;
        var to = parseInt(document.getElementById('cm-to-story').value) || 2;
        if (from === to) { alert('원본과 대상 층이 같습니다.'); return; }
        copyStoryPattern_internal(from, to);
    }
    closeCopyMirrorPanel();
}

function _doCopyOffset(dx, dy, dz) {
    var model = window._v2Model;
    if (!model || typeof selectedMeshSet === 'undefined') return;
    // 노드 제외 — 요소만 복사
    var ids = Array.from(selectedMeshSet)
        .filter(function(m) { return m.userData?.elementData?.type !== 'node'; })
        .map(function(m) { return m.userData?.elementData?.id; })
        .filter(Boolean);
    if (ids.length === 0) { alert('요소를 먼저 선택하세요 (노드만으로는 복사할 수 없습니다).'); return; }
    if (typeof pushUndo === 'function') pushUndo();

    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });
    var nextNid = 1; model.nodes.forEach(function(n) { if (n.id >= nextNid) nextNid = n.id + 1; });
    var nextEid = 1; model.elements.forEach(function(e) { if (e.id >= nextEid) nextEid = e.id + 1; });

    var copyNodeMap = {};
    var elems = model.elements.filter(function(e) { return ids.indexOf(e.id) >= 0; });
    elems.forEach(function(e) {
        [e.node_i, e.node_j].forEach(function(nid) {
            if (copyNodeMap[nid]) return;
            var orig = nodeMap[nid]; if (!orig) return;
            var existing = model.nodes.find(function(n) {
                return Math.abs(n.x-(orig.x+dx))<0.01 && Math.abs(n.y-(orig.y+dy))<0.01 && Math.abs(n.z-(orig.z+dz))<0.01;
            });
            if (existing) { copyNodeMap[nid] = existing.id; }
            else {
                var newN = {id:nextNid++,x:orig.x+dx,y:orig.y+dy,z:orig.z+dz,support:orig.support,story:orig.story,mass:null};
                model.nodes.push(newN); nodeMap[newN.id]=newN; copyNodeMap[nid]=newN.id;
            }
        });
    });
    var copied = 0;
    elems.forEach(function(e) {
        var ni=copyNodeMap[e.node_i],nj=copyNodeMap[e.node_j];
        if (!ni||!nj||ni===nj) return;
        model.elements.push({id:nextEid++,node_i:ni,node_j:nj,elem_type:e.elem_type,section:e.section,material:e.material,release_i:e.release_i,release_j:e.release_j,beta_angle:e.beta_angle||0});
        copied++;
    });
    if (typeof refreshEditPreview === 'function') refreshEditPreview();
    if (typeof setStatus === 'function') setStatus('Copied '+copied+' elements (offset:'+dx+','+dy+','+dz+')', 'success');
}

function _doMirror(axis, val) {
    var model = window._v2Model;
    if (!model || typeof selectedMeshSet === 'undefined') return;
    var ids = Array.from(selectedMeshSet)
        .filter(function(m) { return m.userData?.elementData?.type !== 'node'; })
        .map(function(m) { return m.userData?.elementData?.id; })
        .filter(Boolean);
    if (ids.length === 0) { alert('요소를 먼저 선택하세요.'); return; }
    if (typeof pushUndo === 'function') pushUndo();

    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });
    var nextNid = 1; model.nodes.forEach(function(n) { if (n.id >= nextNid) nextNid = n.id + 1; });
    var nextEid = 1; model.elements.forEach(function(e) { if (e.id >= nextEid) nextEid = e.id + 1; });

    var copyNodeMap = {};
    var elems = model.elements.filter(function(e) { return ids.indexOf(e.id) >= 0; });
    elems.forEach(function(e) {
        [e.node_i, e.node_j].forEach(function(nid) {
            if (copyNodeMap[nid]) return;
            var orig = nodeMap[nid]; if (!orig) return;
            var mx=orig.x, my=orig.y;
            if (axis==='X') my=2*val-orig.y; else mx=2*val-orig.x;
            var existing = model.nodes.find(function(n) {
                return Math.abs(n.x-mx)<0.01 && Math.abs(n.y-my)<0.01 && Math.abs(n.z-orig.z)<0.01;
            });
            if (existing) { copyNodeMap[nid]=existing.id; }
            else {
                var newN={id:nextNid++,x:Math.round(mx*1e6)/1e6,y:Math.round(my*1e6)/1e6,z:orig.z,support:orig.support,story:orig.story,mass:null};
                model.nodes.push(newN); nodeMap[newN.id]=newN; copyNodeMap[nid]=newN.id;
            }
        });
    });
    var copied = 0;
    elems.forEach(function(e) {
        var ni=copyNodeMap[e.node_i],nj=copyNodeMap[e.node_j];
        if (!ni||!nj||ni===nj) return;
        model.elements.push({id:nextEid++,node_i:ni,node_j:nj,elem_type:e.elem_type,section:e.section,material:e.material,release_i:e.release_i,release_j:e.release_j,beta_angle:e.beta_angle||0});
        copied++;
    });
    if (typeof refreshEditPreview === 'function') refreshEditPreview();
    if (typeof setStatus === 'function') setStatus('Mirrored '+copied+' elements ('+axis+'='+val+')', 'success');
}

function copyStoryPattern_internal(from, to) {
    var model = window._v2Model;
    if (!model) return;
    var elevs = model.story_elevations || [];
    if (from >= elevs.length || to >= elevs.length) { alert('층 범위 초과'); return; }
    var dz = elevs[to] - elevs[from];
    if (typeof pushUndo === 'function') pushUndo();

    var nodeMap = {};
    model.nodes.forEach(function(n) { nodeMap[n.id] = n; });
    var fromElems = model.elements.filter(function(e) {
        var ni=nodeMap[e.node_i],nj=nodeMap[e.node_j];
        if (!ni||!nj) return false;
        return ni.story===from || nj.story===from;
    });
    if (fromElems.length === 0) { alert('원본 층에 요소가 없습니다.'); return; }

    var nextNid=1; model.nodes.forEach(function(n){if(n.id>=nextNid)nextNid=n.id+1;});
    var nextEid=1; model.elements.forEach(function(e){if(e.id>=nextEid)nextEid=e.id+1;});
    var copyNodeMap = {};
    fromElems.forEach(function(e) {
        [e.node_i,e.node_j].forEach(function(nid) {
            if (copyNodeMap[nid]) return;
            var orig=nodeMap[nid]; if (!orig) return;
            var nz=orig.z+dz;
            var existing=model.nodes.find(function(n){return Math.abs(n.x-orig.x)<0.01&&Math.abs(n.y-orig.y)<0.01&&Math.abs(n.z-nz)<0.01;});
            if (existing) { copyNodeMap[nid]=existing.id; }
            else {
                var ns=orig.story!=null?orig.story+(to-from):null;
                var newN={id:nextNid++,x:orig.x,y:orig.y,z:nz,support:null,story:ns,mass:null};
                model.nodes.push(newN); nodeMap[newN.id]=newN; copyNodeMap[nid]=newN.id;
            }
        });
    });
    var copied=0;
    fromElems.forEach(function(e) {
        var ni=copyNodeMap[e.node_i],nj=copyNodeMap[e.node_j];
        if (!ni||!nj||ni===nj) return;
        model.elements.push({id:nextEid++,node_i:ni,node_j:nj,elem_type:e.elem_type,section:e.section,material:e.material,release_i:e.release_i,release_j:e.release_j,beta_angle:e.beta_angle||0});
        copied++;
    });
    if (typeof refreshEditPreview === 'function') refreshEditPreview();
    if (typeof setStatus === 'function') setStatus('Story '+from+' → '+to+': '+copied+' elements copied', 'success');
}

// ─── 드래그 이동 ──────────────────────────────────────────────────────────
function _initCMDrag() {
    var panel = document.getElementById('copy-mirror-panel');
    var titlebar = document.getElementById('cm-titlebar');
    if (!panel || !titlebar || titlebar._dragInited) return;
    titlebar._dragInited = true;

    var ox = 0, oy = 0, sx = 0, sy = 0;
    titlebar.addEventListener('mousedown', function(e) {
        e.preventDefault();
        sx = e.clientX; sy = e.clientY;
        ox = panel.offsetLeft; oy = panel.offsetTop;
        function onMove(ev) {
            panel.style.left = (ox + ev.clientX - sx) + 'px';
            panel.style.top = (oy + ev.clientY - sy) + 'px';
        }
        function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
}
