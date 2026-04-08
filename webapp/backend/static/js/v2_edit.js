// ═══════════════════════════════════════════════════════════════════════════════
// V2 MODEL EDITING — Append to editor3d_v2.js
// ═══════════════════════════════════════════════════════════════════════════════

let editMode = 'view';
let addElemFirstNode = null;
let undoStack = [];
let redoStack = [];
const MAX_UNDO = 30;

const EDIT_HINTS = {
    view: 'View mode — rotate/zoom/click members',
    addNode: 'Add Node — click on 3D plane to place node',
    addElement: 'Add Element — click start node, then end node',
    delete: 'Delete — click node or element to remove',
    move: 'Move — click node, enter new coordinates',
};

function showEditToolbar() {
    const tb = document.getElementById('edit-toolbar');
    if (tb) { tb.style.display = 'flex'; populateZLevelSelector(); }
}

function setEditMode(mode) {
    editMode = mode;
    document.querySelectorAll('.edit-btn[data-mode]').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode));
    const hint = document.getElementById('edit-hint');
    if (hint) hint.textContent = EDIT_HINTS[mode] || '';
    const vc = document.getElementById('viewer-container');
    vc.className = vc.className.replace(/mode-\w+/g, '');
    if (mode !== 'view') vc.classList.add('mode-' + mode);
    const zSel = document.getElementById('edit-z-level');
    if (zSel) zSel.style.display = mode === 'addNode' ? 'inline-block' : 'none';
    // 카메라 컨트롤: 편집 모드에서도 줌/패닝 허용, 좌클릭 회전만 비활성화
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
            controls.enabled = true;
            controls.enableZoom = true;
            controls.enablePan = true;
            controls.enableRotate = true;
            // 좌클릭=편집, 우클릭=회전, 중간=줌
            controls.mouseButtons.LEFT = -1;  // 좌클릭 비활성 (편집용)
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
    refreshEditPreview();
}
function editRedo() {
    if (!redoStack.length) return;
    undoStack.push(JSON.stringify(window._v2Model));
    window._v2Model = JSON.parse(redoStack.pop());
    refreshEditPreview();
}
function refreshEditPreview() {
    if (window._v2Model && typeof buildV2PreviewScene === 'function')
        buildV2PreviewScene(window._v2Model);
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'z') { e.preventDefault(); editUndo(); }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); editRedo(); }
    if (e.key === 'Escape') setEditMode('view');
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
    if (!window._v2Model || editMode === 'view') return;
    var rect = renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    if (editMode === 'addNode') handleAddNode(mouse);
    else if (editMode === 'addElement') handleAddElement(mouse);
    else if (editMode === 'delete') handleDelete(mouse);
    else if (editMode === 'move') handleMoveNode(mouse);
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
    setStatus('Node N' + nid + ' added at (' + sx + ', ' + sy + ', ' + targetZ + ')', 'success');
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
    if (editMode !== 'addNode' || !window._v2Model) { removeGhostNode(); return; }

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
    refreshEditPreview();
    updateSnapGrid();
    setStatus('Node N' + nid + ' created at (' + x + ', ' + y + ', ' + z + ')' +
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

// ─── Move Node ──────────────────────────────────────────────────────────
function handleMoveNode(mouse) {
    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var spheres = previewMeshes.filter(function(m){return m.geometry && m.geometry.type === 'SphereGeometry';});
    var hits = rc.intersectObjects(spheres);
    if (!hits.length) return;
    var pos = hits[0].object.position;
    var node = findClosestV2Node(pos.x, -pos.z, pos.y);
    if (!node) return;
    var nx = prompt('N' + node.id + ' X (current: ' + node.x + '):', node.x);
    if (nx === null) return;
    var ny = prompt('N' + node.id + ' Y (current: ' + node.y + '):', node.y);
    if (ny === null) return;
    var nz = prompt('N' + node.id + ' Z (current: ' + node.z + '):', node.z);
    if (nz === null) return;
    pushUndo();
    node.x = parseFloat(nx); node.y = parseFloat(ny); node.z = parseFloat(nz);
    refreshEditPreview();
    setStatus('Node N' + node.id + ' moved', 'success');
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
            if (step === 2 && window._v2Model) showEditToolbar();
            else {
                var tb = document.getElementById('edit-toolbar');
                if (tb) tb.style.display = 'none';
                removeSnapGrid();
            }
        };
    }
    // Canvas events
    document.addEventListener('DOMContentLoaded', function() {
        var canvas = document.getElementById('three-canvas');
        if (canvas) {
            // Edit click
            canvas.addEventListener('click', function(e) {
                if (editMode !== 'view' && window._v2Model) handleEditClick(e);
            });
            // Ghost node on mousemove
            canvas.addEventListener('mousemove', function(e) {
                updateGhostNode(e);
            });
            canvas.addEventListener('mouseleave', function() {
                removeGhostNode();
            });
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
