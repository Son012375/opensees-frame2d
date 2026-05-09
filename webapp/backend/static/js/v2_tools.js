// ═══════════════════════════════════════════════════════════════════════════════
// V2 TOOL PALETTE + BEAM RELEASE + SUPPORT CONDITIONS
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Tool Palette: Model Info Update ────────────────────────────────────
function updateToolModelInfo() {
    var el = document.getElementById('tool-model-info');
    if (!el || !window._v2Model) return;
    var m = window._v2Model;
    var nNodes = (m.nodes || []).length;
    var nElems = (m.elements || []).length;
    var nCols = (m.elements || []).filter(function(e){return e.elem_type === 'column';}).length;
    var nBeams = (m.elements || []).filter(function(e){return e.elem_type === 'beam';}).length;
    var nBraces = (m.elements || []).filter(function(e){return e.elem_type === 'brace';}).length;
    var nStories = (m.story_elevations || []).length - 1;
    el.innerHTML =
        '<div class="tool-info-item"><div class="num">' + nNodes + '</div><div class="lbl">Nodes</div></div>' +
        '<div class="tool-info-item"><div class="num">' + nElems + '</div><div class="lbl">Elems</div></div>' +
        '<div class="tool-info-item"><div class="num">' + nCols + '</div><div class="lbl">Cols</div></div>' +
        '<div class="tool-info-item"><div class="num">' + nBeams + '</div><div class="lbl">Beams</div></div>' +
        '<div class="tool-info-item"><div class="num">' + nBraces + '</div><div class="lbl">Braces</div></div>' +
        '<div class="tool-info-item"><div class="num">' + nStories + '</div><div class="lbl">Stories</div></div>';
    // Sync Z-level selector in tool palette
    var zSelTool = document.getElementById('edit-z-level-tool');
    var zSelMain = document.getElementById('edit-z-level');
    if (zSelTool && zSelMain) zSelTool.innerHTML = zSelMain.innerHTML;
}

// Override refreshEditPreview to also update tool info
(function() {
    var _origRefresh = refreshEditPreview;
    refreshEditPreview = function() {
        _origRefresh();
        updateToolModelInfo();
    };
})();


// ═══════════════════════════════════════════════════════════════════════════════
// BEAM RELEASE: 6-DOF Hexagon UI
// ═══════════════════════════════════════════════════════════════════════════════

var DOF_LABELS = ['Vx','Vy','Vz','Rx','Ry','Rz'];

function handleRelease(mouse) {
    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var allObjs = previewMeshes.filter(function(m){return m.geometry;});
    var hits = rc.intersectObjects(allObjs);
    if (!hits.length) return;
    var pt = hits[0].point;
    var elem = findClosestV2Element(pt.x, -pt.z, pt.y);
    if (!elem) return;
    showReleaseDialog(elem);
}

function showReleaseDialog(elem) {
    closeEditDialog();
    var riState = parseReleaseDOFs(elem.release_i);
    var rjState = parseReleaseDOFs(elem.release_j);

    var d = document.createElement('div');
    d.id = 'edit-dialog';
    d.style.cssText = 'left:50%;top:50%;transform:translate(-50%,-50%);min-width:300px';
    d.innerHTML =
        '<h4>Beam Release: E' + elem.id + ' (' + elem.elem_type + ')</h4>' +
        '<p style="font-size:11px;color:#888;margin-bottom:8px">' + elem.section + '</p>' +
        '<div style="display:flex;gap:20px;justify-content:center;margin:12px 0">' +
        '<div style="text-align:center">' +
        '<div style="font-size:11px;font-weight:600;margin-bottom:6px">i-end (N' + elem.node_i + ')</div>' +
        buildHexGrid('ri', riState) +
        '</div>' +
        '<div style="text-align:center">' +
        '<div style="font-size:11px;font-weight:600;margin-bottom:6px">j-end (N' + elem.node_j + ')</div>' +
        buildHexGrid('rj', rjState) +
        '</div>' +
        '</div>' +
        '<div style="font-size:10px;color:#888;text-align:center;margin-bottom:10px">' +
        'Click to toggle: ' +
        '<span style="background:#1a237e;color:#fff;padding:1px 6px;border-radius:2px;font-size:9px">Fixed</span> / ' +
        '<span style="border:1px solid #ccc;padding:1px 6px;border-radius:2px;font-size:9px">Released</span>' +
        '</div>' +
        '<div style="display:flex;gap:8px;justify-content:center;margin-bottom:10px">' +
        '<button style="font-size:10px;padding:3px 8px;border:1px solid #ddd;border-radius:3px;cursor:pointer;background:#f5f5f5" onclick="presetRelease(\'pin_i\')">Pin i-end</button>' +
        '<button style="font-size:10px;padding:3px 8px;border:1px solid #ddd;border-radius:3px;cursor:pointer;background:#f5f5f5" onclick="presetRelease(\'pin_j\')">Pin j-end</button>' +
        '<button style="font-size:10px;padding:3px 8px;border:1px solid #ddd;border-radius:3px;cursor:pointer;background:#f5f5f5" onclick="presetRelease(\'pin_both\')">Pin Both</button>' +
        '<button style="font-size:10px;padding:3px 8px;border:1px solid #ddd;border-radius:3px;cursor:pointer;background:#f5f5f5" onclick="presetRelease(\'fixed\')">All Fixed</button>' +
        '</div>' +
        '<div class="dialog-buttons">' +
        '<button class="btn-cancel" onclick="closeEditDialog()">Cancel</button>' +
        '<button class="btn-ok" onclick="confirmRelease(' + elem.id + ')">Apply</button>' +
        '</div>';
    document.getElementById('viewer-container').appendChild(d);
}

function buildHexGrid(prefix, state) {
    var html = '<div class="release-hex">';
    for (var i = 0; i < 6; i++) {
        var cls = state[i] ? 'fixed' : 'free';
        html += '<div class="release-hex-cell ' + cls + '" ' +
            'id="' + prefix + '_' + i + '" ' +
            'onclick="toggleReleaseCell(\'' + prefix + '\',' + i + ')">' +
            DOF_LABELS[i] + '</div>';
    }
    html += '</div>';
    return html;
}

function toggleReleaseCell(prefix, idx) {
    var cell = document.getElementById(prefix + '_' + idx);
    if (!cell) return;
    cell.classList.toggle('fixed');
    cell.classList.toggle('free');
}

function presetRelease(preset) {
    var allFixed = [true,true,true,true,true,true];
    var pinned = [true,true,true,false,false,false]; // translations fixed, rotations free
    if (preset === 'pin_i') {
        setHexState('ri', pinned);
        setHexState('rj', allFixed);
    } else if (preset === 'pin_j') {
        setHexState('ri', allFixed);
        setHexState('rj', pinned);
    } else if (preset === 'pin_both') {
        setHexState('ri', pinned);
        setHexState('rj', pinned);
    } else if (preset === 'fixed') {
        setHexState('ri', allFixed);
        setHexState('rj', allFixed);
    }
}

function setHexState(prefix, state) {
    for (var i = 0; i < 6; i++) {
        var cell = document.getElementById(prefix + '_' + i);
        if (!cell) continue;
        cell.classList.remove('fixed', 'free');
        cell.classList.add(state[i] ? 'fixed' : 'free');
    }
}

function parseReleaseDOFs(releaseVal) {
    var state = [true, true, true, true, true, true]; // all fixed
    if (!releaseVal) return state;
    if (releaseVal === 'moment_y') { state[4] = false; }
    else if (releaseVal === 'moment_z') { state[5] = false; }
    else if (releaseVal === 'moment_yz') { state[4] = false; state[5] = false; }
    else if (releaseVal === 'all') { state[3] = false; state[4] = false; state[5] = false; }
    return state;
}

function dofStateToReleaseType(state) {
    var allFixed = state.every(function(s){return s;});
    if (allFixed) return null;
    var rxFree = !state[3], ryFree = !state[4], rzFree = !state[5];
    if (rxFree && ryFree && rzFree) return 'all';
    if (ryFree && rzFree) return 'moment_yz';
    if (ryFree) return 'moment_y';
    if (rzFree) return 'moment_z';
    return 'moment_yz';
}

function confirmRelease(elemId) {
    var elem = window._v2Model.elements.find(function(e){return e.id === elemId;});
    if (!elem) return;
    var riState = [], rjState = [];
    for (var i = 0; i < 6; i++) {
        var ci = document.getElementById('ri_' + i);
        riState.push(ci ? ci.classList.contains('fixed') : true);
        var cj = document.getElementById('rj_' + i);
        rjState.push(cj ? cj.classList.contains('fixed') : true);
    }
    pushUndo();
    elem.release_i = dofStateToReleaseType(riState);
    elem.release_j = dofStateToReleaseType(rjState);
    closeEditDialog();
    refreshEditPreview();
    setStatus('E' + elemId + ' release: i=' + (elem.release_i||'fixed') + ', j=' + (elem.release_j||'fixed'), 'success');
}


// ═══════════════════════════════════════════════════════════════════════════════
// SUPPORT CONDITIONS: Node boundary
// ═══════════════════════════════════════════════════════════════════════════════

function handleSupport(mouse) {
    var rc = new THREE.Raycaster();
    rc.setFromCamera(mouse, camera);
    var spheres = previewMeshes.filter(function(m){return m.geometry && m.geometry.type === 'SphereGeometry';});
    var hits = rc.intersectObjects(spheres);
    if (!hits.length) return;
    var pos = hits[0].object.position;
    var node = findClosestV2Node(pos.x, -pos.z, pos.y);
    if (!node) return;
    showSupportDialog(node);
}

function showSupportDialog(node) {
    closeEditDialog();
    var current = node.support || '';
    var d = document.createElement('div');
    d.id = 'edit-dialog';
    d.style.cssText = 'left:50%;top:50%;transform:translate(-50%,-50%);min-width:240px';

    var opts = [
        {v:'', label:'Free (No support)', dof:'all free'},
        {v:'fixed', label:'Fixed (6-DOF)', dof:'Vx Vy Vz Rx Ry Rz = all constrained'},
        {v:'pinned', label:'Pinned (3-DOF)', dof:'Vx Vy Vz constrained, Rx Ry Rz free'},
        {v:'roller_x', label:'Roller X', dof:'Vy Vz constrained, Vx Rx Ry Rz free'},
        {v:'roller_y', label:'Roller Y', dof:'Vx Vz constrained, Vy Rx Ry Rz free'},
    ];
    var optHtml = opts.map(function(o) {
        return '<option value="' + o.v + '"' + (current===o.v?' selected':'') + '>' + o.label + '</option>';
    }).join('');

    d.innerHTML =
        '<h4>Support: N' + node.id + '</h4>' +
        '<p style="font-size:11px;color:#888;margin-bottom:8px">(' + node.x + ', ' + node.y + ', ' + node.z + ')</p>' +
        '<label style="font-size:12px;font-weight:600">Condition</label>' +
        '<select id="dlg-support-type" style="width:100%;padding:6px 8px;margin:4px 0 8px;border:1px solid #ddd;border-radius:4px;font-size:12px">' + optHtml + '</select>' +
        '<div id="support-dof-preview" style="padding:8px;background:#f5f5f5;border-radius:4px;font-family:monospace;font-size:11px;margin-bottom:10px">' +
        getSupportDOFText(current) + '</div>' +
        '<div class="dialog-buttons">' +
        '<button class="btn-cancel" onclick="closeEditDialog()">Cancel</button>' +
        '<button class="btn-ok" onclick="confirmSupport(' + node.id + ')">Apply</button>' +
        '</div>';
    document.getElementById('viewer-container').appendChild(d);
    document.getElementById('dlg-support-type').addEventListener('change', function() {
        document.getElementById('support-dof-preview').innerHTML = getSupportDOFText(this.value);
    });
}

function getSupportDOFText(type) {
    var map = {
        '':        'Vx:○ Vy:○ Vz:○ Rx:○ Ry:○ Rz:○\nAll DOF free',
        'fixed':   'Vx:● Vy:● Vz:● Rx:● Ry:● Rz:●\nAll DOF constrained',
        'pinned':  'Vx:● Vy:● Vz:● Rx:○ Ry:○ Rz:○\nTranslation fixed, Rotation free',
        'roller_x':'Vx:○ Vy:● Vz:● Rx:○ Ry:○ Rz:○\nX-direction free',
        'roller_y':'Vx:● Vy:○ Vz:● Rx:○ Ry:○ Rz:○\nY-direction free',
    };
    return '<pre style="margin:0;white-space:pre-wrap">' + (map[type||''] || map['']) + '</pre>';
}

function confirmSupport(nodeId) {
    var node = window._v2Model.nodes.find(function(n){return n.id === nodeId;});
    if (!node) return;
    var val = document.getElementById('dlg-support-type').value;
    pushUndo();
    node.support = val || null;
    closeEditDialog();
    refreshEditPreview();
    setStatus('N' + nodeId + ' support: ' + (val || 'free'), 'success');
}


// ═══════════════════════════════════════════════════════════════════════════════
// SOLID SECTION 3D RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

// sectionPropsCache는 editor3d_v2.js에서 이미 선언됨 — 재사용
window.solidMeshes = window.solidMeshes || [];
window.solidMode = window.solidMode || false;

function _applySolidMode(enabled) {
    window.solidMode = !!enabled;
    if (window.solidMode) {
        _hideWireMeshes();
        loadSectionPropsAndBuild().then(function() {
            // DC Colors가 활성화되어 있으면 solid에도 적용
            var chkDC = document.getElementById('toggle-dc-colors');
            if (chkDC && chkDC.checked && typeof currentResult !== 'undefined'
                && currentResult && currentResult.member_checks
                && typeof applyDesignCheckColors === 'function') {
                applyDesignCheckColors(currentResult.member_checks);
            }
            // Deformed shape가 활성화되어 있으면 solid에도 적용
            var chkDef = document.getElementById('toggle-deformed');
            if (chkDef && chkDef.checked && typeof _applyDeformedIfEnabled === 'function') {
                _applyDeformedIfEnabled();
            }
        });
    } else {
        removeSolidMeshes();
        _showWireMeshes();
    }
    // 두 체크박스 동기화
    var chkStep2 = document.getElementById('chk-solid-section');
    var chkTop = document.getElementById('chk-solid-section-top');
    if (chkStep2) chkStep2.checked = window.solidMode;
    if (chkTop) chkTop.checked = window.solidMode;
}

function toggleSolidSection() {
    var chk = document.getElementById('chk-solid-section');
    _applySolidMode(chk ? chk.checked : false);
}

function toggleSolidSectionTop() {
    var chk = document.getElementById('chk-solid-section-top');
    _applySolidMode(chk ? chk.checked : false);
}

function _hideWireMeshes() {
    // 해석 후 결과 씬 + preview 씬 모두 숨김
    // memberMeshes/nodeMeshes는 editor3d_v2.js에서 let으로 선언되지만 같은 전역 스코프
    try {
        memberMeshes.forEach(function(item) { if (item.mesh) item.mesh.visible = false; });
    } catch(e) {}
    try {
        nodeMeshes.forEach(function(m) { if (m) m.visible = false; });
    } catch(e) {}
    scene.traverse(function(obj) {
        if (obj.userData && obj.userData._isPreviewElement) obj.visible = false;
    });
}

function _showWireMeshes() {
    try {
        memberMeshes.forEach(function(item) { if (item.mesh) item.mesh.visible = true; });
    } catch(e) {}
    try {
        nodeMeshes.forEach(function(m) { if (m) m.visible = true; });
    } catch(e) {}
    scene.traverse(function(obj) {
        if (obj.userData && obj.userData._isPreviewElement) obj.visible = true;
    });
}

function removeSolidMeshes() {
    (window.solidMeshes || []).forEach(function(m) { scene.remove(m); });
    window.solidMeshes = [];
    window._solidMeshMap = {};
}

async function loadSectionPropsAndBuild() {
    if (!window._v2Model) return;
    // 사용된 단면 목록
    var sections = [];
    window._v2Model.elements.forEach(function(e) {
        if (sections.indexOf(e.section) === -1) sections.push(e.section);
    });

    // 단면 치수를 별도 캐시에 저장 (sectionPropsCache는 API 전체 응답)
    if (!window._sectionDims) window._sectionDims = {};

    for (var i = 0; i < sections.length; i++) {
        var secName = sections[i];
        if (window._sectionDims[secName]) continue;

        // 기존 캐시에서 치수 추출 시도
        var cached = sectionPropsCache[secName];
        if (cached && (cached.h_mm || cached.H || cached.h)) {
            window._sectionDims[secName] = {
                h: cached.h_mm || cached.H || cached.h,
                b: cached.b_mm || cached.B || cached.b,
                tw: cached.tw_mm || cached.tw || 8,
                tf: cached.tf_mm || cached.tf || 12,
            };
            continue;
        }

        // API 조회
        try {
            var resp = await fetch('/api/sections/properties/' + encodeURIComponent(secName));
            var data = await resp.json();
            if (data && (data.h_mm || data.H || data.h)) {
                window._sectionDims[secName] = {
                    h: data.h_mm || data.H || data.h || 300,
                    b: data.b_mm || data.B || data.b || 150,
                    tw: data.tw_mm || data.tw || 8,
                    tf: data.tf_mm || data.tf || 12,
                };
            }
        } catch(e) { /* skip */ }
    }

    console.log('Section dims loaded:', Object.keys(window._sectionDims).length, window._sectionDims);
    buildSolidMeshes();
}

function buildHSectionShape(h, b, tw, tf) {
    // H형강 단면 Shape (중심 (0,0))
    var hw = h / 2, bw = b / 2, twh = tw / 2, tfv = tf;
    var s = new THREE.Shape();
    s.moveTo(-bw, -hw);
    s.lineTo(bw, -hw);
    s.lineTo(bw, -hw + tfv);
    s.lineTo(twh, -hw + tfv);
    s.lineTo(twh, hw - tfv);
    s.lineTo(bw, hw - tfv);
    s.lineTo(bw, hw);
    s.lineTo(-bw, hw);
    s.lineTo(-bw, hw - tfv);
    s.lineTo(-twh, hw - tfv);
    s.lineTo(-twh, -hw + tfv);
    s.lineTo(-bw, -hw + tfv);
    s.lineTo(-bw, -hw);
    return s;
}

function buildSHSShape(b, t) {
    // SHS/RHS 사각 중공 단면 Shape (중심 (0,0))
    var hw = b / 2;
    var outer = new THREE.Shape();
    outer.moveTo(-hw, -hw);
    outer.lineTo(hw, -hw);
    outer.lineTo(hw, hw);
    outer.lineTo(-hw, hw);
    outer.lineTo(-hw, -hw);
    // 내부 중공 (hole)
    var iw = hw - t;
    if (iw > 0) {
        var hole = new THREE.Path();
        hole.moveTo(-iw, -iw);
        hole.lineTo(iw, -iw);
        hole.lineTo(iw, iw);
        hole.lineTo(-iw, iw);
        hole.lineTo(-iw, -iw);
        outer.holes.push(hole);
    }
    return outer;
}

function buildRHSShape(h, b, t) {
    // RHS 직사각 중공 단면 Shape (중심 (0,0))
    var hh = h / 2, bh = b / 2;
    var outer = new THREE.Shape();
    outer.moveTo(-bh, -hh);
    outer.lineTo(bh, -hh);
    outer.lineTo(bh, hh);
    outer.lineTo(-bh, hh);
    outer.lineTo(-bh, -hh);
    var ih = hh - t, ib = bh - t;
    if (ih > 0 && ib > 0) {
        var hole = new THREE.Path();
        hole.moveTo(-ib, -ih);
        hole.lineTo(ib, -ih);
        hole.lineTo(ib, ih);
        hole.lineTo(-ib, ih);
        hole.lineTo(-ib, -ih);
        outer.holes.push(hole);
    }
    return outer;
}

function buildCHSShape(d, t) {
    // CHS 원형 중공 단면 Shape (중심 (0,0))
    var ro = d / 2;
    var outer = new THREE.Shape();
    outer.absarc(0, 0, ro, 0, Math.PI * 2, false);
    var ri = ro - t;
    if (ri > 0) {
        var hole = new THREE.Path();
        hole.absarc(0, 0, ri, 0, Math.PI * 2, true);
        outer.holes.push(hole);
    }
    return outer;
}

function buildLShape(a, b, t) {
    // L형강 (ㄱ) 단면 Shape (중심 ≈ 도심 근사, 좌하단 기준)
    var s = new THREE.Shape();
    // 수직 플랜지 (좌측)
    s.moveTo(0, 0);
    s.lineTo(t, 0);
    s.lineTo(t, a - t);
    s.lineTo(0, a - t);
    s.lineTo(0, 0);
    // 수평 플랜지 (하단)
    var s2 = new THREE.Shape();
    s2.moveTo(0, 0);
    s2.lineTo(b, 0);
    s2.lineTo(b, t);
    s2.lineTo(0, t);
    s2.lineTo(0, 0);
    // 합치기: L자 외곽선을 하나의 Shape로
    var ls = new THREE.Shape();
    ls.moveTo(-b/2, -a/2);
    ls.lineTo(-b/2 + b, -a/2);
    ls.lineTo(-b/2 + b, -a/2 + t);
    ls.lineTo(-b/2 + t, -a/2 + t);
    ls.lineTo(-b/2 + t, -a/2 + a);
    ls.lineTo(-b/2, -a/2 + a);
    ls.lineTo(-b/2, -a/2);
    return ls;
}

function buildChannelShape(h, b, tw, tf) {
    // ㄷ형강 (Channel) 단면 Shape (중심 (0,0))
    var hh = h/2, bh = b;
    var s = new THREE.Shape();
    // 외곽 C자
    s.moveTo(0, -hh);
    s.lineTo(bh, -hh);
    s.lineTo(bh, -hh + tf);
    s.lineTo(tw, -hh + tf);
    s.lineTo(tw, hh - tf);
    s.lineTo(bh, hh - tf);
    s.lineTo(bh, hh);
    s.lineTo(0, hh);
    s.lineTo(0, -hh);
    return s;
}

/** 단면명에서 타입 판별 */
function _detectSectionType(name) {
    if (!name) return 'H';
    if (name.startsWith('\u25a1-') || name.startsWith('□-')) {
        // SHS: □-BxBxt  /  RHS: □-HxBxt (H≠B)
        var m = name.match(/(\d+\.?\d*)[xX](\d+\.?\d*)[xX](\d+\.?\d*)/);
        if (m) {
            var a = parseFloat(m[1]), b = parseFloat(m[2]), t = parseFloat(m[3]);
            if (Math.abs(a - b) < 1) return 'SHS';
            return 'RHS';
        }
        return 'SHS';
    }
    if (name.startsWith('\u25cb-') || name.startsWith('○-')) return 'CHS';
    if (name.startsWith('L-')) return 'L';
    if (name.startsWith('TFC-') || name.startsWith('PFC-')) return 'C';
    if (name.startsWith('T-')) return 'T';
    return 'H';
}

/** 단면명에서 치수 파싱 (SHS/RHS/CHS) */
function _parseSectionDims(name) {
    var m = name.match(/(\d+\.?\d*)[xX](\d+\.?\d*)[xX](\d+\.?\d*)/);
    if (m) return { a: parseFloat(m[1]), b: parseFloat(m[2]), t: parseFloat(m[3]) };
    m = name.match(/(\d+\.?\d*)[xX](\d+\.?\d*)/);
    if (m) return { a: parseFloat(m[1]), b: parseFloat(m[2]), t: null };
    return null;
}

function buildSolidMeshes() {
    removeSolidMeshes();
    if (!window._v2Model || !window.solidMode) return;

    var nodeMap = {};
    window._v2Model.nodes.forEach(function(n) { nodeMap[n.id] = n; });

    var TYPE_COLORS = {column: 0x4285f4, beam: 0x34a853, brace: 0xfbbc04};

    var built = 0, skipped = 0;
    window._v2Model.elements.forEach(function(elem) {
        var ni = nodeMap[elem.node_i];
        var nj = nodeMap[elem.node_j];
        if (!ni || !nj) { skipped++; return; }

        var sType = _detectSectionType(elem.section);
        var props = (window._sectionDims || {})[elem.section];
        var shape;

        if (sType === 'SHS') {
            var dims = _parseSectionDims(elem.section);
            var b_m = ((props && props.b) || (dims && dims.a) || 150) / 1000;
            var t_m = ((props && props.t_mm) || (dims && dims.t) || b_m * 50) / 1000;
            shape = buildSHSShape(b_m, t_m);
        } else if (sType === 'RHS') {
            var dims = _parseSectionDims(elem.section);
            var h_m = ((props && props.h) || (dims && dims.a) || 200) / 1000;
            var b_m = ((props && props.b) || (dims && dims.b) || 100) / 1000;
            var t_m = ((props && props.t_mm) || (dims && dims.t) || 6) / 1000;
            shape = buildRHSShape(h_m, b_m, t_m);
        } else if (sType === 'CHS') {
            var dims = _parseSectionDims(elem.section);
            var d_m = ((props && props.h) || (dims && dims.a) || 200) / 1000;
            var t_m = ((props && props.t_mm) || (dims && dims.b) || 6) / 1000;
            shape = buildCHSShape(d_m, t_m);
        } else if (sType === 'L') {
            var dims = _parseSectionDims(elem.section);
            var a_m = ((props && props.h) || (dims && dims.a) || 100) / 1000;
            var b_m = ((props && props.b) || (dims && dims.b) || a_m * 1000) / 1000;
            var t_m = ((props && props.tw) || (dims && dims.t) || 10) / 1000;
            shape = buildLShape(a_m, b_m, t_m);
        } else if (sType === 'C') {
            var h_m = ((props && props.h) || 200) / 1000;
            var b_m = ((props && props.b) || 80) / 1000;
            var tw_m = ((props && props.tw) || 7) / 1000;
            var tf_m = ((props && props.tf) || 11) / 1000;
            shape = buildChannelShape(h_m, b_m, tw_m, tf_m);
        } else {
            // H형강 / I형강 / 기타
            if (!props || !props.h || !props.b) {
                var match = elem.section.match(/H-?(\d+)[xX×](\d+)/);
                if (match) {
                    props = { h: parseInt(match[1]), b: parseInt(match[2]), tw: 8, tf: 12 };
                } else {
                    skipped++; return;
                }
            }
            var h_m = props.h / 1000;
            var b_m = props.b / 1000;
            var tw_m = (props.tw || props.b * 0.05) / 1000;
            var tf_m = (props.tf || props.h * 0.05) / 1000;
            shape = buildHSectionShape(h_m, b_m, tw_m, tf_m);
        }

        // 부재 길이
        var dx = nj.x - ni.x, dy = nj.y - ni.y, dz = nj.z - ni.z;
        var length = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (length < 0.01) return;

        // Extrude
        var extrudeSettings = { depth: length, bevelEnabled: false };
        var geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);

        var color = TYPE_COLORS[elem.elem_type] || 0x999999;
        var material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: false,
            shininess: 60,
            side: THREE.DoubleSide,
        });
        // 엣지 윤곽선 추가 (형상 명확화)
        var edgeMat = new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.3 });

        var mesh = new THREE.Mesh(geometry, material);

        // 위치/회전: 시작점에 놓고 부재 방향으로 회전
        // Three.js 좌표: (x, z, -y)
        var startPos = new THREE.Vector3(ni.x, ni.z, -ni.y);
        var endPos = new THREE.Vector3(nj.x, nj.z, -nj.y);
        var dir = new THREE.Vector3().subVectors(endPos, startPos).normalize();

        // ExtrudeGeometry는 Z 방향으로 돌출됨 → 부재 방향으로 회전
        var defaultDir = new THREE.Vector3(0, 0, 1);
        var quat = new THREE.Quaternion().setFromUnitVectors(defaultDir, dir);

        mesh.position.copy(startPos);
        mesh.setRotationFromQuaternion(quat);
        mesh.userData._solidElementId = elem.id;
        mesh.userData._solidOrigColor = color;
        mesh.userData._solidOrigLen = length;

        scene.add(mesh);
        window.solidMeshes.push(mesh);
        window._solidMeshMap = window._solidMeshMap || {};
        window._solidMeshMap[elem.id] = mesh;
        // 엣지 윤곽선 — mesh의 자식으로 추가 (mesh 변형 시 함께 이동)
        var edges = new THREE.EdgesGeometry(geometry, 15);
        var edgeLine = new THREE.LineSegments(edges, edgeMat);
        mesh.add(edgeLine);  // 자식으로 추가 (position/rotation 상속)
        window.solidMeshes.push(edgeLine);
        built++;
    });
    console.log('Solid meshes: built=' + built + ', skipped=' + skipped + ', total in scene=' + solidMeshes.length);
}

// Solid meshes도 편집 시 갱신
(function() {
    var _origRefresh2 = refreshEditPreview;
    refreshEditPreview = function() {
        _origRefresh2();
        if (window.solidMode) buildSolidMeshes();
    };
})();


// ─── Init: update tool palette when step 2 shown ────────────────────────
(function() {
    var _prevShowToolbar = showEditToolbar;
    showEditToolbar = function() {
        _prevShowToolbar();
        updateToolModelInfo();
    };
})();
