/* ============================================================================
 * figma_explorer.js — Model Explorer 라이브 트리 + Analysis Control 실데이터
 *                     + Result Tables 팝업 (Figma UI Lab, wiring map Tier 1-1/1-3/1-5/1-6)
 * ----------------------------------------------------------------------------
 * 데이터 소스 (정적 목업 → 실데이터):
 *   - 해석 전: window._v2Model (IFC 파싱/DRAW/프로젝트 로드 직후부터 존재)
 *   - 해석 후: currentResult (case/combo, member_checks, modal, drifts...)
 * 실시간 연동: 모든 모델 변경 경로가 buildV2PreviewScene(프리뷰/편집/IFC) 또는
 * buildScene(해석 후)을 부르므로 두 함수를 래핑(figma_plan과 동일 패턴, 체이닝)해
 * 디바운스 리빌드한다. 엔진 JS 무수정.
 *
 * 트리 클릭 → 선택 동선은 3D 클릭과 같은 경로를 태운다:
 *   단일 부재 = highlightMesh + showMemberProperties(figma_plan 랩이 플랜/편집패널 동기화)
 *   그룹(층/타입/단면/재료) = clearAllSelection 후 일괄 highlight → 리본 M/S '선택 부재 적용'과 연동
 * ========================================================================== */
(function () {
    'use strict';

    /* ---- engine guards ------------------------------------------------------ */
    function getModel() { try { return window._v2Model || null; } catch (_) { return null; } }
    function getResult() { try { return currentResult || null; } catch (_) { return null; } }
    function status(msg, kind) { try { if (typeof setStatus === 'function') setStatus(msg, kind || 'success'); } catch (_) {} }
    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }
    var TYPE_LABEL = { column: 'Columns', beam_x: 'Beams X', beam_y: 'Beams Y', beam: 'Beams', brace: 'Braces' };
    var LEAF_CAP = 200;   // 대형 모델에서 leaf 폭주 방지 (초과분은 '… +N more')

    /* ---- 선택 디스패치 -------------------------------------------------------- */
    function meshEntries() {
        try { return (typeof memberMeshes !== 'undefined' && memberMeshes) ? memberMeshes : []; } catch (_) { return []; }
    }
    function edOf(md) { return md.elementData || (md.mesh && md.mesh.userData && md.mesh.userData.elementData); }
    function clearSel() { try { if (typeof window.clearAllSelection === 'function') window.clearAllSelection(); } catch (_) {} }
    function selectElemById(id) {
        clearSel();
        var mds = meshEntries();
        for (var i = 0; i < mds.length; i++) {
            var ed = edOf(mds[i]);
            if (ed && ed.type !== 'node' && String(ed.id) === String(id)) {
                try { if (typeof highlightMesh === 'function') highlightMesh(mds[i].mesh); } catch (_) {}
                try { selectedMesh = mds[i].mesh; } catch (_) {}
                try { if (typeof window.showMemberProperties === 'function') window.showMemberProperties(ed); } catch (_) {}
                return true;
            }
        }
        // 씬에 메시가 없으면(드문 경우) 모델에서 직접 속성 패널만
        var m = getModel();
        var e = m && (m.elements || []).find(function (x) { return String(x.id) === String(id); });
        if (e && typeof window.showMemberProperties === 'function') {
            try { window.showMemberProperties({ type: e.elem_type, id: e.id, section: e.section, ni: e.node_i, nj: e.node_j }); } catch (_) {}
            return true;
        }
        return false;
    }
    function selectGroup(pred, label) {
        clearSel();
        var n = 0;
        meshEntries().forEach(function (md) {
            var ed = edOf(md);
            if (ed && ed.type !== 'node' && pred(ed)) {
                try { if (typeof highlightMesh === 'function') highlightMesh(md.mesh); n++; } catch (_) {}
            }
        });
        try { if (typeof updateSelectionCount === 'function') updateSelectionCount(); } catch (_) {}
        status(label + ' — 부재 ' + n + '개 선택' + (n ? ' (리본 M/S로 일괄 적용 가능)' : ''), n ? 'success' : 'warning');
    }
    function selectStory(s) {
        clearSel();
        try { if (typeof selectByStory === 'function') { selectByStory(String(s)); return; } } catch (_) {}
        status('층 선택 기능을 사용할 수 없습니다', 'warning');
    }

    /* ---- 트리 ----------------------------------------------------------------
     * 기존 목업의 .tree-row / level-N / open 클래스를 그대로 재사용해 룩 유지.
     * 브랜치 = 헤더 row + 자식 컨테이너(.tree-kids). openKeys로 접힘 상태 보존. */
    var openKeys = { project: 1, structure: 1, elements: 1, properties: 1 };

    function branchRow(parent, key, label, level, count, extraCls) {
        var row = el('div', 'tree-row level-' + level + (openKeys[key] ? ' open' : '') + (extraCls ? ' ' + extraCls : ''), label);
        if (count != null) row.appendChild(el('span', null, String(count)));
        var kids = el('div', 'tree-kids');
        if (!openKeys[key]) kids.style.display = 'none';
        row.addEventListener('click', function () {
            var open = row.classList.toggle('open');
            openKeys[key] = open ? 1 : 0;
            kids.style.display = open ? '' : 'none';
        });
        parent.appendChild(row);
        parent.appendChild(kids);
        return kids;
    }
    function leafRow(parent, label, level, count, onclick, extraCls) {
        var row = el('div', 'tree-row leaf level-' + level + (extraCls ? ' ' + extraCls : ''), label);
        if (count != null) row.appendChild(el('span', null, String(count)));
        if (onclick) {
            row.classList.add('clickable');
            row.addEventListener('click', function (ev) { ev.stopPropagation(); onclick(); });
        }
        parent.appendChild(row);
        return row;
    }

    function buildTree() {
        var host = document.getElementById('fig-mx-tree');
        if (!host) return;
        var m = getModel();
        var r = getResult();
        host.innerHTML = '';

        if (!m || !(m.nodes || []).length) {
            var hint = el('div', 'tree-hint',
                '모델이 없습니다 — IFC 가져오기, 자연어 입력 또는 해석 실행 후 구조 트리가 여기에 표시됩니다.');
            host.appendChild(hint);
            return;
        }

        var nodes = m.nodes || [], elems = m.elements || [];

        // 집계
        var stories = [];
        (function () {
            var s = {};
            nodes.forEach(function (n) { if (n.story != null) s[n.story] = 1; });
            stories = Object.keys(s).map(Number).sort(function (a, b) { return a - b; });
        })();
        var byType = {}, bySection = {}, byMaterial = {};
        elems.forEach(function (e) {
            var t = e.elem_type || 'etc';
            (byType[t] = byType[t] || []).push(e);
            if (e.section) (bySection[e.section] = bySection[e.section] || []).push(e);
            var mat = e.material || m.material_name;
            if (mat) (byMaterial[mat] = byMaterial[mat] || []).push(e);
        });

        var root = branchRow(host, 'project', 'Project', 0);
        var st = branchRow(root, 'structure', 'Structure', 1);

        // Stories
        var uSt = branchRow(st, 'stories', 'Stories', 2, stories.length);
        stories.forEach(function (s) {
            leafRow(uSt, s === 0 ? 'Base (Z=0)' : s + 'F', 3, null, function () { selectStory(s); });
        });

        // Nodes (카운트 전용 — 절점 leaf 폭주 방지)
        leafRow(st, 'Nodes', 2, nodes.length, null);

        // Elements → 타입별 브랜치 → 부재 leaf
        var uEl = branchRow(st, 'elements', 'Elements', 2, elems.length);
        Object.keys(byType).sort().forEach(function (t) {
            var list = byType[t];
            var uT = branchRow(uEl, 'type:' + t, TYPE_LABEL[t] || t, 3, list.length);
            leafRow(uT, '▸ 전체 선택', 3, null, function () {
                selectGroup(function (ed) { return ed.type === t || (t === 'beam' && ed.type === 'beam_x'); }, TYPE_LABEL[t] || t);
            }, 'selall');
            list.slice(0, LEAF_CAP).forEach(function (e) {
                leafRow(uT, '#' + e.id + ' · ' + (e.section || '-'), 3, null, function () { selectElemById(e.id); });
            });
            if (list.length > LEAF_CAP) leafRow(uT, '… +' + (list.length - LEAF_CAP) + ' more', 3, null, null, 'dim');
        });

        // Properties
        var pr = branchRow(root, 'properties', 'Properties', 1);
        var matKeys = Object.keys(byMaterial).sort();
        var uMat = branchRow(pr, 'materials', 'Materials', 2, matKeys.length);
        matKeys.forEach(function (k) {
            leafRow(uMat, k, 3, byMaterial[k].length, function () {
                selectGroup(function (ed) {
                    var me = elems.find(function (x) { return x.id === ed.id; });
                    return me && (me.material || m.material_name) === k;
                }, '재료 ' + k);
            });
        });
        var secKeys = Object.keys(bySection).sort();
        var uSec = branchRow(pr, 'sections', 'Sections', 2, secKeys.length);
        secKeys.forEach(function (k) {
            leafRow(uSec, k, 3, bySection[k].length, function () {
                selectGroup(function (ed) { return ed.section === k; }, '단면 ' + k);
            });
        });

        // Load Cases (해석 후)
        var cases = r ? (r.case_names || []).concat(r.combo_names || []) : [];
        if (cases.length) {
            var uLC = branchRow(root, 'loadcases', 'Load Cases', 1, cases.length);
            (r.case_names || []).forEach(function (c) { leafRow(uLC, c, 2, null, null); });
            if ((r.combo_names || []).length) {
                var uCombo = branchRow(uLC, 'combos', 'Combinations', 2, r.combo_names.length);
                r.combo_names.forEach(function (c) { leafRow(uCombo, c, 3, null, null); });
            }
        } else {
            leafRow(root, 'Load Cases', 1, null, null, 'dim').title = '해석 후 표시됩니다';
        }

        // Analysis Results (해석 후 — Design Check OK/NG)
        var checks = r && r.member_checks ? r.member_checks : null;
        if (checks && Object.keys(checks).length) {
            var ok = [], ng = [];
            Object.keys(checks).forEach(function (id) {
                var stt = (checks[id].status || '').toUpperCase();
                (stt === 'NG' ? ng : ok).push(id);
            });
            var uAR = branchRow(root, 'results', 'Analysis Results', 1, ok.length + ng.length);
            if (ng.length) {
                var uNG = branchRow(uAR, 'results-ng', 'NG', 2, ng.length, 'ng');
                ng.slice(0, LEAF_CAP).forEach(function (id) {
                    var e = elems.find(function (x) { return String(x.id) === String(id); });
                    leafRow(uNG, '#' + id + ' · ' + ((e && e.section) || '-'), 3, null, function () { selectElemById(id); }, 'ng');
                });
            }
            leafRow(uAR, 'OK', 2, ok.length, function () {
                selectGroup(function (ed) { var c = checks[ed.id] || checks[String(ed.id)]; return c && (c.status || '').toUpperCase() !== 'NG'; }, 'Design Check OK');
            }, 'ok');
        } else {
            leafRow(root, 'Analysis Results', 1, null, null, 'dim').title = '해석 후 표시됩니다';
        }
    }

    /* ---- Analysis Control (우측 패널) ------------------------------------------ */
    function setTxt(id, v, cls) {
        var e = document.getElementById(id);
        if (!e) return;
        e.textContent = v;
        e.classList.remove('ok', 'ng');
        if (cls) e.classList.add(cls);
    }
    function overallCheck(r) {
        // design_check 요약: 부재 NG or 층간변위 NG → NG
        try {
            var dc = r.design_check;
            if (dc) {
                var ngN = dc.member_check && dc.member_check.summary ? (dc.member_check.summary.ng || 0) : 0;
                var driftNG = dc.drift_check && String(dc.drift_check.status || '').toUpperCase() === 'NG';
                if (ngN > 0 || driftNG) return 'NG';
                return 'OK';
            }
        } catch (_) {}
        try {
            var ch = r.member_checks || {};
            var ids = Object.keys(ch);
            if (!ids.length) return null;
            return ids.some(function (id) { return (ch[id].status || '').toUpperCase() === 'NG'; }) ? 'NG' : 'OK';
        } catch (_) {}
        return null;
    }
    function updateAnalysisControl() {
        var m = getModel(), r = getResult();
        setTxt('fig-ac-size', m ? ((m.nodes || []).length + ' nodes · ' + (m.elements || []).length + ' elems') : '—');
        if (r) {
            var oc = overallCheck(r);
            setTxt('fig-ac-check', oc || 'Done', oc === 'NG' ? 'ng' : 'ok');
            setTxt('fig-ac-cases', (r.case_names || []).length || '—');
            setTxt('fig-ac-combos', (r.combo_names || []).length || '—');
            setTxt('fig-ac-solver', 'Linear Static' + (r.seismic_method === 'RSA' ? ' + RSA' : ' (ELF)'));
        } else {
            setTxt('fig-ac-check', m ? 'Ready' : '—', m ? 'ok' : null);
            setTxt('fig-ac-cases', '—');
            setTxt('fig-ac-combos', '—');
            setTxt('fig-ac-solver', '—');
        }
        var rep = document.getElementById('fig-btn-report');
        if (rep) rep.classList.toggle('disabled', !(r && r.report_url));
        var exp = document.getElementById('fig-btn-export');
        if (exp) {
            var hasJob = false;
            try { hasJob = !!currentJobId; } catch (_) {}
            exp.classList.toggle('disabled', !hasJob);
        }
    }
    window.figmaOpenReport = function () {
        var r = getResult();
        if (r && r.report_url) window.open(r.report_url, '_blank');
        else status('리포트가 없습니다 — 먼저 해석을 실행하세요', 'warning');
    };
    window.figmaExportTables = function () {
        var hasJob = false;
        try { hasJob = !!currentJobId; } catch (_) {}
        if (!hasJob) { status('내보낼 결과가 없습니다 — 먼저 해석을 실행하세요', 'warning'); return; }
        try { if (typeof exportToExcel === 'function') exportToExcel(); } catch (e) { console.warn('[figma-explorer] export', e); }
    };

    /* ---- Result Tables (실카운트 + 경량 테이블 팝업) ----------------------------- */
    function firstCase(r) { return (r.case_names || [])[0] || (r.combo_names || [])[0] || null; }
    function tableSpec(kind, r, caseName) {
        // returns {title, rows, caseScoped}
        if (kind === 'drift') {
            var cd = (r.case_data || {})[caseName] || {};
            return { title: 'Story Drift — ' + caseName, rows: cd.story_drifts || [], caseScoped: true };
        }
        if (kind === 'reaction') {
            var cd2 = (r.case_data || {})[caseName] || {};
            return { title: 'Base Reaction — ' + caseName, rows: cd2.reactions || [], caseScoped: true };
        }
        if (kind === 'force') {
            return { title: 'Member Force — ' + caseName, rows: (r.member_forces || {})[caseName] || [], caseScoped: true };
        }
        if (kind === 'dc') {
            var ch = r.member_checks || {};
            var rows = Object.keys(ch).map(function (id) {
                var c = ch[id];
                return { member: '#' + id, status: c.status, ratio: c.interaction_ratio != null ? c.interaction_ratio : c.ratio, governing: c.governing_combo || c.combo || '' };
            });
            return { title: 'Design Check (' + rows.length + ')', rows: rows, caseScoped: false };
        }
        if (kind === 'modal') {
            var ma = r.modal_analysis || {};
            return { title: 'Modal Period (' + ((ma.modes || []).length) + ' modes)', rows: ma.modes || [], caseScoped: false };
        }
        return { title: kind, rows: [], caseScoped: false };
    }
    function fmtCell(v) {
        if (v == null) return '-';
        if (typeof v === 'number') {
            if (Number.isInteger(v)) return String(v);
            var a = Math.abs(v);
            return v.toFixed(a >= 100 ? 1 : a >= 1 ? 2 : 4);
        }
        if (typeof v === 'object') return JSON.stringify(v);
        return String(v);
    }
    var rtOverlay = null;
    function ensureRtOverlay() {
        if (rtOverlay) return;
        rtOverlay = document.createElement('div');
        rtOverlay.id = 'fig-rt-overlay';
        document.body.appendChild(rtOverlay);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && rtOverlay.classList.contains('open')) closeRt();
        }, true);
    }
    function closeRt() { if (rtOverlay) { rtOverlay.classList.remove('open'); rtOverlay.innerHTML = ''; } }
    window.figmaOpenResultTable = function (kind) {
        var r = getResult();
        if (!r) { status('결과가 없습니다 — 먼저 해석을 실행하세요', 'warning'); return; }
        ensureRtOverlay();
        rtOverlay.innerHTML = '';
        rtOverlay.classList.add('open');

        var win = el('div', 'fig-mat-win fig-rt-win');
        var head = el('div', 'fmw-head');
        var title = el('span', null, '');
        var x = el('button', 'fmw-close', '✕');
        x.addEventListener('click', closeRt);
        head.appendChild(title); head.appendChild(x);
        var body = el('div', 'fmw-body');
        win.appendChild(head); win.appendChild(body);
        rtOverlay.appendChild(win);

        var caseName = firstCase(r);
        function render() {
            body.innerHTML = '';
            var spec = tableSpec(kind, r, caseName);
            title.textContent = spec.title;
            if (spec.caseScoped) {
                var selWrap = el('div', 'fmw-row');
                selWrap.appendChild(el('label', null, 'Case / Combo'));
                var sel = document.createElement('select');
                (r.case_names || []).concat(r.combo_names || []).forEach(function (c) {
                    var op = document.createElement('option');
                    op.value = c; op.textContent = c;
                    if (c === caseName) op.selected = true;
                    sel.appendChild(op);
                });
                sel.addEventListener('change', function () { caseName = sel.value; render(); });
                selWrap.appendChild(sel);
                body.appendChild(selWrap);
            }
            var rows = spec.rows;
            if (!rows || !rows.length) {
                body.appendChild(el('div', 'fmw-empty', '이 결과에는 해당 데이터가 없습니다.'));
                return;
            }
            var cols = Object.keys(rows[0]);
            var wrap = el('div', 'fig-rt-tablewrap');
            var tbl = el('table', 'fmw-fytable fig-rt-table');
            var thead = el('thead'); var hr = el('tr');
            cols.forEach(function (c) { hr.appendChild(el('th', null, c)); });
            thead.appendChild(hr); tbl.appendChild(thead);
            var tbody = el('tbody');
            rows.slice(0, 400).forEach(function (row) {
                var tr = el('tr');
                cols.forEach(function (c) { tr.appendChild(el('td', null, fmtCell(row[c]))); });
                tbody.appendChild(tr);
            });
            tbl.appendChild(tbody);
            wrap.appendChild(tbl);
            body.appendChild(wrap);
            if (rows.length > 400) body.appendChild(el('div', 'fmw-warn', '표시 400행 / 전체 ' + rows.length + '행 — 전체는 Export Tables(Excel) 사용'));
        }
        render();
    };
    function updateResultCounts() {
        var r = getResult();
        var c1 = firstCase(r || {});
        function cnt(kind) {
            if (!r) return '—';
            try {
                var n = tableSpec(kind, r, c1).rows.length;
                return n || '—';
            } catch (_) { return '—'; }
        }
        [['fig-rt-drift', 'drift'], ['fig-rt-reaction', 'reaction'], ['fig-rt-force', 'force'], ['fig-rt-dc', 'dc'], ['fig-rt-modal', 'modal']]
            .forEach(function (p) {
                var e = document.getElementById(p[0]);
                if (e) e.textContent = cnt(p[1]);
            });
        document.querySelectorAll('.figma-result-panel .result-row').forEach(function (b) {
            b.classList.toggle('disabled', !r);
        });
    }

    /* ---- 실시간 갱신 훅 -------------------------------------------------------- */
    var refreshTimer = null;
    function scheduleRefresh() {
        if (refreshTimer) return;
        refreshTimer = setTimeout(function () {
            refreshTimer = null;
            try { buildTree(); } catch (e) { console.warn('[figma-explorer] tree', e); }
            try { updateAnalysisControl(); } catch (e) { console.warn('[figma-explorer] control', e); }
            try { updateResultCounts(); } catch (e) { console.warn('[figma-explorer] counts', e); }
        }, 60);
    }
    window.figmaExplorerRefresh = scheduleRefresh;   // 테스트/외부 훅

    function wrapRenderer(name) {
        var orig = window[name];
        if (typeof orig !== 'function' || orig._figmaMxWrapped) return;
        window[name] = function () {
            var out = orig.apply(this, arguments);
            try { scheduleRefresh(); } catch (_) {}
            return out;
        };
        window[name]._figmaMxWrapped = true;
    }

    function init() {
        wrapRenderer('buildV2PreviewScene');
        wrapRenderer('buildScene');
        wrapRenderer('clearPreviewScene');
        scheduleRefresh();
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
