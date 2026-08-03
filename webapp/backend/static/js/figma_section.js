/* ============================================================================
 * figma_section.js  —  Section(Frame Properties) 팝업 (Figma UI Lab, 철골 전용)
 * ----------------------------------------------------------------------------
 * ETABS의 Define > Section Properties > Frame Sections 3단을 재현:
 *   ① Frame Properties      — Type/Filter/Find + 738건 목록 + Click to 버튼
 *   ② Frame Section Property Data — 치수(읽기전용)+단면 스케치+재료 페어링
 *   ③ Show Section Properties     — A/Ix/Iy/J 물성 (J 근사 주의 표기)
 *
 * 데이터 소스 (백엔드 무변경):
 *   - 목록: 엔진 전역 `sectionsList` (= /api/sections/list, KS D 3502/3568
 *     12개 테이블 738건, Supabase)
 *   - 상세: /api/sections/properties/{name} (h/b/tw/tf/t + A/Ix/Iy/J/j_source)
 *
 * 의미론:
 *   - 카탈로그는 읽기 전용 — Add New/Copy/Delete/SD 변환은 비활성(백로그).
 *     사용자 정의 단면은 해석의 name 기반 DB 조회(get_section_3d)가 못 풀므로
 *     UI만 열어주면 조용히 깨진다 → 비활성이 정직한 스코프.
 *   - Material 페어링: ETABS는 단면 속성이 재료를 가짐. 여기선 상세창에서
 *     재료를 고르면 세션 페어링으로 저장하고, '선택 부재에 적용' 시
 *     단면+페어링 재료를 한 undo 스텝으로 함께 적용.
 * ========================================================================== */
(function () {
    'use strict';

    var pairing = {};            // section name -> material name (세션)
    var detailCache = {};        // section name -> /api/sections/properties 응답
    var overlay = null;
    var winStack = [];
    var listWin = null;
    var filterState = { group: '', text: '', find: '' };

    /* ---- engine helpers ----------------------------------------------------- */
    function getModel() { try { return window._v2Model || null; } catch (_) { return null; } }
    function status(msg, kind) { try { if (typeof setStatus === 'function') setStatus(msg, kind || 'success'); } catch (_) {} }
    function catalogGroups() {
        try { return (typeof sectionsList === 'object' && sectionsList) ? sectionsList : {}; } catch (_) { return {}; }
    }
    function selectedIds() {
        var ids = [];
        try { if (typeof getSelectedElementIds === 'function') ids = getSelectedElementIds() || []; } catch (_) {}
        if (!ids.length && window._figmaSelectedElemId != null) ids = [window._figmaSelectedElemId];
        return ids;
    }
    function usageCounts() {
        var m = getModel(), cnt = {};
        ((m && m.elements) || []).forEach(function (e) {
            if (e.section) cnt[e.section] = (cnt[e.section] || 0) + 1;
        });
        return cnt;
    }

    /* ---- 단면 적용 (단면 + 페어링 재료를 한 undo 스텝으로) -------------------- */
    window.figmaApplySectionToSelection = function (name) {
        var model = getModel();
        if (!model || !name) return;
        var ids = selectedIds();
        if (!ids.length) { status('선택된 부재가 없습니다', 'warning'); return; }
        try { if (typeof pushUndo === 'function') pushUndo(); } catch (_) {}
        var mat = pairing[name] || null, n = 0;
        (model.elements || []).forEach(function (e) {
            if (ids.indexOf(e.id) >= 0) { e.section = name; if (mat) e.material = mat; n++; }
        });
        try { if (typeof refreshEditPreview === 'function') refreshEditPreview(); } catch (_) {}
        status('단면 적용: 부재 ' + n + '개 → ' + name + (mat ? ' (+재료 ' + mat + ')' : ''));
    };

    /* ---- shape type (API section_type + prefix 보정) -------------------------- */
    function shapeType(name, apiType) {
        if (name.indexOf('FB-') === 0) return 'FB';        // API가 FB를 H로 오분류
        return apiType || 'H';
    }
    var SHAPE_LABEL = {
        H: 'Steel I/Wide Flange (H형강)', I: 'Steel I (I형강)',
        C: 'Steel Channel (ㄷ형강)', T: 'Steel Tee (T형강)',
        L: 'Steel Angle (ㄱ형강)', FB: 'Flat Bar (구평형강)',
        CHS: 'Steel Pipe (원형강관)', SHS: 'Steel Tube (정사각형강관)',
        RHS: 'Steel Tube (직사각형강관)',
    };

    /* ---- window chrome (figma_material과 동일 룩, 독립 스택) ------------------ */
    function ensureOverlay() {
        if (overlay) return;
        overlay = document.createElement('div');
        overlay.id = 'fig-sec-overlay';
        document.body.appendChild(overlay);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && winStack.length) { e.stopPropagation(); closeTop(); }
        }, true);
    }
    function openWin(title, width) {
        ensureOverlay();
        overlay.classList.add('open');
        var w = document.createElement('div');
        w.className = 'fig-mat-win';
        w.style.width = (width || 470) + 'px';
        var off = winStack.length * 26;
        w.style.transform = 'translate(calc(-50% + ' + off + 'px), calc(-50% + ' + off + 'px))';
        var head = document.createElement('div');
        head.className = 'fmw-head';
        var t = document.createElement('span'); t.textContent = title;
        var x = document.createElement('button'); x.className = 'fmw-close'; x.textContent = '✕';
        x.addEventListener('click', function () { closeWin(w); });
        head.appendChild(t); head.appendChild(x);
        var body = document.createElement('div'); body.className = 'fmw-body';
        w.appendChild(head); w.appendChild(body);
        overlay.appendChild(w);
        winStack.push(w);
        return body;
    }
    function closeWin(w) {
        var i = winStack.indexOf(w);
        if (i >= 0) winStack.splice(i, 1);
        if (w === listWin) listWin = null;
        w.remove();
        if (!winStack.length) overlay.classList.remove('open');
    }
    function closeTop() { if (winStack.length) closeWin(winStack[winStack.length - 1]); }
    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }
    function row(label, node) {
        var r = el('div', 'fmw-row');
        r.appendChild(el('label', null, label));
        r.appendChild(node);
        return r;
    }
    function roInput(value) {
        var i = document.createElement('input');
        i.type = 'text'; i.readOnly = true; i.classList.add('ro');
        i.value = value != null ? value : '';
        return i;
    }
    function btn(label, primary, onclick) {
        var b = el('button', 'fmw-btn' + (primary ? ' primary' : ''), label);
        b.addEventListener('click', onclick);
        return b;
    }
    function disabledBtn(label, note) {
        var b = el('button', 'fmw-btn', label);
        b.disabled = true;
        b.title = note || '미지원 (백로그)';
        return b;
    }

    /* ---- ① Frame Properties --------------------------------------------------- */
    window.figmaOpenSections = function () {
        var groups = catalogGroups();
        if (!Object.keys(groups).length) { status('단면 목록이 아직 로드되지 않았습니다 (/api/sections/list)', 'warning'); return; }
        if (listWin) { renderList(); return; }
        var body = openWin('Frame Properties', 500);
        listWin = winStack[winStack.length - 1];
        buildListSkeleton(body);
        renderList();
    };

    var listBox = null, findInput = null, filterInput = null, typeSel = null, applyBtn = null;
    function buildListSkeleton(body) {
        var groups = catalogGroups();

        body.appendChild(el('div', 'fmw-grouplabel', 'Filter Properties List'));
        var typeRow = el('div', 'fms-filterrow');
        typeRow.appendChild(el('label', null, 'Type'));
        typeSel = document.createElement('select');
        [['', 'All']].concat(Object.keys(groups).map(function (g) { return [g, g]; }))
            .forEach(function (o) {
                var op = document.createElement('option');
                op.value = o[0]; op.textContent = o[1];
                typeSel.appendChild(op);
            });
        typeSel.addEventListener('change', function () { filterState.group = typeSel.value; renderList(); });
        typeRow.appendChild(typeSel);
        body.appendChild(typeRow);

        var fRow = el('div', 'fms-filterrow');
        fRow.appendChild(el('label', null, 'Filter'));
        filterInput = document.createElement('input'); filterInput.type = 'text';
        filterInput.addEventListener('input', function () { filterState.text = filterInput.value.trim(); renderList(); });
        fRow.appendChild(filterInput);
        var clr = btn('Clear', false, function () {
            filterInput.value = ''; filterState.text = '';
            typeSel.value = ''; filterState.group = '';
            renderList();
        });
        clr.classList.add('fms-clear');
        fRow.appendChild(clr);
        body.appendChild(fRow);

        body.appendChild(el('div', 'fmw-grouplabel', 'Properties'));
        var wrap = el('div', 'fmw-cols');
        var left = el('div', 'fmw-listcol');
        left.appendChild(el('div', 'fms-findlabel', 'Find This Property'));
        findInput = document.createElement('input'); findInput.type = 'text'; findInput.className = 'fms-find';
        findInput.addEventListener('input', function () { filterState.find = findInput.value.trim(); renderList(); });
        left.appendChild(findInput);
        listBox = el('div', 'fmw-listbox fms-listbox');
        left.appendChild(listBox);

        var right = el('div', 'fmw-btncol');
        right.appendChild(el('div', 'fmw-grouplabel', 'Click to:'));
        right.appendChild(disabledBtn('Import New Properties…', 'KS 카탈로그(738건) 사용 — import 불필요'));
        right.appendChild(disabledBtn('Add New Property…', '사용자 정의 단면 미지원 (해석이 KS 이름으로 DB 조회) — 백로그'));
        right.appendChild(disabledBtn('Add Copy of Property…', '사용자 정의 단면 미지원 — 백로그'));
        right.appendChild(btn('Modify/Show Property…', true, function () {
            var n = selName(); if (!n) return alertNeedSel();
            openDetail(n);
        }));
        right.appendChild(el('div', 'fmw-sep'));
        right.appendChild(disabledBtn('Delete Property', 'DB 카탈로그 단면은 삭제 불가'));
        right.appendChild(disabledBtn('Delete Multiple Properties…', 'DB 카탈로그 단면은 삭제 불가'));
        right.appendChild(el('div', 'fmw-sep'));
        applyBtn = btn('선택 부재에 적용', false, function () {
            var n = selName(); if (!n) return alertNeedSel();
            window.figmaApplySectionToSelection(n);
            renderList();
        });
        right.appendChild(applyBtn);
        right.appendChild(el('div', 'fmw-sep'));
        right.appendChild(disabledBtn('Convert to SD Section', 'Section Designer 미지원'));
        right.appendChild(disabledBtn('Export to XML File…', '미지원 — 백로그'));
        right.appendChild(el('div', 'fmw-sep'));
        right.appendChild(btn('OK', false, function () { closeWin(listWin); }));

        wrap.appendChild(left); wrap.appendChild(right);
        body.appendChild(wrap);
        body.appendChild(el('div', 'fmw-hint',
            'KS D 3502/3568 단면 카탈로그 (Supabase, 12개 강종 테이블) · 치수/물성 읽기 전용 · 적용 시 페어링된 재료 동시 반영'));
    }
    function selName() {
        var s = listBox ? listBox.querySelector('.fmw-item.sel') : null;
        return s ? s.dataset.name : null;
    }
    function alertNeedSel() { status('목록에서 단면을 먼저 선택하세요', 'warning'); }

    function renderList() {
        if (!listBox) return;
        var groups = catalogGroups(), use = usageCounts();
        var prevSel = selName();
        listBox.innerHTML = '';
        var textLc = filterState.text.toLowerCase();
        var findLc = filterState.find.toLowerCase();
        var total = 0, shown = 0, firstFind = null;
        Object.keys(groups).forEach(function (g) {
            if (filterState.group && g !== filterState.group) return;
            (groups[g] || []).forEach(function (n) {
                total++;
                var lc = n.toLowerCase();
                if (textLc && lc.indexOf(textLc) < 0) return;
                if (findLc && lc.indexOf(findLc) < 0) return;
                shown++;
                var label = n + (use[n] ? '   (부재 ' + use[n] + ')' : '') + (pairing[n] ? '  ⚭' + pairing[n] : '');
                var it = el('div', 'fmw-item', label);
                it.dataset.name = n;
                it.title = g;
                if (n === prevSel) it.classList.add('sel');
                if (findLc && !firstFind) firstFind = it;
                it.addEventListener('click', function () {
                    listBox.querySelectorAll('.fmw-item').forEach(function (o) { o.classList.remove('sel'); });
                    it.classList.add('sel');
                });
                it.addEventListener('dblclick', function () { openDetail(n); });
                listBox.appendChild(it);
            });
        });
        if (!shown) listBox.appendChild(el('div', 'fmw-empty', '(일치하는 단면 없음)'));
        // Find: 첫 일치 자동 선택 + 스크롤
        if (firstFind && !listBox.querySelector('.fmw-item.sel')) {
            firstFind.classList.add('sel');
            firstFind.scrollIntoView({ block: 'nearest' });
        }
        var ids = selectedIds();
        applyBtn.textContent = '선택 부재에 적용 (' + ids.length + ')';
        applyBtn.disabled = !ids.length;
        var cap = listWin.querySelector('.fms-count');
        if (!cap) { cap = el('div', 'fms-count'); listBox.parentNode.appendChild(cap); }
        cap.textContent = shown + ' / ' + total + '건';
    }

    /* ---- ② Frame Section Property Data ---------------------------------------- */
    function fetchDetail(name) {
        if (detailCache[name]) return Promise.resolve(detailCache[name]);
        return fetch('/api/sections/properties/' + encodeURIComponent(name))
            .then(function (r) { return r.json(); })
            .then(function (d) { detailCache[name] = d; return d; });
    }
    function openDetail(name) {
        fetchDetail(name).then(function (d) {
            if (d.error) { status('단면 조회 실패: ' + d.error, 'warning'); return; }
            var body = openWin('Frame Section Property Data', 640);
            var win = winStack[winStack.length - 1];
            var st = shapeType(name, d.section_type);

            var cols = el('div', 'fms-cols');
            var left = el('div', 'fms-left');
            var right = el('div', 'fms-right');

            /* General */
            left.appendChild(el('div', 'fmw-grouplabel', 'General Data'));
            left.appendChild(row('Property Name', roInput(d.name)));
            var matSel = document.createElement('select');
            var matNames = [];
            try { if (typeof window._figmaMaterialNames === 'function') matNames = window._figmaMaterialNames(); } catch (_) {}
            [['', '(미지정 — 부재 재료 유지)']].concat(matNames.map(function (m) { return [m, m]; }))
                .forEach(function (o) {
                    var op = document.createElement('option');
                    op.value = o[0]; op.textContent = o[1];
                    if (o[0] === (pairing[name] || '')) op.selected = true;
                    matSel.appendChild(op);
                });
            var matWrap = el('div', 'fms-matwrap');
            matWrap.appendChild(matSel);
            var dots = btn('…', false, function () {
                try { window.figmaOpenMaterials(); } catch (e) { status('재료 팝업을 열 수 없습니다', 'warning'); }
            });
            dots.classList.add('fms-dots');
            dots.title = '재료 정의 (Define Materials)';
            matWrap.appendChild(dots);
            left.appendChild(row('Material', matWrap));
            var colorIn = roInput('기본 (미지원)'); colorIn.title = '단면별 표시 색상 미지원 — 백로그';
            left.appendChild(row('Display Color', colorIn));

            /* Shape + Source */
            left.appendChild(el('div', 'fmw-grouplabel', 'Shape'));
            left.appendChild(row('Section Shape', roInput(SHAPE_LABEL[st] || st)));
            left.appendChild(el('div', 'fmw-grouplabel', 'Section Property Source'));
            var srcRow = el('div', 'fms-srcrow');
            srcRow.appendChild(el('span', null, 'Source:  KS D ' + (st === 'CHS' || st === 'SHS' || st === 'RHS' ? '3568' : '3502') + ' (Supabase)'));
            var cvt = disabledBtn('Convert To User Defined', '사용자 정의 단면 미지원 — 백로그');
            srcRow.appendChild(cvt);
            left.appendChild(srcRow);

            /* Dimensions */
            left.appendChild(el('div', 'fmw-grouplabel', 'Section Dimensions (mm)'));
            dimRows(st, d).forEach(function (r) { left.appendChild(row(r[0], roInput(r[1]))); });

            /* Right: sketch + modifiers */
            var canvas = document.createElement('canvas');
            canvas.width = 200; canvas.height = 200;
            canvas.className = 'fms-sketch';
            right.appendChild(canvas);
            right.appendChild(el('div', 'fmw-grouplabel', 'Property Modifiers'));
            right.appendChild(disabledBtn('Modify/Show Modifiers…', '강성 수정계수 미지원 — 백로그'));
            right.appendChild(el('div', 'fms-moddefault', 'Currently Default'));

            cols.appendChild(left); cols.appendChild(right);
            body.appendChild(cols);

            var showProps = btn('Show Section Properties…', false, function () { openProps(name, d, st); });
            showProps.classList.add('fms-showprops');
            body.appendChild(showProps);

            var foot = el('div', 'fmw-foot');
            foot.appendChild(btn('OK', true, function () {
                var m = matSel.value;
                if (m) pairing[name] = m; else delete pairing[name];
                closeWin(win);
                renderList();
                status(m ? ('단면 ' + name + ' ⚭ 재료 ' + m + ' 페어링 저장 (적용 시 함께 반영)') : ('단면 ' + name + ' 확인'));
            }));
            foot.appendChild(btn('Cancel', false, function () { closeWin(win); }));
            body.appendChild(foot);

            drawSketch(canvas, st, d);
        }).catch(function (e) { status('단면 조회 실패: ' + e, 'warning'); });
    }
    function dimRows(st, d) {
        if (st === 'CHS') return [['Outside Diameter, D', d.h_mm], ['Wall Thickness, t', d.t_mm]];
        if (st === 'SHS' || st === 'RHS') return [['Total Depth, H', d.h_mm], ['Total Width, B', d.b_mm], ['Wall Thickness, t', d.t_mm]];
        if (st === 'L') return [['Leg A', d.h_mm], ['Leg B', d.b_mm], ['Thickness', d.tw_mm || d.tf_mm]];
        if (st === 'FB') return [['Depth, a', d.h_mm], ['Thickness, t', d.tw_mm || d.b_mm]];
        if (st === 'T') return [['Total Depth', d.h_mm], ['Flange Width', d.b_mm], ['Flange Thickness', d.tf_mm], ['Stem Thickness', d.tw_mm]];
        // H / I / C
        return [['Total Depth', d.h_mm], ['Flange Width', d.b_mm],
                ['Flange Thickness', d.tf_mm], ['Web Thickness', d.tw_mm]];
    }

    /* ---- 단면 스케치 (2축=상향 초록, 3축=좌향 파랑 — ETABS 표기) --------------- */
    function drawSketch(canvas, st, d) {
        var ctx = canvas.getContext('2d');
        var W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#eef3f8';
        ctx.fillRect(0, 0, W, H);

        var h = d.h_mm || 100, b = d.b_mm || h, tf = d.tf_mm || 10, tw = d.tw_mm || 8, t = d.t_mm || tw;
        var box = 132, s = box / Math.max(h, b);
        var cx = W / 2, cy = H / 2;
        var hw = b * s / 2, hh = h * s / 2, ftf = Math.max(tf * s, 2.5), ftw = Math.max(tw * s, 2), ft = Math.max(t * s, 2.5);

        ctx.fillStyle = '#b9c9da';
        ctx.strokeStyle = '#41546b';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        if (st === 'CHS') {
            var R = hh;
            ctx.arc(cx, cy, R, 0, Math.PI * 2);
            ctx.arc(cx, cy, Math.max(R - ft, 1), 0, Math.PI * 2, true);
        } else if (st === 'SHS' || st === 'RHS') {
            ctx.rect(cx - hw, cy - hh, hw * 2, hh * 2);
            ctx.rect(cx - hw + ft, cy - hh + ft, Math.max(hw * 2 - 2 * ft, 1), Math.max(hh * 2 - 2 * ft, 1));
            ctx.fill('evenodd'); ctx.stroke();
            drawAxes(ctx, cx, cy); return;
        } else if (st === 'L') {
            ctx.moveTo(cx - hw, cy - hh);
            ctx.lineTo(cx - hw + ftw, cy - hh);
            ctx.lineTo(cx - hw + ftw, cy + hh - ftw);
            ctx.lineTo(cx + hw, cy + hh - ftw);
            ctx.lineTo(cx + hw, cy + hh);
            ctx.lineTo(cx - hw, cy + hh);
            ctx.closePath();
        } else if (st === 'T') {
            ctx.moveTo(cx - hw, cy - hh);
            ctx.lineTo(cx + hw, cy - hh);
            ctx.lineTo(cx + hw, cy - hh + ftf);
            ctx.lineTo(cx + ftw / 2, cy - hh + ftf);
            ctx.lineTo(cx + ftw / 2, cy + hh);
            ctx.lineTo(cx - ftw / 2, cy + hh);
            ctx.lineTo(cx - ftw / 2, cy - hh + ftf);
            ctx.lineTo(cx - hw, cy - hh + ftf);
            ctx.closePath();
        } else if (st === 'C') {
            ctx.moveTo(cx + hw, cy - hh);
            ctx.lineTo(cx - hw, cy - hh);
            ctx.lineTo(cx - hw, cy + hh);
            ctx.lineTo(cx + hw, cy + hh);
            ctx.lineTo(cx + hw, cy + hh - ftf);
            ctx.lineTo(cx - hw + ftw, cy + hh - ftf);
            ctx.lineTo(cx - hw + ftw, cy - hh + ftf);
            ctx.lineTo(cx + hw, cy - hh + ftf);
            ctx.closePath();
        } else if (st === 'FB') {
            ctx.rect(cx - ftw / 2, cy - hh, ftw, hh * 2);
        } else { // H / I
            ctx.moveTo(cx - hw, cy - hh);
            ctx.lineTo(cx + hw, cy - hh);
            ctx.lineTo(cx + hw, cy - hh + ftf);
            ctx.lineTo(cx + ftw / 2, cy - hh + ftf);
            ctx.lineTo(cx + ftw / 2, cy + hh - ftf);
            ctx.lineTo(cx + hw, cy + hh - ftf);
            ctx.lineTo(cx + hw, cy + hh);
            ctx.lineTo(cx - hw, cy + hh);
            ctx.lineTo(cx - hw, cy + hh - ftf);
            ctx.lineTo(cx - ftw / 2, cy + hh - ftf);
            ctx.lineTo(cx - ftw / 2, cy - hh + ftf);
            ctx.lineTo(cx - hw, cy - hh + ftf);
            ctx.closePath();
        }
        if (st === 'CHS') { ctx.fill('evenodd'); ctx.stroke(); }
        else { ctx.fill(); ctx.stroke(); }
        drawAxes(ctx, cx, cy);
    }
    function drawAxes(ctx, cx, cy) {
        // 2축(상향, 초록) / 3축(좌향, 파랑)
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = '#0c8043'; ctx.fillStyle = '#0c8043';
        arrow(ctx, cx, cy, cx, cy - 46);
        ctx.font = 'bold 11px system-ui'; ctx.fillText('2', cx + 5, cy - 38);
        ctx.strokeStyle = '#1a56db'; ctx.fillStyle = '#1a56db';
        arrow(ctx, cx, cy, cx - 46, cy);
        ctx.fillText('3', cx - 42, cy + 13);
    }
    function arrow(ctx, x1, y1, x2, y2) {
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        var a = Math.atan2(y2 - y1, x2 - x1);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - 7 * Math.cos(a - 0.42), y2 - 7 * Math.sin(a - 0.42));
        ctx.lineTo(x2 - 7 * Math.cos(a + 0.42), y2 - 7 * Math.sin(a + 0.42));
        ctx.closePath(); ctx.fill();
    }

    /* ---- ③ Show Section Properties -------------------------------------------- */
    function openProps(name, d, st) {
        var body = openWin('Property Data — ' + name, 380);
        var win = winStack[winStack.length - 1];
        var tbl = el('table', 'fmw-fytable');
        var head = el('tr');
        ['Property', 'Value'].forEach(function (h) { head.appendChild(el('th', null, h)); });
        tbl.appendChild(head);
        var rows = [
            ['Cross-section Area, A', d.A_cm2 + ' cm²'],
            ['Moment of Inertia, I33 (Ix)', d.Ix_cm4 + ' cm⁴'],
            ['Moment of Inertia, I22 (Iy)', d.Iy_cm4 + ' cm⁴'],
            ['Torsional Constant, J', d.J_cm4 + ' cm⁴' + (d.j_source === 'approximate' ? ' *' : '')],
        ];
        rows.forEach(function (r) {
            var tr = el('tr');
            tr.appendChild(el('td', null, r[0]));
            tr.appendChild(el('td', null, String(r[1])));
            tbl.appendChild(tr);
        });
        body.appendChild(tbl);
        if (d.j_source === 'approximate') {
            body.appendChild(el('div', 'fmw-warn',
                '* J = 개방단면 근사 (1/3)Σbt³ — KS 표값 대비 과소평가 가능. 대칭 평면골조에선 비틀림 비연성이라 해석 영향 없음.'));
        }
        var foot = el('div', 'fmw-foot');
        foot.appendChild(btn('OK', true, function () { closeWin(win); }));
        body.appendChild(foot);
    }
})();
