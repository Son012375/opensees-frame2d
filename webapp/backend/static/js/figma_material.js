/* ============================================================================
 * figma_material.js  —  Material 정의 팝업 (Figma UI Lab, 철골 전용)
 * ----------------------------------------------------------------------------
 * ETABS의 재료 3단 팝업(Define Materials → Add New Material Property →
 * Material Property Data)을 재현. 데이터 소스는 /api/materials/detail
 * (Supabase materials: KS 강종 × 두께구간 fy + fu/density/standard) +
 * 표시 상수(ν=0.3, α=1.2e-5, G=E/2(1+ν)).
 *
 * - 카탈로그(DB)는 읽기 전용, 사용자 편집본은 세션 한정 custom 맵에 저장
 *   (카탈로그를 가리는 shadow). 커스텀 영속화는 백로그.
 * - 부재 반영: window.bulkApplyMaterial() — #bulk-material 스텁 +
 *   getSelectedElementIds() 기반, pushUndo/refreshEditPreview 사용
 *   (엔진 bulkApplySection과 동일 패턴, 엔진 JS 무수정).
 * - 진입: 리본 M(Material) onclick + figma_menu Model 메뉴.
 * - 주의: 콘크리트/철근 타입은 비활성(철골 프레임 전용 스코프).
 *   해석은 전 강종 E=205,000 동일이라 강성엔 무손실이나, 설계검토 fy는
 *   현재 모델 전역 1개(check_member_strengths)라 강종 혼용 반영은 백엔드
 *   백로그(M3)임을 힌트로 표기.
 * ========================================================================== */
(function () {
    'use strict';

    var catalog = null;          // /api/materials/detail 결과 (name -> entry)
    var catalogErr = null;
    var custom = {};             // 세션 커스텀/편집본 (name -> entry, catalog를 가림)
    var defined = {};            // 사용자가 Add New/OK로 '정의'한 이름 (값이 카탈로그와 같아도 목록에 표시)
    var overlay = null;
    var winStack = [];           // 열린 창 스택 (Esc/✕로 top부터 닫기)
    var listWin = null;

    /* ---- engine helpers ----------------------------------------------------- */
    function getModel() { try { return window._v2Model || null; } catch (_) { return null; } }
    function status(msg, kind) { try { if (typeof setStatus === 'function') setStatus(msg, kind || 'success'); } catch (_) {} }
    function selectedIds() {
        var ids = [];
        try { if (typeof getSelectedElementIds === 'function') ids = getSelectedElementIds() || []; } catch (_) {}
        if (!ids.length && window._figmaSelectedElemId != null) ids = [window._figmaSelectedElemId];
        return ids;
    }

    /* ---- catalog ------------------------------------------------------------ */
    function fetchCatalog() {
        return fetch('/api/materials/detail').then(function (r) { return r.json(); }).then(function (d) {
            catalog = {};
            (d.materials || []).forEach(function (m) { catalog[m.name] = m; });
            catalogErr = d.fallback ? (d.error || 'fallback') : null;
            return catalog;
        }).catch(function (e) { catalog = {}; catalogErr = String(e); return catalog; });
    }
    function ensureCatalog(cb) {
        if (catalog) { cb(); return; }
        fetchCatalog().then(cb);
    }
    function findMat(name) {
        if (custom[name]) return custom[name];
        if (catalog && catalog[name]) return catalog[name];
        return null;
    }
    function usageCounts() {
        var m = getModel(), cnt = {};
        ((m && m.elements) || []).forEach(function (e) {
            if (e.material) cnt[e.material] = (cnt[e.material] || 0) + 1;
        });
        return cnt;
    }
    /* Define Materials 목록 = 모델 사용중 ∪ 세션 정의 (ETABS의 '정의된 재료' 개념) */
    function definedNames() {
        var names = {}, use = usageCounts();
        Object.keys(use).forEach(function (n) { names[n] = 1; });
        Object.keys(custom).forEach(function (n) { names[n] = 1; });
        Object.keys(defined).forEach(function (n) { names[n] = 1; });
        return Object.keys(names).sort();
    }
    /* 재료 드롭다운용 전체 이름 (figma_plan 편집 패널에서 사용) */
    window._figmaMaterialNames = function () {
        var names = {}, use = usageCounts();
        Object.keys(use).forEach(function (n) { names[n] = 1; });
        Object.keys(custom).forEach(function (n) { names[n] = 1; });
        Object.keys(catalog || {}).forEach(function (n) { names[n] = 1; });
        return Object.keys(names).sort();
    };

    /* ---- 부재 일괄 재료 적용 (엔진 bulkApplySection 패턴) --------------------- */
    window.bulkApplyMaterial = function () {
        var inp = document.getElementById('bulk-material');
        var v = inp ? inp.value : '';
        var model = getModel();
        if (!v || !model) return;
        var ids = selectedIds();
        if (!ids.length) { status('선택된 부재가 없습니다', 'warning'); return; }
        try { if (typeof pushUndo === 'function') pushUndo(); } catch (_) {}
        var n = 0;
        (model.elements || []).forEach(function (e) {
            if (ids.indexOf(e.id) >= 0) { e.material = v; n++; }
        });
        try { if (typeof refreshEditPreview === 'function') refreshEditPreview(); } catch (_) {}
        status('재료 변경: 부재 ' + n + '개 → ' + v);
    };

    /* ---- window chrome ------------------------------------------------------ */
    function ensureOverlay() {
        if (overlay) return;
        overlay = document.createElement('div');
        overlay.id = 'fig-mat-overlay';
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
        w.style.width = (width || 420) + 'px';
        var off = winStack.length * 28;
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
    function mkInput(value, opts) {
        var i = document.createElement('input');
        i.type = (opts && opts.text) ? 'text' : 'number';
        if (opts && opts.step) i.step = opts.step;
        i.value = value != null ? value : '';
        if (opts && opts.readonly) { i.readOnly = true; i.classList.add('ro'); }
        return i;
    }
    function mkSelect(options, current) {
        var s = document.createElement('select');
        options.forEach(function (o) {
            var op = document.createElement('option');
            op.value = o[0]; op.textContent = o[1];
            if (o.length > 2 && o[2]) op.disabled = true;
            if (o[0] === current) op.selected = true;
            s.appendChild(op);
        });
        return s;
    }
    function btn(label, primary, onclick) {
        var b = el('button', 'fmw-btn' + (primary ? ' primary' : ''), label);
        b.addEventListener('click', onclick);
        return b;
    }

    /* ---- ① Define Materials -------------------------------------------------- */
    window.figmaOpenMaterials = function () {
        ensureCatalog(function () {
            if (listWin) { renderList(); return; }
            var body = openWin('Define Materials', 430);
            listWin = winStack[winStack.length - 1];
            body._isMatList = true;
            renderList();
        });
    };
    function renderList() {
        if (!listWin) return;
        var body = listWin.querySelector('.fmw-body');
        body.innerHTML = '';
        var use = usageCounts();
        var names = definedNames();

        var wrap = el('div', 'fmw-cols');
        var left = el('div', 'fmw-listcol');
        left.appendChild(el('div', 'fmw-grouplabel', 'Materials'));
        var list = el('div', 'fmw-listbox');
        if (!names.length) list.appendChild(el('div', 'fmw-empty', '(정의된 재료 없음 — Add New)'));
        names.forEach(function (n) {
            var it = el('div', 'fmw-item', n + (use[n] ? '   (부재 ' + use[n] + ')' : '') + (custom[n] ? ' *' : ''));
            it.dataset.name = n;
            it.addEventListener('click', function () {
                list.querySelectorAll('.fmw-item').forEach(function (o) { o.classList.remove('sel'); });
                it.classList.add('sel');
            });
            it.addEventListener('dblclick', function () { openDetail(n); });
            list.appendChild(it);
        });
        left.appendChild(list);
        if (catalogErr) left.appendChild(el('div', 'fmw-warn', 'DB 폴백 모드: ' + catalogErr));

        var right = el('div', 'fmw-btncol');
        right.appendChild(el('div', 'fmw-grouplabel', 'Click to:'));
        right.appendChild(btn('Add New Material…', true, openAddNew));
        right.appendChild(btn('Add Copy of Material…', false, function () {
            var n = selName(list); if (!n) return alertNeedSel();
            var src = findMat(n); if (!src) return;
            var copy = JSON.parse(JSON.stringify(src));
            copy.name = uniqueName(n);
            openDetail(null, copy);
        }));
        right.appendChild(btn('Modify/Show Material…', false, function () {
            var n = selName(list); if (!n) return alertNeedSel();
            openDetail(n);
        }));
        var del = btn('Delete Material', false, function () {
            var n = selName(list); if (!n) return alertNeedSel();
            if (use[n]) { status('모델에서 사용 중이라 삭제할 수 없습니다 (' + n + ')', 'warning'); return; }
            delete custom[n]; delete defined[n];   // 목록에서 제거 (DB 카탈로그 원본은 유지)
            renderList();
        });
        right.appendChild(del);
        right.appendChild(el('div', 'fmw-sep'));
        var ids = selectedIds();
        var apply = btn('선택 부재에 적용 (' + ids.length + ')', false, function () {
            var n = selName(list); if (!n) return alertNeedSel();
            var stub = document.getElementById('bulk-material');
            if (stub) stub.value = n;
            window.bulkApplyMaterial();
            renderList();
        });
        if (!ids.length) apply.disabled = true;
        right.appendChild(apply);
        right.appendChild(el('div', 'fmw-sep'));
        right.appendChild(btn('OK', false, function () { closeWin(listWin); }));

        wrap.appendChild(left); wrap.appendChild(right);
        body.appendChild(wrap);
        body.appendChild(el('div', 'fmw-hint',
            '해석: 전 강종 E=205,000 MPa (KDS 14 31) — 강성 동일 반영 · 설계검토 fy는 현재 모델 대표 강종 기준'));

        function selName(l) { var s = l.querySelector('.fmw-item.sel'); return s ? s.dataset.name : null; }
        function alertNeedSel() { status('목록에서 재료를 먼저 선택하세요', 'warning'); }
    }
    function uniqueName(base) {
        var n = base + '-copy', i = 2;
        while (findMat(n)) { n = base + '-copy' + i; i++; }
        return n;
    }

    /* ---- ② Add New Material Property ----------------------------------------- */
    function openAddNew() {
        var body = openWin('Add New Material Property', 380);
        var win = winStack[winStack.length - 1];

        var standards = {};
        Object.keys(catalog || {}).forEach(function (n) {
            standards[catalog[n].standard || 'KS'] = 1;
        });
        var stdKeys = Object.keys(standards).sort();

        var region = mkSelect([['Korea', 'Korea']], 'Korea'); region.disabled = true;
        var type = mkSelect([
            ['Steel', 'Steel'],
            ['Concrete', 'Concrete (미지원 — 철골 전용)', true],
            ['Rebar', 'Rebar (미지원 — 철골 전용)', true],
            ['Other', 'Other (미지원)', true],
        ], 'Steel');
        var std = mkSelect(stdKeys.map(function (s) { return [s, s]; }), stdKeys[0]);
        var grade = mkSelect([], null);
        function refillGrades() {
            grade.innerHTML = '';
            Object.keys(catalog || {}).sort().forEach(function (n) {
                if ((catalog[n].standard || 'KS') !== std.value) return;
                var op = document.createElement('option');
                op.value = n; op.textContent = n;
                grade.appendChild(op);
            });
        }
        std.addEventListener('change', refillGrades);
        refillGrades();

        body.appendChild(row('Region', region));
        body.appendChild(row('Material Type', type));
        body.appendChild(row('Standard', std));
        body.appendChild(row('Grade', grade));

        var foot = el('div', 'fmw-foot');
        foot.appendChild(btn('OK', true, function () {
            var n = grade.value;
            if (!n) { status('Grade를 선택하세요', 'warning'); return; }
            closeWin(win);
            openDetail(n);
        }));
        foot.appendChild(btn('Cancel', false, function () { closeWin(win); }));
        body.appendChild(foot);
    }

    /* ---- ③ Material Property Data -------------------------------------------- */
    function openDetail(name, matOverride) {
        var src = matOverride || findMat(name);
        if (!src) { status('재료 정보를 찾을 수 없습니다: ' + name, 'warning'); return; }
        var mat = JSON.parse(JSON.stringify(src));    // 편집용 복사본
        var body = openWin('Material Property Data', 460);
        var win = winStack[winStack.length - 1];

        /* General */
        body.appendChild(el('div', 'fmw-grouplabel', 'General Data'));
        var nameIn = mkInput(mat.name, { text: true });
        body.appendChild(row('Material Name', nameIn));
        body.appendChild(row('Material Type', mkInput('Steel', { text: true, readonly: true })));
        body.appendChild(row('Directional Symmetry', mkInput('Isotropic', { text: true, readonly: true })));
        body.appendChild(row('Standard', mkInput(mat.standard || '-', { text: true, readonly: true })));

        /* Weight and Mass (연동) */
        body.appendChild(el('div', 'fmw-grouplabel', 'Material Weight and Mass'));
        var wIn = mkInput(mat.unit_weight_kN_m3, { step: '0.01' });
        var mIn = mkInput(mat.density_kg_m3, { step: '1' });
        var G0 = 9.80665;
        wIn.addEventListener('input', function () {
            var v = parseFloat(wIn.value);
            if (v > 0) mIn.value = Math.round(v * 1000 / G0);
        });
        mIn.addEventListener('input', function () {
            var v = parseFloat(mIn.value);
            if (v > 0) wIn.value = (v * G0 / 1000).toFixed(2);
        });
        body.appendChild(row('Weight per Unit Volume (kN/m³)', wIn));
        body.appendChild(row('Mass per Unit Volume (kg/m³)', mIn));

        /* Mechanical (G 자동) */
        body.appendChild(el('div', 'fmw-grouplabel', 'Mechanical Property Data'));
        var eIn = mkInput(mat.E_MPa, { step: '100' });
        var nuIn = mkInput(mat.poisson, { step: '0.01' });
        var gIn = mkInput(mat.G_MPa, { readonly: true });
        function recalcG() {
            var E = parseFloat(eIn.value), nu = parseFloat(nuIn.value);
            gIn.value = (E > 0 && nu > -1) ? Math.round(E / (2 * (1 + nu)) * 10) / 10 : '';
        }
        eIn.addEventListener('input', recalcG);
        nuIn.addEventListener('input', recalcG);
        body.appendChild(row('Modulus of Elasticity, E (MPa)', eIn));
        body.appendChild(row("Poisson's Ratio, ν", nuIn));
        body.appendChild(row('Shear Modulus, G (MPa) [자동]', gIn));
        body.appendChild(row('Thermal Coeff, α (1/°C)', mkInput(mat.thermal_coeff_per_C, { readonly: true })));

        /* Design (KDS/KS) — 두께구간별 fy */
        body.appendChild(el('div', 'fmw-grouplabel', 'Design Property Data (KS/KDS 14 31)'));
        var tbl = el('table', 'fmw-fytable');
        var thead = el('tr');
        ['두께구간 (mm)', 'Fy (MPa)'].forEach(function (h) { thead.appendChild(el('th', null, h)); });
        tbl.appendChild(thead);
        var fyInputs = [];
        (mat.fy_by_thickness || []).forEach(function (b) {
            var tr = el('tr');
            var rng = (b.t_min != null ? b.t_min : 0) + ' < t ≤ ' + ((b.t_max != null && b.t_max < 900) ? b.t_max : '∞');
            tr.appendChild(el('td', null, rng));
            var td = el('td');
            var fi = mkInput(b.fy, { step: '5' });
            fyInputs.push(fi);
            td.appendChild(fi);
            tr.appendChild(td);
            tbl.appendChild(tr);
        });
        body.appendChild(tbl);
        var fuIn = mkInput(mat.fu_MPa, { step: '5' });
        body.appendChild(row('Tensile Strength, Fu (MPa)', fuIn));
        if (mat.fu_max_MPa) body.appendChild(row('Fu 상한 (MPa)', mkInput(mat.fu_max_MPa, { readonly: true })));
        if (mat.elongation_min_pct != null) body.appendChild(row('연신율 최소 (%)', mkInput(mat.elongation_min_pct, { readonly: true })));

        var foot = el('div', 'fmw-foot');
        foot.appendChild(btn('OK', true, function () {
            var newName = (nameIn.value || '').trim();
            var E = parseFloat(eIn.value), nu = parseFloat(nuIn.value);
            var uw = parseFloat(wIn.value), dens = parseFloat(mIn.value), fu = parseFloat(fuIn.value);
            if (!newName) { status('재료 이름을 입력하세요', 'warning'); return; }
            if (!(E > 0) || !(nu > 0 && nu < 0.5) || !(uw > 0)) { status('E/ν/단위중량 값을 확인하세요', 'warning'); return; }
            mat.name = newName;
            mat.E_MPa = E; mat.poisson = nu;
            mat.G_MPa = Math.round(E / (2 * (1 + nu)) * 10) / 10;
            mat.unit_weight_kN_m3 = uw; mat.density_kg_m3 = dens;
            mat.fu_MPa = isNaN(fu) ? mat.fu_MPa : fu;
            var bad = false;
            (mat.fy_by_thickness || []).forEach(function (b, i) {
                var v = parseFloat(fyInputs[i].value);
                if (!(v > 0)) bad = true; else b.fy = v;
            });
            if (bad) { status('Fy 값을 확인하세요', 'warning'); return; }
            // OK = '정의된 재료' 등록. 값이 카탈로그와 다를 때만 custom으로 가림.
            defined[mat.name] = 1;
            var catRef = catalog && catalog[mat.name];
            if (!catRef || JSON.stringify(catRef) !== JSON.stringify(mat)) custom[mat.name] = mat;
            closeWin(win);
            renderList();
            status('재료 저장: ' + mat.name + (custom[mat.name] ? ' (세션 커스텀)' : ''));
        }));
        foot.appendChild(btn('Cancel', false, function () { closeWin(win); }));
        body.appendChild(foot);
    }

    /* ---- init: 카탈로그 선로딩 (리본 M 버튼은 HTML onclick으로 진입) ----------- */
    function init() { fetchCatalog(); }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
