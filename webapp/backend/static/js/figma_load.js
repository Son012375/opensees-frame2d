/* ============================================================================
 * figma_load.js  —  하중 정의 팝업 D/L/E/W (Figma UI Lab)
 * ----------------------------------------------------------------------------
 * ETABS의 수직하중(Define > Shell Uniform Load Sets, 강의자료 p17) +
 * 횡하중(Define > Load Patterns > Auto Lateral Load KDS, p18~20) 흐름을
 * 층/환경 파라미터 팝업 4종으로 재현:
 *   D (Dead)    층별 슬래브 두께 t + 마감 DL(kN/m²)   → stories[].slab_thickness/dead_load_finish
 *   L (Live)    층별 용도 → KDS 41 12 00 활하중 자동    → stories[].usage
 *   E (Seismic) 지역·지반등급·중요도·횡력저항시스템·ELF/RSA → region/site_class/importance/seismic_system/seismic_method
 *   W (Wind)    지역·노풍도                             → region/exposure_category
 *
 * 반영 메커니즘 (엔진 JS 무수정): 모든 해석 경로(직접입력 V1폼·NL·IFC·DRAW)가
 * 최종적으로 fetch('/api/v2/analyze', {config})를 부르므로, window.fetch를
 * 래핑해 요청 직전에 팝업 정의값을 config에 병합한다 — "팝업에서 본 값이
 * 곧 해석에 쓰이는 값" 단일 관문. 추가로 엔진 자체 UI(직접입력 story row,
 * IFC 위저드 select, _v2Model 메타)에도 best-effort로 write-through 해서
 * 레거시 화면과 값이 어긋나 보이지 않게 한다.
 *
 * 활하중 표시값은 신규 GET /api/loads/live-preview — 해석 시 load_generator가
 * 조회하는 것과 동일 경로라 "표시값 == 적용값"이 보장된다.
 * 창 스타일은 Material/Section 팝업의 .fig-mat-win / .fmw-* 룩 공유.
 * ========================================================================== */
(function () {
    'use strict';

    /* ---- 세션 하중 정의 상태 (fetch 병합의 단일 source of truth) ------------- */
    var ov = { env: {}, stories: {} };   // env: {region,...}, stories: {1:{usage,slab_thickness,dead_load_finish}}
    window._figmaLoadOverrides = ov;     // 디버그/테스트 노출

    var livePreview = null;              // /api/loads/live-preview 캐시 (usage -> {value, source})
    var livePreviewErr = null;

    /* ---- engine globals (top-level let/const — bare 이름 접근, 가드) --------- */
    function getModel() { try { return window._v2Model || null; } catch (_) { return null; } }
    function getIfcData() { try { return (typeof ifcEditedData !== 'undefined') ? ifcEditedData : null; } catch (_) { return null; } }
    function getModelSource() { try { return (typeof modelSource !== 'undefined' && modelSource) || ''; } catch (_) { return ''; } }
    function isIFCContext() {
        var d = getIfcData();
        return getModelSource().indexOf('IFC') === 0 && !!(d && Array.isArray(d.stories));
    }
    function usageOptions() {
        try { if (typeof USAGE_OPTIONS !== 'undefined' && USAGE_OPTIONS.length) return USAGE_OPTIONS; } catch (_) {}
        return [
            { value: 'office', label: '사무실 (Office)' }, { value: 'retail', label: '소매점 (Retail)' },
            { value: 'residential', label: '주거 (Residential)' }, { value: 'parking', label: '주차장 (Parking)' },
            { value: 'storage', label: '창고 (Storage)' }, { value: 'hospital', label: '병원 (Hospital)' },
            { value: 'school', label: '학교 (School)' }, { value: 'assembly', label: '집회 (Assembly)' },
            { value: 'corridor', label: '복도 (Corridor)' }, { value: 'mechanical_room', label: '기계실 (Mech.)' },
            { value: 'roof', label: '옥상 (Roof)' },
        ];
    }
    function usageLabel(v) {
        var hit = usageOptions().filter(function (o) { return o.value === v; })[0];
        return hit ? hit.label : v;
    }
    function status(msg, kind) { try { if (typeof setStatus === 'function') setStatus(msg, kind || 'success'); } catch (_) {} }

    /* ---- 현재 컨텍스트에서 층별 하중행 도출 (엔진 runAnalysisV2와 동일 우선순위) */
    function baseStories() {
        var out = [];
        var d = getIfcData();
        if (isIFCContext()) {
            var sels = document.querySelectorAll('.ifc-usage-sel');
            d.stories.forEach(function (_, i) {
                out.push({ story: i + 1, usage: (sels[i] && sels[i].value) || 'office',
                           slab_thickness: 0.15, dead_load_finish: 1.0 });
            });
            return out;
        }
        var m = getModel();
        if (m && m.story_elevations && m.story_elevations.length > 1) {
            var su = m.story_usages || {}, st = m.story_slab_thickness || {}, sdl = m.story_dead_load_finish || {};
            for (var i = 0; i < m.story_elevations.length - 1; i++) {
                var sn = i + 1;
                out.push({
                    story: sn,
                    usage: su[sn] != null ? su[sn] : (su[String(sn)] != null ? su[String(sn)] : 'office'),
                    slab_thickness: st[sn] != null ? st[sn] : (st[String(sn)] != null ? st[String(sn)] : 0.15),
                    dead_load_finish: sdl[sn] != null ? sdl[sn] : (sdl[String(sn)] != null ? sdl[String(sn)] : 1.0),
                });
            }
            return out;
        }
        // 직접입력 폼 story rows (모델 미생성 시에도 기본 3층 행이 존재)
        document.querySelectorAll('#story-list-container .story-row').forEach(function (row, i) {
            var us = row.querySelector('.usage-select');
            out.push({ story: i + 1, usage: (us && us.value) || 'office',
                       slab_thickness: 0.15, dead_load_finish: 1.0 });
        });
        return out;
    }
    function currentStories() {
        return baseStories().map(function (s) {
            var o = ov.stories[s.story] || {};
            return {
                story: s.story,
                usage: o.usage || s.usage,
                slab_thickness: (o.slab_thickness != null) ? o.slab_thickness : s.slab_thickness,
                dead_load_finish: (o.dead_load_finish != null) ? o.dead_load_finish : s.dead_load_finish,
            };
        });
    }

    /* ---- 환경 파라미터 도출 (overrides > model.environment > DOM > 기본값) ---- */
    function domVal(id) { var e = document.getElementById(id); return (e && e.value) || null; }
    function baseEnv() {
        var envM = (getModel() && getModel().environment) || {};
        var ifc = isIFCContext();
        function pick(modelKey, ifcId, inputId, fallback) {
            if (ifc) return domVal(ifcId) || envM[modelKey] || fallback;
            return envM[modelKey] || domVal(inputId) || fallback;
        }
        return {
            region: pick('region', 'ifc-region', 'input-region', '서울'),
            importance: pick('importance', 'ifc-importance', 'input-importance', 'II'),
            site_class: envM.site_class || 'S3',
            seismic_system: envM.seismic_system || 'ordinary_moment_frame',
            exposure_category: envM.exposure_category || 'B',
            seismic_method: (ifc ? domVal('ifc-seismic-method') : domVal('input-seismic-method')) || 'ELF',
            rsa_combination: (ifc ? domVal('ifc-rsa-combination') : domVal('input-rsa-combination')) || 'CQC',
            rsa_direction: (ifc ? domVal('ifc-rsa-direction') : domVal('input-rsa-direction')) || '30pct',
        };
    }
    function currentEnv() {
        var e = baseEnv();
        Object.keys(ov.env).forEach(function (k) { e[k] = ov.env[k]; });
        return e;
    }

    /* ---- write-through: 팝업 적용값을 레거시 UI/모델 메타에도 반영 ------------ */
    function setDom(id, v) { var e = document.getElementById(id); if (e && v != null) e.value = v; }
    function writeThroughEnv(env) {
        setDom('input-region', env.region); setDom('ifc-region', env.region);
        setDom('input-importance', env.importance); setDom('ifc-importance', env.importance);
        setDom('input-seismic-method', env.seismic_method); setDom('ifc-seismic-method', env.seismic_method);
        setDom('input-rsa-combination', env.rsa_combination); setDom('ifc-rsa-combination', env.rsa_combination);
        setDom('input-rsa-direction', env.rsa_direction); setDom('ifc-rsa-direction', env.rsa_direction);
        // RSA 옵션 표시 토글 (직접입력 폼의 inline onchange와 동일 동작)
        var rsa = document.getElementById('rsa-options');
        if (rsa) rsa.style.display = env.seismic_method === 'RSA' ? 'block' : 'none';
        var m = getModel();
        if (m) {
            m.environment = m.environment || {};
            ['region', 'importance', 'site_class', 'seismic_system', 'exposure_category'].forEach(function (k) {
                if (env[k] != null) m.environment[k] = env[k];
            });
        }
    }
    function writeThroughStories(stories) {
        var m = getModel();
        if (m) {
            m.story_usages = m.story_usages || {};
            m.story_slab_thickness = m.story_slab_thickness || {};
            m.story_dead_load_finish = m.story_dead_load_finish || {};
            stories.forEach(function (s) {
                m.story_usages[s.story] = s.usage;
                m.story_slab_thickness[s.story] = s.slab_thickness;
                m.story_dead_load_finish[s.story] = s.dead_load_finish;
            });
        }
        // 직접입력 폼 story rows + IFC 위저드 select의 용도 동기화
        var rows = document.querySelectorAll('#story-list-container .story-row');
        var sels = document.querySelectorAll('.ifc-usage-sel');
        stories.forEach(function (s, i) {
            var us = rows[i] && rows[i].querySelector('.usage-select');
            if (us) us.value = s.usage;
            if (sels[i]) sels[i].value = s.usage;
        });
    }

    /* ---- fetch 병합: 모든 해석 요청의 단일 관문 ------------------------------ */
    function mergeConfig(config) {
        if (!config || typeof config !== 'object') return;
        Object.keys(ov.env).forEach(function (k) { config[k] = ov.env[k]; });
        if (Array.isArray(config.stories)) {
            config.stories.forEach(function (s, i) {
                var num = s.story || (i + 1);
                var o = ov.stories[num];
                if (!o) return;
                s.story = num;   // V1 폼 엔트리({height,usage})에 story 번호 부여 → 백엔드가 slab/finish 인식
                if (o.usage) s.usage = o.usage;
                if (o.slab_thickness != null) s.slab_thickness = o.slab_thickness;
                if (o.dead_load_finish != null) s.dead_load_finish = o.dead_load_finish;
            });
        }
    }
    function wrapFetch() {
        if (window.fetch._figmaLoadMerge) return;
        var of = window.fetch;
        window.fetch = function (url, opts) {
            try {
                if (typeof url === 'string' && url.indexOf('/api/v2/analyze') === 0 &&
                    opts && typeof opts.body === 'string' &&
                    (Object.keys(ov.env).length || Object.keys(ov.stories).length)) {
                    var body = JSON.parse(opts.body);
                    if (body && body.config) {
                        mergeConfig(body.config);
                        opts = Object.assign({}, opts, { body: JSON.stringify(body) });
                    }
                }
            } catch (e) { console.warn('[figma-load] config merge 실패(원본 전송)', e); }
            return of.call(this, url, opts);
        };
        window.fetch._figmaLoadMerge = true;
    }

    /* ---- 활하중 미리보기 (표시값 == 적용값 보장 경로) ------------------------ */
    function ensureLivePreview(cb) {
        if (livePreview) { cb(); return; }
        fetch('/api/loads/live-preview').then(function (r) { return r.json(); }).then(function (d) {
            livePreview = d.live_loads || {};
            livePreviewErr = d.fallback ? (d.error || 'fallback') : null;
            cb();
        }).catch(function (e) { livePreview = {}; livePreviewErr = String(e); cb(); });
    }
    function liveValueText(usage) {
        if (!livePreview) return '…';
        var rec = livePreview[usage];
        if (!rec || rec.value == null) return '자동';
        return Number(rec.value).toFixed(2);
    }

    /* ---- window chrome (Material 팝업 룩 공유) ------------------------------- */
    var overlay = null, winStack = [];
    function ensureOverlay() {
        if (overlay) return;
        overlay = document.createElement('div');
        overlay.id = 'fig-load-overlay';
        document.body.appendChild(overlay);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && winStack.length) { e.stopPropagation(); closeTop(); }
        }, true);
    }
    function openWin(title, width) {
        ensureOverlay();
        overlay.classList.add('open');
        var w = document.createElement('div');
        w.className = 'fig-mat-win fig-load-win';
        w.style.width = (width || 460) + 'px';
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
        w.remove();
        if (!winStack.length) overlay.classList.remove('open');
    }
    function closeTop() { if (winStack.length) closeWin(winStack[winStack.length - 1]); }
    function winOf(node) { for (var p = node; p; p = p.parentElement) if (p.classList && p.classList.contains('fig-mat-win')) return p; return null; }

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
    function mkSelect(options, current) {
        var s = document.createElement('select');
        options.forEach(function (o) {
            var op = document.createElement('option');
            op.value = o[0]; op.textContent = o[1];
            if (o[0] === current) op.selected = true;
            s.appendChild(op);
        });
        return s;
    }
    function mkNum(value, step, min, max) {
        var i = document.createElement('input');
        i.type = 'number'; i.step = step; i.value = value;
        if (min != null) i.min = min;
        if (max != null) i.max = max;
        return i;
    }
    function btn(label, primary, onclick) {
        var b = el('button', 'fmw-btn' + (primary ? ' primary' : ''), label);
        b.type = 'button';
        b.addEventListener('click', onclick);
        return b;
    }
    function foot(body, okLabel, onOk) {
        var f = el('div', 'fmw-foot');
        f.appendChild(btn(okLabel || 'OK', true, function (e) {
            var w = winOf(e.target);
            if (onOk() !== false && w) closeWin(w);
        }));
        f.appendChild(btn('취소', false, function (e) { var w = winOf(e.target); if (w) closeWin(w); }));
        body.appendChild(f);
    }
    function reanalyzeHintText() {
        var has = false;
        try { has = !!currentResult; } catch (_) {}
        return has ? ' · 기존 결과에 반영하려면 Run으로 재해석' : '';
    }

    /* ---- 지역 datalist (자유 입력 허용 — DB는 229 시군구) --------------------- */
    var REGIONS = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
        '수원', '성남', '고양', '용인', '창원', '청주', '전주', '천안', '포항', '강릉', '제주'];
    function mkRegionInput(current) {
        var wrap = document.createElement('div');
        var i = document.createElement('input');
        i.type = 'text'; i.value = current || '서울'; i.setAttribute('list', 'fig-load-regions');
        if (!document.getElementById('fig-load-regions')) {
            var dl = document.createElement('datalist');
            dl.id = 'fig-load-regions';
            REGIONS.forEach(function (r) { var o = document.createElement('option'); o.value = r; dl.appendChild(o); });
            document.body.appendChild(dl);
        }
        wrap.appendChild(i);
        wrap._input = i;
        return wrap;
    }

    /* ---- D: 고정하중 (수직 — Shell Uniform Dead) ------------------------------ */
    function openDead() {
        var stories = currentStories();
        if (!stories.length) { status('층 정보가 없습니다 — 먼저 모델을 생성/가져오세요', 'warning'); return; }
        var body = openWin('고정하중 정의 — Dead (수직하중)', 560);

        body.appendChild(el('div', 'fmw-grouplabel', '층별 바닥 고정하중 (Shell Uniform — Dead)'));
        var tbl = el('table', 'fmw-fytable flw-table');
        var thead = el('thead'); var hr = el('tr');
        ['층', '용도', '슬래브 t (m)', '슬래브 자중 (kN/m²)', '마감 DL (kN/m²)'].forEach(function (h) { hr.appendChild(el('th', null, h)); });
        thead.appendChild(hr); tbl.appendChild(thead);
        var tbody = el('tbody');
        var rowsUI = [];
        stories.forEach(function (s) {
            var tr = el('tr');
            tr.appendChild(el('td', 'flw-c', String(s.story) + 'F'));
            tr.appendChild(el('td', null, usageLabel(s.usage)));
            var tIn = mkNum(s.slab_thickness, '0.01', 0.05, 0.5);
            var selfTd = el('td', 'flw-ro', (24.0 * s.slab_thickness).toFixed(2));
            tIn.addEventListener('input', function () {
                var v = parseFloat(tIn.value);
                selfTd.textContent = isNaN(v) ? '—' : (24.0 * v).toFixed(2);
            });
            var fIn = mkNum(s.dead_load_finish, '0.1', 0, 20);
            var td1 = el('td'); td1.appendChild(tIn); tr.appendChild(td1);
            tr.appendChild(selfTd);
            var td2 = el('td'); td2.appendChild(fIn); tr.appendChild(td2);
            tbody.appendChild(tr);
            rowsUI.push({ story: s.story, tIn: tIn, fIn: fIn });
        });
        tbl.appendChild(tbody);
        body.appendChild(tbl);

        // 전층 일괄 입력
        var bulk = el('div', 'flw-bulk');
        var bt = mkNum('', '0.01', 0.05, 0.5); bt.placeholder = 't (m)';
        var bf = mkNum('', '0.1', 0, 20); bf.placeholder = '마감 DL';
        bulk.appendChild(el('span', null, '전층 일괄:'));
        bulk.appendChild(bt); bulk.appendChild(bf);
        bulk.appendChild(btn('적용', false, function () {
            rowsUI.forEach(function (r) {
                if (bt.value !== '') { r.tIn.value = bt.value; r.tIn.dispatchEvent(new Event('input')); }
                if (bf.value !== '') r.fIn.value = bf.value;
            });
        }));
        body.appendChild(bulk);

        body.appendChild(el('div', 'fmw-hint',
            'Note: Loads are in the gravity direction. 슬래브 자중 = 24.0 kN/m³ × t 자동 산정, ' +
            '부재(보·기둥) 자중은 단면·재료에서 자동. 마감 DL은 바닥면적당 kN/m².'));

        foot(body, '적용 (OK)', function () {
            var bad = false;
            rowsUI.forEach(function (r) {
                var t = parseFloat(r.tIn.value), f = parseFloat(r.fIn.value);
                if (isNaN(t) || t <= 0 || isNaN(f) || f < 0) { bad = true; return; }
                var o = ov.stories[r.story] = ov.stories[r.story] || {};
                o.slab_thickness = t;
                o.dead_load_finish = f;
            });
            if (bad) { alert('슬래브 두께(>0)와 마감 DL(≥0)을 확인해주세요.'); return false; }
            writeThroughStories(currentStories());
            status('고정하중 정의 적용 — 다음 해석부터 반영' + reanalyzeHintText());
        });
    }

    /* ---- L: 활하중 (용도별 KDS 자동) ------------------------------------------ */
    function openLive() {
        var stories = currentStories();
        if (!stories.length) { status('층 정보가 없습니다 — 먼저 모델을 생성/가져오세요', 'warning'); return; }
        var body = openWin('활하중 정의 — Live (KDS 41 12 00)', 520);

        body.appendChild(el('div', 'fmw-grouplabel', '층별 용도 → 활하중 (표 3.2-1 자동 조회)'));
        var tbl = el('table', 'fmw-fytable flw-table');
        var thead = el('thead'); var hr = el('tr');
        ['층', '용도', '활하중 (kN/m²)'].forEach(function (h) { hr.appendChild(el('th', null, h)); });
        thead.appendChild(hr); tbl.appendChild(thead);
        var tbody = el('tbody');
        var rowsUI = [];
        var opts = usageOptions().map(function (o) { return [o.value, o.label]; });
        stories.forEach(function (s) {
            var tr = el('tr');
            tr.appendChild(el('td', 'flw-c', String(s.story) + 'F'));
            var sel = mkSelect(opts, s.usage);
            var vTd = el('td', 'flw-ro flw-c', liveValueText(s.usage));
            sel.addEventListener('change', function () { vTd.textContent = liveValueText(sel.value); });
            var td = el('td'); td.appendChild(sel); tr.appendChild(td);
            tr.appendChild(vTd);
            tbody.appendChild(tr);
            rowsUI.push({ story: s.story, sel: sel, vTd: vTd });
        });
        tbl.appendChild(tbody);
        body.appendChild(tbl);

        var hint = el('div', 'fmw-hint',
            '활하중 값은 용도로부터 KDS 41 12 00 표 3.2-1(하중 DB)에서 자동 조회됩니다 — ' +
            '위 표시값이 해석에 그대로 적용됩니다. 창고 등 중(重)활하중은 지진질량 산정 비율도 자동 반영.');
        body.appendChild(hint);
        if (livePreviewErr) body.appendChild(el('div', 'fmw-warn', '⚠ DB 미리보기 실패(' + livePreviewErr + ') — 해석 시 서버 폴백값 사용'));

        foot(body, '적용 (OK)', function () {
            rowsUI.forEach(function (r) {
                var o = ov.stories[r.story] = ov.stories[r.story] || {};
                o.usage = r.sel.value;
            });
            writeThroughStories(currentStories());
            status('활하중(용도) 정의 적용 — 다음 해석부터 반영' + reanalyzeHintText());
        });

        // 값 채우기 (비동기 — 창은 즉시 열림)
        ensureLivePreview(function () {
            rowsUI.forEach(function (r) { r.vTd.textContent = liveValueText(r.sel.value); });
        });
    }

    /* ---- E: 지진하중 (KDS 41 17 00) ------------------------------------------- */
    var SEISMIC_SYSTEMS = [
        ['ordinary_moment_frame', '철골 보통모멘트골조'],
        ['intermediate_moment_frame', '철골 중간모멘트골조'],
        ['special_moment_frame', '철골 특수모멘트골조'],
        ['ordinary_braced_frame', '철골 보통중심가새골조'],
        ['special_braced_frame', '철골 특수중심가새골조'],
        ['rc_ordinary_moment_frame', 'RC 보통모멘트골조'],
        ['rc_intermediate_moment_frame', 'RC 중간모멘트골조'],
        ['rc_special_moment_frame', 'RC 특수모멘트골조'],
        ['rc_ordinary_shear_wall', 'RC 보통전단벽'],
        ['rc_special_shear_wall', 'RC 특수전단벽'],
    ];
    var SITE_CLASSES = [
        ['S1', 'S1 — 암반 지반'],
        ['S2', 'S2 — 얕고 단단한 지반'],
        ['S3', 'S3 — 얕고 연약한 지반'],
        ['S4', 'S4 — 깊고 단단한 지반'],
        ['S5', 'S5 — 깊고 연약한 지반'],
    ];
    var IMPORTANCE = [['특', '특 (IE=1.5)'], ['I', 'I (IE=1.2)'], ['II', 'II (IE=1.0)']];

    function openSeismic() {
        var env = currentEnv();
        var body = openWin('지진하중 정의 — Seismic (KDS 41 17 00)', 470);

        body.appendChild(el('div', 'fmw-grouplabel', '지역 · 지반 (설계응답스펙트럼)'));
        var regionIn = mkRegionInput(env.region);
        body.appendChild(row('지역 (시·군·구)', regionIn));
        var siteSel = mkSelect(SITE_CLASSES, env.site_class);
        body.appendChild(row('지반등급 (Site Class)', siteSel));
        var impSel = mkSelect(IMPORTANCE, env.importance);
        body.appendChild(row('중요도 (Importance)', impSel));

        body.appendChild(el('div', 'fmw-grouplabel', '횡력저항시스템 (R · Ω₀ · Cd 자동)'));
        var sysSel = mkSelect(SEISMIC_SYSTEMS, env.seismic_system);
        body.appendChild(row('시스템', sysSel));

        body.appendChild(el('div', 'fmw-grouplabel', '해석 방법'));
        var methodSel = mkSelect([['ELF', 'ELF — 등가정적해석'], ['RSA', 'RSA — 응답스펙트럼해석']], env.seismic_method);
        body.appendChild(row('Seismic Method', methodSel));
        var comboSel = mkSelect([['CQC', 'CQC'], ['SRSS', 'SRSS']], env.rsa_combination);
        var dirSel = mkSelect([['30pct', '30% Rule (KDS)'], ['SRSS', 'SRSS']], env.rsa_direction);
        var rsaRows = [row('모드 조합', comboSel), row('방향 조합', dirSel)];
        rsaRows.forEach(function (r) { body.appendChild(r); });
        function syncRsa() {
            rsaRows.forEach(function (r) { r.style.display = methodSel.value === 'RSA' ? '' : 'none'; });
        }
        methodSel.addEventListener('change', syncRsa);
        syncRsa();

        body.appendChild(el('div', 'fmw-hint',
            '지역 → 유효지반가속도 S(위험도 DB, 229 시·군·구), 지반등급 → Fa·Fv, ' +
            'S_DS·S_D1·설계스펙트럼(KDS 17 10 00)과 R·Ω₀·Cd는 해석 시 자동 산정됩니다. ' +
            '지역·중요도는 풍하중 정의와 공유됩니다.'));

        foot(body, '적용 (OK)', function () {
            var region = (regionIn._input.value || '').trim();
            if (!region) { alert('지역을 입력해주세요.'); return false; }
            ov.env.region = region;
            ov.env.site_class = siteSel.value;
            ov.env.importance = impSel.value;
            ov.env.seismic_system = sysSel.value;
            ov.env.seismic_method = methodSel.value;
            ov.env.rsa_combination = comboSel.value;
            ov.env.rsa_direction = dirSel.value;
            writeThroughEnv(currentEnv());
            status('지진하중 정의 적용 — 다음 해석부터 반영' + reanalyzeHintText());
        });
    }

    /* ---- W: 풍하중 (KDS 41 12 00 §5) ------------------------------------------ */
    var EXPOSURES = [
        ['A', 'A — 대도시 도심 (고층 밀집)'],
        ['B', 'B — 시가지 (중·저층 밀집)'],
        ['C', 'C — 개활지 (드문 장애물)'],
        ['D', 'D — 해안 · 평탄 개활지'],
    ];
    function openWind() {
        var env = currentEnv();
        var body = openWin('풍하중 정의 — Wind (KDS 41 12 00 §5)', 470);

        body.appendChild(el('div', 'fmw-grouplabel', 'Wind Load Parameters'));
        var regionIn = mkRegionInput(env.region);
        body.appendChild(row('지역 (→ 기본풍속 V₀ 자동)', regionIn));
        var expSel = mkSelect(EXPOSURES, env.exposure_category);
        body.appendChild(row('노풍도 (Exposure Category)', expSel));
        var impSel = mkSelect(IMPORTANCE, env.importance);
        body.appendChild(row('중요도 (Importance)', impSel));

        body.appendChild(el('div', 'fmw-hint',
            '기본풍속 V₀는 지역으로부터 위험도 DB에서 자동 조회되고, 풍속고도분포계수 Kzr(§5.5)·' +
            '가스트영향계수·풍력계수는 KDS 산식으로 자동 산정됩니다. 지역·중요도는 지진하중 정의와 공유됩니다.'));

        foot(body, '적용 (OK)', function () {
            var region = (regionIn._input.value || '').trim();
            if (!region) { alert('지역을 입력해주세요.'); return false; }
            ov.env.region = region;
            ov.env.exposure_category = expSel.value;
            ov.env.importance = impSel.value;
            writeThroughEnv(currentEnv());
            status('풍하중 정의 적용 — 다음 해석부터 반영' + reanalyzeHintText());
        });
    }

    /* ---- 진입점 -------------------------------------------------------------- */
    window.figmaOpenLoads = function (kind) {
        if (kind === 'dead') openDead();
        else if (kind === 'live') openLive();
        else if (kind === 'seismic') openSeismic();
        else if (kind === 'wind') openWind();
    };

    /* ---- 테스트 훅 ------------------------------------------------------------ */
    window._figmaLoadView = function () {
        return {
            overrides: JSON.parse(JSON.stringify(ov)),
            stories: currentStories(),
            env: currentEnv(),
        };
    };

    wrapFetch();
})();
