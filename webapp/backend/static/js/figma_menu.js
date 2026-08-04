/* ============================================================================
 * figma_menu.js  —  Top menu-bar dropdowns for the Figma UI Lab (/editor-figma)
 * ----------------------------------------------------------------------------
 * Turns the static top menu strip (File / Edit / View / ...) into click-to-open
 * dropdowns whose items call the existing engine functions, organised per the
 * Tier-1 wiring map (docs/figma_ui_lab/tier1_wiring_map.md).
 *
 * Isolation: this file only READS/CALLS engine globals; it never edits the
 * engine JS. Loaded after editor3d_figma.js / figma_edit.js / figma_tools.js.
 *
 * Status conventions per item:
 *   run            -> live: calls an existing engine function (try/caught)
 *   toggle         -> flips a (hidden) checkbox then calls the handler; shows ✓
 *   enabled: fn    -> greyed out until fn() is true (e.g. needs analysis result)
 *   disabled+note  -> backlog / needs DOM porting; greyed with a tooltip
 * ========================================================================== */
(function () {
    'use strict';

    /* ---- safe access to engine globals (top-level `let`s in editor3d_figma.js) */
    function hasResult() { try { return !!currentResult; } catch (_) { return false; } }
    function hasJob() { try { return !!currentJobId; } catch (_) { return false; } }
    function reportUrl() { try { return currentResult && currentResult.report_url; } catch (_) { return null; } }
    function curEditMode() { try { return editMode; } catch (_) { return null; } }
    function call(fnName, ...args) {
        var fn = window[fnName];
        if (typeof fn !== 'function') throw new Error(fnName + '() 를 찾을 수 없습니다');
        return fn.apply(window, args);
    }
    /* engine top-level `let modelSource` — assignable by bare name from this
     * later classic script; guarded in case the engine binding disappears. */
    function setModelSource(v) {
        try { modelSource = v; } catch (_) { console.warn('[figma-menu] modelSource binding 없음'); }
    }

    /* ---- hidden checkbox backing store for toggle handlers that read #toggle-* */
    function ensureCheckbox(id) {
        var cb = document.getElementById(id);
        if (!cb) {
            cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.id = id;
            cb.style.display = 'none';
            document.body.appendChild(cb);
        }
        return cb;
    }
    function cbChecked(id) { var c = document.getElementById(id); return !!(c && c.checked); }

    /* ---- input drawer -----------------------------------------------------
     * The Figma layout hides the whole model-input UI (.legacy-config-ui:
     * 직접 입력 / 자연어 / IFC wizard) with display:none!important, replacing
     * the left panel with the Model Explorer. So switchInputTab() alone shows
     * nothing. We surface that hidden UI as a floating drawer on demand.
     * Moving the nodes preserves their ids + inline handlers; stripping the
     * `legacy-config-ui` class lets their native styles render again.        */
    var inputDrawerBuilt = false;
    var drawerEl = null, drawerTitleEl = null;
    var DRAWER_TITLES = {
        manual: '직접 입력 — 모델 정의',
        nl: '자연어 입력 — 모델 · 하중 자동 생성',
        ifc: 'IFC 가져오기',
    };
    function ensureInputDrawer() {
        if (inputDrawerBuilt) return;
        var overlay = document.createElement('div');
        overlay.id = 'figma-input-overlay';
        overlay.addEventListener('mousedown', function (e) { if (e.target === overlay) closeInputDrawer(); });

        var drawer = document.createElement('div');
        drawer.id = 'figma-input-drawer';
        var head = document.createElement('div');
        head.className = 'fid-head';
        var title = document.createElement('span');
        title.textContent = '모델 · 하중 입력';
        drawerEl = drawer;
        drawerTitleEl = title;
        var close = document.createElement('button');
        close.className = 'fid-close';
        close.textContent = '✕';
        close.addEventListener('click', closeInputDrawer);
        head.appendChild(title);
        head.appendChild(close);

        var body = document.createElement('div');
        body.className = 'fid-body';
        // relocate the hidden config UI + drop the hide-triggering class
        document.querySelectorAll('.legacy-config-ui').forEach(function (el) {
            el.classList.remove('legacy-config-ui');
            body.appendChild(el);
        });

        drawer.appendChild(head);
        drawer.appendChild(body);
        overlay.appendChild(drawer);
        document.body.appendChild(overlay);
        retargetImportButtons();
        inputDrawerBuilt = true;
    }

    /* Import = load & DISPLAY the model only; analysis is the top Run button.
     * Manual/NL terminal "해석 실행" buttons are repointed to preview-only.   */
    function retargetImportButtons() {
        var mBtn = document.querySelector('#tab-manual button[onclick*="runAnalysis"]');
        if (mBtn && !mBtn._figmaImport) {
            mBtn._figmaImport = true;
            mBtn.onclick = function (e) {
                if (e) e.preventDefault();
                // 이전 IFC import의 stale 마커가 Run을 V2(IFC) 경로로
                // 가로채지 않도록 소스를 명시적으로 되돌린다.
                setModelSource('Manual');
                try { call('switchInputTab', 'manual'); call('updateManualPreview'); }
                catch (err) { console.warn('[figma-menu] manual preview', err); }
                closeInputDrawer();
                toast('모델을 뷰에 표시했습니다 · 해석은 Run 버튼');
            };
            mBtn.textContent = '모델 생성 · 뷰에 표시';
        }
        var nBtn = document.querySelector('#tab-nl button[onclick*="runAnalysisFromNL"]');
        if (nBtn && !nBtn._figmaImport) {
            nBtn._figmaImport = true;
            nBtn.onclick = function (e) {
                if (e) e.preventDefault();
                setModelSource('NL');
                try { call('applyResolvedConfig'); } catch (err) { console.warn('[figma-menu] applyResolvedConfig', err); }
                try { call('switchInputTab', 'manual'); call('updateManualPreview'); } catch (err) {}
                closeInputDrawer();
                toast('모델을 뷰에 표시했습니다 · 해석은 Run 버튼');
            };
            nBtn.textContent = '모델 생성 · 뷰에 표시';
        }
    }
    function openInputDrawer(tab) {
        ensureInputDrawer();
        if (drawerTitleEl) drawerTitleEl.textContent = DRAWER_TITLES[tab] || '모델 · 하중 입력';
        // NL 전용 모드: 이전 UI 흔적(탭 전환 UI 등) 없이 자연어 패널만 노출
        if (drawerEl) drawerEl.classList.toggle('fid-nl-mode', tab === 'nl');
        var o = document.getElementById('figma-input-overlay');
        if (o) o.classList.add('open');
        try { call('switchInputTab', tab); } catch (e) { console.warn('[figma-menu] switchInputTab', e); }
    }
    function closeInputDrawer() {
        var o = document.getElementById('figma-input-overlay');
        if (o) o.classList.remove('open');
    }
    /* 리본/독의 Model 탭과 랜딩 딥링크(figma_deeplink.js)가 같은 드로어를
     * 열어야 하므로 노출 — switchInputTab()만으론 아무것도 보이지 않는다. */
    window.figmaOpenInputDrawer = openInputDrawer;

    /* ---- item factories ---------------------------------------------------- */
    function action(label, fnName, opts) {
        opts = opts || {};
        return {
            label: label,
            hint: opts.hint,
            enabled: opts.enabled,
            run: function () { call(fnName, ...(opts.args || [])); }
        };
    }
    function toggleItem(label, cbxId, fnName, opts) {
        opts = opts || {};
        return {
            label: label,
            hint: opts.hint,
            enabled: opts.enabled,
            check: function () { return cbChecked(cbxId); },
            run: function () { var c = ensureCheckbox(cbxId); c.checked = !c.checked; call(fnName); }
        };
    }
    function soon(label, note) { return { label: label, disabled: true, note: note || '준비 중' }; }
    var SEP = { sep: true };

    /* Natural-language input reaches a paid LLM. On a keyless public
       deployment the response is an English billing error that ends up in an
       alert(), so the demo hides the entry — but only the demo. With
       DEMO_MODE off this must behave exactly as it always has, which is the
       whole contract of that flag. window.DEMO_MODE is emitted by
       editor_figma.html from the server-side value. */
    function nlEntry(label) {
        if (window.DEMO_MODE) {
            return soon(label, '자연어 입력은 이 공개 데모에서 비활성화되어 있습니다 (LLM 키 필요)');
        }
        return { label: label, hint: 'NL', run: function () { openInputDrawer('nl'); } };
    }

    /* ---- menu content (keyed by the exact button text in .menu-strip) ------ */
    var MENUS = {
        'File': [
            { label: '직접 입력 (모델 정의)', hint: 'Manual', run: function () { openInputDrawer('manual'); } },
            nlEntry('자연어 입력'),
            { label: 'IFC 가져오기…', hint: 'Import', run: function () { openInputDrawer('ifc'); } },
            SEP,
            { label: '해석 리포트 열기', hint: 'Report', enabled: function () { return !!reportUrl(); },
              run: function () { var u = reportUrl(); if (u) window.open(u, '_blank'); } },
            action('표 내보내기 (Excel)', 'exportToExcel', { hint: 'xlsx', enabled: hasJob }),
            action('도면 내보내기 (DXF)', 'exportToDXF', { hint: 'dxf', enabled: hasJob })
        ],
        'Edit': [
            action('선택 모드', 'setEditMode', { args: ['view'], hint: 'Select' }),
            action('절점 추가', 'setEditMode', { args: ['addNode'], hint: 'Node' }),
            action('부재 추가', 'setEditMode', { args: ['addElement'], hint: 'Element' }),
            SEP,
            action('릴리즈 지정', 'setEditMode', { args: ['release'], hint: 'Release' }),
            action('지점 지정', 'setEditMode', { args: ['support'], hint: 'Support' })
        ],
        'View': [
            action('3D 초기화', 'resetCamera', { hint: 'Home' }),
            action('평면도 (XY)', 'setViewPreset', { args: ['top'] }),
            action('정면도 (XZ)', 'setViewPreset', { args: ['front'] }),
            action('측면도 (YZ)', 'setViewPreset', { args: ['right'] }),
            SEP,
            action('와이어프레임', 'toggleWireframe'),
            action('축 표시', 'toggleAxes'),
            action('그리드 스냅', 'toggleGridSnap'),
            SEP,
            toggleItem('절점 번호', 'toggle-node-numbers', 'toggleNodeNumbers'),
            toggleItem('부재 번호', 'toggle-member-numbers', 'toggleMemberNumbers'),
            toggleItem('변형 형상', 'toggle-deformed', 'toggleDeformedShape', { enabled: hasResult }),
            toggleItem('설계검토 색상 (DC)', 'toggle-dc-colors', 'toggleDesignCheckColors', { enabled: hasResult })
        ],
        'Model': [
            action('재료 정의 (Materials)…', 'figmaOpenMaterials', { hint: 'KS 강종' }),
            action('단면 정의 (Sections)…', 'figmaOpenSections', { hint: 'KS 738' }),
            SEP,
            action('기둥 단면 지정', 'applyGlobalSection', { args: ['column'] }),
            SEP,
            soon('모델 탐색기 새로고침', '트리 렌더 이식 예정 (Tier 1-1)')
        ],
        'Node/Element': [
            action('절점 추가', 'setEditMode', { args: ['addNode'] }),
            action('부재 추가', 'setEditMode', { args: ['addElement'] }),
            action('릴리즈 지정', 'setEditMode', { args: ['release'] })
        ],
        'Boundary': [
            action('지점 지정', 'setEditMode', { args: ['support'] }),
            soon('스프링', '엔진 훅 없음 (백로그)'),
            soon('구속조건', '엔진 훅 없음 (백로그)')
        ],
        'Load': [
            nlEntry('자연어 입력 (모델·하중 자동 생성)'),
            SEP,
            action('고정하중 정의 (Dead)…', 'figmaOpenLoads', { args: ['dead'], hint: 'D' }),
            action('활하중 정의 (Live)…', 'figmaOpenLoads', { args: ['live'], hint: 'L' }),
            action('지진하중 정의 (Seismic)…', 'figmaOpenLoads', { args: ['seismic'], hint: 'E' }),
            action('풍하중 정의 (Wind)…', 'figmaOpenLoads', { args: ['wind'], hint: 'W' })
        ],
        'Analysis': [
            action('해석 실행', 'runAnalysis', { hint: 'Run' }),
            soon('해석 옵션', '옵션 패널 예정')
        ],
        'Results': [
            action('축력도 (N)', 'setDiagramMode', { args: ['axial'], enabled: hasResult }),
            action('전단력도 (V)', 'setDiagramMode', { args: ['shear'], enabled: hasResult }),
            action('모멘트도 (M)', 'setDiagramMode', { args: ['moment'], enabled: hasResult }),
            action('다이어그램 끄기', 'setDiagramMode', { args: ['off'], enabled: hasResult }),
            SEP,
            toggleItem('변형 형상', 'toggle-deformed', 'toggleDeformedShape', { enabled: hasResult }),
            soon('층간변위 (Story Drift)', 'drift 패널 DOM 이식 필요 (Tier 1-2)')
        ],
        'Design': [
            toggleItem('설계검토 색상 (DC)', 'toggle-dc-colors', 'toggleDesignCheckColors', { enabled: hasResult }),
            soon('설계검토 요약', 'dc-summary 패널 연결 예정')
        ],
        'Tools': [
            action('그리드 스냅', 'toggleGridSnap'),
            action('표 내보내기 (Excel)', 'exportToExcel', { enabled: hasJob }),
            action('도면 내보내기 (DXF)', 'exportToDXF', { enabled: hasJob })
        ],
        'Help': [
            { label: '정보 (About)', run: function () {
                alert('OpenSees Frame Studio — Figma UI Lab (/editor-figma)\n격리 실험 UI · 엔진: editor3d_figma.js'); } },
            soon('단축키 도움말', '준비 중')
        ]
    };

    /* ---- dropdown rendering + open/close mechanics ------------------------- */
    var dropdown = null;    // single floating panel reused for all menus
    var openBtn = null;     // currently open menu button

    function makeDropdown() {
        var d = document.createElement('div');
        d.className = 'figma-menu-dropdown';
        document.body.appendChild(d);
        // clicks inside must not bubble to the document-close handler
        d.addEventListener('mousedown', function (e) { e.stopPropagation(); });
        return d;
    }

    function toast(msg) {
        var t = document.getElementById('figma-menu-toast');
        if (!t) {
            t = document.createElement('div');
            t.id = 'figma-menu-toast';
            t.className = 'figma-menu-toast';
            document.body.appendChild(t);
        }
        t.textContent = msg;
        t.classList.add('show');
        clearTimeout(t._h);
        t._h = setTimeout(function () { t.classList.remove('show'); }, 2200);
    }

    function renderItems(items) {
        dropdown.innerHTML = '';
        items.forEach(function (it) {
            if (it.sep) {
                var s = document.createElement('div');
                s.className = 'figma-menu-sep';
                dropdown.appendChild(s);
                return;
            }
            var row = document.createElement('div');
            row.className = 'figma-menu-item';
            var isEnabled = !it.disabled && (typeof it.enabled === 'function' ? it.enabled() : true);
            if (!isEnabled) row.classList.add('disabled');
            if (it.note) row.title = it.note;

            var checked = (typeof it.check === 'function') ? it.check() : null;
            var chk = document.createElement('span');
            chk.className = 'fmi-check';
            chk.textContent = checked ? '✓' : '';
            row.appendChild(chk);

            var lab = document.createElement('span');
            lab.className = 'fmi-label';
            lab.textContent = it.label;
            row.appendChild(lab);

            if (it.hint) {
                var h = document.createElement('span');
                h.className = 'fmi-hint';
                h.textContent = it.hint;
                row.appendChild(h);
            }

            row.addEventListener('click', function () {
                if (!isEnabled) {
                    if (it.note) toast(it.label + ' — ' + it.note);
                    else if (typeof it.enabled === 'function') toast(it.label + ' — 먼저 해석을 실행하세요');
                    return;
                }
                closeMenu();
                try { it.run(); }
                catch (e) { console.warn('[figma-menu]', it.label, e); toast(it.label + ' — 아직 준비 중이거나 해석이 필요합니다'); }
            });
            dropdown.appendChild(row);
        });
    }

    function openMenu(btn) {
        var items = MENUS[btn.textContent.trim()];
        if (!items) return;
        if (!dropdown) dropdown = makeDropdown();
        renderItems(items);
        var r = btn.getBoundingClientRect();
        dropdown.style.visibility = 'hidden';
        dropdown.classList.add('open');
        // clamp to viewport right edge
        var w = dropdown.offsetWidth;
        var left = Math.min(r.left, window.innerWidth - w - 8);
        dropdown.style.left = Math.max(4, left) + 'px';
        dropdown.style.top = r.bottom + 'px';
        dropdown.style.visibility = '';
        if (openBtn && openBtn !== btn) openBtn.classList.remove('menu-open');
        btn.classList.add('menu-open');
        openBtn = btn;
    }

    function closeMenu() {
        if (dropdown) dropdown.classList.remove('open');
        if (openBtn) openBtn.classList.remove('menu-open');
        openBtn = null;
    }

    /* IFC import: after parse, DISPLAY the parsed model + close the drawer.
     * Skip the wizard's geometry/config/analyze steps — Run stays separate. */
    function wrapUploadIFC() {
        if (typeof window.uploadIFC !== 'function' || window.uploadIFC._figmaWrapped) return;
        var orig = window.uploadIFC;
        window.uploadIFC = async function () {
            var prev = window._v2Model;
            await orig.apply(this, arguments);
            var m = window._v2Model;
            // only display on a *successful* parse (produces a new model object)
            if (m && m !== prev && m.nodes && m.nodes.length) {
                // editor_v2에선 위저드의 runAnalysisFromIFCWizard()가 세팅하는
                // source marker를, 위저드를 건너뛰는 이 랩에선 여기서 확정.
                // (없으면 Run이 직접입력 폼의 기본 모델로 해석해버린다.)
                setModelSource('IFC');
                try { call('buildV2PreviewScene', m); }
                catch (e) { console.warn('[figma-menu] IFC preview', e); }
                closeInputDrawer();
                toast('IFC 모델을 뷰에 로드했습니다 · 해석은 Run 버튼');
            }
        };
        window.uploadIFC._figmaWrapped = true;
    }

    /* Run(리본 ▶ / 우측 패널 / Analyze 메뉴)은 인자 없는 runAnalysis()를 부르는데,
     * 엔진의 그 경로는 *직접입력 폼*에서 config를 재구성한다 — IFC import 후엔
     * 기본 모델이 해석되고, 그 updated_model이 _v2Model(가져온 IFC 모델)을
     * 덮어쓰기까지 한다. IFC 소스일 땐 window._v2Model 자체를 POST하는
     * runAnalysisV2()로 라우팅한다. configOverride가 있는 호출(NL/위저드
     * fallback)은 그대로 통과. */
    function wrapRunAnalysis() {
        if (typeof window.runAnalysis !== 'function' || window.runAnalysis._figmaWrapped) return;
        var orig = window.runAnalysis;
        window.runAnalysis = function (configOverride) {
            // A model load in flight means the imported model is not active
            // yet; running now would analyse the manual-input defaults and
            // present the result as if it were the imported model.
            if (window._modelLoading) {
                toast('모델을 불러오는 중입니다 · 잠시 후 실행해 주세요');
                return;
            }
            var src = '';
            try { src = modelSource || ''; } catch (_) {}
            if (!configOverride && src.indexOf('IFC') === 0 && window._v2Model) {
                return call('runAnalysisV2');
            }
            return orig.apply(this, arguments);
        };
        window.runAnalysis._figmaWrapped = true;
    }

    /* ---- 해석 로딩 오버레이 -------------------------------------------------
     * 문제: V2 해석(runAnalysisV2)은 상태바 텍스트만 바꿔서 실행 중인지
     * 구분이 안 되고, 해석 중 Run 연타 시 중복 요청도 나간다.
     * 해법: 화면 중앙 모달 카드(스피너+단계 메시지+경과시간). 오버레이가
     * 클릭을 막아 중복 실행도 차단. 배선은 3곳 —
     *   ① window.showLoading/hideLoading 오버라이드: V1 경로(직접입력 등)가
     *      부르는 뷰포트 스피너를 이 오버레이로 라우팅 (refcount)
     *   ② runAnalysisV2 래핑: V2 경로는 로딩 호출이 아예 없어 함수를 감쌈
     *      (promise resolve/reject 양쪽에서 hide, rethrow 보존)
     *   ③ setStatus 미러: 오버레이 표시 중엔 진행 메시지를 카드에도 표시   */
    var loadOverlay = null, loadTitle = null, loadStatusEl = null, loadElapsed = null;
    var loadCount = 0, loadTimer = null, loadT0 = 0;
    function ensureLoadOverlay() {
        if (loadOverlay) return;
        loadOverlay = document.createElement('div');
        loadOverlay.id = 'figma-loading';
        var card = document.createElement('div');
        card.className = 'fl-card';
        var spin = document.createElement('div'); spin.className = 'fl-spinner';
        loadTitle = document.createElement('div'); loadTitle.className = 'fl-title';
        loadStatusEl = document.createElement('div'); loadStatusEl.className = 'fl-status';
        var bar = document.createElement('div'); bar.className = 'fl-bar';
        bar.appendChild(document.createElement('span'));
        loadElapsed = document.createElement('div'); loadElapsed.className = 'fl-elapsed';
        card.appendChild(spin); card.appendChild(loadTitle); card.appendChild(loadStatusEl);
        card.appendChild(bar); card.appendChild(loadElapsed);
        loadOverlay.appendChild(card);
        document.body.appendChild(loadOverlay);
    }
    function loadingShow(text) {
        ensureLoadOverlay();
        loadCount++;
        loadTitle.textContent = text || '구조해석 실행 중';
        if (loadCount === 1) {
            loadStatusEl.textContent = '';
            loadElapsed.textContent = '경과 0.0 s';
            loadT0 = Date.now();
            loadOverlay.classList.add('open');
            loadTimer = setInterval(function () {
                loadElapsed.textContent = '경과 ' + ((Date.now() - loadT0) / 1000).toFixed(1) + ' s';
            }, 100);
        }
    }
    function loadingHide() {
        if (loadCount > 0) loadCount--;
        if (loadCount === 0 && loadOverlay) {
            loadOverlay.classList.remove('open');
            if (loadTimer) { clearInterval(loadTimer); loadTimer = null; }
        }
    }
    /* ---- 카메라 전체뷰(fit) 수정 ---------------------------------------------
     * 엔진 fitCameraToModel(viewer)는 Three 좌표 (x, z, -y) 규약인데 타겟을
     * (cx, cz, +cy)로 잡는다 — Y부호가 반대라 해석 직후/3D 리셋 때 카메라가
     * 건물 반대편 빈 공간을 본다 (프리뷰 buildV2PreviewScene은 -cy로 올바름).
     * 게다가 viewer.total_*는 원점(0,0,0) 시작을 가정해 오프셋 모델에 취약.
     * → window.fitCameraToModel을 노드 bbox 기반으로 교체 (엔진 무수정,
     *   buildScene/resetCamera/프로젝트 로드 등 모든 호출처가 window 경유). */
    function wrapFitCamera() {
        if (window.fitCameraToModel && window.fitCameraToModel._figmaFixed) return;
        window.fitCameraToModel = function (viewer) {
            var nodes = (window._v2Model && window._v2Model.nodes) || (viewer && viewer.nodes) || [];
            var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
            nodes.forEach(function (n) {
                if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
                if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
                if (n.z < minZ) minZ = n.z; if (n.z > maxZ) maxZ = n.z;
            });
            if (!isFinite(minX)) {   // 노드 없음 — viewer totals 폴백 (부호만 교정)
                if (!viewer) return;
                minX = 0; maxX = viewer.total_width_x || 20;
                minY = 0; maxY = viewer.total_width_y || 20;
                minZ = 0; maxZ = viewer.total_height || 10;
            }
            var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
            try {
                // 바운딩 스피어가 프러스텀에 완전히 들어가는 거리 (FOV·종횡비 반영)
                var ex = (maxX - minX) / 2, ey = (maxY - minY) / 2, ez = (maxZ - minZ) / 2;
                var R = Math.max(Math.sqrt(ex * ex + ey * ey + ez * ez), 2.5);
                var vHalf = (camera.fov || 60) * Math.PI / 360;
                var hHalf = Math.atan(Math.tan(vHalf) * (camera.aspect || 1));
                var dist = R / Math.sin(Math.min(vHalf, hHalf)) * 1.12;
                // 시선 방향 (0.6, 0.4, 0.6) 정규화 유닛벡터
                var ux = 0.6396, uy = 0.4264, uz = 0.6396;
                camera.position.set(cx + dist * ux, cz + dist * uy, -cy + dist * uz);
                controls.target.set(cx, cz, -cy);
                controls.update();
                if (typeof gridHelper !== 'undefined' && gridHelper) gridHelper.position.set(cx, 0, -cy);
            } catch (e) { console.warn('[figma-menu] fitCamera', e); }
        };
        window.fitCameraToModel._figmaFixed = true;
    }

    function wrapLoadingHooks() {
        window.showLoading = function (text) { loadingShow(text); };
        window.hideLoading = function () { loadingHide(); };
        if (typeof window.runAnalysisV2 === 'function' && !window.runAnalysisV2._figmaLoad) {
            var orig = window.runAnalysisV2;
            window.runAnalysisV2 = function () {
                loadingShow('구조해석 실행 중');
                var p;
                try { p = orig.apply(this, arguments); }
                catch (e) { loadingHide(); throw e; }
                return Promise.resolve(p).then(
                    function (r) { loadingHide(); return r; },
                    function (e) { loadingHide(); throw e; });
            };
            window.runAnalysisV2._figmaLoad = true;
        }
        if (typeof window.setStatus === 'function' && !window.setStatus._figmaLoad) {
            var os = window.setStatus;
            window.setStatus = function (text, type) {
                var r = os.apply(this, arguments);
                try { if (loadCount > 0 && loadStatusEl) loadStatusEl.textContent = text || ''; } catch (_) {}
                return r;
            };
            window.setStatus._figmaLoad = true;
        }
    }

    function init() {
        wrapUploadIFC();
        wrapRunAnalysis();
        wrapLoadingHooks();
        wrapFitCamera();
        var btns = document.querySelectorAll('.menu-strip button');
        if (!btns.length) return;
        btns.forEach(function (btn) {
            if (!MENUS[btn.textContent.trim()]) return;
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                if (openBtn === btn) { closeMenu(); }
                else { openMenu(btn); }
            });
            // classic menu-bar behaviour: hover switches while a menu is open
            btn.addEventListener('mouseenter', function () {
                if (openBtn && openBtn !== btn) openMenu(btn);
            });
        });
        document.addEventListener('mousedown', function () { closeMenu(); });
        document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeMenu(); });
        window.addEventListener('resize', closeMenu);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
