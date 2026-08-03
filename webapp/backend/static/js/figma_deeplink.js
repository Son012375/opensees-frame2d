/* ==========================================================================
   Deep links from the landing page.

   The landing page prints the results of a specific model — the editor's
   default 3-story preset — and then invites the visitor to run it themselves.
   For that invitation to mean anything, the link has to open the *same* model,
   already set up, with the Run button obviously next.

   Everything here drives functions that already exist and are already global.
   This file adds no engineering behaviour; it is a URL parser plus a coach
   mark. Delete it and the editor is unchanged.

   Loaded LAST (after figma_explorer.js) and acting on `load`, not
   DOMContentLoaded, so every module has installed its globals and figma_menu
   has finished wrapping runAnalysis/uploadIFC. ?demo=ifc is deliberately NOT
   handled here — editor3d_figma.js owns it, and doing it twice would parse the
   sample file twice.
   ========================================================================== */
(function () {
    'use strict';

    var params = new URLSearchParams(window.location.search);
    if (!params.toString()) return;

    /* Result-table kinds are named for the visitor in the URL and for the
       implementation in figma_explorer.js. `?result=member` is what a person
       would type; 'force' is what tableSpec() branches on. */
    var RESULT_KINDS = {
        drift: 'drift',
        reaction: 'reaction',
        member: 'force',
        force: 'force',
        design: 'dc',
        dc: 'dc',
        modal: 'modal'
    };

    var LOAD_PANELS = {
        'load-dead': 'dead',
        'load-live': 'live',
        'load-seismic': 'seismic',
        'load-wind': 'wind'
    };

    function log(msg, err) {
        if (err) console.warn('[deeplink] ' + msg, err);
        else console.info('[deeplink] ' + msg);
    }

    /* The section dropdowns are filled by an async fetch fired at
       DOMContentLoaded, so on a cold load they can still be empty here.
       Waiting on the DOM state is more honest than waiting on a fixed delay:
       a slow network just means a slightly later prefill, never a silent miss. */
    function whenSectionsReady(fn) {
        var deadline = Date.now() + 8000;
        (function poll() {
            var sel = document.getElementById('input-col-section');
            if (sel && sel.options.length > 1) { fn(true); return; }
            if (Date.now() > deadline) { fn(false); return; }
            setTimeout(poll, 120);
        })();
    }

    function call(name, arg) {
        var fn = window[name];
        if (typeof fn !== 'function') { log(name + ' unavailable'); return false; }
        try { fn(arg); return true; } catch (e) { log(name + ' threw', e); return false; }
    }

    // ── the "same model as the landing" banner ───────────────────────────────
    // The quoted numbers belong to the baseline H-300x300 run and to nothing
    // else. A ?col=/?beam= link is deliberately a DIFFERENT model — the landing
    // knob uses it to send someone to the section that flips the verdict — so
    // asserting the baseline figures there would have the banner contradict the
    // very result it is inviting the visitor to produce.
    function showBenchBanner(overridden) {
        if (document.getElementById('fig-dl-banner')) return;
        var bar = document.createElement('div');
        bar.id = 'fig-dl-banner';
        var msg = overridden
            ? '랜딩에서 고른 단면 <b>' + overridden + '</b>이 적용된 상태입니다 — ' +
              '▶ Run을 누르면 이 단면의 판정을 직접 확인할 수 있습니다.'
            : '랜딩에 실린 것과 <b>같은 모델</b>입니다 — ▶ Run을 누르면 ' +
              '총중량 3916.8 kN · 밑면전단 421.98 kN · 최대 상관비 1.4933이 그대로 나옵니다.';
        bar.innerHTML =
            '<span class="fdl-dot" aria-hidden="true"></span>' +
            '<span>' + msg + '</span>' +
            '<button type="button" class="fdl-x" aria-label="닫기">✕</button>';
        bar.querySelector('.fdl-x').addEventListener('click', function () { bar.remove(); });
        document.body.appendChild(bar);
    }

    // ── coach mark on ▶ Run ──────────────────────────────────────────────────
    // Styles are injected rather than added to editor_figma.css so this whole
    // feature stays in one removable file and does not force a CSS cache bump.
    function injectStyles() {
        if (document.getElementById('fig-dl-style')) return;
        var css = document.createElement('style');
        css.id = 'fig-dl-style';
        css.textContent = [
            '@keyframes fdlPulse{0%{box-shadow:0 0 0 0 rgba(26,115,232,.55)}',
            '70%{box-shadow:0 0 0 12px rgba(26,115,232,0)}',
            '100%{box-shadow:0 0 0 0 rgba(26,115,232,0)}}',
            '.fdl-coach{position:relative;border-radius:6px;animation:fdlPulse 1.9s ease-out 4}',
            '#fig-dl-banner{position:fixed;left:50%;transform:translateX(-50%);bottom:44px;',
            'z-index:900;display:flex;align-items:center;gap:10px;max-width:min(760px,92vw);',
            'padding:10px 14px;border-radius:8px;font-size:12.5px;line-height:1.5;',
            'background:#1A73E8;color:#fff;box-shadow:0 10px 30px rgba(0,0,0,.28)}',
            '#fig-dl-banner b{font-weight:700}',
            '#fig-dl-banner .fdl-dot{width:8px;height:8px;border-radius:50%;background:#fff;flex:0 0 auto}',
            '#fig-dl-banner .fdl-x{margin-left:auto;background:none;border:0;color:#fff;',
            'font-size:13px;cursor:pointer;padding:0 2px;opacity:.8}',
            '#fig-dl-banner .fdl-x:hover{opacity:1}',
            '@media (prefers-reduced-motion:reduce){.fdl-coach{animation:none;',
            'outline:2px solid #1A73E8;outline-offset:2px}}'
        ].join('');
        document.head.appendChild(css);
    }

    function coachRun() {
        var btn = document.querySelector('.ribbon-command.run');
        if (!btn) return;
        btn.classList.add('fdl-coach');
        // The pulse is a pointer, not decoration: once the visitor has done the
        // thing it points at, it stops being either.
        btn.addEventListener('click', function () {
            btn.classList.remove('fdl-coach');
            var bar = document.getElementById('fig-dl-banner');
            if (bar) bar.remove();
        }, { once: true });
    }

    // ── ?demo=bench ──────────────────────────────────────────────────────────
    function openBench(colOverride, beamOverride) {
        var preset = document.getElementById('preset-select');
        if (preset) {
            preset.value = '3story';
            call('applyPreset');           // rebuilds stories, bays and sections
        }
        if (colOverride) setSection('input-col-section', colOverride);
        if (beamOverride) {
            setSection('input-beamx-section', beamOverride);
            setSection('input-beamy-section', beamOverride);
        }
        call('updateManualPreview');
        injectStyles();
        showBenchBanner(colOverride || beamOverride || null);
        coachRun();
    }

    function setSection(id, name) {
        var sel = document.getElementById(id);
        if (!sel) return;
        if (typeof window.setSelectValue === 'function') {
            window.setSelectValue(sel, name);
        } else {
            sel.value = name;
        }
    }

    // ── ?panel= ──────────────────────────────────────────────────────────────
    function openPanel(which) {
        if (which === 'sections') return call('figmaOpenSections');
        if (which === 'materials') return call('figmaOpenMaterials');
        if (LOAD_PANELS[which]) return call('figmaOpenLoads', LOAD_PANELS[which]);
        if (which === 'input' || which === 'manual') return call('figmaOpenInputDrawer', 'manual');
        if (which === 'ifc') return call('figmaOpenInputDrawer', 'ifc');
        log('unknown panel: ' + which);
        return false;
    }

    // ── ?run=1 ───────────────────────────────────────────────────────────────
    // Only auto-run when the server says a run will actually start. A visitor
    // whose first click queues behind three others concludes the tool is
    // broken, and they never see that it was a queue.
    function autoRun() {
        fetch('/api/health/demo')
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (h) {
                if (!h || h.ready === false) {
                    log('auto-run skipped — server busy');
                    return;
                }
                if (window._modelLoading) { setTimeout(autoRun, 800); return; }
                call('runAnalysis');
            })
            .catch(function (e) { log('health probe failed; not auto-running', e); });
    }

    // ── post-analysis params ─────────────────────────────────────────────────
    // These need a result, which does not exist at load. Rather than polling
    // forever, watch for one for a bounded window and then give up quietly.
    function whenResult(fn) {
        var deadline = Date.now() + 180000;   // 3 min covers a queued 12 s run
        (function poll() {
            // `currentResult` is a top-level `let` in editor3d_figma.js, so it
            // is a script binding and never appears on `window`. Read it bare,
            // the way figma_explorer.js does.
            var r = null;
            try { r = currentResult; } catch (_) {}
            if (r) { fn(r); return; }
            if (Date.now() > deadline) return;
            setTimeout(poll, 500);
        })();
    }

    // ── go ───────────────────────────────────────────────────────────────────
    window.addEventListener('load', function () {
        var demo = params.get('demo');
        var col = params.get('col');
        var beam = params.get('beam');
        var panel = params.get('panel');
        var result = params.get('result');

        if (demo === 'bench' || col || beam) {
            whenSectionsReady(function (ready) {
                if (!ready) log('section list never arrived — preset applied without prefill');
                openBench(col, beam);
                if (panel) openPanel(panel);
                if (params.get('run') === '1') autoRun();
            });
        } else {
            // ?demo=ifc and bare ?panel= links: the IFC path is driven by
            // editor3d_figma.js, so only the panel is ours to open.
            //
            // The readiness wait is NOT optional here. /api/sections/list does
            // not block the `load` event, so on a cold open the catalogue is
            // still empty at this point and figmaOpenSections bails with a
            // status line instead of a window — a bare ?panel=sections link
            // would appear to do nothing at all.
            if (panel) {
                whenSectionsReady(function (ready) {
                    if (!ready) log('section list never arrived — opening ' + panel + ' anyway');
                    if (demo === 'ifc') setTimeout(function () { openPanel(panel); }, 1200);
                    else openPanel(panel);
                });
            }
            if (params.get('run') === '1' && demo !== 'ifc') autoRun();
        }

        if (result && RESULT_KINDS[result]) {
            whenResult(function () { call('figmaOpenResultTable', RESULT_KINDS[result]); });
        }
        if (params.get('report') === '1') {
            whenResult(function () { call('figmaOpenReport'); });
        }
    });
})();
