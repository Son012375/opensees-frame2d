/* ============================================================================
 * figma_plan.js  —  Live, interactive Plan View for the Figma UI Lab
 * ----------------------------------------------------------------------------
 * - Renders a real 2D top-view of window._v2Model at a selectable Z-level.
 * - Story selector (▾) pages through each floor.
 * - Hover a member -> reuses the 3D viewer's rich tooltip (showHoverTooltip).
 * - Click a member (plan OR 3D) -> unified selection: plan highlight + 3D
 *   highlight + the visible "Properties of Object" panel.
 * - DRAW (ETABS식 플랜 모델링): ribbon N(Node)/B(Beam) = engine edit modes
 *   addNode/addElement. 이 파일이 plan 캔버스 쪽 구현을 담당 —
 *   절점(1클릭) / 보(2클릭 체인, 방향따라 beam_x/beam_y) / 기둥(1클릭,
 *   현재층→아래층). 스냅: 기존절점 > 그리드축 > 0.5m. 생성은 엔진 프리미티브
 *   (pushUndo/getNextNodeId/getNextElemId/splitElementsAtNodes/
 *   refreshEditPreview)만 사용 -> Ctrl+Z 언두 통합. 3D 쪽 그리기는 엔진의
 *   기존 다이얼로그 흐름 그대로(여기선 setEditMode 래핑으로 편집 활성화만
 *   보장). Engine JS is never edited.
 *
 * Also hardens the selection chain: the Figma reskin dropped #prop-node /
 * #prop-multi / selection-count, so the engine's showMemberProperties /
 * clearAllSelection would throw on any member/node click. Hidden stubs cover
 * the ids; these wrappers additionally feed the *visible* Figma panel and keep
 * the plan highlight in sync with 3D clicks. Engine JS is never edited.
 * ========================================================================== */
(function () {
    'use strict';

    var canvas, ctx, wrap, levelBtn, levelMenu, titleH3;
    var levels = [];
    var levelIndex = 0;
    var EPS = 0.05;

    var _hits = [];
    var selectedElemId = null;
    var _hoverElemId = null;
    var selectedNodeId = null, editSelType = null, editPanel = null;
    var HOVER_SENTINEL = { plan: true };

    /* draw layer state */
    var view = null;                                  // last render()'s world<->screen transform
    var drawState = {
        mode: null,                                   // null | 'node' | 'frame'  (editMode 파생)
        type: 'beam',                                 // frame 타입: 'beam' | 'column'
        pending: null,                                // 보 그리기 시작점 {x,y,node}
        ghost: null,                                  // 마우스 스냅 프리뷰 {x,y,node}
        sections: { beam: null, column: null },       // 타입별 선택 단면 (기본=모델 최빈값)
    };
    var drawPanel = null, drawTypeSel = null, drawSectionSel = null, drawHintEl = null, drawLevelEl = null;

    /* plan solid(실폭) 표시 상태 — 단면 치수 캐시는 엔진(figma_tools)의
     * window._sectionDims {name: {h,b,tw,tf}(mm)} 를 공유 */
    var planSolid = false;
    var _dimsPending = false;

    var C = {
        beamX: '#34a853', beamY: '#fbbc04', beam: '#34a853',
        col: '#4285f4', node: '#5f6368', grid: 'rgba(120,130,145,0.18)',
        support: '#ff6600', text: '#5f6368', sel: '#ff3b30',
    };

    function getModel() { try { return window._v2Model || null; } catch (_) { return null; } }
    function approx(a, b) { return Math.abs((a || 0) - (b || 0)) <= EPS; }
    function fmt(z) { return (Math.round(z * 100) / 100).toFixed(2); }
    function typeLabelOf(t) {
        return t === 'column' ? 'Column' : t === 'beam_x' ? 'Beam X' : t === 'beam_y' ? 'Beam Y'
            : t === 'beam' ? 'Beam' : t === 'brace' ? 'Brace' : (t || 'Element');
    }

    /* ---- levels ------------------------------------------------------------ */
    function computeLevels(model) {
        var set = {};
        (model.nodes || []).forEach(function (n) { set[Math.round((n.z || 0) * 100) / 100] = true; });
        return Object.keys(set).map(Number).sort(function (a, b) { return a - b; });
    }
    function levelLabel(i) { return (i === 0 ? 'Base' : (i + 'F')) + ' : Z = ' + fmt(levels[i]); }

    /* ---- rendering --------------------------------------------------------- */
    function sizeCanvas() {
        var r = wrap.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(r.width * dpr));
        canvas.height = Math.max(1, Math.round(r.height * dpr));
        canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function placeholder(msg) {
        var r = wrap.getBoundingClientRect();
        ctx.clearRect(0, 0, r.width, r.height);
        ctx.fillStyle = C.text; ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(msg, r.width / 2, r.height / 2);
    }

    function render() {
        if (!canvas) return;
        sizeCanvas();
        _hits = [];
        view = null;
        var r = wrap.getBoundingClientRect(), W = r.width, H = r.height;
        ctx.clearRect(0, 0, W, H);

        var model = getModel();
        if (!model || !model.nodes || !model.nodes.length) {
            placeholder('모델을 불러오면 평면도가 표시됩니다 (File → IFC 가져오기)');
            return;
        }
        if (!levels.length) levels = computeLevels(model);
        if (levelIndex >= levels.length) levelIndex = 0;
        var z = levels[levelIndex];

        var nodeMap = {};
        model.nodes.forEach(function (n) { nodeMap[n.id] = n; });

        var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        model.nodes.forEach(function (n) {
            if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
            if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
        });
        if (!isFinite(minX)) { placeholder('좌표 없음'); return; }
        var spanX = Math.max(maxX - minX, 0.001), spanY = Math.max(maxY - minY, 0.001);
        var margin = 42, scale = Math.min((W - 2 * margin) / spanX, (H - 2 * margin) / spanY);
        var offX = (W - spanX * scale) / 2, offY = (H - spanY * scale) / 2;
        function SX(x) { return offX + (x - minX) * scale; }
        function SY(y) { return H - (offY + (y - minY) * scale); }

        var xs = {}, ys = {};
        model.nodes.forEach(function (n) { if (approx(n.z, z)) { xs[Math.round(n.x * 100) / 100] = 1; ys[Math.round(n.y * 100) / 100] = 1; } });
        var xKeys = Object.keys(xs).map(Number), yKeys = Object.keys(ys).map(Number);
        if (!xKeys.length) { model.nodes.forEach(function (n) { xs[Math.round(n.x * 100) / 100] = 1; ys[Math.round(n.y * 100) / 100] = 1; }); xKeys = Object.keys(xs).map(Number); yKeys = Object.keys(ys).map(Number); }
        view = { minX: minX, minY: minY, scale: scale, offX: offX, offY: offY, W: W, H: H, z: z, xKeys: xKeys, yKeys: yKeys, SX: SX, SY: SY };
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        xKeys.forEach(function (x) { ctx.beginPath(); ctx.moveTo(SX(x), SY(minY)); ctx.lineTo(SX(x), SY(maxY)); ctx.stroke(); });
        yKeys.forEach(function (y) { ctx.beginPath(); ctx.moveTo(SX(minX), SY(y)); ctx.lineTo(SX(maxX), SY(y)); ctx.stroke(); });

        var solidOn = planSolid && dimsReady(model);

        (model.elements || []).forEach(function (e) {
            var ni = nodeMap[e.node_i], nj = nodeMap[e.node_j];
            if (!ni || !nj || e.elem_type === 'column') return;
            if (!(approx(ni.z, z) && approx(nj.z, z))) return;
            var ax = SX(ni.x), ay = SY(ni.y), bx = SX(nj.x), by = SY(nj.y), sel = (e.id === selectedElemId);
            var color = sel ? C.sel : (e.elem_type === 'beam_y' ? C.beamY : (e.elem_type === 'brace' ? C.col : C.beamX));
            if (solidOn) {
                drawBeamStrip(e, ax, ay, bx, by, color, scale);
            } else {
                ctx.strokeStyle = color;
                ctx.lineWidth = sel ? 4.5 : 2.5;
                ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
            }
            _hits.push({ kind: 'beam', e: e, ax: ax, ay: ay, bx: bx, by: by });
        });
        ctx.lineWidth = 2.5;

        (model.elements || []).forEach(function (e) {
            if (e.elem_type !== 'column') return;
            var ni = nodeMap[e.node_i], nj = nodeMap[e.node_j];
            if (!ni || !nj) return;
            var at = approx(ni.z, z) ? ni : (approx(nj.z, z) ? nj : null);
            if (!at) return;
            var sx = SX(at.x), sy = SY(at.y), sel = (e.id === selectedElemId), s = sel ? 9 : 6;
            if (solidOn) {
                drawColumnSection(e, sx, sy, sel ? C.sel : C.col, scale);
            } else {
                ctx.fillStyle = sel ? C.sel : C.col;
                ctx.fillRect(sx - s / 2, sy - s / 2, s, s);
            }
            _hits.push({ kind: 'col', e: e, x: sx, y: sy });
        });

        model.nodes.forEach(function (n) {
            if (!approx(n.z, z)) return;
            ctx.fillStyle = (z <= EPS && n.support) ? C.support : C.node;
            ctx.beginPath(); ctx.arc(SX(n.x), SY(n.y), 2.6, 0, Math.PI * 2); ctx.fill();
        });

        ctx.fillStyle = C.text; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left';
        ctx.fillText('X →', W - 34, H - 10);
        ctx.save(); ctx.translate(12, 26); ctx.fillText('Y ↑', 0, 0); ctx.restore();

        drawOverlay();
    }

    /* ---- draw layer: overlay (ghost / rubber-band / pending marker) -------- */
    function drawOverlay() {
        if (!drawState.mode || !view) return;
        var g = drawState.ghost, p = drawState.pending;
        // rubber band: pending -> ghost (보 그리기)
        if (p && g && drawState.mode === 'frame' && drawState.type === 'beam') {
            ctx.save();
            ctx.strokeStyle = '#ff4081'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
            ctx.beginPath(); ctx.moveTo(view.SX(p.x), view.SY(p.y)); ctx.lineTo(view.SX(g.x), view.SY(g.y)); ctx.stroke();
            ctx.restore();
        }
        if (p) {
            ctx.save();
            ctx.strokeStyle = '#ff4081'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(view.SX(p.x), view.SY(p.y), 6, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
        }
        if (g) {
            var gx = view.SX(g.x), gy = view.SY(g.y);
            ctx.save();
            ctx.strokeStyle = g.node ? '#0c8043' : '#ff4081'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(gx, gy, 5, 0, Math.PI * 2); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(gx - 9, gy); ctx.lineTo(gx + 9, gy); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(gx, gy - 9); ctx.lineTo(gx, gy + 9); ctx.stroke();
            ctx.fillStyle = '#1f2937'; ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'left';
            var lbl = '(' + g.x.toFixed(1) + ', ' + g.y.toFixed(1) + ')' + (g.node ? ' N' + g.node.id : '');
            ctx.fillText(lbl, gx + 10, gy - 8);
            ctx.restore();
        }
    }

    /* ---- plan solid(실폭) 렌더 ---------------------------------------------- */
    function distinctSections(model) {
        var s = {};
        (model.elements || []).forEach(function (e) { if (e.section) s[e.section] = 1; });
        return Object.keys(s);
    }
    /* 이름 파싱 폴백: "H-400x200"→{h:400,b:200}, "□-125x125x6", "○-216.3x8" */
    function parseDimsFromName(name) {
        var m = name.match(/^[^\d]*([\d.]+)x([\d.]+)(?:x([\d.]+))?/);
        if (!m) return null;
        var a = parseFloat(m[1]), b = parseFloat(m[2]), c = m[3] ? parseFloat(m[3]) : null;
        if (name.indexOf('○-') === 0) return { h: a, b: a, tw: b, tf: b };      // D x t
        if (name.indexOf('□-') === 0) return { h: a, b: b, tw: c || 6, tf: c || 6 }; // H x B x t
        return { h: a, b: b, tw: 0, tf: 0 };                                     // H형강 등: 폭만 신뢰
    }
    function dimsReady(model) {
        var dims = window._sectionDims || {};
        var missing = distinctSections(model).filter(function (n) { return !dims[n]; });
        if (!missing.length) return true;
        ensurePlanDims(missing);
        return false;
    }
    function ensurePlanDims(missing) {
        if (_dimsPending) return;
        _dimsPending = true;
        if (!window._sectionDims) window._sectionDims = {};
        Promise.all(missing.map(function (name) {
            return fetch('/api/sections/properties/' + encodeURIComponent(name))
                .then(function (r) { return r.json(); })
                .then(function (d) {
                    window._sectionDims[name] = (d && d.h_mm)
                        ? { h: d.h_mm, b: d.b_mm || d.h_mm, tw: d.tw_mm || d.t_mm || 8, tf: d.tf_mm || d.t_mm || 12 }
                        : (parseDimsFromName(name) || { h: 300, b: 150, tw: 8, tf: 12 });
                })
                .catch(function () {
                    window._sectionDims[name] = parseDimsFromName(name) || { h: 300, b: 150, tw: 8, tf: 12 };
                });
        })).then(function () { _dimsPending = false; render(); });
    }
    function drawBeamStrip(e, ax, ay, bx, by, color, scale) {
        var d = (window._sectionDims || {})[e.section];
        var w = Math.max(((d && d.b) || 150) / 1000 * scale, 2);   // 보 평면 폭 = 플랜지 폭 b
        var L = Math.hypot(bx - ax, by - ay);
        ctx.save();
        ctx.translate(ax, ay);
        ctx.rotate(Math.atan2(by - ay, bx - ax));
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = color;
        ctx.fillRect(0, -w / 2, L, w);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(0, -w / 2, L, w);
        ctx.restore();
        // 중심선 (가는 선 유지 — 픽킹/시인성)
        ctx.strokeStyle = color; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    function drawColumnSection(e, sx, sy, color, scale) {
        var d = (window._sectionDims || {})[e.section] || { h: 300, b: 300, tw: 10, tf: 15 };
        var k = scale / 1000;
        var h = Math.max(d.h * k, 4), b = Math.max(d.b * k, 4);
        var tw = Math.max((d.tw || 8) * k, 1.2), tf = Math.max((d.tf || 12) * k, 1.5);
        ctx.save();
        ctx.translate(sx, sy);
        ctx.fillStyle = color;
        if (e.section && e.section.indexOf('○-') === 0) {          // 원형강관: 링
            ctx.beginPath();
            ctx.arc(0, 0, h / 2, 0, Math.PI * 2);
            ctx.arc(0, 0, Math.max(h / 2 - tw, 1), 0, Math.PI * 2, true);
            ctx.fill('evenodd');
        } else if (e.section && e.section.indexOf('□-') === 0) {   // 각형강관: 중공 박스
            ctx.beginPath();
            ctx.rect(-b / 2, -h / 2, b, h);
            ctx.rect(-b / 2 + tw, -h / 2 + tw, Math.max(b - 2 * tw, 1), Math.max(h - 2 * tw, 1));
            ctx.fill('evenodd');
        } else if (d.tf > 0) {                                      // H/I: 단면 글리프
            ctx.fillRect(-b / 2, -h / 2, b, tf);                    // 상부 플랜지
            ctx.fillRect(-b / 2, h / 2 - tf, b, tf);                // 하부 플랜지
            ctx.fillRect(-tw / 2, -h / 2 + tf, tw, Math.max(h - 2 * tf, 1)); // 웹
        } else {
            ctx.fillRect(-b / 2, -h / 2, b, h);
        }
        ctx.restore();
    }

    /* ---- picking ----------------------------------------------------------- */
    function distToSeg(px, py, ax, ay, bx, by) {
        var dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
        if (L2 === 0) return Math.hypot(px - ax, py - ay);
        var t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L2));
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    }
    function hitTest(mx, my) {
        var best = null, bestD = 7;
        _hits.forEach(function (h) {
            var d = h.kind === 'col' ? Math.hypot(mx - h.x, my - h.y) : distToSeg(mx, my, h.ax, h.ay, h.bx, h.by);
            if (d < bestD) { bestD = d; best = h; }
        });
        return best;
    }
    function buildElem(e) { return { type: e.elem_type, id: e.id, section: e.section || '-', ni: e.node_i, nj: e.node_j }; }

    /* ---- draw layer: snapping + creation ----------------------------------- */
    function eng(fnName) {
        var fn = window[fnName];
        if (typeof fn !== 'function') { console.warn('[figma-plan] engine fn missing:', fnName); return undefined; }
        try { return fn.apply(window, Array.prototype.slice.call(arguments, 1)); }
        catch (e) { console.warn('[figma-plan]', fnName, e); return undefined; }
    }
    function status(msg, kind) { try { if (typeof setStatus === 'function') setStatus(msg, kind || 'success'); } catch (_) {} }

    function findNodeAt(model, x, y, z) {
        return (model.nodes || []).find(function (n) {
            return approx(n.z, z) && Math.abs(n.x - x) <= EPS && Math.abs(n.y - y) <= EPS;
        }) || null;
    }
    /* snap ladder: ①기존 절점(12px) ②그리드 축 키(8px) ③0.5m 라운딩 */
    function snapPoint(mx, my) {
        var model = getModel();
        if (!model || !view) return null;
        var z = view.z;
        var best = null, bestD = 12;
        (model.nodes || []).forEach(function (n) {
            if (!approx(n.z, z)) return;
            var d = Math.hypot(view.SX(n.x) - mx, view.SY(n.y) - my);
            if (d < bestD) { bestD = d; best = n; }
        });
        if (best) return { x: best.x, y: best.y, node: best };
        var wx = view.minX + (mx - view.offX) / view.scale;
        var wy = view.minY + (view.H - my - view.offY) / view.scale;
        wx = snapAxis(wx, view.xKeys, mx, view.SX);
        wy = snapAxis(wy, view.yKeys, my, function (v) { return view.SY(v); });
        var n2 = findNodeAt(model, wx, wy, z);
        return { x: wx, y: wy, node: n2 };
    }
    function snapAxis(w, keys, screenPos, toScreen) {
        var best = null, bestD = 8;
        (keys || []).forEach(function (k) {
            var d = Math.abs(toScreen(k) - screenPos);
            if (d < bestD) { bestD = d; best = k; }
        });
        if (best !== null) return best;
        return Math.round(w * 2) / 2;
    }

    function mostCommon(model, types, field) {
        var cnt = {}, best = null;
        (model.elements || []).forEach(function (e) {
            if (types.indexOf(e.elem_type) < 0 || !e[field]) return;
            cnt[e[field]] = (cnt[e[field]] || 0) + 1;
            if (best === null || cnt[e[field]] > cnt[best]) best = e[field];
        });
        return best;
    }
    function drawSectionFor(type) {
        if (drawState.sections[type]) return drawState.sections[type];
        var model = getModel() || { elements: [] };
        return type === 'column'
            ? (mostCommon(model, ['column'], 'section') || 'H-300x300')
            : (mostCommon(model, ['beam_x', 'beam_y', 'beam'], 'section') || 'H-400x200');
    }
    function defaultMaterial(model) {
        return mostCommon(model, ['column', 'beam_x', 'beam_y', 'beam', 'brace'], 'material')
            || model.material_name || 'SS275';
    }
    /* 생성 없이 좌표의 기존 절점 id만 (dup 검사용) */
    function peekNodeId(model, s, z) { var n = findNodeAt(model, s.x, s.y, z); return n ? n.id : null; }
    function ensureNodeAt(model, x, y, z) {
        var n = findNodeAt(model, x, y, z);
        if (n) return n.id;
        var id = eng('getNextNodeId');
        if (id == null) id = (model.nodes || []).reduce(function (m, nn) { return Math.max(m, nn.id || 0); }, 0) + 1;
        model.nodes.push({ id: id, x: x, y: y, z: z, story: null, support: null, mass: null });
        return id;
    }
    function nextElemId(model) {
        var id = eng('getNextElemId');
        if (id == null) id = (model.elements || []).reduce(function (m, e) { return Math.max(m, e.id || 0); }, 0) + 1;
        return id;
    }
    function hasElementBetween(model, a, b) {
        return (model.elements || []).some(function (e) {
            return (e.node_i === a && e.node_j === b) || (e.node_i === b && e.node_j === a);
        });
    }

    function createPlanNode(s) {
        var model = getModel();
        if (!model) return;
        if (s.node) { status('이미 절점이 있습니다 (N' + s.node.id + ')', 'warning'); return; }
        eng('pushUndo');
        var nid = ensureNodeAt(model, s.x, s.y, view.z);
        var splits = eng('splitElementsAtNodes', model) || 0;
        eng('refreshEditPreview');
        status('절점 N' + nid + ' 생성 (' + s.x + ', ' + s.y + ', ' + view.z + ')' + (splits > 0 ? ' [부재 ' + splits + '개 분할]' : ''));
    }
    function createPlanBeam(a, b) {
        var model = getModel();
        if (!model) return;
        if (Math.abs(a.x - b.x) <= EPS && Math.abs(a.y - b.y) <= EPS) return;
        var exI = peekNodeId(model, a, view.z), exJ = peekNodeId(model, b, view.z);
        if (exI !== null && exJ !== null && hasElementBetween(model, exI, exJ)) {
            status('이미 같은 위치에 부재가 있습니다', 'warning'); return;
        }
        eng('pushUndo');
        var ni = ensureNodeAt(model, a.x, a.y, view.z);
        var nj = ensureNodeAt(model, b.x, b.y, view.z);
        // 백엔드 V2 스키마(ElementType)는 column|beam|brace만 허용 —
        // 방향별 beam_x/beam_y는 IFC 파서/뷰어 전용 표기라 여기선 'beam'.
        var eid = nextElemId(model);
        model.elements.push({
            id: eid, node_i: ni, node_j: nj, elem_type: 'beam',
            section: drawSectionFor('beam'), material: defaultMaterial(model),
            release_i: null, release_j: null, beta_angle: 0,
        });
        var splits = eng('splitElementsAtNodes', model) || 0;
        eng('refreshEditPreview');
        status('보 E' + eid + ' 생성: N' + ni + ' → N' + nj + (splits > 0 ? ' [' + splits + '개 분할]' : ''));
    }
    function createPlanColumn(s) {
        var model = getModel();
        if (!model) return;
        if (levelIndex === 0) { status('Base 레벨에는 기둥을 그릴 수 없습니다 — 상부 층 플랜에서 그리세요 (기둥 = 현재층 → 아래층)', 'warning'); return; }
        var zTop = levels[levelIndex], zBot = levels[levelIndex - 1];
        var exT = peekNodeId(model, s, zTop);
        var exB = (function () { var n = findNodeAt(model, s.x, s.y, zBot); return n ? n.id : null; })();
        if (exT !== null && exB !== null && hasElementBetween(model, exB, exT)) {
            status('이미 같은 위치에 기둥이 있습니다', 'warning'); return;
        }
        eng('pushUndo');
        var nb = ensureNodeAt(model, s.x, s.y, zBot);
        var nt = ensureNodeAt(model, s.x, s.y, zTop);
        var eid = nextElemId(model);
        model.elements.push({
            id: eid, node_i: nb, node_j: nt, elem_type: 'column',
            section: drawSectionFor('column'), material: defaultMaterial(model),
            release_i: null, release_j: null, beta_angle: 0,
        });
        var splits = eng('splitElementsAtNodes', model) || 0;
        eng('refreshEditPreview');
        status('기둥 E' + eid + ' 생성: N' + nb + '(' + fmt(zBot) + ') → N' + nt + '(' + fmt(zTop) + ')' + (splits > 0 ? ' [' + splits + '개 분할]' : ''));
    }

    function handleDrawClick(mx, my) {
        var s = snapPoint(mx, my);
        if (!s) return;
        if (drawState.mode === 'node') { createPlanNode(s); return; }
        if (drawState.type === 'column') { createPlanColumn(s); return; }
        if (!drawState.pending) {
            drawState.pending = s;
            setDrawHint('끝 절점 클릭 · 우클릭=선 끊기 · Esc=종료');
            render();
        } else {
            var a = drawState.pending;
            createPlanBeam(a, s);
            drawState.pending = s;                     // ETABS식 체인 드로잉: 끝점에서 이어 그림
        }
    }

    /* ---- draw layer: floating panel ---------------------------------------- */
    function buildDrawPanel(planCanvasWrap) {
        drawPanel = document.createElement('div');
        drawPanel.className = 'fig-draw-panel';
        drawPanel.style.display = 'none';

        var typeLab = document.createElement('span'); typeLab.className = 'fdp-label'; typeLab.textContent = '그리기';
        drawTypeSel = document.createElement('select'); drawTypeSel.className = 'fdp-select';
        [['beam', '보 (Beam)'], ['column', '기둥 (Column)']].forEach(function (o) {
            var op = document.createElement('option'); op.value = o[0]; op.textContent = o[1]; drawTypeSel.appendChild(op);
        });
        drawTypeSel.addEventListener('change', function () {
            drawState.type = drawTypeSel.value;
            drawState.pending = null;
            refreshDrawPanel(); render();
        });

        drawSectionSel = document.createElement('select'); drawSectionSel.className = 'fdp-select fdp-section';
        drawSectionSel.addEventListener('change', function () {
            drawState.sections[drawState.type === 'column' ? 'column' : 'beam'] = drawSectionSel.value;
        });

        drawLevelEl = document.createElement('span'); drawLevelEl.className = 'fdp-level';
        drawHintEl = document.createElement('span'); drawHintEl.className = 'fdp-hint';

        var exit = document.createElement('button'); exit.className = 'fdp-exit'; exit.textContent = '종료 (Esc)';
        exit.addEventListener('click', function () { try { window.setEditMode('view'); } catch (_) {} });

        drawPanel.appendChild(typeLab);
        drawPanel.appendChild(drawTypeSel);
        drawPanel.appendChild(drawSectionSel);
        drawPanel.appendChild(drawLevelEl);
        drawPanel.appendChild(drawHintEl);
        drawPanel.appendChild(exit);
        planCanvasWrap.appendChild(drawPanel);
    }
    function setDrawHint(t) { if (drawHintEl) drawHintEl.textContent = t; }
    function refreshDrawPanel() {
        if (!drawPanel) return;
        if (!drawState.mode) { drawPanel.style.display = 'none'; return; }
        drawPanel.style.display = 'flex';
        var isNode = drawState.mode === 'node';
        drawTypeSel.style.display = isNode ? 'none' : '';
        drawSectionSel.style.display = isNode ? 'none' : '';
        if (!isNode) {
            drawTypeSel.value = drawState.type;
            var secType = drawState.type === 'column' ? 'column' : 'beam';
            var cur = drawSectionFor(secType);
            drawSectionSel.innerHTML = '';
            buildSectionOptions(cur).forEach(function (o) {
                var op = document.createElement('option'); op.value = o[0]; op.textContent = o[1];
                if (o[0] === cur) op.selected = true;
                drawSectionSel.appendChild(op);
            });
        }
        var lvl = levels.length ? levelLabel(levelIndex) : '-';
        drawLevelEl.textContent = isNode ? ('절점 · ' + lvl)
            : (drawState.type === 'column' ? (lvl + ' → 아래층') : lvl);
        setDrawHint(isNode ? '빈 위치 클릭 = 절점 생성 · Esc=종료'
            : (drawState.type === 'column' ? '위치 클릭 = 기둥 생성 · Esc=종료' : '시작 절점 클릭 · Esc=종료'));
    }

    /* ---- draw layer: setEditMode wrap (auto-enable + mode sync) ------------ */
    var CREATE_MODES = { addNode: 1, addElement: 1, 'delete': 1, move: 1, release: 1, support: 1, copy: 1 };
    function onEditModeChanged(mode) {
        drawState.mode = mode === 'addNode' ? 'node' : (mode === 'addElement' ? 'frame' : null);
        drawState.pending = null; drawState.ghost = null;
        if (wrap) wrap.classList.toggle('plan-drawing', !!drawState.mode);
        refreshDrawPanel();
        updateRibbonActive(mode);
        render();
    }
    function updateRibbonActive(mode) {
        document.querySelectorAll('.ribbon-command[onclick], .rail-btn[onclick]').forEach(function (b) {
            var m = (b.getAttribute('onclick') || '').match(/setEditMode\('(\w+)'\)/);
            if (m) b.classList.toggle('active', m[1] === mode);
        });
    }
    function wrapSetEditMode() {
        if (typeof window.setEditMode !== 'function' || window.setEditMode._figmaDrawWrap) return;
        var orig = window.setEditMode;
        window.setEditMode = function (mode) {
            /* 편집(생성) 모드 진입 시 편집 게이트 자동 해제 — 엔진에선 IFC 위저드
             * step2에서만 showEditToolbar()가 켜지고, 해석 후엔 다시 꺼진다.
             * 모델이 있으면 켜 주고(결과 화면이면 프리뷰 씬 재구축), 없으면 안내. */
            if (CREATE_MODES[mode]) {
                var m = getModel();
                if (!m || !m.nodes || !m.nodes.length) {
                    status('모델이 없습니다 — File → 직접입력/IFC로 먼저 모델을 만드세요', 'warning');
                    mode = 'view';
                } else if (!window._editingEnabled) {
                    try { if (typeof showEditToolbar === 'function') showEditToolbar(); } catch (e) { console.warn('[figma-plan] showEditToolbar', e); }
                    try { if (typeof refreshEditPreview === 'function') refreshEditPreview(); } catch (e) { console.warn('[figma-plan] refreshEditPreview', e); }
                }
            }
            var r;
            try { r = orig.call(this, mode); } catch (e) { console.warn('[figma-plan] setEditMode', e); }
            try { onEditModeChanged(mode); } catch (e) { console.warn('[figma-plan] onEditModeChanged', e); }
            return r;
        };
        window.setEditMode._figmaDrawWrap = true;
    }

    /* ---- visible Properties-of-Object panel -------------------------------- */
    function setTxt(id, v) { var el = document.getElementById(id); if (el) el.textContent = v; }
    function fmtEnd(rel) { return (Array.isArray(rel) && (rel[3] || rel[4] || rel[5])) ? 'Pinned' : 'Fixed'; }
    function clearVisiblePanel() {
        setTxt('fig-obj-summary', 'Selected: (none)');
        ['fig-prop-section', 'fig-prop-material', 'fig-prop-length', 'fig-prop-release', 'fig-prop-loadgroup'].forEach(function (id) { setTxt(id, '-'); });
        selectedNodeId = null; editSelType = null; hideEditPanel();
    }
    function updatePropsPanel(sel) {
        if (!sel) { clearVisiblePanel(); return; }
        if (sel.type === 'node') {
            setTxt('fig-obj-summary', 'Selected: Node #' + sel.id);
            setTxt('fig-prop-section', '(node)'); setTxt('fig-prop-material', '-');
            setTxt('fig-prop-length', (sel.x != null) ? ('(' + (+sel.x).toFixed(2) + ', ' + (+sel.y).toFixed(2) + ', ' + (+sel.z).toFixed(2) + ') m') : '-');
            setTxt('fig-prop-release', sel.support || 'free'); setTxt('fig-prop-loadgroup', '-');
            return;
        }
        var etype = sel.elem_type || sel.type;
        var niId = (sel.node_i != null) ? sel.node_i : sel.ni;
        var njId = (sel.node_j != null) ? sel.node_j : sel.nj;
        var model = getModel(), nm = {}; ((model && model.nodes) || []).forEach(function (n) { nm[n.id] = n; });
        var ni = nm[niId], nj = nm[njId];
        var len = (ni && nj) ? Math.hypot(nj.x - ni.x, nj.y - ni.y, nj.z - ni.z) : 0;
        var mat = sel.material || (model && model.material_name);
        if (!mat) { try { mat = currentResult && currentResult.viewer && currentResult.viewer.material_name; } catch (_) {} }
        setTxt('fig-obj-summary', 'Selected: ' + typeLabelOf(etype) + ' #' + sel.id);
        setTxt('fig-prop-section', sel.section || '-');
        setTxt('fig-prop-material', mat || '-');
        setTxt('fig-prop-length', len ? len.toFixed(3) + ' m' : '-');
        setTxt('fig-prop-release', (sel.release_i == null && sel.release_j == null) ? '-' : (fmtEnd(sel.release_i) + '-' + fmtEnd(sel.release_j)));
        setTxt('fig-prop-loadgroup', '-');
    }

    /* ---- unified selection (plan + 3D share native highlight tracking) ----- */
    function sync3DHighlight(id) {
        try {
            if (typeof memberMeshes === 'undefined' || !memberMeshes) return;
            for (var i = 0; i < memberMeshes.length; i++) {
                var md = memberMeshes[i];
                var ed = md.elementData || (md.mesh && md.mesh.userData && md.mesh.userData.elementData);
                if (ed && ed.id === id) {
                    if (typeof highlightMesh === 'function') highlightMesh(md.mesh);
                    try { selectedMesh = md.mesh; } catch (_) {}
                    try { if (typeof selectedMeshSet !== 'undefined' && selectedMeshSet) selectedMeshSet.add(md.mesh); } catch (_) {}
                    return;
                }
            }
        } catch (e) { console.warn('[figma-plan] 3D highlight', e); }
    }
    // plan click entry (elements only)
    function selectElement(e) {
        try { if (typeof clearAllSelection === 'function') clearAllSelection(); } catch (_) {}   // wrapped: clears plan+panel+3D
        if (e) {
            sync3DHighlight(e.id);
            selectedElemId = e.id; editSelType = 'element';
            updatePropsPanel(e); render();
            buildEditPanel(e, null);
        }
        window._figmaSelectedElemId = selectedElemId;
    }
    // 3D click entry (via showMemberProperties wrap; native highlight already applied)
    function figmaOnSelect3D(elemData) {
        if (!elemData) return;
        var model = getModel();
        if (elemData.type === 'node') {
            selectedElemId = null; selectedNodeId = elemData.id; editSelType = 'node';
            var nn = model ? (model.nodes || []).find(function (x) { return x.id === elemData.id; }) : null;
            var nd = nn ? { type: 'node', id: nn.id, x: nn.x, y: nn.y, z: nn.z, support: nn.support } : elemData;
            updatePropsPanel(nd); render(); buildEditPanel(null, nd);
        } else {
            var me = model ? (model.elements || []).find(function (x) { return x.id === elemData.id; }) : null;
            selectedElemId = elemData.id; selectedNodeId = null; editSelType = 'element';
            updatePropsPanel(me || elemData); render(); buildEditPanel(me || elemData, null);
        }
        window._figmaSelectedElemId = selectedElemId;
    }

    /* ---- selection-based editing (Section / Release / Support / Delete) ---- */
    function setStub(id, v) { var el = document.getElementById(id); if (el) el.value = v; }
    function runBulk(fnName) {
        try { if (typeof window[fnName] === 'function') window[fnName](); }
        catch (e) { console.warn('[figma-plan] ' + fnName, e); }
        afterEdit();
    }
    function afterEdit() {
        var model = getModel();
        if (editSelType === 'element' && selectedElemId != null && model) {
            var me = (model.elements || []).find(function (x) { return x.id === selectedElemId; });
            if (me) { selectElement(me); return; }              // re-establish selection on fresh meshes
        }
        if (editSelType === 'node' && selectedNodeId != null && model) {
            var nn = (model.nodes || []).find(function (x) { return x.id === selectedNodeId; });
            if (nn) { var nd = { type: 'node', id: nn.id, x: nn.x, y: nn.y, z: nn.z, support: nn.support }; updatePropsPanel(nd); render(); buildEditPanel(null, nd); return; }
        }
        selectElement(null);                                    // item deleted -> clear
    }
    function buildSectionOptions(current) {
        var opts = [], seen = {};
        try {
            var sl = (typeof sectionsList !== 'undefined') ? sectionsList : null;
            if (sl) Object.keys(sl).forEach(function (t) { (sl[t] || []).forEach(function (nm) { if (!seen[nm]) { seen[nm] = 1; opts.push([nm, nm]); } }); });
        } catch (_) {}
        if (current && !seen[current]) opts.unshift([current, current + ' (현재)']);
        if (!opts.length) opts = [['', '(단면 목록 없음)']];
        return opts;
    }
    function buildMaterialOptions(current) {
        var opts = [], seen = {};
        try {
            var names = (typeof window._figmaMaterialNames === 'function') ? window._figmaMaterialNames() : [];
            names.forEach(function (nm) { if (!seen[nm]) { seen[nm] = 1; opts.push([nm, nm]); } });
        } catch (_) {}
        if (current && !seen[current]) opts.unshift([current, current + ' (현재)']);
        if (!opts.length) opts = [['', '(재료 목록 없음)']];
        return opts;
    }
    function makeSelectRow(label, options, current, onchange) {
        var row = document.createElement('div'); row.className = 'fig-edit-row';
        var lab = document.createElement('span'); lab.className = 'fig-edit-label'; lab.textContent = label; row.appendChild(lab);
        var s = document.createElement('select'); s.className = 'fig-edit-select';
        options.forEach(function (o) { var op = document.createElement('option'); op.value = o[0]; op.textContent = o[1]; if (o[0] === current) op.selected = true; s.appendChild(op); });
        s.addEventListener('change', function () { onchange(s.value); });
        row.appendChild(s); return row;
    }
    function hideEditPanel() { if (editPanel) { editPanel.style.display = 'none'; editPanel.innerHTML = ''; } }
    function buildEditPanel(elem, node) {
        if (!editPanel) return;
        editPanel.innerHTML = '';
        if (!elem && !node) { editPanel.style.display = 'none'; return; }
        editPanel.style.display = '';
        var head = document.createElement('div'); head.className = 'fig-edit-head'; head.textContent = '편집'; editPanel.appendChild(head);
        if (elem) {
            editPanel.appendChild(makeSelectRow('단면', buildSectionOptions(elem.section), elem.section || '', function (v) { if (v) { setStub('bulk-section', v); runBulk('bulkApplySection'); } }));
            editPanel.appendChild(makeSelectRow('재료', buildMaterialOptions(elem.material), elem.material || '', function (v) { if (v) { setStub('bulk-material', v); runBulk('bulkApplyMaterial'); } }));
            editPanel.appendChild(makeSelectRow('릴리즈', [['', '(변경 안 함)'], ['fixed', 'Fixed–Fixed'], ['pin_i', 'Pin i'], ['pin_j', 'Pin j'], ['pin_both', 'Pin 양단']], '', function (v) { if (v) { setStub('bulk-release', v); runBulk('bulkApplyRelease'); } }));
        } else if (node) {
            editPanel.appendChild(makeSelectRow('지점', [['free', 'Free'], ['fixed', 'Fixed'], ['pinned', 'Pinned']], node.support || 'free', function (v) { setStub('bulk-support', v); runBulk('bulkApplySupport'); }));
        }
        var b = document.createElement('button'); b.className = 'fig-edit-del'; b.textContent = '삭제 (Delete)';
        b.addEventListener('click', function () { runBulk('bulkDeleteSelected'); });
        editPanel.appendChild(b);
    }

    function installWraps() {
        if (typeof window.clearAllSelection === 'function' && !window.clearAllSelection._figmaWrap) {
            var oc = window.clearAllSelection;
            window.clearAllSelection = function () {
                var r; try { r = oc.apply(this, arguments); } catch (e) { console.warn('[figma-plan] clearAllSelection', e); }
                try { selectedElemId = null; window._figmaSelectedElemId = null; clearVisiblePanel(); render(); } catch (_) {}
                return r;
            };
            window.clearAllSelection._figmaWrap = true;
        }
        if (typeof window.showMemberProperties === 'function' && !window.showMemberProperties._figmaWrap) {
            var os = window.showMemberProperties;
            window.showMemberProperties = function (elem) {
                var r; try { r = os.apply(this, arguments); } catch (e) { console.warn('[figma-plan] showMemberProperties', e); }
                try { figmaOnSelect3D(elem); } catch (_) {}
                return r;
            };
            window.showMemberProperties._figmaWrap = true;
        }
    }

    /* ---- interaction ------------------------------------------------------- */
    function bindInteractions() {
        canvas.addEventListener('mousemove', function (ev) {
            var r = canvas.getBoundingClientRect();
            if (drawState.mode) {                      // 그리기 중: 스냅 고스트만 (툴팁 억제)
                if (_hoverElemId != null) { _hoverElemId = null; try { hoveredMesh = null; hideHoverTooltip(); } catch (_) {} }
                drawState.ghost = snapPoint(ev.clientX - r.left, ev.clientY - r.top);
                render();
                return;
            }
            var h = hitTest(ev.clientX - r.left, ev.clientY - r.top);
            if (h) {
                canvas.style.cursor = 'pointer';
                if (_hoverElemId !== h.e.id) {
                    _hoverElemId = h.e.id;
                    try { hoveredMesh = HOVER_SENTINEL; } catch (_) {}
                    try { showHoverTooltip(buildElem(h.e), ev.clientX, ev.clientY); } catch (err) { console.warn('[figma-plan] tooltip', err); }
                } else { try { positionHoverTooltip(ev.clientX, ev.clientY); } catch (_) {} }
            } else {
                canvas.style.cursor = '';
                if (_hoverElemId != null) { _hoverElemId = null; try { hoveredMesh = null; hideHoverTooltip(); } catch (_) {} }
            }
        });
        canvas.addEventListener('mouseleave', function () {
            _hoverElemId = null; canvas.style.cursor = '';
            try { hoveredMesh = null; hideHoverTooltip(); } catch (_) {}
            if (drawState.mode) { drawState.ghost = null; render(); }
        });
        canvas.addEventListener('click', function (ev) {
            var r = canvas.getBoundingClientRect();
            if (drawState.mode) { handleDrawClick(ev.clientX - r.left, ev.clientY - r.top); return; }
            var h = hitTest(ev.clientX - r.left, ev.clientY - r.top);
            selectElement(h ? h.e : null);
        });
        canvas.addEventListener('contextmenu', function (ev) {
            if (!drawState.mode) return;
            ev.preventDefault();
            if (drawState.pending) {                   // 우클릭 1회: 체인 끊기
                drawState.pending = null;
                refreshDrawPanel(); render();
            } else {                                   // 우클릭 2회: 그리기 종료
                try { window.setEditMode('view'); } catch (_) {}
            }
        });
    }

    /* ---- story selector ---------------------------------------------------- */
    function rebuildLevelMenu() {
        if (!levelMenu) return;
        levelMenu.innerHTML = '';
        levels.forEach(function (z, i) {
            var it = document.createElement('div');
            it.className = 'plan-story-item' + (i === levelIndex ? ' active' : '');
            it.textContent = levelLabel(i);
            it.addEventListener('click', function (e) {
                e.stopPropagation(); levelIndex = i;
                levelBtn.firstChild.textContent = levelLabel(i) + ' ';
                drawState.pending = null; drawState.ghost = null;    // 레벨 바뀌면 그리던 선 무효
                refreshDrawPanel();
                closeLevelMenu(); render(); rebuildLevelMenu();
            });
            levelMenu.appendChild(it);
        });
    }
    function openLevelMenu() { if (levelMenu) { rebuildLevelMenu(); levelMenu.classList.add('open'); } }
    function closeLevelMenu() { if (levelMenu) levelMenu.classList.remove('open'); }

    /* 테스트/디버그: 현재 플랜 변환(월드→스크린) 파라미터 조회 */
    window._figmaPlanView = function () {
        return view ? { minX: view.minX, minY: view.minY, scale: view.scale, offX: view.offX, offY: view.offY, W: view.W, H: view.H, z: view.z, levelIndex: levelIndex } : null;
    };

    /* ---- refresh + engine render hooks ------------------------------------ */
    window.figmaPlanRefresh = function () {
        var model = getModel();
        levels = model ? computeLevels(model) : [];
        if (levelIndex >= levels.length) levelIndex = 0;
        if (levelBtn) levelBtn.firstChild.textContent = (levels.length ? levelLabel(levelIndex) : 'Base : Z = 0.00') + ' ';
        if (selectedElemId != null && model && !(model.elements || []).some(function (e) { return e.id === selectedElemId; })) {
            selectedElemId = null; updatePropsPanel(null);
        }
        rebuildLevelMenu(); refreshDrawPanel(); render();
    };
    function wrapRenderer(name) {
        var orig = window[name];
        if (typeof orig !== 'function' || orig._figmaPlanWrapped) return;
        window[name] = function () { var out = orig.apply(this, arguments); try { window.figmaPlanRefresh(); } catch (e) { console.warn('[figma-plan]', e); } return out; };
        window[name]._figmaPlanWrapped = true;
    }

    function init() {
        var planWindow = document.querySelector('.plan-window');
        if (!planWindow) return;
        var title = planWindow.querySelector('.viewport-title');
        var planCanvas = planWindow.querySelector('.plan-canvas');
        if (!title || !planCanvas) return;

        titleH3 = title.querySelector('h3');
        if (titleH3) titleH3.textContent = 'Plan View';
        var sel = document.createElement('div'); sel.className = 'plan-story-select';
        levelBtn = document.createElement('button'); levelBtn.className = 'plan-story-btn';
        levelBtn.appendChild(document.createTextNode('Base : Z = 0.00 '));
        var caret = document.createElement('span'); caret.className = 'plan-caret'; caret.textContent = '▾';
        levelBtn.appendChild(caret);
        levelMenu = document.createElement('div'); levelMenu.className = 'plan-story-menu';
        sel.appendChild(levelBtn); sel.appendChild(levelMenu);
        if (titleH3 && titleH3.nextSibling) title.insertBefore(sel, titleH3.nextSibling); else title.appendChild(sel);
        // plan solid(실폭) 토글 — 스토리 셀렉터 옆
        var solidLab = document.createElement('label');
        solidLab.className = 'vp-solid';
        solidLab.title = '부재 실폭(단면) 표시';
        var solidChk = document.createElement('input');
        solidChk.type = 'checkbox';
        solidChk.id = 'fig-plan-solid';
        solidChk.addEventListener('change', function () { planSolid = solidChk.checked; render(); });
        solidLab.appendChild(solidChk);
        solidLab.appendChild(document.createTextNode(' Solid'));
        sel.parentNode.insertBefore(solidLab, sel.nextSibling);
        levelBtn.addEventListener('click', function (e) { e.stopPropagation(); if (levelMenu.classList.contains('open')) closeLevelMenu(); else openLevelMenu(); });
        levelMenu.addEventListener('mousedown', function (e) { e.stopPropagation(); });
        document.addEventListener('mousedown', closeLevelMenu);

        planCanvas.innerHTML = '';
        planCanvas.classList.add('plan-live');
        wrap = planCanvas;
        canvas = document.createElement('canvas'); canvas.className = 'plan-live-canvas';
        planCanvas.appendChild(canvas);
        ctx = canvas.getContext('2d');
        bindInteractions();

        var objPanel = document.querySelector('.figma-object-panel');
        if (objPanel) { editPanel = document.createElement('div'); editPanel.id = 'fig-edit-panel'; editPanel.style.display = 'none'; objPanel.appendChild(editPanel); }

        buildDrawPanel(planCanvas);
        wrapRenderer('buildV2PreviewScene');
        wrapRenderer('buildScene');
        installWraps();
        wrapSetEditMode();

        if (window.ResizeObserver) new ResizeObserver(function () { render(); }).observe(planCanvas);
        window.addEventListener('resize', render);
        window.figmaPlanRefresh();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
