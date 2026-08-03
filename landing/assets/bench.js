/* ==========================================================================
   OpenSees-MCP landing — "the bench"

   This file renders baked results. It computes NOTHING structural: every
   number it shows was produced by scripts/bake_demo_bundle.py running the
   real analysis pipeline, and is read verbatim out of the inlined bundle.
   ========================================================================== */
(function () {
  "use strict";

  var el = document.getElementById("demo-bundle");
  if (!el) return;

  var BUNDLE;
  try {
    BUNDLE = JSON.parse(el.textContent);
  } catch (e) {
    console.error("demo bundle parse failed", e);
    return;
  }

  var manifest = BUNDLE.manifest;
  var variants = BUNDLE.variants;          // { "H-300x300": {...}, ... }
  var order = manifest.ladder.map(function (r) { return r.variant; });
  var DEFAULT = "H-300x300";

  // ── helpers ──────────────────────────────────────────────────────────
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function bind(name) { return $$('[data-bind="' + name + '"]'); }
  function setText(name, value) {
    bind(name).forEach(function (n) { n.textContent = value; });
  }
  function fmt(v, nd) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    return Number(v).toFixed(nd === undefined ? 4 : nd);
  }

  // ── static bits ──────────────────────────────────────────────────────
  setText("commit", manifest.git_commit);
  setText("generated-at", (manifest.generated_at || "").slice(0, 10));

  // ── live-app links ───────────────────────────────────────────────────
  // The landing is static and may be hosted apart from the app, so the app
  // origin comes from one <meta> tag rather than being hardcoded per link.
  (function wireAppLinks() {
    var meta = document.querySelector('meta[name="app-base"]');
    var base = (meta && meta.getAttribute("content") || "").replace(/\/+$/, "");
    $$("[data-app-path]").forEach(function (a) {
      a.href = base + a.getAttribute("data-app-path");
    });
  })();

  // ── the section dropdown ─────────────────────────────────────────────
  var select = $("#sect");
  order.forEach(function (name) {
    var opt = document.createElement("option");
    opt.value = name;
    var row = manifest.ladder.filter(function (r) { return r.variant === name; })[0];
    opt.textContent = name + "   (" + row.overall_status + " · " + fmt(row.max_interaction_ratio, 3) + ")";
    select.appendChild(opt);
  });

  // ── nudge: point at the first variant that flips the verdict to OK ───
  var firstOK = manifest.ladder.filter(function (r) { return r.overall_status === "OK"; })[0];

  function nudgeFor(current) {
    if (!firstOK) return "";
    if (current === DEFAULT) return "↳ " + firstOK.variant + "(으)로 바꿔 보세요";
    if (current === firstOK.variant) return "판정이 뒤집혔습니다";
    return "";
  }

  // ── the closing line under the card ──────────────────────────────────
  // Written from the baked ladder, not by hand: at the point the verdict
  // flips, the governing member stops being a column.
  function footFor(rec) {
    var worst = rec.members.reduce(function (a, b) {
      return (b.interaction || 0) > (a.interaction || 0) ? b : a;
    }, rec.members[0] || {});
    var kind = worst.type === "column" ? "기둥" :
               worst.type === "beam_x" ? "X방향 보" :
               worst.type === "beam_y" ? "Y방향 보" : worst.type;
    var W = (rec.load_chain.seismic || {}).W_kN;
    var V = (rec.load_chain.seismic || {}).V_kN;
    return "지배 부재는 " + worst.story + "층 " + kind +
           " (상관비 " + fmt(worst.interaction, 4) + ").  " +
           "지진질량은 바닥 고정하중 기준이라 단면을 바꿔도 총중량 " + fmt(W, 1) +
           " kN · 밑면전단 " + fmt(V, 1) + " kN은 그대로입니다.";
  }

  // ── render ───────────────────────────────────────────────────────────
  function render(name) {
    var rec = variants[name];
    if (!rec) return;

    // verdict block
    var verdict = $(".verdict");
    verdict.setAttribute("data-status", rec.overall_status);
    setText("overall", rec.overall_status);
    setText("ng-groups", rec.checks.filter(function (c) { return c.status === "NG"; }).length);

    var ms = rec.member_summary || {};
    setText("total", ms.total);
    setText("ng-members", ms.ng);
    setText("max-ratio", fmt(ms.max_interaction_ratio));
    var memberRow = rec.checks.filter(function (c) { return c.key === "member_check"; })[0];
    setText("member-code-ref", memberRow ? memberRow.code_ref : "");
    setText("governing-combo", rec.governing_combo || "—");

    // limits section numbers
    var seis = rec.load_chain.seismic || {};
    setText("W", fmt(seis.W_kN, 1));
    setText("V", fmt(seis.V_kN, 1));

    // frame colouring
    var byId = {};
    rec.members.forEach(function (m) { byId[m.id] = m; });
    $$(".mb").forEach(function (line) {
      var m = byId[Number(line.getAttribute("data-m"))];
      line.classList.remove("is-ng", "is-hot");
      if (!m) return;
      if (m.status === "NG") line.classList.add("is-ng");
      else if ((m.interaction || 0) >= 0.9) line.classList.add("is-hot");
      var kind = m.type === "column" ? "기둥" : m.type === "beam_x" ? "X방향 보" : "Y방향 보";
      line.setAttribute("aria-label",
        m.story + "층 " + kind + " · 상관비 " + fmt(m.interaction, 3) + " · " + m.status);
    });

    // check table
    var body = $("#checks-body");
    body.innerHTML = "";
    rec.checks.forEach(function (c) {
      var tr = document.createElement("tr");
      var status = c.status || "none";
      tr.innerHTML =
        '<td>' + c.label + '</td>' +
        '<td><span class="pill pill--' + status + '">' + (c.status || "미평가") + '</span></td>' +
        '<td class="num">' + (c.ratio === null || c.ratio === undefined ? "—" : fmt(c.ratio, 4)) + '</td>' +
        '<td class="num">' + (c.limit === null || c.limit === undefined ? "—" : c.limit) + '</td>' +
        '<td class="code">' + (c.code_ref || "—") + '</td>' +
        '<td><button class="dispute" type="button" data-anchor="' + c.key + '">이건 다릅니다</button></td>';
      body.appendChild(tr);
    });

    setText("nudge", nudgeFor(name));
    setText("foot", footFor(rec));
  }

  // ?section=H-350x350 makes a given state linkable — useful when someone
  // wants to point at the exact case they are disputing.
  function initialSection() {
    var q = new URLSearchParams(window.location.search).get("section");
    return (q && variants[q]) ? q : DEFAULT;
  }

  var start = initialSection();
  select.value = start;
  select.addEventListener("change", function () {
    render(select.value);
    var url = new URL(window.location.href);
    if (select.value === DEFAULT) url.searchParams.delete("section");
    else url.searchParams.set("section", select.value);
    history.replaceState(null, "", url);
  });
  render(start);

  // Inline dispute buttons — wiring lands with the feedback phase; for now
  // make sure a click is never a dead end.
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest && ev.target.closest(".dispute");
    if (!btn) return;
    var target = document.getElementById("feedback");
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
