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
  var fileSizes = BUNDLE.files || {};      // measured at build time
  var order = manifest.ladder.map(function (r) { return r.variant; });
  // The ladder from the manifest is authoritative; the named baseline is only a
  // preference. A re-bake with a different section list must not leave every
  // render() call returning early against a variant that no longer exists.
  var DEFAULT = Object.prototype.hasOwnProperty.call(variants, "H-300x300")
    ? "H-300x300" : order[0];

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

  // Download sizes come from a stat() at build time, so they cannot drift away
  // from the files actually sitting next to this page.
  (function fillFileSizes() {
    function kb(bytes) { return Math.round(bytes / 1024).toLocaleString() + " KB"; }
    $$("[data-file]").forEach(function (n) {
      var b = fileSizes[n.getAttribute("data-file")];
      if (b) n.textContent = kb(b);
    });
    var rep = $("#report-size");
    if (rep && fileSizes["report_example.html"]) {
      rep.textContent = kb(fileSizes["report_example.html"]);
    }
  })();

  // ── live-app links ───────────────────────────────────────────────────
  // The landing is static and may be hosted apart from the app, so the app
  // origin comes from one <meta> tag rather than being hardcoded per link.
  function appBase() {
    var meta = document.querySelector('meta[name="app-base"]');
    return (meta && meta.getAttribute("content") || "").replace(/\/+$/, "");
  }

  (function wireAppLinks() {
    var base = appBase();
    $$("[data-app-path]").forEach(function (a) {
      a.href = base + a.getAttribute("data-app-path");
    });
  })();

  // The primary CTA promises "run this and get these numbers". Every figure
  // around it follows the dropdown, so the link has to as well — otherwise
  // moving the knob to the section that passes and then clicking through
  // opens the section that fails, which is the precise contradiction this
  // page was rebuilt to remove. Non-default variants ride along as ?col=,
  // which figma_deeplink.js prefills.
  function syncRunCta(name) {
    var cta = $("#run-cta");
    if (!cta) return;
    var path = cta.getAttribute("data-app-path") || "/editor-figma?demo=bench";
    cta.href = appBase() + path + (name === DEFAULT ? "" : "&col=" + encodeURIComponent(name));
    cta.textContent = (name === DEFAULT)
      ? "이 모델 직접 돌려보기 →"
      : name + "(으)로 직접 돌려보기 →";
  }

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
           " kN · 밑면전단 " + fmt(V, 2) + " kN은 그대로입니다.";
  }

  // ── load receipt ─────────────────────────────────────────────────────
  // One row per field the load generator actually recorded. The field name is
  // printed next to the label on purpose: it is the difference between "trust
  // us" and "here is the record you can grep for".
  //   [field, label, unit, role]   role: "" | "in" | "out" | "why"
  var RECEIPTS = [
    {
      key: "seismic", title: "지진하중", code: "KDS 41 17 00", tag: "ELF",
      rows: [
        ["region", "지역", "", "in"],
        ["z", "지진구역계수 z", "g", "in"],
        ["site_class", "지반종류", "", "in"],
        ["IE", "중요도계수 IE", "", "in"],
        ["Fa", "단주기 증폭계수 Fa", "", ""],
        ["Fv", "장주기 증폭계수 Fv", "", ""],
        ["SDS", "설계스펙트럼가속도 SDS", "g", ""],
        ["SD1", "설계스펙트럼가속도 SD1", "g", ""],
        ["seismic_system", "횡력저항시스템", "", "in"],
        ["R", "반응수정계수 R", "", ""],
        ["Cd", "변위증폭계수 Cd", "", ""],
        ["Ct", "주기계수 Ct", "", ""],
        ["hn_m", "건물높이 hn", "m", "in"],
        ["Ta_sec", "근사고유주기 Ta", "s", ""],
        ["Cs", "지진응답계수 Cs", "", ""],
        ["Cs_min", "Cs 하한", "", ""],
        ["Cs_max", "Cs 상한", "", ""],
        ["Cs_governed_by", "Cs 지배", "", "why"],
        ["W_kN", "유효건물중량 W", "kN", ""],
        ["V_kN", "밑면전단력 V = Cs·W", "kN", "out"]
      ]
    },
    {
      key: "wind", title: "풍하중", code: "KDS 41 12 00 §5", tag: "간편법",
      rows: [
        ["region", "지역", "", "in"],
        ["V0_ms", "기본풍속 V0", "m/s", "in"],
        ["exposure", "지표면조도구분", "", "in"],
        ["Kzt", "지형계수 Kzt", "", ""],
        ["Gf", "가스트영향계수 Gf", "", ""],
        ["Cp_total", "풍력계수 Cp (풍상+풍하)", "", ""],
        ["Bx_m", "수풍면 폭 Bx", "m", ""],
        ["By_m", "수풍면 폭 By", "m", ""],
        ["total_Fx_kN", "X방향 총 풍하중", "kN", "out"],
        ["total_Fy_kN", "Y방향 총 풍하중", "kN", "out"]
      ]
    },
    {
      key: "snow", title: "설하중", code: "KDS 41 12 00 §4", tag: "평지붕",
      rows: [
        ["region", "지역", "", "in"],
        ["Sg_kNm2", "기본지상적설하중 Sg", "kN/m²", "in"],
        ["Cb", "기본지붕적설계수 Cb", "", ""],
        ["Ce", "노출계수 Ce", "", ""],
        ["Ct", "온도계수 Ct", "", ""],
        ["Is", "중요도계수 Is", "", ""],
        ["S_formula_kNm2", "산정식 결과 S", "kN/m²", ""],
        ["S_min_kNm2", "최소 적설하중", "kN/m²", ""],
        ["governed_by", "지배", "", "why"],
        ["S_design_kNm2", "설계 적설하중", "kN/m²", "out"],
        ["roof_area_m2", "지붕면적", "m²", ""]
      ]
    }
  ];

  function receiptValue(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v === "number") {
      // Integers stay integers; everything else keeps the baked precision
      // rather than being re-rounded here.
      return Number.isInteger(v) ? String(v) : String(v);
    }
    return String(v);
  }

  function renderReceipts(rec) {
    var host = $("#receipts");
    if (!host) return;
    host.innerHTML = "";
    RECEIPTS.forEach(function (spec) {
      var data = (rec.load_chain || {})[spec.key];
      if (!data) return;

      var card = document.createElement("section");
      card.className = "receipt";
      var rows = spec.rows.filter(function (r) { return data[r[0]] !== undefined; });

      var html =
        '<header class="receipt-head">' +
          '<h3>' + spec.title + '</h3>' +
          '<span class="receipt-code">' + spec.code + '</span>' +
          '<span class="receipt-tag">' + spec.tag + '</span>' +
        '</header><dl class="receipt-rows">';

      rows.forEach(function (r) {
        var field = r[0], label = r[1], unit = r[2], role = r[3];
        html +=
          '<div class="rrow' + (role ? " rrow--" + role : "") + '">' +
            '<dt>' + label + '<code>' + field + '</code></dt>' +
            '<dd>' + receiptValue(data[field]) +
              (unit ? '<span class="ru">' + unit + '</span>' : '') +
            '</dd>' +
          '</div>';
      });

      card.innerHTML = html + "</dl>";
      host.appendChild(card);
    });
  }

  function renderStoryForces(rec) {
    var tb = $("#story-forces tbody");
    if (!tb) return;
    tb.innerHTML = "";
    var rows = (rec.load_chain || {}).seismic_story_forces || [];
    rows.slice().reverse().forEach(function (s) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + s.story + "층</td>" +
        '<td class="num">' + fmt(s.weight_kN, 1) + "</td>" +
        '<td class="num">' + fmt(s.height_m, 1) + "</td>" +
        '<td class="num">' + fmt(s.Cvx, 4) + "</td>" +
        '<td class="num">' + fmt(s.Fx_kN, 2) + "</td>";
      tb.appendChild(tr);
    });
  }

  // ── load combinations ────────────────────────────────────────────────
  // Grouped by what actually appears in the combination name, not by a
  // hand-kept list — if the generator adds a family the count follows it.
  function comboFamily(name) {
    if (/EQ[XY]/.test(name)) return "seismic";
    if (/W[XY]/.test(name)) return "wind";
    return "gravity";
  }

  function renderComboMix(rec) {
    var host = $("#combo-mix");
    if (!host) return;
    var combos = rec.combos || [];
    var counts = { gravity: 0, wind: 0, seismic: 0 };
    combos.forEach(function (c) { counts[comboFamily(c)] += 1; });

    var labels = [
      ["gravity", "중력 (DL·LL·S)"],
      ["wind", "풍하중 포함"],
      ["seismic", "지진 포함 (직교 100/30)"]
    ];
    host.innerHTML = "";
    labels.forEach(function (l) {
      var d = document.createElement("div");
      d.className = "mix mix--" + l[0];
      d.innerHTML = '<strong>' + counts[l[0]] + '</strong><span>' + l[1] + "</span>";
      host.appendChild(d);
    });
  }

  function renderCombos(rec) {
    renderComboMix(rec);
    var list = $("#combo-list");
    if (!list) return;
    var combos = rec.combos || [];
    setText("combo-count", combos.length);
    list.innerHTML = "";
    combos.forEach(function (c) {
      var li = document.createElement("li");
      li.className = "combo" + (c === rec.governing_combo ? " is-gov" : "");
      li.innerHTML = "<code>" + c + "</code>" +
        (c === rec.governing_combo ? '<span class="gov-tag">부재강도 지배</span>' : "");
      list.appendChild(li);
    });
  }

  // ── member roll ──────────────────────────────────────────────────────
  var MEMBER_ROWS_SHOWN = 20;
  var memberFilter = "all";
  var expandedMembers = false;

  function memberKind(m) {
    return m.type === "column" ? "기둥" :
           m.type === "beam_x" ? "X방향 보" :
           m.type === "beam_y" ? "Y방향 보" : (m.type || "—");
  }

  function memberMatches(m) {
    if (memberFilter === "all") return true;
    if (memberFilter === "NG") return m.status === "NG";
    if (memberFilter === "column") return m.type === "column";
    if (memberFilter === "beam") return m.type === "beam_x" || m.type === "beam_y";
    return true;
  }

  function renderMembers(rec) {
    var body = $("#members-body");
    if (!body) return;
    // The bake already sorts by interaction descending; sorting a copy keeps
    // that guarantee even if a future bake stops doing it.
    var all = (rec.members || []).slice().sort(function (a, b) {
      return (b.interaction || 0) - (a.interaction || 0);
    });
    var rows = all.filter(memberMatches);
    var limit = expandedMembers ? rows.length : MEMBER_ROWS_SHOWN;

    body.innerHTML = "";
    rows.slice(0, limit).forEach(function (m) {
      var tr = document.createElement("tr");
      if (m.status === "NG") tr.className = "is-ng";
      tr.innerHTML =
        "<td>#" + m.id + ' <span class="muted">' + memberKind(m) + "</span></td>" +
        "<td>" + (m.story == null ? "—" : m.story + "층") + "</td>" +
        '<td class="code">' + (m.section || "—") + "</td>" +
        '<td><span class="pill pill--' + (m.status || "none") + '">' + (m.status || "—") + "</span></td>" +
        '<td class="num strong">' + fmt(m.interaction) + "</td>" +
        '<td class="num">' + fmt(m.axial) + "</td>" +
        '<td class="num">' + fmt(m.shear) + "</td>" +
        '<td class="code combo-cell">' + (m.governing_combo || "—") + "</td>";
      body.appendChild(tr);
    });

    var ngCount = all.filter(function (m) { return m.status === "NG"; }).length;
    var countEl = $("#member-count");
    if (countEl) {
      // Count what is on screen, not what passed the filter — the two differ
      // until the visitor expands, and claiming 63 while showing 20 is exactly
      // the kind of small lie this page exists to avoid.
      countEl.textContent =
        Math.min(limit, rows.length) + " / " + rows.length + "행 표시 · 전체 " +
        all.length + "개 중 NG " + ngCount + "개";
    }

    var more = $("#members-more");
    if (more) {
      more.innerHTML = "";
      if (rows.length > MEMBER_ROWS_SHOWN) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "more-btn";
        btn.textContent = expandedMembers
          ? "접기"
          : "나머지 " + (rows.length - MEMBER_ROWS_SHOWN) + "개 더 보기";
        btn.addEventListener("click", function () {
          expandedMembers = !expandedMembers;
          renderMembers(rec);
        });
        more.appendChild(btn);
      }
    }
  }

  // ── modal ────────────────────────────────────────────────────────────
  var DIRECTION_KO = {
    "TRAN-X": "X방향 병진", "TRAN-Y": "Y방향 병진", "ROTN-Z": "Z축 회전"
  };

  function renderModal(rec) {
    var md = rec.modal || {};
    var cards = $("#modal-cards");
    if (cards) {
      var fp = md.fundamental_periods || {};
      var cp = md.cumulative_participation || {};
      var items = [
        ["T₁ X방향", fp.T1_x_s, "s", cp.x_pct],
        ["T₁ Y방향", fp.T1_y_s, "s", cp.y_pct],
        ["T₁ Z축 회전", fp.T1_rz_s, "s", cp.rz_pct]
      ];
      cards.innerHTML = "";
      items.forEach(function (it) {
        var d = document.createElement("div");
        d.className = "mcard";
        d.innerHTML =
          '<span class="mcard-k">' + it[0] + "</span>" +
          '<strong class="mcard-v">' + fmt(it[1], 4) + '<span class="ru">' + it[2] + "</span></strong>" +
          '<span class="mcard-s">누적 질량참여 ' + fmt(it[3], 1) + "%</span>";
        cards.appendChild(d);
      });
    }

    var tb = $("#mode-table tbody");
    if (tb) {
      tb.innerHTML = "";
      (md.modes || []).forEach(function (m) {
        var mp = m.mass_participation || {};
        var tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + m.mode + "</td>" +
          '<td class="num">' + fmt(m.period_s, 4) + "</td>" +
          "<td>" + (DIRECTION_KO[m.direction] || m.direction || "—") +
            ' <code class="muted">' + (m.direction || "") + "</code></td>" +
          '<td class="num">' + fmt(m.dominance_pct, 2) + "</td>" +
          '<td class="num">' + fmt(mp.x_pct, 2) + "</td>" +
          '<td class="num">' + fmt(mp.y_pct, 2) + "</td>" +
          '<td class="num">' + fmt(mp.rz_pct, 2) + "</td>";
        tb.appendChild(tr);
      });
    }

    var foot = $("#modal-foot");
    if (foot) {
      var shown = (md.modes || []).length;
      var solved = md.num_modes;
      var mass = md.mass_info || {};
      // Say plainly that the table is a subset — a reader who counts rows and
      // finds fewer than num_modes should not have to wonder which is wrong.
      // Both counts are optional in the bundle (extract_modal returns {} when
      // there was no modal analysis), so neither is interpolated unguarded.
      var lead = (solved && shown)
        ? "해석은 <b>" + solved + "모드</b>를 풀었고 위 표는 상위 " + shown + "개입니다. "
        : (shown ? "위 표는 " + shown + "개 모드입니다. " : "");
      foot.innerHTML = lead +
        "질량 산정 기준 <code>" + (mass.basis || "—") + "</code>" +
        (mass.rigid_diaphragm ? " · 강막 적용" : "") +
        (mass.includes_member_self_weight === false
          ? " · <b>부재 자중 미포함</b>" : "") + ".";
    }
  }

  // ── feedback ─────────────────────────────────────────────────────────
  var FEEDBACK_TO = "mk0010@khu.ac.kr";

  function mailtoHref(topic, checkRow) {
    var rec = variants[select.value] || {};
    var ms = rec.member_summary || {};
    var lines = [
      "지적 내용:",
      "",
      "",
      "— 아래는 어느 화면을 보고 계셨는지 알려주는 정보입니다 (지워도 됩니다) —",
      "기둥 단면: " + select.value,
      "판정: " + (rec.overall_status || "-") +
        " · 최대 상관비 " + fmt(ms.max_interaction_ratio) +
        " · NG " + (ms.ng == null ? "-" : ms.ng) + "/" + (ms.total == null ? "-" : ms.total),
    ];
    if (checkRow) {
      lines.push("해당 검토: " + checkRow.label +
        " (" + (checkRow.status || "미평가") + ", " + (checkRow.code_ref || "-") + ")");
    }
    lines.push("bake: " + manifest.git_commit + " / " + (manifest.generated_at || "").slice(0, 10));

    return "mailto:" + FEEDBACK_TO +
      "?subject=" + encodeURIComponent("[OpenSees-MCP 데모] " + (topic || "지적")) +
      "&body=" + encodeURIComponent(lines.join("\n"));
  }

  // ── is the app awake? ────────────────────────────────────────────────
  // The CTA promises a live run. If the solver is queued or the app is asleep,
  // saying so beforehand costs nothing; letting the visitor find out by waiting
  // costs the visitor.
  function probeHealth() {
    var meta = document.querySelector('meta[name="app-base"]');
    var base = (meta && meta.getAttribute("content") || "").replace(/\/+$/, "");
    var note = $("#run-health");
    if (!note) return;

    fetch(base + "/api/health/demo", { cache: "no-store" })
      .then(function (r) {
        // A 5xx/404 is exactly the case this probe exists for. Returning null
        // and bailing would leave the optimistic "약 12초" note in place — the
        // one outcome worse than not probing at all.
        if (!r.ok) throw new Error("health " + r.status);
        return r.json();
      })
      .then(function (h) {
        if (!h) throw new Error("health payload empty");
        if (h.ready === false) {
          note.innerHTML = "지금 <b>" + h.queue_depth +
            "건이 대기 중</b>입니다. 열어두고 잠시 후 ▶ Run을 눌러도 됩니다.";
          note.classList.add("is-busy");
        } else {
          note.innerHTML = "지금 실행 가능합니다 · 해석에 <b>약 " +
            (h.expected_seconds || 12) + "초</b>가 걸립니다.";
        }
      })
      .catch(function () {
        // No backend reachable — the page is still complete on its own, so
        // this is a footnote, not an error.
        note.innerHTML = "앱 서버가 응답하지 않습니다. " +
          "<b>이 페이지의 내용은 서버 없이도 전부 그대로입니다.</b>";
        note.classList.add("is-busy");
      });
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
    // 2dp, matching the load receipt exactly. The page invites the visitor to
    // compare printed numbers against a live run, so two roundings of the same
    // quantity on one page would be a self-inflicted discrepancy.
    setText("V", fmt(seis.V_kN, 2));

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

    // Everything below is per-variant too: the load chain is identical across
    // the ladder (mass is floor-load based) but the modal periods and every
    // member row change with the column section, so they are re-rendered
    // rather than drawn once.
    renderReceipts(rec);
    renderStoryForces(rec);
    renderCombos(rec);
    expandedMembers = false;
    renderMembers(rec);
    renderModal(rec);
    syncRunCta(name);
    var mail = $("#fb-mail");
    if (mail) mail.href = mailtoHref();

    // How many of the NG members this one combination actually governs. The
    // envelope NG count is over all 36 combinations, so quoting it against a
    // single combination would overstate that combination's reach.
    setText("gov-ng-count", (rec.members || []).filter(function (m) {
      return m.status === "NG" && m.governing_combo === rec.governing_combo;
    }).length);

    setText("nudge", nudgeFor(name));
    setText("foot", footFor(rec));
  }

  // Member filter segments. Delegated so the buttons survive a re-render.
  $$(".seg-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      memberFilter = b.getAttribute("data-filter") || "all";
      $$(".seg-btn").forEach(function (o) { o.classList.toggle("is-on", o === b); });
      expandedMembers = false;
      renderMembers(variants[select.value] || variants[DEFAULT]);
    });
  });

  // ?section=H-350x350 makes a given state linkable — useful when someone
  // wants to point at the exact case they are disputing.
  function initialSection() {
    var q = new URLSearchParams(window.location.search).get("section");
    // hasOwnProperty, not truthiness: `?section=constructor` resolves through
    // the prototype chain to a function, which passes a plain `variants[q]`
    // test and then blows up in render().
    return (q && Object.prototype.hasOwnProperty.call(variants, q)) ? q : DEFAULT;
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
  probeHealth();

  // Inline dispute buttons. A disagreement is worth almost nothing without the
  // state it was formed against, and nobody will retype it — so the click
  // carries the variant, the verdict and the bake commit into the draft.
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest && ev.target.closest(".dispute");
    if (!btn) return;
    var key = btn.getAttribute("data-anchor");
    var rec = variants[select.value] || {};
    var row = (rec.checks || []).filter(function (c) { return c.key === key; })[0];
    var mail = $("#fb-mail");
    if (mail) mail.href = mailtoHref(row ? row.label + " 검토가 다릅니다" : "지적", row);
    var target = document.getElementById("feedback");
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
