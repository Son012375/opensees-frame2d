"""V2 3D 시각화 — 자유 노드-요소 뷰어 (Plotly 기반).

StructuralModel (노드-요소 그래프)을 Plotly 기반 인터랙티브 HTML로 렌더링한다.
기존 visualization_3d.py (격자 기반)와 별도 모듈이며, 동일한 Plotly CDN + 스타일 사용.

Usage:
    from core.visualization_v2 import generate_model_viewer
    path = generate_model_viewer(model)                # 모델만
    path = generate_model_viewer(model, result=result)  # 모델 + 해석결과
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional


def generate_model_viewer(
    model,
    result=None,
    output_path: str | None = None,
    title: str = "Structural Model Viewer",
) -> str:
    """StructuralModel을 인터랙티브 3D HTML로 생성.

    Args:
        model: StructuralModel
        result: Frame3DMultiCaseResult (선택)
        output_path: HTML 파일 경로 (None이면 임시 파일)
        title: HTML 제목

    Returns:
        생성된 HTML 파일 경로
    """
    nodes_data = []
    for n in sorted(model.nodes.values(), key=lambda n: n.id):
        nodes_data.append({
            "id": n.id, "x": round(n.x, 4), "y": round(n.y, 4), "z": round(n.z, 4),
            "story": n.story,
            "support": n.support.value if n.support else None,
        })

    elements_data = []
    for e in sorted(model.elements.values(), key=lambda e: e.id):
        elements_data.append({
            "id": e.id, "ni": e.node_i, "nj": e.node_j,
            "type": e.elem_type.value,
            "section": e.section, "material": e.material,
            "length": round(e.length(model.nodes), 3),
            "release_i": e.release_i.value if e.release_i else None,
            "release_j": e.release_j.value if e.release_j else None,
        })

    summary = model.summary()
    story_elevations = model.story_elevations

    # 해석 결과
    result_data = None
    if result is not None:
        result_data = {"cases": {}, "combos": {}}
        for cname, cr in result.case_results.items():
            result_data["cases"][cname] = {
                "disp": {str(d["node"]): [d["dx_mm"], d["dy_mm"], d["dz_mm"]]
                         for d in cr.nodal_displacements},
                "max_disp_x": cr.max_displacement_x,
                "max_disp_y": cr.max_displacement_y,
                "max_moment": cr.max_moment,
                "drifts": cr.story_drifts,
                "reactions": cr.reactions,
            }
        for cname, cr in result.combo_results.items():
            result_data["combos"][cname] = {
                "disp": {str(d["node"]): [d["dx_mm"], d["dy_mm"], d["dz_mm"]]
                         for d in cr.nodal_displacements},
                "max_disp_x": cr.max_displacement_x,
                "max_disp_y": cr.max_displacement_y,
                "max_moment": cr.max_moment,
                "drifts": cr.story_drifts,
                "reactions": cr.reactions,
            }

    # 검증 텍스트
    validation_lines = []
    if model.validation:
        for issue in model.validation.issues:
            sev = issue.severity.value.upper()
            msg = issue.message
            if issue.default_value:
                msg += f" (default: {issue.default_value})"
            validation_lines.append({"severity": sev, "message": msg})

    payload = {
        "nodes": nodes_data,
        "elements": elements_data,
        "summary": summary,
        "story_elevations": story_elevations,
        "result": result_data,
        "validation": validation_lines,
    }

    data_json = json.dumps(payload, ensure_ascii=False)

    # 메타 텍스트
    meta = (
        f"{summary.get('num_stories', 0)}F | "
        f"Nodes {summary.get('num_nodes', 0)} | "
        f"Elems {summary.get('num_elements', 0)} | "
        f"Sections: {', '.join(summary.get('sections_used', [])[:3])}"
    )

    html = _HTML_TEMPLATE
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("__TITLE__", title)
    html = html.replace("__META_TEXT__", meta)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html")
        os.close(fd)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ============================================================
# HTML Template
# ============================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#333}
.header{background:#1a237e;color:#fff;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:20px;font-weight:600}
.header .meta{font-size:13px;opacity:.85;margin-top:4px}
.tab-bar{display:flex;gap:0;background:#fff;border-bottom:2px solid #1a237e;padding:0 24px}
.tab{padding:10px 20px;border:none;background:none;cursor:pointer;font-size:14px;color:#666;border-bottom:3px solid transparent;margin-bottom:-2px}
.tab.active{color:#1a237e;border-bottom-color:#1a237e;font-weight:600}
.tab:hover{color:#1a237e;background:#f0f0f8}
.tab-content{display:none;padding:20px 24px}
.tab-content.active{display:block}
.controls{display:flex;align-items:center;gap:12px;padding:12px 24px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
.controls label{font-size:13px;font-weight:500;color:#555}
.controls select,.controls input[type=range]{padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:13px}
.controls input[type=range]{width:180px}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.summary-card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.summary-card .label{font-size:11px;color:#888;text-transform:uppercase;margin-bottom:4px}
.summary-card .value{font-size:22px;font-weight:700;color:#1a237e}
.summary-card .detail{font-size:11px;color:#666;margin-top:2px}
.section-title{font-size:16px;font-weight:600;color:#1a237e;margin:20px 0 8px 0;padding-bottom:6px;border-bottom:2px solid #e8eaf6}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-top:12px}
th{background:#f5f5f5;padding:10px 12px;font-size:12px;text-align:right;color:#555;border-bottom:2px solid #ddd}
th:first-child{text-align:left}
td{padding:8px 12px;font-size:13px;text-align:right;border-bottom:1px solid #eee}
td:first-child{text-align:left;font-weight:500}
tr:hover{background:#f8f8ff}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.badge-column{background:#e3f2fd;color:#1565c0}
.badge-beam{background:#e8f5e9;color:#2e7d32}
.badge-brace{background:#fff3e0;color:#e65100}
.warn-item{padding:6px 12px;margin-bottom:4px;border-radius:4px;font-size:12px}
.warn-CRITICAL{background:#ffebee;border-left:4px solid #f44336;color:#c62828}
.warn-WARNING{background:#fff3e0;border-left:4px solid #ff9800;color:#e65100}
.warn-INFO{background:#e3f2fd;border-left:4px solid #42a5f5;color:#1565c0}
.model-section{background:#fff;border-radius:8px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:16px}
.model-section h3{font-size:15px;margin-bottom:12px;color:#1a237e}
#plot3d{width:100%;height:600px}
.drift-bar{height:18px;border-radius:3px;min-width:2px}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>__TITLE__</h1>
    <div class="meta">__META_TEXT__</div>
  </div>
</div>

<div class="tab-bar">
  <button class="tab active" onclick="switchTab('3dview',this)">3D View</button>
  <button class="tab" onclick="switchTab('summary',this)">Summary</button>
  <button class="tab" onclick="switchTab('elements',this)">Elements</button>
  <button class="tab" id="tab-btn-results" onclick="switchTab('results',this)" style="display:none">Results</button>
  <button class="tab" onclick="switchTab('validation',this)">Validation</button>
</div>

<!-- 3D View -->
<div id="tab-3dview" class="tab-content active">
  <div class="controls">
    <label>Display:</label>
    <select id="displayMode" onchange="updatePlot()">
      <option value="type">By Type</option>
      <option value="section">By Section</option>
      <option value="story">By Story</option>
    </select>
    <label style="margin-left:12px">Show:</label>
    <label><input type="checkbox" id="chkNodes" checked onchange="updatePlot()"> Nodes</label>
    <label><input type="checkbox" id="chkLabels" onchange="updatePlot()"> Labels</label>
    <label><input type="checkbox" id="chkSupports" checked onchange="updatePlot()"> Supports</label>
    <label><input type="checkbox" id="chkStories" checked onchange="updatePlot()"> Story Lines</label>
  </div>
  <div id="plot3d"></div>
</div>

<!-- Summary -->
<div id="tab-summary" class="tab-content">
  <div id="summaryContent"></div>
</div>

<!-- Elements -->
<div id="tab-elements" class="tab-content">
  <div id="elementsContent"></div>
</div>

<!-- Results -->
<div id="tab-results" class="tab-content">
  <div id="resultsContent"></div>
</div>

<!-- Validation -->
<div id="tab-validation" class="tab-content">
  <div id="validationContent"></div>
</div>

<script>
const DATA = __DATA_JSON__;
const NODES = DATA.nodes;
const ELEMENTS = DATA.elements;
const SUMMARY = DATA.summary;
const ELEVATIONS = DATA.story_elevations;
const RESULT = DATA.result;
const VALIDATION = DATA.validation;

const nodeMap = {};
NODES.forEach(n => { nodeMap[n.id] = n; });

const TYPE_COLORS = {column:'#1565c0', beam:'#2e7d32', brace:'#e65100'};
const TYPE_WIDTHS = {column:5, beam:3, brace:2};

// Tab switching
function switchTab(id, btn) {
  document.querySelectorAll('.tab-content').forEach(d => d.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
}

// ── 3D Plot (Plotly) ──
function buildTraces() {
  const mode = document.getElementById('displayMode').value;
  const showNodes = document.getElementById('chkNodes').checked;
  const showLabels = document.getElementById('chkLabels').checked;
  const showSupports = document.getElementById('chkSupports').checked;
  const showStories = document.getElementById('chkStories').checked;
  const traces = [];

  // Group elements
  const groups = {};
  ELEMENTS.forEach(e => {
    let key;
    if (mode === 'type') key = e.type;
    else if (mode === 'section') key = e.section;
    else key = 'S' + (nodeMap[e.ni]?.story ?? '?');
    if (!groups[key]) groups[key] = [];
    groups[key].push(e);
  });

  // Section color palette
  const sectionColors = ['#1565c0','#2e7d32','#e65100','#7b1fa2','#c62828','#00838f','#4e342e','#546e7a'];
  const sectionKeys = Object.keys(groups);

  Object.entries(groups).forEach(([key, elems], idx) => {
    const xs = [], ys = [], zs = [], hovers = [];
    elems.forEach(e => {
      const ni = nodeMap[e.ni], nj = nodeMap[e.nj];
      if (!ni || !nj) return;
      xs.push(ni.x, nj.x, null);
      ys.push(ni.y, nj.y, null);
      zs.push(ni.z, nj.z, null);
      hovers.push(
        `E${e.id}: ${e.type}<br>${e.section}<br>L=${e.length}m<br>N${e.ni}→N${e.nj}`,
        `E${e.id}: ${e.type}<br>${e.section}<br>L=${e.length}m<br>N${e.ni}→N${e.nj}`,
        ''
      );
    });

    let color, width;
    if (mode === 'type') {
      color = TYPE_COLORS[key] || '#999';
      width = TYPE_WIDTHS[key] || 3;
    } else {
      color = sectionColors[idx % sectionColors.length];
      width = 4;
    }

    traces.push({
      type:'scatter3d', mode:'lines', x:xs, y:ys, z:zs,
      line:{color:color, width:width},
      name: key,
      hovertext: hovers, hoverinfo:'text',
    });
  });

  // Nodes
  if (showNodes) {
    const nx=[], ny=[], nz=[], nt=[], nc=[];
    NODES.forEach(n => {
      nx.push(n.x); ny.push(n.y); nz.push(n.z);
      nt.push(`N${n.id} (${n.x.toFixed(2)}, ${n.y.toFixed(2)}, ${n.z.toFixed(2)})<br>Story ${n.story}`);
      nc.push(n.support ? '#f44336' : '#999');
    });
    traces.push({
      type:'scatter3d', mode:'markers', x:nx, y:ny, z:nz,
      marker:{size:3, color:nc, opacity:0.8},
      hovertext:nt, hoverinfo:'text', name:'Nodes', showlegend:false,
    });
  }

  // Node labels
  if (showLabels) {
    traces.push({
      type:'scatter3d', mode:'text', x:NODES.map(n=>n.x), y:NODES.map(n=>n.y), z:NODES.map(n=>n.z),
      text:NODES.map(n=>'N'+n.id),
      textfont:{size:9, color:'#666'},
      hoverinfo:'skip', showlegend:false,
    });
  }

  // Supports
  if (showSupports) {
    const sx=[], sy=[], sz=[], st=[], sc=[];
    NODES.filter(n=>n.support).forEach(n => {
      sx.push(n.x); sy.push(n.y); sz.push(n.z);
      st.push(`N${n.id}: ${n.support}`);
      sc.push(n.support==='fixed' ? '#f44336' : '#ff9800');
    });
    if (sx.length) {
      traces.push({
        type:'scatter3d', mode:'markers', x:sx, y:sy, z:sz,
        marker:{size:6, color:sc, symbol:'square', opacity:0.9},
        hovertext:st, hoverinfo:'text', name:'Supports',
      });
    }
  }

  // Story planes
  if (showStories && ELEVATIONS.length) {
    const allX = NODES.map(n=>n.x), allY = NODES.map(n=>n.y);
    const xMin = Math.min(...allX)-1, xMax = Math.max(...allX)+1;
    const yMin = Math.min(...allY)-1, yMax = Math.max(...allY)+1;

    ELEVATIONS.forEach((elev, idx) => {
      traces.push({
        type:'mesh3d',
        x:[xMin,xMax,xMax,xMin], y:[yMin,yMin,yMax,yMax],
        z:[elev,elev,elev,elev],
        i:[0,0], j:[1,2], k:[2,3],
        color:'#e8eaf6', opacity:0.15,
        name: idx===0 ? 'Stories' : undefined,
        showlegend: idx===0,
        hoverinfo:'skip',
      });
    });
  }

  return traces;
}

function updatePlot() {
  const traces = buildTraces();
  const layout = {
    scene: {
      xaxis:{title:'X (m)', gridcolor:'#ddd', zerolinecolor:'#bbb'},
      yaxis:{title:'Y (m)', gridcolor:'#ddd', zerolinecolor:'#bbb'},
      zaxis:{title:'Z (m)', gridcolor:'#ddd', zerolinecolor:'#bbb'},
      aspectmode:'data',
      camera:{eye:{x:1.5, y:1.5, z:1.0}},
    },
    margin:{l:0,r:0,t:0,b:0},
    legend:{x:0.01, y:0.99, bgcolor:'rgba(255,255,255,0.8)', font:{size:12}},
    paper_bgcolor:'#fafafa',
  };
  Plotly.react('plot3d', traces, layout, {responsive:true});
}

// ── Summary Tab ──
function renderSummary() {
  const s = SUMMARY;
  let h = '<div class="summary-grid">';
  h += card('Nodes', s.num_nodes);
  h += card('Elements', s.num_elements);
  h += card('Columns', s.num_columns);
  h += card('Beams', s.num_beams);
  h += card('Braces', s.num_braces);
  h += card('Stories', s.num_stories, s.total_height_m+'m');
  h += '</div>';

  h += '<div class="model-section"><h3>Story Heights</h3><table>';
  h += '<tr><th style="text-align:left">Story</th><th>Elevation (m)</th><th>Height (m)</th></tr>';
  (s.story_elevations_m||[]).forEach((elev, i) => {
    const ht = i < (s.story_heights_m||[]).length ? s.story_heights_m[i] : '-';
    h += `<tr><td>${i===0?'Base':i}</td><td>${elev.toFixed(2)}</td><td>${typeof ht==='number'?ht.toFixed(2):ht}</td></tr>`;
  });
  h += '</table></div>';

  h += '<div class="model-section"><h3>Sections Used</h3><table>';
  h += '<tr><th style="text-align:left">Section</th><th>Count</th></tr>';
  const secCount = {};
  ELEMENTS.forEach(e => { secCount[e.section] = (secCount[e.section]||0)+1; });
  Object.entries(secCount).sort((a,b)=>b[1]-a[1]).forEach(([sec,cnt]) => {
    h += `<tr><td>${sec}</td><td>${cnt}</td></tr>`;
  });
  h += '</table></div>';

  h += '<div class="model-section"><h3>Supports</h3><table>';
  h += '<tr><th style="text-align:left">Node</th><th>Type</th><th>X</th><th>Y</th><th>Z</th></tr>';
  (s.supports||[]).forEach(sp => {
    const n = nodeMap[sp.node_id];
    h += `<tr><td>N${sp.node_id}</td><td>${sp.type}</td><td>${n?n.x.toFixed(2):''}</td><td>${n?n.y.toFixed(2):''}</td><td>${n?n.z.toFixed(2):''}</td></tr>`;
  });
  h += '</table></div>';

  document.getElementById('summaryContent').innerHTML = h;
}

function card(label, value, detail) {
  return `<div class="summary-card"><div class="label">${label}</div><div class="value">${value}</div>${detail?'<div class="detail">'+detail+'</div>':''}</div>`;
}

// ── Elements Tab ──
function renderElements() {
  let h = '<div class="controls" style="border:none;padding:0 0 12px 0">';
  h += '<label>Filter: </label>';
  h += '<select id="elemFilter" onchange="filterElements()">';
  h += '<option value="all">All</option><option value="column">Columns</option><option value="beam">Beams</option><option value="brace">Braces</option>';
  h += '</select></div>';

  h += '<table id="elemTable">';
  h += '<tr><th style="text-align:left">ID</th><th style="text-align:left">Type</th><th style="text-align:left">Section</th><th style="text-align:left">Material</th><th>Length (m)</th><th style="text-align:left">Node i</th><th style="text-align:left">Node j</th><th style="text-align:left">Release i</th><th style="text-align:left">Release j</th></tr>';
  ELEMENTS.forEach(e => {
    const badge = `<span class="badge badge-${e.type}">${e.type}</span>`;
    h += `<tr data-type="${e.type}"><td>${e.id}</td><td>${badge}</td><td>${e.section}</td><td>${e.material}</td><td>${e.length}</td><td>N${e.ni}</td><td>N${e.nj}</td><td>${e.release_i||'-'}</td><td>${e.release_j||'-'}</td></tr>`;
  });
  h += '</table>';
  document.getElementById('elementsContent').innerHTML = h;
}

function filterElements() {
  const filter = document.getElementById('elemFilter').value;
  document.querySelectorAll('#elemTable tr[data-type]').forEach(tr => {
    tr.style.display = (filter==='all' || tr.dataset.type===filter) ? '' : 'none';
  });
}

// ── Results Tab ──
function renderResults() {
  if (!RESULT) return;
  document.getElementById('tab-btn-results').style.display = '';

  let h = '';
  const allCases = {...(RESULT.cases||{}), ...(RESULT.combos||{})};

  Object.entries(allCases).forEach(([name, cr]) => {
    h += `<div class="model-section"><h3>${name}</h3>`;
    h += '<div class="summary-grid">';
    h += card('Max Disp X', cr.max_disp_x+' mm');
    h += card('Max Disp Y', cr.max_disp_y+' mm');
    h += card('Max Moment', cr.max_moment+' kN·m');
    h += '</div>';

    if (cr.drifts && cr.drifts.length) {
      h += '<p class="section-title">Story Drift</p><table>';
      h += '<tr><th style="text-align:left">Story</th><th>Height (m)</th><th>Drift X</th><th>Drift Y</th><th>Resultant</th></tr>';
      cr.drifts.forEach(d => {
        h += `<tr><td>${d.story}</td><td>${d.height_m}</td><td>${d.drift_x.toFixed(6)}</td><td>${d.drift_y.toFixed(6)}</td><td>${d.drift_resultant.toFixed(6)}</td></tr>`;
      });
      h += '</table>';
    }

    if (cr.reactions && cr.reactions.length) {
      h += '<p class="section-title">Reactions</p><table>';
      h += '<tr><th style="text-align:left">Node</th><th>RX (kN)</th><th>RY (kN)</th><th>RZ (kN)</th><th>MX (kN·m)</th><th>MY (kN·m)</th><th>MZ (kN·m)</th></tr>';
      let totRx=0,totRy=0,totRz=0;
      cr.reactions.forEach(r => {
        h += `<tr><td>N${r.node}</td><td>${r.RX_kN}</td><td>${r.RY_kN}</td><td>${r.RZ_kN}</td><td>${r.MX_kNm}</td><td>${r.MY_kNm}</td><td>${r.MZ_kNm}</td></tr>`;
        totRx+=r.RX_kN; totRy+=r.RY_kN; totRz+=r.RZ_kN;
      });
      h += `<tr style="font-weight:700;background:#f5f5f5"><td>Total</td><td>${totRx.toFixed(1)}</td><td>${totRy.toFixed(1)}</td><td>${totRz.toFixed(1)}</td><td>-</td><td>-</td><td>-</td></tr>`;
      h += '</table>';
    }
    h += '</div>';
  });

  document.getElementById('resultsContent').innerHTML = h;
}

// ── Validation Tab ──
function renderValidation() {
  let h = '';
  if (!VALIDATION || !VALIDATION.length) {
    h = '<div class="model-section"><p>No validation issues.</p></div>';
  } else {
    h = '<div class="model-section"><h3>IFC Validation</h3>';
    VALIDATION.forEach(v => {
      h += `<div class="warn-item warn-${v.severity}"><b>[${v.severity}]</b> ${v.message}</div>`;
    });
    h += '</div>';
  }
  document.getElementById('validationContent').innerHTML = h;
}

// ── Init ──
updatePlot();
renderSummary();
renderElements();
renderResults();
renderValidation();
</script>
</body>
</html>"""
