"""문서형 구조계산서(HTML) 리포트 생성 모듈.

탭 기반 리포트(visualization_3d.plot_frame_3d_interactive)를 대체하는
문서형(표지 → 사이드바 목차 → 번호 섹션 1~10 → 부록) 레이아웃.

핵심 설계:
- visualization_3d.prepare_3d_viz_data() 를 IMPORT 하여 데이터 가공을 재사용.
- 시각 타깃: docs/report_mockups/calc_report_hybrid.html 의 레이아웃/CSS
  (A4 .page, 좌측 고정 TOC + scroll-spy, 상단 sticky 메트릭 스트립,
   네이비 룰드 테이블, OK/NG pill, §10 callout, @page A4 + @media print).
- 섹션 렌더 로직/필드 접근은 visualization_3d.py 의 render* 함수와 동일.

NOTE: visualization_3d.py 는 수정하지 않음 — 오직 import/read 만 함.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

# 데이터 가공 재사용 (재구현 금지)
from core.visualization_3d import prepare_3d_viz_data


# ============================================================
# Public API
# ============================================================

def plot_frame_3d_calc_report(
    multi_result,
    output_path=None,
    deformation_scale=50.0,
    design_check=None,
    interpretation=None,
    assumptions=None,
    load_result=None,
    cover_info=None,
    data_out_path=None,
) -> str:
    """3D 프레임 해석 결과 → 문서형 구조계산서 HTML.

    Args:
        multi_result: Frame3DMultiCaseResult 객체
        output_path: HTML 파일 경로 (None이면 임시 파일)
        deformation_scale: §6 3D 뷰 변형 배율 기본값 (0~200)
        design_check: run_design_check() 결과 dict (None 허용 — §8/§9 placeholder)
        interpretation: 결과 해석 dict (None 허용 — §10 callout placeholder)
        assumptions: build_assumption_summary() 결과 dict (None 허용)
        load_result: 하중 생성 결과 (Stage 1A 미사용, data["loads"]={})
        cover_info: 표지 정보 dict (None이면 placeholder 기본값)
        data_out_path: 지정 시, 렌더에 쓰인 data dict를 JSON으로 저장.
            표지/도장란을 재해석 없이 다시 주입(render_calc_report_from_data)할 때
            이 사이드카를 읽는다.

    Returns:
        str: 생성된 HTML 파일 경로
    """
    data = build_calc_report_data(
        multi_result,
        design_check=design_check,
        interpretation=interpretation,
        assumptions=assumptions,
        load_result=load_result,
        cover_info=cover_info,
    )

    html = render_calc_report_html(data, deformation_scale)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html")
        os.close(fd)

    _atomic_write(output_path, html)

    # 표지 재주입(render_calc_report_from_data)용 사이드카 저장 (best-effort)
    if data_out_path:
        try:
            with open(data_out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    return output_path


def build_calc_report_data(
    multi_result,
    design_check=None,
    interpretation=None,
    assumptions=None,
    load_result=None,
    cover_info=None,
) -> dict:
    """렌더에 필요한 data dict 빌드 (HTML 템플릿과 분리 — 재렌더/사이드카 저장용)."""
    data = prepare_3d_viz_data(multi_result)

    # visualization_3d 와 동일한 키로 주입 (defensive: None 허용)
    data["design_check"] = design_check if design_check is not None else None
    data["interpretation"] = interpretation if interpretation is not None else None
    data["assumptions"] = assumptions if assumptions is not None else None

    model_info = data.get("model_info", {})

    # 표지 정보 정규화 (placeholder 기본값) + 면적 산정
    data["cover"] = _normalize_cover(cover_info, model_info)
    data["areas"] = _build_areas(cover_info, model_info)

    # Stage 1B: load_result + 단면/재료 조회 → §1/§3/§4/§5 실측치
    reports = (load_result or {}).get("reports", {}) if isinstance(load_result, dict) else {}
    data["loads"] = {
        "gravity": reports.get("gravity"),   # list[per-story dict] | None
        "seismic": reports.get("seismic"),   # dict | None (error dict 가능)
        "wind": reports.get("wind"),         # dict | None (error dict 가능)
        "snow": reports.get("snow"),         # dict | None (error dict 가능)
        "combinations": (load_result or {}).get("load_combinations") if isinstance(load_result, dict) else None,
    }
    # 단면 제원 + Fy/Fu (§3)
    data["sections"] = _build_section_props(multi_result, model_info)

    return data


def _atomic_write(output_path, text) -> str:
    """같은 디렉터리 임시파일에 쓰고 os.replace 로 원자적 교체.

    쓰기 실패(디스크 풀/중단) 시 기존 파일을 손상시키지 않는다 — report.html은
    재해석 없이 갱신되는 유일 산출물이므로 부분쓰기로 깨지면 복구 불가.
    """
    d = os.path.dirname(os.path.abspath(output_path)) or "."
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return output_path


def _json_sanitize(obj):
    """Tier2-14: NaN/Inf float → None 재귀 치환 (json.dumps 비표준 'NaN' 토큰 → 500 방지)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def render_calc_report_html(data, deformation_scale=50.0) -> str:
    """data dict → 완성된 HTML 문자열.

    DATA는 <script> 리터럴 안에 박히므로, 표지 텍스트의 '</script>' 등으로
    스크립트가 조기 종료되어 임의 마크업이 주입되는 것을 막기 위해
    JSON 직렬화 결과에서 < > & 를 \\u 이스케이프(JSON상 동일 디코드)한다.
    Tier2-14: 직렬화 전 NaN/Inf를 None으로 정리해 비표준 JSON으로 인한 500을 막는다.
    """
    data_json = json.dumps(_json_sanitize(data), ensure_ascii=False)
    data_json = (data_json.replace("<", "\\u003c")
                          .replace(">", "\\u003e")
                          .replace("&", "\\u0026"))
    html = _HTML_TEMPLATE
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("__DEFAULT_SCALE__", str(int(deformation_scale)))
    return html


def render_calc_report_from_data(data, cover_info, output_path, deformation_scale=50.0) -> str:
    """캐시된 data dict에 표지/도장란(cover_info)만 다시 주입하여 리포트 재생성.

    재해석 없이 표지·도장란·연면적을 갱신할 때 사용 (webapp report-cover 엔드포인트).
    cover_info 의존 항목(cover/areas)만 재계산하고 나머지 해석 데이터는 보존한다.
    """
    model_info = data.get("model_info", {})
    data["cover"] = _normalize_cover(cover_info, model_info)
    data["areas"] = _build_areas(cover_info, model_info)
    html = render_calc_report_html(data, deformation_scale)
    return _atomic_write(output_path, html)


def _normalize_cover(cover_info, model_info) -> dict:
    """표지 정보 정규화 — 누락 항목은 placeholder. 로고/직인 이미지(data URL)는
    있으면 그대로, 없으면 None (JS가 조건부 렌더)."""
    ci = cover_info if isinstance(cover_info, dict) else {}
    stamp = ci.get("stamp") if isinstance(ci.get("stamp"), dict) else {}

    def _s(d, key, default):
        v = d.get(key)
        return v if (v is not None and v != "") else default

    def _img(d, key):
        """data URL 이미지 — 비어있지 않은 문자열만 통과, 아니면 None."""
        v = d.get(key) if isinstance(d, dict) else None
        return v if (isinstance(v, str) and v.strip()) else None

    _ROLE_KO = {"author": "[작성자]", "reviewer": "[검토자]", "approver": "[승인자]"}

    def _person(role):
        p = stamp.get(role) if isinstance(stamp.get(role), dict) else {}
        return {
            "name": _s(p, "name", _ROLE_KO[role]),
            "qualification": _s(p, "qualification", "구조기술사"),
            "license_no": _s(p, "license_no", "[자격번호]"),
            "seal": _img(p, "seal"),
        }

    return {
        "project_name": _s(ci, "project_name", "[프로젝트명]"),
        "location": _s(ci, "location", "[대지위치]"),
        "client": _s(ci, "client", "[건축주]"),
        "structure_type": _s(ci, "structure_type", "강구조 보통모멘트골조"),
        "date": _s(ci, "date", "[YYYY.MM.DD]"),
        "firm": _s(ci, "firm", "[작성업체]"),
        "logo": _img(ci, "logo"),
        "stamp": {
            "author": _person("author"),
            "reviewer": _person("reviewer"),
            "approver": _person("approver"),
        },
    }


def _build_areas(cover_info, model_info) -> dict:
    """건축면적/연면적 — 사용자 입력 우선, 없으면 외곽치수×층수 자동 산정.

    자동 산정은 외곽(bounding-box) 치수 기준이므로 비정형(L/T자형) 평면은
    실제보다 과대평가될 수 있음 → JS가 "(자동 산정)" 으로 표기.
    사용자 입력(cover_info.building_area / gross_floor_area, m²)이 있으면 우선한다.
    """
    ci = cover_info if isinstance(cover_info, dict) else {}
    num_stories = model_info.get("num_stories") or 0
    wx = model_info.get("total_width_x")
    wy = model_info.get("total_width_y")

    auto_building = None
    auto_gross = None
    if isinstance(wx, (int, float)) and isinstance(wy, (int, float)) and wx > 0 and wy > 0:
        auto_building = round(float(wx) * float(wy), 1)
        if num_stories:
            auto_gross = round(auto_building * num_stories, 1)

    def _pos(v):
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    user_building = _pos(ci.get("building_area"))
    user_gross = _pos(ci.get("gross_floor_area"))

    building_area = user_building if user_building is not None else auto_building
    gross_floor_area = user_gross if user_gross is not None else auto_gross

    return {
        "building_area_m2": building_area,
        "gross_floor_area_m2": gross_floor_area,
        "building_area_source": ("user" if user_building is not None
                                 else ("auto" if auto_building is not None else None)),
        "gross_floor_area_source": ("user" if user_gross is not None
                                    else ("auto" if auto_gross is not None else None)),
    }


# 강종 → (Fy, Fu) MPa. KS D 3503(SS)/3515(SM·SMA)/3866(SN)/SHN, 대표 두께대(t≤16mm) 기준.
# 두께가 커지면 Fy가 낮아지나(예 SM355 t>40 → 335), 표기 명확성 위해 기본대 값 사용.
_FY_FU_TABLE = {
    "SS235": (235, 330), "SS275": (275, 410), "SS315": (315, 490), "SS410": (410, 540),
    "SS400": (235, 400),  # 구 규격 (인장강도 명명)
    "SM275": (275, 410), "SM355": (355, 490), "SM420": (420, 520), "SM460": (460, 570),
    "SM490": (325, 490),  # 구 규격
    "SN275": (275, 410), "SN355": (355, 490), "SN460": (460, 570),
    "SHN275": (275, 410), "SHN355": (355, 490), "SHN420": (420, 520),
    "SMA275": (275, 410), "SMA355": (355, 490),
}


def _lookup_fy_fu(material_name):
    """강종명 → (Fy, Fu) MPa. 매칭 실패 시 (None, None)."""
    if not material_name:
        return None, None
    import re
    key = str(material_name).upper().replace("-", "").replace(" ", "").strip()
    m = re.match(r"(S[A-Z]+\d{3})", key)  # 등급코드만 추출 (예: "SS275(KS)" → "SS275")
    if m:
        key = m.group(1)
    return _FY_FU_TABLE.get(key, (None, None))


def _parse_fy_from_name(material_name) -> "int | None":
    """강종명 끝자리 숫자에서 Fy 추정 (SS275→275, SM355→355). 실패 시 None."""
    if not material_name:
        return None
    import re
    m = re.search(r"(\d{3})\s*$", str(material_name))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _build_section_props(multi_result, model_info) -> dict:
    """§3 단면 제원 + Fy/Fu 빌드 (defensive).

    Fy: multi_result.fy_MPa 우선 → 강종표 → 강종명 끝자리 파싱.
    Fu: 강종표(KS) 조회. 매칭 실패 시 None (JS가 "[입력]" 표기).
    단면: get_section_3d(name) → {A_mm2, Ix_mm4, Iy_mm4, h_mm, b_mm}.
    조회 실패/예외 시 해당 부재는 None (JS가 "[입력]" 표기).
    """
    # Fy / Fu
    material_name = model_info.get("material_name")
    table_fy, fu = _lookup_fy_fu(material_name)
    fy = getattr(multi_result, "fy_MPa", None)
    if fy is None:
        fy = table_fy if table_fy is not None else _parse_fy_from_name(material_name)

    # 단면 제원 (column / beam_x / beam_y)
    sections = {}
    try:
        from core.section_3d import get_section_3d
    except Exception:
        get_section_3d = None
    try:
        from core.design_check import _compute_design_props  # Zx 재계산 재사용
    except Exception:
        _compute_design_props = None

    for role, key in (("column", "column_section"),
                      ("beam_x", "beam_x_section"),
                      ("beam_y", "beam_y_section")):
        name = model_info.get(key) or getattr(multi_result, key, None)
        entry = {"name": name, "A_mm2": None, "Ix_mm4": None,
                 "Iy_mm4": None, "h_mm": None, "b_mm": None, "Zx_mm3": None}
        if name and get_section_3d is not None:
            try:
                sec = get_section_3d(name)
                if sec is not None:
                    entry.update({
                        "name": getattr(sec, "name", name),
                        "A_mm2": getattr(sec, "A", None),
                        "Ix_mm4": getattr(sec, "Ix", None),
                        "Iy_mm4": getattr(sec, "Iy", None),
                        "h_mm": getattr(sec, "h", None),
                        "b_mm": getattr(sec, "b", None),
                    })
                    # Zx(소성단면계수): design_check와 동일 산식으로 재계산
                    if _compute_design_props is not None:
                        try:
                            dp = _compute_design_props(
                                getattr(sec, "A", 0) or 0, getattr(sec, "Ix", 0) or 0,
                                getattr(sec, "Iy", 0) or 0, getattr(sec, "h", 0) or 0,
                                getattr(sec, "b", 0) or 0, getattr(sec, "tw", 0) or 0,
                                getattr(sec, "tf", 0) or 0,
                            )
                            zx = dp.get("Zx")
                            entry["Zx_mm3"] = zx if (zx and zx > 0) else None
                        except Exception:
                            pass
            except Exception:
                pass  # entry는 name만, 나머지 None → JS가 "[입력]"
        sections[role] = entry

    return {"fy_MPa": fy, "fu_MPa": fu, "items": sections}


# ============================================================
# HTML Template (placeholder replacement, no f-string escaping)
# ============================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>구조계산서 (문서형)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root{
    --navy:#16335c;
    --navy-light:#27497a;
    --ink:#1a1a1a;
    --rule:#9aa3ad;
    --rule-thin:#c7ccd2;
    --rule-light:#e2e5e9;
    --shade:#eef1f5;
    --shade-navy:#16335c;
    --ok:#15633a;
    --ng:#9e1b1b;
    --ok-bg:#e9f3ed;
    --ok-bd:#bcd9c8;
    --ng-bg:#fbeaea;
    --ng-bd:#e6c2c2;
    --paper:#ffffff;
    --pagebg:#54595f;
    --navy-soft:#e7edf5;
    --sidebar-w:264px;
  }
  *{ box-sizing:border-box; }
  html{ scroll-behavior:smooth; }
  html,body{
    margin:0; padding:0;
    background:var(--pagebg);
    color:var(--ink);
    font-family:"Malgun Gothic","\B9D1\C740 고\B515","Apple SD Gothic Neo","Noto Sans KR",
                "Segoe UI",Tahoma,sans-serif;
    font-size:10.5pt;
    line-height:1.5;
    -webkit-font-smoothing:antialiased;
  }
  /* 비인쇄 안내 배너 */
  .mock-banner{
    background:#3a2b00; color:#ffe08a; text-align:center;
    font-size:9pt; letter-spacing:.02em; padding:7px 10px;
    border-bottom:1px solid #6b5200;
  }
  /* 좌측 고정 목차 사이드바 — 화면 전용 */
  .sidebar{
    position:fixed; top:0; left:0; bottom:0;
    width:var(--sidebar-w);
    background:#fbfcfd; border-right:1px solid var(--rule-thin);
    overflow-y:auto; padding:18px 16px 40px; z-index:40;
  }
  .sb-title{ font-size:13px; font-weight:800; color:var(--navy); letter-spacing:.01em; }
  .sb-sub{ font-size:11px; color:#6b7280; margin-top:2px; }
  .sb-verdict{
    margin:14px 0 18px; border-radius:6px; padding:10px 12px;
    background:var(--ok-bg); border:1px solid var(--ok-bd); border-left:4px solid var(--ok);
  }
  .sb-verdict.ng{ background:var(--ng-bg); border-color:var(--ng-bd); border-left-color:var(--ng); }
  .sb-verdict .vlabel{ font-size:11px; color:#374151; margin-bottom:5px; }
  .sb-verdict .vbadge{
    display:inline-block; font-size:13px; font-weight:800; color:#fff;
    background:var(--ok); padding:3px 11px; border-radius:5px; letter-spacing:.03em;
  }
  .sb-verdict.ng .vbadge{ background:var(--ng); }
  .toc-label{
    font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.09em;
    color:#6b7280; margin:6px 4px 8px;
  }
  nav.toc ol{ list-style:none; margin:0; padding:0; }
  nav.toc a{
    display:flex; gap:9px; align-items:baseline; text-decoration:none;
    color:#3a4658; font-size:13px; padding:7px 10px; border-radius:6px;
    border-left:2px solid transparent; transition:background .12s, color .12s;
  }
  nav.toc a:hover{ background:var(--navy-soft); color:var(--navy); }
  nav.toc a .num{ font-variant-numeric:tabular-nums; color:#8b94a0; font-weight:700; min-width:16px; }
  nav.toc a.active{ background:var(--navy-soft); color:var(--navy); font-weight:700; border-left-color:var(--navy); }
  nav.toc a.active .num{ color:var(--navy); }
  .main{ margin-left:var(--sidebar-w); }
  /* 상단 sticky 메트릭 스트립 — 화면 전용 */
  .summary-strip{
    position:sticky; top:0; z-index:30;
    background:rgba(255,255,255,.97); backdrop-filter:saturate(140%) blur(2px);
    border-bottom:1px solid var(--rule); padding:11px 22px;
    display:flex; flex-wrap:wrap; gap:10px; align-items:center;
  }
  .strip-title{ font-size:12px; font-weight:800; color:var(--navy); margin-right:6px; letter-spacing:.02em; }
  .chip{
    display:inline-flex; align-items:center; gap:7px; background:#f1f4f8;
    border:1px solid var(--rule-thin); border-radius:20px; padding:5px 12px;
    font-size:12px; color:#3a4658;
  }
  .chip b{ color:var(--navy); font-variant-numeric:tabular-nums; }
  .chip .ck{ font-size:10.5px; color:#6b7280; font-weight:700; }
  .chip.ng{ background:var(--ng-bg); border-color:var(--ng-bd); color:#7f1d1d; }
  .chip.ng b{ color:var(--ng); }
  /* A4 페이지 박스 */
  .page{
    background:var(--paper); width:210mm; min-height:297mm; margin:10mm auto;
    padding:18mm 18mm 16mm 18mm; position:relative;
    box-shadow:0 2px 12px rgba(0,0,0,.35);
    display:flex; flex-direction:column; scroll-margin-top:70px;
  }
  .run-head{
    position:absolute; top:9mm; left:18mm; right:18mm;
    display:flex; justify-content:space-between; align-items:flex-end;
    font-size:8.2pt; color:#444; padding-bottom:3px; border-bottom:.6pt solid var(--navy);
  }
  .run-head .rh-proj{ font-weight:600; }
  .run-head .rh-doc{ font-weight:700; color:var(--navy); letter-spacing:.18em; }
  .run-foot{
    position:absolute; bottom:9mm; left:18mm; right:18mm; text-align:center;
    font-size:8.2pt; color:#555; padding-top:3px; border-top:.6pt solid var(--rule-thin);
  }
  .run-foot .pg-cur{ font-weight:600; color:var(--navy); }
  .page-body{ flex:1 0 auto; }
  h2.sec{
    font-size:13.5pt; color:var(--navy); margin:0 0 12px 0; padding:0 0 6px 0;
    border-bottom:1.6pt solid var(--navy); letter-spacing:.01em;
  }
  h2.sec .sec-no{ display:inline-block; min-width:1.4em; margin-right:.35em; font-weight:800; }
  h3.sub{
    font-size:11pt; color:var(--navy-light); margin:18px 0 7px 0;
    padding-left:9px; border-left:3px solid var(--navy);
  }
  p.note{ font-size:9.3pt; color:#444; margin:6px 0; }
  p.note strong{ color:var(--navy); }
  table.tbl{ width:100%; border-collapse:collapse; margin:9px 0 6px 0; font-size:9.6pt; }
  table.tbl caption{
    caption-side:top; text-align:left; font-size:9.3pt; color:#333;
    font-weight:600; padding:0 0 4px 1px;
  }
  table.tbl th, table.tbl td{ border:.6pt solid var(--rule); padding:4px 7px; vertical-align:middle; }
  table.tbl thead th{
    background:var(--shade-navy); color:#fff; font-weight:600; text-align:center;
    border-color:var(--navy); letter-spacing:.01em;
  }
  table.tbl tbody th{ background:var(--shade); font-weight:600; text-align:left; width:32%; color:#1f2a3a; }
  table.tbl td.num, table.tbl th.num{ text-align:right; font-variant-numeric:tabular-nums; }
  table.tbl td.ctr, table.tbl th.ctr{ text-align:center; }
  table.tbl tbody tr:nth-child(even) td{ background:#fafbfc; }
  table.tbl .unit{ font-weight:400; font-size:8.6pt; opacity:.9; }
  table.tbl tfoot td{ background:var(--shade); font-weight:600; border-color:var(--rule); }
  table.tbl tbody tr.ng-row td{ background:#fbf1f1; }
  table.kv tbody th{ width:34%; }
  table.kv td{ text-align:left; }
  .pill{
    display:inline-block; font-size:8.6pt; font-weight:800; padding:1px 9px;
    border-radius:11px; letter-spacing:.03em; border:1px solid transparent;
  }
  .pill.ok{ background:var(--ok-bg); color:var(--ok); border-color:var(--ok-bd); }
  .pill.ng{ background:var(--ng-bg); color:var(--ng); border-color:var(--ng-bd); }
  .placeholder-box{
    border:1pt dashed var(--rule);
    background:repeating-linear-gradient(45deg,#f3f4f6,#f3f4f6 10px,#eceef1 10px,#eceef1 20px);
    color:#5a6470; text-align:center; font-size:10pt; letter-spacing:.04em;
    display:flex; align-items:center; justify-content:center; min-height:78mm; margin:10px 0;
  }
  .verdict{ border:2pt solid var(--ok); background:#eef6f1; padding:12px 16px; margin:6px 0 14px 0; }
  .verdict.ng{ border-color:var(--ng); background:#fbf1f1; }
  .verdict .vtag{ font-size:12.5pt; font-weight:800; color:var(--ok); letter-spacing:.01em; }
  .verdict.ng .vtag{ color:var(--ng); }
  .verdict .vsub{ font-size:9.3pt; color:#1f3a2a; margin-top:3px; }
  .verdict.ng .vsub{ color:#5a1212; }
  .callout{
    background:#eef6f1; border:.6pt solid var(--ok-bd); border-left:5px solid var(--ok);
    padding:14px 18px; margin:6px 0 14px 0;
  }
  .callout.ng{ background:#fbf1f1; border-color:var(--ng-bd); border-left-color:var(--ng); }
  .callout .c-badge{
    display:inline-block; background:var(--ok); color:#fff; font-size:11pt; font-weight:800;
    padding:5px 13px; border-radius:5px; letter-spacing:.02em; margin-bottom:11px;
  }
  .callout.ng .c-badge{ background:var(--ng); }
  .callout p{ margin:0; color:#22332a; font-size:10pt; line-height:1.78; text-align:justify; }
  .callout.ng p{ color:#3f2222; }
  .callout strong{ color:var(--ok); }
  .callout.ng strong{ color:#7f1d1d; }
  .diag-box{ background:#fff7ec; border:.6pt solid #f0d8ac; border-radius:4px; padding:12px 14px; margin:8px 0; font-size:9.6pt; }
  .diag-box .dh{ font-weight:700; color:#9a5a00; }
  .sugg{
    padding:9px 13px; background:#f6f7f9; border-radius:5px; border-left:4px solid var(--navy);
    margin:7px 0; font-size:9.6pt;
  }
  .sugg .imp{ font-size:8.6pt; color:#888; float:right; }
  ul.warns{ margin:8px 0; padding-left:0; list-style:none; font-size:9.4pt; }
  ul.warns li{ padding:4px 8px; border-bottom:.6pt solid var(--rule-light); display:flex; gap:9px; }
  ul.warns li .wcode{ color:var(--navy); font-weight:700; font-family:"Consolas",monospace; min-width:3em; flex:0 0 auto; }
  /* 표지 */
  .cover{ text-align:center; }
  .cover-top{ border-top:2.4pt solid var(--navy); border-bottom:.8pt solid var(--navy); padding:6mm 0 4mm 0; margin-top:6mm; }
  .cover-kind{ font-size:11pt; letter-spacing:.55em; color:var(--navy-light); }
  .cover-title{ font-size:30pt; font-weight:800; color:var(--navy); letter-spacing:.02em; line-height:1.25; margin:14mm 0 6mm 0; }
  .cover-sub{ font-size:13pt; color:#333; letter-spacing:.35em; }
  .cover-rule{ width:46%; margin:9mm auto; border:0; border-top:.8pt solid var(--rule); }
  table.titleblock{ width:80%; margin:6mm auto 0 auto; border-collapse:collapse; font-size:10.5pt; }
  table.titleblock th, table.titleblock td{ border:.6pt solid var(--rule); padding:7px 12px; }
  table.titleblock th{ background:var(--shade); width:32%; text-align:center; font-weight:600; color:#1f2a3a; letter-spacing:.06em; }
  table.titleblock td{ text-align:left; }
  .stamp-wrap{ margin:14mm auto 0 auto; width:80%; }
  .stamp-cap{ font-size:9.5pt; color:#444; text-align:left; letter-spacing:.1em; margin-bottom:5px; }
  table.stamp{ width:100%; border-collapse:collapse; }
  table.stamp th{ background:var(--navy); color:#fff; font-weight:600; border:.7pt solid var(--navy); padding:5px; font-size:10pt; letter-spacing:.2em; }
  table.stamp td{ border:.7pt solid var(--rule); height:30mm; vertical-align:top; padding:6px 9px; font-size:9pt; color:#555; width:33.33%; text-align:left; }
  table.stamp td .field{ display:block; margin-bottom:9px; }
  table.stamp td .sealspot{ float:right; width:15mm; height:15mm; border:.6pt dotted var(--rule); border-radius:50%; color:#aab; font-size:7.5pt; text-align:center; line-height:15mm; margin-top:2mm; }
  table.toc{ width:100%; border-collapse:collapse; font-size:10.5pt; margin-top:6px; }
  table.toc td{ padding:7px 4px; border-bottom:.6pt dotted var(--rule); }
  table.toc td.toc-no{ width:10%; color:var(--navy); font-weight:700; }
  table.toc td.toc-pg{ width:10%; text-align:right; font-variant-numeric:tabular-nums; color:#333; }
  table.toc tr.toc-app td{ color:#555; }
  .meta-foot{ margin-top:auto; font-size:8.4pt; color:#777; border-top:.6pt solid var(--rule-light); padding-top:5px; }
  #plot3d{ width:100%; height:560px; }
  .viz-controls{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin:8px 0; font-size:9.4pt; }
  .viz-controls select, .viz-controls input{ font-size:9.4pt; }
  .viz-cards{ display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }
  .viz-card{ flex:1 1 110px; border:.6pt solid var(--rule-thin); border-radius:6px; padding:7px 10px; background:#fafbfc; }
  .viz-card .vc-lbl{ font-size:8.4pt; color:#777; text-transform:uppercase; }
  .viz-card .vc-val{ font-size:13pt; font-weight:700; color:var(--navy); }
  .viz-card .vc-det{ font-size:8.2pt; color:#888; }
  .empty-note{ color:#888; font-style:italic; font-size:9.4pt; margin:8px 0; }
  /* 내보내기 버튼 (화면 전용) */
  .export-bar{ position:fixed; right:16px; bottom:16px; z-index:60; display:flex; gap:8px; }
  .exp-btn{
    font-family:inherit; font-size:12px; font-weight:700; color:#fff;
    background:var(--navy); border:1px solid var(--navy); border-radius:7px;
    padding:9px 14px; cursor:pointer; box-shadow:0 3px 10px rgba(0,0,0,.28);
  }
  .exp-btn:hover{ background:var(--navy-light); }
  .exp-btn.ghost{ background:#fff; color:var(--navy); }
  .exp-btn.ghost:hover{ background:var(--navy-soft); }
  @page{ size:A4; margin:14mm; }
  @media print{
    html,body{ background:#fff; font-size:10pt; }
    :root{ --sidebar-w:0px; }
    .mock-banner, .sidebar, .summary-strip{ display:none !important; }
    .main{ margin-left:0 !important; }
    .page{
      width:auto; min-height:auto; margin:0;
      padding:11mm 0 12mm 0; box-shadow:none; page-break-after:always;
    }
    .page:last-child{ page-break-after:auto; }
    .sec-page{ page-break-before:always; }
    .run-head{ position:fixed; top:0; left:0; right:0; border-bottom:.6pt solid var(--navy); }
    .run-foot{ position:fixed; bottom:0; left:0; right:0; }
    .export-bar{ display:none !important; }
    table.tbl, .verdict, .callout, .placeholder-box, .diag-box, .sugg{ page-break-inside:avoid; }
    thead{ display:table-header-group; }
    tr{ page-break-inside:avoid; }
  }
</style>
</head>
<body>

<div class="export-bar">
  <button type="button" class="exp-btn" onclick="window.print()">🖨 PDF로 저장 / 인쇄</button>
  <button type="button" class="exp-btn ghost" onclick="downloadReportHtml()">⬇ HTML 저장</button>
</div>

<aside class="sidebar">
  <div class="sb-title">구조계산서</div>
  <div class="sb-sub" id="sbSub">&nbsp;</div>
  <div id="sbVerdict"></div>
  <div class="toc-label">목차</div>
  <nav class="toc">
    <ol>
      <li><a href="#s1"><span class="num">1</span>구조설계개요</a></li>
      <li><a href="#s2"><span class="num">2</span>적용기준</a></li>
      <li><a href="#s3"><span class="num">3</span>사용재료</a></li>
      <li><a href="#s4"><span class="num">4</span>하중산정</a></li>
      <li><a href="#s5"><span class="num">5</span>하중조합</a></li>
      <li><a href="#s6"><span class="num">6</span>해석개요</a></li>
      <li><a href="#s7"><span class="num">7</span>고유치/동적</a></li>
      <li><a href="#s8"><span class="num">8</span>부재검토</a></li>
      <li><a href="#s9"><span class="num">9</span>층간변위</a></li>
      <li><a href="#s10"><span class="num">10</span>종합결론</a></li>
      <li><a href="#sapp"><span class="num">부록</span>부재력 상세</a></li>
    </ol>
  </nav>
</aside>

<div class="main">

  <div class="summary-strip" id="summaryStrip"></div>

  <!-- 표지 -->
  <section class="page cover" id="coverPage"></section>

  <!-- 목차 페이지 -->
  <section class="page" id="tocPage"></section>

  <!-- 1. 구조설계 개요 -->
  <section class="page sec-page" id="s1">
    <div class="run-head"><span class="rh-proj" id="rh1"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">3</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">1.</span>구조설계 개요</h2><div id="sec1"></div></div>
  </section>

  <!-- 2. 적용 기준 -->
  <section class="page sec-page" id="s2">
    <div class="run-head"><span class="rh-proj" id="rh2"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">4</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">2.</span>적용 기준</h2><div id="sec2"></div></div>
  </section>

  <!-- 3. 사용 재료 -->
  <section class="page sec-page" id="s3">
    <div class="run-head"><span class="rh-proj" id="rh3"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">5</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">3.</span>사용 재료</h2><div id="sec3"></div></div>
  </section>

  <!-- 4. 하중 산정 -->
  <section class="page sec-page" id="s4">
    <div class="run-head"><span class="rh-proj" id="rh4"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">6</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">4.</span>하중 산정</h2><div id="sec4"></div></div>
  </section>

  <!-- 5. 하중 조합 -->
  <section class="page sec-page" id="s5">
    <div class="run-head"><span class="rh-proj" id="rh5"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">7</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">5.</span>하중 조합</h2><div id="sec5"></div></div>
  </section>

  <!-- 6. 해석 개요 -->
  <section class="page sec-page" id="s6">
    <div class="run-head"><span class="rh-proj" id="rh6"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">8</span> -</div>
    <div class="page-body">
      <h2 class="sec"><span class="sec-no">6.</span>해석 개요</h2>
      <div id="sec6"></div>
      <h3 class="sub">6.1 해석 모델 형상 (3D)</h3>
      <div class="viz-controls">
        <label>하중케이스:
          <select id="caseSelect" onchange="onCaseChange()"></select>
        </label>
        <label>변형 배율:
          <input type="range" id="scaleSlider" min="0" max="200" value="__DEFAULT_SCALE__" oninput="onScaleInput()">
        </label>
        <span id="scaleValue">__DEFAULT_SCALE__&times;</span>
      </div>
      <div id="plot3d"></div>
      <div id="vizCards" class="viz-cards"></div>
      <div id="nonlinearBox"></div>
      <p class="note"><strong>주.</strong> 보-기둥 접합은 강접(rigid)으로 모델링하였으며, 변형 형상은 위 배율로 과장 표시된다.</p>
    </div>
  </section>

  <!-- 7. 고유치 / 동적 -->
  <section class="page sec-page" id="s7">
    <div class="run-head"><span class="rh-proj" id="rh7"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">9</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">7.</span>고유치 / 동적해석</h2><div id="sec7"></div></div>
  </section>

  <!-- 8. 부재 검토 -->
  <section class="page sec-page" id="s8">
    <div class="run-head"><span class="rh-proj" id="rh8"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">10</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">8.</span>부재 검토</h2><div id="sec8"></div></div>
  </section>

  <!-- 9. 층간변위 -->
  <section class="page sec-page" id="s9">
    <div class="run-head"><span class="rh-proj" id="rh9"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">11</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">9.</span>층간변위 검토</h2><div id="sec9"></div></div>
  </section>

  <!-- 10. 종합 결론 -->
  <section class="page sec-page" id="s10">
    <div class="run-head"><span class="rh-proj" id="rh10"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">12</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">10.</span>종합 결론</h2><div id="sec10"></div></div>
  </section>

  <!-- 부록 표지 -->
  <section class="page sec-page cover" id="sapp">
    <div class="run-head"><span class="rh-proj" id="rhapp"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">부록</span> -</div>
    <div class="page-body" style="display:flex;flex-direction:column;justify-content:center;">
      <div class="cover-top"><div class="cover-kind">A P P E N D I X</div></div>
      <div class="cover-title" style="font-size:24pt;margin:18mm 0 4mm 0;">부 록 A</div>
      <div class="cover-sub" style="font-size:12pt;">부 재 력 상 세 &nbsp;·&nbsp; 반 력 &nbsp;·&nbsp; 하 중 조 합</div>
      <hr class="cover-rule">
      <p class="note" style="text-align:center;">A.1 전체 부재 검토 일람표 &nbsp;·&nbsp; A.2 지점 반력 &nbsp;·&nbsp; A.3 하중조합별 포락 요약을 수록한다.</p>
    </div>
  </section>

  <!-- 부록 A.1 — 전체 부재 검토 일람표 -->
  <section class="page sec-page" id="sappA1">
    <div class="run-head"><span class="rh-proj"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">부록 A.1</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">A.1</span>전체 부재 검토 일람표</h2><div id="appMembers"></div></div>
  </section>

  <!-- 부록 A.2 — 지점 반력 -->
  <section class="page sec-page" id="sappA2">
    <div class="run-head"><span class="rh-proj"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">부록 A.2</span> -</div>
    <div class="page-body">
      <h2 class="sec"><span class="sec-no">A.2</span>지점 반력</h2>
      <div class="viz-controls"><label>하중조합:
        <select id="appRxnCase" onchange="renderAppReactions()"></select>
      </label></div>
      <div id="appReactions"></div>
    </div>
  </section>

  <!-- 부록 A.3 — 하중조합별 포락 요약 -->
  <section class="page sec-page" id="sappA3">
    <div class="run-head"><span class="rh-proj"></span><span class="rh-doc">구 조 계 산 서</span></div>
    <div class="run-foot">- <span class="pg-cur">부록 A.3</span> -</div>
    <div class="page-body"><h2 class="sec"><span class="sec-no">A.3</span>하중조합별 포락 요약</h2><div id="appCombos"></div></div>
  </section>

</div><!-- /main -->

<script>
var DATA = __DATA_JSON__;
var DEFAULT_SCALE = __DEFAULT_SCALE__;
var SUMMARY = DATA.summary || {};
var currentCase = (DATA.all_names && DATA.all_names.length) ? DATA.all_names[0] : null;
var currentScale = DEFAULT_SCALE;

// ============================================================
// helpers
// ============================================================
function esc(s){ if (s === null || s === undefined) return ''; return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function num(v, d){ if (typeof v !== 'number' || !isFinite(v)) return '-'; return v.toFixed(d===undefined?2:d); }
function pill(ok){ return ok ? '<span class="pill ok">OK</span>' : '<span class="pill ng">NG</span>'; }
function el(id){ return document.getElementById(id); }
function setHTML(id, html){ var e = el(id); if (e) e.innerHTML = html; }

// 내보내기: 자체 완결형 HTML 다운로드 (서버 원본 우선, 실패 시 현재 DOM 직렬화)
function downloadReportHtml(){
  try {
    var a = document.createElement('a');
    a.href = window.location.pathname + window.location.search;  // 같은 출처 → 원본 파일 다운로드
    a.download = '구조계산서.html';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  } catch (e) {
    try {
      var blob = new Blob(['<!DOCTYPE html>\n' + document.documentElement.outerHTML], {type:'text/html'});
      var url = URL.createObjectURL(blob);
      var a2 = document.createElement('a'); a2.href = url; a2.download = '구조계산서.html';
      document.body.appendChild(a2); a2.click(); document.body.removeChild(a2);
      URL.revokeObjectURL(url);
    } catch (e2) { alert('내보내기 실패: ' + e2.message); }
  }
}

// 색상 판정 (활용률)
function utilColor(r){ return r <= 0.7 ? 'var(--ok)' : (r <= 1.0 ? '#9a5a00' : 'var(--ng)'); }

// ============================================================
// Severity (interpretation 우선, 없으면 design_check)
// ============================================================
function getSeverity(){
  var interp = DATA.interpretation, dc = DATA.design_check;
  if (interp && interp.severity) return interp.severity;
  if (dc) return (dc.overall_status === 'OK') ? 'safe' : 'moderate';
  return null;
}
var SEV_TEXT = {safe:'SAFE', marginal:'MARGINAL', moderate:'NG', severe:'NG (SEVERE)'};
var SEV_KO   = {safe:'적합', marginal:'여유부족', moderate:'부적합', severe:'부적합 (심각)'};
function isNG(){ var s = getSeverity(); return s === 'moderate' || s === 'severe'; }

// ============================================================
// Cover / TOC / running heads / sidebar
// ============================================================
function renderCover(){
  var c = DATA.cover || {};
  var st = c.stamp || {};
  function stampCell(role, p){
    p = p || {};
    var seal = p.seal
      ? '<span class="sealspot" style="border-style:solid;border-color:var(--ng);"><img src="' + esc(p.seal) + '" alt="(인)" style="width:100%;height:100%;object-fit:contain;border-radius:50%;"></span>'
      : '<span class="sealspot">(인)</span>';
    return '<td>' + seal
      + '<span class="field">자격 : ' + esc(p.qualification) + '</span>'
      + '<span class="field">성명 : ' + esc(p.name) + '</span>'
      + '<span class="field">자격번호 : ' + esc(p.license_no) + '</span></td>';
  }
  var h = '<div class="run-foot"><span class="pg-cur">표지</span></div>';
  h += '<div class="cover-top"><div class="cover-kind">S T R U C T U R A L &nbsp; C A L C U L A T I O N</div></div>';
  if (c.logo){
    h += '<div style="text-align:center;margin:7mm 0 0;"><img src="' + esc(c.logo) + '" alt="회사 로고" style="max-height:24mm;max-width:60%;object-fit:contain;"></div>';
  }
  h += '<div class="cover-title">' + esc(c.project_name) + '<br>구조계산서</div>';
  h += '<div class="cover-sub">' + esc(c.structure_type) + '</div>';
  h += '<hr class="cover-rule">';
  h += '<table class="titleblock"><tbody>'
    + '<tr><th>대지위치</th><td>' + esc(c.location) + '</td></tr>'
    + '<tr><th>건 축 주</th><td>' + esc(c.client) + '</td></tr>'
    + '<tr><th>구조형식</th><td>' + esc(c.structure_type) + '</td></tr>'
    + '<tr><th>규　　모</th><td>지상 ' + esc((DATA.model_info||{}).num_stories) + '층</td></tr>'
    + '<tr><th>작 성 자</th><td>' + esc(st.author && st.author.name) + ' (자격번호 ' + esc(st.author && st.author.license_no) + ')</td></tr>'
    + '<tr><th>작 성 일</th><td>' + esc(c.date) + '</td></tr>'
    + '</tbody></table>';
  h += '<div class="stamp-wrap"><div class="stamp-cap">도 장 란</div><table class="stamp">'
    + '<thead><tr><th>작　성</th><th>검　토</th><th>승　인</th></tr></thead>'
    + '<tbody><tr>'
    + stampCell('author', st.author) + stampCell('reviewer', st.reviewer) + stampCell('approver', st.approver)
    + '</tr></tbody></table></div>';
  h += '<div class="meta-foot">본 구조계산서는 KDS 41 10 00 외 관련 설계기준에 따라 작성됨. 작성업체: ' + esc(c.firm) + '</div>';
  el('coverPage').innerHTML = h;
}

function renderTOC(){
  var c = DATA.cover || {};
  var h = '<div class="run-head"><span class="rh-proj">' + esc(c.project_name) + '</span><span class="rh-doc">구 조 계 산 서</span></div>';
  h += '<div class="run-foot">- <span class="pg-cur">2</span> -</div>';
  h += '<div class="page-body"><h2 class="sec"><span class="sec-no">　</span>목　　차</h2><table class="toc"><tbody>';
  var rows = [
    ['1.', '구조설계 개요'], ['2.', '적용 기준'], ['3.', '사용 재료'], ['4.', '하중 산정'],
    ['5.', '하중 조합'], ['6.', '해석 개요'], ['7.', '고유치 / 동적해석'], ['8.', '부재 검토'],
    ['9.', '층간변위 검토'], ['10.', '종합 결론']
  ];
  for (var i = 0; i < rows.length; i++){
    h += '<tr><td class="toc-no">' + rows[i][0] + '</td><td>' + rows[i][1] + '</td><td class="toc-pg">' + (i+3) + '</td></tr>';
  }
  h += '<tr class="toc-app"><td class="toc-no">부록</td><td>부록 A — 부재력 상세 / 반력 / 하중도</td><td class="toc-pg">부록</td></tr>';
  h += '</tbody></table></div>';
  el('tocPage').innerHTML = h;
}

function renderRunningHeads(){
  var name = (DATA.cover||{}).project_name || '';
  // 모든 러닝헤드(.rh-proj) 일괄 설정 — 본문 §1~10 + 부록 표지/A.1~A.3 포함
  document.querySelectorAll('.run-head .rh-proj').forEach(function(e){ e.textContent = name; });
}

function renderSidebar(){
  var c = DATA.cover || {};
  var mi = DATA.model_info || {};
  el('sbSub').textContent = (c.project_name || '') + ' · ' + (mi.num_stories || '?') + 'F';
  var sev = getSeverity();
  var v = el('sbVerdict');
  if (!sev){ v.innerHTML = ''; return; }
  var ng = isNG();
  var txt = SEV_TEXT[sev] || sev.toUpperCase();
  v.innerHTML = '<div class="sb-verdict' + (ng?' ng':'') + '"><div class="vlabel">종합판정</div>'
    + '<span class="vbadge">' + txt + '</span></div>';
}

// ============================================================
// 상단 메트릭 스트립 (renderOverviewBanner 로직 재사용)
// ============================================================
function renderStrip(){
  var interp = DATA.interpretation, dc = DATA.design_check, ma = DATA.modal_analysis;
  var strip = el('summaryStrip');
  if (!dc && !interp){ strip.style.display = 'none'; return; }

  var driftR = (dc && dc.summary) ? (dc.summary.max_drift_ratio || 0) : 0;
  var interR = (dc && dc.summary) ? (dc.summary.max_interaction_ratio || 0) : 0;
  var maxUtil = Math.max(driftR, interR);
  var ngU = maxUtil > 1.0;

  var h = '<span class="strip-title">검토요약</span>';
  h += '<span class="chip' + (ngU?' ng':'') + '"><span class="ck">최대 활용률</span><b>' + Math.round(maxUtil*100) + '%</b></span>';

  // 지배 검토
  var governing = (driftR >= interR) ? '층간변위' : '부재';
  h += '<span class="chip"><span class="ck">지배검토</span><b>' + governing + ' ' + maxUtil.toFixed(3) + '</b></span>';

  // 위험층
  var critStory = '-';
  if (interp && interp.drift_interpretation && interp.drift_interpretation.critical_story){
    critStory = interp.drift_interpretation.critical_story + 'F';
  } else if (interp && interp.member_interpretation && interp.member_interpretation.ng_by_story){
    var ngs = interp.member_interpretation.ng_by_story, maxNg = 0, maxS = null;
    for (var s in ngs){ if (ngs[s] > maxNg){ maxNg = ngs[s]; maxS = s; } }
    if (maxS) critStory = maxS + 'F';
  }
  h += '<span class="chip' + (ngU?' ng':'') + '"><span class="ck">위험층</span><b>' + critStory + '</b></span>';

  // 위험부재
  if (interp && interp.member_interpretation && interp.member_interpretation.weakest_link){
    var wl = interp.member_interpretation.weakest_link;
    var wlType = {column:'Col', beam_x:'Bm-X', beam_y:'Bm-Y'}[wl.type] || wl.type;
    var wlNG = wl.ratio > 1.0;
    h += '<span class="chip' + (wlNG?' ng':'') + '"><span class="ck">위험부재</span><b>' + wlType + ' #' + wl.member_id + '</b></span>';
  }

  // T1 / Ta
  if (interp && interp.modal_interpretation){
    var m = interp.modal_interpretation;
    var t1 = m.T1_actual_s || 0, ta = m.Ta_empirical_s || 0, rt = m.T1_Ta_ratio || 0;
    var flexNG = !!m.flexibility_flag;
    h += '<span class="chip' + (flexNG?' ng':'') + '"><span class="ck">T1 / Ta</span><b>' + t1.toFixed(3) + ' / ' + ta.toFixed(3) + ' (' + rt.toFixed(2) + ')</b></span>';
    var dir = m.first_mode_direction || '-';
    h += '<span class="chip"><span class="ck">1차모드</span><b>' + dir + '</b></span>';
  } else if (ma && ma.fundamental_periods){
    var fp = ma.fundamental_periods;
    var t1v = Math.max(fp.T1_x_s || 0, fp.T1_y_s || 0);
    if (t1v > 0) h += '<span class="chip"><span class="ck">T1</span><b>' + t1v.toFixed(3) + 's</b></span>';
  }
  strip.innerHTML = h;
}

// ============================================================
// §1 구조설계 개요
// ============================================================
function renderSec1(){
  var mi = DATA.model_info || {};
  var ms = SUMMARY.model_settings || {};
  var h = '<h3 class="sub">1.1 건축물 개요</h3>';
  h += '<table class="tbl kv"><tbody>';
  h += '<tr><th>규　　모</th><td>지상 ' + esc(mi.num_stories) + '층</td></tr>';
  if ((mi.num_bays_x || 0) > 0 || (mi.num_bays_y || 0) > 0){
    h += '<tr><th>경간 구성</th><td>X방향 ' + esc(mi.num_bays_x) + '경간 × Y방향 ' + esc(mi.num_bays_y) + '경간</td></tr>';
  } else {
    h += '<tr><th>모델 규모</th><td>' + esc(mi.num_nodes) + ' 절점 / ' + esc(mi.num_elements) + ' 요소</td></tr>';
  }
  var stories = mi.stories || [];
  h += '<tr><th>층고 구성</th><td>' + (stories.length ? esc(stories.join(', ')) + ' m' : '[입력]') + '</td></tr>';
  h += '<tr><th>총 높이</th><td>' + num(mi.total_height, 1) + ' m</td></tr>';
  // 건축면적 / 연면적 (사용자 입력 우선, 없으면 외곽치수×층수 자동 산정)
  var ar = DATA.areas || {};
  function areaStr(v, src){
    if (typeof v !== 'number') return '[입력]';
    var s = v.toLocaleString(undefined, {maximumFractionDigits:1}) + ' m²';
    if (src === 'auto') s += ' <span class="unit">(자동 산정)</span>';
    return s;
  }
  h += '<tr><th>건축면적</th><td>' + areaStr(ar.building_area_m2, ar.building_area_source) + '</td></tr>';
  h += '<tr><th>연면적</th><td>' + areaStr(ar.gross_floor_area_m2, ar.gross_floor_area_source) + '</td></tr>';
  h += '<tr><th>구조 형식</th><td>' + esc((DATA.cover||{}).structure_type) + '</td></tr>';
  h += '</tbody></table>';

  // 내진 파라미터 — seismic report (error dict면 무효 처리)
  var sm_seismic = seismicReport();

  h += '<h3 class="sub">1.2 해석 설정</h3>';
  h += '<table class="tbl kv"><tbody>';
  h += '<tr><th>지점 조건</th><td>' + esc(ms.support_type || mi.supports || '[입력]') + '</td></tr>';
  h += '<tr><th>해석 종류</th><td>' + esc(ms.analysis_type || '[입력]') + '</td></tr>';
  h += '<tr><th>기하비선형</th><td>' + esc(ms.geometric_nonlinearity || '[입력]') + '</td></tr>';
  h += '<tr><th>Rigid Diaphragm</th><td>' + (ms.rigid_diaphragm ? '적용' : '미적용') + '</td></tr>';
  h += '</tbody></table>';

  h += '<h3 class="sub">1.3 내진설계 개요</h3>';
  if (sm_seismic){
    var s = sm_seismic;
    var ieLbl = ieToGrade(s.IE);
    h += '<table class="tbl kv"><tbody>';
    h += '<tr><th>지진구역 (지역)</th><td>' + esc(s.region) + '</td></tr>';
    h += '<tr><th>내진등급 / I<sub>e</sub></th><td>' + esc(ieLbl) + ' (I<sub>e</sub> = ' + num(s.IE, 2) + ')</td></tr>';
    h += '<tr><th>지반 종류</th><td>' + esc(s.site_class || '[입력]') + '</td></tr>';
    h += '<tr><th>내진시스템</th><td>' + esc(seismicSystemKO(s.seismic_system)) + '</td></tr>';
    h += '<tr><th>반응수정계수 R</th><td class="num">' + num(s.R, 1) + '</td></tr>';
    h += '<tr><th>설계스펙트럼가속도 S<sub>DS</sub></th><td class="num">' + num(s.SDS, 3) + ' g</td></tr>';
    h += '<tr><th>설계스펙트럼가속도 S<sub>D1</sub></th><td class="num">' + num(s.SD1, 3) + ' g</td></tr>';
    h += '</tbody></table>';
    h += '<p class="note"><strong>주.</strong> 내진 파라미터는 ' + esc(s.code || 'KDS 41 17 00') + ' 등가정적해석법에 따라 산정되었다.</p>';
  } else {
    h += '<p class="empty-note">지진하중 미생성 (지역 미지정 또는 하중조건 미포함) — 내진 파라미터 [입력].</p>';
    h += '<table class="tbl kv"><tbody>';
    h += '<tr><th>내진등급 / I<sub>e</sub></th><td>[입력]</td></tr>';
    h += '<tr><th>반응수정계수 R</th><td>[입력]</td></tr>';
    h += '<tr><th>설계스펙트럼 S<sub>DS</sub> / S<sub>D1</sub></th><td>[입력]</td></tr>';
    h += '</tbody></table>';
  }
  setHTML('sec1', h);
}

// seismic report (유효한 경우만 반환, error dict/누락 → null)
function seismicReport(){
  var s = (DATA.loads || {}).seismic;
  if (!s || s.error || typeof s.SDS !== 'number') return null;
  return s;
}
function windReport(){
  var w = (DATA.loads || {}).wind;
  if (!w || w.error || typeof w.V0_ms !== 'number') return null;
  return w;
}
function snowReport(){
  var s = (DATA.loads || {}).snow;
  if (!s || s.error || typeof s.S_design_kNm2 !== 'number') return null;
  return s;
}
function gravityReport(){
  var g = (DATA.loads || {}).gravity;
  return (Array.isArray(g) && g.length) ? g : null;
}
// IE → 내진등급 라벨 (특/중요/일반)
function ieToGrade(ie){
  if (typeof ie !== 'number') return '[입력]';
  if (ie >= 1.5) return '특 (등급 I)';
  if (ie >= 1.2) return '중요 (등급 II)';
  return '일반 (등급 III)';
}
// 내진시스템 key → 한글
var SEISMIC_SYS_KO = {
  special_moment_frame:'철골 특수모멘트골조', intermediate_moment_frame:'철골 중간모멘트골조',
  ordinary_moment_frame:'철골 보통모멘트골조', rc_special_moment_frame:'RC 특수모멘트골조',
  rc_intermediate_moment_frame:'RC 중간모멘트골조', rc_ordinary_moment_frame:'RC 보통모멘트골조',
  rc_special_shear_wall:'RC 특수전단벽', rc_ordinary_shear_wall:'RC 보통전단벽',
  special_braced_frame:'철골 특수가새골조', ordinary_braced_frame:'철골 보통가새골조'
};
function seismicSystemKO(k){ return SEISMIC_SYS_KO[k] || (k || '[입력]'); }

// ============================================================
// §2 적용 기준
// ============================================================
function renderSec2(){
  var dc = DATA.design_check || {};
  var driftRef = (dc.drift_check && dc.drift_check.code_ref) ? dc.drift_check.code_ref : 'KDS 41 17 00';
  var memberRef = (dc.member_check && dc.member_check.code_ref) ? dc.member_check.code_ref : 'KDS 14 31 00';
  var h = '<table class="tbl"><thead><tr>'
    + '<th class="ctr" style="width:20%">코드번호</th><th>기준 명칭</th><th class="ctr" style="width:26%">적용 범위</th>'
    + '</tr></thead><tbody>'
    + '<tr><td class="ctr">KDS 41 10 00</td><td>건축구조기준 일반사항</td><td class="ctr">총칙·설계법</td></tr>'
    + '<tr><td class="ctr">KDS 41 12 00</td><td>건축물 설계하중</td><td class="ctr">고정·활·설·풍하중</td></tr>'
    + '<tr><td class="ctr">KDS 41 17 00</td><td>건축물 내진설계기준</td><td class="ctr">지진하중·층간변위</td></tr>'
    + '<tr><td class="ctr">KDS 14 31 00</td><td>강구조 설계기준 (LRFD)</td><td class="ctr">부재 강도검토</td></tr>'
    + '</tbody></table>';
  h += '<p class="note"><strong>주.</strong> 층간변위 검토는 <code>' + esc(driftRef) + '</code>, 부재 강도검토는 <code>' + esc(memberRef) + '</code>을 준거로 한다.</p>';
  setHTML('sec2', h);
}

// ============================================================
// §3 사용 재료
// ============================================================
function renderSec3(){
  var mi = DATA.model_info || {};
  var sp = DATA.sections || {};
  var fy = sp.fy_MPa;
  var fyStr = (typeof fy === 'number') ? Math.round(fy).toLocaleString() : '[입력]';
  var fu = sp.fu_MPa;
  var fuStr = (typeof fu === 'number') ? Math.round(fu).toLocaleString() : '[입력]';

  var h = '<h3 class="sub">3.1 강재 물성</h3>';
  h += '<table class="tbl"><thead><tr>'
    + '<th>재료</th><th class="num">탄성계수 E <span class="unit">(MPa)</span></th>'
    + '<th class="num">전단계수 G <span class="unit">(MPa)</span></th>'
    + '<th class="num">항복강도 F<sub>y</sub> <span class="unit">(MPa)</span></th>'
    + '<th class="num">인장강도 F<sub>u</sub> <span class="unit">(MPa)</span></th></tr></thead><tbody>';
  h += '<tr><td>' + esc(mi.material_name || '[입력]') + '</td>'
    + '<td class="num">' + (typeof mi.E_MPa === 'number' ? Math.round(mi.E_MPa).toLocaleString() : '[입력]') + '</td>'
    + '<td class="num">' + (typeof mi.G_MPa === 'number' ? Math.round(mi.G_MPa).toLocaleString() : '[입력]') + '</td>'
    + '<td class="num">' + fyStr + '</td>'
    + '<td class="num">' + fuStr + '</td></tr>';
  h += '</tbody></table>';

  // mm² → cm², mm⁴ → cm⁴ 환산 표기
  function cm2(v){ return (typeof v === 'number') ? (v/100).toFixed(2) : '[입력]'; }        // mm²→cm²
  function cm4(v){ return (typeof v === 'number') ? (v/1e4).toLocaleString(undefined,{maximumFractionDigits:0}) : '[입력]'; } // mm⁴→cm⁴
  function hb(it){ return (typeof it.h_mm === 'number' && typeof it.b_mm === 'number') ? (Math.round(it.h_mm) + '×' + Math.round(it.b_mm)) : '[입력]'; }

  h += '<h3 class="sub">3.2 부재 단면 제원</h3>';
  h += '<table class="tbl"><thead><tr>'
    + '<th>부재</th><th class="ctr">단면 규격</th>'
    + '<th class="num">A <span class="unit">(cm²)</span></th>'
    + '<th class="num">I<sub>x</sub> <span class="unit">(cm⁴)</span></th>'
    + '<th class="num">I<sub>y</sub> <span class="unit">(cm⁴)</span></th>'
    + '<th class="num">Z<sub>x</sub> <span class="unit">(cm³)</span></th>'
    + '<th class="num">H×B <span class="unit">(mm)</span></th></tr></thead><tbody>';
  // mm³ → cm³
  function cm3(v){ return (typeof v === 'number') ? (v/1e3).toLocaleString(undefined,{maximumFractionDigits:0}) : '[입력]'; }
  var rows = [['기둥 (Column)', 'column'], ['보 (X방향)', 'beam_x'], ['보 (Y방향)', 'beam_y']];
  var items = sp.items || {};
  for (var i = 0; i < rows.length; i++){
    var it = items[rows[i][1]] || {};
    var nm = it.name || mi[rows[i][1] + '_section'] || '[입력]';
    h += '<tr><td>' + rows[i][0] + '</td><td class="ctr">' + esc(nm) + '</td>'
      + '<td class="num">' + cm2(it.A_mm2) + '</td>'
      + '<td class="num">' + cm4(it.Ix_mm4) + '</td>'
      + '<td class="num">' + cm4(it.Iy_mm4) + '</td>'
      + '<td class="num">' + cm3(it.Zx_mm3) + '</td>'
      + '<td class="num">' + hb(it) + '</td></tr>';
  }
  h += '</tbody></table>';
  h += '<p class="note"><strong>주.</strong> 단면 제원(A, I<sub>x</sub>, I<sub>y</sub>, H×B)은 KS D 3502 열간압연 H형강 DB값을 적용한다. 소성단면계수 Z<sub>x</sub>는 단면 형상(b·t<sub>f</sub>·h·t<sub>w</sub>)에서 산정(휨강도 검토 M<sub>n</sub>=F<sub>y</sub>·Z<sub>x</sub>에 사용). F<sub>y</sub>·F<sub>u</sub>는 입력 강종(KS D 3503/3515/3866, t≤16mm 기준)에 따른다.</p>';
  setHTML('sec3', h);
}

// ============================================================
// §4 하중 산정 (DL/LL/W/E 블록, load_result 실측치)
// ============================================================
function renderSec4(){
  var h = '';
  h += renderGravityBlock();
  h += renderSnowBlock();
  h += renderWindBlock();
  h += renderSeismicBlock();
  setHTML('sec4', h);
}

// 4.2 적설하중 (S) — snow report
function renderSnowBlock(){
  var s = snowReport();
  var h = '<h3 class="sub">4.2 적설하중 (S)</h3>';
  if (!s){
    return h + '<p class="empty-note">해당 없음 (지역 미지정 또는 기본지상적설하중 데이터 없음).</p>';
  }
  h += '<table class="tbl kv"><tbody>';
  h += '<tr><th>기본지상적설하중 S<sub>g</sub></th><td class="num">' + num(s.Sg_kNm2, 3) + ' kN/m²</td></tr>';
  h += '<tr><th>지붕적설하중계수 C<sub>b</sub></th><td class="num">' + num(s.Cb, 2) + '</td></tr>';
  h += '<tr><th>노출계수 C<sub>e</sub> / 온도계수 C<sub>t</sub></th><td class="num">' + num(s.Ce, 2) + ' / ' + num(s.Ct, 2) + '</td></tr>';
  h += '<tr><th>중요도계수 I<sub>s</sub></th><td class="num">' + num(s.Is, 2) + '</td></tr>';
  h += '<tr><th>지붕적설하중 S (설계값)</th><td class="num">' + num(s.S_design_kNm2, 3) + ' kN/m²</td></tr>';
  h += '<tr><th>적용 위치 / 지붕면적</th><td>' + esc(s.roof_story) + '층(지붕) / ' + num(s.roof_area_m2, 1) + ' m²</td></tr>';
  h += '</tbody></table>';
  h += '<div class="diag-box" style="font-family:monospace;font-size:9pt">'
    + 'S = C<sub>b</sub>·C<sub>e</sub>·C<sub>t</sub>·I<sub>s</sub>·S<sub>g</sub> = ' + num(s.Cb,2) + '×' + num(s.Ce,2) + '×' + num(s.Ct,2) + '×' + num(s.Is,2) + '×' + num(s.Sg_kNm2,3) + ' = ' + num(s.S_formula_kNm2,3) + ' kN/m²<br>'
    + '최소 지붕적설하중 S<sub>min</sub> = ' + num(s.S_min_kNm2,3) + ' kN/m²  →  설계값 = max(S, S<sub>min</sub>) = <b>' + num(s.S_design_kNm2,3) + '</b> kN/m² (' + (s.governed_by === 'minimum' ? '최소값 지배' : '산정식 지배') + ')</div>';
  h += '<p class="note"><strong>주.</strong> ' + esc(s.code || 'KDS 41 12 00 §4') + ' 평지붕 약식 (C<sub>e</sub>=C<sub>t</sub>=1.0 가정). 적설은 최상층 지붕에만 작용하며, 적설조합(1.2D+1.6S 등)으로 검토한다.</p>';
  return h;
}

// 4.1 중력하중 (DL/LL) — gravity report (list[per-story])
function renderGravityBlock(){
  var g = gravityReport();
  var h = '<h3 class="sub">4.1 중력하중 (DL / LL)</h3>';
  if (!g){
    return h + '<p class="empty-note">해당 없음 (입력 조건에 미포함).</p>';
  }
  h += '<table class="tbl"><thead><tr>'
    + '<th class="ctr">층</th><th class="ctr">용도</th>'
    + '<th class="num">고정하중 DL <span class="unit">(kN/m²)</span></th>'
    + '<th class="num">활하중 LL <span class="unit">(kN/m²)</span></th>'
    + '<th>DL 내역 (슬래브+마감+설비)</th></tr></thead><tbody>';
  for (var i = 0; i < g.length; i++){
    var r = g[i], bd = r.DL_breakdown || {};
    var bdStr = (typeof bd.slab_self === 'number')
      ? (num(bd.slab_self,2) + ' + ' + num(bd.finish,2) + ' + ' + num(bd.mep,2))
      : '-';
    h += '<tr><td class="ctr">' + esc(r.story) + '</td>'
      + '<td class="ctr">' + esc(r.usage || '-') + '</td>'
      + '<td class="num">' + num(r.DL_kNm2, 2) + '</td>'
      + '<td class="num">' + num(r.LL_kNm2, 2) + '</td>'
      + '<td>' + bdStr + '</td></tr>';
  }
  h += '</tbody></table>';
  h += '<p class="note"><strong>주.</strong> DL = 슬래브 자중(γ<sub>RC</sub>×t) + 마감재 + 설비(0.5). LL은 KDS 41 12 00 용도별 DB값.</p>';
  return h;
}

// 4.2 풍하중 (W) — wind report
function renderWindBlock(){
  var w = windReport();
  var h = '<h3 class="sub">4.3 풍하중 (W)</h3>';
  if (!w){
    return h + '<p class="empty-note">해당 없음 (입력 조건에 미포함).</p>';
  }
  h += '<table class="tbl kv"><tbody>';
  h += '<tr><th>기본풍속 V<sub>0</sub></th><td class="num">' + num(w.V0_ms, 1) + ' m/s</td></tr>';
  h += '<tr><th>노풍도</th><td>' + esc(w.exposure) + '</td></tr>';
  h += '<tr><th>가스트영향계수 G<sub>f</sub></th><td class="num">' + num(w.Gf, 2) + '</td></tr>';
  h += '<tr><th>풍력계수 C<sub>p</sub> (풍상+풍하)</th><td class="num">' + num(w.Cp_total, 2) + '</td></tr>';
  h += '<tr><th>밑면 풍력 ΣF<sub>x</sub> / ΣF<sub>y</sub></th><td class="num">' + num(w.total_Fx_kN, 1) + ' / ' + num(w.total_Fy_kN, 1) + ' kN</td></tr>';
  h += '</tbody></table>';
  // 산정식 + 층별 (대표 층의 qz/p)
  var det = w.story_detail || [];
  if (det.length){
    var top = det[det.length-1];
    h += '<div class="diag-box" style="font-family:monospace;font-size:9pt">'
      + 'q<sub>z</sub> = 0.5·ρ·(V<sub>0</sub>·K<sub>z</sub>·K<sub>zt</sub>)²   ·   p = q<sub>z</sub>·G<sub>f</sub>·C<sub>p</sub><br>'
      + '예) ' + esc(top.story) + '층: K<sub>z</sub>=' + num(top.Kz,3) + ', q<sub>z</sub>=' + num(top.qz_kNm2,4) + ' kN/m², p=' + num(top.p_kNm2,4) + ' kN/m²'
      + '</div>';
    h += '<table class="tbl"><thead><tr><th class="ctr">층</th>'
      + '<th class="num">중앙높이 <span class="unit">(m)</span></th><th class="num">K<sub>z</sub></th>'
      + '<th class="num">q<sub>z</sub> <span class="unit">(kN/m²)</span></th><th class="num">p <span class="unit">(kN/m²)</span></th>'
      + '<th class="num">F<sub>x</sub> <span class="unit">(kN)</span></th><th class="num">F<sub>y</sub> <span class="unit">(kN)</span></th></tr></thead><tbody>';
    for (var i = det.length-1; i >= 0; i--){
      var d = det[i];
      h += '<tr><td class="ctr">' + esc(d.story) + '</td>'
        + '<td class="num">' + num(d.h_mid_m,2) + '</td>'
        + '<td class="num">' + num(d.Kz,3) + '</td>'
        + '<td class="num">' + num(d.qz_kNm2,4) + '</td>'
        + '<td class="num">' + num(d.p_kNm2,4) + '</td>'
        + '<td class="num">' + num(d.Fx_kN,2) + '</td>'
        + '<td class="num">' + num(d.Fy_kN,2) + '</td></tr>';
    }
    h += '</tbody></table>';
  }
  h += '<p class="note"><strong>주.</strong> ' + esc(w.code || 'KDS 41 12 00 §5') + ' 약식 절차 (K<sub>zt</sub>=1.0 평탄지 가정).</p>';
  return h;
}

// 4.3 지진하중 (E) — seismic report
function renderSeismicBlock(){
  var s = seismicReport();
  var h = '<h3 class="sub">4.4 지진하중 (E) — 등가정적해석법</h3>';
  if (!s){
    return h + '<p class="empty-note">해당 없음 (지역 미지정 또는 입력 조건에 미포함).</p>';
  }
  h += '<table class="tbl kv"><tbody>';
  h += '<tr><th>S<sub>DS</sub> / S<sub>D1</sub></th><td class="num">' + num(s.SDS,3) + ' / ' + num(s.SD1,3) + ' g</td></tr>';
  h += '<tr><th>R / I<sub>e</sub></th><td class="num">' + num(s.R,1) + ' / ' + num(s.IE,2) + '</td></tr>';
  h += '<tr><th>근사고유주기 T<sub>a</sub></th><td class="num">' + num(s.Ta_sec,3) + ' s</td></tr>';
  h += '<tr><th>설계주기 T (= min(T<sub>a</sub>, C<sub>u</sub>T<sub>a</sub>))</th><td class="num">' + num(s.T_sec,3) + ' s</td></tr>';
  h += '<tr><th>지진응답계수 C<sub>s</sub></th><td class="num">' + num(s.Cs,4) + ' (한계 ' + num(s.Cs_min,4) + '~' + num(s.Cs_max,4) + ')</td></tr>';
  h += '<tr><th>유효중량 W</th><td class="num">' + num(s.W_kN,1) + ' kN</td></tr>';
  h += '<tr><th>밑면전단력 V (= C<sub>s</sub>·W)</th><td class="num">' + num(s.V_kN,1) + ' kN</td></tr>';
  h += '</tbody></table>';
  h += '<div class="diag-box" style="font-family:monospace;font-size:9pt">'
    + 'C<sub>s</sub> = S<sub>DS</sub> / (R / I<sub>e</sub>) = ' + num(s.SDS,3) + ' / (' + num(s.R,1) + ' / ' + num(s.IE,2) + ') = ' + num(s.Cs,4) + '<br>'
    + 'V = C<sub>s</sub>·W = ' + num(s.Cs,4) + ' × ' + num(s.W_kN,1) + ' = ' + num(s.V_kN,1) + ' kN</div>';
  // story_forces 표
  var sf = s.story_forces || [];
  if (sf.length){
    h += '<table class="tbl"><thead><tr><th class="ctr">층</th>'
      + '<th class="num">중량 w <span class="unit">(kN)</span></th>'
      + '<th class="num">높이 h <span class="unit">(m)</span></th>'
      + '<th class="num">C<sub>vx</sub></th>'
      + '<th class="num">층전단력 F<sub>x</sub> <span class="unit">(kN)</span></th></tr></thead><tbody>';
    for (var i = sf.length-1; i >= 0; i--){
      var f = sf[i];
      h += '<tr><td class="ctr">' + esc(f.story) + '</td>'
        + '<td class="num">' + num(f.weight_kN,1) + '</td>'
        + '<td class="num">' + num(f.height_m,2) + '</td>'
        + '<td class="num">' + num(f.Cvx,4) + '</td>'
        + '<td class="num">' + num(f.Fx_kN,2) + '</td></tr>';
    }
    h += '</tbody></table>';
  }
  h += '<p class="note"><strong>주.</strong> ' + esc(s.code || 'KDS 41 17 00') + ' 등가정적해석. 수직분포 지수 k=' + num(s.k,2) + ', C<sub>vx</sub>=(w·h<sup>k</sup>)/Σ(w·h<sup>k</sup>). 수직지진성분 E<sub>v</sub>=0.2·S<sub>DS</sub>·D(=' + num(0.2*s.SDS, 3) + '·D)는 지진 하중조합의 고정하중 계수에 반영(1.2→' + num(1.2+0.2*s.SDS, 3) + ', 0.9→' + num(0.9-0.2*s.SDS, 3) + ').</p>';
  return h;
}

// ============================================================
// §5 하중 조합 (load_combinations 계수 표)
// ============================================================
function renderSec5(){
  var combos = (DATA.loads || {}).combinations;   // {name:{case:factor}} | null
  var h = '<p class="note">KDS 41 10 00 강도설계법(LRFD) 하중조합을 적용한다. 부재검토 및 층간변위 검토는 포락(envelope)값을 사용한다.</p>';

  // 등장하는 모든 케이스 헤더 수집
  var caseSet = {}, order = [];
  function addCase(c){ if (!(c in caseSet)){ caseSet[c] = true; order.push(c); } }
  if (combos && typeof combos === 'object'){
    // 일반적 케이스 순서 우선
    var pref = ['DL','LL','S','EQX','EQY','WX','WY'];
    for (var p = 0; p < pref.length; p++) {
      for (var nm in combos){ if (combos[nm] && (pref[p] in combos[nm])) { addCase(pref[p]); break; } }
    }
    for (var nm in combos){ for (var c in combos[nm]) addCase(c); }
  }

  if (combos && order.length){
    h += '<table class="tbl"><thead><tr><th class="ctr" style="width:9%">번호</th><th>조합명</th>';
    for (var i = 0; i < order.length; i++) h += '<th class="num">' + esc(order[i]) + '</th>';
    h += '<th class="ctr">구분</th></tr></thead><tbody>';
    var idx = 1;
    for (var name in combos){
      var f = combos[name] || {};
      // 강도/사용성 구분: 계수>1 또는 지진/풍 포함 → 강도, 전부 1.0 이하 & 중력만 → 사용성
      var maxFactor = 0, hasLateral = false;
      for (var c in f){ maxFactor = Math.max(maxFactor, Math.abs(f[c])); if (['EQX','EQY','WX','WY'].indexOf(c) >= 0) hasLateral = true; }
      var kind = (maxFactor > 1.0 || hasLateral) ? '강도' : '사용성';
      h += '<tr><td class="ctr">C' + idx + '</td><td>' + esc(name) + '</td>';
      for (var j = 0; j < order.length; j++){
        var v = f[order[j]];
        h += '<td class="num">' + (typeof v === 'number' ? v.toFixed(2) : '-') + '</td>';
      }
      h += '<td class="ctr">' + kind + '</td></tr>';
      idx++;
    }
    h += '</tbody></table>';
    h += '<p class="note"><strong>주.</strong> 지진·풍하중 조합은 X·Y 양방향 및 ±부호에 대하여 각각 적용한다.</p>';
  } else {
    // combinations 미제공 시 combo_names로 폴백
    var cn = DATA.combo_names || [];
    if (cn.length){
      h += '<table class="tbl"><thead><tr><th class="ctr" style="width:14%">번호</th><th>하중조합</th></tr></thead><tbody>';
      for (var k = 0; k < cn.length; k++) h += '<tr><td class="ctr">C' + (k+1) + '</td><td>' + esc(cn[k]) + '</td></tr>';
      h += '</tbody></table>';
    } else {
      h += '<p class="empty-note">하중조합 정보 없음.</p>';
    }
  }
  setHTML('sec5', h);
}

// ============================================================
// §6 해석 개요 (표) + 3D 뷰는 별도 init
// ============================================================
function renderSec6(){
  var mi = DATA.model_info || {};
  var ms = SUMMARY.model_settings || {};
  var h = '<table class="tbl kv"><tbody>';
  h += '<tr><th>해석 모델</th><td>3차원 입체 프레임 (3D Frame)</td></tr>';
  h += '<tr><th>절점 수</th><td>' + esc(mi.num_nodes) + ' 개</td></tr>';
  h += '<tr><th>요소 수</th><td>' + esc(mi.num_elements) + ' 개 (보-기둥 요소, ' + esc(mi.num_members) + ' 부재)</td></tr>';
  h += '<tr><th>경계 조건</th><td>' + esc(ms.support_type || mi.supports || '[입력]') + ' (6-DOF)</td></tr>';
  h += '<tr><th>해석 종류</th><td>' + esc(ms.analysis_type || '[입력]') + '</td></tr>';
  h += '<tr><th>요소 유형</th><td>' + esc(ms.element_type || 'elasticBeamColumn') + '</td></tr>';
  h += '<tr><th>해석 소프트웨어</th><td>OpenSees (Open System for Earthquake Engineering Simulation)</td></tr>';
  h += '</tbody></table>';
  setHTML('sec6', h);

  // case select 옵션
  var sel = el('caseSelect');
  if (sel){
    var opts = '';
    var cn = DATA.case_names || [], comb = DATA.combo_names || [];
    if (cn.length){ opts += '<optgroup label="하중케이스">'; for (var i=0;i<cn.length;i++) opts += '<option value="'+esc(cn[i])+'">'+esc(cn[i])+'</option>'; opts += '</optgroup>'; }
    if (comb.length){ opts += '<optgroup label="하중조합">'; for (var j=0;j<comb.length;j++) opts += '<option value="'+esc(comb[j])+'">'+esc(comb[j])+'</option>'; opts += '</optgroup>'; }
    sel.innerHTML = opts;
  }
  renderNonlinearBox();
}

function renderNonlinearBox(){
  var ns = SUMMARY.nonlinear_summary;
  if (!ns){ setHTML('nonlinearBox', ''); return; }
  var h = '<div class="diag-box"><div class="dh">기하비선형 해석 요약</div>';
  h += '<div>변환: <b>' + esc(ns.transformation) + '</b> · 솔버: ' + esc(ns.solver_algorithm) + (ns.fallback_used ? ' (fallback)' : '') + ' · 하중단계: ' + esc(ns.n_load_steps) + '</div>';
  if (ns.force_interpretation){
    var fi = ns.force_interpretation;
    h += '<div style="margin-top:4px;font-size:9pt;color:#555">신뢰도 — 변위/변위비: ' + esc(fi.displacement_drift) + ', 지점반력: ' + esc(fi.base_reactions) + ', 로컬요소력: ' + esc(fi.local_element_forces) + '</div>';
  }
  h += '</div>';
  setHTML('nonlinearBox', h);
}

// ============================================================
// §7 고유치 / 동적 (renderModalAnalysis 필드 재사용)
// ============================================================
function renderSec7(){
  var ma = DATA.modal_analysis;
  if (!ma || !ma.modes || !ma.modes.length){
    setHTML('sec7', '<p class="empty-note">고유치 해석 미수행.</p>');
    return;
  }
  var fp = ma.fundamental_periods || {};
  var h = '<h3 class="sub">7.1 고유주기 및 모드</h3>';
  var hasMP = ma.modes[0] && ma.modes[0].mass_participation;
  h += '<table class="tbl"><thead><tr>'
    + '<th class="ctr">모드</th><th class="num">주기 T <span class="unit">(s)</span></th>'
    + '<th class="num">진동수 <span class="unit">(Hz)</span></th><th class="ctr">주 거동</th>';
  if (hasMP) h += '<th class="num">X (%)</th><th class="num">Y (%)</th><th class="num">RZ (%)</th>';
  h += '</tr></thead><tbody>';
  var dirKO = {'TRAN-X':'X방향 병진', 'TRAN-Y':'Y방향 병진', 'ROTN-Z':'Z축 비틀림'};
  for (var i = 0; i < ma.modes.length; i++){
    var m = ma.modes[i];
    // 필드명 호환: period_s/frequency_hz (renderModalAnalysis), period/mass_participation
    var per = (typeof m.period_s === 'number') ? m.period_s : m.period;
    var fr  = (typeof m.frequency_hz === 'number') ? m.frequency_hz : (per ? 1/per : 0);
    h += '<tr><td class="ctr">' + esc(m.mode) + '</td>'
      + '<td class="num">' + num(per, 4) + '</td>'
      + '<td class="num">' + num(fr, 4) + '</td>'
      + '<td class="ctr">' + esc(dirKO[m.direction] || m.direction) + '</td>';
    if (hasMP){
      var mp = m.mass_participation || {};
      h += '<td class="num">' + num(mp.x_pct, 1) + '</td><td class="num">' + num(mp.y_pct, 1) + '</td><td class="num">' + num(mp.rz_pct, 1) + '</td>';
    }
    h += '</tr>';
  }
  h += '</tbody></table>';

  h += '<h3 class="sub">7.2 기본 주기 및 경험식 비교</h3>';
  h += '<table class="tbl kv"><tbody>';
  if (typeof fp.T1_x_s === 'number') h += '<tr><th>T<sub>1,x</sub> (X방향 병진)</th><td class="num">' + num(fp.T1_x_s, 4) + ' s</td></tr>';
  if (typeof fp.T1_y_s === 'number') h += '<tr><th>T<sub>1,y</sub> (Y방향 병진)</th><td class="num">' + num(fp.T1_y_s, 4) + ' s</td></tr>';
  if (typeof fp.T1_rz_s === 'number') h += '<tr><th>T<sub>1,rz</sub> (비틀림)</th><td class="num">' + num(fp.T1_rz_s, 4) + ' s</td></tr>';
  var cum = ma.cumulative_participation;
  if (cum){
    h += '<tr><th>누적 질량참여율 (X)</th><td class="num">' + num(cum.x_pct, 1) + ' %</td></tr>';
    h += '<tr><th>누적 질량참여율 (Y)</th><td class="num">' + num(cum.y_pct, 1) + ' %</td></tr>';
  }
  // 경험식 비교 — interpretation.modal_interpretation 사용 (있으면)
  var interp = DATA.interpretation;
  if (interp && interp.modal_interpretation){
    var m = interp.modal_interpretation;
    if (typeof m.Ta_empirical_s === 'number') h += '<tr><th>경험식 근사주기 T<sub>a</sub></th><td class="num">' + num(m.Ta_empirical_s, 3) + ' s</td></tr>';
    if (typeof m.T1_Ta_ratio === 'number'){
      var c = m.flexibility_flag ? 'color:var(--ng);font-weight:700;' : '';
      h += '<tr><th>주기 비 (T<sub>1</sub> / T<sub>a</sub>)</th><td class="num" style="' + c + '">' + num(m.T1_Ta_ratio, 2) + '</td></tr>';
    }
  }
  h += '</tbody></table>';
  setHTML('sec7', h);
}

// ============================================================
// §8 부재 검토 (renderDesignCheckTab member 로직 재사용)
// ============================================================
function renderSec8(){
  var dc = DATA.design_check;
  if (!dc || !dc.member_check){
    setHTML('sec8', '<p class="empty-note">부재 강도검토 데이터 없음 — 설계검토 미수행.</p>');
    return;
  }
  var mc = dc.member_check;
  var sm = mc.summary || {};
  var h = '<p class="note">' + esc(mc.code_ref) + '에 따라 부재군별 포락 부재력에 대한 조합내력비(P-M-V 상관)를 산정한다. 내력비 ≤ 1.0 을 만족하여야 한다.</p>';
  h += '<p class="note">총 ' + esc(sm.total) + '개 부재 / OK ' + esc(sm.ok) + ' · NG ' + esc(sm.ng) + ' · 최대 상관비 ' + num(sm.max_interaction_ratio, 3) + '</p>';
  var show = mc.critical_members || (mc.members ? mc.members.slice(0,10) : []);
  if (show.length){
    h += '<table class="tbl"><thead><tr>'
      + '<th class="ctr">부재</th><th class="ctr">유형</th><th class="ctr">단면</th><th class="ctr">층</th>'
      + '<th class="num">P<sub>u</sub> <span class="unit">(kN)</span></th>'
      + '<th class="num">M<sub>ux</sub> <span class="unit">(kN·m)</span></th>'
      + '<th class="num">M<sub>uy</sub> <span class="unit">(kN·m)</span></th>'
      + '<th class="num">내력비</th><th class="ctr">공식</th><th class="ctr">판정</th></tr></thead><tbody>';
    var typeKO = {column:'기둥', beam_x:'보-X', beam_y:'보-Y'};
    for (var i = 0; i < show.length; i++){
      var m = show[i];
      var r = (m.ratios && typeof m.ratios.interaction === 'number') ? m.ratios.interaction : 0;
      var ok = (m.status === 'OK');
      var dmd = m.demand || {};
      h += '<tr' + (ok?'':' class="ng-row"') + '>'
        + '<td class="ctr">#' + esc(m.member_id) + '</td>'
        + '<td class="ctr">' + esc(typeKO[m.type] || m.type) + '</td>'
        + '<td class="ctr">' + esc(m.section) + '</td>'
        + '<td class="ctr">' + esc(m.story) + '</td>'
        + '<td class="num">' + num(dmd.Pu, 1) + '</td>'
        + '<td class="num">' + num(dmd.Mux, 1) + '</td>'
        + '<td class="num">' + num(dmd.Muy, 1) + '</td>'
        + '<td class="num" style="color:' + utilColor(r) + ';font-weight:700">' + num(r, 3) + '</td>'
        + '<td class="ctr" style="font-size:8.4pt">' + esc(m.ratios ? m.ratios.formula : '') + '</td>'
        + '<td class="ctr">' + pill(ok) + '</td></tr>';
    }
    h += '</tbody><tfoot><tr><td colspan="7">최대 상관비 / 부적합(NG) 부재</td>'
      + '<td class="num" style="color:' + (sm.ng>0?'var(--ng)':'var(--ok)') + '">' + num(sm.max_interaction_ratio, 3) + '</td>'
      + '<td colspan="2" class="ctr" style="color:' + (sm.ng>0?'var(--ng)':'var(--ok)') + '"><strong>' + esc(sm.ng) + ' NG</strong></td></tr></tfoot>';
    h += '</table>';
  }
  if (mc.assumptions && mc.assumptions.length){
    h += '<div class="diag-box"><div class="dh">검토 가정</div><ul style="margin:4px 0 0 16px">';
    for (var a = 0; a < mc.assumptions.length; a++) h += '<li>' + esc(mc.assumptions[a]) + '</li>';
    h += '</ul></div>';
  }
  setHTML('sec8', h);
}

// ============================================================
// §9 층간변위 (renderDesignCheckTab drift + renderDriftTab 재사용)
// ============================================================
function renderSec9(){
  var dc = DATA.design_check;
  var h = '';
  if (dc && dc.drift_check){
    var d = dc.drift_check;
    h += '<p class="note">' + esc(d.code_ref) + '에 따라 비탄성 층간변위비가 허용치(' + num(d.allowable, 3) + ', 중요도 ' + esc(d.importance) + ')를 초과하지 않아야 한다. 비율 = 변위비 / 허용 ≤ 1.0.</p>';
    h += '<p class="note">C<sub>d</sub> = ' + esc(d.Cd) + ', I<sub>E</sub> = ' + esc(d.IE) + '</p>';
    var checks = d.checks || [];
    if (checks.length){
      h += '<table class="tbl"><thead><tr>'
        + '<th class="ctr">층</th><th class="ctr">방향</th><th>조합</th>'
        + '<th class="num">탄성변위비</th><th class="num">비탄성변위비</th><th class="num">허용</th>'
        + '<th class="num">비율</th><th class="ctr">판정</th></tr></thead><tbody>';
      for (var i = 0; i < checks.length; i++){
        var c = checks[i];
        var ok = (c.status === 'OK');
        h += '<tr' + (ok?'':' class="ng-row"') + '>'
          + '<td class="ctr">' + esc(c.story) + '</td>'
          + '<td class="ctr">' + esc(c.direction) + '</td>'
          + '<td>' + esc(c.combo) + '</td>'
          + '<td class="num">' + num(c.elastic_drift, 6) + '</td>'
          + '<td class="num">' + num(c.inelastic_drift, 6) + '</td>'
          + '<td class="num">' + num(c.allowable, 3) + '</td>'
          + '<td class="num" style="color:' + utilColor(c.ratio) + ';font-weight:700">' + num(c.ratio, 3) + '</td>'
          + '<td class="ctr">' + pill(ok) + '</td></tr>';
      }
      var crit = d.critical;
      h += '</tbody><tfoot><tr><td colspan="8">최대 비율: '
        + (crit ? (esc(crit.story) + '층 ' + esc(crit.direction) + '방향 — ' + num(crit.ratio, 3) + ' (' + (d.status==='OK'?'OK':'NG') + ')') : num(d.max_ratio, 3))
        + '</td></tr></tfoot></table>';
    }
  } else {
    h += '<p class="empty-note">설계 층간변위 검토 데이터 없음 (지진 하중조합 또는 seismic_report 미제공).</p>';
  }

  // 케이스별 층별 drift (envelope) — case_data[currentCase].drifts 재사용
  if (currentCase && DATA.case_data && DATA.case_data[currentCase]){
    var drifts = DATA.case_data[currentCase].drifts || [];
    if (drifts.length){
      h += '<h3 class="sub">9.1 층별 층간변위 (' + esc(currentCase) + ')</h3>';
      h += '<table class="tbl"><thead><tr>'
        + '<th class="ctr">층</th><th class="num">층고 <span class="unit">(m)</span></th>'
        + '<th class="num">변위비 X</th><th class="num">1/X</th>'
        + '<th class="num">변위비 Y</th><th class="num">1/Y</th></tr></thead><tbody>';
      for (var k = 0; k < drifts.length; k++){
        var dr = drifts[k];
        h += '<tr><td class="ctr">' + esc(dr.story) + '</td>'
          + '<td class="num">' + num(dr.height_m, 1) + '</td>'
          + '<td class="num">' + num(dr.drift_x, 6) + '</td>'
          + '<td class="num">' + (dr.drift_x > 0 ? '1/' + Math.round(1/dr.drift_x) : '-') + '</td>'
          + '<td class="num">' + num(dr.drift_y, 6) + '</td>'
          + '<td class="num">' + (dr.drift_y > 0 ? '1/' + Math.round(1/dr.drift_y) : '-') + '</td></tr>';
      }
      h += '</tbody></table>';
    }
  }
  setHTML('sec9', h);
}

// ============================================================
// §10 종합 결론 (interpretation 서술 + diagnosis + suggestions + warnings)
// ============================================================
function renderSec10(){
  var interp = DATA.interpretation;
  var sev = getSeverity();
  var ng = isNG();
  var h = '';

  // 판정 박스
  var dc = DATA.design_check;

  // Tier1-4: 해석 실패(솔버 비수렴) 게이트 — 0 결과를 '안전'으로 표기하지 않음.
  if (dc && dc.analysis_error){
    var ae = dc.analysis_error || {};
    h += '<div class="verdict ng"><div class="vtag">판정 : 해석 실패 — 결과 신뢰 불가</div>'
      + '<div class="vsub">' + esc(ae.message || '구조해석이 비수렴/특이행렬로 실패하여 변위·부재력이 0으로 산출됨. 설계검토를 보류함.')
      + '</div></div>';
    setHTML('sec10', h);
    return;
  }

  var driftR = (dc && dc.summary) ? (dc.summary.max_drift_ratio || 0) : 0;
  var interR = (dc && dc.summary) ? (dc.summary.max_interaction_ratio || 0) : 0;
  var ngMembers = (dc && dc.summary) ? (dc.summary.ng_members || 0) : 0;
  // A2: 지배 하중조합 표기
  var govDrift = (dc && dc.summary) ? (dc.summary.drift_governing_combo || '') : '';
  var govMember = (dc && dc.summary) ? (dc.summary.member_governing_combo || '') : '';
  // B1: 보 처짐(사용성) — 미검토 시 명시(과대주장 방지)
  var deflR = (dc && dc.summary) ? dc.summary.max_deflection_ratio : null;
  var deflStat = (dc && dc.summary) ? dc.summary.deflection_status : null;
  var deflTxt = (deflStat === 'not_checked' || deflR == null)
    ? ' · 보 처짐 <span style="color:#b45309">미검토</span>'
    : ' · 최대 보 처짐비 ' + num(deflR, 2);
  var verdictTxt = sev ? (SEV_TEXT[sev] + ' — ' + SEV_KO[sev]) : '판정 데이터 없음';
  h += '<div class="verdict' + (ng?' ng':'') + '"><div class="vtag">판정 : ' + esc(verdictTxt) + '</div>';
  h += '<div class="vsub">최대 층간변위비 비율 ' + num(driftR, 2) + (govDrift ? ' <span style="color:#6b7280">(' + esc(govDrift) + ')</span>' : '')
    + ' · 최대 부재 내력비 ' + num(interR, 2) + (govMember ? ' <span style="color:#6b7280">(' + esc(govMember) + ')</span>' : '')
    + ' · 부적합 부재 ' + esc(ngMembers) + '개' + deflTxt + '</div></div>';

  // Tier1-1/2/5: 기둥 K · 보 LTB 지배 · RSA 부재검토 미수행 명시
  var smM = (dc && dc.summary) ? dc.summary : {};
  var memNotes = [];
  if (smM.column_K != null && smM.column_K > 1.0)
    memNotes.push('기둥 유효좌굴길이계수 K=' + num(smM.column_K, 2) + ' (비가새 모멘트골조)');
  if (smM.n_ltb_governed)
    memNotes.push('보 횡-비틀림좌굴(LTB) 지배 ' + esc(smM.n_ltb_governed) + '개');
  if (smM.n_slenderness_ng)
    memNotes.push('<span style="color:#c62828">세장비 한계 초과 ' + esc(smM.n_slenderness_ng) + '개</span> (최대 ' + num(smM.max_slenderness, 0) + ')');
  if (smM.n_noncompact)
    memNotes.push('비콤팩트/세장 단면 ' + esc(smM.n_noncompact) + '개 (F3.2 감소)');
  if (smM.n_compression_slender)
    memNotes.push('<span style="color:#b45309">압축 세장요소 ' + esc(smM.n_compression_slender) + '개</span> (E7 미반영)');
  var rsaU = smM.rsa_unchecked_combos || [];
  if (rsaU.length)
    memNotes.push('<span style="color:#b45309">RSA 조합 ' + esc(rsaU.length) + '개 부재검토 미수행</span> (변위만 산출)');
  if (memNotes.length)
    h += '<div class="vsub" style="margin-top:3px">부재검토 — ' + memNotes.join(' · ') + '</div>';

  // B2: 내진설계범주(SDC) + 횡력저항시스템 적격성
  var syc = dc && dc.system_check;
  if (syc){
    var sysNg = syc.status === 'NG';
    var sysTail = sysNg
      ? '<span style="color:#c62828">부적격</span>' + (syc.issues && syc.issues.length ? ' — ' + esc(syc.issues[0]) : '')
      : '적합 (높이제한 ' + esc((syc.height_limit || {}).limit_desc || '-') + ')';
    h += '<div class="vsub" style="margin-top:3px">내진설계범주 <strong>SDC ' + esc(syc.sdc) + '</strong>'
      + ' · 횡력저항시스템 ' + esc(syc.seismic_system || '-') + ' — ' + sysTail + '</div>';
  }

  // C1/C2/C3: 전역안정(전도)·P-Delta θ·비틀림 비정형
  var sm2 = (dc && dc.summary) ? dc.summary : {};
  var stabParts = [];
  if (sm2.overturning_FS != null)
    stabParts.push('전도 안전율 ' + num(sm2.overturning_FS, 2)
      + (sm2.stability_status === 'NG' ? ' <span style="color:#c62828">NG</span>' : ''));
  if (sm2.max_theta != null)
    stabParts.push('P-Delta θ ' + num(sm2.max_theta, 3)
      + (sm2.pdelta_status === 'NG' ? ' <span style="color:#c62828">불안정</span>' : ''));
  if (sm2.torsion_ratio != null){
    var torKO = {regular:'정형', torsional:'비틀림 비정형', extreme:'극단 비틀림'}[sm2.torsion_classification] || '';
    stabParts.push('δmax/δavg ' + num(sm2.torsion_ratio, 2) + (torKO ? ' (' + torKO + ')' : ''));
  }
  if (stabParts.length)
    h += '<div class="vsub" style="margin-top:3px">전역안정 — ' + stabParts.join(' · ') + '</div>';

  // #10/#12/#8/#13: 풍하중 사용성·수직 비정형·우발 비틀림·미검토 항목
  var extra = [];
  if (sm2.max_wind_drift_ratio != null)
    extra.push('풍하중 사용성 층간변위비 ' + num(sm2.max_wind_drift_ratio, 2)
      + (sm2.wind_drift_status === 'NG' ? ' <span style="color:#c62828">NG</span> (1/400 초과)' : ' (≤1/400)'));
  if (sm2.vertical_irregular){
    var vtKO = {stiffness_soft_story:'강성(연층)', mass:'중량', geometric:'기하'};
    var vts = (sm2.vertical_irregularity || []).map(function(t){ return vtKO[t] || t; });
    extra.push('<span style="color:#b45309">수직 비정형 ' + esc(vts.join(', ')) + '</span> (KDS 표 5.3-2)');
  }
  if (sm2.accidental_torsion_applied === false)
    extra.push('우발 비틀림 ±5% 편심 <span style="color:#b45309">모델 미적용</span> (산정값 §하중)');
  if (extra.length)
    h += '<div class="vsub" style="margin-top:3px">추가검토 — ' + extra.join(' · ') + '</div>';

  // #13: 지진입력 손상 → '안전' 단정 차단 + 미검토 항목 명시 (과대주장 방지)
  if (sm2.seismic_input_error)
    h += '<div class="vsub" style="margin-top:3px;color:#c62828"><strong>주의.</strong> '
      + '지진하중 입력 손상/누락 — 지진검토 미수행, 결과를 \'안전\'으로 단정 불가.</div>';
  var nci = sm2.not_checked_items || [];
  if (nci.length)
    h += '<div class="vsub" style="margin-top:3px;color:#6b7280">미검토(미포함) 항목 : '
      + esc(nci.join(', ')) + '</div>';

  // 종합 검토의견 callout (narration)
  if (interp && interp.summary_ko){
    h += '<div class="callout' + (ng?' ng':'') + '">';
    var badge = sev ? (SEV_KO[sev] + ' (' + SEV_TEXT[sev] + ')') : '검토의견';
    h += '<span class="c-badge">종합 검토의견 — ' + esc(badge) + '</span>';
    h += '<p>' + esc(interp.summary_ko) + '</p></div>';
  } else {
    h += '<p class="empty-note">서술형 종합의견 미생성 (interpretation 미제공).</p>';
  }

  // AI 생성 고지 (C1) — 책임소재 명확화. 도장란 직상단이므로 출처를 명시한다.
  // 필드단위 폴백(C4)으로 일부 항목만 AI 적용될 수 있으므로 applied_fields로 판정한다
  // (요약은 규칙기반 폴백이지만 §10.1 진단 서사는 AI인 부분적용 경우까지 정확히 표기).
  var nm = (interp && interp.narration_meta) || {};
  var aiApplied = nm.applied_fields && nm.applied_fields.length;
  var aiNote = null;
  if (nm.llm_used && !nm.fallback && aiApplied){
    aiNote = '본 종합검토의견 문안은 AI(' + esc(nm.model || 'LLM') + ')가 검증된 해석 결과를 바탕으로 작성한 <strong>초안</strong>이며, 모든 수치·판정은 결정론적 구조해석/설계검토에서 산출됨. 책임기술자의 검토·확정이 필요함.';
  } else if (nm.llm_used && aiApplied){
    aiNote = '본 보고서의 일부 서술(예: 진단 서사)은 AI(' + esc(nm.model || 'LLM') + ')가 작성한 <strong>초안</strong>이며, 나머지 항목은 규칙기반 자동요약으로 대체됨. 모든 수치·판정은 결정론적 구조해석/설계검토에서 산출됨. 책임기술자의 검토·확정이 필요함.';
  } else if (interp && interp.summary_ko){
    aiNote = '본 종합검토의견은 규칙기반 자동 요약(AI 문장화 미적용)이며, 책임기술자의 검토·확정이 필요함.';
  }
  if (aiNote){
    h += '<p class="note" style="font-size:8.4pt;color:#6b7280;border-top:.4pt dotted var(--rule-thin);padding-top:5px;margin-top:8px;"><strong>고지.</strong> ' + aiNote + '</p>';
  }

  // diagnosis (서술형 diagnosis_narrative + 구조화 진단 박스)
  var diagNarr = (interp && interp.diagnosis_narrative_ko) || null;
  if (interp && (interp.diagnosis || diagNarr)){
    h += '<h3 class="sub">10.1 파괴 진단</h3>';
    // B2: AI 진단 서사 (NG일 때만 생성됨). 아래 구조화 박스는 동일 근거의 데이터 표기.
    if (diagNarr){
      h += '<p class="diag-narr" style="margin:0 0 8px;line-height:1.6">' + esc(diagNarr) + '</p>';
    }
    if (interp.diagnosis){
      var dg = interp.diagnosis;
      h += '<div class="diag-box"><div class="dh">주 원인: ' + esc(dg.primary_cause_ko || '-') + '</div>';
      var cf = dg.contributing_factors_ko || [];
      if (cf.length){
        h += '<div style="margin-top:6px"><strong>기여 요인</strong><ul style="margin:4px 0 0 16px">';
        for (var i = 0; i < cf.length; i++) h += '<li>' + esc(cf[i]) + '</li>';
        h += '</ul></div>';
      }
      h += '</div>';
    }
  }

  // suggestions
  if (interp && interp.suggestions && interp.suggestions.length){
    h += '<h3 class="sub">10.2 개선 제안</h3>';
    for (var s = 0; s < interp.suggestions.length; s++){
      var sg = interp.suggestions[s];
      h += '<div class="sugg"><span class="imp">영향도: ' + esc(sg.expected_impact || '-') + '</span>'
        + '<strong>' + esc(sg.message_ko || '') + '</strong>';
      if (sg.target && (sg.current || sg.recommended)){
        var tgt = {column:'기둥', beam_x:'보-X', beam_y:'보-Y', structural_system:'구조시스템'}[sg.target] || sg.target;
        h += '<div style="margin-top:4px;font-size:9pt;color:#555">대상: <b>' + esc(tgt) + '</b>';
        if (sg.current && sg.recommended) h += ' — ' + esc(sg.current) + ' → ' + esc(sg.recommended);
        else if (sg.recommended) h += ' → ' + esc(sg.recommended);
        h += '</div>';
      }
      h += '</div>';
    }
  }

  // 해석 가정 / 경고 목록 (SUMMARY.warnings)
  var w = SUMMARY.warnings || [];
  if (w.length){
    h += '<h3 class="sub">10.3 해석 가정 및 경고</h3><ul class="warns">';
    for (var j = 0; j < w.length; j++){
      var txt = w[j].text_ko || w[j].text || '';
      h += '<li><span class="wcode">' + esc(w[j].code) + '</span><span>' + esc(txt) + '</span></li>';
    }
    h += '</ul>';
  }

  // 검토자 서명 (cover stamp 재사용)
  var st = (DATA.cover || {}).stamp || {};
  h += '<h3 class="sub">10.4 검토자 서명</h3>';
  h += '<table class="tbl"><thead><tr><th class="ctr" style="width:22%">구분</th><th class="ctr">성명</th><th class="ctr">자격번호</th><th class="ctr" style="width:20%">서명 / (인)</th></tr></thead><tbody>';
  function sigRow(label, p){ p = p || {}; return '<tr><td class="ctr">' + label + '</td><td class="ctr">' + esc(p.name) + '</td><td class="ctr">' + esc(p.license_no) + '</td><td class="ctr">　</td></tr>'; }
  h += sigRow('작성자', st.author) + sigRow('검토자', st.reviewer) + sigRow('승인자', st.approver);
  h += '</tbody></table>';

  setHTML('sec10', h);
}

// ============================================================
// §6 3D Plotly viewer (buildPlot / updateDeformed / updateSummary verbatim)
// ============================================================
function buildPlot(caseName, scale){
  var types = ['column', 'beam_x', 'beam_y'];
  var colors = ['#9E9E9E', '#2196F3', '#4CAF50'];
  var defColors = ['#f44336', '#ff5722', '#e91e63'];
  var names = ['Column', 'Beam X', 'Beam Y'];
  var traces = [];

  // Undeformed (traces 0-2)
  for (var t = 0; t < 3; t++){
    var ms = DATA.members[types[t]];
    var x = [], y = [], z = [];
    for (var i = 0; i < ms.length; i++){
      var ni = ms[i][0], nj = ms[i][1];
      var pi = DATA.node_map[String(ni)], pj = DATA.node_map[String(nj)];
      x.push(pi[0], pj[0], null); y.push(pi[1], pj[1], null); z.push(pi[2], pj[2], null);
    }
    traces.push({type:'scatter3d', mode:'lines', x:x, y:y, z:z,
      line:{color:colors[t], width:4}, name:names[t], legendgroup:'undeformed'});
  }
  // Deformed (traces 3-5)
  var cd = DATA.case_data[caseName], disp = cd.disp;
  for (var t = 0; t < 3; t++){
    var ms = DATA.members[types[t]];
    var x = [], y = [], z = [];
    for (var i = 0; i < ms.length; i++){
      var ni = ms[i][0], nj = ms[i][1];
      var pi = DATA.node_map[String(ni)], pj = DATA.node_map[String(nj)];
      var di = disp[String(ni)] || [0,0,0], dj = disp[String(nj)] || [0,0,0];
      x.push(pi[0]+di[0]/1000*scale, pj[0]+dj[0]/1000*scale, null);
      y.push(pi[1]+di[1]/1000*scale, pj[1]+dj[1]/1000*scale, null);
      z.push(pi[2]+di[2]/1000*scale, pj[2]+dj[2]/1000*scale, null);
    }
    traces.push({type:'scatter3d', mode:'lines', x:x, y:y, z:z,
      line:{color:defColors[t], width:2}, name:names[t]+' (deformed)', legendgroup:'deformed'});
  }
  // Nodes (trace 6)
  var nx=[], ny=[], nz=[], texts=[];
  for (var nid in DATA.node_map){
    var p = DATA.node_map[nid], d = disp[nid] || [0,0,0];
    nx.push(p[0]+d[0]/1000*scale); ny.push(p[1]+d[1]/1000*scale); nz.push(p[2]+d[2]/1000*scale);
    texts.push('Node '+nid+'<br>dx: '+d[0].toFixed(2)+' mm<br>dy: '+d[1].toFixed(2)+' mm<br>dz: '+d[2].toFixed(2)+' mm');
  }
  traces.push({type:'scatter3d', mode:'markers', x:nx, y:ny, z:nz,
    marker:{size:2, color:'#333'}, text:texts, hoverinfo:'text', name:'Nodes', showlegend:false});
  // Supports (trace 7)
  var sx=[], sy=[], sz=[];
  for (var i = 0; i < DATA.support_ids.length; i++){
    var p = DATA.node_map[String(DATA.support_ids[i])];
    sx.push(p[0]); sy.push(p[1]); sz.push(p[2]);
  }
  traces.push({type:'scatter3d', mode:'markers', x:sx, y:sy, z:sz,
    marker:{size:5, color:'#FF9800', symbol:'diamond'}, name:'Supports'});

  var layout = {
    scene: { xaxis:{title:'X (m)'}, yaxis:{title:'Y (m)'}, zaxis:{title:'Z (m)'},
      aspectmode:'data', camera:{eye:{x:1.5, y:1.5, z:1.0}} },
    height:560, margin:{l:0,r:0,t:30,b:0}, legend:{x:0,y:1}
  };
  Plotly.newPlot('plot3d', traces, layout, {responsive:true});
}

function updateDeformed(caseName, scale){
  var cd = DATA.case_data[caseName], disp = cd.disp;
  var types = ['column', 'beam_x', 'beam_y'];
  for (var t = 0; t < 3; t++){
    var ms = DATA.members[types[t]];
    var x=[], y=[], z=[];
    for (var i = 0; i < ms.length; i++){
      var ni = ms[i][0], nj = ms[i][1];
      var pi = DATA.node_map[String(ni)], pj = DATA.node_map[String(nj)];
      var di = disp[String(ni)] || [0,0,0], dj = disp[String(nj)] || [0,0,0];
      x.push(pi[0]+di[0]/1000*scale, pj[0]+dj[0]/1000*scale, null);
      y.push(pi[1]+di[1]/1000*scale, pj[1]+dj[1]/1000*scale, null);
      z.push(pi[2]+di[2]/1000*scale, pj[2]+dj[2]/1000*scale, null);
    }
    Plotly.restyle('plot3d', {x:[x], y:[y], z:[z]}, [3+t]);
  }
  var nx=[], ny=[], nz=[], texts=[];
  for (var nid in DATA.node_map){
    var p = DATA.node_map[nid], d = disp[nid] || [0,0,0];
    nx.push(p[0]+d[0]/1000*scale); ny.push(p[1]+d[1]/1000*scale); nz.push(p[2]+d[2]/1000*scale);
    texts.push('Node '+nid+'<br>dx: '+d[0].toFixed(2)+' mm<br>dy: '+d[1].toFixed(2)+' mm<br>dz: '+d[2].toFixed(2)+' mm');
  }
  Plotly.restyle('plot3d', {x:[nx], y:[ny], z:[nz], text:[texts]}, [6]);
}

function updateSummary(caseName){
  var mv = DATA.case_data[caseName].max;
  var fmt = function(v){ return typeof v==='number' ? Math.abs(v).toFixed(2) : v; };
  var driftFmt = function(v){ return v > 0 ? '1/'+Math.round(1/v) : '-'; };
  var h = '';
  h += '<div class="viz-card"><div class="vc-lbl">최대변위 X</div><div class="vc-val">'+fmt(mv.disp_x)+' mm</div><div class="vc-det">Node '+mv.disp_x_node+'</div></div>';
  h += '<div class="viz-card"><div class="vc-lbl">최대변위 Y</div><div class="vc-val">'+fmt(mv.disp_y)+' mm</div><div class="vc-det">Node '+mv.disp_y_node+'</div></div>';
  h += '<div class="viz-card"><div class="vc-lbl">최대변위 Z</div><div class="vc-val">'+fmt(mv.disp_z)+' mm</div><div class="vc-det">Node '+mv.disp_z_node+'</div></div>';
  h += '<div class="viz-card"><div class="vc-lbl">최대층간변위 X</div><div class="vc-val">'+driftFmt(mv.drift_x)+'</div><div class="vc-det">Story '+mv.drift_x_story+'</div></div>';
  h += '<div class="viz-card"><div class="vc-lbl">최대층간변위 Y</div><div class="vc-val">'+driftFmt(mv.drift_y)+'</div><div class="vc-det">Story '+mv.drift_y_story+'</div></div>';
  h += '<div class="viz-card"><div class="vc-lbl">최대모멘트</div><div class="vc-val">'+fmt(mv.moment)+' kN·m</div><div class="vc-det">Elem '+mv.moment_elem+'</div></div>';
  h += '<div class="viz-card"><div class="vc-lbl">최대축력</div><div class="vc-val">'+fmt(mv.axial)+' kN</div><div class="vc-det">Elem '+mv.axial_elem+'</div></div>';
  setHTML('vizCards', h);
}

function onCaseChange(){
  currentCase = el('caseSelect').value;
  updateDeformed(currentCase, currentScale);
  updateSummary(currentCase);
  renderSec9();   // 층별 drift 갱신
}
function onScaleInput(){
  currentScale = parseInt(el('scaleSlider').value);
  el('scaleValue').textContent = currentScale + '×';
  updateDeformed(currentCase, currentScale);
}

// ============================================================
// scroll-spy (IntersectionObserver, verbatim from mockup)
// ============================================================
function initScrollSpy(){
  var links = Array.prototype.slice.call(document.querySelectorAll('nav.toc a'));
  var map = {};
  links.forEach(function(a){
    var id = a.getAttribute('href').slice(1);
    var sec = document.getElementById(id);
    if (sec){ map[id] = a; }
  });
  var sections = Object.keys(map).map(function(id){ return document.getElementById(id); });
  function setActive(id){ links.forEach(function(a){ a.classList.remove('active'); }); if (map[id]){ map[id].classList.add('active'); } }
  var observer = new IntersectionObserver(function(entries){
    var visible = entries.filter(function(e){ return e.isIntersecting; });
    if (visible.length){
      visible.sort(function(a,b){ return a.boundingClientRect.top - b.boundingClientRect.top; });
      setActive(visible[0].target.id);
    }
  }, { rootMargin: '-80px 0px -65% 0px', threshold: 0 });
  sections.forEach(function(s){ observer.observe(s); });
  if (sections.length){ setActive(sections[0].id); }
}

// ============================================================
// 부록 A — 전체 부재 일람표 / 지점 반력 / 조합 포락
// ============================================================
function _typeKO(t){ return {column:'기둥', beam_x:'보-X', beam_y:'보-Y'}[t] || t; }

function renderAppendix(){
  renderAppMembers();
  renderAppReactionsInit();
  renderAppComboSummary();
}

// A.1 전체 부재 검토 일람표 (member_check.members 전부 — §8은 위험부재만)
function renderAppMembers(){
  var mc = (DATA.design_check || {}).member_check;
  if (!mc || !mc.members || !mc.members.length){
    setHTML('appMembers', '<p class="empty-note">부재 검토 데이터 없음 (설계검토 미수행).</p>'); return;
  }
  var ms = mc.members;
  var h = '<p class="note">전체 <b>' + ms.length + '</b>개 부재의 포락 부재력 및 조합내력비 (KDS 14 31 00). 내력비 내림차순. §8은 상위 위험부재만 표시한다.</p>';
  h += '<table class="tbl"><thead><tr>'
    + '<th class="ctr">부재#</th><th class="ctr">유형</th><th class="ctr">단면</th><th class="ctr">층</th>'
    + '<th class="num">P<sub>u</sub> <span class="unit">(kN)</span></th>'
    + '<th class="num">V<sub>u</sub> <span class="unit">(kN)</span></th>'
    + '<th class="num">M<sub>ux</sub> <span class="unit">(kN·m)</span></th>'
    + '<th class="num">M<sub>uy</sub> <span class="unit">(kN·m)</span></th>'
    + '<th class="num">내력비</th><th class="num">전단비</th><th class="ctr">지배조합</th><th class="ctr">판정</th></tr></thead><tbody>';
  for (var i = 0; i < ms.length; i++){
    var m = ms[i], d = m.demand || {}, r = m.ratios || {};
    var ok = (m.status === 'OK');
    h += '<tr' + (ok ? '' : ' class="ng-row"') + '>'
      + '<td class="ctr">#' + esc(m.member_id) + '</td>'
      + '<td class="ctr">' + esc(_typeKO(m.type)) + '</td>'
      + '<td class="ctr">' + esc(m.section) + '</td>'
      + '<td class="ctr">' + esc(m.story) + '</td>'
      + '<td class="num">' + num(d.Pu, 1) + '</td>'
      + '<td class="num">' + num(d.Vu, 1) + '</td>'
      + '<td class="num">' + num(d.Mux, 1) + '</td>'
      + '<td class="num">' + num(d.Muy, 1) + '</td>'
      + '<td class="num" style="color:' + utilColor(r.interaction) + ';font-weight:700">' + num(r.interaction, 3) + '</td>'
      + '<td class="num" style="color:' + utilColor(r.shear) + '">' + num(r.shear, 3) + '</td>'
      + '<td class="ctr" style="font-size:8.2pt">' + esc(m.governing_combo) + '</td>'
      + '<td class="ctr">' + pill(ok) + '</td></tr>';
  }
  h += '</tbody></table>';
  setHTML('appMembers', h);
}

// A.2 지점 반력 — 조합 선택 드롭다운 + 표
function renderAppReactionsInit(){
  var sel = el('appRxnCase');
  var names = (DATA.combo_names && DATA.combo_names.length) ? DATA.combo_names : (DATA.case_names || []);
  if (!sel || !names.length){ setHTML('appReactions', '<p class="empty-note">반력 데이터 없음.</p>'); return; }
  var prefer = ['1.2DL+1.6LL', '1.4DL'];
  var def = names[0];
  for (var p = 0; p < prefer.length; p++){ if (names.indexOf(prefer[p]) >= 0){ def = prefer[p]; break; } }
  var opts = '';
  for (var i = 0; i < names.length; i++){
    opts += '<option value="' + esc(names[i]) + '"' + (names[i] === def ? ' selected' : '') + '>' + esc(names[i]) + '</option>';
  }
  sel.innerHTML = opts;
  renderAppReactions();
}

function renderAppReactions(){
  var sel = el('appRxnCase'); if (!sel) return;
  var name = sel.value;
  var rx = ((DATA.case_data || {})[name] || {}).reactions || [];
  if (!rx.length){ setHTML('appReactions', '<p class="empty-note">선택 조합의 반력 데이터 없음.</p>'); return; }
  var h = '<table class="tbl"><thead><tr>'
    + '<th class="ctr">절점</th><th class="num">X <span class="unit">(m)</span></th><th class="num">Y <span class="unit">(m)</span></th>'
    + '<th class="num">R<sub>X</sub> <span class="unit">(kN)</span></th><th class="num">R<sub>Y</sub> <span class="unit">(kN)</span></th><th class="num">R<sub>Z</sub> <span class="unit">(kN)</span></th>'
    + '<th class="num">M<sub>X</sub> <span class="unit">(kN·m)</span></th><th class="num">M<sub>Y</sub> <span class="unit">(kN·m)</span></th><th class="num">M<sub>Z</sub> <span class="unit">(kN·m)</span></th></tr></thead><tbody>';
  var sx = 0, sy = 0, sz = 0;
  for (var i = 0; i < rx.length; i++){
    var r = rx[i];
    sx += (r.RX_kN || 0); sy += (r.RY_kN || 0); sz += (r.RZ_kN || 0);
    h += '<tr><td class="ctr">' + esc(r.node) + '</td>'
      + '<td class="num">' + num(r.x_m, 1) + '</td><td class="num">' + num(r.y_m, 1) + '</td>'
      + '<td class="num">' + num(r.RX_kN, 2) + '</td><td class="num">' + num(r.RY_kN, 2) + '</td><td class="num">' + num(r.RZ_kN, 2) + '</td>'
      + '<td class="num">' + num(r.MX_kNm, 2) + '</td><td class="num">' + num(r.MY_kNm, 2) + '</td><td class="num">' + num(r.MZ_kNm, 2) + '</td></tr>';
  }
  h += '</tbody><tfoot><tr><td colspan="3" class="ctr">합계 Σ</td>'
    + '<td class="num">' + num(sx, 2) + '</td><td class="num">' + num(sy, 2) + '</td><td class="num">' + num(sz, 2) + '</td>'
    + '<td colspan="3" class="ctr">밑면전단 √(ΣR<sub>x</sub>²+ΣR<sub>y</sub>²) = ' + num(Math.sqrt(sx*sx + sy*sy), 2) + ' kN</td></tr></tfoot></table>';
  h += '<p class="note"><strong>주.</strong> 양(+) 반력은 전역좌표 +방향. ΣR<sub>z</sub>는 해당 조합의 총 연직반력(중력 검증용).</p>';
  setHTML('appReactions', h);
}

// A.3 하중조합별 포락 요약
function renderAppComboSummary(){
  var cd = DATA.case_data || {};
  var names = (DATA.combo_names && DATA.combo_names.length) ? DATA.combo_names : (DATA.case_names || []);
  if (!names.length){ setHTML('appCombos', '<p class="empty-note">하중조합 데이터 없음.</p>'); return; }
  var h = '<p class="note">하중조합별 최대 응답 포락. 변위는 절대 최대값, 층간변위비는 1/x 표기, ΣR<sub>z</sub>는 총 연직반력.</p>';
  h += '<table class="tbl"><thead><tr><th>하중조합</th>'
    + '<th class="num">δ<sub>x</sub> <span class="unit">(mm)</span></th><th class="num">δ<sub>y</sub> <span class="unit">(mm)</span></th><th class="num">δ<sub>z</sub> <span class="unit">(mm)</span></th>'
    + '<th class="num">변위비 X</th><th class="num">변위비 Y</th>'
    + '<th class="num">M<sub>max</sub> <span class="unit">(kN·m)</span></th><th class="num">N<sub>max</sub> <span class="unit">(kN)</span></th>'
    + '<th class="num">ΣR<sub>z</sub> <span class="unit">(kN)</span></th></tr></thead><tbody>';
  function dr(v){ return (v > 0) ? '1/' + Math.round(1/v) : '-'; }
  for (var i = 0; i < names.length; i++){
    var nm = names[i], c = cd[nm] || {}, mx = c.max || {}, rx = c.reactions || [];
    var sz = 0; for (var j = 0; j < rx.length; j++) sz += (rx[j].RZ_kN || 0);
    h += '<tr><td>' + esc(nm) + '</td>'
      + '<td class="num">' + num(Math.abs(mx.disp_x || 0), 2) + '</td>'
      + '<td class="num">' + num(Math.abs(mx.disp_y || 0), 2) + '</td>'
      + '<td class="num">' + num(Math.abs(mx.disp_z || 0), 2) + '</td>'
      + '<td class="num">' + dr(mx.drift_x || 0) + '</td>'
      + '<td class="num">' + dr(mx.drift_y || 0) + '</td>'
      + '<td class="num">' + num(Math.abs(mx.moment || 0), 1) + '</td>'
      + '<td class="num">' + num(Math.abs(mx.axial || 0), 1) + '</td>'
      + '<td class="num">' + num(sz, 1) + '</td></tr>';
  }
  h += '</tbody></table>';
  setHTML('appCombos', h);
}

// ============================================================
// Boot
// ============================================================
function boot(){
  renderCover();
  renderTOC();
  renderRunningHeads();
  renderSidebar();
  renderStrip();
  renderSec1();
  renderSec2();
  renderSec3();
  renderSec4();
  renderSec5();
  renderSec6();
  renderSec7();
  renderSec8();
  renderSec9();
  renderSec10();
  renderAppendix();

  // §6 3D viewer — #plot3d 존재 후 실행
  if (currentCase && DATA.case_data && DATA.case_data[currentCase]){
    var sel = el('caseSelect'); if (sel) sel.value = currentCase;
    el('scaleSlider').value = DEFAULT_SCALE;
    el('scaleValue').textContent = DEFAULT_SCALE + '×';
    try {
      buildPlot(currentCase, DEFAULT_SCALE);
      updateSummary(currentCase);
    } catch (e) {
      setHTML('plot3d', '<p class="empty-note">3D 뷰 생성 실패: ' + esc(e.message) + '</p>');
    }
  } else {
    setHTML('plot3d', '<p class="empty-note">3D 형상 데이터 없음.</p>');
  }

  initScrollSpy();
}

if (document.readyState === 'loading'){
  window.addEventListener('load', boot);
} else {
  boot();
}
</script>
</body>
</html>"""
