"""
3D 프레임 해석 결과 시각화 모듈
- prepare_3d_viz_data(): 공용 데이터 가공 (Streamlit/HTML 재사용)
- plot_frame_3d_interactive(): 독립 HTML 리포트 생성 (Plotly CDN)
"""
from __future__ import annotations

import json
import math
import os
import tempfile


# ============================================================
# Public API
# ============================================================

def prepare_3d_viz_data(multi_result) -> dict:
    """3D 해석 결과 → 시각화 데이터 (JSON 직렬화 가능).

    Args:
        multi_result: Frame3DMultiCaseResult 객체

    Returns:
        dict with node_map, members, support_ids, case_data,
        all_names, case_names, combo_names, model_info, summary
    """
    r = multi_result
    node_map = {str(n["id"]): [n["x_m"], n["y_m"], n["z_m"]] for n in r.nodes}

    members = {"column": [], "beam_x": [], "beam_y": []}
    for m in r.member_info:
        if m["type"] in members:
            members[m["type"]].append([m["ni"], m["nj"]])

    support_ids = [n["id"] for n in r.nodes if abs(n["z_m"]) < 0.001]

    case_data = {}
    case_names = list(r.case_results.keys())
    combo_names = list(r.combo_results.keys())

    for name in case_names:
        case_data[name] = _extract_viz_case(r.case_results[name])
    for name in combo_names:
        case_data[name] = _extract_viz_case(r.combo_results[name])

    return {
        "node_map": node_map,
        "members": members,
        "support_ids": support_ids,
        "case_data": case_data,
        "all_names": case_names + combo_names,
        "case_names": case_names,
        "combo_names": combo_names,
        "model_info": {
            "num_stories": r.num_stories,
            "num_bays_x": r.num_bays_x,
            "num_bays_y": r.num_bays_y,
            "stories": r.stories,
            "bays_x": r.bays_x,
            "bays_y": r.bays_y,
            "total_height": r.total_height,
            "total_width_x": r.total_width_x,
            "total_width_y": r.total_width_y,
            "supports": r.supports,
            "column_section": r.column_section,
            "beam_x_section": r.beam_x_section,
            "beam_y_section": r.beam_y_section,
            "material_name": r.material_name,
            "E_MPa": r.E_MPa,
            "G_MPa": r.G_MPa,
            "num_elements": r.num_elements,
            "num_nodes": len(r.nodes),
            "num_members": len(r.member_info),
        },
        "summary": _build_report_summary(r),
        "modal_analysis": getattr(r, 'modal_analysis', None) or None,
    }


def _extract_viz_case(cr) -> dict:
    """Frame3DCaseResult → 시각화용 dict."""
    disp = {}
    for d in cr.nodal_displacements:
        disp[str(d["node"])] = [d["dx_mm"], d["dy_mm"], d["dz_mm"]]

    return {
        "disp": disp,
        "reactions": cr.reactions,
        "drifts": cr.story_drifts,
        "max": {
            "disp_x": cr.max_displacement_x,
            "disp_x_node": cr.max_displacement_x_node,
            "disp_y": cr.max_displacement_y,
            "disp_y_node": cr.max_displacement_y_node,
            "disp_z": cr.max_displacement_z,
            "disp_z_node": cr.max_displacement_z_node,
            "drift_x": cr.max_drift_x,
            "drift_x_story": cr.max_drift_x_story,
            "drift_y": cr.max_drift_y,
            "drift_y_story": cr.max_drift_y_story,
            "moment": cr.max_moment,
            "moment_elem": cr.max_moment_element,
            "axial": cr.max_axial,
            "axial_elem": cr.max_axial_element,
            "shear": cr.max_shear,
            "shear_elem": cr.max_shear_element,
            "torsion": cr.max_torsion,
            "torsion_elem": cr.max_torsion_element,
        },
    }


# ============================================================
# Stage 1: Report Summary Helpers
# ============================================================

def _build_report_summary(r) -> dict:
    """해석 결과 요약 데이터 생성 (Stage 1 리포트 품질 강화)."""
    summary = {
        "warnings": _get_model_warnings(r),
        "envelope": _compute_envelope(r),
        "story_summary": _compute_story_summary(r),
        "critical_members": _compute_critical_members(r, top_n=10),
        "equilibrium": _compute_equilibrium(r),
        "model_settings": _get_model_settings(r),
        "global_response": validate_3d_global_response(r),
    }
    # 비선형 해석 검증 요약 (기하비선형 시만)
    meta = getattr(r, 'analysis_metadata', {})
    if meta.get('requested_mode') == 'pdelta':
        summary["nonlinear_summary"] = _build_nonlinear_summary(meta, r)
    return summary


def _get_model_warnings(r) -> list[dict]:
    """모델 한계점/경고 목록 생성."""
    warnings = []
    geo_nl = getattr(r, 'geometric_nonlinearity', 'linear')
    meta = getattr(r, 'analysis_metadata', {})
    actual_transf = meta.get('actual_transf', 'Linear')
    if geo_nl == "pdelta":
        warnings.append({
            "code": "W01",
            "text": f"Geometric nonlinearity enabled ({actual_transf} transformation)",
            "text_ko": f"기하비선형 해석 활성화 ({actual_transf} 변환)",
            "severity": "info",
        })
        # Task B: 3D Corotational 힘 해석 경고
        if actual_transf == "Corotational":
            warnings.append({
                "code": "W03",
                "text": ("3D Corotational: displacement/drift results are valid. "
                         "Local element forces are in the corotated frame "
                         "(minor effect at small deformations)."),
                "text_ko": ("3D Corotational: 변위/층간변위 결과 유효. "
                            "로컬 요소력은 회전좌표계 기준 "
                            "(소변위에서 영향 미미)."),
                "severity": "info",
            })
        if meta.get('fallback_used'):
            alg = meta.get('solver_algorithm', 'ModifiedNewton')
            nst = meta.get('n_steps', 50)
            warnings.append({
                "code": "W04",
                "text": f"Solver fallback triggered: {alg} ({nst} steps)",
                "text_ko": f"솔버 Fallback 발생: {alg} ({nst} steps)",
                "severity": "warning",
            })
    else:
        warnings.append({"code": "W01",
                         "text": "Geometric nonlinearity: disabled (linear geometry)",
                         "text_ko": "기하비선형: 비활성화 (선형 기하)",
                         "severity": "info"})
    # W02: 릴리즈 상태에 따라 조건부 표시
    releases = getattr(r, 'member_releases', None) or {}
    if releases:
        beam_parts = []
        col_parts = []
        for k, v in releases.items():
            if v:
                if k == "column":
                    col_parts.append(f"{k}: {v}")
                else:
                    beam_parts.append(f"{k}: {v}")
        if beam_parts:
            desc = ', '.join(beam_parts)
            warnings.append(
                {"code": "W02",
                 "text": f"Member end releases (beam): {desc}",
                 "text_ko": f"부재 릴리즈 (보): {desc}",
                 "severity": "info"})
        if col_parts:
            desc = ', '.join(col_parts)
            warnings.append(
                {"code": "W02a",
                 "text": f"Column end release is experimental and may destabilize frame behavior ({desc})",
                 "text_ko": f"기둥 릴리즈는 실험적이며 골조 안정성에 영향 가능 ({desc})",
                 "severity": "warning"})
    else:
        warnings.append(
            {"code": "W02",
             "text": "All beam-column connections are rigid (no member releases)",
             "text_ko": "모든 보-기둥 접합부 강접 (부재 릴리즈 없음)",
             "severity": "info"})
    warnings.append(
        {"code": "W05", "text": "Floor slabs not modeled (load distribution only)",
         "text_ko": "슬래브 강성 미모델링 (하중 분배만 반영)",
         "severity": "info"})

    # W06: J(비틀림상수) 근사값 사용 여부 — 실제 단면의 j_source 확인
    try:
        from core.section_3d import DEFAULT_SECTIONS_3D
        sec_names = set()
        for attr in ('column_section', 'beam_x_section', 'beam_y_section'):
            sn = getattr(r, attr, '')
            if sn:
                sec_names.add(sn)
        j_sources = {}
        for sn in sec_names:
            sec = DEFAULT_SECTIONS_3D.get(sn)
            if sec:
                j_sources[sn] = getattr(sec, 'j_source', 'unknown')
        approx_secs = [s for s, js in j_sources.items() if js in ("approximate", "fallback")]
        if approx_secs:
            slist = ', '.join(approx_secs)
            warnings.append(
                {"code": "W06",
                 "text": f"Torsional constant (J) uses approximate formula: {slist}",
                 "text_ko": f"비틀림상수(J) 근사식 사용: {slist}",
                 "severity": "warning"})
        else:
            warnings.append(
                {"code": "W06",
                 "text": "Torsional constant (J) from DB/benchmark for all sections",
                 "text_ko": "비틀림상수(J) DB/벤치마크 값 사용",
                 "severity": "info"})
    except Exception:
        warnings.append(
            {"code": "W06", "text": "Torsional constant (J) source not verified",
             "text_ko": "비틀림상수(J) 출처 미확인",
             "severity": "info"})

    warnings.append(
        {"code": "W07", "text": "elasticBeamColumn elements only (no inelastic behavior)",
         "text_ko": "elasticBeamColumn 요소만 사용 (비탄성 거동 없음)",
         "severity": "info"})
    if getattr(r, 'rigid_diaphragm', False):
        warnings.append(
            {"code": "W08", "text": "Rigid diaphragm applied at floor levels",
             "text_ko": "강체 다이어프램 적용 (층 레벨)",
             "severity": "info"})
    return warnings


def _build_nonlinear_summary(meta: dict, r=None) -> dict:
    """기하비선형 해석 검증 요약 데이터 생성 (Stage 6 강화).

    Args:
        meta: analysis_metadata dict
        r: Frame3DMultiCaseResult (선형 비교 증폭비 계산용, optional)
    """
    actual_transf = meta.get("actual_transf", "Unknown")
    is_corotational = (actual_transf == "Corotational")

    summary = {
        "requested_mode": meta.get("requested_mode", "pdelta"),
        "transformation": actual_transf,
        "solver_algorithm": meta.get("solver_algorithm", "Linear"),
        "fallback_used": meta.get("fallback_used", False),
        "n_load_steps": meta.get("n_steps", 1),
        # Task C: force interpretation policy
        "force_interpretation": {
            "displacement_drift": "reliable",
            "base_reactions": "reliable",
            "local_element_forces": "caution — corotated frame" if is_corotational else "reliable",
        },
        "interpretation_notes": [
            "Displacement and drift amplification: valid for interpretation",
            "Base reactions: reliable for global equilibrium verification",
        ] + ([
            "Local element forces are in the corotated frame — direct moment comparison with linear analysis requires caution",
            "For member design, prefer base reaction verification or use linear analysis forces with amplification factors",
        ] if is_corotational else []),
    }

    return summary


def _compute_envelope(r) -> list[dict]:
    """전 케이스/조합 Envelope (각 metric의 최대값 + governing case + 구조화 위치)."""
    # Build element → member mapping
    elem_to_member = {}
    for m in r.member_info:
        for eid in m.get("element_ids", []):
            elem_to_member[eid] = m

    # Build node z lookup + story resolver
    node_z = {n["id"]: n["z_m"] for n in r.nodes}
    cum_h = []
    ch = 0.0
    for h in r.stories:
        ch += h
        cum_h.append(round(ch, 4))

    def z_to_story(z):
        for i, c in enumerate(cum_h):
            if z <= c + 0.01:
                return i + 1
        return len(r.stories)

    all_results = {}
    all_results.update(r.case_results)
    all_results.update(r.combo_results)

    metrics = [
        ("Max Disp. X", "max_displacement_x", "mm", "max_displacement_x_node", "node"),
        ("Max Disp. Y", "max_displacement_y", "mm", "max_displacement_y_node", "node"),
        ("Max Disp. Z", "max_displacement_z", "mm", "max_displacement_z_node", "node"),
        ("Max Drift X", "max_drift_x", "", "max_drift_x_story", "story"),
        ("Max Drift Y", "max_drift_y", "", "max_drift_y_story", "story"),
        ("Max Moment", "max_moment", "kN\u00b7m", "max_moment_element", "element"),
        ("Max Axial", "max_axial", "kN", "max_axial_element", "element"),
        ("Max Shear", "max_shear", "kN", "max_shear_element", "element"),
        ("Max Torsion", "max_torsion", "kN\u00b7m", "max_torsion_element", "element"),
    ]

    envelope = []
    for label, attr, unit, loc_attr, loc_type in metrics:
        max_val = 0.0
        gov_case = "-"
        raw_loc_id = None
        for name, cr in all_results.items():
            val = abs(getattr(cr, attr, 0.0))
            if val > abs(max_val):
                max_val = val
                gov_case = name
                raw_loc_id = getattr(cr, loc_attr, None)

        # Build structured location
        location = _build_location(
            loc_type, raw_loc_id, elem_to_member, node_z, z_to_story)

        envelope.append({
            "metric": label,
            "value": round(max_val, 4),
            "unit": unit,
            "governing_case": gov_case,
            "location": location,
        })

    return envelope


def _build_location(loc_type, raw_id, elem_to_member, node_z, z_to_story):
    """위치 정보를 구조화된 dict로 변환 (Streamlit 재사용 가능)."""
    if raw_id is None:
        return {"type": loc_type, "label": "-"}

    if loc_type == "node":
        z = node_z.get(raw_id)
        story = z_to_story(z) if z is not None else None
        label = "Node %d" % raw_id
        if story is not None:
            label += " (%dF)" % story
        return {"type": "node", "id": raw_id, "story": story, "label": label}

    elif loc_type == "story":
        return {"type": "story", "id": raw_id, "label": "Story %s" % raw_id}

    elif loc_type == "element":
        member = elem_to_member.get(raw_id)
        if member:
            mid = member["member_id"]
            mtype = member["type"]
            ni, nj = member["ni"], member["nj"]
            z_ni = node_z.get(ni, 0)
            z_nj = node_z.get(nj, 0)
            story = z_to_story(max(z_ni, z_nj))
            label = "M%d (%s, %dF, N%d-%d)" % (mid, mtype, story, ni, nj)
            return {
                "type": "member", "member_id": mid, "member_type": mtype,
                "story": story, "node_i": ni, "node_j": nj, "label": label,
            }
        return {"type": "element", "id": raw_id, "label": "Elem %d" % raw_id}

    return {"type": loc_type, "label": str(raw_id)}


def _compute_story_summary(r) -> list[dict]:
    """층별 최대 변위/Drift 요약 + governing case."""
    # Node z lookup
    node_z = {n["id"]: n["z_m"] for n in r.nodes}

    # Cumulative heights
    cum_h = []
    ch = 0.0
    for h in r.stories:
        ch += h
        cum_h.append(round(ch, 4))

    # Map node -> story
    node_to_story = {}
    for nid, z in node_z.items():
        for i, c in enumerate(cum_h):
            if abs(z - c) < 0.01:
                node_to_story[nid] = i + 1
                break

    all_results = {}
    all_results.update(r.case_results)
    all_results.update(r.combo_results)

    stories = []
    for i, h in enumerate(r.stories):
        story_num = i + 1
        entry = {
            "story": story_num,
            "height_m": round(h, 2),
            "cum_height_m": round(cum_h[i], 2),
            "max_dx": 0.0, "gov_dx": "-",
            "max_dy": 0.0, "gov_dy": "-",
            "max_drift_x": 0.0, "gov_drift_x": "-",
            "max_drift_y": 0.0, "gov_drift_y": "-",
        }

        for name, cr in all_results.items():
            # Drift
            for sd in cr.story_drifts:
                if sd["story"] == story_num:
                    if sd["drift_x"] > entry["max_drift_x"]:
                        entry["max_drift_x"] = round(sd["drift_x"], 6)
                        entry["gov_drift_x"] = name
                    if sd["drift_y"] > entry["max_drift_y"]:
                        entry["max_drift_y"] = round(sd["drift_y"], 6)
                        entry["gov_drift_y"] = name

            # Max displacement at this story level
            for nd in cr.nodal_displacements:
                nid = nd["node"]
                if node_to_story.get(nid) == story_num:
                    dx = abs(nd["dx_mm"])
                    dy = abs(nd["dy_mm"])
                    if dx > entry["max_dx"]:
                        entry["max_dx"] = round(dx, 3)
                        entry["gov_dx"] = name
                    if dy > entry["max_dy"]:
                        entry["max_dy"] = round(dy, 3)
                        entry["gov_dy"] = name

        stories.append(entry)

    return stories


def _compute_critical_members(r, top_n: int = 10, sort_by: str = "moment") -> dict:
    """부재별 최대 부재력 + 지배 케이스 (상위 N개).

    Args:
        sort_by: 정렬 기준 - "moment", "axial", "shear", "torsion"
    """
    _sort_keys = {
        "moment": "max_M", "axial": "max_N",
        "shear": "max_V", "torsion": "max_T",
    }
    sort_key = _sort_keys.get(sort_by, "max_M")

    if not r.member_forces:
        return {"sort_basis": sort_by, "top_n": top_n, "items": []}

    # Node z lookup
    node_z = {n["id"]: n["z_m"] for n in r.nodes}

    # Cumulative heights for story mapping
    cum_h = []
    ch = 0.0
    for h in r.stories:
        ch += h
        cum_h.append(ch)

    def get_story(z):
        for i, c in enumerate(cum_h):
            if z <= c + 0.01:
                return i + 1
        return len(r.stories)

    # For each member, find max forces across all cases/combos
    member_max = {}

    for case_name, mf_list in r.member_forces.items():
        for mf in mf_list:
            mid = mf["member_id"]
            if mid not in member_max:
                z_ni = node_z.get(mf["ni"], 0)
                z_nj = node_z.get(mf["nj"], 0)
                member_z = max(z_ni, z_nj)
                member_max[mid] = {
                    "member_id": mid,
                    "type": mf["type"],
                    "story": get_story(member_z),
                    "max_N": 0.0, "gov_N": "-",
                    "max_M": 0.0, "gov_M": "-",
                    "max_V": 0.0, "gov_V": "-",
                    "max_T": 0.0, "gov_T": "-",
                }

            entry = member_max[mid]

            # Max axial
            max_n = max((abs(v) for v in mf["N_kN"]), default=0.0)
            if max_n > entry["max_N"]:
                entry["max_N"] = round(max_n, 2)
                entry["gov_N"] = case_name

            # Max moment (max of My and Mz)
            max_my = max((abs(v) for v in mf["My_kNm"]), default=0.0)
            max_mz = max((abs(v) for v in mf["Mz_kNm"]), default=0.0)
            max_m = max(max_my, max_mz)
            if max_m > entry["max_M"]:
                entry["max_M"] = round(max_m, 2)
                entry["gov_M"] = case_name

            # Max shear
            max_vy = max((abs(v) for v in mf["Vy_kN"]), default=0.0)
            max_vz = max((abs(v) for v in mf["Vz_kN"]), default=0.0)
            max_v = max(max_vy, max_vz)
            if max_v > entry["max_V"]:
                entry["max_V"] = round(max_v, 2)
                entry["gov_V"] = case_name

            # Max torsion
            max_t = max((abs(v) for v in mf["T_kNm"]), default=0.0)
            if max_t > entry["max_T"]:
                entry["max_T"] = round(max_t, 2)
                entry["gov_T"] = case_name

    # Sort by chosen criterion, take top N
    sorted_members = sorted(
        member_max.values(), key=lambda x: x[sort_key], reverse=True)
    return {
        "sort_basis": sort_by,
        "top_n": top_n,
        "items": sorted_members[:top_n],
    }


def _compute_equilibrium(r) -> dict:
    """하중케이스별 외력-반력 평형 검증 (residual 기반).

    외력 합계를 load_cases 정의에서 복원 가능한 경우 residual을 계산.
    복원 불가능한 하중 유형이 포함된 경우 status = 'Partial Check'.
    """
    EQ_TOL = 0.1  # kN

    # --- Step 1: 각 load case의 외력 합계 복원 ---
    # NOTE: ops.nodeReaction()은 nodal 하중(ops.load)만 반영하고,
    # 요소 하중(ops.eleLoad — beamUniform 등)은 반영하지 않음.
    # 따라서 floor_area/floor 하중은 반력 비교에서 제외.
    case_applied = {}

    for case_name, case_loads in r.load_cases.items():
        fx, fy, fz = 0.0, 0.0, 0.0
        computable = True
        has_eleload = False
        for ld in case_loads:
            ltype = ld.get("type", "")
            val = ld.get("value", 0.0)
            if ltype == "lateral_x":
                fx += val
            elif ltype == "lateral_y":
                fy += val
            elif ltype in ("floor_area", "floor"):
                # eleLoad(beamUniform)로 적용 → nodeReaction에 미반영
                has_eleload = True
            elif ltype == "nodal":
                fx += ld.get("fx", 0.0)
                fy += ld.get("fy", 0.0)
                fz += ld.get("fz", 0.0)
            else:
                computable = False
        case_applied[case_name] = {
            "fx": fx, "fy": fy, "fz": fz,
            "computable": computable, "has_eleload": has_eleload}

    # --- Step 2: combo 외력 = 선형중첩 ---
    combo_applied = {}
    for combo_name, factors in r.load_combinations.items():
        fx, fy, fz = 0.0, 0.0, 0.0
        computable = True
        has_eleload = False
        for cn, factor in factors.items():
            ca = case_applied.get(cn, {
                "fx": 0, "fy": 0, "fz": 0,
                "computable": False, "has_eleload": False})
            fx += ca["fx"] * factor
            fy += ca["fy"] * factor
            fz += ca["fz"] * factor
            if not ca["computable"]:
                computable = False
            if ca.get("has_eleload"):
                has_eleload = True
        combo_applied[combo_name] = {
            "fx": fx, "fy": fy, "fz": fz,
            "computable": computable, "has_eleload": has_eleload}

    all_applied = {}
    all_applied.update(case_applied)
    all_applied.update(combo_applied)

    # --- Step 3: 반력과 비교 ---
    all_results = {}
    all_results.update(r.case_results)
    all_results.update(r.combo_results)

    checks = []
    for name, cr in all_results.items():
        sum_rx = sum(rx["RX_kN"] for rx in cr.reactions)
        sum_ry = sum(rx["RY_kN"] for rx in cr.reactions)
        sum_rz = sum(rx["RZ_kN"] for rx in cr.reactions)

        finite = all(math.isfinite(v) for v in [sum_rx, sum_ry, sum_rz])
        applied = all_applied.get(name, {
            "fx": 0, "fy": 0, "fz": 0, "computable": False})

        if not finite:
            status = "WARNING (non-finite reactions)"
            res_fx = res_fy = res_fz = None
        elif applied["computable"]:
            # residual = reaction + applied (should be ~0)
            res_fx = round(sum_rx + applied["fx"], 4)
            res_fy = round(sum_ry + applied["fy"], 4)

            if applied.get("has_eleload"):
                # eleLoad(beamUniform)는 nodeReaction에 미반영 → FZ 비교 불가
                res_fz = None
                max_res = max(abs(res_fx), abs(res_fy))
                if max_res < EQ_TOL:
                    status = "OK (FX/FY verified, FZ has eleLoad)"
                else:
                    status = "FAIL (residual > tol)"
            else:
                res_fz = round(sum_rz + applied["fz"], 4)
                max_res = max(abs(res_fx), abs(res_fy), abs(res_fz))
                status = "OK" if max_res < EQ_TOL else "FAIL (residual > tol)"
        else:
            res_fx = res_fy = res_fz = None
            status = "Partial Check (some load types not computable)"

        checks.append({
            "case": name,
            "applied_fx": round(applied["fx"], 2) if applied["computable"] else None,
            "applied_fy": round(applied["fy"], 2) if applied["computable"] else None,
            "applied_fz": round(applied["fz"], 2) if applied["computable"] else None,
            "reaction_fx": round(sum_rx, 2),
            "reaction_fy": round(sum_ry, 2),
            "reaction_fz": round(sum_rz, 2),
            "residual_fx": res_fx,
            "residual_fy": res_fy,
            "residual_fz": res_fz,
            "finite": finite,
            "tolerance_kN": EQ_TOL,
            "status": status,
        })

    return {"tolerance_kN": EQ_TOL, "checks": checks}


def validate_3d_global_response(r) -> dict:
    """3D 해석 결과의 전역 응답 검증 요약.

    Args:
        r: Frame3DMultiCaseResult

    Returns:
        dict: max_disp, story_drifts, base_reactions, overturning 검증 포함
    """
    result = {
        "max_displacement": {},
        "story_drift_summary": [],
        "base_reaction_summary": {},
        "overturning_check": [],
    }

    # 전 케이스/조합에서 최대 변위
    all_results = {}
    all_results.update(r.case_results)
    all_results.update(r.combo_results)

    max_dx, max_dy, gov_dx, gov_dy = 0.0, 0.0, "", ""
    for name, cr in all_results.items():
        if abs(cr.max_displacement_x) > abs(max_dx):
            max_dx = cr.max_displacement_x
            gov_dx = name
        if abs(cr.max_displacement_y) > abs(max_dy):
            max_dy = cr.max_displacement_y
            gov_dy = name

    result["max_displacement"] = {
        "x_mm": round(max_dx, 3), "governing_x": gov_dx,
        "y_mm": round(max_dy, 3), "governing_y": gov_dy,
        "height_mm": r.total_height * 1000,
        "ratio_x": f"H/{max(1, int(r.total_height * 1000 / max(abs(max_dx), 0.001)))}",
        "ratio_y": f"H/{max(1, int(r.total_height * 1000 / max(abs(max_dy), 0.001)))}",
    }

    # 층별 drift 요약 (모든 케이스/조합 envelope)
    n_stories = len(r.stories)
    drift_env_x = [0.0] * n_stories
    drift_env_y = [0.0] * n_stories
    drift_gov_x = [""] * n_stories
    drift_gov_y = [""] * n_stories

    for name, cr in all_results.items():
        for sd in cr.story_drifts:
            s = sd.get("story", 1) - 1
            if 0 <= s < n_stories:
                dx = abs(sd.get("drift_x", 0.0))
                dy = abs(sd.get("drift_y", 0.0))
                if dx > drift_env_x[s]:
                    drift_env_x[s] = dx
                    drift_gov_x[s] = name
                if dy > drift_env_y[s]:
                    drift_env_y[s] = dy
                    drift_gov_y[s] = name

    for s in range(n_stories):
        result["story_drift_summary"].append({
            "story": s + 1,
            "max_drift_x": round(drift_env_x[s], 6),
            "governing_x": drift_gov_x[s],
            "max_drift_y": round(drift_env_y[s], 6),
            "governing_y": drift_gov_y[s],
        })

    # 기저부 반력 요약 (케이스별)
    for name, cr in r.case_results.items():
        sum_rx = sum(rx["RX_kN"] for rx in cr.reactions)
        sum_ry = sum(rx["RY_kN"] for rx in cr.reactions)
        sum_rz = sum(rx["RZ_kN"] for rx in cr.reactions)
        result["base_reaction_summary"][name] = {
            "sum_RX_kN": round(sum_rx, 2),
            "sum_RY_kN": round(sum_ry, 2),
            "sum_RZ_kN": round(sum_rz, 2),
        }

    # 전도 안정성 확인 (횡력 케이스 — 반력 모멘트 방향 일관성)
    for name, cr in r.case_results.items():
        loads = r.load_cases.get(name, [])
        has_lateral = any(
            ld.get("type", "") in ("lateral_x", "lateral_y") for ld in loads)
        if not has_lateral:
            continue

        sum_rx = sum(rx["RX_kN"] for rx in cr.reactions)
        sum_ry = sum(rx["RY_kN"] for rx in cr.reactions)
        sum_rz = sum(rx["RZ_kN"] for rx in cr.reactions)

        # 기저부 전도 모멘트 (=횡력×높이) vs 반력 수직력 검증
        total_lateral_x = sum(
            ld.get("value", 0.0) for ld in loads
            if ld.get("type") == "lateral_x")
        total_lateral_y = sum(
            ld.get("value", 0.0) for ld in loads
            if ld.get("type") == "lateral_y")

        status = "OK"
        if total_lateral_x != 0 and abs(sum_rx + total_lateral_x) > 0.1:
            status = "WARNING (X reaction mismatch)"
        if total_lateral_y != 0 and abs(sum_ry + total_lateral_y) > 0.1:
            status = "WARNING (Y reaction mismatch)"

        result["overturning_check"].append({
            "case": name,
            "lateral_x_kN": round(total_lateral_x, 2),
            "lateral_y_kN": round(total_lateral_y, 2),
            "reaction_x_kN": round(sum_rx, 2),
            "reaction_y_kN": round(sum_ry, 2),
            "status": status,
        })

    return result


def _get_model_settings(r) -> dict:
    """해석 설정 정보."""
    meta = getattr(r, 'analysis_metadata', {})
    geo_nl = getattr(r, 'geometric_nonlinearity', 'linear')
    actual_transf = meta.get('actual_transf', 'Linear')

    if geo_nl == "pdelta":
        analysis_type = f"Nonlinear Static (3D {actual_transf})"
        geo_nl_desc = f"Enabled — {actual_transf} transformation"
    else:
        analysis_type = "Linear Static"
        geo_nl_desc = "Disabled (linear geometry)"

    settings = {
        "analysis_type": analysis_type,
        "element_type": "elasticBeamColumn",
        "geometric_nonlinearity": geo_nl_desc,
        "rigid_diaphragm": getattr(r, 'rigid_diaphragm', False),
        "mesh_subdivision": r.num_elements_per_member,
        "support_type": r.supports,
        "dof": 6,
    }
    # solver 메타데이터
    if meta:
        settings["solver_algorithm"] = meta.get("solver_algorithm", "Linear")
        settings["fallback_used"] = meta.get("fallback_used", False)
        settings["n_load_steps"] = meta.get("n_steps", 1)
    releases = getattr(r, 'member_releases', None) or {}
    if releases:
        settings["member_releases"] = releases
        # Support level per member type
        support_info = {}
        for k, v in releases.items():
            if v:
                if k in ("beam", "beam_x", "beam_y"):
                    support_info[k] = "supported"
                elif k == "column":
                    support_info[k] = "experimental"
        settings["release_support_level"] = support_info
    return settings


# ============================================================
# HTML Report Generator
# ============================================================

def plot_frame_3d_interactive(multi_result, output_path=None, deformation_scale=50.0, assumptions=None, design_check=None, interpretation=None, paper_mode=False, paper_case=None) -> str:
    """3D 프레임 해석 결과 → 인터랙티브 HTML 리포트.

    Args:
        multi_result: Frame3DMultiCaseResult 객체
        output_path: HTML 파일 경로 (None이면 임시 파일)
        deformation_scale: 변형 배율 기본값 (0~200)
        assumptions: build_assumption_summary() 결과 dict (None이면 탭 미표시)
        design_check: run_design_check() 결과 dict (None이면 탭 미표시)
        paper_mode: True면 논문용 2색 3D 뷰만 생성 (UI 제거)
        paper_case: paper_mode에서 표시할 하중조합 이름 (None이면 첫 번째 조합)

    Returns:
        str: 생성된 HTML 파일 경로
    """
    data = prepare_3d_viz_data(multi_result)

    # Paper mode: 논문용 2색 3D 뷰만 생성
    if paper_mode:
        return _generate_paper_mode_html(data, output_path, deformation_scale, paper_case)

    # Assumptions 데이터 삽입 (있는 경우만)
    if assumptions is not None:
        data["assumptions"] = assumptions
    else:
        data["assumptions"] = None

    mi = data["model_info"]

    # Build case select options
    opts = ''
    if data["case_names"]:
        opts += '<optgroup label="Load Cases">'
        for n in data["case_names"]:
            opts += '<option value="' + n + '">' + n + '</option>'
        opts += '</optgroup>'
    if data["combo_names"]:
        opts += '<optgroup label="Combinations">'
        for n in data["combo_names"]:
            opts += '<option value="' + n + '">' + n + '</option>'
        opts += '</optgroup>'

    # V2 메타: bays가 0이면 노드/요소 수 표시
    if mi["num_bays_x"] > 0 or mi["num_bays_y"] > 0:
        geo_str = (str(mi["num_bays_x"]) + '\u00d7' + str(mi["num_bays_y"]) + ' bays')
    else:
        geo_str = (str(mi.get("num_nodes", "?")) + ' nodes, '
                   + str(mi.get("num_elements", "?")) + ' elems')
    meta = (str(mi["num_stories"]) + 'F | '
            + geo_str + ' | '
            + 'Col: ' + mi["column_section"] + ' | '
            + 'Beam: ' + mi["beam_x_section"] + '/' + mi["beam_y_section"] + ' | '
            + mi["material_name"] + ' (' + str(int(mi["E_MPa"])) + ' MPa)')

    # Assumptions tab (조건부 삽입)
    if assumptions is not None:
        assumptions_tab_btn = '  <button class="tab" id="tab-btn-assumptions" onclick="switchTab(\'assumptions\',this)">\uc785\ub825 \uac00\uc815</button>\n'
        assumptions_tab_div = """
<div id="tab-assumptions" class="tab-content" style="display:none">
  <div id="assumptionsOverview"></div>
  <div id="assumptionsWarnings"></div>
  <div id="occupancyTrace"></div>
  <div id="assumptionsDetail"></div>
</div>
"""
    else:
        assumptions_tab_btn = ''
        assumptions_tab_div = ''

    # Design Check tab (조건부 삽입)
    if design_check is not None:
        data["design_check"] = design_check
        dc_tab_btn = '  <button class="tab" id="tab-btn-designcheck" onclick="switchTab(\'designcheck\',this)">Design Check</button>\n'
        dc_tab_div = """
<div id="tab-designcheck" class="tab-content" style="display:none">
  <div id="dcBanner"></div>
  <div id="dcDriftSection"></div>
  <div id="dcMemberSection"></div>
  <div id="dcIssuesSection"></div>
</div>
"""
    else:
        dc_tab_btn = ''
        dc_tab_div = ''

    # Interpretation tab (조건부 삽입)
    if interpretation is not None:
        data["interpretation"] = interpretation
        interp_tab_btn = '  <button class="tab" id="tab-btn-interpretation" onclick="switchTab(\'interpretation\',this)">Interpretation</button>\n'
        interp_tab_div = """
<div id="tab-interpretation" class="tab-content" style="display:none">
  <div id="interpBanner"></div>
  <div id="interpFindings"></div>
  <div id="interpDiagnosis"></div>
  <div id="interpSuggestions"></div>
</div>
"""
    else:
        interp_tab_btn = ''
        interp_tab_div = ''

    # design_check / interpretation 삽입 후 직렬화
    data_json = json.dumps(data, ensure_ascii=False)

    html = _HTML_TEMPLATE
    html = html.replace('__DATA_JSON__', data_json)
    html = html.replace('__DEFAULT_SCALE__', str(int(deformation_scale)))
    html = html.replace('__META_TEXT__', meta)
    html = html.replace('__CASE_OPTIONS__', opts)
    html = html.replace('__ASSUMPTIONS_TAB_BTN__', assumptions_tab_btn)
    html = html.replace('__ASSUMPTIONS_TAB_DIV__', assumptions_tab_div)
    html = html.replace('__DC_TAB_BTN__', dc_tab_btn)
    html = html.replace('__DC_TAB_DIV__', dc_tab_div)
    html = html.replace('__INTERP_TAB_BTN__', interp_tab_btn)
    html = html.replace('__INTERP_TAB_DIV__', interp_tab_div)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.html')
        os.close(fd)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


# ============================================================
# HTML Template (placeholder replacement, no f-string escaping)
# ============================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D Frame Analysis Results</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#333}
.header{background:#1a237e;color:#fff;padding:16px 24px}
.header h1{font-size:20px;font-weight:600}
.header .meta{font-size:13px;opacity:.85;margin-top:4px}
.controls{display:flex;align-items:center;gap:12px;padding:12px 24px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
.controls label{font-size:13px;font-weight:500;color:#555}
.controls select{padding:6px 10px;border:1px solid #ccc;border-radius:4px;font-size:13px}
.controls input[type=range]{width:200px}
#scaleValue{font-size:13px;font-weight:600;color:#1a237e;min-width:40px}
.warnings-banner{padding:10px 24px;background:#fff3e0;border-bottom:1px solid #ffe0b2}
.warnings-banner .warn-title{font-size:12px;font-weight:600;color:#e65100;margin-bottom:4px;cursor:pointer;user-select:none}
.warnings-banner .warn-list{font-size:12px;color:#bf360c;line-height:1.6}
.warnings-banner .warn-item{display:inline-block;margin-right:16px;white-space:nowrap}
.warnings-banner .warn-item.severity-warning{color:#e65100;font-weight:500}
.warnings-banner .warn-item.severity-info{color:#795548}
.tab-bar{display:flex;gap:0;background:#fff;border-bottom:2px solid #1a237e;padding:0 24px}
.tab{padding:10px 20px;border:none;background:none;cursor:pointer;font-size:14px;color:#666;border-bottom:3px solid transparent;margin-bottom:-2px}
.tab.active{color:#1a237e;border-bottom-color:#1a237e;font-weight:600}
.tab:hover{color:#1a237e;background:#f0f0f8}
.tab-content{padding:20px 24px}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:16px}
.summary-card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.summary-card .label{font-size:11px;color:#888;text-transform:uppercase;margin-bottom:4px}
.summary-card .value{font-size:20px;font-weight:700;color:#1a237e}
.summary-card .detail{font-size:11px;color:#666;margin-top:2px}
.drift-pass{background:#e8f5e9;border-left:4px solid #4CAF50;padding:12px 16px;margin-bottom:16px;border-radius:4px}
.drift-fail{background:#ffebee;border-left:4px solid #f44336;padding:12px 16px;margin-bottom:16px;border-radius:4px}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-top:16px}
th{background:#f5f5f5;padding:10px 12px;font-size:12px;text-align:right;color:#555;border-bottom:2px solid #ddd}
th:first-child{text-align:left}
td{padding:8px 12px;font-size:13px;text-align:right;border-bottom:1px solid #eee}
td:first-child{text-align:left;font-weight:500}
tr:hover{background:#f8f8ff}
.section-title{font-size:16px;font-weight:600;color:#1a237e;margin:24px 0 8px 0;padding-bottom:6px;border-bottom:2px solid #e8eaf6}
.section-title:first-child{margin-top:0}
.gov-case{font-size:11px;color:#1565c0;font-weight:400}
.eq-ok{color:#2e7d32;font-weight:600}
.eq-warn{color:#c62828;font-weight:600}
.settings-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
.settings-grid td{border-right:1px solid #eee}
.export-buttons{display:flex;gap:12px;flex-wrap:wrap}
.export-buttons button{padding:10px 20px;background:#1a237e;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px}
.export-buttons button:hover{background:#283593}
.model-section{background:#fff;border-radius:8px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:16px}
.model-section h3{font-size:15px;margin-bottom:12px;color:#1a237e}
.drift-controls{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.drift-controls select{padding:6px 10px;border:1px solid #ccc;border-radius:4px}
.assumption-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.assumption-badge.user_input{background:#e3f2fd;color:#1565c0}
.assumption-badge.default{background:#fff3e0;color:#e65100}
.assumption-badge.ifc_detected{background:#e8f5e9;color:#2e7d32}
.assumption-badge.calculated{background:#f3e5f5;color:#7b1fa2}
.assumption-badge.fallback{background:#fce4ec;color:#c62828}
.assumption-badge.fallback_unmapped{background:#fce4ec;color:#c62828}
.assumption-badge.fallback_db_miss{background:#fce4ec;color:#c62828}
.assumption-badge.db_lookup{background:#e8f5e9;color:#2e7d32}
.asmp-overview{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
.asmp-stat{background:#fff;border-radius:8px;padding:14px 20px;box-shadow:0 1px 3px rgba(0,0,0,.1);text-align:center;min-width:120px}
.asmp-stat .num{font-size:28px;font-weight:700}
.asmp-stat .lbl{font-size:11px;color:#888;text-transform:uppercase;margin-top:2px}
.asmp-warn-item{padding:8px 12px;margin-bottom:6px;border-radius:4px;font-size:13px}
.asmp-warn-item.severity-warning{background:#fff3e0;border-left:4px solid #ff9800}
.asmp-warn-item.severity-caution{background:#fffde7;border-left:4px solid #fbc02d}
.asmp-warn-item.severity-info{background:#e3f2fd;border-left:4px solid #42a5f5}
.trace-arrow{color:#999;font-size:16px;padding:0 4px}
.asmp-group-title{font-size:14px;font-weight:600;color:#1a237e;margin:16px 0 6px 0;padding:6px 12px;background:#e8eaf6;border-radius:4px;cursor:pointer;user-select:none}
@media print{
  .controls,.tab-bar,.export-buttons,.warnings-banner{display:none}
  .tab-content{display:block!important;page-break-inside:avoid}
  body{background:#fff}
}
</style>
</head>
<body>

<div class="header" style="display:flex;justify-content:space-between;align-items:center">
  <div>
    <h1 id="headerTitle">3D Frame Analysis Results</h1>
    <div class="meta">__META_TEXT__</div>
  </div>
  <button id="langToggle" onclick="toggleLang()" style="background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.4);color:#fff;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;white-space:nowrap">한국어</button>
</div>

<div class="controls">
  <label>Load Case:</label>
  <select id="caseSelect" onchange="onCaseChange()">__CASE_OPTIONS__</select>
  <label>Deformation Scale:</label>
  <input type="range" id="scaleSlider" min="0" max="200" value="__DEFAULT_SCALE__" oninput="onScaleInput()">
  <span id="scaleValue">__DEFAULT_SCALE__&times;</span>
</div>

<div id="warningsBanner" class="warnings-banner"></div>

<div class="tab-bar" id="tabBar">
  <button class="tab active" id="tab-btn-view3d" onclick="switchTab('view3d',this)">3D View</button>
__DC_TAB_BTN____INTERP_TAB_BTN__  <button class="tab" id="tab-btn-summary" onclick="switchTab('summary',this)">Summary</button>
  <button class="tab" id="tab-btn-drift" onclick="switchTab('drift',this)">Story Drift</button>
  <button class="tab" id="tab-btn-reactions" onclick="switchTab('reactions',this)">Reactions</button>
  <button class="tab" id="tab-btn-model" onclick="switchTab('model',this)">Model Info</button>
__ASSUMPTIONS_TAB_BTN__  <button class="tab" id="tab-btn-export" onclick="switchTab('export',this)">Export</button>
</div>

<div id="tab-view3d" class="tab-content">
  <div id="overviewBanner"></div>
  <div id="summaryCards" class="summary-grid"></div>
  <div id="plot3d"></div>
</div>

<div id="tab-summary" class="tab-content" style="display:none">
  <div id="modalAnalysisSection"></div>
  <div id="envelopeSection"></div>
  <div id="storySummarySection"></div>
  <div id="criticalMembersSection"></div>
</div>

<div id="tab-drift" class="tab-content" style="display:none">
  <div class="drift-controls">
    <label>Allowable Drift:</label>
    <select id="driftLimit" onchange="renderDriftTab()">
      <option value="200">1/200 (KDS General)</option>
      <option value="300">1/300</option>
      <option value="400">1/400</option>
    </select>
  </div>
  <div id="driftBanner"></div>
  <div id="driftChart"></div>
  <div id="driftTable"></div>
</div>

<div id="tab-reactions" class="tab-content" style="display:none">
  <div id="reactionsTable"></div>
  <div id="equilibriumSection"></div>
</div>

<div id="tab-model" class="tab-content" style="display:none">
  <div id="modelInfo"></div>
</div>

__ASSUMPTIONS_TAB_DIV__
__DC_TAB_DIV__
__INTERP_TAB_DIV__
<div id="tab-export" class="tab-content" style="display:none">
  <div class="export-buttons">
    <button onclick="exportCSV('displacements')">Displacements CSV</button>
    <button onclick="exportCSV('reactions')">Reactions CSV</button>
    <button onclick="exportCSV('drifts')">Story Drifts CSV</button>
    <button onclick="exportPNG()">3D View PNG</button>
  </div>
</div>

<script>
var DATA = __DATA_JSON__;
var DEFAULT_SCALE = __DEFAULT_SCALE__;
var SUMMARY = DATA.summary;
var currentCase = DATA.all_names[0];
var currentScale = DEFAULT_SCALE;
var currentTab = 'view3d';
var LANG = 'en';

// ============================================================
// i18n Labels
// ============================================================
var LABELS = {
  // Header
  header_title: {en:'3D Frame Analysis Results', ko:'3D \ud504\ub808\uc784 \ud574\uc11d \uacb0\uacfc'},
  // Tab buttons
  tab_view3d: {en:'3D View', ko:'3D \ubdf0'},
  tab_summary: {en:'Summary', ko:'\uc694\uc57d'},
  tab_designcheck: {en:'Design Check', ko:'\uc124\uacc4\uac80\ud1a0'},
  tab_interpretation: {en:'Interpretation', ko:'\uacb0\uacfc \ud574\uc11d'},
  tab_drift: {en:'Story Drift', ko:'\uce35\uac04\ubcc0\uc704'},
  tab_reactions: {en:'Reactions', ko:'\ubc18\ub825'},
  tab_model: {en:'Model Info', ko:'\ubaa8\ub378 \uc815\ubcf4'},
  tab_assumptions: {en:'Assumptions', ko:'\uc785\ub825 \uac00\uc815'},
  tab_export: {en:'Export', ko:'\ub0b4\ubcf4\ub0b4\uae30'},
  // Overview banner
  ov_max_util: {en:'Max Utilization', ko:'\ucd5c\ub300 \ud65c\uc6a9\ub960'},
  ov_governing: {en:'Governing', ko:'\uc9c0\ubc30 \uac80\ud1a0'},
  ov_critical_story: {en:'Critical Story', ko:'\uc704\ud5d8 \uce35'},
  ov_critical_member: {en:'Critical Member', ko:'\uc704\ud5d8 \ubd80\uc7ac'},
  ov_t1_ta: {en:'T1 / Ta', ko:'T1 / Ta'},
  ov_first_mode: {en:'1st Mode', ko:'1\ucc28 \ubaa8\ub4dc'},
  ov_drift: {en:'Drift', ko:'\ubcc0\uc704'},
  ov_member: {en:'Member', ko:'\ubd80\uc7ac'},
  ov_ratio: {en:'ratio', ko:'\ube44\uc728'},
  // Design Check tab
  dc_title: {en:'Design Check', ko:'\uc124\uacc4\uac80\ud1a0'},
  dc_not_available: {en:'Design check data not available.', ko:'\uc124\uacc4\uac80\ud1a0 \ub370\uc774\ud130\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.'},
  dc_max_drift: {en:'Max Drift Ratio', ko:'\ucd5c\ub300 \ubcc0\uc704 \ube44\uc728'},
  dc_max_interact: {en:'Max Interaction Ratio', ko:'\ucd5c\ub300 \uc0c1\uad00\ube44'},
  dc_ng_members: {en:'NG Members', ko:'NG \ubd80\uc7ac'},
  dc_drift_check: {en:'Story Drift Check', ko:'\uce35\uac04\ubcc0\uc704 \uac80\ud1a0'},
  dc_member_check: {en:'Member Strength Check', ko:'\ubd80\uc7ac \uac15\ub3c4 \uac80\ud1a0'},
  dc_story: {en:'Story', ko:'\uce35'},
  dc_dir: {en:'Dir', ko:'\ubc29\ud5a5'},
  dc_combo: {en:'Combo', ko:'\uc870\ud569'},
  dc_elastic: {en:'Elastic', ko:'\ud0c4\uc131'},
  dc_inelastic: {en:'Inelastic', ko:'\ube44\ud0c4\uc131'},
  dc_allowable: {en:'Allowable', ko:'\ud5c8\uc6a9'},
  dc_ratio: {en:'Ratio', ko:'\ube44\uc728'},
  dc_status: {en:'Status', ko:'\ud310\uc815'},
  dc_member: {en:'Member', ko:'\ubd80\uc7ac'},
  dc_type: {en:'Type', ko:'\uc720\ud615'},
  dc_section: {en:'Section', ko:'\ub2e8\uba74'},
  dc_axial: {en:'Axial', ko:'\ucd95\ub825'},
  dc_bending: {en:'Bending', ko:'\ud720'},
  dc_shear: {en:'Shear', ko:'\uc804\ub2e8'},
  dc_interaction: {en:'Interaction', ko:'\uc0c1\uad00'},
  dc_formula: {en:'Formula', ko:'\uacf5\uc2dd'},
  dc_critical_issues: {en:'Critical Issues', ko:'\uc8fc\uc694 \ubb38\uc81c'},
  dc_total: {en:'Total', ko:'\ud569\uacc4'},
  dc_assumptions: {en:'Assumptions', ko:'\uac00\uc815\uc0ac\ud56d'},
  dc_importance: {en:'Importance', ko:'\uc911\uc694\ub3c4'},
  // Interpretation tab
  it_findings: {en:'Key Findings', ko:'\uc8fc\uc694 \ubc1c\uacac'},
  it_evidence: {en:'Evidence Summary', ko:'\uadfc\uac70 \uc694\uc57d'},
  it_max_drift: {en:'Max Drift Ratio', ko:'\ucd5c\ub300 \ubcc0\uc704 \ube44\uc728'},
  it_max_interact: {en:'Max Interaction', ko:'\ucd5c\ub300 \uc0c1\uad00\ube44'},
  it_governing_check: {en:'Governing Check', ko:'\uc9c0\ubc30 \uac80\ud1a0'},
  it_governing_mode: {en:'Governing Mode', ko:'\uc9c0\ubc30 \ud30c\uad34 \ubaa8\ub4dc'},
  it_diagnosis: {en:'Failure Diagnosis', ko:'\ud30c\uad34 \uc9c4\ub2e8'},
  it_primary: {en:'Primary cause', ko:'\uc8fc \uc6d0\uc778'},
  it_contrib: {en:'Contributing factors', ko:'\uae30\uc5ec \uc694\uc778'},
  it_suggestions: {en:'Improvement Suggestions', ko:'\uac1c\uc120 \uc81c\uc548'},
  it_over_flexible: {en:'Over-flexible', ko:'\uacfc\uc720\uc5f0'},
  it_torsion_risk: {en:'Torsion Risk', ko:'\ube44\ud2c0\ub9bc \uc704\ud5d8'},
  it_impact: {en:'impact', ko:'\uc601\ud5a5\ub3c4'},
  it_target: {en:'Target', ko:'\ub300\uc0c1'},
  // Summary tab
  sm_modal: {en:'Modal Analysis (Eigenvalue)', ko:'\ubaa8\ub2ec \ud574\uc11d (\uace0\uc720\uce58)'},
  sm_mode: {en:'Mode', ko:'\ubaa8\ub4dc'},
  sm_period: {en:'Period (s)', ko:'\uc8fc\uae30 (s)'},
  sm_freq: {en:'Freq (Hz)', ko:'\uc9c4\ub3d9\uc218 (Hz)'},
  sm_direction: {en:'Direction', ko:'\ubc29\ud5a5'},
  sm_dominance: {en:'Dominance', ko:'\uc9c0\ubc30\ub3c4'},
  sm_cum_part: {en:'Cumulative Participation', ko:'\ub204\uc801 \ucc38\uc5ec'},
  sm_sufficiency: {en:'90% Sufficiency', ko:'90% \ucda9\uc871'},
  sm_mass_assumptions: {en:'Mass Assumptions', ko:'\uc9c8\ub7c9 \uac00\uc815'},
  sm_total_weight: {en:'Total weight', ko:'\ucd1d \uc911\ub7c9'},
  sm_weight_source: {en:'Weight source', ko:'\uc911\ub7c9 \ucd9c\ucc98'},
  sm_user_weights: {en:'User-provided story weights', ko:'\uc0ac\uc6a9\uc790 \uc81c\uacf5 \uce35\ubcc4 \uc911\ub7c9'},
  sm_estimated_dl: {en:'Estimated from DL load case', ko:'\uace0\uc815\ud558\uc911\uc73c\ub85c\ubd80\ud130 \ucd94\uc815'},
  sm_envelope: {en:'Response Envelope (All Cases & Combinations)', ko:'\uc751\ub2f5 \ud3ec\ub77d\uc120 (\uc804\uccb4 \ucf00\uc774\uc2a4 & \uc870\ud569)'},
  sm_metric: {en:'Metric', ko:'\uc9c0\ud45c'},
  sm_max_value: {en:'Max Value', ko:'\ucd5c\ub300\uac12'},
  sm_unit: {en:'Unit', ko:'\ub2e8\uc704'},
  sm_gov_case: {en:'Governing Case', ko:'\uc9c0\ubc30 \ucf00\uc774\uc2a4'},
  sm_location: {en:'Location', ko:'\uc704\uce58'},
  sm_story_summary: {en:'Story Summary (Envelope)', ko:'\uce35\ubcc4 \uc694\uc57d (\ud3ec\ub77d\uc120)'},
  sm_height: {en:'Height (m)', ko:'\ub192\uc774 (m)'},
  sm_cum_h: {en:'Cum. H (m)', ko:'\ub204\uc801\ub192\uc774 (m)'},
  sm_critical_members: {en:'Critical Members', ko:'\uc704\ud5d8 \ubd80\uc7ac'},
  sm_sorted_by: {en:'sorted by', ko:'\uc815\ub82c \uae30\uc900'},
  sm_member: {en:'Member', ko:'\ubd80\uc7ac'},
  sm_type: {en:'Type', ko:'\uc720\ud615'},
  // Warnings
  wn_limitations: {en:'Modeling Assumptions & Notes', ko:'\ubaa8\ub378\ub9c1 \uac00\uc815 \ubc0f \ucc38\uace0\uc0ac\ud56d'},
  wn_nonlinear: {en:'Geometric Nonlinearity Summary', ko:'\uae30\ud558\ube44\uc120\ud615 \uc694\uc57d'},
  wn_req_mode: {en:'Requested Mode', ko:'\uc694\uccad \ubaa8\ub4dc'},
  wn_transformation: {en:'Transformation', ko:'\ubcc0\ud658'},
  wn_solver: {en:'Solver', ko:'\uc194\ubc84'},
  wn_fallback: {en:'fallback', ko:'\ub300\uccb4'},
  wn_load_steps: {en:'Load Steps', ko:'\ud558\uc911 \ub2e8\uacc4'},
  wn_reliability: {en:'Result Reliability', ko:'\uacb0\uacfc \uc2e0\ub8b0\ub3c4'},
  wn_disp_drift: {en:'Displacement/Drift', ko:'\ubcc0\uc704/\uce35\uac04\ubcc0\uc704'},
  wn_base_rxn: {en:'Base Reactions', ko:'\uc9c0\uc810 \ubc18\ub825'},
  wn_local_forces: {en:'Local Element Forces', ko:'\ub85c\uceec \uc694\uc18c\ub825'},
  // Model Info tab
  mi_analysis_settings: {en:'Analysis Settings', ko:'\ud574\uc11d \uc124\uc815'},
  mi_analysis_type: {en:'Analysis Type', ko:'\ud574\uc11d \uc720\ud615'},
  mi_element_type: {en:'Element Type', ko:'\uc694\uc18c \uc720\ud615'},
  mi_dof: {en:'DOF per Node', ko:'\uc808\uc810\ub2f9 \uc790\uc720\ub3c4'},
  mi_geom_nl: {en:'Geometric Nonlinearity', ko:'\uae30\ud558\ube44\uc120\ud615'},
  mi_solver: {en:'Solver Algorithm', ko:'\uc194\ubc84 \uc54c\uace0\ub9ac\uc998'},
  mi_load_steps: {en:'Load Steps', ko:'\ud558\uc911 \ub2e8\uacc4'},
  mi_rigid_diap: {en:'Rigid Diaphragm', ko:'\uac15\uccb4 \ub2e4\uc774\uc5b4\ud504\ub7a8'},
  mi_applied: {en:'Applied', ko:'\uc801\uc6a9'},
  mi_not_applied: {en:'Not Applied', ko:'\ubbf8\uc801\uc6a9'},
  mi_mesh: {en:'Mesh Subdivision', ko:'\uba54\uc26c \ubd84\ud560'},
  mi_elem_per_member: {en:'elements per member', ko:'\ubd80\uc7ac\ub2f9 \uc694\uc18c \uc218'},
  mi_support_cond: {en:'Support Condition', ko:'\uc9c0\uc810 \uc870\uac74'},
  mi_building_geom: {en:'Building Geometry', ko:'\uac74\ubb3c \uae30\ud558'},
  mi_stories: {en:'Stories', ko:'\uce35\uc218'},
  mi_bays_x: {en:'Bays X', ko:'X\ubc29\ud5a5 \uacbd\uac04'},
  mi_bays_y: {en:'Bays Y', ko:'Y\ubc29\ud5a5 \uacbd\uac04'},
  mi_total_height: {en:'Total Height', ko:'\uc804\uccb4 \ub192\uc774'},
  mi_total_width_x: {en:'Total Width X', ko:'X\ubc29\ud5a5 \uc804\uccb4 \ud3ed'},
  mi_total_width_y: {en:'Total Width Y', ko:'Y\ubc29\ud5a5 \uc804\uccb4 \ud3ed'},
  mi_supports: {en:'Supports', ko:'\uc9c0\uc810'},
  mi_nodes_elems: {en:'Nodes / Members / Elements', ko:'\uc808\uc810 / \ubd80\uc7ac / \uc694\uc18c'},
  mi_sections: {en:'Sections', ko:'\ub2e8\uba74'},
  mi_column: {en:'Column', ko:'\uae30\ub465'},
  mi_beam_x: {en:'Beam X', ko:'X\ubcf4'},
  mi_beam_y: {en:'Beam Y', ko:'Y\ubcf4'},
  mi_material: {en:'Material', ko:'\uc7ac\ub8cc'},
  mi_name: {en:'Name', ko:'\uc774\ub984'},
  mi_load_cases: {en:'Load Cases', ko:'\ud558\uc911 \ucf00\uc774\uc2a4'},
  mi_load_combos: {en:'Load Combinations', ko:'\ud558\uc911 \uc870\ud569'},
  mi_model_limits: {en:'Modeling Assumptions & Notes', ko:'\ubaa8\ub378\ub9c1 \uac00\uc815 \ubc0f \ucc38\uace0\uc0ac\ud56d'},
  mi_code: {en:'Code', ko:'\ucf54\ub4dc'},
  mi_description: {en:'Description', ko:'\uc124\uba85'},
  mi_severity_col: {en:'Severity', ko:'\uc2ec\uac01\ub3c4'},
  mi_warning: {en:'Warning', ko:'\uacbd\uace0'},
  mi_info: {en:'Info', ko:'\uc815\ubcf4'},
  // Drift tab
  dr_pass: {en:'PASS', ko:'\ud1b5\uacfc'},
  dr_fail: {en:'FAIL', ko:'\ubd80\uc801\ud569'},
  dr_max_drift: {en:'Max drift', ko:'\ucd5c\ub300 \ubcc0\uc704'},
  dr_no_lateral: {en:'No lateral drift', ko:'\uc218\ud3c9 \ubcc0\uc704 \uc5c6\uc74c'},
  dr_check: {en:'Check', ko:'\ud310\uc815'},
  dr_drift_ratio: {en:'Drift Ratio', ko:'\ubcc0\uc704 \ube44\uc728'},
  // Reactions tab
  rx_support: {en:'Support Reactions', ko:'\uc9c0\uc810 \ubc18\ub825'},
  rx_sum: {en:'Sum', ko:'\ud569\uacc4'},
  rx_eq_check: {en:'Equilibrium Check', ko:'\ud3c9\ud615 \uac80\uc99d'},
  rx_tolerance: {en:'tolerance', ko:'\ud5c8\uc6a9 \uc624\ucc28'},
  rx_case: {en:'Case', ko:'\ucf00\uc774\uc2a4'},
  rx_applied: {en:'Applied', ko:'\uc801\uc6a9'},
  rx_residual: {en:'Res.', ko:'\uc794\ucc28'},
  // 3D View summary cards
  v3_max_disp: {en:'Max Disp.', ko:'\ucd5c\ub300\ubcc0\uc704'},
  v3_max_drift: {en:'Max Drift', ko:'\ucd5c\ub300\uce35\uac04\ubcc0\uc704'},
  v3_max_moment: {en:'Max Moment', ko:'\ucd5c\ub300\ubaa8\uba58\ud2b8'},
  v3_max_axial: {en:'Max Axial', ko:'\ucd5c\ub300\ucd95\ub825'},
};

function L(key) { return (LABELS[key] && LABELS[key][LANG]) || key; }

function toggleLang() {
  LANG = (LANG === 'en') ? 'ko' : 'en';
  document.getElementById('langToggle').textContent = (LANG === 'en') ? '\ud55c\uad6d\uc5b4' : 'English';
  document.getElementById('headerTitle').textContent = L('header_title');
  // Update tab labels
  var tabMap = {view3d:'tab_view3d',summary:'tab_summary',designcheck:'tab_designcheck',
    interpretation:'tab_interpretation',drift:'tab_drift',reactions:'tab_reactions',
    model:'tab_model',assumptions:'tab_assumptions',export:'tab_export'};
  for (var tid in tabMap) {
    var btn = document.getElementById('tab-btn-'+tid);
    if (btn) btn.textContent = L(tabMap[tid]);
  }
  // Re-render all content sections
  _dcRendered = false;
  _interpRendered = false;
  renderOverviewBanner();
  renderWarnings();
  renderSummaryTab();
  renderDesignCheckTab();
  renderInterpretationTab();
  renderDriftTab();
  renderReactionsTab();
  renderModelTab();
  updateSummary(currentCase);
}

function init() {
  document.getElementById('scaleSlider').value = DEFAULT_SCALE;
  document.getElementById('scaleValue').textContent = DEFAULT_SCALE + '\u00d7';
  renderWarnings();
  renderOverviewBanner();
  buildPlot(currentCase, currentScale);
  updateSummary(currentCase);
  renderSummaryTab();
  renderModelTab();
}

function switchTab(tabName, btn) {
  var tabs = document.querySelectorAll('.tab-content');
  for (var i = 0; i < tabs.length; i++) tabs[i].style.display = 'none';
  var btns = document.querySelectorAll('.tab');
  for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
  document.getElementById('tab-' + tabName).style.display = 'block';
  btn.classList.add('active');
  currentTab = tabName;
  if (tabName === 'drift') renderDriftTab();
  if (tabName === 'reactions') renderReactionsTab();
  if (tabName === 'assumptions') renderAssumptionsTab();
  if (tabName === 'designcheck') renderDesignCheckTab();
  if (tabName === 'interpretation') renderInterpretationTab();
}

function onCaseChange() {
  currentCase = document.getElementById('caseSelect').value;
  updateDeformed(currentCase, currentScale);
  updateSummary(currentCase);
  if (currentTab === 'drift') renderDriftTab();
  if (currentTab === 'reactions') renderReactionsTab();
}

function onScaleInput() {
  currentScale = parseInt(document.getElementById('scaleSlider').value);
  document.getElementById('scaleValue').textContent = currentScale + '\u00d7';
  updateDeformed(currentCase, currentScale);
}

// ============================================================
// Warnings Banner
// ============================================================

function renderWarnings() {
  var w = SUMMARY.warnings;
  if (!w || w.length === 0) return;
  var h = '<div class="warn-title">\u26a0 '+L('wn_limitations')+' (' + w.length + ')</div><div class="warn-list">';
  for (var i = 0; i < w.length; i++) {
    var wText = (LANG==='ko' && w[i].text_ko) ? w[i].text_ko : w[i].text;
    h += '<span class="warn-item severity-' + w[i].severity + '">[' + w[i].code + '] ' + wText + '</span>';
  }
  h += '</div>';
  document.getElementById('warningsBanner').innerHTML = h;
  // Nonlinear summary box
  var ns = SUMMARY.nonlinear_summary;
  if (ns) {
    var nh = '<div class="warn-title">'+L('wn_nonlinear')+'</div>';
    nh += '<table style="width:100%;font-size:12px;border-collapse:collapse;">';
    nh += '<tr><td style="padding:3px 8px;">'+L('wn_req_mode')+'</td><td style="padding:3px 8px;">'+ns.requested_mode+'</td></tr>';
    nh += '<tr><td style="padding:3px 8px;">'+L('wn_transformation')+'</td><td style="padding:3px 8px;font-weight:bold;">'+ns.transformation+'</td></tr>';
    nh += '<tr><td style="padding:3px 8px;">'+L('wn_solver')+'</td><td style="padding:3px 8px;">'+ns.solver_algorithm+(ns.fallback_used?' ('+L('wn_fallback')+')':'')+'</td></tr>';
    nh += '<tr><td style="padding:3px 8px;">'+L('wn_load_steps')+'</td><td style="padding:3px 8px;">'+ns.n_load_steps+'</td></tr>';
    nh += '</table>';
    // Force interpretation policy
    if (ns.force_interpretation) {
      var fi = ns.force_interpretation;
      nh += '<div style="margin-top:6px;font-size:11px;"><b>'+L('wn_reliability')+':</b></div>';
      nh += '<table style="width:100%;font-size:11px;border-collapse:collapse;">';
      nh += '<tr><td style="padding:2px 8px;">'+L('wn_disp_drift')+'</td><td style="padding:2px 8px;color:#2e7d32;">'+fi.displacement_drift+'</td></tr>';
      nh += '<tr><td style="padding:2px 8px;">'+L('wn_base_rxn')+'</td><td style="padding:2px 8px;color:#2e7d32;">'+fi.base_reactions+'</td></tr>';
      nh += '<tr><td style="padding:2px 8px;">'+L('wn_local_forces')+'</td><td style="padding:2px 8px;color:'+(fi.local_element_forces==='reliable'?'#2e7d32':'#e65100')+';">'+fi.local_element_forces+'</td></tr>';
      nh += '</table>';
    }
    if (ns.interpretation_notes && ns.interpretation_notes.length > 0) {
      nh += '<div style="margin-top:4px;font-size:11px;color:#666;">';
      for (var j = 0; j < ns.interpretation_notes.length; j++) {
        nh += '<div style="margin:2px 0;">&bull; '+ns.interpretation_notes[j]+'</div>';
      }
      nh += '</div>';
    }
    document.getElementById('warningsBanner').innerHTML += '<div style="margin-top:8px;padding:8px 12px;background:#e8f5e9;border-left:4px solid #4caf50;border-radius:4px;">'+nh+'</div>';
  }
}

// ============================================================
// Overview Banner (Top-Level Summary Card)
// ============================================================

function renderOverviewBanner() {
  var el = document.getElementById('overviewBanner');
  if (!el) return;

  var interp = DATA.interpretation;
  var dc = DATA.design_check;
  var ma = DATA.modal_analysis;

  // design_check도 interpretation도 없으면 배너 미표시
  if (!dc && !interp) return;

  var h = '';

  // ── Status Banner ──
  var severity = interp ? interp.severity : (dc && dc.overall_status === 'OK' ? 'safe' : 'moderate');
  var sevColors = {safe:'#2e7d32',marginal:'#e65100',moderate:'#c62828',severe:'#4a148c'};
  var sevBg = {safe:'#e8f5e9',marginal:'#fff3e0',moderate:'#ffebee',severe:'#f3e5f5'};
  var sevText = {safe:'SAFE',marginal:'MARGINAL',moderate:'NG',severe:'NG (SEVERE)'};
  var color = sevColors[severity] || '#333';
  var bg = sevBg[severity] || '#f5f5f5';
  var statusText = sevText[severity] || severity.toUpperCase();

  h += '<div style="background:'+bg+';border-left:6px solid '+color+';border-radius:8px;padding:14px 20px;margin-bottom:12px">';
  h += '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">';
  h += '<div style="font-size:24px;font-weight:bold;color:'+color+'">'+statusText+'</div>';
  // summary 문장
  if (interp) {
    var sumText = (LANG==='ko') ? (interp.summary_ko||'') : (interp.summary_en||interp.summary_ko||'');
    if (sumText) h += '<div style="font-size:13px;color:#333;flex:1;min-width:200px">'+sumText+'</div>';
  }
  h += '</div>';

  // ── Key Metrics Row ──
  h += '<div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap">';

  // Max Utilization
  var driftR = (dc && dc.summary) ? (dc.summary.max_drift_ratio || 0) : 0;
  var interR = (dc && dc.summary) ? (dc.summary.max_interaction_ratio || 0) : 0;
  var maxUtil = Math.max(driftR, interR);
  var utilColor = maxUtil <= 0.7 ? '#2e7d32' : (maxUtil <= 1.0 ? '#e65100' : '#c62828');
  h += '<div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:8px 14px;min-width:120px;text-align:center">';
  h += '<div style="font-size:10px;color:#888;text-transform:uppercase">'+L('ov_max_util')+'</div>';
  h += '<div style="font-size:20px;font-weight:bold;color:'+utilColor+'">'+(maxUtil*100).toFixed(0)+'%</div>';
  h += '</div>';

  // Governing Check
  var governing = driftR >= interR ? L('ov_drift') : L('ov_member');
  h += '<div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:8px 14px;min-width:120px;text-align:center">';
  h += '<div style="font-size:10px;color:#888;text-transform:uppercase">'+L('ov_governing')+'</div>';
  h += '<div style="font-size:16px;font-weight:bold;color:#333">'+governing+'</div>';
  var govRatio = Math.max(driftR, interR);
  h += '<div style="font-size:11px;color:#666">'+L('ov_ratio')+' '+govRatio.toFixed(3)+'</div>';
  h += '</div>';

  // Critical Story
  var critStory = '-';
  if (interp && interp.drift_interpretation && interp.drift_interpretation.critical_story) {
    critStory = interp.drift_interpretation.critical_story + 'F';
  } else if (interp && interp.member_interpretation && interp.member_interpretation.ng_by_story) {
    var ngs = interp.member_interpretation.ng_by_story;
    var maxNg = 0; var maxS = null;
    for (var s in ngs) { if (ngs[s] > maxNg) { maxNg = ngs[s]; maxS = s; } }
    if (maxS) critStory = maxS + 'F';
  }
  h += '<div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:8px 14px;min-width:120px;text-align:center">';
  h += '<div style="font-size:10px;color:#888;text-transform:uppercase">'+L('ov_critical_story')+'</div>';
  h += '<div style="font-size:16px;font-weight:bold;color:#333">'+critStory+'</div>';
  h += '</div>';

  // Critical Member (Weakest Link)
  if (interp && interp.member_interpretation && interp.member_interpretation.weakest_link) {
    var wl = interp.member_interpretation.weakest_link;
    var wlType = {column:'Col',beam_x:'Bm-X',beam_y:'Bm-Y'}[wl.type] || wl.type;
    var wlColor = wl.ratio > 1.0 ? '#c62828' : (wl.ratio > 0.7 ? '#e65100' : '#2e7d32');
    h += '<div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:8px 14px;min-width:120px;text-align:center">';
    h += '<div style="font-size:10px;color:#888;text-transform:uppercase">'+L('ov_critical_member')+'</div>';
    h += '<div style="font-size:16px;font-weight:bold;color:'+wlColor+'">'+wlType+' #'+wl.member_id+'</div>';
    h += '<div style="font-size:11px;color:#666">'+wl.story+'F, ratio '+wl.ratio.toFixed(2)+'</div>';
    h += '</div>';
  }

  // T1 / Ta
  if (interp && interp.modal_interpretation) {
    var mi = interp.modal_interpretation;
    var t1 = mi.T1_actual_s || 0;
    var ta = mi.Ta_empirical_s || 0;
    var ratioT = mi.T1_Ta_ratio || 0;
    var t1Color = mi.flexibility_flag ? '#c62828' : '#2e7d32';
    h += '<div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:8px 14px;min-width:120px;text-align:center">';
    h += '<div style="font-size:10px;color:#888;text-transform:uppercase">'+L('ov_t1_ta')+'</div>';
    h += '<div style="font-size:16px;font-weight:bold;color:'+t1Color+'">'+t1.toFixed(3)+'s</div>';
    h += '<div style="font-size:11px;color:#666">Ta='+ta.toFixed(3)+'s ('+ratioT.toFixed(2)+')</div>';
    h += '</div>';
  } else if (ma && ma.fundamental_periods) {
    var fp = ma.fundamental_periods;
    var t1v = Math.max(fp.T1_x_s || 0, fp.T1_y_s || 0);
    if (t1v > 0) {
      h += '<div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:8px 14px;min-width:120px;text-align:center">';
      h += '<div style="font-size:10px;color:#888;text-transform:uppercase">T1</div>';
      h += '<div style="font-size:16px;font-weight:bold;color:#333">'+t1v.toFixed(3)+'s</div>';
      h += '</div>';
    }
  }

  // 1st Mode Direction
  if (interp && interp.modal_interpretation) {
    var dir = interp.modal_interpretation.first_mode_direction || '-';
    var dirColor = dir === 'ROTN-Z' ? '#c62828' : '#1565c0';
    h += '<div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:8px 14px;min-width:120px;text-align:center">';
    h += '<div style="font-size:10px;color:#888;text-transform:uppercase">'+L('ov_first_mode')+'</div>';
    h += '<div style="font-size:16px;font-weight:bold;color:'+dirColor+'">'+dir+'</div>';
    h += '</div>';
  }

  h += '</div>'; // metrics row
  h += '</div>'; // banner
  el.innerHTML = h;
}

// ============================================================
// 3D Plot
// ============================================================

function buildPlot(caseName, scale) {
  var types = ['column', 'beam_x', 'beam_y'];
  var colors = ['#9E9E9E', '#2196F3', '#4CAF50'];
  var defColors = ['#f44336', '#ff5722', '#e91e63'];
  var names = ['Column', 'Beam X', 'Beam Y'];
  var traces = [];

  // Undeformed (traces 0-2)
  for (var t = 0; t < 3; t++) {
    var ms = DATA.members[types[t]];
    var x = [], y = [], z = [];
    for (var i = 0; i < ms.length; i++) {
      var ni = ms[i][0], nj = ms[i][1];
      var pi = DATA.node_map[String(ni)], pj = DATA.node_map[String(nj)];
      x.push(pi[0], pj[0], null);
      y.push(pi[1], pj[1], null);
      z.push(pi[2], pj[2], null);
    }
    traces.push({type:'scatter3d', mode:'lines', x:x, y:y, z:z,
      line:{color:colors[t], width:4}, name:names[t], legendgroup:'undeformed'});
  }

  // Deformed (traces 3-5)
  var cd = DATA.case_data[caseName], disp = cd.disp;
  for (var t = 0; t < 3; t++) {
    var ms = DATA.members[types[t]];
    var x = [], y = [], z = [];
    for (var i = 0; i < ms.length; i++) {
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
  for (var nid in DATA.node_map) {
    var p = DATA.node_map[nid], d = disp[nid] || [0,0,0];
    nx.push(p[0]+d[0]/1000*scale);
    ny.push(p[1]+d[1]/1000*scale);
    nz.push(p[2]+d[2]/1000*scale);
    texts.push('Node '+nid+'<br>dx: '+d[0].toFixed(2)+' mm<br>dy: '+d[1].toFixed(2)+' mm<br>dz: '+d[2].toFixed(2)+' mm');
  }
  traces.push({type:'scatter3d', mode:'markers', x:nx, y:ny, z:nz,
    marker:{size:2, color:'#333'}, text:texts, hoverinfo:'text', name:'Nodes', showlegend:false});

  // Supports (trace 7)
  var sx=[], sy=[], sz=[];
  for (var i = 0; i < DATA.support_ids.length; i++) {
    var p = DATA.node_map[String(DATA.support_ids[i])];
    sx.push(p[0]); sy.push(p[1]); sz.push(p[2]);
  }
  traces.push({type:'scatter3d', mode:'markers', x:sx, y:sy, z:sz,
    marker:{size:5, color:'#FF9800', symbol:'diamond'}, name:'Supports'});

  var layout = {
    scene: {
      xaxis:{title:'X (m)'}, yaxis:{title:'Y (m)'}, zaxis:{title:'Z (m)'},
      aspectmode:'data',
      camera:{eye:{x:1.5, y:1.5, z:1.0}}
    },
    height:700, margin:{l:0,r:0,t:30,b:0}, legend:{x:0,y:1}
  };
  Plotly.newPlot('plot3d', traces, layout, {responsive:true});
}

function updateDeformed(caseName, scale) {
  var cd = DATA.case_data[caseName], disp = cd.disp;
  var types = ['column', 'beam_x', 'beam_y'];

  for (var t = 0; t < 3; t++) {
    var ms = DATA.members[types[t]];
    var x=[], y=[], z=[];
    for (var i = 0; i < ms.length; i++) {
      var ni = ms[i][0], nj = ms[i][1];
      var pi = DATA.node_map[String(ni)], pj = DATA.node_map[String(nj)];
      var di = disp[String(ni)] || [0,0,0], dj = disp[String(nj)] || [0,0,0];
      x.push(pi[0]+di[0]/1000*scale, pj[0]+dj[0]/1000*scale, null);
      y.push(pi[1]+di[1]/1000*scale, pj[1]+dj[1]/1000*scale, null);
      z.push(pi[2]+di[2]/1000*scale, pj[2]+dj[2]/1000*scale, null);
    }
    Plotly.restyle('plot3d', {x:[x], y:[y], z:[z]}, [3+t]);
  }

  // Update nodes (trace 6)
  var nx=[], ny=[], nz=[], texts=[];
  for (var nid in DATA.node_map) {
    var p = DATA.node_map[nid], d = disp[nid] || [0,0,0];
    nx.push(p[0]+d[0]/1000*scale);
    ny.push(p[1]+d[1]/1000*scale);
    nz.push(p[2]+d[2]/1000*scale);
    texts.push('Node '+nid+'<br>dx: '+d[0].toFixed(2)+' mm<br>dy: '+d[1].toFixed(2)+' mm<br>dz: '+d[2].toFixed(2)+' mm');
  }
  Plotly.restyle('plot3d', {x:[nx], y:[ny], z:[nz], text:[texts]}, [6]);
}

// ============================================================
// Summary Cards (3D View tab)
// ============================================================

function updateSummary(caseName) {
  var mv = DATA.case_data[caseName].max;
  var fmt = function(v) { return typeof v==='number' ? Math.abs(v).toFixed(2) : v; };
  var driftFmt = function(v) { return v > 0 ? '1/'+Math.round(1/v) : '-'; };

  var h = '';
  h += '<div class="summary-card"><div class="label">'+L('v3_max_disp')+' X</div><div class="value">'+fmt(mv.disp_x)+' mm</div><div class="detail">Node '+mv.disp_x_node+'</div></div>';
  h += '<div class="summary-card"><div class="label">'+L('v3_max_disp')+' Y</div><div class="value">'+fmt(mv.disp_y)+' mm</div><div class="detail">Node '+mv.disp_y_node+'</div></div>';
  h += '<div class="summary-card"><div class="label">'+L('v3_max_disp')+' Z</div><div class="value">'+fmt(mv.disp_z)+' mm</div><div class="detail">Node '+mv.disp_z_node+'</div></div>';
  h += '<div class="summary-card"><div class="label">'+L('v3_max_drift')+' X</div><div class="value">'+driftFmt(mv.drift_x)+'</div><div class="detail">Story '+mv.drift_x_story+'</div></div>';
  h += '<div class="summary-card"><div class="label">'+L('v3_max_drift')+' Y</div><div class="value">'+driftFmt(mv.drift_y)+'</div><div class="detail">Story '+mv.drift_y_story+'</div></div>';
  h += '<div class="summary-card"><div class="label">'+L('v3_max_moment')+'</div><div class="value">'+fmt(mv.moment)+' kN\u00b7m</div><div class="detail">Elem '+mv.moment_elem+'</div></div>';
  h += '<div class="summary-card"><div class="label">'+L('v3_max_axial')+'</div><div class="value">'+fmt(mv.axial)+' kN</div><div class="detail">Elem '+mv.axial_elem+'</div></div>';
  document.getElementById('summaryCards').innerHTML = h;
}

// ============================================================
// Summary Tab (Envelope + Story Summary + Critical Members)
// ============================================================

function renderSummaryTab() {
  renderModalAnalysis();
  renderEnvelope();
  renderStorySummary();
  renderCriticalMembers();
}

function renderModalAnalysis() {
  var el = document.getElementById('modalAnalysisSection');
  if (!el) return;
  var ma = DATA.modal_analysis;
  if (!ma || !ma.modes || ma.modes.length === 0) {
    el.innerHTML = '';
    return;
  }
  var fp = ma.fundamental_periods || {};
  var h = '<div class="section-title">'+L('sm_modal')+'</div>';

  // 기본 주기 카드
  h += '<div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap">';
  var cards = [
    {label:'T1,x (X-Translation)', val:fp.T1_x_s, color:'#1565c0'},
    {label:'T1,y (Y-Translation)', val:fp.T1_y_s, color:'#2e7d32'},
    {label:'T1,rz (Torsion)', val:fp.T1_rz_s, color:'#e65100'},
  ];
  for (var c = 0; c < cards.length; c++) {
    var cd = cards[c];
    if (!cd.val) continue;
    h += '<div style="background:'+cd.color+';color:#fff;padding:10px 16px;border-radius:6px;min-width:180px;text-align:center">';
    h += '<div style="font-size:11px;opacity:0.85">'+cd.label+'</div>';
    h += '<div style="font-size:22px;font-weight:bold">'+cd.val.toFixed(4)+' s</div>';
    h += '<div style="font-size:11px;opacity:0.85">f = '+(1/cd.val).toFixed(3)+' Hz</div>';
    h += '</div>';
  }
  h += '</div>';

  // diaphragm 자동활성화 경고
  if (ma.diaphragm_auto_enabled) {
    h += '<div style="background:#fff3e0;border-left:4px solid #ff9800;padding:8px 12px;margin-bottom:10px;font-size:12px">';
    h += '<strong>W04:</strong> Rigid diaphragm\uc774 \ubaa8\ub2ec\ud574\uc11d\uc744 \uc704\ud574 \uc790\ub3d9 \ud65c\uc131\ud654\ub418\uc5c8\uc2b5\ub2c8\ub2e4.';
    h += '</div>';
  }

  // 모드 테이블 (질량 참여율 포함)
  var hasMassPart = ma.modes[0] && ma.modes[0].mass_participation;
  h += '<table><thead><tr><th>'+L('sm_mode')+'</th><th>'+L('sm_period')+'</th><th>'+L('sm_freq')+'</th><th>'+L('sm_direction')+'</th>';
  if (hasMassPart) {
    h += '<th>X (%)</th><th>Y (%)</th><th>RZ (%)</th>';
  } else {
    h += '<th>'+L('sm_dominance')+'</th>';
  }
  h += '</tr></thead><tbody>';
  for (var i = 0; i < ma.modes.length; i++) {
    var m = ma.modes[i];
    var dirColor = m.direction === 'TRAN-X' ? '#1565c0' : m.direction === 'TRAN-Y' ? '#2e7d32' : '#e65100';
    var isFundamental = (i < 3) ? 'font-weight:bold;' : '';
    h += '<tr style="'+isFundamental+'">';
    h += '<td>'+m.mode+'</td>';
    h += '<td>'+m.period_s.toFixed(4)+'</td>';
    h += '<td>'+m.frequency_hz.toFixed(4)+'</td>';
    h += '<td style="color:'+dirColor+'">'+m.direction+'</td>';
    if (hasMassPart) {
      var mp = m.mass_participation;
      h += '<td>'+(mp.x_pct||0).toFixed(1)+'</td>';
      h += '<td>'+(mp.y_pct||0).toFixed(1)+'</td>';
      h += '<td>'+(mp.rz_pct||0).toFixed(1)+'</td>';
    } else {
      h += '<td>'+m.dominance_pct.toFixed(1)+'%</td>';
    }
    h += '</tr>';
  }

  // 누적 참여질량 행
  var cum = ma.cumulative_participation;
  if (hasMassPart && cum) {
    var suf = cum.sufficient_90pct;
    var sufColor = suf ? '#2e7d32' : '#c62828';
    var sufLabel = suf ? 'OK (\u226590%)' : 'INSUFFICIENT (<90%)';
    h += '<tr style="font-weight:bold;background:#f5f5f5;border-top:2px solid #333">';
    h += '<td colspan="4" style="text-align:right">'+L('sm_cum_part')+'</td>';
    h += '<td>'+cum.x_pct.toFixed(1)+'</td>';
    h += '<td>'+cum.y_pct.toFixed(1)+'</td>';
    h += '<td>'+cum.rz_pct.toFixed(1)+'</td>';
    h += '</tr>';
    h += '<tr><td colspan="7" style="text-align:right;font-size:11px;color:'+sufColor+'">';
    h += L('sm_sufficiency')+': <strong>'+sufLabel+'</strong></td></tr>';
  }
  h += '</tbody></table>';

  // 질량 정보 노트
  var mi = ma.mass_info;
  if (mi) {
    h += '<div style="background:#f5f5f5;border-radius:4px;padding:8px 12px;margin-top:8px;font-size:11px;color:#555">';
    h += '<strong>'+L('sm_mass_assumptions')+':</strong><ul style="margin:4px 0 0 16px;padding:0">';
    if (mi.notes) {
      for (var n = 0; n < mi.notes.length; n++) {
        h += '<li>'+mi.notes[n]+'</li>';
      }
    }
    if (mi.total_weight_kN) {
      h += '<li>'+L('sm_total_weight')+': '+mi.total_weight_kN.toFixed(1)+' kN</li>';
    }
    h += '</ul></div>';
  }

  // weight_source 표시
  if (ma.weight_source) {
    var wsLabel = ma.weight_source === 'user_provided' ? L('sm_user_weights') : L('sm_estimated_dl');
    h += '<div style="font-size:11px;color:#777;margin-top:4px">'+L('sm_weight_source')+': '+wsLabel+'</div>';
  }

  el.innerHTML = h;
}

function renderEnvelope() {
  var env = SUMMARY.envelope;
  var h = '<div class="section-title">'+L('sm_envelope')+'</div>';
  h += '<table><thead><tr><th>'+L('sm_metric')+'</th><th>'+L('sm_max_value')+'</th><th>'+L('sm_unit')+'</th><th>'+L('sm_gov_case')+'</th><th>'+L('sm_location')+'</th></tr></thead><tbody>';
  for (var i = 0; i < env.length; i++) {
    var e = env[i];
    var valStr = e.value.toFixed(4);
    if (e.metric.indexOf('Drift') >= 0 && e.value > 0) {
      valStr = e.value.toFixed(6) + ' (1/' + Math.round(1/e.value) + ')';
    } else if (e.value >= 1) {
      valStr = e.value.toFixed(2);
    }
    h += '<tr><td>'+e.metric+'</td><td>'+valStr+'</td><td>'+e.unit+'</td>';
    h += '<td><span class="gov-case">'+e.governing_case+'</span></td>';
    h += '<td>'+e.location.label+'</td></tr>';
  }
  h += '</tbody></table>';
  document.getElementById('envelopeSection').innerHTML = h;
}

function renderStorySummary() {
  var ss = SUMMARY.story_summary;
  var h = '<div class="section-title">'+L('sm_story_summary')+'</div>';
  h += '<table><thead><tr><th>'+L('dc_story')+'</th><th>'+L('sm_height')+'</th><th>'+L('sm_cum_h')+'</th>';
  h += '<th>Max dx (mm)</th><th>'+L('sm_gov_case')+'</th>';
  h += '<th>Max dy (mm)</th><th>'+L('sm_gov_case')+'</th>';
  h += '<th>Max Drift X</th><th>1/N</th><th>'+L('sm_gov_case')+'</th>';
  h += '<th>Max Drift Y</th><th>1/N</th><th>'+L('sm_gov_case')+'</th>';
  h += '</tr></thead><tbody>';
  for (var i = 0; i < ss.length; i++) {
    var s = ss[i];
    h += '<tr>';
    h += '<td>'+s.story+'F</td>';
    h += '<td>'+s.height_m.toFixed(1)+'</td>';
    h += '<td>'+s.cum_height_m.toFixed(1)+'</td>';
    h += '<td>'+s.max_dx.toFixed(2)+'</td>';
    h += '<td><span class="gov-case">'+s.gov_dx+'</span></td>';
    h += '<td>'+s.max_dy.toFixed(2)+'</td>';
    h += '<td><span class="gov-case">'+s.gov_dy+'</span></td>';
    h += '<td>'+s.max_drift_x.toFixed(6)+'</td>';
    h += '<td>'+(s.max_drift_x > 0 ? Math.round(1/s.max_drift_x) : '-')+'</td>';
    h += '<td><span class="gov-case">'+s.gov_drift_x+'</span></td>';
    h += '<td>'+s.max_drift_y.toFixed(6)+'</td>';
    h += '<td>'+(s.max_drift_y > 0 ? Math.round(1/s.max_drift_y) : '-')+'</td>';
    h += '<td><span class="gov-case">'+s.gov_drift_y+'</span></td>';
    h += '</tr>';
  }
  h += '</tbody></table>';
  document.getElementById('storySummarySection').innerHTML = h;
}

function renderCriticalMembers() {
  var cmData = SUMMARY.critical_members;
  var cm = cmData.items || [];
  if (!cm || cm.length === 0) {
    document.getElementById('criticalMembersSection').innerHTML = '';
    return;
  }
  var sortLabel = (cmData.sort_basis || 'moment').charAt(0).toUpperCase() + (cmData.sort_basis || 'moment').slice(1);
  var h = '<div class="section-title">'+L('sm_critical_members')+' (Top ' + cm.length + ', '+L('sm_sorted_by')+' ' + sortLabel + ')</div>';
  h += '<table><thead><tr><th>'+L('sm_member')+'</th><th>'+L('sm_type')+'</th><th>'+L('dc_story')+'</th>';
  h += '<th>Max N (kN)</th><th>Gov. Case</th>';
  h += '<th>Max M (kN\u00b7m)</th><th>Gov. Case</th>';
  h += '<th>Max V (kN)</th><th>Gov. Case</th>';
  h += '<th>Max T (kN\u00b7m)</th><th>Gov. Case</th>';
  h += '</tr></thead><tbody>';
  for (var i = 0; i < cm.length; i++) {
    var m = cm[i];
    var typeLabel = m.type === 'column' ? 'Col' : (m.type === 'beam_x' ? 'Bm-X' : 'Bm-Y');
    h += '<tr>';
    h += '<td>M'+m.member_id+'</td>';
    h += '<td>'+typeLabel+'</td>';
    h += '<td>'+m.story+'F</td>';
    h += '<td>'+m.max_N.toFixed(1)+'</td>';
    h += '<td><span class="gov-case">'+m.gov_N+'</span></td>';
    h += '<td>'+m.max_M.toFixed(1)+'</td>';
    h += '<td><span class="gov-case">'+m.gov_M+'</span></td>';
    h += '<td>'+m.max_V.toFixed(1)+'</td>';
    h += '<td><span class="gov-case">'+m.gov_V+'</span></td>';
    h += '<td>'+m.max_T.toFixed(2)+'</td>';
    h += '<td><span class="gov-case">'+m.gov_T+'</span></td>';
    h += '</tr>';
  }
  h += '</tbody></table>';
  document.getElementById('criticalMembersSection').innerHTML = h;
}

// ============================================================
// Story Drift Tab
// ============================================================

function renderDriftTab() {
  var cd = DATA.case_data[currentCase];
  var drifts = cd.drifts;
  var limit = parseInt(document.getElementById('driftLimit').value);
  var allowable = 1.0 / limit;

  // Pass/Fail check
  var maxDrift = 0;
  for (var i = 0; i < drifts.length; i++) {
    var d = Math.max(drifts[i].drift_x, drifts[i].drift_y);
    if (d > maxDrift) maxDrift = d;
  }
  var pass = maxDrift <= allowable;
  var banner = document.getElementById('driftBanner');
  if (maxDrift > 0) {
    banner.className = pass ? 'drift-pass' : 'drift-fail';
    banner.innerHTML = '<strong>' + (pass?L('dr_pass'):L('dr_fail')) + '</strong> \u2014 '+L('dr_max_drift')+' 1/' + Math.round(1/maxDrift) + (pass?' \u2264 ':' > ') + '1/' + limit;
  } else {
    banner.className = 'drift-pass';
    banner.innerHTML = '<strong>'+L('dr_pass')+'</strong> \u2014 '+L('dr_no_lateral');
  }

  // Bar chart
  var stories = [], driftX = [], driftY = [];
  for (var i = 0; i < drifts.length; i++) {
    stories.push('Story ' + drifts[i].story);
    driftX.push(drifts[i].drift_x);
    driftY.push(drifts[i].drift_y);
  }
  var trace1 = {y:stories, x:driftX, type:'bar', orientation:'h', name:'Drift X', marker:{color:'#2196F3'}};
  var trace2 = {y:stories, x:driftY, type:'bar', orientation:'h', name:'Drift Y', marker:{color:'#4CAF50'}};
  Plotly.newPlot('driftChart', [trace1, trace2], {
    barmode:'group',
    height: 40*drifts.length + 120,
    margin:{l:80,r:30,t:30,b:50},
    xaxis:{title:L('dr_drift_ratio')},
    shapes:[{type:'line', x0:allowable, x1:allowable, y0:-0.5, y1:drifts.length-0.5,
             line:{color:'red', width:2, dash:'dash'}}],
    annotations:[{x:allowable, y:drifts.length-0.5, text:'1/'+limit,
                  showarrow:false, yshift:10, font:{color:'red'}}]
  }, {responsive:true});

  // Table
  var t = '<table><thead><tr><th>'+L('dc_story')+'</th><th>'+L('sm_height')+'</th><th>Drift X</th><th>1/Drift X</th><th>Drift Y</th><th>1/Drift Y</th><th>'+L('dr_check')+'</th></tr></thead><tbody>';
  for (var i = 0; i < drifts.length; i++) {
    var d = drifts[i];
    var mxD = Math.max(d.drift_x, d.drift_y);
    var ok = mxD <= allowable;
    var clr = ok ? '#4CAF50' : '#f44336';
    t += '<tr>';
    t += '<td>'+d.story+'</td>';
    t += '<td>'+d.height_m.toFixed(1)+'</td>';
    t += '<td>'+d.drift_x.toFixed(6)+'</td>';
    t += '<td>'+(d.drift_x>0 ? Math.round(1/d.drift_x) : '-')+'</td>';
    t += '<td>'+d.drift_y.toFixed(6)+'</td>';
    t += '<td>'+(d.drift_y>0 ? Math.round(1/d.drift_y) : '-')+'</td>';
    t += '<td style="color:'+clr+';font-weight:600">'+(ok?'OK':'NG')+'</td>';
    t += '</tr>';
  }
  t += '</tbody></table>';
  document.getElementById('driftTable').innerHTML = t;
}

// ============================================================
// Reactions Tab (enhanced with equilibrium check)
// ============================================================

function renderReactionsTab() {
  var reactions = DATA.case_data[currentCase].reactions;
  var h = '<div class="section-title">'+L('rx_support')+' \u2014 ' + currentCase + '</div>';
  h += '<table><thead><tr><th>Node</th><th>X (m)</th><th>Y (m)</th><th>RX (kN)</th><th>RY (kN)</th><th>RZ (kN)</th><th>MX (kN\u00b7m)</th><th>MY (kN\u00b7m)</th><th>MZ (kN\u00b7m)</th></tr></thead><tbody>';
  var sRX=0, sRY=0, sRZ=0;
  for (var i = 0; i < reactions.length; i++) {
    var r = reactions[i];
    h += '<tr>';
    h += '<td>'+r.node+'</td>';
    h += '<td>'+r.x_m.toFixed(2)+'</td>';
    h += '<td>'+r.y_m.toFixed(2)+'</td>';
    h += '<td>'+r.RX_kN.toFixed(2)+'</td>';
    h += '<td>'+r.RY_kN.toFixed(2)+'</td>';
    h += '<td>'+r.RZ_kN.toFixed(2)+'</td>';
    h += '<td>'+r.MX_kNm.toFixed(2)+'</td>';
    h += '<td>'+r.MY_kNm.toFixed(2)+'</td>';
    h += '<td>'+r.MZ_kNm.toFixed(2)+'</td>';
    h += '</tr>';
    sRX += r.RX_kN; sRY += r.RY_kN; sRZ += r.RZ_kN;
  }
  h += '<tr style="font-weight:700;background:#f5f5f5">';
  h += '<td colspan="3">'+L('rx_sum')+'</td>';
  h += '<td>'+sRX.toFixed(2)+'</td><td>'+sRY.toFixed(2)+'</td><td>'+sRZ.toFixed(2)+'</td>';
  h += '<td colspan="3"></td></tr>';
  h += '</tbody></table>';

  // Equilibrium Check (residual-based)
  var eqData = SUMMARY.equilibrium;
  var eq = eqData.checks || [];
  var tol = eqData.tolerance_kN || 0.1;
  h += '<div class="section-title">'+L('rx_eq_check')+' ('+L('rx_tolerance')+': ' + tol + ' kN)</div>';
  h += '<table><thead><tr><th>'+L('rx_case')+'</th>';
  h += '<th>Applied FX</th><th>\u03a3RX</th><th>Res. FX</th>';
  h += '<th>Applied FY</th><th>\u03a3RY</th><th>Res. FY</th>';
  h += '<th>Applied FZ</th><th>\u03a3RZ</th><th>Res. FZ</th>';
  h += '<th>Status</th></tr></thead><tbody>';
  var fmtEq = function(v) { return v !== null ? v.toFixed(2) : '-'; };
  for (var i = 0; i < eq.length; i++) {
    var e = eq[i];
    var cls = e.status === 'OK' ? 'eq-ok' : (e.status.indexOf('FAIL') >= 0 ? 'eq-warn' : '');
    h += '<tr>';
    h += '<td>'+e["case"]+'</td>';
    h += '<td>'+fmtEq(e.applied_fx)+'</td><td>'+fmtEq(e.reaction_fx)+'</td><td>'+fmtEq(e.residual_fx)+'</td>';
    h += '<td>'+fmtEq(e.applied_fy)+'</td><td>'+fmtEq(e.reaction_fy)+'</td><td>'+fmtEq(e.residual_fy)+'</td>';
    h += '<td>'+fmtEq(e.applied_fz)+'</td><td>'+fmtEq(e.reaction_fz)+'</td><td>'+fmtEq(e.residual_fz)+'</td>';
    h += '<td class="'+cls+'">'+e.status+'</td>';
    h += '</tr>';
  }
  h += '</tbody></table>';

  document.getElementById('reactionsTable').innerHTML = h;
}

// ============================================================
// Model Info Tab (enhanced with analysis settings + limitations)
// ============================================================

function renderModelTab() {
  var mi = DATA.model_info;
  var ms = SUMMARY.model_settings;
  var h = '';

  // Analysis Settings
  h += '<div class="model-section"><h3>'+L('mi_analysis_settings')+'</h3><table>';
  h += '<tr><td>'+L('mi_analysis_type')+'</td><td>'+ms.analysis_type+'</td></tr>';
  h += '<tr><td>'+L('mi_element_type')+'</td><td>'+ms.element_type+'</td></tr>';
  h += '<tr><td>'+L('mi_dof')+'</td><td>'+ms.dof+'</td></tr>';
  h += '<tr><td>'+L('mi_geom_nl')+'</td><td>'+ms.geometric_nonlinearity+'</td></tr>';
  if (ms.solver_algorithm) {
    h += '<tr><td>'+L('mi_solver')+'</td><td>'+ms.solver_algorithm+(ms.fallback_used ? ' ('+L('wn_fallback')+')' : '')+'</td></tr>';
    h += '<tr><td>'+L('mi_load_steps')+'</td><td>'+ms.n_load_steps+'</td></tr>';
  }
  h += '<tr><td>'+L('mi_rigid_diap')+'</td><td>'+(ms.rigid_diaphragm ? L('mi_applied') : L('mi_not_applied'))+'</td></tr>';
  h += '<tr><td>'+L('mi_mesh')+'</td><td>'+ms.mesh_subdivision+' '+L('mi_elem_per_member')+'</td></tr>';
  h += '<tr><td>'+L('mi_support_cond')+'</td><td>'+ms.support_type+'</td></tr>';
  h += '</table></div>';

  // Building Geometry
  h += '<div class="model-section"><h3>'+L('mi_building_geom')+'</h3><table>';
  h += '<tr><td>'+L('mi_stories')+'</td><td>'+mi.num_stories+' ('+mi.stories.join(', ')+' m)</td></tr>';
  if (mi.num_bays_x > 0 || mi.num_bays_y > 0) {
    h += '<tr><td>'+L('mi_bays_x')+'</td><td>'+mi.num_bays_x+' ('+mi.bays_x.join(', ')+' m)</td></tr>';
    h += '<tr><td>'+L('mi_bays_y')+'</td><td>'+mi.num_bays_y+' ('+mi.bays_y.join(', ')+' m)</td></tr>';
  } else {
    h += '<tr><td>Nodes</td><td>'+mi.num_nodes+'</td></tr>';
    h += '<tr><td>Elements</td><td>'+mi.num_elements+' ('+mi.num_members+' members)</td></tr>';
  }
  h += '<tr><td>'+L('mi_total_height')+'</td><td>'+mi.total_height.toFixed(1)+' m</td></tr>';
  h += '<tr><td>'+L('mi_total_width_x')+'</td><td>'+mi.total_width_x.toFixed(1)+' m</td></tr>';
  h += '<tr><td>'+L('mi_total_width_y')+'</td><td>'+mi.total_width_y.toFixed(1)+' m</td></tr>';
  h += '<tr><td>'+L('mi_supports')+'</td><td>'+mi.supports+'</td></tr>';
  h += '<tr><td>'+L('mi_nodes_elems')+'</td><td>'+mi.num_nodes+' / '+mi.num_members+' / '+mi.num_elements+'</td></tr>';
  h += '</table></div>';

  // Sections
  h += '<div class="model-section"><h3>'+L('mi_sections')+'</h3><table>';
  h += '<tr><td>'+L('mi_column')+'</td><td>'+mi.column_section+'</td></tr>';
  h += '<tr><td>'+L('mi_beam_x')+'</td><td>'+mi.beam_x_section+'</td></tr>';
  h += '<tr><td>'+L('mi_beam_y')+'</td><td>'+mi.beam_y_section+'</td></tr>';
  h += '</table></div>';

  // Material
  h += '<div class="model-section"><h3>'+L('mi_material')+'</h3><table>';
  h += '<tr><td>'+L('mi_name')+'</td><td>'+mi.material_name+'</td></tr>';
  h += '<tr><td>E</td><td>'+mi.E_MPa.toFixed(0)+' MPa</td></tr>';
  h += '<tr><td>G</td><td>'+mi.G_MPa.toFixed(0)+' MPa</td></tr>';
  h += '</table></div>';

  // Load Cases
  h += '<div class="model-section"><h3>'+L('mi_load_cases')+'</h3><ul>';
  for (var i = 0; i < DATA.case_names.length; i++) h += '<li>'+DATA.case_names[i]+'</li>';
  h += '</ul>';
  if (DATA.combo_names.length > 0) {
    h += '<h3 style="margin-top:12px">'+L('mi_load_combos')+'</h3><ul>';
    for (var i = 0; i < DATA.combo_names.length; i++) h += '<li>'+DATA.combo_names[i]+'</li>';
    h += '</ul>';
  }
  h += '</div>';

  // Model Limitations
  var w = SUMMARY.warnings;
  h += '<div class="model-section"><h3>'+L('mi_model_limits')+'</h3><table>';
  h += '<thead><tr><th>'+L('mi_code')+'</th><th>'+L('mi_description')+'</th><th>'+L('mi_severity_col')+'</th></tr></thead><tbody>';
  for (var i = 0; i < w.length; i++) {
    var wText = (LANG==='ko' && w[i].text_ko) ? w[i].text_ko : w[i].text;
    h += '<tr><td>'+w[i].code+'</td><td style="text-align:left">'+wText+'</td>';
    h += '<td>'+(w[i].severity === 'warning' ? '<span style="color:#e65100;font-weight:600">'+L('mi_warning')+'</span>' : L('mi_info'))+'</td></tr>';
  }
  h += '</tbody></table></div>';

  document.getElementById('modelInfo').innerHTML = h;
}

// ============================================================
// Input Assumptions Tab
// ============================================================

var _assumptionsRendered = false;
function renderAssumptionsTab() {
  if (_assumptionsRendered) return;
  var a = DATA.assumptions;
  if (!a) {
    var el = document.getElementById('assumptionsOverview');
    if (el) el.innerHTML = '<p style="color:#999">\ub370\uc774\ud130 \uc5c6\uc74c</p>';
    return;
  }
  _renderModelOverview(a);
  _renderLoadsInfo(a);
  _renderGroupedWarnings(a);
  _renderOccupancyTrace(a.occupancy_trace || []);
  _renderAsmpDetail(a.assumptions || []);
  _assumptionsRendered = true;
}

// Source badge (Korean labels)
var _SRC_KO = {user_input:'\uc0ac\uc6a9\uc790 \uc785\ub825', 'default':'\uae30\ubcf8\uac12', ifc_detected:'IFC \uac10\uc9c0',
  calculated:'\uc790\ub3d9 \uacc4\uc0b0', fallback:'\ub300\uccb4\uac12', fallback_unmapped:'\ub300\uccb4\uac12', fallback_db_miss:'\ub300\uccb4\uac12', db_lookup:'DB \uc870\ud68c'};
function srcBadge(s) { return '<span class="assumption-badge '+s+'">' + (_SRC_KO[s]||s) + '</span>'; }

function _renderModelOverview(a) {
  var ov = a.model_overview;
  if (!ov) return;
  var h = '<div class="section-title" style="margin-top:0">\ubaa8\ub378 \uac00\uc815 \uc694\uc57d <span style="font-size:12px;color:#888;font-weight:400">Model Assumption Summary</span></div>';
  h += '<div class="model-section" style="margin-bottom:8px"><table style="margin-top:0;box-shadow:none">';
  // Row helper
  function row(label, val, badge) {
    h += '<tr><td style="text-align:left;width:160px;color:#555;font-weight:600">' + label + '</td>';
    h += '<td style="text-align:left">' + val;
    if (badge) h += ' ' + srcBadge(badge);
    h += '</td></tr>';
  }
  // Input source
  var srcLabel = ov.input_source === 'ifc+json' ? 'IFC + JSON' : 'JSON';
  row('\uc785\ub825 \ubc29\uc2dd', srcLabel);
  // Geometry
  row('\uad6c\uc870 \uaddc\ubaa8', ov.num_stories + '\uce35 / \uc804\uccb4 \ub192\uc774 ' + ov.total_height_m.toFixed(1) + 'm');
  row('\uce35\uace0', ov.height_desc);
  if (ov.bays_x_count > 0 || ov.bays_y_count > 0) {
    row('X\ubc29\ud5a5 \uacbd\uac04', ov.bays_x_count + '\uacbd\uac04 (' + ov.bays_x_widths.join(', ') + ' m)');
    row('Y\ubc29\ud5a5 \uacbd\uac04', ov.bays_y_count + '\uacbd\uac04 (' + ov.bays_y_widths.join(', ') + ' m)');
  } else {
    row('\ub178\ub4dc/\uc694\uc18c', (ov.num_nodes||'?') + ' nodes / ' + (ov.num_elements||'?') + ' elements');
  }
  row('\ubc14\ub2e5 \uba74\uc801', ov.floor_area_m2.toFixed(1) + ' m\u00b2');
  // Usage
  row('\uce35\ubcc4 \uc6a9\ub3c4', ov.usage_summary.join(' / '));
  h += '</table></div>';
  // Structural
  h += '<div class="model-section" style="margin-bottom:8px"><table style="margin-top:0;box-shadow:none">';
  row('\uae30\ub465 \ub2e8\uba74', ov.column_section, ov.column_source);
  row('\ubcf4 \ub2e8\uba74 (X)', ov.beam_x_section, ov.beam_source);
  row('\ubcf4 \ub2e8\uba74 (Y)', ov.beam_y_section, ov.beam_source);
  row('\uc7ac\ub8cc', ov.material_name);
  row('\uc9c0\uc810 \uc870\uac74', ov.supports === 'fixed' ? '\uace0\uc815\ub2e8' : '\ud540');
  row('Rigid Diaphragm', ov.rigid_diaphragm ? 'ON' : 'OFF');
  row('\uc2ac\ub798\ube0c \uac15\uc131', ov.slab_stiffness_modeled ? '\ubaa8\ub378\ub9c1 \ud3ec\ud568' : '\ubbf8\ubc18\uc601 (\ud558\uc911\ub9cc)');
  row('\ubcbd\uccb4 \uac15\uc131', ov.wall_stiffness_modeled ? '\ubaa8\ub378\ub9c1 \ud3ec\ud568' : '\ubbf8\ubc18\uc601');
  h += '</table></div>';
  // Environment
  h += '<div class="model-section" style="margin-bottom:8px"><table style="margin-top:0;box-shadow:none">';
  row('\uc9c0\uc5ed', ov.region);
  row('\uc9c0\ubc18 \ubd84\ub958', ov.site_class);
  row('\uc911\uc694\ub3c4', ov.importance_label);
  row('\ub0b4\uc9c4\uc2dc\uc2a4\ud15c', ov.seismic_system_label);
  row('\ud48d\ub178\ucd9c \ubc94\uc8fc', ov.exposure_label);
  h += '</table></div>';
  document.getElementById('assumptionsOverview').innerHTML = h;
}

function _renderLoadsInfo(a) {
  var li = a.loads_info;
  if (!li) return;
  var el = document.getElementById('assumptionsOverview');
  var h = '<div class="section-title">\ud558\uc911 \uc0dd\uc131 \uc694\uc57d <span style="font-size:12px;color:#888;font-weight:400">Load Generation Summary</span></div>';
  h += '<div class="asmp-overview">';
  h += '<div class="asmp-stat"><div class="num" style="color:#1a237e">' + li.load_cases.length + '</div><div class="lbl">\ud558\uc911 \ucf00\uc774\uc2a4</div></div>';
  h += '<div class="asmp-stat"><div class="num" style="color:#1a237e">' + li.num_combinations + '</div><div class="lbl">\ud558\uc911 \uc870\ud569</div></div>';
  h += '<div class="asmp-stat"><div class="num" style="color:#333;font-size:18px">' + li.total_weight_kN.toFixed(0) + '</div><div class="lbl">\ucd1d \uc911\ub7c9 (kN)</div></div>';
  var eqBadge = li.has_seismic ? '<span style="color:#2e7d32;font-weight:600">\u2713 \uc0dd\uc131\ub428</span>' : '<span style="color:#c62828;font-weight:600">\u2717 \ubbf8\uc0dd\uc131</span>';
  var wBadge = li.has_wind ? '<span style="color:#2e7d32;font-weight:600">\u2713 \uc0dd\uc131\ub428</span>' : '<span style="color:#c62828;font-weight:600">\u2717 \ubbf8\uc0dd\uc131</span>';
  h += '<div class="asmp-stat"><div class="num" style="font-size:14px">' + eqBadge + '</div><div class="lbl">\uc9c0\uc9c4\ud558\uc911</div></div>';
  h += '<div class="asmp-stat"><div class="num" style="font-size:14px">' + wBadge + '</div><div class="lbl">\ud48d\ud558\uc911</div></div>';
  h += '<div class="asmp-stat"><div class="num" style="font-size:14px">' + (li.auto_combinations ? 'ON' : 'OFF') + '</div><div class="lbl">\uc790\ub3d9 \uc870\ud569</div></div>';
  h += '</div>';
  // Load case list
  h += '<div style="font-size:13px;color:#555;margin-bottom:16px">\uc0dd\uc131\ub41c \ucf00\uc774\uc2a4: <strong>' + li.load_cases.join(', ') + '</strong></div>';
  el.innerHTML += h;
}

function _renderGroupedWarnings(a) {
  var gw = a.grouped_warnings || [];
  var allW = a.warnings || [];
  if (!gw.length) {
    document.getElementById('assumptionsWarnings').innerHTML = '';
    return;
  }
  var h = '<div class="section-title">\uacbd\uace0 \ubc0f \uc54c\ub9bc <span style="font-size:12px;color:#888;font-weight:400">Warnings (' + allW.length + '\uac74)</span></div>';
  // Grouped summary
  for (var i = 0; i < gw.length; i++) {
    var g = gw[i];
    h += '<div class="asmp-warn-item severity-' + g.severity + '">';
    h += '<strong>[' + g.code + ']</strong> ';
    h += g.summary_ko;
    if (g.count > 1) h += ' <span style="color:#888;font-size:12px">(' + g.count + '\uac74)</span>';
    h += '<br><span style="color:#888;font-size:12px">' + g.summary_en + '</span>';
    h += '</div>';
  }
  // Collapsible detail
  if (allW.length > gw.length) {
    h += '<details style="margin-top:8px"><summary style="cursor:pointer;font-size:13px;color:#1a237e;font-weight:500">\uce35\ubcc4 \uc0c1\uc138 \ubcf4\uae30 (' + allW.length + '\uac74)</summary>';
    for (var j = 0; j < allW.length; j++) {
      var w = allW[j];
      h += '<div class="asmp-warn-item severity-' + w.severity + '" style="margin-left:16px;margin-top:4px">';
      h += '<strong>[' + w.code + ']</strong> ' + w.message_ko;
      h += '</div>';
    }
    h += '</details>';
  }
  document.getElementById('assumptionsWarnings').innerHTML = h;
}

function _renderOccupancyTrace(traces) {
  if (!traces.length) {
    document.getElementById('occupancyTrace').innerHTML = '';
    return;
  }
  var h = '<div class="section-title">\uc6a9\ub3c4 \ub9e4\ud551 \ucd94\uc801 <span style="font-size:12px;color:#888;font-weight:400">Occupancy Mapping Trace (Live Load)</span></div>';
  h += '<table><tr><th style="text-align:left">\uce35</th><th style="text-align:left">\uc0ac\uc6a9\uc790 \uc785\ub825</th>';
  h += '<th></th><th style="text-align:left">\ub0b4\ubd80 \ud0a4</th>';
  h += '<th></th><th style="text-align:left">DB \ud0a4</th>';
  h += '<th style="text-align:right">\ud65c\ud558\uc911</th><th style="text-align:left">\ucd9c\ucc98</th><th style="text-align:left">\uae30\uc900</th></tr>';
  for (var i = 0; i < traces.length; i++) {
    var t = traces[i];
    var dbKey = t.db_primary_key || '-';
    if (t.db_secondary_key) dbKey += ' / ' + t.db_secondary_key;
    h += '<tr>';
    h += '<td style="text-align:left">' + t.story + '\uce35</td>';
    h += '<td style="text-align:left;font-weight:500">' + t.user_input + '</td>';
    h += '<td class="trace-arrow">&rarr;</td>';
    h += '<td style="text-align:left"><code>' + t.internal_key + '</code></td>';
    h += '<td class="trace-arrow">&rarr;</td>';
    h += '<td style="text-align:left"><code>' + dbKey + '</code></td>';
    h += '<td style="text-align:right;font-weight:600">' + t.live_load_kNm2 + ' kN/m\u00b2</td>';
    h += '<td style="text-align:left">' + srcBadge(t.load_source) + '</td>';
    h += '<td style="text-align:left;font-size:11px;color:#888">' + t.kds_ref + '</td>';
    h += '</tr>';
  }
  h += '</table>';
  document.getElementById('occupancyTrace').innerHTML = h;
}

function _renderAsmpDetail(items) {
  if (!items.length) {
    document.getElementById('assumptionsDetail').innerHTML = '';
    return;
  }
  var groups = {};
  var order = [];
  for (var i = 0; i < items.length; i++) {
    var cat = items[i].category;
    if (!groups[cat]) { groups[cat] = []; order.push(cat); }
    groups[cat].push(items[i]);
  }
  var catLabels = {geometry:'\uae30\ud558 (Geometry)', sections:'\ub2e8\uba74 (Sections)', material:'\uc7ac\ub8cc (Material)',
    loads:'\ud558\uc911 (Loads)', seismic:'\ub0b4\uc9c4 (Seismic)', wind:'\ud48d\ud558\uc911 (Wind)', environment:'\ud658\uacbd (Environment)'};

  var h = '<div class="section-title">\ud30c\ub77c\ubbf8\ud130 \uc0c1\uc138 <span style="font-size:12px;color:#888;font-weight:400">Parameter Detail</span></div>';
  for (var g = 0; g < order.length; g++) {
    var cat = order[g];
    var catItems = groups[cat];
    h += '<div class="asmp-group-title">' + (catLabels[cat] || cat) + ' (' + catItems.length + ')</div>';
    h += '<table><tr><th style="text-align:left">\ud30c\ub77c\ubbf8\ud130</th><th style="text-align:left">\uac12</th>';
    h += '<th style="text-align:left">\ucd9c\ucc98</th><th style="text-align:left">\uae30\ubcf8\uac12</th><th style="text-align:left">\ube44\uace0</th></tr>';
    for (var j = 0; j < catItems.length; j++) {
      var it = catItems[j];
      var val = it.value;
      if (val === null || val === undefined) val = '-';
      if (typeof val === 'boolean') val = val ? 'ON' : 'OFF';
      var def = it.default_value;
      if (def === null || def === undefined) def = '-';
      if (typeof def === 'boolean') def = def ? 'ON' : 'OFF';
      h += '<tr>';
      h += '<td style="text-align:left"><code>' + it.param + '</code></td>';
      h += '<td style="text-align:left">' + val + '</td>';
      h += '<td style="text-align:left">' + srcBadge(it.source) + '</td>';
      h += '<td style="text-align:left;color:#888">' + def + '</td>';
      h += '<td style="text-align:left;font-size:12px;color:#666">' + (it.note || '') + '</td>';
      h += '</tr>';
    }
    h += '</table>';
  }
  document.getElementById('assumptionsDetail').innerHTML = h;
}

// ============================================================
// Export
// ============================================================

function exportCSV(type) {
  var cd = DATA.case_data[currentCase];
  var csv = '';
  if (type === 'displacements') {
    csv = 'Node,X_m,Y_m,Z_m,dx_mm,dy_mm,dz_mm\n';
    for (var nid in DATA.node_map) {
      var p = DATA.node_map[nid], d = cd.disp[nid]||[0,0,0];
      csv += nid+','+p[0]+','+p[1]+','+p[2]+','+d[0]+','+d[1]+','+d[2]+'\n';
    }
  } else if (type === 'reactions') {
    csv = 'Node,X_m,Y_m,RX_kN,RY_kN,RZ_kN,MX_kNm,MY_kNm,MZ_kNm\n';
    for (var i = 0; i < cd.reactions.length; i++) {
      var r = cd.reactions[i];
      csv += r.node+','+r.x_m+','+r.y_m+','+r.RX_kN+','+r.RY_kN+','+r.RZ_kN+','+r.MX_kNm+','+r.MY_kNm+','+r.MZ_kNm+'\n';
    }
  } else if (type === 'drifts') {
    csv = 'Story,Height_m,Drift_X,Drift_Y,Drift_Resultant\n';
    for (var i = 0; i < cd.drifts.length; i++) {
      var d = cd.drifts[i];
      csv += d.story+','+d.height_m+','+d.drift_x+','+d.drift_y+','+d.drift_resultant+'\n';
    }
  }
  var blob = new Blob([csv], {type:'text/csv'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = currentCase+'_'+type+'.csv';
  a.click();
}

function exportPNG() {
  Plotly.downloadImage('plot3d', {format:'png', width:1920, height:1080, filename:'3d_frame_'+currentCase});
}

// ============================================================
// Design Check Tab
// ============================================================

var _dcRendered = false;
function renderDesignCheckTab() {
  if (_dcRendered) return;
  var dc = DATA.design_check;
  if (!dc) { document.getElementById('dcBanner').innerHTML = '<p>'+L('dc_not_available')+'</p>'; return; }
  _dcRendered = true;

  // Overall banner
  var overall = dc.overall_status;
  var bannerColor = overall === 'OK' ? '#27ae60' : '#e74c3c';
  var bannerHtml = '<div style="background:'+bannerColor+';color:#fff;padding:16px 24px;border-radius:8px;margin-bottom:20px;font-size:18px;font-weight:bold">'
    + L('dc_title') + ': ' + overall + '</div>';

  // Summary cards
  if (dc.summary) {
    var s = dc.summary;
    bannerHtml += '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">';
    if (s.max_drift_ratio !== null && s.max_drift_ratio !== undefined) {
      var drColor = s.max_drift_ratio <= 0.7 ? '#27ae60' : (s.max_drift_ratio <= 1.0 ? '#f39c12' : '#e74c3c');
      bannerHtml += '<div style="background:#f8f9fa;border-left:4px solid '+drColor+';padding:12px 16px;border-radius:4px;min-width:180px">'
        + '<div style="font-size:12px;color:#666">'+L('dc_max_drift')+'</div>'
        + '<div style="font-size:24px;font-weight:bold;color:'+drColor+'">'+s.max_drift_ratio.toFixed(3)+'</div></div>';
    }
    if (s.max_interaction_ratio !== null && s.max_interaction_ratio !== undefined) {
      var irColor = s.max_interaction_ratio <= 0.7 ? '#27ae60' : (s.max_interaction_ratio <= 1.0 ? '#f39c12' : '#e74c3c');
      bannerHtml += '<div style="background:#f8f9fa;border-left:4px solid '+irColor+';padding:12px 16px;border-radius:4px;min-width:180px">'
        + '<div style="font-size:12px;color:#666">'+L('dc_max_interact')+'</div>'
        + '<div style="font-size:24px;font-weight:bold;color:'+irColor+'">'+s.max_interaction_ratio.toFixed(3)+'</div></div>';
    }
    if (s.ng_members > 0) {
      bannerHtml += '<div style="background:#f8f9fa;border-left:4px solid #e74c3c;padding:12px 16px;border-radius:4px;min-width:180px">'
        + '<div style="font-size:12px;color:#666">'+L('dc_ng_members')+'</div>'
        + '<div style="font-size:24px;font-weight:bold;color:#e74c3c">'+s.ng_members+'</div></div>';
    }
    bannerHtml += '</div>';
  }
  document.getElementById('dcBanner').innerHTML = bannerHtml;

  // Drift check section
  var driftHtml = '';
  if (dc.drift_check) {
    var dchk = dc.drift_check;
    driftHtml += '<h3 style="margin:16px 0 8px">'+L('dc_drift_check')+' (' + dchk.code_ref + ')</h3>';
    driftHtml += '<p style="margin-bottom:8px;color:#666">Cd='+dchk.Cd+', IE='+dchk.IE+', '+L('dc_allowable')+'='+dchk.allowable+' ('+L('dc_importance')+': '+dchk.importance+')</p>';
    if (dchk.checks && dchk.checks.length > 0) {
      driftHtml += '<table style="width:100%;border-collapse:collapse;font-size:13px"><thead><tr style="background:#f0f0f0">'
        + '<th style="padding:6px 8px;text-align:left">'+L('dc_story')+'</th><th>'+L('dc_dir')+'</th><th>'+L('dc_combo')+'</th>'
        + '<th>'+L('dc_elastic')+'</th><th>'+L('dc_inelastic')+'</th><th>'+L('dc_allowable')+'</th><th>'+L('dc_ratio')+'</th><th>'+L('dc_status')+'</th></tr></thead><tbody>';
      for (var i = 0; i < dchk.checks.length; i++) {
        var c = dchk.checks[i];
        var rColor = c.ratio <= 0.7 ? '#27ae60' : (c.ratio <= 1.0 ? '#f39c12' : '#e74c3c');
        var barW = Math.min(c.ratio * 100, 100);
        driftHtml += '<tr style="border-bottom:1px solid #eee">'
          + '<td style="padding:6px 8px">'+c.story+'</td>'
          + '<td style="text-align:center">'+c.direction+'</td>'
          + '<td>'+c.combo+'</td>'
          + '<td style="text-align:right">'+c.elastic_drift.toFixed(6)+'</td>'
          + '<td style="text-align:right">'+c.inelastic_drift.toFixed(6)+'</td>'
          + '<td style="text-align:right">'+c.allowable+'</td>'
          + '<td style="text-align:right"><div style="display:flex;align-items:center;gap:4px;justify-content:flex-end">'
          + '<div style="width:60px;height:8px;background:#eee;border-radius:4px;overflow:hidden">'
          + '<div style="width:'+barW+'%;height:100%;background:'+rColor+';border-radius:4px"></div></div>'
          + '<span style="color:'+rColor+';font-weight:bold">'+c.ratio.toFixed(3)+'</span></div></td>'
          + '<td style="text-align:center;font-weight:bold;color:'+(c.status==='OK'?'#27ae60':'#e74c3c')+'">'+c.status+'</td></tr>';
      }
      driftHtml += '</tbody></table>';
    }
  }
  document.getElementById('dcDriftSection').innerHTML = driftHtml;

  // Member check section
  var memberHtml = '';
  if (dc.member_check) {
    var mchk = dc.member_check;
    memberHtml += '<h3 style="margin:24px 0 8px">'+L('dc_member_check')+' (' + mchk.code_ref + ')</h3>';
    memberHtml += '<p style="margin-bottom:8px;color:#666">'+L('dc_total')+': '+mchk.summary.total+', OK: '+mchk.summary.ok+', NG: '+mchk.summary.ng+'</p>';
    // Show top 10 critical members
    var show = mchk.critical_members || mchk.members.slice(0, 10);
    if (show.length > 0) {
      memberHtml += '<table style="width:100%;border-collapse:collapse;font-size:13px"><thead><tr style="background:#f0f0f0">'
        + '<th style="padding:6px 8px;text-align:left">'+L('dc_member')+'</th><th>'+L('dc_type')+'</th><th>'+L('dc_section')+'</th>'
        + '<th>Pu(kN)</th><th>Mux(kNm)</th><th>Muy(kNm)</th>'
        + '<th>'+L('dc_interaction')+'</th><th>'+L('dc_formula')+'</th><th>'+L('dc_status')+'</th></tr></thead><tbody>';
      for (var i = 0; i < show.length; i++) {
        var m = show[i];
        var r = m.ratios.interaction;
        var mColor = r <= 0.7 ? '#27ae60' : (r <= 1.0 ? '#f39c12' : '#e74c3c');
        var barW = Math.min(r * 100, 100);
        memberHtml += '<tr style="border-bottom:1px solid #eee">'
          + '<td style="padding:6px 8px">#'+m.member_id+'</td>'
          + '<td>'+m.type+'</td><td>'+m.section+'</td>'
          + '<td style="text-align:right">'+m.demand.Pu_kN.toFixed(1)+'</td>'
          + '<td style="text-align:right">'+m.demand.Mux_kNm.toFixed(1)+'</td>'
          + '<td style="text-align:right">'+m.demand.Muy_kNm.toFixed(1)+'</td>'
          + '<td style="text-align:right"><div style="display:flex;align-items:center;gap:4px;justify-content:flex-end">'
          + '<div style="width:80px;height:8px;background:#eee;border-radius:4px;overflow:hidden">'
          + '<div style="width:'+barW+'%;height:100%;background:'+mColor+';border-radius:4px"></div></div>'
          + '<span style="color:'+mColor+';font-weight:bold">'+r.toFixed(3)+'</span></div></td>'
          + '<td style="text-align:center;font-size:11px">'+m.ratios.formula+'</td>'
          + '<td style="text-align:center;font-weight:bold;color:'+(m.status==='OK'?'#27ae60':'#e74c3c')+'">'+m.status+'</td></tr>';
      }
      memberHtml += '</tbody></table>';
    }
    // Assumptions
    if (mchk.assumptions) {
      memberHtml += '<div style="margin-top:16px;padding:12px;background:#fff3cd;border-radius:6px;font-size:12px">'
        + '<strong>'+L('dc_assumptions')+':</strong><ul style="margin:4px 0 0 16px">';
      for (var i = 0; i < mchk.assumptions.length; i++) {
        memberHtml += '<li>'+mchk.assumptions[i]+'</li>';
      }
      memberHtml += '</ul></div>';
    }
  }
  document.getElementById('dcMemberSection').innerHTML = memberHtml;

  // Critical issues
  var issueHtml = '';
  if (dc.critical_issues && dc.critical_issues.length > 0) {
    issueHtml += '<h3 style="margin:24px 0 8px;color:#e74c3c">'+L('dc_critical_issues')+'</h3><ul style="margin-left:16px">';
    for (var i = 0; i < dc.critical_issues.length; i++) {
      var ci = dc.critical_issues[i];
      issueHtml += '<li style="margin-bottom:6px"><strong>['+ci.code_ref+']</strong> '+ci.description+'</li>';
    }
    issueHtml += '</ul>';
  }
  document.getElementById('dcIssuesSection').innerHTML = issueHtml;
}

// ============================================================
// Interpretation Tab
// ============================================================

var _interpRendered = false;
function renderInterpretationTab() {
  if (_interpRendered) return;
  var interp = DATA.interpretation;
  if (!interp || !interp.severity) return;
  _interpRendered = true;

  // 1. Severity Banner
  var sevColors = {safe:'#2e7d32',marginal:'#e65100',moderate:'#c62828',severe:'#4a148c'};
  var sevBg = {safe:'#e8f5e9',marginal:'#fff3e0',moderate:'#ffebee',severe:'#f3e5f5'};
  var h = '<div style="background:'+sevBg[interp.severity]+';border-left:6px solid '+sevColors[interp.severity]+';padding:16px;border-radius:6px;margin-bottom:16px">';
  var sevLabel = (LANG==='ko') ? interp.severity_label.ko : interp.severity_label.en;
  var sevLabelAlt = (LANG==='ko') ? interp.severity_label.en : interp.severity_label.ko;
  h += '<div style="font-size:20px;font-weight:bold;color:'+sevColors[interp.severity]+'">'+sevLabel+' <span style="font-size:14px;font-weight:normal;color:#666">('+sevLabelAlt+')</span></div>';
  var mainSum = (LANG==='ko') ? (interp.summary_ko||'') : (interp.summary_en||'');
  var altSum = (LANG==='ko') ? (interp.summary_en||'') : (interp.summary_ko||'');
  if (mainSum) h += '<div style="margin-top:8px;font-size:13px;color:#333">'+mainSum+'</div>';
  if (altSum) h += '<div style="margin-top:4px;font-size:12px;color:#666">'+altSum+'</div>';
  h += '</div>';
  // Evidence Summary Table
  var dc = DATA.design_check;
  var driftR = (dc && dc.summary) ? (dc.summary.max_drift_ratio || 0) : 0;
  var interR = (dc && dc.summary) ? (dc.summary.max_interaction_ratio || 0) : 0;
  var governing = driftR >= interR ? L('ov_drift') : L('ov_member');
  h += '<div style="margin-top:12px;background:#fff;border:1px solid #e0e0e0;border-radius:6px;overflow:hidden">';
  h += '<table style="width:100%;border-collapse:collapse;font-size:13px">';
  h += '<thead><tr style="background:#f5f5f5"><th colspan="2" style="padding:8px 12px;text-align:left;font-size:12px;color:#666;text-transform:uppercase">'+L('it_evidence')+'</th></tr></thead>';
  h += '<tbody>';
  // Max Drift
  if (dc && dc.drift_check) {
    var dchk = dc.drift_check;
    var critD = dchk.critical || {};
    var drColor = driftR <= 0.7 ? '#2e7d32' : (driftR <= 1.0 ? '#e65100' : '#c62828');
    h += '<tr style="border-bottom:1px solid #f0f0f0"><td style="padding:6px 12px;color:#666;width:160px">'+L('it_max_drift')+'</td>';
    h += '<td style="padding:6px 12px"><span style="font-weight:bold;color:'+drColor+'">'+driftR.toFixed(3)+'</span>';
    if (critD.story) h += ' <span style="color:#888">(Story '+critD.story+', '+critD.direction+'-dir)</span>';
    h += '</td></tr>';
  }
  // Max Interaction
  if (dc && dc.member_check) {
    var mchk = dc.member_check;
    var irColor = interR <= 0.7 ? '#2e7d32' : (interR <= 1.0 ? '#e65100' : '#c62828');
    h += '<tr style="border-bottom:1px solid #f0f0f0"><td style="padding:6px 12px;color:#666">'+L('it_max_interact')+'</td>';
    h += '<td style="padding:6px 12px"><span style="font-weight:bold;color:'+irColor+'">'+interR.toFixed(3)+'</span>';
    if (interp.member_interpretation && interp.member_interpretation.weakest_link) {
      var wk = interp.member_interpretation.weakest_link;
      h += ' <span style="color:#888">('+wk.type+' #'+wk.member_id+', Story '+wk.story+')</span>';
    }
    h += '</td></tr>';
  }
  // Governing Check
  h += '<tr style="border-bottom:1px solid #f0f0f0"><td style="padding:6px 12px;color:#666">'+L('it_governing_check')+'</td>';
  h += '<td style="padding:6px 12px;font-weight:bold">'+governing+' ('+Math.max(driftR,interR).toFixed(3)+')</td></tr>';
  // Governing Failure Mode
  if (interp.member_interpretation && interp.member_interpretation.governing_check) {
    var gm = interp.member_interpretation.governing_check;
    var gmLabels = {axial:'Axial',bending_x:'Major-axis Bending',bending_y:'Minor-axis Bending',shear:'Shear',interaction:'Axial-Bending Interaction'};
    h += '<tr style="border-bottom:1px solid #f0f0f0"><td style="padding:6px 12px;color:#666">'+L('it_governing_mode')+'</td>';
    h += '<td style="padding:6px 12px">'+(gmLabels[gm]||gm)+'</td></tr>';
  }
  // T1 / Ta
  if (interp.modal_interpretation) {
    var mi = interp.modal_interpretation;
    h += '<tr style="border-bottom:1px solid #f0f0f0"><td style="padding:6px 12px;color:#666">'+L('ov_t1_ta')+'</td>';
    h += '<td style="padding:6px 12px">'+(mi.T1_actual_s||0).toFixed(3)+'s / '+(mi.Ta_empirical_s||0).toFixed(3)+'s';
    h += ' <span style="color:#888">('+L('ov_ratio')+' '+(mi.T1_Ta_ratio||0).toFixed(2)+')</span>';
    if (mi.flexibility_flag) h += ' <span style="color:#c62828;font-weight:bold">\u26a0 '+L('it_over_flexible')+'</span>';
    h += '</td></tr>';
    // 1st mode direction
    h += '<tr><td style="padding:6px 12px;color:#666">'+L('ov_first_mode')+'</td>';
    h += '<td style="padding:6px 12px">'+mi.first_mode_direction;
    if (mi.torsion_risk) h += ' <span style="color:#c62828;font-weight:bold">\u26a0 '+L('it_torsion_risk')+'</span>';
    h += '</td></tr>';
  }
  h += '</tbody></table></div>';

  document.getElementById('interpBanner').innerHTML = h;

  // 2. Key Findings
  var fh = '<div class="section-title">'+L('it_findings')+'</div>';
  var sevIcon = {critical:'\u274c',warning:'\u26a0\ufe0f',caution:'\u2139\ufe0f',info:'\u2705'};
  var sevTextColor = {critical:'#c62828',warning:'#e65100',caution:'#1565c0',info:'#2e7d32'};
  if (interp.findings && interp.findings.length > 0) {
    fh += '<div style="display:flex;flex-direction:column;gap:6px">';
    for (var i = 0; i < interp.findings.length; i++) {
      var f = interp.findings[i];
      var icon = sevIcon[f.severity] || '';
      var col = sevTextColor[f.severity] || '#333';
      var fMsg = (LANG==='ko') ? (f.message_ko||f.message_en||'') : (f.message_en||f.message_ko||'');
      fh += '<div style="padding:8px 12px;background:#fafafa;border-radius:4px;border-left:3px solid '+col+'">';
      fh += '<span style="font-size:12px;color:#999">['+f.code+']</span> ';
      fh += '<span style="color:'+col+';font-weight:'+(f.severity==='critical'?'bold':'normal')+'">'+fMsg+'</span>';
      // Inline data evidence
      if (f.data) {
        var parts = [];
        if (f.data.max_ratio !== undefined) parts.push('ratio='+f.data.max_ratio.toFixed(2));
        if (f.data.critical) { var c=f.data.critical; parts.push(c.story+'F '+c.direction+'-dir'); }
        if (f.data.soft_stories) parts.push('story '+f.data.soft_stories);
        if (f.data.weakest) { var w=f.data.weakest; parts.push(w.type+' #'+w.member_id+' ('+w.story+'F, '+w.ratio.toFixed(2)+')'); }
        if (f.data.ng_count) parts.push('NG '+f.data.ng_count);
        if (f.data.T1 !== undefined) parts.push('T1='+f.data.T1.toFixed(3)+'s');
        if (f.data.Ta !== undefined) parts.push('Ta='+f.data.Ta.toFixed(3)+'s');
        if (f.data.governing) parts.push(f.data.governing);
        if (f.data.mode_dir) parts.push('mode='+f.data.mode_dir);
        if (f.data.drift_dir) parts.push('drift='+f.data.drift_dir);
        if (f.data.story !== undefined && f.data.count !== undefined) parts.push(f.data.story+'F: '+f.data.count);
        if (parts.length > 0) {
          fh += ' <span style="font-size:11px;color:#999;margin-left:4px">['+parts.join(', ')+']</span>';
        }
      }
      fh += '</div>';
    }
    fh += '</div>';
  }
  document.getElementById('interpFindings').innerHTML = fh;

  // 3. Diagnosis (NG일 때만)
  var dh = '';
  if (interp.diagnosis) {
    dh = '<div class="section-title">'+L('it_diagnosis')+'</div>';
    dh += '<div style="background:#fff3e0;border-radius:6px;padding:12px;margin-bottom:12px">';
    var diagMain = (LANG==='ko') ? interp.diagnosis.primary_cause_ko : (interp.diagnosis.primary_cause_en||interp.diagnosis.primary_cause_ko);
    var diagAlt = (LANG==='ko') ? (interp.diagnosis.primary_cause_en||'') : (interp.diagnosis.primary_cause_ko||'');
    dh += '<div style="font-size:14px;font-weight:bold;color:#e65100">'+L('it_primary')+': '+diagMain+'</div>';
    if (diagAlt) dh += '<div style="font-size:12px;color:#666;margin-top:2px">'+diagAlt+'</div>';
    var contribArr = (LANG==='ko') ? interp.diagnosis.contributing_factors_ko : (interp.diagnosis.contributing_factors_en||interp.diagnosis.contributing_factors_ko);
    if (contribArr && contribArr.length > 0) {
      dh += '<div style="margin-top:8px;font-size:12px"><strong>'+L('it_contrib')+':</strong><ul style="margin:4px 0 0 16px">';
      for (var j = 0; j < contribArr.length; j++) {
        dh += '<li>'+contribArr[j]+'</li>';
      }
      dh += '</ul></div>';
    }
    dh += '</div>';
  }
  document.getElementById('interpDiagnosis').innerHTML = dh;

  // 4. Suggestions
  var sh = '';
  if (interp.suggestions && interp.suggestions.length > 0) {
    sh = '<div class="section-title">'+L('it_suggestions')+'</div>';
    sh += '<div style="display:flex;flex-direction:column;gap:8px">';
    var impactColors = {high:'#c62828',medium:'#e65100',low:'#1565c0'};
    for (var k = 0; k < interp.suggestions.length; k++) {
      var sg = interp.suggestions[k];
      var ic = impactColors[sg.expected_impact] || '#333';
      var sgMain = (LANG==='ko') ? (sg.message_ko||sg.message_en||'') : (sg.message_en||sg.message_ko||'');
      var sgAlt = (LANG==='ko') ? (sg.message_en||'') : (sg.message_ko||'');
      sh += '<div style="padding:10px 14px;background:#f5f5f5;border-radius:6px;border-left:4px solid '+ic+'">';
      sh += '<div style="display:flex;justify-content:space-between;align-items:center">';
      sh += '<span style="font-weight:bold;font-size:13px">'+sgMain+'</span>';
      sh += '<span style="font-size:11px;color:#999">'+L('it_impact')+': '+sg.expected_impact+'</span>';
      sh += '</div>';
      if (sgAlt) sh += '<div style="font-size:11px;color:#666;margin-top:4px">'+sgAlt+'</div>';
      // Target / Current → Recommended
      if (sg.target && (sg.current || sg.recommended)) {
        var targetLabel = {column:'Column',beam_x:'Beam X',beam_y:'Beam Y',structural_system:'System'}[sg.target] || sg.target;
        sh += '<div style="margin-top:6px;font-size:12px;color:#555;background:#fff;padding:4px 8px;border-radius:4px;border:1px solid #e0e0e0">';
        sh += '<span style="color:#888">'+L('it_target')+': </span><strong>'+targetLabel+'</strong>';
        if (sg.current && sg.recommended) {
          sh += ' &mdash; '+sg.current+' \u2192 '+sg.recommended;
        } else if (sg.recommended) {
          sh += ' \u2192 '+sg.recommended;
        }
        sh += '</div>';
      }
      sh += '</div>';
    }
    sh += '</div>';
  }
  document.getElementById('interpSuggestions').innerHTML = sh;
}

// ============================================================
// Start
// ============================================================
init();
</script>
</body>
</html>"""


# ============================================================
# Paper Mode: 논문용 2색 3D 뷰 HTML 생성
# ============================================================

def _generate_paper_mode_html(data: dict, output_path, deformation_scale: float, paper_case: str | None) -> str:
    """논문용 2색(undeformed 회색 + deformed 파랑) 3D 뷰 HTML 생성."""
    # 표시할 하중 케이스 결정
    if paper_case and paper_case in data["all_names"]:
        case_name = paper_case
    elif data["combo_names"]:
        case_name = data["combo_names"][0]
    else:
        case_name = data["all_names"][0]

    data_json = json.dumps(data, ensure_ascii=False)
    scale = int(deformation_scale)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>3D Frame - Paper Figure</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ margin:0; padding:0; background:#fff; }}
#plot3d {{ width:100vw; height:100vh; }}
</style>
</head>
<body>
<div id="plot3d"></div>
<script>
var DATA = {data_json};
var caseName = "{case_name}";
var scale = {scale};

(function() {{
  var types = ['column', 'beam_x', 'beam_y'];
  var traces = [];

  // Undeformed - single light gray
  var ux=[], uy=[], uz=[];
  for (var t = 0; t < 3; t++) {{
    var ms = DATA.members[types[t]];
    for (var i = 0; i < ms.length; i++) {{
      var ni = ms[i][0], nj = ms[i][1];
      var pi = DATA.node_map[String(ni)], pj = DATA.node_map[String(nj)];
      ux.push(pi[0], pj[0], null);
      uy.push(pi[1], pj[1], null);
      uz.push(pi[2], pj[2], null);
    }}
  }}
  traces.push({{type:'scatter3d', mode:'lines', x:ux, y:uy, z:uz,
    line:{{color:'#BDBDBD', width:3}}, name:'Undeformed', legendgroup:'undeformed'}});

  // Deformed - single dark blue
  var cd = DATA.case_data[caseName], disp = cd.disp;
  var dx=[], dy=[], dz=[];
  for (var t = 0; t < 3; t++) {{
    var ms = DATA.members[types[t]];
    for (var i = 0; i < ms.length; i++) {{
      var ni = ms[i][0], nj = ms[i][1];
      var pi = DATA.node_map[String(ni)], pj = DATA.node_map[String(nj)];
      var di = disp[String(ni)] || [0,0,0], dj = disp[String(nj)] || [0,0,0];
      dx.push(pi[0]+di[0]/1000*scale, pj[0]+dj[0]/1000*scale, null);
      dy.push(pi[1]+di[1]/1000*scale, pj[1]+dj[1]/1000*scale, null);
      dz.push(pi[2]+di[2]/1000*scale, pj[2]+dj[2]/1000*scale, null);
    }}
  }}
  traces.push({{type:'scatter3d', mode:'lines', x:dx, y:dy, z:dz,
    line:{{color:'#1565C0', width:5}}, name:'Deformed', legendgroup:'deformed'}});

  // Supports - black triangles
  var sx=[], sy=[], sz=[];
  for (var i = 0; i < DATA.support_ids.length; i++) {{
    var p = DATA.node_map[String(DATA.support_ids[i])];
    sx.push(p[0]); sy.push(p[1]); sz.push(p[2]);
  }}
  traces.push({{type:'scatter3d', mode:'markers', x:sx, y:sy, z:sz,
    marker:{{size:5, color:'#333', symbol:'diamond'}}, name:'Supports'}});

  var layout = {{
    scene: {{
      xaxis:{{title:'X (m)', gridcolor:'#eee', zerolinecolor:'#ccc'}},
      yaxis:{{title:'Y (m)', gridcolor:'#eee', zerolinecolor:'#ccc'}},
      zaxis:{{title:'Z (m)', gridcolor:'#eee', zerolinecolor:'#ccc'}},
      aspectmode:'data',
      camera:{{eye:{{x:1.8, y:1.8, z:1.2}}}},
      bgcolor:'#ffffff'
    }},
    height:800,
    margin:{{l:0,r:0,t:40,b:0}},
    legend:{{x:0.01, y:0.98, bgcolor:'rgba(255,255,255,0.9)',
             bordercolor:'#ccc', borderwidth:1, font:{{size:13}}}},
    title:{{text:'Load combination: {case_name}  |  Deformation scale: ' + scale + 'x',
            font:{{size:14, color:'#555'}}, x:0.5}},
    paper_bgcolor:'#ffffff',
    plot_bgcolor:'#ffffff'
  }};

  Plotly.newPlot('plot3d', traces, layout, {{responsive:true}});
}})();
</script>
</body>
</html>"""

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.html')
        os.close(fd)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path
