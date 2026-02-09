"""
평형 검증 모듈
해석 결과의 정적 평형 조건을 자동 검증
"""

import numpy as np


def _compute_total_applied_load(load_info: list[dict]) -> float:
    """load_info에서 전체 수직 외력 합 계산 (kN, 아래 방향 양수)"""
    total = 0.0
    for ld in load_info:
        lt = ld.get("type", "uniform")
        if lt == "uniform":
            w = ld.get("value", 0.0)
            length = ld.get("end", 0.0) - ld.get("start", 0.0)
            total += w * length
        elif lt == "point":
            total += ld.get("value", 0.0)
        elif lt == "triangular":
            w1 = ld.get("value", 0.0)
            w2 = ld.get("value_end", 0.0)
            length = ld.get("end", 0.0) - ld.get("start", 0.0)
            total += 0.5 * (w1 + w2) * length
    return total


def _compute_load_moment_about(load_info: list[dict], x_ref: float) -> float:
    """load_info에서 x_ref 기준 모멘트 합 계산 (kN·m, 시계 방향 양수)"""
    total_m = 0.0
    for ld in load_info:
        lt = ld.get("type", "uniform")
        if lt == "uniform":
            w = ld.get("value", 0.0)
            a = ld.get("start", 0.0)
            b = ld.get("end", 0.0)
            length = b - a
            centroid = (a + b) / 2
            total_m += w * length * (centroid - x_ref)
        elif lt == "point":
            P = ld.get("value", 0.0)
            loc = ld.get("location", 0.0)
            total_m += P * (loc - x_ref)
        elif lt == "triangular":
            w1 = ld.get("value", 0.0)
            w2 = ld.get("value_end", 0.0)
            a = ld.get("start", 0.0)
            b = ld.get("end", 0.0)
            length = b - a
            # 사다리꼴: 균일 부분 + 삼각형 부분
            # 균일 w_min의 중심 = (a+b)/2
            # 삼각형 부분의 중심 = a + 2L/3 (w1>w2일 때) 또는 a + L/3 (w2>w1일 때)
            w_min = min(w1, w2)
            w_diff = abs(w1 - w2)
            # 균일 부분
            total_m += w_min * length * ((a + b) / 2 - x_ref)
            # 삼각형 부분
            if w1 >= w2:
                centroid_tri = a + length / 3
            else:
                centroid_tri = a + 2 * length / 3
            total_m += 0.5 * w_diff * length * (centroid_tri - x_ref)
    return total_m


def verify_equilibrium(result) -> dict:
    """
    해석 결과의 정적 평형 검증

    Parameters
    ----------
    result : BeamResult 또는 ContinuousBeamResult

    Returns
    -------
    dict: 검증 결과
        - sum_vertical: ΣV = 0 검증
        - sum_moment: ΣM = 0 검증 (좌단 기준)
        - shear_jumps: 지점별 전단 점프 = 반력 검증
        - all_passed: 전체 통과 여부
    """
    load_info = getattr(result, 'load_info', []) or []
    checks = {}

    # --- 1) ΣV = 0 ---
    total_applied = _compute_total_applied_load(load_info)

    # 반력 합 (위 방향 = 양수)
    if hasattr(result, 'reactions'):
        # 연속보
        total_reaction = sum(r.get("vertical_kN", 0.0) for r in result.reactions)
    else:
        # 단순보
        total_reaction = getattr(result, 'reaction_left', 0.0) + getattr(result, 'reaction_right', 0.0)

    sum_v_error = total_reaction - total_applied
    ref_v = max(total_applied, 1.0)
    sum_v_ok = abs(sum_v_error) < max(0.01 * ref_v, 0.1)
    checks["sum_vertical"] = {
        "description": "ΣV = 0",
        "reaction_sum_kN": round(total_reaction, 3),
        "applied_load_kN": round(total_applied, 3),
        "error_kN": round(sum_v_error, 3),
        "status": "OK" if sum_v_ok else "FAIL",
    }

    # --- 2) ΣM = 0 (좌단 기준) ---
    # 하중에 의한 모멘트 (시계 방향 양수 = 아래 하중이 오른쪽에 있으면 양수)
    if hasattr(result, 'reactions'):
        x_ref = result.reactions[0]["location"] if result.reactions else 0.0
    else:
        x_ref = 0.0

    load_moment = _compute_load_moment_about(load_info, x_ref)

    # 반력에 의한 모멘트 (위 방향 반력이 오른쪽 → 반시계 = 음수)
    reaction_moment = 0.0
    if hasattr(result, 'reactions'):
        for r in result.reactions:
            dist = r["location"] - x_ref
            reaction_moment -= r.get("vertical_kN", 0.0) * dist
            # 모멘트 반력 (고정단)
            reaction_moment -= r.get("moment_kNm", 0.0)
    else:
        span = getattr(result, 'node_positions', [0])
        total_len = span[-1] if span else 0.0
        reaction_moment -= getattr(result, 'reaction_right', 0.0) * total_len
        # 모멘트 반력
        reaction_moment -= getattr(result, 'reaction_moment_left', 0.0)
        reaction_moment += getattr(result, 'reaction_moment_right', 0.0)

    sum_m_error = load_moment + reaction_moment
    ref_m = max(abs(load_moment), 1.0)
    sum_m_ok = abs(sum_m_error) < max(0.01 * ref_m, 0.1)
    checks["sum_moment"] = {
        "description": "ΣM_about_A = 0",
        "error_kNm": round(sum_m_error, 3),
        "status": "OK" if sum_m_ok else "FAIL",
    }

    # --- 3) 전단 점프 = 반력 ---
    shear_jump_checks = []
    node_positions = getattr(result, 'node_positions', [])
    shears_array = getattr(result, 'shears', [])

    if node_positions and shears_array:
        # 지점 위치 목록
        if hasattr(result, 'reactions'):
            support_locs = [r["location"] for r in result.reactions]
            support_reactions = [r.get("vertical_kN", 0.0) for r in result.reactions]
        else:
            total_len = node_positions[-1] if node_positions else 0.0
            support_locs = [0.0, total_len]
            support_reactions = [
                getattr(result, 'reaction_left', 0.0),
                getattr(result, 'reaction_right', 0.0),
            ]

        for loc, reaction in zip(support_locs, support_reactions):
            # 해당 위치에서 전단 값 찾기
            indices = [i for i, x in enumerate(node_positions) if abs(x - loc) < 0.001]
            if len(indices) >= 2:
                # 중간 지점: 좌/우 값 존재
                v_left = shears_array[indices[0]]
                v_right = shears_array[indices[1]]
                jump = abs(v_right - v_left)
            elif len(indices) == 1:
                # 끝 지점
                idx = indices[0]
                jump = abs(shears_array[idx])
            else:
                jump = 0.0

            err = abs(jump - reaction)
            ref = max(reaction, 1.0)
            ok = err < max(0.02 * ref, 0.5)
            shear_jump_checks.append({
                "location_m": round(loc, 3),
                "reaction_kN": round(reaction, 2),
                "shear_jump_kN": round(jump, 2),
                "error_kN": round(err, 2),
                "status": "OK" if ok else "FAIL",
            })

    all_jumps_ok = all(c["status"] == "OK" for c in shear_jump_checks) if shear_jump_checks else True
    checks["shear_jumps"] = {
        "description": "Shear jump = Reaction at supports",
        "details": shear_jump_checks,
        "status": "OK" if all_jumps_ok else "FAIL",
    }

    # --- 전체 판정 ---
    checks["all_passed"] = all(
        checks[k].get("status") == "OK"
        for k in ["sum_vertical", "sum_moment", "shear_jumps"]
    )

    return checks


# ============================================================
# 2D 골조 평형검증
# ============================================================

def verify_frame_equilibrium(
    case_result,
    loads: list[dict],
    stories: list[float],
    bays: list[float],
) -> dict:
    """
    2D 골조 해석 결과의 정적 평형 검증

    Parameters
    ----------
    case_result : Frame2DCaseResult
        단일 케이스 (또는 조합) 해석 결과
    loads : list[dict]
        해당 케이스의 하중 정의
    stories : list[float]
        각 층의 높이 (m)
    bays : list[float]
        각 경간의 폭 (m)

    Returns
    -------
    dict: 검증 결과
        - sum_horizontal: ΣFx = 0 검증
        - sum_vertical: ΣFy = 0 검증
        - sum_moment: ΣM = 0 검증 (기초 좌측 기준)
        - all_passed: 전체 통과 여부
    """
    checks = {}
    n_stories = len(stories)
    total_width = sum(bays)

    # 층 높이 누적 (y좌표)
    y_levels = [0.0]
    for s in stories:
        y_levels.append(y_levels[-1] + s)

    # x 위치
    x_positions = [0.0]
    for b in bays:
        x_positions.append(x_positions[-1] + b)

    # --- 외력 계산 ---
    total_fx_applied = 0.0   # 횡하중 합 (kN, 오른쪽 양수)
    total_fy_applied = 0.0   # 수직하중 합 (kN, 아래 양수)
    total_m_applied = 0.0    # 기초 좌단(0,0) 기준 모멘트 (kN·m)

    for ld in loads:
        ld_type = ld.get("type", "floor")
        story = ld.get("story", 1)

        if ld_type == "floor":
            # 등분포 하중: w (kN/m) × 전체 폭 = 총 수직력
            w = ld.get("value", 0.0)
            force_y = w * total_width
            total_fy_applied += force_y

            # 모멘트: 각 bay별 하중의 도심 기준
            for i, bay_w in enumerate(bays):
                bay_force = w * bay_w
                centroid_x = x_positions[i] + bay_w / 2
                y_story = y_levels[story] if story <= n_stories else y_levels[-1]
                # 아래 방향 하중 × x거리 (시계 양수) - 아래 하중은 y축 기준 모멘트 기여
                total_m_applied += bay_force * centroid_x
                # 수직하중은 x축 기준 모멘트 기여 없음 (2D 기준 y=0 기초에서)
                # 실제로 기초 좌단 기준: M = Fy * x - Fx * y

        elif ld_type == "lateral":
            fx = ld.get("fx", ld.get("value", 0.0))
            total_fx_applied += fx
            y_story = y_levels[story] if story <= n_stories else y_levels[-1]
            # 횡하중 모멘트 (기초 좌단 기준): Fx * y
            total_m_applied += fx * y_story

        elif ld_type == "nodal":
            fx = ld.get("fx", 0.0)
            fy = ld.get("fy", 0.0)
            mz = ld.get("mz", 0.0)
            total_fx_applied += fx
            total_fy_applied += fy
            total_m_applied += mz
            # 노드 위치를 알아야 정확한 모멘트 계산 가능
            # 여기서는 근사적으로 mz만 반영 (노드 위치 미상)

    # --- 반력 계산 ---
    reactions = getattr(case_result, 'reactions', []) or []
    total_rx = sum(r.get("RX_kN", 0.0) for r in reactions)
    total_ry = sum(r.get("RY_kN", 0.0) for r in reactions)
    total_mz = sum(r.get("MZ_kNm", 0.0) for r in reactions)

    # 반력에 의한 기초 좌단 기준 모멘트
    reaction_moment = total_mz
    for r in reactions:
        x = r.get("x_m", 0.0)
        # RY * x (위 방향 반력 × 오른쪽 거리 = 반시계 = 음수)
        reaction_moment += r.get("RY_kN", 0.0) * x

    # --- 1) ΣFx = 0 ---
    # OpenSees 반력 부호: 반력은 외력과 반대 방향
    # total_rx는 반력 (외력 + 반력 = 0 → 반력 = -외력)
    sum_fx_error = total_fx_applied + total_rx
    ref_fx = max(abs(total_fx_applied), 1.0)
    sum_fx_ok = abs(sum_fx_error) < max(0.01 * ref_fx, 0.1)
    checks["sum_horizontal"] = {
        "description": "ΣFx = 0",
        "applied_kN": round(total_fx_applied, 3),
        "reaction_sum_kN": round(total_rx, 3),
        "error_kN": round(sum_fx_error, 3),
        "status": "OK" if sum_fx_ok else "FAIL",
    }

    # --- 2) ΣFy = 0 ---
    # 수직: total_fy_applied는 아래 방향 양수, total_ry는 위 방향 양수 (OpenSees 규약)
    # 평형 시 두 값이 같아야 하므로 차이로 오차 계산
    sum_fy_error = total_fy_applied - total_ry
    ref_fy = max(abs(total_fy_applied), 1.0)
    sum_fy_ok = abs(sum_fy_error) < max(0.01 * ref_fy, 0.1)
    checks["sum_vertical"] = {
        "description": "ΣFy = 0",
        "applied_kN": round(total_fy_applied, 3),
        "reaction_sum_kN": round(total_ry, 3),
        "error_kN": round(sum_fy_error, 3),
        "status": "OK" if sum_fy_ok else "FAIL",
    }

    # --- 3) ΣM = 0 (기초 좌단 기준) ---
    # total_m_applied (하중 모멘트)와 reaction_moment (반력 모멘트)는
    # 같은 부호 규약으로 계산되므로, 평형 시 같은 값 → 차이로 오차 계산
    sum_m_error = total_m_applied - reaction_moment
    ref_m = max(abs(total_m_applied), 1.0)
    sum_m_ok = abs(sum_m_error) < max(0.01 * ref_m, 0.1)
    checks["sum_moment"] = {
        "description": "ΣM_about_base_left = 0",
        "applied_moment_kNm": round(total_m_applied, 3),
        "reaction_moment_kNm": round(reaction_moment, 3),
        "error_kNm": round(sum_m_error, 3),
        "status": "OK" if sum_m_ok else "FAIL",
    }

    # --- 전체 판정 ---
    checks["all_passed"] = all(
        checks[k].get("status") == "OK"
        for k in ["sum_horizontal", "sum_vertical", "sum_moment"]
    )

    return checks
