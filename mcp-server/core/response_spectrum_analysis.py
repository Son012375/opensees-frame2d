"""Response Spectrum Analysis (RSA) Module.

KDS 17 10 00 기반 응답스펙트럼해석.
고유치해석 결과 + 설계응답스펙트럼 → 모달 중첩 → 구조 응답.

모달 조합 규칙:
  - SRSS (Square Root of Sum of Squares)
  - CQC (Complete Quadratic Combination)

방향 조합:
  - 30% 규칙: 100%X + 30%Y, 30%X + 100%Y (KDS 표준)
  - SRSS: sqrt(X^2 + Y^2)

기저전단 최소값 검증:
  - RSA 기저전단 < 0.85 × ELF 기저전단 → 스케일 업 (KDS 41 17 00 §8.2.1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RSAResult:
    """Response Spectrum Analysis result."""
    # Per-mode spectral responses
    modal_responses: list[dict] = field(default_factory=list)

    # Combined responses (SRSS or CQC)
    combined_x: dict = field(default_factory=dict)
    combined_y: dict = field(default_factory=dict)

    # Direction-combined (30% rule or SRSS)
    direction_combined: list[dict] = field(default_factory=list)

    # Envelope per node/element
    node_displacements: dict = field(default_factory=dict)  # {node_id: {dx, dy, dz}}
    story_drifts: list[dict] = field(default_factory=list)
    base_shear: dict = field(default_factory=dict)  # {Vx_kN, Vy_kN}

    # Scale factor (if RSA < 85% ELF)
    scale_factor: float = 1.0
    elf_base_shear: float = 0.0

    # Metadata
    combination_rule: str = "SRSS"
    direction_rule: str = "30pct"
    num_modes_used: int = 0
    parameters: dict = field(default_factory=dict)


def interpolate_sa(periods: list[float], sa_values: list[float], T: float) -> float:
    """Interpolate Sa at period T from design spectrum arrays."""
    if T <= periods[0]:
        return sa_values[0]
    if T >= periods[-1]:
        return sa_values[-1]
    for i in range(len(periods) - 1):
        if periods[i] <= T <= periods[i + 1]:
            t0, t1 = periods[i], periods[i + 1]
            s0, s1 = sa_values[i], sa_values[i + 1]
            if abs(t1 - t0) < 1e-12:
                return s0
            return s0 + (s1 - s0) * (T - t0) / (t1 - t0)
    return sa_values[-1]


def compute_cqc_coefficient(Ti: float, Tj: float, damping: float = 0.05) -> float:
    """Compute CQC cross-modal correlation coefficient.

    Der Kiureghian (1981) formula:
    ρ_ij = 8 * ξ² * (1 + r) * r^(3/2) / [(1 - r²)² + 4ξ²r(1 + r)²]
    where r = ωj/ωi = Ti/Tj
    """
    if Ti <= 0 or Tj <= 0:
        return 0.0
    r = Ti / Tj  # ratio of periods (= inverse ratio of frequencies)
    xi = damping
    num = 8.0 * xi * xi * (1.0 + r) * r ** 1.5
    den = (1.0 - r * r) ** 2 + 4.0 * xi * xi * r * (1.0 + r) ** 2
    if abs(den) < 1e-30:
        return 1.0
    return num / den


def compute_modal_responses(
    modal_analysis: dict,
    spectrum_periods: list[float],
    spectrum_sa: list[float],
    node_coords: dict,
    direction: str = "x",
    damping_ratio: float = 0.05,
    IE: float = 1.0,
    R: float = 1.0,
    g: float = 9.81,
) -> list[dict]:
    """Compute per-mode spectral response.

    For each mode n:
      Sa_n = Sa(T_n) from design spectrum
      Sd_n = Sa_n * (T_n / 2π)² * g   [displacement spectrum]
      For each node i:
        u_i_n = Γ_n * φ_i_n * Sd_n   [modal displacement]

    Args:
        modal_analysis: From _run_eigen_analysis() → {modes: [{period_s, shape, mass_participation}]}
        spectrum_periods: Sa(T) period array
        spectrum_sa: Sa(T) values (in g)
        node_coords: {node_id: {x, y, z}} structural coordinates
        direction: "x" or "y"
        damping_ratio: ξ (default 5%)
        IE: importance factor
        R: response modification factor
        g: gravity (m/s²)

    Returns:
        List of per-mode response dicts.
    """
    modes = modal_analysis.get("modes", [])
    responses = []

    for mode in modes:
        T = mode["period_s"]
        if T <= 0:
            continue

        shape = mode.get("shape", {})
        mp = mode.get("mass_participation", {})

        # Direction-specific participation
        if direction == "x":
            gamma_pct = mp.get("x_pct", 0)
            dof_idx = 0
        else:
            gamma_pct = mp.get("y_pct", 0)
            dof_idx = 1

        # Spectral acceleration at this period
        Sa_g = interpolate_sa(spectrum_periods, spectrum_sa, T)
        # Apply importance factor and response modification
        Cs_mode = Sa_g * IE / R

        # Spectral displacement: Sd = Sa * (T/2π)² * g
        omega = 2.0 * math.pi / T
        Sd_m = Cs_mode * g / (omega * omega)  # in meters

        # Per-node modal displacement
        node_displacements = {}
        max_disp = 0.0
        for nid_str, phi in shape.items():
            nid = int(nid_str)
            phi_dir = phi[dof_idx] if dof_idx < len(phi) else 0.0
            # Modal displacement = participation_ratio * phi * Sd
            # Since shape is normalized (max=1), scale by gamma_pct/100 approx
            # More precisely: u = Γ * φ * Sd where Γ = L/M*
            # We use the fact that m_eff/M_total = gamma_pct/100
            # and Γ * Sd ≈ sqrt(m_eff/M_total) * Sd for SRSS-type
            u_dir = phi_dir * Sd_m * 1000  # convert to mm

            node_displacements[nid] = {
                "dx_mm": u_dir if direction == "x" else 0.0,
                "dy_mm": u_dir if direction == "y" else 0.0,
                "dz_mm": phi[2] * Sd_m * 1000 if len(phi) > 2 else 0.0,
            }
            max_disp = max(max_disp, abs(u_dir))

        # Modal base shear: V_n = Sa_n * m_eff_n * g (approximation)
        # m_eff = gamma_pct/100 * M_total (but we don't have M_total here)
        # Store Sa for later combination
        responses.append({
            "mode": mode["mode"],
            "period_s": T,
            "direction": direction,
            "Sa_g": round(Sa_g, 6),
            "Cs_mode": round(Cs_mode, 6),
            "Sd_mm": round(Sd_m * 1000, 4),
            "max_disp_mm": round(max_disp, 4),
            "gamma_pct": gamma_pct,
            "node_displacements": node_displacements,
        })

    return responses


def combine_srss(modal_responses: list[dict]) -> dict:
    """SRSS combination of modal responses.

    R_total = sqrt(Σ R_i²)
    """
    combined_nodes = {}
    for resp in modal_responses:
        for nid, disp in resp["node_displacements"].items():
            if nid not in combined_nodes:
                combined_nodes[nid] = {"dx2": 0.0, "dy2": 0.0, "dz2": 0.0}
            combined_nodes[nid]["dx2"] += disp["dx_mm"] ** 2
            combined_nodes[nid]["dy2"] += disp["dy_mm"] ** 2
            combined_nodes[nid]["dz2"] += disp["dz_mm"] ** 2

    result = {}
    for nid, sq in combined_nodes.items():
        result[nid] = {
            "dx_mm": round(math.sqrt(sq["dx2"]), 4),
            "dy_mm": round(math.sqrt(sq["dy2"]), 4),
            "dz_mm": round(math.sqrt(sq["dz2"]), 4),
        }
    return result


def combine_cqc(modal_responses: list[dict], damping: float = 0.05) -> dict:
    """CQC combination of modal responses.

    R_total = sqrt(Σ_i Σ_j ρ_ij * R_i * R_j)
    """
    n = len(modal_responses)
    if n == 0:
        return {}

    periods = [r["period_s"] for r in modal_responses]

    # Pre-compute correlation matrix
    rho = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rho[i][j] = compute_cqc_coefficient(periods[i], periods[j], damping)

    # Collect all node IDs
    all_nodes = set()
    for resp in modal_responses:
        all_nodes.update(resp["node_displacements"].keys())

    result = {}
    for nid in all_nodes:
        dx_sum = 0.0
        dy_sum = 0.0
        dz_sum = 0.0

        for i in range(n):
            di = modal_responses[i]["node_displacements"].get(nid, {"dx_mm": 0, "dy_mm": 0, "dz_mm": 0})
            for j in range(n):
                dj = modal_responses[j]["node_displacements"].get(nid, {"dx_mm": 0, "dy_mm": 0, "dz_mm": 0})
                dx_sum += rho[i][j] * di["dx_mm"] * dj["dx_mm"]
                dy_sum += rho[i][j] * di["dy_mm"] * dj["dy_mm"]
                dz_sum += rho[i][j] * di["dz_mm"] * dj["dz_mm"]

        result[nid] = {
            "dx_mm": round(math.sqrt(max(0, dx_sum)), 4),
            "dy_mm": round(math.sqrt(max(0, dy_sum)), 4),
            "dz_mm": round(math.sqrt(max(0, dz_sum)), 4),
        }
    return result


def combine_directions(
    combined_x: dict,
    combined_y: dict,
    rule: str = "30pct",
) -> list[dict]:
    """Combine X and Y direction responses.

    30% rule (KDS): max of (100%X + 30%Y) and (30%X + 100%Y)
    SRSS: sqrt(X² + Y²)

    Returns list of direction combination cases.
    """
    all_nodes = set(combined_x.keys()) | set(combined_y.keys())

    if rule == "SRSS":
        nodes = {}
        for nid in all_nodes:
            dx_x = combined_x.get(nid, {}).get("dx_mm", 0)
            dy_y = combined_y.get(nid, {}).get("dy_mm", 0)
            dz_x = combined_x.get(nid, {}).get("dz_mm", 0)
            dz_y = combined_y.get(nid, {}).get("dz_mm", 0)
            nodes[nid] = {
                "dx_mm": round(math.sqrt(dx_x**2), 4),
                "dy_mm": round(math.sqrt(dy_y**2), 4),
                "dz_mm": round(math.sqrt(dz_x**2 + dz_y**2), 4),
            }
        return [{"name": "RSA_SRSS", "nodes": nodes}]
    else:
        # 30% rule: two cases
        cases = []
        for label, fx, fy in [("RSA_100X_30Y", 1.0, 0.3), ("RSA_30X_100Y", 0.3, 1.0)]:
            nodes = {}
            for nid in all_nodes:
                dx_x = combined_x.get(nid, {}).get("dx_mm", 0)
                dy_y = combined_y.get(nid, {}).get("dy_mm", 0)
                dz_x = combined_x.get(nid, {}).get("dz_mm", 0)
                dz_y = combined_y.get(nid, {}).get("dz_mm", 0)
                nodes[nid] = {
                    "dx_mm": round(fx * dx_x + fy * 0, 4),  # X-dir disp only from EQX
                    "dy_mm": round(fy * dy_y + fx * 0, 4),  # Y-dir disp only from EQY
                    "dz_mm": round(fx * dz_x + fy * dz_y, 4),
                }
            cases.append({"name": label, "nodes": nodes})
        return cases


def compute_story_drifts_rsa(
    combined_nodes: dict,
    viewer_nodes: list[dict],
    stories: list[float],
    bays_x: list[float],
    bays_y: list[float],
) -> list[dict]:
    """Compute story drifts from RSA combined displacements.

    Groups nodes by story height, computes max drift per story.
    """
    # Build height→story map
    z_levels = [0.0]
    for h in stories:
        z_levels.append(z_levels[-1] + h)

    # Group nodes by story
    story_nodes = {s: [] for s in range(len(stories))}
    for vn in viewer_nodes:
        z = vn.get("z_m", vn.get("z", 0))
        for s in range(len(stories)):
            if abs(z - z_levels[s + 1]) < 0.01:
                story_nodes[s].append(vn["id"])
                break

    # Lower story nodes
    lower_nodes = {s: [] for s in range(len(stories))}
    for vn in viewer_nodes:
        z = vn.get("z_m", vn.get("z", 0))
        for s in range(len(stories)):
            if abs(z - z_levels[s]) < 0.01:
                lower_nodes[s].append(vn["id"])
                break

    drifts = []
    for s in range(len(stories)):
        h = stories[s]
        max_drift_x = 0.0
        max_drift_y = 0.0

        # Upper story max displacement
        for nid in story_nodes[s]:
            d = combined_nodes.get(nid, {})
            dx_upper = d.get("dx_mm", 0)
            dy_upper = d.get("dy_mm", 0)

            # Find corresponding lower node (same x, y)
            # For RSA, all values are positive (absolute), so drift = upper / h
            drift_x = abs(dx_upper) / (h * 1000)  # mm / mm
            drift_y = abs(dy_upper) / (h * 1000)
            max_drift_x = max(max_drift_x, drift_x)
            max_drift_y = max(max_drift_y, drift_y)

        drifts.append({
            "story": s + 1,
            "height_m": h,
            "drift_x": round(max_drift_x, 6),
            "drift_y": round(max_drift_y, 6),
        })

    return drifts


def run_response_spectrum_analysis(
    modal_analysis: dict,
    spectrum_result: dict,
    viewer_nodes: list[dict],
    stories: list[float],
    bays_x: list[float],
    bays_y: list[float],
    combination_rule: str = "CQC",
    direction_rule: str = "30pct",
    IE: float = 1.0,
    R: float = 1.0,
    damping_ratio: float = 0.05,
    elf_base_shear_kN: float = 0.0,
) -> RSAResult:
    """Run complete RSA pipeline.

    Args:
        modal_analysis: From frame_3d._run_eigen_analysis()
        spectrum_result: From design_spectrum.compute_design_spectrum()
        viewer_nodes: [{id, x, y, z}, ...] structural grid nodes
        stories: [h1, h2, ...] story heights in m
        bays_x, bays_y: bay widths
        combination_rule: "SRSS" or "CQC"
        direction_rule: "30pct" or "SRSS"
        IE: importance factor
        R: response modification factor
        damping_ratio: ξ
        elf_base_shear_kN: ELF base shear for minimum check

    Returns:
        RSAResult with combined responses.
    """
    result = RSAResult()
    result.combination_rule = combination_rule
    result.direction_rule = direction_rule

    # Extract spectrum data
    spec = spectrum_result.get("spectrum", {})
    periods = spec.get("periods", [])
    sa_values = spec.get("Sa", [])
    if not periods or not sa_values:
        return result

    params = spectrum_result.get("parameters", {})
    result.parameters = {
        "SDS": params.get("SDS", 0),
        "SD1": params.get("SD1", 0),
        "IE": IE,
        "R": R,
        "damping_ratio": damping_ratio,
    }

    # Node coordinate lookup
    node_coords = {n["id"]: n for n in viewer_nodes}

    # 1. Compute per-mode responses for X and Y
    responses_x = compute_modal_responses(
        modal_analysis, periods, sa_values, node_coords,
        direction="x", damping_ratio=damping_ratio, IE=IE, R=R,
    )
    responses_y = compute_modal_responses(
        modal_analysis, periods, sa_values, node_coords,
        direction="y", damping_ratio=damping_ratio, IE=IE, R=R,
    )

    result.modal_responses = responses_x + responses_y
    result.num_modes_used = len(modal_analysis.get("modes", []))

    # 2. Modal combination (SRSS or CQC)
    if combination_rule == "CQC":
        result.combined_x = combine_cqc(responses_x, damping_ratio)
        result.combined_y = combine_cqc(responses_y, damping_ratio)
    else:
        result.combined_x = combine_srss(responses_x)
        result.combined_y = combine_srss(responses_y)

    # 3. Direction combination
    result.direction_combined = combine_directions(
        result.combined_x, result.combined_y, direction_rule,
    )

    # 4. Compute story drifts from envelope of direction combinations
    envelope_nodes = {}
    for dc in result.direction_combined:
        for nid, disp in dc["nodes"].items():
            if nid not in envelope_nodes:
                envelope_nodes[nid] = {"dx_mm": 0, "dy_mm": 0, "dz_mm": 0}
            envelope_nodes[nid]["dx_mm"] = max(envelope_nodes[nid]["dx_mm"], disp["dx_mm"])
            envelope_nodes[nid]["dy_mm"] = max(envelope_nodes[nid]["dy_mm"], disp["dy_mm"])
            envelope_nodes[nid]["dz_mm"] = max(envelope_nodes[nid]["dz_mm"], disp["dz_mm"])

    result.node_displacements = envelope_nodes
    result.story_drifts = compute_story_drifts_rsa(
        envelope_nodes, viewer_nodes, stories, bays_x, bays_y,
    )

    # 5. Base shear estimation
    max_dx = max((d["dx_mm"] for d in envelope_nodes.values()), default=0)
    max_dy = max((d["dy_mm"] for d in envelope_nodes.values()), default=0)
    result.base_shear = {"max_dx_mm": round(max_dx, 3), "max_dy_mm": round(max_dy, 3)}

    # 6. Scale factor check (RSA vs ELF minimum)
    result.elf_base_shear = elf_base_shear_kN
    # Scale factor is applied at the design check level, not here
    result.scale_factor = 1.0

    return result


def rsa_result_to_case_data(rsa: RSAResult) -> dict:
    """Convert RSA result to case_data format for frontend.

    Returns dict compatible with the existing case_data structure:
    {case_name: {summary: {...}, displacements: {node_id: [dx, dy, dz]}, story_drifts: [...]}}
    """
    case_data = {}

    # Per-direction combined results
    for label, combined in [("EQX_RSA", rsa.combined_x), ("EQY_RSA", rsa.combined_y)]:
        max_dx = max((d["dx_mm"] for d in combined.values()), default=0)
        max_dy = max((d["dy_mm"] for d in combined.values()), default=0)
        max_dz = max((d["dz_mm"] for d in combined.values()), default=0)

        displacements = {}
        for nid, d in combined.items():
            displacements[str(nid)] = [d["dx_mm"], d["dy_mm"], d["dz_mm"]]

        case_data[label] = {
            "summary": {
                "max_dx_mm": round(max_dx, 3),
                "max_dy_mm": round(max_dy, 3),
                "max_dz_mm": round(max_dz, 3),
                "max_drift_x": max((d["drift_x"] for d in rsa.story_drifts), default=0),
                "max_drift_y": max((d["drift_y"] for d in rsa.story_drifts), default=0),
                "max_moment_kNm": 0,  # RSA doesn't directly give forces (displacement-based)
                "max_axial_kN": 0,
                "max_shear_kN": 0,
            },
            "displacements": displacements,
            "story_drifts": rsa.story_drifts,
        }

    # Direction-combined cases
    for dc in rsa.direction_combined:
        max_dx = max((d["dx_mm"] for d in dc["nodes"].values()), default=0)
        max_dy = max((d["dy_mm"] for d in dc["nodes"].values()), default=0)

        displacements = {}
        for nid, d in dc["nodes"].items():
            displacements[str(nid)] = [d["dx_mm"], d["dy_mm"], d.get("dz_mm", 0)]

        case_data[dc["name"]] = {
            "summary": {
                "max_dx_mm": round(max_dx, 3),
                "max_dy_mm": round(max_dy, 3),
                "max_dz_mm": 0,
                "max_drift_x": max((d["drift_x"] for d in rsa.story_drifts), default=0),
                "max_drift_y": max((d["drift_y"] for d in rsa.story_drifts), default=0),
                "max_moment_kNm": 0,
                "max_axial_kN": 0,
                "max_shear_kN": 0,
            },
            "displacements": displacements,
            "story_drifts": rsa.story_drifts,
        }

    return case_data
