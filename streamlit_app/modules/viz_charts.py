"""2D 차트 모듈 (Plotly).

층간변위, 반력, 하중 리포트, 엔벨로프 테이블 등.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


def create_story_drift_chart(
    case_result,
    num_stories: int,
    allowable_drift: float = 1 / 200,
    Cd: float = 1.0,
) -> go.Figure:
    """X/Y 방향 층간변위 수평 바 차트.

    Args:
        case_result: Frame3DCaseResult (story_drifts 포함)
        num_stories: 총 층수
        allowable_drift: 허용 층간변위비 (기본 1/200)
        Cd: 변위증폭계수

    Returns:
        수평 바 차트 Figure
    """
    drifts = case_result.story_drifts
    stories = [d["story"] for d in drifts]
    drift_x = [d.get("drift_x", 0.0) for d in drifts]
    drift_y = [d.get("drift_y", 0.0) for d in drifts]

    # Cd 보정
    drift_x_cd = [dx * Cd for dx in drift_x]
    drift_y_cd = [dy * Cd for dy in drift_y]

    # 변위각 계산 (1/drift)
    def drift_to_ratio(d):
        return f"1/{int(1/d)}" if d > 1e-8 else "-"

    fig = go.Figure()

    # X 방향
    fig.add_trace(go.Bar(
        y=[f"{s}F" for s in stories],
        x=drift_x_cd,
        name="Drift X (Cd applied)" if Cd > 1 else "Drift X",
        orientation="h",
        marker_color="#E8524A",
        text=[drift_to_ratio(d) for d in drift_x_cd],
        textposition="outside",
    ))

    # Y 방향
    fig.add_trace(go.Bar(
        y=[f"{s}F" for s in stories],
        x=drift_y_cd,
        name="Drift Y (Cd applied)" if Cd > 1 else "Drift Y",
        orientation="h",
        marker_color="#4A90D9",
        text=[drift_to_ratio(d) for d in drift_y_cd],
        textposition="outside",
    ))

    # 허용치 라인
    fig.add_vline(
        x=allowable_drift,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Allowable: 1/{int(1/allowable_drift)}",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Story Drift Ratio",
        xaxis_title="Drift Ratio",
        yaxis_title="Story",
        barmode="group",
        height=max(400, num_stories * 50 + 100),
        xaxis=dict(range=[0, max(max(drift_x_cd + [0]) * 1.5, allowable_drift * 1.3)]),
        legend=dict(x=0.7, y=0.95),
    )
    return fig


def create_reaction_summary(case_result) -> pd.DataFrame:
    """기초 반력 요약 테이블."""
    rows = []
    for r in case_result.reactions:
        rows.append({
            "Node": r["node"],
            "RX (kN)": round(r.get("RX_kN", 0.0), 2),
            "RY (kN)": round(r.get("RY_kN", 0.0), 2),
            "RZ (kN)": round(r.get("RZ_kN", 0.0), 2),
            "MX (kNm)": round(r.get("MX_kNm", 0.0), 2),
            "MY (kNm)": round(r.get("MY_kNm", 0.0), 2),
            "MZ (kNm)": round(r.get("MZ_kNm", 0.0), 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # 합계 행
        sums = {col: df[col].sum() if col != "Node" else "SUM" for col in df.columns}
        sum_df = pd.DataFrame([sums])
        df = pd.concat([df, sum_df], ignore_index=True)
    return df


def create_envelope_table(result) -> pd.DataFrame:
    """조합별 엔벨로프 최대값 테이블."""
    rows = []
    for combo_name, cr in result.combo_results.items():
        rows.append({
            "Combination": combo_name,
            "Max Disp X (mm)": round(cr.max_displacement_x, 2),
            "Max Disp Y (mm)": round(cr.max_displacement_y, 2),
            "Max Disp Z (mm)": round(cr.max_displacement_z, 2),
            "Max Drift X": f"1/{int(1/cr.max_drift_x)}" if cr.max_drift_x > 1e-8 else "-",
            "Max Drift Y": f"1/{int(1/cr.max_drift_y)}" if cr.max_drift_y > 1e-8 else "-",
            "Max Moment (kNm)": round(cr.max_moment, 1),
            "Max Axial (kN)": round(cr.max_axial, 1),
            "Max Shear (kN)": round(cr.max_shear, 1),
        })
    return pd.DataFrame(rows)


def create_load_report_sections(load_result: dict) -> dict[str, str]:
    """하중 생성 리포트를 섹션별 텍스트로 변환."""
    reports = load_result.get("reports", {})
    sections = {}

    # 중력하중 (list of dicts per story)
    grav = reports.get("gravity", [])
    if grav:
        lines = []
        if isinstance(grav, list):
            for item in grav:
                story = item.get("story", "?")
                lines.append(f"**Story {story} ({item.get('usage', '')}):**")
                lines.append(f"  - DL: {item.get('DL_kNm2', '?')} kN/m²")
                bd = item.get("DL_breakdown", {})
                if bd:
                    parts = ", ".join(f"{k}={v}" for k, v in bd.items())
                    lines.append(f"  - Breakdown: {parts}")
                lines.append(f"  - LL: {item.get('LL_kNm2', '?')} kN/m²")
        elif isinstance(grav, dict):
            for k, v in grav.items():
                lines.append(f"- {k}: {v}")
        sections["Gravity Loads (DL/LL)"] = "\n".join(lines)

    # 지진하중
    seis = reports.get("seismic", {})
    if seis:
        lines = []
        for k, v in seis.items():
            if isinstance(v, dict):
                lines.append(f"**{k}:**")
                for kk, vv in v.items():
                    lines.append(f"  - {kk}: {vv}")
            elif isinstance(v, list):
                lines.append(f"**{k}:** {len(v)} items")
            else:
                lines.append(f"**{k}:** {v}")
        sections["Seismic Loads (EQX/EQY)"] = "\n".join(lines)

    # 풍하중
    wind = reports.get("wind", {})
    if wind:
        lines = []
        for k, v in wind.items():
            if isinstance(v, dict):
                lines.append(f"**{k}:**")
                for kk, vv in v.items():
                    lines.append(f"  - {kk}: {vv}")
            elif isinstance(v, list):
                lines.append(f"**{k}:** {len(v)} items")
            else:
                lines.append(f"**{k}:** {v}")
        sections["Wind Loads (WX/WY)"] = "\n".join(lines)

    # 요약
    summary = load_result.get("summary", {})
    if summary:
        lines = [f"- {k}: {v}" for k, v in summary.items()]
        sections["Summary"] = "\n".join(lines)

    return sections
