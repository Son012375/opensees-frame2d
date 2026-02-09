"""
구조해석 결과 시각화 모듈
matplotlib (정적 PNG/SVG) + plotly (인터랙티브 HTML)

레이아웃:
  상단: [구조형상도 (좌)] [단면 정보 (우)]
  중단: SFD
  하단: BMD
  최하단: Displacement (Position)
"""
from __future__ import annotations

import os
import io
import base64
import math
import tempfile
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # 비GUI 백엔드
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _get_support_info(result) -> list[dict]:
    """결과 객체에서 지점 정보 추출"""
    supports = []
    if hasattr(result, 'num_spans'):
        total_len = 0.0
        sup_types = result.supports if result.supports else []
        supports.append({"x": 0.0, "type": sup_types[0] if sup_types else "pin"})
        for i, span in enumerate(result.spans):
            total_len += span
            st = sup_types[i + 1] if i + 1 < len(sup_types) else "roller"
            supports.append({"x": total_len, "type": st})
    else:
        st = getattr(result, 'support_type', 'simple')
        span = result.node_positions[-1] if result.node_positions else 0
        if st == "simple":
            supports = [{"x": 0, "type": "pin"}, {"x": span, "type": "roller"}]
        elif st == "cantilever":
            supports = [{"x": 0, "type": "fixed"}]
        elif st == "fixed_fixed":
            supports = [{"x": 0, "type": "fixed"}, {"x": span, "type": "fixed"}]
        elif st in ("fixed_pin", "propped_cantilever"):
            supports = [{"x": 0, "type": "fixed"}, {"x": span, "type": "roller"}]
        else:
            supports = [{"x": 0, "type": "pin"}, {"x": span, "type": "roller"}]
    return supports


def _draw_support(ax, x, y, stype, size=0.3):
    """지점 심볼 그리기"""
    if stype == "pin":
        tri = plt.Polygon(
            [[x, y], [x - size * 0.5, y - size], [x + size * 0.5, y - size]],
            closed=True, facecolor='white', edgecolor='black', linewidth=1.5, zorder=5
        )
        ax.add_patch(tri)
        ax.plot([x - size * 0.7, x + size * 0.7], [y - size, y - size], 'k-', linewidth=1.5)
    elif stype == "roller":
        tri = plt.Polygon(
            [[x, y], [x - size * 0.5, y - size * 0.7], [x + size * 0.5, y - size * 0.7]],
            closed=True, facecolor='white', edgecolor='black', linewidth=1.5, zorder=5
        )
        ax.add_patch(tri)
        circle = plt.Circle((x, y - size * 0.7 - size * 0.2), size * 0.2,
                             facecolor='white', edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.plot([x - size * 0.7, x + size * 0.7], [y - size, y - size], 'k-', linewidth=1.5)
    elif stype == "fixed":
        rect_w = size * 0.3
        rect_h = size * 1.2
        rect = patches.FancyBboxPatch(
            (x - rect_w / 2, y - rect_h / 2), rect_w, rect_h,
            boxstyle="square,pad=0", facecolor='gray', edgecolor='black',
            linewidth=1.5, zorder=5
        )
        ax.add_patch(rect)
        for dy in np.linspace(-rect_h / 2, rect_h / 2, 6):
            ax.plot([x - rect_w / 2 - size * 0.2, x - rect_w / 2],
                    [y + dy - size * 0.1, y + dy + size * 0.1], 'k-', linewidth=0.8)


def _draw_loads_mpl(ax, load_info: list, span_total: float, beam_y: float = 0.0):
    """matplotlib: 구조형상도에 하중 표시"""
    if not load_info:
        return
    arrow_h = span_total * 0.06  # 화살표 높이
    arrow_top = beam_y + arrow_h * 1.8  # 화살표 시작 y (위에서 아래로)
    color_uniform = '#D32F2F'
    color_point = '#1565C0'

    for ld in load_info:
        lt = ld.get("type", "uniform")
        if lt == "uniform":
            start, end = ld["start"], ld["end"]
            w = ld["value"]
            n_arrows = max(int((end - start) / span_total * 12), 3)
            xs = np.linspace(start, end, n_arrows)
            for xp in xs:
                ax.annotate('', xy=(xp, beam_y), xytext=(xp, arrow_top),
                            arrowprops=dict(arrowstyle='->', color=color_uniform, lw=1.2))
            # 상단 수평선
            ax.plot([start, end], [arrow_top, arrow_top], color=color_uniform, lw=1.5)
            # 하중값 표시
            mid = (start + end) / 2
            ax.text(mid, arrow_top + arrow_h * 0.3, f'{w} kN/m',
                    ha='center', va='bottom', fontsize=8, color=color_uniform, fontweight='bold')

        elif lt == "point":
            loc = ld["location"]
            pv = ld["value"]
            ax.annotate('', xy=(loc, beam_y), xytext=(loc, arrow_top * 1.3),
                        arrowprops=dict(arrowstyle='->', color=color_point, lw=2.0))
            ax.text(loc, arrow_top * 1.3 + arrow_h * 0.2, f'{pv} kN',
                    ha='center', va='bottom', fontsize=8, color=color_point, fontweight='bold')

        elif lt == "triangular":
            start, end = ld["start"], ld["end"]
            w_start = ld["value"]
            w_end = ld.get("value_end", 0.0)
            w_max = max(abs(w_start), abs(w_end))
            if w_max == 0:
                continue
            n_arrows = max(int((end - start) / span_total * 12), 3)
            xs = np.linspace(start, end, n_arrows)
            for xp in xs:
                ratio = (xp - start) / (end - start) if end != start else 0
                w_local = w_start + (w_end - w_start) * ratio
                h_local = arrow_h * 1.8 * abs(w_local) / w_max
                if h_local > arrow_h * 0.2:
                    ax.annotate('', xy=(xp, beam_y), xytext=(xp, beam_y + h_local),
                                arrowprops=dict(arrowstyle='->', color=color_uniform, lw=1.0))
            # 경사선 (상단 연결)
            y_start = beam_y + arrow_h * 1.8 * abs(w_start) / w_max
            y_end_pt = beam_y + arrow_h * 1.8 * abs(w_end) / w_max
            ax.plot([start, end], [y_start, y_end_pt], color=color_uniform, lw=1.5)
            mid = (start + end) / 2
            ax.text(mid, max(y_start, y_end_pt) + arrow_h * 0.3,
                    f'{w_start}→{w_end} kN/m', ha='center', va='bottom',
                    fontsize=8, color=color_uniform, fontweight='bold')


def _draw_loads_plotly(fig, load_info: list, span_total: float, beam_y: float = 0.0, row: int = 1, col: int = 1):
    """plotly: 구조형상도에 하중 표시"""
    if not load_info:
        return
    arrow_h = span_total * 0.06
    arrow_top = beam_y + arrow_h * 1.8
    color_uniform = '#D32F2F'
    color_point = '#1565C0'

    for ld in load_info:
        lt = ld.get("type", "uniform")
        if lt == "uniform":
            start, end = ld["start"], ld["end"]
            w = ld["value"]
            n_arrows = max(int((end - start) / span_total * 12), 3)
            xs = np.linspace(start, end, n_arrows)
            # 상단 수평선
            fig.add_trace(go.Scatter(
                x=[start, end], y=[arrow_top, arrow_top],
                mode='lines', line=dict(color=color_uniform, width=2),
                hoverinfo='skip', showlegend=False,
            ), row=row, col=col)
            # 화살표 (수직선 + annotation)
            for xp in xs:
                fig.add_trace(go.Scatter(
                    x=[xp, xp], y=[arrow_top, beam_y + arrow_h * 0.15],
                    mode='lines', line=dict(color=color_uniform, width=1.2),
                    hoverinfo='skip', showlegend=False,
                ), row=row, col=col)
            # 화살표 머리 (삼각형 마커로)
            fig.add_trace(go.Scatter(
                x=list(xs), y=[beam_y + arrow_h * 0.05] * len(xs),
                mode='markers', marker=dict(symbol='triangle-down', size=7, color=color_uniform),
                hoverinfo='text', hovertext=f'{w} kN/m', showlegend=False,
            ), row=row, col=col)
            # 하중값 텍스트
            fig.add_annotation(
                x=(start + end) / 2, y=arrow_top + arrow_h * 0.4,
                text=f'{w} kN/m', showarrow=False,
                font=dict(size=10, color=color_uniform, family='Arial Black'),
                row=row, col=col,
            )

        elif lt == "point":
            loc = ld["location"]
            pv = ld["value"]
            top_y = arrow_top * 1.3
            fig.add_trace(go.Scatter(
                x=[loc, loc], y=[top_y, beam_y + arrow_h * 0.15],
                mode='lines', line=dict(color=color_point, width=2.5),
                hoverinfo='skip', showlegend=False,
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[loc], y=[beam_y + arrow_h * 0.05],
                mode='markers', marker=dict(symbol='triangle-down', size=10, color=color_point),
                hoverinfo='text', hovertext=f'{pv} kN', showlegend=False,
            ), row=row, col=col)
            fig.add_annotation(
                x=loc, y=top_y + arrow_h * 0.3,
                text=f'{pv} kN', showarrow=False,
                font=dict(size=10, color=color_point, family='Arial Black'),
                row=row, col=col,
            )

        elif lt == "triangular":
            start, end = ld["start"], ld["end"]
            w_start = ld["value"]
            w_end = ld.get("value_end", 0.0)
            w_max = max(abs(w_start), abs(w_end))
            if w_max == 0:
                continue
            n_arrows = max(int((end - start) / span_total * 12), 3)
            xs = np.linspace(start, end, n_arrows)
            # 경사 상단선
            y_s = beam_y + arrow_h * 1.8 * abs(w_start) / w_max
            y_e = beam_y + arrow_h * 1.8 * abs(w_end) / w_max
            fig.add_trace(go.Scatter(
                x=[start, end], y=[y_s, y_e],
                mode='lines', line=dict(color=color_uniform, width=2),
                hoverinfo='skip', showlegend=False,
            ), row=row, col=col)
            for xp in xs:
                ratio = (xp - start) / (end - start) if end != start else 0
                w_local = w_start + (w_end - w_start) * ratio
                h_local = arrow_h * 1.8 * abs(w_local) / w_max
                if h_local > arrow_h * 0.2:
                    fig.add_trace(go.Scatter(
                        x=[xp, xp], y=[beam_y + h_local, beam_y + arrow_h * 0.15],
                        mode='lines', line=dict(color=color_uniform, width=1.0),
                        hoverinfo='skip', showlegend=False,
                    ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=list(xs), y=[beam_y + arrow_h * 0.05] * len(xs),
                mode='markers', marker=dict(symbol='triangle-down', size=6, color=color_uniform),
                hoverinfo='skip', showlegend=False,
            ), row=row, col=col)
            fig.add_annotation(
                x=(start + end) / 2, y=max(y_s, y_e) + arrow_h * 0.4,
                text=f'{w_start}→{w_end} kN/m', showarrow=False,
                font=dict(size=10, color=color_uniform, family='Arial Black'),
                row=row, col=col,
            )


def _get_section_prefix(section_name: str) -> str:
    """단면 이름에서 prefix 추출"""
    for prefix in ["TFC-", "PFC-", "FB-", "H-", "I-", "T-", "L-", "○-", "□-"]:
        if section_name.startswith(prefix):
            return prefix
    return ""


def _draw_section_sketch(ax, section_name: str):
    """단면 스케치 그리기 (matplotlib axes에)"""
    prefix = _get_section_prefix(section_name)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    color = '#4A90D9'
    lw = 2.5

    if prefix in ("H-", "I-"):
        # H/I형강: 좌우 대칭 I형
        fw = 0.8   # 플랜지 폭
        fh = 0.12  # 플랜지 두께
        wh = 0.8   # 웹 높이
        ww = 0.08  # 웹 두께
        # 상부 플랜지
        ax.add_patch(patches.Rectangle((-fw/2, wh/2), fw, fh, facecolor=color, edgecolor='black', lw=lw))
        # 하부 플랜지
        ax.add_patch(patches.Rectangle((-fw/2, -wh/2-fh), fw, fh, facecolor=color, edgecolor='black', lw=lw))
        # 웹
        ax.add_patch(patches.Rectangle((-ww/2, -wh/2), ww, wh, facecolor=color, edgecolor='black', lw=lw))

    elif prefix in ("TFC-", "PFC-"):
        # ㄷ형강: C-channel
        fw = 0.5
        fh = 0.1
        wh = 0.8
        ww = 0.1
        # 웹 (좌측 수직)
        ax.add_patch(patches.Rectangle((-fw/2, -wh/2-fh), ww, wh+2*fh, facecolor=color, edgecolor='black', lw=lw))
        # 상부 플랜지 (우측으로)
        ax.add_patch(patches.Rectangle((-fw/2+ww, wh/2), fw-ww, fh, facecolor=color, edgecolor='black', lw=lw))
        # 하부 플랜지
        ax.add_patch(patches.Rectangle((-fw/2+ww, -wh/2-fh), fw-ww, fh, facecolor=color, edgecolor='black', lw=lw))

    elif prefix == "T-":
        # T형강
        fw = 0.8
        fh = 0.12
        wh = 0.6
        ww = 0.1
        ax.add_patch(patches.Rectangle((-fw/2, wh/2), fw, fh, facecolor=color, edgecolor='black', lw=lw))
        ax.add_patch(patches.Rectangle((-ww/2, -wh/2), ww, wh, facecolor=color, edgecolor='black', lw=lw))

    elif prefix == "L-":
        # ㄱ형강: L-angle
        leg = 0.8
        t = 0.12
        # 수직 leg
        ax.add_patch(patches.Rectangle((-leg/2, -leg/2), t, leg, facecolor=color, edgecolor='black', lw=lw))
        # 수평 leg
        ax.add_patch(patches.Rectangle((-leg/2, -leg/2), leg, t, facecolor=color, edgecolor='black', lw=lw))

    elif prefix == "○-":
        # 원형강관
        outer = plt.Circle((0, 0), 0.5, facecolor=color, edgecolor='black', lw=lw)
        inner = plt.Circle((0, 0), 0.38, facecolor='white', edgecolor='black', lw=lw)
        ax.add_patch(outer)
        ax.add_patch(inner)

    elif prefix == "□-":
        # 중공형강 (사각)
        outer = patches.Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor=color, edgecolor='black', lw=lw)
        inner = patches.Rectangle((-0.38, -0.38), 0.76, 0.76, facecolor='white', edgecolor='black', lw=lw)
        ax.add_patch(outer)
        ax.add_patch(inner)

    elif prefix == "FB-":
        # 구평형강 (플랫바)
        ax.add_patch(patches.Rectangle((-0.6, -0.15), 1.2, 0.3, facecolor=color, edgecolor='black', lw=lw))

    else:
        # 기본: 사각형
        ax.add_patch(patches.Rectangle((-0.4, -0.5), 0.8, 1.0, facecolor=color, edgecolor='black', lw=lw))


# ============================================================
# Matplotlib (PNG)
# ============================================================

def plot_beam_results(result, output_path: Optional[str] = None,
                      fmt: str = "png", dpi: int = 150) -> str:
    """matplotlib로 해석 결과 시각화"""
    x = result.node_positions
    if not x:
        raise ValueError("시각화 데이터가 없습니다 (node_positions 비어있음)")

    disps = result.displacements
    moments = result.moments
    shears_data = result.shears
    span_total = x[-1]

    section_name = getattr(result, 'section_name', '') or ''
    material_name = getattr(result, 'material_name', '') or ''

    # 레이아웃: 상단(2열: 구조형상 + 단면정보), SFD, BMD, Displacement
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.3, 1, 1, 1],
                          width_ratios=[3, 1], hspace=0.3, wspace=0.05)

    # --- 1) 구조형상도 (좌상단) ---
    ax_struct = fig.add_subplot(gs[0, 0])
    ax_struct.set_title('Structure', fontsize=11, fontweight='bold')
    ax_struct.plot(x, [0] * len(x), 'k-', linewidth=3, zorder=3)
    supports = _get_support_info(result)
    hinge_locs = getattr(result, 'hinge_locations', []) or []
    hinge_locs_set = set(round(h, 3) for h in hinge_locs)  # 비교를 위해 반올림
    for s in supports:
        # 힌지가 있는 위치는 지점 심볼 건너뜀
        if round(s["x"], 3) in hinge_locs_set:
            continue
        _draw_support(ax_struct, s["x"], 0, s["type"], size=span_total * 0.03)
    # 힌지 마커 표시
    for h_x in hinge_locs:
        hinge_radius = span_total * 0.015
        hinge_circle = plt.Circle((h_x, 0), hinge_radius,
                                   facecolor='white', edgecolor='#D32F2F',
                                   linewidth=2, zorder=6)
        ax_struct.add_patch(hinge_circle)
    # 하중 표시
    _load_info = getattr(result, 'load_info', []) or []
    _draw_loads_mpl(ax_struct, _load_info, span_total, beam_y=0.0)
    # 경간 길이 표시
    dim_y = -span_total * 0.07  # 보 아래쪽
    if hasattr(result, 'num_spans') and result.spans:
        x_start = 0.0
        for sp_len in result.spans:
            mid_x = x_start + sp_len / 2
            ax_struct.annotate('', xy=(x_start, dim_y), xytext=(x_start + sp_len, dim_y),
                               arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
            ax_struct.text(mid_x, dim_y - span_total * 0.015, f'{sp_len:.1f} m',
                           ha='center', va='top', fontsize=9, color='#555', fontweight='bold')
            x_start += sp_len
    else:
        mid_x = span_total / 2
        ax_struct.annotate('', xy=(0, dim_y), xytext=(span_total, dim_y),
                           arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
        ax_struct.text(mid_x, dim_y - span_total * 0.015, f'{span_total:.1f} m',
                       ha='center', va='top', fontsize=9, color='#555', fontweight='bold')
    ax_struct.set_xlim(-span_total * 0.05, span_total * 1.05)
    ax_struct.set_ylim(-span_total * 0.14, span_total * 0.18)
    ax_struct.set_aspect('equal')
    ax_struct.axis('off')

    # --- 2) 단면 정보 (우상단) ---
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    # 단면 스케치
    # 상단 절반에 스케치, 하단 절반에 텍스트
    ax_sketch = ax_info.inset_axes([0.1, 0.35, 0.8, 0.6])
    _draw_section_sketch(ax_sketch, section_name)

    # 텍스트 정보
    info_text = f"Section: {section_name}\nMaterial: {material_name}"
    ax_info.text(0.5, 0.18, info_text, transform=ax_info.transAxes,
                 fontsize=10, ha='center', va='center',
                 fontweight='bold', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#CCCCCC'))

    # =================================================================
    # Label priority & collision avoidance helper (PNG / Matplotlib)
    # -----------------------------------------------------------------
    # Priority: (1) global max/min → (2) internal-support hogging
    #           → (3) span sagging max → (4) skip (hover-only in HTML)
    #
    # Internal-support moment definition  ── **Option A**
    #   Diagram shows ONE representative value per support
    #   (= section moment at continuous cross-section).
    #   BL / BR split values are provided in the console table and
    #   in the HTML hover tooltip only.
    # =================================================================

    def _mpl_collision_offset(new_xy, placed, dy_base=14, dx_nudge=8):
        """Return (dx, dy) offset-points that avoid *placed* labels.

        *placed* is a list of (x_data, y_data, dx_pts, dy_pts) already drawn.
        Uses staircase y-offset (+dy, -dy, +2dy …) and small dx nudge.
        """
        nx, ny = new_xy
        dx, dy = 0, dy_base
        for step in range(6):
            sign = 1 if step % 2 == 0 else -1
            trial_dy = dy_base * (step // 2 + 1) * sign
            trial_dx = dx_nudge * (step // 2) * (1 if step % 2 == 0 else -1)
            conflict = False
            for (px, py, pdx, pdy) in placed:
                # rough screen-distance check (offset points are ~1 px)
                if abs((nx - px) / max(1, abs(px)) * 100) < 3 and abs(trial_dy - pdy) < dy_base * 0.8:
                    conflict = True
                    break
            if not conflict:
                return trial_dx, trial_dy
            dx, dy = trial_dx, trial_dy
        return dx, dy  # fallback

    # --- 3) SFD ---
    ax_sfd = fig.add_subplot(gs[1, :])
    ax_sfd.set_title('Shear Force Diagram (kN)', fontsize=11, fontweight='bold')
    ax_sfd.fill_between(x, shears_data, 0, alpha=0.3, color='red')
    ax_sfd.plot(x, shears_data, 'r-', linewidth=1.5)
    ax_sfd.axhline(y=0, color='black', linewidth=0.8)
    abs_shears = [abs(v) for v in shears_data]
    max_v_idx = abs_shears.index(max(abs_shears))
    max_v_val = shears_data[max_v_idx]
    max_v_x = x[max_v_idx]
    ax_sfd.annotate(f'{max_v_val:.1f}', xy=(max_v_x, max_v_val),
                    fontsize=9, color='red', fontweight='bold',
                    textcoords="offset points", xytext=(12, 18 if max_v_val >= 0 else -22),
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax_sfd.set_xlim(-span_total * 0.05, span_total * 1.05)
    ax_sfd.set_ylabel('V (kN)')
    ax_sfd.grid(True, alpha=0.3)
    ax_sfd.text(0.99, 0.97, 'V > 0: ↑ on left face', transform=ax_sfd.transAxes,
                fontsize=7, ha='right', va='top', color='#888', style='italic')

    # --- 4) BMD  (priority-based labelling) ---
    ax_bmd = fig.add_subplot(gs[2, :])
    ax_bmd.set_title('Bending Moment Diagram (kN·m)', fontsize=11, fontweight='bold')
    ax_bmd.fill_between(x, moments, 0, alpha=0.3, color='blue')
    ax_bmd.plot(x, moments, 'b-', linewidth=1.5)
    ax_bmd.axhline(y=0, color='black', linewidth=0.8)

    _bmd_placed = []  # collision tracker: [(x, y, dx_pts, dy_pts), ...]

    # ── P1: global max |M| ──
    abs_moments = [abs(m) for m in moments]
    max_m_idx = abs_moments.index(max(abs_moments))
    max_m_val = moments[max_m_idx]
    max_m_x = x[max_m_idx]

    # ── P2: internal-support hogging moments (Option A: 1 value) ──
    _internal_support_labels = []  # [(loc, m_val, label_str), ...]
    if hasattr(result, 'reactions'):
        reactions = getattr(result, 'reactions', [])
        for idx_r, r in enumerate(reactions):
            if idx_r == 0 or idx_r == len(reactions) - 1:
                continue
            loc = r.get("location", 0.0)
            m_left = r.get("moment_left_kNm", None)
            m_right = r.get("moment_right_kNm", None)
            if m_left is not None and m_right is not None:
                if abs(m_left) > 0.1 or abs(m_right) > 0.1:
                    label = chr(65 + idx_r)
                    rep_val = m_left  # representative (Option A)
                    closest_idx = min(range(len(x)), key=lambda i: abs(x[i] - loc))
                    m_val = moments[closest_idx]
                    _internal_support_labels.append((loc, m_val, f'{label}:{rep_val:.1f}'))

    # (P3 span sagging labels: omitted from PNG to keep diagram clean.
    #  Values are in the console summary table.)

    # Draw P1 — global max (only if not at an internal support)
    _is_at_support = any(abs(max_m_x - sl) < span_total * 0.02
                         for sl, _, _ in _internal_support_labels)
    if not _is_at_support:
        dy0 = 18 if max_m_val >= 0 else -22
        dx0, dy0 = _mpl_collision_offset((max_m_x, max_m_val), _bmd_placed, dy_base=abs(dy0))
        if max_m_val < 0:
            dy0 = -abs(dy0)
        ax_bmd.annotate(f'{max_m_val:.1f}', xy=(max_m_x, max_m_val),
                        fontsize=9, color='blue', fontweight='bold',
                        textcoords="offset points", xytext=(dx0 + 10, dy0),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=0.8))
        _bmd_placed.append((max_m_x, max_m_val, dx0 + 10, dy0))

    # Draw P2 — internal-support hogging
    for loc, m_val, txt in _internal_support_labels:
        dy_base = -20 if m_val < 0 else 14
        dx, dy = _mpl_collision_offset((loc, m_val), _bmd_placed, dy_base=abs(dy_base))
        if m_val < 0:
            dy = -abs(dy)
        ax_bmd.annotate(txt, xy=(loc, m_val),
                        fontsize=8, color='darkred', fontweight='bold',
                        textcoords="offset points", xytext=(dx, dy), ha='center')
        _bmd_placed.append((loc, m_val, dx, dy))

    ax_bmd.set_xlim(-span_total * 0.05, span_total * 1.05)
    ax_bmd.set_ylabel('M (kN·m)')
    ax_bmd.grid(True, alpha=0.3)
    ax_bmd.text(0.99, 0.97, 'M > 0: sagging (tension at bottom)', transform=ax_bmd.transAxes,
                fontsize=7, ha='right', va='top', color='#888', style='italic')
    # Option A 정의 범례
    ax_bmd.text(0.01, 0.97, 'Support label = section moment (BL/BR in table)',
                transform=ax_bmd.transAxes, fontsize=6.5, ha='left', va='top',
                color='#999', style='italic')

    # --- 5) Displacement ---
    ax_disp = fig.add_subplot(gs[3, :])
    ax_disp.set_title('Displacement Diagram (mm)', fontsize=11, fontweight='bold')
    ax_disp.fill_between(x, disps, 0, alpha=0.3, color='green')
    ax_disp.plot(x, disps, 'g-', linewidth=1.5)
    ax_disp.axhline(y=0, color='black', linewidth=0.8)
    abs_disps = [abs(d) for d in disps]
    max_d_idx = abs_disps.index(max(abs_disps))
    max_d_val = disps[max_d_idx]
    max_d_x = x[max_d_idx]
    ax_disp.annotate(f'{abs(max_d_val):.3f}', xy=(max_d_x, max_d_val),
                     fontsize=9, color='green', fontweight='bold',
                     textcoords="offset points", xytext=(12, -22 if max_d_val <= 0 else 18),
                     arrowprops=dict(arrowstyle='->', color='green', lw=0.8))
    ax_disp.set_xlim(-span_total * 0.05, span_total * 1.05)
    ax_disp.set_xlabel('Position (m)')
    ax_disp.set_ylabel('δ (mm)')
    ax_disp.grid(True, alpha=0.3)
    ax_disp.text(0.99, 0.97, 'δ < 0: downward', transform=ax_disp.transAxes,
                 fontsize=7, ha='right', va='top', color='#888', style='italic')

    # 평형 검증 패널
    try:
        from core.verification import verify_equilibrium
        eq = verify_equilibrium(result)
        marks = []
        for key in ["sum_vertical", "sum_moment", "shear_jumps"]:
            if key in eq:
                s = eq[key]
                icon = "✓" if s["status"] == "OK" else "✗"
                marks.append(f"{icon} {s['description']} ({s['status']})")
        eq_text = "  |  ".join(marks)
        fig.text(0.5, 0.005, eq_text, ha='center', va='bottom', fontsize=8,
                 color='#333', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F8F8', edgecolor='#CCC'))
    except Exception:
        pass

    # 처짐 판정 패널
    try:
        defl_ratio = getattr(result, 'deflection_limit_ratio', 300)
        span_results = getattr(result, 'span_results', [])
        if span_results:
            defl_parts = []
            for sr in span_results:
                sp_len = sr.get("span_length", 0)
                d_max = sr.get("max_displacement_mm", 0)
                d_allow = sr.get("delta_allow_mm", sp_len * 1000 / defl_ratio if defl_ratio > 0 else 0)
                status = sr.get("deflection_status", "OK" if d_max <= d_allow else "NG")
                icon = "✓" if status == "OK" else "✗"
                defl_parts.append(f"{icon} Span{sr.get('span_index',0)+1}({sp_len}m): δ={d_max:.2f}mm, L/{defl_ratio}={d_allow:.1f}mm → {status}")
            defl_text = "  |  ".join(defl_parts)
            fig.text(0.5, -0.015, f"Deflection check (L/{defl_ratio}): {defl_text}", ha='center', va='top', fontsize=7,
                     color='#333', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FFF0', edgecolor='#9C9'))
    except Exception:
        pass

    # 모델 정보 패널
    try:
        e_gpa = round(getattr(result, 'E_MPa', 0) / 1000, 1)
        ix = getattr(result, 'Ix_mm4', 0)
        zx = round(getattr(result, 'Zx_mm3', 0), 1)
        n_elem = getattr(result, 'num_elements_per_span', getattr(result, 'num_elements', 0))
        model_text = f"E={e_gpa}GPa | Ix={ix:.3e}mm⁴ | Zx={zx:.0f}mm³ | Element=elasticBeamColumn | Elem/span={n_elem}"
        fig.text(0.5, -0.035, model_text, ha='center', va='top', fontsize=6.5,
                 color='#666', fontfamily='monospace')
    except Exception:
        pass

    fig.suptitle('Beam Analysis Results', fontsize=14, fontweight='bold', y=0.99)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=f'.{fmt}')
        os.close(fd)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', format=fmt)
    plt.close(fig)
    return output_path


# ============================================================
# Plotly (HTML)
# ============================================================

def _section_sketch_svg(section_name: str) -> str:
    """단면 스케치를 SVG 문자열로 생성 (plotly annotation용)"""
    prefix = _get_section_prefix(section_name)
    color = '#4A90D9'
    stroke = '#333333'
    sw = 2  # stroke-width

    # viewBox: -60 -60 120 120 (center at 0,0)
    shapes = []

    if prefix in ("H-", "I-"):
        fw, fh, wh, ww = 50, 8, 50, 6
        shapes.append(f'<rect x="{-fw//2}" y="{-wh//2-fh}" width="{fw}" height="{fh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<rect x="{-fw//2}" y="{wh//2}" width="{fw}" height="{fh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<rect x="{-ww//2}" y="{-wh//2}" width="{ww}" height="{wh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
    elif prefix in ("TFC-", "PFC-"):
        fw, fh, wh, ww = 35, 7, 50, 7
        shapes.append(f'<rect x="{-fw//2}" y="{-wh//2-fh}" width="{ww}" height="{wh+2*fh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<rect x="{-fw//2+ww}" y="{-wh//2-fh}" width="{fw-ww}" height="{fh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<rect x="{-fw//2+ww}" y="{wh//2}" width="{fw-ww}" height="{fh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
    elif prefix == "T-":
        fw, fh, wh, ww = 50, 8, 40, 7
        shapes.append(f'<rect x="{-fw//2}" y="{-wh//2-fh}" width="{fw}" height="{fh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<rect x="{-ww//2}" y="{-wh//2}" width="{ww}" height="{wh}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
    elif prefix == "L-":
        leg, t = 50, 8
        shapes.append(f'<rect x="{-leg//2}" y="{-leg//2}" width="{t}" height="{leg}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<rect x="{-leg//2}" y="{leg//2-t}" width="{leg}" height="{t}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
    elif prefix == "○-":
        shapes.append(f'<circle cx="0" cy="0" r="35" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<circle cx="0" cy="0" r="26" fill="white" stroke="{stroke}" stroke-width="{sw}"/>')
    elif prefix == "□-":
        shapes.append(f'<rect x="-35" y="-35" width="70" height="70" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        shapes.append(f'<rect x="-26" y="-26" width="52" height="52" fill="white" stroke="{stroke}" stroke-width="{sw}"/>')
    elif prefix == "FB-":
        shapes.append(f'<rect x="-40" y="-10" width="80" height="20" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
    else:
        shapes.append(f'<rect x="-25" y="-35" width="50" height="70" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')

    inner = "\n".join(shapes)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-60 -60 120 120" width="120" height="120">{inner}</svg>'


def _section_sketch_base64(section_name: str) -> str:
    """단면 스케치를 base64 PNG로 생성 (plotly 이미지 삽입용)"""
    fig_s, ax_s = plt.subplots(figsize=(1.5, 1.5), dpi=120)
    _draw_section_sketch(ax_s, section_name)
    buf = io.BytesIO()
    fig_s.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=120)
    plt.close(fig_s)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def plot_beam_results_interactive(result, output_path: Optional[str] = None) -> str:
    """plotly로 인터랙티브 시각화 → HTML 파일"""
    x = result.node_positions
    if not x:
        raise ValueError("시각화 데이터가 없습니다 (node_positions 비어있음)")

    disps = result.displacements
    moments = result.moments
    shears_data = result.shears
    span_total = x[-1]

    section_name = getattr(result, 'section_name', '') or ''
    material_name = getattr(result, 'material_name', '') or ''

    supports = _get_support_info(result)

    # 상단 2열 (구조형상 + 단면정보) + 하단 3행 (SFD, BMD, Displacement)
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=('Structure', 'Section Info',
                        'Shear Force Diagram (kN)', None,
                        'Bending Moment Diagram (kN·m)', None,
                        'Displacement Diagram (mm)', None),
        column_widths=[0.75, 0.25],
        vertical_spacing=0.06,
        row_heights=[0.18, 0.27, 0.27, 0.27],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
        ],
    )

    # 1) 구조형상도 (row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]], y=[0, 0], mode='lines', name='Beam',
        line=dict(color='black', width=4), hoverinfo='skip',
    ), row=1, col=1)

    # 힌지 위치 집합 (지점 심볼 제외용)
    hinge_locs = getattr(result, 'hinge_locations', []) or []
    hinge_locs_set = set(round(h, 3) for h in hinge_locs)

    for s in supports:
        sx, st = s["x"], s["type"]
        # 힌지가 있는 위치는 지점 심볼 건너뜀
        if round(sx, 3) in hinge_locs_set:
            continue
        sz = span_total * 0.03
        if st == "pin":
            fig.add_trace(go.Scatter(
                x=[sx, sx - sz * 0.5, sx + sz * 0.5, sx],
                y=[0, -sz, -sz, 0],
                mode='lines', line=dict(color='black', width=2),
                fill='toself', fillcolor='white', hoverinfo='text',
                hovertext=f'Pin @ {sx:.1f}m',
            ), row=1, col=1)
        elif st == "roller":
            fig.add_trace(go.Scatter(
                x=[sx, sx - sz * 0.5, sx + sz * 0.5, sx],
                y=[0, -sz * 0.7, -sz * 0.7, 0],
                mode='lines', line=dict(color='black', width=2),
                fill='toself', fillcolor='white', hoverinfo='text',
                hovertext=f'Roller @ {sx:.1f}m',
            ), row=1, col=1)
            r = sz * 0.2
            cy = -sz * 0.7 - r
            theta = [i * 2 * math.pi / 20 for i in range(21)]
            fig.add_trace(go.Scatter(
                x=[sx + r * math.cos(t) for t in theta],
                y=[cy + r * math.sin(t) for t in theta],
                mode='lines', line=dict(color='black', width=1.5), hoverinfo='skip',
            ), row=1, col=1)
        elif st == "fixed":
            rw, rh = sz * 0.3, sz * 1.2
            fig.add_trace(go.Scatter(
                x=[sx - rw/2, sx + rw/2, sx + rw/2, sx - rw/2, sx - rw/2],
                y=[-rh/2, -rh/2, rh/2, rh/2, -rh/2],
                mode='lines', line=dict(color='black', width=2),
                fill='toself', fillcolor='gray', hoverinfo='text',
                hovertext=f'Fixed @ {sx:.1f}m',
            ), row=1, col=1)

    # 힌지 마커 표시
    for h_x in hinge_locs:
        hinge_r = span_total * 0.015
        theta = [i * 2 * math.pi / 20 for i in range(21)]
        fig.add_trace(go.Scatter(
            x=[h_x + hinge_r * math.cos(t) for t in theta],
            y=[hinge_r * math.sin(t) for t in theta],
            mode='lines', line=dict(color='#D32F2F', width=2),
            fill='toself', fillcolor='white', hoverinfo='text',
            hovertext=f'Hinge @ {h_x:.2f}m<br>(Moment Release)',
            showlegend=False,
        ), row=1, col=1)

    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)
    y_range = span_total * 0.12
    # 하중 표시
    _load_info = getattr(result, 'load_info', []) or []
    _draw_loads_plotly(fig, _load_info, span_total, beam_y=0.0, row=1, col=1)
    # 경간 길이 표시
    dim_y = -y_range * 0.55
    if hasattr(result, 'num_spans') and result.spans:
        x_start = 0.0
        for sp_len in result.spans:
            mid_x = x_start + sp_len / 2
            # 양쪽 끝 짧은 수직선 + 수평선
            fig.add_trace(go.Scatter(
                x=[x_start, x_start + sp_len], y=[dim_y, dim_y],
                mode='lines', line=dict(color='#555', width=1.2, dash='solid'),
                hoverinfo='skip', showlegend=False,
            ), row=1, col=1)
            for xp in [x_start, x_start + sp_len]:
                fig.add_trace(go.Scatter(
                    x=[xp, xp], y=[dim_y - y_range * 0.06, dim_y + y_range * 0.06],
                    mode='lines', line=dict(color='#555', width=1.2),
                    hoverinfo='skip', showlegend=False,
                ), row=1, col=1)
            fig.add_annotation(
                x=mid_x, y=dim_y - y_range * 0.1,
                text=f'{sp_len:.1f} m', showarrow=False,
                font=dict(size=11, color='#555', family='Arial Black'),
                row=1, col=1,
            )
            x_start += sp_len
    else:
        mid_x = span_total / 2
        fig.add_trace(go.Scatter(
            x=[0, span_total], y=[dim_y, dim_y],
            mode='lines', line=dict(color='#555', width=1.2),
            hoverinfo='skip', showlegend=False,
        ), row=1, col=1)
        for xp in [0, span_total]:
            fig.add_trace(go.Scatter(
                x=[xp, xp], y=[dim_y - y_range * 0.06, dim_y + y_range * 0.06],
                mode='lines', line=dict(color='#555', width=1.2),
                hoverinfo='skip', showlegend=False,
            ), row=1, col=1)
        fig.add_annotation(
            x=mid_x, y=dim_y - y_range * 0.1,
            text=f'{span_total:.1f} m', showarrow=False,
            font=dict(size=11, color='#555', family='Arial Black'),
            row=1, col=1,
        )
    fig.update_yaxes(range=[-y_range * 1.2, y_range * 1.5], row=1, col=1)

    # 2) 단면 정보 (row=1, col=2) — 스케치 이미지 + 텍스트
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    fig.update_xaxes(range=[0, 1], row=1, col=2)

    if section_name or material_name:
        # 단면 스케치 이미지 삽입 (우상단 subplot 영역에)
        if section_name:
            img_uri = _section_sketch_base64(section_name)
            fig.add_layout_image(
                dict(
                    source=img_uri,
                    xref="x2", yref="y2",
                    x=0.5, y=0.95,
                    sizex=0.8, sizey=0.6,
                    xanchor="center", yanchor="top",
                    layer="above",
                )
            )
        # 텍스트 (이미지 아래)
        info_lines = []
        if section_name:
            info_lines.append(f"Section: {section_name}")
        if material_name:
            info_lines.append(f"Material: {material_name}")
        info_text = "<br>".join(info_lines)
        fig.add_annotation(
            text=info_text,
            xref="x2", yref="y2",
            x=0.5, y=0.25, showarrow=False,
            font=dict(size=11, color='#333'),
            align='center', xanchor='center', yanchor='top',
            bgcolor='#F0F4F8', bordercolor='#CCCCCC', borderwidth=1,
            borderpad=6,
        )

    # =================================================================
    # HTML (Plotly) — Priority-based labelling with hover details
    # -----------------------------------------------------------------
    # Label policy (same as PNG):
    #   P1: global max/min  →  P2: internal-support hogging
    #   P3: span sagging max  →  P4: hover only
    #
    # Internal-support moments: **Option A**
    #   Visible label = support letter only ("B", "C").
    #   Hover shows BL / BR split + representative section moment.
    # =================================================================

    # 3) SFD (row=2, colspan=2) — with hovertemplate
    fig.add_trace(go.Scatter(
        x=x, y=shears_data, mode='lines', name='Shear',
        line=dict(color='red', width=2),
        fill='tozeroy', fillcolor='rgba(255,0,0,0.15)',
        hovertemplate='x=%{x:.2f}m<br>V=%{y:.2f} kN<extra></extra>',
    ), row=2, col=1)

    # 4) BMD (row=3, colspan=2) — with hovertemplate
    fig.add_trace(go.Scatter(
        x=x, y=moments, mode='lines', name='Moment',
        line=dict(color='blue', width=2),
        fill='tozeroy', fillcolor='rgba(0,0,255,0.15)',
        hovertemplate=(
            'x=%{x:.2f}m<br>M=%{y:.2f} kN·m<br>'
            '<span style="color:#999;font-size:10px">'
            'Sign: M&gt;0 sagging (tension at bottom). '
            'End moments follow element local convention.</span>'
            '<extra></extra>'
        ),
    ), row=3, col=1)

    # 5) Displacement (row=4, colspan=2) — with hovertemplate
    fig.add_trace(go.Scatter(
        x=x, y=disps, mode='lines', name='Displacement',
        line=dict(color='green', width=2),
        fill='tozeroy', fillcolor='rgba(0,128,0,0.15)',
        hovertemplate='x=%{x:.2f}m<br>δ=%{y:.3f} mm<extra></extra>',
    ), row=4, col=1)

    # ── P1: SFD global max ──
    abs_shears = [abs(v) for v in shears_data]
    max_v_idx = abs_shears.index(max(abs_shears))
    fig.add_annotation(x=x[max_v_idx], y=shears_data[max_v_idx],
                        text=f'{shears_data[max_v_idx]:.1f}', showarrow=True,
                        arrowhead=2, font=dict(color='red', size=11),
                        ax=15, ay=-28 if shears_data[max_v_idx] >= 0 else 28,
                        row=2, col=1)

    # ── P1: Displacement global max ──
    abs_disps = [abs(d) for d in disps]
    max_d_idx = abs_disps.index(max(abs_disps))
    fig.add_annotation(x=x[max_d_idx], y=disps[max_d_idx],
                        text=f'{abs(disps[max_d_idx]):.3f}', showarrow=True,
                        arrowhead=2, font=dict(color='green', size=11),
                        ax=15, ay=28 if disps[max_d_idx] <= 0 else -28,
                        row=4, col=1)

    # 부호 규약 + Option A 정의
    fig.add_annotation(text='V > 0: ↑ on left face', xref='x3 domain', yref='y3 domain',
                       x=1, y=1, showarrow=False, font=dict(size=9, color='#888'),
                       xanchor='right', yanchor='top', row=2, col=1)
    fig.add_annotation(text='M > 0: sagging  |  Support label = section moment (hover for BL/BR)',
                       xref='x5 domain', yref='y5 domain',
                       x=1, y=1, showarrow=False, font=dict(size=8, color='#888'),
                       xanchor='right', yanchor='top', row=3, col=1)
    fig.add_annotation(text='δ < 0: downward', xref='x7 domain', yref='y7 domain',
                       x=1, y=1, showarrow=False, font=dict(size=9, color='#888'),
                       xanchor='right', yanchor='top', row=4, col=1)

    # ── P1: BMD global max |M| ──
    abs_moments = [abs(m) for m in moments]
    max_m_idx = abs_moments.index(max(abs_moments))
    _max_m_x = x[max_m_idx]
    _max_m_val = moments[max_m_idx]

    # ── P2: Internal-support hogging — minimal label + hover detail ──
    _html_support_locs = set()
    if hasattr(result, 'reactions'):
        reactions = getattr(result, 'reactions', [])
        for idx_r, r in enumerate(reactions):
            if idx_r == 0 or idx_r == len(reactions) - 1:
                continue
            loc = r.get("location", 0.0)
            m_left = r.get("moment_left_kNm", None)
            m_right = r.get("moment_right_kNm", None)
            if m_left is not None and m_right is not None:
                if abs(m_left) > 0.1 or abs(m_right) > 0.1:
                    label = chr(65 + idx_r)
                    closest_idx = min(range(len(x)), key=lambda i: abs(x[i] - loc))
                    m_val = moments[closest_idx]
                    _html_support_locs.add(loc)
                    # Hover-capable invisible marker with full BL/BR detail
                    fig.add_trace(go.Scatter(
                        x=[loc], y=[m_val], mode='markers+text',
                        marker=dict(size=8, color='darkred', symbol='diamond'),
                        text=[label],
                        textposition='bottom center' if m_val < 0 else 'top center',
                        textfont=dict(size=11, color='darkred', family='Arial Black'),
                        hovertemplate=(
                            f'<b>Support {label}</b> @ {loc:.1f}m<br>'
                            f'Section M = {m_left:.2f} kN·m<br>'
                            f'M_left = {m_left:.2f} kN·m<br>'
                            f'M_right = {m_right:.2f} kN·m<br>'
                            '<span style="color:#999;font-size:10px">'
                            'Sign: M&gt;0 sagging. End moments follow element local convention.</span>'
                            '<extra></extra>'
                        ),
                        showlegend=False,
                    ), row=3, col=1)

    # Draw P1 global max (only if not at internal support)
    _near_sup_html = any(abs(_max_m_x - sl) < span_total * 0.02 for sl in _html_support_locs)
    if not _near_sup_html:
        fig.add_annotation(x=_max_m_x, y=_max_m_val,
                            text=f'{_max_m_val:.1f}', showarrow=True,
                            arrowhead=2, font=dict(color='blue', size=11),
                            ax=15, ay=-28 if _max_m_val >= 0 else 28,
                            row=3, col=1)

    # ── P3: Per-span max sagging — hover marker ──
    if hasattr(result, 'num_spans') and result.spans:
        x_start = 0.0
        for sp_i, sp in enumerate(result.spans):
            x_end = x_start + sp
            best_m, best_x = 0.0, x_start + sp / 2
            for i, xi in enumerate(x):
                if x_start + sp * 0.05 < xi < x_end - sp * 0.05:
                    if moments[i] > best_m:
                        best_m = moments[i]
                        best_x = xi
            if best_m > 0.1:
                fig.add_trace(go.Scatter(
                    x=[best_x], y=[best_m], mode='markers',
                    marker=dict(size=7, color='#1565C0', symbol='circle'),
                    hovertemplate=(
                        f'<b>Span {sp_i+1} max sagging</b><br>'
                        f'x = {best_x:.2f} m<br>'
                        f'M = {best_m:.2f} kN·m'
                        '<extra></extra>'
                    ),
                    showlegend=False,
                ), row=3, col=1)
                # P3 visible annotation omitted — value available on hover
            x_start = x_end

    fig.update_yaxes(title_text='V (kN)', row=2, col=1)
    fig.update_yaxes(title_text='M (kN·m)', row=3, col=1)
    fig.update_yaxes(title_text='δ (mm)', row=4, col=1)
    fig.update_xaxes(title_text='Position (m)', row=4, col=1)

    # 지점 위치에 수직 점선 추가
    for s in supports:
        for row in range(2, 5):
            fig.add_vline(x=s["x"], line_dash="dot", line_color="gray",
                          opacity=0.5, row=row, col=1)

    # 평형 검증 패널
    try:
        from core.verification import verify_equilibrium
        eq = verify_equilibrium(result)
        marks = []
        for key in ["sum_vertical", "sum_moment", "shear_jumps"]:
            if key in eq:
                s = eq[key]
                icon = "✓" if s["status"] == "OK" else "✗"
                marks.append(f"{icon} {s['description']} ({s['status']})")
        eq_text = "  |  ".join(marks)
        fig.add_annotation(
            text=eq_text, xref='paper', yref='paper',
            x=0.5, y=-0.02, showarrow=False,
            font=dict(size=10, color='#333', family='monospace'),
            bgcolor='#F8F8F8', bordercolor='#CCC', borderwidth=1,
        )
    except Exception:
        pass

    # 처짐 판정 패널
    try:
        defl_ratio = getattr(result, 'deflection_limit_ratio', 300)
        span_results = getattr(result, 'span_results', [])
        if span_results:
            defl_parts = []
            for sr in span_results:
                sp_len = sr.get("span_length", 0)
                d_max = sr.get("max_displacement_mm", 0)
                d_allow = sr.get("delta_allow_mm", sp_len * 1000 / defl_ratio if defl_ratio > 0 else 0)
                status = sr.get("deflection_status", "OK" if d_max <= d_allow else "NG")
                icon = "✓" if status == "OK" else "✗"
                defl_parts.append(f"{icon} Span{sr.get('span_index',0)+1}: δ={d_max:.2f}mm, L/{defl_ratio}={d_allow:.1f}mm → {status}")
            defl_text = " | ".join(defl_parts)
            fig.add_annotation(
                text=f"Deflection (L/{defl_ratio}): {defl_text}", xref='paper', yref='paper',
                x=0.5, y=-0.04, showarrow=False,
                font=dict(size=9, color='#333', family='monospace'),
                bgcolor='#F0FFF0', bordercolor='#9C9', borderwidth=1,
            )
    except Exception:
        pass

    # 모델 정보 패널
    try:
        e_gpa = round(getattr(result, 'E_MPa', 0) / 1000, 1)
        ix = getattr(result, 'Ix_mm4', 0)
        zx = round(getattr(result, 'Zx_mm3', 0), 1)
        n_elem = getattr(result, 'num_elements_per_span', getattr(result, 'num_elements', 0))
        model_text = f"E={e_gpa}GPa | Ix={ix:.3e}mm⁴ | Zx={zx:.0f}mm³ | elasticBeamColumn | Elem/span={n_elem}"
        fig.add_annotation(
            text=model_text, xref='paper', yref='paper',
            x=0.5, y=-0.06, showarrow=False,
            font=dict(size=8, color='#666', family='monospace'),
        )
    except Exception:
        pass

    fig.update_layout(
        height=1000,
        title_text='Beam Analysis Results',
        title_font_size=16,
        showlegend=False,
        template='plotly_white',
        margin=dict(b=120),
    )

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.html')
        os.close(fd)

    fig.write_html(output_path)
    return output_path


# ============================================================
# 2D Frame Visualization (Plotly)
# ============================================================

def plot_frame_2d_interactive(result, output_path: Optional[str] = None,
                               deformation_scale: float = 50.0) -> str:
    """
    2D 골조 해석 결과를 Plotly 인터랙티브 HTML로 시각화

    Parameters
    ----------
    result : Frame2DResult
        2D 골조 해석 결과 객체
    output_path : str, optional
        출력 HTML 파일 경로
    deformation_scale : float
        변형 확대 배율 (기본: 50)

    Returns
    -------
    str : HTML 파일 경로
    """
    from core.frame_2d import Frame2DResult

    nodes = result.nodes
    elements = result.elements
    nodal_disps = result.nodal_displacements
    element_forces = result.element_forces
    reactions = result.reactions

    # 노드 좌표 딕셔너리 생성 (x_m, y_m 또는 x, y 키 지원)
    node_coords = {}
    for n in nodes:
        x = n.get("x_m", n.get("x", 0))
        y = n.get("y_m", n.get("y", 0))
        node_coords[n["id"]] = (x, y)

    # 변위 딕셔너리 생성 (mm → m 변환하여 좌표에 적용)
    disp_dict = {}
    for nd in nodal_disps:
        nid = nd["node"]
        disp_dict[nid] = {
            "dx": nd["dx_mm"] / 1000 * deformation_scale,  # mm to m, scaled
            "dy": nd["dy_mm"] / 1000 * deformation_scale,
        }

    # 요소력 딕셔너리 (부재별 합산용)
    force_dict = {ef["element"]: ef for ef in element_forces}

    # 기하 정보
    total_height = result.total_height
    total_width = result.total_width
    n_stories = result.num_stories
    n_bays = result.num_bays
    n_cols = n_bays + 1

    # 그리드 기반 메인 연결선 생성 (요소 분할과 무관하게 메인 노드 간 연결)
    # 노드 ID 규칙: 왼쪽 아래부터 오른쪽으로, 층별로 위로
    # 노드 ID = story * n_cols + col + 1 (story=0은 바닥, col=0은 왼쪽)
    main_connections = []  # [(ni, nj, type), ...]

    # 기둥: 각 열에서 수직 연결
    for col in range(n_cols):
        for story in range(n_stories):
            ni = story * n_cols + col + 1
            nj = (story + 1) * n_cols + col + 1
            main_connections.append((ni, nj, "column"))

    # 보: 각 층에서 수평 연결
    for story in range(1, n_stories + 1):
        for bay in range(n_bays):
            ni = story * n_cols + bay + 1
            nj = story * n_cols + bay + 2
            main_connections.append((ni, nj, "beam"))

    # Figure 생성 (단일 플롯)
    fig = go.Figure()

    # 색상 정의
    color_column = '#666666'  # 기둥 (회색)
    color_beam = '#2196F3'    # 보 (파란색)
    color_deformed = '#F44336'  # 변형 (빨간색)
    color_load_floor = '#4CAF50'  # 층 하중 (초록)
    color_load_lateral = '#FF9800'  # 횡하중 (주황)

    # 1) 변형 전 프레임 그리기 (메인 연결선 기반)
    column_legend_shown = False
    beam_legend_shown = False
    for ni, nj, etype in main_connections:
        if ni not in node_coords or nj not in node_coords:
            continue

        x0, y0 = node_coords[ni]
        x1, y1 = node_coords[nj]

        color = color_column if etype == "column" else color_beam
        width = 4 if etype == "column" else 3

        hover_text = (
            f"<b>{etype.capitalize()}</b><br>"
            f"Node {ni} → {nj}<br>"
            f"({x0:.1f}, {y0:.1f}) → ({x1:.1f}, {y1:.1f})"
        )

        # 각 타입별 첫 번째에서만 범례 표시
        if etype == "column":
            show_legend = not column_legend_shown
            if show_legend:
                column_legend_shown = True
        else:
            show_legend = not beam_legend_shown
            if show_legend:
                beam_legend_shown = True

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='text',
            hovertext=hover_text,
            name=etype.capitalize(),
            legendgroup=etype,
            showlegend=show_legend,
        ))

    # 2) 변형 형상 그리기 (메인 연결선 기반)
    deformed_legend_shown = False
    for ni, nj, etype in main_connections:
        if ni not in node_coords or nj not in node_coords:
            continue

        x0, y0 = node_coords[ni]
        x1, y1 = node_coords[nj]

        # 변위 적용
        d0 = disp_dict.get(ni, {"dx": 0, "dy": 0})
        d1 = disp_dict.get(nj, {"dx": 0, "dy": 0})

        x0_def = x0 + d0["dx"]
        y0_def = y0 + d0["dy"]
        x1_def = x1 + d1["dx"]
        y1_def = y1 + d1["dy"]

        # 첫 번째에서만 범례 표시
        show_legend = not deformed_legend_shown
        if show_legend:
            deformed_legend_shown = True

        fig.add_trace(go.Scatter(
            x=[x0_def, x1_def], y=[y0_def, y1_def],
            mode='lines',
            line=dict(color=color_deformed, width=2, dash='dash'),
            hoverinfo='skip',
            legendgroup='deformed',
            showlegend=show_legend,
            name=f'Deformed (×{deformation_scale:.0f})',
        ))

    # 3) 노드 마커 (변형 전)
    node_x = [n.get("x_m", n.get("x", 0)) for n in nodes]
    node_y = [n.get("y_m", n.get("y", 0)) for n in nodes]
    node_hover = []
    for n in nodes:
        nid = n["id"]
        nx = n.get("x_m", n.get("x", 0))
        ny = n.get("y_m", n.get("y", 0))
        nd = next((d for d in nodal_disps if d["node"] == nid), None)
        if nd:
            hover = (
                f"<b>Node {nid}</b><br>"
                f"Position: ({nx:.2f}, {ny:.2f}) m<br>"
                f"dx: {nd['dx_mm']:.3f} mm<br>"
                f"dy: {nd['dy_mm']:.3f} mm<br>"
                f"rz: {nd['rz_rad']:.6f} rad"
            )
        else:
            hover = f"<b>Node {nid}</b><br>({nx:.2f}, {ny:.2f}) m"
        node_hover.append(hover)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=6, color='black'),
        hoverinfo='text',
        hovertext=node_hover,
        name='Nodes',
        showlegend=True,
    ))

    # 4) 지점 심볼
    support_type = getattr(result, 'supports', 'fixed') or 'fixed'
    # 바닥층 노드 (y=0)
    base_nodes = [n for n in nodes if abs(n.get("y_m", n.get("y", 0))) < 0.01]

    for bn in base_nodes:
        bx = bn.get("x_m", bn.get("x", 0))
        by = bn.get("y_m", bn.get("y", 0))
        sz = total_height * 0.03

        # 반력 정보
        rxn = next((r for r in reactions if r["node"] == bn["id"]), None)
        rxn_text = ""
        if rxn:
            rxn_text = (
                f"<br><b>Reactions:</b><br>"
                f"RX: {rxn.get('RX_kN', 0):.1f} kN<br>"
                f"RY: {rxn.get('RY_kN', 0):.1f} kN<br>"
                f"MZ: {rxn.get('MZ_kNm', 0):.1f} kN·m"
            )

        if support_type == "fixed":
            # 고정단: 사각형
            fig.add_trace(go.Scatter(
                x=[bx - sz, bx + sz, bx + sz, bx - sz, bx - sz],
                y=[by, by, by - sz * 0.8, by - sz * 0.8, by],
                mode='lines',
                fill='toself',
                fillcolor='#888888',
                line=dict(color='black', width=2),
                hoverinfo='text',
                hovertext=f'<b>Fixed Support</b><br>Node {bn["id"]}{rxn_text}',
                showlegend=False,
            ))
            # 해칭 표시
            for i in range(4):
                hx = bx - sz + i * sz * 0.5
                fig.add_trace(go.Scatter(
                    x=[hx, hx - sz * 0.3],
                    y=[by - sz * 0.8, by - sz * 1.2],
                    mode='lines',
                    line=dict(color='black', width=1),
                    hoverinfo='skip',
                    showlegend=False,
                ))
        else:
            # 핀: 삼각형
            fig.add_trace(go.Scatter(
                x=[bx, bx - sz * 0.5, bx + sz * 0.5, bx],
                y=[by, by - sz, by - sz, by],
                mode='lines',
                fill='toself',
                fillcolor='white',
                line=dict(color='black', width=2),
                hoverinfo='text',
                hovertext=f'<b>Pinned Support</b><br>Node {bn["id"]}{rxn_text}',
                showlegend=False,
            ))

    # 5) 하중 표시
    loads_info = getattr(result, 'loads_info', []) or []
    arrow_scale = total_height * 0.08

    for ld in loads_info:
        ltype = ld.get("type", "")
        story = ld.get("story", 1)

        if ltype == "floor":
            # 층 등분포 하중 (아래 방향 화살표)
            value = ld.get("value", 0)
            y_level = sum(result.stories[:story])

            # 각 보 위에 화살표 표시
            n_arrows = max(3, int(total_width / 2))
            xs = np.linspace(0, total_width, n_arrows)

            for xp in xs:
                fig.add_annotation(
                    x=xp, y=y_level + arrow_scale * 0.2,
                    ax=xp, ay=y_level + arrow_scale,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                    arrowcolor=color_load_floor,
                )

            # 상단 수평선
            fig.add_trace(go.Scatter(
                x=[0, total_width],
                y=[y_level + arrow_scale, y_level + arrow_scale],
                mode='lines',
                line=dict(color=color_load_floor, width=2),
                hoverinfo='text',
                hovertext=f'Floor Load: {value} kN/m @ Story {story}',
                showlegend=False,
            ))

            # 라벨
            fig.add_annotation(
                x=total_width / 2, y=y_level + arrow_scale * 1.3,
                text=f'{value} kN/m',
                showarrow=False,
                font=dict(size=10, color=color_load_floor, family='Arial Black'),
            )

        elif ltype == "lateral":
            # 횡하중 (수평 방향 화살표)
            fx = ld.get("fx", ld.get("value", 0))
            y_level = sum(result.stories[:story])

            # 왼쪽에서 프레임으로 향하는 화살표
            arrow_len = total_width * 0.15
            fig.add_annotation(
                x=0, y=y_level,
                ax=-arrow_len, ay=y_level,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=2, arrowsize=1.5, arrowwidth=2.5,
                arrowcolor=color_load_lateral,
            )

            # 라벨
            fig.add_annotation(
                x=-arrow_len * 0.5, y=y_level + arrow_scale * 0.3,
                text=f'{fx} kN',
                showarrow=False,
                font=dict(size=11, color=color_load_lateral, family='Arial Black'),
            )

    # 6) 결과 정보 표시
    # 최대값 마커
    max_disp_x_node = result.max_displacement_x_node
    max_disp_y_node = result.max_displacement_y_node

    # 최대 수평변위 노드
    if max_disp_x_node:
        nd = next((n for n in nodes if n["id"] == max_disp_x_node), None)
        if nd:
            nd_x = nd.get("x_m", nd.get("x", 0))
            nd_y = nd.get("y_m", nd.get("y", 0))
            d = disp_dict.get(max_disp_x_node, {"dx": 0, "dy": 0})
            fig.add_trace(go.Scatter(
                x=[nd_x + d["dx"]], y=[nd_y + d["dy"]],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='x'),
                text=[f'Max dx: {result.max_displacement_x:.2f} mm'],
                textposition='top right',
                textfont=dict(size=9, color='red'),
                hoverinfo='skip',
                showlegend=False,
            ))

    # 최대 수직변위 노드
    if max_disp_y_node and max_disp_y_node != max_disp_x_node:
        nd = next((n for n in nodes if n["id"] == max_disp_y_node), None)
        if nd:
            nd_x = nd.get("x_m", nd.get("x", 0))
            nd_y = nd.get("y_m", nd.get("y", 0))
            d = disp_dict.get(max_disp_y_node, {"dx": 0, "dy": 0})
            fig.add_trace(go.Scatter(
                x=[nd_x + d["dx"]], y=[nd_y + d["dy"]],
                mode='markers+text',
                marker=dict(size=12, color='blue', symbol='x'),
                text=[f'Max dy: {result.max_displacement_y:.2f} mm'],
                textposition='bottom right',
                textfont=dict(size=9, color='blue'),
                hoverinfo='skip',
                showlegend=False,
            ))

    # 7) 레이아웃 설정
    # 치수선 (층고, 경간)
    dim_offset = total_width * 0.08

    # 층고 치수 (오른쪽)
    y_start = 0
    for i, sh in enumerate(result.stories):
        y_end = y_start + sh
        mid_y = (y_start + y_end) / 2

        fig.add_trace(go.Scatter(
            x=[total_width + dim_offset, total_width + dim_offset],
            y=[y_start, y_end],
            mode='lines',
            line=dict(color='#888', width=1),
            hoverinfo='skip',
            showlegend=False,
        ))
        fig.add_annotation(
            x=total_width + dim_offset * 1.5, y=mid_y,
            text=f'{sh:.1f}m',
            showarrow=False,
            font=dict(size=9, color='#666'),
        )
        y_start = y_end

    # 경간 치수 (아래)
    x_start = 0
    for i, bw in enumerate(result.bays):
        x_end = x_start + bw
        mid_x = (x_start + x_end) / 2

        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[-dim_offset, -dim_offset],
            mode='lines',
            line=dict(color='#888', width=1),
            hoverinfo='skip',
            showlegend=False,
        ))
        fig.add_annotation(
            x=mid_x, y=-dim_offset * 1.5,
            text=f'{bw:.1f}m',
            showarrow=False,
            font=dict(size=9, color='#666'),
        )
        x_start = x_end

    # 8) 정보 패널 (하단)
    info_text = (
        f"<b>2D Frame Analysis Results</b><br>"
        f"Stories: {result.num_stories} | Bays: {result.num_bays} | "
        f"Nodes: {len(nodes)} | Elements: {result.num_elements}<br>"
        f"Column: {result.column_section} | Beam: {result.beam_section} | "
        f"Material: {result.material_name} (E={result.E_MPa/1000:.0f} GPa)"
    )

    fig.add_annotation(
        text=info_text,
        xref='paper', yref='paper',
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=10, color='#333'),
        align='center',
        bgcolor='#F5F5F5',
        bordercolor='#DDD',
        borderwidth=1,
        borderpad=8,
    )

    # 결과 요약 패널
    result_text = (
        f"<b>Max Results:</b> "
        f"dx={result.max_displacement_x:.2f}mm (N{result.max_displacement_x_node}) | "
        f"dy={result.max_displacement_y:.2f}mm (N{result.max_displacement_y_node}) | "
        f"Drift={result.max_drift:.5f}rad (Story {result.max_drift_story}) | "
        f"M={result.max_moment:.1f}kN·m (E{result.max_moment_element}) | "
        f"N={result.max_axial:.1f}kN (E{result.max_axial_element})"
    )

    fig.add_annotation(
        text=result_text,
        xref='paper', yref='paper',
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=9, color='#333', family='monospace'),
        bgcolor='#E3F2FD',
        bordercolor='#90CAF9',
        borderwidth=1,
        borderpad=6,
    )

    # 레이아웃 마무리
    margin_x = total_width * 0.25
    margin_y = total_height * 0.2

    fig.update_layout(
        title=dict(
            text='2D Frame Analysis',
            font=dict(size=16, color='#333'),
            x=0.5,
        ),
        xaxis=dict(
            range=[-margin_x, total_width + margin_x],
            scaleanchor='y',
            scaleratio=1,
            showgrid=True,
            gridcolor='#EEE',
            zeroline=False,
            title='X (m)',
        ),
        yaxis=dict(
            range=[-margin_y, total_height + margin_y],
            showgrid=True,
            gridcolor='#EEE',
            zeroline=False,
            title='Y (m)',
        ),
        height=700,
        width=900,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#CCC',
            borderwidth=1,
        ),
        template='plotly_white',
        margin=dict(t=100, b=100, l=60, r=60),
        hovermode='closest',
    )

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.html')
        os.close(fd)

    fig.write_html(output_path)
    return output_path


# ============================================================
# 2D Frame Multi-Case Visualization (Tab-based HTML)
# ============================================================

def plot_frame_2d_multi_interactive(
    multi_result,
    equilibrium_checks: Optional[dict] = None,
    output_path: Optional[str] = None,
    deformation_scale: float = 50.0,
) -> str:
    """
    멀티케이스 2D 골조 해석 결과를 탭 기반 인터랙티브 HTML로 시각화

    Parameters
    ----------
    multi_result : Frame2DMultiCaseResult
    equilibrium_checks : dict, optional
        케이스별 평형검증 결과 {case_name: verify_frame_equilibrium() 결과}
    output_path : str, optional
    deformation_scale : float

    Returns
    -------
    str : HTML 파일 경로
    """
    import json

    # 기하 데이터
    nodes = multi_result.nodes
    stories = multi_result.stories
    bays = multi_result.bays
    n_stories = multi_result.num_stories
    n_bays = multi_result.num_bays
    n_cols = n_bays + 1
    total_height = multi_result.total_height
    total_width = multi_result.total_width

    # 메인 연결선 (기둥/보)
    main_connections = []
    for col in range(n_cols):
        for story in range(n_stories):
            ni = story * n_cols + col + 1
            nj = (story + 1) * n_cols + col + 1
            main_connections.append({"ni": ni, "nj": nj, "type": "column"})
    for story in range(1, n_stories + 1):
        for bay in range(n_bays):
            ni = story * n_cols + bay + 1
            nj = story * n_cols + bay + 2
            main_connections.append({"ni": ni, "nj": nj, "type": "beam"})

    # 노드 좌표 dict
    node_coords = {}
    for n in nodes:
        node_coords[n["id"]] = {"x": n.get("x_m", n.get("x", 0)), "y": n.get("y_m", n.get("y", 0))}

    # 지점 노드
    base_nodes = [n["id"] for n in nodes if abs(n.get("y_m", n.get("y", 0))) < 0.01]

    # 케이스 + 조합 이름 목록
    case_names = list(multi_result.load_cases.keys())
    combo_names = list(multi_result.load_combinations.keys()) if multi_result.load_combinations else []
    all_names = case_names + combo_names

    # 케이스별 데이터 직렬화
    def _serialize_case(name):
        cr = multi_result.case_results.get(name) or multi_result.combo_results.get(name)
        if cr is None:
            return None
        return {
            "nodal_displacements": cr.nodal_displacements,
            "reactions": cr.reactions,
            "story_drifts": cr.story_drifts,
            "story_data": cr.story_data,
            "max_displacement_x": cr.max_displacement_x,
            "max_displacement_y": cr.max_displacement_y,
            "max_displacement_x_node": cr.max_displacement_x_node,
            "max_displacement_y_node": cr.max_displacement_y_node,
            "max_drift": cr.max_drift,
            "max_drift_story": cr.max_drift_story,
            "max_moment": cr.max_moment,
            "max_moment_element": cr.max_moment_element,
            "max_axial": cr.max_axial,
            "max_axial_element": cr.max_axial_element,
            "max_shear": cr.max_shear,
            "max_shear_element": cr.max_shear_element,
        }

    case_data = {}
    for name in all_names:
        sd = _serialize_case(name)
        if sd:
            case_data[name] = sd

    # 부재력 데이터
    member_forces_data = {}
    for name in all_names:
        mf = multi_result.member_forces.get(name)
        if mf:
            member_forces_data[name] = mf

    # 부재 정보
    member_info = multi_result.member_info

    # 하중케이스 원본 데이터 (하중 표시용)
    load_cases_raw = multi_result.load_cases

    # 평형검증 데이터
    eq_data = equilibrium_checks or {}

    # 층고 누적 y
    y_levels = [0.0]
    for s in stories:
        y_levels.append(y_levels[-1] + s)

    # JSON 직렬화
    geometry_json = json.dumps({
        "nodes": nodes,
        "node_coords": {str(k): v for k, v in node_coords.items()},
        "main_connections": main_connections,
        "base_nodes": base_nodes,
        "stories": stories,
        "bays": bays,
        "y_levels": y_levels,
        "n_stories": n_stories,
        "n_bays": n_bays,
        "n_cols": n_cols,
        "total_height": total_height,
        "total_width": total_width,
        "supports": multi_result.supports,
        "column_section": multi_result.column_section,
        "beam_section": multi_result.beam_section,
        "material_name": multi_result.material_name,
        "E_MPa": multi_result.E_MPa,
        "num_elements": multi_result.num_elements,
        "num_elements_per_member": multi_result.num_elements_per_member,
        "deformation_scale": deformation_scale,
        "column_A_mm2": getattr(multi_result, 'column_A_mm2', 0),
        "column_I_mm4": getattr(multi_result, 'column_I_mm4', 0),
        "column_h_mm": getattr(multi_result, 'column_h_mm', 0),
        "beam_A_mm2": getattr(multi_result, 'beam_A_mm2', 0),
        "beam_I_mm4": getattr(multi_result, 'beam_I_mm4', 0),
        "beam_h_mm": getattr(multi_result, 'beam_h_mm', 0),
        "fy_MPa": getattr(multi_result, 'fy_MPa', 0),
    })
    case_data_json = json.dumps(case_data)
    member_forces_json = json.dumps(member_forces_data)
    member_info_json = json.dumps(member_info)
    load_cases_json = json.dumps(load_cases_raw)
    load_combos_json = json.dumps(multi_result.load_combinations or {})
    eq_data_json = json.dumps(eq_data)
    all_names_json = json.dumps(all_names)
    case_names_json = json.dumps(case_names)
    combo_names_json = json.dumps(combo_names)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2D Frame Multi-Case Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 16px 24px; }}
.header h1 {{ font-size: 20px; margin-bottom: 4px; }}
.header .meta {{ font-size: 12px; opacity: 0.85; }}
.controls {{ background: white; padding: 12px 24px; border-bottom: 1px solid #ddd; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
.controls label {{ font-weight: 600; font-size: 13px; }}
.controls select {{ padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; min-width: 200px; }}
.tab-bar {{ background: white; padding: 0 24px; border-bottom: 2px solid #e0e0e0; display: flex; gap: 0; }}
.tab-btn {{ padding: 12px 20px; border: none; background: none; cursor: pointer; font-size: 13px; font-weight: 600;
           color: #666; border-bottom: 3px solid transparent; transition: all 0.2s; }}
.tab-btn:hover {{ color: #1a237e; background: #f0f0ff; }}
.tab-btn.active {{ color: #1a237e; border-bottom-color: #1a237e; }}
.tab-content {{ display: none; padding: 16px 24px; }}
.tab-content.active {{ display: block; }}
.chart-container {{ background: white; border-radius: 8px; padding: 12px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
th {{ background: #f0f0f0; font-weight: 600; text-align: center; }}
.status-ok {{ color: #2e7d32; font-weight: bold; }}
.status-fail {{ color: #c62828; font-weight: bold; }}
.drift-green {{ background: #e8f5e9; }}
.drift-yellow {{ background: #fff8e1; }}
.drift-red {{ background: #ffebee; }}
.eq-panel {{ margin-top: 16px; padding: 12px; border-radius: 6px; border: 1px solid #ddd; }}
.eq-ok {{ border-color: #4caf50; background: #e8f5e9; }}
.eq-fail {{ border-color: #f44336; background: #ffebee; }}
.member-select {{ display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
.member-select select {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }}
.export-btn {{ display: inline-block; padding: 10px 20px; margin: 8px 8px 8px 0; background: #1a237e; color: white;
              border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600; }}
.export-btn:hover {{ background: #283593; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.summary-card {{ background: white; border-radius: 6px; padding: 12px; border-left: 4px solid #1a237e; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.summary-card .label {{ font-size: 11px; color: #666; margin-bottom: 4px; }}
.summary-card .value {{ font-size: 18px; font-weight: 700; color: #1a237e; }}
.summary-card .detail {{ font-size: 10px; color: #999; margin-top: 2px; }}
.label-toggle {{ display: flex; gap: 16px; margin-bottom: 8px; align-items: center; flex-wrap: wrap; padding: 8px 0; }}
.label-toggle label {{ font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px; user-select: none; }}
.label-toggle input[type="checkbox"] {{ cursor: pointer; accent-color: #1a237e; }}
.label-toggle .hint {{ font-size: 11px; color: #999; margin-left: 8px; font-style: italic; }}
.global-controls {{ display: flex; gap: 8px; margin-left: auto; align-items: center; }}
.global-btn {{ padding: 6px 14px; border: 1px solid #1a237e; background: white; color: #1a237e; border-radius: 4px;
              cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s; }}
.global-btn:hover {{ background: #e8eaf6; }}
.global-btn.active {{ background: #1a237e; color: white; }}
.drift-controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
.drift-controls label {{ font-size: 12px; font-weight: 600; }}
.drift-controls select, .drift-controls input {{ padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }}
.drift-controls input[type="number"] {{ width: 70px; }}
.drift-judgment {{ padding: 8px 12px; border-radius: 4px; font-size: 13px; font-weight: 600; margin-bottom: 12px; }}
.drift-judgment.ok {{ background: #e8f5e9; border: 1px solid #4caf50; color: #2e7d32; }}
.drift-judgment.ng {{ background: #ffebee; border: 1px solid #f44336; color: #c62828; }}
.shear-toggle {{ display: flex; gap: 12px; align-items: center; margin-bottom: 8px; }}
.shear-toggle label {{ font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px; }}
.shear-detail {{ margin: 8px 0; padding: 10px 12px; background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px; font-size: 12px; }}
.shear-detail h4 {{ margin: 0 0 6px; font-size: 13px; color: #333; }}
.shear-detail table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.shear-detail th {{ text-align: left; padding: 3px 8px; border-bottom: 2px solid #ccc; color: #555; font-weight: 600; }}
.shear-detail td {{ padding: 3px 8px; border-bottom: 1px solid #eee; }}
.shear-detail .sum-row {{ font-weight: 700; border-top: 2px solid #555; }}
.shear-detail .note {{ font-size: 11px; color: #666; margin-top: 6px; font-style: italic; }}
.design-review {{ font-size: 12px; }}
.design-review h4 {{ margin: 0 0 8px; font-size: 13px; color: #37474f; }}
.review-item {{ display: flex; align-items: center; gap: 8px; padding: 4px 0; border-bottom: 1px solid #f0f0f0; }}
.review-label {{ flex: 1; color: #555; }}
.review-status {{ font-weight: 600; padding: 2px 8px; border-radius: 3px; font-size: 11px; white-space: nowrap; }}
.review-ok {{ background: #e8f5e9; color: #2e7d32; }}
.review-note {{ background: #fff8e1; color: #f57f17; }}
.review-over {{ background: #ffebee; color: #c62828; }}
.shear-warn-badge {{ background: #fff3e0; color: #e65100; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; cursor: help; border: 1px solid #ffcc80; }}
.member-summary {{ margin-top: 16px; }}
.member-summary h4 {{ margin-bottom: 8px; font-size: 13px; }}
.sign-note {{ font-size: 10px; color: #888; margin-top: 4px; font-style: italic; }}
.envelope-controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
.envelope-controls select {{ padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }}
.img-export-btn {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #455a64; color: white;
                   border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
.img-export-btn:hover {{ background: #546e7a; }}
.report-btn {{ display: inline-block; padding: 10px 20px; margin: 8px 0; background: #37474f; color: white;
               border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600; }}
.report-btn:hover {{ background: #455a64; }}
.model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
.model-section {{ background: white; border-radius: 6px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.model-section h4 {{ font-size: 13px; color: #1a237e; margin-bottom: 8px; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }}
.model-table {{ width: 100%; font-size: 12px; }}
.model-table td:first-child {{ font-weight: 600; color: #555; width: 40%; white-space: nowrap; }}
.model-table td:last-child {{ color: #222; }}
.model-table td {{ padding: 4px 8px; border: none; border-bottom: 1px solid #f0f0f0; text-align: left; }}
.supported {{ color: #2e7d32; font-weight: 600; }}
.not-supported {{ color: #c62828; font-weight: 600; }}
.load-summary {{ margin-top: 12px; }}
.load-summary h4 {{ font-size: 13px; margin-bottom: 8px; }}
@page {{ size: A4 landscape; margin: 10mm; }}
@media print {{
  *, *::before, *::after {{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
  body {{ background: white !important; margin: 0; padding: 0; font-size: 11px; }}
  .controls, .tab-bar, .label-toggle, .drift-controls, .shear-toggle, .member-select,
  .global-controls, .envelope-controls, .export-btn, .img-export-btn, .report-btn,
  button, select, input {{ display: none !important; }}
  .header {{ background: white !important; color: black !important; border-bottom: 2px solid black; }}

  /* Show all tab contents for print */
  .tab-content {{ display: block !important; }}
  .tab-content.print-hide {{ display: none !important; }}
  #tab-export {{ display: none !important; }}

  /* Each tab section starts on a new page (except first) */
  #tab-memberforces, #tab-reactions, #tab-story, #tab-envelope, #tab-model {{
    page-break-before: always;
  }}
  /* Story shear chart + drift table on a separate page from story disp chart */
  .story-shear-section {{
    page-break-before: always;
  }}

  .summary-cards {{ page-break-inside: avoid; break-inside: avoid; margin-bottom: 8px; }}

  /* Chart containers: never split across pages, width follows page */
  .chart-container {{
    box-shadow: none !important;
    border: 1px solid #ccc;
    page-break-inside: avoid;
    break-inside: avoid;
    width: 100%;
    overflow: hidden;
  }}

  /* Tables */
  table {{ page-break-inside: avoid; break-inside: avoid; font-size: 10px; }}
  .member-summary {{ page-break-inside: avoid; break-inside: avoid; }}

  /* Global diagram container: hide if not active */
  #globalDiagramContainer[style*="display: none"] {{ display: none !important; }}
  #globalDiagramContainer[style*="display:none"] {{ display: none !important; }}

  .sign-note {{ font-size: 9px; }}
  .drift-judgment {{ display: block !important; }}

  /* Section titles for print clarity */
  .tab-content::before {{
    display: block;
    font-size: 16px;
    font-weight: 700;
    margin: 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #999;
  }}
  #tab-deformation::before {{ content: ""; }}
  #tab-memberforces::before {{ content: "Member Forces"; }}
  #tab-reactions::before {{ content: "Reactions"; }}
  #tab-story::before {{ content: "Story Response"; }}
  #tab-envelope::before {{ content: "Envelope"; }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>2D Frame Multi-Case Analysis Results</h1>
  <div class="meta">
    {n_stories} Stories / {n_bays} Bays &nbsp;|&nbsp;
    Column: {multi_result.column_section} &nbsp;|&nbsp; Beam: {multi_result.beam_section} &nbsp;|&nbsp;
    Material: {multi_result.material_name} (E={multi_result.E_MPa/1000:.0f} GPa) &nbsp;|&nbsp;
    Elements: {multi_result.num_elements} ({multi_result.num_elements_per_member}/member)
  </div>
</div>

<div class="controls">
  <label>Load Case / Combination:</label>
  <select id="caseSelect" onchange="onCaseChange()"></select>
</div>

<div class="tab-bar">
  <button class="tab-btn active" data-tab="deformation" onclick="showTab('deformation')">Deformation</button>
  <button class="tab-btn" data-tab="memberforces" onclick="showTab('memberforces')">Member Forces</button>
  <button class="tab-btn" data-tab="reactions" onclick="showTab('reactions')">Reactions</button>
  <button class="tab-btn" data-tab="story" onclick="showTab('story')">Story</button>
  <button class="tab-btn" data-tab="envelope" onclick="showTab('envelope')">Envelope</button>
  <button class="tab-btn" data-tab="model" onclick="showTab('model')">Model</button>
  <button class="tab-btn" data-tab="export" onclick="showTab('export')">Export</button>
</div>

<div id="tab-deformation" class="tab-content active">
  <div id="summary-cards" class="summary-cards"></div>
  <div class="label-toggle">
    <label><input type="checkbox" id="chkNodeIds" onchange="toggleLabels()"> Node IDs</label>
    <label><input type="checkbox" id="chkMemberIds" onchange="toggleLabels()"> Member IDs (ni-nj)</label>
    <label><input type="checkbox" id="chkMemberNames" onchange="toggleLabels()"> Member Names</label>
    <label><input type="checkbox" id="chkShowLoads" onchange="toggleLabels()"> Show Loads</label>
    <span class="hint">Click a member to view its forces</span>
  </div>
  <div class="chart-container"><div id="deformPlot" style="width:100%;height:600px;"></div></div>
  <div id="loadSummary" class="load-summary"></div>
</div>

<div id="tab-memberforces" class="tab-content">
  <div class="member-select">
    <label>Type:</label>
    <select id="mfTypeSelect" onchange="onMemberTypeChange()">
      <option value="column">Column</option>
      <option value="beam">Beam</option>
    </select>
    <label>Story:</label>
    <select id="mfStoryFilter" onchange="populateMemberList()">
      <option value="all">All</option>
    </select>
    <label>Line:</label>
    <select id="mfLineFilter" onchange="populateMemberList()">
      <option value="all">All</option>
    </select>
    <label>Member:</label>
    <select id="mfMemberSelect" onchange="onMemberSelect()"></select>
    <div class="global-controls">
      <label style="font-size:12px;font-weight:600;">Global:</label>
      <button class="global-btn" id="globalMBtn" onclick="toggleGlobalDiagram('M')">M Diagram</button>
      <button class="global-btn" id="globalVBtn" onclick="toggleGlobalDiagram('V')">V Diagram</button>
      <button class="global-btn" id="globalNBtn" onclick="toggleGlobalDiagram('N')">N Diagram</button>
    </div>
  </div>
  <div class="chart-container"><div id="memberForcePlot" style="width:100%;height:700px;"></div></div>
  <div id="memberSummary" class="member-summary"></div>
  <div class="chart-container" id="globalDiagramContainer" style="display:none;">
    <div id="globalDiagramPlot" style="width:100%;height:600px;"></div>
  </div>
</div>

<div id="tab-reactions" class="tab-content">
  <div id="reactionTable"></div>
  <div id="equilibriumPanel"></div>
</div>

<div id="tab-story" class="tab-content">
  <div class="drift-controls">
    <label>Drift Limit:</label>
    <select id="driftLimitSelect" onchange="onDriftLimitChange()">
      <option value="200">1/200</option>
      <option value="300">1/300</option>
      <option value="400" selected>1/400</option>
      <option value="custom">Custom</option>
    </select>
    <input type="number" id="driftLimitCustom" min="50" max="2000" value="400" style="display:none;" onchange="onDriftLimitChange()">
  </div>
  <div id="driftJudgment" class="drift-judgment ok"></div>
  <div class="chart-container"><div id="storyDispPlot" style="width:100%;height:400px;"></div></div>
  <div class="shear-toggle">
    <label style="font-weight:600;">Shear Method:</label>
    <select id="shearMethodSelect" onchange="renderStory(caseData[currentCase])">
      <option value="reaction" selected>Reaction-based (ΣRX)</option>
      <option value="element_signed">Element-based (signed)</option>
      <option value="element_abs">Element-based (|V|)</option>
    </select>
    <label style="margin-left:12px;"><input type="checkbox" id="chkAbsShear" onchange="renderStory(caseData[currentCase])"> Use |V| (absolute shear)</label>
  </div>
  <div id="shearNote" style="display:none;color:#666;font-size:12px;padding:4px 8px;background:#FFF8E1;border-radius:4px;margin:4px 0;"></div>
  <div class="chart-container story-shear-section"><div id="storyShearPlot" style="width:100%;height:350px;"></div></div>
  <div id="storyShearDetail" class="shear-detail" style="display:none;"></div>
  <h3 style="margin:16px 0 8px;">Story Drift Table</h3>
  <div id="driftTable"></div>
  <div id="designReview" class="design-review" style="margin-top:16px;"></div>
</div>

<div id="tab-envelope" class="tab-content">
  <div class="envelope-controls">
    <label style="font-weight:600;">Category:</label>
    <select id="envCategory" onchange="renderEnvelope()">
      <option value="displacement">Node Displacement</option>
      <option value="drift">Story Drift</option>
      <option value="memberforces">Member Forces</option>
      <option value="reactions">Reactions</option>
    </select>
    <label style="font-weight:600;margin-left:8px;">Type:</label>
    <select id="envMemberType" onchange="renderEnvelope()" style="display:none;">
      <option value="all">All</option>
      <option value="column">Column</option>
      <option value="beam">Beam</option>
    </select>
    <label style="font-weight:600;margin-left:8px;">Sort:</label>
    <select id="envSort" onchange="renderEnvelope()">
      <option value="default">Default</option>
      <option value="Mmax_desc">|M|max desc</option>
      <option value="Mpos_desc">M+ max desc</option>
      <option value="Mneg_desc">M- min desc</option>
      <option value="Vmax_desc">|V|max desc</option>
      <option value="Nmax_desc">|N|max desc</option>
    </select>
  </div>
  <div id="envelopeContent"></div>
</div>

<div id="tab-model" class="tab-content">
  <h3 style="margin-bottom:16px;">Model &amp; Assumptions</h3>
  <div class="model-grid">
    <div class="model-section">
      <h4>Units</h4>
      <table class="model-table">
        <tr><td>Force</td><td>kN</td></tr>
        <tr><td>Length</td><td>m (geometry), mm (section, displacement)</td></tr>
        <tr><td>Stress</td><td>MPa (N/mm&sup2;)</td></tr>
        <tr><td>Moment</td><td>kN&middot;m</td></tr>
        <tr><td>Rotation</td><td>rad</td></tr>
      </table>
    </div>
    <div class="model-section">
      <h4>Material</h4>
      <table class="model-table" id="modelMaterialTable"></table>
    </div>
    <div class="model-section">
      <h4>Column Section</h4>
      <table class="model-table" id="modelColTable"></table>
    </div>
    <div class="model-section">
      <h4>Beam Section</h4>
      <table class="model-table" id="modelBeamTable"></table>
    </div>
    <div class="model-section">
      <h4>Geometry</h4>
      <table class="model-table" id="modelGeomTable"></table>
    </div>
    <div class="model-section">
      <h4>Analysis</h4>
      <table class="model-table">
        <tr><td>Type</td><td>Linear Static</td></tr>
        <tr><td>Dimension</td><td>2D Frame (3 DOF/node: UX, UY, RZ)</td></tr>
        <tr><td>Element</td><td>Euler&ndash;Bernoulli Beam (elasticBeamColumn)</td></tr>
        <tr><td>Sub-elements/member</td><td id="modelSubElem"></td></tr>
        <tr><td>Solver</td><td>OpenSeesPy</td></tr>
      </table>
    </div>
    <div class="model-section">
      <h4>Sign Convention</h4>
      <table class="model-table">
        <tr><td>Axial (N)</td><td>+ Tension, &minus; Compression</td></tr>
        <tr><td>Shear (V)</td><td>+ acts upward on left face</td></tr>
        <tr><td>Moment (M)</td><td>+ Sagging (tension at bottom)</td></tr>
        <tr><td>Displacement</td><td>+ rightward (dx), + upward (dy)</td></tr>
      </table>
    </div>
    <div class="model-section">
      <h4>Load Cases</h4>
      <table class="model-table" id="modelLoadCases"></table>
    </div>
    <div class="model-section" id="modelCombosSection" style="display:none;">
      <h4>Load Combinations</h4>
      <table class="model-table" id="modelCombos"></table>
    </div>
    <div class="model-section">
      <h4>Capabilities &amp; Limitations</h4>
      <table class="model-table">
        <tr><td>Floor distributed load</td><td class="supported">Supported</td></tr>
        <tr><td>Lateral point load</td><td class="supported">Supported</td></tr>
        <tr><td>Linear superposition (combos)</td><td class="supported">Supported</td></tr>
        <tr><td>Multi-story / multi-bay</td><td class="supported">Supported (up to 10&times;5)</td></tr>
        <tr><td>Fixed / Pinned supports</td><td class="supported">Supported</td></tr>
        <tr><td>End release (hinge)</td><td class="not-supported">Not supported</td></tr>
        <tr><td>Rigid offset</td><td class="not-supported">Not supported</td></tr>
        <tr><td>Shear deformation (Timoshenko)</td><td class="not-supported">Not supported</td></tr>
        <tr><td>P-Delta (geometric nonlinearity)</td><td class="not-supported">Not supported</td></tr>
        <tr><td>Self-weight auto-load</td><td class="not-supported">Not supported</td></tr>
      </table>
    </div>
  </div>
</div>

<div id="tab-export" class="tab-content">
  <h3>CSV Export</h3>
  <p style="margin:8px 0;font-size:13px;color:#666;">Download analysis results for the currently selected load case/combination.</p>
  <button class="export-btn" onclick="exportCSV('nodes')">Nodes CSV</button>
  <button class="export-btn" onclick="exportCSV('reactions')">Reactions CSV</button>
  <button class="export-btn" onclick="exportCSV('members')">Member Forces CSV</button>
  <button class="export-btn" onclick="exportCSV('story')">Story Data CSV</button>
  <button class="export-btn" onclick="exportCSV('extrema')">Member Extrema CSV</button>
  <hr style="margin:16px 0;">
  <h3>Image Export (PNG)</h3>
  <p style="margin:8px 0;font-size:13px;color:#666;">Save current plots as PNG images.</p>
  <button class="img-export-btn" onclick="exportPNG('deformPlot','deformation')">Deformation PNG</button>
  <button class="img-export-btn" onclick="exportPNG('memberForcePlot','member_forces')">Member Forces PNG</button>
  <button class="img-export-btn" onclick="exportPNG('globalDiagramPlot','global_diagram')">Global Diagram PNG</button>
  <button class="img-export-btn" onclick="exportPNG('storyDispPlot','story_disp')">Story Disp. PNG</button>
  <button class="img-export-btn" onclick="exportPNG('storyShearPlot','story_shear')">Story Shear PNG</button>
  <hr style="margin:16px 0;">
  <h3>Report</h3>
  <button class="report-btn" onclick="printReport()">Print / Save as PDF</button>
</div>

<script>
// ============================================================
// Embedded Data
// ============================================================
const geometry = {geometry_json};
const caseData = {case_data_json};
const memberForcesData = {member_forces_json};
const memberInfo = {member_info_json};
const loadCasesRaw = {load_cases_json};
const eqData = {eq_data_json};
const allNames = {all_names_json};
const caseNames = {case_names_json};
const comboNames = {combo_names_json};
const loadCombinations = {load_combos_json};

let currentCase = allNames[0] || '';
let currentTab = 'deformation';
let selectedMemberId = null;
let markedSPos = null;
let globalDiagramType = null;
let deformClickAttached = false;
let driftLimitDenom = 400;

// Build member lookup by ni-nj
const memberLookup = {{}};
memberInfo.forEach(m => {{ memberLookup[m.ni + '-' + m.nj] = m; }});

function getMemberName(m) {{
  const loc = m.location;
  if (m.type === 'column') {{
    return 'C' + (loc.col + 1) + '-' + loc.story + 'F';
  }} else {{
    return 'B' + (loc.bay + 1) + '-' + loc.story + 'F';
  }}
}}

// ============================================================
// Init
// ============================================================
function init() {{
  const sel = document.getElementById('caseSelect');
  if (caseNames.length > 0) {{
    const optg1 = document.createElement('optgroup');
    optg1.label = 'Load Cases';
    caseNames.forEach(n => {{
      const o = document.createElement('option');
      o.value = n; o.textContent = n;
      optg1.appendChild(o);
    }});
    sel.appendChild(optg1);
  }}
  if (comboNames.length > 0) {{
    const optg2 = document.createElement('optgroup');
    optg2.label = 'Combinations';
    comboNames.forEach(n => {{
      const o = document.createElement('option');
      o.value = n; o.textContent = n;
      optg2.appendChild(o);
    }});
    sel.appendChild(optg2);
  }}
  sel.value = currentCase;
  _populateFilterOptions();
  renderModelTab();
  renderAll();
}}

function renderModelTab() {{
  const g = geometry;
  const fmt = (v, d) => typeof v === 'number' ? v.toLocaleString(undefined, {{maximumFractionDigits: d || 0}}) : '-';
  const fmtE = v => typeof v === 'number' && v > 0 ? v.toExponential(3) : '-';

  // Material
  document.getElementById('modelMaterialTable').innerHTML =
    `<tr><td>Name</td><td>${{g.material_name}}</td></tr>` +
    `<tr><td>E (MPa)</td><td>${{fmt(g.E_MPa, 0)}}</td></tr>` +
    `<tr><td>fy (MPa)</td><td>${{fmt(g.fy_MPa, 0)}}</td></tr>`;

  // Column section
  document.getElementById('modelColTable').innerHTML =
    `<tr><td>Name</td><td>${{g.column_section}}</td></tr>` +
    `<tr><td>A (mm&sup2;)</td><td>${{fmt(g.column_A_mm2, 1)}}</td></tr>` +
    `<tr><td>I (mm&sup4;)</td><td>${{fmtE(g.column_I_mm4)}}</td></tr>` +
    `<tr><td>h (mm)</td><td>${{fmt(g.column_h_mm, 0)}}</td></tr>`;

  // Beam section
  document.getElementById('modelBeamTable').innerHTML =
    `<tr><td>Name</td><td>${{g.beam_section}}</td></tr>` +
    `<tr><td>A (mm&sup2;)</td><td>${{fmt(g.beam_A_mm2, 1)}}</td></tr>` +
    `<tr><td>I (mm&sup4;)</td><td>${{fmtE(g.beam_I_mm4)}}</td></tr>` +
    `<tr><td>h (mm)</td><td>${{fmt(g.beam_h_mm, 0)}}</td></tr>`;

  // Geometry
  document.getElementById('modelGeomTable').innerHTML =
    `<tr><td>Configuration</td><td>${{g.n_stories}} story &times; ${{g.n_bays}} bay</td></tr>` +
    `<tr><td>Story Heights (m)</td><td>${{g.stories.map(s => s.toFixed(1)).join(', ')}}</td></tr>` +
    `<tr><td>Bay Widths (m)</td><td>${{g.bays.map(b => b.toFixed(1)).join(', ')}}</td></tr>` +
    `<tr><td>Total Height (m)</td><td>${{g.total_height.toFixed(1)}}</td></tr>` +
    `<tr><td>Total Width (m)</td><td>${{g.total_width.toFixed(1)}}</td></tr>` +
    `<tr><td>Supports</td><td>${{g.supports}}</td></tr>`;

  // Sub-elements
  document.getElementById('modelSubElem').textContent = g.num_elements_per_member;

  // Load Cases
  let lcHtml = '<tr><th style="text-align:left;">Case</th><th style="text-align:left;">Loads</th></tr>';
  Object.entries(loadCasesRaw).forEach(([cn, loads]) => {{
    const desc = loads.map(ld => {{
      if (ld.type === 'floor') return `Floor ${{ld.story}}F: ${{ld.value}} kN/m`;
      if (ld.type === 'lateral') return `Lateral ${{ld.story}}F: Fx=${{ld.fx}} kN`;
      if (ld.type === 'point') return `Point N${{ld.node}}: Fy=${{ld.fy||0}} kN`;
      return JSON.stringify(ld);
    }}).join('<br>');
    lcHtml += `<tr><td>${{cn}}</td><td style="font-size:11px;">${{desc}}</td></tr>`;
  }});
  document.getElementById('modelLoadCases').innerHTML = lcHtml;

  // Load Combinations (formula + source)
  if (comboNames.length > 0) {{
    document.getElementById('modelCombosSection').style.display = '';
    let comboHtml = '<tr><th style="text-align:left;">Combination</th><th style="text-align:left;">Formula</th><th style="text-align:left;">Source</th></tr>';
    Object.entries(loadCombinations).forEach(([cn, factors]) => {{
      const fEntries = Object.entries(factors).filter(([k]) => k !== '_source');
      const fStr = fEntries.map(([k,v]) => `${{v}}\u00d7${{k}}`).join(' + ');
      const source = factors._source || 'User-defined';
      comboHtml += `<tr><td>${{cn}}</td><td style="font-size:11px;">${{fStr}}</td><td style="font-size:11px;color:#666;">${{source}}</td></tr>`;
    }});
    document.getElementById('modelCombos').innerHTML = comboHtml;
  }}
}}

function onCaseChange() {{
  currentCase = document.getElementById('caseSelect').value;
  renderAll();
}}

function showTab(tab) {{
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => {{
    b.classList.toggle('active', b.getAttribute('data-tab') === tab);
  }});
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  renderAll();
}}

function renderAll() {{
  const cd = caseData[currentCase];
  if (!cd) return;
  if (currentTab === 'deformation') {{ renderDeformation(cd); renderSummaryCards(cd); }}
  if (currentTab === 'memberforces') {{ populateMemberList(); if (globalDiagramType) renderGlobalDiagram(); }}
  if (currentTab === 'reactions') {{ renderReactions(cd); renderEquilibrium(); }}
  if (currentTab === 'story') {{ renderStory(cd); }}
  if (currentTab === 'envelope') {{ renderEnvelope(); }}
}}

// ============================================================
// Summary Cards
// ============================================================
function renderSummaryCards(cd) {{
  const div = document.getElementById('summary-cards');
  const driftStr = cd.max_drift > 0 ? '1/' + Math.round(1/cd.max_drift) : '-';
  div.innerHTML = `
    <div class="summary-card"><div class="label">Max Horizontal Disp.</div><div class="value">${{cd.max_displacement_x.toFixed(2)}} mm</div><div class="detail">Node ${{cd.max_displacement_x_node}}</div></div>
    <div class="summary-card"><div class="label">Max Vertical Disp.</div><div class="value">${{cd.max_displacement_y.toFixed(2)}} mm</div><div class="detail">Node ${{cd.max_displacement_y_node}}</div></div>
    <div class="summary-card"><div class="label">Max Drift Ratio</div><div class="value">${{driftStr}}</div><div class="detail">Story ${{cd.max_drift_story}} (${{cd.max_drift.toFixed(6)}} rad)</div></div>
    <div class="summary-card"><div class="label">Max Moment</div><div class="value">${{cd.max_moment.toFixed(1)}} kN&middot;m</div><div class="detail">Element ${{cd.max_moment_element}}</div></div>
    <div class="summary-card"><div class="label">Max Axial</div><div class="value">${{cd.max_axial.toFixed(1)}} kN</div><div class="detail">Element ${{cd.max_axial_element}}</div></div>
    <div class="summary-card"><div class="label">Max Shear</div><div class="value">${{cd.max_shear.toFixed(1)}} kN</div><div class="detail">Element ${{cd.max_shear_element}}</div></div>
  `;
}}

// ============================================================
// Deformation Tab
// ============================================================
function renderDeformation(cd) {{
  const traces = [];
  const nc = geometry.node_coords;
  const scale = geometry.deformation_scale;

  // Displacement map
  const dispMap = {{}};
  cd.nodal_displacements.forEach(d => {{
    dispMap[d.node] = {{ dx: d.dx_mm / 1000 * scale, dy: d.dy_mm / 1000 * scale }};
  }});

  // Undeformed frame (clickable, with customdata for member identification)
  let colLegend = false, beamLegend = false;
  geometry.main_connections.forEach(c => {{
    const a = nc[c.ni], b = nc[c.nj];
    if (!a || !b) return;
    const isCol = c.type === 'column';
    const show = isCol ? !colLegend : !beamLegend;
    if (isCol) colLegend = true; else beamLegend = true;

    const mInfo = memberLookup[c.ni + '-' + c.nj];
    const mId = mInfo ? mInfo.id : null;
    const mName = mInfo ? getMemberName(mInfo) : (c.type + ' ' + c.ni + '-' + c.nj);
    const isSelected = mId !== null && mId === selectedMemberId;

    traces.push({{
      x: [a.x, b.x], y: [a.y, b.y], mode: 'lines',
      line: {{ color: isSelected ? '#FF6F00' : (isCol ? '#666' : '#2196F3'),
              width: isSelected ? 7 : (isCol ? 4 : 3) }},
      name: isCol ? 'Column' : 'Beam', legendgroup: c.type,
      showlegend: show, hoverinfo: 'text',
      hovertext: [`${{mName}} (N${{c.ni}}-N${{c.nj}})`, `${{mName}} (N${{c.ni}}-N${{c.nj}})`],
      customdata: [[mId, c.ni, c.nj, c.type, mName], [mId, c.ni, c.nj, c.type, mName]],
    }});
  }});

  // Deformed frame
  let defLegend = false;
  geometry.main_connections.forEach(c => {{
    const a = nc[c.ni], b = nc[c.nj];
    if (!a || !b) return;
    const da = dispMap[c.ni] || {{dx:0,dy:0}};
    const db = dispMap[c.nj] || {{dx:0,dy:0}};
    const show = !defLegend;
    defLegend = true;
    traces.push({{
      x: [a.x + da.dx, b.x + db.dx], y: [a.y + da.dy, b.y + db.dy],
      mode: 'lines', line: {{ color: '#F44336', width: 2, dash: 'dash' }},
      name: `Deformed (x${{scale}})`, legendgroup: 'deformed',
      showlegend: show, hoverinfo: 'skip',
    }});
  }});

  // Nodes
  const nodeX = [], nodeY = [], nodeHover = [];
  geometry.nodes.forEach(n => {{
    const x = n.x_m !== undefined ? n.x_m : n.x;
    const y = n.y_m !== undefined ? n.y_m : n.y;
    nodeX.push(x); nodeY.push(y);
    const d = cd.nodal_displacements.find(dd => dd.node === n.id);
    nodeHover.push(d
      ? `<b>Node ${{n.id}}</b><br>dx: ${{d.dx_mm.toFixed(3)}} mm<br>dy: ${{d.dy_mm.toFixed(3)}} mm<br>rz: ${{d.rz_rad.toFixed(6)}} rad`
      : `Node ${{n.id}}`);
  }});
  traces.push({{
    x: nodeX, y: nodeY, mode: 'markers',
    marker: {{ size: 5, color: 'black' }},
    hoverinfo: 'text', hovertext: nodeHover,
    name: 'Nodes', showlegend: true,
  }});

  // Support symbols
  geometry.base_nodes.forEach(bn => {{
    const c = nc[bn];
    if (!c) return;
    const sz = geometry.total_height * 0.03;
    if (geometry.supports === 'fixed') {{
      traces.push({{
        x: [c.x-sz, c.x+sz, c.x+sz, c.x-sz, c.x-sz],
        y: [c.y, c.y, c.y-sz*0.8, c.y-sz*0.8, c.y],
        mode: 'lines', fill: 'toself', fillcolor: '#888',
        line: {{ color: 'black', width: 2 }}, showlegend: false,
        hoverinfo: 'text', hovertext: `Fixed Support N${{bn}}`,
      }});
    }} else {{
      traces.push({{
        x: [c.x, c.x-sz*0.5, c.x+sz*0.5, c.x],
        y: [c.y, c.y-sz, c.y-sz, c.y],
        mode: 'lines', fill: 'toself', fillcolor: 'white',
        line: {{ color: 'black', width: 2 }}, showlegend: false,
        hoverinfo: 'text', hovertext: `Pin Support N${{bn}}`,
      }});
    }}
  }});

  // Load arrows (toggled by checkbox)
  const arrowAnnotations = [];
  const tw = geometry.total_width;
  const th = geometry.total_height;
  const aScale = th * 0.08;
  const showLoads = document.getElementById('chkShowLoads').checked;

  if (showLoads) {{
    // Gather effective loads: for combos, factor each component
    let effectiveLoads = [];
    const comboFactors = loadCombinations[currentCase];
    if (comboFactors) {{
      Object.entries(comboFactors).forEach(([cn, factor]) => {{
        const cLoads = loadCasesRaw[cn] || [];
        cLoads.forEach(ld => {{
          const scaled = Object.assign({{}}, ld);
          if (scaled.value !== undefined) scaled.value = scaled.value * factor;
          if (scaled.fx !== undefined) scaled.fx = scaled.fx * factor;
          if (scaled.fy !== undefined) scaled.fy = scaled.fy * factor;
          scaled._source = `${{factor}}\u00d7${{cn}}`;
          effectiveLoads.push(scaled);
        }});
      }});
    }} else {{
      effectiveLoads = (loadCasesRaw[currentCase] || []).map(ld => ({{ ...ld, _source: currentCase }}));
    }}

    effectiveLoads.forEach(ld => {{
      const ltype = ld.type || '';
      const story = ld.story || 1;
      const yLev = geometry.y_levels[story] || 0;

      if (ltype === 'floor') {{
        const w = ld.value || 0;
        const nArr = Math.max(3, Math.floor(tw / 2));
        for (let i = 0; i <= nArr; i++) {{
          const xp = tw * i / nArr;
          arrowAnnotations.push({{
            x: xp, y: yLev + aScale * 0.2,
            ax: xp, ay: yLev + aScale,
            xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
            showarrow: true, arrowhead: 2, arrowsize: 1.2, arrowwidth: 1.5,
            arrowcolor: '#4CAF50',
          }});
        }}
        traces.push({{
          x: [0, tw], y: [yLev + aScale, yLev + aScale],
          mode: 'lines', line: {{ color: '#4CAF50', width: 2 }},
          showlegend: false, hoverinfo: 'text',
          hovertext: `Floor Load: ${{w.toFixed(1)}} kN/m @ Story ${{story}} (${{ld._source}})`,
        }});
        arrowAnnotations.push({{
          x: tw/2, y: yLev + aScale*1.3, text: `${{w.toFixed(1)}} kN/m`,
          showarrow: false, font: {{ size: 10, color: '#4CAF50', family: 'Arial Black' }},
        }});
      }} else if (ltype === 'lateral') {{
        const fx = ld.fx !== undefined ? ld.fx : (ld.value || 0);
        const aLen = tw * 0.15;
        arrowAnnotations.push({{
          x: 0, y: yLev, ax: -aLen, ay: yLev,
          xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
          showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 2.5,
          arrowcolor: '#FF9800',
        }});
        arrowAnnotations.push({{
          x: -aLen*0.5, y: yLev + aScale*0.3,
          text: `${{fx.toFixed(1)}} kN`, showarrow: false,
          font: {{ size: 11, color: '#FF9800', family: 'Arial Black' }},
        }});
      }}
    }});
  }}

  // Label traces (toggled by checkboxes)
  const showNIds = document.getElementById('chkNodeIds').checked;
  const showMIds = document.getElementById('chkMemberIds').checked;
  const showMNames = document.getElementById('chkMemberNames').checked;

  if (showNIds) {{
    const lx = [], ly = [], lt = [];
    geometry.nodes.forEach(n => {{
      lx.push(n.x_m !== undefined ? n.x_m : n.x);
      ly.push(n.y_m !== undefined ? n.y_m : n.y);
      lt.push('N' + n.id);
    }});
    traces.push({{
      x: lx, y: ly, mode: 'text', text: lt, textposition: 'top right',
      textfont: {{ size: 9, color: '#1a237e', family: 'monospace' }},
      showlegend: false, hoverinfo: 'skip', name: '_nodeLabels',
    }});
  }}

  if (showMIds || showMNames) {{
    const lx = [], ly = [], lt = [];
    geometry.main_connections.forEach(c => {{
      const a = nc[c.ni], b = nc[c.nj];
      if (!a || !b) return;
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      const dx = b.x - a.x, dy = b.y - a.y;
      const len = Math.sqrt(dx*dx + dy*dy);
      const offMag = th * 0.018;
      const offX = len > 0 ? -dy / len * offMag : offMag;
      const offY = len > 0 ? dx / len * offMag : 0;
      const mInfo = memberLookup[c.ni + '-' + c.nj];
      let label = '';
      if (showMNames && mInfo) {{ label = getMemberName(mInfo); }}
      else if (showMIds) {{ label = c.ni + '-' + c.nj; }}
      lx.push(mx + offX); ly.push(my + offY); lt.push(label);
    }});
    traces.push({{
      x: lx, y: ly, mode: 'text', text: lt, textposition: 'middle center',
      textfont: {{ size: 9, color: '#444', family: 'monospace' }},
      showlegend: false, hoverinfo: 'skip', name: '_memberLabels',
    }});
  }}

  // Dimension lines
  const dimOff = tw * 0.08;
  let yStart = 0;
  geometry.stories.forEach((sh, i) => {{
    const yEnd = yStart + sh;
    traces.push({{
      x: [tw + dimOff, tw + dimOff], y: [yStart, yEnd],
      mode: 'lines', line: {{ color: '#888', width: 1 }},
      showlegend: false, hoverinfo: 'skip',
    }});
    arrowAnnotations.push({{
      x: tw + dimOff*1.5, y: (yStart+yEnd)/2,
      text: `${{sh.toFixed(1)}}m`, showarrow: false,
      font: {{ size: 9, color: '#666' }},
    }});
    yStart = yEnd;
  }});
  let xStart = 0;
  geometry.bays.forEach((bw, i) => {{
    const xEnd = xStart + bw;
    traces.push({{
      x: [xStart, xEnd], y: [-dimOff, -dimOff],
      mode: 'lines', line: {{ color: '#888', width: 1 }},
      showlegend: false, hoverinfo: 'skip',
    }});
    arrowAnnotations.push({{
      x: (xStart+xEnd)/2, y: -dimOff*1.5,
      text: `${{bw.toFixed(1)}}m`, showarrow: false,
      font: {{ size: 9, color: '#666' }},
    }});
    xStart = xEnd;
  }});

  const marginX = tw * 0.25;
  const marginY = th * 0.2;

  Plotly.react('deformPlot', traces, {{
    title: {{ text: `Deformed Shape - ${{currentCase}}`, font: {{ size: 14 }} }},
    xaxis: {{ range: [-marginX, tw + marginX], scaleanchor: 'y', scaleratio: 1, showgrid: true, gridcolor: '#eee', zeroline: false, title: 'X (m)' }},
    yaxis: {{ range: [-marginY, th + marginY], showgrid: true, gridcolor: '#eee', zeroline: false, title: 'Y (m)' }},
    showlegend: true, legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)', bordercolor: '#ccc', borderwidth: 1 }},
    template: 'plotly_white', hovermode: 'closest',
    annotations: arrowAnnotations,
  }}, {{ responsive: true }});

  // Attach click handler once
  if (!deformClickAttached) {{
    document.getElementById('deformPlot').on('plotly_click', function(data) {{
      if (!data.points || data.points.length === 0) return;
      const pt = data.points[0];
      if (pt.customdata && pt.customdata.length > 0) {{
        const mId = pt.customdata[0];
        if (mId) selectMemberById(mId);
      }}
    }});
    deformClickAttached = true;
  }}

  renderLoadSummary();
}}

function renderLoadSummary() {{
  const div = document.getElementById('loadSummary');
  if (!div) return;
  const showLoads = document.getElementById('chkShowLoads').checked;
  if (!showLoads) {{ div.innerHTML = ''; return; }}

  let rows = [];
  let sumFx = 0, sumFy = 0;
  const comboFactors = loadCombinations[currentCase];
  if (comboFactors) {{
    Object.entries(comboFactors).forEach(([cn, factor]) => {{
      const cLoads = loadCasesRaw[cn] || [];
      cLoads.forEach(ld => {{
        const ltype = ld.type || '';
        const story = ld.story || '-';
        let desc = '', fx = 0, fy = 0;
        if (ltype === 'floor') {{
          const w = (ld.value || 0) * factor;
          fy = -w * geometry.total_width;
          desc = `Floor ${{story}}F: ${{w.toFixed(1)}} kN/m (${{factor}}&times;${{cn}})`;
        }} else if (ltype === 'lateral') {{
          fx = ((ld.fx !== undefined ? ld.fx : 0) * factor);
          desc = `Lateral ${{story}}F: ${{fx.toFixed(1)}} kN (${{factor}}&times;${{cn}})`;
        }}
        sumFx += fx; sumFy += fy;
        rows.push({{ story, desc, fx, fy }});
      }});
    }});
  }} else {{
    const loads = loadCasesRaw[currentCase] || [];
    loads.forEach(ld => {{
      const ltype = ld.type || '';
      const story = ld.story || '-';
      let desc = '', fx = 0, fy = 0;
      if (ltype === 'floor') {{
        const w = ld.value || 0;
        fy = -w * geometry.total_width;
        desc = `Floor ${{story}}F: ${{w}} kN/m`;
      }} else if (ltype === 'lateral') {{
        fx = ld.fx !== undefined ? ld.fx : 0;
        desc = `Lateral ${{story}}F: ${{fx}} kN`;
      }}
      sumFx += fx; sumFy += fy;
      rows.push({{ story, desc, fx, fy }});
    }});
  }}

  if (rows.length === 0) {{ div.innerHTML = ''; return; }}
  let html = '<h4>Load Summary - ' + currentCase + '</h4>';
  html += '<table style="width:auto;"><tr><th style="text-align:left;">Description</th><th>&Sigma;Fx (kN)</th><th>&Sigma;Fy (kN)</th></tr>';
  rows.forEach(r => {{
    html += `<tr><td style="text-align:left;">${{r.desc}}</td><td>${{r.fx.toFixed(1)}}</td><td>${{r.fy.toFixed(1)}}</td></tr>`;
  }});
  html += `<tr style="font-weight:bold;background:#e3f2fd;"><td style="text-align:left;">Total</td><td>${{sumFx.toFixed(1)}}</td><td>${{sumFy.toFixed(1)}}</td></tr>`;
  html += '</table>';
  div.innerHTML = html;
}}

// ============================================================
// Member Forces Tab
// ============================================================
function _populateFilterOptions() {{
  const mType = document.getElementById('mfTypeSelect').value;
  const storySel = document.getElementById('mfStoryFilter');
  const lineSel = document.getElementById('mfLineFilter');
  const prevStory = storySel.value;
  const prevLine = lineSel.value;

  // Collect unique stories and lines for current type
  const stories = new Set();
  const lines = new Set();
  memberInfo.forEach(m => {{
    if (m.type !== mType) return;
    const loc = m.location || {{}};
    if (loc.story !== undefined) stories.add(loc.story);
    if (mType === 'column' && loc.col !== undefined) lines.add(loc.col);
    if (mType === 'beam' && loc.bay !== undefined) lines.add(loc.bay);
  }});

  storySel.innerHTML = '<option value="all">All</option>';
  [...stories].sort((a, b) => a - b).forEach(s => {{
    const o = document.createElement('option');
    o.value = s; o.textContent = s + 'F';
    storySel.appendChild(o);
  }});

  const linePrefix = mType === 'column' ? 'C' : 'B';
  lineSel.innerHTML = '<option value="all">All</option>';
  [...lines].sort((a, b) => a - b).forEach(l => {{
    const o = document.createElement('option');
    o.value = l; o.textContent = linePrefix + (l + 1);
    lineSel.appendChild(o);
  }});

  // Restore previous if still valid
  if (prevStory !== 'all' && storySel.querySelector(`option[value="${{prevStory}}"]`)) storySel.value = prevStory;
  else storySel.value = 'all';
  if (prevLine !== 'all' && lineSel.querySelector(`option[value="${{prevLine}}"]`)) lineSel.value = prevLine;
  else lineSel.value = 'all';
}}

function populateMemberList() {{
  const mType = document.getElementById('mfTypeSelect').value;
  const storyFilter = document.getElementById('mfStoryFilter').value;
  const lineFilter = document.getElementById('mfLineFilter').value;
  const sel = document.getElementById('mfMemberSelect');
  const prevVal = sel.value;
  sel.innerHTML = '';
  memberInfo.forEach(m => {{
    if (m.type !== mType) return;
    const loc = m.location || {{}};
    // Story filter
    if (storyFilter !== 'all' && loc.story !== parseInt(storyFilter)) return;
    // Line/Bay filter
    if (lineFilter !== 'all') {{
      const lineIdx = parseInt(lineFilter);
      if (mType === 'column' && loc.col !== lineIdx) return;
      if (mType === 'beam' && loc.bay !== lineIdx) return;
    }}
    const mName = getMemberName(m);
    const label = (mType === 'column' ? 'Column' : 'Beam') + ': ' + mName + ' (N' + m.ni + '-N' + m.nj + ')';
    const o = document.createElement('option');
    o.value = m.id; o.textContent = label;
    sel.appendChild(o);
  }});
  if (prevVal && sel.querySelector(`option[value="${{prevVal}}"]`)) {{
    sel.value = prevVal;
  }}
  renderMemberForces();
  if (globalDiagramType) renderGlobalDiagram();
}}
function onMemberTypeChange() {{ _populateFilterOptions(); populateMemberList(); }}
function onMemberSelect() {{ renderMemberForces(); if (globalDiagramType) renderGlobalDiagram(); }}

function renderMemberForces() {{
  const mfData = memberForcesData[currentCase];
  if (!mfData) {{ Plotly.purge('memberForcePlot'); return; }}
  const selId = parseInt(document.getElementById('mfMemberSelect').value);
  selectedMemberId = isNaN(selId) ? null : selId;
  const mf = mfData.find(m => m.member_id === selId);
  if (!mf) {{ Plotly.purge('memberForcePlot'); return; }}

  const s = mf.s;
  const mInfo = memberInfo.find(mi => mi.id === mf.member_id);
  const mName = mInfo ? getMemberName(mInfo) : (mf.type + ' ' + mf.ni + '-' + mf.nj);
  const title = (mf.type === 'column' ? 'Column' : 'Beam') + ': ' + mName + ' (N' + mf.ni + '-N' + mf.nj + ', L=' + mf.length_m.toFixed(2) + 'm)';

  const subTraces = [
    {{ x: s, y: mf.N_kN, mode: 'lines', name: 'Axial N', fill: 'tozeroy', fillcolor: 'rgba(76,175,80,0.15)',
       line: {{ color: '#4CAF50', width: 2 }}, xaxis: 'x', yaxis: 'y',
       hovertemplate: 's=%{{x:.3f}}m<br>N=%{{y:.2f}} kN<extra></extra>' }},
    {{ x: s, y: mf.V_kN, mode: 'lines', name: 'Shear V', fill: 'tozeroy', fillcolor: 'rgba(244,67,54,0.15)',
       line: {{ color: '#F44336', width: 2 }}, xaxis: 'x2', yaxis: 'y2',
       hovertemplate: 's=%{{x:.3f}}m<br>V=%{{y:.2f}} kN<extra></extra>' }},
    {{ x: s, y: mf.M_kNm, mode: 'lines', name: 'Moment M', fill: 'tozeroy', fillcolor: 'rgba(33,150,243,0.15)',
       line: {{ color: '#2196F3', width: 2 }}, xaxis: 'x3', yaxis: 'y3',
       hovertemplate: 's=%{{x:.3f}}m<br>M=%{{y:.2f}} kN&middot;m<extra></extra>' }},
  ];

  // s-position marker shapes
  const sShapes = [];
  if (markedSPos !== null) {{
    ['y', 'y2', 'y3'].forEach(yr => {{
      sShapes.push({{
        type: 'line', xref: yr === 'y' ? 'x' : (yr === 'y2' ? 'x2' : 'x3'), yref: yr,
        x0: markedSPos, x1: markedSPos, y0: 0, y1: 1, yref: yr + ' domain',
        line: {{ color: '#FF6F00', width: 2, dash: 'dash' }},
      }});
    }});
  }}

  Plotly.react('memberForcePlot', subTraces, {{
    title: {{ text: `Member Forces - ${{currentCase}} - ${{title}}`, font: {{ size: 13 }} }},
    xaxis:  {{ domain: [0, 1], anchor: 'y',  title: '' }},
    xaxis2: {{ domain: [0, 1], anchor: 'y2', title: '' }},
    xaxis3: {{ domain: [0, 1], anchor: 'y3', title: 's (m)' }},
    yaxis:  {{ domain: [0.72, 1.0], title: 'N (kN)' }},
    yaxis2: {{ domain: [0.38, 0.66], title: 'V (kN)' }},
    yaxis3: {{ domain: [0.0, 0.30], title: 'M (kN&middot;m)' }},
    showlegend: true, template: 'plotly_white',
    legend: {{ x: 1.01, y: 1, xanchor: 'left' }},
    margin: {{ l: 60, r: 110, t: 40, b: 40 }},
    shapes: sShapes,
  }}, {{ responsive: true }});

  // Clear marker after rendering
  markedSPos = null;

  // K-3: Member summary table (end forces + extrema)
  renderMemberSummary(mf);
}}

function renderMemberSummary(mf) {{
  const div = document.getElementById('memberSummary');
  if (!mf) {{ div.innerHTML = ''; return; }}
  const s = mf.s, N = mf.N_kN, V = mf.V_kN, M = mf.M_kNm;
  const L = mf.length_m;
  const last = s.length - 1;

  // End forces
  const Ni = N[0], Nj = N[last], Vi = V[0], Vj = V[last], Mi = M[0], Mj = M[last];

  // Extrema
  let Nmax = -Infinity, Nmin = Infinity, Vmax = -Infinity, Vmin = Infinity, Mmax = -Infinity, Mmin = Infinity;
  let sNmax=0, sNmin=0, sVmax=0, sVmin=0, sMmax=0, sMmin=0;
  for (let i = 0; i < s.length; i++) {{
    if (N[i] > Nmax) {{ Nmax = N[i]; sNmax = s[i]; }}
    if (N[i] < Nmin) {{ Nmin = N[i]; sNmin = s[i]; }}
    if (V[i] > Vmax) {{ Vmax = V[i]; sVmax = s[i]; }}
    if (V[i] < Vmin) {{ Vmin = V[i]; sVmin = s[i]; }}
    if (M[i] > Mmax) {{ Mmax = M[i]; sMmax = s[i]; }}
    if (M[i] < Mmin) {{ Mmin = M[i]; sMmin = s[i]; }}
  }}

  const mInfo = memberInfo.find(mi => mi.id === mf.member_id);
  const mName = mInfo ? getMemberName(mInfo) : '';
  const f2 = v => v.toFixed(2);
  const f3 = v => v.toFixed(3);

  let html = `<h4>${{mName}} End Forces &amp; Extrema (L=${{f2(L)}}m)</h4>`;
  html += '<table><tr><th></th><th>N (kN)</th><th>V (kN)</th><th>M (kN&middot;m)</th></tr>';
  html += `<tr><td style="text-align:left;">i-end (s=0)</td><td>${{f2(Ni)}}</td><td>${{f2(Vi)}}</td><td>${{f2(Mi)}}</td></tr>`;
  html += `<tr><td style="text-align:left;">j-end (s=${{f2(L)}})</td><td>${{f2(Nj)}}</td><td>${{f2(Vj)}}</td><td>${{f2(Mj)}}</td></tr>`;
  html += `<tr style="background:#e3f2fd;"><td style="text-align:left;">Max</td>`;
  html += `<td>${{f2(Nmax)}} <small>@${{f3(sNmax)}}m</small></td>`;
  html += `<td>${{f2(Vmax)}} <small>@${{f3(sVmax)}}m</small></td>`;
  html += `<td>${{f2(Mmax)}} <small>@${{f3(sMmax)}}m</small></td></tr>`;
  html += `<tr style="background:#fff3e0;"><td style="text-align:left;">Min</td>`;
  html += `<td>${{f2(Nmin)}} <small>@${{f3(sNmin)}}m</small></td>`;
  html += `<td>${{f2(Vmin)}} <small>@${{f3(sVmin)}}m</small></td>`;
  html += `<td>${{f2(Mmin)}} <small>@${{f3(sMmin)}}m</small></td></tr>`;
  html += '</table>';
  html += '<div class="sign-note">Sign: N&gt;0 tension, V&gt;0 per element local, M&gt;0 sagging (tension at bottom).</div>';
  div.innerHTML = html;
}}

// ============================================================
// Label Toggle & Member Click
// ============================================================
function toggleLabels() {{
  if (currentTab === 'deformation') {{
    const cd = caseData[currentCase];
    if (cd) renderDeformation(cd);
  }}
}}

function selectMemberById(id) {{
  selectedMemberId = id;
  const m = memberInfo.find(mi => mi.id === id);
  if (!m) return;

  // Re-render deformation with highlight
  const cd = caseData[currentCase];
  if (cd && currentTab === 'deformation') renderDeformation(cd);

  // Update Member Forces dropdowns and switch tab
  document.getElementById('mfTypeSelect').value = m.type;
  populateMemberList();
  document.getElementById('mfMemberSelect').value = id;
  renderMemberForces();
  showTab('memberforces');
}}

// ============================================================
// Global Diagram
// ============================================================
function toggleGlobalDiagram(type) {{
  if (globalDiagramType === type) {{
    globalDiagramType = null;
    document.getElementById('globalMBtn').classList.remove('active');
    document.getElementById('globalVBtn').classList.remove('active');
    document.getElementById('globalNBtn').classList.remove('active');
    document.getElementById('globalDiagramContainer').style.display = 'none';
    return;
  }}
  globalDiagramType = type;
  document.getElementById('globalMBtn').classList.toggle('active', type === 'M');
  document.getElementById('globalVBtn').classList.toggle('active', type === 'V');
  document.getElementById('globalNBtn').classList.toggle('active', type === 'N');
  document.getElementById('globalDiagramContainer').style.display = 'block';
  renderGlobalDiagram();
}}

function renderGlobalDiagram() {{
  if (!globalDiagramType) return;
  const mfData = memberForcesData[currentCase];
  if (!mfData) return;

  const nc = geometry.node_coords;
  const traces = [];
  const annotations = [];
  const tw = geometry.total_width;
  const th = geometry.total_height;

  // Gray frame baseline
  geometry.main_connections.forEach(c => {{
    const a = nc[c.ni], b = nc[c.nj];
    if (!a || !b) return;
    traces.push({{
      x: [a.x, b.x], y: [a.y, b.y], mode: 'lines',
      line: {{ color: '#bbb', width: 2 }},
      showlegend: false, hoverinfo: 'skip',
    }});
  }});

  // Supports
  geometry.base_nodes.forEach(bn => {{
    const c = nc[bn];
    if (!c) return;
    const sz = th * 0.02;
    if (geometry.supports === 'fixed') {{
      traces.push({{
        x: [c.x-sz, c.x+sz, c.x+sz, c.x-sz, c.x-sz],
        y: [c.y, c.y, c.y-sz*0.8, c.y-sz*0.8, c.y],
        mode: 'lines', fill: 'toself', fillcolor: '#aaa',
        line: {{ color: '#888', width: 1 }}, showlegend: false, hoverinfo: 'skip',
      }});
    }} else {{
      traces.push({{
        x: [c.x, c.x-sz*0.5, c.x+sz*0.5, c.x],
        y: [c.y, c.y-sz, c.y-sz, c.y],
        mode: 'lines', fill: 'toself', fillcolor: 'white',
        line: {{ color: '#888', width: 1 }}, showlegend: false, hoverinfo: 'skip',
      }});
    }}
  }});

  // Find global max for scaling
  let maxVal = 0;
  mfData.forEach(mf => {{
    const arr = globalDiagramType === 'M' ? mf.M_kNm : (globalDiagramType === 'V' ? mf.V_kN : mf.N_kN);
    arr.forEach(v => {{ if (Math.abs(v) > maxVal) maxVal = Math.abs(v); }});
  }});
  if (maxVal === 0) maxVal = 1;
  const diagramScale = th * 0.12 / maxVal;

  const colorMap = {{ 'M': '#2196F3', 'V': '#F44336', 'N': '#4CAF50' }};
  const fillMap = {{ 'M': 'rgba(33,150,243,0.12)', 'V': 'rgba(244,67,54,0.12)', 'N': 'rgba(76,175,80,0.12)' }};
  const unitMap = {{ 'M': 'kN*m', 'V': 'kN', 'N': 'kN' }};
  const color = colorMap[globalDiagramType];
  const fillColor = fillMap[globalDiagramType];
  const unit = unitMap[globalDiagramType];

  // Draw diagram polygon per member
  const selId = parseInt(document.getElementById('mfMemberSelect').value);

  mfData.forEach(mf => {{
    const a = nc[mf.ni], b = nc[mf.nj];
    if (!a || !b) return;

    const arr = globalDiagramType === 'M' ? mf.M_kNm : (globalDiagramType === 'V' ? mf.V_kN : mf.N_kN);
    const sArr = mf.s;
    const L = mf.length_m;
    if (L === 0) return;

    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.sqrt(dx*dx + dy*dy);
    if (len === 0) return;
    const tx = dx / len, ty = dy / len;
    const nx = -ty, ny = tx;

    const polyX = [a.x], polyY = [a.y];
    let maxAbsVal = 0, maxIdx = 0;
    for (let i = 0; i < sArr.length; i++) {{
      const t = sArr[i] / L;
      const baseX = a.x + dx * t;
      const baseY = a.y + dy * t;
      polyX.push(baseX + nx * arr[i] * diagramScale);
      polyY.push(baseY + ny * arr[i] * diagramScale);
      if (Math.abs(arr[i]) > maxAbsVal) {{ maxAbsVal = Math.abs(arr[i]); maxIdx = i; }}
    }}
    polyX.push(b.x); polyY.push(b.y);

    const isSelected = mf.member_id === selId;
    const mInfo = memberInfo.find(mi => mi.id === mf.member_id);
    const mName = mInfo ? getMemberName(mInfo) : (mf.type + ' ' + mf.ni + '-' + mf.nj);

    traces.push({{
      x: polyX, y: polyY, mode: 'lines', fill: 'toself',
      fillcolor: isSelected ? (globalDiagramType === 'M' ? 'rgba(33,150,243,0.3)' : (globalDiagramType === 'V' ? 'rgba(244,67,54,0.3)' : 'rgba(76,175,80,0.3)')) : fillColor,
      line: {{ color: isSelected ? '#FF6F00' : color, width: isSelected ? 2 : 1 }},
      showlegend: false, hoverinfo: 'text',
      hovertext: mName + '<br>Max |' + globalDiagramType + '| = ' + maxAbsVal.toFixed(2) + ' ' + unit,
    }});

    // Max value label (only for significant values > 10% of global max)
    if (maxAbsVal > maxVal * 0.1) {{
      const t = sArr[maxIdx] / L;
      const baseX = a.x + dx * t;
      const baseY = a.y + dy * t;
      annotations.push({{
        x: baseX + nx * arr[maxIdx] * diagramScale,
        y: baseY + ny * arr[maxIdx] * diagramScale,
        text: arr[maxIdx].toFixed(1), showarrow: false,
        font: {{ size: 8, color: color }},
      }});
    }}
  }});

  const marginX = tw * 0.3;
  const marginY = th * 0.25;
  const titleMap = {{ 'M': 'Moment', 'V': 'Shear', 'N': 'Axial' }};

  Plotly.react('globalDiagramPlot', traces, {{
    title: {{ text: 'Global ' + titleMap[globalDiagramType] + ' Diagram - ' + currentCase + ' (' + unit + ')', font: {{ size: 13 }} }},
    xaxis: {{ range: [-marginX, tw + marginX], scaleanchor: 'y', scaleratio: 1, showgrid: true, gridcolor: '#eee', zeroline: false, title: 'X (m)' }},
    yaxis: {{ range: [-marginY, th + marginY], showgrid: true, gridcolor: '#eee', zeroline: false, title: 'Y (m)' }},
    showlegend: false, template: 'plotly_white', hovermode: 'closest',
    annotations: annotations,
  }}, {{ responsive: true }});
}}

// ============================================================
// Reactions Tab
// ============================================================
function renderReactions(cd) {{
  let html = '<table><tr><th>Node</th><th>X (m)</th><th>RX (kN)</th><th>RY (kN)</th><th>MZ (kN&middot;m)</th></tr>';
  let sumRX = 0, sumRY = 0, sumMZ = 0;
  cd.reactions.forEach(r => {{
    sumRX += r.RX_kN; sumRY += r.RY_kN; sumMZ += r.MZ_kNm;
    html += `<tr><td>${{r.node}}</td><td>${{r.x_m.toFixed(2)}}</td><td>${{r.RX_kN.toFixed(2)}}</td><td>${{r.RY_kN.toFixed(2)}}</td><td>${{r.MZ_kNm.toFixed(2)}}</td></tr>`;
  }});
  html += `<tr style="font-weight:bold;background:#e3f2fd;"><td colspan="2">&Sigma;</td><td>${{sumRX.toFixed(2)}}</td><td>${{sumRY.toFixed(2)}}</td><td>${{sumMZ.toFixed(2)}}</td></tr>`;
  html += '</table>';
  document.getElementById('reactionTable').innerHTML = html;
}}

function renderEquilibrium() {{
  const eq = eqData[currentCase];
  if (!eq) {{ document.getElementById('equilibriumPanel').innerHTML = ''; return; }}
  const passed = eq.all_passed;
  let html = `<div class="eq-panel ${{passed ? 'eq-ok' : 'eq-fail'}}">`;
  html += `<h3 style="margin-bottom:8px;">Equilibrium Check: ${{passed ? 'ALL PASSED' : 'FAILED'}}</h3>`;
  html += '<table><tr><th>Check</th><th>Applied</th><th>Reaction</th><th>Error</th><th>Status</th></tr>';

  const sh = eq.sum_horizontal || {{}};
  html += `<tr><td>${{sh.description||''}}</td><td>${{(sh.applied_kN||0).toFixed(3)}}</td><td>${{(sh.reaction_sum_kN||0).toFixed(3)}}</td><td>${{(sh.error_kN||0).toFixed(3)}}</td><td class="${{sh.status==='OK'?'status-ok':'status-fail'}}">${{sh.status||''}}</td></tr>`;

  const sv = eq.sum_vertical || {{}};
  html += `<tr><td>${{sv.description||''}}</td><td>${{(sv.applied_kN||0).toFixed(3)}}</td><td>${{(sv.reaction_sum_kN||0).toFixed(3)}}</td><td>${{(sv.error_kN||0).toFixed(3)}}</td><td class="${{sv.status==='OK'?'status-ok':'status-fail'}}">${{sv.status||''}}</td></tr>`;

  const sm = eq.sum_moment || {{}};
  html += `<tr><td>${{sm.description||''}}</td><td>${{(sm.applied_moment_kNm||0).toFixed(3)}}</td><td>${{(sm.reaction_moment_kNm||0).toFixed(3)}}</td><td>${{(sm.error_kNm||0).toFixed(3)}}</td><td class="${{sm.status==='OK'?'status-ok':'status-fail'}}">${{sm.status||''}}</td></tr>`;

  html += '</table></div>';
  document.getElementById('equilibriumPanel').innerHTML = html;
}}

// ============================================================
// Story Tab (K-1, K-2)
// ============================================================
function onDriftLimitChange() {{
  const sel = document.getElementById('driftLimitSelect');
  const customInput = document.getElementById('driftLimitCustom');
  if (sel.value === 'custom') {{
    customInput.style.display = '';
    driftLimitDenom = parseInt(customInput.value) || 400;
  }} else {{
    customInput.style.display = 'none';
    driftLimitDenom = parseInt(sel.value);
  }}
  const cd = caseData[currentCase];
  if (cd) renderStory(cd);
}}

function renderStory(cd) {{
  const sd = cd.story_data || {{}};
  const sDisp = sd.story_displacements || [];
  const sShear = sd.story_shears || [];
  const drifts = cd.story_drifts || [];
  const limit = 1.0 / driftLimitDenom;

  // Drift judgment
  let maxDrift = 0, maxStory = 0;
  drifts.forEach(d => {{ if (d.drift > maxDrift) {{ maxDrift = d.drift; maxStory = d.story; }} }});
  const invMax = maxDrift > 0 ? Math.round(1/maxDrift) : Infinity;
  const passed = maxDrift <= limit;
  const jDiv = document.getElementById('driftJudgment');
  jDiv.className = 'drift-judgment ' + (passed ? 'ok' : 'ng');
  jDiv.textContent = 'Max drift = 1/' + (invMax === Infinity ? '\u221e' : invMax) +
    ' at Story ' + maxStory + ', Limit = 1/' + driftLimitDenom +
    ' \u2192 ' + (passed ? 'OK' : 'Exceeds Limit');

  // Story displacement profile
  const dispValues = [0, ...sDisp.map(d => d.dx_avg_mm)];
  const yValues = [0, ...sDisp.map(d => d.y_m)];
  const storyLabels = ['Base', ...sDisp.map(d => 'Story ' + d.story)];

  Plotly.react('storyDispPlot', [{{
    x: dispValues, y: yValues, mode: 'lines+markers',
    line: {{ color: '#1a237e', width: 2 }},
    marker: {{ size: 8, color: '#1a237e' }},
    text: storyLabels,
    hovertemplate: '%{{text}}<br>Disp: %{{x:.3f}} mm<br>Height: %{{y:.2f}} m<extra></extra>',
  }}], {{
    title: {{ text: `Story Displacement Profile - ${{currentCase}}`, font: {{ size: 13 }} }},
    xaxis: {{ title: 'Average Horizontal Displacement (mm)' }},
    yaxis: {{ title: 'Height (m)' }},
    template: 'plotly_white', showlegend: false,
  }}, {{ responsive: true }});

  // Story shear bar chart (O: dual-method with reaction-based)
  const useAbs = document.getElementById('chkAbsShear').checked;
  const shearMethod = document.getElementById('shearMethodSelect').value;
  const hasLateral = sShear.length > 0 && sShear[0].has_lateral;
  const shearNote = document.getElementById('shearNote');

  // Gravity-case label change
  const shearTitle = hasLateral ? 'Story Shear' : 'Column Cut Force (no lateral load)';
  if (!hasLateral && sShear.length > 0) {{
    shearNote.style.display = 'block';
    shearNote.textContent = 'No lateral loads applied in this case. Values represent column shear cut forces, not conventional story shear.';
  }} else {{
    shearNote.style.display = 'none';
  }}

  if (sShear.length > 0) {{
    const shearVals = sShear.map(s => {{
      if (shearMethod === 'reaction') return s.shear_rxn_kN !== undefined ? s.shear_rxn_kN : 0;
      if (shearMethod === 'element_abs') return s.shear_kN_abs !== undefined ? s.shear_kN_abs : s.shear_kN;
      return s.shear_kN_signed !== undefined ? s.shear_kN_signed : s.shear_kN;
    }});
    const methodLabel = shearMethod === 'reaction' ? 'Reaction' :
                        shearMethod === 'element_abs' ? 'Element |V|' : 'Element Signed';
    const barColor = shearMethod === 'reaction' ? '#43A047' :
                     shearMethod === 'element_abs' ? '#FF7043' : '#F44336';

    // Enhanced tooltip with all 3 values
    const hoverTexts = sShear.map(s => {{
      const rxn = s.shear_rxn_kN !== undefined ? s.shear_rxn_kN : 0;
      const signed = s.shear_kN_signed !== undefined ? s.shear_kN_signed : s.shear_kN;
      const absV = s.shear_kN_abs !== undefined ? s.shear_kN_abs : s.shear_kN;
      return 'Story ' + s.story +
        '<br>Reaction-based: ' + rxn.toFixed(2) + ' kN' +
        '<br>Element (signed): ' + signed.toFixed(2) + ' kN' +
        '<br>Element (|V|): ' + absV.toFixed(2) + ' kN';
    }});
    Plotly.react('storyShearPlot', [{{
      x: shearVals,
      y: sShear.map(s => 'Story ' + s.story),
      type: 'bar', orientation: 'h',
      marker: {{ color: barColor }},
      hoverinfo: 'text', hovertext: hoverTexts,
    }}], {{
      title: {{ text: `${{shearTitle}} (${{methodLabel}}) - ${{currentCase}}`, font: {{ size: 13 }} }},
      xaxis: {{ title: (hasLateral ? 'Story Shear' : 'Cut Force') + ' (kN)' }},
      yaxis: {{ title: '' }},
      template: 'plotly_white', showlegend: false,
    }}, {{ responsive: true }});

    // Click handler → show column cut force detail
    const shearPlotEl = document.getElementById('storyShearPlot');
    shearPlotEl.removeAllListeners && shearPlotEl.removeAllListeners('plotly_click');
    shearPlotEl.on('plotly_click', function(evtData) {{
      if (!evtData || !evtData.points || !evtData.points.length) return;
      const pt = evtData.points[0];
      const storyIdx = sShear.length - pt.pointIndex;
      _showColumnDetail(storyIdx, cd);
    }});
  }}

  // Column detail panel (hidden until bar click)
  document.getElementById('storyShearDetail').style.display = 'none';

  // Shear method consistency check per story (reaction vs element)
  const shearConsistency = _checkShearConsistency(sShear, hasLateral);

  // Drift table (K-1: user-defined limit) + shear consistency badge
  let html = '<table><tr><th>Story</th><th>Height (m)</th><th>Drift Ratio</th><th>1/Drift</th><th>Limit 1/' + driftLimitDenom + '</th><th>Status</th><th>Shear Check</th></tr>';
  drifts.forEach(d => {{
    const invDrift = d.drift > 0 ? Math.round(1/d.drift) : Infinity;
    const ok = d.drift <= limit;
    const cls = ok ? (invDrift > driftLimitDenom * 1.5 ? 'drift-green' : 'drift-yellow') : 'drift-red';
    const status = ok ? 'OK' : 'Exceeds';
    const sc = shearConsistency[d.story];
    let shearBadge = '';
    if (sc && sc.warn) {{
      shearBadge = `<span class="shear-warn-badge" title="${{sc.tip}}">\u26a0 Check</span>`;
    }} else if (sc && !sc.skip) {{
      shearBadge = '<span style="color:#2e7d32;font-size:11px;">OK</span>';
    }}
    html += `<tr class="${{cls}}"><td>Story ${{d.story}}</td><td>${{d.height_m.toFixed(1)}}</td><td>${{d.drift.toFixed(6)}}</td><td>1/${{invDrift === Infinity ? '&infin;' : invDrift}}</td><td>1/${{driftLimitDenom}}</td><td>${{status}}</td><td style="text-align:center;">${{shearBadge}}</td></tr>`;
  }});
  // Base Shear row from reactions
  const baseShearRxn = (cd.reactions || []).reduce((s, r) => s + (r.RX_kN || 0), 0);
  html += `<tr style="border-top:2px solid #333;font-weight:600;"><td colspan="2">Base Shear (\u03A3RX)</td><td colspan="5">${{baseShearRxn.toFixed(2)}} kN</td></tr>`;
  html += '</table>';
  document.getElementById('driftTable').innerHTML = html;

  // Design review notes
  _renderDesignReview(drifts, sShear, limit, hasLateral);
}}

// Shear method consistency: compare reaction-based vs element-based per story
function _checkShearConsistency(sShear, hasLateral) {{
  const result = {{}};
  const threshold = 0.05; // 5% relative error
  sShear.forEach(s => {{
    const story = s.story;
    if (!hasLateral) {{
      result[story] = {{ warn: false, skip: true, tip: '' }};
      return;
    }}
    const rxn = s.shear_rxn_kN !== undefined ? s.shear_rxn_kN : null;
    const signed = s.shear_kN_signed !== undefined ? s.shear_kN_signed : null;
    const absV = s.shear_kN_abs !== undefined ? s.shear_kN_abs : null;
    if (rxn === null || rxn === 0) {{
      result[story] = {{ warn: false, skip: true, tip: '' }};
      return;
    }}
    const errSigned = Math.abs((signed - rxn) / rxn);
    const errAbs = Math.abs((absV - Math.abs(rxn)) / Math.abs(rxn));
    let warns = [];
    if (errSigned > threshold) {{
      warns.push('Element signed vs Reaction: ' + (errSigned * 100).toFixed(1) + '% — column shears may have opposing signs');
    }}
    if (errAbs > threshold) {{
      warns.push('Element |V| vs |Reaction|: ' + (errAbs * 100).toFixed(1) + '% — check section plane definition');
    }}
    result[story] = {{ warn: warns.length > 0, skip: false, tip: warns.join('; ') }};
  }});
  return result;
}}

// Column cut force detail for a clicked story bar
function _showColumnDetail(storyIdx, cd) {{
  const mfData = memberForcesData[currentCase] || [];
  const cols = mfData.filter(m => m.type === 'column' && m.location && m.location.story === storyIdx);
  const detailDiv = document.getElementById('storyShearDetail');
  if (!cols.length) {{ detailDiv.style.display = 'none'; return; }}

  let tbl = `<h4>Story ${{storyIdx}} \u2014 Column Cut Forces (${{currentCase}})</h4>`;
  tbl += '<table><tr><th>Column</th><th>V_i (kN)</th><th>V_j (kN)</th><th>N (kN)</th></tr>';
  let sumVi = 0, sumVj = 0;
  cols.forEach(m => {{
    const name = getMemberName(m);
    const Vi = m.V_kN ? m.V_kN[0] : 0;
    const Vj = m.V_kN ? m.V_kN[m.V_kN.length - 1] : 0;
    const Ni = m.N_kN ? m.N_kN[0] : 0;
    sumVi += Vi;
    sumVj += Vj;
    tbl += `<tr><td>${{name}}</td><td>${{Vi.toFixed(2)}}</td><td>${{Vj.toFixed(2)}}</td><td>${{Ni.toFixed(2)}}</td></tr>`;
  }});
  tbl += `<tr class="sum-row"><td>\u03A3 (i-end)</td><td>${{sumVi.toFixed(2)}}</td><td>${{sumVj.toFixed(2)}}</td><td></td></tr>`;
  tbl += '</table>';
  tbl += `<div class="note">Story shear = sum of column cut forces at the section plane. V_i = shear at column base (i-end), V_j = shear at column top (j-end).</div>`;
  detailDiv.innerHTML = tbl;
  detailDiv.style.display = 'block';
}}

// Design review — informational drift and shear trend checks
function _renderDesignReview(drifts, sShear, limit, hasLateral) {{
  const div = document.getElementById('designReview');
  if (!drifts.length) {{ div.innerHTML = ''; return; }}

  let items = '';

  // 1. Drift checks against multiple reference limits
  const maxDrift = Math.max(...drifts.map(d => d.drift));
  const invMax = maxDrift > 0 ? 1 / maxDrift : Infinity;

  const driftChecks = [
    {{ denom: 400, label: 'H/400 (general)' }},
    {{ denom: 500, label: 'H/500 (sensitive finish)' }},
  ];
  driftChecks.forEach(chk => {{
    const lim = 1.0 / chk.denom;
    let cls, txt;
    if (maxDrift <= lim * 0.8) {{ cls = 'review-ok'; txt = 'OK'; }}
    else if (maxDrift <= lim) {{ cls = 'review-note'; txt = 'Review'; }}
    else {{ cls = 'review-over'; txt = 'Exceeds'; }}
    items += `<div class="review-item"><span class="review-label">Drift vs ${{chk.label}} (1/${{chk.denom}}): max 1/${{invMax === Infinity ? '\u221e' : Math.round(invMax)}}</span><span class="review-status ${{cls}}">${{txt}}</span></div>`;
  }});

  // 2. Story shear trend consistency (for lateral cases only)
  if (hasLateral && sShear.length > 1) {{
    const rxnVals = sShear.map(s => s.shear_rxn_kN || 0);
    let monotonic = true;
    for (let i = 0; i < rxnVals.length - 1; i++) {{
      if (Math.abs(rxnVals[i]) < Math.abs(rxnVals[i + 1]) - 0.01) {{ monotonic = false; break; }}
    }};
    const cls = monotonic ? 'review-ok' : 'review-note';
    const txt = monotonic ? 'OK' : 'Review';
    const desc = monotonic
      ? 'Story shear increases monotonically toward base \u2014 consistent with typical lateral load distribution.'
      : 'Story shear does not increase monotonically toward base. Verify applied lateral load pattern.';
    items += `<div class="review-item"><span class="review-label">${{desc}}</span><span class="review-status ${{cls}}">${{txt}}</span></div>`;
  }}

  // 3. Soft-story drift pattern check
  if (drifts.length > 1) {{
    const driftVals = drifts.map(d => d.drift);
    const softStories = [];
    for (let i = 0; i < driftVals.length; i++) {{
      const below = i > 0 ? driftVals[i - 1] : null;
      const above = i < driftVals.length - 1 ? driftVals[i + 1] : null;
      const neighbors = [below, above].filter(v => v !== null && v > 0);
      if (neighbors.length > 0 && driftVals[i] > 0) {{
        const avgNeighbor = neighbors.reduce((a, b) => a + b, 0) / neighbors.length;
        if (driftVals[i] > avgNeighbor * 1.5) {{
          softStories.push(drifts[i].story);
        }}
      }}
    }}
    if (softStories.length > 0) {{
      items += `<div class="review-item"><span class="review-label">Possible soft-story behavior at Story ${{softStories.join(', ')}} \u2014 drift ratio exceeds 1.5\u00d7 adjacent stories. Check stiffness distribution.</span><span class="review-status review-note">Review</span></div>`;
    }}
  }}

  // 4. Shear irregularity check (for lateral cases)
  if (hasLateral && sShear.length > 1) {{
    const rxnVals = sShear.map(s => Math.abs(s.shear_rxn_kN || 0));
    const irregStories = [];
    for (let i = 1; i < rxnVals.length; i++) {{
      if (rxnVals[i - 1] > 0 && rxnVals[i] > rxnVals[i - 1] * 1.5) {{
        irregStories.push(sShear[i].story);
      }}
    }}
    if (irregStories.length > 0) {{
      items += `<div class="review-item"><span class="review-label">Story shear jump at Story ${{irregStories.join(', ')}} \u2014 check lateral force distribution or stiffness irregularity.</span><span class="review-status review-note">Review</span></div>`;
    }}
  }}

  if (items) {{
    div.innerHTML = '<h4>Design Review Notes</h4>' + items;
  }} else {{
    div.innerHTML = '';
  }}
}}

// ============================================================
// Envelope Tab (Phase L)
// ============================================================
function computeEnvelope() {{
  const env = {{ displacement: [], drift: [], memberforces: [], reactions: [] }};
  const nodeMap = {{}};
  allNames.forEach(cn => {{
    const cd = caseData[cn];
    if (!cd) return;
    cd.nodal_displacements.forEach(d => {{
      const key = d.node;
      if (!nodeMap[key]) nodeMap[key] = {{ node: d.node, x_m: d.x_m, y_m: d.y_m,
        dx_max: -Infinity, dx_min: Infinity, dy_max: -Infinity, dy_min: Infinity,
        dx_max_case: '', dx_min_case: '', dy_max_case: '', dy_min_case: '' }};
      const nm = nodeMap[key];
      if (d.dx_mm > nm.dx_max) {{ nm.dx_max = d.dx_mm; nm.dx_max_case = cn; }}
      if (d.dx_mm < nm.dx_min) {{ nm.dx_min = d.dx_mm; nm.dx_min_case = cn; }}
      if (d.dy_mm > nm.dy_max) {{ nm.dy_max = d.dy_mm; nm.dy_max_case = cn; }}
      if (d.dy_mm < nm.dy_min) {{ nm.dy_min = d.dy_mm; nm.dy_min_case = cn; }}
    }});
  }});
  env.displacement = Object.values(nodeMap);

  const driftMap = {{}};
  allNames.forEach(cn => {{
    const cd = caseData[cn];
    if (!cd) return;
    (cd.story_drifts || []).forEach(d => {{
      const key = d.story;
      if (!driftMap[key]) driftMap[key] = {{ story: d.story, height_m: d.height_m,
        drift_max: 0, drift_max_case: '' }};
      if (d.drift > driftMap[key].drift_max) {{ driftMap[key].drift_max = d.drift; driftMap[key].drift_max_case = cn; }}
    }});
  }});
  env.drift = Object.values(driftMap).sort((a,b) => a.story - b.story);

  const mfMap = {{}};
  allNames.forEach(cn => {{
    const mfData = memberForcesData[cn];
    if (!mfData) return;
    mfData.forEach(mf => {{
      const key = mf.member_id;
      if (!mfMap[key]) {{
        const mi = memberInfo.find(m => m.id === key);
        mfMap[key] = {{ id: key, type: mf.type, ni: mf.ni, nj: mf.nj, name: mi ? getMemberName(mi) : '',
          location: mi ? mi.location : {{}}, length_m: mi ? mi.length_m : 0,
          Nmax: -Infinity, Nmin: Infinity, Vmax: -Infinity, Vmin: Infinity, Mmax: -Infinity, Mmin: Infinity,
          Nmax_case:'', Nmin_case:'', Vmax_case:'', Vmin_case:'', Mmax_case:'', Mmin_case:'',
          sNmax:0, sNmin:0, sVmax:0, sVmin:0, sMmax:0, sMmin:0 }};
      }}
      const em = mfMap[key];
      for (let i = 0; i < mf.s.length; i++) {{
        if (mf.N_kN[i] > em.Nmax) {{ em.Nmax = mf.N_kN[i]; em.Nmax_case = cn; em.sNmax = mf.s[i]; }}
        if (mf.N_kN[i] < em.Nmin) {{ em.Nmin = mf.N_kN[i]; em.Nmin_case = cn; em.sNmin = mf.s[i]; }}
        if (mf.V_kN[i] > em.Vmax) {{ em.Vmax = mf.V_kN[i]; em.Vmax_case = cn; em.sVmax = mf.s[i]; }}
        if (mf.V_kN[i] < em.Vmin) {{ em.Vmin = mf.V_kN[i]; em.Vmin_case = cn; em.sVmin = mf.s[i]; }}
        if (mf.M_kNm[i] > em.Mmax) {{ em.Mmax = mf.M_kNm[i]; em.Mmax_case = cn; em.sMmax = mf.s[i]; }}
        if (mf.M_kNm[i] < em.Mmin) {{ em.Mmin = mf.M_kNm[i]; em.Mmin_case = cn; em.sMmin = mf.s[i]; }}
      }}
    }});
  }});
  env.memberforces = Object.values(mfMap);

  const rxnMap = {{}};
  allNames.forEach(cn => {{
    const cd = caseData[cn];
    if (!cd) return;
    cd.reactions.forEach(r => {{
      const key = r.node;
      if (!rxnMap[key]) rxnMap[key] = {{ node: r.node, x_m: r.x_m,
        RX_max: -Infinity, RX_min: Infinity, RY_max: -Infinity, RY_min: Infinity, MZ_max: -Infinity, MZ_min: Infinity,
        RX_max_case:'', RX_min_case:'', RY_max_case:'', RY_min_case:'', MZ_max_case:'', MZ_min_case:'' }};
      const rm = rxnMap[key];
      if (r.RX_kN > rm.RX_max) {{ rm.RX_max = r.RX_kN; rm.RX_max_case = cn; }}
      if (r.RX_kN < rm.RX_min) {{ rm.RX_min = r.RX_kN; rm.RX_min_case = cn; }}
      if (r.RY_kN > rm.RY_max) {{ rm.RY_max = r.RY_kN; rm.RY_max_case = cn; }}
      if (r.RY_kN < rm.RY_min) {{ rm.RY_min = r.RY_kN; rm.RY_min_case = cn; }}
      if (r.MZ_kNm > rm.MZ_max) {{ rm.MZ_max = r.MZ_kNm; rm.MZ_max_case = cn; }}
      if (r.MZ_kNm < rm.MZ_min) {{ rm.MZ_min = r.MZ_kNm; rm.MZ_min_case = cn; }}
    }});
  }});
  env.reactions = Object.values(rxnMap);
  return env;
}}

function renderEnvelope() {{
  const cat = document.getElementById('envCategory').value;
  const mTypeFilter = document.getElementById('envMemberType').value;
  const sortBy = document.getElementById('envSort').value;
  document.getElementById('envMemberType').style.display = cat === 'memberforces' ? '' : 'none';
  document.getElementById('envSort').style.display = cat === 'memberforces' ? '' : 'none';

  const env = computeEnvelope();
  const div = document.getElementById('envelopeContent');
  const f2 = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) : '-';
  const f3 = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(3) : '-';
  let html = '';

  if (cat === 'displacement') {{
    html = '<table><tr><th>Node</th><th>X(m)</th><th>Y(m)</th><th>dx max</th><th>Case</th><th>dx min</th><th>Case</th><th>dy max</th><th>Case</th><th>dy min</th><th>Case</th></tr>';
    env.displacement.sort((a,b) => a.node - b.node).forEach(d => {{
      html += `<tr style="cursor:pointer;" onclick="goToCase('${{d.dx_max_case}}')"><td>${{d.node}}</td><td>${{f2(d.x_m)}}</td><td>${{f2(d.y_m)}}</td>`;
      html += `<td>${{f3(d.dx_max)}}</td><td style="font-size:10px;">${{d.dx_max_case}}</td>`;
      html += `<td>${{f3(d.dx_min)}}</td><td style="font-size:10px;">${{d.dx_min_case}}</td>`;
      html += `<td>${{f3(d.dy_max)}}</td><td style="font-size:10px;">${{d.dy_max_case}}</td>`;
      html += `<td>${{f3(d.dy_min)}}</td><td style="font-size:10px;">${{d.dy_min_case}}</td></tr>`;
    }});
    html += '</table>';
  }} else if (cat === 'drift') {{
    const limit = 1.0 / driftLimitDenom;
    html = '<table><tr><th>Story</th><th>Height(m)</th><th>Max Drift</th><th>1/Drift</th><th>Case</th><th>Limit 1/' + driftLimitDenom + '</th><th>Status</th></tr>';
    env.drift.forEach(d => {{
      const inv = d.drift_max > 0 ? Math.round(1/d.drift_max) : Infinity;
      const ok = d.drift_max <= limit;
      const cls = ok ? 'drift-green' : 'drift-red';
      html += `<tr class="${{cls}}" style="cursor:pointer;" onclick="goToCase('${{d.drift_max_case}}')"><td>Story ${{d.story}}</td><td>${{f2(d.height_m)}}</td><td>${{d.drift_max.toFixed(6)}}</td><td>1/${{inv === Infinity ? '&infin;' : inv}}</td><td style="font-size:10px;">${{d.drift_max_case}}</td><td>1/${{driftLimitDenom}}</td><td>${{ok?'OK':'Exceeds'}}</td></tr>`;
    }});
    html += '</table>';
  }} else if (cat === 'memberforces') {{
    let data = env.memberforces;
    if (mTypeFilter !== 'all') data = data.filter(m => m.type === mTypeFilter);
    if (sortBy === 'Mmax_desc') data.sort((a,b) => Math.max(Math.abs(b.Mmax),Math.abs(b.Mmin)) - Math.max(Math.abs(a.Mmax),Math.abs(a.Mmin)));
    else if (sortBy === 'Mpos_desc') data.sort((a,b) => b.Mmax - a.Mmax);
    else if (sortBy === 'Mneg_desc') data.sort((a,b) => a.Mmin - b.Mmin);
    else if (sortBy === 'Vmax_desc') data.sort((a,b) => Math.max(Math.abs(b.Vmax),Math.abs(b.Vmin)) - Math.max(Math.abs(a.Vmax),Math.abs(a.Vmin)));
    else if (sortBy === 'Nmax_desc') data.sort((a,b) => Math.max(Math.abs(b.Nmax),Math.abs(b.Nmin)) - Math.max(Math.abs(a.Nmax),Math.abs(a.Nmin)));

    html = '<table><tr><th>Member</th><th>Type</th><th>Location</th><th>L(m)</th>';
    html += '<th>Nmax(kN)</th><th>@s/Case</th><th>Nmin(kN)</th><th>@s/Case</th>';
    html += '<th>Vmax(kN)</th><th>@s/Case</th><th>Vmin(kN)</th><th>@s/Case</th>';
    html += '<th>Mmax(kN*m)</th><th>@s/Case</th><th>Mmin(kN*m)</th><th>@s/Case</th></tr>';
    data.forEach(m => {{
      const bg = m.id === selectedMemberId ? 'background:#fff3e0;' : '';
      const loc = m.location || {{}};
      const locStr = m.type === 'column' ? `${{loc.story||''}}F-C${{(loc.col||0)+1}}` : `${{loc.story||''}}F-B${{(loc.bay||0)+1}}`;
      const esc = c => c.replace(/'/g, "\\\\'");
      html += `<tr style="${{bg}}">`;
      html += `<td>${{m.name}}</td><td>${{m.type}}</td><td>${{locStr}}</td><td>${{f2(m.length_m)}}</td>`;
      html += `<td style="cursor:pointer;color:#1565c0;" onclick="goToMember(${{m.id}},'${{esc(m.Nmax_case)}}',${{m.sNmax}})">${{f2(m.Nmax)}}</td>`;
      html += `<td style="font-size:10px;">${{f3(m.sNmax)}}m/${{m.Nmax_case}}</td>`;
      html += `<td style="cursor:pointer;color:#1565c0;" onclick="goToMember(${{m.id}},'${{esc(m.Nmin_case)}}',${{m.sNmin}})">${{f2(m.Nmin)}}</td>`;
      html += `<td style="font-size:10px;">${{f3(m.sNmin)}}m/${{m.Nmin_case}}</td>`;
      html += `<td style="cursor:pointer;color:#1565c0;" onclick="goToMember(${{m.id}},'${{esc(m.Vmax_case)}}',${{m.sVmax}})">${{f2(m.Vmax)}}</td>`;
      html += `<td style="font-size:10px;">${{f3(m.sVmax)}}m/${{m.Vmax_case}}</td>`;
      html += `<td style="cursor:pointer;color:#1565c0;" onclick="goToMember(${{m.id}},'${{esc(m.Vmin_case)}}',${{m.sVmin}})">${{f2(m.Vmin)}}</td>`;
      html += `<td style="font-size:10px;">${{f3(m.sVmin)}}m/${{m.Vmin_case}}</td>`;
      html += `<td style="cursor:pointer;color:#1565c0;" onclick="goToMember(${{m.id}},'${{esc(m.Mmax_case)}}',${{m.sMmax}})">${{f2(m.Mmax)}}</td>`;
      html += `<td style="font-size:10px;">${{f3(m.sMmax)}}m/${{m.Mmax_case}}</td>`;
      html += `<td style="cursor:pointer;color:#1565c0;" onclick="goToMember(${{m.id}},'${{esc(m.Mmin_case)}}',${{m.sMmin}})">${{f2(m.Mmin)}}</td>`;
      html += `<td style="font-size:10px;">${{f3(m.sMmin)}}m/${{m.Mmin_case}}</td>`;
      html += '</tr>';
    }});
    html += '</table>';
  }} else if (cat === 'reactions') {{
    html = '<table><tr><th>Node</th><th>X(m)</th><th>RX max</th><th>Case</th><th>RX min</th><th>Case</th>';
    html += '<th>RY max</th><th>Case</th><th>RY min</th><th>Case</th>';
    html += '<th>MZ max</th><th>Case</th><th>MZ min</th><th>Case</th></tr>';
    env.reactions.sort((a,b) => a.node - b.node).forEach(r => {{
      html += `<tr style="cursor:pointer;" onclick="goToCase('${{r.RY_max_case}}')"><td>N${{r.node}}</td><td>${{f2(r.x_m)}}</td>`;
      html += `<td>${{f2(r.RX_max)}}</td><td style="font-size:10px;">${{r.RX_max_case}}</td>`;
      html += `<td>${{f2(r.RX_min)}}</td><td style="font-size:10px;">${{r.RX_min_case}}</td>`;
      html += `<td>${{f2(r.RY_max)}}</td><td style="font-size:10px;">${{r.RY_max_case}}</td>`;
      html += `<td>${{f2(r.RY_min)}}</td><td style="font-size:10px;">${{r.RY_min_case}}</td>`;
      html += `<td>${{f2(r.MZ_max)}}</td><td style="font-size:10px;">${{r.MZ_max_case}}</td>`;
      html += `<td>${{f2(r.MZ_min)}}</td><td style="font-size:10px;">${{r.MZ_min_case}}</td></tr>`;
    }});
    html += '</table>';
  }}
  div.innerHTML = html;
}}

function goToCase(caseName) {{
  if (!caseName || !caseData[caseName]) return;
  document.getElementById('caseSelect').value = caseName;
  currentCase = caseName;
  showTab('deformation');
}}

function goToMember(memberId, caseName, sPos) {{
  if (caseName && caseData[caseName]) {{
    document.getElementById('caseSelect').value = caseName;
    currentCase = caseName;
  }}
  markedSPos = (typeof sPos === 'number') ? sPos : null;
  selectMemberById(memberId);
}}

// ============================================================
// Export Tab (M-1, M-2)
// ============================================================
function exportCSV(type) {{
  const cd = caseData[currentCase];
  if (!cd) return;
  let csv = '';
  const name = currentCase.replace(/[^a-zA-Z0-9]/g, '_');

  if (type === 'nodes') {{
    csv = 'node,x_m,y_m,dx_mm,dy_mm,rz_rad\\n';
    cd.nodal_displacements.forEach(d => {{
      csv += `${{d.node}},${{d.x_m}},${{d.y_m}},${{d.dx_mm}},${{d.dy_mm}},${{d.rz_rad}}\\n`;
    }});
  }} else if (type === 'reactions') {{
    csv = 'node,x_m,RX_kN,RY_kN,MZ_kNm\\n';
    cd.reactions.forEach(r => {{
      csv += `${{r.node}},${{r.x_m}},${{r.RX_kN}},${{r.RY_kN}},${{r.MZ_kNm}}\\n`;
    }});
  }} else if (type === 'members') {{
    const mfData = memberForcesData[currentCase] || [];
    csv = 'member_id,type,ni,nj,s_m,N_kN,V_kN,M_kNm\\n';
    mfData.forEach(mf => {{
      for (let i = 0; i < mf.s.length; i++) {{
        csv += `${{mf.member_id}},${{mf.type}},${{mf.ni}},${{mf.nj}},${{mf.s[i]}},${{mf.N_kN[i]}},${{mf.V_kN[i]}},${{mf.M_kNm[i]}}\\n`;
      }}
    }});
  }} else if (type === 'story') {{
    const sd = cd.story_data || {{}};
    csv = 'story,y_m,dx_left_mm,dx_right_mm,dx_avg_mm,drift,shear_signed_kN,shear_abs_kN\\n';
    const sDisp = sd.story_displacements || [];
    const sShear = sd.story_shears || [];
    const drifts = cd.story_drifts || [];
    for (let i = 0; i < sDisp.length; i++) {{
      const d = sDisp[i];
      const sh = sShear[i] || {{}};
      const dr = drifts[i] || {{}};
      const sSigned = sh.shear_kN_signed !== undefined ? sh.shear_kN_signed : sh.shear_kN;
      const sAbs = sh.shear_kN_abs !== undefined ? sh.shear_kN_abs : sh.shear_kN;
      csv += `${{d.story}},${{d.y_m}},${{d.dx_left_mm}},${{d.dx_right_mm}},${{d.dx_avg_mm}},${{dr.drift||0}},${{sSigned}},${{sAbs}}\\n`;
    }}
  }} else if (type === 'extrema') {{
    const env = computeEnvelope();
    csv = 'member,type,ni,nj,Nmax_kN,sNmax_m,Nmax_case,Nmin_kN,sNmin_m,Nmin_case,Vmax_kN,sVmax_m,Vmax_case,Vmin_kN,sVmin_m,Vmin_case,Mmax_kNm,sMmax_m,Mmax_case,Mmin_kNm,sMmin_m,Mmin_case\\n';
    env.memberforces.forEach(m => {{
      csv += `${{m.name}},${{m.type}},${{m.ni}},${{m.nj}},${{m.Nmax}},${{m.sNmax}},${{m.Nmax_case}},${{m.Nmin}},${{m.sNmin}},${{m.Nmin_case}},${{m.Vmax}},${{m.sVmax}},${{m.Vmax_case}},${{m.Vmin}},${{m.sVmin}},${{m.Vmin_case}},${{m.Mmax}},${{m.sMmax}},${{m.Mmax_case}},${{m.Mmin}},${{m.sMmin}},${{m.Mmin_case}}\\n`;
    }});
  }}

  const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${{type}}_${{name}}.csv`;
  link.click();
}}

function exportPNG(divId, filename) {{
  const el = document.getElementById(divId);
  if (!el || !el.data) {{ alert('Plot not available. Render it first.'); return; }}
  Plotly.toImage(el, {{ format: 'png', width: 1600, height: 900 }}).then(url => {{
    const link = document.createElement('a');
    link.href = url;
    link.download = filename + '_' + currentCase.replace(/[^a-zA-Z0-9]/g, '_') + '.png';
    link.click();
  }});
}}

// ============================================================
// Print Support – render all tabs, resize Plotly plots for A4 landscape
// ============================================================
let _printPrevTab = null;
const _printOrigLayout = {{}};
let _printPrepared = false;

// A4 landscape: 297×210mm, margins 10mm each → printable 277×190mm
// At 96 CSS-px/inch: 277/25.4*96 ≈ 1047px wide, 190/25.4*96 ≈ 718px tall
// Browser header/footer ≈ 40px, section title ≈ 35px → safe chart height ≈ 520px
const PRINT_W = 1047;
const PRINT_MAX_H = 520;

// Original inline heights (must match HTML inline styles)
const _plotOrigH = {{
  deformPlot: 600,
  memberForcePlot: 700,
  globalDiagramPlot: 600,
  storyDispPlot: 400,
  storyShearPlot: 350
}};

const _printPlotIds = ['deformPlot','memberForcePlot','globalDiagramPlot','storyDispPlot','storyShearPlot'];

function _relayoutForPrint() {{
  _printPlotIds.forEach(function(id) {{
    const el = document.getElementById(id);
    if (!el || !el._fullLayout) return;

    const fl = el._fullLayout;
    _printOrigLayout[id] = {{ width: fl.width, height: fl.height }};

    // Use print page width, keep original design height (clamp if too tall)
    const targetW = PRINT_W;
    const targetH = Math.min(_plotOrigH[id] || 500, PRINT_MAX_H);

    el.style.width = targetW + 'px';
    el.style.height = targetH + 'px';
    Plotly.relayout(el, {{ width: targetW, height: targetH, autosize: false }});
  }});
}}

function _restoreFromPrint() {{
  // Restore plot dimensions
  _printPlotIds.forEach(function(id) {{
    const el = document.getElementById(id);
    if (!el || !el._fullLayout) return;

    const orig = _printOrigLayout[id];
    el.style.width = '100%';
    el.style.height = _plotOrigH[id] + 'px';

    if (orig) {{
      Plotly.relayout(el, {{ width: orig.width, height: orig.height, autosize: true }});
    }} else {{
      Plotly.relayout(el, {{ autosize: true }});
    }}
    Plotly.Plots.resize(el);
  }});

  // Restore tab visibility (remove inline display override)
  document.querySelectorAll('.tab-content').forEach(function(c) {{
    c.style.display = '';
  }});

  // Restore active tab
  if (_printPrevTab) {{
    showTab(_printPrevTab);
    _printPrevTab = null;
  }}
  _printPrepared = false;
}}

// Called by the Print button – renders all, relayouts, waits, then prints
function printReport() {{
  _printPrevTab = currentTab;
  const cd = caseData[currentCase];
  if (!cd) return;

  // 1. Make ALL tabs visible so Plotly can measure container widths
  document.querySelectorAll('.tab-content').forEach(function(c) {{
    c.style.display = 'block';
  }});

  // 2. Render every tab so all plots exist in DOM
  renderDeformation(cd);
  renderSummaryCards(cd);
  populateMemberList();
  renderReactions(cd);
  renderEquilibrium();
  renderStory(cd);
  renderEnvelope();

  // 3. Relayout for print dimensions (now widths are correct)
  _relayoutForPrint();
  _printPrepared = true;

  // 4. Wait for Plotly to finish painting, then open print dialog
  requestAnimationFrame(function() {{
    setTimeout(function() {{ window.print(); }}, 250);
  }});
}}

// Guard: if user presses Ctrl+P (bypassing the button)
window.addEventListener('beforeprint', function() {{
  if (_printPrepared) return;
  _printPrevTab = currentTab;
  const cd = caseData[currentCase];
  if (!cd) return;
  document.querySelectorAll('.tab-content').forEach(function(c) {{
    c.style.display = 'block';
  }});
  renderDeformation(cd);
  renderSummaryCards(cd);
  populateMemberList();
  renderReactions(cd);
  renderEquilibrium();
  renderStory(cd);
  renderEnvelope();
  _relayoutForPrint();
}});

window.addEventListener('afterprint', function() {{
  _restoreFromPrint();
}});

// ============================================================
// Start
// ============================================================
init();
</script>
</body>
</html>"""

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.html')
        os.close(fd)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path
