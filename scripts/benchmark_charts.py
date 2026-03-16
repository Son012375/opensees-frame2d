"""Midas Gen 벤치마크 비교 차트 생성 (발표용)"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = Path(__file__).parent.parent / "docs" / "benchmark_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def chart1_displacement_comparison():
    """Chart 1: 케이스별 변위 비교 (OpenSees vs Midas)"""
    fig, ax = plt.subplots(figsize=(12, 6))

    cases = ['Case 1\n2D 단순보', 'Case 2\n2D 1층 포탈', 'Case 3\n2D 3층',
             'Case 4\n3D 2층', 'Case 5 Linear\n3D 5층', 'Case 5 PDelta\n3D 5층']
    opensees = [5.425, 1.388, 18.444, 5.377, 33.648, 33.937]
    midas    = [5.425, 1.388, 18.444, 5.201, 33.648, 33.946]

    x = np.arange(len(cases))
    w = 0.35

    bars1 = ax.bar(x - w/2, opensees, w, label='OpenSees', color='#2196F3', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, midas, w, label='Midas Gen', color='#FF9800', edgecolor='white', linewidth=0.5)

    # 차이 표시
    diffs = [abs(o - m) / max(abs(o), abs(m)) * 100 if max(abs(o), abs(m)) > 0 else 0
             for o, m in zip(opensees, midas)]
    for i, (d, o, m) in enumerate(zip(diffs, opensees, midas)):
        y_pos = max(o, m) + max(opensees) * 0.03
        color = '#4CAF50' if d < 1.0 else '#FF5722'
        label = f'{d:.2f}%' if d > 0.001 else 'MATCH'
        ax.text(i, y_pos, label, ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=color)

    ax.set_ylabel('최대 변위 (mm)', fontsize=13)
    ax.set_title('OpenSees vs Midas Gen — 케이스별 최대 변위 비교', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=10)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, max(opensees) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'chart1_displacement.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUT_DIR / 'chart1_displacement.png'}")


def chart2_diff_percentage():
    """Chart 2: 전체 메트릭 차이율 분포"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: 케이스별 최대 차이 ──
    cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5']
    max_diffs = [0.00, 0.02, 0.04, 3.90, 0.05]
    colors = ['#4CAF50' if d <= 1.0 else '#FF9800' for d in max_diffs]

    bars = ax1.barh(cases, max_diffs, color=colors, edgecolor='white', height=0.6)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='허용 오차 1%')

    for bar, d in zip(bars, max_diffs):
        x_pos = bar.get_width() + 0.1
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{d:.2f}%', va='center', fontsize=11, fontweight='bold')

    ax1.set_xlabel('최대 차이 (%)', fontsize=12)
    ax1.set_title('케이스별 최대 차이율', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 5.0)
    ax1.legend(fontsize=11)
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.3)

    # ── Right: 전체 메트릭 OK/CHECK 비율 ──
    labels = ['OK\n(< 1%)', 'CHECK\n(≥ 1%)']
    sizes = [100, 12]
    colors_pie = ['#4CAF50', '#FF9800']
    explode = (0.03, 0.08)

    wedges, texts, autotexts = ax2.pie(
        sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct=lambda pct: f'{int(round(pct/100*112))}건\n({pct:.1f}%)',
        startangle=90, textprops={'fontsize': 12},
        pctdistance=0.55,
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(11)

    ax2.set_title(f'전체 112개 메트릭 판정 분포', fontsize=14, fontweight='bold')

    plt.tight_layout(w_pad=3)
    plt.savefig(OUT_DIR / 'chart2_diff_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUT_DIR / 'chart2_diff_distribution.png'}")


def chart3_moment_comparison():
    """Chart 3: 기둥 기초 모멘트 비교"""
    fig, ax = plt.subplots(figsize=(10, 6))

    cases = ['Case 2\n1층 포탈', 'Case 3\n3층', 'Case 4\n3D 2층',
             'Case 5 Linear\n3D 5층', 'Case 5 PDelta\n3D 5층']
    opensees = [51.105, 213.08, 92.094, 296.13, 298.34]
    midas    = [51.11,  213.0,  93.56,  296.13, 298.34]

    x = np.arange(len(cases))
    w = 0.35

    ax.bar(x - w/2, opensees, w, label='OpenSees', color='#2196F3', edgecolor='white')
    ax.bar(x + w/2, midas, w, label='Midas Gen', color='#FF9800', edgecolor='white')

    diffs = [abs(o - m) / max(abs(o), abs(m)) * 100 for o, m in zip(opensees, midas)]
    for i, (d, o, m) in enumerate(zip(diffs, opensees, midas)):
        y_pos = max(o, m) + max(opensees) * 0.03
        color = '#4CAF50' if d < 1.0 else '#FF5722'
        label = f'{d:.2f}%' if d > 0.005 else 'MATCH'
        ax.text(i, y_pos, label, ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    ax.set_ylabel('기둥 기초 모멘트 (kN·m)', fontsize=13)
    ax.set_title('OpenSees vs Midas Gen — 기둥 기초 모멘트 비교', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(opensees) * 1.20)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'chart3_moment.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUT_DIR / 'chart3_moment.png'}")


def chart4_pdelta_amplification():
    """Chart 4: Case 5 P-Delta 층별 변위 증폭비"""
    fig, ax = plt.subplots(figsize=(8, 6))

    stories = ['1F', '2F', '3F', '4F', '5F']
    opensees_amp = [1.00797, 1.00900, 1.00908, 1.00849, 1.00788]
    midas_amp    = [1.00848, 1.00908, 1.00947, 1.00884, 1.00827]

    x = np.arange(len(stories))
    w = 0.3

    ax.bar(x - w/2, opensees_amp, w, label='OpenSees (Corotational)', color='#2196F3', edgecolor='white')
    ax.bar(x + w/2, midas_amp, w, label='Midas Gen (P-Delta)', color='#FF9800', edgecolor='white')

    # 차이 표시
    for i, (o, m) in enumerate(zip(opensees_amp, midas_amp)):
        d = abs(o - m) / m * 100
        ax.text(i, max(o, m) + 0.0003, f'{d:.3f}%', ha='center', fontsize=9,
                fontweight='bold', color='#4CAF50')

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('변위 증폭비 (PDelta / Linear)', fontsize=12)
    ax.set_title('Case 5 — 층별 P-Delta 변위 증폭비 비교', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(stories, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0.999, 1.012)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'chart4_pdelta_amp.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUT_DIR / 'chart4_pdelta_amp.png'}")


def chart5_case4_3d_detail():
    """Chart 5: Case 4 (3D) 상세 비교 — 평형력 일치 vs 변위 차이"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: 평형력 (정확 일치) ──
    metrics = ['Base X\n(kN)', 'Base Z\n(kN)']
    ops_vals = [120.0, 560.0]
    mid_vals = [120.0, 560.0]

    x = np.arange(len(metrics))
    w = 0.3
    ax1.bar(x - w/2, ops_vals, w, label='OpenSees', color='#2196F3', edgecolor='white')
    ax1.bar(x + w/2, mid_vals, w, label='Midas Gen', color='#FF9800', edgecolor='white')

    for i in range(len(metrics)):
        ax1.text(i, max(ops_vals[i], mid_vals[i]) + 15, 'MATCH\n0.00%',
                ha='center', fontsize=11, fontweight='bold', color='#4CAF50')

    ax1.set_title('평형력 — 정확 일치', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 700)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Right: 변위/모멘트 (~3% 차이) ──
    metrics2 = ['Roof dx\n(mm)', '1F dx\n(mm)', 'Col My\n(kN·m)', 'Beam My\n(kN·m)']
    ops2 = [5.377, 1.998, 92.09, 29.18]
    mid2 = [5.201, 1.921, 93.56, 28.60]
    diffs2 = [abs(o-m)/max(o,m)*100 for o,m in zip(ops2, mid2)]

    x2 = np.arange(len(metrics2))
    ax2.bar(x2 - w/2, ops2, w, label='OpenSees', color='#2196F3', edgecolor='white')
    ax2.bar(x2 + w/2, mid2, w, label='Midas Gen', color='#FF9800', edgecolor='white')

    for i, d in enumerate(diffs2):
        y_pos = max(ops2[i], mid2[i]) + max(ops2) * 0.04
        ax2.text(i, y_pos, f'{d:.1f}%', ha='center', fontsize=10,
                fontweight='bold', color='#FF9800')

    ax2.set_title('변위/모멘트 — ~3% 차이 (요소 정식화)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics2, fontsize=10)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, max(ops2) * 1.25)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Case 4 (3D 2층) — 평형력 일치, 변위 ~3% 차이 분석',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'chart5_case4_detail.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUT_DIR / 'chart5_case4_detail.png'}")


def chart6_summary_heatmap():
    """Chart 6: 전체 종합 요약 — 케이스별 메트릭 현황"""
    fig, ax = plt.subplots(figsize=(10, 5))

    cases = ['Case 1\n2D 단순보', 'Case 2\n2D 1층 포탈', 'Case 3\n2D 3층',
             'Case 4\n3D 2층', 'Case 5\n3D 5층 PDelta']
    ok_counts   = [5, 16, 18, 21, 40]
    check_counts = [0, 0, 0, 12, 0]
    total = [o+c for o, c in zip(ok_counts, check_counts)]
    max_diffs = [0.00, 0.02, 0.04, 3.90, 0.05]

    x = np.arange(len(cases))
    w = 0.5

    bars_ok = ax.bar(x, ok_counts, w, label='OK (< 1%)', color='#4CAF50', edgecolor='white')
    bars_check = ax.bar(x, check_counts, w, bottom=ok_counts, label='CHECK (≥ 1%)',
                        color='#FF9800', edgecolor='white')

    # 총 메트릭 수 + 최대 차이 표시
    for i, (t, d) in enumerate(zip(total, max_diffs)):
        verdict = 'MATCH' if d < 1.0 else f'AGREE\n~{d:.0f}%'
        v_color = '#2E7D32' if d < 1.0 else '#E65100'
        ax.text(i, t + 1.5, f'{t}건\n{verdict}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=v_color)

    ax.set_ylabel('비교 메트릭 수', fontsize=13)
    ax.set_title('벤치마크 종합 — 112개 메트릭, 100 OK + 12 CHECK',
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=10)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, max(total) * 1.35)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'chart6_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUT_DIR / 'chart6_summary.png'}")


if __name__ == '__main__':
    print("벤치마크 차트 생성 중...\n")
    chart1_displacement_comparison()
    chart2_diff_percentage()
    chart3_moment_comparison()
    chart4_pdelta_amplification()
    chart5_case4_3d_detail()
    chart6_summary_heatmap()
    print(f"\n완료! 총 6개 차트 → {OUT_DIR}")
