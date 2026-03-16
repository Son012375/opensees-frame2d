"""
Rigid Diaphragm 비교: OpenSees vs Midas Gen
test1_3x3bay_2story 건물 모델
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server'))

from core.ops_compat import ops
from core.frame_3d import analyze_frame_3d_multi, _generate_frame_3d_geometry
from core.section_3d import get_section_3d, DEFAULT_SECTIONS_3D
from core.simple_beam import get_material_from_db, DEFAULT_MATERIALS

# ============================================================
# 건물 파라미터
# ============================================================
stories = [4.0, 4.0]
bays_x = [4.5, 6.5, 6.0]  # total 17m (from IFC)
bays_y = [4.0, 6.0, 5.0]  # total 15m (from IFC)
column_section = 'H-400x400'    # IFC: H400X197 = H-400x400x13x21
beam_x_section = 'H-250x250'    # IFC: H250X72.4 = H-250x250x9x14
beam_y_section = 'H-250x250'
material_name = 'SS275'

# Midas Gen 결과 (test1_3x3bay_2story.xlsx)
midas_periods = [0.29, 0.2181, 0.211, 0.0887, 0.0698, 0.0643]
midas_dirs = ['TRAN-Y', 'ROTN-Z', 'TRAN-X', 'TRAN-Y', 'ROTN-Z', 'TRAN-X']
midas_pcts = [
    (0.0, 96.66, 0.19),
    (14.36, 0.17, 79.01),
    (78.64, 0.02, 14.31),
    (0.0, 3.14, 0.003),
    (0.06, 0.006, 6.39),
    (6.93, 0.0, 0.09),
]

midas_drifts = {
    'EX': {'1F': 0.0025, '2F': 0.0039},
    'WX': {'1F': 0.0010, '2F': 0.0014},
}

midas_reactions = {
    'DL': 1729.18,
    'LL': 637.50,
    'EX': -306.544,
    'WX': -130.544,
}

# Midas 층질량 (kN/g)
midas_mass_2F = 154.48
midas_mass_Roof = 15.56

# ============================================================
# 하중 구성 (Midas와 동일 조건)
# ============================================================
floor_area = sum(bays_x) * sum(bays_y)
member_selfweight = 428.7  # kN (기둥+보 자중 추정)
DL_intensity = (midas_reactions['DL'] - member_selfweight) / floor_area
LL_intensity = midas_reactions['LL'] / floor_area

# EQX 분배
W_2F = midas_mass_2F * 9.81
W_Roof = midas_mass_Roof * 9.81
Cv_2F = (W_2F * 4) / (W_2F * 4 + W_Roof * 8)
EQX_total = abs(midas_reactions['EX'])
F_2F_eq = EQX_total * Cv_2F
F_Roof_eq = EQX_total * (1 - Cv_2F)

# WX 분배 (qz 비례)
WX_total = abs(midas_reactions['WX'])
F_2F_wx = WX_total * 0.55
F_Roof_wx = WX_total * 0.45

load_cases = {
    'DL': [{'type': 'floor_area', 'story': 1, 'value': DL_intensity}],
    'LL': [{'type': 'floor_area', 'story': 1, 'value': LL_intensity}],
    'EQX': [
        {'type': 'lateral_x', 'story': 1, 'value': F_2F_eq},
        {'type': 'lateral_x', 'story': 2, 'value': F_Roof_eq},
    ],
    'WX': [
        {'type': 'lateral_x', 'story': 1, 'value': F_2F_wx},
        {'type': 'lateral_x', 'story': 2, 'value': F_Roof_wx},
    ],
}

print(f"건물: {len(stories)}층, {len(bays_x)}x{len(bays_y)} bay ({sum(bays_x):.0f}m x {sum(bays_y):.0f}m)")
print(f"바닥면적: {floor_area:.0f} m2, DL={DL_intensity:.2f} kN/m2, LL={LL_intensity:.2f} kN/m2")
print(f"EQX: 1F={F_2F_eq:.1f}kN, 2F={F_Roof_eq:.1f}kN (합={EQX_total:.1f}kN)")
print()

# ============================================================
# 1. 정적해석 비교: OFF vs ON
# ============================================================
print("=" * 80)
print("  정적해석 비교: rigid_diaphragm OFF vs ON vs Midas")
print("=" * 80)

static_results = {}
for rd in [False, True]:
    result = analyze_frame_3d_multi(
        stories=stories, bays_x=bays_x, bays_y=bays_y,
        load_cases=load_cases, supports='fixed',
        column_section=column_section,
        beam_x_section=beam_x_section,
        beam_y_section=beam_y_section,
        material_name=material_name,
        num_elements_per_member=1,
        rigid_diaphragm=rd,
    )
    label = 'ON' if rd else 'OFF'
    static_results[label] = result

    print(f"\n--- rigid_diaphragm = {label} ---")
    for cn, cr in result.case_results.items():
        rz = sum(r['RZ_kN'] for r in cr.reactions)
        rx = sum(r['RX_kN'] for r in cr.reactions)
        drifts = sorted(cr.story_drifts, key=lambda x: x['story'])

        if cn in ['EQX', 'WX']:
            midas_key = cn.replace('EQX', 'EX')
            midas_val = midas_reactions.get(midas_key, 0)
            diff = (abs(rx) - abs(midas_val)) / abs(midas_val) * 100 if midas_val else 0
            drift_str = ', '.join([
                f"{sd['story']}F={sd['drift_x'] * sd['height_m'] * 1000:.2f}mm"
                for sd in drifts
            ])
            print(f"  {cn}: FX={rx:.1f}kN (Midas={midas_val:.1f}, diff={diff:+.1f}%), drifts=[{drift_str}]")
        else:
            midas_val = midas_reactions.get(cn, 0)
            diff = (rz - midas_val) / midas_val * 100 if midas_val else 0
            print(f"  {cn}: FZ={rz:.1f}kN (Midas={midas_val:.1f}, diff={diff:+.1f}%)")

# 층간변위 비교 상세
print("\n")
print("=" * 80)
print("  EQX 층간변위 상세 비교")
print("=" * 80)
print(f"  {'층':>4}  {'OFF(mm)':>8}  {'ON(mm)':>8}  {'Midas(mm)':>10}  {'OFF-Midas':>10}  {'ON-Midas':>10}")
print("  " + "-" * 60)
for case in ['EQX', 'WX']:
    midas_key = case.replace('EQX', 'EX')
    if midas_key not in midas_drifts:
        continue
    print(f"\n  [{case}]")
    for label in ['OFF', 'ON']:
        cr = static_results[label].case_results[case]

    drifts_off = sorted(static_results['OFF'].case_results[case].story_drifts, key=lambda x: x['story'])
    drifts_on = sorted(static_results['ON'].case_results[case].story_drifts, key=lambda x: x['story'])

    for sd_off, sd_on in zip(drifts_off, drifts_on):
        s = sd_off['story']
        h_mm = sd_off['height_m'] * 1000
        off_mm = sd_off['drift_x'] * h_mm
        on_mm = sd_on['drift_x'] * h_mm
        floor_key = f"{s}F"
        midas_mm = midas_drifts[midas_key][floor_key] * 1000
        diff_off = (off_mm - midas_mm) / midas_mm * 100 if midas_mm > 0 else 0
        diff_on = (on_mm - midas_mm) / midas_mm * 100 if midas_mm > 0 else 0
        print(f"  {floor_key:>4}  {off_mm:>8.2f}  {on_mm:>8.2f}  {midas_mm:>10.1f}  {diff_off:>+9.1f}%  {diff_on:>+9.1f}%")

# ============================================================
# 2. 고유치해석: Rigid Diaphragm
# ============================================================
print("\n")
print("=" * 80)
print("  고유치해석 비교 (Rigid Diaphragm + Midas 동일 질량)")
print("=" * 80)

nodes, connections, node_grid, base_nodes = _generate_frame_3d_geometry(stories, bays_x, bays_y)

col_sec = get_section_3d(column_section) or DEFAULT_SECTIONS_3D.get(column_section)
bx_sec = get_section_3d(beam_x_section) or DEFAULT_SECTIONS_3D.get(beam_x_section)
by_sec = get_section_3d(beam_y_section) or DEFAULT_SECTIONS_3D.get(beam_y_section)
mat_obj = get_material_from_db(material_name) or DEFAULT_MATERIALS.get('SS275')
E = mat_obj.E
G = E / (2.0 * (1.0 + 0.3))

n_cols_x = len(bays_x) + 1
n_cols_y = len(bays_y) + 1
n_stories = len(stories)
g_acc = 9810.0

story_weights_kN = [midas_mass_2F * 9.81, midas_mass_Roof * 9.81]

ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

for node in nodes:
    ops.node(node.id, node.x * 1000, node.y * 1000, node.z * 1000)
for bn in base_nodes:
    ops.fix(bn, 1, 1, 1, 1, 1, 1)

ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)
ops.geomTransf('Linear', 2, 0.0, 0.0, 1.0)
ops.geomTransf('Linear', 3, 0.0, 0.0, 1.0)

for i, (ni, nj, etype) in enumerate(connections, 1):
    if etype == 'column':
        sec, transf = col_sec, 1
    elif etype == 'beam_x':
        sec, transf = bx_sec, 2
    else:
        sec, transf = by_sec, 3
    ops.element('elasticBeamColumn', i, ni, nj,
                sec.A, E, G, sec.J, sec.Ix, sec.Iy, transf)

# Rigid Diaphragm + 질량 (트리뷰터리 면적 기반 분산)
master_nodes = {}
floor_masses = []

# 트리뷰터리 면적 계산
x_coords = [0.0]
for bx in bays_x:
    x_coords.append(x_coords[-1] + bx)
y_coords = [0.0]
for by in bays_y:
    y_coords.append(y_coords[-1] + by)

trib_areas = {}
for cx in range(n_cols_x):
    dx_left = (x_coords[cx] - x_coords[cx-1]) / 2.0 if cx > 0 else 0
    dx_right = (x_coords[cx+1] - x_coords[cx]) / 2.0 if cx < n_cols_x - 1 else 0
    for cy in range(n_cols_y):
        dy_left = (y_coords[cy] - y_coords[cy-1]) / 2.0 if cy > 0 else 0
        dy_right = (y_coords[cy+1] - y_coords[cy]) / 2.0 if cy < n_cols_y - 1 else 0
        trib_areas[(cx, cy)] = (dx_left + dx_right) * (dy_left + dy_right)
total_trib = sum(trib_areas.values())

for s in range(1, n_stories + 1):
    cx_m = n_cols_x // 2
    cy_m = n_cols_y // 2
    master_nid = node_grid[(s, cx_m, cy_m)]
    master_nodes[s] = master_nid

    slave_nids = [node_grid[(s, cx, cy)]
                  for cx in range(n_cols_x) for cy in range(n_cols_y)
                  if node_grid[(s, cx, cy)] != master_nid]
    ops.rigidDiaphragm(3, master_nid, *slave_nids)

    W_N = story_weights_kN[s - 1] * 1000.0
    M_floor = W_N / g_acc

    # 트리뷰터리 면적 기반으로 각 절점에 질량 분배
    # Rigid Diaphragm이 자동으로 정확한 회전관성 계산
    for cx in range(n_cols_x):
        for cy in range(n_cols_y):
            nid = node_grid[(s, cx, cy)]
            frac = trib_areas[(cx, cy)] / total_trib
            m_node = M_floor * frac
            ops.mass(nid, m_node, m_node, 1.0e-6, 0.0, 0.0, 0.0)

    floor_masses.append(M_floor)

# 고유치해석
eigenvalues = ops.eigen(6)

print(f"\n{'Mode':>4}  {'OPS T(s)':>10}  {'Midas T(s)':>10}  {'차이(%)':>8}  {'OPS 방향':>14}  {'Midas 방향':>14}  {'일치':>4}")
print("-" * 80)

for i in range(6):
    omega = math.sqrt(eigenvalues[i])
    T = 2 * math.pi / omega
    T_midas = midas_periods[i]
    diff = (T - T_midas) / T_midas * 100

    # 마스터 절점 모드 형상으로 방향 판별
    sum_ux2, sum_uy2, sum_rz2 = 0.0, 0.0, 0.0
    for s in range(1, n_stories + 1):
        nid = master_nodes[s]
        ux = ops.nodeEigenvector(nid, i + 1, 1)
        uy = ops.nodeEigenvector(nid, i + 1, 2)
        rz = ops.nodeEigenvector(nid, i + 1, 6)
        sum_ux2 += ux**2
        sum_uy2 += uy**2
        sum_rz2 += rz**2

    total = sum_ux2 + sum_uy2 + sum_rz2
    if total > 1e-30:
        if sum_ux2 >= sum_uy2 and sum_ux2 >= sum_rz2:
            direction = 'TRAN-X'
        elif sum_uy2 >= sum_rz2:
            direction = 'TRAN-Y'
        else:
            direction = 'ROTN-Z'
    else:
        direction = 'N/A'

    dir_match = 'O' if direction == midas_dirs[i] else 'X'
    print(f"{i+1:>4}  {T:>10.4f}  {T_midas:>10.4f}  {diff:>+8.2f}%  {direction:>14}  {midas_dirs[i]:>14}  {dir_match:>4}")

print("\nDone!")
