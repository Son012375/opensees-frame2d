"""
비틀림 모드 디버깅:
  Case A: 마스터 절점에 질량 집중 (현재 방식)
  Case B: 모든 절점에 질량 분산 (Midas Gen 방식)
"""
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server'))
from core.ops_compat import ops
from core.frame_3d import _generate_frame_3d_geometry
from core.section_3d import get_section_3d, DEFAULT_SECTIONS_3D
from core.simple_beam import get_material_from_db, DEFAULT_MATERIALS


def build_and_solve(mass_mode="distributed"):
    """모델 구축 + 고유치해석"""
    stories = [5.0] + [4.0] * 10
    bays_x = [6.5, 2.5, 4.0, 2.0, 13.0]
    bays_y = [6.5, 6.5, 2.0, 6.5, 6.5]
    story_weights_kN = [4390.4] + [3998.4] * 10

    nodes, connections, node_grid, base_nodes = _generate_frame_3d_geometry(stories, bays_x, bays_y)

    col_sec = get_section_3d("H-400x400") or DEFAULT_SECTIONS_3D.get("H-300x300")
    bx_sec = get_section_3d("H-500x200") or DEFAULT_SECTIONS_3D.get("H-400x200")
    by_sec = get_section_3d("H-500x200") or DEFAULT_SECTIONS_3D.get("H-400x200")
    mat = get_material_from_db("SS275") or DEFAULT_MATERIALS.get("SS275")
    E = mat.E
    G = E / 2.6

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
        if etype == "column":
            sec, transf = col_sec, 1
        elif etype == "beam_x":
            sec, transf = bx_sec, 2
        else:
            sec, transf = by_sec, 3
        ops.element('elasticBeamColumn', i, ni, nj,
                    sec.A, E, G, sec.J, sec.Ix, sec.Iy, transf)

    n_cols_x = len(bays_x) + 1
    n_cols_y = len(bays_y) + 1
    nodes_per_floor = n_cols_x * n_cols_y
    n_stories = len(stories)
    g_acc = 9810.0

    # Rigid Diaphragm
    master_nodes = {}
    for s in range(1, n_stories + 1):
        cx_m, cy_m = n_cols_x // 2, n_cols_y // 2
        master_nid = node_grid[(s, cx_m, cy_m)]
        master_nodes[s] = master_nid
        slave_nids = [node_grid[(s, cx, cy)]
                      for cx in range(n_cols_x) for cy in range(n_cols_y)
                      if node_grid[(s, cx, cy)] != master_nid]
        ops.rigidDiaphragm(3, master_nid, *slave_nids)

    # 질량 배정
    Lx_mm = sum(bays_x) * 1000
    Ly_mm = sum(bays_y) * 1000

    floor_masses = []
    for s in range(1, n_stories + 1):
        W_N = story_weights_kN[s - 1] * 1000.0
        M_floor = W_N / g_acc

        if mass_mode == "concentrated":
            # Case A: 마스터 절점에 집중
            I_rot = M_floor * (Lx_mm**2 + Ly_mm**2) / 12.0
            ops.mass(master_nodes[s], M_floor, M_floor, 1e-6, 0.0, 0.0, I_rot)
            floor_masses.append((M_floor, I_rot))

        elif mass_mode == "distributed":
            # Case B: 모든 절점에 분산 (Midas Gen과 동일)
            m_per_node = M_floor / nodes_per_floor
            for cx in range(n_cols_x):
                for cy in range(n_cols_y):
                    nid = node_grid[(s, cx, cy)]
                    ops.mass(nid, m_per_node, m_per_node, 1e-6, 0.0, 0.0, 0.0)
            # 분산질량의 실효 회전관성 계산 (참고용)
            I_eff = 0.0
            xm = nodes[master_nodes[s]-1].x * 1000
            ym = nodes[master_nodes[s]-1].y * 1000
            for cx in range(n_cols_x):
                for cy in range(n_cols_y):
                    nid = node_grid[(s, cx, cy)]
                    nd = nodes[nid - 1]
                    dx = nd.x * 1000 - xm
                    dy = nd.y * 1000 - ym
                    I_eff += m_per_node * (dx**2 + dy**2)
            floor_masses.append((M_floor, I_eff))

    # 고유치해석
    eigenvalues = ops.eigen(15)

    # 모달 참여질량 계산 (올바른 방법)
    total_mass = sum(fm[0] for fm in floor_masses)
    total_I = sum(fm[1] for fm in floor_masses)

    results = []
    sum_pct_x, sum_pct_y, sum_pct_rz = 0, 0, 0

    for i in range(15):
        omega = math.sqrt(eigenvalues[i])
        T = 2 * math.pi / omega

        # 모드 형상 (마스터 절점만)
        phi_ux, phi_uy, phi_rz = [], [], []
        for s in range(1, n_stories + 1):
            nid = master_nodes[s]
            phi_ux.append(ops.nodeEigenvector(nid, i+1, 1))
            phi_uy.append(ops.nodeEigenvector(nid, i+1, 2))
            phi_rz.append(ops.nodeEigenvector(nid, i+1, 6))

        # 일반화 질량
        gen_mass = 0
        for s_idx in range(n_stories):
            m_t, i_r = floor_masses[s_idx]
            gen_mass += m_t * (phi_ux[s_idx]**2 + phi_uy[s_idx]**2) + i_r * phi_rz[s_idx]**2

        # 참여계수 & 유효질량
        Lx = sum(floor_masses[s][0] * phi_ux[s] for s in range(n_stories))
        Ly = sum(floor_masses[s][0] * phi_uy[s] for s in range(n_stories))
        Lrz = sum(floor_masses[s][1] * phi_rz[s] for s in range(n_stories))

        if gen_mass > 1e-30:
            meff_x = Lx**2 / gen_mass
            meff_y = Ly**2 / gen_mass
            meff_rz = Lrz**2 / gen_mass
        else:
            meff_x = meff_y = meff_rz = 0

        pct_x = meff_x / total_mass * 100 if total_mass > 0 else 0
        pct_y = meff_y / total_mass * 100 if total_mass > 0 else 0
        pct_rz = meff_rz / total_I * 100 if total_I > 0 else 0

        sum_pct_x += pct_x
        sum_pct_y += pct_y
        sum_pct_rz += pct_rz

        # 주방향 판별
        if pct_x >= pct_y and pct_x >= pct_rz:
            direction = f"TRAN-X"
        elif pct_y >= pct_rz:
            direction = f"TRAN-Y"
        else:
            direction = f"ROTN-Z"

        results.append({
            'mode': i+1, 'T': T, 'dir': direction,
            'pct_x': pct_x, 'pct_y': pct_y, 'pct_rz': pct_rz,
            'sum_x': sum_pct_x, 'sum_y': sum_pct_y, 'sum_rz': sum_pct_rz,
        })

    return results, floor_masses


# ============================================================
# 실행
# ============================================================
midas_periods = [2.4681, 1.9915, 1.9219, 0.8133, 0.6515,
                 0.6215, 0.4742, 0.3754, 0.3506, 0.3316,
                 0.2595, 0.2526, 0.2395, 0.2036, 0.1949]
midas_dirs = [
    "TRAN-Y", "ROTN-Z", "TRAN-X", "TRAN-Y", "ROTN-Z",
    "TRAN-X", "TRAN-Y", "ROTN-Z", "TRAN-X", "TRAN-Y",
    "ROTN-Z", "TRAN-Y", "TRAN-X", "TRAN-Y", "ROTN-Z"
]
midas_pcts = [
    (0, 77.0, 8.3), (0, 8.5, 75.6), (82.6, 0, 0), (0, 8.7, 1.0), (0, 0.9, 9.2),
    (10.9, 0, 0), (0, 2.6, 0.3), (0, 0.3, 2.9), (3.5, 0, 0), (0, 1.0, 0.1),
    (0, 0.1, 1.2), (0, 0.4, 0.1), (1.6, 0, 0), (0, 0.2, 0.0), (0, 0.0, 0.7)
]

for case_name, mass_mode in [("Case A: 마스터 집중질량", "concentrated"),
                               ("Case B: 분산질량 (Midas방식)", "distributed")]:
    print(f"\n{'='*100}")
    print(f"  {case_name}")
    print(f"{'='*100}")

    results, floor_masses = build_and_solve(mass_mode)

    print(f"\n  1F 회전관성: {floor_masses[0][1]:.3e} N·mm·s²")
    print(f"  2F 회전관성: {floor_masses[1][1]:.3e} N·mm·s²")

    print(f"\n{'Mode':>4}  {'OPS T(s)':>9}  {'방향':>8}  {'TX(%)':>7}  {'TY(%)':>7}  {'RZ(%)':>7}  {'ΣTX':>7}  {'ΣTY':>7}  {'ΣRZ':>7}  │  {'Midas T':>8}  {'방향':>8}  {'차이':>7}")
    print("-" * 110)

    for r in results:
        i = r['mode'] - 1
        diff = (r['T'] - midas_periods[i]) / midas_periods[i] * 100
        print(f"{r['mode']:>4}  {r['T']:>9.4f}  {r['dir']:>8}  {r['pct_x']:>7.2f}  {r['pct_y']:>7.2f}  {r['pct_rz']:>7.2f}  {r['sum_x']:>7.1f}  {r['sum_y']:>7.1f}  {r['sum_rz']:>7.1f}  │  {midas_periods[i]:>8.4f}  {midas_dirs[i]:>8}  {diff:>+7.2f}%")
