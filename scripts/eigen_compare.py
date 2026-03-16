"""
OpenSees 고유치해석 - Midas Gen 비교 검증
4차시도 건물 모델 (11층, 5x5 bay, 28m x 28m)
- Rigid Diaphragm 적용
"""
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server'))
from core.ops_compat import ops
from core.frame_3d import _generate_frame_3d_geometry
from core.section_3d import get_section_3d, DEFAULT_SECTIONS_3D
from core.simple_beam import get_material_from_db, DEFAULT_MATERIALS

# ============================================================
# 건물 파라미터
# ============================================================
stories = [5.0] + [4.0] * 10
bays_x = [6.5, 2.5, 4.0, 2.0, 13.0]
bays_y = [6.5, 6.5, 2.0, 6.5, 6.5]

column_section = "H-400x400"
beam_x_section = "H-500x200"
beam_y_section = "H-500x200"
material_name = "SS275"

story_weights_kN = [4390.4] + [3998.4] * 10

# ============================================================
# 모델 구축
# ============================================================
nodes, connections, node_grid, base_nodes = _generate_frame_3d_geometry(stories, bays_x, bays_y)

col_sec = get_section_3d(column_section) or DEFAULT_SECTIONS_3D.get("H-300x300")
bx_sec = get_section_3d(beam_x_section) or DEFAULT_SECTIONS_3D.get("H-400x200")
by_sec = get_section_3d(beam_y_section) or DEFAULT_SECTIONS_3D.get("H-400x200")
mat = get_material_from_db(material_name) or DEFAULT_MATERIALS.get("SS275")

E = mat.E  # MPa
G = E / (2.0 * (1.0 + 0.3))

ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

for node in nodes:
    ops.node(node.id, node.x * 1000, node.y * 1000, node.z * 1000)

for bn in base_nodes:
    ops.fix(bn, 1, 1, 1, 1, 1, 1)

ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)  # column
ops.geomTransf('Linear', 2, 0.0, 0.0, 1.0)  # beam_x
ops.geomTransf('Linear', 3, 0.0, 0.0, 1.0)  # beam_y

elem_id = 1
for ni, nj, etype in connections:
    if etype == "column":
        sec, transf = col_sec, 1
    elif etype == "beam_x":
        sec, transf = bx_sec, 2
    else:
        sec, transf = by_sec, 3
    os_Iy = sec.Ix
    os_Iz = sec.Iy
    ops.element('elasticBeamColumn', elem_id, ni, nj,
                sec.A, E, G, sec.J, os_Iy, os_Iz, transf)
    elem_id += 1

n_cols_x = len(bays_x) + 1  # 6
n_cols_y = len(bays_y) + 1  # 6
print(f"모델: {len(nodes)} nodes, {elem_id - 1} elements")

# ============================================================
# Rigid Diaphragm 적용
# ============================================================
# 각 층의 중앙 절점을 마스터로, 나머지를 슬레이브로 설정
# rigidDiaphragm(perpDirn, masterNode, *slaveNodes)
# perpDirn = 3 (Z축 수직 = 수평면 다이어프램)

master_nodes = {}
for s in range(1, len(stories) + 1):
    # 중앙 그리드 (cx=3, cy=3 근처) 를 마스터로
    cx_m = n_cols_x // 2  # 3
    cy_m = n_cols_y // 2  # 3
    master_nid = node_grid[(s, cx_m, cy_m)]
    master_nodes[s] = master_nid

    slave_nids = []
    for cx in range(n_cols_x):
        for cy in range(n_cols_y):
            nid = node_grid[(s, cx, cy)]
            if nid != master_nid:
                slave_nids.append(nid)

    # rigidDiaphragm: 수평면 (perpDirn=3 = Z축 수직)
    ops.rigidDiaphragm(3, master_nid, *slave_nids)

print(f"Rigid Diaphragm: {len(master_nodes)} floors")

# ============================================================
# 질량 배정
# ============================================================
g_acc = 9810.0  # mm/s²

# X,Y 좌표 (m)
x_coords = [0.0]
for bx in bays_x:
    x_coords.append(x_coords[-1] + bx)
y_coords = [0.0]
for by in bays_y:
    y_coords.append(y_coords[-1] + by)

Lx_mm = sum(bays_x) * 1000  # 28000 mm
Ly_mm = sum(bays_y) * 1000  # 28000 mm

nodes_per_floor = n_cols_x * n_cols_y  # 36

for s in range(1, len(stories) + 1):
    W_N = story_weights_kN[s - 1] * 1000.0
    M_floor = W_N / g_acc  # 총 층 질량

    # 마스터 절점에 총 층 질량과 회전관성 배정
    master_nid = master_nodes[s]
    # 회전관성: I = M * (Lx² + Ly²) / 12 (직사각형 평면)
    I_rot = M_floor * (Lx_mm**2 + Ly_mm**2) / 12.0

    ops.mass(master_nid, M_floor, M_floor, 1.0e-6, 0.0, 0.0, I_rot)

# ============================================================
# 고유치해석
# ============================================================
num_modes = 15

try:
    eigenvalues = ops.eigen(num_modes)
except Exception as e:
    print(f"기본 eigen 실패: {e}")
    try:
        eigenvalues = ops.eigen('-genBandArpack', num_modes)
    except Exception as e2:
        print(f"genBandArpack도 실패: {e2}")
        eigenvalues = ops.eigen('-fullGenLapack', num_modes)

# ============================================================
# 결과 출력
# ============================================================
midas_periods = [2.4681, 1.9915, 1.9219, 0.8133, 0.6515,
                 0.6215, 0.4742, 0.3754, 0.3506, 0.3316,
                 0.2595, 0.2526, 0.2395, 0.2036, 0.1949]

midas_directions = [
    "TRAN-Y  77.0%",
    "ROTN-Z  75.6%",
    "TRAN-X  82.6%",
    "TRAN-Y   8.7%",
    "ROTN-Z   9.2%",
    "TRAN-X  10.9%",
    "TRAN-Y   2.6%",
    "ROTN-Z   2.9%",
    "TRAN-X   3.5%",
    "TRAN-Y   1.0%",
    "ROTN-Z   1.2%",
    "TRAN-Y   0.4%",
    "TRAN-X   1.6%",
    "TRAN-Y   0.2%",
    "ROTN-Z   0.7%",
]

def get_mode_direction(mode_num):
    """마스터 절점 기반 모드 방향 판별"""
    sum_ux2 = 0.0
    sum_uy2 = 0.0
    sum_rz2 = 0.0
    for s in range(1, len(stories) + 1):
        nid = master_nodes[s]
        ux = ops.nodeEigenvector(nid, mode_num, 1)
        uy = ops.nodeEigenvector(nid, mode_num, 2)
        rz = ops.nodeEigenvector(nid, mode_num, 6)
        sum_ux2 += ux**2
        sum_uy2 += uy**2
        sum_rz2 += rz**2

    total = sum_ux2 + sum_uy2 + sum_rz2
    if total < 1e-30:
        return "N/A"
    px = sum_ux2 / total * 100
    py = sum_uy2 / total * 100
    prz = sum_rz2 / total * 100
    if px >= py and px >= prz:
        return f"TRAN-X {px:5.1f}%"
    elif py >= prz:
        return f"TRAN-Y {py:5.1f}%"
    else:
        return f"ROTN-Z {prz:5.1f}%"


print("\n" + "=" * 95)
print("  OpenSees (Rigid Diaphragm) vs Midas Gen  고유치해석 비교")
print("=" * 95)
print(f"\n{'Mode':>4}  {'OPS T(s)':>10}  {'Midas T(s)':>10}  {'차이(%)':>8}  {'OPS 방향':>18}  {'Midas 방향':>16}")
print("-" * 95)

for i in range(num_modes):
    if eigenvalues[i] <= 0:
        print(f"{i+1:>4}  {'N/A':>10}  {midas_periods[i]:>10.4f}")
        continue
    omega = math.sqrt(eigenvalues[i])
    T_ops = 2.0 * math.pi / omega
    T_midas = midas_periods[i]
    diff = (T_ops - T_midas) / T_midas * 100
    direction = get_mode_direction(i + 1)
    print(f"{i+1:>4}  {T_ops:>10.4f}  {T_midas:>10.4f}  {diff:>+8.2f}%  {direction:>18}  {midas_directions[i]:>16}")

print("-" * 95)

# 모드 형상 (마스터 절점)
print("\n[모드 형상 - 마스터 절점 변위]")
print(f"{'Story':>5}  {'H(m)':>6}  {'M1-UX':>10}  {'M1-UY':>10}  {'M1-RZ':>10}  {'M2-UX':>10}  {'M2-UY':>10}  {'M2-RZ':>10}  {'M3-UX':>10}  {'M3-UY':>10}  {'M3-RZ':>10}")
print("-" * 115)

for s in range(1, len(stories) + 1):
    z_m = sum(stories[:s])
    nid = master_nodes[s]
    vals = []
    for mode in range(1, 4):
        ux = ops.nodeEigenvector(nid, mode, 1)
        uy = ops.nodeEigenvector(nid, mode, 2)
        rz = ops.nodeEigenvector(nid, mode, 6)
        vals.extend([ux, uy, rz])
    print(f"{s:>5}  {z_m:>6.1f}  {vals[0]:>10.6f}  {vals[1]:>10.6f}  {vals[2]:>10.4e}  {vals[3]:>10.6f}  {vals[4]:>10.6f}  {vals[5]:>10.4e}  {vals[6]:>10.6f}  {vals[7]:>10.6f}  {vals[8]:>10.4e}")

# 질량 검증
print(f"\n[질량 검증]")
total_mass = sum(w * 1000.0 / g_acc for w in story_weights_kN)
print(f"  총 질량: {total_mass:.1f} N*s2/mm  (= {total_mass * g_acc / 1000:.1f} kN)")
print(f"  회전관성(층당): I = M_floor × (28000² + 28000²) / 12")
print(f"  1F 회전관성: {story_weights_kN[0] * 1000 / g_acc * (28000**2 + 28000**2) / 12:.2e} N*mm*s²")
