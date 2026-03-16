"""실제 IFC 파일로 parse_ifc() 검증."""
import sys
import json

sys.path.insert(0, "mcp-server")
from core.ifc_parser import parse_ifc

IFC_PATH = r"C:\Users\youm\Documents\카카오톡 받은 파일\2차시도_260121.ifc"

print("=" * 60)
print("IFC Parser Test: 2차시도_260121.ifc")
print("=" * 60)

result = parse_ifc(IFC_PATH)

# 층 정보
print("\n--- Stories ---")
for s in result["stories"]:
    slab_t = s.get("slab_thickness_mm")
    slab_str = f", slab={slab_t:.0f}mm" if slab_t else ""
    print(f"  {s['name']}: height={s['height']}m, elev={s['elevation']}m{slab_str}")

# 그리드
print(f"\n--- Grid (source: {result['grid_source']}) ---")
print(f"  grid_x: {result['grid_x']} (m)")
print(f"  grid_y: {result['grid_y']} (m)")
print(f"  bays_x: {result['bays_x']} (m)")
print(f"  bays_y: {result['bays_y']} (m)")

# 벽체 정보
print(f"\n--- Walls ({len(result.get('walls', []))} on base floor) ---")
for w in result.get("walls", []):
    print(f"  Wall {w['tag']}: ({w['x_m']},{w['y_m']}m) "
          f"dir={w['orientation']} len={w['length_m']}m "
          f"t={w['thickness_mm']}mm h={w['height_m']}m")

# 슬래브
print(f"\n--- Slabs ({len(result.get('slabs', []))}) ---")
for s in result.get("slabs", []):
    print(f"  z={s['elevation_mm']:.0f}mm, t={s['thickness_mm']:.0f}mm, type={s['type']}")

# 경고
print(f"\n--- Warnings ({len(result['warnings'])}) ---")
for w in result["warnings"]:
    print(f"  - {w}")

# 기대값 검증
print("\n--- Validation ---")
expected_bays_x = [4.0, 4.0, 4.0]
expected_bays_y = [5.0, 5.0]
expected_stories = 3  # 1F, 2F, Roof

ok = True
if len(result["stories"]) != expected_stories:
    print(f"  FAIL: stories={len(result['stories'])}, expected={expected_stories}")
    ok = False
else:
    print(f"  PASS: stories count = {expected_stories}")

# bays는 tolerance 때문에 정확히 안 맞을 수 있으므로 반올림 비교
if result["bays_x"]:
    rounded_bx = [round(b, 0) for b in result["bays_x"]]
    if rounded_bx == [round(b, 0) for b in expected_bays_x]:
        print(f"  PASS: bays_x ~ {expected_bays_x}")
    else:
        print(f"  CHECK: bays_x={result['bays_x']}, expected~{expected_bays_x}")
        ok = False
else:
    print(f"  FAIL: bays_x is empty")
    ok = False

if result["bays_y"]:
    rounded_by = [round(b, 0) for b in result["bays_y"]]
    if rounded_by == [round(b, 0) for b in expected_bays_y]:
        print(f"  PASS: bays_y ~ {expected_bays_y}")
    else:
        print(f"  CHECK: bays_y={result['bays_y']}, expected~{expected_bays_y}")
        ok = False
else:
    print(f"  FAIL: bays_y is empty")
    ok = False

if ok:
    print("\n>>> ALL CHECKS PASSED <<<")
else:
    print("\n>>> SOME CHECKS NEED REVIEW <<<")
