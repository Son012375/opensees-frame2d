"""IFC 파일 탐색 스크립트."""
import re
import ifcopenshell
import ifcopenshell.util.placement as ifc_p

IFC_PATH = r"C:\Users\youm\Documents\카카오톡 받은 파일\2차시도_260121.ifc"

ifc = ifcopenshell.open(IFC_PATH)

# 1. 층 정보 (ifcopenshell 자체 디코딩 확인)
print("=== IfcBuildingStorey ===")
storeys = ifc.by_type("IfcBuildingStorey")
for s in sorted(storeys, key=lambda x: x.Elevation if x.Elevation else 0):
    name_bytes = s.Name.encode("utf-8", errors="replace") if s.Name else b""
    print(f"  Name: {s.Name!r}, Elevation: {s.Elevation:.1f} mm")

# 2. 벽 위치 분석 → 그리드 추론
print("\n=== Wall positions (all) ===")
walls = ifc.by_type("IfcWallStandardCase")
wall_data = []
for w in walls:
    matrix = ifc_p.get_local_placement(w.ObjectPlacement)
    x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]

    # profile
    profile_x, profile_y, depth = None, None, None
    if w.Representation:
        for rep in w.Representation.Representations:
            for item in rep.Items:
                if item.is_a("IfcExtrudedAreaSolid"):
                    depth = item.Depth
                    profile = item.SweptArea
                    if profile.is_a("IfcRectangleProfileDef"):
                        profile_x = profile.XDim
                        profile_y = profile.YDim

    wall_data.append({
        "tag": w.Tag, "x": x, "y": y, "z": z,
        "profile_x": profile_x, "profile_y": profile_y, "depth": depth,
        "name": w.Name
    })
    print(f"  Wall {w.Tag}: pos=({x:.0f},{y:.0f},{z:.0f}) "
          f"profile=({profile_x:.0f}x{profile_y:.0f}) height={depth:.0f}")

# 3. 벽 위치에서 그리드 추론
print("\n=== Grid inference from walls ===")
# 1층 벽만 (z=0)
floor1_walls = [w for w in wall_data if abs(w["z"]) < 100]

x_coords = set()
y_coords = set()
for w in floor1_walls:
    x_coords.add(round(w["x"]))
    y_coords.add(round(w["y"]))
    # 벽 끝점도 추가 (profile이 있으면)
    if w["profile_x"] and w["profile_y"]:
        # 벽이 X방향인지 Y방향인지 판단
        if w["profile_x"] > w["profile_y"]:
            # X방향 벽 (길이가 X)
            x_coords.add(round(w["x"] + w["profile_x"]))
        else:
            # Y방향 벽 (길이가 Y)
            y_coords.add(round(w["y"] + w["profile_y"]))

print(f"  X coords: {sorted(x_coords)}")
print(f"  Y coords: {sorted(y_coords)}")

# 4. 슬래브 boundary 추출
print("\n=== Slab boundaries ===")
slabs = ifc.by_type("IfcSlab")
for sl in slabs:
    matrix = ifc_p.get_local_placement(sl.ObjectPlacement)
    x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]

    thickness = None
    boundary_pts = []
    if sl.Representation:
        for rep in sl.Representation.Representations:
            for item in rep.Items:
                if item.is_a("IfcExtrudedAreaSolid"):
                    thickness = item.Depth
                    profile = item.SweptArea
                    if profile.is_a("IfcArbitraryClosedProfileDef"):
                        curve = profile.OuterCurve
                        if curve.is_a("IfcPolyline"):
                            for pt in curve.Points:
                                boundary_pts.append(pt.Coordinates)

    print(f"  Slab {sl.Tag} at z={z:.0f}, thickness={thickness}")
    if boundary_pts:
        print(f"    Boundary points: {boundary_pts}")

# 5. 단위 확인
print("\n=== Units ===")
units = ifc.by_type("IfcSIUnit")
for u in units:
    if u.UnitType in ("LENGTHUNIT", "AREAUNIT", "FORCEUNIT"):
        prefix = u.Prefix if u.Prefix else ""
        print(f"  {u.UnitType}: {prefix}{u.Name}")
