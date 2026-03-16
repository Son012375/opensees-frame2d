"""4차시도 IFC 파일 탐색."""
import ifcopenshell
import ifcopenshell.util.placement as ifc_p

IFC_PATH = r"C:\Users\youm\Documents\카카오톡 받은 파일\4차시도_260121.ifc"
ifc = ifcopenshell.open(IFC_PATH)

# 1. 스키마 + 엔티티 개수
print(f"Schema: {ifc.schema}")
entity_types = {}
for e in ifc:
    t = e.is_a()
    entity_types[t] = entity_types.get(t, 0) + 1

structural = ['IfcBuilding', 'IfcBuildingStorey', 'IfcColumn', 'IfcBeam',
              'IfcSlab', 'IfcWallStandardCase', 'IfcWall', 'IfcMember',
              'IfcDoor', 'IfcWindow', 'IfcRoof', 'IfcStairFlight',
              'IfcMaterial', 'IfcMaterialProfileSet', 'IfcMaterialLayerSet']
print("\n=== Structural entities ===")
for t in structural:
    if t in entity_types:
        print(f"  {t}: {entity_types[t]}")

# 2. 층 정보
print("\n=== IfcBuildingStorey ===")
storeys = ifc.by_type("IfcBuildingStorey")
for s in sorted(storeys, key=lambda x: x.Elevation if x.Elevation else 0):
    print(f"  {s.Name!r}: elev={s.Elevation:.0f} mm")

# 3. 기둥
columns = ifc.by_type("IfcColumn")
print(f"\n=== IfcColumn ({len(columns)}) ===")
for col in columns[:10]:
    matrix = ifc_p.get_local_placement(col.ObjectPlacement)
    x, y, z = float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])
    print(f"  {col.Name!r}: pos=({x:.0f}, {y:.0f}, {z:.0f})")

# 4. 보
beams = ifc.by_type("IfcBeam")
print(f"\n=== IfcBeam ({len(beams)}) ===")
for b in beams[:10]:
    matrix = ifc_p.get_local_placement(b.ObjectPlacement)
    x, y, z = float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])
    print(f"  {b.Name!r}: pos=({x:.0f}, {y:.0f}, {z:.0f})")

# 5. 벽
walls = ifc.by_type("IfcWallStandardCase") or ifc.by_type("IfcWall")
print(f"\n=== Walls ({len(walls)}) ===")
for w in walls[:10]:
    matrix = ifc_p.get_local_placement(w.ObjectPlacement)
    x, y, z = float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])
    # profile
    profile_x, profile_y, depth = None, None, None
    if w.Representation:
        for rep in w.Representation.Representations:
            for item in rep.Items:
                if item.is_a("IfcExtrudedAreaSolid"):
                    depth = float(item.Depth)
                    p = item.SweptArea
                    if p.is_a("IfcRectangleProfileDef"):
                        profile_x = float(p.XDim)
                        profile_y = float(p.YDim)
    # direction
    local_x_gx = float(matrix[0][0])
    local_x_gy = float(matrix[1][0])
    d = "X" if abs(local_x_gx) > abs(local_x_gy) else "Y"
    print(f"  {w.Tag}: pos=({x:.0f},{y:.0f},{z:.0f}) "
          f"profile=({profile_x}x{profile_y}) h={depth} dir={d}")

# 6. 슬래브
slabs = ifc.by_type("IfcSlab")
print(f"\n=== IfcSlab ({len(slabs)}) ===")
seen = set()
for sl in slabs:
    if sl.Tag in seen:
        continue
    seen.add(sl.Tag)
    matrix = ifc_p.get_local_placement(sl.ObjectPlacement)
    z = float(matrix[2][3])
    thickness = None
    if sl.Representation:
        for rep in sl.Representation.Representations:
            for item in rep.Items:
                if item.is_a("IfcExtrudedAreaSolid"):
                    thickness = float(item.Depth)
    stype = getattr(sl, "PredefinedType", "?")
    print(f"  {sl.Tag}: z={z:.0f}mm, t={thickness}mm, type={stype}")

# 7. 재료
mats = ifc.by_type("IfcMaterial")
print(f"\n=== IfcMaterial ({len(mats)}) ===")
for m in mats:
    print(f"  {m.Name!r}")
