"""Generate a synthetic, IP-free IFC file for the paper's §4 application example.

Purpose
-------
`docs/paper1_open_source_alternative/validation/ifc_example_model.json` is the
frozen node-element model behind the paper's §4/§5 IFC application example and
its ETABS cross-validation (see `_export_ifc_example_for_etabs.py`). That JSON
was NOT produced by parsing an IFC file — it was generated directly from a
JSON config (`example_input.json`) via `BuildingModel.from_json()`. That means
the paper's foregrounded "IFC -> node-element model" pipeline steps
(`core/ifc_parser.py::parse_ifc` for the V1 grid path, and
`core/ifc_parser_v2.py::parse_ifc_v2` + `snap_beams_to_columns` for the V2
node-element product path) were never exercised against a shareable artifact
for this example.

This script closes that gap: it authors a clean, minimal, from-scratch IFC
file (3-story, 3x2 bay steel moment frame; columns + beams only; no walls or
slabs) that reproduces the SAME geometry/section/material inputs that were fed
into `BuildingModel.from_json()` originally, but expressed as real
IfcColumn/IfcBeam/IfcBuildingStorey entities with IfcIShapeProfileDef sections.
Both parser paths recover the identical 48-node / 87-member regular grid (348
analysis elements at num_elements_per_member=4).

Schema
------
Written as **IFC2X3** — the schema the manuscript prose (§4, P121) states for
the example ("IFC 2x3"). Notable IFC2X3-vs-IFC4 authoring differences handled
here:
  * `IfcIShapeProfileDef.Position` (IfcAxis2Placement2D) is MANDATORY in
    IFC2X3 (optional in IFC4) — supplied for every profile.
  * `IfcColumn`/`IfcBeam` have 8 attributes in IFC2X3 (no `PredefinedType`,
    which IFC4 adds as a 9th) — created via keyword args so both schemas work.
  * `IfcMaterial` takes only `Name` in IFC2X3 (IFC4 adds Description/Category).
The parser code (`core/ifc_parser.py`, `core/ifc_parser_v2.py`) reads only
schema-invariant constructs (ObjectPlacement, IfcBuildingStorey.Elevation,
IfcIShapeProfileDef dims, IfcExtrudedAreaSolid geometry,
IfcRelAssociatesMaterial), so both paths behave identically across schemas.

Geometry authoring
------------------
Each member is a **proper perpendicular extrusion**: the ObjectPlacement local
frame is rotated so local +Z runs along the member axis (i->j), the profile
sits in the local XY plane, and the IfcExtrudedAreaSolid extrudes along local
+Z. This yields clean, valid 3D beam/column solids (no degenerate/oblique
sheets), with the I-section strong axis (OverallDepth) oriented vertically for
the horizontal beams — i.e. a file that also renders correctly in an IFC
viewer.

Realistic column-face offset (drives the V2 connectivity-repair demo)
--------------------------------------------------------------------
Beam ends are pulled back from the column centreline by the column half-depth
(H-300x300 -> 150 mm) so beams frame into the column FACE, exactly as a BIM
authoring tool exports them. This offset (0.15 m) is larger than the V2
node-merge tolerance (10 mm) but smaller than the snap tolerance (0.5 m), so
`parse_ifc_v2` extracts beam endpoints as separate joint nodes, raises the
`IFC_DISCONNECTED_JOINTS` validation issue, and `snap_beams_to_columns()`
repairs connectivity back onto the 48-node column grid — demonstrating the
node-element extraction + connectivity-repair product path on a physically
meaningful artifact. Columns are authored full length with ends exactly on the
grid, so they define the grid nodes.

How this maps onto parser behaviour
-----------------------------------
- Coordinates authored in millimetres (IfcSIUnit LENGTHUNIT = MILLI.METRE);
  matches the V1 parser's default `tolerance=200.0` mm cluster window.
- 4 IfcBuildingStorey: "1F"@0, "2F"@3000, "3F"@6000, "RF"@9000 (mm). Naming
  matters: `_filter_above_ground_stories` keeps `\\d+F` names; `_is_roof_level`
  matches "RF" so the 4th storey is consumed as a roof marker -> exactly 3
  "stories" of 3.0 m each (matching ifc_example_model.json), not 4.
- Column ObjectPlacement.Location = the column base node (mm); V1's
  `_get_column_positions()` clusters these into the 4x3 grid.
- Every element's IfcExtrudedAreaSolid.SweptArea is an IfcIShapeProfileDef with
  OverallDepth/OverallWidth/WebThickness/FlangeThickness -> `_ishape_to_dict()`
  reads real dimensions directly (no ProfileName parsing / DB fuzzy-match).
- One IfcMaterial "SS275" associated with all members;
  `_normalize_material_name()` matches it.

Usage
-----
    python scripts/_generate_synthetic_ifc_example.py

Writes:
    docs/paper1_open_source_alternative/validation/ifc_example.ifc
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import ifcopenshell
import ifcopenshell.guid as guid

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "docs" / "paper1_open_source_alternative" / "validation"
MODEL_JSON = VAL_DIR / "ifc_example_model.json"
OUT_IFC = VAL_DIR / "ifc_example.ifc"

SCHEMA = "IFC2X3"

# IfcBuildingStorey names chosen to satisfy ifc_parser's naming heuristics:
#   - "NF" (\d+F) survives _filter_above_ground_stories' "level" filter
#   - "RF" is recognized by _is_roof_level() -> consumed as roof marker,
#     leaving exactly 3 "stories" entries (not 4) each of height 3.0 m.
STORY_NAMES = ["1F", "2F", "3F", "RF"]


def _axis_and_refdir(u: tuple[float, float, float]) -> tuple[tuple, tuple]:
    """Return (Axis, RefDirection) for an ObjectPlacement whose local +Z is the
    member axis `u`, oriented so the I-section OverallDepth (strong axis) is
    vertical for horizontal members.

    column  (+Z): local Z=(0,0,1), local X=(1,0,0)
    beam_x  (+X): local Z=(1,0,0), local X=(0,1,0) -> local Y=(0,0,1)=global Z
    beam_y  (+Y): local Z=(0,1,0), local X=(-1,0,0)-> local Y=(0,0,1)=global Z
    """
    ux, uy, uz = u
    if abs(uz) > 0.9:            # vertical member (column)
        return (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    if abs(ux) > 0.9:           # beam along X
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    # beam along Y
    return (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)


def main() -> int:
    data = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    members = data["members"]
    sections = data["sections"]
    section_map = data["section_map"]
    stories_m = data["stories"]  # [3.0, 3.0, 3.0]
    material_name = data["_meta"]["material"]

    assert len(stories_m) == 3, f"expected 3 stories, got {len(stories_m)}"
    elevations_mm = [0.0]
    for h in stories_m:
        elevations_mm.append(elevations_mm[-1] + h * 1000.0)
    # elevations_mm = [0, 3000, 6000, 9000]

    # Beam column-face offset = half the column section depth (mm).
    # 0.15 m: > V2 node-merge tol (0.01 m) so detected, < V2 snap tol (0.5 m) so repaired.
    col_section_name = section_map["column"]
    beam_end_offset_mm = float(sections[col_section_name]["h_mm"]) / 2.0  # 150 mm

    f = ifcopenshell.file(schema=SCHEMA)

    # ── units ──
    length_unit = f.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Prefix="MILLI", Name="METRE")
    area_unit = f.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE")
    volume_unit = f.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
    unit_assignment = f.create_entity("IfcUnitAssignment", Units=[length_unit, area_unit, volume_unit])

    # ── geometric representation context ──
    origin = f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    z_dir = f.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    x_dir = f.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    world_placement = f.create_entity(
        "IfcAxis2Placement3D", Location=origin, Axis=z_dir, RefDirection=x_dir
    )
    context = f.create_entity(
        "IfcGeometricRepresentationContext", ContextType="Model",
        CoordinateSpaceDimension=3, Precision=1e-5,
        WorldCoordinateSystem=world_placement,
    )
    body_subcontext = f.create_entity(
        "IfcGeometricRepresentationSubContext", ContextIdentifier="Body",
        ContextType="Model", ParentContext=context, TargetView="MODEL_VIEW",
    )

    # ── project / site / building hierarchy ──
    project = f.create_entity(
        "IfcProject", GlobalId=guid.new(), Name="ifc_example — 3-story steel frame",
        Description="Synthetic reproduction bundle for Paper 1 §4 IFC application example",
        RepresentationContexts=[context], UnitsInContext=unit_assignment,
    )
    site = f.create_entity("IfcSite", GlobalId=guid.new(), Name="Site")
    f.create_entity(
        "IfcRelAggregates", GlobalId=guid.new(), RelatingObject=project, RelatedObjects=[site]
    )
    building = f.create_entity("IfcBuilding", GlobalId=guid.new(), Name="Building")
    f.create_entity(
        "IfcRelAggregates", GlobalId=guid.new(), RelatingObject=site, RelatedObjects=[building]
    )

    storeys = []
    for name, elev in zip(STORY_NAMES, elevations_mm):
        s = f.create_entity("IfcBuildingStorey", GlobalId=guid.new(), Name=name, Elevation=elev)
        storeys.append(s)
    f.create_entity(
        "IfcRelAggregates", GlobalId=guid.new(), RelatingObject=building, RelatedObjects=storeys
    )

    # ── material ──
    material = f.create_entity("IfcMaterial", Name=material_name)

    # ── section profiles (one IfcIShapeProfileDef per named section) ──
    # IFC2X3: Position (IfcAxis2Placement2D) is a REQUIRED attribute.
    profiles = {}
    for name, dims in sections.items():
        p2d_origin = f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0))
        p2d_xdir = f.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0))
        pos2d = f.create_entity("IfcAxis2Placement2D", Location=p2d_origin, RefDirection=p2d_xdir)
        profiles[name] = f.create_entity(
            "IfcIShapeProfileDef", ProfileType="AREA", ProfileName=name, Position=pos2d,
            OverallWidth=float(dims["b_mm"]), OverallDepth=float(dims["h_mm"]),
            WebThickness=float(dims["tw_mm"]), FlangeThickness=float(dims["tf_mm"]),
        )

    def make_element(kind: str, name: str, tag, ni: int, nj: int, section_name: str):
        pi = (nodes[ni]["x"] * 1000.0, nodes[ni]["y"] * 1000.0, nodes[ni]["z"] * 1000.0)
        pj = (nodes[nj]["x"] * 1000.0, nodes[nj]["y"] * 1000.0, nodes[nj]["z"] * 1000.0)
        dx, dy, dz = pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]
        full_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        u = (dx / full_len, dy / full_len, dz / full_len)

        # beams frame into the column face -> pull both ends in by the offset;
        # columns run full length with ends exactly on grid nodes.
        off = beam_end_offset_mm if kind == "beam" else 0.0
        start = (pi[0] + u[0] * off, pi[1] + u[1] * off, pi[2] + u[2] * off)
        depth = full_len - 2.0 * off

        axis, refdir = _axis_and_refdir(u)
        loc = f.create_entity("IfcCartesianPoint", Coordinates=start)
        placement3d = f.create_entity(
            "IfcAxis2Placement3D", Location=loc,
            Axis=f.create_entity("IfcDirection", DirectionRatios=axis),
            RefDirection=f.create_entity("IfcDirection", DirectionRatios=refdir),
        )
        obj_placement = f.create_entity("IfcLocalPlacement", RelativePlacement=placement3d)

        # Sketch plane = identity in the element's local frame (profile in local
        # XY, normal = local Z = member axis). Extrude along local +Z -> a clean
        # perpendicular extrusion of length `depth`.
        sk_loc = f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
        sk_place = f.create_entity("IfcAxis2Placement3D", Location=sk_loc)
        extrude_dir = f.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        solid = f.create_entity(
            "IfcExtrudedAreaSolid", SweptArea=profiles[section_name], Position=sk_place,
            ExtrudedDirection=extrude_dir, Depth=depth,
        )
        shape_rep = f.create_entity(
            "IfcShapeRepresentation", ContextOfItems=body_subcontext,
            RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid],
        )
        prod_shape = f.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])

        entity_type = "IfcColumn" if kind == "column" else "IfcBeam"
        return f.create_entity(
            entity_type, GlobalId=guid.new(), Name=name, Tag=str(tag),
            ObjectPlacement=obj_placement, Representation=prod_shape,
        )

    columns_by_story: dict[int, list] = {1: [], 2: [], 3: []}
    beams_by_story: dict[int, list] = {1: [], 2: [], 3: []}
    all_elements = []

    for m in members:
        kind = "column" if m["type"] == "column" else "beam"
        section_name = section_map[m["type"]]
        elem = make_element(kind, f"{m['type']}_{m['member_id']}", m["member_id"],
                            m["ni"], m["nj"], section_name)
        all_elements.append(elem)
        story = m["story"]
        (columns_by_story if kind == "column" else beams_by_story)[story].append(elem)

    # ── spatial containment (not required by the parser; included for a
    #     semantically valid / human-reviewable IFC) ──
    #   columns of story i -> storey i (they rise FROM that level)
    #   beams of story i   -> storey i+1 (they sit AT the top of that story)
    for story_idx in (1, 2, 3):
        cols = columns_by_story[story_idx]
        if cols:
            f.create_entity(
                "IfcRelContainedInSpatialStructure", GlobalId=guid.new(),
                RelatedElements=cols, RelatingStructure=storeys[story_idx - 1],
            )
        beams = beams_by_story[story_idx]
        if beams:
            f.create_entity(
                "IfcRelContainedInSpatialStructure", GlobalId=guid.new(),
                RelatedElements=beams, RelatingStructure=storeys[story_idx],
            )

    # ── material association (single relationship covering all members) ──
    f.create_entity(
        "IfcRelAssociatesMaterial", GlobalId=guid.new(),
        RelatedObjects=all_elements, RelatingMaterial=material,
    )

    OUT_IFC.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(OUT_IFC))
    n_cols = sum(len(v) for v in columns_by_story.values())
    n_beams = sum(len(v) for v in beams_by_story.values())
    print(f"[OK] wrote {OUT_IFC.relative_to(ROOT)}")
    print(f"     schema={SCHEMA}  columns={n_cols}  beams={n_beams}"
          f"  total_elements={len(all_elements)}  beam_end_offset={beam_end_offset_mm:.0f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
