"""Minimal IFC4 fixture builder for V2 parser tests.

Generates a tiny IFC building (4 columns + 2 beams, 1 story) at import time so
the parser test suite has something to run against in CI even when the original
``data/debug_uploaded.ifc`` is not committed to the repo.

The fixture is cached as ``tests/fixtures/minimal_v2_building.ifc`` on first
build; subsequent test runs reuse it. Delete the file to force regeneration.

Geometry intent (so test assertions can rely on these counts):
    - 1 story (Ground @ 0 m, Level 1 @ 3 m)
    - 4 columns (3 m tall) at corners of a 6x6 m square — generates 8 nodes
    - 2 beams (parallel to Y, 6 m long) at the top, intentionally offset 0.2 m
      in X from the column centerlines so the parser raises
      ``IFC_DISCONNECTED_JOINTS`` (and the snap test has something to snap)
    - No IfcSlab → ``IFC_NO_SLAB`` validation issue
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.api
    HAS_IFC = True
except ImportError:  # ifcopenshell missing — fixture cannot be built
    HAS_IFC = False


FIXTURE_PATH = Path(__file__).resolve().parent / "minimal_v2_building.ifc"

# Geometry constants (m)
SPAN = 6.0
STORY_HEIGHT = 3.0
COLUMN_SECTION_MM = 400.0  # 400x400 column
BEAM_DEPTH_MM = 400.0
BEAM_WIDTH_MM = 200.0
BEAM_OFFSET_M = 0.2  # intentional disconnect (snap target)

# Corner column positions (XY in m), z = 0
COLUMN_POSITIONS: list[Tuple[float, float]] = [
    (0.0, 0.0),
    (SPAN, 0.0),
    (0.0, SPAN),
    (SPAN, SPAN),
]

# Beams along Y at the top, intentionally offset in X by +BEAM_OFFSET_M
BEAM_LINES: list[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = [
    (
        (0.0 + BEAM_OFFSET_M, 0.0, STORY_HEIGHT),
        (0.0 + BEAM_OFFSET_M, SPAN, STORY_HEIGHT),
    ),
    (
        (SPAN + BEAM_OFFSET_M, 0.0, STORY_HEIGHT),
        (SPAN + BEAM_OFFSET_M, SPAN, STORY_HEIGHT),
    ),
]


def _placement_matrix(translation_m: Tuple[float, float, float],
                      z_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                      x_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    """Build a 4x4 placement matrix with the given Z (extrusion) axis."""
    z = np.array(z_axis, dtype=float)
    z /= np.linalg.norm(z)
    x = np.array(x_axis, dtype=float)
    # Re-orthogonalize x w.r.t. z
    x -= z * np.dot(x, z)
    if np.linalg.norm(x) < 1e-9:
        # Pick any non-parallel reference
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = ref - z * np.dot(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.eye(4)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = translation_m
    return mat


def build_minimal_ifc(path: Path = FIXTURE_PATH, force: bool = False) -> Path:
    """Create (or reuse) the minimal IFC fixture and return its path.

    Raises:
        ImportError: if ifcopenshell is not installed.
    """
    if not HAS_IFC:
        raise ImportError(
            "ifcopenshell is required to build the IFC fixture. "
            "Install it with `pip install ifcopenshell`."
        )
    path = Path(path)
    if path.exists() and not force:
        return path

    api = ifcopenshell.api.run
    f = api("project.create_file", version="IFC4")
    project = api("root.create_entity", f, ifc_class="IfcProject", name="V2TestProject")
    # Real Revit IFCs ship in millimetres; the V2 parser expects geometry in mm
    # via the placement fallback and in meters via ifcopenshell.geom (which
    # converts to project units automatically).
    api("unit.assign_unit", f, length={"is_metric": True, "raw": "MILLIMETERS"})
    model_ctx = api("context.add_context", f, context_type="Model")
    body_ctx = api(
        "context.add_context", f,
        context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=model_ctx,
    )

    site = api("root.create_entity", f, ifc_class="IfcSite", name="Site")
    api("aggregate.assign_object", f, products=[site], relating_object=project)
    building = api("root.create_entity", f, ifc_class="IfcBuilding", name="Bldg")
    api("aggregate.assign_object", f, products=[building], relating_object=site)

    storey_ground = api("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Ground")
    storey_ground.Elevation = 0.0
    storey_l1 = api("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level1")
    # IfcBuildingStorey.Elevation is in the project length unit (mm here).
    storey_l1.Elevation = STORY_HEIGHT * 1000.0
    api("aggregate.assign_object", f, products=[storey_ground, storey_l1], relating_object=building)

    # Reuse a single column profile (400x400 rectangular)
    col_profile = f.create_entity(
        "IfcRectangleProfileDef",
        ProfileType="AREA",
        ProfileName="C-400x400",
        XDim=COLUMN_SECTION_MM,
        YDim=COLUMN_SECTION_MM,
    )

    columns = []
    for i, (x, y) in enumerate(COLUMN_POSITIONS, start=1):
        col = api("root.create_entity", f, ifc_class="IfcColumn", name=f"C{i}")
        # is_si=True means "the matrix translation is given in meters"; the
        # function converts to file units (mm) internally.
        api("geometry.edit_object_placement", f, product=col,
            matrix=_placement_matrix((x, y, 0.0)), is_si=True)
        # depth is in the file's length unit (mm).
        rep = api("geometry.add_profile_representation", f, context=body_ctx,
                  profile=col_profile, depth=STORY_HEIGHT * 1000.0)
        api("geometry.assign_representation", f, product=col, representation=rep)
        columns.append(col)
    api("spatial.assign_container", f, products=columns, relating_structure=storey_ground)

    # Beams along Y (extrusion direction is local +Z, so rotate local +Z onto global +Y)
    beam_profile = f.create_entity(
        "IfcRectangleProfileDef",
        ProfileType="AREA",
        ProfileName="B-400x200",
        XDim=BEAM_WIDTH_MM,
        YDim=BEAM_DEPTH_MM,
    )

    beams = []
    for i, (start, _end) in enumerate(BEAM_LINES, start=1):
        beam = api("root.create_entity", f, ifc_class="IfcBeam", name=f"B{i}")
        # Local Z (extrusion) along global +Y; local X along global -Z so the
        # axis frame stays right-handed.
        matrix = _placement_matrix(start, z_axis=(0.0, 1.0, 0.0), x_axis=(0.0, 0.0, -1.0))
        api("geometry.edit_object_placement", f, product=beam, matrix=matrix, is_si=True)
        rep = api("geometry.add_profile_representation", f, context=body_ctx,
                  profile=beam_profile, depth=SPAN * 1000.0)
        api("geometry.assign_representation", f, product=beam, representation=rep)
        beams.append(beam)
    api("spatial.assign_container", f, products=beams, relating_structure=storey_l1)

    path.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(path))
    return path


if __name__ == "__main__":  # manual rebuild
    out = build_minimal_ifc(force=True)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
