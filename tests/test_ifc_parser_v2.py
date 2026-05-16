"""V2 IFC Parser tests.

This file is split into two layers:

1.  ``TestPureHelpers`` and ``TestModelJsonRoundtrip`` exercise the pure-Python
    parts of the V2 pipeline (classification, joint detection, snap, JSON
    roundtrip). They do **not** require ifcopenshell and always run in CI so
    the IFC pipeline has at least one meaningful guarantee even on a slim
    runner.

2.  ``TestParseFixtureIFC`` parses a tiny IFC4 file generated on demand by
    :mod:`tests.fixtures.minimal_ifc` (4 columns + 2 beams, 1 story). The
    fixture used to ship as ``data/debug_uploaded.ifc`` but was removed from
    the repo. It is regenerated lazily and skipped only when ifcopenshell is
    not installed at all.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# mcp-server is also added by conftest.py at the repo root, but keep this
# line so the file can also be run directly with `python -m pytest <file>`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-server"))

from core.structural_model import (
    StructuralModel,
    ElementType,
    SupportType,
    Severity,
)

try:
    import ifcopenshell  # noqa: F401
    HAS_IFCOPENSHELL = True
except ImportError:
    HAS_IFCOPENSHELL = False


# ──────────────────────────────────────────────────────────────
#  Pure-Python helpers (always run, no IFC required)
# ──────────────────────────────────────────────────────────────

class TestPureHelpers:
    """Unit tests for V2 helper functions that do not need a real IFC file."""

    def test_classify_vertical_member_is_column(self):
        import numpy as np
        from core.ifc_parser_v2 import _classify_by_direction
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 0.0, 3.0])
        assert _classify_by_direction(start, end) == ElementType.COLUMN

    def test_classify_horizontal_member_is_beam(self):
        import numpy as np
        from core.ifc_parser_v2 import _classify_by_direction
        start = np.array([0.0, 0.0, 3.0])
        end = np.array([6.0, 0.0, 3.0])
        assert _classify_by_direction(start, end) == ElementType.BEAM

    def test_classify_diagonal_member_is_brace(self):
        import numpy as np
        from core.ifc_parser_v2 import _classify_by_direction
        # 45 degrees from vertical — outside both column and beam thresholds
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([3.0, 0.0, 3.0])
        assert _classify_by_direction(start, end) == ElementType.BRACE

    def test_classify_zero_length_defaults_to_beam(self):
        import numpy as np
        from core.ifc_parser_v2 import _classify_by_direction
        # < 1 mm length — short-circuit branch
        zero = np.array([0.0, 0.0, 0.0])
        assert _classify_by_direction(zero, zero) == ElementType.BEAM

    def test_detect_disconnected_joints_finds_offset_beam(self):
        """Beam endpoints offset from column nodes should be detected."""
        from core.ifc_parser_v2 import _detect_disconnected_joints

        model = StructuralModel()
        # Column from (0,0,0) → (0,0,3)
        c0 = model.add_node(0.0, 0.0, 0.0, support=SupportType.FIXED).id
        c1 = model.add_node(0.0, 0.0, 3.0).id
        model.add_element(node_i=c0, node_j=c1, elem_type=ElementType.COLUMN)
        # Beam at z=3 but offset by 0.2 m in X (disconnected from column top)
        b0 = model.add_node(0.2, 0.0, 3.0).id
        b1 = model.add_node(0.2, 6.0, 3.0).id
        model.add_element(node_i=b0, node_j=b1, elem_type=ElementType.BEAM)

        candidates = _detect_disconnected_joints(model, snap_tolerance=0.5)
        # b0 is within 0.5 m of c1 (the column top) → 1 snap candidate
        assert len(candidates) >= 1
        ids = {c["beam_node_id"] for c in candidates}
        assert b0 in ids

    def test_snap_beams_to_columns_reduces_nodes(self):
        """snap_beams_to_columns should remove orphan beam nodes."""
        from core.ifc_parser_v2 import snap_beams_to_columns

        model = StructuralModel()
        c0 = model.add_node(0.0, 0.0, 0.0, support=SupportType.FIXED).id
        c1 = model.add_node(0.0, 0.0, 3.0).id
        model.add_element(node_i=c0, node_j=c1, elem_type=ElementType.COLUMN)

        b0 = model.add_node(0.2, 0.0, 3.0).id
        b1 = model.add_node(0.2, 6.0, 3.0).id
        model.add_element(node_i=b0, node_j=b1, elem_type=ElementType.BEAM)

        nodes_before = len(model.nodes)
        elements_before = len(model.elements)
        snapped = snap_beams_to_columns(model, snap_tolerance=0.5)

        assert snapped >= 1
        # The beam end that snapped onto the column top should no longer
        # exist as a separate node; element count is preserved.
        assert len(model.nodes) < nodes_before
        assert len(model.elements) == elements_before

    def test_snap_is_idempotent(self):
        """Running snap twice should not change the model further."""
        from core.ifc_parser_v2 import snap_beams_to_columns

        model = StructuralModel()
        c0 = model.add_node(0.0, 0.0, 0.0, support=SupportType.FIXED).id
        c1 = model.add_node(0.0, 0.0, 3.0).id
        model.add_element(node_i=c0, node_j=c1, elem_type=ElementType.COLUMN)
        b0 = model.add_node(0.2, 0.0, 3.0).id
        b1 = model.add_node(0.2, 6.0, 3.0).id
        model.add_element(node_i=b0, node_j=b1, elem_type=ElementType.BEAM)

        snap_beams_to_columns(model, snap_tolerance=0.5)
        nodes_after_first = len(model.nodes)
        snap_beams_to_columns(model, snap_tolerance=0.5)
        assert len(model.nodes) == nodes_after_first


# ──────────────────────────────────────────────────────────────
#  StructuralModel JSON roundtrip (no IFC required)
# ──────────────────────────────────────────────────────────────

class TestModelJsonRoundtrip:
    def test_roundtrip_preserves_structure(self):
        model = StructuralModel()
        n1 = model.add_node(0.0, 0.0, 0.0, support=SupportType.FIXED).id
        n2 = model.add_node(0.0, 0.0, 3.0).id
        n3 = model.add_node(6.0, 0.0, 3.0).id
        model.add_element(node_i=n1, node_j=n2, elem_type=ElementType.COLUMN,
                          section="H-300x300", material="SS275")
        model.add_element(node_i=n2, node_j=n3, elem_type=ElementType.BEAM,
                          section="H-400x200", material="SS275")
        model.story_elevations = [0.0, 3.0]

        data = model.to_json()
        rebuilt = StructuralModel.from_json(data)

        assert len(rebuilt.nodes) == len(model.nodes)
        assert len(rebuilt.elements) == len(model.elements)
        assert rebuilt.story_elevations == model.story_elevations
        # Element references survive
        for orig_id, orig in model.elements.items():
            new = rebuilt.elements[orig_id]
            assert new.section == orig.section
            assert new.material == orig.material
            assert new.elem_type == orig.elem_type


# ──────────────────────────────────────────────────────────────
#  Fixture-driven IFC parse (only when ifcopenshell available)
# ──────────────────────────────────────────────────────────────

skipif_no_ifc = pytest.mark.skipif(
    not HAS_IFCOPENSHELL,
    reason="ifcopenshell is not installed in this environment",
)


@pytest.fixture(scope="module")
def fixture_ifc_path():
    """Build (or reuse) the small IFC4 fixture for parser smoke tests."""
    if not HAS_IFCOPENSHELL:
        pytest.skip("ifcopenshell missing — fixture cannot be built")
    from tests.fixtures.minimal_ifc import build_minimal_ifc
    return str(build_minimal_ifc())


@skipif_no_ifc
class TestParseFixtureIFC:
    """End-to-end smoke tests against the generated fixture IFC."""

    def test_parse_returns_model_and_validation(self, fixture_ifc_path):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, validation = parse_ifc_v2(fixture_ifc_path)
        assert isinstance(model, StructuralModel)
        # CRITICAL issues should never appear on the fixture
        assert validation.is_valid

    def test_extracted_element_counts(self, fixture_ifc_path):
        """Fixture has exactly 4 IfcColumn + 2 IfcBeam."""
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(fixture_ifc_path)
        cols = model.elements_by_type(ElementType.COLUMN)
        beams = model.elements_by_type(ElementType.BEAM)
        assert len(cols) == 4
        assert len(beams) == 2
        assert len(model.elements_by_type(ElementType.BRACE)) == 0

    def test_no_failed_elements(self, fixture_ifc_path):
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(fixture_ifc_path)
        assert validation.failed_elements == 0

    def test_no_slab_warning_present(self, fixture_ifc_path):
        """The fixture deliberately omits IfcSlab → IFC_NO_SLAB."""
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(fixture_ifc_path)
        codes = {i.code for i in validation.issues}
        assert "IFC_NO_SLAB" in codes

    # Note: IFC_DISCONNECTED_JOINTS is not asserted on the fixture because
    # ifcopenshell.api stores rectangular profiles in meters but placements
    # in millimetres for IFC4 files built this way, which yields mixed-unit
    # node coordinates. The disconnect-detection code path is exercised in
    # ``TestPureHelpers.test_detect_disconnected_joints_finds_offset_beam``
    # with clean meter coordinates.

    def test_base_nodes_are_fixed(self, fixture_ifc_path):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(fixture_ifc_path)
        base = model.base_nodes()
        # 4 corner columns share 4 base nodes
        assert len(base) >= 4
        for n in base:
            assert n.support == SupportType.FIXED

    def test_story_elevations_extracted(self, fixture_ifc_path):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(fixture_ifc_path)
        # At least one ground level + one upper level
        assert len(model.story_elevations) >= 2

    def test_stats_info_recorded(self, fixture_ifc_path):
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(fixture_ifc_path)
        stats = [i for i in validation.issues if i.code == "IFC_STATS"]
        assert len(stats) == 1
        assert stats[0].severity == Severity.INFO
