"""IFC Parser V2 테스트.

실제 IFC 파일(data/debug_uploaded.ifc)을 사용하여
V2 파서의 노드-요소 추출, 검증, 스냅 기능을 검증한다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-server"))

from core.structural_model import (
    StructuralModel,
    ElementType,
    SupportType,
    Severity,
)

# IFC 파일 경로
IFC_PATH = str(Path(__file__).resolve().parent.parent / "data" / "debug_uploaded.ifc")
HAS_IFC_FILE = Path(IFC_PATH).exists()

try:
    import ifcopenshell
    HAS_IFCOPENSHELL = True
except ImportError:
    HAS_IFCOPENSHELL = False

skipif_no_ifc = pytest.mark.skipif(
    not (HAS_IFC_FILE and HAS_IFCOPENSHELL),
    reason="IFC file or ifcopenshell not available",
)


# ──────────────────────────────────────────────────────────────
#  파싱 기본
# ──────────────────────────────────────────────────────────────

@skipif_no_ifc
class TestParseBasic:
    def test_parse_returns_model_and_validation(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, validation = parse_ifc_v2(IFC_PATH)
        assert isinstance(model, StructuralModel)
        assert validation.is_valid

    def test_correct_element_counts(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(IFC_PATH)
        assert len(model.elements_by_type(ElementType.COLUMN)) == 32
        assert len(model.elements_by_type(ElementType.BEAM)) == 17
        assert len(model.elements_by_type(ElementType.BRACE)) == 0

    def test_no_failed_elements(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(IFC_PATH)
        assert validation.failed_elements == 0

    def test_story_detection(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(IFC_PATH)
        assert model.num_stories == 2
        assert len(model.story_elevations) == 3
        assert abs(model.story_elevations[0] - 0.0) < 0.01
        assert abs(model.story_elevations[1] - 4.0) < 0.01
        assert abs(model.story_elevations[2] - 8.0) < 0.01

    def test_base_nodes_fixed(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(IFC_PATH)
        base = model.base_nodes()
        assert len(base) == 16
        for n in base:
            assert n.support == SupportType.FIXED

    def test_sections_extracted(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(IFC_PATH)
        sections = set(e.section for e in model.elements.values())
        assert "H-400x408" in sections or "H-250x250" in sections

    def test_column_lengths(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        model, _ = parse_ifc_v2(IFC_PATH)
        for e in model.elements.values():
            if e.elem_type == ElementType.COLUMN:
                L = e.length(model.nodes)
                assert 3.5 < L < 4.5, f"Column {e.id} length {L}m out of range"


# ──────────────────────────────────────────────────────────────
#  검증
# ──────────────────────────────────────────────────────────────

@skipif_no_ifc
class TestValidation:
    def test_validation_issues_present(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(IFC_PATH)
        assert len(validation.issues) > 0

    def test_no_slab_warning(self):
        """이 IFC에는 IfcSlab가 없으므로 경고가 있어야 함."""
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(IFC_PATH)
        codes = [i.code for i in validation.issues]
        assert "IFC_NO_SLAB" in codes

    def test_disconnected_joints_detected(self):
        """스냅 전에 보-기둥 접합 불일치가 감지되어야 함."""
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(IFC_PATH)
        codes = [i.code for i in validation.issues]
        assert "IFC_DISCONNECTED_JOINTS" in codes

    def test_needs_user_input(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(IFC_PATH)
        assert len(validation.needs_user_input) > 0

    def test_stats_info(self):
        from core.ifc_parser_v2 import parse_ifc_v2
        _, validation = parse_ifc_v2(IFC_PATH)
        info = [i for i in validation.issues if i.code == "IFC_STATS"]
        assert len(info) == 1


# ──────────────────────────────────────────────────────────────
#  스냅
# ──────────────────────────────────────────────────────────────

@skipif_no_ifc
class TestSnapBeamsToColumns:
    def test_snap_reduces_nodes(self):
        from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
        model, _ = parse_ifc_v2(IFC_PATH)
        nodes_before = len(model.nodes)
        snapped = snap_beams_to_columns(model)
        assert snapped > 0
        assert len(model.nodes) < nodes_before

    def test_beams_connected_after_snap(self):
        from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
        model, _ = parse_ifc_v2(IFC_PATH)
        snap_beams_to_columns(model)

        for e in model.elements.values():
            if e.elem_type == ElementType.BEAM:
                cols_at_i = [
                    ee for ee in model.connected_elements(e.node_i)
                    if ee.elem_type == ElementType.COLUMN
                ]
                cols_at_j = [
                    ee for ee in model.connected_elements(e.node_j)
                    if ee.elem_type == ElementType.COLUMN
                ]
                assert len(cols_at_i) > 0, f"Beam {e.id} node_i has no columns"
                assert len(cols_at_j) > 0, f"Beam {e.id} node_j has no columns"

    def test_beam_z_matches_column_after_snap(self):
        from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
        model, _ = parse_ifc_v2(IFC_PATH)
        snap_beams_to_columns(model)

        for e in model.elements.values():
            if e.elem_type == ElementType.BEAM:
                ni = model.nodes[e.node_i]
                nj = model.nodes[e.node_j]
                # 보 노드가 층 표고와 일치해야 함 (기둥 상단)
                assert any(
                    abs(ni.z - elev) < 0.5 for elev in model.story_elevations
                ), f"Beam {e.id} node_i z={ni.z} not near any story"

    def test_element_count_unchanged_after_snap(self):
        from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
        model, _ = parse_ifc_v2(IFC_PATH)
        elems_before = len(model.elements)
        snap_beams_to_columns(model)
        assert len(model.elements) == elems_before

    def test_no_duplicate_snap(self):
        """스냅을 두 번 해도 추가 변경 없음."""
        from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
        model, _ = parse_ifc_v2(IFC_PATH)
        snap_beams_to_columns(model)
        nodes_after_first = len(model.nodes)
        snap_beams_to_columns(model)
        assert len(model.nodes) == nodes_after_first


# ──────────────────────────────────────────────────────────────
#  JSON 왕복
# ──────────────────────────────────────────────────────────────

@skipif_no_ifc
class TestJsonRoundtrip:
    def test_roundtrip_preserves_structure(self):
        from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
        model, _ = parse_ifc_v2(IFC_PATH)
        snap_beams_to_columns(model)

        data = model.to_json()
        model2 = StructuralModel.from_json(data)

        assert len(model2.nodes) == len(model.nodes)
        assert len(model2.elements) == len(model.elements)
        assert model2.story_elevations == model.story_elevations
