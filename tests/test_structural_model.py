"""StructuralModel V2 IR 테스트."""
import json
import math
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-server"))

from core.structural_model import (
    StructuralModel,
    StructuralNode,
    StructuralElement,
    ElementType,
    SupportType,
    ReleaseType,
    IFCValidationResult,
    ValidationIssue,
    Severity,
)


# ──────────────────────────────────────────────────────────────
#  노드 CRUD
# ──────────────────────────────────────────────────────────────

class TestNodeCRUD:
    def test_add_node(self):
        m = StructuralModel()
        n = m.add_node(x=0.0, y=0.0, z=0.0, support="fixed")
        assert n.id == 1
        assert n.support == SupportType.FIXED
        assert len(m.nodes) == 1

    def test_add_node_auto_id(self):
        m = StructuralModel()
        n1 = m.add_node(0, 0, 0)
        n2 = m.add_node(1, 0, 0)
        assert n1.id == 1
        assert n2.id == 2

    def test_add_node_custom_id(self):
        m = StructuralModel()
        n = m.add_node(0, 0, 0, node_id=100)
        assert n.id == 100
        n2 = m.add_node(1, 0, 0)  # auto-increment after 100
        assert n2.id == 101

    def test_add_duplicate_id_raises(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1)
        with pytest.raises(ValueError, match="이미 존재"):
            m.add_node(1, 0, 0, node_id=1)

    def test_remove_node(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1)
        m.remove_node(1)
        assert len(m.nodes) == 0

    def test_remove_connected_node_raises(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1)
        m.add_node(1, 0, 0, node_id=2)
        m.add_element(1, 2, elem_type="beam")
        with pytest.raises(ValueError, match="연결된 요소"):
            m.remove_node(1)

    def test_move_node(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1)
        m.move_node(1, 5.0, 3.0, 4.0)
        assert m.nodes[1].x == 5.0
        assert m.nodes[1].y == 3.0
        assert m.nodes[1].z == 4.0

    def test_node_distance(self):
        n1 = StructuralNode(id=1, x=0, y=0, z=0)
        n2 = StructuralNode(id=2, x=3, y=4, z=0)
        assert abs(n1.distance_to(n2) - 5.0) < 1e-9


# ──────────────────────────────────────────────────────────────
#  요소 CRUD
# ──────────────────────────────────────────────────────────────

class TestElementCRUD:
    def _model_with_nodes(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1, support="fixed")
        m.add_node(0, 0, 4, node_id=2)
        m.add_node(8, 0, 4, node_id=3)
        return m

    def test_add_element(self):
        m = self._model_with_nodes()
        e = m.add_element(1, 2, elem_type="column", section="H-300x300")
        assert e.id == 1
        assert e.elem_type == ElementType.COLUMN
        assert e.section == "H-300x300"

    def test_add_element_invalid_node_raises(self):
        m = self._model_with_nodes()
        with pytest.raises(ValueError, match="존재하지 않습니다"):
            m.add_element(1, 999)

    def test_add_element_same_node_raises(self):
        m = self._model_with_nodes()
        with pytest.raises(ValueError, match="같을 수 없습니다"):
            m.add_element(1, 1)

    def test_remove_element(self):
        m = self._model_with_nodes()
        m.add_element(1, 2, elem_id=10)
        m.remove_element(10)
        assert len(m.elements) == 0

    def test_update_element(self):
        m = self._model_with_nodes()
        m.add_element(1, 2, elem_id=1)
        m.update_element(1, section="H-500x200", release_i="moment_y")
        assert m.elements[1].section == "H-500x200"
        assert m.elements[1].release_i == ReleaseType.MOMENT_Y

    def test_element_length(self):
        m = self._model_with_nodes()
        e = m.add_element(1, 2, elem_type="column")
        assert abs(e.length(m.nodes) - 4.0) < 1e-9

    def test_direction_vector(self):
        m = self._model_with_nodes()
        # column: (0,0,0) → (0,0,4) → direction = (0,0,1)
        e = m.add_element(1, 2, elem_type="column")
        dx, dy, dz = e.direction_vector(m.nodes)
        assert abs(dz - 1.0) < 1e-9

    def test_release_types(self):
        m = self._model_with_nodes()
        e = m.add_element(2, 3, release_i="moment_yz", release_j="all")
        assert e.release_i == ReleaseType.MOMENT_YZ
        assert e.release_j == ReleaseType.ALL_MOMENT


# ──────────────────────────────────────────────────────────────
#  노드 병합
# ──────────────────────────────────────────────────────────────

class TestNodeMerge:
    def test_merge_close_nodes(self):
        m = StructuralModel()
        m.add_node(0.0, 0.0, 0.0, node_id=1, support="fixed")
        m.add_node(0.005, 0.003, 0.0, node_id=2)  # 1과 매우 가까움
        m.add_node(8.0, 0.0, 0.0, node_id=3)
        m.add_element(2, 3, elem_id=1)

        merge_map = m.merge_close_nodes(tolerance=0.01)

        assert 2 in merge_map
        assert merge_map[2] == 1
        assert 2 not in m.nodes  # 병합됨
        assert m.elements[1].node_i == 1  # 참조 업데이트됨
        assert m.nodes[1].support == SupportType.FIXED  # support 유지

    def test_no_merge_when_distant(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1)
        m.add_node(1, 0, 0, node_id=2)
        merge_map = m.merge_close_nodes(tolerance=0.01)
        assert len(merge_map) == 0
        assert len(m.nodes) == 2


# ──────────────────────────────────────────────────────────────
#  층 감지 & 요소 분류
# ──────────────────────────────────────────────────────────────

class TestStoryDetection:
    def _simple_frame(self):
        """2층 1경간 단순 프레임."""
        m = StructuralModel()
        # 기초 (z=0)
        m.add_node(0, 0, 0, node_id=1, support="fixed")
        m.add_node(8, 0, 0, node_id=2, support="fixed")
        # 1층 (z=4)
        m.add_node(0, 0, 4, node_id=3)
        m.add_node(8, 0, 4, node_id=4)
        # 2층 (z=7.5)
        m.add_node(0, 0, 7.5, node_id=5)
        m.add_node(8, 0, 7.5, node_id=6)
        # 기둥
        m.add_element(1, 3, elem_type="column")
        m.add_element(2, 4, elem_type="column")
        m.add_element(3, 5, elem_type="column")
        m.add_element(4, 6, elem_type="column")
        # 보
        m.add_element(3, 4, elem_type="beam")
        m.add_element(5, 6, elem_type="beam")
        return m

    def test_detect_stories(self):
        m = self._simple_frame()
        elevations = m.detect_stories()
        assert len(elevations) == 3  # z=0, 4.0, 7.5
        assert abs(elevations[0] - 0.0) < 0.01
        assert abs(elevations[1] - 4.0) < 0.01
        assert abs(elevations[2] - 7.5) < 0.01

    def test_num_stories(self):
        m = self._simple_frame()
        m.detect_stories()
        assert m.num_stories == 2

    def test_story_heights(self):
        m = self._simple_frame()
        m.detect_stories()
        heights = m.story_heights
        assert len(heights) == 2
        assert abs(heights[0] - 4.0) < 0.01
        assert abs(heights[1] - 3.5) < 0.01

    def test_nodes_at_story(self):
        m = self._simple_frame()
        m.detect_stories()
        base = m.nodes_at_story(0)
        assert len(base) == 2
        floor1 = m.nodes_at_story(1)
        assert len(floor1) == 2

    def test_classify_elements(self):
        m = self._simple_frame()
        # 일부러 모든 요소를 beam으로 두고 자동 분류
        for e in m.elements.values():
            e.elem_type = ElementType.BEAM
        m.classify_elements()
        columns = m.elements_by_type(ElementType.COLUMN)
        beams = m.elements_by_type(ElementType.BEAM)
        assert len(columns) == 4
        assert len(beams) == 2


# ──────────────────────────────────────────────────────────────
#  바닥 면적
# ──────────────────────────────────────────────────────────────

class TestFloorArea:
    def test_rectangular_floor(self):
        m = StructuralModel()
        # 8m x 6m 사각형
        m.add_node(0, 0, 4, node_id=1, story=1)
        m.add_node(8, 0, 4, node_id=2, story=1)
        m.add_node(8, 6, 4, node_id=3, story=1)
        m.add_node(0, 6, 4, node_id=4, story=1)
        m.story_elevations = [0.0, 4.0]
        area = m.floor_area_at_story(1)
        assert abs(area - 48.0) < 0.1


# ──────────────────────────────────────────────────────────────
#  JSON 직렬화
# ──────────────────────────────────────────────────────────────

class TestSerialization:
    def _sample_model(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1, support="fixed")
        m.add_node(8, 0, 0, node_id=2, support="fixed")
        m.add_node(0, 0, 4, node_id=3)
        m.add_node(8, 0, 4, node_id=4)
        m.add_element(1, 3, elem_type="column", section="H-300x300")
        m.add_element(2, 4, elem_type="column", section="H-300x300")
        m.add_element(3, 4, elem_type="beam", section="H-400x200")
        m.detect_stories()
        m.region = "서울"
        m.importance = "II"
        m.story_usages = {1: "office"}
        return m

    def test_to_json_roundtrip(self):
        m1 = self._sample_model()
        data = m1.to_json()
        m2 = StructuralModel.from_json(data)

        assert len(m2.nodes) == len(m1.nodes)
        assert len(m2.elements) == len(m1.elements)
        assert m2.region == "서울"
        assert m2.nodes[1].support == SupportType.FIXED
        assert m2.elements[1].elem_type == ElementType.COLUMN

    def test_save_load_file(self):
        m1 = self._sample_model()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        m1.save(path)
        m2 = StructuralModel.load(path)
        assert len(m2.nodes) == len(m1.nodes)
        assert len(m2.elements) == len(m1.elements)
        Path(path).unlink()

    def test_json_version(self):
        m = self._sample_model()
        data = m.to_json()
        assert data["version"] == "2.0"


# ──────────────────────────────────────────────────────────────
#  IFC 검증 결과
# ──────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_result(self):
        v = IFCValidationResult(
            is_valid=True,
            extracted_nodes=100,
            extracted_elements=150,
        )
        assert "추출 완료" in v.summary_text()

    def test_critical_issue(self):
        v = IFCValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    code="IFC_NO_COLUMN",
                    message="IfcColumn 요소를 찾을 수 없습니다",
                )
            ],
        )
        assert not v.is_valid
        assert len(v.critical_issues) == 1
        assert "CRITICAL" in v.summary_text()

    def test_warning_with_default(self):
        v = IFCValidationResult(
            is_valid=True,
            extracted_nodes=50,
            extracted_elements=60,
            issues=[
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="IFC_NO_PROFILE",
                    message="23개 보의 단면 정보 없음",
                    element_ids=[1, 2, 3],
                    default_value="H-400x200",
                )
            ],
        )
        text = v.summary_text()
        assert "WARNING" in text
        assert "H-400x200" in text


# ──────────────────────────────────────────────────────────────
#  요약
# ──────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_keys(self):
        m = StructuralModel()
        m.add_node(0, 0, 0, node_id=1, support="fixed")
        m.add_node(0, 0, 4, node_id=2)
        m.add_element(1, 2, elem_type="column")
        m.detect_stories()
        m.region = "서울"

        s = m.summary()
        assert s["num_nodes"] == 2
        assert s["num_elements"] == 1
        assert s["num_columns"] == 1
        assert s["num_beams"] == 0
        assert s["region"] == "서울"
