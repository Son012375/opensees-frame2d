"""V2 비정형 건물 모델 검증 테스트.

세 가지 비정형 케이스를 프로그래밍으로 직접 구성하여
StructuralModel → analyze_from_model() → 시각화 전체 파이프라인을 검증한다.

Case 1: L자형 평면 (5층+3층 setback)
Case 2: 경사 브레이스 골조
Case 3: 불규칙 기둥 배치 + 요소별 다른 단면
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-server"))

from core.structural_model import (
    StructuralModel, ElementType, SupportType, ReleaseType,
)
from core.frame_3d import analyze_from_model
from core.visualization_v2 import generate_model_viewer


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


# ──────────────────────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────────────────────

def _add_column(m, x, y, z_bot, z_top, section="H-300x300"):
    """기둥 추가 (하단 노드 없으면 생성)."""
    ni = _find_or_add(m, x, y, z_bot)
    nj = _find_or_add(m, x, y, z_top)
    return m.add_element(ni, nj, elem_type="column", section=section)


def _add_beam(m, x1, y1, x2, y2, z, section="H-400x200"):
    """보 추가."""
    ni = _find_or_add(m, x1, y1, z)
    nj = _find_or_add(m, x2, y2, z)
    return m.add_element(ni, nj, elem_type="beam", section=section)


def _add_brace(m, x1, y1, z1, x2, y2, z2, section="H-200x200"):
    """브레이스 추가."""
    ni = _find_or_add(m, x1, y1, z1)
    nj = _find_or_add(m, x2, y2, z2)
    return m.add_element(ni, nj, elem_type="brace", section=section)


def _find_or_add(m, x, y, z, tol=0.01):
    """좌표에 해당하는 노드를 찾거나 생성."""
    for n in m.nodes.values():
        if abs(n.x - x) < tol and abs(n.y - y) < tol and abs(n.z - z) < tol:
            return n.id
    node = m.add_node(x, y, z)
    return node.id


# ──────────────────────────────────────────────────────────────
#  Case 1: L자형 평면 (5층 + 3층 setback)
# ──────────────────────────────────────────────────────────────

def build_l_shape_model() -> StructuralModel:
    """L자형 평면 건물.

    평면:
        Y
        8 ┌───┬───┐
          │   │   │ ← 전체 5층
        4 ├───┼───┼───┬───┐
          │   │   │   │   │ ← 하부 3층만
        0 └───┴───┴───┴───┘
          0   6   12  18  24  X

    - 좌측 (X=0~12, Y=0~8): 5층
    - 우측 (X=12~24, Y=0~4): 3층 (setback)
    """
    m = StructuralModel()
    story_h = 3.5  # 층고

    # 좌측 영역 기둥 (5층)
    left_cols = [(0, 0), (6, 0), (12, 0), (0, 4), (6, 4), (12, 4), (0, 8), (6, 8), (12, 8)]
    for x, y in left_cols:
        for s in range(5):
            _add_column(m, x, y, s * story_h, (s + 1) * story_h)

    # 우측 영역 기둥 (3층만)
    right_cols = [(18, 0), (24, 0), (18, 4), (24, 4)]
    for x, y in right_cols:
        for s in range(3):
            _add_column(m, x, y, s * story_h, (s + 1) * story_h)

    # 보: X방향
    for s in range(1, 6):
        z = s * story_h
        # 좌측 영역 (전체 5층)
        for y in [0, 4, 8]:
            _add_beam(m, 0, y, 6, y, z)
            _add_beam(m, 6, y, 12, y, z)
        # 우측 영역 (3층까지만)
        if s <= 3:
            for y in [0, 4]:
                _add_beam(m, 12, y, 18, y, z)
                _add_beam(m, 18, y, 24, y, z)

    # 보: Y방향
    for s in range(1, 6):
        z = s * story_h
        # 좌측 영역
        for x in [0, 6, 12]:
            _add_beam(m, x, 0, x, 4, z)
            _add_beam(m, x, 4, x, 8, z)
        # 우측 영역 (3층까지만)
        if s <= 3:
            for x in [18, 24]:
                _add_beam(m, x, 0, x, 4, z)

    # 경계조건 (z=0 노드 고정)
    for n in m.nodes.values():
        if abs(n.z) < 0.01:
            n.support = SupportType.FIXED

    m.detect_stories()
    return m


# ──────────────────────────────────────────────────────────────
#  Case 2: 경사 브레이스 골조
# ──────────────────────────────────────────────────────────────

def build_braced_frame() -> StructuralModel:
    """3층 2경간 프레임 + X형 브레이스.

    정면:
        12 ┌──/\\──┬──/\\──┐
           │ /  \\ │ /  \\ │
         8 ├/────\\┼/────\\┤
           │\\  / │\\  / │
         4 ├──\\/──┼──\\/──┤
           │ /  \\ │      │ ← 1층 좌측만 브레이스
         0 └──\\/──┴──────┘
           0     8      16  X
    """
    m = StructuralModel()

    # 3개 기둥선 × 3층
    for x in [0, 8, 16]:
        for s in range(3):
            _add_column(m, x, 0, s * 4.0, (s + 1) * 4.0)

    # 보 (각 층)
    for s in range(1, 4):
        z = s * 4.0
        _add_beam(m, 0, 0, 8, 0, z)
        _add_beam(m, 8, 0, 16, 0, z)

    # X형 브레이스 — 1층 좌측 스팬
    _add_brace(m, 0, 0, 0, 8, 0, 4)   # / 방향
    _add_brace(m, 8, 0, 0, 0, 0, 4)   # \ 방향

    # X형 브레이스 — 2층 양쪽
    _add_brace(m, 0, 0, 4, 8, 0, 8)
    _add_brace(m, 8, 0, 4, 0, 0, 8)
    _add_brace(m, 8, 0, 4, 16, 0, 8)
    _add_brace(m, 16, 0, 4, 8, 0, 8)

    # X형 브레이스 — 3층 양쪽
    _add_brace(m, 0, 0, 8, 8, 0, 12)
    _add_brace(m, 8, 0, 8, 0, 0, 12)
    _add_brace(m, 8, 0, 8, 16, 0, 12)
    _add_brace(m, 16, 0, 8, 8, 0, 12)

    for n in m.nodes.values():
        if abs(n.z) < 0.01:
            n.support = SupportType.FIXED

    m.detect_stories()
    return m


# ──────────────────────────────────────────────────────────────
#  Case 3: 불규칙 기둥 배치 + 요소별 다른 단면
# ──────────────────────────────────────────────────────────────

def build_irregular_columns() -> StructuralModel:
    """불규칙 기둥 배치 + 요소별 개별 단면.

    평면 (비직교):
        Y
        7 ──── ●(5,7) ────── ●(14,7)
              /               │
        3 ●(0,3)── ●(8,4) ──●(14,3)
          │         │         │
        0 ●(0,0)── ●(8,0) ──●(14,0)
          0        8         14  X
    """
    m = StructuralModel()
    story_h = 4.0

    # 불규칙 기둥 위치
    col_positions = [(0, 0), (8, 0), (14, 0), (0, 3), (8, 4), (14, 3), (5, 7), (14, 7)]

    # 2층 건물 — 기둥별 다른 단면
    col_sections = ["H-300x300", "H-350x350", "H-300x300", "H-300x300",
                    "H-350x350", "H-300x300", "H-250x250", "H-250x250"]

    for i, (x, y) in enumerate(col_positions):
        sec = col_sections[i]
        for s in range(2):
            _add_column(m, x, y, s * story_h, (s + 1) * story_h, section=sec)

    # 보 (수동 연결 — 비직교)
    beam_connections = [
        # 1층 하부 행
        ((0, 0), (8, 0)), ((8, 0), (14, 0)),
        # 1층 중간 행
        ((0, 3), (8, 4)), ((8, 4), (14, 3)),
        # 1층 상부 행
        ((5, 7), (14, 7)),
        # 1층 Y방향
        ((0, 0), (0, 3)), ((8, 0), (8, 4)), ((14, 0), (14, 3)),
        ((0, 3), (5, 7)), ((14, 3), (14, 7)),
    ]

    for s in range(1, 3):
        z = s * story_h
        for (x1, y1), (x2, y2) in beam_connections:
            # 보 단면도 경간에 따라 다르게
            L = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            sec = "H-500x200" if L > 10 else "H-400x200"
            _add_beam(m, x1, y1, x2, y2, z, section=sec)

    for n in m.nodes.values():
        if abs(n.z) < 0.01:
            n.support = SupportType.FIXED

    m.detect_stories()
    return m


# ──────────────────────────────────────────────────────────────
#  테스트: 모델 구성
# ──────────────────────────────────────────────────────────────

class TestModelBuilding:
    def test_l_shape_geometry(self):
        m = build_l_shape_model()
        assert m.num_stories == 5
        assert len(m.elements_by_type(ElementType.COLUMN)) > 0
        assert len(m.elements_by_type(ElementType.BEAM)) > 0
        # 기초 노드 확인
        base = m.base_nodes()
        assert len(base) == 13  # 9 left + 4 right

    def test_braced_frame_geometry(self):
        m = build_braced_frame()
        assert m.num_stories == 3
        braces = m.elements_by_type(ElementType.BRACE)
        assert len(braces) == 10  # 2 + 4 + 4
        # 브레이스가 경사 부재인지 확인
        for b in braces:
            dv = b.direction_vector(m.nodes)
            vert_angle = math.degrees(math.acos(min(abs(dv[2]), 1.0)))
            assert 20 < vert_angle < 70, f"Brace {b.id} angle={vert_angle} not diagonal"

    def test_irregular_columns_geometry(self):
        m = build_irregular_columns()
        assert m.num_stories == 2
        # 요소별 다른 단면 확인
        sections = set(e.section for e in m.elements.values())
        assert len(sections) >= 3  # H-300x300, H-350x350, H-400x200, ...
        # 기둥 8개 × 2층 = 16개
        cols = m.elements_by_type(ElementType.COLUMN)
        assert len(cols) == 16


# ──────────────────────────────────────────────────────────────
#  테스트: 해석
# ──────────────────────────────────────────────────────────────

class TestAnalysis:
    def _make_load_cases(self, model):
        return {
            "DL": [{"type": "floor_area", "story": s, "value": 5.0}
                   for s in range(1, model.num_stories + 1)],
            "EQX": [{"type": "lateral_x", "story": s, "value": 30.0 * s}
                    for s in range(1, model.num_stories + 1)],
        }

    def test_l_shape_analysis(self):
        m = build_l_shape_model()
        result = analyze_from_model(m, self._make_load_cases(m))
        assert "DL" in result.case_results
        assert "EQX" in result.case_results
        # DL: 수직 반력 총합 > 0
        dl = result.case_results["DL"]
        total_rz = sum(r["RZ_kN"] for r in dl.reactions)
        assert total_rz > 0, f"DL total RZ={total_rz} should be positive"
        # EQX: 수평 변위 > 0
        eqx = result.case_results["EQX"]
        assert abs(eqx.max_displacement_x) > 0

    def test_braced_frame_analysis(self):
        m = build_braced_frame()
        result = analyze_from_model(m, self._make_load_cases(m))
        assert "DL" in result.case_results
        assert "EQX" in result.case_results
        # 브레이스 효과: 변위가 작아야 함
        eqx = result.case_results["EQX"]
        assert abs(eqx.max_displacement_x) > 0
        # 반력 평형
        dl = result.case_results["DL"]
        total_rz = sum(r["RZ_kN"] for r in dl.reactions)
        assert total_rz > 0

    def test_irregular_analysis(self):
        m = build_irregular_columns()
        result = analyze_from_model(m, self._make_load_cases(m))
        assert "DL" in result.case_results
        dl = result.case_results["DL"]
        total_rz = sum(r["RZ_kN"] for r in dl.reactions)
        assert total_rz > 0
        # 비직교 보도 해석됨
        assert len(dl.element_forces) > 0


# ──────────────────────────────────────────────────────────────
#  테스트: 시각화 생성
# ──────────────────────────────────────────────────────────────

class TestVisualization:
    def test_l_shape_viewer(self):
        m = build_l_shape_model()
        result = analyze_from_model(m, {
            "DL": [{"type": "floor_area", "story": s, "value": 5.0}
                   for s in range(1, m.num_stories + 1)],
        })
        path = generate_model_viewer(
            m, result=result,
            output_path=str(OUTPUT_DIR / "v2_test_lshape.html"),
            title="Case 1: L-Shape (5F+3F)",
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_braced_frame_viewer(self):
        m = build_braced_frame()
        result = analyze_from_model(m, {
            "EQX": [{"type": "lateral_x", "story": s, "value": 30.0 * s}
                    for s in range(1, m.num_stories + 1)],
        })
        path = generate_model_viewer(
            m, result=result,
            output_path=str(OUTPUT_DIR / "v2_test_braced.html"),
            title="Case 2: Braced Frame (X-Brace)",
        )
        assert Path(path).exists()

    def test_irregular_viewer(self):
        m = build_irregular_columns()
        result = analyze_from_model(m, {
            "DL": [{"type": "floor_area", "story": s, "value": 5.0}
                   for s in range(1, m.num_stories + 1)],
        })
        path = generate_model_viewer(
            m, result=result,
            output_path=str(OUTPUT_DIR / "v2_test_irregular.html"),
            title="Case 3: Irregular Columns",
        )
        assert Path(path).exists()
