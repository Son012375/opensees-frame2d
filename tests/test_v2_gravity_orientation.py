"""V2 중력하중 정정 + 방향-강건화 회귀 테스트.

배경 (2026-07-01):
    OpenSees `eleLoad -beamUniform`은 모멘트접합 보에서 분포하중 고정단력을
    i-end로 편향 오귀속한다(대칭구조+대칭중력에 42% 비대칭해). V1 경로는 등가절점
    하중(_apply_beam_udl)+요소력 q0복원(_localforce_with_q0)으로 정정됐고, 이 테스트는
    **V2 경로**(analyze_from_model)에도 동일 정정이 적용됐는지 검증한다.

핵심 검증 3가지:
    (1) 대칭성 — 대칭 사각패널의 4코너 연직반력이 동일해야 한다(버그 시 비대칭).
    (2) 정렬-불변성 — 보 요소를 i→j = -X/-Y로 뒤집어 생성해도 반력/변위가 동일해야
        한다. V1 격자는 항상 +X/+Y라 미노출이지만 V2(IFC/웹 에디터)는 임의 정렬이므로
        _apply_beam_udl의 전역-모멘트 부호가 좌표차로 결정돼야 한다(하드코딩 시 실패).
    (3) 총하중 보존 — Σ연직반력 = w_area × 바닥면적.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-server"))

from core.structural_model import StructuralModel, SupportType
from core.frame_3d import analyze_from_model


STORY_H = 3.5
COL_SEC = "H-300x300"
BEAM_SEC = "H-400x200"


def _build_square_panel(a: float = 8.0, reverse_beams: bool = False) -> StructuralModel:
    """1층 단일 정사각 패널(a×a). 4 기둥(고정) + 4 상단보.

    reverse_beams=True면 보를 i→j = -X/-Y 방향으로 생성(노드순서 뒤집기)하여
    방향-강건화를 시험한다. 기하/하중은 동일하므로 결과가 동일해야 정상.
    """
    m = StructuralModel()
    corners = [(0.0, 0.0), (a, 0.0), (a, a), (0.0, a)]

    # 노드: 각 코너의 base(z=0)와 top(z=story_h)
    base_ids = {}
    top_ids = {}
    for (x, y) in corners:
        base_ids[(x, y)] = m.add_node(x, y, 0.0, support=SupportType.FIXED).id
        top_ids[(x, y)] = m.add_node(x, y, STORY_H).id

    # 기둥 4개
    for (x, y) in corners:
        m.add_element(base_ids[(x, y)], top_ids[(x, y)],
                      elem_type="column", section=COL_SEC)

    # 상단보 4개 (패널 둘레). 정방향: 좌→우, 하→상.
    edges = [
        ((0.0, 0.0), (a, 0.0)),   # beam_x (y=0)
        ((0.0, a), (a, a)),       # beam_x (y=a)
        ((0.0, 0.0), (0.0, a)),   # beam_y (x=0)
        ((a, 0.0), (a, a)),       # beam_y (x=a)
    ]
    for (p_i, p_j) in edges:
        if reverse_beams:
            p_i, p_j = p_j, p_i   # i→j 를 -X/-Y 로 뒤집기
        m.add_element(top_ids[p_i], top_ids[p_j],
                      elem_type="beam", section=BEAM_SEC)

    m.detect_stories()
    return m


def _analyze(model, w_area=10.0):
    """floor_area 중력 해석 후 반력·보모멘트·횡변위 요약을 반환.

    Returns dict:
        rz     : {(x,y): RZ_kN}                        코너 연직반력
        beamMy : sorted [max|My_kNm| per 보 부재]      방향-불변 물리량(정렬로 i/j무관)
        latx   : max |dx_mm|,  laty: max |dy_mm|       중력하 기생 횡변위
    """
    result = analyze_from_model(
        model, {"DL": [{"type": "floor_area", "story": 1, "value": w_area}]}
    )
    dl = result.case_results["DL"]
    coord = {n.id: (round(n.x, 3), round(n.y, 3)) for n in model.nodes.values()}
    beam_my = sorted(
        round(max(abs(v) for v in ef.get("My_kNm", [0.0])), 4)
        for ef in result.member_forces["DL"]
        if ef["type"] in ("beam_x", "beam_y")
    )
    latx = max(abs(d["dx_mm"]) for d in dl.nodal_displacements)
    laty = max(abs(d["dy_mm"]) for d in dl.nodal_displacements)
    return {
        "rz": {coord[r["node"]]: r["RZ_kN"] for r in dl.reactions},
        "beamMy": beam_my,
        "latx": round(latx, 5),
        "laty": round(laty, 5),
    }


def _reactions_rz(model, w_area=10.0):
    return _analyze(model, w_area)["rz"]


class TestV2GravitySymmetry:
    def test_square_panel_corner_symmetry(self):
        """대칭 사각패널: 4코너 연직반력 동일(중력버그 정정 검증)."""
        a, w = 8.0, 10.0
        rz = _reactions_rz(_build_square_panel(a), w)
        vals = list(rz.values())
        assert len(vals) == 4
        avg = sum(vals) / 4
        # 4코너 대칭 → 각 코너가 총하중의 1/4을 담당, 상호 편차 <1%
        for v in vals:
            assert v == pytest.approx(avg, rel=0.01), f"corner asymmetry: {rz}"
        # 총하중 보존
        assert sum(vals) == pytest.approx(w * a * a, rel=0.01)
        # 코너당 ≈ w·a²/4
        assert avg == pytest.approx(w * a * a / 4, rel=0.02)

    def test_beam_orientation_invariance(self):
        """보 요소를 -X/-Y로 뒤집어도 결과가 동일해야 한다(방향-강건화 검증).

        하드코딩된 +m0@i / -m0@j 라면 뒤집힌 보에서 등가 고정단모멘트 부호가 반대로
        적용된다. 대칭 사각패널의 **연직반력**은 이 오류에 둔감하지만(모멘트→기둥휨,
        축력 무관), **보 내부모멘트·중력하 기생 횡변위**는 오염된다(웹 에디터에서
        -X/-Y로 그린 보의 설계력이 틀리는 실제 버그). 좌표차 부호 결정 시에만 일치.
        [경험 확인 2026-07-01: 구 하드코딩은 reversed에서 beamMy +3~6%, latx 변동.]
        """
        a, w = 8.0, 10.0
        fwd = _analyze(_build_square_panel(a, reverse_beams=False), w)
        rev = _analyze(_build_square_panel(a, reverse_beams=True), w)
        # 연직반력
        assert set(fwd["rz"].keys()) == set(rev["rz"].keys())
        for k in fwd["rz"]:
            assert fwd["rz"][k] == pytest.approx(rev["rz"][k], rel=1e-6, abs=1e-4)
        # 보 내부모멘트(민감 물리량) — 방향 뒤집어도 동일해야 함
        assert fwd["beamMy"] == pytest.approx(rev["beamMy"], rel=1e-5, abs=1e-3), (
            f"orientation-dependent beam moments: fwd={fwd['beamMy']} rev={rev['beamMy']}"
        )
        # 중력하 기생 횡변위 — 방향 불변
        assert fwd["latx"] == pytest.approx(rev["latx"], rel=1e-4, abs=1e-4)
        assert fwd["laty"] == pytest.approx(rev["laty"], rel=1e-4, abs=1e-4)

    def test_reversed_panel_still_symmetric(self):
        """뒤집힌 보 모델도 4코너 대칭이 유지돼야 한다."""
        a, w = 8.0, 10.0
        rz = _reactions_rz(_build_square_panel(a, reverse_beams=True), w)
        vals = list(rz.values())
        avg = sum(vals) / 4
        for v in vals:
            assert v == pytest.approx(avg, rel=0.01), f"reversed asymmetry: {rz}"
        assert sum(vals) == pytest.approx(w * a * a, rel=0.01)
