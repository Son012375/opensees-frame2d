"""2-way slab load distribution tests.

Tests for _panel_beam_contribution() helper and
integration tests verifying total load preservation.
"""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.frame_3d import _panel_beam_contribution


# ============================================================
# Unit tests: _panel_beam_contribution
# ============================================================

class TestPanelBeamContribution:
    """_panel_beam_contribution 헬퍼 단위 테스트."""

    def test_square_panel_equal(self):
        """정방형 패널: beam_x == beam_y (50/50)."""
        w = 10.0
        L = 8.0
        cx = _panel_beam_contribution(w, L, L, "beam_x")
        cy = _panel_beam_contribution(w, L, L, "beam_y")
        assert cx == pytest.approx(cy)
        # w * a / 4 = 10 * 8 / 4 = 20
        assert cx == pytest.approx(20.0)

    def test_square_total_load_preserved(self):
        """정방형 패널: 총 하중 보존."""
        w, Lx, Ly = 10.0, 8.0, 8.0
        cx = _panel_beam_contribution(w, Lx, Ly, "beam_x")
        cy = _panel_beam_contribution(w, Lx, Ly, "beam_y")
        total = 2 * cx * Lx + 2 * cy * Ly
        assert total == pytest.approx(w * Lx * Ly)

    def test_rectangular_2to1(self):
        """2:1 직사각형: 단변 보 < 장변 보."""
        w = 1.0
        Lx, Ly = 6.0, 12.0  # beam_x is short
        cx = _panel_beam_contribution(w, Lx, Ly, "beam_x")
        cy = _panel_beam_contribution(w, Lx, Ly, "beam_y")
        # short: w*a/4 = 1*6/4 = 1.5
        assert cx == pytest.approx(1.5)
        # long: w*a*(2b-a)/(4b) = 1*6*(24-6)/(4*12) = 108/48 = 2.25
        assert cy == pytest.approx(2.25)
        # total preserved
        total = 2 * cx * Lx + 2 * cy * Ly
        assert total == pytest.approx(w * Lx * Ly)

    def test_rectangular_3to1(self):
        """3:1 직사각형: 거의 1-way에 가까움."""
        w = 1.0
        Lx, Ly = 4.0, 12.0
        cx = _panel_beam_contribution(w, Lx, Ly, "beam_x")
        cy = _panel_beam_contribution(w, Lx, Ly, "beam_y")
        # short: 1*4/4 = 1.0
        assert cx == pytest.approx(1.0)
        # long: 1*4*(24-4)/(4*12) = 80/48 ≈ 1.6667
        assert cy == pytest.approx(80.0 / 48.0)
        total = 2 * cx * Lx + 2 * cy * Ly
        assert total == pytest.approx(w * Lx * Ly)

    def test_reversed_direction(self):
        """Lx > Ly: beam_x가 장변, beam_y가 단변."""
        w = 1.0
        Lx, Ly = 10.0, 6.0  # beam_x is long, beam_y is short
        cx = _panel_beam_contribution(w, Lx, Ly, "beam_x")
        cy = _panel_beam_contribution(w, Lx, Ly, "beam_y")
        # beam_y is short (span=6, a=6): w*6/4 = 1.5
        assert cy == pytest.approx(1.5)
        # beam_x is long (span=10, a=6, b=10): 1*6*(20-6)/(4*10) = 84/40 = 2.1
        assert cx == pytest.approx(2.1)
        total = 2 * cx * Lx + 2 * cy * Ly
        assert total == pytest.approx(w * Lx * Ly)

    def test_degenerate_zero(self):
        """패널 치수 0 → 기여 0."""
        assert _panel_beam_contribution(10.0, 0.0, 8.0, "beam_x") == 0.0
        assert _panel_beam_contribution(10.0, 8.0, 0.0, "beam_y") == 0.0
        assert _panel_beam_contribution(10.0, 0.0, 0.0, "beam_x") == 0.0

    def test_negative_dimension(self):
        """음수 치수 → 기여 0."""
        assert _panel_beam_contribution(10.0, -5.0, 8.0, "beam_x") == 0.0

    @pytest.mark.parametrize("Lx,Ly", [
        (3.0, 3.0), (4.0, 6.0), (5.0, 10.0), (8.0, 8.0),
        (6.0, 4.0), (10.0, 5.0), (7.5, 9.2), (3.0, 12.0),
    ])
    def test_total_load_parametric(self, Lx, Ly):
        """다양한 패널 치수에서 총 하중 보존."""
        w = 5.0
        cx = _panel_beam_contribution(w, Lx, Ly, "beam_x")
        cy = _panel_beam_contribution(w, Lx, Ly, "beam_y")
        total = 2 * cx * Lx + 2 * cy * Ly
        assert total == pytest.approx(w * Lx * Ly, rel=1e-10)

    def test_uniform_vs_2way_square(self):
        """정방형에서 50/50과 2-way 결과 동일."""
        w, L = 10.0, 8.0
        # 50/50: w_area_half * trib_y = 0.5 * 10 * (8/2) = 25...
        # no — for single panel edge beam: trib = L/2, so w_line = 0.5 * w * L/2 = w*L/4 = 20
        # 2-way: w*a/4 = 10*8/4 = 20
        cx_2way = _panel_beam_contribution(w, L, L, "beam_x")
        cx_uniform = w * L / 4.0  # 50% * w * (L/2) = w*L/4
        assert cx_2way == pytest.approx(cx_uniform)


# ============================================================
# Integration tests: full analysis with 2-way vs uniform
# ============================================================

class TestSlabDistributionIntegration:
    """2-way 분배 통합 테스트 (OpenSees 해석 포함)."""

    def _run_analysis(self, bays_x, bays_y, w_area, slab_distribution):
        """단순 1층 빌딩 해석 실행, 총 반력 반환."""
        from core.frame_3d import analyze_frame_3d_multi
        load_cases = {
            "DL": [{"type": "floor_area", "story": 1, "value": w_area}],
        }
        multi = analyze_frame_3d_multi(
            stories=[3.5],
            bays_x=bays_x,
            bays_y=bays_y,
            load_cases=load_cases,
            supports="fixed",
            column_section="H-300x300",
            beam_x_section="H-400x200",
            beam_y_section="H-400x200",
            num_elements_per_member=4,
            slab_distribution=slab_distribution,
        )
        return multi

    @staticmethod
    def _total_rz_kN(reactions):
        """반력 리스트에서 RZ 총합 (kN)."""
        return sum(abs(r["RZ_kN"]) for r in reactions)

    def test_total_reaction_square_2way(self):
        """정방형 1bay: 총 반력 = w × 면적."""
        w = 10.0
        bays_x, bays_y = [8.0], [8.0]
        multi = self._run_analysis(bays_x, bays_y, w, "2way")
        cr = multi.case_results["DL"]
        total_rz_kN = self._total_rz_kN(cr.reactions)
        expected = w * 8.0 * 8.0
        assert total_rz_kN == pytest.approx(expected, rel=0.01)

    def test_total_reaction_rectangular_2way(self):
        """직사각형 1bay: 총 반력 = w × 면적 (2-way)."""
        w = 10.0
        bays_x, bays_y = [6.0], [10.0]
        multi = self._run_analysis(bays_x, bays_y, w, "2way")
        cr = multi.case_results["DL"]
        total_rz_kN = self._total_rz_kN(cr.reactions)
        expected = w * 6.0 * 10.0
        assert total_rz_kN == pytest.approx(expected, rel=0.01)

    def test_total_reaction_rectangular_uniform(self):
        """직사각형 1bay: 총 반력 = w × 면적 (uniform)."""
        w = 10.0
        bays_x, bays_y = [6.0], [10.0]
        multi = self._run_analysis(bays_x, bays_y, w, "uniform")
        cr = multi.case_results["DL"]
        total_rz_kN = self._total_rz_kN(cr.reactions)
        expected = w * 6.0 * 10.0
        assert total_rz_kN == pytest.approx(expected, rel=0.01)

    def test_2way_vs_uniform_square_identical(self):
        """정방형: 2-way와 uniform 반력 동일."""
        w = 10.0
        bays_x, bays_y = [8.0], [8.0]
        m_2way = self._run_analysis(bays_x, bays_y, w, "2way")
        m_unif = self._run_analysis(bays_x, bays_y, w, "uniform")
        r_2way = m_2way.case_results["DL"].reactions
        r_unif = m_unif.case_results["DL"].reactions
        for r2, ru in zip(r_2way, r_unif):
            assert r2["RZ_kN"] == pytest.approx(ru["RZ_kN"], abs=0.01)

    def test_2way_vs_uniform_rectangular_differ(self):
        """직사각형 단일 패널: 슬래브 분배 차이는 보 모멘트에 나타난다.

        2-way와 uniform의 차이는 각 보에 배분되는 선하중(→ 보 휨모멘트)에 있다.
        단일 대칭 패널(4코너 대칭)에서는 코너 연직반력이 분배방식과 무관하게
        총하중/4로 동일하다(대칭성). 과거에는 OpenSees `eleLoad` 반력 오귀속 버그로
        인해 코너 반력이 분배방식에 따라 다르게 나왔으나(2026-07-01 등가절점하중
        방식으로 수정), 이는 물리적으로 잘못된 거동이었다. 따라서 이 테스트는
        (a) 총반력 동일, (b) 대칭 코너 반력 동일, (c) 보 휨모멘트는 상이 를 검증한다.
        """
        w = 10.0
        bays_x, bays_y = [6.0], [10.0]
        m_2way = self._run_analysis(bays_x, bays_y, w, "2way")
        m_unif = self._run_analysis(bays_x, bays_y, w, "uniform")
        r_2way = m_2way.case_results["DL"].reactions
        r_unif = m_unif.case_results["DL"].reactions
        # (a) 총합 동일
        total_2way = self._total_rz_kN(r_2way)
        total_unif = self._total_rz_kN(r_unif)
        assert total_2way == pytest.approx(total_unif, rel=0.01)
        # (b) 대칭 단일 패널: 코너 반력은 분배방식과 무관하게 동일 (correct physics)
        for r2, ru in zip(r_2way, r_unif):
            assert r2["RZ_kN"] == pytest.approx(ru["RZ_kN"], abs=0.05)
        # (c) 보 휨모멘트는 분배방식에 따라 달라야 한다 (슬래브 분배의 물리적 효과)
        def _beam_max_My(multi):
            return [
                max(abs(v) for v in mf["My_kNm"])
                for mf in multi.member_forces["DL"]
                if mf["type"] in ("beam_x", "beam_y")
            ]
        b2 = _beam_max_My(m_2way)
        bu = _beam_max_My(m_unif)
        assert max(abs(a - b) for a, b in zip(b2, bu)) > 0.1  # kN·m

    def test_multibay_total_reaction(self):
        """2x2 bay 직사각형: 총 반력 보존."""
        w = 8.0
        bays_x = [6.0, 8.0]
        bays_y = [5.0, 10.0]
        multi = self._run_analysis(bays_x, bays_y, w, "2way")
        cr = multi.case_results["DL"]
        total_rz_kN = self._total_rz_kN(cr.reactions)
        floor_area = sum(bays_x) * sum(bays_y)
        expected = w * floor_area
        assert total_rz_kN == pytest.approx(expected, rel=0.01)

    def test_metadata_contains_slab_distribution(self):
        """analysis_metadata에 slab_distribution 포함."""
        multi = self._run_analysis([8.0], [8.0], 10.0, "2way")
        assert multi.analysis_metadata["slab_distribution"] == "2way"
        multi2 = self._run_analysis([8.0], [8.0], 10.0, "uniform")
        assert multi2.analysis_metadata["slab_distribution"] == "uniform"
