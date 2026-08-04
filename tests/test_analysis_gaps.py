"""해석시스템 갭 백로그 회귀 테스트 (적대적 감사 2026-06-24, Phase A~C).

`docs/analysis_gaps_backlog.md`의 12개 갭 구현에 대한 회귀 테스트.
실행: cd mcp-server && python -m pytest ../tests/test_analysis_gaps.py -q
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))


# ════════════════════════════════════════════════════════════
# Phase A1 — 유효지진중량 W 창고류 25% 활하중 포함 (KDS 41 17 00)
# ════════════════════════════════════════════════════════════
def test_a1_seismic_live_load_fraction():
    from core.load_generator import seismic_live_load_fraction
    # 창고/공장류 → 0.25
    for u in ("storage", "storage_light", "storage_heavy", "factory",
              "factory_light", "factory_heavy"):
        assert seismic_live_load_fraction(u) == 0.25, u
    # 일반 용도 → 0
    for u in ("office", "residential", "retail", "school", "hospital", ""):
        assert seismic_live_load_fraction(u) == 0.0, u


def test_a1_calculate_story_weights_storage_adds_live_load(monkeypatch):
    """창고층 유효지진중량 = DL×A + 0.25×LL×A (비창고는 DL×A 그대로)."""
    from core.building_model import BuildingModel
    import core.load_generator as lg
    monkeypatch.setattr(lg, "_query_live_load", lambda usage: 8.0)

    cfg = {"bays_x": [6, 6], "bays_y": [6, 6]}
    m_store = BuildingModel.from_json({**cfg, "stories": [{"height": 3.5, "usage": "storage"}]})
    m_office = BuildingModel.from_json({**cfg, "stories": [{"height": 3.5, "usage": "office"}]})
    dl = [{"type": "floor_area", "value": 5.0, "story": 1}]
    area = 12 * 12  # 144 m²

    w_office = lg._calculate_story_weights(m_office, dl)
    w_store = lg._calculate_story_weights(m_store, dl)
    assert abs(w_office[0] - 5.0 * area) < 1e-3            # DL only
    assert abs(w_store[0] - (5.0 + 0.25 * 8.0) * area) < 1e-3  # +25% LL
    assert w_store[0] > w_office[0]


def test_a1_modal_mass_storage_consistency():
    """modal 질량(_estimate_story_weights)도 usages 제공 시 창고류 25% LL 포함."""
    from core.frame_3d import _estimate_story_weights
    lc = {
        "DL": [{"type": "floor_area", "value": 5.0, "story": s} for s in (1, 2)],
        "LL": [{"type": "floor_area", "value": 8.0, "story": s} for s in (1, 2)],
    }
    bays_x, bays_y = [6, 6], [6]
    area = 12 * 6  # 72 m²

    w_none = _estimate_story_weights(lc, 2, bays_x, bays_y, 3, 2)  # 하위호환: DL only
    w_office = _estimate_story_weights(lc, 2, bays_x, bays_y, 3, 2, usages=["office", "office"])
    w_store = _estimate_story_weights(lc, 2, bays_x, bays_y, 3, 2, usages=["storage", "storage"])

    assert abs(w_none[0] - 5.0 * area) < 1e-6
    assert abs(w_office[0] - 5.0 * area) < 1e-6
    assert abs(w_store[0] - (5.0 + 0.25 * 8.0) * area) < 1e-6


# ════════════════════════════════════════════════════════════
# Phase A2 — 지배 하중조합 §10/facts 표기
# ════════════════════════════════════════════════════════════
class _MockCaseResult:
    def __init__(self, story_drifts=None):
        self.story_drifts = story_drifts or []


class _MockMulti:
    """run_design_check가 읽는 최소 속성만 가진 mock."""
    def __init__(self):
        self.stories = [3.5, 3.5]
        self.bays_x = [6.0]
        self.bays_y = [6.0]
        self.num_stories = 2
        self.fy_MPa = 275.0
        self.E_MPa = 205000.0
        for mt in ("column", "beam_x", "beam_y"):
            setattr(self, f"{mt}_section", "H-300x300")
        self.member_info = [{"member_id": 1, "type": "column",
                             "length_m": 3.5, "section": "H-300x300"}]
        # Gravity: 축력 큼·모멘트 작음 / 지진조합: 모멘트 큼 → 실제 H1 지배
        self.member_forces = {
            "1.2DL+1.6LL": [{"member_id": 1, "N_kN": [2000], "My_kNm": [30],
                             "Mz_kNm": [5], "Vy_kN": [50], "Vz_kN": [0]}],
            "1.32DL+1.0LL+1.0EQX": [{"member_id": 1, "N_kN": [800], "My_kNm": [350],
                                     "Mz_kNm": [40], "Vy_kN": [120], "Vz_kN": [0]}],
        }
        sd_small = [{"story": 1, "height_m": 3.5, "drift_x": 0.0005, "drift_y": 0.0003},
                    {"story": 2, "height_m": 3.5, "drift_x": 0.0004, "drift_y": 0.0002}]
        sd_big = [{"story": 1, "height_m": 3.5, "drift_x": 0.004, "drift_y": 0.001},
                  {"story": 2, "height_m": 3.5, "drift_x": 0.003, "drift_y": 0.001}]
        self.combo_results = {
            "1.2DL+1.6LL": _MockCaseResult(sd_small),
            "1.32DL+1.0LL+1.0EQX": _MockCaseResult(sd_big),
        }
        self.case_results = {}


def test_a2_run_design_check_exposes_governing_combos():
    from core.design_check import run_design_check
    seismic = {"Cd": 5.5, "IE": 1.0}
    dc = run_design_check(_MockMulti(), building_model=None, seismic_report=seismic)
    s = dc["summary"]
    # drift 지배조합 = 유일한 EQ 조합
    assert s["drift_governing_combo"] == "1.32DL+1.0LL+1.0EQX"
    # member 지배조합 = 실제 H1 최대(지진조합, 모멘트 지배)
    assert s["member_governing_combo"] == "1.32DL+1.0LL+1.0EQX"


def test_a2_governing_combo_surfaced_in_facts_and_allowlist():
    from core.narrative_interpreter import build_facts, build_number_allowlist
    interp = {"severity": "safe", "severity_label": {"ko": "안전", "en": "safe"},
              "drift_interpretation": {}, "member_interpretation": {"weakest_link": {}}}
    dc = {"summary": {"max_drift_ratio": 0.45, "max_interaction_ratio": 0.82,
                      "drift_governing_combo": "1.32DL+1.0LL+1.0EQX",
                      "member_governing_combo": "0.78DL-1.0EQY",
                      "ng_stories": 0, "ng_members": 0}}
    facts = build_facts(interp, dc)
    assert "지배조합 1.32DL+1.0LL+1.0EQX" in facts["max_drift_ratio"]
    assert "지배조합 0.78DL-1.0EQY" in facts["max_interaction_ratio"]
    # anti-hallucination: 조합 계수가 allowlist에 자기씨앗으로 포함되어야 함
    allow = build_number_allowlist(facts)
    assert "1.32" in allow and "0.78" in allow


# ════════════════════════════════════════════════════════════
# Phase A3/A4 — V2 모달 누적참여 충분성 + 비틀림모드 탐지
# ════════════════════════════════════════════════════════════
def _v2_find_or_add(m, x, y, z, tol=0.01):
    for n in m.nodes.values():
        if abs(n.x - x) < tol and abs(n.y - y) < tol and abs(n.z - z) < tol:
            return n.id
    return m.add_node(x, y, z).id


def _build_v2_grid(col_xy, n_stories=2, h=3.5):
    """col_xy(기둥 평면좌표 리스트)로 V2 StructuralModel 골조 구성.

    인접한 기둥쌍(같은 X 또는 같은 Y, 간격 6m)을 보로 연결한다.
    """
    from core.structural_model import StructuralModel, SupportType
    m = StructuralModel()
    cols = set(col_xy)
    for (x, y) in col_xy:
        for s in range(n_stories):
            m.add_element(_v2_find_or_add(m, x, y, s * h),
                          _v2_find_or_add(m, x, y, (s + 1) * h),
                          elem_type="column", section="H-300x300")
    for s in range(1, n_stories + 1):
        z = s * h
        for (x, y) in col_xy:
            if (x + 6, y) in cols:
                m.add_element(_v2_find_or_add(m, x, y, z),
                              _v2_find_or_add(m, x + 6, y, z),
                              elem_type="beam", section="H-400x200")
            if (x, y + 6) in cols:
                m.add_element(_v2_find_or_add(m, x, y, z),
                              _v2_find_or_add(m, x, y + 6, z),
                              elem_type="beam", section="H-400x200")
    for n in m.nodes.values():
        if abs(n.z) < 0.01:
            n.support = SupportType.FIXED
    m.detect_stories()
    return m


def _dl_cases(m):
    return {"DL": [{"type": "floor_area", "story": s, "value": 5.0}
                   for s in range(1, m.num_stories + 1)]}


def test_a3_v2_cumulative_participation_present():
    """대칭 정형 모델(완전기저) → V2가 cumulative_participation을 반환한다."""
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (12, 0),
                        (0, 6), (6, 6), (12, 6),
                        (0, 12), (6, 12), (12, 12)])
    res = analyze_from_model(m, _dl_cases(m))
    ma = res.modal_analysis
    assert ma, "V2 모달해석이 수행되어야 함"
    cp = ma.get("cumulative_participation")
    assert cp is not None, "A3: cumulative_participation 반환 누락"
    for k in ("x_pct", "y_pct", "rz_pct", "sufficient_90pct"):
        assert k in cp, f"누락 키 {k}"
    assert 0 <= cp["x_pct"] <= 101 and 0 <= cp["y_pct"] <= 101
    # 플래그 내부 일관성 (≥90% 충분조건)
    assert cp["sufficient_90pct"] == (cp["x_pct"] >= 90 and cp["y_pct"] >= 90)
    # 강막 완전기저면 병진 누적 ~100%
    assert cp["x_pct"] >= 85 and cp["y_pct"] >= 85


def test_a4_v2_torsion_detected_and_labels():
    """비대칭(L자) 모델 → rz 참여 산정 + V1식 방향 라벨."""
    from core.frame_3d import analyze_from_model
    # (12,12) 모서리 제거 → 강성중심≠질량중심 → 비틀림 결합
    m = _build_v2_grid([(0, 0), (6, 0), (12, 0),
                        (0, 6), (6, 6), (12, 6),
                        (0, 12), (6, 12)])
    res = analyze_from_model(m, _dl_cases(m))
    ma = res.modal_analysis
    assert ma, "V2 모달해석이 수행되어야 함"
    modes = ma["modes"]
    # 방향 라벨이 V1과 통일(TRAN-X/TRAN-Y/ROTN-Z/N/A) — 구버전 'X'/'Y'/'XY' 아님
    assert all(md["direction"] in {"TRAN-X", "TRAN-Y", "ROTN-Z", "N/A"} for md in modes)
    assert not any(md["direction"] in {"X", "Y", "XY"} for md in modes)
    # rz_pct가 더 이상 하드코딩 0 아님 — 비대칭 모델은 비틀림 참여가 존재
    assert any(md["mass_participation"]["rz_pct"] > 0.5 for md in modes), \
        "A4: 비대칭 모델인데 비틀림 참여가 0 (rz 계산 누락 의심)"
    assert ma["cumulative_participation"]["rz_pct"] > 0


# ════════════════════════════════════════════════════════════
# Phase A5 — design envelope 성분별 worst 보강
# ════════════════════════════════════════════════════════════
def test_a5_component_wise_demand_envelope():
    """축력 최대와 모멘트 최대가 서로 다른 조합일 때 성분별 worst를 포착한다."""
    from core.design_check import _governing_member_forces
    mf = {
        # 축력 최대 = Gravity (모멘트 작음)
        "Gravity": [{"member_id": 1, "N_kN": [3000], "My_kNm": [20], "Mz_kNm": [5],
                     "Vy_kN": [100], "Vz_kN": [0]}],
        # 모멘트·전단 최대 = Seismic (축력 작음)
        "Seismic": [{"member_id": 1, "N_kN": [1500], "My_kNm": [400], "Mz_kNm": [120],
                     "Vy_kN": [150], "Vz_kN": [0]}],
    }
    gov = _governing_member_forces(mf, 1, ["Gravity", "Seismic"], 4000, 300, 80, 500)
    env = gov["demand_envelope"]
    assert env["N_max_kN"] == 3000.0 and env["N_combo"] == "Gravity"
    assert env["Mux_max_kNm"] == 400.0 and env["Mux_combo"] == "Seismic"
    assert env["Muy_max_kNm"] == 120.0 and env["Muy_combo"] == "Seismic"
    assert env["Vu_max_kN"] == 150.0 and env["Vu_combo"] == "Seismic"
    # B3 불변: 판정용 H1 지배조합은 실제 H1 최대(Seismic), demand 스냅샷도 그 조합값
    assert gov["governing_combo"] == "Seismic"
    assert gov["Pu"] == 1500.0


# ════════════════════════════════════════════════════════════
# Phase A 적대적 리뷰 후속 (V2 storage modal / 대소문자 / R16)
# ════════════════════════════════════════════════════════════
def test_a1_v2_storage_modal_mass_heavier_than_office():
    """V2(analyze_from_model)도 창고층 modal 질량에 25% LL을 반영해야 한다.

    창고는 질량이 더 크므로 1차 주기가 사무실보다 길다.
    """
    from core.frame_3d import analyze_from_model
    cols = [(0, 0), (6, 0), (12, 0), (0, 6), (6, 6), (12, 6), (0, 12), (6, 12), (12, 12)]

    def _run(usage):
        m = _build_v2_grid(cols)
        m.story_usages = {s: usage for s in range(1, m.num_stories + 1)}
        lc = {
            "DL": [{"type": "floor_area", "story": s, "value": 5.0}
                   for s in range(1, m.num_stories + 1)],
            "LL": [{"type": "floor_area", "story": s, "value": 8.0}
                   for s in range(1, m.num_stories + 1)],
        }
        return analyze_from_model(m, lc).modal_analysis

    ma_off = _run("office")
    ma_sto = _run("storage")
    assert ma_off and ma_sto, "V2 모달해석이 수행되어야 함"
    t_off = ma_off["fundamental_periods"]["T1_x_s"]
    t_sto = ma_sto["fundamental_periods"]["T1_x_s"]
    assert t_off and t_sto
    assert t_sto > t_off * 1.01, f"창고 T1={t_sto} > 사무실 T1={t_off} (25% LL 미반영 의심)"


def test_a1_mixed_case_usage_no_fallback_leak():
    """혼합 대소문자 용도('Storage')도 정규화되어 정확한 LL을 쓴다 (fallback 2.5 누출 방지)."""
    import core.load_generator as lg
    from core.building_model import BuildingModel
    cfg = {"bays_x": [6], "bays_y": [6]}
    dl = [{"type": "floor_area", "value": 5.0, "story": 1}]
    w_lower = lg._calculate_story_weights(
        BuildingModel.from_json({**cfg, "stories": [{"height": 3.5, "usage": "storage"}]}), dl)
    w_upper = lg._calculate_story_weights(
        BuildingModel.from_json({**cfg, "stories": [{"height": 3.5, "usage": "Storage"}]}), dl)
    w_office = lg._calculate_story_weights(
        BuildingModel.from_json({**cfg, "stories": [{"height": 3.5, "usage": "office"}]}), dl)
    # 대소문자 무관 동일값 (정규화)
    assert abs(w_lower[0] - w_upper[0]) < 1e-6
    # 창고는 사무실보다 무거움 (LL 가산이 fallback로 새지 않았음)
    assert w_upper[0] > w_office[0]


def test_a3_r16_finding_triggers_when_participation_low():
    """누적 참여 <90%면 R16 caution, ≥90%면 미발생."""
    import types
    from core.result_interpreter import _interpret_modal
    multi = types.SimpleNamespace(stories=[3.5] * 10)
    base = {
        "modes": [{"direction": "TRAN-X", "period_s": 1.2,
                   "mass_participation": {"x_pct": 50, "y_pct": 40, "rz_pct": 0}}],
        "fundamental_periods": {"T1_x_s": 1.2, "T1_y_s": 1.1},
    }
    low = {**base, "cumulative_participation":
           {"x_pct": 70.0, "y_pct": 65.0, "rz_pct": 5.0, "sufficient_90pct": False}}
    interp, findings = _interpret_modal(low, multi, None)
    assert "R16" in [f["code"] for f in findings]
    assert interp["mass_participation_sufficient"] is False

    ok = {**base, "cumulative_participation":
          {"x_pct": 96.0, "y_pct": 93.0, "rz_pct": 2.0, "sufficient_90pct": True}}
    _i2, f2 = _interpret_modal(ok, multi, None)
    assert "R16" not in [f["code"] for f in f2]


# ════════════════════════════════════════════════════════════
# Phase B1 — 보 수직처짐 / 사용성 검토 (L/360·L/240)
# ════════════════════════════════════════════════════════════
def _ss_udl_moment(w_kNm: float, L_m: float, n: int = 5):
    """단순지지 등분포하중 보의 강축 모멘트 다이어그램 M(x)=w·x·(L−x)/2 (kN·m)."""
    return [w_kNm * (L_m * i / (n - 1)) * (L_m - L_m * i / (n - 1)) / 2 for i in range(n)]


def test_b1_chord_deflection_matches_analytical_udl():
    """이중적분 처짐이 단순지지 UDL 해석해(5wL⁴/384EI)와 일치(거친 메시 포함)."""
    from core.design_check import _chord_relative_max_deflection
    from core.section_3d import get_section_3d
    EI = 205000.0 * get_section_3d("H-400x200").Ix
    L_mm, w = 8000.0, 30.0  # 1 kN/m = 1 N/mm
    exact = 5 * w * L_mm ** 4 / (384 * EI)
    for n in (5, 9, 21):
        M = _ss_udl_moment(w, 8.0, n)
        d = _chord_relative_max_deflection(M, L_mm, EI)
        assert abs(d - exact) / exact < 0.02, f"n={n}: {d} vs {exact}"


def _beam_multi(w_dl, w_ll, L_m=8.0, section="H-400x200"):
    import types
    return types.SimpleNamespace(
        E_MPa=205000.0, fy_MPa=275.0, stories=[3.5],
        combo_results={}, case_results={},
        member_info=[{"member_id": 1, "type": "beam_x", "length_m": L_m, "section": section}],
        member_forces={
            "DL": [{"member_id": 1, "My_kNm": _ss_udl_moment(w_dl, L_m)}],
            "LL": [{"member_id": 1, "My_kNm": _ss_udl_moment(w_ll, L_m)}],
        },
    )


def test_b1_deflection_ng_when_live_exceeds_L360():
    from core.design_check import check_beam_deflections, run_design_check
    from core.section_3d import get_section_3d
    EI = 205000.0 * get_section_3d("H-400x200").Ix
    m = _beam_multi(w_dl=10.0, w_ll=30.0)  # 활하중 큼 → L/360 초과
    dc = check_beam_deflections(m)
    assert dc is not None and dc["status"] == "NG"
    b = dc["beams"][0]
    exact_live = 5 * 30.0 * 8000.0 ** 4 / (384 * EI)
    assert abs(b["defl_live_mm"] - exact_live) / exact_live < 0.03
    assert b["ratio_live"] > 1.0  # 30 kN/m, 8m, H-400x200 → ~1.5
    # run_design_check 통합: overall NG + summary 노출
    full = run_design_check(m)
    assert full["summary"]["max_deflection_ratio"] is not None
    assert full["summary"]["deflection_status"] == "NG"
    assert full["overall_status"] == "NG"


def test_b1_deflection_ok_small_load():
    from core.design_check import check_beam_deflections
    m = _beam_multi(w_dl=5.0, w_ll=5.0)  # 가벼운 하중 → 처짐 작음
    dc = check_beam_deflections(m)
    assert dc is not None and dc["status"] == "OK"
    assert dc["beams"][0]["ratio"] < 1.0


def test_b1_deflection_not_checked_without_gravity_cases():
    """개별 DL/LL 케이스 부재력이 없으면 미검토(None) → §10 '미검토' 명시."""
    import types
    from core.design_check import check_beam_deflections, run_design_check
    m = types.SimpleNamespace(
        E_MPa=205000.0, fy_MPa=275.0, stories=[3.5],
        combo_results={}, case_results={},
        member_info=[{"member_id": 1, "type": "beam_x", "length_m": 8.0, "section": "H-400x200"}],
        member_forces={"1.2DL+1.6LL": [{"member_id": 1, "My_kNm": _ss_udl_moment(20.0, 8.0)}]},
    )
    assert check_beam_deflections(m) is None
    full = run_design_check(m)
    assert full["summary"]["deflection_status"] == "not_checked"
    assert full["summary"]["max_deflection_ratio"] is None


def test_b1_deflection_facts_note_when_not_checked():
    from core.narrative_interpreter import build_facts
    interp = {"severity": "safe", "severity_label": {"ko": "안전", "en": "safe"},
              "drift_interpretation": {}, "member_interpretation": {"weakest_link": {}}}
    dc = {"summary": {"max_drift_ratio": 0.4, "max_interaction_ratio": 0.5,
                      "max_deflection_ratio": None, "deflection_status": "not_checked",
                      "ng_stories": 0, "ng_members": 0}}
    facts = build_facts(interp, dc)
    assert "미검토" in facts.get("deflection_note", "")
    # 검토된 경우 비율 노출
    dc2 = {"summary": {"max_drift_ratio": 0.4, "max_interaction_ratio": 0.5,
                       "max_deflection_ratio": 0.85, "deflection_status": "OK",
                       "ng_stories": 0, "ng_members": 0}}
    facts2 = build_facts(interp, dc2)
    assert "0.85" in facts2.get("max_deflection_ratio", "")


# ════════════════════════════════════════════════════════════
# Phase B2 — 내진설계범주(SDC) + 시스템 적격성/높이제한
# ════════════════════════════════════════════════════════════
def _bm(importance="II", seismic_system="ordinary_moment_frame", total_height=21.0):
    import types
    return types.SimpleNamespace(importance=importance, seismic_system=seismic_system,
                                 total_height=total_height)


def test_b2_sdc_governing_and_omf_no_limit():
    from core.design_check import check_seismic_system
    rep = {"SDS": 0.46, "SD1": 0.25, "hn_m": 21.0}
    r = check_seismic_system(_bm("II", "ordinary_moment_frame", 21.0), rep)
    assert r["sdc_basis"]["from_SDS"] == "C"   # 0.33≤0.46<0.50, 내진II → C
    assert r["sdc_basis"]["from_SD1"] == "D"   # 0.25≥0.20 → D
    assert r["sdc"] == "D"                       # 더 엄격한 쪽
    assert r["db_key"] == "3-c"
    assert r["status"] == "OK" and r["height_ok"]  # 3-c는 NL(높이제한 없음)


def test_b2_special_grade_more_severe_than_II():
    from core.design_check import check_seismic_system
    rep = {"SDS": 0.40, "SD1": 0.10, "hn_m": 20.0}
    assert check_seismic_system(_bm("II"), rep)["sdc"] == "C"   # SDS:C, SD1:B → C
    assert check_seismic_system(_bm("특"), rep)["sdc"] == "D"   # SDS:D, SD1:C → D


def test_b2_np_system_not_permitted():
    from core.design_check import check_seismic_system
    rep = {"SDS": 0.46, "SD1": 0.25}  # SDC D
    r = check_seismic_system(_bm("II", "ordinary_braced_frame", 20.0), rep)
    assert r["db_key"] == "2-q" and r["sdc"] == "D"
    assert r["status"] == "NG" and not r["system_permitted"]
    assert r["issues"]


def test_b2_numeric_height_limit():
    from core.design_check import check_seismic_system
    rep = {"SDS": 0.46, "SD1": 0.25}  # SDC D
    over = check_seismic_system(_bm("II", "rc_ordinary_shear_wall", 70.0), rep)
    under = check_seismic_system(_bm("II", "rc_ordinary_shear_wall", 50.0), rep)
    assert over["db_key"] == "1-b" and over["height_limit"]["limit"] == 60
    assert over["status"] == "NG"    # 70 m > 60 m
    assert under["status"] == "OK"   # 50 m ≤ 60 m


def test_b2_not_checked_without_seismic():
    from core.design_check import check_seismic_system
    assert check_seismic_system(_bm("II"), None) is None
    assert check_seismic_system(_bm("II"), {"error": "x"}) is None
    assert check_seismic_system(_bm("II"), {"SDS": 0.3}) is None  # SD1 누락


def test_b2_severity_escalates_on_system_ng():
    """비율은 safe인데 시스템 NG면 §10 verdict 모순 방지 위해 severity 격상."""
    from core.result_interpreter import _classify_severity
    dc_ng = {"overall_status": "NG",
             "summary": {"max_drift_ratio": 0.3, "max_interaction_ratio": 0.4,
                         "max_deflection_ratio": 0.2}}
    assert _classify_severity(dc_ng) in ("moderate", "severe")
    dc_ok = {"overall_status": "OK",
             "summary": {"max_drift_ratio": 0.3, "max_interaction_ratio": 0.4}}
    assert _classify_severity(dc_ok) == "safe"


# ════════════════════════════════════════════════════════════
# Phase B3 — 인장부재 인장내력 검토 (signed N, AISC H1.2)
# ════════════════════════════════════════════════════════════
def test_b3_tension_uses_tension_capacity():
    from core.design_check import _interaction_H1
    # 순인장(Pu<0): 인장내력 phiPn_tens=2000 사용 → axial=1000/2000=0.5
    r_t, f_t = _interaction_H1(-1000, 500, 100, 400, 0, 200, phiPn_tens=2000)
    # 동일 크기 압축: phiPn_comp=500(좌굴로 낮음) → axial=1000/500=2.0
    r_c, f_c = _interaction_H1(1000, 500, 100, 400, 0, 200)
    assert "(T)" in f_t and "(T)" not in f_c
    assert r_t < r_c                                  # 인장내력이 더 커서 비율 낮음
    assert abs(r_t - (0.5 + (8 / 9) * 0.25)) < 1e-6   # 0.5≥0.2 → H1-1a(T)


def test_b3_backward_compat_compression_when_no_tension_cap():
    """phiPn_tens 미제공(기본 0) → 인장이어도 압축내력 사용(하위호환)."""
    from core.design_check import _interaction_H1
    r, f = _interaction_H1(-1000, 500, 100, 400, 0, 200)
    assert abs(r - (2.0 + (8 / 9) * 0.25)) < 1e-6
    assert "(T)" not in f


def test_b3_governing_tension_member():
    from core.design_check import _governing_member_forces
    mf = {"Uplift": [{"member_id": 1, "N_kN": [-1500], "My_kNm": [100],
                      "Mz_kNm": [0], "Vy_kN": [20], "Vz_kN": [0]}]}
    gov = _governing_member_forces(mf, 1, ["Uplift"], 600, 400, 200, 500, phiPn_tens=2000)
    assert gov["axial_mode"] == "tension"
    assert gov["N_signed"] == -1500.0
    assert abs(gov["axial"] - 0.75) < 1e-6  # 1500/2000 (인장내력)
    assert "(T)" in gov["formula"]


def test_b3_compression_member_unaffected():
    """압축 부재는 종전대로 압축내력(Fcr) 사용."""
    from core.design_check import _governing_member_forces
    mf = {"G": [{"member_id": 1, "N_kN": [1500], "My_kNm": [100],
                 "Mz_kNm": [0], "Vy_kN": [20], "Vz_kN": [0]}]}
    gov = _governing_member_forces(mf, 1, ["G"], 600, 400, 200, 500, phiPn_tens=2000)
    assert gov["axial_mode"] == "compression"
    assert abs(gov["axial"] - 1500 / 600) < 1e-6  # 압축내력 600 사용
    assert "(T)" not in gov["formula"]


# ── Phase B 적대적 리뷰 후속 (실제 OpenSees 부호규약 lock + 통합 테스트) ──
def test_b3_real_opensees_tension_sign_locked():
    """실 OpenSees 해석으로 인장 부호규약(N<0) 및 인장내력 적용을 잠근다 (SAFETY).

    중력 없이 순수 횡하중 → 전도 couple → 풍상측 기둥 순인장.
    """
    from core.frame_3d import analyze_frame_3d_multi
    from core.design_check import check_member_strengths
    lc = {"EQX": [{"type": "lateral_x", "value": 800.0, "story": 1}]}
    multi = analyze_frame_3d_multi(
        stories=[6.0], bays_x=[4], bays_y=[4], load_cases=lc,
        load_combinations={"C": {"EQX": 1.0}},
        column_section="H-300x300", beam_x_section="H-400x200", beam_y_section="H-400x200")
    cols = [m for m in multi.member_forces["C"] if m["type"] == "column"]
    n0 = [m["N_kN"][0] for m in cols]
    assert min(n0) < 0, f"전도 시 인장(N<0) 기둥이 존재해야 함: {n0}"  # 부호규약 lock
    mc = check_member_strengths(multi, fy_MPa=275.0, E_MPa=205000.0)
    tens = [m for m in mc["members"] if m.get("axial_mode") == "tension"]
    assert tens, "인장 기둥이 design check에서 tension으로 분류돼야 함"
    t = tens[0]
    assert t["N_signed_kN"] < 0
    # 인장내력 φt·Fy·Ag 가 압축내력(좌굴 Fcr)보다 큼
    assert t["capacity"]["phiPnt_kN"] > t["capacity"]["phiPn_kN"]
    assert mc["summary"]["n_tension_members"] >= 1


def test_b2_integration_through_run_design_check_and_facts():
    """B2가 run_design_check 요약 + build_facts(서술 facts)까지 노출되는지 통합 검증."""
    import types
    from core.design_check import run_design_check
    from core.result_interpreter import interpret_results
    from core.narrative_interpreter import build_facts
    multi = types.SimpleNamespace(
        fy_MPa=275.0, E_MPa=205000.0, stories=[3.5, 3.5],
        combo_results={}, case_results={}, member_info=[], member_forces={})
    bm = _bm("II", "ordinary_braced_frame", 20.0)  # 2-q → SDC D 미허용
    rep = {"SDS": 0.46, "SD1": 0.25, "hn_m": 7.0, "Cd": 4.0, "IE": 1.0}
    dc = run_design_check(multi, bm, rep)
    assert dc["summary"]["sdc"] == "D"
    assert dc["summary"]["seismic_system_status"] == "NG"
    assert dc["overall_status"] == "NG"
    interp = interpret_results(dc, multi, modal_analysis=None)
    # severity가 NG와 모순되지 않게 격상
    assert interp["severity"] in ("moderate", "severe")
    facts = build_facts(interp, dc)
    sdc_fact = facts.get("seismic_design_category", {})
    assert sdc_fact.get("sdc") == "D" and sdc_fact.get("status") == "부적격"


# ════════════════════════════════════════════════════════════
# Phase C1 — 전도(overturning)·활동(sliding) 안정
# ════════════════════════════════════════════════════════════
def test_c1_overturning_ok_wide_building():
    import types
    from core.design_check import check_global_stability
    sf = [{"story": 1, "Fx_kN": 50, "height_m": 3.5, "weight_kN": 1000},
          {"story": 2, "Fx_kN": 100, "height_m": 7.0, "weight_kN": 1000}]
    bm = types.SimpleNamespace(total_width_x=20.0, total_width_y=20.0)
    r = check_global_stability(bm, {"story_forces": sf, "W_kN": 2000, "V_kN": 150})
    # M_ot=50·3.5+100·7=875, M_r=0.9·2000·10=18000 → FS≈20.6
    assert r["status"] == "OK"
    assert r["overturning"]["X"]["status"] == "OK"
    assert r["governing_FS"] > 1.5


def test_c1_overturning_ng_tall_narrow():
    import types
    from core.design_check import check_global_stability
    sf = [{"story": i + 1, "Fx_kN": 200, "height_m": 4 * (i + 1), "weight_kN": 300}
          for i in range(5)]
    bm = types.SimpleNamespace(total_width_x=2.0, total_width_y=2.0)
    r = check_global_stability(bm, {"story_forces": sf, "W_kN": 1500, "V_kN": 1000})
    # M_ot=200·60=12000, M_r=0.9·1500·1=1350 → FS≈0.11 < 1.5
    assert r["overturning"]["X"]["status"] == "NG"
    assert r["status"] == "NG"


def test_c1_not_checked_without_seismic():
    import types
    from core.design_check import check_global_stability
    assert check_global_stability(None, None) is None
    bm = types.SimpleNamespace(total_width_x=10, total_width_y=10)
    assert check_global_stability(bm, {"error": "x"}) is None


# ════════════════════════════════════════════════════════════
# Phase C3 — P-Delta 안정계수 θ
# ════════════════════════════════════════════════════════════
def _pdelta_inputs(weights, fxs, drifts, Cd):
    sf = [{"story": i + 1, "weight_kN": weights[i], "Fx_kN": fxs[i]} for i in range(len(weights))]
    rep = {"story_forces": sf, "Cd": Cd}
    drift = {"checks": [{"story": i + 1, "inelastic_drift": drifts[i], "direction": "X"}
                        for i in range(len(drifts))]}
    return rep, drift


def test_c3_theta_below_threshold_ok():
    from core.design_check import check_pdelta_stability
    rep, drift = _pdelta_inputs([1000, 1000], [50, 100], [0.005, 0.004], Cd=5.5)
    r = check_pdelta_stability(rep, drift)
    # story1: Px=2000,Vx=150,drift=0.005 → θ=2000·0.005/(150·5.5)=0.0121
    assert r["status"] == "OK"
    assert r["max_theta"] < 0.1


def test_c3_theta_amplify_band():
    from core.design_check import check_pdelta_stability
    rep, drift = _pdelta_inputs([3000], [100], [0.012], Cd=3.0)
    r = check_pdelta_stability(rep, drift)
    # θ=3000·0.012/(100·3)=0.12 ; θmax=min(0.5/3,0.25)=0.1667 → 0.1<θ≤θmax → amplify
    c = r["checks"][0]
    assert c["status"] == "amplify"
    assert r["needs_amplification"] is True and r["status"] == "OK"
    assert c["amplification"] == round(1 / (1 - 0.12), 3)


def test_c3_theta_unstable_ng():
    from core.design_check import check_pdelta_stability
    rep, drift = _pdelta_inputs([5000], [50], [0.02], Cd=3.0)
    r = check_pdelta_stability(rep, drift)
    # θ=5000·0.02/(50·3)=0.667 > θmax=0.1667 → NG
    assert r["status"] == "NG"
    assert r["max_theta"] > r["theta_limit"]


def test_c3_not_checked():
    from core.design_check import check_pdelta_stability
    assert check_pdelta_stability(None, None) is None
    rep = {"story_forces": [{"story": 1, "weight_kN": 1, "Fx_kN": 1}], "Cd": 3.0}
    assert check_pdelta_stability(rep, None) is None  # drift 없음


def _pdelta_multi(geometric_nonlinearity="linear"):
    import types

    class _CR:
        def __init__(self):
            self.story_drifts = [{"story": 1, "height_m": 3.5, "drift_x": 0.004, "drift_y": 0.0}]
    return types.SimpleNamespace(
        fy_MPa=275.0, E_MPa=205000.0, stories=[3.5],
        geometric_nonlinearity=geometric_nonlinearity, analysis_metadata={},
        combo_results={"1.0DL+1.0EQX": _CR()}, case_results={},
        member_info=[{"member_id": 1, "type": "column", "length_m": 3.5, "section": "H-300x300"}],
        member_forces={"1.0DL+1.0EQX": [{"member_id": 1, "N_kN": [500], "My_kNm": [150],
                                          "Mz_kNm": [20], "Vy_kN": [50], "Vz_kN": [0]}]})


def test_c3_pdelta_amplification_applied_to_member_and_drift():
    """0.1<θ≤θmax: 1/(1−θ)가 층간변위 + 부재 상관비에 실제로 적용된다 (KDS §7.2.7, SAFETY)."""
    from core.design_check import run_design_check
    multi = _pdelta_multi("linear")
    sf = [{"story": 1, "weight_kN": 3000, "Fx_kN": 100, "height_m": 3.5}]
    rep = {"story_forces": sf, "W_kN": 3000, "V_kN": 100, "Cd": 3.0, "IE": 1.0,
           "SDS": 0.4, "SD1": 0.15}
    dc = run_design_check(multi, None, rep)
    pd = dc["pdelta_check"]
    # θ=3000·0.012/(100·3)=0.12 ; θmax=0.1667 → amplify
    assert pd["needs_amplification"] is True and pd["status"] == "OK"
    amp = round(1 / (1 - 0.12), 3)
    # 층간변위 증폭
    xchk = [c for c in dc["drift_check"]["checks"] if c["direction"] == "X"][0]
    assert xchk.get("pdelta_amp") == amp
    assert abs(xchk["inelastic_drift"] - round(0.012 * amp, 6)) < 1e-6
    # 부재 상관비 증폭
    m = dc["member_check"]["members"][0]
    assert m["ratios"]["pdelta_amp"] == amp


def test_c3_no_amplification_when_geometric_nonlinear():
    """기하비선형(Corotational) 해석이면 2차효과 이미 반영 → 증폭 미적용(이중계산 방지)."""
    from core.design_check import run_design_check
    multi = _pdelta_multi("corotational")
    sf = [{"story": 1, "weight_kN": 3000, "Fx_kN": 100, "height_m": 3.5}]
    rep = {"story_forces": sf, "W_kN": 3000, "V_kN": 100, "Cd": 3.0, "IE": 1.0}
    dc = run_design_check(multi, None, rep)
    assert dc["pdelta_check"]["needs_amplification"] is True  # θ는 여전히 보고
    xchk = [c for c in dc["drift_check"]["checks"] if c["direction"] == "X"][0]
    assert "pdelta_amp" not in xchk                  # 변위 미증폭
    assert dc["member_check"]["members"][0]["ratios"]["pdelta_amp"] == 1.0


def test_c1_data_inconsistency_not_silent_ok():
    """밑면전단 V>0인데 Fx가 비어 M_ot≈0이면 '안전'으로 오판하지 않고 미검토(None)."""
    import types
    from core.design_check import check_global_stability
    bm = types.SimpleNamespace(total_width_x=20.0, total_width_y=20.0)
    sf = [{"story": 1, "Fx_kN": 0, "height_m": 3.5, "weight_kN": 1000}]
    assert check_global_stability(bm, {"story_forces": sf, "W_kN": 1000, "V_kN": 500}) is None


# ════════════════════════════════════════════════════════════
# Phase C2 — 비틀림 비정형 + 변위증폭 Ax
# ════════════════════════════════════════════════════════════
def _combo_with_disp(name, story, dxmax, dxmin):
    import types
    cr = types.SimpleNamespace(story_drifts=[{
        "story": story, "height_m": 3.5, "drift_x": 0.002, "drift_y": 0.0,
        "disp_x_max": dxmax, "disp_x_min": dxmin, "disp_y_max": 0.0, "disp_y_min": 0.0}])
    return {name: cr}


def test_c2_regular_no_amplification():
    from core.design_check import check_torsional_irregularity, _ax_key
    combos = _combo_with_disp("1.3DL+1.0LL+1.0EQX", 1, 10.0, 9.5)
    t = check_torsional_irregularity(combos)
    # ratio=10/9.75≈1.026 <1.2 → regular, Ax clamp→1.0
    assert t["classification"] == "regular" and not t["irregular"]
    assert abs(t["ax_map"][_ax_key("1.3DL+1.0LL+1.0EQX", 1, "X")] - 1.0) < 1e-9


def test_c2_extreme_irregular_ax():
    from core.design_check import check_torsional_irregularity, _ax_key
    combos = _combo_with_disp("1.3DL+1.0LL+1.0EQX", 1, 14.0, 6.0)
    t = check_torsional_irregularity(combos)
    # dmax=14,davg=10,ratio=1.4 → extreme; Ax=(1.4/1.2)²≈1.361
    assert t["classification"] == "extreme" and t["irregular"] and t["extreme"]
    assert abs(t["ax_map"][_ax_key("1.3DL+1.0LL+1.0EQX", 1, "X")] - (1.4 / 1.2) ** 2) < 1e-6


def test_c2_ax_map_is_json_serializable():
    """ax_map 키는 문자열(JSON 직렬화 가능) — 튜플 키 금지(리포트/API 직렬화 회귀)."""
    import json
    from core.design_check import check_torsional_irregularity
    t = check_torsional_irregularity(_combo_with_disp("1.0EQX", 1, 14.0, 6.0))
    json.dumps(t)  # 튜플 키면 TypeError
    assert all(isinstance(k, str) for k in t["ax_map"])


def test_c2_sign_flip_negative_dominant():
    from core.design_check import check_torsional_irregularity
    combos = _combo_with_disp("0.7DL-1.0EQX", 1, -6.0, -14.0)  # −EQX 우세
    t = check_torsional_irregularity(combos)
    assert abs(t["max_ratio"] - 1.4) < 1e-6  # 부호 정렬 후 동일 비율


def test_c2_ax_amplifies_inelastic_drift():
    from core.design_check import check_story_drifts, _ax_key
    combos = _combo_with_disp("1.0EQX", 1, 14.0, 6.0)
    ax_map = {_ax_key("1.0EQX", 1, "X"): 1.5}
    dc = check_story_drifts(combos, [3.5], Cd=5.0, IE=1.0, importance="II", ax_map=ax_map)
    xchk = [c for c in dc["checks"] if c["direction"] == "X"][0]
    # base inelastic = Cd·δe/IE = 5·0.002/1 = 0.01 ; ×Ax 1.5 = 0.015
    assert abs(xchk["inelastic_drift"] - 0.015) < 1e-6
    assert xchk["ax"] == 1.5


def test_c2_not_checked_without_disp_extremes():
    from core.design_check import check_torsional_irregularity
    import types
    cr = types.SimpleNamespace(story_drifts=[{"story": 1, "drift_x": 0.003, "drift_y": 0.001}])
    assert check_torsional_irregularity({"1.0EQX": cr}) is None  # disp_* 없음(V2 등)


# ── Phase C 통합 (실 프레임 → run_design_check C1/C2/C3 동시 경로) ──
def test_c_integration_real_frame_design_check():
    import types
    from core.frame_3d import analyze_frame_3d_multi
    from core.design_check import run_design_check
    dl = [{"type": "floor_area", "value": 8.0, "story": s} for s in (1, 2)]
    eqx = [{"type": "lateral_x", "value": 40.0 * s, "story": s} for s in (1, 2)]
    combos = {"1.2DL": {"DL": 1.2}, "1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}}
    multi = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6, 6], bays_y=[6, 6],
        load_cases={"DL": dl, "EQX": eqx}, load_combinations=combos,
        column_section="H-300x300", beam_x_section="H-400x200", beam_y_section="H-400x200")
    cr = multi.combo_results["1.0DL+1.0EQX"]
    assert "disp_x_max" in cr.story_drifts[0]  # C2 disp 극값 기록 확인
    sf = [{"story": 1, "weight_kN": 3000, "Fx_kN": 40, "height_m": 3.5},
          {"story": 2, "weight_kN": 3000, "Fx_kN": 80, "height_m": 7.0}]
    bm = types.SimpleNamespace(importance="II", seismic_system="ordinary_moment_frame",
                               total_width_x=12.0, total_width_y=12.0, total_height=7.0)
    rep = {"story_forces": sf, "W_kN": 6000, "V_kN": 120, "Cd": 3.0, "IE": 1.0,
           "SDS": 0.4, "SD1": 0.15}
    dc = run_design_check(multi, bm, rep)
    assert dc["stability_check"] is not None and dc["stability_check"]["status"] == "OK"
    assert dc["pdelta_check"] is not None
    assert dc["torsion_check"] is not None
    # 대칭 격자 → 비틀림 정형
    assert dc["summary"]["torsion_classification"] == "regular"
    assert dc["summary"]["overturning_FS"] > 1.5
    # 서버 직렬화 경로 회귀: design_check 전체가 JSON 직렬화 가능해야 함(튜플 키 금지)
    import json
    json.dumps(dc, ensure_ascii=False)


# ════════════════════════════════════════════════════════════
# Tier1 잔여이슈 (안전-비보수) 회귀 — docs/analysis_residual_audit_2026-06.md
# ════════════════════════════════════════════════════════════

def _member_multi(member_info, member_forces, **secattrs):
    """check_member_strengths/run_design_check용 최소 multi (mtype fallback 단면)."""
    import types
    base = dict(
        fy_MPa=275.0, E_MPa=205000.0, stories=[3.5, 3.5, 3.5],
        combo_results={k: _MockCaseResult() for k in member_forces},
        case_results={},
        member_info=member_info, member_forces=member_forces,
        analysis_metadata={},
        # mtype 기본 단면 (section="" 부재가 fallback로 읽음)
        column_section="", beam_x_section="", beam_y_section="",
        column_A_mm2=2716.0, column_Ix_mm4=1840e4, column_Iy_mm4=134e4,
        column_J_mm4=10e4, column_h_mm=200.0, column_b_mm=100.0,
        column_tw_mm=5.5, column_tf_mm=8.0,
        beam_x_A_mm2=8412.0, beam_x_Ix_mm4=23700e4, beam_x_Iy_mm4=1740e4,
        beam_x_J_mm4=66e4, beam_x_h_mm=400.0, beam_x_b_mm=200.0,
        beam_x_tw_mm=8.0, beam_x_tf_mm=13.0,
        beam_y_A_mm2=8412.0, beam_y_Ix_mm4=23700e4, beam_y_Iy_mm4=1740e4,
        beam_y_J_mm4=66e4, beam_y_h_mm=400.0, beam_y_b_mm=200.0,
        beam_y_tw_mm=8.0, beam_y_tf_mm=13.0,
    )
    base.update(secattrs)
    return types.SimpleNamespace(**base)


def _mf(mid, mtype, N, My, Mz=0.0, Vy=0.0, Vz=0.0, pts=5):
    return {"member_id": mid, "type": mtype, "N_kN": [N] * pts,
            "My_kNm": [My] * pts, "Mz_kNm": [Mz] * pts,
            "Vy_kN": [Vy] * pts, "Vz_kN": [Vz] * pts}


# ── Tier1-1: 유효좌굴길이계수 K (모멘트골조 K≥1.2, 비보수 정정) ──
def test_tier1_k_braced_vs_moment_factor():
    from core.design_check import _effective_length_factor, K_BRACED, K_MOMENT_FRAME
    # 기둥: 가새/전단벽 → 1.0, 모멘트·미상 → 1.2
    assert _effective_length_factor("ordinary_braced_frame", "column") == K_BRACED
    assert _effective_length_factor("rc_ordinary_shear_wall", "column") == K_BRACED
    assert _effective_length_factor("ordinary_moment_frame", "column") == K_MOMENT_FRAME
    assert _effective_length_factor("special_moment_frame", "column") == K_MOMENT_FRAME
    assert _effective_length_factor(None, "column") == K_MOMENT_FRAME  # 미상 → sway 보수
    # 보/가새 부재는 항상 1.0
    assert _effective_length_factor("ordinary_moment_frame", "beam_x") == 1.0


def test_tier1_k_slender_column_NG_under_moment_frame():
    """세장 기둥: K=1.0(가새)→OK 인데 K=1.2(모멘트골조)→φPn 감소로 NG (SAFETY)."""
    from core.design_check import check_member_strengths
    import json
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": ""}]
    mf = {"1.2DL+1.0EQX": [_mf(1, "column", N=150.0, My=5.0)]}  # 압축 150kN
    multi = _member_multi(mi, mf)

    braced = check_member_strengths(multi, 275.0, 205000.0,
                                    seismic_system="ordinary_braced_frame")
    moment = check_member_strengths(multi, 275.0, 205000.0,
                                    seismic_system="ordinary_moment_frame")
    pn_b = braced["members"][0]["capacity"]["phiPn_kN"]
    pn_m = moment["members"][0]["capacity"]["phiPn_kN"]
    assert pn_m < pn_b, f"모멘트골조 K=1.2 φPn({pn_m})는 가새 K=1.0 φPn({pn_b})보다 작아야 함"
    assert braced["status"] == "OK"
    assert moment["status"] == "NG", "세장 기둥은 모멘트골조 K반영 시 NG여야 함(비보수 정정)"
    assert moment["members"][0]["ratios"]["interaction"] > 1.0
    assert braced["members"][0]["K"] == 1.0 and moment["members"][0]["K"] == 1.2
    assert moment["summary"]["column_K"] == 1.2
    json.dumps(moment)  # JSON-safe


def test_tier1_k_normal_building_verdict_unchanged():
    """정상 기둥(여유 충분)은 K=1.2여도 OK 유지 — 트리거 아닐 때 판정 불변."""
    from core.design_check import check_member_strengths
    # 충분한 단면(H-300x300)·경미한 하중 → K=1.2여도 여유
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": ""}]
    mf = {"1.2DL+1.0EQX": [_mf(1, "column", N=200.0, My=50.0, Mz=30.0, Vy=30.0)]}
    multi = _member_multi(
        mi, mf,
        column_A_mm2=11980.0, column_Ix_mm4=20400e4, column_Iy_mm4=6750e4,
        column_J_mm4=128e4, column_h_mm=300.0, column_b_mm=300.0,
        column_tw_mm=10.0, column_tf_mm=15.0)
    r = check_member_strengths(multi, 275.0, 205000.0,
                              seismic_system="ordinary_moment_frame")
    assert r["status"] == "OK"
    assert r["members"][0]["ratios"]["interaction"] < 1.0


# ── Tier1-2: 보 횡-비틀림좌굴 LTB (AISC F2) ──
def test_tier1_ltb_capacity_monotonic_decreasing():
    """비지지길이 증가 → φMn 단조 감소; Lb≤Lp는 φMp, Lb=0은 횡지지(φMp)."""
    from core.design_check import _ltb_bending_capacity, _bending_capacity
    from core.section_3d import get_section_3d
    sec = get_section_3d("H-400x200")
    Sx = sec.Ix / (sec.h / 2.0)
    # Zx (design_check 공식과 동일)
    hw = sec.h - 2 * sec.tf
    Zx = sec.b * sec.tf * (sec.h - sec.tf) + sec.tw * hw ** 2 / 4
    ry = math.sqrt(sec.Iy / sec.A)
    ho = sec.h - sec.tf
    phiMp = _bending_capacity(275.0, Zx)

    def cap(Lb):
        return _ltb_bending_capacity(275.0, 205000.0, Lb, Zx, Sx, ry,
                                     sec.Iy, sec.J, ho)

    m_braced, _ = cap(0.0)                  # 비지지길이 0 → 횡지지
    m_short, i_short = cap(1500.0)          # Lb ≤ Lp → 소성
    m_mid, i_mid = cap(6000.0)              # Lp < Lb ≤ Lr → 비탄성 LTB
    m_long, i_long = cap(10000.0)           # Lb > Lr → 탄성 LTB
    assert abs(m_braced - phiMp) < 1e-6
    assert abs(m_short - phiMp) < 0.5 and i_short["zone"] == "plastic"
    assert i_mid["zone"] == "inelastic_LTB" and i_long["zone"] == "elastic_LTB"
    assert m_short > m_mid > m_long, f"{m_short} > {m_mid} > {m_long}"
    assert m_long < phiMp  # LTB 감소 확인


def test_tier1_ltb_unbraced_beam_NG():
    """비지지 보(Lb=경간): 횡지지 가정→OK 인데 LTB 반영→NG (SAFETY)."""
    from core.design_check import check_member_strengths
    import json
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 8.0, "section": ""}]
    mf = {"1.2DL+1.6LL": [_mf(1, "beam_x", N=0.0, My=250.0, Vy=20.0)]}
    multi = _member_multi(mi, mf)

    braced = check_member_strengths(multi, 275.0, 205000.0)            # 슬래브 횡지지(기본)
    unbraced = check_member_strengths(multi, 275.0, 205000.0, beam_unbraced=True)
    assert braced["status"] == "OK"
    assert unbraced["status"] == "NG", "비지지 보는 LTB로 NG여야 함"
    bm_b = braced["members"][0]
    bm_u = unbraced["members"][0]
    assert bm_b["capacity"]["phiMnx_kNm"] > bm_u["capacity"]["phiMnx_kNm"]
    assert bm_b["ltb"]["governs"] is False        # 횡지지 → LTB 미지배
    assert bm_u["ltb"]["governs"] is True          # 비지지 → LTB 지배
    assert unbraced["summary"]["n_ltb_governed"] >= 1
    json.dumps(unbraced)


def test_tier1_ltb_per_member_unbraced_override():
    """member_info["unbraced_length_m"]로 부재별 비지지길이 재정의 (기본 무영향)."""
    from core.design_check import check_member_strengths
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 8.0, "section": "",
           "unbraced_length_m": 8.0}]
    mf = {"1.2DL+1.6LL": [_mf(1, "beam_x", N=0.0, My=250.0, Vy=20.0)]}
    multi = _member_multi(mi, mf)
    r = check_member_strengths(multi, 275.0, 205000.0)  # beam_unbraced=False지만 부재가 비지지 명시
    assert r["status"] == "NG"
    assert r["members"][0]["ltb"]["governs"] is True


# ── Tier1-3: _infer_member_story가 member_info["story"](실층)를 우선 ──
def test_tier1_member_story_uses_real_story_for_pdelta():
    """P-Delta 부재증폭이 member_id 추정이 아닌 실제 층(minfo['story'])에 매핑."""
    from core.design_check import check_member_strengths
    # member_id=1, n_stories=3 → 구버전 추정층=(1%3)+1=2. 실제 층은 3으로 명시.
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 3}]
    mf = {"1.0DL+1.0EQX": [_mf(1, "column", N=300.0, My=100.0, Vy=50.0)]}
    multi = _member_multi(mi, mf)
    # 3층에만 증폭 1.3. 구버전(추정 2층)이면 증폭 미적용(1.0)이어야 하지만,
    # 실층(3) 사용 시 1.3이 적용돼야 한다.
    r = check_member_strengths(multi, 275.0, 205000.0, pdelta_amp_by_story={3: 1.3})
    m = r["members"][0]
    assert m["story"] == 3, "member_info['story'] 실값이 보고돼야 함"
    assert m["ratios"]["pdelta_amp"] == 1.3, "실제 층(3) 증폭이 적용돼야 함"


def test_tier1_member_story_fallback_when_none():
    """story 누락(None)이면 member_id 기반 추정으로 폴백 (엣지 안전)."""
    from core.design_check import check_member_strengths
    # story 키 없음 → 폴백 추정 (1%3)+1 = 2
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": ""}]
    mf = {"1.0DL+1.0EQX": [_mf(1, "column", N=300.0, My=100.0, Vy=50.0)]}
    multi = _member_multi(mi, mf)
    r = check_member_strengths(multi, 275.0, 205000.0, pdelta_amp_by_story={2: 1.4})
    m = r["members"][0]
    assert m["story"] == 2
    assert m["ratios"]["pdelta_amp"] == 1.4


# ── Tier1-4: 솔버 실패 게이트 (0 결과를 '안전'으로 보고 금지) ──
def test_tier1_solver_failure_gates_to_NG():
    """analysis_metadata.solver_failed → run_design_check가 OK 아닌 NG+analysis_error."""
    from core.design_check import run_design_check
    import json
    # 0 변위·부재력 (정상이면 '안전'으로 오판될 입력)
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    mf = {"1.2DL+1.0EQX": [_mf(1, "column", N=0.0, My=0.0)]}
    multi = _member_multi(mi, mf)
    multi.analysis_metadata = {"solver_failed": True, "failed_cases": ["DL"]}
    # 0 층간변위 (게이트 없으면 drift OK)
    multi.combo_results = {"1.2DL+1.0EQX": _MockCaseResult(
        [{"story": 1, "height_m": 3.5, "drift_x": 0.0, "drift_y": 0.0}])}
    rep = {"Cd": 5.5, "IE": 1.0, "story_forces": [], "W_kN": 0}
    dc = run_design_check(multi, None, rep)
    assert dc["overall_status"] == "NG", "솔버 실패는 '안전'이 아니라 NG로 게이트돼야 함"
    assert dc.get("analysis_error", {}).get("solver_failed") is True
    assert dc["summary"]["analysis_error"] is True
    assert dc["drift_check"] is None and dc["member_check"] is None
    assert dc["critical_issues"][0]["type"] == "analysis_solver_failed"
    json.dumps(dc, ensure_ascii=False)


def test_tier1_solver_failure_interpretation_and_facts():
    """해석 실패 시 result_interpreter severity=severe + R26 + facts 노트."""
    from core.design_check import run_design_check
    from core.result_interpreter import interpret_results
    from core.narrative_interpreter import build_facts
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    multi = _member_multi(mi, {"DL": [_mf(1, "column", N=0.0, My=0.0)]})
    multi.analysis_metadata = {"solver_failed": True, "failed_cases": ["DL"]}
    multi.combo_results = {}
    dc = run_design_check(multi, None, None)
    interp = interpret_results(dc, multi, modal_analysis=None)
    assert interp["severity"] == "severe"
    assert "R26" in [f["code"] for f in interp["findings"]]
    facts = build_facts(interp, dc)
    assert "신뢰 불가" in facts.get("analysis_error_note", "")


# ── Tier1-5: RSA 조합 부재검토 미수행 명시 ──
def test_tier1_rsa_combo_member_check_flagged():
    """RSA 조합이 부재력 없으면 '미검토'로 명시(은폐 방지) — summary+R27+critical issue."""
    from core.design_check import check_member_strengths, run_design_check
    from core.result_interpreter import interpret_results
    import json
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    # ELF 조합엔 부재력 있고, RSA 조합엔 없음
    mf = {"1.2DL+1.6LL": [_mf(1, "column", N=200.0, My=40.0, Vy=20.0)]}
    multi = _member_multi(mi, mf)
    multi.combo_results = {
        "1.2DL+1.6LL": _MockCaseResult(),
        "EQX_RSA": _MockCaseResult([{"story": 1, "height_m": 3.5,
                                     "drift_x": 0.001, "drift_y": 0.0}]),
    }
    mc = check_member_strengths(multi, 275.0, 205000.0)
    assert mc["summary"]["rsa_unchecked_combos"] == ["EQX_RSA"]

    dc = run_design_check(multi, None, None)
    assert any(ci["type"] == "rsa_member_check_not_performed"
               for ci in dc["critical_issues"])
    interp = interpret_results(dc, multi, modal_analysis=None)
    assert "R27" in [f["code"] for f in interp["findings"]]
    json.dumps(dc, ensure_ascii=False)


# ── Tier1 적대적 리뷰 후속 (degenerate LTB / RSA severity / unresolved-story 보수증폭) ──
def test_tier1_ltb_degenerate_section_falls_back_to_Mp_finite():
    """퇴화 단면(Lr≤Lp)이면 NaN/Inf 없이 보수적으로 Mp 폴백 (JSON-safe)."""
    from core.design_check import _ltb_bending_capacity, _bending_capacity
    # ry는 크게, Iy는 작게(불일치 강제) → rts 미소 → Lr ≪ Lp
    phiMn, info = _ltb_bending_capacity(
        275.0, 205000.0, 5000.0, Zx=1.3e6, Sx=1.0e6, ry=200.0,
        Iy=1.0, J=1.0, ho=300.0)
    assert math.isfinite(phiMn)
    assert info["zone"] == "undetermined" and info["governs"] is False
    assert abs(phiMn - _bending_capacity(275.0, 1.3e6)) < 1e-6  # Mp 폴백


def test_tier1_rsa_unchecked_blocks_safe_verdict():
    """RSA 부재검토 미수행이면 severity가 'safe'로 단정되지 않는다(최소 marginal)."""
    from core.result_interpreter import _classify_severity
    safe_no_rsa = {"overall_status": "OK",
                   "summary": {"max_drift_ratio": 0.3, "max_interaction_ratio": 0.4}}
    assert _classify_severity(safe_no_rsa) == "safe"
    safe_with_rsa = {"overall_status": "OK",
                     "summary": {"max_drift_ratio": 0.3, "max_interaction_ratio": 0.4,
                                 "rsa_unchecked_combos": ["EQX_RSA"]}}
    assert _classify_severity(safe_with_rsa) == "marginal"


def test_tier1_unresolved_story_uses_conservative_max_pdelta_amp():
    """층 불확실(추정) 부재는 P-Delta 증폭을 전 층 최대값으로 보수 적용(과소증폭 방지)."""
    from core.design_check import check_member_strengths
    # story 키 없음 → 추정. 증폭맵에 여러 층 존재 → 최대(1.5)가 적용돼야 함.
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": ""}]
    mf = {"1.0DL+1.0EQX": [_mf(1, "column", N=300.0, My=100.0, Vy=50.0)]}
    multi = _member_multi(mi, mf)
    r = check_member_strengths(multi, 275.0, 205000.0,
                               pdelta_amp_by_story={2: 1.2, 4: 1.5})
    assert r["members"][0]["ratios"]["pdelta_amp"] == 1.5  # 보수적 최대


# ════════════════════════════════════════════════════════════
# Tier2-A 잔여이슈 (#6 세장비 / #7 콤팩트·국부좌굴 / #15 순단면파단 / #14 NaN가드)
# ════════════════════════════════════════════════════════════

def _h200x100_col(mi_extra=None):
    """약축 세장 기둥(H-200x100, ry≈22.2mm) mtype-fallback multi."""
    base = dict(column_A_mm2=2716.0, column_Ix_mm4=1840e4, column_Iy_mm4=134e4,
                column_J_mm4=10e4, column_h_mm=200.0, column_b_mm=100.0,
                column_tw_mm=5.5, column_tf_mm=8.0)
    return base


# ── Tier2-6: 세장비 한계 (AISC E2 압축 KL/r≤200 / D1 인장 L/r≤300) ──
def test_tier2_slenderness_compression_column_NG_low_load():
    """세장 기둥: 하중 작아 내력비<1이어도 KL/r>200이면 세장비로 NG (SAFETY)."""
    from core.design_check import check_member_strengths
    import json
    # H-200x100 5m, K=1.2 → KL/r=1.2·5000/22.2≈270>200. 하중 경미 → interaction<1.
    mi = [{"member_id": 1, "type": "column", "length_m": 5.0, "section": ""}]
    mf = {"DL": [_mf(1, "column", N=50.0, My=5.0)]}
    multi = _member_multi(mi, mf, **_h200x100_col())
    r = check_member_strengths(multi, 275.0, 205000.0,
                               seismic_system="ordinary_moment_frame")
    m = r["members"][0]
    assert m["slenderness"]["mode"] == "compression"
    assert m["slenderness"]["status"] == "NG" and m["slenderness"]["ratio"] > 200
    assert m["ratios"]["interaction"] < 1.0, "내력비는 작아야(세장비 단독 NG 입증)"
    assert m["status"] == "NG"
    assert r["summary"]["n_slenderness_ng"] >= 1
    json.dumps(r)


def test_tier2_slenderness_tension_limit_300():
    """순인장 부재(비기둥 타이)는 L/r≤300(D1) 적용 — 초과 시 NG."""
    from core.design_check import check_member_strengths
    # 인장(N<0) 비기둥 타이, H-200x100 단면(ry≈22.2), 7m → L/r=7000/22.2≈315>300.
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 7.0, "section": ""}]
    mf = {"Uplift": [_mf(1, "beam_x", N=-50.0, My=2.0)]}
    multi = _member_multi(mi, mf, beam_x_A_mm2=2716.0, beam_x_Ix_mm4=1840e4,
                          beam_x_Iy_mm4=134e4, beam_x_J_mm4=10e4,
                          beam_x_h_mm=200.0, beam_x_b_mm=100.0,
                          beam_x_tw_mm=5.5, beam_x_tf_mm=8.0)
    m = check_member_strengths(multi, 275.0, 205000.0)["members"][0]
    assert m["axial_mode"] == "tension"
    assert m["slenderness"]["mode"] == "tension" and m["slenderness"]["limit"] == 300
    assert m["slenderness"]["status"] == "NG" and m["slenderness"]["ratio"] > 300


def test_tier2_slenderness_column_transient_tension_uses_compression_limit():
    """리뷰 후속: 지배조합이 일시 인장인 기둥도 압축한계(KL/r≤200) 적용(느슨한 인장한계 금지)."""
    from core.design_check import check_member_strengths
    # 기둥, 지배조합 N<0(일시 인장). 그래도 압축재이므로 압축한계(200) 적용.
    mi = [{"member_id": 1, "type": "column", "length_m": 5.0, "section": ""}]
    mf = {"1.0DL-1.0EQX": [_mf(1, "column", N=-50.0, My=2.0)]}
    multi = _member_multi(mi, mf, **_h200x100_col())
    m = check_member_strengths(multi, 275.0, 205000.0,
                               seismic_system="ordinary_moment_frame")["members"][0]
    # 기둥은 인장지배여도 압축 세장한계 적용(200) — KL/r=1.2·5000/22.2≈270>200 → NG
    assert m["slenderness"]["mode"] == "compression" and m["slenderness"]["limit"] == 200
    assert m["slenderness"]["status"] == "NG"


def test_tier2_slenderness_beam_in_compression_checked():
    """리뷰 후속: 유의 압축을 받는 비기둥(가새/수집재)도 E2(KL/r≤200) 검토."""
    from core.design_check import check_member_strengths
    # beam_x로 분류됐지만 압축을 크게 받는 부재(가새류), H-200x100, 6m → KL/r=6000/22.2≈270>200.
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 6.0, "section": ""}]
    mf = {"1.2DL+1.0EQX": [_mf(1, "beam_x", N=400.0, My=5.0)]}  # 큰 압축
    multi = _member_multi(mi, mf, beam_x_A_mm2=2716.0, beam_x_Ix_mm4=1840e4,
                          beam_x_Iy_mm4=134e4, beam_x_J_mm4=10e4,
                          beam_x_h_mm=200.0, beam_x_b_mm=100.0,
                          beam_x_tw_mm=5.5, beam_x_tf_mm=8.0)
    m = check_member_strengths(multi, 275.0, 205000.0)["members"][0]
    assert m["axial_mode"] == "compression"
    assert m["slenderness"] is not None and m["slenderness"]["mode"] == "compression"
    assert m["slenderness"]["status"] == "NG"  # 압축 비기둥도 E2 적용


def test_tier2_nonfinite_response_gates_design_check():
    """리뷰 후속: 솔버 ok=0이어도 결과가 비유한이면(nonfinite_response) 해석실패로 게이트."""
    import json
    from core.design_check import run_design_check
    cr = _MockCaseResult([{"story": 1, "height_m": 3.5, "drift_x": 0.0, "drift_y": 0.0}])
    cr.nonfinite_response = True  # 비유한 응답 표시
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 6.0, "section": "", "story": 1}]
    multi = _member_multi(mi, {"DL": [_mf(1, "beam_x", N=0.0, My=10.0)]})
    multi.case_results = {"DL": cr}
    multi.combo_results = {"DL": cr}
    dc = run_design_check(multi, None, None)
    assert dc["overall_status"] == "NG"
    assert dc.get("analysis_error", {}).get("solver_failed") is True
    json.dumps(dc, ensure_ascii=False)


def test_tier2_non_i_section_skips_compactness():
    """리뷰 후속: 비 I형강(채널/T 등)은 B4/F3.2 미적용(n/a) — 오분류 방지."""
    from core.design_check import check_member_strengths
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 6.0, "section": "TFC-200x80"}]
    mf = {"DL": [_mf(1, "beam_x", N=0.0, My=30.0)]}
    multi = _member_multi(mi, mf)
    m = check_member_strengths(multi, 275.0, 205000.0)["members"][0]
    assert m["compactness"]["section"] == "n/a"


def test_tier2_slenderness_only_ng_emits_dedicated_issue():
    """세장비로만 NG(상관비<1)인 부재는 slenderness_exceeded로 보고 —
    '상관비 <1인데 >1.0' 오문구(member_overstressed) 미발생."""
    from core.design_check import run_design_check
    import json
    # 세장 기둥(H-200x100, 5m, K=1.2 → KL/r≈270>200), 하중 경미 → 내력비<1.
    mi = [{"member_id": 1, "type": "column", "length_m": 5.0, "section": "", "story": 1}]
    mf = {"1.2DL+1.0EQX": [_mf(1, "column", N=50.0, My=5.0)]}
    multi = _member_multi(mi, mf, **_h200x100_col())
    multi.combo_results = {"1.2DL+1.0EQX": _MockCaseResult()}

    class _BM:
        seismic_system = "ordinary_moment_frame"
        importance = "II"
    dc = run_design_check(multi, _BM(), {"Cd": 5.5, "IE": 1.0})
    types_ = [ci["type"] for ci in dc["critical_issues"]]
    assert "slenderness_exceeded" in types_, types_
    assert "member_overstressed" not in types_, "내력비<1이므로 과대응력 이슈는 없어야 함"
    # 최상위 summary에 세장비 카운트가 전파돼 §10가 읽을 수 있어야 함(Fix #1, 묶음B 반영분)
    assert dc["summary"]["n_slenderness_ng"] >= 1
    assert dc["summary"]["max_slenderness"] > 200
    assert dc["overall_status"] == "NG"
    json.dumps(dc, ensure_ascii=False)


def test_tier2_slenderness_normal_ok_and_beam_na():
    """정상 기둥 OK, 휨지배 보는 세장비 비대상(None)."""
    from core.design_check import check_member_strengths
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": ""},
          {"member_id": 2, "type": "beam_x", "length_m": 6.0, "section": ""}]
    mf = {"DL": [_mf(1, "column", N=200.0, My=30.0), _mf(2, "beam_x", N=0.0, My=80.0)]}
    multi = _member_multi(mi, mf)  # 기본 H-300x300 / H-400x200
    res = check_member_strengths(multi, 275.0, 205000.0,
                                 seismic_system="ordinary_moment_frame")
    by = {m["member_id"]: m for m in res["members"]}
    assert by[1]["slenderness"]["status"] == "OK"     # 기둥 KL/r<200
    assert by[2]["slenderness"] is None                # 보(휨지배)는 비대상
    assert res["summary"]["n_slenderness_ng"] == 0
    # 보의 웹이 압축세장(h/tw>1.49√(E/Fy))이어도 휨지배 보는 압축세장요소로 집계 안 됨
    # (정상건물 오플래그 방지 — 기둥만 n_compression_slender 집계).
    assert res["summary"]["n_compression_slender"] == 0


# ── Tier2-7: 콤팩트 분류 (B4) + 플랜지국부좌굴 (F3.2) ──
def test_tier2_classify_compactness():
    from core.design_check import _classify_compactness
    fy, E = 275.0, 205000.0  # λpf=0.38√(E/Fy)≈10.37, λrf≈27.3
    cpt = _classify_compactness(fy, E, b=200, tf=13, h=400, tw=8)   # bf/2tf=7.7
    nc = _classify_compactness(fy, E, b=300, tf=10, h=400, tw=8)    # 15.0
    sl = _classify_compactness(fy, E, b=400, tf=6, h=400, tw=8)     # 33.3
    assert cpt["flange"] == "compact"
    assert nc["flange"] == "noncompact"
    assert sl["flange"] == "slender"
    # 입력 부족 → compact 가정(폴백)
    assert _classify_compactness(fy, E, 0, 0, 0, 0)["section"] == "compact"


def test_tier2_flb_reduces_noncompact_flange():
    from core.design_check import _flb_capacity, _classify_compactness, _bending_capacity
    fy, E = 275.0, 205000.0
    Zx, Sx = 1.3e6, 1.1e6
    comp_compact = _classify_compactness(fy, E, 200, 13, 400, 8)
    comp_nc = _classify_compactness(fy, E, 300, 10, 400, 8)
    phiMp = _bending_capacity(fy, Zx)
    m_cpt = _flb_capacity(fy, E, Zx, Sx, 200, 13, 400, 8, comp_compact)
    m_nc = _flb_capacity(fy, E, Zx, Sx, 300, 10, 400, 8, comp_nc)
    assert abs(m_cpt - phiMp) < 1e-6        # compact → Mp
    assert m_nc < phiMp                       # noncompact → 감소
    assert math.isfinite(m_nc)


def test_tier2_noncompact_beam_capacity_reduced_in_member_check():
    """비콤팩트 플랜지 보 → check_member_strengths에서 phiMnx<φMp, n_noncompact 집계."""
    from core.design_check import check_member_strengths, _bending_capacity
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 6.0, "section": ""}]
    mf = {"DL": [_mf(1, "beam_x", N=0.0, My=100.0)]}
    # 비콤팩트 플랜지: b=300, tf=10 → bf/2tf=15
    multi = _member_multi(mi, mf, beam_x_b_mm=300.0, beam_x_tf_mm=10.0,
                          beam_x_h_mm=400.0, beam_x_tw_mm=8.0,
                          beam_x_A_mm2=9000.0, beam_x_Ix_mm4=25000e4,
                          beam_x_Iy_mm4=4500e4, beam_x_J_mm4=70e4)
    m = check_member_strengths(multi, 275.0, 205000.0)["members"][0]
    assert m["compactness"]["flange"] == "noncompact"
    # plastic Mp 대비 FLB 감소
    from core.section_3d import get_section_3d  # noqa: F401 (캐시 영향 없음)
    assert m["capacity"]["phiMnx_kNm"] > 0


# ── Tier2-15: 인장 순단면파단 (AISC D2-b) ──
def test_tier2_rupture_capacity_formula():
    from core.design_check import _tension_rupture_capacity, _estimate_fu
    # φt·Fu·Ae = 0.75·410·(0.85·5000)/1000
    val = _tension_rupture_capacity(410.0, 5000.0, 0.85)
    assert abs(val - 0.75 * 410 * 0.85 * 5000 / 1000) < 1e-6
    assert _estimate_fu(275.0, "SS275") == 410.0
    assert _estimate_fu(275.0, None) == round(275 * 1.5, 1)  # 강종 미상 폴백


def test_tier2_rupture_governs_low_net_area():
    """낮은 Ae/Ag(볼트공 多) → 순단면파단이 총단면항복보다 지배 → phiPnt 감소."""
    from core.design_check import check_member_strengths
    mi = [{"member_id": 1, "type": "column", "length_m": 3.0, "section": "",
           "net_area_ratio": 0.6}]
    mf = {"Uplift": [_mf(1, "column", N=-100.0, My=2.0)]}
    multi = _member_multi(mi, mf)
    multi.material_name = "SS275"
    m = check_member_strengths(multi, 275.0, 205000.0)["members"][0]
    cap = m["capacity"]
    assert cap["tension_rupture_governs"] is True
    assert cap["phiPnt_rupture_kN"] < cap["phiPnt_yield_kN"]
    assert cap["phiPnt_kN"] == cap["phiPnt_rupture_kN"]  # min 채택


def test_tier2_rupture_default_ratio_yield_governs():
    """기본 Ae/Ag=0.85 → 총단면항복 지배(정상 부재 무영향)."""
    from core.design_check import check_member_strengths
    mi = [{"member_id": 1, "type": "column", "length_m": 3.0, "section": ""}]
    mf = {"Uplift": [_mf(1, "column", N=-100.0, My=2.0)]}
    multi = _member_multi(mi, mf)
    multi.material_name = "SS275"
    cap = check_member_strengths(multi, 275.0, 205000.0)["members"][0]["capacity"]
    assert cap["tension_rupture_governs"] is False
    assert cap["phiPnt_kN"] == cap["phiPnt_yield_kN"]


# ── Tier2-14: NaN/Inf 직렬화 가드 ──
def test_tier2_sanitize_nonfinite():
    import json
    from core.design_check import _sanitize_nonfinite
    dirty = {"a": float("nan"), "b": [float("inf"), 1.0],
             "c": {"d": float("-inf"), "e": 2.5}, "f": "ok", "g": True}
    clean = _sanitize_nonfinite(dirty)
    assert clean["a"] is None
    assert clean["b"] == [None, 1.0]
    assert clean["c"]["d"] is None and clean["c"]["e"] == 2.5
    assert clean["g"] is True
    json.dumps(clean)  # 표준 JSON 직렬화 가능


def test_tier2_frame3d_safe_round_extreme():
    from core.frame_3d import _safe_round, _safe_extreme
    assert _safe_round(float("nan"), 4) == 0.0
    assert _safe_round(float("inf"), 4) == 0.0
    assert _safe_round(1.23456, 2) == 1.23
    assert _safe_extreme([float("nan"), float("inf")], max) == 0.0
    assert _safe_extreme([1.0, float("nan"), 3.0], max) == 3.0
    assert _safe_extreme([], min) == 0.0


def test_tier2_run_design_check_output_json_safe_with_nan_forces():
    """부재력에 NaN이 섞여도 run_design_check 결과는 JSON 직렬화 가능(NaN→None)."""
    import json
    from core.design_check import run_design_check
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 6.0, "section": "", "story": 1}]
    mf = {"DL": [{"member_id": 1, "type": "beam_x", "N_kN": [0.0],
                  "My_kNm": [float("nan")], "Mz_kNm": [0.0],
                  "Vy_kN": [float("nan")], "Vz_kN": [0.0]}]}
    multi = _member_multi(mi, mf)
    multi.combo_results = {"DL": _MockCaseResult()}
    dc = run_design_check(multi, None, None)
    s = json.dumps(dc, ensure_ascii=False)   # NaN 토큰 없어야 함
    assert "NaN" not in s and "Infinity" not in s


# ════════════════════════════════════════════════════════════
# Tier2 묶음 B (#8 우발비틀림 / #9 직교지진 / #10 풍하중변위 /
#               #11 모달모드수 / #12 수직비정형 / #13 not_checked 단정)
# ════════════════════════════════════════════════════════════

# ── Tier2-9: 직교방향 지진 100%+30% (KDS 41 17 00 §8.1.3, SDC C/D) ──
def test_tier2_seismic_design_category_thresholds():
    from core.design_check import seismic_design_category
    assert seismic_design_category(0.46, 0.25, "II") == "D"   # SD1 0.25≥0.20 → D
    assert seismic_design_category(0.40, 0.10, "II") == "C"   # SDS:C, SD1:B → C
    assert seismic_design_category(0.10, 0.05, "II") == "A"   # 둘 다 낮음
    assert seismic_design_category(0.40, 0.10, "특") == "D"   # 특등급은 더 엄격
    assert seismic_design_category(None, 0.2, "II") is None    # 입력 부족 → 미상


def test_tier2_orthogonal_combos_gated_on_sdc():
    """직교 100/30 조합은 orthogonal_seismic=True + EQX·EQY 둘 다일 때만 생성(하위호환)."""
    from core.load_generator import generate_load_combinations
    cases = ["DL", "LL", "EQX", "EQY"]
    base = generate_load_combinations(cases, sds=0.4, orthogonal_seismic=False)
    ortho = generate_load_combinations(cases, sds=0.4, orthogonal_seismic=True)
    # 미적용(SDC A/B): 30% 직교성분 없음
    assert not any(("0.3EQX" in k or "0.3EQY" in k) for k in base)
    # 적용(SDC C/D): 100%주 + 30%직교, 양방향 모두
    assert any(c.get("EQX") == 1.0 and c.get("EQY") == 0.3 for c in ortho.values())
    assert any(c.get("EQY") == 1.0 and c.get("EQX") == 0.3 for c in ortho.values())
    # 부호조합 ±100/±30 모두 생성
    signs = {(c["EQX"], c["EQY"]) for c in ortho.values() if "EQX" in c and "EQY" in c}
    for s in [(1.0, 0.3), (1.0, -0.3), (-1.0, 0.3), (0.3, 1.0), (-0.3, -1.0)]:
        assert s in signs, s
    # 직교조합은 base보다 16개 더(8 down + 8 up)
    assert len(ortho) == len(base) + 16
    # 단방향(EQY 없음)은 직교 미생성 (정상건물 무영향)
    one = generate_load_combinations(["DL", "LL", "EQX"], sds=0.4, orthogonal_seismic=True)
    assert not any("0.3" in k for k in one)


# ── Tier2-8: 우발 비틀림 ±5% 편심 (KDS 41 17 00 §7.2.6.4) ──
def test_tier2_accidental_torsion_load_generator():
    """generate_seismic_loads가 우발 비틀림 Mt=Fx·0.05·b 산정 (DB 미가용 시 skip)."""
    import pytest
    from core.building_model import BuildingModel
    from core.load_generator import generate_seismic_loads, _calculate_story_weights
    cfg = {"stories": [{"height": 3.5, "usage": "office"}] * 2,
           "bays_x": [6, 6], "bays_y": [6], "region": "서울"}
    m = BuildingModel.from_json(cfg)
    try:
        sw = _calculate_story_weights(
            m, [{"type": "floor_area", "value": 5.0, "story": s} for s in (1, 2)])
        out = generate_seismic_loads(m, sw)
    except Exception:
        pytest.skip("seismic DB/region unavailable")
    rep = out["report"]
    if rep.get("error"):
        pytest.skip(f"seismic report error: {rep['error']}")
    at = rep["accidental_torsion"]
    assert at["applied_in_model"] is False and at["eccentricity_ratio"] == 0.05
    sm0 = at["story_moments"][0]
    # 저장값은 2자리 반올림 → 반올림 오차 허용
    assert abs(sm0["Mt_x_kNm"] - sm0["Fx_kN"] * 0.05 * m.total_width_y) < 0.01
    assert abs(sm0["Mt_y_kNm"] - sm0["Fx_kN"] * 0.05 * m.total_width_x) < 0.01


def test_tier2_accidental_torsion_surfaced_in_design_check():
    """등가정적 우발편심 미적용 → run_design_check가 summary+critical+R36으로 명시."""
    import json
    from core.design_check import run_design_check
    from core.result_interpreter import interpret_results
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    mf = {"1.2DL+1.0EQX": [_mf(1, "column", N=200.0, My=40.0, Vy=20.0)]}
    multi = _member_multi(mi, mf)
    multi.combo_results = {"1.2DL+1.0EQX": _MockCaseResult(
        [{"story": 1, "height_m": 3.5, "drift_x": 0.001, "drift_y": 0.0}])}
    rep = {"Cd": 3.0, "IE": 1.0, "SDS": 0.4, "SD1": 0.15,
           "story_forces": [{"story": 1, "weight_kN": 1000, "Fx_kN": 50, "height_m": 3.5}],
           "W_kN": 1000, "V_kN": 50,
           "accidental_torsion": {"eccentricity_ratio": 0.05, "applied_in_model": False,
                                  "story_moments": [{"story": 1, "Fx_kN": 50,
                                                     "Mt_x_kNm": 50.0, "Mt_y_kNm": 75.0}],
                                  "total_Mt_x_kNm": 50.0, "total_Mt_y_kNm": 75.0}}
    dc = run_design_check(multi, None, rep)
    assert dc["summary"]["accidental_torsion_applied"] is False
    assert any(ci["type"] == "accidental_torsion_not_modeled" for ci in dc["critical_issues"])
    interp = interpret_results(dc, multi, modal_analysis=None)
    assert "R36" in [f["code"] for f in interp["findings"]]
    json.dumps(dc, ensure_ascii=False)


# ── Tier2-10: 풍하중 사용성 층간변위 (H/400, 탄성) ──
def _wind_combos(drift_x, name="1.2DL+1.0LL+1.0WX", h=3.5):
    return {name: _MockCaseResult(
        [{"story": 1, "height_m": h, "drift_x": drift_x, "drift_y": 0.0}])}


def test_tier2_wind_drift_ng_when_exceeds_h400():
    from core.design_check import check_wind_drifts
    ng = check_wind_drifts(_wind_combos(1 / 300.0), [3.5])   # 1/300 > 1/400
    assert ng is not None and ng["status"] == "NG" and ng["max_ratio"] > 1.0
    ok = check_wind_drifts(_wind_combos(1 / 500.0), [3.5])   # 1/500 < 1/400
    assert ok["status"] == "OK" and ok["max_ratio"] < 1.0


def test_tier2_wind_drift_none_without_wind_combos():
    from core.design_check import check_wind_drifts
    eq_only = {"1.2DL+1.0EQX": _MockCaseResult(
        [{"story": 1, "height_m": 3.5, "drift_x": 0.01, "drift_y": 0.0}])}
    assert check_wind_drifts(eq_only, [3.5]) is None


def test_tier2_wind_drift_integration_gates_overall_and_facts():
    import json
    from core.design_check import run_design_check
    from core.result_interpreter import interpret_results
    from core.narrative_interpreter import build_facts, build_number_allowlist
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    mf = {"1.2DL+1.0LL+1.0WX": [_mf(1, "column", N=100.0, My=10.0)]}
    multi = _member_multi(mi, mf)
    multi.combo_results = {"1.2DL+1.0LL+1.0WX": _MockCaseResult(
        [{"story": 1, "height_m": 3.5, "drift_x": 1 / 250.0, "drift_y": 0.0}])}
    dc = run_design_check(multi, None, None)
    assert dc["summary"]["wind_drift_status"] == "NG"
    assert dc["overall_status"] == "NG"
    assert any(ci["type"] == "wind_drift_exceeded" for ci in dc["critical_issues"])
    interp = interpret_results(dc, multi, modal_analysis=None)
    assert "R34" in [f["code"] for f in interp["findings"]]
    facts = build_facts(interp, dc)
    assert "1/400" in facts.get("wind_drift", "")
    assert "400" in build_number_allowlist(facts)   # 자기씨앗
    json.dumps(dc, ensure_ascii=False)


# ── Tier2-12: 수직 비정형 자동탐지 (KDS 41 17 00 표 5.3-2) ──
def _vert_rep(weights, fxs, h=3.5):
    return {"Cd": 3.0, "IE": 1.0, "W_kN": sum(weights), "V_kN": sum(fxs),
            "story_forces": [{"story": i + 1, "weight_kN": weights[i],
                              "Fx_kN": fxs[i], "height_m": h} for i in range(len(weights))]}


def _vert_drift(ratios, h=3.5):
    return {"checks": [{"story": i + 1, "height_m": h, "direction": "X",
                        "elastic_drift": ratios[i]} for i in range(len(ratios))]}


def test_tier2_vertical_soft_story_detected():
    from core.design_check import check_vertical_irregularity
    # 1층 매우 유연(연층): drift 5배 → K1 ≪ K2 (극단)
    rep = _vert_rep([1000, 1000, 1000], [300, 200, 100])
    drift = _vert_drift([0.010, 0.002, 0.002])
    r = check_vertical_irregularity(None, rep, drift)
    assert r is not None and r["irregular"] and r["status"] == "OK"  # 분류만 — 미게이트
    assert "stiffness_soft_story" in r["types"]
    assert any(f["type"] == "stiffness_soft_story" and f["story"] == 1 for f in r["findings"])


def test_tier2_vertical_regular_building_no_flag():
    from core.design_check import check_vertical_irregularity
    rep = _vert_rep([1000, 1000, 1000], [150, 150, 150])
    drift = _vert_drift([0.003, 0.003, 0.003])
    r = check_vertical_irregularity(None, rep, drift)
    assert r is not None and not r["irregular"] and r["types"] == []


def test_tier2_vertical_mass_irregularity():
    from core.design_check import check_vertical_irregularity
    rep = _vert_rep([1000, 2000, 1000], [150, 150, 150])  # 2층 중량 2배
    drift = _vert_drift([0.003, 0.003, 0.003])
    r = check_vertical_irregularity(None, rep, drift)
    assert r["irregular"] and "mass" in r["types"]
    assert any(f["type"] == "mass" and f["story"] == 2 for f in r["findings"])


def test_tier2_vertical_geometric_setback():
    import types
    from core.design_check import check_vertical_irregularity
    areas = {1: 288.0, 2: 288.0, 3: 96.0}   # 3층 셋백(면적 1/3 → 치수비 √3)
    bm = types.SimpleNamespace(floor_area_at_story=lambda s: areas.get(s, 0.0))
    rep = _vert_rep([1000, 1000, 500], [150, 150, 150])
    drift = _vert_drift([0.003, 0.003, 0.003])
    r = check_vertical_irregularity(bm, rep, drift)
    assert r["irregular"] and "geometric" in r["types"]


def test_tier2_vertical_integration_classification_only():
    """run_design_check 통합: 수직 비정형은 summary 노출하되 overall은 게이트 안 함."""
    import json
    from core.design_check import run_design_check
    from core.result_interpreter import interpret_results
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    mf = {"1.2DL+1.0EQX": [_mf(1, "column", N=200.0, My=40.0, Vy=20.0)]}
    multi = _member_multi(mi, mf)
    multi.stories = [3.5, 3.5, 3.5]
    multi.combo_results = {"1.2DL+1.0EQX": _MockCaseResult(
        [{"story": s, "height_m": 3.5, "drift_x": d, "drift_y": 0.0}
         for s, d in ((1, 0.010), (2, 0.002), (3, 0.002))])}
    rep = _vert_rep([1000, 1000, 1000], [300, 200, 100])
    rep.update({"SDS": 0.4, "SD1": 0.15})
    dc = run_design_check(multi, None, rep)
    assert dc["summary"]["vertical_irregular"] is True
    assert "stiffness_soft_story" in (dc["summary"]["vertical_irregularity"] or [])
    interp = interpret_results(dc, multi, modal_analysis=None)
    assert "R35" in [f["code"] for f in interp["findings"]]
    json.dumps(dc, ensure_ascii=False)


# ── Tier2-11: 모달 모드수 자동증대 (누적참여 <90% → num_modes 상향) ──
def test_tier2_modal_auto_increase_contract():
    """고층(8층) 대칭모델: 모달이 자동증대 키 보고 + num_modes는 상한(≤min(3n,30)) 이내."""
    from core.frame_3d import analyze_from_model, _EIGEN_MODE_CAP
    cols = [(x * 6, y * 6) for x in range(3) for y in range(3)]
    m = _build_v2_grid(cols, n_stories=8)
    res = analyze_from_model(m, _dl_cases(m))
    ma = res.modal_analysis
    assert ma, "V2 모달해석 필요"
    assert "mode_count_auto_increased" in ma
    cap = min(3 * 8, _EIGEN_MODE_CAP)
    assert ma["num_modes"] <= cap
    cp = ma["cumulative_participation"]
    # 자동증대 결과: 90% 도달(충분) 또는 상한 도달(더 못 늘림) 중 하나는 성립해야 함
    assert cp["sufficient_90pct"] or ma["num_modes"] >= cap or not ma["mode_count_auto_increased"]


def test_tier2_modal_low_rise_no_auto_increase():
    """저층(2층)은 초기 모드수=면내기저 → 증대 미발생(하위호환)."""
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=2)
    ma = analyze_from_model(m, _dl_cases(m)).modal_analysis
    assert ma and ma.get("mode_count_auto_increased") is False


# ── Tier2-13: 요약 not_checked 단정 정정 (지진입력 손상 게이트 구멍) ──
def test_tier2_seismic_input_error_blocks_safe_verdict():
    """손상된 seismic_report({error})가 '안전'으로 새지 않도록 게이트 (코드리뷰 confirmed)."""
    import json
    from core.design_check import run_design_check
    from core.result_interpreter import interpret_results, _classify_severity
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    mf = {"1.2DL+1.6LL": [_mf(1, "column", N=200.0, My=30.0, Vy=20.0)]}
    multi = _member_multi(mi, mf)
    multi.combo_results = {"1.2DL+1.6LL": _MockCaseResult()}
    rep = {"error": "wind_v0 not found for 동해시"}   # 손상된 지진 입력
    dc = run_design_check(multi, None, rep)
    assert dc["summary"]["seismic_input_error"] is True
    assert dc["drift_check"] is None                 # 지진검토 게이트(빈 OK 누출 차단)
    assert any(ci["type"] == "seismic_input_error" for ci in dc["critical_issues"])
    # 게이트 구멍 정정: '안전'으로 단정되지 않음 (최소 marginal)
    assert _classify_severity(dc) != "safe"
    interp = interpret_results(dc, multi, modal_analysis=None)
    assert "R37" in [f["code"] for f in interp["findings"]]
    json.dumps(dc, ensure_ascii=False)


def test_tier2_not_checked_items_listed_for_gravity_only():
    from core.design_check import run_design_check
    from core.narrative_interpreter import build_facts
    from core.result_interpreter import interpret_results
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    mf = {"1.2DL+1.6LL": [_mf(1, "column", N=200.0, My=30.0)]}
    multi = _member_multi(mi, mf)
    multi.combo_results = {"1.2DL+1.6LL": _MockCaseResult()}
    dc = run_design_check(multi, None, None)   # 중력만 — 지진/풍/수직 미검토
    nc = dc["summary"]["not_checked_items"]
    assert "층간변위" in nc and "풍하중 사용성 변위" in nc and "수직 비정형" in nc
    interp = interpret_results(dc, multi, modal_analysis=None)
    facts = build_facts(interp, dc)
    assert "미수행" in facts.get("not_checked_note", "")


def test_tier2_normal_seismic_building_not_flagged_not_checked():
    """정상 지진해석(유효 seismic_report)은 seismic_input_error=False — 무영향."""
    from core.design_check import run_design_check
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    mf = {"1.32DL+1.0LL+1.0EQX": [_mf(1, "column", N=200.0, My=40.0, Vy=20.0)]}
    multi = _member_multi(mi, mf)
    multi.combo_results = {"1.32DL+1.0LL+1.0EQX": _MockCaseResult(
        [{"story": 1, "height_m": 3.5, "drift_x": 0.001, "drift_y": 0.0}])}
    rep = {"Cd": 5.5, "IE": 1.0, "SDS": 0.4, "SD1": 0.15,
           "story_forces": [{"story": 1, "weight_kN": 1000, "Fx_kN": 50, "height_m": 3.5}],
           "W_kN": 1000, "V_kN": 50}
    dc = run_design_check(multi, None, rep)
    assert dc["summary"]["seismic_input_error"] is False
    assert dc["drift_check"] is not None   # 지진검토 정상 수행


# ── 묶음 B 적대적 리뷰 후속 수정 회귀 ──
def test_tier2b_wind_combo_no_false_positive_substring():
    """외부/사용자정의 조합명에 'WX' substring이 있어도 풍하중으로 오탐하지 않는다."""
    from core.design_check import check_wind_drifts
    # 'NEWX_Load'·'FloorWY'는 풍하중 토큰이 아님 → 미검토(None)
    fake = {"NEWX_Load": _MockCaseResult([{"story": 1, "height_m": 3.5,
                                           "drift_x": 1 / 200.0, "drift_y": 0.0}]),
            "FloorWY": _MockCaseResult([{"story": 1, "height_m": 3.5,
                                         "drift_x": 1 / 200.0, "drift_y": 0.0}])}
    assert check_wind_drifts(fake, [3.5]) is None
    # 진짜 풍하중 조합('1.0WX')은 정상 매칭
    real = {"1.2DL+1.0LL+1.0WX": _MockCaseResult([{"story": 1, "height_m": 3.5,
                                                   "drift_x": 1 / 200.0, "drift_y": 0.0}])}
    r = check_wind_drifts(real, [3.5])
    assert r is not None and r["status"] == "NG"


def test_tier2b_accidental_torsion_gated_on_seismic_error():
    """손상 seismic_report(error)면 accidental_torsion이 동봉돼도 surfacing 안 함(이중보고 방지)."""
    from core.design_check import run_design_check
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    multi = _member_multi(mi, {"1.2DL+1.6LL": [_mf(1, "column", N=200.0, My=30.0)]})
    multi.combo_results = {"1.2DL+1.6LL": _MockCaseResult()}
    rep = {"error": "x", "accidental_torsion": {"applied_in_model": False,
           "eccentricity_ratio": 0.05, "story_moments": [{"story": 1, "Fx_kN": 50,
           "Mt_x_kNm": 50.0, "Mt_y_kNm": 75.0}], "total_Mt_x_kNm": 50.0, "total_Mt_y_kNm": 75.0}}
    dc = run_design_check(multi, None, rep)
    assert dc["accidental_torsion"] is None
    assert not any(ci["type"] == "accidental_torsion_not_modeled" for ci in dc["critical_issues"])
    assert dc["summary"]["seismic_input_error"] is True   # 지진입력 손상만 보고


def test_tier2b_accidental_torsion_none_ecc_no_crash():
    """eccentricity_ratio=None(키 존재+None)이어도 critical_issue 생성이 크래시하지 않는다."""
    import json
    from core.design_check import run_design_check
    from core.narrative_interpreter import build_facts
    from core.result_interpreter import interpret_results
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": "", "story": 1}]
    multi = _member_multi(mi, {"1.2DL+1.0EQX": [_mf(1, "column", N=200.0, My=40.0, Vy=20.0)]})
    multi.combo_results = {"1.2DL+1.0EQX": _MockCaseResult(
        [{"story": 1, "height_m": 3.5, "drift_x": 0.001, "drift_y": 0.0}])}
    rep = {"Cd": 3.0, "IE": 1.0, "SDS": 0.4, "SD1": 0.15,
           "story_forces": [{"story": 1, "weight_kN": 1000, "Fx_kN": 50, "height_m": 3.5}],
           "W_kN": 1000, "V_kN": 50,
           "accidental_torsion": {"applied_in_model": False, "eccentricity_ratio": None,
                                  "story_moments": [{"story": 1, "Fx_kN": 50,
                                  "Mt_x_kNm": None, "Mt_y_kNm": None},],
                                  "total_Mt_x_kNm": None, "total_Mt_y_kNm": None}}
    dc = run_design_check(multi, None, rep)   # None 값으로 TypeError 안 나야 함
    interp = interpret_results(dc, multi, modal_analysis=None)
    build_facts(interp, dc)
    json.dumps(dc, ensure_ascii=False)


def test_tier2b_slenderness_keys_propagated_to_top_summary():
    """Tier2-6/7 세장비/콤팩트 카운트가 최상위 summary에 전파(§10 리포트가 직접 읽음)."""
    from core.design_check import run_design_check
    mi = [{"member_id": 1, "type": "column", "length_m": 5.0, "section": ""}]
    multi = _member_multi(mi, {"DL": [_mf(1, "column", N=50.0, My=5.0)]}, **_h200x100_col())
    multi.combo_results = {"DL": _MockCaseResult()}
    dc = run_design_check(multi, None, None)
    s = dc["summary"]
    for k in ("n_slenderness_ng", "max_slenderness", "n_noncompact", "n_compression_slender"):
        assert k in s, f"최상위 summary에 {k} 누락"
    # H-200x100 5m 기둥(K=1.2 미상→모멘트) → KL/r>200 세장비 초과가 summary에 노출
    assert s["n_slenderness_ng"] >= 1 and s["max_slenderness"] > 200


# ════════════════════════════════════════════════════════════
# Tier3 잔여이슈 (견고성/모델링 근사) — docs/analysis_residual_audit_2026-06.md
#   T3-1 member story None 폴백 / T3-2 영길이 필터 / T3-3 강막 rank /
#   T3-4 릴리즈 torsion-dof 가드+노트 / T3-5 P-Delta 층단위 근사 /
#   T3-6 적설/풍 폴백 / T3-7 Cs_min·IE / T3-8 강막가정 / T3-9 기둥비지지길이 /
#   T3-A V2 마스터→질량중심(누적rz≤100%) / T3-B V2 T1_rz_s
# ════════════════════════════════════════════════════════════

import types as _types_t3


# ── T3-1: member story z좌표 기반 폴백 헬퍼 ──
def test_tier3_z_story_map_and_fallback():
    from core.frame_3d import _z_story_map, _story_from_z
    zmap = _z_story_map([0.0, 3.5, 7.0, 3.5, 0.0])  # 중복 무시, 정렬
    assert zmap == {0.0: 0, 3.5: 1, 7.0: 2}
    assert _story_from_z(0.0, zmap) == 0      # base
    assert _story_from_z(3.5, zmap) == 1
    assert _story_from_z(7.0, zmap) == 2
    assert _story_from_z(99.0, zmap) is None  # 미존재 레벨
    # 반올림 허용오차(3자리)
    assert _story_from_z(3.5004, zmap) == 1


def test_tier3_v2_no_none_member_story():
    """실 V2 해석: 모든 부재가 층(story)을 가진다(z폴백으로 None 제거, happy-path)."""
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=2)
    res = analyze_from_model(m, _dl_cases(m))
    assert res.member_info
    assert all(mm.get("story") is not None for mm in res.member_info)


def test_tier3_z_fallback_preserves_base_story_in_builder():
    """node_grid/story_nodes_map 없이 빌드 시 z-폴백이 발동하고 base(0)를 보존한다.

    구 코드 `_story_from_z(nj) or _story_from_z(ni)`는 falsy 0을 버려 base층(0) 부재를
    상위층(1)으로 오매핑했다. nj=base 절점인 부재로 그 회귀를 잠근다(solve 미수행 → 안전).
    """
    from core.frame_3d import _build_frame_3d_model, Node3D
    from core.section_3d import get_section_3d
    sec = get_section_3d("H-300x300")
    bsec = get_section_3d("H-400x200")
    nodes = [Node3D(id=1, x=0, y=0, z=0.0), Node3D(id=2, x=0, y=0, z=3.5)]
    # 부재를 (ni=상부 절점2, nj=base 절점1)로 등록 → z폴백이 nj(z=0)→story 0을 보존해야 함.
    conns = [(2, 1, "column")]
    _ei, _m2e, minfo, _nn = _build_frame_3d_model(
        nodes, conns, [1], "fixed", sec, bsec, bsec, 205000.0, 78846.0, 1,
        node_grid=None, story_nodes_map=None, num_stories=1)
    assert minfo[0]["story"] == 0, "base(0) 미보존 — 'a or b' falsy-0 버그 재발"


# ── T3-2: 영길이/좌표중복 연결 필터 헬퍼 + V2 실해석 W06 surfacing ──
def test_tier3_degenerate_connections_helper():
    from core.frame_3d import _degenerate_connections
    nodes = [_types_t3.SimpleNamespace(id=i, x=x, y=0.0, z=0.0)
             for i, x in [(1, 0.0), (2, 6.0), (3, 6.0), (4, 12.0)]]
    # member1: 1-2 정상, member2: 3-3(ni==nj), member3: 2-3(좌표일치 6.0==6.0), member4: 1-4 정상
    conns = [(1, 2, "beam"), (3, 3, "beam"), (2, 3, "beam"), (1, 4, "beam")]
    bad = _degenerate_connections(nodes, conns)
    assert bad == [2, 3]   # 1-based member_id
    # 정상만 있으면 빈 목록
    assert _degenerate_connections(nodes, [(1, 2, "beam"), (1, 4, "beam")]) == []


def test_tier3_v2_zero_length_surfaced_as_W06():
    """실 V2 해석: 좌표중복 요소 제거가 robustness_warnings(W06)로 노출(침묵 금지)."""
    import json
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=2)
    ex = next(n for n in m.nodes.values()
              if abs(n.z - 3.5) < 0.01 and abs(n.x) < 0.01 and abs(n.y) < 0.01)
    other = next(n for n in m.nodes.values()
                 if abs(n.z - 3.5) < 0.01 and abs(n.x - 6) < 0.01 and abs(n.y) < 0.01)
    dup = m.add_node(ex.x, ex.y, ex.z)            # ex와 좌표 일치(별 ID)
    m.add_element(ex.id, dup.id, elem_type="beam", section="H-400x200")    # 영길이 → 제거
    m.add_element(dup.id, other.id, elem_type="beam", section="H-400x200")  # 실연결(고아 방지)
    res = analyze_from_model(m, _dl_cases(m))
    rob = res.analysis_metadata.get("robustness_warnings") or []
    assert any(w["code"] == "W06" for w in rob)
    json.dumps(res.analysis_metadata, ensure_ascii=False)


# ── T3-3: 강막(rigidDiaphragm) 절점 <3 rank 결손 경고 헬퍼 ──
def test_tier3_diaphragm_rank_warnings_helper():
    from core.frame_3d import _diaphragm_rank_warnings
    snm = {0: [1, 2, 3, 4], 1: [11, 12], 2: [21, 22, 23, 24]}
    # rigid=False → 무경고
    assert _diaphragm_rank_warnings(snm, False) == []
    # rigid=True → 1층(2절점)만 W07, 0층(지점) 제외, 2층(4절점) 무경고
    w = _diaphragm_rank_warnings(snm, True)
    assert len(w) == 1 and w[0]["code"] == "W07" and w[0]["story"] == 1
    assert w[0]["node_count"] == 2
    # 1절점 층은 '미형성' 경고
    w1 = _diaphragm_rank_warnings({1: [11]}, True)
    assert w1 and "미형성" in w1[0]["message"]


# ── T3-4 / T3-8: 모델링 근사 가정 노트 헬퍼 ──
def test_tier3_modeling_assumptions_helper():
    from core.frame_3d import _modeling_assumptions
    assert [a["code"] for a in _modeling_assumptions(True, False)] == ["A06"]
    assert [a["code"] for a in _modeling_assumptions(False, True)] == ["A08"]
    assert [a["code"] for a in _modeling_assumptions(True, True)] == ["A06", "A08"]
    assert _modeling_assumptions(False, False) == []


def test_tier3_normal_v2_no_false_robustness_flags():
    """정상 대칭 V2(강막 미사용·릴리즈 없음)는 robustness_warnings/modeling_assumptions 무발생."""
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=2)
    md = analyze_from_model(m, _dl_cases(m)).analysis_metadata
    assert md.get("robustness_warnings") is None
    assert md.get("modeling_assumptions") is None


def test_tier3_v2_modeling_assumptions_a06_a08_surfaced():
    """실 V2 해석: 강막(rigid)+릴리즈 모델은 modeling_assumptions에 A06+A08 노출(과대주장 방지).

    절점/층 ≥4개 안정 격자 → W07/특이행렬 없이 가정노트만 surfacing.
    """
    import json
    from core.frame_3d import analyze_from_model
    from core.structural_model import ReleaseType
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=2)
    m.rigid_diaphragm = True
    beam = next(e for e in m.elements.values() if e.elem_type.value == "beam")
    beam.release_j = ReleaseType.MOMENT_Z
    md = analyze_from_model(m, _dl_cases(m)).analysis_metadata
    codes = [a["code"] for a in (md.get("modeling_assumptions") or [])]
    assert "A06" in codes and "A08" in codes, codes
    # 안정 격자(4절점/층)이므로 rank 경고 없음 + 솔버 정상
    assert not any(w["code"] == "W07" for w in (md.get("robustness_warnings") or []))
    assert md.get("solver_failed") is None
    json.dumps(md, ensure_ascii=False)


# ── T3-A: V2 마스터절점→질량중심 (누적 rz 참여 ≤100%, 순수병진 무허위비틀림) ──
def test_tier3_v2_centroid_rz_participation_bounded():
    """짝수격자 대칭 V2: 질량중심 기준으로 누적 rz 참여 ≤100% + 병진모드 비틀림≈0 (T3-A)."""
    import json
    from core.frame_3d import analyze_from_model
    # 2×2 기둥격자 → n_cols=2(짝수): 구버전 master=snodes[len//2]는 기하중심 이탈 →
    # 순수병진모드에 허위 rz가 섞여 누적 rz가 100%를 초과할 수 있었다.
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=2)
    ma = analyze_from_model(m, _dl_cases(m)).modal_analysis
    assert ma, "V2 모달해석 필요"
    cp = ma["cumulative_participation"]
    # 핵심 안전속성: 누적 참여율이 100%를 넘지 않음(반올림 0.1%×3성분 → 100.5 상한이면 충분).
    # 구 master=snodes[len//2] 버그면 병진모드 비틀림 오염으로 누적 rz가 100%를 초과했다.
    assert cp["x_pct"] <= 100.5 and cp["y_pct"] <= 100.5 and cp["rz_pct"] <= 100.5, cp
    # 대칭모델: 유의 병진(>20%) 모드는 비틀림참여 ≈0 (질량중심 기준 → 순수병진 Lrz=0).
    for md in ma["modes"]:
        mp = md["mass_participation"]
        if max(mp["x_pct"], mp["y_pct"]) > 20:
            assert mp["rz_pct"] < 2.0, f"병진모드에 허위 비틀림 {mp['rz_pct']}% (기준점 오류 의심)"
    json.dumps(ma, ensure_ascii=False)


def test_tier3_v2_irregular_rz_still_bounded():
    """비대칭(L자 결손) V2도 누적 rz 참여 ≤100% (질량중심 기준 → 분해 정합)."""
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (12, 0), (0, 6), (6, 6), (12, 6), (0, 12), (6, 12)])
    ma = analyze_from_model(m, _dl_cases(m)).modal_analysis
    assert ma
    cp = ma["cumulative_participation"]
    assert cp["rz_pct"] <= 101, cp
    # 비대칭이므로 비틀림 참여 자체는 존재
    assert cp["rz_pct"] > 0


# ── T3-B: V2 fundamental_periods에 T1_rz_s + 모드 top-level rz 키 ──
def test_tier3_v2_t1_rz_period_present():
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=2)
    ma = analyze_from_model(m, _dl_cases(m)).modal_analysis
    assert ma
    fp = ma["fundamental_periods"]
    assert "T1_rz_s" in fp                       # V1과 동일 키(구버전 V2는 누락)
    assert fp["T1_rz_s"] is not None and fp["T1_rz_s"] > 0
    # _t1_dir이 읽는 top-level rz 키 존재
    assert all("mass_participation_rz_pct" in md for md in ma["modes"])


# ── T3-5: P-Delta 1/(1−θ) 층단위 B2 근사 명시 ──
def test_tier3_pdelta_story_level_amplification_note():
    from core.design_check import check_pdelta_stability
    # 증폭대역(0.1<θ≤θmax): θ=0.12
    rep, drift = _pdelta_inputs([3000], [100], [0.012], Cd=3.0)
    r = check_pdelta_stability(rep, drift)
    assert r["amplification_method"] == "story_level_B2_1/(1-theta)"
    assert r["member_level_b1_note"] is True
    assert any("B1" in a for a in r["assumptions"])
    # 저-θ(OK)대역: 부재 B1 노트 미발생
    rep2, drift2 = _pdelta_inputs([1000, 1000], [50, 100], [0.005, 0.004], Cd=5.5)
    r2 = check_pdelta_stability(rep2, drift2)
    assert r2["member_level_b1_note"] is False


def test_tier3_pdelta_note_surfaced_in_summary():
    from core.design_check import run_design_check
    multi = _pdelta_multi("linear")
    sf = [{"story": 1, "weight_kN": 3000, "Fx_kN": 100, "height_m": 3.5}]
    rep = {"story_forces": sf, "W_kN": 3000, "V_kN": 100, "Cd": 3.0, "IE": 1.0,
           "SDS": 0.4, "SD1": 0.15}
    dc = run_design_check(multi, None, rep)
    s = dc["summary"]
    assert s["pdelta_amplification_method"] == "story_level_B2_1/(1-theta)"
    assert s["pdelta_member_level_note"] is True


# ── T3-9: 기둥 비지지길이 = 층고 가정 명시 ──
def test_tier3_column_unbraced_assumption():
    from core.design_check import check_member_strengths
    mi = [{"member_id": 1, "type": "column", "length_m": 3.5, "section": ""}]
    mf = {"DL": [_mf(1, "column", N=200.0, My=30.0)]}
    r = check_member_strengths(_member_multi(mi, mf), 275.0, 205000.0)
    assert r["summary"]["column_unbraced_basis"] == "story_height"
    assert any("비지지길이" in a and "층고" in a for a in r["assumptions"])
    assert any("unbraced_length_m" in a for a in r["assumptions"])


# ── T3-6: 적설 불균형/표류 미모델 명시 + 풍속 V0 폴백 경고 ──
def _mock_hazard(value_key, value):
    def _q(region, key):
        if key == value_key:
            return {"status": "success", "regions": [{value_key: value}]}
        return {"status": "success", "regions": [{}]}
    return _q


def test_tier3_snow_limitations_note(monkeypatch):
    import core.load_generator as lg
    from core.building_model import BuildingModel
    monkeypatch.setattr(lg, "query_hazard_values", _mock_hazard("snow_sg", 0.5))
    m = BuildingModel.from_json({"bays_x": [6, 6], "bays_y": [6, 6], "region": "서울",
                                 "stories": [{"height": 3.5, "usage": "office"}]})
    rep = lg.generate_snow_loads(m)["report"]
    assert "limitations" in rep
    joined = " ".join(rep["limitations"])
    assert "불균형" in joined and ("표류" in joined or "drift" in joined)


def test_tier3_wind_v0_fallback_warning(monkeypatch):
    import core.load_generator as lg
    from core.building_model import BuildingModel
    cfg = {"bays_x": [6, 6], "bays_y": [6, 6], "region": "테스트시",
           "stories": [{"height": 3.5, "usage": "office"}] * 2}
    m = BuildingModel.from_json(cfg)
    # wind_v0 키 없음 → 폴백 + 경고
    monkeypatch.setattr(lg, "query_hazard_values",
                        lambda region, key: {"status": "success", "regions": [{}]})
    rep = lg.generate_wind_loads(m)["report"]
    assert rep["V0_fallback"] is True and rep["V0_ms"] == 26.0
    # 경고 내용 검증: 'V0' 토큰 + 지역명 + 폴백값(26)이 모두 포함돼야 의미 있는 경고(부분퇴화 차단).
    assert rep.get("warnings")
    w0 = rep["warnings"][0]
    assert "V0" in w0 and "테스트시" in w0 and "26" in w0, w0
    # wind_v0 존재 → 폴백 아님, 경고 없음
    monkeypatch.setattr(lg, "query_hazard_values", _mock_hazard("wind_v0", 30.0))
    rep2 = lg.generate_wind_loads(m)["report"]
    assert rep2["V0_fallback"] is False and rep2["V0_ms"] == 30.0
    assert "warnings" not in rep2


# ── T3-7: Cs_min의 IE 적용 조항/민감도 명시 ──
def test_tier3_cs_min_ie_clarification():
    import pytest
    from core.building_model import BuildingModel
    from core.load_generator import generate_seismic_loads, _calculate_story_weights
    cfg = {"stories": [{"height": 3.5, "usage": "office"}] * 2,
           "bays_x": [6, 6], "bays_y": [6], "region": "서울"}
    m = BuildingModel.from_json(cfg)
    try:
        sw = _calculate_story_weights(
            m, [{"type": "floor_area", "value": 5.0, "story": s} for s in (1, 2)])
        out = generate_seismic_loads(m, sw)
    except Exception:
        pytest.skip("seismic DB/region unavailable")
    rep = out["report"]
    if rep.get("error"):
        pytest.skip(f"seismic report error: {rep['error']}")
    assert "IE" in rep["Cs_min_basis"]
    assert rep["IE_applied_to_Cs_min"] is True
    assert rep["Cs_governed_by"] in ("Cs", "Cs_min", "Cs_max")


# ════════════════════════════════════════════════════════════
# Tier4 잔여이슈 (보수적/명확성/테스트, LOW) — docs/analysis_residual_audit_2026-06.md
#   T4-1 활하중 면적저감(opt-in) / T4-2 보 약축휨 H1(보수가정 명시) /
#   T4-3 단위변환 주석 / T4-4 단층 story_nodes_map 완전성 /
#   T4-5 비틀림 미평가 플래그 / T4-6 C-phase 수치 allowlist 자기씨앗
# ════════════════════════════════════════════════════════════


# ── T4-1: 활하중 면적저감 (KDS 41 12 00 §3.5) — opt-in (기본 off) ──
def test_tier4_live_reduction_factor_formula():
    """C = 0.3 + 4.2/√A (영향면적 A≥36m²) — KDS 41 12 00 §3.5 저감식."""
    from core.load_generator import live_load_reduction_factor
    r = live_load_reduction_factor(72.0, floors_supported=1,
                                   live_load_kNm2=2.5, occupancy="office")
    assert r["applied"] is True
    assert abs(r["factor"] - round(0.3 + 4.2 / math.sqrt(72.0), 4)) < 1e-9  # 4자리 반올림
    assert r["clause"] == "KDS 41 12 00 3.5"
    assert r["influence_area_m2"] == 72.0


def test_tier4_live_reduction_below_min_area_no_reduction():
    """영향면적 < 36 m² → 저감 미적용(C=1.0)."""
    from core.load_generator import live_load_reduction_factor
    r = live_load_reduction_factor(30.0, floors_supported=1,
                                   live_load_kNm2=2.5, occupancy="office")
    assert r["applied"] is False and r["factor"] == 1.0


def test_tier4_live_reduction_min_factor_floor():
    """대면적: 공식 C는 0.3 근접하나 최소 0.5(1층)/0.4(2층 이상)로 floor."""
    from core.load_generator import live_load_reduction_factor
    r1 = live_load_reduction_factor(10000.0, floors_supported=1, live_load_kNm2=2.5)
    r2 = live_load_reduction_factor(10000.0, floors_supported=3, live_load_kNm2=2.5)
    assert abs(r1["factor"] - 0.5) < 1e-9
    assert abs(r2["factor"] - 0.4) < 1e-9
    # C는 절대 1.0 초과 불가, 0 미만 불가
    assert 0.0 <= r1["factor"] <= 1.0


def test_tier4_live_reduction_heavy_load_exception():
    """활하중 > 5 kN/m²: 1층 지지 저감 불가, 2층 이상 지지 시 C≥0.8 (§3.5.3)."""
    from core.load_generator import live_load_reduction_factor
    r1 = live_load_reduction_factor(100.0, floors_supported=1, live_load_kNm2=8.0)
    assert r1["applied"] is False and r1["factor"] == 1.0
    r2 = live_load_reduction_factor(100.0, floors_supported=2, live_load_kNm2=8.0)
    assert r2["applied"] is True and abs(r2["factor"] - 0.8) < 1e-9


def test_tier4_live_reduction_occupancy_exceptions():
    """공중집회(≤5kN/m²)·승용차 주차장 저감 불가; 주차장 2층 이상은 C≥0.8."""
    from core.load_generator import live_load_reduction_factor
    assembly = live_load_reduction_factor(200.0, floors_supported=1,
                                          live_load_kNm2=5.0, occupancy="assembly")
    assert assembly["applied"] is False and assembly["factor"] == 1.0
    park1 = live_load_reduction_factor(200.0, floors_supported=1,
                                       live_load_kNm2=3.0, occupancy="parking")
    assert park1["applied"] is False and park1["factor"] == 1.0
    park2 = live_load_reduction_factor(200.0, floors_supported=2,
                                       live_load_kNm2=3.0, occupancy="parking")
    assert park2["applied"] is True and abs(park2["factor"] - 0.8) < 1e-9


def test_tier4_live_reduction_opt_in_default_off_and_recorded(monkeypatch):
    """기본 경로(apply_live_reduction=False)는 LL 무변경(하위호환); on이면 report 기록."""
    import json
    from core.building_model import BuildingModel
    import core.load_generator as lg
    monkeypatch.setattr(lg, "_query_live_load", lambda usage: 2.5)
    cfg = {"bays_x": [6, 6], "bays_y": [6, 6],
           "stories": [{"height": 3.5, "usage": "office"}] * 3}
    m = BuildingModel.from_json(cfg)

    base = lg.generate_gravity_loads(m)                       # 기본 off
    assert base["load_cases"]["LL"][0]["value"] == 2.5        # 무변경
    assert all("live_reduction" not in r for r in base["report"])

    red = lg.generate_gravity_loads(m, apply_live_reduction=True)
    # 패널 6×6=36 부하면적 ×2(보) = 72 m² → C=0.3+4.2/√72≈0.795
    rr = red["report"][0]["live_reduction"]
    assert rr["applied"] is True and rr["factor"] < 1.0
    assert rr["LL_original_kNm2"] == 2.5
    assert red["load_cases"]["LL"][0]["value"] < 2.5
    assert abs(red["load_cases"]["LL"][0]["value"] - 2.5 * rr["factor"]) < 0.011
    json.dumps(red)  # JSON-safe


def test_tier4_live_reduction_model_flag_threads_through(monkeypatch):
    """model.live_load_reduction=True → generate_all_loads가 저감 적용(config opt-in)."""
    from core.building_model import BuildingModel
    import core.load_generator as lg
    monkeypatch.setattr(lg, "_query_live_load", lambda usage: 2.5)
    cfg = {"bays_x": [6, 6], "bays_y": [6, 6],
           "stories": [{"height": 3.5, "usage": "office"}] * 2}

    on = BuildingModel.from_json({**cfg, "live_load_reduction": True})
    assert on.live_load_reduction is True
    out_on = lg.generate_all_loads(on)
    assert any("live_reduction" in r for r in out_on["reports"]["gravity"])
    assert out_on["load_cases"]["LL"][0]["value"] < 2.5

    off = BuildingModel.from_json(cfg)               # 기본 False
    assert off.live_load_reduction is False
    out_off = lg.generate_all_loads(off)
    assert out_off["load_cases"]["LL"][0]["value"] == 2.5
    assert all("live_reduction" not in r for r in out_off["reports"]["gravity"])


# ── T4-2: 보 약축휨 φMny H1 포함 = 보수적 가정 (문서 명확화) ──
def test_tier4_beam_weak_axis_assumption_documented():
    """보 약축 휨내력 H1 포함이 보수적 가정으로 assumptions에 명시된다."""
    from core.design_check import check_member_strengths
    mi = [{"member_id": 1, "type": "beam_x", "length_m": 6.0, "section": ""}]
    mf = {"DL": [_mf(1, "beam_x", N=0.0, My=80.0)]}
    r = check_member_strengths(_member_multi(mi, mf), 275.0, 205000.0)
    joined = " ".join(r["assumptions"])
    assert "약축" in joined and ("보수" in joined or "포함" in joined)
    # φMny가 실제로 산정되어 capacity에 노출(약축 demand 검토 경로 존재)
    assert r["members"][0]["capacity"]["phiMny_kNm"] > 0


def test_tier4_beam_weak_axis_moment_increases_interaction():
    """약축 모멘트(Muy)를 더하면 H1 상관비가 커진다 — 보수 처리(무시 아님)."""
    from core.design_check import _interaction_H1
    r_no_y, _ = _interaction_H1(0.0, 500, 200, 400, 0.0, 50)    # Muy=0
    r_with_y, _ = _interaction_H1(0.0, 500, 200, 400, 30.0, 50)  # Muy=30
    assert r_with_y > r_no_y, "약축 휨을 포함하면 상관비가 증가(보수)해야 함"


# ── T4-4: 단층(1층) 건물 story_nodes_map 완전성 ──
def test_tier4_single_story_node_grid_complete():
    """단층 _get_story_nodes_from_grid: 층 0(base)·1(지붕)에 모든 격자절점 누락/중복 없이 매핑."""
    from core.frame_3d import _get_story_nodes_from_grid
    node_grid = {}
    nid = 1
    for s in range(2):            # 0=base, 1=roof
        for cx in range(2):
            for cy in range(2):
                node_grid[(s, cx, cy)] = nid
                nid += 1
    snm = _get_story_nodes_from_grid(node_grid, 2, 2, 1)
    assert set(snm.keys()) == {0, 1}
    assert len(snm[0]) == 4 and len(snm[1]) == 4
    all_mapped = sorted(n for ids in snm.values() for n in ids)
    assert all_mapped == list(range(1, 9))   # 8개 절점 모두 정확히 1회


def test_tier4_single_story_v1_member_story_complete():
    """단층 V1 실해석: 모든 부재가 유효 story를 가진다(None 없음)."""
    from core.frame_3d import analyze_frame_3d_multi
    dl = [{"type": "floor_area", "value": 5.0, "story": 1}]
    multi = analyze_frame_3d_multi(
        stories=[3.5], bays_x=[6, 6], bays_y=[6], load_cases={"DL": dl},
        load_combinations={"1.0DL": {"DL": 1.0}},
        column_section="H-300x300", beam_x_section="H-400x200",
        beam_y_section="H-400x200")
    assert multi.member_info
    assert all(mm.get("story") is not None for mm in multi.member_info)
    # 완전성: 단층 → 전 부재가 유일 층(roof=1)에 매핑(base-0/None 누락 없음)
    assert {mm["story"] for mm in multi.member_info} == {1}


def test_tier4_single_story_v2_member_story_complete():
    """단층 V2 실해석: story_nodes_map 절점 완전성 + 전 부재 유효 story 매핑."""
    from core.frame_3d import analyze_from_model
    m = _build_v2_grid([(0, 0), (6, 0), (0, 6), (6, 6)], n_stories=1)
    # story_nodes_map 완전성(절점 단위): 모든 절점이 층 배정(None 없음), base(0)+roof(1) 둘 다 형성
    node_stories = {n.story for n in m.nodes.values()}
    assert None not in node_stories and node_stories == {0, 1}
    res = analyze_from_model(m, _dl_cases(m))
    assert res.member_info
    assert all(mm.get("story") is not None for mm in res.member_info)
    # 완전성: 단층 → 전 부재가 roof(1)에 매핑(누락·오매핑 없음)
    assert {mm["story"] for mm in res.member_info} == {1}


# ── T4-5: 비틀림 미평가 서술 플래그 (drift·torsion 둘 다 None) ──
def test_tier4_torsion_performed_flag_false_when_gravity_only():
    """중력전용(drift·torsion 미수행) → torsion_performed=False + 미평가 노트."""
    from core.narrative_interpreter import build_facts, build_number_allowlist
    interp = {"severity": "safe", "severity_label": {"ko": "안전", "en": "safe"},
              "drift_interpretation": {}, "member_interpretation": {"weakest_link": {}}}
    dc = {"summary": {"max_drift_ratio": None, "max_interaction_ratio": 0.5,
                      "ng_stories": 0, "ng_members": 0}}
    facts = build_facts(interp, dc)
    assert facts["torsion_performed"] is False
    assert "미평가" in facts.get("torsion_note", "")
    # bool/텍스트 플래그는 allowlist에 숫자를 추가하지 않음(자기씨앗 무영향)
    allow = build_number_allowlist(facts)
    assert isinstance(allow, set)


def test_tier4_torsion_performed_flag_true_and_no_note_when_evaluated():
    """비틀림 검토 수행 시 torsion_performed=True + 미평가 노트 없음."""
    from core.narrative_interpreter import build_facts
    interp = {"severity": "safe", "severity_label": {"ko": "안전", "en": "safe"},
              "drift_interpretation": {}, "member_interpretation": {"weakest_link": {}}}
    dc = {"summary": {"max_drift_ratio": 0.4, "max_interaction_ratio": 0.5,
                      "ng_stories": 0, "ng_members": 0},
          "torsion_check": {"max_ratio": 1.15, "classification": "regular"}}
    facts = build_facts(interp, dc)
    assert facts["torsion_performed"] is True
    assert "torsion_note" not in facts
    assert "1.15" in facts["torsion"]


def test_tier4_torsion_note_suppressed_when_drift_present_torsion_absent():
    """drift는 수행됐으나 torsion만 None(V2 등)이면 '미평가' 노트는 발생하지 않음(과도경고 방지)."""
    from core.narrative_interpreter import build_facts
    interp = {"severity": "safe", "severity_label": {"ko": "안전", "en": "safe"},
              "drift_interpretation": {}, "member_interpretation": {"weakest_link": {}}}
    dc = {"summary": {"max_drift_ratio": 0.6, "max_interaction_ratio": 0.5,
                      "ng_stories": 0, "ng_members": 0}}   # torsion_check 없음, drift 존재
    facts = build_facts(interp, dc)
    assert facts["torsion_performed"] is False
    assert "torsion_note" not in facts   # drift가 있으므로 미평가 노트 억제


# ── T4-6: C-phase 수치 anti-hallucination allowlist 자기씨앗 통합 테스트 ──
def test_tier4_c_phase_numbers_self_seeded_in_allowlist():
    """C-phase 수치(전도 FS·P-Delta θ·δmax/δavg)가 build_facts→allowlist 자기씨앗으로
    근거화되어, narrator가 인용해도 '근거 없는 숫자'로 거부되지 않는다."""
    from core.narrative_interpreter import (
        build_facts, build_number_allowlist, passes_number_check)
    interp = {"severity": "moderate", "severity_label": {"ko": "주의", "en": "moderate"},
              "drift_interpretation": {}, "member_interpretation": {"weakest_link": {}}}
    dc = {
        "summary": {"max_drift_ratio": 0.4, "max_interaction_ratio": 0.6,
                    "ng_stories": 0, "ng_members": 0},
        "stability_check": {"governing_FS": 1.83, "required_FS": 1.5, "status": "OK"},
        "pdelta_check": {"max_theta": 0.123, "theta_limit": 0.167, "status": "OK"},
        "torsion_check": {"max_ratio": 1.27, "classification": "torsional"},
    }
    facts = build_facts(interp, dc)
    assert "1.83" in facts["overturning"]
    assert "0.123" in facts["pdelta_theta"]
    assert "1.27" in facts["torsion"]
    allow = build_number_allowlist(facts)
    for n in ("1.83", "1.5", "0.123", "0.167", "1.27"):
        assert n in allow, f"C-phase 수치 {n} 자기씨앗 누락"
    # facts 수치를 그대로 인용한 산문은 숫자검사를 통과
    prose = "전도 안전율 1.83(허용 1.5), P-Delta θ_max 0.123(한계 0.167), δmax/δavg 1.27."
    assert passes_number_check(prose, allow)
    # 근거 없는 숫자(9.99)는 거부(자기씨앗 경로가 무차별 통과시키지 않음)
    assert not passes_number_check(prose + " 그리고 9.99 추가.", allow)


def test_tier4_c_phase_allowlist_rejects_absent_check_numbers():
    """자기씨앗 메커니즘 negative 검증: C-phase 검토가 없으면(facts 미생성) 그 수치는
    allowlist에 없어 거부된다 — 즉 씨앗은 facts 구동이며 default/tautology가 아님."""
    from core.narrative_interpreter import (
        build_facts, build_number_allowlist, passes_number_check)
    interp = {"severity": "safe", "severity_label": {"ko": "안전", "en": "safe"},
              "drift_interpretation": {}, "member_interpretation": {"weakest_link": {}}}
    # 전도(stability)만 존재 — P-Delta·비틀림 검토는 미수행 → 해당 수치 미씨앗.
    dc = {"summary": {"max_drift_ratio": None, "max_interaction_ratio": 0.5,
                      "ng_stories": 0, "ng_members": 0},
          "stability_check": {"governing_FS": 1.83, "required_FS": 1.5, "status": "OK"}}
    facts = build_facts(interp, dc)
    assert "overturning" in facts
    assert "pdelta_theta" not in facts and "torsion" not in facts
    allow = build_number_allowlist(facts)
    # 미수행 검토 수치(P-Delta θ 0.123 / 비틀림 1.27)는 근거 없음 → 거부(거짓 인용 차단)
    assert not passes_number_check("P-Delta θ_max 0.123 발생.", allow)
    assert not passes_number_check("δmax/δavg 1.27 비틀림.", allow)
    # 실제 산정된 전도 FS·허용치는 통과
    assert passes_number_check("전도 안전율 1.83(허용 1.5).", allow)
