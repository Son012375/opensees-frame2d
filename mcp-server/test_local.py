"""
로컬 테스트 스크립트
MCP 서버 없이 직접 함수 호출 테스트
"""

from core.simple_beam import (
    analyze_simple_beam,
    get_available_sections,
    get_available_materials,
    get_section_properties,
    get_material_properties,
)
from core.continuous_beam import analyze_continuous_beam
from core.frame_2d import analyze_frame_2d, analyze_frame_2d_multi


def main():
    print("=" * 60)
    print("OpenSeesPy 단순보 해석 테스트")
    print("=" * 60)

    # 사용 가능한 단면/재료 목록
    print("\n[사용 가능한 단면]")
    print(get_available_sections())

    print("\n[사용 가능한 재료]")
    print(get_available_materials())

    # 단면 정보 조회
    print("\n[H-400x200 단면 정보]")
    print(get_section_properties("H-400x200"))

    # 재료 정보 조회
    print("\n[SS275 재료 정보]")
    print(get_material_properties("SS275"))

    # 등분포하중 해석
    print("\n" + "=" * 60)
    print("테스트 1: 등분포하중")
    print("  - 스팬: 6m")
    print("  - 하중: 10 kN/m (등분포)")
    print("  - 단면: H-400x200")
    print("=" * 60)

    result1 = analyze_simple_beam(
        span=6.0,
        load_type="uniform",
        load_value=10.0,
        section_name="H-400x200",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result1.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result1.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result1.max_shear:.2f} kN")
    print(f"  좌측 반력: {result1.reaction_left:.2f} kN")
    print(f"  우측 반력: {result1.reaction_right:.2f} kN")
    print(f"  최대 응력: {result1.max_stress:.2f} MPa")

    # 이론값 비교 (등분포하중)
    w = 10.0  # kN/m
    L = 6.0   # m
    E = 205000  # MPa
    I = 23700e4  # mm⁴

    # 이론값
    M_theory = w * L**2 / 8
    delta_theory = 5 * w * L**4 / (384 * E * I) * 1e12  # mm
    R_theory = w * L / 2

    print(f"\n이론값 비교:")
    print(f"  모멘트 - 해석: {result1.max_moment:.2f}, 이론: {M_theory:.2f} kN·m")
    print(f"  처짐 - 해석: {result1.max_displacement:.3f}, 이론: {delta_theory:.3f} mm")
    print(f"  반력 - 해석: {result1.reaction_left:.2f}, 이론: {R_theory:.2f} kN")

    # 중앙 집중하중 해석
    print("\n" + "=" * 60)
    print("테스트 2: 중앙 집중하중")
    print("  - 스팬: 6m")
    print("  - 하중: 60 kN (중앙 집중)")
    print("  - 단면: H-400x200")
    print("=" * 60)

    result2 = analyze_simple_beam(
        span=6.0,
        load_type="point_center",
        load_value=60.0,
        section_name="H-400x200",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result2.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result2.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result2.max_shear:.2f} kN")
    print(f"  좌측 반력: {result2.reaction_left:.2f} kN")
    print(f"  우측 반력: {result2.reaction_right:.2f} kN")
    print(f"  최대 응력: {result2.max_stress:.2f} MPa")

    # 이론값 (중앙집중하중)
    P = 60.0  # kN
    M_theory2 = P * L / 4
    delta_theory2 = P * L**3 / (48 * E * I) * 1e12  # mm
    R_theory2 = P / 2

    print(f"\n이론값 비교:")
    print(f"  모멘트 - 해석: {result2.max_moment:.2f}, 이론: {M_theory2:.2f} kN·m")
    print(f"  처짐 - 해석: {result2.max_displacement:.3f}, 이론: {delta_theory2:.3f} mm")
    print(f"  반력 - 해석: {result2.reaction_left:.2f}, 이론: {R_theory2:.2f} kN")

    # 삼각분포하중 해석
    print("\n" + "=" * 60)
    print("테스트 3: 삼각분포하중")
    print("  - 스팬: 6m")
    print("  - 하중: 좌측 10 kN/m → 우측 0 kN/m")
    print("  - 단면: H-400x200x8x13")
    print("=" * 60)

    result3 = analyze_simple_beam(
        span=6.0,
        load_type="triangular",
        load_value=10.0,
        load_value_end=0.0,
        section_name="H-400x200x8x13",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result3.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result3.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result3.max_shear:.2f} kN")
    print(f"  좌측 반력: {result3.reaction_left:.2f} kN")
    print(f"  우측 반력: {result3.reaction_right:.2f} kN")

    # 이론값 (삼각분포: R_left = wL/3, R_right = wL/6, M_max = wL²/(9√3))
    import math
    w_tri = 10.0
    R_left_theory3 = w_tri * L / 3
    R_right_theory3 = w_tri * L / 6
    M_theory3 = w_tri * L**2 / (9 * math.sqrt(3))
    print(f"\n이론값 비교:")
    print(f"  모멘트 - 해석: {result3.max_moment:.2f}, 이론: {M_theory3:.2f} kN·m")
    print(f"  반력(좌) - 해석: {result3.reaction_left:.2f}, 이론: {R_left_theory3:.2f} kN")
    print(f"  반력(우) - 해석: {result3.reaction_right:.2f}, 이론: {R_right_theory3:.2f} kN")

    # 부분 등분포하중 해석
    print("\n" + "=" * 60)
    print("테스트 4: 부분 등분포하중")
    print("  - 스팬: 6m")
    print("  - 하중: 10 kN/m (1m ~ 4m 구간)")
    print("  - 단면: H-400x200x8x13")
    print("=" * 60)

    result4 = analyze_simple_beam(
        span=6.0,
        load_type="partial_uniform",
        load_value=10.0,
        load_start=1.0,
        load_end=4.0,
        section_name="H-400x200x8x13",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result4.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result4.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result4.max_shear:.2f} kN")
    print(f"  좌측 반력: {result4.reaction_left:.2f} kN")
    print(f"  우측 반력: {result4.reaction_right:.2f} kN")

    # 조합하중 해석
    print("\n" + "=" * 60)
    print("테스트 5: 조합하중")
    print("  - 스팬: 6m")
    print("  - 하중: 등분포 5 kN/m + 중앙 집중 30 kN")
    print("  - 단면: H-400x200x8x13")
    print("=" * 60)

    result5 = analyze_simple_beam(
        span=6.0,
        load_type="combined",
        loads=[
            {"type": "uniform", "value": 5.0},
            {"type": "point", "value": 30.0, "location": 3.0},
        ],
        section_name="H-400x200x8x13",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result5.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result5.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result5.max_shear:.2f} kN")
    print(f"  좌측 반력: {result5.reaction_left:.2f} kN")
    print(f"  우측 반력: {result5.reaction_right:.2f} kN")

    # 이론값 (조합: 등분포 + 중앙집중)
    M_comb = 5.0 * L**2 / 8 + 30.0 * L / 4
    R_comb = 5.0 * L / 2 + 30.0 / 2
    print(f"\n이론값 비교:")
    print(f"  모멘트 - 해석: {result5.max_moment:.2f}, 이론: {M_comb:.2f} kN·m")
    print(f"  반력 - 해석: {result5.reaction_left:.2f}, 이론: {R_comb:.2f} kN")

    # 캔틸레버 + 등분포하중 해석
    print("\n" + "=" * 60)
    print("테스트 6: 캔틸레버 + 등분포하중")
    print("  - 스팬: 6m")
    print("  - 하중: 10 kN/m (등분포)")
    print("  - 경계조건: cantilever (좌측 고정)")
    print("  - 단면: H-400x200x8x13")
    print("=" * 60)

    result6 = analyze_simple_beam(
        span=6.0,
        load_type="uniform",
        load_value=10.0,
        support_type="cantilever",
        section_name="H-400x200x8x13",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result6.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result6.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result6.max_shear:.2f} kN")
    print(f"  좌측 반력: {result6.reaction_left:.2f} kN")
    print(f"  좌측 모멘트 반력: {result6.reaction_moment_left:.2f} kN·m")

    # 이론값 (캔틸레버 등분포: M=wL²/2, V=wL, delta=wL⁴/(8EI))
    M_cant = w * L**2 / 2
    V_cant = w * L
    delta_cant = w * L**4 / (8 * E * I) * 1e12  # mm
    R_cant = w * L
    print(f"\n이론값 비교:")
    print(f"  모멘트 - 해석: {result6.max_moment:.2f}, 이론: {M_cant:.2f} kN·m")
    print(f"  전단력 - 해석: {result6.max_shear:.2f}, 이론: {V_cant:.2f} kN")
    print(f"  처짐 - 해석: {result6.max_displacement:.3f}, 이론: {delta_cant:.3f} mm")
    print(f"  반력 - 해석: {result6.reaction_left:.2f}, 이론: {R_cant:.2f} kN")

    # 양단고정 + 등분포하중 해석
    print("\n" + "=" * 60)
    print("테스트 7: 양단고정 + 등분포하중")
    print("  - 스팬: 6m")
    print("  - 하중: 10 kN/m (등분포)")
    print("  - 경계조건: fixed_fixed (양단 고정)")
    print("  - 단면: H-400x200x8x13")
    print("=" * 60)

    result7 = analyze_simple_beam(
        span=6.0,
        load_type="uniform",
        load_value=10.0,
        support_type="fixed_fixed",
        section_name="H-400x200x8x13",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result7.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result7.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result7.max_shear:.2f} kN")
    print(f"  좌측 반력: {result7.reaction_left:.2f} kN")
    print(f"  좌측 모멘트 반력: {result7.reaction_moment_left:.2f} kN·m")
    print(f"  우측 모멘트 반력: {result7.reaction_moment_right:.2f} kN·m")

    # 이론값 (양단고정 등분포: M_end=wL²/12, M_mid=wL²/24, delta=wL⁴/(384EI))
    M_fixed_end = w * L**2 / 12
    M_fixed_mid = w * L**2 / 24
    delta_fixed = w * L**4 / (384 * E * I) * 1e12  # mm
    R_fixed = w * L / 2
    print(f"\n이론값 비교:")
    print(f"  단부 모멘트 - 해석: {result7.reaction_moment_left:.2f}, 이론: {M_fixed_end:.2f} kN·m")
    print(f"  처짐 - 해석: {result7.max_displacement:.3f}, 이론: {delta_fixed:.3f} mm")
    print(f"  반력 - 해석: {result7.reaction_left:.2f}, 이론: {R_fixed:.2f} kN")

    # 일단고정-일단핀 + 등분포하중 해석
    print("\n" + "=" * 60)
    print("테스트 8: 일단고정-일단핀 + 등분포하중")
    print("  - 스팬: 6m")
    print("  - 하중: 10 kN/m (등분포)")
    print("  - 경계조건: fixed_pin (좌측 고정, 우측 롤러)")
    print("  - 단면: H-400x200x8x13")
    print("=" * 60)

    result8 = analyze_simple_beam(
        span=6.0,
        load_type="uniform",
        load_value=10.0,
        support_type="fixed_pin",
        section_name="H-400x200x8x13",
        material_name="SS275",
    )

    print(f"\n결과:")
    print(f"  최대 처짐: {result8.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result8.max_moment:.2f} kN·m")
    print(f"  최대 전단력: {result8.max_shear:.2f} kN")
    print(f"  좌측 반력: {result8.reaction_left:.2f} kN")
    print(f"  우측 반력: {result8.reaction_right:.2f} kN")
    print(f"  좌측 모멘트 반력: {result8.reaction_moment_left:.2f} kN·m")

    # 이론값 (일단고정-일단핀 등분포: R_fixed=5wL/8, R_pin=3wL/8, M_fixed=wL²/8)
    R_fp_left = 5 * w * L / 8
    R_fp_right = 3 * w * L / 8
    M_fp_fixed = w * L**2 / 8
    print(f"\n이론값 비교:")
    print(f"  고정단 모멘트 - 해석: {result8.reaction_moment_left:.2f}, 이론: {M_fp_fixed:.2f} kN·m")
    print(f"  반력(좌) - 해석: {result8.reaction_left:.2f}, 이론: {R_fp_left:.2f} kN")
    print(f"  반력(우) - 해석: {result8.reaction_right:.2f}, 이론: {R_fp_right:.2f} kN")

    # 다양한 단면 테스트 (Phase 1-3)
    print("\n" + "=" * 60)
    print("테스트 9: I형강 단면")
    print("  - 스팬: 6m, 등분포 10 kN/m, 단면: I-300x150")
    print("=" * 60)

    result9 = analyze_simple_beam(
        span=6.0, load_type="uniform", load_value=10.0,
        section_name="I-300x150", material_name="SS275",
    )
    print(f"  최대 처짐: {result9.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result9.max_moment:.2f} kN·m")
    print(f"  최대 응력: {result9.max_stress:.2f} MPa")

    print("\n" + "=" * 60)
    print("테스트 10: ㄷ형강(PFC) 단면")
    print("  - 스팬: 4m, 등분포 5 kN/m, 단면: PFC-200x90")
    print("=" * 60)

    result10 = analyze_simple_beam(
        span=4.0, load_type="uniform", load_value=5.0,
        section_name="PFC-200x90", material_name="SS275",
    )
    print(f"  최대 처짐: {result10.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result10.max_moment:.2f} kN·m")
    print(f"  최대 응력: {result10.max_stress:.2f} MPa")

    print("\n" + "=" * 60)
    print("테스트 11: 원형강관(CHS) 단면")
    print("  - 스팬: 4m, 등분포 5 kN/m, 단면: ○-114.3x4.0")
    print("=" * 60)

    result11 = analyze_simple_beam(
        span=4.0, load_type="uniform", load_value=5.0,
        section_name="○-114.3x4.0", material_name="SS275",
    )
    print(f"  최대 처짐: {result11.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result11.max_moment:.2f} kN·m")
    print(f"  최대 응력: {result11.max_stress:.2f} MPa")

    print("\n" + "=" * 60)
    print("테스트 12: 정사각형중공형강(SHS) 단면")
    print("  - 스팬: 4m, 등분포 5 kN/m, 단면: □-100x100x4.0")
    print("=" * 60)

    result12 = analyze_simple_beam(
        span=4.0, load_type="uniform", load_value=5.0,
        section_name="□-100x100x4.0", material_name="SS275",
    )
    print(f"  최대 처짐: {result12.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result12.max_moment:.2f} kN·m")
    print(f"  최대 응력: {result12.max_stress:.2f} MPa")

    print("\n" + "=" * 60)
    print("테스트 13: 등변ㄱ형강 단면")
    print("  - 스팬: 3m, 등분포 3 kN/m, 단면: L-100x100x10")
    print("=" * 60)

    result13 = analyze_simple_beam(
        span=3.0, load_type="uniform", load_value=3.0,
        section_name="L-100x100x10", material_name="SS275",
    )
    print(f"  최대 처짐: {result13.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result13.max_moment:.2f} kN·m")
    print(f"  최대 응력: {result13.max_stress:.2f} MPa")

    # 단면 정보 조회 테스트 (다양한 타입)
    print("\n" + "=" * 60)
    print("단면 정보 조회 테스트")
    print("=" * 60)
    for name in ["I-300x150", "PFC-200x90", "○-114.3x4.0", "□-100x100x4.0", "L-100x100x10"]:
        props = get_section_properties(name)
        print(f"\n  [{name}]")
        print(f"    {props}")

    # ============================================================
    # Phase 2-1: 다경간 연속보 테스트
    # ============================================================

    print("\n" + "=" * 60)
    print("테스트 14: 2경간 연속보 (대칭, 등분포)")
    print("  - 경간: [6, 6] m, 전경간 등분포 10 kN/m")
    print("  - 이론: 중간지점 모멘트 = wL²/8 = 45 kN·m")
    print("=" * 60)

    result14 = analyze_continuous_beam(
        spans=[6.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
    )
    print(f"  총 길이: {result14.total_length} m")
    print(f"  최대 처짐: {result14.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result14.max_moment:.2f} kN·m (위치: {result14.max_moment_location:.2f} m)")
    print(f"  최대 전단력: {result14.max_shear:.2f} kN")
    print(f"  최대 응력: {result14.max_stress:.2f} MPa")
    print(f"  반력:")
    for r in result14.reactions:
        print(f"    지점 {r['support_index']} ({r['type']}, {r['location']}m): {r['vertical_kN']} kN")
    print(f"  경간별:")
    for sr in result14.span_results:
        print(f"    경간 {sr['span_index']}: M_max={sr['max_moment_kNm']} kN·m, V_max={sr['max_shear_kN']} kN, δ_max={sr['max_displacement_mm']} mm")

    # 이론값 비교
    w, L = 10.0, 6.0
    M_mid_theory = w * L**2 / 8
    R_mid_theory = 5 * w * L / 4
    print(f"\n이론값 비교:")
    print(f"  중간지점 모멘트 - 이론: {M_mid_theory:.2f} kN·m")
    print(f"  중간지점 반력 - 해석: {result14.reactions[1]['vertical_kN']}, 이론: {R_mid_theory:.2f} kN")

    print("\n" + "=" * 60)
    print("테스트 15: 3경간 연속보 (비대칭)")
    print("  - 경간: [6, 8, 6] m")
    print("  - 1경간: 등분포 10 kN/m, 2경간: 등분포 15 kN/m, 3경간: 집중 50 kN @ 3m")
    print("=" * 60)

    result15 = analyze_continuous_beam(
        spans=[6.0, 8.0, 6.0],
        loads=[
            {"span_index": 0, "type": "uniform", "value": 10.0},
            {"span_index": 1, "type": "uniform", "value": 15.0},
            {"span_index": 2, "type": "point", "value": 50.0, "location": 3.0},
        ],
    )
    print(f"  총 길이: {result15.total_length} m")
    print(f"  최대 처짐: {result15.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result15.max_moment:.2f} kN·m (위치: {result15.max_moment_location:.2f} m)")
    print(f"  최대 전단력: {result15.max_shear:.2f} kN")
    print(f"  반력:")
    for r in result15.reactions:
        print(f"    지점 {r['support_index']} ({r['type']}, {r['location']}m): {r['vertical_kN']} kN")
    print(f"  경간별:")
    for sr in result15.span_results:
        print(f"    경간 {sr['span_index']}: M_max={sr['max_moment_kNm']} kN·m, V_max={sr['max_shear_kN']} kN, δ_max={sr['max_displacement_mm']} mm")

    print("\n" + "=" * 60)
    print("테스트 16: 2경간 양단 고정 연속보")
    print("  - 경간: [6, 6] m, 전경간 등분포 10 kN/m, 양단 fixed")
    print("=" * 60)

    result16 = analyze_continuous_beam(
        spans=[6.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
        supports=["fixed", "pin", "fixed"],
    )
    print(f"  최대 처짐: {result16.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result16.max_moment:.2f} kN·m")
    print(f"  반력:")
    for r in result16.reactions:
        m_str = f", M={r['moment_kNm']} kN·m" if r['moment_kNm'] > 0 else ""
        print(f"    지점 {r['support_index']} ({r['type']}, {r['location']}m): {r['vertical_kN']} kN{m_str}")

    print("\n" + "=" * 60)
    print("테스트 17: 4경간 연속보")
    print("  - 경간: [5, 7, 7, 5] m, 전경간 등분포 8 kN/m")
    print("=" * 60)

    result17 = analyze_continuous_beam(
        spans=[5.0, 7.0, 7.0, 5.0],
        loads=[{"type": "uniform", "value": 8.0}],
    )
    print(f"  총 길이: {result17.total_length} m")
    print(f"  최대 처짐: {result17.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result17.max_moment:.2f} kN·m")
    print(f"  반력:")
    for r in result17.reactions:
        print(f"    지점 {r['support_index']} ({r['location']}m): {r['vertical_kN']} kN")
    print(f"  경간별:")
    for sr in result17.span_results:
        print(f"    경간 {sr['span_index']} ({sr['span_length']}m): M={sr['max_moment_kNm']}, V={sr['max_shear_kN']}, δ={sr['max_displacement_mm']} mm")

    # === 시각화 테스트 ===
    from core.visualization import plot_beam_results, plot_beam_results_interactive, plot_frame_2d_interactive
    import os

    print("\n" + "=" * 60)
    print("테스트 18: 단경간 등분포 시각화 (PNG + HTML)")
    print("=" * 60)

    result18 = analyze_simple_beam(span=6.0, load_type="uniform", load_value=10.0)
    print(f"  배열 길이: positions={len(result18.node_positions)}, disps={len(result18.displacements)}, moments={len(result18.moments)}, shears={len(result18.shears)}")

    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(output_dir, exist_ok=True)

    png_path = plot_beam_results(result18, os.path.join(output_dir, "test18_simple.png"))
    html_path = plot_beam_results_interactive(result18, os.path.join(output_dir, "test18_simple.html"))
    print(f"  PNG: {png_path} (exists={os.path.exists(png_path)}, size={os.path.getsize(png_path)} bytes)")
    print(f"  HTML: {html_path} (exists={os.path.exists(html_path)}, size={os.path.getsize(html_path)} bytes)")

    print("\n" + "=" * 60)
    print("테스트 19: 3경간 연속보 시각화 (PNG + HTML)")
    print("=" * 60)

    result19 = analyze_continuous_beam(
        spans=[6.0, 8.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
    )
    print(f"  배열 길이: positions={len(result19.node_positions)}, disps={len(result19.displacements)}, moments={len(result19.moments)}, shears={len(result19.shears)}")

    png_path2 = plot_beam_results(result19, os.path.join(output_dir, "test19_continuous.png"))
    html_path2 = plot_beam_results_interactive(result19, os.path.join(output_dir, "test19_continuous.html"))
    print(f"  PNG: {png_path2} (exists={os.path.exists(png_path2)}, size={os.path.getsize(png_path2)} bytes)")
    print(f"  HTML: {html_path2} (exists={os.path.exists(html_path2)}, size={os.path.getsize(html_path2)} bytes)")

    # === 평형 검증 + 새깅/호깅 테스트 ===
    from core.verification import verify_equilibrium

    print("\n" + "=" * 60)
    print("테스트 20: 단순보 평형 검증 + 새깅/호깅")
    print("  - 6m 등분포 10 kN/m, 단순보")
    print("=" * 60)

    result20 = analyze_simple_beam(span=6.0, load_type="uniform", load_value=10.0)
    eq20 = verify_equilibrium(result20)
    print(f"  M+(sagging): {result20.max_moment_positive:.2f} kN·m at {result20.max_moment_positive_location:.2f} m")
    print(f"  M-(hogging): {result20.max_moment_negative:.2f} kN·m at {result20.max_moment_negative_location:.2f} m")
    print(f"  Zx: {result20.Zx_mm3:.1f} mm³")
    print(f"  평형 검증:")
    print(f"    ΣV=0: {eq20['sum_vertical']['status']} (err={eq20['sum_vertical']['error_kN']:.3f} kN)")
    print(f"    ΣM=0: {eq20['sum_moment']['status']} (err={eq20['sum_moment']['error_kNm']:.3f} kN·m)")
    print(f"    Shear jumps: {eq20['shear_jumps']['status']}")
    print(f"    All passed: {eq20['all_passed']}")

    print("\n" + "=" * 60)
    print("테스트 21: 3경간 연속보 평형 검증 + 새깅/호깅")
    print("  - [6, 8, 6] m 전경간 등분포 12 kN/m")
    print("=" * 60)

    result21 = analyze_continuous_beam(
        spans=[6.0, 8.0, 6.0],
        loads=[{"type": "uniform", "value": 12.0}],
    )
    eq21 = verify_equilibrium(result21)
    print(f"  M+(sagging): {result21.max_moment_positive:.2f} kN·m at {result21.max_moment_positive_location:.2f} m")
    print(f"  M-(hogging): {result21.max_moment_negative:.2f} kN·m at {result21.max_moment_negative_location:.2f} m")
    print(f"  경간별:")
    for sr in result21.span_results:
        print(f"    경간 {sr['span_index']}: M+={sr.get('max_moment_positive_kNm', 0)} / M-={sr.get('max_moment_negative_kNm', 0)} / V={sr['max_shear_kN']} / δ={sr['max_displacement_mm']} mm")
    print(f"  평형 검증:")
    print(f"    ΣV=0: {eq21['sum_vertical']['status']} (reaction={eq21['sum_vertical']['reaction_sum_kN']}, load={eq21['sum_vertical']['applied_load_kN']})")
    print(f"    ΣM=0: {eq21['sum_moment']['status']}")
    print(f"    Shear jumps: {eq21['shear_jumps']['status']}")
    for sj in eq21['shear_jumps']['details']:
        print(f"      x={sj['location_m']}m: R={sj['reaction_kN']}kN, jump={sj['shear_jump_kN']}kN → {sj['status']}")
    print(f"    All passed: {eq21['all_passed']}")

    print("\n" + "=" * 60)
    print("테스트 22: 캔틸레버 새깅/호깅 (호깅만)")
    print("  - 4m 등분포 10 kN/m, cantilever")
    print("=" * 60)

    result22 = analyze_simple_beam(span=4.0, load_type="uniform", load_value=10.0, support_type="cantilever")
    eq22 = verify_equilibrium(result22)
    print(f"  M+(sagging): {result22.max_moment_positive:.2f} kN·m")
    print(f"  M-(hogging): {result22.max_moment_negative:.2f} kN·m at {result22.max_moment_negative_location:.2f} m")
    print(f"  평형 검증: All passed = {eq22['all_passed']}")

    # === 내부지점 좌/우 모멘트 + 처짐 판정 + 모델 정보 ===

    print("\n" + "=" * 60)
    print("테스트 23: 내부지점 좌/우 모멘트 (대칭 하중)")
    print("  - [6, 8, 6] m 전경간 등분포 10 kN/m → B_left ~= B_right")
    print("=" * 60)

    result23 = analyze_continuous_beam(
        spans=[6.0, 8.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
    )
    print("  Support Moments (end moments):")
    for r in result23.reactions:
        label = chr(65 + r["support_index"])
        m_left = r.get("moment_left_kNm", 0.0)
        m_right = r.get("moment_right_kNm", 0.0)
        print(f"    M_{label}_left  = {m_left:.2f} kN·m")
        print(f"    M_{label}_right = {m_right:.2f} kN·m")
    # 대칭 하중이므로 내부 지점 좌/우 근접 확인
    for r in result23.reactions:
        if r["support_index"] > 0 and r["support_index"] < len(result23.reactions) - 1:
            m_l = r.get("moment_left_kNm", 0.0)
            m_r = r.get("moment_right_kNm", 0.0)
            diff = abs(m_l - m_r)
            label = chr(65 + r["support_index"])
            print(f"    → {label}: |left - right| = {diff:.3f} kN·m {'(~= same, OK)' if diff < 1.0 else '(DIFF!)'}")

    print("\n" + "=" * 60)
    print("테스트 24: 내부지점 좌/우 모멘트 (비대칭 하중)")
    print("  - [6, 8, 6] m, 2경간에만 UDL 15 kN/m → B_left != B_right 예상")
    print("=" * 60)

    result24 = analyze_continuous_beam(
        spans=[6.0, 8.0, 6.0],
        loads=[{"span_index": 1, "type": "uniform", "value": 15.0}],
    )
    print("  Support Moments (end moments):")
    for r in result24.reactions:
        label = chr(65 + r["support_index"])
        m_left = r.get("moment_left_kNm", 0.0)
        m_right = r.get("moment_right_kNm", 0.0)
        print(f"    M_{label}_left  = {m_left:.2f} kN·m")
        print(f"    M_{label}_right = {m_right:.2f} kN·m")
    # 비대칭이므로 B_left != B_right 확인
    for r in result24.reactions:
        if r["support_index"] > 0 and r["support_index"] < len(result24.reactions) - 1:
            m_l = r.get("moment_left_kNm", 0.0)
            m_r = r.get("moment_right_kNm", 0.0)
            diff = abs(m_l - m_r)
            label = chr(65 + r["support_index"])
            same_or_diff = "(DIFF, expected)" if diff > 0.5 else "(~= same)"
            print(f"    → {label}: |left - right| = {diff:.3f} kN·m {same_or_diff}")

    print("\n" + "=" * 60)
    print("테스트 25: 처짐 판정 (L/300)")
    print("  - [6, 8, 6] m 전경간 등분포 10 kN/m")
    print("=" * 60)

    result25 = analyze_continuous_beam(
        spans=[6.0, 8.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
        deflection_limit=300,
    )
    print("  Deflection check (serviceability):")
    for sr in result25.span_results:
        sp = sr["span_index"] + 1
        sp_len = sr["span_length"]
        d_max = sr["max_displacement_mm"]
        d_allow = sr.get("delta_allow_mm", 0)
        status = sr.get("deflection_status", "?")
        print(f"    Span {sp} ({sp_len}m): δ_max = {d_max:.3f} mm, δ_allow(L/300) = {d_allow:.1f} mm → {status}")
    print(f"  deflection_limit_ratio = {result25.deflection_limit_ratio}")

    print("\n" + "=" * 60)
    print("테스트 26: 모델 신뢰성 정보")
    print("  - result25의 메타정보 확인")
    print("=" * 60)

    print("  Model info:")
    print(f"    Material: E = {result25.E_MPa / 1000:.1f} GPa, fy = {result25.fy_MPa} MPa")
    print(f"    Section: Ix = {result25.Ix_mm4:.3e} mm⁴ (used in analysis)")
    print(f"             Zx = {result25.Zx_mm3:.1f} mm³")
    print(f"             h  = {result25.h_mm:.0f} mm")
    print(f"    Numerical: Element = elasticBeamColumn")
    print(f"               Elements/span = {result25.num_elements_per_span}")
    print(f"               Total elements = {result25.num_elements_per_span * result25.num_spans}")
    print(f"               Load method = eleLoad -beamUniform (global Y)")
    if result25.num_elements_per_span < 10:
        print(f"    ⚠ Warning: Low mesh density may affect displacement accuracy.")

    # 시각화 테스트 (좌/우 라벨 + 처짐판정 + 모델정보 패널 확인)
    print("\n  Generating visualization with new panels...")
    png_25 = plot_beam_results(result25, os.path.join(output_dir, "test25_defl_check.png"))
    html_25 = plot_beam_results_interactive(result25, os.path.join(output_dir, "test25_defl_check.html"))
    print(f"    PNG: {png_25} (exists={os.path.exists(png_25)})")
    print(f"    HTML: {html_25} (exists={os.path.exists(html_25)})")

    # 비대칭 시각화
    png_24 = plot_beam_results(result24, os.path.join(output_dir, "test24_asymmetric.png"))
    html_24 = plot_beam_results_interactive(result24, os.path.join(output_dir, "test24_asymmetric.html"))
    print(f"    Asymmetric PNG: {png_24} (exists={os.path.exists(png_24)})")
    print(f"    Asymmetric HTML: {html_24} (exists={os.path.exists(html_24)})")

    # ============================================================
    # Phase 2-2: 내부 힌지 테스트 (Gerber Beam)
    # ============================================================

    print("\n" + "=" * 60)
    print("테스트 27: 2경간 연속보 + 중간 힌지 (Gerber Beam)")
    print("  - 경간: [6, 6] m, 전경간 등분포 10 kN/m")
    print("  - 힌지: 지점 B(인덱스 1)에 힌지")
    print("  - 예상: 힌지 위치에서 모멘트 = 0")
    print("=" * 60)

    result27 = analyze_continuous_beam(
        spans=[6.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
        hinges=[1],  # 지점 B에 힌지
    )
    print(f"  총 길이: {result27.total_length} m")
    print(f"  힌지 인덱스: {result27.hinges}")
    print(f"  힌지 위치: {result27.hinge_locations} m")
    print(f"  최대 처짐: {result27.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result27.max_moment:.2f} kN·m (위치: {result27.max_moment_location:.2f} m)")
    print(f"  M+(sagging): {result27.max_moment_positive:.2f} kN·m at {result27.max_moment_positive_location:.2f} m")
    print(f"  M-(hogging): {result27.max_moment_negative:.2f} kN·m at {result27.max_moment_negative_location:.2f} m")
    print(f"  반력:")
    for r in result27.reactions:
        label = chr(65 + r["support_index"])
        hinge_str = " (HINGE)" if r.get("has_hinge", False) else ""
        m_l = r.get("moment_left_kNm", 0.0)
        m_r = r.get("moment_right_kNm", 0.0)
        print(f"    지점 {label}{hinge_str}: R={r['vertical_kN']} kN, M_left={m_l:.2f}, M_right={m_r:.2f} kN·m")
    print(f"  경간별:")
    for sr in result27.span_results:
        print(f"    경간 {sr['span_index']}: M_max={sr['max_moment_kNm']} kN·m, V_max={sr['max_shear_kN']} kN")

    # 힌지 위치에서 모멘트 확인
    for r in result27.reactions:
        if r.get("has_hinge", False):
            label = chr(65 + r["support_index"])
            m_l = abs(r.get("moment_left_kNm", 0.0))
            m_r = abs(r.get("moment_right_kNm", 0.0))
            print(f"\n  힌지 검증 ({label}):")
            print(f"    M_left = {m_l:.3f} kN·m (expected ~0)")
            print(f"    M_right = {m_r:.3f} kN·m (expected ~0)")
            if m_l < 0.5 and m_r < 0.5:
                print(f"    → OK: 힌지 위치에서 모멘트 해제 확인")
            else:
                print(f"    → WARNING: 힌지 위치에서 모멘트가 0이 아님")

    # 노드별 결과 테이블 출력 및 CSV 저장
    print(f"\n  노드별 결과 (처음 10개):")
    print(f"    {'node':>4} {'x_m':>8} {'DY_mm':>12} {'RZ_rad':>12} {'M_kNm':>10} {'V_kN':>10}")
    print(f"    {'-'*4} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for i in range(min(10, len(result27.node_positions))):
        print(f"    {i+1:>4} {result27.node_positions[i]:>8.3f} {result27.displacements[i]:>12.6f} {result27.rotations[i]:>12.6f} {result27.moments[i]:>10.3f} {result27.shears[i]:>10.3f}")
    if len(result27.node_positions) > 10:
        print(f"    ... (총 {len(result27.node_positions)}개 노드)")

    # CSV 파일로 저장
    csv_27 = os.path.join(output_dir, "test27_nodal_results.csv")
    with open(csv_27, "w") as f:
        f.write("node,x_m,DY_mm,RZ_rad,M_kNm,V_kN\n")
        for i in range(len(result27.node_positions)):
            f.write(f"{i+1},{result27.node_positions[i]:.6f},{result27.displacements[i]:.6f},{result27.rotations[i]:.6f},{result27.moments[i]:.6f},{result27.shears[i]:.6f}\n")
    print(f"\n  CSV: {csv_27} (exists={os.path.exists(csv_27)})")

    # 시각화 생성
    png_27 = plot_beam_results(result27, os.path.join(output_dir, "test27_gerber_2span.png"))
    html_27 = plot_beam_results_interactive(result27, os.path.join(output_dir, "test27_gerber_2span.html"))
    print(f"  PNG: {png_27} (exists={os.path.exists(png_27)})")
    print(f"  HTML: {html_27} (exists={os.path.exists(html_27)})")

    print("\n" + "=" * 60)
    print("테스트 28: 3경간 연속보 + 2개 힌지")
    print("  - 경간: [6, 8, 6] m, 전경간 등분포 10 kN/m")
    print("  - 힌지: 지점 B(1), C(2)에 힌지")
    print("=" * 60)

    result28 = analyze_continuous_beam(
        spans=[6.0, 8.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
        hinges=[1, 2],  # 지점 B, C에 힌지
    )
    print(f"  총 길이: {result28.total_length} m")
    print(f"  힌지 인덱스: {result28.hinges}")
    print(f"  힌지 위치: {result28.hinge_locations} m")
    print(f"  최대 처짐: {result28.max_displacement:.3f} mm")
    print(f"  최대 모멘트: {result28.max_moment:.2f} kN·m")
    print(f"  반력:")
    for r in result28.reactions:
        label = chr(65 + r["support_index"])
        hinge_str = " (HINGE)" if r.get("has_hinge", False) else ""
        m_l = r.get("moment_left_kNm", 0.0)
        m_r = r.get("moment_right_kNm", 0.0)
        print(f"    지점 {label}{hinge_str}: R={r['vertical_kN']} kN, M_left={m_l:.2f}, M_right={m_r:.2f} kN·m")

    # CSV 파일로 저장
    csv_28 = os.path.join(output_dir, "test28_nodal_results.csv")
    with open(csv_28, "w") as f:
        f.write("node,x_m,DY_mm,RZ_rad,M_kNm,V_kN\n")
        for i in range(len(result28.node_positions)):
            f.write(f"{i+1},{result28.node_positions[i]:.6f},{result28.displacements[i]:.6f},{result28.rotations[i]:.6f},{result28.moments[i]:.6f},{result28.shears[i]:.6f}\n")
    print(f"\n  CSV: {csv_28} (exists={os.path.exists(csv_28)})")

    # 시각화 생성
    png_28 = plot_beam_results(result28, os.path.join(output_dir, "test28_gerber_3span.png"))
    html_28 = plot_beam_results_interactive(result28, os.path.join(output_dir, "test28_gerber_3span.html"))
    print(f"  PNG: {png_28} (exists={os.path.exists(png_28)})")
    print(f"  HTML: {html_28} (exists={os.path.exists(html_28)})")

    print("\n" + "=" * 60)
    print("테스트 29: 힌지 vs 비힌지 비교")
    print("  - 경간: [6, 6] m, 전경간 등분포 10 kN/m")
    print("=" * 60)

    result29_no_hinge = analyze_continuous_beam(
        spans=[6.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
    )
    result29_hinge = analyze_continuous_beam(
        spans=[6.0, 6.0],
        loads=[{"type": "uniform", "value": 10.0}],
        hinges=[1],
    )
    print(f"  비교:")
    print(f"                    힌지 없음     힌지 있음")
    print(f"    최대 처짐:      {result29_no_hinge.max_displacement:8.3f} mm  {result29_hinge.max_displacement:8.3f} mm")
    print(f"    최대 모멘트:    {result29_no_hinge.max_moment:8.2f} kN·m  {result29_hinge.max_moment:8.2f} kN·m")
    print(f"    중간지점 반력:  {result29_no_hinge.reactions[1]['vertical_kN']:8.2f} kN  {result29_hinge.reactions[1]['vertical_kN']:8.2f} kN")
    # 힌지가 있으면 처짐이 더 크고, 모멘트 분포가 달라짐

    # ============================================================
    # Phase 3-1: 2D 골조 테스트
    # ============================================================

    print("\n" + "=" * 60)
    print("테스트 30: 1층 1경간 골조 (Portal Frame)")
    print("  - 층고: [3.5] m, 경간: [6.0] m")
    print("  - 보에 등분포 20 kN/m")
    print("=" * 60)

    result30 = analyze_frame_2d(
        stories=[3.5],
        bays=[6.0],
        loads=[{"type": "floor", "story": 1, "value": 20.0}],
        supports="fixed",
    )
    print(f"  층수: {result30.num_stories}, 경간수: {result30.num_bays}")
    print(f"  총 높이: {result30.total_height} m, 총 폭: {result30.total_width} m")
    print(f"  노드 수: {len(result30.nodes)}, 요소 수: {result30.num_elements}")
    print(f"  최대 수평변위: {result30.max_displacement_x:.3f} mm (node {result30.max_displacement_x_node})")
    print(f"  최대 수직변위: {result30.max_displacement_y:.3f} mm (node {result30.max_displacement_y_node})")
    print(f"  최대 모멘트: {result30.max_moment:.2f} kN·m (elem {result30.max_moment_element})")
    print(f"  최대 전단력: {result30.max_shear:.2f} kN (elem {result30.max_shear_element})")
    print(f"  반력:")
    for r in result30.reactions:
        print(f"    Node {r['node']}: RY={r['RY_kN']:.2f} kN, MZ={r['MZ_kNm']:.2f} kN·m")

    # 시각화 생성
    html_30 = plot_frame_2d_interactive(result30, os.path.join(output_dir, "test30_portal_frame.html"))
    print(f"  HTML: {html_30} (exists={os.path.exists(html_30)})")

    print("\n" + "=" * 60)
    print("테스트 31: 2층 2경간 골조 + 횡하중")
    print("  - 층고: [3.5, 3.2] m, 경간: [6.0, 8.0] m")
    print("  - 각 층 등분포 15 kN/m + 2층 횡하중 30 kN")
    print("=" * 60)

    result31 = analyze_frame_2d(
        stories=[3.5, 3.2],
        bays=[6.0, 8.0],
        loads=[
            {"type": "floor", "story": 1, "value": 15.0},
            {"type": "floor", "story": 2, "value": 15.0},
            {"type": "lateral", "story": 2, "fx": 30.0},
        ],
        supports="fixed",
    )
    print(f"  층수: {result31.num_stories}, 경간수: {result31.num_bays}")
    print(f"  노드 수: {len(result31.nodes)}, 요소 수: {result31.num_elements}")
    print(f"  최대 수평변위: {result31.max_displacement_x:.3f} mm (node {result31.max_displacement_x_node})")
    print(f"  층간변위각: {result31.max_drift:.6f} rad ({result31.max_drift_story}층)")
    print(f"  최대 모멘트: {result31.max_moment:.2f} kN·m (elem {result31.max_moment_element})")
    print(f"  최대 축력: {result31.max_axial:.2f} kN (elem {result31.max_axial_element})")
    print(f"  반력:")
    for r in result31.reactions:
        print(f"    Node {r['node']}: RX={r['RX_kN']:.2f} kN, RY={r['RY_kN']:.2f} kN, MZ={r['MZ_kNm']:.2f} kN·m")

    # 시각화 생성
    html_31 = plot_frame_2d_interactive(result31, os.path.join(output_dir, "test31_2story_lateral.html"))
    print(f"  HTML: {html_31} (exists={os.path.exists(html_31)})")

    print("\n" + "=" * 60)
    print("테스트 32: 3층 3경간 골조")
    print("  - 층고: [4.0, 3.5, 3.5] m, 경간: [6.0, 8.0, 6.0] m")
    print("  - 전층 등분포 20 kN/m")
    print("=" * 60)

    result32 = analyze_frame_2d(
        stories=[4.0, 3.5, 3.5],
        bays=[6.0, 8.0, 6.0],
        loads=[
            {"type": "floor", "story": 1, "value": 20.0},
            {"type": "floor", "story": 2, "value": 20.0},
            {"type": "floor", "story": 3, "value": 20.0},
        ],
        supports="fixed",
        column_section="H-350x350",
        beam_section="H-500x200",
    )
    print(f"  층수: {result32.num_stories}, 경간수: {result32.num_bays}")
    print(f"  총 높이: {result32.total_height} m, 총 폭: {result32.total_width} m")
    print(f"  노드 수: {len(result32.nodes)}, 요소 수: {result32.num_elements}")
    print(f"  최대 수직변위: {result32.max_displacement_y:.3f} mm")
    print(f"  최대 모멘트: {result32.max_moment:.2f} kN·m")
    print(f"  최대 축력: {result32.max_axial:.2f} kN")

    # 시각화 생성
    html_32 = plot_frame_2d_interactive(result32, os.path.join(output_dir, "test32_3story_3bay.html"))
    print(f"  HTML: {html_32} (exists={os.path.exists(html_32)})")

    # ============================================================
    # Phase 4: 멀티케이스 / 하중조합 / 부재력 다이어그램 테스트
    # ============================================================

    from core.frame_2d import analyze_frame_2d_multi
    from core.visualization import plot_frame_2d_multi_interactive
    from core.verification import verify_frame_equilibrium

    print("\n" + "=" * 60)
    print("테스트 33: 멀티케이스 (DL + EQX 분리) — 3층 2경간")
    print("  - 층고: [4.0, 3.5, 3.5] m, 경간: [6.0, 6.0] m")
    print("  - DL: 전층 등분포 15 kN/m")
    print("  - EQX: 3층 횡하중 50 kN")
    print("=" * 60)

    multi33 = analyze_frame_2d_multi(
        stories=[4.0, 3.5, 3.5],
        bays=[6.0, 6.0],
        load_cases={
            "DL": [
                {"type": "floor", "story": 1, "value": 15.0},
                {"type": "floor", "story": 2, "value": 15.0},
                {"type": "floor", "story": 3, "value": 15.0},
            ],
            "EQX": [
                {"type": "lateral", "story": 3, "value": 50.0},
            ],
        },
        supports="fixed",
    )
    print(f"  케이스 수: {len(multi33.case_results)}")
    print(f"  부재 수: {len(multi33.member_info)}")
    for cn, cr in multi33.case_results.items():
        print(f"  [{cn}] max_dx={cr.max_displacement_x:.3f}mm, max_dy={cr.max_displacement_y:.3f}mm, "
              f"max_M={cr.max_moment:.2f}kN·m, max_N={cr.max_axial:.2f}kN")
    print(f"  부재력 데이터 케이스 수: {len(multi33.member_forces)}")
    for cn, mf_list in multi33.member_forces.items():
        print(f"    [{cn}] 부재 수: {len(mf_list)}, 첫 부재 s점 수: {len(mf_list[0]['s']) if mf_list else 0}")

    print("\n" + "=" * 60)
    print("테스트 34: 하중조합 (1.2DL + 1.0EQX)")
    print("  - 선형중첩 검증")
    print("=" * 60)

    multi34 = analyze_frame_2d_multi(
        stories=[4.0, 3.5, 3.5],
        bays=[6.0, 6.0],
        load_cases={
            "DL": [
                {"type": "floor", "story": 1, "value": 15.0},
                {"type": "floor", "story": 2, "value": 15.0},
                {"type": "floor", "story": 3, "value": 15.0},
            ],
            "EQX": [
                {"type": "lateral", "story": 3, "value": 50.0},
            ],
        },
        load_combinations={
            "1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0},
            "1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0},
        },
        supports="fixed",
    )
    print(f"  조합 수: {len(multi34.combo_results)}")
    for cn, cr in multi34.combo_results.items():
        print(f"  [{cn}] max_dx={cr.max_displacement_x:.3f}mm, max_M={cr.max_moment:.2f}kN·m")

    # 선형중첩 검증: 1.0DL+1.0EQX의 변위 = DL 변위 + EQX 변위
    dl_cr = multi34.case_results["DL"]
    eq_cr = multi34.case_results["EQX"]
    combo_cr = multi34.combo_results["1.0DL+1.0EQX"]
    # 최상층 좌측 노드의 수평변위로 비교
    n_cols_34 = len(multi34.bays) + 1
    top_left_node = len(multi34.stories) * n_cols_34 + 1
    dl_dx = next((d["dx_mm"] for d in dl_cr.nodal_displacements if d["node"] == top_left_node), 0)
    eq_dx = next((d["dx_mm"] for d in eq_cr.nodal_displacements if d["node"] == top_left_node), 0)
    combo_dx = next((d["dx_mm"] for d in combo_cr.nodal_displacements if d["node"] == top_left_node), 0)
    superpose_err = abs(combo_dx - (dl_dx + eq_dx))
    print(f"\n  선형중첩 검증 (Node {top_left_node} dx):")
    print(f"    DL={dl_dx:.4f} + EQX={eq_dx:.4f} = {dl_dx+eq_dx:.4f} mm")
    print(f"    Combo={combo_dx:.4f} mm")
    print(f"    Error={superpose_err:.6f} mm → {'OK' if superpose_err < 0.001 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("테스트 35: 부재력 다이어그램 연속성 검증")
    print("  - sub-element 경계에서 M 값 연속 확인")
    print("=" * 60)

    mf_dl = multi34.member_forces.get("DL", [])
    if mf_dl:
        # 첫 번째 보 부재 확인
        beam_mf = next((m for m in mf_dl if m["type"] == "beam"), None)
        if beam_mf:
            s = beam_mf["s"]
            M = beam_mf["M_kNm"]
            print(f"  부재 {beam_mf['member_id']} ({beam_mf['type']}), L={beam_mf['length_m']:.2f}m")
            print(f"    s점 수: {len(s)}")
            # sub-element 경계에서 연속성 확인 (j-end[k] ≈ i-end[k+1])
            n_sub = len(beam_mf["sub_element_ids"])
            max_disc = 0.0
            for k in range(n_sub - 1):
                j_idx = 2 * k + 1  # j-end of sub-element k
                i_idx = 2 * (k + 1)  # i-end of sub-element k+1
                disc = abs(M[j_idx] - M[i_idx])
                max_disc = max(max_disc, disc)
            print(f"    M 연속성 검증: max discontinuity = {max_disc:.6f} kN·m → {'OK' if max_disc < 0.01 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("테스트 36: 골조 평형검증 (ΣFx, ΣFy, ΣM)")
    print("=" * 60)

    for cn in ["DL", "EQX"]:
        cr = multi34.case_results[cn]
        loads = multi34.load_cases[cn]
        eq = verify_frame_equilibrium(cr, loads, multi34.stories, multi34.bays)
        passed = eq["all_passed"]
        print(f"  [{cn}] all_passed={passed}")
        for key in ["sum_horizontal", "sum_vertical", "sum_moment"]:
            chk = eq[key]
            print(f"    {chk['description']}: error={chk.get('error_kN', chk.get('error_kNm', 0)):.4f} → {chk['status']}")

    print("\n" + "=" * 60)
    print("테스트 37: 층전단 검증")
    print("  - EQX 케이스: 각 층 기둥 전단합 = 해당층 이상 횡하중 합")
    print("=" * 60)

    eq_case = multi34.case_results["EQX"]
    sd = eq_case.story_data
    shears = sd.get("story_shears", [])
    print(f"  층전단력:")
    for ss in shears:
        print(f"    Story {ss['story']}: V={ss['shear_kN']:.2f} kN")
    # EQX에는 3층에만 50kN 횡하중
    # 따라서 각 층의 전단은 모두 50kN이어야 함 (3층 이상 하중의 합)
    if shears:
        for ss in shears:
            err = abs(ss["shear_kN"] - 50.0)
            print(f"    Story {ss['story']}: |V - 50| = {err:.2f} kN → {'OK' if err < 1.0 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("테스트 38: 하위호환 — 기존 단일 loads 파라미터")
    print("=" * 60)

    result38 = analyze_frame_2d(
        stories=[3.5],
        bays=[6.0],
        loads=[{"type": "floor", "story": 1, "value": 20.0}],
        supports="fixed",
    )
    print(f"  (하위호환) 층수: {result38.num_stories}, 요소수: {result38.num_elements}")
    print(f"  max_dy={result38.max_displacement_y:.3f}mm, max_M={result38.max_moment:.2f}kN·m")
    print(f"  반력수: {len(result38.reactions)}")
    # result30과 동일해야 함
    dy_diff = abs(result38.max_displacement_y - result30.max_displacement_y)
    print(f"  vs Test30 dy 차이: {dy_diff:.6f} mm → {'OK' if dy_diff < 0.001 else 'FAIL'}")

    # 탭형 HTML 시각화 생성 (Test 34 결과)
    print("\n" + "=" * 60)
    print("테스트 39: 탭형 HTML 시각화 생성")
    print("=" * 60)

    eq_checks_34 = {}
    for cn in ["DL", "EQX"]:
        cr = multi34.case_results[cn]
        loads = multi34.load_cases[cn]
        eq_checks_34[cn] = verify_frame_equilibrium(cr, loads, multi34.stories, multi34.bays)

    html_multi = plot_frame_2d_multi_interactive(
        multi34,
        equilibrium_checks=eq_checks_34,
        output_path=os.path.join(output_dir, "test39_multi_case_tabs.html"),
    )
    print(f"  HTML: {html_multi}")
    print(f"  파일 존재: {os.path.exists(html_multi)}")
    print(f"  파일 크기: {os.path.getsize(html_multi) / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
