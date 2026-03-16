"""Building Analysis 페이지.

사이드바 입력 → BuildingModel → 자동하중 → 3D 해석 → 시각화
"""
import streamlit as st
import json
import traceback

st.set_page_config(page_title="Building Analysis", page_icon="🏢", layout="wide")

# 모듈 임포트
from modules.config_builder import (
    USAGE_DISPLAY, SEISMIC_SYSTEM_DISPLAY, SITE_CLASSES,
    IMPORTANCE_LEVELS, COMMON_SECTIONS, COMMON_REGIONS,
    build_config,
)
from modules.analysis_runner import run_building_analysis
from modules.viz_3d import (
    create_3d_model_figure,
    create_deformed_figure,
    create_force_colored_figure,
)
from modules.viz_charts import (
    create_story_drift_chart,
    create_reaction_summary,
    create_envelope_table,
    create_load_report_sections,
)


st.title("Building Analysis")
st.caption("KDS 기반 자동하중 생성 + 3D Frame 해석")


# ============================================================
# 사이드바 입력
# ============================================================
usage_keys = list(USAGE_DISPLAY.keys())

# ── session_state 초기화: 동적 층 목록 ──
if "stories_list" not in st.session_state:
    # 기본 3층 (1F 4m retail, 2F 3.5m office, 3F 3.5m roof)
    st.session_state.stories_list = [
        {"height": 4.0, "usage": "office", "dead_load_finish": 1.0, "slab_thickness": 0.15},
        {"height": 3.5, "usage": "office", "dead_load_finish": 1.0, "slab_thickness": 0.15},
        {"height": 3.5, "usage": "roof",   "dead_load_finish": 1.0, "slab_thickness": 0.15},
    ]


def _add_story():
    """층 추가 (마지막 층 설정 복사, 용도는 office)."""
    last = st.session_state.stories_list[-1] if st.session_state.stories_list else {}
    st.session_state.stories_list.append({
        "height": last.get("height", 3.5),
        "usage": "office",
        "dead_load_finish": last.get("dead_load_finish", 1.0),
        "slab_thickness": last.get("slab_thickness", 0.15),
    })


def _remove_story(idx: int):
    """특정 층 삭제."""
    if len(st.session_state.stories_list) > 1:
        st.session_state.stories_list.pop(idx)


with st.sidebar:
    st.header("Building Parameters")

    # --- 기하 ---
    st.subheader("Geometry")
    col1, col2 = st.columns(2)
    with col1:
        num_bays_x = st.number_input("Bays X", 1, 10, 3, 1)
        bay_width_x = st.number_input("Bay Width X (m)", 3.0, 15.0, 6.0, 0.5)
    with col2:
        num_bays_y = st.number_input("Bays Y", 1, 10, 3, 1)
        bay_width_y = st.number_input("Bay Width Y (m)", 3.0, 15.0, 6.0, 0.5)

    # --- 층 설정 (동적) ---
    # Story N = N층 기둥높이 + 상부 슬래브(N+1층 바닥) 하중
    st.subheader(f"Stories ({len(st.session_state.stories_list)}F)")
    st.caption("각 층 = 기둥 높이 + 상부 슬래브 하중 (Usage → 해당 층 천장 슬래브에 적용)")

    num_stories = len(st.session_state.stories_list)
    for i, s in enumerate(st.session_state.stories_list):
        is_top = (i == num_stories - 1)
        # 라벨: "1F: Col 4.0m, Slab → 2F floor (office)" / 최상층은 "3F: Col 3.5m, Slab → Roof"
        if is_top:
            slab_label = "Roof"
        else:
            slab_label = f"{i+2}F floor"
        expander_label = f"{i+1}F: Col {s['height']}m, Slab → {slab_label} ({s['usage']})"

        with st.expander(expander_label, expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                s["height"] = st.number_input(
                    "Column Height (m)", 2.5, 8.0, s["height"], 0.1, key=f"sh_{i}",
                )
            with c2:
                cur_idx = usage_keys.index(s["usage"]) if s["usage"] in usage_keys else 0
                s["usage"] = st.selectbox(
                    "Slab Usage (→ Live Load)", usage_keys,
                    format_func=lambda x: USAGE_DISPLAY[x],
                    index=cur_idx, key=f"su_{i}",
                )
            c3, c4 = st.columns(2)
            with c3:
                s["dead_load_finish"] = st.number_input(
                    "Slab Finish DL (kN/m²)", 0.0, 5.0, s["dead_load_finish"], 0.5, key=f"dl_{i}",
                )
            with c4:
                s["slab_thickness"] = st.number_input(
                    "Slab t (m)", 0.10, 0.30, s["slab_thickness"], 0.01, key=f"st_{i}",
                )
            if len(st.session_state.stories_list) > 1:
                st.button(f"Remove {i+1}F", key=f"rm_{i}",
                          on_click=_remove_story, args=(i,), type="secondary")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button("+ Add", on_click=_add_story, use_container_width=True)
    with btn_col2:
        st.button(
            "- Remove Last", use_container_width=True,
            on_click=_remove_story, args=(len(st.session_state.stories_list) - 1,),
            disabled=len(st.session_state.stories_list) <= 1,
        )

    num_stories = len(st.session_state.stories_list)
    total_h = sum(s["height"] for s in st.session_state.stories_list)
    st.caption(f"Total: {num_stories}F, {total_h:.1f}m")

    # --- 단면 ---
    st.subheader("Sections")
    column_section = st.selectbox("Column Section", COMMON_SECTIONS, index=11)  # H-400x400
    beam_section = st.selectbox("Beam Section", COMMON_SECTIONS, index=19)  # H-500x200

    # --- 환경/내진 ---
    st.subheader("Environment")
    region = st.selectbox("Region", COMMON_REGIONS, index=0)
    site_class = st.selectbox("Site Class", SITE_CLASSES, index=2)  # S3
    importance = st.selectbox("Importance", IMPORTANCE_LEVELS, index=2)  # II
    seismic_system = st.selectbox(
        "Seismic System",
        list(SEISMIC_SYSTEM_DISPLAY.keys()),
        format_func=lambda x: SEISMIC_SYSTEM_DISPLAY[x],
        index=2,  # ordinary_moment_frame
    )

    st.divider()

    # --- 해석 실행 ---
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)


# ============================================================
# 해석 실행
# ============================================================
if run_btn:
    config = {
        "stories": [dict(s) for s in st.session_state.stories_list],
        "bays_x": [bay_width_x] * num_bays_x,
        "bays_y": [bay_width_y] * num_bays_y,
        "column_section": column_section,
        "beam_x_section": beam_section,
        "beam_y_section": beam_section,
        "material_name": "SS275",
        "region": region,
        "site_class": site_class,
        "importance": importance,
        "seismic_system": seismic_system,
        "supports": "fixed",
        "auto_combinations": True,
        "num_elements_per_member": 4,
    }

    st.session_state["building_config"] = config

    # 예상 요소 수: (기둥 + 보) × 4 sub-elements
    n_cols = (num_bays_x + 1) * (num_bays_y + 1)
    n_beams = num_bays_x * (num_bays_y + 1) + num_bays_y * (num_bays_x + 1)
    est_elements = (n_cols + n_beams) * num_stories * 4
    est_time = max(5, est_elements * 0.04)  # ~0.04s per element

    with st.spinner(f"Analyzing... (~{est_elements} elements, estimated {est_time:.0f}s)"):
        try:
            model, load_result, multi = run_building_analysis(config)
            st.session_state["building_model"] = model
            st.session_state["load_result"] = load_result
            st.session_state["analysis_result"] = multi
            st.session_state["analysis_error"] = None
            st.success(
                f"Analysis complete: {multi.num_elements} elements, "
                f"{len(multi.case_results)} cases, "
                f"{len(multi.combo_results)} combinations"
            )
        except Exception as e:
            st.session_state["analysis_error"] = str(e)
            st.error(f"Analysis failed: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())


# ============================================================
# 결과 시각화
# ============================================================
if "analysis_result" in st.session_state and st.session_state.get("analysis_error") is None:
    multi = st.session_state["analysis_result"]
    model = st.session_state["building_model"]
    load_result = st.session_state["load_result"]

    # 건물 요약 정보
    summary = model.summary()
    info_cols = st.columns(5)
    with info_cols[0]:
        st.metric("Stories", summary["num_stories"])
    with info_cols[1]:
        st.metric("Height", f"{summary['total_height_m']:.1f} m")
    with info_cols[2]:
        st.metric("Plan", f"{summary['total_width_x_m']:.1f} x {summary['total_width_y_m']:.1f} m")
    with info_cols[3]:
        st.metric("Elements", multi.num_elements)
    with info_cols[4]:
        st.metric("Load Cases", f"{len(multi.case_results)} + {len(multi.combo_results)} combos")

    st.divider()

    # 탭
    tab_model, tab_deform, tab_drift, tab_force, tab_load, tab_summary = st.tabs([
        "3D Model", "Deformed Shape", "Story Drift", "Member Forces", "Load Report", "Summary"
    ])

    # --- 3D 모델 ---
    with tab_model:
        fig_model = create_3d_model_figure(multi)
        st.plotly_chart(fig_model, use_container_width=True, key="model_3d")

    # --- 변형 형상 ---
    with tab_deform:
        # 케이스/조합 선택
        all_cases = list(multi.case_results.keys()) + list(multi.combo_results.keys())
        selected_case = st.selectbox("Load Case / Combination", all_cases, key="deform_case")
        scale = st.slider("Deformation Scale", 1.0, 200.0, 50.0, 5.0, key="deform_scale")

        if selected_case in multi.case_results:
            cr = multi.case_results[selected_case]
        else:
            cr = multi.combo_results[selected_case]

        fig_deform = create_deformed_figure(multi, cr, scale=scale)
        st.plotly_chart(fig_deform, use_container_width=True, key="deform_3d")

        # 최대 변위 표시
        d_cols = st.columns(3)
        with d_cols[0]:
            st.metric("Max Disp X", f"{cr.max_displacement_x:.2f} mm")
        with d_cols[1]:
            st.metric("Max Disp Y", f"{cr.max_displacement_y:.2f} mm")
        with d_cols[2]:
            st.metric("Max Disp Z", f"{cr.max_displacement_z:.2f} mm")

    # --- 층간변위 ---
    with tab_drift:
        drift_case = st.selectbox("Load Case / Combination", all_cases, key="drift_case")

        if drift_case in multi.case_results:
            cr_drift = multi.case_results[drift_case]
        else:
            cr_drift = multi.combo_results[drift_case]

        fig_drift = create_story_drift_chart(cr_drift, multi.num_stories)
        st.plotly_chart(fig_drift, use_container_width=True, key="drift_chart")

        # 최대 층간변위
        d_cols = st.columns(2)
        with d_cols[0]:
            if cr_drift.max_drift_x > 1e-8:
                st.metric("Max Drift X", f"1/{int(1/cr_drift.max_drift_x)}")
            else:
                st.metric("Max Drift X", "-")
        with d_cols[1]:
            if cr_drift.max_drift_y > 1e-8:
                st.metric("Max Drift Y", f"1/{int(1/cr_drift.max_drift_y)}")
            else:
                st.metric("Max Drift Y", "-")

    # --- 부재력 ---
    with tab_force:
        force_case = st.selectbox("Load Case / Combination", all_cases, key="force_case")
        force_comp = st.selectbox(
            "Force Component",
            ["My", "Mz", "N", "Vy", "Vz", "T"],
            key="force_comp",
        )

        if force_case in multi.case_results:
            cr_force = multi.case_results[force_case]
        else:
            cr_force = multi.combo_results[force_case]

        fig_force = create_force_colored_figure(multi, cr_force, force_component=force_comp)
        st.plotly_chart(fig_force, use_container_width=True, key="force_3d")

        # 최대값
        f_cols = st.columns(4)
        with f_cols[0]:
            st.metric("Max Moment", f"{cr_force.max_moment:.1f} kNm")
        with f_cols[1]:
            st.metric("Max Axial", f"{cr_force.max_axial:.1f} kN")
        with f_cols[2]:
            st.metric("Max Shear", f"{cr_force.max_shear:.1f} kN")
        with f_cols[3]:
            st.metric("Max Torsion", f"{cr_force.max_torsion:.1f} kNm")

    # --- 하중 리포트 ---
    with tab_load:
        sections = create_load_report_sections(load_result)
        for title, content in sections.items():
            with st.expander(title, expanded=False):
                st.markdown(content)

        # JSON 설정 표시
        with st.expander("Building Config (JSON)", expanded=False):
            st.json(st.session_state.get("building_config", {}))

    # --- 요약 ---
    with tab_summary:
        st.subheader("Envelope (All Combinations)")
        env_df = create_envelope_table(multi)
        if not env_df.empty:
            st.dataframe(env_df, use_container_width=True, hide_index=True)

        st.subheader("Reactions")
        # 첫 번째 조합의 반력
        first_combo = list(multi.combo_results.keys())[0] if multi.combo_results else None
        if first_combo:
            react_case = st.selectbox("Reaction Case", all_cases, key="react_case")
            if react_case in multi.case_results:
                cr_react = multi.case_results[react_case]
            else:
                cr_react = multi.combo_results[react_case]
            react_df = create_reaction_summary(cr_react)
            st.dataframe(react_df, use_container_width=True, hide_index=True)

else:
    st.info("Configure building parameters in the sidebar and click **Run Analysis** to start.")
