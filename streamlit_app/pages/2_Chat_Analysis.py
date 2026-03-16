"""Chat Analysis 페이지 — 멀티스텝 IFC 워크플로우.

Phase 1: IFC 업로드 또는 자연어로 형상 입력 → 3D 미리보기 → 수정 → 확인
Phase 2: 자연어로 용도/환경/단면 입력 → 설정 요약 → 확인
Phase 3: 해석 실행 → 결과 시각화
"""
import streamlit as st
import json
import os
import traceback

st.set_page_config(page_title="Chat Analysis", page_icon="💬", layout="wide")

from modules.analysis_runner import run_building_analysis, merge_geometry_and_usage
from modules.geometry_preview import build_geometry_preview
from modules.ifc_handler import (
    parse_uploaded_ifc,
    ifc_data_to_geometry_config,
    format_geometry_summary,
)
from modules.chat_prompts import (
    GEOMETRY_MODIFY_PROMPT,
    GEOMETRY_DESCRIBE_PROMPT,
    USAGE_ENV_PROMPT,
    detect_intent,
)
from modules.viz_3d import (
    create_3d_model_figure,
    create_deformed_figure,
    create_force_colored_figure,
)
from modules.viz_charts import (
    create_story_drift_chart,
    create_envelope_table,
    create_reaction_summary,
    create_load_report_sections,
)


# ============================================================
# Session State 초기화
# ============================================================
_DEFAULTS = {
    "chat_step": "geometry",
    "ifc_uploaded": False,
    "ifc_raw_data": None,
    "geometry_config": None,
    "usage_config": None,
    "full_config": None,
    "chat_messages": [],
    "chat_model": None,
    "chat_load_result": None,
    "chat_analysis_result": None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Claude API
# ============================================================
def get_claude_response(user_message: str, system_prompt: str, api_key: str) -> str:
    """Claude API 호출."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
    except ImportError:
        st.error("anthropic 패키지가 설치되지 않았습니다. `pip install anthropic`")
        return ""
    except Exception as e:
        st.error(f"Claude API Error: {e}")
        return ""


def parse_json_response(text: str) -> dict | None:
    """응답에서 JSON 추출."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing failed: {e}")
        return None


# ============================================================
# Step Indicator
# ============================================================
def render_step_indicator():
    """상단 단계 표시."""
    step = st.session_state["chat_step"]
    step_order = ["geometry", "usage", "analysis", "done"]
    cur_idx = step_order.index(step) if step in step_order else 0

    cols = st.columns(3)
    labels = [
        ("1. Geometry", "geometry"),
        ("2. Usage / Environment", "usage"),
        ("3. Analysis", "analysis"),
    ]
    for col, (label, sid) in zip(cols, labels):
        s_idx = step_order.index(sid)
        if s_idx < cur_idx:
            col.success(f"✓ {label}")
        elif s_idx == cur_idx:
            col.info(f"▶ **{label}**")
        else:
            col.caption(f"  {label}")


# ============================================================
# 채팅 이력 렌더링
# ============================================================
def render_chat_history():
    """채팅 메시지 표시."""
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "config_json" in msg:
                with st.expander("Config JSON"):
                    st.json(msg["config_json"])


# ============================================================
# 3D 미리보기 렌더링
# ============================================================
def render_geometry_preview():
    """현재 geometry_config로 3D 미리보기 표시."""
    geo = st.session_state.get("geometry_config")
    if not geo or not geo.get("stories") or not geo.get("bays_x") or not geo.get("bays_y"):
        st.info("IFC 파일을 업로드하거나 채팅으로 형상을 설명해주세요.")
        return

    story_heights = [s["height"] for s in geo["stories"]]
    col_sec = geo.get("column_section", "H-300x300")
    beam_sec = geo.get("beam_section", "H-400x200")
    mock = build_geometry_preview(
        story_heights, geo["bays_x"], geo["bays_y"],
        column_section=col_sec,
        beam_x_section=geo.get("beam_x_section", beam_sec),
        beam_y_section=geo.get("beam_y_section", beam_sec),
    )
    fig = create_3d_model_figure(mock)
    fig.update_layout(title="Geometry Preview", height=500)
    st.plotly_chart(fig, use_container_width=True, key="geo_preview_3d")

    # 요약 메트릭
    mc = st.columns(3)
    with mc[0]:
        st.metric("Stories", f"{len(geo['stories'])}F")
    with mc[1]:
        st.metric("Height", f"{sum(s['height'] for s in geo['stories']):.1f}m")
    with mc[2]:
        total_x = sum(geo["bays_x"])
        total_y = sum(geo["bays_y"])
        st.metric("Plan", f"{total_x:.0f}×{total_y:.0f}m")


# ============================================================
# 사이드바
# ============================================================
api_key = os.environ.get("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=api_key,
        type="password",
        help="Claude API key (sk-ant-...)",
    )
    if api_key_input:
        api_key = api_key_input

    st.divider()

    # IFC 업로드
    st.subheader("IFC Upload")
    uploaded_ifc = st.file_uploader(
        "Upload IFC file",
        type=["ifc"],
        help="Revit/BIM IFC export",
        disabled=st.session_state["chat_step"] != "geometry",
    )

    if uploaded_ifc and not st.session_state["ifc_uploaded"]:
        with st.spinner("Parsing IFC..."):
            try:
                ifc_data = parse_uploaded_ifc(uploaded_ifc)
                st.session_state["ifc_raw_data"] = ifc_data
                st.session_state["ifc_uploaded"] = True

                geo = ifc_data_to_geometry_config(ifc_data)
                st.session_state["geometry_config"] = geo

                summary = format_geometry_summary(geo)
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": (
                        f"IFC 파일에서 형상을 추출했습니다.\n\n{summary}\n\n"
                        "수정이 필요하면 채팅으로 요청해주세요.\n"
                        "형상이 맞으면 **Confirm Geometry** 버튼을 누르거나 '확인'이라고 입력해주세요."
                    ),
                })

                if ifc_data.get("warnings"):
                    for w in ifc_data["warnings"]:
                        st.warning(w)
            except Exception as e:
                st.error(f"IFC parsing failed: {e}")
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": f"IFC 파싱 실패: {e}\n\n채팅으로 형상을 직접 설명해주세요.",
                })
        st.rerun()

    st.divider()

    # Phase별 예시
    step = st.session_state["chat_step"]
    if step == "geometry":
        st.markdown("""
**Phase 1: 형상 입력**

예시:
- "10층 건물, 6m x 8m 경간, 3x2 bay"
- "1층 높이를 5m로 변경"
- "X방향 경간 2개 추가"
        """)
    elif step == "usage":
        st.markdown("""
**Phase 2: 용도/환경 입력**

예시:
- "1-3층 사무실, 4층 식당, 서울"
- "H-400x400 기둥, H-600x200 보"
- "내진등급 II, S3 지반"
        """)

    st.divider()
    if st.button("Reset Workflow", use_container_width=True):
        for k in list(_DEFAULTS.keys()):
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()


# ============================================================
# 메인 영역
# ============================================================
st.title("Chat Analysis")
st.caption("IFC 업로드 또는 자연어로 건물을 설명 → 형상 확인 → 용도/환경 입력 → 구조해석")

render_step_indicator()
st.divider()

step = st.session_state["chat_step"]


# ============================================================
# Phase 1: Geometry
# ============================================================
if step == "geometry":
    left_col, right_col = st.columns([1, 1])

    with left_col:
        render_chat_history()

        # Geometry 요약
        geo = st.session_state.get("geometry_config")
        if geo and geo.get("stories"):
            with st.expander("Current Geometry", expanded=True):
                st.markdown(format_geometry_summary(geo))

            if st.button("Confirm Geometry ▶", type="primary", use_container_width=True):
                st.session_state["chat_step"] = "usage"

                # IFC에서 감지된 단면/재료 안내
                detected_info = ""
                if geo.get("column_section") or geo.get("beam_section") or geo.get("material_name"):
                    parts = []
                    if geo.get("column_section"):
                        parts.append(f"기둥: {geo['column_section']}")
                    if geo.get("beam_section"):
                        parts.append(f"보: {geo['beam_section']}")
                    if geo.get("material_name"):
                        parts.append(f"재료: {geo['material_name']}")
                    detected_info = (
                        "\n\nIFC에서 감지된 단면/재료: **" + ", ".join(parts) + "**\n"
                        "변경하려면 채팅에서 지정해주세요. 미지정 시 이 값이 사용됩니다."
                    )

                    # 대체 경고
                    for sub in geo.get("section_substitutions", []):
                        detected_info += (
                            f"\n⚠ {sub['original']} → {sub['substituted']}로 대체됨 "
                            f"({sub.get('reason', '')})"
                        )

                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": (
                        "형상이 확인되었습니다.\n\n"
                        "이제 **용도, 지역, 단면, 내진시스템**을 설명해주세요.\n\n"
                        "예: \"1-3층 사무실, 4층 식당, 최상층 옥상, 서울, H-400x400 기둥\""
                        + detected_info
                    ),
                })
                st.rerun()

    with right_col:
        render_geometry_preview()

    # 채팅 입력
    user_input = st.chat_input("건물 형상을 설명하거나 수정을 요청하세요")

    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        if not api_key:
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": "API key가 설정되지 않았습니다. 사이드바에서 API key를 입력해주세요.",
            })
            st.rerun()

        geo = st.session_state.get("geometry_config")
        intent = detect_intent(user_input)

        # 확인 의도 + geometry가 있으면 Phase 2로
        if intent == "confirm" and geo and geo.get("stories"):
            st.session_state["chat_step"] = "usage"

            detected_info = ""
            if geo.get("column_section") or geo.get("beam_section") or geo.get("material_name"):
                parts = []
                if geo.get("column_section"):
                    parts.append(f"기둥: {geo['column_section']}")
                if geo.get("beam_section"):
                    parts.append(f"보: {geo['beam_section']}")
                if geo.get("material_name"):
                    parts.append(f"재료: {geo['material_name']}")
                detected_info = (
                    "\n\nIFC에서 감지된 단면/재료: **" + ", ".join(parts) + "**\n"
                    "변경하려면 채팅에서 지정해주세요."
                )
                for sub in geo.get("section_substitutions", []):
                    detected_info += (
                        f"\n⚠ {sub['original']} → {sub['substituted']}로 대체됨"
                    )

            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": (
                    "형상이 확인되었습니다.\n\n"
                    "이제 **용도, 지역, 단면, 내진시스템**을 설명해주세요.\n\n"
                    "예: \"1-3층 사무실, 최상층 옥상, 서울, H-400x400 기둥\""
                    + detected_info
                ),
            })
        else:
            # Claude로 geometry 생성/수정
            if geo and geo.get("stories"):
                prompt = GEOMETRY_MODIFY_PROMPT.format(
                    current_geometry=json.dumps(geo, ensure_ascii=False, indent=2)
                )
            else:
                prompt = GEOMETRY_DESCRIBE_PROMPT

            with st.spinner("Claude가 형상을 분석 중..."):
                response_text = get_claude_response(user_input, prompt, api_key)

            if response_text:
                new_geo = parse_json_response(response_text)
                if new_geo and new_geo.get("stories") and new_geo.get("bays_x"):
                    st.session_state["geometry_config"] = new_geo
                    summary = format_geometry_summary(new_geo)
                    action = "업데이트" if geo else "생성"
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": (
                            f"형상이 {action}되었습니다.\n\n{summary}\n\n"
                            "수정이 필요하면 계속 요청하세요. "
                            "맞으면 **Confirm Geometry** 버튼 또는 '확인'을 입력하세요."
                        ),
                    })
                else:
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": "형상 파싱에 실패했습니다. 다시 설명해주세요.",
                    })

        st.rerun()


# ============================================================
# Phase 2: Usage / Environment
# ============================================================
elif step == "usage":
    left_col, right_col = st.columns([1, 1])

    with left_col:
        render_chat_history()

        # 설정 요약
        usage_cfg = st.session_state.get("usage_config")
        if usage_cfg:
            with st.expander("Current Config", expanded=True):
                geo = st.session_state["geometry_config"]
                geo_stories = geo.get("stories", [])
                usage_stories = usage_cfg.get("stories_usage", [])

                lines = []
                for i, gs in enumerate(geo_stories):
                    usage = usage_stories[i]["usage"] if i < len(usage_stories) else "?"
                    lines.append(f"- {i+1}F: {gs['height']:.1f}m — **{usage}**")
                lines.append(f"\n**Column:** {usage_cfg.get('column_section', '?')}")
                lines.append(f"**Beam X:** {usage_cfg.get('beam_x_section', '?')}")
                lines.append(f"**Beam Y:** {usage_cfg.get('beam_y_section', '?')}")
                lines.append(f"**Region:** {usage_cfg.get('region', '?')}")
                lines.append(f"**Site Class:** {usage_cfg.get('site_class', '?')}")
                lines.append(f"**Importance:** {usage_cfg.get('importance', '?')}")
                lines.append(f"**Seismic System:** {usage_cfg.get('seismic_system', '?')}")

                st.markdown("\n".join(lines))

            if st.button("Run Analysis ▶", type="primary", use_container_width=True):
                geo = st.session_state["geometry_config"]
                full = merge_geometry_and_usage(geo, usage_cfg)
                st.session_state["full_config"] = full
                st.session_state["chat_step"] = "analysis"
                st.rerun()

    with right_col:
        render_geometry_preview()

    # 채팅 입력
    user_input = st.chat_input("용도, 지역, 단면, 내진시스템을 설명해주세요")

    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        if not api_key:
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": "API key가 설정되지 않았습니다.",
            })
            st.rerun()

        geo = st.session_state["geometry_config"]
        geo_summary = format_geometry_summary(geo)
        num_stories = len(geo.get("stories", []))

        prompt = USAGE_ENV_PROMPT.format(
            geometry_summary=geo_summary,
            num_stories=num_stories,
        )

        with st.spinner("Claude가 설정을 분석 중..."):
            response_text = get_claude_response(user_input, prompt, api_key)

        if response_text:
            usage_cfg = parse_json_response(response_text)
            if usage_cfg and usage_cfg.get("stories_usage"):
                st.session_state["usage_config"] = usage_cfg

                # 요약 생성
                usage_stories = usage_cfg.get("stories_usage", [])
                usage_lines = []
                for i, us in enumerate(usage_stories):
                    usage_lines.append(f"  {i+1}F: {us.get('usage', '?')}")

                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": (
                        f"설정이 생성되었습니다.\n\n"
                        f"**용도:**\n" + "\n".join(usage_lines) + "\n\n"
                        f"**기둥:** {usage_cfg.get('column_section', '?')}\n"
                        f"**보:** {usage_cfg.get('beam_x_section', '?')} / {usage_cfg.get('beam_y_section', '?')}\n"
                        f"**지역:** {usage_cfg.get('region', '?')}\n"
                        f"**내진시스템:** {usage_cfg.get('seismic_system', '?')}\n\n"
                        "수정이 필요하면 말씀해주세요. "
                        "맞으면 **Run Analysis** 버튼을 눌러주세요."
                    ),
                    "config_json": usage_cfg,
                })
            else:
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": "설정 파싱에 실패했습니다. 다시 설명해주세요.",
                })

        st.rerun()


# ============================================================
# Phase 3: Analysis
# ============================================================
elif step == "analysis":
    full_config = st.session_state.get("full_config")
    if not full_config:
        st.error("Config가 없습니다. Reset 후 다시 시도하세요.")
        st.stop()

    with st.spinner("Analyzing... (KDS DB + OpenSees)"):
        try:
            model, load_result, multi = run_building_analysis(full_config)
            st.session_state["chat_model"] = model
            st.session_state["chat_load_result"] = load_result
            st.session_state["chat_analysis_result"] = multi
            st.session_state["chat_step"] = "done"
            st.rerun()
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            # usage 단계로 돌아가기
            if st.button("← Back to Usage"):
                st.session_state["chat_step"] = "usage"
                st.rerun()


# ============================================================
# Phase Done: 결과 시각화
# ============================================================
elif step == "done":
    multi = st.session_state.get("chat_analysis_result")
    model = st.session_state.get("chat_model")
    load_result = st.session_state.get("chat_load_result")

    if not multi:
        st.error("결과가 없습니다.")
        st.stop()

    # 요약 메트릭
    summary = model.summary()
    info_cols = st.columns(5)
    with info_cols[0]:
        st.metric("Stories", summary["num_stories"])
    with info_cols[1]:
        st.metric("Height", f"{summary['total_height_m']:.1f}m")
    with info_cols[2]:
        st.metric("Plan", f"{summary['total_width_x_m']:.1f}×{summary['total_width_y_m']:.1f}m")
    with info_cols[3]:
        st.metric("Elements", multi.num_elements)
    with info_cols[4]:
        st.metric("Load Cases", f"{len(multi.case_results)}+{len(multi.combo_results)}")

    st.divider()

    # 탭
    tab_model, tab_deform, tab_drift, tab_force, tab_load, tab_summary = st.tabs([
        "3D Model", "Deformed Shape", "Story Drift", "Member Forces", "Load Report", "Summary",
    ])

    all_cases = list(multi.case_results.keys()) + list(multi.combo_results.keys())

    with tab_model:
        fig = create_3d_model_figure(multi)
        st.plotly_chart(fig, use_container_width=True, key="chat_result_model")

    with tab_deform:
        sel = st.selectbox("Load Case / Combination", all_cases, key="chat_deform_case")
        scale = st.slider("Deformation Scale", 1.0, 200.0, 50.0, 5.0, key="chat_deform_scale")
        cr = multi.case_results.get(sel) or multi.combo_results.get(sel)
        if cr:
            fig = create_deformed_figure(multi, cr, scale=scale)
            st.plotly_chart(fig, use_container_width=True, key="chat_deform_3d")
            d_cols = st.columns(3)
            with d_cols[0]:
                st.metric("Max Disp X", f"{cr.max_displacement_x:.2f} mm")
            with d_cols[1]:
                st.metric("Max Disp Y", f"{cr.max_displacement_y:.2f} mm")
            with d_cols[2]:
                st.metric("Max Disp Z", f"{cr.max_displacement_z:.2f} mm")

    with tab_drift:
        drift_sel = st.selectbox("Load Case / Combination", all_cases, key="chat_drift_case")
        cr_drift = multi.case_results.get(drift_sel) or multi.combo_results.get(drift_sel)
        if cr_drift:
            fig = create_story_drift_chart(cr_drift, multi.num_stories)
            st.plotly_chart(fig, use_container_width=True, key="chat_drift_chart")

    with tab_force:
        force_sel = st.selectbox("Load Case / Combination", all_cases, key="chat_force_case")
        force_comp = st.selectbox("Force Component", ["My", "Mz", "N", "Vy", "Vz", "T"], key="chat_force_comp")
        cr_force = multi.case_results.get(force_sel) or multi.combo_results.get(force_sel)
        if cr_force:
            fig = create_force_colored_figure(multi, cr_force, force_component=force_comp)
            st.plotly_chart(fig, use_container_width=True, key="chat_force_3d")

    with tab_load:
        if load_result:
            sections = create_load_report_sections(load_result)
            for title, content in sections.items():
                with st.expander(title, expanded=False):
                    st.markdown(content)
            with st.expander("Full Config (JSON)", expanded=False):
                st.json(st.session_state.get("full_config", {}))

    with tab_summary:
        st.subheader("Envelope (All Combinations)")
        env_df = create_envelope_table(multi)
        if not env_df.empty:
            st.dataframe(env_df, use_container_width=True, hide_index=True)

        st.subheader("Reactions")
        if multi.combo_results:
            react_sel = st.selectbox("Reaction Case", all_cases, key="chat_react_case")
            cr_react = multi.case_results.get(react_sel) or multi.combo_results.get(react_sel)
            if cr_react:
                react_df = create_reaction_summary(cr_react)
                st.dataframe(react_df, use_container_width=True, hide_index=True)
