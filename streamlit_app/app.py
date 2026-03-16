"""OpenSees-MCP Streamlit Prototype.

3D 구조해석 시각화 + KDS DB 기반 자동하중 생성 데모.
"""
import streamlit as st

st.set_page_config(
    page_title="OpenSees-MCP | Structural Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OpenSees-MCP Structural Analysis")

st.markdown("""
**KDS 기반 자동 구조해석 시스템 프로토타입**

이 시스템은 한국건설기준(KDS) Supabase DB를 기반으로 설계하중을 자동 생성하고,
OpenSeesPy로 3D 골조 해석을 수행합니다.

---

**페이지 안내:**
- **Building Analysis**: 건물 파라미터 입력 → 자동 하중 생성 → 3D 해석 → 시각화
- **Chat Analysis**: 자연어로 건물 설명 → Claude가 파싱 → 자동 해석 → 시각화

**차별성:**
- 설계기준 DB (Supabase 712건 load_params + 2290건 hazard) 기반 동적 조회
- MCP 구조화된 도구 실행 (LLM 코드 생성 대비 환각 방지)
- KDS 41 12 00, KDS 41 17 00, KDS 17 10 00 자동 적용
""")

# 간단한 시스템 상태 표시
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Analysis Engine", "OpenSeesPy")
with col2:
    st.metric("Design Code", "KDS (Korean)")
with col3:
    st.metric("DB Records", "3,002")
