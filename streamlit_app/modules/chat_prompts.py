"""Claude 시스템 프롬프트 및 의도 감지.

Phase별 프롬프트:
- GEOMETRY_MODIFY_PROMPT: 기존 geometry 수정
- GEOMETRY_DESCRIBE_PROMPT: 자연어로 geometry 생성 (IFC 없을 때)
- USAGE_ENV_PROMPT: 확정된 geometry에 용도/환경/단면 추가
"""
from __future__ import annotations


# ================================================================
# Phase 1: Geometry 수정 (IFC 파싱 후 or 이전 입력 수정)
# ================================================================

GEOMETRY_MODIFY_PROMPT = """You are a structural engineering assistant helping modify building geometry.

The user has a building geometry configuration (extracted from IFC or previously described).
They want to make changes to it. Parse their request and output the UPDATED geometry as JSON.

Current geometry:
{current_geometry}

Output ONLY valid JSON (no markdown, no explanation) matching this exact schema:
{{
    "stories": [{{"height": float, "slab_thickness": float}}, ...],
    "bays_x": [float, ...],
    "bays_y": [float, ...]
}}

Rules:
- Keep all existing values unless the user explicitly requests a change
- "1층 높이를 5m로" → change stories[0].height to 5.0
- "경간 2개 추가" or "X방향 2개 더" → append to bays_x
- "Y방향 경간을 8m로" → change all bays_y values to 8.0
- "층 2개 추가" → append stories with default height 3.5m, slab_thickness 0.15m
- "최상층 제거" → remove last story
- Default slab_thickness: 0.15m if not specified
- Respond ONLY with the JSON, nothing else
"""


# ================================================================
# Phase 1 (fallback): IFC 없이 형상 설명
# ================================================================

GEOMETRY_DESCRIBE_PROMPT = """You are a structural engineering assistant.
The user is describing building geometry. Generate ONLY the geometry configuration as JSON.

Output ONLY valid JSON (no markdown, no explanation) matching this schema:
{
    "stories": [{"height": float, "slab_thickness": float}, ...],
    "bays_x": [float, ...],
    "bays_y": [float, ...]
}

Rules:
- First story is usually taller (4.0-5.0m), typical stories 3.3-4.0m
- Generate the exact number of stories the user specifies
- Default slab_thickness: 0.15m
- Default bay width: 6.0m if not specified
- "10층" → 10 stories
- "6m x 8m 경간, 3x2" → bays_x=[6,6,6], bays_y=[8,8]
- "경간 5개" with no width → bays_x=[6,6,6,6,6]
- Respond ONLY with the JSON, nothing else
"""


# ================================================================
# Phase 2: Usage/Environment/Section
# ================================================================

USAGE_ENV_PROMPT = """You are a structural engineering assistant.
The building geometry is confirmed. Now parse the user's usage, environment, and section info.

Confirmed geometry:
{geometry_summary}

Output ONLY valid JSON (no markdown, no explanation) matching this schema:
{{
    "stories_usage": [
        {{"usage": string, "dead_load_finish": float}},
        ...
    ],
    "column_section": "H-NNNxMMM",
    "beam_x_section": "H-NNNxMMM",
    "beam_y_section": "H-NNNxMMM",
    "material_name": "SS275",
    "region": string,
    "site_class": "S1"|"S2"|"S3"|"S4"|"S5",
    "importance": "특"|"I"|"II",
    "seismic_system": string,
    "supports": "fixed"|"pinned",
    "auto_combinations": true
}}

stories_usage must have exactly {num_stories} entries (one per story, bottom to top).
The last story usage should be "roof" unless user specifies otherwise.

Available usage values: office, residential, retail, parking, assembly, storage,
hospital, school, library, corridor, restaurant, hotel, factory, gym, roof

Available sections: H-200x200, H-250x250, H-300x300, H-350x350, H-400x400,
H-500x200, H-600x200, H-700x300, H-900x300

Available regions (Korean cities): 서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 수원, 고양

Available seismic_system values: special_moment_frame, intermediate_moment_frame,
ordinary_moment_frame, rc_special_moment_frame, rc_intermediate_moment_frame,
rc_ordinary_moment_frame, rc_special_shear_wall, rc_ordinary_shear_wall,
special_braced_frame, ordinary_braced_frame

Rules:
- "서울" or "Seoul" → region = "서울"
- "1-3층 사무실, 4층 식당" → first 3 = office, 4th = restaurant, rest = office
- Last story → usage = "roof" (unless user says otherwise)
- Default material: SS275
- Default site_class: S3
- Default importance: II
- Default dead_load_finish: 1.0 kN/m²
- Use "ordinary_moment_frame" for steel, "rc_ordinary_moment_frame" for RC
- Respond ONLY with the JSON, nothing else
"""


# ================================================================
# 의도 감지
# ================================================================

CONFIRMATION_KEYWORDS = [
    "확인", "완료", "confirm", "ok", "좋아", "괜찮", "진행",
    "네", "예", "ㅇㅇ", "go", "다음", "next", "proceed",
    "맞아", "그대로", "이대로",
]

MODIFICATION_KEYWORDS = [
    "수정", "변경", "바꿔", "change", "modify", "update",
    "추가", "제거", "삭제", "add", "remove", "delete",
    "늘려", "줄여", "높이", "층을", "경간", "bay",
]


def detect_intent(message: str) -> str:
    """사용자 메시지에서 의도 감지.

    Returns:
        "confirm" - 현재 상태 확인/승인
        "modify" - 수정 요청
        "unclear" - 판단 불가 (수정으로 처리)
    """
    msg_lower = message.lower().strip()

    # 수정 키워드 우선 (수정 요청이 확인보다 우선)
    for kw in MODIFICATION_KEYWORDS:
        if kw in msg_lower:
            return "modify"

    for kw in CONFIRMATION_KEYWORDS:
        if kw in msg_lower:
            return "confirm"

    return "unclear"
