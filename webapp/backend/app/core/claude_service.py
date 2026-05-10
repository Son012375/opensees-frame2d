"""Claude API service: 자연어 → BuildingIntent (V2)."""
from __future__ import annotations
import os
import json
import re
from typing import Dict, Any

import anthropic

# Claude API key from environment
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def check_api_key() -> bool:
    """Check if Claude API key is configured."""
    return bool(CLAUDE_API_KEY)


# Building (3D) — Natural Language → BuildingIntent
# ══════════════════════════════════════════════════════════════════════════

BUILDING_SYSTEM_PROMPT = """당신은 한국 구조공학 전문가입니다. 사용자의 자연어 건물 설명을 BuildingIntent JSON으로 변환합니다.

## 출력 형식 (JSON만 출력, 설명 없이)
```json
{
  "stories": [
    {"floor_start": 1, "floor_end": 1, "usage_raw": "근린생활시설", "height": 4.5},
    {"floor_start": 2, "floor_end": 5, "usage_raw": "사무실", "height": 3.5}
  ],
  "region_raw": "서울 강남",
  "num_bays_x": 3,
  "num_bays_y": 2,
  "bay_widths_x": null,
  "bay_widths_y": null,
  "typical_bay_width": 8.0,
  "column_section": null,
  "beam_section": null,
  "material": null,
  "supports": "fixed",
  "importance": "II"
}
```

## 필드 규칙
1. **stories** (필수): 층 범위 배열. floor_start/floor_end로 범위 지정, usage_raw는 한국어 그대로 유지
   - height: 사용자가 명시한 층고(m). 명시하지 않으면 null (시스템이 기본값 적용)
   - 예: "2~5층 오피스" → {"floor_start": 2, "floor_end": 5, "usage_raw": "오피스", "height": null}
2. **region_raw** (선택): 지역명 한국어 그대로 (예: "서울", "부산 해운대", "대전"). 없으면 null
3. **num_bays_x / num_bays_y** (선택): 경간 수. 없으면 null → 시스템이 기본값 적용
4. **bay_widths_x / bay_widths_y** (선택): 구체적 경간 폭 배열 (m). 없으면 null
5. **typical_bay_width** (선택): 일반적 경간 폭 (m). 없으면 null → 기본 8.0m
6. **column_section / beam_section** (선택): 단면명. 없으면 null
7. **material** (선택): 재료명. 없으면 null (기본 SS275)
8. **supports**: "fixed" 또는 "pinned" (기본 "fixed")
9. **importance**: "특", "I", "II" (기본 "II")

## 용도 한국어 매핑 (usage_raw에 한국어 그대로 입력)
- 사무실, 오피스, 업무시설 → 사무실
- 근생, 근린생활시설, 상가 → 근린생활시설
- 주거, 아파트, 공동주택 → 주거
- 주차장 → 주차장
- 창고 → 창고
- 병원 → 병원
- 학교, 교육시설 → 학교
- 복도, 계단실 → 복도
- 기계실 → 기계실
- 옥상 → 옥상

## 규칙
1. 사용자가 언급하지 않은 값은 null로 설정 (절대 임의로 채우지 않음)
2. 층 범위가 겹치지 않도록 정리
3. 경간 수를 "3×2" 또는 "3경간"으로 표현할 수 있음 → num_bays_x/y로 분리
4. 반드시 유효한 JSON만 출력 (```json 없이, 설명 없이)

## 예시

입력: "서울 강남, 1층 근생, 2~5층 오피스, 3×2 경간"
출력: {"stories":[{"floor_start":1,"floor_end":1,"usage_raw":"근린생활시설","height":null},{"floor_start":2,"floor_end":5,"usage_raw":"사무실","height":null}],"region_raw":"서울 강남","num_bays_x":3,"num_bays_y":2,"bay_widths_x":null,"bay_widths_y":null,"typical_bay_width":null,"column_section":null,"beam_section":null,"material":null,"supports":"fixed","importance":"II"}

입력: "부산, 10층 업무시설, 8m 경간 3개"
출력: {"stories":[{"floor_start":1,"floor_end":10,"usage_raw":"업무시설","height":null}],"region_raw":"부산","num_bays_x":3,"num_bays_y":null,"bay_widths_x":null,"bay_widths_y":null,"typical_bay_width":8.0,"column_section":null,"beam_section":null,"material":null,"supports":"fixed","importance":"II"}

입력: "대전, 1층 주차장 4.5m, 2~3층 사무실, H-350x350 기둥"
출력: {"stories":[{"floor_start":1,"floor_end":1,"usage_raw":"주차장","height":4.5},{"floor_start":2,"floor_end":3,"usage_raw":"사무실","height":null}],"region_raw":"대전","num_bays_x":null,"num_bays_y":null,"bay_widths_x":null,"bay_widths_y":null,"typical_bay_width":null,"column_section":"H-350x350","beam_section":null,"material":null,"supports":"fixed","importance":"II"}
"""


def parse_building(user_input: str) -> Dict[str, Any]:
    """Convert natural language building description to BuildingIntent JSON.

    Args:
        user_input: Korean natural language description

    Returns:
        BuildingIntent dict ready for resolve_building_config()

    Raises:
        ValueError: If API key not set or parsing fails
    """
    if not CLAUDE_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=BUILDING_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    raw_text = message.content[0].text if message.content and message.content[0].text else ""
    response_text = raw_text.strip()
    if not response_text:
        raise ValueError("Claude API 응답이 비어 있습니다.")

    # Extract JSON
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
    else:
        json_str = response_text

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n응답: {response_text}")

    # Validate: stories is required
    if "stories" not in result or not result["stories"]:
        raise ValueError("필수 필드 누락: stories")

    return result
