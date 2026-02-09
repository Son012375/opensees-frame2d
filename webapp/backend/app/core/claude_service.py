"""
Claude API Service for natural language to Frame2D input conversion
"""
from __future__ import annotations
import os
import json
import re
from typing import Optional, Dict, Any

import anthropic

# Claude API key from environment
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# System prompt for Frame2D input conversion
SYSTEM_PROMPT = """당신은 구조공학 전문가입니다. 사용자의 자연어 설명을 2D 프레임 해석 입력 JSON으로 변환합니다.

## 출력 형식 (JSON)
```json
{
  "stories": [3.5, 3.2, 3.2],  // 층고 배열 (m), 아래층부터
  "bays": [6.0, 6.0],          // 경간 배열 (m), 왼쪽부터
  "column_section": "H-300x300",  // 기둥 단면
  "beam_section": "H-400x200",    // 보 단면
  "material_name": "SS275",       // 재료
  "supports": "fixed",            // fixed 또는 pinned
  "num_elements_per_member": 4,
  "load_cases": {
    "DL": [
      {"type": "floor", "story": 1, "value": 20},  // 바닥하중 kN/m
      {"type": "floor", "story": 2, "value": 20}
    ],
    "EQX": [
      {"type": "lateral", "story": 1, "fx": 30},   // 횡하중 kN
      {"type": "lateral", "story": 2, "fx": 60}
    ]
  },
  "load_combinations": {
    "1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0}
  }
}
```

## 규칙
1. stories: 최대 10층, 각 층고 0~20m
2. bays: 최대 5경간, 각 경간 0~30m
3. 단면 옵션: H-250x250, H-300x300, H-350x350, H-400x400 (기둥) / H-300x150, H-350x175, H-400x200, H-500x200 (보)
4. 재료 옵션: SS275, SS400, SM490
5. 하중 타입: floor (바닥분포하중), lateral (횡하중)
6. 사용자가 명시하지 않은 값은 합리적인 기본값 사용
7. 반드시 유효한 JSON만 출력 (설명 없이)

## 예시
입력: "3층 2경간 건물, 층고 3.5m, 경간 6m, 바닥하중 25kN/m"
출력: {"stories":[3.5,3.5,3.5],"bays":[6.0,6.0],"column_section":"H-300x300","beam_section":"H-400x200","material_name":"SS275","supports":"fixed","num_elements_per_member":4,"load_cases":{"DL":[{"type":"floor","story":1,"value":25},{"type":"floor","story":2,"value":25},{"type":"floor","story":3,"value":25}]},"load_combinations":null}
"""


def parse_natural_language(user_input: str) -> Dict[str, Any]:
    """
    Convert natural language description to Frame2D input JSON using Claude API

    Args:
        user_input: Natural language description in Korean or English

    Returns:
        Dictionary with Frame2D input parameters

    Raises:
        ValueError: If API key not set or parsing fails
    """
    if not CLAUDE_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    # Extract JSON from response
    response_text = message.content[0].text.strip()

    # Try to find JSON in response (in case there's extra text)
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
    else:
        json_str = response_text

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n응답: {response_text}")

    # Validate required fields
    required_fields = ["stories", "bays"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"필수 필드 누락: {field}")

    # Set defaults for optional fields
    defaults = {
        "column_section": "H-300x300",
        "beam_section": "H-400x200",
        "material_name": "SS275",
        "supports": "fixed",
        "num_elements_per_member": 4,
        "load_cases": {"DL": []},
        "load_combinations": None
    }

    for key, default_value in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_value

    return result


def check_api_key() -> bool:
    """Check if Claude API key is configured"""
    return bool(CLAUDE_API_KEY)
