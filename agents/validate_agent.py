"""Validate Agent - 데이터 검증."""

import json
from .base_agent import BaseAgent


class ValidateAgent(BaseAgent):
    """정규화된 데이터를 검증하는 Agent."""

    def __init__(self):
        super().__init__("ValidateAgent")
        self.validation_rules = {
            "live_load": {"min": 0.5, "max": 25.0, "unit": "kN/m²"},
            "dead_load": {"min": 0.1, "max": 50.0, "unit": "kN/m²"},
            "wind_speed": {"min": 20.0, "max": 50.0, "unit": "m/s"},
            "snow_load": {"min": 0.3, "max": 5.0, "unit": "kN/m²"},
        }

    def get_system_prompt(self) -> str:
        rules_str = json.dumps(self.validation_rules, indent=2)

        return f"""You are a data validation specialist for structural engineering load parameters.

Your task is to validate extracted data for:
1. Value range correctness
2. Duplicates (same primary_key + code_version)
3. Missing required fields
4. Logical consistency

## Validation Rules
{rules_str}

## Required Fields
- record_type (NOT NULL)
- primary_key (NOT NULL)
- value (NOT NULL, must be number)
- unit (NOT NULL)
- source.code_id (NOT NULL)

## Output Format
Return ONLY a JSON object:
{{
  "total_records": 50,
  "passed": 45,
  "failed": 2,
  "needs_review": 3,
  "issues": [
    {{
      "severity": "error",
      "issue_type": "out_of_range",
      "record_ids": ["uuid1"],
      "description": "값 30.0은 live_load 범위(0.5~25.0) 초과",
      "suggested_action": "값 확인 필요"
    }}
  ],
  "validated_records": [
    {{ ...record with validation_status added... }}
  ]
}}

## Severity Levels
- error: 적재 불가
- warning: 적재 가능하나 검토 필요
- info: 참고 사항

Return ONLY valid JSON."""

    def process(self, input_data: dict) -> dict:
        """레코드 검증.

        Args:
            input_data: {"records": 정규화된 레코드 배열}

        Returns:
            검증 결과
        """
        records = input_data.get("records", [])

        if not records:
            return {
                "total_records": 0,
                "passed": 0,
                "failed": 0,
                "needs_review": 0,
                "issues": [],
                "validated_records": []
            }

        self.log(f"검증 시작: {len(records)}개 레코드")

        # LLM으로 검증
        user_prompt = f"""다음 레코드들을 검증해주세요.

## 레코드 목록
{json.dumps(records, ensure_ascii=False, indent=2)}

---
각 레코드에 대해:
1. 값 범위 검증
2. 필수 필드 확인
3. 중복 탐지 (같은 primary_key)
4. needs_review가 true인 항목 별도 분류

검증 결과를 JSON으로 반환하세요."""

        result = self.call_llm(user_prompt)

        self.log(f"완료: 통과 {result.get('passed', 0)}, 실패 {result.get('failed', 0)}, 검토필요 {result.get('needs_review', 0)}")

        return result
