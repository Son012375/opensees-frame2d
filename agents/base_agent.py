"""Base Agent class for KDS parameter extraction pipeline."""

from abc import ABC, abstractmethod
from anthropic import Anthropic
import json
import re
from typing import Any


class BaseAgent(ABC):
    """모든 Agent의 기본 클래스."""

    def __init__(self, name: str, model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = Anthropic()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Agent별 시스템 프롬프트 반환."""
        pass

    @abstractmethod
    def process(self, input_data: dict) -> dict:
        """메인 처리 로직."""
        pass

    def call_llm(self, user_prompt: str, expect_json: bool = True) -> Any:
        """Claude API 호출.

        Args:
            user_prompt: 유저 프롬프트
            expect_json: True면 JSON 파싱, False면 텍스트 반환

        Returns:
            파싱된 JSON 또는 텍스트
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": user_prompt}]
        )

        content = response.content[0].text

        if expect_json:
            return self._parse_json_response(content)
        return content

    def _parse_json_response(self, text: str) -> dict:
        """응답에서 JSON 추출."""
        # JSON 블록 찾기
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1)
        else:
            # { } 블록 찾기
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON 파싱 실패: {e}")
            print(f"원본 텍스트: {text[:500]}...")
            return {"error": "JSON 파싱 실패", "raw": text}

    def log(self, message: str):
        """로깅."""
        print(f"[{self.name}] {message}")
