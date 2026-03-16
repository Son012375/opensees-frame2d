"""Table Agent - PDF 표에서 데이터 추출."""

import fitz  # PyMuPDF
from .base_agent import BaseAgent


class TableAgent(BaseAgent):
    """PDF 표에서 구조화된 데이터를 추출하는 Agent."""

    def __init__(self):
        super().__init__("TableAgent")

    def get_system_prompt(self) -> str:
        return """You are a table extraction specialist for Korean Building Code (KDS) documents.

Your task is to extract structured data from tables, specifically:
- Load values (활하중, 고정하중)
- Units (kN/m², kPa, etc.)
- Conditions or notes

## Output Format
Return ONLY a JSON object:
{
  "table_id": "표 X.X-X",
  "title": "표 제목",
  "headers": ["용도", "등분포활하중", "집중활하중"],
  "rows": [
    {
      "용도": "사무실",
      "등분포활하중": {"value": 2.5, "unit": "kN/m²"},
      "집중활하중": {"value": 4.5, "unit": "kN"},
      "conditions": []
    }
  ],
  "confidence": 0.95,
  "notes": ["추출 시 발견한 이슈"]
}

## Rules
- Parse numbers correctly (2.5, 2,500, etc.)
- Separate value and unit
- conditions: text in parentheses or footnotes
- confidence: 1.0 (clear) to 0.5 (uncertain)
- Return ONLY valid JSON"""

    def extract_page_text(self, pdf_path: str, page_num: int) -> str:
        """특정 페이지의 텍스트 추출."""
        doc = fitz.open(pdf_path)
        if page_num <= 0 or page_num > len(doc):
            doc.close()
            return ""

        page = doc[page_num - 1]  # 0-indexed
        text = page.get_text()
        doc.close()
        return text

    def process(self, input_data: dict) -> dict:
        """표에서 데이터 추출.

        Args:
            input_data: {
                "file_path": "path/to/pdf",
                "table_meta": {"table_id": "표 3.1-1", "title": "...", "page": 12}
            }

        Returns:
            추출된 표 데이터
        """
        pdf_path = input_data["file_path"]
        table_meta = input_data["table_meta"]
        page_num = table_meta.get("page", 1)

        self.log(f"표 추출: {table_meta.get('table_id', 'Unknown')} (페이지 {page_num})")

        # 해당 페이지 및 전후 페이지 텍스트 추출
        texts = []
        for p in range(max(1, page_num - 1), page_num + 2):
            text = self.extract_page_text(pdf_path, p)
            if text:
                texts.append(f"--- 페이지 {p} ---\n{text}")

        page_text = "\n".join(texts)

        # LLM으로 표 추출
        user_prompt = f"""다음 페이지에서 표 데이터를 추출해주세요.

## 찾아야 할 표
- 표 번호: {table_meta.get('table_id', '알 수 없음')}
- 표 제목: {table_meta.get('title', '알 수 없음')}

## 페이지 내용
{page_text}

---
위 표의 모든 행과 열을 JSON으로 추출하세요.
각 셀에서 숫자 값과 단위를 분리하세요.
JSON만 반환하세요."""

        result = self.call_llm(user_prompt)

        # 메타 추가
        result["source_page"] = page_num
        result["table_meta"] = table_meta

        self.log(f"완료: {len(result.get('rows', []))}개 행 추출, 확신도: {result.get('confidence', 0)}")

        return result
