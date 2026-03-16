"""Document Agent - PDF 문서 메타데이터 추출."""

import fitz  # PyMuPDF
from uuid import uuid4
from .base_agent import BaseAgent


class DocumentAgent(BaseAgent):
    """PDF에서 KDS 문서 메타데이터를 추출하는 Agent."""

    def __init__(self):
        super().__init__("DocumentAgent")

    def get_system_prompt(self) -> str:
        return """You are a Korean Building Code (KDS) document analyzer.

Your task is to extract metadata from KDS PDF documents:
1. KDS code number (e.g., "KDS 41 12 00")
2. Document title in Korean
3. Version date (개정일 or 시행일)
4. List of tables with their IDs, titles, and page numbers

## Output Format
Return ONLY a JSON object (no markdown, no explanation):
{
  "code_id": "KDS XX XX XX",
  "code_title": "문서 제목",
  "version_date": "YYYY-MM-DD",
  "sections": [
    {"id": "3.1", "title": "절 제목", "level": 1}
  ],
  "tables": [
    {"table_id": "표 X.X-X", "title": "표 제목", "page": 10, "section_id": "3.1"}
  ]
}

## Important
- Table IDs follow pattern "표 X.X-X" (e.g., 표 3.1-1)
- Page numbers should be actual PDF page numbers
- If unsure about a value, use null
- Return ONLY valid JSON, no other text"""

    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 15) -> str:
        """PDF에서 텍스트 추출."""
        self.log(f"PDF 열기: {pdf_path}")
        doc = fitz.open(pdf_path)

        text_parts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text_parts.append(f"--- 페이지 {i+1} ---\n{page.get_text()}")

        doc.close()
        return "\n".join(text_parts)

    def process(self, input_data: dict) -> dict:
        """PDF에서 메타데이터 추출.

        Args:
            input_data: {"file_path": "path/to/pdf"}

        Returns:
            문서 메타데이터 (code_id, tables, 등)
        """
        pdf_path = input_data["file_path"]

        # PDF 텍스트 추출
        self.log("텍스트 추출 중...")
        pdf_text = self.extract_text_from_pdf(pdf_path)

        # LLM으로 메타데이터 추출
        self.log("Claude로 메타데이터 분석 중...")
        user_prompt = f"""다음 KDS 문서에서 메타데이터를 추출해주세요.

## 문서 내용
{pdf_text}

---
위 문서에서 다음을 JSON으로 추출:
1. KDS 코드 번호 (code_id)
2. 문서 제목 (code_title)
3. 개정일/시행일 (version_date, YYYY-MM-DD 형식)
4. 주요 절 목록 (sections)
5. 표 목록 (tables) - 표 번호, 제목, 페이지 번호

JSON만 반환하세요."""

        result = self.call_llm(user_prompt)

        # 결과에 메타 추가
        result["document_id"] = str(uuid4())
        result["file_path"] = pdf_path

        self.log(f"완료: {result.get('code_id', 'Unknown')} - 표 {len(result.get('tables', []))}개 발견")

        return result
