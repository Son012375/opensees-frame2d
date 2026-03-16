"""Normalize Agent - 추출된 데이터 정규화."""

import json
from pathlib import Path
from uuid import uuid4
from .base_agent import BaseAgent


class NormalizeAgent(BaseAgent):
    """추출된 데이터를 정규화하는 Agent."""

    def __init__(self):
        super().__init__("NormalizeAgent")
        self.occupancy_mapping = self._load_occupancy_mapping()

    def _load_occupancy_mapping(self) -> dict:
        """용도 매핑 로드."""
        mapping_path = Path(__file__).parent.parent / "data" / "mapping" / "occupancy.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 기본 매핑
        return {
            "office": ["사무실", "업무시설", "사무소", "오피스", "업무용"],
            "residential": ["주거", "주택", "아파트", "공동주택", "단독주택", "기숙사"],
            "retail": ["소매점", "상점", "판매시설", "매장", "점포", "백화점"],
            "assembly": ["집회", "집회시설", "공연장", "극장", "강당", "회의실"],
            "storage": ["창고", "창고시설", "저장", "물류", "적재"],
            "parking": ["주차장", "주차시설", "주차", "차고"],
            "hospital": ["병원", "의료시설", "병실", "진료", "수술실"],
            "school": ["학교", "교육시설", "학원", "강의실", "교실"],
            "restaurant": ["식당", "음식점", "식음료", "주방", "레스토랑"],
            "library": ["도서관", "열람실", "서고"],
            "corridor": ["복도", "계단", "통로", "로비"],
            "balcony": ["발코니", "옥상", "테라스"],
        }

    def get_system_prompt(self) -> str:
        mapping_str = json.dumps(self.occupancy_mapping, ensure_ascii=False, indent=2)

        return f"""You are a data normalization specialist for Korean Building Code parameters.

Your task is to:
1. Map Korean occupancy/usage names to standardized keys
2. Ensure units are in kN/m² format
3. Attach source metadata

## Occupancy Mapping
{mapping_str}

## Unit Conversions
- kgf/m² → kN/m² (multiply by 0.00981)
- tf/m² → kN/m² (multiply by 9.81)
- kPa = kN/m² (1:1)

## Output Format
Return ONLY a JSON array:
[
  {{
    "record_id": "uuid",
    "record_type": "live_load",
    "primary_key": "office",
    "display_name_ko": "사무실",
    "value": 2.5,
    "unit": "kN/m²",
    "load_subtype": "distributed",
    "source": {{
      "code_id": "KDS 41 12 00",
      "code_version": "2022-10-11",
      "clause_id": "3.1.1",
      "table_id": "표 3.1-1"
    }},
    "confidence": 0.95,
    "mapping_method": "exact",
    "needs_review": false
  }}
]

## Mapping Methods
- "exact": display name matches exactly
- "alias": matches known alias
- "fuzzy": similar but not exact (set needs_review: true)
- "unknown": no match found (set needs_review: true)

Return ONLY valid JSON array."""

    def process(self, input_data: dict) -> list:
        """추출된 표 데이터를 정규화.

        Args:
            input_data: {
                "tables": [표 추출 결과들],
                "source": 문서 메타데이터
            }

        Returns:
            정규화된 레코드 배열
        """
        tables = input_data.get("tables", [])
        source = input_data.get("source", {})

        self.log(f"정규화 시작: {len(tables)}개 표")

        all_records = []

        for table in tables:
            if "error" in table:
                continue

            # LLM으로 정규화
            user_prompt = f"""다음 표 데이터를 정규화해주세요.

## 출처 정보
- 문서: {source.get('code_id', 'Unknown')} ({source.get('version_date', 'Unknown')})
- 표: {table.get('table_id', 'Unknown')}

## 표 데이터
{json.dumps(table, ensure_ascii=False, indent=2)}

---
위 데이터의 각 행을 정규화된 레코드로 변환하세요.
각 레코드에 고유한 record_id (UUID)를 생성하세요.
JSON 배열만 반환하세요."""

            result = self.call_llm(user_prompt)

            if isinstance(result, list):
                all_records.extend(result)
            elif isinstance(result, dict) and "records" in result:
                all_records.extend(result["records"])

        self.log(f"완료: {len(all_records)}개 레코드 정규화됨")

        return all_records
