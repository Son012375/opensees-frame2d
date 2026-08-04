"""cad_parser — 2D CAD 도면(라스터)에서 StructuralModel JSON을 생성.

기존 OpenSees-MCP 프레임워크에 영향을 주지 않는 standalone 패키지.
출력 JSON은 V2 에디터(`📂 Load` 버튼)가 그대로 받는 `.v2proj.json` 포맷.

공개 API
--------
- `parse_drawing_set(...)` (W4에서 구현 완료 예정)
- `preprocess.load_sheet`, `vectorize.extract_line_segments` 등 단계별 함수
"""
from __future__ import annotations

from . import (
    builder,
    fallback,
    grid_detector,
    member_extract,
    preprocess,
    registration,
    schemas,
    vectorize,
)

__all__ = [
    "builder",
    "fallback",
    "grid_detector",
    "member_extract",
    "preprocess",
    "registration",
    "schemas",
    "vectorize",
]
__version__ = "0.4.0-w4"
