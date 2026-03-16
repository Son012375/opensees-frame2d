# KDS 파라미터 DB - 구현 계획

> **버전:** 1.0.0
> **작성일:** 2026-02-10
> **목표:** PDF 1개로 전체 파이프라인 PoC 완료

---

## 1. PoC 대상 문서

### 1.1 PDF 정보

| 항목 | 값 |
|------|-----|
| **파일명** | KDS 41 12 00_건축물 설계하중.pdf |
| **위치** | `C:\Users\youm\Downloads\` |
| **문서 ID** | KDS 41 12 00 |
| **제목** | 건축구조기준 하중 |
| **버전** | 2022-10-11 (예상, 확인 필요) |

### 1.2 주요 추출 대상

| 표 번호 | 제목 | 데이터 유형 |
|---------|------|-------------|
| 표 3.1-1 | 건축물 바닥의 용도별 활하중 | live_load |
| 표 3.1-2 | 특수용도 활하중 | live_load (special) |
| 표 3.2-1 | 고정하중 (구조재료) | dead_load |
| 표 4.x | 하중조합 계수 | combination_factor |

---

## 2. 구현 단계

### Phase 1: 환경 설정 (Day 1)

#### 1.1 의존성 설치

```bash
# PDF 파싱
pip install PyMuPDF pdfplumber camelot-py[cv] tabula-py

# LLM
pip install anthropic

# Supabase
pip install supabase

# 유틸리티
pip install pandas python-dotenv
```

#### 1.2 프로젝트 구조

```
opensees-MCP/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Agent 기본 클래스
│   ├── document_agent.py       # Agent 1
│   ├── table_agent.py          # Agent 2
│   ├── normalize_agent.py      # Agent 3
│   ├── validate_agent.py       # Agent 4
│   └── loader_agent.py         # Agent 5
├── pipeline/
│   ├── __init__.py
│   ├── runner.py               # 파이프라인 실행기
│   └── config.py               # 설정
├── utils/
│   ├── __init__.py
│   ├── pdf_utils.py            # PDF 파싱 유틸
│   ├── supabase_client.py      # Supabase 클라이언트
│   └── prompts.py              # 프롬프트 로더
├── data/
│   ├── mapping/
│   │   ├── occupancy.json      # 용도 매핑
│   │   └── region.json         # 지역 매핑
│   └── output/                 # 추출 결과 저장
├── tests/
│   └── test_agents.py
├── docs/
│   └── kds-db/                 # 설계 문서 (현재 폴더)
└── main.py                     # 진입점
```

#### 1.3 환경 변수

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-xxx
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJxxx
KDS_PDF_PATH=C:\Users\youm\Downloads\KDS 41 12 00_건축물 설계하중.pdf
```

---

### Phase 2: Agent 개별 구현 (Day 2-3)

#### 2.1 Base Agent

```python
# agents/base_agent.py
from abc import ABC, abstractmethod
from anthropic import Anthropic
import json

class BaseAgent(ABC):
    def __init__(self, name: str, model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = Anthropic()

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def process(self, input_data: dict) -> dict:
        pass

    def call_llm(self, user_prompt: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": user_prompt}]
        )
        return json.loads(response.content[0].text)
```

#### 2.2 Document Agent 구현

```python
# agents/document_agent.py
import fitz  # PyMuPDF
from .base_agent import BaseAgent

class DocumentAgent(BaseAgent):
    def __init__(self):
        super().__init__("DocumentAgent")

    def get_system_prompt(self) -> str:
        return """..."""  # 04_PROMPT_TEMPLATES.md에서 로드

    def process(self, input_data: dict) -> dict:
        pdf_path = input_data["file_path"]

        # PDF에서 텍스트 추출
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:10]:  # 처음 10페이지
            text += page.get_text()

        # LLM 호출
        result = self.call_llm(f"문서 내용:\n{text}")

        return {
            "document_id": str(uuid4()),
            "file_path": pdf_path,
            **result
        }
```

#### 2.3 Table Agent 구현

```python
# agents/table_agent.py
import pdfplumber
from .base_agent import BaseAgent

class TableAgent(BaseAgent):
    def __init__(self):
        super().__init__("TableAgent")

    def process(self, input_data: dict) -> dict:
        pdf_path = input_data["file_path"]
        table_meta = input_data["table_meta"]

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[table_meta["page"] - 1]
            tables = page.extract_tables()

            if tables:
                # 가장 큰 표 선택 (휴리스틱)
                table = max(tables, key=lambda t: len(t) * len(t[0]) if t else 0)

                # LLM으로 구조화
                result = self.call_llm(f"표 데이터:\n{table}")
                return result

        return {"error": "표 추출 실패", "confidence": 0.0}
```

---

### Phase 3: 파이프라인 연결 (Day 4)

#### 3.1 Pipeline Runner

```python
# pipeline/runner.py
from agents import DocumentAgent, TableAgent, NormalizeAgent, ValidateAgent, LoaderAgent

class PipelineRunner:
    def __init__(self, config: dict):
        self.config = config
        self.agents = {
            "document": DocumentAgent(),
            "table": TableAgent(),
            "normalize": NormalizeAgent(),
            "validate": ValidateAgent(),
            "loader": LoaderAgent(),
        }

    def run(self, pdf_path: str, mode: str = "dry_run") -> dict:
        results = {}

        # Stage 1: Document
        print("[1/5] 문서 메타데이터 추출...")
        doc_result = self.agents["document"].process({"file_path": pdf_path})
        results["document"] = doc_result

        # Stage 2: Tables
        print("[2/5] 표 추출...")
        table_results = []
        for table_meta in doc_result.get("tables", []):
            table_result = self.agents["table"].process({
                "file_path": pdf_path,
                "table_meta": table_meta
            })
            table_results.append(table_result)
        results["tables"] = table_results

        # Stage 3: Normalize
        print("[3/5] 정규화...")
        normalized = self.agents["normalize"].process({
            "tables": table_results,
            "source": doc_result
        })
        results["normalized"] = normalized

        # Stage 4: Validate
        print("[4/5] 검증...")
        validation = self.agents["validate"].process({
            "records": normalized
        })
        results["validation"] = validation

        # Stage 5: Load
        print(f"[5/5] 적재 ({mode})...")
        load_result = self.agents["loader"].process({
            "records": normalized,
            "validation": validation,
            "mode": mode
        })
        results["load"] = load_result

        return results
```

#### 3.2 실행

```python
# main.py
from pipeline.runner import PipelineRunner
import os

def main():
    pdf_path = os.getenv("KDS_PDF_PATH")

    runner = PipelineRunner({})

    # Dry-run 먼저
    results = runner.run(pdf_path, mode="dry_run")

    print("\n=== Dry-run 결과 ===")
    print(f"문서: {results['document']['code_id']}")
    print(f"표: {len(results['tables'])}개 추출")
    print(f"레코드: {len(results['normalized'])}개 정규화")
    print(f"검증: {results['validation']['passed']}개 통과")

    # 확인 후 실제 적재
    if input("\n실제로 적재하시겠습니까? (y/n): ").lower() == "y":
        results = runner.run(pdf_path, mode="upsert")
        print(f"적재 완료: {results['load']['summary']}")

if __name__ == "__main__":
    main()
```

---

### Phase 4: Supabase 설정 (Day 5)

#### 4.1 테이블 생성

1. Supabase 콘솔 → SQL Editor
2. [03_SUPABASE_SCHEMA.md](./03_SUPABASE_SCHEMA.md)의 CREATE 문 실행
3. RLS 정책 설정

#### 4.2 클라이언트 구현

```python
# utils/supabase_client.py
from supabase import create_client
import os

class SupabaseClient:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.client = create_client(url, key)

    def upsert_load_param(self, record: dict) -> dict:
        return self.client.table("load_params").upsert(
            record,
            on_conflict="param_type,primary_key,secondary_key,code_id,code_version"
        ).execute()

    def get_existing_records(self, code_id: str, code_version: str) -> list:
        return self.client.table("load_params").select("*").eq(
            "code_id", code_id
        ).eq(
            "code_version", code_version
        ).execute().data
```

---

### Phase 5: 테스트 및 개선 (Day 6-7)

#### 5.1 테스트 케이스

```python
# tests/test_agents.py
import pytest

def test_document_agent_extracts_metadata():
    agent = DocumentAgent()
    result = agent.process({"file_path": TEST_PDF_PATH})

    assert result["code_id"] == "KDS 41 12 00"
    assert "tables" in result
    assert len(result["tables"]) > 0

def test_table_agent_extracts_live_load():
    agent = TableAgent()
    result = agent.process({
        "file_path": TEST_PDF_PATH,
        "table_meta": {"table_id": "표 3.1-1", "page": 12}
    })

    assert result["confidence"] > 0.8
    assert len(result["rows"]) > 0

def test_normalize_maps_office():
    agent = NormalizeAgent()
    result = agent.process({
        "tables": [{"cells": [{"raw_value": "사무실"}]}],
        "source": {}
    })

    assert result[0]["primary_key"] == "office"
```

#### 5.2 개선 항목

| 항목 | 설명 | 우선순위 |
|------|------|----------|
| OCR fallback | 표 추출 실패 시 이미지 OCR | Medium |
| Retry 로직 | LLM 호출 실패 시 재시도 | High |
| 병렬 처리 | 여러 표 동시 추출 | Low |
| 리뷰 UI | needs_review 항목 수동 확인 | Medium |

---

## 3. 일정 요약

| Day | 작업 | 산출물 |
|-----|------|--------|
| 1 | 환경 설정, 프로젝트 구조 | `agents/`, `pipeline/`, `.env` |
| 2 | Document Agent, Table Agent | 개별 Agent 동작 확인 |
| 3 | Normalize Agent, Validate Agent, Loader Agent | 전체 Agent 완성 |
| 4 | 파이프라인 연결 | `main.py` 동작 |
| 5 | Supabase 스키마 적용, 연동 | DB 적재 확인 |
| 6-7 | 테스트, 개선, 문서화 | 안정화 |

---

## 4. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| PDF 표 파싱 실패 | 데이터 추출 불가 | OCR fallback, 수동 입력 |
| KDS 문서 구조 변경 | 파싱 로직 수정 필요 | 문서별 파싱 룰 분리 |
| LLM 비용 증가 | 예산 초과 | 캐싱, 배치 처리 |
| Supabase 무료 한도 | 적재 실패 | 로컬 JSON 백업 |

---

## 5. 다음 단계 (PoC 후)

1. **추가 KDS 문서 적용**
   - KDS 41 12 10 (풍하중)
   - KDS 41 12 20 (적설하중)
   - KDS 41 17 00 (내진설계)

2. **RAG 연동**
   - 조항 텍스트 청킹
   - 벡터 임베딩 (OpenAI ada-002)
   - pgvector 저장

3. **자동 업데이트**
   - KCSC 모니터링 (신규 개정판 감지)
   - 변경 diff 생성
   - 알림 시스템

4. **OpenSees-MCP 연동**
   - `analyze_frame_2d`에서 Supabase 조회
   - 자연어 입력 → 하중 파라미터 자동 적용

---

## 6. 관련 문서

- [01_AGENT_ARCHITECTURE.md](./01_AGENT_ARCHITECTURE.md) - Agent 스펙
- [02_DATA_FLOW.md](./02_DATA_FLOW.md) - 데이터 흐름
- [03_SUPABASE_SCHEMA.md](./03_SUPABASE_SCHEMA.md) - DB 스키마
- [04_PROMPT_TEMPLATES.md](./04_PROMPT_TEMPLATES.md) - 프롬프트 템플릿
