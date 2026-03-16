"""Loader Agent - Supabase 적재."""

import json
import os
from datetime import datetime
from .base_agent import BaseAgent


class LoaderAgent(BaseAgent):
    """검증된 데이터를 Supabase에 적재하는 Agent."""

    def __init__(self):
        super().__init__("LoaderAgent")
        self.supabase = None
        self._init_supabase()

    def _init_supabase(self):
        """Supabase 클라이언트 초기화."""
        try:
            from supabase import create_client

            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")

            if url and key:
                self.supabase = create_client(url, key)
                self.log("Supabase 연결됨")
            else:
                self.log("Supabase 환경변수 없음 - dry_run 모드만 가능")
        except ImportError:
            self.log("supabase 패키지 없음 - dry_run 모드만 가능")

    def get_system_prompt(self) -> str:
        return """You are a database loading specialist.

Your task is to prepare data for Supabase insertion:
1. Format records for the load_params table
2. Generate upsert operations
3. Ensure data integrity

## load_params Table Schema
- id: UUID (auto-generated)
- param_type: VARCHAR (live_load, dead_load, etc.)
- primary_key: VARCHAR (office, residential, etc.)
- display_name_ko: VARCHAR
- value: DECIMAL
- unit: VARCHAR
- code_id: VARCHAR
- code_version: DATE
- clause_id: VARCHAR
- table_id: VARCHAR
- confidence: DECIMAL
- needs_review: BOOLEAN
- created_at: TIMESTAMPTZ

## Output Format
Return ONLY a JSON object:
{
  "mode": "dry_run",
  "operations": [
    {
      "operation": "insert",
      "table": "load_params",
      "data": { ...record... }
    }
  ],
  "summary": {
    "insert": 10,
    "update": 0,
    "skip": 2
  }
}

Return ONLY valid JSON."""

    def process(self, input_data: dict) -> dict:
        """데이터 적재.

        Args:
            input_data: {
                "records": 검증된 레코드들,
                "validation": 검증 결과,
                "mode": "dry_run" | "upsert"
            }

        Returns:
            적재 결과
        """
        records = input_data.get("records", [])
        validation = input_data.get("validation", {})
        mode = input_data.get("mode", "dry_run")

        # 검증 통과한 레코드만
        validated_records = validation.get("validated_records", records)

        # 실패하거나 검토 필요한 레코드 제외
        issues = validation.get("issues", [])
        failed_ids = set()
        for issue in issues:
            if issue.get("severity") == "error":
                failed_ids.update(issue.get("record_ids", []))

        loadable_records = [
            r for r in validated_records
            if r.get("record_id") not in failed_ids
            and not r.get("needs_review", False)
        ]

        self.log(f"적재 대상: {len(loadable_records)}개 레코드 (모드: {mode})")

        if mode == "dry_run":
            return self._dry_run(loadable_records)
        else:
            return self._upsert(loadable_records)

    def _dry_run(self, records: list) -> dict:
        """Dry-run: 적재 미리보기."""
        operations = []
        for record in records:
            db_record = self._to_db_format(record)
            operations.append({
                "operation": "insert",
                "table": "load_params",
                "data": db_record
            })

        return {
            "mode": "dry_run",
            "operations": operations,
            "summary": {
                "insert": len(operations),
                "update": 0,
                "skip": 0
            },
            "message": f"{len(operations)}개 레코드가 적재될 예정입니다."
        }

    def _upsert(self, records: list) -> dict:
        """실제 Supabase 적재."""
        if not self.supabase:
            return {
                "mode": "upsert",
                "error": "Supabase 연결 안됨",
                "summary": {"insert": 0, "update": 0, "skip": len(records)}
            }

        inserted = 0
        errors = []

        for record in records:
            try:
                db_record = self._to_db_format(record)
                self.supabase.table("load_params").upsert(db_record).execute()
                inserted += 1
            except Exception as e:
                errors.append({"record_id": record.get("record_id"), "error": str(e)})

        return {
            "mode": "upsert",
            "summary": {
                "insert": inserted,
                "update": 0,  # upsert이므로 구분 어려움
                "skip": len(errors)
            },
            "errors": errors if errors else None
        }

    def _to_db_format(self, record: dict) -> dict:
        """레코드를 DB 형식으로 변환."""
        source = record.get("source", {})

        return {
            "param_type": record.get("record_type"),
            "primary_key": record.get("primary_key"),
            "display_name_ko": record.get("display_name_ko"),
            "display_name_en": record.get("display_name_en"),
            "value": record.get("value"),
            "unit": record.get("unit", "kN/m²"),
            "code_id": source.get("code_id"),
            "code_version": source.get("code_version"),
            "clause_id": source.get("clause_id"),
            "table_id": source.get("table_id"),
            "confidence": record.get("confidence", 1.0),
            "needs_review": record.get("needs_review", False),
            "created_at": datetime.utcnow().isoformat(),
            "created_by": "agent_pipeline"
        }
