#!/usr/bin/env python
"""Supabase Loader - KDS 파라미터 데이터를 Supabase에 적재하는 헬퍼 스크립트.

Usage:
    python scripts/supabase_loader.py --mode dry_run --input data/kds_output/03_normalized.json
    python scripts/supabase_loader.py --mode upsert --input data/kds_output/03_normalized.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_records(input_path: str) -> list:
    """JSON 파일에서 레코드 로드."""
    path = Path(input_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 배열이면 직접 반환, 아니면 records 키 확인
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]

    print(f"[ERROR] Unexpected format: expected array or {{records: []}}")
    sys.exit(1)


def to_db_format(record: dict) -> dict:
    """레코드를 Supabase load_params 테이블 형식으로 변환."""
    source = record.get("source", {})

    return {
        "param_type": record.get("record_type") or record.get("param_type"),
        "param_subtype": record.get("load_subtype") or record.get("param_subtype"),
        "primary_key": record.get("primary_key"),
        "secondary_key": record.get("secondary_key"),
        "display_name_ko": record.get("display_name_ko"),
        "display_name_en": record.get("display_name_en"),
        "value": record.get("value"),
        "value_min": record.get("value_min"),
        "value_max": record.get("value_max"),
        "unit": record.get("unit", "kN/m\u00b2"),
        "conditions": record.get("conditions", []),
        "notes": record.get("notes"),
        "code_id": source.get("code_id") or record.get("code_id"),
        "code_version": source.get("code_version") or record.get("code_version"),
        "clause_id": source.get("clause_id") or record.get("clause_id"),
        "table_id": source.get("table_id") or record.get("table_id"),
        "confidence": record.get("confidence", 1.0),
        "needs_review": record.get("needs_review", False),
        "created_by": "agent_team_pipeline",
    }


def dry_run(records: list, output_path: str = None) -> dict:
    """Dry-run: 적재 미리보기."""
    operations = []
    for record in records:
        db_record = to_db_format(record)
        operations.append({
            "operation": "upsert",
            "table": "load_params",
            "data": db_record,
        })

    result = {
        "mode": "dry_run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "operations": operations,
        "summary": {
            "upsert": len(operations),
            "skip": 0,
        },
    }

    # 미리보기 출력
    print(f"\n{'=' * 60}")
    print(f"DRY RUN - {len(operations)} records to upsert")
    print(f"{'=' * 60}")

    for i, op in enumerate(operations[:10]):
        d = op["data"]
        print(f"  [{i+1}] {d.get('param_type')}/{d.get('primary_key')}: "
              f"{d.get('value')} {d.get('unit')} "
              f"(conf: {d.get('confidence')}, review: {d.get('needs_review')})")

    if len(operations) > 10:
        print(f"  ... and {len(operations) - 10} more")

    # 결과 저장
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nPreview saved to: {output_path}")

    return result


def upsert(records: list, output_path: str = None) -> dict:
    """실제 Supabase 적재."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("[ERROR] SUPABASE_URL and SUPABASE_KEY environment variables required")
        sys.exit(1)

    try:
        from supabase import create_client
    except ImportError:
        print("[ERROR] supabase package not installed. Run: pip install supabase")
        sys.exit(1)

    client = create_client(url, key)

    inserted = 0
    errors = []

    print(f"\n{'=' * 60}")
    print(f"UPSERT - {len(records)} records")
    print(f"{'=' * 60}")

    for i, record in enumerate(records):
        try:
            db_record = to_db_format(record)
            client.table("load_params").upsert(
                db_record,
                on_conflict="param_type,param_subtype,primary_key,secondary_key,code_id,code_version"
            ).execute()
            inserted += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(records)}] inserted...")
        except Exception as e:
            errors.append({
                "record": record.get("primary_key", "unknown"),
                "error": str(e),
            })
            print(f"  [ERROR] {record.get('primary_key', '?')}: {e}")

    result = {
        "mode": "upsert",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "summary": {
            "inserted": inserted,
            "errors": len(errors),
        },
        "errors": errors if errors else None,
    }

    print(f"\nInserted: {inserted}")
    if errors:
        print(f"Errors: {len(errors)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"Result saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Load KDS parameters into Supabase")
    parser.add_argument("--mode", choices=["dry_run", "upsert"], default="dry_run",
                        help="Execution mode (default: dry_run)")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", help="Output result JSON path (default: data/kds_output/05_load_result.json)")

    args = parser.parse_args()
    output = args.output or "data/kds_output/05_load_result.json"

    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    # needs_review 필터링
    loadable = [r for r in records if not r.get("needs_review", False)]
    review = [r for r in records if r.get("needs_review", False)]

    if review:
        print(f"Skipping {len(review)} records flagged for review")

    if args.mode == "dry_run":
        dry_run(loadable, output)
    else:
        upsert(loadable, output)


if __name__ == "__main__":
    main()
