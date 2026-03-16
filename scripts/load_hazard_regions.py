#!/usr/bin/env python
"""Hazard Region Loader - 시/군/구별 Sg/V₀ 값을 Supabase hazard_region_values 테이블에 적재.

Usage:
    python scripts/load_hazard_regions.py --mode dry_run --input data/hazard_region_template.json
    python scripts/load_hazard_regions.py --mode upsert --input data/hazard_region_template.json
    python scripts/load_hazard_regions.py --mode upsert --input data/hazard_region_template.json --output data/kds_output/05_hazard_region_result.json
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


def load_template(input_path: str) -> dict:
    """템플릿 JSON 파일 로드."""
    path = Path(input_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "regions" not in data or "metadata" not in data:
        print("[ERROR] Expected format: {metadata: {...}, regions: [...]}")
        sys.exit(1)

    return data


def regions_to_db_records(data: dict) -> list:
    """템플릿의 region 항목들을 hazard_region_values DB 레코드로 변환.

    각 region에서 sg와 v0가 null이 아닌 경우 별도 레코드로 생성.
    """
    meta = data["metadata"]
    records = []

    for region in data["regions"]:
        sido = region["sido"]
        sigungu = region["sigungu"]
        lat = region.get("lat")
        lon = region.get("lon")

        # Sg (지상적설하중)
        if region.get("sg") is not None:
            records.append({
                "hazard_type": "snow_sg",
                "region_sido": sido,
                "region_sigungu": sigungu,
                "value": region["sg"],
                "unit": meta.get("snow_unit", "kN/m2"),
                "lat": lat,
                "lon": lon,
                "source_code": meta.get("source_code", "KDS 41 12 00"),
                "source_ref": meta.get("snow_ref", "그림 4.2-1"),
                "source_version": meta.get("source_version", "2022-10-11"),
                "verified_by": meta.get("verified_by", "midas_gen"),
            })

        # V₀ (기본풍속)
        if region.get("v0") is not None:
            records.append({
                "hazard_type": "wind_v0",
                "region_sido": sido,
                "region_sigungu": sigungu,
                "value": region["v0"],
                "unit": meta.get("wind_unit", "m/s"),
                "lat": lat,
                "lon": lon,
                "source_code": meta.get("source_code", "KDS 41 12 00"),
                "source_ref": meta.get("wind_ref", "그림 5.5-1"),
                "source_version": meta.get("source_version", "2022-10-11"),
                "verified_by": meta.get("verified_by", "midas_gen"),
            })

    return records


def print_summary(regions: list):
    """입력 데이터 요약 출력."""
    total = len(regions)
    sg_filled = sum(1 for r in regions if r.get("sg") is not None)
    v0_filled = sum(1 for r in regions if r.get("v0") is not None)
    sg_null = total - sg_filled
    v0_null = total - v0_filled

    print(f"\n{'=' * 60}")
    print(f"Template Summary: {total} regions")
    print(f"  Sg filled: {sg_filled} / {total}  (empty: {sg_null})")
    print(f"  V0 filled: {v0_filled} / {total}  (empty: {v0_null})")
    print(f"  DB records to create: {sg_filled + v0_filled}")
    print(f"{'=' * 60}")


def dry_run(records: list, output_path: str = None) -> dict:
    """Dry-run: 적재 미리보기."""
    print(f"\n{'=' * 60}")
    print(f"DRY RUN - {len(records)} records to upsert")
    print(f"{'=' * 60}")

    # 유형별 집계
    snow_count = sum(1 for r in records if r["hazard_type"] == "snow_sg")
    wind_count = sum(1 for r in records if r["hazard_type"] == "wind_v0")
    print(f"  snow_sg: {snow_count} records")
    print(f"  wind_v0: {wind_count} records")

    # 샘플 출력
    print(f"\nSample records (first 10):")
    for i, rec in enumerate(records[:10]):
        print(f"  [{i+1}] {rec['hazard_type']} | "
              f"{rec['region_sido']} {rec['region_sigungu']}: "
              f"{rec['value']} {rec['unit']}")

    if len(records) > 10:
        print(f"  ... and {len(records) - 10} more")

    result = {
        "mode": "dry_run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "summary": {
            "snow_sg": snow_count,
            "wind_v0": wind_count,
        },
    }

    if output_path:
        _save_result(result, output_path)

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
    print(f"UPSERT - {len(records)} records into hazard_region_values")
    print(f"{'=' * 60}")

    for i, record in enumerate(records):
        try:
            client.table("hazard_region_values").upsert(
                record,
                on_conflict="hazard_type,region_sido,region_sigungu"
            ).execute()
            inserted += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(records)}] inserted...")
        except Exception as e:
            errors.append({
                "region": f"{record.get('region_sido')} {record.get('region_sigungu')}",
                "hazard_type": record.get("hazard_type"),
                "error": str(e),
            })
            print(f"  [ERROR] {record.get('region_sido')} "
                  f"{record.get('region_sigungu')}: {e}")

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
        _save_result(result, output_path)

    return result


def _save_result(result: dict, output_path: str):
    """결과를 JSON 파일로 저장."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"Result saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load hazard region values (Sg, V₀) into Supabase"
    )
    parser.add_argument("--mode", choices=["dry_run", "upsert"], default="dry_run",
                        help="Execution mode (default: dry_run)")
    parser.add_argument("--input", required=True,
                        help="Input template JSON file path")
    parser.add_argument("--output",
                        help="Output result JSON path "
                             "(default: data/kds_output/05_hazard_region_result.json)")

    args = parser.parse_args()
    output = args.output or "data/kds_output/05_hazard_region_result.json"

    # 템플릿 로드
    data = load_template(args.input)
    regions = data["regions"]
    print(f"Loaded {len(regions)} regions from {args.input}")

    # 요약 출력
    print_summary(regions)

    # DB 레코드 변환 (null 값은 자동으로 건너뜀)
    records = regions_to_db_records(data)

    if not records:
        print("\n[WARN] No filled values found (all sg/v0 are null). Nothing to load.")
        print("       Fill in the template with Midas Gen values first.")
        return

    if args.mode == "dry_run":
        dry_run(records, output)
    else:
        upsert(records, output)


if __name__ == "__main__":
    main()
