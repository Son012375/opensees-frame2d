#!/usr/bin/env python
"""KDS 17 10 00 지진구역 지역매핑 → hazard_region_values 적재 스크립트.

기존 229개 시/군/구 지역 데이터에 seismic_zone 레코드를 추가한다.
표 4.2-1 지진구역 행정구역 매핑 기반:
  - Zone I (z=0.11): 서울/인천/대전/부산/대구/울산/광주/세종 + 경기/충북/충남/경북/경남/전북/전남/강원남부
  - Zone II (z=0.07): 강원 북부, 제주

Usage:
    python scripts/load_seismic_zone_mapping.py --mode dry_run
    python scripts/load_seismic_zone_mapping.py --mode upsert
"""

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

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = BASE_DIR / "data" / "hazard_region_template.json"

# KDS 17 10 00 표 4.2-1 강원 북부 시/군 목록
GANGWON_NORTH = {
    "홍천군", "철원군", "화천군", "횡성군", "평창군",
    "양구군", "양양군", "인제군", "고성군", "춘천시", "속초시",
}

# 강원 남부 시/군 목록 (확인용)
GANGWON_SOUTH = {
    "영월군", "정선군", "삼척시", "강릉시", "동해시", "원주시", "태백시",
}


def determine_zone(sido: str, sigungu: str) -> tuple:
    """시/도, 시/군/구로 지진구역(I/II)과 구역계수(z) 결정.

    Returns:
        (zone, z_value): ("I", 0.11) or ("II", 0.07)
    """
    # 제주 → Zone II
    if "제주" in sido:
        return ("II", 0.07)

    # 강원도 → 북부/남부 구분
    if "강원" in sido:
        if sigungu in GANGWON_NORTH:
            return ("II", 0.07)
        else:
            # 강원 남부 또는 미분류 → Zone I
            return ("I", 0.11)

    # 그 외 전부 Zone I
    return ("I", 0.11)


def build_seismic_zone_records(template_data: dict) -> list:
    """템플릿의 229개 지역에 대해 seismic_zone 레코드 생성."""
    records = []
    zone_stats = {"I": 0, "II": 0}

    for region in template_data["regions"]:
        sido = region["sido"]
        sigungu = region["sigungu"]
        zone, z_val = determine_zone(sido, sigungu)
        zone_stats[zone] += 1

        records.append({
            "hazard_type": "seismic_zone",
            "region_sido": sido,
            "region_sigungu": sigungu,
            "value": z_val,
            "unit": "g",
            "lat": region.get("lat"),
            "lon": region.get("lon"),
            "source_code": "KDS 17 10 00",
            "source_ref": "표 4.2-1, 표 4.2-2",
            "source_version": "2024-01-25",
            "verified_by": "kds_17_10_00_table_4.2-1",
            "notes": f"지진구역 {zone} (z={z_val})",
        })

    print(f"\nZone distribution: Zone I = {zone_stats['I']}, Zone II = {zone_stats['II']}")
    return records


def dry_run(records: list, output_path: Path):
    """Dry-run 미리보기."""
    print(f"\n{'=' * 60}")
    print(f"DRY RUN - {len(records)} seismic_zone records")
    print(f"{'=' * 60}")

    zone_i = [r for r in records if r["value"] == 0.11]
    zone_ii = [r for r in records if r["value"] == 0.07]
    print(f"  Zone I  (z=0.11): {len(zone_i)} regions")
    print(f"  Zone II (z=0.07): {len(zone_ii)} regions")

    print(f"\nZone II regions:")
    for r in zone_ii:
        print(f"  {r['region_sido']} {r['region_sigungu']}")

    print(f"\nSample Zone I (first 10):")
    for r in zone_i[:10]:
        print(f"  {r['region_sido']} {r['region_sigungu']}: z={r['value']}")

    result = {
        "mode": "dry_run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "zone_I_count": len(zone_i),
        "zone_II_count": len(zone_ii),
        "zone_II_regions": [
            f"{r['region_sido']} {r['region_sigungu']}" for r in zone_ii
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResult saved to: {output_path}")


def upsert(records: list, output_path: Path):
    """Supabase 적재."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("[ERROR] SUPABASE_URL and SUPABASE_KEY required")
        sys.exit(1)

    from supabase import create_client
    client = create_client(url, key)

    inserted = 0
    errors = []

    print(f"\n{'=' * 60}")
    print(f"UPSERT - {len(records)} seismic_zone records")
    print(f"{'=' * 60}")

    for i, record in enumerate(records):
        try:
            client.table("hazard_region_values").upsert(
                record,
                on_conflict="hazard_type,region_sido,region_sigungu"
            ).execute()
            inserted += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(records)}] inserted...")
        except Exception as e:
            errors.append({
                "region": f"{record['region_sido']} {record['region_sigungu']}",
                "error": str(e),
            })
            print(f"  [ERROR] {record['region_sido']} {record['region_sigungu']}: {e}")

    print(f"\nInserted: {inserted}")
    if errors:
        print(f"Errors: {len(errors)}")

    result = {
        "mode": "upsert",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "inserted": inserted,
        "errors": errors if errors else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Result saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load seismic zone mapping to Supabase")
    parser.add_argument("--mode", choices=["dry_run", "upsert"], default="dry_run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        BASE_DIR / "data" / "kds_output" / "05_seismic_zone_result.json"
    )

    # 기존 템플릿 로드
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template_data = json.load(f)

    print(f"Loaded {len(template_data['regions'])} regions from template")

    # 지진구역 레코드 생성
    records = build_seismic_zone_records(template_data)

    if args.mode == "dry_run":
        dry_run(records, output_path)
    else:
        upsert(records, output_path)


if __name__ == "__main__":
    main()
