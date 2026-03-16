#!/usr/bin/env python
"""KS_MAP 국가지진위험지도 PGA 데이터 → hazard_region_values 적재 스크립트.

KSD.npz (공주대 구조시스템연구실, NEMA 2013 국가지진위험지도 디지털화)에서
229개 시/군/구 좌표에 대한 재현주기별 PGA(%g)를 보간 계산하여 Supabase에 적재.

기능:
  1. 7개 재현주기(50~4800년)에 대한 PGA 신규 적재 (seismic_pga_XXXX)
  2. 기존 seismic_zone 레코드에 실제 PGA 정보 notes 업데이트

Usage:
    python scripts/load_seismic_pga.py --mode dry_run
    python scripts/load_seismic_pga.py --mode upsert
    python scripts/load_seismic_pga.py --mode upsert --update-zone-notes
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = BASE_DIR / "data" / "hazard_region_template.json"
KSD_PATH = Path("C:/Users/youm/Downloads/KS_MAP/KSD.npz")
OUTPUT_DIR = BASE_DIR / "data" / "kds_output"

# 재현주기 정의
RETURN_PERIODS = {
    "0050": {"years": 50, "desc": "10% in 5yr"},
    "0100": {"years": 100, "desc": "10% in 10yr"},
    "0200": {"years": 200, "desc": "10% in 20yr"},
    "0500": {"years": 500, "desc": "10% in 50yr, 기본설계지진(DBE)"},
    "1000": {"years": 1000, "desc": "10% in 100yr"},
    "2400": {"years": 2400, "desc": "10% in 250yr, 최대고려지진(MCE)"},
    "4800": {"years": 4800, "desc": "2% in 100yr"},
}

# 강원 북부 (Zone II)
GANGWON_NORTH = {
    "홍천군", "철원군", "화천군", "횡성군", "평창군",
    "양구군", "양양군", "인제군", "고성군", "춘천시", "속초시",
}


def load_ksd_data(ksd_path: Path) -> dict:
    """KSD.npz 로드."""
    if not ksd_path.exists():
        print(f"[ERROR] KSD.npz not found: {ksd_path}")
        sys.exit(1)

    data = np.load(str(ksd_path), allow_pickle=True)
    print(f"Loaded KSD.npz: {len(data.keys())} arrays")
    return data


def load_regions(template_path: Path) -> tuple:
    """229개 시/군/구 템플릿 로드."""
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    regions = template["regions"]
    coords = np.array([[r["lon"], r["lat"]] for r in regions])
    print(f"Loaded {len(regions)} regions from template")
    return regions, coords


def interpolate_pga(ksd_data, coords: np.ndarray, return_period: str) -> np.ndarray:
    """등고선 포인트에서 좌표별 PGA 보간 (linear + nearest fallback)."""
    key = f"KSD_2013_{return_period}"
    arr = ksd_data[key]
    points = arr[:, :2]  # (lon, lat)
    values = arr[:, 2]   # PGA(%g)

    pga_linear = griddata(points, values, coords, method="linear")
    pga_nearest = griddata(points, values, coords, method="nearest")

    # linear 실패 시 nearest fallback
    result = np.where(np.isnan(pga_linear), pga_nearest, pga_linear)
    fallback_count = np.isnan(pga_linear).sum()

    return result, fallback_count


def determine_zone(sido: str, sigungu: str) -> tuple:
    """지진구역 판별."""
    if "제주" in sido:
        return ("II", 0.07)
    if "강원" in sido:
        if sigungu in GANGWON_NORTH:
            return ("II", 0.07)
        return ("I", 0.11)
    return ("I", 0.11)


def build_pga_records(regions: list, coords: np.ndarray, ksd_data) -> list:
    """7개 재현주기에 대한 PGA 레코드 생성."""
    records = []
    stats = {}

    for rp_code, rp_info in RETURN_PERIODS.items():
        pga_values, fallback_count = interpolate_pga(ksd_data, coords, rp_code)
        hazard_type = f"seismic_pga_{rp_code}"

        stats[hazard_type] = {
            "count": len(regions),
            "fallback": fallback_count,
            "min": round(float(pga_values.min()), 2),
            "max": round(float(pga_values.max()), 2),
            "mean": round(float(pga_values.mean()), 2),
        }

        # fallback 여부를 한 번에 계산
        pga_linear_check = griddata(
            ksd_data[f"KSD_2013_{rp_code}"][:, :2],
            ksd_data[f"KSD_2013_{rp_code}"][:, 2],
            coords, method="linear"
        )
        is_fallback_arr = np.isnan(pga_linear_check)

        for i, region in enumerate(regions):
            pga_pct = round(float(pga_values[i]), 2)
            pga_g = round(pga_pct / 100.0, 4)

            notes = f"재현주기 {rp_info['years']}년 PGA={pga_pct}%g ({pga_g}g)"
            if rp_code == "0500":
                notes += " [기본설계지진 DBE]"
            elif rp_code == "2400":
                notes += " [최대고려지진 MCE]"
            if is_fallback_arr[i]:
                notes += " (nearest fallback)"

            records.append({
                "hazard_type": hazard_type,
                "region_sido": region["sido"],
                "region_sigungu": region["sigungu"],
                "value": pga_pct,
                "unit": "%g",
                "lat": region.get("lat"),
                "lon": region.get("lon"),
                "source_code": "NEMA 2013",
                "source_ref": f"국가지진위험지도 재현주기 {rp_info['years']}년",
                "source_version": "2013",
                "verified_by": "KS_MAP (Kongju Univ. SSL)",
                "notes": notes,
            })

    return records, stats


def build_zone_update_records(regions: list, coords: np.ndarray, ksd_data) -> list:
    """기존 seismic_zone 레코드의 notes를 실제 PGA 정보로 업데이트."""
    pga500, _ = interpolate_pga(ksd_data, coords, "0500")
    pga2400, _ = interpolate_pga(ksd_data, coords, "2400")

    records = []
    for i, region in enumerate(regions):
        sido = region["sido"]
        sigungu = region["sigungu"]
        zone, z_val = determine_zone(sido, sigungu)

        pga500_g = round(float(pga500[i]) / 100.0, 4)
        pga2400_g = round(float(pga2400[i]) / 100.0, 4)

        notes = (
            f"지진구역 {zone} (z={z_val}), "
            f"실제PGA: 500yr={pga500_g}g, 2400yr={pga2400_g}g "
            f"(NEMA 2013 via KS_MAP)"
        )

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
            "verified_by": "kds_17_10_00_table_4.2-1 + KS_MAP",
            "notes": notes,
        })

    return records


def dry_run(pga_records: list, zone_records: list, stats: dict, output_path: Path):
    """Dry-run 미리보기."""
    print(f"\n{'=' * 70}")
    print(f"DRY RUN - PGA 적재 미리보기")
    print(f"{'=' * 70}")

    print(f"\n[신규 PGA 레코드: {len(pga_records)}건]")
    for ht, s in stats.items():
        print(f"  {ht}: {s['count']}건, "
              f"PGA={s['min']}~{s['max']}%g (mean={s['mean']}), "
              f"nearest_fallback={s['fallback']}건")

    print(f"\n[기존 seismic_zone 업데이트: {len(zone_records)}건]")

    # 샘플 출력 - PGA
    print(f"\nPGA 샘플 (서울 종로구, 7개 재현주기):")
    for rec in pga_records:
        if rec["region_sigungu"] == "종로구":
            print(f"  {rec['hazard_type']}: {rec['value']}%g - {rec['notes']}")

    # 샘플 출력 - Zone 업데이트
    print(f"\nZone 업데이트 샘플 (첫 5건):")
    for rec in zone_records[:5]:
        print(f"  {rec['region_sido']} {rec['region_sigungu']}: {rec['notes']}")

    result = {
        "mode": "dry_run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pga_records": len(pga_records),
        "zone_update_records": len(zone_records),
        "stats": stats,
        "sample_pga": [r for r in pga_records if r["region_sigungu"] == "종로구"],
        "sample_zone": zone_records[:5],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResult saved to: {output_path}")


def upsert(pga_records: list, zone_records: list, stats: dict,
           update_zone: bool, output_path: Path):
    """Supabase 적재."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("[ERROR] SUPABASE_URL and SUPABASE_KEY required")
        sys.exit(1)

    from supabase import create_client
    client = create_client(url, key)

    inserted = 0
    updated = 0
    errors = []

    # 1. PGA 레코드 적재
    print(f"\n{'=' * 70}")
    print(f"UPSERT - {len(pga_records)} PGA records")
    print(f"{'=' * 70}")

    batch_size = 50
    for i in range(0, len(pga_records), batch_size):
        batch = pga_records[i:i + batch_size]
        try:
            client.table("hazard_region_values").upsert(
                batch,
                on_conflict="hazard_type,region_sido,region_sigungu"
            ).execute()
            inserted += len(batch)
            print(f"  [{min(i + batch_size, len(pga_records))}/{len(pga_records)}] "
                  f"PGA records inserted...")
        except Exception as e:
            # batch 실패 시 개별 처리
            for record in batch:
                try:
                    client.table("hazard_region_values").upsert(
                        record,
                        on_conflict="hazard_type,region_sido,region_sigungu"
                    ).execute()
                    inserted += 1
                except Exception as e2:
                    errors.append({
                        "hazard_type": record["hazard_type"],
                        "region": f"{record['region_sido']} {record['region_sigungu']}",
                        "error": str(e2),
                    })
                    print(f"  [ERROR] {record['hazard_type']} "
                          f"{record['region_sido']} {record['region_sigungu']}: {e2}")

    print(f"PGA inserted: {inserted}")

    # 2. Zone notes 업데이트
    if update_zone and zone_records:
        print(f"\n{'=' * 70}")
        print(f"UPDATE - {len(zone_records)} seismic_zone notes")
        print(f"{'=' * 70}")

        for i in range(0, len(zone_records), batch_size):
            batch = zone_records[i:i + batch_size]
            try:
                client.table("hazard_region_values").upsert(
                    batch,
                    on_conflict="hazard_type,region_sido,region_sigungu"
                ).execute()
                updated += len(batch)
                print(f"  [{min(i + batch_size, len(zone_records))}/{len(zone_records)}] "
                      f"zone records updated...")
            except Exception as e:
                for record in batch:
                    try:
                        client.table("hazard_region_values").upsert(
                            record,
                            on_conflict="hazard_type,region_sido,region_sigungu"
                        ).execute()
                        updated += 1
                    except Exception as e2:
                        errors.append({
                            "hazard_type": "seismic_zone",
                            "region": f"{record['region_sido']} {record['region_sigungu']}",
                            "error": str(e2),
                        })

        print(f"Zone updated: {updated}")

    # 결과 저장
    result = {
        "mode": "upsert",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pga_records": len(pga_records),
        "pga_inserted": inserted,
        "zone_updated": updated,
        "stats": stats,
        "errors": errors if errors else None,
    }

    if errors:
        print(f"\nTotal errors: {len(errors)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResult saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load KS_MAP seismic PGA data to Supabase"
    )
    parser.add_argument("--mode", choices=["dry_run", "upsert"], default="dry_run")
    parser.add_argument("--ksd", default=str(KSD_PATH),
                        help=f"Path to KSD.npz (default: {KSD_PATH})")
    parser.add_argument("--update-zone-notes", action="store_true",
                        help="Also update existing seismic_zone notes with actual PGA")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        OUTPUT_DIR / "05_seismic_pga_result.json"
    )

    # 데이터 로드
    ksd_data = load_ksd_data(Path(args.ksd))
    regions, coords = load_regions(TEMPLATE_PATH)

    # PGA 레코드 생성
    print("\nInterpolating PGA for all return periods...")
    pga_records, stats = build_pga_records(regions, coords, ksd_data)
    print(f"Generated {len(pga_records)} PGA records")

    # Zone 업데이트 레코드 생성
    zone_records = build_zone_update_records(regions, coords, ksd_data)
    print(f"Generated {len(zone_records)} zone update records")

    if args.mode == "dry_run":
        dry_run(pga_records, zone_records, stats, output_path)
    else:
        upsert(pga_records, zone_records, stats,
               args.update_zone_notes, output_path)


if __name__ == "__main__":
    main()
