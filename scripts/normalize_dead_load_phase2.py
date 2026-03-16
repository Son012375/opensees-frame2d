#!/usr/bin/env python
"""Phase 2 고정하중 데이터 정규화 스크립트.

5개 추출 파일을 통합하고, 단위 변환 및 검증을 수행하여
Supabase 적재용 정규화 JSON을 생성한다.

단위 변환 규칙:
- t/m³ → kN/m³: × 9.81
- kgf/m² → kN/m²: × 0.00981
- kgf/m²/mm → kN/m²/mm: × 0.00981
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "kds_output" / "02_tables_extracted"
OUTPUT_DIR = BASE_DIR / "data" / "kds_output"

# 입력 파일 목록
INPUT_FILES = [
    "material_properties.json",
    "material_weights.json",
    "roof_assemblies.json",
    "ceiling_floor_assemblies.json",
    "wall_railing_fireproofing.json",
]

# Phase 1 기존 데이터 (중복 체크용)
PHASE1_FILE = OUTPUT_DIR / "03_dead_load_normalized.json"


def load_phase1_keys() -> set:
    """Phase 1 기존 데이터의 키 세트 로드."""
    if not PHASE1_FILE.exists():
        return set()
    with open(PHASE1_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data if isinstance(data, list) else [])
    keys = set()
    for r in records:
        pk = r.get("primary_key", "")
        sk = r.get("secondary_key") or ""
        st = r.get("param_subtype") or r.get("load_subtype", "")
        keys.add((st, pk, sk))
    return keys


def normalize_unit_string(unit: str) -> str:
    """단위 문자열 정규화 (인코딩 차이 통일)."""
    # ² = U+00B2, ³ = U+00B3
    replacements = {
        "m\u00b2": "m²", "m\u00b3": "m³",
        "m2": "m²", "m3": "m³",
    }
    for old, new in replacements.items():
        unit = unit.replace(old, new)
    return unit


def get_target_unit(from_unit: str) -> tuple:
    """원본 단위 → (변환 계수, 목표 단위). None이면 변환 불필요."""
    u = normalize_unit_string(from_unit)
    mapping = {
        "t/m³":       (9.81,    "kN/m³"),
        "kgf/m²":     (0.00981, "kN/m²"),
        "kgf/m²/mm":  (0.00981, "kN/m²/mm"),
        "kgf/m":      (0.00981, "kN/m"),
        "kgf/개":     (0.00981, "kN/개"),
        "kgf/m³":     (0.00981, "kN/m³"),
    }
    return mapping.get(u)


def convert_unit(value, from_unit: str) -> tuple:
    """단위 변환. Returns (converted_value, new_unit)."""
    norm = normalize_unit_string(from_unit)

    # 이미 kN 단위면 패스
    if norm.startswith("kN/"):
        return value, norm

    target = get_target_unit(norm)
    if target is None:
        print(f"  [WARN] Unknown unit: {from_unit} (normalized: {norm}), keeping as-is")
        return value, norm

    factor, new_unit = target
    if value is None:
        return None, new_unit
    return round(value * factor, 4), new_unit


def validate_record(record: dict) -> list:
    """레코드 검증. 문제 목록 반환."""
    issues = []

    # 필수 필드 체크
    for field in ["primary_key", "param_subtype", "source"]:
        if not record.get(field):
            issues.append(f"Missing required field: {field}")

    # 값 존재 체크
    if record.get("value") is None and record.get("value_min") is None:
        issues.append("No value or value_min")

    # 범위 체크 (변환 후 kN 단위)
    unit = record.get("unit", "")
    value = record.get("value")

    if value is not None:
        if "kN/m³" in unit and not (0.5 <= value <= 300):
            issues.append(f"Unit weight out of range: {value} {unit}")
        elif "kN/m²" in unit and "/mm" not in unit and not (0.001 <= value <= 50):
            issues.append(f"Area weight out of range: {value} {unit}")

    return issues


def normalize_all():
    """모든 추출 파일을 통합 정규화."""
    phase1_keys = load_phase1_keys()
    print(f"Phase 1 existing keys: {len(phase1_keys)}")

    all_records = []
    stats = {"total_input": 0, "converted": 0, "phase1_overlap": 0, "issues": 0}

    for filename in INPUT_FILES:
        filepath = EXTRACTED_DIR / filename
        if not filepath.exists():
            print(f"[SKIP] {filename} not found")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("records", data if isinstance(data, list) else [])
        print(f"\n--- {filename}: {len(records)} records ---")
        stats["total_input"] += len(records)

        for record in records:
            # record_type 통일
            record["record_type"] = "dead_load"

            # param_subtype 정리 (record_type → param_type 매핑 제거)
            if "record_type" in record and record.get("param_type"):
                del record["param_type"]

            # 단위 변환 - 항상 단위 문자열 업데이트
            orig_unit = record.get("unit", "")
            target = get_target_unit(normalize_unit_string(orig_unit))

            if target:
                factor, new_unit = target
                record["unit"] = new_unit
                if record.get("value") is not None:
                    record["value"] = round(record["value"] * factor, 4)
                if record.get("value_min") is not None:
                    record["value_min"] = round(record["value_min"] * factor, 4)
                if record.get("value_max") is not None:
                    record["value_max"] = round(record["value_max"] * factor, 4)
            else:
                # 이미 kN이거나 알 수 없는 단위
                record["unit"] = normalize_unit_string(orig_unit)

            stats["converted"] += 1

            # Phase 1 중복 체크
            pk = record.get("primary_key", "")
            sk = record.get("secondary_key") or ""
            st = record.get("param_subtype", "")
            if (st, pk, sk) in phase1_keys:
                record["notes"] = (record.get("notes") or "") + " [Phase 1 중복 - 확인 필요]"
                record["needs_review"] = True
                stats["phase1_overlap"] += 1

            # 검증
            issues = validate_record(record)
            if issues:
                record["needs_review"] = True
                record["notes"] = (record.get("notes") or "") + f" [검증: {'; '.join(issues)}]"
                stats["issues"] += 1

            all_records.append(record)

    # 중복 키 탐지
    key_counts = Counter()
    for r in all_records:
        key = (r.get("param_subtype", ""), r.get("primary_key", ""), r.get("secondary_key") or "")
        key_counts[key] += 1

    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    if duplicates:
        print(f"\n[WARN] {len(duplicates)} duplicate keys found:")
        for k, v in list(duplicates.items())[:10]:
            print(f"  {k}: {v} records")

    # 통계
    subtypes = Counter(r["param_subtype"] for r in all_records)
    review_count = sum(1 for r in all_records if r.get("needs_review", False))
    loadable_count = len(all_records) - review_count

    print(f"\n{'=' * 60}")
    print(f"NORMALIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total input records:  {stats['total_input']}")
    print(f"Converted records:    {stats['converted']}")
    print(f"Phase 1 overlaps:     {stats['phase1_overlap']}")
    print(f"Validation issues:    {stats['issues']}")
    print(f"Duplicate keys:       {len(duplicates)}")
    print(f"Loadable (no review): {loadable_count}")
    print(f"Needs review:         {review_count}")
    print(f"\nBy param_subtype:")
    for st, count in sorted(subtypes.items()):
        print(f"  {st}: {count}")

    # 출력
    output = {
        "metadata": {
            "source_document": "건축물의 하중기준 및 해설 (대한건축학회, 2000)",
            "pipeline_stage": "03_normalized",
            "normalized_at": datetime.now().isoformat(),
            "total_records": len(all_records),
            "loadable_records": loadable_count,
            "needs_review_records": review_count,
            "phase1_overlaps": stats["phase1_overlap"],
            "description": "Phase 2 고정하중 - 건축용 재료 밀도/단위중량 + 마감재/조합중량",
            "source_tables": "해표 2.1~2.13",
            "unit_conversions": {
                "t/m³ → kN/m³": "× 9.81",
                "kgf/m² → kN/m²": "× 0.00981"
            }
        },
        "records": all_records
    }

    output_path = OUTPUT_DIR / "03_dead_load_phase2_normalized.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {output_path}")
    return output


if __name__ == "__main__":
    normalize_all()
