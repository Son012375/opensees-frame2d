#!/usr/bin/env python
"""KDS 17 10 00 내진설계 일반 데이터 정규화 스크립트.

추출된 seismic_general_kds17.json을 Supabase load_params 테이블 적재용
정규화 JSON으로 변환한다.

특성:
- 성능목표: conditions에 재현주기, 성능수준, 내진등급 저장
- 지진구역: conditions에 구역, 행정구역, regions 목록 저장
- 구역계수/위험도계수: value 필드에 수치값 저장
- 지반분류: conditions에 분류기준 저장
- 스펙트럼 파라미터: value 필드 + formula conditions
- 포락함수: conditions에 tr/tm/td 시간값 저장
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "kds_output" / "02_tables_extracted"
OUTPUT_DIR = BASE_DIR / "data" / "kds_output"

INPUT_FILE = "seismic_general_kds17.json"


def normalize_record(record: dict) -> dict:
    """추출 레코드를 load_params 테이블 형식으로 정규화."""
    source = record.get("source", {})
    subtype = record.get("param_subtype", "")

    normalized = {
        "param_type": "seismic_design",
        "param_subtype": subtype,
        "primary_key": record.get("primary_key", ""),
        "secondary_key": record.get("secondary_key"),
        "display_name_ko": record.get("display_name_ko", ""),
        "display_name_en": record.get("display_name_en"),
        "value": record.get("value"),
        "value_min": record.get("value_min"),
        "value_max": record.get("value_max"),
        "unit": record.get("unit", ""),
        "conditions": record.get("conditions", []),
        "notes": record.get("notes"),
        "code_id": source.get("code_id", "KDS 17 10 00"),
        "code_version": source.get("code_version", "2024-01-25"),
        "clause_id": source.get("clause_id"),
        "table_id": source.get("table_id"),
        "confidence": record.get("confidence", 1.0),
        "needs_review": record.get("needs_review", False),
    }

    # 성능목표 - required_grade를 conditions에 추가
    if subtype == "performance_objective":
        conds = list(normalized.get("conditions") or [])
        if record.get("required_grade"):
            conds.append(f"required_grade={record['required_grade']}")
        normalized["conditions"] = conds

    # 지진구역 - regions 배열을 conditions에 추가
    elif subtype == "seismic_zone":
        conds = list(normalized.get("conditions") or [])
        if record.get("regions"):
            conds.append(f"regions={','.join(record['regions'])}")
        normalized["conditions"] = conds

    # 스펙트럼가속도 수식 - formula를 conditions에 추가
    elif subtype == "spectral_acceleration_formula":
        conds = list(normalized.get("conditions") or [])
        if record.get("formula"):
            conds.append(f"formula={record['formula']}")
        normalized["conditions"] = conds

    # 감쇠보정계수 - formula를 conditions에 추가
    elif subtype == "damping_correction":
        conds = list(normalized.get("conditions") or [])
        if record.get("formula"):
            conds.append(f"formula={record['formula']}")
        normalized["conditions"] = conds

    # 포락함수 지속시간 - tr/tm/td를 conditions에 추가
    elif subtype == "envelope_duration":
        conds = list(normalized.get("conditions") or [])
        if record.get("rise_time_tr") is not None:
            conds.append(f"tr={record['rise_time_tr']}")
        if record.get("strong_motion_tm") is not None:
            conds.append(f"tm={record['strong_motion_tm']}")
        if record.get("decay_time_td") is not None:
            conds.append(f"td={record['decay_time_td']}")
        normalized["conditions"] = conds

    return normalized


def validate_record(record: dict) -> list:
    """레코드 검증."""
    issues = []

    if not record.get("primary_key"):
        issues.append("Missing primary_key")
    if not record.get("param_subtype"):
        issues.append("Missing param_subtype")
    if not record.get("code_id"):
        issues.append("Missing code_id")

    # 값이 있어야 하는 subtype 체크
    value_subtypes = {
        "zone_coefficient", "risk_coefficient", "spectral_parameter",
    }
    subtype = record.get("param_subtype", "")
    if subtype in value_subtypes and record.get("value") is None:
        issues.append(f"Missing value for {subtype}")

    # 구역계수 범위 체크
    if subtype == "zone_coefficient":
        v = record.get("value")
        if v is not None and not (0.01 <= v <= 1.0):
            issues.append(f"Zone coefficient out of range: {v}")

    # 위험도계수 범위 체크
    if subtype == "risk_coefficient":
        v = record.get("value")
        if v is not None and not (0.1 <= v <= 5.0):
            issues.append(f"Risk coefficient out of range: {v}")

    return issues


def normalize_all():
    """추출 파일 정규화."""
    filepath = EXTRACTED_DIR / INPUT_FILE
    if not filepath.exists():
        print(f"[ERROR] {INPUT_FILE} not found at {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    print(f"Input: {len(records)} records from {INPUT_FILE}")

    all_records = []
    stats = {"total_input": len(records), "normalized": 0, "issues": 0}

    for record in records:
        normalized = normalize_record(record)

        issues = validate_record(normalized)
        if issues:
            normalized["needs_review"] = True
            normalized["notes"] = (normalized.get("notes") or "") + f" [검증: {'; '.join(issues)}]"
            stats["issues"] += 1

        all_records.append(normalized)
        stats["normalized"] += 1

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
    print(f"NORMALIZATION SUMMARY (KDS 17 10 00)")
    print(f"{'=' * 60}")
    print(f"Total input records:  {stats['total_input']}")
    print(f"Normalized records:   {stats['normalized']}")
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
            "source_document": "KDS 17 10 00 내진설계 일반",
            "pipeline_stage": "03_normalized",
            "normalized_at": datetime.now().isoformat(),
            "total_records": len(all_records),
            "loadable_records": loadable_count,
            "needs_review_records": review_count,
            "description": "내진설계 일반 - 성능목표, 지진구역, 구역계수, 위험도계수, 지반분류, 스펙트럼 파라미터, 감쇠보정계수, 포락함수",
            "source_tables": "표 4.1-1 ~ 표 4.2-9 (8개 표, Fa/Fv 중복 제외)",
        },
        "records": all_records,
    }

    output_path = OUTPUT_DIR / "03_seismic_general_normalized.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {output_path}")
    return output


if __name__ == "__main__":
    normalize_all()
