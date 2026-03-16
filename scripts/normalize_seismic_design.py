#!/usr/bin/env python
"""KDS 41 17 00 내진설계 데이터 정규화 스크립트.

5개 추출 파일을 통합하여 Supabase load_params 테이블 적재용
정규화 JSON을 생성한다.

내진설계 데이터 특성:
- 단일값 레코드 (IE, Fa, Fv, Cu, drift): value 필드 사용
- 다중값 레코드 (R/Ω₀/Cd, ap/Rp/ω₀p): conditions JSON에 통합 저장
- 범주형 레코드 (설계범주, 비정형성): conditions JSON에 정의 저장
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
    "seismic_basic_params.json",
    "seismic_design_category.json",
    "seismic_system_coefficients.json",
    "seismic_analysis_params.json",
    "seismic_nonstructural.json",
]


def normalize_record(record: dict) -> dict:
    """추출 레코드를 load_params 테이블 형식으로 정규화."""
    source = record.get("source", {})
    subtype = record.get("param_subtype", "")

    # 기본 구조
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
        "code_id": source.get("code_id", "KDS 41 17 00"),
        "code_version": source.get("code_version", "2019-11-18"),
        "clause_id": source.get("clause_id"),
        "table_id": source.get("table_id"),
        "confidence": record.get("confidence", 1.0),
        "needs_review": record.get("needs_review", False),
    }

    # === 다중값 레코드 처리 ===

    # 지진력저항시스템 (R, Ω₀, Cd, 높이제한)
    if subtype == "seismic_force_resisting_system":
        extra = {}
        if record.get("R") is not None:
            extra["R"] = record["R"]
        if record.get("omega_0") is not None:
            extra["omega_0"] = record["omega_0"]
        if record.get("Cd") is not None:
            extra["Cd"] = record["Cd"]
        if record.get("height_limit_AB") is not None:
            extra["height_limit_AB_m"] = record["height_limit_AB"]
        if record.get("height_limit_C") is not None:
            extra["height_limit_C_m"] = record["height_limit_C"]
        if record.get("height_limit_D") is not None:
            extra["height_limit_D_m"] = record["height_limit_D"]

        # R을 대표값으로 저장
        normalized["value"] = record.get("R")
        # conditions에 모든 계수를 추가
        conds = list(normalized.get("conditions") or [])
        for k, v in extra.items():
            conds.append(f"{k}={v}")

        # 카테고리 정보 추가
        sec_key = record.get("secondary_key", "")
        normalized["conditions"] = conds

    # 비구조요소 계수 (ap, Rp, ω₀p)
    elif subtype in ("nonstructural_arch_coeff", "nonstructural_mep_coeff"):
        extra = {}
        if record.get("ap") is not None:
            extra["ap"] = record["ap"]
        if record.get("Rp") is not None:
            extra["Rp"] = record["Rp"]
        if record.get("omega_0p") is not None:
            extra["omega_0p"] = record["omega_0p"]

        # ap를 대표값으로 저장
        normalized["value"] = record.get("ap")
        conds = list(normalized.get("conditions") or [])
        for k, v in extra.items():
            conds.append(f"{k}={v}")
        if record.get("category_ko"):
            conds.append(f"category={record['category_ko']}")
        normalized["conditions"] = conds

    # 설계범주 (A/B/C/D)
    elif subtype in ("seismic_design_category_Sds", "seismic_design_category_Sd1"):
        dc = record.get("design_category", "")
        conds = list(normalized.get("conditions") or [])
        conds.append(f"design_category={dc}")
        normalized["conditions"] = conds

    # 비정형성 (정의, 관련조항)
    elif subtype in ("plan_irregularity", "vertical_irregularity"):
        conds = list(normalized.get("conditions") or [])
        if record.get("definition"):
            conds.append(f"definition={record['definition']}")
        if record.get("related_clause"):
            conds.append(f"related_clause={','.join(record['related_clause'])}")
        if record.get("applicable_design_category"):
            conds.append(f"applicable_category={','.join(record['applicable_design_category'])}")
        normalized["conditions"] = conds

    # 근사고유주기 계수 (Ct, x)
    elif subtype == "approximate_period_Ct":
        conds = list(normalized.get("conditions") or [])
        if record.get("Ct") is not None:
            conds.append(f"Ct={record['Ct']}")
            normalized["value"] = record["Ct"]
        if record.get("x") is not None:
            conds.append(f"x={record['x']}")
        if record.get("formula"):
            conds.append(f"formula={record['formula']}")
        if record.get("note"):
            normalized["notes"] = record["note"]
        normalized["conditions"] = conds

    # 해석법 (permitted_methods)
    elif subtype == "analysis_method_D":
        conds = list(normalized.get("conditions") or [])
        if record.get("permitted_methods"):
            conds.append(f"permitted_methods={','.join(record['permitted_methods'])}")
        if record.get("description"):
            conds.append(f"description={record['description']}")
        normalized["conditions"] = conds

    return normalized


def validate_record(record: dict) -> list:
    """레코드 검증. 문제 목록 반환."""
    issues = []

    # 필수 필드
    if not record.get("primary_key"):
        issues.append("Missing primary_key")
    if not record.get("param_subtype"):
        issues.append("Missing param_subtype")
    if not record.get("code_id"):
        issues.append("Missing code_id")

    # 값 존재 체크 (일부 레코드는 조건/정의만 있을 수 있음)
    subtype = record.get("param_subtype", "")
    value_subtypes = {
        "importance_factor", "site_coefficient_Fa", "site_coefficient_Fv",
        "period_upper_limit_Cu", "allowable_drift",
    }
    if subtype in value_subtypes and record.get("value") is None:
        issues.append(f"Missing value for {subtype}")

    # R 범위 체크
    if subtype == "seismic_force_resisting_system":
        r_val = record.get("value")
        if r_val is not None and not (1.0 <= r_val <= 10.0):
            issues.append(f"R out of range: {r_val}")

    return issues


def normalize_all():
    """모든 추출 파일을 통합 정규화."""
    all_records = []
    stats = {"total_input": 0, "normalized": 0, "issues": 0}

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
            normalized = normalize_record(record)

            # 검증
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
    print(f"NORMALIZATION SUMMARY")
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
            "source_document": "KDS 41 17 00 건축물 내진설계기준",
            "pipeline_stage": "03_normalized",
            "normalized_at": datetime.now().isoformat(),
            "total_records": len(all_records),
            "loadable_records": loadable_count,
            "needs_review_records": review_count,
            "description": "내진설계기준 파라미터 - 중요도계수, 지반계수, 설계범주, 시스템계수, 비정형성, 해석법, 비구조요소 등",
            "source_tables": "표 2.2-1 ~ 표 18.4-1 (15개 표)",
        },
        "records": all_records,
    }

    output_path = OUTPUT_DIR / "03_seismic_design_normalized.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {output_path}")
    return output


if __name__ == "__main__":
    normalize_all()
