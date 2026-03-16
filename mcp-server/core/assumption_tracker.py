"""가정 확인 (Assumption Confirmation) 모듈.

BuildingModel + 원본 config를 비교하여 각 파라미터의 출처를 추적한다.
- user_input: 사용자가 config에 명시적으로 제공
- default: BuildingModel/StoryInfo 기본값 사용
- ifc_detected: IFC 파일에서 자동 감지
- calculated: 코드 내 계산식으로 도출 (하드코딩 상수 포함)
- fallback: DB 조회 실패 시 대체값 사용
- llm_inferred: LLM이 자연어에서 추론한 값 (NL 파싱 계층)
- fuzzy_matched: Python fuzzy match로 해석된 값 (지역명 등)
- composite_mapped: 복합 용도 매핑 (예: 근린생활시설 → retail)

Usage:
    from core.assumption_tracker import build_assumption_summary
    assumptions = build_assumption_summary(model, config, loads_summary)
"""
from __future__ import annotations


# ============================================================
# 필드 기본값 정의 (BuildingModel 기본값과 일치해야 함)
# ============================================================

# (category, default_value)
_FIELD_DEFAULTS = {
    "column_section": ("sections", "H-300x300"),
    "beam_x_section": ("sections", "H-400x200"),
    "beam_y_section": ("sections", "H-400x200"),
    "material_name": ("material", "SS275"),
    "supports": ("geometry", "fixed"),
    "region": ("environment", ""),
    "site_class": ("environment", "S3"),
    "importance": ("environment", "II"),
    "seismic_system": ("seismic", "ordinary_moment_frame"),
    "seismic_direction": ("seismic", "both"),
    "exposure_category": ("wind", "B"),
    "num_elements_per_member": ("geometry", 4),
    "auto_combinations": ("loads", True),
    "rigid_diaphragm": ("geometry", False),
    "member_releases": ("geometry", {}),
    "geometric_nonlinearity": ("geometry", "linear"),
}

_STORY_FIELD_DEFAULTS = {
    "usage": ("loads", "office"),
    "dead_load_finish": ("loads", 1.0),
    "slab_thickness": ("geometry", 0.15),
}


# ============================================================
# 메인 함수
# ============================================================

def build_assumption_summary(
    model,
    config: dict,
    loads_summary: dict,
    input_source: str = "json",
    ifc_fields: set | None = None,
    nl_resolution: dict | None = None,
) -> dict:
    """BuildingModel 파라미터 출처 추적 + 경고 + 용도 매핑 추적.

    Args:
        model: BuildingModel 인스턴스
        config: 원본 사용자 config dict
        loads_summary: generate_all_loads()["summary"]
        input_source: "json" 또는 "ifc+json"
        ifc_fields: IFC에서 감지된 필드명 set (input_source="ifc+json"일 때)
        nl_resolution: resolve_building_config()의 resolution_report dict.
            있으면 해당 파라미터의 source를 llm_inferred/fuzzy_matched/composite_mapped로 오버라이드.

    Returns:
        dict with assumptions, warnings, occupancy_trace, input_source, config_keys_provided
    """
    if ifc_fields is None:
        ifc_fields = set()

    # NL resolution에서 출처 오버라이드 맵 구축
    _nl_source_overrides = _build_nl_source_overrides(nl_resolution) if nl_resolution else {}

    assumptions = []
    provided_keys = list(config.keys())

    # ── Top-level 필드 ──
    for field_name, (category, default_val) in _FIELD_DEFAULTS.items():
        actual = getattr(model, field_name, None)

        if field_name in _nl_source_overrides:
            source = _nl_source_overrides[field_name]
        elif field_name in config:
            source = "user_input"
        elif field_name in ifc_fields:
            source = "ifc_detected"
        else:
            source = "default"

        assumptions.append({
            "param": field_name,
            "category": category,
            "value": actual,
            "source": source,
            "default_value": default_val if source != "user_input" else None,
            "note": None,
        })

    # ── Per-story 필드 ──
    raw_stories = config.get("stories", [])
    for i, story in enumerate(model.stories):
        raw = raw_stories[i] if i < len(raw_stories) else {}
        if isinstance(raw, (int, float)):
            raw = {}  # 높이만 제공된 경우

        for field_name, (category, default_val) in _STORY_FIELD_DEFAULTS.items():
            actual = getattr(story, field_name, None)
            story_param_key = f"stories[{i}].{field_name}"

            if story_param_key in _nl_source_overrides:
                source = _nl_source_overrides[story_param_key]
            elif field_name in raw:
                source = "user_input"
            else:
                source = "default"

            # IFC 경로: slab_thickness가 IFC에서 감지될 수 있음
            if source == "default" and field_name == "slab_thickness" and "stories" in ifc_fields:
                source = "ifc_detected"

            assumptions.append({
                "param": f"stories[{i}].{field_name}",
                "category": category,
                "value": actual,
                "source": source,
                "default_value": default_val if source != "user_input" else None,
                "note": None,
            })

    # ── 하드코딩 계산 상수 ──
    assumptions.extend([
        {
            "param": "rc_unit_weight",
            "category": "loads",
            "value": 24.0,
            "source": "calculated",
            "default_value": None,
            "note": "KDS 철근콘크리트 단위중량 (kN/m³)",
        },
        {
            "param": "mep_load",
            "category": "loads",
            "value": 0.5,
            "source": "calculated",
            "default_value": None,
            "note": "설비 하중 경험치 (kN/m²)",
        },
    ])

    # ── 중요도계수 (자동 매핑) ──
    if "importance_factor" not in config:
        assumptions.append({
            "param": "importance_factor",
            "category": "seismic",
            "value": getattr(model, "importance_factor", 1.0),
            "source": "calculated",
            "default_value": None,
            "note": f"importance='{getattr(model, 'importance', 'II')}'에서 자동 매핑",
        })

    warnings = _collect_warnings(model, config)
    occupancy_trace = _trace_occupancy(model, config)
    model_overview = _build_model_overview(model, config, input_source)
    grouped_warnings = _group_warnings(warnings)

    return {
        "assumptions": assumptions,
        "warnings": warnings,
        "grouped_warnings": grouped_warnings,
        "occupancy_trace": occupancy_trace,
        "model_overview": model_overview,
        "loads_info": {
            "load_cases": loads_summary.get("load_cases", []),
            "num_combinations": loads_summary.get("num_combinations", 0),
            "total_weight_kN": loads_summary.get("total_weight_kN", 0),
            "has_seismic": any(c.startswith("EQ") for c in loads_summary.get("load_cases", [])),
            "has_wind": any(c.startswith("W") for c in loads_summary.get("load_cases", [])),
            "auto_combinations": getattr(model, "auto_combinations", True),
        },
        "input_source": input_source,
        "config_keys_provided": provided_keys,
    }


# ============================================================
# NL Resolution 출처 오버라이드
# ============================================================

def _build_nl_source_overrides(nl_resolution: dict) -> dict[str, str]:
    """resolve_building_config의 resolution_report에서 출처 오버라이드 맵을 구축.

    Returns:
        {param_key: source_string} 맵.
        예: {"region": "fuzzy_matched", "stories[0].usage": "composite_mapped"}
    """
    overrides: dict[str, str] = {}

    # 용도 매칭 결과 → per-story usage 출처
    for match in nl_resolution.get("occupancy_matches", []):
        story_idx = match.get("story")
        match_type = match.get("match_type", "")
        if story_idx is not None:
            key = f"stories[{story_idx}].usage"
            if match_type == "composite":
                overrides[key] = "composite_mapped"
            elif match_type in ("exact", "normalized", "substring"):
                overrides[key] = "llm_inferred"

    # 지역 매칭 결과
    region_match = nl_resolution.get("region_match")
    if region_match:
        match_type = region_match.get("match_type", "")
        if match_type in ("exact", "partial"):
            overrides["region"] = "fuzzy_matched"

    # 기본값 적용된 항목
    for default_item in nl_resolution.get("defaults_applied", []):
        param = default_item.get("param", "")
        source = default_item.get("source", "default")
        if source == "llm_inferred":
            overrides[param] = "llm_inferred"

    return overrides


# ============================================================
# 경고 수집
# ============================================================

def _collect_warnings(model, config: dict) -> list[dict]:
    """입력 누락/모호성 경고 생성."""
    warnings = []

    # A01: region 미지정
    if "region" not in config or not config.get("region"):
        warnings.append({
            "code": "A01",
            "severity": "warning",
            "message_ko": "지역(region)이 지정되지 않아 지진/풍하중이 생성되지 않습니다.",
            "message_en": "No region specified — seismic and wind loads will NOT be generated.",
            "param": "region",
            "story": None,
        })

    # A02: story usage 미지정
    raw_stories = config.get("stories", [])
    for i, story in enumerate(model.stories):
        raw = raw_stories[i] if i < len(raw_stories) else {}
        if isinstance(raw, (int, float)):
            raw = {}
        if "usage" not in raw:
            warnings.append({
                "code": "A02",
                "severity": "info",
                "message_ko": f"{story.story}층: 용도 미지정, 'office'로 가정합니다.",
                "message_en": f"Story {story.story}: usage not specified, assumed 'office'.",
                "param": "usage",
                "story": story.story,
            })

    # A03: 모든 층 동일 높이
    heights = [s.height for s in model.stories]
    if len(heights) > 1 and len(set(heights)) == 1:
        warnings.append({
            "code": "A03",
            "severity": "info",
            "message_ko": f"모든 층의 높이가 동일합니다 ({heights[0]}m).",
            "message_en": f"All stories have identical height ({heights[0]}m).",
            "param": "stories",
            "story": None,
        })

    # A04: column_section 미지정
    if "column_section" not in config:
        warnings.append({
            "code": "A04",
            "severity": "caution",
            "message_ko": "기둥 단면 미지정, 기본값 H-300x300 사용.",
            "message_en": "Column section not specified, using default H-300x300.",
            "param": "column_section",
            "story": None,
        })

    # A05: beam_section 미지정
    if "beam_x_section" not in config and "beam_y_section" not in config:
        warnings.append({
            "code": "A05",
            "severity": "caution",
            "message_ko": "보 단면 미지정, 기본값 H-400x200 사용.",
            "message_en": "Beam section not specified, using default H-400x200.",
            "param": "beam_x_section",
            "story": None,
        })

    # A06: site_class 미지정
    if "site_class" not in config:
        warnings.append({
            "code": "A06",
            "severity": "info",
            "message_ko": "지반 분류 미지정, 기본값 S3 사용.",
            "message_en": "Site class not specified, using default S3.",
            "param": "site_class",
            "story": None,
        })

    # A07: seismic_system 미지정
    if "seismic_system" not in config:
        warnings.append({
            "code": "A07",
            "severity": "caution",
            "message_ko": "내진시스템 미지정, 보통모멘트골조(R=3.5) 사용.",
            "message_en": "Seismic system not specified, using ordinary_moment_frame (R=3.5).",
            "param": "seismic_system",
            "story": None,
        })

    # A08: exposure_category 미지정
    if "exposure_category" not in config:
        warnings.append({
            "code": "A08",
            "severity": "info",
            "message_ko": "풍노출 범주 미지정, 기본값 B (교외) 사용.",
            "message_en": "Wind exposure not specified, using B (suburban).",
            "param": "exposure_category",
            "story": None,
        })

    # A09: slab_thickness 미지정
    for i, story in enumerate(model.stories):
        raw = raw_stories[i] if i < len(raw_stories) else {}
        if isinstance(raw, (int, float)):
            raw = {}
        if "slab_thickness" not in raw:
            warnings.append({
                "code": "A09",
                "severity": "info",
                "message_ko": f"{story.story}층: 슬래브 두께 미지정, 150mm 가정.",
                "message_en": f"Story {story.story}: slab thickness not specified, assumed 150mm.",
                "param": "slab_thickness",
                "story": story.story,
            })

    # A10: usage가 매핑에 없음
    from core.load_generator import USAGE_TO_LIVE_LOAD_KEY
    for story in model.stories:
        if story.usage not in USAGE_TO_LIVE_LOAD_KEY:
            warnings.append({
                "code": "A10",
                "severity": "warning",
                "message_ko": f"{story.story}층: 용도 '{story.usage}'가 매핑에 없습니다.",
                "message_en": f"Story {story.story}: usage '{story.usage}' not in occupancy mapping.",
                "param": "usage",
                "story": story.story,
            })

    # A11: importance 미지정
    if "importance" not in config:
        warnings.append({
            "code": "A11",
            "severity": "info",
            "message_ko": "중요도 등급 미지정, II등급(Ie=1.0) 가정.",
            "message_en": "Importance class not specified, assumed II (Ie=1.0).",
            "param": "importance",
            "story": None,
        })

    # A12: stories가 숫자 리스트 (자동 생성)
    if raw_stories and all(isinstance(s, (int, float)) for s in raw_stories):
        warnings.append({
            "code": "A12",
            "severity": "info",
            "message_ko": f"층 정보 자동 생성: {len(raw_stories)}개 층, 높이만 제공됨.",
            "message_en": f"Stories auto-generated: {len(raw_stories)} stories, heights only.",
            "param": "stories",
            "story": None,
        })

    # A13: member_releases 관련 경고
    releases = getattr(model, "member_releases", None) or {}
    if releases:
        has_column = "column" in releases and releases["column"]
        beam_types = [k for k in releases if k != "column" and releases.get(k)]
        if beam_types:
            desc = ", ".join(f"{k}={releases[k]}" for k in beam_types)
            warnings.append({
                "code": "A13",
                "severity": "info",
                "message_ko": f"보 부재 릴리즈 적용: {desc}.",
                "message_en": f"Beam member releases applied: {desc}.",
                "param": "member_releases",
                "story": None,
            })
        if has_column:
            warnings.append({
                "code": "A13a",
                "severity": "warning",
                "message_ko": "기둥 릴리즈는 실험적 기능이며 골조 안정성에 영향을 줄 수 있습니다.",
                "message_en": "Column end release is experimental and may destabilize frame behavior.",
                "param": "member_releases",
                "story": None,
            })

    return warnings


# ============================================================
# 모델 개요 (사람이 읽기 쉬운 요약)
# ============================================================

_SEISMIC_SYSTEM_LABELS = {
    "special_moment_frame": "특수모멘트골조 (R=8)",
    "intermediate_moment_frame": "중간모멘트골조 (R=4.5)",
    "ordinary_moment_frame": "보통모멘트골조 (R=3.5)",
    "rc_special_moment_frame": "RC 특수모멘트골조 (R=8)",
    "rc_intermediate_moment_frame": "RC 중간모멘트골조 (R=7)",
    "rc_ordinary_moment_frame": "RC 보통모멘트골조 (R=8)",
    "rc_special_shear_wall": "RC 특수전단벽 (R=5)",
    "rc_ordinary_shear_wall": "RC 보통전단벽 (R=4)",
    "special_braced_frame": "특수가새골조 (R=8)",
    "ordinary_braced_frame": "보통가새골조 (R=1.5)",
}

_EXPOSURE_LABELS = {
    "A": "A (대도시 중심)",
    "B": "B (교외/주거)",
    "C": "C (개활지)",
    "D": "D (해안)",
}

_IMPORTANCE_LABELS = {
    "특": "특등급 (Ie=1.5)",
    "I": "I등급 (Ie=1.2)",
    "II": "II등급 (Ie=1.0)",
}


def _build_model_overview(model, config: dict, input_source: str) -> dict:
    """핵심 모델 정보를 사람이 바로 이해할 수 있는 형태로 요약."""
    stories = model.stories

    # 층별 용도 요약 (연속된 동일 용도는 묶어서 표시)
    usage_runs = []
    for s in stories:
        if usage_runs and usage_runs[-1]["usage"] == s.usage:
            usage_runs[-1]["to"] = s.story
            usage_runs[-1]["count"] += 1
        else:
            usage_runs.append({
                "usage": s.usage,
                "from": s.story,
                "to": s.story,
                "count": 1,
            })

    usage_summary = []
    for run in usage_runs:
        if run["count"] == 1:
            usage_summary.append(f"{run['from']}층: {run['usage']}")
        else:
            usage_summary.append(
                f"{run['from']}~{run['to']}층: {run['usage']}")

    # 층고 요약
    heights = [s.height for s in stories]
    if len(set(heights)) == 1:
        height_desc = f"전층 {heights[0]}m"
    else:
        height_desc = " / ".join(f"{h}m" for h in heights)

    seismic_sys = getattr(model, "seismic_system", "ordinary_moment_frame")
    exposure = getattr(model, "exposure_category", "B")
    importance = getattr(model, "importance", "II")

    return {
        "num_stories": model.num_stories,
        "total_height_m": model.total_height,
        "height_desc": height_desc,
        "bays_x_count": len(model.bays_x),
        "bays_x_widths": model.bays_x,
        "bays_y_count": len(model.bays_y),
        "bays_y_widths": model.bays_y,
        "floor_area_m2": model.floor_area,
        "usage_summary": usage_summary,
        "region": getattr(model, "region", "") or "(미지정)",
        "site_class": getattr(model, "site_class", "S3"),
        "importance": importance,
        "importance_label": _IMPORTANCE_LABELS.get(importance, importance),
        "seismic_system": seismic_sys,
        "seismic_system_label": _SEISMIC_SYSTEM_LABELS.get(seismic_sys, seismic_sys),
        "exposure_category": exposure,
        "exposure_label": _EXPOSURE_LABELS.get(exposure, exposure),
        "column_section": model.column_section,
        "column_source": "user_input" if "column_section" in config else "default",
        "beam_x_section": model.beam_x_section,
        "beam_y_section": model.beam_y_section,
        "beam_source": "user_input" if ("beam_x_section" in config or "beam_y_section" in config) else "default",
        "material_name": model.material_name,
        "supports": getattr(model, "supports", "fixed"),
        "rigid_diaphragm": getattr(model, "rigid_diaphragm", False),
        "slab_stiffness_modeled": False,  # 현재 미구현
        "wall_stiffness_modeled": False,  # 현재 미구현
        "input_source": input_source,
    }


# ============================================================
# 경고 그룹화
# ============================================================

def _group_warnings(warnings: list[dict]) -> list[dict]:
    """동일 code의 per-story 경고를 묶어서 요약 메시지 생성.

    Returns:
        list of {code, severity, summary_ko, summary_en, count, stories, param}
    """
    from collections import OrderedDict

    groups: dict[str, list[dict]] = OrderedDict()
    for w in warnings:
        key = w["code"]
        if key not in groups:
            groups[key] = []
        groups[key].append(w)

    result = []
    for code, items in groups.items():
        count = len(items)
        severity = items[0]["severity"]
        param = items[0].get("param")

        # story가 있는 항목들의 층 번호 수집
        story_nums = [w["story"] for w in items if w.get("story") is not None]

        if count == 1:
            # 단일 항목 — 원본 메시지 그대로
            result.append({
                "code": code,
                "severity": severity,
                "summary_ko": items[0]["message_ko"],
                "summary_en": items[0]["message_en"],
                "count": 1,
                "stories": story_nums,
                "param": param,
            })
        else:
            # 복수 항목 — 요약 메시지 생성
            if story_nums:
                stories_str = ", ".join(str(s) for s in story_nums)
                summary_ko = _make_grouped_ko(code, count, stories_str, items)
                summary_en = _make_grouped_en(code, count, stories_str, items)
            else:
                summary_ko = items[0]["message_ko"]
                summary_en = items[0]["message_en"]

            result.append({
                "code": code,
                "severity": severity,
                "summary_ko": summary_ko,
                "summary_en": summary_en,
                "count": count,
                "stories": story_nums,
                "param": param,
            })

    return result


def _make_grouped_ko(code: str, count: int, stories_str: str, items: list) -> str:
    """코드별 한국어 그룹 요약 메시지."""
    msg_map = {
        "A02": f"{count}개 층({stories_str}층)에서 용도 미지정 → 'office'로 가정",
        "A09": f"{count}개 층({stories_str}층)에서 슬래브 두께 미지정 → 150mm 가정",
        "A10": f"{count}개 층({stories_str}층)에서 용도가 매핑에 없음",
    }
    return msg_map.get(code, f"{count}개 항목: {items[0]['message_ko']}")


def _make_grouped_en(code: str, count: int, stories_str: str, items: list) -> str:
    """코드별 영어 그룹 요약 메시지."""
    msg_map = {
        "A02": f"{count} stories (F{stories_str}): usage not specified, assumed 'office'",
        "A09": f"{count} stories (F{stories_str}): slab thickness not specified, assumed 150mm",
        "A10": f"{count} stories (F{stories_str}): usage not in occupancy mapping",
    }
    return msg_map.get(code, f"{count} items: {items[0]['message_en']}")


# ============================================================
# 용도 매핑 추적
# ============================================================

def _trace_occupancy(model, config: dict) -> list[dict]:
    """Per-story 활하중 매핑 체인 추적."""
    from core.load_generator import (
        USAGE_TO_LIVE_LOAD_KEY,
        _query_live_load_traced,
    )

    raw_stories = config.get("stories", [])
    traces = []

    for i, story in enumerate(model.stories):
        raw = raw_stories[i] if i < len(raw_stories) else {}
        if isinstance(raw, (int, float)):
            user_input = None
        elif isinstance(raw, dict):
            user_input = raw.get("usage", None)
        else:
            user_input = None

        internal_key = story.usage
        key_info = USAGE_TO_LIVE_LOAD_KEY.get(internal_key)
        if key_info:
            pk, sk = key_info
        else:
            pk, sk = internal_key, None

        # DB 조회 (traced)
        traced = _query_live_load_traced(internal_key)

        traces.append({
            "story": story.story,
            "user_input": user_input if user_input else "(not specified)",
            "internal_key": internal_key,
            "db_primary_key": traced.get("db_primary_key", pk),
            "db_secondary_key": traced.get("db_secondary_key", sk),
            "live_load_kNm2": traced["value"],
            "load_source": traced["source"],
            "kds_ref": "KDS 41 12 00 표 3.1-1",
        })

    return traces
