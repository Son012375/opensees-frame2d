"""자연어 건물 설계 의도 → 구조 해석 config 변환 모듈.

BuildingIntent (Claude 추출) → BuildingAnalysisInput.config (검증됨)

주요 기능:
1. 용도명 매칭: 한국어 raw text → canonical occupancy key
2. 지역명 매칭: raw region text → Supabase region_sido/sigungu
3. 층 범위 확장: StoryIntent ranges → 개별 StoryInfo
4. 기본값 채움 + 가정 추적
5. 검증 + 경고 생성

Usage:
    from core.nl_resolver import resolve_building_config
    result = resolve_building_config(intent_dict)
    # result["config"] → analyze_building에 전달
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# 데이터 로드
# ============================================================

_OCCUPANCY_MAP: dict[str, str] | None = None
_OCCUPANCY_DATA: dict | None = None


def _load_occupancy_data() -> tuple[dict[str, str], dict]:
    """occupancy.json → (flat_alias_map, raw_data) 로드.

    flat_alias_map: {한국어alias: canonical_key}
    raw_data: 원본 JSON (maps_to, note 등 포함)
    """
    global _OCCUPANCY_MAP, _OCCUPANCY_DATA
    if _OCCUPANCY_MAP is not None:
        return _OCCUPANCY_MAP, _OCCUPANCY_DATA

    json_path = Path(__file__).resolve().parent.parent.parent / "data" / "mapping" / "occupancy.json"
    if not json_path.exists():
        _OCCUPANCY_MAP = {}
        _OCCUPANCY_DATA = {}
        return _OCCUPANCY_MAP, _OCCUPANCY_DATA

    with open(json_path, "r", encoding="utf-8") as f:
        _OCCUPANCY_DATA = json.load(f)

    _OCCUPANCY_MAP = {}
    for canonical_key, entry in _OCCUPANCY_DATA.items():
        # canonical key 자체도 매핑
        _OCCUPANCY_MAP[canonical_key] = canonical_key
        # 영어 이름
        en = entry.get("en", "")
        if en:
            _OCCUPANCY_MAP[en.lower()] = canonical_key
        # 한국어 별칭
        for alias in entry.get("ko", []):
            _OCCUPANCY_MAP[alias] = canonical_key

    return _OCCUPANCY_MAP, _OCCUPANCY_DATA


# ============================================================
# 복합 용도 매핑 (건축법 용어 → 구조 하중 기준)
# ============================================================

_COMPOSITE_USAGE_MAP = {
    "근린생활시설": "retail",
    "근생": "retail",
    "제1종근린생활시설": "retail",
    "제2종근린생활시설": "retail",
    "1종근생": "retail",
    "2종근생": "retail",
    "근린상업": "retail",
    "기계실": "mechanical_room",
    "전기실": "mechanical_room",
    "공조실": "mechanical_room",
    "서버실": "mechanical_room",
    "전산실": "mechanical_room",
    "옥탑": "roof",
    "옥탑층": "roof",
    "펜트하우스": "roof",
    "필로티": "parking",
    "지하주차장": "parking",
    "피난층": "corridor",
    "근무시설": "office",
}

# 용도별 기본 층고 오버라이드 (m)
_USAGE_HEIGHT_DEFAULTS = {
    "mechanical_room": 3.0,
    "roof": 3.0,
    "parking": 3.3,
}

# 기계실 등 특수 용도의 dead_load_finish 기본값 (kN/m²)
_USAGE_DEAD_LOAD_OVERRIDE = {
    "mechanical_room": 2.0,
}


# ============================================================
# 1. 용도명 해석
# ============================================================

def resolve_usage(raw_text: str) -> dict:
    """한국어 용도명 → canonical occupancy key.

    매칭 순서:
    1. exact match (occupancy.json alias)
    2. composite match (_COMPOSITE_USAGE_MAP)
    3. normalized match (접미사 제거 후 재시도)
    4. substring match (alias 포함 관계)
    5. unresolved

    Returns:
        {
            "canonical_key": str,
            "resolved_to": str,      # maps_to 적용 후 최종 키
            "raw_input": str,
            "match_type": "alias_exact"|"composite"|"alias_normalized"|"alias_substring"|"unresolved",
            "confidence": float,
            "note": str | None,
        }
    """
    alias_map, raw_data = _load_occupancy_data()
    text = raw_text.strip()

    # 1) Exact alias match
    if text in alias_map:
        canonical = alias_map[text]
        return _finalize_usage(canonical, text, "alias_exact", 1.0, raw_data)

    # 2) Composite map (건축법 용어)
    if text in _COMPOSITE_USAGE_MAP:
        resolved = _COMPOSITE_USAGE_MAP[text]
        return _finalize_usage(resolved, text, "composite", 0.9, raw_data,
                               note=f"'{text}' → {resolved} (복합 용도 매핑)")

    # 3) Normalized (접미사 제거: 시설, 실, 용)
    normalized = re.sub(r'(시설|실|용)$', '', text).strip()
    if normalized and normalized in alias_map:
        canonical = alias_map[normalized]
        return _finalize_usage(canonical, text, "alias_normalized", 0.85, raw_data)
    if normalized and normalized in _COMPOSITE_USAGE_MAP:
        resolved = _COMPOSITE_USAGE_MAP[normalized]
        return _finalize_usage(resolved, text, "composite", 0.85, raw_data,
                               note=f"'{text}' → '{normalized}' → {resolved}")

    # 4) Substring match
    for alias, canonical in alias_map.items():
        if len(alias) >= 2 and (alias in text or text in alias):
            return _finalize_usage(canonical, text, "alias_substring", 0.7, raw_data,
                                   note=f"부분 매칭: '{text}' ↔ '{alias}'")

    # 5) Unresolved
    return {
        "canonical_key": text,
        "resolved_to": text,
        "raw_input": raw_text,
        "match_type": "unresolved",
        "confidence": 0.0,
        "note": f"'{text}' 매핑 실패. 기본 활하중(2.5 kN/m²) 적용 예정.",
    }


def _finalize_usage(canonical: str, raw: str, match_type: str, confidence: float,
                    raw_data: dict, note: str | None = None) -> dict:
    """maps_to 체인 해석 후 최종 결과 반환."""
    entry = raw_data.get(canonical, {})
    resolved_to = entry.get("maps_to", canonical)
    if note is None and "note" in entry:
        note = entry["note"]
    return {
        "canonical_key": canonical,
        "resolved_to": resolved_to,
        "raw_input": raw,
        "match_type": match_type,
        "confidence": confidence,
        "note": note,
    }


# ============================================================
# 2. 지역명 해석
# ============================================================

def resolve_region(raw_text: str) -> dict:
    """지역명 → Supabase hazard_region_values fuzzy match.

    Returns:
        {
            "resolved_region": str | None,
            "region_sido": str | None,
            "region_sigungu": str | None,
            "raw_input": str,
            "match_type": "exact"|"partial"|"ambiguous"|"not_found"|"error",
            "candidates": list[dict],
            "hazard_preview": dict | None,
        }
    """
    text = raw_text.strip()
    if not text:
        return {
            "resolved_region": None,
            "raw_input": raw_text,
            "match_type": "not_specified",
            "candidates": [],
            "hazard_preview": None,
        }

    try:
        from core.kds_loads import query_hazard_values
    except ImportError:
        return {
            "resolved_region": None,
            "raw_input": raw_text,
            "match_type": "error",
            "candidates": [],
            "hazard_preview": None,
        }

    # 공백으로 분리하여 더 정밀한 검색
    parts = text.split()

    try:
        result = query_hazard_values(text)
    except Exception:
        return {
            "resolved_region": None,
            "raw_input": raw_text,
            "match_type": "error",
            "candidates": [],
            "hazard_preview": None,
        }

    if result.get("status") != "success" or result.get("count", 0) == 0:
        # 부분 검색 시도 (공백 분리 시 마지막 부분으로)
        if len(parts) > 1:
            try:
                result = query_hazard_values(parts[-1])
            except Exception:
                pass

    if result.get("status") != "success" or result.get("count", 0) == 0:
        return {
            "resolved_region": None,
            "raw_input": raw_text,
            "match_type": "not_found",
            "candidates": [],
            "hazard_preview": None,
        }

    # 결과에서 고유 (sido, sigungu) 추출
    regions = result.get("regions", [])
    unique_regions = {}
    for r in regions:
        sido = r.get("region_sido", "")
        sigungu = r.get("region_sigungu", "")
        key = (sido, sigungu)
        if key not in unique_regions:
            unique_regions[key] = {
                "region_sido": sido,
                "region_sigungu": sigungu,
                "snow_sg": r.get("snow_sg"),
                "wind_v0": r.get("wind_v0"),
            }

    candidates = list(unique_regions.values())

    # 다중 부분 필터링: "부산 해운대" → sido에 "부산" AND sigungu에 "해운대"
    if len(candidates) > 1 and len(parts) > 1:
        filtered = candidates
        for part in parts:
            filtered = [
                c for c in filtered
                if part in c["region_sido"] or part in c["region_sigungu"]
            ]
        if filtered:
            candidates = filtered

    if len(candidates) == 1:
        c = candidates[0]
        return {
            "resolved_region": f"{c['region_sido']} {c['region_sigungu']}",
            "region_sido": c["region_sido"],
            "region_sigungu": c["region_sigungu"],
            "raw_input": raw_text,
            "match_type": "partial" if text != f"{c['region_sido']} {c['region_sigungu']}" else "exact",
            "candidates": candidates,
            "hazard_preview": {
                "snow_sg": c.get("snow_sg"),
                "wind_v0": c.get("wind_v0"),
            },
        }

    # 다중 후보: ambiguous
    return {
        "resolved_region": None,
        "raw_input": raw_text,
        "match_type": "ambiguous",
        "candidates": [
            {"region": f"{c['region_sido']} {c['region_sigungu']}",
             "region_sido": c["region_sido"],
             "region_sigungu": c["region_sigungu"]}
            for c in candidates[:10]
        ],
        "hazard_preview": None,
    }


# ============================================================
# 3. 층 범위 확장
# ============================================================

def expand_story_intents(
    intents: list[dict],
    num_stories: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """StoryIntent 범위 → 개별 층 config + 경고.

    Args:
        intents: [{"floor_start": 1, "floor_end": 1, "usage_raw": "근생", "height": None}, ...]
        num_stories: 총 층수 (명시 시). None이면 intents에서 추론.

    Returns:
        (stories_list, warnings)
        stories_list: [{"height": 4.0, "usage": "retail", ...}, ...]  (1층부터 순서대로)
        warnings: [{"code": "NL03", "severity": "info", "message": "..."}, ...]
    """
    warnings = []

    # 총 층수 결정
    max_floor = 0
    for intent in intents:
        fs = intent.get("floor_start", 1)
        fe = intent.get("floor_end", fs)
        max_floor = max(max_floor, fe)
    if num_stories and num_stories > max_floor:
        max_floor = num_stories

    # 층별 usage 매핑
    floor_map: dict[int, dict] = {}
    occupancy_matches = []

    for intent in intents:
        fs = intent.get("floor_start", 1)
        fe = intent.get("floor_end", fs)
        usage_raw = intent.get("usage_raw", "office")
        height = intent.get("height")
        dlf = intent.get("dead_load_finish")
        slab = intent.get("slab_thickness")

        # 용도 해석
        usage_result = resolve_usage(usage_raw)
        resolved_usage = usage_result["resolved_to"]
        occupancy_matches.append({
            "floors": f"{fs}-{fe}" if fs != fe else str(fs),
            **usage_result,
        })

        for floor in range(fs, fe + 1):
            if floor in floor_map:
                warnings.append({
                    "code": "NL08",
                    "severity": "caution",
                    "message": f"{floor}층 용도가 중복 지정됨. 마지막 값({usage_raw}) 적용.",
                })
            floor_map[floor] = {
                "usage": resolved_usage,
                "usage_raw": usage_raw,
                "usage_source": "llm_inferred",
                "usage_match_type": usage_result["match_type"],
                "height": height,
                "dead_load_finish": dlf,
                "slab_thickness": slab,
            }

    # 갭 채움
    height_defaults_used = False
    stories = []
    for floor in range(1, max_floor + 1):
        if floor in floor_map:
            info = floor_map[floor]
        else:
            info = {
                "usage": "office",
                "usage_raw": None,
                "usage_source": "default",
                "usage_match_type": "default",
                "height": None,
                "dead_load_finish": None,
                "slab_thickness": None,
            }
            warnings.append({
                "code": "NL09",
                "severity": "info",
                "message": f"{floor}층 용도 미지정 → 기본값(office) 적용.",
            })

        usage = info["usage"]

        # 층고 결정
        if info["height"] is not None:
            h = info["height"]
            h_source = "user_specified"
        elif usage in _USAGE_HEIGHT_DEFAULTS:
            h = _USAGE_HEIGHT_DEFAULTS[usage]
            h_source = f"default_{usage}"
            height_defaults_used = True
        elif floor == 1:
            h = 4.0
            h_source = "default_first_floor"
            height_defaults_used = True
        else:
            h = 3.5
            h_source = "default_standard"
            height_defaults_used = True

        # dead_load_finish 결정
        if info["dead_load_finish"] is not None:
            dlf = info["dead_load_finish"]
        elif usage in _USAGE_DEAD_LOAD_OVERRIDE:
            dlf = _USAGE_DEAD_LOAD_OVERRIDE[usage]
            warnings.append({
                "code": "NL04",
                "severity": "info",
                "message": f"{floor}층({usage}) dead_load_finish {dlf} kN/m² 적용 (설비 추가하중).",
            })
        else:
            dlf = None  # BuildingModel 기본값 사용

        story = {"height": h, "usage": usage}
        if dlf is not None:
            story["dead_load_finish"] = dlf
        if info["slab_thickness"] is not None:
            story["slab_thickness"] = info["slab_thickness"]

        # 메타 (resolution_report용, config에는 포함 안됨)
        story["_meta"] = {
            "floor": floor,
            "height_source": h_source,
            "usage_source": info["usage_source"],
            "usage_raw": info["usage_raw"],
        }

        stories.append(story)

    if height_defaults_used:
        warnings.append({
            "code": "NL03",
            "severity": "info",
            "message": "층고 미지정 → 1층 4.0m, 기준층 3.5m, 기계실/옥상 3.0m 적용.",
        })

    return stories, occupancy_matches, warnings


# ============================================================
# 4. 경간 해석
# ============================================================

def resolve_bays(intent: dict) -> tuple[list[float], list[float], str, list[dict]]:
    """경간 정보 해석.

    Returns:
        (bays_x, bays_y, source, warnings)
    """
    warnings = []
    bays_x = intent.get("bays_x")
    bays_y = intent.get("bays_y")

    if bays_x and bays_y:
        return bays_x, bays_y, "user_specified", warnings

    typical_width = intent.get("typical_bay_width", 8.0)
    num_x = intent.get("num_bays_x")
    num_y = intent.get("num_bays_y")

    if num_x and not bays_x:
        bays_x = [typical_width] * num_x
    if num_y and not bays_y:
        bays_y = [typical_width] * num_y

    if bays_x and bays_y:
        return bays_x, bays_y, "inferred", warnings

    # 기본값
    if not bays_x:
        bays_x = [8.0, 8.0]
    if not bays_y:
        bays_y = [8.0]

    warnings.append({
        "code": "NL02",
        "severity": "caution",
        "message": f"경간 정보 미지정 → 기본값 bays_x={bays_x}, bays_y={bays_y} 적용.",
    })

    return bays_x, bays_y, "default", warnings


# ============================================================
# 5. 메인 해석 함수
# ============================================================

def resolve_building_config(intent_dict: dict) -> dict:
    """BuildingIntent → validated config + resolution report.

    Args:
        intent_dict: Claude가 추출한 BuildingIntent JSON.

    Returns:
        {
            "status": "resolved" | "needs_clarification",
            "config": dict | None,
            "resolution_report": {
                "occupancy_matches": [...],
                "region_match": {...},
                "stories_expanded": [...],
                "bays_resolved": {...},
                "defaults_applied": [...],
            },
            "warnings": [...],
            "clarification_needed": [...],
        }
    """
    all_warnings = []
    clarification_needed = []
    defaults_applied = []

    # ── 1. 층 확장 ──
    story_intents = intent_dict.get("stories", [])
    num_stories = intent_dict.get("num_stories")
    stories, occupancy_matches, story_warnings = expand_story_intents(story_intents, num_stories)
    all_warnings.extend(story_warnings)

    # 미매핑 용도 경고
    for match in occupancy_matches:
        if match["match_type"] == "unresolved":
            all_warnings.append({
                "code": "NL05",
                "severity": "warning",
                "message": f"{match['floors']}층 용도 '{match['raw_input']}' 매핑 실패. 기본 활하중 적용.",
            })
        elif match["match_type"] == "composite":
            all_warnings.append({
                "code": "NL01",
                "severity": "info",
                "message": f"{match['floors']}층: {match.get('note', '')}",
            })
        elif match["match_type"] == "alias_substring":
            all_warnings.append({
                "code": "NL12",
                "severity": "caution",
                "message": (
                    f"{match['floors']}층: '{match['raw_input']}' → {match['resolved_to']} "
                    f"({match.get('note', '')}). 의도한 용도가 맞는지 확인 필요."
                ),
            })

    # ── 2. 지역 해석 ──
    region_raw = intent_dict.get("region_raw", "")
    region_result = resolve_region(region_raw)

    if region_result["match_type"] == "ambiguous":
        clarification_needed.append({
            "type": "ambiguous_region",
            "raw_input": region_raw,
            "candidates": region_result["candidates"],
            "question": f"'{region_raw}'이(가) 다음 중 어디를 의미합니까?",
        })
    elif region_result["match_type"] == "not_found":
        all_warnings.append({
            "code": "NL11",
            "severity": "warning",
            "message": f"지역 '{region_raw}' DB에서 찾을 수 없습니다.",
        })
    elif region_result["match_type"] == "not_specified":
        all_warnings.append({
            "code": "NL07",
            "severity": "info",
            "message": "지역 미지정 → 지진/풍하중 미생성.",
        })

    # ── 3. 경간 해석 ──
    bays_x, bays_y, bays_source, bays_warnings = resolve_bays(intent_dict)
    all_warnings.extend(bays_warnings)
    if bays_source == "default":
        defaults_applied.append({"param": "bays_x", "value": bays_x, "source": "default"})
        defaults_applied.append({"param": "bays_y", "value": bays_y, "source": "default"})

    # ── 4. clarification 필요 시 조기 반환 ──
    if clarification_needed:
        return {
            "status": "needs_clarification",
            "config": None,
            "resolution_report": {
                "occupancy_matches": occupancy_matches,
                "region_match": region_result,
            },
            "warnings": all_warnings,
            "clarification_needed": clarification_needed,
        }

    # ── 5. config 조립 ──
    # 층 정보 (메타 제거)
    config_stories = []
    stories_expanded = []
    for s in stories:
        meta = s.pop("_meta", {})
        config_stories.append(s)
        stories_expanded.append({
            "story": meta.get("floor"),
            "height": s["height"],
            "usage": s["usage"],
            "height_source": meta.get("height_source"),
            "usage_source": meta.get("usage_source"),
            "usage_raw": meta.get("usage_raw"),
        })

    config: dict[str, Any] = {
        "stories": config_stories,
        "bays_x": bays_x,
        "bays_y": bays_y,
    }

    # 지역 정보
    if region_result.get("region_sigungu"):
        config["region"] = region_result["region_sigungu"]

    # 선택적 파라미터 (intent에서 직접 전달)
    _optional_keys = [
        "site_class", "importance", "seismic_system", "exposure_category",
        "column_section", "beam_x_section", "beam_y_section",
        "material_name", "supports", "auto_combinations",
        "rigid_diaphragm", "geometric_nonlinearity",
    ]
    for key in _optional_keys:
        val = intent_dict.get(key)
        if val is not None:
            config[key] = val
        else:
            # 기본값 추적
            from core.assumption_tracker import _FIELD_DEFAULTS
            if key in _FIELD_DEFAULTS:
                _, default_val = _FIELD_DEFAULTS[key]
                defaults_applied.append({"param": key, "value": default_val, "source": "default"})

    return {
        "status": "resolved",
        "config": config,
        "resolution_report": {
            "occupancy_matches": occupancy_matches,
            "region_match": {
                "raw": region_raw,
                "resolved": region_result.get("resolved_region"),
                "match_type": region_result["match_type"],
                "hazard_preview": region_result.get("hazard_preview"),
            },
            "stories_expanded": stories_expanded,
            "bays_resolved": {
                "bays_x": bays_x,
                "bays_y": bays_y,
                "source": bays_source,
            },
            "defaults_applied": defaults_applied,
        },
        "warnings": all_warnings,
        "clarification_needed": [],
    }
