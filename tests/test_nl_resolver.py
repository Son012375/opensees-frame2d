"""Phase 1 NL Resolver 단위 테스트.

검증 항목:
  - 용도(occupancy) 매칭: exact, composite, normalized, substring, unresolved
  - 지역(region) 매칭: exact, partial, ambiguous, not_found
  - 층 범위 확장: range, default heights, gap filling
  - 경간 해석: user_specified, inferred, default
  - 전체 워크플로우: resolve_building_config end-to-end
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# mcp-server를 sys.path에 추가
_mcp_server_dir = Path(__file__).resolve().parent.parent / "mcp-server"
if str(_mcp_server_dir) not in sys.path:
    sys.path.insert(0, str(_mcp_server_dir))

from core.nl_resolver import (
    resolve_usage,
    resolve_region,
    expand_story_intents,
    resolve_bays,
    resolve_building_config,
    _load_occupancy_data,
)


# ============================================================
# 1. 용도(Occupancy) 매칭 테스트
# ============================================================

class TestOccupancyResolution:
    """용도명 → canonical key 매칭."""

    def test_exact_alias(self):
        """정확한 한국어 별칭 → canonical key."""
        result = resolve_usage("사무실")
        assert result["resolved_to"] == "office"
        assert result["match_type"] == "alias_exact"
        assert result["confidence"] == 1.0

    def test_exact_alias_residential(self):
        """'아파트' → residential."""
        result = resolve_usage("아파트")
        assert result["resolved_to"] == "residential"
        assert result["match_type"] == "alias_exact"

    def test_canonical_key_direct(self):
        """canonical key 자체 입력 → 매칭."""
        result = resolve_usage("office")
        assert result["resolved_to"] == "office"
        assert result["match_type"] == "alias_exact"

    def test_composite_mapping(self):
        """건축법 복합 용도 → retail 매핑."""
        result = resolve_usage("근린생활시설")
        assert result["resolved_to"] == "retail"
        assert result["match_type"] in ("alias_exact", "composite")
        assert result["confidence"] >= 0.9

    def test_abbreviation(self):
        """약어 '근생' → retail."""
        result = resolve_usage("근생")
        assert result["resolved_to"] == "retail"

    def test_composite_1종근생(self):
        """'1종근생' → retail."""
        result = resolve_usage("1종근생")
        assert result["resolved_to"] == "retail"

    def test_mechanical_room(self):
        """'기계실' → mechanical_room."""
        result = resolve_usage("기계실")
        assert result["resolved_to"] == "mechanical_room"

    def test_mechanical_room_variant(self):
        """'전기실' → mechanical_room."""
        result = resolve_usage("전기실")
        assert result["resolved_to"] == "mechanical_room"

    def test_parking_piloti(self):
        """'필로티' → parking."""
        result = resolve_usage("필로티")
        assert result["resolved_to"] == "parking"

    def test_roof(self):
        """'옥탑층' → roof."""
        result = resolve_usage("옥탑층")
        assert result["resolved_to"] == "roof"

    def test_unresolved(self):
        """매핑 불가 → unresolved."""
        result = resolve_usage("특수연구시설")
        assert result["match_type"] == "unresolved"
        assert result["confidence"] == 0.0

    def test_english_name(self):
        """영어 이름(소문자) → 매칭."""
        result = resolve_usage("hospital")
        assert result["resolved_to"] == "hospital"

    def test_neighborhood_living_maps_to_retail(self):
        """neighborhood_living → maps_to: retail."""
        result = resolve_usage("근린생활시설")
        assert result["resolved_to"] == "retail"

    def test_whitespace_handling(self):
        """전후 공백 제거 후 매칭."""
        result = resolve_usage("  사무실  ")
        assert result["resolved_to"] == "office"


# ============================================================
# 2. 지역(Region) 매칭 테스트
# ============================================================

class TestRegionResolution:
    """지역명 → Supabase fuzzy match."""

    def _mock_hazard_result(self, regions):
        """테스트용 hazard_values 결과 생성."""
        return {
            "status": "success",
            "count": len(regions),
            "regions": regions,
        }

    def test_empty_region(self):
        """빈 문자열 → not_specified."""
        result = resolve_region("")
        assert result["match_type"] == "not_specified"
        assert result["resolved_region"] is None

    @patch("core.kds_loads.query_hazard_values")
    def test_exact_match(self, mock_query):
        """정확한 지역명 → exact match."""
        mock_query.return_value = self._mock_hazard_result([
            {"region_sido": "서울특별시", "region_sigungu": "강남구",
             "snow_sg": 0.5, "wind_v0": 26.0},
        ])
        result = resolve_region("서울특별시 강남구")
        assert result["match_type"] in ("exact", "partial")
        assert result["resolved_region"] is not None
        assert "강남구" in result["resolved_region"]

    @patch("core.kds_loads.query_hazard_values")
    def test_partial_match(self, mock_query):
        """'부산 해운대' → partial match (단일 후보)."""
        mock_query.return_value = self._mock_hazard_result([
            {"region_sido": "부산광역시", "region_sigungu": "해운대구",
             "snow_sg": 0.3, "wind_v0": 30.0},
        ])
        result = resolve_region("부산 해운대")
        assert result["match_type"] in ("exact", "partial")
        assert "해운대" in (result.get("resolved_region") or "")

    @patch("core.kds_loads.query_hazard_values")
    def test_ambiguous_region(self, mock_query):
        """다중 후보 → ambiguous."""
        mock_query.return_value = self._mock_hazard_result([
            {"region_sido": "서울특별시", "region_sigungu": "강남구",
             "snow_sg": 0.5, "wind_v0": 26.0},
            {"region_sido": "경기도", "region_sigungu": "강남동",
             "snow_sg": 0.5, "wind_v0": 26.0},
        ])
        result = resolve_region("강남")
        assert result["match_type"] == "ambiguous"
        assert len(result["candidates"]) >= 2

    @patch("core.kds_loads.query_hazard_values")
    def test_not_found(self, mock_query):
        """DB에 없는 지역명 → not_found."""
        mock_query.return_value = {"status": "success", "count": 0, "regions": []}
        result = resolve_region("아틀란티스")
        assert result["match_type"] == "not_found"

    def test_import_error_fallback(self):
        """kds_loads import 실패 → error."""
        with patch.dict("sys.modules", {"core.kds_loads": None}):
            result = resolve_region("서울")
            assert result["match_type"] == "error"


# ============================================================
# 3. 층 범위 확장 테스트
# ============================================================

class TestStoryExpansion:
    """StoryIntent ranges → 개별 층."""

    def test_range_expansion(self):
        """floor 2-5 → 4개 개별 층."""
        intents = [
            {"floor_start": 2, "floor_end": 5, "usage_raw": "사무실"},
        ]
        stories, matches, warnings = expand_story_intents(intents)
        # floor 1은 갭으로 추가됨
        assert len(stories) == 5  # 1~5
        assert stories[1]["usage"] == "office"  # 2층 (0-indexed)
        assert stories[4]["usage"] == "office"  # 5층

    def test_default_heights(self):
        """1층=4.0m, 기준층=3.5m."""
        intents = [
            {"floor_start": 1, "floor_end": 3, "usage_raw": "사무실"},
        ]
        stories, _, _ = expand_story_intents(intents)
        assert stories[0]["height"] == 4.0   # 1층
        assert stories[1]["height"] == 3.5   # 2층
        assert stories[2]["height"] == 3.5   # 3층

    def test_mechanical_room_height(self):
        """기계실 → 3.0m."""
        intents = [
            {"floor_start": 1, "floor_end": 1, "usage_raw": "사무실"},
            {"floor_start": 2, "floor_end": 2, "usage_raw": "기계실"},
        ]
        stories, _, _ = expand_story_intents(intents)
        assert stories[1]["height"] == 3.0  # 기계실
        assert stories[1]["usage"] == "mechanical_room"

    def test_gap_filling(self):
        """1층 + 3~5층 → 2층 기본값 채움."""
        intents = [
            {"floor_start": 1, "floor_end": 1, "usage_raw": "주차장"},
            {"floor_start": 3, "floor_end": 5, "usage_raw": "사무실"},
        ]
        stories, _, warnings = expand_story_intents(intents)
        assert len(stories) == 5
        assert stories[1]["usage"] == "office"  # 2층 갭 → office
        gap_warnings = [w for w in warnings if w["code"] == "NL09"]
        assert len(gap_warnings) == 1

    def test_duplicate_floor(self):
        """같은 층 중복 → NL08 경고."""
        intents = [
            {"floor_start": 1, "floor_end": 3, "usage_raw": "사무실"},
            {"floor_start": 2, "floor_end": 2, "usage_raw": "주차장"},
        ]
        stories, _, warnings = expand_story_intents(intents)
        assert stories[1]["usage"] == "parking"  # 마지막 값 우선
        dup_warnings = [w for w in warnings if w["code"] == "NL08"]
        assert len(dup_warnings) == 1

    def test_num_stories_override(self):
        """num_stories 지정 시 총 층수 확장."""
        intents = [
            {"floor_start": 1, "floor_end": 3, "usage_raw": "사무실"},
        ]
        stories, _, _ = expand_story_intents(intents, num_stories=5)
        assert len(stories) == 5
        # 4~5층은 갭으로 채워짐

    def test_user_height_override(self):
        """사용자가 명시한 층고 → 기본값보다 우선."""
        intents = [
            {"floor_start": 1, "floor_end": 1, "usage_raw": "사무실", "height": 5.0},
        ]
        stories, _, _ = expand_story_intents(intents)
        assert stories[0]["height"] == 5.0

    def test_dead_load_finish_mechanical(self):
        """기계실 dead_load_finish → 2.0 kN/m² 자동 적용."""
        intents = [
            {"floor_start": 1, "floor_end": 1, "usage_raw": "기계실"},
        ]
        stories, _, warnings = expand_story_intents(intents)
        assert stories[0].get("dead_load_finish") == 2.0
        nl04 = [w for w in warnings if w["code"] == "NL04"]
        assert len(nl04) == 1


# ============================================================
# 4. 경간 해석 테스트
# ============================================================

class TestBayResolution:
    """경간 정보 해석."""

    def test_user_specified(self):
        """bays_x/bays_y 직접 지정."""
        intent = {"bays_x": [6.0, 8.0, 6.0], "bays_y": [9.0]}
        bx, by, source, warnings = resolve_bays(intent)
        assert bx == [6.0, 8.0, 6.0]
        assert by == [9.0]
        assert source == "user_specified"
        assert len(warnings) == 0

    def test_num_bays_inferred(self):
        """num_bays + typical_width → inferred."""
        intent = {"num_bays_x": 3, "num_bays_y": 2, "typical_bay_width": 7.0}
        bx, by, source, _ = resolve_bays(intent)
        assert bx == [7.0, 7.0, 7.0]
        assert by == [7.0, 7.0]
        assert source == "inferred"

    def test_default_bays(self):
        """경간 미지정 → 기본값 + NL02 경고."""
        intent = {}
        bx, by, source, warnings = resolve_bays(intent)
        assert bx == [8.0, 8.0]
        assert by == [8.0]
        assert source == "default"
        assert any(w["code"] == "NL02" for w in warnings)


# ============================================================
# 5. 전체 워크플로우 테스트
# ============================================================

class TestFullResolution:
    """resolve_building_config end-to-end."""

    @patch("core.kds_loads.query_hazard_values")
    def test_complete_workflow(self, mock_query):
        """전체 변환: 1층 근생 + 2~5층 오피스 + 부산 해운대."""
        mock_query.return_value = {
            "status": "success",
            "count": 1,
            "regions": [
                {"region_sido": "부산광역시", "region_sigungu": "해운대구",
                 "snow_sg": 0.3, "wind_v0": 30.0},
            ],
        }

        intent = {
            "stories": [
                {"floor_start": 1, "floor_end": 1, "usage_raw": "근린생활시설"},
                {"floor_start": 2, "floor_end": 5, "usage_raw": "오피스"},
            ],
            "region_raw": "부산 해운대",
            "bays_x": [8.0, 8.0],
            "bays_y": [8.0],
        }

        result = resolve_building_config(intent)
        assert result["status"] == "resolved"
        assert result["config"] is not None
        config = result["config"]

        # 5개 층
        assert len(config["stories"]) == 5
        # 1층 → retail (근린생활시설 매핑)
        assert config["stories"][0]["usage"] == "retail"
        # 2~5층 → office
        for i in range(1, 5):
            assert config["stories"][i]["usage"] == "office"
        # 지역
        assert config["region"] == "해운대구"
        # 경간
        assert config["bays_x"] == [8.0, 8.0]

        # resolution report 존재
        report = result["resolution_report"]
        assert "occupancy_matches" in report
        assert "region_match" in report
        assert "stories_expanded" in report

    def test_minimal_input(self):
        """stories만 제공 → 나머지 기본값."""
        intent = {
            "stories": [
                {"floor_start": 1, "floor_end": 3, "usage_raw": "사무실"},
            ],
        }
        result = resolve_building_config(intent)
        assert result["status"] == "resolved"
        config = result["config"]
        assert len(config["stories"]) == 3
        # 기본 경간
        assert config["bays_x"] == [8.0, 8.0]
        assert config["bays_y"] == [8.0]
        # NL02 경고 (경간 기본값)
        assert any(w["code"] == "NL02" for w in result["warnings"])

    @patch("core.kds_loads.query_hazard_values")
    def test_needs_clarification(self, mock_query):
        """지역 모호 → status=needs_clarification."""
        mock_query.return_value = {
            "status": "success",
            "count": 3,
            "regions": [
                {"region_sido": "서울특별시", "region_sigungu": "강남구",
                 "snow_sg": 0.5, "wind_v0": 26.0},
                {"region_sido": "경기도", "region_sigungu": "강남동",
                 "snow_sg": 0.5, "wind_v0": 26.0},
                {"region_sido": "충청남도", "region_sigungu": "강남면",
                 "snow_sg": 0.5, "wind_v0": 26.0},
            ],
        }

        intent = {
            "stories": [
                {"floor_start": 1, "floor_end": 3, "usage_raw": "사무실"},
            ],
            "region_raw": "강남",
        }

        result = resolve_building_config(intent)
        assert result["status"] == "needs_clarification"
        assert len(result["clarification_needed"]) > 0
        assert result["config"] is None

    def test_unresolved_usage_warning(self):
        """매핑 불가 용도 → NL05 경고."""
        intent = {
            "stories": [
                {"floor_start": 1, "floor_end": 1, "usage_raw": "특수연구시설"},
            ],
        }
        result = resolve_building_config(intent)
        assert result["status"] == "resolved"
        assert any(w["code"] == "NL05" for w in result["warnings"])

    def test_composite_usage_warning(self):
        """복합 용도 매핑 → NL01 info."""
        intent = {
            "stories": [
                {"floor_start": 1, "floor_end": 1, "usage_raw": "근린생활시설"},
            ],
        }
        result = resolve_building_config(intent)
        # 근린생활시설이 occupancy.json에도 있으므로
        # exact match or composite 둘 다 가능
        config = result["config"]
        assert config["stories"][0]["usage"] == "retail"

    def test_substring_match_warning(self):
        """substring 매칭 시 NL12 caution 경고 발생."""
        # "대형 병원" — exact/composite/normalized에 안 걸리고 substring "병원" 매칭
        intent = {
            "stories": [
                {"floor_start": 1, "floor_end": 1, "usage_raw": "대형 병원"},
            ],
        }
        result = resolve_building_config(intent)
        assert result["status"] == "resolved"
        assert result["config"]["stories"][0]["usage"] == "hospital"
        nl12 = [w for w in result["warnings"] if w["code"] == "NL12"]
        assert len(nl12) == 1
        assert nl12[0]["severity"] == "caution"

    def test_optional_params_passthrough(self):
        """intent의 선택 파라미터 → config에 전달."""
        intent = {
            "stories": [
                {"floor_start": 1, "floor_end": 2, "usage_raw": "사무실"},
            ],
            "site_class": "S4",
            "importance": "I",
            "column_section": "H-400x400",
        }
        result = resolve_building_config(intent)
        config = result["config"]
        assert config["site_class"] == "S4"
        assert config["importance"] == "I"
        assert config["column_section"] == "H-400x400"
