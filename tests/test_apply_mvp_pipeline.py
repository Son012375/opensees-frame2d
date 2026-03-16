"""End-to-end tests for the Apply MVP pipeline.

All Supabase calls are mocked — tests run without network.
OpenSees analysis is guarded by ``OPENSEESPY_AVAILABLE``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

# Re-use existing mock infrastructure from the adapter tests
from tests.test_load_combo_adapter import MockSupabase, ULS3_DB_ROWS

from apply_mvp.load_case_builder import build_load_cases, lookup_live_load
from apply_mvp.opensees_3d_mvp import OPENSEESPY_AVAILABLE, build_and_analyze
from apply_mvp.runner import run_pipeline
from apply_mvp.schemas import (
    BuildingInfo,
    CodeRef,
    ComboRequest,
    FloorInfo,
    LoadsUser,
    MvpInput,
)
from core.contract_interpreter import ContractInterpreter

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_YAML = ROOT / "docs" / "specs" / "load_contract.yaml"
NORMALIZED_DB = ROOT / "data" / "kds_output" / "03_normalized.json"


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_input() -> MvpInput:
    return MvpInput(
        code=CodeRef(code_id="KDS 41 12 00", code_version="2022-10-11"),
        combo_request=ComboRequest(param_subtype="uls", combo_id="ULS3"),
        building=BuildingInfo(
            name="TestBuilding",
            floors=[
                FloorInfo(level=1, height_m=3.5, occupancy_key="office",
                          floor_area_m2=400.0, edge_beam_total_length_m=80.0),
                FloorInfo(level=2, height_m=3.5, occupancy_key="office",
                          floor_area_m2=400.0, edge_beam_total_length_m=80.0),
            ],
        ),
        loads_user=LoadsUser(
            DL_kN_per_m2=5.0,
            SDL_kN_per_m2=1.0,
            wind_base_kN_per_m2=1.2,
            snow_base_kN_per_m2=0.8,
            eq_base_kN_per_m2=0.7,
            use_direction_expansion=True,
        ),
    )


@pytest.fixture
def interp() -> ContractInterpreter:
    return ContractInterpreter(CONTRACT_YAML)


@pytest.fixture
def db_records() -> list[dict]:
    """Load normalised records once for all tests."""
    import json
    with open(NORMALIZED_DB, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("records", [])


# ═══════════════════════════════════════════════════════════════════════
# 1. Schema tests
# ═══════════════════════════════════════════════════════════════════════

class TestSchemas:
    def test_from_dict_roundtrip(self):
        raw = {
            "code": {"code_id": "KDS 41 12 00", "code_version": "2022-10-11"},
            "combo_request": {"param_subtype": "uls", "combo_id": "ULS3"},
            "building": {
                "name": "Test",
                "floors": [
                    {"level": 1, "height_m": 3.5, "occupancy_key": "office",
                     "floor_area_m2": 200.0, "edge_beam_total_length_m": 40.0},
                ],
            },
            "loads_user": {"DL_kN_per_m2": 5.0, "SDL_kN_per_m2": 1.0},
        }
        inp = MvpInput.from_dict(raw)
        assert inp.building.name == "Test"
        assert len(inp.building.floors) == 1
        assert inp.building.floors[0].occupancy_key == "office"
        assert inp.code.code_id == "KDS 41 12 00"

    def test_from_dict_defaults(self):
        inp = MvpInput.from_dict({})
        assert inp.code.code_id == "KDS 41 12 00"
        assert inp.combo_request.combo_id == "ULS3"
        assert inp.building.name == "unnamed"


# ═══════════════════════════════════════════════════════════════════════
# 2. Load-case builder tests
# ═══════════════════════════════════════════════════════════════════════

class TestLookupLiveLoad:
    def test_office_from_local_db(self, db_records):
        val, src = lookup_live_load("office", db_records=db_records)
        assert val is not None
        assert val == 2.5  # office_general = 2.5 kN/m2
        assert src == "local_db"

    def test_residential_from_local_db(self, db_records):
        val, src = lookup_live_load("residential", db_records=db_records)
        assert val is not None
        assert val == 2.0
        assert src == "local_db"

    def test_unknown_occupancy_returns_none(self, db_records):
        val, src = lookup_live_load("underwater_base", db_records=db_records)
        assert val is None
        assert src == "not_found"


class TestBuildLoadCases:
    def test_basic_output_structure(self, sample_input, db_records):
        result = build_load_cases(sample_input, db_records=db_records)
        assert "DL" in result.load_cases_global
        assert "LL" in result.load_cases_global
        assert "WIND" in result.load_cases_global
        assert result.ll_source == "local_db"

    def test_ll_value_from_db(self, sample_input, db_records):
        result = build_load_cases(sample_input, db_records=db_records)
        assert result.load_cases_global["LL"] == 2.5

    def test_load_application_plan_per_floor(self, sample_input, db_records):
        result = build_load_cases(sample_input, db_records=db_records)
        assert 1 in result.load_application_plan
        assert 2 in result.load_application_plan
        plan_f1 = result.load_application_plan[1]
        assert "DL" in plan_f1
        assert plan_f1["DL"]["type"] == "line"
        # DL: 5.0 kN/m2 * 400 m2 / 80 m = 25 kN/m
        assert math.isclose(plan_f1["DL"]["w_kN_per_m"], 25.0, rel_tol=1e-3)

    def test_fallback_when_db_missing(self, sample_input):
        # Pass empty DB records
        result = build_load_cases(sample_input, db_records=[])
        assert result.load_cases_global["LL"] == 2.5  # default fallback
        assert result.ll_source == "default_fallback"
        assert any("needs_review" in w for w in result.warnings)

    def test_user_fallback(self, sample_input):
        sample_input.loads_user.LL_fallback_kN_per_m2 = 3.0
        result = build_load_cases(sample_input, db_records=[])
        assert result.load_cases_global["LL"] == 3.0
        assert result.ll_source == "user_fallback"


# ═══════════════════════════════════════════════════════════════════════
# 3. ContractInterpreter integration
# ═══════════════════════════════════════════════════════════════════════

class TestComboIntegration:
    def test_resolve_combo_with_built_load_cases(self, sample_input, interp, db_records):
        """load_cases from builder → ContractInterpreter → total_effect is non-zero."""
        lc = build_load_cases(sample_input, db_records=db_records)

        # Use the same ULS3 combo rows from the adapter test
        combo_rows = [
            {"primary_key": "DL",   "value": 1.2, "conditions": []},
            {"primary_key": "RLL",  "value": 1.6, "conditions": [
                {"kind": "alt", "group": "roof_or_snow",
                 "candidates": ["RLL", "SNOW"], "policy": "max_effect"}
            ]},
            {"primary_key": "SNOW", "value": 1.6, "conditions": [
                {"kind": "alt", "group": "roof_or_snow",
                 "candidates": ["RLL", "SNOW"], "policy": "max_effect"}
            ]},
            {"primary_key": "LL",   "value": 1.0, "conditions": [
                {"kind": "alt", "group": "live_or_wind",
                 "candidates": ["LL", "WIND"], "policy": "max_effect"},
                {"kind": "reduction", "target": "LL", "allowed": True,
                 "factor": 0.5, "ref": "live_reduction",
                 "condition_ko": "base LL ≤ 5.0kN/m2"}
            ]},
            {"primary_key": "WIND", "value": 0.5, "conditions": [
                {"kind": "alt", "group": "live_or_wind",
                 "candidates": ["LL", "WIND"], "policy": "max_effect"}
            ]},
        ]

        result = interp.resolve_combo(
            combo_rows,
            dict(lc.load_cases_global),
            combo_id="ULS3",
        )

        # total_effect must be non-zero
        assert result.total_effect > 0
        # LL factor should be 0.5 (reduction applied)
        assert result.applied_factors["LL"] == 0.5
        # WIND should be zeroed (alt: LL wins over WIND)
        assert result.applied_factors["WIND"] == 0.0

        # Alt group decisions are in summary
        assert "roof_or_snow" in result.summary["alt_groups"]
        assert "live_or_wind" in result.summary["alt_groups"]

        # Reductions recorded
        reductions = result.summary["reductions"]
        ll_reds = [r for r in reductions if r["target"] == "LL"]
        assert len(ll_reds) == 1
        assert ll_reds[0]["applied"] is True

    def test_total_effect_expected_value(self, interp, db_records):
        """Verify the golden value: ULS3 with DL=5.0, LL=2.5 → 8.85 kN/m2."""
        load_cases = {"DL": 5.0, "LL": 2.5, "RLL": 1.0, "SNOW": 0.0, "WIND": 0.0}
        combo_rows = [
            {"primary_key": "DL",   "value": 1.2, "conditions": []},
            {"primary_key": "RLL",  "value": 1.6, "conditions": [
                {"kind": "alt", "group": "roof_or_snow",
                 "candidates": ["RLL", "SNOW"], "policy": "max_effect"}
            ]},
            {"primary_key": "SNOW", "value": 1.6, "conditions": [
                {"kind": "alt", "group": "roof_or_snow",
                 "candidates": ["RLL", "SNOW"], "policy": "max_effect"}
            ]},
            {"primary_key": "LL",   "value": 1.0, "conditions": [
                {"kind": "alt", "group": "live_or_wind",
                 "candidates": ["LL", "WIND"], "policy": "max_effect"},
                {"kind": "reduction", "target": "LL", "allowed": True,
                 "factor": 0.5, "ref": "live_reduction"}
            ]},
            {"primary_key": "WIND", "value": 0.5, "conditions": [
                {"kind": "alt", "group": "live_or_wind",
                 "candidates": ["LL", "WIND"], "policy": "max_effect"}
            ]},
        ]
        result = interp.resolve_combo(combo_rows, load_cases, combo_id="ULS3")
        assert math.isclose(result.total_effect, 8.85, rel_tol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 4. OpenSees 3D MVP (guarded)
# ═══════════════════════════════════════════════════════════════════════

class TestOpenSees3DMVP:
    @pytest.mark.skipif(not OPENSEESPY_AVAILABLE, reason="openseespy not installed")
    def test_basic_analysis_runs(self):
        """Minimal 3D frame with gravity loads should converge."""
        result = build_and_analyze(
            stories=[3.5, 3.5],
            bay_x=20.0,
            bay_y=20.0,
            beam_loads={1: 10.0, 2: 10.0},
        )
        assert result.success is True
        assert result.num_nodes == 12  # 3 levels × 4 corners
        assert result.num_elements > 0
        assert result.max_disp_z_mm != 0.0
        assert result.equilibrium_error_pct < 1.0  # <1% error

    @pytest.mark.skipif(not OPENSEESPY_AVAILABLE, reason="openseespy not installed")
    def test_zero_load_zero_disp(self):
        """Zero beam loads → zero displacement."""
        result = build_and_analyze(
            stories=[3.5],
            bay_x=10.0,
            bay_y=10.0,
            beam_loads={1: 0.0},
        )
        assert result.success is True
        assert abs(result.max_disp_z_mm) < 1e-6

    def test_without_openseespy(self):
        """When openseespy is missing, result contains skip message."""
        if OPENSEESPY_AVAILABLE:
            pytest.skip("openseespy is installed; can't test missing path")
        result = build_and_analyze(
            stories=[3.5],
            bay_x=10.0,
            bay_y=10.0,
            beam_loads={1: 10.0},
        )
        assert result.success is False
        assert any("not installed" in e for e in result.errors)


# ═══════════════════════════════════════════════════════════════════════
# 5. Full pipeline (mocked DB + combo)
# ═══════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_pipeline_produces_result(self, sample_input, monkeypatch):
        """Full pipeline runs without exceptions and produces a result."""
        # Monkeypatch combo-row fetcher to use the fallback example JSON
        import apply_mvp.runner as runner_mod

        # Force Supabase path to return empty so fallback is used
        monkeypatch.setattr(
            runner_mod, "_get_combo_rows_supabase",
            lambda _: ([], ""),
        )

        result = run_pipeline(sample_input)

        # Basics
        assert result.combo_source == "example_json"
        assert result.total_effect > 0
        assert result.applied_factors.get("DL") == 1.2

        # Load cases were built
        assert result.load_cases is not None
        assert result.load_cases.load_cases_global["LL"] == 2.5

        # Alt groups resolved
        assert "roof_or_snow" in result.combo_summary.get("alt_groups", {})

    def test_pipeline_no_combo_rows_errors(self, sample_input, monkeypatch):
        """Pipeline with no combo rows reports an error."""
        import apply_mvp.runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "_get_combo_rows_supabase",
            lambda _: ([], ""),
        )
        monkeypatch.setattr(
            runner_mod, "_get_combo_rows_fallback",
            lambda: ([], ""),
        )

        result = run_pipeline(sample_input)
        assert len(result.errors) > 0
        assert "No combo rows" in result.errors[0]
