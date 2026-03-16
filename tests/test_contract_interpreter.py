"""Tests for the KDS Contract Interpreter.

Validates combo resolution against docs/specs/examples/combo_apply_example.json
and the rules in LOAD_DB_CONTRACT.md section 4.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.contract_interpreter import ContractInterpreter, load_example

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_YAML = ROOT / "docs" / "specs" / "load_contract.yaml"
COMBO_EXAMPLE = ROOT / "docs" / "specs" / "examples" / "combo_apply_example.json"


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def interp():
    """Interpreter loaded with the project contract YAML."""
    return ContractInterpreter(CONTRACT_YAML)


@pytest.fixture
def uls3_example():
    """ULS3 combo_apply example data."""
    return load_example(COMBO_EXAMPLE)


# ── Helpers ──────────────────────────────────────────────────────────

def _combo_rows(ex: dict) -> list[dict]:
    return ex["step_6_combo_records"]


def _load_cases(ex: dict) -> dict[str, float]:
    return {k: float(v) for k, v in ex["calculation"]["load_cases_kN_m2"].items()}


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Full ULS3 scenario (regression against combo_apply_example.json)
# ═══════════════════════════════════════════════════════════════════════

class TestULS3Scenario:
    """Core regression tests — the example JSON is the oracle."""

    def test_total_effect_matches(self, interp, uls3_example):
        result = interp.resolve_combo(
            _combo_rows(uls3_example),
            _load_cases(uls3_example),
            combo_id="ULS3",
        )
        assert math.isclose(result.total_effect, 8.85, rel_tol=1e-6)

    def test_applied_factors(self, interp, uls3_example):
        result = interp.resolve_combo(
            _combo_rows(uls3_example),
            _load_cases(uls3_example),
            combo_id="ULS3",
        )
        af = result.applied_factors
        assert af["DL"] == 1.2
        assert af["RLL"] == 1.6
        assert af["LL"] == 0.5   # reduced from 1.0
        assert af["SNOW"] == 0.0  # zeroed by alt
        assert af["WIND"] == 0.0  # zeroed by alt

    def test_roof_or_snow_selects_one(self, interp, uls3_example):
        result = interp.resolve_combo(
            _combo_rows(uls3_example),
            _load_cases(uls3_example),
            combo_id="ULS3",
        )
        grp = result.summary["alt_groups"]["roof_or_snow"]
        assert len(grp["selected"]) == 1
        sel = grp["selected"][0]
        assert sel in ("RLL", "SNOW")
        # With SNOW=0, RLL=1.0 -> RLL wins
        assert sel == "RLL"

    def test_live_or_wind_selects_one(self, interp, uls3_example):
        result = interp.resolve_combo(
            _combo_rows(uls3_example),
            _load_cases(uls3_example),
            combo_id="ULS3",
        )
        grp = result.summary["alt_groups"]["live_or_wind"]
        assert len(grp["selected"]) == 1
        sel = grp["selected"][0]
        assert sel in ("LL", "WIND")
        # LL effect=1.0*2.5=2.5 > WIND effect=0.5*0.0=0.0
        assert sel == "LL"

    def test_ll_reduction_applied(self, interp, uls3_example):
        result = interp.resolve_combo(
            _combo_rows(uls3_example),
            _load_cases(uls3_example),
            combo_id="ULS3",
        )
        reds = result.summary["reductions"]
        assert len(reds) >= 1
        ll_red = [r for r in reds if r["target"] == "LL"]
        assert len(ll_red) == 1
        assert ll_red[0]["applied"] is True
        assert ll_red[0]["original_factor"] == 1.0
        assert ll_red[0]["new_factor"] == 0.5

    def test_selected_rows_have_nonzero_factors(self, interp, uls3_example):
        result = interp.resolve_combo(
            _combo_rows(uls3_example),
            _load_cases(uls3_example),
            combo_id="ULS3",
        )
        for sr in result.selected_rows:
            assert sr["factor"] != 0.0


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Alt selection with non-zero WIND
# ═══════════════════════════════════════════════════════════════════════

class TestAltWithWind:
    """Varying WIND value to verify the max_effect algorithm."""

    def test_ll_still_wins_moderate_wind(self, interp, uls3_example):
        lc = _load_cases(uls3_example)
        lc["WIND"] = 1.2
        result = interp.resolve_combo(
            _combo_rows(uls3_example), lc, combo_id="ULS3",
        )
        grp = result.summary["alt_groups"]["live_or_wind"]
        # LL: 1.0*2.5=2.5 > WIND: 0.5*1.2=0.6
        assert grp["selected"] == ["LL"]

    def test_wind_wins_high_wind(self, interp, uls3_example):
        lc = _load_cases(uls3_example)
        lc["WIND"] = 10.0
        lc["LL"] = 0.5
        result = interp.resolve_combo(
            _combo_rows(uls3_example), lc, combo_id="ULS3",
        )
        grp = result.summary["alt_groups"]["live_or_wind"]
        # LL: 1.0*0.5=0.5 < WIND: 0.5*10.0=5.0
        assert grp["selected"] == ["WIND"]
        assert result.applied_factors["LL"] == 0.0
        assert result.applied_factors["WIND"] == 0.5

    def test_reduction_skipped_when_ll_zeroed(self, interp, uls3_example):
        """When WIND wins alt, LL factor=0 -> reduction is a no-op."""
        lc = _load_cases(uls3_example)
        lc["WIND"] = 10.0
        lc["LL"] = 0.5
        result = interp.resolve_combo(
            _combo_rows(uls3_example), lc, combo_id="ULS3",
        )
        reds = [r for r in result.summary["reductions"] if r["target"] == "LL"]
        assert len(reds) == 1
        assert reds[0]["applied"] is False


# ═══════════════════════════════════════════════════════════════════════
# Test 3: kind=derived (informational only)
# ═══════════════════════════════════════════════════════════════════════

class TestDerived:
    """Derived conditions must not affect calculation; only appear in summary."""

    def _rows_with_derived(self):
        return [
            {"primary_key": "DL", "value": 0.9, "conditions": []},
            {"primary_key": "WIND", "value": 0.45, "conditions": [
                {"kind": "derived", "expression": "0.75*0.6",
                 "base_factors": [0.75, 0.6]},
            ]},
        ]

    def test_derived_recorded_in_summary(self, interp):
        result = interp.resolve_combo(
            self._rows_with_derived(), {"DL": 5.0, "WIND": 1.2},
        )
        assert "derived" in result.summary
        assert len(result.summary["derived"]) == 1
        d = result.summary["derived"][0]
        assert d["primary_key"] == "WIND"
        assert d["expression"] == "0.75*0.6"
        assert d["base_factors"] == [0.75, 0.6]

    def test_derived_no_calculation_impact(self, interp):
        result = interp.resolve_combo(
            self._rows_with_derived(), {"DL": 5.0, "WIND": 1.2},
        )
        expected = 0.9 * 5.0 + 0.45 * 1.2  # 4.5 + 0.54 = 5.04
        assert math.isclose(result.total_effect, expected, rel_tol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Test 4: WIND/EQ direction expansion
# ═══════════════════════════════════════════════════════════════════════

class TestDirectionExpansion:
    """Direction expansion creates sub-keys and records them in summary."""

    def test_wind_expanded_to_4_directions(self, interp):
        rows = [
            {"primary_key": "DL", "value": 0.9, "conditions": []},
            {"primary_key": "WIND", "value": 1.0, "conditions": []},
        ]
        result = interp.resolve_combo(
            rows, {"DL": 5.0, "WIND": 1.2},
            options={"expand_directions": True},
        )
        exp = result.summary.get("direction_expansion", {})
        assert "WIND" in exp.get("expanded_keys", {})
        dirs = exp["expanded_keys"]["WIND"]["directions"]
        for dk in ("WIND_X_POS", "WIND_X_NEG", "WIND_Y_POS", "WIND_Y_NEG"):
            assert dk in dirs

    def test_eq_expanded_to_2_directions(self, interp):
        rows = [
            {"primary_key": "DL", "value": 1.2, "conditions": []},
            {"primary_key": "EQ", "value": 1.0, "conditions": []},
        ]
        result = interp.resolve_combo(
            rows, {"DL": 5.0, "EQ": 0.8},
            options={"expand_directions": True},
        )
        exp = result.summary.get("direction_expansion", {})
        assert "EQ" in exp.get("expanded_keys", {})
        dirs = exp["expanded_keys"]["EQ"]["directions"]
        for dk in ("EQ_X", "EQ_Y"):
            assert dk in dirs

    def test_no_expansion_by_default(self, interp):
        rows = [
            {"primary_key": "WIND", "value": 1.0, "conditions": []},
        ]
        result = interp.resolve_combo(rows, {"WIND": 1.2})
        assert "direction_expansion" not in result.summary


# ═══════════════════════════════════════════════════════════════════════
# Test 5: includes (F/T/H processing policy)
# ═══════════════════════════════════════════════════════════════════════

class TestIncludes:
    """Includes are recorded with Phase 1 policy; no calc impact."""

    def test_includes_recorded(self, interp):
        rows = [
            {"primary_key": "DL", "value": 1.2, "conditions": [
                {"includes": ["F", "T"],
                 "note": "F(fluid) T(temperature) same factor"},
            ]},
        ]
        result = interp.resolve_combo(rows, {"DL": 5.0})
        assert "includes" in result.summary
        inc = result.summary["includes"]
        assert len(inc) == 1
        assert inc[0]["includes"] == ["F", "T"]
        assert "Phase 1" in inc[0]["policy"]

    def test_includes_no_calculation_impact(self, interp):
        rows = [
            {"primary_key": "DL", "value": 1.2, "conditions": [
                {"includes": ["F"], "note": "F included"},
            ]},
        ]
        result = interp.resolve_combo(rows, {"DL": 5.0})
        assert math.isclose(result.total_effect, 6.0, rel_tol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Contract YAML loading
# ═══════════════════════════════════════════════════════════════════════

class TestContractLoading:
    """Verify load_contract.yaml is parsed correctly."""

    def test_contract_loaded(self, interp):
        assert interp.contract
        assert interp.contract["meta"]["contract_version"] == "1.0.0"

    def test_direction_rules_present(self, interp):
        assert "WIND" in interp.direction_rules
        assert "EQ" in interp.direction_rules

    def test_conditions_schema_present(self, interp):
        schema = interp.contract.get("conditions_schema", {})
        for kind in ("alt", "reduction", "derived", "includes"):
            assert kind in schema


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Robustness checks."""

    def test_empty_combo(self, interp):
        result = interp.resolve_combo([], {"DL": 5.0})
        assert result.total_effect == 0.0
        assert result.applied_factors == {}

    def test_missing_load_case_treated_as_zero(self, interp):
        rows = [
            {"primary_key": "DL", "value": 1.2, "conditions": []},
            {"primary_key": "EQ", "value": 1.0, "conditions": []},
        ]
        result = interp.resolve_combo(rows, {"DL": 5.0})  # EQ missing
        assert math.isclose(result.total_effect, 6.0, rel_tol=1e-6)

    def test_all_zero_alt_selects_first_candidate(self, interp):
        """Contract rule: all effects=0 -> first candidate wins."""
        rows = [
            {"primary_key": "RLL", "value": 1.6, "conditions": [
                {"kind": "alt", "group": "roof_or_snow",
                 "candidates": ["RLL", "SNOW"], "choose": 1,
                 "policy": "max_effect"},
            ]},
            {"primary_key": "SNOW", "value": 1.6, "conditions": [
                {"kind": "alt", "group": "roof_or_snow",
                 "candidates": ["RLL", "SNOW"], "choose": 1,
                 "policy": "max_effect"},
            ]},
        ]
        result = interp.resolve_combo(rows, {"RLL": 0.0, "SNOW": 0.0})
        grp = result.summary["alt_groups"]["roof_or_snow"]
        assert grp["selected"] == ["RLL"]

    def test_reduction_not_allowed(self, interp):
        rows = [
            {"primary_key": "LL", "value": 1.0, "conditions": [
                {"kind": "reduction", "target": "LL",
                 "allowed": False, "factor": 0.5},
            ]},
        ]
        result = interp.resolve_combo(rows, {"LL": 2.5})
        assert result.applied_factors["LL"] == 1.0

    def test_global_reduction_disabled(self, interp):
        rows = [
            {"primary_key": "LL", "value": 1.0, "conditions": [
                {"kind": "reduction", "target": "LL",
                 "allowed": True, "factor": 0.5},
            ]},
        ]
        result = interp.resolve_combo(
            rows, {"LL": 2.5},
            options={"reduction_allowed": False},
        )
        assert result.applied_factors["LL"] == 1.0

    def test_conditions_as_single_dict(self, interp):
        """Conditions can be a single dict instead of a list."""
        rows = [
            {"primary_key": "LL", "value": 1.0,
             "conditions": {"kind": "reduction", "target": "LL",
                            "allowed": True, "factor": 0.5}},
        ]
        result = interp.resolve_combo(rows, {"LL": 2.5})
        assert result.applied_factors["LL"] == 0.5

    def test_conditions_none(self, interp):
        rows = [{"primary_key": "DL", "value": 1.4, "conditions": None}]
        result = interp.resolve_combo(rows, {"DL": 5.0})
        assert math.isclose(result.total_effect, 7.0, rel_tol=1e-6)

    def test_no_contract_yaml_still_works(self):
        """Interpreter without YAML can still resolve basic combos."""
        interp = ContractInterpreter()
        rows = [{"primary_key": "DL", "value": 1.4, "conditions": []}]
        result = interp.resolve_combo(rows, {"DL": 5.0})
        assert math.isclose(result.total_effect, 7.0, rel_tol=1e-6)
