"""Tests for the Supabase load_combo adapter.

Uses a lightweight mock Supabase client that mimics the
``.table().select().eq().execute()`` chaining pattern of ``supabase-py``.
No real network calls are made.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from adapters.load_combo_adapter import (
    _normalize_conditions_from_db,
    fetch_combo_rows,
    normalize_combo_rows,
    resolve_combo_from_supabase,
)
from core.contract_interpreter import ContractInterpreter

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_YAML = ROOT / "docs" / "specs" / "load_contract.yaml"


# ═══════════════════════════════════════════════════════════════════════
# Mock Supabase client
# ═══════════════════════════════════════════════════════════════════════

class _MockResponse:
    """Mimics ``APIResponse`` from supabase-py."""

    def __init__(self, data: list[dict]):
        self.data = data


class _MockQuery:
    """Mimics the builder chain: ``.select().eq().eq().execute()``."""

    def __init__(self, rows: list[dict]):
        self._rows = list(rows)
        self._filters: dict[str, Any] = {}

    def select(self, columns: str = "*") -> "_MockQuery":
        return self

    def eq(self, column: str, value: Any) -> "_MockQuery":
        self._filters[column] = value
        return self

    def execute(self) -> _MockResponse:
        result = self._rows
        for col, val in self._filters.items():
            result = [r for r in result if r.get(col) == val]
        return _MockResponse(result)


class MockSupabase:
    """Lightweight Supabase client mock — no network, just in-memory filter."""

    def __init__(self, tables: dict[str, list[dict]] | None = None):
        self._tables = tables or {}

    def table(self, name: str) -> _MockQuery:
        return _MockQuery(self._tables.get(name, []))


# ═══════════════════════════════════════════════════════════════════════
# Shared test data — ULS3 rows as they would appear in Supabase
# ═══════════════════════════════════════════════════════════════════════

_BASE = {
    "param_type": "load_combo",
    "param_subtype": "uls",
    "secondary_key": "ULS3",
    "unit": "-",
    "code_id": "KDS 41 12 00",
    "code_version": "2022-10-11",
    "needs_review": False,
}

ULS3_DB_ROWS: list[dict] = [
    {**_BASE, "primary_key": "DL", "value": 1.2, "conditions": []},
    {
        **_BASE,
        "primary_key": "RLL",
        "value": 1.6,
        "conditions": [
            {
                "kind": "alt",
                "group": "roof_or_snow",
                "candidates": ["RLL", "SNOW"],
                "choose": 1,
                "policy": "max_effect",
            },
        ],
    },
    {
        **_BASE,
        "primary_key": "SNOW",
        "value": 1.6,
        "conditions": [
            {
                "kind": "alt",
                "group": "roof_or_snow",
                "candidates": ["RLL", "SNOW"],
                "choose": 1,
                "policy": "max_effect",
            },
        ],
    },
    {
        **_BASE,
        "primary_key": "LL",
        "value": 1.0,
        "conditions": [
            {
                "kind": "alt",
                "group": "live_or_wind",
                "candidates": ["LL", "WIND"],
                "choose": 1,
                "policy": "max_effect",
            },
            {
                "kind": "reduction",
                "target": "LL",
                "allowed": True,
                "factor": 0.5,
                "ref": "live_reduction",
                "condition_ko": "기본등분포활하중 ≤5.0kN/m² (주차장·공공집회 제외)",
            },
        ],
    },
    {
        **_BASE,
        "primary_key": "WIND",
        "value": 0.5,
        "conditions": [
            {
                "kind": "alt",
                "group": "live_or_wind",
                "candidates": ["LL", "WIND"],
                "choose": 1,
                "policy": "max_effect",
            },
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def interp():
    return ContractInterpreter(CONTRACT_YAML)


@pytest.fixture
def mock_client():
    return MockSupabase(tables={"load_combo": ULS3_DB_ROWS})


@pytest.fixture
def load_cases():
    return {"DL": 5.0, "LL": 2.5, "RLL": 1.0, "SNOW": 0.0, "WIND": 0.0}


# ═══════════════════════════════════════════════════════════════════════
# 1. fetch_combo_rows
# ═══════════════════════════════════════════════════════════════════════

class TestFetchComboRows:
    def test_returns_correct_count(self, mock_client):
        rows = fetch_combo_rows(
            mock_client,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="uls",
            combo_id="ULS3",
        )
        assert len(rows) == 5
        assert all(r["secondary_key"] == "ULS3" for r in rows)

    def test_wrong_combo_returns_empty(self, mock_client):
        rows = fetch_combo_rows(
            mock_client,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="uls",
            combo_id="ULS999",
        )
        assert rows == []

    def test_wrong_subtype_returns_empty(self, mock_client):
        rows = fetch_combo_rows(
            mock_client,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="sls",
            combo_id="ULS3",
        )
        assert rows == []

    def test_missing_code_id_raises(self, mock_client):
        with pytest.raises(ValueError, match="code_id"):
            fetch_combo_rows(
                mock_client,
                code_id="",
                code_version="2022-10-11",
                param_subtype="uls",
                combo_id="ULS3",
            )

    def test_missing_code_version_raises(self, mock_client):
        with pytest.raises(ValueError, match="code_version"):
            fetch_combo_rows(
                mock_client,
                code_id="KDS 41 12 00",
                code_version="",
                param_subtype="uls",
                combo_id="ULS3",
            )

    def test_missing_param_subtype_raises(self, mock_client):
        with pytest.raises(ValueError, match="param_subtype"):
            fetch_combo_rows(
                mock_client,
                code_id="KDS 41 12 00",
                code_version="2022-10-11",
                param_subtype="",
                combo_id="ULS3",
            )

    def test_missing_combo_id_raises(self, mock_client):
        with pytest.raises(ValueError, match="combo_id"):
            fetch_combo_rows(
                mock_client,
                code_id="KDS 41 12 00",
                code_version="2022-10-11",
                param_subtype="uls",
                combo_id="",
            )


# ═══════════════════════════════════════════════════════════════════════
# 2. normalize_combo_rows
# ═══════════════════════════════════════════════════════════════════════

class TestNormalizeComboRows:
    # --- normal path ---

    def test_normal_rows(self):
        rows, skipped = normalize_combo_rows(ULS3_DB_ROWS)
        assert len(rows) == 5
        assert len(skipped) == 0
        for r in rows:
            assert "primary_key" in r
            assert isinstance(r["value"], float)
            assert isinstance(r["conditions"], list)

    # --- value=None ---

    def test_value_none_skipped(self):
        extra = [{**_BASE, "primary_key": "RAIN", "value": None, "conditions": []}]
        rows, skipped = normalize_combo_rows(ULS3_DB_ROWS + extra)
        assert len(rows) == 5
        assert len(skipped) == 1
        assert skipped[0]["primary_key"] == "RAIN"
        assert "value is None" in skipped[0]["_skip_reason"]

    # --- needs_review ---

    def test_needs_review_skipped_by_default(self):
        extra = [
            {
                **_BASE,
                "primary_key": "EQ",
                "value": 1.0,
                "conditions": [],
                "needs_review": True,
                "review_note": "unclear coefficient",
            },
        ]
        rows, skipped = normalize_combo_rows(ULS3_DB_ROWS + extra)
        assert len(rows) == 5
        assert len(skipped) == 1
        assert "needs_review" in skipped[0]["_skip_reason"]

    def test_needs_review_kept_when_flag_off(self):
        db = [
            {"primary_key": "EQ", "value": 1.0, "conditions": [], "needs_review": True},
        ]
        rows, skipped = normalize_combo_rows(db, skip_needs_review=False)
        assert len(rows) == 1
        assert len(skipped) == 0

    # --- missing primary_key ---

    def test_missing_primary_key_skipped(self):
        db = [{"value": 1.0, "conditions": []}]
        rows, skipped = normalize_combo_rows(db)
        assert len(rows) == 0
        assert len(skipped) == 1
        assert "missing primary_key" in skipped[0]["_skip_reason"]

    # --- conditions formats ---

    def test_conditions_dict(self):
        db = [
            {
                "primary_key": "LL",
                "value": 1.0,
                "conditions": {"kind": "reduction", "target": "LL", "allowed": True, "factor": 0.5},
            },
        ]
        rows, _ = normalize_combo_rows(db)
        assert isinstance(rows[0]["conditions"], list)
        assert len(rows[0]["conditions"]) == 1
        assert rows[0]["conditions"][0]["kind"] == "reduction"

    def test_conditions_none(self):
        db = [{"primary_key": "DL", "value": 1.4, "conditions": None}]
        rows, _ = normalize_combo_rows(db)
        assert rows[0]["conditions"] == []

    def test_conditions_json_string(self):
        db = [
            {
                "primary_key": "DL",
                "value": 1.2,
                "conditions": '[{"kind": "alt", "group": "test"}]',
            },
        ]
        rows, _ = normalize_combo_rows(db)
        assert isinstance(rows[0]["conditions"], list)
        assert rows[0]["conditions"][0]["kind"] == "alt"

    def test_mixed_conditions_formats(self):
        db = [
            {"primary_key": "A", "value": 1.0, "conditions": None},
            {"primary_key": "B", "value": 2.0, "conditions": {"kind": "derived"}},
            {"primary_key": "C", "value": 3.0, "conditions": [{"kind": "alt", "group": "g"}]},
            {"primary_key": "D", "value": 4.0, "conditions": "[]"},
        ]
        rows, skipped = normalize_combo_rows(db)
        assert len(rows) == 4
        assert len(skipped) == 0
        assert rows[0]["conditions"] == []
        assert rows[1]["conditions"] == [{"kind": "derived"}]
        assert rows[2]["conditions"] == [{"kind": "alt", "group": "g"}]
        assert rows[3]["conditions"] == []


# ═══════════════════════════════════════════════════════════════════════
# 3. _normalize_conditions_from_db (edge cases)
# ═══════════════════════════════════════════════════════════════════════

class TestNormalizeConditionsFromDB:
    def test_none(self):
        assert _normalize_conditions_from_db(None) == []

    def test_empty_list(self):
        assert _normalize_conditions_from_db([]) == []

    def test_dict(self):
        assert _normalize_conditions_from_db({"kind": "alt"}) == [{"kind": "alt"}]

    def test_json_string_list(self):
        assert _normalize_conditions_from_db('[{"kind": "alt"}]') == [{"kind": "alt"}]

    def test_json_string_dict(self):
        assert _normalize_conditions_from_db('{"kind": "reduction"}') == [{"kind": "reduction"}]

    def test_invalid_string(self):
        assert _normalize_conditions_from_db("not json") == []

    def test_integer(self):
        assert _normalize_conditions_from_db(42) == []


# ═══════════════════════════════════════════════════════════════════════
# 4. resolve_combo_from_supabase (full integration)
# ═══════════════════════════════════════════════════════════════════════

class TestResolveComboFromSupabase:
    def test_full_uls3(self, interp, mock_client, load_cases):
        result = resolve_combo_from_supabase(
            interp,
            mock_client,
            load_cases,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="uls",
            combo_id="ULS3",
        )
        assert math.isclose(result.total_effect, 8.85, rel_tol=1e-6)
        assert result.combo_id == "ULS3"

        # adapter provenance in summary
        ad = result.summary["adapter"]
        assert ad["source"] == "supabase"
        assert ad["combo_id"] == "ULS3"
        assert ad["fetched"] == 5
        assert ad["usable"] == 5
        assert ad["skipped"] == 0

    def test_with_skipped_null_value(self, interp, load_cases):
        extra = [{**_BASE, "primary_key": "RAIN", "value": None, "conditions": []}]
        client = MockSupabase(tables={"load_combo": ULS3_DB_ROWS + extra})
        result = resolve_combo_from_supabase(
            interp,
            client,
            load_cases,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="uls",
            combo_id="ULS3",
        )
        # RAIN is skipped, total is still 8.85
        assert math.isclose(result.total_effect, 8.85, rel_tol=1e-6)
        ad = result.summary["adapter"]
        assert ad["skipped"] == 1
        assert ad["skipped_details"][0]["primary_key"] == "RAIN"

    def test_with_skipped_needs_review(self, interp, load_cases):
        extra = [
            {
                **_BASE,
                "primary_key": "EQ",
                "value": 1.0,
                "conditions": [],
                "needs_review": True,
                "review_note": "uncertain",
            },
        ]
        client = MockSupabase(tables={"load_combo": ULS3_DB_ROWS + extra})
        result = resolve_combo_from_supabase(
            interp,
            client,
            load_cases,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="uls",
            combo_id="ULS3",
        )
        assert math.isclose(result.total_effect, 8.85, rel_tol=1e-6)
        ad = result.summary["adapter"]
        assert ad["skipped"] == 1
        assert "needs_review" in ad["skipped_details"][0]["reason"]

    def test_empty_result_for_unknown_combo(self, interp, mock_client, load_cases):
        result = resolve_combo_from_supabase(
            interp,
            mock_client,
            load_cases,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="uls",
            combo_id="NONEXISTENT",
        )
        assert result.total_effect == 0.0
        assert result.summary["adapter"]["fetched"] == 0

    def test_alt_and_reduction_work_through_adapter(self, interp, mock_client, load_cases):
        """Verify that adapter normalisation preserves conditions for interpreter."""
        result = resolve_combo_from_supabase(
            interp,
            mock_client,
            load_cases,
            code_id="KDS 41 12 00",
            code_version="2022-10-11",
            param_subtype="uls",
            combo_id="ULS3",
        )
        # Alt groups resolved
        assert "roof_or_snow" in result.summary["alt_groups"]
        assert "live_or_wind" in result.summary["alt_groups"]
        # Reduction applied
        reds = [r for r in result.summary["reductions"] if r["target"] == "LL"]
        assert len(reds) == 1
        assert reds[0]["applied"] is True
        assert result.applied_factors["LL"] == 0.5
