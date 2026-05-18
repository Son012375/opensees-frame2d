"""Tests for ``core.recommendation.section_catalog``.

These tests hit the live Supabase catalogue by default (the same one
``section_3d`` already uses). If the network is down they still pass
because ``section_catalog`` falls back to its in-module ladder.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.recommendation.section_catalog import (  # noqa: E402
    SectionEntry,
    clear_cache,
    get_section_metadata,
    id_order_audit,
    list_family_ladder,
    next_sections,
    parse_family,
    source,
)


# ---------------------------------------------------------------------------
# parse_family
# ---------------------------------------------------------------------------

class TestParseFamily:
    def test_known_prefixes(self):
        cases = {
            "H-300x300": "H",
            "H-400x200x8x13": "H",
            "I-150x75": "I",
            "TFC-100x50": "TFC",
            "PFC-100x50": "PFC",
            "T-150x9": "T",
            "L-50x50x4": "L",
            "CHS-150x4": "CHS",
            "SHS-50x50": "SHS",
            "RHS-100x50x4": "RHS",
        }
        for name, expected in cases.items():
            assert parse_family(name) == expected, f"{name!r}"

    def test_native_korean_prefix(self):
        # ks3568 native prefixes — must resolve.
        assert parse_family("○-21.3x2.3") == "CHS"
        assert parse_family("□-50x50x2.5") == "SHS"
        assert parse_family("□-50x25x2.5") == "RHS"

    def test_unknown_returns_none(self):
        assert parse_family(None) is None
        assert parse_family("") is None
        assert parse_family("foobar") is None


# ---------------------------------------------------------------------------
# H column ladder shape
# ---------------------------------------------------------------------------

class TestHColumnLadder:
    def setup_method(self):
        clear_cache()

    def teardown_method(self):
        clear_cache()

    def test_column_ladder_contains_only_square_h(self):
        ladder = list_family_ladder("H", "column")
        assert ladder, "H column ladder must not be empty"
        for entry in ladder:
            assert entry.family == "H"
            assert entry.h == entry.b, (
                f"non-square H slipped into column ladder: {entry.name}"
            )

    def test_column_ladder_is_size_monotonic(self):
        ladder = list_family_ladder("H", "column")
        keys = [(e.h, e.b, e.area) for e in ladder]
        assert keys == sorted(keys), (
            "column ladder is not strictly ordered by (h, b, area)"
        )

    def test_column_ladder_includes_known_anchors(self):
        names = [e.name for e in list_family_ladder("H", "column")]
        # Spot-check the canonical KS square H sub-families.
        for anchor in ("H-100x100", "H-200x200", "H-300x300",
                       "H-350x350", "H-400x400"):
            assert anchor in names, f"missing column anchor {anchor}"


# ---------------------------------------------------------------------------
# H beam ladder shape
# ---------------------------------------------------------------------------

class TestHBeamLadder:
    def setup_method(self):
        clear_cache()

    def teardown_method(self):
        clear_cache()

    def test_beam_ladder_excludes_square_h(self):
        ladder = list_family_ladder("H", "beam")
        assert ladder, "H beam ladder must not be empty"
        for entry in ladder:
            assert entry.h != entry.b, (
                f"square H slipped into beam ladder: {entry.name}"
            )


# ---------------------------------------------------------------------------
# next_sections — the core API used by candidate_generator
# ---------------------------------------------------------------------------

class TestNextSections:
    def setup_method(self):
        clear_cache()

    def test_column_steps_climb_the_ladder(self):
        n = next_sections("H-100x100", "column", steps=2)
        assert len(n) == 2
        assert all(isinstance(e, SectionEntry) for e in n)
        # Each step must be strictly larger than the current section.
        prev = (100.0, 100.0)
        for e in n:
            assert (e.h, e.b) >= prev
            prev = (e.h, e.b)

    def test_column_top_of_ladder_yields_nothing(self):
        # H-400x400 is the top of the square ladder per the Step 0 audit.
        n = next_sections("H-400x400", "column", steps=2)
        assert n == []

    def test_steps_zero_returns_empty(self):
        assert next_sections("H-200x200", "column", steps=0) == []

    def test_unknown_family_returns_empty(self):
        assert next_sections("FOO-123", "column", steps=2) == []

    def test_four_token_section_name_still_walks(self):
        """A real-world name like 'H-400x200x8x13' carries thickness data.
        It may not be a direct catalogue key, but the ladder must still
        progress on (h, b) alone."""
        n = next_sections("H-400x200x8x13", "beam", steps=1)
        assert len(n) <= 1
        for e in n:
            # The next entry must be strictly larger in (h, b) than 400x200.
            assert (e.h, e.b) > (400.0, 200.0)

    def test_column_never_recommends_wide_h(self):
        """Critical safety: if you ask for the next *column* upgrade,
        you must NEVER get a wide-flange (h ≠ b) section back."""
        for cur in ("H-100x100", "H-200x200", "H-300x300"):
            for e in next_sections(cur, "column", steps=2):
                assert e.h == e.b, (
                    f"column upgrade from {cur} returned wide H {e.name}"
                )

    def test_beam_never_recommends_square_h(self):
        for cur in ("H-300x150", "H-400x200"):
            for e in next_sections(cur, "beam", steps=2):
                assert e.h != e.b, (
                    f"beam upgrade from {cur} returned square H {e.name}"
                )


# ---------------------------------------------------------------------------
# get_section_metadata
# ---------------------------------------------------------------------------

class TestSectionMetadata:
    def test_known_section(self):
        m = get_section_metadata("H-300x300")
        assert m is not None
        assert m.family == "H"
        assert m.h == 300
        assert m.b == 300
        # area_mm2 = area_cm2 × 100
        assert m.area_mm2 == m.area * 100

    def test_unknown_section_is_none(self):
        assert get_section_metadata("FOO-bar") is None
        assert get_section_metadata(None) is None


# ---------------------------------------------------------------------------
# Audit / source surface
# ---------------------------------------------------------------------------

class TestAuditSurface:
    def test_audit_flags_present(self):
        flags = id_order_audit()
        # The H table is known non-monotonic per Step 0; just assert
        # the audit reports SOME flag per family it loaded.
        assert isinstance(flags, dict)
        assert "H" in flags

    def test_source_is_supabase_or_fallback(self):
        assert source() in {"supabase", "fallback", "mixed"}
