"""Unit tests for ``services.analysis_context.build_compact_subset``.

Phase 0 Step 0-2 added five chat-tool lookups to ``analysis_context_cache``.
The helper is pure-functional, so we cover it without spinning up FastAPI
or OpenSees.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

from app.services.analysis_context import build_compact_subset  # noqa: E402


ENV = {
    "max_dx_mm": 12.3,
    "max_dy_mm": 8.1,
    "max_dz_mm": 0.5,
    "max_drift_x": 0.0042,
    "max_drift_y": 0.0031,
    "max_moment_kNm": 145.2,
    "max_shear_kN": 88.4,
    "max_axial_kN": 312.0,
}

MEMBER_INFO = [
    {
        "member_id": 1,
        "section": "H-300x300",
        "story": 1,
        "type": "column",
        "length_mm": 3500,
        "element_ids": [101, 102, 103, 104],
    },
    {
        "member_id": 2,
        "section": "H-400x200",
        "story": 1,
        "type": "beam_x",
        "length_mm": 6000,
        "element_ids": [201, 202],
    },
]

MEMBER_CHECK = {
    "status": "NG",
    "members": [
        {
            "member_id": 1,
            "status": "OK",
            "ratios": {"interaction": 0.74, "shear": 0.31, "axial": 0.5, "bending": 0.6},
        },
        {
            "member_id": 2,
            "status": "NG",
            "ratios": {"interaction": 1.08, "shear": 0.42, "axial": 0.1, "bending": 0.95},
        },
        # No matching member_info entry — should be silently skipped.
        {"member_id": 99, "status": "OK", "ratios": {"interaction": 0.1}},
    ],
}

MODAL = {
    "num_modes": 12,
    "fundamental_periods": {"T1_x_s": 0.85, "T1_y_s": 0.81, "T1_rz_s": 0.62},
    "modes": [
        {
            "mode": 1,
            "period_s": 0.85,
            "direction": "TRAN-X",
            "mass_participation_x_pct": 78.4,
            "mass_participation_y_pct": 0.2,
        },
        {
            "mode": 2,
            "period_s": 0.81,
            "direction": "TRAN-Y",
            "mass_participation": {"x_pct": 0.1, "y_pct": 79.2, "rz_pct": 0},
        },
        {
            "mode": 3,
            "period_s": 0.62,
            "direction": "ROTN-Z",
            "mass_participation_x_pct": 0.0,
            "mass_participation_y_pct": 0.0,
        },
        # A fourth mode should be dropped — we only keep top 3.
        {"mode": 4, "period_s": 0.42, "direction": "TRAN-X"},
    ],
}


def _build():
    return build_compact_subset(
        env=ENV,
        member_info_list=MEMBER_INFO,
        member_check=MEMBER_CHECK,
        modal_analysis=MODAL,
        material_name="SM355",
        num_stories=3,
        num_elements=6,
    )


def test_returns_chat_tool_keys():
    out = _build()
    # Two index variants per metric (elem_id + member_id) so the chat
    # tool can resolve either lookup style — see analysis_context's
    # _index_member_info_by_member_id docstring.
    assert set(out.keys()) == {
        "analysis_summary",
        "envelope",
        "member_info_by_elem_id",
        "member_info_by_member_id",
        "member_ratios_by_elem_id",
        "member_ratios_by_member_id",
        "modal_summary",
    }


def test_analysis_summary_scalar_fields():
    s = _build()["analysis_summary"]
    assert s["max_disp"] == {"dx_mm": 12.3, "dy_mm": 8.1, "dz_mm": 0.5}
    assert s["max_drift"] == {"x": 0.0042, "y": 0.0031}
    assert s["max_force"]["moment_kNm"] == 145.2
    assert s["ng_count"] == 1  # only member_id=2 is NG
    assert s["num_stories"] == 3
    assert s["num_elements"] == 6


def test_envelope_is_independent_copy():
    """Cache mutations must not leak back into the source ``env`` dict."""
    out = _build()
    out["envelope"]["max_dx_mm"] = 999.9
    assert ENV["max_dx_mm"] == 12.3


def test_member_info_indexed_by_elem_id():
    info = _build()["member_info_by_elem_id"]
    # All 6 element_ids resolved
    assert set(info.keys()) == {"101", "102", "103", "104", "201", "202"}
    # Sub-elements of member 1 share section/story/etype
    assert info["101"]["section"] == "H-300x300"
    assert info["101"]["story"] == 1
    assert info["101"]["etype"] == "column"
    assert info["101"]["material"] == "SM355"
    assert info["201"]["section"] == "H-400x200"
    assert info["201"]["etype"] == "beam_x"


def test_design_envelope_keyed_by_elem_id_not_member_id():
    """The chat tool gets element_id from the 3D viewer click. Keying by
    member_id would force an extra hop — by_elem_id collapses both lookups."""
    env_by_eid = _build()["member_ratios_by_elem_id"]
    # Member 1's four sub-elements all see member 1's ratios
    assert env_by_eid["101"]["ratio_interaction"] == 0.74
    assert env_by_eid["104"]["status"] == "OK"
    # Member 2 (NG) propagates to both its sub-elements
    assert env_by_eid["201"]["status"] == "NG"
    assert env_by_eid["201"]["ratio_interaction"] == 1.08
    # member_id=99 had no matching member_info → no element entries leaked
    assert all(v["member_id"] in (1, 2) for v in env_by_eid.values())


def test_member_info_indexed_by_member_id():
    """The 3D viewer sends ``member_id`` (its mesh userData carries
    minfo[member_id]) so the chat tool MUST be able to resolve a
    member_id lookup. Without this map a click on column #19 was
    mis-resolved to the member owning sub-element #19 (column #5)."""
    info_by_mid = _build()["member_info_by_member_id"]
    assert set(info_by_mid.keys()) == {"1", "2"}
    assert info_by_mid["1"]["section"] == "H-300x300"
    assert info_by_mid["1"]["story"] == 1
    assert info_by_mid["1"]["etype"] == "column"
    assert info_by_mid["2"]["section"] == "H-400x200"
    assert info_by_mid["2"]["etype"] == "beam_x"


def test_design_ratios_indexed_by_member_id():
    ratios_by_mid = _build()["member_ratios_by_member_id"]
    assert ratios_by_mid["1"]["ratio_interaction"] == 0.74
    assert ratios_by_mid["1"]["status"] == "OK"
    assert ratios_by_mid["2"]["status"] == "NG"
    assert ratios_by_mid["2"]["ratio_interaction"] == 1.08


def test_member_id_index_does_not_collide_with_elem_id_index():
    """Regression for the 'every member is 1층' bug: a member_id (1) and a
    sub-element id (101) live in DIFFERENT maps so the chat tool never
    has to disambiguate by accident. The element_id map keys with
    sub-element ids only, never member_ids."""
    out = _build()
    # member_id=1 has sub-elements 101-104. The elem_id map must NOT have
    # a key '1' (would collide with member id lookup if both shared a map).
    assert "1" not in out["member_info_by_elem_id"]
    assert "1" in out["member_info_by_member_id"]


def test_modal_summary_keeps_top_three():
    m = _build()["modal_summary"]
    assert m["num_modes"] == 12
    assert m["fundamental_periods"]["T1_x_s"] == 0.85
    assert len(m["top_modes"]) == 3
    # Reads both legacy ``mass_participation_x_pct`` and nested
    # ``mass_participation.x_pct`` shapes.
    assert m["top_modes"][0]["mass_x_pct"] == 78.4
    assert m["top_modes"][1]["mass_y_pct"] == 79.2


def test_modal_summary_preserves_zero_participation():
    """Mode 3 in the fixture is a pure torsion mode with 0 % translation
    participation. A ``a or b`` fallback would drop the 0.0 to None — the
    helper must coalesce on ``is not None`` so chat answers can still say
    "mode 3 is torsion-dominant" instead of "mode 3 participation unknown".
    """
    top3 = _build()["modal_summary"]["top_modes"][2]
    assert top3["mode"] == 3
    assert top3["mass_x_pct"] == 0.0  # NOT None
    assert top3["mass_y_pct"] == 0.0


def test_handles_missing_optional_inputs():
    out = build_compact_subset(
        env=None,
        member_info_list=None,
        member_check=None,
        modal_analysis=None,
    )
    assert out["analysis_summary"]["ng_count"] == 0
    assert out["envelope"] == {}
    assert out["member_info_by_elem_id"] == {}
    assert out["member_ratios_by_elem_id"] == {}
    assert out["modal_summary"] is None
