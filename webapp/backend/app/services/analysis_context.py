"""In-process cache for ``/api/v2/analyze`` baselines.

``analysis_context_cache[analysis_id]`` holds the model JSON, load cases,
combos, building model, seismic report, design-check baseline, and
candidate-by-id index that the recommendation endpoints need to reproduce
a run. Entries expire after :data:`_ANALYSIS_CONTEXT_TTL_SEC`.

The cache and its lock live here (rather than in ``main_simple``) so the
chat router can read them without pulling main_simple's full transitive
import set.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

analysis_context_cache: dict[str, dict] = {}
_ANALYSIS_CONTEXT_TTL_SEC = 30 * 60  # 30 minutes
_ANALYSIS_CONTEXT_LOCK = threading.Lock()


def _purge_expired_analysis_contexts() -> None:
    """Drop entries past their TTL. Called opportunistically on each
    cache write — no background thread."""
    now = time.time()
    with _ANALYSIS_CONTEXT_LOCK:
        expired = [
            k for k, v in analysis_context_cache.items()
            if v.get("expires_at", 0) < now
        ]
        for k in expired:
            analysis_context_cache.pop(k, None)


def _get_analysis_context(analysis_id: str) -> Optional[dict]:
    """TTL-aware read for ``analysis_context_cache``.

    Returns the cached context if present and unexpired, otherwise
    evicts the (expired) entry and returns ``None``. Both the POST
    endpoint and the background worker call this single helper so the
    TTL is enforced on every read — not only when /api/v2/analyze
    happens to run and triggers the bulk purge.
    """
    now = time.time()
    with _ANALYSIS_CONTEXT_LOCK:
        ctx = analysis_context_cache.get(analysis_id)
        if ctx is None:
            return None
        if ctx.get("expires_at", 0) < now:
            analysis_context_cache.pop(analysis_id, None)
            return None
        return ctx


# Public aliases — callers outside ``main_simple`` should prefer these
# names. The underscore-prefixed originals are kept so existing imports
# (including the pytest suite) continue to work unchanged.
purge_expired = _purge_expired_analysis_contexts
get_context = _get_analysis_context


# ---------------------------------------------------------------------------
# Compact subset for the chat router (Phase 0 Step 0-2)
# ---------------------------------------------------------------------------
#
# The chat router's `inspect_selection` and `get_analysis_summary` tools need
# fast, small lookups by element_id and a scalar overview of the last run.
# Computing these from the full ``model_json`` + ``multi.member_forces`` per
# request would be wasteful, so /api/v2/analyze pre-derives them once and
# stows the result inside ``analysis_context_cache[analysis_id]``.

def _count_ng_members(member_check: Optional[dict]) -> int:
    if not member_check:
        return 0
    members = member_check.get("members") or []
    return sum(1 for m in members if m.get("status") == "NG")


def _build_info(m: dict, material_name: Optional[str]) -> dict:
    return {
        "member_id": m.get("member_id"),
        "section": m.get("section"),
        "material": material_name,
        "story": m.get("story"),
        "etype": m.get("type") or m.get("elem_type"),
        "length_mm": m.get("length_mm"),
    }


def _index_member_info(
    member_info_list: Optional[list[dict]],
    material_name: Optional[str],
) -> dict[str, dict]:
    """Explode ``multi.member_info`` (one entry per member, each carrying a
    list of sub-element ids) into a lookup keyed by OpenSees sub-element
    id. The 3D viewer doesn't expose sub-element ids today (selection
    sends ``member_id`` — see :func:`_index_member_info_by_member_id`),
    but the keys are kept so a future feature can resolve an explicit
    sub-element reference (e.g. peak-force probe at element midspan)."""
    result: dict[str, dict] = {}
    if not member_info_list:
        return result
    for m in member_info_list:
        info = _build_info(m, material_name)
        for eid in (m.get("element_ids") or []):
            result[str(eid)] = info
    return result


def _index_member_info_by_member_id(
    member_info_list: Optional[list[dict]],
    material_name: Optional[str],
) -> dict[str, dict]:
    """Same payload as :func:`_index_member_info` but keyed by ``member_id``.

    The V2 Editor's 3D viewer attaches ``member_id`` to each mesh
    (``viewer_elements[i].id = minfo["member_id"]`` in main_simple.py)
    so a click sends member ids in ``ui_context.selected_element_ids``
    despite the key name. Without this lookup, ``inspect_selection``
    would resolve a click on column #19 to whichever member happens to
    have a sub-element id of 19 — which (at num_elements_per_member=4)
    is column #5 of the 1st story. That's the source of the "every
    member is 1층" misreport seen in live smoke."""
    result: dict[str, dict] = {}
    if not member_info_list:
        return result
    for m in member_info_list:
        mid = m.get("member_id")
        if mid is None:
            continue
        result[str(mid)] = _build_info(m, material_name)
    return result


def _build_ratios_by_mid(member_check: Optional[dict]) -> dict:
    """Pull ratios+status off design_check's member_check, keyed by
    member_id. Internal helper for both per-element and per-member
    lookups."""
    if not member_check:
        return {}
    by_mid: dict = {}
    for m in (member_check.get("members") or []):
        mid = m.get("member_id")
        ratios = m.get("ratios") or {}
        by_mid[mid] = {
            "member_id": mid,
            "status": m.get("status", "OK"),
            "ratio_interaction": ratios.get("interaction", 0),
            "ratio_shear": ratios.get("shear", 0),
            "ratio_axial": ratios.get("axial", 0),
            "ratio_bending": ratios.get("bending", 0),
        }
    return by_mid


def _design_ratios_by_elem(
    member_check: Optional[dict],
    member_info_list: Optional[list[dict]],
) -> dict[str, dict]:
    """Per sub-element_id lookup for design ratios. See sibling
    :func:`_design_ratios_by_member_id` for the keying used by the live
    chat viewer; this map is retained for potential future tools that
    operate on raw OpenSees sub-elements."""
    if not member_check or not member_info_list:
        return {}
    by_mid = _build_ratios_by_mid(member_check)
    out: dict[str, dict] = {}
    for m_info in member_info_list:
        mid = m_info.get("member_id")
        env = by_mid.get(mid)
        if env is None:
            continue
        for eid in (m_info.get("element_ids") or []):
            out[str(eid)] = env
    return out


def _design_ratios_by_member_id(
    member_check: Optional[dict],
) -> dict[str, dict]:
    """Same content as :func:`_design_ratios_by_elem` but keyed by
    member_id — what the 3D viewer sends in ``ui_context``. Required
    for the chat ``inspect_selection`` tool to return ratios that
    actually match the clicked column instead of the sub-element with
    a colliding numeric id."""
    by_mid = _build_ratios_by_mid(member_check)
    return {str(mid): env for mid, env in by_mid.items() if mid is not None}


def _coalesce(*values):
    """Return the first value that is not ``None``. Unlike ``a or b`` this
    preserves meaningful falsy numbers like ``0.0`` — important for modal
    mass participation, where 0 % is a real measurement, not a missing one.
    """
    for v in values:
        if v is not None:
            return v
    return None


def _modal_subset(modal_analysis: Optional[dict]) -> Optional[dict]:
    if not modal_analysis:
        return None
    modes = (modal_analysis.get("modes") or [])[:3]
    return {
        "num_modes": modal_analysis.get("num_modes", 0),
        "fundamental_periods": modal_analysis.get("fundamental_periods", {}),
        "top_modes": [
            {
                "mode": _coalesce(m.get("mode"), m.get("mode_num")),
                "period_s": m.get("period_s"),
                "direction": m.get("direction"),
                "mass_x_pct": _coalesce(
                    m.get("mass_participation_x_pct"),
                    (m.get("mass_participation") or {}).get("x_pct"),
                ),
                "mass_y_pct": _coalesce(
                    m.get("mass_participation_y_pct"),
                    (m.get("mass_participation") or {}).get("y_pct"),
                ),
            }
            for m in modes
        ],
    }


def build_compact_subset(
    *,
    env: Optional[dict],
    member_info_list: Optional[list[dict]],
    member_check: Optional[dict],
    modal_analysis: Optional[dict],
    material_name: Optional[str] = None,
    num_stories: int = 0,
    num_elements: int = 0,
) -> dict:
    """Derive the five chat-tool lookups for ``analysis_context_cache``.

    Pure transformation — no I/O, no global state. Safe to call from any
    thread. Returns a dict with the keys ``analysis_summary``, ``envelope``,
    ``member_info_by_elem_id``, ``member_ratios_by_elem_id`` (design
    interaction/shear/axial/bending ratios, not raw N/V/M), and
    ``modal_summary``. Raw force envelopes by element are intentionally not
    pre-derived — if a future tool needs them, compute on demand from
    ``multi.member_forces`` to keep this cache slice small.
    """
    env = env or {}
    return {
        "analysis_summary": {
            "max_disp": {
                "dx_mm": env.get("max_dx_mm", 0),
                "dy_mm": env.get("max_dy_mm", 0),
                "dz_mm": env.get("max_dz_mm", 0),
            },
            "max_drift": {
                "x": env.get("max_drift_x", 0),
                "y": env.get("max_drift_y", 0),
            },
            "max_force": {
                "moment_kNm": env.get("max_moment_kNm", 0),
                "shear_kN": env.get("max_shear_kN", 0),
                "axial_kN": env.get("max_axial_kN", 0),
            },
            "ng_count": _count_ng_members(member_check),
            "num_stories": num_stories,
            "num_elements": num_elements,
        },
        "envelope": dict(env),
        # Two indexes per metric: the live viewer sends member_id (mesh
        # userData carries minfo["member_id"]), but the keys read like
        # "element_ids" so future code paths that surface a real sub-
        # element id can still resolve via the _by_elem_id maps.
        "member_info_by_elem_id": _index_member_info(member_info_list, material_name),
        "member_info_by_member_id": _index_member_info_by_member_id(
            member_info_list, material_name,
        ),
        "member_ratios_by_elem_id": _design_ratios_by_elem(
            member_check, member_info_list,
        ),
        "member_ratios_by_member_id": _design_ratios_by_member_id(member_check),
        "modal_summary": _modal_subset(modal_analysis),
    }
