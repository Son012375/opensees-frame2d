"""Load-case builder for the Apply MVP pipeline.

Responsibilities:
  A. Look up live-load intensity from local DB JSON (or Supabase).
  B. Apply simplified live-load reduction if allowed.
  C. Convert floor area loads (kN/m2) → edge-beam line loads (kN/m).
  D. Produce ``load_cases_global`` (for ContractInterpreter) and
     ``load_application_plan`` (for OpenSees model).

Design rule: **no LLM-generated numbers** — every value is either from
the DB, from ``loads_user``, or from an explicit fallback with a warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from apply_mvp.schemas import FloorInfo, LoadCasesResult, LoadsUser, MvpInput

logger = logging.getLogger(__name__)

# Default fallback live-load when DB has no match and user didn't supply one
_LL_FALLBACK_KN_M2 = 2.5

# Path to the normalised live-load DB extracted from KDS
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "kds_output" / "03_normalized.json"


# ======================================================================
# DB lookup helpers
# ======================================================================

def _load_local_db(path: Path | None = None) -> list[dict]:
    """Load normalised records from the local JSON DB."""
    p = path or _DEFAULT_DB_PATH
    if not p.exists():
        logger.warning("Local DB not found at %s", p)
        return []
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("records", data) if isinstance(data, dict) else data


def lookup_live_load(
    occupancy_key: str,
    *,
    db_records: list[dict] | None = None,
    supabase_client: Any = None,
) -> tuple[float | None, str]:
    """Return (value_kN_m2, source) for the given occupancy_key.

    Strategy:
      1. If ``supabase_client`` is provided, try remote query first.
      2. Else fall back to local JSON DB.
      3. Among matches, pick the "general" variant (secondary_key containing
         the occupancy + '_general' or the first distributed match).

    Returns
    -------
    (value, source)
        *source* is ``"supabase"``, ``"local_db"``, or ``"not_found"``.
    """
    # --- Supabase path (future; skip if no client) ---
    if supabase_client is not None:
        try:
            resp = (
                supabase_client
                .table("load_params")
                .select("value, primary_key, secondary_key")
                .eq("param_type", "live_load")
                .eq("param_subtype", "distributed")
                .eq("primary_key", occupancy_key)
                .execute()
            )
            rows = resp.data or []
            if rows:
                # prefer _general variant
                for r in rows:
                    sk = r.get("secondary_key", "") or ""
                    if "general" in sk or not sk:
                        return float(r["value"]), "supabase"
                return float(rows[0]["value"]), "supabase"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase lookup failed (%s); falling back to local DB", exc)

    # --- Local DB path ---
    records = db_records if db_records is not None else _load_local_db()
    candidates = [
        r for r in records
        if r.get("primary_key") == occupancy_key
        and r.get("record_type") == "live_load"
        and r.get("param_subtype") == "distributed"
        and r.get("value") is not None
    ]
    if candidates:
        # prefer _general variant
        for c in candidates:
            sk = c.get("secondary_key", "") or ""
            if "general" in sk or not sk:
                return float(c["value"]), "local_db"
        return float(candidates[0]["value"]), "local_db"

    return None, "not_found"


# ======================================================================
# Main builder
# ======================================================================

def build_load_cases(
    mvp_input: MvpInput,
    *,
    supabase_client: Any = None,
    db_records: list[dict] | None = None,
) -> LoadCasesResult:
    """Create ``load_cases_global`` and ``load_application_plan``.

    Parameters
    ----------
    mvp_input :
        Parsed MVP input.
    supabase_client :
        Optional Supabase client for live-load lookup.
    db_records :
        Pre-loaded local DB records (avoids re-reading file each call).
    """
    loads = mvp_input.loads_user
    warnings: list[str] = []
    ll_source = ""

    # --- A. Determine LL intensity (kN/m2) from DB ---
    # Use the first floor's occupancy_key as the representative key.
    occ_key = (
        mvp_input.building.floors[0].occupancy_key
        if mvp_input.building.floors
        else "office"
    )

    ll_val, src = lookup_live_load(
        occ_key,
        db_records=db_records,
        supabase_client=supabase_client,
    )

    if ll_val is not None:
        ll_source = src
        logger.info("LL from %s: %.2f kN/m2 (occupancy=%s)", src, ll_val, occ_key)
    elif loads.LL_fallback_kN_per_m2 is not None:
        ll_val = loads.LL_fallback_kN_per_m2
        ll_source = "user_fallback"
        warnings.append(
            f"LL not found in DB for '{occ_key}'; using user fallback {ll_val} kN/m2"
        )
    else:
        ll_val = _LL_FALLBACK_KN_M2
        ll_source = "default_fallback"
        warnings.append(
            f"LL not found in DB for '{occ_key}'; needs_review — using default {ll_val} kN/m2"
        )

    # --- B. Simplified live-load reduction ---
    # MVP: if floor_area_m2 > 37.2 m2 (4 * 9.3 m2 tributary threshold),
    # flag reduction as "allowed".  The actual reduction factor comes from
    # the combo conditions (ContractInterpreter), not computed here.
    # We just note it for logging.
    for fl in mvp_input.building.floors:
        if fl.floor_area_m2 > 37.2:
            logger.info(
                "Floor %d: area=%.1f m2 > 37.2 -> live-load reduction eligible",
                fl.level,
                fl.floor_area_m2,
            )

    # --- C. Build load_cases_global (representative kN/m2 scalars) ---
    load_cases_global: dict[str, float] = {
        "DL":   loads.DL_kN_per_m2,
        "SDL":  loads.SDL_kN_per_m2,
        "LL":   ll_val,
        "RLL":  0.0,   # MVP: no roof modelling → 0
        "SNOW": loads.snow_base_kN_per_m2,
        "WIND": loads.wind_base_kN_per_m2,
        "EQ":   loads.eq_base_kN_per_m2,
    }

    # --- D. Build per-floor load_application_plan ---
    load_application_plan: dict[int, dict] = {}
    for fl in mvp_input.building.floors:
        plan: dict[str, dict] = {}
        for case_key, intensity in load_cases_global.items():
            total_kN = intensity * fl.floor_area_m2
            w = total_kN / fl.edge_beam_total_length_m if fl.edge_beam_total_length_m > 0 else 0.0
            plan[case_key] = {
                "type": "line",
                "intensity_kN_per_m2": intensity,
                "total_kN": round(total_kN, 4),
                "w_kN_per_m": round(w, 4),
            }
        load_application_plan[fl.level] = plan

    return LoadCasesResult(
        load_cases_global=load_cases_global,
        load_application_plan=load_application_plan,
        warnings=warnings,
        ll_source=ll_source,
    )
