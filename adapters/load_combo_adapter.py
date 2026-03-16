"""Supabase adapter for load_combo view -> ContractInterpreter.

Bridges the Supabase ``load_combo`` view (backed by ``load_params``) with
``ContractInterpreter.resolve_combo``.

The query and normalisation helpers are intentionally generic so that future
param_type adapters (snow_load, wind_load, …) can reuse them with minimal
duplication.

Usage::

    from supabase import create_client
    from core.contract_interpreter import ContractInterpreter
    from adapters.load_combo_adapter import resolve_combo_from_supabase

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    interp = ContractInterpreter("docs/specs/load_contract.yaml")

    load_cases = {"DL": 5.0, "LL": 2.5, "RLL": 1.0, "SNOW": 0.0, "WIND": 0.0}

    result = resolve_combo_from_supabase(
        interp, client, load_cases,
        code_id="KDS 41 12 00",
        code_version="2022-10-11",
        param_subtype="uls",
        combo_id="ULS3",
    )
    print(result.total_effect)  # 8.85
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.contract_interpreter import ComboResult, ContractInterpreter

logger = logging.getLogger(__name__)


# ======================================================================
# Fetch
# ======================================================================

def fetch_combo_rows(
    client: Any,
    *,
    code_id: str,
    code_version: str,
    param_subtype: str,
    combo_id: str,
    table_or_view: str = "load_combo",
) -> list[dict]:
    """Query a Supabase view/table and return raw DB rows.

    Parameters
    ----------
    client :
        A ``supabase.Client`` (or any object supporting the
        ``.table().select().eq().execute()`` chain).
    code_id :
        Design code identifier, e.g. ``"KDS 41 12 00"``.
    code_version :
        Code version date, e.g. ``"2022-10-11"``.
    param_subtype :
        ``"uls"`` or ``"sls"``.
    combo_id :
        Combination key, e.g. ``"ULS3"``.
    table_or_view :
        Supabase table/view name (default ``"load_combo"``).

    Returns
    -------
    list[dict]
        Raw rows as returned by Supabase.

    Raises
    ------
    ValueError
        If any required parameter is empty/falsy.
    """
    if not code_id:
        raise ValueError("code_id is required")
    if not code_version:
        raise ValueError("code_version is required")
    if not param_subtype:
        raise ValueError("param_subtype is required (e.g. 'uls' or 'sls')")
    if not combo_id:
        raise ValueError("combo_id is required")

    response = (
        client
        .table(table_or_view)
        .select("*")
        .eq("secondary_key", combo_id)
        .eq("param_subtype", param_subtype)
        .eq("code_id", code_id)
        .eq("code_version", code_version)
        .execute()
    )
    return response.data


# ======================================================================
# Normalize
# ======================================================================

def normalize_combo_rows(
    db_rows: list[dict],
    *,
    skip_needs_review: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Convert raw Supabase rows to ContractInterpreter-ready ``combo_rows``.

    Parameters
    ----------
    db_rows :
        Raw rows from :func:`fetch_combo_rows`.
    skip_needs_review :
        If *True* (default), rows with ``needs_review=True`` are moved to
        the *skipped* list instead of being included.

    Returns
    -------
    (usable, skipped)
        *usable* — ``list[dict]`` ready for ``resolve_combo``.
        *skipped* — rows that were excluded, each augmented with
        ``_skip_reason``.
    """
    usable: list[dict] = []
    skipped: list[dict] = []

    for row in db_rows:
        pk = row.get("primary_key")
        if not pk:
            skipped.append({**row, "_skip_reason": "missing primary_key"})
            continue

        needs_review = row.get("needs_review", False)
        if needs_review and skip_needs_review:
            note = row.get("review_note", "")
            skipped.append({
                **row,
                "_skip_reason": f"needs_review=True (review_note: {note})",
            })
            continue

        value = row.get("value")
        if value is None:
            skipped.append({**row, "_skip_reason": "value is None"})
            continue

        usable.append({
            "primary_key": pk,
            "value": float(value),
            "conditions": _normalize_conditions_from_db(row.get("conditions")),
            "notes": row.get("notes"),
        })

    if skipped:
        logger.info(
            "normalize_combo_rows: %d rows skipped (of %d total)",
            len(skipped),
            len(db_rows),
        )

    return usable, skipped


def _normalize_conditions_from_db(raw: Any) -> list[dict]:
    """Robustly normalise the ``conditions`` column from DB.

    Handles every shape that Supabase / PostgreSQL jsonb might deliver:

    * ``None``            -> ``[]``
    * ``list[dict]``      -> pass-through
    * ``dict``            -> ``[dict]``
    * JSON *string*       -> parse then re-normalise
    * Anything else       -> ``[]`` (with a warning)
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [_ensure_dict(item) for item in raw]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("conditions: unparseable string %r", raw)
            return []
        return _normalize_conditions_from_db(parsed)
    logger.warning("conditions: unexpected type %s", type(raw).__name__)
    return []


def _ensure_dict(item: Any) -> dict:
    """Best-effort coercion of a single condition element to ``dict``."""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {"_raw": item}


# ======================================================================
# Integration
# ======================================================================

def resolve_combo_from_supabase(
    interp: ContractInterpreter,
    client: Any,
    load_cases: dict[str, float],
    *,
    code_id: str,
    code_version: str,
    param_subtype: str,
    combo_id: str,
    options: dict | None = None,
) -> ComboResult:
    """One-call convenience: fetch → normalise → resolve.

    Parameters
    ----------
    interp :
        A loaded :class:`ContractInterpreter`.
    client :
        Supabase client (or mock).
    load_cases :
        Unfactored load-case values.
    code_id, code_version, param_subtype, combo_id :
        Filter parameters forwarded to :func:`fetch_combo_rows`.
    options :
        Passed through to ``interp.resolve_combo``.

    Returns
    -------
    ComboResult
        The interpreter result, with ``summary["adapter"]`` containing
        fetch/skip metadata.
    """
    db_rows = fetch_combo_rows(
        client,
        code_id=code_id,
        code_version=code_version,
        param_subtype=param_subtype,
        combo_id=combo_id,
    )

    combo_rows, skipped = normalize_combo_rows(db_rows)

    if not combo_rows:
        logger.warning(
            "No usable combo rows for %s (fetched %d, skipped %d)",
            combo_id,
            len(db_rows),
            len(skipped),
        )

    result = interp.resolve_combo(
        combo_rows,
        load_cases,
        combo_id=combo_id,
        options=options,
    )

    # Attach adapter provenance to the summary
    result.summary["adapter"] = {
        "source": "supabase",
        "table_or_view": "load_combo",
        "code_id": code_id,
        "code_version": code_version,
        "param_subtype": param_subtype,
        "combo_id": combo_id,
        "fetched": len(db_rows),
        "usable": len(combo_rows),
        "skipped": len(skipped),
        "skipped_details": [
            {
                "primary_key": s.get("primary_key", "?"),
                "reason": s.get("_skip_reason", "?"),
            }
            for s in skipped
        ]
        if skipped
        else [],
    }

    return result
