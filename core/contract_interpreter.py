"""KDS Load Combination Contract Interpreter.

Deterministically resolves load combinations per LOAD_DB_CONTRACT rules (section 4).
Reads load_contract.yaml for schema definitions and direction expansion rules.

Usage::

    from core.contract_interpreter import ContractInterpreter

    interp = ContractInterpreter("docs/specs/load_contract.yaml")

    combo_rows = [
        {"primary_key": "DL",   "value": 1.2, "conditions": []},
        {"primary_key": "RLL",  "value": 1.6, "conditions": [
            {"kind": "alt", "group": "roof_or_snow",
             "candidates": ["RLL", "SNOW"], "choose": 1, "policy": "max_effect"}
        ]},
        {"primary_key": "SNOW", "value": 1.6, "conditions": [
            {"kind": "alt", "group": "roof_or_snow",
             "candidates": ["RLL", "SNOW"], "choose": 1, "policy": "max_effect"}
        ]},
        {"primary_key": "LL",   "value": 1.0, "conditions": [
            {"kind": "alt", "group": "live_or_wind",
             "candidates": ["LL", "WIND"], "choose": 1, "policy": "max_effect"},
            {"kind": "reduction", "target": "LL", "allowed": True,
             "factor": 0.5, "ref": "live_reduction",
             "condition_ko": "base LL <= 5.0 kN/m2"}
        ]},
        {"primary_key": "WIND", "value": 0.5, "conditions": [
            {"kind": "alt", "group": "live_or_wind",
             "candidates": ["LL", "WIND"], "choose": 1, "policy": "max_effect"}
        ]},
    ]

    load_cases = {"DL": 5.0, "LL": 2.5, "RLL": 1.0, "SNOW": 0.0, "WIND": 0.0}

    result = interp.resolve_combo(combo_rows, load_cases, combo_id="ULS3")
    print(result.applied_factors)  # {'DL': 1.2, 'RLL': 1.6, 'SNOW': 0.0, 'LL': 0.5, 'WIND': 0.0}
    print(result.total_effect)     # 8.85
    print(result.summary)          # structured resolution log
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ComboResult:
    """Result of resolving a single load combination."""

    combo_id: str | None
    applied_factors: dict[str, float]
    total_effect: float
    summary: dict[str, Any]
    selected_rows: list[dict]


# ======================================================================
# Main interpreter
# ======================================================================

class ContractInterpreter:
    """Deterministic interpreter for KDS load combination contract.

    Loads ``load_contract.yaml``, then resolves combo rows against load
    cases following LOAD_DB_CONTRACT section 4 rules:
    ``alt``, ``reduction``, ``derived``, ``includes``.
    """

    def __init__(self, contract_path: str | Path | None = None):
        self.contract: dict = {}
        self.direction_rules: dict = {}
        if contract_path is not None:
            self.load_contract(contract_path)

    # ------------------------------------------------------------------
    # Contract loading
    # ------------------------------------------------------------------

    def load_contract(self, path: str | Path) -> dict:
        """Parse *load_contract.yaml* and cache direction expansion rules."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            self.contract = yaml.safe_load(fh)
        self.direction_rules = (
            self.contract.get("direction_expansion", {}).get("rules", {})
        )
        return self.contract

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_combo(
        self,
        combo_rows: list[dict],
        load_cases: dict[str, float],
        *,
        combo_id: str | None = None,
        options: dict | None = None,
    ) -> ComboResult:
        """Resolve one load combination against given load cases.

        Parameters
        ----------
        combo_rows :
            Each dict has ``primary_key`` (str), ``value`` (float),
            and optional ``conditions`` (list[dict] | dict | None).
        load_cases :
            Unfactored load-case values, e.g. ``{"DL": 5.0, "LL": 2.5}``.
        combo_id :
            Label like ``"ULS3"`` (informational).
        options :
            * ``expand_directions`` (bool, default False) -- expand WIND/EQ.
            * ``reduction_allowed`` (bool, default True) -- apply reductions.

        Returns
        -------
        ComboResult
            ``applied_factors``, ``total_effect``, ``summary``,
            ``selected_rows``.
        """
        opts = options or {}
        expand_dirs = opts.get("expand_directions", False)
        reduction_allowed_global = opts.get("reduction_allowed", True)

        summary: dict[str, Any] = {
            "alt_groups": {},
            "reductions": [],
            "derived": [],
            "includes": [],
        }

        # Step 0 -- direction expansion (MVP: duplicate base values)
        work_lc = dict(load_cases)
        if expand_dirs:
            summary["direction_expansion"] = self._expand_directions(work_lc)

        # Step 1 -- build factor map & normalise conditions
        factors: dict[str, float] = {}
        conds_map: dict[str, list[dict]] = {}
        for row in combo_rows:
            pk = row["primary_key"]
            factors[pk] = float(row["value"])
            conds_map[pk] = _normalize_conditions(row.get("conditions"))

        # Step 2 -- resolve alt groups
        alt_groups = _collect_alt_groups(conds_map)
        for group_name, ginfo in alt_groups.items():
            sel = self._resolve_alt(
                ginfo["candidates"], factors, work_lc, ginfo["policy"],
            )
            # zero-out non-selected candidates
            for cand in ginfo["candidates"]:
                if cand not in sel["selected"]:
                    factors[cand] = 0.0
            summary["alt_groups"][group_name] = sel

        # Step 3 -- apply reductions (after alt -- zeroed targets skipped)
        if reduction_allowed_global:
            for pk, conds in conds_map.items():
                for cond in conds:
                    if cond.get("kind") == "reduction":
                        rr = _apply_reduction(cond, factors)
                        summary["reductions"].append(rr)

        # Step 4 -- record derived & includes (informational only)
        for pk, conds in conds_map.items():
            for cond in conds:
                kind = cond.get("kind")
                if kind == "derived":
                    summary["derived"].append({
                        "primary_key": pk,
                        "expression": cond.get("expression"),
                        "base_factors": cond.get("base_factors"),
                        "value_used": factors.get(pk),
                    })
                # includes: no kind field, just an "includes" array
                if "includes" in cond and kind is None:
                    summary["includes"].append({
                        "primary_key": pk,
                        "includes": cond["includes"],
                        "note": cond.get("note", ""),
                        "policy": "absorbed_into_DL (Phase 1)",
                    })

        # Step 5 -- compute total effect = SUM(factor * load_value)
        selected_rows: list[dict] = []
        total = 0.0
        for pk, fac in factors.items():
            lv = work_lc.get(pk, 0.0)
            contrib = fac * lv
            total += contrib
            if fac != 0.0:
                selected_rows.append({
                    "primary_key": pk,
                    "factor": fac,
                    "load_value": lv,
                    "contribution": round(contrib, 10),
                })

        # Tidy summary -- remove empty optional sections
        for key in ("derived", "includes", "direction_expansion"):
            if key in summary and not summary.get(key):
                summary.pop(key, None)

        return ComboResult(
            combo_id=combo_id,
            applied_factors=dict(factors),
            total_effect=round(total, 10),
            summary=summary,
            selected_rows=selected_rows,
        )

    # ------------------------------------------------------------------
    # Alt resolution
    # ------------------------------------------------------------------

    def _resolve_alt(
        self,
        candidates: list[str],
        factors: dict[str, float],
        load_cases: dict[str, float],
        policy: str,
    ) -> dict:
        """Resolve one alt group via *max_effect* or *envelope*."""
        effects: dict[str, dict] = {}
        for c in candidates:
            f = factors.get(c, 0.0)
            lv = load_cases.get(c, 0.0)
            effects[c] = {"factor": f, "load": lv, "effect": f * lv}

        # Select candidate with largest effect
        best = max(candidates, key=lambda c: effects[c]["effect"])

        # Tie-break: all effects == 0 -> first candidate wins (contract rule)
        if all(e["effect"] == 0 for e in effects.values()):
            best = candidates[0]

        reason = (
            f"effect({best})={effects[best]['effect']:.6g} is max"
            if effects[best]["effect"] != 0
            else f"all effects=0, default to first candidate ({best})"
        )

        return {
            "policy": policy,
            "candidates": effects,
            "selected": [best],
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Direction expansion (MVP)
    # ------------------------------------------------------------------

    def _expand_directions(self, load_cases: dict[str, float]) -> dict:
        """Expand WIND/EQ into directional sub-keys (MVP: copy base value)."""
        info: dict[str, Any] = {"expanded_keys": {}}
        for base_key, rule in self.direction_rules.items():
            if base_key not in load_cases:
                continue
            base_val = load_cases[base_key]
            dir_keys = rule.get("expand_to", [])
            created = {}
            for dk in dir_keys:
                load_cases.setdefault(dk, base_val)
                created[dk] = load_cases[dk]
            info["expanded_keys"][base_key] = {
                "original_value": base_val,
                "directions": created,
            }
        return info


# ======================================================================
# Module-level helpers (stateless)
# ======================================================================

def _normalize_conditions(raw: Any) -> list[dict]:
    """Ensure conditions is always a list[dict]."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return list(raw)
    return []


def _collect_alt_groups(conds_map: dict[str, list[dict]]) -> dict:
    """Gather unique alt groups from all conditions."""
    groups: dict[str, dict] = {}
    for _pk, conds in conds_map.items():
        for c in conds:
            if c.get("kind") != "alt":
                continue
            g = c["group"]
            if g in groups:
                continue
            groups[g] = {
                "candidates": list(c["candidates"]),
                "policy": c.get("policy", "max_effect"),
                "choose": c.get("choose", 1),
            }
    return groups


def _apply_reduction(cond: dict, factors: dict[str, float]) -> dict:
    """Apply a reduction condition to *factors* (mutates in-place).

    Only applied when ``allowed=True`` **and** the target has a non-zero
    factor (i.e. was not zeroed by a prior alt resolution).
    """
    target = cond["target"]
    allowed = cond.get("allowed", False)
    new_factor = cond.get("factor", 1.0)
    original = factors.get(target, 0.0)

    result: dict[str, Any] = {
        "target": target,
        "allowed": allowed,
        "original_factor": original,
        "reduction_factor": new_factor,
        "applied": False,
        "condition_ko": cond.get("condition_ko", ""),
        "ref": cond.get("ref", ""),
    }

    if allowed and target in factors and factors[target] != 0.0:
        factors[target] = new_factor
        result["applied"] = True
        result["new_factor"] = new_factor
        result["reason"] = f"{target}: {original} -> {new_factor}"
    else:
        parts = ["not applied"]
        if not allowed:
            parts.append("(not allowed)")
        elif factors.get(target, 0.0) == 0.0:
            parts.append(f"(target {target} factor=0)")
        result["reason"] = " ".join(parts)

    return result


# ======================================================================
# Convenience
# ======================================================================

def load_example(path: str | Path) -> dict:
    """Load an example JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
