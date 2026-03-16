"""Apply MVP pipeline runner -- CLI entry point.

Executes the full end-to-end flow:

  1. Load JSON input
  2. Build load_cases via LoadCaseBuilder (DB lookup)
  3. Fetch/fallback combo rows → ContractInterpreter.resolve_combo
  4. Apply factored loads to minimal 3D OpenSees frame
  5. Print structured summary

Usage::

    python -m apply_mvp.runner --input data/apply_mvp_input.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from core.xxx` works
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from apply_mvp.load_case_builder import build_load_cases
from apply_mvp.opensees_3d_mvp import OPENSEESPY_AVAILABLE, build_and_analyze
from apply_mvp.schemas import MvpInput, PipelineResult
from core.contract_interpreter import ContractInterpreter

logger = logging.getLogger("apply_mvp")

# ======================================================================
# Combo-row sourcing helpers
# ======================================================================

_CONTRACT_YAML = _PROJECT_ROOT / "docs" / "specs" / "load_contract.yaml"
_EXAMPLE_COMBO = _PROJECT_ROOT / "docs" / "specs" / "examples" / "combo_apply_example.json"


def _get_combo_rows_supabase(
    mvp_input: MvpInput,
) -> tuple[list[dict], str]:
    """Try Supabase; return (combo_rows, source_label)."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return [], ""

    try:
        from supabase import create_client  # type: ignore[import-untyped]
        from adapters.load_combo_adapter import fetch_combo_rows, normalize_combo_rows

        client = create_client(url, key)
        raw = fetch_combo_rows(
            client,
            code_id=mvp_input.code.code_id,
            code_version=mvp_input.code.code_version,
            param_subtype=mvp_input.combo_request.param_subtype,
            combo_id=mvp_input.combo_request.combo_id,
        )
        rows, _skipped = normalize_combo_rows(raw)
        if rows:
            return rows, "supabase"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase fetch failed: %s", exc)
    return [], ""


def _get_combo_rows_fallback() -> tuple[list[dict], str]:
    """Load combo rows from the bundled example JSON."""
    if not _EXAMPLE_COMBO.exists():
        return [], ""
    with open(_EXAMPLE_COMBO, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("step_6_combo_records", [])
    return rows, "example_json"


# ======================================================================
# Main pipeline
# ======================================================================

def run_pipeline(mvp_input: MvpInput) -> PipelineResult:
    """Execute the full Apply MVP pipeline.

    Steps:
      1. JSON input → MvpInput (already done by caller)
      2. LoadCaseBuilder → load_cases_global + load_application_plan
      3. Supabase / fallback → combo_rows
      4. ContractInterpreter.resolve_combo → ComboResult
      5. Factored line loads → OpenSees 3D MVP → results
    """
    result = PipelineResult()

    # --- Summarise input ---
    result.input_summary = {
        "building": mvp_input.building.name,
        "floors": len(mvp_input.building.floors),
        "combo_id": mvp_input.combo_request.combo_id,
        "param_subtype": mvp_input.combo_request.param_subtype,
        "code": f"{mvp_input.code.code_id} ({mvp_input.code.code_version})",
    }

    # ---- Step 2: Build load cases ----
    logger.info("=" * 60)
    logger.info("STEP 2: Build load cases")
    lc = build_load_cases(mvp_input)
    result.load_cases = lc
    logger.info("  LL source: %s", lc.ll_source)
    logger.info("  load_cases_global: %s", lc.load_cases_global)
    if lc.warnings:
        for w in lc.warnings:
            logger.warning("  [WARN] %s", w)

    # ---- Step 3: Fetch combo rows ----
    logger.info("=" * 60)
    logger.info("STEP 3: Fetch combo rows")

    combo_rows, combo_source = _get_combo_rows_supabase(mvp_input)
    if not combo_rows:
        combo_rows, combo_source = _get_combo_rows_fallback()
    if not combo_rows:
        msg = "No combo rows available (Supabase not configured, example JSON missing)"
        result.errors.append(msg)
        logger.error(msg)
        return result

    result.combo_source = combo_source
    logger.info("  Combo source: %s (%d rows)", combo_source, len(combo_rows))

    # ---- Step 4: Resolve combo ----
    logger.info("=" * 60)
    logger.info("STEP 4: Resolve combo via ContractInterpreter")

    interp = ContractInterpreter()
    if _CONTRACT_YAML.exists():
        interp.load_contract(_CONTRACT_YAML)

    options = {}
    if mvp_input.loads_user.use_direction_expansion:
        options["expand_directions"] = True

    combo_result = interp.resolve_combo(
        combo_rows,
        dict(lc.load_cases_global),  # copy to avoid mutation
        combo_id=mvp_input.combo_request.combo_id,
        options=options,
    )

    result.applied_factors = combo_result.applied_factors
    result.total_effect = combo_result.total_effect
    result.combo_summary = combo_result.summary

    logger.info("  applied_factors: %s", combo_result.applied_factors)
    logger.info("  total_effect: %.4f kN/m2", combo_result.total_effect)

    # Log alt & reduction decisions
    for gname, ginfo in combo_result.summary.get("alt_groups", {}).items():
        logger.info("  ALT [%s]: selected=%s -- %s", gname, ginfo.get("selected"), ginfo.get("reason"))
    for red in combo_result.summary.get("reductions", []):
        logger.info(
            "  REDUCTION [%s]: applied=%s, %s",
            red.get("target"), red.get("applied"), red.get("reason"),
        )

    # ---- Step 5: OpenSees 3D analysis ----
    logger.info("=" * 60)
    logger.info("STEP 5: OpenSees 3D MVP analysis")

    if not OPENSEESPY_AVAILABLE:
        msg = "openseespy not installed -- skipping 3D analysis"
        logger.warning(msg)
        result.opensees_result = {"skipped": True, "reason": msg}
        return result

    # Build factored beam loads per storey
    # Combined load per storey = Σ (factor_k × w_k_per_m)
    floors = mvp_input.building.floors
    storey_heights = [f.height_m for f in floors]

    # Determine bay size from first floor geometry
    # MVP assumption: square bay, edge_beam_total_length / 4 = each beam span
    first_floor = floors[0] if floors else None
    if first_floor:
        beam_span = first_floor.edge_beam_total_length_m / 4.0
    else:
        beam_span = 8.0  # fallback

    beam_loads_per_storey: dict[int, float] = {}
    for fl in floors:
        plan = lc.load_application_plan.get(fl.level, {})
        w_combined = 0.0
        for case_key, factor in combo_result.applied_factors.items():
            if factor == 0.0:
                continue
            case_plan = plan.get(case_key)
            if case_plan is None:
                continue
            w_combined += factor * case_plan["w_kN_per_m"]
        beam_loads_per_storey[fl.level] = round(w_combined, 4)
        logger.info(
            "  Floor %d: combined factored w = %.4f kN/m",
            fl.level, w_combined,
        )

    os_result = build_and_analyze(
        stories=storey_heights,
        bay_x=beam_span,
        bay_y=beam_span,
        beam_loads=beam_loads_per_storey,
    )

    result.opensees_result = {
        "success": os_result.success,
        "num_nodes": os_result.num_nodes,
        "num_elements": os_result.num_elements,
        "max_disp_z_mm": os_result.max_disp_z_mm,
        "max_disp_x_mm": os_result.max_disp_x_mm,
        "max_disp_y_mm": os_result.max_disp_y_mm,
        "max_disp_z_node": os_result.max_disp_z_node,
        "applied_load_total_kN": os_result.applied_load_total_kN,
        "reaction_total_kN": os_result.reaction_total_kN,
        "equilibrium_error_pct": os_result.equilibrium_error_pct,
        "column_axial_kN": os_result.column_axial_kN,
        "errors": os_result.errors,
    }
    if os_result.errors:
        result.errors.extend(os_result.errors)

    for line in os_result.log:
        logger.info("  [OS] %s", line)

    return result


# ======================================================================
# CLI
# ======================================================================

def _print_summary(result: PipelineResult) -> None:
    """Pretty-print pipeline result to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  APPLY MVP PIPELINE - RESULT SUMMARY")
    print(sep)

    print(f"\n[Input]")
    for k, v in result.input_summary.items():
        print(f"  {k}: {v}")

    if result.load_cases:
        print(f"\n[Load Cases Global] (kN/m2)")
        for k, v in result.load_cases.load_cases_global.items():
            print(f"  {k}: {v}")
        print(f"  LL source: {result.load_cases.ll_source}")
        if result.load_cases.warnings:
            for w in result.load_cases.warnings:
                print(f"  [!] {w}")

    print(f"\n[Combo Resolution]")
    print(f"  Source: {result.combo_source}")
    print(f"  Applied factors: {result.applied_factors}")
    print(f"  Total effect: {result.total_effect} kN/m2")

    # Alt groups
    for gname, ginfo in result.combo_summary.get("alt_groups", {}).items():
        print(f"  ALT [{gname}]: selected={ginfo.get('selected')} -- {ginfo.get('reason')}")
    for red in result.combo_summary.get("reductions", []):
        status = "APPLIED" if red.get("applied") else "not applied"
        print(f"  REDUCTION [{red.get('target')}]: {status} -- {red.get('reason')}")

    print(f"\n[OpenSees 3D Result]")
    osr = result.opensees_result
    if osr.get("skipped"):
        print(f"  SKIPPED: {osr.get('reason')}")
    else:
        print(f"  Success: {osr.get('success')}")
        print(f"  Nodes/Elements: {osr.get('num_nodes')}/{osr.get('num_elements')}")
        print(f"  Max vertical disp: {osr.get('max_disp_z_mm')} mm (node {osr.get('max_disp_z_node')})")
        print(f"  Max lateral X: {osr.get('max_disp_x_mm')} mm")
        print(f"  Max lateral Y: {osr.get('max_disp_y_mm')} mm")
        print(f"  Applied load total: {osr.get('applied_load_total_kN')} kN")
        print(f"  Reaction total:     {osr.get('reaction_total_kN')} kN")
        print(f"  Equilibrium error:  {osr.get('equilibrium_error_pct')}%")
        if osr.get("errors"):
            for e in osr["errors"]:
                print(f"  ERROR: {e}")

    if result.errors:
        print(f"\n[Errors]")
        for e in result.errors:
            print(f"  - {e}")

    print(f"\n{sep}\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply MVP Pipeline -- end-to-end load->combo->OpenSees verification"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to MVP input JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    with open(input_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    mvp_input = MvpInput.from_dict(raw)

    # Run
    result = run_pipeline(mvp_input)
    _print_summary(result)

    # Also dump full result as JSON for debugging
    out_path = input_path.with_name("apply_mvp_result.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "input_summary": result.input_summary,
                "load_cases_global": result.load_cases.load_cases_global if result.load_cases else {},
                "ll_source": result.load_cases.ll_source if result.load_cases else "",
                "combo_source": result.combo_source,
                "applied_factors": result.applied_factors,
                "total_effect": result.total_effect,
                "combo_summary": result.combo_summary,
                "opensees_result": result.opensees_result,
                "errors": result.errors,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Full result written to %s", out_path)

    return 0 if not result.errors else 1


if __name__ == "__main__":
    sys.exit(main())
