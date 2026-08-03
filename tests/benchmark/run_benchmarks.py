"""Run all benchmark cases and generate comparison tables.

Usage:
    python tests/benchmark/run_benchmarks.py              # run all cases
    python tests/benchmark/run_benchmarks.py case1 case3   # run specific cases
    python tests/benchmark/run_benchmarks.py --template    # create Midas templates
"""
from __future__ import annotations

import sys
import os
import json
import time

# Ensure mcp-server is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mcp-server'))

from cases import ALL_CASES, OPTIONAL_CASES
from extract import EXTRACTORS
from compare import (
    load_midas_results,
    compare_metrics,
    format_table,
    save_opensees_results,
    create_midas_template,
)


def _get_case(case_id: str) -> tuple[str, callable]:
    """Look up case from ALL_CASES or OPTIONAL_CASES."""
    if case_id in ALL_CASES:
        return ALL_CASES[case_id]
    if case_id in OPTIONAL_CASES:
        return OPTIONAL_CASES[case_id]
    raise KeyError(f"Unknown case: {case_id}")


def run_single(case_id: str) -> tuple[str, list[dict]]:
    """Run a single benchmark case and return formatted output."""
    name, runner = _get_case(case_id)
    extractor = EXTRACTORS[case_id]

    print(f"  Running {case_id}: {name} ...", end=" ", flush=True)
    t0 = time.time()
    results = runner()
    dt = time.time() - t0
    print(f"done ({dt:.2f}s)")

    extracted = extractor(results)
    midas = load_midas_results(case_id)
    compared = compare_metrics(extracted, midas)

    save_opensees_results(case_id, extracted)

    table = format_table(f"{case_id.upper()}: {name}", compared)
    return table, compared


def main():
    args = sys.argv[1:]

    # Template mode
    if "--template" in args:
        all_known = {**ALL_CASES, **OPTIONAL_CASES}
        requested = [a for a in args if a in all_known]
        if not requested:
            requested = list(ALL_CASES.keys())  # legacy default
        print("\nCreating Midas Gen result templates...")
        for case_id in requested:
            name, runner = all_known[case_id]
            extractor = EXTRACTORS[case_id]
            # Run to get metric names
            results = runner()
            extracted = extractor(results)
            fpath = create_midas_template(case_id, extracted)
            print(f"  {case_id}: {fpath}")
        print("\nDone. Fill in Midas Gen values and re-run.")
        return

    # Select cases (optional cases only run when explicitly requested)
    all_known = {**ALL_CASES, **OPTIONAL_CASES}
    case_ids = [a for a in args if a in all_known]
    if not case_ids:
        case_ids = list(ALL_CASES.keys())

    print(f"\n{'='*80}")
    print(f"  OpenSees Benchmark Suite - {len(case_ids)} cases")
    print(f"{'='*80}\n")

    all_tables = []
    total_ok = 0
    total_check = 0
    total_fail = 0
    total_pending = 0
    total_metrics = 0

    for case_id in case_ids:
        table, compared = run_single(case_id)
        all_tables.append(table)

        for r in compared:
            total_metrics += 1
            if r["status"] == "OK":
                total_ok += 1
            elif r["status"] == "CHECK":
                total_check += 1
            elif r["status"] == "FAIL":
                total_fail += 1
            else:
                total_pending += 1

    # Print all tables
    for t in all_tables:
        print(t)

    # Grand summary
    print(f"\n{'='*80}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*80}")
    print(f"  Cases run: {len(case_ids)}")
    print(f"  Total metrics: {total_metrics}")
    print(f"  OK: {total_ok}  |  CHECK: {total_check}  |  FAIL: {total_fail}  |  PENDING: {total_pending}")

    if total_pending == total_metrics:
        print(f"\n  All metrics PENDING - Midas Gen results not yet entered.")
        print(f"  Run with --template to create JSON templates.")
    elif total_fail > 0:
        print(f"\n  {total_fail} metrics exceed FAIL threshold (>5%) - investigate.")
    elif total_check > 0:
        print(f"\n  {total_check} metrics in CHECK band (1-5%) - review needed.")
    else:
        print(f"\n  All compared metrics within tolerance.")

    print(f"{'='*80}\n")

    # Save summary JSON
    out_dir = os.path.join(os.path.dirname(__file__), "opensees_results")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "cases_run": len(case_ids),
        "total_metrics": total_metrics,
        "ok": total_ok,
        "check": total_check,
        "fail": total_fail,
        "pending": total_pending,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
