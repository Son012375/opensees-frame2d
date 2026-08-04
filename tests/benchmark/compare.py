"""Comparison utility for OpenSees vs Midas Gen results.

Loads Midas Gen results from JSON, computes differences, and generates
a formatted comparison table.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


MIDAS_RESULTS_DIR = Path(__file__).parent / "midas_results"


def load_midas_results(case_id: str) -> dict | None:
    """Load manually entered Midas Gen results from JSON file.

    Expected file: tests/benchmark/midas_results/{case_id}.json
    Format: {"metric_name": value, ...}
    """
    fpath = MIDAS_RESULTS_DIR / f"{case_id}.json"
    if not fpath.exists():
        return None
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_metrics(
    extracted: list[dict],
    midas: dict | None,
    tolerance_pct: float = 1.0,
    fail_threshold_pct: float | None = 5.0,
) -> list[dict]:
    """Compare OpenSees metrics against Midas Gen results.

    Args:
        extracted: list of {"metric", "unit", "opensees"} dicts
        midas: dict of {"metric_name": value} from Midas Gen (or None)
        tolerance_pct: ≤ this percent → OK (default 1%)
        fail_threshold_pct: > this percent → FAIL. Default 5%. Pass None to
            disable FAIL (legacy 2-level OK/CHECK behavior).

    Returns:
        list of comparison rows with diff_pct and status. Status is one of
        OK / CHECK / FAIL / PENDING.
    """
    rows = []
    for item in extracted:
        metric = item["metric"]
        unit = item["unit"]
        ops_val = item["opensees"]

        row = {
            "metric": metric,
            "unit": unit,
            "opensees": ops_val,
            "midas": None,
            "diff_pct": None,
            "status": "PENDING",
        }

        if midas and metric in midas and midas[metric] is not None:
            m_val = midas[metric]
            row["midas"] = m_val

            # Compute percentage difference
            ref = max(abs(ops_val), abs(m_val))
            if ref > 1e-12:
                diff = abs(ops_val - m_val) / ref * 100.0
                row["diff_pct"] = diff
                if diff <= tolerance_pct:
                    row["status"] = "OK"
                elif fail_threshold_pct is not None and diff > fail_threshold_pct:
                    row["status"] = "FAIL"
                else:
                    row["status"] = "CHECK"
            else:
                row["diff_pct"] = 0.0
                row["status"] = "OK"

        rows.append(row)
    return rows


def format_table(case_name: str, rows: list[dict]) -> str:
    """Format comparison results as a readable ASCII table."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {case_name}")
    lines.append(f"{'='*80}")

    # Header
    header = f"{'Metric':<30} {'Unit':<8} {'OpenSees':>14} {'Midas':>14} {'Diff%':>8} {'Status':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for row in rows:
        ops_str = f"{row['opensees']:14.6f}" if isinstance(row['opensees'], (int, float)) else f"{row['opensees']!s:>14}"
        midas_str = f"{row['midas']:14.6f}" if isinstance(row['midas'], (int, float)) else f"{'-':>14}"
        diff_str = f"{row['diff_pct']:7.3f}%" if row['diff_pct'] is not None else f"{'-':>8}"
        status = row['status']

        lines.append(f"{row['metric']:<30} {row['unit']:<8} {ops_str} {midas_str} {diff_str} {status:>8}")

    # Summary
    total = len(rows)
    ok = sum(1 for r in rows if r['status'] == 'OK')
    check = sum(1 for r in rows if r['status'] == 'CHECK')
    fail = sum(1 for r in rows if r['status'] == 'FAIL')
    pending = sum(1 for r in rows if r['status'] == 'PENDING')
    lines.append("-" * 80)
    lines.append(f"  Total: {total} | OK: {ok} | CHECK: {check} | FAIL: {fail} | PENDING: {pending}")

    return "\n".join(lines)


def save_opensees_results(case_id: str, extracted: list[dict]) -> str:
    """Save OpenSees results to JSON for reference.

    Returns the file path.
    """
    out_dir = MIDAS_RESULTS_DIR.parent / "opensees_results"
    out_dir.mkdir(exist_ok=True)
    fpath = out_dir / f"{case_id}.json"

    data = {item["metric"]: item["opensees"] for item in extracted}
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(fpath)


def create_midas_template(case_id: str, extracted: list[dict]) -> str:
    """Create a blank Midas Gen results template JSON for manual entry.

    Returns the file path.
    """
    MIDAS_RESULTS_DIR.mkdir(exist_ok=True)
    fpath = MIDAS_RESULTS_DIR / f"{case_id}.json"

    if fpath.exists():
        return str(fpath)  # Don't overwrite existing results

    data = {item["metric"]: None for item in extracted}
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(fpath)
