"""Step 0 verification: ks3502 section table ordering monotonicity.

Goal: figure out whether row id order matches (h, b, area) order.
Output: a printable summary used to seed section_catalog.py docstring.

One-shot script — kept under scripts/ with leading underscore.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.section_3d import _get_supabase  # type: ignore


TABLES = [
    ("ks3502", "h_beam_sections", "H"),
    ("ks3502", "i_beam_sections", "I"),
    ("ks3502", "tfc_channel_sections", "TFC"),
    ("ks3502", "pfc_channel_sections", "PFC"),
    ("ks3502", "t_beam_sections", "T"),
    ("ks3502", "equal_angle_sections", "L"),
    ("ks3568", "chs_sections", "CHS"),
    ("ks3568", "shs_sections", "SHS"),
    ("ks3568", "rhs_hollow_sections", "RHS"),
]


def check_table(schema: str, table: str, label: str) -> dict:
    sb = _get_supabase()
    try:
        res = sb.schema(schema).table(table).select("*").execute()
    except Exception as e:
        return {"label": label, "error": str(e), "n": 0}

    rows = res.data or []
    if not rows:
        return {"label": label, "n": 0}

    sample_cols = sorted(rows[0].keys())

    h_col = "h" if "h" in sample_cols else ("d" if "d" in sample_cols else ("a" if "a" in sample_cols else None))
    b_col = "b" if "b" in sample_cols else h_col
    a_col = "area" if "area" in sample_cols else None
    has_id = "id" in sample_cols

    if not has_id or h_col is None or b_col is None or a_col is None:
        return {
            "label": label,
            "n": len(rows),
            "cols": sample_cols,
            "skip": "missing required columns",
        }

    # sort by id
    by_id = sorted(rows, key=lambda r: (r.get("id") or 0))

    def key_sort(r):
        return (
            (r.get(h_col) or 0),
            (r.get(b_col) or 0),
            (r.get(a_col) or 0),
        )

    by_size = sorted(rows, key=key_sort)

    # check monotonicity of (h,b,area) when iterating in id order
    inversions = 0
    prev = None
    for r in by_id:
        k = key_sort(r)
        if prev is not None and k < prev:
            inversions += 1
        prev = k

    # check if id ordering equals size ordering (by name)
    same_order_names = [r["name"] for r in by_id] == [r["name"] for r in by_size]

    # sample first 5 and last 5
    first5 = [
        {"id": r.get("id"), "name": r.get("name"), "h": r.get(h_col),
         "b": r.get(b_col), "area": r.get(a_col)}
        for r in by_id[:5]
    ]
    last5 = [
        {"id": r.get("id"), "name": r.get("name"), "h": r.get(h_col),
         "b": r.get(b_col), "area": r.get(a_col)}
        for r in by_id[-5:]
    ]

    return {
        "label": label,
        "schema": schema,
        "table": table,
        "n": len(rows),
        "h_col": h_col,
        "b_col": b_col,
        "area_col": a_col,
        "inversions_under_id_order": inversions,
        "id_order_equals_size_order": same_order_names,
        "first5_by_id": first5,
        "last5_by_id": last5,
        "cols": sample_cols,
    }


def find_square_h(rows):
    """For H beam table: how many H-WxW square sections exist?"""
    sq = []
    for r in rows:
        n = r.get("name") or ""
        # like "H-200x200x..." or "H-200x200"
        try:
            body = n.split("H-")[1]
            parts = body.split("x")
            w = int(parts[0])
            d = int(parts[1])
            if w == d:
                sq.append((n, r.get("h"), r.get("b"), r.get("area")))
        except Exception:
            pass
    return sq


def main():
    import json

    results = []
    for schema, table, label in TABLES:
        r = check_table(schema, table, label)
        results.append(r)

    print("=" * 78)
    print("Section Table Ordering Audit (Step 0)")
    print("=" * 78)
    for r in results:
        print()
        print(f"--- {r.get('label')} ({r.get('schema','?')}.{r.get('table','?')}) ---")
        if "error" in r:
            print("  ERROR:", r["error"])
            continue
        if r.get("n", 0) == 0:
            print("  empty / inaccessible")
            continue
        if "skip" in r:
            print(f"  SKIP: {r['skip']}, cols={r['cols']}")
            continue
        print(f"  rows = {r['n']}")
        print(f"  cols used: h={r['h_col']}, b={r['b_col']}, area={r['area_col']}")
        print(f"  inversions under id-order (h,b,area): {r['inversions_under_id_order']}")
        print(f"  id_order_equals_size_order: {r['id_order_equals_size_order']}")
        print("  first 5 by id:")
        for s in r["first5_by_id"]:
            print(f"    id={s['id']:>4} {s['name']:<25} h={s['h']} b={s['b']} A={s['area']}")
        print("  last 5 by id:")
        for s in r["last5_by_id"]:
            print(f"    id={s['id']:>4} {s['name']:<25} h={s['h']} b={s['b']} A={s['area']}")

    # Special: H square section catalog for column ladder
    sb = _get_supabase()
    try:
        res = sb.schema("ks3502").table("h_beam_sections").select("*").execute()
        rows = res.data or []
        sq = find_square_h(rows)
        sq_unique = {}
        for n, h, b, area in sq:
            key = (h, b)
            if key not in sq_unique or area > sq_unique[key][1]:
                sq_unique[key] = (n, area)
        print()
        print("=" * 78)
        print(f"H square (W=B) sub-families (de-duped by (h,b), {len(sq_unique)} groups)")
        print("=" * 78)
        for (h, b), (n, area) in sorted(sq_unique.items()):
            print(f"  h={h} b={b}  representative: {n}  area={area}")
    except Exception as e:
        print("\nSquare H lookup failed:", e)


if __name__ == "__main__":
    main()
