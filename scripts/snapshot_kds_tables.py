"""Dump the KDS reference tables to a local snapshot.

The snapshot is what `core.kds_cache` serves when Supabase is unreachable, so
run this while the database is up and commit the result. Values come from the
database itself — offline answers therefore match online answers.

    python scripts/snapshot_kds_tables.py
    python scripts/snapshot_kds_tables.py --verify   # compare, don't write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "mcp-server"))

from core.kds_cache import CACHE_DIR, MIRRORED_TABLES, cache_path  # noqa: E402

PAGE = 1000


def fetch_all(client, schema: str, table: str) -> list[dict]:
    """Page through a table — PostgREST caps a single response at 1000 rows."""
    rows: list[dict] = []
    start = 0
    while True:
        q = client.schema(schema).table(table) if schema != "public" \
            else client.table(table)
        resp = q.select("*").range(start, start + PAGE - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        start += PAGE
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="compare live tables against the snapshot without writing")
    args = ap.parse_args()

    from core.section_3d import _get_supabase  # reuses the configured credentials
    client = _get_supabase()
    # Unwrap: snapshotting must talk to the real database, never to the cache.
    real = getattr(client, "_real", client)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    drift: list[str] = []

    for schema, table in MIRRORED_TABLES:
        try:
            rows = fetch_all(real, schema, table)
        except Exception as exc:
            print(f"  {schema}.{table:38s} SKIP ({type(exc).__name__}: {exc})")
            continue

        path = cache_path(schema, table)
        if args.verify:
            if not path.exists():
                drift.append(f"{schema}.{table}: no snapshot")
                print(f"  {schema}.{table:38s} MISSING  (live {len(rows)})")
                continue
            old = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
            same = old == rows
            if not same:
                drift.append(f"{schema}.{table}: {len(old)} -> {len(rows)}")
            print(f"  {schema}.{table:38s} {'OK  ' if same else 'DRIFT'} "
                  f"(snapshot {len(old)}, live {len(rows)})")
        else:
            payload = {
                "schema": schema,
                "table": table,
                "captured_at": datetime.now(timezone.utc).astimezone().isoformat(
                    timespec="seconds"),
                "row_count": len(rows),
                "rows": rows,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False,
                                       separators=(",", ":")), encoding="utf-8")
            print(f"  {schema}.{table:38s} {len(rows):5d} rows  "
                  f"{path.stat().st_size / 1024:7.1f} KB")
        total += len(rows)

    if args.verify:
        print(f"\n{'DRIFT DETECTED' if drift else 'snapshot matches live database'}")
        for d in drift:
            print("  " + d)
        return 1 if drift else 0

    size = sum(p.stat().st_size for p in CACHE_DIR.glob("*.json")) / 1024
    print(f"\n{total} rows across {len(MIRRORED_TABLES)} tables -> {CACHE_DIR} "
          f"({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
