"""Bake the landing page's downloadable deliverables.

``bake_demo_bundle.py`` produces the *numbers* the landing renders. This script
produces the *files* it hands out: the Korean structural calculation report, the
Excel result workbook and the DXF drawing set.

Those three are generated inside the FastAPI endpoints rather than by importable
functions, so this drives a running server instead of re-implementing them —
which also means the artifact a visitor downloads came out of exactly the code
path a visitor's own run would use.

The model is ``bake_demo_bundle.BASE_CONFIG``: the editor's default 3-story
preset, the same one the landing prints results for and the same one
``/editor-figma?demo=bench`` opens. One source of truth for the demo model.

    # terminal 1
    cd webapp/backend && python -m uvicorn app.main_simple:app --port 8099
    # terminal 2
    python scripts/bake_demo_artifacts.py --server http://127.0.0.1:8099

Writes into ``landing/files/`` and prints a manifest the page can quote.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover - dependency is in requirements
    print("[artifacts] requests is required: pip install requests", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_demo_bundle import BASE_CONFIG  # noqa: E402

#: Analysis is multi-second and queues behind any other run on the solver thread.
ANALYZE_TIMEOUT = 300


def _die(msg: str) -> None:
    print(f"[artifacts] {msg}", file=sys.stderr)
    raise SystemExit(1)


def analyze(server: str) -> dict:
    """Run the bench model through the same endpoint the editor's Run uses."""
    print(f"[artifacts] POST {server}/api/v2/analyze  (bench preset) ...", flush=True)
    r = requests.post(f"{server}/api/v2/analyze",
                      json={"config": BASE_CONFIG}, timeout=ANALYZE_TIMEOUT)
    if r.status_code != 200:
        _die(f"analyze failed: HTTP {r.status_code} {r.text[:400]}")
    data = r.json()
    if data.get("status") != "success":
        _die(f"analyze returned status={data.get('status')}: {str(data)[:400]}")

    dc = data.get("design_check") or {}
    summary = (dc.get("member_check") or {}).get("summary") or {}
    print(f"  job {data.get('job_id')}  overall={dc.get('overall_status')} "
          f"max_ratio={summary.get('max_interaction_ratio')} "
          f"NG={summary.get('ng')}/{summary.get('total')}")
    return data


def fetch(server: str, path: str, out: Path, *, method: str = "GET",
          json_body: dict | None = None, expect: str | None = None) -> int:
    r = (requests.get(f"{server}{path}", timeout=180) if method == "GET"
         else requests.post(f"{server}{path}", json=json_body, timeout=180))
    if r.status_code != 200:
        print(f"  ! {path} -> HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        return 0
    if expect and expect not in (r.headers.get("content-type") or ""):
        print(f"  ! {path} -> unexpected content-type "
              f"{r.headers.get('content-type')!r}", file=sys.stderr)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print(f"  {out.name:24s} {len(r.content):>9,} B")
    return len(r.content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8099",
                    help="a running instance of app.main_simple")
    ap.add_argument("--out", default="landing/files")
    args = ap.parse_args()

    server = args.server.rstrip("/")
    out = Path(args.out)

    data = analyze(server)
    job_id = data.get("job_id")
    if not job_id:
        _die("analyze response carried no job_id")

    sizes = {
        "report_example.html": fetch(
            server, f"/api/jobs/{job_id}/report", out / "report_example.html"),
        "tables_example.xlsx": fetch(
            server, f"/api/export/excel/{job_id}", out / "tables_example.xlsx",
            expect="spreadsheetml"),
        "drawing_example.dxf": fetch(
            server, "/api/export/dxf", out / "drawing_example.dxf",
            method="POST", json_body={"model": data.get("updated_model")}),
    }

    if not all(sizes.values()):
        _die("at least one artifact failed — landing/files/ left partially updated")

    # A tiny sidecar so the landing page can print real sizes and counts instead
    # of numbers typed into HTML that quietly go stale after the next bake.
    dc = data.get("design_check") or {}
    summary = (dc.get("member_check") or {}).get("summary") or {}
    manifest = {
        "job_id": job_id,
        "overall_status": dc.get("overall_status"),
        "max_interaction_ratio": summary.get("max_interaction_ratio"),
        "ng": summary.get("ng"),
        "total": summary.get("total"),
        "combos": len(data.get("combo_names") or []),
        "cases": len(data.get("case_names") or []),
        "files": {k: v for k, v in sizes.items()},
    }
    (out / "artifacts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[artifacts] wrote {out}/artifacts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
