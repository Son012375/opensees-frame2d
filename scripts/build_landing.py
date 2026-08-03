"""Assemble landing/index.html from the template + the baked demo bundle.

The page ships the analysis results *inlined*, so it renders with no backend,
no fetch, and no CORS — it works from file:// and from any static host. The
frame drawing is inlined as SVG for the same reason: with JavaScript disabled
the visitor still sees the structure.

    python scripts/bake_demo_bundle.py --mcp-root <...>   # produce the numbers
    python scripts/build_landing.py                        # produce the page
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TEMPLATE_MARK_SVG = "<!--{{FRAME_SVG}}-->"
TEMPLATE_MARK_JSON = "<!--{{BUNDLE_JSON}}-->"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/demo_bundle")
    ap.add_argument("--template", default="landing/index.template.html")
    ap.add_argument("--out", default="landing/index.html")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle)
    template = Path(args.template)
    out = Path(args.out)

    manifest_path = bundle_dir / "manifest.json"
    svg_path = bundle_dir / "frame.svg"
    if not manifest_path.exists() or not svg_path.exists():
        print(f"[build] missing {manifest_path} or {svg_path} — run the bake first",
              file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    variants: dict[str, dict] = {}
    for row in manifest["ladder"]:
        path = bundle_dir / row["file"]
        if not path.exists():
            print(f"[build] missing variant file {path}", file=sys.stderr)
            return 1
        variants[row["variant"]] = json.loads(path.read_text(encoding="utf-8"))

    # Downloadable sizes, measured rather than typed into the HTML. Numbers
    # written by hand go stale the first time an artifact is re-baked, and a
    # page whose pitch is "these are the real files" cannot afford a wrong
    # file size sitting next to the download link.
    files_dir = out.parent / "files"
    file_sizes = {
        p.name: p.stat().st_size
        for p in sorted(files_dir.glob("*"))
        if p.is_file() and p.suffix != ".json"
    } if files_dir.is_dir() else {}

    payload = {"manifest": manifest, "variants": variants, "files": file_sizes}
    # "</script>" inside the JSON would close the host <script> tag early.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) \
        .replace("</", "<\\/")

    html = template.read_text(encoding="utf-8")
    for mark in (TEMPLATE_MARK_SVG, TEMPLATE_MARK_JSON):
        if mark not in html:
            print(f"[build] template is missing {mark}", file=sys.stderr)
            return 1

    html = html.replace(TEMPLATE_MARK_SVG, svg_path.read_text(encoding="utf-8"))
    html = html.replace(TEMPLATE_MARK_JSON, blob)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    kb = out.stat().st_size / 1024
    print(f"[build] {out}  {kb:.0f} KB  "
          f"({len(variants)} variants, commit {manifest['git_commit']})")
    for row in manifest["ladder"]:
        print(f"         {row['variant']:12s} {row['overall_status']:3s} "
              f"ratio={row['max_interaction_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
