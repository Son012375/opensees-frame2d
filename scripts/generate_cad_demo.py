"""UI 실습용 CAD 데모 도면 + v2proj.json 생성.

실행:
    python scripts/generate_cad_demo.py

산출:
    outputs/cad_demo/
      ├── plan_typical.png     합성 평면도 (4×3 그리드, 12 컬럼)
      ├── building.v2proj.json V2 UI 'Load' 버튼이 받는 파일
      └── building.report.json CADExtractionReport
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "cad_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_demo_plan(
    out_path: Path,
    width: int = 1000,
    height: int = 800,
    grid_xs: list[int] = [200, 400, 600, 800],
    grid_ys: list[int] = [200, 400, 600],
    column_size_px: int = 20,
) -> None:
    """4×3 그리드 + 모든 교점에 컬럼이 있는 합성 평면도."""
    img = np.full((height, width), 255, dtype=np.uint8)
    half = column_size_px // 2
    gap = half + 4

    # 그리드 라인은 컬럼 영역을 우회
    for x in grid_xs:
        prev = 100
        for gy in grid_ys:
            cv2.line(img, (x, prev), (x, gy - gap), 0, 1)
            prev = gy + gap
        cv2.line(img, (x, prev), (x, height - 100), 0, 1)
    for y in grid_ys:
        prev = 100
        for gx in grid_xs:
            cv2.line(img, (prev, y), (gx - gap, y), 0, 1)
            prev = gx + gap
        cv2.line(img, (prev, y), (width - 100, y), 0, 1)

    # 컬럼 정사각형
    for gx in grid_xs:
        for gy in grid_ys:
            cv2.rectangle(img, (gx - half, gy - half), (gx + half, gy + half), 0, -1)

    # 라벨 (PIL TTF) — 시각 검토용. CLI는 manual labels 사용하므로 OCR 안 거침.
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except OSError:
            font = ImageFont.load_default()
        x_labels = ["A", "B", "C", "D"]
        y_labels = ["1", "2", "3"]
        for x, lab in zip(grid_xs, x_labels):
            draw.text((x - 10, 30), lab, fill=0, font=font)
            draw.text((x - 10, height - 60), lab, fill=0, font=font)
        for y, lab in zip(grid_ys, y_labels):
            draw.text((30, y - 15), lab, fill=0, font=font)
            draw.text((width - 60, y - 15), lab, fill=0, font=font)
        img = np.array(pil)
    except ImportError:
        pass

    cv2.imwrite(str(out_path), img)
    print(f"[OK] wrote {out_path} ({width}x{height})")


def make_demo_elevation(
    out_path: Path,
    width: int = 1000,
    height: int = 700,
    grid_xs: list[int] = (200, 500, 800),
    grid_labels: tuple[str, ...] = ("1", "2", "3"),
    floor_ys: list[int] = (600, 500, 400, 300, 200, 100),
    line_width: int = 2,
    column_size_px: int = 20,
) -> None:
    """입면도 — 수직 라인(평면 한 축 그리드) + 수평 floor 라인 + 보."""
    img = np.full((height, width), 255, dtype=np.uint8)
    grid_xs = list(grid_xs)
    grid_labels = list(grid_labels)

    # 수직 그리드 라인 (=평면의 1,2,3)
    for x in grid_xs:
        cv2.line(img, (x, floor_ys[-1] - 30), (x, floor_ys[0] + 30), 0, line_width)

    # 수평 floor 라인 (보로도 작용 — 그리드 인접 보 후보)
    for y in floor_ys:
        cv2.line(img, (grid_xs[0] - 40, y), (grid_xs[-1] + 40, y), 0, line_width)

    # 컬럼: 각 (grid_x, floor_y) 교점에 작은 사각형
    half = column_size_px // 2
    for gx in grid_xs:
        for gy in floor_ys:
            cv2.rectangle(img, (gx - half, gy - half), (gx + half, gy + half), 0, -1)

    # 라벨
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        for x, lab in zip(grid_xs, grid_labels):
            draw.text((x - 8, 30), lab, fill=0, font=font)
            draw.text((x - 8, height - 70), lab, fill=0, font=font)
        story_labels = ["RF", "5F", "4F", "3F", "2F", "1F"]
        for y, lab in zip(floor_ys, story_labels):
            draw.text((30, y - 12), lab, fill=0, font=font)
        img = np.array(pil)
    except ImportError:
        pass

    cv2.imwrite(str(out_path), img)
    print(f"[OK] wrote {out_path} ({width}x{height})")


def main() -> int:
    plan_path = OUT_DIR / "plan_typical.png"
    elev_a_path = OUT_DIR / "elev_A.png"
    elev_1_path = OUT_DIR / "elev_1.png"
    v2proj_path = OUT_DIR / "building.v2proj.json"

    # 1) 평면 + 입면 두 장 생성
    make_demo_plan(plan_path)
    # A열 입면: 수직 라인 3개 = 평면 Y그리드 (1,2,3)
    make_demo_elevation(
        elev_a_path, grid_xs=(200, 500, 800), grid_labels=("1", "2", "3")
    )
    # 1통 입면: 수직 라인 4개 = 평면 X그리드 (A,B,C,D)
    make_demo_elevation(
        elev_1_path, grid_xs=(150, 350, 550, 750), grid_labels=("A", "B", "C", "D")
    )

    # 2) cad_parser CLI 호출 — 평면 1장 + 입면 2장 (A열, 1통)
    mcp_server = ROOT / "mcp-server"
    cmd = [
        sys.executable, "-m", "core.cad_parser",
        "--plan", f"{plan_path}:1,2,3,4,5",   # 1F~5F 동일 typical
        "--elevation", f"{elev_a_path}:vertical_grid:A",
        "--elevation", f"{elev_1_path}:horizontal_grid:1",
        "--grid-spacing-x", "8.0",
        "--grid-spacing-y", "6.0",
        "--grid-labels-x", "A,B,C,D",
        "--grid-labels-y", "1,2,3",
        "--story-elevations", "0,4,7.5,11,14.5,18",  # base + 5 floors
        "--typical-column", "H-400x400",
        "--typical-beam-x", "H-500x200",
        "--typical-beam-y", "H-400x200",
        "--material", "SS275",
        "--region", "서울 강남",
        "--importance", "II",
        "--output", str(v2proj_path),
    ]
    env = {**__import__("os").environ, "PYTHONPATH": str(mcp_server)}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if result.returncode != 0:
        print("CLI failed:", result.stderr)
        return 1
    print(result.stdout)

    # 3) 결과 미리보기
    proj = json.loads(v2proj_path.read_text(encoding="utf-8"))
    print()
    print("=== Generated demo summary ===")
    print(f"  plan image:  {plan_path}")
    print(f"  v2proj.json: {v2proj_path}")
    print(f"  nodes:       {len(proj['model']['nodes'])}")
    print(f"  elements:    {len(proj['model']['elements'])}")
    print(f"  stories:     {len(proj['model']['story_elevations'])} (base + 5F)")
    print()
    print("UI usage:")
    print("  1) Open V2 3D Building Editor page")
    print(f"  2) Top-right [Load] button -> select {v2proj_path.name}")
    print("  3) Model appears in 3D editor -> review -> click Analyze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
