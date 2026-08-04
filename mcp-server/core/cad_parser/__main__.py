"""CLI 진입점 — `python -m core.cad_parser ...`

사용 예:
    python -m core.cad_parser --config drawing_set.yaml --output building.v2proj.json

또는 인자 직접:
    python -m core.cad_parser \\
        --plan plan_1F.png:0 --plan plan_typical.png:1,2,3,4 \\
        --grid-spacing-x 8.0 --grid-spacing-y 6.0 \\
        --grid-labels-x A,B,C,D --grid-labels-y 1,2,3 \\
        --story-elevations 0,4,7.5,11,14.5,18 \\
        --typical-column H-400x400 --typical-beam-x H-500x200 --typical-beam-y H-400x200 \\
        --output building.v2proj.json

YAML config 형식:
```yaml
plans:
  - file: plan_1F.png
    stories: [0]
  - file: plan_typical.png
    stories: [1, 2, 3, 4]
grid:
  spacing_x_m: 8.0
  spacing_y_m: 6.0
  labels_x: [A, B, C, D]
  labels_y: [1, 2, 3]
story_elevations_m: [0, 4, 7.5, 11, 14.5, 18]
typical_sections:
  column: H-400x400
  beam_x: H-500x200
  beam_y: H-400x200
  material: SS275
environment:
  region: "서울 강남"
  importance: II
output: building.v2proj.json
```
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m core.cad_parser")
    p.add_argument("--config", type=Path, help="YAML config 파일 경로")
    p.add_argument(
        "--plan", action="append", default=[],
        help="평면도 시트: `path:story_list` (예: `plan_typical.png:1,2,3`). 반복 가능.",
    )
    p.add_argument(
        "--elevation", action="append", default=[],
        help="입면도 시트: `path:orth_axis:transverse_label` "
             "(예: `elev_A.png:vertical_grid:A`). orth_axis ∈ "
             "{vertical_grid,horizontal_grid}. 반복 가능.",
    )
    p.add_argument("--grid-spacing-x", type=float, help="vertical 그리드(A,B,C) 간격 (m)")
    p.add_argument("--grid-spacing-y", type=float, help="horizontal 그리드(1,2,3) 간격 (m)")
    p.add_argument("--grid-labels-x", type=str, help="콤마 구분 (예: 'A,B,C,D'). 좌표 오름차순.")
    p.add_argument("--grid-labels-y", type=str, help="콤마 구분 (예: '1,2,3').")
    p.add_argument("--story-elevations", type=str, help="콤마 구분 (m). 첫 값은 base=0 권장.")
    p.add_argument("--typical-column", type=str, default="H-400x400")
    p.add_argument("--typical-beam-x", type=str, default="H-500x200")
    p.add_argument("--typical-beam-y", type=str, default="H-400x200")
    p.add_argument("--material", type=str, default="SS275")
    p.add_argument("--region", type=str, default="")
    p.add_argument("--importance", type=str, default="II")
    p.add_argument("--output", type=Path, default=Path("cad_output.v2proj.json"))
    p.add_argument("--report-output", type=Path, default=None,
                   help="CADExtractionReport JSON 경로 (default: <output>.report.json)")
    return p.parse_args()


def _load_yaml_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "YAML config를 쓰려면 PyYAML이 필요합니다: pip install pyyaml"
        ) from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _config_from_cli(args: argparse.Namespace) -> dict:
    """CLI 인자 → config dict (YAML과 동일 스키마)."""
    if not args.plan:
        raise ValueError("--plan 옵션이 최소 1개 필요합니다 (또는 --config 사용)")
    plans = []
    for p in args.plan:
        # Windows 드라이브 콜론(`C:\…`) 회피 위해 마지막 `:` 기준 분할
        if ":" not in p:
            raise ValueError(f"--plan 형식 'path:story_list', got {p!r}")
        path_part, story_part = p.rsplit(":", 1)
        try:
            stories = [int(s.strip()) for s in story_part.split(",") if s.strip()]
        except ValueError as e:
            raise ValueError(
                f"--plan 의 story 부분이 정수 콤마 리스트가 아닙니다: {story_part!r} "
                f"(예: 'plan.png:1,2,3')"
            ) from e
        plans.append({"file": path_part, "stories": stories})

    if not args.grid_spacing_x or not args.grid_spacing_y:
        raise ValueError("--grid-spacing-x 와 --grid-spacing-y 모두 필요합니다.")
    if not args.grid_labels_x or not args.grid_labels_y:
        raise ValueError("--grid-labels-x 와 --grid-labels-y 모두 필요합니다.")
    if not args.story_elevations:
        raise ValueError("--story-elevations 필요합니다.")

    elevations = []
    for e in args.elevation:
        parts = e.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"--elevation 형식 'path:orth_axis:transverse_label', got {e!r}"
            )
        path_part, orth, trans = parts
        if orth not in ("vertical_grid", "horizontal_grid"):
            raise ValueError(f"orth_axis must be vertical_grid|horizontal_grid, got {orth!r}")
        elevations.append({
            "file": path_part,
            "orth_axis": orth,
            "transverse_label": trans.strip(),
        })

    return {
        "plans": plans,
        "elevations": elevations,
        "grid": {
            "spacing_x_m": args.grid_spacing_x,
            "spacing_y_m": args.grid_spacing_y,
            "labels_x": [s.strip() for s in args.grid_labels_x.split(",")],
            "labels_y": [s.strip() for s in args.grid_labels_y.split(",")],
        },
        "story_elevations_m": [
            float(s.strip()) for s in args.story_elevations.split(",")
        ],
        "typical_sections": {
            "column": args.typical_column,
            "beam_x": args.typical_beam_x,
            "beam_y": args.typical_beam_y,
            "material": args.material,
        },
        "environment": {"region": args.region, "importance": args.importance},
    }


def _run_pipeline(config: dict, output: Path, report_output: Optional[Path]) -> None:
    """파이프라인 실행: 평면 로딩 → 그리드 검출 → 정합 → 컬럼 추출 → builder → JSON 저장."""
    # 지연 import (CLI 도움말 출력 시 무거운 의존성 안 끌어들임)
    from . import (
        builder, grid_detector, member_extract, preprocess, registration, vectorize,
    )
    from .schemas import CADExtractionReport, TypicalSectionSpec

    grid_cfg = config["grid"]
    typ = config["typical_sections"]
    typical = TypicalSectionSpec(
        column=typ["column"],
        beam_x=typ["beam_x"],
        beam_y=typ["beam_y"],
        material=typ.get("material", "SS275"),
    )

    # 각 평면 처리
    plan_grids: dict[str, "schemas.GridSet"] = {}
    polygons_per_plan: dict[str, list] = {}
    sheet_id_to_stories: dict[str, list[int]] = {}
    report = CADExtractionReport()

    for plan_cfg in config["plans"]:
        path = Path(plan_cfg["file"])
        sheet_id = plan_cfg.get("sheet_id", path.stem)
        sheet = preprocess.load_sheet(path, sheet_id=sheet_id, kind="plan")
        binary = preprocess.binarize(sheet)

        grid = grid_detector.detect_grid(binary, min_line_length=200)
        # manual 라벨 부여 (좌표 오름차순)
        grid = grid_detector.assign_labels_manual(
            grid,
            vertical_labels=grid_cfg["labels_x"],
            horizontal_labels=grid_cfg["labels_y"],
        )
        plan_grids[sheet_id] = grid

        polygons = vectorize.extract_polygons(binary, min_area=80.0, max_area=5000.0)
        polygons_per_plan[sheet_id] = polygons

        sheet_id_to_stories[sheet_id] = list(plan_cfg.get("stories", []))
        report.sheets_processed.append(sheet_id)

    # 정합
    registered = registration.register_plans(
        plan_grids=plan_grids,
        grid_spacing_x_m=grid_cfg["spacing_x_m"],
        grid_spacing_y_m=grid_cfg["spacing_y_m"],
        story_elevations_m=config["story_elevations_m"],
    )
    report.registration_rmse_px = max(registered.rmse_px.values()) if registered.rmse_px else 0.0

    # 컬럼 추출
    columns = member_extract.extract_column_candidates(
        polygons_per_plan=polygons_per_plan,
        plan_grid_per_plan=plan_grids,
        sheet_id_to_stories=sheet_id_to_stories,
        max_dist_px=30.0,
        min_area_px=80.0,
        max_area_px=5000.0,
    )
    report.detected_columns = len(columns)

    # 입면도 처리 (옵셔널) → 보 추출
    n_stories = len(config["story_elevations_m"]) - 1   # base 제외 층 수
    beam_candidates: list = []
    for elev_cfg in config.get("elevations", []) or []:
        path = Path(elev_cfg["file"])
        sheet_id = elev_cfg.get("sheet_id", path.stem)
        sheet = preprocess.load_sheet(path, sheet_id=sheet_id, kind="elevation")
        binary = preprocess.binarize(sheet)

        elev_grid = grid_detector.detect_grid(binary, min_line_length=200)
        # 입면 vertical_lines 라벨 부여 — orth_axis에 따라 평면의 다른 축 라벨 사용
        orth = elev_cfg["orth_axis"]
        if orth == "vertical_grid":
            v_labels = grid_cfg["labels_y"]
        else:
            v_labels = grid_cfg["labels_x"]
        # horizontal_lines = 층 라인. 좌표 오름차순(위→아래) → reverse=True로 base→top
        n_h = len(elev_grid.horizontal_lines)
        if n_h == 0:
            report.warnings.append(f"{sheet_id}: 입면 horizontal grid 검출 실패")
            continue
        sorted_h_idx = sorted(
            range(n_h),
            key=lambda i: elev_grid.horizontal_lines[i].coord_px,
            reverse=True,   # 아래(큰 y) = base
        )
        elev_story_labels = [None] * n_h
        for story_idx, h_idx in enumerate(sorted_h_idx):
            elev_story_labels[h_idx] = story_idx   # 0=base, 1=1F, ...
        # vertical_lines 라벨도 좌표 오름차순으로 manual 부여
        if len(v_labels) != len(elev_grid.vertical_lines):
            report.warnings.append(
                f"{sheet_id}: vertical lines 수({len(elev_grid.vertical_lines)})와 "
                f"labels({len(v_labels)}) mismatch - skipped"
            )
            continue
        sorted_v_idx = sorted(
            range(len(elev_grid.vertical_lines)),
            key=lambda i: elev_grid.vertical_lines[i].coord_px,
        )
        for v_idx, lab_idx in enumerate(sorted_v_idx):
            elev_grid.vertical_lines[lab_idx].label = v_labels[v_idx]

        # affine — 추후 픽셀→world 변환 필요 시 사용
        try:
            M3_elev = registration.link_elevation_to_plan(
                elev_grid, orth, registered, story_labels=elev_story_labels,
            )
            registered.elevation_affines[sheet_id] = M3_elev
        except ValueError as e:
            report.warnings.append(f"{sheet_id}: elevation affine failed - {e}")

        # 보 추출: 입면 long horizontals
        segments = vectorize.extract_line_segments(
            binary, min_length=120, threshold=80
        )
        horiz, _vert, _diag = vectorize.split_by_orientation(segments, tolerance_deg=3.0)
        beams = member_extract.extract_beam_candidates(
            elevation_horiz_segments=horiz,
            elevation_grid=elev_grid,
            elevation_orth_axis=orth,
            transverse_label=elev_cfg["transverse_label"],
            story_labels=elev_story_labels,
            floor_tolerance_px=30.0,
            min_span_ratio=0.5,
        )
        beam_candidates.extend(beams)
        report.sheets_processed.append(sheet_id)

    report.detected_beams = len(beam_candidates)

    # 빌더
    model_dict = builder.build_structural_model_dict(
        registered=registered,
        column_candidates=columns,
        beam_candidates=beam_candidates,
        typical_sections=typical,
        environment=config.get("environment"),
    )
    v2proj = builder.wrap_v2proj(model_dict)

    # 저장
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(v2proj, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = report_output or output.with_suffix(".report.json")
    report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {output}")
    print(f"  nodes={len(model_dict['nodes'])}  elements={len(model_dict['elements'])}")
    print(f"  columns={report.detected_columns}  beams={report.detected_beams}")
    print(f"  registration RMSE = {report.registration_rmse_px:.4f} m")
    if report.warnings:
        print(f"  warnings: {len(report.warnings)}")
        for w in report.warnings[:5]:
            print(f"    - {w}")
    print(f"  report: {report_path}")


def main() -> int:
    args = _parse_args()
    try:
        if args.config:
            config = _load_yaml_config(args.config)
            output = Path(config.get("output", args.output))
        else:
            config = _config_from_cli(args)
            output = args.output
        _run_pipeline(config, output, args.report_output)
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
