"""IFC 파일 업로드 처리.

Streamlit file_uploader → tempfile → parse_ifc() → cleanup.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

MCP_SERVER_PATH = str(Path(__file__).resolve().parent.parent.parent / "mcp-server")


def _ensure_path():
    if MCP_SERVER_PATH not in sys.path:
        sys.path.insert(0, MCP_SERVER_PATH)


def parse_uploaded_ifc(uploaded_file) -> dict:
    """Streamlit UploadedFile → parse_ifc() 결과.

    Args:
        uploaded_file: st.file_uploader 반환값

    Returns:
        parse_ifc() 결과 dict (stories, bays_x, bays_y, warnings 등)
    """
    _ensure_path()
    from core.ifc_parser import parse_ifc

    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        result = parse_ifc(tmp_path)

        # 디버그: IFC 복사본 저장 (단면 추출 진단용)
        debug_dir = Path(__file__).resolve().parent.parent.parent / "data"
        debug_dir.mkdir(exist_ok=True)
        debug_path = debug_dir / "debug_uploaded.ifc"
        import shutil
        shutil.copy2(tmp_path, str(debug_path))
    finally:
        os.unlink(tmp_path)

    return result


def ifc_data_to_geometry_config(ifc_data: dict) -> dict:
    """parse_ifc() 출력 → geometry config dict.

    Returns:
        {"stories": [{"height": float, "slab_thickness": float}, ...],
         "bays_x": [...], "bays_y": [...]}
    """
    stories = []
    for s in ifc_data.get("stories", []):
        story = {"height": s["height"]}
        slab_t = s.get("slab_thickness_mm")
        if slab_t is not None:
            story["slab_thickness"] = round(slab_t / 1000.0, 4)
        else:
            story["slab_thickness"] = 0.15
        stories.append(story)

    result = {
        "stories": stories,
        "bays_x": ifc_data.get("bays_x", []),
        "bays_y": ifc_data.get("bays_y", []),
    }

    # IFC에서 감지된 단면/재료 포함
    detected = ifc_data.get("detected_sections", {})
    if detected.get("column"):
        result["column_section"] = detected["column"]
    if detected.get("beam"):
        result["beam_section"] = detected["beam"]
    if ifc_data.get("detected_material"):
        result["material_name"] = ifc_data["detected_material"]
    if ifc_data.get("section_substitutions"):
        result["section_substitutions"] = ifc_data["section_substitutions"]

    return result


def format_geometry_summary(geo: dict) -> str:
    """기하 config → 사람이 읽을 수 있는 마크다운 요약."""
    stories = geo.get("stories", [])
    bays_x = geo.get("bays_x", [])
    bays_y = geo.get("bays_y", [])

    total_h = sum(s["height"] for s in stories)
    total_x = sum(bays_x)
    total_y = sum(bays_y)

    lines = [
        f"**Stories:** {len(stories)}F, Total Height: {total_h:.1f}m",
    ]
    for i, s in enumerate(stories, 1):
        slab_t = s.get("slab_thickness", 0.15)
        lines.append(f"- {i}F: height={s['height']:.1f}m, slab={slab_t*1000:.0f}mm")

    lines.append(f"**Bays X:** {len(bays_x)} bays {bays_x}")
    lines.append(f"**Bays Y:** {len(bays_y)} bays {bays_y}")
    lines.append(f"**Plan:** {total_x:.1f} x {total_y:.1f}m = {total_x*total_y:.0f}m²")

    # 감지된 단면/재료 표시
    col_sec = geo.get("column_section")
    beam_sec = geo.get("beam_section")
    mat = geo.get("material_name")
    if col_sec or beam_sec or mat:
        lines.append("")
        lines.append("**Detected from IFC:**")
        if col_sec:
            lines.append(f"- Column: {col_sec}")
        if beam_sec:
            lines.append(f"- Beam: {beam_sec}")
        if mat:
            lines.append(f"- Material: {mat}")

    # 대체 경고 표시
    substitutions = geo.get("section_substitutions", [])
    for sub in substitutions:
        lines.append(
            f"- ⚠ {sub['original']} → {sub['substituted']}로 대체 "
            f"({sub.get('reason', '')})"
        )

    return "\n".join(lines)
