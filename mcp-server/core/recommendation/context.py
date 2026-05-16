"""Resolve member/location context from analysis_metadata + design_check row.

The recommendation pipeline gets a heterogeneous ``analysis_metadata`` dict
from the API layer. This module flattens it into:

    * a ``MemberContextIndex`` keyed by ``member_id``
    * helpers that return ``StructuralIssue``-ready context kwargs

Everything is defensive — missing fields just yield ``None``. We never
raise from here because the analysis itself already succeeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MemberContext:
    member_type: Optional[str] = None
    section: Optional[str] = None
    material: Optional[str] = None
    story: Optional[int] = None
    level: Optional[float] = None
    connected_node_ids: list[int] = field(default_factory=list)
    grid_location: Optional[str] = None
    coordinates: Optional[dict[str, Any]] = None

    def merge(self, other: "MemberContext") -> None:
        """Fill ``self`` from ``other`` for any field still unset."""
        for k in ("member_type", "section", "material",
                  "story", "level", "grid_location", "coordinates"):
            if getattr(self, k) is None and getattr(other, k) is not None:
                setattr(self, k, getattr(other, k))
        if not self.connected_node_ids and other.connected_node_ids:
            self.connected_node_ids = list(other.connected_node_ids)


@dataclass
class MemberContextIndex:
    """Per-member context, plus the global material name as fallback."""
    by_member: dict[int, MemberContext] = field(default_factory=dict)
    default_material: Optional[str] = None

    def get(self, member_id: Optional[int]) -> MemberContext:
        if member_id is None:
            return MemberContext()
        ctx = self.by_member.get(int(member_id))
        if ctx is None:
            ctx = MemberContext()
            self.by_member[int(member_id)] = ctx
        if ctx.material is None and self.default_material:
            ctx.material = self.default_material
        return ctx


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _ctx_from_member_info_row(row: dict[str, Any]) -> tuple[Optional[int], MemberContext]:
    mid = _coerce_int(row.get("member_id"))
    if mid is None:
        return None, MemberContext()

    # Node ids vary by source: "ni"/"nj" (frame_3d) or "node_i"/"node_j"
    nodes: list[int] = []
    for key in ("ni", "node_i", "i", "i_node"):
        v = _coerce_int(row.get(key))
        if v is not None:
            nodes.append(v)
            break
    for key in ("nj", "node_j", "j", "j_node"):
        v = _coerce_int(row.get(key))
        if v is not None:
            nodes.append(v)
            break

    coords = None
    # Some upstream rows include endpoint coords; preserve them when present.
    if "coords" in row and isinstance(row["coords"], dict):
        coords = dict(row["coords"])
    elif "x_m" in row and "y_m" in row:
        coords = {"x": row.get("x_m"), "y": row.get("y_m"), "z": row.get("z_m")}

    return mid, MemberContext(
        member_type=row.get("type") or row.get("elem_type") or row.get("member_type"),
        section=row.get("section"),
        material=row.get("material"),
        story=_coerce_int(row.get("story")),
        level=_coerce_float(row.get("level") or row.get("z")),
        connected_node_ids=nodes,
        grid_location=row.get("grid") or row.get("grid_location"),
        coordinates=coords,
    )


def _ctx_from_updated_model_element(elem: dict[str, Any]) -> tuple[Optional[int], MemberContext]:
    """Element entry from ``updated_model.elements`` / V2 StructuralModel."""
    mid = _coerce_int(elem.get("id") or elem.get("member_id") or elem.get("element_id"))
    if mid is None:
        return None, MemberContext()
    nodes: list[int] = []
    for key in ("ni", "nj", "node_i", "node_j", "i", "j"):
        v = _coerce_int(elem.get(key))
        if v is not None and v not in nodes:
            nodes.append(v)
    return mid, MemberContext(
        member_type=elem.get("elem_type") or elem.get("type"),
        section=elem.get("section"),
        material=elem.get("material"),
        connected_node_ids=nodes,
    )


def build_context_index(metadata: Optional[dict[str, Any]]) -> MemberContextIndex:
    """Walk ``analysis_metadata`` and produce a per-member context index.

    Tolerant of the various shapes V2 emits:
        metadata = {
            "member_info": [ {member_id, type, section, ni, nj, …}, ... ],
            "updated_model": { "elements": {...}, "nodes": {...} },
            "building": {...},                       # ignored for member ctx
            "material_name": "SS275",
            "stories_heights": [4.0, 3.5, ...],      # for level/story
            ...
        }
    """
    idx = MemberContextIndex()
    if not isinstance(metadata, dict):
        return idx

    idx.default_material = (
        metadata.get("material_name")
        or metadata.get("default_material")
        or None
    )

    # 1) member_info — typically the richest source (sections, types, nodes).
    member_info = metadata.get("member_info")
    if isinstance(member_info, list):
        for row in member_info:
            if not isinstance(row, dict):
                continue
            mid, ctx = _ctx_from_member_info_row(row)
            if mid is None:
                continue
            existing = idx.by_member.get(mid, MemberContext())
            existing.merge(ctx)
            idx.by_member[mid] = existing

    # 2) updated_model.elements — V2 StructuralModel JSON.
    um = metadata.get("updated_model")
    if isinstance(um, dict):
        elements = um.get("elements")
        if isinstance(elements, dict):
            iterable = elements.values()
        elif isinstance(elements, list):
            iterable = elements
        else:
            iterable = []
        for elem in iterable:
            if not isinstance(elem, dict):
                continue
            mid, ctx = _ctx_from_updated_model_element(elem)
            if mid is None:
                continue
            existing = idx.by_member.get(mid, MemberContext())
            existing.merge(ctx)
            idx.by_member[mid] = existing

        # Node coordinates — used to fill `coordinates` for endpoints.
        nodes = um.get("nodes")
        if isinstance(nodes, dict):
            node_xyz: dict[int, dict[str, Any]] = {}
            for k, n in nodes.items():
                nid = _coerce_int(k if isinstance(k, (int, str)) else n.get("id"))
                if nid is None or not isinstance(n, dict):
                    continue
                node_xyz[nid] = {
                    "x": n.get("x"),
                    "y": n.get("y"),
                    "z": n.get("z"),
                }
            for mid, ctx in idx.by_member.items():
                if ctx.coordinates is None and ctx.connected_node_ids:
                    ni = ctx.connected_node_ids[0]
                    if ni in node_xyz:
                        ctx.coordinates = node_xyz[ni]

    return idx


def context_to_kwargs(ctx: MemberContext) -> dict[str, Any]:
    """Map MemberContext fields onto StructuralIssue's location kwargs."""
    return {
        "member_type": ctx.member_type,
        "section": ctx.section,
        "material": ctx.material,
        "story": ctx.story,
        "level": ctx.level,
        "connected_node_ids": list(ctx.connected_node_ids),
        "grid_location": ctx.grid_location,
        "coordinates": dict(ctx.coordinates) if isinstance(ctx.coordinates, dict) else None,
    }
