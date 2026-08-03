"""추출 결과 → StructuralModel JSON dict 빌더.

핵심 흐름:
1. RegisteredFrame (world 좌표) + ColumnCandidate 리스트 + TypicalSectionSpec
2. (grid_x, grid_y, story) 노드 dedup → node_id 부여
3. 각 story-pair 별 컬럼 element 생성
4. StructuralModel.from_json() 입력 스키마 dict 반환
5. V2 UI(`📂 Load`)가 받는 `.v2proj.json` 래퍼는 `wrap_v2proj()` 별도 함수

회귀 안전: 본 모듈은 StructuralModel을 import하지 않는다 — dict만 빌드.
하지만 정확성 검증을 위해 caller가 `StructuralModel.from_json(result)` 호출해 round-trip 검증 가능.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .member_extract import merge_columns_across_stories
from .schemas import (
    BeamCandidate,
    ColumnCandidate,
    RegisteredFrame,
    TypicalSectionSpec,
)


_DEFAULT_ENV = {
    "region": "",
    "site_class": "S3",
    "importance": "II",
    "importance_factor": 1.0,
    "seismic_system": "ordinary_moment_frame",
    "seismic_direction": "both",
    "exposure_category": "B",
}


def build_structural_model_dict(
    registered: RegisteredFrame,
    column_candidates: list[ColumnCandidate],
    beam_candidates: list[BeamCandidate],
    typical_sections: TypicalSectionSpec,
    environment: Optional[dict] = None,
    story_usages: Optional[dict[int, str]] = None,
    story_slab_thickness: Optional[dict[int, float]] = None,
    analysis_options: Optional[dict] = None,
) -> dict:
    """StructuralModel.from_json()의 입력 스키마와 동일한 dict 반환.

    Args:
        registered: RegisteredFrame (world_grid_z[0]=base=0m, world_grid_z[N]=N층 elevation)
        column_candidates: ColumnCandidate 리스트.
            **story 의미: 1-based 층 번호**. story=1은 "1층 컬럼"(base→1F),
            story=N → 노드 elev_idx (N-1, N), i-end = world_grid_z[N-1], j-end = world_grid_z[N].
            merge_columns_across_stories 결과의 (story_from, story_to)는 "N층부터 M층까지" 의미.
        beam_candidates: BeamCandidate 리스트 (W5 결과; 비어 있어도 됨)
        typical_sections: 사용자 입력 typical 단면
        environment: region/site_class/importance/… (None이면 default)
        story_usages: {1: "office", 2: "office", …} (None이면 모든 층 "office")
        story_slab_thickness: {1: 0.15, …} (None이면 모든 층 0.15 m)
        analysis_options: {"num_elements_per_member", "rigid_diaphragm", "geometric_nonlinearity"}

    Returns:
        StructuralModel.from_json()이 받는 dict.
    """
    if not registered.world_grid_z:
        raise ValueError("registered.world_grid_z (story elevations) is empty")

    merged_cols = merge_columns_across_stories(column_candidates)
    n_elevations = len(registered.world_grid_z)  # base 포함 elevation 개수

    # ── 노드 dedup ──
    # key=(grid_x, grid_y, elev_idx), value=node_id (1-based)
    # elev_idx 0 = base, elev_idx N = N층 elevation
    node_ids: dict[tuple[str, str, int], int] = {}
    node_records: list[dict] = []
    next_node_id = 1

    def _ensure_node(xl: str, yl: str, elev_idx: int) -> Optional[int]:
        nonlocal next_node_id
        if elev_idx < 0 or elev_idx >= n_elevations:
            return None
        if xl not in registered.world_grid_x or yl not in registered.world_grid_y:
            return None
        key = (xl, yl, elev_idx)
        if key in node_ids:
            return node_ids[key]

        nid = next_node_id
        next_node_id += 1
        node_ids[key] = nid
        x = registered.world_grid_x[xl]
        y = registered.world_grid_y[yl]
        z = registered.world_grid_z[elev_idx]
        node_records.append({
            "id": nid,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "story": int(elev_idx),     # StructuralNode.story: 0=base, N=N층
            "support": "fixed" if elev_idx == 0 else None,
            "mass": None,
        })
        return nid

    # ── 컬럼 elements ──
    elem_records: list[dict] = []
    next_elem_id = 1

    for c in merged_cols:
        # ColumnCandidate(story_from=N, story_to=M): N층 ~ M층 컬럼 (1-based)
        # 필요한 elevation 인덱스: (N-1) ~ M (즉 M-N+2 개)
        # element는 각 층 K ∈ [N, M] 에 대해 [elev_idx=K-1 → K]
        for eidx in range(c.story_from - 1, c.story_to + 1):
            _ensure_node(c.grid_x_label, c.grid_y_label, eidx)

        for story_n in range(c.story_from, c.story_to + 1):
            ni = node_ids.get((c.grid_x_label, c.grid_y_label, story_n - 1))
            nj = node_ids.get((c.grid_x_label, c.grid_y_label, story_n))
            if ni is None or nj is None:
                continue
            elem_records.append({
                "id": next_elem_id,
                "node_i": ni,
                "node_j": nj,
                "elem_type": "column",
                "section": typical_sections.column,
                "material": typical_sections.material,
                "release_i": None,
                "release_j": None,
                "beta_angle": 0.0,
            })
            next_elem_id += 1

    # ── 보 elements ──
    for b in beam_candidates:
        # span_along=vertical_grid: A↔B 사이 보 (X 방향 보)
        #   노드 i = (from_label=A, transverse_label, elev_idx=story)
        #   노드 j = (to_label=B,   transverse_label, elev_idx=story)
        # span_along=horizontal_grid: 1↔2 사이 보 (Y 방향 보)
        #   노드 i = (transverse_label, from_label=1, elev_idx=story)
        #   노드 j = (transverse_label, to_label=2,   elev_idx=story)
        # 보는 그 층의 상부 슬래브에 있으므로 elev_idx = story (1-based 그대로)
        if b.span_along == "vertical_grid":
            ni = _ensure_node(b.from_label, b.transverse_label, b.story)
            nj = _ensure_node(b.to_label,   b.transverse_label, b.story)
            section = typical_sections.beam_x
        elif b.span_along == "horizontal_grid":
            ni = _ensure_node(b.transverse_label, b.from_label, b.story)
            nj = _ensure_node(b.transverse_label, b.to_label,   b.story)
            section = typical_sections.beam_y
        else:
            continue
        if ni is None or nj is None:
            continue
        elem_records.append({
            "id": next_elem_id,
            "node_i": ni,
            "node_j": nj,
            "elem_type": "beam",
            "section": section,
            "material": typical_sections.material,
            "release_i": None,
            "release_j": None,
            "beta_angle": 0.0,
        })
        next_elem_id += 1

    # ── 층 정보 ──
    # story_elevations: 노드의 z 좌표 시퀀스 (base 포함)
    story_elevations = list(registered.world_grid_z)
    if story_usages is None:
        story_usages = {i: "office" for i in range(1, n_elevations)}
    if story_slab_thickness is None:
        story_slab_thickness = {i: 0.15 for i in range(1, n_elevations)}

    env = dict(_DEFAULT_ENV)
    if environment:
        env.update(environment)

    opts = {
        "num_elements_per_member": 4,
        "rigid_diaphragm": False,
        "geometric_nonlinearity": "linear",
    }
    if analysis_options:
        opts.update(analysis_options)

    return {
        "version": "2.0",
        "nodes": node_records,
        "elements": elem_records,
        "story_elevations": story_elevations,
        "story_usages": {str(k): v for k, v in story_usages.items()},
        "story_slab_thickness": {str(k): v for k, v in story_slab_thickness.items()},
        "story_dead_load_finish": {},
        "environment": env,
        "analysis_options": opts,
    }


def wrap_v2proj(model_dict: dict, source_label: str = "cad_parser") -> dict:
    """V2 UI의 `📂 Load` 버튼이 받는 .v2proj.json 래퍼.

    UI의 `loadProject()`는 `project.model.nodes` 존재만 검증하고 `config`/`analysis`는 옵셔널.
    """
    return {
        "version": 3,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "model": model_dict,
    }
