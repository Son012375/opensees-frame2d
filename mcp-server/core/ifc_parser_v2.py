"""V2 IFC 파서 — Node-Element 직접 추출.

V1(격자 기반)과 달리 IFC 부재에서 노드 좌표와 요소 연결을 직접 추출한다.
Midas Gen 방식: 노드 먼저 → 요소(i,j) 연결 → 사용자 편집 가능.

지원 IFC 엔티티:
  - IfcColumn → column 요소
  - IfcBeam → beam 요소
  - IfcMember → brace 요소 (경사 부재)
  - IfcBuildingStorey → 층 표고 참조
  - IfcSlab → 슬래브 두께
  - IfcProfileDef → 단면 정보
  - IfcRelAssociatesMaterial → 재료 정보

Dependencies:
  pip install ifcopenshell numpy

Usage:
    from core.ifc_parser_v2 import parse_ifc_v2
    model, validation = parse_ifc_v2("building.ifc")
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.placement as ifc_placement
    HAS_IFC = True
except ImportError:
    HAS_IFC = False

# ifcopenshell.geom 설정 (글로벌 좌표계, m 단위 자동 변환)
_GEOM_SETTINGS = None

def _get_geom_settings():
    """ifcopenshell.geom 설정 (싱글턴)."""
    global _GEOM_SETTINGS
    if _GEOM_SETTINGS is None and HAS_IFC:
        _GEOM_SETTINGS = ifcopenshell.geom.settings()
        _GEOM_SETTINGS.set(_GEOM_SETTINGS.USE_WORLD_COORDS, True)
    return _GEOM_SETTINGS

from core.structural_model import (
    StructuralModel,
    StructuralNode,
    StructuralElement,
    ElementType,
    SupportType,
    IFCValidationResult,
    ValidationIssue,
    Severity,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  좌표 변환
# ──────────────────────────────────────────────────────────────

def _compute_endpoints_geometry(element, settings) -> tuple[np.ndarray, np.ndarray] | None:
    """IFC 구조 부재의 글로벌 시작/끝점 좌표 계산 (geometry 기반, m).

    ifcopenshell.geom으로 실제 3D 형상을 생성한 뒤,
    부재 축 방향(가장 긴 방향)의 양 끝 중심점을 시작/끝점으로 결정한다.

    이 방법은 Revit의 보 Extrusion 방향 문제를 우회한다:
    - Revit 보: ObjectPlacement에 회전 없음 + Extrusion이 로컬 Z축
    - geometry 기반: 실제 글로벌 형상에서 축 방향을 직접 추출

    Returns:
        (start_xyz, end_xyz) in m, 또는 None
    """
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)

        if len(verts) < 4:
            return None

        # 부재 축 방향 = bounding box에서 가장 긴 방향
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        ranges = bbox_max - bbox_min

        # 가장 긴 축 = 부재 축
        main_axis = int(np.argmax(ranges))
        member_length = ranges[main_axis]

        if member_length < 0.01:  # 1cm 미만
            return None

        # 축 방향 양쪽 끝 정점 그룹의 중심점
        axis_vals = verts[:, main_axis]
        axis_min = bbox_min[main_axis]
        axis_max = bbox_max[main_axis]

        # 양 끝 10% 범위의 정점 → 중심
        threshold = member_length * 0.1
        start_mask = axis_vals < (axis_min + threshold)
        end_mask = axis_vals > (axis_max - threshold)

        start_pts = verts[start_mask]
        end_pts = verts[end_mask]

        if len(start_pts) == 0 or len(end_pts) == 0:
            return None

        # 단면 중심 (축 직교 방향의 중심)
        other_axes = [i for i in range(3) if i != main_axis]
        start_center = np.zeros(3)
        end_center = np.zeros(3)

        start_center[main_axis] = axis_min
        end_center[main_axis] = axis_max

        for ax in other_axes:
            start_center[ax] = (start_pts[:, ax].min() + start_pts[:, ax].max()) / 2
            end_center[ax] = (end_pts[:, ax].min() + end_pts[:, ax].max()) / 2

        return start_center, end_center

    except Exception as e:
        logger.warning(f"Geometry 추출 실패 [{element.GlobalId}]: {e}")
        return None


def _compute_endpoints_placement(element) -> tuple[np.ndarray, np.ndarray] | None:
    """Placement 기반 끝점 계산 (fallback, mm).

    기둥처럼 Extrusion 방향이 명확한 경우에 사용.
    """
    try:
        if not element.ObjectPlacement:
            return None
        matrix = np.array(
            ifc_placement.get_local_placement(element.ObjectPlacement),
            dtype=float,
        )
    except Exception:
        return None

    start = matrix[:3, 3].copy()

    # Extrusion 길이
    length = None
    local_dir = np.array([0.0, 0.0, 1.0])
    try:
        if element.Representation:
            for rep in element.Representation.Representations:
                for item in rep.Items:
                    extruded = None
                    if item.is_a("IfcExtrudedAreaSolid"):
                        extruded = item
                    elif item.is_a("IfcMappedItem"):
                        src = item.MappingSource.MappedRepresentation
                        for si in src.Items:
                            if si.is_a("IfcExtrudedAreaSolid"):
                                extruded = si
                                break
                    if extruded:
                        length = float(extruded.Depth)
                        d = extruded.ExtrudedDirection.DirectionRatios
                        local_dir = np.array([float(d[0]), float(d[1]), float(d[2])])
                        break
                if length:
                    break
    except Exception:
        pass

    if length is None or length < 1.0:
        return None

    rotation = matrix[:3, :3]
    global_dir = rotation @ local_dir
    norm = np.linalg.norm(global_dir)
    if norm < 1e-9:
        return None
    global_dir /= norm

    end = start + global_dir * length
    # mm → m 변환
    return start / 1000.0, end / 1000.0


# ──────────────────────────────────────────────────────────────
#  단면/재료 추출 (V1 재사용)
# ──────────────────────────────────────────────────────────────

def _get_element_section_name(element) -> str | None:
    """IFC 요소에서 단면 이름 추출.

    탐색 순서:
    1. IfcProfileDef.ProfileName (직접)
    2. IfcTypeObject의 프로파일
    3. IfcMaterialProfileSetUsage의 프로파일
    """
    # V1 함수 재사용
    from core.ifc_parser import _get_element_profile
    profile = _get_element_profile(element)
    if profile and profile.get("profile_name"):
        return profile["profile_name"]

    # 프로파일 치수로 이름 생성
    if profile:
        h, b = round(profile["h"]), round(profile["b"])
        return f"H-{h}x{b}"

    return None


def _get_element_material_name(element) -> str | None:
    """IFC 요소에서 재료명 추출."""
    from core.ifc_parser import _normalize_material_name

    try:
        for rel in element.HasAssociations or []:
            if rel.is_a("IfcRelAssociatesMaterial"):
                mat = rel.RelatingMaterial
                if mat.is_a("IfcMaterial"):
                    return _normalize_material_name(mat.Name or "")
                elif mat.is_a("IfcMaterialProfileSetUsage"):
                    prof_set = mat.ForProfileSet
                    for mp in prof_set.MaterialProfiles or []:
                        if mp.Material:
                            name = _normalize_material_name(mp.Material.Name or "")
                            if name:
                                return name
                elif mat.is_a("IfcMaterialLayerSetUsage"):
                    for layer in mat.ForLayerSet.MaterialLayers or []:
                        if layer.Material:
                            name = _normalize_material_name(layer.Material.Name or "")
                            if name:
                                return name
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
#  층 정보 추출
# ──────────────────────────────────────────────────────────────

def _get_storey_elevations(ifc_file) -> list[tuple[str, float]]:
    """IfcBuildingStorey에서 층 이름과 표고(mm) 추출."""
    storeys = ifc_file.by_type("IfcBuildingStorey")
    result = []
    for s in storeys:
        try:
            elev = float(s.Elevation) if s.Elevation is not None else 0.0
            result.append((s.Name or f"Story_{len(result)}", elev))
        except (TypeError, ValueError):
            continue
    return sorted(result, key=lambda x: x[1])


def _get_slab_thickness(ifc_file) -> float | None:
    """IfcSlab에서 슬래브 두께 추출 (mm). 가장 흔한 두께 반환."""
    slabs = ifc_file.by_type("IfcSlab")
    thicknesses = []
    for slab in slabs:
        try:
            if not slab.Representation:
                continue
            for rep in slab.Representation.Representations:
                for item in rep.Items:
                    if item.is_a("IfcExtrudedAreaSolid"):
                        thicknesses.append(float(item.Depth))
        except Exception:
            continue
    if not thicknesses:
        return None
    # 가장 흔한 두께
    from collections import Counter
    counter = Counter(round(t, 0) for t in thicknesses)
    return counter.most_common(1)[0][0]


# ──────────────────────────────────────────────────────────────
#  IFC 검증
# ──────────────────────────────────────────────────────────────

def validate_ifc(ifc_file) -> IFCValidationResult:
    """IFC 파일의 구조해석 적합성 검증.

    Returns:
        IFCValidationResult — CRITICAL 이슈가 있으면 is_valid=False
    """
    result = IFCValidationResult()
    issues = result.issues

    # 1. 구조 부재 존재 확인
    columns = ifc_file.by_type("IfcColumn")
    beams = ifc_file.by_type("IfcBeam")
    members = ifc_file.by_type("IfcMember")

    total_structural = len(columns) + len(beams) + len(members)

    if total_structural == 0:
        # IfcBuildingElementProxy 확인 (Revit 매핑 오류 대응)
        proxies = ifc_file.by_type("IfcBuildingElementProxy")
        if proxies:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="IFC_PROXY_ONLY",
                message=(
                    f"구조 부재(IfcColumn/IfcBeam)가 없고 "
                    f"IfcBuildingElementProxy {len(proxies)}개만 발견됨. "
                    "Revit에서 IFC 내보내기 시 카테고리 매핑을 확인하세요."
                ),
            ))
        else:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="IFC_NO_STRUCTURAL",
                message="구조 부재(IfcColumn, IfcBeam, IfcMember)를 찾을 수 없습니다.",
            ))
        result.is_valid = False
        return result

    if not columns:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_NO_COLUMN",
            message=f"IfcColumn이 없습니다 (IfcBeam {len(beams)}개, IfcMember {len(members)}개만 존재).",
        ))

    if not beams:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_NO_BEAM",
            message=f"IfcBeam이 없습니다 (IfcColumn {len(columns)}개, IfcMember {len(members)}개만 존재).",
        ))

    # 2. 좌표 정보 (ObjectPlacement) 확인
    no_placement = []
    for entity_type, entities in [("IfcColumn", columns), ("IfcBeam", beams), ("IfcMember", members)]:
        for elem in entities:
            if not elem.ObjectPlacement:
                no_placement.append(elem.GlobalId)

    if no_placement:
        if len(no_placement) == total_structural:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="IFC_NO_PLACEMENT",
                message="모든 구조 부재에 ObjectPlacement(좌표)가 없습니다.",
            ))
            result.is_valid = False
            return result
        else:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                code="IFC_PARTIAL_PLACEMENT",
                message=f"{len(no_placement)}개 부재에 ObjectPlacement가 없습니다 (해당 부재 제외).",
                element_ids=no_placement[:20],  # 처음 20개만
            ))

    # 3. 층 정보 확인
    storeys = ifc_file.by_type("IfcBuildingStorey")
    if not storeys:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_NO_STOREY",
            message="IfcBuildingStorey가 없습니다. Z좌표 클러스터링으로 층을 자동 감지합니다.",
        ))

    # 4. 단면 정보 확인
    no_profile_count = 0
    no_profile_ids = []
    for entities in [columns, beams, members]:
        for elem in entities:
            section = _get_element_section_name(elem)
            if not section:
                no_profile_count += 1
                no_profile_ids.append(elem.GlobalId)

    if no_profile_count > 0:
        pct = round(no_profile_count / total_structural * 100, 1)
        issue = ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_NO_PROFILE",
            message=f"{no_profile_count}개 부재({pct}%)의 단면 정보가 없습니다.",
            element_ids=no_profile_ids[:20],
            default_value="H-400x200 (보), H-300x300 (기둥)",
        )
        issues.append(issue)
        result.needs_user_input.append(issue)

    # 5. 재료 정보 확인
    no_material_count = 0
    for entities in [columns, beams, members]:
        for elem in entities:
            mat = _get_element_material_name(elem)
            if not mat:
                no_material_count += 1

    if no_material_count > 0:
        pct = round(no_material_count / total_structural * 100, 1)
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_NO_MATERIAL",
            message=f"{no_material_count}개 부재({pct}%)의 재료 정보가 없습니다.",
            default_value="SS275",
        ))

    # 6. 슬래브 확인
    slabs = ifc_file.by_type("IfcSlab")
    if not slabs:
        issue = ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_NO_SLAB",
            message="IfcSlab가 없습니다. 슬래브 두께를 수동으로 입력해야 합니다.",
            default_value="200mm",
        )
        issues.append(issue)
        result.needs_user_input.append(issue)

    # 7. 통계
    issues.append(ValidationIssue(
        severity=Severity.INFO,
        code="IFC_STATS",
        message=(
            f"IFC 구조 부재: 기둥 {len(columns)}개, 보 {len(beams)}개, "
            f"기타 {len(members)}개, 슬래브 {len(slabs)}개, "
            f"층 {len(storeys)}개"
        ),
    ))

    return result


# ──────────────────────────────────────────────────────────────
#  메인 파서
# ──────────────────────────────────────────────────────────────

def _classify_by_direction(start: np.ndarray, end: np.ndarray, angle_threshold: float = 30.0) -> ElementType:
    """시작/끝점에서 요소 유형 자동 분류.

    Args:
        start, end: 좌표 (m)
        angle_threshold: 수직/수평 판별 각도 (degree)
    """
    diff = end - start
    length = np.linalg.norm(diff)
    if length < 0.001:  # 1mm 미만
        return ElementType.BEAM

    dz = abs(diff[2]) / length
    vert_angle = math.degrees(math.acos(min(dz, 1.0)))

    if vert_angle <= angle_threshold:
        return ElementType.COLUMN
    elif vert_angle >= (90.0 - angle_threshold):
        return ElementType.BEAM
    else:
        return ElementType.BRACE


def _find_or_create_node(
    model: StructuralModel,
    x: float, y: float, z: float,
    node_map: dict[tuple[int, int, int], int],
    tolerance_m: float = 0.01,
) -> int:
    """좌표에 해당하는 노드를 찾거나 새로 생성.

    tolerance 기준으로 반올림한 키로 빠르게 검색.

    Args:
        model: StructuralModel
        x, y, z: 좌표 (m)
        node_map: (rx, ry, rz) → node_id 매핑
        tolerance_m: 노드 병합 허용 오차 (m, 기본 10mm)

    Returns:
        node_id
    """
    # 반올림 키 (tolerance 기반)
    rkey = (
        round(x / tolerance_m),
        round(y / tolerance_m),
        round(z / tolerance_m),
    )

    if rkey in node_map:
        return node_map[rkey]

    # 새 노드 생성
    node = model.add_node(x=round(x, 4), y=round(y, 4), z=round(z, 4))
    node_map[rkey] = node.id
    return node.id


def parse_ifc_v2(
    ifc_path: str,
    tolerance_mm: float = 10.0,
    default_beam_section: str = "H-400x200",
    default_column_section: str = "H-300x300",
    default_material: str = "SS275",
) -> tuple[StructuralModel, IFCValidationResult]:
    """IFC 파일에서 Node-Element 기반 StructuralModel 생성.

    Args:
        ifc_path: IFC 파일 경로
        tolerance_mm: 노드 병합 허용 오차 (mm)
        default_beam_section: 단면 미지정 보의 기본 단면
        default_column_section: 단면 미지정 기둥의 기본 단면
        default_material: 재료 미지정 부재의 기본 재료

    Returns:
        (StructuralModel, IFCValidationResult) 튜플
    """
    if not HAS_IFC:
        raise ImportError("ifcopenshell이 필요합니다: pip install ifcopenshell")

    ifc_file = ifcopenshell.open(ifc_path)

    # ── Step 1: 검증 ──
    validation = validate_ifc(ifc_file)
    if not validation.is_valid:
        return StructuralModel(), validation

    # ── Step 2: 노드 & 요소 추출 ──
    model = StructuralModel()
    node_map: dict[tuple[int, int, int], int] = {}
    _seen_edges: set[tuple[int, int]] = set()  # 중복 요소 방지
    failed_count = 0
    tolerance_m = tolerance_mm / 1000.0

    # Geometry 설정 (글로벌 좌표, m 단위)
    geom_settings = _get_geom_settings()

    # 처리할 IFC 엔티티 목록
    entity_groups = [
        ("IfcColumn", ifc_file.by_type("IfcColumn")),
        ("IfcBeam", ifc_file.by_type("IfcBeam")),
        ("IfcMember", ifc_file.by_type("IfcMember")),
    ]

    for entity_type_name, entities in entity_groups:
        for elem in entities:
            # 시작/끝점 계산 (geometry 기반 우선, fallback: placement)
            endpoints = _compute_endpoints_geometry(elem, geom_settings)
            if endpoints is None:
                endpoints = _compute_endpoints_placement(elem)
            if endpoints is None:
                failed_count += 1
                logger.warning(
                    f"좌표 변환 실패: {entity_type_name} [{elem.GlobalId}]"
                )
                continue

            start, end = endpoints  # (m 단위)

            # 노드 생성/병합
            ni = _find_or_create_node(model, start[0], start[1], start[2],
                                       node_map, tolerance_m)
            nj = _find_or_create_node(model, end[0], end[1], end[2],
                                       node_map, tolerance_m)

            # 같은 노드로 병합된 경우 (길이 0) → skip
            if ni == nj:
                failed_count += 1
                logger.warning(
                    f"길이 0 요소 skip: {entity_type_name} [{elem.GlobalId}]"
                )
                continue

            # 중복 요소 skip (같은 ni-nj 쌍)
            edge_key = (min(ni, nj), max(ni, nj))
            if edge_key in _seen_edges:
                logger.warning(
                    f"중복 요소 skip: {entity_type_name} [{elem.GlobalId}] "
                    f"nodes {edge_key} (기존 elem과 동일)"
                )
                continue
            _seen_edges.add(edge_key)

            # 요소 유형 분류
            elem_type = _classify_by_direction(start, end)

            # 단면 추출
            section = _get_element_section_name(elem)
            if not section:
                if elem_type == ElementType.COLUMN:
                    section = default_column_section
                else:
                    section = default_beam_section

            # 재료 추출
            material = _get_element_material_name(elem) or default_material

            # 요소 추가
            model.add_element(
                node_i=ni, node_j=nj,
                elem_type=elem_type,
                section=section,
                material=material,
            )

    # ── Step 3: 층 감지 ──
    # IFC 층 정보 활용
    storey_data = _get_storey_elevations(ifc_file)
    if storey_data:
        # IFC 층 표고를 mm → m 변환 후 설정
        model.story_elevations = [round(elev / 1000.0, 4) for _, elev in storey_data]
        # 각 노드에 가장 가까운 층 할당
        for node in model.nodes.values():
            best_story = 0
            best_dist = float("inf")
            for i, elev in enumerate(model.story_elevations):
                dist = abs(node.z - elev)
                if dist < best_dist:
                    best_dist = dist
                    best_story = i
            node.story = best_story
    else:
        # IfcBuildingStorey 없으면 Z 클러스터링
        model.detect_stories(tolerance=0.3)

    # ── Step 4: 기초 노드 자동 경계조건 ──
    if model.story_elevations:
        base_elev = model.story_elevations[0]
        for node in model.nodes.values():
            if node.story == 0 and abs(node.z - base_elev) < 0.1:
                node.support = SupportType.FIXED

    # ── Step 5: 슬래브 두께 ──
    slab_t = _get_slab_thickness(ifc_file)
    if slab_t is not None:
        t_m = round(slab_t / 1000.0, 4)
        for story_idx in range(1, len(model.story_elevations)):
            model.story_slab_thickness[story_idx] = t_m

    # ── Step 6: 보-기둥 접합 검증 ──
    snap_candidates = _detect_disconnected_joints(model, snap_tolerance=0.5)
    if snap_candidates:
        issue = ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_DISCONNECTED_JOINTS",
            message=(
                f"보 끝점 {len(snap_candidates)}개가 가장 가까운 기둥 노드와 "
                f"떨어져 있습니다 (단면 오프셋 추정). "
                "구조해석을 위해 보 끝점을 기둥 노드에 스냅하는 것을 권장합니다."
            ),
            default_value="snap_to_columns",
        )
        validation.issues.append(issue)
        validation.needs_user_input.append(issue)

    # ── Step 7: 검증 결과 업데이트 ──
    validation.extracted_nodes = len(model.nodes)
    validation.extracted_elements = len(model.elements)
    validation.failed_elements = failed_count
    model.validation = validation

    if failed_count > 0:
        validation.issues.append(ValidationIssue(
            severity=Severity.WARNING,
            code="IFC_FAILED_ELEMENTS",
            message=f"{failed_count}개 부재의 좌표 변환에 실패했습니다.",
        ))

    logger.info(
        f"IFC V2 파싱 완료: 노드 {len(model.nodes)}, "
        f"요소 {len(model.elements)}, 실패 {failed_count}"
    )

    return model, validation


# ──────────────────────────────────────────────────────────────
#  접합 검증 & 스냅
# ──────────────────────────────────────────────────────────────

def _detect_disconnected_joints(
    model: StructuralModel,
    snap_tolerance: float = 0.5,
) -> list[dict]:
    """보 끝점이 기둥 노드에 연결되지 않은 경우를 감지.

    Args:
        model: StructuralModel
        snap_tolerance: 스냅 대상 최대 거리 (m)

    Returns:
        스냅 후보 리스트: [{"beam_node_id", "column_node_id", "distance"}]
    """
    # 기둥 노드 수집 (양 끝)
    column_node_ids = set()
    for e in model.elements.values():
        if e.elem_type == ElementType.COLUMN:
            column_node_ids.add(e.node_i)
            column_node_ids.add(e.node_j)

    # 보 끝 노드 수집
    beam_end_node_ids = set()
    for e in model.elements.values():
        if e.elem_type == ElementType.BEAM:
            beam_end_node_ids.add(e.node_i)
            beam_end_node_ids.add(e.node_j)

    # 기둥에 이미 연결된 보 노드는 제외
    disconnected_beam_nodes = beam_end_node_ids - column_node_ids

    if not disconnected_beam_nodes:
        return []

    # 각 disconnected 보 노드에 대해 가장 가까운 기둥 노드 찾기
    candidates = []
    col_nodes = [model.nodes[nid] for nid in column_node_ids if nid in model.nodes]

    for bn_id in disconnected_beam_nodes:
        bn = model.nodes.get(bn_id)
        if bn is None:
            continue
        best_dist = float("inf")
        best_col_id = None
        for cn in col_nodes:
            d = bn.distance_to(cn)
            if d < best_dist:
                best_dist = d
                best_col_id = cn.id
        if best_col_id is not None and best_dist <= snap_tolerance:
            candidates.append({
                "beam_node_id": bn_id,
                "column_node_id": best_col_id,
                "distance_m": round(best_dist, 4),
            })

    return candidates


def snap_beams_to_columns(model: StructuralModel, snap_tolerance: float = 0.5) -> int:
    """보 끝점을 가장 가까운 기둥 노드에 스냅 (사용자 확인 후 호출).

    보 끝점 노드를 기둥 노드로 병합하고, 요소 참조를 업데이트한다.

    Args:
        model: StructuralModel
        snap_tolerance: 스냅 대상 최대 거리 (m)

    Returns:
        스냅된 노드 수
    """
    candidates = _detect_disconnected_joints(model, snap_tolerance)
    if not candidates:
        return 0

    # beam_node → column_node 매핑
    snap_map: dict[int, int] = {}
    for c in candidates:
        snap_map[c["beam_node_id"]] = c["column_node_id"]

    # 요소의 노드 참조 업데이트
    for elem in model.elements.values():
        if elem.node_i in snap_map:
            elem.node_i = snap_map[elem.node_i]
        if elem.node_j in snap_map:
            elem.node_j = snap_map[elem.node_j]

    # 스냅된 노드 삭제 (더 이상 참조되지 않는 것만)
    all_referenced = set()
    for elem in model.elements.values():
        all_referenced.add(elem.node_i)
        all_referenced.add(elem.node_j)

    removed = 0
    for old_id in snap_map:
        if old_id not in all_referenced and old_id in model.nodes:
            del model.nodes[old_id]
            removed += 1

    return len(candidates)
