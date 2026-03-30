"""IFC 파일에서 건물 기하 정보를 추출하는 모듈.

Revit 등 BIM 소프트웨어에서 내보낸 IFC 파일을 파싱하여
정규 그리드 기반 BuildingModel로 변환한다.

지원 구조 유형:
  - 골조 구조: IfcColumn 위치 → X/Y 그리드 감지
  - 벽식 구조: IfcWallStandardCase 위치 → X/Y 그리드 감지

주요 추출 항목:
  - IfcBuildingStorey → 층 높이
  - IfcColumn 또는 IfcWallStandardCase → 그리드 감지
  - IfcSlab → 슬래브 두께
  - IfcColumn/IfcBeam → IfcIShapeProfileDef → 단면 치수
  - IfcRelAssociatesMaterial → 재료명

Dependencies:
  pip install ifcopenshell

Usage:
    from core.ifc_parser import parse_ifc
    ifc_data = parse_ifc("building.ifc")
    model = BuildingModel.from_ifc("building.ifc", config)
"""
from __future__ import annotations

import re
from typing import Optional

try:
    import ifcopenshell
    import ifcopenshell.util.placement as ifc_placement
    HAS_IFC = True
except ImportError:
    HAS_IFC = False


def _cluster_coords(coords: list[float], tolerance: float = 200.0) -> list[float]:
    """좌표 리스트에서 클러스터 중심점 추출.

    tolerance (mm) 이내의 좌표를 동일 그리드라인으로 간주한다.
    """
    if not coords:
        return []

    sorted_c = sorted(float(c) for c in coords)
    clusters = [[sorted_c[0]]]

    for c in sorted_c[1:]:
        if abs(c - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    return [sum(cl) / len(cl) for cl in clusters]


def _detect_grid(coords: list[float], tolerance: float = 200.0) -> list[float]:
    """좌표 리스트에서 그리드 간격 감지."""
    centers = _cluster_coords(coords, tolerance)
    if len(centers) < 2:
        return []

    bays = []
    for i in range(len(centers) - 1):
        bay_mm = centers[i + 1] - centers[i]
        bays.append(round(bay_mm / 1000.0, 3))

    return bays


def _detect_zones_from_occupancy(
    col_positions: list[tuple[float, float, float]],
    grid_x_centers: list[float],
    grid_y_centers: list[float],
    tolerance: float = 200.0,
) -> list[dict] | None:
    """기둥 점유 격자를 분석하여 비정형 zones를 감지.

    모든 격자 교차점에 기둥이 있으면 None (정형).
    빈 교차점이 있으면 직사각형 존 목록을 반환.

    알고리즘:
    1. 점유 격자 생성
    2. 행별 X-범위 프로파일 계산
    3. 동일 프로파일의 연속 행을 그룹으로 묶기
    4. 각 그룹을 존으로 변환 (인접 그룹과 1행 중첩하여 노드 공유)

    Returns:
        None (정형) or list of zone dicts
    """
    nx = len(grid_x_centers)
    ny = len(grid_y_centers)
    if nx < 2 or ny < 2:
        return None

    # 점유 격자 생성
    col_xy = [(p[0], p[1]) for p in col_positions]
    occupied = [[False] * ny for _ in range(nx)]

    for ix in range(nx):
        for iy in range(ny):
            gx, gy = grid_x_centers[ix], grid_y_centers[iy]
            for cx, cy in col_xy:
                if abs(cx - gx) <= tolerance and abs(cy - gy) <= tolerance:
                    occupied[ix][iy] = True
                    break

    # 전부 점유 → 정형
    total = nx * ny
    occupied_count = sum(occupied[ix][iy] for ix in range(nx) for iy in range(ny))
    if occupied_count >= total:
        return None
    if occupied_count < 4:
        return None

    # 행별 점유 ix 집합 (내부 갭 포함)
    row_occ_sets = {}
    for iy in range(ny):
        occ = frozenset(ix for ix in range(nx) if occupied[ix][iy])
        row_occ_sets[iy] = occ

    # 행별 X-범위 프로파일: (ix_min, ix_max)
    # 내부 갭 감지를 위해 실제 점유 set도 비교
    row_profiles = []
    for iy in range(ny):
        occ = row_occ_sets[iy]
        if occ:
            row_profiles.append((iy, min(occ), max(occ)))
        else:
            row_profiles.append((iy, -1, -1))

    # 동일 X-범위의 연속 행 그룹화
    # 내부 갭(1~2개 빠진 기둥)은 무시하고 (ix_min, ix_max) 범위로 그룹화.
    # 내부 갭은 별도 경고로 보고됨.
    groups = []  # [(iy_start, iy_end, ix_min, ix_max)]
    for iy, ix_min, ix_max in row_profiles:
        if ix_min < 0:
            continue
        if (groups and groups[-1][3] == ix_max and groups[-1][2] == ix_min
                and iy == groups[-1][1] + 1):
            groups[-1] = (groups[-1][0], iy, ix_min, ix_max)
        else:
            groups.append((iy, iy, ix_min, ix_max))

    if len(groups) <= 1:
        return None  # 단일 그룹 → 정형

    # 존 생성 (인접 그룹과 1행 중첩으로 노드 공유)
    zones = []
    zone_id = ord('A')

    for gi, (iy_s, iy_e, ix_s, ix_e) in enumerate(groups):
        actual_iy_s = iy_s
        actual_iy_e = iy_e

        # 이후 그룹만 이전 그룹의 마지막 행을 포함하여 경계 공유.
        # 이전 그룹은 확장하지 않음 (확장하면 빈 교차점까지 포함됨).
        if gi > 0:
            prev_iy_e = groups[gi - 1][1]
            if iy_s == prev_iy_e + 1:
                actual_iy_s = prev_iy_e  # 이전 그룹 마지막 행 포함

        # bays 계산
        z_bays_x = [
            round((grid_x_centers[i + 1] - grid_x_centers[i]) / 1000.0, 3)
            for i in range(ix_s, ix_e)
        ]
        z_bays_y = [
            round((grid_y_centers[j + 1] - grid_y_centers[j]) / 1000.0, 3)
            for j in range(actual_iy_s, actual_iy_e)
        ]

        if not z_bays_x or not z_bays_y:
            continue

        zones.append({
            "id": chr(zone_id),
            "bays_x": z_bays_x,
            "bays_y": z_bays_y,
            "origin_x": round(grid_x_centers[ix_s] / 1000.0, 3),
            "origin_y": round(grid_y_centers[actual_iy_s] / 1000.0, 3),
            "story_from": 1,
            "story_to": None,
        })
        zone_id += 1

    if len(zones) <= 1:
        return None

    return zones


def _get_per_storey_columns(ifc_file, tolerance: float = 200.0) -> dict[str, list[tuple[float, float]]]:
    """IFC에서 층별 기둥 (x, y) 위치 추출.

    Returns:
        {storey_name: [(x_mm, y_mm), ...]}
    """
    rels = ifc_file.by_type("IfcRelContainedInSpatialStructure")
    result = {}
    for rel in rels:
        structure = rel.RelatingStructure
        if not structure.is_a("IfcBuildingStorey"):
            continue
        cols = []
        for elem in rel.RelatedElements:
            if elem.is_a("IfcColumn") and elem.ObjectPlacement:
                try:
                    m = ifcopenshell.util.placement.get_local_placement(
                        elem.ObjectPlacement
                    )
                    cols.append((float(m[0][3]), float(m[1][3])))
                except Exception:
                    pass
        if cols:
            result[structure.Name] = list(set(cols))
    return result


def _refine_zones_by_storey(
    zones: list[dict],
    storey_cols: dict[str, list[tuple[float, float]]],
    stories: list[dict],
    grid_x_mm: list[float],
    grid_y_mm: list[float],
    tolerance: float,
    warnings: list[str],
):
    """층별 기둥 데이터로 zone의 story_to를 정밀 설정 (v2: 경계/고유 분리).

    각 zone 영역 내 기둥이 사라지는 최초 층을 감지하여 story_to를 설정.
    경계(다른 zone과 공유) 기둥은 제외하고, zone 고유 영역 기둥만 판단.
    또한 내부 갭(빠진 기둥)을 경고로 보고.
    """
    storey_names = [s.get("name", f"{i+1}F") for i, s in enumerate(stories)]

    # 다른 zone과의 경계 그리드 포인트 계산 (공유 영역)
    def _zone_grid_pts(zone_dict):
        ox = zone_dict["origin_x"] * 1000
        oy = zone_dict["origin_y"] * 1000
        wx = sum(zone_dict["bays_x"]) * 1000
        wy = sum(zone_dict["bays_y"]) * 1000
        pts = set()
        for gx in grid_x_mm:
            for gy in grid_y_mm:
                if (ox - tolerance <= gx <= ox + wx + tolerance
                        and oy - tolerance <= gy <= oy + wy + tolerance):
                    pts.add((round(gx, 0), round(gy, 0)))
        return pts

    zone_pts = {i: _zone_grid_pts(z) for i, z in enumerate(zones)}

    for zi, zone in enumerate(zones):
        all_pts = zone_pts[zi]
        if not all_pts:
            continue

        # 다른 zone과 공유하는 경계 포인트
        shared_pts = set()
        for zj in range(len(zones)):
            if zj != zi:
                shared_pts |= (all_pts & zone_pts[zj])

        # zone 고유 포인트 (경계 제외)
        unique_pts = all_pts - shared_pts
        n_total = len(all_pts)
        n_unique = len(unique_pts)

        # 층별로 zone 영역 내 기둥 수 확인
        last_active_story = len(stories)
        for si, sname in enumerate(storey_names):
            story_num = si + 1
            cols = storey_cols.get(sname, [])

            # zone 전체 영역 내 기둥
            found_all = set()
            for cx, cy in cols:
                pt = (round(cx, 0), round(cy, 0))
                if pt in all_pts:
                    found_all.add(pt)

            zone_cols = len(found_all)

            # 기둥이 하나도 없으면 확실히 비활성
            if zone_cols == 0:
                last_active_story = story_num - 1
                break

            # zone 고유 영역 기둥 확인
            unique_found = found_all & unique_pts
            if n_unique > 0 and len(unique_found) == 0:
                # 고유 영역 기둥 전멸 → 이 zone은 이전 층까지만
                last_active_story = story_num - 1
                break

            # 전체 기둥이 75% 미만이면 비활성 (고유 영역이 없는 경우 fallback)
            if zone_cols < n_total * 0.75:
                last_active_story = story_num - 1
                break

            # 내부 갭 감지
            if 0 < zone_cols < n_total:
                missing = n_total - zone_cols
                warnings.append(
                    f"Zone {zone['id']} {sname}: "
                    f"격자 교차점 {n_total}개 중 {missing}개에 기둥 없음 (내부 갭)"
                )

        if last_active_story < len(stories):
            zone["story_to"] = last_active_story
            warnings.append(
                f"Zone {zone['id']}: {last_active_story}층까지만 기둥 존재 "
                f"(story_to={last_active_story} 자동 설정)"
            )


def _get_column_positions(ifc_file) -> list[tuple[float, float, float]]:
    """IFC 파일에서 기둥 위치 (x, y, z) 추출."""
    positions = []
    columns = ifc_file.by_type("IfcColumn")

    for col in columns:
        try:
            matrix = ifc_placement.get_local_placement(col.ObjectPlacement)
            x = float(matrix[0][3])
            y = float(matrix[1][3])
            z = float(matrix[2][3])
            positions.append((x, y, z))
        except Exception:
            continue

    return positions


def _get_wall_global_direction(matrix) -> str | None:
    """벽의 placement 회전행렬에서 글로벌 방향 판단.

    프로파일의 로컬 X축(길이 방향)이 글로벌 X/Y 중
    어느 축에 더 가까운지 판단한다.
    """
    # 회전행렬의 첫 번째 열 = 로컬 X축의 글로벌 방향
    local_x_gx = float(matrix[0][0])
    local_x_gy = float(matrix[1][0])

    if abs(local_x_gx) > abs(local_x_gy):
        return "x"  # 벽이 글로벌 X방향으로 배치
    elif abs(local_x_gy) > abs(local_x_gx):
        return "y"  # 벽이 글로벌 Y방향으로 배치
    return None


def _get_wall_grid_coords(ifc_file, base_elevation: float = 0.0,
                          tolerance: float = 200.0
                          ) -> tuple[list[float], list[float], list[dict]]:
    """벽체 위치에서 그리드 좌표 추출.

    회전행렬로 벽의 글로벌 방향을 판단하고,
    벽 위치에서 그리드라인을 추론한다.

    - X방향 벽(로컬X ∥ 글로벌X): 해당 벽의 Y좌표 = Y 그리드라인
    - Y방향 벽(로컬X ∥ 글로벌Y): 해당 벽의 X좌표 = X 그리드라인
    """
    walls = ifc_file.by_type("IfcWallStandardCase")
    if not walls:
        walls = ifc_file.by_type("IfcWall")

    x_coords = []
    y_coords = []
    wall_info = []

    for w in walls:
        try:
            matrix = ifc_placement.get_local_placement(w.ObjectPlacement)
            x = float(matrix[0][3])
            y = float(matrix[1][3])
            z = float(matrix[2][3])

            # 기준층 벽만 사용
            if abs(z - base_elevation) > tolerance:
                continue

            # 형상 추출 (프로파일 XDim = 길이, YDim = 두께)
            profile_x, profile_y, depth = None, None, None
            if w.Representation:
                for rep in w.Representation.Representations:
                    for item in rep.Items:
                        if item.is_a("IfcExtrudedAreaSolid"):
                            depth = float(item.Depth)
                            profile = item.SweptArea
                            if profile.is_a("IfcRectangleProfileDef"):
                                profile_x = float(profile.XDim)
                                profile_y = float(profile.YDim)

            # 회전행렬 기반 방향 판단
            orientation = _get_wall_global_direction(matrix)

            info = {
                "tag": w.Tag, "x": x, "y": y, "z": z,
                "length": profile_x, "thickness": profile_y,
                "height": depth, "orientation": orientation,
                "name": w.Name,
            }
            wall_info.append(info)

            # 그리드 좌표 수집
            if orientation == "x":
                # X방향 벽 → Y좌표가 그리드라인
                y_coords.append(y)
            elif orientation == "y":
                # Y방향 벽 → X좌표가 그리드라인
                x_coords.append(x)

        except Exception:
            continue

    return x_coords, y_coords, wall_info


def _get_storey_elevations(ifc_file) -> list[tuple[str, float]]:
    """IFC 파일에서 층 표고 추출."""
    storeys = ifc_file.by_type("IfcBuildingStorey")
    result = []
    for s in storeys:
        elev = s.Elevation if s.Elevation is not None else 0.0
        result.append((s.Name or f"Story_{len(result)+1}", float(elev)))

    result.sort(key=lambda x: x[1])
    return result


def _get_slab_info(ifc_file) -> list[dict]:
    """IFC 파일에서 슬래브 정보 추출."""
    slabs = ifc_file.by_type("IfcSlab")
    result = []
    seen = set()

    for sl in slabs:
        tag = sl.Tag
        if tag in seen:
            continue
        seen.add(tag)

        try:
            matrix = ifc_placement.get_local_placement(sl.ObjectPlacement)
            z = float(matrix[2][3])

            thickness = None
            if sl.Representation:
                for rep in sl.Representation.Representations:
                    for item in rep.Items:
                        if item.is_a("IfcExtrudedAreaSolid"):
                            thickness = float(item.Depth)

            slab_type = getattr(sl, "PredefinedType", "FLOOR") or "FLOOR"
            result.append({
                "elevation_mm": z,
                "thickness_mm": thickness,
                "type": slab_type,
            })
        except Exception:
            continue

    return result


def _is_roof_level(name: str) -> bool:
    """레벨명이 지붕/옥상 레벨인지 판단.

    Revit IFC에서 지붕 레벨은 건물 최상부를 나타내며,
    별도의 층(story)이 아닌 마지막 층의 높이 계산에만 사용해야 한다.

    지원 키워드:
      한국어: 지붕, 옥상
      영문: roof, RF, RFL, PH, parapet
    """
    if not name:
        return False
    name_lower = name.lower().strip()

    # 키워드 포함 검사
    roof_keywords = ["roof", "지붕", "옥상", "parapet"]
    for kw in roof_keywords:
        if kw in name_lower:
            return True

    # 패턴 매칭: RF, RFL, PH (단독 또는 숫자 접미)
    if re.match(r"^(rf|rfl|ph)\d*$", name_lower):
        return True

    return False


def _filter_above_ground_stories(storey_data: list[tuple[str, float]]
                                 ) -> list[tuple[str, float]]:
    """지상층만 필터링.

    기준:
    1. elevation < -100 → 지하/기초 제외
    2. 기초 관련 키워드 (Fnd, B.O., T.O.) 포함 시 제외
    3. 동일 표고 중복 제거 (숫자F > Roof > 기타 우선)
    """
    foundation_keywords = ["fnd", "b.o.", "t.o.", "foundation", "footing"]
    # Revit 참조 레벨 키워드 (한국어 "레벨" = Level)
    reference_keywords = ["\ub808\ubca8", "level"]  # 레벨, level

    # 1단계: 음수 표고 + 기초 키워드 제거
    filtered = []
    for name, elev in storey_data:
        name_lower = (name or "").lower()
        if elev < -100:
            continue
        if any(kw in name_lower for kw in foundation_keywords):
            continue
        # Revit 참조 레벨 제외 (숫자+F 패턴은 유지)
        if any(kw in name_lower for kw in reference_keywords):
            if not re.match(r"\d+F", name):
                continue
        filtered.append((name, elev))

    # 2단계: 동일 표고 중복 제거
    by_elev: dict[float, list[tuple[str, float]]] = {}
    for name, elev in filtered:
        key = round(elev, 0)
        by_elev.setdefault(key, []).append((name, elev))

    result = []
    for key in sorted(by_elev.keys()):
        candidates = by_elev[key]
        if len(candidates) == 1:
            result.append(candidates[0])
        else:
            # 숫자+F 패턴 우선 (1F, 2F, ...), 그 다음 Roof
            floor_pattern = [c for c in candidates if re.match(r"\d+F", c[0])]
            roof_pattern = [c for c in candidates if "roof" in c[0].lower()]
            if floor_pattern:
                result.append(floor_pattern[0])
            elif roof_pattern:
                result.append(roof_pattern[0])
            else:
                result.append(candidates[0])

    return result


def _ishape_to_dict(profile) -> dict | None:
    """IfcIShapeProfileDef → dict 변환 헬퍼."""
    if not profile.is_a("IfcIShapeProfileDef"):
        return None
    return {
        "h": float(profile.OverallDepth),
        "b": float(profile.OverallWidth),
        "tw": float(profile.WebThickness),
        "tf": float(profile.FlangeThickness),
        "profile_name": profile.ProfileName or "",
    }


def _parse_revit_profile_name(name: str) -> dict | None:
    """Revit 스타일 프로파일명에서 H형강 치수 파싱.

    Revit IFC4 Reference View는 IfcArbitraryClosedProfileDef에
    "H{높이}X{중량(kg/m)}" 형식으로 프로파일명을 저장한다.

    예: "H400X197" → {"h": 400, "weight": 197.0}
        "H250X72.4" → {"h": 250, "weight": 72.4}
    """
    if not name:
        return None
    m = re.match(r"[Hh](\d+)[Xx](\d+\.?\d*)", name.strip())
    if m:
        return {"h": float(m.group(1)), "weight": float(m.group(2))}
    return None


def _resolve_profile_by_weight(h_mm: float, weight_kg: float) -> dict | None:
    """높이(mm)와 중량(kg/m)으로 Supabase DB에서 H형강 단면 조회.

    Returns:
        {"h", "b", "tw", "tf", "profile_name", "db_name"} 또는 None
    """
    try:
        from core.section_3d import _get_supabase
        supabase = _get_supabase()
        result = (
            supabase.schema("ks3502").table("h_beam_sections")
            .select("name, h, b, t1, t2, weight")
            .gte("h", h_mm - 10)
            .lte("h", h_mm + 10)
            .gte("weight", weight_kg - 3.0)
            .lte("weight", weight_kg + 3.0)
            .execute()
        )
        if not result.data:
            return None

        # 중량이 가장 가까운 것 선택
        best = min(
            result.data,
            key=lambda r: abs((r.get("weight") or 0) - weight_kg),
        )
        return {
            "h": float(best["h"]),
            "b": float(best["b"]),
            "tw": float(best.get("t1") or 0),
            "tf": float(best.get("t2") or 0),
            "profile_name": best["name"],
            "db_name": best["name"],
        }
    except Exception:
        return None


def _extract_profile_from_item(item) -> dict | None:
    """representation item에서 프로파일 정보 재귀 추출.

    지원 경로:
      - IfcExtrudedAreaSolid → SweptArea:
        - IfcIShapeProfileDef: 직접 치수 추출
        - IfcArbitraryClosedProfileDef: ProfileName 파싱 → DB 중량 매칭
      - IfcMappedItem → MappingSource → MappedRepresentation → Items (재귀)
      - IfcBooleanClippingResult → FirstOperand (재귀)
    """
    if item.is_a("IfcExtrudedAreaSolid"):
        profile = item.SweptArea
        # 1순위: 파라메트릭 I형강
        result = _ishape_to_dict(profile)
        if result:
            return result
        # 2순위: Revit 폴리라인 프로파일 (IfcArbitraryClosedProfileDef)
        profile_name = getattr(profile, "ProfileName", None)
        if profile_name:
            parsed = _parse_revit_profile_name(profile_name)
            if parsed:
                resolved = _resolve_profile_by_weight(
                    parsed["h"], parsed["weight"]
                )
                if resolved:
                    return resolved

    if item.is_a("IfcMappedItem"):
        mapped_rep = item.MappingSource.MappedRepresentation
        for sub_item in getattr(mapped_rep, "Items", []) or []:
            result = _extract_profile_from_item(sub_item)
            if result:
                return result

    if item.is_a("IfcBooleanClippingResult") or item.is_a("IfcBooleanResult"):
        first = getattr(item, "FirstOperand", None)
        if first:
            return _extract_profile_from_item(first)

    return None


def _get_ishape_from_representation(element) -> dict | None:
    """IfcElement의 Representation에서 IfcIShapeProfileDef 추출.

    지원 경로:
      - 직접: Items → IfcExtrudedAreaSolid → SweptArea
      - IFC4 MappedItem: Items → IfcMappedItem → ... → SweptArea
      - Boolean: Items → IfcBooleanClippingResult → FirstOperand → SweptArea
    """
    if not element.Representation:
        return None
    for rep in element.Representation.Representations:
        for item in rep.Items:
            result = _extract_profile_from_item(item)
            if result:
                return result
    return None


def _get_ishape_from_type(element) -> dict | None:
    """IfcElement의 Type에서 IfcIShapeProfileDef 추출.

    경로: Element → IsTypedBy → IfcColumnType/IfcBeamType
           → RepresentationMaps → Items (재귀 탐색)
    """
    for rel in getattr(element, "IsTypedBy", []):
        etype = rel.RelatingType
        for rep_map in getattr(etype, "RepresentationMaps", []) or []:
            mapped_rep = rep_map.MappedRepresentation
            for item in getattr(mapped_rep, "Items", []) or []:
                result = _extract_profile_from_item(item)
                if result:
                    return result
    return None


def _get_ishape_from_material(element) -> dict | None:
    """IfcElement의 재료 프로파일에서 IfcIShapeProfileDef 추출.

    Revit IFC4 Reference View 주요 경로:
      Element ← IfcRelAssociatesMaterial → IfcMaterialProfileSetUsage
        → ForProfileSet → MaterialProfiles → Profile → IfcIShapeProfileDef
    """
    for rel in getattr(element, "HasAssociations", []):
        if not rel.is_a("IfcRelAssociatesMaterial"):
            continue
        mat = rel.RelatingMaterial

        # IfcMaterialProfileSetUsage → ForProfileSet
        profile_set = None
        if mat.is_a("IfcMaterialProfileSetUsage"):
            profile_set = mat.ForProfileSet
        elif mat.is_a("IfcMaterialProfileSet"):
            profile_set = mat

        if profile_set:
            for mp in getattr(profile_set, "MaterialProfiles", []) or []:
                profile = getattr(mp, "Profile", None)
                if profile and profile.is_a("IfcIShapeProfileDef"):
                    return _ishape_to_dict(profile)

    return None


def _get_element_profile(element) -> dict | None:
    """IfcElement에서 I형강 프로파일 추출 (모든 경로 시도).

    시도 순서:
      1. Representation (직접/MappedItem/Boolean)
      2. Type (RepresentationMaps)
      3. Material (IfcMaterialProfileSetUsage) ← IFC4 주요 경로
    """
    profile = _get_ishape_from_representation(element)
    if profile:
        return profile
    profile = _get_ishape_from_type(element)
    if profile:
        return profile
    profile = _get_ishape_from_material(element)
    if profile:
        return profile
    return None


def _extract_section_profiles(ifc_file) -> dict:
    """IfcColumn/IfcBeam에서 I형강 단면 프로파일 추출.

    Returns:
        {"columns": [{"h", "b", "tw", "tf", "profile_name"}, ...],
         "beams":   [{"h", "b", "tw", "tf", "profile_name"}, ...]}
    """
    columns_profiles = []
    beams_profiles = []

    # 기둥 프로파일
    for col in ifc_file.by_type("IfcColumn"):
        profile = _get_element_profile(col)
        if profile:
            columns_profiles.append(profile)

    # 보 프로파일
    for beam in ifc_file.by_type("IfcBeam"):
        profile = _get_element_profile(beam)
        if profile:
            beams_profiles.append(profile)

    return {"columns": columns_profiles, "beams": beams_profiles}


def _normalize_material_name(raw_name: str) -> str | None:
    """IFC 재료명을 표준 한국 강재 등급으로 정규화.

    "Steel SS275", "강재 SS275", "SS 275" → "SS275"
    """
    if not raw_name:
        return None
    # SS/SM/SN + 숫자 패턴
    m = re.search(r"(SS|SM|SN|SHN|HSA)\s*(\d{3,4})", raw_name, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    # S + 숫자 (유럽 규격 S275 등)
    m2 = re.search(r"\bS\s*(\d{3})\b", raw_name)
    if m2:
        grade = m2.group(1)
        return f"SS{grade}"
    return None


def _extract_materials(ifc_file) -> list[str]:
    """IfcRelAssociatesMaterial에서 구조 재료명 추출.

    Returns:
        정규화된 재료명 리스트 (중복 제거). 예: ["SS275"]
    """
    material_names = set()

    for rel in ifc_file.by_type("IfcRelAssociatesMaterial"):
        mat = rel.RelatingMaterial
        # IfcMaterial 직접
        if mat.is_a("IfcMaterial"):
            normalized = _normalize_material_name(mat.Name or "")
            if normalized:
                material_names.add(normalized)
        # IfcMaterialLayerSetUsage → IfcMaterialLayerSet → IfcMaterialLayer
        elif mat.is_a("IfcMaterialLayerSetUsage"):
            layer_set = mat.ForLayerSet
            for layer in layer_set.MaterialLayers:
                if layer.Material:
                    normalized = _normalize_material_name(layer.Material.Name or "")
                    if normalized:
                        material_names.add(normalized)
        # IfcMaterialLayerSet
        elif mat.is_a("IfcMaterialLayerSet"):
            for layer in mat.MaterialLayers:
                if layer.Material:
                    normalized = _normalize_material_name(layer.Material.Name or "")
                    if normalized:
                        material_names.add(normalized)
        # IfcMaterialProfileSetUsage → IfcMaterialProfileSet
        elif mat.is_a("IfcMaterialProfileSetUsage"):
            prof_set = mat.ForProfileSet
            for mp in prof_set.MaterialProfiles or []:
                if mp.Material:
                    normalized = _normalize_material_name(mp.Material.Name or "")
                    if normalized:
                        material_names.add(normalized)

    return sorted(material_names)


def _most_common_profile(profiles: list[dict]) -> dict | None:
    """프로파일 리스트에서 가장 많이 등장하는 h×b 조합 반환."""
    if not profiles:
        return None
    freq: dict[tuple, list[dict]] = {}
    for p in profiles:
        key = (round(p["h"]), round(p["b"]))
        freq.setdefault(key, []).append(p)
    # 가장 빈도 높은 그룹
    best_group = max(freq.values(), key=len)
    # 그룹 내 평균 tw, tf
    return {
        "h": best_group[0]["h"],
        "b": best_group[0]["b"],
        "tw": round(sum(p["tw"] for p in best_group) / len(best_group), 1),
        "tf": round(sum(p["tf"] for p in best_group) / len(best_group), 1),
    }


def parse_ifc(ifc_path: str, tolerance: float = 200.0) -> dict:
    """IFC 파일에서 건물 기하 정보 추출.

    골조 구조(IfcColumn)와 벽식 구조(IfcWall) 모두 지원.
    기둥이 있으면 기둥 기반, 없으면 벽체 기반으로 그리드를 감지한다.
    """
    if not HAS_IFC:
        raise ImportError(
            "ifcopenshell 패키지가 필요합니다. "
            "설치: pip install ifcopenshell"
        )

    ifc_file = ifcopenshell.open(ifc_path)
    warnings = []

    # ── 1. 슬래브 정보 ──
    slab_info = _get_slab_info(ifc_file)

    # ── 2. 층 정보 ──
    raw_storey_data = _get_storey_elevations(ifc_file)
    storey_data = _filter_above_ground_stories(raw_storey_data)

    # 지붕 레벨 분리: 마지막 레벨이 지붕이면 높이 계산에만 사용
    roof_elevation = None
    if len(storey_data) >= 2 and _is_roof_level(storey_data[-1][0]):
        roof_name, roof_elev = storey_data[-1]
        roof_elevation = roof_elev
        storey_data = storey_data[:-1]
        warnings.append(
            f"'{roof_name}' 레벨(EL {roof_elev/1000:.1f}m)을 "
            f"지붕으로 인식하여 최상층 높이 계산에 사용"
        )

    stories = []
    for i in range(len(storey_data)):
        name, elev = storey_data[i]
        if i + 1 < len(storey_data):
            height = (storey_data[i + 1][1] - elev) / 1000.0
        elif roof_elevation is not None:
            # 지붕 표고로 마지막 층 높이 계산
            height = (roof_elevation - elev) / 1000.0
        else:
            if stories:
                height = stories[-1]["height"]
            else:
                height = 3.0
            warnings.append(f"최상층 '{name}' 높이는 추정값 ({height}m)")

        # 슬래브 두께 매칭
        slab_thickness = None
        for s in slab_info:
            if abs(s["elevation_mm"] - elev) < tolerance:
                slab_thickness = s["thickness_mm"]
                break

        stories.append({
            "name": name,
            "height": round(height, 3),
            "elevation": round(elev / 1000.0, 3),
            "slab_thickness_mm": slab_thickness,
        })

    # ── 2.5. 기둥 없는 최상층 → 옥상 레벨로 변환 ──
    # IFC 최상층에 기둥이 0개면 옥상 슬래브 레벨로 판단
    if len(stories) >= 2 and roof_elevation is None:
        try:
            _storey_cols = _get_per_storey_columns(ifc_file, tolerance)
            last_story_name = stories[-1]["name"]
            last_story_cols = _storey_cols.get(last_story_name, [])
            if len(last_story_cols) == 0:
                # 최상층에 기둥 없음 → 옥상 레벨
                roof_story = stories.pop()
                # 이전 최상층의 높이를 옥상 표고로 재계산
                if stories:
                    prev_elev = stories[-1]["elevation"]
                    stories[-1]["height"] = round(roof_story["elevation"] - prev_elev, 3)
                warnings.append(
                    f"'{roof_story['name']}' 레벨에 기둥이 없어 옥상으로 판단하여 "
                    f"층 수에서 제외 (최상층 높이를 "
                    f"{stories[-1]['height'] if stories else '?'}m으로 재계산)"
                )
        except Exception:
            pass

    # ── 3. 그리드 감지 ──
    col_positions = _get_column_positions(ifc_file)
    grid_source = "column"
    wall_info = []
    num_walls = (len(ifc_file.by_type("IfcWallStandardCase"))
                 + len(ifc_file.by_type("IfcWall")))

    if col_positions:
        x_coords = [p[0] for p in col_positions]
        y_coords = [p[1] for p in col_positions]
    else:
        grid_source = "wall"
        base_elev = storey_data[0][1] if storey_data else 0.0
        x_coords, y_coords, wall_info = _get_wall_grid_coords(
            ifc_file, base_elevation=base_elev, tolerance=tolerance
        )
        if wall_info:
            warnings.append(
                f"IfcColumn 없음 - IfcWall {len(wall_info)}개에서 그리드 추론"
            )
        else:
            warnings.append("IfcColumn, IfcWall 모두 찾을 수 없습니다.")
            return {
                "stories": stories,
                "bays_x": [], "bays_y": [],
                "grid_x": [], "grid_y": [],
                "num_columns": 0, "num_walls": num_walls,
                "grid_source": grid_source,
                "slabs": slab_info, "walls": [],
                "detected_sections": {"column": None, "beam": None},
                "detected_material": None,
                "section_substitutions": [],
                "section_profiles": {"columns": [], "beams": []},
                "warnings": warnings,
            }

    # 그리드 감지
    bays_x = _detect_grid(x_coords, tolerance)
    bays_y = _detect_grid(y_coords, tolerance)

    grid_x_centers = _cluster_coords(x_coords, tolerance)
    grid_y_centers = _cluster_coords(y_coords, tolerance)
    grid_x = [round(c / 1000.0, 3) for c in grid_x_centers]
    grid_y = [round(c / 1000.0, 3) for c in grid_y_centers]

    # 비정형 검사 — 경간 편차 경고
    if bays_x:
        mean_bx = sum(bays_x) / len(bays_x)
        max_dev = max(abs(b - mean_bx) for b in bays_x)
        if max_dev > mean_bx * 0.3:
            warnings.append(
                f"X방향 경간 폭 편차가 큽니다: {bays_x}. "
                f"비정형 그리드일 수 있습니다."
            )
    if bays_y:
        mean_by = sum(bays_y) / len(bays_y)
        max_dev = max(abs(b - mean_by) for b in bays_y)
        if max_dev > mean_by * 0.3:
            warnings.append(
                f"Y방향 경간 폭 편차가 큽니다: {bays_y}. "
                f"비정형 그리드일 수 있습니다."
            )

    # 비정형 검사 — 점유 격자 기반 zone 감지
    detected_zones = None
    if grid_source == "column" and col_positions and len(grid_x_centers) >= 2 and len(grid_y_centers) >= 2:
        detected_zones = _detect_zones_from_occupancy(
            col_positions, grid_x_centers, grid_y_centers, tolerance
        )
        if detected_zones:
            warnings.append(
                f"비정형 평면 감지: {len(detected_zones)}개 존 "
                f"({', '.join(z['id'] for z in detected_zones)}). "
                f"일부 격자 교차점에 기둥이 없습니다."
            )

    # 층별 기둥 분석 — zone의 story_to 자동 설정 + 내부 갭 경고
    if detected_zones and grid_source == "column":
        try:
            storey_col_positions = _get_per_storey_columns(ifc_file, tolerance)
            if storey_col_positions:
                _refine_zones_by_storey(
                    detected_zones, storey_col_positions, stories,
                    grid_x_centers, grid_y_centers, tolerance, warnings,
                )
        except Exception:
            pass

    # ── 4. 단면 프로파일 추출 ──
    section_profiles = _extract_section_profiles(ifc_file)
    detected_sections = {"column": None, "beam": None}
    section_substitutions = []

    num_ifc_columns = len(ifc_file.by_type("IfcColumn"))
    num_ifc_beams = len(ifc_file.by_type("IfcBeam"))

    col_profile = _most_common_profile(section_profiles["columns"])
    beam_profile = _most_common_profile(section_profiles["beams"])

    # 프로파일 추출 실패 경고
    if num_ifc_columns > 0 and not section_profiles["columns"]:
        warnings.append(
            f"IfcColumn {num_ifc_columns}개 존재하나 "
            f"I형강 프로파일(IfcIShapeProfileDef)을 추출하지 못했습니다. "
            f"기본 단면이 사용됩니다."
        )
    if num_ifc_beams > 0 and not section_profiles["beams"]:
        warnings.append(
            f"IfcBeam {num_ifc_beams}개 존재하나 "
            f"I형강 프로파일(IfcIShapeProfileDef)을 추출하지 못했습니다. "
            f"기본 단면이 사용됩니다."
        )

    for member_type, profile in [("column", col_profile), ("beam", beam_profile)]:
        if not profile:
            continue
        # db_name이 있으면 이미 DB에서 해결됨 (Revit 중량 매칭)
        if profile.get("db_name"):
            detected_sections[member_type] = profile["db_name"]
            continue
        # 그 외: h, b, tw, tf로 DB 조회
        try:
            from core.section_3d import find_section_by_dims
            name, sub_info = find_section_by_dims(
                profile["h"], profile["b"],
                profile["tw"], profile["tf"],
            )
            if name:
                detected_sections[member_type] = name
                if sub_info:
                    sub_info["member_type"] = member_type
                    section_substitutions.append(sub_info)
        except Exception:
            pass

    # ── 5. 재료 추출 ──
    materials = _extract_materials(ifc_file)
    detected_material = materials[0] if materials else None

    result = {
        "stories": stories,
        "bays_x": bays_x,
        "bays_y": bays_y,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "num_columns": len(col_positions),
        "num_walls": num_walls,
        "grid_source": grid_source,
        "detected_zones": detected_zones,  # None(정형) or list of zone dicts
        "slabs": slab_info,
        "walls": [
            {
                "tag": w["tag"],
                "x_m": round(w["x"] / 1000, 3),
                "y_m": round(w["y"] / 1000, 3),
                "orientation": w["orientation"],
                "length_m": round(w["length"] / 1000, 3) if w["length"] else None,
                "thickness_mm": round(w["thickness"], 1) if w["thickness"] else None,
                "height_m": round(w["height"] / 1000, 3) if w["height"] else None,
            }
            for w in wall_info[:50]
        ],
        "detected_sections": detected_sections,
        "detected_material": detected_material,
        "section_substitutions": section_substitutions,
        "section_profiles": {
            "columns": section_profiles["columns"][:10],
            "beams": section_profiles["beams"][:10],
        },
        "warnings": warnings,
    }

    if col_positions:
        result["column_positions"] = [
            {"x_m": round(p[0]/1000, 3), "y_m": round(p[1]/1000, 3),
             "z_m": round(p[2]/1000, 3)}
            for p in col_positions[:50]
        ]

    return result
