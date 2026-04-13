"""V2 건물 모델 중간 표현 (Node-Element 기반).

Midas Gen 방식과 동일하게 노드를 먼저 배치하고,
요소(i,j)를 연결하는 자유 그래프 기반 IR.

V1(BuildingModel)과 달리 격자(bays_x/y) 기반이 아니므로:
- 비정형 건물 (경사 부재, 불규칙 평면) 자연스럽게 표현
- 요소별 개별 릴리즈, 단면, 재료 지정 가능
- 사용자가 노드/요소를 직접 편집 가능

Usage:
    # IFC에서 생성
    model = StructuralModel.from_ifc("building.ifc")

    # JSON에서 생성
    model = StructuralModel.from_json(config)

    # 사용자 편집
    model.add_node(x=10.0, y=0.0, z=4.0)
    model.add_element(node_i=1, node_j=15, elem_type="brace", section="H-200x200")
    model.remove_element(42)

    # 해석
    result = analyze_from_model(model, load_cases, load_combinations)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


# ──────────────────────────────────────────────────────────────
#  열거형 / 상수
# ──────────────────────────────────────────────────────────────

class ElementType(str, Enum):
    """구조 요소 유형."""
    COLUMN = "column"
    BEAM = "beam"
    BRACE = "brace"


class SupportType(str, Enum):
    """경계 조건 유형."""
    FIXED = "fixed"       # 6-DOF 구속
    PINNED = "pinned"     # 병진 3-DOF 구속, 회전 자유
    ROLLER_X = "roller_x" # Y,Z 구속, X 자유
    ROLLER_Y = "roller_y" # X,Z 구속, Y 자유
    FREE = "free"         # 구속 없음


class ReleaseType(str, Enum):
    """부재 릴리즈 유형."""
    MOMENT_Y = "moment_y"    # 강축 모멘트 릴리즈
    MOMENT_Z = "moment_z"    # 약축 모멘트 릴리즈
    MOMENT_YZ = "moment_yz"  # 양축 모멘트 릴리즈
    ALL_MOMENT = "all"       # 모멘트 + 비틀림


# ──────────────────────────────────────────────────────────────
#  노드
# ──────────────────────────────────────────────────────────────

@dataclass
class StructuralNode:
    """구조 해석용 노드.

    Attributes:
        id: 고유 노드 ID (1-based)
        x, y, z: 글로벌 좌표 (m)
        story: 층 번호 (0=기초, 1=1층, ...) — Z 클러스터링으로 자동감지
        support: 경계조건 (None이면 자유 노드)
        mass: 추가 집중 질량 (ton) — 없으면 None
    """
    id: int
    x: float
    y: float
    z: float
    story: int | None = None
    support: SupportType | None = None
    mass: float | None = None

    @property
    def coords(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: "StructuralNode") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


# ──────────────────────────────────────────────────────────────
#  요소
# ──────────────────────────────────────────────────────────────

@dataclass
class StructuralElement:
    """구조 해석용 요소 (line element).

    Attributes:
        id: 고유 요소 ID (1-based)
        node_i: 시작 노드 ID (i-end)
        node_j: 끝 노드 ID (j-end)
        elem_type: 요소 유형 (column, beam, brace)
        section: 단면 이름 (예: "H-400x200", "H-300x300")
        material: 재료 이름 (예: "SS275", "SM490")
        release_i: i-end 릴리즈 (None이면 강접)
        release_j: j-end 릴리즈 (None이면 강접)
        beta_angle: 단면 회전각 (degree) — 기본 0
    """
    id: int
    node_i: int
    node_j: int
    elem_type: ElementType
    section: str = "H-400x200"
    material: str = "SS275"
    release_i: ReleaseType | None = None
    release_j: ReleaseType | None = None
    beta_angle: float = 0.0

    @property
    def is_vertical(self) -> bool:
        """수직 요소인지 (기둥 판별용)."""
        return self.elem_type == ElementType.COLUMN

    def length(self, nodes: dict[int, StructuralNode]) -> float:
        """요소 길이 계산 (m)."""
        ni = nodes[self.node_i]
        nj = nodes[self.node_j]
        return ni.distance_to(nj)

    def direction_vector(self, nodes: dict[int, StructuralNode]) -> tuple[float, float, float]:
        """i→j 방향 단위벡터."""
        ni = nodes[self.node_i]
        nj = nodes[self.node_j]
        L = self.length(nodes)
        if L < 1e-9:
            return (0.0, 0.0, 0.0)
        return (
            (nj.x - ni.x) / L,
            (nj.y - ni.y) / L,
            (nj.z - ni.z) / L,
        )


# ──────────────────────────────────────────────────────────────
#  IFC 검증 결과
# ──────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """IFC 검증 이슈 하나."""
    severity: Severity
    code: str        # "IFC_NO_COLUMN", "IFC_NO_PROFILE", ...
    message: str     # 사용자에게 표시할 메시지
    element_ids: list[int] = field(default_factory=list)
    default_value: str | None = None  # 기본값 제안 (WARNING 시)


@dataclass
class IFCValidationResult:
    """IFC 파일 검증 결과.

    해석 전에 사용자에게 피드백을 제공하기 위한 구조체.
    """
    is_valid: bool = True              # CRITICAL 이슈 없으면 True
    issues: list[ValidationIssue] = field(default_factory=list)

    # 추출 통계
    extracted_nodes: int = 0
    extracted_elements: int = 0
    failed_elements: int = 0

    # 사용자 입력 필요 항목
    needs_user_input: list[ValidationIssue] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def info_items(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    def summary_text(self) -> str:
        """사용자에게 보여줄 요약 텍스트."""
        lines = []

        if self.is_valid:
            lines.append(
                f"IFC 추출 완료: 노드 {self.extracted_nodes}개, "
                f"요소 {self.extracted_elements}개"
            )
        else:
            lines.append("IFC 검증 실패 — 해석을 진행할 수 없습니다.")

        if self.failed_elements > 0:
            lines.append(f"  변환 실패: {self.failed_elements}개 부재")

        for issue in self.critical_issues:
            lines.append(f"  [CRITICAL] {issue.message}")

        for issue in self.warnings:
            msg = f"  [WARNING] {issue.message}"
            if issue.default_value:
                msg += f" (기본값: {issue.default_value})"
            lines.append(msg)

        for issue in self.info_items:
            lines.append(f"  [INFO] {issue.message}")

        if self.needs_user_input:
            lines.append("\n사용자 확인 필요:")
            for q in self.needs_user_input:
                lines.append(f"  - {q.message}")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  StructuralModel — 핵심 IR
# ──────────────────────────────────────────────────────────────

@dataclass
class StructuralModel:
    """V2 건물 모델 중간 표현 (Node-Element 기반).

    노드와 요소의 자유 그래프로 구조물을 표현한다.
    격자 제약 없이 비정형 구조물도 표현 가능.
    """

    # ── 기하 (자유 노드-요소 그래프) ──
    nodes: dict[int, StructuralNode] = field(default_factory=dict)
    elements: dict[int, StructuralElement] = field(default_factory=dict)

    # ── 층 정보 (Z 클러스터링 자동감지 + 수동 수정) ──
    story_elevations: list[float] = field(default_factory=list)  # [0.0, 4.0, 7.5, ...]
    story_usages: dict[int, str] = field(default_factory=dict)   # {1: "retail", 2: "office"}
    story_slab_thickness: dict[int, float] = field(default_factory=dict)  # {1: 0.15, 2: 0.15}
    story_dead_load_finish: dict[int, float] = field(default_factory=dict)  # {1: 1.5, 2: 1.0}

    # ── 환경/설계 메타 (V1과 동일, 하중생성용) ──
    region: str = ""
    site_class: str = "S3"
    importance: str = "II"
    importance_factor: float = 1.0
    seismic_system: str = "ordinary_moment_frame"
    seismic_direction: str = "both"
    exposure_category: str = "B"

    # ── 해석 옵션 ──
    num_elements_per_member: int = 4
    rigid_diaphragm: bool = False
    geometric_nonlinearity: str = "linear"  # "linear" or "pdelta"

    # ── 검증 결과 (IFC에서 생성 시 포함) ──
    validation: IFCValidationResult | None = None

    # ── 내부 카운터 ──
    _next_node_id: int = field(default=1, repr=False)
    _next_elem_id: int = field(default=1, repr=False)

    # ──────────────────────────────────────────────
    #  노드 CRUD
    # ──────────────────────────────────────────────

    def add_node(
        self,
        x: float,
        y: float,
        z: float,
        support: SupportType | str | None = None,
        story: int | None = None,
        node_id: int | None = None,
    ) -> StructuralNode:
        """노드 추가. node_id 미지정 시 자동 부여."""
        if isinstance(support, str) and support:
            support = SupportType(support)

        nid = node_id if node_id is not None else self._next_node_id
        if nid in self.nodes:
            raise ValueError(f"노드 ID {nid} 이미 존재합니다.")

        node = StructuralNode(
            id=nid, x=x, y=y, z=z,
            story=story, support=support,
        )
        self.nodes[nid] = node
        self._next_node_id = max(self._next_node_id, nid + 1)
        return node

    def remove_node(self, node_id: int) -> None:
        """노드 삭제. 연결된 요소가 있으면 에러."""
        connected = [
            e.id for e in self.elements.values()
            if e.node_i == node_id or e.node_j == node_id
        ]
        if connected:
            raise ValueError(
                f"노드 {node_id}에 연결된 요소가 있습니다: {connected}. "
                "요소를 먼저 삭제하세요."
            )
        del self.nodes[node_id]

    def move_node(self, node_id: int, x: float, y: float, z: float) -> None:
        """노드 좌표 변경."""
        node = self.nodes[node_id]
        node.x = x
        node.y = y
        node.z = z

    def merge_close_nodes(self, tolerance: float = 0.01) -> dict[int, int]:
        """근접 노드 병합 (tolerance in m).

        Returns:
            old_id → new_id 매핑 (병합된 노드만)
        """
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: (n.z, n.x, n.y))
        merge_map: dict[int, int] = {}  # old → keep

        for i, ni in enumerate(sorted_nodes):
            if ni.id in merge_map:
                continue
            for j in range(i + 1, len(sorted_nodes)):
                nj = sorted_nodes[j]
                if nj.id in merge_map:
                    continue
                # Z 차이가 tolerance보다 크면 이후 노드는 더 클 수 밖에 없음
                if nj.z - ni.z > tolerance:
                    break
                if ni.distance_to(nj) <= tolerance:
                    merge_map[nj.id] = ni.id
                    # support 병합: 둘 중 하나가 support이면 유지
                    if nj.support and not ni.support:
                        ni.support = nj.support

        if not merge_map:
            return merge_map

        # 요소의 노드 참조 업데이트
        for elem in self.elements.values():
            if elem.node_i in merge_map:
                elem.node_i = merge_map[elem.node_i]
            if elem.node_j in merge_map:
                elem.node_j = merge_map[elem.node_j]

        # 병합된 노드 삭제
        for old_id in merge_map:
            del self.nodes[old_id]

        return merge_map

    # ──────────────────────────────────────────────
    #  요소 CRUD
    # ──────────────────────────────────────────────

    def add_element(
        self,
        node_i: int,
        node_j: int,
        elem_type: ElementType | str = ElementType.BEAM,
        section: str = "H-400x200",
        material: str = "SS275",
        release_i: ReleaseType | str | None = None,
        release_j: ReleaseType | str | None = None,
        beta_angle: float = 0.0,
        elem_id: int | None = None,
    ) -> StructuralElement:
        """요소 추가."""
        if node_i not in self.nodes:
            raise ValueError(f"노드 {node_i}이 존재하지 않습니다.")
        if node_j not in self.nodes:
            raise ValueError(f"노드 {node_j}이 존재하지 않습니다.")
        if node_i == node_j:
            raise ValueError("시작/끝 노드가 같을 수 없습니다.")

        if isinstance(elem_type, str):
            elem_type = ElementType(elem_type)
        if isinstance(release_i, str) and release_i:
            release_i = ReleaseType(release_i)
        if isinstance(release_j, str) and release_j:
            release_j = ReleaseType(release_j)

        eid = elem_id if elem_id is not None else self._next_elem_id
        if eid in self.elements:
            raise ValueError(f"요소 ID {eid} 이미 존재합니다.")

        elem = StructuralElement(
            id=eid, node_i=node_i, node_j=node_j,
            elem_type=elem_type, section=section, material=material,
            release_i=release_i, release_j=release_j,
            beta_angle=beta_angle,
        )
        self.elements[eid] = elem
        self._next_elem_id = max(self._next_elem_id, eid + 1)
        return elem

    def remove_element(self, elem_id: int) -> None:
        """요소 삭제."""
        if elem_id not in self.elements:
            raise ValueError(f"요소 ID {elem_id}이 존재하지 않습니다.")
        del self.elements[elem_id]

    def update_element(self, elem_id: int, **kwargs) -> None:
        """요소 속성 수정.

        Example:
            model.update_element(5, section="H-500x200", release_i="moment_y")
        """
        elem = self.elements[elem_id]
        for key, val in kwargs.items():
            if key == "elem_type" and isinstance(val, str):
                val = ElementType(val)
            elif key in ("release_i", "release_j") and isinstance(val, str) and val:
                val = ReleaseType(val)
            if hasattr(elem, key):
                setattr(elem, key, val)
            else:
                raise ValueError(f"StructuralElement에 '{key}' 속성이 없습니다.")

    # ──────────────────────────────────────────────
    #  층 자동 감지
    # ──────────────────────────────────────────────

    def detect_stories(self, tolerance: float = 0.3) -> list[float]:
        """Z 좌표 클러스터링으로 층 자동 감지.

        Args:
            tolerance: Z 좌표 그룹핑 허용 오차 (m)

        Returns:
            감지된 층 표고 리스트 (오름차순)
        """
        z_values = sorted(set(round(n.z, 3) for n in self.nodes.values()))
        if not z_values:
            return []

        # 근접 Z값 클러스터링
        clusters: list[list[float]] = [[z_values[0]]]
        for z in z_values[1:]:
            if z - clusters[-1][-1] <= tolerance:
                clusters[-1].append(z)
            else:
                clusters.append([z])

        # 클러스터 평균 → 층 표고
        elevations = [sum(c) / len(c) for c in clusters]

        self.story_elevations = elevations

        # 각 노드에 story 번호 할당 (0=기초/최저, 1=1층, ...)
        for node in self.nodes.values():
            best_story = 0
            best_dist = float("inf")
            for i, elev in enumerate(elevations):
                dist = abs(node.z - elev)
                if dist < best_dist:
                    best_dist = dist
                    best_story = i
            node.story = best_story

        return elevations

    # ──────────────────────────────────────────────
    #  노드 병합 (A-1)
    # ──────────────────────────────────────────────

    def merge_nearby_nodes(self, tolerance: float = 0.0) -> int:
        """근접 노드를 병합하여 보-기둥 접합을 보장.

        Args:
            tolerance: 병합 거리 (m). 0이면 자동 계산 (최대 단면 높이 기반).

        Returns:
            병합된 노드 수.
        """
        if tolerance <= 0:
            # 적응형: 사용 단면의 최대 높이 × 0.8
            # H-400 → 0.32m, H-250 → 0.20m, H-1000 → 0.80m
            max_h = 0.0
            for elem in self.elements.values():
                sec_name = elem.section
                # 단면 이름에서 높이 추출 (H-400x200 → 400mm)
                parts = sec_name.replace("H-", "").replace("h-", "").split("x")
                try:
                    h_mm = float(parts[0])
                    if h_mm > max_h:
                        max_h = h_mm
                except (ValueError, IndexError):
                    pass
            if max_h > 0:
                tolerance = max_h / 1000.0 * 0.8  # mm → m, 80% of section height
            else:
                tolerance = 0.30  # fallback

            # 최소 부재 길이의 절반을 초과하지 않도록 안전장치
            min_len = float('inf')
            for elem in self.elements.values():
                ni, nj = self.nodes.get(elem.node_i), self.nodes.get(elem.node_j)
                if ni and nj:
                    L = math.sqrt((nj.x-ni.x)**2 + (nj.y-ni.y)**2 + (nj.z-ni.z)**2)
                    if L > 0.01 and L < min_len:
                        min_len = L
            if min_len < float('inf'):
                tolerance = min(tolerance, min_len * 0.4)

        tol_sq = tolerance ** 2
        # 지점 노드 우선, ID 순으로 정렬
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: (0 if n.support else 1, n.id)
        )

        merge_map: dict[int, int] = {}  # old_id → canonical_id
        absorbed: set[int] = set()

        for i, ni in enumerate(sorted_nodes):
            if ni.id in absorbed:
                continue
            for nj in sorted_nodes[i + 1:]:
                if nj.id in absorbed:
                    continue
                dx = ni.x - nj.x
                dy = ni.y - nj.y
                dz = ni.z - nj.z
                if dx * dx + dy * dy + dz * dz < tol_sq:
                    merge_map[nj.id] = ni.id
                    absorbed.add(nj.id)
                    # 속성 전파
                    if nj.support and not ni.support:
                        ni.support = nj.support
                    if nj.story is not None and ni.story is None:
                        ni.story = nj.story

        if not merge_map:
            return 0

        # 요소 참조 갱신
        for elem in self.elements.values():
            if elem.node_i in merge_map:
                elem.node_i = merge_map[elem.node_i]
            if elem.node_j in merge_map:
                elem.node_j = merge_map[elem.node_j]

        # 0-길이 요소 제거
        to_remove = [eid for eid, e in self.elements.items() if e.node_i == e.node_j]
        for eid in to_remove:
            del self.elements[eid]

        # 중복 요소 제거
        seen: set[tuple[int, int]] = set()
        dup_remove = []
        for eid, e in self.elements.items():
            key = (min(e.node_i, e.node_j), max(e.node_i, e.node_j))
            if key in seen:
                dup_remove.append(eid)
            else:
                seen.add(key)
        for eid in dup_remove:
            del self.elements[eid]

        # 흡수된 노드 제거
        for nid in absorbed:
            del self.nodes[nid]

        return len(merge_map)

    # ──────────────────────────────────────────────
    #  연결성 검증 (A-2)
    # ──────────────────────────────────────────────

    def validate_connectivity(self) -> list[dict]:
        """노드-요소 연결성을 검증하여 경고 목록을 반환.

        검출 항목:
        - orphan: 어떤 요소에도 연결되지 않은 노드 (지점 아닌)
        - dangling: 하나의 요소에만 연결 + 지점 아닌 노드 (무지지 자유단)
        - disconnected: 요소 그래프가 여러 연결 성분으로 분리됨

        Returns:
            list of {type, severity, node_ids/component_ids, message}
        """
        warnings: list[dict] = []

        # 노드별 연결 요소 수
        conn_count: dict[int, int] = {nid: 0 for nid in self.nodes}
        for e in self.elements.values():
            if e.node_i in conn_count:
                conn_count[e.node_i] += 1
            if e.node_j in conn_count:
                conn_count[e.node_j] += 1

        # Orphan nodes (0 connections, no support)
        orphans = [
            nid for nid, cnt in conn_count.items()
            if cnt == 0 and not self.nodes[nid].support
        ]
        if orphans:
            warnings.append({
                "type": "orphan_node",
                "severity": "warning",
                "node_ids": orphans,
                "message": f"{len(orphans)}개 고립 노드 (요소 미연결): N{orphans[:5]}",
            })

        # Dangling nodes (1 connection, no support)
        dangling = [
            nid for nid, cnt in conn_count.items()
            if cnt == 1 and not self.nodes[nid].support
        ]
        if dangling:
            warnings.append({
                "type": "dangling_node",
                "severity": "info",
                "node_ids": dangling,
                "message": f"{len(dangling)}개 자유단 노드 (단일 요소 연결, 무지지): N{dangling[:5]}",
            })

        # Connected components (Union-Find)
        if self.nodes and self.elements:
            parent: dict[int, int] = {nid: nid for nid in self.nodes}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for e in self.elements.values():
                if e.node_i in parent and e.node_j in parent:
                    union(e.node_i, e.node_j)

            components: dict[int, list[int]] = {}
            for nid in self.nodes:
                root = find(nid)
                components.setdefault(root, []).append(nid)

            if len(components) > 1:
                sizes = sorted(
                    [(root, len(nids)) for root, nids in components.items()],
                    key=lambda x: -x[1],
                )
                warnings.append({
                    "type": "disconnected",
                    "severity": "error",
                    "components": [
                        {"root": root, "size": sz} for root, sz in sizes
                    ],
                    "message": (
                        f"모델이 {len(components)}개 분리 성분으로 구성됨: "
                        f"크기 {[sz for _, sz in sizes]}"
                    ),
                })

        return warnings

    # ──────────────────────────────────────────────
    #  요소 자동 분류
    # ──────────────────────────────────────────────

    def classify_elements(self, angle_threshold: float = 30.0) -> None:
        """요소의 elem_type을 기하학적으로 자동 분류.

        - 수직에 가까우면 (Z축과 angle < threshold) → column
        - 수평에 가까우면 → beam
        - 그 외 → brace

        Args:
            angle_threshold: 수직/수평 판별 각도 (degree)
        """
        for elem in self.elements.values():
            dx, dy, dz = elem.direction_vector(self.nodes)
            vert_angle = math.degrees(math.acos(min(abs(dz), 1.0)))

            if vert_angle <= angle_threshold:
                elem.elem_type = ElementType.COLUMN
            elif vert_angle >= (90.0 - angle_threshold):
                elem.elem_type = ElementType.BEAM
            else:
                elem.elem_type = ElementType.BRACE

    # ──────────────────────────────────────────────
    #  쿼리 헬퍼
    # ──────────────────────────────────────────────

    @property
    def num_stories(self) -> int:
        """층 수 (기초 제외)."""
        if not self.story_elevations:
            return 0
        return len(self.story_elevations) - 1  # 기초(0) 제외

    @property
    def total_height(self) -> float:
        if not self.story_elevations:
            return 0.0
        return max(self.story_elevations) - min(self.story_elevations)

    @property
    def story_heights(self) -> list[float]:
        """각 층의 층고 리스트 (1층~최상층)."""
        if len(self.story_elevations) < 2:
            return []
        return [
            round(self.story_elevations[i + 1] - self.story_elevations[i], 4)
            for i in range(len(self.story_elevations) - 1)
        ]

    def nodes_at_story(self, story: int) -> list[StructuralNode]:
        """특정 층의 노드 목록."""
        return [n for n in self.nodes.values() if n.story == story]

    def base_nodes(self) -> list[StructuralNode]:
        """기초(최하층) 노드 목록."""
        if not self.story_elevations:
            return []
        return self.nodes_at_story(0)

    def elements_by_type(self, elem_type: ElementType) -> list[StructuralElement]:
        """특정 유형의 요소 목록."""
        return [e for e in self.elements.values() if e.elem_type == elem_type]

    def elements_at_story(self, story: int) -> list[StructuralElement]:
        """특정 층의 요소 목록 (하단 노드 기준)."""
        story_node_ids = {n.id for n in self.nodes_at_story(story)}
        return [
            e for e in self.elements.values()
            if e.node_i in story_node_ids or e.node_j in story_node_ids
        ]

    def connected_elements(self, node_id: int) -> list[StructuralElement]:
        """특정 노드에 연결된 요소 목록."""
        return [
            e for e in self.elements.values()
            if e.node_i == node_id or e.node_j == node_id
        ]

    def floor_area_at_story(self, story: int) -> float:
        """특정 층의 바닥 면적 (m²).

        해당 층 노드들의 convex hull 면적으로 근사.
        보다 정확한 면적은 슬래브 요소 추가 시 개선.
        """
        nodes = self.nodes_at_story(story)
        if len(nodes) < 3:
            return 0.0

        # 2D convex hull (Shoelace formula)
        points = [(n.x, n.y) for n in nodes]
        return self._convex_hull_area(points)

    @staticmethod
    def _convex_hull_area(points: list[tuple[float, float]]) -> float:
        """2D 볼록 껍질 면적 (Andrew's monotone chain)."""
        pts = sorted(set(points))
        if len(pts) < 3:
            return 0.0

        def cross(O, A, B):
            return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

        # Lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return 0.0

        # Shoelace
        area = 0.0
        n = len(hull)
        for i in range(n):
            j = (i + 1) % n
            area += hull[i][0] * hull[j][1]
            area -= hull[j][0] * hull[i][1]
        return abs(area) / 2.0

    # ──────────────────────────────────────────────
    #  V1 BuildingModel 호환 어댑터
    # ──────────────────────────────────────────────

    def to_building_model(self):
        """V1 BuildingModel 호환 객체 생성 (load_generator용).

        load_generator.generate_all_loads()가 요구하는 인터페이스를 제공한다.
        실제 BuildingModel을 import하지 않고 duck-typing으로 호환.
        """
        from core.building_model import BuildingModel, StoryInfo

        stories = []
        for i, h in enumerate(self.story_heights, start=1):
            stories.append(StoryInfo(
                story=i,
                height=h,
                usage=self.story_usages.get(i, "office"),
                dead_load_finish=self.story_dead_load_finish.get(i, 1.0),
                slab_thickness=self.story_slab_thickness.get(i, 0.15),
            ))

        # bays 추정: 층별 X/Y 방향 보 간격에서 추출
        bays_x, bays_y = self._estimate_bays()

        bm = BuildingModel(
            stories=stories,
            bays_x=bays_x,
            bays_y=bays_y,
            column_section=self._most_common_section(ElementType.COLUMN) or "H-300x300",
            beam_x_section=self._most_common_section(ElementType.BEAM) or "H-400x200",
            beam_y_section=self._most_common_section(ElementType.BEAM) or "H-400x200",
            material_name=self._most_common_material() or "SS275",
            supports="fixed",
            region=self.region,
            site_class=self.site_class,
            importance=self.importance,
            importance_factor=self.importance_factor,
            seismic_system=self.seismic_system,
            seismic_direction=self.seismic_direction,
            exposure_category=self.exposure_category,
            num_elements_per_member=self.num_elements_per_member,
            auto_combinations=True,
            rigid_diaphragm=self.rigid_diaphragm,
            geometric_nonlinearity=self.geometric_nonlinearity,
        )

        # floor_area_at_story override — convex hull 기반
        original_floor_area = bm.floor_area_at_story
        model_ref = self

        def patched_floor_area(story):
            area = model_ref.floor_area_at_story(story)
            return area if area > 0 else original_floor_area(story)

        bm.floor_area_at_story = patched_floor_area

        return bm

    def _estimate_bays(self) -> tuple[list[float], list[float]]:
        """보 배치에서 bay 간격 추정."""
        if self.num_stories < 1:
            return [8.0], [8.0]

        # 1층 보의 X/Y 좌표에서 간격 추출
        story1_nodes = self.nodes_at_story(1)
        if not story1_nodes:
            story1_nodes = self.nodes_at_story(0)

        xs = sorted(set(round(n.x, 2) for n in story1_nodes))
        ys = sorted(set(round(n.y, 2) for n in story1_nodes))

        bays_x = [round(xs[i+1] - xs[i], 3) for i in range(len(xs)-1)] if len(xs) >= 2 else [8.0]
        bays_y = [round(ys[i+1] - ys[i], 3) for i in range(len(ys)-1)] if len(ys) >= 2 else [8.0]

        # 0 이하 제거
        bays_x = [b for b in bays_x if b > 0.1] or [8.0]
        bays_y = [b for b in bays_y if b > 0.1] or [8.0]

        return bays_x, bays_y

    def _most_common_section(self, elem_type: ElementType) -> str | None:
        """특정 유형 요소의 가장 흔한 단면."""
        from collections import Counter
        sections = [e.section for e in self.elements.values() if e.elem_type == elem_type]
        if not sections:
            return None
        return Counter(sections).most_common(1)[0][0]

    def _most_common_material(self) -> str | None:
        """가장 흔한 재료."""
        from collections import Counter
        materials = [e.material for e in self.elements.values()]
        if not materials:
            return None
        return Counter(materials).most_common(1)[0][0]

    # ──────────────────────────────────────────────
    #  직렬화 (JSON)
    # ──────────────────────────────────────────────

    def to_json(self) -> dict:
        """JSON 직렬화."""
        return {
            "version": "2.0",
            "nodes": [
                {
                    "id": n.id, "x": n.x, "y": n.y, "z": n.z,
                    "story": n.story,
                    "support": n.support.value if n.support else None,
                    "mass": n.mass,
                }
                for n in sorted(self.nodes.values(), key=lambda n: n.id)
            ],
            "elements": [
                {
                    "id": e.id, "node_i": e.node_i, "node_j": e.node_j,
                    "elem_type": e.elem_type.value,
                    "section": e.section, "material": e.material,
                    "release_i": e.release_i.value if e.release_i else None,
                    "release_j": e.release_j.value if e.release_j else None,
                    "beta_angle": e.beta_angle,
                }
                for e in sorted(self.elements.values(), key=lambda e: e.id)
            ],
            "story_elevations": self.story_elevations,
            "story_usages": self.story_usages,
            "story_slab_thickness": self.story_slab_thickness,
            "story_dead_load_finish": self.story_dead_load_finish,
            "environment": {
                "region": self.region,
                "site_class": self.site_class,
                "importance": self.importance,
                "importance_factor": self.importance_factor,
                "seismic_system": self.seismic_system,
                "seismic_direction": self.seismic_direction,
                "exposure_category": self.exposure_category,
            },
            "analysis_options": {
                "num_elements_per_member": self.num_elements_per_member,
                "rigid_diaphragm": self.rigid_diaphragm,
                "geometric_nonlinearity": self.geometric_nonlinearity,
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> "StructuralModel":
        """JSON에서 StructuralModel 생성."""
        model = cls()

        # 노드
        for nd in data.get("nodes", []):
            model.add_node(
                x=nd["x"], y=nd["y"], z=nd["z"],
                support=nd.get("support"),
                story=nd.get("story"),
                node_id=nd["id"],
            )
            if nd.get("mass"):
                model.nodes[nd["id"]].mass = nd["mass"]

        # 요소
        for ed in data.get("elements", []):
            model.add_element(
                node_i=ed["node_i"], node_j=ed["node_j"],
                elem_type=ed.get("elem_type", "beam"),
                section=ed.get("section", "H-400x200"),
                material=ed.get("material", "SS275"),
                release_i=ed.get("release_i"),
                release_j=ed.get("release_j"),
                beta_angle=ed.get("beta_angle", 0.0),
                elem_id=ed["id"],
            )

        # 층 정보
        model.story_elevations = data.get("story_elevations", [])
        model.story_usages = {
            int(k): v for k, v in data.get("story_usages", {}).items()
        }
        model.story_slab_thickness = {
            int(k): v for k, v in data.get("story_slab_thickness", {}).items()
        }
        model.story_dead_load_finish = {
            int(k): v for k, v in data.get("story_dead_load_finish", {}).items()
        }

        # 환경
        env = data.get("environment", {})
        model.region = env.get("region", "")
        model.site_class = env.get("site_class", "S3")
        model.importance = env.get("importance", "II")
        model.importance_factor = env.get("importance_factor", 1.0)
        model.seismic_system = env.get("seismic_system", "ordinary_moment_frame")
        model.seismic_direction = env.get("seismic_direction", "both")
        model.exposure_category = env.get("exposure_category", "B")

        # 해석 옵션
        opts = data.get("analysis_options", {})
        model.num_elements_per_member = opts.get("num_elements_per_member", 4)
        model.rigid_diaphragm = opts.get("rigid_diaphragm", False)
        model.geometric_nonlinearity = opts.get("geometric_nonlinearity", "linear")

        return model

    def save(self, path: str | Path) -> None:
        """JSON 파일로 저장."""
        Path(path).write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "StructuralModel":
        """JSON 파일에서 로드."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(data)

    # ──────────────────────────────────────────────
    #  요약
    # ──────────────────────────────────────────────

    def summary(self) -> dict:
        """모델 요약 정보."""
        columns = self.elements_by_type(ElementType.COLUMN)
        beams = self.elements_by_type(ElementType.BEAM)
        braces = self.elements_by_type(ElementType.BRACE)

        sections_used = sorted(set(e.section for e in self.elements.values()))
        materials_used = sorted(set(e.material for e in self.elements.values()))

        return {
            "num_nodes": len(self.nodes),
            "num_elements": len(self.elements),
            "num_columns": len(columns),
            "num_beams": len(beams),
            "num_braces": len(braces),
            "num_stories": self.num_stories,
            "total_height_m": round(self.total_height, 2),
            "story_heights_m": self.story_heights,
            "story_elevations_m": self.story_elevations,
            "sections_used": sections_used,
            "materials_used": materials_used,
            "supports": [
                {"node_id": n.id, "type": n.support.value}
                for n in sorted(self.nodes.values(), key=lambda n: n.id)
                if n.support
            ],
            "region": self.region,
            "importance": self.importance,
            "rigid_diaphragm": self.rigid_diaphragm,
            "geometric_nonlinearity": self.geometric_nonlinearity,
        }
