"""건물 모델 중간 표현 (Intermediate Representation).

IFC 또는 JSON 설정에서 파싱된 건물 정보를 통합하는 데이터 구조.
BuildingModel은 하중 자동 생성 및 frame_3d 해석의 입력으로 사용된다.

Usage:
    config = {...}
    model = BuildingModel.from_json(config)
    # → load_generator에서 하중 자동 생성
    # → frame_3d.analyze_frame_3d_multi()로 해석

비정형 건물:
    zones 필드를 사용하여 L자형, setback 등 비정형 평면을 정의할 수 있다.
    각 zone은 독립적인 직교 그리드이며, 경계에서 노드가 병합된다.
    zones가 없으면 bays_x/bays_y로 단일 정형 그리드를 생성한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class StoryInfo:
    """층 정보."""
    story: int              # 1-based (1 = 최하층)
    height: float           # m (층고)
    usage: str = "office"   # 용도: office, residential, retail, parking, ...
    dead_load_finish: float = 1.0   # kN/m², 마감재 추가 고정하중
    slab_thickness: float = 0.15    # m, 슬래브 두께 (자중 계산용)


@dataclass
class ZoneInfo:
    """비정형 건물의 존(zone) 정보.

    각 존은 원점 오프셋이 있는 직교 서브그리드이며,
    적용 층 범위를 지정하여 setback 등을 표현한다.
    """
    id: str                          # 존 식별자 (예: "A", "B")
    bays_x: list[float] = field(default_factory=list)   # m
    bays_y: list[float] = field(default_factory=list)   # m
    origin_x: float = 0.0           # m, 글로벌 X 오프셋
    origin_y: float = 0.0           # m, 글로벌 Y 오프셋
    story_from: int = 1             # 시작 층 (1-based, inclusive)
    story_to: int | None = None     # 끝 층 (None = 최상층까지)

    @property
    def width_x(self) -> float:
        return sum(self.bays_x)

    @property
    def width_y(self) -> float:
        return sum(self.bays_y)

    @property
    def area(self) -> float:
        return self.width_x * self.width_y

    def x_coords(self) -> list[float]:
        """존 로컬 X 좌표 → 글로벌 좌표."""
        coords = [self.origin_x]
        for bx in self.bays_x:
            coords.append(coords[-1] + bx)
        return coords

    def y_coords(self) -> list[float]:
        """존 로컬 Y 좌표 → 글로벌 좌표."""
        coords = [self.origin_y]
        for by in self.bays_y:
            coords.append(coords[-1] + by)
        return coords

    def active_at_story(self, story: int) -> bool:
        """해당 층에서 이 존이 활성화되는지 확인."""
        if story < self.story_from:
            return False
        if self.story_to is not None and story > self.story_to:
            return False
        return True


def _rect_intersection_area(a: ZoneInfo, b: ZoneInfo) -> float:
    """두 존의 직교 교차 영역 면적."""
    ax0, ax1 = a.origin_x, a.origin_x + a.width_x
    ay0, ay1 = a.origin_y, a.origin_y + a.width_y
    bx0, bx1 = b.origin_x, b.origin_x + b.width_x
    by0, by1 = b.origin_y, b.origin_y + b.width_y
    dx = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    dy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return dx * dy


@dataclass
class BuildingModel:
    """건물 모델 중간 표현.

    IFC 파싱 또는 JSON 설정에서 생성되며,
    하중 자동 생성 → frame_3d 해석의 입력으로 사용된다.
    """

    # ── 기하 (정규 그리드 — zones 미사용 시) ──
    stories: list[StoryInfo] = field(default_factory=list)
    bays_x: list[float] = field(default_factory=list)   # m
    bays_y: list[float] = field(default_factory=list)   # m

    # ── 비정형 기하 (존 기반) ──
    zones: list[ZoneInfo] = field(default_factory=list)  # 비어있으면 정형 그리드

    # ── 단면/재료 ──
    column_section: str = "H-300x300"
    beam_x_section: str = "H-400x200"
    beam_y_section: str = "H-400x200"
    material_name: str = "SS275"

    # ── 경계조건 ──
    supports: Literal["fixed", "pinned"] = "fixed"

    # ── 위치/환경 (하중 자동생성용) ──
    region: str = ""                # "서울", "강남구", "부산" 등
    site_class: str = "S3"          # S1~S5
    importance: str = "II"          # 특, I, II
    importance_factor: float = 1.0

    # ── 내진 (등가정적해석법) ──
    seismic_system: str = "ordinary_moment_frame"   # → R, Ω₀, Cd 조회 키
    seismic_direction: str = "both"                 # "x", "y", "both"

    # ── 풍하중 ──
    exposure_category: str = "B"    # A, B, C, D

    # ── 분석 옵션 ──
    num_elements_per_member: int = 4
    auto_combinations: bool = True  # True면 KDS 표준 조합 자동 생성
    rigid_diaphragm: bool = False   # True면 층별 강체 다이어프램 적용
    member_releases: dict = field(default_factory=dict)  # 부재 릴리즈 (힌지)
    geometric_nonlinearity: str = "linear"  # "linear" or "pdelta"
    slab_distribution: str = "2way"         # "2way" (45° yield line) or "uniform" (50/50)

    # ── 자동 계산 속성 ──

    @property
    def is_irregular(self) -> bool:
        """비정형 건물(존 기반)인지 여부."""
        return len(self.zones) > 0

    @property
    def num_stories(self) -> int:
        return len(self.stories)

    @property
    def total_height(self) -> float:
        return sum(s.height for s in self.stories)

    @property
    def total_width_x(self) -> float:
        if self.is_irregular:
            all_x = []
            for z in self.zones:
                all_x.extend(z.x_coords())
            return max(all_x) - min(all_x) if all_x else 0.0
        return sum(self.bays_x)

    @property
    def total_width_y(self) -> float:
        if self.is_irregular:
            all_y = []
            for z in self.zones:
                all_y.extend(z.y_coords())
            return max(all_y) - min(all_y) if all_y else 0.0
        return sum(self.bays_y)

    @property
    def floor_area(self) -> float:
        """1층 바닥 면적 (m²). 비정형 시 존 면적 합산 (중복 제외 근사)."""
        if self.is_irregular:
            return self.floor_area_at_story(1)
        return self.total_width_x * self.total_width_y

    def get_active_zones(self, story: int) -> list[ZoneInfo]:
        """해당 층에서 활성화된 존 목록."""
        return [z for z in self.zones if z.active_at_story(story)]

    def floor_area_at_story(self, story: int) -> float:
        """특정 층의 바닥 면적 (m²).

        비정형 시 활성 존의 면적 합산 (inclusion-exclusion으로 중첩 보정).
        """
        if not self.is_irregular:
            return self.total_width_x * self.total_width_y

        active = self.get_active_zones(story)
        if not active:
            return 0.0

        # Inclusion-exclusion: 합집합 면적 = Σ개별 - Σ쌍교차
        total = sum(z.area for z in active)
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                total -= _rect_intersection_area(active[i], active[j])
        return max(total, 0.0)

    @property
    def story_heights_list(self) -> list[float]:
        """frame_3d 입력용 층고 리스트 (아래→위)."""
        return [s.height for s in self.stories]

    @property
    def cumulative_heights(self) -> list[float]:
        """각 층의 누적 높이 (지면 기준)."""
        h = 0.0
        result = []
        for s in self.stories:
            h += s.height
            result.append(h)
        return result

    @classmethod
    def from_json(cls, config: dict) -> "BuildingModel":
        """JSON 설정 딕셔너리에서 BuildingModel 생성.

        Args:
            config: 건물 설정 JSON. 필수 키: stories, bays_x, bays_y.

        Example config:
            {
                "stories": [
                    {"height": 4.0, "usage": "retail", "dead_load_finish": 1.5},
                    {"height": 3.5, "usage": "office"},
                    {"height": 3.5, "usage": "office"},
                ],
                "bays_x": [8.0, 8.0, 8.0],
                "bays_y": [8.0, 8.0],
                "column_section": "H-300x300",
                "region": "서울",
                "site_class": "S3",
                "importance": "II",
                "auto_combinations": true
            }
        """
        # 필수 필드 검증
        if "stories" not in config:
            raise ValueError("config에 'stories' 키가 필요합니다.")

        has_zones = "zones" in config and config["zones"]
        if not has_zones:
            # 정형 건물: bays_x/bays_y 필수
            if "bays_x" not in config:
                raise ValueError("config에 'bays_x' 키가 필요합니다.")
            if "bays_y" not in config:
                raise ValueError("config에 'bays_y' 키가 필요합니다.")

        # 층 정보 파싱
        story_list = []
        for i, s in enumerate(config["stories"], start=1):
            if isinstance(s, dict):
                story_list.append(StoryInfo(
                    story=i,
                    height=s["height"],
                    usage=s.get("usage", "office"),
                    dead_load_finish=s.get("dead_load_finish", 1.0),
                    slab_thickness=s.get("slab_thickness", 0.15),
                ))
            elif isinstance(s, (int, float)):
                # 단순히 높이만 제공된 경우
                story_list.append(StoryInfo(story=i, height=float(s)))
            else:
                raise ValueError(f"stories[{i-1}] 형식 오류: {type(s)}")

        # 존(zone) 파싱
        zone_list = []
        if has_zones:
            for zd in config["zones"]:
                zone_list.append(ZoneInfo(
                    id=zd["id"],
                    bays_x=zd["bays_x"],
                    bays_y=zd["bays_y"],
                    origin_x=zd.get("origin_x", 0.0),
                    origin_y=zd.get("origin_y", 0.0),
                    story_from=zd.get("story_from", 1),
                    story_to=zd.get("story_to", None),
                ))

        # 비정형 시 bays_x/bays_y를 첫 번째 존에서 추출 (호환용)
        bays_x = config.get("bays_x", zone_list[0].bays_x if zone_list else [])
        bays_y = config.get("bays_y", zone_list[0].bays_y if zone_list else [])

        # 중요도계수 자동 매핑
        importance = config.get("importance", "II")
        importance_factor = config.get("importance_factor", None)
        if importance_factor is None:
            _ie_map = {"특": 1.5, "I": 1.2, "II": 1.0}
            importance_factor = _ie_map.get(importance, 1.0)

        return cls(
            stories=story_list,
            bays_x=bays_x,
            bays_y=bays_y,
            zones=zone_list,
            column_section=config.get("column_section", "H-300x300"),
            beam_x_section=config.get("beam_x_section", "H-400x200"),
            beam_y_section=config.get("beam_y_section", "H-400x200"),
            material_name=config.get("material_name", "SS275"),
            supports=config.get("supports", "fixed"),
            region=config.get("region", ""),
            site_class=config.get("site_class", "S3"),
            importance=importance,
            importance_factor=importance_factor,
            seismic_system=config.get("seismic_system", "ordinary_moment_frame"),
            seismic_direction=config.get("seismic_direction", "both"),
            exposure_category=config.get("exposure_category", "B"),
            num_elements_per_member=config.get("num_elements_per_member", 4),
            auto_combinations=config.get("auto_combinations", True),
            rigid_diaphragm=config.get("rigid_diaphragm", False),
            member_releases=config.get("member_releases", {}),
            geometric_nonlinearity=config.get("geometric_nonlinearity", "linear"),
            slab_distribution=config.get("slab_distribution", "2way"),
        )

    @classmethod
    def from_ifc(cls, ifc_path: str, config: dict) -> "BuildingModel":
        """IFC 파일 + JSON 설정에서 BuildingModel 생성.

        IFC에서 기하 정보(stories, bays_x, bays_y)를 추출하고,
        config에서 용도/환경 정보를 보완한다.

        Args:
            ifc_path: IFC 파일 경로
            config: 추가 설정 (usage 등 IFC에 없는 정보)
        """
        from core.ifc_parser import parse_ifc

        ifc_data = parse_ifc(ifc_path)

        # IFC 기하 → config에 병합 (IFC 우선, config로 보완)
        merged = dict(config)

        # stories: IFC에서 높이, config에서 usage 보완
        ifc_stories = ifc_data.get("stories", [])
        cfg_stories = config.get("stories", [])

        merged_stories = []
        for i, ifc_s in enumerate(ifc_stories):
            story_cfg = {}
            story_cfg["height"] = ifc_s["height"]

            # IFC에서 추출된 슬래브 두께 (mm → m)
            slab_t_mm = ifc_s.get("slab_thickness_mm")
            if slab_t_mm is not None:
                story_cfg["slab_thickness"] = round(slab_t_mm / 1000.0, 4)

            # config stories에서 용도 등 보완 (config가 있으면 IFC보다 우선)
            if i < len(cfg_stories) and isinstance(cfg_stories[i], dict):
                story_cfg["usage"] = cfg_stories[i].get("usage", "office")
                story_cfg["dead_load_finish"] = cfg_stories[i].get("dead_load_finish", 1.0)
                if "slab_thickness" in cfg_stories[i]:
                    story_cfg["slab_thickness"] = cfg_stories[i]["slab_thickness"]
            merged_stories.append(story_cfg)

        merged["stories"] = merged_stories

        # bays: IFC 우선
        if ifc_data.get("bays_x"):
            merged["bays_x"] = ifc_data["bays_x"]
        if ifc_data.get("bays_y"):
            merged["bays_y"] = ifc_data["bays_y"]

        # 단면: IFC 추출값 → config 지정값 → 기본값
        detected = ifc_data.get("detected_sections", {})
        if detected.get("column") and "column_section" not in config:
            merged["column_section"] = detected["column"]
        if detected.get("beam") and "beam_x_section" not in config:
            merged["beam_x_section"] = detected["beam"]
        if detected.get("beam") and "beam_y_section" not in config:
            merged["beam_y_section"] = detected["beam"]

        # 재료: IFC 추출값 → config 지정값 → 기본값
        if ifc_data.get("detected_material") and "material_name" not in config:
            merged["material_name"] = ifc_data["detected_material"]

        return cls.from_json(merged)

    def to_frame3d_kwargs(self) -> dict:
        """frame_3d.analyze_frame_3d_multi()에 전달할 키워드 인자 생성.

        load_cases와 load_combinations은 별도로 제공해야 함.
        """
        kwargs = {
            "stories": self.story_heights_list,
            "bays_x": self.bays_x,
            "bays_y": self.bays_y,
            "supports": self.supports,
            "column_section": self.column_section,
            "beam_x_section": self.beam_x_section,
            "beam_y_section": self.beam_y_section,
            "material_name": self.material_name,
            "num_elements_per_member": self.num_elements_per_member,
            "rigid_diaphragm": self.rigid_diaphragm,
            "member_releases": self.member_releases or None,
            "geometric_nonlinearity": self.geometric_nonlinearity,
            "slab_distribution": self.slab_distribution,
        }
        if self.is_irregular:
            kwargs["zones"] = [
                {
                    "id": z.id,
                    "bays_x": z.bays_x,
                    "bays_y": z.bays_y,
                    "origin_x": z.origin_x,
                    "origin_y": z.origin_y,
                    "story_from": z.story_from,
                    "story_to": z.story_to,
                }
                for z in self.zones
            ]
        return kwargs

    def summary(self) -> dict:
        """모델 요약 정보."""
        info = {
            "num_stories": self.num_stories,
            "total_height_m": self.total_height,
            "total_width_x_m": self.total_width_x,
            "total_width_y_m": self.total_width_y,
            "floor_area_m2": self.floor_area,
            "bays_x": self.bays_x,
            "bays_y": self.bays_y,
            "usages": [s.usage for s in self.stories],
            "column_section": self.column_section,
            "beam_x_section": self.beam_x_section,
            "beam_y_section": self.beam_y_section,
            "material": self.material_name,
            "region": self.region,
            "site_class": self.site_class,
            "importance": self.importance,
            "is_irregular": self.is_irregular,
        }
        if self.is_irregular:
            info["zones"] = [
                {"id": z.id, "bays_x": z.bays_x, "bays_y": z.bays_y,
                 "origin": [z.origin_x, z.origin_y],
                 "stories": f"{z.story_from}-{z.story_to or self.num_stories}",
                 "area_m2": z.area}
                for z in self.zones
            ]
            info["floor_areas_per_story"] = [
                round(self.floor_area_at_story(s), 2) for s in range(1, self.num_stories + 1)
            ]
        return info
