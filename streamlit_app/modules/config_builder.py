"""UI 입력 → BuildingModel JSON config 변환 모듈."""
from __future__ import annotations

# 용도 한/영 매핑 (드롭다운 표시용)
USAGE_DISPLAY = {
    "office": "Office (사무실)",
    "residential": "Residential (주거)",
    "retail": "Retail (판매시설)",
    "parking": "Parking (주차장)",
    "assembly": "Assembly (집회시설)",
    "storage": "Storage (창고)",
    "hospital": "Hospital (병원)",
    "school": "School (학교)",
    "library": "Library (도서관)",
    "corridor": "Corridor (복도/계단)",
    "restaurant": "Restaurant (식당)",
    "hotel": "Hotel (숙박)",
    "factory": "Factory (공장)",
    "gym": "Gym (체육관)",
    "roof": "Roof (지붕)",
}

# 내진시스템 한/영 매핑
SEISMIC_SYSTEM_DISPLAY = {
    "special_moment_frame": "Steel Special Moment Frame (R=8)",
    "intermediate_moment_frame": "Steel Intermediate Moment Frame (R=4.5)",
    "ordinary_moment_frame": "Steel Ordinary Moment Frame (R=3.5)",
    "rc_special_moment_frame": "RC Special Moment Frame (R=8)",
    "rc_intermediate_moment_frame": "RC Intermediate Moment Frame (R=7)",
    "rc_ordinary_moment_frame": "RC Ordinary Moment Frame (R=8)",
    "rc_special_shear_wall": "RC Special Shear Wall (R=5)",
    "rc_ordinary_shear_wall": "RC Ordinary Shear Wall (R=4)",
    "special_braced_frame": "Steel Special Braced Frame (R=8)",
    "ordinary_braced_frame": "Steel Ordinary Braced Frame (R=1.5)",
}

# 지반등급
SITE_CLASSES = ["S1", "S2", "S3", "S4", "S5"]

# 중요도
IMPORTANCE_LEVELS = ["특", "I", "II"]

# 대표 단면 목록
COMMON_SECTIONS = [
    "H-200x200", "H-244x175", "H-250x250", "H-294x200",
    "H-300x300", "H-340x250", "H-344x348", "H-350x350",
    "H-386x299", "H-390x300", "H-394x398", "H-400x400",
    "H-414x405", "H-428x407", "H-440x300", "H-458x417",
    "H-482x300", "H-488x300", "H-496x432", "H-500x200",
    "H-582x300", "H-588x300", "H-594x302", "H-600x200",
    "H-692x300", "H-700x300", "H-792x300", "H-900x300",
]

# 대표 지역 목록
COMMON_REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "수원", "고양", "용인", "성남", "청주", "천안", "전주", "포항",
    "제주", "창원", "김해", "강릉",
]


def build_config(
    num_stories: int,
    story_height: float,
    first_story_height: float | None,
    num_bays_x: int,
    bay_width_x: float,
    num_bays_y: int,
    bay_width_y: float,
    usage: str,
    column_section: str,
    beam_x_section: str,
    beam_y_section: str,
    material_name: str,
    region: str,
    site_class: str,
    importance: str,
    seismic_system: str,
    slab_thickness: float = 0.15,
    dead_load_finish: float = 1.0,
) -> dict:
    """UI 입력값들을 BuildingModel JSON config로 변환.

    Returns:
        BuildingModel.from_json()에 맞는 dict
    """
    # 층 정보 생성
    stories = []
    for i in range(num_stories):
        h = first_story_height if (i == 0 and first_story_height) else story_height
        # 최상층은 roof
        u = "roof" if i == num_stories - 1 else usage
        stories.append({
            "height": h,
            "usage": u,
            "dead_load_finish": dead_load_finish,
            "slab_thickness": slab_thickness,
        })

    config = {
        "stories": stories,
        "bays_x": [bay_width_x] * num_bays_x,
        "bays_y": [bay_width_y] * num_bays_y,
        "column_section": column_section,
        "beam_x_section": beam_x_section,
        "beam_y_section": beam_y_section,
        "material_name": material_name,
        "region": region,
        "site_class": site_class,
        "importance": importance,
        "seismic_system": seismic_system,
        "supports": "fixed",
        "auto_combinations": True,
        "num_elements_per_member": 4,
    }
    return config
