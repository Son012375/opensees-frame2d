#!/usr/bin/env python
"""별표4 (건축물의 구조기준 등에 관한 규칙) 지역별 기본지상설하중 데이터를 템플릿에 매핑.

NOTE: 별표4는 2009년 이후 삭제된 구 규정이며,
      KDS 41 12 00 (그림 4.2-1)의 등고선 지도와 값이 다를 수 있음.
      참고용 baseline으로 활용하며, verified_by를 'byulpyo4_deleted_2009'로 표기.

별표4 원문 데이터:
  0.5 kN/m2: 서울, 부산, 대구, 광주, 대전, 울산, 수원, 춘천, 서산, 청주,
             포항, 군산, 전주, 충무(=통영), 목포, 여수, 제주, 서귀포, 진주, 울진, 이천
  0.8 kN/m2: 인천
  2.0 kN/m2: 속초
  3.0 kN/m2: 강릉
  7.0 kN/m2: 울릉도, 대관령(=평창군)
  기타 미기재 지역: 0.5 (기본값)
"""

import json
from pathlib import Path

# ============================================================
# 별표4 데이터: 도시명 → Sg (kN/m2) 매핑
# ============================================================

# 별표4에 명시된 도시 → Sg 매핑
# 도시명을 현행 시/군/구명으로 변환
BYULPYO4_EXPLICIT = {
    # 0.5 그룹 (명시 지역)
    "서울": 0.5,
    "부산": 0.5,
    "대구": 0.5,
    "광주": 0.5,
    "대전": 0.5,
    "울산": 0.5,
    "수원시": 0.5,
    "춘천시": 0.5,
    "서산시": 0.5,
    "청주시": 0.5,
    "포항시": 0.5,
    "군산시": 0.5,
    "전주시": 0.5,
    "통영시": 0.5,  # 충무 → 통영시
    "목포시": 0.5,
    "여수시": 0.5,
    "제주시": 0.5,  # 제주 → 제주시 (제주특별자치도)
    "서귀포시": 0.5,
    "진주시": 0.5,
    "울진군": 0.5,
    "이천시": 0.5,
    # 0.8 그룹
    "인천": 0.8,
    # 2.0 그룹
    "속초시": 2.0,
    # 3.0 그룹
    "강릉시": 3.0,
    # 7.0 그룹
    "울릉군": 7.0,
    "평창군": 7.0,  # 대관령 → 평창군
}

# 광역시/특별시 전체에 동일 값 적용
METRO_SG = {
    "서울특별시": 0.5,
    "부산광역시": 0.5,
    "대구광역시": 0.5,
    "광주광역시": 0.5,
    "대전광역시": 0.5,
    "울산광역시": 0.5,
    "인천광역시": 0.8,
    "세종특별자치시": 0.5,  # 미기재 → 대전(0.5) 인접, 기본값
    "제주특별자치도": 0.5,  # 제주/서귀포 모두 0.5
}

# 강원도: 산간 지역은 KDS 등고선에서 실제로 더 높을 수 있음
# 별표4 기준 + 인접 지역 추정
GANGWON_SG = {
    "강릉시": 3.0,    # 별표4 명시
    "속초시": 2.0,    # 별표4 명시
    "평창군": 7.0,    # 별표4 대관령 = 7.0
    "춘천시": 0.5,    # 별표4 명시
    # 나머지 강원도: 산간 특성상 0.5보다 높을 가능성 있지만,
    # 별표4 기본값 적용 후 needs_review 처리
    "원주시": 0.5,
    "동해시": 2.0,    # 속초(2.0)/강릉(3.0) 인접 해안, 중간값 추정
    "삼척시": 2.0,    # 동해시 인접
    "태백시": 3.0,    # 고산지대, 평창(7.0)/강릉(3.0) 인접
    "양양군": 2.0,    # 속초(2.0) 인접
    "고성군": 2.0,    # 속초(2.0) 인접
    "인제군": 1.0,    # 내륙 산간, 속초/춘천 사이
    "홍천군": 0.5,    # 춘천(0.5) 인접
    "횡성군": 0.5,    # 원주(0.5) 인접
    "영월군": 1.0,    # 태백(3.0)/원주(0.5) 사이 산간
    "정선군": 3.0,    # 태백(3.0)/평창(7.0) 인접 산간
    "철원군": 0.5,    # 춘천(0.5) 인접
    "화천군": 0.5,    # 춘천(0.5) 인접
    "양구군": 0.5,    # 춘천(0.5) 인접
}

# 경상북도: 울릉군 특수
GYEONGBUK_SG = {
    "울릉군": 7.0,    # 별표4 명시
    "울진군": 0.5,    # 별표4 명시
    "포항시": 0.5,    # 별표4 명시
}

DEFAULT_SG = 0.5  # 별표4 미기재 지역 기본값


def get_sg(sido: str, sigungu: str) -> float:
    """시/도 + 시/군/구로 Sg 조회."""
    # 광역시/특별시/특별자치도
    if sido in METRO_SG:
        return METRO_SG[sido]

    # 강원도 (특수 지역 다수)
    if "강원" in sido:
        return GANGWON_SG.get(sigungu, DEFAULT_SG)

    # 경상북도 (울릉군 특수)
    if sido == "경상북도":
        return GYEONGBUK_SG.get(sigungu, DEFAULT_SG)

    # 나머지: 별표4 명시 도시 확인
    if sigungu in BYULPYO4_EXPLICIT:
        return BYULPYO4_EXPLICIT[sigungu]

    # 미기재 → 기본값 0.5
    return DEFAULT_SG


# 강원도 산간 지역: KDS 등고선 기준으로 재검증 필요
NEEDS_REVIEW_REGIONS = {
    ("강원특별자치도", "태백시"),
    ("강원특별자치도", "정선군"),
    ("강원특별자치도", "평창군"),
    ("강원특별자치도", "인제군"),
    ("강원특별자치도", "영월군"),
    ("강원특별자치도", "동해시"),
    ("강원특별자치도", "삼척시"),
    ("강원특별자치도", "양양군"),
    ("강원특별자치도", "고성군"),
    ("경상북도", "울릉군"),
}


def main():
    template_path = Path("data/hazard_region_template.json")
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 메타데이터 업데이트
    data["metadata"]["snow_source"] = (
        "건축물의 구조기준 등에 관한 규칙 별표4 (지상적설하중의 기본값)"
    )
    data["metadata"]["snow_note"] = (
        "2009년 이후 삭제된 구 규정 기준값. "
        "KDS 41 12 00 (그림 4.2-1) 등고선 지도와 값이 다를 수 있으므로 참고용으로 활용. "
        "별표4 미기재 지역은 기본값 0.5 적용. "
        "강원 산간 등 고설지역은 실제 Sg가 더 높을 수 있음 (needs_review)."
    )
    data["metadata"]["sg_verified_by"] = "byulpyo4_deleted_2009"

    filled = 0
    review_count = 0
    sg_distribution = {}

    for region in data["regions"]:
        sg = get_sg(region["sido"], region["sigungu"])
        region["sg"] = sg
        filled += 1

        # 분포 집계
        sg_distribution[sg] = sg_distribution.get(sg, 0) + 1

        # needs_review 체크
        if (region["sido"], region["sigungu"]) in NEEDS_REVIEW_REGIONS:
            region["sg_needs_review"] = True
            review_count += 1

    # 결과 저장
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Sg mapping done: {filled}/{len(data['regions'])} regions")
    print(f"\nSg distribution:")
    for sg_val in sorted(sg_distribution.keys()):
        print(f"  {sg_val} kN/m2: {sg_distribution[sg_val]} regions")
    print(f"\nNeeds review (mountain/island): {review_count} regions")
    print("All regions mapped!")


if __name__ == "__main__":
    main()
