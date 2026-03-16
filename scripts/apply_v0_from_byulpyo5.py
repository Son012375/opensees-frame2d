#!/usr/bin/env python
"""별표5 (건축물의 구조기준 등에 관한 규칙) 지역별 기본풍속 데이터를 템플릿에 매핑.

NOTE: 별표5는 구 규칙 기준이며, KDS 41 12 00 (재현기간 500년)과 값이 다를 수 있음.
      참고용 기준값으로 활용하며, verified_by를 'byulpyo5_old_regulation'으로 표기.
"""

import json
from pathlib import Path

# ============================================================
# 별표5 데이터: 시/군/구 → V₀ (m/s) 매핑
# ============================================================

# 광역시/특별시 전체 적용
METRO_V0 = {
    "서울특별시": 30,
    "부산광역시": 40,
    "인천광역시": 30,
    "대구광역시": 25,
    "광주광역시": 30,
    "대전광역시": 30,
    "울산광역시": 35,
    "세종특별자치시": 35,  # 연기군(조치원읍) = 35
    "제주특별자치도": 40,  # 전지역
}

# 경기도 (30 그룹)
GYEONGGI_30 = {
    "과천시", "구리시", "군포시", "김포시", "부천시", "수원시",
    "시흥시", "안산시", "안성시", "안양시", "오산시", "의왕시", "평택시",
    # 별표5 미기재 → 인접지역 기준 30 추정
    "고양시", "광명시", "화성시",
}

# 경기도 (25 그룹)
GYEONGGI_25 = {
    "광주시", "동두천시", "성남시", "양평군", "여주시", "용인시",
    "의정부시", "이천시", "파주시", "포천시", "하남시",
    # 별표5 미기재 → 인접지역 기준 25 추정
    "가평군", "남양주시", "양주시", "연천군",
}

# 강원도
GANGWON_MAP = {
    "강릉시": 40, "속초시": 40, "양양군": 40,
    "고성군": 35, "동해시": 35, "삼척시": 35,
    "원주시": 25, "춘천시": 25, "홍천군": 25, "태백시": 25,
    "철원군": 25, "평창군": 25, "영월군": 25, "정선군": 25,
    "인제군": 25, "화천군": 25, "양구군": 25,
    "횡성군": 25,  # 미기재 → 원주/홍천 인접 25
}

# 충청북도
CHUNGBUK_MAP = {
    "청주시": 35,
    "증평군": 30, "진천군": 30,
    "충주시": 25, "제천시": 25, "보은군": 25, "옥천군": 25,
    "영동군": 25, "괴산군": 25, "음성군": 25, "단양군": 25,
}

# 충청남도
CHUNGNAM_MAP = {
    "천안시": 35, "아산시": 35, "서산시": 35, "보령시": 35,
    "서천군": 35, "태안군": 35, "홍성군": 35,
    "당진시": 30,
    "공주시": 25, "논산시": 25, "금산군": 25, "부여군": 25, "청양군": 25,
    "계룡시": 30,  # 미기재 → 대전(30) 인접
    "예산군": 30,  # 미기재 → 홍성(35)/아산(35) 인접, 내륙 보정
}

# 경상북도
GYEONGBUK_MAP = {
    "포항시": 45, "울릉군": 45,
    "울진군": 35, "경주시": 35,
    "영덕군": 30,
    "김천시": 25, "안동시": 25, "구미시": 25, "영주시": 25,
    "영천시": 25, "상주시": 25, "문경시": 25, "경산시": 25,
    "군위군": 25, "의성군": 25, "청송군": 25, "영양군": 25,
    "청도군": 25, "고령군": 25, "예천군": 25, "봉화군": 25,
    "성주군": 25,  # 미기재 → 고령/김천 인접
    "칠곡군": 25,  # 미기재 → 구미/대구 인접
}

# 경상남도
GYEONGNAM_MAP = {
    "창원시": 35, "김해시": 35, "양산시": 35, "거제시": 35,
    "통영시": 35, "사천시": 35, "고성군": 35, "남해군": 35,
    "함안군": 30, "밀양시": 25,
    "진주시": 25, "거창군": 25, "창녕군": 25, "합천군": 25,
    "함양군": 25, "산청군": 25, "의령군": 25,
    "하동군": 30,  # 미기재 → 남해(35)/산청(25) 중간, 해안
}

# 전북특별자치도
JEONBUK_MAP = {
    "군산시": 40,
    "익산시": 35,
    "전주시": 25, "정읍시": 25, "남원시": 25, "고창군": 25,
    "부안군": 25, "임실군": 25, "순창군": 25, "진안군": 25,
    "장수군": 25, "무주군": 25,
    "완주군": 25,  # 미기재 → 전주 인접
    "김제시": 25,  # 미기재 → 전주/군산 사이, 내륙
}

# 전라남도
JEONNAM_MAP = {
    "여수시": 35, "목포시": 35, "완도군": 35, "진도군": 35,
    "해남군": 35, "고흥군": 35,
    "순천시": 30, "광양시": 30, "나주시": 30, "화순군": 30,
    "영암군": 30, "장흥군": 30, "보성군": 30, "무안군": 30,
    "함평군": 30, "영광군": 30, "강진군": 30,
    "담양군": 25, "구례군": 25,
    "장성군": 25,  # 미기재 → 담양/광주 인접
    "신안군": 35,  # 미기재 → 목포 인접 도서
    "곡성군": 25,  # 미기재 → 구례/순창 인접
}


def get_v0(sido: str, sigungu: str) -> int | None:
    """시/도 + 시/군/구로 V₀ 조회."""
    # 광역시/특별시/특별자치도
    if sido in METRO_V0:
        return METRO_V0[sido]

    # 도 단위
    if sido == "경기도":
        if sigungu in GYEONGGI_30:
            return 30
        if sigungu in GYEONGGI_25:
            return 25
        return None

    if "강원" in sido:
        return GANGWON_MAP.get(sigungu)

    if sido == "충청북도":
        return CHUNGBUK_MAP.get(sigungu)

    if sido == "충청남도":
        return CHUNGNAM_MAP.get(sigungu)

    if sido == "경상북도":
        return GYEONGBUK_MAP.get(sigungu)

    if sido == "경상남도":
        return GYEONGNAM_MAP.get(sigungu)

    if "전북" in sido:
        return JEONBUK_MAP.get(sigungu)

    if sido == "전라남도":
        return JEONNAM_MAP.get(sigungu)

    return None


def main():
    template_path = Path("data/hazard_region_template.json")
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 메타데이터 업데이트
    data["metadata"]["wind_source"] = "건축물의 구조기준 등에 관한 규칙 별표5 (제13조관련)"
    data["metadata"]["wind_note"] = (
        "구 규칙 기준값. KDS 41 12 00 (재현기간 500년) 등고선 지도와 "
        "값이 다를 수 있으므로 참고용으로 활용. "
        "별표5 미기재 지역은 인접 지역 기준 추정값."
    )
    data["metadata"]["verified_by"] = "byulpyo5_old_regulation"

    filled = 0
    missing = []

    for region in data["regions"]:
        v0 = get_v0(region["sido"], region["sigungu"])
        if v0 is not None:
            region["v0"] = v0
            filled += 1
        else:
            missing.append(f"{region['sido']} {region['sigungu']}")

    # 결과 저장
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"V0 mapping done: {filled}/{len(data['regions'])} regions")
    if missing:
        print(f"\nMissing {len(missing)}:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All regions mapped!")


if __name__ == "__main__":
    main()
