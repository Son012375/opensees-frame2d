"""3D 해석용 단면 물성 조회 모듈.

기존 simple_beam.py의 get_section_from_db()를 확장하여
Iy(약축), J(비틀림상수)를 추가로 반환한다.

- Iy: 대부분 테이블에 iy 컬럼으로 존재
- J: 중공단면(CHS/SHS/RHS)은 it 컬럼, 개방단면은 근사식으로 계산
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from supabase import create_client, Client

# Supabase 설정 (simple_beam.py와 동일)
SUPABASE_URL = "https://kuqoajdqvfsobhhlnyyg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt1cW9hamRxdmZzb2JoaGxueXlnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkzNjA1MjksImV4cCI6MjA4NDkzNjUyOX0.5-rSgPXFrMi4xxdrmxgqZdQnl1U2ta-qDYd2IpLMHzM"

_supabase_client: Optional[Client] = None


def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


@dataclass
class BeamSection3D:
    """3D 해석용 단면 물성."""
    name: str
    A: float     # mm²
    Ix: float    # mm⁴ (강축 단면2차모멘트)
    Iy: float    # mm⁴ (약축 단면2차모멘트)
    J: float     # mm⁴ (비틀림상수)
    h: float     # mm (단면 높이)
    b: float     # mm (단면 폭)
    j_source: str = "db"  # "db", "approximate", "fallback"
    tw: float = 0.0  # mm (웹 두께), 0 = 미확인
    tf: float = 0.0  # mm (플랜지 두께), 0 = 미확인


# ── 테이블 매핑 ──
# (schema, table, area_col, ix_col, iy_col, j_col, h_col, b_col, tw_col, tf_col)
# j_col=None → 개방단면, J 근사 계산
SECTION_3D_MAP = {
    "H-":   ("ks3502", "h_beam_sections",   "area", "ix", "iy", None, "h", "b", "t1", "t2"),
    "I-":   ("ks3502", "i_beam_sections",    "area", "ix", "iy", None, "h", "b", "t1", "t2"),
    "TFC-": ("ks3502", "tfc_channel_sections", "area", "ix", "iy", None, "h", "b", "t1", "t2"),
    "PFC-": ("ks3502", "pfc_channel_sections", "area", "ix", "iy", None, "h", "b", "t1", "t2"),
    "T-":   ("ks3502", "t_beam_sections",    "area", "is_val", "iy", None, "h", "b", "t1", "t2"),
    "○-":   ("ks3568", "chs_sections",       "area", "i", "i", "it", "d", "d", None, None),
    # L-, FB-, □- 는 별도 처리
}

# ㄱ형강 (J 근사: bt³/3)
ANGLE_3D_TABLES = [
    ("ks3502", "equal_angle_sections", "area", "ix_max", "iy_min", "a", "a"),
    ("ks3502", "unequal_angle_sections", "area", "ix", "iy", "a", "b"),
    ("ks3502", "unequal_leg_thickness_angle_sections", "area", "ix", "iy", "a", "b"),
]

# □ 중공단면 (it 컬럼 보유)
SQUARE_3D_TABLES = [
    ("ks3568", "shs_sections", "area", "i", "i", "it", "b", "b"),
    ("ks3568", "rhs_hollow_sections", "area", "ix", "iy", "it", "h", "b"),
]


def _compute_j_open_section(h_mm: float, b_mm: float,
                             tw_mm: float, tf_mm: float) -> float:
    """개방 H/I 단면 J 근사: J ≈ (2×b×tf³ + d×tw³) / 3, d = h - 2tf."""
    d = h_mm - 2.0 * tf_mm
    if d < 0:
        d = 0.0
    return (2.0 * b_mm * tf_mm ** 3 + d * tw_mm ** 3) / 3.0


def _query_section_3d(supabase, schema: str, table: str,
                       section_name: str, mapping: dict) -> Optional[dict]:
    """Supabase에서 단면 row 전체를 조회."""
    query = supabase.schema(schema).table(table).select("*").eq("name", section_name)
    result = query.execute()
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


def _row_to_section_3d(row: dict, ix_col: str, iy_col: str,
                        j_col: Optional[str], h_col: str, b_col: str,
                        tw_col: Optional[str] = None,
                        tf_col: Optional[str] = None) -> BeamSection3D:
    """DB row → BeamSection3D 변환."""
    A = row.get("area", 0) * 100          # cm² → mm²
    Ix = row.get(ix_col, 0) * 10000       # cm⁴ → mm⁴
    Iy = row.get(iy_col, 0) * 10000       # cm⁴ → mm⁴
    h = row.get(h_col, 0)                 # mm
    b = row.get(b_col, 0)                 # mm

    # J 결정
    if j_col and row.get(j_col) is not None:
        J = row[j_col] * 10000            # cm⁴ → mm⁴
        j_source = "db"
    elif tw_col and tf_col and row.get(tw_col) and row.get(tf_col):
        tw = row[tw_col]  # mm
        tf = row[tf_col]  # mm
        J = _compute_j_open_section(h, b, tw, tf)
        j_source = "approximate"
    else:
        # 최후 fallback: J = Ix의 1% (매우 보수적)
        J = Ix * 0.01
        j_source = "fallback"

    # tw, tf 저장 (design check용)
    tw_val = row.get(tw_col, 0) if tw_col else 0.0
    tf_val = row.get(tf_col, 0) if tf_col else 0.0

    return BeamSection3D(
        name=row.get("name", "unknown"),
        A=A, Ix=Ix, Iy=Iy, J=J, h=h, b=b,
        j_source=j_source,
        tw=tw_val or 0.0,
        tf=tf_val or 0.0,
    )


# ── 기본 단면 (Supabase 실패 시 fallback) ──
DEFAULT_SECTIONS_3D = {
    "H-300x300": BeamSection3D(
        name="H-300x300", A=11980, Ix=20400e4, Iy=6750e4,
        J=128e4, h=300, b=300, j_source="fallback",
    ),
    "H-400x200": BeamSection3D(
        name="H-400x200", A=8412, Ix=23700e4, Iy=1740e4,
        J=66e4, h=400, b=200, j_source="fallback",
    ),
    "H-200x100": BeamSection3D(
        name="H-200x100", A=2716, Ix=1840e4, Iy=134e4,
        J=8.5e4, h=200, b=100, j_source="fallback",
    ),
}


def find_section_by_dims(
    h_mm: float, b_mm: float,
    tw_mm: float | None = None, tf_mm: float | None = None,
    tolerance_mm: float = 2.0,
) -> tuple[str | None, dict | None]:
    """치수로 표준 H형강 단면명 역조회.

    IFC 프로파일(IfcIShapeProfileDef)에서 추출한 h, b, tw, tf로
    Supabase h_beam_sections에서 단면명을 찾는다.

    Args:
        h_mm: 단면 높이 (mm)
        b_mm: 단면 폭 (mm)
        tw_mm: 웹 두께 (mm), 선택
        tf_mm: 플랜지 두께 (mm), 선택
        tolerance_mm: 허용 오차 (mm)

    Returns:
        (section_name, substitution_info)
        - 정확 매칭: ("H-400x200x8x13", None)
        - 근사 매칭: ("H-400x200", {"original": "H-410x200x9x14",
                                     "substituted": "H-400x200",
                                     "reason": "..."})
        - 실패: (None, None)
    """
    try:
        supabase = _get_supabase()
        schema, table = "ks3502", "h_beam_sections"
        original_desc = f"H-{int(h_mm)}x{int(b_mm)}"
        if tw_mm and tf_mm:
            original_desc = f"H-{int(h_mm)}x{int(b_mm)}x{int(tw_mm)}x{int(tf_mm)}"

        # 1차: h, b 정확 매칭
        result = (
            supabase.schema(schema).table(table)
            .select("name, h, b, t1, t2")
            .gte("h", h_mm - tolerance_mm)
            .lte("h", h_mm + tolerance_mm)
            .gte("b", b_mm - tolerance_mm)
            .lte("b", b_mm + tolerance_mm)
            .execute()
        )

        if result.data:
            candidates = result.data

            # h, b 가장 가까운 순으로 정렬 (정확 매칭 우선)
            candidates.sort(
                key=lambda c: abs(c["h"] - h_mm) + abs(c["b"] - b_mm)
            )

            # tw, tf로 정확 매칭 시도
            if tw_mm is not None and tf_mm is not None:
                exact = [
                    c for c in candidates
                    if (c.get("t1") is not None
                        and abs(c["t1"] - tw_mm) <= tolerance_mm
                        and c.get("t2") is not None
                        and abs(c["t2"] - tf_mm) <= tolerance_mm)
                ]
                if exact:
                    return (exact[0]["name"], None)

                if len(candidates) > 1:
                    # tw/tf 가장 가까운 것 선택
                    best = min(
                        candidates,
                        key=lambda c: (
                            abs((c.get("t1") or 0) - tw_mm)
                            + abs((c.get("t2") or 0) - tf_mm)
                        ),
                    )
                    return (best["name"], {
                        "original": original_desc,
                        "substituted": best["name"],
                        "reason": f"h×b 매칭, tw/tf 최근접 대체 "
                                  f"(IFC: tw={tw_mm}, tf={tf_mm} → "
                                  f"DB: t1={best.get('t1')}, t2={best.get('t2')})",
                    })

            # 단일 결과 또는 tw/tf 미제공 → 가장 가까운 h×b
            return (candidates[0]["name"], None)

        # 2차: h만으로 근접 매칭 (±20mm)
        result2 = (
            supabase.schema(schema).table(table)
            .select("name, h, b, t1, t2")
            .gte("h", h_mm - 20)
            .lte("h", h_mm + 20)
            .order("h")
            .execute()
        )

        if result2.data:
            # b도 가장 가까운 것
            best = min(
                result2.data,
                key=lambda c: abs(c["h"] - h_mm) + abs(c["b"] - b_mm),
            )
            return (best["name"], {
                "original": original_desc,
                "substituted": best["name"],
                "reason": f"정확한 단면 미발견, h×b 기준 최근접 매칭 "
                          f"(IFC: {int(h_mm)}x{int(b_mm)} → "
                          f"DB: {best['h']}x{best['b']})",
            })

        return (None, None)

    except Exception as e:
        print(f"[section_3d] find_section_by_dims 조회 실패: {e}")
        return (None, None)


def get_section_3d(section_name: str) -> Optional[BeamSection3D]:
    """3D 해석용 단면 물성 조회.

    Supabase에서 A, Ix, Iy, J(또는 근사), h, b를 반환.
    실패 시 DEFAULT_SECTIONS_3D fallback.
    """
    try:
        supabase = _get_supabase()

        # prefix 매칭
        matched_prefix = None
        for prefix in sorted(SECTION_3D_MAP.keys(), key=len, reverse=True):
            if section_name.startswith(prefix):
                matched_prefix = prefix
                break

        # L- prefix (ㄱ형강)
        if section_name.startswith("L-"):
            for schema, table, a_col, ix_col, iy_col, h_col, b_col in ANGLE_3D_TABLES:
                row = _query_section_3d(supabase, schema, table, section_name, {})
                if row:
                    sec = _row_to_section_3d(
                        row, ix_col, iy_col, None, h_col, b_col,
                        tw_col="t" if "t" in row else None,
                        tf_col="t" if "t" in row else None,
                    )
                    return sec
            return DEFAULT_SECTIONS_3D.get(section_name)

        # □- prefix (SHS/RHS)
        if section_name.startswith("□-"):
            for schema, table, a_col, ix_col, iy_col, j_col, h_col, b_col in SQUARE_3D_TABLES:
                row = _query_section_3d(supabase, schema, table, section_name, {})
                if row:
                    return _row_to_section_3d(row, ix_col, iy_col, j_col, h_col, b_col)
            return DEFAULT_SECTIONS_3D.get(section_name)

        # FB- (Flat Bar) - 간단히 처리
        if section_name.startswith("FB-"):
            row = _query_section_3d(supabase, "ks3502", "flat_bar_sections", section_name, {})
            if row:
                return _row_to_section_3d(row, "is_max", "iu", None, "a", "t")
            return DEFAULT_SECTIONS_3D.get(section_name)

        if matched_prefix is None:
            return DEFAULT_SECTIONS_3D.get(section_name)

        # 표준 테이블 조회
        schema, table, a_col, ix_col, iy_col, j_col, h_col, b_col, tw_col, tf_col = SECTION_3D_MAP[matched_prefix]
        row = _query_section_3d(supabase, schema, table, section_name, {})
        if row:
            return _row_to_section_3d(row, ix_col, iy_col, j_col, h_col, b_col, tw_col, tf_col)

    except Exception as e:
        print(f"[section_3d] Supabase 조회 실패: {e}")

    # fallback
    return DEFAULT_SECTIONS_3D.get(section_name)
