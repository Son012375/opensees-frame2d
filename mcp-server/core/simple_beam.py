"""
단순보 해석 모듈
OpenSeesPy를 사용한 2D 단순보 정적 해석
Supabase DB 연동
"""
from __future__ import annotations

import openseespy.opensees as ops
from dataclasses import dataclass, field
from typing import Literal, Optional
from supabase import create_client, Client

# Supabase 설정
SUPABASE_URL = "https://kuqoajdqvfsobhhlnyyg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt1cW9hamRxdmZzb2JoaGxueXlnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkzNjA1MjksImV4cCI6MjA4NDkzNjUyOX0.5-rSgPXFrMi4xxdrmxgqZdQnl1U2ta-qDYd2IpLMHzM"

# Supabase 클라이언트 (lazy initialization)
_supabase_client: Optional[Client] = None

def get_supabase() -> Client:
    """Supabase 클라이언트 반환 (싱글톤)"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


@dataclass
class BeamSection:
    """보 단면 정보"""
    name: str
    A: float      # 단면적 (mm²)
    I: float      # 단면2차모멘트 (mm⁴)
    h: float      # 단면 높이 (mm)

    @property
    def Zx(self) -> float:
        """강축 단면계수 (mm³)"""
        return self.I / (self.h / 2) if self.h > 0 else 0.0


@dataclass
class Material:
    """재료 물성"""
    name: str
    E: float      # 탄성계수 (MPa = N/mm²)
    fy: float     # 항복강도 (MPa)


@dataclass
class BeamResult:
    """해석 결과"""
    max_displacement: float     # 최대 처짐 (mm)
    max_moment: float          # 최대 모멘트 (kN·m)
    max_shear: float           # 최대 전단력 (kN)
    reaction_left: float       # 좌측 반력 (kN)
    reaction_right: float      # 우측 반력 (kN)
    max_stress: float          # 최대 응력 (MPa)
    reaction_moment_left: float = 0.0   # 좌측 모멘트 반력 (kN·m)
    reaction_moment_right: float = 0.0  # 우측 모멘트 반력 (kN·m)
    support_type: str = "simple"        # 경계조건 타입
    max_displacement_location: float = 0.0  # 최대 처짐 위치 (m)
    max_moment_location: float = 0.0        # 최대 모멘트 위치 (m)
    max_shear_location: float = 0.0         # 최대 전단력 위치 (m)
    # 단면/재료 정보
    section_name: str = ""
    material_name: str = ""
    # 시각화용 배열
    node_positions: list[float] = field(default_factory=list)  # m
    displacements: list[float] = field(default_factory=list)   # mm
    moments: list[float] = field(default_factory=list)         # kN·m (노드별)
    shears: list[float] = field(default_factory=list)          # kN (노드별)
    # 하중 정보 (시각화용)
    load_info: list[dict] = field(default_factory=list)
    # 새깅/호깅 분리
    max_moment_positive: float = 0.0       # M+(sagging) kN·m
    max_moment_positive_location: float = 0.0
    max_moment_negative: float = 0.0       # M-(hogging) kN·m (음수)
    max_moment_negative_location: float = 0.0
    # 해석 메타정보
    E_MPa: float = 0.0
    Ix_mm4: float = 0.0
    h_mm: float = 0.0
    Zx_mm3: float = 0.0
    fy_MPa: float = 0.0
    num_elements: int = 0
    deflection_limit_ratio: int = 300  # 허용처짐 기준 분모 (L/N)


# 단면 테이블 매핑: (prefix) → (schema, table, area_col, I_col, h_col)
# I_col: 강축 단면2차모멘트, h_col: 단면 높이(응력 계산용)
SECTION_TABLE_MAP = {
    "H-":   ("ks3502", "h_beam_sections",   "area", "ix",     "h"),
    "I-":   ("ks3502", "i_beam_sections",    "area", "ix",     "h"),
    "TFC-": ("ks3502", "tfc_channel_sections", "area", "ix",   "h"),
    "PFC-": ("ks3502", "pfc_channel_sections", "area", "ix",   "h"),
    "T-":   ("ks3502", "t_beam_sections",    "area", "is_val", "h"),
    "L-":   ("ks3502", None,                 "area", None,     None),  # 복수 테이블 → 별도 처리
    "FB-":  ("ks3502", "flat_bar_sections",  "area", "is_max", "a"),
    "○-":   ("ks3568", "chs_sections",       "area", "i",      "d"),
    "□-":   ("ks3568", None,                 "area", None,     None),  # shs or rhs → 별도 처리
}

# ㄱ형강 테이블 목록 (L- prefix, 순서대로 조회 시도)
ANGLE_TABLES = [
    ("ks3502", "equal_angle_sections",                   "area", "ix_max", "a"),
    ("ks3502", "unequal_angle_sections",                 "area", "ix",     "a"),
    ("ks3502", "unequal_leg_thickness_angle_sections",   "area", "ix",     "a"),
]

# □ prefix 테이블 (shs → rhs 순서로 시도)
SQUARE_TABLES = [
    ("ks3568", "shs_sections",         "area", "i",  "b"),
    ("ks3568", "rhs_hollow_sections",  "area", "ix", "h"),
]


# 폴백용 기본 단면 DB (Supabase 연결 실패 시)
DEFAULT_SECTIONS = {
    "H-200x100x5.5x8": BeamSection("H-200x100x5.5x8", A=2716, I=1840e4, h=200),
    "H-300x150x6.5x9": BeamSection("H-300x150x6.5x9", A=4678, I=7210e4, h=300),
    "H-400x200x8x13": BeamSection("H-400x200x8x13", A=8412, I=23700e4, h=400),
    "H-500x200x10x16": BeamSection("H-500x200x10x16", A=11190, I=47800e4, h=500),
}

# 폴백용 기본 재료 DB
DEFAULT_MATERIALS = {
    "SS275": Material("SS275", E=205000, fy=275),
    "SS235": Material("SS235", E=205000, fy=235),
}


def _query_section_table(supabase, schema: str, table: str, section_name: str,
                          area_col: str, i_col: str, h_col: str) -> Optional[BeamSection]:
    """단일 테이블에서 단면 조회 후 BeamSection 반환"""
    query = supabase.schema(schema).table(table).select("*").eq("name", section_name)
    result = query.execute()
    if result.data and len(result.data) > 0:
        row = result.data[0]
        return BeamSection(
            name=row["name"],
            A=row[area_col] * 100,    # cm² → mm²
            I=row[i_col] * 10000,     # cm⁴ → mm⁴
            h=row[h_col]              # mm (h, a, d, b 등 — 이미 mm 단위)
        )
    return None


def get_section_from_db(section_name: str) -> Optional[BeamSection]:
    """Supabase에서 단면 정보 조회 (이름 prefix로 테이블 자동 선택)"""
    try:
        supabase = get_supabase()

        # prefix 매칭 (긴 prefix 먼저 — TFC-, PFC-, FB- 등)
        matched_prefix = None
        for prefix in sorted(SECTION_TABLE_MAP.keys(), key=len, reverse=True):
            if section_name.startswith(prefix):
                matched_prefix = prefix
                break

        if matched_prefix is None:
            return None

        schema, table, area_col, i_col, h_col = SECTION_TABLE_MAP[matched_prefix]

        # L- prefix: 여러 ㄱ형강 테이블 순회
        if matched_prefix == "L-":
            for s, t, ac, ic, hc in ANGLE_TABLES:
                result = _query_section_table(supabase, s, t, section_name, ac, ic, hc)
                if result is not None:
                    return result
            return None

        # □- prefix: shs → rhs 순회
        if matched_prefix == "□-":
            for s, t, ac, ic, hc in SQUARE_TABLES:
                result = _query_section_table(supabase, s, t, section_name, ac, ic, hc)
                if result is not None:
                    return result
            return None

        # 단일 테이블 조회
        return _query_section_table(supabase, schema, table, section_name, area_col, i_col, h_col)

    except Exception as e:
        print(f"Supabase 조회 실패: {e}")
    return None


def get_material_from_db(material_name: str) -> Optional[Material]:
    """Supabase에서 재료 정보 조회 (두께별 다중 행 중 첫 번째 구간 사용)"""
    try:
        supabase = get_supabase()
        result = supabase.schema("ks3502").table("materials").select("*").eq("name", material_name).order("t_min").limit(1).execute()
        if result.data and len(result.data) > 0:
            row = result.data[0]
            return Material(
                name=row["name"],
                E=row["e"],
                fy=row["fy"]
            )
    except Exception as e:
        print(f"Supabase 조회 실패: {e}")
    return None


def analyze_simple_beam(
    span: float,
    load_type: Literal["uniform", "point_center", "point", "triangular", "partial_uniform", "combined"],
    load_value: float = 0.0,
    support_type: Literal["simple", "cantilever", "fixed_fixed", "fixed_pin", "propped_cantilever"] = "simple",
    section_name: str = "H-400x200x8x13",
    material_name: str = "SS275",
    point_location: float | None = None,
    load_start: float | None = None,
    load_end: float | None = None,
    load_value_end: float | None = None,
    loads: list[dict] | None = None,
    num_elements: int = 20,
    deflection_limit: int = 300,
) -> BeamResult:
    """
    단순보 해석 수행

    Parameters
    ----------
    span : float
        스팬 길이 (m)
    load_type : str
        하중 타입 ("uniform": 등분포, "point_center": 중앙 집중, "point": 임의 위치 집중)
    load_value : float
        하중 크기 (uniform: kN/m, point: kN)
    section_name : str
        단면 이름 (기본: H-400x200)
    material_name : str
        재료 이름 (기본: SS275)
    point_location : float, optional
        집중하중 위치 (m), point 타입일 때만 사용
    num_elements : int
        요소 분할 수 (기본: 10)

    Returns
    -------
    BeamResult
        해석 결과 (처짐, 모멘트, 전단력, 반력, 응력)
    """

    # 단면/재료 가져오기 (Supabase 우선, 폴백으로 하드코딩 DB)
    section = get_section_from_db(section_name)
    if section is None:
        section = DEFAULT_SECTIONS.get(section_name)
    if section is None:
        available = get_available_sections()
        raise ValueError(f"Unknown section: {section_name}. Available: {available}")

    material = get_material_from_db(material_name)
    if material is None:
        material = DEFAULT_MATERIALS.get(material_name)
    if material is None:
        available = get_available_materials()
        raise ValueError(f"Unknown material: {material_name}. Available: {available}")

    # 단위 변환: m → mm
    span_mm = span * 1000

    # OpenSees 모델 초기화
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)  # 2D, 3 DOF (Ux, Uy, Rz)

    # 노드 생성
    dx = span_mm / num_elements
    for i in range(num_elements + 1):
        ops.node(i + 1, i * dx, 0.0)

    # 경계조건
    if support_type == "simple":
        ops.fix(1, 1, 1, 0)                    # 좌: 핀 (Ux, Uy 구속)
        ops.fix(num_elements + 1, 0, 1, 0)     # 우: 롤러 (Uy 구속)
    elif support_type == "cantilever":
        ops.fix(1, 1, 1, 1)                    # 좌: 고정단 (Ux, Uy, Rz 구속)
    elif support_type == "fixed_fixed":
        ops.fix(1, 1, 1, 1)                    # 좌: 고정
        ops.fix(num_elements + 1, 1, 1, 1)     # 우: 고정
    elif support_type in ("fixed_pin", "propped_cantilever"):
        ops.fix(1, 1, 1, 1)                    # 좌: 고정
        ops.fix(num_elements + 1, 0, 1, 0)     # 우: 롤러

    # 기하변환
    ops.geomTransf('Linear', 1)

    # 요소 생성
    E = material.E
    A = section.A
    I = section.I

    for i in range(num_elements):
        ops.element('elasticBeamColumn', i + 1, i + 1, i + 2, A, E, I, 1)

    # 하중 적용
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    if load_type == "uniform":
        # 등분포하중 (kN/m → N/mm)
        w = load_value * 1000 / 1000  # kN/m → N/mm
        for i in range(num_elements):
            ops.eleLoad('-ele', i + 1, '-type', '-beamUniform', -w)

    elif load_type == "point_center":
        # 중앙 집중하중
        mid_node = num_elements // 2 + 1
        P = load_value * 1000  # kN → N
        ops.load(mid_node, 0, -P, 0)

    elif load_type == "point":
        # 임의 위치 집중하중
        if point_location is None:
            point_location = span / 2
        loc_mm = point_location * 1000
        node_idx = int(loc_mm / dx) + 1
        node_idx = max(1, min(node_idx, num_elements + 1))
        P = load_value * 1000  # kN → N
        ops.load(node_idx, 0, -P, 0)

    elif load_type == "triangular":
        # 삼각분포하중: load_value(좌측) → load_value_end(우측)
        # 기본: load_value=최대값, load_value_end=0 (좌측 최대 → 우측 0)
        w_start = load_value  # kN/m
        w_end = load_value_end if load_value_end is not None else 0.0  # kN/m
        for i in range(num_elements):
            x_mid = (i + 0.5) * dx
            ratio = x_mid / span_mm
            w_local = w_start + (w_end - w_start) * ratio
            w_nmm = w_local * 1000 / 1000  # kN/m → N/mm
            ops.eleLoad('-ele', i + 1, '-type', '-beamUniform', -w_nmm)

    elif load_type == "partial_uniform":
        # 부분 등분포하중: load_start ~ load_end 구간만
        a_mm = (load_start if load_start is not None else 0.0) * 1000
        b_mm = (load_end if load_end is not None else span) * 1000
        w = load_value * 1000 / 1000  # kN/m → N/mm
        for i in range(num_elements):
            x1 = i * dx
            x2 = (i + 1) * dx
            if x2 > a_mm and x1 < b_mm:
                ops.eleLoad('-ele', i + 1, '-type', '-beamUniform', -w)

    elif load_type == "combined":
        # 조합하중: loads 리스트의 각 항목을 순차 적용
        for ld in (loads or []):
            ld_type = ld.get("type", "point")
            ld_value = ld.get("value", 0.0)
            if ld_type == "uniform":
                w = ld_value * 1000 / 1000  # kN/m → N/mm
                for i in range(num_elements):
                    ops.eleLoad('-ele', i + 1, '-type', '-beamUniform', -w)
            elif ld_type == "point":
                loc = ld.get("location", span / 2) * 1000  # m → mm
                node_idx = int(loc / dx) + 1
                node_idx = max(1, min(node_idx, num_elements + 1))
                P = ld_value * 1000  # kN → N
                ops.load(node_idx, 0, -P, 0)
            elif ld_type == "triangular":
                w_s = ld_value
                w_e = ld.get("value_end", 0.0)
                for i in range(num_elements):
                    x_mid = (i + 0.5) * dx
                    ratio = x_mid / span_mm
                    w_local = w_s + (w_e - w_s) * ratio
                    w_nmm = w_local * 1000 / 1000
                    ops.eleLoad('-ele', i + 1, '-type', '-beamUniform', -w_nmm)
            elif ld_type == "partial_uniform":
                a_mm = ld.get("start", 0.0) * 1000
                b_mm = ld.get("end", span) * 1000
                w = ld_value * 1000 / 1000
                for i in range(num_elements):
                    x1 = i * dx
                    x2 = (i + 1) * dx
                    if x2 > a_mm and x1 < b_mm:
                        ops.eleLoad('-ele', i + 1, '-type', '-beamUniform', -w)

    # 해석 설정
    ops.system('BandGen')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')

    # 해석 수행
    ops.analyze(1)
    ops.reactions()

    # 결과 추출
    # 최대 처짐 및 위치
    displacements = [ops.nodeDisp(i + 1, 2) for i in range(num_elements + 1)]
    min_disp_idx = displacements.index(min(displacements))
    max_disp = min(displacements)  # 아래쪽이 음수
    max_disp_loc = min_disp_idx * dx / 1000  # mm → m

    # 반력 (위로 향하는 반력을 양수로)
    reaction_left = abs(ops.nodeReaction(1, 2)) / 1000  # N → kN
    reaction_right = abs(ops.nodeReaction(num_elements + 1, 2)) / 1000 if support_type != "cantilever" else 0.0
    # 모멘트 반력 (고정단이 있는 경우)
    reaction_moment_left = abs(ops.nodeReaction(1, 3)) / 1e6 if support_type in ("cantilever", "fixed_fixed", "fixed_pin", "propped_cantilever") else 0.0  # N·mm → kN·m
    reaction_moment_right = abs(ops.nodeReaction(num_elements + 1, 3)) / 1e6 if support_type == "fixed_fixed" else 0.0

    # 요소력에서 최대 모멘트/전단력 및 위치 추출
    moments = []
    moment_locs = []
    shears = []
    shear_locs = []
    for i in range(num_elements):
        forces = ops.eleForce(i + 1)
        # elasticBeamColumn 2D: [N1, V1, M1, N2, V2, M2]
        x_start = i * dx / 1000       # m
        x_end = (i + 1) * dx / 1000   # m
        shears.extend([abs(forces[1]) / 1000, abs(forces[4]) / 1000])
        shear_locs.extend([x_start, x_end])
        moments.extend([abs(forces[2]) / 1e6, abs(forces[5]) / 1e6])
        moment_locs.extend([x_start, x_end])

    max_moment = max(moments) if moments else 0.0
    max_moment_idx = moments.index(max_moment) if moments else 0
    max_moment_loc = moment_locs[max_moment_idx] if moment_locs else 0.0

    max_shear = max(shears) if shears else 0.0
    max_shear_idx = shears.index(max_shear) if shears else 0
    max_shear_loc = shear_locs[max_shear_idx] if shear_locs else 0.0

    # 최대 응력 (휨응력)
    y_max = section.h / 2  # mm
    max_stress = (max_moment * 1e6) * y_max / section.I  # MPa

    # 시각화용 배열: 노드별 위치/변위/모멘트/전단력
    node_positions = [i * dx / 1000 for i in range(num_elements + 1)]  # m
    disp_array = [d for d in displacements]  # mm (이미 계산됨)
    # 노드별 모멘트/전단력: 각 노드에서 인접 요소의 평균값 사용
    node_moments = [0.0] * (num_elements + 1)
    node_shears = [0.0] * (num_elements + 1)
    for i in range(num_elements):
        forces = ops.eleForce(i + 1)
        m_i = forces[2] / 1e6    # N·mm → kN·m (부호 유지)
        m_j = -forces[5] / 1e6   # 반대쪽 부호 반전
        v_i = -forces[1] / 1000  # N → kN (부호 유지)
        v_j = forces[4] / 1000
        if i == 0:
            node_moments[0] = m_i
            node_shears[0] = v_i
        node_moments[i + 1] = m_j
        node_shears[i + 1] = v_j

    # 새깅/호깅 분리 (부호 있는 node_moments에서 추출)
    # OpenSees 2D beamColumn: 배열 음수 = sagging(하부인장), 배열 양수 = hogging(상부인장)
    # 구조공학 관례: M+ = sagging, M- = hogging → 부호 반전하여 저장
    sagging = [(-m, node_positions[i]) for i, m in enumerate(node_moments) if m < 0]
    hogging = [(-m, node_positions[i]) for i, m in enumerate(node_moments) if m > 0]
    max_m_pos = max(sagging, key=lambda t: t[0]) if sagging else (0.0, 0.0)
    max_m_neg = min(hogging, key=lambda t: t[0]) if hogging else (0.0, 0.0)

    # 하중 정보 구성 (시각화용)
    _load_info = []
    if load_type == "uniform":
        _load_info.append({"type": "uniform", "value": load_value, "start": 0.0, "end": span})
    elif load_type == "point_center":
        _load_info.append({"type": "point", "value": load_value, "location": span / 2})
    elif load_type == "point":
        _load_info.append({"type": "point", "value": load_value, "location": point_location if point_location else span / 2})
    elif load_type == "triangular":
        _load_info.append({"type": "triangular", "value": load_value, "value_end": load_value_end if load_value_end is not None else 0.0, "start": 0.0, "end": span})
    elif load_type == "partial_uniform":
        _load_info.append({"type": "uniform", "value": load_value, "start": load_start if load_start is not None else 0.0, "end": load_end if load_end is not None else span})
    elif load_type == "combined" and loads:
        for ld in loads:
            lt = ld.get("type", "uniform")
            lv = ld.get("value", 0)
            if lt == "uniform":
                _load_info.append({"type": "uniform", "value": lv, "start": 0.0, "end": span})
            elif lt == "point":
                _load_info.append({"type": "point", "value": lv, "location": ld.get("location", span / 2)})
            elif lt == "triangular":
                _load_info.append({"type": "triangular", "value": lv, "value_end": ld.get("value_end", 0.0), "start": 0.0, "end": span})
            elif lt == "partial_uniform":
                _load_info.append({"type": "uniform", "value": lv, "start": ld.get("start", 0.0), "end": ld.get("end", span)})

    return BeamResult(
        max_displacement=abs(max_disp),
        max_moment=max_moment,
        max_shear=max_shear,
        reaction_left=reaction_left,
        reaction_right=reaction_right,
        max_stress=max_stress,
        reaction_moment_left=reaction_moment_left,
        reaction_moment_right=reaction_moment_right,
        support_type=support_type,
        max_displacement_location=max_disp_loc,
        max_moment_location=max_moment_loc,
        max_shear_location=max_shear_loc,
        section_name=section_name,
        material_name=material_name,
        load_info=_load_info,
        max_moment_positive=max_m_pos[0],
        max_moment_positive_location=max_m_pos[1],
        max_moment_negative=max_m_neg[0],
        max_moment_negative_location=max_m_neg[1],
        E_MPa=material.E,
        Ix_mm4=section.I,
        h_mm=section.h,
        Zx_mm3=section.Zx,
        fy_MPa=material.fy,
        num_elements=num_elements,
        deflection_limit_ratio=deflection_limit,
        node_positions=node_positions,
        displacements=disp_array,
        moments=node_moments,
        shears=node_shears,
    )


def get_available_sections() -> dict[str, list[str]]:
    """사용 가능한 전체 단면 목록 반환 (테이블별 분류)"""
    ALL_TABLES = [
        ("ks3502", "h_beam_sections", "H형강"),
        ("ks3502", "i_beam_sections", "I형강"),
        ("ks3502", "tfc_channel_sections", "경사두께ㄷ형강"),
        ("ks3502", "pfc_channel_sections", "평행플랜지ㄷ형강"),
        ("ks3502", "t_beam_sections", "T형강"),
        ("ks3502", "equal_angle_sections", "등변ㄱ형강"),
        ("ks3502", "unequal_angle_sections", "부등변ㄱ형강"),
        ("ks3502", "unequal_leg_thickness_angle_sections", "부등변부등두께ㄱ형강"),
        ("ks3502", "flat_bar_sections", "구평형강"),
        ("ks3568", "chs_sections", "원형강관"),
        ("ks3568", "shs_sections", "정사각형중공형강"),
        ("ks3568", "rhs_hollow_sections", "직사각형중공형강"),
    ]
    try:
        supabase = get_supabase()
        result_dict = {}
        for schema, table, label in ALL_TABLES:
            result = supabase.schema(schema).table(table).select("name").execute()
            if result.data:
                result_dict[label] = [row["name"] for row in result.data]
        if result_dict:
            return result_dict
    except Exception as e:
        print(f"Supabase 조회 실패: {e}")
    # 폴백
    return {"H형강": list(DEFAULT_SECTIONS.keys())}


def get_available_materials() -> list[str]:
    """사용 가능한 재료 목록 반환 (중복 제거)"""
    try:
        supabase = get_supabase()
        result = supabase.schema("ks3502").table("materials").select("name").order("name").execute()
        if result.data:
            return list(dict.fromkeys([row["name"] for row in result.data]))
    except Exception as e:
        print(f"Supabase 조회 실패: {e}")
    # 폴백
    return list(DEFAULT_MATERIALS.keys())



def get_section_properties(section_name: str) -> dict:
    """단면 정보 조회 (이름 prefix로 테이블 자동 선택, 전체 컬럼 반환)"""
    try:
        supabase = get_supabase()

        # 조회할 테이블 목록 결정
        tables_to_try = []
        for prefix in sorted(SECTION_TABLE_MAP.keys(), key=len, reverse=True):
            if section_name.startswith(prefix):
                schema, table, _, _, _ = SECTION_TABLE_MAP[prefix]
                if prefix == "L-":
                    tables_to_try = [(s, t) for s, t, _, _, _ in ANGLE_TABLES]
                elif prefix == "□-":
                    tables_to_try = [(s, t) for s, t, _, _, _ in SQUARE_TABLES]
                elif table is not None:
                    tables_to_try = [(schema, table)]
                break

        for schema, table in tables_to_try:
            result = supabase.schema(schema).table(table).select("*").eq("name", section_name).execute()
            if result.data and len(result.data) > 0:
                row = result.data[0]
                # 공통 필드 + 원본 데이터 전체 반환
                props = {"source": "supabase"}
                props.update(row)
                # id, created_at 제거
                props.pop("id", None)
                props.pop("created_at", None)
                return props

    except Exception as e:
        print(f"Supabase 조회 실패: {e}")

    # 폴백
    section = DEFAULT_SECTIONS.get(section_name)
    if section is None:
        return {"error": f"Unknown section: {section_name}"}
    return {
        "name": section.name,
        "A_mm2": section.A,
        "Ix_mm4": section.I,
        "h_mm": section.h,
        "source": "local"
    }


def get_material_properties(material_name: str) -> dict:
    """재료 정보 조회 (두께별 전체 행 반환)"""
    try:
        supabase = get_supabase()
        result = supabase.schema("ks3502").table("materials").select("*").eq("name", material_name).order("t_min").execute()
        if result.data and len(result.data) > 0:
            rows = []
            for row in result.data:
                rows.append({
                    "t_min_mm": row["t_min"],
                    "t_max_mm": row["t_max"],
                    "fy_MPa": row["fy"],
                    "fu_min_MPa": row.get("fu_min"),
                    "fu_max_MPa": row.get("fu_max"),
                    "elongation_min": row.get("elongation_min"),
                })
            return {
                "name": result.data[0]["name"],
                "E_MPa": result.data[0]["e"],
                "density_kg_m3": result.data[0].get("density", 7850),
                "standard": result.data[0].get("standard"),
                "thickness_grades": rows,
                "source": "supabase"
            }
    except Exception as e:
        print(f"Supabase 조회 실패: {e}")

    # 폴백
    material = DEFAULT_MATERIALS.get(material_name)
    if material is None:
        return {"error": f"Unknown material: {material_name}"}
    return {
        "name": material.name,
        "E_MPa": material.E,
        "fy_MPa": material.fy,
        "source": "local"
    }
