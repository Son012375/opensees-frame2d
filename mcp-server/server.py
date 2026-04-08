"""
OpenSeesPy MCP Server
구조해석을 위한 Model Context Protocol 서버
"""

import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent
from pydantic import BaseModel, Field
from typing import Literal
import json
import base64
from datetime import datetime

from core.simple_beam import (
    analyze_simple_beam,
    get_available_sections,
    get_available_materials,
    get_section_properties,
    get_material_properties,
)
from core.continuous_beam import analyze_continuous_beam
from core.frame_2d import analyze_frame_2d_multi
from core.visualization import plot_beam_results, plot_beam_results_interactive, plot_frame_2d_multi_interactive
from core.visualization_3d import plot_frame_3d_interactive
from core.verification import verify_frame_equilibrium
from core.verification import verify_equilibrium
from core.kds_loads import query_design_loads, query_load_combinations, query_hazard_values
from core.design_spectrum import compute_design_spectrum
from core.frame_3d import analyze_frame_3d_multi
from core.building_model import BuildingModel
from core.load_generator import generate_all_loads

def _build_enhanced_response(result, input_data, is_continuous=False) -> dict:
    """해석 결과에 input_summary, moment_summary, equilibrium_check, design_check 추가"""
    extra = {}

    # 1) input_summary
    if is_continuous:
        span_config = " + ".join(f"{s} m" for s in input_data.spans)
        support_labels = [chr(65 + i) for i in range(len(input_data.spans) + 1)]
        supports_str = ", ".join(
            f"{label}: {st}" for label, st in zip(support_labels, result.supports)
        )
        loads_desc = []
        for ld in input_data.loads:
            si = ld.get("span_index", "all")
            lt = ld.get("type", "uniform")
            lv = ld.get("value", 0)
            if si == "all" or si is None:
                prefix = "All spans"
            else:
                prefix = f"Span {si + 1}"
            if lt == "uniform":
                loads_desc.append(f"{prefix}: UDL {lv} kN/m")
            elif lt == "point":
                loc = ld.get("location", "mid")
                loads_desc.append(f"{prefix}: Point {lv} kN @ {loc} m")
            else:
                loads_desc.append(f"{prefix}: {lt} {lv}")
        n_elem = getattr(result, 'num_elements_per_span', 20)
    else:
        span_config = f"{input_data.span} m"
        supports_str = f"A: {result.support_type.split('_')[0]}, B: {result.support_type.split('_')[-1] if '_' in result.support_type else 'roller'}"
        lt = input_data.load_type
        lv = input_data.load_value
        if lt == "uniform":
            loads_desc = [f"UDL {lv} kN/m"]
        elif lt in ("point", "point_center"):
            loc = input_data.point_location or input_data.span / 2
            loads_desc = [f"Point {lv} kN @ {loc} m"]
        elif lt == "combined":
            loads_desc = [str(ld) for ld in (input_data.loads or [])]
        else:
            loads_desc = [f"{lt} {lv}"]
        n_elem = getattr(result, 'num_elements', 20)

    extra["input_summary"] = {
        "span_config": span_config,
        "supports": supports_str,
        "loads": loads_desc,
        "section": {
            "name": result.section_name,
            "source": "Supabase",
            "Ix_mm4": result.Ix_mm4,
            "Zx_mm3": round(result.Zx_mm3, 1),
            "h_mm": result.h_mm,
        },
        "material": {
            "name": result.material_name,
            "E_MPa": result.E_MPa,
            "fy_MPa": result.fy_MPa,
        },
        "mesh": {
            "element_type": "elasticBeamColumn (Euler-Bernoulli)",
            "elements_per_span": n_elem,
            "integration": "Linear static (LoadControl)",
            **({"warning": "Low mesh density — consider 20+ elements per span"} if n_elem < 10 else {}),
        },
    }

    # 2) moment_summary
    m_summary = {
        "max_sagging": f"{result.max_moment_positive:.2f} kN·m at {result.max_moment_positive_location:.2f} m",
        "max_hogging": f"{result.max_moment_negative:.2f} kN·m at {result.max_moment_negative_location:.2f} m",
    }
    if is_continuous:
        m_summary["span_table"] = [
            {
                "span": sr["span_index"] + 1,
                "M_pos_kNm": sr.get("max_moment_positive_kNm", 0.0),
                "M_neg_kNm": sr.get("max_moment_negative_kNm", 0.0),
                "V_max_kN": sr["max_shear_kN"],
                "delta_max_mm": sr["max_displacement_mm"],
            }
            for sr in result.span_results
        ]
    extra["moment_summary"] = m_summary

    # 3) equilibrium_check
    try:
        extra["equilibrium_check"] = verify_equilibrium(result)
    except Exception as e:
        extra["equilibrium_check"] = {"error": str(e)}

    # 4) design_check
    if result.Zx_mm3 > 0:
        sigma_max = (result.max_moment * 1e6) / result.Zx_mm3  # MPa
        fy = result.fy_MPa
        extra["design_check"] = {
            "Zx_mm3": round(result.Zx_mm3, 1),
            "sigma_max_MPa": round(sigma_max, 2),
            "fy_MPa": fy,
            "utilization_ratio": round(sigma_max / fy, 3) if fy > 0 else None,
            "safety_factor": round(fy / sigma_max, 2) if sigma_max > 0 else None,
        }

    # 5) support_moments — 내부지점 좌/우 모멘트 (연속보만)
    if is_continuous and hasattr(result, 'reactions'):
        support_labels = [chr(65 + i) for i in range(len(result.reactions))]
        support_moments = []
        for i, r in enumerate(result.reactions):
            m_left = r.get("moment_left_kNm", 0.0)
            m_right = r.get("moment_right_kNm", 0.0)
            # 첫/끝 지점은 한쪽만 의미 있음
            if i == 0:
                m_left = 0.0  # 좌단 왼쪽에는 요소 없음
            if i == len(result.reactions) - 1:
                m_right = 0.0  # 우단 오른쪽에는 요소 없음
            if abs(m_left) > 0.01 or abs(m_right) > 0.01:
                support_moments.append({
                    "support": support_labels[i],
                    "location_m": r["location"],
                    "M_left_kNm": m_left,
                    "M_right_kNm": m_right,
                })
        if support_moments:
            extra["support_moments"] = support_moments

    # 6) deflection_check — 경간별 처짐 판정
    defl_ratio = getattr(result, 'deflection_limit_ratio', 300)
    if is_continuous:
        defl_spans = []
        for sr in result.span_results:
            defl_spans.append({
                "span": sr["span_index"] + 1,
                "L_m": sr["span_length"],
                "delta_max_mm": sr["max_displacement_mm"],
                "delta_allow_mm": sr.get("delta_allow_mm", round(sr["span_length"] * 1000 / defl_ratio, 1)),
                "status": sr.get("deflection_status", "OK"),
            })
        extra["deflection_check"] = {
            "criterion": f"L/{defl_ratio}",
            "spans": defl_spans,
        }
    else:
        span_len = getattr(input_data, 'span', 0.0)
        delta_allow = span_len * 1000 / defl_ratio if defl_ratio > 0 else 0.0
        delta_max = result.max_displacement
        extra["deflection_check"] = {
            "criterion": f"L/{defl_ratio}",
            "spans": [{
                "span": 1,
                "L_m": span_len,
                "delta_max_mm": round(delta_max, 3),
                "delta_allow_mm": round(delta_allow, 1),
                "status": "OK" if delta_max <= delta_allow else "NG",
            }],
        }

    # 7) model_info — 모델 신뢰성 정보
    if is_continuous:
        elems_per_span = [n_elem] * len(input_data.spans)
        total_elems = n_elem * len(input_data.spans)
    else:
        elems_per_span = [n_elem]
        total_elems = n_elem
    model_info = {
        "material": {
            "E_GPa": round(result.E_MPa / 1000, 1),
            "fy_MPa": result.fy_MPa,
        },
        "section": {
            "Ix_mm4": result.Ix_mm4,
            "Zx_mm3": round(result.Zx_mm3, 1),
            "h_mm": result.h_mm,
        },
        "numerical": {
            "element_type": "elasticBeamColumn",
            "elements_per_span": elems_per_span,
            "total_elements": total_elems,
            "load_method": "eleLoad -beamUniform (global Y)",
        },
    }
    if n_elem < 10:
        model_info["warning"] = "Low mesh density may affect displacement accuracy. Consider 20+ elements per span."
    extra["model_info"] = model_info

    return extra


# 시각화 출력 디렉토리
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MCP 서버 생성
server = Server("opensees-mcp")


# Tool 입력 스키마 정의
class SimpleBeamInput(BaseModel):
    span: float = Field(..., description="스팬 길이 (m)")
    load_type: Literal["uniform", "point_center", "point", "triangular", "partial_uniform", "combined"] = Field(
        ..., description="하중 타입: uniform(등분포), point_center(중앙집중), point(임의위치집중), triangular(삼각분포), partial_uniform(부분등분포), combined(조합하중)"
    )
    load_value: float = Field(default=0.0, description="하중 크기 (uniform/triangular: kN/m, point: kN). combined일 때는 0")
    support_type: Literal["simple", "cantilever", "fixed_fixed", "fixed_pin", "propped_cantilever"] = Field(
        default="simple",
        description="경계조건: simple(단순보), cantilever(캔틸레버), fixed_fixed(양단고정), fixed_pin(일단고정-일단핀), propped_cantilever(지지캔틸레버)"
    )
    section_name: str = Field(default="H-400x200x8x13", description="단면 이름. prefix로 단면 종류 구분: H-(H형강), I-(I형강), TFC-(경사두께ㄷ형강), PFC-(평행플랜지ㄷ형강), T-(T형강), L-(ㄱ형강), FB-(구평형강), ○-(원형강관), □-(중공형강). 예: H-400x200x8x13, I-300x150, L-100x100x10, □-200x200x8")
    material_name: str = Field(default="SS275", description="재료 이름 (예: SS275)")
    point_location: float | None = Field(default=None, description="집중하중 위치 (m), point 타입일 때만 사용")
    load_start: float | None = Field(default=None, description="부분하중 시작 위치 (m), partial_uniform 타입일 때 사용")
    load_end: float | None = Field(default=None, description="부분하중 끝 위치 (m), partial_uniform 타입일 때 사용")
    load_value_end: float | None = Field(default=None, description="삼각분포 끝단 하중값 (kN/m), triangular 타입일 때 사용 (기본: 0)")
    loads: list[dict] | None = Field(default=None, description="조합하중 리스트, combined 타입일 때 사용. 예: [{\"type\":\"uniform\",\"value\":5},{\"type\":\"point\",\"value\":30,\"location\":3}]")
    deflection_limit: int = Field(default=300, description="허용처짐 기준 분모 (L/N). 기본 300. 예: 250, 300, 360, 500")


class ContinuousBeamInput(BaseModel):
    spans: list[float] = Field(..., description="각 경간 길이 리스트 (m), 예: [6, 8, 6]. 2~5경간 지원")
    loads: list[dict] = Field(..., description="하중 리스트. 각 항목: {span_index(0-based, 생략시 전경간), type(uniform/point/triangular/partial_uniform), value(kN/m 또는 kN), location(경간 내 m, point용), value_end(삼각분포 끝단), start/end(부분등분포 구간)}")
    supports: list[str] | None = Field(default=None, description="지점 조건 리스트 (len=경간수+1). pin/roller/fixed/free. 기본: 첫지점 pin, 중간 pin, 끝 roller")
    hinges: list[int] | None = Field(default=None, description="내부 힌지를 추가할 지점 인덱스 리스트 (1-based, 중간 지점만 허용). 예: [1]은 지점 B에 힌지, [1,2]는 지점 B와 C에 힌지. 힌지가 있으면 해당 위치에서 모멘트가 전달되지 않음")
    section_name: str = Field(default="H-400x200x8x13", description="단면 이름")
    material_name: str = Field(default="SS275", description="재료 이름")
    deflection_limit: int = Field(default=300, description="허용처짐 기준 분모 (L/N). 기본 300")


class Frame2DInput(BaseModel):
    stories: list[float] = Field(..., description="각 층의 높이 리스트 (m), 아래에서 위로. 예: [3.5, 3.2] = 2층 건물")
    bays: list[float] = Field(..., description="각 경간의 폭 리스트 (m), 왼쪽에서 오른쪽으로. 예: [6.0, 8.0] = 2경간")
    loads: list[dict] | None = Field(default=None, description="(단일 케이스용) 하중 리스트. load_cases와 함께 사용 불가. 각 항목: {type(floor/lateral/nodal), story(1-based), value(kN/m 또는 kN), fx/fy(nodal용)}")
    load_cases: dict[str, list[dict]] | None = Field(default=None, description="(멀티 케이스) 하중케이스 딕셔너리. 예: {\"DL\": [{type:\"floor\",story:1,value:15}], \"EQX\": [{type:\"lateral\",story:3,value:50}]}")
    load_combinations: dict[str, dict[str, float]] | None = Field(default=None, description="하중조합. 예: {\"1.2DL+1.0EQX\": {\"DL\":1.2, \"EQX\":1.0}}")
    supports: Literal["fixed", "pinned"] = Field(default="fixed", description="기초 지점 조건: fixed(고정) 또는 pinned(핀)")
    column_section: str = Field(default="H-300x300", description="기둥 단면")
    beam_section: str = Field(default="H-400x200", description="보 단면")
    material_name: str = Field(default="SS275", description="재료 이름")
    member_releases: dict | None = Field(default=None, description="부재 단부 릴리즈 (힌지). 예: {\"beam\": \"both\"} = 모든 보 양단 핀. 값: \"i\"(시작단), \"j\"(끝단), \"both\"(양단), null(강절)")
    geometric_nonlinearity: str = Field(default="linear", description="기하비선형 해석 옵션: \"linear\"(기본, 1차 선형) 또는 \"pdelta\"(P-Delta 2차 효과 포함)")


class SectionQueryInput(BaseModel):
    section_name: str = Field(..., description="조회할 단면 이름 (예: H-400x200x8x13)")


class MaterialQueryInput(BaseModel):
    material_name: str = Field(..., description="조회할 재료 이름 (예: SS275)")


class DesignLoadInput(BaseModel):
    param_type: str = Field(..., description="하중 유형: dead_load(고정하중), live_load(활하중), snow_load(설하중), wind_load(풍하중), live_reduction(활하중저감), roof_live_reduction(지붕활하중저감), similar_live_load(유사활하중)")
    param_subtype: str | None = Field(default=None, description="세부 유형 (예: unit_weight, distributed, concentrated, base_coefficient 등). 생략 시 전체 조회")
    keyword: str | None = Field(default=None, description="primary_key 검색어 (부분 매칭). 예: 'office'(사무실), 'steel'(강재), 'concrete'(콘크리트)")


class LoadCombinationInput(BaseModel):
    limit_state: str | None = Field(default=None, description="한계상태: 'uls'(극한한계상태) 또는 'sls'(사용한계상태). 생략 시 전체 조회")


class HazardValueInput(BaseModel):
    region_name: str = Field(..., description="지역명 (시/도 또는 시/군/구). 부분 매칭 지원. 예: '서울', '강남구', '부산'")
    hazard_type: str | None = Field(default=None, description="위험계수 유형: 'snow_sg'(지상적설하중) 또는 'wind_v0'(기본풍속). 생략 시 둘 다 반환")


class DesignSpectrumInput(BaseModel):
    region: str = Field(..., description="지역명 (시/군/구). 예: '종로구', '강남구', '부산', '제주시'")
    site_class: str = Field(default="S3", description="지반종류: S1(암반), S2(얕고 단단한), S3(깊고 단단한), S4(깊고 연약한), S5(깊고 연약한 특수)")
    importance_factor: float = Field(default=1.0, description="위험도계수 I. 내진특등급=1.4, I등급=1.2, II등급=1.0")
    damping_ratio: float = Field(default=0.05, description="감쇠비 (기본 5%=0.05)")
    period_end: float = Field(default=5.0, description="스펙트럼 계산 끝 주기 (sec)")
    period_step: float = Field(default=0.01, description="주기 간격 (sec)")


class Frame3DInput(BaseModel):
    stories: list[float] = Field(..., description="각 층의 높이 리스트 (m), 아래에서 위로. 예: [3.5, 3.2] = 2층 건물")
    bays_x: list[float] = Field(..., description="X방향 경간 폭 리스트 (m). 예: [6.0, 8.0] = 2경간")
    bays_y: list[float] = Field(..., description="Y방향 경간 폭 리스트 (m). 예: [6.0] = 1경간")
    load_cases: dict[str, list[dict]] = Field(..., description="하중케이스 딕셔너리. 예: {\"DL\": [{\"type\":\"floor\",\"story\":1,\"value\":15}], \"EQX\": [{\"type\":\"lateral_x\",\"story\":2,\"value\":50}]}")
    load_combinations: dict[str, dict[str, float]] | None = Field(default=None, description="하중조합. 예: {\"1.2DL+1.0EQX\": {\"DL\":1.2, \"EQX\":1.0}}")
    supports: Literal["fixed", "pinned"] = Field(default="fixed", description="기초 지점 조건: fixed(고정) 또는 pinned(핀)")
    column_section: str = Field(default="H-300x300", description="기둥 단면")
    beam_x_section: str = Field(default="H-400x200", description="X방향 보 단면")
    beam_y_section: str = Field(default="H-400x200", description="Y방향 보 단면")
    material_name: str = Field(default="SS275", description="재료 이름")
    num_elements_per_member: int = Field(default=4, description="부재당 요소 분할 수 (기본 4)")
    rigid_diaphragm: bool = Field(default=False, description="강체 다이어프램 적용 (층별 수평면 강성 구속)")
    member_releases: dict | None = Field(default=None, description="부재 단부 릴리즈 (힌지). 예: {\"beam_x\": \"both\"} = X보 양단 핀. 값: \"i\"(시작단), \"j\"(끝단), \"both\"(양단), null(강절)")
    geometric_nonlinearity: str = Field(default="linear", description="기하비선형 해석 옵션: \"linear\"(기본, 1차 선형) 또는 \"pdelta\"(P-Delta 2차 효과 포함)")
    modal_analysis: bool = Field(default=False, description="고유치해석 수행 여부. True 시 rigid_diaphragm이 자동 활성화됩니다. 1~3차 고유주기, 지배방향 등을 반환합니다.")
    story_weights_kN: list[float] | None = Field(default=None, description="층별 중력하중 (kN). 고유치해석 시 질량 산정에 사용. None이면 DL 하중에서 자동 추정.")


class BuildingAnalysisInput(BaseModel):
    config: dict = Field(..., description="""건물 설정 JSON. 필수 키: stories, bays_x, bays_y.

stories 형식: [{"height": 4.0, "usage": "retail", "dead_load_finish": 1.5}, {"height": 3.5, "usage": "office"}]
  - height (필수): 층고 (m)
  - usage: 용도 (office, residential, retail, parking, hospital, school, library, corridor, restaurant, hotel, factory, gym, storage, assembly, roof, balcony)
  - dead_load_finish: 마감재 하중 (kN/m², 기본 1.0)
  - slab_thickness: 슬래브 두께 (m, 기본 0.15)

bays_x: X방향 경간 폭 리스트 (m). 예: [8.0, 8.0]
bays_y: Y방향 경간 폭 리스트 (m). 예: [8.0, 8.0]

선택 키:
  - column_section: 기둥 단면 (기본: H-300x300)
  - beam_x_section: X보 단면 (기본: H-400x200)
  - beam_y_section: Y보 단면 (기본: H-400x200)
  - material_name: 재료명 (기본: SS275)
  - supports: fixed/pinned (기본: fixed)
  - region: 지역명 (예: "서울"). 지정 시 지진/풍하중 자동 생성
  - site_class: 지반종류 S1~S5 (기본: S3)
  - importance: 중요도 등급 특/I/II (기본: II)
  - seismic_system: 내진시스템 (기본: ordinary_moment_frame)
    (special_moment_frame, intermediate_moment_frame, rc_special_moment_frame 등)
  - exposure_category: 풍하중 노풍도 A/B/C/D (기본: B)
  - auto_combinations: 자동 하중조합 생성 (기본: true)
  - rigid_diaphragm: 강체 다이어프램 적용 (기본: false). true면 층별 수평면 강성 구속
  - geometric_nonlinearity: "linear"(기본) 또는 "pdelta"(P-Delta 2차 효과)""")
    ifc_path: str | None = Field(default=None, description="IFC 파일 경로 (미구현, 추후 지원 예정)")


# ── V2 Input Models ──

class ParseIFCV2Input(BaseModel):
    ifc_path: str = Field(..., description="IFC 파일 경로")
    tolerance_mm: float = Field(10.0, description="노드 병합 허용 오차 (mm, 기본 10)")
    default_column_section: str = Field("H-300x300", description="단면 미지정 기둥의 기본 단면")
    default_beam_section: str = Field("H-400x200", description="단면 미지정 보의 기본 단면")
    auto_snap: bool = Field(False, description="True면 보-기둥 접합을 자동 스냅")


class SnapModelJointsInput(BaseModel):
    model_json: dict = Field(..., description="StructuralModel JSON (parse_ifc_v2의 model 출력)")
    snap_tolerance: float = Field(0.5, description="스냅 최대 거리 (m, 기본 0.5)")


class AnalyzeModelV2Input(BaseModel):
    model_json: dict = Field(..., description="StructuralModel JSON")
    load_cases: dict = Field(..., description="""하중 케이스. 예:
{
  "DL": [{"type": "floor_area", "story": 1, "value": 6.3}],
  "EQX": [{"type": "lateral_x", "story": 1, "value": 50.0}]
}
지원 타입: floor_area(kN/m²), lateral_x/lateral_y(kN), nodal(직접)""")
    load_combinations: Optional[dict] = Field(None, description='하중 조합. 예: {"1.2DL+EQX": {"DL": 1.2, "EQX": 1.0}}')


class ResolveBuildingConfigInput(BaseModel):
    intent: dict = Field(..., description="""Claude가 자연어에서 추출한 건물 설계 의도 JSON.

필수 키:
  stories: 층별 용도 의도 리스트.
    각 항목: {floor_start: int, floor_end: int, usage_raw: str, height: float|null}
    - floor_start/floor_end: 층 범위 (1-based, inclusive). 단일층이면 동일 값.
    - usage_raw: 사용자 원문 (한국어). 예: "근린생활시설", "오피스", "기계실"
    - height: 층고 (m). 미언급 시 null → 1층 4.0m, 기준층 3.5m, 기계실 3.0m 적용

선택 키:
  num_stories: 총 층수 (stories 범위에서 추론 가능하면 생략)
  bays_x: X방향 경간 [m] 리스트. 예: [8.0, 8.0]
  bays_y: Y방향 경간 [m] 리스트
  num_bays_x: X방향 경간 수 (bays_x 대신 사용 가능)
  num_bays_y: Y방향 경간 수
  typical_bay_width: 대표 경간 폭 (m). 기본 8.0
  region_raw: 지역명 원문. 예: "부산 해운대", "서울 강남"
  site_class: S1~S5
  importance: 특/I/II
  seismic_system: 내진시스템 키 (special_moment_frame, ordinary_moment_frame 등)
  exposure_category: A/B/C/D
  column_section, beam_x_section, beam_y_section, material_name, supports
  rigid_diaphragm, geometric_nonlinearity

예시:
{
  "stories": [
    {"floor_start": 1, "floor_end": 1, "usage_raw": "근린생활시설"},
    {"floor_start": 2, "floor_end": 5, "usage_raw": "오피스"},
    {"floor_start": 6, "floor_end": 6, "usage_raw": "기계실"}
  ],
  "region_raw": "부산 해운대",
  "bays_x": [8.0, 8.0],
  "bays_y": [8.0]
}""")


# Tool 목록 정의
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="analyze_simple_beam",
            description="""보 정적 해석을 수행합니다. 다양한 경계조건과 하중 타입을 지원합니다.

입력:
- span: 스팬 길이 (m)
- load_type: 하중 타입
  - uniform: 등분포하중
  - point_center: 중앙 집중하중
  - point: 임의 위치 집중하중 (point_location으로 위치 지정)
  - triangular: 삼각분포하중 (load_value=좌측값, load_value_end=우측값)
  - partial_uniform: 부분 등분포하중 (load_start-load_end 구간)
  - combined: 조합하중 (loads 리스트로 여러 하중 동시 적용)
- load_value: 하중 크기 (등분포/삼각: kN/m, 집중: kN)
- support_type: 경계조건 (기본: simple)
  - simple: 단순보 (핀-롤러)
  - cantilever: 캔틸레버 (좌측 고정, 우측 자유)
  - fixed_fixed: 양단 고정
  - fixed_pin: 일단 고정-일단 핀
  - propped_cantilever: 지지 캔틸레버
- section_name: 단면 (기본: H-400x200x8x13). 이름 prefix로 종류 구분:
  H-(H형강), I-(I형강), TFC-(ㄷ형강), PFC-(ㄷ형강), T-(T형강),
  L-(ㄱ형강), FB-(구평형강), ○-(원형강관), □-(중공형강)
- material_name: 재료 (기본: SS275)

출력:
- 최대 처짐, 최대 모멘트, 최대 전단력, 지점 반력(모멘트 반력 포함), 최대 응력""",
            inputSchema=SimpleBeamInput.model_json_schema(),
        ),
        Tool(
            name="analyze_continuous_beam",
            description="""다경간 연속보 정적 해석을 수행합니다 (2~5경간).

입력:
- spans: 각 경간 길이 리스트 (m), 예: [6, 8, 6]
- loads: 하중 리스트. 각 항목:
  - span_index (int, optional): 적용 경간 (0-based). 생략 시 전 경간 적용
  - type: uniform/point/triangular/partial_uniform
  - value: 하중 크기 (kN/m 또는 kN)
  - location: 집중하중 위치 (경간 내 m, point용)
  - value_end: 삼각분포 끝단값 (kN/m)
  - start, end: 부분등분포 구간 (경간 내 m)
- supports: 지점 조건 리스트 (pin/roller/fixed/free). 기본: 첫 pin + 중간 pin + 끝 roller
- hinges: 내부 힌지 지점 인덱스 리스트 (1-based). 예: [1]은 지점 B에 힌지 (Gerber보)
  힌지가 있으면 해당 위치에서 모멘트가 0이 됨 (회전 자유)
- section_name, material_name

출력:
- 전체 최대 처짐/모멘트/전단력/응력
- 지점별 반력 (수직, 모멘트, 힌지 여부)
- 경간별 최대 처짐/모멘트/전단력
- 힌지 위치 정보""",
            inputSchema=ContinuousBeamInput.model_json_schema(),
        ),
        Tool(
            name="analyze_frame_2d",
            description="""2D 골조(프레임) 정적 해석을 수행합니다. 멀티 하중케이스 및 하중조합을 지원합니다.

입력:
- stories: 각 층 높이 리스트 (m), 아래→위. 예: [3.5, 3.2] = 2층
- bays: 각 경간 폭 리스트 (m), 좌→우. 예: [6.0, 8.0] = 2경간
- loads: (단일 케이스) 하중 리스트. load_cases와 동시 사용 불가
  - type: "floor" (층 등분포), "lateral" (횡하중), "nodal" (절점하중)
  - story: 적용 층 (1-based)
  - value: 하중 크기 (floor: kN/m, lateral: kN)
- load_cases: (멀티 케이스) 하중케이스 딕셔너리
  예: {"DL": [{"type":"floor","story":1,"value":15}], "EQX": [{"type":"lateral","story":3,"value":50}]}
- load_combinations: 하중조합 (선형중첩)
  예: {"1.2DL+1.0EQX": {"DL":1.2, "EQX":1.0}}
- supports: 기초 조건 ("fixed" 또는 "pinned")
- column_section: 기둥 단면 (기본: H-300x300)
- beam_section: 보 단면 (기본: H-400x200)

출력:
- 케이스/조합별: 노드 변위, 층간변위각, 요소력, 지점 반력, 최대값
- 부재력 다이어그램 (N/V/M)
- 층별 분석 (변위 프로파일, 층전단력)
- 평형검증 (ΣFx, ΣFy, ΣM)
- 탭 기반 인터랙티브 HTML 시각화
- CSV 내보내기""",
            inputSchema=Frame2DInput.model_json_schema(),
        ),
        Tool(
            name="get_section_properties",
            description="표준 단면의 단면 특성을 조회합니다. H형강, I형강, ㄷ형강(TFC/PFC), ㄱ형강, T형강, 구평형강, 원형강관, 정사각형/직사각형 중공형강을 지원합니다.",
            inputSchema=SectionQueryInput.model_json_schema(),
        ),
        Tool(
            name="get_material_properties",
            description="표준 재료(SS275, SM355 등)의 물성치(E, fy)를 조회합니다. 두께별 항복강도를 반환합니다.",
            inputSchema=MaterialQueryInput.model_json_schema(),
        ),
        Tool(
            name="list_available_sections",
            description="사용 가능한 전체 단면 목록을 반환합니다. 12개 테이블(H형강, I형강, ㄷ형강, ㄱ형강, T형강, 구평형강, 원형강관, 중공형강 등) 약 700개 단면.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_materials",
            description="사용 가능한 재료 목록을 반환합니다.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_design_loads",
            description="""KDS 설계하중 파라미터를 조회합니다 (Supabase DB, 454건+).

입력:
- param_type (필수): 하중 유형
  - dead_load: 고정하중 (재료 단위중량, 마감재 중량)
  - live_load: 활하중 (용도별 등분포/집중하중)
  - snow_load: 설하중 계수 (Cb, Ce, Ct, Is)
  - wind_load: 풍하중 계수 (외압/내압/풍력계수)
  - live_reduction: 활하중 저감계수
  - roof_live_reduction: 지붕활하중 저감계수
  - similar_live_load: 유사활하중 (난간, 칸막이벽, 차량방호)
- param_subtype (선택): 세부 유형 필터
  - dead_load → unit_weight, density, material_weight, finishing_weight 등
  - live_load → distributed, concentrated, heavy_vehicle
  - snow_load → Cb, Ce, Ct, Is
- keyword (선택): primary_key 부분 매칭 검색어 (예: 'office', 'concrete')

출력:
- 매칭 레코드: display_name_ko, value, unit, conditions, 출처(code_id, clause_id) 등

예시:
- 사무실 활하중: param_type="live_load", keyword="office"
- 콘크리트 단위중량: param_type="dead_load", keyword="concrete"
- 설하중 노출계수: param_type="snow_load", param_subtype="Ce"
- 풍하중 외압계수: param_type="wind_load", param_subtype="external_pressure_wall" """,
            inputSchema=DesignLoadInput.model_json_schema(),
        ),
        Tool(
            name="get_load_combinations",
            description="""KDS 하중조합식을 조회합니다 (KDS 41 12 00 §1.7, 46건).

입력:
- limit_state (선택): 'uls' (극한한계상태, 23건) 또는 'sls' (사용한계상태, 23건). 생략 시 전체 반환.

출력:
- 하중조합 목록: combo_id, 한국어/영어 명칭, 적용조건, 출처

예시:
- 극한한계상태 조합: limit_state="uls"
- 사용한계상태 조합: limit_state="sls" """,
            inputSchema=LoadCombinationInput.model_json_schema(),
        ),
        Tool(
            name="get_hazard_values",
            description="""지역별 지상적설하중(Sg)과 기본풍속(V₀)을 조회합니다 (458건, 229개 시/군/구).

입력:
- region_name (필수): 지역명. 시/도 또는 시/군/구 이름 (부분 매칭). 예: '서울', '강남구', '부산'
- hazard_type (선택): 'snow_sg' (지상적설하중, kN/m²) 또는 'wind_v0' (기본풍속, m/s). 생략 시 둘 다.

출력:
- 매칭 지역별: region_sido, region_sigungu, Sg (kN/m²), V₀ (m/s)

예시:
- 서울 전체: region_name="서울"
- 강남구 풍속: region_name="강남구", hazard_type="wind_v0"
- 부산 적설하중: region_name="부산", hazard_type="snow_sg" """,
            inputSchema=HazardValueInput.model_json_schema(),
        ),
        Tool(
            name="get_design_spectrum",
            description="""KDS 17 10 00 설계응답스펙트럼을 계산합니다.

지역명과 지반종류를 입력하면 가속도 표준설계응답스펙트럼 Sa(T) 곡선을 생성합니다.
Supabase DB에서 구역계수(z), 지반증폭계수(Fa, Fv)를 조회하여 계산합니다.

입력:
- region (필수): 지역명 (시/군/구). 예: '종로구', '강남구', '부산', '제주시'
- site_class: 지반종류 S1~S5 (기본 S3)
- importance_factor: 위험도계수 I (기본 1.0). 내진특등급=1.4, I등급=1.2, II등급=1.0
- damping_ratio: 감쇠비 (기본 0.05 = 5%)
- period_end: 스펙트럼 끝 주기 (기본 5.0초)
- period_step: 주기 간격 (기본 0.01초)

출력:
- spectrum: {periods, Sa, unit, count} - 스펙트럼 곡선 데이터
- parameters: {z, Fa, Fv, SDS, SD1, T0, Ts, TL, ...} - 스펙트럼 정의 파라미터
- pga: 재현주기별 PGA 정보 (있는 경우)
- opensees_input: OpenSeesPy responseSpectrumAnalysis 입력용 데이터

근거: KDS 17 10 00 §4.2.1.4, 표 4.2-5, 표 4.2-6""",
            inputSchema=DesignSpectrumInput.model_json_schema(),
        ),
        Tool(
            name="analyze_frame_3d",
            description="""3D 골조(프레임) 정적 해석을 수행합니다. X/Y 양방향 다경간, 멀티 하중케이스 및 하중조합을 지원합니다.

입력:
- stories: 각 층 높이 리스트 (m), 아래→위. 예: [3.5, 3.2] = 2층
- bays_x: X방향 경간 폭 리스트 (m). 예: [6.0, 8.0] = 2경간
- bays_y: Y방향 경간 폭 리스트 (m). 예: [6.0] = 1경간
- load_cases (필수): 하중케이스 딕셔너리
  하중 유형:
  - floor: 층 보에 등분포하중 (kN/m). story의 모든 보에 적용
  - floor_area: 바닥 면적하중 (kN/m²). tributary width로 보 선하중 변환
  - lateral_x: X방향 횡하중 (kN). story 노드에 균등 분배
  - lateral_y: Y방향 횡하중 (kN). story 노드에 균등 분배
  - nodal: 절점하중 (node, fx, fy, fz, mx, my, mz)
  예: {"DL": [{"type":"floor","story":1,"value":15}], "EQX": [{"type":"lateral_x","story":2,"value":100}]}
- load_combinations: 하중조합 (선형중첩)
  예: {"1.2DL+1.0EQX": {"DL":1.2, "EQX":1.0}}
- supports: 기초 조건 ("fixed" 또는 "pinned")
- column_section: 기둥 단면 (기본: H-300x300)
- beam_x_section: X방향 보 단면 (기본: H-400x200)
- beam_y_section: Y방향 보 단면 (기본: H-400x200)

좌표계: X=수평(bay_x), Y=수평(bay_y), Z=수직(높이)

출력:
- 케이스/조합별: 6-DOF 노드 변위, 12성분 요소력, 6성분 반력
- X/Y 양방향 층간변위각
- 부재력 다이어그램 (N, Vy, Vz, T, My, Mz)""",
            inputSchema=Frame3DInput.model_json_schema(),
        ),
        Tool(
            name="analyze_building",
            description="""건물 자동 해석: JSON 설정 → 하중 자동 생성 → 3D 프레임 해석.

건물의 기하 정보와 위치/용도를 입력하면:
1. 층별 용도에 따라 DB에서 설계하중 자동 조회 (KDS 41 12 00)
2. 지역 정보로 지진하중 자동 계산 (KDS 41 17 00 등가정적해석법)
3. 기본풍속 조회 + 풍하중 자동 계산 (KDS 41 12 00 §5)
4. KDS 표준 하중조합 자동 생성 (18개 조합)
5. OpenSeesPy 3D 프레임 해석 수행

생성되는 하중케이스:
  - DL: 고정하중 (슬래브 자중 + 마감재 + 설비)
  - LL: 활하중 (용도별 DB 조회: 사무실 2.5, 소매점 5.0 kN/m² 등)
  - EQX/EQY: 등가정적 지진하중 (region 지정 시)
  - WX/WY: 풍하중 (region 지정 시)

예시 config:
{
  "stories": [{"height":4.0,"usage":"retail"},{"height":3.5,"usage":"office"},{"height":3.5,"usage":"office"}],
  "bays_x": [8.0, 8.0],
  "bays_y": [8.0, 8.0],
  "region": "서울",
  "site_class": "S3",
  "importance": "II"
}""",
            inputSchema=BuildingAnalysisInput.model_json_schema(),
        ),
        Tool(
            name="resolve_building_config",
            description="""자연어 건물 설계 의도 → 구조 해석 config 변환 (검증 포함).

사용자의 자연어 입력에서 Claude가 추출한 건물 설계 의도(BuildingIntent)를
검증된 analyze_building config로 변환합니다.

주요 기능:
1. 한국어 용도명 → 표준 occupancy key 매핑 (30개 DB 키 + 확장 별칭)
   예: "근린생활시설"→retail, "기계실"→mechanical_room, "오피스"→office
2. 지역명 → Supabase 229개 시군구 fuzzy 매칭
   예: "부산 해운대"→부산광역시 해운대구
3. 층 범위 → 개별 층 config 확장
   예: floor_start=2, floor_end=5 → 4개 개별 층
4. 미지정 파라미터 기본값 채움 + 가정 추적
5. 경고 생성 (복합 매핑, 미매핑, 기본값 등)

응답:
- status: "resolved" (바로 해석 가능) 또는 "needs_clarification" (사용자 확인 필요)
- config: analyze_building에 전달할 최종 config
- resolution_report: 매핑/변환 상세
- warnings: 주의사항
- clarification_needed: 사용자에게 물어볼 질문 (지역 모호성, 용도 미매핑 등)

2단계 워크플로우:
  1. resolve_building_config → config 생성 + 사용자 확인
  2. analyze_building(config) → 해석 실행""",
            inputSchema=ResolveBuildingConfigInput.model_json_schema(),
        ),

        # ── V2 Tools ──
        Tool(
            name="parse_ifc_v2",
            description="""IFC 파일을 노드-요소 기반 구조 모델로 변환합니다 (V2 파이프라인).

V1(격자 기반)과 달리 IFC에서 부재의 시작/끝 좌표를 직접 추출하여
비정형 건물(경사 부재, 불규칙 평면, setback)을 자연스럽게 표현합니다.

동작:
1. IFC 검증: 구조 부재(IfcColumn/IfcBeam/IfcMember) 존재 확인
2. 좌표 추출: ifcopenshell.geom으로 글로벌 끝점 좌표 계산
3. 노드 병합: tolerance 기준 근접 노드 통합
4. 층 감지: IfcBuildingStorey 또는 Z좌표 클러스터링
5. 검증 리포트: 누락 단면, 재료, 슬래브 등 WARNING 생성

응답:
- model: StructuralModel JSON (노드, 요소, 층 정보)
- validation: 검증 결과 (CRITICAL/WARNING/INFO)
- needs_user_input: 사용자 확인 필요 항목 (슬래브 두께, 보-기둥 스냅 등)

다음 단계: snap_model_joints → analyze_model_v2""",
            inputSchema=ParseIFCV2Input.model_json_schema(),
        ),
        Tool(
            name="snap_model_joints",
            description="""V2 모델의 보-기둥 접합부를 스냅합니다.

IFC에서 보의 끝점이 기둥 노드와 미세하게 떨어져 있는 경우
(단면 오프셋 등), 보 끝점을 가장 가까운 기둥 노드에 병합합니다.

parse_ifc_v2에서 IFC_DISCONNECTED_JOINTS 경고가 나온 경우 호출하세요.
사용자에게 스냅 여부를 확인한 후 호출하는 것을 권장합니다.""",
            inputSchema=SnapModelJointsInput.model_json_schema(),
        ),
        Tool(
            name="analyze_model_v2",
            description="""V2 StructuralModel에서 직접 3D 구조 해석을 수행합니다.

노드-요소 기반 자유 그래프 모델을 OpenSees로 해석합니다.
격자(bays_x/y) 제약 없이 비정형 건물도 해석 가능합니다.

지원:
- 요소별 개별 단면/재료/릴리즈
- 경사 브레이스 (brace 요소)
- 하중: floor_area(kN/m²), lateral_x/y(kN), nodal(6-DOF)
- 하중조합 (선형 중첩)
- 강체 다이어프램, P-Delta

응답: 케이스별 변위, 부재력, 반력, 층간변위 + HTML 리포트""",
            inputSchema=AnalyzeModelV2Input.model_json_schema(),
        ),
    ]


# Tool 호출 핸들러
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "analyze_simple_beam":
            # 입력 검증
            input_data = SimpleBeamInput(**arguments)

            # 해석 수행
            result = analyze_simple_beam(
                span=input_data.span,
                load_type=input_data.load_type,
                load_value=input_data.load_value,
                support_type=input_data.support_type,
                section_name=input_data.section_name,
                material_name=input_data.material_name,
                point_location=input_data.point_location,
                load_start=input_data.load_start,
                load_end=input_data.load_end,
                load_value_end=input_data.load_value_end,
                loads=input_data.loads,
                deflection_limit=input_data.deflection_limit,
            )

            # 시각화 생성
            diagram_png = None
            diagram_html = None
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                diagram_png = plot_beam_results(result, output_path=os.path.join(OUTPUT_DIR, f"simple_{ts}.png"))
                diagram_html = plot_beam_results_interactive(result, output_path=os.path.join(OUTPUT_DIR, f"simple_{ts}.html"))
            except Exception as viz_err:
                print(f"시각화 생성 실패: {viz_err}")

            # 결과 포맷팅
            response = {
                "status": "success",
                "input": {
                    "span": f"{input_data.span} m",
                    "load_type": input_data.load_type,
                    "load_value": f"{input_data.load_value} {'kN/m' if input_data.load_type == 'uniform' else 'kN'}",
                    "section": input_data.section_name,
                    "material": input_data.material_name,
                },
                "results": {
                    "support_type": result.support_type,
                    "max_displacement": f"{result.max_displacement:.3f} mm",
                    "max_displacement_location": f"{result.max_displacement_location:.2f} m",
                    "max_moment": f"{result.max_moment:.2f} kN·m",
                    "max_moment_location": f"{result.max_moment_location:.2f} m",
                    "max_shear": f"{result.max_shear:.2f} kN",
                    "max_shear_location": f"{result.max_shear_location:.2f} m",
                    "reaction_left": f"{result.reaction_left:.2f} kN",
                    "reaction_right": f"{result.reaction_right:.2f} kN",
                    **({"reaction_moment_left": f"{result.reaction_moment_left:.2f} kN·m"} if result.reaction_moment_left > 0 else {}),
                    **({"reaction_moment_right": f"{result.reaction_moment_right:.2f} kN·m"} if result.reaction_moment_right > 0 else {}),
                    "max_stress": f"{result.max_stress:.2f} MPa",
                },
            }
            if diagram_png and os.path.exists(diagram_png):
                response["diagram_png"] = diagram_png
            if diagram_html and os.path.exists(diagram_html):
                response["diagram_html"] = diagram_html
            # 확장 응답 추가
            response.update(_build_enhanced_response(result, input_data, is_continuous=False))
            contents = [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]
            if diagram_png and os.path.exists(diagram_png):
                with open(diagram_png, "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                contents.append(ImageContent(type="image", data=b64, mimeType="image/png"))
            return contents

        elif name == "analyze_continuous_beam":
            input_data = ContinuousBeamInput(**arguments)
            result = analyze_continuous_beam(
                spans=input_data.spans,
                loads=input_data.loads,
                supports=input_data.supports,
                hinges=input_data.hinges,
                section_name=input_data.section_name,
                material_name=input_data.material_name,
                deflection_limit=input_data.deflection_limit,
            )
            # 시각화 생성
            diagram_png = None
            diagram_html = None
            nodal_csv = None
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                diagram_png = plot_beam_results(result, output_path=os.path.join(OUTPUT_DIR, f"continuous_{ts}.png"))
                diagram_html = plot_beam_results_interactive(result, output_path=os.path.join(OUTPUT_DIR, f"continuous_{ts}.html"))
                # nodal_results CSV 저장
                nodal_csv = os.path.join(OUTPUT_DIR, f"continuous_{ts}_nodal.csv")
                with open(nodal_csv, "w", encoding="utf-8") as f:
                    f.write("node,x_m,DY_mm,RZ_rad,M_kNm,V_kN\n")
                    for i in range(len(result.node_positions)):
                        f.write(f"{i+1},{result.node_positions[i]:.6f},{result.displacements[i]:.6f},{result.rotations[i]:.6f},{result.moments[i]:.6f},{result.shears[i]:.6f}\n")
            except Exception as viz_err:
                print(f"시각화 생성 실패: {viz_err}")

            response = {
                "status": "success",
                "input": {
                    "spans": [f"{s} m" for s in input_data.spans],
                    "num_spans": len(input_data.spans),
                    "section": input_data.section_name,
                    "material": input_data.material_name,
                    **({"hinges": input_data.hinges} if input_data.hinges else {}),
                },
                "results": {
                    "total_length": f"{result.total_length:.2f} m",
                    "supports": result.supports,
                    **({"hinge_locations": [f"{loc:.2f} m" for loc in result.hinge_locations]} if result.hinge_locations else {}),
                    "max_displacement": f"{result.max_displacement:.3f} mm",
                    "max_displacement_location": f"{result.max_displacement_location:.2f} m",
                    "max_moment": f"{result.max_moment:.2f} kN·m",
                    "max_moment_location": f"{result.max_moment_location:.2f} m",
                    "max_shear": f"{result.max_shear:.2f} kN",
                    "max_shear_location": f"{result.max_shear_location:.2f} m",
                    "max_stress": f"{result.max_stress:.2f} MPa",
                    "reactions": result.reactions,
                    "span_results": result.span_results,
                    "nodal_results": [
                        {
                            "node": i + 1,
                            "x_m": round(result.node_positions[i], 3),
                            "DY_mm": round(result.displacements[i], 6),
                            "RZ_rad": round(result.rotations[i], 6),
                            "M_kNm": round(result.moments[i], 3),
                            "V_kN": round(result.shears[i], 3),
                        }
                        for i in range(len(result.node_positions))
                    ],
                },
            }
            if diagram_png and os.path.exists(diagram_png):
                response["diagram_png"] = diagram_png
            if diagram_html and os.path.exists(diagram_html):
                response["diagram_html"] = diagram_html
            if nodal_csv and os.path.exists(nodal_csv):
                response["nodal_csv"] = nodal_csv
            # 확장 응답 추가
            response.update(_build_enhanced_response(result, input_data, is_continuous=True))
            contents = [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]
            if diagram_png and os.path.exists(diagram_png):
                with open(diagram_png, "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                contents.append(ImageContent(type="image", data=b64, mimeType="image/png"))
            return contents

        elif name == "analyze_frame_2d":
            input_data = Frame2DInput(**arguments)

            # 하중케이스 결정: load_cases > loads (하위호환)
            if input_data.load_cases:
                load_cases = input_data.load_cases
            elif input_data.loads:
                load_cases = {"LC1": input_data.loads}
            else:
                raise ValueError("loads 또는 load_cases 중 하나는 필수입니다.")

            # 멀티케이스 해석
            multi = analyze_frame_2d_multi(
                stories=input_data.stories,
                bays=input_data.bays,
                load_cases=load_cases,
                supports=input_data.supports,
                column_section=input_data.column_section,
                beam_section=input_data.beam_section,
                material_name=input_data.material_name,
                load_combinations=input_data.load_combinations,
                member_releases=input_data.member_releases,
                geometric_nonlinearity=input_data.geometric_nonlinearity,
            )

            # 평형검증 (케이스별)
            eq_checks = {}
            for case_name, case_loads in load_cases.items():
                cr = multi.case_results.get(case_name)
                if cr:
                    try:
                        eq_checks[case_name] = verify_frame_equilibrium(
                            cr, case_loads, input_data.stories, input_data.bays,
                        )
                    except Exception:
                        pass

            # 시각화 생성
            diagram_html = None
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                diagram_html = plot_frame_2d_multi_interactive(
                    multi,
                    equilibrium_checks=eq_checks,
                    output_path=os.path.join(OUTPUT_DIR, f"frame2d_multi_{ts}.html"),
                )
            except Exception as viz_err:
                print(f"프레임 시각화 생성 실패: {viz_err}")

            # 케이스별 결과 요약
            def _case_summary(cr):
                return {
                    "max_displacement_x": f"{cr.max_displacement_x:.3f} mm",
                    "max_displacement_x_node": cr.max_displacement_x_node,
                    "max_displacement_y": f"{cr.max_displacement_y:.3f} mm",
                    "max_displacement_y_node": cr.max_displacement_y_node,
                    "max_drift": f"{cr.max_drift:.6f} rad",
                    "max_drift_story": cr.max_drift_story,
                    "max_moment": f"{cr.max_moment:.2f} kN·m",
                    "max_moment_element": cr.max_moment_element,
                    "max_axial": f"{cr.max_axial:.2f} kN",
                    "max_axial_element": cr.max_axial_element,
                    "max_shear": f"{cr.max_shear:.2f} kN",
                    "max_shear_element": cr.max_shear_element,
                    "reactions": cr.reactions,
                }

            results_by_case = {}
            for cn, cr in multi.case_results.items():
                results_by_case[cn] = _case_summary(cr)

            results_by_combo = {}
            for cn, cr in multi.combo_results.items():
                results_by_combo[cn] = _case_summary(cr)

            response = {
                "status": "success",
                "input": {
                    "stories": [f"{s} m" for s in input_data.stories],
                    "bays": [f"{b} m" for b in input_data.bays],
                    "num_stories": multi.num_stories,
                    "num_bays": multi.num_bays,
                    "column_section": input_data.column_section,
                    "beam_section": input_data.beam_section,
                    "material": input_data.material_name,
                    "supports": input_data.supports,
                },
                "geometry": {
                    "total_height": f"{multi.total_height:.2f} m",
                    "total_width": f"{multi.total_width:.2f} m",
                    "num_elements": multi.num_elements,
                },
                "load_cases": list(multi.case_results.keys()),
                "load_combinations": list(multi.combo_results.keys()),
                "results_by_case": results_by_case,
                "results_by_combo": results_by_combo,
                "equilibrium_check": eq_checks,
                "analysis_metadata": getattr(multi, 'analysis_metadata', {}),
            }
            if diagram_html and os.path.exists(diagram_html):
                response["diagram_html"] = diagram_html
            return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        elif name == "get_section_properties":
            input_data = SectionQueryInput(**arguments)
            result = get_section_properties(input_data.section_name)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "get_material_properties":
            input_data = MaterialQueryInput(**arguments)
            result = get_material_properties(input_data.material_name)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "list_available_sections":
            sections = get_available_sections()
            return [TextContent(type="text", text=json.dumps(sections, ensure_ascii=False))]

        elif name == "list_available_materials":
            materials = get_available_materials()
            return [TextContent(type="text", text=json.dumps({"materials": materials}, ensure_ascii=False))]

        elif name == "get_design_loads":
            input_data = DesignLoadInput(**arguments)
            result = query_design_loads(
                param_type=input_data.param_type,
                param_subtype=input_data.param_subtype,
                keyword=input_data.keyword,
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "get_load_combinations":
            input_data = LoadCombinationInput(**arguments)
            result = query_load_combinations(
                limit_state=input_data.limit_state,
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "get_hazard_values":
            input_data = HazardValueInput(**arguments)
            result = query_hazard_values(
                region_name=input_data.region_name,
                hazard_type=input_data.hazard_type,
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "get_design_spectrum":
            input_data = DesignSpectrumInput(**arguments)
            result = compute_design_spectrum(
                region=input_data.region,
                site_class=input_data.site_class,
                importance_factor=input_data.importance_factor,
                damping_ratio=input_data.damping_ratio,
                period_end=input_data.period_end,
                period_step=input_data.period_step,
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "analyze_frame_3d":
            input_data = Frame3DInput(**arguments)

            multi = analyze_frame_3d_multi(
                stories=input_data.stories,
                bays_x=input_data.bays_x,
                bays_y=input_data.bays_y,
                load_cases=input_data.load_cases,
                supports=input_data.supports,
                column_section=input_data.column_section,
                beam_x_section=input_data.beam_x_section,
                beam_y_section=input_data.beam_y_section,
                material_name=input_data.material_name,
                num_elements_per_member=input_data.num_elements_per_member,
                load_combinations=input_data.load_combinations,
                rigid_diaphragm=input_data.rigid_diaphragm,
                member_releases=input_data.member_releases,
                geometric_nonlinearity=input_data.geometric_nonlinearity,
                modal_analysis=input_data.modal_analysis,
                story_weights_kN=input_data.story_weights_kN,
            )

            # 케이스별 결과 요약
            def _case_summary_3d(cr):
                return {
                    "max_displacement_x": f"{cr.max_displacement_x:.3f} mm",
                    "max_displacement_x_node": cr.max_displacement_x_node,
                    "max_displacement_y": f"{cr.max_displacement_y:.3f} mm",
                    "max_displacement_y_node": cr.max_displacement_y_node,
                    "max_displacement_z": f"{cr.max_displacement_z:.3f} mm",
                    "max_displacement_z_node": cr.max_displacement_z_node,
                    "max_drift_x": f"{cr.max_drift_x:.6f} rad",
                    "max_drift_x_story": cr.max_drift_x_story,
                    "max_drift_y": f"{cr.max_drift_y:.6f} rad",
                    "max_drift_y_story": cr.max_drift_y_story,
                    "max_moment": f"{cr.max_moment:.2f} kN·m",
                    "max_moment_element": cr.max_moment_element,
                    "max_axial": f"{cr.max_axial:.2f} kN",
                    "max_axial_element": cr.max_axial_element,
                    "max_shear": f"{cr.max_shear:.2f} kN",
                    "max_shear_element": cr.max_shear_element,
                    "max_torsion": f"{cr.max_torsion:.2f} kN·m",
                    "max_torsion_element": cr.max_torsion_element,
                    "reactions": cr.reactions,
                    "story_drifts": cr.story_drifts,
                }

            results_by_case = {}
            for cn, cr in multi.case_results.items():
                results_by_case[cn] = _case_summary_3d(cr)

            results_by_combo = {}
            for cn, cr in multi.combo_results.items():
                results_by_combo[cn] = _case_summary_3d(cr)

            response = {
                "status": "success",
                "input": {
                    "stories": [f"{s} m" for s in input_data.stories],
                    "bays_x": [f"{b} m" for b in input_data.bays_x],
                    "bays_y": [f"{b} m" for b in input_data.bays_y],
                    "num_stories": multi.num_stories,
                    "num_bays_x": multi.num_bays_x,
                    "num_bays_y": multi.num_bays_y,
                    "column_section": input_data.column_section,
                    "beam_x_section": input_data.beam_x_section,
                    "beam_y_section": input_data.beam_y_section,
                    "material": input_data.material_name,
                    "supports": input_data.supports,
                },
                "geometry": {
                    "total_height": f"{multi.total_height:.2f} m",
                    "total_width_x": f"{multi.total_width_x:.2f} m",
                    "total_width_y": f"{multi.total_width_y:.2f} m",
                    "num_elements": multi.num_elements,
                    "coordinate_system": "X=bay_x, Y=bay_y, Z=height(up)",
                },
                "sections": {
                    "column": {
                        "A_mm2": multi.column_A_mm2,
                        "Ix_mm4": multi.column_Ix_mm4,
                        "Iy_mm4": multi.column_Iy_mm4,
                        "J_mm4": multi.column_J_mm4,
                    },
                    "beam_x": {
                        "A_mm2": multi.beam_x_A_mm2,
                        "Ix_mm4": multi.beam_x_Ix_mm4,
                        "Iy_mm4": multi.beam_x_Iy_mm4,
                        "J_mm4": multi.beam_x_J_mm4,
                    },
                    "beam_y": {
                        "A_mm2": multi.beam_y_A_mm2,
                        "Ix_mm4": multi.beam_y_Ix_mm4,
                        "Iy_mm4": multi.beam_y_Iy_mm4,
                        "J_mm4": multi.beam_y_J_mm4,
                    },
                },
                "load_cases": list(multi.case_results.keys()),
                "load_combinations": list(multi.combo_results.keys()),
                "results_by_case": results_by_case,
                "results_by_combo": results_by_combo,
                "analysis_metadata": getattr(multi, 'analysis_metadata', {}),
            }

            # 고유치해석 결과 포함
            if multi.modal_analysis:
                response["modal_analysis"] = multi.modal_analysis

            # HTML 리포트 생성
            try:
                html_path = plot_frame_3d_interactive(multi)
                response["html_report_path"] = html_path
            except Exception:
                pass

            return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        elif name == "resolve_building_config":
            input_data = ResolveBuildingConfigInput(**arguments)
            from core.nl_resolver import resolve_building_config
            result = resolve_building_config(input_data.intent)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "analyze_building":
            input_data = BuildingAnalysisInput(**arguments)

            # 1. BuildingModel 생성
            model = BuildingModel.from_json(input_data.config)

            # 2. 하중 자동 생성
            load_result = generate_all_loads(model)

            # 2.5. 가정 확인 (Assumption Confirmation)
            from core.assumption_tracker import build_assumption_summary
            assumptions = build_assumption_summary(
                model, input_data.config, load_result["summary"]
            )

            # 3. frame_3d 해석 (모달해석 자동 활성화)
            kwargs = model.to_frame3d_kwargs()
            kwargs["load_cases"] = load_result["load_cases"]
            kwargs["load_combinations"] = load_result["load_combinations"]
            kwargs["modal_analysis"] = True

            multi = analyze_frame_3d_multi(**kwargs)

            # 4. 결과 요약
            def _case_summary_bldg(cr):
                return {
                    "max_displacement_x_mm": round(cr.max_displacement_x, 3),
                    "max_displacement_y_mm": round(cr.max_displacement_y, 3),
                    "max_displacement_z_mm": round(cr.max_displacement_z, 3),
                    "max_drift_x": round(cr.max_drift_x, 6),
                    "max_drift_y": round(cr.max_drift_y, 6),
                    "max_moment_kNm": round(cr.max_moment, 2),
                    "max_axial_kN": round(cr.max_axial, 2),
                    "max_shear_kN": round(cr.max_shear, 2),
                    "story_drifts": cr.story_drifts,
                }

            results_by_case = {cn: _case_summary_bldg(cr) for cn, cr in multi.case_results.items()}
            results_by_combo = {cn: _case_summary_bldg(cr) for cn, cr in multi.combo_results.items()}

            # 5. Envelope (최대값 across all combos)
            env = {"max_dx_mm": 0, "max_dy_mm": 0, "max_dz_mm": 0,
                   "max_drift_x": 0, "max_drift_y": 0,
                   "max_moment_kNm": 0, "max_axial_kN": 0, "max_shear_kN": 0,
                   "governing_combo_drift_x": "", "governing_combo_drift_y": "",
                   "governing_combo_moment": ""}
            for cn, cr in multi.combo_results.items():
                if abs(cr.max_displacement_x) > abs(env["max_dx_mm"]):
                    env["max_dx_mm"] = round(cr.max_displacement_x, 3)
                if abs(cr.max_displacement_y) > abs(env["max_dy_mm"]):
                    env["max_dy_mm"] = round(cr.max_displacement_y, 3)
                if abs(cr.max_displacement_z) > abs(env["max_dz_mm"]):
                    env["max_dz_mm"] = round(cr.max_displacement_z, 3)
                if cr.max_drift_x > env["max_drift_x"]:
                    env["max_drift_x"] = round(cr.max_drift_x, 6)
                    env["governing_combo_drift_x"] = cn
                if cr.max_drift_y > env["max_drift_y"]:
                    env["max_drift_y"] = round(cr.max_drift_y, 6)
                    env["governing_combo_drift_y"] = cn
                if cr.max_moment > env["max_moment_kNm"]:
                    env["max_moment_kNm"] = round(cr.max_moment, 2)
                    env["governing_combo_moment"] = cn
                if cr.max_axial > env["max_axial_kN"]:
                    env["max_axial_kN"] = round(cr.max_axial, 2)
                if cr.max_shear > env["max_shear_kN"]:
                    env["max_shear_kN"] = round(cr.max_shear, 2)

            # Design check
            try:
                from core.design_check import run_design_check
                seismic_rpt = load_result["reports"].get("seismic")
                dc_result = run_design_check(multi, model, seismic_rpt)
            except Exception:
                dc_result = None

            # Result interpretation
            interpretation = None
            if dc_result is not None:
                try:
                    from core.result_interpreter import interpret_results
                    interpretation = interpret_results(
                        dc_result, multi,
                        modal_analysis=multi.modal_analysis or None,
                    )
                except Exception:
                    pass

            response = {
                "status": "success",
                "building": model.summary(),
                "assumptions": assumptions,
                "load_generation": {
                    "summary": load_result["summary"],
                    "load_cases": list(load_result["load_cases"].keys()),
                    "load_combinations": list(load_result["load_combinations"].keys()),
                    "reports": load_result["reports"],
                },
                "analysis": {
                    "num_elements": multi.num_elements,
                    "results_by_case": results_by_case,
                    "results_by_combo": results_by_combo,
                    "envelope": env,
                    "analysis_metadata": getattr(multi, 'analysis_metadata', {}),
                },
            }
            if dc_result is not None:
                response["design_check"] = dc_result
            if interpretation is not None:
                response["interpretation"] = interpretation
            if multi.modal_analysis:
                response["modal_analysis"] = multi.modal_analysis

            # HTML 리포트 생성
            try:
                html_path = plot_frame_3d_interactive(
                    multi, assumptions=assumptions,
                    design_check=dc_result, interpretation=interpretation,
                )
                response["html_report_path"] = html_path
            except Exception:
                pass

            return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        # ── V2 Tools ──

        elif name == "parse_ifc_v2":
            input_data = ParseIFCV2Input(**arguments)
            from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
            from core.visualization_v2 import generate_model_viewer

            model, validation = parse_ifc_v2(
                input_data.ifc_path,
                tolerance_mm=input_data.tolerance_mm,
                default_beam_section=input_data.default_beam_section,
                default_column_section=input_data.default_column_section,
            )

            if input_data.auto_snap and validation.is_valid:
                snapped = snap_beams_to_columns(model)
            else:
                snapped = 0

            # HTML 뷰어 생성
            html_path = None
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_path = generate_model_viewer(
                    model,
                    output_path=os.path.join(OUTPUT_DIR, f"v2_model_{ts}.html"),
                    title="IFC V2 Model",
                )
            except Exception:
                pass

            response = {
                "status": "success" if validation.is_valid else "validation_failed",
                "model": model.to_json(),
                "validation": {
                    "is_valid": validation.is_valid,
                    "summary": validation.summary_text(),
                    "extracted_nodes": validation.extracted_nodes,
                    "extracted_elements": validation.extracted_elements,
                    "failed_elements": validation.failed_elements,
                    "issues": [
                        {"severity": i.severity.value, "code": i.code,
                         "message": i.message, "default_value": i.default_value}
                        for i in validation.issues
                    ],
                    "needs_user_input": [
                        {"code": i.code, "message": i.message, "default_value": i.default_value}
                        for i in validation.needs_user_input
                    ],
                },
                "snapped_nodes": snapped,
                "summary": model.summary(),
            }
            if html_path:
                response["html_viewer_path"] = html_path

            return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        elif name == "snap_model_joints":
            input_data = SnapModelJointsInput(**arguments)
            from core.structural_model import StructuralModel
            from core.ifc_parser_v2 import snap_beams_to_columns

            model = StructuralModel.from_json(input_data.model_json)
            nodes_before = len(model.nodes)
            snapped = snap_beams_to_columns(model, snap_tolerance=input_data.snap_tolerance)

            response = {
                "status": "success",
                "snapped_count": snapped,
                "nodes_before": nodes_before,
                "nodes_after": len(model.nodes),
                "model": model.to_json(),
                "summary": model.summary(),
            }
            return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        elif name == "analyze_model_v2":
            input_data = AnalyzeModelV2Input(**arguments)
            from core.structural_model import StructuralModel
            from core.frame_3d import analyze_from_model
            from core.visualization_v2 import generate_model_viewer

            model = StructuralModel.from_json(input_data.model_json)
            result = analyze_from_model(
                model,
                load_cases=input_data.load_cases,
                load_combinations=input_data.load_combinations,
            )

            # HTML 리포트
            html_path = None
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_path = generate_model_viewer(
                    model, result=result,
                    output_path=os.path.join(OUTPUT_DIR, f"v2_analysis_{ts}.html"),
                    title="V2 Analysis Results",
                )
            except Exception:
                pass

            # 결과 요약
            response = {
                "status": "success",
                "model_summary": model.summary(),
                "cases": {},
                "combos": {},
            }
            for cname, cr in result.case_results.items():
                response["cases"][cname] = {
                    "max_displacement_x_mm": cr.max_displacement_x,
                    "max_displacement_y_mm": cr.max_displacement_y,
                    "max_displacement_z_mm": cr.max_displacement_z,
                    "max_moment_kNm": cr.max_moment,
                    "max_axial_kN": cr.max_axial,
                    "max_shear_kN": cr.max_shear,
                    "total_reaction_RZ_kN": round(sum(r["RZ_kN"] for r in cr.reactions), 1),
                    "story_drifts": cr.story_drifts,
                    "num_reactions": len(cr.reactions),
                }
            for cname, cr in result.combo_results.items():
                response["combos"][cname] = {
                    "max_displacement_x_mm": cr.max_displacement_x,
                    "max_moment_kNm": cr.max_moment,
                    "story_drifts": cr.story_drifts,
                }

            if html_path:
                response["html_report_path"] = html_path

            return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        error_response = {"status": "error", "message": str(e)}
        return [TextContent(type="text", text=json.dumps(error_response, ensure_ascii=False))]


# 서버 실행
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
