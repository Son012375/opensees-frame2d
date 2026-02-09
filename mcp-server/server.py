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
from core.verification import verify_frame_equilibrium
from core.verification import verify_equilibrium

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


class SectionQueryInput(BaseModel):
    section_name: str = Field(..., description="조회할 단면 이름 (예: H-400x200x8x13)")


class MaterialQueryInput(BaseModel):
    material_name: str = Field(..., description="조회할 재료 이름 (예: SS275)")


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
