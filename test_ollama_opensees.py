# -*- coding: utf-8 -*-
"""
Ollama + OpenSeesPy 연동 테스트
로컬 LLM으로 구조화된 출력 테스트
"""
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
import openseespy.opensees as ops

# Pydantic 모델 정의
class SimpleBeamParams(BaseModel):
    """단순보 파라미터"""
    span_length: float = Field(..., description="보의 길이 (m)")
    load_magnitude: float = Field(..., description="등분포하중 크기 (kN/m)")
    section_height: float = Field(default=0.4, description="단면 높이 (m)")
    material: str = Field(default="SS400", description="재료명")

class AnalysisResponse(BaseModel):
    """응답 모델"""
    params: SimpleBeamParams
    explanation: str = Field(..., description="파라미터 추출 설명")

def test_ollama_structured_output():
    """Ollama로 구조화된 출력 테스트"""

    # Ollama 클라이언트 설정
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        ),
        mode=instructor.Mode.JSON
    )

    user_input = "6m 길이의 단순보에 10kN/m 등분포하중을 적용해줘"

    print("=" * 50)
    print("Ollama 구조화된 출력 테스트")
    print("=" * 50)
    print(f"\n[입력] {user_input}")

    try:
        response = client.chat.completions.create(
            model="qwen2.5:3b",  # 설치된 모델 사용
            messages=[
                {
                    "role": "system",
                    "content": """You are a structural engineering assistant.
Extract beam parameters from the user's description.
- span_length: beam length in meters
- load_magnitude: distributed load in kN/m (positive value)
- section_height: section height in meters (default 0.4m if not specified)
- material: material name (default SS400 if not specified)"""
                },
                {"role": "user", "content": user_input}
            ],
            response_model=AnalysisResponse,
            temperature=0.1
        )

        print(f"\n[추출된 파라미터]")
        print(f"  스팬 길이: {response.params.span_length} m")
        print(f"  등분포하중: {response.params.load_magnitude} kN/m")
        print(f"  설명: {response.explanation}")
        return response.params

    except Exception as e:
        print(f"\n[에러] {e}")
        return None

def run_opensees_with_params(params: SimpleBeamParams):
    """추출된 파라미터로 OpenSees 해석 실행"""

    print("\n" + "=" * 50)
    print("OpenSeesPy 해석 실행")
    print("=" * 50)

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    L = params.span_length
    w = params.load_magnitude * 1000  # kN/m -> N/m
    E = 2.1e11
    A = 0.0085
    I = 0.00023

    ops.node(1, 0.0, 0.0)
    ops.node(2, L, 0.0)
    ops.fix(1, 1, 1, 0)
    ops.fix(2, 0, 1, 0)
    ops.geomTransf('Linear', 1)
    ops.element('elasticBeamColumn', 1, 1, 2, A, E, I, 1)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.eleLoad('-ele', 1, '-type', '-beamUniform', -w)
    ops.system('BandGen')
    ops.constraints('Plain')
    ops.numberer('Plain')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1)
    ops.analysis('Static')
    ops.analyze(1)
    ops.reactions()

    reaction = ops.nodeReaction(1)[1]

    print(f"\n[해석 결과]")
    print(f"  지점 반력: {abs(reaction)/1000:.2f} kN")
    print(f"  이론값: {w*L/2/1000:.2f} kN")
    print("\n[OK] 파이프라인 테스트 완료!")

if __name__ == "__main__":
    params = test_ollama_structured_output()
    if params:
        run_opensees_with_params(params)
    else:
        # 수동 파라미터로 테스트
        print("\n[Ollama 연결 실패 - 수동 파라미터로 진행]")
        manual_params = SimpleBeamParams(span_length=6.0, load_magnitude=10.0)
        run_opensees_with_params(manual_params)
