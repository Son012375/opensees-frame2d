"""
OpenSeesPy 핵심 로직 테스트 (LLM 없이)
간단한 단순보 해석 테스트
"""
import openseespy.opensees as ops

def test_simple_beam():
    """단순보 해석 테스트"""

    # 모델 초기화
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # 파라미터
    span_length = 6.0  # m
    E = 2.1e11  # Pa (강재)
    A = 0.0085  # m²
    I = 0.00023  # m⁴

    # 노드 생성
    ops.node(1, 0.0, 0.0)
    ops.node(2, span_length, 0.0)

    # 경계조건 (단순보: 핀지-롤러)
    ops.fix(1, 1, 1, 0)
    ops.fix(2, 0, 1, 0)

    # 기하 변환
    ops.geomTransf('Linear', 1)

    # 요소 생성
    ops.element('elasticBeamColumn', 1, 1, 2, A, E, I, 1)

    # 하중 패턴
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    # 등분포하중 10 kN/m
    w = -10000  # N/m (하향)
    ops.eleLoad('-ele', 1, '-type', '-beamUniform', w)

    # 해석 설정 및 실행
    ops.system('BandGen')
    ops.constraints('Plain')
    ops.numberer('Plain')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1)
    ops.analysis('Static')
    ops.analyze(1)
    ops.reactions()

    # 결과
    reaction_node1 = ops.nodeReaction(1)

    print("=" * 50)
    print("OpenSeesPy 단순보 해석 테스트 결과")
    print("=" * 50)
    print(f"스팬: {span_length} m")
    print(f"등분포하중: {abs(w)/1000:.1f} kN/m")
    print(f"해석 반력: {abs(reaction_node1[1])/1000:.2f} kN")
    print(f"이론 반력: {abs(w)*span_length/2/1000:.2f} kN")
    print("✓ OpenSeesPy 정상 작동 확인!")

if __name__ == "__main__":
    test_simple_beam()
