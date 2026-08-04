"""
ETABS 23 API 연결 테스트 스크립트

사용법:
    # ETABS가 이미 열려 있을 때:
    python scripts/test_etabs_connection.py --mode attach

    # ETABS를 새로 실행할 때:
    python scripts/test_etabs_connection.py --mode launch

    # 기존 모델 파일 열기:
    python scripts/test_etabs_connection.py --mode launch --model "D:/path/to/model.edb"
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.etabs_api import ETABSClient


def test_connection(mode: str, model_path: str = None):
    print(f"\n{'='*50}")
    print(f"ETABS 23 API 연결 테스트 (mode={mode})")
    print(f"{'='*50}")

    try:
        if mode == "attach":
            print("▶ 실행 중인 ETABS에 연결 중...")
            client = ETABSClient.attach()
        else:
            print("▶ ETABS 새 인스턴스 실행 중...")
            client = ETABSClient.launch(model_path=model_path, visible=True)

        print("✅ 연결 성공!")

        # 기본 정보 출력
        print(f"  단위 설정: kN, m, °C")
        client.set_units("kN_m_C")

        info = client.get_model_info()
        print(f"\n[모델 정보]")
        print(f"  절점 수:       {info['n_joints']}")
        print(f"  프레임 수:     {info['n_frames']}")
        print(f"  면 요소 수:    {info['n_areas']}")
        print(f"  하중 패턴 수:  {info['n_load_patterns']}")
        print(f"  하중 패턴:     {info['load_patterns']}")
        print(f"  층 수:         {info['n_stories']}")
        print(f"  층 이름:       {info['stories']}")

        print(f"\n✅ 모든 기본 API 호출 성공!")
        return client

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["attach", "launch"], default="attach")
    parser.add_argument("--model", default=None, help=".edb 파일 경로")
    args = parser.parse_args()

    client = test_connection(args.mode, args.model)

    if client and args.mode == "launch":
        input("\n[Enter]를 누르면 ETABS를 종료합니다...")
        client.close()
