#!/usr/bin/env python
"""KDS Parameter Extraction Pipeline - 메인 실행 스크립트."""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from pipeline import PipelineRunner, PipelineConfig


def main():
    """메인 함수."""
    # .env 파일 로드
    load_dotenv()

    # PDF 경로 (환경변수 또는 직접 지정)
    pdf_path = os.getenv(
        "KDS_PDF_PATH",
        r"C:\Users\youm\Downloads\KDS 41 12 00_건축물 설계하중.pdf"
    )

    # 파일 존재 확인
    if not Path(pdf_path).exists():
        print(f"[ERROR] PDF file not found: {pdf_path}")
        print("\n다음 방법 중 하나로 PDF 경로를 지정하세요:")
        print("1. 환경변수: set KDS_PDF_PATH=경로")
        print("2. .env 파일에 KDS_PDF_PATH=경로 추가")
        print("3. 이 스크립트의 pdf_path 변수 직접 수정")
        return

    # 설정
    config = PipelineConfig(
        mode="dry_run",  # 첫 실행은 dry_run으로
        save_intermediate=True,
        output_dir="data/output"
    )

    # 파이프라인 실행
    runner = PipelineRunner(config)
    results = runner.run(pdf_path)

    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 결과 요약")
    print("=" * 60)

    if results.get("success"):
        doc = results.get("document", {})
        print(f"문서: {doc.get('code_id', 'Unknown')} - {doc.get('code_title', '')}")
        print(f"버전: {doc.get('version_date', 'Unknown')}")

        tables = results.get("tables", [])
        print(f"\n표 추출: {len(tables)}개")

        normalized = results.get("normalized", [])
        print(f"정규화 레코드: {len(normalized)}개")

        validation = results.get("validation", {})
        print(f"검증 통과: {validation.get('passed', 0)}개")

        load = results.get("load", {})
        if load.get("mode") == "dry_run":
            print(f"\n[Dry-run] 실제 적재하려면 mode='upsert'로 재실행하세요")
            operations = load.get("operations", [])
            if operations:
                print(f"적재 예정 레코드 샘플:")
                for op in operations[:3]:
                    data = op.get("data", {})
                    print(f"  - {data.get('display_name_ko', '?')}: {data.get('value', '?')} {data.get('unit', '')}")
    else:
        print(f"❌ 오류 발생: {results.get('error', 'Unknown')}")


def test_document_agent():
    """Document Agent만 테스트."""
    load_dotenv()

    pdf_path = os.getenv(
        "KDS_PDF_PATH",
        r"C:\Users\youm\Downloads\KDS 41 12 00_건축물 설계하중.pdf"
    )

    if not Path(pdf_path).exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return

    from agents import DocumentAgent

    print("[TEST] Document Agent")
    print("-" * 40)

    agent = DocumentAgent()
    result = agent.process({"file_path": pdf_path})

    print(f"\n[Document Info]")
    print(f"  Code ID: {result.get('code_id', 'Unknown')}")
    print(f"  Title: {result.get('code_title', 'Unknown')}")
    print(f"  Version: {result.get('version_date', 'Unknown')}")

    tables = result.get("tables", [])
    print(f"\n[Tables Found]: {len(tables)}")
    for t in tables[:5]:  # 처음 5개만
        print(f"  - {t.get('table_id', '?')}: {t.get('title', '?')} (p.{t.get('page', '?')})")

    return result


if __name__ == "__main__":
    # 전체 파이프라인 실행
    # main()

    # 또는 Document Agent만 테스트
    test_document_agent()
