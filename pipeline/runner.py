"""Pipeline Runner - Agent 파이프라인 실행."""

import json
from pathlib import Path
from datetime import datetime

from agents import (
    DocumentAgent,
    TableAgent,
    NormalizeAgent,
    ValidateAgent,
    LoaderAgent,
)
from .config import PipelineConfig


class PipelineRunner:
    """KDS 파라미터 추출 파이프라인."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.agents = {
            "document": DocumentAgent(),
            "table": TableAgent(),
            "normalize": NormalizeAgent(),
            "validate": ValidateAgent(),
            "loader": LoaderAgent(),
        }

    def run(self, pdf_path: str, mode: str = None) -> dict:
        """전체 파이프라인 실행.

        Args:
            pdf_path: PDF 파일 경로
            mode: 실행 모드 ("dry_run" 또는 "upsert")

        Returns:
            파이프라인 결과
        """
        mode = mode or self.config.mode
        results = {
            "started_at": datetime.now().isoformat(),
            "pdf_path": pdf_path,
            "mode": mode,
        }

        print("=" * 60)
        print(f"KDS Parameter Extraction Pipeline")
        print(f"PDF: {pdf_path}")
        print(f"Mode: {mode}")
        print("=" * 60)

        try:
            # Stage 1: Document Analysis
            print("\n[1/5] Extracting document metadata...")
            doc_result = self.agents["document"].process({"file_path": pdf_path})
            results["document"] = doc_result

            if "error" in doc_result:
                print(f"  [ERROR] {doc_result['error']}")
                return results

            print(f"  [OK] Document: {doc_result.get('code_id', 'Unknown')}")
            print(f"  [OK] Tables found: {len(doc_result.get('tables', []))}")

            # Stage 2: Table Extraction
            print("\n[2/5] Extracting table data...")
            tables = doc_result.get("tables", [])
            table_results = []

            for i, table_meta in enumerate(tables):
                print(f"  [{i+1}/{len(tables)}] {table_meta.get('table_id', 'Unknown')}...")
                try:
                    table_result = self.agents["table"].process({
                        "file_path": pdf_path,
                        "table_meta": table_meta
                    })
                    table_results.append(table_result)
                    rows = len(table_result.get("rows", []))
                    conf = table_result.get("confidence", 0)
                    print(f"       -> {rows} rows, confidence: {conf:.2f}")
                except Exception as e:
                    print(f"       -> [ERROR] {e}")
                    if not self.config.skip_failed_tables:
                        raise

            results["tables"] = table_results
            print(f"  [OK] Total {len(table_results)} tables extracted")

            # Stage 3: Normalization
            print("\n[3/5] Normalizing data...")
            normalized = self.agents["normalize"].process({
                "tables": table_results,
                "source": doc_result
            })
            results["normalized"] = normalized
            print(f"  [OK] {len(normalized)} records normalized")

            # Stage 4: Validation
            print("\n[4/5] Validating data...")
            validation = self.agents["validate"].process({
                "records": normalized
            })
            results["validation"] = validation
            print(f"  [OK] Passed: {validation.get('passed', 0)}")
            print(f"  [WARN] Needs review: {validation.get('needs_review', 0)}")
            print(f"  [FAIL] Failed: {validation.get('failed', 0)}")

            # Stage 5: Loading
            print(f"\n[5/5] Loading data ({mode})...")
            load_result = self.agents["loader"].process({
                "records": normalized,
                "validation": validation,
                "mode": mode
            })
            results["load"] = load_result
            summary = load_result.get("summary", {})
            print(f"  [OK] Insert: {summary.get('insert', 0)}")
            print(f"  [OK] Update: {summary.get('update', 0)}")
            print(f"  [SKIP] Skip: {summary.get('skip', 0)}")

            results["completed_at"] = datetime.now().isoformat()
            results["success"] = True

        except Exception as e:
            print(f"\n[ERROR] Pipeline error: {e}")
            results["error"] = str(e)
            results["success"] = False

        # 결과 저장
        if self.config.save_intermediate:
            self._save_results(results)

        print("\n" + "=" * 60)
        print("Pipeline completed!")
        print("=" * 60)

        return results

    def _save_results(self, results: dict):
        """결과를 JSON 파일로 저장."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_result_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n[SAVED] Results: {filepath}")

    def run_single_agent(self, agent_name: str, input_data: dict) -> dict:
        """단일 Agent 실행 (테스트용).

        Args:
            agent_name: "document", "table", "normalize", "validate", "loader"
            input_data: Agent 입력 데이터

        Returns:
            Agent 결과
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        return self.agents[agent_name].process(input_data)
