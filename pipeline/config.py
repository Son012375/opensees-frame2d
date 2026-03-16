"""Pipeline configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """파이프라인 설정."""

    # 실행 모드
    mode: str = "dry_run"  # "dry_run" | "upsert"

    # 모델 설정
    model: str = "claude-sonnet-4-20250514"

    # 확신도 임계값
    confidence_threshold_accept: float = 0.9  # 이상이면 자동 승인
    confidence_threshold_review: float = 0.7  # 이하면 리뷰 필요

    # 처리 옵션
    max_pages_to_extract: int = 20
    skip_failed_tables: bool = True
    skip_needs_review: bool = True

    # 출력
    save_intermediate: bool = True
    output_dir: str = "data/output"
