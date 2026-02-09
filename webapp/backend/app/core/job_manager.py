"""
Job management utilities
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import JOBS_DIR
from app.models.schemas import JobStatus, JobResponse, JobSummary, Frame2DInput


def create_job(input_data: Frame2DInput) -> str:
    """Create a new job and return job_id"""
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input
    input_path = job_dir / "input.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data.model_dump(), f, indent=2, ensure_ascii=False)

    # Save initial status
    status_path = job_dir / "status.json"
    status_data = {
        "job_id": job_id,
        "status": JobStatus.QUEUED.value,
        "created_at": datetime.now().isoformat(),
        "progress": 0,
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    return job_id


def get_job_status(job_id: str) -> Optional[JobResponse]:
    """Get job status and summary if available"""
    job_dir = JOBS_DIR / job_id
    status_path = job_dir / "status.json"

    if not status_path.exists():
        return None

    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check for summary
    summary = None
    summary_path = job_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            summary = JobSummary(**summary_data)

    # Check for report
    report_url = None
    report_path = job_dir / "report.html"
    if report_path.exists():
        report_url = f"/api/jobs/{job_id}/report"

    return JobResponse(
        job_id=job_id,
        status=JobStatus(data.get("status", "queued")),
        progress=data.get("progress"),
        summary=summary,
        report_url=report_url,
        error=data.get("error"),
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at"),
    )


def update_job_status(
    job_id: str,
    status: JobStatus,
    progress: Optional[int] = None,
    error: Optional[str] = None,
):
    """Update job status"""
    job_dir = JOBS_DIR / job_id
    status_path = job_dir / "status.json"

    if not status_path.exists():
        return

    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["status"] = status.value
    if progress is not None:
        data["progress"] = progress
    if error is not None:
        data["error"] = error
    if status in (JobStatus.DONE, JobStatus.FAILED):
        data["completed_at"] = datetime.now().isoformat()

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_job_summary(job_id: str, summary: dict):
    """Save analysis summary"""
    job_dir = JOBS_DIR / job_id
    summary_path = job_dir / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def get_job_input(job_id: str) -> Optional[dict]:
    """Get job input data"""
    job_dir = JOBS_DIR / job_id
    input_path = job_dir / "input.json"

    if not input_path.exists():
        return None

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_job_report_path(job_id: str) -> Optional[Path]:
    """Get path to job report HTML"""
    job_dir = JOBS_DIR / job_id
    report_path = job_dir / "report.html"

    if report_path.exists():
        return report_path
    return None


def list_jobs(limit: int = 20) -> list[JobResponse]:
    """List recent jobs"""
    jobs = []
    if not JOBS_DIR.exists():
        return jobs

    job_dirs = sorted(JOBS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    for job_dir in job_dirs[:limit]:
        if job_dir.is_dir():
            job_status = get_job_status(job_dir.name)
            if job_status:
                jobs.append(job_status)

    return jobs
