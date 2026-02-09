"""
API Routes for Frame2D analysis
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.models.schemas import Frame2DInput, JobCreateResponse, JobResponse, JobListResponse
from app.core.job_manager import create_job, get_job_status, list_jobs
from app.core.config import JOBS_DIR
from app.tasks.analysis_tasks import run_frame2d_analysis

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/jobs", response_model=JobCreateResponse)
async def create_analysis_job(input_data: Frame2DInput):
    """
    Create a new Frame2D analysis job
    """
    # Create job directory and save input
    job_id = create_job(input_data.model_dump())

    # Queue the analysis task
    run_frame2d_analysis.delay(job_id)

    return JobCreateResponse(
        job_id=job_id,
        message="Job queued successfully"
    )


@router.get("/jobs", response_model=JobListResponse)
async def get_jobs(limit: int = 20):
    """
    List recent jobs
    """
    jobs = list_jobs(limit=limit)
    return JobListResponse(jobs=jobs)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get job status and results
    """
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/report")
async def get_report(job_id: str):
    """
    Get the HTML report for a completed job
    """
    report_path = JOBS_DIR / job_id / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=f"frame2d_report_{job_id}.html"
    )


@router.get("/jobs/{job_id}/log")
async def get_log(job_id: str):
    """
    Get analysis log for debugging
    """
    log_path = JOBS_DIR / job_id / "analysis.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {"job_id": job_id, "log": content}
