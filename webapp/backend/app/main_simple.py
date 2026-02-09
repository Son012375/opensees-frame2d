"""
Simplified FastAPI app - No Redis/Celery required
Runs analysis synchronously (blocking) - for testing only
"""
import sys
import json
import uuid
import traceback
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse

from pydantic import BaseModel
from typing import Optional

from app.models.schemas import Frame2DInput, JobCreateResponse, JobResponse, JobStatus, JobSummary
from app.core.config import MCP_SERVER_PATH, JOBS_DIR
from app.core.claude_service import parse_natural_language, check_api_key


class NaturalLanguageInput(BaseModel):
    """Natural language input for Claude parsing"""
    text: str


class ParseResponse(BaseModel):
    """Response from natural language parsing"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

# Application
app = FastAPI(title="Frame2D Analysis (Simple Mode)")

# Static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# In-memory job storage
jobs_db = {}


@app.post("/api/jobs", response_model=JobCreateResponse)
async def create_job(input_data: Frame2DInput):
    """Create and immediately run analysis (synchronous)"""
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input (use dict() for Pydantic v1 compatibility)
    with open(job_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_data.dict(), f, indent=2)

    # Initialize job
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": JobStatus.RUNNING,
        "progress": 10,
        "created_at": datetime.now().isoformat(),
    }

    try:
        # Add MCP server to path
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))

        from core.frame_2d import analyze_frame_2d_multi
        from core.visualization import plot_frame_2d_multi_interactive

        # Convert load_cases (use dict() for Pydantic v1 compatibility)
        load_cases = {}
        for case_name, loads in input_data.load_cases.items():
            load_cases[case_name] = [
                {k: v for k, v in ld.dict().items() if v is not None}
                for ld in loads
            ]

        # Run analysis
        result = analyze_frame_2d_multi(
            stories=input_data.stories,
            bays=input_data.bays,
            load_cases=load_cases,
            supports=input_data.supports,
            column_section=input_data.column_section,
            beam_section=input_data.beam_section,
            material_name=input_data.material_name,
            num_elements_per_member=input_data.num_elements_per_member,
            load_combinations=input_data.load_combinations,
        )

        # Generate report
        report_path = str(job_dir / "report.html")
        plot_frame_2d_multi_interactive(result, output_path=report_path)

        # Extract summary
        all_cases = list(result.case_results.keys()) + list(result.combo_results.keys())
        if all_cases:
            first_case = all_cases[0]
            cr = result.case_results.get(first_case) or result.combo_results.get(first_case)
            base_shear = sum(abs(r.get("RX_kN", 0)) for r in cr.reactions)

            summary = JobSummary(
                max_displacement_x_mm=cr.max_displacement_x,
                max_displacement_y_mm=cr.max_displacement_y,
                max_drift=cr.max_drift,
                max_drift_story=cr.max_drift_story,
                max_moment_kNm=cr.max_moment,
                max_shear_kN=cr.max_shear,
                max_axial_kN=cr.max_axial,
                base_shear_kN=base_shear,
                num_stories=result.num_stories,
                num_bays=result.num_bays,
            )
            jobs_db[job_id]["summary"] = summary

        jobs_db[job_id]["status"] = JobStatus.DONE
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_db[job_id]["report_url"] = f"/api/jobs/{job_id}/report"

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error"] = str(e)
        print(f"Analysis error: {traceback.format_exc()}")

    return JobCreateResponse(job_id=job_id, message="Analysis completed")


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**jobs_db[job_id])


@app.get("/api/jobs/{job_id}/report")
async def get_report(job_id: str):
    report_path = JOBS_DIR / job_id / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(report_path), media_type="text/html")


# Page routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
async def job_status_page(request: Request, job_id: str):
    if job_id not in jobs_db:
        return templates.TemplateResponse("partials/job_not_found.html", {"request": request})
    job = JobResponse(**jobs_db[job_id])
    return templates.TemplateResponse("job_status.html", {"request": request, "job": job})


@app.get("/partials/job-status/{job_id}", response_class=HTMLResponse)
async def job_status_partial(request: Request, job_id: str):
    if job_id not in jobs_db:
        return templates.TemplateResponse("partials/job_not_found.html", {"request": request})
    job = JobResponse(**jobs_db[job_id])
    return templates.TemplateResponse("partials/job_card.html", {"request": request, "job": job})


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "simple (no Redis)"}


# ─────────────────────────────────────────────────────────────────────────────
# Claude API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/claude/status")
async def claude_status():
    """Check if Claude API is configured"""
    return {"available": check_api_key()}


@app.post("/api/claude/parse", response_model=ParseResponse)
async def parse_natural_language_input(input_data: NaturalLanguageInput):
    """
    Parse natural language description to Frame2D input JSON using Claude API
    """
    try:
        result = parse_natural_language(input_data.text)
        return ParseResponse(success=True, data=result)
    except ValueError as e:
        return ParseResponse(success=False, error=str(e))
    except Exception as e:
        return ParseResponse(success=False, error=f"파싱 오류: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("Frame2D Analysis - Simple Mode (No Redis)")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
