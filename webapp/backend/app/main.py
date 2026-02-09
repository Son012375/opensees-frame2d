"""
FastAPI main application for Frame2D analysis web app
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.api.routes import router as api_router
from app.core.job_manager import list_jobs, get_job_status
from app.core.config import JOBS_DIR

# Application
app = FastAPI(
    title="Frame2D Analysis Web App",
    description="Web interface for 2D frame structural analysis using OpenSees",
    version="1.0.0"
)

# Static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Include API routes
app.include_router(api_router)


# ─────────────────────────────────────────────────────────────────────────────
# Page Routes (Jinja2 + HTMX)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Main page - analysis input form
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    """
    Job history page
    """
    jobs = list_jobs(limit=50)
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
async def job_status_page(request: Request, job_id: str):
    """
    Job status page (for HTMX polling)
    """
    job = get_job_status(job_id)
    if not job:
        return templates.TemplateResponse("partials/job_not_found.html", {"request": request})
    return templates.TemplateResponse("job_status.html", {"request": request, "job": job})


@app.get("/partials/job-status/{job_id}", response_class=HTMLResponse)
async def job_status_partial(request: Request, job_id: str):
    """
    Partial template for HTMX polling
    """
    job = get_job_status(job_id)
    if not job:
        return templates.TemplateResponse("partials/job_not_found.html", {"request": request})
    return templates.TemplateResponse("partials/job_card.html", {"request": request, "job": job})


@app.get("/partials/jobs-list", response_class=HTMLResponse)
async def jobs_list_partial(request: Request):
    """
    Partial template for jobs list refresh
    """
    jobs = list_jobs(limit=50)
    return templates.TemplateResponse("partials/jobs_list.html", {"request": request, "jobs": jobs})


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    from app.core.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
