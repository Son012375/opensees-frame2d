"""
OpenSees Structural Analysis Platform
Simplified FastAPI app - No Redis/Celery required
Runs analysis synchronously (blocking)
"""
import sys
import json
import uuid
import traceback
from pathlib import Path
from datetime import datetime

# Load .env from project root (for ANTHROPIC_API_KEY etc.)
from dotenv import load_dotenv
_project_root = Path(__file__).resolve().parents[3]  # app/main_simple.py -> opensees-MCP
load_dotenv(_project_root / ".env")

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.models.schemas import Frame2DInput, JobCreateResponse, JobResponse, JobStatus, JobSummary
from app.core.config import MCP_SERVER_PATH, JOBS_DIR
from app.core.claude_service import parse_natural_language, parse_simple_beam, parse_continuous_beam, parse_building, check_api_key


class NaturalLanguageInput(BaseModel):
    """Natural language input for Claude parsing"""
    text: str


class ParseResponse(BaseModel):
    """Response from natural language parsing"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class SimpleBeamInput(BaseModel):
    """Simple beam analysis input"""
    span: float
    load_type: str = "uniform"
    load_value: float = 20.0
    support_type: str = "simple"
    section_name: str = "H-400x200x8x13"
    material_name: str = "SS275"
    point_location: Optional[float] = None
    load_start: Optional[float] = None
    load_end: Optional[float] = None
    load_value_end: Optional[float] = None
    num_elements: int = 20
    deflection_limit: int = 300


class ContinuousBeamInput(BaseModel):
    """Continuous beam analysis input"""
    spans: List[float]
    loads: List[dict]
    supports: Optional[List[str]] = None
    hinges: Optional[List[int]] = None
    section_name: str = "H-400x200x8x13"
    material_name: str = "SS275"
    num_elements_per_span: int = 20
    deflection_limit: int = 300


class BuildingInput(BaseModel):
    """Building analysis input (same config as MCP analyze_building)"""
    config: Dict[str, Any]


class BuildingModification(BaseModel):
    """Modification for re-analysis"""
    column_section: Optional[str] = None
    beam_x_section: Optional[str] = None
    beam_y_section: Optional[str] = None
    material_name: Optional[str] = None


# Application
app = FastAPI(title="OpenSees Structural Analysis Platform")

# Static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# In-memory job storage
jobs_db = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Page Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - Analysis type selection"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/frame2d", response_class=HTMLResponse)
async def frame2d_page(request: Request):
    """Frame2D analysis input page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/simple-beam", response_class=HTMLResponse)
async def simple_beam_page(request: Request):
    """Simple beam analysis input page"""
    return templates.TemplateResponse("simple_beam.html", {"request": request})


@app.get("/continuous-beam", response_class=HTMLResponse)
async def continuous_beam_page(request: Request):
    """Continuous beam analysis input page"""
    return templates.TemplateResponse("continuous_beam.html", {"request": request})


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_list_page(request: Request):
    """Jobs history page"""
    all_jobs = [JobResponse(**job) for job in jobs_db.values()]
    all_jobs.sort(key=lambda x: x.created_at or "", reverse=True)
    return templates.TemplateResponse("jobs_list.html", {"request": request, "jobs": all_jobs})


# ═══════════════════════════════════════════════════════════════════════════════
# Frame2D API
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/jobs", response_model=JobCreateResponse)
async def create_frame2d_job(input_data: Frame2DInput):
    """Create and immediately run Frame2D analysis (synchronous)"""
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input
    with open(job_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_data.dict(), f, indent=2)

    # Initialize job
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": JobStatus.RUNNING,
        "progress": 10,
        "created_at": datetime.now().isoformat(),
        "analysis_type": "frame2d",
    }

    try:
        # Add MCP server to path
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))

        from core.frame_2d import analyze_frame_2d_multi
        from core.visualization import plot_frame_2d_multi_interactive

        # Convert load_cases
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


# ═══════════════════════════════════════════════════════════════════════════════
# Simple Beam API
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/simple-beam/jobs", response_model=JobCreateResponse)
async def create_simple_beam_job(input_data: SimpleBeamInput):
    """Create and run Simple Beam analysis"""
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input
    with open(job_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_data.dict(), f, indent=2)

    # Initialize job
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": JobStatus.RUNNING,
        "progress": 10,
        "created_at": datetime.now().isoformat(),
        "analysis_type": "simple_beam",
    }

    try:
        # Add MCP server to path
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))

        from core.simple_beam import analyze_simple_beam
        from core.visualization import plot_beam_results_interactive

        # Run analysis
        result = analyze_simple_beam(
            span=input_data.span,
            load_type=input_data.load_type,
            load_value=input_data.load_value,
            support_type=input_data.support_type,
            section_name=input_data.section_name,
            material_name=input_data.material_name,
            point_location=input_data.point_location,
            load_start=input_data.load_start,
            load_end=input_data.load_end,
            load_value_end=input_data.load_value_end,
            num_elements=input_data.num_elements,
            deflection_limit=input_data.deflection_limit,
        )

        # Generate report
        report_path = str(job_dir / "report.html")
        plot_beam_results_interactive(result, output_path=report_path)

        # Create summary for simple beam
        # Calculate span from node positions or input
        span_m = input_data.span
        if result.node_positions:
            span_m = max(result.node_positions)

        # Calculate deflection check
        allowable_mm = (span_m * 1000) / result.deflection_limit_ratio
        deflection_ok = result.max_displacement <= allowable_mm

        summary = {
            "span_m": span_m,
            "max_displacement_mm": result.max_displacement,
            "max_moment_kNm": result.max_moment,
            "max_shear_kN": result.max_shear,
            "support_type": result.support_type,
            "section_name": result.section_name,
            "deflection_check": deflection_ok,
            "allowable_deflection_mm": allowable_mm,
        }
        jobs_db[job_id]["beam_summary"] = summary

        jobs_db[job_id]["status"] = JobStatus.DONE
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_db[job_id]["report_url"] = f"/api/jobs/{job_id}/report"

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error"] = str(e)
        print(f"Simple beam analysis error: {traceback.format_exc()}")

    return JobCreateResponse(job_id=job_id, message="Analysis completed")


@app.get("/simple-beam/jobs/{job_id}/status", response_class=HTMLResponse)
async def simple_beam_job_status_page(request: Request, job_id: str):
    if job_id not in jobs_db:
        return templates.TemplateResponse("partials/job_not_found.html", {"request": request})
    job = jobs_db[job_id]
    return templates.TemplateResponse("simple_beam_status.html", {"request": request, "job": job})


# ═══════════════════════════════════════════════════════════════════════════════
# Continuous Beam API
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/continuous-beam/jobs", response_model=JobCreateResponse)
async def create_continuous_beam_job(input_data: ContinuousBeamInput):
    """Create and run Continuous Beam analysis"""
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input
    with open(job_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_data.dict(), f, indent=2)

    # Initialize job
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": JobStatus.RUNNING,
        "progress": 10,
        "created_at": datetime.now().isoformat(),
        "analysis_type": "continuous_beam",
    }

    try:
        # Add MCP server to path
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))

        from core.continuous_beam import analyze_continuous_beam
        from core.visualization import plot_beam_results_interactive

        # Set default supports if not provided
        supports = input_data.supports
        if not supports:
            # Default: pin at start, rollers in middle, roller at end
            num_supports = len(input_data.spans) + 1
            supports = ["pin"] + ["roller"] * (num_supports - 1)

        # Run analysis
        result = analyze_continuous_beam(
            spans=input_data.spans,
            loads=input_data.loads,
            supports=supports,
            hinges=input_data.hinges,
            section_name=input_data.section_name,
            material_name=input_data.material_name,
            num_elements_per_span=input_data.num_elements_per_span,
            deflection_limit=input_data.deflection_limit,
        )

        # Generate report
        report_path = str(job_dir / "report.html")
        plot_beam_results_interactive(result, output_path=report_path)

        # Create summary for continuous beam
        total_span = sum(input_data.spans)

        # Calculate deflection check
        allowable_mm = (total_span * 1000) / result.deflection_limit_ratio
        deflection_ok = result.max_displacement <= allowable_mm

        summary = {
            "num_spans": len(input_data.spans),
            "total_span_m": total_span,
            "spans_m": input_data.spans,
            "max_displacement_mm": result.max_displacement,
            "max_moment_kNm": result.max_moment,
            "max_shear_kN": result.max_shear,
            "section_name": result.section_name,
            "deflection_check": deflection_ok,
            "allowable_deflection_mm": allowable_mm,
        }
        jobs_db[job_id]["beam_summary"] = summary

        jobs_db[job_id]["status"] = JobStatus.DONE
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_db[job_id]["report_url"] = f"/api/jobs/{job_id}/report"

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error"] = str(e)
        print(f"Continuous beam analysis error: {traceback.format_exc()}")

    return JobCreateResponse(job_id=job_id, message="Analysis completed")


@app.get("/continuous-beam/jobs/{job_id}/status", response_class=HTMLResponse)
async def continuous_beam_job_status_page(request: Request, job_id: str):
    if job_id not in jobs_db:
        return templates.TemplateResponse("partials/job_not_found.html", {"request": request})
    job = jobs_db[job_id]
    return templates.TemplateResponse("continuous_beam_status.html", {"request": request, "job": job})


# ═══════════════════════════════════════════════════════════════════════════════
# Claude API endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/claude/status")
async def claude_status():
    """Check if Claude API is configured"""
    return {"available": check_api_key()}


@app.post("/api/claude/parse", response_model=ParseResponse)
async def parse_natural_language_input(input_data: NaturalLanguageInput):
    """Parse natural language to Frame2D input JSON"""
    try:
        result = parse_natural_language(input_data.text)
        return ParseResponse(success=True, data=result)
    except ValueError as e:
        return ParseResponse(success=False, error=str(e))
    except Exception as e:
        return ParseResponse(success=False, error=f"파싱 오류: {str(e)}")


@app.post("/api/claude/parse-beam", response_model=ParseResponse)
async def parse_beam_natural_language(input_data: NaturalLanguageInput):
    """Parse natural language to Simple Beam input JSON"""
    try:
        result = parse_simple_beam(input_data.text)
        return ParseResponse(success=True, data=result)
    except ValueError as e:
        return ParseResponse(success=False, error=str(e))
    except Exception as e:
        return ParseResponse(success=False, error=f"파싱 오류: {str(e)}")


@app.post("/api/claude/parse-continuous-beam", response_model=ParseResponse)
async def parse_continuous_beam_natural_language(input_data: NaturalLanguageInput):
    """Parse natural language to Continuous Beam input JSON"""
    try:
        result = parse_continuous_beam(input_data.text)
        return ParseResponse(success=True, data=result)
    except ValueError as e:
        return ParseResponse(success=False, error=str(e))
    except Exception as e:
        return ParseResponse(success=False, error=f"파싱 오류: {str(e)}")


@app.post("/api/claude/parse-building")
async def parse_building_natural_language(input_data: NaturalLanguageInput):
    """Parse natural language to BuildingIntent, then resolve to config"""
    try:
        # Step 1: Claude → BuildingIntent
        intent = parse_building(input_data.text)

        # Step 2: resolve_building_config
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))
        from core.nl_resolver import resolve_building_config
        resolved = resolve_building_config(intent)

        return {
            "success": True,
            "intent": intent,
            "resolved": resolved,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"파싱 오류: {str(e)}"}


class BuildingResolveInput(BaseModel):
    """Direct intent → config resolution (no Claude)"""
    intent: Dict[str, Any]


@app.post("/api/building/resolve-config")
async def resolve_building_config_api(body: BuildingResolveInput):
    """Resolve BuildingIntent to validated config (without Claude)"""
    try:
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))
        from core.nl_resolver import resolve_building_config
        resolved = resolve_building_config(body.intent)
        return {"success": True, "resolved": resolved}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3D Building Editor
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/editor", response_class=HTMLResponse)
async def editor_page(request: Request):
    """3D Building Editor page (Phase A)"""
    return templates.TemplateResponse("editor.html", {"request": request})


@app.get("/api/sections/list")
async def list_sections():
    """Return available sections grouped by type"""
    try:
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))
        from core.simple_beam import get_available_sections
        sections = get_available_sections()
        return {"sections": sections}
    except Exception as e:
        return {"sections": {"H형강": ["H-200x200", "H-250x250", "H-300x300", "H-350x350", "H-400x400"]},
                "fallback": True, "error": str(e)}


@app.get("/api/materials/list")
async def list_materials():
    """Return available materials"""
    try:
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))
        from core.simple_beam import get_available_materials
        materials = get_available_materials()
        return {"materials": materials}
    except Exception as e:
        return {"materials": ["SS275", "SS400", "SM355", "SM490"],
                "fallback": True, "error": str(e)}


@app.post("/api/building/analyze")
async def analyze_building_api(input_data: BuildingInput):
    """Run 3D building analysis and return results for the editor"""
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input config
    with open(job_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(input_data.config, f, indent=2, ensure_ascii=False)

    jobs_db[job_id] = {
        "job_id": job_id,
        "status": JobStatus.RUNNING,
        "progress": 10,
        "created_at": datetime.now().isoformat(),
        "analysis_type": "building_3d",
    }

    try:
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))

        from core.building_model import BuildingModel
        from core.load_generator import generate_all_loads
        from core.frame_3d import analyze_frame_3d_multi

        # 1. BuildingModel
        model = BuildingModel.from_json(input_data.config)
        jobs_db[job_id]["progress"] = 20

        # 2. Auto load generation
        load_result = generate_all_loads(model)
        jobs_db[job_id]["progress"] = 40

        # 3. Run 3D analysis
        kwargs = model.to_frame3d_kwargs()
        kwargs["load_cases"] = load_result["load_cases"]
        kwargs["load_combinations"] = load_result["load_combinations"]
        kwargs["modal_analysis"] = True

        multi = analyze_frame_3d_multi(**kwargs)
        jobs_db[job_id]["progress"] = 70

        # 4. Design check
        dc_result = None
        try:
            from core.design_check import run_design_check
            seismic_rpt = load_result["reports"].get("seismic")
            dc_result = run_design_check(multi, model, seismic_rpt)
        except Exception:
            pass

        # 5. Result interpretation
        interpretation = None
        if dc_result is not None:
            try:
                from core.result_interpreter import interpret_results
                interpretation = interpret_results(
                    dc_result, multi,
                    modal_analysis=multi.modal_analysis or None,
                )
            except Exception:
                pass

        jobs_db[job_id]["progress"] = 85

        # 6. Extract 3D model data for viewer
        # Use structural grid nodes (not sub-element internal nodes)
        viewer_nodes = []
        for n in multi.nodes:
            if isinstance(n, dict):
                viewer_nodes.append({
                    "id": n["id"],
                    "x": n.get("x_m", n.get("x", 0)),
                    "y": n.get("y_m", n.get("y", 0)),
                    "z": n.get("z_m", n.get("z", 0)),
                })
            else:
                viewer_nodes.append({"id": n.id, "x": n.x, "y": n.y, "z": n.z})

        # Build structural members from grid topology
        # (sub-elements reference internal OpenSees nodes not in viewer_nodes)
        viewer_elements = []
        node_by_pos = {}  # (x,y,z) -> node_id
        for vn in viewer_nodes:
            key = (round(vn["x"], 3), round(vn["y"], 3), round(vn["z"], 3))
            node_by_pos[key] = vn["id"]

        stories_h = list(multi.stories)
        bays_x = list(multi.bays_x)
        bays_y = list(multi.bays_y)
        nx = len(bays_x) + 1
        ny = len(bays_y) + 1
        ns = len(stories_h)

        # Compute grid coordinates
        x_coords = [0.0]
        for bx in bays_x:
            x_coords.append(x_coords[-1] + bx)
        y_coords = [0.0]
        for by_ in bays_y:
            y_coords.append(y_coords[-1] + by_)
        z_coords = [0.0]
        for sh in stories_h:
            z_coords.append(z_coords[-1] + sh)

        member_id = 0
        # Columns: vertical members
        for s in range(ns):
            for iy in range(ny):
                for ix in range(nx):
                    key_i = (round(x_coords[ix], 3), round(y_coords[iy], 3), round(z_coords[s], 3))
                    key_j = (round(x_coords[ix], 3), round(y_coords[iy], 3), round(z_coords[s + 1], 3))
                    ni_id = node_by_pos.get(key_i)
                    nj_id = node_by_pos.get(key_j)
                    if ni_id and nj_id:
                        member_id += 1
                        viewer_elements.append({
                            "id": member_id, "ni": ni_id, "nj": nj_id,
                            "type": "column", "section": multi.column_section,
                        })

        # Beams X: along X direction at each story level
        for s in range(1, ns + 1):
            for iy in range(ny):
                for ix in range(nx - 1):
                    key_i = (round(x_coords[ix], 3), round(y_coords[iy], 3), round(z_coords[s], 3))
                    key_j = (round(x_coords[ix + 1], 3), round(y_coords[iy], 3), round(z_coords[s], 3))
                    ni_id = node_by_pos.get(key_i)
                    nj_id = node_by_pos.get(key_j)
                    if ni_id and nj_id:
                        member_id += 1
                        viewer_elements.append({
                            "id": member_id, "ni": ni_id, "nj": nj_id,
                            "type": "beam_x", "section": multi.beam_x_section,
                        })

        # Beams Y: along Y direction at each story level
        for s in range(1, ns + 1):
            for iy in range(ny - 1):
                for ix in range(nx):
                    key_i = (round(x_coords[ix], 3), round(y_coords[iy], 3), round(z_coords[s], 3))
                    key_j = (round(x_coords[ix], 3), round(y_coords[iy + 1], 3), round(z_coords[s], 3))
                    ni_id = node_by_pos.get(key_i)
                    nj_id = node_by_pos.get(key_j)
                    if ni_id and nj_id:
                        member_id += 1
                        viewer_elements.append({
                            "id": member_id, "ni": ni_id, "nj": nj_id,
                            "type": "beam_y", "section": multi.beam_y_section,
                        })

        # 7. Envelope across combos
        env = {"max_dx_mm": 0, "max_dy_mm": 0, "max_dz_mm": 0,
               "max_drift_x": 0, "max_drift_y": 0,
               "max_moment_kNm": 0, "max_axial_kN": 0, "max_shear_kN": 0}
        for cn, cr in multi.combo_results.items():
            if abs(cr.max_displacement_x) > abs(env["max_dx_mm"]):
                env["max_dx_mm"] = round(cr.max_displacement_x, 3)
            if abs(cr.max_displacement_y) > abs(env["max_dy_mm"]):
                env["max_dy_mm"] = round(cr.max_displacement_y, 3)
            if abs(cr.max_displacement_z) > abs(env["max_dz_mm"]):
                env["max_dz_mm"] = round(cr.max_displacement_z, 3)
            if cr.max_drift_x > env["max_drift_x"]:
                env["max_drift_x"] = round(cr.max_drift_x, 6)
            if cr.max_drift_y > env["max_drift_y"]:
                env["max_drift_y"] = round(cr.max_drift_y, 6)
            if cr.max_moment > env["max_moment_kNm"]:
                env["max_moment_kNm"] = round(cr.max_moment, 2)
            if cr.max_axial > env["max_axial_kN"]:
                env["max_axial_kN"] = round(cr.max_axial, 2)
            if cr.max_shear > env["max_shear_kN"]:
                env["max_shear_kN"] = round(cr.max_shear, 2)

        # 8. Design check per-member results for coloring
        # Map design check member_id (1-based) to viewer structural member IDs
        member_checks = {}
        if dc_result and "member_check" in dc_result:
            ms = dc_result["member_check"]
            for mem in ms.get("members", []):
                mid = mem.get("member_id", mem.get("element_id", 0))
                ratios = mem.get("ratios", {})
                interaction = ratios.get("interaction", ratios.get("H1", 0))
                check_info = {
                    "status": mem.get("status", "OK"),
                    "interaction_ratio": interaction,
                    "governing": mem.get("governing_combo", ""),
                    "type": mem.get("type", ""),
                    "section": mem.get("section", ""),
                }
                # viewer member IDs match design check member_id (both 1-based structural members)
                member_checks[str(mid)] = check_info

        response = {
            "job_id": job_id,
            "status": "success",
            "building": model.summary(),
            "config": input_data.config,
            "viewer": {
                "nodes": viewer_nodes,
                "elements": viewer_elements,
                "stories": list(multi.stories),
                "bays_x": list(multi.bays_x),
                "bays_y": list(multi.bays_y),
                "total_height": multi.total_height,
                "total_width_x": multi.total_width_x,
                "total_width_y": multi.total_width_y,
                "column_section": multi.column_section,
                "beam_x_section": multi.beam_x_section,
                "beam_y_section": multi.beam_y_section,
                "material_name": multi.material_name,
            },
            "envelope": env,
            "design_check": dc_result,
            "interpretation": interpretation,
            "member_checks": member_checks,
            "modal_analysis": multi.modal_analysis,
        }

        # Generate HTML report
        report_url = None
        try:
            from core.visualization_3d import plot_frame_3d_interactive
            report_path = str(job_dir / "report.html")
            plot_frame_3d_interactive(
                multi,
                output_path=report_path,
                design_check=dc_result,
                interpretation=interpretation,
            )
            report_url = f"/api/jobs/{job_id}/report"
            jobs_db[job_id]["report_url"] = report_url
        except Exception as report_err:
            import traceback as tb
            print(f"Report generation warning: {report_err}")
            tb.print_exc()

        response["report_url"] = report_url

        # Store for re-analysis
        jobs_db[job_id]["config"] = input_data.config
        jobs_db[job_id]["response"] = response
        jobs_db[job_id]["status"] = JobStatus.DONE
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()

        return response

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error"] = str(e)
        print(f"Building analysis error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/building/parse-ifc")
async def parse_ifc_upload(file: UploadFile = File(...)):
    """Parse uploaded IFC file and extract building geometry"""
    import tempfile, os

    if not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="IFC 파일(.ifc)만 업로드할 수 있습니다.")

    # Save to temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ifc")
    try:
        content = await file.read()
        os.write(tmp_fd, content)
        os.close(tmp_fd)

        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))
        from core.ifc_parser import parse_ifc

        ifc_data = parse_ifc(tmp_path)

        # Convert to editor-friendly format
        stories = []
        for s in ifc_data.get("stories", []):
            stories.append({
                "name": s.get("name", ""),
                "height": s.get("height", 3.5),
                "usage": "office",  # IFC doesn't carry usage info
                "slab_thickness_mm": s.get("slab_thickness_mm"),
            })

        # Compute grid coordinates from bay dimensions (for 3D wireframe)
        bays_x = ifc_data.get("bays_x", [])
        bays_y = ifc_data.get("bays_y", [])
        grid_x = [0.0]
        for bx in bays_x:
            grid_x.append(grid_x[-1] + bx)
        grid_y = [0.0]
        for by in bays_y:
            grid_y.append(grid_y[-1] + by)

        result = {
            "success": True,
            "stories": stories,
            "bays_x": bays_x,
            "bays_y": bays_y,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "detected_sections": ifc_data.get("detected_sections", {}),
            "detected_material": ifc_data.get("detected_material"),
            "grid_source": ifc_data.get("grid_source", "unknown"),
            "num_columns": ifc_data.get("num_columns", 0),
            "num_walls": ifc_data.get("num_walls", 0),
            "warnings": ifc_data.get("warnings", []),
            "summary": {
                "num_stories": len(stories),
                "num_bays_x": len(bays_x),
                "num_bays_y": len(bays_y),
                "total_height": sum(s.get("height", 0) for s in stories),
                "filename": file.filename,
            },
        }
        return result

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="ifcopenshell 패키지가 설치되지 않았습니다. pip install ifcopenshell"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IFC 파싱 오류: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.patch("/api/building/{job_id}")
async def reanalyze_building(job_id: str, modifications: BuildingModification):
    """Re-analyze building with modified sections/materials"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    original_config = jobs_db[job_id].get("config")
    if not original_config:
        raise HTTPException(status_code=400, detail="No original config found")

    # Apply modifications to config copy
    new_config = json.loads(json.dumps(original_config))
    if modifications.column_section:
        new_config["column_section"] = modifications.column_section
    if modifications.beam_x_section:
        new_config["beam_x_section"] = modifications.beam_x_section
    if modifications.beam_y_section:
        new_config["beam_y_section"] = modifications.beam_y_section
    if modifications.material_name:
        new_config["material_name"] = modifications.material_name

    # Re-run analysis with new config
    new_input = BuildingInput(config=new_config)
    return await analyze_building_api(new_input)


@app.get("/api/building/{job_id}")
async def get_building_result(job_id: str):
    """Get stored building analysis result"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    response = jobs_db[job_id].get("response")
    if not response:
        raise HTTPException(status_code=400, detail="No results available")
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "simple (no Redis)"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("OpenSees Structural Analysis Platform")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
