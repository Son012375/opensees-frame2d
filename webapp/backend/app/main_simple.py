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
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

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

# Demo auth - disabled for now
# from app.core.auth import check_demo_auth, make_auth_response, set_auth_cookie

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

@app.get("/model-v2", response_class=HTMLResponse)
async def model_v2_page(request: Request):
    """V2 Node-Element Model page (simple version)"""
    return templates.TemplateResponse("model_v2.html", {"request": request})


@app.get("/editor-v2", response_class=HTMLResponse)
async def editor_v2_page(request: Request):
    """V2 3D Building Editor (V1 UI + V2 IFC Parser)"""
    return templates.TemplateResponse("editor_v2.html", {"request": request})


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


@app.get("/api/sections/properties/{section_name:path}")
async def get_section_props(section_name: str):
    """Return section properties (A, Ix, Iy, J, h, b, tw, tf) for a given section"""
    try:
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))
        from core.section_3d import get_section_3d
        sec = get_section_3d(section_name)
        # 단면 타입 판별
        stype = "H"
        t_mm = 0.0
        if section_name.startswith("□-") or section_name.startswith("\u25a1-"):
            stype = "SHS" if abs(sec.h - sec.b) < 1 else "RHS"
            t_mm = sec.tw  # SHS/RHS: tw에 두께 저장됨
        elif section_name.startswith("○-") or section_name.startswith("\u25cb-"):
            stype = "CHS"
            t_mm = sec.tw
        elif section_name.startswith("L-"):
            stype = "L"
        elif section_name.startswith("TFC-") or section_name.startswith("PFC-"):
            stype = "C"
        elif section_name.startswith("T-"):
            stype = "T"
        return {
            "name": sec.name,
            "section_type": stype,
            "A_cm2": round(sec.A / 100, 2),
            "Ix_cm4": round(sec.Ix / 10000, 1),
            "Iy_cm4": round(sec.Iy / 10000, 1),
            "J_cm4": round(sec.J / 10000, 1),
            "h_mm": round(sec.h, 1),
            "b_mm": round(sec.b, 1),
            "tw_mm": round(sec.tw, 1),
            "tf_mm": round(sec.tf, 1),
            "t_mm": round(t_mm, 1),
            "j_source": sec.j_source,
        }
    except Exception as e:
        return {"error": str(e), "name": section_name}


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

        # Build structural members from member_info (supports both regular and irregular)
        viewer_elements = []
        _section_map = {
            "column": multi.column_section,
            "beam_x": multi.beam_x_section,
            "beam_y": multi.beam_y_section,
        }
        viewer_node_ids = {vn["id"] for vn in viewer_nodes}
        for minfo in (multi.member_info or []):
            ni_id = minfo.get("ni", 0)
            nj_id = minfo.get("nj", 0)
            etype = minfo.get("elem_type", "column")
            if ni_id in viewer_node_ids and nj_id in viewer_node_ids:
                viewer_elements.append({
                    "id": minfo["member_id"],
                    "ni": ni_id,
                    "nj": nj_id,
                    "type": etype,
                    "section": _section_map.get(etype, ""),
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

        # 7b. Per-case/combo results for frontend case selector
        case_names = list(multi.case_results.keys())
        combo_names = list(multi.combo_results.keys())
        case_data = {}

        def _extract_case_summary(cr):
            """Extract summary + per-node displacements from a case result."""
            summary = {
                "max_dx_mm": round(cr.max_displacement_x, 3),
                "max_dy_mm": round(cr.max_displacement_y, 3),
                "max_dz_mm": round(getattr(cr, 'max_displacement_z', 0), 3),
                "max_drift_x": round(cr.max_drift_x, 6),
                "max_drift_y": round(cr.max_drift_y, 6),
                "max_moment_kNm": round(cr.max_moment, 2),
                "max_axial_kN": round(cr.max_axial, 2),
                "max_shear_kN": round(cr.max_shear, 2),
            }
            displacements = {}
            for nd in cr.nodal_displacements:
                nid = nd.get("node_id", nd.get("node", 0))
                displacements[str(nid)] = [
                    round(nd.get("dx_mm", 0), 3),
                    round(nd.get("dy_mm", 0), 3),
                    round(nd.get("dz_mm", 0), 3),
                ]
            drifts = []
            for sd in cr.story_drifts:
                drifts.append({
                    "story": sd.get("story", 0),
                    "drift_x": round(sd.get("drift_x", 0), 6),
                    "drift_y": round(sd.get("drift_y", 0), 6),
                })
            return {"summary": summary, "displacements": displacements, "story_drifts": drifts}

        for cn, cr in multi.case_results.items():
            case_data[cn] = _extract_case_summary(cr)
        for cn, cr in multi.combo_results.items():
            case_data[cn] = _extract_case_summary(cr)

        # 7c. Response Spectrum Analysis (if requested)
        rsa_result = None
        seismic_method = input_data.config.get("seismic_method", "ELF")
        if seismic_method == "RSA" and multi.modal_analysis:
            try:
                from core.design_spectrum import compute_design_spectrum
                from core.response_spectrum_analysis import (
                    run_response_spectrum_analysis, rsa_result_to_case_data,
                )
                # Get seismic system parameters
                seismic_sys = input_data.config.get("seismic_system", "ordinary_moment_frame")
                R_map = {"special_moment_frame": 8.0, "intermediate_moment_frame": 4.5,
                         "ordinary_moment_frame": 3.5, "special_concentric_braced": 8.0}
                R = R_map.get(seismic_sys, 3.5)
                IE = model.importance_factor
                region = model.region or "서울"
                site_class = model.site_class or "S3"

                spectrum = compute_design_spectrum(region, site_class, IE)
                if spectrum.get("status") == "success":
                    rsa_result = run_response_spectrum_analysis(
                        modal_analysis=multi.modal_analysis,
                        spectrum_result=spectrum,
                        viewer_nodes=viewer_nodes,
                        stories=list(multi.stories),
                        bays_x=list(multi.bays_x),
                        bays_y=list(multi.bays_y),
                        combination_rule=input_data.config.get("rsa_combination", "CQC"),
                        direction_rule=input_data.config.get("rsa_direction", "30pct"),
                        IE=IE, R=R,
                    )
                    # Merge RSA case_data
                    rsa_cases = rsa_result_to_case_data(rsa_result)
                    case_data.update(rsa_cases)
                    case_names.append("__RSA__")  # Marker for frontend
                    combo_names.extend(rsa_cases.keys())
            except Exception as e:
                import traceback
                traceback.print_exc()

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
                "is_irregular": multi.analysis_metadata.get("irregular", False),
                "zones": input_data.config.get("zones"),
            },
            "envelope": env,
            "case_names": case_names,
            "combo_names": combo_names,
            "case_data": case_data,
            "design_check": dc_result,
            "interpretation": interpretation,
            "member_checks": member_checks,
            "modal_analysis": multi.modal_analysis,
            "rsa": {
                "enabled": rsa_result is not None,
                "combination_rule": rsa_result.combination_rule if rsa_result else None,
                "direction_rule": rsa_result.direction_rule if rsa_result else None,
                "num_modes_used": rsa_result.num_modes_used if rsa_result else 0,
                "parameters": rsa_result.parameters if rsa_result else {},
                "story_drifts": rsa_result.story_drifts if rsa_result else [],
            } if seismic_method == "RSA" else None,
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

    # File size limit (50MB)
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기가 50MB를 초과합니다.")

    # Save to temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ifc")
    try:
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

        detected_zones = ifc_data.get("detected_zones")  # zone-based irregular detection

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
            "detected_zones": detected_zones,
            "is_irregular": detected_zones is not None,
            "summary": {
                "num_stories": len(stories),
                "num_bays_x": len(bays_x),
                "num_bays_y": len(bays_y),
                "total_height": sum(s.get("height", 0) for s in stories),
                "filename": file.filename,
                "is_irregular": detected_zones is not None,
                "num_zones": len(detected_zones) if detected_zones else 0,
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
# V2 Node-Element Pipeline APIs
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v2/parse-ifc")
async def parse_ifc_v2_api(file: UploadFile = File(...)):
    """V2: IFC → Node-Element StructuralModel 변환"""
    import tempfile, os

    if not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="IFC 파일(.ifc)만 업로드할 수 있습니다.")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기가 50MB를 초과합니다.")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ifc")
    try:
        os.write(tmp_fd, content)
        os.close(tmp_fd)

        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))
        from core.ifc_parser_v2 import parse_ifc_v2
        from core.visualization_v2 import generate_model_viewer

        model, validation = parse_ifc_v2(tmp_path)

        # HTML 뷰어 생성
        viewer_path = None
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            viewer_dir = JOBS_DIR / "v2_viewers"
            viewer_dir.mkdir(parents=True, exist_ok=True)
            viewer_path = generate_model_viewer(
                model,
                output_path=str(viewer_dir / f"model_{ts}.html"),
                title=f"IFC Model: {file.filename}",
            )
        except Exception:
            pass

        result = {
            "success": validation.is_valid,
            "filename": file.filename,
            "model": model.to_json(),
            "validation": {
                "is_valid": validation.is_valid,
                "summary": validation.summary_text(),
                "extracted_nodes": validation.extracted_nodes,
                "extracted_elements": validation.extracted_elements,
                "failed_elements": validation.failed_elements,
                "issues": [
                    {"severity": i.severity.value, "code": i.code,
                     "message": i.message, "default_value": i.default_value}
                    for i in validation.issues
                ],
                "needs_user_input": [
                    {"code": i.code, "message": i.message, "default_value": i.default_value}
                    for i in validation.needs_user_input
                ],
            },
            "summary": model.summary(),
        }
        if viewer_path:
            result["viewer_url"] = f"/api/v2/viewer/{Path(viewer_path).name}"

        return result

    except ImportError:
        raise HTTPException(status_code=500, detail="ifcopenshell 패키지가 필요합니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IFC V2 파싱 오류: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/api/v2/snap-joints")
async def snap_joints_api(request: Request):
    """V2: 보-기둥 접합 스냅"""
    body = await request.json()
    model_json = body.get("model")
    snap_tolerance = body.get("snap_tolerance", 0.5)

    if not model_json:
        raise HTTPException(status_code=400, detail="model JSON이 필요합니다.")

    if str(MCP_SERVER_PATH) not in sys.path:
        sys.path.insert(0, str(MCP_SERVER_PATH))
    from core.structural_model import StructuralModel
    from core.ifc_parser_v2 import snap_beams_to_columns

    model = StructuralModel.from_json(model_json)
    nodes_before = len(model.nodes)
    snapped = snap_beams_to_columns(model, snap_tolerance=snap_tolerance)

    return {
        "success": True,
        "snapped_count": snapped,
        "nodes_before": nodes_before,
        "nodes_after": len(model.nodes),
        "model": model.to_json(),
        "summary": model.summary(),
    }


@app.post("/api/v2/analyze")
async def analyze_v2_api(request: Request):
    """V2: 통합 해석 (KDS 하중자동생성 + 해석 + 설계검토 + 리포트).

    V1의 /api/building/analyze와 동일한 기능을 V2 StructuralModel에서 수행.
    """
    body = await request.json()
    model_json = body.get("model")
    user_config = body.get("config", {})

    if not model_json:
        raise HTTPException(status_code=400, detail="model JSON이 필요합니다.")

    if str(MCP_SERVER_PATH) not in sys.path:
        sys.path.insert(0, str(MCP_SERVER_PATH))

    from core.structural_model import StructuralModel
    from core.frame_3d import analyze_from_model

    model = StructuralModel.from_json(model_json)

    # 요소 분류 확인 + 자동 분류
    from core.structural_model import ElementType as _ET
    type_set = set(e.elem_type for e in model.elements.values())
    print(f"[V2] Element types before classify: {[t.value for t in type_set]}")
    if len(type_set) == 1 and len(model.elements) > 3:
        model.classify_elements()
        type_set2 = set(e.elem_type for e in model.elements.values())
        print(f"[V2] Auto-classified: {[t.value for t in type_set2]}")

    # A-1: 근접 노드 병합 (보-기둥 접합 보장)
    merged = model.merge_nearby_nodes()  # 적응형: 단면 높이 기반 자동 계산
    if merged > 0:
        print(f"[V2] Merged {merged} nearby nodes")

    # 보-보 교차점 자동 분할
    intersections = model.split_at_intersections()
    if intersections > 0:
        print(f"[V2] Created {intersections} intersection nodes, model: {len(model.nodes)} nodes, {len(model.elements)} elems")

    # 사용자 config 반영
    for key in ["region", "site_class", "importance", "seismic_system", "exposure_category", "geometric_nonlinearity"]:
        if user_config.get(key):
            setattr(model, key, user_config[key])
    if user_config.get("importance"):
        model.importance_factor = {"특": 1.5, "I": 1.2, "II": 1.0}.get(model.importance, 1.0)
    if user_config.get("rigid_diaphragm") is not None:
        model.rigid_diaphragm = user_config["rigid_diaphragm"]

    # 층별 용도/슬래브 설정
    for sc in user_config.get("stories", []):
        s_idx = sc.get("story")
        if s_idx:
            if sc.get("usage"): model.story_usages[s_idx] = sc["usage"]
            if sc.get("slab_thickness"): model.story_slab_thickness[s_idx] = sc["slab_thickness"]
            if sc.get("dead_load_finish") is not None: model.story_dead_load_finish[s_idx] = sc["dead_load_finish"]

    # 기본값 채우기
    for s_idx in range(1, model.num_stories + 1):
        model.story_usages.setdefault(s_idx, "office")
        model.story_slab_thickness.setdefault(s_idx, 0.15)
        model.story_dead_load_finish.setdefault(s_idx, 1.0)

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: 하중 생성 (V1 load_generator 사용) ──
        building_model = model.to_building_model()

        from core.load_generator import generate_all_loads
        load_result = generate_all_loads(building_model)
        load_cases = load_result["load_cases"]
        load_combinations = load_result.get("load_combinations", {})

        # ── Step 2: 해석 ──
        multi = analyze_from_model(model, load_cases, load_combinations)

        # ── Step 3: 설계검토 ──
        dc_result = None
        interpretation = None
        try:
            from core.design_check import run_design_check
            seismic_rpt = load_result.get("reports", {}).get("seismic")
            dc_result = run_design_check(multi, building_model, seismic_rpt)
        except Exception as e:
            print(f"Design check: {e}")

        try:
            from core.result_interpreter import interpret_results
            if dc_result:
                interpretation = interpret_results(dc_result, multi,
                    modal_analysis=getattr(multi, 'modal_analysis', None) or None)
        except Exception:
            pass

        # ── Step 4: HTML 리포트 ──
        report_path = str(job_dir / "report.html")
        try:
            from core.visualization_3d import plot_frame_3d_interactive
            plot_frame_3d_interactive(multi, output_path=report_path,
                                      design_check=dc_result, interpretation=interpretation)
        except Exception as viz_err:
            print(f"V1 report failed ({viz_err}), trying V2 viewer")
            try:
                from core.visualization_v2 import generate_model_viewer
                generate_model_viewer(model, result=multi, output_path=report_path,
                                      title="V2 Analysis Results")
            except Exception:
                pass

        report_url = f"/api/jobs/{job_id}/report" if Path(report_path).exists() else None

        # ── Step 5: 응답 ──
        env = {"max_dx_mm": 0, "max_dy_mm": 0, "max_dz_mm": 0,
               "max_drift_x": 0, "max_drift_y": 0,
               "max_moment_kNm": 0, "max_axial_kN": 0, "max_shear_kN": 0}
        for cr in list(multi.case_results.values()) + list(multi.combo_results.values()):
            env["max_dx_mm"] = max(env["max_dx_mm"], abs(cr.max_displacement_x))
            env["max_dy_mm"] = max(env["max_dy_mm"], abs(cr.max_displacement_y))
            env["max_dz_mm"] = max(env["max_dz_mm"], abs(cr.max_displacement_z))
            env["max_drift_x"] = max(env["max_drift_x"], abs(cr.max_drift_x))
            env["max_drift_y"] = max(env["max_drift_y"], abs(cr.max_drift_y))
            env["max_moment_kNm"] = max(env["max_moment_kNm"], abs(cr.max_moment))
            env["max_axial_kN"] = max(env["max_axial_kN"], abs(cr.max_axial))
            env["max_shear_kN"] = max(env["max_shear_kN"], abs(cr.max_shear))

        case_data = {}
        for cname, cr in {**multi.case_results, **multi.combo_results}.items():
            # 변위 데이터 (에디터 deformed shape용)
            disp_map = {}
            for d in cr.nodal_displacements:
                disp_map[str(d["node"])] = [d["dx_mm"], d["dy_mm"], d["dz_mm"]]
            case_data[cname] = {
                "summary": {"max_dx_mm": cr.max_displacement_x, "max_dy_mm": cr.max_displacement_y,
                            "max_moment_kNm": cr.max_moment, "max_axial_kN": cr.max_axial},
                "story_drifts": cr.story_drifts,
                "reactions": cr.reactions,
                "displacements": disp_map,
            }

        # ── Response Spectrum Analysis (if requested) ──
        rsa_result = None
        seismic_method = user_config.get("seismic_method", "ELF")
        modal_analysis_result = getattr(multi, 'modal_analysis', None) or None
        extra_combo_names = []
        if seismic_method == "RSA" and modal_analysis_result:
            try:
                from core.design_spectrum import compute_design_spectrum
                from core.response_spectrum_analysis import (
                    run_response_spectrum_analysis, rsa_result_to_case_data,
                )
                seismic_sys = user_config.get("seismic_system", model.seismic_system)
                R_map = {"special_moment_frame": 8.0, "intermediate_moment_frame": 4.5,
                         "ordinary_moment_frame": 3.5, "special_concentric_braced": 8.0}
                R = R_map.get(seismic_sys, 3.5)
                IE = model.importance_factor
                region = model.region or "서울"
                site_class = model.site_class or "S3"

                spectrum = compute_design_spectrum(region, site_class, IE)
                if spectrum.get("status") == "success":
                    # V2 viewer_nodes format
                    viewer_nodes_rsa = [
                        {"id": n.id, "x_m": n.x, "y_m": n.y, "z_m": n.z}
                        for n in sorted(model.nodes.values(), key=lambda n: n.id)
                    ]
                    rsa_result = run_response_spectrum_analysis(
                        modal_analysis=modal_analysis_result,
                        spectrum_result=spectrum,
                        viewer_nodes=viewer_nodes_rsa,
                        stories=list(multi.stories),
                        bays_x=list(multi.bays_x) if multi.bays_x else [8.0],
                        bays_y=list(multi.bays_y) if multi.bays_y else [8.0],
                        combination_rule=user_config.get("rsa_combination", "CQC"),
                        direction_rule=user_config.get("rsa_direction", "30pct"),
                        IE=IE, R=R,
                    )
                    # Merge RSA case_data
                    rsa_cases = rsa_result_to_case_data(rsa_result)
                    case_data.update(rsa_cases)
                    extra_combo_names = list(rsa_cases.keys())
            except Exception as rsa_err:
                print(f"V2 RSA error: {rsa_err}")
                import traceback
                traceback.print_exc()

        member_checks = {}
        if dc_result and "member_check" in dc_result:
            for mem in dc_result["member_check"].get("members", []):
                member_checks[str(mem.get("member_id", ""))] = {
                    "status": mem.get("status", "OK"),
                    "interaction_ratio": mem.get("ratios", {}).get("interaction", 0),
                }

        response = {
            "job_id": job_id,
            "status": "success",
            "pipeline": "v2_node_element",
            "building": model.summary(),
            "updated_model": model.to_json(),  # split 후 모델 (JS 동기화용)
            "envelope": env,
            "case_names": list(multi.case_results.keys()) + (["__RSA__"] if rsa_result else []),
            "combo_names": list(multi.combo_results.keys()) + extra_combo_names,
            "case_data": case_data,
            "load_summary": load_result.get("summary", {}),
            "load_cases_raw": load_result.get("load_cases", {}),
            "member_forces": multi.member_forces,
            "member_info": multi.member_info,
            "design_check": dc_result,
            "interpretation": interpretation,
            "member_checks": member_checks,
            "modal_analysis": modal_analysis_result,
            "rsa": rsa_result,
            "seismic_method": seismic_method,
            "report_url": report_url,
        }

        # Job DB 저장
        jobs_db[job_id] = {
            "job_id": job_id, "status": "done",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "report_url": report_url,
            "summary": {"max_displacement_x_mm": env["max_dx_mm"], "max_displacement_y_mm": env["max_dy_mm"],
                        "max_drift": env["max_drift_x"], "max_drift_story": 1,
                        "max_moment_kNm": env["max_moment_kNm"], "max_shear_kN": env["max_shear_kN"],
                        "max_axial_kN": env["max_axial_kN"], "base_shear_kN": 0,
                        "num_stories": model.num_stories, "num_bays": 0},
        }

        # Export용 결과 디스크 저장 (case_data, member_forces, envelope 등)
        try:
            export_payload = {
                "job_id": job_id,
                "created_at": jobs_db[job_id]["created_at"],
                "envelope": env,
                "case_names": response["case_names"],
                "combo_names": response["combo_names"],
                "case_data": case_data,
                "member_forces": multi.member_forces,
                "member_info": multi.member_info,
                "nodes_info": {str(n.id): [n.x, n.y, n.z] for n in model.nodes.values()},
                "num_stories": model.num_stories,
                "num_nodes": len(model.nodes),
                "num_elements": len(model.elements),
                "analysis_type": "v2_building",
                "seismic_method": seismic_method,
            }
            with open(job_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(export_payload, f, ensure_ascii=False, default=str)
        except Exception as save_err:
            print(f"Result save failed: {save_err}")

        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"V2 해석 오류: {str(e)}")


@app.get("/api/v2/viewer/{filename}")
async def serve_v2_viewer(filename: str):
    """V2 HTML 뷰어/리포트 서빙"""
    viewer_path = JOBS_DIR / "v2_viewers" / filename
    if not viewer_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(viewer_path), media_type="text/html")


# ═══════════════════════════════════════════════════════════════════════════════
# Excel Export
# ═══════════════════════════════════════════════════════════════════════════════

# 케이스 그룹별 헤더 배경/글자 색 (hex, 6자리)
_GROUP_COLORS = {
    'Info':   ('37474F', 'FFFFFF'),  # 짙은 회색
    'DL':     ('0097A7', 'FFFFFF'),  # 시안
    'LL':     ('F57C00', 'FFFFFF'),  # 주황
    'EQ':     ('C62828', 'FFFFFF'),  # 빨강
    'Wind':   ('6A1B9A', 'FFFFFF'),  # 보라
    'RSA':    ('1565C0', 'FFFFFF'),  # 파랑
    'Combo0': ('E8F5E9', '000000'),  # 연녹 1
    'Combo1': ('F1F8E9', '000000'),  # 연녹 2
}


def _case_group(cname: str) -> str:
    """케이스 이름 → 그룹 분류."""
    if cname == 'Info':
        return 'Info'
    cu = cname.upper()
    if 'RSA' in cu:
        return 'RSA'
    if 'WIND' in cu or cu.startswith('W'):
        return 'Wind'
    if 'EQ' in cu or cu.startswith('S_'):
        return 'EQ'
    if cu.startswith('DL') or cu == 'DEAD':
        return 'DL'
    if cu.startswith('LL') or cu == 'LIVE':
        return 'LL'
    return 'Combo'


def _style_wide_header(ws, case_order, info_col_names, n_metrics):
    """2행 MultiIndex 헤더에 그룹별 색 적용.

    case_order: Info 제외한 케이스 이름 리스트
    info_col_names: ['node_id', 'x_m', ...] (Info 레벨 컬럼 이름들)
    n_metrics: 각 case 블록의 컬럼 수
    """
    from openpyxl.styles import PatternFill, Font, Alignment

    col = 1
    combo_idx = 0
    # Info 컬럼 먼저
    bg, fg = _GROUP_COLORS['Info']
    info_fill = PatternFill('solid', fgColor=bg)
    info_font = Font(bold=True, color=fg)
    for _ in info_col_names:
        for r in (1, 2):
            cell = ws.cell(row=r, column=col)
            cell.fill = info_fill
            cell.font = info_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
        col += 1

    # 각 case 블록
    for cname in case_order:
        grp = _case_group(cname)
        if grp == 'Combo':
            fill_key = f'Combo{combo_idx % 2}'
            combo_idx += 1
        else:
            fill_key = grp
        bg, fg = _GROUP_COLORS[fill_key]
        fill = PatternFill('solid', fgColor=bg)
        font = Font(bold=True, color=fg)
        for r in (1, 2):
            for c in range(col, col + n_metrics):
                cell = ws.cell(row=r, column=c)
                cell.fill = fill
                cell.font = font
                cell.alignment = Alignment(horizontal='center', vertical='center')
        col += n_metrics

    ws.freeze_panes = 'A3'


def _write_wide_sheet(writer, sheet_name, case_names, row_keys,
                      info_col_names, info_values, metric_keys, fetch_value):
    """openpyxl로 직접 wide sheet 쓰기 (MultiIndex + index=False 제약 회피).

    헤더 1행: Info 컬럼은 각 컬럼명 + case 블록은 첫 셀에 케이스명(merge)
    헤더 2행: Info 컬럼은 비움 + case 블록은 metric 이름들
    3행~: 데이터
    """
    from openpyxl import Workbook
    # pandas ExcelWriter.sheets는 기존 워크북 공유 → 새 시트 생성
    wb = writer.book
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(sheet_name)
    writer.sheets[sheet_name] = ws

    n_info = len(info_col_names)
    n_metrics = len(metric_keys)

    # 1행 헤더: Info 컬럼명 + case별 merge 셀
    for i, name in enumerate(info_col_names):
        ws.cell(row=1, column=i + 1, value=name)
    col = n_info + 1
    for cname in case_names:
        ws.cell(row=1, column=col, value=cname)
        if n_metrics > 1:
            ws.merge_cells(start_row=1, start_column=col,
                           end_row=1, end_column=col + n_metrics - 1)
        col += n_metrics

    # 2행 헤더: Info 컬럼은 이미 1행이 이름이라 빈 셀(또는 동일)도 가능 —
    # 보기 좋게 Info는 1행 이름 그대로 두고 2행도 동일값으로 유지 (merge 없이)
    for i, name in enumerate(info_col_names):
        ws.cell(row=2, column=i + 1, value=name)
    col = n_info + 1
    for _ in case_names:
        for m in metric_keys:
            ws.cell(row=2, column=col, value=m)
            col += 1

    # Info 컬럼 1~2행 세로 병합
    for i in range(n_info):
        ws.merge_cells(start_row=1, start_column=i + 1,
                       end_row=2, end_column=i + 1)

    # 3행~ 데이터
    for ri, rk in enumerate(row_keys):
        r = ri + 3
        for ci, name in enumerate(info_col_names):
            ws.cell(row=r, column=ci + 1, value=info_values[name][ri])
        col = n_info + 1
        for cname in case_names:
            for m in metric_keys:
                ws.cell(row=r, column=col, value=fetch_value(cname, rk, m))
                col += 1


@app.get("/api/export/excel/{job_id}")
async def export_excel(job_id: str):
    """해석 결과를 Excel 다중 시트(.xlsx)로 다운로드. Wide Pivot 구조."""
    result_path = JOBS_DIR / job_id / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="해석 결과를 찾을 수 없습니다. 해석을 다시 실행해주세요.")

    try:
        import pandas as pd
        from io import BytesIO
    except ImportError:
        raise HTTPException(status_code=500, detail="pandas/openpyxl 미설치. `pip install pandas openpyxl` 실행 필요.")

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    case_data = data.get("case_data", {}) or {}
    member_forces = data.get("member_forces", {}) or {}
    envelope = data.get("envelope", {}) or {}
    nodes_info = data.get("nodes_info", {}) or {}

    def _coord(nid, axis):
        """nid의 x/y/z (없으면 None)."""
        v = nodes_info.get(str(nid))
        if not v:
            return None
        return v[axis] if axis < len(v) else None

    def _nid_sort_key(nid):
        try:
            return (0, int(nid))
        except (ValueError, TypeError):
            return (1, str(nid))

    # ─── 1) Displacements (Wide) ─────────────────────────────
    disp_metrics = ["dx_mm", "dy_mm", "dz_mm", "rx_rad", "ry_rad", "rz_rad"]
    disp_case_order = list(case_data.keys())
    all_disp_nodes = set()
    for cd in case_data.values():
        all_disp_nodes.update((cd.get("displacements") or {}).keys())
    all_disp_nodes = sorted(all_disp_nodes, key=_nid_sort_key)

    def _disp_get(cname, nid, m):
        vals = (case_data.get(cname, {}).get("displacements") or {}).get(str(nid))
        if not vals:
            return None
        idx = disp_metrics.index(m)
        return vals[idx] if idx < len(vals) else None

    disp_info_cols = ["node_id", "x_m", "y_m", "z_m"]
    disp_info_values = {
        "node_id": list(all_disp_nodes),
        "x_m": [_coord(n, 0) for n in all_disp_nodes],
        "y_m": [_coord(n, 1) for n in all_disp_nodes],
        "z_m": [_coord(n, 2) for n in all_disp_nodes],
    }

    # ─── 2) Member Forces (Wide) ─────────────────────────────
    force_metrics = ["N_kN", "Vy_kN", "Vz_kN", "T_kNm", "My_kNm", "Mz_kNm"]
    force_case_order = list(member_forces.keys())

    # 모든 부재 수집 + 첫 등장 case에서 info 추출
    member_info_map = {}  # member_id -> {ni, nj, type, section}
    for cname in force_case_order:
        mf_list = member_forces.get(cname, [])
        if not isinstance(mf_list, list):
            continue
        for mf in mf_list:
            mid = mf.get("member_id")
            if mid is None or mid in member_info_map:
                continue
            member_info_map[mid] = {
                "ni": mf.get("ni"),
                "nj": mf.get("nj"),
                "type": mf.get("type"),
                "section": mf.get("section"),
            }
    all_members = sorted(member_info_map.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)

    # member_id → case → {metric: value} 맵 사전 구축 (O(N) lookup)
    force_lookup = {}  # (cname, mid) -> mf dict
    for cname in force_case_order:
        mf_list = member_forces.get(cname, [])
        if not isinstance(mf_list, list):
            continue
        for mf in mf_list:
            mid = mf.get("member_id")
            if mid is not None:
                force_lookup[(cname, mid)] = mf

    def _signed_peak(v):
        """스칼라면 그대로, 배열이면 절댓값 최대(부호 유지)."""
        if isinstance(v, (list, tuple)):
            if not v:
                return None
            vals = [x for x in v if x is not None]
            if not vals:
                return None
            return max(vals, key=lambda x: abs(x))
        return v

    def _force_get(cname, mid, m):
        mf = force_lookup.get((cname, mid))
        if not mf:
            return None
        return _signed_peak(mf.get(m))

    force_info_cols = ["member_id", "ni", "nj", "type", "section"]
    force_info_values = {
        "member_id": list(all_members),
        "ni": [member_info_map[m]["ni"] for m in all_members],
        "nj": [member_info_map[m]["nj"] for m in all_members],
        "type": [member_info_map[m]["type"] for m in all_members],
        "section": [member_info_map[m]["section"] for m in all_members],
    }

    # ─── 3) Reactions (Wide) ─────────────────────────────────
    react_metrics = ["RX_kN", "RY_kN", "RZ_kN", "MX_kNm", "MY_kNm", "MZ_kNm"]
    react_case_order = list(case_data.keys())

    # 지점 노드 수집 + 각 case의 lookup 구축
    react_lookup = {}  # (cname, node_id) -> r dict
    all_react_nodes = set()
    for cname in react_case_order:
        reactions = case_data.get(cname, {}).get("reactions") or []
        for r in reactions:
            nid = r.get("node_id") if "node_id" in r else r.get("node")
            if nid is None:
                continue
            all_react_nodes.add(str(nid))
            react_lookup[(cname, str(nid))] = r
    all_react_nodes = sorted(all_react_nodes, key=_nid_sort_key)

    def _react_get(cname, nid, m):
        r = react_lookup.get((cname, str(nid)))
        if not r:
            return None
        return _signed_peak(r.get(m))

    def _react_coord(nid, axis):
        """반력 노드 좌표: nodes_info 우선, fallback은 첫 case의 reaction 항목."""
        v = nodes_info.get(str(nid))
        if v:
            return v[axis] if axis < len(v) else None
        # fallback: 첫 등장 case의 좌표
        for cname in react_case_order:
            r = react_lookup.get((cname, str(nid)))
            if r:
                keys = ["x_m", "y_m", "z_m"]
                return r.get(keys[axis])
        return None

    react_info_cols = ["node_id", "x_m", "y_m", "z_m"]
    react_info_values = {
        "node_id": list(all_react_nodes),
        "x_m": [_react_coord(n, 0) for n in all_react_nodes],
        "y_m": [_react_coord(n, 1) for n in all_react_nodes],
        "z_m": [_react_coord(n, 2) for n in all_react_nodes],
    }

    # ─── 4) Envelope (기존 long-form 유지) ───────────────────
    env_rows = [{"metric": k, "value": v} for k, v in envelope.items()]
    df_env = pd.DataFrame(env_rows)

    # ─── 5) Metadata ─────────────────────────────────────────
    meta_rows = [
        {"key": "job_id", "value": data.get("job_id", "")},
        {"key": "created_at", "value": data.get("created_at", "")},
        {"key": "analysis_type", "value": data.get("analysis_type", "")},
        {"key": "seismic_method", "value": data.get("seismic_method", "")},
        {"key": "num_stories", "value": data.get("num_stories", 0)},
        {"key": "num_nodes", "value": data.get("num_nodes", 0)},
        {"key": "num_elements", "value": data.get("num_elements", 0)},
        {"key": "n_cases", "value": len(data.get("case_names", []))},
        {"key": "n_combos", "value": len(data.get("combo_names", []))},
    ]
    df_meta = pd.DataFrame(meta_rows)

    # ─── Excel 쓰기 ──────────────────────────────────────────
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        # Envelope/Metadata는 단순 long-form → pandas로
        df_env.to_excel(writer, sheet_name='Envelope', index=False)
        df_meta.to_excel(writer, sheet_name='Metadata', index=False)

        # Wide sheets: openpyxl 직접 쓰기 (MultiIndex+index=False 미지원 회피)
        _write_wide_sheet(writer, 'Displacements', disp_case_order, all_disp_nodes,
                          disp_info_cols, disp_info_values, disp_metrics, _disp_get)
        _write_wide_sheet(writer, 'Member_Forces', force_case_order, all_members,
                          force_info_cols, force_info_values, force_metrics, _force_get)
        _write_wide_sheet(writer, 'Reactions', react_case_order, all_react_nodes,
                          react_info_cols, react_info_values, react_metrics, _react_get)

        # 헤더 그룹별 색상 + freeze pane
        _style_wide_header(writer.sheets['Displacements'],
                           disp_case_order, disp_info_cols, len(disp_metrics))
        _style_wide_header(writer.sheets['Member_Forces'],
                           force_case_order, force_info_cols, len(force_metrics))
        _style_wide_header(writer.sheets['Reactions'],
                           react_case_order, react_info_cols, len(react_metrics))

        # 시트 순서: Displacements, Member_Forces, Reactions, Envelope, Metadata
        wb = writer.book
        desired = ['Displacements', 'Member_Forces', 'Reactions', 'Envelope', 'Metadata']
        wb._sheets = [wb[n] for n in desired if n in wb.sheetnames]

        # 열 너비 자동 조정 (MergedCell 회피: 인덱스 기반 순회)
        from openpyxl.utils import get_column_letter
        from openpyxl.cell.cell import MergedCell
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            max_col = ws.max_column
            for col_idx in range(1, max_col + 1):
                col_letter = get_column_letter(col_idx)
                max_len = 0
                for row_idx in range(1, ws.max_row + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if isinstance(cell, MergedCell):
                        continue
                    val = str(cell.value) if cell.value is not None else ""
                    if len(val) > max_len:
                        max_len = len(val)
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 32)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename=analysis_{job_id}.xlsx'}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DXF Export (층별 평면도)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/export/dxf")
async def export_dxf(request: Request):
    """V2 모델을 층별 평면 DXF로 내보내기."""
    import ezdxf
    from io import BytesIO

    body = await request.json()
    model = body.get("model")
    if not model:
        return JSONResponse({"error": "모델 데이터 없음"}, status_code=400)

    nodes = model.get("nodes", [])
    elements = model.get("elements", [])
    elevations = model.get("story_elevations", [])

    if not nodes or not elements:
        return JSONResponse({"error": "노드/요소 없음"}, status_code=400)

    node_map = {n["id"]: n for n in nodes}
    n_stories = len(elevations) - 1 if len(elevations) > 1 else 1

    # ── 모델 크기 산정 (배치 offset용) ──
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    model_w = max(xs) - min(xs) if xs else 20
    model_h = max(ys) - min(ys) if ys else 20
    x_min = min(xs) if xs else 0
    x_max = max(xs) if xs else 20
    y_min = min(ys) if ys else 0
    y_max = max(ys) if ys else 20

    # 테두리 박스 margin
    # 좌측: 그리드연장(5%) + 축번호(3%) + 개별치수선(5%) + 전체치수선(5%) + 여유(4%) = 22%
    # 하단: 동일 = 22%
    # 상단: 제목박스(8%) + 여유(4%) = 12%
    # 우측: 여유(5%)
    md = max(model_w, model_h)
    margin_left = md * 0.28
    margin_right = md * 0.08
    margin_bottom = md * 0.26
    margin_top = md * 0.18
    plan_h = model_h + margin_bottom + margin_top
    plan_w = model_w + margin_left + margin_right
    spacing = plan_h + md * 0.10  # 평면도 간 Y간격

    # ── DXF 문서 생성 ──
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()

    # 색상 (DXF ACI color index)
    COL_GRID = 8       # 회색 (그리드선)
    COL_COLUMN = 1     # 빨강
    COL_BEAM = 3       # 초록
    COL_TEXT = 7       # 흰/검
    COL_DIM = 2        # 노랑
    COL_SUPPORT = 6    # 마젠타
    COL_FRAME = 5      # 파랑 (테두리)

    col_sym_size = 0.3  # m

    # 치수 스타일 설정 (모델 크기 비례)
    dim_scale = max(model_w, model_h) * 0.025
    # 소수점 구분자를 "."로 강제 (기본은 locale에 따라 "," 사용)
    doc.header["$DIMDSEP"] = ord(".")
    if "STRUCT_DIM" not in doc.dimstyles:
        dimstyle = doc.dimstyles.new("STRUCT_DIM")
        dimstyle.dxf.dimtxt = dim_scale       # 치수 텍스트 높이
        dimstyle.dxf.dimasz = dim_scale * 0.8 # 화살표 크기
        dimstyle.dxf.dimexe = dim_scale * 0.4 # 연장선 돌출
        dimstyle.dxf.dimexo = dim_scale * 0.3 # 원점 오프셋
        dimstyle.dxf.dimgap = dim_scale * 0.3
        dimstyle.dxf.dimclrt = COL_DIM
        dimstyle.dxf.dimdec = 2               # 소수 자릿수
        dimstyle.dxf.dimdsep = ord(".")       # 소수점 = "."

    # 층 루프: 0=GL, 1=1F, 2=2F, ...
    for story_idx in range(0, n_stories + 1):
        is_gl = (story_idx == 0)
        z_level = elevations[story_idx] if story_idx < len(elevations) else story_idx * 3.5
        z_tol = 0.01
        label = "GL" if is_gl else f"{story_idx}F"

        layer_frame = f"{label}_FRAME"
        layer_grid = f"{label}_GRID"
        layer_col = f"{label}_COL"
        layer_beam = f"{label}_BEAM"
        layer_text = f"{label}_TEXT"
        layer_dim = f"{label}_DIM"

        for lname, lcolor in [
            (layer_frame, COL_FRAME), (layer_grid, COL_GRID),
            (layer_col, COL_COLUMN), (layer_beam, COL_BEAM),
            (layer_text, COL_TEXT), (layer_dim, COL_DIM),
        ]:
            if lname not in doc.layers:
                doc.layers.add(lname, color=lcolor)

        # 배치 offset: 아래에서 위로 GL → 1F → 2F → ...
        ox = 0
        oy = story_idx * spacing

        # 해당 층 노드
        story_nodes = [n for n in nodes if abs(n["z"] - z_level) < z_tol]
        story_node_ids = {n["id"] for n in story_nodes}

        # 해당 층 요소 (GL은 요소 없음)
        story_elements = [
            e for e in elements
            if e.get("node_i") in story_node_ids and e.get("node_j") in story_node_ids
        ] if not is_gl else []

        # 기둥: 하부 노드가 아래층, 상부 노드가 이 층 (GL은 없음)
        column_elements = []
        if not is_gl:
            z_below = elevations[story_idx - 1] if story_idx - 1 >= 0 else 0
            below_ids = {n["id"] for n in nodes if abs(n["z"] - z_below) < z_tol}
            column_elements = [
                e for e in elements
                if e.get("elem_type") == "column"
                and ((e["node_i"] in below_ids and e["node_j"] in story_node_ids)
                     or (e["node_j"] in below_ids and e["node_i"] in story_node_ids))
            ]

        # ── 테두리 박스 ──
        fx1 = ox + x_min - margin_left
        fy1 = oy + y_min - margin_bottom
        fx2 = ox + x_max + margin_right
        fy2 = oy + y_max + margin_top
        msp.add_lwpolyline(
            [(fx1, fy1), (fx2, fy1), (fx2, fy2), (fx1, fy2)],
            close=True,
            dxfattribs={"layer": layer_frame, "lineweight": 50}
        )
        # 내부 박스 (여백)
        inner_m = md * 0.012
        msp.add_lwpolyline(
            [(fx1+inner_m, fy1+inner_m), (fx2-inner_m, fy1+inner_m),
             (fx2-inner_m, fy2-inner_m), (fx1+inner_m, fy2-inner_m)],
            close=True,
            dxfattribs={"layer": layer_frame, "lineweight": 25}
        )
        # 제목 박스 (우상단) — 텍스트가 충분히 들어갈 크기
        title_text = f"{label} PLAN  EL. {z_level:.1f}m"
        title_text_h = md * 0.028  # 텍스트 높이
        title_box_h = title_text_h * 2.2  # 텍스트 높이의 2.2배 (상하 여유)
        # 대략적인 텍스트 너비 계산 (AutoCAD txt 폰트: 글자당 폭 ≈ height × 0.9)
        text_width_est = len(title_text) * title_text_h * 0.9
        title_box_w = max(md * 0.5, text_width_est + title_text_h * 3.0)
        tb_x2 = fx2 - inner_m
        tb_y2 = fy2 - inner_m
        tb_x1 = tb_x2 - title_box_w
        tb_y1 = tb_y2 - title_box_h
        msp.add_lwpolyline(
            [(tb_x1, tb_y1), (tb_x2, tb_y1), (tb_x2, tb_y2), (tb_x1, tb_y2)],
            close=True,
            dxfattribs={"layer": layer_frame}
        )
        t = msp.add_text(
            title_text,
            dxfattribs={
                "layer": layer_text, "height": title_text_h,
                "color": COL_TEXT,
            }
        )
        t.set_placement(
            ((tb_x1 + tb_x2) / 2, (tb_y1 + tb_y2) / 2),
            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
        )

        # ── 그리드선 + 축번호 ──
        story_xs = sorted(set(round(n["x"], 3) for n in (story_nodes if story_nodes else nodes)))
        story_ys = sorted(set(round(n["y"], 3) for n in (story_nodes if story_nodes else nodes)))
        grid_ext = max(model_w, model_h) * 0.05

        for i, gx in enumerate(story_xs):
            msp.add_line(
                (ox + gx, oy + y_min - grid_ext),
                (ox + gx, oy + y_max + grid_ext),
                dxfattribs={"layer": layer_grid, "linetype": "CENTER"}
            )
            t = msp.add_text(
                f"X{i+1}",
                dxfattribs={"layer": layer_text, "height": dim_scale * 1.2, "color": COL_DIM}
            )
            t.set_placement(
                (ox + gx, oy + y_min - grid_ext - dim_scale * 1.5),
                align=ezdxf.enums.TextEntityAlignment.CENTER,
            )

        for j, gy in enumerate(story_ys):
            msp.add_line(
                (ox + x_min - grid_ext, oy + gy),
                (ox + x_max + grid_ext, oy + gy),
                dxfattribs={"layer": layer_grid, "linetype": "CENTER"}
            )
            t = msp.add_text(
                f"Y{j+1}",
                dxfattribs={"layer": layer_text, "height": dim_scale * 1.2, "color": COL_DIM}
            )
            t.set_placement(
                (ox + x_min - grid_ext - dim_scale * 0.8, oy + gy),
                align=ezdxf.enums.TextEntityAlignment.MIDDLE_RIGHT,
            )

        # ── 보 ──
        for e in story_elements:
            if e.get("elem_type") == "column":
                continue
            ni = node_map.get(e["node_i"])
            nj = node_map.get(e["node_j"])
            if not ni or not nj:
                continue
            msp.add_line(
                (ox + ni["x"], oy + ni["y"]),
                (ox + nj["x"], oy + nj["y"]),
                dxfattribs={"layer": layer_beam}
            )

        # ── 기둥 심볼 ──
        col_positions = set()
        for e in column_elements:
            ni = node_map.get(e["node_i"])
            nj = node_map.get(e["node_j"])
            top_node = nj if (nj and nj["id"] in story_node_ids) else ni
            if not top_node:
                continue
            pos_key = (round(top_node["x"], 3), round(top_node["y"], 3))
            if pos_key in col_positions:
                continue
            col_positions.add(pos_key)
            cx, cy = ox + top_node["x"], oy + top_node["y"]
            hs = col_sym_size / 2
            msp.add_lwpolyline(
                [(cx-hs, cy-hs), (cx+hs, cy-hs), (cx+hs, cy+hs), (cx-hs, cy+hs)],
                close=True,
                dxfattribs={"layer": layer_col}
            )

        # ── GL: 지점 심볼 (삼각형) ──
        if is_gl:
            for n in story_nodes:
                if not n.get("support"):
                    continue
                sx, sy = ox + n["x"], oy + n["y"]
                ts = 0.25
                msp.add_lwpolyline(
                    [(sx, sy), (sx - ts, sy - ts*1.5), (sx + ts, sy - ts*1.5)],
                    close=True,
                    dxfattribs={"layer": layer_col, "color": COL_SUPPORT}
                )

        # ── 치수선 (bay 간격) ──
        # X방향 치수 (하단): 연속 치수
        if len(story_xs) >= 2:
            dim_y = oy + y_min - grid_ext - dim_scale * 3.5
            for i in range(len(story_xs) - 1):
                x1 = ox + story_xs[i]
                x2 = ox + story_xs[i + 1]
                dim = msp.add_linear_dim(
                    base=(0, dim_y),
                    p1=(x1, oy + y_min),
                    p2=(x2, oy + y_min),
                    angle=0,
                    dimstyle="STRUCT_DIM",
                    dxfattribs={"layer": layer_dim},
                )
                dim.render()
            # 전체 치수 (overall)
            dim_y_all = dim_y - dim_scale * 2.5
            dim = msp.add_linear_dim(
                base=(0, dim_y_all),
                p1=(ox + story_xs[0], oy + y_min),
                p2=(ox + story_xs[-1], oy + y_min),
                angle=0,
                dimstyle="STRUCT_DIM",
                dxfattribs={"layer": layer_dim},
            )
            dim.render()

        # Y방향 치수 (좌측)
        if len(story_ys) >= 2:
            dim_x = ox + x_min - grid_ext - dim_scale * 3.5
            for j in range(len(story_ys) - 1):
                y1 = oy + story_ys[j]
                y2 = oy + story_ys[j + 1]
                dim = msp.add_linear_dim(
                    base=(dim_x, 0),
                    p1=(ox + x_min, y1),
                    p2=(ox + x_min, y2),
                    angle=90,
                    dimstyle="STRUCT_DIM",
                    dxfattribs={"layer": layer_dim},
                )
                dim.render()
            # 전체 치수
            dim_x_all = dim_x - dim_scale * 2.5
            dim = msp.add_linear_dim(
                base=(dim_x_all, 0),
                p1=(ox + x_min, oy + story_ys[0]),
                p2=(ox + x_min, oy + story_ys[-1]),
                angle=90,
                dimstyle="STRUCT_DIM",
                dxfattribs={"layer": layer_dim},
            )
            dim.render()

    # ═══════════════════════════════════════════════════════════
    # 입면도 (Elevations)
    # ═══════════════════════════════════════════════════════════
    total_z = elevations[-1] if elevations else 0
    model_h_z = total_z  # 입면도의 세로 크기 = 전체 높이
    # 평면도 우측으로 offset (평면 폭 + margin)
    elev_col1_ox = plan_w + md * 0.20  # X방향 입면도 컬럼
    elev_col2_ox = elev_col1_ox + (model_w + margin_left + margin_right) + md * 0.20  # Y방향 입면도 컬럼

    # 입면도 그리기 공통 헬퍼
    def _draw_elevation(ox, oy, label_text, horiz_coords, hmin, hmax,
                        draw_columns, draw_beams, layer_prefix, axis_prefix):
        """입면도 1개 그리기 (공통 로직).
        horiz_coords: 수평축 그리드 좌표 (축번호용)
        hmin, hmax: 수평축 범위
        draw_columns: [(x_pos, z_start, z_end)] 기둥 리스트
        draw_beams: [(x_start, x_end, z)] 보 리스트
        """
        # 레이어
        lf = f"{layer_prefix}_FRAME"
        lg = f"{layer_prefix}_GRID"
        lc = f"{layer_prefix}_COL"
        lb = f"{layer_prefix}_BEAM"
        lt = f"{layer_prefix}_TEXT"
        ld = f"{layer_prefix}_DIM"
        for name, cc in [(lf, COL_FRAME), (lg, COL_GRID), (lc, COL_COLUMN),
                         (lb, COL_BEAM), (lt, COL_TEXT), (ld, COL_DIM)]:
            if name not in doc.layers:
                doc.layers.add(name, color=cc)

        # 테두리
        fx1 = ox + hmin - margin_left
        fy1 = oy - margin_bottom
        fx2 = ox + hmax + margin_right
        fy2 = oy + total_z + margin_top
        msp.add_lwpolyline(
            [(fx1, fy1), (fx2, fy1), (fx2, fy2), (fx1, fy2)],
            close=True,
            dxfattribs={"layer": lf, "lineweight": 50}
        )
        # 내부 박스
        inner_mm = md * 0.012
        msp.add_lwpolyline(
            [(fx1+inner_mm, fy1+inner_mm), (fx2-inner_mm, fy1+inner_mm),
             (fx2-inner_mm, fy2-inner_mm), (fx1+inner_mm, fy2-inner_mm)],
            close=True,
            dxfattribs={"layer": lf, "lineweight": 25}
        )
        # 제목 박스
        t_title_h = md * 0.028
        tb_h = t_title_h * 2.2
        tw_est = len(label_text) * t_title_h * 0.9
        tb_w = max(md * 0.5, tw_est + t_title_h * 3.0)
        tbx2 = fx2 - inner_mm
        tby2 = fy2 - inner_mm
        tbx1 = tbx2 - tb_w
        tby1 = tby2 - tb_h
        msp.add_lwpolyline(
            [(tbx1, tby1), (tbx2, tby1), (tbx2, tby2), (tbx1, tby2)],
            close=True,
            dxfattribs={"layer": lf}
        )
        tt = msp.add_text(
            label_text,
            dxfattribs={"layer": lt, "height": t_title_h, "color": COL_TEXT}
        )
        tt.set_placement(
            ((tbx1 + tbx2) / 2, (tby1 + tby2) / 2),
            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
        )

        gext = md * 0.05

        # 수직 그리드 (수평 위치: horiz_coords) + 축번호
        for i, hx in enumerate(horiz_coords):
            msp.add_line(
                (ox + hx, oy - gext),
                (ox + hx, oy + total_z + gext),
                dxfattribs={"layer": lg, "linetype": "CENTER"}
            )
            t = msp.add_text(
                f"{axis_prefix}{i+1}",  # X1/X2... or Y1/Y2...
                dxfattribs={"layer": lt, "height": dim_scale * 1.2, "color": COL_DIM}
            )
            t.set_placement(
                (ox + hx, oy - gext - dim_scale * 1.5),
                align=ezdxf.enums.TextEntityAlignment.CENTER,
            )

        # 수평 그리드 (각 층 z) + 층 라벨 (GL/1F/2F...)
        for s_idx, ez in enumerate(elevations):
            msp.add_line(
                (ox + hmin - gext, oy + ez),
                (ox + hmax + gext, oy + ez),
                dxfattribs={"layer": lg, "linetype": "CENTER"}
            )
            story_label = "GL" if s_idx == 0 else f"{s_idx}F"
            t = msp.add_text(
                story_label,
                dxfattribs={"layer": lt, "height": dim_scale * 1.2, "color": COL_DIM}
            )
            t.set_placement(
                (ox + hmin - gext - dim_scale * 0.8, oy + ez),
                align=ezdxf.enums.TextEntityAlignment.MIDDLE_RIGHT,
            )

        # 기둥 (수직 LINE)
        for hx, z_start, z_end in draw_columns:
            msp.add_line(
                (ox + hx, oy + z_start),
                (ox + hx, oy + z_end),
                dxfattribs={"layer": lc}
            )

        # 보 (수평 LINE)
        for hx_start, hx_end, z in draw_beams:
            msp.add_line(
                (ox + hx_start, oy + z),
                (ox + hx_end, oy + z),
                dxfattribs={"layer": lb}
            )

        # 수평 치수선 (bay 간격)
        if len(horiz_coords) >= 2:
            d_y = oy - gext - dim_scale * 3.5
            for i in range(len(horiz_coords) - 1):
                dim = msp.add_linear_dim(
                    base=(0, d_y),
                    p1=(ox + horiz_coords[i], oy),
                    p2=(ox + horiz_coords[i + 1], oy),
                    angle=0,
                    dimstyle="STRUCT_DIM",
                    dxfattribs={"layer": ld},
                )
                dim.render()
            d_y_all = d_y - dim_scale * 2.5
            dim = msp.add_linear_dim(
                base=(0, d_y_all),
                p1=(ox + horiz_coords[0], oy),
                p2=(ox + horiz_coords[-1], oy),
                angle=0,
                dimstyle="STRUCT_DIM",
                dxfattribs={"layer": ld},
            )
            dim.render()

        # 수직 치수선 (층고)
        if len(elevations) >= 2:
            d_x = ox + hmin - gext - dim_scale * 3.5
            for s_idx in range(len(elevations) - 1):
                dim = msp.add_linear_dim(
                    base=(d_x, 0),
                    p1=(ox + hmin, oy + elevations[s_idx]),
                    p2=(ox + hmin, oy + elevations[s_idx + 1]),
                    angle=90,
                    dimstyle="STRUCT_DIM",
                    dxfattribs={"layer": ld},
                )
                dim.render()
            d_x_all = d_x - dim_scale * 2.5
            dim = msp.add_linear_dim(
                base=(d_x_all, 0),
                p1=(ox + hmin, oy + elevations[0]),
                p2=(ox + hmin, oy + elevations[-1]),
                angle=90,
                dimstyle="STRUCT_DIM",
                dxfattribs={"layer": ld},
            )
            dim.render()

    # X방향 입면도 (Y_j 라인별) — 좌→우: X1, X2... / 상→하: Y_j
    all_xs = sorted(set(round(n["x"], 3) for n in nodes))
    all_ys = sorted(set(round(n["y"], 3) for n in nodes))
    y_tol = 0.01

    elev_x_h = total_z + margin_bottom + margin_top
    elev_x_spacing = elev_x_h + md * 0.10

    for j, y_line in enumerate(all_ys):
        # 해당 Y 라인 상의 노드
        at_line_nodes = [n for n in nodes if abs(n["y"] - y_line) < y_tol]
        at_line_ids = {n["id"] for n in at_line_nodes}
        if len(at_line_nodes) < 2:
            continue

        # 기둥: node_i, node_j 모두 이 라인, elem_type=column
        draw_columns = []
        for e in elements:
            if e.get("elem_type") != "column":
                continue
            if e["node_i"] not in at_line_ids or e["node_j"] not in at_line_ids:
                continue
            ni = node_map[e["node_i"]]
            nj = node_map[e["node_j"]]
            if abs(ni["x"] - nj["x"]) > y_tol:
                continue  # 경사 기둥 배제
            draw_columns.append((ni["x"], min(ni["z"], nj["z"]), max(ni["z"], nj["z"])))

        # 보: node_i, node_j 모두 이 라인, elem_type=beam_x or beam
        draw_beams = []
        for e in elements:
            et = e.get("elem_type", "")
            if et == "column":
                continue
            if e["node_i"] not in at_line_ids or e["node_j"] not in at_line_ids:
                continue
            ni = node_map[e["node_i"]]
            nj = node_map[e["node_j"]]
            if abs(ni["z"] - nj["z"]) > y_tol:
                continue  # 수평 보만
            draw_beams.append((min(ni["x"], nj["x"]), max(ni["x"], nj["x"]), ni["z"]))

        # 배치
        ox = elev_col1_ox
        oy = j * elev_x_spacing
        label_text = f"Y{j+1} ELEVATION"
        layer_prefix = f"ELEV_Y{j+1}"
        _draw_elevation(ox, oy, label_text, all_xs, x_min, x_max,
                        draw_columns, draw_beams, layer_prefix, "X")

    # Y방향 입면도 (X_i 라인별)
    elev_y_w = model_h  # Y방향 입면도의 가로 = Y치수
    for i, x_line in enumerate(all_xs):
        at_line_nodes = [n for n in nodes if abs(n["x"] - x_line) < y_tol]
        at_line_ids = {n["id"] for n in at_line_nodes}
        if len(at_line_nodes) < 2:
            continue

        draw_columns = []
        for e in elements:
            if e.get("elem_type") != "column":
                continue
            if e["node_i"] not in at_line_ids or e["node_j"] not in at_line_ids:
                continue
            ni = node_map[e["node_i"]]
            nj = node_map[e["node_j"]]
            if abs(ni["y"] - nj["y"]) > y_tol:
                continue
            draw_columns.append((ni["y"], min(ni["z"], nj["z"]), max(ni["z"], nj["z"])))

        draw_beams = []
        for e in elements:
            et = e.get("elem_type", "")
            if et == "column":
                continue
            if e["node_i"] not in at_line_ids or e["node_j"] not in at_line_ids:
                continue
            ni = node_map[e["node_i"]]
            nj = node_map[e["node_j"]]
            if abs(ni["z"] - nj["z"]) > y_tol:
                continue
            draw_beams.append((min(ni["y"], nj["y"]), max(ni["y"], nj["y"]), ni["z"]))

        ox = elev_col2_ox
        oy = i * elev_x_spacing
        label_text = f"X{i+1} ELEVATION"
        layer_prefix = f"ELEV_X{i+1}"
        _draw_elevation(ox, oy, label_text, all_ys, y_min, y_max,
                        draw_columns, draw_beams, layer_prefix, "Y")

    # ── Extents 및 뷰 자동 설정 (AutoCAD 열 때 도면이 보이도록) ──
    try:
        from ezdxf import zoom
        zoom.extents(msp, factor=1.1)
    except Exception as ex:
        print(f"Warning: zoom.extents failed: {ex}")

    # ── 파일 출력 (ezdxf는 파일 경로 필요) ──
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix='.dxf', delete=False)
    tmp.close()
    doc.saveas(tmp.name)
    buf = BytesIO(open(tmp.name, 'rb').read())
    os.unlink(tmp.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/dxf",
        headers={"Content-Disposition": "attachment; filename=structural_plan.dxf"}
    )


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
