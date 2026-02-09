"""
Celery tasks for Frame2D analysis
"""
import sys
import json
import traceback
from pathlib import Path

from app.tasks.celery_app import celery_app
from app.core.config import MCP_SERVER_PATH, JOBS_DIR
from app.core.job_manager import update_job_status, save_job_summary, get_job_input
from app.models.schemas import JobStatus


@celery_app.task(bind=True, name="run_frame2d_analysis")
def run_frame2d_analysis(self, job_id: str):
    """
    Execute Frame2D analysis using existing MCP server code
    """
    job_dir = JOBS_DIR / job_id
    log_path = job_dir / "analysis.log"

    def log(msg: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

    try:
        # Update status to running
        update_job_status(job_id, JobStatus.RUNNING, progress=5)
        log(f"Starting analysis for job {job_id}")

        # Add MCP server to path
        if str(MCP_SERVER_PATH) not in sys.path:
            sys.path.insert(0, str(MCP_SERVER_PATH))

        # Import analysis functions
        from core.frame_2d import analyze_frame_2d_multi
        from core.visualization import plot_frame_2d_multi_interactive

        update_job_status(job_id, JobStatus.RUNNING, progress=10)
        log("Imported analysis modules")

        # Load input
        input_data = get_job_input(job_id)
        if not input_data:
            raise ValueError("Input data not found")

        log(f"Input: {json.dumps(input_data, indent=2)}")

        # Convert load_cases to expected format
        load_cases = {}
        for case_name, loads in input_data.get("load_cases", {}).items():
            load_cases[case_name] = [
                {k: v for k, v in ld.items() if v is not None}
                for ld in loads
            ]

        update_job_status(job_id, JobStatus.RUNNING, progress=20)
        log("Prepared load cases")

        # Run analysis
        log("Running Frame2D analysis...")
        result = analyze_frame_2d_multi(
            stories=input_data["stories"],
            bays=input_data["bays"],
            load_cases=load_cases,
            supports=input_data.get("supports", "fixed"),
            column_section=input_data.get("column_section", "H-300x300"),
            beam_section=input_data.get("beam_section", "H-400x200"),
            material_name=input_data.get("material_name", "SS275"),
            num_elements_per_member=input_data.get("num_elements_per_member", 4),
            load_combinations=input_data.get("load_combinations"),
        )

        update_job_status(job_id, JobStatus.RUNNING, progress=60)
        log("Analysis completed, generating report...")

        # Generate HTML report
        report_path = str(job_dir / "report.html")
        plot_frame_2d_multi_interactive(result, output_path=report_path)

        update_job_status(job_id, JobStatus.RUNNING, progress=90)
        log(f"Report generated: {report_path}")

        # Extract summary from first case or first combo
        all_cases = list(result.case_results.keys()) + list(result.combo_results.keys())
        if all_cases:
            first_case = all_cases[0]
            if first_case in result.case_results:
                cr = result.case_results[first_case]
            else:
                cr = result.combo_results[first_case]

            # Calculate base shear from reactions
            base_shear = sum(abs(r.get("RX_kN", 0)) for r in cr.reactions)

            summary = {
                "max_displacement_x_mm": cr.max_displacement_x,
                "max_displacement_y_mm": cr.max_displacement_y,
                "max_drift": cr.max_drift,
                "max_drift_story": cr.max_drift_story,
                "max_moment_kNm": cr.max_moment,
                "max_shear_kN": cr.max_shear,
                "max_axial_kN": cr.max_axial,
                "base_shear_kN": base_shear,
                "num_stories": result.num_stories,
                "num_bays": result.num_bays,
            }
            save_job_summary(job_id, summary)
            log(f"Summary: {json.dumps(summary, indent=2)}")

        # Mark as done
        update_job_status(job_id, JobStatus.DONE, progress=100)
        log("Job completed successfully")

        return {"status": "success", "job_id": job_id}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        log(f"ERROR: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        return {"status": "failed", "job_id": job_id, "error": str(e)}
