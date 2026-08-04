# conftest.py — pytest root marker
#
# pytest collects every test_*.py before any one of them runs, so each
# test file's local `sys.path.insert(...)` happens too late for the
# initial import step. Adding the project root and the mcp-server
# folder to sys.path here makes `from core.xxx import ...` work during
# collection regardless of which test file goes first.
#
# Note: both `core/` (legacy MVP contract_interpreter) and
# `mcp-server/core/` (V2 analysis engine) exist as namespace packages
# (no __init__.py) so PEP 420 merges them at import time.
import os
import sys
from pathlib import Path

# matplotlib picks an interactive backend (TkAgg on Windows) when a display is
# available, and the report-rendering tests then create and tear down Tcl
# interpreters from whatever thread pytest happens to be on. That fails
# intermittently with `_tkinter.TclError` — a *different* test each run, because
# the ordering is randomised, which makes the failure look like a real
# regression somewhere new every time. Nothing under test needs a window.
# Must be set before anything imports matplotlib (visualization.py does).
os.environ.setdefault("MPLBACKEND", "Agg")

_ROOT = Path(__file__).resolve().parent
_MCP_SERVER = _ROOT / "mcp-server"

for _p in (_ROOT, _MCP_SERVER):
    sp = str(_p)
    if _p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

# Script-style "tests" that run analyses at module load time and call
# sys.exit() on failure. They terminate the whole pytest session, so
# exclude them from auto-discovery. They can still be executed
# directly with `python tests/test_xxx.py`.
#
# The MVP-era tests reference modules (`adapters/`, `core.contract_interpreter`)
# that were intentionally removed in commit a8c7f6f ("cleanup legacy code")
# but the test files themselves were left behind. They pass locally only when
# the deleted source still exists in the working copy. Skip them in CI; they
# can be revived if/when the legacy pipeline is reintroduced.
collect_ignore = [
    "tests/test_pdelta_validation.py",
    "tests/test_stage5_metadata.py",
    "tests/test_stage6_building_nonlinear.py",
    "tests/test_apply_mvp_pipeline.py",
    "tests/test_contract_interpreter.py",
    "tests/test_load_combo_adapter.py",
]
