"""Minimal 3D frame OpenSees model for the Apply MVP pipeline.

Creates a 2-storey, 1-bay × 1-bay 3D frame (4 columns, 4 beams per storey),
applies factored line loads on beams, runs a linear-static analysis, and
extracts basic results (max displacement, column axial forces).

This module is deliberately simple — the goal is to prove that the
load-combination pipeline feeds into OpenSees without errors, not to
produce publication-quality results.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _setup_opensees_dll_path() -> None:
    """Add DLL directories required by the ``opensees`` package on Windows.

    The ``opensees`` (Claudio Perez) package ships ``OpenSeesRT.dll`` which
    depends on MKL/Intel DLLs typically installed under ``Library/bin``
    in conda-style environments.  Tcl's ``load`` command uses the PATH
    environment variable, so we prepend the necessary directories.
    """
    if sys.platform != "win32":
        return
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    site_pkgs = os.path.join(sys.prefix, "Lib", "site-packages")
    opensees_bin = os.path.join(site_pkgs, "opensees", "bin")
    dirs_to_add = [d for d in (opensees_bin, lib_bin) if os.path.isdir(d)]
    if dirs_to_add:
        current = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(dirs_to_add) + os.pathsep + current


# Guard import — try multiple backends
_setup_opensees_dll_path()

ops = None  # type: ignore[assignment]
OPENSEESPY_AVAILABLE = False
_OPS_BACKEND = ""

# 1) Try the new 'opensees' package (supports Python 3.12+)
try:
    import opensees.openseespy as _ops
    ops = _ops
    OPENSEESPY_AVAILABLE = True
    _OPS_BACKEND = "opensees"
except (ImportError, RuntimeError, OSError):
    pass

# 2) Fallback: legacy 'openseespy' package (Python <=3.12)
if not OPENSEESPY_AVAILABLE:
    try:
        import openseespy.opensees as _ops  # type: ignore[no-redef]
        ops = _ops
        OPENSEESPY_AVAILABLE = True
        _OPS_BACKEND = "openseespy"
    except (ImportError, RuntimeError):
        pass

if OPENSEESPY_AVAILABLE:
    logger.info("OpenSees backend: %s", _OPS_BACKEND)
else:
    logger.info("OpenSees not available (neither 'opensees' nor 'openseespy' found)")


# ======================================================================
# Result container
# ======================================================================

@dataclass
class Frame3DMvpResult:
    """Lightweight result container for the MVP 3D analysis."""
    success: bool = False
    num_nodes: int = 0
    num_elements: int = 0
    max_disp_z_mm: float = 0.0           # max vertical displacement (mm)
    max_disp_x_mm: float = 0.0           # max lateral X (mm)
    max_disp_y_mm: float = 0.0           # max lateral Y (mm)
    max_disp_z_node: int = 0
    column_axial_kN: dict[int, float] = field(default_factory=dict)  # elem -> axial
    beam_shear_kN: dict[int, float] = field(default_factory=dict)    # elem -> max shear
    applied_load_total_kN: float = 0.0
    reaction_total_kN: float = 0.0
    equilibrium_error_pct: float = 0.0
    errors: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


# ======================================================================
# Geometry helpers
# ======================================================================

def _generate_3d_frame_geometry(
    stories: list[float],
    bay_x: float,
    bay_y: float,
) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, int, str]]]:
    """Generate nodes and element connections for a 1×1-bay 3D frame.

    Columns are vertical (Z-up).  Beams run along X and Y at each storey.

    Node numbering:
        storey i (0=base): nodes 4*i+1 … 4*i+4
        corners per storey:
          +1 → (0,    0,    z)
          +2 → (bay_x,0,    z)
          +3 → (bay_x,bay_y,z)
          +4 → (0,    bay_y,z)

    Returns
    -------
    nodes : {node_id: (x, y, z)}
    connections : [(ni, nj, elem_type)]   elem_type ∈ {"column", "beam_x", "beam_y"}
    """
    nodes: dict[int, tuple[float, float, float]] = {}
    connections: list[tuple[int, int, str]] = []

    n_stories = len(stories)
    z = 0.0
    for s in range(n_stories + 1):  # 0=base .. n_stories
        base_id = s * 4 + 1
        nodes[base_id]     = (0.0,   0.0,   z)
        nodes[base_id + 1] = (bay_x, 0.0,   z)
        nodes[base_id + 2] = (bay_x, bay_y, z)
        nodes[base_id + 3] = (0.0,   bay_y, z)
        if s < n_stories:
            z += stories[s]

    # Columns: connect storey s corner j to storey s+1 corner j
    for s in range(n_stories):
        for j in range(4):
            ni = s * 4 + 1 + j
            nj = (s + 1) * 4 + 1 + j
            connections.append((ni, nj, "column"))

    # Beams at each storey level (s >= 1)
    for s in range(1, n_stories + 1):
        b = s * 4 + 1
        # X-direction beams
        connections.append((b, b + 1, "beam_x"))      # bottom edge
        connections.append((b + 3, b + 2, "beam_x"))   # top edge
        # Y-direction beams
        connections.append((b, b + 3, "beam_y"))        # left edge
        connections.append((b + 1, b + 2, "beam_y"))    # right edge

    return nodes, connections


# ======================================================================
# Model builder + analyser
# ======================================================================

def build_and_analyze(
    stories: list[float],
    bay_x: float,
    bay_y: float,
    beam_loads: dict[int, float],
    *,
    E_MPa: float = 200_000.0,
    col_A_mm2: float = 14_200.0,     # H-300×300
    col_I_mm4: float = 204_000_000.0,
    beam_A_mm2: float = 8_410.0,     # H-400×200
    beam_Iz_mm4: float = 237_000_000.0,
    beam_Iy_mm4: float = 17_400_000.0,
) -> Frame3DMvpResult:
    """Build a minimal 3D frame, apply loads, and run linear-static analysis.

    Parameters
    ----------
    stories :
        Heights per storey (m), e.g. [3.5, 3.5].
    bay_x, bay_y :
        Bay widths (m) in X and Y directions.
    beam_loads :
        {storey_level: w_kN_per_m} — uniform line load on ALL beams at that level.
        ``storey_level`` is 1-based.

    Returns
    -------
    Frame3DMvpResult
    """
    res = Frame3DMvpResult()

    if not OPENSEESPY_AVAILABLE:
        res.errors.append("openseespy not installed — skipping analysis")
        return res

    nodes, connections = _generate_3d_frame_geometry(stories, bay_x, bay_y)
    n_stories = len(stories)
    base_nodes = list(range(1, 5))  # nodes 1..4

    try:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        # --- Nodes ---
        for nid, (x, y, z) in nodes.items():
            ops.node(nid, x, y, z)
        res.num_nodes = len(nodes)
        res.log.append(f"Created {len(nodes)} nodes")

        # --- Boundary conditions (fixed base) ---
        for bn in base_nodes:
            ops.fix(bn, 1, 1, 1, 1, 1, 1)

        # --- Material and sections ---
        E = E_MPa * 1e3  # kN/m2  (1 MPa = 1e3 kN/m2 ... wait no)
        # Actually: 1 MPa = 1 N/mm2.  OpenSees in kN, m:
        # E(kN/m2) = E(MPa) * 1e3  ... no.
        # 1 MPa = 1e6 Pa = 1e6 N/m2 = 1e3 kN/m2.  So E(kN/m2) = E_MPa * 1e3.
        E_kN_m2 = E_MPa * 1e3   # kN/m2
        G_kN_m2 = E_kN_m2 / (2.0 * (1.0 + 0.3))  # ν=0.3

        # Convert mm2 → m2, mm4 → m4
        col_A = col_A_mm2 * 1e-6
        col_I = col_I_mm4 * 1e-12
        beam_A = beam_A_mm2 * 1e-6
        beam_Iz = beam_Iz_mm4 * 1e-12
        beam_Iy = beam_Iy_mm4 * 1e-12
        # Torsional constant J — simplified as sum of Ix + Iy
        col_J = 2.0 * col_I
        beam_J = beam_Iz + beam_Iy

        # Geometric transformations
        # Columns: local z along member axis (vertical), vecxz = (0, 0, 1)
        # We use a Linear transform for columns with vecxz perpendicular to the member
        ops.geomTransf("Linear", 1, 1, 0, 0)  # columns (vertical) — vecxz = global X
        ops.geomTransf("Linear", 2, 0, 0, 1)  # beams in X-dir — vecxz = global Z
        ops.geomTransf("Linear", 3, 0, 0, 1)  # beams in Y-dir — vecxz = global Z

        # --- Elements (elasticBeamColumn) ---
        elem_id = 1
        elem_map: dict[int, tuple[int, int, str]] = {}
        for ni, nj, etype in connections:
            if etype == "column":
                transf = 1
                A, Iz, Iy, J = col_A, col_I, col_I, col_J
            elif etype == "beam_x":
                transf = 2
                A, Iz, Iy, J = beam_A, beam_Iz, beam_Iy, beam_J
            else:  # beam_y
                transf = 3
                A, Iz, Iy, J = beam_A, beam_Iz, beam_Iy, beam_J

            ops.element(
                "elasticBeamColumn", elem_id,
                ni, nj,
                A, E_kN_m2, G_kN_m2,
                J, Iy, Iz,
                transf,
            )
            elem_map[elem_id] = (ni, nj, etype)
            elem_id += 1

        res.num_elements = elem_id - 1
        res.log.append(f"Created {res.num_elements} elements")

        # --- Loads ---
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)

        total_applied = 0.0
        for storey_level, w in beam_loads.items():
            if w == 0.0:
                continue
            # Find beams at this storey level
            for eid, (ni, nj, etype) in elem_map.items():
                if etype.startswith("beam"):
                    # Determine which storey this beam belongs to
                    storey = (ni - 1) // 4  # storey of the i-node (1-based in connection)
                    if storey == storey_level:
                        # Compute beam length
                        xi, yi, zi = nodes[ni]
                        xj, yj, zj = nodes[nj]
                        L = math.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
                        # Apply uniform load in -Z direction (gravity)
                        # eleLoad: Wz in local coord = -w (downward)
                        ops.eleLoad("-ele", eid, "-type", "-beamUniform", 0.0, -w, 0.0)
                        total_applied += w * L
                        res.log.append(
                            f"  elem {eid} ({etype}): w={w:.3f} kN/m, L={L:.3f} m, "
                            f"total={w*L:.3f} kN"
                        )

        res.applied_load_total_kN = round(total_applied, 4)
        res.log.append(f"Total applied load: {total_applied:.4f} kN")

        # --- Analysis ---
        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        ok = ops.analyze(1)

        if ok != 0:
            res.errors.append(f"Analysis failed with code {ok}")
            return res

        res.success = True
        res.log.append("Analysis completed successfully")

        # --- Extract results ---
        max_dz = 0.0
        max_dz_node = 0
        max_dx = 0.0
        max_dy = 0.0
        for nid in nodes:
            if nid in base_nodes:
                continue
            dx, dy, dz = ops.nodeDisp(nid, 1), ops.nodeDisp(nid, 2), ops.nodeDisp(nid, 3)
            if abs(dz) > abs(max_dz):
                max_dz = dz
                max_dz_node = nid
            if abs(dx) > abs(max_dx):
                max_dx = dx
            if abs(dy) > abs(max_dy):
                max_dy = dy

        res.max_disp_z_mm = round(max_dz * 1000, 4)   # m → mm
        res.max_disp_z_node = max_dz_node
        res.max_disp_x_mm = round(max_dx * 1000, 4)
        res.max_disp_y_mm = round(max_dy * 1000, 4)

        # Column axial forces (element local force index 1 = axial at i-node)
        for eid, (ni, nj, etype) in elem_map.items():
            forces = ops.eleForce(eid)  # 12 DOFs for 3D beam-column
            if etype == "column":
                # forces[2] = local axial for vertical member (Fz at i-node)
                # For a vertical column, axial ≈ forces[2] (local z = global Z)
                axial = forces[2]  # positive = tension in OpenSees convention
                res.column_axial_kN[eid] = round(axial, 4)
            elif etype.startswith("beam"):
                # Shear in beams: forces[1] for Vy, forces[2] for Vz at i-node
                shear_z = abs(forces[1])
                shear_y = abs(forces[2])
                res.beam_shear_kN[eid] = round(max(shear_z, shear_y), 4)

        # Reactions — must call reactions() first to compute them
        ops.reactions()
        total_reaction_z = 0.0
        for bn in base_nodes:
            rz = ops.nodeReaction(bn, 3)  # reaction in Z
            total_reaction_z += rz
        res.reaction_total_kN = round(total_reaction_z, 4)

        # Equilibrium check
        if total_applied > 0:
            # reaction_z is in the direction resisting gravity → should be positive
            err = abs(abs(total_reaction_z) - total_applied) / total_applied * 100
            res.equilibrium_error_pct = round(err, 4)

        res.log.append(
            f"Max vertical disp: {res.max_disp_z_mm:.4f} mm (node {max_dz_node})"
        )
        res.log.append(f"Reactions ΣFz: {res.reaction_total_kN:.4f} kN")
        res.log.append(f"Equilibrium error: {res.equilibrium_error_pct:.4f}%")

    except Exception as exc:
        res.errors.append(f"OpenSees error: {exc}")
        logger.exception("OpenSees 3D MVP analysis failed")
    finally:
        if OPENSEESPY_AVAILABLE:
            try:
                ops.wipe()
            except Exception:
                pass

    return res
