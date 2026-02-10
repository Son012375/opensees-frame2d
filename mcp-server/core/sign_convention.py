"""
Sign Convention Transformation Utilities for Structural Analysis Visualization

This module provides functions to convert internal forces from OpenSees local
coordinate convention to the textbook/MIDAS structural engineering convention.

===================================================================================
BACKGROUND: OpenSees vs. Textbook Sign Convention
===================================================================================

OpenSees elasticBeamColumn element:
- eleForce(tag) returns [N_i, V_i, M_i, N_j, V_j, M_j]
- These are nodal reaction forces (equilibrating forces)
- Sign follows element local coordinate system (x: i→j, y: perpendicular)

Textbook/MIDAS Convention (target for visualization):
- Shear V:
    - Left cut section, looking at left portion
    - V > 0: upward shear on left face (↑)
    - For a simply supported beam with uniform downward load: SFD starts (+) at left, ends (-) at right

- Moment M:
    - M > 0: sagging (tension at bottom, compression at top) - concave upward
    - M < 0: hogging (tension at top, compression at bottom) - concave downward
    - For a simply supported beam with downward load: positive moment at midspan

===================================================================================
TRANSFORMATION RULES
===================================================================================

For horizontal beams (typical case):
    V_textbook = -V_opensees
    M_textbook = -M_opensees

Why the sign flip?
- OpenSees returns forces that the element exerts on nodes (equilibrating reaction)
- Textbook convention uses internal force acting on the cut section
- These are opposite in direction

===================================================================================
USAGE
===================================================================================

This module is designed for visualization ONLY. The analysis results (stored in
Result dataclasses) should retain OpenSees convention for numerical accuracy
and interoperability.

Apply transformations at the plotting stage:
    V_plot, M_plot = transform_to_textbook_convention(V_raw, M_raw)

===================================================================================
"""

from typing import List, Tuple, Union
import numpy as np


# ==============================================================================
# Core Transformation Functions
# ==============================================================================

def transform_shear_to_textbook(V_opensees: Union[List[float], np.ndarray]) -> List[float]:
    """
    Transform shear force array from OpenSees to textbook convention.

    Textbook convention: V > 0 means upward shear on left cut face.

    Parameters
    ----------
    V_opensees : list or np.ndarray
        Shear force values in OpenSees local coordinate convention (kN)

    Returns
    -------
    list[float]
        Shear force values in textbook convention (kN)
        V_textbook = -V_opensees

    Notes
    -----
    Internal forces follow OpenSees local coordinate convention.
    Visualization follows textbook/MIDAS sign convention.
    """
    if isinstance(V_opensees, np.ndarray):
        return (-V_opensees).tolist()
    return [-v for v in V_opensees]


def transform_moment_to_textbook(M_opensees: Union[List[float], np.ndarray]) -> List[float]:
    """
    Transform bending moment array from OpenSees to textbook convention.

    Textbook convention: M > 0 means sagging (tension at bottom).

    Parameters
    ----------
    M_opensees : list or np.ndarray
        Bending moment values in OpenSees local coordinate convention (kN·m)

    Returns
    -------
    list[float]
        Bending moment values in textbook convention (kN·m)
        M_textbook = -M_opensees

    Notes
    -----
    Internal forces follow OpenSees local coordinate convention.
    Visualization follows textbook/MIDAS sign convention.
    """
    if isinstance(M_opensees, np.ndarray):
        return (-M_opensees).tolist()
    return [-m for m in M_opensees]


def transform_to_textbook_convention(
    V_opensees: Union[List[float], np.ndarray],
    M_opensees: Union[List[float], np.ndarray]
) -> Tuple[List[float], List[float]]:
    """
    Transform both shear and moment arrays to textbook convention.

    This is the main entry point for visualization code.

    Parameters
    ----------
    V_opensees : list or np.ndarray
        Shear force values in OpenSees convention (kN)
    M_opensees : list or np.ndarray
        Bending moment values in OpenSees convention (kN·m)

    Returns
    -------
    tuple[list[float], list[float]]
        (V_textbook, M_textbook) - forces in textbook convention

    Example
    -------
    >>> V_raw = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -8.0, -10.0]
    >>> M_raw = [0.0, 9.0, 16.0, 21.0, 24.0, 25.0, 24.0, 21.0, 16.0, 9.0, 0.0]
    >>> V_plot, M_plot = transform_to_textbook_convention(V_raw, M_raw)
    >>> # V_plot will be negated: [-10, -8, ..., 8, 10]
    >>> # M_plot will be negated (for typical sagging beam)
    """
    V_textbook = transform_shear_to_textbook(V_opensees)
    M_textbook = transform_moment_to_textbook(M_opensees)
    return V_textbook, M_textbook


# ==============================================================================
# Convenience Functions for Single Values
# ==============================================================================

def shear_to_textbook(v: float) -> float:
    """Transform single shear value to textbook convention."""
    return -v


def moment_to_textbook(m: float) -> float:
    """Transform single moment value to textbook convention."""
    return -m


# ==============================================================================
# Sign Convention Labels (for UI/Charts)
# ==============================================================================

SIGN_CONVENTION_LABELS = {
    "shear": {
        "convention": "textbook",
        "positive": "↑ on left face",
        "description": "V > 0: upward shear on left cut face",
        "annotation": "V > 0: ↑ (left cut, textbook convention)",
    },
    "moment": {
        "convention": "textbook",
        "positive": "sagging (tension at bottom)",
        "description": "M > 0: sagging moment (concave up)",
        "annotation": "M > 0: sagging (textbook convention)",
    },
}


def get_sfd_annotation() -> str:
    """Get annotation text for SFD chart."""
    return SIGN_CONVENTION_LABELS["shear"]["annotation"]


def get_bmd_annotation() -> str:
    """Get annotation text for BMD chart."""
    return SIGN_CONVENTION_LABELS["moment"]["annotation"]


# ==============================================================================
# Frame Member Transformations (for 2D/3D frames)
# ==============================================================================

def transform_member_forces_to_textbook(
    s_positions: List[float],
    N_raw: List[float],
    V_raw: List[float],
    M_raw: List[float],
    member_type: str = "beam"
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Transform member force diagram data to textbook convention.

    For frames, the transformation depends on member orientation.
    Currently applies standard beam transformation (future: consider 3D rotation).

    Parameters
    ----------
    s_positions : list[float]
        Position along member (m)
    N_raw : list[float]
        Axial force in OpenSees convention (kN)
    V_raw : list[float]
        Shear force in OpenSees convention (kN)
    M_raw : list[float]
        Bending moment in OpenSees convention (kN·m)
    member_type : str
        "beam" or "column" - for future orientation-specific handling

    Returns
    -------
    tuple
        (s, N_plot, V_plot, M_plot) - ready for visualization

    Notes
    -----
    Axial force sign is typically kept as-is (tension positive).
    Shear and moment are transformed to textbook convention.

    Future 3D extension: Add rotation matrix for arbitrary member orientation.
    """
    # Axial force: keep same convention (tension = positive)
    N_plot = list(N_raw)

    # Shear and moment: apply textbook transformation
    V_plot = transform_shear_to_textbook(V_raw)
    M_plot = transform_moment_to_textbook(M_raw)

    return s_positions, N_plot, V_plot, M_plot


# ==============================================================================
# Documentation Constants
# ==============================================================================

OPENSEES_CONVENTION_DOC = """
OpenSees Local Coordinate Convention:
- eleForce returns nodal equilibrating forces
- Signs follow element local coordinate system
- For 2D beam: x along element axis (i→j), y perpendicular
"""

TEXTBOOK_CONVENTION_DOC = """
Textbook/MIDAS Structural Engineering Convention:
- Shear: V > 0 = upward on left cut face
- Moment: M > 0 = sagging (tension at bottom)
- Axial: N > 0 = tension
"""

TRANSFORMATION_DOC = """
Transformation Rule (beam elements):
    V_textbook = -V_opensees
    M_textbook = -M_opensees

This transformation is applied ONLY at the visualization layer.
Analysis results retain OpenSees convention for numerical consistency.
"""


# ==============================================================================
# Member Direction Enforcement (for consistent sign conventions)
# ==============================================================================

def enforce_beam_direction(
    ni_x: float, ni_y: float, nj_x: float, nj_y: float
) -> tuple[bool, bool]:
    """
    Check and enforce beam direction: i=left, j=right.

    For horizontal beams, the node with smaller x-coordinate should be i-node.
    This ensures consistent SFD/BMD representation where left is always i-end.

    Parameters
    ----------
    ni_x, ni_y : float
        Current i-node coordinates
    nj_x, nj_y : float
        Current j-node coordinates

    Returns
    -------
    tuple[bool, bool]
        (is_valid, needs_swap)
        - is_valid: True if this is a valid horizontal beam (x differs)
        - needs_swap: True if nodes should be swapped to enforce i=left

    Example
    -------
    >>> is_valid, needs_swap = enforce_beam_direction(10.0, 3.0, 0.0, 3.0)
    >>> # needs_swap = True because ni_x > nj_x
    """
    # Check if beam is horizontal (same y, different x)
    is_horizontal = abs(ni_y - nj_y) < 1e-6 and abs(ni_x - nj_x) > 1e-6

    if not is_horizontal:
        # Not a horizontal beam, no swap needed (could be inclined)
        return False, False

    # For horizontal beam: i should be at smaller x (left)
    needs_swap = ni_x > nj_x
    return True, needs_swap


def enforce_column_direction(
    ni_x: float, ni_y: float, nj_x: float, nj_y: float
) -> tuple[bool, bool]:
    """
    Check and enforce column direction: i=bottom, j=top.

    For vertical columns, the node with smaller y-coordinate should be i-node.
    This ensures consistent local coordinate system where i is at the base.

    Parameters
    ----------
    ni_x, ni_y : float
        Current i-node coordinates
    nj_x, nj_y : float
        Current j-node coordinates

    Returns
    -------
    tuple[bool, bool]
        (is_valid, needs_swap)
        - is_valid: True if this is a valid vertical column (y differs)
        - needs_swap: True if nodes should be swapped to enforce i=bottom

    Example
    -------
    >>> is_valid, needs_swap = enforce_column_direction(0.0, 6.0, 0.0, 0.0)
    >>> # needs_swap = True because ni_y > nj_y
    """
    # Check if column is vertical (same x, different y)
    is_vertical = abs(ni_x - nj_x) < 1e-6 and abs(ni_y - nj_y) > 1e-6

    if not is_vertical:
        # Not a vertical column, no swap needed (could be inclined)
        return False, False

    # For vertical column: i should be at smaller y (bottom)
    needs_swap = ni_y > nj_y
    return True, needs_swap


def enforce_member_direction(
    ni_x: float, ni_y: float, nj_x: float, nj_y: float,
    member_type: str
) -> tuple[bool, bool]:
    """
    Check and enforce member direction based on member type.

    Parameters
    ----------
    ni_x, ni_y : float
        Current i-node coordinates
    nj_x, nj_y : float
        Current j-node coordinates
    member_type : str
        "beam" or "column"

    Returns
    -------
    tuple[bool, bool]
        (is_valid, needs_swap)
    """
    if member_type.lower() == "beam":
        return enforce_beam_direction(ni_x, ni_y, nj_x, nj_y)
    elif member_type.lower() == "column":
        return enforce_column_direction(ni_x, ni_y, nj_x, nj_y)
    else:
        # Unknown type, assume no swap needed
        return True, False


def swap_element_forces(
    N_i: float, V_i: float, M_i: float,
    N_j: float, V_j: float, M_j: float
) -> tuple[float, float, float, float, float, float]:
    """
    Swap i-end and j-end forces when element direction is reversed.

    When swapping nodes (i↔j), the forces must also be adjusted:
    - Axial and shear signs are reversed (direction flips)
    - Moment signs are unchanged (rotation sense preserved)

    Parameters
    ----------
    N_i, V_i, M_i : float
        Forces at original i-end
    N_j, V_j, M_j : float
        Forces at original j-end

    Returns
    -------
    tuple[float, ...]
        (N_i_new, V_i_new, M_i_new, N_j_new, V_j_new, M_j_new)
        Forces after swapping, with appropriate sign changes
    """
    # After swap: new_i = old_j, new_j = old_i
    # Axial and shear reverse sign (direction of positive axis flips)
    # Moment preserves sign (rotation sense is invariant)
    return (-N_j, -V_j, M_j, -N_i, -V_i, M_i)


# ==============================================================================
# Frame Member Force Diagram Data Structure
# ==============================================================================

MEMBER_FORCE_FIELDS = {
    "s_m": "Position along member (m) from i-end",
    "N_kN": "Axial force (kN), tension positive - OpenSees convention",
    "V_kN": "Shear force (kN) - OpenSees local y convention",
    "M_kNm": "Bending moment (kN·m) - OpenSees convention",
}

VISUALIZATION_TRANSFORM = {
    "N": lambda n: n,           # Axial: no change
    "V": lambda v: -v,          # Shear: negate for textbook
    "M": lambda m: -m,          # Moment: negate for textbook (sagging positive)
}
