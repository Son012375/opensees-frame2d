"""Input / output schemas for the Apply MVP pipeline.

Uses plain dataclasses (no pydantic dependency) so the module
stays lightweight and runs anywhere Python 3.10+ is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ======================================================================
# Input schemas
# ======================================================================

@dataclass
class CodeRef:
    """Design-code identifier."""
    code_id: str = "KDS 41 12 00"
    code_version: str = "2022-10-11"


@dataclass
class ComboRequest:
    """Which combination to resolve."""
    param_subtype: str = "uls"       # "uls" | "sls"
    combo_id: str = "ULS3"


@dataclass
class FloorInfo:
    """Single-floor geometry + usage."""
    level: int
    height_m: float
    occupancy_key: str
    floor_area_m2: float
    edge_beam_total_length_m: float


@dataclass
class BuildingInfo:
    """Minimal building description."""
    name: str
    floors: list[FloorInfo] = field(default_factory=list)


@dataclass
class LoadsUser:
    """User-supplied load intensities (kN/m2) and flags."""
    DL_kN_per_m2: float = 5.0
    SDL_kN_per_m2: float = 1.0
    wind_base_kN_per_m2: float = 1.2
    snow_base_kN_per_m2: float = 0.8
    eq_base_kN_per_m2: float = 0.7
    LL_fallback_kN_per_m2: float | None = None
    use_direction_expansion: bool = True


@dataclass
class MvpInput:
    """Root input schema for the Apply MVP pipeline."""
    code: CodeRef = field(default_factory=CodeRef)
    combo_request: ComboRequest = field(default_factory=ComboRequest)
    building: BuildingInfo = field(default_factory=BuildingInfo)
    loads_user: LoadsUser = field(default_factory=LoadsUser)

    @classmethod
    def from_dict(cls, d: dict) -> "MvpInput":
        """Construct from a raw JSON-like dict."""
        code = CodeRef(**d.get("code", {}))
        combo = ComboRequest(**d.get("combo_request", {}))

        bld_raw = d.get("building", {})
        floors = [FloorInfo(**f) for f in bld_raw.get("floors", [])]
        building = BuildingInfo(name=bld_raw.get("name", "unnamed"), floors=floors)

        loads = LoadsUser(**d.get("loads_user", {}))
        return cls(code=code, combo_request=combo, building=building, loads_user=loads)


# ======================================================================
# Output schemas (lightweight — just dicts wrapped for clarity)
# ======================================================================

@dataclass
class LoadCasesResult:
    """Output of LoadCaseBuilder."""
    load_cases_global: dict[str, float]        # case_key -> intensity (kN/m2)
    load_application_plan: dict[int, dict]     # floor_level -> {case_key: {type, w_kN_per_m}}
    warnings: list[str] = field(default_factory=list)
    ll_source: str = ""                        # "db" | "fallback" | "user"


@dataclass
class PipelineResult:
    """Final output of the full pipeline run."""
    input_summary: dict[str, Any] = field(default_factory=dict)
    load_cases: LoadCasesResult | None = None
    combo_source: str = ""                     # "supabase" | "example_json"
    combo_summary: dict[str, Any] = field(default_factory=dict)
    applied_factors: dict[str, float] = field(default_factory=dict)
    total_effect: float = 0.0
    opensees_result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
