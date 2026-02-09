"""
Pydantic models for Frame2D Web API
Compatible with existing MCP schema
Python 3.8 compatible
"""
from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from enum import Enum


class LoadType(str, Enum):
    FLOOR = "floor"
    LATERAL = "lateral"
    POINT = "point"


class LoadItem(BaseModel):
    """Single load item"""
    type: LoadType
    story: int = Field(ge=1, le=10)
    value: Optional[float] = None  # for floor loads (kN/m)
    fx: Optional[float] = None     # for lateral loads (kN)
    fy: Optional[float] = None     # for point loads (kN)
    node: Optional[int] = None     # for point loads


class Frame2DInput(BaseModel):
    """Input schema for Frame2D analysis - compatible with MCP"""
    # Geometry
    stories: List[float] = Field(..., min_length=1, max_length=10, description="Story heights in meters")
    bays: List[float] = Field(..., min_length=1, max_length=5, description="Bay widths in meters")

    # Sections & Material
    column_section: str = Field(default="H-300x300", description="Column section name")
    beam_section: str = Field(default="H-400x200", description="Beam section name")
    material_name: str = Field(default="SS275", description="Material name")

    # Supports
    supports: str = Field(default="fixed")

    # Analysis options
    num_elements_per_member: int = Field(default=4, ge=1, le=10)

    # Load cases
    load_cases: Dict[str, List[LoadItem]] = Field(
        default_factory=lambda: {"DL": []},
        description="Load case name -> list of loads"
    )

    # Load combinations (optional)
    load_combinations: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Combination name -> {case_name: factor}"
    )

    @validator('stories')
    def validate_stories(cls, v):
        for h in v:
            if h <= 0 or h > 20:
                raise ValueError(f"Story height must be 0 < h <= 20m, got {h}")
        return v

    @validator('bays')
    def validate_bays(cls, v):
        for w in v:
            if w <= 0 or w > 30:
                raise ValueError(f"Bay width must be 0 < w <= 30m, got {w}")
        return v


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobSummary(BaseModel):
    """Analysis result summary"""
    max_displacement_x_mm: float
    max_displacement_y_mm: float
    max_drift: float
    max_drift_story: int
    max_moment_kNm: float
    max_shear_kN: float
    max_axial_kN: float
    base_shear_kN: float
    num_stories: int
    num_bays: int


class JobResponse(BaseModel):
    """Response for job status query"""
    job_id: str
    status: JobStatus
    progress: Optional[int] = None  # 0-100
    summary: Optional[JobSummary] = None
    report_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobCreateResponse(BaseModel):
    """Response for job creation"""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Job queued"


class JobListResponse(BaseModel):
    """Response for job list query"""
    jobs: List[JobResponse]
