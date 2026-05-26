"""Chat tool implementations.

Each module under here exports ``ToolSpec`` instances at the module
top level. ``tool_registry.default_registry()`` knows which to wire in
at each Phase. Tools MUST NOT mutate model state — preview-only paths
go through ``services.recommendation_jobs.preview_apply`` (Phase C).
"""
