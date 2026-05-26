"""Shared in-process state for the FastAPI app.

Modules in this package own the dicts, locks, and worker pools that more
than one router (currently /api/v2/analyze, /api/v2/recommendations/*, and
the upcoming /api/v2/chat/*) needs to reach. Keeping them out of
``main_simple`` avoids the import cycles that would otherwise appear when a
new router wants to read the same cache.
"""
