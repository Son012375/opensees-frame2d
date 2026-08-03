"""The one place OpenSees is allowed to run.

Two problems, one fix.

**Correctness.** OpenSees keeps process-global model state. The recommendation
evaluator already serialised its own reanalysis runs through a single-worker
pool, but the ``/api/building/analyze`` and ``/api/v2/analyze`` endpoints called
the solver straight from the event loop. Two visitors analysing at the same
time — or one visitor analysing while an evaluation job ran — could interleave
inside the same global model. That does not raise; it returns numbers that look
plausible and are wrong. For a tool whose entire pitch is "check my numbers",
that is the worst possible failure.

**Availability.** Those same calls are synchronous and multi-second (measured:
~6-9 s for the 3-story preset, ~13 s for the 87-member IFC example). Running
them on the event loop froze the whole process for the duration — with one
worker, every other visitor's page load, health check included, waited in line.

Routing every solver entry point through this single-worker pool serialises the
global state *and* frees the event loop, because the work now happens on a
worker thread while the loop keeps serving.

Runs queue rather than run in parallel. That is deliberate: queueing is slower,
interleaved global state is wrong.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: OpenSees' process-global state means this must stay 1.
MAX_PARALLEL_SOLVER = 1

solver_executor = ThreadPoolExecutor(
    max_workers=MAX_PARALLEL_SOLVER,
    thread_name_prefix="opensees",
)


async def run_solver(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Await ``fn(*args, **kwargs)`` on the solver thread.

    Use for anything that touches OpenSees global state or blocks for more than
    a moment (the solve itself, and the design check that reads its results).
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        solver_executor, functools.partial(fn, *args, **kwargs)
    )


def queue_depth() -> int:
    """Runs waiting for the solver thread — 0 when idle.

    Used by the demo health check so the landing page can say "busy" instead of
    handing a visitor a request that will sit in a queue.
    """
    q = getattr(solver_executor, "_work_queue", None)
    return q.qsize() if q is not None else 0
