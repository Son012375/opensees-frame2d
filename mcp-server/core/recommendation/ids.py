"""Deterministic ID generation for issues and candidates.

Goals
-----
* Same analysis input → same ``issue_id`` / ``candidate_id`` every run.
  Required so the future UI can let users select a candidate, re-analyze,
  and have the *next* response still reference the same candidate.
* IDs must be safe as URL segments and JSON object keys: ASCII-only,
  lowercase, no whitespace, no path-confusing characters.
* If the seed information is insufficient (e.g. member_id missing,
  combo missing), fall back to a short stable digest of *whatever*
  identifying material we do have, then to a UUID — but only as last
  resort, after logging via the caller's warning sink.

The slug rules are simple on purpose: ``[a-z0-9_-]+`` only, length-capped.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Optional


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_SEG = 24


def slugify(value: Any, *, max_len: int = _MAX_SEG) -> str:
    """Lowercase ascii slug. Returns ``""`` for None/empty."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    # Replace any run of non-alphanumeric with a single underscore.
    s = _SLUG_RE.sub("_", s)
    s = s.strip("_-")
    if not s:
        return ""
    return s[:max_len]


def _digest(parts: list[str]) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:10]


def make_issue_id(
    *,
    issue_type: str,
    member_id: Optional[int] = None,
    story: Optional[int] = None,
    direction: Optional[str] = None,
    governing_combo: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    """Build a deterministic ``iss_…`` id.

    The slug shape varies by issue type but the prefix is always
    ``iss_<issue_type>_``. We include only the discriminating fields
    so identical issues collapse, but never collide across different
    members / stories / combos.

    Examples
    --------
    >>> make_issue_id(issue_type="strength_exceeded", member_id=7,
    ...               governing_combo="1.2DL+1.0LL")
    'iss_strength_exceeded_m7_1_2dl_1_0ll'

    >>> make_issue_id(issue_type="drift_exceeded", story=3, direction="X",
    ...               governing_combo="1.0DL+1.0LL+1.0EQX")
    'iss_drift_exceeded_s3_x_1_0dl_1_0ll_1_0eqx'
    """
    segs: list[str] = ["iss", slugify(issue_type)]

    if member_id is not None:
        segs.append(f"m{member_id}")
    if story is not None:
        segs.append(f"s{story}")
    if direction:
        d = slugify(direction)
        if d:
            segs.append(d)
    if governing_combo:
        c = slugify(governing_combo)
        if c:
            segs.append(c)
    if extra:
        e = slugify(extra)
        if e:
            segs.append(e)

    # If we have *nothing* discriminating beyond the type, fall back to a
    # short uuid so multiple typeless issues don't collapse onto one id.
    if len(segs) == 2:
        segs.append(uuid.uuid4().hex[:8])

    return "_".join(s for s in segs if s)


def make_candidate_id(*, issue_id: str, action_type: str) -> str:
    """Build a deterministic ``cand_…`` id derived from the issue id."""
    base = slugify(issue_id, max_len=64) or "noissue"
    act = slugify(action_type) or "action"
    candidate = f"cand_{base}_{act}"
    # Guard against absurdly long ids (very long combo names etc.).
    if len(candidate) > 96:
        candidate = f"cand_{_digest([issue_id, action_type])}_{act}"
    return candidate
