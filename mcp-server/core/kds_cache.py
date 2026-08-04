"""Offline mirror of the KDS reference tables.

Why this exists
---------------
Every load/section lookup reads Supabase. A free-tier project auto-pauses after
a period of inactivity, and ``design_spectrum._get_zone_coefficient`` had no
fallback — so a paused database turned *every* analysis into
``[Errno 11001] getaddrinfo failed``. For a link handed to strangers that is the
most expensive failure there is: the visitor concludes the tool does not work.

What this does
--------------
Wraps the Supabase client. While the database answers, queries run against it
and results are **byte-identical to before** — this layer is transparent. When a
query raises a transport error, the same query is replayed against a local JSON
snapshot of the table. The snapshot is produced from the database itself
(``scripts/snapshot_kds_tables.py``), so offline answers match online answers;
this is a mirror, not a substitute set of values.

After one transport failure the client stays offline for ``_OFFLINE_TTL``
seconds. Without that, every one of the hundreds of lookups in a single analysis
would sit through its own connection timeout.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# core/ -> mcp-server/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(os.environ.get("KDS_CACHE_DIR", _REPO_ROOT / "data" / "kds_cache"))

# Every (schema, table) the runtime reads.
MIRRORED_TABLES: tuple[tuple[str, str], ...] = (
    ("public", "hazard_region_values"),
    ("public", "load_params"),
    ("public", "materials"),
    ("ks3502", "h_beam_sections"),
    ("ks3502", "i_beam_sections"),
    ("ks3502", "tfc_channel_sections"),
    ("ks3502", "pfc_channel_sections"),
    ("ks3502", "t_beam_sections"),
    ("ks3502", "equal_angle_sections"),
    ("ks3502", "unequal_angle_sections"),
    ("ks3502", "unequal_leg_thickness_angle_sections"),
    ("ks3502", "flat_bar_sections"),
    ("ks3568", "chs_sections"),
    ("ks3568", "shs_sections"),
    ("ks3568", "rhs_hollow_sections"),
)

_OFFLINE_TTL = float(os.environ.get("KDS_CACHE_OFFLINE_TTL", "60"))

# KDS_CACHE_MODE
#   auto   (default) database first, snapshot only when the database fails.
#          Behaviour is unchanged from before this module existed.
#   prefer snapshot first, database only for tables with no snapshot. The
#          reference tables are a frozen code edition, so this is safe — and it
#          removes hundreds of network round-trips per analysis (measured:
#          9.5s -> 6.3s for the 3-story preset) as well as the outage risk.
#          This is the right setting for a public demo deployment.
#   only   snapshot only; never touch the network. Missing table -> error.
_MODE = os.environ.get("KDS_CACHE_MODE", "auto").strip().lower()
if _MODE not in {"auto", "prefer", "only"}:
    _MODE = "auto"


def mode() -> str:
    return _MODE

_lock = threading.Lock()
_offline_until: float = 0.0
_warned = False
_table_cache: dict[tuple[str, str], list[dict]] = {}


def cache_path(schema: str, table: str) -> Path:
    return CACHE_DIR / f"{schema}.{table}.json"


def load_table(schema: str, table: str) -> list[dict] | None:
    """Rows from the local snapshot, or None when no snapshot exists."""
    key = (schema, table)
    if key in _table_cache:
        return _table_cache[key]
    path = cache_path(schema, table)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"] if isinstance(payload, dict) else payload
    except Exception as exc:                                    # pragma: no cover
        logger.warning("KDS cache unreadable (%s): %s", path, exc)
        return None
    _table_cache[key] = rows
    return rows


def _is_transport_error(exc: BaseException) -> bool:
    """Network-shaped failure (DNS, refused, timeout) rather than a query error.

    A 4xx from PostgREST means the query itself is wrong — falling back would
    hide a real bug, so those must keep raising.
    """
    name = type(exc).__name__
    if name in {"ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException",
                "NetworkError", "RemoteProtocolError", "PoolTimeout", "ProxyError"}:
        return True
    text = f"{name}: {exc}".lower()
    return any(s in text for s in (
        "getaddrinfo", "connection refused", "temporary failure in name resolution",
        "timed out", "connection reset", "network is unreachable",
        "max retries exceeded", "name or service not known",
    ))


def _go_offline(exc: BaseException) -> None:
    global _offline_until, _warned
    with _lock:
        _offline_until = time.time() + _OFFLINE_TTL
        if not _warned:
            _warned = True
            logger.warning(
                "KDS database unreachable (%s) — serving reference tables from the "
                "local snapshot in %s for the next %.0fs.",
                exc, CACHE_DIR, _OFFLINE_TTL,
            )


def is_offline() -> bool:
    return time.time() < _offline_until


def reset_offline_state() -> None:
    """Test hook — forget that the database was unreachable."""
    global _offline_until, _warned
    with _lock:
        _offline_until = 0.0
        _warned = False


# ── offline query evaluation ────────────────────────────────────────────────
# Only the operators the codebase actually uses are implemented. An unknown
# operator raises rather than silently returning wrong rows.

def _as_number(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _like_match(value: Any, pattern: str) -> bool:
    if value is None:
        return False
    text = str(value).lower()
    pat = pattern.lower()
    if pat.startswith("%") and pat.endswith("%"):
        return pat[1:-1] in text
    if pat.startswith("%"):
        return text.endswith(pat[1:])
    if pat.endswith("%"):
        return text.startswith(pat[:-1])
    return text == pat


def _apply_op(rows: list[dict], op: str, args: tuple, kwargs: dict) -> list[dict]:
    if op == "eq":
        col, val = args[0], args[1]
        return [r for r in rows if str(r.get(col)) == str(val)]
    if op == "neq":
        col, val = args[0], args[1]
        return [r for r in rows if str(r.get(col)) != str(val)]
    if op == "ilike":
        col, pat = args[0], args[1]
        return [r for r in rows if _like_match(r.get(col), pat)]
    if op in ("gte", "lte", "gt", "lt"):
        col, val = args[0], args[1]
        bound = _as_number(val)
        out = []
        for r in rows:
            cur = _as_number(r.get(col))
            if cur is None or bound is None:
                continue
            if ((op == "gte" and cur >= bound) or (op == "lte" and cur <= bound)
                    or (op == "gt" and cur > bound) or (op == "lt" and cur < bound)):
                out.append(r)
        return out
    if op == "or_":
        # "col.ilike.%x%,col2.ilike.%y%"
        keep: list[dict] = []
        clauses = [c for c in str(args[0]).split(",") if c.strip()]
        for r in rows:
            for clause in clauses:
                parts = clause.split(".", 2)
                if len(parts) != 3:
                    continue
                col, sub, val = parts
                if sub == "ilike" and _like_match(r.get(col), val):
                    keep.append(r)
                    break
                if sub == "eq" and str(r.get(col)) == val:
                    keep.append(r)
                    break
        return keep
    if op == "order":
        col = args[0]
        desc = bool(kwargs.get("desc", False))
        return sorted(
            rows,
            key=lambda r: (r.get(col) is None, _as_number(r.get(col))
                           if _as_number(r.get(col)) is not None else str(r.get(col))),
            reverse=desc,
        )
    if op == "limit":
        return rows[: int(args[0])]
    if op in ("select", "schema", "table"):
        return rows
    raise NotImplementedError(f"KDS cache: unsupported query operator '{op}'")


def query_offline(schema: str, table: str, ops: Iterable[tuple]) -> list[dict]:
    rows = load_table(schema, table)
    if rows is None:
        raise LookupError(
            f"KDS database is unreachable and no local snapshot exists for "
            f"{schema}.{table}. Run: python scripts/snapshot_kds_tables.py"
        )
    out = list(rows)
    for op, args, kwargs in ops:
        out = _apply_op(out, op, args, kwargs)
    return out


# ── resilient client ────────────────────────────────────────────────────────

class _ResilientQuery:
    """Records the chained operators while building the real PostgREST query."""

    _CHAINABLE = ("select", "eq", "neq", "ilike", "like", "gte", "lte", "gt",
                  "lt", "or_", "order", "limit", "range", "in_", "is_", "not_")

    def __init__(self, real, schema: str, table: str, ops: list[tuple] | None = None):
        self._real = real
        self._schema = schema
        self._table = table
        self._ops = ops or []

    def __getattr__(self, name):
        if name not in self._CHAINABLE:
            return getattr(self._real, name)

        def _chain(*args, **kwargs):
            real_next = getattr(self._real, name)(*args, **kwargs) if self._real else None
            return _ResilientQuery(real_next, self._schema, self._table,
                                   self._ops + [(name, args, kwargs)])
        return _chain

    def _snapshot_available(self) -> bool:
        return load_table(self._schema, self._table) is not None

    def execute(self):
        # snapshot-first modes
        if _MODE == "only" or (_MODE == "prefer" and self._snapshot_available()):
            return SimpleNamespace(
                data=query_offline(self._schema, self._table, self._ops), count=None
            )

        if self._real is not None and not is_offline():
            try:
                return self._real.execute()
            except Exception as exc:
                if not _is_transport_error(exc):
                    raise
                _go_offline(exc)
        return SimpleNamespace(
            data=query_offline(self._schema, self._table, self._ops), count=None
        )


class ResilientClient:
    """Drop-in wrapper around a Supabase client with a local-snapshot fallback."""

    def __init__(self, real, schema: str = "public"):
        self._real = real
        self._schema = schema

    def schema(self, name: str) -> "ResilientClient":
        real = None
        if self._real is not None:
            try:
                real = self._real.schema(name)
            except Exception:                                    # pragma: no cover
                real = None
        return ResilientClient(real, name)

    def table(self, name: str) -> _ResilientQuery:
        real = None
        if self._real is not None:
            try:
                real = self._real.table(name)
            except Exception:                                    # pragma: no cover
                real = None
        return _ResilientQuery(real, self._schema, name)

    # anything else (auth, storage, ...) passes straight through
    def __getattr__(self, name):
        return getattr(self._real, name)


def wrap(client, schema: str = "public") -> ResilientClient:
    """Wrap a Supabase client so reference lookups survive an outage."""
    if isinstance(client, ResilientClient):
        return client
    return ResilientClient(client, schema)
