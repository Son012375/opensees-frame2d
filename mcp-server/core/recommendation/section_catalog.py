"""Section family ladder for the recommendation layer.

Step 0 audit (run via scripts/_verify_section_ordering.py on
2026-05-18 against the project Supabase instance):

    table                     rows  inversions(id-order vs h,b,area)
    ks3502.h_beam_sections      94   13   <-- id NOT monotonic
    ks3502.i_beam_sections      20    0
    ks3502.tfc_channel_sections 16    0
    ks3502.pfc_channel_sections  8    0
    ks3502.t_beam_sections      11    2
    ks3502.equal_angle_sections 51    0
    ks3568.chs_sections        235    0
    ks3568.shs_sections        n/a   (different schema — no h column)
    ks3568.rhs_hollow_sections 132    1

Authority: (family, h, b, area) ASC. The DB id is recorded only as
an audit hint; ``id_order_is_monotonic`` is exposed in the cache.

Public API
----------
list_family_ladder(family, member_type)       -> list[SectionEntry]
next_sections(current_section, member_type, steps=2) -> list[SectionEntry]
parse_family(section_name)                    -> str | None
get_section_metadata(section_name)            -> SectionMeta | None
clear_cache()                                 -> None    (test helper)
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional


# ---------------------------------------------------------------------------
# Family identification
# ---------------------------------------------------------------------------

# Canonical family keys used inside this module. They do NOT need to match
# the DB schema/table names. Each maps to a list of (schema, table, kind)
# triples queried in order. ``kind`` controls per-table column mapping
# because ks3568 SHS has no 'h' column.
_FAMILY_TABLES: dict[str, list[tuple[str, str, str]]] = {
    "H":   [("ks3502", "h_beam_sections", "h_b")],
    "I":   [("ks3502", "i_beam_sections", "h_b")],
    "TFC": [("ks3502", "tfc_channel_sections", "h_b")],
    "PFC": [("ks3502", "pfc_channel_sections", "h_b")],
    "T":   [("ks3502", "t_beam_sections", "h_b")],
    "L":   [
        ("ks3502", "equal_angle_sections", "angle_eq"),
        ("ks3502", "unequal_angle_sections", "angle_uneq"),
        ("ks3502", "unequal_leg_thickness_angle_sections", "angle_uneq"),
    ],
    "CHS": [("ks3568", "chs_sections", "chs")],
    "SHS": [("ks3568", "shs_sections", "shs")],
    "RHS": [("ks3568", "rhs_hollow_sections", "h_b")],
}

# Section name prefix → family. Order matters: longer prefixes first.
_PREFIX_TO_FAMILY: list[tuple[str, str]] = [
    ("TFC-", "TFC"),
    ("PFC-", "PFC"),
    ("CHS-", "CHS"),
    ("SHS-", "SHS"),
    ("RHS-", "RHS"),
    ("H-",   "H"),
    ("I-",   "I"),
    ("T-",   "T"),
    ("L-",   "L"),
    ("○-",   "CHS"),  # ks3568 native prefix
    ("□-",   "SHS"),  # ks3568 native prefix — may also be RHS, resolved by name
]


def parse_family(section_name: str | None) -> Optional[str]:
    """Best-effort family identifier from a section name.

    The ``□-`` prefix is ambiguous (SHS vs RHS). We disambiguate by
    looking for the ``WxBxT`` shape after the prefix: if W==B we treat
    it as SHS, otherwise RHS.
    """
    if not section_name:
        return None
    name = section_name.strip()
    for prefix, fam in _PREFIX_TO_FAMILY:
        if name.startswith(prefix):
            if prefix == "□-":
                tail = name[len(prefix):]
                try:
                    parts = tail.split("x")
                    w = float(parts[0])
                    b = float(parts[1])
                    return "SHS" if abs(w - b) < 1e-6 else "RHS"
                except (IndexError, ValueError):
                    return "SHS"
            return fam
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionEntry:
    """One ladder entry. ``area`` is in cm² (raw DB unit, NOT mm²)."""
    name: str
    family: str
    h: float          # mm
    b: float          # mm
    area: float       # cm² (DB native; multiply by 100 for mm²)
    db_id: Optional[int] = None


@dataclass(frozen=True)
class SectionMeta:
    """Lookup-friendly metadata, used by the evaluator's weight proxy."""
    name: str
    family: str
    h: float
    b: float
    area: float       # cm²
    area_mm2: float   # mm²


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class _CatalogCache:
    """Per-family parsed ladder + audit hints."""
    by_family: dict[str, list[SectionEntry]]     # family → ladder (h,b,area) ASC
    by_name: dict[str, SectionEntry]
    # Audit hints from Step 0 — exposed in tests / logs.
    id_order_is_monotonic: dict[str, bool]
    source: str   # "supabase" | "fallback" | "mixed"


_cache: Optional[_CatalogCache] = None
_cache_lock = Lock()


def clear_cache() -> None:
    """Test helper. Drops the in-process catalog cache."""
    global _cache
    with _cache_lock:
        _cache = None


def _get_cache() -> _CatalogCache:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = _build_cache()
        return _cache


# ---------------------------------------------------------------------------
# DB row → SectionEntry per table kind
# ---------------------------------------------------------------------------

def _row_to_entry(row: dict, family: str, kind: str) -> Optional[SectionEntry]:
    name = row.get("name")
    if not name:
        return None
    db_id = row.get("id")

    if kind == "h_b":
        h = row.get("h")
        b = row.get("b")
    elif kind == "angle_eq":
        # ks3502.equal_angle_sections: column is 'a', square cross section
        h = row.get("a")
        b = row.get("a")
    elif kind == "angle_uneq":
        h = row.get("a")
        b = row.get("b")
    elif kind == "chs":
        # ks3568.chs_sections: outer diameter 'd'
        h = row.get("d")
        b = row.get("d")
    elif kind == "shs":
        # ks3568.shs_sections: square hollow, only 'b' is the side
        h = row.get("b")
        b = row.get("b")
    else:
        return None

    if h is None or b is None:
        return None
    area = row.get("area")
    if area is None:
        return None

    try:
        return SectionEntry(
            name=str(name),
            family=family,
            h=float(h),
            b=float(b),
            area=float(area),
            db_id=int(db_id) if db_id is not None else None,
        )
    except (TypeError, ValueError):
        return None


def _query_family(supabase, family: str) -> list[SectionEntry]:
    out: list[SectionEntry] = []
    for schema, table, kind in _FAMILY_TABLES[family]:
        try:
            res = supabase.schema(schema).table(table).select("*").execute()
        except Exception:
            continue
        for row in (res.data or []):
            entry = _row_to_entry(row, family, kind)
            if entry is not None:
                out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Fallback (used when Supabase is unreachable)
# ---------------------------------------------------------------------------

_FALLBACK_H_SQUARE_SQ: list[SectionEntry] = [
    # 정사각 H — column ladder 핵심 (Step 0 audit 기준 10 그룹)
    SectionEntry("H-100x100", "H", 100, 100, 21.9),
    SectionEntry("H-125x125", "H", 125, 125, 30.31),
    SectionEntry("H-150x150", "H", 150, 150, 40.14),
    SectionEntry("H-175x175", "H", 175, 175, 51.21),
    SectionEntry("H-200x200", "H", 200, 200, 63.53),
    SectionEntry("H-250x250", "H", 250, 250, 92.18),
    SectionEntry("H-300x300", "H", 300, 300, 119.8),
    SectionEntry("H-350x350", "H", 350, 350, 173.9),
    SectionEntry("H-400x400", "H", 400, 400, 218.7),
]
_FALLBACK_H_WIDE: list[SectionEntry] = [
    # 대표 H beam (h ≠ b)
    SectionEntry("H-200x100", "H", 200, 100,  26.67),
    SectionEntry("H-250x125", "H", 250, 125,  37.66),
    SectionEntry("H-300x150", "H", 300, 150,  46.78),
    SectionEntry("H-350x175", "H", 350, 175,  62.91),
    SectionEntry("H-400x200", "H", 400, 200,  83.37),
    SectionEntry("H-450x200", "H", 450, 200,  95.43),
    SectionEntry("H-500x200", "H", 500, 200, 111.9),
    SectionEntry("H-600x200", "H", 600, 200, 131.7),
]


def _fallback_entries() -> dict[str, list[SectionEntry]]:
    return {
        "H": list(_FALLBACK_H_SQUARE_SQ + _FALLBACK_H_WIDE),
    }


# ---------------------------------------------------------------------------
# Build cache
# ---------------------------------------------------------------------------

def _build_cache() -> _CatalogCache:
    by_family: dict[str, list[SectionEntry]] = {}
    monotonic: dict[str, bool] = {}
    source = "supabase"

    supabase = _try_get_supabase()
    if supabase is None:
        source = "fallback"
        by_family = _fallback_entries()
    else:
        for family in _FAMILY_TABLES.keys():
            entries = _query_family(supabase, family)
            if entries:
                by_family[family] = entries

        if not by_family:
            source = "fallback"
            by_family = _fallback_entries()

    # Sort + monotonicity audit
    for family, entries in by_family.items():
        # Detect monotonicity in raw DB id order BEFORE sorting (audit).
        id_ordered = sorted(entries, key=lambda e: (e.db_id is None, e.db_id or 0))
        prev = None
        inv = 0
        for e in id_ordered:
            k = (e.h, e.b, e.area)
            if prev is not None and k < prev:
                inv += 1
            prev = k
        monotonic[family] = (inv == 0)

        # Then sort by canonical (h, b, area).
        entries.sort(key=lambda e: (e.h, e.b, e.area, e.name))
        by_family[family] = entries

    # Build name→entry index. If a name appears in multiple families
    # (shouldn't happen with these tables), the first family wins
    # deterministically by iteration order.
    by_name: dict[str, SectionEntry] = {}
    for entries in by_family.values():
        for e in entries:
            by_name.setdefault(e.name, e)

    return _CatalogCache(
        by_family=by_family,
        by_name=by_name,
        id_order_is_monotonic=monotonic,
        source=source,
    )


def _try_get_supabase():
    try:
        from core.section_3d import _get_supabase  # type: ignore
        return _get_supabase()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Member-type filter rule
# ---------------------------------------------------------------------------

def _is_h_square(entry: SectionEntry) -> bool:
    """True iff entry is an H section with strict h == b.

    Strict equality on purpose: KS H-208x202 / H-248x249 / H-298x299 are
    wide-flange rolled sections that happen to land near-square, but
    they are NOT in the H-WxW square column family (where W == W
    exactly: H-100x100 ... H-400x400). Letting them sneak into a column
    ladder produces inconsistent step sizes — see Step 0 audit.
    """
    return entry.family == "H" and entry.h == entry.b


def _passes_member_filter(entry: SectionEntry, member_type: str) -> bool:
    """Whether ``entry`` belongs in the preferred sub-ladder for ``member_type``.

    H family is partitioned into two strict sub-ladders:
        * column → square H only (h ≈ b)
        * beam   → wide H only (h > b)

    Mixing them across a step would produce nonsensical recommendations
    (e.g. promoting a 400x400 column to a 100x50 wide section). For all
    other families we don't have a meaningful split, so everything is
    eligible.
    """
    if entry.family != "H":
        return True
    if member_type == "column":
        return _is_h_square(entry)
    if member_type and member_type.startswith("beam"):
        return not _is_h_square(entry) and entry.h > entry.b
    # Unknown member_type → no filter (return everything for inspection).
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_family_ladder(family: str, member_type: str) -> list[SectionEntry]:
    """Entries of ``family`` eligible for ``member_type``, smallest→largest.

    For H family this returns a *strict* sub-ladder (square only for
    columns; wide only for beams). For other families all entries are
    returned. The result is sorted by (h, b, area) and contains no
    repeats. Returns a *new* list — caller may mutate freely.
    """
    cache = _get_cache()
    entries = cache.by_family.get(family, [])
    if not entries:
        return []
    filtered = [e for e in entries if _passes_member_filter(e, member_type)]
    return sorted(filtered, key=lambda e: (e.h, e.b, e.area, e.name))


def get_section_metadata(section_name: str | None) -> Optional[SectionMeta]:
    if not section_name:
        return None
    cache = _get_cache()
    entry = cache.by_name.get(section_name)
    if entry is None:
        return None
    return SectionMeta(
        name=entry.name,
        family=entry.family,
        h=entry.h,
        b=entry.b,
        area=entry.area,
        area_mm2=entry.area * 100.0,
    )


def next_sections(
    current_section: str,
    member_type: str,
    steps: int = 2,
) -> list[SectionEntry]:
    """Return the next ``steps`` entries strictly larger than ``current_section``.

    "Larger" is judged in the same family ladder, using the
    member-type-preferred ordering. If the current section is unknown
    or already the largest, returns an empty list.

    The result never repeats ``current_section`` and may contain fewer
    than ``steps`` entries near the top of the ladder.
    """
    if not current_section or steps <= 0:
        return []

    family = parse_family(current_section)
    if family is None:
        return []

    ladder = list_family_ladder(family, member_type)
    if not ladder:
        return []

    # Find the current entry. If exact name isn't catalogued (e.g.
    # "H-400x200x8x13" but only "H-400x200" is in DB) fall back to
    # matching on (h, b) by parsing the name.
    current_idx = next(
        (i for i, e in enumerate(ladder) if e.name == current_section), None
    )
    if current_idx is None:
        # Try (h, b) inferred from the name.
        cur_meta = _infer_hb_from_name(current_section)
        if cur_meta is None:
            return []
        # Pick the first ladder entry with (h, b, area) strictly larger
        # than the inferred dims under the preferred order.
        target = (cur_meta[0], cur_meta[1])
        idx = None
        for i, e in enumerate(ladder):
            if (e.h, e.b) >= target:
                idx = i
                if (e.h, e.b) > target:
                    break
        if idx is None:
            return []
        # Result starts at the first strictly-larger entry.
        result: list[SectionEntry] = []
        for e in ladder[idx:]:
            if (e.h, e.b) > target:
                result.append(e)
                if len(result) >= steps:
                    break
        return result

    # Standard case: known section, walk forward in the preferred order.
    return ladder[current_idx + 1:current_idx + 1 + steps]


def _infer_hb_from_name(section_name: str) -> Optional[tuple[float, float]]:
    """Parse 'H-400x200x8x13' → (400, 200). None if shape is unfamiliar."""
    try:
        body = section_name.split("-", 1)[1]
    except (IndexError, AttributeError):
        return None
    parts = body.split("x")
    if len(parts) < 1:
        return None
    try:
        h = float(parts[0])
        b = float(parts[1]) if len(parts) >= 2 else h
        return (h, b)
    except ValueError:
        return None


def id_order_audit() -> dict[str, bool]:
    """Per-family flag: True if DB id order matches (h,b,area) order.

    Exposed for debugging / migration safety — the ladder itself does
    NOT rely on this being True.
    """
    return dict(_get_cache().id_order_is_monotonic)


def source() -> str:
    """One of "supabase" / "fallback" / "mixed"."""
    return _get_cache().source
