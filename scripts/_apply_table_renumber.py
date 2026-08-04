"""Part A step 1 - unlink the two broken SEQ table-caption fields to plain text and
renumber every table caption + body reference by +1, opening the 'Table 1' slot for
the Related-Work comparison table (inserted separately).

The two captions 'Benchmark case specifications' and 'Example building specifications
and load summary' use a ` SEQ 그림 \\* ARABIC ` field ('그림' = Korean *Figure*) with a
cached display value — broken (a global F9 would mis-number them). They are unlinked to
static text here (author-approved) so the whole table sequence is plain text.

Final order:  1 = comparison (separate), 2 IFC-map, 3 KDS-load, 4 OpenSees-cfg,
5 Benchmark-specs, 5a Case-1, 6 Benchmark-results, 7 Case-4-ablation,
8 Example-building, 9 Node-element-IFC, 10 L-shape.

Backup -> ..._v2.pre_table_backup.docx.  Run with --apply (default dry-run).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "paper1_open_source_alternative" / "drafts" / "open_source_alternative_review_draft_v2.docx"
BACKUP = DOC.with_name("open_source_alternative_review_draft_v2.pre_table_backup.docx")

# locator (unique, number-free where possible) -> new SEQ number
UNLINK = [
    ("Benchmark case specifications", "5"),
    ("Example building specifications and load summary", "8"),
]

# plain-text caption renumbers: (locator, old, new)
CAPTIONS = [
    ("IFC entity to node-element analysis mapping", "Table 1. IFC entity", "Table 2. IFC entity"),
    ("KDS-based load automation summary", "Table 2. KDS-based load", "Table 3. KDS-based load"),
    ("OpenSeesPy frame model configuration parameters", "Table 3. OpenSeesPy frame model", "Table 4. OpenSeesPy frame model"),
    ("Case 1 three-way comparison", "Table 4a. Case 1 three-way", "Table 5a. Case 1 three-way"),
    ("Benchmark comparison results against Midas Gen by case", "Table 5. Benchmark comparison results", "Table 6. Benchmark comparison results"),
    ("Case 4 discrepancy ablation. Story Drift 1", "Table 6. Case 4 discrepancy ablation", "Table 7. Case 4 discrepancy ablation"),
    ("Node-element IFC application example and preliminary analysis results", "Table 8. Node-element IFC application", "Table 9. Node-element IFC application"),
    ("L-shaped application example: model size and gravity-load", "Table 9. L-shaped application example", "Table 10. L-shaped application example"),
]

# body references: (locator, old, new)
BODY = [
    ("KDS load generation. Table 1 summarizes", "Table 1 summarizes", "Table 2 summarizes"),
    ("Table 5 summarizes the benchmark comparison", "Table 5 summarizes", "Table 6 summarizes"),
    ("comparison is presented in Table 4a", "in Table 4a", "in Table 5a"),
    ("the candidate drivers (Table 6, Figure 4)", "(Table 6, Figure 4)", "(Table 7, Figure 4)"),
]


def find_one(paras, locator):
    hits = [p for p in paras if locator in p.text]
    if len(hits) != 1:
        raise RuntimeError(f"locator matched {len(hits)} (want 1): {locator!r}")
    return hits[0]


def unlink_seq(p, new_num):
    in_field = False
    after_sep = False
    result = None
    remove = []
    for r in p.runs:
        fc = r._r.find(qn('w:fldChar'))
        it = r._r.find(qn('w:instrText'))
        if fc is not None:
            ct = fc.get(qn('w:fldCharType'))
            remove.append(r)
            if ct == 'begin':
                in_field, after_sep = True, False
            elif ct == 'separate':
                after_sep = True
            elif ct == 'end':
                in_field, after_sep = False, False
        elif it is not None:
            remove.append(r)
        else:
            if in_field and after_sep and result is None:
                result = r
    if result is None:
        return False
    result.text = str(new_num)
    for r in remove:
        r._r.getparent().remove(r._r)
    return True


def replace_run(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    return False


def main(apply: bool) -> int:
    d = docx.Document(str(DOC))
    paras = d.paragraphs
    log, ok = [], True

    # guard: already renumbered? (a 'Table 10' caption means done)
    if any(p.text.strip().startswith("Table 10.") for p in paras):
        print("ABORT: 'Table 10.' already present (already renumbered).")
        return 1

    for loc, num in UNLINK:
        try:
            p = find_one(paras, loc)
            done = unlink_seq(p, num)
            log.append(f"[{'OK ' if done else 'MISS'}] UNLINK {loc[:40]!r} -> Table {num}.  now: {p.text.strip()[:38]!r}")
            ok &= done
        except RuntimeError as e:
            log.append(f"[FAIL] {e}"); ok = False

    for loc, old, new in CAPTIONS:
        try:
            p = find_one(paras, loc)
            done = replace_run(p, old, new)
            log.append(f"[{'OK ' if done else 'MISS'}] CAP  {old!r} -> {new.split('.')[0]!r}")
            ok &= done
        except RuntimeError as e:
            log.append(f"[FAIL] {e}"); ok = False

    for loc, old, new in BODY:
        try:
            p = find_one(paras, loc)
            done = replace_run(p, old, new)
            log.append(f"[{'OK ' if done else 'MISS'}] BODY {old!r} -> {new!r}")
            ok &= done
        except RuntimeError as e:
            log.append(f"[FAIL] {e}"); ok = False

    print("\n".join(log))
    print(f"\nresult: {'ALL OK' if ok else 'SOME FAILED'}")
    if not ok:
        print("ABORT: not saving.")
        return 1
    if not apply:
        print("\nDRY-RUN (pass --apply).")
        return 0
    shutil.copy2(DOC, BACKUP)
    print(f"backup -> {BACKUP.name}")
    d.save(str(DOC))
    print(f"SAVED -> {DOC.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(apply="--apply" in sys.argv))
