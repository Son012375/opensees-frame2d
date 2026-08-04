"""Task 2 - reorder Fig8/Table9 (the Case-4 ablation artefacts) into document order.

Document order before:  Figures 1,2,3,[8],4,5,6,7   Tables ...,5,[9],6(SEQ),7,8
The ablation Figure 8 sits between Fig 3 and the old Fig 4; the ablation Table 9
sits between Table 5 and the old Table 6. They must take the Fig 4 / Table 6 slot,
shifting the originals back by one:
   Fig:  8->4, 4->5, 5->6, 6->7, 7->8
   Tbl:  9->6, 7->8, 8->9     (old Table 6 -> 7 is a SEQ field, handled in Word)

Only PLAIN-TEXT captions and body references are edited here. The two SEQ-field
captions (old Table 4 [unchanged] and old Table 6 [-> 7]) are NOT touched: they use
a ` SEQ 그림 \\* ARABIC ` field (note: '그림' = Korean for *Figure*) with cached
display values, so a global F9 would mis-compute them. See the printed advisory.

Run with --apply to write; default dry-run. Idempotent-safe: aborts if an 'old'
token is missing (prevents double application).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "paper1_open_source_alternative" / "drafts" / "open_source_alternative_review_draft_v2.docx"
BACKUP = DOC.with_name("open_source_alternative_review_draft_v2.pre_figtab_backup.docx")

# (number-free unique locator, [(old, new) run-scoped replacements])
EDITS = [
    # ablation body ref: Table 9 -> 6, Figure 8 -> 4 (same run)
    ("controlled ablation of the candidate drivers",
     [("(Table 9, Figure 8)", "(Table 6, Figure 4)")]),
    # ablation Table caption 9 -> 6
    ("discrepancy ablation. Story Drift 1",
     [("Table 9. Case 4", "Table 6. Case 4")]),
    # ablation Figure caption 8 -> 4
    ("discrepancy attribution: Story 1 drift",
     [("Figure 8. Case 4", "Figure 4. Case 4")]),
    # body: IFC deformed shape Figure 4 -> 5
    ("deformed shape under the governing seismic load combination is illustrated",
     [("illustrated in Figure 4.", "illustrated in Figure 5.")]),
    # caption: IFC deformed shape Figure 4 -> 5
    ("deformed shape of the IFC-derived example building under the governing",
     [("Figure 4. Three-dimensional", "Figure 5. Three-dimensional")]),
    # body: screening summary Figure 5 -> 6
    ("drift and member-strength screening results across the evaluated",
     [("presented in Figure 5.", "presented in Figure 6.")]),
    # caption: screening summary Figure 5 -> 6
    ("Summary of preliminary review outputs for the IFC-derived",
     [("Figure 5. Summary", "Figure 6. Summary")]),
    # caption: L-shape zone decomposition Figure 6 -> 7
    ("Zone-based decomposition of the L-shaped plan example",
     [("Figure 6. Zone-based", "Figure 7. Zone-based")]),
    # caption: L-shape model Figure 7 -> 8
    ("node-element model assembled from the zone decomposition",
     [("Figure 7. L-shaped", "Figure 8. L-shaped")]),
    # caption: IFC application results Table 7 -> 8
    ("Node-element IFC application example and preliminary analysis results",
     [("Table 7. Node-element", "Table 8. Node-element")]),
    # caption: L-shape results Table 8 -> 9
    ("L-shaped application example: model size and gravity-load",
     [("Table 8. L-shaped", "Table 9. L-shaped")]),
]


def find_para(paras, locator):
    hits = [p for p in paras if locator in p.text]
    if len(hits) != 1:
        raise RuntimeError(f"locator matched {len(hits)} paragraphs (want 1): {locator!r}")
    return hits[0]


def replace_in_run(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    return False


def main(apply: bool) -> int:
    d = docx.Document(str(DOC))
    paras = d.paragraphs
    log, ok = [], True
    for locator, edits in EDITS:
        try:
            p = find_para(paras, locator)
        except RuntimeError as e:
            log.append(f"[FAIL-LOCATE] {e}")
            ok = False
            continue
        for old, new in edits:
            applied = replace_in_run(p, old, new)
            log.append(f"[{'OK ' if applied else 'MISS'}] {locator[:42]!r}  {old!r} -> {new.split('.')[0]!r}")
            if not applied:
                ok = False
    print("\n".join(log))
    print(f"\nresult: {'ALL OK' if ok else 'SOME FAILED'}")
    if not ok:
        print("ABORT: not saving.")
        return 1
    if not apply:
        print("\nDRY-RUN (pass --apply to write).")
        return 0
    shutil.copy2(DOC, BACKUP)
    print(f"backup -> {BACKUP.name}")
    d.save(str(DOC))
    print(f"SAVED -> {DOC.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(apply="--apply" in sys.argv))
