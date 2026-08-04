"""Task 1 - cross-reference cleanup for the v2 manuscript.

The user manually renumbered the Heading-1 sections (inserting an empty Section 2
slot for Related Work), but the in-text 'Section N' cross-references were only
partially / inconsistently updated. Ground truth was derived from the pre-renumber
canonical backup (..._v2.pre_g2insert_backup.docx): the correct post-renumber value
of every section reference is its first component + 1 (for N >= 2).

This script fixes ONLY the 15 paragraphs whose current value disagrees with that
ground truth. Edits are plain-text run replacements scoped to each paragraph; no
Word field (EndNote citation / SEQ caption) is touched. A backup is written first.

Run with --apply to write; default is dry-run.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "paper1_open_source_alternative" / "drafts" / "open_source_alternative_review_draft_v2.docx"
BACKUP = DOC.with_name("open_source_alternative_review_draft_v2.pre_xref_backup.docx")

NEW_ROADMAP = (
    "The remainder of this paper is organized as follows. Section 2 reviews related "
    "work and positions the present contribution against recent systems; Section 3 "
    "presents the proposed open-source workflow and node-element methodology, including "
    "zone-based decomposition for orthogonal irregular plans and the disclosure of "
    "workflow-introduced simplifications. Section 4 reports the benchmark validation "
    "against Midas Gen. Section 5 presents an IFC-derived regular application example "
    "(Sections 5.1–5.2) and a node-element L-shaped irregular-plan demonstration "
    "(Section 5.3). Section 6 discusses the position of the workflow within its "
    "benchmark-validated scope and its limitations, and Section 7 concludes the paper. "
    "Appendix A provides the clause-by-clause hand-check of the KDS load generation."
)

# Each entry: (paragraph-locator-substring, list of (old, new) run-scoped edits).
# 'old' must be wholly contained in a single run of the matched paragraph.
EDITS = [
    ("This paper presents an open-source pipeline that integrates the three capabilities",
     [("benchmark agreement reported in Section 3 is not",
       "benchmark agreement reported in Section 4 is not")]),

    ("The benchmark-validated scope is intentionally bounded.",
     [("decomposition described in Section 2.3.3;",
       "decomposition described in Section 3.3.3;")]),

    ("The contributions of the paper are threefold.",
     [("L-shaped node-element demonstration in Section 4.3.",
       "L-shaped node-element demonstration in Section 5.3.")]),

    ("The remainder of this paper is organized as follows.",
     [("__WHOLE_RUN0__", NEW_ROADMAP)]),

    ("As a supplementary check, Case 5 compares the results of a five-story",
     [("validation scope defined in Section 2.4.2.",
       "validation scope defined in Section 3.4.2.")]),

    ("For the structural configurations and loading conditions tested in this study",
     [("the discussion in Section 5.1 of how",
       "the discussion in Section 6.1 of how")]),

    # split runs: run == '4.' must become '3.'  (current 4.5.1 -> target 3.5.1)
    ("Within the implemented preliminary screening checks, no limit-state violation",
     [("__EXACT_RUN__4.", "3.")]),

    ("Distinct in scope from the benchmark validation reported",
     [("benchmark validation reported in Section 3, this example",
       "benchmark validation reported in Section 4, this example"),
      ("validation campaign identified in Section 5.3.",
       "validation campaign identified in Section 6.3.")]),

    ("For the structural class examined",
     [("IFC application example in Section 4 exercises",
       "IFC application example in Section 5 exercises")]),

    ("are presented as end-to-end execution demonstrations under the same workflow as the benchmarked cases",
     [("application examples in Section 4 are presented",
       "application examples in Section 5 are presented"),
      ("reported in Section 3; the demonstrations",
       "reported in Section 4; the demonstrations")]),

    ("This appendix records the clause-by-clause verification",
     [("steel frame example described in Section 4.1.",
       "steel frame example described in Section 5.1.")]),

    ("Status symbols used throughout the tables",
     [("simplification disclosed in Section 2.4.",
       "simplification disclosed in Section 3.4.")]),

    ("The MEP allowance (A5) is a workflow constant",
     [("workflow simplifications disclosed in Section 2.4.",
       "workflow simplifications disclosed in Section 3.4.")]),

    ("Rows D2, D7, and D8 are the three wind-related items",
     [("workflow simplifications disclosed in Section 2.4. Specifically",
       "workflow simplifications disclosed in Section 3.4. Specifically")]),

    ("Across all 49 individually checked quantities",
     [("workflow simplifications disclosed in Section 2.4 rather than",
       "workflow simplifications disclosed in Section 3.4 rather than")]),
]


def find_para(paras, locator):
    hits = [p for p in paras if locator in p.text]
    if len(hits) != 1:
        raise RuntimeError(f"locator matched {len(hits)} paragraphs (want 1): {locator!r}")
    return hits[0]


def apply_edit(p, old, new):
    if old == "__WHOLE_RUN0__":
        p.runs[0].text = new
        return True
    if old.startswith("__EXACT_RUN__"):
        token = old[len("__EXACT_RUN__"):]
        for r in p.runs:
            if r.text == token:
                r.text = new
                return True
        return False
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    return False


def main(apply: bool) -> int:
    d = docx.Document(str(DOC))
    paras = d.paragraphs
    log = []
    ok = True
    for locator, edits in EDITS:
        try:
            p = find_para(paras, locator)
        except RuntimeError as e:
            log.append(f"[FAIL-LOCATE] {e}")
            ok = False
            continue
        for old, new in edits:
            applied = apply_edit(p, old, new)
            tag = "OK " if applied else "MISS"
            if not applied:
                ok = False
            disp = old if not old.startswith("__") else old
            log.append(f"[{tag}] {locator[:45]!r}  <<{disp[:55]}>>")

    print("\n".join(log))
    print(f"\nlocator/edit results: {'ALL OK' if ok else 'SOME FAILED'}")
    if not ok:
        print("ABORT: not saving (fix the spec).")
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
