# -*- coding: utf-8 -*-
"""Tier-2 (reproducibility) fixes to the Data-availability paragraph P180.

Field-safe: both targets are single plain-text runs of P180 (run[0] = IFC-model
availability, run[1] = version/pinned sentence). No Zotero fldChar/instrText touched.

  RM-2  version accuracy + make 'pinned' true: Python 3.12 -> 3.13(3.12+), name the
        exact tested versions, point to the reproduction requirements file.
  IFC-SRC (A-hybrid): disclose that the shared IFC application-example model is a
        clean IFC2X3 reproduction equivalent (parses via the Section 3.3 node-element
        + connectivity-repair pipeline), provided in place of the original Revit export.
        (User confirmed the study's example WAS a real parsed Revit IFC export, so
        P121/P122 stay; only the shared-artifact nature is disclosed here.)

Usage: python scripts/_apply_tier2_p180_2026-07-07.py [--apply]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "paper1_open_source_alternative" / "drafts" / "open_source_alternative_review_draft_v2.docx"
BACKUP = DOC.with_name("open_source_alternative_review_draft_v2.pre_tier2p180_2026-07-07.docx")

VER_OLD = ("Python 3.12 on OpenSeesPy, ifcopenshell, and NumPy, with package versions "
           "pinned in the repository")
VER_NEW = ("Python 3.13 (3.12+) on OpenSeesPy 3.3.0.1.1, ifcopenshell 0.8.4, and NumPy 2.3, "
           "with exact package versions pinned in a reproduction requirements file in the "
           "repository")

IFC_OLD = "the IFC application-example model, the five benchmark case definitions"
IFC_NEW = ("the IFC application-example model (a clean IFC 2x3 file that parses through the "
           "Section 3.3 node-element and connectivity-repair pipeline to the analysis graph "
           "used in the study, provided in place of the original vendor Revit export, together "
           "with the resulting node-element model), the five benchmark case definitions")


def _repl_in_para_runs(p, old, new, tag):
    hits = [r for r in p.runs if old in r.text]
    if len(hits) != 1:
        raise SystemExit(f"ABORT [{tag}]: {old!r} found in {len(hits)} runs (want 1).")
    hits[0].text = hits[0].text.replace(old, new)


def main(apply: bool) -> int:
    d = docx.Document(str(DOC))
    p180 = None
    for p in d.paragraphs:
        if ("openly available" in p.text and "IFC application-example model" in p.text
                and "Data availability" not in p.text):
            p180 = p
            break
    if p180 is None:
        raise SystemExit("ABORT: P180 not found.")
    if "reproduction requirements file" in p180.text or "clean IFC 2x3 file" in p180.text:
        raise SystemExit("ABORT: Tier-2 P180 fix already applied.")

    _repl_in_para_runs(p180, VER_OLD, VER_NEW, "RM-2/version")
    _repl_in_para_runs(p180, IFC_OLD, IFC_NEW, "IFC-SRC/availability")

    print("Planned P180 edits:")
    print("  [RM-2]    Python 3.12 -> 3.13(3.12+) + exact versions + reproduction requirements file")
    print("  [IFC-SRC] IFC application-example model -> clean IFC2X3 repro equivalent (A-hybrid)")

    if not apply:
        print("\nDRY-RUN (no save). Re-run with --apply.")
        print("\n--- P180 after edits ---")
        print(p180.text)
        return 0

    shutil.copy2(DOC, BACKUP)
    print(f"\nbackup -> {BACKUP.name}")
    d.save(str(DOC))
    print(f"SAVED -> {DOC.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(apply="--apply" in sys.argv))
