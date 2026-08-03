# Case 6 — L-shape Midas Gen Modeling Handoff

Use this document together with [_case6_lshape_spec.md](_case6_lshape_spec.md) (binding spec) to build the L-shape model in Midas Gen and produce a results JSON that compares directly against the OpenSees runner.

**OpenSees runner**: [tests/benchmark/case6_lshape.py](../tests/benchmark/case6_lshape.py)
**Auto-generated load tables**: [tests/benchmark/case6_lshape_loadtables.json](../tests/benchmark/case6_lshape_loadtables.json) — single source of truth for per-element line loads and per-node lateral forces. **Open this JSON alongside Midas modeling.**
**Midas results JSON to fill**: [tests/benchmark/midas_results/case6_lshape.json](../tests/benchmark/midas_results/case6_lshape.json) — 24 keys, currently all `null`.

---

## A. Verification snapshot from runner (2026-06-04)

Runner verified the spec via internal equilibrium check:

| Item | Value | Note |
|---|---|---|
| n_nodes | 70 | matches spec §1 |
| n_elements | 135 | 57 columns + 78 beams |
| n_base_fixed | 13 | all z=0 nodes FIXED |
| stories_detected | 5 | |
| E (MPa, actually used) | **205,000** | DEFAULT_MATERIALS fallback (DB has no SS275); spec §4 updated |
| DL Σ exported line loads (kN) | 3,457.8 | input total |
| DL Σ base reaction RZ (kN) | 3,457.81 | response total |
| DL internal equilibrium | 0.0003% | < 0.5% threshold ✓ |
| LL Σ exported line loads (kN) | 1,695.0 | input total |
| LL Σ base reaction RZ (kN) | 1,695.01 | response total |
| LL internal equilibrium | 0.0006% | < 0.5% threshold ✓ |
| EQX base shear RX | −500.000 kN | total 500 kN input → reaction −500 ✓ |
| EQY base shear RY | −500.000 kN | ✓ |

Midas modeler should reproduce these checks once the Midas model is built (before exporting result values).

---

## B. Midas modeling order (recommended)

1. **Units & material**: kN, mm. Define material SS275 with **E = 205,000 MPa**, ν = 0.3, fy = 275 MPa (spec §4).
2. **Sections**: define `H-300x300` (column) and `H-400x200` (beam) using DB property values in spec §3 (A, Ix, Iy, J, h, b, tw, tf). If Midas DB resolves to different J values, override to spec values.
3. **Nodes**: create the 70-node grid per spec §1.1 table. Easiest: type in coordinates from the table — Midas will assign IDs 1..70 in the same order.
4. **Supports**: fix all 13 base nodes (FIXED 6-DOF). Base node IDs in spec §2.
5. **Beta angles**: columns β=0°, beams β=0° (spec §5).
6. **Element generation**: connect column elements (base→1F, 1F→2F, ...) for each column line; X-beams and Y-beams per spec §1.3. Total 135 elements.
7. **Rigid floor diaphragm**: per-story slave list from spec §1.2 (binding). Per-story master suggested in §D below. **At 4F and 5F, the constraint set is Zone A nodes only — do not include any Zone B nodes (Zone B does not have 4F or 5F).**
8. **DL/LL beam line loads**: type in each beam's w_line value from `case6_lshape_loadtables.json` (`DL_line_loads_kNm` and `LL_line_loads_kNm` arrays). DO NOT use Midas "floor load" or "slab area pressure" features.
9. **EQX/EQY nodal forces**: apply joint loads from `lateral_force_table_kN` in the JSON. Apply Fx (EQX) or Fy (EQY) to every node in the per-story `node_ids` list — on the **physical perimeter/grid slave nodes**, NOT on the diaphragm reference master node (spec §6, BINDING).
10. **Run linear static**: four independent linear elastic load cases. No P-Delta, no combinations.
11. **Extract results**: see §E below for what to copy where.

---

## C. Per-node lateral force table (EQX / EQY)

V_base = 500 kN per direction; per-story Fi = V × hi / Σhj; split equally across the story's slave nodes.

| Story | z (m) | Fi (kN) | n_nodes | per-node force (kN) | Node IDs (apply Fx for EQX, Fy for EQY to each) |
|---|---|---|---|---|---|
| 1F | 3.5 | 33.3333 | 13 | 2.564103 | 2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 60, 64, 68 |
| 2F | 7.0 | 66.6667 | 13 | 5.128205 | 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 61, 65, 69 |
| 3F | 10.5 | 100.0000 | 13 | 7.692308 | 4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 62, 66, 70 |
| 4F | 14.0 | 133.3333 | 9 | 14.814815 | 5, 11, 17, 23, 29, 35, 41, 47, 53 |
| 5F | 17.5 | 166.6667 | 9 | 18.518519 | 6, 12, 18, 24, 30, 36, 42, 48, 54 |
| Σ | — | 500.0000 | — | — | — |

---

## D. Rigid floor diaphragm — master node per story (runner-picked)

The OpenSees runner uses `master = snodes_sorted[len(snodes)//2]` ([mcp-server/core/frame_3d.py:510](../mcp-server/core/frame_3d.py#L510)). Resulting picks for the L-shape:

| Story | Master node ID | Coordinates (x, y, z) m | Position |
|---|---|---|---|
| 1F | 38 | (0, 8, 3.5) | Zone A NW corner |
| 2F | 39 | (0, 8, 7.0) | Zone A NW corner |
| 3F | 40 | (0, 8, 10.5) | Zone A NW corner |
| 4F | 29 | (6, 4, 14.0) | Zone A interior (4F has only 9 nodes; median is interior) |
| 5F | 30 | (6, 4, 17.5) | Zone A interior |

**Master node choice does not affect physical response of the rigid diaphragm** — it is a kinematic reference. The Midas modeler may pick any of the listed slave nodes as the rigid-floor master, but the simplest deterministic choice is to use the same coordinates the runner picked. Slave list (all other nodes at that story, from spec §1.2) is the part that must match exactly.

---

## E. Result extraction — what fills each key in `midas_results/case6_lshape.json`

Diagnostic node coordinates (re-list for Midas query):

- **Zone A corner base** (0, 0, 0) — Midas node = OpenSees node 1
- **Shared boundary base** (12, 0, 0) — OpenSees node 13
- **Zone B far corner base** (24, 0, 0) — OpenSees node 59
- **Zone A far corner 5F** (0, 8, 17.5) — OpenSees node 42
- **Zone B far corner 3F** (24, 0, 10.5) — OpenSees node 62
- **Zone B re-entrant 3F** (24, 4, 10.5) — OpenSees node 70

Diagnostic columns:

- **Corner column 1F**: at (0, 0), spans z = 0 → 3.5. Read My, Mz at the **i-end (base, z=0)** under EQX.
- **Setback boundary column 3F→4F transition**: at (12, 4), spans z = 10.5 → 14.0. Read My, Mz at the **j-end (upper, z=14.0)** under EQX. (The runner uses the OpenSees convention where i-end has lower z; the corresponding Midas member end is the one at z=14.0 m.)

### JSON key → Midas query → value

| JSON key | Midas query |
|---|---|
| DL Base SumFz (kN) | DL case: Σ reaction Fz across all 13 base nodes |
| DL Reaction ZoneA_corner Fz (kN) | DL case: Fz at base node (0,0,0) |
| DL Reaction SharedBoundary Fz (kN) | DL case: Fz at base node (12,0,0) |
| DL Reaction ZoneB_far Fz (kN) | DL case: Fz at base node (24,0,0) |
| DL ZoneB 3F Far Corner dz (mm) | DL case: vertical disp at node (24,0,10.5) |
| LL Base SumFz (kN) | same as DL but for LL case |
| LL Reaction ZoneA_corner Fz (kN) | LL: Fz at (0,0,0) |
| LL Reaction SharedBoundary Fz (kN) | LL: Fz at (12,0,0) |
| LL Reaction ZoneB_far Fz (kN) | LL: Fz at (24,0,0) |
| LL ZoneB 3F Far Corner dz (mm) | LL: dz at (24,0,10.5) |
| EQX ZoneA Far 5F dx (mm) | EQX: dx at (0,8,17.5) |
| EQX ZoneA Far 5F dy (mm) | EQX: dy at (0,8,17.5) |
| EQX Max StoryDrift X (ratio) | EQX: max story drift envelope in X (interstory drift / story height) |
| EQX Base Shear Fx (kN) | EQX: Σ reaction Fx (will be ~ −500) |
| EQY ZoneA Far 5F dx (mm) | EQY: dx at (0,8,17.5) |
| EQY ZoneA Far 5F dy (mm) | EQY: dy at (0,8,17.5) |
| EQY Max StoryDrift Y (ratio) | EQY: max story drift envelope in Y |
| EQY Base Shear Fy (kN) | EQY: Σ reaction Fy (~ −500) |
| EQX Torsion (A_5F dx - B_3F dx) (mm) | EQX: dx at (0,8,17.5) − dx at (24,4,10.5) |
| EQY Torsion (A_5F dy - B_3F dy) (mm) | EQY: dy at (0,8,17.5) − dy at (24,4,10.5) |
| EQX CornerCol1F Base My (kNm) | EQX: My at i-end (base, z=0) of column at (0,0) z=0→3.5 |
| EQX CornerCol1F Base Mz (kNm) | EQX: Mz at i-end of same column |
| EQX SetbackCol 3F->4F UpperEnd My (kNm) | EQX: My at j-end (top, z=14.0) of column at (12,4) z=10.5→14.0 |
| EQX SetbackCol 3F->4F UpperEnd Mz (kNm) | EQX: Mz at j-end of same column |

### Sign convention

- OpenSees reports `RX_kN`, `RY_kN`, `RZ_kN` reactions as response forces (so EQX reaction Fx is reported as −500 since the applied force was +500). Midas reports vary by output format; modeler should ensure sign convention matches before populating JSON.
- Downward = −Fz everywhere.
- Member moments `My_i_kNm`, `Mz_i_kNm` follow OpenSees local axis convention. If Midas reports in a different local-axis convention, the modeler should align signs before populating JSON.
- For drift ratios, sign is removed (`abs`) before reporting.

---

## F. Comparison workflow (Step 4)

Once `midas_results/case6_lshape.json` is filled:

```
python tests/benchmark/run_benchmarks.py case6_lshape
```

Output is a table with one row per metric showing `OpenSees`, `Midas`, `Diff%`, and `Status`. Status logic (extended in this campaign — 3 levels):

| Status | Diff threshold |
|---|---|
| OK | ≤ 1.0% |
| CHECK | 1.0% < diff ≤ 5.0% |
| FAIL | > 5.0% |
| PENDING | Midas value is null (un-filled) |

Per Decision-Gate criteria in `C:\Users\youm\.claude\plans\scie-shiny-sparrow.md` §Step 5:

- **Scenario A** (paper-integration eligible): ≥ 80% OK + 0 FAIL
- **Scenario B** (paper unchanged): FAIL ≥ 3 OR systemic CHECK pattern
- **Scenario C** (partial; ablation needed): 1–2 FAIL OR borderline systemic CHECK

---

## G. Out-of-spec items deliberately not loaded

- Member self-weight (DL pressure handles vertical mass)
- Combinations (1.2DL + 1.6LL etc.) — each case run independently
- Mass-source / dynamic / spectrum — Case 6 v1 is static only
- P-Δ / Corotational — linear elastic only
- Wind / snow / temperature

---

**End of handoff.** After Midas modeling and JSON fill, trigger Step 4 (comparison) and then Step 5 (Decision Gate per plan).
