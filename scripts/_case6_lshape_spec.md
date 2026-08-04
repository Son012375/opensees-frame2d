# Case 6 — L-shape Decision-Gate Validation Spec

**Purpose**: Single source of truth so the OpenSees runner and the Midas Gen modeler describe the exact same structural model and load. This is a **solver/model response comparison** (not nominal-pressure-conservation). Both engines receive the same per-element beam line loads and per-node lateral force vectors — the OpenSees runner exports those tables, and Midas inputs them verbatim. No floor-load auto-distribution on either side.

**Status**: Step 1 deliverable of the Case 6 plan (`C:\Users\youm\.claude\plans\scie-shiny-sparrow.md`). Pending user confirm before Step 2 (OpenSees runner) and Step 3 (Midas handoff).

**Units**: Internally OpenSees uses N, mm. All values in this spec are stated in m / kN / kN·m (reporting units). Conversion is applied inside the runner.

---

## 1. Geometry & Topology

L-shape plan: 5-story left wing (Zone A) + 3-story right wing setback (Zone B), sharing the column line at x = 12 m, y ∈ {0, 4}.

```
Y(m)
 8 ┌───┬───┐                                  Zone A: 5 stories (12 m × 8 m)
   │ A │ A │                                  Zone B: 3 stories (12 m × 4 m, setback)
 4 ├───┼───┼───┬───┐                          Shared boundary (x=12): supports both zones at y∈{0,4}
   │ A │ A │ B │ B │                          Story height: 3.5 m (uniform)
 0 └───┴───┴───┴───┘
   0   6   12  18  24  X(m)
```

**Story levels**: z = 0.0, 3.5, 7.0, 10.5, 14.0, 17.5 m
- 1F, 2F, 3F: full L-shape footprint (both zones)
- 4F, 5F: Zone A only (12 m × 8 m)
- 5F is the roof of Zone A; 3F is the roof of Zone B

**Column grid (x, y) — 13 vertical lines total**

| Zone | (x, y) coordinates | # of column lines | Stories present |
|------|---|---|---|
| A (left) | (0,0), (6,0), (12,0), (0,4), (6,4), (12,4), (0,8), (6,8), (12,8) | 9 | 1F→5F (5) |
| B (right) | (18,0), (24,0), (18,4), (24,4) | 4 | 1F→3F (3) |

**Element count (must match `build_l_shape_model()` in [tests/test_v2_irregular.py:66](tests/test_v2_irregular.py#L66))**:
- Columns: 9 × 5 + 4 × 3 = 45 + 12 = **57**
- Beams: see §1.3 = **78**
- Nodes: see §1.2 = **70**

### 1.1 Node-ID convention (used in tables below)

Node IDs are auto-assigned in the order shown in [build_l_shape_model()](tests/test_v2_irregular.py#L66). For each (x, y) the nodes are created bottom-up. Per column line:

- Zone A column lines (9 of them, each spans z = 0 → 17.5): 6 nodes per line, total **54 nodes**
- Zone B column lines (4 of them, each spans z = 0 → 10.5): 4 nodes per line, total **16 nodes**

Therefore: **70 nodes total**.

Node IDs follow the (x, y) ordering of `left_cols` / `right_cols`:

| Column line | x, y | Stack node IDs (low z → high z) |
|---|---|---|
| L1 | (0, 0) | 1, 2, 3, 4, 5, 6 |
| L2 | (6, 0) | 7, 8, 9, 10, 11, 12 |
| L3 | (12, 0) | 13, 14, 15, 16, 17, 18 |
| L4 | (0, 4) | 19, 20, 21, 22, 23, 24 |
| L5 | (6, 4) | 25, 26, 27, 28, 29, 30 |
| L6 | (12, 4) | 31, 32, 33, 34, 35, 36 |
| L7 | (0, 8) | 37, 38, 39, 40, 41, 42 |
| L8 | (6, 8) | 43, 44, 45, 46, 47, 48 |
| L9 | (12, 8) | 49, 50, 51, 52, 53, 54 |
| R1 | (18, 0) | 55, 56, 57, 58 |
| R2 | (24, 0) | 59, 60, 61, 62 |
| R3 | (18, 4) | 63, 64, 65, 66 |
| R4 | (24, 4) | 67, 68, 69, 70 |

z-index within each stack: 0 = z=0 (base), 1 = z=3.5 (1F), 2 = z=7.0 (2F), 3 = z=10.5 (3F), 4 = z=14.0 (4F, A only), 5 = z=17.5 (5F, A only).

### 1.2 Per-story node lists (binding for diaphragm + lateral load application)

| Story | z (m) | Node IDs | Count |
|---|---|---|---|
| Base | 0.0 | 1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 59, 63, 67 | 13 |
| 1F | 3.5 | 2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 60, 64, 68 | 13 |
| 2F | 7.0 | 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 61, 65, 69 | 13 |
| 3F | 10.5 | 4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 62, 66, 70 | 13 |
| 4F | 14.0 | 5, 11, 17, 23, 29, 35, 41, 47, 53 | 9 |
| 5F | 17.5 | 6, 12, 18, 24, 30, 36, 42, 48, 54 | 9 |

Total: 13 + 13 + 13 + 13 + 9 + 9 = **70 nodes** ✓

### 1.3 Beams (78 total)

Per story s (z = 3.5 × s):

**X-direction beams**:
- Zone A — at every story 1F→5F: for y ∈ {0, 4, 8}: two segments (0↔6) and (6↔12). → 3 × 2 = 6 per story
- Zone B — only at 1F→3F: for y ∈ {0, 4}: two segments (12↔18) and (18↔24). → 2 × 2 = 4 per story

**Y-direction beams**:
- Zone A — at every story 1F→5F: for x ∈ {0, 6, 12}: two segments (y=0↔4) and (y=4↔8). → 3 × 2 = 6 per story
- Zone B — only at 1F→3F: for x ∈ {18, 24}: one segment (y=0↔4). → 2 × 1 = 2 per story

Totals:
- 1F, 2F, 3F: (6 + 4) X + (6 + 2) Y = 18 beams × 3 = 54
- 4F, 5F: 6 X + 6 Y = 12 beams × 2 = 24
- **Total beams = 78** ✓

---

## 2. Boundary Conditions

All 13 base nodes (z = 0): **FIXED 6-DOF** (Tx, Ty, Tz, Rx, Ry, Rz). No releases, no springs. Base node IDs: **1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 59, 63, 67**.

---

## 3. Sections (DB-resolved from KS D 3502 / h_beam_sections)

| Property | Column **H-300×300** | Beam **H-400×200** |
|---|---|---|
| Section name (KS D 3502) | H-300×300×10×15 | H-400×200×8×13 |
| A (mm²) | 11,980 | 8,412 |
| Ix — strong axis (mm⁴) | 2.04 × 10⁸ | 2.37 × 10⁸ |
| Iy — weak axis (mm⁴) | 6.75 × 10⁷ | 1.74 × 10⁷ |
| J — torsion (mm⁴) | 7.65 × 10⁵ | 6.6 × 10⁵ |
| h overall depth (mm) | 300 | 400 |
| b flange width (mm) | 300 | 200 |
| tw web thickness (mm) | 10 | 8 |
| tf flange thickness (mm) | 15 | 13 |

All 57 columns use H-300×300. All 78 beams use H-400×200. No section overrides.

---

## 4. Material — SS275

| Property | Value |
|---|---|
| E (modulus of elasticity) | **205,000 MPa (205 GPa)** ← verified from runner |
| ν (Poisson ratio) | 0.3 |
| G (shear modulus) | E / (2(1 + ν)) = 78,846 MPa |
| fy (yield strength) | 275 MPa |
| ρ (density) | 7,850 kg/m³ |
| Damping | not used (linear elastic static) |

**E verification (2026-06-04)**: `tests/benchmark/case6_lshape.py` resolves SS275 via `get_material_from_db()` → `DEFAULT_MATERIALS["SS275"]` fallback (`mcp-server/core/simple_beam.py:148`). DB lookup returns no SS275 row in this environment, so the DEFAULT value E = 205,000 MPa is used. **Midas modeler must use E = 205,000 MPa** for SS275 to match the OpenSees runner. (KS standard is E ≈ 205 GPa for SS275; the cases.py-internal SECTION dict used 210 GPa, but that path is not exercised here.)

---

## 5. Local Axes (beta angle / vecxz)

Local-2 axis = strong bending axis (about Ix); local-3 = weak axis.

| Member group | vecxz vector (X, Y, Z global) | Strong axis aligned with |
|---|---|---|
| Columns (all 57) | (1, 0, 0) | Global X (Ix → bending about Y, Iy → about X) |
| X-direction beams | (0, 0, 1) | Vertical (strong axis resists gravity) |
| Y-direction beams | (0, 0, 1) | Vertical (strong axis resists gravity) |

**Midas equivalent (beta angle convention)**:
- Columns: β = 0° (default orientation, strong axis along global X)
- All beams: β = 0° (default — strong axis vertical)

Modeler must verify that Midas's resulting section orientation places the 400 mm beam depth vertically (Ix = 2.37 × 10⁸ mm⁴ resisting bending in the vertical plane) and the 300 × 300 column with its strong axis (Ix) oriented to resist bending about the global Y axis.

---

## 6. Rigid Diaphragm Configuration

**Direction**: normal to Z (horizontal slab plane), `perpDirn = 3` in OpenSees.

**State**: ON at every story.

**Slave node list per story** — exactly the nodes in §1.2 (excluding base). Do NOT artificially extend Zone B's diaphragm into 4F/5F. The 4F and 5F diaphragm covers only Zone A's 9 nodes; the 3F diaphragm extends across the full L-shape (13 nodes including Zone B's 4 grid nodes).

**Master node**: the OpenSees runner (`analyze_from_model` → `_build_v2_model`) auto-selects master as `snodes[len(snodes)//2]` of the sorted story-node list ([mcp-server/core/frame_3d.py:510](mcp-server/core/frame_3d.py#L510)). Step 2 will run the model and print the chosen master ID per story into the Midas handoff doc (Step 3). Midas modeler shall configure its rigid-floor constraint at the same coordinate. **Master-node choice does not change physical response** as long as the slave list (§1.2) and gravity/lateral application points (§7, §8) are consistent — both engines should converge regardless of master.

**Lateral force application point** (BINDING):
Lateral nodal forces (§7.3) are applied to the **physical story slave nodes**, not to the diaphragm master. Both engines must follow this rule — Midas modeler must enter joint loads on the actual perimeter/grid nodes (the slave list), NOT on the rigid-floor reference point.

---

## 7. Load Cases

Four independent static linear load cases, **no combinations** (each case run in isolation for clean per-case comparison).

### 7.1 DL — Dead Load

Pressure: **5.1 kN/m²** applied over the **existing slab footprint** of each floor:
- 1F slab: full L-shape area (Zone A 12 × 8 + Zone B 12 × 4 = 144 m²)
- 2F slab: same
- 3F slab: same
- 4F slab: Zone A only (12 × 8 = 96 m²)
- 5F slab: Zone A only (96 m²)

Distribution method: OpenSees runner converts pressure → beam line loads via two-way tributary-width estimate ([mcp-server/core/load_generator.py](mcp-server/core/load_generator.py) `floor_area` handler). The runner exports the resulting per-element line-load table (kN/m per beam element). That exported table is the single source of truth — Midas modeler types in each element's w_line directly (Midas auto-distribution OFF).

Direction: −Fz (downward), in global frame, applied as uniform beam load along the element.

### 7.2 LL — Live Load

Pressure: **2.5 kN/m²**, same footprint coverage and distribution as DL. Direction: −Fz.

### 7.3 EQX, EQY — Equivalent Lateral

**Total base shear**: V = 500 kN (applied independently in X and Y).

**Per-story force distribution** — linear by height (inverted triangle):

`Fi = V × hi / Σhj` where hi = z-coordinate of story i, summed over stories that receive lateral force (all 5 stories receive lateral; both zones are loaded at the floors where they exist).

Σhi = 3.5 + 7.0 + 10.5 + 14.0 + 17.5 = **52.5 m**

| Story | hi (m) | Fi (kN) total | # nodes | Per-node force (kN) |
|---|---|---|---|---|
| 1F | 3.5 | **33.333** | 13 | 2.5641 |
| 2F | 7.0 | **66.667** | 13 | 5.1282 |
| 3F | 10.5 | **100.000** | 13 | 7.6923 |
| 4F | 14.0 | **133.333** | 9 | 14.8148 |
| 5F | 17.5 | **166.667** | 9 | 18.5185 |
| Σ | — | **500.000** | — | — |

**EQX**: per-node force applied as +Fx in global frame on every node listed in §1.2 for that story (Zone A + Zone B nodes share the same per-node force at 1F-3F).

**EQY**: identical magnitudes, applied as +Fy.

Each per-node Fx (or Fy) is applied at the physical slave node, NOT at the master (§6 binding rule).

---

## 8. Sign & Unit Conventions

- All reported results: kN (forces), kN·m (moments), mm (displacements), radians (rotations).
- Downward = −Fz in global frame.
- Diaphragm rotation Rz: counter-clockwise positive when viewed from +Z.
- Sign of element forces: standard OpenSees `eleForce` (axial tension positive). Midas results to be re-signed as needed at JSON-fill time (handoff doc enumerates the convention).

---

## 9. Geometric Nonlinearity

**Linear elastic only** (linear geometric transformation, no PDelta, no Corotational) for Case 6 v1.

Rationale: L-shape decision-gate is about model-response agreement under linear static — Corotational/PDelta diagnostics are deferred to Step 5 ablation if Scenario C triggers.

OpenSees: `geomTransf('Linear', tag, *vecxz)` for all elements.
Midas: Linear Static analysis, no P-Delta secondary effect.

---

## 10. Cross-engine matching summary (one-page reference for both modelers)

| Item | Value / rule |
|---|---|
| Geometry | L-shape per §1; Zone A 12×8×5F, Zone B 12×4×3F, h_story=3.5 m |
| Node count | 70 (13 base + 13×3 + 9×2 floor) |
| Element count | 135 (57 columns + 78 beams) |
| Supports | All z=0 nodes fully FIXED |
| Column section | H-300×300×10×15 (SS275) |
| Beam section | H-400×200×8×13 (SS275) |
| Material | E=210 GPa, ν=0.3, G=80,769 MPa |
| Local axis | Columns vecxz=(1,0,0); beams vecxz=(0,0,1) |
| Diaphragm | Rigid floor, partial-floor at 4F/5F (Zone A nodes only) |
| Load 1 DL | 5.1 kN/m² → beam line load (table from runner) |
| Load 2 LL | 2.5 kN/m² → beam line load (table from runner) |
| Load 3 EQX | V=500 kN, inverted-triangle, per-node Fx on slaves |
| Load 4 EQY | V=500 kN, inverted-triangle, per-node Fy on slaves |
| Nonlinearity | Linear elastic only |
| Run mode | 4 independent linear static cases |

---

## 11. Out-of-spec for Case 6 v1 (deferred unless Scenario C)

- Combinations (1.2DL+1.6LL, etc.) — not run.
- Mass-source / response spectrum / eigenvalue — not run.
- P-Δ / Corotational — not run.
- Member releases — not used (all rigid connections).
- Wind / snow / temperature — not loaded.
- Member self-weight (handled by DL pressure, no separate density-based gravity).

---

**End of spec.** Step 2 (OpenSees runner) will be written against this spec exactly. The runner output produces: (a) the per-element beam line load table for DL & LL, (b) the per-node lateral force table for EQX & EQY, (c) the auto-picked diaphragm master ID per story, (d) the OpenSees-side metric JSON. Items (a)–(c) feed the Midas handoff doc (Step 3); item (d) goes into the comparison table (Step 4).
