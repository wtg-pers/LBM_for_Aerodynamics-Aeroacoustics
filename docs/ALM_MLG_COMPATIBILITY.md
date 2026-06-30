# ALM + MLG Compatibility Review

**Date:** 2026-04-08
**Status:** Open — Requires design decision before implementation
**Reviewers:** LBM Development Team
**Related files:** `src/actuator/`, `src/grid/`, `src/solver/simulation.py`

---

## 1. Background

The solver currently supports two independent subsystems:

- **ALM (Actuator Line Model):** Computes aerodynamic forces on rotating blades
  using BEM theory and spreads them onto the LBM grid via Gaussian regularization.
  Applied during `Simulation.advance()` Step 2-4 (body force → Guo correction → collision).

- **MLG (Multi-Level Grid):** Nested grid refinement with 2:1 ratio per level.
  Uses convective (acoustic) scaling with nested time-stepping and
  f^eq/f^neq decomposition for inter-level coupling.

Both subsystems work correctly in isolation:
- ALM + single grid (BGK or Cumulant): **validated**
- MLG + obstacle (HalfwayBounceBack): **validated**
- **ALM + MLG: NOT validated — 5 compatibility issues identified**

This document details each issue with code-level analysis for team discussion.

---

## 2. Current Architecture

### 2.1 ALM Execution in Simulation.advance()

```
Simulation.advance():
    Step 1: rho, u = macroscopic.compute(f)
    Step 2: body_force = ALM.step(u, dt=1.0)        # BEM + Gaussian spreading
    Step 3: u += body_force / (2*rho)                # Guo velocity correction
    Step 4: collision.collide(f, f_post, rho, u, tau, body_force)
            → Cumulant: moment sign-flip + Guo source term S_i
            → BGK: Guo source term S_i
    Step 5: streaming.compute(f_post, f_new)
    Step 6: BC (domain + obstacle)
    Step 7: f, f_new = f_new, f                      # buffer swap
```

Source: `src/solver/simulation.py:166-235`

### 2.2 MLG Nested Time-Stepping

```
MultiLevelGrid.advance():
    save f_prev[0]
    L0.advance()                    # includes ALM if al_model is set
    _advance_fine(level_k=1):
        save f_prev[1]
        L1.advance()                # al_model=None (no ALM)
        C→F coupling (half-step)
        [recurse into L2 if exists]

        save f_prev[1]
        L1.advance()                # al_model=None (no ALM)
        C→F coupling (full-step)
        [recurse into L2 if exists]

        F→C coupling (L1→L0)        # overwrites L0 excised region
```

Source: `src/grid/multi_level_grid.py:124-241`

### 2.3 ALM Assignment in MLG Construction

```python
# src/solver/setup.py — _build_mlg_simulation()

# Level 0: ALM assigned
sim_0 = Simulation(..., al_model=self.al_model)

# Fine levels: ALM explicitly excluded
sim_k = Simulation(..., al_model=None)   # "ALM on Level 0 only"
```

Source: `src/solver/setup.py:886-975`

---

## 3. Identified Issues

### Issue 1: ALM Applied Only on Level 0

**Severity:** HIGH
**Files:** `src/solver/setup.py:975`

**Current behavior:**
Fine level Simulations are created with `al_model=None`. Only Level 0
computes and applies ALM body forces. If the rotor blades are physically
located within a fine-grid region, that region receives NO direct body force.

**Physical consequence:**
The fine grid resolves the flow field at higher resolution but is unaware
of the actuator force. The coarse-grid force is applied on L0, but
subsequently overwritten by F→C coupling (see Issue 2).

**What should happen:**
Actuator forces should be present on the level(s) where they are
physically relevant. Options:

| Approach | Description | Complexity |
|----------|-------------|------------|
| A. Multi-level ALM | Each level has its own ALM instance with resolution-appropriate kernel | High |
| B. Force upsampling | Compute ALM on L0, interpolate force field to fine levels before coupling | Medium |
| C. Finest-level ALM | ALM runs only on the finest level containing the rotor; propagate to coarser via F→C | Medium |

---

### Issue 2: F→C Coupling Overwrites ALM Body Force Effect

**Severity:** CRITICAL
**Files:** `src/grid/coupling.py:201-245`, `src/grid/multi_level_grid.py:239-241`

**Current behavior:**
After L0.advance() (which includes ALM), fine levels advance WITHOUT body
forces. Then F→C coupling replaces the distribution function in L0's excised
region with fine-grid data:

```python
# coupling.py — fine_to_coarse()
# Extracts fine f at coarse nodes, rescales f_neq, overwrites coarse f
f_coarse[..., excised] = f_eq + rescale(f_neq_fine)
```

**Physical consequence:**
The ALM body force was embedded in L0's collision step:
```
f*_L0 = f - omega*(f - f_eq) + S_i(F_ALM)
```
After streaming + BC, f_L0 contains the force effect. But F→C coupling
**completely replaces** f_L0 in the excised region with fine-grid data
that has **no ALM forcing**. The force effect is lost.

**Diagram:**
```
L0.advance()  →  f_L0 contains ALM force effect
    ↓
Fine levels advance (no ALM) → f_fine has NO force
    ↓
F→C: f_L0[excised] = reconstruct(f_fine)  →  ALM effect ERASED
```

**What should happen:**
The fine grid should also carry the force effect. Options:

1. Apply ALM on fine levels directly (solves Issue 1 simultaneously)
2. Transfer body force field during C→F coupling (add force as separate field)
3. Encode force in f_neq before coupling (complex, may violate conservation)

---

### Issue 3: Gaussian Kernel Width Fixed at Coarse Resolution

**Severity:** HIGH
**Files:** `src/actuator/blade.py:319-335`, `src/actuator/spreading.py`

**Current behavior:**
The Gaussian kernel width epsilon is computed once at initialization:

```python
# blade.py — set_lattice_spacing()
self.marker_epsilon = np.maximum(
    self.marker_chord / 4.0,     # chord-based minimum
    2.0 * dx                     # resolution-based minimum
)
```

where `dx` is the **Level 0 lattice spacing** (= 1.0 lu). This epsilon is
stored in physical/L0 units and never rescaled for fine levels.

**Physical consequence:**
The regularized force distribution has a physical width determined by epsilon.
On a fine grid, the same physical epsilon corresponds to MORE grid points:

```
L0: epsilon = 2.0 lu_0, kernel covers ~6 cells per dimension
L1: epsilon = 2.0 lu_0 = 4.0 lu_1, kernel covers ~12 cells per dimension
L2: epsilon = 2.0 lu_0 = 8.0 lu_2, kernel covers ~24 cells per dimension
```

This is physically correct IF the intent is to maintain the same physical
kernel width. However, a common practice in refined ALM is to **reduce**
the kernel width on finer grids to capture sharper force distributions:

```
epsilon_k = max(chord/4, 2.0 * dx_k)
```

**What should happen:**
The epsilon strategy needs to be explicitly chosen:

| Strategy | Formula | Effect |
|----------|---------|--------|
| Fixed physical width | epsilon = const | Same physical smoothing across levels |
| Resolution-adaptive | epsilon_k = max(c/4, 2*dx_k) | Sharper forces on finer grids |
| Chord-only | epsilon = c/4 | Independent of grid, physics-driven |

If ALM is moved to fine levels (Issue 1), the `set_lattice_spacing(dx)` call
must receive the fine level's dx, not L0's.

---

### Issue 4: Velocity Interpolation Uses Coarse-Level Data

**Severity:** HIGH
**Files:** `src/actuator/actuator_line.py:284-290`, `src/solver/simulation.py:264-269`

**Current behavior:**
The ALM interpolates velocity from the LBM grid to blade marker positions:

```python
# simulation.py — _compute_body_force()
return self.al_model.step(u, dt=1.0)  # u is Level 0's velocity

# actuator_line.py — step()
u_markers = interpolate_velocity_batch_gpu(
    u_field,        # ← L0 velocity (coarse)
    positions,      # ← blade markers in physical coords
    epsilon_all,    # ← kernel width (coarse)
)
```

The ALM always receives Level 0's velocity field, even when blade markers
are located within a fine-grid region that has a higher-resolution solution.

**Physical consequence:**
- Blade forces are computed from a coarse velocity field
- Fine-grid velocity gradients (important for local angle of attack) are NOT captured
- The BEM lookup (CL, CD vs alpha) uses an inaccurate local velocity → incorrect forces

**Velocity resolution comparison:**
```
Blade marker at position (100, 80, 80):
  L0 velocity:  averaged over 1.0³ lu³ cell   → misses boundary layer detail
  L1 velocity:  averaged over 0.5³ lu³ cell    → 8× finer sampling
  L2 velocity:  averaged over 0.25³ lu³ cell   → 64× finer sampling
```

**What should happen:**
Before ALM execution, assemble a "composite velocity" from the finest
available level at each spatial location. Or, if ALM runs on a fine level
(Issue 1 fix), it naturally uses that level's velocity.

---

### Issue 5: No Temporal Synchronization Between ALM and Fine Substeps

**Severity:** MEDIUM
**Files:** `src/grid/multi_level_grid.py:156-241`, `src/actuator/actuator_line.py:235-320`

**Current behavior:**
The nested time-stepping advances levels at different rates, but the
rotor only updates once per coarse step:

```
Coarse step (dt_0 = 1.0):
  L0.advance()  →  ALM.step(u, dt=1.0)     # rotor rotates by omega*1.0
  L1.advance()  →  no ALM                    # L1 first half-step
  L1.advance()  →  no ALM                    # L1 second half-step
  L2.advance()  →  no ALM (×4)              # L2 quarter-steps
```

**Physical consequence:**
- The rotor blade position is updated only once per coarse step
- Fine levels see a **frozen rotor** during their substeps
- At high rotation rates, this creates temporal discontinuity:

```
Example: omega = 0.1 rad/lt, dt_0 = 1.0
  Coarse step: blade rotates 0.1 rad (correct)
  Fine L1 half-step 1: blade at same position (should be +0.05 rad)
  Fine L1 half-step 2: blade at same position (should be +0.1 rad)
```

- Force field on fine grids is time-invariant within a coarse step
- This may cause spurious oscillations in force coefficients

**What should happen:**

| Approach | Description | Accuracy |
|----------|-------------|----------|
| A. Sub-step ALM | Execute ALM at each fine level's dt | Best, most complex |
| B. Interpolated position | Extrapolate blade position for fine substeps | Good, moderate complexity |
| C. Accept frozen | Document limitation, valid when omega*dt_fine << 1 | Simplest |

**Applicability criterion for approach C:**
```
omega * dt_finest << 1 radian
→ For Level k: omega * dt_0 / 2^k << 1
→ Acceptable when blade tip barely moves during finest substep
```

---

## 4. Interaction Matrix

How the issues compound when ALM and MLG are combined:

```
                      Issue 1        Issue 2         Issue 3       Issue 4       Issue 5
                    (no fine ALM)  (F→C overwrites)  (epsilon)   (velocity)    (temporal)
                    ─────────────  ────────────────  ──────────  ────────────  ──────────
Fine grid accuracy      ✗              ✗               ✗            ✗            ✗
Force conservation      -              ✗               -            -            -
BEM accuracy            -              -               -            ✗            ✗
Grid convergence        ✗              ✗               ✗            ✗            -
Time accuracy           -              -               -            -            ✗
```

**Key insight:** Issues 1 and 2 are interdependent. Fixing Issue 1 (ALM on
fine levels) automatically resolves Issue 2 (force preservation), and
partially resolves Issues 3 and 4.

---

## 5. Recommended Implementation Strategy

### Phase 1: Finest-Level ALM (resolves Issues 1, 2, 4)

Apply ALM on the **finest level containing the rotor**, not on L0.

```
Current:
  L0.advance() → ALM(L0.u) → force on L0 → overwritten by F→C

Proposed:
  L0.advance() → no ALM
  Lk.advance() → ALM(Lk.u) → force on Lk → preserved (finest level)
  F→C: fine data (with force) → overwrites coarse (correct)
```

**Implementation steps:**
1. In `_build_mlg_simulation()`, find the finest level whose domain
   contains the rotor swept volume
2. Assign `al_model` to that level instead of L0
3. Adjust `al_model.set_lattice_spacing(dx_k)` for the fine level
4. Velocity interpolation automatically uses Lk's velocity

**Advantages:**
- Force is computed at the highest available resolution
- No force loss through F→C coupling (force is on the finest level)
- Velocity interpolation uses the best available data
- Minimal code changes (reassign `al_model` to different level)

**Limitations:**
- ALM force does NOT directly appear on coarser levels (only via F→C)
- Coarse levels outside the fine region have no body force

### Phase 2: Resolution-Adaptive Kernel (resolves Issue 3)

After Phase 1 assigns ALM to a fine level:

```python
# blade.py — update set_lattice_spacing()
def set_lattice_spacing(self, dx: float) -> None:
    """dx: lattice spacing of the level where ALM operates"""
    self.marker_epsilon = np.maximum(
        self.marker_chord / 4.0,
        2.0 * dx   # ← now uses fine level's dx
    )
```

This is a one-line change once Phase 1 determines which level's dx to use.

### Phase 3: Temporal Synchronization (resolves Issue 5)

Two sub-options depending on accuracy requirements:

**Phase 3a (Simple):** Accept frozen rotor during substeps.
Document the validity criterion: `omega * dt_finest << 1 rad`.
For typical wind turbine cases, this is usually satisfied.

**Phase 3b (Advanced):** Sub-step ALM on the finest level.
Each fine substep calls `ALM.step(u, dt=dt_fine)` with fractional dt.
Requires the rotor to advance by `omega * dt_fine` per substep.

---

## 6. Validation Plan

After implementation, validate with:

1. **Uniform flow + ALM (no sphere):**
   - Compare single grid vs 2-level MLG thrust/torque
   - Expected: identical within interpolation error

2. **NTNU BT1 rotor (existing config):**
   - Compare C_T, C_P: single grid vs MLG with rotor in fine region
   - Check force history continuity

3. **Grid convergence:**
   - MLG 2-level vs 3-level: C_T should converge
   - Force should improve with finer grid at rotor location

---

## 7. References

- Guo, Zheng, Shi, "Discrete lattice effects on the forcing term in the
  lattice Boltzmann method", Phys. Rev. E 65, 046308, 2002
- Geier et al., "The cumulant lattice Boltzmann equation in three dimensions",
  Comp. Math. Appl. 70(4), 2015
- Lagrava Sandoval, "Revisiting grid refinement algorithms for the lattice
  Boltzmann method", PhD thesis, University of Geneva, 2012
- Watanabe et al., "Large-eddy simulation of wind turbine wake using
  lattice Boltzmann method with actuator line model", 2018

---

## 8. Open Questions for Discussion

1. **Should ALM operate on a single level or multiple levels simultaneously?**
   - Single finest level is simpler but creates a hard boundary
   - Multi-level requires force consistency across grid interfaces

2. **How should the Gaussian kernel width scale with grid level?**
   - Fixed physical width (conservative) vs adaptive (more accurate)?
   - Does chord-based epsilon make level-scaling irrelevant?

3. **Is temporal sub-stepping worth the complexity?**
   - For typical TSR and grid levels, is the frozen-rotor error negligible?
   - What rotation rate / finest dt combination makes it necessary?

4. **Should body force be a coupling variable?**
   - Currently only f is transferred between levels
   - Adding body force as a separate coupled field would enable coarse-level
     forcing even when ALM runs on fine level only
