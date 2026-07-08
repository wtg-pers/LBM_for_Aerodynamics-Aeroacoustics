# Faithful Dağ relaxation + Kleine method audit (2026-07-03)

Methodology: implement each paper's method faithfully as a CONTROLLED baseline,
then layer improvements as documented, separable options. This note (a) fixes our
Dağ relaxation to the standard scheme, and (b) audits our Kleine vs the paper.

## Background — three methods (Kleine 2022 §1)
- **(A) Dağ & Sørensen 2020 / Meyer Forsting**: FULL inner iteration each timestep
  (steps ii-viii), relaxation for stability, iterate to convergence.
- **(B) Martínez-Tossas & Meneveau 2019**: ONE update per step using the previous
  step's Γ + a relaxation-weighted average of the correction velocity. "Equivalent
  to a first iteration"; "for steady simulations, converges to the steady solution."
- **(C) Kleine 2022**: non-iterative LINEAR SOLVE of the same fixed point.

All three converge to the SAME steady fixed point (the full self-consistent
correction) — the controlled baseline for hover.

## (a) Dağ relaxation — was BROKEN, now faithful
Our code did `u_n += relax·w_new` (single scaled pass). The standard scheme (B)
is a weighted average:
    w = relax·w_new + (1−relax)·w_prev
At steady state w_prev = w, so w = w_new = FULL correction (relax-independent).
Our `relax·w_new` alone converged to **relax·full = UNDER-applied** (relax leaked
into the physics → no variable control). This was the "arbitrary optimization".

FIX (`_compute_bem_forces` Dağ branch + `_dag_w_prev` state): weighted-average
relaxation. Verified: for fixed w_new, `relax·w_new+(1−relax)·w_prev` → w_new for
relax∈{0.2,0.5,0.9} (all reach 0.0186); old `relax·w_new` stuck at relax·0.0186.
Now relax only sets convergence speed/stability; the answer is the full correction.

(Method A = full inner iteration is available on request; for STEADY hover (B)
converges to the same answer far more cheaply, so it is the baseline.)

## (b) Kleine audit — our implementation vs Kleine (2022)
### Faithful (matches the paper)
- Non-iterative solve `[I − diag(b)A]ΔΓ = Γ†−Γⁿ⁻¹` (Eq 5.14-5.15). ✓
- Linearize from the previous-step Γ (step i / §5 "start from a previous time step"). ✓
- Free wake advected by the CFD velocity (step ii, free-vortex option). ✓
- Trailed vorticity now edge-based = Dağ Eq 17-18 / discretized lifting line (post-fix). ✓
- Forces from the CORRECTED velocities (paper step; caller re-queries CL/CD at α_corr). ✓

### Deviations (ours) — classify before trusting a comparison
| # | deviation | paper | impact |
|---|-----------|-------|--------|
| 1 | **Cl slope = central finite-diff (Δ=1°)** | Kleine §6.1: 3rd-order/PCHIP smooth slope | minor; matters near stall/kinks. documented in polar_slope.py |
| 2 | **Re/Mach frozen at †** | step v: interp at local Re each iter | minor linearization choice (Re varies slowly) |
| 3 | **wake Γ applied from current Γ (no u_mp/u_mc split)** | Eq 5.5-5.6 freezes prev-wake u_mp | **exact for steady hover** (ΔΓ→0); error only if unsteady |
| 4 | **rebuild_every > 1** (fast=5) | rebuild every step | APPROXIMATION (wake ~frozen N steps). rebuild_every=1 = exact/controlled |
| 5 | **safety-net → Dağ fallback** on stiff/non-finite solve (|w|>0.5·u_tan) | not in paper | can silently mask a stiff Kleine step as Dağ |
| 6 | n_w=50 ring truncation | finite wake (Dağ: 2 rev) | modeling choice, both truncate |

### Recommendation for a CONTROLLED Kleine baseline
- Use **rebuild_every=1** (exact) as the baseline; the requested rebuild_every=5
  "fast" is an APPROXIMATION to compare against it (not the baseline).
- Be aware the safety-net can convert a stiff Kleine step into a Dağ step — watch
  the fallback rate; for a clean comparison, log/monitor it.
- #1-3 are small, documented linearization choices; acceptable for hover but note
  them. None is an "arbitrary optimization" of the physics like the Dağ relax bug was.

## Not changed here
Method A (full inner iteration) not implemented (B suffices for steady). Kleine
deviations #1-3 left as-is (documented, minor). #4 exposed via config (=1 for control).
