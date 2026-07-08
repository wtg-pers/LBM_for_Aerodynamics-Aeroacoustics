# Kleine smearing correction — same edge-based bug fix (2026-07-03)

The Kleine non-iterative correction (`method="kleine"`) shared the SAME
marker-dΓ/dr trailed-vorticity operator as Dağ, so it had the same bug (tip &
root vortices dropped → weak). Fixed both the straight and the free-wake paths.

## What was wrong
`smearing_correction.influence_matrix` built the influence `A = −(1/4π)·Δr·(Kmat @ G)`
with `G = _gradient_matrix` (marker dΓ/dr). The free wake did `A = B @ (Δr·G)` —
same `G`, filaments shed from the MARKERS. Both dropped the tip/root edge vortices.

## Fix
1. **`edge_operator(r, eps, dr)`** (new) → `(E, r_edge, eps_edge)`. `E` (N+1, N)
   maps Γ → the N+1 edge circulations `Γw(j)=Γ(j)−Γ(j−1)` with Γ padded 0
   (tip vortex −Γ_tip, root vortex; Σ Γw = 0).
2. **`influence_matrix`** → `A = +(1/4π)·Kmat_edge @ E` (edge-based, +sign).
   `_gradient_matrix` kept but deprecated.
3. **Free wake** (`actuator_line.py`): shed from the panel EDGES
   (`_edge_positions_3d`, `_shed_edge_idx`), `A = B_edge @ E`. `wake_markers="tip"`
   now sheds ONLY the tip closure edge (the tip vortex).

## Validation (test_kleine_freewake_edge.py, and inline)
- `influence_matrix @ Γ` == `_viscous_core_correction` (edge Dağ): max diff 1e-17. ✓
- Σ(E @ Γ) = 0 (circulation conserved); tip edge Γw = −Γ_tip exactly. ✓
- **Free-wake straight limit: `B_edge @ E` == `influence_matrix`, rel err 4e-6.** ✓
  (the free wake just curves those filaments; the operator is correct.)
- `_edge_positions_3d`: root/tip closure edges at r∓Δr/2. ✓
- `_shed_edge_idx`: "all"→None, "tip"→[N]. ✓
- Kleine solve (real run scale): self-consistent to ~1.5% (linearization error),
  **tip Δα −2.8° (de-loads)**, |w|=0.005 (< safety net). ✓
- **ρ(diag(b)·A) = 7.4 > 1**: the naive within-step iteration diverges — this is
  exactly the stiff-coupling regime Kleine's linear solve exists for. The old
  (buggy, weak) A had small ρ → "stable" but WRONG. The strong tip coupling is
  physical; the non-iterative solve handles it, safety net (|w|>0.5·u_tan →
  Dağ fallback) covers any stiff step (and the fallback is now also edge-fixed).

## Kleine vs Dağ (why compare)
Dağ single-pass uses the over-loaded Γ → overshoots (needs relax). Kleine solves
the self-consistent Γ in one linear system → tip w_corr smaller (~0.005 vs the
single-pass 0.019) and NO relax. So Kleine should give a cleaner tip de-load.

## Not migrated
The old marker-based free-wake dev scripts (patch_notes/alm_marker_distribution/
test_q2_tipwake.py) exercised the pre-fix shedding and are superseded.

## Re-run configs (configs/0703_dag_edge_fix/)
- `kleine_free_fast_endpoint.py`      — free wake, rebuild_every=5, n_w=50, all edges
- `kleine_free_tip_nw2_endpoint.py`   — free wake, tip edge only, n_w=2, rebuild_every=1
(gauss, endpoint, prandtl off, 25rev, NASA deck, light — compare vs the dagfix cases.)
