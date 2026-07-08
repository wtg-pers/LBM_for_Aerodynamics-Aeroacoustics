# Dağ viscous-core correction — implementation bug fix (edge-based, 2026-07-03)

## Symptom
The Dağ ε-smearing correction did almost nothing on the light grid: Task-2
`dag` gave CT/σ 0.1017 vs pureALM 0.1012 (≈ identical), and the tip over-loading
(M²Cn ~2.7× measured) was untouched. Marker distribution (uniform/cosine/
endpoint) also failed to move the tip → the tip over-loading is the induced
deficit (F1), and the correction meant to fix it was inert.

## Root cause = two implementation deviations from Dağ & Sørensen (2020)
Paper Eq. 17-18 (verified against the PDF, page 9/15):

    w_corr(i) = +(1/4π) Σ_{j=1}^{N+1} Γw(j)/d_ij · exp(−(d_ij/ε)²)     (17)
    Γw(j) = Γ(j) − Γ(j−1)      d_ij = r_i − r_edge_j                   (18,20)
    "For the tip vortices, the magnitudes are equal to the bound
     circulations closest to the tips."

Trailed vortices live on the **N+1 panel EDGES**; padding Γ=0 outside the blade
makes the outermost edge shed the **tip vortex Γw = −Γ_tip** and the innermost
the root vortex, so **Σ Γw = 0** (circulation conserved).

Our old `_viscous_core_correction`:
1. **Used dΓ/dr at the N markers** (`np.gradient`) instead of edge Γw. `np.gradient`
   at the last marker is the interior slope — it **never sees the Γ→0 drop past
   the tip**, so the **tip & root vortices were silently dropped**. Net trailed
   circulation Σ(dΓ/dr)·dr = +0.14 ≠ 0 (unphysical).
2. **Sign −(1/4π)** instead of +(1/4π).

Consequence: tip w_corr = +0.0006 vs the ~0.005 needed to remove the +3° tip
induced deficit → **10-30× too weak** → correction inert.

### Numerical audit (patch_notes/.../dag_audit.py, on HVAB run Γ)
| r/R | Γ | OLD (marker,−) | PAPER (edge,+) | needed |
|-----|-----|------|------|------|
| 0.95 | 0.43 | −0.0024 | +0.0052 | ~0.0053 |
| 0.99 | 0.31 | +0.0006 | **+0.0186** | (tip, most) |

Σ Γw: paper (edges) = 0 ✓ ; old (markers) = +0.14 ✗.

### Corollary — the earlier "option 3 closure gives upwash → shelve" was a sign artifact
Option 3 correctly ADDED the tip node but kept the −(1/4π) sign → flipped the
(now-present) tip vortex to upwash. Closure was never the problem; our sign was.

## Fix
`_viscous_core_correction` reimplemented edge-based (Eq. 17-18, +1/4π). N markers
→ N+1 edges (root, interior midpoints, tip); Γw = diff([0, Γ, 0]); +1/(4π).
target="opt" and scalar/array dr preserved. The old marker/endpoint_closure
branches are removed (the edge form always closes the ends).

### Validation (test_dag_edge_fix.py, calls the real edited method)
- Σ Γw = −4e-18 ≈ 0 (circulation conserved). ✓
- smooth synthetic Γ: correction mostly positive (downwash), sign OK. ✓
- HVAB run Γ: **tip w_corr +0.0186 (was +0.0006) = 31× stronger, correct (downwash) sign.** ✓
- scalar dr == array dr (uniform); target="opt" runs & is weaker. ✓

### Observation — mid-tip structure (not pure noise)
On the run Γ the correction wiggles at r/R 0.90-0.97 (±0.003). This tracks REAL
blade features: taper start (r/R 0.950, chord 10.9→9.7 lu) and airfoil transition
(r/R 0.975, RC6-08→RC6-08T, chord 9.7→8.2) shed real trailed vorticity. The
dominant tip correction (+0.019) is robust to light Γ-smoothing (+0.0183). Not
smoothed in the shipped fix (faithful to the paper); smoothing is a fallback if a
run shows marker-to-marker α noise.

## relax
The single-pass tip w_corr (~0.019) uses the over-loaded Γ and would swing φ from
0.6°→11° in one step (~3× the needed). LBM feedback self-limits (α↓→Γ↓→tip
vortex↓→w_corr↓), but **relax=0.5** under-relaxation damps the transient. Tune up
(→1.0) if under-corrected, down if noisy. Exposed in the config.

## NOT yet fixed — Kleine path
`smearing_correction.influence_matrix` (used by method="kleine") has the SAME
marker-dΓ/dr basis and needs the same edge treatment. Deferred; these runs use
method="dag".

## Re-run (configs/0703_dag_edge_fix/)
gauss sampler, dagfix (relax 0.5), 3 distributions:
- `dagfix_uniform_gauss.py`  — CONTROL vs old 260630/dag_csv (isolate code-fix)
- `dagfix_endpoint_gauss.py` — endpoint (tip marker at r/R 1.0)
- `dagfix_cosine_gauss.py`   — cosine(both)
Expected if the fix works: tip M²Cn drops from ~0.39 toward the measured ~0.15.
