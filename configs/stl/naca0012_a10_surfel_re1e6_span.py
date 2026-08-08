"""NACA0012 a=10 Re 1e6, WIDE SPAN — span-widening arm (patch 59 sec. 3).

The registered experiment: identical to naca0012_a10_surfel_re1e6 (full
stack, unseeded) except the slab thickness. Patch 59 closed the seed arm
on branch (2) — an IC seed is washed out in 485 L0 steps and the BL never
regenerates it — and the arithmetic that explained the failure was the
SPAN, while a resolved-LES attached BL needs >= 2-3 delta to hold
outer-layer eddies that are not correlated with their own periodic image.

  nz_frac  Nz [L0]  L3 span [nodes]  span/delta   total nodes
    0.04      4           25            1.19        11.29 M  <- 51/53/57/59
    0.09      9           65            3.10        28.30 M  <- THIS
    0.16     16          121            5.76        52.13 M  <- registered
                                                                first, NOT
                                                                RUNNABLE
  (delta = 21 fine cells, patch 57.)

★ 0.16 -> 0.09 (patch 60 sec. 3b/9b): the 0.16 grid needs a 35.7 GiB
working set and does not fit a 24 GiB card — it OOMs on the cluster and,
under WSL2's WDDM paging, merely collapses to 0.55 MLUPS locally instead
of erroring. Measured model over four grids (0.5% accurate):

    working set = nodes x 736 B ;  VRAM ~ working set x 1.10-1.18 + 0.9 GiB

Only 238 of those 736 B/node are accounted for (f+f_new 216, rho/u/nu_t
20, masks ~2); the unexplained ~470 B/node is what forced the span down
and is registered as a separate track. The physics requirement on record
is ">= 3 delta" (patch 59 sec. 3), not "0.16c" — 0.09 (3.10 delta) is the
largest span meeting it on 24 GiB. Margin over the floor is 0.03 delta.

REQUIRES a >= 24 GiB GPU with nothing else resident (pool reserves
23.6 GiB = 96%). Below that, patch 60 sec. 9b halts the arm rather than
shrinking further — a smaller span falls back to the regime patch 59
already rejected.

★ The span is quoted from the L3 NODE count, not Nz*8: OverlapRegion
builds fine levels node-based (fine_shape = extent*ratio + 1), so a
z-flush region spans (Nz-1)*2^k + 1 nodes and each level wraps at its own
period (Nz=16: L0 16.000 / L1 15.500 / L2 15.250 / L3 15.125 L0 lu).
Patch 59's "32 cells = 1.52 delta" is therefore 25 nodes = 1.19 delta,
and this arm gives 5.76 delta rather than 6.10. Invisible while the
solution is z-invariant, which is why 51-59 never saw it — see
patch_notes/surfel/60 sec. 2 for the registered handling (read P1 inside
L3 only; no cross-level spanwise phase comparison).

0.16 rather than the recommended floor 0.15c: it is the base builder's
own default slab (the validated 6e6 production geometry), it keeps Nz
even for the mid-slice symmetrization and the 2^k refinement, and it
costs 7% over 0.15c.

Registered verdict (patch 59 sec. 3):
  (1) spontaneous 3D turbulence + separation shrinks -> the span was the
      only remaining constraint -> resume the Re ladder (3e6 / 6e6);
  (2) statistically unchanged vs patch 57 -> the residual constraint
      narrows to the supply self-reference (sigma ~ local tau_w, patch
      53) -> non-local supply becomes the research item.
Read: force history (csv, interval 20) unsteadiness + broadband share,
surface Cp/Cf separation topology vs 57, and the spanwise two-point
correlation / energy of the L3 volume snapshots (the new measurement —
1.52 delta could not decorrelate by construction).

CAVEAT (deviation from the registered null hypothesis, not from the
config): this arm is unseeded, but the solver enforces the z-invariant
prism contract BY CONSTRUCTION (mask symmetrized to the mid slice,
geometry_manager.py:384; ibb q broadcast, setup.py:884). The only
3D content available to an unseeded run is the surfel cut-cell dV /
facet-CSR wobble, which is NOT z-symmetrized — deterministic,
tessellation-periodic, and at the level of the measured Fz/Fy ~ 1.3e-3.
A null result therefore cannot separate "span is not the constraint"
from "no 3D content was ever present". The seeded twin
(naca0012_a10_surfel_re1e6_span_seed.py) plants the content and is the
sharper discriminator; run it if only one run fits.

Cost (c=100, nz_frac 0.09) — all measured, not extrapolated:
28.30 M nodes, 161,974,142 cell-updates per coarse step, f + f_new
5,830 MiB, working set 19,967 MiB, pool reserve 23,583 MiB, 153.6 MLUPS
= 1.05 s/step -> 2.9 h for 10k steps. Checkpoints every 2500 steps allow
a restart-extension if the verdict window is ambiguous.
Throughput is the memory gate in disguise (patch 60 G6): a run reporting
much under 100 MLUPS is paging, not computing — stop and re-check VRAM.

Preflight first: configs/stl/naca0012_a10_surfel_re1e6_span_smoke.py
(same span fraction and full stack at c=50, ~7 M cells, 300 steps) —
patch 50 sec. 4e cost a cluster run to a sliver-dV divergence that only
a same-geometry preflight could have caught.

Run (cluster, single GPU — std path, NO LBM_ESOTERIC):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py --config configs/stl/naca0012_a10_surfel_re1e6_span.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The Re-1e6 twin (patch 57/59 arm) and, through it, the base builder.
_twin = _load("re1e6_twin", "naca0012_a10_surfel_re1e6.py")
_base = _twin._m

SPAN_FRAC = 0.09          # Nz = 9 L0 cells -> L3 span 65 nodes = 3.10 delta
                          # (patch 60 sec. 9b; 0.16 = 5.76 delta does not fit)


def build_span(c=100, steps=10000, folder=None, frac=None):
    """Wide-span twin of naca0012_a10_surfel_re1e6 at chord = c L0 cells.

    Every non-span knob is COPIED from the twin's own config rather than
    restated, so the two configs cannot drift apart silently.

    frac overrides SPAN_FRAC. The reduced-chord preflight needs it: Nz is
    round(frac*c), so the production 0.09 lands on Nz=4 at c=50 (round-half
    -to-even) — i.e. the OLD 57/59 slab, which previews nothing. The
    preflight passes frac so it keeps the production SLAB THICKNESS in L0
    lu and only halves the chord resolution.
    """
    cfg = _base._build(c, wall_bc="surfel",
                       nz_frac=SPAN_FRAC if frac is None else frac)

    # twin parity: nu (Re knob, patch 54 convention), SGS, surfel block.
    # NOT force_calculation.reference — its span_length must follow the
    # new Nz so Cl/Cd stay per-unit-span comparable with patch 57.
    cfg["physics"]["nu"] = _twin.config["physics"]["nu"]
    cfg["sgs"] = dict(_twin.config["sgs"])
    cfg["internal_geometry"]["stl"]["surfel"] = dict(
        _twin.config["internal_geometry"]["stl"]["surfel"])

    cfg["time"] = dict(cfg["time"], max_steps=int(steps))

    folder = folder or "results_naca0012_a10_surfel_re1e6_span"
    cfg["output"] = dict(
        cfg["output"],
        output_dir=f"./{folder}/vtk",
        checkpoint_dir=f"./{folder}/checkpoints",
        csv_dir=f"./{folder}/csv",
    )
    return cfg


config = build_span()
