"""NACA0012 a=10 Re 1e6, WIDE SPAN + solenoidal IC seed — the sharp arm.

Cross of the two discriminators: the span of
naca0012_a10_surfel_re1e6_span.py (5.76 delta, L3 nodes) with the
patch-58 seed that ran at 1.19 delta. Registered in
patch_notes/surfel/60 — read that first; sec. 2 carries a defect this
arm is the first to activate (per-level spanwise wrap periods differ),
and sec. 9 is a user decision that gates the run.
It is the direct continuation of the 58/59 experiment
because the ONLY thing that changes versus that arm is the span — seed
spectrum, amplitude, box (x/y), mode count and RNG seed are all
byte-identical to naca0012_a10_surfel_re1e6_seed.py.

Why this and not the unseeded arm: patch 59 established that the IC seed
is one-shot (convective washout in 485 L0 steps) and concluded "an IC
seed is not the answer at any span". That is correct about SUSTAINING
turbulence, but the seed still has the job nothing else does here — it
BREAKS THE SPANWISE SYMMETRY. The solver enforces the z-invariant prism
by construction (mask symmetrized to the mid slice; ibb q broadcast), so
an unseeded run's only 3D content is the un-symmetrized surfel cut-cell
dV wobble at the Fz/Fy ~ 1.3e-3 level. Seeded, the verdict is clean
either way:
  (1) 3D content survives the 485-step washout and the separated shear
      layer regenerates it -> the span WAS the constraint; separation
      should shrink -> resume the Re ladder (3e6 / 6e6);
  (2) it washes out again at 6.1 delta -> the span is exonerated too, and
      the residual constraint is the supply self-reference (sigma ~ local
      tau_w, patch 53) -> non-local supply.

Two knobs move with the span, both forced by geometry rather than chosen:
  - box_lu z = the full new span (the envelope has no z taper; z is
    periodic).
  - k_z quantization: initializer defaults span_z_lu = Nz, so dkz =
    2*pi/16 instead of 2*pi/4. The isotropic mode directions now round
    onto low harmonics as well, i.e. the wider span AUTOMATICALLY admits
    the delta-scale and longer spanwise wavelengths that 1.19 delta could
    not represent (measured: max lambda_z 4 -> 16 L0 lu, modes with
    lambda_z >= delta 30/64 -> 34/64). The drawn spectrum (0.5-2.0 lu) is
    deliberately left alone so the arm stays a pure span comparison.

Run (cluster, single GPU — std path, NO LBM_ESOTERIC):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py \\
        --config configs/stl/naca0012_a10_surfel_re1e6_span_seed.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "re1e6_span_base", os.path.join(_here, "naca0012_a10_surfel_re1e6_span.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.config

# Patch-58 seed verbatim, except z = full span (periodic, no taper).
_NZ = float(config["grid"]["Nz"])
_seed_58 = importlib.util.spec_from_file_location(
    "re1e6_seed_twin", os.path.join(_here, "naca0012_a10_surfel_re1e6_seed.py"))
_s = importlib.util.module_from_spec(_seed_58)
_seed_58.loader.exec_module(_s)

_pert = dict(_s.config["initial_perturbation"])
_pert["box_lu"] = list(_pert["box_lu"][:4]) + [0.0, _NZ]
config["initial_perturbation"] = _pert

_folder = "results_naca0012_a10_surfel_re1e6_span_seed"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
