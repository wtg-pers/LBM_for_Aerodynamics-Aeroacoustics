"""NACA0012 a=10 Re 1e6, FULL 5.76-delta span + seed — 2-GPU z-slab MPI.

The patch-60 ORIGINAL span (Nz=16 = 121 L3 nodes = 5.76 delta) that the
24 GiB single-GPU ceiling forced down to 3.10 delta (60 sec. 3b), now
reachable on 2x24 GiB via the surfel z-slab MPI path (patch 64):
V1 residency bridge per slab (63), slab-filtered surfel (64 sec. 2,
ghost=4), tau_out margin exchange (64 sec. 11). Verdict registration and
discriminators are patch 60 sec. 5 UNCHANGED (P1 read with the |R|
repair of patch 61); the 3.10-delta run's verdict is patch 61 branch D.

Feasibility note (64 sec. 9-10): Nz=16 gives L0 z=16 — the wrap window
(own 8 + 2*ghost 8) EXACTLY fits at 2 ranks; Nz=14 does NOT (window
would exceed the axis). Do not launch with more than 2 ranks (L0 own
would drop below the window bound); do not lower ghost below 4 (the
surfel stencil chain consumes 4 cells/substep — 64 sec. 8).

Known gap at launch (62 sec. 7 / 64): the MPI path writes NO surface
VTK — P3 (Cf topology) is deferred to the surface-gather follow-up;
P0/P1 (L3 .vti), P2 (force history) and P4 (Cl) are all produced.

Run (cluster, ONE node with 2x24 GiB). `--mca pml ucx` is REQUIRED —
without it OpenMPI 5.x drops the UCX PML and the halo runs the ob1
slow path (~1.4 GB/s measured: halo_complete was 80% of a 3.1 s/step
run, 64 sec. 19c; the hpc_upgrade runbook 18 flag was lost in the
sec. 12 transcription):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \\
        --config configs/stl/naca0012_a10_surfel_re1e6_span16_seed.py \\
        --axis z --ghost 4 --cuda-aware 1 --gpu 0,1

Preflight gates on the setup header (patch 60 G4 updated): 4-level
surfel + tau-model ON + Force Level 3 + 52,130,349 total nodes + the
[mpi] surfel-slab lines showing symmetric taum wire sizes on both ranks.
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "re1e6_span_base", os.path.join(_here, "naca0012_a10_surfel_re1e6_span.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build_span(
    frac=0.16,                                   # Nz = 16 — the original span
    folder="results_naca0012_a10_surfel_re1e6_span16_seed")

# Patch-58 seed verbatim, except z = full span (periodic, no taper) —
# identical composition to the 3.10-delta seed arm (span_seed config).
_NZ = float(config["grid"]["Nz"])
_seed_58 = importlib.util.spec_from_file_location(
    "re1e6_seed_twin", os.path.join(_here, "naca0012_a10_surfel_re1e6_seed.py"))
_s = importlib.util.module_from_spec(_seed_58)
_seed_58.loader.exec_module(_s)

_pert = dict(_s.config["initial_perturbation"])
_pert["box_lu"] = list(_pert["box_lu"][:4]) + [0.0, _NZ]
config["initial_perturbation"] = _pert
