"""HVAB hover c10 M0.65 — mlg4 + BGK collision + dyn_smag: collision A/B.

Counterpart to the cumulant relaxation sweep: tests whether the ~84%
non-SGS share of the tip-vortex peak decay (in-level half-life anchor:
cumulant+dyn_smag 222 deg, cumulant-only 263 deg) is cumulant-specific or
collision-generic (= pure resolution):
    BGK+SGS half-life ~ 260 deg  -> collision-generic (resolution floor)
    much longer/shorter          -> collision operator is a real lever
Requires the esoteric-BGK extension (body force + local-omega dyn_smag,
patch 21) — MPI runner path only. Everything else frozen at mlg4 (ALM on
L3, iso/iso gaussian, KSAS, n64, 20 rev, dyn_smag ON like the original
mlg4 so the comparison is BGK+SGS vs cumulant+SGS).

STABILITY RISK: BGK at tau ~= 0.5; dyn_smag nu_t stabilizes sheared
regions but quiescent regions run bare BGK. bench5_bgk_sgs smoke first;
if it diverges, this comparison needs a raised-nu operating point instead.

4-GPU run (65.1M cells):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mlg4_bgk.py \\
        --steps 25140 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 2 \\
        --ckpt-every 25140 --csv mlg4_bgk.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    eps_correction=None, prandtl_loss=False,
    sampling=None,
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_bgk",
)
# numerics.collision takes precedence over simulation.collision_model
config["numerics"]["collision"] = "bgk"
config["simulation"]["collision_model"] = "bgk"
