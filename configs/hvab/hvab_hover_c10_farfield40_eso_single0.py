"""HVAB hover c10 M0.65 — far-field D40, single-GPU comparison twin (GPU 0).

Physics-identical to hvab_hover_c10_farfield40_eso.py (CASE 1: pure ALM,
isotropic Gaussian, N=64, corrections OFF, NASA OVERFLOW deck) with only
the run_tag changed so the result directory does not collide with the
single-GPU case-1 archive (comparison target).

Launch (patch_notes/hpc_upgrade/18_m5_cluster_runbook.md §2):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mpi4.py \
        --steps 1257 --log-every 16 --cuda-aware 1 \
        --csv mpi4_case1_rev1.csv
1 rev = 1257 coarse steps (25 rev = 31425). The MPI runner writes the
thrust CSV only (VTK/checkpoint assembly is a documented M5 follow-up).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None,
    prandtl_loss=False,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="nasa_overflow",
    run_tag="farfield40_eso_single0",
)
