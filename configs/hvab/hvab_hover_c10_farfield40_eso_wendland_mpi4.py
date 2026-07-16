"""HVAB hover c10 M0.65 — far-field D40, CASE 1': pure ALM, WENDLAND kernel.

beta-kernel D40 matrix member (patch_notes/alm_beta_kernel/01_design #14.5):
physics-identical to hvab_hover_c10_farfield40_eso_mpi4.py (CASE 1 4-rank)
except actuator_line.kernel = Wendland C2 — eps-equivalent support
R_s = sqrt(7.5) eps replaces the gaussian 3-eps cutoff automatically
(spreading + sampling; corrections OFF, so no deficit kernel in play).
Judged against the case-1 gaussian archive: peak-region (0.85-0.97R)
M2cn and CT (success criteria: 00_handoff #5.4).

Launch (runbook 18 §2):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \
        --config configs/hvab/hvab_hover_c10_farfield40_eso_wendland_mpi4.py \
        --steps 31425 --log-every 16 --cuda-aware 1 \
        --csv mpi4_case1w.csv
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
    run_tag="farfield40_eso_case1w_wendland_mpi4",
)
config["actuator_line"]["kernel"] = {"type": "wendland"}
