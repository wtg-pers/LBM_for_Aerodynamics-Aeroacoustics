"""HVAB hover c10 M0.65 — GRID A/B: the FINAL formulation on mlg5.

Identical formulation to the current baseline mlg4_k17l_la (K17L ILES +
coupled LineAverage 0.25c/N80/EMA6 + iso variable-eps spreading, pure ALM,
KSAS, n64, 30 rev) with ONLY the grid changed: preset farfield40 = mlg5,
ALM on the L4 slab (dx_fine = L0/16, tip chord 16.7 cells, core birth
~0.29 lu; L4->L3 interface at z/R -0.11).

Purpose (user 2026-07-23): isolate the grid-resolution contribution to the
FM overprediction (+0.055 at matched CT/sigma). kappa hypothesis predicts
mlg5's ~2x finer vortex core deposits more wake KE (~ Gamma^2 ln(1/r_c))
-> kappa up -> FM DOWN toward EXP. Prior hint at all-one+SGS formulation:
s1(mlg5) FM 0.761 vs mlg4 0.772 (-0.011).

Judge: CT / CP-decomposition kappa / FM vs mlg4_k17l_la; spanwise M2cn;
wake metrics (expect the 125-deg core jump to return - interface present).

4-GPU run (91.6M cells, ~4.7 GB/GPU at 4 ranks — can share GPUs with the
tip-loss sweep per current headroom):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mlg5_k17l_la.py \\
        --steps 37710 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 5 \\
        --ckpt-every 37710 --csv mlg5_k17l_la.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "lineavg", "profile": "constant",
              "n_points": 80, "rs_chord_factor": 0.25,
              "time_filter_deg": 6.0},
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=30,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg5_k17l_la",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
