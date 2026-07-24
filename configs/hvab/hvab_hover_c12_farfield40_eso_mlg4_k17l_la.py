"""HVAB hover c12 M0.65 — FINAL formulation, second collective point.

Identical formulation to the c10 mlg4_k17l_la anchor (K17L ILES +
LineAverage 0.25c/N80 + iso variable-eps spreading, pure ALM, KSAS, n64);
ONLY collective changes 10 -> 12 deg.

Purpose (2026-07-23): loading-independence test of the structural FM bias
+ first extra point of the workshop CT/sigma-FM curve. EXP c12 =
(CT 0.009952, CT/sigma 0.0963, FM 0.7297 — the FM peak). If the kappa
deficit is structural, dFM should stay ~+0.05 here too (judge at matched
CT/sigma, expected to land near the EXP c14 loading). No interaction with
the concurrent la_kleine run — pure parallel value.

4-GPU run (25 rev, last-5-rev averaging; can share GPUs with la_kleine):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c12_farfield40_eso_mlg4_k17l_la.py \\
        --steps 31425 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 5 \\
        --ckpt-every 31425 --csv mlg4_k17l_la_c12.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=12.0, mtip=0.65, preset="farfield40_mlg4",
    use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "lineavg", "profile": "constant",
              "n_points": 80, "rs_chord_factor": 0.25,
              "time_filter_deg": 6.0},
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_k17l_la_c12",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
