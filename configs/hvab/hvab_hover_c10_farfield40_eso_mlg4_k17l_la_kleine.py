"""HVAB hover c10 M0.65 — FINAL formulation + Kleine smearing correction.

Identical to the mlg4_k17l_la baseline (K17L ILES + LineAverage 0.25c/N80
+ iso variable-eps spreading, pure ALM, KSAS, n64) with ONLY the Kleine
non-iterative smearing correction (straight wake) re-enabled.

Purpose (2026-07-23): kappa-deficit fix candidate. CP budget: archB-level
dFM(+0.015) needs CP +4.6e-5; kappa 1.03->1.10 alone gives exactly that
(+4.8e-5 CP_ind). This is the CLEAN re-test of the era-B archB idea — the
old retirement verdict was confounded by the dissipative formulation
(all-one+SGS+gaussian sampling); radial truncation stays OFF (retired).
Judge with CP decomposition (kappa target 1.10-1.13), not FM alone.
Watch _kleine_fallback_count in the log (unstable-solve -> Dag fallback).

4-GPU run (25 rev; la anchor converges by rev 16, last-5-rev averaging):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mlg4_k17l_la_kleine.py \\
        --steps 31425 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 5 \\
        --ckpt-every 31425 --csv mlg4_k17l_la_kleine.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    use_sgs=False,
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "target": "inviscid"},
    prandtl_loss=False,
    sampling={"mode": "lineavg", "profile": "constant",
              "n_points": 80, "rs_chord_factor": 0.25,
              "time_filter_deg": 6.0},
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_k17l_la_kleine",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
