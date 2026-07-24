"""HVAB hover c10 M0.65 — archB + Shen g0.3 mlg4, K17L collision A/B.

Identical to the workshop archB c10 point (all-one cumulant + dyn_smag,
gaussian sampling/spreading + radial truncation + Kleine straight + Shen
g0.3, KSAS, n64, 25 rev) with ONLY the collision switched to K17L
(omega_345 = 0, limiter 0.01). SGS stays ON.

Purpose (2026-07-24): measure the K17L x SGS-ON interaction and test the
compensating-balance prediction — expected FM +0.02~0.03 (off the EXP
curve upward), CT +1~3%, sharper wake cores with partial return of
over-descent. Companion evidence case for the workshop ("dissipation-
correction balance is understood").

4-GPU run:
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas_mlg4_shen_g030_k17l.py \\
        --steps 31425 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 5 \\
        --ckpt-every 31425 --csv archB_mlg4_shen_g030_k17l.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss={"enabled": True, "model": "shen", "g": 0.3,
                  "tip": True, "root": False, "eps_offset": False},
    radial_truncation=True,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_archB_ksas_mlg4_shen_g030_k17l",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
