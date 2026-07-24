"""HVAB hover c10 M0.65 — archB + Shen g=0.3, mlg4 grid (speed variant).

Identical formulation to archB_ksas_shen_g030 (all-one cumulant + dyn_smag
SGS, iso gaussian sampling/spreading + Merabet radial truncation + Kleine
straight + Shen g=0.3, KSAS psu, n64, 25 rev); ONLY the grid changes to
mlg4 (ALM on L3, no L4 slab, ~2.4x cheaper per rev).

Purpose (2026-07-23): if mlg4 reproduces the mlg5 archB+shen point, run
the workshop point-hunt on mlg4. Judge against the KNOWN grid offset of
the all-one+SGS combo (mlg4 reads CT +1.4%, FM +0.011 vs mlg5: mlg4
0.00980/0.7721 vs s1 0.00966/0.7612) — a difference of that size is the
expected offset, not a formulation change.

4-GPU run:
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas_mlg4_shen_g030.py \\
        --steps 31425 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 5 \\
        --ckpt-every 31425 --csv archB_mlg4_shen_g030.csv
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
    run_tag="farfield40_eso_archB_ksas_mlg4_shen_g030",
)
