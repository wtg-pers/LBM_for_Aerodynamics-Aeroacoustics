"""HVAB hover c10 M0.65 — mlg4 + K17L: Geier-parametrized third-order rates.

Cumulant relaxation sweep, point 1 (the decisive one): omega_3/4/5 at the
Geier 2017 Part I fourth-order-accurate limit for omega_1 -> 2 (all -> 0,
the (omega_1-2) factor in eqs. 111-113), stabilized by the eq.(116)
third-order limiter with lambda = 0.01 (K17L, the 2021-TGV workhorse:
"steeper low-pass than WALE" = dissipation only at the highest wavenumbers).
A/B baseline = mlg4_nosgs (all-one, half-life 263 deg). SGS off in both so
the collision effect is isolated. omega_6-10 stay 1.0; the A/B fourth-order
equilibrium corrections of the full K17 are NOT implemented (out of scope —
this tests the dissipation lever, not formal fourth-order accuracy).

Judge: in-level half-life vs 263 deg / core radius / trajectory (loading-
causality predicts unchanged) / CT. Precedent: Geier 2017 II ran K17L on
NESTED grids (sphere drag crisis) — MLG compatibility expected.

4-GPU run (65.1M cells):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mlg4_k17l.py \
        --steps 25140 --log-every 64 --cuda-aware 1 --dist-init \
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 2 \
        --ckpt-every 25140 --csv mlg4_k17l.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling=None,
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_k17l",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
