"""3-tier smoke — PURE FLOW with uniform inflow (no ALM, no body).

bench5 topology, U=0.05 lu inflow along +x (eq faces + sponge u_inf), so the
flow-tier progress line (rho_mean / u_max) shows real dynamics and the
nonzero-velocity eq/sponge esoteric BC paths get exercised. NOT a physics
case — code-path smoke for the unified runner.

    LBM_ESOTERIC=1 python main_mpi.py --config configs/hpc_bench/bench_flow_uniform.py \
        --steps 32 --log-every 8 --dist-init --devices 0
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config

U_LU = 0.05
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench_flow_uniform",
)
config["actuator_line"]["enabled"] = False
for face, bc in config["boundaries"].items():
    if bc["method"] == "eq":
        bc["velocity"] = [U_LU, 0.0, 0.0]           # [lu] — kernel-direct
    elif bc["method"] == "sponge":
        bc["velocity"] = [U_LU, 0.0, 0.0]
config["physics"]["initial_flow_velocity"] = [U_LU, 0.0, 0.0]   # [lu]
