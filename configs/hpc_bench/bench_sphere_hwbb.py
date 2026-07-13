"""3-tier smoke — SOLID BODY: sphere (HWBB) in uniform inflow, 5-level MLG.

bench5 topology + D=2 L0-lu sphere at the rotor-hub position (16.5, 24, 24),
fully inside the L4 slab (x[15,18], D=32 fine cells across). U=0.05 lu
inflow. Exercises: esoteric SOLID skip (implicit HWBB), solid nodes across
the decomposition cut, MLG coupling over a solid interior. NOT a physics
case (L4-edge margin 0.25D < the 0.5D padding rule; Cd would be shifted) —
code-path smoke: the acceptance is bit-vs-production, not Cd.

REPLICATED INIT ONLY (--dist-init fail-fasts on obstacles):
    LBM_ESOTERIC=1 python main_mpi.py --config configs/hpc_bench/bench_sphere_hwbb.py \
        --steps 32 --log-every 8 --devices 0
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config

U_LU = 0.05
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench_sphere_hwbb",
)
config["actuator_line"]["enabled"] = False
for face, bc in config["boundaries"].items():
    bc["velocity"] = [U_LU, 0.0, 0.0]
config["physics"]["initial_flow_velocity"] = [U_LU, 0.0, 0.0]
config["internal_geometry"] = {
    "sphere": {"enabled": True, "center": (16.5, 24.0, 24.0),
               "radius": 1.0, "wall_bc": "hwbb"},
}
