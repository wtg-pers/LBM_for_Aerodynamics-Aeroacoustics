"""S4 analytic twin — bench_stl_sphere_hwbb with the ANALYTIC sphere.

Same body (R = 1.0 L0 lu at (16.5, 24, 24)), same CORRECTED physical-unit
BC velocities (see the unit-schema note in bench_stl_sphere_hwbb.py).
This is the reference run for the S4 relative comparison
    Cd(STL icosphere) vs Cd(analytic sphere)  <= 5 %
on identical grid/BC/IC. The legacy configs/hpc_bench/bench_sphere_hwbb
must NOT be used as the S4 reference: its lattice-unit BC values collapse
to u_bc ~ 2e-5 lu under the physical-unit schema (bit-anchor only).

Run (single GPU, standard path):
    python main.py --config configs/stl/bench_ana_sphere_hwbb.py --max-steps 6000
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "bench_stl_sphere_hwbb.py")
_spec = importlib.util.spec_from_file_location("bench_ana_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

config["internal_geometry"] = {
    "sphere": {"enabled": True, "center": (16.5, 24.0, 24.0),
               "radius": 1.0, "wall_bc": "hwbb"},
}

_folder = "results_bench_ana_sphere_hwbb"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
