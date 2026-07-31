"""STL grid-check smoke — bench_sphere_hwbb twin, analytic sphere -> STL.

The D=2 L0-lu sphere at (16.5, 24, 24) is replaced by an inscribed
icosphere STL (R=25 file units x scale 0.04 = R=1.0 L0 lu = 16 L4 cells,
s=4, chord error ~3e-4 lu). Everything else is the bench5 5-level
topology.

Dry-run target for to_claude/stl_grid_check.py (SimulationSetup built,
f never allocated). device_mode=cpu so it runs on the local machine.
NOT a physics case — bench legacy layout has L4 slab margin 0.5 lu =
0.25 D < the 0.5 D padding rule, so the setup padding WARNING on
x_low/x_high is EXPECTED; the C2F/F2C band violation must NOT fire.

Run:
    python -m to_claude.stl_grid_check --config configs/stl/stl_sphere_gridcheck.py
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here))
_base = os.path.join(_repo, "configs", "hpc_bench", "bench_sphere_hwbb.py")
_spec = importlib.util.spec_from_file_location("stl_gridcheck_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

config["simulation"]["device_mode"] = "cpu"

config["internal_geometry"] = {
    "stl": {
        "enabled": True,
        "file": os.path.join(_repo, "input_files", "geom",
                             "icosphere_r25_s4.stl"),
        "scale_to_lu": 1.0 / 25.0,        # R=25 file units -> R=1.0 L0 lu
        "center_lu": (16.5, 24.0, 24.0),  # bench sphere twin position
        "wall_bc": "hwbb",
    },
}

# MEM force needs the 'stl' obstacle meta wiring (track stage S3).
config["force_calculation"] = {"enabled": False}

_folder = "results_stl_gridcheck"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
    clear_previous=True,
)
