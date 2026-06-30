"""Smoke test: 500 steps to verify SGS+MLG+VTK paths work end-to-end."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cyl_Re3900_ibb_mlg3_3d_v2_smag import config

# Override for smoke test
config["time"] = {
    "max_steps": 500, "output_interval": 250,
    "logging_interval": 100, "checkpoint_interval": 1_000_000,
    "conservation_interval": 250,
}
config["force_calculation"]["start_step"] = 100
config["force_calculation"]["interval"] = 10
config["convergence"] = {"enabled": False}

_folder = "result_cyl_Re3900_smag_smoke"
config["output"] = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32",
            "variables": ["density", "velocity", "solid_mask", "nu_t"]},
    "checkpoint": {"enabled": False},
}
