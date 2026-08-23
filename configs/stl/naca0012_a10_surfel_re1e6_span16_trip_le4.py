"""span16 + trip + leading-edge L4 partial-body box (patch 74).

The LE suction-peak deficit (~20 %, patches 69-71) is h-invariant and
dp-clean — the resolution arm. This config adds a 5th level (dx/16 of
L0) over the leading edge only: x/c -0.04..+0.15, wrapping the LE and
both surfaces, cutting the wing at its downstream face (partial-body
surfel, patch 74: full-STL facet build, finest-wins ownership by
fine_region, C2F dead fill, march-axis auto-pick, dv_min 0.5 sliver
floor, tau_model forced OFF on the partial level only).

SINGLE GPU (the partial-body + MPI slab wiring is a registered
follow-up): span16 (52M) + LE L4 (~16M) ~ 14 GiB, fits 24 GiB.

Pre-registered discriminators (73 sec. 3): R1 Cp_min(p_state, x/c
0.005-0.01) -4.21 -> <= -4.8 confirms the resolution mechanism;
R2 Cd_p <= 0.015; R3 flat-plate x/c 0.2-0.9 within 10 %; R4 Cl 0.96-
0.98; R5 no response -> pressure-gradient wall function / STL LE
resolution next.

Run (cluster, ONE GPU):
    LBM_ESOTERIC=1 python main.py \
        --config configs/stl/naca0012_a10_surfel_re1e6_span16_trip_le4.py \
        --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "span16_trip_base",
    os.path.join(_here, "naca0012_a10_surfel_re1e6_span16_trip.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = dict(_m.config)
mlg = dict(config["mlg"])
levels = [dict(L) for L in mlg["levels"]]
levels.append({"region": {"x_min": 297, "x_max": 316,
                          "y_min": 303, "y_max": 315,
                          "z_min": 0, "z_max": 15}})
mlg["levels"] = levels
mlg["num_levels"] = 5
mlg["wall_coupling"] = {"mode": "exclude", "wall_margin": 1,
                        "apply_to": "c2f"}
config["mlg"] = mlg
config["internal_geometry"] = dict(config["internal_geometry"])
config["internal_geometry"]["stl"] = dict(config["internal_geometry"]["stl"])
config["internal_geometry"]["stl"]["surfel"] = dict(
    config["internal_geometry"]["stl"]["surfel"], partial_body=True)

_folder = "results_naca0012_a10_surfel_re1e6_span16_trip_le4"
config["output"] = dict(config["output"],
                        output_dir=f"./{_folder}/vtk",
                        checkpoint_dir=f"./{_folder}/checkpoints",
                        csv_dir=f"./{_folder}/csv")
