"""span16 + trip, wall-law sampling height h_law = 4 (patch 71 h-sweep).

One-knob clone of naca0012_a10_surfel_re1e6_span16_trip (h_law = 3,
the campaign default): everything else verbatim — same seed, trip
strip, span, grid. Pre-registered discriminator (patch 70 verdict):
the LE suction-peak deficit (~20 %, Cp_min -4.2 vs ref -5.3) is
physical; if Cp_min / Cd_p respond to h while the flat-plate region
(x/c 0.2-0.9 Cf, Cp) stays put, the curvature-to-h ratio is the
mechanism; no response -> LE resolution (L4 box / chord sweep).

Band (tau-model) geometry is facet-only (eta), independent of h.
Slab sample envelope: reach = max(h_law, sample_h) + 1 <= ghost 4.

Run (cluster, ONE node, --mca pml ucx REQUIRED, 64 sec. 19c):
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \\
        --config configs/stl/naca0012_a10_surfel_re1e6_span16_trip_h4.py \\
        --axis z --ghost 4 --cuda-aware 1 --gpu 0,1
"""
import importlib.util
import os

H_LAW = 4.0

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "span16_trip_base",
    os.path.join(_here, "naca0012_a10_surfel_re1e6_span16_trip.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = dict(_m.config)
config["internal_geometry"] = dict(config["internal_geometry"])
config["internal_geometry"]["stl"] = dict(config["internal_geometry"]["stl"])
config["internal_geometry"]["stl"]["surfel"] = dict(
    config["internal_geometry"]["stl"]["surfel"], h_law=H_LAW)

_folder = "results_naca0012_a10_surfel_re1e6_span16_trip_h4"
config["output"] = dict(config["output"],
                        output_dir=f"./{_folder}/vtk",
                        checkpoint_dir=f"./{_folder}/checkpoints",
                        csv_dir=f"./{_folder}/csv")
