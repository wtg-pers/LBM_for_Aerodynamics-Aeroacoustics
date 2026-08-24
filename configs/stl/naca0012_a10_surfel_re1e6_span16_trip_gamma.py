"""span16_trip + intermittency-weighted wall function (patch 80) — γ arm.

ONE knob over naca0012_a10_surfel_re1e6_span16_trip: per-facet gamma
blending tau_w = (1-g)·tau_lam + g·tau_Musker (src/boundary/surfel.py
build_facet_intermittency; kernel mirror in surfel_d3q27). Geometry-
fixed gamma — the forced/known-transition logic of patch 72 §2, with
the user-corrected physics of patch 71 §4:

  suction  x_tr = 0.02, width 0.02  — natural transition x/c 0.01-0.03
      at Re 1e6 (the trip at 0.05-0.15 is downstream of it): only a
      tiny LE strip is laminar-weighted; the side is effectively fully
      turbulent, so this is NOT an LE-deficit lever (that claim was
      withdrawn in 71 §4).
  pressure x_tr = 0.60, width 0.20  — favorable-gradient laminar run,
      transition x/c 0.5-0.7. THIS is gamma's real basis: the h sweep
      (71 §3) measured turbulent-level Cf on the laminar pressure side
      = Musker over-supply, ~4-5% of Cd.

Registered caveat (72 §2): tau_lam is the linear viscous law at the
UNCHANGED log-layer sample h_law = 3 (the h=1-2 laminar sample needs
the adaptive-h track, deferred to step ③ by the 71 §4 order). At that
h the linear law under-reads Blasius by ~2x — directionally right,
magnitude conservative. Falkner-Skan tau_lam is the registered
refinement.

Pre-registered discriminators (readout at 10k vs span16_trip):
  P1  pressure-side Cf (x/c 0.1-0.5) drops toward laminar level
      (order 0.5-1e-3 vs the measured turbulent ~2-3e-3).
  P2  Cd down ~4-5% (Cd_f-side), Cl unchanged within ±1%.
  P3  suction side x/c > 0.05 unchanged (g==1 there: same bits).
  P4  LE Cp_min unchanged (suction strip is 2% chord; if Cp_min moves
      > 0.05 the gamma strip is interacting with the trip — flag).

Run (cluster, ONE node with 2x24 GiB) — --mca pml ucx REQUIRED:
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \\
        --config configs/stl/naca0012_a10_surfel_re1e6_span16_trip_gamma.py \\
        --axis z --ghost 4 --cuda-aware 1 --gpu 0,1
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
_folder = "results_naca0012_a10_surfel_re1e6_span16_trip_gamma"
config["output"] = dict(config["output"],
                        output_dir=f"./{_folder}/vtk",
                        checkpoint_dir=f"./{_folder}/checkpoints",
                        csv_dir=f"./{_folder}/csv")

_geo = dict(config["internal_geometry"])
_stl = dict(_geo["stl"])
_stl["surfel"] = dict(_stl["surfel"], intermittency={
    "suction":  {"x_tr": 0.02, "width": 0.02},
    "pressure": {"x_tr": 0.60, "width": 0.20},
})
_geo["stl"] = _stl
config["internal_geometry"] = _geo
