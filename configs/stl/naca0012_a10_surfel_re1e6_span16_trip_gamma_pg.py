"""span16_trip_gamma + pressure-gradient wall function (patch 81) — PG arm.

ONE knob over naca0012_a10_surfel_re1e6_span16_trip_gamma: per-facet
tau_turb = tau_Musker * R with R = tau_TBLE(beta)/tau_TBLE(0)
(Wang & Moin 2002 simplified TBLE, linear stress tau(y) = tau_w +
beta*y; beta = dp/ds from two tangential rho probes at h_law, ds = 2).
Ratio form keeps the Musker family baseline: R == 1 exactly at beta = 0.
Composition with patch 80: tau = (1-g) tau_lam + g (tau_Musker * R).

A-priori map from the trip baseline (81 sec. 0): LE band R 0.30-0.65
(the action zone — equilibrium Musker over-supplies tau right after the
suction peak, p+*y+ up to 10), mid-chord R 0.86-0.93, aft R 0.87-0.98,
pre-peak favorable bin R 1.41. No tau-collapse bins predicted.

Pre-registered discriminators (readout vs span16_trip_gamma, patch 81
sec. 4): Q1 LE Cp_min -4.25 -> deepen >= 0.3 (toward -5.3) / Q2
mid-chord Cf down <= 30%, mid Cp moved <= 2% / Q3 Cd_p 0.034 down,
Cl toward 0.98 / Q4 pressure-side laminar zone untouched / Q5
tau_w-clamp (incipient-separation) facet share < 2%.
Kill/escalate (sec. 5): Q1 overshoot (Cp_min < -6) or Q2 Cp > 2%
-> arm B (convection-balanced TBLE).

Run (cluster, ONE node with 2x24 GiB) — --mca pml ucx REQUIRED:
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \\
        --config configs/stl/naca0012_a10_surfel_re1e6_span16_trip_gamma_pg.py \\
        --axis z --ghost 4 --cuda-aware 1 --gpu 0,1
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "span16_trip_gamma_base",
    os.path.join(_here, "naca0012_a10_surfel_re1e6_span16_trip_gamma.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = dict(_m.config)
_folder = "results_naca0012_a10_surfel_re1e6_span16_trip_gamma_pg"
config["output"] = dict(config["output"],
                        output_dir=f"./{_folder}/vtk",
                        checkpoint_dir=f"./{_folder}/checkpoints",
                        csv_dir=f"./{_folder}/csv")

_geo = dict(config["internal_geometry"])
_stl = dict(_geo["stl"])
_stl["surfel"] = dict(_stl["surfel"], pressure_gradient={"ds": 2.0})
_geo["stl"] = _stl
config["internal_geometry"] = _geo
