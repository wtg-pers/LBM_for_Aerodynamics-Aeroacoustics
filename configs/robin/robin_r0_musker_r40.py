"""ROBIN rotor-off R0 at R = 40 L0 cells — resolution one-knob (patch robin/06).

Identical stack to robin_r0_musker (v1c, Musker h3, tau_model, overlap_cap,
crease_mode=noslip, Smagorinsky Cs 0.1, same domain/box fractions, same BCs);
ONLY the lattice resolution changes: R 32 -> 40 L0 cells (dx0 39.4 mm, dx3
4.92 mm, L = 640 fine cells; nose-tip -> station 1 = 16.5 fine cells,
afterbody lower-corner radius ~0.02R = 6.5 fine cells). Nodes ~91.5M
(x1.95), working set ~67 GB (x1.15 + 0.9 GiB -> ~78 GiB): DGX Spark unified
128 GB only — NOT a 24 GB card. Time: dt0 scales with dx0 -> 1122 L0 steps
per body flow-through; 11250 steps = 10 FT (R0 had 9000 = 10 FT); expect
~2.4x the R0 wall time (~16 h at 72 MLUPS).

Pre-registered readout (robin/06): same window rule (last >= 2.2 FT =
surface files step >= 8500: 8500/9000/.../11000/11249 = 2.45 FT), same tool/r_s. Discriminators vs R0(r=32):
  X1 nose offset  : st1-2 mean(sim-exp) -0.088/-0.048 -> shrinks by >= 50 %
                    => resolution; unchanged (|d| < 0.01) => p_state/model.
  X2 afterbody    : st11 phi156 -0.09 (exp -0.20) and st12 bottom -0.02
                    (exp -0.10) -> move toward exp by >= 0.03 AND a lifted
                    underside vortex pair appears in the L3 planes (05 sec.
                    2 method) => corner resolution; else => wall model.
  X3 D1 overall rms 0.045 -> report; pylon 8-11 (0.034) must not degrade.
  X0 gates as 03 sec. 1 (window Cd oscillation < 5 %, L/R < 0.02, no NaN).

Run (DGX Spark, single GPU, main dir):
    LBM_ESOTERIC=1 python main.py --config configs/robin/robin_r0_musker_r40.py --gpu 0
Readout:
    python -m src.utilities.robin_anchor --point 90 --config configs/robin/robin_r0_musker_r40.py \\
        --surface "results_robin_r0_musker_r40/vtk/surface_000{08500,09000,09500,10000,10500,11000,11249}.vtk" \\
        --plot robin_r40_cp.png --csv robin_r40_cp.csv
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=40, tag="robin_r0_musker_r40", max_steps=11250,
                  output_interval=500)
config["time"]["checkpoint_interval"] = 2500

if __name__ == "__main__":
    _m.report(config, "robin_r0_musker_r40")
