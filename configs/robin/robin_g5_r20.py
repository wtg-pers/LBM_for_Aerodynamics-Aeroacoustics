"""ROBIN rotor-off grid campaign Phase 2b — 5-level ladder, r = 20.

User decision (0902, robin/15 s6b): test r20-5 BEFORE adopting r20-4 as
the reference grid. finest = R/320 (boom 16.0 cells) over the SAME finest
body box as robin_g4_r20 (GRID4_L3_BOX_R, now L4) with a new L3 cushion
(GRID5_L3_BOX_R) -> the g4/g5 pair isolates one halving of the finest dx.
Cost ~1.6x the r32-4 production compute (~37M nodes, working set ~26 GiB
-> DGX Spark single GPU ~11 h, or 2-rank z-slab). p_sample_h stays 1.5
(registered); the h*(R/320) calibration point comes from the offline
volume h-probe at readout (robin/15 s3c rig procedure).

Pre-registered readout: representability gate -> Cd_p/Cz + Cp(sknh) ->
h-probe (3rd point of the h*(dx) curve) -> recalibrated composite vs
g4-recalibrated and the r32-4 control.

Run (main dir; z-slab 2-rank also admissible, cut z~60):
    LBM_ESOTERIC=1 python main.py --config configs/robin/robin_g5_r20.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=20, tag="robin_g5_r20", max_steps=5600,
                  output_interval=280, num_levels=5, ladder_boxes=True)

if __name__ == "__main__":
    _m.report(config, "robin_g5_r20")
