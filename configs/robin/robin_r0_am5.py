"""ROBIN rotor-off alpha = -5 deg — TM-80051 Run 12 point 89 (patch robin/07).

ONE knob over robin_r0_musker (the R0 reference stack, R = 32): the body is
pitched by rotation_deg = (0, -5, 0) about its bbox centre (nose UP for
positive alpha; the build report prints nose/tail z as the check), the
inflow stays along +x. The L1-L3 boxes are rebuilt from the rotated bbox
with the alpha = 0 pads (_robin_base.build). Anchor: point 89 = alpha_F
-5, beta 0, 81.7 kt (Table IV); readout `--point 89`.

    LBM_ESOTERIC=1 python main.py --config configs/robin/robin_r0_am5.py --gpu 0
    python -m src.utilities.robin_anchor --point 89 --config configs/robin/robin_r0_am5.py \\
        --surface "results_robin_r0_am5/vtk/surface_0000[6-8]*.vtk" --plot robin_r0_am5_cp.png --csv robin_r0_am5_cp.csv
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

ALPHA_DEG = -5.0
config = _m.build(r_lu0=32, tag="robin_r0_am5", max_steps=9000, output_interval=500,
                  rotation_deg=(0.0, ALPHA_DEG, 0.0))

if __name__ == "__main__":
    _m.report(config, "robin_r0_am5")
