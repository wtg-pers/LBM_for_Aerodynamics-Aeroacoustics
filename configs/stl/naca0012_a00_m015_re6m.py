"""NACA0012 infinite wing at a = 0 — Cd-sign discriminator.

Same wing STL / grid / BCs as naca0012_a10_m015_re6m.py; the baked
+10-deg pitch is cancelled by Rz(+10) applied after the Rx(-90) axis
swap (rotation order Rz@Ry@Rx; verified numerically: chordline angle
-0.000 deg, y extent = pure thickness 12 lu, chord 100.00 lu).

WHY: the a=10 run measures Cd ~= -0.037 locked negative while Cl ~0.85
is sane. Flat-plate decomposition brackets Cd between +Cl*tan(a) ~
+0.148 (no LE suction recovered) and 0 (ideal full suction) — the
negative value means ~25% suction OVER-recovery, which is either an LE
discretization artifact or a MEM-force defect. At a = 0 the LE suction
mechanism vanishes and a symmetric airfoil MUST measure Cd >= 0
(small): a negative Cd here convicts the force measurement; a small
positive one points to LE-resolution physics at a = 10.

    LBM_ESOTERIC=1 mpirun -n 2 python main.py \
        --config configs/stl/naca0012_a00_m015_re6m.py --gpu 2,3 --dist-init
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_a00_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100)

config["internal_geometry"]["stl"]["rotation_deg"] = (-90.0, 0.0, 10.0)

_folder = "results_naca0012_a00_m015_re6m"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
