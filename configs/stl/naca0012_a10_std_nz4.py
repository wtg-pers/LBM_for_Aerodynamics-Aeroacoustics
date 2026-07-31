"""NACA0012 infinite wing a=10 — thin slab (Nz=4), STANDARD-path force.

WHY THIS VARIANT (2026-08-01 audit): the esoteric force diagnostics
carry systematic biases on this case family — measured on the c=50
a=0 twin at identical flow (rho diff ~6e-5, i.e. dynamics agree):

    std IBB  +0.0087   std HWBB  +0.0102   (mutually consistent, sane)
    eso IBB  -0.0388   eso HWBB  +0.0488   (opposite-sign biases)

The single-level force gates (S5 G2/G3) all pass, so the defect lives
in a configuration the gates never covered (span-through links and/or
level composition) — repair is a separate track. Until then the
STANDARD path is the trusted force reference (S4 sphere 1.19% vs
analytic; a=0 wing +0.009 physical).

The PROVEN z-slice invariance (<= 7e-7, non-growing) makes Nz a pure
cost knob: Nz=4 gives the identical quasi-2D physics at ~13.1M cells
-> the STANDARD path fits ONE GPU (~5.4 GB), no MPI needed:

    python main.py --config configs/stl/naca0012_a10_std_nz4.py --gpu 2

(Do NOT set LBM_ESOTERIC for this run — the point is the std force.)
Reference: Ladson/TMR a=10: Cl ~= 1.09, Cd ~= 0.0123 (absolute Cd
comparison stays conservative: friction unresolved at Re 6e6).
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_nz4_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)          # Nz = 4

_folder = "results_naca0012_a10_std_nz4"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
