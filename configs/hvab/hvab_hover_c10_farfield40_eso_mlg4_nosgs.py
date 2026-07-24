"""HVAB hover c10 M0.65 — mlg4 + SGS OFF: wake-dissipation source A/B.

Every run so far (factorial / l4wake / afix / mlg4) had dyn_smag ON. The
measured in-level tip-vortex decay (half-life ~220 deg, dx-independent
L3=L4) is the combined cumulant+SGS+resolution floor; this case removes the
SGS to split it:
    half-life up   -> nu_t contributes to core decay (SGS model lever)
    unchanged      -> cumulant-intrinsic floor (collision A/B next)
The old 6-case SGS on/off A/B judged only CT ("nu_t irrelevant") — this is
the wake-metric re-examination that memo flagged as open.

Everything else frozen at mlg4 (ALM on L3, interface-free near wake, iso/iso
gaussian, KSAS, n64, 20 rev). Judge with extract_tip_vortex_trajectory.py:
in-level half-life vs 220 deg anchor, core radius, trajectory (expect
unchanged — loading-driven), plus CT/CP as loading cross-check.

Stability note: pure cumulant at tau ~= 0.5 with nu_t = 0; if it diverges,
re-run with numerics.nan_trap enabled to locate, but cumulant should hold.

4-GPU run (65.1M cells):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mlg4_nosgs.py \\
        --steps 25140 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 2 \\
        --ckpt-every 25140 --csv mlg4_nosgs.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling=None,
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_nosgs",
)
