"""HVAB hover c10 M0.65 — mlg4: NO L4 slab, ALM on L3 (interface-free near wake).

Complementary to the l4wake A/B: instead of moving the L4->L3 interface
deeper, this removes it — L3 (dx=L0/8) hosts the ALM and covers the whole
near wake (z/R +0.275 to -0.525) as a single level. Separates interface
effects from resolution effects for both loading and wake metrics.

Physical eps is unchanged (0.25c, chord-based); eps/dx = 3.3 (inboard) to
2.02 (tip marker) — legal but right at the 2dx floor at the tip. Marker
spacing n64 -> dr = 1.84 L3 cells < eps (overlap ok). Everything else
frozen at fact_s1_isoiso (pure ALM iso/iso, gaussian, KSAS, 20 rev).

Judge (vs s1 / l4wake03/04 / EXP): core-radius jump absent by construction —
does the in-L3 intrinsic decay half-life match the in-L4 220 deg anchor?
Trajectory descent? Loading shift from the coarser fine level (D-sweep
suggests CT up) is expected and is a *reference* observation, not the axis.

4-GPU run (65.1M cells, ~3.4 GB/GPU at 4 ranks):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mlg4.py \\
        --steps 25140 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 2 \\
        --ckpt-every 25140 --csv mlg4.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    eps_correction=None, prandtl_loss=False,
    sampling=None,
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4",
)
