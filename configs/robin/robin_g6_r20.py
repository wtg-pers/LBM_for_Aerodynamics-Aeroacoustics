"""ROBIN rotor-off grid campaign — 6-level ladder, r = 20 (robin/16 s10,
user: "last test"). finest = R/640 (boom 32.0 cells) over the SAME body box
as robin_g4_r20 / robin_g5_r20 (GRID4_L3_BOX_R -> L5), GRID5_L3_BOX_R as
the L4 cushion, and two GRID6 cushions (L2 enlarged to -0.35R fore / L3
new) because r = 20 leaves a single integer L0 node between the frozen L2
edge and the body box. Every parent->child margin >= 8 parent cells.

Cost (report(): node accounting, 736 B/node):
    L5 189.9 M + L4 58.4 M + L3 9.6 M + L2 1.9 M + L1 0.9 M + L0 4.0 M
    = 264.9 M nodes, working set ~182 GiB, ~7.1 G updates/step
    = 14 x robin_g5_r20 (503 M) = 37 x the r32-4 production run.
    -> NOT a single-GPU job and NOT a 2-rank z-slab job (105 GiB/rank).
    A uniform z-slab cut (cut pitch = 120/n L0 cells) always drops the
    12-L0-cell-thick body box into <= 2 ranks, so more z ranks do not
    help (mpi_axis PLAN s6: the 1-D slab ceiling). Feasible route = x-cut
    (mpi_axis S2, in progress): body box x-span 48 L0 cells -> 4 ranks x
    ~50 GiB VRAM (80 GB class), ~10 h at 295 MLUPS/rank.
    Affordable 4th ladder point WITHOUT x-cut: r25-5 (R/400, ~74 M nodes,
    ~51 GiB working set, 2-rank z-slab or Spark single, ~2 x g5 cost).

Readout: same as robin/16 (gate -> Cd_p/Cz -> projected h-probe [expect
h* ~ 1.0-1.1 cells again] -> selective Cp at p_sample_h 1.1 -> ladder
point 4). p_sample_h 1.1 / derived kh* (robin/16 s8) from the base.

Run (main dir; needs the x-cut MPI or a >= 200 GB single device):
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 4 python main.py --mpi \\
        --config configs/robin/robin_g6_r20.py --axis x --ghost 8 --cuda-aware 1 --gpu 0,1,2,3
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=20, tag="robin_g6_r20", max_steps=5600,
                  output_interval=280, num_levels=6, ladder_boxes=True)

if __name__ == "__main__":
    _m.report(config, "robin_g6_r20")
