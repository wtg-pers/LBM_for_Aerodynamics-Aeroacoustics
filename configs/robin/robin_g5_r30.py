"""ROBIN rotor-off grid campaign — 5-level ladder, r = 30 (4th ladder point).

User decision (0903, mpi_axis/06 s7): g6_r20 (R/640, 265M) exceeds the
anode1 4x24GB envelope under the V1 bridge (three walls measured and the
runtime 2F staging remains structural until V2), so the 4th ladder point
moves to the 5-level ladder at r = 30: finest = R/480 (boom 24.0 cells)
over the SAME body box family as g4/g5 (GRID4_L3_BOX_R -> L4, GRID5
L3 cushion) -> sits between g5_r20 (R/320) and g6_r20 (R/640) on the
h*(dx) / Cd_p ladder, ~3.4x the g5_r20 nodes.

Cost (see report()): ~127M nodes -- NOT a single-GPU job on 24 GB.
4-rank x-cut: eso residency ~3.4 GiB/rank + L4 slab 2F staging ~4.4 GiB
-> peak ~14-15 GiB/rank (comfortable; B1 per-level cuts engage only if
the shared-cut partitioner is infeasible -- either path is registered).
p_sample_h 1.1 + derived kh* (robin/16 s8) via the base.

Run (main dir, 4-rank x-cut, dist-init REQUIRED at this size):
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 4 python main.py --mpi \\
        --config configs/robin/robin_g5_r30.py \\
        --axis x --ghost 8 --cuda-aware 1 --gpu 0,1,2,3 --dist-init
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

# steps/output scale with r (same physical window as the g2/g4/g5 ladder:
# dt ~ dx -> r30 = 5600 * 30/20 = 8400, VTK cadence likewise)
config = _m.build(r_lu0=30, tag="robin_g5_r30", max_steps=8400,
                  output_interval=420, num_levels=5, ladder_boxes=True)

if __name__ == "__main__":
    _m.report(config, "robin_g5_r30")
