"""V2-4 measurement gate on the g5 grid (surfel/84 §3, 3-GPU plan 0904).

robin_g5_r20 physics (5-level ladder, R/320, ~37.7M nodes) trimmed to a
600-step A/B measurement: V1 vs V2 (LBM_SURFEL_V2=1) step time + memory
census + section profile, checkpoint at 250/500 for the streaming-writer
and slab-restore verification (robin/16 §12·§12c) at scale.

Run (main dir, 3 GPUs — see surfel/84 §3 runbook):
  # A: V1 baseline
  LBM_BUILD_CENSUS=1 LBM_MEM_CENSUS=1 LBM_ESOTERIC=1 \\
  mpirun --mca pml ucx -n 3 python main.py --mpi \\
      --config configs/robin/robin_g5_v24_gate.py \\
      --axis x --ghost 8 --cuda-aware 1 --gpu 0,1,2 --dist-init --profile \\
      --results-dir results_g5_v24_v1
  # B: V2 band sandwich
  LBM_SURFEL_V2=1 LBM_BUILD_CENSUS=1 LBM_MEM_CENSUS=1 LBM_ESOTERIC=1 \\
  mpirun --mca pml ucx -n 3 python main.py --mpi \\
      --config configs/robin/robin_g5_v24_gate.py \\
      --axis x --ghost 8 --cuda-aware 1 --gpu 0,1,2 --dist-init --profile \\
      --results-dir results_g5_v24_v2
  # C: slab-restore verification (resume B from the step-500 checkpoint)
  LBM_SURFEL_V2=1 LBM_ESOTERIC=1 \\
  mpirun --mca pml ucx -n 3 python main.py --mpi \\
      --config configs/robin/robin_g5_v24_gate.py \\
      --axis x --ghost 8 --cuda-aware 1 --gpu 0,1,2 --dist-init \\
      --results-dir results_g5_v24_v2 --restart-latest --extend 100
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=20, tag="robin_g5_v24", max_steps=600,
                  output_interval=300, num_levels=5, ladder_boxes=True)
config["time"]["logging_interval"] = 50
config["time"]["checkpoint_interval"] = 250

if __name__ == "__main__":
    _m.report(config, "robin_g5_v24")
