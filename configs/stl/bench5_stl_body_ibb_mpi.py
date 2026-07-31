"""bench5_stl_body_ibb MPI twin — kleine STRAIGHT wake (S7 rank gate).

Distributed ALM raises NotImplementedError for kleine wake='free' (wake
point velocity sampling across ranks is unsupported by design); the
1<->2-rank equivalence gate therefore runs this straight-wake twin on
BOTH sides (n=1 vs n=2, same model -> pure MPI-correctness comparison).
The CT/CP noise-band gate vs bench5_baseline stays on the free-wake
primary config, single GPU. Decomposition axis is y: the n=2 cut
(y=24) passes through the body center.

    LBM_ESOTERIC=1 mpirun -n 2 python main.py \
        --config configs/stl/bench5_stl_body_ibb_mpi.py \
        --max-steps 32 --gpu 0,1 --verify
    # n=1 side of the rank gate (--mpi forces the MPI runner at world=1):
    LBM_ESOTERIC=1 mpirun -n 1 python main.py \
        --config configs/stl/bench5_stl_body_ibb_mpi.py --mpi --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "bench5_stl_body_ibb.py")
_spec = importlib.util.spec_from_file_location("bench5_stl_body_base_mpi", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

config["actuator_line"]["eps_correction"] = dict(
    config["actuator_line"]["eps_correction"], wake="straight")

_folder = "results_bench5_stl_body_ibb_mpi"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
