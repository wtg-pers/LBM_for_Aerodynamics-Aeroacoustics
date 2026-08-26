"""ROBIN rotor-off — LOCAL smoke twin (R = 12 L0 cells, ~2.4M nodes).

Same builder/boxes/BCs/surfel stack as robin_r0_musker; only r_lu0 and the
step count differ. Purpose (02 sec. 4 gates): STL placement + orientation,
surfel build (n_miss, dropped volume), surface channel presence, the
robin_anchor overlay pipeline, single-GPU vs MPI output parity. NOT a
physics run (dx3 = R/96, nose ~5 fine cells).

    python main.py --config configs/robin/robin_smoke.py --gpu 0
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \\
        --config configs/robin/robin_smoke.py --axis z --ghost 8 --cuda-aware 1 --gpu 0,1
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=12, tag="robin_smoke", max_steps=200, output_interval=100)
config["time"]["logging_interval"] = 20
config["time"]["checkpoint_interval"] = 200

if __name__ == "__main__":
    _m.report(config, "robin_smoke")
