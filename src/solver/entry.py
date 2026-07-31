"""Unified entry point: single-GPU and MPI runs through one dispatcher.

`python main.py` (or `mpirun -n 1 python main.py`) takes the single-process
path — exactly the historical main.py loop (2D/standard/fused/esoteric
opt-in, CUDA graph eligible), and never imports mpi4py.
`mpirun -n N>1 python main.py` takes the distributed path
(src/parallel/driver.py).

IMPORT RULE: this module must stay light at import time — no cupy, no
mpi4py, no src.solver.setup at module top. The MPI branch must be able to
initialize MPI before any CUDA context is created (UCX device-direct), and
the single branch must never pay the mpi4py import.
"""

import os
import sys


def detect_world_size() -> int:
    """MPI world size from launcher env, 1 when not under an MPI launcher.

    Only OMPI_COMM_WORLD_SIZE (OpenMPI) and PMI_SIZE (MPICH/hydra) are
    trusted. Deliberately NOT SLURM_NTASKS: sbatch exports it to every
    process in an allocation, so a serial `python main.py` inside a 4-task
    batch script would falsely take the MPI path. `--mpi` is the escape
    hatch for exotic launchers (e.g. srun --mpi=pmix).
    """
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        v = os.environ.get(var, "")
        if v.isdigit():
            return int(v)
    return 1


def _reject_mpi_only_flags(args) -> None:
    """Error out (exit 2) on MPI-only flags at world size == 1.

    Silently ignoring them would be a silent downgrade of user intent."""
    from src.io.args_parser import MPI_ONLY_DESTS
    bad = [flag for dest, (flag, default) in MPI_ONLY_DESTS.items()
           if getattr(args, dest, default) != default]
    if bad:
        sys.exit(f"error: {', '.join(sorted(bad))}: MPI-only flag(s) — "
                 f"launch with mpirun -n N>1 (or --mpi)")


def run_single(args):
    """Single-process run — the historical main.py body, moved verbatim."""
    from src.solver.setup import SimulationSetup
    from src.solver.initializer import SolverInitializer

    print("=" * 70)
    print(" LBM Solver for Aerodynamics & Aeroacoustics")
    print("=" * 70)

    # [1] Setup — 시뮬레이션 환경 구성
    setup = SimulationSetup(args)

    # [2] Build & Initialize — 상세 로그는 파일로
    setup.start_log_capture()
    sim = setup.build_simulation()
    output = setup.build_output_manager()

    initializer = SolverInitializer(setup)
    start_step, end_step = initializer.initialize(sim, args)
    setup.stop_log_capture()

    if start_step >= end_step:
        return True

    # [3] Execution — 시간 루프
    print(f"\n[6] Running Simulation")
    print("=" * 70)

    output.start(start_step, end_step)

    for step in range(start_step, end_step):
        sim.advance()
        if output.process(step, sim) == 'stop':
            break

    # [4] Finalize
    return output.finalize(sim)


def main(argv=None):
    from src.io.args_parser import parse_args
    args = parse_args(argv)

    nr = detect_world_size()

    if args.list_gpus:
        if nr > 1:
            sys.exit("error: --list-gpus under an MPI launcher is not "
                     "supported (run it as a plain `python main.py "
                     "--list-gpus`)")
        from src.utilities.device import print_gpu_info
        print_gpu_info()
        return

    if nr > 1 or args.mpi:
        from src.parallel.driver import run_mpi
        return run_mpi(args)

    _reject_mpi_only_flags(args)
    ids = getattr(args, 'gpu_ids', None)
    if ids and len(ids) > 1:
        sys.exit(f"error: --gpu {','.join(map(str, ids))}: multiple GPU ids "
                 f"need an MPI launch — mpirun -n {len(ids)} python main.py "
                 f"... (a single-process run takes one id)")
    return run_single(args)
