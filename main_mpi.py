"""Multi-GPU MPI entry (patch 17 M5).

    mpirun -n <R> python main_mpi.py --config <cfg.py> --steps <N> \
        [--axis auto|x|y|z] [--ghost 3] [--cuda-aware 0|1] \
        [--csv out.csv] [--log-every 8] [--verify]

One rank per GPU (device = node-local rank). Every rank runs the full
deterministic production build, keeps its slab, and enters the lockstep
distributed loop (src/parallel/runner.py). --verify gathers the owned
assemblies to rank 0 after the run and compares against a fresh single-rank
production MultiLevelGrid.advance() of the same case (bench5-sized cases
only — the reference runs on rank 0's GPU next to its slab).

CUDA-awareness: --cuda-aware 1 passes CuPy buffers to MPI directly (cluster
OpenMPI 5.0.5 + UCX, task1-verified; requires the serialized thread level
set below BEFORE mpi4py import). --cuda-aware 0 stages through host numpy
(any MPI, e.g. local MPICH functional smoke). Default: LBM_MPI_CUDA env
(0 if unset).
"""
import os
import sys

os.environ.setdefault("LBM_ESOTERIC", "1")
for _k in ("MLG_CUDA_GRAPH", "MLG_PROFILE", "MLG_NVTX"):
    os.environ.pop(_k, None)

import mpi4py                                    # noqa: E402
mpi4py.rc.thread_level = 'serialized'            # BEFORE MPI import (task1)
from mpi4py import MPI                           # noqa: E402


def parse_cli(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, required=True,
                    help="coarse (L0) steps")
    ap.add_argument("--axis", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--devices", default=None,
                    help="comma-separated GPU ids per node-local rank, "
                         "e.g. '0,1' or '2,3' (default: local_rank %% ndev)")
    ap.add_argument("--ghost", type=int, default=3)
    ap.add_argument("--cuda-aware", default=os.environ.get("LBM_MPI_CUDA", "0"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--log-every", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    return ap.parse_args(argv)


def build(config, dev: int):
    sys.argv = ["mpi", "--config", config, "--gpu", str(dev),
                "--no-vtk", "--no-force"]
    from src.io.args_parser import parse_args
    from src.solver.setup import SimulationSetup
    from src.solver.initializer import SolverInitializer
    a = parse_args()
    s = SimulationSetup(a)
    s.start_log_capture()
    mlg = s.build_simulation()
    SolverInitializer(s).initialize(mlg, a)
    s.stop_log_capture()
    mlg._graph_enabled = False
    return mlg


def main():
    args = parse_cli(sys.argv[1:])
    comm = MPI.COMM_WORLD
    rank, nr = comm.Get_rank(), comm.Get_size()

    # device = node-local rank (mpirun may also pin via CUDA_VISIBLE_DEVICES)
    local = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    import cupy as cp
    ndev = cp.cuda.runtime.getDeviceCount()
    if args.devices:
        ids = [int(d) for d in args.devices.split(",")]
        dev = ids[local % len(ids)]
        if dev >= ndev:
            raise ValueError(f"--devices {args.devices}: id {dev} >= "
                             f"visible device count {ndev}")
    else:
        dev = local % ndev
    cp.cuda.Device(dev).use()

    if rank != 0:                                 # quiet non-root builds
        sys.stdout = open(os.devnull, "w")
    mlg = build(args.config, dev)
    sys.stdout = sys.__stdout__

    from src.parallel import MPITransport
    from src.parallel.alm_dist import MPIAllreduce
    from src.parallel.runner import DistributedMLGRunner

    axis = None if args.axis == "auto" else "xyz".index(args.axis)
    transport = MPITransport(comm, cuda_aware=str(args.cuda_aware) == "1")
    runner = DistributedMLGRunner(
        mlg, transport, rank, nr, allreduce=MPIAllreduce(comm),
        axis=axis, ghost=args.ghost)

    # free the full build (runner kept slabs + the ALM model)
    import gc
    del mlg
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    if rank == 0:
        print(f"[mpi] ranks={nr} axis={'xyz'[runner.axis]} "
              f"ghost={args.ghost} cuda_aware={args.cuda_aware}", flush=True)
        for k, p in enumerate(runner.parts):
            print(f"[mpi]   L{k} rank0 {p}", flush=True)
        csv_f = open(args.csv, "w") if args.csv else None
        if csv_f:
            csv_f.write("step,time_lu,thrust,torque,power,C_T,C_P,FM\n")

    def on_log(s, rn):
        if rank != 0:
            return
        line = f"[mpi] step {s}/{args.steps}"
        if rn.last_interval:
            sps = rn.last_interval["s_per_step"]
            eta_h = (args.steps - s) * sps / 3600.0
            line += f"  {sps:.3f}s/step  ETA {eta_h:.2f}h"
        if rn.model is not None:
            perf = rn.model.get_rotor_performance()
            line += (f"  T={perf['thrust']:.6e}  CT={perf['C_T']:.6e}"
                     f"  FM={perf['FM']:.4f}")
            if csv_f:
                csv_f.write(f"{s},{perf['time']},{perf['thrust']},"
                            f"{perf['torque']},{perf['power']},"
                            f"{perf['C_T']},{perf['C_P']},{perf['FM']}\n")
                csv_f.flush()
        print(line, flush=True)

    stats = runner.run(args.steps, log_every=args.log_every, on_log=on_log)
    comm.Barrier()
    if rank == 0:
        print(f"[mpi] done: {stats['steps']} coarse steps, "
              f"{stats['wall_s']:.2f}s ({stats['s_per_step']:.3f} s/step)",
              flush=True)
        if csv_f:
            csv_f.close()

    if args.verify:
        verify(comm, rank, nr, runner, args)
    MPI.Finalize()


def verify(comm, rank, nr, runner, args):
    """Gather owned assemblies to rank 0; compare vs fresh 1-rank reference."""
    import cupy as cp
    import numpy as np
    NL = runner.NL
    for k in range(NL):
        mine = cp.asnumpy(runner.owned_f_std(k))
        if rank == 0:
            pieces = [mine]
            for r in range(1, nr):
                meta = comm.recv(source=r, tag=100 + k)
                buf = np.empty(meta[0], np.float32)
                comm.Recv(buf, source=r, tag=200 + k)
                pieces.append(buf.reshape(meta[1]))
            setattr(runner, f"_asm{k}",
                    np.concatenate(pieces, axis=1 + runner.axis))
        else:
            comm.send((mine.size, mine.shape), dest=0, tag=100 + k)
            comm.Send(np.ascontiguousarray(mine), dest=0, tag=200 + k)
    if rank != 0:
        return
    print("[verify] building 1-rank reference...", flush=True)
    ref = build(args.config, 0)
    for _ in range(args.steps):
        ref.advance()
    ok = True
    for k in range(NL):
        rf = cp.asnumpy(ref.get_level(k).physical_f)
        d = np.abs(getattr(runner, f"_asm{k}") - rf)
        df = float(d.max())
        bit = bool(df == 0.0)
        ok = ok and (df < 1e-4)
        print(f"[verify] L{k}: max|df|={df:.3e}  bit={bit}", flush=True)
        if not bit:
            # localize: owning rank per diff y tells devices apart; x range
            # tells sponge (high x) from bulk; count tells isolated vs band
            nz = np.argwhere(d > 0)
            gb = getattr(runner, "bounds", None)
            ys = np.unique(nz[:, 1 + runner.axis])
            own = [int(np.searchsorted(gb, y, side="right") - 1)
                   if gb is not None else -1 for y in ys[:12]]
            print(f"[verify]   diff cells={len(nz)}  "
                  f"axis-coords={ys[:12].tolist()} -> owner rank {own}",
                  flush=True)
            for row in nz[:5]:
                print(f"[verify]   (q,x,y,z)={tuple(int(v) for v in row)}  "
                      f"dist={d[tuple(row)]:.3e}", flush=True)
    print(f"[verify] RESULT: {'PASS' if ok else 'FAIL'} "
          "(gate: max|df|<1e-4; bit expected at 2 ranks)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)      # fail-fast: never leave peers hanging
