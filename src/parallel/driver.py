"""Distributed (MPI) driver — the multi-rank branch of the unified entry.

Ported from the historical main_mpi.py (patch 17 M5) into src/ so the MPI
driver can never again be missed by an src-only deploy (patch_notes/
hpc_upgrade/20_mpi_result_csv_nut_vtk.md incident class). Since stage C5
the loop runs the FULL OutputManager pipeline via MPIOutputManager
(src/parallel/mpi_output.py): force_history.csv with the properly
rescaled fine-level reference, conservation/convergence with a
rank-invariant 'stop' verdict, VTK/checkpoint/finalize — one output
source for both entry paths.

    mpirun -n <R> python main.py --config <cfg.py> --steps <N> \
        [--axis auto|x|y|z] [--ghost 3] [--cuda-aware 0|1] \
        [--csv out.csv] [--log-every 8] [--verify]

One rank per GPU (device = node-local rank). Every rank runs the full
deterministic production build, keeps its slab, and enters the lockstep
distributed loop (src/parallel/runner.py). --verify gathers the owned
assemblies to rank 0 after the run and compares against a fresh
single-rank production MultiLevelGrid.advance() of the same case.

CUDA-awareness: --cuda-aware 1 passes CuPy buffers to MPI directly
(cluster OpenMPI 5.0.5 + UCX; requires the serialized thread level set
BEFORE mpi4py import). --cuda-aware 0 stages through host numpy (any MPI,
e.g. local MPICH functional smoke). Default: LBM_MPI_CUDA env.

IMPORT RULE: mpi4py is imported only inside run_mpi(), thread_level first
(OpenMPI 5.x drops the UCX PML otherwise — src/parallel/halo.py). cupy is
imported after MPI init so no CUDA context predates MPI_Init.
"""

import argparse
import os
import sys


def _env_hygiene() -> None:
    """MPI-path env contract, applied before any src import reads env."""
    if os.environ.get("LBM_ESOTERIC") == "0":
        sys.exit("[mpi] env error: LBM_ESOTERIC=0 is set, but the "
                 "distributed runner operates on esoteric-pull state. Unset "
                 "it (it defaults to 1 under MPI) or run single-GPU.")
    os.environ.setdefault("LBM_ESOTERIC", "1")
    if "MLG_CUDA_GRAPH" in os.environ:
        print("[mpi] note: MLG_CUDA_GRAPH ignored under MPI (CUDA graphs "
              "are unsupported in the distributed loop); single-GPU runs "
              "keep it.", flush=True)
    for k in ("MLG_CUDA_GRAPH", "MLG_PROFILE", "MLG_NVTX"):
        os.environ.pop(k, None)


def run_mpi(args):
    _env_hygiene()
    import mpi4py                                    # noqa: E402
    mpi4py.rc.thread_level = 'serialized'            # BEFORE MPI import
    from mpi4py import MPI                           # noqa: E402
    try:
        return _run(args, MPI)
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        MPI.COMM_WORLD.Abort(1)   # fail-fast: never leave peers hanging


def _fail_fast_config(setup, mlg) -> None:
    """Reject configs the distributed runner cannot represent — explicit
    errors instead of deep AttributeError/KeyError deaths (no-silent rule)."""
    cfg = getattr(setup, 'config_path', None) or 'config'
    if not getattr(setup, '_mlg_enabled', False):
        raise ValueError(
            f"[mpi] config error: MPI runs require a multi-level grid "
            f"(mlg.enabled=true) — DistributedMLGRunner decomposes "
            f"MultiLevelGrid only. Run it single-GPU: python main.py")
    if getattr(getattr(setup, 'lattice', None), 'dim', 3) == 2:
        raise ValueError(
            f"[mpi] config error: MPI runs support 3D (D3Q27) only; "
            f"{cfg} is 2D. Run it single-GPU: python main.py")
    al = getattr(setup, 'al_model', None)
    if al is not None and type(al).__name__ == 'MultiRotorManager':
        raise ValueError(
            "[mpi] config error: multi-rotor ALM (actuator_line.rotors) is "
            "unsupported under MPI — the distributed sampler binds exactly "
            "one rotor's F_grid (src/parallel/runner.py). Run single-GPU.")
    lvl0 = mlg.get_level(0)
    if not getattr(lvl0, '_use_esoteric', False):
        raise ValueError(
            "[mpi] config error: esoteric-pull could not be enabled for "
            "this config (requires GPU + 3D + BGK/Cumulant + precision "
            "float32 — see csv/setup_log.txt for the fallback warning). "
            "The MPI runner requires it.")


def _build(args, dev: int, with_writers: bool = False,
           use_restart: bool = True, dist_init=None,
           io_role: str = 'writer'):
    """Replicated production build. No sys.argv shim — a Namespace copy of
    the unified args, so directory/clear flags reach setup too.

    io_role='silent' (rank != 0): no directories/clear, no writers, no
    setup_log/CSV headers. CheckpointManager still built (restore path)."""
    di = args.dist_init if dist_init is None else dist_init
    os.environ["LBM_DIST_INIT"] = "1" if di else "0"
    bargs = argparse.Namespace(**vars(args))
    bargs.gpu = dev
    bargs.no_vtk = (not with_writers) or args.no_vtk
    if not use_restart:
        bargs.restart = None
        bargs.restart_latest = False
    from src.solver.setup import SimulationSetup
    from src.solver.initializer import SolverInitializer
    s = SimulationSetup(bargs, io_role=io_role)
    s.start_log_capture()
    mlg = s.build_simulation()
    SolverInitializer(s).initialize(mlg, bargs)
    s.stop_log_capture()
    _fail_fast_config(s, mlg)
    mlg._graph_enabled = False
    return mlg, s


def _collect_solid_masks(mlg_obj, n_levels):
    """Static per-level solid masks (host numpy) for the VTK channel."""
    masks = []
    for k in range(n_levels):
        ob = getattr(mlg_obj.get_level(k), "obstacle_bc", None)
        sm = getattr(ob, "solid_mask", None) if ob is not None else None
        if sm is not None and hasattr(sm, "get"):
            sm = sm.get()
        masks.append(sm)
    return masks


def _run(args, MPI):
    import numpy as np

    if args.steps is None:
        sys.exit("error: --steps N is required for MPI runs "
                 "(unifies into --max-steps in a later stage)")
    log_every = 8 if args.log_every is None else args.log_every
    vtk_every = args.vtk_every or 0
    ckpt_every = args.ckpt_every or 0
    cuda_aware = (args.cuda_aware if args.cuda_aware is not None
                  else os.environ.get("LBM_MPI_CUDA", "0"))

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

    if rank != 0:
        # permanently quiet: every print in the shared output pipeline is
        # rank-0-only by silencing the others at the source (stderr kept
        # for tracebacks)
        sys.stdout = open(os.devnull, "w")
    if args.dist_init and (args.restart or args.restart_latest):
        raise ValueError("--dist-init + restart: restore path loads full "
                         "fields (use replicated build for restarts for now)")
    mlg, setup = _build(args, dev,
                        with_writers=(rank == 0),
                        io_role=('writer' if rank == 0 else 'silent'))

    from src.parallel import MPITransport
    from src.parallel.alm_dist import MPIAllreduce
    from src.parallel.runner import DistributedMLGRunner
    from src.parallel.mpi_output import MPIOutputManager

    NL = mlg.num_levels
    level_shapes = [tuple(mlg.get_level(k).domain_shape) for k in range(NL)]

    output = setup.build_output_manager(
        manager_cls=MPIOutputManager,
        comm=comm, rank=rank, nr=nr, mpi_mod=MPI,
        log_every=log_every, vtk_every=vtk_every, ckpt_every=ckpt_every,
        vtk_fields_last=args.vtk_fields_last,
        dense_csv_path=args.csv)

    axis = None if args.axis == "auto" else "xyz".index(args.axis)
    transport = MPITransport(comm, cuda_aware=str(cuda_aware) == "1")
    runner = DistributedMLGRunner(
        mlg, transport, rank, nr, allreduce=MPIAllreduce(comm),
        axis=axis, ghost=args.ghost)

    # rank-invariant body level: the local nt==1 scan can miss on a rank
    # whose slab holds no solid cells -> mismatched collectives (the old
    # tier hazard). MAX-allreduce the scan result.
    if nr > 1:
        bl = np.array([-1.0 if runner.body_level is None
                       else float(runner.body_level)])
        out_bl = np.empty_like(bl)
        comm.Allreduce(bl, out_bl, op=MPI.MAX)
        runner.body_level = None if out_bl[0] < 0 else int(out_bl[0])

    solid_masks = _collect_solid_masks(mlg, NL) if rank == 0 else None
    start_step = runner.completed_step + 1
    output.bind_runner(runner, level_shapes, solid_masks,
                       dist_init=args.dist_init, start_step=start_step)

    # the MEM link/mask arrays are dead weight on the MPI path (forces come
    # from eso_mem_force + Allreduce via forces_override)
    if setup.force_mgr is not None:
        setup.force_mgr.release_device_state()

    # free the full build (runner kept slabs + the ALM model; the output
    # manager kept writers/monitors/paths)
    import gc
    del mlg, setup
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    if start_step > args.steps:
        if rank == 0:
            print(f"[mpi] nothing to do: restored step {start_step - 1} "
                  f">= --steps {args.steps}", flush=True)
        MPI.Finalize()
        return
    if rank == 0:
        print(f"[mpi] ranks={nr} axis={'xyz'[runner.axis]} "
              f"ghost={args.ghost} cuda_aware={cuda_aware}"
              + (f"  RESTART from {runner.completed_step}"
                 if runner.completed_step else ""), flush=True)
        for k, p in enumerate(runner.parts):
            print(f"[mpi]   L{k} rank0 {p}", flush=True)

    if args.profile:
        runner.profile = {}
    runner.last_interval = None

    # legacy-inclusive step labels (start..steps) until the C8 semantics
    # unification; OutputManager sees the exclusive end start..steps+1
    output.start(start_step, args.steps + 1)
    for s_ in range(start_step, args.steps + 1):
        runner.step_coarse()
        if output.process(s_, runner) == 'stop':
            break
    cp.cuda.runtime.deviceSynchronize()

    result = output.finalize(runner)

    if args.profile:
        prof = comm.gather(runner.profile, root=0)
        if rank == 0:
            keys = sorted({k for d in prof for k in d})
            print("[profile] per-rank seconds over the whole run:",
                  flush=True)
            for k in keys:
                vals = " ".join(f"{d.get(k, 0.0):7.2f}" for d in prof)
                print(f"[profile]   {k:14s} {vals}", flush=True)

    if args.verify:
        _verify(comm, rank, nr, runner, args)
    MPI.Finalize()
    return result


def _verify(comm, rank, nr, runner, args):
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
    ref, _ = _build(args, 0, use_restart=False, dist_init=False)
    for _ in range(args.steps):
        ref.advance()
    ok = True
    for k in range(NL):
        rf = cp.asnumpy(ref.get_level(k).physical_f)
        d = np.abs(getattr(runner, f"_asm{k}") - rf)
        df = float(d.max())
        bit = bool(df == 0.0)
        ok = ok and (bit if args.strict_bit else df < 1e-4)
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
