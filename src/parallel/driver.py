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
    # Multi-rotor and multi-block are both supported since patch 18: the
    # runner indexes every per-grid list by BLOCK uid, derives each block's
    # partition from its PARENT BLOCK's, lets a rank own zero cells of a
    # block (skipping it with its subtree), and gives each ALM block its own
    # sub-communicator. A MultiRotorView reaching here is therefore fine —
    # each block's own manager does the stepping, the view only reports.
    if getattr(setup, '_numerics_cfg', {}).get('esoteric', None) is False:
        raise ValueError(
            "[mpi] config error: numerics.esoteric=false, but the "
            "distributed runner operates on esoteric-pull state. Remove "
            "the key (or set true) for MPI runs, or run single-GPU.")
    # EVERY level must be esoteric — _init_esoteric can fail (e.g. OOM)
    # and fall back to the standard path PER LEVEL with only a warning;
    # a level-0-only check let a fine-level fallback slip through to a
    # broken distributed run (observed: L3 eso-init OOM on a 94M case).
    for b in mlg.iter_blocks():
        if not getattr(b.sim, '_use_esoteric', False):
            raise ValueError(
                f"[mpi] config error: esoteric-pull is not active on "
                f"{b.label} (requires GPU + 3D + BGK/Cumulant + precision "
                f"float32; an init failure such as OOM also falls back — "
                f"see csv/setup_log.txt for the warning). The MPI runner "
                f"requires esoteric state on every level; for large cases "
                f"use --dist-init to avoid full-domain init allocations.")
    # wall_bc='ibb' under MPI: supported since STL track S6 — the runner
    # builds a slab-filtered esoteric deposit-rewrite pass per level
    # (build_slab_ibb) and routes the MEM force through its scatter.


def resolve_local_device(args, local_rank: int, ndev: int,
                         verbose: bool = False) -> int:
    """Node-local rank -> GPU id under the unified --gpu flag.

    --gpu ID[,ID,...] is the single interface (rank r -> id[r % len]);
    --devices is a deprecated alias kept for old scripts. Giving both is
    an error (no silent precedence). Unset: local_rank % ndev.
    """
    ids = list(getattr(args, 'gpu_ids', None) or [])
    if getattr(args, 'devices', None):
        if ids:
            raise ValueError(
                "--gpu and --devices are the same flag (--devices is a "
                "deprecated alias) — give only --gpu ID[,ID,...]")
        if verbose:
            print("[mpi] warning: --devices is deprecated — use "
                  "--gpu ID[,ID,...]", flush=True)
        ids = [int(d) for d in args.devices.split(",")]
    if ids:
        dev = ids[local_rank % len(ids)]
        if dev >= ndev or dev < 0:
            raise ValueError(f"--gpu/--devices: id {dev} outside visible "
                             f"device range [0, {ndev - 1}]")
        return dev
    return local_rank % ndev


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
    start_step, end_step = SolverInitializer(s).initialize(mlg, bargs)
    s.stop_log_capture()
    _fail_fast_config(s, mlg)
    mlg._graph_enabled = False
    return mlg, s, start_step, end_step


def _collect_solid_masks(mlg_obj):
    """Static per-BLOCK solid masks (host numpy) for the VTK channel,
    level-major so the index is the block uid."""
    masks = []
    for b in mlg_obj.iter_blocks():
        ob = getattr(b.sim, "obstacle_bc", None)
        sm = getattr(ob, "solid_mask", None) if ob is not None else None
        if sm is not None and hasattr(sm, "get"):
            sm = sm.get()
        masks.append(sm)
    return masks


def _run(args, MPI):
    import numpy as np

    # unified step semantics (C8): 0-based EXCLUSIVE range, exactly like
    # the single-GPU loop. --max-steps absolute > --extend relative >
    # config time.max_steps. --steps N is a deprecated alias of
    # --max-steps N: same advance count, labels shift by -1.
    if args.steps is not None:
        if args.max_steps is not None:
            sys.exit("error: --steps conflicts with --max-steps "
                     "(--steps is a deprecated alias)")
        print(f"[mpi] warning: --steps is deprecated — use --max-steps "
              f"{args.steps} (same advance count; step labels are now "
              f"0-based)", flush=True)
        args.max_steps = args.steps
        args.steps = None
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
    dev = resolve_local_device(args, local, ndev, verbose=(rank == 0))
    cp.cuda.Device(dev).use()

    if rank != 0:
        # permanently quiet: every print in the shared output pipeline is
        # rank-0-only by silencing the others at the source (stderr kept
        # for tracebacks)
        sys.stdout = open(os.devnull, "w")
    if args.dist_init and (args.restart or args.restart_latest):
        raise ValueError("--dist-init + restart: restore path loads full "
                         "fields (use replicated build for restarts for now)")
    mlg, setup, start_step, end_step = _build(
        args, dev, with_writers=(rank == 0),
        io_role=('writer' if rank == 0 else 'silent'))

    from src.parallel import MPITransport
    from src.parallel.alm_dist import MPIAllreduce
    from src.parallel.runner import DistributedMLGRunner
    from src.parallel.mpi_output import MPIOutputManager

    # per-BLOCK shapes, level-major (index == block uid)
    block_shapes = [tuple(b.sim.domain_shape) for b in mlg.iter_blocks()]

    output = setup.build_output_manager(
        manager_cls=MPIOutputManager,
        comm=comm, rank=rank, nr=nr, mpi_mod=MPI,
        log_every=log_every, vtk_every=vtk_every, ckpt_every=ckpt_every,
        vtk_fields_last=args.vtk_fields_last,
        dense_csv_path=args.csv)

    axis = None if args.axis == "auto" else "xyz".index(args.axis)
    transport = MPITransport(comm, cuda_aware=str(cuda_aware) == "1")

    # esoteric IBB needs ghost >= 5 (deposit rewrite adds a dependency hop
    # on top of the kernel store — see build_slab_ibb). Auto-bump so
    # wall_bc='ibb' configs run correctly with the default --ghost 3.
    ghost = args.ghost
    from src.boundary.interpolated_wall import InterpolatedBounceBack
    if any(isinstance(getattr(b.sim, 'obstacle_bc', None),
                      InterpolatedBounceBack) for b in mlg.iter_blocks()):
        if ghost < 5:
            if rank == 0:
                print(f"[mpi] esoteric IBB: ghost {ghost} -> 5 "
                      f"(rewrite dependency-hop requirement)", flush=True)
            ghost = 5

    runner = DistributedMLGRunner(
        mlg, transport, rank, nr, allreduce=MPIAllreduce(comm),
        axis=axis, ghost=ghost, comm=comm,
        cut_policy=getattr(args, 'cut_policy', 'balanced'))

    # rank-invariant body BLOCK: the local nt==1 scan can miss on a rank
    # whose slab holds no solid cells (or that owns none of the body block
    # at all) -> mismatched collectives (the old tier hazard). MAX-allreduce
    # the scan result; uids are level-major so MAX picks the finest.
    if nr > 1:
        bl = np.array([-1.0 if runner.body_block is None
                       else float(runner.body_block)])
        out_bl = np.empty_like(bl)
        comm.Allreduce(bl, out_bl, op=MPI.MAX)
        runner.body_block = None if out_bl[0] < 0 else int(out_bl[0])

    solid_masks = _collect_solid_masks(mlg) if rank == 0 else None
    assert runner.completed_step == start_step, \
        (runner.completed_step, start_step)
    output.bind_runner(runner, block_shapes, solid_masks,
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

    if start_step >= end_step:
        if rank == 0:
            print(f"[mpi] nothing to do: {start_step} completed steps >= "
                  f"end step {end_step}", flush=True)
        MPI.Finalize()
        return True
    if rank == 0:
        print(f"[mpi] ranks={nr} axis={'xyz'[runner.axis]} "
              f"ghost={args.ghost} cuda_aware={cuda_aware}"
              + (f"  RESTART: {runner.completed_step} steps done, "
                 f"resuming at label {start_step}"
                 if runner.completed_step else ""), flush=True)
        if runner.completed_step:
            print("[mpi] note: step labels are 0-based (unified). A "
                  "checkpoint written by the pre-unification main_mpi "
                  "resumes one step late — convert once with: "
                  "step -= 1 in the npz (see docs/manual/09).", flush=True)
        for b, p in zip(runner.blocks, runner.parts):
            print(f"[mpi]   {b.label} rank0 {p}"
                  + ("" if p.own_count else "  (owns nothing)"), flush=True)

    if args.profile:
        runner.profile = {}
    runner.last_interval = None

    output.start(start_step, end_step)
    for step in range(start_step, end_step):
        runner.step_coarse()
        if output.process(step, runner) == 'stop':
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
        _verify(comm, rank, nr, runner, args, n_advances=end_step)
    MPI.Finalize()
    return result


def _verify(comm, rank, nr, runner, args, n_advances=None):
    """Gather owned assemblies to rank 0; compare vs fresh 1-rank reference.

    The reference is built with io_role='silent' and restart disabled:
    no directories are created/cleared and no CSV headers are rewritten,
    so a config with clear_previous=True can no longer wipe the results
    the distributed run just wrote (historical hazard)."""
    import cupy as cp
    import numpy as np
    from src.parallel.output import _gather_block
    from src.parallel.runner import TAG_VERIFY
    asm = {}
    for uid in range(len(runner.blocks)):
        own = runner.owned_f_std_block(uid)
        asm[uid] = _gather_block(
            comm, rank, nr, runner.parts[uid],
            None if own is None else cp.asnumpy(own),
            TAG_VERIFY + 4 * uid, runner.n_pop, pre_sliced=True)
    if rank != 0:
        return
    print("[verify] building 1-rank reference...", flush=True)
    ref, _s, _a, ref_end = _build(args, 0, use_restart=False,
                                  dist_init=False, io_role='silent')
    n_adv = ref_end if n_advances is None else n_advances
    for _ in range(n_adv):
        ref.advance()
    ok = True
    for uid, rb in enumerate(ref.iter_blocks()):
        rf = cp.asnumpy(rb.sim.physical_f)
        d = np.abs(asm[uid] - rf)
        df = float(d.max())
        bit = bool(df == 0.0)
        ok = ok and (bit if args.strict_bit else df < 1e-4)
        print(f"[verify] {rb.label}: max|df|={df:.3e}  bit={bit}", flush=True)
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
