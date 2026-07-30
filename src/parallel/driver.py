"""Distributed (MPI) driver — the multi-rank branch of the unified entry.

Ported from the historical main_mpi.py (patch 17 M5) into src/ so the MPI
driver can never again be missed by an src-only deploy (patch_notes/
hpc_upgrade/20_mpi_result_csv_nut_vtk.md incident class).

    mpirun -n <R> python main.py --config <cfg.py> --steps <N> \
        [--axis auto|x|y|z] [--ghost 3] [--cuda-aware 0|1] \
        [--csv out.csv] [--log-every 8] [--verify]

One rank per GPU (device = node-local rank). Every rank runs the full
deterministic production build, keeps its slab, and enters the lockstep
distributed loop (src/parallel/runner.py). --verify gathers the owned
assemblies to rank 0 after the run and compares against a fresh single-rank
production MultiLevelGrid.advance() of the same case.

CUDA-awareness: --cuda-aware 1 passes CuPy buffers to MPI directly (cluster
OpenMPI 5.0.5 + UCX; requires the serialized thread level set BEFORE mpi4py
import). --cuda-aware 0 stages through host numpy (any MPI, e.g. local
MPICH functional smoke). Default: LBM_MPI_CUDA env (0 if unset).

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
           use_restart: bool = True, dist_init=None):
    """Replicated production build. No sys.argv shim — a Namespace copy of
    the unified args, so directory/clear flags now reach setup too."""
    di = args.dist_init if dist_init is None else dist_init
    os.environ["LBM_DIST_INIT"] = "1" if di else "0"
    bargs = argparse.Namespace(**vars(args))
    bargs.gpu = dev
    # ForceManager needs full-domain f_post; MPI body forces come from the
    # esoteric MEM kernel. Scheduled to flip in stage C5 (forces_override).
    bargs.no_force = True
    bargs.no_vtk = (not with_writers) or args.no_vtk
    if not use_restart:
        bargs.restart = None
        bargs.restart_latest = False
    from src.solver.setup import SimulationSetup
    from src.solver.initializer import SolverInitializer
    s = SimulationSetup(bargs)
    s.start_log_capture()
    mlg = s.build_simulation()
    SolverInitializer(s).initialize(mlg, bargs)
    s.stop_log_capture()
    _fail_fast_config(s, mlg)
    mlg._graph_enabled = False
    return mlg, s


def _run(args, MPI):
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

    want_out = bool(vtk_every or ckpt_every)
    if rank != 0:                                 # quiet non-root builds
        sys.stdout = open(os.devnull, "w")
    if args.dist_init and (args.restart or args.restart_latest):
        raise ValueError("--dist-init + restart: restore path loads full "
                         "fields (use replicated build for restarts for now)")
    mlg, setup = _build(args, dev,
                        with_writers=(rank == 0 and want_out))
    sys.stdout = sys.__stdout__

    from src.parallel import MPITransport
    from src.parallel.alm_dist import MPIAllreduce
    from src.parallel.runner import DistributedMLGRunner

    axis = None if args.axis == "auto" else "xyz".index(args.axis)
    transport = MPITransport(comm, cuda_aware=str(cuda_aware) == "1")
    runner = DistributedMLGRunner(
        mlg, transport, rank, nr, allreduce=MPIAllreduce(comm),
        axis=axis, ghost=args.ghost)

    def _collect_solid_masks(mlg_obj, n_levels):
        """Static per-level solid masks (host numpy) for the VTK bridge."""
        masks = []
        for k in range(n_levels):
            ob = getattr(mlg_obj.get_level(k), "obstacle_bc", None)
            sm = getattr(ob, "solid_mask", None) if ob is not None else None
            if sm is not None and hasattr(sm, "get"):
                sm = sm.get()
            masks.append(sm)
        return masks

    # output bridge (production writers) BEFORE freeing the build
    from src.parallel.output import Rank0OutputBridge
    bridge = None
    if want_out:
        bridge = Rank0OutputBridge(
            comm, rank, nr,
            mlg_vtk_writer=(getattr(setup, "_mlg_vtk_writer", None)
                            if rank == 0 else None),
            checkpoint_mgr=(getattr(setup, "checkpoint_mgr", None)
                            if rank == 0 else None),
            sim_params=(getattr(setup, "sim_params", None)
                        if rank == 0 else None),
            tau=float(mlg.get_level(0).tau),
            marker_vtk_writer=(getattr(setup, "marker_vtk_writer", None)
                               if rank == 0 else None),
            alm_marker_origin=(getattr(setup, "_alm_marker_origin", None)
                               if rank == 0 else None),
            alm_marker_spacing=(getattr(setup, "_alm_marker_spacing", None)
                                if rank == 0 else None),
            solid_masks=(_collect_solid_masks(mlg, runner.NL)
                         if rank == 0 else None))

    # body-tier normalization source: keep the reference dict from the
    # ALREADY-PARSED setup.config — re-executing the config module later
    # would double-run any config side effects.
    _force_ref = dict(getattr(setup, "config", {})
                      .get("force_calculation", {}).get("reference", {}))

    # result-folder rotor CSVs (rank 0): setup wrote the headers; the MPI
    # loop must append the rows itself (OutputManager.process never runs
    # here — the 0718/0721 result folders were header-only because of this).
    perf_csv_path = (getattr(setup, "perf_csv_path", None)
                     if rank == 0 else None)
    blade_csv_dir = (getattr(setup, "blade_csv_dir", None)
                     if rank == 0 else None)

    # free the full build (runner kept slabs + the ALM model)
    import gc
    del mlg, setup
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    start_step = runner.completed_step + 1
    if start_step > args.steps:
        if rank == 0:
            print(f"[mpi] nothing to do: restored step {start_step - 1} "
                  f">= --steps {args.steps}", flush=True)
        MPI.Finalize()
        return
    csv_f = None
    if rank == 0:
        print(f"[mpi] ranks={nr} axis={'xyz'[runner.axis]} "
              f"ghost={args.ghost} cuda_aware={cuda_aware}"
              + (f"  RESTART from {runner.completed_step}"
                 if runner.completed_step else ""), flush=True)
        for k, p in enumerate(runner.parts):
            print(f"[mpi]   L{k} rank0 {p}", flush=True)
        if args.csv:
            append = runner.completed_step > 0 and os.path.exists(args.csv)
            csv_f = open(args.csv, "a" if append else "w")
            if not append:
                if runner.model is not None:
                    csv_f.write("step,time_lu,thrust,torque,power,"
                                "C_T,C_P,FM\n")
                elif runner.body_level is not None:
                    csv_f.write("step,Fx,Fy,Fz,CD,CL,CS\n")
                else:
                    csv_f.write("step,rho_mean,u_max\n")

    # case tier for the progress line: ALM -> CT/CP/FM; solid body ->
    # CD/CL/CS from the esoteric MEM-force kernel; plain flow ->
    # rho_mean / u_max on the finest level.
    if runner.model is not None:
        tier = "alm"
    elif runner.body_level is not None:
        tier = "body"
        # normalization: force_calculation.reference in the BODY level's
        # lattice units (production 3D convention A = char_length * span)
        _q_rho = float(_force_ref.get("rho", 1.0))
        _q_u = float(_force_ref.get("velocity", 0.05))
        _q_A = float(_force_ref.get("char_length", 1.0)) * \
            float(_force_ref.get("span_length", 1.0))
        _qA = 0.5 * _q_rho * _q_u * _q_u * _q_A
    else:
        tier = "flow"

    def flow_stats():
        """COLLECTIVE: owned FLUID-cell rho mean + |u| max on the finest
        level. SOLID cells are masked — the kernel returns before writing
        rho/u there, so their buffers are UNINITIALIZED."""
        L = runner.lv[-1]
        sl = runner.parts[-1].owned_local()
        fluid = (L.nt.reshape(L.dims)[sl] != 1)
        rho = L.rho[sl]
        u = L.u[(slice(None),) + sl]
        s_loc = float(rho[fluid].sum())
        n_loc = float(fluid.sum())
        usq = (u * u).sum(axis=0)
        usq = cp.where(fluid, usq, 0.0)
        m_loc = float(cp.sqrt(usq.max()))
        if nr > 1:
            import numpy as _np
            buf = _np.array([s_loc, n_loc, m_loc])
            out = _np.empty_like(buf)
            comm.Allreduce(buf[:2], out[:2])           # sum
            comm.Allreduce(buf[2:], out[2:], op=MPI.MAX)
            s_loc, n_loc, m_loc = out[0], out[1], out[2]
        return s_loc / max(n_loc, 1.0), m_loc

    def on_log(s, rn, stats=None):
        if rank != 0:
            return
        line = f"[mpi] step {s}/{args.steps}"
        if rn.last_interval:
            sps = rn.last_interval["s_per_step"]
            eta_h = (args.steps - s) * sps / 3600.0
            line += f"  {sps:.3f}s/step  ETA {eta_h:.2f}h"
        if tier == "alm":
            perf = rn.model.get_rotor_performance()
            line += (f"  CT={perf['C_T']:.6e}  CP={perf['C_P']:.6e}"
                     f"  FM={perf['FM']:.4f}")
            if csv_f:
                csv_f.write(f"{s},{perf['time']},{perf['thrust']},"
                            f"{perf['torque']},{perf['power']},"
                            f"{perf['C_T']},{perf['C_P']},{perf['FM']}\n")
                csv_f.flush()
            # result-folder CSVs (same schema/cadence as the single-GPU loop)
            if perf_csv_path:
                from src.solver.output_manager import (
                    log_rotor_performance_row, log_blade_diagnostics_rows)
                log_rotor_performance_row(rn.model, perf_csv_path, s)
                if blade_csv_dir:
                    log_blade_diagnostics_rows(rn.model, blade_csv_dir, s)
        elif tier == "body" and stats is not None:
            F = stats
            cd, cl, cs = F[0] / _qA, F[1] / _qA, F[2] / _qA
            line += f"  CD={cd:.4f}  CL={cl:.4f}  CS={cs:.4f}"
            if csv_f:
                csv_f.write(f"{s},{F[0]},{F[1]},{F[2]},{cd},{cl},{cs}\n")
                csv_f.flush()
        elif stats is not None:
            rho_m, u_max = stats
            line += f"  rho={rho_m:.6f}  u_max={u_max:.5f}"
            if csv_f:
                csv_f.write(f"{s},{rho_m},{u_max}\n")
                csv_f.flush()
        print(line, flush=True)

    if args.profile:
        runner.profile = {}

    # main loop (collective output cadences need every rank in lockstep)
    import time as _time
    t0 = _time.perf_counter()
    t_last = t0
    runner.last_interval = None
    n_todo = args.steps - start_step + 1
    for s_ in range(start_step, args.steps + 1):
        runner.step_coarse()
        if log_every and s_ % log_every == 0:
            cp.cuda.runtime.deviceSynchronize()
            now = _time.perf_counter()
            runner.last_interval = {
                "s_per_step": (now - t_last) / log_every,
                "elapsed_s": now - t0}
            t_last = now
            if tier == "body":
                F_loc = runner.mem_force_local()
                if nr > 1:
                    import numpy as _np
                    F_tot = _np.empty_like(F_loc)
                    comm.Allreduce(F_loc, F_tot)
                else:
                    F_tot = F_loc
                stats = F_tot
            elif tier == "flow":
                stats = flow_stats()
            else:
                stats = None
            on_log(s_, runner, stats)
        if bridge is not None and vtk_every and s_ % vtk_every == 0:
            # --vtk-fields-last N: level-field files only for the last N vtk
            # events (the ~GB/snapshot cost); marker VTPs keep the full
            # rev-locked history.
            fields = (args.vtk_fields_last <= 0 or
                      s_ > args.steps - args.vtk_fields_last * vtk_every)
            bridge.write_vtk(s_, runner, fields=fields)
        if bridge is not None and ckpt_every and s_ % ckpt_every == 0:
            bridge.save_checkpoint(s_, runner)
    cp.cuda.runtime.deviceSynchronize()
    dt_all = _time.perf_counter() - t0
    stats = {"steps": n_todo, "wall_s": dt_all,
             "s_per_step": dt_all / max(n_todo, 1)}
    comm.Barrier()
    if args.profile:
        prof = comm.gather(runner.profile, root=0)
        if rank == 0:
            keys = sorted({k for d in prof for k in d})
            print("[profile] per-rank seconds over the whole run:",
                  flush=True)
            for k in keys:
                vals = " ".join(f"{d.get(k, 0.0):7.2f}" for d in prof)
                print(f"[profile]   {k:14s} {vals}", flush=True)
    if rank == 0:
        print(f"[mpi] done: {stats['steps']} coarse steps, "
              f"{stats['wall_s']:.2f}s ({stats['s_per_step']:.3f} s/step)",
              flush=True)
        if csv_f:
            csv_f.close()

    if args.verify:
        _verify(comm, rank, nr, runner, args)
    MPI.Finalize()


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
