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
    ap.add_argument("--profile", action="store_true",
                    help="per-section wall-time attribution (adds sync overhead)")
    ap.add_argument("--restart", default=None,
                    help="checkpoint file to resume from (absolute steps)")
    ap.add_argument("--restart-latest", action="store_true",
                    help="resume from the latest checkpoint")
    ap.add_argument("--dist-init", action="store_true",
                    help="slab-scoped initialization: no full-size device "
                         "fields per rank (uniform-IC cases; lifts 4-GPU "
                         "capacity to ~4x a single GPU)")
    ap.add_argument("--vtk-every", type=int, default=0,
                    help="coarse steps between assembled VTK writes (0=off)")
    ap.add_argument("--vtk-fields-last", type=int, default=0,
                    help="write level-field VTK only for the last N vtk "
                         "events; marker VTPs still every --vtk-every (0=all)")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="coarse steps between assembled checkpoints (0=off)")
    ap.add_argument("--strict-bit", action="store_true",
                    help="verify passes ONLY on bit-identity (pure-LBM cases; "
                         "reviewer F-4)")
    return ap.parse_args(argv)


def build(config, dev: int, with_writers: bool = False,
          restart=None, restart_latest=False, dist_init=False):
    os.environ["LBM_DIST_INIT"] = "1" if dist_init else "0"
    sys.argv = ["mpi", "--config", config, "--gpu", str(dev), "--no-force"]
    if not with_writers:
        sys.argv.append("--no-vtk")      # writers only needed on rank 0
    if restart:
        sys.argv += ["--restart", restart]
    elif restart_latest:
        sys.argv.append("--restart-latest")
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
    return mlg, s


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

    want_out = bool(args.vtk_every or args.ckpt_every)
    if rank != 0:                                 # quiet non-root builds
        sys.stdout = open(os.devnull, "w")
    if args.dist_init and (args.restart or args.restart_latest):
        raise ValueError("--dist-init + restart: restore path loads full "
                         "fields (use replicated build for restarts for now)")
    mlg, setup = build(args.config, dev,
                       with_writers=(rank == 0 and want_out),
                       restart=args.restart,
                       restart_latest=args.restart_latest,
                       dist_init=args.dist_init)
    sys.stdout = sys.__stdout__

    from src.parallel import MPITransport
    from src.parallel.alm_dist import MPIAllreduce
    from src.parallel.runner import DistributedMLGRunner

    axis = None if args.axis == "auto" else "xyz".index(args.axis)
    transport = MPITransport(comm, cuda_aware=str(args.cuda_aware) == "1")
    runner = DistributedMLGRunner(
        mlg, transport, rank, nr, allreduce=MPIAllreduce(comm),
        axis=axis, ghost=args.ghost)

    def _collect_solid_masks(mlg_obj, n_levels):
        """Static per-level solid masks (host numpy) for the VTK bridge.

        Captured BEFORE the full build is freed; pre-fix the bridge
        hardwired obstacle_bc=None and MPI VTK silently lost solid_mask."""
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

    # body-tier normalization source (review #3, R3-4.3): keep the reference
    # dict from the ALREADY-PARSED setup.config — re-executing the config
    # module later would double-run any config side effects.
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
    if rank == 0:
        print(f"[mpi] ranks={nr} axis={'xyz'[runner.axis]} "
              f"ghost={args.ghost} cuda_aware={args.cuda_aware}"
              + (f"  RESTART from {runner.completed_step}"
                 if runner.completed_step else ""), flush=True)
        for k, p in enumerate(runner.parts):
            print(f"[mpi]   L{k} rank0 {p}", flush=True)
        csv_f = None
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
    # CD/CL/CS from the esoteric MEM-force kernel (halfway-BB convention,
    # same formula as the standard-path mem_force_d3q27); plain flow ->
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
        rho/u there, so their buffers are UNINITIALIZED (caught by the
        sphere smoke: a rank reported u_max=1.8 from cp.empty garbage)."""
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
        if args.log_every and s_ % args.log_every == 0:
            cp.cuda.runtime.deviceSynchronize()
            now = _time.perf_counter()
            runner.last_interval = {
                "s_per_step": (now - t_last) / args.log_every,
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
        if bridge is not None and args.vtk_every and \
                s_ % args.vtk_every == 0:
            # --vtk-fields-last N: level-field files only for the last N vtk
            # events (the ~GB/snapshot cost); marker VTPs keep the full
            # rev-locked history (the spanwise/convergence analysis input).
            fields = (args.vtk_fields_last <= 0 or
                      s_ > args.steps - args.vtk_fields_last * args.vtk_every)
            bridge.write_vtk(s_, runner, fields=fields)
        if bridge is not None and args.ckpt_every and \
                s_ % args.ckpt_every == 0:
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
            print("[profile] per-rank seconds over the whole run:", flush=True)
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
    ref, _ = build(args.config, 0)          # dist_init=False: full fields
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


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)      # fail-fast: never leave peers hanging
