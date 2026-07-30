"""SPMD distributed MLG runner (patch 17 M5).

One process = one rank = one GPU. Each rank performs the FULL deterministic
production build (SimulationSetup + SolverInitializer — bit-identical state
on every rank), keeps only its wrap-sliced slab per level plus the rank's
ActuatorLineModel, frees the rest, then drives the gate-proven distributed
loop: per-level v1 halo exchange + rank-local MLG coupling + distributed ALM
(M3 partial-sum allreduce). The recursion below is the SPMD restructuring of
gates G-M2b/M3's run_distributed (loops over ranks removed — every rank
executes the same sequence in lockstep; collectives meet inside step()).

Memory note: the full-domain build peaks like a single-GPU run of the same
case (each rank has its own GPU, so this is admissible whenever the case
fits one GPU — D40 does). Cases beyond single-GPU size need a distributed
initializer (documented follow-up, not needed for the M5 validation).
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import cupy as cp

from src.parallel.partition import Partition1D, choose_axis
from src.parallel.halo import HaloBandExchangerV1
from src.parallel.local_level import LocalLevel, extract_level
from src.parallel.mlg_coupling import (
    RankLocalCouplingV1, fine_range_from_coarse)
from src.parallel.alm_dist import make_distributed_sampler
from src.kernels.esoteric_d3q27 import (
    esoteric_gather_std, esoteric_gather_std_region)


class DistributedMLGRunner:
    """Drive ONE rank of a decomposed MLG run (transport = the wire)."""

    def __init__(self, mlg, transport, rank: int, n_ranks: int,
                 allreduce=None, axis: Optional[int] = None,
                 ghost: int = 3) -> None:
        self.rank, self.nr = rank, n_ranks
        if n_ranks == 1:
            # Single rank needs NO halo at all: with ghost=0 the local array
            # IS the domain and the kernel's %N wrap IS the periodicity —
            # identical to the production single-GPU path. (The previous
            # ghost=3 self-exchange was correct but paid the MPI transport;
            # without mpirun/UCX the ob1 fallback made it ~7x slower.)
            ghost = 0
        NL = mlg.num_levels
        couplings = mlg._couplings
        shapes = [mlg.get_level(k).domain_shape for k in range(NL)]
        boxes = []
        for k in range(NL - 1):
            fdc = couplings[k]._region.fine_domain_coarse
            boxes.append(((fdc.x_start, fdc.x_end),
                          (fdc.y_start, fdc.y_end),
                          (fdc.z_start, fdc.z_end)))
        from src.parallel.partition import (
            choose_axis_balanced, balance_cuts)
        if axis is None:
            self.axis, bounds = choose_axis_balanced(
                shapes, boxes, n_ranks, ghost)
        else:
            self.axis = int(axis)
            bounds = balance_cuts(shapes, boxes, self.axis, n_ranks, ghost)
        self.bounds = bounds

        # partitions: L0 load-balanced cuts (all inside the innermost box
        # span -> every rank owns every level), finer cuts DERIVED down
        self.parts: List[Partition1D] = [
            Partition1D.from_range(shapes[0], n_ranks, rank, self.axis,
                                   bounds[rank], bounds[rank + 1] -
                                   bounds[rank], ghost=ghost)]
        for k in range(1, NL):
            lo, hi = boxes[k - 1][self.axis]
            nf = couplings[k - 1]._region.fine_shape[self.axis]
            f0_, fc_ = fine_range_from_coarse(self.parts[k - 1], lo, hi, nf)
            if fc_ < ghost:
                raise ValueError(
                    f"L{k} rank{rank}: own={fc_} < ghost={ghost} on axis "
                    f"{self.axis} — pick another axis or fewer ranks")
            self.parts.append(Partition1D.from_range(
                shapes[k], n_ranks, rank, self.axis, f0_, fc_, ghost=ghost))

        # distributed ALM (M3 hooks) on the finest level — wired BEFORE the
        # slab loop so the model's GLOBAL-shape f64 F_grid (2.5GB at 400M
        # scale) is replaced and freed before slabs start accumulating
        self.model = None
        alm_lev = NL - 1
        if getattr(mlg.get_level(alm_lev), "al_model", None) is not None:
            if allreduce is None:
                raise ValueError("ALM case needs an allreduce adapter")
            m = mlg.get_level(alm_lev).al_model
            p = self.parts[alm_lev]
            off = np.zeros(3)
            off[self.axis] = p.own_start - p.ghost
            m._global_domain_shape = tuple(shapes[alm_lev])
            m.domain_shape = tuple(p.local_shape)
            m._F_grid = cp.zeros((3,) + tuple(p.local_shape), cp.float64)
            m._grid_offset = off
            m._velocity_sampler = make_distributed_sampler(
                allreduce, rank, p, cp)
            self.model = m
            cp.get_default_memory_pool().free_all_blocks()
        self.alm_lev = alm_lev

        # restart: continue esoteric parity + step numbering from the
        # restored state. The initializer sets L0 step_count = start_step
        # = the number of coarse advances already done (0-based exclusive
        # step convention, unified in stage C8) — completed_step IS that
        # count. The old `max(0, sc-1)` belonged to the legacy 1-based
        # main_mpi labels; under the unified convention it desynced the
        # esoteric parity t0 for single-GPU-written checkpoints.
        sc = int(getattr(mlg.get_level(0), "step_count", 0) or 0)
        self.completed_step = sc

        # local slabs: extract views level-by-level and RELEASE each source
        # level's device arrays right after its slab is built — keeps the
        # transient peak at (t=0 build state + one slab), shrinking per level
        # (the f copies of the first version OOMed 24GB at D40).
        self.lv: List[LocalLevel] = []
        for k in range(NL):
            lev = mlg.get_level(k)
            ld = extract_level(lev)
            self.lv.append(LocalLevel(ld, self.parts[k],
                                      t0=self.completed_step * (2 ** k)))
            del ld
            for a in ("f", "f_post", "f_prev", "rho", "u",
                      "_eso_node_type", "_eso_bc_rho", "_eso_bc_ux",
                      "_eso_bc_uy", "_eso_bc_uz"):
                if hasattr(lev, a):
                    try:
                        setattr(lev, a, None)
                    except AttributeError:
                        pass                     # read-only property
            cp.get_default_memory_pool().free_all_blocks()
        self.ex = None if n_ranks == 1 else \
            [HaloBandExchangerV1(self.parts[k], transport, cp,
                                 tag_base=16 * k)
             for k in range(NL)]
        self.rlc = [RankLocalCouplingV1(couplings[k], self.parts[k],
                                        self.parts[k + 1], cp)
                    for k in range(NL - 1)]
        self._fprev = [None] * (NL - 1)
        self.NL = NL
        self.profile = None          # dict -> per-section seconds (opt-in)
        # halo scheduling state (backlog #5): ghosts_fresh -> skip idempotent
        # re-exchanges of unchanged mem (the A-syncs after a same-cycle
        # D-sync, ~30% of all rounds); posted -> band already in flight
        # (early post fired right after the last mutation, so the transfer
        # hides under the child's fprev+advance)
        self._fresh = [False] * NL
        self._posted = [False] * NL
        # solid-body diagnostics: finest level carrying SOLID nodes
        self.body_level = None
        for k in range(NL - 1, -1, -1):
            if bool((self.lv[k].nt == 1).any()):
                self.body_level = k
                break

    # ── plumbing ─────────────────────────────────────────────────────
    def _tic(self):
        if self.profile is not None:
            cp.cuda.get_current_stream().synchronize()
            return time.perf_counter()
        return 0.0

    def _toc(self, key, t0):
        if self.profile is not None:
            cp.cuda.get_current_stream().synchronize()
            self.profile[key] = self.profile.get(key, 0.0) \
                + time.perf_counter() - t0

    def _touch(self, k: int) -> None:
        assert not self._posted[k], \
            f"L{k} mutated with a band in flight (scheduling bug)"
        self._fresh[k] = False

    def _post(self, k: int) -> None:
        """Early post: bands gathered from the FINAL state before the next
        consumer; MPITransport also pre-posts the Irecv (true overlap)."""
        if self.ex is None or self._fresh[k] or self._posted[k]:
            return
        t0 = self._tic()
        self.ex[k].post(self.lv[k].mem, self.lv[k].t)
        self._posted[k] = True
        self._toc("halo_post", t0)

    def _sync(self, k: int) -> None:
        if self.ex is None or self._fresh[k]:
            return                    # NR=1: no halo / unchanged: idempotent
        if not self._posted[k]:
            t0 = self._tic()
            self.ex[k].post(self.lv[k].mem, self.lv[k].t)
            self._toc("halo_post", t0)
        t0 = self._tic()
        self.ex[k].complete(self.lv[k].mem, self.lv[k].t)
        self._posted[k] = False
        self._fresh[k] = True
        self._toc("halo_complete", t0)

    def _save_fprev(self, k: int) -> None:
        t0 = self._tic()
        reg = self.rlc[k].coarse_block_region_local()
        self._fprev[k] = (esoteric_gather_std_region(
            cp, self.lv[k].mem, self.lv[k].t, reg) if reg else None)
        self._toc("fprev", t0)

    def _advance_level(self, k: int) -> None:
        L = self.lv[k]
        if k == self.alm_lev and self.model is not None:
            t0 = self._tic()
            L.macro_pre_pass()                      # rho/u for ALM sampling
            F = self.model.step(L.u, dt=1.0)        # collective inside
            self._toc("alm", t0)
            t0 = self._tic()
            L.advance(force=F.astype(cp.float32) if F.dtype != cp.float32
                      else F)
            self._toc("kernel", t0)
        else:
            t0 = self._tic()
            L.advance()
            self._toc("kernel", t0)
        self._touch(k)

    def _advance_fine(self, k: int) -> None:
        # NOTE sync cadence: ghosts must be fresh for (a) the fprev gather
        # (block extends into ghosts), (b) the kernel, (c) the coupling
        # reads. One sync covers fprev+advance back-to-back — the mem is
        # unchanged between them (the first version synced twice; exchange
        # of unchanged mem is idempotent, so dropping it is bit-neutral).
        has_finer = k + 1 < self.NL
        self._sync(k)
        if has_finer:
            self._save_fprev(k)
        self._advance_level(k)
        self._sync(k - 1)
        t0 = self._tic()
        self.rlc[k - 1].c2f(self.lv[k - 1].mem, self.lv[k].mem,
                            self.lv[k - 1].t, self.lv[k].t,
                            is_half_step=True,
                            f_prev_sub_loc=self._fprev[k - 1])
        self._toc("coupling", t0)
        self._touch(k)                # c2f wrote our strips
        if has_finer:
            self._post(k)             # child completes at its B-sync while
            self._advance_fine(k + 1)  # ...its fprev+advance run
        self._sync(k)
        if has_finer:
            self._save_fprev(k)
        self._advance_level(k)
        t0 = self._tic()
        self.rlc[k - 1].c2f(self.lv[k - 1].mem, self.lv[k].mem,
                            self.lv[k - 1].t, self.lv[k].t,
                            is_half_step=False)
        self._toc("coupling", t0)
        self._touch(k)
        if has_finer:
            self._post(k)
            self._advance_fine(k + 1)
        self._sync(k)
        t0 = self._tic()
        self.rlc[k - 1].f2c(self.lv[k].mem, self.lv[k - 1].mem,
                            self.lv[k].t, self.lv[k - 1].t)
        self._toc("coupling", t0)
        self._touch(k - 1)            # f2c wrote the coarse excised rows

    # ── public API ───────────────────────────────────────────────────
    def step_coarse(self) -> None:
        """One L0 step (= 2^k substeps on level k), lockstep across ranks."""
        self._sync(0)
        if self.NL > 1:
            self._save_fprev(0)
        self._advance_level(0)
        if self.NL > 1:
            self._post(0)             # L0 bands ride under L1's advance
            self._advance_fine(1)

    def run(self, n_coarse: int, log_every: int = 0, on_log=None) -> dict:
        t0 = time.perf_counter()
        t_last = t0
        self.last_interval = None      # {"s_per_step", "elapsed_s"} per log
        for s in range(1, n_coarse + 1):
            self.step_coarse()
            if log_every and on_log and s % log_every == 0:
                cp.cuda.runtime.deviceSynchronize()
                now = time.perf_counter()
                self.last_interval = {
                    "s_per_step": (now - t_last) / log_every,
                    "elapsed_s": now - t0,
                }
                t_last = now
                on_log(s, self)
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        return {"steps": n_coarse, "wall_s": dt, "s_per_step": dt / n_coarse}

    def mem_force_local(self):
        """OWNED-cell MEM force partial on the body level (lattice units of
        that level); allreduce + normalization happen in the caller.
        Diagnostic tier (atomicAdd accumulation)."""
        from src.kernels.esoteric_d3q27 import eso_mem_force
        k = self.body_level
        L = self.lv[k]
        p = self.parts[k]
        cb = [(0, d) for d in L.dims]
        cb[p.axis] = (p.ghost, p.ghost + p.own_count)
        return eso_mem_force(cp, L.mem, L.nt, L.t, cb)

    def owned_f_std(self, k: int):
        """Owned slab of level k in standard physical ordering (GPU)."""
        return esoteric_gather_std(cp, self.lv[k].mem, self.lv[k].t)[
            (slice(None),) + self.parts[k].owned_local()]
