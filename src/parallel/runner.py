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

        # local slabs: extract views level-by-level and RELEASE each source
        # level's device arrays right after its slab is built — keeps the
        # transient peak at (t=0 build state + one slab), shrinking per level
        # (the f copies of the first version OOMed 24GB at D40).
        self.lv: List[LocalLevel] = []
        for k in range(NL):
            lev = mlg.get_level(k)
            ld = extract_level(lev)
            self.lv.append(LocalLevel(ld, self.parts[k]))
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
        self.ex = [HaloBandExchangerV1(self.parts[k], transport, cp,
                                       tag_base=16 * k)
                   for k in range(NL)]
        self.rlc = [RankLocalCouplingV1(couplings[k], self.parts[k],
                                        self.parts[k + 1], cp)
                    for k in range(NL - 1)]
        self._fprev = [None] * (NL - 1)
        self.NL = NL

        # distributed ALM (M3 hooks) on the finest level, if present
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
            cp.get_default_memory_pool().free_all_blocks()   # old global F_grid
        self.alm_lev = alm_lev

    # ── plumbing ─────────────────────────────────────────────────────
    def _sync(self, k: int) -> None:
        self.ex[k].post(self.lv[k].mem, self.lv[k].t)
        self.ex[k].complete(self.lv[k].mem, self.lv[k].t)

    def _save_fprev(self, k: int) -> None:
        reg = self.rlc[k].coarse_block_region_local()
        self._fprev[k] = (esoteric_gather_std_region(
            cp, self.lv[k].mem, self.lv[k].t, reg) if reg else None)

    def _advance_level(self, k: int) -> None:
        L = self.lv[k]
        if k == self.alm_lev and self.model is not None:
            L.macro_pre_pass()                      # rho/u for ALM sampling
            F = self.model.step(L.u, dt=1.0)        # collective inside
            L.advance(force=F.astype(cp.float32) if F.dtype != cp.float32
                      else F)
        else:
            L.advance()

    def _advance_fine(self, k: int) -> None:
        has_finer = k + 1 < self.NL
        if has_finer:
            self._sync(k); self._save_fprev(k)
        self._sync(k)
        self._advance_level(k)
        self._sync(k - 1)
        self.rlc[k - 1].c2f(self.lv[k - 1].mem, self.lv[k].mem,
                            self.lv[k - 1].t, self.lv[k].t,
                            is_half_step=True,
                            f_prev_sub_loc=self._fprev[k - 1])
        if has_finer:
            self._advance_fine(k + 1)
        if has_finer:
            self._sync(k); self._save_fprev(k)
        self._sync(k)
        self._advance_level(k)
        self.rlc[k - 1].c2f(self.lv[k - 1].mem, self.lv[k].mem,
                            self.lv[k - 1].t, self.lv[k].t,
                            is_half_step=False)
        if has_finer:
            self._advance_fine(k + 1)
        self._sync(k)
        self.rlc[k - 1].f2c(self.lv[k].mem, self.lv[k - 1].mem,
                            self.lv[k].t, self.lv[k - 1].t)

    # ── public API ───────────────────────────────────────────────────
    def step_coarse(self) -> None:
        """One L0 step (= 2^k substeps on level k), lockstep across ranks."""
        self._sync(0)
        if self.NL > 1:
            self._save_fprev(0)
        self._advance_level(0)
        if self.NL > 1:
            self._advance_fine(1)

    def run(self, n_coarse: int, log_every: int = 0, on_log=None) -> dict:
        t0 = time.perf_counter()
        for s in range(1, n_coarse + 1):
            self.step_coarse()
            if log_every and on_log and s % log_every == 0:
                cp.cuda.runtime.deviceSynchronize()
                on_log(s, self)
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        return {"steps": n_coarse, "wall_s": dt, "s_per_step": dt / n_coarse}

    def owned_f_std(self, k: int):
        """Owned slab of level k in standard physical ordering (GPU)."""
        return esoteric_gather_std(cp, self.lv[k].mem, self.lv[k].t)[
            (slice(None),) + self.parts[k].owned_local()]
