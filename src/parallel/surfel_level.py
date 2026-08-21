"""Rank-local surfel level — slab-scoped V1 residency bridge (patch 64).

The MPI stack's LocalLevel is the esoteric fused stepper; surfel levels
run the VALIDATED bridge chain instead (patch 63: gather -> std surfel
advance -> scatter), instantiated on the slab shape. This adapter gives
the runner the LocalLevel interface surface it actually uses (mem, t,
advance, nt, nt_c2f/nt_f2c, wall_mask/mail/args, dims) while a slab
Simulation does the physics — zero new physics code.

Halo contract: the runner syncs BEFORE every advance and the halo is a
wrap-window slice exchange of eso mem — gate s14 E6 proved bit-exact own
strips under exactly that model (the 'perfect halo' arm). The surfel
stencil chain consumes 4 ghost cells per substep (advect 1 + facet cell
spread 2 + trilinear 1) — the runner enforces ghost >= 4 for surfel
levels (patch 64 sec. 8).
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from src.boundary.surfel_eso import build_slab_surfel


class SurfelSlabLevel:
    """LocalLevel-interface adapter over a slab-scoped bridge Simulation."""

    def __init__(self, lev, part, t0: int = 0) -> None:
        import os
        if os.environ.get('LBM_ESOTERIC', '0') != '1':
            raise RuntimeError(
                "surfel MPI runs on the eso residency bridge — launch "
                "with LBM_ESOTERIC=1 (patch_notes/surfel/63)")
        n_ax = part.global_shape[part.axis]
        self._full = part.own_count == n_ax          # NR=1 (ghost=0) path
        sb_full = lev.obstacle_bc
        if self._full:
            # single-rank: the replicated build's sim IS the validated
            # bridge — reuse it whole (bit-trivially the 1-GPU path)
            if not getattr(lev, '_use_esoteric', False):
                raise RuntimeError(
                    "single-rank surfel slab expects the bridged sim")
            self.sim = lev
            self.sb = sb_full
            self.surfel_live = sb_full.d_live
            live_h = sb_full.live_h
        else:
            # f window slice FIRST (64 sec. 15): the full-domain f
            # (216 B/node, 6.35 GiB at span16 L3) used to stay resident
            # through the whole slab-surfel build while only its window
            # survives it. Slicing here and dropping the full array buys
            # the build the full-minus-window difference in headroom;
            # the eso/std conversion itself stays where it was (after
            # the full-surfel release below).
            idx = cp.asarray(
                np.arange(part.own_start - part.ghost,
                          part.own_start + part.own_count + part.ghost)
                % n_ax)
            take = [slice(None)] * 4
            take[1 + part.axis] = idx
            f_slab = cp.ascontiguousarray(lev.f[tuple(take)])
            was_eso = bool(getattr(lev, '_use_esoteric', False))
            lev.f = None
            cp.get_default_memory_pool().free_all_blocks()

            self.sb = build_slab_surfel(
                sb_full, part.axis, part.own_start, part.own_count,
                ghost=part.ghost, consume=True)
            slab_shape = self.sb.shape
            self.surfel_live = self.sb.d_live
            live_h = self.sb.live_h

            # tau-band margin exchange wires (patch 64 stage ii) — must
            # precede the full-surfel release below (reads full CSR / W)
            self.taum = None
            self._tt = None
            if getattr(self.sb, 'tau_model_on', False):
                self._taum_wire(sb_full, part)

            # Release the FULL-build surfel DEVICE arrays (64 sec. 13/14):
            # g_field alone is 216 B/node full-domain. Ordering is load-
            # bearing — releasing BEFORE the slab mem conversion keeps
            # the L3 peak from stacking (full surfel + slab arrays +
            # conversion transient) on one card, which was the second
            # span16 OOM. Host state (live_h, dV_h, facets) survives for
            # the driver's solid-mask collection.
            k_full = sb_full.kernel
            for a_ in ('g_field', 'Q', 'indptr', 'cell', 'wgt', 'nrm',
                       'area', 'cen', 'Vsum', 'G_in', 'G_out', 'tau_out',
                       'fb_out', 'u_wm', 'utau_prev'):
                setattr(k_full, a_, None)
            for a_ in ('d_live', 'd_dead', 'd_dV', '_solid_mask_dev',
                       '_tb_W', 'd_tb_cells', 'd_tb_fs', 'd_tb_normal',
                       '_tb_tau_ext'):
                if hasattr(sb_full, a_):
                    setattr(sb_full, a_, None)
            cp.get_default_memory_pool().free_all_blocks()

            if was_eso:
                # window slice of the full eso mem (taken above):
                # interior-identical to a slab-local conversion; the
                # ghost-rim slot difference is refreshed by the first
                # sync (E6 halo model)
                mem = f_slab
            else:
                # MPI build keeps f STANDARD (64 sec. 13): convert THIS
                # slab only. Parity t0 is even by construction (fresh 0;
                # restarts are even-checkpoint-only — 64 sec. 12 G5).
                from src.kernels.esoteric_d3q27 import esoteric_scatter_std
                mem = esoteric_scatter_std(cp, f_slab, t0)
            del f_slab

            from src.solver.simulation import Simulation
            from src.boundary.domain_bc_manager import DomainBCManager

            class _NoStream:
                def compute(self, *a, **k):
                    raise AssertionError(
                        "surfel slab called streaming.compute")

            bc = DomainBCManager.for_shape(
                lev.bc_manager, lev.xp, lev.bc_manager.lattice, slab_shape)
            self.sim = Simulation(
                xp=lev.xp, macroscopic=lev.macroscopic,
                collision=lev.collision, streaming=_NoStream(),
                bc_manager=bc, tau=lev.tau, domain_shape=slab_shape,
                obstacle_bc=self.sb,
                sgs_cfg=getattr(lev, '_sgs_cfg', None),
            )
            self.sim.adopt_esoteric_surfel_slab(mem, t0)

        if self._full:
            self.taum = None
            self._tt = None

        self.dims = tuple(int(d) for d in self.sim.domain_shape)
        # runner diagnostics: nt==1 marks SOLID (body_block scan)
        self.nt = cp.asarray((~np.asarray(live_h, dtype=bool))
                             .astype(np.int8).ravel())
        # coupling skip inputs — mirror the single-GPU bridge exactly:
        # coupling_skip_nt falls through to None there (no _eso_node_type
        # on bridge sims), and the F2C solid handling is the surfel merge
        # branch (patch-50 semantics), not skip_solid_nt.
        self.nt_c2f = None
        self.nt_f2c = None
        self.wall_mask = 0
        self.wall_mail = None

    # ── LocalLevel interface surface used by the runner ──────────────
    @property
    def mem(self):
        return self.sim.f

    @property
    def t(self) -> int:
        return self.sim._esoteric_step

    @property
    def rho(self):
        """Macro state of the LAST bridge advance (VTK gather seam)."""
        return self.sim.rho

    @property
    def u(self):
        return self.sim.u

    @property
    def wall_args(self) -> dict:
        return {'wall_mask': self.wall_mask, 'wall_mail': self.wall_mail}

    def advance(self, force=None) -> None:
        if force is not None:
            raise NotImplementedError(
                "surfel slab + ALM body force is S8b+ scope")
        self.sim.advance()
        # Per-advance scratch discipline (64 sec. 16): Q (216 B/node
        # f64, fill(0)-ed at every apply) and _f_post (fully rewritten
        # by the collide kernel every advance) carry NO cross-substep
        # state — both re-materialize through their existing lazy paths.
        # Left resident they stack per LEVEL (Q+f_post of L0-L2 alone
        # was 5.1 GiB under the L3 advance = the span16 run-state OOM);
        # freed here the run peak holds one level's scratch, and the
        # pool round-trip is noise next to the advance itself.
        self.sim._f_post = None
        self.sb.kernel.Q = None
        # ...and RETURN the freed blocks to the device (64 sec. 17): the
        # census pinned the 4th span16 OOM on pool-HELD churn, not live
        # arrays — L0-L2 advances leave ~6.5 GiB of freed Q/std/f_post
        # fragments (largest 1.85) that can never serve L3's contiguous
        # 3.78 GiB f64 Q, while they starve cudaMalloc of device room
        # (cluster CuPy raises without an effective free-and-retry).
        # Cost: tens of large cudaMalloc/Free per coarse step — noise
        # next to the advances themselves.
        cp.get_default_memory_pool().free_all_blocks()

    # ── tau-band margin exchange (patch 64 stage ii) ─────────────────
    def _taum_wire(self, sb_full, part) -> None:
        from src.boundary.surfel_eso import (
            band_needed_gids, facet_anchor_axis)
        a, g = part.axis, part.ghost
        counts = np.asarray(getattr(part, '_counts'), dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
        n_ax = sb_full.shape[a]
        anchor = facet_anchor_axis(sb_full, a)

        def owned_gids(r):
            return np.flatnonzero(((anchor - starts[r]) % n_ax) < counts[r])

        my_needed = self.sb.tb_needed_gids
        kept = self.sb.facet_gids
        kept_row = np.full(int(anchor.size), -1, dtype=np.int64)
        kept_row[kept] = np.arange(kept.size)
        slot_of = np.full(int(anchor.size), -1, dtype=np.int64)
        slot_of[my_needed] = np.arange(my_needed.size)

        nbrs = sorted({part.neighbor(s) for s in (0, 1)} - {None, part.rank})
        wires = []
        for nb in nbrs:
            nb_needed = band_needed_gids(
                sb_full, a, int(starts[nb]), int(counts[nb]), g)
            send = np.intersect1d(owned_gids(part.rank), nb_needed)
            recv = np.intersect1d(my_needed, owned_gids(nb))
            if kept_row[send].min(initial=0) < 0:
                raise AssertionError(
                    "taum wire: an owned facet is not kept locally")
            rk = kept_row[recv]
            kern_sel = np.flatnonzero(rk >= 0)
            ph_sel = np.flatnonzero(rk < 0)
            ph_slots = slot_of[recv[ph_sel]]
            if ph_slots.min(initial=0) < 0:
                raise AssertionError(
                    "taum wire: received phantom gid missing a tau_ext slot")
            wires.append((int(nb), {
                'send_rows': cp.asarray(kept_row[send]),
                'n_recv': int(recv.size),
                'kern_sel': cp.asarray(kern_sel),
                'kern_rows': cp.asarray(rk[kern_sel]),
                'ext_sel': cp.asarray(ph_sel),
                'ext_slots': cp.asarray(ph_slots),
            }))
        self.taum = wires or None

    def taum_bind(self, transport, rank: int, tag: int) -> None:
        self._tt = (transport, int(rank), int(tag))

    def taum_post(self) -> None:
        if not self.taum or self._tt is None:
            return
        tr, rank, tag = self._tt
        k = self.sb.kernel
        for nb, w in self.taum:
            tr.post(rank, nb, tag, cp.ascontiguousarray(
                k.tau_out[w['send_rows']]))
        tr.commit()
        # prepost the Irecv (after commit — the R3-1 ordering contract):
        # without it the peer's taum Isend stays unmatched until its NEXT
        # sync of this block, and any OTHER block's flush (Waitall is
        # transport-global) blocks on it first — measured deadlock, both
        # ranks in halo.flush at a child's _sync.
        if hasattr(tr, 'prepost'):
            for nb, w in self.taum:
                tr.prepost(nb, rank, tag, shape=(w['n_recv'],),
                           dtype=np.float64)

    def taum_complete(self) -> None:
        if not self.taum or self._tt is None:
            return
        tr, rank, tag = self._tt
        k = self.sb.kernel
        for nb, w in self.taum:
            arr = tr.collect(nb, rank, tag, shape=(w['n_recv'],),
                             dtype=np.float64)
            arr = cp.asarray(arr)
            if int(w['kern_rows'].size):
                k.tau_out[w['kern_rows']] = arr[w['kern_sel']]
            if int(w['ext_slots'].size):
                self.sb._tb_tau_ext[w['ext_slots']] = arr[w['ext_sel']]
        # NO flush here: taum messages exceed the eager threshold
        # (rendezvous), so every rank must reach its taum collect BEFORE
        # any rank blocks in a flush — the runner calls taum_complete
        # ahead of the halo's complete(), whose flush then covers these
        # sends too (deadlock observed with the reversed order).

    def macro_pre_pass(self) -> None:
        raise NotImplementedError("surfel slab has no ALM macro pre-pass")

    def last_force(self):
        """Owned-facet body force (SlabSurfelBoundary dedup; full sb on
        the single-rank path — there every facet is owned)."""
        return np.asarray(self.sb.last_force(), dtype=float)
