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
        if not getattr(lev, '_use_esoteric', False):
            raise RuntimeError(
                "surfel MPI runs on the eso residency bridge — launch "
                "with LBM_ESOTERIC=1 (patch_notes/surfel/63)")
        n_ax = part.global_shape[part.axis]
        self._full = part.own_count == n_ax          # NR=1 (ghost=0) path
        sb_full = lev.obstacle_bc
        if self._full:
            # single-rank: the replicated build's sim IS the validated
            # bridge — reuse it whole (bit-trivially the 1-GPU path)
            self.sim = lev
            self.sb = sb_full
            self.surfel_live = sb_full.d_live
            live_h = sb_full.live_h
        else:
            self.sb = build_slab_surfel(
                sb_full, part.axis, part.own_start, part.own_count,
                ghost=part.ghost)
            slab_shape = self.sb.shape
            idx = cp.asarray(
                np.arange(part.own_start - part.ghost,
                          part.own_start + part.own_count + part.ghost)
                % n_ax)
            take = [slice(None)] * 4
            take[1 + part.axis] = idx
            mem = cp.ascontiguousarray(lev.f[tuple(take)])

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
            self.surfel_live = self.sb.d_live
            live_h = self.sb.live_h
        # tau-band margin exchange wires (patch 64 stage ii): per unique
        # neighbor, the OWNED facets whose tau_out the neighbor's band
        # references — both sides derive the sets from the replicated
        # build, so they agree without communication.
        self.taum = None
        self._tt = None
        if not self._full and getattr(self.sb, 'tau_model_on', False):
            self._taum_wire(sb_full, part)

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
