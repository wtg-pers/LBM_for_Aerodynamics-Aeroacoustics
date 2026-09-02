"""Rank-local MLG coupling over decomposed esoteric levels (patch 17, M2 v1).

Wraps a GLOBAL GridCoupling (unchanged geometry + math primitives: fused
rescale kernel, cubic upsample, f_neq filter, macroscopic/f_eq) and performs
C2F / F2C on ONE rank's local slabs of the coarse and fine levels.

Correctness margins (why owned results are BIT-identical to 1-rank):
  * C2F: the local coarse read block extends 2 rows beyond the owned range
    along the split axis (ghost=2 fresh by schedule). The cubic upsample's
    one-sided stencils only occur within 1 coarse row of the BLOCK edge, so
    every fine node inside the rank's OWNED strip portion sees the same
    centered stencil values as the global computation. At true box ends the
    block edge coincides with the global edge -> the same one-sided stencil.
  * F2C: the strided fine read extends 1 coarse row beyond the owned write
    range, so the 7/19-point f_neq filter is centered (== global) on every
    written row; non-split axes cover the full box (identical edge handling).
  * Fine partitions are DERIVED from coarse cuts (fine = 2*(coarse-box_lo)),
    so strip/excised regions decompose without overlap or gaps.

The per-node math is delegated to the wrapped GridCoupling instance — no
duplicated physics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.kernels.esoteric_d3q27 import (
    esoteric_gather_std_region, esoteric_scatter_std_region)


def _ax_tuple(axis: int, sl: slice, other=slice(None)) -> Tuple[slice, ...]:
    out = [other] * 3
    out[axis] = sl
    return tuple(out)


def fine_range_from_coarse(part_c, box_lo: int, box_hi: int,
                           nf: int) -> Tuple[int, int]:
    """Owned fine range [start, count] derived from a coarse owned range.

    Coarse cut c (global) maps to fine index 2*(c - box_lo), clipped to the
    fine grid [0, nf). The LAST rank owning box rows absorbs the +1 end node
    of the co-located (2E+1) fine grid.
    """
    c0 = max(part_c.own_start, box_lo)
    c1 = min(part_c.own_start + part_c.own_count, box_hi + 1)
    if c1 <= c0:
        return 0, 0
    f0 = 2 * (c0 - box_lo)
    f1 = 2 * (c1 - box_lo)
    if c1 == box_hi + 1:          # end of box -> include the last (+1) node
        f1 = nf
    return f0, f1 - f0


class RankLocalCouplingV1:
    """C2F/F2C for one adjacent level pair on one rank's local slabs."""

    #: patch 76: coarse-slab live mask (3D, device) for the partial-body
    #: C2F dead fill; None = whole-body level (no-op, bit-preserving)
    _dead_fill_live = None
    #: patch 77: coarse-slab F2C wall-shell keep mask (flat, device);
    #: None = whole-body pair (no-op)
    _f2c_wall_keep = None

    def __init__(self, gc, part_c, part_f, xp, transport=None, rank=0,
                 n_ranks=1, tag_base=0) -> None:
        if part_c.axis != part_f.axis:
            raise ValueError("coarse/fine partitions must share the axis")
        self._gc = gc
        self._pc = part_c
        self._pf = part_f
        self._xp = xp
        self._a = part_c.axis
        r = gc._region
        box = r.fine_domain_coarse
        self._box_lo = (box.x_start, box.y_start, box.z_start)[self._a]
        self._box_hi = (box.x_end, box.y_end, box.z_end)[self._a]
        self._nf_ax = r.fine_shape[self._a]
        # Phase B1 (mpi_axis/04): with a transport, the partitions may be
        # INDEPENDENT per level — reads are assembled from the local wrap
        # window plus row strips exchanged with the owning ranks. Without
        # one (legacy shared-cut path), the derived-alignment contract is
        # enforced exactly as before and every read stays rank-local.
        self._tr = transport
        self._rank = int(rank)
        self._nr = int(n_ranks)
        self._tag_c = int(tag_base)          # c2f/fprev coarse-row strips
        self._tag_f = int(tag_base) + 1      # f2c fine-row strips
        if transport is None:
            # consistency: fine partition must be the coarse-derived one
            f0, fc = fine_range_from_coarse(
                part_c, self._box_lo, self._box_hi, self._nf_ax)
            if (f0, fc) != (part_f.own_start, part_f.own_count):
                raise ValueError(
                    f"fine partition {part_f.own_start, part_f.own_count}"
                    f" != coarse-derived {(f0, fc)}")
        else:
            cc = np.asarray(part_c._counts, dtype=np.int64)
            cf = np.asarray(part_f._counts, dtype=np.int64)
            self._starts_c = np.concatenate([[0], np.cumsum(cc)])[:-1]
            self._counts_c = cc
            self._starts_f = np.concatenate([[0], np.cumsum(cf)])[:-1]
            self._counts_f = cf
            ex = gc.excised_spatial_slices
            self._ex_lo = int(ex[self._a].start)
            self._ex_hi = int(ex[self._a].stop)
            self._span_z = (
                self._a == 2
                and getattr(gc, '_flush_faces', {}).get('z_min', False)
                and gc._flush_faces.get('z_max', False))

    # ── Phase B1 row tables (replicated — both sides compute the same
    #    sets from the shared counts, so strips match with no handshake) ─
    def _need_c(self, r) -> Tuple[int, int]:
        """Coarse rows [lo, hi) rank r's OWNED-fine strips read for C2F /
        f_prev: parents of its fine range +-2 (the M2 stencil margin),
        clipped to the box. On derived partitions this equals the legacy
        coarse_block_range() row for row."""
        fs = int(self._starts_f[r])
        fc = int(self._counts_f[r])
        if fc == 0:
            return (self._box_lo, self._box_lo)
        lo = max(self._box_lo, self._box_lo + fs // 2 - 2)
        hi = min(self._box_hi + 1,
                 self._box_lo + (fs + fc + 1) // 2 + 2)
        return (lo, hi)

    def _need_f_rows(self, r):
        """(fine row list, pad_lo, pad_hi, w_lo, w_hi) rank r reads for
        F2C: 2x strided rows for its coarse write range +-1, plus the
        span-z wrap margin rows (patch 64 sec. 10 conditions mirrored)."""
        w_lo = max(self._ex_lo, int(self._starts_c[r]))
        w_hi = min(self._ex_hi,
                   int(self._starts_c[r]) + int(self._counts_c[r]))
        if w_hi <= w_lo:
            return np.empty(0, np.int64), 0, 0, w_lo, w_hi
        rr_lo = max(w_lo - 1, self._box_lo)
        rr_hi = min(w_hi + 1, self._box_hi + 1)
        g0 = 2 * (rr_lo - self._box_lo)
        g1 = 2 * (rr_hi - 1 - self._box_lo) + 1
        rows = np.arange(g0, g1, 2, dtype=np.int64)
        pad_lo = pad_hi = 0
        if self._span_z:
            n_c = self._box_hi + 1 - self._box_lo
            pre = post = None
            if w_lo - 1 < self._box_lo:
                pre = 2 * ((w_lo - 1 - self._box_lo) % n_c)
                pad_lo = 1
            if w_hi + 1 > self._box_hi + 1:
                post = 2 * ((w_hi - self._box_lo) % n_c)
                pad_hi = 1
            parts = ([np.asarray([pre], np.int64)] if pre is not None
                     else []) + [rows] + \
                    ([np.asarray([post], np.int64)] if post is not None
                     else [])
            rows = np.concatenate(parts)
        return rows, pad_lo, pad_hi, w_lo, w_hi

    @staticmethod
    def _win_pos(part, rows):
        """Local window position of each global row; -1 when absent."""
        N = int(part.global_shape[part.axis])
        W = int(part.own_count) + 2 * int(part.ghost)
        pos = (np.asarray(rows, np.int64)
               - (int(part.own_start) - int(part.ghost))) % N
        return np.where(pos < W, pos, np.int64(-1))

    @staticmethod
    def _peer_pos(starts, counts, ghost, N, r, rows):
        """_win_pos for PEER rank r, from the replicated tables."""
        W = int(counts[r]) + 2 * ghost
        pos = (np.asarray(rows, np.int64)
               - (int(starts[r]) - ghost)) % N
        return np.where(pos < W, pos, np.int64(-1))

    def _gather_rows(self, mem, t, rows_pos, sub, wall):
        """Gather local window rows (positions rows_pos, ascending order
        of the caller's row list) as std blocks — one region gather per
        maximal constant-stride run, concatenated along the axis."""
        xp, a = self._xp, self._a
        out = []
        i, n = 0, len(rows_pos)
        while i < n:
            j = i
            step = 1
            if j + 1 < n:
                step = int(rows_pos[j + 1] - rows_pos[j])
                if step <= 0:
                    step = 1
            while (j + 1 < n
                   and rows_pos[j + 1] - rows_pos[j] == step and step > 0):
                j += 1
            sl = list(sub)
            sl[a] = slice(int(rows_pos[i]), int(rows_pos[j]) + 1,
                          step if step > 1 else 1)
            out.append(esoteric_gather_std_region(xp, mem, t, tuple(sl),
                                                  **wall))
            i = j + 1
        if not out:
            return None
        return out[0] if len(out) == 1 else xp.concatenate(out, axis=1 + a)

    def _exchange(self, mem, t, wall, part, starts, counts, need_rows_of,
                  my_rows, my_pos, sub, ext_shape, tag):
        """One symmetric strip-exchange round. Returns {index_in_my_rows:
        received block} per peer. Sends are flushed before return (the
        persistent (dst, tag) send buffer is reused next round)."""
        xp, a, tr = self._xp, self._a, self._tr
        N = int(part.global_shape[part.axis])
        g = int(part.ghost)
        # sends: peer's needed rows outside ITS window, owned by me
        posted = False
        for p in range(self._nr):
            if p == self._rank:
                continue
            prow = need_rows_of(p)
            if prow.size == 0:
                continue
            ppos = self._peer_pos(starts, counts, g, N, p, prow)
            m0 = int(starts[self._rank])
            m1 = m0 + int(counts[self._rank])
            srow = prow[(ppos < 0) & (prow >= m0) & (prow < m1)]
            if srow.size == 0:
                continue
            spos = srow - (int(part.own_start) - g)   # owned rows: no wrap
            buf = self._gather_rows(mem, t, spos, sub, wall)
            tr.post(self._rank, p, tag, buf)
            posted = True
        tr.commit()
        if not posted and hasattr(xp, 'cuda'):
            # prepost/Recv hazard guard (halo R3-1): commit()'s stream sync
            # is skipped when nothing was staged, but the persistent recv
            # buffer may still be read by the previous round's assembly.
            xp.cuda.get_current_stream().synchronize()
        # receives: my needed rows outside MY window, grouped by owner
        recv = {}
        if my_rows.size:
            owner = np.searchsorted(np.concatenate(
                [starts, [N]]), my_rows, side='right') - 1
            for p in range(self._nr):
                if p == self._rank:
                    continue
                sel = np.flatnonzero((my_pos < 0) & (owner == p))
                if sel.size == 0:
                    continue
                shape = list(ext_shape)
                shape[1 + a] = int(sel.size)
                arr = tr.collect(p, self._rank, tag, shape=tuple(shape),
                                 dtype=mem.dtype)
                recv[int(sel[0])] = (sel, arr)
        tr.flush()
        return recv

    def _sub_ext_shape(self, mem, sub, n_pop):
        ext = [n_pop]
        for d in range(3):
            ext.append(len(range(*sub[d].indices(mem.shape[1 + d])))
                       if d != self._a else 0)
        return ext

    def fetch_coarse_block(self, mem_c, t_c, wall_c):
        """Assembled std block over the coarse rows THIS rank's C2F/f_prev
        needs (legacy: the rank-local block, bit-identical gather)."""
        wc = wall_c or {}
        if self._tr is None:
            reg = self.coarse_block_region_local()
            if reg is None:
                return None
            return esoteric_gather_std_region(self._xp, mem_c, t_c, reg,
                                              **wc)
        xp, a = self._xp, self._a
        lo, hi = self._need_c(self._rank)
        rows = np.arange(lo, hi, dtype=np.int64)
        pos = self._win_pos(self._pc, rows)
        sub = list(self._gc.coarse_sub_spatial_slices)
        ext = self._sub_ext_shape(mem_c, sub, int(mem_c.shape[0]))
        recv = self._exchange(
            mem_c, t_c, wc, self._pc, self._starts_c, self._counts_c,
            lambda p: np.arange(*self._need_c(p), dtype=np.int64),
            rows, pos, sub, ext, self._tag_c)
        return self._assemble(mem_c, t_c, wc, rows, pos, sub, recv)

    def fetch_fine_rows(self, mem_f, t_f, wall_f):
        """(assembled strided fine block, pad_lo, pad_hi) for THIS rank's
        F2C — engaged path only. Runs the exchange even when this rank
        writes nothing (peers may need its rows)."""
        wf = wall_f or {}
        xp, a = self._xp, self._a
        rows, pad_lo, pad_hi, _w0, _w1 = self._need_f_rows(self._rank)
        pos = self._win_pos(self._pf, rows)
        sub = list(self._gc.fine_at_coarse_spatial_slices)
        ext = self._sub_ext_shape(mem_f, sub, int(mem_f.shape[0]))
        recv = self._exchange(
            mem_f, t_f, wf, self._pf, self._starts_f, self._counts_f,
            lambda p: self._need_f_rows(p)[0],
            rows, pos, sub, ext, self._tag_f)
        if rows.size == 0:
            return None, pad_lo, pad_hi
        return (self._assemble(mem_f, t_f, wf, rows, pos, sub, recv),
                pad_lo, pad_hi)

    def _assemble(self, mem, t, wall, rows, pos, sub, recv):
        """Stitch local runs + received strips into row order."""
        xp, a = self._xp, self._a
        pieces = {}
        i, n = 0, rows.size
        while i < n:
            if pos[i] < 0:
                i += 1
                continue
            j = i
            step = 1
            if j + 1 < n and pos[j + 1] >= 0:
                step = int(pos[j + 1] - pos[j])
            while (j + 1 < n and pos[j + 1] >= 0 and step > 0
                   and pos[j + 1] - pos[j] == step
                   and rows[j + 1] - rows[j] == step):
                j += 1
            blk = self._gather_rows(mem, t, pos[i:j + 1], sub, wall)
            pieces[i] = blk
            i = j + 1
        for k, (sel, arr) in recv.items():
            # received strips arrive in ascending row order; split into
            # maximal contiguous index runs of my row list
            runs = np.split(np.arange(sel.size),
                            np.flatnonzero(np.diff(sel) > 1) + 1)
            off = 0
            for run in runs:
                sl = [slice(None)] * arr.ndim
                sl[1 + a] = slice(off, off + len(run))
                pieces[int(sel[run[0]])] = arr[tuple(sl)]
                off += len(run)
        parts = [pieces[k] for k in sorted(pieces)]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else xp.concatenate(
            parts, axis=1 + self._a)

    # ── coordinate translation (global -> local, along split axis) ──
    def _l_c(self, g: int) -> int:
        return g - (self._pc.own_start - self._pc.ghost)

    def _l_f(self, g: int) -> int:
        return g - (self._pf.own_start - self._pf.ghost)

    # ── coarse read block along the axis: owned∩box extended by 2 ──
    def coarse_block_range(self) -> Optional[Tuple[int, int]]:
        lo = max(self._box_lo, self._pc.own_start - 2)
        hi = min(self._box_hi + 1, self._pc.own_start + self._pc.own_count + 2)
        return (lo, hi) if hi > lo else None

    def coarse_block_region_local(self) -> Optional[Tuple[slice, ...]]:
        """Spatial region (LOCAL coarse coords) to gather for C2F/f_prev."""
        rng = self.coarse_block_range()
        if rng is None:
            return None
        sub = list(self._gc.coarse_sub_spatial_slices)
        sub[self._a] = slice(self._l_c(rng[0]), self._l_c(rng[1]))
        return tuple(sub)

    # =================================================================
    def c2f(self, mem_c, mem_f, t_c: int, t_f: int, *, is_half_step: bool,
            f_prev_sub_loc=None, nt_f=None,
            wall_c=None, wall_f=None) -> None:
        """Rank-local coarse->fine: strips ∩ owned-fine written in place.

        nt_f: fine slab node-type (raveled int8) — coupling writes carry
        FOREIGN values, which at SOLID cells land on live bounce-deposit
        slots (patch 12); pass it so those entries are skipped.
        wall_c/wall_f: slab-local implicit-wall view args of the coarse/
        fine LocalLevel (eso_wall §4-5b) — gathers/scatters at wall rows
        must go through the swap-slot/mailbox LOAD view, exactly as the
        single-GPU MLG coupling does since patch 04."""
        if self._pf.own_count == 0 and self._tr is None:
            return
        gc, xp, a = self._gc, self._xp, self._a
        wc, wf = (wall_c or {}), (wall_f or {})
        # B1 engaged: the block covers the FINE-derived needed rows and is
        # assembled local+exchanged (fetch runs even with own_count 0 —
        # peers may need this rank's coarse rows). Legacy: bit-identical
        # single local gather over the derived block.
        f_sub = self.fetch_coarse_block(mem_c, t_c, wc)
        if self._pf.own_count == 0:
            return
        rng = (self.coarse_block_range() if self._tr is None
               else self._need_c(self._rank))
        if self._dead_fill_live is not None:
            # partial-body fine level (patch 76 = the runner port of the
            # patch-74 C2F dead fill): the band crosses the body, so the
            # interpolation stencil reaches coarse DEAD cells (surfel
            # f = 0 -> rho collapse -> NaN on the fine level). Fill them
            # with the rest-state equilibrium — same constant as the
            # single-GPU MultiLevelGrid fill, so own strips stay bit-
            # comparable.
            if self._tr is not None:
                raise NotImplementedError(
                    "partial-body C2F dead fill is not wired for the "
                    "per-level exchange path (mpi_axis/04 sec. 3)")
            from src.grid.multi_level_grid import _rest_feq_27
            region_loc = self.coarse_block_region_local()
            dead = ~self._dead_fill_live[region_loc].astype(bool)
            w27 = _rest_feq_27(xp, f_sub.dtype)
            f_sub = xp.where(dead[None], w27[:, None, None, None], f_sub)

        # temporal interp + f_eq/f_neq rescale (same primitives as coupling.py)
        if gc._fused_rescale is not None:
            coarse_nodes = gc._fused_rescale.c2f(
                f_sub, f_prev_sub_loc, is_half_step, gc._factor_c2f)
        else:
            if is_half_step:
                f_sub = 0.5 * (f_prev_sub_loc + f_sub)
            rho, u = gc._compute_macroscopic(f_sub)
            f_eq = gc._compute_f_eq(rho, u)
            coarse_nodes = f_eq + gc._factor_c2f * (f_sub - f_eq)

        # Boundary-only upsample (production strips_out parity, patch 17
        # M5 pass 3): the v1 full-volume block upsample wrote ~30x more
        # fine volume than the 6 written strips need (D40 profile: coupling
        # ~1.0 s/step of 3.0). Per face, upsample only the thin coarse slab
        # from the LOCAL block; the M2 margin argument carries over
        # unchanged (the block extends 2 coarse rows past owned, so every
        # owned strip node sees the same centered/one-sided stencils as the
        # full-volume result -> bit).
        blk_f0 = 2 * (rng[0] - self._box_lo)      # block origin, global fine
        B = coarse_nodes.shape[1 + a]
        own0 = self._pf.own_start
        own1 = own0 + self._pf.own_count
        for c_sl, w_sl, r_sl in gc._bnd_face_specs:
            strip = list(w_sl[1:])                 # global fine, spatial
            fa = next(i for i, sl_ in enumerate(strip)
                      if not (sl_.start is None and sl_.stop is None))
            s_ax = strip[a]
            lo = 0 if s_ax.start is None else s_ax.start
            hi = self._nf_ax if s_ax.stop is None else s_ax.stop
            lo, hi = max(lo, own0), min(hi, own1)  # ∩ owned fine rows
            if hi <= lo:
                continue
            c_spatial = list(c_sl[1:])             # box coords, thin on fa
            r_spatial = [slice(None)] * 3
            if fa == a:
                # face normal to the split axis: translate the slab rows to
                # local block indices (containment guaranteed: owning strip
                # rows implies own>=ghost coarse rows reach past the slab)
                cb = c_spatial[a]
                lo_c = cb.start + self._box_lo - rng[0]
                hi_c = cb.stop + self._box_lo - rng[0]
                if lo_c < 0 or hi_c > B:
                    raise AssertionError(
                        f"c2f face slab [{lo_c},{hi_c}) outside local block "
                        f"B={B} (own_count too small?)")
                c_spatial[a] = slice(lo_c, hi_c)
                off = lo - (0 if s_ax.start is None else s_ax.start)
                r0 = r_sl[1 + fa].start or 0
                r_spatial[fa] = slice(r0 + off, r0 + off + (hi - lo))
            else:
                c_spatial[a] = slice(None)         # full local block rows
                r_spatial[fa] = r_sl[1 + fa]
                r_spatial[a] = slice(lo - blk_f0, hi - blk_f0)
            slab = gc._upsample_block(
                coarse_nodes[(slice(None),) + tuple(c_spatial)])
            vals = slab[(slice(None),) + tuple(r_spatial)]
            strip[a] = slice(lo, hi)
            dst = list(strip)
            dst[a] = slice(self._l_f(lo), self._l_f(hi))
            esoteric_scatter_std_region(xp, mem_f, vals, t_f, tuple(dst),
                                        skip_solid_nt=nt_f, **wf)

    # =================================================================
    def f2c(self, mem_f, mem_c, t_f: int, t_c: int, nt_c=None,
            wall_c=None, wall_f=None, surfel_live_c=None) -> None:
        """Rank-local fine->coarse: excised ∩ owned-coarse written in place.

        nt_c: coarse slab node-type — the excised region contains the body
        on coarse levels; without the skip the restriction's solid-cell
        values overwrite live bounce deposits (patch 12 root cause).
        wall_c/wall_f: see c2f (eso_wall §4-5b).
        surfel_live_c: coarse slab live mask (flat) — SURFEL coarse level
        (patch 64): replaces the hwbb skip with the patch-50 semantics
        (finite-only fine feedback + dead re-zero), mirroring the single-
        GPU bridge branch in MultiLevelGrid._coupling_f2c bit-for-bit
        (same pointwise ops on the same windows)."""
        gc, xp, a = self._gc, self._xp, self._a
        wc, wf = (wall_c or {}), (wall_f or {})
        r = gc._region
        R = r.REFINE_RATIO
        ex = gc.excised_spatial_slices                    # global coarse
        ex_lo = ex[a].start
        ex_hi = ex[a].stop                                # exclusive
        w_lo = max(ex_lo, self._pc.own_start)
        w_hi = min(ex_hi, self._pc.own_start + self._pc.own_count)
        if self._tr is not None:
            # B1 engaged: assembled strided block (local + exchanged) with
            # the span-z wrap margins folded into the row set. Runs even
            # when this rank writes nothing — peers may need its rows.
            f_fine_at, pad_lo, pad_hi = self.fetch_fine_rows(mem_f, t_f, wf)
            if w_hi <= w_lo:
                return
            rr_lo = max(w_lo - 1, self._box_lo)
        else:
            if w_hi <= w_lo:
                return
            # strided fine read rows: coarse [w_lo-1, w_hi+1) ∩ box
            # (filter halo 1)
            rr_lo = max(w_lo - 1, self._box_lo)
            rr_hi = min(w_hi + 1, self._box_hi + 1)
            # global strided region over the FULL box on non-split axes
            f_at = list(gc.fine_at_coarse_spatial_slices)  # (0::R,)*3 global
            g0 = 2 * (rr_lo - self._box_lo)
            g1 = 2 * (rr_hi - 1 - self._box_lo) + 1
            f_at[a] = slice(self._l_f(g0), self._l_f(g0) + (g1 - g0), R)
            f_fine_at = esoteric_gather_std_region(xp, mem_f, t_f,
                                                   tuple(f_at), **wf)

        # ── span-through z: WRAP filter margins (patch 64 sec. 10) ──
        # The single-GPU f_neq filter is z-PERIODIC on z-flush regions
        # (coupling._filter_f_neq, xp.roll — the 0812 seam-periodicity
        # patch). The box clamp above turns that roll into a FALSE wrap
        # between the local block's own edge rows: the rows written next
        # to the GLOBAL z-ends read the cut-side margin instead of the
        # true periodic neighbor (measured: G2 step-1 divergence at each
        # rank's wrap-side own rows, cut side clean). Fix: gather the
        # true wrap margin row(s) — present in this rank's fine slab
        # GHOSTS — and concatenate, so the roll's block-edge wrap only
        # ever touches unwritten margin rows. Wrap arithmetic is over
        # the STRIDED coarse-co-located rows (period n_c), NOT linear
        # fine index — with the odd node-based nf (2n-1) a linear 2*c
        # extension would alias fine row 1 for the wrap of row 0.
        # z-literal is CORRECT, not a z-cut assumption (mpi_axis/01 #3):
        # the single-GPU _filter_f_neq periodic variant is itself z-flush
        # only (grid/coupling.py "Flush-z span-through prism"), so the
        # wrap margin is needed exactly when cut==z==both-flush. If an
        # x/y-flush filter variant is ever added there, revise this gate
        # in the same patch. Engaged (B1) runs fold these margin rows into
        # the fetch_fine_rows row set instead (_need_f_rows).
        if self._tr is not None:
            span_z = False
        else:
            pad_lo = pad_hi = 0
            span_z = (a == 2
                      and getattr(gc, '_flush_faces', {}).get('z_min', False)
                      and gc._flush_faces.get('z_max', False))
        if span_z:
            n_c = self._box_hi + 1 - self._box_lo

            def _fine_margin_row(m_c):
                g = 2 * ((m_c - self._box_lo) % n_c)
                loc = (g - (self._pf.own_start - self._pf.ghost)) \
                    % self._nf_ax
                if not (0 <= loc < self._pf.local_shape[a]):
                    raise ValueError(
                        "span-z F2C wrap margin row outside the fine "
                        "slab window — ghost too small for the filter")
                r_at = list(gc.fine_at_coarse_spatial_slices)
                r_at[a] = slice(loc, loc + 1)
                return esoteric_gather_std_region(
                    xp, mem_f, t_f, tuple(r_at), **wf)

            parts = [f_fine_at]
            if w_lo - 1 < self._box_lo:
                parts.insert(0, _fine_margin_row(w_lo - 1))
                pad_lo = 1
            if w_hi + 1 > self._box_hi + 1:
                parts.append(_fine_margin_row(w_hi))
                pad_hi = 1
            if pad_lo or pad_hi:
                f_fine_at = xp.concatenate(parts, axis=1 + a)

        # same fused feq/fneq primitive as GridCoupling.fine_to_coarse —
        # the decomposition MUST share the production op or the two sides
        # split by kernel-vs-elementwise rounding (few f32 ulps, G-M2b)
        f_eq, f_neq_raw = gc._feq_fneq(f_fine_at)
        f_neq = gc._filter_f_neq(f_neq_raw)
        recon = f_eq + gc._factor_f2c * f_neq

        # excised extraction: global excised offsets within the box on the
        # non-split axes (gc._excised_local_slices), write-rows on the axis.
        exl = list(gc._excised_local_slices[1:])          # spatial, box-local
        exl[a] = slice(w_lo - rr_lo + pad_lo, w_hi - rr_lo + pad_lo)
        block = recon[(slice(None),) + tuple(exl)]
        dst = list(ex)
        dst[a] = slice(self._l_c(w_lo), self._l_c(w_hi))
        if surfel_live_c is not None:
            cur = esoteric_gather_std_region(xp, mem_c, t_c, tuple(dst),
                                             **wc)
            block = xp.where(xp.isnan(block),
                             cur.astype(block.dtype, copy=False), block)
            if self._f2c_wall_keep is not None:
                k3 = self._f2c_wall_keep.reshape(
                    mem_c.shape[1:])[tuple(dst)]
                block = xp.where((k3 > 0)[None],
                                 cur.astype(block.dtype, copy=False),
                                 block)
            live = surfel_live_c.reshape(mem_c.shape[1:])[tuple(dst)]
            block = xp.where((live > 0)[None], block,
                             block.dtype.type(0.0))
            esoteric_scatter_std_region(
                xp, mem_c, block.astype(cur.dtype, copy=False), t_c,
                tuple(dst), **wc)
        else:
            esoteric_scatter_std_region(xp, mem_c, block, t_c, tuple(dst),
                                        skip_solid_nt=nt_c, **wc)
