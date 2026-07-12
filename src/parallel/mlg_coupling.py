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

    def __init__(self, gc, part_c, part_f, xp) -> None:
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
        # consistency: fine partition must be the coarse-derived one
        f0, fc = fine_range_from_coarse(part_c, self._box_lo, self._box_hi,
                                        self._nf_ax)
        if (f0, fc) != (part_f.own_start, part_f.own_count):
            raise ValueError(
                f"fine partition {part_f.own_start, part_f.own_count} != "
                f"coarse-derived {(f0, fc)}")

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
            f_prev_sub_loc=None) -> None:
        """Rank-local coarse->fine: strips ∩ owned-fine written in place."""
        if self._pf.own_count == 0:
            return
        gc, xp, a = self._gc, self._xp, self._a
        rng = self.coarse_block_range()
        region_loc = self.coarse_block_region_local()
        f_sub = esoteric_gather_std_region(xp, mem_c, t_c, region_loc)

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

        # local full-volume upsample of the block (v1; slab-scoped is M4)
        fine_block = gc._upsample_block(coarse_nodes)
        blk_f0 = 2 * (rng[0] - self._box_lo)      # block origin, global fine

        own0 = self._pf.own_start
        own1 = own0 + self._pf.own_count
        for (sx, sy, sz) in gc._fine_bnd_slices.values():
            strip = [sx, sy, sz]                   # global fine, spatial
            s_ax = strip[a]
            lo = 0 if s_ax.start is None else s_ax.start
            hi = self._nf_ax if s_ax.stop is None else s_ax.stop
            lo, hi = max(lo, own0), min(hi, own1)  # ∩ owned fine rows
            if hi <= lo:
                continue
            strip[a] = slice(lo, hi)
            # values from the local upsampled block
            blk = [slice(s.start, s.stop) if s.start is not None or
                   s.stop is not None else slice(None) for s in strip]
            blk[a] = slice(lo - blk_f0, hi - blk_f0)
            vals = fine_block[(slice(None),) + tuple(blk)]
            # destination in LOCAL fine coords
            dst = list(strip)
            dst[a] = slice(self._l_f(lo), self._l_f(hi))
            esoteric_scatter_std_region(xp, mem_f, vals, t_f, tuple(dst))

    # =================================================================
    def f2c(self, mem_f, mem_c, t_f: int, t_c: int) -> None:
        """Rank-local fine->coarse: excised ∩ owned-coarse written in place."""
        gc, xp, a = self._gc, self._xp, self._a
        r = gc._region
        R = r.REFINE_RATIO
        ex = gc.excised_spatial_slices                    # global coarse
        ex_lo = ex[a].start
        ex_hi = ex[a].stop                                # exclusive
        w_lo = max(ex_lo, self._pc.own_start)
        w_hi = min(ex_hi, self._pc.own_start + self._pc.own_count)
        if w_hi <= w_lo:
            return
        # strided fine read rows: coarse [w_lo-1, w_hi+1) ∩ box (filter halo 1)
        rr_lo = max(w_lo - 1, self._box_lo)
        rr_hi = min(w_hi + 1, self._box_hi + 1)
        # global strided region over the FULL box on non-split axes
        f_at = list(gc.fine_at_coarse_spatial_slices)     # (0::R,)*3 global
        g0 = 2 * (rr_lo - self._box_lo)
        g1 = 2 * (rr_hi - 1 - self._box_lo) + 1
        f_at[a] = slice(self._l_f(g0), self._l_f(g0) + (g1 - g0), R)
        f_fine_at = esoteric_gather_std_region(xp, mem_f, t_f, tuple(f_at))

        # same fused feq/fneq primitive as GridCoupling.fine_to_coarse —
        # the decomposition MUST share the production op or the two sides
        # split by kernel-vs-elementwise rounding (few f32 ulps, G-M2b)
        f_eq, f_neq_raw = gc._feq_fneq(f_fine_at)
        f_neq = gc._filter_f_neq(f_neq_raw)
        recon = f_eq + gc._factor_f2c * f_neq

        # excised extraction: global excised offsets within the box on the
        # non-split axes (gc._excised_local_slices), write-rows on the axis.
        exl = list(gc._excised_local_slices[1:])          # spatial, box-local
        exl[a] = slice(w_lo - rr_lo, w_hi - rr_lo)
        block = recon[(slice(None),) + tuple(exl)]
        dst = list(ex)
        dst[a] = slice(self._l_c(w_lo), self._l_c(w_hi))
        esoteric_scatter_std_region(xp, mem_c, block, t_c, tuple(dst))
