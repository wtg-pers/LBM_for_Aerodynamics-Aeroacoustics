"""Esoteric+surfel bridge — static rewrite sets and staging boxes.

patch_notes/surfel/63 (V1, phase-exact sandwich). This module holds the
BUILD-TIME pieces: the deviation support M, the rewrite mask R, and the
axis-aligned stage/deposit boxes the per-substep bridge operates on.

Key fact (63 sec. 0): surfel_advect is identical to plain pull streaming
wherever dV=1, g_field=0, Q=0 and live=1, and the support of every
deviation is static (supp Q is inside the facet CSR cell set). So the
set of cells whose streamed value the bridge must rewrite,

    R = M ∪ { y : ∃i, y - c_i ∈ M },

is computable once at build.

V1 stages/deposits on axis-aligned BOXES (the existing region-scoped
esoteric gather/scatter primitives take slice regions): depositing the
whole box interior is valid because inside the box the staged chain IS
the std chain — box cells outside R simply get values that match the
std path even more closely. R is used to VERIFY containment, not to
mask writes.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from src.boundary.surfel_transport import C27

Region = Tuple[slice, slice, slice]


def _as_host_bool(a) -> np.ndarray:
    a = a.get() if hasattr(a, 'get') else np.asarray(a)
    return np.asarray(a).astype(bool)


def deviation_support(dV, g_field, live, shape,
                      extra_cells: Sequence = ()) -> np.ndarray:
    """M = {dV != 1} | {any_i g != 0} | {~live} | extra cells (flat idx).

    dV: (N,) cut-cell volumes; g_field: (27, N) facet interception
    fractions; live: (N,) uint8/bool; extra_cells: flat indices whose
    populations the band machinery touches (facet CSR cells — supp(Q)
    and the gather set — plus the tau-model injection band).
    """
    n = int(np.prod(shape))

    def _host(a, dtype):
        return np.asarray(a.get() if hasattr(a, 'get') else a, dtype=dtype)

    dV_h = _host(dV, np.float64).reshape(n)
    g_h = _host(g_field, np.float64).reshape(27, n)
    live_h = _as_host_bool(live).reshape(n)
    M = (dV_h != 1.0) | (g_h != 0.0).any(axis=0) | (~live_h)
    for cells in extra_cells:
        M[_host(cells, np.int64).reshape(-1)] = True
    return M.reshape(shape)


def rewrite_mask(M: np.ndarray) -> np.ndarray:
    """R = M | (cells that PULL from M): roll(M, +c_i) covers y with
    M[y - c_i] (advect kernel reads s = y - c_i, %N wrap on every axis
    — the roll wrap is kernel-faithful)."""
    R = M.copy()
    for c in C27:
        if not c.any():
            continue
        R |= np.roll(M, shift=tuple(int(v) for v in c), axis=(0, 1, 2))
    return R


def source_mask(R: np.ndarray) -> np.ndarray:
    """Cells some R-cell pulls from: roll(R, -c_i)."""
    S = R.copy()
    for c in C27:
        if not c.any():
            continue
        S |= np.roll(R, shift=tuple(-int(v) for v in c), axis=(0, 1, 2))
    return S


def _axis_box(mask_1d: np.ndarray, margin: int) -> slice:
    """Tight index range covering the mask on one axis, +margin.

    Conservative: a mask touching BOTH ends of the axis (wrap-spanning
    body, or margin crossing the boundary) collapses to the full axis —
    the periodic wrap of the advect stencil makes a split box unsound.
    """
    n = mask_1d.size
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return slice(0, 0)
    lo, hi = int(idx[0]) - margin, int(idx[-1]) + 1 + margin
    if lo < 0 or hi > n:
        return slice(0, n)
    return slice(lo, hi)


def stage_and_deposit_boxes(M: np.ndarray, stage_margin: int = 2,
                            deposit_margin: int = 1
                            ) -> Tuple[Region, Region]:
    """Axis-aligned (stage, deposit) boxes for the V1 bridge.

    deposit ⊇ R (verified by the caller via rewrite_mask) and every
    advect source of a deposit cell lies inside stage — guaranteed by
    stage_margin >= deposit_margin + 1 (stencil reach 1).
    """
    if stage_margin < deposit_margin + 1:
        raise ValueError("stage box must exceed deposit box by the "
                         "advect stencil reach (1 cell)")
    ax = [M.any(axis=tuple(a for a in range(3) if a != k))
          for k in range(3)]
    stage = tuple(_axis_box(ax[k], stage_margin) for k in range(3))
    dep = tuple(_axis_box(ax[k], deposit_margin) for k in range(3))
    return stage, dep


def verify_containment(M: np.ndarray, stage: Region, dep: Region) -> None:
    """Raise unless R ⊆ deposit box and sources(deposit) ⊆ stage box.

    Bridge soundness proof obligation (63 sec. 1) — run once at build.
    """
    shape = M.shape
    R = rewrite_mask(M)
    inside_dep = np.zeros(shape, dtype=bool)
    inside_dep[dep] = True
    if (R & ~inside_dep).any():
        raise ValueError("rewrite set R escapes the deposit box — "
                         "wrap-spanning body? (surfel_eso._axis_box)")
    dep_mask = inside_dep
    need = source_mask(dep_mask)
    inside_stage = np.zeros(shape, dtype=bool)
    inside_stage[stage] = True
    if (need & ~inside_stage).any():
        raise ValueError("advect sources of the deposit box escape the "
                         "stage box — margins inconsistent")
