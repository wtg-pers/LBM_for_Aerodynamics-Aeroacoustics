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


def _facet_segments(k):
    """(seg, fac_lo) of the kernel CSR (facet-major, asserted)."""
    from src.boundary.surfel_transport import N_PAIR
    seg = np.asarray(k.indptr.get() if hasattr(k.indptr, 'get')
                     else k.indptr).reshape(-1)
    fac_lo = seg[:-1].reshape(k.n_f, N_PAIR)[:, 0]
    if fac_lo[0] != 0 or not (np.diff(fac_lo) >= 0).all():
        raise AssertionError("surfel CSR is not facet-major contiguous")
    return seg, fac_lo


def _cells_axis(k, full_shape, axis) -> np.ndarray:
    """Per-CSR-cell global axis coordinate (host)."""
    cell_h = np.asarray(k.cell.get() if hasattr(k.cell, 'get') else k.cell,
                        dtype=np.int64)
    Nz = full_shape[2]
    if axis == 2:
        return cell_h % Nz
    if axis == 1:
        return (cell_h // Nz) % full_shape[1]
    return cell_h // (full_shape[1] * Nz)


def _window_inv(n_ax: int, own_start: int, own_count: int,
                ghost: int) -> np.ndarray:
    idx = np.arange(own_start - ghost, own_start + own_count + ghost) % n_ax
    inv = np.full(n_ax, -1, dtype=np.int64)
    inv[idx] = np.arange(idx.size)
    return inv


def facet_keep_mask(sb, axis: int, own_start: int, own_count: int,
                    ghost: int) -> np.ndarray:
    """Bool per FULL-build facet: all CSR cells inside the wrap window.

    The exchange wiring (patch 64 stage ii) evaluates this for NEIGHBOR
    windows too — both ranks derive each other's sets from the same
    replicated build, so the sets agree without communication.
    """
    k = sb.kernel
    _seg, fac_lo = _facet_segments(k)
    ax = _cells_axis(k, sb.shape, axis)
    inv = _window_inv(sb.shape[axis], own_start, own_count, ghost)
    return np.minimum.reduceat(inv[ax], fac_lo) >= 0


def facet_anchor_axis(sb, axis: int) -> np.ndarray:
    """Global axis coord of each facet's FIRST CSR cell (ownership anchor)."""
    k = sb.kernel
    _seg, fac_lo = _facet_segments(k)
    return _cells_axis(k, sb.shape, axis)[fac_lo]


def band_needed_gids(sb, axis: int, own_start: int, own_count: int,
                     ghost: int) -> np.ndarray:
    """Facet gids the tau-band W rows INSIDE the window reference.

    == the tau_ext column set of the slab built on that window (the slab
    build uses the same row filter and the same W), so the wire builder
    can evaluate any rank's needed set from the replicated build.
    """
    tb = np.asarray(sb.d_tb_cells.get() if hasattr(sb.d_tb_cells, 'get')
                    else sb.d_tb_cells, dtype=np.int64)
    Nz = sb.shape[2]
    if axis == 2:
        ax = tb % Nz
    elif axis == 1:
        ax = (tb // Nz) % sb.shape[1]
    else:
        ax = tb // (sb.shape[1] * Nz)
    inv = _window_inv(sb.shape[axis], own_start, own_count, ghost)
    rows = np.flatnonzero(inv[ax] >= 0)
    W = sb._tb_W.get()
    return np.unique(W[rows].tocoo().col)


def build_slab_surfel(sb, axis: int, own_start: int, own_count: int,
                      ghost: int = 3):
    """Slab-filtered SurfelBoundary clone (patch 64 sec. 2).

    Slices the BUILT full-domain surfel state onto a wrap window
    [own_start-ghost, own_start+own_count+ghost) along `axis` — never
    rebuilds geometry (the prism march's periodic wrap would be wrong at
    slab edges; slicing is bit-faithful to the replicated build). The
    build_slab_ibb pattern (runner S6) applied to the surfel stack.

    Facets are kept iff ALL their CSR cells lie inside the window; the
    build asserts that every facet relevant to OWN-correct output (any
    cell within own+-1) is kept, and that the wall-law sample envelope
    of those facets stays inside the window along the slab axis.

    Returns a SlabSurfelBoundary whose hot chain (sanitize/mask/tau_sgs/
    inject/apply_and_advect/zero_dead) is the production one on slab
    arrays. Surface output (write_surface/facet_traction) is guarded off
    — the MPI surface channel is a follow-up (64 sec. 7).
    """
    from types import SimpleNamespace
    from src.boundary.surfel_boundary import SurfelBoundary
    from src.kernels.surfel_d3q27 import SurfelKernelD3Q27
    from src.boundary.surfel_transport import N_PAIR

    xp = sb.xp
    full_shape = sb.shape
    n_ax = full_shape[axis]
    idx = np.arange(own_start - ghost, own_start + own_count + ghost) % n_ax
    W = idx.size
    if W > n_ax:
        raise ValueError(
            f"slab surfel: window own {own_count} + 2*ghost {ghost} = {W} "
            f"exceeds the axis extent {n_ax} — the wrap window would "
            f"duplicate cells. Axis too thin for this rank count/ghost "
            f"(production note: span Nz=16 gives L0 z=16 = exactly "
            f"feasible at 2 ranks; Nz=14 is NOT — patch 64)")
    slab_shape = tuple(W if a == axis else full_shape[a] for a in range(3))
    n_slab = int(np.prod(slab_shape))
    inv = np.full(n_ax, -1, dtype=np.int64)
    inv[idx] = np.arange(W)

    def cells_slice(arr):
        """(…, N) device/host array -> (…, N_slab), wrap window on axis."""
        lead = arr.shape[:-1]
        a3 = arr.reshape(lead + full_shape)
        take = [slice(None)] * (len(lead) + 3)
        mod = np if isinstance(arr, np.ndarray) else xp
        take[len(lead) + axis] = mod.asarray(idx)
        return a3[tuple(take)].reshape(lead + (n_slab,)).copy()

    def remap_flat(cells_global):
        """Global flat cell index -> slab flat; -1 outside the window."""
        c = np.asarray(cells_global.get() if hasattr(cells_global, 'get')
                       else cells_global, dtype=np.int64)
        Nx, Ny, Nz = full_shape
        z = c % Nz
        y = (c // Nz) % Ny
        x = c // (Ny * Nz)
        co = np.stack([x, y, z])
        w = inv[co[axis]]
        co = co.copy()
        co[axis] = w
        out = (co[0] * slab_shape[1] + co[1]) * slab_shape[2] + co[2]
        out[w < 0] = -1
        return out

    k = sb.kernel
    if getattr(k, 'wm_mode', 0) != 0:
        raise NotImplementedError(
            "slab surfel: the wall-model input filter carries per-facet "
            "state (u_wm/utau_prev) that is not margin-exchanged — wm off "
            "only under MPI (patch 64 sec. 9)")
    # ── facet filter: keep iff ALL CSR cells inside the window ──────
    # The CSR is facet-major contiguous ((facet, pair) key order), so
    # per-facet segments are [fac_lo[f], fac_lo[f+1]) and reduceat is
    # exact — no python loop (span16 L3 has O(1e5) facets).
    indptr_h = np.asarray(k.indptr.get())
    cell_h = np.asarray(k.cell.get(), dtype=np.int64)
    wgt_h = np.asarray(k.wgt.get())
    cell_slab = remap_flat(cell_h)
    n_f = k.n_f
    seg = indptr_h.reshape(-1)                      # (n_f*N_PAIR + 1,)
    fac_lo = seg[:-1].reshape(n_f, N_PAIR)[:, 0]
    if fac_lo[0] != 0 or not (np.diff(fac_lo) >= 0).all():
        raise AssertionError("surfel CSR is not facet-major contiguous")
    keep = np.minimum.reduceat(cell_slab, fac_lo) >= 0
    kept = np.flatnonzero(keep)

    # own-relevance assertion: facets with any cell at own+-1 must be kept
    Nz_ = full_shape[2]
    ax_of = (cell_h % Nz_ if axis == 2 else
             (cell_h // Nz_) % full_shape[1] if axis == 1 else
             cell_h // (full_shape[1] * Nz_))
    rel = ((ax_of - (own_start - 1)) % n_ax) < (own_count + 2)
    facet_rel = np.logical_or.reduceat(rel, fac_lo)
    bad = facet_rel & ~keep
    if bad.any():
        raise ValueError(
            f"slab surfel: {int(bad.sum())} facets touch the own range "
            f"but have cells outside the window — ghost {ghost} too small")

    # wall-law sample envelope along the slab axis (kernel samples at
    # cen + sample_h*n with trilinear support 1). Checked only for OWN-
    # relevant facets — window-edge ghost facets are recomputed junk the
    # deposit never uses, and their envelope legitimately leaves the
    # window.
    cen_h = np.asarray(k.cen.get())
    nrm_h = np.asarray(k.nrm.get())
    chk = np.flatnonzero(keep & facet_rel)
    reach = (float(k.f.sample_h) + float(k.h_law)) * np.abs(
        nrm_h[chk][:, axis]) + 1.0
    pos_w = (cen_h[chk][:, axis] - (own_start - ghost)) % n_ax
    if ((pos_w - reach < 0) | (pos_w + reach > W)).any():
        raise ValueError(
            "slab surfel: wall-law sample envelope of an own-relevant "
            f"facet escapes the window along axis {axis} — raise ghost "
            f"above {ghost} (normals have a large slab-axis component)")

    # CSR rebuild for kept facets — vectorized row mask over the
    # facet-major arrays (key order preserved by construction)
    facet_of_row = np.repeat(np.arange(n_f), np.diff(fac_lo, append=seg[-1]))
    row_keep = keep[facet_of_row]
    new_cell = cell_slab[row_keep]
    new_wgt = wgt_h[row_keep]
    counts_all = np.diff(seg)                       # per (facet, pair) key
    key_keep = np.repeat(keep, N_PAIR)
    new_indptr = np.concatenate(
        [[0], np.cumsum(counts_all[key_keep], dtype=np.int64)])

    # ── slab kernel (shared RawKernels; per-facet rows filtered) ────
    sk = SurfelKernelD3Q27.__new__(SurfelKernelD3Q27)
    sk._k = k._k
    sk.block = k.block
    for name in ('mode', 'law_id', 'law_iters', 'h_law', 'nu',
                 'y_plus_min', 'fric_dir', 'fb_mode', 'wm_mode', 'wm_tf',
                 '_per'):
        setattr(sk, name, getattr(k, name))
    sk.n_f = int(kept.size)
    sk.shape = slab_shape
    sk.N = n_slab
    sk.indptr = xp.asarray(new_indptr)
    sk.cell = xp.asarray(new_cell.astype(np.int32))
    sk.wgt = xp.asarray(new_wgt)
    sk.nrm = xp.asarray(np.ascontiguousarray(nrm_h[kept]))
    sk.area = k.area[xp.asarray(kept)].copy()
    cen_slab = cen_h[kept].copy()
    cen_slab[:, axis] = (cen_slab[:, axis] - (own_start - ghost)) % n_ax
    sk.cen = xp.asarray(np.ascontiguousarray(cen_slab))
    sk.Vsum = k.Vsum[xp.asarray(kept)].copy()
    sk.g_field = cells_slice(k.g_field)
    sk.G_in = xp.zeros((sk.n_f, 27), dtype=xp.float64)
    sk.G_out = xp.zeros((sk.n_f, 27), dtype=xp.float64)
    sk.Q = xp.zeros((27, n_slab), dtype=xp.float64)
    sk.tau_out = xp.zeros(sk.n_f, dtype=xp.float64)
    sk.fb_out = xp.zeros(sk.n_f, dtype=xp.uint8)
    sk.u_wm = xp.zeros((sk.n_f, 3), dtype=xp.float64)
    sk.utau_prev = xp.zeros(sk.n_f, dtype=xp.float64)
    sk._wm_seed = 1
    sk.f = SimpleNamespace(
        sample_h=k.f.sample_h,
        cdotn=np.ascontiguousarray(np.asarray(k.f.cdotn)[kept]))

    # ── slab boundary object ────────────────────────────────────────
    out = SlabSurfelBoundary.__new__(SlabSurfelBoundary)
    out.xp = xp
    out.shape = slab_shape
    out.kernel = sk
    out.n_facets = sk.n_f
    out.d_live = cells_slice(sb.d_live)
    out.d_dead = cells_slice(sb.d_dead)
    out.d_dV = cells_slice(sb.d_dV)
    out.dV_h = cells_slice(np.ascontiguousarray(
        sb.dV_h.reshape(-1))).reshape(slab_shape)
    out.live_h = cells_slice(np.ascontiguousarray(
        sb.live_h.reshape(-1))).reshape(slab_shape)
    out._solid_mask_dev = None
    out._d_CC = None
    out._force = None
    for name in ('params', 'dv_min', 'tau_model_on', 'collide_path',
                 'q_inf', 'p_ref', 'coord_origin', 'coord_spacing'):
        setattr(out, name, getattr(sb, name))
    out._taum_summary = getattr(sb, '_taum_summary', '')
    out.facets = sk.f                       # hot chain reads sample_h only
    out.surfels = None
    out.triangles_lu = None                 # write_surface guarded off

    # tau-model band (patch 64 stage ii): window row filter + PHANTOM
    # column extension. tau_out is one-substep persistent per-facet state
    # (64 sec. 9: finite ghost cannot close it), so the W column space is
    # extended to EVERY facet the kept rows reference — columns whose
    # facet lies outside the window become phantom slots fed exclusively
    # by the per-substep neighbor exchange (SurfelSlabLevel.taum_*), and
    # kept-but-neighbor-owned columns are refreshed by the same exchange.
    if getattr(sb, 'tau_model_on', False):
        tb_slab = remap_flat(sb.d_tb_cells)
        rows = np.flatnonzero(tb_slab >= 0)
        out.d_tb_cells = xp.asarray(tb_slab[rows])
        out.d_tb_fs = sb.d_tb_fs[xp.asarray(rows)].copy()
        out.d_tb_normal = sb.d_tb_normal[xp.asarray(rows)].copy()
        out.d_tb_C = sb.d_tb_C
        out.d_tb_Wl = sb.d_tb_Wl
        Wh = sb._tb_W.get()                 # cupyx csr -> scipy (build-time)
        Wr = Wh[rows]
        needed = np.unique(Wr.tocoo().col)  # global gids, sorted
        kept_pos = np.full(n_f, -1, dtype=np.int64)
        kept_pos[kept] = np.arange(kept.size)
        loc = kept_pos[needed]
        local_slots = np.flatnonzero(loc >= 0)
        phantom_slots = np.flatnonzero(loc < 0)
        out.tb_needed_gids = needed
        out.tb_phantom_gids = needed[phantom_slots]
        out.d_tb_local_slots = xp.asarray(local_slots)
        out.d_tb_local_rows = xp.asarray(loc[local_slots])
        out.d_tb_phantom_slots = xp.asarray(phantom_slots)
        out._tb_tau_ext = xp.zeros(needed.size, dtype=xp.float64)
        import cupyx.scipy.sparse as _cs
        out._tb_W = _cs.csr_matrix(Wr[:, needed].tocsr())
        out._sig_last = None
    else:
        out._sig_last = None

    # ownership (deterministic across ranks): facet anchor = its FIRST
    # CSR cell's slab-axis coordinate, owned iff inside the own range
    anchor_ax = ax_of[fac_lo[kept]]
    out.facet_owned = xp.asarray(
        (((anchor_ax - own_start) % n_ax) < own_count))
    out.facet_gids = kept
    out.slab_axis = axis
    return out


from src.boundary.surfel_boundary import SurfelBoundary as _SB  # noqa: E402


class SlabSurfelBoundary(_SB):
    """SurfelBoundary on a wrap-window slab (patch 64).

    Built ONLY by build_slab_surfel (never __init__). Differences from
    the full object: `last_force` sums OWNED facets only (ghost-strip
    facets are deterministic duplicates — the cross-rank Allreduce then
    counts the body force once; the partial-sum order makes this an
    ulp-level match to the 1-rank sum, not bitwise — 64 sec. 3), and the
    surface output channel is guarded off until the MPI surface gather.
    """

    def inject_tau_model(self, f_post, u, tau_bar: float) -> None:
        """Band injection through the EXTENDED tau vector (stage ii).

        Kept columns are copied from the kernel state (the exchange has
        already overwritten neighbor-owned kept rows in kernel.tau_out);
        phantom columns keep whatever taum_complete received. Step 0 (no
        exchange yet) is all-zeros on both = the single-GPU convention.
        """
        ext = self._tb_tau_ext
        ext[self.d_tb_local_slots] = self.kernel.tau_out[self.d_tb_local_rows]
        super().inject_tau_model(f_post, u, tau_bar, tau_vec=ext)

    def last_force(self):
        if getattr(self, '_force', None) is None:
            return np.zeros(3)              # no facet pass yet this run
        k = self.kernel
        xp = self.xp
        from src.boundary.surfel_transport import C27 as _C
        cdotn = xp.asarray(k.f.cdotn)
        own = self.facet_owned[:, None]
        G = ((k.G_in * (cdotn < 0) - k.G_out * (cdotn > 0)) * own
             ).sum(axis=0)
        F = xp.asarray(_C.astype(np.float64)).T @ G
        return np.asarray(F.get() if hasattr(F, 'get') else F, dtype=float)

    def write_surface(self, path: str) -> int:
        raise NotImplementedError(
            "slab surfel: surface output is the MPI surface-gather "
            "follow-up (patch_notes/surfel/64 sec. 7)")

    def facet_traction(self, *a, **k):
        raise NotImplementedError(
            "slab surfel: facet_traction is rank-local — use the MPI "
            "surface gather (patch_notes/surfel/64 sec. 7)")


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
