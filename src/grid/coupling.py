"""
Grid Coupling Engine — Coarse↔Fine Data Transfer

Performs the two-way coupling between adjacent grid levels in the
multi-domain grid refinement system.

Array Convention:
    This solver uses (Q, Nx, Ny, Nz) array layout throughout.
    axis 0 = Q (velocity directions)
    axis 1 = x (streamwise)
    axis 2 = y (wall-normal)
    axis 3 = z (spanwise)

Physical Process (Lagrava Sandoval, Sec. 4.4):

    Coarse → Fine (C→F):
        1. Extract f from coarse overlap region
        2. Temporal interpolation (for half-step): f = 0.5·(f_prev + f_now)
        3. Compute ρ, u → f^eq → f^neq = f - f^eq
        4. Rescale: f^neq_fine = (τ_f / 2τ_c) · f^neq_coarse   (Eq. 4.14)
        5. Place at even fine indices + interpolate odd indices
        6. Write result to fine boundary strip

    Fine → Coarse (F→C):
        1. Extract f from fine at coarse-coincident nodes
        2. Filter f^neq (low-pass, ρ and u NOT filtered)
        3. Rescale: f^neq_coarse = (2τ_c / τ_f) · f^neq_fine   (Eq. 4.15)
        4. Reconstruct + write to coarse overlap strip

Reference:
    Lagrava Sandoval, Sec. 4.3.3, Sec. 4.4
    Cheylan et al. (2019), Eq. 11-13

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

# Phase 1a Stage A: boundary-only C→F upsamples just a thin coarse slab per fine
# face instead of the whole fine volume. Bit-identical on the written strips
# (verified CPU), but measured NEUTRAL on GPU (C2F.L4 unchanged: the upsample
# was not the C→F bottleneck — the macro/f_eq decomposition on the full coarse
# sub, or launch/sync, dominates). So DEFAULT is the proven full-volume path;
# boundary-only is opt-in (MLG_C2F_BOUNDARY=1) until steps 1-4 are also
# boundary-scoped and GPU bit-identity is confirmed. See 02_phase1a_stageA.
_C2F_BOUNDARY_ONLY = os.environ.get("MLG_C2F_BOUNDARY", "0") == "1"

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.grid.overlap_manager import OverlapRegion
    from src.grid.level_scaling import LevelScaler
    from src.grid.interpolation import InterpolationScheme


class GridCoupling:
    """Two-way coupling engine for one pair of adjacent grid levels.

    All arrays follow the solver convention: f shape = (Q, Nx, Ny, Nz).

    Args:
        xp: Array module (numpy or cupy).
        lattice: Lattice model providing c (dim, Q), w (Q,), cs2.
        region: OverlapRegion from Phase A (index geometry).
        scaler: LevelScaler (τ/ν rescaling between levels).
        interpolation: Spatial interpolation scheme (cubic recommended).
        filter_level: F→C low-pass filter depth (0/1/2).
    """

    def __init__(
        self,
        xp: 'ModuleType',
        lattice: object,
        region: 'OverlapRegion',
        scaler: 'LevelScaler',
        interpolation: 'InterpolationScheme',
        filter_level: int = 1,
    ) -> None:
        self._xp = xp
        self._region = region
        self._scaler = scaler
        self._interp = interpolation
        self._filter_level = filter_level

        # ── CUDA interpolation kernel (if available) ─────────────
        self._cuda_interp = None
        if xp.__name__ == 'cupy':
            try:
                from src.kernels.interpolation_d3q27 import CubicInterpolationKernel3D
                self._cuda_interp = CubicInterpolationKernel3D()
            except Exception:
                pass

        # ── Lattice constants on device ──────────────────────────
        self._dtype = lattice.dtype
        self._c = xp.asarray(lattice.c, dtype=self._dtype)    # (dim, Q)
        w = xp.asarray(lattice.w, dtype=self._dtype)           # (Q,)
        self._w = w
        self._cs2: float = float(lattice.cs2)
        self._Q: int = int(lattice.Q)
        self._dim: int = int(lattice.dim)

        # Broadcast weights: (Q,) → (Q, 1, 1, 1)
        self._w_bc = w.reshape((-1,) + (1,) * 3)

        # ── Precompute rescaling factors ─────────────────────────
        lc = region.level_coarse
        self._factor_c2f = scaler.rescaling_factor_c2f(lc)
        self._factor_f2c = scaler.rescaling_factor_f2c(lc)

        # ── Fine domain slices on coarse grid ────────────────────
        # Solver array: f.shape = (Q, Nx, Ny, Nz)
        # axis 1 = x, axis 2 = y, axis 3 = z
        fdc = region.fine_domain_coarse
        self._coarse_sub_slices = (
            slice(None),                              # Q
            slice(fdc.x_start, fdc.x_end + 1),        # x
            slice(fdc.y_start, fdc.y_end + 1),        # y
            slice(fdc.z_start, fdc.z_end + 1),        # z
        )

        # ── Fine boundary slices in (x, y, z) order ─────────────
        Nx_f, Ny_f, Nz_f = region.fine_shape
        ow_f = region.overlap_width * region.REFINE_RATIO

        self._fine_bnd_slices = {
            'x_min': (slice(0, ow_f),            slice(None),              slice(None)),
            'x_max': (slice(Nx_f - ow_f, Nx_f),  slice(None),              slice(None)),
            'y_min': (slice(None),                slice(0, ow_f),           slice(None)),
            'y_max': (slice(None),                slice(Ny_f - ow_f, Ny_f), slice(None)),
            'z_min': (slice(None),                slice(None),              slice(0, ow_f)),
            'z_max': (slice(None),                slice(None),              slice(Nz_f - ow_f, Nz_f)),
        }

        # ── F→C target: excised region only (NOT full fine_domain) ─
        # Overlap strip retains native coarse values for smooth transition.
        # Excised region = fine_region (user-specified, without overlap).
        excised = region.excised
        self._excised_coarse_slices = (
            slice(None),                                    # Q
            slice(excised.x_start, excised.x_end + 1),      # x
            slice(excised.y_start, excised.y_end + 1),      # y
            slice(excised.z_start, excised.z_end + 1),      # z
        )

        # Local offset within f_reconstructed (which covers fine_domain_coarse)
        # to extract only the excised sub-region.
        # Use actual offset (not overlap_width) to handle boundary clipping.
        ox = excised.x_start - fdc.x_start
        oy = excised.y_start - fdc.y_start
        oz = excised.z_start - fdc.z_start
        ex_nx = excised.x_end - excised.x_start + 1
        ex_ny = excised.y_end - excised.y_start + 1
        ex_nz = excised.z_end - excised.z_start + 1
        self._excised_local_slices = (
            slice(None),                 # Q
            slice(ox, ox + ex_nx),       # x
            slice(oy, oy + ex_ny),       # y
            slice(oz, oz + ex_nz),       # z
        )

        # ── Phase 1a Stage A: boundary-only C→F face specs ───────
        # Each of the 6 fine faces needs only a thin coarse slab: `cw` coarse
        # nodes deep on the face-normal axis, full on the two tangential axes.
        # cw = ow_f/2 + 2 covers the cubic stencil (radius 3 fine = 1.5 coarse),
        # so the written strip is bit-identical to the full-volume upsample.
        # Each spec = (coarse-slab slice, fine-write slice, slab-read slice), 4D.
        R = region.REFINE_RATIO
        ow_f_b = region.overlap_width * R
        Cx, Cy, Cz = region.fine_domain_coarse.shape
        Nx_fb, Ny_fb, Nz_fb = region.fine_shape
        cw = ow_f_b // 2 + 2

        def _face(axis: int, side: str, C: int, Nf: int):
            cwx = min(cw, C)              # coarse nodes on the normal axis
            wl = min(ow_f_b, Nf)         # fine strip thickness written
            sl = 2 * cwx - 1             # upsampled slab length (normal axis)
            if side == 'min':
                c, w, r = slice(0, cwx), slice(0, wl), slice(0, wl)
            else:
                c, w, r = slice(C - cwx, C), slice(Nf - wl, Nf), slice(sl - wl, sl)
            a = slice(None)
            if axis == 1:
                return ((a, c, a, a), (a, w, a, a), (a, r, a, a))
            if axis == 2:
                return ((a, a, c, a), (a, a, w, a), (a, a, r, a))
            return ((a, a, a, c), (a, a, a, w), (a, a, a, r))

        self._bnd_face_specs = [
            _face(1, 'min', Cx, Nx_fb), _face(1, 'max', Cx, Nx_fb),
            _face(2, 'min', Cy, Ny_fb), _face(2, 'max', Cy, Ny_fb),
            _face(3, 'min', Cz, Nz_fb), _face(3, 'max', Cz, Nz_fb),
        ]

    # =================================================================
    # Public: Coarse → Fine
    # =================================================================

    def coarse_to_fine(
        self,
        f_coarse: 'npt.NDArray',
        f_fine: 'npt.NDArray',
        *,
        is_half_step: bool = False,
        f_coarse_prev: Optional['npt.NDArray'] = None,
    ) -> None:
        """Inject coarse data into the fine grid boundary strip.

        Args:
            f_coarse: shape (Q, Nx_c, Ny_c, Nz_c).
            f_fine: shape (Q, Nx_f, Ny_f, Nz_f). Modified IN-PLACE.
            is_half_step: True for temporal interpolation.
            f_coarse_prev: Previous coarse f (required if is_half_step).
        """
        xp = self._xp

        # ── 1. Extract coarse sub-volume ─────────────────────────
        f_sub = f_coarse[self._coarse_sub_slices].copy()

        # ── 2. Temporal interpolation (half-step) ────────────────
        if is_half_step:
            if f_coarse_prev is None:
                raise ValueError(
                    "f_coarse_prev is required for half-step temporal "
                    "interpolation."
                )
            f_sub_prev = f_coarse_prev[self._coarse_sub_slices]
            f_sub = 0.5 * (f_sub_prev + f_sub)

        # ── 3. Decompose: f^eq + f^neq ──────────────────────────
        rho, u = self._compute_macroscopic(f_sub)
        f_eq = self._compute_f_eq(rho, u)
        f_neq = f_sub - f_eq

        # ── 4. Rescale f^neq for fine level ──────────────────────
        f_neq_fine = self._factor_c2f * f_neq

        # ── 5+6. Upsample & write the 6 fine boundary strips ─────
        # Boundary-only (Phase 1a Stage A): upsample a thin coarse slab per
        # face rather than the full fine volume. Bit-identical on the written
        # strips; MLG_C2F_FULL=1 restores the full-volume path.
        coarse_nodes = f_eq + f_neq_fine
        if _C2F_BOUNDARY_ONLY:
            self._upsample_boundary_into(coarse_nodes, f_fine)
        else:
            f_fine_full = self._upsample_block(coarse_nodes)
            for (sx, sy, sz) in self._fine_bnd_slices.values():
                f_fine[:, sx, sy, sz] = f_fine_full[:, sx, sy, sz]

    # =================================================================
    # Public: Fine → Coarse
    # =================================================================

    def fine_to_coarse(
        self,
        f_fine: 'npt.NDArray',
        f_coarse: 'npt.NDArray',
    ) -> None:
        """Overwrite EXCISED coarse nodes with fine-grid data.

        Only the excised region (interior of fine_region) is overwritten.
        The overlap strip retains its native coarse-computed values,
        ensuring a smooth transition at the grid interface.

        Physical process:
            1. Extract fine f at coarse-coincident nodes (full fine domain)
            2. Compute ρ, u → f^eq, f^neq
            3. Filter f^neq (low-pass, ρ/u NOT filtered)
            4. Rescale f^neq for coarse level
            5. Write ONLY to excised coarse nodes (not overlap strip)

        Args:
            f_fine: shape (Q, Nx_f, Ny_f, Nz_f). Read only.
            f_coarse: shape (Q, Nx_c, Ny_c, Nz_c). Modified IN-PLACE.
        """
        R = self._region.REFINE_RATIO

        # ── Extract coarse-coincident fine nodes ─────────────────
        f_fine_at_coarse = f_fine[:, 0::R, 0::R, 0::R]

        # ── Compute macroscopic (unfiltered) ─────────────────────
        rho_f, u_f = self._compute_macroscopic(f_fine_at_coarse)
        f_eq = self._compute_f_eq(rho_f, u_f)
        f_neq_f = f_fine_at_coarse - f_eq

        # ── Filter f^neq ────────────────────────────────────────
        f_neq_filtered = self._filter_f_neq(f_neq_f)

        # ── Rescale for coarse level ─────────────────────────────
        f_neq_c = self._factor_f2c * f_neq_filtered

        # ── Reconstruct full fine_domain_coarse region ───────────
        f_reconstructed = f_eq + f_neq_c

        # ── Write ONLY excised region to coarse ──────────────────
        # Overlap strip keeps native coarse values → smooth interface
        f_coarse[self._excised_coarse_slices] = \
            f_reconstructed[self._excised_local_slices]

    # =================================================================
    # Private: Equilibrium & Macroscopic
    # =================================================================

    def _compute_macroscopic(self, f: 'npt.NDArray') -> tuple:
        """ρ = Σ f_i,  u = (Σ c_i f_i) / ρ"""
        xp = self._xp
        rho = xp.sum(f, axis=0)
        momentum = xp.einsum('dq,q...->d...', self._c, f)
        u = momentum / rho
        return rho, u

    def _compute_f_eq(self, rho: 'npt.NDArray', u: 'npt.NDArray') -> 'npt.NDArray':
        """Maxwellian equilibrium: f^eq = w·ρ·(1 + cu/cs² + ...)"""
        xp = self._xp
        cu = xp.einsum('dq,d...->q...', self._c, u)
        usqr = xp.sum(u * u, axis=0)
        cs2 = self._cs2
        return self._w_bc * rho * (
            1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * usqr / cs2
        )

    # =================================================================
    # Private: Upsample coarse → fine resolution
    # =================================================================

    def _upsample_block(self, coarse_nodes: 'npt.NDArray') -> 'npt.NDArray':
        """Upsample a coarse block (Q, bx, by, bz) → fine (Q, 2bx-1, 2by-1,
        2bz-1): place coarse at even fine indices, cubic-interpolate the rest
        (axis 1=x, 2=y, 3=z). Block edges are treated as domain boundaries by
        the interp stencil, so a written value is bit-identical to the
        full-volume result wherever the block covers that value's stencil.
        """
        xp = self._xp
        _, bx, by, bz = coarse_nodes.shape
        nx, ny, nz = 2 * bx - 1, 2 * by - 1, 2 * bz - 1

        f_fine = xp.zeros((self._Q, nx, ny, nz), dtype=coarse_nodes.dtype)
        f_fine[:, 0::2, 0::2, 0::2] = coarse_nodes

        if self._cuda_interp is not None:
            self._cuda_interp.interpolate(f_fine, self._Q, nx, ny, nz)
        else:
            f_fine = self._interp.interpolate_1d(xp, f_fine, axis=1)
            f_fine = self._interp.interpolate_1d(xp, f_fine, axis=2)
            f_fine = self._interp.interpolate_1d(xp, f_fine, axis=3)

        return f_fine

    def _upsample_boundary_into(
        self, coarse_nodes: 'npt.NDArray', f_fine: 'npt.NDArray',
    ) -> None:
        """Boundary-only C→F (Phase 1a Stage A): upsample one thin coarse slab
        per fine face and write the 6 boundary strips of `f_fine` in place.

        Bit-identical to `_upsample_block` over the full volume on the written
        strips — each face slab is full on its two tangential axes and `cw`
        coarse nodes deep on the normal axis, where `cw = ow_f/2 + 2` guarantees
        the cubic stencil of every written fine node is inside the slab (and the
        slab's outer edge coincides with the true domain boundary, so the
        boundary stencil matches). The fine interior, which C→F never writes, is
        skipped. Verified: scratchpad/phase1a_stageA_ab.py.
        """
        for c_sl, w_sl, r_sl in self._bnd_face_specs:
            slab = self._upsample_block(coarse_nodes[c_sl])
            f_fine[w_sl] = slab[r_sl]

    # =================================================================
    # Private: F→C Low-pass Filter
    # =================================================================

    def _filter_f_neq(self, f_neq: 'npt.NDArray') -> 'npt.NDArray':
        """Low-pass filter on f^neq. Array: (Q, Nx, Ny, Nz)."""
        if self._filter_level == 0:
            return f_neq

        xp = self._xp
        filtered = f_neq.copy()

        # axes: 1=x, 2=y, 3=z. Interior only.
        center = f_neq[:, 1:-1, 1:-1, 1:-1]
        xm  = f_neq[:, :-2,  1:-1, 1:-1]
        xp_ = f_neq[:, 2:,   1:-1, 1:-1]
        ym  = f_neq[:, 1:-1, :-2,  1:-1]
        yp_ = f_neq[:, 1:-1, 2:,   1:-1]
        zm  = f_neq[:, 1:-1, 1:-1, :-2]
        zp_ = f_neq[:, 1:-1, 1:-1, 2:]

        if self._filter_level == 1:
            filtered[:, 1:-1, 1:-1, 1:-1] = (
                center + xm + xp_ + ym + yp_ + zm + zp_
            ) / 7.0
        elif self._filter_level >= 2:
            xy_mm = f_neq[:, :-2,  :-2,  1:-1]
            xy_mp = f_neq[:, :-2,  2:,   1:-1]
            xy_pm = f_neq[:, 2:,   :-2,  1:-1]
            xy_pp = f_neq[:, 2:,   2:,   1:-1]
            xz_mm = f_neq[:, :-2,  1:-1, :-2]
            xz_mp = f_neq[:, :-2,  1:-1, 2:]
            xz_pm = f_neq[:, 2:,   1:-1, :-2]
            xz_pp = f_neq[:, 2:,   1:-1, 2:]
            yz_mm = f_neq[:, 1:-1, :-2,  :-2]
            yz_mp = f_neq[:, 1:-1, :-2,  2:]
            yz_pm = f_neq[:, 1:-1, 2:,   :-2]
            yz_pp = f_neq[:, 1:-1, 2:,   2:]

            filtered[:, 1:-1, 1:-1, 1:-1] = (
                center + xm + xp_ + ym + yp_ + zm + zp_
                + xy_mm + xy_mp + xy_pm + xy_pp
                + xz_mm + xz_mp + xz_pm + xz_pp
                + yz_mm + yz_mp + yz_pm + yz_pp
            ) / 19.0

        return filtered