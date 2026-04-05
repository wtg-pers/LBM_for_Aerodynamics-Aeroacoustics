"""
Grid Coupling Engine — Coarse↔Fine Data Transfer

Performs the two-way coupling between adjacent grid levels in the
multi-domain grid refinement system.

Physical Process (Lagrava Sandoval, Sec. 4.4):

    Coarse → Fine (C→F):
        At the fine grid boundary strip, coarse data is injected:
        1. Extract f from coarse overlap region
        2. Temporal interpolation (for half-step): f = 0.5·(f_prev + f_now)
        3. Compute ρ, u → f^eq → f^neq = f - f^eq
        4. Rescale: f^neq_fine = (τ_f / 2τ_c) · f^neq_coarse   (Eq. 4.14)
        5. Place at even fine indices (coarse-coincident nodes)
        6. Interpolate odd fine indices (non-coincident nodes)
        7. Write result to fine boundary strip

    Fine → Coarse (F→C):
        At the coarse overlap strip, fine data overwrites coarse:
        1. Extract f from fine nodes corresponding to coarse overlap
        2. Filter f^neq (low-pass, ρ and u NOT filtered)  (Lagrava Ch.7)
        3. Rescale: f^neq_coarse = (2τ_c / τ_f) · f^neq_fine   (Eq. 4.15)
        4. Reconstruct: f_coarse = f^eq(ρ_f, u_f) + f^neq_coarse
        5. Write to coarse overlap nodes

    Key invariants:
        - f^eq depends only on (ρ, u), which are continuous → no rescaling
        - f^neq ∝ ∇u → must be rescaled between resolution levels
        - Filtering removes fine-scale information that coarse cannot resolve

Dependencies:
    - OverlapRegion (Phase A): index mapping and slice geometry
    - LevelScaler: rescaling factors (τ_f / 2τ_c) and (2τ_c / τ_f)
    - InterpolationScheme: spatial interpolation (cubic or compact 2nd order)

Reference:
    Lagrava Sandoval, Sec. 4.3.3 (rescaling), Sec. 4.4 (coupling detail)
    Cheylan et al. (2019), Eq. 11-13 (grid refinement in LBM)

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.grid.overlap_manager import OverlapRegion
    from src.grid.level_scaling import LevelScaler
    from src.grid.interpolation import InterpolationScheme


class GridCoupling:
    """Two-way coupling engine for one pair of adjacent grid levels.

    Handles the complete data transfer between a coarse level and
    its immediately finer neighbor. One GridCoupling instance exists
    per adjacent level pair.

    All array operations are vectorized (CuPy/NumPy compatible).
    The coupling is collision-model independent — it operates entirely
    in distribution function space using f^eq / f^neq decomposition.

    Args:
        xp: Array module (numpy or cupy).
        lattice: Lattice model providing c (dim, Q), w (Q,), cs2.
        region: OverlapRegion from Phase A (index geometry).
        scaler: LevelScaler (τ/ν rescaling between levels).
        interpolation: Spatial interpolation scheme (cubic recommended).
        filter_level: Neighbor depth for F→C low-pass filter.
                      0 = no filter, 1 = face neighbors (default),
                      2 = face + edge neighbors.

    Example:
        >>> coupling = GridCoupling(xp, lattice, region, scaler,
        ...                         CubicInterpolation(), filter_level=1)
        >>> # In nested time-stepping loop:
        >>> coupling.coarse_to_fine(f_c, f_f, is_half_step=True, f_c_prev=f_c_old)
        >>> # ... fine steps ...
        >>> coupling.fine_to_coarse(f_f, f_c)
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

        # ── Lattice constants on device ──────────────────────────
        self._c = xp.asarray(lattice.c, dtype=xp.float64)    # (dim, Q)
        w = xp.asarray(lattice.w, dtype=xp.float64)           # (Q,)
        self._w = w
        self._cs2: float = float(lattice.cs2)
        self._Q: int = int(lattice.Q)
        self._dim: int = int(lattice.dim)

        # Broadcast weights for f_eq computation: (Q,) → (Q, 1, 1, 1)
        self._w_bc = w.reshape((-1,) + (1,) * 3)

        # ── Precompute rescaling factors ─────────────────────────
        lc = region.level_coarse
        self._factor_c2f = scaler.rescaling_factor_c2f(lc)
        self._factor_f2c = scaler.rescaling_factor_f2c(lc)

        # ── Fine domain slices on coarse grid ────────────────────
        fdc = region.fine_domain_coarse
        self._coarse_sub_slices = (
            slice(None),  # Q dimension
            slice(fdc.z_start, fdc.z_end + 1),
            slice(fdc.y_start, fdc.y_end + 1),
            slice(fdc.x_start, fdc.x_end + 1),
        )

        # ── Fine boundary slices (where C→F data is injected) ───
        self._fine_bnd_slices = region.get_fine_boundary_slices()

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

        This is called BEFORE each fine-grid collide-and-stream step.
        For the first fine step (half-step), temporal interpolation is
        used to estimate coarse data at the intermediate time.

        Physical process:
            1. Extract coarse sub-volume covering fine domain
            2. Temporal interpolation (if half-step)
            3. Decompose: f = f^eq(ρ,u) + f^neq
            4. Rescale f^neq for fine level
            5. Upsample to fine resolution (even indices + interpolation)
            6. Write ONLY the boundary strips of f_fine

        Args:
            f_coarse: Coarse distribution, shape (Q, Nz_c, Ny_c, Nx_c).
            f_fine: Fine distribution, shape (Q, Nz_f, Ny_f, Nx_f).
                    Modified IN-PLACE (boundary strip overwritten).
            is_half_step: True for the first of two fine steps.
                          Requires f_coarse_prev for temporal interpolation.
            f_coarse_prev: Coarse distribution at previous timestep.
                           Required when is_half_step=True.
        """
        xp = self._xp

        # ── 1. Extract coarse sub-volume ─────────────────────────
        f_sub = f_coarse[self._coarse_sub_slices].copy()

        # ── 2. Temporal interpolation (half-step) ────────────────
        if is_half_step:
            if f_coarse_prev is None:
                raise ValueError(
                    "f_coarse_prev is required for half-step temporal "
                    "interpolation. Store f before the coarse step."
                )
            f_sub_prev = f_coarse_prev[self._coarse_sub_slices]
            f_sub = 0.5 * (f_sub_prev + f_sub)

        # ── 3. Decompose: f^eq + f^neq ──────────────────────────
        rho, u = self._compute_macroscopic(f_sub)
        f_eq = self._compute_f_eq(rho, u)
        f_neq = f_sub - f_eq

        # ── 4. Rescale f^neq for fine level ──────────────────────
        f_neq_fine = self._factor_c2f * f_neq

        # ── 5. Upsample to fine resolution ───────────────────────
        f_fine_full = self._upsample_to_fine(f_eq, f_neq_fine)

        # ── 6. Write boundary strips to f_fine ───────────────────
        for face_name, slices in self._fine_bnd_slices.items():
            sz, sy, sx = slices
            f_fine[:, sz, sy, sx] = f_fine_full[:, sz, sy, sx]

    # =================================================================
    # Public: Fine → Coarse
    # =================================================================

    def fine_to_coarse(
        self,
        f_fine: 'npt.NDArray',
        f_coarse: 'npt.NDArray',
    ) -> None:
        """Overwrite coarse overlap nodes with fine-grid data.

        This is called AFTER the fine grid completes its two sub-steps
        (reaching the same physical time as coarse). The fine data is
        filtered, rescaled, and written to the coarse overlap strip.

        Physical process:
            1. Extract fine f at coarse-coincident overlap nodes
            2. Filter f^neq (low-pass on f^neq only, NOT on ρ/u)
            3. Compute ρ, u from fine f (unfiltered for macroscopic)
            4. Compute f^eq(ρ, u)
            5. Rescale filtered f^neq for coarse level
            6. Reconstruct: f_coarse = f^eq + rescaled f^neq
            7. Write to coarse overlap strip

        Args:
            f_fine: Fine distribution, shape (Q, Nz_f, Ny_f, Nx_f).
                    Read only.
            f_coarse: Coarse distribution, shape (Q, Nz_c, Ny_c, Nx_c).
                      Modified IN-PLACE (overlap strip overwritten).
        """
        xp = self._xp
        region = self._region
        R = region.REFINE_RATIO

        # ── Extract coarse-coincident fine nodes ─────────────────
        # These are the fine nodes at even indices that map to coarse nodes
        # within the fine domain.
        f_fine_at_coarse = f_fine[:, 0::R, 0::R, 0::R]

        # ── Compute macroscopic from unfiltered fine f ───────────
        # ρ, u must NOT be filtered (Lagrava Ch.7)
        rho_f, u_f = self._compute_macroscopic(f_fine_at_coarse)
        f_eq = self._compute_f_eq(rho_f, u_f)
        f_neq_f = f_fine_at_coarse - f_eq

        # ── Filter f^neq ────────────────────────────────────────
        f_neq_filtered = self._filter_f_neq(f_neq_f)

        # ── Rescale for coarse level ─────────────────────────────
        f_neq_c = self._factor_f2c * f_neq_filtered

        # ── Reconstruct coarse f ─────────────────────────────────
        f_reconstructed = f_eq + f_neq_c

        # ── Write to coarse overlap strip ────────────────────────
        # Map fine-at-coarse indices back to coarse grid indices
        fdc = region.fine_domain_coarse
        # f_fine_at_coarse has shape matching the coarse sub-volume
        # that covers the fine domain
        f_coarse[self._coarse_sub_slices] = f_reconstructed

    # =================================================================
    # Private: Equilibrium & Macroscopic
    # =================================================================

    def _compute_macroscopic(
        self, f: 'npt.NDArray',
    ) -> tuple:
        """Compute density and velocity from distribution function.

        ρ = Σ_i f_i
        u = (Σ_i c_i f_i) / ρ

        Args:
            f: Distribution function, shape (Q, Nz, Ny, Nx).

        Returns:
            (rho, u): Density (Nz,Ny,Nx), velocity (dim,Nz,Ny,Nx).
        """
        xp = self._xp
        rho = xp.sum(f, axis=0)
        momentum = xp.einsum('dq,q...->d...', self._c, f)
        u = momentum / rho
        return rho, u

    def _compute_f_eq(
        self, rho: 'npt.NDArray', u: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Compute Maxwellian equilibrium distribution.

        f^eq_i = w_i · ρ · (1 + c·u/cs² + (c·u)²/(2cs⁴) − |u|²/(2cs²))

        This is collision-model independent — BGK, Cumulant, and MRT
        all share the same second-order equilibrium.

        Args:
            rho: Density, shape (Nz, Ny, Nx).
            u: Velocity, shape (dim, Nz, Ny, Nx).

        Returns:
            f_eq: Equilibrium distribution, shape (Q, Nz, Ny, Nx).
        """
        xp = self._xp
        cu = xp.einsum('dq,d...->q...', self._c, u)
        usqr = xp.sum(u * u, axis=0)
        cs2 = self._cs2

        f_eq = self._w_bc * rho * (
            1.0
            + cu / cs2
            + 0.5 * cu * cu / (cs2 * cs2)
            - 0.5 * usqr / cs2
        )
        return f_eq

    # =================================================================
    # Private: Upsample coarse → fine resolution
    # =================================================================

    def _upsample_to_fine(
        self,
        f_eq: 'npt.NDArray',
        f_neq_fine: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Create fine-resolution f from coarse-resolution (f_eq, f_neq).

        Process:
            1. Allocate fine-sized array
            2. Place f = f_eq + f_neq at even indices (coarse-coincident)
            3. Interpolate odd indices axis-by-axis using InterpolationScheme

        The interpolation is done as a tensor product of 1D passes:
            Pass 1 (x): fills nodes at (even_z, even_y, odd_x)
            Pass 2 (y): fills nodes at (even_z, odd_y, any_x)
            Pass 3 (z): fills nodes at (odd_z, any_y, any_x)

        After all three passes, every fine node has a value.

        Args:
            f_eq: Equilibrium at coarse resolution, shape (Q, nz, ny, nx).
            f_neq_fine: Rescaled non-equilibrium, same shape as f_eq.

        Returns:
            Fine-resolution distribution, shape (Q, Nz_f, Ny_f, Nx_f).
        """
        xp = self._xp
        Nx_f, Ny_f, Nz_f = self._region.fine_shape

        # Reconstruct f at coarse nodes
        f_coarse_nodes = f_eq + f_neq_fine

        # Allocate fine array
        f_fine = xp.zeros((self._Q, Nz_f, Ny_f, Nx_f), dtype=f_eq.dtype)

        # Place at even indices (coarse-coincident nodes)
        f_fine[:, 0::2, 0::2, 0::2] = f_coarse_nodes

        # Interpolate odd indices: three sequential 1D passes
        # axis=3 → x, axis=2 → y, axis=1 → z
        f_fine = self._interp.interpolate_1d(xp, f_fine, axis=3)
        f_fine = self._interp.interpolate_1d(xp, f_fine, axis=2)
        f_fine = self._interp.interpolate_1d(xp, f_fine, axis=1)

        return f_fine

    # =================================================================
    # Private: F→C Low-pass Filter
    # =================================================================

    def _filter_f_neq(
        self, f_neq: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Apply low-pass filter to non-equilibrium distribution.

        Removes fine-scale fluctuations that the coarse grid cannot
        resolve. Only f^neq is filtered — ρ and u are NOT filtered
        to avoid artificial viscosity increase (Lagrava Ch.7).

        Filter levels:
            0: No filtering (direct copy)
            1: Face neighbors (6 neighbors + center) / 7
            2: Face + edge neighbors (18 neighbors + center) / 19

        The filter is applied only to interior nodes where all neighbors
        exist. Boundary nodes retain their unfiltered values.

        Args:
            f_neq: Non-equilibrium distribution, shape (Q, Nz, Ny, Nx).

        Returns:
            Filtered f_neq, same shape.
        """
        if self._filter_level == 0:
            return f_neq

        xp = self._xp
        filtered = f_neq.copy()

        if self._filter_level >= 1:
            # ── Face neighbors (±1 along each axis) ──────────────
            # Interior nodes only (boundary retains original values)
            center = f_neq[:, 1:-1, 1:-1, 1:-1]
            xm = f_neq[:, 1:-1, 1:-1, :-2]
            xp_ = f_neq[:, 1:-1, 1:-1, 2:]
            ym = f_neq[:, 1:-1, :-2, 1:-1]
            yp_ = f_neq[:, 1:-1, 2:, 1:-1]
            zm = f_neq[:, :-2, 1:-1, 1:-1]
            zp_ = f_neq[:, 2:, 1:-1, 1:-1]

            if self._filter_level == 1:
                # 6 face neighbors + center = 7 nodes
                filtered[:, 1:-1, 1:-1, 1:-1] = (
                    center + xm + xp_ + ym + yp_ + zm + zp_
                ) / 7.0

            elif self._filter_level >= 2:
                # 6 face + 12 edge neighbors + center = 19 nodes
                # Edge neighbors (diagonal in 2D plane)
                xy_mm = f_neq[:, 1:-1, :-2, :-2]
                xy_mp = f_neq[:, 1:-1, :-2, 2:]
                xy_pm = f_neq[:, 1:-1, 2:, :-2]
                xy_pp = f_neq[:, 1:-1, 2:, 2:]
                xz_mm = f_neq[:, :-2, 1:-1, :-2]
                xz_mp = f_neq[:, :-2, 1:-1, 2:]
                xz_pm = f_neq[:, 2:, 1:-1, :-2]
                xz_pp = f_neq[:, 2:, 1:-1, 2:]
                yz_mm = f_neq[:, :-2, :-2, 1:-1]
                yz_mp = f_neq[:, :-2, 2:, 1:-1]
                yz_pm = f_neq[:, 2:, :-2, 1:-1]
                yz_pp = f_neq[:, 2:, 2:, 1:-1]

                filtered[:, 1:-1, 1:-1, 1:-1] = (
                    center + xm + xp_ + ym + yp_ + zm + zp_
                    + xy_mm + xy_mp + xy_pm + xy_pp
                    + xz_mm + xz_mp + xz_pm + xz_pp
                    + yz_mm + yz_mp + yz_pm + yz_pp
                ) / 19.0

        return filtered