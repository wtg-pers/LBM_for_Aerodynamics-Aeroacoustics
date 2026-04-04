"""
Level Scaling for Multi-Level Grid

Handles rescaling of physical quantities (τ, ν, f^neq) between adjacent
grid levels under convective (acoustic) scaling.

Physical Background (Lagrava Sandoval, Sec. 4.3.2–4.3.3):
    Given a refinement factor of 2 between adjacent levels:
        δx_f = δx_c / 2    (spatial step halved)
        δt_f = δt_c / 2    (temporal step halved, convective scaling)

    Convective scaling preserves:
        - Velocity:  u_f = u_c           (continuous across interface)
        - Density:   ρ_f = ρ_c           (continuous across interface)
        - Pressure:  p_f = p_c           (continuous across interface)

    But viscosity must be rescaled to keep ν_phys invariant:
        ν_lat(k) = cs² · (τ_k - 0.5)    (lattice-unit viscosity at level k)
        ν_phys   = ν_lat(k) · δx_k² / δt_k = const across all levels

    This yields the relaxation time at level k:
        τ_k = 2^k · (τ_0 - 0.5) + 0.5

    Adjacent-level recurrence (Cheylan et al., Eq. 12):
        τ_{k+1} = 2 · τ_k - 0.5

    The non-equilibrium part of the distribution function is rescaled:
        f^neq_{i,f} = (τ_f / 2τ_c) · f^neq_{i,c}    (coarse → fine, Eq. 4.14)
        f^neq_{i,c} = (2τ_c / τ_f) · f^neq_{i,f}     (fine → coarse, Eq. 4.15)

    Note: f^eq depends only on (ρ, u), which are continuous across the
    interface, so f^eq needs NO rescaling.

Convention:
    - Level 0 = coarsest grid (reference level)
    - Level M-1 = finest grid
    - "coarse" and "fine" always refer to an adjacent pair (level k, level k+1)

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


@dataclass(frozen=True)
class LevelUnits:
    """Physical parameters for a single grid level.

    All quantities are in lattice units of the respective level.

    Attributes:
        level: Grid level index (0 = coarsest).
        dx: Spatial step relative to the reference level 0.       [m]
        dt: Temporal step relative to the reference level 0.       [s]
        nu: Kinematic viscosity in this level's lattice units.     [Δx²/Δt]
        tau: BGK relaxation time in this level's lattice units.    [Δt]
        omega: Relaxation rate ω = 1/τ.                            [1/Δt]
    """
    level: int
    dx: float
    dt: float
    nu: float
    tau: float
    omega: float


class LevelScaler:
    """Computes level-wise scaling of τ, ν, and f^neq for multi-level grids.

    Supports an arbitrary number of levels (up to max_levels) with a
    fixed refinement ratio of 2 between adjacent levels. All conversions
    follow the convective (acoustic) scaling convention.

    Physical Scaling Relations (refinement ratio = 2):
        Level k → Level k+1:
            δx_{k+1} = δx_k / 2
            δt_{k+1} = δt_k / 2
            ν_lat_{k+1} = 2 · ν_lat_k  (lattice-unit viscosity doubles)
            τ_{k+1} = 2 · τ_k - 0.5    (Cheylan et al., Eq. 12)

    Usage:
        >>> scaler = LevelScaler(tau_0=0.56, num_levels=5)
        >>> scaler.get_level_units(0).tau  # coarsest
        0.56
        >>> scaler.get_level_units(4).tau  # finest
        >>> scaler.rescale_f_neq_c2f(f_neq_c, level_coarse=2)  # level 2→3

    Args:
        tau_0: Relaxation time on the coarsest level (level 0).    [Δt_0]
        num_levels: Total number of grid levels (M).
        dx_0: Spatial step on level 0 (default 1.0 in lattice units).
        dt_0: Temporal step on level 0 (default 1.0 in lattice units).
        cs2: Speed of sound squared (1/3 for standard LBM).

    Raises:
        ValueError: If num_levels < 1 or tau_0 <= 0.5.
    """

    # Refinement factor between adjacent levels
    REFINEMENT_RATIO: int = 2

    def __init__(
        self,
        tau_0: float,
        num_levels: int = 1,
        dx_0: float = 1.0,
        dt_0: float = 1.0,
        cs2: float = 1.0 / 3.0,
    ) -> None:
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if tau_0 <= 0.5:
            raise ValueError(
                f"tau_0 must be > 0.5 for stability, got {tau_0}. "
                f"(ω = 1/τ must be < 2)"
            )

        self._num_levels: int = num_levels
        self._cs2: float = cs2

        # ── Precompute units for every level ─────────────────────
        # Physical viscosity invariance (convective scaling):
        #   ν_phys = cs² · (τ_k - 0.5) · δx_k² / δt_k = const
        #
        # With δx_k = δx_0/2^k, δt_k = δt_0/2^k:
        #   (τ_k - 0.5) · δx_k = const   (since δx²/δt = δx · δx/δt = δx · const)
        #   → (τ_k - 0.5) = 2^k · (τ_0 - 0.5)
        #   → τ_k = 2^k · (τ_0 - 0.5) + 0.5
        #
        # Recurrence between adjacent levels (Cheylan et al., Eq. 12):
        #   τ_{k+1} = 2 · τ_k - 0.5
        self._level_units: list[LevelUnits] = []

        for k in range(num_levels):
            factor = 2 ** k
            dx_k = dx_0 / factor
            dt_k = dt_0 / factor

            # τ_k = 2^k · (τ_0 - 0.5) + 0.5
            tau_k = factor * (tau_0 - 0.5) + 0.5
            nu_k = cs2 * (tau_k - 0.5)       # lattice viscosity at level k
            omega_k = 1.0 / tau_k

            self._level_units.append(
                LevelUnits(
                    level=k,
                    dx=dx_k,
                    dt=dt_k,
                    nu=nu_k,
                    tau=tau_k,
                    omega=omega_k,
                )
            )

    # =================================================================
    # Access
    # =================================================================

    @property
    def num_levels(self) -> int:
        """Total number of grid levels (M)."""
        return self._num_levels

    def get_level_units(self, level: int) -> LevelUnits:
        """Get the physical parameters for a given level.

        Args:
            level: Grid level index (0 = coarsest, M-1 = finest).

        Returns:
            LevelUnits dataclass with dx, dt, nu, tau, omega.

        Raises:
            IndexError: If level is out of range [0, M-1].
        """
        if not 0 <= level < self._num_levels:
            raise IndexError(
                f"Level {level} out of range [0, {self._num_levels - 1}]"
            )
        return self._level_units[level]

    # =================================================================
    # f^neq Rescaling
    # =================================================================

    def rescaling_factor_c2f(self, level_coarse: int) -> float:
        """Compute the rescaling factor for f^neq from coarse to fine.

        f^neq_{fine} = factor · f^neq_{coarse}

        From Lagrava Eq. 4.14:
            factor = τ_f / (2 · τ_c)

        where τ_f is the fine level's relaxation time and τ_c is the
        coarse level's relaxation time.

        Args:
            level_coarse: Index of the coarse level (fine = level_coarse + 1).

        Returns:
            Rescaling factor (dimensionless).
        """
        self._validate_adjacent_pair(level_coarse)
        tau_c = self._level_units[level_coarse].tau
        tau_f = self._level_units[level_coarse + 1].tau
        return tau_f / (2.0 * tau_c)

    def rescaling_factor_f2c(self, level_coarse: int) -> float:
        """Compute the rescaling factor for f^neq from fine to coarse.

        f^neq_{coarse} = factor · f^neq_{fine}

        From Lagrava Eq. 4.15:
            factor = 2 · τ_c / τ_f

        Args:
            level_coarse: Index of the coarse level (fine = level_coarse + 1).

        Returns:
            Rescaling factor (dimensionless).
        """
        self._validate_adjacent_pair(level_coarse)
        tau_c = self._level_units[level_coarse].tau
        tau_f = self._level_units[level_coarse + 1].tau
        return (2.0 * tau_c) / tau_f

    def rescale_f_neq_c2f(
        self,
        f_neq_coarse: 'npt.NDArray',
        level_coarse: int,
    ) -> 'npt.NDArray':
        """Rescale non-equilibrium distribution from coarse to fine level.

        Physical meaning: The non-equilibrium part is proportional to
        velocity gradients, which scale differently between resolution
        levels. This rescaling ensures continuity of the stress tensor
        across the grid interface.

        f^neq_{i,fine} = (τ_f / 2τ_c) · f^neq_{i,coarse}   (Lagrava Eq. 4.14)

        Args:
            f_neq_coarse: Non-equilibrium distributions on the coarse grid.
                          Shape: (Q, ...) where Q = number of velocities.
            level_coarse: Index of the coarse level.

        Returns:
            Rescaled f^neq for the fine level. Same shape as input.
        """
        factor = self.rescaling_factor_c2f(level_coarse)
        return factor * f_neq_coarse

    def rescale_f_neq_f2c(
        self,
        f_neq_fine: 'npt.NDArray',
        level_coarse: int,
    ) -> 'npt.NDArray':
        """Rescale non-equilibrium distribution from fine to coarse level.

        f^neq_{i,coarse} = (2τ_c / τ_f) · f^neq_{i,fine}   (Lagrava Eq. 4.15)

        Args:
            f_neq_fine: Non-equilibrium distributions on the fine grid.
                        Shape: (Q, ...) where Q = number of velocities.
            level_coarse: Index of the coarse level.

        Returns:
            Rescaled f^neq for the coarse level. Same shape as input.
        """
        factor = self.rescaling_factor_f2c(level_coarse)
        return factor * f_neq_fine

    # =================================================================
    # Convenience: reconstruct full f from (f_eq, f_neq) with rescaling
    # =================================================================

    def reconstruct_f_c2f(
        self,
        f_eq: 'npt.NDArray',
        f_neq_coarse: 'npt.NDArray',
        level_coarse: int,
    ) -> 'npt.NDArray':
        """Reconstruct fine-level distribution from coarse-level data.

        f_{i,fine} = f^eq_i(ρ, u) + (τ_f / 2τ_c) · f^neq_{i,coarse}

        Since f^eq depends only on (ρ, u) which are continuous across the
        interface, the same f^eq is used for both coarse and fine levels.

        Args:
            f_eq: Equilibrium distribution (same for both levels).
                  Shape: (Q, ...).
            f_neq_coarse: Non-equilibrium part from the coarse grid.
                          Shape: (Q, ...).
            level_coarse: Index of the coarse level.

        Returns:
            Full distribution for the fine grid. Shape: (Q, ...).
        """
        return f_eq + self.rescale_f_neq_c2f(f_neq_coarse, level_coarse)

    def reconstruct_f_f2c(
        self,
        f_eq: 'npt.NDArray',
        f_neq_fine: 'npt.NDArray',
        level_coarse: int,
    ) -> 'npt.NDArray':
        """Reconstruct coarse-level distribution from fine-level data.

        f_{i,coarse} = f^eq_i(ρ, u) + (2τ_c / τ_f) · f^neq_{i,fine}

        Args:
            f_eq: Equilibrium distribution (same for both levels).
                  Shape: (Q, ...).
            f_neq_fine: Non-equilibrium part from the fine grid.
                        Shape: (Q, ...).
            level_coarse: Index of the coarse level.

        Returns:
            Full distribution for the coarse grid. Shape: (Q, ...).
        """
        return f_eq + self.rescale_f_neq_f2c(f_neq_fine, level_coarse)

    # =================================================================
    # Utility
    # =================================================================

    def summary(self) -> str:
        """Return a human-readable summary of all levels."""
        lines = [
            f"LevelScaler: {self._num_levels} levels, "
            f"refinement ratio = {self.REFINEMENT_RATIO}",
            f"{'Level':>6} {'dx':>10} {'dt':>10} {'nu':>12} "
            f"{'tau':>10} {'omega':>10}",
            "-" * 65,
        ]
        for lu in self._level_units:
            lines.append(
                f"{lu.level:>6d} {lu.dx:>10.6f} {lu.dt:>10.6f} "
                f"{lu.nu:>12.8f} {lu.tau:>10.6f} {lu.omega:>10.6f}"
            )
        return "\n".join(lines)

    def _validate_adjacent_pair(self, level_coarse: int) -> None:
        """Validate that level_coarse and level_coarse+1 both exist."""
        if not 0 <= level_coarse < self._num_levels - 1:
            raise IndexError(
                f"Adjacent pair ({level_coarse}, {level_coarse + 1}) invalid. "
                f"Valid coarse levels: [0, {self._num_levels - 2}]"
            )

    def __repr__(self) -> str:
        return (
            f"LevelScaler(tau_0={self._level_units[0].tau:.4f}, "
            f"num_levels={self._num_levels})"
        )