"""
Multi-Level Grid — Nested Time-Stepping Orchestrator

Manages M grid levels with recursive nested time-stepping, providing
a unified advance() interface compatible with the existing Simulation
class. One call to advance() performs exactly one coarse timestep,
which internally triggers 2^k fine steps at each level k.

Physical Process (Lagrava Sandoval, Sec. 4.4.3, Sec. 5.1.2):

    One coarse timestep (t → t + δt_c):

    ┌─ Level 0 ────────────────────────────────────────────────────┐
    │  1. Save f_prev (for temporal interpolation)                 │
    │  2. Collide + Stream + BC  (full domain, t → t+δt_c)        │
    │                                                              │
    │  ┌─ Level 1 (2 fine steps) ──────────────────────────────┐   │
    │  │  3. C→F half-step  (temporal interp: f = avg(f_prev,f))│   │
    │  │  4. Collide + Stream + BC  (t → t+δt_f)               │   │
    │  │     ┌─ Level 2 (2 finer steps) ────────────────────┐  │   │
    │  │     │  ... (recursive)                              │  │   │
    │  │     └───────────────────────────────────────────────┘  │   │
    │  │  5. C→F full-step  (no temporal interp needed)         │   │
    │  │  6. Collide + Stream + BC  (t+δt_f → t+δt_c)          │   │
    │  │     ┌─ Level 2 (2 finer steps) ────────────────────┐  │   │
    │  │     │  ... (recursive)                              │  │   │
    │  │     └───────────────────────────────────────────────┘  │   │
    │  │  7. F→C: overwrite coarse overlap with fine result     │   │
    │  └───────────────────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────────────────┘

Integration:
    MultiLevelGrid exposes the same interface as Simulation:
        advance(), rho, u, f, step_count
    so that main.py and OutputManager work without modification.

    main.py loop (unchanged):
        for step in range(start, end):
            sim.advance()          # sim = MultiLevelGrid or Simulation
            output.process(step, sim)

Reference:
    Lagrava Sandoval, Ch. 5.1 (implementation ideas)
    Geier et al. (2015), Sec. 6 (5-level sphere)

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.solver.simulation import Simulation
    from src.grid.coupling import GridCoupling


class MultiLevelGrid:
    """Orchestrates nested time-stepping across M grid levels.

    Owns M Simulation objects (one per level) and M-1 GridCoupling
    objects (one per adjacent pair). Provides a Simulation-compatible
    interface so that the main time loop is unaware of multi-level
    structure.

    Nested time-stepping cost per coarse step:
        Level 0:  1 step
        Level 1:  2 steps
        Level 2:  4 steps
        ...
        Level k:  2^k steps
        Total sub-steps: 2^M - 1

    Args:
        levels: List of Simulation objects, index 0 = coarsest.
                Each must have set_distribution() already called.
        couplings: List of GridCoupling objects, index i couples
                   levels[i] (coarse) with levels[i+1] (fine).
                   Length must be len(levels) - 1.

    Raises:
        ValueError: If levels/couplings lengths are inconsistent,
                    or if any level is not ready.

    Example:
        >>> mlg = MultiLevelGrid(levels=[sim_0, sim_1], couplings=[coupling_01])
        >>> for step in range(1000):
        ...     mlg.advance()
        ...     print(f"Step {step}: ρ_max = {mlg.rho.max():.6f}")
    """

    def __init__(
        self,
        levels: List['Simulation'],
        couplings: List['GridCoupling'],
    ) -> None:
        # ── Validate ─────────────────────────────────────────────
        if len(couplings) != len(levels) - 1:
            raise ValueError(
                f"Need exactly {len(levels)-1} couplings for {len(levels)} "
                f"levels, got {len(couplings)}."
            )
        for i, lev in enumerate(levels):
            if not lev.is_ready:
                raise ValueError(
                    f"Level {i} Simulation is not ready. "
                    f"Call set_distribution() first."
                )

        self._levels = levels
        self._couplings = couplings
        self._num_levels = len(levels)
        self._step_count = 0

        # ── Allocate f_prev buffers for temporal interpolation ───
        # Each level (except the finest) needs f_prev for C→F half-step.
        # f_prev[k] stores level k's f before the collide-stream step.
        self._f_prev: List[Optional['npt.NDArray']] = []
        for i, lev in enumerate(levels):
            if i < self._num_levels - 1:
                xp = lev.xp
                self._f_prev.append(xp.copy(lev.f))
            else:
                self._f_prev.append(None)  # finest level doesn't need f_prev

    # =================================================================
    # Simulation-compatible interface
    # =================================================================

    def advance(self) -> None:
        """Perform one coarse timestep with nested fine stepping.

        After this call:
            - All levels are at time t + δt_0
            - Level 0's f has been updated with F→C feedback
            - self.rho, self.u reflect Level 0's macroscopic state
            - self.step_count is incremented

        Physical time advanced: δt_0 (one coarsest-level timestep).
        Total sub-steps performed: 2^(M-1) + 2^(M-2) + ... + 1 = 2^M - 1.
        """
        coarse = self._levels[0]

        # ── Save f_prev for level 0 (temporal interpolation) ─────
        xp = coarse.xp
        xp.copyto(self._f_prev[0], coarse.f)

        # ── Advance coarse level (full domain) ───────────────────
        coarse.advance()

        # ── Recursively advance all finer levels ─────────────────
        if self._num_levels > 1:
            self._advance_fine(level_k=1)

        self._step_count += 1

    def _advance_fine(self, level_k: int) -> None:
        """Recursively advance level_k (two fine steps per coarse step).

        This implements the nested time-stepping algorithm from
        Lagrava Sec. 4.4.3. For each coarse step at level k-1,
        level k performs two steps, with C→F coupling before each
        and F→C coupling after both.

        If level k+1 exists, it is recursively advanced (2 steps)
        within each of level k's steps — yielding 4 steps total
        for level k+1 per coarse step.

        Args:
            level_k: Fine level index (1, 2, ..., M-1).
        """
        coupling = self._couplings[level_k - 1]
        sim_coarse = self._levels[level_k - 1]
        sim_fine = self._levels[level_k]
        has_finer = (level_k + 1 < self._num_levels)

        # ═════════════════════════════════════════════════════════
        # Fine step #1: t → t + δt_f  (half of coarse step)
        # ═════════════════════════════════════════════════════════

        # C→F coupling with temporal interpolation (half-step)
        coupling.coarse_to_fine(
            sim_coarse.f, sim_fine.f,
            is_half_step=True,
            f_coarse_prev=self._f_prev[level_k - 1],
        )

        # Save f_prev for this level (if even finer levels exist)
        if has_finer:
            xp = sim_fine.xp
            xp.copyto(self._f_prev[level_k], sim_fine.f)

        # Advance fine level
        sim_fine.advance()

        # Recurse into finer levels
        if has_finer:
            self._advance_fine(level_k + 1)

        # ═════════════════════════════════════════════════════════
        # Fine step #2: t + δt_f → t + δt_c  (second half)
        # ═════════════════════════════════════════════════════════

        # C→F coupling without temporal interpolation (full step)
        coupling.coarse_to_fine(
            sim_coarse.f, sim_fine.f,
            is_half_step=False,
        )

        # Save f_prev for this level (if even finer levels exist)
        if has_finer:
            xp = sim_fine.xp
            xp.copyto(self._f_prev[level_k], sim_fine.f)

        # Advance fine level
        sim_fine.advance()

        # Recurse into finer levels
        if has_finer:
            self._advance_fine(level_k + 1)

        # ═════════════════════════════════════════════════════════
        # F→C feedback: overwrite coarse overlap with fine data
        # ═════════════════════════════════════════════════════════
        coupling.fine_to_coarse(sim_fine.f, sim_coarse.f)

    # =================================================================
    # Properties (delegate to Level 0 for Simulation compatibility)
    # =================================================================

    @property
    def rho(self) -> Optional['npt.NDArray']:
        """Density field from the coarsest level (Level 0).

        This is what OutputManager and VTK writers access.
        For multi-level VTK output (Phase E), each level's rho
        will be accessed separately via get_level().
        """
        return self._levels[0].rho

    @property
    def u(self) -> Optional['npt.NDArray']:
        """Velocity field from the coarsest level (Level 0)."""
        return self._levels[0].u

    @property
    def f(self) -> Optional['npt.NDArray']:
        """Distribution function from the coarsest level (Level 0)."""
        return self._levels[0].f

    @property
    def f_post(self) -> Optional['npt.NDArray']:
        """Post-collision distribution from Level 0."""
        return self._levels[0].f_post

    @property
    def body_force(self) -> Optional['npt.NDArray']:
        """Body force from Level 0 (for ALM compatibility)."""
        return self._levels[0].body_force

    @property
    def step_count(self) -> int:
        """Number of coarse steps completed."""
        return self._step_count

    @property
    def is_ready(self) -> bool:
        """Whether all levels have distributions set."""
        return all(lev.is_ready for lev in self._levels)

    @property
    def tau(self) -> float:
        """Relaxation time of Level 0."""
        return self._levels[0].tau

    @property
    def domain_shape(self) -> Tuple[int, ...]:
        """Domain shape of Level 0."""
        return self._levels[0].domain_shape

    # =================================================================
    # Level access (for multi-level output, diagnostics)
    # =================================================================

    @property
    def num_levels(self) -> int:
        """Total number of grid levels."""
        return self._num_levels

    def get_level(self, k: int) -> 'Simulation':
        """Get the Simulation object for level k.

        Useful for multi-level VTK output where each level's
        fields need to be written separately.

        Args:
            k: Level index (0 = coarsest).

        Returns:
            Simulation object for that level.
        """
        if not 0 <= k < self._num_levels:
            raise IndexError(
                f"Level {k} out of range [0, {self._num_levels - 1}]"
            )
        return self._levels[k]

    def get_coupling(self, k: int) -> 'GridCoupling':
        """Get the GridCoupling for level pair (k, k+1).

        Args:
            k: Coarse level index in the pair.

        Returns:
            GridCoupling for the pair (k, k+1).
        """
        if not 0 <= k < self._num_levels - 1:
            raise IndexError(
                f"Coupling {k} out of range [0, {self._num_levels - 2}]"
            )
        return self._couplings[k]

    # =================================================================
    # Diagnostics
    # =================================================================

    def summary(self) -> str:
        """Human-readable summary of the multi-level grid."""
        lines = [
            f"MultiLevelGrid: {self._num_levels} levels",
            f"  Coarse step count: {self._step_count}",
            f"  Sub-steps per coarse step: {2**self._num_levels - 1}",
            "",
        ]
        for i, lev in enumerate(self._levels):
            sub_steps = 2 ** i
            lines.append(
                f"  Level {i}: shape={lev.domain_shape}, "
                f"tau={lev.tau:.4f}, "
                f"steps_per_coarse={sub_steps}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MultiLevelGrid(num_levels={self._num_levels}, "
            f"step_count={self._step_count})"
        )