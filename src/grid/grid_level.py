"""
Grid Level — Single Resolution Level for Multi-Level Grid

Each GridLevel is a self-contained LBM domain at a specific resolution,
owning its own distribution functions (f), macroscopic fields (ρ, u),
and references to its LBM operators (collision, streaming, BC).

Physical Context:
    In a multi-level grid system with M levels (0=coarsest, M-1=finest),
    each level has its own:
        - Spatial resolution:  δx_k = δx_0 / 2^k
        - Temporal resolution: δt_k = δt_0 / 2^k  (convective scaling)
        - Viscosity:           ν_k  = 2^k · ν_0
        - Relaxation time:     τ_k  from LevelScaler

    The level covers a specific physical region (bounding box) of the
    overall simulation domain. Only the coarsest level (level 0) spans
    the full domain; finer levels cover progressively smaller regions
    around areas of interest (e.g., near solid boundaries).

Design Principles:
    - GridLevel is collision-model agnostic (works with BGK, cumulant, MRT)
    - Each level manages its own arrays; no global arrays
    - Coupling between levels is handled externally (by MultiLevelGrid)
    - Streaming is local to each level (no cross-level streaming)

Reference:
    Lagrava Sandoval, Ch. 5.1: "Main implementation ideas"
    Geier et al., Sec. 6: 5-level grid for sphere simulation

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.grid.level_scaling import LevelUnits


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in physical coordinates.

    Defines the spatial extent of a grid level in the global
    coordinate system.

    Attributes:
        x_min: Minimum x-coordinate.  [m]
        x_max: Maximum x-coordinate.  [m]
        y_min: Minimum y-coordinate.  [m]
        y_max: Maximum y-coordinate.  [m]
        z_min: Minimum z-coordinate.  [m]
        z_max: Maximum z-coordinate.  [m]
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def extent(self) -> Tuple[float, float, float]:
        """Physical size in each direction (Lx, Ly, Lz) [m]."""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    def contains(self, x: float, y: float, z: float) -> bool:
        """Check if a point lies within this bounding box."""
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )

    def overlaps(self, other: 'BoundingBox') -> bool:
        """Check if two bounding boxes have any spatial overlap."""
        return (
            self.x_min < other.x_max and self.x_max > other.x_min
            and self.y_min < other.y_max and self.y_max > other.y_min
            and self.z_min < other.z_max and self.z_max > other.z_min
        )


class GridLevel:
    """Single resolution level in the multi-level grid system.

    A GridLevel is the fundamental unit of the MLG. It owns:
        - Distribution functions f (Q, Nz, Ny, Nx) on device (CuPy/NumPy)
        - Macroscopic fields ρ (Nz, Ny, Nx) and u (3, Nz, Ny, Nx)
        - Grid metadata (dx, dt, tau, bounding box)
        - References to LBM operators (set externally by the solver setup)

    The grid dimensions (Nx, Ny, Nz) are computed from the bounding box
    and spatial resolution δx_k.

    Args:
        xp: Array module (numpy or cupy).
        level: Grid level index (0 = coarsest).
        level_units: Physical parameters from UnitConverter.
        bounding_box: Physical extent of this level.
        Q: Number of lattice velocities (27 for D3Q27).
        dim: Number of spatial dimensions (3).

    Example:
        >>> from src.grid.level_scaling import LevelScaler
        >>> scaler = LevelScaler(tau_0=0.56, num_levels=3)
        >>> level_0 = GridLevel(np, 0, scaler.get_level_units(0), bbox, Q=27)
        >>> level_0.shape
        (10, 5, 5)
    """

    def __init__(
        self,
        xp: 'ModuleType',
        level: int,
        level_units: 'LevelUnits',
        bounding_box: BoundingBox,
        Q: int = 27,
        dim: int = 3,
    ) -> None:
        self.xp = xp
        self.level: int = level
        self.units: LevelUnits = level_units
        self.bbox: BoundingBox = bounding_box
        self.Q: int = Q
        self.dim: int = dim

        # ── Compute grid dimensions from bbox and dx ─────────────
        Lx, Ly, Lz = bounding_box.extent
        dx = level_units.dx

        self.Nx: int = round(Lx / dx)
        self.Ny: int = round(Ly / dx)
        self.Nz: int = round(Lz / dx)

        if self.Nx < 1 or self.Ny < 1 or self.Nz < 1:
            raise ValueError(
                f"Level {level}: Grid dimensions ({self.Nx}, {self.Ny}, {self.Nz}) "
                f"invalid. Bounding box extent ({Lx}, {Ly}, {Lz}) too small "
                f"for dx={dx}."
            )

        # ── Allocate arrays on device ────────────────────────────
        # Distribution function: f[q, z, y, x]
        # Using (Q, Nz, Ny, Nx) convention consistent with existing solver
        self.f: npt.NDArray = xp.zeros(
            (Q, self.Nz, self.Ny, self.Nx), dtype=xp.float64
        )

        # Macroscopic fields
        self.rho: npt.NDArray = xp.ones(
            (self.Nz, self.Ny, self.Nx), dtype=xp.float64
        )
        self.u: npt.NDArray = xp.zeros(
            (dim, self.Nz, self.Ny, self.Nx), dtype=xp.float64
        )

        # ── Solid mask (True = solid node) ───────────────────────
        self.solid_mask: npt.NDArray = xp.zeros(
            (self.Nz, self.Ny, self.Nx), dtype=xp.bool_
        )

        # ── Coupling interface storage ───────────────────────────
        # Previous timestep f at coupling boundaries (for temporal interpolation)
        # Allocated lazily by the coupling module when needed
        self._f_prev: Optional[npt.NDArray] = None

        # ── LBM operator references (set by solver setup) ───────
        # These are not owned by GridLevel; they are shared or configured
        # externally. GridLevel only holds references for convenience.
        self._collision = None
        self._streaming = None
        self._boundaries: list = []

    # =================================================================
    # Properties
    # =================================================================

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid dimensions (Nz, Ny, Nx) — matches array layout."""
        return (self.Nz, self.Ny, self.Nx)

    @property
    def num_nodes(self) -> int:
        """Total number of lattice nodes on this level."""
        return self.Nz * self.Ny * self.Nx

    @property
    def memory_bytes(self) -> int:
        """Estimated GPU/CPU memory usage in bytes."""
        n = self.num_nodes
        f_bytes = self.Q * n * 8  # float64
        field_bytes = (1 + self.dim) * n * 8  # ρ + u
        mask_bytes = n  # bool
        # f_prev if allocated
        prev_bytes = self.Q * n * 8 if self._f_prev is not None else 0
        return f_bytes + field_bytes + mask_bytes + prev_bytes

    @property
    def memory_mb(self) -> float:
        """Estimated memory usage in MB."""
        return self.memory_bytes / (1024 * 1024)

    @property
    def dx(self) -> float:
        """Spatial step for this level [m]."""
        return self.units.dx

    @property
    def dt(self) -> float:
        """Temporal step for this level [s]."""
        return self.units.dt

    @property
    def tau(self) -> float:
        """Relaxation time for this level [Δt]."""
        return self.units.tau

    @property
    def omega(self) -> float:
        """Relaxation rate for this level [1/Δt]."""
        return self.units.omega

    # =================================================================
    # Temporal interpolation buffer (for MLG coupling)
    # =================================================================

    def allocate_f_prev(self) -> None:
        """Allocate buffer for previous-timestep distributions.

        Required for temporal interpolation in coarse-to-fine coupling.
        The coarse level must store f at time t₀ and t₁ to interpolate
        at the fine grid's intermediate time t₀ + δt_f = t₀ + δt_c/2.

        See Lagrava Sandoval, Sec. 5.1.2, Fig. 5.5.
        """
        if self._f_prev is None:
            self._f_prev = self.xp.zeros_like(self.f)

    def save_f_to_prev(self) -> None:
        """Copy current f to the previous-timestep buffer.

        Called before a collide-and-stream step to preserve the
        state at time t₀ for temporal interpolation.
        """
        if self._f_prev is None:
            self.allocate_f_prev()
        self.xp.copyto(self._f_prev, self.f)

    @property
    def f_prev(self) -> Optional['npt.NDArray']:
        """Previous-timestep distribution (None if not allocated)."""
        return self._f_prev

    # =================================================================
    # LBM Operator References
    # =================================================================

    def set_collision(self, collision_op: Any) -> None:
        """Attach a collision operator to this level.

        Args:
            collision_op: Any CollisionOperator subclass instance.
        """
        self._collision = collision_op

    def set_streaming(self, streaming_op: Any) -> None:
        """Attach a streaming operator to this level.

        Args:
            streaming_op: Streaming operator instance.
        """
        self._streaming = streaming_op

    def set_boundaries(self, boundaries: list) -> None:
        """Attach boundary conditions to this level.

        Args:
            boundaries: List of BoundaryCondition instances.
        """
        self._boundaries = boundaries

    @property
    def collision(self) -> Any:
        """Get the collision operator for this level."""
        if self._collision is None:
            raise RuntimeError(
                f"Level {self.level}: Collision operator not set. "
                f"Call set_collision() first."
            )
        return self._collision

    @property
    def streaming(self) -> Any:
        """Get the streaming operator for this level."""
        if self._streaming is None:
            raise RuntimeError(
                f"Level {self.level}: Streaming operator not set. "
                f"Call set_streaming() first."
            )
        return self._streaming

    @property
    def boundaries(self) -> list:
        """Get boundary conditions for this level."""
        return self._boundaries

    # =================================================================
    # Initialization
    # =================================================================

    def initialize_equilibrium(
        self,
        rho_init: float = 1.0,
        u_init: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Initialize distributions to equilibrium state.

        Sets ρ and u to uniform values, then computes f = f^eq(ρ, u).

        Args:
            rho_init: Initial density [lattice units, dimensionless].
            u_init: Initial velocity (ux, uy, uz) [Δx/Δt].
                    Default (0, 0, 0).
        """
        xp = self.xp
        self.rho[:] = rho_init

        if u_init is not None:
            self.u[0, :] = u_init[0]
            self.u[1, :] = u_init[1]
            self.u[2, :] = u_init[2]
        else:
            self.u[:] = 0.0

        # Compute f^eq using the collision operator
        f_eq = self.collision.compute_equilibrium(self.rho, self.u)
        xp.copyto(self.f, f_eq)

    # =================================================================
    # Coordinate Mapping
    # =================================================================

    def physical_to_index(
        self, x: float, y: float, z: float
    ) -> Tuple[int, int, int]:
        """Convert physical coordinates to grid indices.

        Args:
            x, y, z: Physical coordinates [m].

        Returns:
            (iz, iy, ix): Grid indices in array convention (z, y, x).
        """
        dx = self.units.dx
        ix = round((x - self.bbox.x_min) / dx)
        iy = round((y - self.bbox.y_min) / dx)
        iz = round((z - self.bbox.z_min) / dx)
        return (iz, iy, ix)

    def index_to_physical(
        self, iz: int, iy: int, ix: int
    ) -> Tuple[float, float, float]:
        """Convert grid indices to physical coordinates.

        Args:
            iz, iy, ix: Grid indices.

        Returns:
            (x, y, z): Physical coordinates [m].
        """
        dx = self.units.dx
        x = self.bbox.x_min + ix * dx
        y = self.bbox.y_min + iy * dx
        z = self.bbox.z_min + iz * dx
        return (x, y, z)

    # =================================================================
    # Info & Debug
    # =================================================================

    def summary(self) -> str:
        """Human-readable summary of this grid level."""
        lines = [
            f"GridLevel {self.level}:",
            f"  Shape:     ({self.Nz} × {self.Ny} × {self.Nx}) = "
            f"{self.num_nodes:,} nodes",
            f"  Memory:    {self.memory_mb:.1f} MB",
            f"  dx={self.dx:.6f}, dt={self.dt:.6f}",
            f"  tau={self.tau:.6f}, omega={self.omega:.6f}, "
            f"nu={self.units.nu:.8f}",
            f"  BBox:      x=[{self.bbox.x_min}, {self.bbox.x_max}], "
            f"y=[{self.bbox.y_min}, {self.bbox.y_max}], "
            f"z=[{self.bbox.z_min}, {self.bbox.z_max}]",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GridLevel(level={self.level}, shape={self.shape}, "
            f"tau={self.tau:.4f}, nodes={self.num_nodes:,})"
        )