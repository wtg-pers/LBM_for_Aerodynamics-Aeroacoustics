"""
Grid Level (2D) — Single Resolution Level for 2D Multi-Level Grid

2D counterpart of `src/grid/grid_level.py`. Each GridLevel2D owns its
distribution functions f, macroscopic fields ρ and u, and references to
LBM operators for a single resolution level.

Array layout:
    f         (Q, Nx, Ny)         — matches StreamingPull / HalfwayBounceBack
    rho       (Nx, Ny)
    u         (dim=2, Nx, Ny)
    solid_mask(Nx, Ny)

LevelUnits and LevelScaler are shared with the 3D code (dimension-
agnostic scalar quantities).

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Any

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.grid.level_scaling import LevelUnits


@dataclass
class BoundingBox2D:
    """Axis-aligned 2D bounding box in physical coordinates.

    Attributes:
        x_min, x_max: x-coordinate extent.  [m]
        y_min, y_max: y-coordinate extent.  [m]
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def extent(self) -> Tuple[float, float]:
        """(Lx, Ly) physical size [m]."""
        return (self.x_max - self.x_min, self.y_max - self.y_min)

    def contains(self, x: float, y: float) -> bool:
        return (self.x_min <= x <= self.x_max
                and self.y_min <= y <= self.y_max)

    def overlaps(self, other: 'BoundingBox2D') -> bool:
        return (self.x_min < other.x_max and self.x_max > other.x_min
                and self.y_min < other.y_max and self.y_max > other.y_min)


class GridLevel2D:
    """Single resolution level in a 2D multi-level grid.

    Mirrors `GridLevel` (3D) but with 2D arrays and a 2D bounding box.
    LevelUnits is the same dataclass used in 3D.

    Args:
        xp: Array module (numpy or cupy).
        level: Grid level index (0 = coarsest).
        level_units: Physical parameters from LevelScaler.
        bounding_box: 2D physical extent of this level.
        Q: Number of lattice velocities (9 for D2Q9).
        dim: Spatial dimensions (2).

    Example:
        >>> from src.grid.level_scaling import LevelScaler
        >>> scaler = LevelScaler(tau_0=0.56, num_levels=3)
        >>> bbox = BoundingBox2D(0, 1.0, 0, 0.5)
        >>> g = GridLevel2D(np, 0, scaler.get_level_units(0), bbox, Q=9)
        >>> g.shape
        (Nx, Ny)
    """

    def __init__(
        self,
        xp: 'ModuleType',
        level: int,
        level_units: 'LevelUnits',
        bounding_box: BoundingBox2D,
        Q: int = 9,
        dim: int = 2,
        dtype=None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        import numpy as _np
        self.xp = xp
        self.level: int = level
        self.units = level_units
        self.bbox: BoundingBox2D = bounding_box
        self.Q: int = Q
        self.dim: int = dim
        self._dtype = dtype if dtype is not None else _np.float64

        # ── Grid dimensions: `shape` override (MLG uses fine_shape
        #    from OverlapRegion2D directly, which is node count = cells+1)
        #    else compute from bbox/dx (cell count convention).
        if shape is not None:
            self.Nx, self.Ny = int(shape[0]), int(shape[1])
        else:
            Lx, Ly = bounding_box.extent
            dx = level_units.dx
            self.Nx = round(Lx / dx)
            self.Ny = round(Ly / dx)

        if self.Nx < 1 or self.Ny < 1:
            raise ValueError(
                f"Level {level}: Grid dimensions ({self.Nx}, {self.Ny}) "
                f"invalid."
            )

        # ── Allocate arrays (convention: (Q, Nx, Ny) / (Nx, Ny)) ────
        self.f = xp.zeros((Q, self.Nx, self.Ny), dtype=self._dtype)
        self.rho = xp.ones((self.Nx, self.Ny), dtype=self._dtype)
        self.u = xp.zeros((dim, self.Nx, self.Ny), dtype=self._dtype)
        self.solid_mask = xp.zeros((self.Nx, self.Ny), dtype=xp.bool_)

        # Coupling: previous-timestep buffer (lazy)
        self._f_prev = None

        # LBM operator references (set externally)
        self._collision = None
        self._streaming = None
        self._boundaries: list = []

    # =================================================================
    # Properties
    # =================================================================

    @property
    def shape(self) -> Tuple[int, int]:
        """(Nx, Ny) — matches array layout."""
        return (self.Nx, self.Ny)

    @property
    def num_nodes(self) -> int:
        return self.Nx * self.Ny

    @property
    def memory_bytes(self) -> int:
        n = self.num_nodes
        bpe = 8 if self._dtype == __import__('numpy').float64 else 4
        f_bytes = self.Q * n * bpe
        field_bytes = (1 + self.dim) * n * bpe
        mask_bytes = n
        prev_bytes = self.Q * n * bpe if self._f_prev is not None else 0
        return f_bytes + field_bytes + mask_bytes + prev_bytes

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    @property
    def dx(self) -> float:
        return self.units.dx

    @property
    def dt(self) -> float:
        return self.units.dt

    @property
    def tau(self) -> float:
        return self.units.tau

    @property
    def omega(self) -> float:
        return self.units.omega

    # =================================================================
    # Temporal interpolation buffer (MLG coupling)
    # =================================================================

    def allocate_f_prev(self) -> None:
        """Allocate previous-timestep f buffer for temporal interpolation."""
        if self._f_prev is None:
            self._f_prev = self.xp.zeros_like(self.f)

    def save_f_to_prev(self) -> None:
        if self._f_prev is None:
            self.allocate_f_prev()
        self.xp.copyto(self._f_prev, self.f)

    @property
    def f_prev(self):
        return self._f_prev

    # =================================================================
    # LBM operator references
    # =================================================================

    def set_collision(self, collision_op: Any) -> None:
        self._collision = collision_op

    def set_streaming(self, streaming_op: Any) -> None:
        self._streaming = streaming_op

    def set_boundaries(self, boundaries: list) -> None:
        self._boundaries = boundaries

    @property
    def collision(self) -> Any:
        if self._collision is None:
            raise RuntimeError(
                f"Level {self.level}: collision op not set.")
        return self._collision

    @property
    def streaming(self) -> Any:
        if self._streaming is None:
            raise RuntimeError(
                f"Level {self.level}: streaming op not set.")
        return self._streaming

    @property
    def boundaries(self) -> list:
        return self._boundaries

    # =================================================================
    # Initialization
    # =================================================================

    def initialize_equilibrium(
        self,
        rho_init: float = 1.0,
        u_init: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Initialize f to Maxwellian equilibrium with uniform (ρ, u)."""
        xp = self.xp
        self.rho[:] = rho_init
        if u_init is not None:
            self.u[0, :] = u_init[0]
            self.u[1, :] = u_init[1]
        else:
            self.u[:] = 0.0
        f_eq = self.collision.compute_equilibrium(self.rho, self.u)
        xp.copyto(self.f, f_eq)

    # =================================================================
    # Coordinate Mapping
    # =================================================================

    def physical_to_index(
        self, x: float, y: float
    ) -> Tuple[int, int]:
        dx = self.units.dx
        ix = round((x - self.bbox.x_min) / dx)
        iy = round((y - self.bbox.y_min) / dx)
        return (ix, iy)

    def index_to_physical(
        self, ix: int, iy: int
    ) -> Tuple[float, float]:
        dx = self.units.dx
        return (self.bbox.x_min + ix * dx,
                self.bbox.y_min + iy * dx)

    # =================================================================
    # Info
    # =================================================================

    def summary(self) -> str:
        return "\n".join([
            f"GridLevel2D {self.level}:",
            f"  Shape:  ({self.Nx} × {self.Ny}) = {self.num_nodes:,} nodes",
            f"  Memory: {self.memory_mb:.1f} MB",
            f"  dx={self.dx:.6f}, dt={self.dt:.6f}",
            f"  tau={self.tau:.6f}, omega={self.omega:.6f}, "
            f"nu={self.units.nu:.8f}",
            f"  BBox:   x=[{self.bbox.x_min}, {self.bbox.x_max}], "
            f"y=[{self.bbox.y_min}, {self.bbox.y_max}]",
        ])

    def __repr__(self) -> str:
        return (f"GridLevel2D(level={self.level}, shape={self.shape}, "
                f"tau={self.tau:.4f}, nodes={self.num_nodes:,})")
