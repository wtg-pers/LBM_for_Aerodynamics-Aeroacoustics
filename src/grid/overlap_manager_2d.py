"""
Overlap Manager (2D) — Multi-Domain Grid Overlap Geometry for 2D MLG

2D counterpart of `src/grid/overlap_manager.py`. Parallel class hierarchy
(`IndexBox2D`, `OverlapRegion2D`, `OverlapManager2D`) preserving the 3D
semantics while operating on 2D arrays of shape (Nx, Ny).

Design principles:
    - Zero impact on 3D MLG code: this file is imported only when
      building a 2D multi-level simulation.
    - Array convention for 2D: (Q, Nx, Ny) for populations,
      (Nx, Ny) for scalar fields — matches StreamingPull/HalfwayBounceBack.
    - Slices returned as (x_slice, y_slice).

See `overlap_manager.py` for the 3D original and the multi-domain
conceptual picture (Lagrava Sandoval, Ch.5).

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class IndexBox2D:
    """Axis-aligned 2D box in integer index space.

    Inclusive start/end on each axis: contains nodes at
    [x_start, x_end] × [y_start, y_end].
    """
    x_start: int
    x_end: int
    y_start: int
    y_end: int

    @property
    def shape(self) -> Tuple[int, int]:
        """(Nx, Ny) — number of nodes along each axis."""
        return (
            self.x_end - self.x_start + 1,
            self.y_end - self.y_start + 1,
        )

    @property
    def num_nodes(self) -> int:
        nx, ny = self.shape
        return nx * ny

    def contains(self, ix: int, iy: int) -> bool:
        return (
            self.x_start <= ix <= self.x_end
            and self.y_start <= iy <= self.y_end
        )

    def as_slices(self) -> Tuple[slice, slice]:
        """Return (x_slice, y_slice) for indexing (Nx, Ny) arrays."""
        return (
            slice(self.x_start, self.x_end + 1),
            slice(self.y_start, self.y_end + 1),
        )

    def expanded(self, width: int) -> 'IndexBox2D':
        return IndexBox2D(
            x_start=self.x_start - width,
            x_end=self.x_end + width,
            y_start=self.y_start - width,
            y_end=self.y_end + width,
        )

    def shrunk(self, width: int) -> 'IndexBox2D':
        return self.expanded(-width)

    def clipped(self, bounds: 'IndexBox2D') -> 'IndexBox2D':
        return IndexBox2D(
            x_start=max(self.x_start, bounds.x_start),
            x_end=min(self.x_end, bounds.x_end),
            y_start=max(self.y_start, bounds.y_start),
            y_end=min(self.y_end, bounds.y_end),
        )

    def is_valid(self) -> bool:
        return (self.x_start <= self.x_end
                and self.y_start <= self.y_end)


class OverlapRegion2D:
    """Spatial relationship between one pair of adjacent 2D grid levels.

    Mirrors `OverlapRegion` (3D) but on 2D. See 3D docstring for the
    conceptual picture (fine_region / overlap buffer / excised / fine
    grid dimensions).

    Array convention:
        - Coarse arrays indexed as (Nx_c, Ny_c)
        - Fine   arrays indexed as (Nx_f, Ny_f) where Nx_f = fdx·R + 1
    """

    REFINE_RATIO: int = 2   # fixed; matches 3D

    def __init__(
        self,
        level_coarse: int,
        coarse_shape: Tuple[int, int],
        fine_region: IndexBox2D,
        overlap_width: int = 2,
    ) -> None:
        self.level_coarse: int = level_coarse
        self.level_fine: int = level_coarse + 1
        self.overlap_width: int = overlap_width

        Nx_c, Ny_c = coarse_shape
        self._coarse_bounds = IndexBox2D(0, Nx_c - 1, 0, Ny_c - 1)

        self._validate(fine_region, coarse_shape, overlap_width)

        self.fine_region: IndexBox2D = fine_region

        # Fine domain in coarse coords = fine_region + overlap buffer, clipped.
        self.fine_domain_coarse: IndexBox2D = fine_region.expanded(overlap_width)
        self.fine_domain_coarse = self.fine_domain_coarse.clipped(
            self._coarse_bounds
        )

        # Fine grid dimensions
        fdx = self.fine_domain_coarse.x_end - self.fine_domain_coarse.x_start
        fdy = self.fine_domain_coarse.y_end - self.fine_domain_coarse.y_start
        self.fine_shape: Tuple[int, int] = (
            fdx * self.REFINE_RATIO + 1,
            fdy * self.REFINE_RATIO + 1,
        )

        self.excised: IndexBox2D = fine_region

        self.fine_origin_coarse: Tuple[int, int] = (
            self.fine_domain_coarse.x_start,
            self.fine_domain_coarse.y_start,
        )

    # =================================================================
    # Index Mapping: Coarse ↔ Fine
    # =================================================================

    def coarse_to_fine(self, ic: int, jc: int) -> Tuple[int, int]:
        """Convert coarse index to fine index (even-index, coincident)."""
        ox, oy = self.fine_origin_coarse
        i_f = (ic - ox) * self.REFINE_RATIO
        j_f = (jc - oy) * self.REFINE_RATIO

        Nx_f, Ny_f = self.fine_shape
        if not (0 <= i_f < Nx_f and 0 <= j_f < Ny_f):
            raise ValueError(
                f"Coarse index ({ic},{jc}) maps to fine ({i_f},{j_f}) "
                f"which is outside fine grid {self.fine_shape}."
            )
        return (i_f, j_f)

    def fine_to_coarse(self, i_f: int, j_f: int) -> Tuple[float, float]:
        """Convert fine index to coarse coords (may be fractional)."""
        ox, oy = self.fine_origin_coarse
        ic = i_f / self.REFINE_RATIO + ox
        jc = j_f / self.REFINE_RATIO + oy
        return (ic, jc)

    def is_coarse_coincident(self, i_f: int, j_f: int) -> bool:
        """True if fine node coincides with a coarse node (even indices)."""
        return (i_f % 2 == 0) and (j_f % 2 == 0)

    # =================================================================
    # Coarse Active Region
    # =================================================================

    def is_coarse_active(self, ic: int, jc: int) -> bool:
        """True if coarse node is active (not excised by fine)."""
        return not self.excised.contains(ic, jc)

    def compute_coarse_active_mask(self, xp: 'object') -> 'object':
        """Boolean mask: True = coarse computes, False = excised.

        Returned shape matches (Nx_c, Ny_c) convention for 2D arrays.
        """
        bounds = self._coarse_bounds
        Nx_c = bounds.x_end + 1
        Ny_c = bounds.y_end + 1

        mask = xp.ones((Nx_c, Ny_c), dtype=xp.bool_)
        sx, sy = self.excised.as_slices()
        mask[sx, sy] = False
        return mask

    # =================================================================
    # Fine Boundary Faces (coarse → fine coupling targets)
    # =================================================================

    def get_fine_boundary_slices(self) -> Dict[str, Tuple[slice, slice]]:
        """Slices for each face of the fine grid boundary.

        Each face is a 1D strip in the fine grid, `overlap_width * R`
        nodes thick on the perpendicular axis. Returned slices index
        (Nx_f, Ny_f) arrays.
        """
        Nx_f, Ny_f = self.fine_shape
        ow_f = self.overlap_width * self.REFINE_RATIO

        return {
            'x_min': (slice(0, ow_f),               slice(None)),
            'x_max': (slice(Nx_f - ow_f, Nx_f),     slice(None)),
            'y_min': (slice(None),                  slice(0, ow_f)),
            'y_max': (slice(None),                  slice(Ny_f - ow_f, Ny_f)),
        }

    # =================================================================
    # Coarse Overlap Strips (fine → coarse feedback targets)
    # =================================================================

    def get_coarse_overlap_slices(self) -> Dict[str, Tuple[slice, slice]]:
        """Slices for each overlap strip on the coarse grid.

        These strips receive fine-to-coarse feedback after the fine
        grid advances. Returned slices index (Nx_c, Ny_c) arrays.
        """
        fr = self.fine_region
        ow = self.overlap_width

        return {
            'x_min': (
                slice(fr.x_start - ow, fr.x_start),
                slice(fr.y_start,      fr.y_end + 1),
            ),
            'x_max': (
                slice(fr.x_end + 1, fr.x_end + 1 + ow),
                slice(fr.y_start,   fr.y_end + 1),
            ),
            'y_min': (
                slice(fr.x_start - ow, fr.x_end + 1 + ow),
                slice(fr.y_start - ow, fr.y_start),
            ),
            'y_max': (
                slice(fr.x_start - ow, fr.x_end + 1 + ow),
                slice(fr.y_end + 1,    fr.y_end + 1 + ow),
            ),
        }

    # =================================================================
    # Summary
    # =================================================================

    def summary(self) -> str:
        Nx_f, Ny_f = self.fine_shape
        fine_nodes = Nx_f * Ny_f
        excised_nodes = self.excised.num_nodes
        return "\n".join([
            f"OverlapRegion2D: Level {self.level_coarse} ↔ Level {self.level_fine}",
            f"  Fine region (coarse idx): "
            f"x[{self.fine_region.x_start},{self.fine_region.x_end}] "
            f"y[{self.fine_region.y_start},{self.fine_region.y_end}]",
            f"  Fine domain (with overlap): "
            f"x[{self.fine_domain_coarse.x_start},{self.fine_domain_coarse.x_end}] "
            f"y[{self.fine_domain_coarse.y_start},{self.fine_domain_coarse.y_end}]",
            f"  Fine grid shape: ({Nx_f}, {Ny_f}) = {fine_nodes:,} nodes",
            f"  Overlap width: {self.overlap_width} coarse cells "
            f"({self.overlap_width * self.REFINE_RATIO} fine cells)",
            f"  Excised coarse nodes: {excised_nodes:,}",
        ])

    # =================================================================
    # Validation
    # =================================================================

    def _validate(
        self,
        fine_region: IndexBox2D,
        coarse_shape: Tuple[int, int],
        overlap_width: int,
    ) -> None:
        Nx_c, Ny_c = coarse_shape
        bounds = IndexBox2D(0, Nx_c - 1, 0, Ny_c - 1)

        if overlap_width < 1:
            raise ValueError(
                f"overlap_width must be >= 1, got {overlap_width}."
            )

        if not fine_region.is_valid():
            raise ValueError(f"fine_region invalid: {fine_region}")

        fr = fine_region
        if not (bounds.contains(fr.x_start, fr.y_start)
                and bounds.contains(fr.x_end, fr.y_end)):
            raise ValueError(
                f"fine_region {fine_region} extends outside coarse domain "
                f"(0..{Nx_c-1}, 0..{Ny_c-1})."
            )

        expanded = fine_region.expanded(overlap_width)
        if not (bounds.contains(expanded.x_start, expanded.y_start)
                and bounds.contains(expanded.x_end, expanded.y_end)):
            import warnings
            warnings.warn(
                f"Fine region + overlap extends beyond coarse domain. "
                f"Overlap will be clipped at domain boundaries.",
                stacklevel=3,
            )

        nx, ny = fine_region.shape
        if nx < 1 or ny < 1:
            raise ValueError(
                f"fine_region must span >= 1 cell per axis. Got ({nx}, {ny})."
            )


class OverlapManager2D:
    """Container for all 2D overlap regions in an M-level grid system."""

    def __init__(self) -> None:
        self._regions: List[OverlapRegion2D] = []

    @property
    def num_pairs(self) -> int:
        return len(self._regions)

    def add_level_pair(
        self,
        coarse_shape: Tuple[int, int],
        fine_region: IndexBox2D,
        overlap_width: int = 2,
    ) -> OverlapRegion2D:
        level_coarse = len(self._regions)
        region = OverlapRegion2D(
            level_coarse=level_coarse,
            coarse_shape=coarse_shape,
            fine_region=fine_region,
            overlap_width=overlap_width,
        )
        self._regions.append(region)
        return region

    def get_region(self, level_coarse: int) -> OverlapRegion2D:
        if not 0 <= level_coarse < len(self._regions):
            raise IndexError(
                f"Level pair {level_coarse} not found. "
                f"Available: [0, {len(self._regions) - 1}]"
            )
        return self._regions[level_coarse]

    def summary(self) -> str:
        lines = [
            f"OverlapManager2D: {self.num_pairs} level pair(s)",
            "=" * 60,
        ]
        for region in self._regions:
            lines.append(region.summary())
            lines.append("")
        return "\n".join(lines)
