"""
Overlap Manager — Multi-Domain Grid Overlap Geometry

Manages the spatial relationship between adjacent grid levels in a
multi-domain grid refinement system.

Multi-Domain Structure (Lagrava Sandoval, Ch.5):
    Level 0 (coarse) covers the full domain EXCEPT where Level 1 replaces it.
    Level 1 (fine) covers a sub-region at 2× resolution, overlapping with
    Level 0 by a thin buffer strip on every face.

    1D example (overlap_width=2):

        Level 0:  ██ ██ ▓▓ ▓▓ ░░ ░░ ░░ ▓▓ ▓▓ ██ ██
        Level 1:        ●● ●● ●● ●● ●● ●● ●● ●● ●●
                        ^overlap^  excised   ^overlap^

        ██ = coarse only     (Level 0 active, no fine)
        ▓▓ = overlap buffer  (both grids exist, coupling here)
        ░░ = excised         (Level 0 inactive, fine only)
        ●● = fine grid       (Level 1 active)

    The overlap buffer ensures that streaming at the domain boundary
    does not produce unknown populations (Lagrava Sec. 4.3.4, Fig. 4.6).

Definitions:
    fine_region:  User-specified sub-region in coarse index space
                  where fine resolution is desired.
    overlap:      Buffer strip of width `overlap_width` coarse cells
                  added around the fine_region on every face.
    fine_domain:  fine_region expanded by overlap = actual fine grid extent.
    excised:      Interior of fine_region (coarse nodes replaced by fine).
    active_coarse: Full domain minus excised = where coarse actually computes.

Reference:
    Lagrava Sandoval, Sec. 4.3.4 (overlap zone), Sec. 5.1.1 (refine op)
    Geier et al. (2015), Fig. 10 (5-level sphere)

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class IndexBox:
    """Axis-aligned box in integer index space.

    Represents a rectangular region of grid nodes using inclusive
    start/end indices along each axis.

    Convention:
        All indices are INCLUSIVE: the box contains nodes at
        indices [x_start, x_end] × [y_start, y_end] × [z_start, z_end].

    Attributes:
        x_start: First x-index (inclusive).
        x_end:   Last x-index (inclusive).
        y_start: First y-index (inclusive).
        y_end:   Last y-index (inclusive).
        z_start: First z-index (inclusive).
        z_end:   Last z-index (inclusive).
    """
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    z_start: int
    z_end: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Number of nodes along each axis (Nx, Ny, Nz)."""
        return (
            self.x_end - self.x_start + 1,
            self.y_end - self.y_start + 1,
            self.z_end - self.z_start + 1,
        )

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in this box."""
        nx, ny, nz = self.shape
        return nx * ny * nz

    def contains(self, ix: int, iy: int, iz: int) -> bool:
        """Check if an index triple lies within this box."""
        return (
            self.x_start <= ix <= self.x_end
            and self.y_start <= iy <= self.y_end
            and self.z_start <= iz <= self.z_end
        )

    def as_slices(self) -> Tuple[slice, slice, slice]:
        """Convert to NumPy/CuPy slices for array indexing.

        Returns:
            (z_slice, y_slice, x_slice) matching array layout (Nz, Ny, Nx).
        """
        return (
            slice(self.z_start, self.z_end + 1),
            slice(self.y_start, self.y_end + 1),
            slice(self.x_start, self.x_end + 1),
        )

    def expanded(self, width: int) -> 'IndexBox':
        """Return a new box expanded by `width` cells on every face."""
        return IndexBox(
            x_start=self.x_start - width,
            x_end=self.x_end + width,
            y_start=self.y_start - width,
            y_end=self.y_end + width,
            z_start=self.z_start - width,
            z_end=self.z_end + width,
        )

    def intersects(self, other: 'IndexBox') -> bool:
        """Do the two inclusive boxes share at least one cell?"""
        return (self.x_start <= other.x_end and other.x_start <= self.x_end
                and self.y_start <= other.y_end and other.y_start <= self.y_end
                and self.z_start <= other.z_end and other.z_start <= self.z_end)

    def shrunk(self, width: int) -> 'IndexBox':
        """Return a new box shrunk by `width` cells on every face."""
        return self.expanded(-width)

    def clipped(self, bounds: 'IndexBox') -> 'IndexBox':
        """Clip this box to lie within `bounds`."""
        return IndexBox(
            x_start=max(self.x_start, bounds.x_start),
            x_end=min(self.x_end, bounds.x_end),
            y_start=max(self.y_start, bounds.y_start),
            y_end=min(self.y_end, bounds.y_end),
            z_start=max(self.z_start, bounds.z_start),
            z_end=min(self.z_end, bounds.z_end),
        )

    def is_valid(self) -> bool:
        """Check that start <= end on all axes."""
        return (
            self.x_start <= self.x_end
            and self.y_start <= self.y_end
            and self.z_start <= self.z_end
        )


class OverlapRegion:
    """Spatial relationship between one pair of adjacent grid levels.

    Given the coarse grid dimensions and the user-specified fine region,
    this class computes all derived geometric quantities: fine grid size,
    overlap buffer positions, excised coarse region, and index mapping
    functions between the two grids.

    Physical picture (1D, overlap_width=2):

        Coarse grid (Nx_c nodes):
          0   1   2   3   4   5   6   7   8   9  10  11  12
          ██  ██  ██  ▓▓  ▓▓  ░░  ░░  ░░  ▓▓  ▓▓  ██  ██  ██

        fine_region = [5, 7]  (user wants fine here)
        overlap extends fine to [3, 9]  (2 coarse cells each side)
        excised = [5, 7]  (coarse stops computing here)

        Fine grid (13 nodes, 2× resolution of coarse [3,9]):
          0  1  2  3  4  5  6  7  8  9  10 11 12
          ↕     ↕     ↕     ↕     ↕     ↕     ↕
          3    4     5     6     7     8     9    ← coarse coords

    Args:
        level_coarse: Index of the coarse level.
        coarse_shape: Coarse grid dimensions (Nx_c, Ny_c, Nz_c).
        fine_region: User-specified region in coarse index space
                     where fine resolution is desired.
        overlap_width: Number of coarse cells of overlap on each face.
                       Must be >= 1 (minimum for streaming safety).
                       Recommended: 2 (allows cubic interpolation stencil).

    Raises:
        ValueError: If fine_region is outside coarse domain or overlap
                    extends beyond coarse boundaries.
    """

    # Refinement ratio (fixed at 2 for now)
    REFINE_RATIO: int = 2

    def __init__(
        self,
        level_coarse: int,
        coarse_shape: Tuple[int, int, int],
        fine_region: IndexBox,
        overlap_width: int = 2,
    ) -> None:
        self.level_coarse: int = level_coarse
        self.level_fine: int = level_coarse + 1
        self.overlap_width: int = overlap_width

        # ── Coarse domain bounds ─────────────────────────────────
        Nx_c, Ny_c, Nz_c = coarse_shape
        self._coarse_bounds = IndexBox(0, Nx_c - 1, 0, Ny_c - 1, 0, Nz_c - 1)

        # ── Validate inputs ──────────────────────────────────────
        self._validate(fine_region, coarse_shape, overlap_width)

        # ── Store fine region (user-specified, in coarse indices) ─
        self.fine_region: IndexBox = fine_region

        # ── Fine domain = fine_region + overlap buffer ───────────
        # This is the actual extent of the fine grid (in coarse coords)
        self.fine_domain_coarse: IndexBox = fine_region.expanded(overlap_width)

        # Clip to coarse bounds (fine cannot extend beyond full domain)
        self.fine_domain_coarse = self.fine_domain_coarse.clipped(
            self._coarse_bounds
        )

        # ── Flush faces: first-class designed state ──────────────
        # A face where fine_region sits ON the domain boundary has no
        # coarse-only zone beyond it: the fine face IS a domain face, the
        # coupling band there is width 0 BY DESIGN (span-through prism /
        # boundary-flush regions), and C2F/F2C skip it. Not an accident
        # to warn about — _validate only warns for PARTIAL clipping.
        fr_, fd_ = fine_region, self.fine_domain_coarse
        self.flush_faces: Dict[str, bool] = {
            'x_min': fr_.x_start == fd_.x_start,
            'x_max': fr_.x_end == fd_.x_end,
            'y_min': fr_.y_start == fd_.y_start,
            'y_max': fr_.y_end == fd_.y_end,
            'z_min': fr_.z_start == fd_.z_start,
            'z_max': fr_.z_end == fd_.z_end,
        }

        # ── Fine grid dimensions ─────────────────────────────────
        # Each coarse interval becomes REFINE_RATIO fine intervals
        fdx = self.fine_domain_coarse.x_end - self.fine_domain_coarse.x_start
        fdy = self.fine_domain_coarse.y_end - self.fine_domain_coarse.y_start
        fdz = self.fine_domain_coarse.z_end - self.fine_domain_coarse.z_start

        self.fine_shape: Tuple[int, int, int] = (
            fdx * self.REFINE_RATIO + 1,
            fdy * self.REFINE_RATIO + 1,
            fdz * self.REFINE_RATIO + 1,
        )

        # ── Excised region (coarse does NOT compute here) ────────
        # In multi-domain, coarse removes the interior of fine_region.
        # The overlap strip is kept in coarse for coupling.
        self.excised: IndexBox = fine_region

        # ── Fine origin in coarse coordinates ────────────────────
        # fine index 0 corresponds to this coarse coordinate
        self.fine_origin_coarse: Tuple[int, int, int] = (
            self.fine_domain_coarse.x_start,
            self.fine_domain_coarse.y_start,
            self.fine_domain_coarse.z_start,
        )

        # Display label only; set by OverlapManager.add_region.
        self.name: str = f"L{self.level_fine}"

    # =================================================================
    # Index Mapping: Coarse ↔ Fine
    # =================================================================

    def coarse_to_fine(
        self, ic: int, jc: int, kc: int,
    ) -> Tuple[int, int, int]:
        """Convert a coarse-grid index to the corresponding fine-grid index.

        Only coarse nodes within the fine domain have a corresponding
        fine node (at even fine indices). Coarse nodes outside raise
        ValueError.

        Mapping:
            i_fine = (ic - fine_origin_x) × REFINE_RATIO

        The result is always an EVEN fine index (coarse-coincident node).
        Odd fine indices have no coarse counterpart and need interpolation.

        Args:
            ic, jc, kc: Coarse grid indices.

        Returns:
            (i_fine, j_fine, k_fine): Corresponding fine grid indices.

        Raises:
            ValueError: If (ic, jc, kc) is outside the fine domain.
        """
        ox, oy, oz = self.fine_origin_coarse
        i_f = (ic - ox) * self.REFINE_RATIO
        j_f = (jc - oy) * self.REFINE_RATIO
        k_f = (kc - oz) * self.REFINE_RATIO

        # Validate fine indices are in range
        Nx_f, Ny_f, Nz_f = self.fine_shape
        if not (0 <= i_f < Nx_f and 0 <= j_f < Ny_f and 0 <= k_f < Nz_f):
            raise ValueError(
                f"Coarse index ({ic},{jc},{kc}) maps to fine ({i_f},{j_f},{k_f}) "
                f"which is outside fine grid {self.fine_shape}."
            )
        return (i_f, j_f, k_f)

    def fine_to_coarse(
        self, i_f: int, j_f: int, k_f: int,
    ) -> Tuple[float, float, float]:
        """Convert a fine-grid index to coarse-grid coordinates.

        Fine nodes at EVEN indices map to exact coarse nodes (integer result).
        Fine nodes at ODD indices lie between coarse nodes (fractional result).

        Mapping:
            ic = i_fine / REFINE_RATIO + fine_origin_x

        Args:
            i_f, j_f, k_f: Fine grid indices.

        Returns:
            (ic, jc, kc): Coarse-grid coordinates (may be fractional).
        """
        ox, oy, oz = self.fine_origin_coarse
        ic = i_f / self.REFINE_RATIO + ox
        jc = j_f / self.REFINE_RATIO + oy
        kc = k_f / self.REFINE_RATIO + oz
        return (ic, jc, kc)

    def is_coarse_coincident(self, i_f: int, j_f: int, k_f: int) -> bool:
        """Check if a fine node coincides with a coarse node.

        Coarse-coincident fine nodes are at EVEN indices on ALL axes.
        These receive direct rescaling (no spatial interpolation needed).
        Odd-index fine nodes require spatial interpolation from neighbors.

        Args:
            i_f, j_f, k_f: Fine grid indices.

        Returns:
            True if this fine node has a corresponding coarse node.
        """
        return (i_f % 2 == 0) and (j_f % 2 == 0) and (k_f % 2 == 0)

    # =================================================================
    # Coarse Active Region
    # =================================================================

    def is_coarse_active(self, ic: int, jc: int, kc: int) -> bool:
        """Check if a coarse node is active (not excised) in multi-domain.

        Active coarse nodes = full domain minus the excised interior.
        The overlap buffer nodes ARE active in the coarse grid.

        Args:
            ic, jc, kc: Coarse grid indices.

        Returns:
            True if this coarse node should participate in collision/streaming.
        """
        return not self.excised.contains(ic, jc, kc)

    def compute_coarse_active_mask(self, xp: 'object') -> 'object':
        """Create a boolean mask for active coarse nodes.

        True = active (coarse computes here).
        False = excised (fine replaces this node).

        Args:
            xp: Array module (numpy or cupy).

        Returns:
            Boolean array of shape (Nz_c, Ny_c, Nx_c). True = active.
        """
        bounds = self._coarse_bounds
        Nx_c = bounds.x_end + 1
        Ny_c = bounds.y_end + 1
        Nz_c = bounds.z_end + 1

        mask = xp.ones((Nz_c, Ny_c, Nx_c), dtype=xp.bool_)
        ez, ey, ex = self.excised.as_slices()
        mask[ez, ey, ex] = False
        return mask

    # =================================================================
    # Fine Boundary Faces (where coupling BC is applied)
    # =================================================================

    def get_fine_boundary_slices(self) -> Dict[str, Tuple[slice, slice, slice]]:
        """Get array slices for each face of the fine grid boundary.

        These are the fine-grid nodes where coarse→fine coupling data
        must be injected. Each face is a 2D slab of the fine grid,
        `overlap_width * REFINE_RATIO` nodes thick.

        Returns:
            Dict mapping face name → (z_slice, y_slice, x_slice) for
            indexing into fine-grid arrays of shape (Nz_f, Ny_f, Nx_f).
            Face names: 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        """
        Nx_f, Ny_f, Nz_f = self.fine_shape
        ow_f = self.overlap_width * self.REFINE_RATIO  # overlap in fine cells

        return {
            'x_min': (slice(None), slice(None), slice(0, ow_f)),
            'x_max': (slice(None), slice(None), slice(Nx_f - ow_f, Nx_f)),
            'y_min': (slice(None), slice(0, ow_f), slice(None)),
            'y_max': (slice(None), slice(Ny_f - ow_f, Ny_f), slice(None)),
            'z_min': (slice(0, ow_f), slice(None), slice(None)),
            'z_max': (slice(Nz_f - ow_f, Nz_f), slice(None), slice(None)),
        }

    # =================================================================
    # Coarse Overlap Faces (where fine→coarse feedback is applied)
    # =================================================================

    def get_coarse_overlap_slices(self) -> Dict[str, Tuple[slice, slice, slice]]:
        """Get array slices for each overlap strip on the coarse grid.

        These are the coarse-grid nodes where fine→coarse coupling data
        is written back after the fine grid advances.

        Returns:
            Dict mapping face name → (z_slice, y_slice, x_slice) for
            indexing into coarse-grid arrays of shape (Nz_c, Ny_c, Nx_c).
        """
        fr = self.fine_region
        ow = self.overlap_width

        return {
            'x_min': (
                slice(fr.z_start, fr.z_end + 1),
                slice(fr.y_start, fr.y_end + 1),
                slice(fr.x_start - ow, fr.x_start),
            ),
            'x_max': (
                slice(fr.z_start, fr.z_end + 1),
                slice(fr.y_start, fr.y_end + 1),
                slice(fr.x_end + 1, fr.x_end + 1 + ow),
            ),
            'y_min': (
                slice(fr.z_start, fr.z_end + 1),
                slice(fr.y_start - ow, fr.y_start),
                slice(fr.x_start - ow, fr.x_end + 1 + ow),
            ),
            'y_max': (
                slice(fr.z_start, fr.z_end + 1),
                slice(fr.y_end + 1, fr.y_end + 1 + ow),
                slice(fr.x_start - ow, fr.x_end + 1 + ow),
            ),
            'z_min': (
                slice(fr.z_start - ow, fr.z_start),
                slice(fr.y_start - ow, fr.y_end + 1 + ow),
                slice(fr.x_start - ow, fr.x_end + 1 + ow),
            ),
            'z_max': (
                slice(fr.z_end + 1, fr.z_end + 1 + ow),
                slice(fr.y_start - ow, fr.y_end + 1 + ow),
                slice(fr.x_start - ow, fr.x_end + 1 + ow),
            ),
        }

    # =================================================================
    # Summary
    # =================================================================

    def summary(self) -> str:
        """Human-readable summary of overlap geometry."""
        Nx_f, Ny_f, Nz_f = self.fine_shape
        fine_nodes = Nx_f * Ny_f * Nz_f
        excised_nodes = self.excised.num_nodes

        lines = [
            f"OverlapRegion: Level {self.level_coarse} ↔ Level {self.level_fine}",
            f"  Fine region (coarse idx): "
            f"x[{self.fine_region.x_start},{self.fine_region.x_end}] "
            f"y[{self.fine_region.y_start},{self.fine_region.y_end}] "
            f"z[{self.fine_region.z_start},{self.fine_region.z_end}]",
            f"  Fine domain (with overlap): "
            f"x[{self.fine_domain_coarse.x_start},{self.fine_domain_coarse.x_end}] "
            f"y[{self.fine_domain_coarse.y_start},{self.fine_domain_coarse.y_end}] "
            f"z[{self.fine_domain_coarse.z_start},{self.fine_domain_coarse.z_end}]",
            f"  Fine grid shape: ({Nx_f}, {Ny_f}, {Nz_f}) = {fine_nodes:,} nodes",
            f"  Overlap width: {self.overlap_width} coarse cells "
            f"({self.overlap_width * self.REFINE_RATIO} fine cells)",
            f"  Excised coarse nodes: {excised_nodes:,}",
        ]
        return "\n".join(lines)

    # =================================================================
    # Validation
    # =================================================================

    def _validate(
        self,
        fine_region: IndexBox,
        coarse_shape: Tuple[int, int, int],
        overlap_width: int,
    ) -> None:
        """Validate inputs for geometric consistency."""
        Nx_c, Ny_c, Nz_c = coarse_shape
        bounds = IndexBox(0, Nx_c - 1, 0, Ny_c - 1, 0, Nz_c - 1)

        if overlap_width < 1:
            raise ValueError(
                f"overlap_width must be >= 1, got {overlap_width}. "
                f"At least 1 coarse cell overlap is required for streaming."
            )

        if not fine_region.is_valid():
            raise ValueError(
                f"fine_region is invalid (start > end on some axis): "
                f"{fine_region}"
            )

        # Fine region must be inside the coarse domain
        fr = fine_region
        if not (bounds.contains(fr.x_start, fr.y_start, fr.z_start)
                and bounds.contains(fr.x_end, fr.y_end, fr.z_end)):
            raise ValueError(
                f"fine_region {fine_region} extends outside coarse domain "
                f"(0..{Nx_c-1}, 0..{Ny_c-1}, 0..{Nz_c-1})."
            )

        # Overlap beyond the coarse boundary is clipped. Two distinct cases:
        #   EXACT FLUSH — the region bound sits ON the domain bound: the
        #     face is a domain face, the band is 0 by design and coupling
        #     skips it (span-through prism configs). Silent.
        #   PARTIAL clip — the region bound is within (0, overlap_width)
        #     of the boundary: a truncated band still couples there with
        #     a shortened stencil. Warn (genuine accuracy concern).
        partial = []
        for ax, lo_r, hi_r, n_ax in (('x', fine_region.x_start,
                                      fine_region.x_end, Nx_c),
                                     ('y', fine_region.y_start,
                                      fine_region.y_end, Ny_c),
                                     ('z', fine_region.z_start,
                                      fine_region.z_end, Nz_c)):
            if 0 < lo_r < overlap_width:
                partial.append(f"{ax}_min")
            if n_ax - 1 - overlap_width < hi_r < n_ax - 1:
                partial.append(f"{ax}_max")
        if partial:
            import warnings
            warnings.warn(
                f"Fine region + overlap is PARTIALLY clipped at the domain "
                f"boundary on face(s) {partial}: the coupling band there is "
                f"narrower than overlap_width, which may reduce "
                f"interpolation accuracy. Either inset the region by "
                f">= {overlap_width} cells or place it flush on the "
                f"boundary (flush faces skip coupling by design).",
                stacklevel=3,
            )

        # Fine region must be large enough (at least 1 cell on each axis)
        nx, ny, nz = fine_region.shape
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(
                f"fine_region must span at least 1 cell on each axis. "
                f"Got shape: ({nx}, {ny}, {nz})"
            )


class OverlapManager:
    """Container for all overlap regions in an M-level grid system.

    Manages M-1 OverlapRegion objects, one per adjacent level pair.
    Provides unified access to overlap geometry for the coupling engine.

    Usage:
        >>> mgr = OverlapManager()
        >>> mgr.add_level_pair(
        ...     coarse_shape=(200, 100, 100),
        ...     fine_region=IndexBox(40, 160, 20, 80, 20, 80),
        ...     overlap_width=2,
        ... )
        >>> mgr.get_region(0).fine_shape
        (244, 124, 124)

    Attributes:
        num_pairs: Number of level pairs registered.
    """

    def __init__(self) -> None:
        self._regions: List[OverlapRegion] = []

    @property
    def num_pairs(self) -> int:
        """Number of registered regions (== level pairs for a chain)."""
        return len(self._regions)

    def add_region(
        self,
        coarse_shape: Tuple[int, int, int],
        fine_region: IndexBox,
        level_coarse: int,
        overlap_width: int = 2,
        name: str = "",
    ) -> OverlapRegion:
        """Register one refinement block's coupling geometry.

        `level_coarse` is EXPLICIT. Deriving it from the list position (as
        add_level_pair does) silently conflates "how many regions exist" with
        "which level this one belongs to", which stops being true the moment a
        level hosts more than one block.
        """
        region = OverlapRegion(
            level_coarse=level_coarse,
            coarse_shape=coarse_shape,
            fine_region=fine_region,
            overlap_width=overlap_width,
        )
        region.name = name or f"L{level_coarse + 1}"
        self._regions.append(region)
        return region

    def add_level_pair(
        self,
        coarse_shape: Tuple[int, int, int],
        fine_region: IndexBox,
        overlap_width: int = 2,
    ) -> OverlapRegion:
        """Register the next adjacent level pair — CHAIN ONLY.

        Level pairs must be added in order: (0,1), then (1,2), etc.
        The coarse_shape for pair (k, k+1) should be the fine_shape
        of pair (k-1, k) — i.e., each fine grid becomes the coarse
        grid for the next finer level.

        Kept for the single-block path and existing callers; it derives the
        level from the number of regions already registered, so it cannot be
        mixed with multi-block levels. Use add_region(level_coarse=...) there.

        Args:
            coarse_shape: Dimensions of the coarse grid (Nx, Ny, Nz).
            fine_region: Sub-region in coarse indices where fine is desired.
            overlap_width: Overlap buffer width in coarse cells.

        Returns:
            The created OverlapRegion.
        """
        levels = [r.level_coarse for r in self._regions]
        if len(set(levels)) != len(levels):
            raise ValueError(
                "add_level_pair() derives the level from the region count, so "
                "it is chain-only; this manager already has more than one "
                "region on some level. Use add_region(level_coarse=...).")
        return self.add_region(coarse_shape=coarse_shape,
                               fine_region=fine_region,
                               level_coarse=len(self._regions),
                               overlap_width=overlap_width)

    def regions_at(self, level_coarse: int) -> List[OverlapRegion]:
        """Every region whose coarse side is `level_coarse` (config order)."""
        return [r for r in self._regions if r.level_coarse == level_coarse]

    def get_region(self, level_coarse: int) -> OverlapRegion:
        """Get THE OverlapRegion for a level pair — single-block only.

        Args:
            level_coarse: Index of the coarse level in the pair.

        Returns:
            OverlapRegion for pair (level_coarse, level_coarse + 1).

        Raises:
            IndexError: If level_coarse is out of range.
            ValueError: If that level hosts several blocks — callers must then
                use regions_at() and handle each, since picking the first would
                silently ignore the rest.
        """
        found = self.regions_at(level_coarse)
        if not found:
            n = max((r.level_coarse for r in self._regions), default=-1)
            raise IndexError(
                f"Level pair {level_coarse} not found. Available: [0, {n}]")
        if len(found) > 1:
            raise ValueError(
                f"get_region({level_coarse}) is ambiguous: that level hosts "
                f"{len(found)} blocks ({', '.join(r.name for r in found)}). "
                f"Use regions_at({level_coarse}).")
        return found[0]

    def summary(self) -> str:
        """Human-readable summary of all overlap regions."""
        lines = [
            f"OverlapManager: {self.num_pairs} level pair(s)",
            "=" * 60,
        ]
        for region in self._regions:
            lines.append(region.summary())
            lines.append("")
        return "\n".join(lines)

# =============================================================================
# Body vs. coupling-band diagnosis (STL track S2; S0 carryover)
# =============================================================================

def body_coupling_band_report(solid_mask_np, region) -> Dict[str, list]:
    """Diagnose an obstacle mask against a fine level's C2F/F2C band.

    The outer band of a fine level — between the fine-domain edge and the
    user's fine_region edge (overlap_width coarse cells = 2*overlap_width
    fine cells per face, 0 on faces clipped at the domain boundary) — hosts
    the C2F/F2C coupling stencils. Solid cells inside it corrupt the
    interpolation (coupling reads/writes through the body): hard error.
    Additionally, body surface closer than 0.5*L_body to the fine_region
    edge couples the interface into the boundary layer and shifts Cd
    non-physically (MLG region padding record): warning.

    Args:
        solid_mask_np: fine-level solid mask, HOST numpy bool array
            (2D or 3D, matching the region's dimensionality).
        region: OverlapRegion / OverlapRegion2D of the pair (k-1, k).

    Returns:
        {'violations': [(face, n_solid_cells), ...],
         'padding_warnings': [(face, distance_cells, min_required), ...]}
        Both empty == clean. Faces are 'x_low', 'x_high', 'y_low', ...
    """
    ratio = int(getattr(region, 'REFINE_RATIO', 2))
    fd, fr = region.fine_domain_coarse, region.fine_region
    axes = ['x', 'y']
    if solid_mask_np.ndim == 3:
        axes.append('z')

    violations: list = []
    pad_warns: list = []
    solid_any = bool(solid_mask_np.any())
    if not solid_any:
        return {'violations': violations, 'padding_warnings': pad_warns}

    # Body bbox (fine-local cells) and size for the 0.5*L_body rule.
    idx = np.nonzero(solid_mask_np)
    bb_lo = [int(a.min()) for a in idx]
    bb_hi = [int(a.max()) for a in idx]
    l_body = max(hi - lo + 1 for lo, hi in zip(bb_lo, bb_hi))

    for ax_i, ax in enumerate(axes):
        w_lo = (getattr(fr, f'{ax}_start') - getattr(fd, f'{ax}_start')) * ratio
        w_hi = (getattr(fd, f'{ax}_end') - getattr(fr, f'{ax}_end')) * ratio
        n_ax = solid_mask_np.shape[ax_i]

        sl_all = [slice(None)] * solid_mask_np.ndim
        if w_lo > 0:
            sl = list(sl_all)
            sl[ax_i] = slice(0, w_lo)
            n_bad = int(solid_mask_np[tuple(sl)].sum())
            if n_bad:
                violations.append((f'{ax}_low', n_bad))
            else:
                dist = bb_lo[ax_i] - w_lo  # cells from fine_region edge
                if dist < 0.5 * l_body:
                    pad_warns.append((f'{ax}_low', dist, 0.5 * l_body))
        if w_hi > 0:
            sl = list(sl_all)
            sl[ax_i] = slice(n_ax - w_hi, n_ax)
            n_bad = int(solid_mask_np[tuple(sl)].sum())
            if n_bad:
                violations.append((f'{ax}_high', n_bad))
            else:
                dist = (n_ax - 1 - w_hi) - bb_hi[ax_i]
                if dist < 0.5 * l_body:
                    pad_warns.append((f'{ax}_high', dist, 0.5 * l_body))

    return {'violations': violations, 'padding_warnings': pad_warns}
