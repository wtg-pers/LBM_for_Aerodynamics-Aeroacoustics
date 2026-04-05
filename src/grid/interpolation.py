"""
Spatial Interpolation Schemes for Grid Coupling (Vectorized)

When transferring data from a coarse grid to a fine grid, fine nodes
that do not coincide with any coarse node must be filled by interpolation.

Available Schemes:
    CubicInterpolation:
        4-point symmetric cubic interpolation (Lagrava Eq. 4.23).
        Fourth-order accurate. RECOMMENDED — avoids mass loss at interface.
        Stencil: g(x) = 9/16·(g(x+h) + g(x-h)) - 1/16·(g(x+3h) + g(x-3h))

    CompactSecondOrderInterpolation:
        2-point mean interpolation (Lagrava Eq. 4.22).
        Second-order accurate. Causes mass loss. For testing only.
        Stencil: g(x) = (g(x+h) + g(x-h)) / 2

    Boundary nodes use asymmetric 3-point scheme (Lagrava Eq. 4.24):
        g(x) = 3/8·g(x-h) + 3/4·g(x+h) - 1/8·g(x+3h)

Performance:
    All operations are fully vectorized (no Python for-loops).
    All odd indices along the interpolation axis are filled simultaneously.

Reference:
    Lagrava Sandoval, Sec. 4.4.4

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class InterpolationScheme(ABC):
    """Abstract base class for spatial interpolation schemes."""

    @abstractmethod
    def interpolate_1d(
        self,
        xp: 'ModuleType',
        data: 'npt.NDArray',
        axis: int,
    ) -> 'npt.NDArray':
        """Interpolate missing values along one axis.

        Even indices (0, 2, 4, ...) are known (coarse-coincident).
        Odd indices (1, 3, 5, ...) are filled by interpolation.

        Args:
            xp: Array module (numpy or cupy).
            data: Array with known values at even positions along `axis`.
            axis: Axis along which to interpolate.

        Returns:
            Array with all odd indices filled. Same shape as input.
        """
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Formal order of accuracy."""
        ...


def _make_slice(ndim: int, axis: int, idx) -> tuple:
    """Create an indexing tuple selecting `idx` along `axis`."""
    s = [slice(None)] * ndim
    s[axis] = idx
    return tuple(s)


class CubicInterpolation(InterpolationScheme):
    """4-point symmetric cubic interpolation (Lagrava Eq. 4.23).

    Fourth-order accurate, fully vectorized.

    Interior stencil:
        g(x) = 9/16 · [g(x+h) + g(x-h)] - 1/16 · [g(x+3h) + g(x-3h)]

    Boundary stencil (asymmetric 3rd order, Lagrava Eq. 4.24):
        g(x) = 3/8·g(x-h) + 3/4·g(x+h) - 1/8·g(x+3h)
    """

    @property
    def order(self) -> int:
        return 4

    def interpolate_1d(
        self,
        xp: 'ModuleType',
        data: 'npt.NDArray',
        axis: int,
    ) -> 'npt.NDArray':
        """Vectorized cubic interpolation along one axis."""
        result = data.copy()
        n = data.shape[axis]
        nd = data.ndim
        sl = lambda idx: _make_slice(nd, axis, idx)

        if n < 3:
            return result  # nothing to interpolate

        # ── Interior odd indices: 3, 5, ..., n-4 ────────────────
        # Full 4-point symmetric stencil (neighbors at i±1, i±3)
        if n >= 7:
            # Odd indices where i-3 >= 0 and i+3 < n
            odd_int   = sl(slice(3, n - 3, 2))
            left_near = sl(slice(2, n - 4, 2))    # i-1
            right_near = sl(slice(4, n - 2, 2))   # i+1
            left_far  = sl(slice(0, n - 6, 2))    # i-3
            right_far = sl(slice(6, n,     2))    # i+3

            result[odd_int] = (
                9.0 / 16.0 * (data[left_near] + data[right_near])
                - 1.0 / 16.0 * (data[left_far] + data[right_far])
            )

        # ── Left boundary: index 1 ──────────────────────────────
        if n >= 5:
            # Asymmetric 3-point (Eq. 4.24, left boundary)
            result[sl(1)] = (
                3.0 / 8.0 * data[sl(0)]
                + 3.0 / 4.0 * data[sl(2)]
                - 1.0 / 8.0 * data[sl(4)]
            )
        elif n >= 3:
            # Fallback: 2-point linear
            result[sl(1)] = 0.5 * (data[sl(0)] + data[sl(2)])

        # ── Right boundary: index n-2 (if it's odd) ─────────────
        if n >= 5 and (n - 2) % 2 == 1:
            # Asymmetric 3-point (Eq. 4.24, right boundary)
            result[sl(n - 2)] = (
                3.0 / 8.0 * data[sl(n - 1)]
                + 3.0 / 4.0 * data[sl(n - 3)]
                - 1.0 / 8.0 * data[sl(n - 5)]
            )
        elif n >= 3 and (n - 2) % 2 == 1:
            result[sl(n - 2)] = 0.5 * (data[sl(n - 3)] + data[sl(n - 1)])

        return result


class CompactSecondOrderInterpolation(InterpolationScheme):
    """2-point mean interpolation (Lagrava Eq. 4.22).

    Second-order accurate, fully vectorized.
    WARNING: Causes mass loss at grid interfaces.

    Stencil: g(x) = (g(x+h) + g(x-h)) / 2
    """

    @property
    def order(self) -> int:
        return 2

    def interpolate_1d(
        self,
        xp: 'ModuleType',
        data: 'npt.NDArray',
        axis: int,
    ) -> 'npt.NDArray':
        """Vectorized 2-point mean interpolation."""
        result = data.copy()
        n = data.shape[axis]
        nd = data.ndim
        sl = lambda idx: _make_slice(nd, axis, idx)

        if n < 3:
            return result

        # All odd indices: 1, 3, 5, ..., n-2
        odd      = sl(slice(1, n - 1, 2))
        left     = sl(slice(0, n - 2, 2))    # i-1
        right    = sl(slice(2, n,     2))    # i+1

        result[odd] = 0.5 * (data[left] + data[right])

        return result