"""
Spatial Interpolation Schemes for Grid Coupling

When transferring data from a coarse grid to a fine grid, fine nodes
that do not coincide with any coarse node must be filled by interpolation.
This module provides the interpolation stencils.

Available Schemes:
    CubicInterpolation:
        4-point symmetric cubic interpolation (Lagrava Eq. 4.23).
        Fourth-order accurate. RECOMMENDED — avoids mass loss at interface.
        Stencil: g(x) = 9/16·(g(x+h) + g(x-h)) - 1/16·(g(x+3h) + g(x-3h))

    CompactSecondOrderInterpolation:
        2-point mean interpolation (Lagrava Eq. 4.22).
        Second-order accurate. Causes mass loss (pressure jump) at interface.
        Stencil: g(x) = (g(x+h) + g(x-h)) / 2

    For boundary nodes with insufficient neighbors, an asymmetric
    3-point scheme is used (Lagrava Eq. 4.24):
        g(x) = 3/8·g(x-h) + 3/4·g(x+h) - 1/8·g(x+3h)

Physical Context:
    In the multi-domain grid refinement, the fine grid has nodes at
    positions x_f = 0, h, 2h, 3h, ... where h = δx_f. The coarse grid
    has nodes at positions 0, 2h, 4h, ... (every other fine node).
    Fine nodes at odd positions (h, 3h, 5h, ...) need interpolation.

Reference:
    Lagrava Sandoval, Sec. 4.4.4 (2D/3D interpolation schemas)

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
    """Abstract base class for spatial interpolation schemes.

    All schemes interpolate values at fine-grid nodes that lie between
    coarse-grid nodes along one spatial direction (1D interpolation).
    Multi-dimensional interpolation is built from sequential 1D passes.
    """

    @abstractmethod
    def interpolate_1d(
        self,
        xp: 'ModuleType',
        data: 'npt.NDArray',
        axis: int,
    ) -> 'npt.NDArray':
        """Interpolate missing values along one axis.

        Given an array where values at even indices (0, 2, 4, ...)
        are known (from the coarse grid) and odd indices (1, 3, 5, ...)
        need filling, compute the interpolated values.

        Args:
            xp: Array module (numpy or cupy).
            data: Array with known values at even-indexed positions
                  along the specified axis. Shape: arbitrary.
            axis: Axis along which to interpolate.

        Returns:
            Array with interpolated values filled at odd indices.
            Same shape as input.
        """
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Formal order of accuracy of this interpolation scheme."""
        ...


class CubicInterpolation(InterpolationScheme):
    """4-point symmetric cubic interpolation (Lagrava Eq. 4.23).

    Fourth-order accurate. Uses 4 neighboring coarse-grid values to
    interpolate each fine-grid value. This is the RECOMMENDED scheme
    for grid coupling in LBM as it preserves mass conservation across
    the grid interface.

    Stencil (interior nodes):
        g(x) = 9/16 · [g(x+h) + g(x-h)] - 1/16 · [g(x+3h) + g(x-3h)]

    Stencil (boundary nodes, asymmetric 3rd order, Lagrava Eq. 4.24):
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
        """Interpolate using 4-point symmetric cubic stencil.

        Args:
            xp: Array module (numpy or cupy).
            data: Array with known values at even indices along `axis`.
            axis: Interpolation axis.

        Returns:
            Array with all values filled (even = original, odd = interpolated).
        """
        result = data.copy()
        n = data.shape[axis]

        # Helper: slice along axis
        def _sl(idx):
            """Create a slice object selecting index `idx` along `axis`."""
            s = [slice(None)] * data.ndim
            s[axis] = idx
            return tuple(s)

        # Interior odd indices: i = 1, 3, 5, ..., n-2
        # Coarse neighbors at i-1, i+1 (distance h), i-3, i+3 (distance 3h)
        for i in range(1, n - 1, 2):
            # Check if we have enough neighbors for 4-point stencil
            has_left_far = (i - 2) >= 0    # i-3 doesn't exist in fine coords,
            has_right_far = (i + 2) < n    # but i-2 and i+2 are coarse nodes

            # Actually, in our convention:
            # Even indices = coarse nodes: 0, 2, 4, ...
            # Odd indices = fine-only nodes: 1, 3, 5, ...
            # Neighbors of odd index i:
            #   nearest coarse: i-1, i+1
            #   far coarse:     i-3, i+3
            has_left_far = (i - 3) >= 0
            has_right_far = (i + 3) < n

            if has_left_far and has_right_far:
                # Full 4-point symmetric cubic (Eq. 4.23)
                result[_sl(i)] = (
                    9.0 / 16.0 * (data[_sl(i - 1)] + data[_sl(i + 1)])
                    - 1.0 / 16.0 * (data[_sl(i - 3)] + data[_sl(i + 3)])
                )
            elif not has_left_far and has_right_far:
                # Left boundary: asymmetric 3-point (Eq. 4.24, mirrored)
                result[_sl(i)] = (
                    3.0 / 8.0 * data[_sl(i - 1)]
                    + 3.0 / 4.0 * data[_sl(i + 1)]
                    - 1.0 / 8.0 * data[_sl(i + 3)]
                )
            elif has_left_far and not has_right_far:
                # Right boundary: asymmetric 3-point
                result[_sl(i)] = (
                    3.0 / 8.0 * data[_sl(i + 1)]
                    + 3.0 / 4.0 * data[_sl(i - 1)]
                    - 1.0 / 8.0 * data[_sl(i - 3)]
                )
            else:
                # Fallback: 2-point linear (only 2 neighbors)
                result[_sl(i)] = 0.5 * (data[_sl(i - 1)] + data[_sl(i + 1)])

        return result


class CompactSecondOrderInterpolation(InterpolationScheme):
    """2-point mean interpolation (Lagrava Eq. 4.22).

    Second-order accurate. Simple average of two nearest coarse neighbors.
    WARNING: Causes mass loss (pressure discontinuity) at grid interfaces.
    Provided for comparison testing only.

    Stencil:
        g(x) = (g(x+h) + g(x-h)) / 2
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
        """Interpolate using 2-point mean stencil.

        Args:
            xp: Array module (numpy or cupy).
            data: Array with known values at even indices along `axis`.
            axis: Interpolation axis.

        Returns:
            Array with all values filled.
        """
        result = data.copy()
        n = data.shape[axis]

        def _sl(idx):
            s = [slice(None)] * data.ndim
            s[axis] = idx
            return tuple(s)

        for i in range(1, n - 1, 2):
            result[_sl(i)] = 0.5 * (data[_sl(i - 1)] + data[_sl(i + 1)])

        return result