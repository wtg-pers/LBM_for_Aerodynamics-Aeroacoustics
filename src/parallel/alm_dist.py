"""Distributed ALM sampling — Gaussian partial sums + allreduce (patch 17 M3).

Protocol: every rank computes the Gaussian-weighted numerator/denominator of
the integral velocity sampling over its OWNED cells only (the marker stencil
box clips at the owned-slab edge, so ownership is exact and ghost cells are
never double counted). The tiny (N,4) sums are allreduced; u = num/den is
then IDENTICAL on every rank, and the whole BEM/correction pipeline runs
replicated with zero further communication. Markers whose stencil lies
entirely inside one rank reproduce the single-rank sums bit-for-bit (same
box, same reduction order); straddling markers differ only in fp summation
order.

ThreadAllreduce mirrors the MPI.Allreduce control flow in-process (barrier +
shared slots) so the gate exercises the exact production call structure.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

import numpy as np

from src.actuator.interpolation import interpolate_velocity_batch_gpu


def alm_partial_sums(u_owned_view, positions_owned, epsilon, n_cut, xp):
    """(num (N,3), den (N,)) over the owned slab. positions in VIEW coords."""
    return interpolate_velocity_batch_gpu(
        u_owned_view, positions_owned, epsilon, xp=xp, n_cut=n_cut,
        return_sums=True)


class ThreadAllreduce:
    """In-process sum-allreduce across N rank-threads (gate transport)."""

    def __init__(self, n_ranks: int, timeout: float = 120.0) -> None:
        self.n = n_ranks
        self._timeout = timeout
        self._slots: List[Optional[tuple]] = [None] * n_ranks
        self._bar_in = threading.Barrier(n_ranks)
        self._bar_out = threading.Barrier(n_ranks)

    def allreduce(self, rank: int, num: np.ndarray, den: np.ndarray):
        self._slots[rank] = (num, den)
        self._bar_in.wait(self._timeout)
        num_t = sum(s[0] for s in self._slots)
        den_t = sum(s[1] for s in self._slots)
        self._bar_out.wait(self._timeout)          # all read before reuse
        return num_t, den_t


def make_distributed_sampler(allred: ThreadAllreduce, rank: int, part,
                             xp) -> Callable:
    """Build the ActuatorLineModel._velocity_sampler for one rank.

    Called INSIDE step() with (u_field_local, positions_grid_local, eps,
    active, n_cut): positions are already local-array coords (grid offset
    applied by the model); the owned VIEW starts `ghost` cells in along the
    split axis, so shift once more before the partial sums.
    """
    ax = part.axis
    ghost = part.ghost
    owned = part.owned_local()

    def sampler(u_field, positions_grid, epsilon, active, n_cut):
        u_owned = u_field[(slice(None),) + owned]
        pos = positions_grid.copy()
        pos[:, ax] -= ghost
        num, den = alm_partial_sums(u_owned, pos, epsilon, n_cut, xp)
        num_t, den_t = allred.allreduce(rank, num, den)
        return num_t / np.maximum(den_t, 1e-30)[:, None]

    return sampler
