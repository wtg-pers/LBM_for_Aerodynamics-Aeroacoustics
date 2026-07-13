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

KNOWN RESIDUAL (review F-1): the partial sums themselves go through CuPy
reductions inside interpolate_velocity_batch_gpu — shapes are rank-local, so
a device/library strategy change can move results by fp last-bits (same
class as the coupling §6 finding). This stays within the ALM fp-lastbit
verification tier (the allreduce already reassociates), but it is why the
ALM gate is tolerance-based, not bit.
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


class MPIAllreduce:
    """comm.Allreduce adapter — same interface as ThreadAllreduce."""

    def __init__(self, comm) -> None:
        self._comm = comm

    def allreduce(self, rank: int, num: np.ndarray, den: np.ndarray):
        n = den.shape[0]
        send = np.empty(4 * n, np.float64)
        send[:3 * n] = num.ravel()
        send[3 * n:] = den
        recv = np.empty_like(send)
        self._comm.Allreduce(send, recv)
        return recv[:3 * n].reshape(n, 3), recv[3 * n:]


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

    def sampler(u_field, positions_grid, epsilon, active, n_cut):
        # OWNED ownership via clip bounds on the FULL local array (backlog
        # #3: the RawKernel path takes bounds directly — no view, no
        # position shift; cell set identical to the old owned-view clip)
        cb = [(0, d) for d in u_field.shape[1:]]
        cb[ax] = (ghost, ghost + part.own_count)
        num, den = interpolate_velocity_batch_gpu(
            u_field, positions_grid, epsilon, xp=xp, n_cut=n_cut,
            return_sums=True, clip_bounds=tuple(cb))
        num_t, den_t = allred.allreduce(rank, num, den)
        return num_t / np.maximum(den_t, 1e-30)[:, None]

    return sampler
