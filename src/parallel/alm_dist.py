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

F-1 RESOLUTION (backlog #3): the partial sums now go through the
alm_sample_markers RawKernel — per-marker fixed-order accumulation with a
fixed tree reduction, so the LIBRARY-strategy shape-dependence (the §6 bug
class review F-1 flagged) is gone. What remains is the inherent protocol
reassociation (owned-subset sums + allreduce vs one full-box sum), which is
the declared fp-lastbit tier and why the ALM gate stays tolerance-based.
The residual only returns if the non-kernel fallback path is forced
(LBM_ALM_SAMPLE_KERNEL=0).
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

import numpy as np

from src.actuator.interpolation import interpolate_velocity_batch_gpu


def alm_partial_sums(u_owned_view, positions_owned, epsilon, n_cut, xp,
                     kernel_spec=None):
    """(num (N,3), den (N,)) over the owned slab. positions in VIEW coords."""
    return interpolate_velocity_batch_gpu(
        u_owned_view, positions_owned, epsilon, xp=xp, n_cut=n_cut,
        return_sums=True, kernel_spec=kernel_spec)


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

    def sampler(u_field, positions_grid, epsilon, active, n_cut,
                kernel_spec=None, aniso=None):
        # OWNED ownership via clip bounds on the FULL local array (backlog
        # #3: the RawKernel path takes bounds directly — no view, no
        # position shift; cell set identical to the old owned-view clip)
        cb = [(0, d) for d in u_field.shape[1:]]
        cb[ax] = (ghost, ghost + part.own_count)
        if aniso is not None:
            # Anisotropic 3-axis partial sums over owned cells (same clip
            # bounds); the (N,4) sums allreduce identically to gaussian.
            from src.actuator.interpolation import _sample_aniso
            num, den = _sample_aniso(
                u_field, positions_grid, aniso['ec'], aniso['et'], aniso['er'],
                aniso['eps_c'], aniso['eps_t'], aniso['eps_r'], xp, n_cut,
                return_sums=True, clip_bounds=tuple(cb))
            if xp.__name__ != 'numpy':
                num = xp.asnumpy(num); den = xp.asnumpy(den)
        else:
            num, den = interpolate_velocity_batch_gpu(
                u_field, positions_grid, epsilon, xp=xp, n_cut=n_cut,
                return_sums=True, clip_bounds=tuple(cb), kernel_spec=kernel_spec)
        num_t, den_t = allred.allreduce(rank, num, den)
        return num_t / np.maximum(den_t, 1e-30)[:, None]

    return sampler
