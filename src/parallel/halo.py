"""Esoteric halo exchange v1 — physical-band protocol (axis-generic).

Protocol (patch 17 §1): ghost = 2 cells per side along the split axis.
BEFORE every kernel step (lockstep parity t across ranks):
    1. gather own EDGE bands (2 owned cells/side) to physical standard-ordered
       f via esoteric_gather_std_region  (bit-exact primitive, patch 15 e2);
    2. exchange bands with the side neighbors;
    3. scatter the received bands into the local GHOST bands
       (esoteric_scatter_std_region at the same parity t).
The kernel then runs over the FULL local array (owned + ghosts): ghost layer 1
is computed redundantly from synced inputs (its STORE into the owned edge is
exactly what the neighbor's real cell would produce); ghost layer 2 computes
garbage at its outer face and is overwritten by the next exchange. Owned cells
therefore evolve BIT-identically to the single-rank run (per-cell kernel work
is independent; the gather/scatter permutations are exact).

Transports: MPITransport (CUDA-aware mpi4py, cluster) and LoopbackTransport
(in-process post office — enables N-"rank" verification without mpi4py; same
code path except the wire). Two-phase API (post / complete) so all ranks
gather from the pre-exchange state before any scatter happens.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from src.kernels.esoteric_d3q27 import (
    esoteric_gather_std_region, esoteric_scatter_std_region)


class LoopbackTransport:
    """In-process post office for local N-'rank' verification (no MPI)."""

    def __init__(self) -> None:
        self._box: Dict[Tuple[int, int, int], object] = {}

    def post(self, src: int, dst: int, tag: int, arr) -> None:
        key = (src, dst, tag)
        if key in self._box:
            raise RuntimeError(f"duplicate message {key}")
        self._box[key] = arr

    def collect(self, src: int, dst: int, tag: int):
        return self._box.pop((src, dst, tag))


class MPITransport:
    """CUDA-aware mpi4py transport (cluster; task1-verified UCX device-direct).

    NOTE (task1): `mpi4py.rc.thread_level = 'serialized'` MUST be set before
    importing mpi4py.MPI or Open MPI 5.x drops the UCX PML (no CUDA-aware).
    That guard lives in the runner, not here.
    """

    def __init__(self, comm) -> None:
        self._comm = comm
        self._pending = []

    def post(self, src: int, dst: int, tag: int, arr) -> None:
        self._pending.append(self._comm.Isend(arr, dest=dst, tag=tag))

    def collect(self, src: int, dst: int, tag: int):
        raise NotImplementedError(
            "MPI path uses exchange() with pre-allocated recv buffers (M5)")


class HaloBandExchangerV1:
    """Physical-band esoteric halo (v1). One instance per (level-)field."""

    def __init__(self, partition, transport, xp) -> None:
        self._p = partition
        self._t = transport
        self._xp = xp

    # -- phase A: everyone gathers + posts from the pre-exchange state --
    def post(self, f_mem, t_step: int) -> None:
        p = self._p
        for side in (0, 1):
            nbr = p.neighbor(side)
            if nbr is None:
                continue
            band = esoteric_gather_std_region(
                self._xp, f_mem, t_step, p.edge_band(side))
            # tag encodes the side AS SEEN BY THE RECEIVER (opposite side)
            self._t.post(p.rank, nbr, tag=(1 - side), arr=band)

    # -- phase B: everyone receives + scatters into its ghosts --
    def complete(self, f_mem, t_step: int) -> None:
        p = self._p
        for side in (0, 1):
            nbr = p.neighbor(side)
            if nbr is None:
                continue
            band = self._t.collect(nbr, p.rank, tag=side)
            esoteric_scatter_std_region(
                self._xp, f_mem, band, t_step, p.ghost_band(side))
