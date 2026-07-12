"""Esoteric halo exchange v1 — physical-band protocol (axis-generic).

Protocol (patch 17 §1): ghost = 2 cells per side along the split axis
(3 when SGS gradient stencils read u beyond the collided cell).
BEFORE every kernel step (lockstep parity t across ranks):
    1. gather own EDGE bands (ghost owned cells/side) to physical standard-
       ordered f via esoteric_gather_std_region (bit-exact primitive);
    2. exchange bands with the side neighbors;
    3. scatter the received bands into the local GHOST bands
       (esoteric_scatter_std_region at the same parity t).
The kernel then runs over the FULL local array (owned + ghosts): ghost layer 1
is computed redundantly from synced inputs (its STORE into the owned edge is
exactly what the neighbor's real cell would produce); outer ghost layers
compute garbage at their outer face and are overwritten by the next exchange.
Owned cells therefore evolve BIT-identically to the single-rank run (per-cell
kernel work is independent; the gather/scatter permutations are exact).

Transports: MPITransport (mpi4py; CUDA-aware device-direct or host-staged)
and LoopbackTransport (in-process post office — enables N-"rank"
verification without mpi4py; same code path except the wire). Two-phase API
(post / complete) so all ranks gather from the pre-exchange state before any
scatter happens. `tag_base` namespaces concurrent exchangers (one per MLG
level) on a shared communicator.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

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

    def collect(self, src: int, dst: int, tag: int, shape=None, dtype=None):
        return self._box.pop((src, dst, tag))

    def commit(self) -> None:
        pass

    def flush(self) -> None:
        pass


class MPITransport:
    """mpi4py transport — CUDA-aware device-direct or host-staged.

    cuda_aware=True passes CuPy buffers straight to Isend/Recv (UCX
    device-direct; task1-verified OpenMPI 5.0.5 on the cluster). With
    cuda_aware=False buffers are staged through host numpy (works on any
    MPI, e.g. the local MPICH functional smoke).

    NOTE (task1): `mpi4py.rc.thread_level = 'serialized'` MUST be set before
    importing mpi4py.MPI or Open MPI 5.x drops the UCX PML (no CUDA-aware).
    That guard lives in the runner entry, not here.

    Receive sizes are deterministic (both sides know the band/pack shapes),
    so `collect` takes shape+dtype and allocates the buffer — no size
    handshake on the wire. Call `flush()` once per exchange round to retire
    the Isend requests (send buffers are held until then).
    """

    def __init__(self, comm, cuda_aware: bool = True) -> None:
        self._comm = comm
        self._cuda_aware = bool(cuda_aware)
        self._pending = []          # (request, buffer kept alive)
        self._staged = []           # (dst, tag) staged before commit()
        self._sbuf = {}             # persistent send buffers per (dst, tag)
        self._rbuf = {}             # persistent recv buffers per (src, tag)

    def post(self, src: int, dst: int, tag: int, arr) -> None:
        """Stage a message. Persistent per-(dst,tag) buffers keep addresses
        stable so the UCX registration cache hits; the device copy is queued
        async and ONE stream sync in commit() covers the whole round (the
        first version did a full deviceSynchronize per message — ~hundreds
        of pipeline drains per coarse step, the M5 scaling killer)."""
        import cupy as cp
        key = (dst, tag)
        if self._cuda_aware:
            buf = self._sbuf.get(key)
            if buf is None or buf.shape != arr.shape:
                buf = cp.empty(arr.shape, arr.dtype)
                self._sbuf[key] = buf
            buf[...] = arr
        else:
            buf = np.ascontiguousarray(cp.asnumpy(arr))   # implicit sync
            self._sbuf[key] = buf
        self._staged.append(key)

    def commit(self) -> None:
        """One stream sync, then Isend everything staged this round."""
        if not self._staged:
            return
        if self._cuda_aware:
            import cupy as cp
            cp.cuda.get_current_stream().synchronize()
        for key in self._staged:
            dst, tag = key
            buf = self._sbuf[key]
            self._pending.append(
                (self._comm.Isend(buf, dest=dst, tag=tag), buf))
        self._staged.clear()

    def collect(self, src: int, dst: int, tag: int, shape=None, dtype=None):
        import cupy as cp
        if shape is None:
            raise ValueError("MPITransport.collect needs shape+dtype")
        key = (src, tag)
        if self._cuda_aware:
            buf = self._rbuf.get(key)
            if buf is None or buf.shape != tuple(shape):
                buf = cp.empty(shape, dtype)
                self._rbuf[key] = buf
            self._comm.Recv(buf, source=src, tag=tag)
            return buf
        buf = self._rbuf.get(key)
        if buf is None or buf.shape != tuple(shape):
            buf = np.empty(shape, dtype)
            self._rbuf[key] = buf
        self._comm.Recv(buf, source=src, tag=tag)
        return cp.asarray(buf)

    def flush(self) -> None:
        if self._pending:
            from mpi4py import MPI
            MPI.Request.Waitall([r for r, _ in self._pending])
            self._pending.clear()


class HaloBandExchangerV1:
    """Physical-band esoteric halo (v1). One instance per (level-)field."""

    def __init__(self, partition, transport, xp, tag_base: int = 0) -> None:
        self._p = partition
        self._t = transport
        self._xp = xp
        self._tb = int(tag_base)
        band = list(partition.local_shape)
        band[partition.axis] = partition.ghost
        self._band_shape = (27,) + tuple(band)

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
            self._t.post(p.rank, nbr, tag=self._tb + (1 - side), arr=band)
        self._t.commit()

    # -- phase B: everyone receives + scatters into its ghosts --
    def complete(self, f_mem, t_step: int) -> None:
        p = self._p
        for side in (0, 1):
            nbr = p.neighbor(side)
            if nbr is None:
                continue
            band = self._t.collect(nbr, p.rank, tag=self._tb + side,
                                   shape=self._band_shape, dtype=np.float32)
            esoteric_scatter_std_region(
                self._xp, f_mem, band, t_step, p.ghost_band(side))
        self._t.flush()
