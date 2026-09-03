"""Rank-0 assembled field output for the MPI runner (patch 17 M5b).

VTK and checkpoints reuse the PRODUCTION writers byte-for-byte: rank 0 keeps
the setup's MLGVTKWriter and CheckpointManager (built during its own full
build), every rank sends its owned slabs (host-staged numpy), and rank 0
assembles the global per-BLOCK arrays and calls the same write()/save() the
single-GPU loop calls. Checkpoints are therefore restartable by BOTH the
single-GPU path and the MPI runner (each rank restores the global state
through the normal initializer, then slab-extracts as always).

All entry points are COLLECTIVE — every rank must call them at the same
coarse step (non-root ranks only send).

Block trees (patch 18): the unit of assembly is a BLOCK, not a level, and a
rank may own zero cells of one. Such a rank still joins every gather; it
contributes a zero-length piece, which concatenates away. Skipping the call
instead would make the collective rank-dependent.

(The old Rank0OutputBridge lived here as a second, unreferenced assembly path
duplicating MPIOutputManager. It was removed with the block refactor rather
than block-ified blind: an untested duplicate that silently disagrees with the
live path is the failure class this track exists to remove.)
"""

from __future__ import annotations

from typing import List

import numpy as np
import cupy as cp


def _gather_block(comm, rank: int, nr: int, part, arr_local, tag0: int,
                  n_chan: int, pre_sliced: bool = False,
                  dtype=np.float32):
    """Assemble owned slabs of a (C, x, y, z) LOCAL array on rank 0.

    part:       this rank's Partition1D for the block being gathered.
    arr_local:  None when this rank owns nothing of the block (a zero-length
                piece is sent so the collective stays rank-independent).
    pre_sliced: the caller already passes OWNED-only data (e.g.
                runner.owned_f_std_block) — slicing twice drops 2*ghost rows
                per rank (caught by the bench5 npz shape check).
    """
    if arr_local is None:
        shp = list(part.global_shape)
        shp[part.axis] = 0
        own = np.empty((n_chan,) + tuple(shp), dtype)
    else:
        own = (arr_local if pre_sliced
               else arr_local[(slice(None),) + part.owned_local()])
        own = cp.asnumpy(own) if hasattr(own, "get") else np.asarray(own)
    if rank != 0:
        comm.send((own.shape, own.dtype.str), dest=0, tag=tag0)
        comm.Send(np.ascontiguousarray(own), dest=0, tag=tag0 + 1)
        return None
    pieces = [own]
    for r in range(1, nr):
        shape, dt = comm.recv(source=r, tag=tag0)
        buf = np.empty(shape, dtype=np.dtype(dt))
        comm.Recv(buf, source=r, tag=tag0 + 1)      # count 0 is a legal Recv
        pieces.append(buf)
    return np.concatenate(pieces, axis=1 + part.axis)


# ── checkpoint f STREAMS (robin/16 s12) ──────────────────────────────
# The old checkpoint gather (_gather_block on runner.owned_f_std_block)
# materialised a full std copy of every rank's slab on the device, moved
# it to the host and assembled the WHOLE level on rank 0 -- rank-0 host
# peak = every level's f (29 GB float32 on the 6-level ROBIN grid) -> the
# first checkpoint OOM'd the host. The stream below hands rank 0 one
# (xs, Ny, Nz) piece of one population at a time, in the .npy C order
# (q-major, then x), so CheckpointManager writes each piece as it arrives.
# Every rank walks the SAME plan; a rank that owns nothing of a block
# sends zero-length pieces so the message pattern stays rank-invariant.

def ckpt_stream_plan(global_shape, n_pop: int, max_cells: int = 3_000_000):
    """[(q, x0, x1)] in .npy C order for a (Q, Nx, Ny, Nz) level."""
    Nx, Ny, Nz = (int(v) for v in global_shape)
    step = max(1, max_cells // max(Ny * Nz, 1))
    return [(q, x0, min(x0 + step, Nx))
            for q in range(int(n_pop)) for x0 in range(0, Nx, step)]


def _ckpt_piece(lev, part, q: int, x0: int, x1: int):
    """This rank's OWNED part of global x-slab [x0, x1) of population q,
    as a host array + its placement (gx0, s0, s1) in the (x1-x0, Ny, Nz)
    chunk: rows gx0.. along x when the cut axis is x, columns s0..s1 along
    the cut axis otherwise. Zero-size when the rank owns nothing there."""
    Nx, Ny, Nz = (int(v) for v in part.global_shape)
    a, s, n, g = part.axis, part.own_start, part.own_count, part.ghost
    empty = (np.empty((0, Ny, Nz), np.float32), (0, 0, 0))
    if lev is None or n <= 0:
        return empty
    if a == 0:
        gx0, gx1 = max(x0, s), min(x1, s + n)
        if gx1 <= gx0:
            return empty
        region = (slice(gx0 - s + g, gx1 - s + g), slice(None), slice(None))
        place = (gx0 - x0, 0, 0)
    elif a == 1:
        region = (slice(x0, x1), slice(g, g + n), slice(None))
        place = (0, s, s + n)
    else:
        region = (slice(x0, x1), slice(None), slice(g, g + n))
        place = (0, s, s + n)
    from src.kernels.esoteric_d3q27 import esoteric_gather_std_region
    blk = esoteric_gather_std_region(cp, lev.mem, lev.t, region)
    piece = cp.asnumpy(blk[q])
    del blk
    return np.ascontiguousarray(piece), place


def ckpt_send_block(comm, rank: int, part, lev, uid: int, n_pop: int,
                    tag0: int) -> None:
    """Non-root side of one block's stream: produce and send every piece
    of the plan, in order (blocks until rank 0 consumes them)."""
    for q, x0, x1 in ckpt_stream_plan(part.global_shape, n_pop):
        piece, place = _ckpt_piece(lev, part, q, x0, x1)
        comm.send((piece.shape, piece.dtype.str, place), dest=0, tag=tag0)
        comm.Send(piece, dest=0, tag=tag0 + 1)


def ckpt_lazy_block(comm, rank: int, nr: int, part, lev, uid: int,
                    n_pop: int, tag0: int):
    """Rank-0 side: a LazyArray whose chunks are assembled from this rank's
    own piece + one piece per other rank, per plan entry, as the checkpoint
    writer asks for them."""
    from src.io.checkpoint import LazyArray
    Nx, Ny, Nz = (int(v) for v in part.global_shape)
    a = part.axis

    def _place(chunk, piece, place):
        if piece.size == 0:
            return
        gx0, s0, s1 = place
        if a == 0:
            chunk[gx0:gx0 + piece.shape[0]] = piece
        elif a == 1:
            chunk[:, s0:s1] = piece
        else:
            chunk[:, :, s0:s1] = piece

    def _chunks():
        for q, x0, x1 in ckpt_stream_plan(part.global_shape, n_pop):
            chunk = np.empty((x1 - x0, Ny, Nz), np.float32)
            piece, place = _ckpt_piece(lev, part, q, x0, x1)
            _place(chunk, piece, place)
            for src in range(1, nr):
                shape, dt, place = comm.recv(source=src, tag=tag0)
                buf = np.empty(shape, dtype=np.dtype(dt))
                comm.Recv(buf, source=src, tag=tag0 + 1)
                _place(chunk, buf, place)
            yield chunk
    return LazyArray((int(n_pop), Nx, Ny, Nz), np.float32, _chunks)


class _MaskCarrier:
    """Minimal duck for MLGVTKWriter's `obstacle_bc.solid_mask` read."""

    def __init__(self, solid_mask):
        self.solid_mask = solid_mask


class _LevelView:
    """Duck-typed grid for MLGVTKWriter (needs .rho/.u/.nu_t/.obstacle_bc)."""

    def __init__(self, rho, u, nu_t=None, solid_mask=None):
        self.rho = rho
        self.u = u
        self.nu_t = nu_t
        # solid_mask is static -> captured once at bridge construction on
        # rank 0 (pre-fix this was hardwired None, silently dropping the
        # solid_mask array from every MPI VTK snapshot).
        self.obstacle_bc = (None if solid_mask is None
                            else _MaskCarrier(solid_mask))


class _BlockView:
    """Duck-typed GridBlock: what MLGVTKWriter reads off the tree."""

    __slots__ = ("level", "index", "uid", "sim")

    def __init__(self, level: int, index: int, uid: int, sim):
        self.level = level
        self.index = index
        self.uid = uid
        self.sim = sim


class _MLGView:
    """Duck-typed MultiLevelGrid over the assembled per-block fields.

    Exposes iter_blocks() so the writer takes its block-tree path: with the
    level-only duck it zipped block metadata against `get_level(k)` and wrote
    the SAME grid into every block of a multi-block level.
    """

    def __init__(self, blocks: List[_BlockView]):
        self._blocks = list(blocks)

    def iter_blocks(self):
        return iter(self._blocks)

    @property
    def num_levels(self) -> int:
        return max((b.level for b in self._blocks), default=0) + 1

    def get_level(self, k: int):
        hits = [b for b in self._blocks if b.level == k]
        if len(hits) != 1:
            raise ValueError(
                f"get_level({k}) is ambiguous: level {k} holds {len(hits)} "
                f"blocks — iterate iter_blocks() instead")
        return hits[0].sim
