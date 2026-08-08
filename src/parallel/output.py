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
