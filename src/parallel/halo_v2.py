"""Esoteric halo v2 — raw parity-slot-subset exchange (patch 17 M4).

Ghost = 1. AFTER each local step at parity t, exactly the 9 axis-crossing
populations per boundary cell are exchanged per face, in two groups:

  A  (outbox): our edge cell's STORE deposited the outgoing population into
     our GHOST plane at slot (i+1 if t odd else i) for pairs with
     c_i[axis] = +sign(face). The receiver copies it into his OWNED edge
     plane at the SAME slot — the esoteric parity swap guarantees his next
     LOAD (parity t+1) reads that exact slot as the arrived population.
  B  (residents read remotely): the neighbor's next LOAD reads our OWNED
     edge plane at slot (i if t odd else i+1) for pairs with
     c_i[axis] = -sign(face) (i.e. direction i+1 points at him). We send
     those; he stores them into his GHOST plane at the SAME slot.

Both directions are same-slot copies — no relabelling — because the pull
swap already encodes the streaming. Traffic per face pair per step:
9 slot-planes each way, vs v1's 2-cell x 27-slot physical bands (54) -> 6x
less. Ghost cells are marked SOLID in node_type so the kernel never collides
them (they are pure transit mailboxes).

Scope (M4): single-level streaming. SGS needs a u edge-plane exchange and
MLG coupling needs physical bands at coupling instants (v1 machinery) —
that hybrid is deferred until acoustic-scale measurements justify it
(patch 17 §5: measure, then optimise). Production default remains v1.
"""

from __future__ import annotations

from typing import List, Tuple

from src.kernels.esoteric_d3q27 import CX_ESO, CY_ESO, CZ_ESO


def _slot_tables(axis: int) -> Tuple[List[List[int]], List[List[int]]]:
    """A[t&1], B[t&1] slot lists for the +axis face (mirror for -axis).

    A_plus(t)  = { i+1 if odd else i  : c_i[axis] = +1 }
    B_plus(t)  = { i   if odd else i+1: c_i[axis] = -1 }
    """
    c = (CX_ESO, CY_ESO, CZ_ESO)[axis]
    a_even, a_odd, b_even, b_odd = [], [], [], []
    for p in range(13):
        i = 2 * p + 1
        if c[i] == +1:
            a_even.append(i); a_odd.append(i + 1)
        elif c[i] == -1:
            b_even.append(i + 1); b_odd.append(i)
    return [a_even, a_odd], [b_even, b_odd]


class SlotHaloExchangerV2:
    """Post/complete slot-subset exchange for one esoteric field (ghost=1)."""

    def __init__(self, partition, transport, xp, tag_base: int = 0) -> None:
        if partition.ghost != 1:
            raise ValueError("v2 slot halo requires ghost=1")
        if partition.own_count < 2:
            raise ValueError("v2 slot halo requires own_count >= 2")
        self._p = partition
        self._t = transport
        self._xp = xp
        a = partition.axis
        A, B = _slot_tables(a)
        self._A, self._B = A, B
        g, n = partition.ghost, partition.own_count
        # plane index along the axis (local coords)
        self._ghost_pl = {0: 0, 1: g + n}          # low/high ghost plane
        self._edge_pl = {0: g, 1: g + n - 1}       # low/high owned edge
        self._tb = int(tag_base)
        pl = list(partition.local_shape)
        del pl[a]
        self._plane_shape = tuple(pl)
        self.bytes_sent = 0

    def _plane(self, f_mem, slots, pl):
        """(len(slots), *plane_shape) contiguous pack of one axis-plane."""
        sl = [slice(None)] * 4
        sl[1 + self._p.axis] = pl
        return self._xp.ascontiguousarray(f_mem[tuple(sl)][slots])

    def _put(self, f_mem, slots, pl, vals) -> None:
        sl = [slice(None)] * 4
        sl[1 + self._p.axis] = pl
        view = f_mem[tuple(sl)]
        view[slots] = vals                          # advanced-index write

    def post(self, f_mem, t_step: int) -> None:
        """AFTER the kernel step at parity t_step."""
        par = t_step & 1
        for side in (0, 1):                         # 0=low(-a), 1=high(+a)
            nbr = self._p.neighbor(side)
            if nbr is None:
                continue
            # Low face mirrors the +a tables with flipped parity label:
            #   A_low(t) = {c_i[a]=-1 pairs, slot i+1 odd / i even} = B[1-par]
            #   B_low(t) = {c_i[a]=+1 pairs, slot i odd / i+1 even} = A[1-par]
            A = self._B[1 - par] if side == 0 else self._A[par]
            B = self._A[1 - par] if side == 0 else self._B[par]
            pa = self._plane(f_mem, A, self._ghost_pl[side])
            pb = self._plane(f_mem, B, self._edge_pl[side])
            self.bytes_sent += pa.nbytes + pb.nbytes
            self._t.post(self._p.rank, nbr,
                         tag=self._tb + (((1 - side) << 1) | 0), arr=pa)
            self._t.post(self._p.rank, nbr,
                         tag=self._tb + (((1 - side) << 1) | 1), arr=pb)
        if hasattr(self._t, "commit"):
            self._t.commit()

    def complete(self, f_mem, t_step: int) -> None:
        par = t_step & 1
        for side in (0, 1):
            nbr = self._p.neighbor(side)
            if nbr is None:
                continue
            # Neighbor's A-pack -> OUR owned edge plane; his B-pack -> OUR
            # ghost plane; both at the sender's slot labels (his face =
            # 1-side), which are complementary to the slots our own STORE
            # wrote there — no collision, the esoteric layout separates the
            # two populations of each pair by construction.
            ra = self._recv_A(side, par)
            rb = self._recv_B(side, par)
            import numpy as _np
            pa = self._t.collect(nbr, self._p.rank,
                                 tag=self._tb + ((side << 1) | 0),
                                 shape=(len(ra),) + self._plane_shape,
                                 dtype=_np.float32)
            pb = self._t.collect(nbr, self._p.rank,
                                 tag=self._tb + ((side << 1) | 1),
                                 shape=(len(rb),) + self._plane_shape,
                                 dtype=_np.float32)
            self._put(f_mem, ra, self._edge_pl[side], pa)
            self._put(f_mem, rb, self._ghost_pl[side], pb)
        self._t.flush()

    # -- receive-side slot labels: the sender computed labels for HIS face
    #    (opposite side); by the same formulas those are:
    def _recv_A(self, side: int, par: int):
        return self._B[1 - par] if side == 1 else self._A[par]

    def _recv_B(self, side: int, par: int):
        return self._A[1 - par] if side == 1 else self._B[par]
