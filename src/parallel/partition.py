"""1D domain partition for multi-GPU decomposition (axis-generic).

Design: patch_notes/hpc_upgrade/17_multigpu_design.md. The decomposition axis
is a PARAMETER (config `parallel.axis` in {auto, x, y, z}) — inflow/rotation
axes differ per case, so nothing here may assume a particular axis.

Reimplementation: the original src/parallel was never committed (blueprint
docs/MULTI_GPU_DESIGN.md only).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = {v: k for k, v in AXIS_INDEX.items()}
# auto tie-break: balance first (strict >), ties fall to the CHEAPEST-halo
# axis. Arrays are (Q, x, y, z) C-order, so a halo band cut along x is a
# fully contiguous plane, along y it is Nz-long contiguous chunks, and along
# z it is 2-element fragments (worst for gather/scatter + MPI packing).
# Hence contiguity-descending scan order x, y, z — the first axis to reach
# the best balance value wins.
_AUTO_ORDER = (0, 1, 2)


def choose_axis(level_shapes: Sequence[Tuple[int, int, int]],
                pair_boxes: Optional[Sequence] = None,
                n_ranks: int = 2) -> int:
    """Auto axis rule.

    With `pair_boxes` (per level pair: the fine_domain_coarse (lo, hi)
    INCLUSIVE bounds per axis, in the coarse level's coords): SIMULATE the
    derived cut chain per axis and pick the axis minimizing the worst-rank
    share of total updates (level cells x 2^k). Raw extents are NOT enough:
    bench5's x extents look splittable (56 >= 48) but the nested boxes
    collapse the x chain — L3/L4 land entirely on one rank (G-M2b finding).

    Without boxes: fall back to maximizing the minimum per-level extent.
    Ties fall to _AUTO_ORDER (halo-contiguity descending x, y, z).
    """
    if pair_boxes is None:
        best_ax, best_val = _AUTO_ORDER[0], -1
        for ax in _AUTO_ORDER:
            val = min(int(s[ax]) for s in level_shapes)
            if val > best_val:
                best_ax, best_val = ax, val
        return best_ax

    from src.parallel.mlg_coupling import fine_range_from_coarse
    nl = len(level_shapes)
    weights = []                       # per-level updates: cells * 2^k
    for k, s in enumerate(level_shapes):
        weights.append(float(s[0] * s[1] * s[2]) * (2.0 ** k))
    total = sum(weights)

    best_ax, best_worst = _AUTO_ORDER[0], float("inf")
    for ax in _AUTO_ORDER:
        loads = [0.0] * n_ranks
        ok = True
        try:
            parts = [Partition1D(level_shapes[0], n_ranks, r, ax)
                     for r in range(n_ranks)]
        except ValueError:
            continue
        for r in range(n_ranks):
            loads[r] += weights[0] * parts[r].own_count / level_shapes[0][ax]
        for k in range(1, nl):
            lo, hi = pair_boxes[k - 1][ax]
            nf = level_shapes[k][ax]
            nxt = []
            for r in range(n_ranks):
                f0, fc = fine_range_from_coarse(parts[r], lo, hi, nf)
                loads[r] += weights[k] * fc / nf
                nxt.append(Partition1D.from_range(
                    level_shapes[k], n_ranks, r, ax, f0, fc))
            parts = nxt
        worst = max(loads) / total
        if ok and worst < best_worst - 1e-12:
            best_ax, best_worst = ax, worst
    return best_ax


class Partition1D:
    """Contiguous 1D split of a (Nx, Ny, Nz) grid along `axis` with ghosts.

    Local array layout along the split axis:
        [0, ghost)                      low-side ghost band
        [ghost, ghost + own)            owned cells (global [start, start+own))
        [ghost + own, ghost*2 + own)    high-side ghost band

    Slices returned by the helpers are 3-tuples of spatial slices (full
    slice(None) on the two non-split axes) — directly consumable by the
    esoteric region gather/scatter primitives.
    """

    def __init__(self, global_shape: Tuple[int, int, int], n_ranks: int,
                 rank: int, axis: int, ghost: int = 2,
                 periodic: bool = True) -> None:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0/1/2, got {axis}")
        if not (0 <= rank < n_ranks):
            raise ValueError(f"rank {rank} outside n_ranks {n_ranks}")
        N = int(global_shape[axis])
        if n_ranks > N:
            raise ValueError(f"n_ranks {n_ranks} > axis extent {N}")
        self.global_shape = tuple(int(v) for v in global_shape)
        self.n_ranks = int(n_ranks)
        self.rank = int(rank)
        self.axis = int(axis)
        self.ghost = int(ghost)
        self.periodic = bool(periodic)

        # near-equal contiguous split (M2 adds the coarse-aligned rule)
        base, rem = divmod(N, n_ranks)
        counts = [base + (1 if r < rem else 0) for r in range(n_ranks)]
        self.own_count: int = counts[rank]
        self.own_start: int = sum(counts[:rank])       # global index
        self._counts: List[int] = counts

        ls = list(self.global_shape)
        ls[axis] = self.own_count + 2 * self.ghost
        self.local_shape: Tuple[int, int, int] = tuple(ls)

    # ── slice helpers (3-tuple spatial slices) ───────────────────────
    def _ax(self, sl: slice) -> Tuple[slice, slice, slice]:
        out = [slice(None)] * 3
        out[self.axis] = sl
        return tuple(out)

    def owned_local(self) -> Tuple[slice, slice, slice]:
        """Owned cells in LOCAL coordinates."""
        return self._ax(slice(self.ghost, self.ghost + self.own_count))

    def owned_global(self) -> Tuple[slice, slice, slice]:
        """Owned cells in GLOBAL coordinates."""
        return self._ax(slice(self.own_start, self.own_start + self.own_count))

    def edge_band(self, side: int) -> Tuple[slice, slice, slice]:
        """First/last `ghost` OWNED cells (local coords) — what we SEND."""
        g, n = self.ghost, self.own_count
        if side == 0:
            return self._ax(slice(g, 2 * g))
        return self._ax(slice(g + n - g, g + n))

    def ghost_band(self, side: int) -> Tuple[slice, slice, slice]:
        """Ghost cells on `side` (local coords) — where we RECEIVE."""
        g, n = self.ghost, self.own_count
        if side == 0:
            return self._ax(slice(0, g))
        return self._ax(slice(g + n, 2 * g + n))

    def neighbor(self, side: int) -> Optional[int]:
        """Rank on `side` (0=low, 1=high); None at a non-periodic boundary."""
        r = self.rank + (1 if side == 1 else -1)
        if self.periodic:
            return r % self.n_ranks
        return r if 0 <= r < self.n_ranks else None

    @classmethod
    def from_range(cls, global_shape: Tuple[int, int, int], n_ranks: int,
                   rank: int, axis: int, own_start: int, own_count: int,
                   ghost: int = 2, periodic: bool = True) -> "Partition1D":
        """Partition with an EXPLICIT owned range (MLG coarse-aligned cuts).

        Fine-level partitions are DERIVED from the coarse cuts (fine index =
        2*(coarse - box_lo)), not split independently — this keeps coupling
        regions rank-local (patch 17 M2)."""
        p = cls(global_shape, n_ranks, rank, axis, ghost=ghost,
                periodic=periodic)
        p.own_start = int(own_start)
        p.own_count = int(own_count)
        ls = list(p.global_shape)
        ls[axis] = p.own_count + 2 * p.ghost
        p.local_shape = tuple(ls)
        return p

    def __repr__(self) -> str:
        return (f"Partition1D(axis={AXIS_NAME[self.axis]}, rank={self.rank}/"
                f"{self.n_ranks}, own=[{self.own_start},"
                f"{self.own_start + self.own_count}), ghost={self.ghost})")
