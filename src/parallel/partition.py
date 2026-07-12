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
# auto tie-break preference (see patch 17): y, then z, then x.
_AUTO_ORDER = (1, 2, 0)


def choose_axis(level_shapes: Sequence[Tuple[int, int, int]]) -> int:
    """Auto axis rule: maximize the MINIMUM per-level extent along the axis.

    This guarantees the dominant (finest) level splits evenly — e.g. the
    farfield40 L4 rotor slab is only 57 cells thick in x, so x would starve
    all but one rank; y/z (681) split cleanly.
    """
    best_ax, best_val = _AUTO_ORDER[0], -1
    for ax in _AUTO_ORDER:
        val = min(int(s[ax]) for s in level_shapes)
        if val > best_val:
            best_ax, best_val = ax, val
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

    def __repr__(self) -> str:
        return (f"Partition1D(axis={AXIS_NAME[self.axis]}, rank={self.rank}/"
                f"{self.n_ranks}, own=[{self.own_start},"
                f"{self.own_start + self.own_count}), ghost={self.ghost})")
