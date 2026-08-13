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


def level_spans_L0(level_shapes, pair_boxes, axis):
    """Per-level [A_k, B_k) span in fractional L0 coordinates along `axis`.

    Composed from the nested fine_domain_coarse boxes: a level-k coordinate
    x maps to level k-1 as box_lo + x/2 (fine_range_from_coarse inverse).
    """
    spans = [(0.0, float(level_shapes[0][axis]))]
    lo_acc, scale = 0.0, 1.0
    for k in range(1, len(level_shapes)):
        lo, _hi = pair_boxes[k - 1][axis]
        lo_acc = lo_acc + lo * scale
        scale *= 0.5
        spans.append((lo_acc, lo_acc + level_shapes[k][axis] * scale))
    return spans


def chain_owns(level_shapes, pair_boxes, axis, bounds, ghost=3):
    """Exact per-rank chain from explicit L0 cut bounds.

    Returns (min_own, worst_share, owns_last): the smallest own_count over
    all (rank, level), the worst-rank share of total updates, and the
    finest-level own counts. Uses the same fine_range_from_coarse math as
    the runner, so feasibility here == feasibility there.
    """
    from src.parallel.mlg_coupling import fine_range_from_coarse
    nl, nr = len(level_shapes), len(bounds) - 1
    weights = [float(s[0] * s[1] * s[2]) * (2.0 ** k)
               for k, s in enumerate(level_shapes)]
    total = sum(weights)
    loads = [0.0] * nr
    min_own = 10 ** 9
    owns_last = []
    for r in range(nr):
        part = Partition1D.from_range(
            level_shapes[0], nr, r, axis, bounds[r],
            bounds[r + 1] - bounds[r], ghost=ghost)
        min_own = min(min_own, part.own_count)
        loads[r] += weights[0] * part.own_count / level_shapes[0][axis]
        for k in range(1, nl):
            lo, hi = pair_boxes[k - 1][axis]
            nf = level_shapes[k][axis]
            f0, fc = fine_range_from_coarse(part, lo, hi, nf)
            min_own = min(min_own, fc)
            loads[r] += weights[k] * fc / nf
            part = Partition1D.from_range(
                level_shapes[k], nr, r, axis, f0, fc, ghost=ghost)
            if k == nl - 1:
                owns_last.append(fc)
    return min_own, max(loads) / total, owns_last


def balance_cuts(level_shapes, pair_boxes, axis, n_ranks, ghost=3):
    """Load-balanced L0 cut bounds [0, c1, ..., c_{R-1}, N0] along `axis`.

    Even L0 splits fail on nested MLG cases: the fine boxes concentrate in
    the center, so outer ranks own ZERO fine cells (farfield40 at 4 ranks —
    all axes infeasible). The fix and the balance are the same move: place
    every cut STRICTLY INSIDE the innermost box's L0 span. Nesting then
    guarantees every rank intersects every level, and since the finest
    levels carry most updates (L3+L4 = 86% at farfield40), the load-quantile
    positions land there anyway.

    Cuts are the quantiles of the per-L0-cell update-density profile,
    clamped into the innermost span with spacing >= ghost (interior L0 owns)
    and verified with the exact chain (raises if infeasible).
    """
    import numpy as _np
    n0 = level_shapes[0][axis]
    if n_ranks == 1:
        return [0, n0]
    spans = level_spans_L0(level_shapes, pair_boxes, axis)
    w = _np.zeros(n0)
    for k, (A, B) in enumerate(spans):
        upd = float(level_shapes[k][0] * level_shapes[k][1]
                    * level_shapes[k][2]) * (2.0 ** k)
        dens = upd / (B - A)
        for iy in range(int(A), min(n0, int(_np.ceil(B)))):
            ov = min(iy + 1.0, B) - max(float(iy), A)
            if ov > 0:
                w[iy] += dens * ov
    cum = _np.concatenate([[0.0], _np.cumsum(w)])
    cuts = [int(_np.searchsorted(cum, cum[-1] * q / n_ranks))
            for q in range(1, n_ranks)]
    lo_lim = int(_np.floor(spans[-1][0])) + 1
    hi_lim = int(_np.ceil(spans[-1][1])) - 1
    gap = max(int(ghost), 2)
    for i in range(len(cuts)):
        cuts[i] = max(cuts[i], lo_lim + i * gap)
        if i > 0:
            cuts[i] = max(cuts[i], cuts[i - 1] + gap)
    for i in range(len(cuts) - 1, -1, -1):
        cuts[i] = min(cuts[i], hi_lim - (len(cuts) - 1 - i) * gap)
        if i < len(cuts) - 1:
            cuts[i] = min(cuts[i], cuts[i + 1] - gap)
    bounds = [0] + cuts + [n0]
    if any(b1 - b0 < ghost for b0, b1 in zip(bounds, bounds[1:])):
        raise ValueError(f"axis {AXIS_NAME[axis]}: cannot place {n_ranks - 1} "
                         f"cuts with spacing >= {ghost} inside the innermost "
                         f"box span [{lo_lim},{hi_lim}]")
    min_own, worst, _ = chain_owns(
        level_shapes, pair_boxes, axis, bounds, ghost)
    if min_own < ghost:
        raise ValueError(f"axis {AXIS_NAME[axis]}: balanced cuts {cuts} give "
                         f"min own {min_own} < ghost {ghost}")
    return bounds


class _OwnedRange:
    """Minimal stand-in for Partition1D inside the ownership walk."""

    __slots__ = ("own_start", "own_count")

    def __init__(self, own_start: int, own_count: int) -> None:
        self.own_start = int(own_start)
        self.own_count = int(own_count)


def block_axis_box(block, axis: int) -> Tuple[int, int]:
    """(lo, hi) INCLUSIVE fine_domain_coarse bounds of `block` on `axis`,
    in its PARENT's index space."""
    fdc = block.region.fine_domain_coarse
    return ((fdc.x_start, fdc.x_end), (fdc.y_start, fdc.y_end),
            (fdc.z_start, fdc.z_end))[axis]


def derive_block_ranges(blocks, axis: int, bounds: Sequence[int]) -> List[List[
        Tuple[int, int]]]:
    """[uid][rank] -> (own_start, own_count) in each BLOCK's own index space.

    The L0 cuts are replicated, so every rank reproduces every other rank's
    ownership of every block with no communication — which is exactly what
    Partition1D.neighbor() needs in order to skip the ranks that hold nothing
    of a refinement block (a block covering part of the domain is owned by a
    contiguous SUBSET of ranks).

    Requires level-major uids, i.e. a parent's uid precedes its children's.
    """
    from src.parallel.mlg_coupling import fine_range_from_coarse
    nr = len(bounds) - 1
    out: List[List[Tuple[int, int]]] = []
    for uid, b in enumerate(blocks):
        if b.parent is None:
            out.append([(int(bounds[r]), int(bounds[r + 1] - bounds[r]))
                        for r in range(nr)])
            continue
        if b.parent.uid >= uid:
            raise ValueError(
                f"block '{b.name}' (uid {uid}) precedes its parent "
                f"(uid {b.parent.uid}); block lists must be level-major")
        lo, hi = block_axis_box(b, axis)
        nf = b.region.fine_shape[axis]
        pr = out[b.parent.uid]
        out.append([fine_range_from_coarse(_OwnedRange(*pr[r]), lo, hi, nf)
                    for r in range(nr)])
    return out


def choose_axis_balanced(level_shapes, pair_boxes, n_ranks, ghost=3,
                         exclude_axes=()):
    """Axis + balanced bounds minimizing the worst-rank update share.

    The runner's production rule (M5). Falls back across axes: infeasible
    axes (cuts can't fit inside the innermost span) are rejected. Ties break
    to _AUTO_ORDER (halo-contiguity descending x, y, z).

    exclude_axes: axes never considered — the runner passes the eso
    implicit-wall axes (decomposing a wall axis puts the halo wrap where
    the de-periodization severed traffic; hard-unsupported)."""
    best = None
    for ax in _AUTO_ORDER:
        if ax in exclude_axes:
            continue
        try:
            bounds = balance_cuts(level_shapes, pair_boxes, ax,
                                  n_ranks, ghost)
        except ValueError:
            continue
        _, worst, _ = chain_owns(level_shapes, pair_boxes, ax, bounds, ghost)
        if best is None or worst < best[2] - 1e-12:
            best = (ax, bounds, worst)
    if best is None:
        raise ValueError(f"no feasible split axis for {n_ranks} ranks")
    return best[0], best[1]


# =============================================================================
# Block-tree decomposition (patch 18): several blocks per level
# =============================================================================
# The chain rule above ("place every cut inside the innermost box so every rank
# owns every level") has no meaning once a level hosts blocks scattered across
# the domain — there is no single innermost box. What replaces it:
#
#   * balance the WEIGHTED cell count (cells x 2^level = updates per coarse
#     step), taken from the blocks' own L0 spans;
#   * allow a rank to own ZERO cells of a block (the runner skips it with its
#     subtree), but never a sliver: 0 < own < ghost has no valid halo;
#   * cuts land on positions that are safe for EVERY block at once, so the
#     sliver condition cannot appear between any pair of cuts.
#
# Why per-cut safety is sufficient for every rank's piece: a rank's piece of a
# block is F(c_hi) - F(c_lo) with F the block's cumulative fine-cell count.
# If both cuts are safe then each of F(c_lo), F(c_hi) is 0, N_b, or at least
# `ghost` away from both; and interior L0 cells contribute 2^level >= 1 fine
# cells each while cuts are >= ghost apart in L0 — so a middle piece is either
# empty or >= ghost. The exact table is verified at the end regardless.


def block_spans_L0(blocks, axis: int) -> List[Tuple[float, float]]:
    """[(A, B)] — each block's extent along `axis` in fractional L0 units."""
    return [(float(b.origin[axis]),
             float(b.origin[axis]) + b.shape[axis] * b.spacing)
            for b in blocks]


def _tree_weight_profile(blocks, axis: int, n0: int):
    """Per-L0-cell update density summed over all blocks."""
    import numpy as _np
    w = _np.zeros(n0)
    for b, (A, B) in zip(blocks, block_spans_L0(blocks, axis)):
        upd = float(b.shape[0] * b.shape[1] * b.shape[2]) * (2.0 ** b.level)
        dens = upd / (B - A)
        for i in range(max(0, int(_np.floor(A))),
                       min(n0, int(_np.ceil(B)))):
            ov = min(i + 1.0, B) - max(float(i), A)
            if ov > 0:
                w[i] += dens * ov
    return w


def _block_cumulative(blocks, axis: int, n0: int):
    """(cum, totals): cum[uid][c] = fine cells of `uid` inside L0 [0, c).

    Computed with the SAME derivation the runner uses, so "safe" here and
    "feasible" there cannot drift apart.
    """
    import numpy as _np
    nb = len(blocks)
    cum = _np.zeros((nb, n0 + 1), dtype=_np.int64)
    for c in range(n0 + 1):
        table = derive_block_ranges(blocks, axis, [0, c, n0])
        for uid in range(nb):
            cum[uid, c] = table[uid][0][1]
    return cum, cum[:, n0].copy()


def _safe_cut_mask(cum, totals, ghost: int, keep_whole=()):
    """Boolean mask over L0 positions 0..n0 of cuts valid for every block."""
    import numpy as _np
    safe = _np.ones(cum.shape[1], dtype=bool)
    keep = set(int(u) for u in keep_whole)
    for uid in range(cum.shape[0]):
        F = cum[uid]
        N = int(totals[uid])
        if uid in keep:
            safe &= (F == 0) | (F == N)        # never split this block
            continue
        safe &= ~(((F > 0) & (F < ghost)) |
                  (((N - F) > 0) & ((N - F) < ghost)))
    return safe


def _nearest_safe(safe, target: int, lo: int, hi: int) -> Optional[int]:
    """Safe position closest to `target` within [lo, hi]."""
    if hi < lo:
        return None
    target = min(max(int(target), lo), hi)
    for d in range(0, hi - lo + 1):
        for c in (target - d, target + d):
            if lo <= c <= hi and safe[c]:
                return int(c)
    return None


def balance_cuts_tree(blocks, axis: int, n_ranks: int, ghost: int = 3,
                      keep_whole=()) -> List[int]:
    """Weighted L0 cut bounds for a block TREE along `axis`.

    keep_whole: uids that must land entirely inside one rank (no cut may pass
    through them). Used by the 'aligned' cut policy for the blocks carrying an
    ALM model — a rotor block owned by a single rank needs neither halo
    exchange nor a sampling allreduce.
    """
    import numpy as _np
    n0 = int(blocks[0].shape[axis])
    if n_ranks == 1:
        return [0, n0]
    w = _tree_weight_profile(blocks, axis, n0)
    cw = _np.concatenate([[0.0], _np.cumsum(w)])
    cum, totals = _block_cumulative(blocks, axis, n0)
    safe = _safe_cut_mask(cum, totals, ghost, keep_whole)

    gap = max(int(ghost), 2)
    cuts: List[int] = []
    lo = gap
    for i in range(1, n_ranks):
        target = int(_np.searchsorted(cw, cw[-1] * i / n_ranks))
        hi = n0 - gap * (n_ranks - i)
        c = _nearest_safe(safe, target, lo, hi)
        if c is None:
            raise ValueError(
                f"axis {AXIS_NAME[axis]}: no cut position for rank boundary "
                f"{i}/{n_ranks} in [{lo},{hi}] leaves every block either "
                f"unowned or with >= {ghost} cells"
                + (f" while keeping blocks {sorted(keep_whole)} whole"
                   if keep_whole else "")
                + " — try another axis, fewer ranks, or a smaller --ghost")
        cuts.append(c)
        lo = c + gap
    bounds = [0] + cuts + [n0]

    # exact check with the runner's own derivation (belt: the mask argument
    # above is a proof, this is the measurement)
    for uid, rows in enumerate(derive_block_ranges(blocks, axis, bounds)):
        for r, (_s, cnt) in enumerate(rows):
            if 0 < cnt < ghost:
                raise ValueError(
                    f"axis {AXIS_NAME[axis]}: cuts {cuts} leave rank{r} with "
                    f"{cnt} cells of block '{blocks[uid].name}' "
                    f"(< ghost {ghost})")
    return bounds


def tree_loads(blocks, axis: int, bounds: Sequence[int]) -> List[float]:
    """Per-rank cell-updates per coarse step, exact from the ownership table."""
    table = derive_block_ranges(blocks, axis, bounds)
    nr = len(bounds) - 1
    loads = [0.0] * nr
    for uid, b in enumerate(blocks):
        cross = 1
        for d in range(3):
            if d != axis:
                cross *= int(b.shape[d])
        wgt = float(cross) * (2.0 ** b.level)
        for r in range(nr):
            loads[r] += wgt * table[uid][r][1]
    return loads


def choose_axis_balanced_tree(blocks, n_ranks: int, ghost: int = 3,
                              keep_whole=(), exclude_axes=()):
    """(axis, bounds) minimizing the worst-rank update share for a tree.

    exclude_axes: see choose_axis_balanced (eso implicit-wall axes)."""
    best = None
    for ax in _AUTO_ORDER:
        if ax in exclude_axes:
            continue
        try:
            bounds = balance_cuts_tree(blocks, ax, n_ranks, ghost, keep_whole)
        except ValueError:
            continue
        loads = tree_loads(blocks, ax, bounds)
        worst = max(loads) / max(sum(loads), 1e-30)
        if best is None or worst < best[2] - 1e-12:
            best = (ax, bounds, worst)
    if best is None:
        raise ValueError(
            f"no feasible split axis for {n_ranks} ranks over "
            f"{len(blocks)} blocks (ghost={ghost})")
    return best[0], best[1]


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
                 periodic: bool = True, allow_empty: bool = False) -> None:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0/1/2, got {axis}")
        if not (0 <= rank < n_ranks):
            raise ValueError(f"rank {rank} outside n_ranks {n_ranks}")
        N = int(global_shape[axis])
        if n_ranks > N and not allow_empty:
            # allow_empty: the caller supplies explicit ranges (from_range), so
            # the even-split feasibility check is meaningless — a refinement
            # block smaller than the rank count is legal, most ranks own none.
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
        """Rank on `side` (0=low, 1=high); None at a non-periodic boundary.

        Ranks holding ZERO cells of this grid are skipped over. A refinement
        block that covers only part of the domain is owned by a contiguous
        subset of ranks, and the ones outside it have no array to exchange
        with — posting a halo band to them would leave the send unmatched.
        With every rank owning cells (the single-block chain) this is exactly
        rank +- 1 as before.
        """
        step = 1 if side == 1 else -1
        counts = self._counts
        if self.periodic:
            r = (self.rank + step) % self.n_ranks
            for _ in range(self.n_ranks):
                if r == self.rank:
                    return None          # sole owner: no partner
                if counts[r] != 0:
                    return r
                r = (r + step) % self.n_ranks
            return None
        r = self.rank + step
        while 0 <= r < self.n_ranks:
            if counts[r] != 0:
                return r
            r += step
        return None

    @classmethod
    def from_range(cls, global_shape: Tuple[int, int, int], n_ranks: int,
                   rank: int, axis: int, own_start: int, own_count: int,
                   ghost: int = 2, periodic: bool = True,
                   all_counts: Optional[Sequence[int]] = None) -> "Partition1D":
        """Partition with an EXPLICIT owned range (MLG coarse-aligned cuts).

        Fine-level partitions are DERIVED from the coarse cuts (fine index =
        2*(coarse - box_lo)), not split independently — this keeps coupling
        regions rank-local (patch 17 M2)."""
        p = cls(global_shape, n_ranks, rank, axis, ghost=ghost,
                periodic=periodic, allow_empty=True)
        p.own_start = int(own_start)
        p.own_count = int(own_count)
        ls = list(p.global_shape)
        ls[axis] = p.own_count + 2 * p.ghost
        p.local_shape = tuple(ls)
        # Owned counts for EVERY rank. __init__ filled an even split, which is
        # wrong for a derived range; neighbor() needs the real ones to know
        # which ranks hold nothing of this grid. Every rank can compute all of
        # them locally from the shared cuts, so this costs no communication.
        if all_counts is not None:
            p._counts = [int(c) for c in all_counts]
        else:
            p._counts[rank] = p.own_count
        return p

    def __repr__(self) -> str:
        return (f"Partition1D(axis={AXIS_NAME[self.axis]}, rank={self.rank}/"
                f"{self.n_ranks}, own=[{self.own_start},"
                f"{self.own_start + self.own_count}), ghost={self.ghost})")
