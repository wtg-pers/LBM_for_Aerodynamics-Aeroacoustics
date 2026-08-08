"""
Block tree — the MLG topology, generalized from a chain to a tree.

WHY
---
MLG has always been a strictly nested telescope: exactly one refinement box per
level, level k's box inside level k-1's. Three assumptions encode that, and all
three break as soon as a level hosts two boxes:

    level index k  ==  levels list index k  ==  coupling index k-1

That is fine for one body in the middle of a domain. It cannot express eight
rotors scattered 3 m apart, each wanting its own small fine box: a single box
covering all of them at the target resolution costs 277 M cells where eight
separate boxes cost 71 M.

This module replaces the flat lists with an explicit tree. A CHAIN is just the
degenerate tree with one child per node, and `build_chain` produces exactly
that — so the existing recursion, buffer sizes and allocation order are
reproduced element for element, which is what makes the refactor bit-identical.

WHAT LIVES ON A BLOCK
---------------------
Geometry (`shape`/`origin`/`spacing`/`global_box`/`region`) is filled while the
grid is laid out; runtime state (`sim`/`coupling`/`f_prev`) while it is built.
One class for both phases on purpose: a separate spec class would give origins
two sources of truth, which the VTK writer already suffers from (it re-derives
origins that setup has already computed).

`f_prev` belongs to the CHILD, not the parent. It holds the parent's pre-step
distribution over *this* block's coarse sub-volume, which is what the half-step
C2F interpolates against. With one child per level the old `_f_prev[k]` meant
the same thing; with several children each needs its own window on the parent.
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

__all__ = ["GridBlock", "build_chain", "iter_blocks", "blocks_at", "depth"]


@dataclass
class GridBlock:
    """One refinement block: a grid, its coupling to its parent, its children."""

    # ── identity ────────────────────────────────────────────────
    level: int                       # 0 = coarsest
    index: int = 0                   # position among blocks at this level
    uid: int = 0                     # position in the level-major flat list
    name: str = ""

    # ── geometry, in L0 lattice units ───────────────────────────
    shape: Tuple[int, int, int] = (0, 0, 0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: float = 1.0             # = 2 ** -level
    region: Optional[object] = None  # OverlapRegion to the parent; None = root

    # ── topology ────────────────────────────────────────────────
    parent: Optional["GridBlock"] = None
    children: List["GridBlock"] = field(default_factory=list)

    # ── runtime ─────────────────────────────────────────────────
    sim: Optional[object] = None
    coupling: Optional[object] = None    # to the parent; None = root
    f_prev: Optional[object] = None      # parent's pre-step f over this block

    def __post_init__(self):
        if not self.name:
            self.name = f"L{self.level}" if self.index == 0 \
                else f"L{self.level}b{self.index}"

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def label(self) -> str:
        """Display label: plain level name for chains, block-qualified else."""
        return f"L{self.level}" if self.index == 0 and not self._has_siblings \
            else f"L{self.level}.b{self.index} '{self.name}'"

    @property
    def _has_siblings(self) -> bool:
        return self.parent is not None and len(self.parent.children) > 1

    @property
    def global_box(self) -> Tuple[int, int, int, int, int, int]:
        """Integer extent in THIS LEVEL's global lattice.

        Exact integer arithmetic (origin is a multiple of spacing by
        construction), so sibling-disjointness tests never depend on a float
        tolerance, and it is the same quantity the .vth writer needs.
        """
        lo = [int(round(o / self.spacing)) for o in self.origin]
        return (lo[0], lo[0] + self.shape[0] - 1,
                lo[1], lo[1] + self.shape[1] - 1,
                lo[2], lo[2] + self.shape[2] - 1)

    def __repr__(self) -> str:
        return (f"GridBlock({self.label}, shape={self.shape}, "
                f"origin={tuple(round(o, 3) for o in self.origin)}, "
                f"children={len(self.children)})")


# =============================================================================
# Construction
# =============================================================================
def build_chain(levels: Sequence[object],
                couplings: Sequence[object]) -> GridBlock:
    """Degenerate tree (one child per node) from the legacy flat lists.

    This is the compatibility path: `MultiLevelGrid(levels, couplings)` still
    takes the old arguments and routes them through here, so every existing
    caller keeps working and produces a tree whose traversal order is the old
    list order.
    """
    root = GridBlock(level=0, index=0, uid=0, name="L0",
                     shape=tuple(levels[0].domain_shape),
                     origin=(0.0, 0.0, 0.0), spacing=1.0,
                     sim=levels[0])
    prev = root
    for k in range(1, len(levels)):
        cpl = couplings[k - 1]
        region = getattr(cpl, 'region', None) or getattr(cpl, '_region', None)
        origin = prev.origin
        if region is not None:
            fdc = region.fine_domain_coarse
            origin = (prev.origin[0] + fdc.x_start * prev.spacing,
                      prev.origin[1] + fdc.y_start * prev.spacing,
                      prev.origin[2] + fdc.z_start * prev.spacing)
        blk = GridBlock(level=k, index=0, uid=k, name=f"L{k}",
                        shape=tuple(levels[k].domain_shape),
                        origin=origin, spacing=0.5 ** k,
                        region=region, parent=prev, sim=levels[k],
                        coupling=cpl)
        prev.children.append(blk)
        prev = blk
    return root


# =============================================================================
# Traversal
# =============================================================================
def iter_blocks(root: GridBlock) -> Iterator[GridBlock]:
    """Level-major, config order within a level, root first.

    Load-bearing: for a chain this is exactly the old `levels` list order, so
    lazy buffer allocation happens in the same sequence with the same sizes.
    That keeps the memory-pool layout — and therefore CUDA-graph pointer
    stability — identical to the pre-refactor code.
    """
    frontier = [root]
    while frontier:
        for b in frontier:
            yield b
        nxt: List[GridBlock] = []
        for b in frontier:
            nxt.extend(b.children)
        frontier = nxt


def blocks_at(root: GridBlock, level: int) -> List[GridBlock]:
    return [b for b in iter_blocks(root) if b.level == level]


def depth(root: GridBlock) -> int:
    """Number of levels (root-only tree = 1)."""
    return max(b.level for b in iter_blocks(root)) + 1


# =============================================================================
# Validation
# =============================================================================
def _parent_interior(block: GridBlock) -> Optional[Tuple[int, ...]]:
    """The parent's own `fine_region`, in PARENT-LOCAL index space.

    Outside this box lies the parent's own coupling band — cells its own parent
    overwrites twice per coarse step. A grandchild placed there would read
    interpolated rather than computed values, and would restrict back into
    cells that are overwritten immediately afterwards.
    """
    p = block.parent
    if p is None or p.region is None:
        return None                     # parent is the root: whole domain
    fr, fd = p.region.fine_region, p.region.fine_domain_coarse
    r = getattr(p.region, "REFINE_RATIO", 2)
    return ((fr.x_start - fd.x_start) * r, (fr.x_end - fd.x_start) * r,
            (fr.y_start - fd.y_start) * r, (fr.y_end - fd.y_start) * r,
            (fr.z_start - fd.z_start) * r, (fr.z_end - fd.z_start) * r)


def validate_block_tree(root: GridBlock, overlap_width: int = 2,
                        nesting: str = "warn") -> List[str]:
    """Check the two rules that make a multi-block level well posed.

    Rule B (ERROR) — blocks on the SAME LEVEL must have disjoint
      `fine_domain_coarse` extents, cousins included (different parents still
      share physical space).
      The strictly necessary condition is weaker and asymmetric: A's `excised`
      must miss B's `fine_domain_coarse`, since a sibling's F2C write can only
      reach a sibling through the parent. Disjointness on `fine_region` alone
      is NOT enough — two blocks three coarse cells apart are region-disjoint
      yet A's excised write lands inside B's read band, making the answer
      depend on sibling order. The stronger rule is enforced because it (i)
      implies the necessary one and removes any order dependence, (ii) avoids
      two fine grids evolving the same physical volume independently with only
      their excised parts fed back — leaving two different answers in the
      shared band with no principled way to pick one, and (iii) matches
      vtkOverlappingAMR, which requires non-overlapping boxes per level.

    Rule A (WARN by default) — a block must nest inside its parent's own
      `fine_region`, not merely inside the parent's array. Ships as a warning
      first so an existing config cannot be rejected without evidence; promote
      with nesting="error".

    Returns the list of warnings; raises ValueError on a Rule B violation.
    """
    warnings: List[str] = []
    blocks = list(iter_blocks(root))

    # ── Rule B: same-level disjointness ─────────────────────────
    by_level = {}
    for b in blocks:
        by_level.setdefault(b.level, []).append(b)
    for level, group in by_level.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, c = group[i], group[j]
                ab, cb = a.global_box, c.global_box
                overlap = all(ab[2 * d] <= cb[2 * d + 1]
                              and cb[2 * d] <= ab[2 * d + 1] for d in range(3))
                if overlap:
                    raise ValueError(
                        f"MLG blocks '{a.name}' and '{c.name}' both on level "
                        f"{level} overlap: {ab} vs {cb} (level-{level} lattice "
                        f"indices, overlap band included).\n"
                        f"  Two fine grids would evolve the same physical "
                        f"volume independently while only their excised parts "
                        f"feed back — the shared band would hold two different "
                        f"answers. Separate the regions.")
                # Two boxes are separated along the axis that separates them
                # BEST, so take the max over axes — a min would report the
                # overlap on the axes they happen to share and never warn.
                gap = max(max(cb[2 * d] - ab[2 * d + 1] - 1,
                              ab[2 * d] - cb[2 * d + 1] - 1) for d in range(3))
                if 0 <= gap < 2 * overlap_width:
                    warnings.append(
                        f"blocks '{a.name}' and '{c.name}' (level {level}) are "
                        f"only {gap} cells apart; the coarse solution between "
                        f"them has less than {2 * overlap_width} cells to relax")

    # ── Rule A: nest inside the parent's own fine_region ────────
    for b in blocks:
        interior = _parent_interior(b)
        if interior is None or b.region is None:
            continue
        fd = b.region.fine_domain_coarse
        got = (fd.x_start, fd.x_end, fd.y_start, fd.y_end, fd.z_start, fd.z_end)
        bad = [ax for d, ax in enumerate("xyz")
               if got[2 * d] < interior[2 * d] or got[2 * d + 1] > interior[2 * d + 1]]
        if bad:
            msg = (f"block '{b.name}' (level {b.level}) reaches into its "
                   f"parent's coupling band on axis/axes {','.join(bad)}: "
                   f"{got} vs parent interior {interior} (parent-local). Cells "
                   f"there are overwritten by the parent's own parent twice per "
                   f"coarse step.")
            if nesting == "error":
                raise ValueError(msg)
            warnings.append(msg)

    return warnings
