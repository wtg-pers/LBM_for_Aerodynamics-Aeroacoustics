"""Wall-aware MLG coupling — exclude the wall neighbourhood from C2F/F2C.

Why this exists
---------------
The C2F/F2C transfer assumes every cell it reads and writes carries a
hydrodynamic state. Cells at a wall do not:

  * SOLID cells hold the implicit-HWBB bounce DEPOSITS of their fluid
    neighbours (patch 12), not a distribution. Writing a coupling value
    there lands foreign data on a live bounce slot.
  * The first FLUID layer next to a wall is owned by the wall BC — its
    populations are (re)written by the bounce / IBB deposit pass every
    substep. A coupling write there overwrites the wall treatment.

Only the first of those is handled today (`skip_solid_nt`, patch 12).
`setup._check_body_vs_coupling_band` closes the rest by REFUSING to build
when the body intersects the C2F band at all. That refusal is what forbids
per-rotor fine blocks on a vehicle whose airframe reaches the block faces
(`configs/octo8/octo8_blocks8_preview.py` records the measurement).

This module implements the alternative the LBM community uses for that
case — OpenLB's "only couple bulk cells": mark the wall neighbourhood
non-eligible and let the coupling skip it, instead of forbidding the
layout. It is an EXPERIMENT, off by default:

    mlg:
      wall_coupling:
        mode: "strict"     # default — today's hard rejection, bit-identical
        # mode: "exclude"  # skip the wall neighbourhood, allow the layout
        # mode: "allow"    # accept the layout, coupling UNCHANGED
        wall_margin: 1     # fluid layers next to solid also excluded
        apply_to: "c2f"    # which transfer the exclusion covers

Why apply_to defaults to C2F only (measured, rig_cut at 4 rev)
---------------------------------------------------------------
The conflict this module exists to fix is one-directional. C2F writes
INTO the fine boundary strip, so at a wall it overwrites populations the
fine level's own wall BC just produced. F2C writes the other way, into
the coarse excised region — and there the wall-adjacent coarse FLUID
cells legitimately want the fine level's near-wall solution; only the
coarse SOLID cells were ever broken, and patch 12 already skips those.

Excluding both directions is much the larger intervention, and it is the
wrong one. On the rig the C2F band protection is 379 cells (0.9% of the
band), while the same margin blocks F2C at 1,082 coarse fluid cells
wrapping the WHOLE body — starving the coarse level of fine data
everywhere, not just at the cut. Body force vs the enclosed-body
reference, 4 revolutions:

    allow  (no exclusion)             -4.5%
    both,  wall_margin=1              -9.3%    1,082 coarse cells blocked
    both,  wall_margin=2             -25.3%    2,572 coarse cells blocked

Monotone in the margin, and away from the reference — the signature of
starvation, not of a fixed artifact. Hence 'c2f'. 'both' and 'f2c' remain
selectable because that comparison is the evidence and should stay
reproducible.

'allow' exists so the experiment has a control. Cutting a body out of the
fine level changes two things at once: the coupling now runs at a wall,
AND the cut-off part loses resolution. Comparing 'exclude' against a body
fully enclosed by the fine level cannot separate them. 'allow' holds the
geometry fixed and varies only the coupling, so ('allow' - 'exclude') is
what the wall exclusion actually buys.

★ And that quantity CHANGES SIGN WITH THE WALL BC (rig, 4 rev, body
force vs each leg's own enclosed-body reference):

    wall_bc    allow     exclude    exclusion buys
    hwbb      -4.63%     -2.73%        +1.90  (helps)
    ibb       -5.54%     -8.43%        -2.89  (hurts)

Mechanism: IBB runs a deposit-rewrite pass after every fused launch, so
it already corrects whatever C2F wrote at a wall-adjacent cell. Excluding
those cells then buys no correction and costs the outer inflow — pure
starvation. HWBB has no such after-pass, so blocking the overwrite is a
net gain there.

So 'exclude' is NOT a general improvement, and 'allow' is not merely a
control: on an IBB case it is the better of the two. Pick by measurement
against an enclosed-body reference, per case. Both sit ~5% off that
reference here, because neither addresses the READ side.

What this does NOT do (read the caveat before trusting a result)
----------------------------------------------------------------
Only the WRITE side is excluded. The coupling still READS through the
body: the coarse face slab feeding the cubic upsample may contain solid
cells, so fine strip cells further than `wall_margin` from the wall
receive values interpolated across the body. Fixing that needs either a
ghost-node fill (Daeian et al., CPC 2025) or a mask-renormalised
interpolation, both of which change the coupling stencil itself.

Sequencing this way is deliberate: the write-side exclusion is the cheap
prescription the literature actually states, it carries no rank-invariance
risk (see below), and its result tells us whether the expensive read-side
work is justified. `report_band_exclusion` prints what fraction of each
C2F band face is being skipped so a nonsense layout is visible at build
time rather than after a cluster run.

Rank invariance
---------------
The dilation is computed ONCE on the level's full solid mask and then
sliced per rank exactly like `node_type`. It is therefore a global
geometric property, not a rank-local one — no halo width enters, and an
n-rank run sees the same flags as a 1-rank run by construction. Deriving
it from a rank-local slab instead would make the flags depend on the cut
(the ghost band is 2 cells; a margin >= 2 would differ at slab edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

VALID_MODES = ("strict", "allow", "exclude")
VALID_DIRECTIONS = ("c2f", "f2c", "both")

_SOLID = 1          # node_type / skip-flag value the coupling scatter skips


@dataclass(frozen=True)
class WallCouplingPolicy:
    """How the MLG coupling treats cells at a wall."""

    mode: str = "strict"
    wall_margin: int = 1
    apply_to: str = "c2f"

    def applies(self, direction: str) -> bool:
        """Whether the wall exclusion covers this transfer direction."""
        if not self.excludes_wall:
            return False
        return self.apply_to in (direction, "both")

    @property
    def relaxes_guard(self) -> bool:
        """True when a body intersecting the C2F band is accepted."""
        return self.mode in ("allow", "exclude")

    @property
    def excludes_wall(self) -> bool:
        """True when the wall neighbourhood is skipped instead of rejected."""
        return self.mode == "exclude"

    def describe(self) -> str:
        if self.mode == "strict":
            return "strict (body may not touch the C2F band)"
        if self.mode == "allow":
            return ("allow (band intersection accepted, coupling UNCHANGED)")
        return (f"exclude on {self.apply_to.upper()} (skip solid + "
                f"{self.wall_margin} fluid layer"
                f"{'s' if self.wall_margin != 1 else ''} at the wall)")


def check_margin_vs_band(policy: WallCouplingPolicy, overlap_width: int,
                         refine_ratio: int = 2) -> None:
    """Reject a wall_margin that eats the C2F band it is protecting.

    C2F writes a strip `overlap_width * refine_ratio` fine cells deep.
    Excluding `wall_margin` layers from BOTH the wall side and (through
    the body's two faces) the strip's usable depth leaves the fine
    boundary with no coarse inflow, and those cells then coast on stale
    populations. Measured on the rig with overlap_width=2 (band 4 cells),
    body force vs the enclosed-body reference:

        wall_margin=1   -2.73%    (unprotected control: -4.63%)
        wall_margin=2  -108.20%   <- band half gone, solution destroyed

    So the margin must stay strictly below overlap_width.
    """
    if not policy.excludes_wall or not policy.applies('c2f'):
        return
    band = overlap_width * refine_ratio
    if 2 * policy.wall_margin >= band:
        raise ValueError(
            f"mlg.wall_coupling.wall_margin={policy.wall_margin} is too large "
            f"for overlap_width={overlap_width}: the C2F band is {band} fine "
            f"cells deep, and excluding {policy.wall_margin} layer(s) leaves "
            f"the fine boundary without coarse inflow (measured: margin=2 at "
            f"overlap_width=2 moved the body force by -108%). Use "
            f"wall_margin < {overlap_width}, or raise mlg.overlap_width.")


def parse_wall_coupling(mlg_config: Optional[dict]) -> WallCouplingPolicy:
    """Build the policy from the `mlg.wall_coupling` config block."""
    cfg = (mlg_config or {}).get('wall_coupling', {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"mlg.wall_coupling must be a dict, got {type(cfg).__name__}")
    unknown = set(cfg) - {'mode', 'wall_margin', 'apply_to'}
    if unknown:
        raise ValueError(
            f"mlg.wall_coupling: unknown key(s) {sorted(unknown)}; "
            f"valid keys are 'mode', 'wall_margin', 'apply_to'")

    mode = cfg.get('mode', 'strict')
    if mode not in VALID_MODES:
        raise ValueError(
            f"mlg.wall_coupling.mode must be one of {VALID_MODES}, "
            f"got {mode!r}")

    margin = int(cfg.get('wall_margin', 1))
    if margin < 0:
        raise ValueError(
            f"mlg.wall_coupling.wall_margin must be >= 0, got {margin}")
    if mode == 'exclude' and margin == 0:
        # (mode='allow' is exactly this, and says so in its name)
        raise ValueError(
            "mlg.wall_coupling: mode='exclude' with wall_margin=0 is exactly "
            "the existing solid-only skip (patch 12) and would relax the band "
            "guard for no added protection. Use wall_margin >= 1, or keep "
            "mode='strict'.")

    apply_to = cfg.get('apply_to', 'c2f')
    if apply_to not in VALID_DIRECTIONS:
        raise ValueError(
            f"mlg.wall_coupling.apply_to must be one of {VALID_DIRECTIONS}, "
            f"got {apply_to!r}")
    return WallCouplingPolicy(mode=mode, wall_margin=margin,
                              apply_to=apply_to)


# =====================================================================
# Mask construction
# =====================================================================

def _axis_slice(ndim: int, axis: int, sl: slice) -> tuple:
    out = [slice(None)] * ndim
    out[axis] = sl
    return tuple(out)


def dilate_chebyshev(xp, solid, margin: int):
    """Dilate a boolean mask by `margin` in the Chebyshev (D3Q27) ball.

    Separable: one pass per axis per unit of margin (6 shifted ORs in 3D
    instead of 26), chained so the result is the box structuring element
    the D3Q27 stencil actually reaches. Slice-based, NOT roll-based — a
    body touching a domain face must not wrap onto the opposite face.
    """
    if margin <= 0:
        return solid
    out = solid
    ndim = solid.ndim
    for _ in range(margin):
        for axis in range(ndim):
            if out.shape[axis] < 2:
                continue
            cur = out
            nxt = cur.copy()
            lo = _axis_slice(ndim, axis, slice(0, -1))
            hi = _axis_slice(ndim, axis, slice(1, None))
            nxt[hi] |= cur[lo]
            nxt[lo] |= cur[hi]
            out = nxt
    return out


def build_coupling_skip(xp, solid_mask, margin: int):
    """Skip flags for the coupling scatter: 1 = do not couple, 0 = couple.

    Same {0, 1} int8 contract as the `skip_solid_nt` argument of
    `esoteric_scatter_std_region` (it tests `== 1`), so consumers only
    swap WHICH array they pass — the scatter itself is untouched.

    Returns None when the level carries no solid cells (nothing to skip;
    the caller then keeps its existing node-type array).
    """
    if solid_mask is None:
        return None
    m = xp.asarray(solid_mask, dtype=bool)
    if not bool(m.any()):
        return None
    return dilate_chebyshev(xp, m, margin).astype(xp.int8).ravel()


def wall_skip_nt(sim, direction: str):
    """The wall-aware skip flags for `sim`, or None when they don't apply.

    None whenever the policy is strict/allow, the level has no body, or
    the policy does not cover `direction`. For the STANDARD-layout
    coupling path, which has never masked its writes, that None keeps the
    path byte-for-byte as it was: patch 12's solid skip is esoteric-only
    (a standard solid cell holds a real, if unused, distribution, so
    writing it is harmless).
    """
    skip = getattr(sim, '_coupling_skip_nt', None)
    if skip is None:
        return None
    dirs = getattr(sim, '_coupling_skip_dirs', ())
    return skip if direction in dirs else None


def coupling_skip_nt(sim, direction: str):
    """The skip flags an ESOTERIC coupling scatter should use for `sim`.

    Falls back to the level's esoteric node type — the patch-12 solid-only
    skip, which this path has always applied — whenever the wall-aware mask
    does not apply. Callers therefore never branch on the policy.
    """
    skip = wall_skip_nt(sim, direction)
    if skip is not None:
        return skip
    return getattr(sim, '_eso_node_type', None)


def attach_coupling_skip(sim, solid_mask, policy: WallCouplingPolicy) -> int:
    """Build and attach `sim._coupling_skip_nt`; return the flagged count.

    No-op (returns 0) under the default strict policy, so a strict build
    allocates nothing and behaves bit-identically to before this module.
    """
    sim._coupling_skip_nt = None
    sim._coupling_skip_dirs = ()
    if not policy.excludes_wall:
        return 0
    skip = build_coupling_skip(sim.xp, solid_mask, policy.wall_margin)
    if skip is None:
        return 0
    sim._coupling_skip_nt = skip
    sim._coupling_skip_dirs = tuple(
        d for d in ("c2f", "f2c") if policy.applies(d))
    return int(skip.sum())


# =====================================================================
# Build-time diagnostic
# =====================================================================

def report_band_exclusion(solid_mask_np, region, margin: int) -> list:
    """Per-face C2F band exclusion under `mode='exclude'`.

    The band is the strip between the fine-domain edge and the user's
    fine_region edge — the zone C2F writes. A face where most cells are
    excluded has no coarse->fine inflow left and the layout is nonsense;
    this is what makes that visible at build time.

    Returns [(face, n_band_cells, n_excluded), ...] for faces with a
    non-zero band, worst first.
    """
    import numpy as np

    ratio = int(getattr(region, 'REFINE_RATIO', 2))
    fd, fr = region.fine_domain_coarse, region.fine_region
    axes = ['x', 'y'] + (['z'] if solid_mask_np.ndim == 3 else [])

    solid = np.asarray(solid_mask_np, dtype=bool)
    if not solid.any():
        return []

    excluded = dilate_chebyshev(np, solid, margin)

    out = []
    for ax_i, ax in enumerate(axes):
        w_lo = (getattr(fr, f'{ax}_start') - getattr(fd, f'{ax}_start')) * ratio
        w_hi = (getattr(fd, f'{ax}_end') - getattr(fr, f'{ax}_end')) * ratio
        n_ax = solid.shape[ax_i]
        for face, w, sl in (
            (f'{ax}_low', w_lo, slice(0, w_lo)),
            (f'{ax}_high', w_hi, slice(n_ax - w_hi, n_ax)),
        ):
            if w <= 0:
                continue
            idx = [slice(None)] * solid.ndim
            idx[ax_i] = sl
            blk = excluded[tuple(idx)]
            n_ex = int(blk.sum())
            if n_ex:
                out.append((face, int(blk.size), n_ex))
    out.sort(key=lambda t: -t[2] / t[1])
    return out
