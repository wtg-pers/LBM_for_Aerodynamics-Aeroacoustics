"""
Multi-Level Grid — Nested Time-Stepping Orchestrator

Manages M grid levels with recursive nested time-stepping, providing
a unified advance() interface compatible with the existing Simulation
class. One call to advance() performs exactly one coarse timestep,
which internally triggers 2^k fine steps at each level k.

Physical Process (Lagrava Sandoval, Sec. 4.4.3, Sec. 5.1.2):

    One coarse timestep (t → t + δt_c):

    ┌─ Level 0 ────────────────────────────────────────────────────┐
    │  1. Save f_prev (for temporal interpolation)                 │
    │  2. Collide + Stream + BC  (full domain, t → t+δt_c)        │
    │                                                              │
    │  ┌─ Level 1 (2 fine steps) ──────────────────────────────┐   │
    │  │  3. C→F half-step  (temporal interp: f = avg(f_prev,f))│   │
    │  │  4. Collide + Stream + BC  (t → t+δt_f)               │   │
    │  │     ┌─ Level 2 (2 finer steps) ────────────────────┐  │   │
    │  │     │  ... (recursive)                              │  │   │
    │  │     └───────────────────────────────────────────────┘  │   │
    │  │  5. C→F full-step  (no temporal interp needed)         │   │
    │  │  6. Collide + Stream + BC  (t+δt_f → t+δt_c)          │   │
    │  │     ┌─ Level 2 (2 finer steps) ────────────────────┐  │   │
    │  │     │  ... (recursive)                              │  │   │
    │  │     └───────────────────────────────────────────────┘  │   │
    │  │  7. F→C: overwrite coarse overlap with fine result     │   │
    │  └───────────────────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────────────────┘

Integration:
    MultiLevelGrid exposes the same interface as Simulation:
        advance(), rho, u, f, step_count
    so that main.py and OutputManager work without modification.

    main.py loop (unchanged):
        for step in range(start, end):
            sim.advance()          # sim = MultiLevelGrid or Simulation
            output.process(step, sim)

Reference:
    Lagrava Sandoval, Ch. 5.1 (implementation ideas)
    Geier et al. (2015), Sec. 6 (5-level sphere)

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from src.utilities.step_profiler import timed as _prof, step_done as _prof_step

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.solver.simulation import Simulation
    from src.grid.coupling import GridCoupling


def _release_surfel_scratch(sim) -> None:
    """Per-advance scratch discipline for single-GPU surfel bridge
    levels (patch 75 — the same lifecycle rule SurfelSlabLevel.advance
    applies on the MPI path since 64 sec. 16): _f_post is fully
    rewritten by the collide kernel every advance and kernel.Q is
    fill(0)-ed at every apply, so neither carries cross-substep state —
    both re-materialize through their existing lazy paths. Left
    resident they stack per LEVEL (~108 B/node each): the span16+LE-L4
    single-GPU run held every level's f_post at once (~7 GiB) and
    OOMed. Ref-drop only — no pool flush (64 sec. 19)."""
    if not (getattr(sim, '_use_surfel', False)
            and getattr(sim, '_use_esoteric', False)):
        return
    sim._f_post = None
    k = getattr(getattr(sim, 'obstacle_bc', None), 'kernel', None)
    if k is not None:
        k.Q = None


def _rest_feq_27(xp, dtype):
    """D3Q27 rest-state equilibrium (rho=1, u=0) = the lattice weights,
    in surfel_transport's C27 ordering (patch 74 C2F dead fill)."""
    import numpy as _np
    from src.boundary.surfel_transport import C27 as _C27
    w = _np.array([8 / 27, 2 / 27, 1 / 54, 1 / 216])[
        _np.einsum('qd,qd->q', _C27, _C27)]
    return xp.asarray(w.astype(_np.dtype(dtype).type))


def _eso_wall_args(sim):
    """Implicit-domain-wall view args for the eso region gather/scatter
    (eso_wall patch 04). MLG coupling must read/write wall rows through
    the swap-slot/mailbox LOAD view, not the raw periodic bijection —
    the raw view hands stale wrap values to the C2F interpolation at
    wall rows (e.g. the octo8 bottom band reading L0's ground rows)."""
    return {'wall_mask': getattr(sim, '_eso_wall_mask', 0),
            'wall_mail': getattr(sim, '_eso_wall_mail', None)}


class MultiLevelGrid:
    """Orchestrates nested time-stepping across M grid levels.

    Owns M Simulation objects (one per level) and M-1 GridCoupling
    objects (one per adjacent pair). Provides a Simulation-compatible
    interface so that the main time loop is unaware of multi-level
    structure.

    Nested time-stepping cost per coarse step:
        Level 0:  1 step
        Level 1:  2 steps
        Level 2:  4 steps
        ...
        Level k:  2^k steps
        Total sub-steps: 2^M - 1

    Args:
        levels: List of Simulation objects, index 0 = coarsest.
                Each must have set_distribution() already called.
        couplings: List of GridCoupling objects, index i couples
                   levels[i] (coarse) with levels[i+1] (fine).
                   Length must be len(levels) - 1.

    Raises:
        ValueError: If levels/couplings lengths are inconsistent,
                    or if any level is not ready.

    Example:
        >>> mlg = MultiLevelGrid(levels=[sim_0, sim_1], couplings=[coupling_01])
        >>> for step in range(1000):
        ...     mlg.advance()
        ...     print(f"Step {step}: ρ_max = {mlg.rho.max():.6f}")
    """

    def __init__(
        self,
        levels: List['Simulation'],
        couplings: List['GridCoupling'],
    ) -> None:
        # ── Validate structure ────────────────────────────────────
        if len(couplings) != len(levels) - 1:
            raise ValueError(
                f"Need exactly {len(levels)-1} couplings for {len(levels)} "
                f"levels, got {len(couplings)}."
            )

        # A chain is the degenerate tree (one child per node). Routing the
        # legacy arguments through build_chain keeps every existing caller
        # working AND makes the traversal order identical to the old list
        # order — which is what keeps this refactor bit-identical.
        from src.grid.block_tree import build_chain
        self._bind_tree(build_chain(levels, couplings))
        self._step_count = 0

        # ── f_prev buffers (allocated lazily on first advance()) ─
        # Each block except the root needs its parent's pre-step f over its own
        # coarse sub-volume, for the C→F half-step. Cannot allocate here
        # because f may not be set yet (Layer 1→2). Lives on the block: with
        # several children per parent, each needs its own window.
        self._f_prev_initialized: bool = False
        # (profiler section names are set by _bind_tree)

        # ── Phase 1c — CUDA Graphs (S1: pure-LBM whole coarse-step) ──────
        # advance() = ~2^M-1 sim.advance() + coupling → 150-300 kernel launches
        # per coarse step. Phase 0 verdict = launch-bound (3620 ms ≫ 25-50 ms
        # bandwidth floor). Capturing the entire recursion into one CUDA graph
        # and replaying it removes per-launch/driver overhead (08 design; PoC
        # 2.6×). ONLY valid for pure-LBM: ALM keeps a CPU-BEM in the L4 loop
        # (uncapturable) and reallocates rho/u; see _graph_precheck().
        self._graph_enabled = os.environ.get("MLG_CUDA_GRAPH", "0") == "1"
        self._graph = None                 # captured cp.cuda.Graph (None until captured)
        self._graph_stream = None          # non-blocking capture/replay stream
        self._graph_warmup = 0             # eager steps done before capture
        self._graph_failed = False         # permanent eager fallback after a failure
        self._graph_ok: Optional[bool] = None   # precheck result (lazy, cached)
        # Eager warmup before capture: allocate lazy buffers (rho/u, f_prev),
        # JIT-compile every RawKernel, and stabilise memory-pool pointers so the
        # captured graph binds fixed device addresses.
        self._GRAPH_WARMUP = int(os.environ.get("MLG_CUDA_GRAPH_WARMUP", "3"))

    # =================================================================
    # Block tree
    # =================================================================

    @classmethod
    def from_tree(cls, root) -> 'MultiLevelGrid':
        """Build from an explicit block tree (levels may hold several blocks)."""
        obj = cls.__new__(cls)
        obj._bind_tree(root)
        obj._step_count = 0
        obj._f_prev_initialized = False
        obj._graph_enabled = os.environ.get("MLG_CUDA_GRAPH", "0") == "1"
        obj._graph = None
        obj._graph_stream = None
        obj._graph_warmup = 0
        obj._graph_failed = False
        obj._graph_ok = None
        obj._GRAPH_WARMUP = int(os.environ.get("MLG_CUDA_GRAPH_WARMUP", "3"))
        return obj

    def _bind_tree(self, root) -> None:
        from src.grid.block_tree import depth, iter_blocks
        self._root = root
        self._blocks = list(iter_blocks(root))          # level-major
        self._num_levels = depth(root)
        # Profiler sections stay LEVEL-indexed: step_profiler accumulates by
        # name, so sibling blocks sum into one section — which is what the
        # launch-bound vs bandwidth attribution wants.
        self._pn_adv = tuple(f"L{k}.advance" for k in range(self._num_levels))
        self._pn_c2f = tuple(f"C2F.L{k}" for k in range(self._num_levels))
        self._pn_f2c = tuple(f"F2C.L{k}" for k in range(self._num_levels))
        self._pn_fprev = tuple(f"fprev.L{k}" for k in range(self._num_levels))

    @property
    def root(self):
        return self._root

    def iter_blocks(self):
        return iter(self._blocks)

    def blocks_at(self, level: int) -> list:
        return [b for b in self._blocks if b.level == level]

    def num_blocks_at(self, level: int) -> int:
        return sum(1 for b in self._blocks if b.level == level)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def is_multiblock(self) -> bool:
        return any(self.num_blocks_at(k) > 1 for k in range(self._num_levels))

    def block_by_uid(self, uid: int):
        return self._blocks[uid]

    @property
    def _levels(self) -> list:
        """Flat list of Simulations, level-major. Compatibility shim."""
        return [b.sim for b in self._blocks]

    @property
    def _couplings(self) -> list:
        """Flat list of GridCouplings, level-major (root has none)."""
        return [b.coupling for b in self._blocks[1:]]

    def _one_block(self, level: int, what: str):
        """The single block at `level`, or a pointed error."""
        blocks = self.blocks_at(level)
        if not 0 <= level < self._num_levels:
            raise IndexError(
                f"Level {level} out of range [0, {self._num_levels - 1}]")
        if len(blocks) != 1:
            raise ValueError(
                f"{what}({level}) is ambiguous: level {level} hosts "
                f"{len(blocks)} blocks ({', '.join(b.name for b in blocks)}). "
                f"Use blocks_at({level}) / get_block({level}, j) and handle "
                f"each block explicitly — returning block 0 would silently "
                f"drop the rest.")
        return blocks[0]

    def get_block(self, level: int, index: int = 0):
        return self.blocks_at(level)[index]

    # =================================================================
    # Simulation-compatible interface
    # =================================================================

    def advance(self) -> None:
        """Perform one coarse timestep with nested fine stepping.

        After this call:
            - All levels are at time t + δt_0
            - Level 0's f has been updated with F→C feedback
            - self.rho, self.u reflect Level 0's macroscopic state
            - self.step_count is incremented

        Physical time advanced: δt_0 (one coarsest-level timestep).
        Total sub-steps performed: 2^(M-1) + 2^(M-2) + ... + 1 = 2^M - 1.

        Execution path (Phase 1c S1):
            - MLG_CUDA_GRAPH=1 + pure-LBM eligible → after a few eager warmup
              steps, the whole coarse-step kernel sequence is captured into one
              CUDA graph and every subsequent call is a single graph.launch().
              Algorithmically identical (same kernels, same order), so the
              per-step result is bit-identical to the eager path.
            - otherwise → the eager recursion (_advance_eager).
        """
        if not self._graph_should_use():
            self._advance_eager()
            return

        # ── Fast path: replay the captured whole-step graph ──────
        if self._graph is not None:
            self._graph.launch(stream=self._graph_stream)
            self._graph_stream.synchronize()
            self._graph_replay_bookkeeping()
            return

        # ── Warmup (eager) to allocate lazy buffers & JIT kernels ─
        if self._graph_warmup < self._GRAPH_WARMUP:
            self._advance_eager()
            self._graph_warmup += 1
            return

        # ── Capture the whole coarse-step on this call, then run it ─
        self._graph_capture_and_launch()

    # ── Esoteric bridge (Phase e2: REGION-scoped) ────────────────
    # Coupling only reads the coarse sub-volume / strided fine nodes and
    # only writes the 6 fine strips / the coarse excised block — so for
    # esoteric levels we gather/scatter exactly those regions (parity =
    # _esoteric_step, the next LOAD's view). No full-field temporaries.
    # Standard levels take the exact pre-existing coupling path.
    # Phase e1: f_prev stores ONLY the coarse sub-volume C2F actually
    # reads (identical values -> bit-preserving; ~14x smaller).

    @staticmethod
    def _scoped(coupling) -> bool:
        """3D coupling with the region-scoped API (2D coupling = legacy)."""
        return hasattr(coupling, 'coarse_sub_spatial_slices')

    def _f_prev_src(self, block) -> 'npt.NDArray':
        """Sub-volume of `block`'s PARENT physical f that block's C2F reads.

        Scoped by the block's own coupling, so siblings read disjoint windows
        of the same parent. Esoteric levels gather through the region kernel
        (parity = the next LOAD's view); 2D legacy couplings lack the scoped
        API and take the full field exactly as before.
        """
        lev = block.parent.sim
        cpl = block.coupling
        if not self._scoped(cpl):
            return lev.f
        sp = cpl.coarse_sub_spatial_slices
        if getattr(lev, '_use_esoteric', False):
            from src.kernels.esoteric_d3q27 import esoteric_gather_std_region
            return esoteric_gather_std_region(
                lev.xp, lev.f, lev._esoteric_step, sp,
                **_eso_wall_args(lev))
        return lev.f[(slice(None),) + sp]

    def _coupling_c2f(self, coupling, sim_coarse, sim_fine, *,
                      is_half_step: bool, f_prev) -> None:
        """C2F with per-level esoteric scoping (see class comment)."""
        if not self._scoped(coupling):     # 2D legacy path, unchanged
            coupling.coarse_to_fine(
                sim_coarse.f, sim_fine.f, is_half_step=is_half_step,
                f_coarse_prev=f_prev)
            return
        if getattr(sim_coarse, '_use_esoteric', False):
            from src.kernels.esoteric_d3q27 import esoteric_gather_std_region
            f_c = esoteric_gather_std_region(
                sim_coarse.xp, sim_coarse.f, sim_coarse._esoteric_step,
                coupling.coarse_sub_spatial_slices,
                **_eso_wall_args(sim_coarse))
            c_is_sub = True
        else:
            f_c = sim_coarse.f
            c_is_sub = False
        if getattr(sim_fine, '_use_surfel', False) \
                and getattr(getattr(sim_fine, 'obstacle_bc', None),
                            'partial_body', False) \
                and getattr(sim_coarse, '_use_surfel', False):
            # Partial-body fine level (patch 74): the C2F band crosses
            # the body, so interpolation stencils reach coarse DEAD
            # cells (surfel convention f = 0) — rho of the interpolant
            # collapses and the fine collide NaNs (measured: first NaN
            # exactly at the first post-C2F substep). Fill dead coarse
            # cells with the rest-state equilibrium (bounded, weights):
            # the polluted interpolants land only within the coupling
            # stencil reach of the wall, where the FINE facets impose
            # the physics. Whole-body fine levels never reach dead
            # coarse cells here (band guard geometry) — no-op there.
            xp = sim_coarse.xp
            if c_is_sub:
                sub = coupling.coarse_sub_spatial_slices
                dead_c = ~sim_coarse.obstacle_bc.d_live.reshape(
                    sim_coarse.domain_shape)[sub].astype(bool)
                w = _rest_feq_27(xp, f_c.dtype)
                f_c = xp.where(dead_c[None], w[:, None, None, None], f_c)
            else:
                dead_c = ~sim_coarse.obstacle_bc.d_live.reshape(
                    sim_coarse.domain_shape).astype(bool)
                w = _rest_feq_27(xp, f_c.dtype)
                f_c = xp.where(dead_c[None], w[:, None, None, None], f_c)
        if getattr(sim_fine, '_use_esoteric', False):
            from src.kernels.esoteric_d3q27 import esoteric_scatter_std_region
            strips: list = []
            coupling.coarse_to_fine(
                f_c, None, is_half_step=is_half_step, f_coarse_prev=f_prev,
                f_coarse_is_sub=c_is_sub, f_coarse_prev_is_sub=True,
                strips_out=strips)
            from src.grid.wall_coupling import coupling_skip_nt
            skip_f = coupling_skip_nt(sim_fine, 'c2f')
            for sp_sl, vals in strips:
                # skip_solid_nt: coupling writes carry FOREIGN values — at
                # solid cells they would land on live bounce-deposit slots
                # (patch 12 root cause; f1d proof). Under
                # mlg.wall_coupling.mode='exclude' the same argument also
                # carries the wall-adjacent fluid layer, whose populations
                # belong to the wall BC (src/grid/wall_coupling.py).
                esoteric_scatter_std_region(
                    sim_fine.xp, sim_fine.f, vals,
                    sim_fine._esoteric_step, sp_sl,
                    skip_solid_nt=skip_f,
                    **_eso_wall_args(sim_fine))
        else:
            from src.grid.wall_coupling import wall_skip_nt
            coupling.coarse_to_fine(
                f_c, sim_fine.f, is_half_step=is_half_step,
                f_coarse_prev=f_prev,
                f_coarse_is_sub=c_is_sub, f_coarse_prev_is_sub=True,
                skip_nt=wall_skip_nt(sim_fine, 'c2f'))

    def _coupling_f2c(self, coupling, sim_fine, sim_coarse) -> None:
        """F2C with per-level esoteric scoping (see class comment)."""
        if not self._scoped(coupling):     # 2D legacy path, unchanged
            coupling.fine_to_coarse(sim_fine.f, sim_coarse.f)
            return
        if getattr(sim_fine, '_use_esoteric', False):
            from src.kernels.esoteric_d3q27 import esoteric_gather_std_region
            f_f = esoteric_gather_std_region(
                sim_fine.xp, sim_fine.f, sim_fine._esoteric_step,
                coupling.fine_at_coarse_spatial_slices,
                **_eso_wall_args(sim_fine))
            f_is_at = True
        else:
            f_f = sim_fine.f
            f_is_at = False
        if getattr(sim_coarse, '_use_esoteric', False) \
                and getattr(sim_coarse, '_use_surfel', False):
            # Surfel coarse level under the eso residency bridge
            # (patch_notes/surfel/63): the patch-50 surfel F2C semantics
            # (finite-only fine feedback + dead re-zero) MUST win over the
            # hwbb-style skip_solid_nt branch below — the branch order
            # alone routed the bridge there, which scattered the
            # restriction's NaN (fine dead rho = 0) raw into the excised
            # block (first activated by the mlg2 bridge smoke, step 0).
            # Same patch-50 fix, phrased as read-modify-write on the
            # physical view (the coarse mem is eso-resident): writing
            # physical zeros at dead cells is well-defined — dead f = 0
            # IS the surfel convention (no bounce deposits to protect,
            # unlike the hwbb aliasing case).
            from src.kernels.esoteric_d3q27 import (
                esoteric_gather_std_region, esoteric_scatter_std_region)
            xp = sim_coarse.xp
            block = coupling.fine_to_coarse(
                f_f, None, f_fine_is_at_coarse=f_is_at,
                return_excised=True)
            sl = coupling.excised_spatial_slices
            cur = esoteric_gather_std_region(
                xp, sim_coarse.f, sim_coarse._esoteric_step, sl,
                **_eso_wall_args(sim_coarse))
            merged = xp.where(xp.isnan(block),
                              cur.astype(block.dtype, copy=False), block)
            live = sim_coarse.obstacle_bc.d_live.reshape(
                sim_coarse.domain_shape)[sl]
            merged = xp.where((live > 0)[None], merged,
                              merged.dtype.type(0.0))
            esoteric_scatter_std_region(
                xp, sim_coarse.f, merged.astype(cur.dtype, copy=False),
                sim_coarse._esoteric_step, sl,
                **_eso_wall_args(sim_coarse))
        elif getattr(sim_coarse, '_use_esoteric', False):
            from src.kernels.esoteric_d3q27 import esoteric_scatter_std_region
            block = coupling.fine_to_coarse(
                f_f, None, f_fine_is_at_coarse=f_is_at, return_excised=True)
            # skip_solid_nt: the excised region CONTAINS the body on coarse
            # levels; without the skip, the restriction's solid-cell values
            # overwrite the fluid neighbours' live deposits and the coarse
            # wall bounces garbage every cycle (patch 12 root cause).
            from src.grid.wall_coupling import coupling_skip_nt
            esoteric_scatter_std_region(
                sim_coarse.xp, sim_coarse.f, block,
                sim_coarse._esoteric_step, coupling.excised_spatial_slices,
                skip_solid_nt=coupling_skip_nt(sim_coarse, 'f2c'),
                **_eso_wall_args(sim_coarse))
        elif getattr(sim_coarse, '_use_surfel', False):
            # Surfel coarse level: the std restriction has no solid skip
            # and writes THROUGH the body. Fine dead cells hold rho = 0,
            # so the rescale is NaN wherever its stencil touches the fine
            # body — on coarse DEAD cells (harmless, advect re-zeroes)
            # but also on the coarse LIVE cut-cell shell whose fine
            # co-located neighborhood is dead (measured: stationary NaN
            # shell, patch 50 §4; same family as the esoteric
            # skip_solid_nt above — patch 12 root cause; hwbb was blind
            # to it only because its solid cells keep nonzero f).
            # Fix at the write: accept the fine feedback where it is
            # finite, keep the coarse level's OWN surfel-advanced state
            # where the restriction touched the body. Then re-zero dead
            # cells by assignment (mask-multiply would keep NaN).
            xp = sim_coarse.xp
            block = coupling.fine_to_coarse(
                f_f, None, f_fine_is_at_coarse=f_is_at,
                return_excised=True)
            view = sim_coarse.f[
                (slice(None),) + coupling.excised_spatial_slices]
            xp.copyto(view, block.astype(view.dtype, copy=False),
                      where=~xp.isnan(block))
            sim_coarse.obstacle_bc.zero_dead(sim_coarse.f)
        else:
            from src.grid.wall_coupling import wall_skip_nt
            coupling.fine_to_coarse(
                f_f, sim_coarse.f, f_fine_is_at_coarse=f_is_at,
                skip_nt=wall_skip_nt(sim_coarse, 'f2c'))

    def _advance_eager(self) -> None:
        """Eager (non-graph) coarse timestep: the nested-stepping recursion."""
        # ── Lazy initialization of f_prev buffers ────────────────
        # Deferred from __init__ because f is set by Initializer (Layer 2).
        # Level-major order, so for a chain the allocation SEQUENCE (sizes and
        # order) is byte-for-byte the old one — memory-pool layout and hence
        # CUDA-graph pointer stability are preserved.
        if not self._f_prev_initialized:
            for b in self._blocks:
                if not b.sim.is_ready:
                    raise RuntimeError(
                        f"Level {b.level} Simulation is not ready. "
                        f"Call set_distribution() on all levels first."
                    )
            for b in self._blocks:
                if b.parent is not None:
                    b.f_prev = b.sim.xp.ascontiguousarray(self._f_prev_src(b))
            self._f_prev_initialized = True

        root = self._root

        # ── Snapshot the root for each of its children ───────────
        # Taken BEFORE the root advances: this is the t-level state the
        # half-step C2F interpolates against.
        xp = root.sim.xp
        for ch in root.children:
            with _prof(self._pn_fprev[0]):
                xp.copyto(ch.f_prev, self._f_prev_src(ch))

        # ── Advance coarse level (full domain) ───────────────────
        with _prof(self._pn_adv[0]):
            root.sim.advance()
        _release_surfel_scratch(root.sim)

        # ── Recursively advance all finer blocks ─────────────────
        for ch in root.children:
            self._advance_block(ch)

        self._step_count += 1
        _prof_step()

    # =================================================================
    # Phase 1c — CUDA Graph capture / replay (S1: pure-LBM whole-step)
    # =================================================================

    def _graph_should_use(self) -> bool:
        """Whether this advance() should take the CUDA-graph path.

        Runs the (dynamic) precheck once and caches it. Returns False for any
        ineligible run so advance() transparently uses the eager recursion.
        """
        if not self._graph_enabled or self._graph_failed:
            return False
        if self._graph_ok is None:
            ok, reason = self._graph_precheck()
            self._graph_ok = ok
            if ok:
                print("[MLG] CUDA graph mode ON (pure-LBM whole coarse-step); "
                      f"capture after {self._GRAPH_WARMUP} eager warmup step(s).")
            else:
                print(f"[MLG] CUDA graph mode requested but ineligible: {reason}"
                      " — using eager path.")
        return self._graph_ok

    def _graph_precheck(self) -> Tuple[bool, str]:
        """Static eligibility of this MLG for whole-step graph capture (S1).

        All must hold or capture is unsafe:
          - cupy backend (graphs are a CUDA feature);
          - no host↔device sync inside the captured region — MLG_PROFILE/NVTX
            (per-section deviceSynchronize) and nan_trap (bool(xp.any(...)) .get)
            both break capture, so both must be off;
          - pure-LBM only: any level with an ALM model routes L4 through a
            CPU-in-loop BEM and reallocates rho/u each step (08 §1) → the whole
            coarse step is not a fixed pure-GPU sequence.
        """
        if self._root.sim.xp.__name__ != "cupy":
            return False, "numpy backend (CPU)"
        if "MLG_PROFILE" in os.environ or os.environ.get("MLG_NVTX") == "1":
            return False, "MLG_PROFILE/MLG_NVTX active (per-section sync)"
        for _b in self._blocks:
            k, lev = _b.label, _b.sim
            if getattr(lev, "al_model", None) is not None:
                return False, f"{k} has ALM (S1 is pure-LBM only; see S2)"
            if getattr(lev, "nan_trap_enabled", False):
                return False, f"{k} nan_trap enabled (host sync per step)"
            if getattr(lev, "_use_esoteric", False):
                return False, (f"{k} esoteric (parity alternates per step; "
                               "whole-step capture unsupported)")
            if getattr(lev, "_use_surfel", False):
                return False, (f"{k} surfel (facet force readback is a "
                               "host sync every apply — breaks capture)")
        return True, "ok"

    def _graph_capture_and_launch(self) -> None:
        """Capture one whole coarse-step into a CUDA graph, then execute it.

        Capture records the kernel sequence onto a non-blocking stream WITHOUT
        executing it; the counters bumped by the record run are therefore
        correct for exactly one step, and graph.launch() then applies the GPU
        work. Any failure permanently disables the graph path (eager fallback)
        after restoring counters and running one clean eager step.
        """
        import cupy as cp
        snap = (self._step_count, [lev.step_count for lev in self._levels])
        try:
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                self._advance_eager()          # records ops; not yet executed
                graph = stream.end_capture()
            graph.launch(stream=stream)         # apply the recorded coarse step
            stream.synchronize()
            self._graph = graph
            self._graph_stream = stream
            print("[MLG] CUDA graph captured — replaying whole coarse-step "
                  f"(warmup done at step {self._step_count}).")
        except Exception as e:                  # pragma: no cover (hardware path)
            # Record run bumped counters but did NOT advance GPU state (ops were
            # only recorded). Restore counters, then run one real eager step.
            self._step_count, level_counts = snap
            for lev, c in zip(self._levels, level_counts):
                lev.step_count = c
            try:
                stream.end_capture()            # best-effort: close aborted capture
            except Exception:
                pass
            self._graph = None
            self._graph_failed = True
            print(f"[MLG] CUDA graph capture FAILED ({type(e).__name__}: {e}); "
                  "falling back to eager for the rest of the run.")
            self._advance_eager()

    def _graph_replay_bookkeeping(self) -> None:
        """Advance Python-side counters after a graph replay.

        graph.launch() re-executes only the recorded GPU ops; the host-side
        step counters (incremented inside the record run, not the replay) must
        be advanced manually. Level k performs 2^k sub-steps per coarse step.
        """
        for b in self._blocks:
            b.sim.step_count += (1 << b.level)
        self._step_count += 1
        _prof_step()

    def _advance_fine(self, level_k: int) -> None:
        """Recursively advance level_k (two fine steps per coarse step).

        Algorithm (Lagrava Sec. 4.4.3, corrected ordering):
            For each fine step:
                1. fine.advance()  — collide + stream (boundary gets invalid data)
                2. C→F coupling    — fix boundary with coarse data (acts as BC)
                3. Recurse into finer levels (if any)
            After both fine steps:
                4. F→C coupling    — feed fine result back to coarse excised region

        C→F is applied AFTER advance (not before) because:
            - Streaming (pull) at fine boundary creates invalid incoming populations
            - C→F immediately fixes these = acts as boundary condition
            - Corrected boundary is ready for the NEXT step's collision
            - This preserves mass conservation across the grid interface

        Temporal interpolation for C→F:
            - After fine step #1 (at t+δt_f):   half-step interpolation
            - After fine step #2 (at t+δt_c):   full-step (no interpolation)

        Args:
            level_k: Fine level index (1, 2, ..., M-1).
        """
        self._advance_block(self._one_block(level_k, "_advance_fine"))

    def _advance_block(self, b) -> None:
        """Two sub-steps of block `b`, then ONE F→C into its parent.

        Identical algorithm to the old per-level version (see _advance_fine's
        docstring); the only change is that a parent may now have several
        children, so the single statements become one-iteration loops. With one
        child per level the emitted GPU call sequence is unchanged.

        Sibling ORDER is numerically irrelevant by construction: a sibling's
        only write into the parent is its own `excised` box and its only read
        is its own `fine_domain_coarse`, and the block-tree validator keeps
        those disjoint across siblings.
        """
        cpl, parent, lv = b.coupling, b.parent, b.level
        xp = b.sim.xp

        for is_half in (True, False):
            # Snapshot MY state for each of MY children, pre-advance
            for g in b.children:
                with _prof(self._pn_fprev[lv]):
                    xp.copyto(g.f_prev, self._f_prev_src(g))

            # Advance (collide + stream)
            with _prof(self._pn_adv[lv]):
                b.sim.advance()
            _release_surfel_scratch(b.sim)

            # C→F AFTER advance: it acts as the boundary condition.
            # half-step -> temporal interpolation against the parent snapshot;
            # full-step -> the parent's current state, no interpolation.
            with _prof(self._pn_c2f[lv]):
                self._coupling_c2f(cpl, parent.sim, b.sim,
                                   is_half_step=is_half,
                                   f_prev=(b.f_prev if is_half else None))

            for g in b.children:
                self._advance_block(g)

        # ═════════════════════════════════════════════════════════
        # F→C feedback: overwrite parent's excised region with fine data
        # ═════════════════════════════════════════════════════════
        with _prof(self._pn_f2c[lv]):
            self._coupling_f2c(cpl, b.sim, parent.sim)

    # =================================================================
    # Properties (delegate to Level 0 for Simulation compatibility)
    # =================================================================

    @property
    def rho(self) -> Optional['npt.NDArray']:
        """Density field from the coarsest level (Level 0).

        This is what OutputManager and VTK writers access.
        For multi-level VTK output (Phase E), each level's rho
        will be accessed separately via get_level().
        """
        return self._root.sim.rho

    @property
    def u(self) -> Optional['npt.NDArray']:
        """Velocity field from the coarsest level (Level 0)."""
        return self._root.sim.u

    @property
    def f(self) -> Optional['npt.NDArray']:
        """Distribution function from the coarsest level (Level 0)."""
        return self._root.sim.f

    @property
    def f_post(self) -> Optional['npt.NDArray']:
        """Post-collision distribution from Level 0."""
        return self._root.sim.f_post

    @property
    def body_force(self) -> Optional['npt.NDArray']:
        """Body force from Level 0 (for ALM compatibility)."""
        return self._root.sim.body_force

    @property
    def step_count(self) -> int:
        """Number of coarse steps completed."""
        return self._step_count

    @property
    def is_ready(self) -> bool:
        """Whether all levels have distributions set."""
        return all(b.sim.is_ready for b in self._blocks)

    @property
    def physical_f(self) -> Optional['npt.NDArray']:
        """Level-0 f in standard ordering/layout (checkpoint/IO safe)."""
        return self._root.sim.physical_f

    @property
    def tau(self) -> float:
        """Relaxation time of Level 0."""
        return self._root.sim.tau

    @property
    def domain_shape(self) -> Tuple[int, ...]:
        """Domain shape of Level 0."""
        return self._root.sim.domain_shape

    # =================================================================
    # Level access (for multi-level output, diagnostics)
    # =================================================================

    @property
    def num_levels(self) -> int:
        """Total number of grid levels."""
        return self._num_levels

    def get_level(self, k: int) -> 'Simulation':
        """Get the Simulation object for level k.

        Useful for multi-level VTK output where each level's
        fields need to be written separately.

        Args:
            k: Level index (0 = coarsest).

        Returns:
            Simulation object for that level.
        """
        return self._one_block(k, "get_level").sim

    def get_coupling(self, k: int) -> 'GridCoupling':
        """Get the GridCoupling for level pair (k, k+1).

        Args:
            k: Coarse level index in the pair.

        Returns:
            GridCoupling for the pair (k, k+1).
        """
        if not 0 <= k < self._num_levels - 1:
            raise IndexError(
                f"Coupling {k} out of range [0, {self._num_levels - 2}]"
            )
        return self._one_block(k + 1, "get_coupling").coupling

    # =================================================================
    # Diagnostics
    # =================================================================

    def summary(self) -> str:
        """Human-readable summary of the multi-level grid."""
        lines = [
            f"MultiLevelGrid: {self._num_levels} levels",
            f"  Coarse step count: {self._step_count}",
            f"  Sub-steps per coarse step: {2**self._num_levels - 1}",
            "",
        ]
        for b in self._blocks:
            lines.append(
                f"  {b.label}: shape={b.sim.domain_shape}, "
                f"tau={b.sim.tau:.4f}, "
                f"steps_per_coarse={2 ** b.level}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MultiLevelGrid(num_levels={self._num_levels}, "
            f"step_count={self._step_count})"
        )