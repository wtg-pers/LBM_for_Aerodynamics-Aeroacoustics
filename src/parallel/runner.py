"""SPMD distributed MLG runner (patch 17 M5; block tree since patch 18).

One process = one rank = one GPU. Each rank performs the FULL deterministic
production build (SimulationSetup + SolverInitializer — bit-identical state
on every rank), keeps only its wrap-sliced slab per BLOCK plus the rank's
ActuatorLineModel(s), frees the rest, then drives the gate-proven distributed
loop: per-block v1 halo exchange + rank-local MLG coupling + distributed ALM
(M3 partial-sum allreduce). The recursion below is the SPMD restructuring of
gates G-M2b/M3's run_distributed (loops over ranks removed — every rank
executes the same sequence in lockstep; collectives meet inside step()).

BLOCK indexing (patch 18): every per-grid list is indexed by BLOCK uid, not
by level. A level may host several refinement blocks, so "level k" no longer
names one grid. For a chain uid == level, and the emitted call sequence,
tags and buffers are unchanged — which is what keeps single-box cases
bit-identical.

A rank may own ZERO cells of a block: refinement blocks cover part of the
domain, so a contiguous SUBSET of ranks owns each one. An unowned block is
skipped WITH ITS WHOLE SUBTREE (children live inside their parent, so they
are unowned too) and no slab is ever allocated for it.

Memory note: the full-domain build peaks like a single-GPU run of the same
case (each rank has its own GPU, so this is admissible whenever the case
fits one GPU — D40 does). Cases beyond single-GPU size need a distributed
initializer (documented follow-up, not needed for the M5 validation).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import cupy as cp

from src.parallel.partition import (
    Partition1D, AXIS_NAME, derive_block_ranges)
from src.parallel.halo import HaloBandExchangerV1
from src.parallel.local_level import LocalLevel, extract_level
from src.parallel.mlg_coupling import RankLocalCouplingV1
from src.parallel.alm_dist import make_distributed_sampler
from src.kernels.esoteric_d3q27 import (
    esoteric_gather_std, esoteric_gather_std_region)


# ── MPI tag layout ───────────────────────────────────────────────────
# One namespace per BLOCK uid. MPI only guarantees tags up to 32767, so the
# bases are spaced to stay inside that on any implementation (1024 blocks per
# channel group). Before the block tree these were 16*level / 300+4*level,
# which would have collided once uid passed 18.
TAG_HALO = 0            # + 4*uid   : halo bands, one tag per side
TAG_FIELD = 4096        # + 8*uid   : VTK gathers (rho, u, nu_t)
TAG_CKPT = 12288        # + 4*uid   : checkpoint f gather
TAG_MACRO = 20480       # + 4*uid   : checkpoint/final rho+u gather
TAG_VERIFY = 24576      # + 4*uid   : --verify assembly

# Model state that every output channel reads. Broadcast from each block's
# owners to the rest at reporting cadence (see sync_alm_reporting): with a
# per-block sub-communicator only the owners run a rotor, so nobody else's
# copy advances on its own.
_ALM_REPORT_MODEL = ("_step_count", "_ramp_factor", "_last_positions",
                     "_last_bem_result", "_last_forces_global",
                     "_last_eps_samp", "_last_eps_spread")
_ALM_REPORT_ROTOR = ("theta", "time", "n_revolutions")


class _RunBlock:
    """Topology-only view of a GridBlock.

    The runner must NOT hold the built Simulations: the driver drops the whole
    replicated build right after construction, and a reference kept here would
    pin every level's obstacle links, collision objects and BC tables for the
    life of the run. Everything the loop needs from a block is its identity and
    its place in the tree.
    """

    __slots__ = ("uid", "level", "index", "name", "label", "parent",
                 "children")

    def __init__(self, b) -> None:
        self.uid = b.uid
        self.level = b.level
        self.index = b.index
        self.name = b.name
        self.label = b.label
        self.parent: Optional["_RunBlock"] = None
        self.children: List["_RunBlock"] = []

    def __repr__(self) -> str:
        return f"_RunBlock({self.label}, uid={self.uid})"


def _run_tree(blocks) -> List[_RunBlock]:
    """Level-major _RunBlock list mirroring `blocks` (uid == list index)."""
    nodes = [_RunBlock(b) for b in blocks]
    for b, n in zip(blocks, nodes):
        if b.parent is not None:
            n.parent = nodes[b.parent.uid]
            nodes[b.parent.uid].children.append(n)
    return nodes


class DistributedMLGRunner:
    """Drive ONE rank of a decomposed MLG run (transport = the wire)."""

    def __init__(self, mlg, transport, rank: int, n_ranks: int,
                 allreduce=None, axis: Optional[int] = None,
                 ghost: int = 3, comm=None,
                 cut_policy: str = "balanced") -> None:
        self.rank, self.nr = rank, n_ranks
        if cut_policy not in ("balanced", "aligned"):
            raise ValueError(f"cut_policy must be balanced|aligned, "
                             f"got {cut_policy!r}")
        self.cut_policy = cut_policy
        if n_ranks == 1:
            # Single rank needs NO halo at all: with ghost=0 the local array
            # IS the domain and the kernel's %N wrap IS the periodicity —
            # identical to the production single-GPU path. (The previous
            # ghost=3 self-exchange was correct but paid the MPI transport;
            # without mpirun/UCX the ob1 fallback made it ~7x slower.)
            ghost = 0
        self._comm = comm
        self.ghost = ghost

        # ── block tree ───────────────────────────────────────────
        # `src` is the built tree, used for geometry during construction only;
        # `self.blocks` is the topology-only mirror that survives the build.
        src = list(mlg.iter_blocks())                  # level-major
        for i, b in enumerate(src):
            if b.uid != i:
                raise ValueError(
                    f"block '{b.name}' has uid {b.uid} at flat position {i}; "
                    f"the runner indexes every per-grid list by uid")
        self.blocks = _run_tree(src)
        self.nb = len(src)
        NL = mlg.num_levels
        self.NL = NL
        self.multiblock = bool(getattr(mlg, "is_multiblock", False))
        shapes = [tuple(b.sim.domain_shape) for b in src]

        self.axis, self.bounds = self._decompose(src, shapes, axis,
                                                 n_ranks, ghost)
        bounds = self.bounds

        # ── per-block partitions ─────────────────────────────────
        # Child ranges are DERIVED from the parent BLOCK's range with the same
        # fine_range_from_coarse the coupling asserts on, so coupling regions
        # stay rank-local. Every rank derives the full ownership table (cuts
        # are replicated) and hands it to from_range, so neighbor() can skip
        # ranks that hold nothing of this block.
        table = derive_block_ranges(src, self.axis, bounds)
        self.parts: List[Partition1D] = []
        for uid, b in enumerate(src):
            own_start, own_count = table[uid][rank]
            if 0 < own_count < ghost:
                raise ValueError(
                    f"{b.label}: rank{rank} owns {own_count} cells < ghost "
                    f"{ghost} on axis {AXIS_NAME[self.axis]} — pick another "
                    f"axis or fewer ranks")
            self.parts.append(Partition1D.from_range(
                shapes[uid], n_ranks, rank, self.axis, own_start, own_count,
                ghost=ghost, all_counts=[c for _s, c in table[uid]]))
        self.owns: List[bool] = [p.own_count > 0 for p in self.parts]
        if not self.owns[0]:
            raise ValueError(
                f"rank{rank} owns no cells of the root grid — the L0 cuts "
                f"{bounds} are degenerate")

        # ── distributed ALM (M3 hooks), per block ────────────────
        # Wired BEFORE the slab loop so the models' GLOBAL-shape f64 F_grids
        # (2.5GB at 400M scale) are replaced and freed before slabs start
        # accumulating. setup places each rotor on the finest BLOCK that
        # contains its disk, so several blocks may each carry a model.
        # The sampler's partial-sum allreduce must run over exactly the ranks
        # that own the block: a rank that owns none has nothing to contribute
        # and cannot be made to arrive at the collective (it skips the whole
        # subtree), so a full-communicator reduction would hang. comm.Split is
        # itself collective, hence the loop runs over ALL ALM blocks on ALL
        # ranks, in uid order, before anyone drops out.
        self.alm: Dict[int, object] = {}          # owned blocks: stepped here
        self._alm_all: Dict[int, object] = {}     # every ALM block (reporting)
        self._alm_root: Dict[int, int] = {}       # lowest owning world rank
        self._alm_comms: List[object] = []
        for uid, b in enumerate(src):
            al = getattr(b.sim, "al_model", None)
            if al is None:
                continue
            if allreduce is None:
                raise ValueError("ALM case needs an allreduce adapter")
            self._alm_all[uid] = al
            counts = [c for _s, c in table[uid]]
            self._alm_root[uid] = next(r for r, c in enumerate(counts) if c)
            red = allreduce
            if comm is not None and n_ranks > 1:
                sub = comm.Split(1 if self.owns[uid] else 0, rank)
                self._alm_comms.append(sub)
                if self.owns[uid]:
                    from src.parallel.alm_dist import MPIAllreduce
                    red = MPIAllreduce(sub)
            if not self.owns[uid]:
                continue
            p = self.parts[uid]
            off = np.zeros(3)
            off[self.axis] = p.own_start - p.ghost
            loc = tuple(p.local_shape)
            # Bind EVERY rotor, not just the first: MultiRotorManager.step
            # superposes each model's OWN _F_grid, so every model needs its
            # own rank-local buffer, offset and sampler.
            #
            # Setting domain_shape and leaving the buffers None is the whole
            # binding: both are allocated lazily on the first step, and by
            # then they read the rank-local shape set here. Allocating them
            # eagerly (here or in the model's __init__) put a full-domain f64
            # grid per rotor on the card during the build — 6.8 GB for
            # octo8's 8 rotors, discarded immediately.
            for m in (getattr(al, "models", None) or [al]):
                m._global_domain_shape = shapes[uid]
                m.domain_shape = loc
                m._F_grid = None
                m._grid_offset = off
                m._velocity_sampler = make_distributed_sampler(
                    red, rank, p, cp)
            if hasattr(al, "models"):        # manager: rank-local accumulator
                al.domain_shape = loc
                al._F_total = None
            self.alm[uid] = al
            cp.get_default_memory_pool().free_all_blocks()
        # RANK-INVARIANT, deliberately: taken from the replicated tree, not
        # from what this rank happens to own. The output manager selects its
        # channel tier from this, and every tier reaches different
        # collectives — with the owned-only version, a rank owning no ALM
        # block picked the 'flow' tier and hung on its Allreduce while the
        # others sat in the ALM bcast (4 ranks, 2 rotor blocks, one sole
        # owner each).
        self.has_alm = bool(self._alm_all)
        # Back-compat handle: the single model of a single-ALM-block run.
        self.model = next(iter(self._alm_all.values())) \
            if len(self._alm_all) == 1 else None

        # restart: continue esoteric parity + step numbering from the
        # restored state. The initializer sets L0 step_count = start_step
        # = the number of coarse advances already done (0-based exclusive
        # step convention, unified in stage C8) — completed_step IS that
        # count. The old `max(0, sc-1)` belonged to the legacy 1-based
        # main_mpi labels; under the unified convention it desynced the
        # esoteric parity t0 for single-GPU-written checkpoints.
        sc = int(getattr(src[0].sim, "step_count", 0) or 0)
        self.completed_step = sc

        # ── local slabs ──────────────────────────────────────────
        # Extract block-by-block and RELEASE each source grid's device arrays
        # right after — keeps the transient peak at (t=0 build state + one
        # slab), shrinking per block (the f copies of the first version OOMed
        # 24GB at D40). Unowned blocks get no slab at all, but their source
        # arrays are freed just the same.
        self.lv: List[Optional[LocalLevel]] = []
        # nu_t presence, read off the REPLICATED build before the arrays are
        # freed: the VTK gather is collective, so whether a block has an
        # eddy-viscosity channel may not depend on who owns it.
        self.has_nut: List[bool] = []
        for uid, b in enumerate(src):
            lev = b.sim
            self.has_nut.append(
                str((getattr(lev, "_sgs_cfg", None) or {}).get("model", "off"))
                == "dyn_smag")
            if self.owns[uid]:
                ld = extract_level(lev)
                self.lv.append(LocalLevel(
                    ld, self.parts[uid],
                    t0=self.completed_step * (2 ** b.level)))
                del ld
            else:
                self.lv.append(None)
            for a in ("f", "f_post", "f_prev", "rho", "u",
                      "_eso_node_type", "_eso_bc_rho", "_eso_bc_ux",
                      "_eso_bc_uy", "_eso_bc_uz"):
                if hasattr(lev, a):
                    try:
                        setattr(lev, a, None)
                    except AttributeError:
                        pass                     # read-only property
            cp.get_default_memory_pool().free_all_blocks()

        self.n_pop = int(self.lv[0].mem.shape[0])      # 27 (D3Q27)

        self.ex: Optional[List[Optional[HaloBandExchangerV1]]] = None
        if n_ranks > 1:
            self.ex = [
                HaloBandExchangerV1(self.parts[uid], transport, cp,
                                    tag_base=TAG_HALO + 4 * uid)
                if self.owns[uid] else None
                for uid in range(self.nb)]
        self.rlc: List[Optional[RankLocalCouplingV1]] = [None] * self.nb
        for uid, b in enumerate(src):
            if b.parent is None or not self.owns[uid]:
                continue
            self.rlc[uid] = RankLocalCouplingV1(
                b.coupling, self.parts[b.parent.uid], self.parts[uid], cp)
        self._fprev: List[Optional[object]] = [None] * self.nb
        self.profile = None          # dict -> per-section seconds (opt-in)
        # halo scheduling state (backlog #5): ghosts_fresh -> skip idempotent
        # re-exchanges of unchanged mem (the A-syncs after a same-cycle
        # D-sync, ~30% of all rounds); posted -> band already in flight
        # (early post fired right after the last mutation, so the transfer
        # hides under the child's fprev+advance)
        self._fresh = [False] * self.nb
        self._posted = [False] * self.nb

        # solid-body diagnostics: finest OWNED block carrying SOLID nodes.
        # Reported as a uid; the driver MAX-allreduces it so the choice is
        # rank-invariant even when this rank's slab holds no solid cell.
        self.body_block: Optional[int] = None
        for uid in range(self.nb - 1, -1, -1):
            if self.owns[uid] and bool((self.lv[uid].nt == 1).any()):
                self.body_block = uid
                break

        # esoteric IBB (STL track S6): slab-filtered deposit-rewrite pass
        # per block, built from the replicated InterpolatedBounceBack links
        # (obstacle_bc survives the array free above; link arrays are tiny).
        # rewrite runs right after every L.advance().
        from src.boundary.interpolated_wall import InterpolatedBounceBack
        from src.kernels.esoteric_ibb_d3q27 import build_slab_ibb
        self.ibb: List[Optional[object]] = [None] * self.nb
        for uid, b in enumerate(src):
            if not self.owns[uid]:
                continue
            ob = getattr(b.sim, "obstacle_bc", None)
            if isinstance(ob, InterpolatedBounceBack):
                self.ibb[uid] = build_slab_ibb(cp, ob, self.parts[uid],
                                               shapes[uid])
                n_l = self.ibb[uid].n_links if self.ibb[uid] is not None else 0
                print(f"[mpi] eso IBB {b.label}: {n_l:,} slab links "
                      f"(rank {rank})", flush=True)

    # ── decomposition ────────────────────────────────────────────────
    def _decompose(self, src, shapes, axis, n_ranks, ghost):
        """(axis, L0 cut bounds). Chains keep the pre-tree code path verbatim
        so single-box cases decompose exactly as before.

        `keep_whole`: the 'aligned' cut policy keeps every ALM-carrying block
        inside one rank, which costs balance but removes that block's halo
        exchange and sampling allreduce entirely."""
        from src.parallel.partition import (
            choose_axis_balanced, balance_cuts,
            choose_axis_balanced_tree, balance_cuts_tree)
        if self.multiblock:
            keep = tuple(b.uid for b in src
                         if getattr(b.sim, "al_model", None) is not None) \
                if self.cut_policy == "aligned" else ()
            if axis is None:
                return choose_axis_balanced_tree(src, n_ranks, ghost, keep)
            axis = int(axis)
            return axis, balance_cuts_tree(src, axis, n_ranks, ghost, keep)
        boxes = []
        for b in src[1:]:
            fdc = b.region.fine_domain_coarse
            boxes.append(((fdc.x_start, fdc.x_end),
                          (fdc.y_start, fdc.y_end),
                          (fdc.z_start, fdc.z_end)))
        if axis is None:
            return choose_axis_balanced(shapes, boxes, n_ranks, ghost)
        axis = int(axis)
        return axis, balance_cuts(shapes, boxes, axis, n_ranks, ghost)

    # ── plumbing ─────────────────────────────────────────────────────
    def _tic(self):
        if self.profile is not None:
            cp.cuda.get_current_stream().synchronize()
            return time.perf_counter()
        return 0.0

    def _toc(self, key, t0):
        if self.profile is not None:
            cp.cuda.get_current_stream().synchronize()
            self.profile[key] = self.profile.get(key, 0.0) \
                + time.perf_counter() - t0

    def _touch(self, uid: int) -> None:
        assert not self._posted[uid], \
            f"{self.blocks[uid].label} mutated with a band in flight " \
            f"(scheduling bug)"
        self._fresh[uid] = False

    def _post(self, uid: int) -> None:
        """Early post: bands gathered from the FINAL state before the next
        consumer; MPITransport also pre-posts the Irecv (true overlap)."""
        if self.ex is None or self.ex[uid] is None \
                or self._fresh[uid] or self._posted[uid]:
            return
        t0 = self._tic()
        self.ex[uid].post(self.lv[uid].mem, self.lv[uid].t)
        self._posted[uid] = True
        self._toc("halo_post", t0)

    def _sync(self, uid: int) -> None:
        if self.ex is None or self.ex[uid] is None or self._fresh[uid]:
            return                    # NR=1: no halo / unchanged: idempotent
        if not self._posted[uid]:
            t0 = self._tic()
            self.ex[uid].post(self.lv[uid].mem, self.lv[uid].t)
            self._toc("halo_post", t0)
        t0 = self._tic()
        self.ex[uid].complete(self.lv[uid].mem, self.lv[uid].t)
        self._posted[uid] = False
        self._fresh[uid] = True
        self._toc("halo_complete", t0)

    def _save_fprev(self, child) -> None:
        """Snapshot the PARENT's pre-step state over `child`'s coarse
        sub-volume. It belongs to the child: with several children each needs
        its own window on the same parent."""
        t0 = self._tic()
        uid, puid = child.uid, child.parent.uid
        reg = self.rlc[uid].coarse_block_region_local()
        self._fprev[uid] = (esoteric_gather_std_region(
            cp, self.lv[puid].mem, self.lv[puid].t, reg) if reg else None)
        self._toc("fprev", t0)

    def _advance_level(self, uid: int) -> None:
        L = self.lv[uid]
        al = self.alm.get(uid)
        if al is not None:
            t0 = self._tic()
            L.macro_pre_pass()                      # rho/u for ALM sampling
            F = al.step(L.u, dt=1.0)                # collective inside
            self._toc("alm", t0)
            t0 = self._tic()
            L.advance(force=F.astype(cp.float32) if F.dtype != cp.float32
                      else F)
            self._toc("kernel", t0)
        else:
            t0 = self._tic()
            L.advance()
            self._toc("kernel", t0)
        if self.ibb[uid] is not None:
            # IBB deposit rewrite for the step the kernel just ran
            # (L.advance incremented t; parity = t - 1). Must precede any
            # halo post / coupling gather of this block's mem.
            t0 = self._tic()
            self.ibb[uid].rewrite(L.mem, L.t - 1)
            self._toc("ibb", t0)
        self._touch(uid)

    def _advance_block(self, b) -> None:
        """Two sub-steps of block `b`, then ONE F->C into its parent.

        NOTE sync cadence: ghosts must be fresh for (a) the fprev gather
        (block extends into ghosts), (b) the kernel, (c) the coupling
        reads. One sync covers fprev+advance back-to-back — the mem is
        unchanged between them (the first version synced twice; exchange
        of unchanged mem is idempotent, so dropping it is bit-neutral).

        The parent sync before C2F is unconditional. In a chain the second
        one is a no-op (nothing touched the parent in between, so _fresh
        short-circuits it) — but with SIBLING blocks the parent HAS been
        rewritten by the previous sibling's F2C, and that sync is what makes
        the read see it.

        ★ SCHEDULE vs OWNERSHIP. The halo exchange is a PAIRWISE collective,
        so every rank owning a block must reach the same exchanges in the
        same order. That schedule is therefore driven by the TREE — the
        recursion walks every child and the _sync/_touch bookkeeping on a
        block runs whenever this rank owns THAT block — while only the GPU
        work is driven by ownership of the block itself.

        Making the walk ownership-driven instead deadlocks, and only in
        asymmetric ownership: with parent P shared by ranks r0, r1 where r1
        owns a child of P and r0 owns none, r1's child contributes two extra
        touch/sync rounds on P that r0 never performs, and the two ranks
        wait on each other's bands forever. Symmetric layouts (each rank
        owning exactly one child) hide it completely — 2 and 4 ranks on the
        test rig both passed before this was found at 4 ranks with the
        'aligned' cut policy, where two ranks own no rotor block at all.

        Marking a block stale for a child this rank does not own costs one
        idempotent exchange of unchanged data; the peer that DOES own the
        child genuinely needs it.
        """
        uid, puid = b.uid, b.parent.uid
        mine, p_mine = self.owns[uid], self.owns[puid]
        for is_half in (True, False):
            if mine:
                self._sync(uid)
                for g in b.children:
                    if self.owns[g.uid]:
                        self._save_fprev(g)
                self._advance_level(uid)
            if p_mine:
                self._sync(puid)
            if mine:
                t0 = self._tic()
                self.rlc[uid].c2f(self.lv[puid].mem, self.lv[uid].mem,
                                  self.lv[puid].t, self.lv[uid].t,
                                  is_half_step=is_half,
                                  f_prev_sub_loc=(self._fprev[uid] if is_half
                                                  else None),
                                  nt_f=self.lv[uid].nt)
                self._toc("coupling", t0)
                self._touch(uid)      # c2f wrote our strips
            for g in b.children:
                self._post(uid)       # child completes at its B-sync while
                self._advance_block(g)  # ...its fprev+advance run
        if mine:
            self._sync(uid)
            t0 = self._tic()
            self.rlc[uid].f2c(self.lv[uid].mem, self.lv[puid].mem,
                              self.lv[uid].t, self.lv[puid].t,
                              nt_c=self.lv[puid].nt)
            self._toc("coupling", t0)
        if p_mine:
            self._touch(puid)         # f2c wrote the coarse excised rows

    # ── public API ───────────────────────────────────────────────────
    def step_coarse(self) -> None:
        """One L0 step (= 2^k substeps on level k), lockstep across ranks."""
        root = self.blocks[0]
        self._sync(0)
        for g in root.children:
            if self.owns[g.uid]:
                self._save_fprev(g)
        self._advance_level(0)
        for g in root.children:       # tree-driven, see _advance_block
            self._post(0)             # L0 bands ride under the child's advance
            self._advance_block(g)

    def run(self, n_coarse: int, log_every: int = 0, on_log=None) -> dict:
        t0 = time.perf_counter()
        t_last = t0
        self.last_interval = None      # {"s_per_step", "elapsed_s"} per log
        for s in range(1, n_coarse + 1):
            self.step_coarse()
            if log_every and on_log and s % log_every == 0:
                cp.cuda.runtime.deviceSynchronize()
                now = time.perf_counter()
                self.last_interval = {
                    "s_per_step": (now - t_last) / log_every,
                    "elapsed_s": now - t0,
                }
                t_last = now
                on_log(s, self)
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        return {"steps": n_coarse, "wall_s": dt, "s_per_step": dt / n_coarse}

    # ── reporting ────────────────────────────────────────────────────
    def sync_alm_reporting(self) -> None:
        """Give every rank the OWNERS' state for every ALM block.

        COLLECTIVE over the full communicator; call it before any channel
        that reads an ALM model (progress line, rotor CSV, marker VTP,
        finalize) and never per step.

        Why it is needed: with a per-block sub-communicator only the ranks
        owning a block run its rotors, so on every other rank that block's
        models are frozen at their build state. Output reads the models on
        rank 0, which owns some blocks and not others — without this the CSV
        and marker VTP would silently mix live and never-stepped rotors.
        Cheap: a few KB per rotor (marker-sized arrays), at log cadence.

        The listed attributes are the complete set the output path reads;
        loads and kinematics both, since a non-owner advances neither.
        """
        if self._comm is None or self.nr == 1 or not self._alm_all:
            return
        for uid in sorted(self._alm_all):
            al = self._alm_all[uid]
            root = self._alm_root[uid]
            models = getattr(al, "models", None) or [al]
            payload = None
            if self.rank == root:
                payload = [
                    ({a: getattr(m, a, None) for a in _ALM_REPORT_MODEL},
                     {a: getattr(m.rotor, a, None) for a in _ALM_REPORT_ROTOR})
                    for m in models]
            payload = self._comm.bcast(payload, root=root)
            if self.rank == root:
                continue
            for m, (ms, rs) in zip(models, payload):
                for a, v in ms.items():
                    setattr(m, a, v)
                for a, v in rs.items():
                    setattr(m.rotor, a, v)

    # ── diagnostics / assembly seams ─────────────────────────────────
    @property
    def body_level(self) -> Optional[int]:
        """Level of the body block (kept for the level-shaped consumers)."""
        return None if self.body_block is None \
            else self.blocks[self.body_block].level

    def blocks_at(self, level: int) -> List[int]:
        """uids of the blocks on `level`, config order."""
        return [b.uid for b in self.blocks if b.level == level]

    def mem_force_local(self):
        """OWNED-cell MEM force partial on the body block (lattice units of
        that block's level); allreduce + normalization happen in the caller.
        Diagnostic tier (atomicAdd accumulation)."""
        uid = self.body_block
        if uid is None or not self.owns[uid]:
            return np.zeros(3)        # this rank holds none of the body block
        if self.ibb[uid] is not None:
            # IBB: deposits are already REWRITTEN (Bouzidi values) — the
            # HWBB whole-domain kernel would read 2*val != f* + val. The
            # scatter pass accumulated the exact owned-clip MEM force of
            # the body block's most recent substep.
            return self.ibb[uid].last_force()
        from src.kernels.esoteric_d3q27 import eso_mem_force
        L = self.lv[uid]
        p = self.parts[uid]
        cb = [(0, d) for d in L.dims]
        cb[p.axis] = (p.ghost, p.ghost + p.own_count)
        return eso_mem_force(cp, L.mem, L.nt, L.t, cb)

    def owned_f_std_block(self, uid: int):
        """Owned slab of BLOCK uid in standard physical ordering (GPU), or
        None when this rank owns nothing of it."""
        if not self.owns[uid]:
            return None
        return esoteric_gather_std(cp, self.lv[uid].mem, self.lv[uid].t)[
            (slice(None),) + self.parts[uid].owned_local()]
