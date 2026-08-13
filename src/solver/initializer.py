"""
Solver Initializer — Physical State Initialization

This module prepares the distribution function f for execution.

Modes:
    1. Fresh start (single grid):  f = f_eq(ρ₀, u₀)
    2. Fresh start (MLG):          f_k = f_eq(ρ₀, u₀) for each level k
    3. Restart (single grid):      f = checkpoint data
    4. Restart (MLG):              f_k = checkpoint data for each level k

Author: LBM Development Team
Date: 2026-03 (MLG checkpoint restart: 2026-04)
"""

import os
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    from src.solver.simulation import Simulation
    from src.solver.setup import SimulationSetup
    from src.grid.multi_level_grid import MultiLevelGrid


def _iter_grids(mlg):
    """(level, Simulation, block) for every grid, level-major.

    A level may hold several refinement blocks, so iterating levels and calling
    get_level(k) is no longer well defined — that accessor now refuses to guess.
    """
    if hasattr(mlg, 'iter_blocks'):
        return [(b.level, b.sim, b) for b in mlg.iter_blocks()]
    return [(k, mlg.get_level(k), None) for k in range(mlg.num_levels)]


class SolverInitializer:
    """Initialize solver physical state (distribution function f).

    Handles fresh start and checkpoint restart for both
    single-grid Simulation and multi-level MultiLevelGrid.
    """

    def __init__(self, setup: 'SimulationSetup') -> None:
        self._setup = setup
        self.xp: 'ModuleType' = setup.xp

    def initialize(self, sim: Any, args: Any) -> Tuple[int, int]:
        """Initialize distribution function and determine step range."""
        from src.grid.multi_level_grid import MultiLevelGrid
        is_mlg = isinstance(sim, MultiLevelGrid)

        should_restart = (
            (hasattr(args, 'restart_latest') and args.restart_latest)
            or (hasattr(args, 'restart') and args.restart)
        )

        # ── Step 1: Create or restore f ──────────────────────────
        if is_mlg:
            if should_restart:
                start_step = self._restart_mlg(sim, args)
            else:
                start_step = self._initialize_mlg(sim)
            self._check_mlg_wall_masks(sim)
        else:
            f, start_step = self._create_distribution(args)

        # ── Step 2: Determine end step ───────────────────────────
        end_step = self._determine_end_step(args, start_step)

        if start_step >= end_step:
            print(f"\n  ⚠️  start_step ({start_step}) >= end_step ({end_step})")
            print(f"      Use --extend N or --max-steps N to continue")
            return start_step, end_step

        total_steps = end_step - start_step
        print(f"  Steps to run: {total_steps} ({start_step} → {end_step - 1})")

        # ── Step 3: Wire f into Simulation ───────────────────────
        if not is_mlg:
            restored = getattr(self, '_single_restart_state', None)
            # from_checkpoint: the eso init must skip the IC seeders on a
            # restart (the scattered checkpoint f already holds the true
            # bounce/wall deposits — seeding would overwrite them; see
            # Simulation.set_distribution / eso_wall §4-3).
            sim.set_distribution(f, from_checkpoint=restored is not None)
            sim.step_count = start_step
            self._restore_wall_mail(sim, restored)

        # ── Step 4: Conservation initialization ──────────────────
        if os.environ.get("LBM_DIST_INIT", "0") == "1":
            pass                        # no full fields exist (MPI runner)
        elif self._setup.conservation_mgr and self._setup.conservation_mgr.enabled:
            # compute_density, NOT compute: the momentum half of compute() was
            # discarded here, and it is the expensive half. tensordot(c, f)
            # promotes f to c's float64, so the throwaway momentum cost an
            # astype of the WHOLE distribution — 6.94 GB at octo8's L0
            # (27 x 32.14M x 8 B), plus the momentum and velocity arrays, all
            # for a value the next line drops. It was the second wall the
            # octo8 MPI build hit. Bit-identical: compute() derives rho with
            # the same xp.sum(f, axis=0).
            rho_init = self._setup.macro.compute_density(sim.f)
            self._setup.conservation_mgr.initialize(rho_init, step=start_step)

        # ── Step 4b: Open CSV files (with start_step for restart) ─
        # (io_role='silent' MPI ranks never open result CSVs; their
        # perf_csv_path/blade_csv_dir are already None from setup)
        if (self._setup.force_mgr is not None
                and getattr(self._setup, 'is_io_rank', True)):
            self._setup.force_mgr.open_csv(start_step=start_step)

        # Rotor CSV: open with start_step awareness
        if self._setup.perf_csv_path is not None:
            path = self._setup.perf_csv_path
            header = self._setup._perf_csv_header
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if start_step > 0 and os.path.exists(path):
                # Restart: keep rows < start_step
                kept = []
                with open(path, 'r') as f:
                    _ = f.readline()  # skip old header
                    for line in f:
                        if line.strip():
                            step_val = int(line.split(',')[0])
                            if step_val < start_step:
                                kept.append(line)
                with open(path, 'w') as f:
                    f.write(header)
                    f.writelines(kept)
                print(f"  Rotor CSV: {path} (kept {len(kept)} rows, "
                      f"appending from step {start_step})")
            else:
                # Fresh start: write header only
                with open(path, 'w') as f:
                    f.write(header)
                print(f"  Rotor CSV: {path}")

        # Blade diagnostics CSV (per-marker files)
        if self._setup.blade_csv_dir is not None:
            blade_dir = self._setup.blade_csv_dir
            header = self._setup._blade_csv_header
            os.makedirs(blade_dir, exist_ok=True)

            # Determine n_markers from ALM
            al = self._setup.al_model
            if hasattr(al, 'rotor'):
                n_markers = al.rotor.markers_per_blade
            elif hasattr(al, 'models'):
                n_markers = al.models[0].rotor.markers_per_blade
            else:
                # n_markers = 0 here used to silently skip every blade CSV.
                raise TypeError(
                    f"ALM model {type(al).__name__} exposes neither .rotor "
                    "nor .models — cannot size the blade CSV files")

            for j in range(n_markers):
                path = os.path.join(blade_dir, f'{j}.csv')
                if start_step > 0 and os.path.exists(path):
                    kept = []
                    with open(path, 'r') as f:
                        _ = f.readline()
                        for line in f:
                            if line.strip():
                                step_val = int(line.split(',')[0])
                                if step_val < start_step:
                                    kept.append(line)
                    with open(path, 'w') as f:
                        f.write(header)
                        f.writelines(kept)
                else:
                    with open(path, 'w') as f:
                        f.write(header)
            print(f"  Blade CSV: {blade_dir}/ ({n_markers} marker files)")

        # Blade geometry snapshot (actual interpolated values, written once)
        if self._setup.al_model is not None and self._setup.blade_csv_dir is not None:
            al = self._setup.al_model
            models = (
                [al.models[i] for i in range(al.n_rotors)]
                if hasattr(al, 'models')
                else [al]
            )
            for model in models:
                rotor = model.rotor
                blade = rotor.blades[0]
                geo_path = os.path.join(
                    self._setup.blade_csv_dir, 'blade_geometry.csv',
                )
                with open(geo_path, 'w') as f:
                    f.write('marker,r_R,r_lu,chord_lu,twist_deg,epsilon_lu,active\n')
                    for j in range(blade.n_markers):
                        f.write(
                            f"{j},"
                            f"{blade.marker_r[j] / rotor.radius:.4f},"
                            f"{blade.marker_r[j]:.4f},"
                            f"{blade.marker_chord[j]:.4f},"
                            f"{blade.marker_twist[j]:.3f},"
                            f"{blade.marker_epsilon[j]:.4f},"
                            f"{blade.marker_active[j]}\n"
                        )
                print(f"  Blade geometry: {geo_path}")

        # ── Step 5: Info ─────────────────────────────────────────
        if hasattr(sim, 'print_info'):
            sim.print_info()

        return start_step, end_step

    # =====================================================================
    # Single-grid
    # =====================================================================

    def _create_distribution(self, args: Any) -> Tuple[Any, int]:
        setup = self._setup
        xp = self.xp
        if args.restart_latest:
            return self._restart_latest(xp, setup)
        if args.restart:
            return self._restart_from(args.restart, xp, setup)
        return self._fresh_start(xp, setup)

    def _restart_latest(self, xp, setup) -> Tuple[Any, int]:
        print(f"\n[5] Restarting from latest checkpoint...")
        if setup.checkpoint_mgr is None:
            raise RuntimeError("Cannot restart: checkpoints are disabled")
        setup.checkpoint_mgr.print_available()
        state = setup.checkpoint_mgr.load_latest()
        self._single_restart_state = state
        f = xp.asarray(state['f'])
        completed_step = state['step']
        start_step = completed_step + 1
        print(f"  Loaded step {completed_step}, resuming from step {start_step}")
        return f, start_step

    def _restart_from(self, path, xp, setup) -> Tuple[Any, int]:
        print(f"\n[5] Restarting from: {path}")
        if setup.checkpoint_mgr is None:
            from src.io.checkpoint import CheckpointManager
            setup.checkpoint_mgr = CheckpointManager(
                output_dir=setup.checkpoint_dir, xp=xp)
        state = setup.checkpoint_mgr.load(path)
        self._single_restart_state = state
        f = xp.asarray(state['f'])
        completed_step = state['step']
        start_step = completed_step + 1
        print(f"  Loaded step {completed_step}, resuming from step {start_step}")
        return f, start_step

    def _check_mlg_wall_masks(self, mlg) -> None:
        """eso implicit walls on MLG: ROOT level only (eso_wall patch 04).

        A fine sim with a wall mask means one of its faces sits FLUSH on
        an implicit wall face (flush-face BC inheritance converted it).
        Its coupling band on that axis would need the wrap slots the
        de-periodization severs — the open-face mailbox generalization
        is pending. A hard error beats a silently EQ-degraded fine wall
        under a real L0 wall (mixed semantics within one hierarchy).
        octo8 keeps fine bottoms overlap_width off the ground by design
        (l1_zmin), so only L0 carries the wall there.
        """
        if hasattr(mlg, 'iter_blocks'):
            fine = [(b.level, getattr(b, 'label', f'L{b.level}'), b.sim)
                    for b in mlg.iter_blocks() if b.level > 0]
        else:
            fine = [(k, f'L{k}', mlg.get_level(k))
                    for k in range(1, mlg.num_levels)]
        for lvl, label, s in fine:
            if getattr(s, '_eso_wall_mask', 0):
                raise NotImplementedError(
                    f"eso domain wall on fine level {lvl} ({label}): a "
                    "fine block flush on an implicit wall face is not "
                    "supported — keep the fine region off the wall so "
                    "the wall stays on L0 (octo8: l1_zmin = "
                    "overlap_width). patch_notes/eso_wall/PLAN.md §4-5")

    def _restore_wall_mail(self, sim, state) -> None:
        """eso domain-wall mailbox restore + series-consistency guard.

        The mailbox is checkpoint state OUTSIDE the 27N f slots
        (extra_wall_mail_L0, eso_wall §4-3). Silent mismatches here are
        the restart-units-switch class of bug (a2f6abe): the wall
        semantics changing mid-series must be a hard error, never a
        quiet EQ/implicit mix.
        """
        mask = getattr(sim, '_eso_wall_mask', 0)
        if state is None:
            return
        mail = state.get('wall_mail_L0')
        if mask and mail is None:
            raise RuntimeError(
                "restart: this config has eso implicit domain walls but "
                "the checkpoint carries no wall_mail_L0 key — it was "
                "written by the EQ-degradation era (or a std run). A "
                "series cannot switch wall semantics mid-way; start a "
                "fresh series (patch_notes/eso_wall/PLAN.md §2-4)")
        if not mask and mail is not None:
            raise RuntimeError(
                "restart: the checkpoint carries an eso wall mailbox "
                "(wall_mail_L0) but this config builds no implicit wall "
                "faces — wall semantics changed mid-series; start a "
                "fresh series (patch_notes/eso_wall/PLAN.md §2-4)")
        if not mask:
            return
        ck_mask = int(state.get('wall_mask_L0', 0))
        if ck_mask and ck_mask != mask:
            raise RuntimeError(
                f"restart: wall faces changed mid-series (checkpoint "
                f"mask {ck_mask:#04x} vs config {mask:#04x})")
        if getattr(sim, '_eso_wall_mail', None) is None:
            # dist-init metadata path: no device allocation exists —
            # validate against the layout and STASH the host array;
            # extract_level hands it to LocalLevel, which wrap-slices
            # its slab (eso_wall §4-5b).
            import numpy as _np
            from src.kernels.esoteric_d3q27 import eso_wall_mail_layout
            _, tot = eso_wall_mail_layout(mask, tuple(sim.domain_shape))
            arr = _np.asarray(mail, dtype=_np.float32).ravel()
            if arr.size != tot:
                raise RuntimeError(
                    f"restart: wall mailbox size mismatch (checkpoint "
                    f"{arr.size} vs layout {tot}) — grid or wall "
                    "layout changed mid-series")
            sim._dist_restart_wall_mail = arr
            print(f"  [esoteric] wall mailbox staged for slab restore "
                  f"({arr.size * 4 / 1e6:.1f} MB)")
            return
        arr = sim.xp.asarray(mail, dtype=sim.xp.float32).ravel()
        if arr.shape != sim._eso_wall_mail.shape:
            raise RuntimeError(
                f"restart: wall mailbox size mismatch (checkpoint "
                f"{arr.shape} vs config {sim._eso_wall_mail.shape}) — "
                "grid or wall layout changed mid-series")
        sim._eso_wall_mail[:] = arr
        print(f"  [esoteric] wall mailbox restored "
              f"({arr.size * 4 / 1e6:.1f} MB)")

    def _fresh_start(self, xp, setup) -> Tuple[Any, int]:
        print(f"\n[5] Initializing Flow Field (Fresh Start)...")
        # `initial_flow_velocity` is injected by setup._extract_physics
        # from `physics.U_inf` * `physics.flow_direction`.
        physics_config = setup._physics_config
        flow_vel = physics_config.get('initial_flow_velocity', 0.0)
        dtype = setup.compute_dtype

        if setup.lattice.dim == 2:
            rho0 = xp.ones((setup.Nx, setup.Ny), dtype=dtype)
            u0 = xp.zeros((2, setup.Nx, setup.Ny), dtype=dtype)
        else:
            rho0 = xp.ones((setup.Nx, setup.Ny, setup.Nz), dtype=dtype)
            u0 = xp.zeros((3, setup.Nx, setup.Ny, setup.Nz), dtype=dtype)

        if isinstance(flow_vel, (list, tuple)):
            for d in range(min(len(flow_vel), setup.lattice.dim)):
                u0[d] = flow_vel[d]
            print(f"  Initial velocity: {flow_vel} [Δx/Δt]")
        else:
            u0[0] = flow_vel
            print(f"  Initial velocity: {flow_vel} [Δx/Δt] (x-dir)")

        f = setup.collision.compute_equilibrium(rho0, u0)
        self._apply_perturbation(f, level=None)
        print(f"  Initial total mass: {float(xp.sum(f)):.6f}")
        return f, 0

    def _apply_perturbation(self, f, level, block=None) -> None:
        """Seed-discrimination arm (patch 58): solenoidal IC perturbation.

        Config block `initial_perturbation` (top level; absent/disabled =
        bit-identical init). Fresh starts only — restarts keep their
        checkpointed state. Defined in GLOBAL L0-lu coordinates, so every
        MLG level evaluates the same physical field in its own frame.
        """
        setup = self._setup
        pcfg = setup.config.get('initial_perturbation', {})
        if not (isinstance(pcfg, dict) and pcfg.get('enabled', False)):
            return
        if setup.lattice.dim != 3:
            raise NotImplementedError("initial_perturbation is 3D-only")
        from src.utilities.initial_perturbation import apply_to_level
        cfg = dict(pcfg)
        cfg.setdefault('span_z_lu', float(setup.Nz))
        cfg.setdefault('taper_lu', 2.0)
        if level is None or level == 0:
            origin, dx = (0.0, 0.0, 0.0), 1.0
        else:
            origin = tuple(float(o)
                           for o in setup._mlg_level_origins[level])
            dx = float(setup._mlg_scaler.get_level_units(level).dx)
        flow = setup._physics_config.get('initial_flow_velocity',
                                         [0.0, 0.0, 0.0])
        n = apply_to_level(self.xp, f, setup.collision, cfg,
                          origin=origin, dx=dx, rho0=1.0, u0=list(flow))
        tag = 'L0' if level in (None, 0) else f'L{level}'
        print(f"  [perturbation] {tag}: {n:,} cells seeded "
              f"(sigma_u={cfg['sigma_u']:g}, seed={cfg['seed']}, "
              f"modes={cfg['n_modes']})")

    # =====================================================================
    # Multi-Level Grid
    # =====================================================================

    def _initialize_mlg(self, mlg: 'MultiLevelGrid') -> int:
        """Fresh start: all levels get f = f_eq(ρ₀, u₀)."""
        setup = self._setup
        xp = self.xp
        physics_config = setup._physics_config
        flow_vel = physics_config.get('initial_flow_velocity', [0.0, 0.0, 0.0])
        dim = setup.lattice.dim

        print(f"\n[5] Initializing MultiLevelGrid ({mlg.num_levels} levels)")

        dtype = setup.compute_dtype

        if os.environ.get("LBM_DIST_INIT", "0") == "1":
            _p = setup.config.get('initial_perturbation', {})
            if isinstance(_p, dict) and _p.get('enabled', False):
                # dist-init broadcasts ONE equilibrium vector per slab —
                # a spatial perturbation cannot ride it; failing loudly
                # beats silently un-seeded science (patch 58).
                raise NotImplementedError(
                    "initial_perturbation + LBM_DIST_INIT is unsupported "
                    "(dist-init broadcasts a uniform IC)")
            # Distributed init (patch 17 backlog #4): host metadata only,
            # no device fields. The uniform IC (rho=1, u=flow_vel const)
            # makes slab f a broadcast of ONE equilibrium vector — computed
            # by the runner per slab, bit-equal to the full elementwise
            # equilibrium (same per-cell math).
            for k, level_sim, _blk in _iter_grids(mlg):
                level_sim.init_esoteric_metadata_host()
                level_sim._dist_init_ic = (1.0, list(flow_vel)
                                           if isinstance(flow_vel,
                                                         (list, tuple))
                                           else [float(flow_vel), 0.0, 0.0])
                print(f"  Level {k}: dist-init metadata (host), "
                      f"tau={level_sim.tau:.6f}")
            return 0

        for k, level_sim, _blk in _iter_grids(mlg):
            shape = level_sim.domain_shape

            rho_0 = xp.ones(shape, dtype=dtype)
            u_0 = xp.zeros((dim,) + shape, dtype=dtype)
            if isinstance(flow_vel, (list, tuple)):
                for d in range(min(len(flow_vel), dim)):
                    u_0[d] = flow_vel[d]
            else:
                u_0[0] = flow_vel

            f_k = self._equilibrium_lowmem(level_sim, rho_0, u_0, dtype)
            del rho_0, u_0
            self._apply_perturbation(f_k, k, block=_blk)
            level_sim.set_distribution(f_k)
            mem_mb = f_k.nbytes / (1024 * 1024)
            del f_k
            # Return this level's init transients to the driver so the pool
            # high-water stays ~(live + one level's temps). Without this, a
            # 91.6M-cell 5-level build peaked at 52.6 GB pool (would OOM on
            # a native-Linux 24GB card; WSL2 oversubscription masked it).
            if xp.__name__ == 'cupy':
                import cupy as _cp
                _cp.get_default_memory_pool().free_all_blocks()

            print(f"  Level {k}: shape={shape}, τ={level_sim.tau:.6f}, "
                  f"f size={mem_mb:.1f} MB")

        total_nodes = sum(
            sim_k.f.size // setup.lattice.Q
            for _, sim_k, _ in _iter_grids(mlg)
        )
        print(f"  Total nodes across all levels: {total_nodes:,}")
        return 0

    @staticmethod
    def _equilibrium_lowmem(level_sim, rho_0, u_0, dtype,
                            max_chunk_nodes: int = 4_000_000):
        """f_eq via compute_equilibrium in x-slabs (BIT-IDENTICAL).

        compute_equilibrium is pointwise, so slab-chunking along x changes
        nothing numerically while capping the (Q, N)-sized broadcasting
        temporaries at (Q, chunk). A monolithic call on a 26M-node level
        transiently needs ~4x f-size (>11 GB at D40-L4); chunked it is
        ~4 x (Q x chunk x 4B) ~= 1.7 GB. Small levels take one slab.
        """
        xp = level_sim.xp
        shape = rho_0.shape                       # (Nx, Ny, Nz) or (Nx, Ny)
        n_per_x = 1
        for d in shape[1:]:
            n_per_x *= int(d)
        step = max(1, max_chunk_nodes // max(n_per_x, 1))
        if step >= shape[0]:
            return level_sim.collision.compute_equilibrium(rho_0, u_0)
        # First slab also tells us Q (no reliance on collision internals).
        first = level_sim.collision.compute_equilibrium(
            rho_0[:step], u_0[:, :step])
        f = xp.empty((first.shape[0],) + tuple(shape), dtype=dtype)
        f[:, :step] = first
        del first
        for x0 in range(step, shape[0], step):
            sl = slice(x0, min(x0 + step, shape[0]))
            f[:, sl] = level_sim.collision.compute_equilibrium(
                rho_0[sl], u_0[:, sl])
        return f

    def _restart_mlg(self, mlg: 'MultiLevelGrid', args: Any) -> int:
        """Restart MLG from checkpoint.

        Checkpoint stores:
            'f' = Level 0 distribution
            'f_level_1', 'f_level_2', ... = fine level distributions
            'num_levels' = number of levels saved

        If checkpoint has fewer levels than current config, missing
        levels are initialized with equilibrium.
        """
        setup = self._setup
        xp = self.xp
        # LBM_DIST_INIT: the MPI runner keeps only its slab, so the restored
        # field must NOT be uploaded whole to every rank's device — that
        # upload is the only reason --dist-init and restart used to be
        # mutually exclusive. Hand the HOST array to the level instead and
        # let extract_level wrap-slice this rank's part out of it (host RAM
        # holds the full field; the device never sees more than a slab).
        dist = os.environ.get("LBM_DIST_INIT", "0") == "1"

        def _restore(level_sim, arr_host, tag):
            if dist:
                level_sim.init_esoteric_metadata_host()
                level_sim._dist_restart_f = arr_host
            else:
                # from_checkpoint: skip the IC seeders — the scattered
                # checkpoint f already holds the true bounce/wall
                # deposits (same latent restore-seeding bug class as the
                # single-grid path; eso_wall patch 02/04).
                level_sim.set_distribution(xp.asarray(arr_host),
                                           from_checkpoint=True)
            print(f"  {tag}: restored from checkpoint"
                  + (" (slab-scoped)" if dist else ""))

        if setup.checkpoint_mgr is None:
            raise RuntimeError("Cannot restart: checkpoints are disabled")

        print(f"\n[5] Restarting MultiLevelGrid from checkpoint...")
        setup.checkpoint_mgr.print_available()

        # ── Load checkpoint ──────────────────────────────────────
        if hasattr(args, 'restart') and args.restart:
            state = setup.checkpoint_mgr.load(args.restart)
        else:
            state = setup.checkpoint_mgr.load_latest()

        completed_step = state['step']
        start_step = completed_step + 1
        saved_num_levels = int(state.get('num_levels', 1))

        print(f"  Loaded step {completed_step}, "
              f"saved levels: {saved_num_levels}, "
              f"current levels: {mlg.num_levels}")

        # ── Restore Level 0 ──────────────────────────────────────
        _l0 = mlg.get_level(0)
        _restore(_l0, state['f'], "Level 0")
        _l0.step_count = start_step
        # eso implicit-wall mailbox (extra_wall_mail_L0) + series-
        # consistency guard — root level only (fine walls rejected by
        # _check_mlg_wall_masks). Replicated: device inject; dist-init:
        # host stash for the slab restore (both inside the helper).
        self._restore_wall_mail(_l0, state)

        # ── Restore fine levels ──────────────────────────────────
        _blks = list(mlg.iter_blocks()) if hasattr(mlg, 'iter_blocks') else None
        if _blks is not None:
            _per = {}
            for _b in _blks:
                _per[_b.level] = _per.get(_b.level, 0) + 1
            _targets = [(b.level, b.sim,
                         f"f_level_{b.level}"
                         + ("" if _per[b.level] <= 1 else f"_b{b.index}"))
                        for b in _blks if b.level > 0]
        else:
            _targets = [(k, mlg.get_level(k), f'f_level_{k}')
                        for k in range(1, mlg.num_levels)]

        for k, level_sim, key in _targets:
            if key not in state:
                # A missing level key means the checkpoint's block layout
                # differs from this config's (the '_b<index>' suffix only
                # appears when a level has MULTIPLE blocks), or the level
                # count changed. Continuing used to reset the level to
                # uniform equilibrium and keep running — a silently wrong
                # flow field. Refuse instead.
                _avail = sorted(str(_k) for _k in state.keys()
                                if str(_k).startswith('f_level_'))
                raise ValueError(
                    f"Restart: checkpoint has no key '{key}' for level {k} "
                    f"(fine-level keys present: {_avail or 'none'}). "
                    "Restart with the same block layout / level count the "
                    "checkpoint was written with.")
            _restore(level_sim, state[key], f"Level {k}")

        # ── ALM: fast-forward rotor kinematics (restart bug fix) ──
        # theta/time/_step_count previously reset to 0 on restart, which
        # snapped the blades back to azimuth 0 AND re-applied the force
        # ramp. Replaying rotor.advance() reproduces the kinematic state
        # EXACTLY (same fp accumulation as the uninterrupted run).
        al = setup.al_model
        if al is not None:
            # Sub-steps per coarse step = 2^level of the block each rotor
            # actually sits on. Walking BLOCKS rather than levels is what makes
            # this right in two ways: hardcoding num_levels-1 desynced the
            # azimuth by 2^(NL-1-alm_lev) fine steps when a rotor landed on an
            # intermediate level, and reading it back with get_level(k) raised
            # outright once a level hosted several blocks. Rotors on blocks of
            # DIFFERENT levels now each get their own rate — one shared
            # alm_lev would have silently desynced all but one of them.
            pairs = []                       # (model, level)
            if hasattr(mlg, 'iter_blocks'):
                for _b in mlg.iter_blocks():
                    _m = getattr(_b.sim, 'al_model', None)
                    if _m is None:
                        continue
                    for _mm in (getattr(_m, 'models', None) or [_m]):
                        pairs.append((_mm, _b.level))
            if not pairs:
                pairs = [(m, 0) for m in
                         (al.models if hasattr(al, 'models') else [al])]
            # Checkpoint 'step' is the 0-based LABEL of the last processed
            # step -> coarse advances done = label + 1 = start_step. The
            # historical `completed_step * sub` was one coarse step short
            # for label-convention checkpoints (it was calibrated to the
            # pre-unification main_mpi count convention) — azimuth lagged
            # 2^(NL-1) fine steps on every single-GPU MLG ALM restart.
            for m_, lev_ in pairs:
                t_fine = start_step * (2 ** lev_)
                for _ in range(t_fine):
                    m_.rotor.advance(1.0)
                m_._step_count = t_fine
            m0, lev0 = pairs[0]
            _levs = sorted({lv for _, lv in pairs})
            print(f"  ALM: {len(pairs)} rotor(s) fast-forwarded "
                  f"{start_step * (2 ** lev0)} fine steps @L{lev0}"
                  + (f" (levels present: {_levs})" if len(_levs) > 1 else "")
                  + f" (theta[0]={m0.rotor.theta[0]:.4f} rad, "
                  f"ramp done={start_step * (2 ** lev0) >= m0.ramp_steps})")

        print(f"  Resuming from step {start_step}")
        return start_step

    # =====================================================================
    # Utilities
    # =====================================================================

    def _determine_end_step(self, args: Any, start_step: int) -> int:
        if args.max_steps is not None:
            end_step = args.max_steps
            print(f"\n  End step (--max-steps): {end_step}")
        elif args.extend is not None:
            end_step = start_step + args.extend
            print(f"\n  Extending by {args.extend} steps: "
                  f"{start_step} → {end_step}")
        else:
            end_step = self._setup.config_max_steps
            print(f"\n  End step (from config): {end_step}")
        return end_step