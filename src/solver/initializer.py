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
            sim.set_distribution(f)
            sim.step_count = start_step

        # ── Step 4: Conservation initialization ──────────────────
        if os.environ.get("LBM_DIST_INIT", "0") == "1":
            pass                        # no full fields exist (MPI runner)
        elif self._setup.conservation_mgr and self._setup.conservation_mgr.enabled:
            f_for_monitor = sim.f
            rho_init, _ = self._setup.macro.compute(f_for_monitor)
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
                n_markers = 0

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
        f = xp.asarray(state['f'])
        completed_step = state['step']
        start_step = completed_step + 1
        print(f"  Loaded step {completed_step}, resuming from step {start_step}")
        return f, start_step

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
        print(f"  Initial total mass: {float(xp.sum(f)):.6f}")
        return f, 0

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
            # Distributed init (patch 17 backlog #4): host metadata only,
            # no device fields. The uniform IC (rho=1, u=flow_vel const)
            # makes slab f a broadcast of ONE equilibrium vector — computed
            # by the runner per slab, bit-equal to the full elementwise
            # equilibrium (same per-cell math).
            for k in range(mlg.num_levels):
                level_sim = mlg.get_level(k)
                level_sim.init_esoteric_metadata_host()
                level_sim._dist_init_ic = (1.0, list(flow_vel)
                                           if isinstance(flow_vel,
                                                         (list, tuple))
                                           else [float(flow_vel), 0.0, 0.0])
                print(f"  Level {k}: dist-init metadata (host), "
                      f"tau={level_sim.tau:.6f}")
            return 0

        for k in range(mlg.num_levels):
            level_sim = mlg.get_level(k)
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
            mlg.get_level(k).f.size // setup.lattice.Q
            for k in range(mlg.num_levels)
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
        f0 = xp.asarray(state['f'])
        mlg.get_level(0).set_distribution(f0)
        mlg.get_level(0).step_count = start_step
        print(f"  Level 0: restored from checkpoint")

        # ── Restore fine levels ──────────────────────────────────
        for k in range(1, mlg.num_levels):
            level_sim = mlg.get_level(k)
            key = f'f_level_{k}'

            if key in state:
                f_k = xp.asarray(state[key])
                level_sim.set_distribution(f_k)
                print(f"  Level {k}: restored from checkpoint")
            else:
                # Not in checkpoint → initialize from equilibrium
                print(f"  Level {k}: not in checkpoint, init equilibrium")
                shape = level_sim.domain_shape
                dim = setup.lattice.dim
                dtype = setup.compute_dtype
                rho_0 = xp.ones(shape, dtype=dtype)
                u_0 = xp.zeros((dim,) + shape, dtype=dtype)

                physics_config = setup._physics_config
                flow_vel = physics_config.get('initial_flow_velocity',
                                              [0.0, 0.0, 0.0])
                if isinstance(flow_vel, (list, tuple)):
                    for d in range(min(len(flow_vel), dim)):
                        u_0[d] = flow_vel[d]
                else:
                    u_0[0] = flow_vel

                f_k = level_sim.collision.compute_equilibrium(rho_0, u_0)
                level_sim.set_distribution(f_k)

        # ── ALM: fast-forward rotor kinematics (restart bug fix) ──
        # theta/time/_step_count previously reset to 0 on restart, which
        # snapped the blades back to azimuth 0 AND re-applied the force
        # ramp. Replaying rotor.advance() reproduces the kinematic state
        # EXACTLY (same fp accumulation as the uninterrupted run).
        al = setup.al_model
        if al is not None:
            models = al.models if hasattr(al, 'models') else [al]
            sub = 2 ** (mlg.num_levels - 1)      # ALM lives on the finest
            # Checkpoint 'step' is the 0-based LABEL of the last processed
            # step -> coarse advances done = label + 1 = start_step. The
            # historical `completed_step * sub` was one coarse step short
            # for label-convention checkpoints (it was calibrated to the
            # pre-unification main_mpi count convention) — azimuth lagged
            # 2^(NL-1) fine steps on every single-GPU MLG ALM restart.
            t_fine = start_step * sub
            for m_ in models:
                for _ in range(t_fine):
                    m_.rotor.advance(1.0)
                m_._step_count = t_fine
            print(f"  ALM: rotor fast-forwarded {t_fine} fine steps "
                  f"(theta[0]={models[0].rotor.theta[0]:.4f} rad, "
                  f"ramp done={t_fine >= models[0].ramp_steps})")

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