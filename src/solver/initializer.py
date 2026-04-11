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
            # On restart with Esoteric, f is already in Esoteric layout
            if should_restart and hasattr(self, '_last_checkpoint_state'):
                eso_step = self._last_checkpoint_state.get('esoteric_step', None)
                if eso_step is not None:
                    sim._esoteric_f_already_set = True
            sim.set_distribution(f)
            sim.step_count = start_step

        # ── Step 4: Conservation initialization ──────────────────
        if self._setup.conservation_mgr and self._setup.conservation_mgr.enabled:
            f_for_monitor = sim.f
            rho_init, _ = self._setup.macro.compute(f_for_monitor)
            self._setup.conservation_mgr.initialize(rho_init, step=start_step)

        # ── Step 4b: Open CSV files (with start_step for restart) ─
        if self._setup.force_mgr is not None:
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

        # ── Step 4c: Esoteric parity restore on restart ─────────
        if (start_step > 0
                and hasattr(sim, '_use_esoteric') and sim._use_esoteric):
            # Restore esoteric_step from checkpoint extra_data
            # The checkpoint f is already in Esoteric memory layout
            if hasattr(self, '_last_checkpoint_state'):
                eso_step = self._last_checkpoint_state.get('esoteric_step', start_step)
                sim._esoteric_step = eso_step
                print(f"  Esoteric parity restored: step {eso_step}")
            else:
                sim._esoteric_step = start_step

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
        self._last_checkpoint_state = state  # store for esoteric parity
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
        physics_config = setup.sim_params.get('physics', {})
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
        physics_config = setup.sim_params.get('physics', {})
        flow_vel = physics_config.get('initial_flow_velocity', [0.0, 0.0, 0.0])
        dim = setup.lattice.dim

        print(f"\n[5] Initializing MultiLevelGrid ({mlg.num_levels} levels)")

        dtype = setup.compute_dtype

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

            f_k = level_sim.collision.compute_equilibrium(rho_0, u_0)
            level_sim.set_distribution(f_k)

            mem_mb = f_k.nbytes / (1024 * 1024)
            print(f"  Level {k}: shape={shape}, τ={level_sim.tau:.4f}, "
                  f"f size={mem_mb:.1f} MB")

        total_nodes = sum(
            mlg.get_level(k).f.size // setup.lattice.Q
            for k in range(mlg.num_levels)
        )
        print(f"  Total nodes across all levels: {total_nodes:,}")
        return 0

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

                physics_config = setup.sim_params.get('physics', {})
                flow_vel = physics_config.get('initial_flow_velocity',
                                              [0.0, 0.0, 0.0])
                if isinstance(flow_vel, (list, tuple)):
                    for d in range(min(len(flow_vel), dim)):
                        u_0[d] = flow_vel[d]
                else:
                    u_0[0] = flow_vel

                f_k = level_sim.collision.compute_equilibrium(rho_0, u_0)
                level_sim.set_distribution(f_k)

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