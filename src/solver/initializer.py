"""
Solver Initializer — Physical State Initialization

This module prepares the distribution function f for execution.
It answers the question:
    "솔버에 맞는 초기 f를 어떻게 준비할까?"

Three modes:
    1. Fresh start (single grid):  f = f_eq(ρ₀, u₀)
    2. Fresh start (MLG):          f_k = f_eq(ρ₀, u₀) for each level k
    3. Restart:                    f = checkpoint data + resume step

Design Principle:
    "솔버 모드(BGK/Cumulant/기타)에 따라 달라지는가?" → Yes → Initializer

Author: LBM Development Team
Date: 2026-03 (MLG integration: 2026-04)
"""

from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    from src.solver.simulation import Simulation
    from src.solver.setup import SimulationSetup
    from src.grid.multi_level_grid import MultiLevelGrid


class SolverInitializer:
    """Initialize solver physical state (distribution function f).

    Handles fresh start (equilibrium) and checkpoint restart for both
    single-grid Simulation and multi-level MultiLevelGrid.

    After initialize(), the Simulation/MultiLevelGrid is ready for advance().

    Usage:
        >>> initializer = SolverInitializer(setup)
        >>> start_step, end_step = initializer.initialize(sim, args)
    """

    def __init__(self, setup: 'SimulationSetup') -> None:
        """Create initializer from a completed setup.

        Args:
            setup: SimulationSetup with all components constructed
        """
        self._setup = setup
        self.xp: 'ModuleType' = setup.xp

    def initialize(
        self,
        sim: Any,
        args: Any,
    ) -> Tuple[int, int]:
        """Initialize distribution function and determine step range.

        Handles both single-grid Simulation and MultiLevelGrid.

        Performs:
            1. Detect sim type (Simulation or MultiLevelGrid)
            2. Create or restore f (fresh start / checkpoint)
            3. Determine end_step from args + config
            4. Wire f into Simulation/MLG (set_distribution)
            5. Initialize conservation monitor

        Args:
            sim: Simulation or MultiLevelGrid object
            args: Parsed CLI arguments

        Returns:
            (start_step, end_step) tuple  [steps]

        Raises:
            RuntimeError: If restart requested but checkpoints disabled
        """
        # ── Detect MLG ───────────────────────────────────────────
        from src.grid.multi_level_grid import MultiLevelGrid
        is_mlg = isinstance(sim, MultiLevelGrid)

        # ── Step 1: Create or restore f ──────────────────────────
        if is_mlg:
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
        print(f"  Steps to run: {total_steps} "
              f"({start_step} → {end_step - 1})")

        # ── Step 3: Wire f into Simulation ───────────────────────
        if not is_mlg:
            sim.set_distribution(f)
            sim.step_count = start_step

        # ── Step 4: Conservation initialization ──────────────────
        if self._setup.conservation_mgr and self._setup.conservation_mgr.enabled:
            # For MLG, use Level 0's f; for single grid, use f directly
            f_for_monitor = sim.f
            rho_init, _ = self._setup.macro.compute(f_for_monitor)
            self._setup.conservation_mgr.initialize(rho_init, step=start_step)

        # ── Step 5: Info ─────────────────────────────────────────
        if hasattr(sim, 'print_info'):
            sim.print_info()

        return start_step, end_step

    # =====================================================================
    # Private: Single-grid initialization
    # =====================================================================

    def _create_distribution(self, args: Any) -> Tuple[Any, int]:
        """Create initial distribution function (single grid).

        Returns:
            (f, start_step): Distribution array and starting step index

        Strategy selection:
            - restart_latest → load latest checkpoint
            - restart <path> → load specific checkpoint
            - else           → f_eq(ρ₀, u₀) via Maxwellian equilibrium
        """
        setup = self._setup
        xp = self.xp

        if args.restart_latest:
            return self._restart_latest(xp, setup)

        if args.restart:
            return self._restart_from(args.restart, xp, setup)

        return self._fresh_start(xp, setup)

    def _restart_latest(
        self, xp: 'ModuleType', setup: 'SimulationSetup',
    ) -> Tuple[Any, int]:
        """Restore from latest checkpoint."""
        print(f"\n[5] Restarting from latest checkpoint...")

        if setup.checkpoint_mgr is None:
            raise RuntimeError("Cannot restart: checkpoints are disabled")

        setup.checkpoint_mgr.print_available()
        state = setup.checkpoint_mgr.load_latest()
        f = xp.asarray(state['f'])

        completed_step = state['step']
        start_step = completed_step + 1
        print(f"  Loaded step {completed_step}, "
              f"resuming from step {start_step}")

        return f, start_step

    def _restart_from(
        self,
        path: str,
        xp: 'ModuleType',
        setup: 'SimulationSetup',
    ) -> Tuple[Any, int]:
        """Restore from a specific checkpoint file."""
        print(f"\n[5] Restarting from: {path}")

        if setup.checkpoint_mgr is None:
            from src.io.checkpoint import CheckpointManager
            setup.checkpoint_mgr = CheckpointManager(
                output_dir=setup.checkpoint_dir, xp=xp,
            )

        state = setup.checkpoint_mgr.load(path)
        f = xp.asarray(state['f'])

        completed_step = state['step']
        start_step = completed_step + 1
        print(f"  Loaded step {completed_step}, "
              f"resuming from step {start_step}")

        return f, start_step

    def _fresh_start(
        self, xp: 'ModuleType', setup: 'SimulationSetup',
    ) -> Tuple[Any, int]:
        """Create f from collision operator's equilibrium.

        For BGK solver:
            f = f_eq(ρ₀, u₀)
            where f_eq = w_i · ρ · (1 + 3(c·u) + 4.5(c·u)² - 1.5|u|²)

        Each collision model provides its own compute_equilibrium(),
        so Cumulant solver can use a different equilibrium if needed.
        """
        print(f"\n[5] Initializing Flow Field (Fresh Start)...")

        physics_config = setup.sim_params.get('physics', {})
        flow_vel = physics_config.get('initial_flow_velocity', 0.0)

        # ── Allocate ρ₀ = 1, u₀ = flow_vel ──
        if setup.lattice.dim == 2:
            rho0 = xp.ones(
                (setup.Nx, setup.Ny), dtype=xp.float64,
            )                                                # [dimensionless]
            u0 = xp.zeros(
                (2, setup.Nx, setup.Ny), dtype=xp.float64,
            )                                                # [Δx/Δt]
        else:
            rho0 = xp.ones(
                (setup.Nx, setup.Ny, setup.Nz), dtype=xp.float64,
            )                                                # [dimensionless]
            u0 = xp.zeros(
                (3, setup.Nx, setup.Ny, setup.Nz), dtype=xp.float64,
            )                                                # [Δx/Δt]

        # Initial velocity: scalar → x-direction (legacy),
        #                   list   → per-component
        if isinstance(flow_vel, (list, tuple)):
            for d in range(min(len(flow_vel), setup.lattice.dim)):
                u0[d] = flow_vel[d]                          # [Δx/Δt]
            print(f"  Initial velocity: {flow_vel} [Δx/Δt]")
        else:
            u0[0] = flow_vel                                 # [Δx/Δt]
            print(f"  Initial velocity: {flow_vel} [Δx/Δt] (x-dir)")

        # ── Compute f_eq via collision operator ──
        f = setup.collision.compute_equilibrium(rho0, u0)    # [dimensionless]
        print(f"  Initial total mass: {float(xp.sum(f)):.6f}")

        return f, 0  # start_step = 0

    # =====================================================================
    # Private: Multi-Level Grid initialization
    # =====================================================================

    def _initialize_mlg(
        self,
        mlg: 'MultiLevelGrid',
    ) -> int:
        """Initialize all levels of a MultiLevelGrid with equilibrium.

        Each level gets f = f_eq(ρ₀, u₀) at its own resolution.
        The initial velocity is the same across all levels — physical
        velocity is continuous under convective scaling, so u₀ is
        identical in lattice units for all levels.

        Args:
            mlg: MultiLevelGrid with uninitialized Simulation objects.

        Returns:
            start_step: Always 0 for fresh start.
        """
        setup = self._setup
        xp = self.xp

        physics_config = setup.sim_params.get('physics', {})
        flow_vel = physics_config.get('initial_flow_velocity', [0.0, 0.0, 0.0])
        dim = setup.lattice.dim

        print(f"\n[5] Initializing MultiLevelGrid ({mlg.num_levels} levels)")

        for k in range(mlg.num_levels):
            level_sim = mlg.get_level(k)
            shape = level_sim.domain_shape

            # ── Build ρ₀ = 1, u₀ = initial_flow_velocity ─────────
            rho_0 = xp.ones(shape, dtype=xp.float64)
            u_0 = xp.zeros((dim,) + shape, dtype=xp.float64)

            if isinstance(flow_vel, (list, tuple)):
                for d in range(min(len(flow_vel), dim)):
                    u_0[d] = flow_vel[d]
            else:
                u_0[0] = flow_vel

            # ── f = f_eq(ρ₀, u₀) ─────────────────────────────────
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

        return 0  # fresh start

    # =====================================================================
    # Private: Utilities
    # =====================================================================

    def _determine_end_step(self, args: Any, start_step: int) -> int:
        """Determine end step from CLI args or config.

        Priority: --max-steps > --extend > config max_steps

        Args:
            args: CLI arguments
            start_step: Starting step index

        Returns:
            end_step: One past the last step to run  [steps]
        """
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