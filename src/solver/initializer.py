"""
Solver Initializer — Physical State Initialization

This module prepares the distribution function f for execution.
It answers the question:
    "솔버에 맞는 초기 f를 어떻게 준비할까?"

Two modes:
    1. Fresh start:  f = f_eq(ρ₀, u₀) from Maxwellian equilibrium
    2. Restart:       f = checkpoint data + resume step

Future extension:
    - Cumulant collision may require different equilibrium computation
    - 2-step interpolation needs (f_{n}, f_{n-1}) pair

Design Principle:
    "솔버 모드(BGK/Cumulant/기타)에 따라 달라지는가?" → Yes → Initializer

Author: LBM Development Team
Date: 2026-03
"""

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    from src.solver.simulation import Simulation
    from src.solver.setup import SimulationSetup


class SolverInitializer:
    """Initialize solver physical state (distribution function f).

    Handles fresh start (equilibrium) and checkpoint restart.
    After initialize(), the Simulation is ready for advance().

    Usage:
        >>> initializer = SolverInitializer(setup)
        >>> start_step = initializer.initialize(sim, args)
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
        sim: 'Simulation',
        args: Any,
    ) -> Tuple[int, int]:
        """Initialize distribution function and determine step range.

        Performs:
            1. Create or restore f (fresh start / checkpoint)
            2. Determine end_step from args + config
            3. Wire f into Simulation (set_distribution)
            4. Initialize conservation monitor
            5. Print simulation info

        Args:
            sim: Simulation object (from setup.build_simulation())
            args: Parsed CLI arguments

        Returns:
            (start_step, end_step) tuple  [steps]

        Raises:
            RuntimeError: If restart requested but checkpoints disabled
        """
        # ── Step 1: Create or restore f ──
        f, start_step = self._create_distribution(args)

        # ── Step 2: Determine end step ──
        end_step = self._determine_end_step(args, start_step)

        if start_step >= end_step:
            print(f"\n  ⚠️  start_step ({start_step}) >= end_step ({end_step})")
            print(f"      Use --extend N or --max-steps N to continue")
            return start_step, end_step

        total_steps = end_step - start_step
        print(f"  Steps to run: {total_steps} "
              f"({start_step} → {end_step - 1})")

        # ── Step 3: Wire f into Simulation ──
        sim.set_distribution(f)
        sim.step_count = start_step

        # ── Step 4: Conservation initialization ──
        rho_init, _ = self._setup.macro.compute(f)
        self._setup.conservation_mgr.initialize(rho_init, step=start_step)

        # ── Step 5: Info ──
        sim.print_info()

        return start_step, end_step

    # =====================================================================
    # Private
    # =====================================================================

    def _create_distribution(self, args: Any) -> Tuple[Any, int]:
        """Create initial distribution function.

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
        """Create f from Maxwellian equilibrium.

        For BGK solver:
            f = f_eq(ρ₀, u₀)
            where f_eq = w_i · ρ · (1 + 3(c·u) + 4.5(c·u)² - 1.5|u|²)

        Future: Cumulant solver may use a different equilibrium.
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

        # ── Compute f_eq ──
        f = setup.equilibrium.compute(rho0, u0)              # [dimensionless]
        print(f"  Initial total mass: {float(xp.sum(f)):.6f}")

        return f, 0  # start_step = 0

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