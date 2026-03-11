"""
LBM Solver for Aerodynamics & Aeroacoustics

Unified entry point for all simulation types:
  - Channel flow (Poiseuille, lid-driven cavity)
  - External flow (cylinder, airfoil)
  - Wind turbine (Actuator Line model)

Features are activated/deactivated via config:
  - Actuator Line: config["actuator_line"]["enabled"] = True/False
  - Force calculation: config["force_calculation"]["enabled"] = True/False
  - Internal geometry: config["internal_geometry"]

Usage:
    python main.py --config configs/input_config.py
    python main.py --config configs/NTNU_BT1_config.py
    python main.py --restart-latest --extend 10000
"""

import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from src.io.args_parser import parse_args
from src.solver.setup import SimulationSetup


# =============================================================================
# Main Simulation
# =============================================================================
def main():
    args = parse_args()

    # GPU 목록 출력 후 종료
    if args.list_gpus:
        from src.utilities.device import print_gpu_info
        print_gpu_info()
        return

    print("="*70)
    print(" LBM Solver for Aerodynamics & Aeroacoustics")
    print("="*70)

    # =========================================================================
    # [1] Setup — 시뮬레이션 환경 구성
    # =========================================================================
    setup = SimulationSetup(args)
    sim    = setup.build_simulation()
    output = setup.build_output_manager()

    # =========================================================================
    # [2] Initialization — 초기 분포 함수 (f)
    #     TODO(M4): SolverInitializer로 이동 예정
    # =========================================================================
    xp = setup.xp
    start_step = 0

    if args.restart_latest:
        print(f"\n[5] Restarting from latest checkpoint...")
        if setup.checkpoint_mgr is None:
            raise RuntimeError("Cannot restart: checkpoints are disabled")
        setup.checkpoint_mgr.print_available()
        state = setup.checkpoint_mgr.load_latest()
        f_old = xp.asarray(state['f'])

        completed_step = state['step']
        start_step = completed_step + 1
        print(f"  Loaded step {completed_step}, resuming from step {start_step}")

    elif args.restart:
        print(f"\n[5] Restarting from: {args.restart}")
        if setup.checkpoint_mgr is None:
            from src.io.checkpoint import CheckpointManager
            setup.checkpoint_mgr = CheckpointManager(
                output_dir=setup.checkpoint_dir, xp=xp,
            )
        state = setup.checkpoint_mgr.load(args.restart)
        f_old = xp.asarray(state['f'])

        completed_step = state['step']
        start_step = completed_step + 1
        print(f"  Loaded step {completed_step}, resuming from step {start_step}")

    else:
        print(f"\n[5] Initializing Flow Field (Fresh Start)...")
        physics_config = setup.sim_params.get('physics', {})
        flow_vel = physics_config.get('initial_flow_velocity', 0.0)

        if setup.lattice.dim == 2:
            rho0 = xp.ones((setup.Nx, setup.Ny), dtype=xp.float64)
            u0 = xp.zeros((2, setup.Nx, setup.Ny), dtype=xp.float64)
        else:
            rho0 = xp.ones(
                (setup.Nx, setup.Ny, setup.Nz), dtype=xp.float64,
            )
            u0 = xp.zeros(
                (3, setup.Nx, setup.Ny, setup.Nz), dtype=xp.float64,
            )

        # Initial velocity: scalar → x-direction (legacy), list → per-component
        if isinstance(flow_vel, (list, tuple)):
            for d in range(min(len(flow_vel), setup.lattice.dim)):
                u0[d] = flow_vel[d]    # [Δx/Δt]
            print(f"  Initial velocity: {flow_vel} [Δx/Δt]")
        else:
            u0[0] = flow_vel           # [Δx/Δt] x-direction (legacy)
            print(f"  Initial velocity: {flow_vel} [Δx/Δt] (x-dir)")

        f_old = setup.equilibrium.compute(rho0, u0)
        print(f"  Initial total mass: {float(xp.sum(f_old)):.6f}")

    # ── Determine End Step ──
    if args.max_steps is not None:
        end_step = args.max_steps
        print(f"\n  End step (--max-steps): {end_step}")
    elif args.extend is not None:
        end_step = start_step + args.extend
        print(f"\n  Extending by {args.extend} steps: {start_step} → {end_step}")
    else:
        end_step = setup.config_max_steps
        print(f"\n  End step (from config): {end_step}")

    if start_step >= end_step:
        print(f"\n  ⚠️  start_step ({start_step}) >= end_step ({end_step})")
        print(f"      Use --extend N or --max-steps N to continue")
        return True

    total_steps = end_step - start_step
    print(f"  Steps to run: {total_steps} ({start_step} → {end_step - 1})")

    # ── Wire f into Simulation ──
    sim.set_distribution(f_old)
    sim.step_count = start_step

    # ── Conservation initialization (needs f_old) ──
    rho_init, _ = setup.macro.compute(f_old)
    setup.conservation_mgr.initialize(rho_init, step=start_step)

    sim.print_info()

    # =========================================================================
    # [3] Execution — 시간 루프
    # =========================================================================
    print(f"\n[6] Running Simulation")
    print("="*70)

    output.start(start_step, end_step)

    for step in range(start_step, end_step):
        sim.advance()
        action = output.process(step, sim)
        if action == 'stop':
            break

    # =========================================================================
    # [4] Finalize
    # =========================================================================
    return output.finalize(sim)


if __name__ == "__main__":
    main()