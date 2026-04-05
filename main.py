"""
LBM Solver for Aerodynamics & Aeroacoustics

Unified entry point for all simulation types:
  - Channel flow (Poiseuille, lid-driven cavity)
  - External flow (cylinder, airfoil)
  - Wind turbine (Actuator Line model)

Usage:
    python main.py --config configs/input_config.py
    python main.py --config configs/NTNU_BT1_config.py
    python main.py --restart-latest --extend 10000
"""

import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from src.io.args_parser import parse_args
from src.solver.setup import SimulationSetup
from src.solver.initializer import SolverInitializer


def main():
    args = parse_args()

    # GPU 목록 출력 후 종료
    if args.list_gpus:
        from src.utilities.device import print_gpu_info
        print_gpu_info()
        return

    print("=" * 70)
    print(" LBM Solver for Aerodynamics & Aeroacoustics")
    print("=" * 70)

    # [1] Setup — 시뮬레이션 환경 구성
    setup = SimulationSetup(args)
    sim    = setup.build_simulation()
    output = setup.build_output_manager()

    # [2] Initialization — 솔버 물리 상태 초기화
    initializer = SolverInitializer(setup)
    start_step, end_step = initializer.initialize(sim, args)

    if start_step >= end_step:
        return True

    # [3] Execution — 시간 루프
    print(f"\n[6] Running Simulation")
    print("=" * 70)

    output.start(start_step, end_step)

    for step in range(start_step, end_step):
        sim.advance()
        if output.process(step, sim) == 'stop':
            break
    
    import numpy as np
    u_np = sim.u.get() if hasattr(sim.u, 'get') else sim.u
    np.save('velocity_field.npy', u_np)

    # [4] Finalize
    return output.finalize(sim)


if __name__ == "__main__":
    main()