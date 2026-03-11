"""
Solver Package — Core simulation orchestration.

Modules:
    simulation: Simulation class (single-step physics)
    output_manager: OutputManager (I/O, monitoring)
    setup:      SimulationSetup (environment config)      — M3
    initializer: SolverInitializer (physics initialization) — M4
"""

from src.solver.simulation import Simulation
from src.solver.output_manager import OutputManager

__all__ = ['Simulation', 'OutputManager']