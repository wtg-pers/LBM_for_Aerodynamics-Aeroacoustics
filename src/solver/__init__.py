"""
Solver Package — Core simulation orchestration.

Modules:
    simulation: Simulation class (single-step physics)
    setup:      SimulationSetup (environment config)      — M3
    initializer: SolverInitializer (physics initialization) — M4
    output_manager: OutputManager (I/O, monitoring)        — M2
"""

from src.solver.simulation import Simulation

__all__ = ['Simulation']