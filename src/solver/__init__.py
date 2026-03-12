"""
Solver Package — Core simulation orchestration.

Modules:
    simulation:     Simulation class (single-step physics)
    output_manager: OutputManager (I/O, monitoring)
    setup:          SimulationSetup (environment construction)
    initializer:    SolverInitializer (physics initialization)
"""

from src.solver.simulation import Simulation
from src.solver.output_manager import OutputManager
from src.solver.setup import SimulationSetup
from src.solver.initializer import SolverInitializer

__all__ = [
    'Simulation', 'OutputManager', 'SimulationSetup', 'SolverInitializer',
]