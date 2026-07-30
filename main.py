"""
LBM Solver for Aerodynamics & Aeroacoustics — unified entry point.

Single-GPU and MPI runs dispatch from the same command:
    python main.py --config configs/my_config.py            # single GPU
    python main.py --restart-latest --extend 10000
    mpirun -n 4 python main.py --config cfg.py --steps 2000  # multi GPU

All logic lives in src/ (src/solver/entry.py, src/parallel/driver.py) so an
src-only deploy can never miss the driver again.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.solver.entry import main

if __name__ == "__main__":
    main()
