"""Multi-GPU domain decomposition (patch_notes/hpc_upgrade/17).

Axis-generic 1D partition + esoteric halos (v1 physical-band, v2 raw-slot)
+ rank-local MLG coupling + distributed ALM + SPMD runner.
"""
from src.parallel.partition import Partition1D, choose_axis, AXIS_INDEX
from src.parallel.halo import (
    HaloBandExchangerV1, LoopbackTransport, MPITransport)

__all__ = ["Partition1D", "choose_axis", "AXIS_INDEX",
           "HaloBandExchangerV1", "LoopbackTransport", "MPITransport"]
