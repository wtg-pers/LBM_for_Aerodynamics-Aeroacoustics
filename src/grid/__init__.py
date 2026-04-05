"""
Multi-Level Grid (MLG) System for LBM

Modules:
    level_scaling      — Level-wise τ, ν, f^neq rescaling between grid levels
    grid_level         — Single resolution level (BoundingBox, arrays)
    interpolation      — Cubic (4th order) and compact 2nd-order schemes
    overlap_manager    — Multi-domain overlap geometry and index mapping
    coupling           — Coarse↔Fine two-way data transfer engine
    multi_level_grid   — Nested time-stepping orchestrator

Author: LBM Development Team
Date: 2026-04
"""

from src.grid.level_scaling import LevelScaler, LevelUnits
from src.grid.grid_level import GridLevel, BoundingBox
from src.grid.interpolation import CubicInterpolation, CompactSecondOrderInterpolation
from src.grid.overlap_manager import OverlapRegion, OverlapManager, IndexBox
from src.grid.coupling import GridCoupling
from src.grid.multi_level_grid import MultiLevelGrid

__all__ = [
    "LevelScaler", "LevelUnits",
    "GridLevel", "BoundingBox",
    "CubicInterpolation", "CompactSecondOrderInterpolation",
    "OverlapRegion", "OverlapManager", "IndexBox",
    "GridCoupling",
    "MultiLevelGrid",
]