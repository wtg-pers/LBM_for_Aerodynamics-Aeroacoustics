"""
Post-Processing Utilities for LBM Solver

Modules:
    - vortex_id: Q-criterion, Lambda2, vorticity for vortex visualization
"""

from .vortex_id import (
    compute_velocity_gradient_3d,
    compute_strain_rotation,
    compute_q_criterion,
    compute_lambda2,
    compute_vorticity_3d,
    compute_vorticity_magnitude,
    compute_all_vortex_criteria,
    normalize_q_criterion,
    normalize_lambda2,
)

__all__ = [
    'compute_velocity_gradient_3d',
    'compute_strain_rotation',
    'compute_q_criterion',
    'compute_lambda2',
    'compute_vorticity_3d',
    'compute_vorticity_magnitude',
    'compute_all_vortex_criteria',
    'normalize_q_criterion',
    'normalize_lambda2',
]