"""
CUDA Kernels for LBM Solver

Fused GPU kernels that replace multiple CuPy array operations
with a single kernel launch per physical step.

Available kernels:
    - bgk_d3q27: Fused macroscopic + Guo correction + BGK collision (D3Q27)

Usage:
    Kernels are automatically selected by Simulation.advance() when
    running on GPU. CPU (numpy) falls back to the original Python path.
"""
