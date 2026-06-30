"""
Streaming Operator for LBM (Pull Scheme)

This module implements the streaming step using the pull scheme,
where each node pulls distributions from its neighbors.

Physical Process:
    In streaming, distributions propagate along their velocity directions:
        f_i(x, t+Δt) = f_i^post(x - c_i, t)

    This is the pull scheme: we read from source locations (x - c_i).

Implementation:
    Source indices are precomputed for vectorized operations,
    enabling efficient GPU execution without Python loops.

Supports:
    - 2D (D2Q9) and 3D (D3Q27, D3Q19) lattice models
    - Periodic boundaries (implicit via modulo arithmetic)
    - NumPy and CuPy backends

References:
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 4

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class StreamingPull:
    """Vectorized Pull-scheme Streaming Operator
    
    The pull scheme reads from neighboring cells:
        f_i(x, t+1) = f_i^post(x - c_i, t)
    
    This is implemented by precomputing source indices, enabling
    fully vectorized operations without Python loops.
    
    Supports both 2D and 3D domains.
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model containing velocity vectors
        shape: Domain shape (Nx, Ny) for 2D or (Nx, Ny, Nz) for 3D
        dim: Spatial dimension (2 or 3)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice', 
                 shape: tuple) -> None:
        """Initialize streaming operator
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        """
        self.xp = xp
        self.lattice = lattice
        self.dim = lattice.dim
        self.shape = shape
        self.Q = lattice.Q

        # Skip precomputing index arrays if CUDA streaming kernel is available
        # (saves 4 × Q × N × 4B of GPU memory)
        self._use_cuda = False
        if xp.__name__ == 'cupy' and self.dim == 3:
            try:
                from src.kernels.streaming_d3q27 import StreamingKernelD3Q27
                self._cuda_kernel = StreamingKernelD3Q27()
                self._use_cuda = True
            except Exception:
                self._precompute_indices()
        elif xp.__name__ == 'cupy' and self.dim == 2 and self.Q == 9:
            try:
                from src.kernels.streaming_d2q9 import StreamingKernelD2Q9
                self._cuda_kernel = StreamingKernelD2Q9()
                self._use_cuda = True
            except Exception:
                self._precompute_indices()
        else:
            self._precompute_indices()

    def _precompute_indices(self) -> None:
        """Precompute source indices for pull scheme
        
        For pull scheme: src[i, x] = (x - c_i) mod N
        
        This allows vectorized streaming without loops.
        """
        xp = self.xp
        c = xp.asarray(self.lattice.c)

        if self.dim == 2:
            self._precompute_indices_2d(c)
        else:
            self._precompute_indices_3d(c)
    
    def _precompute_indices_2d(self, c: 'npt.NDArray') -> None:
        """Precompute 2D source indices
        
        Args:
            c: Velocity vectors, shape (2, Q)  [Δx/Δt]
        """
        xp = self.xp
        Nx, Ny = self.shape
        
        # Create coordinate grids  [lattice units]
        ix = xp.arange(Nx, dtype=xp.int32)
        iy = xp.arange(Ny, dtype=xp.int32)
        IX, IY = xp.meshgrid(ix, iy, indexing='ij')
        
        # Compute source indices: (Q, Nx, Ny)
        # Pull scheme: source = current - velocity (with periodic wrap)
        self.src_x = (IX[None, :, :] - c[0, :, None, None] + Nx) % Nx
        self.src_y = (IY[None, :, :] - c[1, :, None, None] + Ny) % Ny
        
        self.src_x = self.src_x.astype(xp.int32)
        self.src_y = self.src_y.astype(xp.int32)
        
        # Direction indices for advanced indexing
        self.q_idx = xp.arange(self.Q, dtype=xp.int32)[:, None, None]
        self.q_idx = xp.broadcast_to(self.q_idx, (self.Q, Nx, Ny))
    
    def _precompute_indices_3d(self, c: 'npt.NDArray') -> None:
        """Precompute 3D source indices
        
        Args:
            c: Velocity vectors, shape (3, Q)  [Δx/Δt]
        """
        xp = self.xp
        Nx, Ny, Nz = self.shape
        
        # Create coordinate grids  [lattice units]
        ix = xp.arange(Nx, dtype=xp.int32)
        iy = xp.arange(Ny, dtype=xp.int32)
        iz = xp.arange(Nz, dtype=xp.int32)
        IX, IY, IZ = xp.meshgrid(ix, iy, iz, indexing='ij')
        
        # Compute source indices: (Q, Nx, Ny, Nz)
        self.src_x = (IX[None, :, :, :] - c[0, :, None, None, None] + Nx) % Nx
        self.src_y = (IY[None, :, :, :] - c[1, :, None, None, None] + Ny) % Ny
        self.src_z = (IZ[None, :, :, :] - c[2, :, None, None, None] + Nz) % Nz
        
        self.src_x = self.src_x.astype(xp.int32)
        self.src_y = self.src_y.astype(xp.int32)
        self.src_z = self.src_z.astype(xp.int32)
        
        # Direction indices for advanced indexing
        self.q_idx = xp.arange(self.Q, dtype=xp.int32)[:, None, None, None]
        self.q_idx = xp.broadcast_to(self.q_idx, (self.Q, Nx, Ny, Nz))
    
    def compute(self, f_post: 'npt.NDArray', f_next: 'npt.NDArray') -> None:
        """Perform streaming step (out-of-place)

        Implements: f_i(x, t+1) = f_i^post(x - c_i, t)

        Args:
            f_post: Post-collision distribution, shape (Q, Nx, Ny[, Nz])
                   [dimensionless]
            f_next: Output buffer for streamed distribution
                   [dimensionless] - Modified in-place
        """
        if self._use_cuda:
            if self.dim == 3:
                Nx, Ny, Nz = self.shape
                self._cuda_kernel.launch(f_post, f_next, Nx, Ny, Nz)
            else:
                Nx, Ny = self.shape
                self._cuda_kernel.launch(f_post, f_next, Nx, Ny)
        elif self.dim == 2:
            f_next[:] = f_post[self.q_idx, self.src_x, self.src_y]
        else:
            f_next[:] = f_post[self.q_idx, self.src_x, self.src_y, self.src_z]
    
    def stream_inplace(self, f: 'npt.NDArray') -> 'npt.NDArray':
        """Streaming with internal temporary buffer
        
        Creates a new array for the result. Use compute() for
        pre-allocated buffers to avoid memory allocation overhead.
        
        Args:
            f: Distribution function  [dimensionless]
            
        Returns:
            Streamed distribution (new array)  [dimensionless]
        """
        f_next = self.xp.empty_like(f)
        self.compute(f, f_next)
        return f_next