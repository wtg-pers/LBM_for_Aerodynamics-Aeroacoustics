"""
Outlet Boundary Conditions for LBM

This module implements outlet boundary conditions using pressure relaxation
and non-equilibrium extrapolation methods.

Supports both 2D and 3D domains.

Physical Principle:
    At the outlet, we allow waves to exit with minimal reflection.
    - Pressure relaxation: Smoothly relaxes density to target value
    - Non-equilibrium extrapolation: Preserves viscous stress information

References:
    - Latt & Chopard, Mathematics and Computers in Simulation 72, 2006
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Union, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class PressureRelaxationOutlet(BoundaryCondition):
    """Pressure Relaxation Outlet Boundary Condition
    
    Non-reflecting outlet with density/pressure relaxation:
        ρ_boundary = ρ_target + (1-K)(ρ_interior - ρ_target)
    
    where K is the relaxation coefficient:
        K=0: Fully non-reflecting (pure extrapolation)
        K=1: Fixed pressure (strong reflection)
        K~0.1-0.3: Good compromise for most flows
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 rho_target: float = 1.0,
                 relax_coeff: float = 0.1,
                 shape: Optional[tuple] = None) -> None:
        """Initialize pressure relaxation outlet
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            location: Boundary location (xmax typically)
            rho_target: Target density  [dimensionless]
            relax_coeff: Relaxation coefficient K (0-1)  [dimensionless]
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_target = rho_target
        self.K = relax_coeff
        self.shape = shape
        self.dim = lattice.dim
        
        self._setup_boundary_shape()
        self._precompute_coeffs()
    
    def _setup_boundary_shape(self) -> None:
        """Setup boundary shape based on location and dimension"""
        if self.shape is None:
            raise ValueError("shape must be provided for outlet BC")
        
        loc = self.location.value
        
        if self.dim == 2:
            Nx, Ny = self.shape
            if loc in ['xmin', 'xmax', 'west', 'east']:
                self.boundary_shape = (Ny,)
            else:
                self.boundary_shape = (Nx,)
        else:
            Nx, Ny, Nz = self.shape
            if loc in ['xmin', 'xmax', 'west', 'east']:
                self.boundary_shape = (Ny, Nz)
            elif loc in ['ymin', 'ymax', 'south', 'north']:
                self.boundary_shape = (Nx, Nz)
            else:
                self.boundary_shape = (Nx, Ny)
    
    def _precompute_coeffs(self) -> None:
        """Precompute coefficients"""
        xp = self.xp
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
        self.c_all = self.c.astype(xp.float64)
        self.w_all = self.w.astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply pressure relaxation outlet
        
        Algorithm:
        1. Extract f at interior node
        2. Compute macroscopic (ρ, u) at interior
        3. Compute f^eq at interior → f^neq = f - f^eq
        4. Apply pressure relaxation: ρ_bc = ρ_target + (1-K)(ρ_int - ρ_target)
        5. Extrapolate velocity: u_bc = u_interior
        6. Reconstruct: f = f^eq(ρ_bc, u_bc) + f^neq(interior)
        
        Args:
            f: Distribution function, modified in-place
        """
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply for 2D domain"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        
        # Get interior slice
        if loc in ['xmin', 'west']:
            f_interior = f[:, 1, :]
        elif loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, :]
        elif loc in ['ymin', 'south']:
            f_interior = f[:, :, 1]
        else:  # ymax, north
            f_interior = f[:, :, Ny-2]
        
        # Compute macroscopic at interior
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # Compute f^eq at interior
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # f^neq at interior
        f_neq_int = f_interior - f_eq_int
        
        # Pressure relaxation
        rho_boundary = self.rho_target + (1 - self.K) * (rho_int - self.rho_target)
        u_boundary = u_int  # Extrapolate velocity
        
        # Compute f^eq at boundary for incoming directions
        usqr = xp.sum(u_boundary**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_boundary)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_boundary = w_in * rho_boundary * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        # Reconstruct: f = f^eq(boundary) + f^neq(interior)
        incoming = self.incoming_indices
        f_incoming = f_eq_boundary + f_neq_int[incoming]
        
        # Apply
        if loc in ['xmin', 'west']:
            f[incoming, 0, :] = f_incoming
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :] = f_incoming
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0] = f_incoming
        else:  # ymax, north
            f[incoming, :, Ny-1] = f_incoming
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply for 3D domain"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        
        # Get interior slice
        if loc in ['xmin', 'west']:
            f_interior = f[:, 1, :, :]
        elif loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, :, :]
        elif loc in ['ymin', 'south']:
            f_interior = f[:, :, 1, :]
        elif loc in ['ymax', 'north']:
            f_interior = f[:, :, Ny-2, :]
        elif loc in ['zmin', 'bottom']:
            f_interior = f[:, :, :, 1]
        else:  # zmax, top
            f_interior = f[:, :, :, Nz-2]
        
        # Compute macroscopic at interior
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # Compute f^eq at interior
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # f^neq at interior
        f_neq_int = f_interior - f_eq_int
        
        # Pressure relaxation
        rho_boundary = self.rho_target + (1 - self.K) * (rho_int - self.rho_target)
        u_boundary = u_int
        
        # Compute f^eq at boundary for incoming directions
        usqr = xp.sum(u_boundary**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_boundary)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_boundary = w_in * rho_boundary * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        # Reconstruct
        incoming = self.incoming_indices
        f_incoming = f_eq_boundary + f_neq_int[incoming]
        
        # Apply
        if loc in ['xmin', 'west']:
            f[incoming, 0, :, :] = f_incoming
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :, :] = f_incoming
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0, :] = f_incoming
        elif loc in ['ymax', 'north']:
            f[incoming, :, Ny-1, :] = f_incoming
        elif loc in ['zmin', 'bottom']:
            f[incoming, :, :, 0] = f_incoming
        else:  # zmax, top
            f[incoming, :, :, Nz-1] = f_incoming


class ConvectiveOutlet(BoundaryCondition):
    """Convective (Advective) Outlet Boundary Condition
    
    Advects distributions out of the domain:
        ∂f/∂t + U_convective * ∂f/∂n = 0
    
    Discretized as:
        f_i^{n+1}(boundary) = f_i^n(boundary) - U_c * Δt/Δx * (f_i^n(boundary) - f_i^n(interior))
    
    Good for vortex-dominated flows where structures exit the domain.
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 convective_velocity: float = 0.1,
                 shape: Optional[tuple] = None) -> None:
        """Initialize convective outlet
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            location: Boundary location
            convective_velocity: Advection velocity  [Δx/Δt]
            shape: Domain shape  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.U_conv = convective_velocity
        self.shape = shape
        self.dim = lattice.dim
        
        # Store previous boundary values for time derivative
        self._f_prev = None
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply convective outlet condition"""
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply for 2D domain"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        incoming = self.incoming_indices
        
        # Get boundary and interior slices
        if loc in ['xmin', 'west']:
            f_boundary = f[incoming, 0, :]
            f_interior = f[incoming, 1, :]
        elif loc in ['xmax', 'east']:
            f_boundary = f[incoming, Nx-1, :]
            f_interior = f[incoming, Nx-2, :]
        elif loc in ['ymin', 'south']:
            f_boundary = f[incoming, :, 0]
            f_interior = f[incoming, :, 1]
        else:  # ymax, north
            f_boundary = f[incoming, :, Ny-1]
            f_interior = f[incoming, :, Ny-2]
        
        # Initialize previous state if needed
        if self._f_prev is None:
            self._f_prev = f_boundary.copy()
        
        # Convective update: f_new = f_old - U * (f_boundary - f_interior)
        f_new = self._f_prev - self.U_conv * (self._f_prev - f_interior)
        
        # Store for next time step
        self._f_prev = f_new.copy()
        
        # Apply
        if loc in ['xmin', 'west']:
            f[incoming, 0, :] = f_new
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :] = f_new
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0] = f_new
        else:
            f[incoming, :, Ny-1] = f_new
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply for 3D domain"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        incoming = self.incoming_indices
        
        # Get boundary and interior slices
        if loc in ['xmin', 'west']:
            f_boundary = f[incoming, 0, :, :]
            f_interior = f[incoming, 1, :, :]
        elif loc in ['xmax', 'east']:
            f_boundary = f[incoming, Nx-1, :, :]
            f_interior = f[incoming, Nx-2, :, :]
        elif loc in ['ymin', 'south']:
            f_boundary = f[incoming, :, 0, :]
            f_interior = f[incoming, :, 1, :]
        elif loc in ['ymax', 'north']:
            f_boundary = f[incoming, :, Ny-1, :]
            f_interior = f[incoming, :, Ny-2, :]
        elif loc in ['zmin', 'bottom']:
            f_boundary = f[incoming, :, :, 0]
            f_interior = f[incoming, :, :, 1]
        else:  # zmax, top
            f_boundary = f[incoming, :, :, Nz-1]
            f_interior = f[incoming, :, :, Nz-2]
        
        # Initialize previous state if needed
        if self._f_prev is None:
            self._f_prev = f_boundary.copy()
        
        # Convective update
        f_new = self._f_prev - self.U_conv * (self._f_prev - f_interior)
        self._f_prev = f_new.copy()
        
        # Apply
        if loc in ['xmin', 'west']:
            f[incoming, 0, :, :] = f_new
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :, :] = f_new
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0, :] = f_new
        elif loc in ['ymax', 'north']:
            f[incoming, :, Ny-1, :] = f_new
        elif loc in ['zmin', 'bottom']:
            f[incoming, :, :, 0] = f_new
        else:
            f[incoming, :, :, Nz-1] = f_new


class NeumannOutlet(BoundaryCondition):
    """Zero-Gradient Extrapolation Outlet
    
    Simple first-order extrapolation:
        f_i(boundary) = f_i(interior)
    
    Minimal reflection but can cause mass drift.
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 shape: Optional[tuple] = None) -> None:
        super().__init__(xp, lattice, location)
        self.shape = shape
        self.dim = lattice.dim
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply zero-gradient extrapolation"""
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply for 2D domain"""
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        incoming = self.incoming_indices
        
        if loc in ['xmin', 'west']:
            f[incoming, 0, :] = f[incoming, 1, :]
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :] = f[incoming, Nx-2, :]
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0] = f[incoming, :, 1]
        else:  # ymax, north
            f[incoming, :, Ny-1] = f[incoming, :, Ny-2]
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply for 3D domain"""
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        incoming = self.incoming_indices
        
        if loc in ['xmin', 'west']:
            f[incoming, 0, :, :] = f[incoming, 1, :, :]
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :, :] = f[incoming, Nx-2, :, :]
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0, :] = f[incoming, :, 1, :]
        elif loc in ['ymax', 'north']:
            f[incoming, :, Ny-1, :] = f[incoming, :, Ny-2, :]
        elif loc in ['zmin', 'bottom']:
            f[incoming, :, :, 0] = f[incoming, :, :, 1]
        else:  # zmax, top
            f[incoming, :, :, Nz-1] = f[incoming, :, :, Nz-2]