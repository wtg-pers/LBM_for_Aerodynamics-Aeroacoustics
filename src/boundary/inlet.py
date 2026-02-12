"""
Inlet Boundary Conditions for LBM

This module implements inlet boundary conditions using equilibrium
and non-equilibrium approaches.

Supports both 2D and 3D domains.

Physical Principle:
    At the inlet, we specify target velocity and density.
    - Equilibrium inlet: f_i = f_i^eq(ρ_target, u_target)
    - Non-equilibrium inlet: f_i = f_i^eq(target) + f_i^neq(extrapolated)

The non-equilibrium approach preserves viscous stress information
and provides better mass conservation.

Corner Safety:
    When set_wall_info() is called, f_neq at wall-adjacent nodes on the
    inlet face is zeroed, preventing density accumulation at corners.
    See base.py for detailed explanation.

References:
    - Latt et al., Computers & Mathematics with Applications 81, 2021
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Union, Optional
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class EquilibriumInlet(BoundaryCondition):
    """Equilibrium Inlet Boundary Condition
    
    Sets incoming distributions to equilibrium values.
    Simple but may cause mass drift with wall boundaries.
    
    Supports both 2D and 3D domains.
    
    Use NonEquilibriumInlet for better mass conservation.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 velocity: Union[float, 'npt.NDArray'] = 0.1,
                 density: float = 1.0,
                 shape: Optional[tuple] = None) -> None:
        """Initialize equilibrium inlet
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            location: Boundary location (xmin, xmax, etc.)
            velocity: Inlet velocity (scalar or array)  [Δx/Δt]
            density: Target density  [dimensionless]
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.density = density
        self.shape = shape
        self.dim = lattice.dim
        self._setup_velocity(velocity)
        self._precompute_equilibrium_coeffs()
    
    def _setup_velocity(self, velocity: Union[float, 'npt.NDArray']) -> None:
        """Setup velocity array at inlet boundary"""
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for inlet BC")
        
        # Parse shape based on dimension
        if self.dim == 2:
            Nx, Ny = self.shape
        else:
            Nx, Ny, Nz = self.shape
        
        loc = self.location.value
        
        # Determine boundary shape based on location and dimension
        if self.dim == 2:
            if loc in ['xmin', 'xmax', 'west', 'east']:
                boundary_shape = (Ny,)
                normal_dir = 0
            else:  # ymin, ymax
                boundary_shape = (Nx,)
                normal_dir = 1
        else:  # 3D
            if loc in ['xmin', 'xmax', 'west', 'east']:
                boundary_shape = (Ny, Nz)
                normal_dir = 0
            elif loc in ['ymin', 'ymax', 'south', 'north']:
                boundary_shape = (Nx, Nz)
                normal_dir = 1
            else:  # zmin, zmax
                boundary_shape = (Nx, Ny)
                normal_dir = 2
        
        self.normal_dir = normal_dir
        self.boundary_shape = boundary_shape
        
        # Create velocity array: (dim, *boundary_shape)
        if isinstance(velocity, (int, float)):
            self.u_inlet = xp.zeros((self.dim,) + boundary_shape, dtype=xp.float64)
            sign = 1 if loc in ['xmin', 'ymin', 'zmin', 'west', 'south', 'bottom'] else -1
            self.u_inlet[normal_dir] = sign * float(velocity)
        else:
            self.u_inlet = xp.asarray(velocity, dtype=xp.float64)
    
    def _precompute_equilibrium_coeffs(self) -> None:
        """Precompute coefficients for equilibrium calculation"""
        xp = self.xp
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply equilibrium inlet condition
        
        Sets incoming distributions to f_eq(ρ_target, u_target).
        No f_neq involved, so no corner safety needed.
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny) or (Q, Nx, Ny, Nz)
               Modified in-place.
        """
        xp = self.xp
        f_eq_incoming = self._compute_equilibrium_incoming()
        incoming = self.incoming_indices
        loc = self.location.value
        
        if self.dim == 2:
            Nx, Ny = f.shape[1], f.shape[2]
            
            if loc in ['xmin', 'west']:
                f[incoming, 0, :] = f_eq_incoming
            elif loc in ['xmax', 'east']:
                f[incoming, Nx-1, :] = f_eq_incoming
            elif loc in ['ymin', 'south']:
                f[incoming, :, 0] = f_eq_incoming
            elif loc in ['ymax', 'north']:
                f[incoming, :, Ny-1] = f_eq_incoming
        else:  # 3D
            Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
            
            if loc in ['xmin', 'west']:
                f[incoming, 0, :, :] = f_eq_incoming
            elif loc in ['xmax', 'east']:
                f[incoming, Nx-1, :, :] = f_eq_incoming
            elif loc in ['ymin', 'south']:
                f[incoming, :, 0, :] = f_eq_incoming
            elif loc in ['ymax', 'north']:
                f[incoming, :, Ny-1, :] = f_eq_incoming
            elif loc in ['zmin', 'bottom']:
                f[incoming, :, :, 0] = f_eq_incoming
            elif loc in ['zmax', 'top']:
                f[incoming, :, :, Nz-1] = f_eq_incoming
    
    def _compute_equilibrium_incoming(self) -> 'npt.NDArray':
        """Compute equilibrium for incoming directions"""
        xp = self.xp
        rho = self.density
        u = self.u_inlet
        
        usqr = xp.sum(u**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u)
        
        w_bc = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq = w_bc * rho * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        return f_eq

    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply equilibrium inlet for 2D domain (wall-safe)"""
        xp = self.xp
        incoming = self.incoming_indices
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        
        f_eq_incoming = self._compute_equilibrium_incoming()
        
        # Wall-safe slicing: exclude wall nodes on this face
        (fs,) = self._get_face_fluid_slices()
        
        if loc in ['xmin', 'west']:
            f[incoming, 0, fs] = f_eq_incoming[:, fs]
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, fs] = f_eq_incoming[:, fs]
        elif loc in ['ymin', 'south']:
            f[incoming, fs, 0] = f_eq_incoming[:, fs]
        elif loc in ['ymax', 'north']:
            f[incoming, fs, Ny-1] = f_eq_incoming[:, fs]
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply equilibrium inlet for 3D domain (wall-safe)"""
        xp = self.xp
        incoming = self.incoming_indices
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        
        f_eq_incoming = self._compute_equilibrium_incoming()
        
        # Wall-safe slicing: exclude wall nodes on this face
        (fs1, fs2) = self._get_face_fluid_slices()
        
        if loc in ['xmin', 'west']:
            f[incoming, 0, fs1, fs2] = f_eq_incoming[:, fs1, fs2]
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, fs1, fs2] = f_eq_incoming[:, fs1, fs2]
        elif loc in ['ymin', 'south']:
            f[incoming, fs1, 0, fs2] = f_eq_incoming[:, fs1, fs2]
        elif loc in ['ymax', 'north']:
            f[incoming, fs1, Ny-1, fs2] = f_eq_incoming[:, fs1, fs2]
        elif loc in ['zmin', 'bottom']:
            f[incoming, fs1, fs2, 0] = f_eq_incoming[:, fs1, fs2]
        elif loc in ['zmax', 'top']:
            f[incoming, fs1, fs2, Nz-1] = f_eq_incoming[:, fs1, fs2]


class NonEquilibriumInlet(BoundaryCondition):
    """Non-Equilibrium Extrapolation Inlet
    
    Preserves non-equilibrium part for better mass conservation:
        f_i(inlet) = f_i^eq(ρ_target, u_target) + f_i^neq(extrapolated)
    
    The non-equilibrium part is extrapolated from interior nodes,
    carrying viscous stress information.
    
    Corner Safety:
        When set_wall_info() is called, wall-adjacent nodes on the inlet
        face have their f_neq zeroed. This prevents density accumulation
        at corners where the inlet meets wall boundaries.
        
        Fluid nodes:         f = f_eq(target) + f_neq(interior)
        Wall-adjacent nodes: f = f_eq(target)  [pure equilibrium]
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 velocity: Union[float, 'npt.NDArray'] = 0.1,
                 density: float = 1.0,
                 shape: Optional[tuple] = None) -> None:
        """Initialize non-equilibrium inlet
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            location: Boundary location (xmin, xmax, etc.)
            velocity: Inlet velocity  [Δx/Δt]
            density: Target density  [dimensionless]
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.density = density
        self.shape = shape
        self.dim = lattice.dim
        self._setup_velocity(velocity)
        self._precompute_coeffs()
    
    def _setup_velocity(self, velocity: Union[float, 'npt.NDArray']) -> None:
        """Setup velocity array at inlet"""
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for inlet BC")
        
        # Parse shape based on dimension
        if self.dim == 2:
            Nx, Ny = self.shape
        else:
            Nx, Ny, Nz = self.shape
        
        loc = self.location.value
        
        # Determine boundary shape
        if self.dim == 2:
            if loc in ['xmin', 'xmax', 'west', 'east']:
                boundary_shape = (Ny,)
                normal_dir = 0
            else:
                boundary_shape = (Nx,)
                normal_dir = 1
        else:
            if loc in ['xmin', 'xmax', 'west', 'east']:
                boundary_shape = (Ny, Nz)
                normal_dir = 0
            elif loc in ['ymin', 'ymax', 'south', 'north']:
                boundary_shape = (Nx, Nz)
                normal_dir = 1
            else:
                boundary_shape = (Nx, Ny)
                normal_dir = 2
        
        self.normal_dir = normal_dir
        self.boundary_shape = boundary_shape
        
        # Create velocity array
        if isinstance(velocity, (int, float)):
            self.u_inlet = xp.zeros((self.dim,) + boundary_shape, dtype=xp.float64)
            sign = 1 if loc in ['xmin', 'ymin', 'zmin', 'west', 'south', 'bottom'] else -1
            self.u_inlet[normal_dir] = sign * float(velocity)
        else:
            self.u_inlet = xp.asarray(velocity, dtype=xp.float64)
    
    def _precompute_coeffs(self) -> None:
        """Precompute coefficients"""
        xp = self.xp
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
        
        # All directions for equilibrium computation
        self.c_all = self.c.astype(xp.float64)
        self.w_all = self.w.astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply non-equilibrium inlet condition
        
        Algorithm:
        1. Get distribution at interior node (one node inside from boundary)
        2. Compute macroscopic at interior → ρ_int, u_int
        3. Compute f_eq at interior → f_neq = f - f_eq
        4. ★ Sanitize f_neq: zero at wall-adjacent nodes (corner safety)
        5. Set inlet: f = f_eq(target) + f_neq(sanitized)
        
        Args:
            f: Distribution function, modified in-place
        """
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply non-equilibrium inlet for 2D domain (wall-safe)
        
        Only applies to fluid nodes; wall nodes are left to domain_wall BC.
        """
        xp = self.xp
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        
        # Wall-safe slicing
        (fs,) = self._get_face_fluid_slices()
        
        # Get interior slice (fluid nodes only)
        if loc in ['xmin', 'west']:
            f_interior = f[:, 1, fs]
        elif loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, fs]
        elif loc in ['ymin', 'south']:
            f_interior = f[:, fs, 1]
        else:  # ymax, north
            f_interior = f[:, fs, Ny-2]
        
        # Compute macroscopic at interior (fluid only)
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # Compute equilibrium at interior
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # Non-equilibrium at interior
        f_neq_int = f_interior - f_eq_int
        
        # Equilibrium at boundary (target values, fluid nodes only)
        rho_target = self.density
        u_target = self.u_inlet[:, fs]  # slice velocity to match fluid region
        
        usqr_target = xp.sum(u_target**2, axis=0)
        cu_target = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*(u_target.ndim - 1))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu_target + 4.5*(cu_target**2) - 1.5*usqr_target)
        
        # f = f_eq(target) + f_neq(interior)
        incoming = self.incoming_indices
        f_incoming = f_eq_target + f_neq_int[incoming]
        
        # Apply only to fluid nodes
        if loc in ['xmin', 'west']:
            f[incoming, 0, fs] = f_incoming
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, fs] = f_incoming
        elif loc in ['ymin', 'south']:
            f[incoming, fs, 0] = f_incoming
        else:  # ymax, north
            f[incoming, fs, Ny-1] = f_incoming
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply non-equilibrium inlet for 3D domain (wall-safe)
        
        Only applies to fluid nodes; wall nodes are left to domain_wall BC.
        """
        xp = self.xp
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        
        # Wall-safe slicing: (ys, zs) for xmin/xmax, etc.
        (fs1, fs2) = self._get_face_fluid_slices()
        
        # Get interior slice (fluid nodes only)
        if loc in ['xmin', 'west']:
            f_interior = f[:, 1, fs1, fs2]
        elif loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, fs1, fs2]
        elif loc in ['ymin', 'south']:
            f_interior = f[:, fs1, 1, fs2]
        elif loc in ['ymax', 'north']:
            f_interior = f[:, fs1, Ny-2, fs2]
        elif loc in ['zmin', 'bottom']:
            f_interior = f[:, fs1, fs2, 1]
        else:  # zmax, top
            f_interior = f[:, fs1, fs2, Nz-2]
        
        # Compute macroscopic at interior (fluid only)
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # Compute equilibrium at interior
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # Non-equilibrium at interior
        f_neq_int = f_interior - f_eq_int
        
        # Equilibrium at boundary (target values, fluid nodes only)
        rho_target = self.density
        u_target = self.u_inlet[:, fs1, fs2]  # slice velocity to match fluid region
        
        usqr_target = xp.sum(u_target**2, axis=0)
        cu_target = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*(u_target.ndim - 1))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu_target + 4.5*(cu_target**2) - 1.5*usqr_target)
        
        # f = f_eq(target) + f_neq(interior)
        incoming = self.incoming_indices
        f_incoming = f_eq_target + f_neq_int[incoming]
        
        # Apply only to fluid nodes
        if loc in ['xmin', 'west']:
            f[incoming, 0, fs1, fs2] = f_incoming
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, fs1, fs2] = f_incoming
        elif loc in ['ymin', 'south']:
            f[incoming, fs1, 0, fs2] = f_incoming
        elif loc in ['ymax', 'north']:
            f[incoming, fs1, Ny-1, fs2] = f_incoming
        elif loc in ['zmin', 'bottom']:
            f[incoming, fs1, fs2, 0] = f_incoming
        else:  # zmax, top
            f[incoming, fs1, fs2, Nz-1] = f_incoming


# Alias for backward compatibility
EquilibriumInletProfile = EquilibriumInlet