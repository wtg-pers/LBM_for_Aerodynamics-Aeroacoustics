"""
Farfield/Freestream Boundary Conditions for LBM

This module implements non-reflecting boundary conditions suitable
for external aerodynamic simulations.

Supports both 2D and 3D simulations.

Available Methods:
    - FreestreamBC: Non-equilibrium extrapolation with pressure relaxation
    - SpongeLayer: Gradual damping toward freestream state
    - AmbientBC: Zero-gradient with minimal reflection

Physical Principle:
    These BCs aim to minimize wave reflections at artificial boundaries
    by smoothly transitioning the solution to freestream conditions.

References:
    - Latt & Chopard, Mathematics and Computers in Simulation 72, 2006
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Union, Tuple
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class FreestreamBC(BoundaryCondition):
    """Freestream Boundary Condition with Pressure Relaxation
    
    Non-equilibrium extrapolation method that:
    - Fixes velocity to freestream value
    - Relaxes pressure/density toward freestream
    - Preserves non-equilibrium (viscous stress) from interior
    
    Supports both 2D and 3D domains.
    
    Algorithm:
        ρ_bc = ρ∞ + (1-K)(ρ_int - ρ∞)
        u_bc = U∞ (fixed)
        f = f^eq(ρ_bc, U∞) + f^neq(interior)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 u_inf: Union[float, Tuple] = 0.1,
                 rho_inf: float = 1.0,
                 relax_coeff: float = 0.1,
                 shape: Optional[tuple] = None) -> None:
        """Initialize freestream BC
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            location: Boundary location
            u_inf: Freestream velocity (scalar or tuple)  [Δx/Δt]
            rho_inf: Freestream density  [dimensionless]
            relax_coeff: Pressure relaxation coefficient K (0-1)
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_inf = rho_inf
        self.K = relax_coeff
        self.shape = shape
        self.dim = lattice.dim
        
        self._setup_velocity(u_inf)
        self._precompute_coefficients()
    
    def _setup_velocity(self, u_inf: Union[float, Tuple]) -> None:
        """Setup freestream velocity array"""
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for FreestreamBC")
        
        loc = self.location.value
        
        # Determine boundary shape based on dimension and location
        if self.dim == 2:
            Nx, Ny = self.shape
            if loc in ['xmin', 'xmax', 'west', 'east']:
                boundary_shape = (Ny,)
            else:
                boundary_shape = (Nx,)
        else:
            Nx, Ny, Nz = self.shape
            if loc in ['xmin', 'xmax', 'west', 'east']:
                boundary_shape = (Ny, Nz)
            elif loc in ['ymin', 'ymax', 'south', 'north']:
                boundary_shape = (Nx, Nz)
            else:
                boundary_shape = (Nx, Ny)
        
        self.boundary_shape = boundary_shape
        
        # Create velocity array
        self.u_inf_array = xp.zeros((self.dim,) + boundary_shape, dtype=xp.float64)
        
        if isinstance(u_inf, (int, float)):
            # Scalar → assume flow in x-direction
            self.u_inf_array[0, ...] = float(u_inf)
        else:
            # Tuple
            for d in range(min(len(u_inf), self.dim)):
                self.u_inf_array[d, ...] = float(u_inf[d])
    
    def _precompute_coefficients(self) -> None:
        """Precompute equilibrium coefficients"""
        xp = self.xp
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
        self.c_all = self.c.astype(xp.float64)
        self.w_all = self.w.astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply freestream boundary condition"""
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply for 2D domain"""
        xp = self.xp
        Nx, Ny = f.shape[1], f.shape[2]
        loc = self.location.value
        
        # Get interior distribution
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
        
        f_neq_int = f_interior - f_eq_int
        
        # Target: pressure relaxation, fixed velocity
        rho_target = self.rho_inf + (1 - self.K) * (rho_int - self.rho_inf)
        u_target = self.u_inf_array
        
        # f^eq at boundary for incoming directions
        usqr = xp.sum(u_target**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        incoming = self.incoming_indices
        f_incoming = f_eq_target + f_neq_int[incoming]
        
        # Apply
        if loc in ['xmin', 'west']:
            f[incoming, 0, :] = f_incoming
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :] = f_incoming
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0] = f_incoming
        else:
            f[incoming, :, Ny-1] = f_incoming
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply for 3D domain"""
        xp = self.xp
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        loc = self.location.value
        
        # Get interior distribution
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
        
        f_neq_int = f_interior - f_eq_int
        
        # Target
        rho_target = self.rho_inf + (1 - self.K) * (rho_int - self.rho_inf)
        u_target = self.u_inf_array
        
        # f^eq at boundary
        usqr = xp.sum(u_target**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        incoming = self.incoming_indices
        f_incoming = f_eq_target + f_neq_int[incoming]
        
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
        else:
            f[incoming, :, :, Nz-1] = f_incoming
    
    def get_info(self) -> str:
        """Return BC information string"""
        dim_str = "2D" if self.dim == 2 else "3D"
        return (f"FreestreamBC ({dim_str}) at {self.location.value}:\n"
                f"    ρ∞={self.rho_inf}, K={self.K}")


class AmbientBC(BoundaryCondition):
    """Ambient (Zero-Gradient) Boundary Condition
    
    Simple extrapolation with optional pressure correction.
    Extrapolates velocity from interior, applies target pressure.
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 rho_ambient: float = 1.0,
                 shape: Optional[tuple] = None) -> None:
        """Initialize ambient BC
        
        Args:
            xp: Array module
            lattice: Lattice model
            location: Boundary location
            rho_ambient: Target ambient density  [dimensionless]
            shape: Domain shape  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_ambient = rho_ambient
        self.shape = shape
        self.dim = lattice.dim
        
        self._precompute_coefficients()
    
    def _precompute_coefficients(self) -> None:
        """Precompute coefficients"""
        xp = self.xp
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
        self.c_all = self.c.astype(xp.float64)
        self.w_all = self.w.astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply ambient BC"""
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply for 2D domain"""
        xp = self.xp
        Nx, Ny = f.shape[1], f.shape[2]
        loc = self.location.value
        
        # Get interior distribution
        if loc in ['xmin', 'west']:
            f_interior = f[:, 1, :]
        elif loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, :]
        elif loc in ['ymin', 'south']:
            f_interior = f[:, :, 1]
        else:
            f_interior = f[:, :, Ny-2]
        
        # Extrapolate velocity from interior
        rho_int = xp.sum(f_interior, axis=0)
        u_ext = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # Use ambient density
        rho_bc = self.rho_ambient
        
        # Compute equilibrium
        usqr = xp.sum(u_ext**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_ext)
        
        boundary_shape = f_interior.shape[1:]
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(boundary_shape))
        f_eq = w_in * rho_bc * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        incoming = self.incoming_indices
        
        if loc in ['xmin', 'west']:
            f[incoming, 0, :] = f_eq
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :] = f_eq
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0] = f_eq
        else:
            f[incoming, :, Ny-1] = f_eq
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply for 3D domain"""
        xp = self.xp
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        loc = self.location.value
        
        # Get interior distribution
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
        else:
            f_interior = f[:, :, :, Nz-2]
        
        # Extrapolate velocity
        rho_int = xp.sum(f_interior, axis=0)
        u_ext = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        rho_bc = self.rho_ambient
        
        usqr = xp.sum(u_ext**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_ext)
        
        boundary_shape = f_interior.shape[1:]
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(boundary_shape))
        f_eq = w_in * rho_bc * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        incoming = self.incoming_indices
        
        if loc in ['xmin', 'west']:
            f[incoming, 0, :, :] = f_eq
        elif loc in ['xmax', 'east']:
            f[incoming, Nx-1, :, :] = f_eq
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0, :] = f_eq
        elif loc in ['ymax', 'north']:
            f[incoming, :, Ny-1, :] = f_eq
        elif loc in ['zmin', 'bottom']:
            f[incoming, :, :, 0] = f_eq
        else:
            f[incoming, :, :, Nz-1] = f_eq
    
    def get_info(self) -> str:
        """Return BC information string"""
        dim_str = "2D" if self.dim == 2 else "3D"
        return f"AmbientBC ({dim_str}) at {self.location.value}: ρ_ambient={self.rho_ambient}"


class SpongeLayer(BoundaryCondition):
    """Sponge Layer Boundary Condition
    
    Gradually damps the solution toward freestream state
    over a specified thickness.
    
    Supports both 2D and 3D domains.
    
    Algorithm:
        f = (1 - σ) · f_computed + σ · f_target
        
    where σ increases from 0 to σ_max over the sponge thickness.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 thickness: int = 20,
                 sigma_max: float = 0.5,
                 u_inf: Union[float, Tuple] = 0.1,
                 rho_inf: float = 1.0,
                 profile: str = 'quadratic',
                 shape: Optional[tuple] = None) -> None:
        """Initialize sponge layer
        
        Args:
            xp: Array module
            lattice: Lattice model
            location: Boundary location
            thickness: Sponge layer thickness  [Δx]
            sigma_max: Maximum damping coefficient (0-1)
            u_inf: Freestream velocity  [Δx/Δt]
            rho_inf: Freestream density
            profile: Damping profile ('linear', 'quadratic', 'exponential')
            shape: Domain shape  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.thickness = thickness
        self.sigma_max = sigma_max
        self.rho_inf = rho_inf
        self.profile = profile
        self.shape = shape
        self.dim = lattice.dim
        
        self._setup_velocity(u_inf)
        self._setup_damping_profile()
    
    def _setup_velocity(self, u_inf: Union[float, Tuple]) -> None:
        """Setup freestream velocity"""
        xp = self.xp
        self.u_inf = xp.zeros(self.dim, dtype=xp.float64)
        
        if isinstance(u_inf, (int, float)):
            self.u_inf[0] = float(u_inf)
        else:
            for d in range(min(len(u_inf), self.dim)):
                self.u_inf[d] = float(u_inf[d])
    
    def _setup_damping_profile(self) -> None:
        """Setup damping coefficient profile σ(x)"""
        xp = self.xp
        loc = self.location.value
        
        if self.shape is None:
            raise ValueError("shape must be provided for SpongeLayer")
        
        # Determine axis and create coordinate
        if loc in ['xmin', 'xmax', 'west', 'east']:
            self.axis = 0
            N = self.shape[0]
        elif loc in ['ymin', 'ymax', 'south', 'north']:
            self.axis = 1
            N = self.shape[1]
        else:
            self.axis = 2
            N = self.shape[2]
        
        # Create sponge region slice and damping profile
        t = self.thickness
        
        if loc in ['xmin', 'ymin', 'zmin', 'west', 'south', 'bottom']:
            self.sponge_slice = slice(0, t)
            xi = xp.linspace(1.0, 0.0, t)
        else:
            self.sponge_slice = slice(N - t, N)
            xi = xp.linspace(0.0, 1.0, t)
        
        # Compute damping profile
        self.sigma = self._compute_profile(xi)
        
        # Reshape for broadcasting
        if self.dim == 2:
            if self.axis == 0:
                self.sigma = self.sigma.reshape(-1, 1)
            else:
                self.sigma = self.sigma.reshape(1, -1)
        else:
            if self.axis == 0:
                self.sigma = self.sigma.reshape(-1, 1, 1)
            elif self.axis == 1:
                self.sigma = self.sigma.reshape(1, -1, 1)
            else:
                self.sigma = self.sigma.reshape(1, 1, -1)
    
    def _compute_profile(self, xi: 'npt.NDArray') -> 'npt.NDArray':
        """Compute damping coefficient from normalized coordinate"""
        if self.profile == 'linear':
            g = xi
        elif self.profile == 'quadratic':
            g = xi ** 2
        elif self.profile == 'polynomial':
            g = 3 * xi**2 - 2 * xi**3
        elif self.profile == 'exponential':
            g = 1 - self.xp.exp(-5 * xi)
        else:
            raise ValueError(f"Unknown profile: {self.profile}")
        
        return self.sigma_max * g
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply sponge layer damping"""
        xp = self.xp
        
        # Compute target equilibrium
        rho_target = self.rho_inf
        u_target = self.u_inf
        
        usqr = float(xp.sum(u_target**2))
        cu = xp.einsum('d,di->i', u_target, self.c.astype(xp.float64))
        f_target = self.w * rho_target * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        # Reshape for broadcasting
        if self.dim == 2:
            f_target = f_target.reshape(-1, 1, 1)
        else:
            f_target = f_target.reshape(-1, 1, 1, 1)
        
        # Extract sponge region
        if self.dim == 2:
            if self.axis == 0:
                sponge_region = (slice(None), self.sponge_slice, slice(None))
            else:
                sponge_region = (slice(None), slice(None), self.sponge_slice)
        else:
            if self.axis == 0:
                sponge_region = (slice(None), self.sponge_slice, slice(None), slice(None))
            elif self.axis == 1:
                sponge_region = (slice(None), slice(None), self.sponge_slice, slice(None))
            else:
                sponge_region = (slice(None), slice(None), slice(None), self.sponge_slice)
        
        # Apply damping
        f[sponge_region] = (1 - self.sigma) * f[sponge_region] + self.sigma * f_target
    
    def get_info(self) -> str:
        """Return BC information string"""
        dim_str = "2D" if self.dim == 2 else "3D"
        return (f"SpongeLayer ({dim_str}) at {self.location.value}:\n"
                f"    thickness={self.thickness}, σ_max={self.sigma_max}\n"
                f"    profile={self.profile}")


# Backward compatibility alias
class CharacteristicFarfield(FreestreamBC):
    """DEPRECATED: Use FreestreamBC instead."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "CharacteristicFarfield is deprecated. Use FreestreamBC instead.",
            DeprecationWarning, stacklevel=2
        )
        super().__init__(*args, **kwargs)