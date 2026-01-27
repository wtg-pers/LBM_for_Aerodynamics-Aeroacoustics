"""
Equilibrium Inlet Boundary Condition for LBM

This module implements a simple but robust inlet boundary condition
that sets unknown distributions to their equilibrium values.

Physical Principle:
    At the inlet, we specify a target velocity u_target and density ρ_target.
    The unknown (incoming) distributions are set to equilibrium:
    
        f_i(x_inlet) = f_i^eq(ρ_target, u_target)  for incoming directions
    
    Known distributions (from inside the domain) are left unchanged.

Advantages:
    - Exact mass and momentum specification at boundary
    - Simple implementation with clear physical meaning
    - Numerically stable
    - No over-determined system issues (unlike Zou-He in 3D)

Disadvantages:
    - Non-equilibrium (viscous) information is lost at boundary
    - May cause slight discontinuity at inlet
    - Not suitable for fully-developed flow inlet

Best Used For:
    - Far-field boundaries
    - Uniform flow inlets
    - Cases where inlet is far from region of interest

Author: LBM Development Team
Date: 2026-01
"""

from typing import TYPE_CHECKING, Union, Callable, Optional
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class EquilibriumInlet(BoundaryCondition):
    """Equilibrium Inlet Boundary Condition
    
    Sets incoming distributions at the inlet to equilibrium values
    based on specified velocity and density.
    
    The equilibrium distribution is:
        f_i^eq = w_i * ρ * (1 + 3(c_i·u) + 4.5(c_i·u)² - 1.5|u|²)
    
    Attributes:
        velocity: Target velocity at inlet (can be scalar, array, or function)
        density: Target density at inlet (default 1.0)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 velocity: Union[float, 'npt.NDArray', Callable],
                 density: float = 1.0,
                 shape: Optional[tuple] = None) -> None:
        """Initialize equilibrium inlet
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            location: Boundary location (WEST, EAST, etc.)
            velocity: Inlet velocity, can be:
                     - float: uniform velocity in normal direction
                     - array: velocity field at boundary (dim, Ny, Nz) for WEST
                     - callable: function(y, z) -> velocity
            density: Target density (default 1.0)
            shape: Domain shape (Nx, Ny, Nz), required for array allocation
        """
        super().__init__(xp, lattice, location)
        
        self.density = density
        self.shape = shape
        
        # Process velocity input
        self._setup_velocity(velocity)
        
        # Precompute equilibrium coefficients for efficiency
        self._precompute_equilibrium_coeffs()
    
    def _setup_velocity(self, velocity: Union[float, 'npt.NDArray', Callable]) -> None:
        """Setup velocity field at inlet"""
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for inlet BC")
        
        Nx, Ny, Nz = self.shape
        
        # Determine boundary shape based on location
        if self.location in [BoundaryLocation.WEST, BoundaryLocation.EAST]:
            boundary_shape = (Ny, Nz)
            normal_dir = 0  # x-direction
        elif self.location in [BoundaryLocation.SOUTH, BoundaryLocation.NORTH]:
            boundary_shape = (Nx, Nz)
            normal_dir = 1  # y-direction
        else:  # TOP, BOTTOM
            boundary_shape = (Nx, Ny)
            normal_dir = 2  # z-direction
        
        self.normal_dir = normal_dir
        self.boundary_shape = boundary_shape
        
        # Create velocity array (3, *boundary_shape)
        if callable(velocity):
            # Function input - evaluate on grid
            self.u_inlet = self._evaluate_velocity_function(velocity, boundary_shape)
        elif isinstance(velocity, (int, float)):
            # Scalar input - uniform velocity in normal direction
            self.u_inlet = xp.zeros((3,) + boundary_shape, dtype=xp.float64)
            # Set velocity in the direction INTO the domain
            sign = 1 if self.location in [BoundaryLocation.WEST, 
                                           BoundaryLocation.SOUTH, 
                                           BoundaryLocation.BOTTOM] else -1
            self.u_inlet[normal_dir] = sign * float(velocity)
        else:
            # Array input - use directly
            self.u_inlet = xp.asarray(velocity, dtype=xp.float64)
            if self.u_inlet.shape != (3,) + boundary_shape:
                raise ValueError(f"Velocity array shape {self.u_inlet.shape} "
                               f"doesn't match expected {(3,) + boundary_shape}")
    
    def _evaluate_velocity_function(self, func: Callable, 
                                    shape: tuple) -> 'npt.NDArray':
        """Evaluate velocity function on boundary grid"""
        xp = self.xp
        
        # Create coordinate arrays based on boundary orientation
        if self.location in [BoundaryLocation.WEST, BoundaryLocation.EAST]:
            # Boundary is (Ny, Nz) shaped
            Ny, Nz = shape
            y = xp.arange(Ny, dtype=xp.float64)
            z = xp.arange(Nz, dtype=xp.float64)
            Y, Z = xp.meshgrid(y, z, indexing='ij')
            u = func(Y, Z)
        elif self.location in [BoundaryLocation.SOUTH, BoundaryLocation.NORTH]:
            Nx, Nz = shape
            x = xp.arange(Nx, dtype=xp.float64)
            z = xp.arange(Nz, dtype=xp.float64)
            X, Z = xp.meshgrid(x, z, indexing='ij')
            u = func(X, Z)
        else:
            Nx, Ny = shape
            x = xp.arange(Nx, dtype=xp.float64)
            y = xp.arange(Ny, dtype=xp.float64)
            X, Y = xp.meshgrid(x, y, indexing='ij')
            u = func(X, Y)
        
        return xp.asarray(u, dtype=xp.float64)
    
    def _precompute_equilibrium_coeffs(self) -> None:
        """Precompute equilibrium calculation coefficients for incoming directions"""
        xp = self.xp
        
        # Only compute for incoming directions
        incoming = self.incoming_indices
        
        # Get velocity vectors for incoming directions: (3, n_incoming)
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        
        # Get weights for incoming directions: (n_incoming,)
        self.w_incoming = self.w[incoming].astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply equilibrium inlet condition
        
        Sets f_i = f_i^eq(ρ_target, u_target) for incoming directions at inlet.
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny, Nz)
        """
        xp = self.xp
        
        # Get boundary slice
        Nx, Ny, Nz = f.shape[1:]
        
        # Compute equilibrium for incoming directions
        f_eq_incoming = self._compute_equilibrium_incoming()
        
        # Apply to distribution function
        # Index into f for incoming directions at boundary
        incoming = self.incoming_indices
        
        if self.location == BoundaryLocation.WEST:
            f[incoming, 0, :, :] = f_eq_incoming
        elif self.location == BoundaryLocation.EAST:
            f[incoming, Nx-1, :, :] = f_eq_incoming
        elif self.location == BoundaryLocation.SOUTH:
            f[incoming, :, 0, :] = f_eq_incoming
        elif self.location == BoundaryLocation.NORTH:
            f[incoming, :, Ny-1, :] = f_eq_incoming
        elif self.location == BoundaryLocation.BOTTOM:
            f[incoming, :, :, 0] = f_eq_incoming
        elif self.location == BoundaryLocation.TOP:
            f[incoming, :, :, Nz-1] = f_eq_incoming
    
    def _compute_equilibrium_incoming(self) -> 'npt.NDArray':
        """Compute equilibrium distributions for incoming directions
        
        Returns:
            f_eq: Equilibrium for incoming directions, shape (n_incoming, *boundary_shape)
        """
        xp = self.xp
        
        rho = self.density
        u = self.u_inlet  # (3, *boundary_shape)
        
        # |u|² for equilibrium formula
        usqr = xp.sum(u**2, axis=0)  # (*boundary_shape,)
        
        # c·u for each incoming direction
        # c_incoming: (3, n_incoming), u: (3, *boundary_shape)
        # Result: (n_incoming, *boundary_shape)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u)
        
        # Equilibrium: f_i^eq = w_i * ρ * (1 + 3*cu + 4.5*cu² - 1.5*usqr)
        # w_incoming: (n_incoming,) -> broadcast to (n_incoming, 1, 1)
        w_bc = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        
        f_eq = w_bc * rho * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        return f_eq
    
    def set_velocity(self, velocity: Union[float, 'npt.NDArray']) -> None:
        """Update inlet velocity (for time-varying BCs)
        
        Args:
            velocity: New velocity (scalar or array)
        """
        self._setup_velocity(velocity)


class EquilibriumInletProfile(EquilibriumInlet):
    """Equilibrium Inlet with Parabolic (Poiseuille) Profile
    
    Implements a parabolic velocity profile:
        u(y, z) = u_max * (1 - (r/R)²)
    
    where r is distance from center and R is the half-width.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 u_max: float,
                 shape: tuple,
                 density: float = 1.0) -> None:
        """Initialize parabolic inlet
        
        Args:
            xp: Array module
            lattice: Lattice model
            location: Boundary location
            u_max: Maximum (centerline) velocity
            shape: Domain shape (Nx, Ny, Nz)
            density: Target density
        """
        Nx, Ny, Nz = shape
        
        # Create parabolic profile function
        if location in [BoundaryLocation.WEST, BoundaryLocation.EAST]:
            # Flow in x-direction, profile in y-z plane
            def profile(Y, Z):
                u = xp.zeros((3, Y.shape[0], Y.shape[1]), dtype=xp.float64)
                # Parabolic in y and z
                y_center = (Ny - 1) / 2.0
                z_center = (Nz - 1) / 2.0
                R_y = (Ny - 1) / 2.0
                R_z = (Nz - 1) / 2.0
                
                r_y = (Y - y_center) / R_y
                r_z = (Z - z_center) / R_z
                
                # Parabolic profile: u = u_max * (1 - r²)
                profile_val = u_max * (1 - r_y**2) * (1 - r_z**2)
                profile_val = xp.maximum(profile_val, 0)  # Clip negative values
                
                u[0] = profile_val if location == BoundaryLocation.WEST else -profile_val
                return u
            
            velocity = profile
        else:
            raise NotImplementedError("Parabolic profile only implemented for WEST/EAST")
        
        super().__init__(xp, lattice, location, velocity, density, shape)