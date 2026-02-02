# """
# Equilibrium Inlet Boundary Condition for LBM

# This module implements a simple but robust inlet boundary condition
# that sets unknown distributions to their equilibrium values.

# Physical Principle:
#     At the inlet, we specify a target velocity u_target and density ρ_target.
#     The unknown (incoming) distributions are set to equilibrium:
    
#         f_i(x_inlet) = f_i^eq(ρ_target, u_target)  for incoming directions
    
#     Known distributions (from inside the domain) are left unchanged.

# Advantages:
#     - Exact mass and momentum specification at boundary
#     - Simple implementation with clear physical meaning
#     - Numerically stable
#     - No over-determined system issues (unlike Zou-He in 3D)

# Disadvantages:
#     - Non-equilibrium (viscous) information is lost at boundary
#     - May cause slight discontinuity at inlet
#     - Not suitable for fully-developed flow inlet

# Best Used For:
#     - Far-field boundaries
#     - Uniform flow inlets
#     - Cases where inlet is far from region of interest

# Author: LBM Development Team
# Date: 2026-01
# """

# from typing import TYPE_CHECKING, Union, Callable, Optional
# import numpy as np

# if TYPE_CHECKING:
#     from types import ModuleType
#     import numpy.typing as npt

# from .base import BoundaryCondition, BoundaryLocation


# class EquilibriumInlet(BoundaryCondition):
#     """Equilibrium Inlet Boundary Condition
    
#     Sets incoming distributions at the inlet to equilibrium values
#     based on specified velocity and density.
    
#     The equilibrium distribution is:
#         f_i^eq = w_i * ρ * (1 + 3(c_i·u) + 4.5(c_i·u)² - 1.5|u|²)
    
#     Attributes:
#         velocity: Target velocity at inlet (can be scalar, array, or function)
#         density: Target density at inlet (default 1.0)
#     """
    
#     def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
#                  location: BoundaryLocation,
#                  velocity: Union[float, 'npt.NDArray', Callable],
#                  density: float = 1.0,
#                  shape: Optional[tuple] = None) -> None:
#         """Initialize equilibrium inlet
        
#         Args:
#             xp: Array module (numpy or cupy)
#             lattice: Lattice model
#             location: Boundary location (WEST, EAST, etc.)
#             velocity: Inlet velocity, can be:
#                      - float: uniform velocity in normal direction
#                      - array: velocity field at boundary (dim, Ny, Nz) for WEST
#                      - callable: function(y, z) -> velocity
#             density: Target density (default 1.0)
#             shape: Domain shape (Nx, Ny, Nz), required for array allocation
#         """
#         super().__init__(xp, lattice, location)
        
#         self.density = density
#         self.shape = shape
        
#         # Process velocity input
#         self._setup_velocity(velocity)
        
#         # Precompute equilibrium coefficients for efficiency
#         self._precompute_equilibrium_coeffs()
    
#     def _setup_velocity(self, velocity: Union[float, 'npt.NDArray', Callable]) -> None:
#         """Setup velocity field at inlet"""
#         xp = self.xp
        
#         if self.shape is None:
#             raise ValueError("shape must be provided for inlet BC")
        
#         Nx, Ny, Nz = self.shape
        
#         # Determine boundary shape based on location
#         if self.location in [BoundaryLocation.WEST, BoundaryLocation.EAST]:
#             boundary_shape = (Ny, Nz)
#             normal_dir = 0  # x-direction
#         elif self.location in [BoundaryLocation.SOUTH, BoundaryLocation.NORTH]:
#             boundary_shape = (Nx, Nz)
#             normal_dir = 1  # y-direction
#         else:  # TOP, BOTTOM
#             boundary_shape = (Nx, Ny)
#             normal_dir = 2  # z-direction
        
#         self.normal_dir = normal_dir
#         self.boundary_shape = boundary_shape
        
#         # Create velocity array (3, *boundary_shape)
#         if callable(velocity):
#             # Function input - evaluate on grid
#             self.u_inlet = self._evaluate_velocity_function(velocity, boundary_shape)
#         elif isinstance(velocity, (int, float)):
#             # Scalar input - uniform velocity in normal direction
#             self.u_inlet = xp.zeros((3,) + boundary_shape, dtype=xp.float64)
#             # Set velocity in the direction INTO the domain
#             sign = 1 if self.location in [BoundaryLocation.WEST, 
#                                            BoundaryLocation.SOUTH, 
#                                            BoundaryLocation.BOTTOM] else -1
#             self.u_inlet[normal_dir] = sign * float(velocity)
#         else:
#             # Array input - use directly
#             self.u_inlet = xp.asarray(velocity, dtype=xp.float64)
#             if self.u_inlet.shape != (3,) + boundary_shape:
#                 raise ValueError(f"Velocity array shape {self.u_inlet.shape} "
#                                f"doesn't match expected {(3,) + boundary_shape}")
    
#     def _evaluate_velocity_function(self, func: Callable, 
#                                     shape: tuple) -> 'npt.NDArray':
#         """Evaluate velocity function on boundary grid"""
#         xp = self.xp
        
#         # Create coordinate arrays based on boundary orientation
#         if self.location in [BoundaryLocation.WEST, BoundaryLocation.EAST]:
#             # Boundary is (Ny, Nz) shaped
#             Ny, Nz = shape
#             y = xp.arange(Ny, dtype=xp.float64)
#             z = xp.arange(Nz, dtype=xp.float64)
#             Y, Z = xp.meshgrid(y, z, indexing='ij')
#             u = func(Y, Z)
#         elif self.location in [BoundaryLocation.SOUTH, BoundaryLocation.NORTH]:
#             Nx, Nz = shape
#             x = xp.arange(Nx, dtype=xp.float64)
#             z = xp.arange(Nz, dtype=xp.float64)
#             X, Z = xp.meshgrid(x, z, indexing='ij')
#             u = func(X, Z)
#         else:
#             Nx, Ny = shape
#             x = xp.arange(Nx, dtype=xp.float64)
#             y = xp.arange(Ny, dtype=xp.float64)
#             X, Y = xp.meshgrid(x, y, indexing='ij')
#             u = func(X, Y)
        
#         return xp.asarray(u, dtype=xp.float64)
    
#     def _precompute_equilibrium_coeffs(self) -> None:
#         """Precompute equilibrium calculation coefficients for incoming directions"""
#         xp = self.xp
        
#         # Only compute for incoming directions
#         incoming = self.incoming_indices
        
#         # Get velocity vectors for incoming directions: (3, n_incoming)
#         self.c_incoming = self.c[:, incoming].astype(xp.float64)
        
#         # Get weights for incoming directions: (n_incoming,)
#         self.w_incoming = self.w[incoming].astype(xp.float64)
    
#     def apply(self, f: 'npt.NDArray', **kwargs) -> None:
#         """Apply equilibrium inlet condition
        
#         Sets f_i = f_i^eq(ρ_target, u_target) for incoming directions at inlet.
        
#         Args:
#             f: Distribution function, shape (Q, Nx, Ny, Nz)
#         """
#         xp = self.xp
        
#         # Get boundary slice
#         Nx, Ny, Nz = f.shape[1:]
        
#         # Compute equilibrium for incoming directions
#         f_eq_incoming = self._compute_equilibrium_incoming()
        
#         # Apply to distribution function
#         # Index into f for incoming directions at boundary
#         incoming = self.incoming_indices
        
#         if self.location == BoundaryLocation.WEST:
#             f[incoming, 0, :, :] = f_eq_incoming
#         elif self.location == BoundaryLocation.EAST:
#             f[incoming, Nx-1, :, :] = f_eq_incoming
#         elif self.location == BoundaryLocation.SOUTH:
#             f[incoming, :, 0, :] = f_eq_incoming
#         elif self.location == BoundaryLocation.NORTH:
#             f[incoming, :, Ny-1, :] = f_eq_incoming
#         elif self.location == BoundaryLocation.BOTTOM:
#             f[incoming, :, :, 0] = f_eq_incoming
#         elif self.location == BoundaryLocation.TOP:
#             f[incoming, :, :, Nz-1] = f_eq_incoming
    
#     def _compute_equilibrium_incoming(self) -> 'npt.NDArray':
#         """Compute equilibrium distributions for incoming directions
        
#         Returns:
#             f_eq: Equilibrium for incoming directions, shape (n_incoming, *boundary_shape)
#         """
#         xp = self.xp
        
#         rho = self.density
#         u = self.u_inlet  # (3, *boundary_shape)
        
#         # |u|² for equilibrium formula
#         usqr = xp.sum(u**2, axis=0)  # (*boundary_shape,)
        
#         # c·u for each incoming direction
#         # c_incoming: (3, n_incoming), u: (3, *boundary_shape)
#         # Result: (n_incoming, *boundary_shape)
#         cu = xp.einsum('di,d...->i...', self.c_incoming, u)
        
#         # Equilibrium: f_i^eq = w_i * ρ * (1 + 3*cu + 4.5*cu² - 1.5*usqr)
#         # w_incoming: (n_incoming,) -> broadcast to (n_incoming, 1, 1)
#         w_bc = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        
#         f_eq = w_bc * rho * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
#         return f_eq
    
#     def set_velocity(self, velocity: Union[float, 'npt.NDArray']) -> None:
#         """Update inlet velocity (for time-varying BCs)
        
#         Args:
#             velocity: New velocity (scalar or array)
#         """
#         self._setup_velocity(velocity)


# class EquilibriumInletProfile(EquilibriumInlet):
#     """Equilibrium Inlet with Parabolic (Poiseuille) Profile
    
#     Implements a parabolic velocity profile:
#         u(y, z) = u_max * (1 - (r/R)²)
    
#     where r is distance from center and R is the half-width.
#     """
    
#     def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
#                  location: BoundaryLocation,
#                  u_max: float,
#                  shape: tuple,
#                  density: float = 1.0) -> None:
#         """Initialize parabolic inlet
        
#         Args:
#             xp: Array module
#             lattice: Lattice model
#             location: Boundary location
#             u_max: Maximum (centerline) velocity
#             shape: Domain shape (Nx, Ny, Nz)
#             density: Target density
#         """
#         Nx, Ny, Nz = shape
        
#         # Create parabolic profile function
#         if location in [BoundaryLocation.WEST, BoundaryLocation.EAST]:
#             # Flow in x-direction, profile in y-z plane
#             def profile(Y, Z):
#                 u = xp.zeros((3, Y.shape[0], Y.shape[1]), dtype=xp.float64)
#                 # Parabolic in y and z
#                 y_center = (Ny - 1) / 2.0
#                 z_center = (Nz - 1) / 2.0
#                 R_y = (Ny - 1) / 2.0
#                 R_z = (Nz - 1) / 2.0
                
#                 r_y = (Y - y_center) / R_y
#                 r_z = (Z - z_center) / R_z
                
#                 # Parabolic profile: u = u_max * (1 - r²)
#                 profile_val = u_max * (1 - r_y**2) * (1 - r_z**2)
#                 profile_val = xp.maximum(profile_val, 0)  # Clip negative values
                
#                 u[0] = profile_val if location == BoundaryLocation.WEST else -profile_val
#                 return u
            
#             velocity = profile
#         else:
#             raise NotImplementedError("Parabolic profile only implemented for WEST/EAST")
        
#         super().__init__(xp, lattice, location, velocity, density, shape)
"""
Inlet Boundary Conditions for LBM

This module implements inlet boundary conditions with options for
mass conservation. The key insight is that pure equilibrium inlet
destroys non-equilibrium information, potentially causing mass drift.

Options:
    1. EquilibriumInlet: Simple, sets f = f_eq (may cause mass drift)
    2. NonEquilibriumInlet: Preserves non-eq part (better mass conservation)

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
    
    Sets incoming distributions to equilibrium values.
    Simple but may cause mass drift with wall boundaries.
    
    Use NonEquilibriumInlet for better mass conservation.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 velocity: Union[float, 'npt.NDArray'] = 0.1,
                 density: float = 1.0,
                 shape: Optional[tuple] = None) -> None:
        super().__init__(xp, lattice, location)
        
        self.density = density
        self.shape = shape
        self._setup_velocity(velocity)
        self._precompute_equilibrium_coeffs()
    
    def _setup_velocity(self, velocity: Union[float, 'npt.NDArray']) -> None:
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for inlet BC")
        
        Nx, Ny, Nz = self.shape
        
        # Determine boundary shape based on location
        loc = self.location.value
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
            self.u_inlet = xp.zeros((3,) + boundary_shape, dtype=xp.float64)
            sign = 1 if loc in ['xmin', 'ymin', 'zmin', 'west', 'south', 'bottom'] else -1
            self.u_inlet[normal_dir] = sign * float(velocity)
        else:
            self.u_inlet = xp.asarray(velocity, dtype=xp.float64)
    
    def _precompute_equilibrium_coeffs(self) -> None:
        xp = self.xp
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        
        f_eq_incoming = self._compute_equilibrium_incoming()
        incoming = self.incoming_indices
        loc = self.location.value
        
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
        xp = self.xp
        rho = self.density
        u = self.u_inlet
        
        usqr = xp.sum(u**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u)
        
        w_bc = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq = w_bc * rho * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        return f_eq


class NonEquilibriumInlet(BoundaryCondition):
    """Non-Equilibrium Extrapolation Inlet
    
    Preserves non-equilibrium part for better mass conservation:
        f_i(inlet) = f_i^eq(ρ_target, u_target) + f_i^neq(extrapolated)
    
    The non-equilibrium part is extrapolated from interior nodes,
    carrying viscous stress information.
    
    Recommended for simulations with wall boundaries where mass
    conservation is important.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 velocity: Union[float, 'npt.NDArray'] = 0.1,
                 density: float = 1.0,
                 shape: Optional[tuple] = None) -> None:
        super().__init__(xp, lattice, location)
        
        self.density = density
        self.shape = shape
        self._setup_velocity(velocity)
        self._precompute_coeffs()
    
    def _setup_velocity(self, velocity: Union[float, 'npt.NDArray']) -> None:
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for inlet BC")
        
        Nx, Ny, Nz = self.shape
        loc = self.location.value
        
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
        
        if isinstance(velocity, (int, float)):
            self.u_inlet = xp.zeros((3,) + boundary_shape, dtype=xp.float64)
            sign = 1 if loc in ['xmin', 'ymin', 'zmin', 'west', 'south', 'bottom'] else -1
            self.u_inlet[normal_dir] = sign * float(velocity)
        else:
            self.u_inlet = xp.asarray(velocity, dtype=xp.float64)
    
    def _precompute_coeffs(self) -> None:
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
        1. Get distribution at interior node (x=1 for xmin inlet)
        2. Compute macroscopic at interior
        3. Compute f_eq at interior
        4. Extract non-equilibrium: f_neq = f - f_eq
        5. Set inlet: f = f_eq(target) + f_neq(interior)
        """
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        loc = self.location.value
        
        # Get interior slice for extrapolation
        if loc in ['xmin', 'west']:
            f_interior = f[:, 1, :, :]      # (Q, Ny, Nz)
            boundary_idx = 0
        elif loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, :, :]
            boundary_idx = Nx - 1
        elif loc in ['ymin', 'south']:
            f_interior = f[:, :, 1, :]
            boundary_idx = 0
        elif loc in ['ymax', 'north']:
            f_interior = f[:, :, Ny-2, :]
            boundary_idx = Ny - 1
        elif loc in ['zmin', 'bottom']:
            f_interior = f[:, :, :, 1]
            boundary_idx = 0
        else:  # zmax, top
            f_interior = f[:, :, :, Nz-2]
            boundary_idx = Nz - 1
        
        # Compute macroscopic at interior
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # Compute equilibrium at interior
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # Non-equilibrium at interior
        f_neq_int = f_interior - f_eq_int
        
        # Compute equilibrium at boundary (target values)
        rho_target = self.density
        u_target = self.u_inlet
        
        usqr_target = xp.sum(u_target**2, axis=0)
        cu_target = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu_target + 4.5*(cu_target**2) - 1.5*usqr_target)
        
        # f = f_eq(target) + f_neq(interior)
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
        elif loc in ['zmax', 'top']:
            f[incoming, :, :, Nz-1] = f_incoming


# Alias for backward compatibility
EquilibriumInletProfile = EquilibriumInlet  # Parabolic profile to be implemented later