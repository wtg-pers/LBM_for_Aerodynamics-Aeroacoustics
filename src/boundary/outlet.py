# """
# Characteristic (Open) Outlet Boundary Condition for LBM

# This module implements a non-reflecting outlet boundary condition based on
# characteristic wave analysis. Essential for aeroacoustics simulations where
# acoustic wave reflections at boundaries must be minimized.

# Physical Background:
#     In compressible flow, information propagates along characteristic lines
#     with speeds λ = u ± c_s. At an outflow boundary:
    
#     - Outgoing characteristics (λ > 0): information from interior → extrapolate
#     - Incoming characteristics (λ < 0): must be specified to avoid reflection
    
#     The non-reflecting condition sets incoming wave amplitude based on:
#     1. Zero gradient (fully non-reflecting, but may drift)
#     2. Target pressure relaxation (partially non-reflecting, stable)

# LBM Implementation:
#     The distribution function is decomposed into:
#     - Equilibrium part: carries macroscopic information
#     - Non-equilibrium part: carries viscous stress information
    
#     For incoming directions at the outlet:
#         f_i = f_i^eq(ρ_target, u_extrapolated) + f_i^neq(extrapolated)
    
#     With relaxation towards target pressure:
#         ρ_target = ρ₀ + K * (ρ_extrapolated - ρ₀)
    
#     K = 0: Fully non-reflecting (pressure floats)
#     K = 1: Pressure fixed (fully reflecting)
#     K ~ 0.1-0.25: Partially non-reflecting (recommended)

# References:
#     - Izquierdo & Fueyo, J. Comp. Phys. 227, 2008
#     - Lou et al., Phys. Rev. E 91, 2015
#     - Latt et al., Phys. Rev. E 77, 2008

# Author: LBM Development Team
# Date: 2026-01
# """

# from typing import TYPE_CHECKING, Optional
# import numpy as np

# if TYPE_CHECKING:
#     from types import ModuleType
#     import numpy.typing as npt

# from .base import BoundaryCondition, BoundaryLocation


# class CharacteristicOutlet(BoundaryCondition):
#     """Characteristic (Non-Reflecting) Outlet Boundary Condition
    
#     Minimizes acoustic wave reflections at the outlet using characteristic
#     decomposition. Suitable for aeroacoustic simulations.
    
#     The boundary condition:
#     1. Extrapolates velocity from interior (zero gradient)
#     2. Relaxes density towards target value (partial reflection control)
#     3. Reconstructs incoming distributions from equilibrium + extrapolated non-eq
    
#     Attributes:
#         rho_target: Target outlet density (pressure = ρ * cs²)
#         relax_coeff: Relaxation coefficient K ∈ [0, 1]
#                     K=0: fully non-reflecting
#                     K=1: pressure fixed (reflecting)
#     """
    
#     def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
#                  location: BoundaryLocation,
#                  rho_target: float = 1.0,
#                  relax_coeff: float = 0.1,
#                  shape: Optional[tuple] = None) -> None:
#         """Initialize characteristic outlet
        
#         Args:
#             xp: Array module (numpy or cupy)
#             lattice: Lattice model
#             location: Boundary location (typically EAST for x-outlet)
#             rho_target: Target density at outlet (default 1.0)
#             relax_coeff: Relaxation coefficient (default 0.1, recommended 0.05-0.25)
#             shape: Domain shape (Nx, Ny, Nz)
#         """
#         super().__init__(xp, lattice, location)
        
#         self.rho_target = rho_target
#         self.K = relax_coeff  # Relaxation coefficient
#         self.shape = shape
        
#         # Get outgoing directions (for non-equilibrium extrapolation)
#         self.outgoing_indices = self._get_outgoing_indices()
        
#         # Precompute coefficients
#         self._precompute_coefficients()
        
#     def _get_outgoing_indices(self) -> 'npt.NDArray':
#         """Get indices of distributions leaving the domain"""
#         xp = self.xp
#         c = self.c
        
#         # Outgoing = opposite of incoming (leaving the domain)
#         if self.location == BoundaryLocation.WEST:
#             mask = c[0, :] < 0
#         elif self.location == BoundaryLocation.EAST:
#             mask = c[0, :] > 0
#         elif self.location == BoundaryLocation.SOUTH:
#             mask = c[1, :] < 0
#         elif self.location == BoundaryLocation.NORTH:
#             mask = c[1, :] > 0
#         elif self.location == BoundaryLocation.BOTTOM:
#             mask = c[2, :] < 0
#         elif self.location == BoundaryLocation.TOP:
#             mask = c[2, :] > 0
        
#         return xp.where(mask)[0]
    
#     def _precompute_coefficients(self) -> None:
#         """Precompute equilibrium coefficients"""
#         xp = self.xp
        
#         # Velocity vectors for incoming directions
#         incoming = self.incoming_indices
#         self.c_incoming = self.c[:, incoming].astype(xp.float64)
#         self.w_incoming = self.w[incoming].astype(xp.float64)
        
#         # Store opposite direction mapping for non-eq bounce-back
#         opp = xp.asarray(self.lattice.opp)
#         self.opp_incoming = opp[incoming]
    
#     def apply(self, f: 'npt.NDArray', **kwargs) -> None:
#         """Apply characteristic outlet condition
        
#         Algorithm:
#         1. Extrapolate macroscopic variables from interior
#         2. Apply pressure relaxation: ρ = ρ_target + K*(ρ_extrap - ρ_target)
#         3. Compute equilibrium with relaxed ρ and extrapolated u
#         4. Extrapolate non-equilibrium from interior
#         5. Reconstruct: f_i = f_i^eq + f_i^neq
        
#         Args:
#             f: Distribution function, shape (Q, Nx, Ny, Nz)
#         """
#         xp = self.xp
#         Nx, Ny, Nz = f.shape[1:]
        
#         # =====================================================================
#         # Step 1: Extrapolate from interior (1st order)
#         # =====================================================================
#         if self.location == BoundaryLocation.EAST:
#             # Outlet at x = Nx-1, extrapolate from x = Nx-2
#             f_interior = f[:, Nx-2, :, :]  # (Q, Ny, Nz)
#             f_interior2 = f[:, Nx-3, :, :] if Nx > 2 else f_interior
#             boundary_idx = Nx - 1
#         elif self.location == BoundaryLocation.WEST:
#             f_interior = f[:, 1, :, :]
#             f_interior2 = f[:, 2, :, :] if Nx > 2 else f_interior
#             boundary_idx = 0
#         elif self.location == BoundaryLocation.NORTH:
#             f_interior = f[:, :, Ny-2, :]
#             f_interior2 = f[:, :, Ny-3, :] if Ny > 2 else f_interior
#             boundary_idx = Ny - 1
#         elif self.location == BoundaryLocation.SOUTH:
#             f_interior = f[:, :, 1, :]
#             f_interior2 = f[:, :, 2, :] if Ny > 2 else f_interior
#             boundary_idx = 0
#         elif self.location == BoundaryLocation.TOP:
#             f_interior = f[:, :, :, Nz-2]
#             f_interior2 = f[:, :, :, Nz-3] if Nz > 2 else f_interior
#             boundary_idx = Nz - 1
#         elif self.location == BoundaryLocation.BOTTOM:
#             f_interior = f[:, :, :, 1]
#             f_interior2 = f[:, :, :, 2] if Nz > 2 else f_interior
#             boundary_idx = 0
        
#         # =====================================================================
#         # Step 2: Compute macroscopic at interior point
#         # =====================================================================
#         rho_interior = xp.sum(f_interior, axis=0)  # (Ny, Nz) or similar
        
#         # Momentum
#         c_float = self.c.astype(xp.float64)
#         momentum_interior = xp.tensordot(c_float, f_interior, axes=(1, 0))
#         u_interior = momentum_interior / (rho_interior + 1e-10)
        
#         # =====================================================================
#         # Step 3: Apply pressure relaxation
#         # =====================================================================
#         # Partially non-reflecting: ρ relaxes towards target
#         # K = 0: fully non-reflecting (ρ = ρ_interior, pressure floats)
#         # K = 1: pressure fixed (ρ = ρ_target)
#         rho_boundary = self.rho_target + (1 - self.K) * (rho_interior - self.rho_target)
        
#         # Use extrapolated velocity (zero gradient for velocity)
#         u_boundary = u_interior
        
#         # =====================================================================
#         # Step 4: Compute equilibrium for incoming directions
#         # =====================================================================
#         usqr = xp.sum(u_boundary**2, axis=0)
        
#         # c·u for incoming directions
#         cu = xp.einsum('di,d...->i...', self.c_incoming, u_boundary)
        
#         # Equilibrium
#         w_bc = self.w_incoming.reshape((-1,) + (1,) * (rho_boundary.ndim))
#         f_eq_incoming = w_bc * rho_boundary * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
#         # =====================================================================
#         # Step 5: Extrapolate non-equilibrium part
#         # =====================================================================
#         # Compute f_eq at interior for incoming directions
#         cu_int = xp.einsum('di,d...->i...', self.c_incoming, u_interior)
#         f_eq_int = w_bc * rho_interior * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) 
#                                           - 1.5*xp.sum(u_interior**2, axis=0))
        
#         # Non-equilibrium at interior
#         incoming = self.incoming_indices
#         f_neq_interior = f_interior[incoming] - f_eq_int
        
#         # =====================================================================
#         # Step 6: Reconstruct and apply
#         # =====================================================================
#         # f = f_eq(boundary) + f_neq(extrapolated from interior)
#         f_incoming = f_eq_incoming + f_neq_interior
        
#         # Apply to distribution
#         if self.location == BoundaryLocation.EAST:
#             f[incoming, Nx-1, :, :] = f_incoming
#         elif self.location == BoundaryLocation.WEST:
#             f[incoming, 0, :, :] = f_incoming
#         elif self.location == BoundaryLocation.NORTH:
#             f[incoming, :, Ny-1, :] = f_incoming
#         elif self.location == BoundaryLocation.SOUTH:
#             f[incoming, :, 0, :] = f_incoming
#         elif self.location == BoundaryLocation.TOP:
#             f[incoming, :, :, Nz-1] = f_incoming
#         elif self.location == BoundaryLocation.BOTTOM:
#             f[incoming, :, :, 0] = f_incoming


# class ConvectiveOutlet(BoundaryCondition):
#     """Convective (Advective) Outlet Boundary Condition
    
#     Assumes structures convect out at a specified velocity:
#         ∂f_i/∂t + U_c * ∂f_i/∂n = 0
    
#     Discretized:
#         f_i^{n+1}(boundary) = f_i^n(boundary) - U_c*Δt/Δx * [f(boundary) - f(interior)]
    
#     For LBM with Δt/Δx = 1:
#         f_i(boundary) = f_i(boundary) * (1 - U_c) + f_i(interior) * U_c
    
#     Simpler alternative to characteristic BC, good for vortex shedding.
    
#     Attributes:
#         U_c: Convective velocity (typically mean inlet velocity)
#     """
    
#     def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
#                  location: BoundaryLocation,
#                  convective_velocity: float = 0.1,
#                  shape: Optional[tuple] = None) -> None:
#         """Initialize convective outlet
        
#         Args:
#             xp: Array module
#             lattice: Lattice model
#             location: Boundary location
#             convective_velocity: Advection speed (typically inlet velocity)
#             shape: Domain shape
#         """
#         super().__init__(xp, lattice, location)
        
#         self.U_c = min(convective_velocity, 1.0)  # Stability: U_c <= 1
#         self.shape = shape
    
#     def apply(self, f: 'npt.NDArray', **kwargs) -> None:
#         """Apply convective outlet condition
        
#         Args:
#             f: Distribution function
#         """
#         xp = self.xp
#         Nx, Ny, Nz = f.shape[1:]
        
#         incoming = self.incoming_indices
#         U_c = self.U_c
        
#         if self.location == BoundaryLocation.EAST:
#             # Convect from interior (Nx-2) to boundary (Nx-1)
#             f_int = f[incoming, Nx-2, :, :]
#             f_bnd = f[incoming, Nx-1, :, :]
#             f[incoming, Nx-1, :, :] = (1 - U_c) * f_bnd + U_c * f_int
            
#         elif self.location == BoundaryLocation.WEST:
#             f_int = f[incoming, 1, :, :]
#             f_bnd = f[incoming, 0, :, :]
#             f[incoming, 0, :, :] = (1 - U_c) * f_bnd + U_c * f_int
            
#         elif self.location == BoundaryLocation.NORTH:
#             f_int = f[incoming, :, Ny-2, :]
#             f_bnd = f[incoming, :, Ny-1, :]
#             f[incoming, :, Ny-1, :] = (1 - U_c) * f_bnd + U_c * f_int
            
#         elif self.location == BoundaryLocation.SOUTH:
#             f_int = f[incoming, :, 1, :]
#             f_bnd = f[incoming, :, 0, :]
#             f[incoming, :, 0, :] = (1 - U_c) * f_bnd + U_c * f_int
            
#         elif self.location == BoundaryLocation.TOP:
#             f_int = f[incoming, :, :, Nz-2]
#             f_bnd = f[incoming, :, :, Nz-1]
#             f[incoming, :, :, Nz-1] = (1 - U_c) * f_bnd + U_c * f_int
            
#         elif self.location == BoundaryLocation.BOTTOM:
#             f_int = f[incoming, :, :, 1]
#             f_bnd = f[incoming, :, :, 0]
#             f[incoming, :, :, 0] = (1 - U_c) * f_bnd + U_c * f_int


# class ExtrapolationOutlet(BoundaryCondition):
#     """Simple Extrapolation Outlet
    
#     First-order extrapolation (zero gradient):
#         f_i(boundary) = f_i(interior)
    
#     Simplest outlet, but can cause reflections.
#     Use as baseline or for testing.
#     """
    
#     def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
#                  location: BoundaryLocation,
#                  order: int = 1) -> None:
#         """Initialize extrapolation outlet
        
#         Args:
#             xp: Array module
#             lattice: Lattice model
#             location: Boundary location
#             order: Extrapolation order (1 or 2)
#         """
#         super().__init__(xp, lattice, location)
#         self.order = order
    
#     def apply(self, f: 'npt.NDArray', **kwargs) -> None:
#         """Apply extrapolation outlet"""
#         xp = self.xp
#         Nx, Ny, Nz = f.shape[1:]
        
#         incoming = self.incoming_indices
        
#         if self.order == 1:
#             # Zero gradient: f(boundary) = f(interior)
#             if self.location == BoundaryLocation.EAST:
#                 f[incoming, Nx-1, :, :] = f[incoming, Nx-2, :, :]
#             elif self.location == BoundaryLocation.WEST:
#                 f[incoming, 0, :, :] = f[incoming, 1, :, :]
#             elif self.location == BoundaryLocation.NORTH:
#                 f[incoming, :, Ny-1, :] = f[incoming, :, Ny-2, :]
#             elif self.location == BoundaryLocation.SOUTH:
#                 f[incoming, :, 0, :] = f[incoming, :, 1, :]
#             elif self.location == BoundaryLocation.TOP:
#                 f[incoming, :, :, Nz-1] = f[incoming, :, :, Nz-2]
#             elif self.location == BoundaryLocation.BOTTOM:
#                 f[incoming, :, :, 0] = f[incoming, :, :, 1]
                
#         elif self.order == 2:
#             # Linear extrapolation: f(b) = 2*f(b-1) - f(b-2)
#             if self.location == BoundaryLocation.EAST:
#                 f[incoming, Nx-1, :, :] = 2*f[incoming, Nx-2, :, :] - f[incoming, Nx-3, :, :]
#             elif self.location == BoundaryLocation.WEST:
#                 f[incoming, 0, :, :] = 2*f[incoming, 1, :, :] - f[incoming, 2, :, :]
#             # ... similar for other faces

"""
Outlet Boundary Conditions for LBM

This module implements outlet boundary conditions with non-reflecting
properties for aeroacoustic simulations.

Options:
    1. CharacteristicOutlet: Non-reflecting, preserves non-eq (recommended)
    2. ConvectiveOutlet: Simple advection-based
    3. ExtrapolationOutlet: Zero-gradient extrapolation

Author: LBM Development Team
Date: 2026-01
"""

from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class CharacteristicOutlet(BoundaryCondition):
    """Characteristic (Non-Reflecting) Outlet Boundary Condition
    
    Minimizes acoustic wave reflections at the outlet.
    Preserves non-equilibrium part from interior.
    
    f_i(outlet) = f_i^eq(ρ_relaxed, u_extrapolated) + f_i^neq(extrapolated)
    
    where:
        ρ_relaxed = ρ_target + (1 - K) * (ρ_interior - ρ_target)
        K = 0: fully non-reflecting (pressure floats)
        K = 1: pressure fixed (fully reflecting)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 rho_target: float = 1.0,
                 relax_coeff: float = 0.1,
                 shape: Optional[tuple] = None) -> None:
        super().__init__(xp, lattice, location)
        
        self.rho_target = rho_target
        self.K = relax_coeff
        self.shape = shape
        
        self._precompute_coefficients()
    
    def _precompute_coefficients(self) -> None:
        xp = self.xp
        
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
        
        self.c_all = self.c.astype(xp.float64)
        self.w_all = self.w.astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply characteristic outlet condition"""
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        loc = self.location.value
        
        # Get interior distribution
        if loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, :, :]
            boundary_idx = Nx - 1
        elif loc in ['xmin', 'west']:
            f_interior = f[:, 1, :, :]
            boundary_idx = 0
        elif loc in ['ymax', 'north']:
            f_interior = f[:, :, Ny-2, :]
            boundary_idx = Ny - 1
        elif loc in ['ymin', 'south']:
            f_interior = f[:, :, 1, :]
            boundary_idx = 0
        elif loc in ['zmax', 'top']:
            f_interior = f[:, :, :, Nz-2]
            boundary_idx = Nz - 1
        else:
            f_interior = f[:, :, :, 1]
            boundary_idx = 0
        
        # Macroscopic at interior
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # Equilibrium at interior
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # Non-equilibrium
        f_neq_int = f_interior - f_eq_int
        
        # Relaxed density
        rho_boundary = self.rho_target + (1 - self.K) * (rho_int - self.rho_target)
        u_boundary = u_int  # Zero gradient for velocity
        
        # Equilibrium at boundary
        usqr = xp.sum(u_boundary**2, axis=0)
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_boundary)
        w_in = self.w_incoming.reshape((-1,) + (1,)*(rho_boundary.ndim))
        f_eq_boundary = w_in * rho_boundary * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        # f = f_eq(boundary) + f_neq(interior)
        incoming = self.incoming_indices
        f_incoming = f_eq_boundary + f_neq_int[incoming]
        
        # Apply
        if loc in ['xmax', 'east']:
            f[incoming, Nx-1, :, :] = f_incoming
        elif loc in ['xmin', 'west']:
            f[incoming, 0, :, :] = f_incoming
        elif loc in ['ymax', 'north']:
            f[incoming, :, Ny-1, :] = f_incoming
        elif loc in ['ymin', 'south']:
            f[incoming, :, 0, :] = f_incoming
        elif loc in ['zmax', 'top']:
            f[incoming, :, :, Nz-1] = f_incoming
        else:
            f[incoming, :, :, 0] = f_incoming


class ConvectiveOutlet(BoundaryCondition):
    """Convective (Advective) Outlet
    
    f(boundary) = (1 - U_c) * f(boundary) + U_c * f(interior)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 convective_velocity: float = 0.1,
                 shape: Optional[tuple] = None) -> None:
        super().__init__(xp, lattice, location)
        self.U_c = min(convective_velocity, 1.0)
        self.shape = shape
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        incoming = self.incoming_indices
        U_c = self.U_c
        loc = self.location.value
        
        if loc in ['xmax', 'east']:
            f_int = f[incoming, Nx-2, :, :]
            f_bnd = f[incoming, Nx-1, :, :]
            f[incoming, Nx-1, :, :] = (1 - U_c) * f_bnd + U_c * f_int
        elif loc in ['xmin', 'west']:
            f_int = f[incoming, 1, :, :]
            f_bnd = f[incoming, 0, :, :]
            f[incoming, 0, :, :] = (1 - U_c) * f_bnd + U_c * f_int


class ExtrapolationOutlet(BoundaryCondition):
    """Simple Zero-Gradient Outlet
    
    f(boundary) = f(interior)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 order: int = 1) -> None:
        super().__init__(xp, lattice, location)
        self.order = order
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        incoming = self.incoming_indices
        loc = self.location.value
        
        if loc in ['xmax', 'east']:
            f[incoming, Nx-1, :, :] = f[incoming, Nx-2, :, :]
        elif loc in ['xmin', 'west']:
            f[incoming, 0, :, :] = f[incoming, 1, :, :]