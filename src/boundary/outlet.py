"""
Outlet Boundary Conditions for LBM

This module implements outlet boundary conditions for LBM simulations.

Available BCs:
==============

1. PressureRelaxationOutlet (formerly CharacteristicOutlet)
   - Physics: Pressure relaxation + Non-equilibrium extrapolation
   - NOT a true characteristic BC (no LODI wave analysis)
   - Suitable for: steady/unsteady external flows

2. ConvectiveOutlet
   - Physics: Advective transport equation ∂f/∂t + U_c ∂f/∂n = 0
   - Suitable for: vortex shedding, convection-dominated flows

3. NeumannOutlet (formerly ExtrapolationOutlet)
   - Physics: Zero-gradient (Neumann) condition ∂f/∂n = 0
   - Simplest outlet, may cause reflections
   - Suitable for: testing, simple flows

Note:
-----
For true non-reflecting boundary conditions based on characteristic
wave analysis (LODI equations), see characteristic.py (to be implemented).

References:
-----------
- Guo et al., Phys. Rev. E 65, 2002 (NEQ extrapolation)
- Lou et al., Phys. Rev. E 91, 2015 (Convective BC)

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class PressureRelaxationOutlet(BoundaryCondition):
    """Pressure Relaxation Outlet with Non-Equilibrium Extrapolation
    
    Physical Principle:
    -------------------
    This BC combines two mechanisms:
    
    1. **Pressure Relaxation**: Density (≈pressure) relaxes toward target value
       
           ρ_boundary = ρ_target + (1 - K)(ρ_interior - ρ_target)
       
       where K is the relaxation coefficient:
       - K = 0: Fully non-reflecting (pressure floats freely)
       - K = 1: Pressure fixed at ρ_target (fully reflecting)
       - K ~ 0.1: Recommended for most applications
    
    2. **Non-Equilibrium Extrapolation** (Guo et al., 2002):
       Preserves viscous stress information from interior:
       
           f_i = f_i^eq(ρ_boundary, u_interior) + f_i^neq(interior)
       
       where f^neq = f - f^eq carries viscous/stress information.
    
    Mathematical Formulation:
    -------------------------
    For incoming distributions at outlet:
    
        f_i(outlet) = f_i^eq(ρ_relaxed, u_extrapolated) + f_i^neq(extrapolated)
    
    where:
        ρ_relaxed = ρ_target + (1 - K)(ρ_interior - ρ_target)  [pressure control]
        u_boundary = u_interior                                 [zero velocity gradient]
        f^neq = f - f^eq                                        [non-equilibrium part]
    
    Important Note:
    ---------------
    Despite the former name "CharacteristicOutlet", this BC does NOT implement
    true characteristic analysis (LODI equations). It is a pressure-based BC
    with non-equilibrium correction, which provides adequate non-reflecting
    properties for many applications but is not theoretically optimal for
    acoustic wave problems.
    
    For true characteristic BCs, see CharacteristicBC in characteristic.py
    (to be implemented based on Izquierdo & Fueyo, J. Comp. Phys. 2008).
    
    Attributes:
        rho_target: Target density at outlet  [dimensionless, ρ/ρ₀]
        K: Relaxation coefficient  [dimensionless, 0 ≤ K ≤ 1]
           K=0: non-reflecting, K=1: fixed pressure
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
    
    Example:
        >>> # Outlet with 10% pressure relaxation (typical for external flows)
        >>> bc = PressureRelaxationOutlet(
        ...     xp, lattice, 'xmax',
        ...     rho_target=1.0,
        ...     relax_coeff=0.1,  # K=0.1: mostly non-reflecting
        ...     shape=(200, 100, 100)
        ... )
    
    References:
        - Guo et al., Phys. Rev. E 65, 046308, 2002 (NEQ extrapolation)
        - Izquierdo & Fueyo, J. Comp. Phys. 227, 2008 (pressure relaxation concept)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 rho_target: float = 1.0,
                 relax_coeff: float = 0.1,
                 shape: Optional[tuple] = None) -> None:
        """Initialize pressure relaxation outlet
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D3Q27, D3Q19, etc.)
            location: Boundary location ('xmax', 'xmin', etc.)
            rho_target: Target outlet density  [dimensionless, ρ/ρ₀]
                       In LBM: pressure p = ρ·c_s² = ρ/3
            relax_coeff: Pressure relaxation coefficient K  [dimensionless]
                        K=0: fully non-reflecting (pressure floats)
                        K=1: pressure fixed (fully reflecting)
                        Recommended: K=0.05~0.25 for external flows
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_target = rho_target  # [dimensionless]
        self.K = relax_coeff          # [dimensionless]
        self.shape = shape            # [lattice units]
        
        self._precompute_coefficients()
    
    def _precompute_coefficients(self) -> None:
        """Precompute equilibrium coefficients for efficiency"""
        xp = self.xp
        
        # Coefficients for incoming directions only
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)  # [Δx/Δt]
        self.w_incoming = self.w[incoming].astype(xp.float64)     # [dimensionless]
        
        # Coefficients for all directions (needed for interior f_eq)
        self.c_all = self.c.astype(xp.float64)  # [Δx/Δt]
        self.w_all = self.w.astype(xp.float64)  # [dimensionless]
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply pressure relaxation outlet condition
        
        Algorithm:
        ----------
        1. Extract f at interior node (one layer inside boundary)
        2. Compute macroscopic (ρ_int, u_int) at interior
        3. Compute f^eq at interior → extract f^neq = f - f^eq
        4. Apply pressure relaxation: ρ_target = ρ_target + (1-K)(ρ_int - ρ_target)
        5. Set velocity: u_boundary = u_interior (zero gradient)
        6. Compute f^eq at boundary with (ρ_relaxed, u_interior)
        7. Reconstruct: f = f^eq(boundary) + f^neq(interior)
        
        Args:
            f: Distribution function after streaming, shape (Q, Nx, Ny, Nz)
               [dimensionless] - Modified in-place
        """
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        loc = self.location.value
        
        # =====================================================================
        # Step 1: Get interior distribution (one layer inside)
        # =====================================================================
        if loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, :, :]  # (Q, Ny, Nz)
        elif loc in ['xmin', 'west']:
            f_interior = f[:, 1, :, :]
        elif loc in ['ymax', 'north']:
            f_interior = f[:, :, Ny-2, :]
        elif loc in ['ymin', 'south']:
            f_interior = f[:, :, 1, :]
        elif loc in ['zmax', 'top']:
            f_interior = f[:, :, :, Nz-2]
        else:  # zmin, bottom
            f_interior = f[:, :, :, 1]
        
        # =====================================================================
        # Step 2: Compute macroscopic at interior
        # =====================================================================
        # ρ = Σ f_i  [dimensionless]
        rho_int = xp.sum(f_interior, axis=0)
        
        # u = Σ(c_i · f_i) / ρ  [Δx/Δt]
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # =====================================================================
        # Step 3: Compute f^eq at interior, extract f^neq
        # =====================================================================
        # f^eq = w_i · ρ · (1 + 3(c·u) + 4.5(c·u)² - 1.5|u|²)  [dimensionless]
        usqr_int = xp.sum(u_int**2, axis=0)  # |u|²  [Δx²/Δt²]
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)  # c·u  [Δx²/Δt²]
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # f^neq = f - f^eq (carries viscous stress information)  [dimensionless]
        f_neq_int = f_interior - f_eq_int
        
        # =====================================================================
        # Step 4 & 5: Relaxed density and extrapolated velocity
        # =====================================================================
        # Pressure relaxation: ρ_boundary = ρ_target + (1-K)(ρ_int - ρ_target)
        # K=0: ρ_boundary = ρ_int (fully non-reflecting)
        # K=1: ρ_boundary = ρ_target (fixed pressure)
        rho_boundary = self.rho_target + (1 - self.K) * (rho_int - self.rho_target)
        
        # Zero velocity gradient: u_boundary = u_interior
        u_boundary = u_int
        
        # =====================================================================
        # Step 6: Compute f^eq at boundary for incoming directions
        # =====================================================================
        usqr = xp.sum(u_boundary**2, axis=0)  # [Δx²/Δt²]
        cu = xp.einsum('di,d...->i...', self.c_incoming, u_boundary)  # [Δx²/Δt²]
        w_in = self.w_incoming.reshape((-1,) + (1,)*(rho_boundary.ndim))
        f_eq_boundary = w_in * rho_boundary * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        # =====================================================================
        # Step 7: Reconstruct f = f^eq(boundary) + f^neq(interior)
        # =====================================================================
        incoming = self.incoming_indices
        f_incoming = f_eq_boundary + f_neq_int[incoming]
        
        # Apply to distribution at boundary
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
        else:  # zmin, bottom
            f[incoming, :, :, 0] = f_incoming
    
    def get_info(self) -> str:
        """Return information string about this BC"""
        return (f"PressureRelaxationOutlet at {self.location.value}:\n"
                f"    ρ_target={self.rho_target}, K={self.K}\n"
                f"    (K=0: non-reflecting, K=1: fixed pressure)")


class ConvectiveOutlet(BoundaryCondition):
    """Convective (Advective) Outlet Boundary Condition
    
    Physical Principle:
    -------------------
    Assumes flow structures convect out of the domain at a specified
    advection velocity U_c. This is based on the 1D advection equation:
    
        ∂f/∂t + U_c · ∂f/∂n = 0
    
    where n is the outward normal direction.
    
    Mathematical Formulation:
    -------------------------
    Discretized with first-order upwind:
    
        f_i^{n+1}(boundary) = f_i^n(boundary) - U_c·Δt/Δx · [f(boundary) - f(interior)]
    
    For LBM with Δt/Δx = 1:
    
        f(boundary) = (1 - U_c) · f(boundary) + U_c · f(interior)
    
    This is essentially a weighted average between current boundary value
    and interior value, with U_c controlling the blending ratio.
    
    Physical Interpretation:
    ------------------------
    - U_c = 0: No advection, boundary value unchanged (unstable!)
    - U_c = 1: Full advection, boundary copies interior (zero gradient)
    - U_c = u_inlet: Structures exit at inlet velocity (typical choice)
    
    Suitable Applications:
    ----------------------
    - Vortex shedding simulations (vortices convect at mean flow speed)
    - Wake flows where structures travel at approximately U_inlet
    - Unsteady flows with coherent structures
    
    Limitations:
    ------------
    - Assumes structures travel at U_c (may not be accurate for all cases)
    - Not truly non-reflecting for acoustic waves
    - Requires knowledge of appropriate convection velocity
    
    Attributes:
        U_c: Convective (advection) velocity  [Δx/Δt]
             Typically set to mean inlet velocity
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
    
    Example:
        >>> # Outlet for flow with inlet velocity u_inlet = 0.1
        >>> bc = ConvectiveOutlet(
        ...     xp, lattice, 'xmax',
        ...     convective_velocity=0.1,  # Same as inlet velocity
        ...     shape=(200, 100, 100)
        ... )
    
    References:
        - Lou et al., Phys. Rev. E 91, 2015
        - Orlanski, J. Comp. Phys. 21, 1976 (original concept)
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
            convective_velocity: Advection speed  [Δx/Δt]
                                Typically equals mean inlet velocity
                                Stability requires U_c ≤ 1
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        # Stability constraint: U_c ≤ 1 (CFL condition)
        self.U_c = min(convective_velocity, 1.0)  # [Δx/Δt]
        self.shape = shape
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply convective outlet condition
        
        Algorithm:
        ----------
        For each incoming direction at outlet:
            f(boundary) = (1 - U_c) · f(boundary) + U_c · f(interior)
        
        This convects structures from interior to boundary at velocity U_c.
        
        Args:
            f: Distribution function after streaming, shape (Q, Nx, Ny, Nz)
               [dimensionless] - Modified in-place
        """
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        incoming = self.incoming_indices
        U_c = self.U_c
        loc = self.location.value
        
        if loc in ['xmax', 'east']:
            f_int = f[incoming, Nx-2, :, :]  # Interior
            f_bnd = f[incoming, Nx-1, :, :]  # Boundary (current)
            f[incoming, Nx-1, :, :] = (1 - U_c) * f_bnd + U_c * f_int
            
        elif loc in ['xmin', 'west']:
            f_int = f[incoming, 1, :, :]
            f_bnd = f[incoming, 0, :, :]
            f[incoming, 0, :, :] = (1 - U_c) * f_bnd + U_c * f_int
            
        elif loc in ['ymax', 'north']:
            f_int = f[incoming, :, Ny-2, :]
            f_bnd = f[incoming, :, Ny-1, :]
            f[incoming, :, Ny-1, :] = (1 - U_c) * f_bnd + U_c * f_int
            
        elif loc in ['ymin', 'south']:
            f_int = f[incoming, :, 1, :]
            f_bnd = f[incoming, :, 0, :]
            f[incoming, :, 0, :] = (1 - U_c) * f_bnd + U_c * f_int
            
        elif loc in ['zmax', 'top']:
            f_int = f[incoming, :, :, Nz-2]
            f_bnd = f[incoming, :, :, Nz-1]
            f[incoming, :, :, Nz-1] = (1 - U_c) * f_bnd + U_c * f_int
            
        else:  # zmin, bottom
            f_int = f[incoming, :, :, 1]
            f_bnd = f[incoming, :, :, 0]
            f[incoming, :, :, 0] = (1 - U_c) * f_bnd + U_c * f_int
    
    def get_info(self) -> str:
        """Return information string about this BC"""
        return (f"ConvectiveOutlet at {self.location.value}:\n"
                f"    U_c={self.U_c} [Δx/Δt]")


class NeumannOutlet(BoundaryCondition):
    """Zero-Gradient (Neumann) Outlet Boundary Condition
    
    Physical Principle:
    -------------------
    Applies the Neumann (zero-gradient) condition at the outlet:
    
        ∂f_i/∂n = 0
    
    This is the simplest outlet condition, implemented as first-order
    extrapolation from interior:
    
        f_i(boundary) = f_i(interior)
    
    Physical Interpretation:
    ------------------------
    - Assumes flow is fully developed at outlet (no normal gradients)
    - All flow structures pass through unchanged
    - No physical information prescribed at boundary
    
    Advantages:
    -----------
    - Extremely simple implementation
    - No parameters to tune
    - Works well for fully developed, steady flows
    
    Limitations:
    ------------
    - Can cause significant acoustic wave reflections
    - May lead to instabilities in unsteady flows
    - No mechanism to control pressure drift
    
    Suitable Applications:
    ----------------------
    - Fully developed channel flow (Poiseuille flow)
    - Steady-state simulations where outlet is far from region of interest
    - Debugging and testing (simplest BC)
    
    Not Suitable For:
    -----------------
    - Aeroacoustic simulations (wave reflections)
    - Vortex shedding near outlet
    - Any flow where acoustic non-reflection is important
    
    Attributes:
        order: Extrapolation order (1 or 2)  [dimensionless]
               1: f(boundary) = f(interior)       (first-order)
               2: f(boundary) = 2f(int) - f(int2)  (second-order, linear extrap)
    
    Example:
        >>> # Simple zero-gradient outlet
        >>> bc = NeumannOutlet(xp, lattice, 'xmax', order=1)
    
    Mathematical Note:
    ------------------
    The Neumann condition ∂f/∂n = 0 is discretized as:
    
    First-order:
        (f_boundary - f_interior) / Δx = 0
        → f_boundary = f_interior
    
    Second-order:
        (-3f_boundary + 4f_int1 - f_int2) / (2Δx) = 0
        → f_boundary = (4f_int1 - f_int2) / 3 ≈ 2f_int1 - f_int2
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 order: int = 1) -> None:
        """Initialize Neumann (zero-gradient) outlet
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            location: Boundary location
            order: Extrapolation order  [dimensionless]
                   1: First-order (f_bnd = f_int)
                   2: Second-order (f_bnd = 2f_int1 - f_int2)
        """
        super().__init__(xp, lattice, location)
        self.order = order
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply Neumann (zero-gradient) outlet condition
        
        First-order: f(boundary) = f(interior)
        Second-order: f(boundary) = 2·f(interior) - f(interior2)
        
        Args:
            f: Distribution function after streaming, shape (Q, Nx, Ny, Nz)
               [dimensionless] - Modified in-place
        """
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        incoming = self.incoming_indices
        loc = self.location.value
        
        if self.order == 1:
            # Zero gradient: f(boundary) = f(interior)
            if loc in ['xmax', 'east']:
                f[incoming, Nx-1, :, :] = f[incoming, Nx-2, :, :]
            elif loc in ['xmin', 'west']:
                f[incoming, 0, :, :] = f[incoming, 1, :, :]
            elif loc in ['ymax', 'north']:
                f[incoming, :, Ny-1, :] = f[incoming, :, Ny-2, :]
            elif loc in ['ymin', 'south']:
                f[incoming, :, 0, :] = f[incoming, :, 1, :]
            elif loc in ['zmax', 'top']:
                f[incoming, :, :, Nz-1] = f[incoming, :, :, Nz-2]
            else:  # zmin, bottom
                f[incoming, :, :, 0] = f[incoming, :, :, 1]
                
        elif self.order == 2:
            # Linear extrapolation: f(b) = 2·f(b-1) - f(b-2)
            if loc in ['xmax', 'east']:
                f[incoming, Nx-1, :, :] = (2*f[incoming, Nx-2, :, :] 
                                            - f[incoming, Nx-3, :, :])
            elif loc in ['xmin', 'west']:
                f[incoming, 0, :, :] = 2*f[incoming, 1, :, :] - f[incoming, 2, :, :]
            elif loc in ['ymax', 'north']:
                f[incoming, :, Ny-1, :] = (2*f[incoming, :, Ny-2, :] 
                                            - f[incoming, :, Ny-3, :])
            elif loc in ['ymin', 'south']:
                f[incoming, :, 0, :] = 2*f[incoming, :, 1, :] - f[incoming, :, 2, :]
            elif loc in ['zmax', 'top']:
                f[incoming, :, :, Nz-1] = (2*f[incoming, :, :, Nz-2] 
                                            - f[incoming, :, :, Nz-3])
            else:  # zmin, bottom
                f[incoming, :, :, 0] = 2*f[incoming, :, :, 1] - f[incoming, :, :, 2]
    
    def get_info(self) -> str:
        """Return information string about this BC"""
        return (f"NeumannOutlet at {self.location.value}:\n"
                f"    order={self.order} (∂f/∂n = 0)")


# =============================================================================
# Backward Compatibility Aliases (Deprecated)
# =============================================================================
# These aliases maintain backward compatibility with old code.
# They will be removed in a future version.

class CharacteristicOutlet(PressureRelaxationOutlet):
    """DEPRECATED: Use PressureRelaxationOutlet instead.
    
    This class was misnamed - it does NOT implement true characteristic
    (LODI-based) boundary conditions. The actual implementation is
    pressure relaxation with non-equilibrium extrapolation.
    
    For true characteristic BCs, use CharacteristicBC (to be implemented).
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "CharacteristicOutlet is deprecated and will be removed in v2.0. "
            "Use PressureRelaxationOutlet instead. "
            "Note: This BC does NOT implement true characteristic analysis.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class ExtrapolationOutlet(NeumannOutlet):
    """DEPRECATED: Use NeumannOutlet instead.
    
    Renamed to NeumannOutlet to accurately reflect the physics:
    this implements the Neumann (zero-gradient) boundary condition.
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "ExtrapolationOutlet is deprecated and will be removed in v2.0. "
            "Use NeumannOutlet instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)