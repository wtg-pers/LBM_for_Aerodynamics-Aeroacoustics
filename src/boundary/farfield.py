"""
Far-field Boundary Conditions for LBM

This module implements far-field boundary conditions for external aerodynamics
and aeroacoustics simulations.

Available BCs:
==============

1. FreestreamBC (formerly CharacteristicFarfield)
   - Physics: Fixed freestream velocity U∞ + pressure relaxation
   - NOT a true characteristic BC (no LODI wave analysis)
   - Suitable for: external aerodynamics with U∞ ≠ 0

2. SpongeLayer
   - Physics: Gradual damping toward target state over buffer zone
   - Based on: f = (1-σ)f + σf_target with smooth damping function
   - Suitable for: aeroacoustics, wave absorption

Note on Naming:
---------------
The former "CharacteristicFarfield" was renamed to "FreestreamBC" because:
- It does NOT implement true characteristic analysis (LODI equations)
- The actual physics is: fixed freestream velocity + pressure relaxation
- The name "Freestream" accurately describes what it does

For hover simulations (U∞ = 0), use PressureRelaxationOutlet with K~0.5
which allows velocity to develop freely while controlling pressure.

Note on Former "AmbientBC":
---------------------------
AmbientBC has been removed as it was functionally identical to 
PressureRelaxationOutlet. For pressure-based open boundaries where
velocity should develop freely, use PressureRelaxationOutlet with
appropriate K value (typically K=0.3~0.5).

References:
-----------
- Guo et al., Phys. Rev. E 65, 2002 (NEQ extrapolation)
- Freund, J. Comp. Phys. 143, 1997 (Sponge layer)

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
    """Freestream Boundary Condition for External Aerodynamics
    
    Physical Principle:
    -------------------
    This BC is designed for external flow simulations (forward flight, wind tunnel)
    where a known freestream velocity U∞ must be maintained at far-field boundaries.
    
    Key mechanisms:
    
    1. **Fixed Freestream Velocity**: Boundary velocity is SET to U∞
       
           u_boundary = U∞  (FIXED, not extrapolated!)
       
       This is crucial for external flows - if velocity were extrapolated from
       interior (which may contain wake or separated flow), the driving force
       for the flow would decay.
    
    2. **Pressure Relaxation**: Density relaxes toward freestream value
       
           ρ_boundary = ρ∞ + (1 - K)(ρ_interior - ρ∞)
       
       K controls the reflection:
       - K = 0: Fully non-reflecting (pressure floats)
       - K = 1: Pressure fixed (fully reflecting)
       - K ~ 0.1: Typical for external aerodynamics
    
    3. **Non-Equilibrium Extrapolation** (Guo et al., 2002):
       Preserves viscous information from interior:
       
           f = f^eq(ρ_relaxed, U∞) + f^neq(interior)
    
    Mathematical Formulation:
    -------------------------
    For incoming distributions at far-field:
    
        f_i(farfield) = f_i^eq(ρ_relaxed, U∞) + f_i^neq(extrapolated)
    
    where:
        ρ_relaxed = ρ∞ + (1 - K)(ρ_interior - ρ∞)  [pressure relaxation]
        u_boundary = U∞                             [velocity FIXED to freestream!]
        f^neq = f_interior - f^eq_interior          [non-equilibrium part]
    
    Key Difference from PressureRelaxationOutlet:
    ----------------------------------------------
    - PressureRelaxationOutlet: u_boundary = u_interior (velocity extrapolated)
    - FreestreamBC: u_boundary = U∞ (velocity FIXED to freestream)
    
    This distinction is critical for maintaining proper flow physics in
    external aerodynamics simulations.
    
    Important Note:
    ---------------
    Despite the former name "CharacteristicFarfield", this BC does NOT implement
    true characteristic analysis (LODI equations). It is a freestream-fixed BC
    with pressure relaxation, which works well for many external flow applications
    but is not theoretically optimal for acoustic problems.
    
    For true characteristic BCs, see CharacteristicBC in characteristic.py
    (to be implemented).
    
    When to Use:
    ------------
    - External aerodynamics (airfoil, cylinder, vehicle)
    - Forward flight simulations (U∞ ≠ 0)
    - Wind tunnel simulations
    - Any case where freestream velocity must be maintained
    
    When NOT to Use:
    ----------------
    - Hover simulations (U∞ = 0) - use PressureRelaxationOutlet instead
    - Internal flows (pipe, channel)
    - Cases where velocity should develop freely
    
    Attributes:
        rho_inf: Freestream density  [dimensionless, ρ/ρ₀]
        u_inf: Freestream velocity   [Δx/Δt]
        K: Pressure relaxation coefficient  [dimensionless, 0 ≤ K ≤ 1]
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
    
    Example:
        >>> # Far-field with U∞ = 0.1 in x-direction
        >>> bc = FreestreamBC(
        ...     xp, lattice, 'ymin',
        ...     rho_inf=1.0,
        ...     u_inf=0.1,        # Scalar → (0.1, 0, 0)
        ...     relax_coeff=0.1,  # K=0.1: mostly non-reflecting
        ...     shape=(200, 100, 100)
        ... )
    
    References:
        - Guo et al., Phys. Rev. E 65, 046308, 2002 (NEQ extrapolation)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: Union[str, BoundaryLocation],
                 rho_inf: float = 1.0,
                 u_inf: Union[float, Tuple[float, float, float]] = 0.1,
                 relax_coeff: float = 0.1,
                 shape: Optional[Tuple[int, int, int]] = None) -> None:
        """Initialize freestream boundary condition
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            location: Boundary location ('ymin', 'ymax', 'zmin', 'zmax')
            rho_inf: Freestream density  [dimensionless, ρ/ρ₀]
            u_inf: Freestream velocity, either:
                   - scalar: interpreted as u_x component, i.e., (u_inf, 0, 0)
                   - tuple: (u_x, u_y, u_z)  [Δx/Δt]
            relax_coeff: Pressure relaxation coefficient K  [dimensionless]
                        K=0: non-reflecting, K=1: fixed pressure
                        Typical: K=0.05~0.2 for external flows
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_inf = rho_inf  # [dimensionless]
        self.K = relax_coeff    # [dimensionless]
        self.shape = shape      # [lattice units]
        
        # Process velocity input
        self._setup_velocity(u_inf)
        
        # Precompute coefficients
        self._precompute_coefficients()
    
    def _setup_velocity(self, u_inf: Union[float, Tuple[float, float, float]]) -> None:
        """Setup freestream velocity array"""
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for FreestreamBC")
        
        Nx, Ny, Nz = self.shape
        loc = self.location.value
        
        # Determine boundary shape based on location
        if loc in ['xmin', 'xmax', 'west', 'east']:
            boundary_shape = (Ny, Nz)
        elif loc in ['ymin', 'ymax', 'south', 'north']:
            boundary_shape = (Nx, Nz)
        else:  # zmin, zmax
            boundary_shape = (Nx, Ny)
        
        self.boundary_shape = boundary_shape
        
        # Create velocity array: (3, *boundary_shape)  [Δx/Δt]
        self.u_inf_array = xp.zeros((3,) + boundary_shape, dtype=xp.float64)
        
        if isinstance(u_inf, (int, float)):
            # Scalar → assume flow in x-direction: (u_inf, 0, 0)
            self.u_inf_array[0, :, :] = float(u_inf)
        else:
            # Tuple (u_x, u_y, u_z)
            self.u_inf_array[0, :, :] = float(u_inf[0])
            self.u_inf_array[1, :, :] = float(u_inf[1])
            self.u_inf_array[2, :, :] = float(u_inf[2])
    
    def _precompute_coefficients(self) -> None:
        """Precompute equilibrium coefficients for efficiency"""
        xp = self.xp
        
        # For incoming directions
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)  # [Δx/Δt]
        self.w_incoming = self.w[incoming].astype(xp.float64)     # [dimensionless]
        
        # For all directions (computing interior equilibrium)
        self.c_all = self.c.astype(xp.float64)  # [Δx/Δt]
        self.w_all = self.w.astype(xp.float64)  # [dimensionless]
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply freestream boundary condition
        
        Algorithm:
        ----------
        1. Extract f at interior node (one layer inside boundary)
        2. Compute macroscopic (ρ_int, u_int) at interior
        3. Compute f^eq at interior → extract f^neq = f - f^eq
        4. Apply pressure relaxation: ρ_target = ρ∞ + (1-K)(ρ_int - ρ∞)
        5. Set velocity: u_boundary = U∞ (FIXED to freestream!)
        6. Compute f^eq at boundary with (ρ_relaxed, U∞)
        7. Reconstruct: f = f^eq(ρ_target, U∞) + f^neq(interior)
        
        Args:
            f: Distribution function after streaming, shape (Q, Nx, Ny, Nz)
               [dimensionless] - Modified in-place
        """
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        loc = self.location.value
        
        # =====================================================================
        # Step 1: Get interior distribution
        # =====================================================================
        if loc in ['xmin', 'west']:
            f_interior = f[:, 1, :, :]       # (Q, Ny, Nz)
        elif loc in ['xmax', 'east']:
            f_interior = f[:, Nx-2, :, :]
        elif loc in ['ymin', 'south']:
            f_interior = f[:, :, 1, :]       # (Q, Nx, Nz)
        elif loc in ['ymax', 'north']:
            f_interior = f[:, :, Ny-2, :]
        elif loc in ['zmin', 'bottom']:
            f_interior = f[:, :, :, 1]       # (Q, Nx, Ny)
        else:  # zmax, top
            f_interior = f[:, :, :, Nz-2]
        
        # =====================================================================
        # Step 2: Compute macroscopic at interior
        # =====================================================================
        # ρ = Σ f_i  [dimensionless]
        rho_int = xp.sum(f_interior, axis=0)
        
        # u = Σ(c_i · f_i) / ρ  [Δx/Δt]
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # =====================================================================
        # Step 3: Compute f^eq at interior → f^neq
        # =====================================================================
        usqr_int = xp.sum(u_int**2, axis=0)  # [Δx²/Δt²]
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)  # [Δx²/Δt²]
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        # f^neq carries viscous/stress information  [dimensionless]
        f_neq_int = f_interior - f_eq_int
        
        # =====================================================================
        # Step 4 & 5: Target density (relaxed) and velocity (FIXED to U∞)
        # =====================================================================
        # Pressure relaxation
        rho_target = self.rho_inf + (1 - self.K) * (rho_int - self.rho_inf)
        
        # KEY: Velocity is FIXED to freestream, NOT extrapolated from interior!
        u_target = self.u_inf_array  # [Δx/Δt]
        
        # =====================================================================
        # Step 6: Compute f^eq at boundary for incoming directions
        # =====================================================================
        usqr_target = xp.sum(u_target**2, axis=0)  # [Δx²/Δt²]
        cu_target = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu_target + 4.5*(cu_target**2) - 1.5*usqr_target)
        
        # =====================================================================
        # Step 7: Reconstruct f = f^eq(target) + f^neq(interior)
        # =====================================================================
        incoming = self.incoming_indices
        f_incoming = f_eq_target + f_neq_int[incoming]
        
        # Apply to distribution at boundary
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
    
    def get_info(self) -> str:
        """Return BC information string"""
        u_str = f"({float(self.u_inf_array[0,0,0]):.3f}, " \
                f"{float(self.u_inf_array[1,0,0]):.3f}, " \
                f"{float(self.u_inf_array[2,0,0]):.3f})"
        return (f"FreestreamBC at {self.location.value}:\n"
                f"    ρ∞={self.rho_inf}, U∞={u_str} [Δx/Δt], K={self.K}\n"
                f"    (velocity FIXED to freestream, pressure relaxed)")


class SpongeLayer(BoundaryCondition):
    """Sponge Layer (Buffer Zone) for Wave Damping
    
    Physical Principle:
    -------------------
    Gradually damps flow perturbations toward a target state over a buffer
    zone (sponge region). This is a robust method for absorbing waves and
    preventing reflections at boundaries.
    
    Within the sponge zone (thickness δ from boundary):
    
        f = (1 - σ(x)) · f_computed + σ(x) · f_target
    
    where σ(x) is a smooth damping function that varies from 0 (at the
    interface with physical domain) to σ_max (at the boundary).
    
    Damping Function:
    -----------------
    σ(x) = σ_max · g(ξ), where ξ = (x - x_start) / δ  [0 ≤ ξ ≤ 1]
    
    Available profiles g(ξ):
    
    - **linear**: g(ξ) = ξ
      Simple but has discontinuous derivative at interface.
    
    - **quadratic**: g(ξ) = ξ²
      Smoother transition, commonly used.
    
    - **polynomial** (default): g(ξ) = 6ξ⁵ - 15ξ⁴ + 10ξ³
      C² continuous (smooth first and second derivatives).
      Minimizes spurious reflections. Recommended for aeroacoustics.
    
    - **exponential**: g(ξ) = (1 - e^(-αξ)) / (1 - e^(-α))
      Steep rise near boundary, good for strong damping.
    
    Physical Interpretation:
    ------------------------
    - Near physical domain (ξ ≈ 0): σ ≈ 0, flow evolves naturally
    - Near boundary (ξ → 1): σ → σ_max, flow forced toward target
    - Smooth transition prevents spurious reflections
    
    Trade-offs:
    -----------
    Advantages:
    - Very robust for complex flows
    - Excellent for acoustic absorption
    - Predictable behavior, easy to tune
    
    Disadvantages:
    - Requires buffer zone (increases domain size)
    - Target state must be known (freestream or ambient)
    - Damping zone is "non-physical"
    
    Suitable Applications:
    ----------------------
    - Aeroacoustic simulations (absorb acoustic waves)
    - External flows with strong unsteadiness
    - Cases where other BCs fail or cause reflections
    - Any simulation requiring robust wave absorption
    
    Attributes:
        thickness: Sponge layer thickness  [lattice units]
        sigma_max: Maximum damping coefficient  [dimensionless, 0 ≤ σ ≤ 1]
        rho_inf: Target density  [dimensionless]
        u_inf: Target velocity  [Δx/Δt]
        profile: Damping profile type
    
    Example:
        >>> # Sponge layer at outlet with 20 lattice units thickness
        >>> bc = SpongeLayer(
        ...     xp, lattice, 'xmax',
        ...     thickness=20,           # 20 lattice units buffer
        ...     strength=0.8,           # σ_max = 0.8
        ...     rho_inf=1.0,
        ...     u_inf=(0.1, 0, 0),       # Target = freestream
        ...     profile='polynomial',   # Smooth C² transition
        ...     shape=(200, 100, 100)
        ... )
    
    References:
        - Freund, J. Comp. Phys. 143, 1997 (original sponge layer concept)
        - Colonius & Lele, Prog. Aerosp. Sci. 40, 2004 (aeroacoustics review)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: Union[str, BoundaryLocation],
                 thickness: int = 10,
                 strength: float = 0.5,
                 rho_inf: float = 1.0,
                 u_inf: Union[float, Tuple[float, float, float]] = 0.0,
                 profile: str = 'polynomial',
                 shape: Optional[Tuple[int, int, int]] = None) -> None:
        """Initialize sponge layer
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            location: Boundary location
            thickness: Sponge layer thickness  [lattice units]
                      Typical: 10-50 lattice units
            strength: Maximum damping coefficient σ_max  [dimensionless, 0-1]
                     σ_max=0.5: moderate damping
                     σ_max=0.8-1.0: strong damping
            rho_inf: Target freestream density  [dimensionless]
            u_inf: Target freestream velocity (scalar or tuple)  [Δx/Δt]
            profile: Damping profile type:
                    'linear', 'quadratic', 'polynomial' (recommended), 'exponential'
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.thickness = thickness                        # [lattice units]
        self.sigma_max = min(max(strength, 0.0), 1.0)    # [dimensionless]
        self.rho_inf = rho_inf                           # [dimensionless]
        self.profile = profile.lower()
        self.shape = shape
        
        # Setup target velocity
        self._setup_target_velocity(u_inf)
        
        # Precompute damping coefficient array
        self._precompute_damping()
    
    def _setup_target_velocity(self, u_inf: Union[float, Tuple[float, float, float]]) -> None:
        """Setup target velocity for sponge zone"""
        xp = self.xp
        
        if isinstance(u_inf, (int, float)):
            # Scalar → (u_inf, 0, 0)
            self.u_inf = xp.array([float(u_inf), 0.0, 0.0], dtype=xp.float64)
        else:
            self.u_inf = xp.array([float(u_inf[0]), float(u_inf[1]), float(u_inf[2])], 
                                   dtype=xp.float64)
    
    def _precompute_damping(self) -> None:
        """Precompute damping coefficient array within sponge zone
        
        Creates σ array shaped for broadcasting with f[Q, Nx, Ny, Nz].
        σ = 0 at inner edge (interface with physical domain)
        σ = σ_max at boundary
        """
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for SpongeLayer")
        
        Nx, Ny, Nz = self.shape
        loc = self.location.value
        delta = self.thickness
        
        # Create normalized coordinate ξ ∈ [0, 1]
        # ξ = 0 at interface with physical domain
        # ξ = 1 at boundary
        
        if loc in ['xmin', 'west']:
            # Sponge from x=0 to x=delta-1
            # ξ = (delta-1-x)/(delta-1): 1 at x=0, 0 at x=delta-1
            x = xp.arange(min(delta, Nx), dtype=xp.float64)
            xi = (delta - 1 - x) / max(delta - 1, 1)
            self.sponge_slice = slice(0, min(delta, Nx))
            self.axis = 0
            
        elif loc in ['xmax', 'east']:
            # Sponge from x=Nx-delta to x=Nx-1
            # ξ = 0 at x=Nx-delta, ξ = 1 at x=Nx-1
            start = max(Nx - delta, 0)
            x = xp.arange(delta, dtype=xp.float64)
            xi = x / max(delta - 1, 1)
            self.sponge_slice = slice(start, Nx)
            self.axis = 0
            
        elif loc in ['ymin', 'south']:
            y = xp.arange(min(delta, Ny), dtype=xp.float64)
            xi = (delta - 1 - y) / max(delta - 1, 1)
            self.sponge_slice = slice(0, min(delta, Ny))
            self.axis = 1
            
        elif loc in ['ymax', 'north']:
            start = max(Ny - delta, 0)
            y = xp.arange(delta, dtype=xp.float64)
            xi = y / max(delta - 1, 1)
            self.sponge_slice = slice(start, Ny)
            self.axis = 1
            
        elif loc in ['zmin', 'bottom']:
            z = xp.arange(min(delta, Nz), dtype=xp.float64)
            xi = (delta - 1 - z) / max(delta - 1, 1)
            self.sponge_slice = slice(0, min(delta, Nz))
            self.axis = 2
            
        else:  # zmax, top
            start = max(Nz - delta, 0)
            z = xp.arange(delta, dtype=xp.float64)
            xi = z / max(delta - 1, 1)
            self.sponge_slice = slice(start, Nz)
            self.axis = 2
        
        # Compute damping profile σ = σ_max · g(ξ)
        sigma = self._compute_profile(xi)
        
        # Reshape for broadcasting: (delta,) → (delta, 1, 1) etc.
        if self.axis == 0:
            self.sigma = sigma.reshape(-1, 1, 1)
        elif self.axis == 1:
            self.sigma = sigma.reshape(1, -1, 1)
        else:  # axis == 2
            self.sigma = sigma.reshape(1, 1, -1)
    
    def _compute_profile(self, xi: 'npt.NDArray') -> 'npt.NDArray':
        """Compute damping profile σ(ξ) = σ_max · g(ξ)
        
        Args:
            xi: Normalized coordinate  [dimensionless, 0 ≤ ξ ≤ 1]
            
        Returns:
            Damping coefficients σ  [dimensionless]
        """
        xp = self.xp
        xi = xp.clip(xi, 0, 1)
        
        if self.profile == 'linear':
            # g(ξ) = ξ
            # Simple but C⁰ only (discontinuous derivative)
            g = xi
            
        elif self.profile == 'quadratic':
            # g(ξ) = ξ²
            # C¹ continuous at ξ=0
            g = xi**2
            
        elif self.profile == 'polynomial':
            # g(ξ) = 6ξ⁵ - 15ξ⁴ + 10ξ³
            # This is the unique polynomial satisfying:
            #   g(0) = 0, g(1) = 1
            #   g'(0) = g'(1) = 0
            #   g''(0) = g''(1) = 0
            # C² continuous, minimizes spurious reflections
            g = 6*xi**5 - 15*xi**4 + 10*xi**3
            
        elif self.profile == 'exponential':
            # g(ξ) = (1 - e^{-αξ}) / (1 - e^{-α})
            # Normalized exponential rise, steep near boundary
            alpha = 5.0  # Steepness parameter
            g = (1 - xp.exp(-alpha * xi)) / (1 - xp.exp(-alpha))
            
        else:
            raise ValueError(f"Unknown profile: {self.profile}. "
                           f"Choose from: linear, quadratic, polynomial, exponential")
        
        return self.sigma_max * g
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply sponge layer damping
        
        Blends computed distribution toward target equilibrium:
        
            f = (1 - σ) · f_computed + σ · f_target
        
        where f_target is equilibrium at (ρ∞, U∞).
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny, Nz)
               [dimensionless] - Modified in-place
        """
        xp = self.xp
        
        # Compute target equilibrium distribution f^eq(ρ∞, U∞)
        rho_target = self.rho_inf
        u_target = self.u_inf  # [Δx/Δt]
        
        usqr = float(xp.sum(u_target**2))  # |U∞|²  [Δx²/Δt²]
        cu = xp.einsum('d,di->i', u_target, self.c.astype(xp.float64))  # c·U∞
        
        # f^eq = w_i · ρ · (1 + 3(c·u) + 4.5(c·u)² - 1.5|u|²)  [dimensionless]
        f_target = self.w * rho_target * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        # Reshape f_target for broadcasting: (Q,) → (Q, 1, 1, 1)
        f_target = f_target.reshape(-1, 1, 1, 1)
        
        # Extract sponge region indices
        if self.axis == 0:
            sponge_region = (slice(None), self.sponge_slice, slice(None), slice(None))
        elif self.axis == 1:
            sponge_region = (slice(None), slice(None), self.sponge_slice, slice(None))
        else:  # axis == 2
            sponge_region = (slice(None), slice(None), slice(None), self.sponge_slice)
        
        # Apply damping: f = (1 - σ) · f + σ · f_target
        f[sponge_region] = (1 - self.sigma) * f[sponge_region] + self.sigma * f_target
    
    def get_info(self) -> str:
        """Return BC information string"""
        u_str = f"({float(self.u_inf[0]):.3f}, {float(self.u_inf[1]):.3f}, {float(self.u_inf[2]):.3f})"
        return (f"SpongeLayer at {self.location.value}:\n"
                f"    thickness={self.thickness} [Δx], σ_max={self.sigma_max}\n"
                f"    ρ∞={self.rho_inf}, U∞={u_str} [Δx/Δt]\n"
                f"    profile={self.profile}")


# =============================================================================
# Backward Compatibility Aliases (Deprecated)
# =============================================================================

class CharacteristicFarfield(FreestreamBC):
    """DEPRECATED: Use FreestreamBC instead.
    
    This class was misnamed - it does NOT implement true characteristic
    (LODI-based) boundary conditions. The actual implementation maintains
    fixed freestream velocity with pressure relaxation.
    
    For true characteristic BCs, see CharacteristicBC (to be implemented).
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "CharacteristicFarfield is deprecated and will be removed in v2.0. "
            "Use FreestreamBC instead. "
            "Note: This BC does NOT implement true characteristic analysis.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


# Note: AmbientBC has been removed.
# For pressure-based open boundaries with free velocity, use:
#   PressureRelaxationOutlet with K=0.3~0.5
# from .outlet import PressureRelaxationOutlet