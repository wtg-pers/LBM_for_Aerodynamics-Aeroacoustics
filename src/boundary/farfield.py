"""
Far-field Boundary Conditions for LBM

This module implements far-field boundary conditions for external aerodynamics
and aeroacoustics simulations. Different BCs are needed for different flight
conditions.

Physical Background:
====================

1. Forward Flight (U∞ ≠ 0):
   - Freestream provides energy input
   - Far-field velocity should be maintained at U∞
   - Use: CharacteristicFarfield

2. Hover (U∞ = 0):
   - No freestream; rotor generates all flow
   - Far-field velocity should develop freely based on rotor-induced flow
   - Pressure relaxes to ambient
   - Use: AmbientBC

3. Complex Unsteady Flows:
   - Acoustic wave propagation must not reflect at boundaries
   - Buffer zone gradually damps perturbations
   - Use: SpongeLayer

BC Selection Guide:
==================

    Forward Flight (U∞ ≠ 0):
    ┌───────────────────────────────────────────────────┐
    │   Inlet (xmin):     NonEquilibriumInlet          │
    │   Outlet (xmax):    CharacteristicOutlet         │
    │   Far-field (y,z):  CharacteristicFarfield       │
    └───────────────────────────────────────────────────┘

    Hover (U∞ = 0):
    ┌───────────────────────────────────────────────────┐
    │   All boundaries:   AmbientBC                    │
    │   (pressure-based, velocity free to develop)     │
    └───────────────────────────────────────────────────┘

References:
    - Poinsot & Lele, J. Comp. Phys. 101, 1992 (LODI/NRBC)
    - Izquierdo & Fueyo, J. Comp. Phys. 227, 2008 (LBM characteristic BC)
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


class CharacteristicFarfield(BoundaryCondition):
    """Characteristic Far-field BC for Forward Flight (U∞ ≠ 0)
    
    Maintains freestream velocity U∞ at far-field boundaries while allowing
    pressure/density perturbations to pass through with minimal reflection.
    
    Key Difference from CharacteristicOutlet:
    -----------------------------------------
    - CharacteristicOutlet: u_boundary = u_interior (velocity extrapolated)
    - CharacteristicFarfield: u_boundary = U∞ (velocity FIXED to freestream)
    
    This is crucial for external flow: if velocity extrapolates from interior
    (which may have wake or separated flow), the freestream decays and the
    driving force for vortex shedding disappears.
    
    Algorithm:
    ----------
    For incoming distributions at far-field:
    
        f_i = f_i^eq(ρ_relaxed, U∞) + f_i^neq(extrapolated)
        
    where:
        ρ_relaxed = ρ∞ + (1 - K)(ρ_interior - ρ∞)   [pressure relaxation]
        u_boundary = U∞                              [freestream velocity FIXED]
        
        K = 0: fully non-reflecting (pressure floats)
        K = 1: pressure fixed at p∞ (fully reflecting)
        K ~ 0.1: recommended for aerodynamics
    
    Attributes:
        rho_inf: Freestream density  [dimensionless, ρ/ρ₀]
        u_inf: Freestream velocity   [lattice units, Δx/Δt]
        K: Relaxation coefficient    [dimensionless]
        
    Example:
        >>> # ymin far-field with U∞ = 0.1 in x-direction
        >>> bc = CharacteristicFarfield(
        ...     xp, lattice, 'ymin',
        ...     rho_inf=1.0, 
        ...     u_inf=0.1,  # scalar → (0.1, 0, 0)
        ...     relax_coeff=0.1,
        ...     shape=(200, 100, 100)
        ... )
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: Union[str, BoundaryLocation],
                 rho_inf: float = 1.0,
                 u_inf: Union[float, Tuple[float, float, float]] = 0.1,
                 relax_coeff: float = 0.1,
                 shape: Optional[Tuple[int, int, int]] = None) -> None:
        """Initialize characteristic far-field BC
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            location: Boundary location ('ymin', 'ymax', 'zmin', 'zmax')
            rho_inf: Freestream density  [dimensionless]
            u_inf: Freestream velocity, either:
                   - scalar: interpreted as u_x component
                   - tuple: (u_x, u_y, u_z)  [lattice units]
            relax_coeff: Pressure relaxation coefficient K  [dimensionless]
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_inf = rho_inf
        self.K = relax_coeff
        self.shape = shape
        
        # Process velocity input
        self._setup_velocity(u_inf)
        
        # Precompute coefficients
        self._precompute_coefficients()
    
    def _setup_velocity(self, u_inf: Union[float, Tuple[float, float, float]]) -> None:
        """Setup freestream velocity array"""
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for farfield BC")
        
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
        
        # Create velocity array (3, *boundary_shape)
        self.u_inf_array = xp.zeros((3,) + boundary_shape, dtype=xp.float64)
        
        if isinstance(u_inf, (int, float)):
            # Scalar → assume flow in x-direction
            self.u_inf_array[0, :, :] = float(u_inf)
        else:
            # Tuple (u_x, u_y, u_z)
            self.u_inf_array[0, :, :] = float(u_inf[0])
            self.u_inf_array[1, :, :] = float(u_inf[1])
            self.u_inf_array[2, :, :] = float(u_inf[2])
    
    def _precompute_coefficients(self) -> None:
        """Precompute equilibrium coefficients"""
        xp = self.xp
        
        # For incoming directions
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
        
        # For all directions (computing interior equilibrium)
        self.c_all = self.c.astype(xp.float64)
        self.w_all = self.w.astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply characteristic far-field boundary condition
        
        Algorithm:
        1. Extract interior distribution (one layer inside boundary)
        2. Compute macroscopic at interior (ρ_int, u_int)
        3. Compute equilibrium at interior → extract f_neq
        4. Relax density: ρ_target = ρ∞ + (1-K)(ρ_int - ρ∞)
        5. Set velocity: u_target = U∞ (FIXED, not extrapolated!)
        6. Compute equilibrium at boundary with (ρ_target, U∞)
        7. Reconstruct: f = f_eq(ρ_target, U∞) + f_neq(interior)
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny, Nz)
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
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # =====================================================================
        # Step 3: Compute equilibrium at interior → f_neq
        # =====================================================================
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        f_neq_int = f_interior - f_eq_int
        
        # =====================================================================
        # Step 4 & 5: Target density (relaxed) and velocity (FIXED to U∞)
        # =====================================================================
        rho_target = self.rho_inf + (1 - self.K) * (rho_int - self.rho_inf)
        u_target = self.u_inf_array  # ← KEY: Fixed to freestream, NOT extrapolated!
        
        # =====================================================================
        # Step 6: Compute equilibrium at boundary for incoming directions
        # =====================================================================
        usqr_target = xp.sum(u_target**2, axis=0)
        cu_target = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu_target + 4.5*(cu_target**2) - 1.5*usqr_target)
        
        # =====================================================================
        # Step 7: Reconstruct: f = f_eq(target) + f_neq(interior)
        # =====================================================================
        incoming = self.incoming_indices
        f_incoming = f_eq_target + f_neq_int[incoming]
        
        # Apply to distribution
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
        return (f"CharacteristicFarfield at {self.location.value}:\n"
                f"    ρ∞={self.rho_inf}, U∞={u_str}, K={self.K}")


class AmbientBC(BoundaryCondition):
    """Ambient (Pressure-based Open) BC for Hover Conditions (U∞ = 0)
    
    For hover simulations where the rotor generates all flow, there is no
    meaningful freestream velocity to maintain. This BC:
    
    - Relaxes pressure (density) towards ambient p∞
    - Allows velocity to develop freely based on interior flow
    
    Key Difference from CharacteristicFarfield:
    -------------------------------------------
    - CharacteristicFarfield: u_boundary = U∞ (FIXED freestream)
    - AmbientBC: u_boundary = u_interior (FREE, extrapolated)
    
    This is essential for hover because:
    - Rotor induces inflow from above and downwash below
    - Velocity direction varies by boundary location
    - Forcing U∞ = 0 would artificially damp the induced flow
    
    Physical Picture (Hover):
    -------------------------
            
              Far-field (top)
                  ↓ ↓ ↓
              Induced inflow
                  ↓ ↓ ↓
            ┌─────────────┐
            │    ROTOR    │  ← rotating blades
            └─────────────┘
                  ↓ ↓ ↓
                Downwash
                  ↓ ↓ ↓
              Far-field (bottom)
    
    At each boundary:
    - Top: inflow develops naturally (induced by rotor)
    - Bottom: outflow develops naturally (downwash)
    - Sides: complex pattern adapts to local flow
    
    Algorithm:
    ----------
        f_i = f_i^eq(ρ_relaxed, u_interior) + f_i^neq(extrapolated)
        
    where:
        ρ_relaxed = ρ_ambient + (1 - K)(ρ_interior - ρ_ambient)  [pressure control]
        u_boundary = u_interior                                   [velocity FREE!]
    
    Attributes:
        rho_ambient: Ambient density (= p_ambient in LBM)  [dimensionless]
        K: Relaxation coefficient  [dimensionless]
    
    Example:
        >>> # Hover simulation with ambient BC on all boundaries
        >>> bc_top = AmbientBC(xp, lattice, 'zmax', rho_ambient=1.0, relax_coeff=0.5)
        >>> bc_bottom = AmbientBC(xp, lattice, 'zmin', rho_ambient=1.0, relax_coeff=0.5)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: Union[str, BoundaryLocation],
                 rho_ambient: float = 1.0,
                 relax_coeff: float = 0.5,
                 shape: Optional[Tuple[int, int, int]] = None) -> None:
        """Initialize ambient BC
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            location: Boundary location
            rho_ambient: Ambient density (= ambient pressure in LBM units)
                        [dimensionless, ρ/ρ₀]
            relax_coeff: Pressure relaxation coefficient K  [dimensionless]
                        K ~ 0.5 recommended for hover (stronger than farfield)
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_ambient = rho_ambient
        self.K = relax_coeff
        self.shape = shape
        
        # Determine boundary shape for array allocation
        self._setup_boundary_shape()
        
        # Precompute coefficients
        self._precompute_coefficients()
    
    def _setup_boundary_shape(self) -> None:
        """Determine boundary shape based on location"""
        if self.shape is None:
            raise ValueError("shape must be provided for AmbientBC")
        
        Nx, Ny, Nz = self.shape
        loc = self.location.value
        
        if loc in ['xmin', 'xmax', 'west', 'east']:
            self.boundary_shape = (Ny, Nz)
        elif loc in ['ymin', 'ymax', 'south', 'north']:
            self.boundary_shape = (Nx, Nz)
        else:  # zmin, zmax
            self.boundary_shape = (Nx, Ny)
    
    def _precompute_coefficients(self) -> None:
        """Precompute equilibrium coefficients"""
        xp = self.xp
        
        # For incoming directions
        incoming = self.incoming_indices
        self.c_incoming = self.c[:, incoming].astype(xp.float64)
        self.w_incoming = self.w[incoming].astype(xp.float64)
        
        # For all directions
        self.c_all = self.c.astype(xp.float64)
        self.w_all = self.w.astype(xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply ambient BC
        
        Algorithm:
        1. Get interior distribution
        2. Compute macroscopic at interior (ρ_int, u_int)
        3. Compute f_neq at interior
        4. Set target: ρ_target = relaxed, u_target = u_interior (FREE!)
        5. Reconstruct: f = f_eq(ρ_target, u_interior) + f_neq
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny, Nz)
        """
        xp = self.xp
        Nx, Ny, Nz = f.shape[1:]
        loc = self.location.value
        
        # =====================================================================
        # Step 1: Get interior distribution
        # =====================================================================
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
        
        # =====================================================================
        # Step 2: Compute macroscopic at interior
        # =====================================================================
        rho_int = xp.sum(f_interior, axis=0)
        u_int = xp.einsum('di,i...->d...', self.c_all, f_interior) / (rho_int + 1e-10)
        
        # =====================================================================
        # Step 3: Compute f_neq at interior
        # =====================================================================
        usqr_int = xp.sum(u_int**2, axis=0)
        cu_int = xp.einsum('di,d...->i...', self.c_all, u_int)
        w_bc = self.w_all.reshape((-1,) + (1,)*(f_interior.ndim - 1))
        f_eq_int = w_bc * rho_int * (1.0 + 3.0*cu_int + 4.5*(cu_int**2) - 1.5*usqr_int)
        
        f_neq_int = f_interior - f_eq_int
        
        # =====================================================================
        # Step 4: Target values
        #   - Density: relaxed towards ambient
        #   - Velocity: extrapolated from interior (FREE!)
        # =====================================================================
        rho_target = self.rho_ambient + (1 - self.K) * (rho_int - self.rho_ambient)
        u_target = u_int  # ← KEY DIFFERENCE: velocity is FREE, not fixed!
        
        # =====================================================================
        # Step 5: Compute equilibrium at boundary for incoming directions
        # =====================================================================
        usqr_target = xp.sum(u_target**2, axis=0)
        cu_target = xp.einsum('di,d...->i...', self.c_incoming, u_target)
        w_in = self.w_incoming.reshape((-1,) + (1,)*len(self.boundary_shape))
        f_eq_target = w_in * rho_target * (1.0 + 3.0*cu_target + 4.5*(cu_target**2) - 1.5*usqr_target)
        
        # =====================================================================
        # Step 6: Reconstruct
        # =====================================================================
        incoming = self.incoming_indices
        f_incoming = f_eq_target + f_neq_int[incoming]
        
        # Apply to distribution
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
        return (f"AmbientBC at {self.location.value}:\n"
                f"    ρ_ambient={self.rho_ambient}, K={self.K}\n"
                f"    (velocity free, pressure relaxed to ambient)")


class SpongeLayer(BoundaryCondition):
    """Sponge Layer (Buffer Zone) for Wave Damping
    
    Gradually damps flow perturbations towards a target state over a buffer
    zone. Essential for aeroacoustics where acoustic waves must not reflect.
    
    Algorithm:
    ----------
    Within the sponge zone (thickness δ from boundary):
    
        f = (1 - σ(x)) · f_computed + σ(x) · f_target
        
    where σ(x) is the damping function:
    
        σ(x) = σ_max · g(ξ),   ξ = (x - x_start) / δ
        
    Damping profiles g(ξ):
        - Linear:      g(ξ) = ξ
        - Quadratic:   g(ξ) = ξ²
        - Polynomial:  g(ξ) = 6ξ⁵ - 15ξ⁴ + 10ξ³  (smoothest)
        - Exponential: g(ξ) = 1 - exp(-αξ) / (1 - exp(-α))
    
    Physical Intuition:
    -------------------
    - Near physical domain: σ ≈ 0, flow develops naturally
    - Near boundary: σ → σ_max, flow forced towards target
    - Smooth transition prevents spurious reflections
    
    Trade-offs:
    -----------
    - More robust than characteristic BCs for complex flows
    - Requires buffer zone thickness (domain overhead)
    - Target state must be known (freestream or ambient)
    
    Attributes:
        thickness: Sponge layer thickness  [lattice units]
        strength: Maximum damping σ_max  [dimensionless, 0-1]
        rho_inf: Target density  [dimensionless]
        u_inf: Target velocity  [lattice units]
        profile: Damping profile type
        
    Example:
        >>> # Sponge layer at outlet with 20 lattice units thickness
        >>> sponge = SpongeLayer(
        ...     xp, lattice, 'xmax',
        ...     thickness=20, strength=0.8,
        ...     rho_inf=1.0, u_inf=(0.1, 0, 0),
        ...     profile='polynomial',
        ...     shape=(200, 100, 100)
        ... )
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
            strength: Maximum damping coefficient σ_max  [0, 1]
            rho_inf: Target freestream density  [dimensionless]
            u_inf: Target freestream velocity (scalar or tuple)  [lattice units]
            profile: Damping profile ('linear', 'quadratic', 'polynomial', 'exponential')
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.thickness = thickness
        self.sigma_max = min(max(strength, 0.0), 1.0)  # Clip to [0, 1]
        self.rho_inf = rho_inf
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
            self.u_inf = xp.array([float(u_inf), 0.0, 0.0], dtype=xp.float64)
        else:
            self.u_inf = xp.array([float(u_inf[0]), float(u_inf[1]), float(u_inf[2])], 
                                   dtype=xp.float64)
    
    def _precompute_damping(self) -> None:
        """Precompute damping coefficient array within sponge zone"""
        xp = self.xp
        
        if self.shape is None:
            raise ValueError("shape must be provided for SpongeLayer")
        
        Nx, Ny, Nz = self.shape
        loc = self.location.value
        delta = self.thickness
        
        # Create damping array based on location
        # σ = 0 at inner edge of sponge, σ = σ_max at boundary
        
        if loc in ['xmin', 'west']:
            # Sponge from x=0 to x=delta-1
            x = xp.arange(min(delta, Nx), dtype=xp.float64)
            xi = (delta - 1 - x) / max(delta - 1, 1)  # 1 at x=0, 0 at x=delta-1
            self.sponge_slice = slice(0, min(delta, Nx))
            self.axis = 0
            
        elif loc in ['xmax', 'east']:
            # Sponge from x=Nx-delta to x=Nx-1
            start = max(Nx - delta, 0)
            x = xp.arange(delta, dtype=xp.float64)
            xi = x / max(delta - 1, 1)  # 0 at start, 1 at x=Nx-1
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
        
        # Compute damping profile
        sigma = self._compute_profile(xi)
        
        # Reshape for broadcasting: add dimensions for other axes
        # For axis=0: shape (delta, 1, 1)
        # For axis=1: shape (1, delta, 1)
        # For axis=2: shape (1, 1, delta)
        if self.axis == 0:
            self.sigma = sigma.reshape(-1, 1, 1)
        elif self.axis == 1:
            self.sigma = sigma.reshape(1, -1, 1)
        else:
            self.sigma = sigma.reshape(1, 1, -1)
    
    def _compute_profile(self, xi: 'npt.NDArray') -> 'npt.NDArray':
        """Compute damping profile σ(ξ)
        
        Args:
            xi: Normalized coordinate [0, 1]
            
        Returns:
            Damping coefficients σ
        """
        xp = self.xp
        xi = xp.clip(xi, 0, 1)
        
        if self.profile == 'linear':
            g = xi
        elif self.profile == 'quadratic':
            g = xi**2
        elif self.profile == 'polynomial':
            # Smooth profile: g(ξ) = 6ξ⁵ - 15ξ⁴ + 10ξ³
            g = 6*xi**5 - 15*xi**4 + 10*xi**3
        elif self.profile == 'exponential':
            alpha = 5.0  # Steepness parameter
            g = (1 - xp.exp(-alpha * xi)) / (1 - xp.exp(-alpha))
        else:
            raise ValueError(f"Unknown profile: {self.profile}")
        
        return self.sigma_max * g
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply sponge layer damping
        
        Blends computed distribution towards target:
            f = (1 - σ) · f_computed + σ · f_target
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny, Nz)
        """
        xp = self.xp
        
        # Compute target equilibrium distribution
        rho_target = self.rho_inf
        u_target = self.u_inf
        
        usqr = float(xp.sum(u_target**2))
        cu = xp.einsum('d,di->i', u_target, self.c.astype(xp.float64))
        f_target = self.w * rho_target * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
        
        # Reshape f_target for broadcasting: (Q,) → (Q, 1, 1, 1)
        f_target = f_target.reshape(-1, 1, 1, 1)
        
        # Extract sponge region
        if self.axis == 0:
            sponge_region = (slice(None), self.sponge_slice, slice(None), slice(None))
        elif self.axis == 1:
            sponge_region = (slice(None), slice(None), self.sponge_slice, slice(None))
        else:
            sponge_region = (slice(None), slice(None), slice(None), self.sponge_slice)
        
        # Apply damping: f = (1 - σ) · f + σ · f_target
        f[sponge_region] = (1 - self.sigma) * f[sponge_region] + self.sigma * f_target
    
    def get_info(self) -> str:
        """Return BC information string"""
        u_str = f"({float(self.u_inf[0]):.3f}, {float(self.u_inf[1]):.3f}, {float(self.u_inf[2]):.3f})"
        return (f"SpongeLayer at {self.location.value}:\n"
                f"    thickness={self.thickness}, σ_max={self.sigma_max}\n"
                f"    ρ∞={self.rho_inf}, U∞={u_str}\n"
                f"    profile={self.profile}")