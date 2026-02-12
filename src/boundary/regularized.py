"""
Regularized Boundary Conditions for LBM

This module implements Regularized Boundary Conditions following the
approach used by Palabos (Latt & Chopard, 2006; Malaspinas et al., 2011).

Key Principle:
    Unlike partial reconstruction (Zou-He style) which only replaces
    "unknown" distributions, the Regularized BC replaces ALL Q populations
    at boundary nodes using Chapman-Enskog expansion:

        f_i = f_i^eq(ρ, u) + f_i^(1)(Π^neq)     for ALL i = 0..Q-1

    This eliminates corner/edge conflicts entirely because a single
    formula determines all populations — there is nothing to "collide" with.

Physical Basis (Chapman-Enskog):
    f_i ≈ f_i^eq + f_i^(1)

    where the first-order non-equilibrium part is:

        f_i^(1) = (w_i / 2cs⁴) × Σ_αβ Q_iαβ × Π^neq_αβ

        Q_iαβ = c_iα × c_iβ - cs² × δ_αβ       [dimensionless]
        Π^neq_αβ = non-equilibrium stress tensor  [Δx²/Δt² density units]

    Π^neq is estimated from the interior neighbor node.

Advantages over Partial Reconstruction:
    - No known/unknown direction distinction → no corner conflicts
    - Stable at high Re (used in Palabos for Re up to 5×10⁴)
    - Naturally handles face, edge, and corner nodes with same formula
    - Consistent strain rate computation at boundaries

References:
    - Latt & Chopard, Math. Comp. Sim. 72, 2006 (original regularized LBM)
    - Malaspinas, Chopard, Latt, Comp. Fluids 49, 2011 (GRBC for corners)
    - Latt et al., Phys. Rev. E 77, 2008 (straight velocity boundaries)

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Union, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


def compute_Pi_neq(xp: 'ModuleType', f: 'npt.NDArray', 
                   c: 'npt.NDArray', w: 'npt.NDArray',
                   cs2: float) -> 'npt.NDArray':
    """Compute non-equilibrium stress tensor Π^neq from distribution function
    
    Physical meaning:
        Π^neq_αβ = Σ_i (f_i - f_i^eq) × c_iα × c_iβ
        
        This captures the viscous stress at a given node.
        In the Navier-Stokes limit: Π^neq_αβ ≈ -2ρ cs² (τ - 0.5) S_αβ
        where S_αβ is the strain rate tensor.
    
    Args:
        xp: Array module (numpy or cupy)
        f: Distribution function at a node/slice, shape (Q, ...)
        c: Lattice velocities, shape (dim, Q)  [Δx/Δt]
        w: Lattice weights, shape (Q,)  [dimensionless]
        cs2: Speed of sound squared = 1/3  [Δx²/Δt²]
        
    Returns:
        Π^neq tensor, shape (dim, dim, ...)  [density × Δx²/Δt²]
    """
    # Macroscopic quantities
    rho = xp.sum(f, axis=0)                          # [density]
    u = xp.einsum('di,i...->d...', c.astype(xp.float64), f) / (rho + 1e-30)  # [Δx/Δt]
    
    # Equilibrium distributions
    usqr = xp.sum(u * u, axis=0)                     # [Δx²/Δt²]
    cu = xp.einsum('di,d...->i...', c.astype(xp.float64), u)  # [Δx/Δt]
    
    ndim_extra = f.ndim - 1  # number of spatial dims in f
    w_shape = (-1,) + (1,) * ndim_extra
    w_bc = w.reshape(w_shape)
    
    f_eq = w_bc * rho * (1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * usqr / cs2)
    
    # Non-equilibrium part
    f_neq = f - f_eq  # [density units]
    
    # Stress tensor: Π^neq_αβ = Σ_i f_neq_i × c_iα × c_iβ
    dim = c.shape[0]
    spatial_shape = f.shape[1:]
    Pi_neq = xp.zeros((dim, dim) + spatial_shape, dtype=xp.float64)
    
    c_float = c.astype(xp.float64)
    for alpha in range(dim):
        for beta in range(alpha, dim):
            # Σ_i f_neq_i × c_iα × c_iβ
            cc = c_float[alpha] * c_float[beta]  # (Q,)
            cc_shape = (-1,) + (1,) * ndim_extra
            val = xp.sum(f_neq * cc.reshape(cc_shape), axis=0)
            Pi_neq[alpha, beta] = val
            if alpha != beta:
                Pi_neq[beta, alpha] = val  # symmetric
    
    return Pi_neq


def reconstruct_f_regularized(xp: 'ModuleType',
                               rho: 'npt.NDArray',
                               u: 'npt.NDArray',
                               Pi_neq: 'npt.NDArray',
                               c: 'npt.NDArray',
                               w: 'npt.NDArray',
                               cs2: float) -> 'npt.NDArray':
    """Reconstruct ALL distribution functions using regularized approach
    
    f_i = f_i^eq(ρ, u) + f_i^(1)(Π^neq)
    
    where:
        f_i^(1) = (w_i / 2cs⁴) × Σ_αβ Q_iαβ × Π^neq_αβ
        Q_iαβ = c_iα × c_iβ - cs² × δ_αβ
    
    Args:
        xp: Array module
        rho: Target density, shape (...)  [density]
        u: Target velocity, shape (dim, ...)  [Δx/Δt]
        Pi_neq: Non-equilibrium stress tensor, shape (dim, dim, ...)  [density×Δx²/Δt²]
        c: Lattice velocities, shape (dim, Q)  [Δx/Δt]
        w: Lattice weights, shape (Q,)  [dimensionless]
        cs2: Speed of sound squared  [Δx²/Δt²]
        
    Returns:
        Reconstructed distribution, shape (Q, ...)  [density]
    """
    Q = w.shape[0]
    dim = c.shape[0]
    spatial_shape = rho.shape
    ndim_extra = len(spatial_shape)
    
    c_float = c.astype(xp.float64)
    
    # =========================================================================
    # 1) Equilibrium part: f_i^eq(ρ, u)
    # =========================================================================
    usqr = xp.sum(u * u, axis=0)  # [Δx²/Δt²]
    cu = xp.einsum('di,d...->i...', c_float, u)  # [Δx/Δt] → (Q, ...)
    
    w_shape = (-1,) + (1,) * ndim_extra
    w_bc = w.reshape(w_shape)
    
    # f^eq = w_i ρ (1 + cu/cs² + cu²/(2cs⁴) - u²/(2cs²))
    f_eq = w_bc * rho * (1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * usqr / cs2)
    
    # =========================================================================
    # 2) Non-equilibrium part: f_i^(1) = (w_i / 2cs⁴) Σ_αβ Q_iαβ Π^neq_αβ
    #    Q_iαβ = c_iα c_iβ - cs² δ_αβ
    # =========================================================================
    cs4 = cs2 * cs2
    
    # Compute Q_iαβ × Π^neq_αβ contracted over α,β for each i
    f_neq = xp.zeros((Q,) + spatial_shape, dtype=xp.float64)
    
    for alpha in range(dim):
        for beta in range(dim):
            # Q_iαβ = c_iα c_iβ - cs² δ_αβ
            cc = c_float[alpha] * c_float[beta]  # (Q,)
            delta = cs2 if alpha == beta else 0.0
            Q_iab = cc - delta  # (Q,)
            
            Q_iab_shape = (-1,) + (1,) * ndim_extra
            f_neq += Q_iab.reshape(Q_iab_shape) * Pi_neq[alpha, beta]
    
    f_neq = w_bc / (2.0 * cs4) * f_neq
    
    return f_eq + f_neq


class RegularizedInlet(BoundaryCondition):
    """Regularized Velocity Inlet Boundary Condition
    
    Replaces ALL Q distributions at inlet nodes using:
        f_i = f_i^eq(ρ_target, u_target) + f_i^(1)(Π^neq_interior)
    
    This is equivalent to Palabos's createLocalBoundaryCondition with
    velocity Dirichlet condition.
    
    Key advantage: NO corner/edge conflicts. A single formula determines
    all Q populations, so there is nothing for wall BC to "fight" over.
    
    Supports both 2D and 3D domains.
    
    Algorithm:
        1. Extract Π^neq from interior neighbor (1 node inside domain)
        2. Reconstruct all Q distributions at boundary:
           f_i = f_i^eq(ρ_target, u_inlet) + f_i^(1)(Π^neq)
        
    References:
        Latt & Chopard, Math. Comp. Sim. 72, 2006
        Malaspinas et al., Comp. Fluids 49, 2011
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 velocity: Union[float, 'npt.NDArray'] = 0.1,
                 density: float = 1.0,
                 shape: Optional[tuple] = None) -> None:
        """Initialize regularized inlet
        
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
        self.cs2 = lattice.cs2
        
        if shape is None:
            raise ValueError("shape must be provided for RegularizedInlet")
        
        self._setup_velocity(velocity)
    
    def _get_incoming_indices(self) -> 'npt.NDArray':
        """Override: not used for regularized BC (we replace ALL populations)
        
        We keep the incoming indices for backward compatibility with base class,
        but they are not used in the apply logic.
        """
        return super()._get_incoming_indices()
    
    def _setup_velocity(self, velocity: Union[float, 'npt.NDArray']) -> None:
        """Setup velocity field at the inlet boundary
        
        Creates u_target array of shape (dim, *boundary_shape)
        """
        xp = self.xp
        loc = self.location.value
        
        if self.dim == 2:
            Nx, Ny = self.shape
            if loc in ['xmin', 'xmax', 'west', 'east']:
                boundary_shape = (Ny,)
                normal_dir = 0
            else:
                boundary_shape = (Nx,)
                normal_dir = 1
        else:
            Nx, Ny, Nz = self.shape
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
        
        # Inward sign: +1 if min face, -1 if max face
        sign = 1 if loc in ['xmin', 'ymin', 'zmin', 'west', 'south', 'bottom'] else -1
        
        if isinstance(velocity, (int, float)):
            self.u_target = xp.zeros((self.dim,) + boundary_shape, dtype=xp.float64)
            self.u_target[normal_dir] = sign * float(velocity)
        else:
            self.u_target = xp.asarray(velocity, dtype=xp.float64)
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply regularized inlet: replace ALL populations at boundary
        
        Physical process:
            1. Read interior neighbor distributions
            2. Compute Π^neq at interior (viscous stress information)
            3. Reconstruct ALL Q distributions at boundary using
               target (ρ, u) and interior Π^neq
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny) or (Q, Nx, Ny, Nz)
               Modified in-place.
        """
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply regularized inlet for 2D domain"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        
        # =====================================================================
        # Step 1: Get interior neighbor distributions
        # =====================================================================
        if loc in ['xmin', 'west']:
            f_int = f[:, 1, :]       # shape (Q, Ny) — 1 node inside from x=0
        elif loc in ['xmax', 'east']:
            f_int = f[:, Nx-2, :]    # 1 node inside from x=Nx-1
        elif loc in ['ymin', 'south']:
            f_int = f[:, :, 1]       # shape (Q, Nx)
        else:  # ymax, north
            f_int = f[:, :, Ny-2]
        
        # =====================================================================
        # Step 2: Compute Π^neq at interior (viscous stress tensor)
        # =====================================================================
        Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
        # shape: (dim, dim, Ny) or (dim, dim, Nx)
        
        # =====================================================================
        # Step 3: Reconstruct ALL Q distributions at boundary
        # =====================================================================
        f_reconstructed = reconstruct_f_regularized(
            xp, 
            rho=xp.full(self.boundary_shape, self.density, dtype=xp.float64),
            u=self.u_target,
            Pi_neq=Pi_neq,
            c=self.c, w=self.w, cs2=self.cs2
        )
        # shape: (Q, Ny) or (Q, Nx)
        
        # =====================================================================
        # Step 4: Write ALL populations at boundary (full replacement)
        # =====================================================================
        if loc in ['xmin', 'west']:
            f[:, 0, :] = f_reconstructed
        elif loc in ['xmax', 'east']:
            f[:, Nx-1, :] = f_reconstructed
        elif loc in ['ymin', 'south']:
            f[:, :, 0] = f_reconstructed
        else:  # ymax, north
            f[:, :, Ny-1] = f_reconstructed
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply regularized inlet for 3D domain"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        
        # Step 1: Interior neighbor
        if loc in ['xmin', 'west']:
            f_int = f[:, 1, :, :]
        elif loc in ['xmax', 'east']:
            f_int = f[:, Nx-2, :, :]
        elif loc in ['ymin', 'south']:
            f_int = f[:, :, 1, :]
        elif loc in ['ymax', 'north']:
            f_int = f[:, :, Ny-2, :]
        elif loc in ['zmin', 'bottom']:
            f_int = f[:, :, :, 1]
        else:  # zmax, top
            f_int = f[:, :, :, Nz-2]
        
        # Step 2: Π^neq from interior
        Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
        
        # Step 3: Reconstruct all
        f_reconstructed = reconstruct_f_regularized(
            xp,
            rho=xp.full(self.boundary_shape, self.density, dtype=xp.float64),
            u=self.u_target,
            Pi_neq=Pi_neq,
            c=self.c, w=self.w, cs2=self.cs2
        )
        
        # Step 4: Full replacement
        if loc in ['xmin', 'west']:
            f[:, 0, :, :] = f_reconstructed
        elif loc in ['xmax', 'east']:
            f[:, Nx-1, :, :] = f_reconstructed
        elif loc in ['ymin', 'south']:
            f[:, :, 0, :] = f_reconstructed
        elif loc in ['ymax', 'north']:
            f[:, :, Ny-1, :] = f_reconstructed
        elif loc in ['zmin', 'bottom']:
            f[:, :, :, 0] = f_reconstructed
        else:  # zmax, top
            f[:, :, :, Nz-1] = f_reconstructed
    
    def get_info(self) -> str:
        """Return information string"""
        u_mag = float(self.xp.max(self.xp.abs(self.u_target)))
        return (f"Regularized Inlet at {self.location.value}: "
                f"u={u_mag:.4f}, ρ={self.density:.4f}, "
                f"dim={'2D' if self.dim == 2 else '3D'} "
                f"(full reconstruction, all {self.Q} populations)")


class RegularizedOutlet(BoundaryCondition):
    """Regularized Pressure Outlet Boundary Condition
    
    Replaces ALL Q distributions at outlet nodes using:
        f_i = f_i^eq(ρ_bc, u_interior) + f_i^(1)(Π^neq_interior)
    
    with pressure relaxation:
        ρ_bc = ρ_target + (1 - K)(ρ_interior - ρ_target)
    
    Key advantage: Same as RegularizedInlet — no corner/edge conflicts.
    
    Supports both 2D and 3D domains.
    
    References:
        Latt & Chopard, Math. Comp. Sim. 72, 2006
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 rho_target: float = 1.0,
                 relax_coeff: float = 0.1,
                 shape: Optional[tuple] = None) -> None:
        """Initialize regularized outlet
        
        Args:
            xp: Array module
            lattice: Lattice model
            location: Boundary location
            rho_target: Target density  [dimensionless]
            relax_coeff: Pressure relaxation coefficient K  [dimensionless]
                K=0: Fully non-reflecting (pure extrapolation)
                K=1: Fixed pressure (strong reflection)
                K~0.1-0.3: Good compromise
            shape: Domain shape  [lattice units]
        """
        super().__init__(xp, lattice, location)
        
        self.rho_target = rho_target
        self.K = relax_coeff
        self.shape = shape
        self.dim = lattice.dim
        self.cs2 = lattice.cs2
        
        if shape is None:
            raise ValueError("shape must be provided for RegularizedOutlet")
        
        self._setup_boundary_shape()
    
    def _setup_boundary_shape(self) -> None:
        """Determine boundary face shape"""
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
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply regularized outlet: replace ALL populations at boundary
        
        Physical process:
            1. Read interior neighbor distributions
            2. Compute macroscopic (ρ, u) at interior
            3. Apply pressure relaxation: ρ_bc = ρ_target + (1-K)(ρ_int - ρ_target)
            4. Compute Π^neq at interior
            5. Reconstruct ALL Q distributions using (ρ_bc, u_int, Π^neq)
        
        Args:
            f: Distribution function, modified in-place
        """
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply regularized outlet for 2D"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        
        # Step 1: Interior neighbor
        if loc in ['xmin', 'west']:
            f_int = f[:, 1, :]
        elif loc in ['xmax', 'east']:
            f_int = f[:, Nx-2, :]
        elif loc in ['ymin', 'south']:
            f_int = f[:, :, 1]
        else:
            f_int = f[:, :, Ny-2]
        
        # Step 2: Macroscopic at interior
        rho_int = xp.sum(f_int, axis=0)
        c_float = self.c.astype(xp.float64)
        u_int = xp.einsum('di,i...->d...', c_float, f_int) / (rho_int + 1e-30)
        
        # Step 3: Pressure relaxation
        rho_bc = self.rho_target + (1.0 - self.K) * (rho_int - self.rho_target)
        
        # Step 4: Π^neq from interior
        Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
        
        # Step 5: Full reconstruction
        f_reconstructed = reconstruct_f_regularized(
            xp, rho=rho_bc, u=u_int, Pi_neq=Pi_neq,
            c=self.c, w=self.w, cs2=self.cs2
        )
        
        # Write all populations
        if loc in ['xmin', 'west']:
            f[:, 0, :] = f_reconstructed
        elif loc in ['xmax', 'east']:
            f[:, Nx-1, :] = f_reconstructed
        elif loc in ['ymin', 'south']:
            f[:, :, 0] = f_reconstructed
        else:
            f[:, :, Ny-1] = f_reconstructed
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply regularized outlet for 3D"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        
        # Step 1: Interior neighbor
        if loc in ['xmin', 'west']:
            f_int = f[:, 1, :, :]
        elif loc in ['xmax', 'east']:
            f_int = f[:, Nx-2, :, :]
        elif loc in ['ymin', 'south']:
            f_int = f[:, :, 1, :]
        elif loc in ['ymax', 'north']:
            f_int = f[:, :, Ny-2, :]
        elif loc in ['zmin', 'bottom']:
            f_int = f[:, :, :, 1]
        else:
            f_int = f[:, :, :, Nz-2]
        
        # Step 2: Macroscopic at interior
        rho_int = xp.sum(f_int, axis=0)
        c_float = self.c.astype(xp.float64)
        u_int = xp.einsum('di,i...->d...', c_float, f_int) / (rho_int + 1e-30)
        
        # Step 3: Pressure relaxation
        rho_bc = self.rho_target + (1.0 - self.K) * (rho_int - self.rho_target)
        
        # Step 4: Π^neq
        Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
        
        # Step 5: Full reconstruction
        f_reconstructed = reconstruct_f_regularized(
            xp, rho=rho_bc, u=u_int, Pi_neq=Pi_neq,
            c=self.c, w=self.w, cs2=self.cs2
        )
        
        # Write all populations
        if loc in ['xmin', 'west']:
            f[:, 0, :, :] = f_reconstructed
        elif loc in ['xmax', 'east']:
            f[:, Nx-1, :, :] = f_reconstructed
        elif loc in ['ymin', 'south']:
            f[:, :, 0, :] = f_reconstructed
        elif loc in ['ymax', 'north']:
            f[:, :, Ny-1, :] = f_reconstructed
        elif loc in ['zmin', 'bottom']:
            f[:, :, :, 0] = f_reconstructed
        else:
            f[:, :, :, Nz-1] = f_reconstructed
    
    def get_info(self) -> str:
        """Return information string"""
        return (f"Regularized Outlet at {self.location.value}: "
                f"ρ_target={self.rho_target:.4f}, K={self.K:.4f}, "
                f"dim={'2D' if self.dim == 2 else '3D'} "
                f"(full reconstruction, all {self.Q} populations)")


class RegularizedWall(BoundaryCondition):
    """Regularized No-Slip Wall Boundary Condition
    
    Replaces ALL Q distributions at wall nodes using:
        f_i = f_i^eq(ρ_extrapolated, u=0) + f_i^(1)(Π^neq_interior)
    
    This replaces bounce-back for domain walls. The density ρ is
    extrapolated from the interior to maintain consistency.
    
    For internal obstacles, standard bounce-back remains appropriate.
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 shape: tuple,
                 u_wall: Optional['npt.NDArray'] = None) -> None:
        """Initialize regularized wall
        
        Args:
            xp: Array module
            lattice: Lattice model
            location: Boundary location (ymin, ymax, etc.)
            shape: Domain shape  [lattice units]
            u_wall: Wall velocity (for moving walls)  [Δx/Δt]
                    Default: None (no-slip, u=0)
        """
        super().__init__(xp, lattice, location)
        
        self.shape = shape
        self.dim = lattice.dim
        self.cs2 = lattice.cs2
        
        self._setup_boundary_shape()
        
        # Wall velocity (default: no-slip = 0)
        if u_wall is not None:
            self.u_wall = xp.asarray(u_wall, dtype=xp.float64)
        else:
            self.u_wall = xp.zeros((self.dim,) + self.boundary_shape, dtype=xp.float64)
    
    def _setup_boundary_shape(self) -> None:
        """Determine boundary face shape"""
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
    
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply regularized wall: replace ALL populations
        
        Physical process:
            1. Read interior neighbor distributions
            2. Extrapolate density from interior
            3. Compute Π^neq at interior
            4. Reconstruct ALL populations with (ρ_ext, u_wall=0, Π^neq)
        """
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply regularized wall for 2D"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny = f.shape[1], f.shape[2]
        
        # Interior neighbor
        if loc in ['xmin', 'west']:
            f_int = f[:, 1, :]
        elif loc in ['xmax', 'east']:
            f_int = f[:, Nx-2, :]
        elif loc in ['ymin', 'south']:
            f_int = f[:, :, 1]
        else:
            f_int = f[:, :, Ny-2]
        
        # Extrapolate density from interior
        rho_ext = xp.sum(f_int, axis=0)
        
        # Π^neq from interior
        Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
        
        # Reconstruct with wall velocity (typically 0)
        f_reconstructed = reconstruct_f_regularized(
            xp, rho=rho_ext, u=self.u_wall, Pi_neq=Pi_neq,
            c=self.c, w=self.w, cs2=self.cs2
        )
        
        # Write
        if loc in ['xmin', 'west']:
            f[:, 0, :] = f_reconstructed
        elif loc in ['xmax', 'east']:
            f[:, Nx-1, :] = f_reconstructed
        elif loc in ['ymin', 'south']:
            f[:, :, 0] = f_reconstructed
        else:
            f[:, :, Ny-1] = f_reconstructed
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply regularized wall for 3D"""
        xp = self.xp
        loc = self.location.value
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        
        # Interior neighbor
        if loc in ['xmin', 'west']:
            f_int = f[:, 1, :, :]
        elif loc in ['xmax', 'east']:
            f_int = f[:, Nx-2, :, :]
        elif loc in ['ymin', 'south']:
            f_int = f[:, :, 1, :]
        elif loc in ['ymax', 'north']:
            f_int = f[:, :, Ny-2, :]
        elif loc in ['zmin', 'bottom']:
            f_int = f[:, :, :, 1]
        else:
            f_int = f[:, :, :, Nz-2]
        
        rho_ext = xp.sum(f_int, axis=0)
        Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
        
        f_reconstructed = reconstruct_f_regularized(
            xp, rho=rho_ext, u=self.u_wall, Pi_neq=Pi_neq,
            c=self.c, w=self.w, cs2=self.cs2
        )
        
        if loc in ['xmin', 'west']:
            f[:, 0, :, :] = f_reconstructed
        elif loc in ['xmax', 'east']:
            f[:, Nx-1, :, :] = f_reconstructed
        elif loc in ['ymin', 'south']:
            f[:, :, 0, :] = f_reconstructed
        elif loc in ['ymax', 'north']:
            f[:, :, Ny-1, :] = f_reconstructed
        elif loc in ['zmin', 'bottom']:
            f[:, :, :, 0] = f_reconstructed
        else:
            f[:, :, :, Nz-1] = f_reconstructed
    
    def get_info(self) -> str:
        """Return information string"""
        u_mag = float(self.xp.max(self.xp.abs(self.u_wall)))
        wall_type = "moving" if u_mag > 0 else "no-slip"
        return (f"Regularized Wall ({wall_type}) at {self.location.value}: "
                f"dim={'2D' if self.dim == 2 else '3D'} "
                f"(full reconstruction, all {self.Q} populations)")