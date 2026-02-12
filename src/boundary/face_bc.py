"""
Face Boundary Conditions for Domain Boundaries

Implements the "flat node" BC layer of the Palabos architecture.
Each face BC operates ONLY on flat nodes — edge/corner nodes are
handled separately by corner_bc.py with pure equilibrium (f = f_eq).

Available Face BCs:
    VelocityDirichletBC:  Prescribed velocity (inlet)
        - Equilibrium mode:   f = f_eq(ρ₀, u_target)        [external flow]
        - Regularized mode:   f = f_eq + f^(1)(Π^neq_int)   [channel flow]
    
    PressureDirichletBC:  Prescribed pressure (outlet)
        - ρ_bc = ρ_target + (1-K)(ρ_int - ρ_target)  [pressure relaxation]
        - f = f_eq(ρ_bc, u_int) + f^(1)(Π^neq_int)
    
    WallBC:  No-slip wall (domain boundary)
        - f = f_eq(ρ_ext, u=0) + f^(1)(Π^neq_int)
        - Replaces DomainWallBounceBack for domain walls
    
    FreestreamDirichletBC:  Far-field (both ρ and u prescribed)
        - ρ_bc = ρ∞ + (1-K)(ρ_int - ρ∞),  u = U∞
        - f = f_eq(ρ_bc, U∞) + f^(1)(Π^neq_int)

All BCs receive their slice from NodeMap at initialization, ensuring
they never touch edge/corner nodes.

References:
    - Latt & Chopard, Math. Comp. Sim. 72, 2006
    - Malaspinas, Chopard, Latt, Comp. Fluids 49, 2011

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .bc_config import FaceConfig, FaceLocation, BCType
from .node_map import NodeMap
from .regularized_utils import compute_f_eq, compute_Pi_neq, reconstruct_f_regularized


# =============================================================================
# Base Face BC
# =============================================================================

class FaceBC:
    """Base class for face boundary conditions.
    
    All face BCs share the pattern:
        1. Read interior neighbor distributions
        2. Determine target (ρ, u)
        3. Reconstruct f (equilibrium or regularized)
        4. Write to flat nodes only (edge/corner excluded)
    
    Subclasses implement _get_target_rho_u() to define what ρ and u
    to prescribe at the boundary.
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model
        config: FaceConfig for this face
        node_map: NodeMap with flat masks
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig, node_map: NodeMap) -> None:
        self.xp = xp
        self.lattice = lattice
        self.config = config
        self.node_map = node_map
        
        self.dim = lattice.dim
        self.Q = lattice.Q
        self.c = xp.asarray(lattice.c)
        self.w = xp.asarray(lattice.w)
        self.cs2 = lattice.cs2
        
        self.location = config.location
        self.use_regularized = config.use_regularized
    
    def apply(self, f: 'npt.NDArray', f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply face BC to flat nodes only.
        
        Args:
            f: Distribution after streaming, modified in-place. shape (Q, Nx, Ny[, Nz])
            f_post: Post-collision distribution (unused by most BCs)
        """
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply BC on flat nodes of a 2D face."""
        xp = self.xp
        loc = self.location
        
        # Get flat slice (edge nodes excluded)
        face_sl = self.node_map.get_face_slice_2d(loc)
        int_sl = self.node_map.get_interior_slice_2d(loc)
        
        # Step 1: Interior neighbor distributions
        f_int = f[:, int_sl[0], int_sl[1]]  # shape (Q, N_flat)
        
        # Step 2: Determine target ρ, u
        rho_t, u_t = self._get_target_rho_u(f_int)
        
        # Step 3: Reconstruct
        if self.use_regularized:
            Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
            f_new = reconstruct_f_regularized(
                xp, rho_t, u_t, Pi_neq, self.c, self.w, self.cs2
            )
        else:
            f_new = compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)
        
        # Step 4: Write ONLY to flat nodes
        f[:, face_sl[0], face_sl[1]] = f_new
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply BC on flat nodes of a 3D face."""
        xp = self.xp
        loc = self.location
        
        face_sl = self.node_map.get_face_slice_3d(loc)
        int_sl = self.node_map.get_interior_slice_3d(loc)
        
        f_int = f[:, int_sl[0], int_sl[1], int_sl[2]]
        
        rho_t, u_t = self._get_target_rho_u(f_int)
        
        if self.use_regularized:
            Pi_neq = compute_Pi_neq(xp, f_int, self.c, self.w, self.cs2)
            f_new = reconstruct_f_regularized(
                xp, rho_t, u_t, Pi_neq, self.c, self.w, self.cs2
            )
        else:
            f_new = compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)
        
        f[:, face_sl[0], face_sl[1], face_sl[2]] = f_new
    
    def _get_target_rho_u(self, f_int: 'npt.NDArray') -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Determine target (ρ, u) at this boundary.
        
        Must be implemented by subclasses.
        
        Args:
            f_int: Interior neighbor distribution, shape (Q, ...)
            
        Returns:
            (rho, u) where rho has shape (...) and u has shape (dim, ...)
        """
        raise NotImplementedError
    
    def get_info(self) -> str:
        """Return human-readable info string."""
        mode = "regularized" if self.use_regularized else "equilibrium"
        return (f"{self.__class__.__name__} at {self.location.value}: "
                f"{self.config.bc_type.value}, mode={mode}")


# =============================================================================
# Velocity Dirichlet (Inlet)
# =============================================================================

class VelocityDirichletBC(FaceBC):
    """Velocity Dirichlet Boundary Condition (Inlet)
    
    Prescribes velocity u = u_target at the boundary face.
    
    Two modes:
        Equilibrium (external flow, far inlet):
            f = f_eq(ρ₀, u_target)
            Simple, robust, no corner issues.
            
        Regularized (channel flow):
            f = f_eq(ρ₀, u_target) + f^(1)(Π^neq from interior)
            Preserves viscous stress for developed flow profiles.
    
    Density is prescribed as ρ₀ (typically 1.0).
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig, node_map: NodeMap) -> None:
        super().__init__(xp, lattice, config, node_map)
        
        # Build target velocity array on flat nodes
        self._setup_velocity()
    
    def _setup_velocity(self) -> None:
        """Setup velocity array matching flat node shape."""
        xp = self.xp
        velocity = self.config.velocity
        
        # Determine flat node shape by indexing a dummy array
        if self.dim == 2:
            face_sl = self.node_map.get_face_slice_2d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1]].shape
        else:
            face_sl = self.node_map.get_face_slice_3d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1], face_sl[2]].shape
        
        # Create velocity array: shape (dim, *flat_shape)
        self.u_target = xp.zeros((self.dim,) + flat_shape, dtype=xp.float64)
        
        if isinstance(velocity, (int, float)):
            # Scalar → x-direction (streamwise default)
            self.u_target[0, ...] = float(velocity)
        elif isinstance(velocity, (list, tuple)):
            for d in range(min(len(velocity), self.dim)):
                self.u_target[d, ...] = float(velocity[d])
        
        self.rho_target = self.config.density
    
    def _get_target_rho_u(self, f_int: 'npt.NDArray') -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Target: ρ = ρ₀ (prescribed), u = u_target (prescribed)."""
        xp = self.xp
        spatial_shape = f_int.shape[1:]
        
        rho = xp.full(spatial_shape, self.rho_target, dtype=xp.float64)
        return rho, self.u_target
    
    def get_info(self) -> str:
        u_mag = float(self.xp.max(self.xp.abs(self.u_target)))
        mode = "regularized" if self.use_regularized else "equilibrium"
        return (f"VelocityDirichletBC at {self.location.value}: "
                f"u={u_mag:.4f}, ρ={self.rho_target:.4f}, mode={mode}")


# =============================================================================
# Pressure Dirichlet (Outlet)
# =============================================================================

class PressureDirichletBC(FaceBC):
    """Pressure Dirichlet Boundary Condition (Outlet)
    
    Prescribes density ρ = ρ_target with pressure relaxation:
        ρ_bc = ρ_target + (1 - K)(ρ_int - ρ_target)
    
    Velocity is extrapolated from the interior neighbor.
    
    Pressure relaxation parameter K:
        K = 0:   Pure extrapolation (fully non-reflecting)
        K = 1:   Fixed pressure (strong reflection)
        K ~ 0.1: Good compromise (recommended)
    
    Always uses regularized reconstruction (f = f_eq + f^(1)).
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig, node_map: NodeMap) -> None:
        super().__init__(xp, lattice, config, node_map)
        
        self.rho_target = config.density        # [dimensionless]
        self.K = config.relax_coeff             # [dimensionless]
    
    def _get_target_rho_u(self, f_int: 'npt.NDArray') -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Target: ρ = relaxed, u = extrapolated from interior."""
        xp = self.xp
        
        # Interior macroscopics
        rho_int = xp.sum(f_int, axis=0)
        c_float = self.c.astype(xp.float64)
        u_int = xp.einsum('di,i...->d...', c_float, f_int) / (rho_int + 1e-30)
        
        # Pressure relaxation:  ρ_bc = ρ_target + (1-K)(ρ_int - ρ_target)
        rho_bc = self.rho_target + (1.0 - self.K) * (rho_int - self.rho_target)
        
        return rho_bc, u_int
    
    def get_info(self) -> str:
        return (f"PressureDirichletBC at {self.location.value}: "
                f"ρ_target={self.rho_target:.4f}, K={self.K:.3f}")


# =============================================================================
# Wall BC (Domain Boundary No-Slip)
# =============================================================================

class WallBC(FaceBC):
    """No-Slip Wall Boundary Condition (Domain Boundary)
    
    Prescribes u = 0 (or u = u_wall for moving walls).
    Density is extrapolated from the interior neighbor.
    
    Replaces the old DomainWallBounceBack with a regularized approach:
        f = f_eq(ρ_ext, u_wall) + f^(1)(Π^neq from interior)
    
    For internal obstacles (cylinders, airfoils), use HalfwayBounceBack
    from wall.py instead — it correctly handles curved boundaries.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig, node_map: NodeMap,
                 u_wall: Optional['npt.NDArray'] = None) -> None:
        super().__init__(xp, lattice, config, node_map)
        
        # Wall velocity (default: no-slip u=0)
        if self.dim == 2:
            face_sl = self.node_map.get_face_slice_2d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1]].shape
        else:
            face_sl = self.node_map.get_face_slice_3d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1], face_sl[2]].shape
        
        if u_wall is not None:
            self.u_wall = xp.asarray(u_wall, dtype=xp.float64)
        else:
            self.u_wall = xp.zeros((self.dim,) + flat_shape, dtype=xp.float64)
    
    def _get_target_rho_u(self, f_int: 'npt.NDArray') -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Target: ρ = extrapolated from interior, u = 0 (no-slip)."""
        xp = self.xp
        
        # Extrapolate density from interior
        rho_ext = xp.sum(f_int, axis=0)
        
        return rho_ext, self.u_wall
    
    def get_info(self) -> str:
        u_mag = float(self.xp.max(self.xp.abs(self.u_wall)))
        wall_type = "moving" if u_mag > 0 else "no-slip"
        return (f"WallBC ({wall_type}) at {self.location.value}")


# =============================================================================
# Freestream Dirichlet (Far-field)
# =============================================================================

class FreestreamDirichletBC(FaceBC):
    """Freestream Boundary Condition
    
    Both velocity and density are prescribed, with density relaxation:
        ρ_bc = ρ∞ + (1 - K)(ρ_int - ρ∞)
        u_bc = U∞ (fixed)
    
    Suitable for far-field boundaries in external aerodynamics.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig, node_map: NodeMap) -> None:
        super().__init__(xp, lattice, config, node_map)
        
        self.rho_inf = config.density
        self.K = config.relax_coeff
        
        # Build velocity array
        self._setup_velocity()
    
    def _setup_velocity(self) -> None:
        """Setup freestream velocity array."""
        xp = self.xp
        velocity = self.config.velocity
        
        if self.dim == 2:
            face_sl = self.node_map.get_face_slice_2d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1]].shape
        else:
            face_sl = self.node_map.get_face_slice_3d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1], face_sl[2]].shape
        
        self.u_inf = xp.zeros((self.dim,) + flat_shape, dtype=xp.float64)
        
        if isinstance(velocity, (int, float)):
            self.u_inf[0, ...] = float(velocity)
        elif isinstance(velocity, (list, tuple)):
            for d in range(min(len(velocity), self.dim)):
                self.u_inf[d, ...] = float(velocity[d])
    
    def _get_target_rho_u(self, f_int: 'npt.NDArray') -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Target: ρ = relaxed toward ρ∞, u = U∞ (fixed)."""
        xp = self.xp
        
        rho_int = xp.sum(f_int, axis=0)
        rho_bc = self.rho_inf + (1.0 - self.K) * (rho_int - self.rho_inf)
        
        return rho_bc, self.u_inf
    
    def get_info(self) -> str:
        u_mag = float(self.xp.max(self.xp.abs(self.u_inf)))
        return (f"FreestreamDirichletBC at {self.location.value}: "
                f"u∞={u_mag:.4f}, ρ∞={self.rho_inf:.4f}, K={self.K:.3f}")


# =============================================================================
# Factory
# =============================================================================

def create_face_bc(xp: 'ModuleType', lattice: 'object',
                   config: FaceConfig, node_map: NodeMap) -> FaceBC:
    """Create the appropriate FaceBC subclass from FaceConfig.
    
    Args:
        xp: Array module
        lattice: Lattice model
        config: FaceConfig describing this face
        node_map: NodeMap with flat/edge/corner classification
        
    Returns:
        FaceBC subclass instance
        
    Raises:
        ValueError: If bc_type is not supported as a face BC
    """
    bc_type = config.bc_type
    
    if bc_type == BCType.VELOCITY:
        return VelocityDirichletBC(xp, lattice, config, node_map)
    
    elif bc_type == BCType.PRESSURE:
        return PressureDirichletBC(xp, lattice, config, node_map)
    
    elif bc_type == BCType.WALL:
        return WallBC(xp, lattice, config, node_map)
    
    elif bc_type == BCType.FREESTREAM:
        return FreestreamDirichletBC(xp, lattice, config, node_map)
    
    elif bc_type == BCType.SPONGE:
        # Sponge is handled separately (volume-based, not face-based)
        raise ValueError(
            "Sponge layer is not a face BC. "
            "It should be handled separately as a volume-based damping layer."
        )
    
    else:
        raise ValueError(f"Unsupported BCType for face BC: {bc_type}")