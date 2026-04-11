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
    
    WallBC:  No-slip wall via regularized reconstruction (domain boundary)
        - f = f_eq(ρ_ext, u=0) + f^(1)(Π^neq_int)
    
    DomainBounceBackBC:  No-slip wall via half-way bounce-back (domain boundary)
        - f_i(bdy) = f_ī(bdy) from post-collision state
    
    NeumannBC:  Zero-gradient extrapolation (open boundary)
        - f[:, boundary] = f[:, interior_neighbor]

All BCs receive their slice from NodeMap at initialization, ensuring
they never touch edge/corner nodes.

References:
    - Latt & Chopard, Math. Comp. Sim. 72, 2006
    - Malaspinas, Chopard, Latt, Comp. Fluids 49, 2011
    - Ladd, J. Fluid Mech. 271, 1994 (bounce-back)
    - Yang, Z., Comp. Math. Appl. 65, 2013 (Neumann outflow)

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
    
    Standard (regularized) face BCs share the pattern:
        1. Read interior neighbor distributions
        2. Determine target (ρ, u)
        3. Reconstruct f (equilibrium or regularized)
        4. Write to flat nodes only (edge/corner excluded)
    
    Subclasses implement _get_target_rho_u() to define what ρ and u
    to prescribe at the boundary.
    
    Non-standard BCs (DomainBounceBackBC, NeumannBC) override apply()
    directly, as their physics does not fit the reconstruct pattern.
    
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
            f_post: Post-collision distribution (needed by bounce-back BCs)
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

        # Equilibrium with constant target: use cached f_eq
        if not self.use_regularized and hasattr(self, '_cached_f_eq_3d'):
            f[:, face_sl[0], face_sl[1], face_sl[2]] = self._cached_f_eq_3d
            return

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
        
        Must be implemented by subclasses that use the regularized framework.
        
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

        if self.dim == 2:
            face_sl = self.node_map.get_face_slice_2d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1]].shape
        else:
            face_sl = self.node_map.get_face_slice_3d(self.location)
            dummy = xp.zeros(self.node_map.domain_shape)
            flat_shape = dummy[face_sl[0], face_sl[1], face_sl[2]].shape

        self.u_target = xp.zeros((self.dim,) + flat_shape, dtype=xp.float64)
        self.rho_target = xp.full(flat_shape, self.config.density, dtype=xp.float64)

        if isinstance(velocity, (int, float)):
            self.u_target[0, ...] = float(velocity)
        elif isinstance(velocity, (list, tuple)):
            for d in range(min(len(velocity), self.dim)):
                self.u_target[d, ...] = float(velocity[d])

        # Cache f_eq for equilibrium mode (constant target → same every step)
        if not self.use_regularized and self.dim == 3:
            from src.boundary.regularized_utils import compute_f_eq
            self._cached_f_eq_3d = compute_f_eq(
                xp, self.rho_target, self.u_target,
                self.c, self.w, self.cs2,
            )
    
    def _get_target_rho_u(self, f_int: 'npt.NDArray') -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Target: ρ = ρ₀ (prescribed), u = u_target (prescribed)."""
        return self.rho_target, self.u_target
    
    def get_info(self) -> str:
        u_mag = float(self.xp.max(self.xp.abs(self.u_target)))
        mode = "regularized" if self.use_regularized else "equilibrium"
        return (f"VelocityDirichletBC at {self.location.value}: "
                f"u={u_mag:.4f}, ρ₀={self.config.density:.4f} ({mode})")


# =============================================================================
# Pressure Dirichlet (Outlet)
# =============================================================================

class PressureDirichletBC(FaceBC):
    """Pressure Dirichlet Boundary Condition (Outlet)
    
    Prescribes density ρ (= pressure / cs²) at the boundary.
    Velocity is extrapolated from the interior neighbor.
    
    Pressure relaxation:
        ρ_bc = ρ_target + (1 - K)(ρ_int - ρ_target)  [dimensionless]
    
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
# Wall BC (Regularized Domain Wall)
# =============================================================================

class WallBC(FaceBC):
    """No-Slip Wall Boundary Condition — Regularized (Domain Boundary)
    
    Prescribes u = 0 (or u = u_wall for moving walls).
    Density is extrapolated from the interior neighbor.
    
    Reconstruction:
        f = f_eq(ρ_ext, u_wall) + f^(1)(Π^neq from interior)
    
    For internal obstacles (cylinders, airfoils), use HalfwayBounceBack
    from wall.py instead — it correctly handles curved boundaries.
    
    For standard bounce-back at domain walls, use DomainBounceBackBC.
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
        return (f"WallBC ({wall_type}, regularized) at {self.location.value}")


# =============================================================================
# Domain Bounce-Back (Standard Half-way Bounce-Back for Domain Walls)
# =============================================================================

class DomainBounceBackBC(FaceBC):
    """Standard Half-way Bounce-Back for Domain Wall Boundaries
    
    Physics:
        f_i(x_b, t+Δt) = f_ī(x_b, t)    for incoming directions i
        
    where ī = opposite(i), and f_ī(x_b, t) comes from post-collision
    state (f_post, before streaming).
    
    The wall is located at Δx/2 outside the boundary node,
    giving exact no-slip for Poiseuille flow.
    
    Unlike WallBC (regularized), this does NOT reconstruct f.
    It simply reflects incoming populations from f_post.
    
    Note:
        - ONLY overwrites incoming populations; outgoing remain as streamed.
        - Edge/corner nodes are handled by CornerBC (not touched here).
    
    For internal obstacles (cylinders, airfoils), use HalfwayBounceBack
    from wall.py — it handles curved boundaries with link-wise q values.
    
    References:
        - Ladd, J. Fluid Mech. 271, 1994
        - Kruger et al., "The Lattice Boltzmann Method", Ch. 5
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig, node_map: NodeMap) -> None:
        super().__init__(xp, lattice, config, node_map)
        self._setup_bounce_back()
    
    def _setup_bounce_back(self) -> None:
        """Precompute incoming direction indices and their opposites.
        
        Incoming directions: those whose lattice velocity points
        INTO the domain at this boundary face.
            - For min face (xmin): c[axis, i] > 0  (entering from left)
            - For max face (xmax): c[axis, i] < 0  (entering from right)
        """
        xp = self.xp
        c = self.lattice.c           # (dim, Q) numpy array
        opp = self.lattice.opp       # (Q,) opposite index array
        axis = self.location.axis
        is_min = self.location.is_min
        
        # Incoming = populations arriving FROM outside the domain
        incoming = []
        opposites = []
        
        for i in range(self.Q):
            c_normal = c[axis, i]    # component along face normal  [lattice velocity]
            if (is_min and c_normal > 0) or (not is_min and c_normal < 0):
                incoming.append(i)
                opposites.append(opp[i])
        
        # Store as arrays for vectorized indexing
        self.incoming = xp.asarray(incoming, dtype=xp.int64)    # populations TO fill
        self.opposites = xp.asarray(opposites, dtype=xp.int64)  # populations TO read
    
    def apply(self, f: 'npt.NDArray',
              f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply bounce-back on flat nodes of this wall face.
        
        For each incoming direction i with opposite ī:
            f[i, face_nodes] = f_post[ī, face_nodes]
        
        This overwrites ONLY the incoming populations, leaving
        outgoing populations (which streamed correctly) untouched.
        
        Physical process:
            1. Pre-streaming: population ī was heading toward the wall
            2. At the wall (Δx/2 outside): it bounces back
            3. Post-streaming: population i at boundary node = bounced ī
        
        Args:
            f: Post-streaming distribution, modified in-place (Q, Nx, Ny[, Nz])
            f_post: Post-collision distribution (Q, Nx, Ny[, Nz])
                    Required for correct half-way bounce-back.
                    Falls back to f if None (on-node BB, less accurate).
        """
        if self.dim == 2:
            face_sl = self.node_map.get_face_slice_2d(self.location)
            if f_post is not None:
                for k in range(len(self.incoming)):
                    i_in = int(self.incoming[k])
                    i_opp = int(self.opposites[k])
                    f[i_in, face_sl[0], face_sl[1]] = \
                        f_post[i_opp, face_sl[0], face_sl[1]]
            else:
                # In-place: save outgoing first
                saved = {}
                for k in range(len(self.incoming)):
                    i_opp = int(self.opposites[k])
                    saved[i_opp] = f[i_opp, face_sl[0], face_sl[1]].copy()
                for k in range(len(self.incoming)):
                    i_in = int(self.incoming[k])
                    i_opp = int(self.opposites[k])
                    f[i_in, face_sl[0], face_sl[1]] = saved[i_opp]
        else:
            face_sl = self.node_map.get_face_slice_3d(self.location)
            if f_post is not None:
                for k in range(len(self.incoming)):
                    i_in = int(self.incoming[k])
                    i_opp = int(self.opposites[k])
                    f[i_in, face_sl[0], face_sl[1], face_sl[2]] = \
                        f_post[i_opp, face_sl[0], face_sl[1], face_sl[2]]
            else:
                saved = {}
                for k in range(len(self.incoming)):
                    i_opp = int(self.opposites[k])
                    saved[i_opp] = f[i_opp, face_sl[0], face_sl[1], face_sl[2]].copy()
                for k in range(len(self.incoming)):
                    i_in = int(self.incoming[k])
                    i_opp = int(self.opposites[k])
                    f[i_in, face_sl[0], face_sl[1], face_sl[2]] = saved[i_opp]
    
    def get_info(self) -> str:
        n_inc = len(self.incoming)
        return (f"DomainBounceBackBC at {self.location.value}: "
                f"{n_inc} incoming directions, half-way BB")


# =============================================================================
# Neumann (Zero-Gradient Extrapolation) — INCOMING ONLY
# =============================================================================

class NeumannBC(FaceBC):
    """Zero-Gradient (Neumann) Boundary Condition
    
    Physics:
        ∂f_i/∂n = 0  at the boundary, for incoming directions only.
        
    Implementation:
        f[i, boundary] = f[i, interior_neighbor]   for incoming i only
        
    "Incoming" = populations arriving from outside the domain:
        - min face (e.g. xmin): c[axis, i] < 0  (coming from left/outside)
        - max face (e.g. xmax): c[axis, i] > 0  (coming from right/outside)
        
    Wait — careful with the convention!
    After streaming (pull scheme), boundary nodes have:
        - Outgoing pops (pointing away from domain): correctly pulled from interior ✓
        - Tangential pops (parallel to face): correctly pulled ✓
        - Incoming pops (pointing into domain): pulled via periodic wrap → garbage ✗
    
    "Incoming to the boundary node from outside" means:
        - For xmax face: c_x < 0 populations (they would come from x > N-1, outside)
        - For xmin face: c_x > 0 populations (they would come from x < 0, outside)
    
    So incoming = populations whose lattice velocity points INTO the domain
    (same definition as DomainBounceBackBC).
    
    Why only incoming?
        - Outgoing populations were correctly streamed from interior → preserve them
        - Overwriting outgoing would distort the mass flux leaving the domain
        - This causes cumulative mass drift → divergence
    
    Best suited for:
        - Low-Re outflows where reflected waves are weak
        - Quick validation runs  
        - Situations where PressureDirichletBC is too reflective
    
    Limitation:
        - Does not control outlet pressure → mass may drift slowly over time
        - More reflective than pressure relaxation at high Ma
    
    References:
        - Yang, Z., "Lattice Boltzmann outflow treatments...",
          Comp. Math. Appl. 65, 2013
        - Latt et al., "Efficient supersonic flows through high-order
          guided equilibrium with lattice Boltzmann", Phil. Trans. R. Soc. A, 2020
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig, node_map: NodeMap) -> None:
        super().__init__(xp, lattice, config, node_map)
        self._setup_incoming()
    
    def _setup_incoming(self) -> None:
        """Precompute incoming direction indices.
        
        Incoming = populations that would arrive from OUTSIDE the domain.
        After streaming with periodic wrap, these contain garbage values.
        
        Convention (same as DomainBounceBackBC):
            min face (xmin): c[axis, i] > 0  → population enters from left
            max face (xmax): c[axis, i] < 0  → population enters from right
        """
        c = self.lattice.c           # (dim, Q) numpy/cupy array
        axis = self.location.axis
        is_min = self.location.is_min
        
        incoming = []
        for i in range(self.Q):
            c_normal = c[axis, i]    # velocity component along face normal
            if (is_min and c_normal > 0) or (not is_min and c_normal < 0):
                incoming.append(i)
        
        self.incoming = incoming     # list of int indices
    
    def apply(self, f: 'npt.NDArray',
              f_post: Optional['npt.NDArray'] = None) -> None:
        """Copy incoming distributions from interior neighbor to boundary.
        
        For each incoming direction i:
            f[i, face] = f[i, interior]
        
        Outgoing and tangential populations are left untouched — they were
        correctly populated by streaming.
        
        Args:
            f: Post-streaming distribution, modified in-place
            f_post: Unused (Neumann doesn't need pre-streaming state)
        """
        if self.dim == 2:
            face_sl = self.node_map.get_face_slice_2d(self.location)
            int_sl = self.node_map.get_interior_slice_2d(self.location)
            for i in self.incoming:
                f[i, face_sl[0], face_sl[1]] = f[i, int_sl[0], int_sl[1]]
        else:
            face_sl = self.node_map.get_face_slice_3d(self.location)
            int_sl = self.node_map.get_interior_slice_3d(self.location)
            for i in self.incoming:
                f[i, face_sl[0], face_sl[1], face_sl[2]] = \
                    f[i, int_sl[0], int_sl[1], int_sl[2]]
    
    def get_info(self) -> str:
        n_inc = len(self.incoming)
        return (f"NeumannBC (zero-gradient, {n_inc} incoming) "
                f"at {self.location.value}")


# =============================================================================
# Factory
# =============================================================================

def create_face_bc(xp: 'ModuleType', lattice: 'object',
                   config: FaceConfig, node_map: NodeMap) -> FaceBC:
    """Create the appropriate FaceBC subclass from FaceConfig.
    
    Routing logic:
        1. Method-first: check for special BCs (bounce_back, neumann)
           that bypass the regularized reconstruction framework.
        2. BCType-based: route to regularized BC classes.
    
    Args:
        xp: Array module
        lattice: Lattice model
        config: FaceConfig describing this face
        node_map: NodeMap with flat/edge/corner classification
        
    Returns:
        FaceBC subclass instance
        
    Raises:
        ValueError: If bc_type/method combination is not supported
    """
    method = config.method.lower()
    bc_type = config.bc_type
    
    # ── Special BCs: bypass regularized framework ──
    
    if method in ('bounce_back', 'hwbb'):
        return DomainBounceBackBC(xp, lattice, config, node_map)
    
    if bc_type == BCType.NEUMANN:
        return NeumannBC(xp, lattice, config, node_map)
    
    # ── Standard regularized framework ──
    
    if bc_type == BCType.VELOCITY:
        return VelocityDirichletBC(xp, lattice, config, node_map)
    
    elif bc_type == BCType.PRESSURE:
        return PressureDirichletBC(xp, lattice, config, node_map)
    
    elif bc_type == BCType.WALL:
        return WallBC(xp, lattice, config, node_map)
    
    elif bc_type == BCType.SPONGE:
        raise ValueError(
            "Sponge layer is volume-based, not a face BC. "
            "It is handled separately by DomainBCManager."
        )
    
    else:
        raise ValueError(f"Unsupported BCType: {bc_type} (method='{method}')")