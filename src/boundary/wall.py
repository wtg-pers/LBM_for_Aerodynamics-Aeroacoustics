"""
Half-way Bounce-Back Wall Boundary Condition for LBM

This module implements the half-way bounce-back scheme for solid walls
and internal obstacles (e.g., cylinders, spheres, arbitrary geometries).

Physical Principle:
    The wall is assumed to be located halfway between fluid and solid nodes.
    When a distribution streams towards a solid node, it bounces back to
    the originating fluid node in the opposite direction.
    
    For a fluid node x_f with solid neighbor in direction i:
        f_ī(x_f, t+Δt) = f_i*(x_f, t)
    
    where f_i* is the post-collision distribution and ī is the opposite of i.

Implementation:
    This is applied AFTER streaming. We identify fluid nodes that have
    solid neighbors, and for each such direction, we set the incoming
    distribution (which would have come from the solid) to the outgoing
    distribution from the previous step.

Accuracy:
    - Second-order accurate when wall is exactly at q = 0.5
    - First-order for other wall positions (use Bouzidi for better accuracy)

References:
    - Ladd, J. Fluid Mech. 271, 1994
    - Sukop & Thorne, "Lattice Boltzmann Modeling", 2006

Author: LBM Development Team
Date: 2026-01
"""

from typing import TYPE_CHECKING, Tuple, Optional, Callable
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class HalfwayBounceBack:
    """Half-way Bounce-Back for Internal Walls and Obstacles
    
    Implements no-slip wall boundary condition using the half-way
    bounce-back scheme. Suitable for internal obstacles like cylinders,
    spheres, or arbitrary geometries defined by a solid mask.
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model
        solid_mask: Boolean array, True = solid node
        opp: Opposite direction indices
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 solid_mask: 'npt.NDArray',
                 periodic_axes: Optional[Tuple[int, ...]] = None) -> None:
        """Initialize half-way bounce-back

        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q19, D3Q27)
            solid_mask: Boolean array of shape (Nx, Ny, Nz) or (Nx, Ny)
                       True = solid (wall), False = fluid
            periodic_axes: axes whose wrap is a contract — see
                q_fraction.compute_needs_bounce. None = legacy torus
                (every axis wraps); production callers pass it explicitly.
        """
        self.xp = xp
        self.lattice = lattice
        self.solid_mask = xp.asarray(solid_mask, dtype=bool)
        self.c = xp.asarray(lattice.c)
        self.opp = xp.asarray(lattice.opp)
        self.Q = lattice.Q
        self.dim = lattice.dim
        self._periodic_axes = periodic_axes

        # Precompute boundary information for efficiency
        self._precompute_boundary_links()

    def _precompute_boundary_links(self) -> None:
        """Precompute fluid-solid boundary links

        For each fluid node adjacent to a solid, identify which directions
        point into solid nodes. These are the directions that need bounce-back.

        Delegates to q_fraction.compute_needs_bounce — the single link
        enumeration (its dense form). Keeping a private roll loop here is
        how the seam (torus wrap) policy drifts between HWBB, IBB and the
        force calculator.
        """
        from src.boundary.q_fraction import compute_needs_bounce

        self.needs_bounce = compute_needs_bounce(
            self.xp, self.lattice, self.solid_mask,
            periodic_axes=self._periodic_axes,
        )
        self.n_boundary_links = int(self.xp.sum(self.needs_bounce))
    
    def apply(self, f: 'npt.NDArray', f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply half-way bounce-back boundary condition.

        Physical process:
            At fluid nodes adjacent to solid, the outgoing distribution
            toward the wall bounces back:  f_ī(x) = f_i*(x)

        Two modes:
            f_post provided: f_source = f_post (separate post-collision array)
            f_post = None:   f_source = f itself (post-collision stored in f,
                             e.g. streaming-fused kernel). Uses save-then-write
                             to avoid direction-pair conflicts.

        Args:
            f: Distribution function, shape (Q, Nx, Ny[, Nz]).
               Modified in-place.
            f_post: Post-collision distribution. If None, reads from f.
        """
        xp = self.xp
        opp = self.opp

        if f_post is not None:
            # Standard path: separate source array, no conflict
            for i in range(1, self.Q):
                i_opp = int(opp[i])
                mask = self.needs_bounce[i]
                if xp.any(mask):
                    f[i_opp][mask] = f_post[i][mask]
        else:
            # In-place path: f is both source and target.
            # Save outgoing distributions first to avoid direction-pair
            # conflicts (e.g. writing f[opp_q] before reading f[opp_q]
            # when both q and opp_q are boundary at the same node).
            saved = {}
            for i in range(1, self.Q):
                mask = self.needs_bounce[i]
                if xp.any(mask):
                    saved[i] = f[i][mask].copy()

            for i, vals in saved.items():
                i_opp = int(opp[i])
                f[i_opp][self.needs_bounce[i]] = vals
    
    def apply_simple(self, f: 'npt.NDArray') -> None:
        """Simplified bounce-back using symmetry
        
        For the half-way bounce-back, when a fluid node's neighbor is solid:
        - After streaming, f[i] at that node is invalid (came from solid)
        - We set f[i] = f[opp[i]] (bounce back)
        
        This simplified version swaps distributions in boundary links.
        Works well for steady-state but may have small errors for transient.
        
        Args:
            f: Distribution function after streaming (modified in-place)
        """
        xp = self.xp
        opp = self.opp
        
        # For each direction pair (i, opp[i]), swap at boundary links
        # Only process each pair once (i < opp[i])
        processed = set()
        
        for i in range(1, self.Q):
            i_opp = int(opp[i])
            
            if (i, i_opp) in processed or (i_opp, i) in processed:
                continue
            processed.add((i, i_opp))
            
            # Nodes where direction i points to solid
            mask_i = self.needs_bounce[i]
            # Nodes where direction opp[i] points to solid  
            mask_opp = self.needs_bounce[i_opp]
            
            # Swap f[i] and f[opp[i]] at boundary nodes
            # But only where the respective direction hits a wall
            
            # At nodes in mask_i: f[i] is invalid, should be f[opp[i]]
            # At nodes in mask_opp: f[opp[i]] is invalid, should be f[i]
            
            if xp.any(mask_i):
                temp = f[i][mask_i].copy()
                f[i][mask_i] = f[i_opp][mask_i]
                # Only swap back if mask_opp overlaps
                # Actually for internal walls, they shouldn't overlap
            
            if xp.any(mask_opp):
                f[i_opp][mask_opp] = f[i][mask_opp]
    
    def reset_solid_nodes(self, f: 'npt.NDArray', 
                          rho: Optional['npt.NDArray'] = None,
                          u: Optional['npt.NDArray'] = None) -> None:
        """Reset distribution inside solid nodes to equilibrium
        
        This prevents accumulation of non-physical values inside solid regions.
        Should be called every time step after apply().
        
        Args:
            f: Distribution function (modified in-place)
            rho: Density field (for computing equilibrium). If None, uses rho=1.
            u: Velocity field (for computing equilibrium). If None, uses u=0.
        """
        xp = self.xp
        w = xp.asarray(self.lattice.w)
        
        # Default: rho=1, u=0 inside solid
        if rho is None:
            rho_solid = 1.0
        else:
            rho_solid = rho[self.solid_mask] if hasattr(rho, '__getitem__') else rho
        
        # Set f to equilibrium with u=0 inside solid
        # f_eq = w_i * rho * (1 + 0 + 0 - 0) = w_i * rho
        for i in range(self.Q):
            f[i][self.solid_mask] = w[i] * rho_solid
    
    def apply_with_reset(self, f: 'npt.NDArray', 
                         f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply bounce-back and reset solid nodes in one call
        
        Convenience method that combines apply() and reset_solid_nodes().
        
        Args:
            f: Distribution function after streaming (modified in-place)
            f_post: Post-collision distribution (optional)
        """
        self.apply(f, f_post)
        self.reset_solid_nodes(f)
    
    def mask_velocity(self, u: 'npt.NDArray') -> None:
        """Set velocity to zero inside solid nodes
        
        Should be called after computing macroscopic variables.
        
        Args:
            u: Velocity field, shape (dim, Nx, Ny[, Nz]), modified in-place
        """
        for d in range(self.dim):
            u[d][self.solid_mask] = 0.0
    
    def mask_density(self, rho: 'npt.NDArray', rho_solid: float = 1.0) -> None:
        """Set density to fixed value inside solid nodes
        
        Args:
            rho: Density field, shape (Nx, Ny[, Nz]), modified in-place
            rho_solid: Density value inside solid (default 1.0)
        """
        rho[self.solid_mask] = rho_solid
    
    def get_solid_mask(self) -> 'npt.NDArray':
        """Return the solid mask array"""
        return self.solid_mask
    
    def get_info(self) -> str:
        """Return information string about the boundary"""
        solid_count = int(self.xp.sum(self.solid_mask))
        total_nodes = self.solid_mask.size
        
        return (f"Half-way Bounce-Back Wall:\n"
                f"  Solid nodes: {solid_count:,} ({100*solid_count/total_nodes:.2f}%)\n"
                f"  Boundary links: {self.n_boundary_links:,}")


class MovingWallBounceBack(HalfwayBounceBack):
    """Half-way Bounce-Back with Moving Wall Velocity
    
    Extension of HWBB for walls with non-zero velocity (e.g., lid-driven cavity).
    
    The bounce-back formula becomes:
        f_ī(x_f, t+Δt) = f_i*(x_f, t) - 2 * w_i * ρ * (c_i · u_wall) / cs²
    
    where u_wall is the wall velocity.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 solid_mask: 'npt.NDArray',
                 wall_velocity: 'npt.NDArray') -> None:
        """Initialize moving wall bounce-back
        
        Args:
            xp: Array module
            lattice: Lattice model
            solid_mask: Boolean solid mask
            wall_velocity: Wall velocity, shape (dim,) for uniform
                          or (dim, Nx, Ny, Nz) for spatially varying
        """
        super().__init__(xp, lattice, solid_mask)
        
        self.w = xp.asarray(lattice.w)
        self.cs2 = lattice.cs2
        
        # Process wall velocity
        wall_velocity = xp.asarray(wall_velocity, dtype=xp.float64)
        if wall_velocity.ndim == 1:
            # Uniform velocity - broadcast to spatial shape
            self.u_wall = wall_velocity.reshape((self.dim,) + (1,)*len(solid_mask.shape))
        else:
            self.u_wall = wall_velocity
    
    def apply(self, f: 'npt.NDArray', rho: Optional['npt.NDArray'] = None,
              f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply moving wall bounce-back
        
        Args:
            f: Distribution function after streaming
            rho: Density field (needed for momentum correction)
            f_post: Post-collision distribution (optional)
        """
        xp = self.xp
        opp = self.opp
        c = self.c.astype(xp.float64)
        w = self.w
        
        if rho is None:
            rho = xp.sum(f, axis=0)
        
        if f_post is not None:
            # Standard path: separate source array
            for i in range(1, self.Q):
                i_opp = int(opp[i])
                mask = self.needs_bounce[i]
                if xp.any(mask):
                    c_dot_u = xp.sum(c[:, i:i+1] * self.u_wall, axis=0)
                    correction = 2 * w[i] * rho * c_dot_u / self.cs2
                    f[i_opp][mask] = f_post[i][mask] - correction[mask]
        else:
            # In-place path: save outgoing distributions first
            saved = {}
            for i in range(1, self.Q):
                mask = self.needs_bounce[i]
                if xp.any(mask):
                    saved[i] = f[i][mask].copy()

            for i, vals in saved.items():
                i_opp = int(opp[i])
                mask = self.needs_bounce[i]
                c_dot_u = xp.sum(c[:, i:i+1] * self.u_wall, axis=0)
                correction = 2 * w[i] * rho * c_dot_u / self.cs2
                f[i_opp][mask] = vals - correction[mask]
