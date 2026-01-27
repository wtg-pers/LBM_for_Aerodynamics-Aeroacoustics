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
                 solid_mask: 'npt.NDArray') -> None:
        """Initialize half-way bounce-back
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q19, D3Q27)
            solid_mask: Boolean array of shape (Nx, Ny, Nz) or (Nx, Ny)
                       True = solid (wall), False = fluid
        """
        self.xp = xp
        self.lattice = lattice
        self.solid_mask = xp.asarray(solid_mask, dtype=bool)
        self.c = xp.asarray(lattice.c)
        self.opp = xp.asarray(lattice.opp)
        self.Q = lattice.Q
        self.dim = lattice.dim
        
        # Precompute boundary information for efficiency
        self._precompute_boundary_links()
    
    def _precompute_boundary_links(self) -> None:
        """Precompute fluid-solid boundary links
        
        For each fluid node adjacent to a solid, identify which directions
        point into solid nodes. These are the directions that need bounce-back.
        
        This precomputation allows vectorized application of bounce-back.
        """
        xp = self.xp
        c = self.c
        solid = self.solid_mask
        
        shape = solid.shape
        
        if self.dim == 3:
            Nx, Ny, Nz = shape
            
            # For each direction i, find fluid nodes whose neighbor in direction i is solid
            # These are the "boundary links" that need bounce-back
            
            # Store as list of (direction, fluid_indices) pairs
            # Or more efficiently: store masks for each direction
            
            self.bounce_mask = []  # List of (Q,) masks for each spatial point
            
            # Create shifted solid masks for each direction
            # If solid_mask shifted by -c_i has True at x, then x+c_i is solid
            # So fluid node x streaming in direction i hits a wall
            
            self.needs_bounce = xp.zeros((self.Q,) + shape, dtype=bool)
            
            for i in range(self.Q):
                if i == 0:  # Rest direction, no streaming
                    continue
                    
                cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
                
                # Shift solid mask by -c_i (opposite direction)
                # This tells us: for each point x, is x + c_i solid?
                shifted_solid = xp.roll(
                    xp.roll(
                        xp.roll(solid, -cx, axis=0),
                        -cy, axis=1),
                    -cz, axis=2)
                
                # Boundary link: fluid node where neighbor in direction i is solid
                # needs_bounce[i, x, y, z] = True means:
                #   - (x, y, z) is fluid
                #   - (x + cx, y + cy, z + cz) is solid
                #   - So f_i at (x,y,z) should bounce back to f_opp[i]
                self.needs_bounce[i] = (~solid) & shifted_solid
            
            # Count boundary links for info
            self.n_boundary_links = int(xp.sum(self.needs_bounce))
            
        elif self.dim == 2:
            Nx, Ny = shape
            
            self.needs_bounce = xp.zeros((self.Q,) + shape, dtype=bool)
            
            for i in range(self.Q):
                if i == 0:
                    continue
                    
                cx, cy = int(c[0, i]), int(c[1, i])
                
                shifted_solid = xp.roll(
                    xp.roll(solid, -cx, axis=0),
                    -cy, axis=1)
                
                self.needs_bounce[i] = (~solid) & shifted_solid
            
            self.n_boundary_links = int(xp.sum(self.needs_bounce))
    
    def apply(self, f: 'npt.NDArray', f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply half-way bounce-back boundary condition
        
        IMPORTANT: This must be called AFTER streaming.
        
        The bounce-back requires the post-collision distribution from
        the previous time step. Two usage patterns:
        
        Pattern 1 (Simple): 
            Store f_post before streaming, pass to apply()
            
        Pattern 2 (Memory efficient):
            For symmetric equilibrium, use current f as approximation
        
        Args:
            f: Distribution function after streaming, shape (Q, Nx, Ny[, Nz])
               Modified in-place.
            f_post: Post-collision distribution from before streaming.
                   If None, uses f itself (approximation, less accurate).
        """
        xp = self.xp
        opp = self.opp
        
        if f_post is None:
            # Approximation: use current f
            # This works because for half-way BB, we need f_i* from the same node
            # After streaming, f[opp[i], x] came from neighbor, but we want
            # the value that was at x before streaming
            # This approximation works for steady flows
            f_source = f
        else:
            f_source = f_post
        
        # Apply bounce-back for each direction
        for i in range(1, self.Q):  # Skip rest direction
            i_opp = int(opp[i])
            
            # Where this direction needs bounce-back
            mask = self.needs_bounce[i]
            
            if xp.any(mask):
                # f_opp[i](x) = f_i*(x)  (what was going out comes back)
                # Since we're after streaming, we need to use f_source
                # which has the pre-streaming (post-collision) values
                
                # Actually, for proper HWBB after streaming:
                # The distribution f[i] at boundary fluid nodes came from
                # solid nodes (invalid). We replace it with bounced value.
                # f[opp[i], x] = f_source[i, x]
                
                f[i_opp][mask] = f_source[i][mask]
    
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
        
        f_source = f if f_post is None else f_post
        if f_post is None:
            import warnings
            warnings.warn("f_post not provided; using post-streaming f as approximation. "
                        "This is only accurate for steady-state flows.", 
                        UserWarning)
            f_source = f
        
        for i in range(1, self.Q):
            i_opp = int(opp[i])
            mask = self.needs_bounce[i]
            
            if xp.any(mask):
                # c_i · u_wall
                c_dot_u = xp.sum(c[:, i:i+1] * self.u_wall, axis=0)
                
                # Momentum correction term
                correction = 2 * w[i] * rho * c_dot_u / self.cs2
                
                # Bounce-back with correction
                f[i_opp][mask] = f_source[i][mask] - correction[mask]


# =============================================================================
# Geometry Creation Utilities
# =============================================================================

def create_sphere_mask(xp, shape: Tuple[int, ...], 
                       center: Tuple[float, ...], 
                       radius: float) -> 'npt.NDArray':
    """Create a spherical solid mask
    
    Args:
        xp: Array module
        shape: Domain shape (Nx, Ny, Nz)
        center: Sphere center (x, y, z)
        radius: Sphere radius
        
    Returns:
        Boolean mask, True inside sphere
    """
    Nx, Ny, Nz = shape
    cx, cy, cz = center
    
    x = xp.arange(Nx)
    y = xp.arange(Ny)
    z = xp.arange(Nz)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    distance = xp.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
    
    return distance <= radius


def create_cylinder_mask(xp, shape: Tuple[int, ...],
                         center: Tuple[float, float],
                         radius: float,
                         axis: str = 'z') -> 'npt.NDArray':
    """Create a cylindrical solid mask
    
    Args:
        xp: Array module
        shape: Domain shape (Nx, Ny, Nz)
        center: Cylinder center in the plane perpendicular to axis
                For axis='z': (x_center, y_center)
                For axis='x': (y_center, z_center)
                For axis='y': (x_center, z_center)
        radius: Cylinder radius
        axis: Cylinder axis direction ('x', 'y', or 'z')
        
    Returns:
        Boolean mask, True inside cylinder
    """
    Nx, Ny, Nz = shape
    
    x = xp.arange(Nx)
    y = xp.arange(Ny)
    z = xp.arange(Nz)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    if axis == 'z':
        cx, cy = center
        distance = xp.sqrt((X - cx)**2 + (Y - cy)**2)
    elif axis == 'x':
        cy, cz = center
        distance = xp.sqrt((Y - cy)**2 + (Z - cz)**2)
    elif axis == 'y':
        cx, cz = center
        distance = xp.sqrt((X - cx)**2 + (Z - cz)**2)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    
    return distance <= radius


def create_box_mask(xp, shape: Tuple[int, ...],
                    corner_min: Tuple[int, ...],
                    corner_max: Tuple[int, ...]) -> 'npt.NDArray':
    """Create a rectangular box solid mask
    
    Args:
        xp: Array module
        shape: Domain shape (Nx, Ny, Nz)
        corner_min: Minimum corner (x_min, y_min, z_min)
        corner_max: Maximum corner (x_max, y_max, z_max)
        
    Returns:
        Boolean mask, True inside box
    """
    Nx, Ny, Nz = shape
    x_min, y_min, z_min = corner_min
    x_max, y_max, z_max = corner_max
    
    x = xp.arange(Nx)
    y = xp.arange(Ny)
    z = xp.arange(Nz)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    mask = ((X >= x_min) & (X <= x_max) &
            (Y >= y_min) & (Y <= y_max) &
            (Z >= z_min) & (Z <= z_max))
    
    return mask


def create_channel_walls_mask(xp, shape: Tuple[int, ...],
                               wall_directions: str = 'yz') -> 'npt.NDArray':
    """Create wall masks for channel boundaries
    
    Args:
        xp: Array module
        shape: Domain shape (Nx, Ny, Nz)
        wall_directions: Which directions have walls
                        'y' = walls at y=0 and y=Ny-1
                        'z' = walls at z=0 and z=Nz-1
                        'yz' = both (rectangular channel)
                        
    Returns:
        Boolean mask, True at wall nodes
    """
    Nx, Ny, Nz = shape
    mask = xp.zeros(shape, dtype=bool)
    
    if 'y' in wall_directions:
        mask[:, 0, :] = True
        mask[:, -1, :] = True
    
    if 'z' in wall_directions:
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    
    if 'x' in wall_directions:
        mask[0, :, :] = True
        mask[-1, :, :] = True
    
    return mask
