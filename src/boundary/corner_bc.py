"""
Edge and Corner Node Boundary Conditions

Applies pure equilibrium distribution at edge/corner nodes where
multiple boundary faces intersect:

    f_i = f_eq(ρ_target, u_target)    for ALL i = 0..Q-1
    
    Π^neq = 0.  No stress tensor needed.

This is the Palabos approach (Malaspinas et al., 2011):
    - Edge/corner nodes are a single point (2D) or line (3D edge)
    - Their impact on the solution is O(1/N) → negligible
    - The next collision step naturally generates correct f_neq
    - Far simpler and more robust than trying to average Π^neq

Target (ρ, u) Resolution Rules:
    When 2 or 3 faces meet, their prescribed values may conflict.
    "Wall" below means the no-slip kind (hwbb, bounce_back). A slip /
    symmetry face is treated as transparent to tangential flow — only
    its face-normal velocity component is zeroed, after the non-wall
    targets are resolved.

    Priority (highest → lowest):
        1. No-slip WALL → u = 0; ρ from PRESSURE > SPONGE > extrapolate
        2. SPONGE       → (ρ∞, U∞)  [far-field reference wins over Dirichlet,
                          consistent with Phase 3 sponge volume damping]
        3. VELOCITY ∩ PRESSURE → ρ from PRESSURE, u from VELOCITY
        4. VELOCITY     → (ρ_inlet, u_inlet)
        5. PRESSURE     → (avg ρ_p, u_extrap)
        6. Fallback     → (ρ_extrap, u_extrap)   [Neumann or pure extrapolation]
        + Slip WALL applied last: zero face-normal component of resolved u.

    Examples (after priority resolution):

    | Combination              | ρ               | u                  |
    |-------------------------|-----------------|--------------------|
    | Wall ∩ Wall              | extrapolate     | 0                  |
    | Wall ∩ Velocity          | extrapolate     | 0  (wall wins)     |
    | Wall ∩ Pressure          | ρ_p_target      | 0  (wall wins)     |
    | Wall ∩ Sponge            | ρ∞              | 0  (wall wins)     |
    | Sponge ∩ Velocity        | ρ∞              | U∞ (sponge wins)   |
    | Sponge ∩ Pressure        | ρ∞              | U∞ (sponge wins)   |
    | Sponge ∩ Neumann         | ρ∞              | U∞                 |
    | Velocity ∩ Pressure      | ρ_p_target      | u_inlet            |
    | Velocity ∩ Velocity      | ρ_inlet         | u_inlet            |
    | Pressure ∩ Pressure      | avg(ρ_p)        | extrapolate        |
    | Slip ∩ Velocity          | ρ_inlet         | u_inlet, u[s]=0    |
    | Slip ∩ Sponge            | ρ∞              | U∞,     u[s]=0     |
    | Slip ∩ Neumann           | extrapolate     | u_ext,  u[s]=0     |
    | Slip ∩ Slip (3D)         | extrapolate     | u_ext,  normals 0  |
    | Slip ∩ Slip (2D corner)  | extrapolate     | (0, 0) effective   |
    | Neumann + Neumann/etc.   | extrapolate     | extrapolate        |

References:
    - Malaspinas, Chopard, Latt, Comp. Fluids 49, 2011

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, List, Tuple, Optional
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .bc_config import FaceConfig, BCType
from .node_map import NodeMap, EdgeNode, CornerNode
from .regularized_utils import compute_f_eq


class CornerBC:
    """Applies f = f_eq at all edge and corner nodes.
    
    This class is initialized with the NodeMap and then called each
    time step to overwrite edge/corner distributions with equilibrium.
    
    The physical justification:
        1. Edge/corner nodes have neighbors in 2+ "outside" directions
           → extracting Π^neq from interior is ambiguous and inaccurate
        2. A single node's f_neq error is O(1/N) on the global solution
        3. BGK collision at the next step generates correct f_neq naturally
        4. Equilibrium at edges prevents any BC conflict by construction
    
    Args:
        xp: Array module (numpy or cupy)
        lattice: Lattice model (D2Q9 or D3Q27)
        node_map: NodeMap with classified edge/corner nodes
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 node_map: NodeMap) -> None:
        self.xp = xp
        self.lattice = lattice
        self.node_map = node_map
        
        self.dim = lattice.dim
        self.Q = lattice.Q
        self.c = xp.asarray(lattice.c)
        self.w = xp.asarray(lattice.w)
        self.cs2 = lattice.cs2
        
        self.domain_shape = node_map.domain_shape

        # ── Pre-cache edge/corner f_eq where possible ────────────
        # Edges/corners with constant target (no extrapolation needed)
        # get their f_eq computed once at init, not every step.
        self._cached_edges = []    # (index_tuple, f_eq_array)
        self._uncached_edges = []  # edges that need runtime extrapolation
        self._cached_corners = []
        self._uncached_corners = []

        self._precache_edges_and_corners()

    def _precache_edges_and_corners(self) -> None:
        """Pre-compute f_eq for edges/corners with constant targets."""
        from src.boundary.regularized_utils import compute_f_eq
        xp = self.xp

        for edge in self.node_map.edge_nodes:
            faces = [edge.face_a, edge.face_b]
            if hasattr(edge, 'face_c'):
                faces.append(edge.face_c)
            types = set(fc.bc_type for fc in faces)

            # Check if target is constant (no extrapolation needed).
            # Cases that DO need runtime extrapolation:
            #   - NEUMANN present                       (Branch 6, both extrap)
            #   - PRESSURE alone (no V/W/S)             (Branch 5, u extrap)
            #   - No-slip WALL without PRESSURE/SPONGE  (Branch 1, ρ extrap;
            #     WALL+VELOCITY also lands here — wall pins u=0, but ρ has no
            #     constant reference)
            # SPONGE always pins (ρ∞, U∞) → constant. VELOCITY+PRESSURE constant.
            needs_extrap = (
                BCType.NEUMANN in types
                or (BCType.PRESSURE in types
                    and BCType.VELOCITY not in types
                    and BCType.SPONGE not in types
                    and BCType.WALL not in types)
                or (BCType.WALL in types
                    and BCType.PRESSURE not in types
                    and BCType.SPONGE not in types)
            )

            if not needs_extrap and self.dim == 3:
                # Constant target: compute f_eq once
                # Use dummy rho_ext to resolve target
                free_ax = edge.free_axis
                n_free = self.domain_shape[free_ax]
                rho_dummy = xp.ones(n_free, dtype=xp.float64)
                u_dummy = xp.zeros((self.dim, n_free), dtype=xp.float64)
                rho_t, u_t = self._resolve_target(faces, rho_dummy, u_dummy)
                f_eq = compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)

                # Build index
                fixed_axes = sorted(edge.position.keys())
                edge_idx = [None, None, None]
                for ax in range(3):
                    if ax == free_ax:
                        edge_idx[ax] = slice(None)
                    else:
                        edge_idx[ax] = edge.position[ax]

                self._cached_edges.append((edge_idx, f_eq))
            else:
                self._uncached_edges.append(edge)

        for corner in self.node_map.corner_nodes:
            faces = [corner.face_a, corner.face_b, corner.face_c]
            types = set(fc.bc_type for fc in faces)

            # Same extrapolation rules as edges (see comment above).
            needs_extrap = (
                BCType.NEUMANN in types
                or (BCType.PRESSURE in types
                    and BCType.VELOCITY not in types
                    and BCType.SPONGE not in types
                    and BCType.WALL not in types)
                or (BCType.WALL in types
                    and BCType.PRESSURE not in types
                    and BCType.SPONGE not in types)
            )

            if not needs_extrap:
                rho_dummy = xp.ones(1, dtype=xp.float64)
                u_dummy = xp.zeros((self.dim, 1), dtype=xp.float64)
                rho_t, u_t = self._resolve_target(faces, rho_dummy, u_dummy)
                f_eq = compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)

                pos = corner.position
                self._cached_corners.append(
                    ((pos[0], pos[1], pos[2]), f_eq.ravel()))
            else:
                self._uncached_corners.append(corner)

    # =========================================================================
    # Main Apply
    # =========================================================================

    def apply(self, f: 'npt.NDArray') -> None:
        """Apply equilibrium BC at all edge and corner nodes.

        Pre-cached edges/corners: direct write (fast).
        Uncached: runtime computation (original path).
        """
        # Cached edges: direct write, no computation
        for edge_idx, f_eq in self._cached_edges:
            f[:, edge_idx[0], edge_idx[1], edge_idx[2]] = f_eq

        # Cached corners: direct write
        for (ix, iy, iz), f_eq in self._cached_corners:
            f[:, ix, iy, iz] = f_eq

        # Uncached edges: original path (with extrapolation)
        for edge in self._uncached_edges:
            self._apply_edge(f, edge)

        # Uncached corners: original path
        for corner in self._uncached_corners:
            self._apply_corner(f, corner)
    
    # =========================================================================
    # Edge Nodes
    # =========================================================================
    
    def _apply_edge(self, f: 'npt.NDArray', edge: EdgeNode) -> None:
        """Apply f = f_eq at an edge node (or line of nodes in 3D).
        
        Args:
            f: Distribution function, modified in-place
            edge: EdgeNode with position and face configs
        """
        xp = self.xp
        pos = edge.position
        
        if self.dim == 2:
            # 2D edge = single point
            ix, iy = pos[0], pos[1]
            
            # Get diagonal interior neighbor for extrapolation
            offset_x = edge.face_a.location.inward_offset if edge.face_a.location.axis == 0 else edge.face_b.location.inward_offset
            offset_y = edge.face_a.location.inward_offset if edge.face_a.location.axis == 1 else edge.face_b.location.inward_offset
            
            f_diag = f[:, ix + offset_x, iy + offset_y]
            rho_ext = xp.sum(f_diag, axis=0)
            c_float = self.c.astype(xp.float64)
            u_ext = xp.einsum('di,i->d', c_float, f_diag) / (rho_ext + 1e-30)
            
            # Resolve target (ρ, u)
            rho_t, u_t = self._resolve_target(
                [edge.face_a, edge.face_b], rho_ext, u_ext
            )
            
            # f = f_eq (scalar node → shapes are (Q,))
            f_eq = compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)
            f[:, ix, iy] = f_eq
        
        else:
            # 3D edge = line of nodes along free_axis
            free_ax = edge.free_axis
            fixed_axes = sorted(pos.keys())  # 2 fixed axes
            
            # Get inward offsets for the 2 fixed axes
            offsets = {}
            for face in (edge.face_a, edge.face_b):
                offsets[face.location.axis] = face.location.inward_offset
            
            # Build index for edge line and diagonal interior line
            edge_idx = [None, None, None]
            diag_idx = [None, None, None]
            
            for ax in range(3):
                if ax == free_ax:
                    # Free axis: iterate over all nodes
                    # Trim endpoints if they are corners (handled separately)
                    edge_idx[ax] = slice(None)
                    diag_idx[ax] = slice(None)
                elif ax in offsets:
                    edge_idx[ax] = pos[ax]
                    diag_idx[ax] = pos[ax] + offsets[ax]
                else:
                    edge_idx[ax] = pos[ax]
                    diag_idx[ax] = pos[ax]
            
            f_diag = f[:, diag_idx[0], diag_idx[1], diag_idx[2]]
            rho_ext = xp.sum(f_diag, axis=0)
            c_float = self.c.astype(xp.float64)
            u_ext = xp.einsum('di,i...->d...', c_float, f_diag) / (rho_ext + 1e-30)
            
            rho_t, u_t = self._resolve_target(
                [edge.face_a, edge.face_b], rho_ext, u_ext
            )
            
            f_eq = compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)
            f[:, edge_idx[0], edge_idx[1], edge_idx[2]] = f_eq
    
    # =========================================================================
    # Corner Nodes (3D only)
    # =========================================================================
    
    def _apply_corner(self, f: 'npt.NDArray', corner: CornerNode) -> None:
        """Apply f = f_eq at a single corner node (3D only).
        
        Args:
            f: Distribution function, modified in-place
            corner: CornerNode with position and 3 face configs
        """
        xp = self.xp
        pos = corner.position
        ix, iy, iz = pos[0], pos[1], pos[2]
        
        # Diagonal interior neighbor (offset in all 3 directions)
        ox = corner.face_a.location.inward_offset  # face_a is x-face
        oy = corner.face_b.location.inward_offset  # face_b is y-face
        oz = corner.face_c.location.inward_offset  # face_c is z-face
        
        f_diag = f[:, ix + ox, iy + oy, iz + oz]
        rho_ext = xp.sum(f_diag, axis=0)
        c_float = self.c.astype(xp.float64)
        u_ext = xp.einsum('di,i->d', c_float, f_diag) / (rho_ext + 1e-30)
        
        rho_t, u_t = self._resolve_target(
            [corner.face_a, corner.face_b, corner.face_c], rho_ext, u_ext
        )
        
        f_eq = compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)
        f[:, ix, iy, iz] = f_eq
    
    # =========================================================================
    # Target Resolution
    # =========================================================================
    
    # Methods that represent a free-slip / symmetry wall (WALL-typed).
    # These zero only the face-normal velocity component at corners,
    # leaving tangential components to be set by other faces.
    _SLIP_METHODS = frozenset({'slip', 'symmetry', 'free_slip'})

    @classmethod
    def _is_slip(cls, fc) -> bool:
        return (fc.bc_type == BCType.WALL
                and fc.method.lower() in cls._SLIP_METHODS)

    @classmethod
    def _is_noslip_wall(cls, fc) -> bool:
        return (fc.bc_type == BCType.WALL
                and fc.method.lower() not in cls._SLIP_METHODS)

    def _apply_slip_walls(self, u_t, slip_walls):
        """Zero the face-normal velocity component for each slip wall face.

        Returns u_t unchanged if no slip walls are present. Otherwise copies
        first so callers' extrapolation arrays are not mutated in place.

        Works for both scalar-node (u_t shape (dim,)) and line/edge
        (u_t shape (dim, N)) u_t arrays.
        """
        if not slip_walls:
            return u_t
        u_t = u_t.copy() if hasattr(u_t, 'copy') else self.xp.array(u_t)
        for fc in slip_walls:
            u_t[fc.location.axis] = 0
        return u_t

    def _resolve_target(self, faces, rho_ext, u_ext):
        """Resolve target (ρ, u) when multiple faces meet.

        Priority rules (physical basis):
            1. No-slip WALL → u = 0; ρ from PRESSURE > SPONGE > extrapolate
               (no-slip is a hard physical constraint; pressure target wins
               for ρ if available, else sponge ρ∞, else interior extrapolation)
            2. SPONGE → (ρ∞, U∞) from its freestream target
               (placed above VELOCITY/PRESSURE so corners shared with a sponge
               buffer track far-field truth, consistent with Phase 3 volume
               damping — avoids Phase 2/Phase 3 disagreement)
            3. VELOCITY ∩ PRESSURE → ρ from PRESSURE, u from VELOCITY
            4. VELOCITY → (ρ_inlet, u_inlet)
            5. PRESSURE → (avg ρ_p, u_extrap)
            6. Fallback (Neumann/etc.) → (ρ_extrap, u_extrap)
            + Slip WALL applied last: zero only the face-normal component of
              the resolved u, leaving tangential flow untouched.

        Args:
            faces: List of FaceConfig objects meeting at this node
            rho_ext: Extrapolated density from diagonal interior  [dimensionless]
            u_ext: Extrapolated velocity from diagonal interior  [Δx/Δt]

        Returns:
            (rho_target, u_target)
        """
        xp = self.xp

        types = set(fc.bc_type for fc in faces)

        # Classify wall faces: no-slip (hwbb/bounce_back) vs slip (specular).
        # No-slip walls force u=0 entirely; slip walls only zero the
        # wall-normal component and are applied last.
        slip_walls = [fc for fc in faces if self._is_slip(fc)]
        has_noslip_wall = any(self._is_noslip_wall(fc) for fc in faces)

        is_array = (
            isinstance(rho_ext, np.ndarray)
            or (hasattr(rho_ext, 'ndim') and rho_ext.ndim > 0)
        )

        def _broadcast_rho(value: float):
            if is_array:
                return xp.full_like(rho_ext, float(value))
            return xp.asarray(value, dtype=xp.float64)

        def _zero_u():
            if is_array:
                return xp.zeros((self.dim,) + rho_ext.shape, dtype=xp.float64)
            return xp.zeros(self.dim, dtype=xp.float64)

        def _build_u_from_velocity(vel):
            u_t = _zero_u()
            if isinstance(vel, (int, float)):
                u_t[0] = float(vel)
            elif isinstance(vel, (list, tuple)):
                for d in range(min(len(vel), self.dim)):
                    u_t[d] = float(vel[d])
            return u_t

        # --- 1) No-slip wall present → u = 0 always ---
        # ρ priority: PRESSURE target > SPONGE ρ∞ > interior extrapolation.
        # No-slip pins u; ρ falls back to the most authoritative far-field
        # reference adjacent to this corner.
        if has_noslip_wall:
            u_t = _zero_u()

            if BCType.PRESSURE in types:
                pf = next(fc for fc in faces if fc.bc_type == BCType.PRESSURE)
                rho_t = _broadcast_rho(pf.density)
            elif BCType.SPONGE in types:
                sf = next(fc for fc in faces if fc.bc_type == BCType.SPONGE)
                rho_t = _broadcast_rho(sf.density)
            else:
                rho_t = rho_ext

            return rho_t, u_t

        # --- 2) SPONGE present → freestream target (ρ∞, U∞) ---
        # Placed above VELOCITY/PRESSURE so corners shared with a sponge buffer
        # track the far-field reference. This keeps Phase 2 (corner BC) and
        # Phase 3 (sponge volume damping) writing the same target — eliminating
        # the σ-dependent fight that would otherwise occur at those nodes.
        if BCType.SPONGE in types:
            sf = next(fc for fc in faces if fc.bc_type == BCType.SPONGE)
            rho_t = _broadcast_rho(sf.density)
            u_t = _build_u_from_velocity(sf.velocity)
            return rho_t, self._apply_slip_walls(u_t, slip_walls)

        # --- 3) VELOCITY ∩ PRESSURE ---
        if BCType.VELOCITY in types and BCType.PRESSURE in types:
            vf = next(fc for fc in faces if fc.bc_type == BCType.VELOCITY)
            pf = next(fc for fc in faces if fc.bc_type == BCType.PRESSURE)
            rho_t = _broadcast_rho(pf.density)
            u_t = _build_u_from_velocity(vf.velocity)
            return rho_t, self._apply_slip_walls(u_t, slip_walls)

        # --- 4) VELOCITY only ---
        if BCType.VELOCITY in types:
            vf = next(fc for fc in faces if fc.bc_type == BCType.VELOCITY)
            rho_t = _broadcast_rho(vf.density)
            u_t = _build_u_from_velocity(vf.velocity)
            return rho_t, self._apply_slip_walls(u_t, slip_walls)

        # --- 5) PRESSURE only (avg density from all pressure faces) ---
        if BCType.PRESSURE in types:
            pressure_faces = [fc for fc in faces if fc.bc_type == BCType.PRESSURE]
            avg_rho = sum(fc.density for fc in pressure_faces) / len(pressure_faces)
            rho_t = _broadcast_rho(avg_rho)
            return rho_t, self._apply_slip_walls(u_ext, slip_walls)

        # --- 6) Neumann / Fallback: pure extrapolation ---
        # NEUMANN faces prescribe nothing (∂f/∂n = 0 is handled by face BC).
        # At edge/corner nodes, use the extrapolated values from the diagonal
        # interior neighbor. Also serves as the general fallback.
        return rho_ext, self._apply_slip_walls(u_ext, slip_walls)
    
    # =========================================================================
    # Info
    # =========================================================================
    
    def get_info(self) -> str:
        """Return summary string."""
        ne = self.node_map.n_edges
        nc = self.node_map.n_corners
        return f"CornerBC: {ne} edges, {nc} corners → f = f_eq (pure equilibrium)"