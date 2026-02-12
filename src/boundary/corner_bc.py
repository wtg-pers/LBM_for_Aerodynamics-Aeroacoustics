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
    Resolution follows physical priority:
    
    | Combination           | ρ               | u                |
    |----------------------|-----------------|------------------|
    | Wall ∩ Wall          | extrapolate     | 0                |
    | Wall ∩ Velocity      | extrapolate     | 0  (wall wins)   |
    | Wall ∩ Pressure      | ρ_target        | 0  (wall wins)   |
    | Wall ∩ Freestream    | ρ_target        | 0  (wall wins)   |
    | Velocity ∩ Pressure  | ρ_target        | u_inlet          |
    | Velocity ∩ Freestream| ρ_freestream    | u_inlet          |
    | Pressure ∩ Pressure  | avg(ρ_target)   | extrapolate      |
    | Freestream ∩ any     | ρ∞              | U∞               |

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
    
    # =========================================================================
    # Main Apply
    # =========================================================================
    
    def apply(self, f: 'npt.NDArray') -> None:
        """Apply equilibrium BC at all edge and corner nodes.
        
        Called AFTER face BCs in the time loop. Since face BCs only write
        to flat nodes, this fills the remaining edge/corner nodes.
        
        Args:
            f: Distribution function, modified in-place. shape (Q, Nx, Ny[, Nz])
        """
        # Edges
        for edge in self.node_map.edge_nodes:
            self._apply_edge(f, edge)
        
        # Corners (3D only)
        for corner in self.node_map.corner_nodes:
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
    
    def _resolve_target(self, faces: List[FaceConfig],
                        rho_ext: 'npt.NDArray',
                        u_ext: 'npt.NDArray') -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Resolve target (ρ, u) when multiple faces meet.
        
        Priority rules (physical basis):
            1. WALL present → u = 0 (no-slip dominates velocity)
            2. FREESTREAM present → use ρ∞ and U∞ (far-field overrides)
            3. VELOCITY provides u; PRESSURE provides ρ
            4. Fallback: extrapolate from diagonal interior
        
        Args:
            faces: List of FaceConfig objects meeting at this node
            rho_ext: Extrapolated density from diagonal interior  [density]
            u_ext: Extrapolated velocity from diagonal interior  [Δx/Δt]
            
        Returns:
            (rho_target, u_target)
        """
        xp = self.xp
        
        types = set(fc.bc_type for fc in faces)
        
        # --- Freestream overrides everything ---
        if BCType.FREESTREAM in types:
            fs = next(fc for fc in faces if fc.bc_type == BCType.FREESTREAM)
            rho_t = xp.asarray(fs.density, dtype=xp.float64)
            
            # Build u from freestream velocity
            if isinstance(rho_ext, (int, float)) or rho_ext.ndim == 0:
                u_t = xp.zeros(self.dim, dtype=xp.float64)
            else:
                u_t = xp.zeros((self.dim,) + rho_ext.shape, dtype=xp.float64)
            
            vel = fs.velocity
            if isinstance(vel, (int, float)):
                u_t[0] = float(vel)
            elif isinstance(vel, (list, tuple)):
                for d in range(min(len(vel), self.dim)):
                    u_t[d] = float(vel[d])
            
            # But if wall also present, force u = 0
            if BCType.WALL in types:
                u_t[...] = 0.0
            
            return rho_t, u_t
        
        # --- Wall present → u = 0 always ---
        has_wall = BCType.WALL in types
        
        if has_wall:
            if isinstance(rho_ext, (int, float)) or rho_ext.ndim == 0:
                u_t = xp.zeros(self.dim, dtype=xp.float64)
            else:
                u_t = xp.zeros((self.dim,) + rho_ext.shape, dtype=xp.float64)
            
            # Density from pressure face if available, else extrapolate
            if BCType.PRESSURE in types:
                pf = next(fc for fc in faces if fc.bc_type == BCType.PRESSURE)
                rho_t = xp.asarray(pf.density, dtype=xp.float64)
                if isinstance(rho_ext, np.ndarray) or (hasattr(rho_ext, 'ndim') and rho_ext.ndim > 0):
                    rho_t = xp.full_like(rho_ext, float(pf.density))
            else:
                rho_t = rho_ext
            
            return rho_t, u_t
        
        # --- Velocity ∩ Pressure ---
        if BCType.VELOCITY in types and BCType.PRESSURE in types:
            vf = next(fc for fc in faces if fc.bc_type == BCType.VELOCITY)
            pf = next(fc for fc in faces if fc.bc_type == BCType.PRESSURE)
            
            rho_t = xp.asarray(pf.density, dtype=xp.float64)
            if isinstance(rho_ext, np.ndarray) or (hasattr(rho_ext, 'ndim') and rho_ext.ndim > 0):
                rho_t = xp.full_like(rho_ext, float(pf.density))
            
            if isinstance(rho_ext, (int, float)) or rho_ext.ndim == 0:
                u_t = xp.zeros(self.dim, dtype=xp.float64)
            else:
                u_t = xp.zeros((self.dim,) + rho_ext.shape, dtype=xp.float64)
            
            vel = vf.velocity
            if isinstance(vel, (int, float)):
                u_t[0] = float(vel)
            elif isinstance(vel, (list, tuple)):
                for d in range(min(len(vel), self.dim)):
                    u_t[d] = float(vel[d])
            
            return rho_t, u_t
        
        # --- Velocity ∩ Velocity ---
        if BCType.VELOCITY in types:
            vf = next(fc for fc in faces if fc.bc_type == BCType.VELOCITY)
            rho_t = xp.asarray(vf.density, dtype=xp.float64)
            if isinstance(rho_ext, np.ndarray) or (hasattr(rho_ext, 'ndim') and rho_ext.ndim > 0):
                rho_t = xp.full_like(rho_ext, float(vf.density))
            
            if isinstance(rho_ext, (int, float)) or rho_ext.ndim == 0:
                u_t = xp.zeros(self.dim, dtype=xp.float64)
            else:
                u_t = xp.zeros((self.dim,) + rho_ext.shape, dtype=xp.float64)
            
            vel = vf.velocity
            if isinstance(vel, (int, float)):
                u_t[0] = float(vel)
            elif isinstance(vel, (list, tuple)):
                for d in range(min(len(vel), self.dim)):
                    u_t[d] = float(vel[d])
            
            return rho_t, u_t
        
        # --- Pressure ∩ Pressure ---
        if BCType.PRESSURE in types:
            pressure_faces = [fc for fc in faces if fc.bc_type == BCType.PRESSURE]
            avg_rho = sum(fc.density for fc in pressure_faces) / len(pressure_faces)
            
            rho_t = xp.asarray(avg_rho, dtype=xp.float64)
            if isinstance(rho_ext, np.ndarray) or (hasattr(rho_ext, 'ndim') and rho_ext.ndim > 0):
                rho_t = xp.full_like(rho_ext, avg_rho)
            
            return rho_t, u_ext
        
        # --- Fallback: pure extrapolation ---
        return rho_ext, u_ext
    
    # =========================================================================
    # Info
    # =========================================================================
    
    def get_info(self) -> str:
        """Return summary string."""
        ne = self.node_map.n_edges
        nc = self.node_map.n_corners
        return f"CornerBC: {ne} edges, {nc} corners → f = f_eq (pure equilibrium)"