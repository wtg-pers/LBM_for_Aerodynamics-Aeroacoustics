"""
Edge/Corner Regularized Boundary Conditions for LBM

This module implements the Generalized Regularized Boundary Condition (GRBC)
for edge and corner nodes — where two or more boundary faces intersect.

Problem Solved:
    At edge/corner nodes, face BCs extract Π^neq from a single interior
    neighbor that may itself be close to another wall, yielding inaccurate
    stress estimates. This causes under-resolution artifacts at corners.

Palabos Approach (Malaspinas et al., Comp. Fluids 49, 2011):
    1. Classify each domain node as FLAT / EDGE / CORNER by counting how
       many boundary faces it belongs to.
    2. For EDGE nodes (2 faces meet):
       - Extract Π^neq from multiple interior neighbors (along each
         face's inward normal) and AVERAGE them.
       - Determine (ρ, u) based on the BC types that meet there.
    3. For CORNER nodes (3 faces meet, 3D only):
       - Average Π^neq from 3 interior neighbors (or use diagonal).
    4. Reconstruct f_i = f_i^eq(ρ, u) + f_i^(1)(Π^neq_averaged).

Direction-Agnostic Design:
    This module makes NO assumption about which axis is streamwise.
    Any face (xmin, xmax, ymin, ymax, zmin, zmax) can be inlet, outlet,
    wall, or freestream. All logic is derived from the face location and
    its assigned BC type.

Integration:
    Applied AFTER face BCs, overwriting edge/corner nodes with properly
    averaged reconstructions. Application order in main loop:
        1. domain_walls.apply_all(f_new, f_post)
        2. bc_manager.apply_all(f_new)          ← face BCs
        3. edge_corner_mgr.apply(f_new)         ← THIS MODULE (overwrites corners)
        4. obstacle_bc.apply_with_reset(...)

References:
    - Malaspinas, Chopard, Latt, Comp. Fluids 49, 2011
    - Latt & Chopard, Math. Comp. Sim. 72, 2006
    - Latt et al., Palabos (open-source LBM framework)

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Dict, Any, List, Tuple, Optional
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .regularized import compute_Pi_neq, reconstruct_f_regularized


# =============================================================================
# BC Type Classification
# =============================================================================

class BCType(Enum):
    """Boundary condition type for face classification
    
    Used to determine (ρ, u) at edge/corner nodes.
    """
    VELOCITY = auto()     # Inlet: specifies u (ρ extrapolated from interior)
    PRESSURE = auto()     # Outlet: specifies ρ (u extrapolated from interior)
    WALL = auto()         # No-slip wall: u = u_wall (typically 0)
    FREESTREAM = auto()   # Far-field: specifies both ρ and u
    PERIODIC = auto()     # No BC needed at this face
    UNKNOWN = auto()


# Method strings → BCType mapping
_METHOD_TO_TYPE: Dict[str, BCType] = {
    # Velocity (inlet) methods
    'equilibrium': BCType.VELOCITY,
    'eq': BCType.VELOCITY,
    'non_equilibrium': BCType.VELOCITY,
    'neq': BCType.VELOCITY,
    'non_eq': BCType.VELOCITY,
    'regularized_inlet': BCType.VELOCITY,
    'reg_inlet': BCType.VELOCITY,
    # Pressure (outlet) methods
    'pressure_relaxation': BCType.PRESSURE,
    'regularized_outlet': BCType.PRESSURE,
    'reg_outlet': BCType.PRESSURE,
    'convective': BCType.PRESSURE,
    'neumann': BCType.PRESSURE,
    'extrapolation': BCType.PRESSURE,
    # Wall methods
    'bounce_back': BCType.WALL,
    'hwbb': BCType.WALL,
    'wall': BCType.WALL,
    'halfway': BCType.WALL,
    'regularized_wall': BCType.WALL,
    'reg_wall': BCType.WALL,
    # Far-field
    'freestream': BCType.FREESTREAM,
    'farfield': BCType.FREESTREAM,
    'sponge': BCType.FREESTREAM,
    # Periodic
    'periodic': BCType.PERIODIC,
    'none': BCType.PERIODIC,
    '': BCType.PERIODIC,
}


def classify_method(method: str) -> BCType:
    """Classify BC method string into BCType
    
    Args:
        method: BC method name (case-insensitive)
        
    Returns:
        BCType enum value
    """
    return _METHOD_TO_TYPE.get(method.lower().strip(), BCType.UNKNOWN)


# =============================================================================
# Direction-Agnostic Face Utilities
# =============================================================================

@dataclass
class FaceInfo:
    """Information about a boundary face
    
    Direction-agnostic: derived entirely from location string.
    
    Attributes:
        location: Face name ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        axis: Coordinate axis (0=x, 1=y, 2=z)  [dimensionless]
        is_min: True if min face (index=0), False if max face
        index: Grid index at this face  [lattice units]
        inward_offset: +1 for min faces, -1 for max faces  [lattice units]
        bc_type: BCType classification
        velocity: Inlet velocity (only for VELOCITY type)  [Δx/Δt]
        density: Target density (only for PRESSURE type)  [dimensionless]
        relax_coeff: Pressure relaxation K (only for PRESSURE type)  [dimensionless]
    """
    location: str
    axis: int
    is_min: bool
    index: int
    inward_offset: int
    bc_type: BCType
    velocity: float = 0.0       # [Δx/Δt]
    density: float = 1.0        # [dimensionless]
    relax_coeff: float = 0.1    # [dimensionless]


def _parse_face_location(location: str, shape: Tuple[int, ...]) -> Tuple[int, bool, int, int]:
    """Parse face location string into axis, is_min, grid_index, inward_offset
    
    Direction-agnostic: handles any face regardless of flow direction.
    
    Args:
        location: Face name (e.g., 'xmin', 'ymax')
        shape: Domain shape (Nx, Ny[, Nz])  [lattice units]
        
    Returns:
        (axis, is_min, grid_index, inward_offset)
    """
    # Normalize aliases
    _alias = {
        'west': 'xmin', 'east': 'xmax',
        'south': 'ymin', 'north': 'ymax',
        'bottom': 'zmin', 'top': 'zmax',
    }
    loc = _alias.get(location.lower(), location.lower())
    
    _axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis = _axis_map[loc[0]]
    is_min = loc.endswith('min')
    
    grid_index = 0 if is_min else shape[axis] - 1
    inward_offset = 1 if is_min else -1
    
    return axis, is_min, grid_index, inward_offset


def build_face_info(location: str, bc_config: Dict[str, Any],
                    shape: Tuple[int, ...]) -> FaceInfo:
    """Build FaceInfo from config dictionary
    
    Args:
        location: Face location string
        bc_config: Boundary config dict with 'method', 'velocity', 'rho', etc.
        shape: Domain shape  [lattice units]
        
    Returns:
        FaceInfo with all fields populated
    """
    axis, is_min, grid_index, inward_offset = _parse_face_location(location, shape)
    
    method = bc_config.get('method', '').lower()
    bc_type = classify_method(method)
    
    velocity = bc_config.get('velocity', bc_config.get('u_inf', 0.0))
    density = bc_config.get('rho', bc_config.get('density', 1.0))
    relax_coeff = bc_config.get('k', bc_config.get('relax_coeff', 0.1))
    
    return FaceInfo(
        location=location.lower(),
        axis=axis,
        is_min=is_min,
        index=grid_index,
        inward_offset=inward_offset,
        bc_type=bc_type,
        velocity=float(velocity),
        density=float(density),
        relax_coeff=float(relax_coeff),
    )


# =============================================================================
# Edge/Corner Node Identification
# =============================================================================

@dataclass
class IntersectionNode:
    """An edge or corner node where multiple BC faces meet
    
    Attributes:
        faces: List of FaceInfo objects meeting at this node
        node_type: 'edge' (2 faces) or 'corner' (3 faces)
        position: Grid coordinate of this node, per axis  [lattice units]
        interior_neighbors: List of (axis, offset) pairs for Π^neq extraction
    """
    faces: List[FaceInfo]
    node_type: str           # 'edge' or 'corner'
    position: Dict[int, int]  # axis → grid index  [lattice units]
    
    @property
    def n_faces(self) -> int:
        return len(self.faces)


def find_intersections(face_infos: List[FaceInfo],
                       dim: int) -> List[IntersectionNode]:
    """Find all edge/corner nodes from the boundary face list
    
    Direction-agnostic: works for any combination of face assignments.
    
    2D: up to 4 edge nodes (where any 2 faces on different axes meet)
    3D: up to 12 edge nodes + 8 corner nodes
    
    Only creates intersection nodes where BOTH faces have non-periodic BCs.
    
    Args:
        face_infos: List of FaceInfo for all configured faces
        dim: Spatial dimension (2 or 3)
        
    Returns:
        List of IntersectionNode objects
    """
    # Group faces by axis
    faces_by_axis: Dict[int, List[FaceInfo]] = {}
    for fi in face_infos:
        if fi.bc_type == BCType.PERIODIC:
            continue  # Skip periodic faces — no BC needed
        faces_by_axis.setdefault(fi.axis, []).append(fi)
    
    intersections: List[IntersectionNode] = []
    axes = sorted(faces_by_axis.keys())
    
    # ── Edges: pairs of faces on DIFFERENT axes ──
    for i_idx in range(len(axes)):
        for j_idx in range(i_idx + 1, len(axes)):
            axis_a, axis_b = axes[i_idx], axes[j_idx]
            for fa in faces_by_axis[axis_a]:
                for fb in faces_by_axis[axis_b]:
                    position = {fa.axis: fa.index, fb.axis: fb.index}
                    intersections.append(IntersectionNode(
                        faces=[fa, fb],
                        node_type='edge',
                        position=position,
                    ))
    
    # ── Corners (3D only): triplets of faces on 3 different axes ──
    if dim == 3 and len(axes) >= 3:
        for i_idx in range(len(axes)):
            for j_idx in range(i_idx + 1, len(axes)):
                for k_idx in range(j_idx + 1, len(axes)):
                    axis_a = axes[i_idx]
                    axis_b = axes[j_idx]
                    axis_c = axes[k_idx]
                    for fa in faces_by_axis[axis_a]:
                        for fb in faces_by_axis[axis_b]:
                            for fc in faces_by_axis[axis_c]:
                                position = {
                                    fa.axis: fa.index,
                                    fb.axis: fb.index,
                                    fc.axis: fc.index,
                                }
                                intersections.append(IntersectionNode(
                                    faces=[fa, fb, fc],
                                    node_type='corner',
                                    position=position,
                                ))
    
    return intersections


# =============================================================================
# Target (ρ, u) Resolution at Intersections
# =============================================================================

def resolve_target_rho_u(
    node: IntersectionNode,
    rho_interior: float,
    u_interior: 'npt.NDArray',
    dim: int
) -> Tuple[float, 'npt.NDArray']:
    """Determine target (ρ, u) at an edge/corner node
    
    Priority rules (following Palabos convention):
        1. WALL face present → u = u_wall (= 0 for no-slip)  [Δx/Δt]
        2. VELOCITY face present → sets velocity along that face's normal  [Δx/Δt]
        3. PRESSURE face present → sets ρ (with optional relaxation)  [dimensionless]
        4. Anything unspecified → extrapolated from interior
    
    Combined rules:
        - WALL ∩ WALL         : u = 0, ρ = ρ_interior
        - WALL ∩ VELOCITY     : u = 0 (wall dominates), ρ = ρ_interior
        - WALL ∩ PRESSURE     : u = 0 (wall), ρ = ρ_target (pressure)
        - VELOCITY ∩ PRESSURE : u = u_inlet, ρ = ρ_target
        - VELOCITY ∩ VELOCITY : average velocities
        - PRESSURE ∩ PRESSURE : average densities, u = u_interior
        - FREESTREAM ∩ *      : ρ = ρ_fs, u = u_fs
    
    This is direction-agnostic: velocity components are set based on
    each face's normal axis, not hardcoded x/y/z assignments.
    
    Args:
        node: IntersectionNode with face information
        rho_interior: Density from interior neighbor  [dimensionless]
        u_interior: Velocity from interior neighbor, shape (dim,)  [Δx/Δt]
        dim: Spatial dimension
        
    Returns:
        (rho_target, u_target) where u_target has shape (dim,)
    """
    import numpy as np_local
    
    types = [f.bc_type for f in node.faces]
    
    has_wall = BCType.WALL in types
    has_velocity = BCType.VELOCITY in types
    has_pressure = BCType.PRESSURE in types
    has_freestream = BCType.FREESTREAM in types
    
    # Start with interior extrapolation as default
    rho_target = float(rho_interior)
    u_target = np_local.array(u_interior, dtype=np_local.float64).copy()
    
    # ── FREESTREAM dominates everything ──
    if has_freestream:
        for face in node.faces:
            if face.bc_type == BCType.FREESTREAM:
                rho_target = face.density
                # Set velocity along this face's normal
                sign = 1 if face.is_min else -1
                u_target[face.axis] = sign * face.velocity
        return rho_target, u_target
    
    # ── WALL present → u = 0 (no-slip) ──
    if has_wall:
        u_target[:] = 0.0  # [Δx/Δt]  No-slip
        
        # If pressure BC also present, use its density
        if has_pressure:
            pressure_faces = [f for f in node.faces if f.bc_type == BCType.PRESSURE]
            rho_target = pressure_faces[0].density  # [dimensionless]
            # Apply relaxation: ρ = ρ_target + (1-K)(ρ_int - ρ_target)
            K = pressure_faces[0].relax_coeff
            rho_target = rho_target + (1.0 - K) * (rho_interior - rho_target)
        # else: ρ = ρ_interior (already set)
        
        return rho_target, u_target
    
    # ── VELOCITY ∩ PRESSURE ──
    if has_velocity and has_pressure:
        # Velocity face → set u along its normal
        for face in node.faces:
            if face.bc_type == BCType.VELOCITY:
                sign = 1 if face.is_min else -1
                u_target[face.axis] = sign * face.velocity  # [Δx/Δt]
        # Pressure face → set ρ
        for face in node.faces:
            if face.bc_type == BCType.PRESSURE:
                rho_target = face.density  # [dimensionless]
                K = face.relax_coeff
                rho_target = rho_target + (1.0 - K) * (rho_interior - rho_target)
        return rho_target, u_target
    
    # ── VELOCITY ∩ VELOCITY ──
    if has_velocity and not has_pressure:
        for face in node.faces:
            if face.bc_type == BCType.VELOCITY:
                sign = 1 if face.is_min else -1
                u_target[face.axis] = sign * face.velocity  # [Δx/Δt]
        return rho_target, u_target
    
    # ── PRESSURE ∩ PRESSURE ──
    if has_pressure and not has_velocity:
        pressure_faces = [f for f in node.faces if f.bc_type == BCType.PRESSURE]
        rho_avg = sum(f.density for f in pressure_faces) / len(pressure_faces)
        K_avg = sum(f.relax_coeff for f in pressure_faces) / len(pressure_faces)
        rho_target = rho_avg + (1.0 - K_avg) * (rho_interior - rho_avg)
        # u stays as interior extrapolation
        return rho_target, u_target
    
    # ── Fallback: full extrapolation from interior ──
    return rho_target, u_target


# =============================================================================
# Main Manager Class
# =============================================================================

class RegularizedEdgeCornerManager:
    """Manages edge/corner nodes with averaged Π^neq reconstruction
    
    Follows Palabos's Generalized Regularized BC (Malaspinas et al., 2011).
    
    At each edge/corner node:
        1. Identify interior neighbors along each participating face's normal
        2. Extract Π^neq from each interior neighbor
        3. Average the Π^neq tensors
        4. Determine target (ρ, u) based on BC type priority
        5. Reconstruct all Q populations:
           f_i = f_i^eq(ρ_target, u_target) + f_i^(1)(Π^neq_avg)
    
    Direction-agnostic: any face can be inlet/outlet/wall/freestream.
    
    Usage:
        >>> manager = RegularizedEdgeCornerManager(xp, lattice, boundaries_config, shape)
        >>> # In time loop, AFTER face BCs:
        >>> manager.apply(f)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 boundaries_config: Dict[str, Dict[str, Any]],
                 shape: Tuple[int, ...],
                 verbose: bool = True) -> None:
        """Initialize edge/corner manager from boundary configuration
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            boundaries_config: Full boundary config dictionary
                e.g. {"inlet": {"location": "xmin", "method": "regularized_inlet", ...},
                       "outlet": {"location": "xmax", ...},
                       "wall_s": {"location": "ymin", "method": "regularized_wall"}, ...}
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
            verbose: Print detected edge/corner info
        """
        self.xp = xp
        self.lattice = lattice
        self.shape = shape
        self.dim = lattice.dim
        self.c = xp.asarray(lattice.c, dtype=xp.float64)
        self.w = xp.asarray(lattice.w, dtype=xp.float64)
        self.cs2 = lattice.cs2
        self.Q = lattice.Q
        
        # ── Step 1: Parse all face configurations ──
        self.face_infos = self._parse_boundaries(boundaries_config)
        
        # ── Step 2: Find edge/corner intersections ──
        self.intersections = find_intersections(self.face_infos, self.dim)
        
        if verbose:
            self._print_info()
    
    def _parse_boundaries(self, config: Dict[str, Dict[str, Any]]) -> List[FaceInfo]:
        """Parse boundary config into FaceInfo list
        
        Args:
            config: Boundary configuration dictionary
            
        Returns:
            List of FaceInfo for each configured face
        """
        face_infos = []
        for bc_name, bc_config in config.items():
            location = bc_config.get('location', bc_name)
            fi = build_face_info(location, bc_config, self.shape)
            face_infos.append(fi)
        return face_infos
    
    def _print_info(self) -> None:
        """Print detected edge/corner nodes"""
        n_edges = sum(1 for n in self.intersections if n.node_type == 'edge')
        n_corners = sum(1 for n in self.intersections if n.node_type == 'corner')
        
        print(f"  Edge/Corner Manager ({self.dim}D):")
        print(f"    Faces: {len(self.face_infos)} configured")
        print(f"    Edges: {n_edges}, Corners: {n_corners}")
        
        for node in self.intersections:
            face_strs = [f"{f.location}({f.bc_type.name})" for f in node.faces]
            pos_str = ", ".join(f"{'xyz'[a]}={v}" for a, v in sorted(node.position.items()))
            print(f"      {node.node_type}: {' ∩ '.join(face_strs)} → ({pos_str})")
    
    @property
    def n_intersections(self) -> int:
        return len(self.intersections)
    
    def apply(self, f: 'npt.NDArray') -> None:
        """Apply regularized reconstruction at all edge/corner nodes
        
        Must be called AFTER face BCs to overwrite corners with proper
        averaged Π^neq reconstruction.
        
        Physical process at each edge/corner node:
            1. For each participating face, look at the interior neighbor
               (1 node inward along that face's normal)
            2. Compute Π^neq at each interior neighbor
            3. Average the Π^neq tensors
            4. Determine target (ρ, u) based on BC type priorities
            5. Reconstruct: f_i = f_i^eq(ρ, u) + f_i^(1)(Π^neq_avg)
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny[, Nz])
               Modified in-place at edge/corner nodes.
        """
        if self.dim == 2:
            self._apply_2d(f)
        else:
            self._apply_3d(f)
    
    # =========================================================================
    # 2D Implementation
    # =========================================================================
    
    def _apply_2d(self, f: 'npt.NDArray') -> None:
        """Apply edge BC for all 2D edge nodes (corners in 2D sense)
        
        In 2D, "edges" are single-point intersections of two faces.
        E.g., inlet(xmin) ∩ wall(ymin) → node (0, 0)
        """
        xp = self.xp
        Nx, Ny = f.shape[1], f.shape[2]
        
        for node in self.intersections:
            if node.node_type != 'edge':
                continue
            
            # ── Step 1: Get node position ──
            # position dict has {axis: index} for participating faces
            # For non-participating axes, the edge extends along that axis
            # But in 2D with 2 faces on different axes, the node is a single point
            ix = node.position.get(0, None)  # x-index (or None if no x-face)
            iy = node.position.get(1, None)  # y-index
            
            if ix is None or iy is None:
                continue  # Need both axes defined for a point node in 2D
            
            # ── Step 2: Extract Π^neq from interior neighbors ──
            Pi_neq_sum = xp.zeros((self.dim, self.dim), dtype=xp.float64)
            rho_sum = 0.0
            u_sum = xp.zeros(self.dim, dtype=xp.float64)
            n_neighbors = 0
            
            for face in node.faces:
                # Interior neighbor: shift 1 node inward along this face's normal
                ni_x = ix + (face.inward_offset if face.axis == 0 else 0)
                ni_y = iy + (face.inward_offset if face.axis == 1 else 0)
                
                # Bounds check
                if 0 <= ni_x < Nx and 0 <= ni_y < Ny:
                    f_neighbor = f[:, ni_x, ni_y]  # shape (Q,)
                    
                    # Compute Π^neq at this neighbor
                    # Need to reshape for compute_Pi_neq: expects (Q, ...) with spatial dims
                    f_nb = f_neighbor.reshape(self.Q, 1)
                    Pi_neq_nb = compute_Pi_neq(xp, f_nb, self.c, self.w, self.cs2)
                    Pi_neq_sum += Pi_neq_nb[:, :, 0]
                    
                    # Macroscopic at neighbor
                    rho_nb = float(xp.sum(f_neighbor))
                    u_nb = xp.einsum('di,i->d', self.c, f_neighbor) / (rho_nb + 1e-30)
                    rho_sum += rho_nb
                    u_sum += u_nb
                    n_neighbors += 1
            
            if n_neighbors == 0:
                continue
            
            # ── Step 3: Average Π^neq ──
            Pi_neq_avg = Pi_neq_sum / n_neighbors  # [density × Δx²/Δt²]
            rho_interior = rho_sum / n_neighbors    # [dimensionless]
            u_interior = u_sum / n_neighbors        # [Δx/Δt]
            
            # Also check diagonal interior neighbor for better estimate
            diag_x = ix + node.faces[0].inward_offset if node.faces[0].axis == 0 else ix + node.faces[1].inward_offset if node.faces[1].axis == 0 else ix
            diag_y = iy + node.faces[0].inward_offset if node.faces[0].axis == 1 else iy + node.faces[1].inward_offset if node.faces[1].axis == 1 else iy
            
            if 0 <= diag_x < Nx and 0 <= diag_y < Ny:
                f_diag = f[:, diag_x, diag_y]
                f_diag_r = f_diag.reshape(self.Q, 1)
                Pi_neq_diag = compute_Pi_neq(xp, f_diag_r, self.c, self.w, self.cs2)
                
                # Blend: 50% face-averaged + 50% diagonal
                Pi_neq_avg = 0.5 * Pi_neq_avg + 0.5 * Pi_neq_diag[:, :, 0]
                
                # Use diagonal interior for ρ, u extrapolation (cleaner estimate)
                rho_interior = float(xp.sum(f_diag))
                u_interior = xp.einsum('di,i->d', self.c, f_diag) / (rho_interior + 1e-30)
            
            # ── Step 4: Determine target (ρ, u) ──
            u_interior_np = np.array(u_interior.get() if hasattr(u_interior, 'get') else u_interior)
            rho_target, u_target = resolve_target_rho_u(
                node, rho_interior, u_interior_np, self.dim
            )
            
            # ── Step 5: Reconstruct all Q populations ──
            rho_arr = xp.array([rho_target], dtype=xp.float64)
            u_arr = xp.asarray(u_target.reshape(self.dim, 1), dtype=xp.float64)
            Pi_neq_arr = Pi_neq_avg.reshape(self.dim, self.dim, 1)
            
            f_reconstructed = reconstruct_f_regularized(
                xp, rho=rho_arr, u=u_arr,
                Pi_neq=Pi_neq_arr,
                c=self.c, w=self.w, cs2=self.cs2
            )  # shape (Q, 1)
            
            # Write to the edge node
            f[:, ix, iy] = f_reconstructed[:, 0]
    
    # =========================================================================
    # 3D Implementation
    # =========================================================================
    
    def _apply_3d(self, f: 'npt.NDArray') -> None:
        """Apply edge/corner BC for all 3D intersection nodes
        
        Edges in 3D extend along one axis (line of nodes).
        Corners in 3D are single points.
        """
        xp = self.xp
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        shape_3d = (Nx, Ny, Nz)
        
        for node in self.intersections:
            # ── Determine fixed and free axes ──
            fixed_axes = sorted(node.position.keys())
            all_axes = [0, 1, 2]
            free_axes = [a for a in all_axes if a not in fixed_axes]
            
            if node.node_type == 'edge':
                # Edge: 2 fixed axes, 1 free axis → line of nodes
                if len(free_axes) != 1:
                    continue
                free_axis = free_axes[0]
                free_size = shape_3d[free_axis]
                
                self._apply_3d_edge(f, node, fixed_axes, free_axis, free_size,
                                    Nx, Ny, Nz)
            
            elif node.node_type == 'corner':
                # Corner: 3 fixed axes → single node
                if len(fixed_axes) != 3:
                    continue
                self._apply_3d_corner(f, node, Nx, Ny, Nz)
    
    def _apply_3d_edge(self, f: 'npt.NDArray', node: IntersectionNode,
                       fixed_axes: List[int], free_axis: int, free_size: int,
                       Nx: int, Ny: int, Nz: int) -> None:
        """Apply regularized BC along a 3D edge (line of nodes)
        
        An edge in 3D is where 2 faces meet, extending along the free axis.
        E.g., xmin ∩ ymin → edge along z-axis at (0, 0, 0:Nz)
        """
        xp = self.xp
        shape_3d = (Nx, Ny, Nz)
        
        # Fixed positions
        pos = [0, 0, 0]
        for ax in fixed_axes:
            pos[ax] = node.position[ax]
        
        # Extract the edge line of f values: shape (Q, free_size)
        idx = [slice(None)] * 4  # [Q, x, y, z]
        for ax in fixed_axes:
            idx[ax + 1] = pos[ax]
        idx[free_axis + 1] = slice(None)
        f_edge = f[tuple(idx)]  # shape (Q, free_size)
        
        # ── Collect Π^neq from interior neighbors ──
        Pi_neq_sum = xp.zeros((self.dim, self.dim, free_size), dtype=xp.float64)
        rho_sum = xp.zeros(free_size, dtype=xp.float64)
        u_sum = xp.zeros((self.dim, free_size), dtype=xp.float64)
        n_neighbors = 0
        
        for face in node.faces:
            # Interior neighbor: shift along this face's normal
            ni_pos = list(pos)
            ni_pos[face.axis] += face.inward_offset
            
            if 0 <= ni_pos[face.axis] < shape_3d[face.axis]:
                idx_nb = [slice(None)] * 4
                for ax in fixed_axes:
                    idx_nb[ax + 1] = ni_pos[ax]
                idx_nb[free_axis + 1] = slice(None)
                
                f_nb = f[tuple(idx_nb)]  # shape (Q, free_size)
                
                Pi_neq_nb = compute_Pi_neq(xp, f_nb, self.c, self.w, self.cs2)
                Pi_neq_sum += Pi_neq_nb
                
                rho_nb = xp.sum(f_nb, axis=0)
                u_nb = xp.einsum('di,i...->d...', self.c, f_nb) / (rho_nb + 1e-30)
                rho_sum += rho_nb
                u_sum += u_nb
                n_neighbors += 1
        
        if n_neighbors == 0:
            return
        
        Pi_neq_avg = Pi_neq_sum / n_neighbors
        rho_interior = rho_sum / n_neighbors
        u_interior = u_sum / n_neighbors
        
        # ── Also blend diagonal interior neighbor ──
        diag_pos = list(pos)
        for face in node.faces:
            diag_pos[face.axis] += face.inward_offset
        
        all_valid = all(0 <= diag_pos[ax] < shape_3d[ax] for ax in fixed_axes)
        if all_valid:
            idx_diag = [slice(None)] * 4
            for ax in range(3):
                if ax in fixed_axes:
                    idx_diag[ax + 1] = diag_pos[ax]
                else:
                    idx_diag[ax + 1] = slice(None)
            
            f_diag = f[tuple(idx_diag)]  # (Q, free_size)
            Pi_neq_diag = compute_Pi_neq(xp, f_diag, self.c, self.w, self.cs2)
            
            Pi_neq_avg = 0.5 * Pi_neq_avg + 0.5 * Pi_neq_diag
            
            rho_interior = xp.sum(f_diag, axis=0)
            u_interior = xp.einsum('di,i...->d...', self.c, f_diag) / (rho_interior + 1e-30)
        
        # ── Resolve target (ρ, u) for the whole edge line ──
        rho_int_mean = float(xp.mean(rho_interior))
        u_int_mean = np.zeros(self.dim)
        for d in range(self.dim):
            u_int_mean[d] = float(xp.mean(u_interior[d]))
        
        rho_target_scalar, u_target_vec = resolve_target_rho_u(
            node, rho_int_mean, u_int_mean, self.dim
        )
        
        # Build target arrays for the edge line
        # For PRESSURE types, apply per-node relaxation
        has_pressure = any(f.bc_type == BCType.PRESSURE for f in node.faces)
        has_wall = any(f.bc_type == BCType.WALL for f in node.faces)
        
        if has_pressure and not has_wall:
            # Per-node pressure relaxation
            pressure_face = next(f for f in node.faces if f.bc_type == BCType.PRESSURE)
            K = pressure_face.relax_coeff
            rho_target = pressure_face.density + (1.0 - K) * (rho_interior - pressure_face.density)
        elif has_wall and has_pressure:
            pressure_face = next(f for f in node.faces if f.bc_type == BCType.PRESSURE)
            K = pressure_face.relax_coeff
            rho_target = pressure_face.density + (1.0 - K) * (rho_interior - pressure_face.density)
        else:
            rho_target = rho_interior  # Extrapolate from interior
        
        # Velocity target
        u_target = xp.zeros((self.dim, free_size), dtype=xp.float64)
        if has_wall:
            u_target[:] = 0.0  # [Δx/Δt]
        else:
            for face in node.faces:
                if face.bc_type == BCType.VELOCITY:
                    sign = 1 if face.is_min else -1
                    u_target[face.axis, :] = sign * face.velocity
                elif face.bc_type in (BCType.PRESSURE, BCType.UNKNOWN):
                    u_target[face.axis, :] = u_interior[face.axis]
        
        # ── Reconstruct ──
        f_reconstructed = reconstruct_f_regularized(
            xp, rho=rho_target, u=u_target,
            Pi_neq=Pi_neq_avg,
            c=self.c, w=self.w, cs2=self.cs2
        )
        
        # ── Write back to edge ──
        f[tuple(idx)] = f_reconstructed
    
    def _apply_3d_corner(self, f: 'npt.NDArray', node: IntersectionNode,
                         Nx: int, Ny: int, Nz: int) -> None:
        """Apply regularized BC at a 3D corner node (single point)
        
        Where 3 faces meet: e.g., xmin ∩ ymin ∩ zmin → (0, 0, 0)
        """
        xp = self.xp
        shape_3d = (Nx, Ny, Nz)
        
        ix = node.position[0]
        iy = node.position[1]
        iz = node.position[2]
        
        # ── Collect Π^neq from interior neighbors ──
        Pi_neq_sum = xp.zeros((self.dim, self.dim), dtype=xp.float64)
        rho_sum = 0.0
        u_sum = xp.zeros(self.dim, dtype=xp.float64)
        n_neighbors = 0
        
        for face in node.faces:
            ni = [ix, iy, iz]
            ni[face.axis] += face.inward_offset
            
            if 0 <= ni[0] < Nx and 0 <= ni[1] < Ny and 0 <= ni[2] < Nz:
                f_nb = f[:, ni[0], ni[1], ni[2]].reshape(self.Q, 1)
                Pi_neq_nb = compute_Pi_neq(xp, f_nb, self.c, self.w, self.cs2)
                Pi_neq_sum += Pi_neq_nb[:, :, 0]
                
                rho_nb = float(xp.sum(f[:, ni[0], ni[1], ni[2]]))
                u_nb = xp.einsum('di,i->d', self.c, f[:, ni[0], ni[1], ni[2]]) / (rho_nb + 1e-30)
                rho_sum += rho_nb
                u_sum += u_nb
                n_neighbors += 1
        
        if n_neighbors == 0:
            return
        
        Pi_neq_avg = Pi_neq_sum / n_neighbors
        rho_interior = rho_sum / n_neighbors
        u_interior = u_sum / n_neighbors
        
        # Diagonal interior
        diag = [
            ix + next((f.inward_offset for f in node.faces if f.axis == 0), 0),
            iy + next((f.inward_offset for f in node.faces if f.axis == 1), 0),
            iz + next((f.inward_offset for f in node.faces if f.axis == 2), 0),
        ]
        
        if 0 <= diag[0] < Nx and 0 <= diag[1] < Ny and 0 <= diag[2] < Nz:
            f_diag = f[:, diag[0], diag[1], diag[2]].reshape(self.Q, 1)
            Pi_neq_diag = compute_Pi_neq(xp, f_diag, self.c, self.w, self.cs2)
            Pi_neq_avg = 0.5 * Pi_neq_avg + 0.5 * Pi_neq_diag[:, :, 0]
            
            rho_interior = float(xp.sum(f[:, diag[0], diag[1], diag[2]]))
            u_interior = xp.einsum('di,i->d', self.c,
                                   f[:, diag[0], diag[1], diag[2]]) / (rho_interior + 1e-30)
        
        # Resolve target
        u_int_np = np.array(u_interior.get() if hasattr(u_interior, 'get') else u_interior)
        rho_target, u_target = resolve_target_rho_u(
            node, rho_interior, u_int_np, self.dim
        )
        
        # Reconstruct
        rho_arr = xp.array([rho_target], dtype=xp.float64)
        u_arr = xp.asarray(u_target.reshape(self.dim, 1), dtype=xp.float64)
        Pi_neq_arr = Pi_neq_avg.reshape(self.dim, self.dim, 1)
        
        f_reconstructed = reconstruct_f_regularized(
            xp, rho=rho_arr, u=u_arr,
            Pi_neq=Pi_neq_arr,
            c=self.c, w=self.w, cs2=self.cs2
        )
        
        f[:, ix, iy, iz] = f_reconstructed[:, 0]
    
    # =========================================================================
    # Info / Debug
    # =========================================================================
    
    def get_info(self) -> str:
        """Return summary string"""
        n_edges = sum(1 for n in self.intersections if n.node_type == 'edge')
        n_corners = sum(1 for n in self.intersections if n.node_type == 'corner')
        dim_str = "2D" if self.dim == 2 else "3D"
        
        lines = [f"RegularizedEdgeCornerManager ({dim_str}):"]
        lines.append(f"  Configured faces: {len(self.face_infos)}")
        lines.append(f"  Edge intersections: {n_edges}")
        lines.append(f"  Corner intersections: {n_corners}")
        
        for node in self.intersections:
            faces_str = " ∩ ".join(f"{f.location}({f.bc_type.name})" for f in node.faces)
            pos_str = ", ".join(f"{'xyz'[a]}={v}" for a, v in sorted(node.position.items()))
            lines.append(f"    {node.node_type}: {faces_str} @ ({pos_str})")
        
        return "\n".join(lines)