"""
Node Classification Map for Domain Boundaries

Implements the Palabos "One Node, One Dynamics" principle by classifying
every boundary node at initialization time:

    Flat:   node belongs to exactly 1 boundary face
    Edge:   node belongs to exactly 2 faces (line in 3D, point in 2D)
    Corner: node belongs to exactly 3 faces (point in 3D only)

Each face BC operates ONLY on its flat nodes.
Edge/corner nodes receive pure equilibrium: f = f_eq(ρ_target, u_target).

This classification is computed once and never changes during simulation.

         Palabos Node Map (2D example)
    ┌─────────────────────────────────────┐
    │ E ─ W ─ W ─ W ─ W ─ W ─ W ─ W ─ E │  ymax
    │ │                               │ │
    │ I   ·   ·   ·   ·   ·   ·   ·   O │
    │ I   ·   ·   ·   ·   ·   ·   ·   O │
    │ │                               │ │
    │ E ─ W ─ W ─ W ─ W ─ W ─ W ─ W ─ E │  ymin
    └─────────────────────────────────────┘
     xmin                             xmax

    E = Edge (2 faces meet)  → f = f_eq
    I = Inlet flat node      → face BC
    O = Outlet flat node     → face BC
    W = Wall flat node       → face BC

Author: LBM Development Team
Date: 2026-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .bc_config import FaceConfig, FaceLocation, BCType


# =============================================================================
# Data Structures for Edge/Corner Nodes
# =============================================================================

@dataclass
class EdgeNode:
    """A node where exactly 2 boundary faces meet.
    
    In 2D: a single point (e.g., inlet ∩ wall → corner of domain).
    In 3D: a line of nodes along the shared axis.
    
    Attributes:
        face_a: First FaceConfig
        face_b: Second FaceConfig
        position: Dict mapping axis→index for the fixed axes
                  e.g. {0: 0, 1: 0} means x=0, y=0 (the free axis is z)
        free_axis: The axis along which edge extends (3D only, None for 2D)
    """
    face_a: FaceConfig
    face_b: FaceConfig
    position: Dict[int, int]    # axis → grid index
    free_axis: Optional[int] = None


@dataclass
class CornerNode:
    """A node where exactly 3 boundary faces meet (3D only).
    
    Always a single point — all 3 axes are fixed.
    
    Attributes:
        face_a, face_b, face_c: The three FaceConfig objects
        position: Dict mapping all 3 axes to grid indices
                  e.g. {0: 0, 1: 0, 2: 0}
    """
    face_a: FaceConfig
    face_b: FaceConfig
    face_c: FaceConfig
    position: Dict[int, int]    # axis → grid index (all 3 axes)


# =============================================================================
# Node Map
# =============================================================================

class NodeMap:
    """Classifies boundary nodes into flat / edge / corner categories.
    
    Computed once at initialization. Provides:
        - flat_masks:   {FaceLocation: boolean mask} for each face
                        True = flat node (face BC applies here)
                        False = edge/corner (excluded from face BC)
        - edge_nodes:   List[EdgeNode]
        - corner_nodes: List[CornerNode] (3D only)
    
    Args:
        domain_shape: (Nx, Ny) for 2D or (Nx, Ny, Nz) for 3D  [lattice units]
        face_configs: List of FaceConfig for all non-periodic faces
        dim: Spatial dimension (2 or 3)
    """
    
    def __init__(self, domain_shape: Tuple[int, ...],
                 face_configs: List[FaceConfig],
                 dim: int) -> None:
        self.domain_shape = domain_shape
        self.face_configs = face_configs
        self.dim = dim
        
        # Build lookup: FaceLocation → FaceConfig
        self._face_map: Dict[FaceLocation, FaceConfig] = {
            fc.location: fc for fc in face_configs
        }
        
        # Results
        self.flat_masks: Dict[FaceLocation, 'tuple'] = {}  # slicing tuples
        self.edge_nodes: List[EdgeNode] = []
        self.corner_nodes: List[CornerNode] = []
        
        # Build
        self._classify()
    
    # =========================================================================
    # Public Properties
    # =========================================================================
    
    @property
    def n_edges(self) -> int:
        return len(self.edge_nodes)
    
    @property
    def n_corners(self) -> int:
        return len(self.corner_nodes)
    
    def get_face_config(self, location: FaceLocation) -> Optional[FaceConfig]:
        """Get FaceConfig for a given location, or None if periodic."""
        return self._face_map.get(location)
    
    # =========================================================================
    # Classification Logic
    # =========================================================================
    
    def _classify(self) -> None:
        """Main classification: find edges, corners, build flat masks."""
        if self.dim == 2:
            self._classify_2d()
        else:
            self._classify_3d()
    
    def _classify_2d(self) -> None:
        """Classify nodes for 2D domain.
        
        2D domain has 4 possible faces (xmin, xmax, ymin, ymax).
        Edges are points where 2 faces on DIFFERENT axes meet.
        No corners in 2D.
        
        Face layout:
            xmin face: nodes (0, 0..Ny-1)     → flat mask excludes y-endpoints if walls
            xmax face: nodes (Nx-1, 0..Ny-1)   
            ymin face: nodes (0..Nx-1, 0)       → flat mask excludes x-endpoints if inlet/outlet
            ymax face: nodes (0..Nx-1, Ny-1)
        """
        Nx, Ny = self.domain_shape
        fm = self._face_map
        
        # --- Find all edge nodes (2 faces on different axes) ---
        # Possible edge pairs: (x-face, y-face)
        x_faces = [loc for loc in fm if loc.axis == 0]  # xmin, xmax
        y_faces = [loc for loc in fm if loc.axis == 1]  # ymin, ymax
        
        for x_loc in x_faces:
            for y_loc in y_faces:
                # These two faces meet at a single point
                ix = 0 if x_loc == FaceLocation.XMIN else Nx - 1
                iy = 0 if y_loc == FaceLocation.YMIN else Ny - 1
                
                edge = EdgeNode(
                    face_a=fm[x_loc],
                    face_b=fm[y_loc],
                    position={0: ix, 1: iy},
                    free_axis=None,  # No free axis in 2D
                )
                self.edge_nodes.append(edge)
        
        # --- Build flat masks (slices that exclude edge nodes) ---
        edge_positions = set()
        for e in self.edge_nodes:
            edge_positions.add((e.position[0], e.position[1]))
        
        for loc, fc in fm.items():
            axis = loc.axis
            gi = 0 if loc.is_min else self.domain_shape[axis] - 1
            
            if axis == 0:
                # x-face: spans y-direction. Exclude y-indices at edges.
                y_start, y_end = 0, Ny
                for e in self.edge_nodes:
                    if e.position[0] == gi:
                        ey = e.position[1]
                        if ey == 0:
                            y_start = 1
                        elif ey == Ny - 1:
                            y_end = Ny - 1
                self.flat_masks[loc] = (y_start, y_end)
            else:
                # y-face: spans x-direction. Exclude x-indices at edges.
                x_start, x_end = 0, Nx
                for e in self.edge_nodes:
                    if e.position[1] == gi:
                        ex = e.position[0]
                        if ex == 0:
                            x_start = 1
                        elif ex == Nx - 1:
                            x_end = Nx - 1
                self.flat_masks[loc] = (x_start, x_end)
    
    def _classify_3d(self) -> None:
        """Classify nodes for 3D domain.
        
        3D domain has up to 6 faces. 
        Edges: where 2 faces on different axes meet → line of nodes.
        Corners: where 3 faces on all 3 different axes meet → single point.
        """
        Nx, Ny, Nz = self.domain_shape
        fm = self._face_map
        shape_by_axis = {0: Nx, 1: Ny, 2: Nz}
        
        # Group faces by axis
        faces_by_axis: Dict[int, List[FaceLocation]] = {0: [], 1: [], 2: []}
        for loc in fm:
            faces_by_axis[loc.axis].append(loc)
        
        # --- Find corners (3 faces, all on different axes) ---
        for x_loc in faces_by_axis[0]:
            for y_loc in faces_by_axis[1]:
                for z_loc in faces_by_axis[2]:
                    ix = 0 if x_loc == FaceLocation.XMIN else Nx - 1
                    iy = 0 if y_loc == FaceLocation.YMIN else Ny - 1
                    iz = 0 if z_loc == FaceLocation.ZMIN else Nz - 1
                    
                    corner = CornerNode(
                        face_a=fm[x_loc],
                        face_b=fm[y_loc],
                        face_c=fm[z_loc],
                        position={0: ix, 1: iy, 2: iz},
                    )
                    self.corner_nodes.append(corner)
        
        # --- Find edges (2 faces on different axes) ---
        # For each pair of axes, find face combinations
        axis_pairs = [(0, 1), (0, 2), (1, 2)]
        
        for ax_a, ax_b in axis_pairs:
            ax_free = 3 - ax_a - ax_b  # the remaining axis
            
            for loc_a in faces_by_axis[ax_a]:
                for loc_b in faces_by_axis[ax_b]:
                    idx_a = 0 if loc_a.is_min else shape_by_axis[ax_a] - 1
                    idx_b = 0 if loc_b.is_min else shape_by_axis[ax_b] - 1
                    
                    edge = EdgeNode(
                        face_a=fm[loc_a],
                        face_b=fm[loc_b],
                        position={ax_a: idx_a, ax_b: idx_b},
                        free_axis=ax_free,
                    )
                    self.edge_nodes.append(edge)
        
        # --- Build flat masks ---
        # For each face, determine the range that excludes edge/corner nodes
        # A face at axis=a spans the other 2 axes. We need to trim those axes
        # wherever another face exists.
        
        for loc, fc in fm.items():
            axis = loc.axis
            other_axes = sorted([a for a in range(3) if a != axis])
            
            # For each transverse axis, determine start/end
            ranges = {}
            for oa in other_axes:
                start = 0
                end = shape_by_axis[oa]
                
                # Check if there's a face at min/max of this axis
                min_loc = FaceLocation(('xmin', 'ymin', 'zmin')[oa])
                max_loc = FaceLocation(('xmax', 'ymax', 'zmax')[oa])
                
                if min_loc in fm:
                    start = 1
                if max_loc in fm:
                    end = shape_by_axis[oa] - 1
                
                ranges[oa] = (start, end)
            
            self.flat_masks[loc] = {oa: ranges[oa] for oa in other_axes}
    
    # =========================================================================
    # Slice Generators (used by face BCs)
    # =========================================================================
    
    def get_face_slice_2d(self, location: FaceLocation) -> Tuple[slice, slice]:
        """Get (x_slice, y_slice) to index flat nodes on a 2D face.
        
        The returned slices index into f[:, x, y] to select ONLY flat nodes
        on the given face (edge nodes excluded).
        
        Args:
            location: Which face
            
        Returns:
            (x_slice, y_slice) for indexing f[:, x_slice, y_slice]
        """
        Nx, Ny = self.domain_shape
        axis = location.axis
        gi = 0 if location.is_min else self.domain_shape[axis] - 1
        
        flat_range = self.flat_masks.get(location, None)
        
        if axis == 0:
            # x-face at x=gi, flat range is for y
            if flat_range is not None:
                y_start, y_end = flat_range
            else:
                y_start, y_end = 0, Ny
            return (gi, slice(y_start, y_end))
        else:
            # y-face at y=gi, flat range is for x
            if flat_range is not None:
                x_start, x_end = flat_range
            else:
                x_start, x_end = 0, Nx
            return (slice(x_start, x_end), gi)
    
    def get_face_slice_3d(self, location: FaceLocation) -> Tuple[
            'slice_or_int', 'slice_or_int', 'slice_or_int']:
        """Get (x_slice, y_slice, z_slice) to index flat nodes on a 3D face.
        
        Returns:
            (x_slice, y_slice, z_slice) for indexing f[:, x, y, z]
        """
        Nx, Ny, Nz = self.domain_shape
        axis = location.axis
        gi = 0 if location.is_min else self.domain_shape[axis] - 1
        
        flat_ranges = self.flat_masks.get(location, {})
        
        # Build slices for each axis
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = gi  # Fix the normal axis
        
        for oa in sorted(flat_ranges.keys()):
            start, end = flat_ranges[oa]
            slices[oa] = slice(start, end)
        
        return tuple(slices)
    
    def get_interior_slice_2d(self, location: FaceLocation) -> Tuple[int, slice]:
        """Get slice for interior neighbor nodes (1 node inside from face).
        
        Same transverse range as the flat mask, but shifted 1 node inward.
        """
        face_slice = self.get_face_slice_2d(location)
        offset = location.inward_offset
        axis = location.axis
        gi = 0 if location.is_min else self.domain_shape[axis] - 1
        
        if axis == 0:
            return (gi + offset, face_slice[1])
        else:
            return (face_slice[0], gi + offset)
    
    def get_interior_slice_3d(self, location: FaceLocation) -> Tuple:
        """Get slice for interior neighbor nodes in 3D."""
        face_slice = list(self.get_face_slice_3d(location))
        axis = location.axis
        gi = 0 if location.is_min else self.domain_shape[axis] - 1
        offset = location.inward_offset
        
        face_slice[axis] = gi + offset
        return tuple(face_slice)
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    def summary(self) -> str:
        """Return human-readable summary of node classification."""
        lines = [f"  NodeMap ({self.dim}D, shape={self.domain_shape}):"]
        
        lines.append(f"    Faces: {len(self.face_configs)}")
        for fc in self.face_configs:
            lines.append(f"      {fc.location.value}: {fc.bc_type.value} "
                        f"({fc.method})")
        
        lines.append(f"    Edges: {self.n_edges}")
        for e in self.edge_nodes:
            fa = e.face_a.location.value
            fb = e.face_b.location.value
            pos = ', '.join(f"{'xyz'[k]}={v}" for k, v in sorted(e.position.items()))
            if e.free_axis is not None:
                pos += f", free={'xyz'[e.free_axis]}"
            lines.append(f"      {fa} ∩ {fb} @ ({pos})")
        
        lines.append(f"    Corners: {self.n_corners}")
        for c in self.corner_nodes:
            fa = c.face_a.location.value
            fb = c.face_b.location.value
            fc_loc = c.face_c.location.value
            pos = ', '.join(f"{'xyz'[k]}={v}" for k, v in sorted(c.position.items()))
            lines.append(f"      {fa} ∩ {fb} ∩ {fc_loc} @ ({pos})")
        
        return '\n'.join(lines)