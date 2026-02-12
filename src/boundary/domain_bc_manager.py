"""
Domain Boundary Condition Manager

Unified orchestrator for all domain boundary conditions, following
the Palabos "One Node, One Dynamics" principle.

Architecture:
    1. Parse config → FaceConfig objects
    2. Build NodeMap → classify every boundary node (flat/edge/corner)
    3. Create FaceBCs → each operates ONLY on flat nodes
    4. Create CornerBC → applies f = f_eq at edge/corner nodes
    5. Create SpongeLayerBCs → volume-based damping (if configured)

Time Loop:
    bc_manager.apply_all(f, f_post)
        Phase 1: Face BCs on flat nodes  (regularized, equilibrium, BB, neumann)
        Phase 2: CornerBC on edge/corner nodes  (pure equilibrium)
        Phase 3: Sponge layers on buffer zone volumes  (damping toward f_eq)

Usage:
    from src.boundary.domain_bc_manager import DomainBCManager
    
    bc_manager = DomainBCManager(xp, lattice, boundaries_config, domain_shape)
    
    # In time loop:
    bc_manager.apply_all(f_new, f_post)

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .bc_config import (
    FaceConfig, FaceLocation, BCType,
    parse_all_boundaries,
)
from .node_map import NodeMap
from .face_bc import FaceBC, create_face_bc
from .corner_bc import CornerBC
from .sponge import SpongeLayerBC


class DomainBCManager:
    """Unified domain boundary condition manager.
    
    Encapsulates the complete BC system:
        - Node classification (NodeMap)
        - Face BCs (flat nodes only)
        - Edge/corner BC (pure equilibrium)
        - Sponge layers (volume damping)
    
    Guarantees "One Node, One Dynamics": every boundary node is written
    to by exactly one BC, exactly once per time step (Phase 1+2).
    Sponge layers in Phase 3 further modify the buffer zone.
    
    Args:
        xp: Array module (numpy or cupy)
        lattice: Lattice model (D2Q9, D3Q27)
        boundaries_config: Raw config dictionary from input file
        domain_shape: (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        verbose: Print initialization summary
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 boundaries_config: Dict[str, Dict[str, Any]],
                 domain_shape: Tuple[int, ...],
                 verbose: bool = True) -> None:
        self.xp = xp
        self.lattice = lattice
        self.domain_shape = domain_shape
        self.dim = lattice.dim
        
        # =====================================================================
        # Phase 1: Parse configuration
        # =====================================================================
        self.face_configs: List[FaceConfig] = parse_all_boundaries(boundaries_config)
        
        if verbose:
            print(f"  Parsed {len(self.face_configs)} boundary faces:")
            for fc in self.face_configs:
                self._print_face_config(fc)
        
        # =====================================================================
        # Phase 2: Build node map (flat/edge/corner classification)
        # =====================================================================
        self.node_map = NodeMap(domain_shape, self.face_configs, self.dim)
        
        if verbose:
            print(self.node_map.summary())
        
        # =====================================================================
        # Phase 3: Create face BCs (flat nodes only) + sponge layers
        # =====================================================================
        self.face_bcs: List[FaceBC] = []
        self.sponge_layers: List[SpongeLayerBC] = []
        
        for fc in self.face_configs:
            if fc.bc_type == BCType.PERIODIC:
                continue
            
            if fc.bc_type == BCType.SPONGE:
                sponge = SpongeLayerBC(xp, lattice, fc, domain_shape)
                self.sponge_layers.append(sponge)
                if verbose:
                    print(f"    {sponge.get_info()}")
                continue
            
            face_bc = create_face_bc(xp, lattice, fc, self.node_map)
            self.face_bcs.append(face_bc)
        
        if verbose:
            print(f"  Face BCs created: {len(self.face_bcs)}")
            for fbc in self.face_bcs:
                print(f"    {fbc.get_info()}")
            if self.sponge_layers:
                print(f"  Sponge layers: {len(self.sponge_layers)}")
        
        # =====================================================================
        # Phase 4: Create corner BC (edge/corner → f = f_eq)
        # =====================================================================
        self.corner_bc: Optional[CornerBC] = None
        
        n_edge = self.node_map.n_edges
        n_corner = self.node_map.n_corners
        
        if n_edge > 0 or n_corner > 0:
            self.corner_bc = CornerBC(xp, lattice, self.node_map)
            if verbose:
                print(f"  {self.corner_bc.get_info()}")
        else:
            if verbose:
                print("  Edge/Corner: none (all faces periodic or single face)")
    
    # =========================================================================
    # Time Loop Interface
    # =========================================================================
    
    def apply_all(self, f: 'npt.NDArray',
                  f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply all domain BCs in correct order.
        
        Phase 1: Face BCs write to flat nodes only
        Phase 2: CornerBC writes to edge/corner nodes (f = f_eq)
        Phase 3: Sponge layers damp buffer zone volumes
        
        Each boundary node is written EXACTLY ONCE by Phase 1+2.
        Sponge in Phase 3 further modifies the buffer zone.
        
        Args:
            f: Distribution after streaming, modified in-place
            f_post: Post-collision distribution (needed for bounce-back BCs)
        """
        # Phase 1: Face BCs (flat nodes only)
        for face_bc in self.face_bcs:
            face_bc.apply(f, f_post)
        
        # Phase 2: Edge/corner nodes → f = f_eq
        if self.corner_bc is not None:
            self.corner_bc.apply(f)
        
        # Phase 3: Sponge layers (volume damping, applied last)
        for sponge in self.sponge_layers:
            sponge.apply(f)
    
    # =========================================================================
    # Query Interface
    # =========================================================================
    
    @property
    def n_face_bcs(self) -> int:
        """Number of face BCs."""
        return len(self.face_bcs)
    
    @property
    def n_sponge_layers(self) -> int:
        """Number of sponge layers."""
        return len(self.sponge_layers)
    
    @property
    def n_edges(self) -> int:
        """Number of edge nodes/lines."""
        return self.node_map.n_edges
    
    @property
    def n_corners(self) -> int:
        """Number of corner nodes."""
        return self.node_map.n_corners
    
    def has_walls(self) -> bool:
        """Check if any wall BCs are present."""
        return any(fc.bc_type == BCType.WALL for fc in self.face_configs)
    
    def get_wall_locations(self) -> List[str]:
        """Get list of wall face location strings."""
        return [fc.location.value for fc in self.face_configs 
                if fc.bc_type == BCType.WALL]
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    @staticmethod
    def _print_face_config(fc: FaceConfig) -> None:
        """Print a single FaceConfig for verbose output."""
        parts = [f"    {fc.location.value}: {fc.method}"]
        
        if fc.bc_type == BCType.VELOCITY:
            parts.append(f"u={fc.velocity}")
            parts.append(f"ρ={fc.density}")
            mode = "equilibrium" if not fc.use_regularized else "regularized"
            parts.append(f"({mode})")
        elif fc.bc_type == BCType.PRESSURE:
            parts.append(f"ρ_target={fc.density}")
            parts.append(f"K={fc.relax_coeff}")
        elif fc.bc_type == BCType.WALL:
            mode = "bounce-back" if not fc.use_regularized else "regularized"
            parts.append(f"({mode})")
        elif fc.bc_type == BCType.NEUMANN:
            parts.append("(zero-gradient)")
        elif fc.bc_type == BCType.SPONGE:
            L = fc.extra.get('thickness', 20)
            sigma = fc.extra.get('sigma_max', 0.5)
            parts.append(f"L={L}, σ_max={sigma}")
        
        print(', '.join(parts))
    
    def get_summary(self) -> str:
        """Return full summary string."""
        lines = [
            f"DomainBCManager ({self.dim}D):",
            f"  Face BCs: {self.n_face_bcs}",
        ]
        for fbc in self.face_bcs:
            lines.append(f"    {fbc.get_info()}")
        
        if self.sponge_layers:
            lines.append(f"  Sponge layers: {self.n_sponge_layers}")
            for sl in self.sponge_layers:
                lines.append(f"    {sl.get_info()}")
        
        if self.corner_bc:
            lines.append(f"  {self.corner_bc.get_info()}")
        else:
            lines.append("  Corners: none")
        
        return '\n'.join(lines)