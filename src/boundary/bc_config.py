"""
Boundary Condition Configuration Module

Defines the fundamental types and data structures for the Palabos-style
BC architecture ("One Node, One Dynamics").

Responsibilities:
    1. BCType / FaceLocation enumerations
    2. FaceConfig dataclass — one per domain boundary face
    3. Config parsing — raw user dict → List[FaceConfig]

Each domain boundary face is described by exactly one FaceConfig.
At initialization time, every boundary node is classified as:
    - Flat:   belongs to exactly 1 non-periodic face  → face BC
    - Edge:   belongs to exactly 2 non-periodic faces → f = f_eq
    - Corner: belongs to exactly 3 non-periodic faces → f = f_eq (3D only)
This classification is immutable during the simulation.

Supported Methods:
    Velocity Inlets:
        'regularized_inlet' / 'reg_inlet'   — f = f_eq + f^(1)(Π^neq)
        'equilibrium' / 'eq'                — f = f_eq(ρ₀, u_target)
    
    Pressure Outlets:
        'regularized_outlet' / 'reg_outlet' — ρ relaxation + f^(1)
    
    Domain Walls:
        'regularized_wall' / 'reg_wall'     — f = f_eq(ρ_ext, 0) + f^(1)
        'bounce_back' / 'hwbb'              — f_i = f_ī (half-way BB)
    
    Zero-Gradient:
        'neumann' / 'zero_gradient'         — f[bdy] = f[interior]
    
    Sponge Layer:
        'sponge' / 'sponge_layer'           — volume damping toward f_eq
    
    Periodic:
        'periodic' / 'none'                 — no BC (streaming handles it)

Author: LBM Development Team
Date: 2026-02
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any


# =============================================================================
# BCType — physical boundary condition classification
# =============================================================================

class BCType(Enum):
    """Physical type of a boundary condition.
    
    Determines what macroscopic quantity is prescribed at the boundary face
    and how (ρ, u) conflicts are resolved at edge/corner nodes.
    
    Types:
        VELOCITY:   u is given (Dirichlet for velocity — inlet)
        PRESSURE:   ρ is given (Dirichlet for pressure — outlet)
        WALL:       u = 0 or u = u_wall (no-slip — domain wall)
        NEUMANN:    ∂f/∂n = 0 (zero-gradient — open boundary)
        SPONGE:     volumetric damping layer (not face-based)
        PERIODIC:   no explicit BC (handled by streaming periodicity)
    
    Edge/corner priority (highest → lowest):
        WALL > VELOCITY > PRESSURE > NEUMANN > PERIODIC
    """
    VELOCITY   = "velocity"
    PRESSURE   = "pressure"
    WALL       = "wall"
    NEUMANN    = "neumann"
    SPONGE     = "sponge"
    PERIODIC   = "periodic"


# =============================================================================
# FaceLocation — domain boundary face identifiers
# =============================================================================

class FaceLocation(Enum):
    """Domain boundary face identifiers.
    
    Convention:
        axis 0 = x,  axis 1 = y,  axis 2 = z
        'min' = index 0 side,  'max' = last index side
    
    Example (2D domain, Nx=100, Ny=40):
        XMIN: face at x=0,    spanning y=0..39
        XMAX: face at x=99,   spanning y=0..39
        YMIN: face at y=0,    spanning x=0..99
        YMAX: face at y=39,   spanning x=0..99
    """
    XMIN = "xmin"
    XMAX = "xmax"
    YMIN = "ymin"
    YMAX = "ymax"
    ZMIN = "zmin"
    ZMAX = "zmax"

    @classmethod
    def from_string(cls, name: str) -> 'FaceLocation':
        """Parse location from string, supporting legacy names.
        
        Supported inputs:
            'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'  (canonical)
            'west', 'east', 'south', 'north', 'bottom', 'top'  (legacy)
        
        Args:
            name: Location string (case-insensitive)
            
        Returns:
            FaceLocation enum value
            
        Raises:
            ValueError: If name is not recognized
        """
        _LEGACY_MAP = {
            'west': 'xmin', 'east': 'xmax',
            'south': 'ymin', 'north': 'ymax',
            'bottom': 'zmin', 'top': 'zmax',
        }
        key = _LEGACY_MAP.get(name.lower().strip(), name.lower().strip())
        
        for member in cls:
            if member.value == key:
                return member
        
        valid = [m.value for m in cls] + list(_LEGACY_MAP.keys())
        raise ValueError(f"Unknown face location: '{name}'. Valid: {valid}")

    @property
    def axis(self) -> int:
        """Normal axis index: xmin/xmax → 0, ymin/ymax → 1, zmin/zmax → 2"""
        return {'xmin': 0, 'xmax': 0,
                'ymin': 1, 'ymax': 1,
                'zmin': 2, 'zmax': 2}[self.value]

    @property
    def is_min(self) -> bool:
        """True if this is the low-index side (grid index = 0)."""
        return self.value.endswith('min')

    @property
    def grid_index(self) -> int:
        """Grid index of this face: 0 for min faces, -1 for max faces.
        
        Note: -1 means "last index" in NumPy slicing. For exact index,
        use domain_shape[self.axis] - 1.
        """
        return 0 if self.is_min else -1

    @property
    def inward_offset(self) -> int:
        """Offset from face to first interior neighbor node.
        
        +1 for min faces (interior is at index 1),
        -1 for max faces (interior is at index N-2).
        """
        return +1 if self.is_min else -1


# =============================================================================
# FaceConfig — configuration for a single domain boundary face
# =============================================================================

@dataclass
class FaceConfig:
    """Configuration for a single domain boundary face.
    
    Created once during initialization and never modified.
    Encapsulates all information needed to:
        - Apply the face BC on flat nodes
        - Resolve (ρ, u) conflicts at edge/corner nodes
    
    Attributes:
        location:        Which face (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
        bc_type:         Physical BC type (VELOCITY, PRESSURE, WALL, etc.)
        method:          Specific method string from config (for logging)
        velocity:        Prescribed velocity — scalar or [ux, uy, uz]  [Δx/Δt]
        density:         Prescribed density  [dimensionless, ρ₀ = 1.0]
        relax_coeff:     Pressure relaxation coefficient K  [dimensionless, 0 < K ≤ 1]
        use_regularized: True → f = f_eq + f^(1)(Π^neq);  False → f = f_eq
        extra:           Additional method-specific parameters (sponge thickness, etc.)
    """
    location: FaceLocation
    bc_type: BCType
    method: str = ""
    velocity: Union[float, List[float]] = 0.0    # [Δx/Δt]
    density: float = 1.0                          # [dimensionless]
    relax_coeff: float = 0.1                      # [dimensionless]
    use_regularized: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Method → BCType classification table
# =============================================================================

# Maps config 'method' strings to (BCType, use_regularized)
#
# use_regularized meaning:
#   True  → f = f_eq + f^(1)(Π^neq)  (full regularized reconstruction)
#   False → method-specific handling   (equilibrium, bounce-back, neumann, etc.)
#
_METHOD_MAP: Dict[str, Tuple[BCType, bool]] = {
    # ── Velocity inlets ──
    'regularized_inlet':    (BCType.VELOCITY, True),    # f = f_eq + f^(1)
    'reg_inlet':            (BCType.VELOCITY, True),
    'equilibrium':          (BCType.VELOCITY, False),   # f = f_eq only
    'eq':                   (BCType.VELOCITY, False),
    
    # ── Pressure outlets (regularized) ──
    'regularized_outlet':   (BCType.PRESSURE, True),    # ρ relaxation + f^(1)
    'reg_outlet':           (BCType.PRESSURE, True),
    
    # ── Domain walls ──
    'regularized_wall':     (BCType.WALL, True),        # f = f_eq(ρ_ext, 0) + f^(1)
    'reg_wall':             (BCType.WALL, True),
    'bounce_back':          (BCType.WALL, False),       # f_i = f_ī (half-way BB)
    'hwbb':                 (BCType.WALL, False),
    
    # ── Zero-gradient outlet ──
    'neumann':              (BCType.NEUMANN, False),    # f[bdy] = f[int]
    'zero_gradient':        (BCType.NEUMANN, False),
    
    # ── Sponge (volume-based damping) ──
    'sponge':               (BCType.SPONGE, True),
    'sponge_layer':         (BCType.SPONGE, True),
    
    # ── Periodic (no BC needed) ──
    'periodic':             (BCType.PERIODIC, False),
    'none':                 (BCType.PERIODIC, False),
}

# Legacy 'type' field → default method mapping
_LEGACY_TYPE_MAP: Dict[str, str] = {
    'inlet':    'regularized_inlet',
    'outlet':   'regularized_outlet',
    'wall':     'regularized_wall',
    'open':     'regularized_outlet',
    'periodic': 'periodic',
}


def classify_method(method: str) -> Tuple[BCType, bool]:
    """Classify a config method string into (BCType, use_regularized).
    
    Args:
        method: Method string from config (case-insensitive)
        
    Returns:
        (BCType, use_regularized) tuple
        
    Raises:
        ValueError: If method is not recognized
    """
    key = method.lower().strip()
    if key in _METHOD_MAP:
        return _METHOD_MAP[key]
    raise ValueError(
        f"Unknown BC method: '{method}'. "
        f"Available: {sorted(set(k for k in _METHOD_MAP.keys()))}"
    )


# =============================================================================
# Config Parsing
# =============================================================================

def parse_face_config(bc_name: str, bc_dict: Dict[str, Any]) -> Optional[FaceConfig]:
    """Parse a single boundary config dictionary into FaceConfig.
    
    Handles both new-style and legacy config formats:
    
    New style:
        {'location': 'xmin', 'method': 'regularized_inlet', 'velocity': 0.1, 'rho': 1.0}
    
    Legacy style (any of these):
        {'type': 'inlet', 'location': 0, 'velocity': 0.1}     ← numeric location → use bc_name
        {'type': 'wall', 'method': 'bounce_back'}
        {'type': 'outlet', 'pressure': 1.0}                    ← 'pressure' key → density
    
    Args:
        bc_name: User-defined name (e.g. 'inlet', 'west', 'ymin')
        bc_dict: Config dictionary
        
    Returns:
        FaceConfig, or None if location cannot be determined
    """
    # =====================================================================
    # Step 1: Determine location
    # =====================================================================
    location_str = bc_dict.get('location')
    
    if isinstance(location_str, (int, float)):
        location_str = bc_name
    elif location_str is None:
        location_str = bc_name
    
    try:
        location = FaceLocation.from_string(str(location_str))
    except ValueError:
        try:
            location = FaceLocation.from_string(bc_name)
        except ValueError:
            print(f"  Warning: Cannot determine location for boundary '{bc_name}', skipping")
            return None
    
    # =====================================================================
    # Step 2: Determine method
    # =====================================================================
    method = bc_dict.get('method', '').lower().strip()
    bc_type_str = bc_dict.get('type', '').lower().strip()
    
    if not method:
        method = _LEGACY_TYPE_MAP.get(bc_type_str, '')
    elif method not in _METHOD_MAP:
        fallback = _LEGACY_TYPE_MAP.get(bc_type_str, '')
        if fallback:
            method = fallback
    
    if not method:
        print(f"  Warning: No 'method'/'type' specified for boundary '{bc_name}', skipping")
        return None
    
    # =====================================================================
    # Step 3: Classify method → (BCType, use_regularized)
    # =====================================================================
    try:
        bc_type, use_regularized = classify_method(method)
    except ValueError as e:
        print(f"  Warning: {e}")
        return None
    
    # Periodic → return minimal FaceConfig
    if bc_type == BCType.PERIODIC:
        return FaceConfig(
            location=location,
            bc_type=BCType.PERIODIC,
            method='periodic',
        )
    
    # =====================================================================
    # Step 4: Extract physical parameters
    # =====================================================================
    # Velocity  [Δx/Δt]
    velocity_raw = bc_dict.get('velocity', bc_dict.get('u_inf', 0.0))
    if isinstance(velocity_raw, (list, tuple)):
        velocity: Union[float, List[float]] = [float(v) for v in velocity_raw]
    else:
        velocity = float(velocity_raw)
    
    # Density  [dimensionless]
    density = float(bc_dict.get('rho', bc_dict.get('density',
                    bc_dict.get('rho_inf', bc_dict.get('pressure', 1.0)))))
    
    # Relaxation coefficient  [dimensionless]
    relax_coeff = float(bc_dict.get('k', bc_dict.get('relax_coeff', 0.1)))
    
    # Extra parameters for special methods
    extra: Dict[str, Any] = {}
    if method in ('sponge', 'sponge_layer'):
        extra['thickness'] = bc_dict.get('thickness', 20)
        extra['sigma_max'] = bc_dict.get('strength', bc_dict.get('sigma_max', 0.5))
    
    return FaceConfig(
        location=location,
        bc_type=bc_type,
        method=method,
        velocity=velocity,
        density=density,
        relax_coeff=relax_coeff,
        use_regularized=use_regularized,
        extra=extra,
    )


def parse_all_boundaries(boundaries_config: Dict[str, Dict[str, Any]]) -> List[FaceConfig]:
    """Parse all boundary configs into a list of FaceConfig.
    
    Args:
        boundaries_config: Full 'boundaries' section from config file
        
    Returns:
        List of FaceConfig (including periodic faces — NodeMap needs them
        to correctly identify which faces are NOT periodic)
    
    Raises:
        ValueError: If duplicate locations are found
        
    Example:
        >>> configs = parse_all_boundaries({
        ...     'inlet':  {'location': 'xmin', 'method': 'regularized_inlet', 'velocity': 0.1},
        ...     'outlet': {'location': 'xmax', 'method': 'regularized_outlet', 'rho': 1.0},
        ...     'ymin':   {'location': 'ymin', 'method': 'bounce_back'},
        ...     'ymax':   {'location': 'ymax', 'method': 'bounce_back'},
        ... })
        >>> [fc.bc_type for fc in configs]
        [VELOCITY, PRESSURE, WALL, WALL]
    """
    face_configs: List[FaceConfig] = []
    
    for bc_name, bc_dict in boundaries_config.items():
        fc = parse_face_config(bc_name, bc_dict)
        if fc is not None:
            face_configs.append(fc)
    
    # Validate: no duplicate locations
    seen: set = set()
    for fc in face_configs:
        if fc.location in seen:
            raise ValueError(f"Duplicate boundary location: {fc.location.value}")
        seen.add(fc.location)
    
    return face_configs