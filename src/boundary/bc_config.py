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
        WALL(no-slip) > SPONGE > VELOCITY > PRESSURE > NEUMANN > PERIODIC
        (Slip WALL applied last: zeroes only face-normal component of u.)
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
            'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'

        Args:
            name: Location string (case-insensitive)

        Returns:
            FaceLocation enum value

        Raises:
            ValueError: If name is not recognized (the legacy compass
            aliases west/east/south/north/bottom/top were removed — no
            active config used them)
        """
        key = name.lower().strip()

        for member in cls:
            if member.value == key:
                return member

        valid = [m.value for m in cls]
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
    'equilibrium':          (BCType.VELOCITY, False),   # f = f_eq only
    'eq':                   (BCType.VELOCITY, False),

    # ── Domain walls ──
    'bounce_back':          (BCType.WALL, False),       # f_i = f_ī (half-way BB)
    'hwbb':                 (BCType.WALL, False),

    # ── Free-slip / symmetry walls (specular reflection) ──
    'slip':                 (BCType.WALL, False),       # f_i = f_{σ(i)}, σ flips c_axis
    'symmetry':             (BCType.WALL, False),
    'free_slip':            (BCType.WALL, False),
    
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

# Domain-BC policy (feedback_domain_bc_policy): eq + neumann (+ sponge)
# only. The regularized family was accepted here for years while its
# reconstruction was never properly finished — rejected with guidance
# until a real implementation lands as its own track.
_REMOVED_METHODS: Dict[str, str] = {
    k: ("regularized BCs are not implemented properly and were removed "
        "from the accepted set (policy: eq + neumann + sponge). A proper "
        "regularized implementation is planned as a separate track — "
        "until then use 'eq' (+ 'sponge') or 'neumann'.")
    for k in ('regularized_inlet', 'reg_inlet', 'regularized_outlet',
              'reg_outlet', 'regularized_wall', 'reg_wall')
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
    if key in _REMOVED_METHODS:
        raise ValueError(f"BC method '{method}': {_REMOVED_METHODS[key]}")
    raise ValueError(
        f"Unknown BC method: '{method}'. "
        f"Available: {sorted(set(k for k in _METHOD_MAP.keys()))}"
    )


# =============================================================================
# Config Parsing
# =============================================================================

def parse_face_config(bc_name: str, bc_dict: Dict[str, Any]) -> FaceConfig:
    """Parse a single boundary config dictionary into FaceConfig.

    Expected format:
        {'location': 'xmin', 'method': 'equilibrium', 'velocity': [0.1, 0.0]}
        {'location': 'xmax', 'method': 'sponge', 'velocity': [0.1, 0],
         'density': 1.0, 'thickness': 20, 'strength': 0.5}

    Args:
        bc_name: User-defined name (e.g. 'inlet', 'west', 'ymin')
        bc_dict: Config dictionary

    Returns:
        FaceConfig — always valid

    Raises:
        ValueError: If location or method is missing or unrecognized, or
                    any key is unknown (fail-fast: no silent drops).
    """
    # =====================================================================
    # Step 0: Key whitelist. A mistyped or removed-alias key ('u_inf',
    # 'rho_inf', 'pressure', 'type') used to be silently ignored with the
    # parameter's default applied in its place — e.g. 'u_inf' bypassed the
    # m/s → LU conversion entirely.
    # =====================================================================
    _KNOWN_KEYS = {'location', 'method', 'velocity', 'density', 'rho',
                   'k', 'relax_coeff', 'thickness', 'strength', 'sigma_max'}
    _unknown = set(bc_dict) - _KNOWN_KEYS
    if _unknown:
        raise ValueError(
            f"Boundary '{bc_name}': unknown key(s) {sorted(_unknown)}. "
            f"Known keys: {sorted(_KNOWN_KEYS)}")

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
    except ValueError as e:
        # No bc_name retry: an explicit-but-mistyped 'location' used to be
        # silently overridden by the dict key.
        raise ValueError(
            f"Boundary '{bc_name}': cannot determine location "
            f"from 'location'={location_str!r}. {e}"
        ) from None
    
    # =====================================================================
    # Step 2: Determine method
    # =====================================================================
    method = bc_dict.get('method', '').lower().strip()
    if not method:
        raise ValueError(
            f"Boundary '{bc_name}': no 'method' specified. "
            f"Got bc_dict={bc_dict}"
        )

    # =====================================================================
    # Step 3: Classify method → (BCType, use_regularized) — fail-fast
    # =====================================================================
    try:
        bc_type, use_regularized = classify_method(method)
    except ValueError as e:
        raise ValueError(f"Boundary '{bc_name}': {e}") from None
    
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
    velocity_raw = bc_dict.get('velocity', 0.0)
    if isinstance(velocity_raw, (list, tuple)):
        velocity: Union[float, List[float]] = [float(v) for v in velocity_raw]
    else:
        velocity = float(velocity_raw)

    # Density  [dimensionless]
    density = float(bc_dict.get('rho', bc_dict.get('density', 1.0)))
    
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
        face_configs.append(parse_face_config(bc_name, bc_dict))
    
    # Validate: no duplicate locations
    seen: set = set()
    for fc in face_configs:
        if fc.location in seen:
            raise ValueError(f"Duplicate boundary location: {fc.location.value}")
        seen.add(fc.location)

    return face_configs


def derive_periodic_axes(
    boundaries_config: Dict[str, Dict[str, Any]],
    dim: int,
) -> Tuple[int, ...]:
    """Axes on which the domain is genuinely a torus.

    An axis is periodic iff NEITHER of its faces carries a non-periodic BC:
    a face absent from the config has no FaceConfig at all, so the pull
    streaming's wrap is what acts there (that IS the periodic BC in this
    code base), and a face given method='periodic'/'none' says so
    explicitly. Any other method (wall, velocity, pressure, neumann,
    sponge) claims the face, and the wrap underneath it is an
    implementation detail the BC overwrites — not a seam contract.

    This is the single source for the `periodic_axes` the boundary-link
    enumeration (q_fraction.compute_boundary_links / compute_needs_bounce)
    must be given on L0: real periodic boxes keep their wrap links (the
    momentum accounting closes through the seam), everything else stops
    inventing neighbours across it.
    """
    axis_faces = {
        0: (FaceLocation.XMIN, FaceLocation.XMAX),
        1: (FaceLocation.YMIN, FaceLocation.YMAX),
        2: (FaceLocation.ZMIN, FaceLocation.ZMAX),
    }
    claimed = {
        fc.location
        for fc in parse_all_boundaries(boundaries_config or {})
        if fc.bc_type != BCType.PERIODIC
    }
    return tuple(
        ax for ax in range(dim)
        if not (axis_faces[ax][0] in claimed or axis_faces[ax][1] in claimed)
    )