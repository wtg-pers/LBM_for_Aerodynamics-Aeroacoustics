"""
Mass Flux Utilities for LBM Solver

This module provides utility functions for computing and verifying
mass flux through domain boundaries. Essential for validating
open boundary condition implementations.

Physical Background:
    Mass flux through a surface:
        ṁ = ∫∫ ρ · (u · n̂) dA
    
    Discretized for LBM:
        ṁ = Σ ρ[face] · u_normal[face]
    
    Mass conservation (integral form):
        dM/dt = ṁ_inlet - ṁ_outlet
    
    At steady state:
        dM/dt ≈ 0  →  ṁ_inlet ≈ ṁ_outlet

Author: LBM Development Team
Date: 2026-01
"""

from typing import TYPE_CHECKING, Tuple, Optional, Dict, List

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


# =============================================================================
# Control Volume Conservation Checker (Recommended)
# =============================================================================

class ControlVolumeChecker:
    """Control Volume based conservation checker
    
    Implements Reynolds Transport Theorem for mass and momentum:
    
        dM_CV/dt = -∮ ρ(u·n̂) dA = Σ(ṁ_in) - Σ(ṁ_out)
        
    where the surface integral is over ALL faces of the control volume.
    
    This is more accurate than simple inlet/outlet comparison because:
    1. Accounts for ALL boundary fluxes (including far-field)
    2. Can check conservation in sub-regions (e.g., around obstacle)
    3. Properly handles open boundaries in all directions
    
    Attributes:
        xp: Array module (numpy or cupy)
        bounds: Control volume bounds (x_min, x_max, y_min, y_max, z_min, z_max)
        solid_mask: Optional solid mask within CV
        
    Example:
        >>> # Check entire domain
        >>> cv = ControlVolumeChecker(xp, domain_shape)
        >>> result = cv.check(rho, u, step)
        >>>
        >>> # Check region around cylinder
        >>> cv_local = ControlVolumeChecker(xp, domain_shape,
        ...     bounds=(50, 150, 30, 70, 0, Nz))  # Box around obstacle
        >>> result = cv_local.check(rho, u, step)
    """
    
    def __init__(self, xp: 'ModuleType',
                 domain_shape: Tuple[int, int, int],
                 bounds: Optional[Tuple[int, int, int, int, int, int]] = None,
                 solid_mask: Optional['npt.NDArray'] = None,
                 exclude_domain_boundary: bool = False) -> None:
        """Initialize Control Volume checker
        
        Args:
            xp: Array module (numpy or cupy)
            domain_shape: Full domain shape (Nx, Ny, Nz)
            bounds: CV bounds (x0, x1, y0, y1, z0, z1). 
                   If None, uses entire domain.
            solid_mask: Solid nodes mask (True = solid)
            exclude_domain_boundary: If True, don't count flux at domain edges
                                    (useful for internal CV checks)
        """
        self.xp = xp
        self.Nx, self.Ny, self.Nz = domain_shape
        self.solid_mask = solid_mask
        self.exclude_domain_boundary = exclude_domain_boundary
        
        # Set bounds (default: entire domain)
        if bounds is None:
            self.x0, self.x1 = 0, self.Nx - 1
            self.y0, self.y1 = 0, self.Ny - 1
            self.z0, self.z1 = 0, self.Nz - 1
        else:
            self.x0, self.x1, self.y0, self.y1, self.z0, self.z1 = bounds
        
        # For tracking
        self.M_prev = None
        self.M_initial = None  # For total drift calculation
        self.step_prev = None
        self.initialized = False
    
    def initialize(self, rho: 'npt.NDArray', step: int = 0) -> float:
        """Initialize with starting state
        
        Args:
            rho: Initial density field
            step: Initial step number
            
        Returns:
            Initial mass in CV
        """
        self.M_prev = self._compute_cv_mass(rho)
        self.M_initial = self.M_prev  # Store for total drift calculation
        self.step_prev = step
        self.initialized = True
        return self.M_prev
    
    def _compute_cv_mass(self, rho: 'npt.NDArray') -> float:
        """Compute total mass within control volume"""
        xp = self.xp
        
        # Extract CV region
        rho_cv = rho[self.x0:self.x1+1, self.y0:self.y1+1, self.z0:self.z1+1]
        
        # Exclude solid nodes if mask provided
        if self.solid_mask is not None:
            mask_cv = self.solid_mask[self.x0:self.x1+1, self.y0:self.y1+1, self.z0:self.z1+1]
            return float(xp.sum(rho_cv[~mask_cv]))
        else:
            return float(xp.sum(rho_cv))
    
    def _compute_face_flux(self, rho: 'npt.NDArray', u: 'npt.NDArray',
                           face: str) -> float:
        """Compute mass flux through one face of the CV
        
        Convention: Positive flux = leaving CV (outward normal)
        
        Args:
            rho: Density field
            u: Velocity field (3, Nx, Ny, Nz)
            face: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'
            
        Returns:
            Mass flux (positive = outflow from CV)
        """
        xp = self.xp
        x0, x1 = self.x0, self.x1
        y0, y1 = self.y0, self.y1
        z0, z1 = self.z0, self.z1
        
        if face == 'xmin':
            # x = x0 face, outward normal = -x direction
            # Outflow = -ρ * u_x (negative u_x leaves through xmin)
            if self.exclude_domain_boundary and x0 == 0:
                return 0.0
            rho_face = rho[x0, y0:y1+1, z0:z1+1]
            u_face = u[0, x0, y0:y1+1, z0:z1+1]
            return float(-xp.sum(rho_face * u_face))  # Negative sign for outward normal
            
        elif face == 'xmax':
            # x = x1 face, outward normal = +x direction
            if self.exclude_domain_boundary and x1 == self.Nx - 1:
                return 0.0
            rho_face = rho[x1, y0:y1+1, z0:z1+1]
            u_face = u[0, x1, y0:y1+1, z0:z1+1]
            return float(xp.sum(rho_face * u_face))
            
        elif face == 'ymin':
            # y = y0 face, outward normal = -y direction
            if self.exclude_domain_boundary and y0 == 0:
                return 0.0
            rho_face = rho[x0:x1+1, y0, z0:z1+1]
            u_face = u[1, x0:x1+1, y0, z0:z1+1]
            return float(-xp.sum(rho_face * u_face))
            
        elif face == 'ymax':
            # y = y1 face, outward normal = +y direction
            if self.exclude_domain_boundary and y1 == self.Ny - 1:
                return 0.0
            rho_face = rho[x0:x1+1, y1, z0:z1+1]
            u_face = u[1, x0:x1+1, y1, z0:z1+1]
            return float(xp.sum(rho_face * u_face))
            
        elif face == 'zmin':
            # z = z0 face, outward normal = -z direction
            if self.exclude_domain_boundary and z0 == 0:
                return 0.0
            rho_face = rho[x0:x1+1, y0:y1+1, z0]
            u_face = u[2, x0:x1+1, y0:y1+1, z0]
            return float(-xp.sum(rho_face * u_face))
            
        elif face == 'zmax':
            # z = z1 face, outward normal = +z direction
            if self.exclude_domain_boundary and z1 == self.Nz - 1:
                return 0.0
            rho_face = rho[x0:x1+1, y0:y1+1, z1]
            u_face = u[2, x0:x1+1, y0:y1+1, z1]
            return float(xp.sum(rho_face * u_face))
        
        else:
            raise ValueError(f"Unknown face: {face}")
    
    def compute_all_fluxes(self, rho: 'npt.NDArray', 
                           u: 'npt.NDArray') -> Dict[str, float]:
        """Compute mass flux through all 6 faces of CV
        
        Returns:
            Dictionary with fluxes for each face (positive = outflow)
        """
        faces = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        fluxes = {}
        
        for face in faces:
            fluxes[face] = self._compute_face_flux(rho, u, face)
        
        # Net outflow (positive = mass leaving CV)
        fluxes['net_outflow'] = sum(fluxes[f] for f in faces)
        
        return fluxes
    
    def check(self, rho: 'npt.NDArray', u: 'npt.NDArray' = None,
              step: int = 0, verbose: bool = False) -> Dict:
        """Check mass conservation in control volume
        
        Tracks actual mass change in the CV over time.
        This is the TRUE conservation check based on Σf_i (all distributions).
        
        Note:
            The velocity field 'u' is optional and only used for diagnostic
            macroscopic flux calculation, which may not match actual LBM 
            transport near boundaries.
        
        Args:
            rho: Current density field (= Σf_i at each node)
            u: Current velocity field (optional, for diagnostic only)
            step: Current step number
            verbose: Print detailed results
            
        Returns:
            Dictionary with conservation metrics
        """
        if not self.initialized:
            self.initialize(rho, step)
            return {'status': 'initialized', 'M_initial': self.M_prev}
        
        # Current CV mass (TRUE conservation metric)
        M_current = self._compute_cv_mass(rho)
        
        # Time interval
        interval = max(step - self.step_prev, 1)
        
        # Mass change (PRIMARY METRIC)
        dM = M_current - self.M_prev
        dM_per_step = dM / interval
        
        # Relative mass change (compared to initial mass)
        mass_drift_percent = (M_current - self.M_initial) / self.M_initial * 100
        
        # Update state
        self.M_prev = M_current
        self.step_prev = step
        
        result = {
            'M_current': M_current,
            'M_initial': self.M_initial,
            'dM': dM,
            'interval': interval,
            'dM_per_step': dM_per_step,
            'mass_drift_percent': mass_drift_percent,
        }
        
        # Optional: Compute macroscopic fluxes (diagnostic only)
        if u is not None:
            fluxes = self.compute_all_fluxes(rho, u)
            result['fluxes'] = fluxes
            result['net_outflow'] = fluxes['net_outflow']
        
        if verbose:
            self._print_result(result)
        
        return result
    
    def _print_result(self, result: Dict) -> None:
        """Print formatted conservation check result"""
        print("\n" + "="*60)
        print(" Control Volume Conservation Check")
        print("="*60)
        print(f" CV bounds: x=[{self.x0}, {self.x1}], y=[{self.y0}, {self.y1}], z=[{self.z0}, {self.z1}]")
        print("-"*60)
        
        # Primary metric: Mass tracking
        print(" Mass Tracking:")
        print(f"   Current mass:  {result['M_current']:.6f}")
        print(f"   Initial mass:  {result['M_initial']:.6f}")
        drift = result['mass_drift_percent']
        print(f"   Total drift:   {drift:+.6f}%", end="")
        
        if abs(drift) < 0.01:
            print("  ✓ Excellent")
        elif abs(drift) < 0.1:
            print("  ✓ Good")
        elif abs(drift) < 1.0:
            print("  ⚠ Acceptable")
        else:
            print("  ✗ Check BCs")
        
        # Rate of change (convergence indicator)
        print(f"   dM (interval): {result['dM']:+.4e}")
        print(f"   dM/step:       {result['dM_per_step']:+.4e}", end="")
        if abs(result['dM_per_step']) < 1e-6:
            print("  (steady state)")
        elif abs(result['dM_per_step']) < 1e-4:
            print("  (converging)")
        else:
            print("  (transient)")
        print("="*60)


def create_obstacle_cv(xp: 'ModuleType',
                       domain_shape: Tuple[int, int, int],
                       obstacle_center: Tuple[float, float],
                       obstacle_radius: float,
                       margin: float = 2.0,
                       solid_mask: Optional['npt.NDArray'] = None) -> 'ControlVolumeChecker':
    """Create a control volume around an obstacle
    
    Useful for checking local conservation near the obstacle.
    The CV extends from (center - radius - margin) to (center + radius + margin)
    in x and y, and spans the full z-direction.
    
    Args:
        xp: Array module
        domain_shape: (Nx, Ny, Nz)
        obstacle_center: (x, y) center of obstacle
        obstacle_radius: Radius of obstacle  [lattice units]
        margin: Extra margin around obstacle  [lattice units]
        solid_mask: Solid mask
        
    Returns:
        ControlVolumeChecker instance for the obstacle region
        
    Example:
        >>> # Check conservation in a box around the cylinder
        >>> cv = create_obstacle_cv(xp, (420, 180, 60), (84, 90), 10, margin=20)
        >>> cv.initialize(rho, step=0)
        >>> result = cv.check(rho, u, step, verbose=True)
    """
    Nx, Ny, Nz = domain_shape
    cx, cy = obstacle_center
    r = obstacle_radius
    
    # CV bounds with margin
    x0 = max(0, int(cx - r - margin))
    x1 = min(Nx - 1, int(cx + r + margin))
    y0 = max(0, int(cy - r - margin))
    y1 = min(Ny - 1, int(cy + r + margin))
    z0 = 0
    z1 = Nz - 1
    
    return ControlVolumeChecker(
        xp, domain_shape,
        bounds=(x0, x1, y0, y1, z0, z1),
        solid_mask=solid_mask,
        exclude_domain_boundary=True  # Internal CV, don't count domain boundary flux
    )


def create_domain_cv(xp: 'ModuleType',
                     domain_shape: Tuple[int, int, int],
                     solid_mask: Optional['npt.NDArray'] = None) -> 'ControlVolumeChecker':
    """Create a control volume for the entire domain
    
    This is the standard conservation check that accounts for ALL 
    boundary fluxes (inlet, outlet, and all far-field boundaries).
    
    Args:
        xp: Array module
        domain_shape: (Nx, Ny, Nz)
        solid_mask: Solid mask
        
    Returns:
        ControlVolumeChecker for entire domain
        
    Example:
        >>> cv = create_domain_cv(xp, (420, 180, 60), solid_mask)
        >>> cv.initialize(rho, step=0)
        >>> # In time loop:
        >>> result = cv.check(rho, u, step, verbose=True)
    """
    return ControlVolumeChecker(
        xp, domain_shape,
        bounds=None,  # Full domain
        solid_mask=solid_mask,
        exclude_domain_boundary=False  # Count all boundary fluxes
    )


# =============================================================================
# Total Mass Computation
# =============================================================================

def compute_total_mass(xp: 'ModuleType',
                       rho: 'npt.NDArray',
                       solid_mask: Optional['npt.NDArray'] = None) -> float:
    """Compute total mass in the fluid domain
    
    Total mass:
        M = Σ ρ[fluid nodes]  [mass]
    
    Args:
        xp: Array module (numpy or cupy)
        rho: Density field, shape (Nx, Ny, Nz)  [dimensionless]
        solid_mask: Boolean mask where True = solid (excluded from sum)
                   If None, all nodes are considered fluid.
        
    Returns:
        Total mass in fluid domain  [dimensionless, sum of ρ]
        
    Example:
        >>> mass = compute_total_mass(xp, rho, solid_mask)
    """
    if solid_mask is not None:
        # Only sum fluid nodes (where solid_mask is False)
        fluid_rho = rho[~solid_mask]
        return float(xp.sum(fluid_rho))
    else:
        return float(xp.sum(rho))


def compute_mass_change_rate(xp: 'ModuleType',
                              rho_current: 'npt.NDArray',
                              rho_previous: 'npt.NDArray',
                              solid_mask: Optional['npt.NDArray'] = None,
                              dt: float = 1.0) -> float:
    """Compute rate of mass change in the domain
    
    Mass change rate:
        dM/dt = (M_current - M_previous) / Δt  [mass/Δt]
    
    Args:
        xp: Array module
        rho_current: Current density field  [dimensionless]
        rho_previous: Previous density field  [dimensionless]
        solid_mask: Boolean solid mask (optional)
        dt: Time step (default 1.0 for lattice units)
        
    Returns:
        Rate of mass change  [mass/Δt]
        Positive = mass accumulating in domain
        Negative = mass leaving domain
    """
    M_current = compute_total_mass(xp, rho_current, solid_mask)
    M_previous = compute_total_mass(xp, rho_previous, solid_mask)
    
    return (M_current - M_previous) / dt


class MassConservationTracker:
    """Tracks mass conservation over time
    
    Stores previous density to compute dM/dt and verifies
    the conservation equation: dM/dt = ṁ_in - ṁ_out
    
    Attributes:
        xp: Array module
        solid_mask: Solid mask for excluding solid nodes
        rho_prev: Previous step density (stored internally)
        M_prev: Previous total mass
        step_prev: Previous step number (for interval calculation)
        
    Example:
        >>> tracker = MassConservationTracker(xp, solid_mask)
        >>> tracker.initialize(rho_initial, step=0)
        >>> 
        >>> # In time loop (every output_interval steps):
        >>> result = tracker.check(rho, u, step=current_step)
        >>> print(f"dM/step = {result['dM_per_step']:.4e}")
    """
    
    def __init__(self, xp: 'ModuleType', 
                 solid_mask: Optional['npt.NDArray'] = None) -> None:
        """Initialize tracker
        
        Args:
            xp: Array module
            solid_mask: Boolean mask (True = solid)
        """
        self.xp = xp
        self.solid_mask = solid_mask
        self.rho_prev = None
        self.M_prev = None
        self.step_prev = None
        self.initialized = False
    
    def initialize(self, rho: 'npt.NDArray', step: int = 0) -> float:
        """Initialize with starting density
        
        Args:
            rho: Initial density field
            step: Initial step number
            
        Returns:
            Initial total mass
        """
        self.rho_prev = self.xp.copy(rho)
        self.M_prev = compute_total_mass(self.xp, rho, self.solid_mask)
        self.step_prev = step
        self.initialized = True
        return self.M_prev
    
    def check(self, rho: 'npt.NDArray', 
              u: 'npt.NDArray',
              step: int,
              inlet: str = 'xmin',
              outlet: str = 'xmax') -> dict:
        """Check mass conservation for current step
        
        Verifies: dM/dt + ṁ_out - ṁ_in ≈ 0
        
        Args:
            rho: Current density field
            u: Current velocity field
            step: Current step number
            inlet: Inlet face name
            outlet: Outlet face name
            
        Returns:
            Dictionary with:
                - M_current: Current total mass
                - M_previous: Previous total mass
                - dM: Mass change (M_current - M_previous) over interval
                - interval: Number of steps since last check
                - dM_per_step: Average mass change per step
                - flux_in: Inlet mass flux (instantaneous)
                - flux_out: Outlet mass flux (instantaneous)
                - net_flux: flux_in - flux_out (per step)
                - expected_dM: net_flux * interval (expected total change)
                - residual: dM - expected_dM (should be ≈ 0)
                - relative_error: |residual| / |expected_dM|
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        
        xp = self.xp
        
        # Current mass
        M_current = compute_total_mass(xp, rho, self.solid_mask)
        
        # Interval (number of steps since last check)
        interval = step - self.step_prev
        if interval <= 0:
            interval = 1  # Avoid division by zero
        
        # Mass change over interval
        dM = M_current - self.M_prev
        dM_per_step = dM / interval
        
        # Instantaneous fluxes (per step)
        flux_in = compute_mass_flux(xp, rho, u, inlet)
        flux_out = compute_mass_flux(xp, rho, u, outlet)
        net_flux = flux_in - flux_out  # Per step
        
        # Expected total dM over interval (assuming constant flux)
        expected_dM = net_flux * interval
        
        # Conservation residual
        residual = dM - expected_dM
        
        # Relative error (compared to expected change)
        rel_error = abs(residual) / (abs(expected_dM) + 1e-10)
        
        # Update stored values
        self.rho_prev = xp.copy(rho)
        self.M_prev = M_current
        self.step_prev = step
        
        return {
            'M_current': M_current,
            'M_previous': self.M_prev,
            'dM': dM,
            'interval': interval,
            'dM_per_step': dM_per_step,
            'flux_in': flux_in,
            'flux_out': flux_out,
            'net_flux': net_flux,
            'expected_dM': expected_dM,
            'residual': residual,
            'relative_error': rel_error
        }
    
    def get_summary(self, result: dict, verbose: bool = True) -> str:
        """Generate summary string from check result
        
        Args:
            result: Dictionary from check()
            verbose: Include detailed output
            
        Returns:
            Formatted summary string
        """
        lines = []
        
        if verbose:
            lines.append("Mass Conservation Check:")
            lines.append(f"  Total mass:      M = {result['M_current']:.6f}")
            lines.append(f"  Interval:        {result['interval']} steps")
            lines.append(f"  Mass change:     dM = {result['dM']:+.6e} (over interval)")
            lines.append(f"  Per-step change: dM/step = {result['dM_per_step']:+.6e}")
            lines.append(f"  Inlet flux:      ṁ_in = {result['flux_in']:.6f} (per step)")
            lines.append(f"  Outlet flux:     ṁ_out = {result['flux_out']:.6f} (per step)")
            lines.append(f"  Net flux:        ṁ_in - ṁ_out = {result['net_flux']:+.6e} (per step)")
            lines.append(f"  Expected dM:     net_flux × interval = {result['expected_dM']:+.6e}")
            lines.append(f"  Residual:        dM - expected = {result['residual']:+.6e}")
            lines.append(f"  Relative error:  {result['relative_error']:.2e}")
            
            if result['relative_error'] < 1e-6:
                lines.append("  Status: ✓ Excellent (near machine precision)")
            elif result['relative_error'] < 1e-3:
                lines.append("  Status: ✓ Very good")
            elif result['relative_error'] < 1e-1:
                lines.append("  Status: ✓ Good")
            else:
                lines.append("  Status: ⚠ Check boundary conditions")
        else:
            lines.append(f"dM/step={result['dM_per_step']:+.2e}, "
                        f"net_flux={result['net_flux']:+.2e}, "
                        f"err={result['relative_error']:.1e}")
        
        return "\n".join(lines)


# =============================================================================
# Mass Flux Computation
# =============================================================================


def compute_mass_flux(xp: 'ModuleType', 
                      rho: 'npt.NDArray', 
                      u: 'npt.NDArray', 
                      face: str = 'xmin') -> float:
    """Compute mass flux through a boundary face
    
    Calculates the integral of ρ·u_n over the specified face,
    where u_n is the velocity component normal to the face.
    
    Mass flux formula:
        ṁ = Σ ρ[face] · u_normal[face]  [mass/Δt]
    
    Args:
        xp: Array module (numpy or cupy)
        rho: Density field, shape (Nx, Ny, Nz)  [dimensionless]
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        face: Boundary face name. Options:
              Coordinate-based (preferred):
              - 'xmin' (x=0, inlet typically)
              - 'xmax' (x=Nx-1, outlet typically)
              - 'ymin' (y=0)
              - 'ymax' (y=Ny-1)
              - 'zmin' (z=0)
              - 'zmax' (z=Nz-1)
              Legacy (also supported):
              - 'west'='xmin', 'east'='xmax'
              - 'south'='ymin', 'north'='ymax'
              - 'bottom'='zmin', 'top'='zmax'
        
    Returns:
        Total mass flux through the face  [mass/Δt]
        Positive = flow in positive coordinate direction
        
    Examples:
        >>> flux_in = compute_mass_flux(xp, rho, u, 'xmin')
        >>> flux_out = compute_mass_flux(xp, rho, u, 'xmax')
        >>> print(f"Inlet: {flux_in:.4f}, Outlet: {flux_out:.4f}")
    
    Note:
        For a properly conserving simulation at steady state,
        flux_inlet should approximately equal flux_outlet.
    """
    face = face.lower()
    
    # Map legacy names to coordinate-based
    legacy_map = {
        'west': 'xmin', 'east': 'xmax',
        'south': 'ymin', 'north': 'ymax',
        'bottom': 'zmin', 'top': 'zmax'
    }
    face = legacy_map.get(face, face)
    
    if face == 'xmin':
        # x = 0 face, normal direction is +x (u[0])
        return float(xp.sum(rho[0, :, :] * u[0, 0, :, :]))
    
    elif face == 'xmax':
        # x = Nx-1 face, normal direction is +x (u[0])
        return float(xp.sum(rho[-1, :, :] * u[0, -1, :, :]))
    
    elif face == 'ymin':
        # y = 0 face, normal direction is +y (u[1])
        return float(xp.sum(rho[:, 0, :] * u[1, :, 0, :]))
    
    elif face == 'ymax':
        # y = Ny-1 face, normal direction is +y (u[1])
        return float(xp.sum(rho[:, -1, :] * u[1, :, -1, :]))
    
    elif face == 'zmin':
        # z = 0 face, normal direction is +z (u[2])
        return float(xp.sum(rho[:, :, 0] * u[2, :, :, 0]))
    
    elif face == 'zmax':
        # z = Nz-1 face, normal direction is +z (u[2])
        return float(xp.sum(rho[:, :, -1] * u[2, :, :, -1]))
    
    else:
        raise ValueError(
            f"Unknown face: '{face}'. "
            f"Valid options: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'"
        )


def verify_mass_flux_balance(xp: 'ModuleType', 
                              rho: 'npt.NDArray', 
                              u: 'npt.NDArray',
                              inlet_face: str = 'xmin',
                              outlet_face: str = 'xmax',
                              verbose: bool = True) -> Tuple[float, float, float]:
    """Verify mass flux balance between inlet and outlet
    
    Computes mass flux at inlet and outlet, and calculates the
    relative imbalance. Useful for validating boundary conditions.
    
    At steady state, mass conservation requires:
        ṁ_inlet ≈ ṁ_outlet
        
    Imbalance = (ṁ_in - ṁ_out) / |ṁ_in| × 100%
    
    Args:
        xp: Array module (numpy or cupy)
        rho: Density field, shape (Nx, Ny, Nz)  [dimensionless]
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        inlet_face: Inlet face name (default: 'west')
        outlet_face: Outlet face name (default: 'east')
        verbose: Print results if True
        
    Returns:
        Tuple of (flux_inlet, flux_outlet, imbalance_percent)
        
    Examples:
        >>> flux_in, flux_out, imbalance = verify_mass_flux_balance(xp, rho, u)
        Inlet flux:  144.0000
        Outlet flux: 143.9856
        Imbalance:   0.0100%
    """
    flux_inlet = compute_mass_flux(xp, rho, u, inlet_face)
    flux_outlet = compute_mass_flux(xp, rho, u, outlet_face)
    
    # Relative imbalance (avoid division by zero)
    imbalance = (flux_inlet - flux_outlet) / (abs(flux_inlet) + 1e-10) * 100

    if verbose:
        print(f"  Inlet flux ({inlet_face}):   {flux_inlet:.6f}")
        print(f"  Outlet flux ({outlet_face}): {flux_outlet:.6f}")
        print(f"  Imbalance:   {imbalance:.4f}%")
        
        # Quality assessment
        if abs(imbalance) < 0.1:
            print(f"  Status: ✓ Excellent mass conservation")
        elif abs(imbalance) < 1.0:
            print(f"  Status: ✓ Good mass conservation")
        elif abs(imbalance) < 5.0:
            print(f"  Status: ⚠ Acceptable, but check BCs")
        else:
            print(f"  Status: ✗ Poor mass conservation, review boundary conditions")
    
    return flux_inlet, flux_outlet, imbalance


def compute_volumetric_flow_rate(xp: 'ModuleType',
                                  u: 'npt.NDArray',
                                  face: str = 'xmin') -> float:
    """Compute volumetric flow rate through a face
    
    Volumetric flow rate:
        Q = ∫∫ u_n dA = Σ u_normal[face]  [Δx³/Δt]
    
    Unlike mass flux, this doesn't weight by density.
    Useful when density variations are small.
    
    Args:
        xp: Array module
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        face: Boundary face name (xmin, xmax, ymin, ymax, zmin, zmax)
        
    Returns:
        Volumetric flow rate  [Δx³/Δt]
    """
    face = face.lower()
    
    # Map legacy names
    legacy_map = {
        'west': 'xmin', 'east': 'xmax',
        'south': 'ymin', 'north': 'ymax',
        'bottom': 'zmin', 'top': 'zmax'
    }
    face = legacy_map.get(face, face)
    
    if face == 'xmin':
        return float(xp.sum(u[0, 0, :, :]))
    elif face == 'xmax':
        return float(xp.sum(u[0, -1, :, :]))
    elif face == 'ymin':
        return float(xp.sum(u[1, :, 0, :]))
    elif face == 'ymax':
        return float(xp.sum(u[1, :, -1, :]))
    elif face == 'zmin':
        return float(xp.sum(u[2, :, :, 0]))
    elif face == 'zmax':
        return float(xp.sum(u[2, :, :, -1]))
    else:
        raise ValueError(f"Unknown face: '{face}'")


def compute_average_velocity(xp: 'ModuleType',
                              u: 'npt.NDArray',
                              face: str = 'xmin') -> float:
    """Compute average velocity magnitude normal to a face
    
    Average normal velocity:
        <u_n> = (Σ u_normal[face]) / N_face  [Δx/Δt]
    
    Args:
        xp: Array module
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        face: Boundary face name (xmin, xmax, ymin, ymax, zmin, zmax)
        
    Returns:
        Average normal velocity  [Δx/Δt]
    """
    face = face.lower()
    
    # Map legacy names
    legacy_map = {
        'west': 'xmin', 'east': 'xmax',
        'south': 'ymin', 'north': 'ymax',
        'bottom': 'zmin', 'top': 'zmax'
    }
    face = legacy_map.get(face, face)
    
    if face in ['xmin', 'xmax']:
        idx = 0 if face == 'xmin' else -1
        face_velocity = u[0, idx, :, :]
    elif face in ['ymin', 'ymax']:
        idx = 0 if face == 'ymin' else -1
        face_velocity = u[1, :, idx, :]
    elif face in ['zmin', 'zmax']:
        idx = 0 if face == 'zmin' else -1
        face_velocity = u[2, :, :, idx]
    else:
        raise ValueError(f"Unknown face: '{face}'")
    
    return float(xp.mean(face_velocity))