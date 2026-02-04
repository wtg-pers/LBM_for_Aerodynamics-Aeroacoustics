"""
Mass Conservation Utilities for LBM Solver

This module provides utility functions and classes for monitoring mass
conservation in LBM simulations. Supports both domain-wide and
control-volume based conservation checks with CSV logging.

Features:
    - Domain-wide mass tracking
    - Multiple Control Volume (CV) monitoring
    - CSV logging for post-processing
    - Configurable tolerance and warnings

Physical Background:
    Mass conservation (integral form):
        dM/dt = ṁ_inlet - ṁ_outlet
    
    At steady state:
        dM/dt ≈ 0  →  ṁ_inlet ≈ ṁ_outlet

Author: LBM Development Team
Date: 2026-02
"""

import os
import csv
from datetime import datetime
from typing import TYPE_CHECKING, Tuple, Optional, Dict, List, Any

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


# =============================================================================
# Control Volume Checker
# =============================================================================

class ControlVolumeChecker:
    """Control Volume based conservation checker
    
    Implements Reynolds Transport Theorem for mass:
        dM_CV/dt = -∮ ρ(u·n̂) dA = Σ(ṁ_in) - Σ(ṁ_out)
    
    Attributes:
        xp: Array module (numpy or cupy)
        name: Identifier for this CV
        bounds: CV bounds (x0, x1, y0, y1, z0, z1)  [lattice units]
        solid_mask: Optional solid mask within CV
    """
    
    def __init__(self, xp: 'ModuleType',
                 domain_shape: Tuple[int, int, int],
                 name: str = "cv",
                 bounds: Optional[Tuple[int, int, int, int, int, int]] = None,
                 solid_mask: Optional['npt.NDArray'] = None) -> None:
        """Initialize Control Volume checker
        
        Args:
            xp: Array module (numpy or cupy)
            domain_shape: Full domain shape (Nx, Ny, Nz)  [lattice units]
            name: Identifier for this CV (for logging)
            bounds: CV bounds (x0, x1, y0, y1, z0, z1)  [lattice units]
                   If None, uses entire domain.
            solid_mask: Solid nodes mask (True = solid)
        """
        self.xp = xp
        self.name = name
        self.Nx, self.Ny, self.Nz = domain_shape
        self.solid_mask = solid_mask
        
        # Set bounds (default: entire domain)
        if bounds is None:
            self.x0, self.x1 = 0, self.Nx - 1
            self.y0, self.y1 = 0, self.Ny - 1
            self.z0, self.z1 = 0, self.Nz - 1
        else:
            self.x0, self.x1, self.y0, self.y1, self.z0, self.z1 = bounds
        
        # Validate bounds
        self._validate_bounds()
        
        # For tracking
        self.M_initial: Optional[float] = None
        self.M_prev: Optional[float] = None
        self.step_prev: Optional[int] = None
        self.initialized: bool = False
    
    def _validate_bounds(self) -> None:
        """Validate CV bounds are within domain"""
        # Clamp to domain bounds (with warning if out of range)
        if self.x0 < 0:
            self.x0 = 0
        if self.x1 >= self.Nx:
            self.x1 = self.Nx - 1
        if self.y0 < 0:
            self.y0 = 0
        if self.y1 >= self.Ny:
            self.y1 = self.Ny - 1
        if self.z0 < 0:
            self.z0 = 0
        if self.z1 >= self.Nz:
            self.z1 = self.Nz - 1
    
    def get_bounds_str(self) -> str:
        """Get formatted bounds string"""
        return f"x=[{self.x0}:{self.x1}], y=[{self.y0}:{self.y1}], z=[{self.z0}:{self.z1}]"
    
    def initialize(self, rho: 'npt.NDArray', step: int = 0) -> float:
        """Initialize with starting state
        
        Args:
            rho: Initial density field  [dimensionless]
            step: Initial step number
            
        Returns:
            Initial mass in CV  [dimensionless]
        """
        self.M_initial = self._compute_cv_mass(rho)
        self.M_prev = self.M_initial
        self.step_prev = step
        self.initialized = True
        return self.M_initial
    
    def _compute_cv_mass(self, rho: 'npt.NDArray') -> float:
        """Compute total mass within control volume
        
        Args:
            rho: Density field (Nx, Ny, Nz)  [dimensionless]
            
        Returns:
            Total mass in CV  [dimensionless]
        """
        xp = self.xp
        
        # Extract CV region
        rho_cv = rho[self.x0:self.x1+1, self.y0:self.y1+1, self.z0:self.z1+1]
        
        # Exclude solid nodes if mask provided
        if self.solid_mask is not None:
            mask_cv = self.solid_mask[self.x0:self.x1+1, 
                                       self.y0:self.y1+1, 
                                       self.z0:self.z1+1]
            return float(xp.sum(rho_cv[~mask_cv]))
        else:
            return float(xp.sum(rho_cv))
    
    def check(self, rho: 'npt.NDArray', step: int = 0) -> Dict[str, Any]:
        """Check mass conservation in control volume
        
        Args:
            rho: Current density field  [dimensionless]
            step: Current step number
            
        Returns:
            Dictionary with conservation metrics:
                - name: CV identifier
                - M_current: Current mass
                - M_initial: Initial mass
                - mass_drift_percent: (M_current - M_initial) / M_initial * 100
                - dM: Mass change since last check
                - dM_per_step: Average mass change per step
        """
        if not self.initialized:
            self.initialize(rho, step)
            return {
                'name': self.name,
                'status': 'initialized',
                'M_initial': self.M_initial,
                'M_current': self.M_initial,
                'mass_drift_percent': 0.0,
                'dM': 0.0,
                'dM_per_step': 0.0,
            }
        
        # Current mass
        M_current = self._compute_cv_mass(rho)
        
        # Time interval
        interval = max(step - self.step_prev, 1)
        
        # Mass change
        dM = M_current - self.M_prev
        dM_per_step = dM / interval
        
        # Total drift from initial
        mass_drift_percent = (M_current - self.M_initial) / (self.M_initial + 1e-16) * 100
        
        # Update state
        self.M_prev = M_current
        self.step_prev = step
        
        return {
            'name': self.name,
            'status': 'checked',
            'M_current': M_current,
            'M_initial': self.M_initial,
            'mass_drift_percent': mass_drift_percent,
            'dM': dM,
            'interval': interval,
            'dM_per_step': dM_per_step,
        }
    
    def reset(self, rho: 'npt.NDArray', step: int = 0) -> None:
        """Reset reference to current state
        
        Args:
            rho: Current density field
            step: Current step number
        """
        self.initialize(rho, step)


# =============================================================================
# Conservation Manager (Config-based)
# =============================================================================

class ConservationManager:
    """Manages conservation checking based on config
    
    Handles both domain-wide and multiple CV-based conservation checks.
    Supports CSV logging for post-processing.
    
    Config Example:
        conservation = {
            "enabled": True,
            "check_interval": 0,  # 0 = use output_interval
            "verbose": 1,         # 0=silent, 1=summary, 2=detailed
            "log_to_csv": True,
            "tolerance": {
                "mass_drift_percent": 1.0,
                "warn_on_exceed": True,
            },
            "domain": {"enabled": True},
            "control_volumes": [
                {"name": "wake", "enabled": True, "bounds": {...}},
            ],
        }
    
    Example:
        >>> manager = ConservationManager(xp, domain_shape, config, solid_mask)
        >>> manager.initialize(rho, step=0)
        >>> # In time loop:
        >>> results = manager.check(rho, step, verbose=True)
    """
    
    def __init__(self, xp: 'ModuleType',
                 domain_shape: Tuple[int, int, int],
                 config: Dict[str, Any],
                 solid_mask: Optional['npt.NDArray'] = None,
                 csv_dir: str = './results/csv') -> None:
        """Initialize ConservationManager
        
        Args:
            xp: Array module (numpy or cupy)
            domain_shape: (Nx, Ny, Nz)  [lattice units]
            config: Conservation config dictionary
            solid_mask: Solid mask (True = solid)
            csv_dir: Directory for CSV output
        """
        self.xp = xp
        self.domain_shape = domain_shape
        self.Nx, self.Ny, self.Nz = domain_shape
        self.config = config
        self.solid_mask = solid_mask
        self.csv_dir = csv_dir
        
        # Parse config
        self.enabled = config.get('enabled', True)
        self.check_interval = config.get('check_interval', 0)  # 0 = use output_interval
        self.verbose = config.get('verbose', 1)
        self.log_to_csv = config.get('log_to_csv', True)
        
        # Tolerance settings
        tolerance_config = config.get('tolerance', {})
        self.mass_drift_tolerance = tolerance_config.get('mass_drift_percent', 1.0)
        self.warn_on_exceed = tolerance_config.get('warn_on_exceed', True)
        
        # Initialize checkers
        self.domain_checker: Optional[ControlVolumeChecker] = None
        self.cv_checkers: List[ControlVolumeChecker] = []
        
        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        self.csv_path: Optional[str] = None
        
        if self.enabled:
            self._setup_checkers()
    
    def _setup_checkers(self) -> None:
        """Setup domain and CV checkers based on config"""
        # Domain-wide checker
        domain_config = self.config.get('domain', {})
        if domain_config.get('enabled', True):
            self.domain_checker = ControlVolumeChecker(
                self.xp,
                self.domain_shape,
                name='domain',
                bounds=None,  # Full domain
                solid_mask=self.solid_mask
            )
        
        # Control Volume checkers
        cv_configs = self.config.get('control_volumes', [])
        for cv_config in cv_configs:
            if not cv_config.get('enabled', True):
                continue
            
            name = cv_config.get('name', f'cv_{len(self.cv_checkers)}')
            bounds_config = cv_config.get('bounds', {})
            
            # Parse bounds
            bounds = self._parse_bounds(bounds_config)
            
            cv = ControlVolumeChecker(
                self.xp,
                self.domain_shape,
                name=name,
                bounds=bounds,
                solid_mask=self.solid_mask
            )
            self.cv_checkers.append(cv)
    
    def _parse_bounds(self, bounds_config: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
        """Parse bounds from config
        
        Args:
            bounds_config: Dictionary with xmin, xmax, ymin, ymax, zmin, zmax
                          Values should be int (Python variables evaluated at config load)
        
        Returns:
            Tuple of (x0, x1, y0, y1, z0, z1)
        """
        x0 = int(bounds_config.get('xmin', 0))
        x1 = int(bounds_config.get('xmax', self.Nx - 1))
        y0 = int(bounds_config.get('ymin', 0))
        y1 = int(bounds_config.get('ymax', self.Ny - 1))
        z0 = int(bounds_config.get('zmin', 0))
        z1 = int(bounds_config.get('zmax', self.Nz - 1))
        
        return (x0, x1, y0, y1, z0, z1)
    
    def _setup_csv_logger(self) -> None:
        """Setup CSV file for logging"""
        if not self.log_to_csv:
            return
        
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'conservation_{timestamp}.csv'
        self.csv_path = os.path.join(self.csv_dir, filename)
        
        # Build header
        headers = ['step', 'time']
        
        if self.domain_checker is not None:
            headers.extend(['domain_mass', 'domain_drift_percent'])
        
        for cv in self.cv_checkers:
            headers.extend([f'{cv.name}_mass', f'{cv.name}_drift_percent'])
        
        # Open file and write header
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(headers)
        self.csv_file.flush()
        
        print(f"    CSV log: {self.csv_path}")
    
    def initialize(self, rho: 'npt.NDArray', step: int = 0) -> None:
        """Initialize all checkers
        
        Args:
            rho: Initial density field
            step: Initial step number
        """
        if not self.enabled:
            return
        
        print("\n  Initializing Conservation Checks...")
        
        if self.domain_checker is not None:
            M0 = self.domain_checker.initialize(rho, step)
            print(f"    Domain: M0 = {M0:.6f}")
        
        for cv in self.cv_checkers:
            M0 = cv.initialize(rho, step)
            print(f"    {cv.name}: M0 = {M0:.6f}, {cv.get_bounds_str()}")
        
        # Setup CSV (after checkers are ready)
        if self.log_to_csv:
            self._setup_csv_logger()
    
    def check(self, rho: 'npt.NDArray', step: int, 
              time: Optional[float] = None,
              verbose: Optional[int] = None) -> Dict[str, Any]:
        """Check conservation for all registered CVs
        
        Args:
            rho: Current density field
            step: Current step number
            time: Physical time (optional, defaults to step)
            verbose: Override verbose level (None = use config)
        
        Returns:
            Dictionary with all results:
                - domain: Domain-wide result (if enabled)
                - cvs: List of CV results
                - any_warning: True if any CV exceeds tolerance
        """
        if not self.enabled:
            return {'enabled': False}
        
        if time is None:
            time = float(step)
        
        if verbose is None:
            verbose = self.verbose
        
        results = {
            'step': step,
            'time': time,
            'domain': None,
            'cvs': [],
            'any_warning': False,
        }
        
        # Domain check
        if self.domain_checker is not None:
            domain_result = self.domain_checker.check(rho, step)
            results['domain'] = domain_result
            
            if abs(domain_result['mass_drift_percent']) > self.mass_drift_tolerance:
                results['any_warning'] = True
        
        # CV checks
        for cv in self.cv_checkers:
            cv_result = cv.check(rho, step)
            results['cvs'].append(cv_result)
            
            if abs(cv_result['mass_drift_percent']) > self.mass_drift_tolerance:
                results['any_warning'] = True
        
        # Print results
        if verbose > 0:
            self._print_results(results, verbose)
        
        # Log to CSV
        if self.log_to_csv and self.csv_writer is not None:
            self._log_to_csv(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any], verbose: int) -> None:
        """Print conservation check results"""
        print("\n" + "="*60)
        print(f" Conservation Check (step {results['step']})")
        print("="*60)
        
        # Domain result
        if results['domain'] is not None:
            d = results['domain']
            drift = d['mass_drift_percent']
            status = self._get_status_symbol(drift)
            
            print(f"\n  [Domain]")
            print(f"    Mass: {d['M_current']:.6f} (initial: {d['M_initial']:.6f})")
            print(f"    Drift: {drift:+.4f}%  {status}")
            
            if verbose >= 2:
                print(f"    dM/step: {d['dM_per_step']:+.4e}")
        
        # CV results
        for cv_result in results['cvs']:
            drift = cv_result['mass_drift_percent']
            status = self._get_status_symbol(drift)
            
            print(f"\n  [{cv_result['name']}]")
            print(f"    Mass: {cv_result['M_current']:.6f} (initial: {cv_result['M_initial']:.6f})")
            print(f"    Drift: {drift:+.4f}%  {status}")
            
            if verbose >= 2:
                print(f"    dM/step: {cv_result['dM_per_step']:+.4e}")
        
        # Warning
        if results['any_warning'] and self.warn_on_exceed:
            print(f"\n  ⚠ WARNING: Mass drift exceeds tolerance ({self.mass_drift_tolerance}%)")
        
        print("="*60)
    
    def _get_status_symbol(self, drift_percent: float) -> str:
        """Get status symbol based on drift magnitude"""
        abs_drift = abs(drift_percent)
        
        if abs_drift < 0.01:
            return "✓ Excellent"
        elif abs_drift < 0.1:
            return "✓ Good"
        elif abs_drift < self.mass_drift_tolerance:
            return "✓ OK"
        else:
            return "✗ Exceeded"
    
    def _log_to_csv(self, results: Dict[str, Any]) -> None:
        """Log results to CSV file"""
        row = [results['step'], results['time']]
        
        # Domain
        if results['domain'] is not None:
            d = results['domain']
            row.extend([d['M_current'], d['mass_drift_percent']])
        
        # CVs
        for cv_result in results['cvs']:
            row.extend([cv_result['M_current'], cv_result['mass_drift_percent']])
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def get_info(self) -> str:
        """Get information string about the manager"""
        lines = ["Conservation Manager:"]
        lines.append(f"  Enabled: {self.enabled}")
        
        if not self.enabled:
            return "\n".join(lines)
        
        lines.append(f"  Tolerance: {self.mass_drift_tolerance}%")
        lines.append(f"  CSV logging: {self.log_to_csv}")
        
        if self.domain_checker is not None:
            lines.append(f"  Domain check: enabled")
        
        if self.cv_checkers:
            lines.append(f"  Control Volumes ({len(self.cv_checkers)}):")
            for cv in self.cv_checkers:
                lines.append(f"    - {cv.name}: {cv.get_bounds_str()}")
        
        return "\n".join(lines)
    
    def close(self) -> None:
        """Close CSV file"""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
    
    def __del__(self):
        """Destructor to ensure CSV file is closed"""
        self.close()


# =============================================================================
# Utility Functions
# =============================================================================

def compute_total_mass(xp: 'ModuleType',
                       rho: 'npt.NDArray',
                       solid_mask: Optional['npt.NDArray'] = None) -> float:
    """Compute total mass in the fluid domain
    
    Args:
        xp: Array module (numpy or cupy)
        rho: Density field, shape (Nx, Ny, Nz)  [dimensionless]
        solid_mask: Boolean mask where True = solid (excluded from sum)
        
    Returns:
        Total mass in fluid domain  [dimensionless]
    """
    if solid_mask is not None:
        return float(xp.sum(rho[~solid_mask]))
    else:
        return float(xp.sum(rho))


def compute_mass_flux(xp: 'ModuleType', 
                      rho: 'npt.NDArray', 
                      u: 'npt.NDArray', 
                      face: str = 'xmin') -> float:
    """Compute mass flux through a boundary face
    
    Mass flux: ṁ = Σ ρ[face] · u_normal[face]  [mass/Δt]
    
    Args:
        xp: Array module (numpy or cupy)
        rho: Density field, shape (Nx, Ny, Nz)  [dimensionless]
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        face: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'
        
    Returns:
        Mass flux through the face  [mass/Δt]
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
        return float(xp.sum(rho[0, :, :] * u[0, 0, :, :]))
    elif face == 'xmax':
        return float(xp.sum(rho[-1, :, :] * u[0, -1, :, :]))
    elif face == 'ymin':
        return float(xp.sum(rho[:, 0, :] * u[1, :, 0, :]))
    elif face == 'ymax':
        return float(xp.sum(rho[:, -1, :] * u[1, :, -1, :]))
    elif face == 'zmin':
        return float(xp.sum(rho[:, :, 0] * u[2, :, :, 0]))
    elif face == 'zmax':
        return float(xp.sum(rho[:, :, -1] * u[2, :, :, -1]))
    else:
        raise ValueError(f"Unknown face: '{face}'")


def verify_mass_flux_balance(xp: 'ModuleType', 
                              rho: 'npt.NDArray', 
                              u: 'npt.NDArray',
                              inlet_face: str = 'xmin',
                              outlet_face: str = 'xmax',
                              verbose: bool = True) -> Tuple[float, float, float]:
    """Verify mass flux balance between inlet and outlet
    
    Args:
        xp: Array module
        rho: Density field  [dimensionless]
        u: Velocity field  [Δx/Δt]
        inlet_face: Inlet face name
        outlet_face: Outlet face name
        verbose: Print results
        
    Returns:
        Tuple of (flux_inlet, flux_outlet, imbalance_percent)
    """
    flux_inlet = compute_mass_flux(xp, rho, u, inlet_face)
    flux_outlet = compute_mass_flux(xp, rho, u, outlet_face)
    
    imbalance = (flux_inlet - flux_outlet) / (abs(flux_inlet) + 1e-10) * 100

    if verbose:
        print(f"  Inlet flux ({inlet_face}):   {flux_inlet:.6f}")
        print(f"  Outlet flux ({outlet_face}): {flux_outlet:.6f}")
        print(f"  Imbalance: {imbalance:.4f}%")
    
    return flux_inlet, flux_outlet, imbalance