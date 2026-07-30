"""
Mass Conservation and Flux Utilities for LBM

This module provides tools for monitoring mass conservation
in LBM simulations, including domain-wide and control volume checks.

Supports both 2D and 3D domains.

Physical Principle:
==================
For incompressible flows, total mass should be conserved:
    dM/dt = 0 (closed system)
    dM/dt = ṁ_in - ṁ_out (open system)

Mass is computed as:
    M = Σ ρ(x) * ΔV

where ΔV = 1 in lattice units.

Mass drift is defined as:
    drift% = (M_current - M_initial) / M_initial * 100

Author: LBM Development Team
Date: 2026-02
"""

import os
import csv
from typing import TYPE_CHECKING, Optional, Dict, List, Any, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class ControlVolume:
    """Control Volume for local mass conservation monitoring
    
    Tracks mass within a specified rectangular region of the domain.
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType',
                 name: str,
                 domain_shape: tuple,
                 bounds: Optional[Tuple] = None,
                 solid_mask: Optional['npt.NDArray'] = None) -> None:
        """Initialize control volume
        
        Args:
            xp: Array module (numpy or cupy)
            name: CV identifier
            domain_shape: (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
            bounds: CV bounds tuple. For 2D: (x0, x1, y0, y1)
                   For 3D: (x0, x1, y0, y1, z0, z1)
            solid_mask: Solid nodes mask (True = solid)
        """
        self.xp = xp
        self.name = name
        self.shape = domain_shape
        self.dim = len(domain_shape)
        self.solid_mask = solid_mask
        
        # Set bounds
        if self.dim == 2:
            Nx, Ny = domain_shape
            if bounds is None:
                self.x0, self.x1 = 0, Nx - 1
                self.y0, self.y1 = 0, Ny - 1
            else:
                self.x0, self.x1, self.y0, self.y1 = bounds
        else:
            Nx, Ny, Nz = domain_shape
            if bounds is None:
                self.x0, self.x1 = 0, Nx - 1
                self.y0, self.y1 = 0, Ny - 1
                self.z0, self.z1 = 0, Nz - 1
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
        if self.dim == 2:
            Nx, Ny = self.shape
            self.x0 = max(0, self.x0)
            self.x1 = min(Nx - 1, self.x1)
            self.y0 = max(0, self.y0)
            self.y1 = min(Ny - 1, self.y1)
        else:
            Nx, Ny, Nz = self.shape
            self.x0 = max(0, self.x0)
            self.x1 = min(Nx - 1, self.x1)
            self.y0 = max(0, self.y0)
            self.y1 = min(Ny - 1, self.y1)
            self.z0 = max(0, self.z0)
            self.z1 = min(Nz - 1, self.z1)
    
    def get_bounds_str(self) -> str:
        """Get formatted bounds string"""
        if self.dim == 2:
            return f"x=[{self.x0}:{self.x1}], y=[{self.y0}:{self.y1}]"
        else:
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
            rho: Density field  [dimensionless]
            
        Returns:
            Total mass in CV  [dimensionless]
        """
        xp = self.xp
        
        # Extract CV region
        if self.dim == 2:
            rho_cv = rho[self.x0:self.x1+1, self.y0:self.y1+1]
            if self.solid_mask is not None:
                mask_cv = self.solid_mask[self.x0:self.x1+1, self.y0:self.y1+1]
                return float(xp.sum(rho_cv[~mask_cv]))
        else:
            rho_cv = rho[self.x0:self.x1+1, self.y0:self.y1+1, self.z0:self.z1+1]
            if self.solid_mask is not None:
                mask_cv = self.solid_mask[self.x0:self.x1+1, 
                                          self.y0:self.y1+1, 
                                          self.z0:self.z1+1]
                return float(xp.sum(rho_cv[~mask_cv]))
        
        return float(xp.sum(rho_cv))
    
    def compute_mass(self, rho: 'npt.NDArray') -> float:
        """Public alias of the CV mass computation (scalar-API entry: the
        MPI path sums owned-slab partials and Allreduces them instead)."""
        return self._compute_cv_mass(rho)

    def initialize_from_mass(self, M: float, step: int = 0) -> float:
        """Scalar-input variant of initialize() — same state transitions."""
        self.M_initial = M
        self.M_prev = M
        self.step_prev = step
        self.initialized = True
        return self.M_initial

    def check_from_mass(self, M_current: float,
                        step: int = 0) -> Dict[str, Any]:
        """Scalar-input variant of check(): all drift bookkeeping from a
        precomputed CV mass. Single source for both paths — check() computes
        the mass then delegates here."""
        if not self.initialized:
            self.initialize_from_mass(M_current, step)
            return {
                'name': self.name,
                'M_current': self.M_initial,
                'M_initial': self.M_initial,
                'mass_drift_percent': 0.0,
                'dM': 0.0,
                'dM_per_step': 0.0
            }

        # Compute metrics
        mass_drift_percent = (M_current - self.M_initial) / self.M_initial * 100 \
                             if abs(self.M_initial) > 1e-10 else 0.0

        dM = M_current - self.M_prev
        steps_elapsed = step - self.step_prev if self.step_prev is not None else 1
        dM_per_step = dM / max(steps_elapsed, 1)

        # Update tracking
        self.M_prev = M_current
        self.step_prev = step

        return {
            'name': self.name,
            'M_current': M_current,
            'M_initial': self.M_initial,
            'mass_drift_percent': mass_drift_percent,
            'dM': dM,
            'dM_per_step': dM_per_step
        }

    def check(self, rho: 'npt.NDArray', step: int = 0) -> Dict[str, Any]:
        """Check mass conservation in control volume

        Args:
            rho: Current density field  [dimensionless]
            step: Current step number

        Returns:
            Dictionary with conservation metrics
        """
        return self.check_from_mass(self._compute_cv_mass(rho), step)


class ConservationManager:
    """Manager for domain-wide and control volume conservation checks
    
    Coordinates multiple control volumes and provides unified
    logging and reporting interface.
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType',
                 domain_shape: tuple,
                 solid_mask: Optional['npt.NDArray'] = None,
                 config: Optional[Dict[str, Any]] = None,
                 csv_dir: str = './results/csv') -> None:
        """Initialize conservation manager
        
        Args:
            xp: Array module (numpy or cupy)
            domain_shape: (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
            solid_mask: Solid nodes mask
            config: Conservation configuration dictionary
            csv_dir: Directory for CSV output
        """
        self.xp = xp
        self.domain_shape = domain_shape
        self.dim = len(domain_shape)
        self.solid_mask = solid_mask
        self.csv_dir = csv_dir
        
        # Parse config
        if config is None:
            config = {}
        
        self.enabled = config.get('enabled', True)
        self.check_interval = config.get('check_interval', 1000)
        self.verbose = config.get('verbose', 1)
        self.log_to_csv = config.get('log_to_csv', True)
        
        tolerance_config = config.get('tolerance', {})
        self.mass_drift_threshold = tolerance_config.get('mass_drift_percent', 1.0)
        self.warn_on_exceed = tolerance_config.get('warn_on_exceed', True)
        
        # Create domain-wide CV
        domain_config = config.get('domain', {})
        self.domain_cv_enabled = domain_config.get('enabled', True)
        
        if self.domain_cv_enabled:
            self.domain_cv = ControlVolume(
                xp, 'domain', domain_shape,
                bounds=None,
                solid_mask=solid_mask
            )
        else:
            self.domain_cv = None
        
        # Create user-defined CVs
        self.control_volumes: List[ControlVolume] = []
        cv_configs = config.get('control_volumes', [])
        
        for cv_config in cv_configs:
            if not cv_config.get('enabled', True):
                continue
            
            name = cv_config.get('name', f'cv_{len(self.control_volumes)}')
            bounds_dict = cv_config.get('bounds', {})
            
            if self.dim == 2:
                bounds = (
                    bounds_dict.get('xmin', 0),
                    bounds_dict.get('xmax', domain_shape[0] - 1),
                    bounds_dict.get('ymin', 0),
                    bounds_dict.get('ymax', domain_shape[1] - 1),
                )
            else:
                bounds = (
                    bounds_dict.get('xmin', 0),
                    bounds_dict.get('xmax', domain_shape[0] - 1),
                    bounds_dict.get('ymin', 0),
                    bounds_dict.get('ymax', domain_shape[1] - 1),
                    bounds_dict.get('zmin', 0),
                    bounds_dict.get('zmax', domain_shape[2] - 1),
                )
            
            cv = ControlVolume(xp, name, domain_shape, bounds, solid_mask)
            self.control_volumes.append(cv)
        
        # History storage
        self.history: List[Dict[str, Any]] = []
        
        # CSV file
        self._csv_file = None
        self._csv_writer = None
    
    def initialize(self, rho: 'npt.NDArray', step: int = 0) -> None:
        """Initialize all control volumes
        
        Args:
            rho: Initial density field
            step: Initial step number
        """
        if not self.enabled:
            return
        
        if self.domain_cv is not None:
            M0 = self.domain_cv.initialize(rho, step)
            print(f"  Domain initial mass: M0 = {M0:.6f}")
        
        for cv in self.control_volumes:
            M0 = cv.initialize(rho, step)
            print(f"  CV '{cv.name}': M0 = {M0:.6f}, {cv.get_bounds_str()}")
        
        # Setup CSV logging (csv_dir=None: non-IO rank — state advances
        # identically but nothing is written)
        if self.log_to_csv and self.csv_dir is not None:
            os.makedirs(self.csv_dir, exist_ok=True)
            csv_path = os.path.join(self.csv_dir, 'mass_conservation.csv')

            # Header: step, domain_drift, [cv_name_drift, ...]
            fieldnames = ['step', 'domain_drift']
            for cv in self.control_volumes:
                fieldnames.append(f'{cv.name}_drift')

            if step > 0 and os.path.exists(csv_path):
                # Restart: keep rows with step < current start_step
                kept_lines = []
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if row and int(row[0]) < step:
                            kept_lines.append(row)

                self._csv_file = open(csv_path, 'w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(fieldnames)
                for row in kept_lines:
                    self._csv_writer.writerow(row)
                self._csv_file.flush()
                print(f"  CSV log: {csv_path} (appending from step {step}, "
                      f"{len(kept_lines)} previous rows kept)")
            else:
                self._csv_file = open(csv_path, 'w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(fieldnames)
                self._csv_file.flush()
                print(f"  CSV log: {csv_path}")
    
    def cv_items(self) -> list:
        """(name, ControlVolume) pairs, domain first — the iteration order
        of check(); the MPI path uses it to build the same-ordered vector
        of slab-partial masses for one SUM Allreduce."""
        items = []
        if self.domain_cv is not None:
            items.append(('domain', self.domain_cv))
        for cv in self.control_volumes:
            items.append((cv.name, cv))
        return items

    def check(self, rho: 'npt.NDArray', step: int,
              verbose: bool = None) -> Dict[str, Any]:
        """Perform conservation check

        Args:
            rho: Current density field
            step: Current step number
            verbose: Override verbosity setting

        Returns:
            Dictionary with results for each CV
        """
        if not self.enabled:
            return {}
        masses = {name: cv.compute_mass(rho) for name, cv in self.cv_items()}
        return self.check_from_masses(masses, step, verbose=verbose)

    def check_from_masses(self, masses: Dict[str, float], step: int,
                          verbose: bool = None) -> Dict[str, Any]:
        """Scalar-input variant of check(): drift bookkeeping + CSV + print
        from precomputed CV masses (keys = cv_items() names). Single source
        for both paths."""
        if not self.enabled:
            return {}

        if verbose is None:
            verbose = self.verbose > 0

        results = {}
        csv_row = [step]
        drift_lines = []

        # Check domain-wide
        if self.domain_cv is not None:
            domain_result = self.domain_cv.check_from_mass(
                masses['domain'], step)
            results['domain'] = domain_result

            drift = domain_result['mass_drift_percent']
            csv_row.append(drift)
            status = self._get_status(drift)
            drift_lines.append(f"  domain: {drift:+.4f}% {status}")

        # Check user-defined CVs
        for cv in self.control_volumes:
            cv_result = cv.check_from_mass(masses[cv.name], step)
            results[cv.name] = cv_result

            drift = cv_result['mass_drift_percent']
            csv_row.append(drift)
            status = self._get_status(drift)
            drift_lines.append(f"  {cv.name}: {drift:+.4f}% {status}")
        
        # Terminal output (블록 형태)
        if verbose and drift_lines:
            print(f"\n  {'─'*40}")
            print(f"  Conservation Check (step {step})")
            print(f"  {'─'*40}")
            for line in drift_lines:
                print(line)
            print(f"  {'─'*40}")
        
        # Write to CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow(csv_row)
            self._csv_file.flush()
        
        # Store in history
        self.history.append({'step': step, 'results': results})
        
        return results
    
    def _get_status(self, drift_percent: float) -> str:
        """Get status symbol based on drift magnitude"""
        abs_drift = abs(drift_percent)
        
        if abs_drift < 0.01:
            return "✓"
        elif abs_drift < 0.1:
            return "✓"
        elif abs_drift < self.mass_drift_threshold:
            return "✓"
        else:
            return "✗"
    
    def close(self) -> None:
        """Close CSV file"""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
    
    def get_info(self) -> str:
        """Return information string"""
        dim_str = "2D" if self.dim == 2 else "3D"
        info = [f"ConservationManager ({dim_str}):"]
        info.append(f"  Enabled: {self.enabled}")
        info.append(f"  Check interval: {self.check_interval}")
        
        if self.domain_cv is not None:
            info.append(f"  Domain CV: enabled")
        
        if self.control_volumes:
            info.append(f"  User CVs: {len(self.control_volumes)}")
            for cv in self.control_volumes:
                info.append(f"    - {cv.name}: {cv.get_bounds_str()}")
        
        return '\n'.join(info)