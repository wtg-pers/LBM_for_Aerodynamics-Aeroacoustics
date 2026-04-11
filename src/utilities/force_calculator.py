"""
Force Calculation Module using Momentum Exchange Method (MEM)

This module implements the Momentum Exchange Method for computing
hydrodynamic forces on solid obstacles in LBM simulations.

Supports both 2D and 3D domains.

Physical Principle:
==================
The momentum exchange method calculates forces by summing the momentum
transferred at fluid-solid boundary links during bounce-back.

For half-way bounce-back, at each boundary link:
    F_link = 2 * c_i * f_i^post(x_fluid)

where:
    c_i = lattice velocity vector  [Δx/Δt]
    f_i^post = post-collision distribution  [dimensionless]
    x_fluid = fluid node adjacent to solid

Total force on obstacle:
    F_total = Σ F_link (sum over all boundary links)

Force Coefficients:
==================
2D:
    Drag coefficient: C_D = F_x / (0.5 * ρ * U² * D)
    Lift coefficient: C_L = F_y / (0.5 * ρ * U² * D)
    Reference area: D (per unit span)

3D:
    Drag coefficient: C_D = F_x / (0.5 * ρ * U² * A)
    Lift coefficient: C_L = F_y / (0.5 * ρ * U² * A)
    Reference area: A = D * L_z

References:
==========
- Ladd, J. Fluid Mech. 271, 1994 (original MEM)
- Mei et al., Phys. Fluids 14, 2002 (improved MEM)
- Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 11

Author: LBM Development Team
Date: 2026-02
"""

import os
import csv
from datetime import datetime
from typing import TYPE_CHECKING, Tuple, Optional, Dict, List, Any, Union

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class MomentumExchangeForce:
    """Momentum Exchange Method for Force Calculation
    
    Computes hydrodynamic forces on solid obstacles using the momentum
    exchange at fluid-solid boundary links.
    
    Supports both 2D and 3D domains.
    
    Algorithm:
    ----------
    1. Identify boundary links: fluid nodes with solid neighbors
    2. For each boundary link (fluid node x_f, direction i → solid):
       F_link = 2 * c_i * f_i^post(x_f)
    3. Sum all F_link to get total force
    
    Example:
        >>> force_calc = MomentumExchangeForce(xp, lattice, solid_mask)
        >>> Fx, Fy = force_calc.compute(f_post)  # 2D
        >>> Fx, Fy, Fz = force_calc.compute(f_post)  # 3D
    """
    
    def __init__(self, xp: 'ModuleType', 
                 lattice: 'Lattice',
                 solid_mask: 'npt.NDArray',
                 wall_bc: Optional['HalfwayBounceBack'] = None) -> None:
        """Initialize momentum exchange force calculator
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            solid_mask: Boolean solid mask, True = solid
            wall_bc: Optional HalfwayBounceBack instance (reuses precomputed links)
        """
        self.xp = xp
        self.lattice = lattice
        self.dim = lattice.dim
        self.Q = lattice.Q
        self.c = xp.asarray(lattice.c, dtype=xp.float64)  # (dim, Q)
        self.opp = xp.asarray(lattice.opp)
        self.solid_mask = xp.asarray(solid_mask, dtype=bool)
        self.shape = solid_mask.shape
        
        # Reuse boundary link info from wall BC if provided
        if wall_bc is not None and hasattr(wall_bc, 'needs_bounce'):
            self.needs_bounce = wall_bc.needs_bounce
        else:
            self._precompute_boundary_links()
        
        # Count boundary links for info
        self.n_boundary_links = int(xp.sum(self.needs_bounce))

        # Try CUDA kernel for 3D
        self._cuda_kernel = None
        if self.dim == 3 and xp.__name__ == 'cupy':
            try:
                from src.kernels.mem_force_d3q27 import MEMForceKernelD3Q27
                self._cuda_kernel = MEMForceKernelD3Q27()
            except Exception:
                pass

        # Storage for time history
        self.force_history: List[Dict[str, Any]] = []
    
    def _precompute_boundary_links(self) -> None:
        """Precompute fluid-solid boundary links
        
        For each direction i, identify fluid nodes whose neighbor
        in direction i is solid. These are the boundary links.
        
        needs_bounce[i, x, y(, z)] = True means:
            - (x, y(, z)) is a fluid node
            - neighbor in direction i is a solid node
            - Distribution f_i at this node bounces back
        """
        xp = self.xp
        c = self.c.astype(xp.int32)
        solid = self.solid_mask
        
        if self.dim == 2:
            self._precompute_boundary_links_2d(c, solid)
        else:
            self._precompute_boundary_links_3d(c, solid)
    
    def _precompute_boundary_links_2d(self, c: 'npt.NDArray', solid: 'npt.NDArray') -> None:
        """Precompute boundary links for 2D domain"""
        xp = self.xp
        Nx, Ny = self.shape
        self.needs_bounce = xp.zeros((self.Q,) + self.shape, dtype=bool)
        
        for i in range(self.Q):
            if i == 0:  # Rest direction
                continue
            
            cx, cy = int(c[0, i]), int(c[1, i])
            
            # Shift solid mask by -c_i
            shifted_solid = xp.roll(
                xp.roll(solid, -cx, axis=0),
                -cy, axis=1)
            
            # Boundary link: fluid node where neighbor is solid
            self.needs_bounce[i] = (~solid) & shifted_solid
    
    def _precompute_boundary_links_3d(self, c: 'npt.NDArray', solid: 'npt.NDArray') -> None:
        """Precompute boundary links for 3D domain"""
        xp = self.xp
        Nx, Ny, Nz = self.shape
        self.needs_bounce = xp.zeros((self.Q,) + self.shape, dtype=bool)
        
        for i in range(self.Q):
            if i == 0:
                continue
            
            cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
            
            shifted_solid = xp.roll(
                xp.roll(
                    xp.roll(solid, -cx, axis=0),
                    -cy, axis=1),
                -cz, axis=2)
            
            self.needs_bounce[i] = (~solid) & shifted_solid
    
    def compute(self, f_post: 'npt.NDArray') -> Tuple:
        """Compute total force on solid using momentum exchange

        Mathematical Formula:
        ---------------------
        F = Σ_{boundary links} 2 * c_i * f_i^post(x_fluid)

        Args:
            f_post: Post-collision distribution, shape (Q, Nx, Ny) or (Q, Nx, Ny, Nz)
                   [dimensionless] - MUST be post-collision, pre-streaming!

        Returns:
            (Fx, Fy) for 2D or (Fx, Fy, Fz) for 3D  [lattice units]
        """
        # Try CUDA kernel for 3D
        if self.dim == 3 and self._cuda_kernel is not None:
            N = 1
            for d in f_post.shape[1:]:
                N *= d
            return self._cuda_kernel.compute(f_post, self.needs_bounce, N)

        # Fallback: Python loop (2D or no CUDA)
        xp = self.xp
        c = self.c

        forces = []
        for d in range(self.dim):
            F_d = 0.0
            for i in range(1, self.Q):
                f_boundary = f_post[i] * self.needs_bounce[i]
                F_d += 2.0 * float(c[d, i]) * float(xp.sum(f_boundary))
            forces.append(F_d)

        return tuple(forces)
    
    def get_coefficients(self, forces: Tuple, 
                         rho_ref: float = 1.0,
                         u_ref: float = 0.1,
                         char_length: float = 20.0,
                         span_length: float = 1.0) -> Dict[str, float]:
        """Convert forces to dimensionless coefficients
        
        Args:
            forces: (Fx, Fy) or (Fx, Fy, Fz)  [lattice units]
            rho_ref: Reference density  [dimensionless]
            u_ref: Reference velocity  [Δx/Δt]
            char_length: Characteristic length D  [Δx]
            span_length: Span length (for 3D, use Lz)  [Δx]
        
        Returns:
            Dictionary with Cd, Cl (and Cz for 3D)
        """
        # Reference area: D for 2D, D*Lz for 3D
        if self.dim == 2:
            A_ref = char_length  # Per unit span
        else:
            A_ref = char_length * span_length
        
        # Dynamic pressure: q = 0.5 * ρ * U²
        q = 0.5 * rho_ref * u_ref**2
        
        # Force coefficients
        Cd = forces[0] / (q * A_ref) if abs(q * A_ref) > 1e-10 else 0.0
        Cl = forces[1] / (q * A_ref) if abs(q * A_ref) > 1e-10 else 0.0
        
        result = {'Cd': Cd, 'Cl': Cl}
        
        if self.dim == 3 and len(forces) > 2:
            Cz = forces[2] / (q * A_ref) if abs(q * A_ref) > 1e-10 else 0.0
            result['Cz'] = Cz
        
        return result
    
    def get_info(self) -> str:
        """Return information about the force calculator"""
        dim_str = "2D" if self.dim == 2 else "3D"
        return (f"MomentumExchangeForce ({dim_str}):\n"
                f"  Boundary links: {self.n_boundary_links:,}")


class ForceManager:
    """High-level manager for force calculation and logging
    
    Combines MomentumExchangeForce with logging, statistics,
    and coefficient computation.
    
    Supports both 2D and 3D domains.
    
    Example:
        >>> force_mgr = ForceManager(xp, lattice, solid_mask, config)
        >>> force_mgr.initialize()
        >>> # In time loop:
        >>> force_mgr.compute_and_log(step, f_post)
        >>> # After simulation:
        >>> stats = force_mgr.get_final_statistics()
    """
    
    def __init__(self, xp: 'ModuleType',
                 lattice: 'Lattice',
                 solid_mask: 'npt.NDArray',
                 config: Dict[str, Any],
                 wall_bc: Optional['HalfwayBounceBack'] = None,
                 csv_dir: str = './results/csv') -> None:
        """Initialize force manager
        
        Args:
            xp: Array module
            lattice: Lattice model
            solid_mask: Solid mask
            config: Force configuration dictionary
            wall_bc: Optional wall BC (reuses boundary link info)
            csv_dir: Directory for CSV output
        """
        self.xp = xp
        self.config = config
        self.csv_dir = csv_dir
        self.dim = lattice.dim
        
        # Parse config
        self.enabled = config.get('enabled', True)
        self.interval = config.get('interval', 1)
        self.start_step = config.get('start_step', 0)
        
        ref_config = config.get('reference', {})
        self.rho_ref = ref_config.get('rho', 1.0)
        self.u_ref = ref_config.get('velocity', 0.1)
        self.char_length = ref_config.get('char_length', 20.0)
        self.span_length = ref_config.get('span_length', 1.0)
        
        log_config = config.get('log', {})
        self.log_enabled = log_config.get('enabled', True)
        self.log_filename = log_config.get('filename', 'force_history')
        
        # Create force calculator
        self.force_calc = MomentumExchangeForce(xp, lattice, solid_mask, wall_bc)
        
        # History storage
        self.history: List[Dict[str, Any]] = []
        
        # CSV file handle
        self._csv_file = None
        self._csv_writer = None
    
    def initialize(self) -> None:
        """Print force manager info. CSV is opened later by open_csv()."""
        if not self.enabled:
            print("  Force calculation: disabled")
            return

        print(f"  {self.force_calc.get_info()}")
        print(f"  Reference: ρ={self.rho_ref}, U={self.u_ref}, D={self.char_length}, Lz={self.span_length}")
        print(f"  Interval: every {self.interval} steps (start from step {self.start_step})")

    def open_csv(self, start_step: int = 0) -> None:
        """Open CSV log file. Called by SolverInitializer after start_step is known.

        Args:
            start_step: First step of this run. If > 0 (restart),
                        existing CSV rows up to start_step are preserved
                        and new data is appended.
        """
        if not self.enabled or not self.log_enabled:
            return

        os.makedirs(self.csv_dir, exist_ok=True)
        csv_path = os.path.join(self.csv_dir, f"{self.log_filename}.csv")

        if self.dim == 2:
            fieldnames = ['step', 'Fx', 'Fy', 'Cd', 'Cl']
        else:
            fieldnames = ['step', 'Fx', 'Fy', 'Fz', 'Cd', 'Cl', 'Cz']

        if start_step > 0 and os.path.exists(csv_path):
            # Restart: keep rows with step < start_step
            kept_lines = []
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['step']) < start_step:
                        kept_lines.append(row)

            self._csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames,
            )
            self._csv_writer.writeheader()
            for row in kept_lines:
                self._csv_writer.writerow(row)
            self._csv_file.flush()
            print(f"  Force CSV: {csv_path} (kept {len(kept_lines)} rows, "
                  f"appending from step {start_step})")
        else:
            self._csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames,
            )
            self._csv_writer.writeheader()
            print(f"  Force CSV: {csv_path}")
    
    def should_compute(self, step: int) -> bool:
        """Check if force should be computed at this step"""
        if not self.enabled:
            return False
        if step < self.start_step:
            return False
        return step % self.interval == 0
    
    def compute_and_log(self, step: int, f_post: 'npt.NDArray',
                        verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Compute forces and log to CSV
        
        Args:
            step: Current time step
            f_post: Post-collision distribution
            verbose: Print results to console
        
        Returns:
            Dictionary with forces and coefficients, or None if skipped
        """
        if not self.should_compute(step):
            return None
        
        # Compute forces
        forces = self.force_calc.compute(f_post)
        
        # Get coefficients
        coeffs = self.force_calc.get_coefficients(
            forces,
            rho_ref=self.rho_ref,
            u_ref=self.u_ref,
            char_length=self.char_length,
            span_length=self.span_length
        )
        
        # Build result dictionary
        if self.dim == 2:
            result = {
                'step': step,
                'Fx': forces[0],
                'Fy': forces[1],
                'Cd': coeffs['Cd'],
                'Cl': coeffs['Cl']
            }
        else:
            result = {
                'step': step,
                'Fx': forces[0],
                'Fy': forces[1],
                'Fz': forces[2],
                'Cd': coeffs['Cd'],
                'Cl': coeffs['Cl'],
                'Cz': coeffs.get('Cz', 0.0)
            }
        
        # Store in history
        self.history.append(result)
        
        # Write to CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow(result)
            self._csv_file.flush()
        
        if verbose:
            print(f"  Step {step}: Cd={coeffs['Cd']:.4f}, Cl={coeffs['Cl']:.4f}")
        
        return result
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """Compute final statistics from force history
        
        Returns:
            Dictionary with mean, std, min, max for each coefficient
        """
        import numpy as np
        
        if len(self.history) == 0:
            return {}
        
        Cd_values = [h['Cd'] for h in self.history]
        Cl_values = [h['Cl'] for h in self.history]
        
        stats = {
            'Cd_mean': float(np.mean(Cd_values)),
            'Cd_std': float(np.std(Cd_values)),
            'Cd_min': float(np.min(Cd_values)),
            'Cd_max': float(np.max(Cd_values)),
            'Cl_mean': float(np.mean(Cl_values)),
            'Cl_std': float(np.std(Cl_values)),
            'Cl_min': float(np.min(Cl_values)),
            'Cl_max': float(np.max(Cl_values)),
            'n_samples': len(self.history)
        }
        
        if self.dim == 3 and 'Cz' in self.history[0]:
            Cz_values = [h['Cz'] for h in self.history]
            stats.update({
                'Cz_mean': float(np.mean(Cz_values)),
                'Cz_std': float(np.std(Cz_values)),
            })
        
        return stats
    
    def print_summary(self, window: Optional[int] = None) -> None:
        """Print force summary statistics
        
        Args:
            window: Number of steps to average (None = last 50% of data)
        """
        if not self.enabled:
            print("  Force calculation: disabled")
            return
        
        if len(self.history) == 0:
            print("  Force calculation: no data recorded")
            return
        
        # Use last 50% of data by default (skip transient)
        if window is None:
            window = max(1, len(self.history) // 2)
        
        # Get statistics from recent data
        import numpy as np
        recent = self.history[-window:]
        
        Cd_values = np.array([h['Cd'] for h in recent])
        Cl_values = np.array([h['Cl'] for h in recent])
        Fx_values = np.array([h['Fx'] for h in recent])
        Fy_values = np.array([h['Fy'] for h in recent])
        
        print("\n" + "="*60)
        print(" Force Summary (time-averaged)")
        print("="*60)
        print(f"  Samples: {len(recent)} (last {window} of {len(self.history)} total)")
        print(f"  Drag coefficient: Cd = {np.mean(Cd_values):.4f} ± {np.std(Cd_values):.4f}")
        print(f"  Lift coefficient: Cl = {np.mean(Cl_values):.4f} ± {np.std(Cl_values):.4f}")
        print(f"  Lift RMS:         Cl_rms = {np.sqrt(np.mean(Cl_values**2)):.4f}")
        print(f"  Force (mean):     Fx = {np.mean(Fx_values):.6f}, Fy = {np.mean(Fy_values):.6f}")
        
        if self.dim == 3 and 'Fz' in recent[0]:
            Fz_values = np.array([h['Fz'] for h in recent])
            print(f"                    Fz = {np.mean(Fz_values):.6f}")
        
        print("="*60)
    
    def close(self) -> None:
        """Close CSV file"""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
    
    def get_info(self) -> str:
        """Return information string"""
        dim_str = "2D" if self.dim == 2 else "3D"
        return (f"ForceManager ({dim_str}):\n"
                f"  Enabled: {self.enabled}\n"
                f"  Interval: {self.interval}\n"
                f"  Start step: {self.start_step}\n"
                f"  {self.force_calc.get_info()}")


# =============================================================================
# Strouhal Number Calculation
# =============================================================================

def compute_strouhal_number(force_history: List[Dict[str, Any]],
                            char_length: float,
                            u_ref: float,
                            component: str = 'Cl',
                            min_periods: int = 3) -> Optional[float]:
    """Compute Strouhal number from force time history using FFT
    
    The Strouhal number characterizes vortex shedding frequency:
        St = f * D / U
    
    where:
        f = dominant frequency of lift oscillation  [1/Δt]
        D = characteristic length (diameter)  [Δx]
        U = reference velocity  [Δx/Δt]
    
    Algorithm:
        1. Extract force component time series
        2. Apply FFT to find dominant frequency
        3. Compute St = f * D / U
    
    Args:
        force_history: List of force records with 'step' and component
        char_length: Characteristic length D  [Δx]
        u_ref: Reference velocity U  [Δx/Δt]
        component: Force component to analyze ('Cl', 'Fy', etc.)
        min_periods: Minimum number of oscillation periods required
        
    Returns:
        Strouhal number, or None if insufficient data
        
    Example:
        >>> St = compute_strouhal_number(force_mgr.history, D=20, u_ref=0.1)
        >>> print(f"Strouhal number: {St:.4f}")  # Expected ~0.16-0.21 for cylinder
    """
    import numpy as np
    
    if len(force_history) < 100:
        return None  # Not enough data
    
    # Extract component values
    if component not in force_history[0]:
        return None
    
    values = np.array([h[component] for h in force_history])
    steps = np.array([h['step'] for h in force_history])
    
    # Determine time step (assume uniform spacing)
    dt = 1.0  # In lattice units, one step = Δt = 1
    if len(steps) > 1:
        dt = float(steps[1] - steps[0])
    
    N = len(values)
    
    # Remove mean (DC component)
    values_centered = values - np.mean(values)
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(N)
    values_windowed = values_centered * window
    
    # FFT
    fft_vals = np.fft.rfft(values_windowed)
    freqs = np.fft.rfftfreq(N, d=dt)
    
    # Find dominant frequency (excluding DC)
    magnitudes = np.abs(fft_vals)
    magnitudes[0] = 0  # Ignore DC
    
    # Find peak
    peak_idx = np.argmax(magnitudes)
    if peak_idx == 0:
        return None  # No oscillation detected
    
    f_dominant = freqs[peak_idx]  # [1/Δt]
    
    # Check if we have enough periods
    T_period = 1.0 / f_dominant if f_dominant > 0 else float('inf')
    n_periods = (N * dt) / T_period
    
    if n_periods < min_periods:
        return None  # Insufficient number of periods
    
    # Strouhal number: St = f * D / U
    St = f_dominant * char_length / u_ref
    
    return float(St)