"""
Force Calculation Module using Momentum Exchange Method (MEM)

This module implements the Momentum Exchange Method for computing
hydrodynamic forces on solid obstacles in LBM simulations.

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
Drag coefficient: C_D = F_x / (0.5 * ρ * U² * A)
Lift coefficient: C_L = F_y / (0.5 * ρ * U² * A)

where:
    A = reference area (D * L_z for 2D cylinder)  [Δx²]
    U = freestream velocity  [Δx/Δt]
    ρ = reference density  [dimensionless]

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
from typing import TYPE_CHECKING, Tuple, Optional, Dict, List, Any

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class MomentumExchangeForce:
    """Momentum Exchange Method for Force Calculation
    
    Computes hydrodynamic forces on solid obstacles using the momentum
    exchange at fluid-solid boundary links.
    
    Algorithm:
    ----------
    1. Identify boundary links: fluid nodes with solid neighbors
    2. For each boundary link (fluid node x_f, direction i → solid):
       F_link = 2 * c_i * f_i^post(x_f)
    3. Sum all F_link to get total force
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model
        solid_mask: Boolean mask (True = solid)
        needs_bounce: Pre-computed boundary link mask from HalfwayBounceBack
    
    Example:
        >>> force_calc = MomentumExchangeForce(xp, lattice, solid_mask)
        >>> Fx, Fy, Fz = force_calc.compute(f_post)
        >>> Cd, Cl = force_calc.get_coefficients(Fx, Fy, u_inf=0.1, D=20, Lz=4)
    """
    
    def __init__(self, xp: 'ModuleType', 
                 lattice: 'Lattice',
                 solid_mask: 'npt.NDArray',
                 wall_bc: Optional['HalfwayBounceBack'] = None) -> None:
        """Initialize momentum exchange force calculator
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D3Q27, D3Q19, etc.)
            solid_mask: Boolean solid mask (Nx, Ny, Nz), True = solid
            wall_bc: Optional HalfwayBounceBack instance (reuses precomputed links)
        """
        self.xp = xp
        self.lattice = lattice
        self.Q = lattice.Q
        self.c = xp.asarray(lattice.c, dtype=xp.float64)  # (3, Q) [Δx/Δt]
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
        
        # Storage for time history
        self.force_history: List[Dict[str, Any]] = []
    
    def _precompute_boundary_links(self) -> None:
        """Precompute fluid-solid boundary links
        
        For each direction i, identify fluid nodes whose neighbor
        in direction i is solid. These are the boundary links.
        
        needs_bounce[i, x, y, z] = True means:
            - (x, y, z) is a fluid node
            - (x + c_x[i], y + c_y[i], z + c_z[i]) is a solid node
            - Distribution f_i at (x,y,z) bounces back
        """
        xp = self.xp
        c = self.c.astype(xp.int32)
        solid = self.solid_mask
        
        Nx, Ny, Nz = self.shape
        self.needs_bounce = xp.zeros((self.Q,) + self.shape, dtype=bool)
        
        for i in range(self.Q):
            if i == 0:  # Rest direction, no streaming/bounce-back
                continue
            
            cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
            
            # Shift solid mask by -c_i to check if neighbor is solid
            # shifted_solid[x,y,z] = solid[x+cx, y+cy, z+cz] (with periodic wrap)
            shifted_solid = xp.roll(
                xp.roll(
                    xp.roll(solid, -cx, axis=0),
                    -cy, axis=1),
                -cz, axis=2)
            
            # Boundary link: fluid node where neighbor in direction i is solid
            self.needs_bounce[i] = (~solid) & shifted_solid
    
    def compute(self, f_post: 'npt.NDArray') -> Tuple[float, float, float]:
        """Compute total force on solid using momentum exchange
        
        Mathematical Formula:
        ---------------------
        F = Σ_{boundary links} 2 * c_i * f_i^post(x_fluid)
        
        Args:
            f_post: Post-collision distribution, shape (Q, Nx, Ny, Nz)
                   [dimensionless] - MUST be post-collision, pre-streaming!
        
        Returns:
            Tuple (Fx, Fy, Fz): Force components in lattice units
                               [ρ₀ Δx⁴/Δt²] in LBM units
        
        Note:
            The sign convention: positive force = force ON the solid FROM fluid
        """
        xp = self.xp
        c = self.c  # (3, Q) [Δx/Δt]
        
        # Initialize force components
        Fx = 0.0  # [ρ₀ Δx⁴/Δt²]
        Fy = 0.0
        Fz = 0.0
        
        # Sum over all directions (skip rest direction i=0)
        for i in range(1, self.Q):
            # Get boundary link mask for this direction
            mask = self.needs_bounce[i]  # (Nx, Ny, Nz)
            
            if xp.any(mask):
                # Extract f_post values at boundary links
                f_boundary = f_post[i][mask]  # [dimensionless]
                
                # Momentum exchange: F_link = 2 * c_i * f_i
                # Sum over all links for this direction
                n_links = float(xp.sum(mask))
                f_sum = float(xp.sum(f_boundary))
                
                Fx += 2.0 * float(c[0, i]) * f_sum
                Fy += 2.0 * float(c[1, i]) * f_sum
                Fz += 2.0 * float(c[2, i]) * f_sum
        
        return Fx, Fy, Fz
    
    def compute_vectorized(self, f_post: 'npt.NDArray') -> Tuple[float, float, float]:
        """Fully vectorized force computation (potentially faster on GPU)
        
        Args:
            f_post: Post-collision distribution (Q, Nx, Ny, Nz) [dimensionless]
        
        Returns:
            Tuple (Fx, Fy, Fz): Force components [lattice units]
        """
        xp = self.xp
        c = self.c  # (3, Q)
        
        # Mask f_post at boundary links: set non-boundary to 0
        # needs_bounce: (Q, Nx, Ny, Nz), f_post: (Q, Nx, Ny, Nz)
        f_boundary = xp.where(self.needs_bounce, f_post, 0.0)  # (Q, Nx, Ny, Nz)
        
        # Sum over spatial dimensions to get f_sum per direction: (Q,)
        f_sum = xp.sum(f_boundary, axis=(1, 2, 3))  # [dimensionless]
        
        # Compute force: F = 2 * Σ_i (c_i * f_sum[i])
        # c: (3, Q), f_sum: (Q,) -> F: (3,)
        F = 2.0 * xp.dot(c, f_sum)  # [lattice units]
        
        return float(F[0]), float(F[1]), float(F[2])
    
    def get_coefficients(self, Fx: float, Fy: float,
                         rho_ref: float = 1.0,
                         u_ref: float = 0.1,
                         char_length: float = 20.0,
                         span_length: float = 1.0) -> Tuple[float, float]:
        """Compute drag and lift coefficients
        
        Mathematical Formula:
        ---------------------
        C_D = F_x / (0.5 * ρ * U² * A)
        C_L = F_y / (0.5 * ρ * U² * A)
        
        where A = D * L_z (reference area for 2D cylinder)
        
        Args:
            Fx, Fy: Force components [lattice units]
            rho_ref: Reference density  [dimensionless, typically 1.0]
            u_ref: Reference velocity (freestream)  [Δx/Δt]
            char_length: Characteristic length (diameter D)  [Δx]
            span_length: Span length in z-direction  [Δx]
        
        Returns:
            Tuple (Cd, Cl): Drag and lift coefficients [dimensionless]
        """
        # Reference area: A = D * Lz  [Δx²]
        A_ref = char_length * span_length
        
        # Dynamic pressure: q = 0.5 * ρ * U²  [ρ₀ Δx²/Δt²]
        q = 0.5 * rho_ref * u_ref**2
        
        # Avoid division by zero
        if abs(q * A_ref) < 1e-15:
            return 0.0, 0.0
        
        # Coefficients [dimensionless]
        Cd = Fx / (q * A_ref)
        Cl = Fy / (q * A_ref)
        
        return Cd, Cl
    
    def record_force(self, step: int, Fx: float, Fy: float, Fz: float,
                     Cd: Optional[float] = None, Cl: Optional[float] = None) -> None:
        """Record force data for time history
        
        Args:
            step: Time step number
            Fx, Fy, Fz: Force components [lattice units]
            Cd, Cl: Optional drag/lift coefficients
        """
        record = {
            'step': step,
            'Fx': Fx,
            'Fy': Fy,
            'Fz': Fz,
        }
        if Cd is not None:
            record['Cd'] = Cd
        if Cl is not None:
            record['Cl'] = Cl
        
        self.force_history.append(record)
    
    def get_statistics(self, window: Optional[int] = None) -> Dict[str, float]:
        """Compute force statistics from history
        
        Args:
            window: Number of recent steps to average (None = all)
        
        Returns:
            Dictionary with mean, std, min, max for each component
        """
        if not self.force_history:
            return {}
        
        # Get recent data
        data = self.force_history[-window:] if window else self.force_history
        
        import numpy as np  # Use numpy for statistics even if main array is cupy
        
        Fx = np.array([d['Fx'] for d in data])
        Fy = np.array([d['Fy'] for d in data])
        Fz = np.array([d['Fz'] for d in data])
        
        stats = {
            'Fx_mean': float(np.mean(Fx)),
            'Fx_std': float(np.std(Fx)),
            'Fy_mean': float(np.mean(Fy)),
            'Fy_std': float(np.std(Fy)),
            'Fz_mean': float(np.mean(Fz)),
            'Fz_std': float(np.std(Fz)),
        }
        
        if 'Cd' in data[0]:
            Cd = np.array([d['Cd'] for d in data])
            Cl = np.array([d['Cl'] for d in data])
            stats['Cd_mean'] = float(np.mean(Cd))
            stats['Cd_std'] = float(np.std(Cd))
            stats['Cl_mean'] = float(np.mean(Cl))
            stats['Cl_std'] = float(np.std(Cl))
            stats['Cl_rms'] = float(np.sqrt(np.mean(Cl**2)))
        
        return stats
    
    def get_info(self) -> str:
        """Return information string about the force calculator"""
        solid_count = int(self.xp.sum(self.solid_mask))
        
        return (f"Momentum Exchange Force Calculator:\n"
                f"  Solid nodes: {solid_count:,}\n"
                f"  Boundary links: {self.n_boundary_links:,}\n"
                f"  History length: {len(self.force_history)}")


class ForceLogger:
    """CSV Logger for Force Time History
    
    Logs force data to CSV file for post-processing and analysis.
    
    Example:
        >>> logger = ForceLogger('./results/csv', 'force_history')
        >>> logger.log(step=1000, Fx=0.01, Fy=0.001, Cd=1.5, Cl=0.1)
        >>> logger.close()
    """
    
    def __init__(self, output_dir: str, 
                 filename: str = 'force_history',
                 include_coefficients: bool = True) -> None:
        """Initialize force logger
        
        Args:
            output_dir: Directory for CSV file
            filename: Base filename (without extension)
            include_coefficients: Include Cd, Cl columns
        """
        self.output_dir = output_dir
        self.include_coefficients = include_coefficients
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filepath = os.path.join(output_dir, f'{filename}_{timestamp}.csv')
        
        # Build header
        headers = ['step', 'time', 'Fx', 'Fy', 'Fz']
        if include_coefficients:
            headers.extend(['Cd', 'Cl'])
        
        # Open file and write header
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(headers)
        self.file.flush()
        
        print(f"    Force log: {self.filepath}")
    
    def log(self, step: int, 
            Fx: float, Fy: float, Fz: float,
            time: Optional[float] = None,
            Cd: Optional[float] = None, 
            Cl: Optional[float] = None) -> None:
        """Log force data for one time step
        
        Args:
            step: Time step number
            Fx, Fy, Fz: Force components [lattice units]
            time: Physical time (defaults to step)
            Cd, Cl: Drag and lift coefficients
        """
        if time is None:
            time = float(step)
        
        row = [step, time, Fx, Fy, Fz]
        
        if self.include_coefficients:
            row.extend([Cd if Cd is not None else 0.0,
                       Cl if Cl is not None else 0.0])
        
        self.writer.writerow(row)
        self.file.flush()
    
    def close(self) -> None:
        """Close the CSV file"""
        if self.file is not None:
            self.file.close()
            self.file = None
    
    def __del__(self):
        self.close()


class ForceManager:
    """High-level manager for force calculation and logging
    
    Combines MomentumExchangeForce with logging and analysis.
    
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
            solid_mask: Solid mask (Nx, Ny, Nz)
            config: Force configuration dictionary:
                    - enabled: bool
                    - interval: int (compute every N steps)
                    - start_step: int (start computing after N steps)
                    - reference: {rho, velocity, char_length, span_length}
            wall_bc: Optional wall BC (reuses boundary link info)
            csv_dir: Directory for CSV output
        """
        self.xp = xp
        self.config = config
        self.csv_dir = csv_dir
        
        # Parse config
        self.enabled = config.get('enabled', True)
        self.interval = config.get('interval', 1)
        self.start_step = config.get('start_step', 0)
        
        ref_config = config.get('reference', {})
        self.rho_ref = ref_config.get('rho', 1.0)
        self.u_ref = ref_config.get('velocity', 0.1)
        self.char_length = ref_config.get('char_length', ref_config.get('area', 20.0))
        self.span_length = ref_config.get('span_length', 1.0)
        
        # Initialize calculator
        if self.enabled:
            self.calculator = MomentumExchangeForce(xp, lattice, solid_mask, wall_bc)
        else:
            self.calculator = None
        
        self.logger: Optional[ForceLogger] = None
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize logger and prepare for force computation"""
        if not self.enabled:
            return
        
        # Create logger
        log_config = self.config.get('log', {})
        if log_config.get('enabled', True):
            self.logger = ForceLogger(
                self.csv_dir,
                filename=log_config.get('filename', 'force_history'),
                include_coefficients=True
            )
        
        self.initialized = True
        
        print(f"  {self.calculator.get_info()}")
        print(f"    Reference: ρ={self.rho_ref}, U={self.u_ref}, D={self.char_length}, Lz={self.span_length}")
    
    def should_compute(self, step: int) -> bool:
        """Check if force should be computed at this step"""
        if not self.enabled:
            return False
        if step < self.start_step:
            return False
        return (step - self.start_step) % self.interval == 0
    
    def compute_and_log(self, step: int, f_post: 'npt.NDArray',
                        verbose: bool = False) -> Optional[Dict[str, float]]:
        """Compute force and log to CSV
        
        Args:
            step: Current time step
            f_post: Post-collision distribution (Q, Nx, Ny, Nz)
            verbose: Print force values
        
        Returns:
            Dictionary with Fx, Fy, Fz, Cd, Cl (or None if not computed)
        """
        if not self.should_compute(step):
            return None
        
        # Compute forces
        Fx, Fy, Fz = self.calculator.compute_vectorized(f_post)
        
        # Compute coefficients
        Cd, Cl = self.calculator.get_coefficients(
            Fx, Fy,
            rho_ref=self.rho_ref,
            u_ref=self.u_ref,
            char_length=self.char_length,
            span_length=self.span_length
        )
        
        # Record to history
        self.calculator.record_force(step, Fx, Fy, Fz, Cd, Cl)
        
        # Log to CSV
        if self.logger is not None:
            self.logger.log(step, Fx, Fy, Fz, time=float(step), Cd=Cd, Cl=Cl)
        
        if verbose:
            print(f"  Step {step}: Fx={Fx:.6f}, Fy={Fy:.6f}, Cd={Cd:.4f}, Cl={Cl:.4f}")
        
        return {
            'Fx': Fx, 'Fy': Fy, 'Fz': Fz,
            'Cd': Cd, 'Cl': Cl
        }
    
    def get_final_statistics(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get final force statistics
        
        Args:
            window: Number of recent steps to average (None = use last 50%)
        
        Returns:
            Statistics dictionary
        """
        if not self.enabled or self.calculator is None:
            return {}
        
        # Default: use last 50% of data for averaging (skip transient)
        if window is None:
            window = len(self.calculator.force_history) // 2
        
        return self.calculator.get_statistics(window)
    
    def print_summary(self, window: Optional[int] = None) -> None:
        """Print force summary statistics"""
        if not self.enabled:
            print("  Force calculation: disabled")
            return
        
        stats = self.get_final_statistics(window)
        
        if not stats:
            print("  Force calculation: no data recorded")
            return
        
        print("\n" + "="*60)
        print(" Force Summary (time-averaged)")
        print("="*60)
        print(f"  Drag coefficient: Cd = {stats.get('Cd_mean', 0):.4f} ± {stats.get('Cd_std', 0):.4f}")
        print(f"  Lift coefficient: Cl = {stats.get('Cl_mean', 0):.4f} ± {stats.get('Cl_std', 0):.4f}")
        print(f"  Lift RMS:         Cl_rms = {stats.get('Cl_rms', 0):.4f}")
        print(f"  Force (mean):     Fx = {stats.get('Fx_mean', 0):.6f}, Fy = {stats.get('Fy_mean', 0):.6f}")
        print("="*60)
    
    def close(self) -> None:
        """Close logger"""
        if self.logger is not None:
            self.logger.close()
    
    def __del__(self):
        self.close()


# =============================================================================
# Strouhal Number Calculation (for vortex shedding frequency)
# =============================================================================

def compute_strouhal_number(force_history: List[Dict[str, Any]],
                            char_length: float,
                            u_ref: float,
                            component: str = 'Cl',
                            min_periods: int = 3) -> Optional[float]:
    """Compute Strouhal number from force time history
    
    St = f * D / U
    
    where f is the dominant frequency of lift oscillation.
    
    Args:
        force_history: List of force records with 'step' and component
        char_length: Characteristic length D  [Δx]
        u_ref: Reference velocity U  [Δx/Δt]
        component: Which force component to analyze ('Cl', 'Fy', etc.)
        min_periods: Minimum periods required for reliable FFT
    
    Returns:
        Strouhal number, or None if cannot be computed
    """
    import numpy as np
    
    if len(force_history) < 10:
        return None
    
    # Extract signal
    if component not in force_history[0]:
        component = 'Fy'  # Fallback to raw force
    
    steps = np.array([d['step'] for d in force_history])
    signal = np.array([d.get(component, d.get('Fy', 0)) for d in force_history])
    
    # Remove mean (DC component)
    signal = signal - np.mean(signal)
    
    if np.std(signal) < 1e-10:
        return None  # No oscillation
    
    # FFT
    n = len(signal)
    dt = 1.0  # Assume uniform time step in lattice units
    
    # Compute FFT
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    
    # Find dominant frequency (excluding DC)
    magnitudes = np.abs(fft_result)
    magnitudes[0] = 0  # Exclude DC
    
    peak_idx = np.argmax(magnitudes)
    dominant_freq = freqs[peak_idx]  # [1/Δt]
    
    # Check if we have enough periods
    period = 1.0 / (dominant_freq + 1e-10)
    n_periods = n / period
    
    if n_periods < min_periods:
        return None  # Not enough data for reliable frequency
    
    # Strouhal number
    St = dominant_freq * char_length / u_ref
    
    return float(St)