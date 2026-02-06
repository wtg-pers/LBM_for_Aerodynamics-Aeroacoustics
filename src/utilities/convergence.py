"""
Convergence Detection Module for LBM Solver

Monitors simulation convergence using Cauchy convergence criterion:
successive window-mean comparison, following SU2 (CONV_CAUCHY) and
Palabos (ValueTracer) approaches.

Cauchy Convergence Criterion:
=============================
    Given a rolling window of N samples, split into two halves:

        μ_old = mean(first half)
        μ_new = mean(second half)

        ε_cauchy = |μ_new - μ_old| / |μ_old|

    Converged when: ε_cauchy < threshold for n_required consecutive checks.

Why Cauchy over CV:
===================
    CV = σ/|μ| measures SPREAD within a window.
    For periodic flows (vortex shedding), Cd oscillates → σ ≠ 0 → CV never → 0.

    Cauchy measures DRIFT of the MEAN between successive windows.
    Even if Cd oscillates (1.35 ± 0.01), the mean stabilizes → ε_cauchy → 0.

    This correctly identifies:
        - Transient phase: mean still shifting → ε_cauchy large → NOT converged
        - Developed phase: mean stabilized → ε_cauchy ≈ 0 → CONVERGED

Convergence Paths:
==================
    Path A (no obstacle): Energy is the sole criterion (Palabos style)
        - Cauchy(E_avg) < ε_energy     [criterion]

    Path B (with obstacle): Cd is the sole criterion (SU2 style)
        - Cauchy(Cd) < ε_Cd            [criterion]
        - Energy, Cl: logged only       [monitor]

Window Sizing:
==============
    T_conv = D / U  [timesteps]
    window_timesteps = T_conv × time_coverage  [timesteps]
    window_samples   = window_timesteps / feed_interval  [samples]

    The window should span enough shedding cycles so that the mean
    is representative. Default time_coverage=50 gives ~10 shedding
    periods for St ≈ 0.2.

Cauchy Threshold Guidance:
==========================
    Cd:     ε = 1e-3   (mean drag stabilized to 0.1% between windows)
    Energy: ε = 1e-5   (domain-averaged energy is very smooth)

    These are much more intuitive than CV thresholds because they
    directly answer: "has the mean stopped drifting?"

IMPORTANT - Feed Intervals:
===========================
    Energy and Force trackers are fed at `force_interval` in main.py.
    Convergence CHECK is done at `check_interval`.

    main.py data flow:
        every force_interval:  feed_energy(), feed_force()
        every check_interval:  feed_divergence_check(), check()

References:
===========
    - SU2: SU2_CFD/src/solvers/CSolver.cpp → CONV_CAUCHY
    - Palabos: src/core/util.h → ValueTracer (relative change of avg)
    - OpenLB: src/utilities/valueTracer.h

Author: LBM Development Team
Date: 2026-02
"""

import os
import csv
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np


# =============================================================================
# Convergence Status Enum
# =============================================================================

class ConvergenceStatus(Enum):
    """Convergence monitor state"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    MAX_STEPS = "max_steps"
    DISABLED = "disabled"


# =============================================================================
# Ring Buffer
# =============================================================================

class RingBuffer:
    """Fixed-size circular buffer for scalar time series
    
    Provides O(1) push and O(1) mean/std via incremental statistics.
    
    Attributes:
        capacity: Buffer size  [samples]
        count: Values currently stored  [samples]
    """
    
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self._data: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self._index: int = 0
        self.count: int = 0
    
    def push(self, value: float) -> None:
        """Append value, overwriting oldest if full."""
        self._data[self._index] = value
        self._index = (self._index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    
    @property
    def is_full(self) -> bool:
        return self.count >= self.capacity
    
    @property
    def values(self) -> np.ndarray:
        """Return values in chronological order (oldest first)."""
        if self.count < self.capacity:
            return self._data[:self.count].copy()
        # Ring buffer is full: unwrap from _index
        return np.roll(self._data, -self._index).copy()
    
    def mean(self) -> float:
        """Mean of all stored values."""
        if self.count == 0:
            return 0.0
        if self.count < self.capacity:
            return float(np.mean(self._data[:self.count]))
        return float(np.mean(self._data))
    
    def std(self) -> float:
        """Standard deviation of all stored values."""
        if self.count < 2:
            return float('inf')
        if self.count < self.capacity:
            return float(np.std(self._data[:self.count]))
        return float(np.std(self._data))
    
    def cv(self) -> float:
        """Coefficient of variation: CV = σ / |μ|  [dimensionless]
        
        Retained for monitoring/logging purposes, NOT for convergence.
        """
        if self.count < 2:
            return float('inf')
        mu = self.mean()
        sigma = self.std()
        if abs(mu) < 1e-30:
            return float('inf') if sigma > 1e-30 else 0.0
        return sigma / abs(mu)
    
    def first_half_mean(self) -> float:
        """Mean of the older half of the buffer (chronological order)."""
        if self.count < 2:
            return 0.0
        vals = self.values
        half = len(vals) // 2
        return float(np.mean(vals[:half]))
    
    def second_half_mean(self) -> float:
        """Mean of the newer half of the buffer (chronological order)."""
        if self.count < 2:
            return 0.0
        vals = self.values
        half = len(vals) // 2
        return float(np.mean(vals[half:]))
    
    def reset(self) -> None:
        self._data[:] = 0.0
        self._index = 0
        self.count = 0


# =============================================================================
# Cauchy Tracker (replaces ScalarTracker)
# =============================================================================

class CauchyTracker:
    """Tracks convergence via Cauchy criterion: relative change of window mean
    
    Cauchy Criterion:
        ε_cauchy = |μ_new - μ_old| / |μ_old|
        
        where μ_old = mean(first half of window)
              μ_new = mean(second half of window)
        
        Converged when ε_cauchy < epsilon for n_required consecutive checks.
    
    Physical Interpretation:
        - For steady flow: Cd stabilizes → μ_new ≈ μ_old → ε_cauchy → 0
        - For periodic flow (shedding): Cd oscillates around a stable mean
          → window averages converge → ε_cauchy → 0
        - During transient: mean is still drifting → ε_cauchy >> epsilon
    
    Comparison with CV:
        CV = σ/|μ| → measures oscillation amplitude (fails for periodic flow)
        Cauchy     → measures mean drift (works for both steady & periodic)
    
    Attributes:
        name: Identifier (e.g., 'Cd', 'avg_energy')
        epsilon: Cauchy threshold  [dimensionless]
        window_size: Buffer capacity  [samples]
        n_required: Consecutive passes needed  [count]
        is_criterion: If True, participates in convergence decision
        feed_interval: Feed rate  [timesteps/sample]
    """
    
    def __init__(self,
                 name: str,
                 epsilon: float,
                 window_size: int,
                 n_required: int = 3,
                 is_criterion: bool = True,
                 feed_interval: int = 1) -> None:
        self.name: str = name
        self.epsilon: float = epsilon
        self.window_size: int = window_size
        self.n_required: int = n_required
        self.is_criterion: bool = is_criterion
        self.feed_interval: int = feed_interval
        
        self.buffer: RingBuffer = RingBuffer(window_size)
        
        # Cauchy state
        self._cauchy_value: float = float('inf')   # current ε_cauchy
        self._consecutive: int = 0                  # consecutive passes
        self._converged: bool = False
        self._total_fed: int = 0
    
    def push(self, value: float) -> None:
        """Feed a new sample into the buffer."""
        self.buffer.push(value)
        self._total_fed += 1
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate Cauchy criterion
        
        Returns:
            Dict with:
                converged: bool
                cauchy: current ε_cauchy value
                mean: current window mean
                std: current window std (for monitoring)
                cv: current CV (for monitoring/logging)
                consecutive: number of consecutive passes
                samples: current sample count
                window_size: buffer capacity
        """
        if not self.buffer.is_full:
            self._cauchy_value = float('inf')
            return self._build_status()
        
        # --- Cauchy criterion: compare first-half mean vs second-half mean ---
        mu_old = self.buffer.first_half_mean()
        mu_new = self.buffer.second_half_mean()
        
        if abs(mu_old) < 1e-30:
            # Denominator near zero: use absolute difference
            self._cauchy_value = abs(mu_new - mu_old)
        else:
            self._cauchy_value = abs(mu_new - mu_old) / abs(mu_old)
        
        # --- Consecutive pass tracking ---
        if self._cauchy_value < self.epsilon:
            self._consecutive += 1
        else:
            self._consecutive = 0
        
        if self._consecutive >= self.n_required:
            self._converged = True
        
        return self._build_status()
    
    def _build_status(self) -> Dict[str, Any]:
        """Build status dict for logging and convergence check."""
        return {
            'name': self.name,
            'converged': self._converged,
            'cauchy': self._cauchy_value,
            'epsilon': self.epsilon,
            'consecutive': self._consecutive,
            'n_required': self.n_required,
            'mean': self.buffer.mean(),
            'std': self.buffer.std(),
            'cv': self.buffer.cv(),
            'samples': self.buffer.count,
            'window_size': self.window_size,
            'is_criterion': self.is_criterion,
            'total_fed': self._total_fed,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status without re-evaluating."""
        return self._build_status()


# =============================================================================
# Auto Window Sizing
# =============================================================================

def compute_auto_window_sizes(
        char_length: float,
        u_ref: float,
        feed_interval: int = 10,
        time_coverage: float = 50.0,
        min_samples: int = 100,
) -> Dict[str, Any]:
    """Compute window size in SAMPLES from physical time scales
    
    Window should span enough convective times (and shedding periods)
    so that the half-window mean is representative.
    
    For Cauchy criterion, each half should cover at least several 
    shedding periods. With time_coverage=50 and St≈0.2:
        T_shed ≈ 5 T_conv → window ≈ 10 shedding periods
        → each half ≈ 5 shedding periods (sufficient for stable mean)
    
    Args:
        char_length: D  [Δx]
        u_ref: U  [Δx/Δt]
        feed_interval: How often trackers are fed  [timesteps/sample]
        time_coverage: Window spans this many T_conv  [dimensionless]
        min_samples: Minimum buffer size  [samples]
    
    Returns:
        Dict with 'window_samples', 'T_conv', diagnostics
    """
    if u_ref < 1e-15:
        return {
            'window_samples': min_samples,
            'T_conv': 0.0,
            'fill_steps': min_samples * feed_interval,
        }
    
    T_conv = char_length / u_ref                        # [timesteps]
    window_timesteps = T_conv * time_coverage            # [timesteps]
    window_samples = max(
        int(window_timesteps / feed_interval),
        min_samples
    )                                                    # [samples]
    
    # Ensure even number for clean half-splitting
    if window_samples % 2 != 0:
        window_samples += 1
    
    return {
        'window_samples': window_samples,
        'T_conv': T_conv,
        'fill_steps': window_samples * feed_interval,    # [timesteps]
    }


# =============================================================================
# Divergence Detector
# =============================================================================

class DivergenceDetector:
    """Detects NaN, Inf, density/velocity blow-up
    
    Instability criteria:
        1. NaN / Inf in fields
        2. |ρ - ρ₀| > density_bound  [dimensionless]
        3. |u| > velocity_bound  [Δx/Δt]  (default: 1/√3 ≈ 0.577)
    """
    
    def __init__(self,
                 density_bound: float = 0.5,
                 velocity_bound: float = 0.577) -> None:
        self.density_bound = density_bound
        self.velocity_bound = velocity_bound
    
    def check(self, rho: Any, u: Optional[Any] = None) -> Dict[str, Any]:
        rho_np = rho.get() if hasattr(rho, 'get') else np.asarray(rho)
        
        if np.isnan(rho_np).any() or np.isinf(rho_np).any():
            return {'diverged': True, 'reason': 'NaN/Inf in density',
                    'details': {'nan_count': int(np.isnan(rho_np).sum())}}
        
        rho_dev = np.max(np.abs(rho_np - 1.0))
        if rho_dev > self.density_bound:
            return {'diverged': True,
                    'reason': f'Density deviation {rho_dev:.4f} > {self.density_bound}',
                    'details': {'rho_min': float(rho_np.min()),
                                'rho_max': float(rho_np.max())}}
        
        if u is not None:
            u_np = u.get() if hasattr(u, 'get') else np.asarray(u)
            if np.isnan(u_np).any() or np.isinf(u_np).any():
                return {'diverged': True, 'reason': 'NaN/Inf in velocity',
                        'details': {}}
            u_max = float(np.sqrt(np.sum(u_np**2, axis=0)).max())
            if u_max > self.velocity_bound:
                return {'diverged': True,
                        'reason': f'Velocity {u_max:.4f} > {self.velocity_bound}',
                        'details': {'max_mach': u_max * np.sqrt(3)}}
        
        return {'diverged': False, 'reason': None, 'details': {}}


# =============================================================================
# Convergence Monitor (Main Interface)
# =============================================================================

class ConvergenceMonitor:
    """Main convergence monitoring system for LBM solver
    
    Architecture:
    =============
    Path A (no obstacle): Energy-based (Palabos style)
        - Cauchy(E_avg) < ε_energy   [criterion]
    
    Path B (with obstacle): Cd-based (SU2 style)
        - Cauchy(Cd) < ε_Cd          [criterion]
        - Energy, Cl                  [monitor]
    
    Cauchy Criterion (replaces CV):
    ===============================
    Instead of checking σ/|μ| < ε (fails for periodic flows),
    we check: |μ_new - μ_old| / |μ_old| < ε
    
    This measures whether the MEAN is still drifting, which works
    correctly for both steady and periodic (vortex shedding) flows.
    
    Feed & Check Separation:
    ========================
    All trackers are FED at `feed_interval` (= force_interval in main.py).
    Convergence is CHECKED at `check_interval` (less frequently).
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 has_obstacle: bool = False,
                 csv_dir: str = './results/csv') -> None:
        """
        Args:
            config: Convergence configuration dict
            has_obstacle: Enable force tracking (Path B)
            csv_dir: CSV output directory
        """
        self.enabled: bool = config.get('enabled', False)
        self.config: Dict[str, Any] = config
        self.has_obstacle: bool = has_obstacle
        self.csv_dir: str = csv_dir
        
        # Actions
        self.on_converged: str = config.get('on_converged', 'checkpoint_and_stop')
        self.on_max_steps: str = config.get('on_max_steps', 'warn')
        self.on_diverged: str = config.get('on_diverged', 'stop_with_checkpoint')
        
        # State
        self._status: ConvergenceStatus = ConvergenceStatus.NOT_STARTED
        self._diverge_reason: Optional[str] = None
        self._check_count: int = 0
        self._converged_step: Optional[int] = None
        
        # Trackers (created in initialize)
        self.energy_tracker: Optional[CauchyTracker] = None
        self.Cd_tracker: Optional[CauchyTracker] = None
        self.Cl_tracker: Optional[CauchyTracker] = None
        self.divergence_detector: DivergenceDetector = DivergenceDetector()
        
        # CSV
        self._csv_writer = None
        self._csv_file = None
    
    def initialize(self,
                   char_length: float,
                   u_ref: float,
                   feed_interval: int = 10) -> None:
        """Initialize trackers with auto-computed window sizes
        
        All trackers share the same feed_interval because both energy
        and force are fed at the same rate in the main loop.
        
        Args:
            char_length: D  [Δx]
            u_ref: U  [Δx/Δt]
            feed_interval: How often data is fed  [timesteps/sample]
                           (= force_interval in main.py)
        """
        if not self.enabled:
            self._status = ConvergenceStatus.DISABLED
            return
        
        stat_config = self.config.get('cauchy', {})
        window_cfg = stat_config.get('window_size', 'auto')
        
        # --- Window size ---
        if window_cfg == 'auto':
            auto_cfg = stat_config.get('auto_window', {})
            sizes = compute_auto_window_sizes(
                char_length=char_length,
                u_ref=u_ref,
                feed_interval=feed_interval,
                time_coverage=auto_cfg.get('time_coverage', 50.0),
                min_samples=auto_cfg.get('min_samples', 100),
            )
            window_samples = sizes['window_samples']
            self._T_conv = sizes['T_conv']
        else:
            window_samples = int(window_cfg)
            # Ensure even
            if window_samples % 2 != 0:
                window_samples += 1
            self._T_conv = char_length / max(u_ref, 1e-15)
        
        self._window_samples = window_samples
        self._feed_interval = feed_interval
        
        # --- Epsilon defaults ---
        # Cauchy thresholds are more intuitive than CV thresholds:
        #   "Has the mean stopped changing by more than X%?"
        # Cd:     1e-3 → mean Cd stable to 0.1% between halves
        # Energy: 1e-5 → domain-averaged energy extremely stable
        epsilon_energy = stat_config.get('epsilon', 1e-5)
        epsilon_Cd = stat_config.get('Cd_epsilon', 1e-3)
        n_required = stat_config.get('n_required', 3)
        
        # --- Create trackers ---
        if self.has_obstacle:
            # Path B: Cd is criterion, energy & Cl are monitoring
            self.energy_tracker = CauchyTracker(
                name='avg_energy',
                epsilon=epsilon_energy,
                window_size=window_samples,
                n_required=n_required,
                is_criterion=False,          # monitoring only
                feed_interval=feed_interval,
            )
            self.Cd_tracker = CauchyTracker(
                name='Cd',
                epsilon=epsilon_Cd,
                window_size=window_samples,
                n_required=n_required,
                is_criterion=True,           # sole convergence criterion
                feed_interval=feed_interval,
            )
            self.Cl_tracker = CauchyTracker(
                name='Cl',
                epsilon=epsilon_Cd,
                window_size=window_samples,
                n_required=n_required,
                is_criterion=False,          # monitoring only (Cl oscillates ±)
                feed_interval=feed_interval,
            )
        else:
            # Path A: Energy is the sole criterion
            self.energy_tracker = CauchyTracker(
                name='avg_energy',
                epsilon=epsilon_energy,
                window_size=window_samples,
                n_required=n_required,
                is_criterion=True,
                feed_interval=feed_interval,
            )
        
        self._status = ConvergenceStatus.RUNNING
        self._init_csv_logger()
        self._print_config(char_length, u_ref, feed_interval,
                           window_samples, n_required)
    
    def _print_config(self, char_length: float, u_ref: float,
                      feed_interval: int, window_samples: int,
                      n_required: int) -> None:
        """Print initialization summary."""
        path = "B (Cd-based, SU2 Cauchy)" if self.has_obstacle \
               else "A (Energy-based, Palabos Cauchy)"
        fill_steps = window_samples * feed_interval
        
        print(f"\n  Convergence Monitor: ENABLED — Path {path}")
        print(f"    Method: Cauchy (window-mean drift)")
        print(f"    T_conv = D/U = {char_length}/{u_ref} = {self._T_conv:.1f} steps")
        print(f"    Window = {window_samples} samples × {feed_interval} steps "
              f"= {fill_steps:,} steps to fill")
        print(f"    Half-window = {window_samples // 2} samples "
              f"(μ_old vs μ_new comparison)")
        print(f"    n_required = {n_required} consecutive passes")
        
        for tracker in [self.energy_tracker, self.Cd_tracker, self.Cl_tracker]:
            if tracker is not None:
                role = "criterion" if tracker.is_criterion else "monitor "
                print(f"    {tracker.name:>12s}: ε = {tracker.epsilon:.1e}  [{role}]")
        
        print(f"    Actions: converged → {self.on_converged}, "
              f"diverged → {self.on_diverged}")
    
    def _init_csv_logger(self) -> None:
        """Initialize CSV logger for convergence history."""
        os.makedirs(self.csv_dir, exist_ok=True)
        csv_path = os.path.join(self.csv_dir, 'convergence_history.csv')
        self._csv_file = open(csv_path, 'w', newline='')
        
        headers = ['step',
                    'energy_cauchy', 'energy_mean', 'energy_cv',
                    'energy_consecutive']
        if self.has_obstacle:
            headers += ['Cd_cauchy', 'Cd_mean', 'Cd_cv', 'Cd_consecutive',
                        'Cd_converged',
                        'Cl_cauchy', 'Cl_mean', 'Cl_cv']
        headers.append('all_converged')
        
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=headers)
        self._csv_writer.writeheader()
    
    # -------------------------------------------------------------------------
    # Data Feeding (called at feed_interval)
    # -------------------------------------------------------------------------
    
    def feed_energy(self, step: int, rho: Any, u: Any) -> None:
        """Feed domain-averaged kinetic energy: E = (1/N) Σ 0.5ρ|u|²
        
        Units: [Δx²/Δt²]  (lattice units)
        
        Call at feed_interval (= force_interval) in main.py.
        """
        if not self.enabled or self.energy_tracker is None:
            return
        if hasattr(rho, 'get'):
            xp = type(rho)
            E = float(0.5 * xp.mean(rho * xp.sum(u * u, axis=0)))
        else:
            E = float(0.5 * np.mean(rho * np.sum(u * u, axis=0)))
        self.energy_tracker.push(E)
    
    def feed_force(self, step: int, Cd: float, Cl: float) -> None:
        """Feed force coefficients. Call at feed_interval."""
        if not self.enabled:
            return
        if self.Cd_tracker is not None:
            self.Cd_tracker.push(Cd)
        if self.Cl_tracker is not None:
            self.Cl_tracker.push(Cl)
    
    def feed_divergence_check(self, rho: Any,
                              u: Optional[Any] = None) -> Dict[str, Any]:
        """Check for divergence. Call at check_interval."""
        if not self.enabled:
            return {'diverged': False, 'reason': None, 'details': {}}
        result = self.divergence_detector.check(rho, u)
        if result['diverged']:
            self._status = ConvergenceStatus.DIVERGED
            self._diverge_reason = result['reason']
        return result
    
    # -------------------------------------------------------------------------
    # Convergence Evaluation (called at check_interval)
    # -------------------------------------------------------------------------
    
    def check(self, step: int) -> Dict[str, Any]:
        """Evaluate Cauchy convergence on all criterion trackers
        
        Each criterion tracker must satisfy:
            |μ_new - μ_old| / |μ_old| < ε  for n_required consecutive checks
        
        Returns:
            Dict with 'converged', 'diverged', 'action', 'trackers',
            'diverge_reason'
        """
        if not self.enabled:
            return {'converged': False, 'diverged': False,
                    'action': 'continue', 'trackers': {},
                    'diverge_reason': None}
        
        self._check_count += 1
        
        trackers = {}
        all_criteria_met = True
        
        for tracker in [self.energy_tracker, self.Cd_tracker, self.Cl_tracker]:
            if tracker is not None:
                status = tracker.evaluate()
                trackers[tracker.name] = status
                if tracker.is_criterion and not status['converged']:
                    all_criteria_met = False
        
        # --- Determine action ---
        if self._status == ConvergenceStatus.DIVERGED:
            action = self.on_diverged
        elif all_criteria_met:
            self._status = ConvergenceStatus.CONVERGED
            self._converged_step = step
            action = self.on_converged
        else:
            action = 'continue'
        
        # --- Log to CSV ---
        self._log_csv(step, trackers, all_criteria_met)
        
        return {
            'converged': (self._status == ConvergenceStatus.CONVERGED),
            'diverged': (self._status == ConvergenceStatus.DIVERGED),
            'action': action,
            'trackers': trackers,
            'diverge_reason': self._diverge_reason,
        }
    
    def _log_csv(self, step: int, trackers: Dict, converged: bool) -> None:
        """Write one row to convergence CSV."""
        if self._csv_writer is None:
            return
        
        row: Dict[str, Any] = {'step': step}
        
        # Energy (always present)
        if 'avg_energy' in trackers:
            e = trackers['avg_energy']
            row['energy_cauchy'] = f"{e['cauchy']:.6e}"
            row['energy_mean'] = f"{e['mean']:.8e}"
            row['energy_cv'] = f"{e['cv']:.6e}"
            row['energy_consecutive'] = e['consecutive']
        
        # Force trackers (Path B only)
        if 'Cd' in trackers:
            cd = trackers['Cd']
            row['Cd_cauchy'] = f"{cd['cauchy']:.6e}"
            row['Cd_mean'] = f"{cd['mean']:.6f}"
            row['Cd_cv'] = f"{cd['cv']:.6e}"
            row['Cd_consecutive'] = cd['consecutive']
            row['Cd_converged'] = cd['converged']
        
        if 'Cl' in trackers:
            cl = trackers['Cl']
            row['Cl_cauchy'] = f"{cl['cauchy']:.6e}"
            row['Cl_mean'] = f"{cl['mean']:.6f}"
            row['Cl_cv'] = f"{cl['cv']:.6e}"
        
        row['all_converged'] = converged
        
        self._csv_writer.writerow(row)
        self._csv_file.flush()
    
    # -------------------------------------------------------------------------
    # Status & Properties
    # -------------------------------------------------------------------------
    
    @property
    def converged(self) -> bool:
        return self._status == ConvergenceStatus.CONVERGED
    
    @property
    def diverged(self) -> bool:
        return self._status == ConvergenceStatus.DIVERGED
    
    @property
    def converged_step(self) -> Optional[int]:
        return self._converged_step
    
    def get_status(self) -> ConvergenceStatus:
        return self._status
    
    def mark_max_steps(self) -> None:
        """Called when max_steps reached without convergence."""
        if self._status == ConvergenceStatus.RUNNING:
            self._status = ConvergenceStatus.MAX_STEPS
    
    def print_summary(self) -> None:
        """Print convergence summary table."""
        print("\n" + "=" * 78)
        print("  Convergence Summary (Cauchy Criterion)")
        print("=" * 78)
        
        status_str = {
            ConvergenceStatus.CONVERGED: "✅ CONVERGED",
            ConvergenceStatus.DIVERGED: "❌ DIVERGED",
            ConvergenceStatus.RUNNING: "🔄 RUNNING",
            ConvergenceStatus.MAX_STEPS: "⚠️  MAX STEPS",
        }.get(self._status, str(self._status))
        
        print(f"  Status: {status_str}")
        if self._converged_step is not None:
            print(f"  Converged at step: {self._converged_step}")
        print(f"  Checks performed: {self._check_count}")
        
        print(f"\n  {'Tracker':>12s}  {'ε_cauchy':>12s}  {'ε_thresh':>10s}  "
              f"{'Pass':>6s}  {'μ':>12s}  {'CV':>10s}  "
              f"{'Fill':>12s}  {'Role'}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*12}  "
              f"{'-'*10}  {'-'*12}  {'-'*10}")
        
        for tracker in [self.energy_tracker, self.Cd_tracker, self.Cl_tracker]:
            if tracker is not None:
                s = tracker.get_status()
                role = "[criterion]" if s['is_criterion'] else "[monitor] "
                conv_mark = f"{s['consecutive']}/{s['n_required']}"
                fill = f"{s['samples']}/{s['window_size']}"
                
                print(f"  {s['name']:>12s}  {s['cauchy']:>12.4e}  "
                      f"{s['epsilon']:>10.1e}  {conv_mark:>6s}  "
                      f"{s['mean']:>12.6f}  {s['cv']:>10.4e}  "
                      f"{fill:>12s}  {role}")
        
        print("=" * 78)
    
    def get_info(self) -> str:
        """One-line info string."""
        if not self.enabled:
            return "Convergence monitor: disabled"
        method = "Cauchy"
        if self.has_obstacle:
            return (f"Convergence: Path B (Cd {method}), "
                    f"window={self._window_samples} samples, "
                    f"n_required={self.Cd_tracker.n_required}")
        return (f"Convergence: Path A (Energy {method}), "
                f"window={self._window_samples} samples, "
                f"n_required={self.energy_tracker.n_required}")
    
    def __bool__(self) -> bool:
        return self.enabled
    
    def close(self) -> None:
        """Close CSV file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
    
    def __del__(self):
        self.close()