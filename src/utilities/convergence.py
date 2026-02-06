"""
Convergence Detection Module for LBM Solver

Monitors simulation convergence using within-window coefficient of variation (CV),
following Palabos (ValueTracer) and SU2 (CONV_CAUCHY) approaches.

Convergence Criterion:
======================
    CV = σ(Q) / |μ(Q)| < ε    within a rolling window of N samples

Convergence Paths:
==================
    Path A (no obstacle): Energy is the sole criterion (Palabos style)
        - CV(E_avg) < ε_energy         [criterion]

    Path B (with obstacle): Cd is the sole criterion (SU2 style)
        - CV(Cd) < ε_Cd                [criterion]
        - Energy, Cl: logged only       [monitor]
        - Rationale: Cd > 0 always → CV well-defined.
          Cl oscillates near zero → CV ill-defined (excluded).
          Energy is a domain average → less sensitive than force.

Window Sizing:
==============
    Window sizes are computed in SAMPLES, accounting for feed intervals:

        T_conv = D / U  [timesteps]
        window_timesteps = T_conv × time_coverage  [timesteps]
        window_samples   = window_timesteps / feed_interval  [samples]

    Example (D=20, U=0.1, force_interval=10):
        T_conv = 200 steps
        time_coverage = 50 (default) → window_timesteps = 10,000
        energy: 10,000 / 10 = 1000 samples → fills at step 10,000
        force:  10,000 / 10 = 1000 samples → fills at step 10,000

Cd Threshold (ε_Cd) Guidance:
============================
    Steady flow (low Re):     CV → 0,      ε_Cd = 1e-3 ~ 1e-2
    Periodic flow (shedding): CV ≈ 0.01,   ε_Cd = 0.02 ~ 0.05
    Default: 0.02 (targets periodic external aerodynamics)

IMPORTANT - Feed Intervals:
===========================
    Energy and Force trackers are both fed at `force_interval` in main.py.
    Convergence CHECK (evaluate & log) is done at `check_interval`.
    This separation ensures buffers fill at the correct rate.

    main.py data flow:
        every force_interval:  feed_energy(), feed_force()
        every check_interval:  feed_divergence_check(), check()

References:
==========
    - Palabos: src/core/util.h → ValueTracer
    - SU2: CONV_CAUCHY criterion
    - OpenLB: src/utilities/valueTracer.h

Author: LBM Development Team
Date: 2026-02
"""

import os
import csv
from enum import Enum
from typing import Optional, Dict, Any
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
        self._data[self._index] = value
        self._index = (self._index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    
    @property
    def is_full(self) -> bool:
        return self.count >= self.capacity
    
    @property
    def values(self) -> np.ndarray:
        if self.count < self.capacity:
            return self._data[:self.count].copy()
        return np.roll(self._data, -self._index).copy()
    
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        if self.count < self.capacity:
            return float(np.mean(self._data[:self.count]))
        return float(np.mean(self._data))
    
    def std(self) -> float:
        if self.count < 2:
            return float('inf')
        if self.count < self.capacity:
            return float(np.std(self._data[:self.count]))
        return float(np.std(self._data))
    
    def cv(self) -> float:
        """CV = σ / |μ|  [dimensionless]"""
        if self.count < 2:
            return float('inf')
        mu = self.mean()
        sigma = self.std()
        if abs(mu) < 1e-30:
            return float('inf') if sigma > 1e-30 else 0.0
        return sigma / abs(mu)
    
    def reset(self) -> None:
        self._data[:] = 0.0
        self._index = 0
        self.count = 0


# =============================================================================
# Scalar Tracker
# =============================================================================

class ScalarTracker:
    """Tracks convergence of a scalar via CV = σ/|μ| < ε
    
    Attributes:
        name: Identifier (e.g., 'Cd', 'avg_energy')
        epsilon: CV threshold  [dimensionless]
        window_size: Buffer capacity  [samples]
        is_criterion: If True, participates in convergence decision
        feed_interval: Feed rate  [timesteps/sample]
    """
    
    def __init__(self, name: str, epsilon: float, window_size: int,
                 is_criterion: bool = True,
                 feed_interval: int = 1) -> None:
        self.name: str = name
        self.epsilon: float = epsilon
        self.window_size: int = window_size
        self.is_criterion: bool = is_criterion
        self.feed_interval: int = feed_interval
        self.buffer: RingBuffer = RingBuffer(window_size)
        self._last_cv: float = float('inf')
        self._converged: bool = False
    
    def push(self, value: float) -> None:
        self.buffer.push(value)
        if self.buffer.is_full:
            self._last_cv = self.buffer.cv()
            self._converged = self._last_cv < self.epsilon
    
    @property
    def cv(self) -> float:
        return self._last_cv
    
    def has_converged(self) -> bool:
        return self._converged
    
    @property
    def fill_steps(self) -> int:
        """Estimated timesteps to fill buffer  [timesteps]"""
        return self.window_size * self.feed_interval
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'cv': self._last_cv,
            'epsilon': self.epsilon,
            'converged': self._converged,
            'is_criterion': self.is_criterion,
            'samples': self.buffer.count,
            'window_size': self.window_size,
            'fill_ratio': self.buffer.count / self.window_size,
            'mean': self.buffer.mean(),
            'std': self.buffer.std(),
            'feed_interval': self.feed_interval,
        }
    
    def reset(self) -> None:
        self.buffer.reset()
        self._last_cv = float('inf')
        self._converged = False


# =============================================================================
# Auto Window Size Calculator
# =============================================================================

def compute_auto_window_sizes(
    char_length: float,
    u_ref: float,
    feed_interval: int = 10,
    time_coverage: float = 50.0,
    min_samples: int = 50,
) -> Dict[str, Any]:
    """Compute window sizes in samples, accounting for feed rate
    
    Both energy and force trackers are fed at the same `feed_interval`
    (= force_interval), so they share the same window size.
    
    T_conv = D / U  [timesteps]
    window_timesteps = T_conv × time_coverage  [timesteps]
    window_samples = window_timesteps / feed_interval  [samples]
    
    Default time_coverage=50:
        For St≈0.2 → T_shed ≈ 5 T_conv → window ≈ 10 shedding periods
    
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
        - CV(E_avg) < ε_energy   [criterion]
    
    Path B (with obstacle): Cd-based (SU2 style)
        - CV(Cd) < ε_Cd          [criterion]
        - Energy                  [monitor]
        - Cl                      [monitor]
    
    Feed & Check Separation:
    ========================
    All trackers are FED at `feed_interval` (= force_interval in main.py).
    Convergence is CHECKED at `check_interval` (less frequently).
    
    This ensures buffers fill at the correct rate regardless of check frequency.
    
    main.py integration:
        every feed_interval:   feed_energy(step, rho, u)
                               feed_force(step, Cd, Cl)    # Path B only
        every check_interval:  feed_divergence_check(rho, u)
                               status = check(step)
    
    Cd Threshold for Periodic Flows:
    ================================
    For vortex shedding (Re > ~47 for cylinder):
        Cd oscillates → CV(Cd) ≈ σ_oscillation / Cd_mean ≈ 0.01
        Default ε_Cd = 0.02 accommodates this
    
    For steady flow (low Re):
        Cd stabilizes → CV → 0
        Use ε_Cd = 1e-3 for stricter check
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
        self.energy_tracker: Optional[ScalarTracker] = None
        self.Cd_tracker: Optional[ScalarTracker] = None
        self.Cl_tracker: Optional[ScalarTracker] = None
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
        
        stat_config = self.config.get('statistical', {})
        window_cfg = stat_config.get('window_size', 'auto')
        
        # --- Window size ---
        if window_cfg == 'auto':
            auto_cfg = stat_config.get('auto_window', {})
            sizes = compute_auto_window_sizes(
                char_length=char_length,
                u_ref=u_ref,
                feed_interval=feed_interval,
                time_coverage=auto_cfg.get('time_coverage', 50.0),
                min_samples=auto_cfg.get('min_samples', 50),
            )
            window_samples = sizes['window_samples']
            self._T_conv = sizes['T_conv']
        else:
            window_samples = int(window_cfg)
            self._T_conv = char_length / max(u_ref, 1e-15)
        
        self._window_samples = window_samples
        self._feed_interval = feed_interval
        
        # --- Epsilon defaults ---
        # Cd: 0.02 = suitable for periodic flows (CV_natural ≈ 0.01)
        # Energy: 1e-4 = domain average converges tightly
        epsilon_energy = stat_config.get('epsilon', 1e-4)
        epsilon_Cd = stat_config.get('Cd_epsilon', 0.02)
        
        # --- Create trackers ---
        if self.has_obstacle:
            # Path B: Cd is criterion, energy & Cl are monitoring
            self.energy_tracker = ScalarTracker(
                name='avg_energy',
                epsilon=epsilon_energy,
                window_size=window_samples,
                is_criterion=False,          # monitoring only
                feed_interval=feed_interval,
            )
            self.Cd_tracker = ScalarTracker(
                name='Cd',
                epsilon=epsilon_Cd,
                window_size=window_samples,
                is_criterion=True,           # sole convergence criterion
                feed_interval=feed_interval,
            )
            self.Cl_tracker = ScalarTracker(
                name='Cl',
                epsilon=epsilon_Cd,
                window_size=window_samples,
                is_criterion=False,          # monitoring only
                feed_interval=feed_interval,
            )
        else:
            # Path A: Energy is the sole criterion
            self.energy_tracker = ScalarTracker(
                name='avg_energy',
                epsilon=epsilon_energy,
                window_size=window_samples,
                is_criterion=True,
                feed_interval=feed_interval,
            )
        
        self._status = ConvergenceStatus.RUNNING
        self._init_csv_logger()
        self._print_config(char_length, u_ref, feed_interval, window_samples)
    
    def _print_config(self, char_length: float, u_ref: float,
                      feed_interval: int, window_samples: int) -> None:
        path = "B (Cd-based, SU2)" if self.has_obstacle else "A (Energy-based, Palabos)"
        fill_steps = window_samples * feed_interval
        
        print(f"\n  Convergence Monitor: ENABLED — Path {path}")
        print(f"    T_conv = D/U = {char_length}/{u_ref} = {self._T_conv:.1f} steps")
        print(f"    Window = {window_samples} samples × {feed_interval} steps "
              f"= {fill_steps:,} steps to fill")
        
        for tracker in [self.energy_tracker, self.Cd_tracker, self.Cl_tracker]:
            if tracker is not None:
                role = "criterion" if tracker.is_criterion else "monitor "
                print(f"    {tracker.name:>12s}: ε = {tracker.epsilon:.1e}  [{role}]")
        
        print(f"    Actions: converged → {self.on_converged}, "
              f"diverged → {self.on_diverged}")
    
    def _init_csv_logger(self) -> None:
        os.makedirs(self.csv_dir, exist_ok=True)
        csv_path = os.path.join(self.csv_dir, 'convergence_history.csv')
        self._csv_file = open(csv_path, 'w', newline='')
        
        headers = ['step', 'energy_cv', 'energy_mean']
        if self.has_obstacle:
            headers += ['Cd_cv', 'Cd_mean', 'Cd_converged', 'Cl_cv', 'Cl_mean']
        headers.append('all_converged')
        
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=headers)
        self._csv_writer.writeheader()
    
    # -------------------------------------------------------------------------
    # Data Feeding (called at feed_interval)
    # -------------------------------------------------------------------------
    
    def feed_energy(self, step: int, rho: Any, u: Any) -> None:
        """Feed domain-averaged kinetic energy: E = (1/N) Σ 0.5ρ|u|²  [Δx²/Δt²]
        
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
    
    def feed_divergence_check(self, rho: Any, u: Optional[Any] = None) -> Dict[str, Any]:
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
        """Evaluate convergence: all criterion trackers must have CV < ε"""
        if not self.enabled:
            return {'converged': False, 'diverged': False,
                    'action': 'continue', 'trackers': {}}
        
        self._check_count += 1
        
        trackers = {}
        all_criteria_met = True
        
        for tracker in [self.energy_tracker, self.Cd_tracker, self.Cl_tracker]:
            if tracker is not None:
                status = tracker.get_status()
                trackers[tracker.name] = status
                if tracker.is_criterion and not status['converged']:
                    all_criteria_met = False
        
        if self._status == ConvergenceStatus.DIVERGED:
            action = self.on_diverged
        elif all_criteria_met:
            self._status = ConvergenceStatus.CONVERGED
            self._converged_step = step
            action = self.on_converged
        else:
            action = 'continue'
        
        self._log_csv(step, trackers, all_criteria_met)
        
        return {
            'converged': all_criteria_met and self._status == ConvergenceStatus.CONVERGED,
            'diverged': self._status == ConvergenceStatus.DIVERGED,
            'diverge_reason': self._diverge_reason,
            'action': action,
            'trackers': trackers,
        }
    
    def _log_csv(self, step: int, trackers: Dict, converged: bool) -> None:
        if self._csv_writer is None:
            return
        row = {'step': step, 'all_converged': converged}
        if 'avg_energy' in trackers:
            t = trackers['avg_energy']
            row['energy_cv'] = f"{t['cv']:.6e}"
            row['energy_mean'] = f"{t['mean']:.6e}"
        if 'Cd' in trackers:
            t = trackers['Cd']
            row['Cd_cv'] = f"{t['cv']:.6e}"
            row['Cd_mean'] = f"{t['mean']:.6e}"
            row['Cd_converged'] = t['converged']
        if 'Cl' in trackers:
            t = trackers['Cl']
            row['Cl_cv'] = f"{t['cv']:.6e}"
            row['Cl_mean'] = f"{t['mean']:.6e}"
        self._csv_writer.writerow(row)
        self._csv_file.flush()
    
    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    
    def get_status(self) -> ConvergenceStatus:
        if not self.enabled:
            return ConvergenceStatus.DISABLED
        return self._status
    
    @property
    def converged(self) -> bool:
        return self._status == ConvergenceStatus.CONVERGED
    
    @property
    def diverged(self) -> bool:
        return self._status == ConvergenceStatus.DIVERGED
    
    @property
    def converged_step(self) -> Optional[int]:
        return self._converged_step
    
    def get_status_string(self, step: int) -> str:
        """One-line status for optional verbose use"""
        parts = []
        for tracker in [self.energy_tracker, self.Cd_tracker]:
            if tracker is not None and tracker.buffer.is_full:
                cv = tracker.cv
                mark = '✓' if tracker.has_converged() else ''
                label = 'E' if tracker.name == 'avg_energy' else tracker.name
                parts.append(f"{label}={cv:.1e}{mark}")
        if not parts:
            return "filling buffers..."
        return " | ".join(parts)
    
    def mark_max_steps(self) -> None:
        if self._status == ConvergenceStatus.RUNNING:
            self._status = ConvergenceStatus.MAX_STEPS
    
    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print(" Convergence Summary")
        print("=" * 70)
        
        if not self.enabled:
            print("  Convergence monitor: disabled")
            return
        
        status_str = {
            ConvergenceStatus.CONVERGED: "CONVERGED ✓",
            ConvergenceStatus.DIVERGED: f"DIVERGED ✗ ({self._diverge_reason})",
            ConvergenceStatus.RUNNING: "NOT CONVERGED (still running)",
            ConvergenceStatus.MAX_STEPS: "NOT CONVERGED (max steps reached)",
            ConvergenceStatus.NOT_STARTED: "NOT STARTED",
        }.get(self._status, str(self._status))
        
        print(f"  Status: {status_str}")
        if self._converged_step is not None:
            print(f"  Converged at step: {self._converged_step}")
        print(f"  Checks performed: {self._check_count}")
        
        for tracker in [self.energy_tracker, self.Cd_tracker, self.Cl_tracker]:
            if tracker is not None:
                s = tracker.get_status()
                role = "[criterion]" if s['is_criterion'] else "[monitor] "
                conv_mark = '✓' if s['converged'] else '✗'
                fill = f"{s['samples']}/{s['window_size']}"
                print(f"  {s['name']:>12s}: CV = {s['cv']:.4e} "
                      f"(ε = {s['epsilon']:.1e}) [{conv_mark}]  "
                      f"μ = {s['mean']:.6f}, σ = {s['std']:.6e}  "
                      f"({fill} samples)  {role}")
        
        print("=" * 70)
    
    def get_info(self) -> str:
        if not self.enabled:
            return "Convergence monitor: disabled"
        if self.has_obstacle:
            return (f"Convergence: Path B (Cd criterion), "
                    f"window={self._window_samples} samples")
        return (f"Convergence: Path A (Energy criterion), "
                f"window={self._window_samples} samples")
    
    def __bool__(self) -> bool:
        return self.enabled
    
    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
    
    def __del__(self):
        self.close()