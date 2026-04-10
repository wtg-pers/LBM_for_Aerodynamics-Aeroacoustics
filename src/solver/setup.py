"""
Simulation Setup — Environment Construction

This module constructs the complete simulation environment from CLI args
and config files. It answers the question:
    "어떤 도구로, 어떤 공간에서, 어떤 조건으로?"

Responsibilities (Layer 1 of the 3-Layer architecture):
    [0] Config loading, device selection, lattice validation
    [1] Domain setup
    [2] Physics parameters extraction
    [3] Boundary conditions (domain + internal obstacle)
    [4] I/O setup (directories, VTK, checkpoint, CSV)
    [5.1-5.4] Monitors & ALM (conservation, force, convergence, ALM)
    [6] Multi-Level Grid (conditional)
    LBM component creation (streaming, equilibrium, macro, collision, forcing)

Produces:
    build_simulation()      → Simulation or MultiLevelGrid (if MLG enabled)
    build_output_manager()  → OutputManager

Does NOT handle (→ SolverInitializer, M4):
    - Initial distribution function (fresh start or checkpoint restore)
    - Solver-mode-specific initialization (BGK vs Cumulant)

Design Principle:
    "솔버 모드에 따라 달라지는가?" → Yes: Initializer, No: Setup

Author: LBM Development Team
Date: 2026-03 (MLG integration: 2026-04)
"""

import os
import sys
import io
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType

from src.lattice import get_lattice
from src.macroscopic.compute import Macroscopic
from src.collision.bgk import BGKCollision
from src.streaming.stream import StreamingPull

from src.io.config_loader import ConfigLoader
from src.io.vtk_writer import VTKWriter
from src.io.checkpoint import CheckpointManager

from src.boundary.domain_bc_manager import DomainBCManager
from src.boundary.wall import HalfwayBounceBack
from src.boundary.geometry_manager import (
    create_geometry_mask, validate_geometry_config,
    create_fine_level_geometry_config,
)

from src.utilities.device import setup_library
from src.utilities.lattice_validation import LatticeValidator
from src.utilities.directory_utils import setup_output_directories
from src.utilities.flux_utils import ConservationManager
from src.utilities.force_calculator import ForceManager
from src.utilities.convergence import ConvergenceMonitor

from src.solver.simulation import Simulation
from src.solver.output_manager import OutputManager

# ── MLG (Multi-Level Grid) imports ───────────────────────────────
from src.grid.multi_level_grid import MultiLevelGrid
from src.grid.coupling import GridCoupling
from src.grid.overlap_manager import OverlapManager, IndexBox
from src.grid.level_scaling import LevelScaler
from src.grid.interpolation import CubicInterpolation, CompactSecondOrderInterpolation


class SimulationSetup:
    """Construct the complete simulation environment from CLI args + config.

    This class orchestrates the creation of ALL objects needed to run a
    simulation. After construction, call build_simulation() and
    build_output_manager() to obtain the two main execution objects.

    Usage:
        >>> setup = SimulationSetup(args)
        >>> sim = setup.build_simulation()
        >>> output = setup.build_output_manager()

    Attributes:
        xp: Array module (numpy or cupy)
        config: Full config dict from config file
        lattice: Lattice model (D2Q9, D3Q27, etc.)
        domain_shape: Grid dimensions (Nx, Ny[, Nz])  [lu]
        tau: Relaxation time  [Δt]
        collision: Collision operator (owns equilibrium + forcing)
        checkpoint_mgr: Checkpoint manager (for initializer to use)
    """

    def __init__(self, args: Any) -> None:
        """Build entire simulation environment from CLI args.

        Default: detailed log → file only, compact summary → terminal.
        Use --verbose to also echo detailed log to terminal.
        """
        self._args = args
        self._verbose = getattr(args, 'verbose', False)

        # ── Capture detailed output to buffer ────────────────────
        self._log_buffer = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._LogTee(
            self._log_buffer, self._original_stdout,
            echo=self._verbose,
        )

        try:
            # ── [0] Configuration Loading ────────────────────────
            self._load_config()

            # ── Device + Lattice ─────────────────────────────────
            self._setup_device_and_lattice()

            # ── [1] Domain ───────────────────────────────────────
            self._setup_domain()

            # ── [2] Physics Parameters ───────────────────────────
            self._extract_physics()

            # ── [3] Boundary Conditions ──────────────────────────
            self._setup_boundaries()

            # ── [4] I/O (directories, writers, CSV) ──────────────
            self._setup_io()

            # ── LBM Components ───────────────────────────────────
            self._create_lbm_components()

            # ── [5.1] Conservation Monitor ───────────────────────
            self._setup_conservation()

            # ── [5.2] Force Calculation ──────────────────────────
            self._setup_force_calculation()

            # ── [5.3] Convergence Monitor ────────────────────────
            self._setup_convergence()

            # ── [5.4] Actuator Line Model ────────────────────────
            self._setup_actuator_line()

            # ── [6] Multi-Level Grid (conditional) ───────────────
            self._setup_mlg()

        finally:
            sys.stdout = self._original_stdout

        # ── Write detailed log to file ───────────────────────────
        log_dir = getattr(self, '_csv_dir', './results/csv')
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, 'setup_log.txt')
        with open(self._log_path, 'w', encoding='utf-8') as f:
            f.write(self._log_buffer.getvalue())

        # ── Print compact summary to terminal ────────────────────
        self._print_summary()

    # =====================================================================
    # Internal: Log capture utility
    # =====================================================================

    class _LogTee:
        """Writes to buffer and optionally echoes to terminal."""

        def __init__(self, buffer, terminal, echo: bool = False):
            self._buffer = buffer
            self._terminal = terminal
            self._echo = echo

        def write(self, text):
            self._buffer.write(text)
            if self._echo:
                self._terminal.write(text)

        def flush(self):
            self._buffer.flush()
            if self._echo:
                self._terminal.flush()

    def start_log_capture(self) -> None:
        """Redirect stdout to log buffer (call before build/init)."""
        self._original_stdout = sys.stdout
        sys.stdout = self._LogTee(
            self._log_buffer, self._original_stdout,
            echo=self._verbose,
        )

    def stop_log_capture(self) -> None:
        """Restore stdout and update log file."""
        sys.stdout = self._original_stdout
        with open(self._log_path, 'w', encoding='utf-8') as f:
            f.write(self._log_buffer.getvalue())

    # =====================================================================
    # Public: Build products
    # =====================================================================

    def build_simulation(self):
        """Create Simulation or MultiLevelGrid (if MLG enabled).

        If MLG is configured in the config file, returns a MultiLevelGrid
        that wraps multiple Simulation objects with nested time-stepping.
        Otherwise, returns a single Simulation as before.

        The returned object has set_distribution() NOT called yet.
        The caller (or SolverInitializer) must initialize f before advance().

        Returns:
            Simulation or MultiLevelGrid (both support advance(), rho, u, f)
        """
        if self._mlg_enabled:
            return self._build_mlg_simulation()

        return Simulation(
            xp=self.xp,
            macroscopic=self.macro,
            collision=self.collision,
            streaming=self.streaming,
            bc_manager=self.domain_bc_mgr,
            tau=self.tau,
            domain_shape=self.domain_shape,
            obstacle_bc=self.obstacle_bc,
            al_model=self.al_model,
        )

    def build_output_manager(self) -> 'OutputManager':
        """Create OutputManager with all I/O components.

        Returns:
            OutputManager ready for start() → process() → finalize()
        """
        return OutputManager(
            xp=self.xp,
            macroscopic=self.macro,
            lattice=self.lattice,
            sim_params=self.sim_params,
            vtk_writer=self.vtk_writer,
            marker_vtk_writer=self.marker_vtk_writer,
            checkpoint_mgr=self.checkpoint_mgr,
            conservation_mgr=self.conservation_mgr,
            force_mgr=self.force_mgr,
            conv_monitor=self.conv_monitor,
            al_model=self.al_model,
            output_interval=self.output_interval,
            check_interval=self.check_interval,
            checkpoint_interval=self.checkpoint_interval,
            tau=self.tau,
            solid_mask_np=self.solid_mask_np,
            perf_csv_path=self.perf_csv_path,
            domain_shape=self.domain_shape,
            L_ref_lu=self.L_ref_lu,
            u_ref_lu=self.u_ref_lu,
            config_path=self._args.config,
            mlg_vtk_writer=self._mlg_vtk_writer if self._mlg_enabled else None,
            mlg_force_level=getattr(self, '_mlg_force_level', None),
        )

    # =====================================================================
    # Private: Setup steps [0]–[5.4] (existing, unchanged)
    # =====================================================================

    def _load_config(self) -> None:
        """[0] Load config file and extract sub-configs."""
        self._config_loader = ConfigLoader(self._args.config)
        self.config = self._config_loader.config

        self.sim_params = self._config_loader.get_simulation_params()
        self._domain_config = self.sim_params.get('domain', {})
        self._physics_config = self.sim_params.get('physics', {})
        self._time_config = self.sim_params.get('time', {})

        self._output_config = self.config.get('output', {})
        self._vtk_config = self._output_config.get('vtk', {})
        self._checkpoint_config = self._output_config.get('checkpoint', {})
        self._conservation_config = self.config.get('conservation', {})
        self._force_config = self.config.get('force_calculation', {})
        self._conv_config = self.config.get('convergence', {})

        self._al_cfg = self.config.get('actuator_line', {})
        self.al_enabled: bool = self._al_cfg.get('enabled', False)

    def _setup_device_and_lattice(self) -> None:
        """Device selection + lattice creation + validation."""
        import numpy as np

        device_mode = self.sim_params.get('device_mode')
        device_id = (self._args.gpu
                     if self._args.gpu is not None
                     else self.sim_params.get('device_id', 0))

        self.xp: 'ModuleType' = setup_library(device_mode, device_id=device_id)

        # ── Computation precision ────────────────────────────────
        precision_str = self.sim_params.get('precision', 'float64')
        _precision_map = {'float32': np.float32, 'float64': np.float64}
        if precision_str not in _precision_map:
            raise ValueError(
                f"Unknown precision: '{precision_str}'. "
                f"Available: 'float32', 'float64'"
            )
        self.compute_dtype = _precision_map[precision_str]
        print(f"\n  Computation precision: {precision_str}")

        lattice_model = self.sim_params.get('lattice_model', 'D3Q27')
        self.lattice = get_lattice(lattice_model, self.xp, dtype=self.compute_dtype)
        self._dimension = self.sim_params.get('dimension')

        # ── Lattice validation (optional) ────────────────────────
        # Skipped by default for known models. Enable via config:
        #   "simulation": {"validate_lattice": true}
        if self.sim_params.get('validate_lattice', False):
            import numpy as _np
            print(f"\n[0] Validating Lattice Model ({lattice_model})...")
            _val_lattice = get_lattice(lattice_model, _np, dtype=_np.float64)
            validator = LatticeValidator(self.xp)
            is_valid, _ = validator.validate_all(
                _val_lattice.c, _val_lattice.w, _val_lattice.cs2, verbose=True,
            )
            if not is_valid:
                raise RuntimeError("Lattice validation failed!")
        else:
            print(f"\n[0] Lattice: {lattice_model} (precision: {precision_str})")

    def _setup_domain(self) -> None:
        """[1] Domain setup."""
        self.Nx = self._domain_config.get('Nx')
        self.Ny = self._domain_config.get('Ny')
        self.Nz = self._domain_config.get('Nz')  # None for 2D

        if self.lattice.dim == 2:
            self.domain_shape: Tuple[int, ...] = (self.Nx, self.Ny)
            print(f"\n[1] Domain Setup (2D)")
            print(f"  Grid: {self.Nx} x {self.Ny}")
            print(f"  Total cells: {self.Nx * self.Ny:,}")
        else:
            self.domain_shape = (self.Nx, self.Ny, self.Nz)
            print(f"\n[1] Domain Setup (3D)")
            print(f"  Grid: {self.Nx} x {self.Ny} x {self.Nz}")
            print(f"  Total cells: {self.Nx * self.Ny * self.Nz:,}")

    def _extract_physics(self) -> None:
        """[2] Extract physics parameters from config."""
        pc = self._physics_config

        self.Re = pc.get('Re')
        self.u_ref_lu: Optional[float] = pc.get('u_ref_lu')  # [Δx/Δt]
        self.L_ref_lu: Optional[float] = pc.get('L_ref_lu')  # [Δx]
        self.u_ref_phys = pc.get('U_ref')                     # [m/s]
        self.nu_lu: float = pc.get('nu_lu')                    # [Δx²/Δt]
        self.tau: float = pc.get('tau')                        # [Δt]

        tc = self._time_config
        fc = self._force_config
        self.config_max_steps: int = tc.get('max_steps', 10000)
        self.output_interval: int = tc.get('output_interval', 500)
        self.checkpoint_interval: int = tc.get('checkpoint_interval', 2000)
        self._force_interval: int = fc.get('interval', 10)

        print(f"\n[2] Physics Parameters")
        print(f"  Re = {self.Re}")
        print(f"  u_ref_lu = {self.u_ref_lu} [Δx/Δt], "
              f"U_ref = {self.u_ref_phys} [m/s]")
        print(f"  L_ref_lu = {self.L_ref_lu} [Δx]")
        print(f"  ν = {self.nu_lu:.6f} [Δx²/Δt], τ = {self.tau:.6f}")

        # Physical unit conversion (needed for Actuator Line)
        self.dx_phys: Optional[float] = None   # [m/lu]
        self.dt_phys: Optional[float] = None   # [s/lt]
        if self.al_enabled:
            units_cfg = self._al_cfg['units']
            self.dx_phys = units_cfg['dx_phys']
            self.dt_phys = units_cfg['dt_phys']
            print(f"  Δx = {self.dx_phys * 1000:.2f} mm, "
                  f"Δt = {self.dt_phys * 1e6:.2f} μs")

    def _setup_boundaries(self) -> None:
        """[3] Domain BC + internal obstacle."""
        print(f"\n[3] Domain Boundary Conditions")

        boundaries_config = self.config.get('boundaries', {})
        self.domain_bc_mgr = DomainBCManager(
            xp=self.xp,
            lattice=self.lattice,
            boundaries_config=boundaries_config,
            domain_shape=self.domain_shape,
            verbose=True,
        )

        # Internal obstacle
        internal_geom = self.config.get('internal_geometry', {})
        is_valid, msg = validate_geometry_config(
            internal_geom, self.domain_shape, self.lattice.dim,
        )
        if not is_valid:
            raise ValueError(f"Invalid geometry configuration: {msg}")

        self._mask, geom_info = create_geometry_mask(
            self.xp, self.lattice, self.domain_shape,
            internal_geom,
            characteristic_length=self.L_ref_lu,
            verbose=True,
        )

        if geom_info['type'] != 'none':
            self.obstacle_bc = HalfwayBounceBack(
                self.xp, self.lattice, self._mask,
            )
        else:
            self.obstacle_bc = None

        self.solid_mask_np = (
            self._mask.get() if hasattr(self._mask, 'get') else self._mask
        )

    def _setup_io(self) -> None:
        """[4] I/O directories, VTK, checkpoint, rotor CSV."""
        print(f"\n[4] I/O Setup")

        args = self._args
        oc = self._output_config

        output_dir = args.output_dir or oc.get('output_dir', './results/vtk')
        self.checkpoint_dir = (
            args.checkpoint_dir or oc.get('checkpoint_dir', './checkpoints')
        )
        self._csv_dir = args.csv_dir or oc.get('csv_dir', './results/csv')

        is_restart = args.restart_latest or args.restart is not None
        clear_previous = args.clear or oc.get('clear_previous', False)

        print(f"  VTK output dir: {output_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  CSV output dir: {self._csv_dir}")

        setup_output_directories(
            output_dir=output_dir,
            checkpoint_dir=self.checkpoint_dir,
            csv_dir=self._csv_dir,
            clear_previous=clear_previous,
            is_restart=is_restart,
        )

        # ── VTK Writer ──
        vc = self._vtk_config
        vtk_enabled = vc.get('enabled', True) and not args.no_vtk
        if vtk_enabled:
            self.vtk_writer = VTKWriter(
                output_dir=output_dir,
                domain_shape=self.domain_shape,
                precision=vc.get('precision', 'float32'),
                compression_level=vc.get('compression_level', 0),
            )
            size_est = self.vtk_writer.get_file_size_estimate()
            print(f"  VTK: enabled ({size_est['estimated_MB']:.2f} MB/file)")
        else:
            self.vtk_writer = None
            print("  VTK: disabled")

        # ── Marker VTP Writer ──
        self.marker_vtk_writer = None
        if vtk_enabled and self.al_enabled:
            from src.io.marker_vtk_writer import MarkerVTPWriter
            marker_dir = os.path.join(output_dir, 'markers')
            self.marker_vtk_writer = MarkerVTPWriter(
                output_dir=marker_dir,
                precision=vc.get('precision', 'float32'),
            )
            print(f"  Marker VTP: enabled ({marker_dir})")

        # ── Checkpoint Manager ──
        cc = self._checkpoint_config
        checkpoint_enabled = cc.get('enabled', True)
        if checkpoint_enabled:
            self.checkpoint_mgr = CheckpointManager(
                output_dir=self.checkpoint_dir,
                prefix='checkpoint',
                keep_last_n=cc.get('keep_last_n', 3),
                xp=self.xp,
            )
            if self._dimension == 2:
                est = self.checkpoint_mgr.get_size_estimate(
                    (self.lattice.Q, self.Nx, self.Ny),
                )
            else:
                est = self.checkpoint_mgr.get_size_estimate(
                    (self.lattice.Q, self.Nx, self.Ny, self.Nz),
                )
            print(f"  Checkpoint: enabled ({est['estimated_MB']:.2f} MB/file)")
        else:
            self.checkpoint_mgr = None
            print("  Checkpoint: disabled")

        # ── Rotor CSV ──
        # CSV is opened by SolverInitializer after start_step is known,
        # so that restart preserves existing data.
        self.perf_csv_path: Optional[str] = None
        self._perf_csv_header = (
            'step,time_lt,time_phys,revolutions,'
            'thrust_lu,torque_lu,power_lu,'
            'C_T,C_P,coeff_mode,u_inf_used,FM\n'
        )
        if self.al_enabled:
            self.perf_csv_path = os.path.join(
                self._csv_dir, 'rotor_performance.csv',
            )
            print(f"  Rotor CSV: {self.perf_csv_path} (opened at init)")

    def _create_lbm_components(self) -> None:
        """Create core LBM operator objects.
    
        Collision Model Selection:
            Reads 'collision_model' from simulation config.
            - "bgk" (default): BGKCollision — works with D2Q9, D3Q27
            - "cumulant":       CumulantCollision — D3Q27 only
    
            Additional cumulant parameters (optional):
            - "omega_bulk": Bulk viscosity rate ω₂  [1/Δt] (default 1.0)
            - "omega_high": High-order rate ω₃-ω₁₀  [1/Δt] (default 1.0)
    
        Config Example:
            "simulation": {
                "collision_model": "cumulant",   # or "bgk"
                "omega_bulk": 1.0,               # optional
                "omega_high": 1.0,               # optional
                ...
            }
        """
        self.streaming = StreamingPull(
            self.xp, self.lattice, self.domain_shape,
        )
        self.macro = Macroscopic(self.xp, self.lattice)
    
        # ── Collision model selection ────────────────────────────
        model_name = self.sim_params.get('collision_model', 'bgk').lower()
    
        if model_name == 'cumulant':
            from src.collision.cumulant import CumulantCollision
            omega_bulk = self.sim_params.get('omega_bulk', 1.0)
            omega_high = self.sim_params.get('omega_high', 1.0)
            self.collision = CumulantCollision(
                self.xp, self.lattice,
                omega_bulk=omega_bulk,
                omega_high=omega_high,
            )
            print(f"  Collision: Cumulant (ω_bulk={omega_bulk}, ω_high={omega_high})")
        elif model_name == 'bgk':
            self.collision = BGKCollision(self.xp, self.lattice)
            print(f"  Collision: BGK")
        else:
            raise ValueError(
                f"Unknown collision model: '{model_name}'. "
                f"Available: 'bgk', 'cumulant'"
            )

    def _setup_conservation(self) -> None:
        """[5.1] Conservation check setup.

        Note: conservation_mgr.initialize() is NOT called here because
        it requires the initial distribution function f, which is set
        by the Initializer (M4).
        """
        print(f"\n[5.1] Conservation Check Setup")

        self.check_interval: int = self._conservation_config.get(
            'check_interval', 0,
        )
        if self.check_interval == 0:
            self.check_interval = self.output_interval

        self.conservation_mgr = ConservationManager(
            xp=self.xp,
            domain_shape=self.domain_shape,
            config=self._conservation_config,
            solid_mask=self._mask,
            csv_dir=self._csv_dir,
        )
        print(f"  {self.conservation_mgr.get_info()}")

    def _setup_force_calculation(self) -> None:
        """[5.2] Force calculation (MEM) setup."""
        print(f"\n[5.2] Force Calculation Setup")

        fc = self._force_config
        force_enabled = (
            fc.get('enabled', False) and self.obstacle_bc is not None
        )

        if force_enabled and not self._args.no_force:
            ref_config = fc.get('reference', {})

            if self.lattice.dim == 2:
                default_span = 1       # 2D: unit span  [Δx]
            else:
                default_span = self.Nz  # [Δx]

            force_calc_config = {
                'enabled': True,
                'interval': self._force_interval,
                'start_step': fc.get('start_step', 0),
                'reference': {
                    'rho': ref_config.get('rho', 1.0),
                    'velocity': ref_config.get('velocity', self.u_ref_lu),
                    'char_length': ref_config.get(
                        'char_length', self.L_ref_lu,
                    ),
                    'span_length': ref_config.get(
                        'span_length', default_span,
                    ),
                },
                'log': {'enabled': True, 'filename': 'force_history'},
            }

            self.force_mgr = ForceManager(
                xp=self.xp,
                lattice=self.lattice,
                solid_mask=self._mask,
                config=force_calc_config,
                wall_bc=self.obstacle_bc,
                csv_dir=self._csv_dir,
            )
            self.force_mgr.initialize()
        else:
            self.force_mgr = None
            if self.obstacle_bc is None:
                print("  Force calculation: disabled (no obstacle)")
            elif self._args.no_force:
                print("  Force calculation: disabled (--no-force)")
            else:
                print("  Force calculation: disabled (config)")

    def _setup_convergence(self) -> None:
        """[5.3] Convergence monitor setup."""
        print(f"\n[5.3] Convergence Monitor Setup")

        has_obstacle = (
            self.obstacle_bc is not None and self.force_mgr is not None
        )
        self.conv_monitor = ConvergenceMonitor(
            config=self._conv_config,
            has_obstacle=has_obstacle,
            csv_dir=self._csv_dir,
        )

        if self.conv_monitor.enabled:
            self.conv_monitor.initialize(
                char_length=self.L_ref_lu,
                u_ref=self.u_ref_lu,
            )
        else:
            print("  Convergence monitor: disabled")

    def _setup_actuator_line(self) -> None:
        """[5.4] Actuator Line Model (conditional)."""
        self.al_model = None
        self._polar_manager = None

        if not self.al_enabled:
            print(f"\n[5.4] Actuator Line: disabled")
            return

        from src.actuator.actuator_line import (
            create_actuator_line_from_config,
            create_multi_rotor_from_config,
        )
        from src.actuator.airfoil_data import create_polar_from_config

        print(f"\n[5.4] Actuator Line Model")

        # ── Load airfoil polar ──
        polar_config = self.config.get('airfoil_polar', {})
        if not polar_config:
            polar_config = {
                "method": "neuralfoil",
                "airfoil_name": "naca0012",
                "Re_target": 1e5,
                "mode": "asb",
            }

        method = polar_config.get('method', 'neuralfoil')
        print(f"  Airfoil polar method: '{method}'")

        polar_query, self._polar_manager = create_polar_from_config(
            polar_config,
        )

        # ── u_inf_lu ──
        pc = self._physics_config
        U_inf_phys = pc.get('U_inf', 0.0)                     # [m/s]
        u_inf_lu = U_inf_phys * self.dt_phys / self.dx_phys    # [Δx/Δt]
        u_inf_lu_arg = u_inf_lu if u_inf_lu > 0 else None

        al_cfg = self._al_cfg

        # ── Detect single vs multi rotor ──
        if 'rotors' in al_cfg:
            print(f"  Mode: MULTI-ROTOR ({len(al_cfg['rotors'])} rotors)")
            self.al_model = create_multi_rotor_from_config(
                config=al_cfg,
                domain_shape=self.domain_shape,
                nu_lattice=self.nu_lu,
                polar_query=polar_query,
                dx_phys=self.dx_phys,
                dt_phys=self.dt_phys,
                u_inf_lu=u_inf_lu_arg,
                coeff_mode=al_cfg.get('coeff_mode', 'auto'),
                xp=self.xp,
            )
            for i, (model, name) in enumerate(
                zip(self.al_model.models, self.al_model.names),
            ):
                print(f"    [{i}] {name}: "
                      f"hub={model.rotor.hub_center}, "
                      f"R={model.rotor.radius:.1f} lu, "
                      f"ω={model.rotor.omega:.6f} rad/lt")
        else:
            print(f"  Mode: SINGLE-ROTOR")
            self.al_model = create_actuator_line_from_config(
                config=al_cfg,
                domain_shape=self.domain_shape,
                nu_lattice=self.nu_lu,
                polar_query=polar_query,
                dx_phys=self.dx_phys,
                dt_phys=self.dt_phys,
                u_inf_lu=u_inf_lu_arg,
                coeff_mode=al_cfg.get('coeff_mode', 'auto'),
                xp=self.xp,
            )
            print(f"    Hub: {self.al_model.rotor.hub_center}")
            print(f"    R={self.al_model.rotor.radius:.1f} lu, "
                  f"ω={self.al_model.rotor.omega:.6f} rad/lt")
            print(f"    Blades: {self.al_model.rotor.n_blades}, "
                  f"Markers: {self.al_model.rotor.total_markers}")

    # =====================================================================
    # [6] Multi-Level Grid Setup (NEW)
    # =====================================================================

    def _setup_mlg(self) -> None:
        """[6] Parse MLG config and prepare multi-level components.

        Reads the 'mlg' section from config. If not present or disabled,
        sets _mlg_enabled = False and returns immediately.
        """
        self._mlg_config = self.config.get('mlg', {})
        self._mlg_enabled: bool = self._mlg_config.get('enabled', False)

        if not self._mlg_enabled:
            self._mlg_vtk_writer = None
            print(f"\n[6] Multi-Level Grid: disabled")
            return

        num_levels = self._mlg_config.get('num_levels', 1)
        overlap_width = self._mlg_config.get('overlap_width', 2)
        interp_name = self._mlg_config.get('interpolation', 'cubic')
        filter_level = self._mlg_config.get('filter_level', 1)

        print(f"\n[6] Multi-Level Grid Setup")
        print(f"  Levels: {num_levels}")
        print(f"  Overlap width: {overlap_width} coarse cells")
        print(f"  Interpolation: {interp_name}")
        print(f"  Filter level: {filter_level}")

        # ── Level scaler ─────────────────────────────────────────
        self._mlg_scaler = LevelScaler(
            tau_0=self.tau, num_levels=num_levels,
        )
        print(f"  Level τ values: ", end="")
        for k in range(num_levels):
            lu = self._mlg_scaler.get_level_units(k)
            print(f"L{k}={lu.tau:.4f} ", end="")
        print()

        # ── Interpolation scheme ─────────────────────────────────
        if interp_name == 'cubic':
            self._mlg_interp = CubicInterpolation()
        elif interp_name == 'compact_second_order':
            self._mlg_interp = CompactSecondOrderInterpolation()
        else:
            raise ValueError(f"Unknown interpolation: '{interp_name}'")

        # ── Overlap manager ──────────────────────────────────────
        # All regions in config are in Level 0 (physical) coordinates.
        # For Level k (k≥2), we convert to parent fine grid local coords.
        self._mlg_overlap_mgr = OverlapManager()
        levels_config = self._mlg_config.get('levels', [])

        coarse_shape = (self.Nx, self.Ny, self.Nz)

        # Track each level's physical origin and spacing for coord conversion
        # Stored as instance variables so _build_mlg_simulation() can access
        # them for fine-level obstacle coordinate transformation.
        self._mlg_level_origins = [(0.0, 0.0, 0.0)]   # Level 0 origin
        self._mlg_level_spacings = [(1.0, 1.0, 1.0)]  # Level 0 spacing

        for k in range(1, num_levels):
            level_cfg = levels_config[k] if k < len(levels_config) else {}
            region_cfg = level_cfg.get('region', {})

            # Config values are in L0 physical coordinates
            x_min_phys = region_cfg['x_min']
            x_max_phys = region_cfg['x_max']
            y_min_phys = region_cfg['y_min']
            y_max_phys = region_cfg['y_max']
            z_min_phys = region_cfg['z_min']
            z_max_phys = region_cfg['z_max']

            # Convert to parent (Level k-1) local coordinates
            po = self._mlg_level_origins[k - 1]     # parent origin in physical coords
            pd = self._mlg_level_spacings[k - 1]    # parent spacing

            local_x_min = round((x_min_phys - po[0]) / pd[0])
            local_x_max = round((x_max_phys - po[0]) / pd[0])
            local_y_min = round((y_min_phys - po[1]) / pd[1])
            local_y_max = round((y_max_phys - po[1]) / pd[1])
            local_z_min = round((z_min_phys - po[2]) / pd[2])
            local_z_max = round((z_max_phys - po[2]) / pd[2])

            fine_region = IndexBox(
                x_start=local_x_min, x_end=local_x_max,
                y_start=local_y_min, y_end=local_y_max,
                z_start=local_z_min, z_end=local_z_max,
            )

            overlap_region = self._mlg_overlap_mgr.add_level_pair(
                coarse_shape=coarse_shape,
                fine_region=fine_region,
                overlap_width=overlap_width,
            )

            # Compute this level's physical origin and spacing
            fdc = overlap_region.fine_domain_coarse
            lu_k = self._mlg_scaler.get_level_units(k)
            new_origin = (
                po[0] + fdc.x_start * pd[0],
                po[1] + fdc.y_start * pd[1],
                po[2] + fdc.z_start * pd[2],
            )
            new_spacing = (lu_k.dx, lu_k.dx, lu_k.dx)
            self._mlg_level_origins.append(new_origin)
            self._mlg_level_spacings.append(new_spacing)

            print(f"  Level {k}: phys region x[{x_min_phys},{x_max_phys}] "
                  f"y[{y_min_phys},{y_max_phys}] z[{z_min_phys},{z_max_phys}]")
            print(f"            fine shape = {overlap_region.fine_shape}, "
                  f"excised = {overlap_region.excised.num_nodes:,} nodes, "
                  f"origin = ({new_origin[0]:.1f}, {new_origin[1]:.1f}, {new_origin[2]:.1f})")

            # Next iteration: fine shape becomes the coarse shape
            coarse_shape = overlap_region.fine_shape

        self._mlg_filter_level = filter_level

        # ── MLG VTK writer ───────────────────────────────────────
        from src.io.mlg_vtk_writer import MLGVTKWriter
        vtk_out_dir = './results/vtk'
        if self.vtk_writer is not None:
            vtk_out_dir = self.vtk_writer.output_dir
        self._mlg_vtk_writer = MLGVTKWriter(
            output_dir=vtk_out_dir,
            coarse_shape=(self.Nx, self.Ny, self.Nz),
            overlap_mgr=self._mlg_overlap_mgr,
            scaler=self._mlg_scaler,
            num_levels=num_levels,
            precision=self._vtk_config.get('precision', 'float32'),
        )
        print(f"  {self._mlg_vtk_writer.get_info()}")

    def _build_mlg_simulation(self):
        """Build a MultiLevelGrid with M Simulation objects.

        Level 0 uses the existing setup components (collision, streaming,
        BC, obstacle). Fine levels get their own streaming operator and
        an empty DomainBCManager (coupling handles their boundaries).

        Returns:
            MultiLevelGrid with all Simulation objects (f not set yet).
        """
        xp = self.xp
        num_levels = self._mlg_config['num_levels']
        simulations = []
        couplings = []

        # ── Level 0: use existing setup ──────────────────────────
        sim_0 = Simulation(
            xp=xp,
            macroscopic=self.macro,
            collision=self.collision,
            streaming=self.streaming,
            bc_manager=self.domain_bc_mgr,
            tau=self.tau,
            domain_shape=self.domain_shape,
            obstacle_bc=self.obstacle_bc,
            al_model=self.al_model,
        )
        simulations.append(sim_0)

        # ── Fine levels ──────────────────────────────────────────
        for k in range(1, num_levels):
            region = self._mlg_overlap_mgr.get_region(k - 1)
            lu = self._mlg_scaler.get_level_units(k)
            fine_shape = region.fine_shape  # (Nx_f, Ny_f, Nz_f)

            # Fine level streaming (different domain shape)
            fine_streaming = StreamingPull(
                xp, self.lattice, fine_shape,
            )

            # Fine level BC: empty (coupling handles boundaries)
            fine_bc_mgr = DomainBCManager(
                xp=xp,
                lattice=self.lattice,
                boundaries_config={},
                domain_shape=fine_shape,
                verbose=False,
            )

            # ── Fine-level obstacle ──────────────────────────────
            # Physical process: same obstacle geometry, higher resolution.
            # The obstacle center/radius in L0 coords are transformed to
            # fine-level local coords using the level's physical origin
            # and grid spacing.
            #
            # Coordinate mapping:
            #   cx_fine = (cx_L0 - origin_x) / dx_fine
            #   radius_fine = radius_L0 / dx_fine
            #
            # This means the obstacle has 2^k times more grid points
            # on level k (e.g., D=20 at L0 → D=40 at L1).
            fine_obstacle_bc = None
            internal_geom = self.config.get('internal_geometry', {})
            if internal_geom:
                fine_origin = self._mlg_level_origins[k]
                fine_geom_config = create_fine_level_geometry_config(
                    geometry_config=internal_geom,
                    fine_origin_phys=fine_origin,
                    fine_shape=fine_shape,
                    dx_fine=lu.dx,
                    dx_coarse=1.0,
                    verbose=True,
                )

                if fine_geom_config:
                    fine_mask, fine_geom_info = create_geometry_mask(
                        xp, self.lattice, fine_shape,
                        fine_geom_config,
                        characteristic_length=None,
                        verbose=True,
                    )
                    n_solid = int(xp.sum(fine_mask))
                    if fine_geom_info['type'] != 'none' and n_solid > 0:
                        fine_obstacle_bc = HalfwayBounceBack(
                            xp, self.lattice, fine_mask,
                        )
                        print(f"    Level {k}: HalfwayBounceBack with "
                              f"{n_solid:,} solid nodes")

            # Fine level simulation
            sim_k = Simulation(
                xp=xp,
                macroscopic=self.macro,     # shared
                collision=self.collision,    # shared (τ passed per-call)
                streaming=fine_streaming,
                bc_manager=fine_bc_mgr,
                tau=lu.tau,
                domain_shape=fine_shape,
                obstacle_bc=fine_obstacle_bc,
                al_model=None,              # ALM on Level 0 only
            )
            simulations.append(sim_k)

            # ── Coupling engine for pair (k-1, k) ────────────────
            coupling_k = GridCoupling(
                xp=xp,
                lattice=self.lattice,
                region=region,
                scaler=self._mlg_scaler,
                interpolation=self._mlg_interp,
                filter_level=self._mlg_filter_level,
            )
            couplings.append(coupling_k)

        # ── Assemble MultiLevelGrid ──────────────────────────────
        mlg = MultiLevelGrid(levels=simulations, couplings=couplings)
        print(f"\n  MultiLevelGrid assembled:")
        print(f"  {mlg.summary()}")

        # ── MLG force: measure on finest level with obstacle ─────
        # Physical reason: the finest level has the most accurate
        # representation of the obstacle surface and flow field.
        # L0 f_post is captured before F→C coupling, so it does not
        # reflect the fine-grid solution.
        self._mlg_force_level: Optional[int] = None
        if self.force_mgr is not None:
            for k in range(num_levels - 1, -1, -1):
                if simulations[k].obstacle_bc is not None and k > 0:
                    lu_k = self._mlg_scaler.get_level_units(k)
                    scale = 1.0 / lu_k.dx   # = 2^k

                    fc = self._force_config
                    ref_config = fc.get('reference', {})
                    fine_force_config = {
                        'enabled': True,
                        'interval': self._force_interval,
                        'start_step': fc.get('start_step', 0),
                        'reference': {
                            'rho': self.force_mgr.rho_ref,
                            'velocity': self.force_mgr.u_ref,
                            'char_length': self.force_mgr.char_length * scale,
                            'span_length': self.force_mgr.span_length * scale,
                        },
                        'log': {'enabled': True, 'filename': 'force_history'},
                    }

                    self.force_mgr.close()
                    self.force_mgr = ForceManager(
                        xp=xp,
                        lattice=self.lattice,
                        solid_mask=simulations[k].obstacle_bc.solid_mask,
                        config=fine_force_config,
                        wall_bc=simulations[k].obstacle_bc,
                        csv_dir=self._csv_dir,
                    )
                    self.force_mgr.initialize()
                    self._mlg_force_level = k
                    print(f"\n  Force measurement: Level {k} "
                          f"(D_fine={self.force_mgr.char_length:.0f} "
                          f"[fine lu])")
                    break

        return mlg

    # =====================================================================
    # Summary (printed to terminal after all setup)
    # =====================================================================

    def _print_summary(self) -> None:
        """Print compact simulation summary to terminal."""
        sep = "-" * 70
        print(f"\n{sep}")
        print(f" Simulation Summary")
        print(f"{sep}")

        # ── Config & Device ──────────────────────────────────────
        device_name = "CPU (NumPy)"
        if self.xp.__name__ == 'cupy':
            try:
                dev = self.xp.cuda.Device()
                device_name = f"{dev.name} (GPU)"
            except Exception:
                device_name = "GPU (CuPy)"

        config_name = os.path.basename(self._args.config)
        lattice_model = self.sim_params.get('lattice_model', 'D3Q27')
        print(f" Config : {config_name}")
        print(f" Device : {device_name}")
        print(f" Lattice: {lattice_model}")

        # ── Grid ─────────────────────────────────────────────────
        Q = self.lattice.Q
        if self._mlg_enabled:
            num_levels = self._mlg_config['num_levels']
            total_nodes = 0
            total_updates = 0
            total_mem = 0.0
            grid_lines = []

            for k in range(num_levels):
                if k == 0:
                    shape = (self.Nx, self.Ny, self.Nz)
                    tau_k = self.tau
                else:
                    region = self._mlg_overlap_mgr.get_region(k - 1)
                    shape = region.fine_shape
                    tau_k = self._mlg_scaler.get_level_units(k).tau

                nodes = shape[0] * shape[1] * shape[2]
                steps = 2 ** k
                updates = nodes * steps
                mem = nodes * Q * 8 / (1024 * 1024)

                total_nodes += nodes
                total_updates += updates
                total_mem += mem

                grid_lines.append(
                    f"          L{k}: {shape[0]:>4}x{shape[1]:<4}x{shape[2]:<4}"
                    f" ({nodes:>9,}) x{steps} = {updates:>11,} updates  "
                    f"tau={tau_k:.4f}"
                )

            interp = self._mlg_config.get('interpolation', 'cubic')
            ow = self._mlg_config.get('overlap_width', 2)
            print(f" Grid   : {num_levels} levels (MLG), "
                  f"{total_nodes:,} total nodes")
            for line in grid_lines:
                print(line)
            print(f"          Total: {total_updates:,} updates/coarse step"
                  f" | {total_mem:.1f} MB")
            print(f"          Interp: {interp} | Overlap: {ow} cells")
        else:
            if self.lattice.dim == 2:
                cells = self.Nx * self.Ny
                print(f" Grid   : {self.Nx} x {self.Ny} = "
                      f"{cells:,} cells")
            else:
                cells = self.Nx * self.Ny * self.Nz
                print(f" Grid   : {self.Nx} x {self.Ny} x {self.Nz} = "
                      f"{cells:,} cells")

        # ── Physics ──────────────────────────────────────────────
        cs = (1.0 / 3.0) ** 0.5
        Ma_str = ""
        if self.u_ref_lu:
            Ma = self.u_ref_lu / cs
            Ma_str = f", Ma={Ma:.3f}"
        print(f" Physics: Re={self.Re}, tau={self.tau:.4f}, "
              f"nu={self.nu_lu:.4f}{Ma_str}")

        # ── BC summary ───────────────────────────────────────────
        bc_cfg = self.config.get('boundaries', {})
        bc_parts = []
        for key, cfg in bc_cfg.items():
            loc = cfg.get('location', key)
            method = cfg.get('method', '?')
            if 'inlet' in method:
                vel = cfg.get('velocity', '?')
                bc_parts.append(f"{loc}=inlet(u={vel})")
            elif 'outlet' in method:
                rho = cfg.get('rho', cfg.get('rho_target', '?'))
                bc_parts.append(f"{loc}=outlet(rho={rho})")
            elif 'wall' in method:
                bc_parts.append(f"{loc}=wall")
            else:
                bc_parts.append(f"{loc}={method}")

        if bc_parts:
            mid = (len(bc_parts) + 1) // 2
            line1 = '  '.join(bc_parts[:mid])
            line2 = '  '.join(bc_parts[mid:])
            print(f" BC     : {line1}")
            if line2:
                print(f"          {line2}")

        # ── Time ─────────────────────────────────────────────────
        ckpt_str = (f"Ckpt: {self.checkpoint_interval}"
                    if self.checkpoint_mgr else "Ckpt: off")
        print(f" Time   : {self.config_max_steps:,} steps | "
              f"VTK: {self.output_interval} | {ckpt_str}")

        # ── Log file ─────────────────────────────────────────────
        print(f" Log    : {self._log_path}")
        print(sep)