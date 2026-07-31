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
from src.boundary.bc_config import BCType
from src.boundary.wall import HalfwayBounceBack
from src.boundary.interpolated_wall import InterpolatedBounceBack
from src.boundary.stl_geometry import compute_q_fraction_triangles
from src.boundary.q_fraction import (
    compute_q_fraction_sphere,
    compute_needs_bounce,
    compute_q_fraction_circle,
    compute_q_fraction_cylinder_axis,
    compute_q_fraction_polyline,
)
from src.boundary.geometry_manager import (
    create_geometry_mask, validate_geometry_config,
    create_fine_level_geometry_config,
)
from src.io.geometry_outline_writer import write_geometry_outline

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
# ── 2D MLG imports (parallel to 3D, used when lattice.dim == 2) ──
from src.grid.coupling_2d import GridCoupling2D
from src.grid.overlap_manager_2d import OverlapManager2D, IndexBox2D


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

    def __init__(self, args: Any, io_role: str = 'writer') -> None:
        """Build entire simulation environment from CLI args.

        Default: detailed log → file only, compact summary → terminal.
        Use --verbose to also echo detailed log to terminal.

        io_role: 'writer' (default — full file IO, current single-GPU
        behavior) or 'silent' (MPI rank != 0: build everything but create
        NO directories and write NO files — kills the N-rank concurrent
        header/clear races. CheckpointManager is still built: the restore
        path needs it on every rank).
        """
        self._args = args
        self._verbose = getattr(args, 'verbose', False)
        assert io_role in ('writer', 'silent'), io_role
        self.io_role = io_role
        self.is_io_rank = (io_role == 'writer')

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

        # ── Write detailed log to file (IO rank only) ────────────
        self._log_path = None
        if self.is_io_rank:
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
        if self._log_path is not None:
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
            sgs_cfg=self._sgs_cfg,
        )

    def build_output_manager(self, manager_cls=None,
                             **extra_kwargs) -> 'OutputManager':
        """Create OutputManager with all I/O components.

        manager_cls/extra_kwargs: the MPI driver passes MPIOutputManager
        plus its comm/cadence kwargs — same wiring, one pipeline.

        Returns:
            OutputManager ready for start() → process() → finalize()
        """
        cls = manager_cls or OutputManager
        return cls(
            **extra_kwargs,
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
            log_interval=self.log_interval,
            check_interval=self.check_interval,
            checkpoint_interval=self.checkpoint_interval,
            tau=self.tau,
            solid_mask_np=self.solid_mask_np,
            perf_csv_path=self.perf_csv_path,
            blade_csv_dir=self.blade_csv_dir,
            domain_shape=self.domain_shape,
            L_ref_lu=self.L_ref_lu,
            u_ref_lu=self.u_ref_lu,
            config_path=self._args.config,
            mlg_vtk_writer=self._mlg_vtk_writer if self._mlg_enabled else None,
            mlg_force_level=getattr(self, '_mlg_force_level', None),
            alm_marker_origin=getattr(self, '_alm_marker_origin', None),
            alm_marker_spacing=getattr(self, '_alm_marker_spacing', None),
        )

    # =====================================================================
    # Private: Setup steps [0]–[5.4] (existing, unchanged)
    # =====================================================================

    def _load_config(self) -> None:
        """[0] Load config file and extract sub-configs.

        Config schema (top-level blocks only — see docs/CONFIG_GUIDE.md):
            simulation, physics, grid, numerics, boundaries,
            internal_geometry, mlg, time, output,
            conservation, convergence, force_calculation,
            actuator_line, airfoil_polar

        The `time` block now holds both step count and all interval cadences:
            time = {
                "max_steps": int,                    # required
                "output_interval": int,              # VTK output cadence
                "logging_interval": int,             # CSV/log cadence
                "checkpoint_interval": int,          # restart snapshot cadence
                "conservation_interval": int,        # conservation check cadence
            }
        """
        self._config_loader = ConfigLoader(self._args.config)
        self.config = self._config_loader.config

        self.sim_params = self.config.get('simulation', {})
        self._physics_config = self.config.get('physics', {})
        self._time_config = self.config.get('time', {})
        self._grid_cfg = self.config.get('grid', {})
        self._numerics_cfg = self.config.get('numerics', {})
        # NaN trap (debug) — DEFAULT OFF. The per-step bool(xp.any(~isfinite)) check
        # forces a GPU→CPU sync that serialises the async kernel pipeline and lowers
        # GPU utilisation, and it only adds value when a run actually diverges. Keep
        # it off for healthy production runs; enable ONLY on a re-run to locate a
        # blow-up. numerics.nan_trap (bool); numerics.nan_check_every (int; 1 = every
        # step, most precise; N = every N steps, cheaper). Set on the Simulation
        # class so every level (single-grid + MLG) shares the setting.
        Simulation.nan_trap_enabled = bool(self._numerics_cfg.get('nan_trap', False))
        Simulation.nan_check_every = max(1, int(self._numerics_cfg.get('nan_check_every', 1)))

        # ── numerics.esoteric → LBM_ESOTERIC bridge ──────────────
        # The streaming memory layout is a numerics choice, so the config
        # may request it (numerics.esoteric: true/false). The env var stays
        # the OVERRIDE (A/B twins and gates flip it per process; the MPI
        # driver setdefaults it to 1 before setup) — config applies only
        # when the env is unset. Downstream (per-level set_distribution)
        # keeps reading the env as the single mechanism.
        eso_cfg = self._numerics_cfg.get('esoteric', None)
        if eso_cfg is not None:
            if bool(eso_cfg):
                # fail fast on preconditions the esoteric path cannot meet
                # (an explicit config request must never silently degrade;
                # env-requested keeps the historical warn+fallback)
                dim = int(self.sim_params.get('dimension', 3))
                prec = str(self.sim_params.get('precision', 'float64'))
                dev = str(self.sim_params.get('device_mode', 'gpu'))
                problems = []
                if dim != 3:
                    problems.append(f"dimension={dim} (3D only)")
                if prec != 'float32':
                    problems.append(f"precision={prec} (float32 only)")
                if dev != 'gpu':
                    problems.append(f"device_mode={dev} (GPU only)")
                # wall_bc='ibb' + esoteric is supported since STL track S5
                # (single-GPU deposit-rewrite pass); MPI still rejects it
                # in the driver's fail-fast until S6.
                if problems:
                    raise ValueError(
                        "numerics.esoteric=true cannot be satisfied: "
                        + "; ".join(problems))
            if "LBM_ESOTERIC" not in os.environ:
                os.environ["LBM_ESOTERIC"] = "1" if eso_cfg else "0"

        self._output_config = self.config.get('output', {})
        self._vtk_config = self._output_config.get('vtk', {})
        self._checkpoint_config = self._output_config.get('checkpoint', {})
        self._conservation_config = self.config.get('conservation', {})
        self._force_config = self.config.get('force_calculation', {})
        self._conv_config = self.config.get('convergence', {})

        self._al_cfg = self.config.get('actuator_line', {})
        self.al_enabled: bool = self._al_cfg.get('enabled', False)
        self._unit_converter = None

        # SGS / turbulence-model config (validated; missing block -> disabled)
        from src.turbulence.sgs import parse_sgs_config
        self._sgs_cfg: dict = parse_sgs_config(self.config)
        if self._sgs_cfg["enabled"]:
            model = self._sgs_cfg["model"]
            if model == "smagorinsky":
                print(f"  SGS turbulence model: Smagorinsky "
                      f"(Cs={self._sgs_cfg['Cs']:.3f})")
            elif model == "wale":
                print(f"  SGS turbulence model: WALE "
                      f"(Cw={self._sgs_cfg['Cw']:.3f})")
            elif model == "dyn_smag":
                print(f"  SGS turbulence model: Dynamic Smagorinsky "
                      f"(Cs_max={self._sgs_cfg['Cs_max']:.2f}, "
                      f"alpha²={self._sgs_cfg['alpha_sq']:.2f})")
            else:
                print(f"  SGS turbulence model: {model}")
        else:
            print("  SGS turbulence model: off")

        self._validate_config_schema()

    def _validate_config_schema(self) -> None:
        """Fail-fast on legacy config format or missing required top-level blocks."""
        # Legacy detection: physics/domain nested inside simulation
        if 'physics' in self.sim_params:
            raise ValueError(
                "Legacy config format detected: 'physics' is nested inside "
                "'simulation'. Move it to the top-level `physics` block.\n"
                "See docs/CONFIG_GUIDE.md for the new format."
            )
        if 'domain' in self.sim_params:
            raise ValueError(
                "Legacy config format detected: 'domain' is nested inside "
                "'simulation'. Move Nx/Ny/Nz to the top-level `grid` block.\n"
                "See docs/CONFIG_GUIDE.md."
            )

        # Legacy detection: separate top-level `interval` block.
        # Migrated → all cadences live inside `time`.
        if 'interval' in self.config:
            raise ValueError(
                "Legacy config format detected: top-level `interval` block. "
                "Cadences are now folded into the unified `time` block:\n"
                "    time = {\n"
                "        \"max_steps\": ...,\n"
                "        \"output_interval\": ...,\n"
                "        \"logging_interval\": ...,\n"
                "        \"checkpoint_interval\": ...,\n"
                "        \"conservation_interval\": ...,\n"
                "    }\n"
                "See docs/CONFIG_GUIDE.md."
            )

        # Required blocks
        missing = [b for b in ('physics', 'grid', 'numerics')
                   if not self.config.get(b)]
        if missing:
            raise ValueError(
                f"Config missing required top-level block(s): {missing}. "
                f"See docs/CONFIG_GUIDE.md."
            )

        # Required physics keys (physical units, not lattice)
        required_physics = ('L_char',)
        missing_p = [k for k in required_physics if k not in self._physics_config]
        if missing_p:
            raise ValueError(
                f"physics block missing required key(s): {missing_p}. "
                f"L_char [m] is required for UnitConverter."
            )

        # Required grid key
        if 'resolution' not in self._grid_cfg and 'resolution' not in self._numerics_cfg:
            raise ValueError(
                "grid (or numerics) missing 'resolution' "
                "[cells per L_char]. Required for UnitConverter."
            )

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
        """[1] Domain setup — read Nx/Ny/Nz from top-level `grid` block."""
        g = self._grid_cfg
        self.Nx = g.get('Nx')
        self.Ny = g.get('Ny')
        self.Nz = g.get('Nz')  # None for 2D

        if self.Nx is None or self.Ny is None:
            raise ValueError("grid block requires Nx and Ny")

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
        """[2] Extract physics parameters via UnitConverter.

        Config schema: top-level `physics` (physical SI units) + `grid`
        (Nx/Ny/[Nz]/resolution) + `numerics` (u_max). UnitConverter derives
        all lattice parameters (tau, nu_lu, dt_phys, ...) from these inputs.
        """
        from src.solver.unit_converter import UnitConverter

        pc = self._physics_config
        tc = self._time_config
        fc = self._force_config

        self.config_max_steps: int = tc.get('max_steps', 10000)
        self.output_interval: int = tc.get('output_interval', 500)
        self.log_interval: int = tc.get('logging_interval', self.output_interval)
        self.checkpoint_interval: int = tc.get('checkpoint_interval', 2000)
        self.check_interval: int = tc.get(
            'conservation_interval', self.output_interval,
        )
        self._force_interval: int = fc.get('interval', 10)

        resolution = (self._numerics_cfg.get('resolution')
                      or self._grid_cfg.get('resolution'))

        grid_cfg = {
            'Nx': self.Nx, 'Ny': self.Ny, 'resolution': resolution,
        }
        if self.Nz is not None:
            grid_cfg['Nz'] = self.Nz

        # Pass the rotor config to the UnitConverter whenever a rotor is defined —
        # the tip_speed sets the velocity scale (dx/dt) and is a PHYSICAL property
        # independent of whether the ALM actually runs. Gating this on `al_enabled`
        # broke pure-LBM diagnostics (ALM disabled → tip_speed=0 → "U_max_phys ~ 0").
        # The ALM *running* is still gated by al_enabled elsewhere (al_model=None).
        _uc_al = self._al_cfg if (isinstance(self._al_cfg, dict)
                                  and self._al_cfg.get('rotor')) else None
        uc = UnitConverter(
            physics=pc,
            grid=grid_cfg,
            numerics=self._numerics_cfg,
            actuator_line=_uc_al,
        )
        self._unit_converter = uc

        # Expose conversion results as setup attributes for downstream modules.
        self.Re = uc.Re
        self.nu_lu = uc.nu_lu
        self.nu_phys = uc.nu_phys                # [m^2/s]
        self.tau = uc.tau
        self.u_ref_lu = uc.u_max
        self.L_ref_lu = float(uc.resolution)
        self.u_ref_phys = uc.U_inf
        self.dx_phys = uc.dx_phys
        self.dt_phys = uc.dt_phys
        self.c_s_phys = uc.c_s_phys              # [m/s] target sound speed

        # Inject initial_flow_velocity for the initializer.
        if 'initial_flow_velocity' not in pc:
            flow_dir = pc.get('flow_direction', [1, 0, 0])
            u = uc.U_inf_lu
            if uc.dim == 2:
                pc['initial_flow_velocity'] = [u * flow_dir[0], u * flow_dir[1]]
            else:
                pc['initial_flow_velocity'] = [
                    u * flow_dir[0], u * flow_dir[1], u * flow_dir[2]]

        uc.print_summary()

    def _setup_boundaries(self) -> None:
        """[3] Domain BC + internal obstacle.

        BC velocities are given in physical units [m/s] and converted to
        lattice units here via UnitConverter.phys_to_lu_velocity().
        """
        import copy
        print(f"\n[3] Domain Boundary Conditions")

        boundaries_config = copy.deepcopy(self.config.get('boundaries', {}))
        uc = self._unit_converter
        for name, bc in boundaries_config.items():
            if 'velocity' in bc:
                v = bc['velocity']
                if isinstance(v, (list, tuple)):
                    bc['velocity'] = [uc.phys_to_lu_velocity(vi) for vi in v]
                else:
                    bc['velocity'] = uc.phys_to_lu_velocity(float(v))

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
            self.obstacle_bc = self._build_obstacle_wall_bc(
                internal_geom, geom_info,
            )
        else:
            self.obstacle_bc = None

        self._geom_info = geom_info  # for outline dump in _setup_io

        self.solid_mask_np = (
            self._mask.get() if hasattr(self._mask, 'get') else self._mask
        )

    def _check_body_vs_coupling_band(self, k, region, fine_mask) -> None:
        """Body vs. C2F/F2C band on MLG fine level k (0.5*L_body rule).

        Solid cells inside the coupling band (fine-domain edge ..
        fine_region edge) corrupt the C2F/F2C stencils -> hard error.
        Body surface closer than 0.5*L_body to the fine_region edge
        couples the interface into the boundary layer (non-physical Cd
        shift) -> warning. STL track S0 carryover.
        """
        import numpy as _np
        from src.grid.overlap_manager import body_coupling_band_report

        solid_np = fine_mask.get() if hasattr(fine_mask, 'get') else fine_mask
        report = body_coupling_band_report(
            _np.asarray(solid_np, dtype=bool), region,
        )
        if report['violations']:
            faces = ', '.join(
                f"{face} ({n} solid cells)"
                for face, n in report['violations']
            )
            raise ValueError(
                f"Level {k}: obstacle intersects the C2F/F2C coupling band "
                f"on face(s): {faces}. Coupling interpolation would read/"
                f"write through the body. Enlarge mlg.levels[{k}].region so "
                f"the fine region encloses the body with >= 0.5*L_body "
                f"padding (MLG region padding rule)."
            )
        for face, dist, need in report['padding_warnings']:
            print(f"    [warn] Level {k}: body surface only {dist} fine "
                  f"cells from fine_region edge '{face}' "
                  f"(< 0.5*L_body = {need:.1f}) — interface couples into "
                  f"the boundary layer, Cd may shift non-physically")

    def _build_obstacle_wall_bc(
        self,
        internal_geom: dict,
        geom_info: dict,
        mask=None,
    ):
        """Pick HWBB or IBB based on internal_geometry.<type>.wall_bc.

        Supported values:
            'hwbb' (default) — half-way bounce-back
            'ibb'            — Bouzidi interpolated bounce-back (2D only here)

        For IBB, q-fraction is computed from the geometry:
            circle  → analytic circle intersection
            airfoil → ray-segment intersection vs polygon_lu stored in info
            other   → warn and use q = 0.5 (≈ HWBB)

        Args:
            internal_geom: full `internal_geometry` block (any-level).
            geom_info: output of create_geometry_mask for this mask.
            mask: solid mask to bind the BC to. If None, uses self._mask
                (single-grid run). For MLG fine levels, pass the
                level-local fine mask so the obstacle BC operates on the
                correct grid resolution.
        """
        if mask is None:
            mask = self._mask

        # Find which internal_geometry sub-dict is enabled
        wall_bc_type = 'hwbb'
        enabled_cfg = {}
        for _k, _v in internal_geom.items():
            if isinstance(_v, dict) and _v.get('enabled', False):
                enabled_cfg = _v
                wall_bc_type = _v.get('wall_bc', 'hwbb').lower()
                break

        if wall_bc_type == 'hwbb':
            print(f"  Wall BC: half-way bounce-back (HWBB)")
            return HalfwayBounceBack(self.xp, self.lattice, mask)

        if wall_bc_type != 'ibb':
            raise ValueError(
                f"Unknown wall_bc='{wall_bc_type}'. "
                f"Expected 'hwbb' or 'ibb'."
            )

        # Compute q-fraction from geometry info (2D and 3D paths)
        needs_bounce = compute_needs_bounce(
            self.xp, self.lattice, mask,
        )

        q_fraction = None
        gtype = geom_info.get('type')
        dim = self.lattice.dim

        if dim == 2 and gtype == 'circle':
            q_fraction = compute_q_fraction_circle(
                self.xp, self.lattice, mask, needs_bounce,
                center=geom_info['center'],
                radius=geom_info['radius'],
            )
            print(f"  Wall BC: Bouzidi IBB (analytic q from circle, 2D)")
        elif dim == 2 and gtype == 'airfoil' and 'polygon_lu' in geom_info:
            x_poly, y_poly = geom_info['polygon_lu']
            q_fraction = compute_q_fraction_polyline(
                self.xp, self.lattice, mask, needs_bounce,
                x_poly=x_poly, y_poly=y_poly,
            )
            print(f"  Wall BC: Bouzidi IBB (q from airfoil polyline, "
                  f"{len(x_poly)} vertices)")
        elif dim == 3 and gtype == 'cylinder':
            axis = geom_info.get('axis', 'z')
            q_fraction = compute_q_fraction_cylinder_axis(
                self.xp, self.lattice, mask, needs_bounce,
                center=geom_info['center'],
                radius=geom_info['radius'],
                axis=axis,
            )
            print(f"  Wall BC: Bouzidi IBB (analytic q from cylinder "
                  f"axis='{axis}', 3D)")
        elif dim == 3 and gtype == 'sphere':
            q_fraction = compute_q_fraction_sphere(
                self.xp, self.lattice, mask, needs_bounce,
                center=geom_info['center'],
                radius=geom_info['radius'],
            )
            print(f"  Wall BC: Bouzidi IBB (analytic q from sphere, 3D)")
        elif dim == 3 and gtype == 'stl' and 'triangles_lu' in geom_info:
            q_fraction = compute_q_fraction_triangles(
                self.xp, self.lattice, mask, needs_bounce,
                triangles_lu=geom_info['triangles_lu'],
            )
            print(f"  Wall BC: Bouzidi IBB (ray-triangle q from STL, "
                  f"{geom_info.get('n_faces', '?')} faces)")
        else:
            print(f"  [warn] wall_bc='ibb' with dim={dim} geom type='{gtype}' "
                  f"has no q-source; using q=0.5 sentinel (≡ HWBB).")

        if os.environ.get('LBM_FORCE_Q_HALF', '0') == '1' and q_fraction is not None:
            q_fraction = self.xp.full_like(q_fraction, 0.5)
            print(f"  [SANITY] LBM_FORCE_Q_HALF=1 → q_fraction overridden to 0.5 "
                  f"(IBB linear formula degenerates to HWBB)")

        bc = InterpolatedBounceBack(
            self.xp, self.lattice, mask, q_fraction=q_fraction,
        )
        print(f"  {bc.get_info()}")
        return bc

    def _setup_io(self) -> None:
        """[4] I/O directories, VTK, checkpoint, rotor CSV."""
        print(f"\n[4] I/O Setup")

        args = self._args
        oc = self._output_config

        # Directory resolution precedence (most specific wins):
        #   --vtk-dir / --csv-dir / --checkpoint-dir   (per-subdir override)
        #   > --results-dir PATH                       (root → PATH/{vtk,csv,checkpoints})
        #   > config `output` block.
        _root = getattr(args, 'results_dir', None)

        def _resolve(sub_flag, sub_name, cfg_key, cfg_default):
            if sub_flag:
                return sub_flag
            if _root:
                return os.path.join(_root, sub_name)
            return oc.get(cfg_key, cfg_default)

        output_dir = _resolve(getattr(args, 'vtk_dir', None), 'vtk',
                              'output_dir', './results/vtk')
        self.checkpoint_dir = _resolve(args.checkpoint_dir, 'checkpoints',
                                       'checkpoint_dir', './checkpoints')
        self._csv_dir = _resolve(args.csv_dir, 'csv', 'csv_dir', './results/csv')

        is_restart = args.restart_latest or args.restart is not None
        clear_previous = args.clear or oc.get('clear_previous', False)

        print(f"  VTK output dir: {output_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  CSV output dir: {self._csv_dir}")

        if self.is_io_rank:
            setup_output_directories(
                output_dir=output_dir,
                checkpoint_dir=self.checkpoint_dir,
                csv_dir=self._csv_dir,
                clear_previous=clear_previous,
                is_restart=is_restart,
            )
        else:
            print("  (io_role=silent: directories/clear/writers owned by "
                  "the IO rank)")

        # ── Geometry outline (one-shot, L0 lu) ──
        gi = getattr(self, '_geom_info', None)
        if (self.is_io_rank and gi is not None
                and gi.get('type', 'none') != 'none'):
            outline_path = os.path.join(output_dir, 'geometry_outline.vtk')
            write_geometry_outline(gi, outline_path)

        # ── VTK Writer ──
        vc = self._vtk_config
        vtk_enabled = (vc.get('enabled', True) and not args.no_vtk
                       and self.is_io_rank)
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
                create_dir=self.is_io_rank,
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
            'rho_ref,area_lu,tip_speed_lu,omega_lu,R_lu,D_lu,n_lu\n'
        )
        self.blade_csv_dir: Optional[str] = None
        self._blade_csv_header = (
            'step,revolutions,blade,r_R,r_lu,chord_lu,eps_lu,twist,'
            'u_n,u_theta,u_rel,phi,alpha,Re,CL,CD,'
            'F_n,F_theta,F_L,F_D,'
            'eps_c,eps_t,eps_r,eps_samp_c,eps_samp_t,eps_samp_r\n'
        )
        if self.al_enabled and self.is_io_rank:
            self.perf_csv_path = os.path.join(
                self._csv_dir, 'rotor_performance.csv',
            )
            self.blade_csv_dir = os.path.join(
                self._csv_dir, 'blade_diagnostics',
            )
            print(f"  Rotor CSV: {self.perf_csv_path} (opened at init)")
            print(f"  Blade CSV: {self.blade_csv_dir}/<marker>.csv (opened at init)")

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
        numerics = self.config.get('numerics', {})
        model_name = (numerics.get('collision')
                      or self.sim_params.get('collision_model', 'bgk')).lower()
    
        if model_name == 'cumulant':
            if self.lattice.dim == 2:
                from src.collision.cumulant_d2q9 import CumulantCollisionD2Q9
                omega_bulk = self.sim_params.get('omega_bulk', None)
                omega_3 = self.sim_params.get('omega_3', 0.6)
                omega_4 = self.sim_params.get('omega_4', 1.4)
                self.collision = CumulantCollisionD2Q9(
                    self.xp, self.lattice,
                    omega_bulk=omega_bulk,
                    omega_3=omega_3,
                    omega_4=omega_4,
                )
                ob_str = f"{omega_bulk}" if omega_bulk is not None else "ω_shear"
                print(f"  Collision: Cumulant D2Q9 "
                      f"(ω_bulk={ob_str}, ω_3={omega_3}, ω_4={omega_4})")
            else:
                from src.collision.cumulant import CumulantCollision
                omega_bulk = self.sim_params.get('omega_bulk', 1.0)
                omega_high = self.sim_params.get('omega_high', 1.0)
                omega_3 = self.sim_params.get('omega_3', None)
                omega_4 = self.sim_params.get('omega_4', None)
                omega_5 = self.sim_params.get('omega_5', None)
                limiter = self.sim_params.get('cumulant_limiter', 0.0)
                self.collision = CumulantCollision(
                    self.xp, self.lattice,
                    omega_bulk=omega_bulk,
                    omega_high=omega_high,
                    omega_3=omega_3, omega_4=omega_4, omega_5=omega_5,
                    limiter=limiter,
                )
                print(f"  Collision: Cumulant D3Q27 (ω_bulk={omega_bulk}, "
                      f"ω_high={omega_high}, "
                      f"ω_345=({self.collision.omega_3}, "
                      f"{self.collision.omega_4}, {self.collision.omega_5}), "
                      f"limiter λ={limiter})")
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

        # check_interval is set by _extract_physics from the interval block.
        self.conservation_mgr = ConservationManager(
            xp=self.xp,
            domain_shape=self.domain_shape,
            config=self._conservation_config,
            solid_mask=self._mask,
            csv_dir=self._csv_dir if self.is_io_rank else None,
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
            # Honor explicit override; otherwise ForceManager auto-enables for
            # any 2D obstacle.
            if 'save_link_forces' in fc:
                force_calc_config['save_link_forces'] = bool(
                    fc['save_link_forces']
                )

            self.force_mgr = ForceManager(
                xp=self.xp,
                lattice=self.lattice,
                solid_mask=self._mask,
                config=force_calc_config,
                wall_bc=self.obstacle_bc,
                csv_dir=self._csv_dir,
                internal_geometry=self.config.get('internal_geometry', {}),
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
            csv_dir=self._csv_dir if self.is_io_rank else None,
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

        # hub_center is given in L0 lattice units; rotor speed is given
        # as rpm. Convert to [m] / [rad/s] for downstream ALM code.
        import copy, math
        al_cfg = copy.deepcopy(self._al_cfg)
        dx = self._unit_converter.dx_phys
        if 'rotor' in al_cfg:
            rotor_cfg = al_cfg['rotor']
            if 'hub_center' in rotor_cfg:
                hc = rotor_cfg['hub_center']
                rotor_cfg['hub_center'] = [h * dx for h in hc]
            if 'rpm' in rotor_cfg and 'omega' not in rotor_cfg:
                rotor_cfg['omega'] = rotor_cfg['rpm'] * 2.0 * math.pi / 60.0

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
                sound_speed=self.c_s_phys,
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
                sound_speed=self.c_s_phys,
            )
            print(f"    Hub: {self.al_model.rotor.hub_center}")
            print(f"    R={self.al_model.rotor.radius:.1f} lu, "
                  f"ω={self.al_model.rotor.omega:.6f} rad/lt")
            print(f"    Blades: {self.al_model.rotor.n_blades}, "
                  f"Markers: {self.al_model.rotor.total_markers}")
            if hasattr(self.al_model.rotor, 'thrust_axis'):
                print(f"    Thrust direction: "
                      f"{self.al_model.rotor.thrust_axis.tolist()}")
                # Canonical-axis invariant (Step 2): the shaft (rotation_axis) must
                # be parallel to the disk normal (thrust_axis) — SIGN-AGNOSTIC, since
                # u_n and every wake correction reference n̂_a=−thrust_axis (not the
                # sign-arbitrary rotation_axis label). A mislabelled/non-parallel
                # config thus fails loudly here instead of silently flipping the
                # velocity triangle. See patch_notes/alm_canonical_axis/.
                import numpy as _np
                _r = _np.asarray(self.al_model.rotor.rotation_axis, dtype=float)
                _t = _np.asarray(self.al_model.rotor.thrust_axis, dtype=float)
                _cos = abs(float(_r @ _t)) / (
                    _np.linalg.norm(_r) * _np.linalg.norm(_t) + 1e-30)
                assert _cos > 1 - 1e-6, (
                    f"ALM axis invariant violated: rotation_axis {_r.tolist()} must "
                    f"be parallel to the disk normal / thrust_direction "
                    f"{_t.tolist()} (|cos|={_cos:.4f}). u_n and wake corrections "
                    f"reference n̂_a=−thrust_axis; a non-parallel shaft is a config error.")

        # Force ramp-up
        ramp = al_cfg.get('ramp_steps', 0)
        if ramp > 0 and self.al_model is not None:
            self.al_model.ramp_steps = ramp
            print(f"    Force ramp: {ramp} steps")

        # Kernel family echo — the D40 case-4' debug showed the log carried no
        # record of which spreading/sampling/deficit family a run used.
        if self.al_model is not None:
            _kern = (al_cfg.get('kernel') or {}).get('type', 'gaussian')
            print(f"    Kernel family: {_kern}")

    # =====================================================================
    # [6] Multi-Level Grid Setup (NEW)
    # =====================================================================

    def _setup_mlg(self) -> None:
        """[6] Parse MLG config and prepare multi-level components.

        Reads the 'mlg' section from config. If not present or disabled,
        sets _mlg_enabled = False and returns immediately.

        Dispatches to 2D variant when lattice.dim == 2.
        """
        self._mlg_config = self.config.get('mlg', {})
        self._mlg_enabled: bool = self._mlg_config.get('enabled', False)

        if not self._mlg_enabled:
            self._mlg_vtk_writer = None
            print(f"\n[6] Multi-Level Grid: disabled")
            return

        # Route to 2D path if lattice is D2Q9
        if self.lattice.dim == 2:
            self._setup_mlg_2d()
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
            print(f"L{k}={lu.tau:.6f} ", end="")
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

        Dispatches to 2D variant when lattice.dim == 2.

        Returns:
            MultiLevelGrid with all Simulation objects (f not set yet).
        """
        if self.lattice.dim == 2:
            return self._build_mlg_simulation_2d()

        xp = self.xp
        num_levels = self._mlg_config['num_levels']
        simulations = []
        couplings = []

        # ── Determine which level gets the ALM ──────────────────
        alm_target_level = 0  # default: coarsest level
        if self.al_model is not None and num_levels > 1:
            from src.actuator.actuator_line import MultiRotorManager
            if isinstance(self.al_model, MultiRotorManager):
                print(f"\n  ALM: Level 0 "
                      f"(multi-rotor fine-level not yet supported)")
            else:
                hub_L0 = self.al_model.rotor.hub_center  # L0 lattice units
                # Check from finest to coarsest
                for k in range(num_levels - 1, 0, -1):
                    region = self._mlg_overlap_mgr.get_region(k - 1)
                    origin = self._mlg_level_origins[k]
                    spacing = self._mlg_level_spacings[k]
                    fs = region.fine_shape
                    # Hub in fine-level local lattice units
                    hub_loc = tuple(
                        (hub_L0[d] - origin[d]) / spacing[d]
                        for d in range(3)
                    )
                    margin = 5.0
                    if all(margin <= hub_loc[d] <= fs[d] - margin
                           for d in range(3)):
                        alm_target_level = k
                        break

                if alm_target_level > 0:
                    print(f"\n  ALM target: Level {alm_target_level} "
                          f"(finest level containing rotor hub)")
                else:
                    print(f"\n  ALM target: Level 0 "
                          f"(hub not inside any fine region)")

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
            al_model=self.al_model if alm_target_level == 0 else None,
            sgs_cfg=self._sgs_cfg,
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

                # Cache so the per-level ForceManager (built later) can
                # supply finest-lattice chord/center to the surface-distribution
                # post-processor via NPZ metadata.
                if fine_geom_config:
                    if not hasattr(self, '_mlg_fine_geom_configs'):
                        self._mlg_fine_geom_configs = {}
                    self._mlg_fine_geom_configs[k] = fine_geom_config

                if fine_geom_config:
                    fine_mask, fine_geom_info = create_geometry_mask(
                        xp, self.lattice, fine_shape,
                        fine_geom_config,
                        characteristic_length=None,
                        verbose=True,
                    )
                    n_solid = int(xp.sum(fine_mask))
                    if fine_geom_info['type'] != 'none' and n_solid > 0:
                        self._check_body_vs_coupling_band(k, region, fine_mask)
                        # Honor wall_bc (hwbb / ibb) on this MLG fine level too —
                        # NOT hardcoded HWBB. Without this, IBB requested in the
                        # config silently downgrades to HWBB on every level.
                        print(f"    Level {k}: building obstacle BC "
                              f"({n_solid:,} solid nodes)")
                        fine_obstacle_bc = self._build_obstacle_wall_bc(
                            internal_geom=internal_geom,
                            geom_info=fine_geom_info,
                            mask=fine_mask,
                        )

            # ── Fine-level ALM (if this is the target level) ─────
            fine_al_k = None
            if k == alm_target_level and self.al_model is not None:
                fine_al_k = self._create_fine_level_alm(k, fine_shape)

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
                al_model=fine_al_k,
                sgs_cfg=self._sgs_cfg,
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

        # ── Update al_model to fine-level for OutputManager ──────
        if alm_target_level > 0:
            self.al_model = simulations[alm_target_level].al_model
            # Marker coordinate transform: fine local → global (L0 units)
            self._alm_marker_origin = self._mlg_level_origins[alm_target_level]
            self._alm_marker_spacing = self._mlg_level_spacings[alm_target_level][0]

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
                        # Preserve the surface-distribution flag from the original
                        # ForceManager. Without this the MLG-replacement ForceManager
                        # (constructed below without `internal_geometry`) defaults to
                        # False and the per-link NPZ is never written.
                        'save_link_forces': self.force_mgr.save_link_forces,
                    }

                    self.force_mgr.close()
                    fine_geom_for_force = getattr(
                        self, '_mlg_fine_geom_configs', {}
                    ).get(k, self.config.get('internal_geometry', {}))
                    self.force_mgr = ForceManager(
                        xp=xp,
                        lattice=self.lattice,
                        solid_mask=simulations[k].obstacle_bc.solid_mask,
                        config=fine_force_config,
                        wall_bc=simulations[k].obstacle_bc,
                        csv_dir=self._csv_dir,
                        internal_geometry=fine_geom_for_force,
                    )
                    self.force_mgr.initialize()
                    self._mlg_force_level = k
                    print(f"\n  Force measurement: Level {k} "
                          f"(D_fine={self.force_mgr.char_length:.0f} "
                          f"[fine lu])")
                    break

        return mlg

    # =================================================================
    # 2D MLG variants (lattice.dim == 2)
    # =================================================================

    def _setup_mlg_2d(self) -> None:
        """2D variant of _setup_mlg. Uses OverlapManager2D + IndexBox2D.

        ALM is intentionally NOT supported in 2D (rotor is a 3D concept).
        """
        num_levels = self._mlg_config.get('num_levels', 1)
        overlap_width = self._mlg_config.get('overlap_width', 2)
        interp_name = self._mlg_config.get('interpolation', 'cubic')
        filter_level = self._mlg_config.get('filter_level', 1)

        print(f"\n[6] Multi-Level Grid Setup (2D)")
        print(f"  Levels: {num_levels}")
        print(f"  Overlap width: {overlap_width} coarse cells")
        print(f"  Interpolation: {interp_name}")
        print(f"  Filter level: {filter_level}")

        # ── Level scaler (dimension-agnostic) ────────────────────
        self._mlg_scaler = LevelScaler(
            tau_0=self.tau, num_levels=num_levels,
        )
        print(f"  Level τ values: ", end="")
        for k in range(num_levels):
            lu = self._mlg_scaler.get_level_units(k)
            print(f"L{k}={lu.tau:.6f} ", end="")
        print()

        # ── Interpolation scheme (dimension-agnostic) ────────────
        if interp_name == 'cubic':
            self._mlg_interp = CubicInterpolation()
        elif interp_name == 'compact_second_order':
            self._mlg_interp = CompactSecondOrderInterpolation()
        else:
            raise ValueError(f"Unknown interpolation: '{interp_name}'")

        # ── Overlap manager (2D) ─────────────────────────────────
        self._mlg_overlap_mgr = OverlapManager2D()
        levels_config = self._mlg_config.get('levels', [])

        coarse_shape = (self.Nx, self.Ny)

        # Track physical origin/spacing per level for obstacle transform
        self._mlg_level_origins = [(0.0, 0.0)]
        self._mlg_level_spacings = [(1.0, 1.0)]

        for k in range(1, num_levels):
            level_cfg = levels_config[k] if k < len(levels_config) else {}
            region_cfg = level_cfg.get('region', {})

            x_min_phys = region_cfg['x_min']
            x_max_phys = region_cfg['x_max']
            y_min_phys = region_cfg['y_min']
            y_max_phys = region_cfg['y_max']

            po = self._mlg_level_origins[k - 1]
            pd = self._mlg_level_spacings[k - 1]

            local_x_min = round((x_min_phys - po[0]) / pd[0])
            local_x_max = round((x_max_phys - po[0]) / pd[0])
            local_y_min = round((y_min_phys - po[1]) / pd[1])
            local_y_max = round((y_max_phys - po[1]) / pd[1])

            fine_region = IndexBox2D(
                x_start=local_x_min, x_end=local_x_max,
                y_start=local_y_min, y_end=local_y_max,
            )

            overlap_region = self._mlg_overlap_mgr.add_level_pair(
                coarse_shape=coarse_shape,
                fine_region=fine_region,
                overlap_width=overlap_width,
            )

            # Compute physical origin/spacing of this level
            fdc = overlap_region.fine_domain_coarse
            lu_k = self._mlg_scaler.get_level_units(k)
            new_origin = (
                po[0] + fdc.x_start * pd[0],
                po[1] + fdc.y_start * pd[1],
            )
            new_spacing = (lu_k.dx, lu_k.dx)
            self._mlg_level_origins.append(new_origin)
            self._mlg_level_spacings.append(new_spacing)

            print(f"  Level {k}: phys region x[{x_min_phys},{x_max_phys}] "
                  f"y[{y_min_phys},{y_max_phys}]")
            print(f"            fine shape = {overlap_region.fine_shape}, "
                  f"excised = {overlap_region.excised.num_nodes:,} nodes, "
                  f"origin = ({new_origin[0]:.1f}, {new_origin[1]:.1f})")

            # Next iteration: fine shape becomes the coarse shape
            coarse_shape = overlap_region.fine_shape

        self._mlg_filter_level = filter_level

        # ── MLG VTK writer (2D) ──────────────────────────────────
        from src.io.mlg_vtk_writer_2d import MLGVTKWriter2D
        vtk_out_dir = './results/vtk'
        if self.vtk_writer is not None:
            vtk_out_dir = self.vtk_writer.output_dir
        self._mlg_vtk_writer = MLGVTKWriter2D(
            output_dir=vtk_out_dir,
            coarse_shape=(self.Nx, self.Ny),
            overlap_mgr=self._mlg_overlap_mgr,
            scaler=self._mlg_scaler,
            num_levels=num_levels,
            precision=self._vtk_config.get('precision', 'float32'),
        )
        print(f"  {self._mlg_vtk_writer.get_info()}")

    def _build_mlg_simulation_2d(self):
        """2D variant of _build_mlg_simulation. No ALM."""
        xp = self.xp
        num_levels = self._mlg_config['num_levels']
        simulations: list = []
        couplings: list = []

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
            al_model=None,  # ALM not supported in 2D
            sgs_cfg=self._sgs_cfg,
        )
        simulations.append(sim_0)

        # Import 2D-specific fine-level geometry transform
        from src.boundary.geometry_manager import (
            create_fine_level_geometry_config_2d,
        )

        # ── Fine levels ──────────────────────────────────────────
        for k in range(1, num_levels):
            region = self._mlg_overlap_mgr.get_region(k - 1)
            lu = self._mlg_scaler.get_level_units(k)
            fine_shape = region.fine_shape  # (Nx_f, Ny_f)

            fine_streaming = StreamingPull(
                xp, self.lattice, fine_shape,
            )

            fine_bc_mgr = DomainBCManager(
                xp=xp,
                lattice=self.lattice,
                boundaries_config={},
                domain_shape=fine_shape,
                verbose=False,
            )

            # ── Fine-level obstacle (2D) ─────────────────────────
            fine_obstacle_bc = None
            internal_geom = self.config.get('internal_geometry', {})
            if internal_geom:
                fine_origin = self._mlg_level_origins[k]
                fine_geom_config = create_fine_level_geometry_config_2d(
                    geometry_config=internal_geom,
                    fine_origin_phys=fine_origin,
                    fine_shape=fine_shape,
                    dx_fine=lu.dx,
                    dx_coarse=1.0,
                    verbose=True,
                )

                # Cache so the per-level ForceManager (built later) can
                # supply finest-lattice chord/center to the surface-distribution
                # post-processor via NPZ metadata.
                if fine_geom_config:
                    if not hasattr(self, '_mlg_fine_geom_configs'):
                        self._mlg_fine_geom_configs = {}
                    self._mlg_fine_geom_configs[k] = fine_geom_config

                if fine_geom_config:
                    fine_mask, fine_geom_info = create_geometry_mask(
                        xp, self.lattice, fine_shape,
                        fine_geom_config,
                        characteristic_length=None,
                        verbose=True,
                    )
                    n_solid = int(xp.sum(fine_mask))
                    if fine_geom_info['type'] != 'none' and n_solid > 0:
                        self._check_body_vs_coupling_band(k, region, fine_mask)
                        # Honor wall_bc (hwbb / ibb) on this MLG fine level too.
                        print(f"    Level {k}: building obstacle BC "
                              f"({n_solid:,} solid nodes)")
                        fine_obstacle_bc = self._build_obstacle_wall_bc(
                            internal_geom=internal_geom,
                            geom_info=fine_geom_info,
                            mask=fine_mask,
                        )

            sim_k = Simulation(
                xp=xp,
                macroscopic=self.macro,
                collision=self.collision,
                streaming=fine_streaming,
                bc_manager=fine_bc_mgr,
                tau=lu.tau,
                domain_shape=fine_shape,
                obstacle_bc=fine_obstacle_bc,
                al_model=None,
                sgs_cfg=self._sgs_cfg,
            )
            simulations.append(sim_k)

            # ── Coupling engine (2D) for pair (k-1, k) ───────────
            coupling_k = GridCoupling2D(
                xp=xp,
                lattice=self.lattice,
                region=region,
                scaler=self._mlg_scaler,
                interpolation=self._mlg_interp,
                filter_level=self._mlg_filter_level,
            )
            couplings.append(coupling_k)

        mlg = MultiLevelGrid(levels=simulations, couplings=couplings)
        print(f"\n  MultiLevelGrid (2D) assembled:")
        print(f"  {mlg.summary()}")

        # ── Force measurement on finest level with obstacle ──────
        self._mlg_force_level: Optional[int] = None
        if self.force_mgr is not None:
            for k in range(num_levels - 1, -1, -1):
                if simulations[k].obstacle_bc is not None and k > 0:
                    lu_k = self._mlg_scaler.get_level_units(k)
                    scale = 1.0 / lu_k.dx   # = 2^k

                    fc = self._force_config
                    fine_force_config = {
                        'enabled': True,
                        'interval': self._force_interval,
                        'start_step': fc.get('start_step', 0),
                        'reference': {
                            'rho': self.force_mgr.rho_ref,
                            'velocity': self.force_mgr.u_ref,
                            'char_length': self.force_mgr.char_length * scale,
                            'span_length': self.force_mgr.span_length,
                        },
                        'log': {'enabled': True, 'filename': 'force_history'},
                        # Preserve surface-distribution flag (see 3D path).
                        'save_link_forces': self.force_mgr.save_link_forces,
                    }

                    self.force_mgr.close()
                    fine_geom_for_force = getattr(
                        self, '_mlg_fine_geom_configs', {}
                    ).get(k, self.config.get('internal_geometry', {}))
                    self.force_mgr = ForceManager(
                        xp=xp,
                        lattice=self.lattice,
                        solid_mask=simulations[k].obstacle_bc.solid_mask,
                        config=fine_force_config,
                        wall_bc=simulations[k].obstacle_bc,
                        csv_dir=self._csv_dir,
                        internal_geometry=fine_geom_for_force,
                    )
                    self.force_mgr.initialize()
                    self._mlg_force_level = k
                    print(f"\n  Force measurement: Level {k} "
                          f"(c_fine={self.force_mgr.char_length:.0f} "
                          f"[fine lu])")
                    break

        return mlg

    def _create_fine_level_alm(self, level_k: int, fine_shape):
        """Create ALM instance for a fine MLG level.

        Coordinate Transform:
            hub_center (global [m]) → fine-level local [m]
            → to_lattice_units(dx_fine, dt_fine) → fine local [lu]

            hub_fine_lu = (hub_global - origin) / dx_fine
                        = (hub_L0 - origin_L0) × 2^k

        Physical scaling (refinement ratio 2):
            dx_fine = dx_L0 / 2^k      [m]
            dt_fine = dt_L0 / 2^k      [s]
            ω_fine  = ω_phys · dt_fine  [rad/lt]  (= ω_L0 / 2^k)
            u_inf   = same as L0       [Δx/Δt]  (convective scaling)
            ν_fine  = ν_L0 · 2^k       [Δx²/Δt] (from τ_fine)
            ε_fine  = max(c/4, 2·Δx)   [fine lu] (c in fine lu = c_L0 · 2^k)

        Args:
            level_k: Fine level index (1, 2, ...).
            fine_shape: (Nx_f, Ny_f, Nz_f) of the fine level domain.

        Returns:
            ActuatorLineModel configured for the fine level.
        """
        import copy
        import math

        lu_k = self._mlg_scaler.get_level_units(level_k)
        origin_L0 = self._mlg_level_origins[level_k]

        # Fine level physical scales
        dx_fine = self.dx_phys * lu_k.dx   # dx_phys / 2^k
        dt_fine = self.dt_phys * lu_k.dt   # dt_phys / 2^k

        # Fine level viscosity (in fine lattice units)
        nu_fine = (1.0 / 3.0) * (lu_k.tau - 0.5)

        # ── Prepare AL config with fine-level hub_center ─────────
        al_cfg = copy.deepcopy(self._al_cfg)
        rotor_cfg = al_cfg.get('rotor', {})

        # hub_center in config is L0 lattice units; convert to [m].
        dx_uc = self._unit_converter.dx_phys
        if 'hub_center' in rotor_cfg:
            hc = rotor_cfg['hub_center']
            rotor_cfg['hub_center'] = [h * dx_uc for h in hc]
        if 'rpm' in rotor_cfg and 'omega' not in rotor_cfg:
            rotor_cfg['omega'] = rotor_cfg['rpm'] * 2.0 * math.pi / 60.0

        # Convert hub to fine-level local [m]
        hub_m = rotor_cfg['hub_center']          # global [m]
        origin_m = [o * self.dx_phys for o in origin_L0]
        hub_local_m = [h - o for h, o in zip(hub_m, origin_m)]
        rotor_cfg['hub_center'] = hub_local_m

        # ── u_inf (same in lattice units across all levels) ──────
        pc = self._physics_config
        U_inf_phys = pc.get('U_inf', 0.0)
        u_inf_lu = (U_inf_phys * dt_fine / dx_fine
                    if U_inf_phys > 0 else None)

        # ── Create fine-level ALM ────────────────────────────────
        from src.actuator.actuator_line import create_actuator_line_from_config

        fine_al = create_actuator_line_from_config(
            config=al_cfg,
            domain_shape=fine_shape,
            nu_lattice=nu_fine,
            polar_query=self.al_model.polar_query,
            dx_phys=dx_fine,
            dt_phys=dt_fine,
            u_inf_lu=u_inf_lu,
            coeff_mode=al_cfg.get('coeff_mode', 'auto'),
            xp=self.xp,
            sound_speed=self.c_s_phys,
        )

        # Ramp steps (scale to fine timesteps for same physical duration)
        ramp = al_cfg.get('ramp_steps', 0)
        if ramp > 0:
            fine_al.ramp_steps = ramp * (2 ** level_k)

        # ── Print info ───────────────────────────────────────────
        print(f"\n  Fine-level ALM (Level {level_k}):")
        print(f"    hub_local [m] = "
              f"[{hub_local_m[0]:.4f}, "
              f"{hub_local_m[1]:.4f}, "
              f"{hub_local_m[2]:.4f}]")
        print(f"    hub_local [lu] = "
              f"({fine_al.rotor.hub_center[0]:.1f}, "
              f"{fine_al.rotor.hub_center[1]:.1f}, "
              f"{fine_al.rotor.hub_center[2]:.1f})")
        print(f"    R = {fine_al.rotor.radius:.1f} fine lu, "
              f"markers = {fine_al.rotor.total_markers}")
        print(f"    omega = {fine_al.rotor.omega:.6f} rad/lt_fine")
        print(f"    dx = {dx_fine*1000:.4f} mm, "
              f"dt = {dt_fine*1e6:.4f} us")
        print(f"    nu = {nu_fine:.6e}, "
              f"tau = {lu_k.tau:.6f}")
        if ramp > 0:
            print(f"    ramp = {fine_al.ramp_steps} fine steps "
                  f"({ramp} coarse)")

        return fine_al

    # =====================================================================
    # Summary (printed to terminal after all setup)
    # =====================================================================

    def _summary_wall_bc_string(self) -> str:
        """Build a one-line description of the obstacle wall BC for the summary.

        Returns empty string when there is no obstacle. For IBB the
        actually-effective branch is reported (Bouzidi linear vs HWBB
        sentinel) since 'wall_bc=ibb' silently degrades to HWBB when no
        q-fraction source is available.
        """
        if self.obstacle_bc is None:
            return ""
        from src.boundary.wall import HalfwayBounceBack
        from src.boundary.interpolated_wall import InterpolatedBounceBack

        ig = self.config.get('internal_geometry', {})
        gtype = next(
            (k for k, v in ig.items()
             if isinstance(v, dict) and v.get('enabled', False)),
            None,
        ) or "unknown"

        if isinstance(self.obstacle_bc, InterpolatedBounceBack):
            q = self.obstacle_bc.q_fraction
            nb = self.obstacle_bc.needs_bounce
            xp = self.xp
            n_links = int(xp.sum(nb))
            n_sentinel = int(xp.sum(nb & (q == 0.5))) if n_links else 0
            mode = "Bouzidi IBB"
            extra = (f" [{n_sentinel}/{n_links} q=0.5 sentinel]"
                     if n_links else "")
            return f"{mode} on {gtype}{extra}"
        if isinstance(self.obstacle_bc, HalfwayBounceBack):
            return f"HWBB on {gtype}"
        return f"{type(self.obstacle_bc).__name__} on {gtype}"

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
                    if self.lattice.dim == 2:
                        shape = (self.Nx, self.Ny)
                    else:
                        shape = (self.Nx, self.Ny, self.Nz)
                    tau_k = self.tau
                else:
                    region = self._mlg_overlap_mgr.get_region(k - 1)
                    shape = region.fine_shape
                    tau_k = self._mlg_scaler.get_level_units(k).tau

                # Works for any dimensionality
                nodes = 1
                for s in shape:
                    nodes *= s
                steps = 2 ** k
                updates = nodes * steps
                mem = nodes * Q * 8 / (1024 * 1024)

                total_nodes += nodes
                total_updates += updates
                total_mem += mem

                if self.lattice.dim == 2:
                    grid_lines.append(
                        f"          L{k}: {shape[0]:>4}x{shape[1]:<4}"
                        f" ({nodes:>9,}) x{steps} = {updates:>11,} updates  "
                        f"tau={tau_k:.4f}"
                    )
                else:
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
        # Physical-unit values (Re is dimensionless; nu in m^2/s;
        # Ma is the physical Mach of the reference velocity vs c_s_phys).
        u_ref_phys_eff = (self.u_ref_phys
                          if self.u_ref_phys and self.u_ref_phys > 0
                          else self.u_ref_lu * self.dx_phys / self.dt_phys)
        Ma_phys = u_ref_phys_eff / self.c_s_phys if self.c_s_phys > 0 else 0.0
        print(f" Physics: Re={self.Re:.0f}, "
              f"nu={self.nu_phys:.4e} m^2/s, "
              f"Ma_phys={Ma_phys:.4f} "
              f"[tau_L0={self.tau:.6f}]")

        # ── Wall BC (internal obstacle) ──────────────────────────
        wall_bc_str = self._summary_wall_bc_string()
        if wall_bc_str:
            print(f" Wall   : {wall_bc_str}")

        # ── SGS turbulence model ────────────────────────────────
        if self._sgs_cfg["enabled"]:
            model = self._sgs_cfg["model"]
            if model == "smagorinsky":
                print(f" SGS    : Smagorinsky (Cs={self._sgs_cfg['Cs']:.3f})")
            elif model == "wale":
                print(f" SGS    : WALE (Cw={self._sgs_cfg['Cw']:.3f})")
            elif model == "dyn_smag":
                print(f" SGS    : Dynamic Smagorinsky "
                      f"(Cs_max={self._sgs_cfg['Cs_max']:.2f})")
            else:
                print(f" SGS    : {model}")
        else:
            print(f" SGS    : off")

        # ── BC summary (reads actually-parsed face_configs) ──────
        bc_cfg = self.config.get('boundaries', {})
        applied_locs = {fc.location.value for fc in self.domain_bc_mgr.face_configs}
        expected_locs = {
            str(cfg.get('location', key)).lower()
            for key, cfg in bc_cfg.items()
        }
        dropped = expected_locs - applied_locs

        bc_parts = []
        for fc in self.domain_bc_mgr.face_configs:
            loc = fc.location.value
            m = fc.method
            if fc.bc_type == BCType.VELOCITY:
                if isinstance(fc.velocity, (list, tuple)):
                    u_str = "[" + ",".join(f"{v:.3f}" for v in fc.velocity) + "]"
                else:
                    u_str = f"{fc.velocity:.3f}"
                bc_parts.append(f"{loc}={m}(u={u_str})")
            elif fc.bc_type == BCType.PRESSURE:
                bc_parts.append(f"{loc}={m}(ρ={fc.density:.3f})")
            elif fc.bc_type == BCType.WALL:
                bc_parts.append(f"{loc}=wall({m})")
            elif fc.bc_type == BCType.SPONGE:
                L = fc.extra.get('thickness', '?')
                s = fc.extra.get('sigma_max', '?')
                bc_parts.append(f"{loc}=sponge(L={L}, σ={s})")
            else:
                bc_parts.append(f"{loc}={m}")

        if bc_parts:
            mid = (len(bc_parts) + 1) // 2
            line1 = '  '.join(bc_parts[:mid])
            line2 = '  '.join(bc_parts[mid:])
            print(f" BC     : {line1}")
            if line2:
                print(f"          {line2}")
        else:
            print(f" BC     : (none applied — all periodic)")

        if dropped:
            print(f" BC WARN: {len(dropped)} boundary face(s) dropped: "
                  f"{sorted(dropped)} — check setup_log.txt")

        # ── ALM summary ──────────────────────────────────────────
        if self.al_model is not None:
            import math as _math
            _model = self.al_model
            # Handle both single and multi-rotor
            _is_multi = hasattr(_model, 'models')
            _models = _model.models if _is_multi else [_model]

            # MLG fine-level scaling factor
            _mlg_scale = 1
            _level_tag = ""
            if self._mlg_enabled:
                _nlev = self._mlg_config.get('num_levels', 1)
                _mlg_scale = 2 ** (_nlev - 1)
                _level_tag = f" @L{_nlev - 1}"

            for _i, _m in enumerate(_models):
                _r = _m.rotor
                _prefix = f" ALM[{_i}]" if _is_multi else " ALM"
                _R_phys = _r.radius * _m.dx_phys
                # Tip chord (last active marker), scaled to fine level
                _blade0 = _r.blades[0]
                _active = _blade0.marker_active
                _tip_idx = max(
                    (j for j in range(len(_active)) if _active[j]),
                    default=len(_active) - 1,
                )
                _c_tip = float(_blade0.marker_chord[_tip_idx]) * _mlg_scale
                _R_lu = _r.radius * _mlg_scale
                print(
                    f"{_prefix} : {_r.n_blades} blades x "
                    f"{_r.markers_per_blade} markers, "
                    f"R={_R_lu:.1f} lu ({_R_phys:.4f} m), "
                    f"c_tip/dx={_c_tip:.2f}{_level_tag}"
                )
                _omega_lu = abs(_r.omega) / _mlg_scale
                _omega_rpm = abs(_r.omega) / _m.dt_phys * 60 / (2 * _math.pi)
                print(
                    f"          omega={_omega_lu:.6f} rad/lt "
                    f"({_omega_rpm:.0f} RPM), "
                    f"tip={_omega_lu*_R_lu:.4f} lu"
                )
                _prandtl = "on" if _m.prandtl_loss else "off"
                _ramp = _m.ramp_steps * _mlg_scale
                print(
                    f"          Prandtl={_prandtl}, "
                    f"gauss_cut={_m.n_cut:.1f}, "
                    f"ramp={_ramp}"
                )

            # Airfoil polar info
            _pc = self.config.get('airfoil_polar', {})
            _method = _pc.get('method', 'neuralfoil')
            if _method == 'multi':
                _names = list(_pc.get('airfoils', {}).keys())
                _af_str = ' + '.join(_names)
                _sample = next(iter(_pc['airfoils'].values()), {})
            else:
                _af_str = _pc.get('airfoil_name', '?')
                _sample = _pc
            # Per-airfoil method drives the label (c81 deck vs NeuralFoil-generated).
            _af_method = _sample.get('method', _method)
            if _af_method == 'c81':
                import os as _os
                _path = _sample.get('path', '')
                _src = _os.path.basename(_os.path.dirname(_path)) or 'data/airfoils'
                print(
                    f" Polar  : {_af_str} (C81 deck source: {_src}, "
                    f"e.g. {_os.path.basename(_path)})"
                )
            else:
                _ncrit = _sample.get('ncrit', 9.0)
                _mode = _sample.get('mode', 'asb')
                _re_min = _sample.get('Re_min', '?')
                _re_max = _sample.get('Re_max', '?')
                _re_steps = _sample.get('Re_steps', '?')
                print(
                    f" Polar  : {_af_str} (NeuralFoil/{_mode}, "
                    f"ncrit={_ncrit}, "
                    f"Re={_re_min}/{_re_max}/{_re_steps})"
                )

        # ── Force calculation ────────────────────────────────────
        if self.force_mgr is not None and self.force_mgr.enabled:
            fm = self.force_mgr
            print(f" Force  : interval={fm.interval}, start={fm.start_step}"
                  f"  (ref: ρ={fm.rho_ref}, U={fm.u_ref:.4f}, "
                  f"D={fm.char_length})")
            if fm.save_link_forces:
                npz_path = os.path.join(fm.csv_dir, 'surface_link_forces.npz')
                print(f"          Cp data → {npz_path}")

        # ── Time ─────────────────────────────────────────────────
        ckpt_str = (f"Ckpt: {self.checkpoint_interval}"
                    if self.checkpoint_mgr else "Ckpt: off")
        dt_str = ""
        if self.dt_phys is not None:
            dt_str = f" | dt={self.dt_phys*1e6:.2f}us"
        print(f" Time   : {self.config_max_steps:,} steps{dt_str} | "
              f"VTK: {self.output_interval} | Log: {self.log_interval} | "
              f"{ckpt_str}")

        # ── Log file ─────────────────────────────────────────────
        print(f" Log    : {self._log_path}")
        print(sep)