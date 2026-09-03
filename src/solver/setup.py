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
from src.boundary.q_fraction import (
    compute_needs_bounce,
    compute_q_fraction_circle,
    compute_q_fraction_polyline,
)
# 3D q sources are imported inside _setup_ibb_links (link cores).
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


def _mlg_regions_of(level_cfg: dict, k: int) -> list:
    """The refinement boxes configured for level k.

    `region: {...}` is sugar for `regions: [{...}]`. Supplying both is an error
    rather than a precedence rule — silently honouring one would hide the other.
    """
    has_one = 'region' in level_cfg
    has_many = 'regions' in level_cfg
    if has_one and has_many:
        raise ValueError(
            f"mlg.levels[{k}] sets BOTH 'region' and 'regions'. Use 'regions' "
            f"(a list) for several blocks, or 'region' for one.")
    if has_many:
        regions = level_cfg['regions']
        if not isinstance(regions, (list, tuple)) or not regions:
            raise ValueError(
                f"mlg.levels[{k}].regions must be a non-empty list of boxes.")
        return list(regions)
    if has_one:
        return [level_cfg['region']]
    raise ValueError(
        f"mlg.levels[{k}] has no refinement box: set 'region' (one) or "
        f"'regions' (several).")


def _mlg_resolve_parent(blocks: list, parent_level: int, box, explicit, name):
    """The block at `parent_level` that contains this box (L0 coordinates).

    Resolution is by CONTAINMENT rather than an explicit index. The containment
    check has to run anyway (a block must nest in its parent), and sibling
    blocks are validated disjoint, so at most one candidate can ever match —
    the mapping is a well-defined partial function, not a heuristic. An index
    would be a second source of truth that drifts when the list is reordered.

    `parent:` is still accepted, as an ASSERTION against the resolved answer.
    """
    x0, x1, y0, y1, z0, z1 = box
    candidates = [b for b in blocks if b.level == parent_level]
    if parent_level == 0:
        hits = candidates                        # the root covers everything
    else:
        hits = []
        for b in candidates:
            r = b.region
            # fine_region / fine_domain_coarse are in the GRANDPARENT's index
            # space, so the offset scales by the grandparent's spacing — not by
            # this block's own, which is half of it.
            gp = b.parent.spacing if b.parent is not None else 1.0
            lo = (b.origin[0] + (r.fine_region.x_start - r.fine_domain_coarse.x_start) * gp,
                  b.origin[1] + (r.fine_region.y_start - r.fine_domain_coarse.y_start) * gp,
                  b.origin[2] + (r.fine_region.z_start - r.fine_domain_coarse.z_start) * gp)
            hi = (b.origin[0] + (r.fine_region.x_end - r.fine_domain_coarse.x_start) * gp,
                  b.origin[1] + (r.fine_region.y_end - r.fine_domain_coarse.y_start) * gp,
                  b.origin[2] + (r.fine_region.z_end - r.fine_domain_coarse.z_start) * gp)
            if (lo[0] <= x0 and x1 <= hi[0] and lo[1] <= y0 and y1 <= hi[1]
                    and lo[2] <= z0 and z1 <= hi[2]):
                hits.append(b)

    if len(hits) != 1:
        avail = ", ".join(f"'{b.name}'" for b in candidates) or "(none)"
        raise ValueError(
            f"MLG block '{name}' x[{x0},{x1}] y[{y0},{y1}] z[{z0},{z1}] is "
            f"contained in {len(hits)} level-{parent_level} block(s); it must "
            f"be exactly 1. Level-{parent_level} blocks: {avail}.\n"
            f"  Widen the parent so it fully contains this box, or move this "
            f"box inside one parent — a block straddling two parents has no "
            f"well-defined coupling.")
    resolved = hits[0]
    if explicit is not None:
        want = (resolved.name == explicit if isinstance(explicit, str)
                else resolved.index == int(explicit))
        if not want:
            raise ValueError(
                f"MLG block '{name}' declares parent={explicit!r} but its box "
                f"lies inside '{resolved.name}'. Fix the box or the declaration.")
    return resolved
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
            # eso implicit domain walls (eso_wall track): single-grid
            # only. MLG member sims + every MPI path keep the loud EQ/
            # SOLID degradation until PLAN sec 4 steps 5-6 wire the
            # coupling region gather/scatter and the runner.
            eso_wall_implicit_ok=True,
        )

    def build_output_manager(self, manager_cls=None,
                             blocks_meta=None,
                             **extra_kwargs) -> 'OutputManager':
        """Create OutputManager with all I/O components.

        manager_cls/extra_kwargs: the MPI driver passes MPIOutputManager
        plus its comm/cadence kwargs — same wiring, one pipeline.

        blocks_meta: GLOBAL block geometry (uid/origin/spacing/shape/
        level/index dicts, level-major) captured by the MPI driver from
        the replicated build. Required when output.planes or
        output.probes runs under MPI — the runner keeps topology only,
        so the geometry must be handed over before the build is dropped.

        Returns:
            OutputManager ready for start() → process() → finalize()
        """
        # Probes/planes under MPI: swap the single-process managers for
        # the distributed ones (owner-rank sampling through the
        # replicated partition map). A requested channel must never
        # silently produce nothing — missing geometry is a hard error,
        # on every rank (the request flags are parsed rank-invariantly).
        probe_mgr = getattr(self, 'probe_mgr', None)
        plane_mgr = getattr(self, 'plane_mgr', None)
        if manager_cls is not None:
            _rank = int(extra_kwargs.get('rank', 0))
            _nr = int(extra_kwargs.get('nr', 1))
            if getattr(self, '_probe_cfg', None):
                if blocks_meta is None:
                    raise ValueError(
                        "output.probes under MPI: the driver must pass "
                        "blocks_meta (global block geometry) to "
                        "build_output_manager")
                from src.io.probe_writer import MPIPressureProbeManager
                probe_mgr = MPIPressureProbeManager(
                    self._probe_cfg, self._csv_dir,
                    self._unit_converter, self._dimension,
                    blocks_meta=blocks_meta, rank=_rank, n_ranks=_nr,
                    comm=extra_kwargs.get('comm'))
            if getattr(self, '_plane_cfg', None):
                if blocks_meta is None:
                    raise ValueError(
                        "output.planes under MPI: the driver must pass "
                        "blocks_meta (global block geometry) to "
                        "build_output_manager")
                from src.io.plane_writer import MPIPlaneWriterManager
                plane_mgr = MPIPlaneWriterManager(
                    self._plane_cfg, self._vtk_output_dir,
                    self._unit_converter, self.field_units,
                    precision=self._vtk_config.get('precision',
                                                   'float32'),
                    blocks_meta=blocks_meta,
                    rank=_rank, n_ranks=_nr)

        cls = manager_cls or OutputManager
        mgr = cls(
            **extra_kwargs,
            xp=self.xp,
            macroscopic=self.macro,
            lattice=self.lattice,
            sim_params=self.sim_params,
            vtk_writer=self.vtk_writer,
            marker_vtk_writer=self.marker_vtk_writer,
            checkpoint_mgr=self.checkpoint_mgr,
            probe_mgr=probe_mgr,
            plane_mgr=plane_mgr,
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
            mlg_force_block=getattr(self, '_mlg_force_block', None),
            alm_marker_origin=getattr(self, '_alm_marker_origin', None),
            alm_marker_spacing=getattr(self, '_alm_marker_spacing', None),
        )
        # output.vtk.fields_start_step: suppress DOMAIN-field VTK before this
        # step; ALM marker VTP keeps the full output_interval cadence.
        # Single-GPU analogue of the MPI driver's --vtk-fields-last.
        mgr.vtk_fields_start_step = int(
            self.config.get('output', {}).get('vtk', {})
            .get('fields_start_step', 0))
        return mgr

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
            _env = os.environ.get("LBM_ESOTERIC")
            _want = "1" if eso_cfg else "0"
            if _env is None:
                os.environ["LBM_ESOTERIC"] = _want
            elif _env != _want:
                # Explicit config vs explicit env used to resolve silently
                # (env won), flipping the streaming layout — and with it
                # memory footprint and the domain-wall BC semantics —
                # behind the user's back. Under MPI the driver setdefaults
                # LBM_ESOTERIC=1, so numerics.esoteric=false + mpirun also
                # lands here (the distributed runner is esoteric-only).
                raise ValueError(
                    f"numerics.esoteric={bool(eso_cfg)} conflicts with "
                    f"LBM_ESOTERIC={_env} in the environment. Unset the "
                    "env var (the config seeds it), or drop the config "
                    "key (env-driven A/B twins run configs without it).")

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

        # Deprecated physics keys (nu-only policy): these were parsed once,
        # then ignored for years — a user hand-editing "Re" saw no effect.
        _dead_phys = ('Re', 'tau', 'omega', 'nu_lu', 'u_ref_lu', 'L_ref_lu',
                      'Re_U_ref', 'Re_L_ref')
        _phys = self.config.get('physics', {})
        _found = [k for k in _dead_phys if k in _phys]
        if _found:
            raise ValueError(
                f"physics block contains removed key(s) {_found}: the "
                "solver takes physics.nu only (Re-targeting: keep L_char "
                "fixed and derive NU = U*L/RE in the config script). "
                "Delete the key(s) — they have been ignored since the "
                "nu-only migration.")

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

        # Missing device_mode has always meant "try GPU" (None fell through
        # to the GPU branch); make that default explicit — setup_library
        # rejects anything other than 'cpu'/'gpu' and no longer falls back
        # to NumPy on GPU failure.
        device_mode = self.sim_params.get('device_mode', 'gpu')
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
                                  and (self._al_cfg.get('rotor')
                                       or self._al_cfg.get('rotors'))) \
            else None
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

        # Which axes are genuinely periodic on L0 — the seam contract the
        # obstacle link enumeration is built against (q_fraction docstring;
        # patch_notes/ibb_sparse/02 sec. 7 (b)). Derived once here, single
        # source: an axis wraps iff neither face carries a non-periodic BC.
        from src.boundary.bc_config import derive_periodic_axes
        self._periodic_axes = derive_periodic_axes(
            self.config.get('boundaries', {}), self.lattice.dim,
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

        Under mlg.wall_coupling.mode='exclude' the wall neighbourhood is
        skipped by the coupling instead (src/grid/wall_coupling.py), so
        the intersection is reported with its per-face cost rather than
        rejected — see that module for what the exclusion does and does
        NOT cover.
        """
        import numpy as _np
        from src.grid.overlap_manager import body_coupling_band_report
        from src.grid.wall_coupling import report_band_exclusion

        solid_np = fine_mask.get() if hasattr(fine_mask, 'get') else fine_mask
        solid_np = _np.asarray(solid_np, dtype=bool)
        report = body_coupling_band_report(solid_np, region)
        policy = self._wall_coupling
        # partial-body surfel level (patch 77): the body crossing the
        # band is the DESIGN (finest-wins partial refinement) — accept
        # without demanding the global wall_coupling mode, and flag the
        # level so _attach_coupling_skip force-attaches its own local
        # C2F skip (the global exclude knob polluted EVERY level's
        # coupling in the first LE-L4 run).
        _enabled_bc = [v for v in self.config.get(
            'internal_geometry', {}).values()
            if isinstance(v, dict) and v.get('enabled', False)]
        _is_surfel_geom = bool(_enabled_bc) and \
            _enabled_bc[0].get('wall_bc', 'hwbb').lower() == 'surfel'
        if report['violations'] and _is_surfel_geom:
            faces = ', '.join(f"{face} ({n} solid cells)"
                              for face, n in report['violations'])
            print(f"    [wall_coupling] Level {k}: partial-body surfel "
                  f"level — body crosses the band on {faces}; local C2F "
                  f"skip auto-attached (patch 77)")
            self._partial_skip_levels.add(int(k))
            report = dict(report, violations=[])
        if report['violations']:
            faces = ', '.join(
                f"{face} ({n} solid cells)"
                for face, n in report['violations']
            )
            if not policy.relaxes_guard:
                raise ValueError(
                    f"Level {k}: obstacle intersects the C2F/F2C coupling "
                    f"band on face(s): {faces}. Coupling interpolation would "
                    f"read/write through the body. Enlarge "
                    f"mlg.levels[{k}].region so the fine region encloses the "
                    f"body with >= 0.5*L_body padding (MLG region padding "
                    f"rule), or set mlg.wall_coupling.mode='exclude' to skip "
                    f"the wall neighbourhood instead (experimental)."
                )
            how = ("wall neighbourhood skipped" if policy.excludes_wall
                   else "coupling UNCHANGED")
            print(f"    [wall_coupling] Level {k}: body intersects the C2F "
                  f"band on {faces} — accepted, {how}")
            for face, n_band, n_ex in report_band_exclusion(
                    solid_np, region,
                    policy.wall_margin if policy.excludes_wall else 0):
                frac = 100.0 * n_ex / n_band
                flag = "  <-- face is mostly excluded" if frac > 50.0 else ""
                print(f"      band {face:7s}: {n_ex:,}/{n_band:,} cells "
                      f"excluded ({frac:.1f}%){flag}")
        for face, dist, need in report['padding_warnings']:
            print(f"    [warn] Level {k}: body surface only {dist} fine "
                  f"cells from fine_region edge '{face}' "
                  f"(< 0.5*L_body = {need:.1f}) — interface couples into "
                  f"the boundary layer, Cd may shift non-physically")

    def _attach_coupling_skip(self, sim, solid_mask, level: int,
                              label: str = "") -> None:
        """Give `sim` the flags its coupling scatters skip.

        Strict policy (default) attaches None, so `coupling_skip_nt` falls
        back to the level's esoteric node type and the run is bit-identical
        to before wall-aware coupling existed.
        """
        from src.grid.wall_coupling import attach_coupling_skip

        n = attach_coupling_skip(sim, solid_mask, self._wall_coupling)
        if n == 0 and level in getattr(self, '_partial_skip_levels', ()):
            # patch 77 (2nd iteration): NO local C2F skip on the partial
            # surfel level. The margin-1 skip starved the wall-adjacent
            # band cells of their only inflow (74's measured 'cut-skip
            # backfire') and the L4 near-wall layer slowly destabilized
            # (~55 coarse steps, whole x-extent). On a SURFEL level the
            # skip is unnecessary: C2F writes into dead cells are healed
            # by zero_dead every substep, writes onto cut cells are
            # bounded by the 0.5 dv floor, and interpolation THROUGH the
            # wall is handled by the C2F dead fill. Solid-write hygiene
            # that hwbb needs (patch 12) does not apply here.
            print(f"    [wall_coupling] L{level}: partial-body surfel — "
                  f"no coupling skip (band cells keep their C2F supply; "
                  f"zero_dead heals solid writes)")
        if n:
            tag = f"L{level}" + (f" '{label}'" if label else "")
            n_solid = int(solid_mask.sum())
            print(f"    [wall_coupling] {tag}: {n:,} cells excluded from "
                  f"coupling ({n_solid:,} solid + "
                  f"{n - n_solid:,} wall-adjacent fluid)")

    def _build_obstacle_wall_bc(
        self,
        internal_geom: dict,
        geom_info: dict,
        mask=None,
        level_ctx: Optional[dict] = None,
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
            level_ctx: MLG fine-level context (surfel path, S8a-2) —
                {'level': k, 'nu_lu': lu.nu, 'region': OverlapRegion}.
                None on L0 / single grid. Also selects the seam policy
                for the link enumeration (see below).
        """
        from src.boundary.q_fraction import detect_cut_faces

        if mask is None:
            mask = self._mask

        # ── Seam policy (periodic_axes) for the link enumeration ─────
        # L0: derived from the domain BCs (an axis wraps iff neither face
        #     carries a non-periodic BC).
        # MLG fine block: every face is a coupling boundary — nothing
        #     wraps by right.
        # Either way the span_through axis is added back: the z-invariant
        # prism's wrap is its correctness contract (geometry_manager),
        # and with a z-invariant mask it changes no link anyway.
        if level_ctx is not None:
            per_axes: Tuple[int, ...] = ()
        else:
            per_axes = getattr(self, '_periodic_axes', None)
        _span = geom_info.get('span_through_axis')
        if _span is not None:
            _span_ax = ({'x': 0, 'y': 1, 'z': 2}[str(_span).lower()]
                        if isinstance(_span, str) else int(_span))
            if per_axes is not None and _span_ax not in per_axes:
                per_axes = tuple(per_axes) + (_span_ax,)

        # Body cut by a non-periodic box face = open body. The links that
        # used to wrap there are suppressed by the enumeration below, so
        # the fact must be carried on the BC: momentum-exchange force on
        # an open body is not the body force (the torus closure used to
        # hide a |F|=869 imbalance — patch_notes/ibb_sparse/02 sec. 3).
        cut_faces = (detect_cut_faces(self.xp, mask, per_axes)
                     if per_axes is not None else ())
        if cut_faces:
            where = ("the coupling faces of this fine block"
                     if level_ctx is not None else "the domain boundary")
            print(f"  [note] body is cut by box face(s) "
                  f"{', '.join(cut_faces)} ({where}) — seam links "
                  f"suppressed, MEM force on this block is not a body "
                  f"force")

        # Find which internal_geometry sub-dict is enabled
        _enabled = [(_k, _v) for _k, _v in internal_geom.items()
                    if isinstance(_v, dict) and _v.get('enabled', False)]
        if len(_enabled) > 1:
            # Silently taking the first (dict order!) let the wall_bc come
            # from one geometry while the mask came from another.
            raise ValueError(
                "internal_geometry: more than one geometry enabled "
                f"({', '.join(k for k, _ in _enabled)}) — enable exactly one")
        wall_bc_type = 'hwbb'
        enabled_cfg = {}
        if _enabled:
            enabled_cfg = _enabled[0][1]
            wall_bc_type = enabled_cfg.get('wall_bc', 'hwbb').lower()

        if wall_bc_type == 'hwbb':
            if cut_faces and os.environ.get('LBM_ESOTERIC', '0') == '1':
                # The esoteric fused kernel bounces via node_type with
                # UNCONDITIONAL periodic address arithmetic (% N) — the
                # link filter above cannot reach it, so ghost bounces
                # across the seam remain. They only ever write the
                # outermost cell layer, which lies inside the C2F/F2C
                # band by construction (overlap_width >= 1 is enforced),
                # so the coupling overwrite contains them — but the seam
                # is NOT honest the way the ibb link path now is. The
                # force guard elsewhere is what keeps this safe to run
                # (rig_cut / wall_coupling exclude track relies on it).
                print(f"  [warn] wall_bc='hwbb' + esoteric with body cut "
                      f"by {', '.join(cut_faces)}: kernel-level periodic "
                      f"addressing keeps ghost bounces on the seam "
                      f"(contained in the coupling band). wall_bc='ibb' "
                      f"is the seam-honest choice for cut bodies.")
            print(f"  Wall BC: half-way bounce-back (HWBB)")
            bc = HalfwayBounceBack(self.xp, self.lattice, mask,
                                   periodic_axes=per_axes)
            bc.cut_faces = cut_faces
            return bc

        if wall_bc_type == 'surfel':
            # Scope guards (patch_notes/surfel/46 sec. 1): STL, standard
            # path, D3Q27, no ALM/SGS. Each violation is an explicit
            # error — never a silent fallback. MLG fine levels are
            # supported since S8a-2 (level_ctx carries the level-local
            # viscosity and the coupling region for the band guards).
            import os as _os
            if geom_info.get('type') != 'stl' \
                    or geom_info.get('triangles_lu') is None:
                raise ValueError(
                    "wall_bc='surfel' needs STL geometry with triangles_lu "
                    f"(got type='{geom_info.get('type')}')")
            if self.lattice.dim != 3:
                raise ValueError("wall_bc='surfel' is D3Q27-only")
            if _os.environ.get('LBM_ESOTERIC', '0') == '1':
                # 46 sec. 3 deferral lifted: V1 residency bridge (gather ->
                # std surfel chain -> scatter, patch_notes/surfel/63).
                # MPI stays guarded below until the slab filter (62 (2)).
                print("  [surfel] esoteric residency bridge active "
                      "(patch_notes/surfel/63 V1)")
            from src.solver.entry import detect_world_size
            if detect_world_size() > 1:
                # patch 64: z-slab surfel MPI — the runner enforces the
                # specifics (axis=z, ghost>=4) and builds slab-scoped
                # bridge Simulations (src/parallel/surfel_level.py).
                print("  [surfel] MPI z-slab path (patch_notes/surfel/64)")
            alm_cfg = self.config.get('actuator_line', {})
            if isinstance(alm_cfg, dict) and alm_cfg.get('enabled', False):
                raise NotImplementedError(
                    "wall_bc='surfel' + ALM is S8b+ scope "
                    "(patch_notes/surfel/46)")
            if self._sgs_cfg.get('enabled', False) \
                    and self._sgs_cfg.get('model') != 'smagorinsky':
                raise NotImplementedError(
                    "wall_bc='surfel' + SGS supports constant-Cs "
                    "'smagorinsky' only (S8b-2: the moment-based Stiebler "
                    "route the wall-model campaign was measured with — "
                    "patch_notes/surfel/54). wale/dyn_smag live in the "
                    "fused-kernel pre-pass the surfel path does not run.")
            from src.boundary.surfel_boundary import (
                SurfelBoundary, check_level_coupling_bands,
            )
            ctx = level_ctx or {}
            scfg = dict(enabled_cfg.get('surfel', {}))
            # Production STL default: dv_min = 1e-2, NOT the adapter's
            # 1e-6 (patch 50 sec. 4c/4e). The advect amplifies any state
            # imbalance by 1/dV at a sliver cell, and for arbitrary STL
            # geometry the cut fraction lands arbitrarily close above
            # the floor. Measured ladder (c=50/c=100 wing):
            #   1e-6 -> dV=1.038e-6 sliver, IC imbalance x1e6, inf in
            #          2 substeps (immediate);
            #   1e-3 -> dV=1.28e-3 sliver AT THE LEADING EDGE survived
            #          the start-up but diverged at coarse step 93 once
            #          the LE suction developed (gain 780 x flow
            #          fluctuation);
            #   1e-2 -> gain <= 100, 8x margin on the measured event —
            #          re-verified on the exact c=100 failure case.
            # The 1e-6 testbed floor was measured on PLANAR channel
            # walls whose dV is quantized far from it. This floor is a
            # REGULARIZATION, not physics: the wall force is facet-
            # accounted (unaffected); dropped slivers are a volume-
            # bookkeeping stairstep reported as `dropped dV`. The
            # structural fix (cut-cell merging) is the registered
            # follow-up. Config may still override.
            scfg.setdefault('dv_min', 1e-2)
            # Partial-body level (patch 74): a fine box that cuts the
            # body keeps only the triangles whose surfel stencils clear
            # its coupling bands (finest-wins partition); the parent
            # level keeps the full wing. Opt-in: without the flag the
            # band guard below still REFUSES a cut body (bit-identical
            # for every existing run).
            # Partial-body level (patch 74): the box cuts the body. The
            # facet set is NOT clipped — the surfel build handles a body
            # leaving the grid natively (facets exist only for in-grid
            # cells; edge-clipped prisms stay mass-consistent through
            # Vsum, the Eq. 7 '<= 1' case), so the wall runs unbroken to
            # the box face. Triangle clipping was measured to blow up in
            # one substep: partial facet coverage (a dropped triangle's
            # cells keep dV < 1 but lose its flux) and wall-less
            # live/dead interfaces are both structural holes. Facets in
            # the C2F band are LEGITIMATE here (the band supplies the
            # outer flow as a BC; the facet supplies the wall) — the
            # band guard is skipped below, and near-wall C2F pollution
            # is the wall_coupling exclude mode's existing job.
            # Ownership (accounting only) is partitioned later: this
            # level owns the triangles inside its fine_region (= the
            # excised region where MLG already deems the fine solution
            # authoritative), the parent keeps the band triangles.
            # Partial-body detection (patch 77): AUTOMATIC, per level —
            # a level is partial iff its box CUTS the body (cut_faces,
            # already detected above). The former config knob
            # 'partial_body' applied through the shared surfel dict to
            # EVERY level, silently forcing tau_model OFF and the 0.5
            # sliver floor on the whole hierarchy (the LE-L4 run froze:
            # Cd -0.089, fluctuations dead). The knob is rejected now.
            if 'partial_body' in scfg:
                raise ValueError(
                    "surfel config: 'partial_body' is not a knob "
                    "(patch 77) — a level is partial-body iff its box "
                    "cuts the STL (detected automatically); remove it")
            partial = bool(cut_faces)
            if partial:
                # patch 74: sliver amplification floor. The C2F band
                # crosses the wall, so band interpolants (bounded but
                # wall-inconsistent) feed cut cells whose advect divides
                # by dV — dv_min 1e-2 leaves gains up to 100x and the
                # measured blow-up cells all sat at dV 0.01-0.04. Floor
                # HALF: gain <= 2. Cost is geometric only (the facet
                # force ledger never uses dV; dropped slivers are the
                # registered volume-bookkeeping stairstep, patch 50) —
                # a half-cell wall displacement at slivers, i.e. the
                # parent level's own resolution.
                import os as _os3
                _floor = float(_os3.environ.get('LBM_PARTIAL_DVMIN', 0.5))
                if float(scfg.get('dv_min', 1e-2)) < _floor:
                    scfg['dv_min'] = _floor
            # patch 77: tau_model STAYS ON for partial levels — the 74
            # force-OFF predated the local C2F skip and the dead fill,
            # and the tau-less L4 fed the parent an alien near-wall
            # regime (F2C garbage growth, L3 blow-up at coarse step 14).
            # The injection/C2F conflict is confined to the band rows
            # inside the outer C2F strips — those rows are excised below.
            if partial:
                # march-axis validity (patch 74): the volumetric march
                # pins each line by max(dV)=1 and requires one fully
                # fluid cell per line. A box the body CROSSES along the
                # march axis has interior lines that never leave the
                # body — measured: chordwise march on the LE box marked
                # the wing core live (36k cells, first-substep blow-up).
                # Pick the first axis (configured first) whose lines all
                # contain solid-free cells, using the level mask.
                import numpy as _np
                from src.boundary.surfel_boundary import _DEFAULTS as _SD
                _solid = _np.asarray(mask.get() if hasattr(mask, 'get')
                                     else mask, dtype=bool)
                _dil = _solid.copy()
                for _ax in range(3):
                    for _sh in (1, -1):
                        _dil |= _np.roll(_solid, _sh, axis=_ax)
                _cand = [int(scfg.get('march_axis', _SD['march_axis']))]
                _cand += [a for a in (1, 2, 0) if a not in _cand]
                for _ax in _cand:
                    _other = tuple(a for a in range(3) if a != _ax)
                    _viol = int((_dil.all(axis=_ax)).sum())
                    if _viol == 0:
                        if _ax != _cand[0]:
                            print(f"  [surfel] partial-body: march_axis "
                                  f"{_cand[0]} has body-spanning lines — "
                                  f"switched to axis {_ax}")
                        scfg['march_axis'] = _ax
                        break
                else:
                    raise ValueError(
                        "surfel partial_body: every march axis has lines "
                        "fully inside the body — the box cannot resolve "
                        "dV (enlarge the box on at least one axis)")
            sb = SurfelBoundary(
                self.xp, tuple(int(s) for s in mask.shape),
                geom_info['triangles_lu'],
                nu_lu=float(ctx.get('nu_lu', self.nu_lu)),
                cfg=scfg,
            )
            sb.partial_body = partial
            if partial and getattr(sb, 'tau_model_on', False) \
                    and ctx.get('region') is not None:
                # excise the tau-band rows inside this level's own C2F
                # band (outer overlap strips on non-flush faces): C2F
                # rewrites those cells every substep and the injection
                # there is the measured 74 conflict. Interior rows keep
                # the full band (they see no C2F).
                import numpy as _np
                reg = ctx['region']
                fr, fd = reg.fine_region, reg.fine_domain_coarse
                ratio = int(getattr(reg, 'REFINE_RATIO', 2))
                cells = _np.asarray(sb.d_tb_cells.get(), dtype=_np.int64)
                shp = sb.shape
                cz = cells % shp[2]
                cy = (cells // shp[2]) % shp[1]
                cx = cells // (shp[1] * shp[2])
                keepm = _np.ones(cells.size, dtype=bool)
                for ax_i, (cc, ax) in enumerate(zip((cx, cy, cz), 'xyz')):
                    w_lo = (getattr(fr, f'{ax}_start')
                            - getattr(fd, f'{ax}_start')) * ratio
                    w_hi = (getattr(fd, f'{ax}_end')
                            - getattr(fr, f'{ax}_end')) * ratio
                    n_ax = int(shp[ax_i])
                    if w_lo > 0:
                        keepm &= cc >= w_lo
                    if w_hi > 0:
                        keepm &= cc <= n_ax - 1 - w_hi
                if not keepm.all():
                    keep = _np.flatnonzero(keepm)
                    sb.d_tb_cells = sb.xp.asarray(cells[keep])
                    sb.d_tb_fs = sb.d_tb_fs[sb.xp.asarray(keep)]
                    sb.d_tb_normal = sb.d_tb_normal[sb.xp.asarray(keep)]
                    sb._tb_W = sb._tb_W[keep]
                    print(f"  [surfel] partial-body L{ctx.get('level')}: "
                          f"{int((~keepm).sum()):,} tau-band rows in the "
                          f"C2F strips excised ({keepm.size:,} total)")
            if partial:
                # invariant: with a valid march every cut cell is cut by
                # a present facet (whole-body levels measure exactly 0)
                import numpy as _np
                _has = _np.zeros(int(_np.prod(sb.shape)), dtype=bool)
                _has[_np.unique(sb.facets._t_cell)] = True
                # tolerance: the march's f64 line accumulation leaves
                # far-field fluid at 1 - eps (band-guard convention)
                _fl = int((sb.live_h.ravel()
                           & (sb.dV_h.ravel() < 1.0 - 1e-9)
                           & ~_has).sum())
                import os as _os2
                if _fl and _os2.environ.get('LBM_PARTIAL_SOFT') == '1':
                    print(f"  [surfel] SOFT: {_fl} facet-less cut cells "
                          f"on L{ctx.get('level')} (march axis "
                          f"{scfg.get('march_axis')})")
                elif _fl:
                    raise AssertionError(
                        f"surfel partial_body: {_fl} facet-less cut "
                        f"cells — dV/facet inconsistency (march axis or "
                        f"geometry problem)")
            sb.tri_global = None
            sb.n_faces_full = sb.n_faces
            sb.tri_owned = None
            if partial and ctx.get('region') is not None:
                import numpy as _np
                reg = ctx['region']
                fr, fd = reg.fine_region, reg.fine_domain_coarse
                ratio = int(getattr(reg, 'REFINE_RATIO', 2))
                v, f = geom_info['triangles_lu']
                cen = _np.asarray(v)[_np.asarray(f)].mean(axis=1)
                own = _np.ones(len(f), dtype=bool)
                for ax_i, ax in enumerate('xyz'):
                    lo = (getattr(fr, f'{ax}_start')
                          - getattr(fd, f'{ax}_start')) * ratio
                    hi_band = (getattr(fd, f'{ax}_end')
                               - getattr(fr, f'{ax}_end')) * ratio
                    n_ax = int(mask.shape[ax_i])
                    if lo > 0:
                        own &= cen[:, ax_i] >= lo
                    if hi_band > 0:
                        own &= cen[:, ax_i] <= n_ax - 1 - hi_band
                sb.tri_owned = _np.flatnonzero(own)
                print(f"  [surfel] partial-body L{ctx.get('level')}: "
                      f"{own.sum():,} of {own.size:,} triangles "
                      f"fine-owned (fine_region interior)")
            if self.xp.__name__ == 'cupy':
                # return the build transients (prism tables, band w_norm)
                # to the driver between per-level surfel builds — the
                # span16 4-level build runs at a few-GB margin (64 §13)
                self.xp.get_default_memory_pool().free_all_blocks()
            if ctx.get('region') is not None:
                # S8a-2 hard guards: span-through prism needs boundary-
                # flush z faces; no surfel stencil may touch the C2F/F2C
                # coupling band (see check_level_coupling_bands).
                check_level_coupling_bands(
                    sb, ctx['region'],
                    span_through_axis=geom_info.get('span_through_axis'),
                    level=ctx.get('level'),
                )
            # Cp/Cf reference for the surface writer (lattice units):
            # q_inf = 0.5 rho0 U_lu^2 from the configured freestream
            import numpy as _np
            _u0 = self.config.get('physics', {}).get(
                'initial_flow_velocity', (0.0, 0.0, 0.0))
            _umag = float(_np.linalg.norm(_np.asarray(_u0, dtype=float)))
            sb.q_inf = 0.5 * _umag ** 2 if _umag > 0.0 else None
            print(f"  Wall BC: {sb.summary()}")
            return sb

        if wall_bc_type != 'ibb':
            raise ValueError(
                f"Unknown wall_bc='{wall_bc_type}'. "
                f"Expected 'hwbb', 'ibb' or 'surfel'."
            )

        gtype = geom_info.get('type')
        dim = self.lattice.dim

        if dim == 3:
            return self._setup_ibb_links(mask, geom_info, gtype,
                                         periodic_axes=per_axes,
                                         cut_faces=cut_faces)

        # ── 2D: dense q_fraction (the D2Q9 IBB kernel indexes it directly) ──
        needs_bounce = compute_needs_bounce(
            self.xp, self.lattice, mask, periodic_axes=per_axes,
        )

        q_fraction = None

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
        else:
            raise ValueError(
                f"wall_bc='ibb' with dim={dim} geom type='{gtype}' has no "
                "q-source — the run would silently degrade to HWBB "
                "(q=0.5 everywhere). Use wall_bc='hwbb' explicitly, or add "
                "a q-source for this geometry type.")

        if os.environ.get('LBM_FORCE_Q_HALF', '0') == '1' and q_fraction is not None:
            q_fraction = self.xp.full_like(q_fraction, 0.5)
            print(f"  [SANITY] LBM_FORCE_Q_HALF=1 → q_fraction overridden to 0.5 "
                  f"(IBB linear formula degenerates to HWBB)")

        bc = InterpolatedBounceBack(
            self.xp, self.lattice, mask, q_fraction=q_fraction,
            periodic_axes=per_axes,
        )
        bc.cut_faces = cut_faces
        print(f"  {bc.get_info()}")
        return bc

    def _setup_ibb_links(self, mask, geom_info, gtype,
                         periodic_axes=None, cut_faces=()):
        """3D wall_bc='ibb' — built entirely in the sparse link representation.

        The dense twin above allocates (Q,)+shape twice (needs_bounce 27 B/cell
        + q_fraction 108 B/cell, plus a host copy of each) to carry ~10 links
        per 1000 cells. Measured build peak was 248 B/cell, which puts v2's L1
        (108.6 M cells) at ~27 GB — past a 24 GB card before a single field is
        allocated. Nothing here is ever of size (Q,)+shape.
        """
        from src.boundary.q_fraction import (
            broadcast_links_mid_slice,
            compute_boundary_links,
            compute_q_fraction_cylinder_axis_links,
            compute_q_fraction_sphere_links,
            count_seam_links,
        )
        from src.boundary.stl_geometry import (
            compute_q_fraction_triangles_links)

        shape = tuple(int(s) for s in mask.shape)
        link_cell, link_dir = compute_boundary_links(
            self.xp, self.lattice, mask, periodic_axes=periodic_axes,
        )
        link_q = None
        _span = geom_info.get('span_through_axis')
        _per_seam = periodic_axes
        if _per_seam is None:
            # Legacy torus enumeration: the seam count is diagnostic only,
            # and span-axis wraps are contractual, not seam crossings.
            _per_seam = ()
            if _span:
                _per_seam = (({'x': 0, 'y': 1, 'z': 2}[str(_span).lower()]
                              if isinstance(_span, str) else int(_span)),)
        n_seam = count_seam_links(self.xp, self.lattice, shape,
                                  link_cell, link_dir,
                                  periodic_axes=_per_seam)
        if periodic_axes is not None and n_seam:
            # By construction the enumeration and the counter share the
            # same wrap predicate — a nonzero count means they drifted.
            raise AssertionError(
                f"seam filter left {n_seam} wrapped links "
                f"(periodic_axes={periodic_axes}) — "
                f"compute_boundary_links/_suppress_wrapped_links and "
                f"count_seam_links disagree")

        if gtype == 'cylinder':
            axis = geom_info.get('axis', 'z')
            link_q = compute_q_fraction_cylinder_axis_links(
                self.xp, self.lattice, shape, link_cell, link_dir,
                center=geom_info['center'],
                radius=geom_info['radius'],
                axis=axis,
            )
            print(f"  Wall BC: Bouzidi IBB (analytic q from cylinder "
                  f"axis='{axis}', 3D)")
        elif gtype == 'sphere':
            link_q = compute_q_fraction_sphere_links(
                self.xp, self.lattice, shape, link_cell, link_dir,
                center=geom_info['center'],
                radius=geom_info['radius'],
            )
            print(f"  Wall BC: Bouzidi IBB (analytic q from sphere, 3D)")
        elif gtype == 'stl' and 'triangles_lu' in geom_info:
            q_stats = {}
            link_q = compute_q_fraction_triangles_links(
                self.xp, self.lattice, shape, link_cell, link_dir,
                triangles_lu=geom_info['triangles_lu'], stats=q_stats,
            )
            print(f"  Wall BC: Bouzidi IBB (ray-triangle q from STL, "
                  f"{geom_info.get('n_faces', '?')} faces)")
            # A ray miss on a watertight mesh is normally a real defect. On a
            # fine block whose face CUTS the body it is not: the roll-based
            # enumeration wraps, so the opposite face's outer cell layer gets
            # links to cells a box away, and those rays cross nothing. Name
            # which one this is instead of leaving the mesh accused.
            n_miss = int(q_stats.get('n_miss', 0))
            if n_seam:
                print(f"  [note] {n_seam:,} of {int(link_cell.size):,} links "
                      f"cross the box seam (roll wrap) — the body is cut by a "
                      f"face; {min(n_seam, n_miss):,} of the {n_miss:,} ray "
                      f"misses are these, not mesh defects")
            span_axis = geom_info.get('span_through_axis')
            if span_axis:
                # z-invariant prism contract: the mask is symmetrized to
                # the mid-slice section, and q must match — per-link q
                # from an unstructured side tessellation wobbles by the
                # chordal sagitta along z, which breaks quasi-2D slice
                # invariance at the wall (observed ~3e-3 after 24 steps).
                # Broadcasting the mid-slice q IS the ideal prism's q.
                ax = ({'x': 0, 'y': 1, 'z': 2}[span_axis.lower()]
                      if isinstance(span_axis, str) else int(span_axis))
                link_q = broadcast_links_mid_slice(
                    self.xp, shape, link_cell, link_dir, link_q, ax,
                )
                print("  span-through prism: q broadcast from mid slice "
                      "(z-invariant wall)")
        else:
            raise ValueError(
                f"wall_bc='ibb' with dim=3 geom type='{gtype}' has no "
                "q-source — the run would silently degrade to HWBB "
                "(q=0.5 everywhere). Use wall_bc='hwbb' explicitly, or add "
                "a q-source for this geometry type.")

        if os.environ.get('LBM_FORCE_Q_HALF', '0') == '1' and link_q is not None:
            link_q = self.xp.full_like(link_q, 0.5)
            print(f"  [SANITY] LBM_FORCE_Q_HALF=1 → q_fraction overridden to 0.5 "
                  f"(IBB linear formula degenerates to HWBB)")

        bc = InterpolatedBounceBack.from_links(
            self.xp, self.lattice, mask, link_cell, link_dir, link_q,
            periodic_axes=periodic_axes,
        )
        bc.cut_faces = tuple(cut_faces)
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

        # ── Output field units (output.units: 'lu' | 'phys') ──
        # One FieldUnits instance is shared by every VTK-family writer
        # (volume 2D/3D, MLG, planes) so the value/name mapping has a
        # single source. Coordinates stay L0-lu in both modes.
        from src.io.field_units import FieldUnits
        _units_mode = oc.get('units', 'phys')
        if _units_mode not in ('lu', 'phys'):
            raise ValueError(
                f"output.units={_units_mode!r}: use 'phys' (default — "
                "p_pa/velocity_ms) or 'lu' (lattice)")
        self.field_units = FieldUnits(_units_mode, self._unit_converter)
        for _ln in self.field_units.summary_lines():
            print(_ln)  # captured -> setup_log.txt (summary repeats it)

        is_restart = args.restart_latest or args.restart is not None
        clear_previous = args.clear or oc.get('clear_previous', False)

        print(f"  VTK output dir: {output_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  CSV output dir: {self._csv_dir}")

        # Only channels that will actually WRITE get their directory
        # created (empty vtk/checkpoints dirs for disabled channels were
        # pure noise); clear_previous still sweeps an EXISTING dir of a
        # disabled channel so stale files can't mix into a later run.
        vc = self._vtk_config
        vtk_enabled = (vc.get('enabled', True) and not args.no_vtk
                       and self.is_io_rank)
        # The MLG writers (built later, in _setup_mlg[_2d]) need the SAME
        # resolved dir and write-intent — a hardcoded './results/vtk'
        # fallback there once pointed a --no-vtk run at a stale series in
        # the CWD and the units guard killed every MPI rank on it.
        self._vtk_output_dir = output_dir
        self._vtk_write_enabled = bool(vtk_enabled)
        ckpt_enabled = self._checkpoint_config.get('enabled', True)
        if self.is_io_rank:
            setup_output_directories(
                output_dir=output_dir if vtk_enabled else None,
                checkpoint_dir=self.checkpoint_dir if ckpt_enabled else None,
                csv_dir=self._csv_dir,
                clear_previous=clear_previous,
                is_restart=is_restart,
                sweep_dirs=[output_dir, self.checkpoint_dir],
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

        # ── VTK Writer ── (vtk_enabled computed above, before dir setup)
        if vtk_enabled:
            self.vtk_writer = VTKWriter(
                output_dir=output_dir,
                domain_shape=self.domain_shape,
                precision=vc.get('precision', 'float32'),
                compression_level=vc.get('compression_level', 0),
                units=self.field_units,
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

        # ── Pressure probes (virtual microphones) ──
        # Parsed on EVERY rank (the MPI guard in build_output_manager must
        # see the request on all ranks to abort consistently); constructed
        # only on the IO rank of a single-process run.
        from src.io.probe_writer import parse_probe_config, PressureProbeManager
        self._probe_cfg = parse_probe_config(oc, self._dimension)
        self.probe_mgr = None
        if self._probe_cfg is not None and self.is_io_rank:
            self.probe_mgr = PressureProbeManager(
                self._probe_cfg, self._csv_dir,
                self._unit_converter, self._dimension)
            n = len(self._probe_cfg['points'])
            print(f"  Probes: {n} point(s), every "
                  f"{self._probe_cfg['interval']} step(s) "
                  f"-> {self.probe_mgr.csv_path}")

        # ── Plane slices (dense-in-time acoustic cuts) ──
        # Same rank rules as the probes (parse everywhere, build on IO
        # rank). Planes write under the VTK root even when the volume
        # channel is disabled — a plane-only acoustic run is legitimate.
        from src.io.plane_writer import parse_plane_config, PlaneWriterManager
        self._plane_cfg = parse_plane_config(oc, self._dimension)
        self.plane_mgr = None
        if self._plane_cfg is not None and self.is_io_rank:
            os.makedirs(output_dir, exist_ok=True)
            self.plane_mgr = PlaneWriterManager(
                self._plane_cfg, output_dir,
                self._unit_converter, self.field_units,
                precision=vc.get('precision', 'float32'))
            print(f"  Planes: {len(self._plane_cfg)} plane(s) "
                  f"-> {os.path.join(output_dir, 'planes')}/")

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
            _cut = getattr(self.obstacle_bc, 'cut_faces', ())
            if _cut:
                # L0 warning, not an error: a body touching a WALL face
                # (ground-mounted) is closed by that wall physically —
                # the wetted-surface MEM force is still meaningful, minus
                # the un-wetted contact patch's baseline-pressure term.
                # On MLG fine blocks (body cut by COUPLING faces) the same
                # condition is a hard error — see the MLG force selection.
                print(f"  [warn] body touches domain face(s) "
                      f"{', '.join(_cut)} — MEM force misses the contact "
                      f"patch (baseline-pressure term); interpret offsets "
                      f"accordingly")
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
        # 'U_inf' PRESENT (even 0.0 — hover is a value, not "unspecified")
        # → pass through; ABSENT → None, and downstream may estimate. The
        # old `> 0` test turned an explicit hover 0.0 into "unspecified",
        # so the CSV u_inf_used silently became a BEM induced-velocity
        # estimate instead of the configured freestream.
        pc = self._physics_config
        if 'U_inf' in pc:
            U_inf_phys = float(pc['U_inf'])                    # [m/s]
            u_inf_lu = U_inf_phys * self.dt_phys / self.dx_phys  # [Δx/Δt]
            u_inf_lu_arg = u_inf_lu
        else:
            U_inf_phys = 0.0
            u_inf_lu = 0.0
            u_inf_lu_arg = None

        # hub_center is given in L0 lattice units; rotor speed is given
        # as rpm. Convert to [m] / [rad/s] for downstream ALM code.
        import copy, math
        al_cfg = copy.deepcopy(self._al_cfg)
        dx = self._unit_converter.dx_phys

        def _rotor_lu_to_phys(rotor_cfg):
            # hub_center is L0 lattice units in config (single AND multi —
            # one contract); rpm -> omega [rad/s] when omega absent.
            if 'hub_center' in rotor_cfg:
                hc = rotor_cfg['hub_center']
                rotor_cfg['hub_center'] = [h * dx for h in hc]
            if 'rpm' in rotor_cfg and 'omega' not in rotor_cfg:
                rotor_cfg['omega'] = rotor_cfg['rpm'] * 2.0 * math.pi / 60.0

        if 'rotor' in al_cfg:
            _rotor_lu_to_phys(al_cfg['rotor'])
        for entry in al_cfg.get('rotors', []):
            _rotor_lu_to_phys(entry['rotor'] if 'rotor' in entry else entry)

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

        # Force ramp-up. Set per MODEL: only ActuatorLineModel.step reads
        # ramp_steps, so assigning it to a MultiRotorManager was inert — every
        # multi-rotor run started at full thrust regardless of the config.
        ramp = al_cfg.get('ramp_steps', 0)
        if ramp > 0 and self.al_model is not None:
            for _m in (getattr(self.al_model, 'models', None)
                       or [self.al_model]):
                _m.ramp_steps = ramp
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
        from src.grid.wall_coupling import parse_wall_coupling

        self._mlg_config = self.config.get('mlg', {})
        self._mlg_enabled: bool = self._mlg_config.get('enabled', False)
        # Parsed before the early returns so every path (disabled, 2D, 3D)
        # has the attribute the band guard reads.
        self._wall_coupling = parse_wall_coupling(self._mlg_config)

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
        if self._wall_coupling.relaxes_guard:
            from src.grid.wall_coupling import check_margin_vs_band
            check_margin_vs_band(self._wall_coupling, overlap_width)
            print(f"  Wall coupling: {self._wall_coupling.describe()}")

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
        self._partial_skip_levels = set()             # patch 77
        self._mlg_level_spacings = [(1.0, 1.0, 1.0)]  # Level 0 spacing

        from src.grid.block_tree import GridBlock, validate_block_tree
        root = GridBlock(level=0, index=0, uid=0, name='L0',
                         shape=coarse_shape, origin=(0.0, 0.0, 0.0),
                         spacing=1.0)
        blocks = [root]

        for k in range(1, num_levels):
            level_cfg = levels_config[k] if k < len(levels_config) else {}
            for j, region_cfg in enumerate(_mlg_regions_of(level_cfg, k)):
                name = region_cfg.get('name') or (
                    f"L{k}" if j == 0 and len(_mlg_regions_of(level_cfg, k)) == 1
                    else f"L{k}b{j}")

                # Config values are in L0 physical coordinates
                x_min_phys = region_cfg['x_min']
                x_max_phys = region_cfg['x_max']
                y_min_phys = region_cfg['y_min']
                y_max_phys = region_cfg['y_max']
                z_min_phys = region_cfg['z_min']
                z_max_phys = region_cfg['z_max']

                parent = _mlg_resolve_parent(
                    blocks, k - 1,
                    (x_min_phys, x_max_phys, y_min_phys,
                     y_max_phys, z_min_phys, z_max_phys),
                    region_cfg.get('parent'), name)

                # Convert to parent-local coordinates
                po = parent.origin           # parent origin, L0 units
                pd = (parent.spacing,) * 3   # parent spacing, L0 units

                local_x_min = round((x_min_phys - po[0]) / pd[0])
                local_x_max = round((x_max_phys - po[0]) / pd[0])
                local_y_min = round((y_min_phys - po[1]) / pd[1])
                local_y_max = round((y_max_phys - po[1]) / pd[1])
                local_z_min = round((z_min_phys - po[2]) / pd[2])
                local_z_max = round((z_max_phys - po[2]) / pd[2])
                # Announce the snap: the octo8 rotor boxes were silently
                # moved by this round() and nobody could see it in the log.
                _snap = max(
                    abs(local_x_min * pd[0] + po[0] - x_min_phys),
                    abs(local_x_max * pd[0] + po[0] - x_max_phys),
                    abs(local_y_min * pd[1] + po[1] - y_min_phys),
                    abs(local_y_max * pd[1] + po[1] - y_max_phys),
                    abs(local_z_min * pd[2] + po[2] - z_min_phys),
                    abs(local_z_max * pd[2] + po[2] - z_max_phys))
                if _snap > 1e-9:
                    print(f"  [note] fine region snapped to the parent grid "
                          f"(max shift {_snap:.4g} L0 cells)")

                fine_region = IndexBox(
                    x_start=local_x_min, x_end=local_x_max,
                    y_start=local_y_min, y_end=local_y_max,
                    z_start=local_z_min, z_end=local_z_max,
                )

                overlap_region = self._mlg_overlap_mgr.add_region(
                    coarse_shape=parent.shape,
                    fine_region=fine_region,
                    level_coarse=k - 1,
                    overlap_width=overlap_width,
                    name=name,
                )

                # Compute this block's physical origin and spacing
                fdc = overlap_region.fine_domain_coarse
                lu_k = self._mlg_scaler.get_level_units(k)
                new_origin = (
                    po[0] + fdc.x_start * pd[0],
                    po[1] + fdc.y_start * pd[1],
                    po[2] + fdc.z_start * pd[2],
                )
                blk = GridBlock(level=k, index=j, uid=len(blocks), name=name,
                                shape=overlap_region.fine_shape,
                                origin=new_origin, spacing=lu_k.dx,
                                region=overlap_region, parent=parent)
                parent.children.append(blk)
                blocks.append(blk)

                _tag = f"Level {k}" if blk.index == 0 else \
                    f"Level {k}.b{j} '{name}'"
                print(f"  {_tag}: phys region x[{x_min_phys},{x_max_phys}] "
                      f"y[{y_min_phys},{y_max_phys}] z[{z_min_phys},{z_max_phys}]"
                      + (f"  parent={parent.name}" if k > 1 else ""))
                print(f"            fine shape = {overlap_region.fine_shape}, "
                      f"excised = {overlap_region.excised.num_nodes:,} nodes, "
                      f"origin = ({new_origin[0]:.1f}, {new_origin[1]:.1f}, "
                      f"{new_origin[2]:.1f})")

        # nesting="error": Rule A violations (child band invading the
        # parent's own region → cells overwritten twice per coarse step)
        # were a print on this manual-placement path but a hard error on
        # the box_packing path — same defect, same verdict now.
        for _w in validate_block_tree(root, overlap_width, nesting="error"):
            print(f"  [warn] MLG blocks: {_w}")

        self._mlg_is_multiblock = any(
            len([b for b in blocks if b.level == k]) > 1
            for k in range(num_levels))
        if self._mlg_is_multiblock:
            print(f"  MLG: multi-block levels active "
                  f"({', '.join(b.name for b in blocks if b.level > 0)})")

        self._mlg_blocks = blocks
        self._mlg_root = root
        # Per-level origin/spacing stay available for the single-block path;
        # multi-block callers must read block.origin instead.
        if all(len([b for b in blocks if b.level == k]) == 1
               for k in range(num_levels)):
            self._mlg_level_origins = [
                next(b.origin for b in blocks if b.level == k)
                for k in range(num_levels)]
            self._mlg_level_spacings = [
                (next(b.spacing for b in blocks if b.level == k),) * 3
                for k in range(num_levels)]

        self._mlg_filter_level = filter_level

        # ── Which level owns each rotor (correctness, not convenience) ──
        self._resolve_alm_levels(num_levels)

        # ── MLG VTK writer ───────────────────────────────────────
        from src.io.mlg_vtk_writer import MLGVTKWriter
        self._mlg_vtk_writer = MLGVTKWriter(
            output_dir=self._vtk_output_dir,
            check_units=self._vtk_write_enabled,
            coarse_shape=(self.Nx, self.Ny, self.Nz),
            overlap_mgr=self._mlg_overlap_mgr,
            scaler=self._mlg_scaler,
            num_levels=num_levels,
            precision=self._vtk_config.get('precision', 'float32'),
            blocks=self._mlg_blocks,
            units=self.field_units,
        )
        print(f"  {self._mlg_vtk_writer.get_info()}")

    def _resolve_alm_levels(self, num_levels: int) -> None:
        """Decide which MLG level receives each rotor's body force.

        Sets `self._alm_rotor_levels` (one level per rotor, config order) and
        `self._alm_target_level` (the common level; Stage 1 requires agreement).

        This is a correctness gate, not a convenience: F2C overwrites a coarse
        level's excised region with fine-level data every coarse step, so a
        rotor forced on a level that a finer level excises has its momentum
        deleted silently — no error, and thrust reads HIGH because the induced
        downwash never develops. See src/grid/alm_placement for the rule and
        the 190-config parity evidence behind it.
        """
        self._alm_rotor_levels = []
        self._alm_rotor_blocks = []
        self._alm_target_level = 0
        if self.al_model is None or num_levels < 2:
            return

        from src.grid.alm_placement import (
            band_report, block_boxes_l0, excision_conflicts, rotor_extent,
            select_level)

        boxes = block_boxes_l0(self._mlg_blocks)
        models = getattr(self.al_model, 'models', None) or [self.al_model]
        names = getattr(self.al_model, 'names', None) or ['rotor']
        policy = str(self._mlg_config.get('alm_band_policy', 'warn')).lower()
        if policy not in ('warn', 'error'):
            # Any unknown string used to fall through to 'warn' — a user
            # who wrote 'strict'/'raise' believed the hard gate was armed.
            raise ValueError(
                f"mlg.alm_band_policy={policy!r}: expected 'warn' or 'error'")

        reports, exts = [], []
        for i, m in enumerate(models):
            nm = names[i] if i < len(names) else f'rotor_{i}'
            ext = rotor_extent(m, name=nm)
            rep = select_level(ext, boxes)
            exts.append(ext)
            reports.append(rep)
            self._alm_rotor_levels.append(rep.level)
            self._alm_rotor_blocks.append(rep.uid)

        # ── All rotors must land on one LEVEL (blocks may differ) ──
        distinct = sorted(set(self._alm_rotor_levels))
        if len(distinct) > 1:
            lines = [f"    rotor {i} '{r.rotor}' hub="
                     f"({exts[i].hub[0]:.1f},{exts[i].hub[1]:.1f},"
                     f"{exts[i].hub[2]:.1f}) lu -> L{r.level}"
                     + (f"   ({r.reason})" if r.level == 0 else "")
                     for i, r in enumerate(reports)]
            raise ValueError(
                "ALM rotor/level split — rotors do not all resolve to one MLG "
                "level:\n" + "\n".join(lines) + "\n"
                "  All rotors must sit inside ONE fine level's region. Enlarge "
                "that level's mlg.levels[k].region to cover every rotor disk, "
                "or move the outliers.")
        self._alm_target_level = distinct[0]

        # ── Check A: a finer level would erase this rotor (hard) ──
        for i, ext in enumerate(exts):
            conflicts = excision_conflicts(ext, self._alm_target_level, boxes)
            if conflicts:
                lv, ov = conflicts[0]
                raise ValueError(
                    f"ALM force would be silently erased: rotor '{ext.name}' is "
                    f"assigned to L{self._alm_target_level}, but L{lv}'s excised "
                    f"region overlaps its Gaussian support by "
                    f"({ov['x']:.2f}, {ov['y']:.2f}, {ov['z']:.2f}) L0 lu.\n"
                    f"  F2C overwrites the coarse excised region with fine data "
                    f"every step, and the fine level carries no ALM, so that "
                    f"momentum never reaches the fluid.\n"
                    f"  Fix: enlarge mlg.levels[{lv}].region so it CONTAINS the "
                    f"rotor disk (then the ALM moves there), or shrink it clear "
                    f"of the rotor.")

        # ── Check B: support spilling into the band / past the domain ──
        _by_uid = {b.uid: b for b in boxes}
        if self._alm_target_level > 0:
            for _i, ext in enumerate(exts):
                box = _by_uid[self._alm_rotor_blocks[_i]]
                rep = band_report(ext, box)
                if not (rep.band_overshoot or rep.domain_overshoot):
                    continue
                msg = (f"ALM rotor '{ext.name}' on L{box.level}: Gaussian support "
                       f"extends past the excised region — ")
                if rep.band_overshoot:
                    msg += "into the coupling band on " + ", ".join(
                        f"{f} by {v:.1f} fine cells"
                        for f, v in rep.band_overshoot.items())
                if rep.domain_overshoot:
                    msg += ("; OUTSIDE the fine domain on " + ", ".join(
                        f"{f} by {v:.1f} fine cells"
                        for f, v in rep.domain_overshoot.items())
                        + " (that force is lost)")
                if policy == 'error':
                    raise ValueError(msg)
                print(f"  [warn] {msg}")

        if self._alm_target_level > 0:
            print(f"\n  ALM target: Level {self._alm_target_level} "
                  f"(finest level containing every rotor disk)")
        else:
            print(f"\n  ALM target: Level 0 ({reports[0].reason})")

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
        return self._build_mlg_simulation_3d()

    def _fine_level_boundaries(self, blk) -> dict:
        """Domain BCs for an MLG fine block — only on its DOMAIN-flush faces.

        A fine block face is a coupling boundary EXCEPT where it coincides
        with the global domain face: there the C2F band collapses to width
        0 by design, and with boundaries_config={} the pull streaming's
        periodic wrap acted as the "BC" — the face pulled populations from
        the OPPOSITE face of the block. For a ground plane that is plainly
        wrong; it is why fine regions were forbidden from touching the
        domain (docs/manual/05 §5.7-5) and why v3 leaves 0..46 mm above
        ground at L0 (patch_notes/mlg_blocks/02 axis (1)).

        Returns the subset of self.config['boundaries'] whose faces this
        block is flush against, velocities converted to lattice units the
        same way _setup_boundaries does. u_lu is level-invariant here
        (acoustic scaling: dt and dx halve together), and density is
        dimensionless, so entries copy verbatim.

        Wall-plane caveat (documented, stage-2 candidate): hwbb places the
        wall 0.5*dx_k below node 0, so a fine wall sits 0.5*(1 - 2^-k) L0
        lu HIGHER than the same wall on L0 (L1: +0.25). The build prints
        the offset; making the planes coincide exactly (q-fraction face
        BC) is a separate, registered step.
        """
        import copy as _copy
        from src.boundary.bc_config import parse_face_config

        dom_faces = blk.domain_faces
        if dom_faces and blk.region is not None:
            # domain_faces tests the ARRAY extent (global_box), which
            # includes the C2F/F2C band: a region parked exactly
            # overlap_width cells off a domain wall puts its BAND on the
            # wall without being flush (octo8 v3 'outwash' z_min — the
            # designed "ground layer stays L0" configuration). The
            # coupling still owns such a face: the parent supplies the
            # band every substep and the parent's own BC handles the
            # wall. Only faces the OverlapRegion designates flush (band
            # width 0 by design) take a fine-level BC.
            _banded = [f for f in dom_faces
                       if not blk.region.flush_faces.get(f, False)]
            if _banded:
                print(f"  MLG block '{blk.name}': face(s) "
                      f"{', '.join(_banded)} touch the domain via the "
                      f"coupling band only (region interior) — no fine "
                      f"BC, parent coupling owns them")
                dom_faces = tuple(f for f in dom_faces
                                  if f not in _banded)
        if not dom_faces:
            return {}

        _SUPPORTED = {'hwbb', 'bounce_back', 'slip', 'symmetry', 'free_slip',
                      'neumann', 'zero_gradient', 'eq', 'equilibrium'}
        _WALLS = {'hwbb', 'bounce_back'}
        _OPPOSITE = {'x_min': 'x_max', 'x_max': 'x_min',
                     'y_min': 'y_max', 'y_max': 'y_min',
                     'z_min': 'z_max', 'z_max': 'z_min'}

        by_loc = {}
        for name, bc in (self.config.get('boundaries', {}) or {}).items():
            fc = parse_face_config(name, bc)
            by_loc[fc.location.value] = (name, bc, fc)

        uc = self._unit_converter
        out = {}
        for face in dom_faces:
            loc = face.replace('_', '')            # 'z_min' -> 'zmin'
            if loc not in by_loc:
                # Domain face without a BC = periodic axis. Flush on BOTH
                # ends is the span_through contract (the wrap IS the
                # physics — no face BC wanted); flush on one end only
                # would wrap this block into itself.
                if _OPPOSITE[face] in dom_faces:
                    continue
                raise ValueError(
                    f"MLG block '{blk.name}' (L{blk.level}) is flush with "
                    f"domain face {loc}, which has no BC (periodic axis) — "
                    f"the block would stream-wrap into itself there. Either "
                    f"span the whole axis (both faces flush) or pull the "
                    f"region off that face.")
            name, bc, fc = by_loc[loc]
            if fc.method not in _SUPPORTED:
                raise NotImplementedError(
                    f"MLG block '{blk.name}' (L{blk.level}) is flush with "
                    f"domain face {loc} (method='{fc.method}') — fine-level "
                    f"domain BC supports {sorted(_SUPPORTED)} for now. "
                    f"sponge needs thickness x2^k and per-dt strength "
                    f"rescaling plus a band-overlap decision; keep the "
                    f"region off this face until that lands.")
            # Band-only contact was filtered above, so every face here is
            # flush by design — the coupling has no band to overwrite the
            # BC with.
            entry = _copy.deepcopy(bc)
            entry['location'] = loc
            if 'velocity' in entry:
                v = entry['velocity']
                if isinstance(v, (list, tuple)):
                    entry['velocity'] = [uc.phys_to_lu_velocity(vi)
                                         for vi in v]
                else:
                    entry['velocity'] = uc.phys_to_lu_velocity(float(v))
            out[name] = entry
            if fc.method in _WALLS:
                shift = 0.5 * (1.0 - blk.spacing)
                print(f"    Level {blk.level} '{blk.name}': {loc} wall "
                      f"(hwbb) — halfway plane at -{0.5 * blk.spacing:g} "
                      f"L0 lu, {shift:g} L0 lu above the L0 wall plane. "
                      f"STD path keeps this offset; the ESO path corrects "
                      f"it to the exact global plane (q-face pass, "
                      f"eso_wall/09)")
        return out

    def _partition_surfel_ownership(self, simulations) -> None:
        """Partial-body levels (patch 74): finest wins. Walk surfel
        levels top-down; every triangle a finer level owns (its clipped
        face set, full-STL ids) is handed over by all coarser levels —
        their facets on those triangles drop out of the force ledger
        and the surface output (the coarse solution there is F2C-
        overwritten fine data anyway). No-op when no level is clipped
        (every existing run: no partition, bit-identical)."""
        import numpy as _np
        surf = [(k, s) for k, s in enumerate(simulations)
                if getattr(getattr(s, 'obstacle_bc', None), 'kind', None)
                == 'surfel']
        if not any(getattr(s.obstacle_bc, 'tri_owned', None) is not None
                   for _, s in surf):
            return
        owned_by_finer = _np.zeros(0, dtype=_np.int64)
        for k, s in sorted(surf, key=lambda t: -t[0]):
            sb = s.obstacle_bc
            mine = (_np.asarray(sb.tri_owned)
                    if getattr(sb, 'tri_owned', None) is not None
                    else _np.arange(sb.n_faces))
            # not-owned = finer-owned + (partial level) its own band
            # triangles outside fine_region — both excluded from THIS
            # level's ledger/surface
            not_mine = _np.union1d(
                owned_by_finer,
                _np.setdiff1d(_np.arange(sb.n_faces), mine,
                              assume_unique=False))
            if not_mine.size:
                n = sb.set_facet_ownership(not_mine)
                print(f"  [surfel] L{k}: {n:,} facets excluded from the "
                      f"ledger ({not_mine.size:,} triangles not owned "
                      f"here)")
            owned_by_finer = _np.union1d(
                owned_by_finer,
                _np.setdiff1d(mine, owned_by_finer))
        # tau-band excision under a PARTIAL child (patch 77): the
        # parent's band cells inside the child's fine_region are not the
        # parent's to evolve (finest wins; F2C rewrites them anyway) —
        # measured: the L3 band rows inside the LE-L4 box blew up at
        # coarse step 14 (persistent tau_out state x per-substep F2C
        # rewrite feedback). Whole-body children are untouched (the
        # L2-under-L3 configuration ran stably for the whole campaign —
        # bit-preserving there).
        import numpy as _np2
        for k, sim in surf:
            child = next((s2 for k2, s2 in surf if k2 == k + 1), None)
            if child is None or not getattr(child.obstacle_bc,
                                            'partial_body', False):
                continue
            sb_p = sim.obstacle_bc
            if not getattr(sb_p, 'tau_model_on', False):
                continue
            blk = next(b for b in self._mlg_blocks if b.level == k + 1)
            fr = blk.region.fine_region      # in PARENT (level-k) coords
            # NOTE (patch 77 iteration): an earlier variant ALSO excised
            # the parent's tau-band rows inside the child's box. With
            # the F2C wall-shell exclusion below, the parent OWNS its
            # near-wall shell there — excising its band recreated the
            # band-less near-wall regime and the run still blew at
            # ~100 steps (measured). The band stays intact.
            # F2C wall-shell exclusion (patch 77): the partial child's
            # near-wall constitution (0.5 sliver floor, skipped band,
            # different BL regime) is NOT what this parent's facet
            # machinery is consistent with — restricting it into the
            # parent's wall shell grew to a blow-up in ~14 coarse steps
            # (measured, band-height cells over the LE box). The parent
            # KEEPS ITS OWN near-wall solution: keep-mask = solid
            # dilated by 3 fluid layers; F2C accepts the child only
            # beyond it. The parent's wall physics there is the
            # campaign-validated one, and the box-interior surface
            # force is L4-owned regardless.
            solid = ~_np2.asarray(sb_p.live_h, dtype=bool)
            dil = solid.copy()
            for _ in range(3):
                nx_ = dil.copy()
                for ax3 in range(3):
                    for sh in (1, -1):
                        nx_ |= _np2.roll(dil, sh, axis=ax3)
                dil = nx_
            sb_p.f2c_wall_keep = sb_p.xp.asarray(
                dil.astype(_np2.uint8).ravel())
            print(f"  [surfel] L{k}: F2C wall-shell keep mask "
                  f"({int(dil.sum()):,} cells = solid+3 layers)")

        # surface output: the topmost level writes the whole wing,
        # merging the coarser levels' owned contributions (output layer)
        self._surfel_levels_for_surface = [s.obstacle_bc for _, s in
                                           sorted(surf, key=lambda t: t[0])]
        # force: the topmost level's ledger (what ForceManager reads)
        # adds every coarser level's OWNED-facet force, rescaled to the
        # top level's lattice units — facet force ~ area ~ dx^2, so
        # F_top = F_j * (2^top / 2^j)^2 (velocity is level-invariant
        # under acoustic scaling; the MLG force rebind uses the same
        # 2^k length scaling for the reference).
        k_top, s_top = max(surf, key=lambda t: t[0])
        s_top.obstacle_bc._force_merge = [
            (s.obstacle_bc, (2.0 ** (k_top - k)) ** 2)
            for k, s in surf if k < k_top]

    def _attach_trip_forcing(self, simulations) -> None:
        """Sustained trip strip (patch 66): attach a TripForcing to every
        surfel level whose box the global strip intersects. Runtime
        forcing, so this runs on FRESH AND RESTARTED builds alike (the
        IC seed is init-only; the trip is not). Config block absent or
        disabled = no attribute = bit-identical (gate T1).
        """
        tcfg = self.config.get('trip_forcing', {})
        if not (isinstance(tcfg, dict) and tcfg.get('enabled', False)):
            return
        if self.lattice.dim != 3:
            raise NotImplementedError("trip_forcing is 3D-only")
        import numpy as np
        from src.utilities.trip_forcing import TripForcing
        cfg = dict(tcfg)
        cfg.setdefault('span_z_lu', float(self.Nz))
        n_att = 0
        surfel_ks = [k for k, s2 in enumerate(simulations)
                     if getattr(getattr(s2, 'obstacle_bc', None),
                                'kind', None) == 'surfel']
        top_k = max(surfel_ks) if surfel_ks else -1
        for k, sim in enumerate(simulations):
            if getattr(getattr(sim, 'obstacle_bc', None),
                       'kind', None) != 'surfel':
                continue
            if k != top_k:
                # patch 77: trip on the FINEST surfel level only. The
                # coarse copies were tolerated as "washed by F2C", but
                # with a partial finest level the coarse kick + per-
                # substep F2C mixing at the strip destabilized L3
                # (blow-up exactly over the strip x-range, coarse step
                # 14-15) — and the finest copy is the physically
                # operative one anyway (finest wins).
                continue
            origin = ((0.0, 0.0, 0.0) if k == 0
                      else tuple(float(o)
                                 for o in self._mlg_level_origins[k]))
            dx = (1.0 if k == 0
                  else float(self._mlg_scaler.get_level_units(k).dx))
            shp = sim.domain_shape
            xg = origin[0] + np.arange(shp[0], dtype=np.float64) * dx
            yg = origin[1] + np.arange(shp[1], dtype=np.float64) * dx
            zg = origin[2] + np.arange(shp[2], dtype=np.float64) * dx
            sim._trip = TripForcing(self.xp, cfg, xg, yg, zg,
                                    level=k, lattice=self.lattice)
            # the MPI slab rebuilds the trip on its wrap window from
            # these args (surfel_level) — caches are lazy on both sides
            sim._trip_args = {'cfg': cfg, 'origin': origin, 'dx': dx,
                              'level': k}
            n_att += 1
        print(f"  [trip] sustained strip attached to {n_att} level(s): "
              f"amp={cfg['amp_lu']:g} [L0 lu], box_xy={cfg['box_lu']}, "
              f"modes={cfg['n_modes']}, seed={cfg['seed']}")

    def _build_mlg_simulation_3d(self) -> "MultiLevelGrid":

        xp = self.xp
        num_levels = self._mlg_config['num_levels']
        simulations = []
        couplings = []

        # ── Which level gets the ALM (decided in _resolve_alm_levels) ──
        alm_target_level = getattr(self, '_alm_target_level', 0)

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
            # eso implicit domain walls on MLG (eso_wall patch 04):
            # ROOT-LEVEL walls only. Fine sims also get the flag so an
            # inherited (flush) wall face converts loudly to a mask —
            # the initializer then hard-errors on any fine-level mask
            # (band-on-wall-axis is unsupported until the open-face
            # mailbox generalization; octo8 keeps fine bottoms 2 lu off
            # the ground by design, so only L0 carries the wall).
            eso_wall_implicit_ok=True,
        )
        simulations.append(sim_0)
        from src.utilities.build_census import build_census
        build_census('setup: L0 sim built')
        self._attach_coupling_skip(sim_0, self._mask, level=0)

        # ── Fine levels ──────────────────────────────────────────
        _blk_alms = []
        for _blk in self._mlg_blocks[1:]:
            k = _blk.level
            region = _blk.region
            lu = self._mlg_scaler.get_level_units(k)
            fine_shape = region.fine_shape  # (Nx_f, Ny_f, Nz_f)

            # Fine level streaming (different domain shape)
            fine_streaming = StreamingPull(
                xp, self.lattice, fine_shape,
            )

            # Fine level BC: coupling handles every face EXCEPT the ones
            # flush with the global domain — those get the domain's own
            # BCs (fine-level domain BC; patch_notes/mlg_blocks/03).
            fine_bounds = self._fine_level_boundaries(_blk)
            fine_bc_mgr = DomainBCManager(
                xp=xp,
                lattice=self.lattice,
                boundaries_config=fine_bounds,
                domain_shape=fine_shape,
                verbose=False,
            )
            if fine_bounds:
                print(f"    Level {k} '{_blk.name}': domain BC on flush "
                      f"face(s) "
                      + ", ".join(f"{e['location']}="
                                  f"{e.get('method', e.get('type', '?'))}"
                                  for e in fine_bounds.values()))

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
            fine_mask_k = None       # this level's solid mask, or None
            internal_geom = self.config.get('internal_geometry', {})
            if internal_geom:
                fine_origin = _blk.origin
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
                    self._mlg_fine_geom_configs[_blk.uid] = fine_geom_config

                if fine_geom_config:
                    fine_mask, fine_geom_info = create_geometry_mask(
                        xp, self.lattice, fine_shape,
                        fine_geom_config,
                        characteristic_length=None,
                        verbose=True,
                    )
                    n_solid = int(xp.sum(fine_mask))
                    if fine_geom_info['type'] != 'none' and n_solid > 0:
                        fine_mask_k = fine_mask
                        self._check_body_vs_coupling_band(k, region, fine_mask)
                        # Honor wall_bc (hwbb / ibb / surfel) on this MLG fine
                        # level too — NOT hardcoded HWBB. Without this, the BC
                        # requested in the config silently downgrades to HWBB
                        # on every level.
                        print(f"    Level {k}: building obstacle BC "
                              f"({n_solid:,} solid nodes)")
                        fine_obstacle_bc = self._build_obstacle_wall_bc(
                            internal_geom=internal_geom,
                            geom_info=fine_geom_info,
                            mask=fine_mask,
                            # surfel (S8a-2): level-local viscosity + the
                            # coupling region for the band guards
                            level_ctx={'level': k, 'nu_lu': lu.nu,
                                       'region': region},
                        )
                        if getattr(fine_obstacle_bc, 'kind', None) \
                                == 'surfel':
                            # surface VTK in the global (L0 lu) frame
                            fine_obstacle_bc.coord_origin = tuple(
                                float(o) for o in fine_origin)
                            fine_obstacle_bc.coord_spacing = float(lu.dx)

            # ── Fine-level ALM (if this is the target level) ─────
            fine_al_k = None
            _rot_idx = [i for i, u in enumerate(
                getattr(self, '_alm_rotor_blocks', [])) if u == _blk.uid]
            if _rot_idx and self.al_model is not None:
                fine_al_k = self._create_fine_level_alm(
                    k, fine_shape, block=_blk, rotor_indices=_rot_idx)
                _blk_alms.append((_blk, fine_al_k, _rot_idx))

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
                # see sim_0: converts loudly. A FLUSH fine wall is
                # legal since the open-face track (eso_wall/08-09):
                # its band faces are declared below and its q-face pass
                # puts the reflection plane on the global ground.
                eso_wall_implicit_ok=True,
            )
            # open-face declarations (eso_wall/09): coupling-band faces
            # (= non-flush faces) may sit opposite an inherited wall on
            # a de-periodized axis; q = 0.5*2^k for flush wall faces.
            _FB = {'x_min': 0, 'x_max': 1, 'y_min': 2, 'y_max': 3,
                   'z_min': 4, 'z_max': 5}
            _flush = (getattr(_blk.region, 'flush_faces', {}) or {}) \
                if _blk.region is not None else {}
            sim_k._eso_band_faces = 0
            for _fname, _bit in _FB.items():
                if not _flush.get(_fname, False):
                    sim_k._eso_band_faces |= 1 << _bit
            sim_k._eso_wall_q_scale = 0.5 * (2 ** k)
            simulations.append(sim_k)
            build_census(f'setup: L{k} sim built')
            _blk.sim = sim_k
            self._attach_coupling_skip(sim_k, fine_mask_k, level=k,
                                       label=_blk.name)

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
            _blk.coupling = coupling_k

        # ── Assemble MultiLevelGrid ──────────────────────────────
        self._mlg_root.sim = simulations[0]
        mlg = MultiLevelGrid.from_tree(self._mlg_root)
        print(f"\n  MultiLevelGrid assembled:")
        print(f"  {mlg.summary()}")

        self._attach_trip_forcing(simulations)
        self._partition_surfel_ownership(simulations)

        # ── Update al_model to fine-level for OutputManager ──────
        if len(_blk_alms) == 1:
            _b, _m, _ = _blk_alms[0]
            self.al_model = _m
            # Marker coordinate transform: fine local → global (L0 units)
            self._alm_marker_origin = _b.origin
            self._alm_marker_spacing = _b.spacing
        elif len(_blk_alms) > 1:
            # Rotors landed on different blocks. Each block's own manager does
            # the stepping; this aggregate exists only so output stays one
            # performance CSV and one marker VTP.
            from src.actuator.actuator_line import MultiRotorView
            _view = MultiRotorView(xp=xp)
            for _b, _m, _idx in _blk_alms:
                _sub = getattr(_m, 'models', None) or [_m]
                _nm = getattr(_m, 'names', None) or [f'rotor_{i}' for i in _idx]
                for _j, _mm in enumerate(_sub):
                    _view.attach(_mm, _nm[_j] if _j < len(_nm) else f'r{_j}',
                                 frame_origin=_b.origin, frame_spacing=_b.spacing)
            self.al_model = _view
            self._alm_marker_origin = None      # frames live on the models
            self._alm_marker_spacing = 1.0
            print(f"\n  ALM: {len(_view.models)} rotors across "
                  f"{len(_blk_alms)} blocks "
                  f"({', '.join(b.name for b, _, _ in _blk_alms)})")

        # ── MLG force: measure on finest level with obstacle ─────
        # Physical reason: the finest level has the most accurate
        # representation of the obstacle surface and flow field.
        # L0 f_post is captured before F→C coupling, so it does not
        # reflect the fine-grid solution.
        self._mlg_force_level: Optional[int] = None
        self._mlg_force_block: Optional[int] = None
        if self.force_mgr is not None:
            # Finest block that carries an obstacle. Scanning blocks (not
            # levels) keeps this right when a level hosts several: if more
            # than one candidate shares the finest level we refuse rather
            # than pick, since one ForceManager measures one body.
            _cands = [b for b in reversed(self._mlg_blocks)
                      if b.level > 0 and b.sim.obstacle_bc is not None]
            if _cands:
                _top = max(b.level for b in _cands)
                _same = [b for b in _cands if b.level == _top]
                if len(_same) > 1:
                    raise ValueError(
                        "force_calculation: level %d has %d blocks carrying an "
                        "obstacle (%s). One ForceManager measures one body — "
                        "split the run or restrict the geometry to one block."
                        % (_top, len(_same), ", ".join(b.name for b in _same)))
            for _fb in _cands[:1]:
                k = _fb.level
                _cut = getattr(_fb.sim.obstacle_bc, 'cut_faces', ())
                if _cut:
                    raise ValueError(
                        f"force_calculation: block '{_fb.name}' (L{k})'s "
                        f"body is cut by box face(s) {', '.join(_cut)} — "
                        f"MEM force on an open body is not the body force "
                        f"(the seam links that used to close the torus are "
                        f"suppressed; patch_notes/ibb_sparse/02 sec. 3). "
                        f"Disable force_calculation or enclose the body in "
                        f"one block.")
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
                ).get(_fb.uid, self.config.get('internal_geometry', {}))
                self.force_mgr = ForceManager(
                    xp=xp,
                    lattice=self.lattice,
                    solid_mask=_fb.sim.obstacle_bc.solid_mask,
                    config=fine_force_config,
                    wall_bc=_fb.sim.obstacle_bc,
                    csv_dir=self._csv_dir,
                    internal_geometry=fine_geom_for_force,
                )
                self.force_mgr.initialize()
                self._mlg_force_level = k
                self._mlg_force_block = _fb.uid
                print(f"\n  Force measurement: {_fb.label} "
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
            _snap = max(
                abs(local_x_min * pd[0] + po[0] - x_min_phys),
                abs(local_x_max * pd[0] + po[0] - x_max_phys),
                abs(local_y_min * pd[1] + po[1] - y_min_phys),
                abs(local_y_max * pd[1] + po[1] - y_max_phys))
            if _snap > 1e-9:
                print(f"  [note] fine region snapped to the parent grid "
                      f"(max shift {_snap:.4g} L0 cells)")

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
        self._mlg_vtk_writer = MLGVTKWriter2D(
            output_dir=self._vtk_output_dir,
            check_units=self._vtk_write_enabled,
            coarse_shape=(self.Nx, self.Ny),
            overlap_mgr=self._mlg_overlap_mgr,
            scaler=self._mlg_scaler,
            num_levels=num_levels,
            precision=self._vtk_config.get('precision', 'float32'),
            units=self.field_units,
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
        from src.utilities.build_census import build_census
        build_census('setup: L0 sim built')

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

            # Fine level BC: empty — 2D is OUTSIDE the fine-level domain
            # BC feature (3D: _fine_level_boundaries). GridCoupling2D
            # writes all 4 faces unconditionally (no flush_faces concept
            # in OverlapManager2D), so a face BC here would be overwritten
            # every substep. Introduce flush faces to the 2D coupling
            # first if this is ever needed.
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
                            # fine block: every face is a coupling boundary,
                            # nothing wraps (seam policy — see the method).
                            level_ctx={'level': k},
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
            build_census(f'setup: L{k} sim built')

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

        build_census('setup: MLG assembled')
        mlg = MultiLevelGrid(levels=simulations, couplings=couplings)
        print(f"\n  MultiLevelGrid (2D) assembled:")
        print(f"  {mlg.summary()}")

        # ── Force measurement on finest level with obstacle ──────
        self._mlg_force_level: Optional[int] = None
        if self.force_mgr is not None:
            for k in range(num_levels - 1, -1, -1):
                if simulations[k].obstacle_bc is not None and k > 0:
                    _cut = getattr(simulations[k].obstacle_bc,
                                   'cut_faces', ())
                    if _cut:
                        raise ValueError(
                            f"force_calculation: level {k}'s body is cut "
                            f"by box face(s) {', '.join(_cut)} — MEM force "
                            f"on an open body is not the body force "
                            f"(patch_notes/ibb_sparse/02 sec. 3). Disable "
                            f"force_calculation or enlarge the region to "
                            f"enclose the body.")
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

    def _create_fine_level_alm(self, level_k: int, fine_shape,
                               block=None, rotor_indices=None):
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
        # Origin of the BLOCK the rotors were assigned to; the
        # per-level list only exists for single-block grids.
        origin_L0 = (block.origin if block is not None
                     else self._mlg_level_origins[level_k])

        # Fine level physical scales
        dx_fine = self.dx_phys * lu_k.dx   # dx_phys / 2^k
        dt_fine = self.dt_phys * lu_k.dt   # dt_phys / 2^k

        # Fine level viscosity (in fine lattice units)
        nu_fine = (1.0 / 3.0) * (lu_k.tau - 0.5)

        # ── Prepare AL config with fine-level hub_center ─────────
        # Handles BOTH shapes: 'rotor' (single) and 'rotors' (multi). Every
        # rotor takes the same two steps — config L0 lu -> global [m], then
        # minus this level's origin -> level-local [m].
        al_cfg = copy.deepcopy(self._al_cfg)
        dx_uc = self._unit_converter.dx_phys
        origin_m = [o * self.dx_phys for o in origin_L0]

        if 'rotors' in al_cfg:
            # Only the rotors assigned to this block; the others are built by
            # their own blocks.
            if rotor_indices is not None:
                al_cfg['rotors'] = [al_cfg['rotors'][i] for i in rotor_indices]
            rotor_cfgs = [e.get('rotor', e) for e in al_cfg['rotors']]
        else:
            rotor_cfgs = [al_cfg['rotor']]

        hub_locals = []
        for rotor_cfg in rotor_cfgs:
            if 'hub_center' in rotor_cfg:
                hc = rotor_cfg['hub_center']
                rotor_cfg['hub_center'] = [h * dx_uc for h in hc]
            if 'rpm' in rotor_cfg and 'omega' not in rotor_cfg:
                rotor_cfg['omega'] = rotor_cfg['rpm'] * 2.0 * math.pi / 60.0
            hub_local_m = [h - o for h, o
                           in zip(rotor_cfg['hub_center'], origin_m)]
            rotor_cfg['hub_center'] = hub_local_m
            hub_locals.append(hub_local_m)
        hub_local_m = hub_locals[0]

        # ── u_inf (same in lattice units across all levels) ──────
        pc = self._physics_config
        U_inf_phys = pc.get('U_inf', 0.0)
        u_inf_lu = (U_inf_phys * dt_fine / dx_fine
                    if U_inf_phys > 0 else None)

        # ── Create fine-level ALM ────────────────────────────────
        # Single-rotor keeps the exact original call (same factory, same
        # argument list) so existing configs stay bit-identical; multi-rotor
        # goes to the multi factory, which returns a MultiRotorManager sized
        # to the fine grid.
        from src.actuator.actuator_line import (
            create_actuator_line_from_config, create_multi_rotor_from_config)

        _factory = (create_multi_rotor_from_config if 'rotors' in al_cfg
                    else create_actuator_line_from_config)
        fine_al = _factory(
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

        # Ramp steps (scale to fine timesteps for same physical duration).
        # Set per MODEL: on a MultiRotorManager the attribute is inert because
        # step() never reads it — only ActuatorLineModel.step does.
        ramp = al_cfg.get('ramp_steps', 0)
        if ramp > 0:
            for _m in (getattr(fine_al, 'models', None) or [fine_al]):
                _m.ramp_steps = ramp * (2 ** level_k)

        # ── Print info ───────────────────────────────────────────
        _btag = f" block '{block.name}'" if block is not None else ""
        print(f"\n  Fine-level ALM (Level {level_k}{_btag}):")
        _fine_models = getattr(fine_al, 'models', None) or [fine_al]
        _names = getattr(fine_al, 'names', None) or ['rotor']
        for _i, _m in enumerate(_fine_models):
            _r = _m.rotor
            _nm = _names[_i] if _i < len(_names) else f'rotor_{_i}'
            _hl = hub_locals[_i] if _i < len(hub_locals) else hub_local_m
            print(f"    [{_i}] {_nm}: hub_local [m] = "
                  f"[{_hl[0]:.4f}, {_hl[1]:.4f}, {_hl[2]:.4f}]"
                  f"  [lu] = ({_r.hub_center[0]:.1f}, "
                  f"{_r.hub_center[1]:.1f}, {_r.hub_center[2]:.1f})")
            print(f"         R = {_r.radius:.1f} fine lu, "
                  f"markers = {_r.total_markers}, "
                  f"omega = {_r.omega:.6f} rad/lt_fine, "
                  f"ramp = {_m.ramp_steps}")
        print(f"    dx = {dx_fine*1000:.4f} mm, "
              f"dt = {dt_fine*1e6:.4f} us")
        print(f"    nu = {nu_fine:.6e}, "
              f"tau = {lu_k.tau:.6f}")

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
            xp = self.xp
            q = self.obstacle_bc.q_fraction
            if q is None:                        # link mode (3D production)
                n_links = int(self.obstacle_bc.n_links)
                n_sentinel = int(xp.sum(
                    self.obstacle_bc.link_q == xp.float32(0.5))) \
                    if n_links else 0
            else:
                nb = self.obstacle_bc.needs_bounce
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
                    # Sum over every block on this level — get_region(k-1)
                    # refuses to guess when there is more than one.
                    _rs = self._mlg_overlap_mgr.regions_at(k - 1)
                    shape = _rs[0].fine_shape
                    tau_k = self._mlg_scaler.get_level_units(k).tau

                # Works for any dimensionality
                nodes = 0
                for _r in ([None] if k == 0 else _rs):
                    _n = 1
                    for s in (shape if k == 0 else _r.fine_shape):
                        _n *= s
                    nodes += _n
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

            # MLG fine-level scaling factor. Must follow the level the ALM
            # actually lands on (_resolve_alm_levels) — assuming the finest
            # level printed "@L3, R=16 lu" for a rotor sitting on L0.
            _mlg_scale = 1
            _level_tag = ""
            if self._mlg_enabled:
                _alv = getattr(self, '_alm_target_level', 0)
                _mlg_scale = 2 ** _alv
                _level_tag = f" @L{_alv}"

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

        # ── Output units (always shown: applied units, or the lattice->
        #    physical recipe with this run's actual constants) ──────
        if getattr(self, 'field_units', None) is not None:
            for line in self.field_units.summary_lines():
                print(line)

        # ── Log file ─────────────────────────────────────────────
        print(f" Log    : {self._log_path}")
        print(sep)