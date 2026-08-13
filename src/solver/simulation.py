"""
LBM Simulation — Single Timestep Physics

This module defines the Simulation class, which encapsulates the core LBM
physics for one timestep. The `advance()` method performs all physical
operations in a clear, readable sequence matching the continuous equations:

    1. Macroscopic (ρ, u from f)
    2. Body Force  (ALM pipeline → F(x))
    3. Velocity Correction (Guo: u += F/(2ρ))
    4. Collision (equilibrium + forcing + relaxation, all internal)
    5. Streaming (pull: f' = f*(x - c_i))
    6. Boundary Conditions (domain faces + internal obstacles)
    7. Buffer Swap (f ← f')

Design Principle:
    "main은 실행 진입점, Simulation이 물리를 서술한다."
    → 물리적 흐름이 advance() 안에서 직관적으로 확인된다.

References:
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017
    - Guo, Zheng, Shi, Phys. Rev. E 65, 046308, 2002

Author: LBM Development Team
Date: 2026-02
"""

import os
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.macroscopic.compute import Macroscopic
    from src.collision.base import CollisionOperator
    from src.streaming.stream import StreamingPull
    from src.boundary.domain_bc_manager import DomainBCManager
    from src.boundary.wall import HalfwayBounceBack


class Simulation:
    """Core LBM simulation state and single-step physics.

    This class owns the distribution function buffers (f, f_post)
    and all physical operators. One call to `advance()` performs exactly
    one LBM timestep. Streaming writes directly to f (no f_new buffer).

    Attributes:
        f:          Current distribution function (Q, Nx, Ny[, Nz])  [dimensionless]
        rho:        Density field (Nx, Ny[, Nz])                     [dimensionless]
        u:          Velocity field (dim, Nx, Ny[, Nz])               [Δx/Δt]
                    (Guo force-corrected when body_force is active)
        body_force: External body force (dim, Nx, Ny[, Nz])          [lattice force / lu³]
                    or None if no ALM is active
        f_post:     Post-collision distribution (read-only access)    [dimensionless]
                    Needed by MEM force calculation (ForceManager)
        step_count: Number of advance() calls since initialization

    Example:
        >>> sim = Simulation(xp=xp, macroscopic=macro, equilibrium=eq,
        ...                  collision=collision, streaming=streaming,
        ...                  forcing_scheme=forcing, bc_manager=bc_mgr,
        ...                  tau=0.6, domain_shape=(100, 50, 50))
        >>> sim.set_distribution(f_init)
        >>> for step in range(1000):
        ...     sim.advance()
        ...     # Access sim.rho, sim.u for output
    """

    def __init__(
        self,
        xp: 'ModuleType',
        macroscopic: 'Macroscopic',
        collision: 'CollisionOperator',
        streaming: 'StreamingPull',
        bc_manager: 'DomainBCManager',
        tau: float,
        domain_shape: Tuple[int, ...],
        obstacle_bc: Optional['HalfwayBounceBack'] = None,
        al_model: Optional[object] = None,
        sgs_cfg: Optional[dict] = None,
        eso_wall_implicit_ok: bool = False,
    ) -> None:
        """Initialize Simulation with all physical operators.

        Args:
            xp: Array module (numpy or cupy)
            macroscopic: Macroscopic variable calculator (ρ, u from f)
            collision: Collision operator (BGK, Cumulant, etc.)
                       Owns equilibrium and forcing internally.
            streaming: Streaming operator (pull scheme)
            bc_manager: Domain boundary condition manager
            tau: Relaxation time  [Δt]
                 ν = cs²·(τ - 0.5) in lattice units  [Δx²/Δt]
            domain_shape: Grid dimensions (Nx, Ny) or (Nx, Ny, Nz)  [lu]
            obstacle_bc: Internal obstacle BC (HalfwayBounceBack), optional
            al_model: Actuator Line model (ActuatorLineModel or
                      MultiRotorManager), optional
            sgs_cfg: Validated SGS config dict with keys
                     {enabled, model, Cs, Cw}, or None for SGS-disabled.
                     Produced by src.turbulence.sgs.parse_sgs_config.
            eso_wall_implicit_ok: opt-in for the esoteric implicit
                     domain-wall faces (face mailbox, eso_wall track).
                     ONLY the single-grid setup path passes True; MLG
                     member sims and every MPI path keep the legacy
                     degradation (loud) until PLAN sec 4 steps 5-6 wire
                     the coupling/runner surfaces.
        """
        # ── Array module ──
        self.xp = xp

        # ── Physical operators ──
        self.macroscopic = macroscopic
        self.collision = collision
        self.streaming = streaming

        # ── Boundary conditions ──
        self.bc_manager = bc_manager
        self.obstacle_bc = obstacle_bc

        # ── Actuator Line ──
        self.al_model = al_model

        # ── SGS / turbulence model ──
        # Default to disabled if caller did not supply a parsed dict.
        self._sgs_cfg = sgs_cfg if sgs_cfg is not None else {
            "enabled": False, "model": "off", "Cs": 0.18, "Cw": 0.5,
        }

        # ── Parameters ──
        self.tau: float = tau                        # [Δt]
        self.domain_shape: Tuple[int, ...] = domain_shape  # [lu]

        # ── State (populated by advance()) ──
        self.rho: Optional['npt.NDArray'] = None     # [dimensionless]
        self.u: Optional['npt.NDArray'] = None       # [Δx/Δt] (force-corrected)
        self.body_force: Optional['npt.NDArray'] = None  # [lattice force / lu³]
        # SGS eddy viscosity buffer (lattice units). Allocated in
        # set_distribution() iff SGS is enabled. None -> kernel skips writing.
        self.nu_t: Optional['npt.NDArray'] = None    # [lu²/lt]

        # ── Distribution buffers (populated by set_distribution()) ──
        self.f: Optional['npt.NDArray'] = None       # Current distribution  [dimensionless]
        self._f_post: Optional['npt.NDArray'] = None # Post-collision buffer [dimensionless]

        # ── Counters ──
        self.step_count: int = 0
        self._is_ready: bool = False

        # ── Fused CUDA kernel (lazy init) ──
        self._fused_kernel = None
        self._use_fused: bool = False
        self._fused_is_cumulant: bool = False

        # ── Surfel wall boundary (S8a, patch_notes/surfel/46-47) ──
        # Detected by the adapter's `kind` tag: the surfel scheme REPLACES
        # streaming, so it owns a dedicated advance path instead of the
        # post-stream apply_with_reset hook — an unknown obstacle type
        # must never silently fall through to the HWBB-style hook.
        self._use_surfel: bool = (
            getattr(obstacle_bc, 'kind', None) == 'surfel')
        # fused-collide surfel path (patch 55) — lazy init/eligibility
        self._surfel_ck = None
        self._sck_rho = None
        self._sck_u = None
        self._surfel_ck_ok: Optional[bool] = None

        # ── Esoteric Pull path (opt-in via env LBM_ESOTERIC; lazy init) ──
        # Single-buffer in-place streaming+collision+BC (Lehmann 2022).
        # Resurrected from commit 8036317. See patch_notes/hpc_upgrade/15.
        self._use_esoteric: bool = False
        self._esoteric_kernel = None
        self._esoteric_is_cumulant: bool = False
        self._esoteric_step: int = 0
        self._esoteric_f_already_set: bool = False
        self._eso_ibb = None   # EsotericIBBD3Q27 (deposit-rewrite IBB, S5)
        # ── eso domain wall (eso_wall track, patch_notes/eso_wall) ──
        # _eso_wall_mask: 6-bit wall faces (0=xmin..5=zmax), set by
        # _build_esoteric_domain_bc iff eso_wall_implicit_ok.
        self._eso_wall_implicit_ok: bool = eso_wall_implicit_ok
        self._eso_wall_mask: int = 0
        self._eso_wall_mail: Optional['npt.NDArray'] = None
        self._eso_from_checkpoint: bool = False   # set_distribution flag
        # open-face track (eso_wall/08-09): band faces = coupling faces
        # the MLG builder declares on fine sims (allowed opposite a wall
        # on a de-periodized axis — the C2F scatter prescribes their
        # inbound mailbox each substep). q scale = 0.5 * 2^level: the
        # Bouzidi q that puts a FLUSH fine wall's reflection plane on
        # the global ground (-0.5 dx0). 0.5 = halfway (L0; pass off).
        self._eso_band_faces: int = 0
        self._eso_wall_q_scale: float = 0.5
        self._eso_wall_face_q: Optional[dict] = None

        # ── WALE pre-pass kernels (lazy init for sgs_model="wale") ──
        # The WALE eddy-viscosity needs u globally before collision so the
        # gradient stencil can read neighbours. We compute (rho, u) via a
        # macro pre-pass, then nu_t via a separate kernel, then run the
        # cumulant collide reading nu_t from a buffer.
        self._macro_kernel = None
        self._wale_kernel = None
        self._u_buf = None      # (dim, N) flat for pre-pass writes
        self._rho_buf = None    # (N,)     flat
        self._nu_t_in = None    # (N,)     fed to cumulant WALE branch

        # ── Obstacle BC kernel (lazy init in set_distribution) ──
        # Only one of _hwbb_kernel / _ibb_kernel is set at a time, depending
        # on isinstance(obstacle_bc, ...). Both None → Python path.
        self._ibb_kernel = None

    # =====================================================================
    # Public Interface
    # =====================================================================

    def set_distribution(self, f: 'npt.NDArray',
                         from_checkpoint: bool = False) -> None:
        """Set distribution function and allocate work buffer.

        from_checkpoint: f is a restored checkpoint state, NOT a fresh
        IC. The esoteric init must then skip BOTH IC seeders (solid
        bounce + domain wall): scattering the checkpointed std f at
        parity 0 already reproduces every bounce/wall read bit-exactly
        (gather/scatter LOAD-view fidelity), and seeding would overwrite
        the true deposits with IC-rule values — the exact failure mode
        the eso_seed_solid_bounce_ic docstring warns about (the single-
        grid restart path used to call it anyway; fixed with the wall
        track's §4-3).

        This is the bridge between Initialization and Execution:
        after calling this, advance() can be called.

        Only one work buffer (f_post) is allocated. Streaming writes
        directly back to f (pull scheme: source=f_post, dest=f, no conflict).

        If running on GPU with BGK D3Q27, a fused CUDA kernel is used
        for macroscopic + collision (single kernel launch instead of ~15).

        Args:
            f: Initial distribution function, shape (Q, Nx, Ny[, Nz])
               [dimensionless]
        """
        self.f = f
        self._is_ready = True
        self._eso_from_checkpoint = bool(from_checkpoint)

        # ── Esoteric Pull (opt-in: env LBM_ESOTERIC=1, BGK D3Q27, GPU) ──
        # Single-buffer in-place streaming+collision+BC. NO _f_post buffer
        # (that is the memory saving). Non-esoteric paths untouched below
        # (zero regression when off). Phase 1a: BGK only. See patch 15.
        self._use_esoteric = False
        if (os.environ.get("LBM_ESOTERIC", "0") == "1"
                and self.xp.__name__ == 'cupy'
                and len(self.domain_shape) == 3):
            from src.collision.bgk import BGKCollision
            from src.collision.cumulant import CumulantCollision
            if isinstance(self.collision, (BGKCollision, CumulantCollision)):
                # IBB is supported on the single-GPU esoteric path since S5
                # (two-phase deposit-rewrite pass built in _init_esoteric).
                # Init failure used to warn and continue on the standard
                # path — on MLG that could leave a MIXED per-level state
                # (some levels esoteric, some standard; the L3 OOM case in
                # driver._fail_fast_config history). Esoteric was requested
                # explicitly (env/config), so failure is fatal.
                self._init_esoteric(f)
        if self._use_esoteric:
            return  # single-buffer: skip _f_post / SGS / fused setup below

        self._f_post = self.xp.empty_like(f)

        # ── SGS eddy-viscosity output buffer ──
        # Allocate iff a non-trivial SGS model is in use, so the kernel can
        # write nu_t per cell for diagnostics / VTK export.
        if self._sgs_cfg["model"] in ("smagorinsky", "wale", "dyn_smag"):
            self.nu_t = self.xp.zeros(self.domain_shape, dtype=f.dtype)
        else:
            self.nu_t = None

        # ── WALE pre-pass scratch buffers ──
        # The cumulant collide WALE branch reads nu_t from a flat (N,) buffer.
        # The pre-pass kernels write into flat (N,) and (dim, N) views.
        if self._sgs_cfg["model"] in ("wale", "dyn_smag"):
            N = 1
            for d in self.domain_shape:
                N *= int(d)
            dim = len(self.domain_shape)
            self._rho_buf = self.xp.empty(N, dtype=f.dtype)
            self._u_buf   = self.xp.empty((dim, N), dtype=f.dtype)
            self._nu_t_in = self.xp.empty(N, dtype=f.dtype)

        # ── Try to initialize CUDA kernels ──
        self._use_fused = False
        self._fused_is_cumulant = False
        # (streaming CUDA kernel is now handled inside StreamingPull)
        self._hwbb_kernel = None

        # Track whether the fused kernel is the D2Q9 variant (different
        # launch signature: omega_3 + omega_4 instead of omega_high).
        self._fused_is_d2q9 = False

        if self.xp.__name__ == 'cupy' and len(self.domain_shape) == 3:
            # Every kernel below takes const float*; RawKernel passes the
            # array pointer without checking dtype, so a float64 field is
            # read as garbage rather than raising. Same contract as the
            # esoteric guard in _init_esoteric.
            import numpy as _np
            if self.f.dtype != _np.float32:
                raise ValueError(
                    f"GPU D3Q27 kernels are float32-only (got "
                    f"{self.f.dtype}); set simulation precision: 'float32', "
                    "or device_mode: 'cpu' for a float64 reference run")
            from src.collision.bgk import BGKCollision
            from src.collision.cumulant import CumulantCollision

            sgs_model = self._sgs_cfg["model"]   # "off" | "smagorinsky" | "wale" | "dyn_smag"
            # Cumulant collide branch: dyn_smag reuses the WALE block (both
            # just read nu_t from a buffer). sgs_model_kernel -> what the
            # cumulant kernel template is built with.
            sgs_model_kernel = "wale" if sgs_model == "dyn_smag" else sgs_model

            # Collision kernel
            if isinstance(self.collision, BGKCollision):
                if self._sgs_cfg["enabled"]:
                    raise NotImplementedError(
                        "SGS turbulence models are wired to the cumulant "
                        "collision only; BGK + SGS is not supported."
                    )
                from src.kernels.bgk_d3q27 import BGKCollideKernelD3Q27
                self._fused_kernel = BGKCollideKernelD3Q27()
                self._use_fused = True
            elif isinstance(self.collision, CumulantCollision):
                # The std fused kernel relaxes ALL higher cumulants at the
                # single rate it receives as omega_high (launch feeds
                # omega_3) and implements no limiter. A config asking for
                # distinct omega_4/omega_5/limiter must not run with those
                # knobs dropped — the esoteric kernel supports them.
                _c = self.collision
                _dropped = []
                if _c.omega_4 != _c.omega_3:
                    _dropped.append(f"omega_4={_c.omega_4}")
                if _c.omega_5 != _c.omega_3:
                    _dropped.append(f"omega_5={_c.omega_5}")
                if _c.omega_high != _c.omega_3:
                    _dropped.append(f"omega_6..10(=omega_high={_c.omega_high})")
                if float(getattr(_c, 'cumulant_limiter', 0.0)) != 0.0:
                    _dropped.append(f"cumulant_limiter={_c.cumulant_limiter}")
                if _dropped:
                    raise ValueError(
                        "std fused D3Q27 cumulant kernel relaxes all higher "
                        f"cumulants at omega_3={_c.omega_3} and has no "
                        "limiter; it cannot honor: " + ", ".join(_dropped)
                        + ". Use the esoteric path (omega_3/omega_4/limiter "
                        "supported), align the values with omega_3, or "
                        "implement per-order rates in the std kernel first.")
                from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
                self._fused_kernel = CumulantCollideKernelD3Q27(sgs_model=sgs_model_kernel)
                self._use_fused = True
                self._fused_is_cumulant = True
                if sgs_model != "off":
                    print(f"  [fused kernel] D3Q27 cumulant + SGS={sgs_model}")

                # WALE / dyn_smag pre-pass kernels (3D)
                # Both feed nu_t to the cumulant collide WALE branch.
                if sgs_model == "wale":
                    from src.kernels.macro_d3q27 import MacroKernelD3Q27
                    from src.kernels.wale_d3q27 import WALEKernelD3Q27
                    self._macro_kernel = MacroKernelD3Q27()
                    self._wale_kernel  = WALEKernelD3Q27()
                elif sgs_model == "dyn_smag":
                    from src.kernels.macro_d3q27 import MacroKernelD3Q27
                    from src.kernels.dyn_smag_d3q27 import DynSmagKernelD3Q27
                    self._macro_kernel = MacroKernelD3Q27()
                    self._wale_kernel  = DynSmagKernelD3Q27()

            # Obstacle BC kernel: dispatch on type (HWBB vs IBB).
            if self.obstacle_bc is not None:
                from src.boundary.wall import HalfwayBounceBack
                if type(self.obstacle_bc) is HalfwayBounceBack:
                    from src.kernels.bounce_back_d3q27 import HWBBKernelD3Q27
                    self._hwbb_kernel = HWBBKernelD3Q27()
                else:
                    from src.boundary.interpolated_wall import InterpolatedBounceBack
                    if isinstance(self.obstacle_bc, InterpolatedBounceBack):
                        from src.kernels.interpolated_wall_d3q27 import IBBKernelD3Q27
                        self._ibb_kernel = IBBKernelD3Q27()

        elif self.xp.__name__ == 'cupy' and len(self.domain_shape) == 2:
            # D2Q9 fused path (currently only Cumulant).
            import numpy as _np

            # The D2Q9 kernel set indexes f as q*N+idx in int32 (the D3Q27
            # set was ported to 64-bit, this one was not): past
            # N = 2^31/9 the flat index overflows and the kernels corrupt
            # memory instead of raising.
            _N_cells = 1
            for _s in self.domain_shape:
                _N_cells *= int(_s)
            if 9 * _N_cells > 2 ** 31:
                raise NotImplementedError(
                    f"D2Q9 CUDA kernels are int32-indexed and this grid "
                    f"(N={_N_cells:,}) exceeds the 2^31/9 ceiling (~238M "
                    "cells per level). 64-bit indexing for the D2Q9 kernel "
                    "set must be implemented before running grids this "
                    "large (follow the D3Q27 64-bit port).")

            # Kernels are float32-only (const float*; RawKernel does not
            # check array dtype — float64 would read as garbage).
            if self.f.dtype != _np.float32:
                raise ValueError(
                    f"GPU D2Q9 kernels are float32-only (got "
                    f"{self.f.dtype}); set simulation precision: 'float32', "
                    "or device_mode: 'cpu' for a float64 reference run")

            from src.collision.cumulant_d2q9 import CumulantCollisionD2Q9

            if isinstance(self.collision, CumulantCollisionD2Q9):
                sgs_model = self._sgs_cfg["model"]
                sgs_model_kernel = "wale" if sgs_model == "dyn_smag" else sgs_model
                from src.kernels.cumulant_d2q9 import CumulantCollideKernelD2Q9
                self._fused_kernel = CumulantCollideKernelD2Q9(
                    sgs_model=sgs_model_kernel,
                )
                self._use_fused = True
                self._fused_is_cumulant = True
                self._fused_is_d2q9 = True

            # WALE / dyn_smag pre-pass kernels (2D)
            if self._sgs_cfg["model"] == "wale":
                from src.kernels.macro_d2q9 import MacroKernelD2Q9
                from src.kernels.wale_d2q9 import WALEKernelD2Q9
                self._macro_kernel = MacroKernelD2Q9()
                self._wale_kernel  = WALEKernelD2Q9()
            elif self._sgs_cfg["model"] == "dyn_smag":
                from src.kernels.macro_d2q9 import MacroKernelD2Q9
                from src.kernels.dyn_smag_d2q9 import DynSmagKernelD2Q9
                self._macro_kernel = MacroKernelD2Q9()
                self._wale_kernel  = DynSmagKernelD2Q9()

            # Obstacle BC kernel: dispatch on exact type.
            # Plain HalfwayBounceBack → HWBBKernelD2Q9
            # InterpolatedBounceBack  → IBBKernelD2Q9
            # MovingWallBounceBack / others → Python path.
            if self.obstacle_bc is not None:
                from src.boundary.wall import HalfwayBounceBack
                if type(self.obstacle_bc) is HalfwayBounceBack:
                    from src.kernels.bounce_back_d2q9 import HWBBKernelD2Q9
                    self._hwbb_kernel = HWBBKernelD2Q9()
                else:
                    from src.boundary.interpolated_wall import InterpolatedBounceBack
                    if isinstance(self.obstacle_bc, InterpolatedBounceBack):
                        from src.kernels.interpolated_wall_d2q9 import IBBKernelD2Q9
                        self._ibb_kernel = IBBKernelD2Q9()

    @property
    def is_ready(self) -> bool:
        """Whether set_distribution() has been called."""
        return self._is_ready

    @property
    def f_post(self) -> Optional['npt.NDArray']:
        """Post-collision distribution (read-only).

        Needed by MEM (Momentum Exchange Method) force calculation.
        Valid after the most recent advance() call.

        In streaming-fused mode (ping-pong), f itself is the post-collision
        state after swap, so f_post returns f. In Esoteric mode there is no
        separate post-collision buffer (single-buffer in-place) -> None.

        Returns:
            Post-collision f, shape (Q, Nx, Ny[, Nz])  [dimensionless]
        """
        if self._use_esoteric:
            return None
        return self._f_post

    def eso_body_force(self) -> 'npt.NDArray':
        """Whole-domain MEM force on the esoteric path (HWBB bodies).

        Same diagnostic-tier kernel the MPI runner uses (eso_mem_force),
        with the clip set to the full domain. Returns (Fx, Fy, Fz) as
        host float64 in lattice units of this level."""
        if not self._use_esoteric:
            raise RuntimeError("eso_body_force() requires the esoteric path")
        if self._eso_ibb is not None:
            # IBB: the HWBB-formula whole-domain kernel would read the
            # REWRITTEN deposits (2*val != f* + val). The scatter pass
            # already accumulated the exact Bouzidi MEM force (f64).
            return self._eso_ibb.last_force()
        from src.kernels.esoteric_d3q27 import eso_mem_force
        cb = [(0, int(d)) for d in self.domain_shape]
        return eso_mem_force(
            self.xp, self.f.reshape((27,) + self.domain_shape),
            self._eso_node_type, self._esoteric_step, cb)

    @property
    def physical_f(self) -> Optional['npt.NDArray']:
        """f in standard ordering + physical layout (checkpoint/IO safe).

        Esoteric mode: gathers the single-buffer memory into a physical
        copy (parity-free on disk -> restart uses the normal fresh-convert
        path in set_distribution). Otherwise returns f itself.
        """
        if self._use_esoteric and self.f is not None:
            if self._eso_wall_mask:
                # Tripwire, NOT the checkpoint path: checkpoints stream
                # the region gather (OutputManager._f_checkpoint_cpu) +
                # the wall_mail_L0 extra key, which round-trips walls
                # bit-exactly. This property's std image is only a LOAD-
                # view bijection — wall-row cross entries carry stale /
                # relocated reflection payload, NOT physical populations
                # — so any consumer doing physics with it (forces, f
                # export, coupling) would be silently wrong at walls.
                # No such consumer exists for single-grid wall cases;
                # keep it loud.
                raise RuntimeError(
                    "physical_f with eso implicit domain walls: the std "
                    "gather image is not physically meaningful at wall "
                    "rows (checkpoints use the region gather + "
                    "wall_mail key instead; patch_notes/eso_wall §4-3)")
            from src.kernels.esoteric_d3q27 import esoteric_gather_std
            return esoteric_gather_std(self.xp, self.f, self._esoteric_step)
        return self.f

    # =====================================================================
    # Diagnostic: NaN/Inf trap
    # =====================================================================

    def _check_nan(self, where: str) -> None:
        """Detect non-finite values in rho/u and abort with location info.

        DEFAULT OFF. The check does a per-step GPU→CPU sync (bool(xp.any(...)))
        that serialises the async kernel pipeline and drops GPU utilisation, yet it
        adds value ONLY when a run actually diverges. So keep it off for healthy
        production runs and enable it ONLY to locate a blow-up on re-run:

            numerics: {nan_trap: true, nan_check_every: 1}    # 1 = every step (precise)

        `nan_check_every` > 1 checks every N steps — cheaper, catches the blow-up
        within N steps (slightly less precise on the first-bad location). Wired in
        src/solver/setup.py from the `numerics` config onto the Simulation class.
        """
        if not getattr(self, 'nan_trap_enabled', False):
            return
        every = int(getattr(self, 'nan_check_every', 1))
        if every > 1 and (self.step_count % every) != 0:
            return
        xp = self.xp
        if self.rho is None or self.u is None:
            return

        rho_bad = bool(xp.any(~xp.isfinite(self.rho)))
        u_bad = bool(xp.any(~xp.isfinite(self.u)))
        if not (rho_bad or u_bad):
            return

        if rho_bad:
            bad_mask = ~xp.isfinite(self.rho)
            field = 'rho'
        else:
            bad_mask = xp.any(~xp.isfinite(self.u), axis=0)
            field = 'u'

        n_bad = int(bad_mask.sum())
        bad_idx = xp.argwhere(bad_mask)
        loc = tuple(int(v) for v in bad_idx[0])

        rho_finite = xp.where(xp.isfinite(self.rho), self.rho, xp.float64(1.0))
        u_sq = xp.where(xp.isfinite(self.u), self.u, xp.float64(0.0)) ** 2
        u_mag_sq = xp.sum(u_sq, axis=0)
        rho_min = float(xp.min(rho_finite))
        rho_max = float(xp.max(rho_finite))
        u_max = float(xp.sqrt(xp.max(u_mag_sq)))

        raise RuntimeError(
            f"[NaN trap @ {where}] step={self.step_count}, "
            f"domain={self.domain_shape}, bad_field='{field}', "
            f"first_bad_loc(x,y,z)={loc}, n_bad={n_bad}, "
            f"rho_finite_range=[{rho_min:.4g}, {rho_max:.4g}], "
            f"|u|_max_finite={u_max:.4g}"
        )

    def advance(self) -> None:
        """Perform one LBM timestep.

        Physical Process (matches continuous equations):
            1. Macroscopic:     ρ, u = moments(f)
            2. Body Force:      F(x) = ALM pipeline (if active)
            3. Velocity Corr:   u += F/(2ρ)         (Guo, 2002)
            4. Collision:       f* = collide(f, ρ, u, τ, F)
                                (equilibrium + forcing internally)
            5. Streaming:       f = f*(x - c_i)     (pull, writes to f directly)
            6. Boundary Cond:   domain faces + obstacles

        After calling advance():
            - self.rho, self.u, self.body_force are updated
            - self.f contains the new distribution
            - self.f_post contains post-collision distribution
            - self.step_count is incremented

        Raises:
            RuntimeError: If set_distribution() has not been called.
        """
        if not self._is_ready:
            raise RuntimeError(
                "Simulation not ready: call set_distribution() first"
            )

        if self._use_surfel:
            # surfel replaces streaming (volumetric advect) — it has its
            # own path and composes with nothing else in S8a
            # (patch_notes/surfel/46 sec. 1; guards raised in setup).
            if self._use_esoteric:
                raise RuntimeError(
                    "wall_bc='surfel' does not support esoteric streaming "
                    "(re-evaluated after S8c — patch_notes/surfel/46)")
            if self._surfel_use_kernel():
                self._advance_surfel_kernel()
            else:
                self._advance_surfel()
        elif self._use_esoteric and self.al_model is None:
            self._advance_esoteric()
        elif self._use_esoteric:
            self._advance_esoteric_with_alm()
        elif self._use_fused and self.al_model is None:
            self._advance_fused()
        elif self._use_fused and self.al_model is not None:
            self._advance_fused_with_alm()
        else:
            self._advance_default()

    # =====================================================================
    # Esoteric Pull path (single-buffer in-place; resurrected from 8036317)
    # Lehmann 2022. Phase 1a = BGK only. Full BC/cumulant/MLG in later phases.
    # =====================================================================

    def _init_esoteric(self, f: 'npt.NDArray') -> None:
        """Initialize Esoteric Pull mode (BGK D3Q27).

        f (standard ordering) -> Esoteric direction ordering -> Esoteric
        memory layout (t=0 even). Builds node_type + BC arrays, allocates
        macroscopic buffers, creates the kernel. No _f_post (single buffer).
        """
        from src.kernels.esoteric_d3q27 import (
            EsotericBGKKernelD3Q27,
            EsotericMacroKernelD3Q27,
            esoteric_scatter_std,
            eso_seed_solid_bounce_ic,
            NODE_SOLID, NODE_EQ_BC, NODE_NEUMANN, NODE_SPONGE,
            _STD_TO_ESO,
        )
        xp = self.xp
        Nx, Ny, Nz = self.domain_shape
        N = Nx * Ny * Nz

        if f.dtype != xp.float32:
            raise ValueError(
                f"esoteric kernels are float32-only (got {f.dtype}); "
                "use precision: float32 or disable LBM_ESOTERIC")

        if not self._esoteric_f_already_set:
            # Fused conversion: see esoteric_scatter_std. The two-call form
            # (convert -> init) held the caller's f, the reordered copy AND
            # the result simultaneously — three full-size arrays of the level
            # being built, 11.5 GB on octo8's 35.5 M-cell L1.
            self.f = esoteric_scatter_std(xp, f, 0)
        else:
            self.f = f
        self._f_post = None  # single-buffer in-place

        node_type = xp.zeros((Nx, Ny, Nz), dtype=xp.int8)   # all FLUID
        bc_rho = xp.ones((Nx, Ny, Nz), dtype=xp.float32)
        bc_ux = xp.zeros((Nx, Ny, Nz), dtype=xp.float32)
        bc_uy = xp.zeros((Nx, Ny, Nz), dtype=xp.float32)
        bc_uz = xp.zeros((Nx, Ny, Nz), dtype=xp.float32)

        if self.obstacle_bc is not None and hasattr(self.obstacle_bc, 'solid_mask'):
            node_type[self.obstacle_bc.solid_mask] = NODE_SOLID

        # Domain faces + sponge (periodic box => no-op). Full fidelity: Phase 1c.
        self._build_esoteric_domain_bc(
            node_type, bc_rho, bc_ux, bc_uy, bc_uz,
            NODE_SOLID, NODE_EQ_BC, NODE_NEUMANN, NODE_SPONGE)

        if self._eso_wall_mask:
            if self._esoteric_f_already_set:
                # MLG/dist restore hands over RAW esoteric memory — that
                # path has no mailbox hand-off wiring (walls are single-
                # grid-only until PLAN §4-5·6), so this must stay
                # unreachable rather than run silently mailbox-less.
                raise RuntimeError(
                    "eso domain wall + raw-memory restore path: mailbox "
                    "hand-off is only wired for the single-grid "
                    "checkpoint restart (patch_notes/eso_wall §4-3)")
            # (9, face_area) f32 segment per wall face, fixed face order
            # xmin..zmax — eso_wall_mail_layout mirrors the in-kernel
            # offset arithmetic (single source for seeding/checkpoint).
            from src.kernels.esoteric_d3q27 import eso_wall_mail_layout
            _, total = eso_wall_mail_layout(self._eso_wall_mask,
                                            (Nx, Ny, Nz))
            self._eso_wall_mail = xp.zeros(total, dtype=xp.float32)
            faces = '+'.join(
                n for b, n in enumerate(
                    ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'))
                if self._eso_wall_mask >> b & 1)
            print(f"  [esoteric] implicit domain wall {faces}: "
                  f"face mailbox {total * 4 / 1e6:.1f} MB")
            # q-face pass (open-face track): a flush FINE wall needs
            # Bouzidi q = 0.5*2^k to land on the global ground plane.
            q = float(getattr(self, '_eso_wall_q_scale', 0.5))
            if abs(q - 0.5) > 1e-12:
                if not (0.5 < q <= 1.0):
                    raise NotImplementedError(
                        f"eso flush wall q={q}: only 0.5 < q <= 1.0 "
                        "(one-hop Bouzidi, flush L1) is wired — a "
                        "flush L2+ wall needs the 2-hop family "
                        "(patch_notes/eso_wall/09, C option)")
                self._eso_wall_face_q = {
                    b: q for b in range(6)
                    if self._eso_wall_mask >> b & 1}
                print(f"  [esoteric] q-face wall pass: q={q} on "
                      f"{faces} (reflection plane = global ground)")

        if not self._esoteric_f_already_set and not self._eso_from_checkpoint:
            # fresh IC only. Checkpoint restore skips BOTH seeders:
            # scattering the checkpointed std f at parity 0 already
            # reproduces every bounce/wall read bit-exactly (LOAD-view
            # fidelity of the gather/scatter bijection; the in-flight
            # wall reflection rides the wrapped row's std entries, the
            # mailbox is restored separately). Seeding on restore would
            # overwrite true deposits with IC-rule values — the exact
            # corruption the solid seeder's docstring forbids (the
            # single-grid restart path used to do it; eso_wall §4-3).
            eso_seed_solid_bounce_ic(
                xp, self.f.reshape((27,) + self.domain_shape), node_type,
                lambda q: f[q], 0)
            if self._eso_wall_mask:
                from src.kernels.esoteric_d3q27 import eso_seed_wall_ic
                eso_seed_wall_ic(
                    xp, self.f.reshape((27,) + self.domain_shape),
                    self._eso_wall_mail, self._eso_wall_mask,
                    lambda q: f[q], 0)

        self._eso_node_type = node_type.ravel()
        self._eso_bc_rho = bc_rho.ravel()
        self._eso_bc_ux = bc_ux.ravel()
        self._eso_bc_uy = bc_uy.ravel()
        self._eso_bc_uz = bc_uz.ravel()
        self._esoteric_step = 0

        self.rho = xp.empty(self.domain_shape, dtype=xp.float32)
        self.u = xp.empty((3,) + self.domain_shape, dtype=xp.float32)

        # Obstacle wall BC on the esoteric path:
        #   HWBB — implicit (SOLID skip + parity-swapped LOAD); optional
        #          in-kernel MEM force via needs_bounce (BGK diagnostics).
        #   IBB  — two-phase deposit-rewrite pass after every fused launch
        #          (S5). Its scatter fuses the MEM force, so the in-kernel
        #          HWBB-formula force MUST stay off for IBB.
        from src.boundary.interpolated_wall import InterpolatedBounceBack
        self._eso_needs_bounce = None
        self._eso_force_out = None
        self._eso_ibb = None
        if isinstance(self.obstacle_bc, InterpolatedBounceBack):
            from src.kernels.esoteric_ibb_d3q27 import EsotericIBBD3Q27
            self._eso_ibb = EsotericIBBD3Q27.from_interpolated_bc(
                xp, self.obstacle_bc, self.domain_shape)
            print(f"  [esoteric] IBB deposit-rewrite pass: "
                  f"{self._eso_ibb.n_links:,} links")
        elif self.obstacle_bc is not None and \
                hasattr(self.obstacle_bc, 'needs_bounce'):
            nb_std = self.obstacle_bc.needs_bounce
            nb_eso = xp.empty_like(nb_std)
            for eso_q in range(27):
                nb_eso[eso_q] = nb_std[_STD_TO_ESO[eso_q]]
            self._eso_needs_bounce = nb_eso.reshape(27, N)
            self._eso_force_out = xp.zeros(3, dtype=xp.float32)

        # Collision-model-specific kernel: cumulant (HVAB) or BGK.
        from src.collision.cumulant import CumulantCollision
        sgs_model = self._sgs_cfg["model"]
        if isinstance(self.collision, CumulantCollision):
            from src.kernels.esoteric_cumulant_d3q27 import (
                EsotericCumulantKernelD3Q27)
            # dyn_smag reuses the WALE kernel branch (same as standard path).
            sgs_kernel = "wale" if sgs_model == "dyn_smag" else sgs_model
            if sgs_kernel not in ("off", "smagorinsky", "wale"):
                raise ValueError(
                    f"esoteric cumulant: unsupported SGS '{sgs_model}'")
            self._esoteric_kernel = EsotericCumulantKernelD3Q27(
                sgs_model=sgs_kernel)
            self._esoteric_is_cumulant = True
            self._eso_omega_bulk = float(self.collision.omega_bulk)
            self._eso_omega_high = float(getattr(self.collision, 'omega_high',
                                                 self.collision.omega_3))
            self._eso_omega_345 = (float(self.collision.omega_3),
                                   float(self.collision.omega_4),
                                   float(self.collision.omega_5))
            self._eso_lambda = float(getattr(self.collision,
                                             'cumulant_limiter', 0.0))
            # SGS buffers (mirrors the standard fused path, f32/esoteric).
            if sgs_model in ("smagorinsky", "wale", "dyn_smag"):
                self.nu_t = xp.zeros(self.domain_shape, dtype=xp.float32)
            if sgs_model in ("wale", "dyn_smag"):
                self._rho_buf = xp.empty(N, dtype=xp.float32)
                self._u_buf = xp.empty((3, N), dtype=xp.float32)
                self._nu_t_in = xp.empty(N, dtype=xp.float32)
                self._eso_macro_kernel = EsotericMacroKernelD3Q27()
                if sgs_model == "wale":
                    from src.kernels.wale_d3q27 import WALEKernelD3Q27
                    self._wale_kernel = WALEKernelD3Q27()
                else:
                    from src.kernels.dyn_smag_d3q27 import DynSmagKernelD3Q27
                    self._wale_kernel = DynSmagKernelD3Q27()
        else:
            if sgs_model not in ("off", "none"):
                raise ValueError(
                    "esoteric BGK + SGS: supported via the MPI runner "
                    "(main_mpi/LocalLevel, dyn_smag only) — the single-GPU "
                    "esoteric path has no BGK pre-pass wiring")
            self._esoteric_kernel = EsotericBGKKernelD3Q27()
            self._esoteric_is_cumulant = False
        self._use_esoteric = True

    def init_esoteric_metadata_host(self) -> None:
        """Distributed-init (patch 17 backlog #4): esoteric BC/node metadata
        ONLY, full-size on HOST numpy (~17 B/cell of RAM), zero device field
        allocation. The distributed runner wrap-slices its slab from these
        host arrays and builds slab f from the uniform-IC equilibrium
        (spatially constant), so no full-size f/rho/u ever exists anywhere —
        this is what lifts the 4-GPU capacity from "one GPU's worth" to
        ~4x (replicated build allocated every full field per rank).
        """
        # wall_bc='ibb' under dist-init: supported since STL track P1 —
        # setup builds the sparse IBB links regardless of how f is
        # initialized (link arrays are tiny), and DistributedMLGRunner
        # slab-filters them via build_slab_ibb (S6) exactly as on the
        # replicated path. The kernel's implicit HWBB at NODE_SOLID is
        # rewritten to Bouzidi by the per-step deposit-rewrite pass.
        import numpy as _np
        from src.kernels.esoteric_d3q27 import (
            NODE_SOLID, NODE_EQ_BC, NODE_NEUMANN, NODE_SPONGE)
        Nx, Ny, Nz = self.domain_shape
        node_type = _np.zeros((Nx, Ny, Nz), dtype=_np.int8)
        if self.obstacle_bc is not None and \
                hasattr(self.obstacle_bc, 'solid_mask'):
            # geometry masks are cell-local functions -> the GPU mask (1B/
            # cell, cheap even at 400M) marks SOLID on the host metadata;
            # needs_bounce is NOT materialized (the MEM-force kernel derives
            # bounce links from node_type on the fly)
            sm = self.obstacle_bc.solid_mask
            sm = sm.get() if hasattr(sm, 'get') else sm
            node_type[sm.reshape(Nx, Ny, Nz)] = 1  # NODE_SOLID
        bc_rho = _np.ones((Nx, Ny, Nz), dtype=_np.float32)
        bc_ux = _np.zeros((Nx, Ny, Nz), dtype=_np.float32)
        bc_uy = _np.zeros((Nx, Ny, Nz), dtype=_np.float32)
        bc_uz = _np.zeros((Nx, Ny, Nz), dtype=_np.float32)
        self._build_esoteric_domain_bc(          # pure indexing: xp-agnostic
            node_type, bc_rho, bc_ux, bc_uy, bc_uz,
            NODE_SOLID, NODE_EQ_BC, NODE_NEUMANN, NODE_SPONGE)
        self._eso_node_type = node_type.ravel()
        self._eso_bc_rho = bc_rho.ravel()
        self._eso_bc_ux = bc_ux.ravel()
        self._eso_bc_uy = bc_uy.ravel()
        self._eso_bc_uz = bc_uz.ravel()
        # q-face pass metadata (open-face track): the dist-init path
        # must carry the SAME face-q dict the full init builds, or a
        # flush fine wall silently runs halfway (-0.25 plane) under
        # --dist-init only — the exact silent-divergence class patch 05
        # caught on the replicated path.
        if self._eso_wall_mask:
            q = float(getattr(self, '_eso_wall_q_scale', 0.5))
            if abs(q - 0.5) > 1e-12:
                if not (0.5 < q <= 1.0):
                    raise NotImplementedError(
                        f"eso flush wall q={q}: only 0.5 < q <= 1.0 is "
                        "wired (patch_notes/eso_wall/09, C option)")
                self._eso_wall_face_q = {
                    b: q for b in range(6)
                    if self._eso_wall_mask >> b & 1}
        from src.collision.cumulant import CumulantCollision
        if isinstance(self.collision, CumulantCollision):
            self._eso_omega_bulk = float(self.collision.omega_bulk)
            self._eso_omega_high = float(getattr(self.collision, 'omega_high',
                                                 self.collision.omega_3))
            self._eso_omega_345 = (float(self.collision.omega_3),
                                   float(self.collision.omega_4),
                                   float(self.collision.omega_5))
            self._eso_lambda = float(getattr(self.collision,
                                             'cumulant_limiter', 0.0))
        else:
            self._eso_omega_bulk = 1.0 / self.tau
            self._eso_omega_high = 1.0 / self.tau
            self._eso_omega_345 = (self._eso_omega_high,) * 3
            self._eso_lambda = 0.0
        self._esoteric_step = 0
        self._use_esoteric = True

    def _build_esoteric_domain_bc(self, node_type, bc_rho, bc_ux, bc_uy, bc_uz,
                                  NODE_SOLID, NODE_EQ_BC, NODE_NEUMANN,
                                  NODE_SPONGE):
        """Map domain face BCs + sponge layers into node_type / bc arrays.

        Faithful port of 8036317. Periodic faces are absent from
        bc_manager.face_bcs, so a periodic box stays all-FLUID (Phase 1a gate).
        Eq-face fidelity validated in Phase 1c (HVAB).

        NOTE(2026-07-30): the sponge branch of the original port was dead
        code from day one — it read `sb._sigma`, which SpongeLayerBC never
        had — so every esoteric run before this fix silently ran WITHOUT
        sponges (faces stayed FLUID -> periodic wrap). Fixed to read
        sigma_slab/_L; gate: patch_notes/entry_unification/gates/
        eso_sponge_twin_gate.py (manual 06 §6.9-①).
        """
        Nx, Ny, Nz = self.domain_shape
        _FBIT = {'xmin': 0, 'west': 0, 'xmax': 1, 'east': 1,
                 'ymin': 2, 'south': 2, 'ymax': 3, 'north': 3,
                 'zmin': 4, 'bottom': 4, 'zmax': 5, 'top': 5}
        wall_mask = 0
        face_methods: dict = {}
        for face_bc in getattr(self.bc_manager, 'face_bcs', []):
            loc = getattr(getattr(face_bc, 'location', None), 'value', None)
            cfg = getattr(face_bc, 'config', None)
            if loc is None or cfg is None:
                continue
            if loc in ('xmin', 'west'):     sl = (0, slice(None), slice(None))
            elif loc in ('xmax', 'east'):   sl = (Nx-1, slice(None), slice(None))
            elif loc in ('ymin', 'south'):  sl = (slice(None), 0, slice(None))
            elif loc in ('ymax', 'north'):  sl = (slice(None), Ny-1, slice(None))
            elif loc in ('zmin', 'bottom'): sl = (slice(None), slice(None), 0)
            elif loc in ('zmax', 'top'):    sl = (slice(None), slice(None), Nz-1)
            else:
                continue
            method = str(getattr(cfg, 'method', '') or '')
            face_methods[_FBIT[loc]] = method
            # NEVER overwrite NODE_SOLID: a span-through body has solid
            # cells ON the face planes; flipping them to a face-BC type
            # would open the wall there (leak). bc_rho/u writes below stay
            # unconditional — solid cells never read them.
            face = node_type[sl]
            nonsolid = face != NODE_SOLID
            is_wall_method = (method == 'hwbb' or 'wall' in method
                              or 'bounce' in method)
            if is_wall_method and self._eso_wall_implicit_ok:
                # eso_wall track proper fix: the row stays FLUID and the
                # kernels reflect every wall-crossing link at the halfway
                # plane node0 −0.5 (parity-swap / face mailbox) — std-
                # identical wall position (Couette gate G2). The whole
                # axis is de-periodized; the opposite face is guarded
                # below.
                wall_mask |= 1 << _FBIT[loc]
                print(f"  * eso domain wall '{loc}' ({method}): implicit "
                      "halfway wall at node0 -0.5 (face mailbox, "
                      "patch_notes/eso_wall)")
                continue
            if 'wall' in method or 'bounce' in method:
                face[nonsolid] = NODE_SOLID
                print(f"  * eso domain wall '{loc}' ({method}): SOLID row "
                      "-> wall at node0 +0.5 and one fluid row lost "
                      "(std halfway puts it at -0.5). Proper-fix track: "
                      "patch_notes/eso_wall/PLAN.md")
            elif 'neumann' in method:
                face[nonsolid] = NODE_NEUMANN
            else:
                face[nonsolid] = NODE_EQ_BC
                if method == 'hwbb':
                    # NOT a wall on this path (MLG/MPI surfaces until
                    # PLAN §4 steps 5-6): 'hwbb' lands here as EQ(u=0)
                    # at node0. Measured (eso_wall_couette_gate,
                    # tau 0.8): effective wall z0 = +0.20 vs std halfway
                    # -0.50 — a tau-DEPENDENT slip wall, not a clean
                    # half-cell. Loud until the wall wiring covers this
                    # path too (patch_notes/eso_wall/PLAN.md).
                    print(f"  * WARNING: eso face '{loc}' method 'hwbb' "
                          "DEGRADES to EQ(u=0) at node0 — measured "
                          "effective wall z0=+0.20 (tau 0.8) vs std "
                          "-0.50. Track: patch_notes/eso_wall/PLAN.md")
                dens = getattr(cfg, 'density', None)
                if dens is not None:
                    bc_rho[sl] = float(dens)
                vel = getattr(cfg, 'velocity', None)
                if isinstance(vel, (int, float)):
                    bc_ux[sl] = float(vel)
                elif isinstance(vel, (list, tuple)):
                    if len(vel) > 0: bc_ux[sl] = float(vel[0])
                    if len(vel) > 1: bc_uy[sl] = float(vel[1])
                    if len(vel) > 2: bc_uz[sl] = float(vel[2])

        # eso wall guard: an axis with a wall face is de-periodized as a
        # WHOLE, so its opposite face must be a wall too or a covered BC
        # row that discards its (now meaningless) cross-face load —
        # eq/sponge overwrite after LOAD, so they qualify; neumann
        # passes the loaded value through (stale wrap slot -> silent
        # corruption) and an absent BC leaves a FLUID row consuming the
        # same stale slots as real data. Both are hard errors
        # (patch_notes/eso_wall/PLAN.md §2-2).
        self._eso_wall_mask = wall_mask
        if wall_mask:
            sponge_bits = 0
            for sb in getattr(self.bc_manager, 'sponge_layers', []):
                b = _FBIT.get(
                    getattr(getattr(sb, 'location', None), 'value', ''))
                if b is not None:
                    sponge_bits |= 1 << b
            _FNAME = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
            for blo in (0, 2, 4):
                pair = (blo, blo + 1)
                if not any(wall_mask >> b & 1 for b in pair):
                    continue
                for b in pair:
                    if wall_mask >> b & 1:
                        continue
                    meth = face_methods.get(b)
                    if meth is not None and 'neumann' in meth:
                        raise ValueError(
                            f"eso domain wall: face '{_FNAME[b]}' is "
                            f"neumann opposite a wall on the same axis "
                            "— passthrough would consume stale "
                            "de-periodized slots. Use eq or sponge "
                            "there (patch_notes/eso_wall/PLAN.md)")
                    if (meth is None and not (sponge_bits >> b & 1)
                            and not (getattr(self, '_eso_band_faces', 0)
                                     >> b & 1)):
                        raise ValueError(
                            f"eso domain wall: face '{_FNAME[b]}' has "
                            "no BC opposite a wall on the same axis — "
                            "the axis is de-periodized, so the face "
                            "needs a wall, eq, sponge, or a DECLARED "
                            "coupling-band face (_eso_band_faces, set "
                            "by the MLG builder for flush fine walls; "
                            "its inbound mailbox is prescribed by the "
                            "C2F scatter each substep). "
                            "(patch_notes/eso_wall/08-09)")

        for sb in getattr(self.bc_manager, 'sponge_layers', []):
            loc = getattr(getattr(sb, 'location', None), 'value', '')
            # Effective thickness _L (= min(thickness, N//2)) and the slab
            # damping profile sigma_slab are what SpongeLayerBC actually
            # stores. A configured sponge that cannot be converted must be a
            # hard error, not a silent skip (the pre-fix code read a
            # never-existing `_sigma` attribute and dropped every sponge on
            # the esoteric path — gate: eso_sponge_twin_gate.py).
            L = int(getattr(sb, '_L', 0) or 0)
            sigma_slab = getattr(sb, 'sigma_slab', None)
            if L <= 0 or sigma_slab is None:
                raise ValueError(
                    f"esoteric BC conversion: sponge at '{loc}' has no readable "
                    f"damping profile (_L={L}, sigma_slab="
                    f"{'None' if sigma_slab is None else 'ok'})")
            sigma_np = (sigma_slab.get() if hasattr(sigma_slab, 'get')
                        else sigma_slab).reshape(-1)
            # sigma_slab is indexed by slab position along the axis
            # (min face: index 0 = boundary row; max face: index L-1 =
            # boundary row). The sl loop below indexes layer_i inward FROM
            # the boundary face, so flip on max faces.
            if not bool(getattr(sb.location, 'is_min', False)):
                sigma_np = sigma_np[::-1]
            rho_inf = float(getattr(sb, 'rho_inf', 1.0))
            u_inf = getattr(sb, 'u_inf', [0.0, 0.0, 0.0])
            u_inf = u_inf.get() if hasattr(u_inf, 'get') else u_inf
            for layer_i in range(L):
                if loc in ('xmax', 'east'):     sl = (Nx-1-layer_i, slice(None), slice(None))
                elif loc in ('xmin', 'west'):   sl = (layer_i, slice(None), slice(None))
                elif loc in ('ymax', 'north'):  sl = (slice(None), Ny-1-layer_i, slice(None))
                elif loc in ('ymin', 'south'):  sl = (slice(None), layer_i, slice(None))
                elif loc in ('zmax', 'top'):    sl = (slice(None), slice(None), Nz-1-layer_i)
                elif loc in ('zmin', 'bottom'): sl = (slice(None), slice(None), layer_i)
                else:
                    continue
                # NOTE: bc_uz is reused as sigma -> sponge target w is
                # implicitly 0 (fine for hover; documented in patch 15).
                row = node_type[sl]
                row[row != NODE_SOLID] = NODE_SPONGE   # keep body cells solid
                bc_rho[sl] = rho_inf
                bc_ux[sl] = float(u_inf[0])
                bc_uy[sl] = float(u_inf[1]) if len(u_inf) > 1 else 0.0
                bc_uz[sl] = float(sigma_np[layer_i])  # reuse bc_uz as sigma

    def _eso_sgs_pre_pass(self) -> None:
        """WALE/dyn_smag pre-pass on Esoteric memory: LOAD-macro -> nu_t.

        Mirrors _wale_pre_pass (3D branch), but the macro comes from the
        Esoteric LOAD replica (the pre-collision state of THIS step)."""
        Nx, Ny, Nz = self.domain_shape
        self._eso_macro_kernel.launch(
            self.f, self._eso_node_type, self._rho_buf, self._u_buf,
            Nx, Ny, Nz, self._esoteric_step,
            wall_mask=self._eso_wall_mask, wall_mail=self._eso_wall_mail)
        self._sgs_mask_solid_u()
        if self._sgs_cfg["model"] == "wale":
            self._wale_kernel.launch(
                self._u_buf[0], self._u_buf[1], self._u_buf[2],
                self._nu_t_in, Nx, Ny, Nz,
                Cw=float(self._sgs_cfg["Cw"]), dx=1.0)
        else:  # dyn_smag
            self._wale_kernel.launch(
                self._u_buf[0], self._u_buf[1], self._u_buf[2],
                self._nu_t_in, Nx, Ny, Nz,
                dx=1.0,
                Cs_max=float(self._sgs_cfg["Cs_max"]),
                alpha_sq=float(self._sgs_cfg["alpha_sq"]))
        self._sgs_wall_damp()

    def _advance_esoteric_with_alm(self) -> None:
        """Esoteric ALM 2-pass (mirrors _advance_fused_with_alm):
        Pass 1 = macro from Esoteric memory (pre-collision, uncorrected u)
        -> ALM body force -> Pass 2 = collision kernel with force (Guo
        velocity correction + source term inside the kernel)."""
        if not self._esoteric_is_cumulant:
            raise RuntimeError("Esoteric ALM requires the cumulant kernel")
        xp = self.xp
        Nx, Ny, Nz = self.domain_shape
        if getattr(self, '_eso_macro_kernel', None) is None:
            from src.kernels.esoteric_d3q27 import EsotericMacroKernelD3Q27
            self._eso_macro_kernel = EsotericMacroKernelD3Q27()
        # Pass 1: rho/u views share memory with the 3D-shaped arrays.
        self._eso_macro_kernel.launch(
            self.f, self._eso_node_type, self.rho.ravel(),
            self.u.reshape(3, -1), Nx, Ny, Nz, self._esoteric_step,
            wall_mask=self._eso_wall_mask, wall_mail=self._eso_wall_mail)
        self._check_nan("esoteric+alm:post-macro")
        body_force = self._compute_body_force(self.u, self.rho)
        if body_force is not None and body_force.dtype != xp.float32:
            body_force = body_force.astype(xp.float32)
        self.body_force = body_force
        self._advance_esoteric()   # consumes self.body_force, then clears it

    def _advance_esoteric(self) -> None:
        """Esoteric Pull advance: 1 kernel launch (load+macro+BC+collide+store)."""
        Nx, Ny, Nz = self.domain_shape
        omega = 1.0 / self.tau
        if self._esoteric_is_cumulant:
            if self._sgs_cfg["model"] in ("wale", "dyn_smag"):
                self._eso_sgs_pre_pass()
            self._esoteric_kernel.launch(
                self.f, self.rho, self.u,
                self._eso_node_type,
                self._eso_bc_rho, self._eso_bc_ux,
                self._eso_bc_uy, self._eso_bc_uz,
                omega, self._eso_omega_bulk, self._eso_omega_high,
                Nx, Ny, Nz,
                t_step=self._esoteric_step,
                force=self.body_force,
                Cs=float(self._sgs_cfg["Cs"]),
                nu_t_out=(self.nu_t.ravel() if self.nu_t is not None else None),
                nu_t_in=getattr(self, '_nu_t_in', None),
                omega_3=getattr(self, '_eso_omega_345',
                                (self._eso_omega_high,) * 3)[0],
                omega_4=getattr(self, '_eso_omega_345',
                                (self._eso_omega_high,) * 3)[1],
                omega_5=getattr(self, '_eso_omega_345',
                                (self._eso_omega_high,) * 3)[2],
                lambda_lim=getattr(self, '_eso_lambda', 0.0),
                wall_mask=self._eso_wall_mask,
                wall_mail=self._eso_wall_mail,
            )
        else:
            force_out = None
            if self._eso_force_out is not None:
                self._eso_force_out.fill(0)
                force_out = self._eso_force_out
            self._esoteric_kernel.launch(
                self.f, self.rho, self.u,
                self._eso_node_type,
                self._eso_bc_rho, self._eso_bc_ux,
                self._eso_bc_uy, self._eso_bc_uz,
                omega, Nx, Ny, Nz,
                t_step=self._esoteric_step,
                needs_bounce=self._eso_needs_bounce,
                force_out=force_out,
                wall_mask=self._eso_wall_mask,
                wall_mail=self._eso_wall_mail,
            )
        # q-face wall pass (open-face track): rewrite the flush wall's
        # halfway deposits with Bouzidi(q) — must precede any gather of
        # this memory (same contract as the IBB rewrite below).
        if self._eso_wall_face_q:
            from src.kernels.esoteric_d3q27 import eso_wall_q_rewrite
            eso_wall_q_rewrite(
                self.xp, self.f, self._eso_wall_mail,
                self._eso_wall_mask, self._eso_wall_face_q,
                self._esoteric_step)
        # IBB (S5): overwrite toward-solid deposits of the step just run
        # with Bouzidi values (2-phase pass; parity = this step's index).
        if self._eso_ibb is not None:
            self._eso_ibb.rewrite(self.f, self._esoteric_step)
        self.body_force = None
        self._esoteric_step += 1
        self.step_count += 1

    def _advance_default(self) -> None:
        """Default advance path using CuPy array operations."""

        # ─── Step 1: Macroscopic Variables ───────────────────────────
        self.rho, self.u = self.macroscopic.compute(self.f)
        self._check_nan("default:post-macro")

        # ─── Step 2: Body Force (ALM pipeline) ──────────────────────
        self.body_force = self._compute_body_force(self.u, self.rho)

        # ─── Step 3: Guo Velocity Correction ────────────────────────
        u = self.u
        if self.body_force is not None:
            u = u + self.body_force / (2.0 * self.rho[None, ...])
            self.u = u

        # ─── Step 4: Collision ───────────────────────────────────────
        self.collision.collide(
            self.f, self._f_post, self.rho, u, self.tau, self.body_force
        )

        # ─── Step 5: Streaming ───────────────────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Step 6: Boundary Conditions ─────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
            self.obstacle_bc.apply_with_reset(self.f, self._f_post)

        self.step_count += 1

    def _advance_surfel(self) -> None:
        """Surfel wall-boundary path (S8a): collide → facet apply → advect.

        Mirrors the validated testbed order exactly
        (patch_notes/surfel/channel_wmles_surfel.py step(); parity gate
        s13): the volumetric advect REPLACES streaming, and the facet
        exchange must run on the post-collision field BEFORE transport
        (inject→apply ordering family, patch_notes/surfel/27). Domain BCs
        run after transport, as in the default path.

        S8a limits (guarded in setup): no ALM, no MLG, no tau-model band,
        standard (non-fused, non-esoteric) operators only.
        """
        sb = self.obstacle_bc

        # 1. macroscopics on the raw field. Solid cells hold f = 0 →
        #    rho = 0 → the division writes NaN there; sanitize with
        #    where() (never mask-multiply: NaN * 0 = NaN — measured,
        #    testbed comment).
        self.rho, self.u = self.macroscopic.compute(self.f)

        if self.al_model is not None:
            raise RuntimeError("wall_bc='surfel' + ALM is S8b+ scope "
                               "(patch_notes/surfel/46)")

        rho_s, u_s = sb.sanitize_macro(self.rho, self.u)

        # 2. collision on the benign-solid state, then zero non-live
        #    populations (cut-cell hygiene before transport).
        #    SGS (S8b-2): constant-Cs moment-based Smagorinsky, the exact
        #    testbed configuration every campaign table was measured with
        #    (setup guards any other model). Per-cell tau field — the
        #    production CumulantCollision accepts it (testbed-validated).
        tau_arg = self.tau
        if self._sgs_cfg["enabled"]:
            tau_arg = sb.tau_sgs(self.f, self.rho, self.u, self.tau,
                                 self._sgs_cfg["Cs"])
        self.collision.collide(
            self.f, self._f_post, rho_s, u_s, tau_arg, None
        )
        sb.mask_post(self._f_post)

        self._surfel_post_collide(sb)

    def _surfel_use_kernel(self) -> bool:
        """Eligibility of the fused-collide surfel path (cached).

        Falls back to the host chain — never silently changes physics:
        s10 pins kernel==host, and the adapter's `collide` knob selects
        explicitly (gates pass 'host' to pin the reference chain).
        """
        if self._surfel_ck_ok is None:
            from src.collision.cumulant import CumulantCollision
            sb = self.obstacle_bc
            if getattr(sb, 'collide_path', 'host') != 'kernel':
                self._surfel_ck_ok = False
            else:
                # collide='kernel' is an explicit request: unmet
                # prerequisites used to route to the host chain silently.
                problems = []
                if self.xp.__name__ != 'cupy':
                    problems.append("device is not cupy")
                if not isinstance(self.collision, CumulantCollision):
                    problems.append(
                        f"collision {type(self.collision).__name__} "
                        "is not Cumulant")
                if self.f is None or self.f.dtype != self.xp.float32:
                    problems.append("f is not float32")
                if problems:
                    raise ValueError(
                        "surfel collide='kernel' requested but unavailable: "
                        + "; ".join(problems)
                        + " — pass collide='host' explicitly for the "
                        "reference chain")
                self._surfel_ck_ok = True
        return self._surfel_ck_ok

    def _advance_surfel_kernel(self) -> None:
        """Fused-collide surfel path (patch 55): macro + SGS + collide in
        ONE launch (CumulantCollideKernelD3Q27, gate s10 — the same
        kernel the testbed's --collide kernel path validated).

        The host collide chain was measured at ~87% of the surfel step
        (testbed note); this branch removes it. Differences from the
        host branch, all testbed-mirrored:
          - the kernel computes macro itself (f32 rho/u out — no f64
            promotion, no separate Macroscopic pass, no sanitize: the
            kernel leaves NaN u / garbage f_post on DEAD cells, so the
            post-collision zeroing must be zero_dead (assignment,
            NaN-safe) — NEVER mask_post's multiply (NaN * 0 = NaN);
          - SGS runs in-kernel from Cs (Stiebler — same math as
            sb.tau_sgs; gate s10 pins host==kernel).
        """
        sb = self.obstacle_bc
        xp = self.xp
        if self._surfel_ck is None:
            from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
            self._surfel_ck = CumulantCollideKernelD3Q27(
                sgs_model=('smagorinsky' if self._sgs_cfg["enabled"]
                           else 'off'))
            n = 1
            for d in self.domain_shape:
                n *= d
            self._sck_rho = xp.zeros(n, dtype=xp.float32)
            self._sck_u = xp.zeros((3, n), dtype=xp.float32)

        n = self._sck_rho.size
        q = self.f.shape[0]
        self._surfel_ck.launch(
            self.f.reshape(q, -1), self._f_post.reshape(q, -1),
            self._sck_rho, self._sck_u, None,
            1.0 / self.tau,
            float(self.collision.omega_bulk),
            float(self.collision.omega_3),
            n, Cs=float(self._sgs_cfg["Cs"]))
        self.rho = self._sck_rho.reshape(self.domain_shape)
        self.u = self._sck_u.reshape((3,) + self.domain_shape)
        sb.zero_dead(self._f_post)

        self._surfel_post_collide(sb)

    def _surfel_post_collide(self, sb) -> None:
        """Shared surfel tail: band injection -> facet apply -> advect ->
        domain BCs (order per patch 27)."""

        # 2b. tau-model band injection (S8b): post-collision, BEFORE the
        #     facet apply — the order is load-bearing (patch 27: apply-
        #     first drains the injected flux outside the facet force
        #     accounting; inject-first closes at 1.000). Band cells are
        #     live FULL cells, so the raw u is finite there.
        if getattr(sb, 'tau_model_on', False):
            sb.inject_tau_model(self._f_post, self.u, self.tau)

        # 3. facet exchange + volumetric transport (writes new field
        #    into self.f). RAW rho/u by design — the wall-law sampler
        #    masks live cells itself (testbed order).
        sb.apply_and_advect(self._f_post, self.f, self.rho, self.u)

        # 4. domain boundary conditions on the post-transport field
        self.bc_manager.apply_all(self.f, self._f_post)

        self.step_count += 1

    def _advance_fused(self) -> None:
        """Fused CUDA kernel path (no ALM): macro + collision in 1 launch."""
        xp = self.xp
        N = 1
        for d in self.domain_shape:
            N *= d

        # Allocate rho/u if needed (first call)
        if self.rho is None:
            self.rho = xp.empty(self.domain_shape, dtype=self.f.dtype)
            self.u = xp.empty((len(self.domain_shape),) + self.domain_shape,
                              dtype=self.f.dtype)

        # ─── WALE / dyn_smag pre-pass: f -> rho, u (flat) -> nu_t_in ──
        if self._sgs_cfg["model"] in ("wale", "dyn_smag"):
            self._wale_pre_pass(N)

        # ─── Fused: macro + collision (1 kernel launch) ─────────────
        omega = 1.0 / self.tau
        if self._fused_is_cumulant:
            if self._fused_is_d2q9:
                ob = self.collision.omega_bulk
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=None,
                    omega_1=omega,
                    omega_bulk=omega if ob is None else ob,
                    omega_3=self.collision.omega_3,
                    omega_4=self.collision.omega_4,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                    nu_t_in=self._nu_t_in,
                )
            else:
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=None,
                    omega_1=omega,
                    omega_bulk=self.collision.omega_bulk,
                    omega_high=self.collision.omega_3,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                    nu_t_in=self._nu_t_in,
                    nut_scale=None,
                )
        else:
            self._fused_kernel.launch(
                f_in=self.f,
                f_post=self._f_post,
                rho_out=self.rho,
                u_out=self.u,
                force=None,
                omega=omega,
                N=N,
            )
        self.body_force = None
        self._check_nan("fused:post-collision")

        # ─── Streaming ───────────────────────────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Boundary Conditions ─────────────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
            self._apply_obstacle_bc()

        self.step_count += 1

    def _wale_pre_pass(self, N: int) -> None:
        """Compute (rho, u) from f and nu_t from u via the SGS pre-pass —
        feeds the cumulant collide WALE branch. Same buffer pipeline
        works for both WALE and Dynamic Smagorinsky; only the second
        kernel launch differs."""
        model = self._sgs_cfg["model"]
        if len(self.domain_shape) == 2:
            Nx, Ny = self.domain_shape
            self._macro_kernel.launch(
                self.f, self._rho_buf,
                self._u_buf[0], self._u_buf[1], N,
            )
            self._sgs_mask_solid_u()
            if model == "wale":
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._nu_t_in,
                    Nx, Ny, Cw=float(self._sgs_cfg["Cw"]), dx=1.0,
                )
            else:  # dyn_smag
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._nu_t_in,
                    Nx, Ny,
                    dx=1.0,
                    Cs_max=float(self._sgs_cfg["Cs_max"]),
                    alpha_sq=float(self._sgs_cfg["alpha_sq"]),
                )
        else:
            Nx, Ny, Nz = self.domain_shape
            self._macro_kernel.launch(
                self.f, self._rho_buf,
                self._u_buf[0], self._u_buf[1], self._u_buf[2], N,
            )
            self._sgs_mask_solid_u()
            if model == "wale":
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._u_buf[2],
                    self._nu_t_in,
                    Nx, Ny, Nz, Cw=float(self._sgs_cfg["Cw"]), dx=1.0,
                )
            else:  # dyn_smag
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._u_buf[2],
                    self._nu_t_in,
                    Nx, Ny, Nz,
                    dx=1.0,
                    Cs_max=float(self._sgs_cfg["Cs_max"]),
                    alpha_sq=float(self._sgs_cfg["alpha_sq"]),
                )
        self._sgs_wall_damp()

    def _sgs_mask_solid_u(self) -> None:
        """No-slip the SGS input velocity at SOLID cells (patch 12, F1e).

        The WALE/dyn_smag gradient (and dyn_smag test-filter) stencils read
        u at every neighbour with NO solid awareness. Without this mask the
        two paths feed them different garbage — std: moments of the
        populations streamed into the wall this step; esoteric: moments of
        deposit + stale fossil slots — giving an O(1) systematic wall-layer
        nu_t difference from step 1 (f1e gate; the eso Cd-bias driver).
        u = 0 is the physical wall value; both paths now see identical,
        correct stencil input. rho at solid is not read by these kernels.
        """
        if self.obstacle_bc is None or \
                not hasattr(self.obstacle_bc, 'solid_mask'):
            return
        idx = getattr(self, '_sgs_solid_idx', None)
        if idx is None:
            idx = self.xp.where(self.obstacle_bc.solid_mask.ravel())[0]
            self._sgs_solid_idx = idx
        if idx.size:
            self._u_buf[:, idx] = 0.0

    def _sgs_wall_damp(self) -> None:
        """Zero nu_t_in within sgs.wall_damp_cells of a SOLID body.

        See parse_sgs_config: the staircase velocity jump at the wall is a
        discretisation artifact the SGS operators mistake for resolved
        strain (f1g probe: O(100-1000) x nu_mol in the wall shell at
        alpha=10). Hard van-Driest-style damping over the stencil reach.
        No-op when wall_damp_cells == 0 (default) or no body.
        """
        n = int(self._sgs_cfg.get("wall_damp_cells", 0))
        if n <= 0 or self.obstacle_bc is None or \
                not hasattr(self.obstacle_bc, 'solid_mask'):
            return
        idx = getattr(self, '_sgs_damp_idx', None)
        if idx is None:
            from cupyx.scipy import ndimage
            shell = ndimage.binary_dilation(
                self.obstacle_bc.solid_mask, iterations=n, brute_force=True)
            idx = self.xp.where(shell.ravel())[0]
            self._sgs_damp_idx = idx
        if idx.size:
            self._nu_t_in[idx] = 0.0

    def _apply_obstacle_bc(self) -> None:
        """Dispatch obstacle BC application across HWBB / IBB / Python paths."""
        N = 1
        for d in self.domain_shape:
            N *= d

        if self._hwbb_kernel is not None:
            self._hwbb_kernel.apply(
                self.f, self._f_post,
                self.obstacle_bc.needs_bounce, N,
            )
            self._hwbb_kernel.reset_solid(
                self.f, self.obstacle_bc.solid_mask, N,
            )
        elif self._ibb_kernel is not None:
            if len(self.domain_shape) == 2:
                Nx, Ny = self.domain_shape
                self._ibb_kernel.apply(
                    self.f, self._f_post,
                    self.obstacle_bc.needs_bounce,
                    self.obstacle_bc.q_fraction,
                    Nx, Ny,
                )
            else:
                Nx, Ny, Nz = self.domain_shape
                self._ibb_kernel.apply(
                    self.f, self._f_post,
                    self.obstacle_bc.link_cell,
                    self.obstacle_bc.link_dir,
                    self.obstacle_bc.link_q,
                    self.obstacle_bc.n_links,
                    Nx, Ny, Nz,
                )
            self._ibb_kernel.reset_solid(
                self.f, self.obstacle_bc.solid_mask, N,
            )
        else:
            self.obstacle_bc.apply_with_reset(self.f, self._f_post)

    def _advance_fused_with_alm(self) -> None:
        """Fused path with ALM: macro → ALM → fused collision."""
        xp = self.xp
        N = 1
        for d in self.domain_shape:
            N *= d

        # ─── Step 1: Macroscopic (needed by ALM before collision) ───
        self.rho, self.u = self.macroscopic.compute(self.f)

        # Ensure rho/u dtype matches f for fused CUDA kernel
        # (macroscopic.compute uses tensordot which promotes to float64
        #  via int32 c × float32 f, but the kernel writes float32)
        dtype = self.f.dtype
        if self.rho.dtype != dtype:
            self.rho = self.rho.astype(dtype)
        if self.u.dtype != dtype:
            self.u = self.u.astype(dtype)

        self._check_nan("fused+alm:post-macro")

        # ─── Step 2: Body Force (ALM pipeline) ──────────────────────
        self.body_force = self._compute_body_force(self.u, self.rho)

        # Ensure dtype matches f (ALM returns float64, kernel expects float32)
        if self.body_force is not None and self.body_force.dtype != self.f.dtype:
            self.body_force = self.body_force.astype(self.f.dtype)

        # ─── Step 3+4: Guo correction + collision (fused kernel) ────
        omega = 1.0 / self.tau
        if self._fused_is_cumulant:
            if self._fused_is_d2q9:
                ob = self.collision.omega_bulk
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=self.body_force,
                    omega_1=omega,
                    omega_bulk=omega if ob is None else ob,
                    omega_3=self.collision.omega_3,
                    omega_4=self.collision.omega_4,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                )
            else:
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=self.body_force,
                    omega_1=omega,
                    omega_bulk=self.collision.omega_bulk,
                    omega_high=self.collision.omega_3,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                    nut_scale=None,
                )
        else:
            self._fused_kernel.launch(
                f_in=self.f,
                f_post=self._f_post,
                rho_out=self.rho,
                u_out=self.u,
                force=self.body_force,
                omega=omega,
                N=N,
            )

        # ─── Step 5: Streaming ───────────────────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Step 6: Boundary Conditions ─────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
            self._apply_obstacle_bc()

        self.step_count += 1

    # =====================================================================
    # Internal
    # =====================================================================

    def _compute_body_force(
        self,
        u: 'npt.NDArray',
        rho: 'npt.NDArray',
    ) -> Optional['npt.NDArray']:
        """Compute body force from Actuator Line Model.

        The ALM pipeline:
            1. Advance rotor (rotation by ωΔt)
            2. Interpolate LBM velocity at blade markers (Gaussian kernel)
            3. Compute BEM forces at each blade element (CL, CD lookup)
            4. Spread forces back to LBM grid (Gaussian kernel)

        Args:
            u: Velocity field (raw, before Guo correction)
               shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
            rho: Density field
               shape (Nx, Ny[, Nz])  [dimensionless]

        Returns:
            Body force field, shape (dim, Nx, Ny[, Nz])  [lattice force / lu³]
            or None if no ALM is active.
        """
        if self.al_model is None:
            return None

        # al_model.step() handles:
        #   rotor.advance(dt) → interpolation → BEM → spreading
        return self.al_model.step(u, dt=1.0)  # [lattice force / lu³]

    # =====================================================================
    # Info / Debug
    # =====================================================================

    def print_info(self) -> None:
        """Print simulation configuration summary."""
        dim = len(self.domain_shape)
        total_cells = 1
        for n in self.domain_shape:
            total_cells *= n

        nu = (1.0 / 3.0) * (self.tau - 0.5)  # [Δx²/Δt]

        print(f"\n{'='*60}")
        print(f"  Simulation Info")
        print(f"{'='*60}")
        print(f"  Domain: {'×'.join(str(n) for n in self.domain_shape)} "
              f"({dim}D, {total_cells:,} cells)")
        print(f"  τ = {self.tau:.6f}, ν = {nu:.6f} [Δx²/Δt]")
        print(f"  Collision: {self.collision.__class__.__name__}")
        print(f"  Streaming: {self.streaming.__class__.__name__}")
        print(f"  ALM: {'active' if self.al_model is not None else 'inactive'}")
        print(f"  Obstacle BC: "
              f"{'active' if self.obstacle_bc is not None else 'none'}")
        print(f"  Ready: {self.is_ready}")
        print(f"{'='*60}")

    def __repr__(self) -> str:
        dim = len(self.domain_shape)
        shape_str = '×'.join(str(n) for n in self.domain_shape)
        return (f"Simulation({dim}D, {shape_str}, "
                f"τ={self.tau:.4f}, "
                f"step={self.step_count}, "
                f"ready={self.is_ready})")