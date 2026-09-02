"""
Sponge Layer Boundary Condition

Volume-based damping that drives distributions toward equilibrium
near domain boundaries. NOT a face BC — operates on a buffer zone
of configurable thickness.

Two-stage application (encapsulated in apply()):
    Stage A — Neumann cleanup at boundary face:
        Streaming with periodic-wrap leaves "wrap-around garbage" in the
        incoming populations of the boundary row. Without cleanup, only a
        σ_max fraction is replaced each step, so (1 − σ_max) of the garbage
        survives → can grow unboundedly. SpongeLayerBC owns its own
        NeumannBC so the partner is guaranteed regardless of σ_max.

    Stage B — Volume damping over the buffer slab:
        f_i(x) ← f_i(x) + σ(x) · [f_eq(ρ∞, U∞) − f_i(x)]
        Equivalently:
        f_i(x) ← (1 − σ(x)) · f_i(x) + σ(x) · f_eq_i(ρ∞, U∞)

        with quadratic ramp:
            σ(x) = σ_max · (d/L)²
            d = distance from inner edge of sponge layer  [lattice units]
            L = sponge thickness                          [lattice units]
            σ_max = maximum damping coefficient           [0 < σ ≤ 1]

Performance:
    Only the slab of nodes within the buffer zone is accessed for the
    volume damping; Stage A only touches the boundary face row. Memory
    and compute scale with  Q × L × (transverse area), not  Q × N_total.

Usage in config:
    "farfield_sponge": {
        "location": "xmax",
        "method": "sponge",
        "velocity": 0.1,        # U∞  [Δx/Δt]
        "rho": 1.0,             # ρ∞  [dimensionless]
        "thickness": 20,        # L   [lattice units]
        "strength": 0.5         # σ_max [dimensionless]
    }

Applied in Phase 3 of the time loop (after face BCs and corner BCs).
SpongeLayerBC is now self-contained: callers no longer need to register
a separate NeumannBC for the sponge face.

References:
    - Israeli & Orszag, J. Comp. Phys. 41, 1981
    - Vergnault et al., Comp. & Fluids 68, 2012

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .bc_config import FaceConfig, FaceLocation, BCType
from .regularized_utils import compute_f_eq


class SpongeLayerBC:
    """Volume-based sponge layer for non-reflecting boundaries.

    Damps distributions toward equilibrium in a buffer zone near
    the domain boundary. The damping increases quadratically from
    the inner edge (σ = 0) to the boundary (σ = σ_max).

    Self-contained: the apply() method first runs a Neumann cleanup on
    the boundary face row (via an internally-held NeumannBC) to remove
    streaming wrap-around garbage, then performs the volume damping.
    Callers do NOT need to register a separate NeumannBC.

    **Slab-only optimisation**: only the L nodes within the buffer
    zone are touched by Stage B (volume damping); Stage A (Neumann
    cleanup) only touches the boundary row.

    Attributes:
        location: Which face the sponge is attached to
        thickness: Buffer zone depth  [lattice units]
        sigma_max: Maximum damping  [dimensionless]
        u_inf: Freestream velocity  [Δx/Δt]
        rho_inf: Freestream density  [dimensionless]
    """

    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig,
                 domain_shape: Tuple[int, ...],
                 node_map: 'object') -> None:
        """Initialize sponge layer.

        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            config: FaceConfig with extra['thickness'] and extra['sigma_max']
            domain_shape: (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
            node_map: NodeMap — provides flat-slice geometry for the
                internal NeumannBC base layer.
        """
        self.xp = xp
        self.lattice = lattice
        self.dim = lattice.dim
        self.Q = lattice.Q
        self.c = xp.asarray(lattice.c)
        self.w = xp.asarray(lattice.w)
        self.cs2 = lattice.cs2

        self.location = config.location
        self.domain_shape = domain_shape

        # Physical parameters
        self.rho_inf = config.density                                # [dimensionless]
        self.thickness = int(config.extra.get('thickness', 20))      # [lattice units]
        self.sigma_max = float(config.extra.get('sigma_max', 0.5))   # [dimensionless]

        # MPI slab cut of the node map (F1 repair, mpi_axis/02): when the
        # sponge axis IS the cut axis, the zone is the window's overlap
        # with the GLOBAL sponge zone and Stage A runs only on the rank
        # owning the global boundary row. None = full-domain (default).
        self._cut = getattr(node_map, 'cut', None)

        # Setup velocity and precompute damping
        self._setup_velocity(config.velocity)
        self._setup_damping()

        # Compose with NeumannBC for Stage A (boundary-face wrap-around cleanup).
        # Composition over duplication: re-uses NeumannBC's incoming-direction
        # logic and slice handling, and guarantees this base partner is always
        # active regardless of σ_max value.
        # Under an MPI cut along this axis, only the rank owning the global
        # boundary row builds the partner — other ranks have no face row to
        # clean (their zone rows never receive wrap-around garbage).
        if node_map.face_active(self.location):
            from .face_bc import NeumannBC
            neumann_config = FaceConfig(
                location=config.location,
                bc_type=BCType.NEUMANN,
                method='neumann',
            )
            self._neumann_base = NeumannBC(xp, lattice, neumann_config, node_map)
        else:
            self._neumann_base = None

    @property
    def active(self) -> bool:
        """False only for a cut-axis sponge with no zone rows and no
        boundary row in this rank's window (the manager then skips it)."""
        return bool(self._zones) or self._neumann_base is not None
    
    def _setup_velocity(self, velocity: Union[float, list]) -> None:
        """Build freestream velocity vector.
        
        Args:
            velocity: Scalar or list of velocity components  [Δx/Δt]
        """
        xp = self.xp
        self.u_inf = xp.zeros(self.dim, dtype=xp.float64)     # [Δx/Δt]
        if isinstance(velocity, (int, float)):
            self.u_inf[0] = float(velocity)
        elif isinstance(velocity, (list, tuple)):
            for d in range(min(len(velocity), self.dim)):
                self.u_inf[d] = float(velocity[d])
    
    def _setup_damping(self) -> None:
        """Precompute slab-local damping profile and target equilibrium.
        
        Only the L nodes within the buffer zone are stored, not the
        full domain.  This reduces memory from O(Q·N_total) to
        O(Q·L·transverse_area).
        
        Damping profile (quadratic ramp):
            σ(x) = σ_max · (d / L)²  [dimensionless]
            
            d = distance from inner edge of sponge  [lattice units]
            L = sponge thickness  [lattice units]
        """
        xp = self.xp
        axis = self.location.axis          # 0=x, 1=y, 2=z
        is_min = self.location.is_min

        # ── MPI cut along THIS axis: global-zone overlap (F1 repair) ──
        # The zone lives in GLOBAL rows [0, L) / [N_glob-L, N_glob); this
        # rank damps the window rows that map into it (<= 2 contiguous
        # local runs — the wrap window can split the zone). σ comes from
        # the GLOBAL profile, so co-owned rows damp identically on every
        # rank. sigma_slab/_slab_idx/f_eq_slab stay None here: their
        # full-zone contract (consumed by the esoteric BC conversion,
        # simulation.py) is unrepresentable on a partial window, and the
        # conversion path never sees a slab manager — its existing hard
        # error covers any future misuse.
        if self._cut is not None and axis == self._cut[0]:
            self._zones = []
            self._slab_idx = None
            self.sigma_slab = None
            self.f_eq_slab = None
            a, s, n_own, g, N = self._cut
            W = self.domain_shape[axis]
            L = min(self.thickness, N // 2)
            self._L = L
            d = xp.arange(L, dtype=xp.float64)
            if is_min:
                zone_start = 0
                sigma_glob = self.sigma_max * ((L - d) / L) ** 2
            else:
                zone_start = N - L
                sigma_glob = self.sigma_max * ((d + 1) / L) ** 2
            glob = (np.arange(W) + s - g) % N
            idxs = np.flatnonzero((glob >= zone_start)
                                  & (glob < zone_start + L))
            runs = []
            if idxs.size:
                brk = np.flatnonzero(np.diff(idxs) > 1)
                lo = np.concatenate([[0], brk + 1])
                hi = np.concatenate([brk, [idxs.size - 1]])
                runs = [(int(idxs[i]), int(idxs[j]) + 1)
                        for i, j in zip(lo, hi)]
            for i0, i1 in runs:
                slab_slices = [slice(None)] * (self.dim + 1)
                slab_slices[axis + 1] = slice(i0, i1)
                sig = sigma_glob[xp.asarray(glob[i0:i1] - zone_start)]
                shape = [1] * (self.dim + 1)
                shape[axis + 1] = i1 - i0
                zone_shape = list(self.domain_shape)
                zone_shape[axis] = i1 - i0
                zone_shape = tuple(zone_shape)
                rho_t = xp.full(zone_shape, self.rho_inf, dtype=xp.float64)
                u_t = xp.zeros((self.dim,) + zone_shape, dtype=xp.float64)
                for d_idx in range(self.dim):
                    u_t[d_idx] = self.u_inf[d_idx]
                self._zones.append((
                    tuple(slab_slices), sig.reshape(shape),
                    compute_f_eq(xp, rho_t, u_t, self.c, self.w, self.cs2)))
            return

        # ── Full-domain / transverse-to-cut path (historical, unchanged) ──
        self._zones = None                 # apply() uses the single slab
        N = self.domain_shape[axis]        # domain size along normal axis  [lu]
        L = min(self.thickness, N // 2)    # cap at half domain  [lu]
        self._L = L                        # store effective thickness

        # ── Build slab index (tuple of slices for f[:, ...]) ──
        # f has shape (Q, Nx, Ny[, Nz])  → axis in f is (axis + 1)
        slab_slices = [slice(None)] * (self.dim + 1)   # start with [:] everywhere

        if is_min:
            # Sponge at min face: nodes [0, L)
            slab_slices[axis + 1] = slice(0, L)
            self._slab_idx = tuple(slab_slices)
            
            # 1D damping: node 0 → σ_max, node L-1 → σ ≈ 0
            d = xp.arange(L, dtype=xp.float64)          # [lu]
            sigma_1d = self.sigma_max * ((L - d) / L) ** 2  # [dimensionless]
        else:
            # Sponge at max face: nodes [N-L, N)
            slab_slices[axis + 1] = slice(N - L, N)
            self._slab_idx = tuple(slab_slices)
            
            # 1D damping: node 0 (=N-L) → σ ≈ 0, node L-1 (=N-1) → σ_max
            d = xp.arange(L, dtype=xp.float64)          # [lu]
            sigma_1d = self.sigma_max * ((d + 1) / L) ** 2  # [dimensionless]
        
        # Reshape for broadcasting over slab: (1, ..., L, ..., 1)
        shape = [1] * (self.dim + 1)
        shape[axis + 1] = L
        self.sigma_slab = sigma_1d.reshape(shape)        # [dimensionless]
        
        # ── Precompute target f_eq for the slab only ──
        slab_shape = list(self.domain_shape)
        slab_shape[axis] = L                              # replace N → L
        slab_shape = tuple(slab_shape)
        
        rho_target = xp.full(slab_shape, self.rho_inf, dtype=xp.float64)
        u_target = xp.zeros((self.dim,) + slab_shape, dtype=xp.float64)
        for d_idx in range(self.dim):
            u_target[d_idx] = self.u_inf[d_idx]           # [Δx/Δt]
        
        self.f_eq_slab = compute_f_eq(
            xp, rho_target, u_target, self.c, self.w, self.cs2
        )  # shape (Q, ..., L, ...)
    
    def apply(self, f: 'npt.NDArray') -> None:
        """Apply sponge BC: Neumann cleanup + volume damping.

        Stage A — Neumann cleanup on boundary face flat nodes:
            Replace incoming populations (those that streamed in via
            wrap-around) with values from the interior neighbour. Without
            this, only σ_max of the wrap-around garbage is overwritten by
            Stage B each step; the rest survives and can grow.

        Stage B — Volume damping over buffer slab:
            f[slab] ← f[slab] + σ · (f_eq∞ − f[slab])
            Only L layers are read/written.

        Args:
            f: Distribution function, modified in-place (Q, Nx, Ny[, Nz])
        """
        # Stage A: clean wrap-around garbage at boundary face (only the
        # rank owning the global boundary row under an MPI cut — F1)
        if self._neumann_base is not None:
            self._neumann_base.apply(f)

        # Stage B: volume damping toward (ρ∞, U∞)
        if self._zones is None:
            f_slab = f[self._slab_idx]
            f_slab += self.sigma_slab * (self.f_eq_slab - f_slab)
        else:
            for slab_idx, sigma, f_eq in self._zones:
                f_slab = f[slab_idx]
                f_slab += sigma * (f_eq - f_slab)
    
    def get_info(self) -> str:
        """Return human-readable info string."""
        u_mag = float(self.xp.max(self.xp.abs(self.u_inf)))
        return (f"SpongeLayerBC at {self.location.value}: "
                f"L={self.thickness}, σ_max={self.sigma_max:.3f}, "
                f"U∞={u_mag:.4f}, ρ∞={self.rho_inf:.4f}")