"""
Sponge Layer Boundary Condition

Volume-based damping that drives distributions toward equilibrium
near domain boundaries. NOT a face BC — operates on a buffer zone
of configurable thickness.

Physics:
    f_i(x) ← f_i(x) + σ(x) · [f_eq(ρ∞, U∞) - f_i(x)]
    
    Equivalently:
    f_i(x) ← (1 - σ(x)) · f_i(x) + σ(x) · f_eq_i(ρ∞, U∞)
    
    where σ(x) is a spatial damping profile:
        σ(x) = σ_max · (d/L)²
        
        d = distance from inner edge of sponge layer  [lattice units]
        L = sponge thickness  [lattice units]
        σ_max = maximum damping coefficient  [dimensionless, 0 < σ ≤ 1]

Usage in config:
    "farfield_sponge": {
        "location": "xmax",
        "method": "sponge",
        "velocity": 0.1,        # U∞  [Δx/Δt]
        "rho": 1.0,             # ρ∞  [dimensionless]
        "thickness": 20,        # L   [lattice units]
        "strength": 0.5         # σ_max [dimensionless]
    }

Applied AFTER face BCs and corner BCs in the time loop (Phase 3).

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

from .bc_config import FaceConfig, FaceLocation
from .regularized_utils import compute_f_eq


class SpongeLayerBC:
    """Volume-based sponge layer for non-reflecting boundaries.
    
    Damps distributions toward equilibrium in a buffer zone near
    the domain boundary. The damping increases quadratically from
    the inner edge (σ = 0) to the boundary (σ = σ_max).
    
    This is NOT a FaceBC subclass — it modifies a volume of nodes,
    not just a single face layer.
    
    Attributes:
        location: Which face the sponge is attached to
        thickness: Buffer zone depth  [lattice units]
        sigma_max: Maximum damping  [dimensionless]
        u_inf: Freestream velocity  [Δx/Δt]
        rho_inf: Freestream density  [dimensionless]
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'object',
                 config: FaceConfig,
                 domain_shape: Tuple[int, ...]) -> None:
        """Initialize sponge layer.
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model
            config: FaceConfig with extra['thickness'] and extra['sigma_max']
            domain_shape: (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
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
        
        # Setup velocity and precompute damping
        self._setup_velocity(config.velocity)
        self._setup_damping()
    
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
        """Precompute spatial damping profile σ(x) and target equilibrium.
        
        Damping profile (quadratic ramp):
            σ(x) = σ_max · (d / L)²  [dimensionless]
            
            d = distance from inner edge of sponge  [lattice units]
            L = sponge thickness  [lattice units]
        
        The profile is stored as a broadcastable array matching f's shape.
        """
        xp = self.xp
        axis = self.location.axis
        is_min = self.location.is_min
        N = self.domain_shape[axis]          # domain size along normal axis  [lattice units]
        L = min(self.thickness, N // 2)      # cap at half domain  [lattice units]
        
        # ── 1D damping profile along the normal axis (vectorized) ──
        sigma_1d = xp.zeros(N, dtype=xp.float64)    # [dimensionless]
        idx = xp.arange(N, dtype=xp.float64)
        
        if is_min:
            # Sponge at min face: node 0 gets σ_max, node L gets σ ≈ 0
            mask = idx < L
            d = L - idx                                         # [lattice units]
            sigma_1d[mask] = self.sigma_max * (d[mask] / L) ** 2  # [dimensionless]
        else:
            # Sponge at max face: node N-1 gets σ_max, node N-1-L gets σ ≈ 0
            mask = idx >= (N - L)
            d = idx - (N - L - 1)                               # [lattice units]
            sigma_1d[mask] = self.sigma_max * (d[mask] / L) ** 2
        
        # Reshape for broadcasting: f has shape (Q, Nx, Ny[, Nz])
        # sigma needs shape (1, ..., N, ..., 1) with N at position axis+1
        shape = [1] * (self.dim + 1)    # +1 for Q axis at position 0
        shape[axis + 1] = N
        self.sigma = sigma_1d.reshape(shape)
        
        # ── Precompute target equilibrium f_eq(ρ∞, U∞) ──
        rho_target = xp.full(self.domain_shape, self.rho_inf, dtype=xp.float64)
        u_target = xp.zeros((self.dim,) + self.domain_shape, dtype=xp.float64)
        for d in range(self.dim):
            u_target[d] = self.u_inf[d]      # [Δx/Δt]
        
        self.f_eq_target = compute_f_eq(
            xp, rho_target, u_target, self.c, self.w, self.cs2
        )  # shape (Q, Nx, Ny[, Nz])
    
    def apply(self, f: 'npt.NDArray') -> None:
        """Apply sponge damping to the distribution function.
        
        f ← f + σ(x) · (f_eq∞ - f)
        
        The damping is strongest at the boundary (σ = σ_max) and
        zero at the inner edge of the sponge layer.
        
        Args:
            f: Distribution function, modified in-place (Q, Nx, Ny[, Nz])
        """
        # Vectorized: σ broadcasts over Q and transverse dimensions
        f += self.sigma * (self.f_eq_target - f)
    
    def get_info(self) -> str:
        """Return human-readable info string."""
        u_mag = float(self.xp.max(self.xp.abs(self.u_inf)))
        return (f"SpongeLayerBC at {self.location.value}: "
                f"L={self.thickness}, σ_max={self.sigma_max:.3f}, "
                f"U∞={u_mag:.4f}, ρ∞={self.rho_inf:.4f}")