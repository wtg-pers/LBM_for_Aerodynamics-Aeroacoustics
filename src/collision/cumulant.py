"""
Cumulant Collision Operator for D3Q27 (Geier et al., 2015)

This module implements the parametrized cumulant collision model for the
D3Q27 lattice. The cumulant approach relaxes statistically independent
quantities (cumulants), providing superior Galilean invariance and
numerical stability compared to BGK and classical MRT.

Physical Process (per timestep, per grid node):
    1. f → central moments κ_αβγ         (Chimera forward, Eqs. 43-45)
    2. κ_αβγ → cumulants C_αβγ           (Eqs. 46-54)
    3. Relax cumulants → C*_αβγ           (Eqs. 55-80)
       - Galilean correction for aliasing  (Eqs. 58-63)
    4. C*_αβγ → κ*_αβγ                    (Eqs. 81-87)
    5. κ*_αβγ → f*                         (Chimera backward, Eqs. 88-96)
    6. Add Guo forcing source term S_i     (distribution-space, Method A)

Chimera Algorithm:
    The Chimera (split) approach factorizes the 27-point moment transform
    into 3 sequential 3-point transforms (one per spatial direction),
    reducing the operation count by a factor of 3.

    Forward:  f_{ijk} --z--> κ_{ij|γ} --y--> κ_{i|βγ} --x--> κ_{αβγ}
    Backward: κ*_{αβγ} --x--> κ*_{i|βγ} --y--> κ*_{ij|γ} --z--> f*_{ijk}

Relaxation Parameters:
    ω₁ = 1/τ  : Shear viscosity rate (ν = cs²(1/ω₁ - 0.5))  [1/Δt]
    ω₂        : Bulk viscosity rate (default = 1.0)            [1/Δt]
    ω₃-ω₁₀   : Higher-order rates (default = 1.0 for stability) [1/Δt]

Galilean Correction (Appendix H in Geier 2015):
    Modifies 2nd-order cumulant equilibria to compensate for the absence
    of C₃₀₀, C₀₃₀, C₀₀₃ on the D3Q27 lattice, reducing velocity-dependent
    viscosity errors from O(Ma²) to O(Ma⁴).

Forcing:
    Method A (distribution-space Guo source term), identical to BGK.
    The cumulant-space forcing (sign flip of 1st-order central moments,
    Eqs. 85-87) is applied when external forces are present.

References:
    - Geier, Schönherr, Pasquali, Krafczyk,
      "The cumulant lattice Boltzmann equation in three dimensions:
       Theory and validation",
      Comp. & Math. with Appl. 70(4), 507-547, 2015
    - Geier, Pasquali, Schönherr,
      "Parametrization of the cumulant lattice Boltzmann method for
       fourth order accurate diffusion Part I/II",
      J. Comp. Phys., 2017

Author: LBM Development Team
Date: 2026-03
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from src.collision.base import CollisionOperator


class CumulantCollision(CollisionOperator):
    """Parametrized Cumulant Collision for D3Q27 (Geier et al., 2015).

    Transforms distributions to cumulant space, relaxes each cumulant
    independently toward its equilibrium value, and transforms back.
    This provides:
      - Superior stability near τ → 0.5  (high Re flows)
      - Galilean invariance to O(Ma⁴) with correction terms
      - Independent control of bulk viscosity
      - No hyper-viscosity artifacts

    Limitation: D3Q27 only (27 DOF required for 27 independent cumulants).

    Usage:
        >>> collision = CumulantCollision(xp, lattice)
        >>> collision.collide(f, f_post, rho, u, tau)
        >>> collision.collide(f, f_post, rho, u, tau, force=F)
        >>> f_eq = collision.compute_equilibrium(rho0, u0)  # for init

    Attributes:
        omega_bulk: Bulk viscosity relaxation rate ω₂  [1/Δt]
        omega_3 .. omega_10: Higher-order rates  [1/Δt]
    """

    def __init__(
        self,
        xp: 'ModuleType',
        lattice: 'Lattice',
        omega_bulk: float = 1.0,
        omega_high: float = 1.0,
    ) -> None:
        """Initialize Cumulant collision operator.

        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (must be D3Q27, Q=27)
            omega_bulk: Bulk viscosity rate ω₂  [1/Δt]
                        Default 1.0 (sets bulk viscosity = shear viscosity).
                        Range: (0, 2].
            omega_high: Default rate for ω₃-ω₁₀  [1/Δt]
                        Default 1.0 (maximum damping of high-order modes).
                        Range: (0, 2].

        Raises:
            ValueError: If lattice is not D3Q27.
        """
        super().__init__(xp, lattice)

        if self.Q != 27 or self.dim != 3:
            raise ValueError(
                f"CumulantCollision requires D3Q27 (got D{self.dim}Q{self.Q}). "
                f"Cumulant needs 27 DOF for 27 independent cumulants."
            )

        # ── Lattice constants on device ──────────────────────────
        self._dtype = lattice.dtype
        self._c = xp.asarray(lattice.c, dtype=self._dtype)   # (3, 27)  [Δx/Δt]
        self._w = xp.asarray(lattice.w, dtype=self._dtype)   # (27,)    [dimensionless]
        self._cs2: float = float(lattice.cs2)                # 1/3      [Δx²/Δt²]
        self._inv_cs2: float = 1.0 / self._cs2              # 3.0      [Δt²/Δx²]
        self._inv_cs4: float = 1.0 / (self._cs2 ** 2)       # 9.0      [Δt⁴/Δx⁴]

        # ── Relaxation parameters ────────────────────────────────
        # ω₁ is computed from tau at each collide() call
        self.omega_bulk: float = omega_bulk     # ω₂  [1/Δt]
        self.omega_3: float = omega_high        # ω₃  [1/Δt]
        self.omega_4: float = omega_high        # ω₄  [1/Δt]
        self.omega_5: float = omega_high        # ω₅  [1/Δt]
        self.omega_6: float = omega_high        # ω₆  [1/Δt]
        self.omega_7: float = omega_high        # ω₇  [1/Δt]
        self.omega_8: float = omega_high        # ω₈  [1/Δt]
        self.omega_9: float = omega_high        # ω₉  [1/Δt]
        self.omega_10: float = omega_high       # ω₁₀ [1/Δt]

        # ── Build (i,j,k) ↔ flat Q index mapping ────────────────
        self._build_ijk_map()

    # =================================================================
    # CollisionOperator Interface
    # =================================================================

    def collide(
        self,
        f_in: 'npt.NDArray',
        f_out: 'npt.NDArray',
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau: float,
        force: Optional['npt.NDArray'] = None,
    ) -> None:
        """Perform cumulant collision step in-place.

        Physical Process (Geier et al., 2015, Eqs. 42-96):
            1. Compute macroscopic velocity with half-force shift  (Eq. 42)
            2. f → central moments (Chimera forward)               (Eqs. 43-45)
            3. Central moments → cumulants                         (Eqs. 46-54)
            4. Relax cumulants with Galilean correction             (Eqs. 55-80)
            5. Cumulants → central moments                         (Eqs. 81-87)
            6. Central moments → f* (Chimera backward)             (Eqs. 88-96)
            7. Add Guo forcing source term (Method A)

        Args:
            f_in:  Pre-collision distribution (27, Nx, Ny, Nz)     [dimensionless]
            f_out: Post-collision result — written in-place          [dimensionless]
            rho:   Density field (Nx, Ny, Nz)                       [dimensionless]
            u:     Velocity (Guo-corrected) (3, Nx, Ny, Nz)         [Δx/Δt]
            tau:   Relaxation time                                   [Δt]
            force: Body force (3, Nx, Ny, Nz) or None               [lattice force / lu³]
        """
        xp = self.xp

        # ── Relaxation rate for shear viscosity ──
        omega_1 = 1.0 / tau  # ω₁ = 1/τ  [1/Δt]

        # ── Extract velocity components ──
        # u is already Guo-corrected by Simulation.advance() Step 3
        ux = u[0]  # (Nx, Ny, Nz)  [Δx/Δt]
        uy = u[1]  # (Nx, Ny, Nz)  [Δx/Δt]
        uz = u[2]  # (Nx, Ny, Nz)  [Δx/Δt]

        # ── Step 1: Reshape f to (3,3,3, Nx,Ny,Nz) for Chimera ──
        f_ijk = self._flat_to_ijk(f_in)  # (3, 3, 3, Nx, Ny, Nz)

        # ── Step 2: Forward central moment transform (Chimera) ───
        # Eqs. 43-45: f_{ijk} → κ_{αβγ}
        kappa = self._forward_chimera(f_ijk, ux, uy, uz)
        # kappa shape: (3, 3, 3, Nx, Ny, Nz)
        # kappa[α, β, γ] = κ_{αβγ}  where index 0→order 0, 1→order 1, 2→order 2

        # ── Step 3: Forward cumulant transform ───────────────────
        # Eqs. 46-54: κ_{αβγ} → C_{αβγ}
        # (Modifies kappa in-place for orders ≥ 4 only; orders 2-3 are identical)
        C = self._forward_cumulant_transform(kappa, rho)

        # ── Step 4: Collision — relax cumulants ──────────────────
        # Eqs. 55-80 with Galilean correction (Eqs. 58-63)
        self._relax_cumulants(C, rho, ux, uy, uz, omega_1)

        # ── Step 5: Backward cumulant transform ──────────────────
        # Eqs. 81-87: C*_{αβγ} → κ*_{αβγ}
        self._backward_cumulant_transform(C, rho)

        # ── Step 5b: Force sign-flip of first moments ────────────
        # Eqs. 85-87: κ*_100 = -κ_100, κ*_010 = -κ_010, κ*_001 = -κ_001
        # This implements the "half-force before + half-force after" scheme.
        if force is not None:
            C[1, 0, 0] = -C[1, 0, 0]  # κ*_100 = -κ_100  [Δx/Δt · dimensionless]
            C[0, 1, 0] = -C[0, 1, 0]  # κ*_010 = -κ_010
            C[0, 0, 1] = -C[0, 0, 1]  # κ*_001 = -κ_001

        # ── Step 6: Backward central moment transform (Chimera) ──
        # Eqs. 88-96: κ*_{αβγ} → f*_{ijk}
        f_star_ijk = self._backward_chimera(C, ux, uy, uz)

        # ── Step 7: Convert back to flat Q indexing ──────────────
        self._ijk_to_flat(f_star_ijk, f_out)

        # ── Step 8: Add Guo forcing source term (Method A) ──────
        # S_i = (1 - 1/(2τ)) · w_i · [(c_i-u)/cs² + (c_i·u)·c_i/cs⁴] · F
        if force is not None:
            f_out += self._guo_source_term(force, rho, u, tau)

    def compute_equilibrium(
        self,
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Compute Maxwellian equilibrium distribution.

        For initialization purposes, the cumulant model uses the same
        standard Maxwellian as BGK. The cumulant-specific higher-order
        corrections only matter during collision, not at initialization.

        Formula (2nd-order truncated Maxwellian):
            f_i^eq = w_i · ρ · [1 + (c_i·u)/cs² + (c_i·u)²/(2cs⁴) - |u|²/(2cs²)]

        Args:
            rho: Density field (Nx, Ny, Nz)                 [dimensionless]
            u:   Velocity field (3, Nx, Ny, Nz)              [Δx/Δt]

        Returns:
            f_eq: Equilibrium distribution (27, Nx, Ny, Nz)  [dimensionless]
        """
        xp = self.xp

        # c_i · u  for each direction i  →  shape (Q, Nx, Ny, Nz)
        # c: (3, Q), u: (3, Nx, Ny, Nz) → einsum → (Q, Nx, Ny, Nz)
        cu = xp.einsum('dq,d...->q...', self._c.astype(self._dtype), u)
        # [Δx/Δt · Δx/Δt = Δx²/Δt²]

        # |u|² = u_x² + u_y² + u_z²  →  shape (Nx, Ny, Nz)
        usq = xp.sum(u * u, axis=0)  # [Δx²/Δt²]

        # f_eq = w_i · ρ · (1 + cu/cs² + cu²/(2·cs⁴) - usq/(2·cs²))
        # Reshape w for broadcasting: (Q,) → (Q, 1, 1, 1)
        w = self._w
        ndim_spatial = rho.ndim
        for _ in range(ndim_spatial):
            w = w[..., None]

        f_eq = w * rho * (
            1.0
            + cu * self._inv_cs2                          # [dimensionless]
            + 0.5 * cu * cu * self._inv_cs4               # [dimensionless]
            - 0.5 * usq * self._inv_cs2                   # [dimensionless]
        )  # [dimensionless]

        return f_eq

    # =================================================================
    # Index Mapping: flat Q ↔ (i,j,k) with i,j,k ∈ {-1,0,1}
    # =================================================================

    def _build_ijk_map(self) -> None:
        """Build bidirectional mapping between flat Q index and (i,j,k).

        Geier's convention: i,j,k ∈ {-1, 0, 1}
        Array indexing:     i_idx = i + 1 ∈ {0, 1, 2}

        Creates:
            _q_to_ijk_idx: (27, 3) array mapping q → (i_idx, j_idx, k_idx)
            _ijk_to_q:     (3, 3, 3) array mapping (i_idx, j_idx, k_idx) → q
        """
        import numpy as np

        c_np = np.array(self.lattice.c.get() if hasattr(self.lattice.c, 'get')
                        else self.lattice.c)  # (3, 27) on CPU

        # q → (i_idx, j_idx, k_idx)
        self._q_to_ijk_idx = np.zeros((27, 3), dtype=np.int32)
        self._ijk_to_q = np.full((3, 3, 3), -1, dtype=np.int32)

        for q in range(27):
            cx, cy, cz = int(c_np[0, q]), int(c_np[1, q]), int(c_np[2, q])
            i_idx, j_idx, k_idx = cx + 1, cy + 1, cz + 1
            self._q_to_ijk_idx[q] = [i_idx, j_idx, k_idx]
            self._ijk_to_q[i_idx, j_idx, k_idx] = q

        # Verify all 27 slots filled
        assert np.all(self._ijk_to_q >= 0), "D3Q27 mapping incomplete!"

        # Store on device
        self._q_to_ijk_idx_dev = self.xp.asarray(self._q_to_ijk_idx)
        self._ijk_to_q_dev = self.xp.asarray(self._ijk_to_q)

    def _flat_to_ijk(self, f: 'npt.NDArray') -> 'npt.NDArray':
        """Reshape f from (27, Nx, Ny, Nz) to (3, 3, 3, Nx, Ny, Nz).

        Uses precomputed mapping: f_ijk[i_idx, j_idx, k_idx, ...] = f[q, ...]

        Args:
            f: Distribution in flat indexing (27, Nx, Ny, Nz)

        Returns:
            f_ijk: Distribution in (i,j,k) indexing (3, 3, 3, Nx, Ny, Nz)
        """
        xp = self.xp
        spatial_shape = f.shape[1:]  # (Nx, Ny, Nz)
        f_ijk = xp.empty((3, 3, 3) + spatial_shape, dtype=f.dtype)

        for q in range(27):
            i, j, k = self._q_to_ijk_idx[q]
            f_ijk[i, j, k] = f[q]

        return f_ijk

    def _ijk_to_flat(
        self,
        f_ijk: 'npt.NDArray',
        f_out: 'npt.NDArray',
    ) -> None:
        """Write f from (3, 3, 3, Nx, Ny, Nz) back to (27, Nx, Ny, Nz).

        Args:
            f_ijk: Distribution in (i,j,k) indexing (3, 3, 3, Nx, Ny, Nz)
            f_out: Output array in flat indexing (27, Nx, Ny, Nz) — written in-place
        """
        for q in range(27):
            i, j, k = self._q_to_ijk_idx[q]
            f_out[q] = f_ijk[i, j, k]

    # =================================================================
    # Forward Chimera Transform: f → central moments
    # =================================================================

    def _forward_chimera(
        self,
        f_ijk: 'npt.NDArray',
        ux: 'npt.NDArray',
        uy: 'npt.NDArray',
        uz: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Forward Chimera: f_{ijk} → central moments κ_{αβγ}.

        Geier et al. (2015), Eqs. 43-45.
        Split transform: z-direction → y-direction → x-direction.

        Each 1D step transforms 3 values {f_{-1}, f_0, f_1} into
        3 central moments {κ_0, κ_1, κ_2} around velocity component v_d:

            κ_0 = f_{-1} + f_0 + f_1
            κ_1 = f_{-1}·(-1-v) + f_0·(-v) + f_1·(1-v)
            κ_2 = f_{-1}·(-1-v)² + f_0·(-v)² + f_1·(1-v)²

        Simplification (for efficiency):
            κ_1 = -f_{-1} + f_1 - v·κ_0
            κ_2 = f_{-1} + f_1 - 2v·κ_1 - v²·κ_0

        Args:
            f_ijk: (3, 3, 3, Nx, Ny, Nz) distribution  [dimensionless]
            ux, uy, uz: Velocity components (Nx, Ny, Nz)  [Δx/Δt]

        Returns:
            kappa: (3, 3, 3, Nx, Ny, Nz) central moments
                   kappa[α, β, γ] = κ_{αβγ}
        """
        xp = self.xp

        # ── Step 1: z-direction (Eq. 43) ─────────────────────────
        # κ_{ij|γ} = Σ_k f_{ijk} (k - w)^γ
        # Input: f_ijk[i, j, k, ...]  Output: kz[i, j, γ, ...]
        kz = self._chimera_1d(f_ijk, uz, axis=2)

        # ── Step 2: y-direction (Eq. 44) ─────────────────────────
        # κ_{i|βγ} = Σ_j κ_{ij|γ} (j - v)^β
        # Input: kz[i, j, γ, ...]  Output: ky[i, β, γ, ...]
        ky = self._chimera_1d(kz, uy, axis=1)

        # ── Step 3: x-direction (Eq. 45) ─────────────────────────
        # κ_{αβγ} = Σ_i κ_{i|βγ} (i - u)^α
        # Input: ky[i, β, γ, ...]  Output: kappa[α, β, γ, ...]
        kappa = self._chimera_1d(ky, ux, axis=0)

        return kappa

    def _chimera_1d(
        self,
        data: 'npt.NDArray',
        vel: 'npt.NDArray',
        axis: int,
    ) -> 'npt.NDArray':
        """Single-direction Chimera transform.

        Given data[..., k, ...] for k_idx ∈ {0,1,2} (corresponding to
        k ∈ {-1, 0, 1}), compute central moments around velocity v:

            m_0 = d_{-1} + d_0 + d_1
            m_1 = -d_{-1} + d_1 - v·m_0
            m_2 = d_{-1} + d_1 - 2v·m_1 - v²·m_0

        Derivation from direct formula:
            m_0 = d_{-1} + d_0 + d_1
            m_1 = d_{-1}(-1-v) + d_0(-v) + d_1(1-v)
                = -d_{-1} + d_1 - v(d_{-1} + d_0 + d_1)
                = -d_{-1} + d_1 - v·m_0
            m_2 = d_{-1}(-1-v)² + d_0 v² + d_1(1-v)²
                = d_{-1}(1+2v+v²) + d_0 v² + d_1(1-2v+v²)
                = (d_{-1}+d_1) + 2v(d_{-1}-d_1) + v²(d_{-1}+d_0+d_1)
                = (d_{-1}+d_1) + 2v(d_{-1}-d_1) + v²·m_0
            Using m_1 = -d_{-1} + d_1 - v·m_0:
                d_{-1} - d_1 = -m_1 - v·m_0
            So m_2 = (d_{-1}+d_1) + 2v(-m_1-v·m_0) + v²·m_0
                   = (d_{-1}+d_1) - 2v·m_1 - v²·m_0

        Args:
            data: (..., 3, ..., Nx, Ny, Nz) input array
                  The size-3 axis is at position `axis`.
            vel: (Nx, Ny, Nz) velocity component  [Δx/Δt]
            axis: Which of the first 3 axes (0, 1, or 2) to transform

        Returns:
            result: Same shape as data, with moments replacing values along `axis`
        """
        xp = self.xp

        # Extract slices along the specified axis
        # axis ∈ {0, 1, 2} refers to one of the 3 ijk axes
        # data shape is (3, 3, 3, Nx, Ny, Nz) or subsets thereof
        slc_m = [slice(None)] * data.ndim  # k_idx = 0 → k = -1
        slc_0 = [slice(None)] * data.ndim  # k_idx = 1 → k = 0
        slc_p = [slice(None)] * data.ndim  # k_idx = 2 → k = +1

        slc_m[axis] = 0
        slc_0[axis] = 1
        slc_p[axis] = 2

        d_m = data[tuple(slc_m)]   # d_{-1}
        d_0 = data[tuple(slc_0)]   # d_0
        d_p = data[tuple(slc_p)]   # d_1

        # Compute central moments
        m0 = d_m + d_0 + d_p                                  # [same unit as data]
        m1 = -d_m + d_p - vel * m0                             # [vel unit × data unit]
        m2 = d_m + d_p - 2.0 * vel * m1 - vel * vel * m0      # [vel² unit × data unit]

        # Write back to result array
        result = xp.empty_like(data)
        slc_out_0 = [slice(None)] * data.ndim
        slc_out_1 = [slice(None)] * data.ndim
        slc_out_2 = [slice(None)] * data.ndim
        slc_out_0[axis] = 0
        slc_out_1[axis] = 1
        slc_out_2[axis] = 2

        result[tuple(slc_out_0)] = m0
        result[tuple(slc_out_1)] = m1
        result[tuple(slc_out_2)] = m2

        return result

    # =================================================================
    # Forward Cumulant Transform: central moments → cumulants
    # =================================================================

    def _forward_cumulant_transform(
        self,
        kappa: 'npt.NDArray',
        rho: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Convert central moments to cumulants.

        Geier et al. (2015), Eqs. 46-54.

        Orders 0-3: cumulants = central moments (identical).
        Orders 4+:  nonlinear corrections involving lower-order products.

        Notation: C_{abc} = c_{abc} · ρ  (unnormalized cumulants, Eq. 46)

        The input kappa is modified IN-PLACE for efficiency:
        after this call, kappa contains cumulants C_{αβγ}.

        Args:
            kappa: (3, 3, 3, Nx, Ny, Nz) central moments  [various units]
                   Modified in-place to contain cumulants.
            rho:   (Nx, Ny, Nz) density  [dimensionless]

        Returns:
            kappa: Same array, now containing cumulants C_{αβγ}
        """
        # Shorthand for readability: K[a,b,c] = κ_{abc}
        K = kappa
        inv_rho = 1.0 / rho  # [1/dimensionless]

        # ─── Orders 0, 1, 2, 3: C = κ (Eqs. 47-50) ────────────
        # No modification needed.

        # ─── Save original 4th-order κ values ────────────────────
        # Eq. 54 (C_{222}) requires ORIGINAL κ_{022}, κ_{202}, κ_{220},
        # κ_{211}, κ_{121}, κ_{112} which get overwritten below.
        # 2nd/3rd order κ = C (identical), so no copies needed for those.
        xp = self.xp
        k220_orig = xp.copy(K[2, 2, 0])
        k202_orig = xp.copy(K[2, 0, 2])
        k022_orig = xp.copy(K[0, 2, 2])
        k211_orig = xp.copy(K[2, 1, 1])
        k121_orig = xp.copy(K[1, 2, 1])
        k112_orig = xp.copy(K[1, 1, 2])

        # ─── Order 4: C_{211} and permutations (Eq. 51) ─────────
        # C_{211} = κ_{211} − (κ_{200}·κ_{011} + 2·κ_{110}·κ_{101}) / ρ
        K[2, 1, 1] -= (K[2, 0, 0] * K[0, 1, 1]
                       + 2.0 * K[1, 1, 0] * K[1, 0, 1]) * inv_rho
        K[1, 2, 1] -= (K[0, 2, 0] * K[1, 0, 1]
                       + 2.0 * K[1, 1, 0] * K[0, 1, 1]) * inv_rho
        K[1, 1, 2] -= (K[0, 0, 2] * K[1, 1, 0]
                       + 2.0 * K[1, 0, 1] * K[0, 1, 1]) * inv_rho

        # C_{220} and permutations (Eq. 52)
        # C_{220} = κ_{220} − (κ_{200}·κ_{020} + 2·κ²_{110}) / ρ
        K[2, 2, 0] -= (K[2, 0, 0] * K[0, 2, 0]
                       + 2.0 * K[1, 1, 0] ** 2) * inv_rho
        K[2, 0, 2] -= (K[2, 0, 0] * K[0, 0, 2]
                       + 2.0 * K[1, 0, 1] ** 2) * inv_rho
        K[0, 2, 2] -= (K[0, 2, 0] * K[0, 0, 2]
                       + 2.0 * K[0, 1, 1] ** 2) * inv_rho

        # ─── Order 5: C_{122} and permutations (Eq. 53) ─────────
        # All RHS terms are 2nd × 3rd order products → C = κ, no issue.
        K[1, 2, 2] -= (K[0, 0, 2] * K[1, 2, 0] + K[0, 2, 0] * K[1, 0, 2]
                       + 4.0 * K[0, 1, 1] * K[1, 1, 1]
                       + 2.0 * (K[1, 0, 1] * K[0, 2, 1]
                                + K[1, 1, 0] * K[0, 1, 2])) * inv_rho
        K[2, 1, 2] -= (K[0, 0, 2] * K[2, 1, 0] + K[2, 0, 0] * K[0, 1, 2]
                       + 4.0 * K[1, 0, 1] * K[1, 1, 1]
                       + 2.0 * (K[0, 1, 1] * K[2, 0, 1]
                                + K[1, 1, 0] * K[1, 0, 2])) * inv_rho
        K[2, 2, 1] -= (K[0, 2, 0] * K[2, 0, 1] + K[2, 0, 0] * K[0, 2, 1]
                       + 4.0 * K[1, 1, 0] * K[1, 1, 1]
                       + 2.0 * (K[0, 1, 1] * K[2, 1, 0]
                                + K[1, 0, 1] * K[1, 2, 0])) * inv_rho

        # ─── Order 6: C_{222} (Eq. 54) ──────────────────────────
        # IMPORTANT: Uses ORIGINAL κ values for 4th-order terms.
        # 2nd/3rd order terms use K directly (since C = κ for order ≤ 3).
        # 4th order terms use saved originals (k220_orig, etc.).
        K[2, 2, 2] -= (
            4.0 * K[1, 1, 1] ** 2
            + K[2, 0, 0] * k022_orig
            + K[0, 2, 0] * k202_orig
            + K[0, 0, 2] * k220_orig
            + 4.0 * (K[0, 1, 1] * k211_orig
                     + K[1, 0, 1] * k121_orig
                     + K[1, 1, 0] * k112_orig)
            + 2.0 * (K[1, 2, 0] * K[1, 0, 2]
                     + K[2, 1, 0] * K[0, 1, 2]
                     + K[2, 0, 1] * K[0, 2, 1])
        ) * inv_rho
        K[2, 2, 2] += (
            16.0 * K[1, 1, 0] * K[1, 0, 1] * K[0, 1, 1]
            + 4.0 * (K[1, 0, 1] ** 2 * K[0, 2, 0]
                     + K[0, 1, 1] ** 2 * K[2, 0, 0]
                     + K[1, 1, 0] ** 2 * K[0, 0, 2])
            + 2.0 * K[2, 0, 0] * K[0, 2, 0] * K[0, 0, 2]
        ) * inv_rho ** 2

        return K

    # =================================================================
    # Collision: Relax Cumulants
    # =================================================================

    def _relax_cumulants(
        self,
        C: 'npt.NDArray',
        rho: 'npt.NDArray',
        ux: 'npt.NDArray',
        uy: 'npt.NDArray',
        uz: 'npt.NDArray',
        omega_1: float,
    ) -> None:
        """Relax cumulants toward equilibrium values.

        Geier et al. (2015), Eqs. 55-80 with Galilean correction (Eqs. 58-63).

        The relaxation is performed in-place on the C array.

        Cumulant equilibria:
            - 2nd order diagonal: C^eq_{200} = ρ/3 (= ρ·cs²)  [dimensionless]
            - All others: C^eq = 0

        Relaxation formula: C* = (1 - ω) · C + ω · C^eq
                           or equivalently: C* = C + ω · (C^eq - C)

        Args:
            C:   (3, 3, 3, Nx, Ny, Nz) cumulants — modified in-place
            rho: (Nx, Ny, Nz) density  [dimensionless]
            ux, uy, uz: Velocity components  [Δx/Δt]
            omega_1: Shear relaxation rate = 1/τ  [1/Δt]
        """
        xp = self.xp
        w1 = omega_1         # ω₁ — shear viscosity
        w2 = self.omega_bulk  # ω₂ — bulk viscosity
        w3 = self.omega_3
        w4 = self.omega_4
        w5 = self.omega_5
        w6 = self.omega_6
        w7 = self.omega_7
        w8 = self.omega_8
        w9 = self.omega_9
        w10 = self.omega_10

        # ── Shorthand for 2nd order cumulants ────────────────────
        C200 = C[2, 0, 0]  # [dimensionless]
        C020 = C[0, 2, 0]
        C002 = C[0, 0, 2]

        # ── Galilean correction: velocity gradients (Eqs. 58-60) ─
        # Dxu ≈ -ω₁/(2ρ)·(2C_{200} - C_{020} - C_{002})
        #        -ω₂/(2ρ)·(C_{200} + C_{020} + C_{002} - ρ)
        # [dimensionless] (strain rate × Δt)
        inv_rho = 1.0 / rho
        Dxu = (-w1 * 0.5 * inv_rho * (2.0 * C200 - C020 - C002)
               - w2 * 0.5 * inv_rho * (C200 + C020 + C002 - rho))
        Dyv = Dxu + 1.5 * w1 * inv_rho * (C200 - C020)  # (Eq. 59)
        Dzw = Dxu + 1.5 * w1 * inv_rho * (C200 - C002)  # (Eq. 60)

        # ── 2nd order off-diagonal: C_{110}, C_{101}, C_{011} (Eqs. 55-57) ──
        # C*_{110} = (1 - ω₁)·C_{110}   [eq = 0]
        C[1, 1, 0] *= (1.0 - w1)
        C[1, 0, 1] *= (1.0 - w1)
        C[0, 1, 1] *= (1.0 - w1)

        # ── 2nd order diagonal with Galilean correction (Eqs. 61-63) ──
        # These are relaxed as trace/traceless combinations:
        #
        # Normal stress differences (traceless, ω₁):
        #   (C*_{200} - C*_{020}) = (1-ω₁)(C_{200} - C_{020})
        #                           - 3ρ(1-ω₁/2)(u²·Dxu - v²·Dyv)
        #
        # Trace (bulk, ω₂):
        #   (C*_{200} + C*_{020} + C*_{002}) = ρ·ω₂ + (1-ω₂)(C_{200}+C_{020}+C_{002})
        #                                      - 3ρ(1-ω₂/2)(u²·Dxu + v²·Dyv + w²·Dzw)

        # Galilean correction contributions  [dimensionless]
        Gal_xx = -3.0 * rho * (1.0 - 0.5 * w1) * ux * ux * Dxu
        Gal_yy = -3.0 * rho * (1.0 - 0.5 * w1) * uy * uy * Dyv
        Gal_zz = -3.0 * rho * (1.0 - 0.5 * w1) * uz * uz * Dzw

        Gal_bulk_xx = -3.0 * rho * (1.0 - 0.5 * w2) * ux * ux * Dxu
        Gal_bulk_yy = -3.0 * rho * (1.0 - 0.5 * w2) * uy * uy * Dyv
        Gal_bulk_zz = -3.0 * rho * (1.0 - 0.5 * w2) * uz * uz * Dzw

        # Relaxed combinations (Eqs. 61-63):
        diff_xy = (1.0 - w1) * (C200 - C020) + (Gal_xx - Gal_yy)  # (Eq. 61)
        diff_xz = (1.0 - w1) * (C200 - C002) + (Gal_xx - Gal_zz)  # (Eq. 62)
        trace = (w2 * rho
                 + (1.0 - w2) * (C200 + C020 + C002)
                 + (Gal_bulk_xx + Gal_bulk_yy + Gal_bulk_zz))       # (Eq. 63)

        # Reconstruct individual C*_{200}, C*_{020}, C*_{002} from:
        #   C*_{200} - C*_{020} = diff_xy
        #   C*_{200} - C*_{002} = diff_xz
        #   C*_{200} + C*_{020} + C*_{002} = trace
        # Solution:
        #   C*_{200} = (trace + diff_xy + diff_xz) / 3
        #   C*_{020} = (trace - 2·diff_xy + diff_xz) / 3  → no:
        # Let me solve correctly:
        #   From diff_xy: C020 = C200 - diff_xy
        #   From diff_xz: C002 = C200 - diff_xz
        #   From trace:   C200 + (C200 - diff_xy) + (C200 - diff_xz) = trace
        #                 3·C200 = trace + diff_xy + diff_xz
        C[2, 0, 0] = (trace + diff_xy + diff_xz) / 3.0
        C[0, 2, 0] = C[2, 0, 0] - diff_xy
        C[0, 0, 2] = C[2, 0, 0] - diff_xz

        # ── 3rd order (Eqs. 64-70) ──────────────────────────────
        # Symmetric combinations (ω₃):
        # C*_{120} + C*_{102} = (1-ω₃)(C_{120} + C_{102})
        sum_120_102 = (1.0 - w3) * (C[1, 2, 0] + C[1, 0, 2])  # (Eq. 64)
        sum_210_012 = (1.0 - w3) * (C[2, 1, 0] + C[0, 1, 2])  # (Eq. 65)
        sum_201_021 = (1.0 - w3) * (C[2, 0, 1] + C[0, 2, 1])  # (Eq. 66)

        # Anti-symmetric combinations (ω₄):
        dif_120_102 = (1.0 - w4) * (C[1, 2, 0] - C[1, 0, 2])  # (Eq. 67)
        dif_210_012 = (1.0 - w4) * (C[2, 1, 0] - C[0, 1, 2])  # (Eq. 68)
        dif_201_021 = (1.0 - w4) * (C[2, 0, 1] - C[0, 2, 1])  # (Eq. 69)

        # Reconstruct individual 3rd order cumulants
        C[1, 2, 0] = 0.5 * (sum_120_102 + dif_120_102)
        C[1, 0, 2] = 0.5 * (sum_120_102 - dif_120_102)
        C[2, 1, 0] = 0.5 * (sum_210_012 + dif_210_012)
        C[0, 1, 2] = 0.5 * (sum_210_012 - dif_210_012)
        C[2, 0, 1] = 0.5 * (sum_201_021 + dif_201_021)
        C[0, 2, 1] = 0.5 * (sum_201_021 - dif_201_021)

        # C*_{111} (ω₅, Eq. 70):
        C[1, 1, 1] *= (1.0 - w5)

        # ── 4th order (Eqs. 71-76) ──────────────────────────────
        # Traceless symmetric (ω₆):
        # C*_{220} - 2C*_{202} + C*_{022} = (1-ω₆)(C_{220} - 2C_{202} + C_{022})
        C220 = C[2, 2, 0]
        C202 = C[2, 0, 2]
        C022 = C[0, 2, 2]

        comb1 = (1.0 - w6) * (C220 - 2.0 * C202 + C022)  # (Eq. 71)
        comb2 = (1.0 - w6) * (C220 + C202 - 2.0 * C022)  # (Eq. 72)
        comb3 = (1.0 - w7) * (C220 + C202 + C022)         # (Eq. 73) — ω₇

        # Reconstruct:
        #   comb1 = C220 - 2·C202 + C022
        #   comb2 = C220 + C202 - 2·C022
        #   comb3 = C220 + C202 + C022
        # Adding comb1 + comb2: 2·C220 - C202 - C022
        # comb3: C220 + C202 + C022
        # → 2·C220 - C202 - C022 + C220 + C202 + C022 = 3·C220
        # So C220 = (comb1 + comb2 + comb3) / 3
        C[2, 2, 0] = (comb1 + comb2 + comb3) / 3.0
        C[2, 0, 2] = C[2, 2, 0] - comb1  # from comb1 = C220 - 2C202 + C022
        # wait, let me redo:
        # comb1 = C220 - 2C202 + C022  →  C022 = comb1 - C220 + 2C202
        # comb2 = C220 + C202 - 2C022
        # comb3 = C220 + C202 + C022
        # From comb2 + comb3: 2C220 + 2C202 - C022 = comb2 + comb3 ... complicated
        # Better: solve the 3×3 system.
        # Let A = C220, B = C202, C_ = C022:
        #   A - 2B + C_  = comb1
        #   A +  B - 2C_ = comb2
        #   A +  B + C_  = comb3
        # From row3 - row1: 3B = comb3 - comb1 → B = (comb3 - comb1) / 3
        # From row3 - row2: 3C_ = comb3 - comb2 → C_ = (comb3 - comb2) / 3
        # From row3: A = comb3 - B - C_ = comb3 - (comb3-comb1)/3 - (comb3-comb2)/3
        #            = (3·comb3 - comb3+comb1 - comb3+comb2)/3 = (comb1+comb2+comb3)/3
        C[2, 2, 0] = (comb1 + comb2 + comb3) / 3.0
        C[2, 0, 2] = (comb3 - comb1) / 3.0
        C[0, 2, 2] = (comb3 - comb2) / 3.0

        # C*_{211}, C*_{121}, C*_{112} (ω₈, Eqs. 74-76):
        C[2, 1, 1] *= (1.0 - w8)
        C[1, 2, 1] *= (1.0 - w8)
        C[1, 1, 2] *= (1.0 - w8)

        # ── 5th order (Eqs. 77-79) ──────────────────────────────
        C[2, 2, 1] *= (1.0 - w9)
        C[2, 1, 2] *= (1.0 - w9)
        C[1, 2, 2] *= (1.0 - w9)

        # ── 6th order (Eq. 80) ──────────────────────────────────
        C[2, 2, 2] *= (1.0 - w10)

    # =================================================================
    # Backward Cumulant Transform: cumulants → central moments
    # =================================================================

    def _backward_cumulant_transform(
        self,
        C: 'npt.NDArray',
        rho: 'npt.NDArray',
    ) -> None:
        """Convert relaxed cumulants back to central moments.

        Geier et al. (2015), Eqs. 81-84.
        Inverse of _forward_cumulant_transform.

        Orders 0-3: no change (cumulants = central moments).
        Orders 4+:  add back nonlinear lower-order products.

        Modified in-place: after this call, C contains κ*_{αβγ}.

        Args:
            C:   (3, 3, 3, Nx, Ny, Nz) relaxed cumulants → central moments
            rho: (Nx, Ny, Nz) density  [dimensionless]
        """
        inv_rho = 1.0 / rho

        # ─── Order 4: κ*_{220} and permutations (Eq. 82) ────────
        # κ*_{220} = C*_{220} + (κ*_{200}·κ*_{020} + 2·κ*²_{110}) / ρ
        # Note: 2nd/3rd order κ* = C* (identical), already relaxed.
        C[2, 2, 0] += (C[2, 0, 0] * C[0, 2, 0]
                       + 2.0 * C[1, 1, 0] ** 2) * inv_rho
        C[2, 0, 2] += (C[2, 0, 0] * C[0, 0, 2]
                       + 2.0 * C[1, 0, 1] ** 2) * inv_rho
        C[0, 2, 2] += (C[0, 2, 0] * C[0, 0, 2]
                       + 2.0 * C[0, 1, 1] ** 2) * inv_rho

        # κ*_{211} and permutations (Eq. 81)
        C[2, 1, 1] += (C[2, 0, 0] * C[0, 1, 1]
                       + 2.0 * C[1, 1, 0] * C[1, 0, 1]) * inv_rho
        C[1, 2, 1] += (C[0, 2, 0] * C[1, 0, 1]
                       + 2.0 * C[1, 1, 0] * C[0, 1, 1]) * inv_rho
        C[1, 1, 2] += (C[0, 0, 2] * C[1, 1, 0]
                       + 2.0 * C[1, 0, 1] * C[0, 1, 1]) * inv_rho

        # ─── Order 5: κ*_{122} and permutations (Eq. 83) ────────
        C[1, 2, 2] += (C[0, 0, 2] * C[1, 2, 0] + C[0, 2, 0] * C[1, 0, 2]
                       + 4.0 * C[0, 1, 1] * C[1, 1, 1]
                       + 2.0 * (C[1, 0, 1] * C[0, 2, 1]
                                + C[1, 1, 0] * C[0, 1, 2])) * inv_rho
        C[2, 1, 2] += (C[0, 0, 2] * C[2, 1, 0] + C[2, 0, 0] * C[0, 1, 2]
                       + 4.0 * C[1, 0, 1] * C[1, 1, 1]
                       + 2.0 * (C[0, 1, 1] * C[2, 0, 1]
                                + C[1, 1, 0] * C[1, 0, 2])) * inv_rho
        C[2, 2, 1] += (C[0, 2, 0] * C[2, 0, 1] + C[2, 0, 0] * C[0, 2, 1]
                       + 4.0 * C[1, 1, 0] * C[1, 1, 1]
                       + 2.0 * (C[0, 1, 1] * C[2, 1, 0]
                                + C[1, 0, 1] * C[1, 2, 0])) * inv_rho

        # ─── Order 6: κ*_{222} (Eq. 84) ─────────────────────────
        inv_rho2 = inv_rho * inv_rho
        C[2, 2, 2] += (
            4.0 * C[1, 1, 1] ** 2
            + C[2, 0, 0] * C[0, 2, 2]
            + C[0, 2, 0] * C[2, 0, 2]
            + C[0, 0, 2] * C[2, 2, 0]
            + 4.0 * (C[0, 1, 1] * C[2, 1, 1]
                     + C[1, 0, 1] * C[1, 2, 1]
                     + C[1, 1, 0] * C[1, 1, 2])
            + 2.0 * (C[1, 2, 0] * C[1, 0, 2]
                     + C[2, 1, 0] * C[0, 1, 2]
                     + C[2, 0, 1] * C[0, 2, 1])
        ) * inv_rho
        C[2, 2, 2] -= (
            16.0 * C[1, 1, 0] * C[1, 0, 1] * C[0, 1, 1]
            + 4.0 * (C[1, 0, 1] ** 2 * C[0, 2, 0]
                     + C[0, 1, 1] ** 2 * C[2, 0, 0]
                     + C[1, 1, 0] ** 2 * C[0, 0, 2])
            + 2.0 * C[2, 0, 0] * C[0, 2, 0] * C[0, 0, 2]
        ) * inv_rho2

    # =================================================================
    # Backward Chimera Transform: central moments → f*
    # =================================================================

    def _backward_chimera(
        self,
        kappa: 'npt.NDArray',
        ux: 'npt.NDArray',
        uy: 'npt.NDArray',
        uz: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Backward Chimera: central moments κ*_{αβγ} → f*_{ijk}.

        Geier et al. (2015), Eqs. 88-96.
        Split transform: x-direction → y-direction → z-direction.

        Each 1D step reconstructs 3 values {f_{-1}, f_0, f_1} from
        3 central moments {κ_0, κ_1, κ_2} around velocity v:

            f_0  = κ_0·(1 - v²) - 2v·κ_1 - κ_2           (Eq. 88)
            f_{-1} = (κ_0·(v²-v) + κ_1·(2v-1) + κ_2) / 2  (Eq. 89)
            f_{+1} = (κ_0·(v²+v) + κ_1·(2v+1) + κ_2) / 2  (Eq. 90)

        Args:
            kappa: (3, 3, 3, Nx, Ny, Nz) central moments  [various]
            ux, uy, uz: Velocity components (Nx, Ny, Nz)  [Δx/Δt]

        Returns:
            f_ijk: (3, 3, 3, Nx, Ny, Nz) post-collision distributions
        """
        # ── Step 1: x-direction (Eqs. 88-90) ────────────────────
        kx = self._inv_chimera_1d(kappa, ux, axis=0)

        # ── Step 2: y-direction (Eqs. 91-93) ────────────────────
        ky = self._inv_chimera_1d(kx, uy, axis=1)

        # ── Step 3: z-direction (Eqs. 94-96) ────────────────────
        f_ijk = self._inv_chimera_1d(ky, uz, axis=2)

        return f_ijk

    def _inv_chimera_1d(
        self,
        data: 'npt.NDArray',
        vel: 'npt.NDArray',
        axis: int,
    ) -> 'npt.NDArray':
        """Single-direction inverse Chimera transform.

        Reconstruct 3 distribution values from 3 central moments:

            f_0   = m_0·(1 - v²) - 2v·m_1 - m_2                 (Eq. 88)
            f_{-1} = (m_0·(v² - v) + m_1·(2v - 1) + m_2) / 2    (Eq. 89)
            f_{+1} = (m_0·(v² + v) + m_1·(2v + 1) + m_2) / 2    (Eq. 90)

        Args:
            data: (..., 3, ..., Nx, Ny, Nz) moment data (size-3 along `axis`)
            vel: (Nx, Ny, Nz) velocity component  [Δx/Δt]
            axis: Which of the first 3 axes to inverse-transform

        Returns:
            result: Same shape, with distribution values replacing moments
        """
        xp = self.xp

        slc_0 = [slice(None)] * data.ndim
        slc_1 = [slice(None)] * data.ndim
        slc_2 = [slice(None)] * data.ndim
        slc_0[axis] = 0
        slc_1[axis] = 1
        slc_2[axis] = 2

        m0 = data[tuple(slc_0)]  # moment order 0
        m1 = data[tuple(slc_1)]  # moment order 1
        m2 = data[tuple(slc_2)]  # moment order 2

        v2 = vel * vel  # v²  [Δx²/Δt²]

        # Eqs. 88-90 (Geier 2015)
        f_center = m0 * (1.0 - v2) - 2.0 * vel * m1 - m2        # f_0   [dimensionless]
        f_minus = 0.5 * (m0 * (v2 - vel) + m1 * (2.0 * vel - 1.0) + m2)  # f_{-1}
        f_plus = 0.5 * (m0 * (v2 + vel) + m1 * (2.0 * vel + 1.0) + m2)   # f_{+1}

        result = xp.empty_like(data)

        slc_out_m = [slice(None)] * data.ndim   # k_idx = 0 → k = -1
        slc_out_0 = [slice(None)] * data.ndim   # k_idx = 1 → k = 0
        slc_out_p = [slice(None)] * data.ndim   # k_idx = 2 → k = +1
        slc_out_m[axis] = 0
        slc_out_0[axis] = 1
        slc_out_p[axis] = 2

        result[tuple(slc_out_m)] = f_minus
        result[tuple(slc_out_0)] = f_center
        result[tuple(slc_out_p)] = f_plus

        return result

    # =================================================================
    # Guo Forcing Source Term (Method A — distribution space)
    # =================================================================

    def _guo_source_term(
        self,
        force: 'npt.NDArray',
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau: float,
    ) -> 'npt.NDArray':
        """Compute Guo forcing source term in distribution space.

        This is identical to the BGK implementation (Method A from handoff).

        Formula (Guo et al., Phys. Rev. E 65, 2002):
            S_i = (1 - 1/(2τ)) · w_i ·
                  [(c_i - u)/cs² + (c_i·u)·c_i/cs⁴] · F

        Args:
            force: Body force (3, Nx, Ny, Nz)  [lattice force / lu³]
            rho:   Density (Nx, Ny, Nz)  [dimensionless]
            u:     Velocity (3, Nx, Ny, Nz)  [Δx/Δt]
            tau:   Relaxation time  [Δt]

        Returns:
            S: Source term (27, Nx, Ny, Nz)  [dimensionless]
        """
        xp = self.xp

        # Pre-factor:  (1 - 1/(2τ))  [dimensionless]
        prefactor = 1.0 - 0.5 / tau

        # c_i · u → (Q, Nx, Ny, Nz)  [Δx²/Δt²]
        cu = xp.einsum('dq,d...->q...', self._c.astype(self._dtype), u)

        # (c_i - u) / cs²  →  (dim, Q, Nx, Ny, Nz)  [Δt/Δx]
        # c_i · u · c_i / cs⁴  →  (dim, Q, Nx, Ny, Nz)
        c_3d = self._c.astype(self._dtype)  # (3, Q)
        # Expand c to broadcast with (Nx, Ny, Nz)
        ndim_spatial = u.ndim - 1
        for _ in range(ndim_spatial):
            c_3d = c_3d[..., None]

        # term = (c_i - u)/cs² + (c_i·u)·c_i/cs⁴
        # shape: (3, Q, Nx, Ny, Nz)
        term = ((c_3d - u[:, None, ...]) * self._inv_cs2
                + cu[None, ...] * c_3d * self._inv_cs4)

        # S_i = prefactor · w_i · Σ_d (term_d · F_d)
        # force: (3, Nx, Ny, Nz), term: (3, Q, Nx, Ny, Nz)
        # → dot over dim axis → (Q, Nx, Ny, Nz)
        S = xp.sum(term * force[:, None, ...], axis=0)  # (Q, Nx, Ny, Nz)

        # Multiply by w_i and prefactor
        w = self._w
        for _ in range(ndim_spatial):
            w = w[..., None]

        S *= prefactor * w  # [dimensionless]

        return S
