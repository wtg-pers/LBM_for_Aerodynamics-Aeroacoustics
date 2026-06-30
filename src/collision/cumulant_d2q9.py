"""
D2Q9 Cumulant Collision with Galilean Correction + Bulk Viscosity

Implements the PALABOS/Geier-style cumulant collision on D2Q9:
    - bulk/shear trace-traceless decomposition  (like PALABOS D2Q9)
    - Galilean correction on 2nd-order diagonals  (like OpenLB D3Q27,
      reduced to 2D via Chapman-Enskog)
    - simple (1 − ω) relaxation for C_11, C_21, C_12, C_22
      (same as PALABOS D2Q9; no Galilean correction on these because
       truncated Maxwellian error is dominant on 2nd-order diagonals)

Algorithm per site:
    1. Forward: f_i → normalized central moments K̂_{mn} → cumulants C_{mn}
    2. Velocity gradients (C-E 1st order):
        Dxu = -ω₁/(4·cs²)·(C_20 - C_02) - ω_b/(4·cs²)·(C_20 + C_02 - 2·cs²)
        Dyv = +ω₁/(4·cs²)·(C_20 - C_02) - ω_b/(4·cs²)·(C_20 + C_02 - 2·cs²)
    3. Galilean corrections:
        Gal_xx = -1/cs² · (1 - ω₁/2) · u² · Dxu   (normalized)
        Gal_bulk_xx = -1/cs² · (1 - ω_b/2) · u² · Dxu
        (similarly for yy)
    4. Relaxation (trace/deviator decomposition):
        trace_post    = ω_b · trace_eq + (1-ω_b)·(C_20+C_02) + (Gal_bulk_xx + Gal_bulk_yy)
        deviator_post = (1-ω₁)·(C_20 - C_02) + (Gal_xx - Gal_yy)
        C*_20 = (trace_post + deviator_post) / 2
        C*_02 = (trace_post - deviator_post) / 2
    5. Simple relaxation: C*_11, C*_21, C*_12, C*_22
    6. Backward: cumulants → central moments → raw moments → f_i*

References:
    - Geier et al. (2015), Comp. & Math. with Appl. 70(4), 507-547
      (D3Q27 cumulant with Galilean correction)
    - PALABOS comprehensiveModelsTemplates2D.h (D2Q9 cumulant, no Gal)
    - Molla (2025), Results in Engineering 27, 106225 (D2Q9 cumulant)

Cross-validated against:
    - Our D3Q27 `cumulant.py`        (bulk + Galilean)
    - PALABOS D2Q9                   (bulk, no Galilean)
    - lbmpy symbolic derivation      (forward/backward identities)

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from src.collision.base import CollisionOperator


class CumulantCollisionD2Q9(CollisionOperator):
    """D2Q9 Cumulant with bulk/shear separation and Galilean correction.

    Attributes:
        omega_bulk: Bulk viscosity rate. If None, defaults to omega_shear
                    (= 1/tau) at each collide() call → pure shear mode.
        omega_3:    3rd-order cumulant relaxation (C_{21}, C_{12}).
                    Default 0.6 (Molla heuristic).
        omega_4:    4th-order cumulant relaxation (C_{22}).
                    Default 1.4 (Molla heuristic).

    Note:
        omega_bulk = None (default) yields identical behavior to the
        original Molla-style implementation. Setting omega_bulk != 1/τ
        enables independent bulk viscosity control (PALABOS-style).
    """

    def __init__(
        self,
        xp: 'ModuleType',
        lattice: 'Lattice',
        omega_bulk: Optional[float] = None,
        omega_3: float = 0.6,
        omega_4: float = 1.4,
    ) -> None:
        super().__init__(xp, lattice)

        if self.Q != 9 or self.dim != 2:
            raise ValueError(
                f"CumulantCollisionD2Q9 requires D2Q9 (got D{self.dim}Q{self.Q})."
            )

        self._dtype = lattice.dtype
        self._c = xp.asarray(lattice.c, dtype=self._dtype)
        self._w = xp.asarray(lattice.w, dtype=self._dtype)
        self._cs2: float = float(lattice.cs2)
        self._inv_cs2: float = 1.0 / self._cs2
        self._inv_cs4: float = 1.0 / (self._cs2 ** 2)

        self.omega_bulk: Optional[float] = (None if omega_bulk is None
                                            else float(omega_bulk))
        self.omega_3: float = float(omega_3)
        self.omega_4: float = float(omega_4)

        self._cx = xp.asarray(lattice.c[0], dtype=self._dtype)
        self._cy = xp.asarray(lattice.c[1], dtype=self._dtype)

    # =================================================================
    # CollisionOperator interface
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
        """Perform D2Q9 cumulant collision step in-place."""
        xp = self.xp
        ux = u[0]
        uy = u[1]

        cs2 = self._cs2
        inv_cs2 = self._inv_cs2
        w1 = 1.0 / tau                                          # shear
        w2 = self.omega_bulk if self.omega_bulk is not None else w1  # bulk
        w3 = self.omega_3
        w4 = self.omega_4

        # ── Step 1: normalized central moments K̂_{mn} ───────────────
        K20h = xp.zeros_like(rho)
        K02h = xp.zeros_like(rho)
        K11h = xp.zeros_like(rho)
        K21h = xp.zeros_like(rho)
        K12h = xp.zeros_like(rho)
        K22h = xp.zeros_like(rho)

        for q in range(9):
            fq = f_in[q]
            dx = self._cx[q] - ux
            dy = self._cy[q] - uy
            fdx = fq * dx
            fdy = fq * dy
            fdxdy = fdx * dy
            K20h += fdx * dx
            K02h += fdy * dy
            K11h += fdxdy
            K21h += fdxdy * dx
            K12h += fdxdy * dy
            K22h += fdxdy * dx * dy

        inv_rho = 1.0 / rho
        K20h *= inv_rho
        K02h *= inv_rho
        K11h *= inv_rho
        K21h *= inv_rho
        K12h *= inv_rho
        K22h *= inv_rho

        # ── Step 2: forward cumulant transform ───────────────────────
        # Orders ≤ 3 identity; C_22 non-trivial.
        C20 = K20h
        C02 = K02h
        C11 = K11h
        C21 = K21h
        C12 = K12h
        C22 = K22h - K20h * K02h - 2.0 * K11h * K11h

        # ── Step 3: velocity gradients (C-E 1st order, 2D) ──────────
        # Dxu = ∂_x u_x,  Dyv = ∂_y u_y
        # Derived from:
        #   C_20 - C_02       = -2·cs²/ω₁   · (∂_x u_x - ∂_y u_y)
        #   C_20 + C_02 - 2cs²= -2·cs²/ω_b  · (∂_x u_x + ∂_y u_y)
        q025_inv_cs2 = 0.25 * inv_cs2     # = 0.75 at cs²=1/3
        shear_dev = C20 - C02
        bulk_dev  = C20 + C02 - 2.0 * cs2
        Dxu = -q025_inv_cs2 * (w1 * shear_dev + w2 * bulk_dev)
        Dyv = +q025_inv_cs2 * w1 * shear_dev - q025_inv_cs2 * w2 * bulk_dev

        # ── Step 4: Galilean correction contributions (normalized) ──
        # In 3D (unnormalized): Gal_xx = -3·ρ·(1-ω/2)·u²·Dxu
        # In 2D (normalized):   Gal_xx = -inv_cs2·(1-ω/2)·u²·Dxu
        # (factor 3 ≡ 1/cs² in both; we drop the ρ since our cumulants
        #  are normalized by ρ throughout the 2D code)
        fac_shear = -inv_cs2 * (1.0 - 0.5 * w1)
        fac_bulk  = -inv_cs2 * (1.0 - 0.5 * w2)
        Gal_xx       = fac_shear * ux * ux * Dxu
        Gal_yy       = fac_shear * uy * uy * Dyv
        Gal_bulk_xx  = fac_bulk  * ux * ux * Dxu
        Gal_bulk_yy  = fac_bulk  * uy * uy * Dyv

        # ── Step 5a: trace/deviator relaxation of C_20, C_02 ────────
        # trace_eq = C_20^eq + C_02^eq = 2·cs² (normalized)
        # deviator_eq = 0
        trace_eq = 2.0 * cs2
        trace    = (w2 * trace_eq
                    + (1.0 - w2) * (C20 + C02)
                    + (Gal_bulk_xx + Gal_bulk_yy))
        deviator = ((1.0 - w1) * (C20 - C02) + (Gal_xx - Gal_yy))

        C20_s = 0.5 * (trace + deviator)
        C02_s = 0.5 * (trace - deviator)

        # ── Step 5b: simple relaxation for remaining cumulants ──────
        C11_s = (1.0 - w1) * C11            # shear rate
        C21_s = (1.0 - w3) * C21            # 3rd order
        C12_s = (1.0 - w3) * C12
        C22_s = (1.0 - w4) * C22            # 4th order

        # ── Step 5c: optional body-force half-step ──────────────────
        if force is not None:
            half_Fx_over_rho = 0.5 * force[0] * inv_rho
            half_Fy_over_rho = 0.5 * force[1] * inv_rho
        else:
            half_Fx_over_rho = 0.0
            half_Fy_over_rho = 0.0

        # ── Step 6: backward cumulant transform ──────────────────────
        K20s = C20_s
        K02s = C02_s
        K11s = C11_s
        K21s = C21_s
        K12s = C12_s
        K22s = C22_s + C20_s * C02_s + 2.0 * C11_s * C11_s
        K10s = half_Fx_over_rho
        K01s = half_Fy_over_rho

        # ── Step 7: central → raw moments (binomial) ────────────────
        mu10p = K10s + ux
        mu01p = K01s + uy
        mu20p = K20s + 2.0 * ux * K10s + ux * ux
        mu02p = K02s + 2.0 * uy * K01s + uy * uy
        mu11p = K11s + ux * K01s + uy * K10s + ux * uy
        mu21p = (K21s
                 + 2.0 * ux * K11s + uy * K20s
                 + ux * ux * K01s + 2.0 * ux * uy * K10s
                 + ux * ux * uy)
        mu12p = (K12s
                 + ux * K02s + 2.0 * uy * K11s
                 + 2.0 * ux * uy * K01s + uy * uy * K10s
                 + ux * uy * uy)
        mu22p = (K22s
                 + 2.0 * uy * K21s + 2.0 * ux * K12s
                 + uy * uy * K20s + ux * ux * K02s + 4.0 * ux * uy * K11s
                 + 2.0 * ux * uy * uy * K10s + 2.0 * ux * ux * uy * K01s
                 + ux * ux * uy * uy)

        # ── Step 8: scale by ρ ──────────────────────────────────────
        mu00 = rho
        mu10 = rho * mu10p
        mu01 = rho * mu01p
        mu20 = rho * mu20p
        mu02 = rho * mu02p
        mu11 = rho * mu11p
        mu21 = rho * mu21p
        mu12 = rho * mu12p
        mu22 = rho * mu22p

        # ── Step 9: inverse M → f* ──────────────────────────────────
        f_out[0] = mu00 - mu20 - mu02 + mu22
        f_out[1] = 0.5 * ( mu10 + mu20 - mu12 - mu22)
        f_out[2] = 0.5 * ( mu01 + mu02 - mu21 - mu22)
        f_out[3] = 0.5 * (-mu10 + mu20 + mu12 - mu22)
        f_out[4] = 0.5 * (-mu01 + mu02 + mu21 - mu22)
        f_out[5] = 0.25 * ( mu11 + mu21 + mu12 + mu22)
        f_out[6] = 0.25 * (-mu11 + mu21 - mu12 + mu22)
        f_out[7] = 0.25 * ( mu11 - mu21 - mu12 + mu22)
        f_out[8] = 0.25 * (-mu11 - mu21 + mu12 + mu22)

    def compute_equilibrium(
        self,
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Standard 2D Maxwellian equilibrium for initialization."""
        xp = self.xp
        usqr = xp.sum(u * u, axis=0)
        cu = xp.einsum('dq,d...->q...', self._c, u)
        w_bc = self._w.reshape((-1,) + (1,) * rho.ndim)
        return w_bc * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)

    def __repr__(self) -> str:
        omega_bulk_repr = ("None(=ω_shear)" if self.omega_bulk is None
                           else f"{self.omega_bulk}")
        return (f"CumulantCollisionD2Q9("
                f"ω_bulk={omega_bulk_repr}, "
                f"ω_3={self.omega_3}, ω_4={self.omega_4})")
