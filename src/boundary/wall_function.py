"""Wall-function closures for under-resolved boundary layers.

Originally built for the WFB track (W0-W3d, closed and deleted 2026-08-13
— see patch_notes/wall_model/). The closures survived the track: the
surfel boundary is the live consumer (surfel_boundary.py WALL_LAWS,
src/kernels/surfel_d3q27.py mirrors the Musker expressions in-kernel).

Implemented closure: equilibrium wall function on the Musker (1979)
composite profile

    u+(y+) = 5.424 atan((2 y+ - 8.15) / 16.7)
             + 4.1693 ln(y+ + 10.6) - 0.8686 ln(y+^2 - 8.15 y+ + 86)
             - C0,        y+ = y u_tau / nu,   u+ = U_t / u_tau

with C0 anchored so u+(0) = 0 exactly (the literature's rounded -3.52
leaves u+(0) = -0.0098; du+/dy+(0) = 1.0003 already matches the viscous
sublayer). Asymptotes: u+ = y+ and the kappa = 0.41 log law.

Planned additions (same interface): Werner-Wengle power law,
non-equilibrium two-layer (Cabot/Wang), pressure-gradient extension
(PowerFLOW-style) — see patch_notes/wall_model/PLAN.md.

DIRECTION CONVENTION (W1 review): the tangent is the projection of the
LOCAL sampled velocity onto the wall plane, u_t = u - (u.n) n — never a
global inflow axis. Valid for separated / reversed / spanwise flow and
future moving bodies; Cd/Cl axis conventions live in postprocessing.

Host numpy f64, vectorized over links (sparse per-step pass).

Author: LBM Development Team
Date: 2026-08 (wall_model track W1; relocated at W2 wrap-up)
"""

from typing import Dict, Optional, Tuple, Type, Union

import numpy as np

# Additive constant anchored so u+(0) = 0 EXACTLY. Value: -3.512013...
_MUSKER_C0 = (5.424 * np.arctan(-8.15 / 16.7)
              + 4.1693 * np.log(10.6) - 0.8686 * np.log(86.0))


def musker_uplus(yplus: np.ndarray) -> np.ndarray:
    """Musker (1979) composite u+(y+), f64, y+ >= 0."""
    yp = np.asarray(yplus, dtype=np.float64)
    return (5.424 * np.arctan((2.0 * yp - 8.15) / 16.7)
            + 4.1693 * np.log(yp + 10.6)
            - 0.8686 * np.log(yp * yp - 8.15 * yp + 86.0)
            - _MUSKER_C0)


def solve_utau(
    u_t: np.ndarray,
    y: np.ndarray,
    nu: float,
    max_iter: int = 80,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert U_t = u_tau * u+(y u_tau / nu) for u_tau (vectorized).

    Damped fixed point  u_tau <- u_tau + 0.5 (U_t / u+(y+) - u_tau)  from
    the viscous-sublayer initial guess sqrt(nu U_t / y) (exact for
    y+ <= ~1). Links with U_t ~ 0 return u_tau = 0 (converged).

    Returns (u_tau, yplus, converged).
    """
    U = np.asarray(u_t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if np.any(yy <= 0.0):
        raise ValueError("wall distance y must be > 0")
    tiny = 1e-14
    live = U > tiny
    ut = np.sqrt(np.maximum(nu * U / yy, 0.0))
    ut = np.where(live, np.maximum(ut, tiny), 0.0)

    for _ in range(max_iter):
        yp = yy * ut / nu
        up = musker_uplus(np.maximum(yp, 0.0))
        ut_new = np.where(live & (up > tiny), U / np.maximum(up, tiny), 0.0)
        step = 0.5 * (ut_new - ut)
        ut_next = np.where(live, ut + step, 0.0)
        done = np.abs(step) <= tol * np.maximum(ut, tiny)
        ut = ut_next
        if bool(np.all(done | ~live)):
            break

    yp = yy * ut / nu
    up = musker_uplus(np.maximum(yp, 0.0))
    resid = np.abs(ut * up - U)
    converged = ~live | (resid <= 1e-8 * np.maximum(U, tiny))
    return ut, yp, converged


def tangential_decompose(
    u_vec: np.ndarray,
    normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(|u_t|, unit tangent t_hat) of u projected onto the wall plane.

    t_hat is the direction of the LOCAL tangential velocity (see module
    doc); zero tangential velocity returns t_hat = 0 (caller treats the
    link as no-shear).
    """
    u = np.asarray(u_vec, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    un = np.einsum('ij,ij->i', u, n)
    ut_vec = u - un[:, None] * n
    mag = np.linalg.norm(ut_vec, axis=1)
    that = np.where(mag[:, None] > 1e-14, ut_vec / np.maximum(
        mag, 1e-300)[:, None], 0.0)
    return mag, that


def wall_shear(
    u_tau: np.ndarray,
    t_hat: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """tau_w vector ON THE FLUID: magnitude rho u_tau^2, direction -t_hat
    (the wall drags the fluid against its local tangential motion).
    Force on the BODY is the negative."""
    mag = np.asarray(rho, dtype=np.float64) * np.asarray(
        u_tau, dtype=np.float64) ** 2
    return -mag[:, None] * np.asarray(t_hat, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════
# Pluggable wall-law interface (W2)
#
# Contract (03 review): a law maps (U_t, y, nu[, dp/ds]) to the friction
# velocity u_tau; tau_w = rho u_tau^2 downstream. `dpds` is reserved for
# the pressure-gradient extension (W6) — equilibrium laws ignore it.
#
# Each law is the HOST REFERENCE of the same-kernel_id branch in
# src/kernels/wfb_d3q27.py: expressions are written in the exact
# left-to-right order the CUDA branch evaluates them (--fmad=false), so
# the laminar law is bit-reproducible and Musker differs only by the
# atan/log ulp gap between libm and CUDA math (W2 gate tolerance).
# ═════════════════════════════════════════════════════════════════════

class ViscousLaw:
    """Laminar stress law tau_w = alpha rho nu U_t / y.

    alpha = 1 is the true viscous law: with nu_eff = nu the WFB virtual
    wall velocity is u_w = 0 (no-slip degeneracy, W1b gate [B]). alpha < 1
    deliberately weakens the drag — the analytic-slip discriminator of
    W1b gate [C] and the W2 accounting gate.
    """

    kernel_id = 0
    kernel_iters = 0

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    @property
    def kernel_p0(self) -> float:
        return self.alpha

    def utau(
        self,
        u_t: np.ndarray,
        y: Union[float, np.ndarray],
        nu: float,
        dpds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        U = np.asarray(u_t, dtype=np.float64)
        return np.sqrt(self.alpha * nu * U / y)


class MuskerLaw:
    """Musker (1979) equilibrium wall function (module top).

    utau() runs the damped fixed point of solve_utau with tol = 0 —
    i.e. the full max_iter sweep — because the CUDA branch iterates a
    fixed count per link (no batch-coupled early exit). Once an element
    is stationary further iterations are no-ops, so the two are
    equivalent; tol = 0 just removes the batch-dependent stop.
    """

    kernel_id = 1
    kernel_p0 = 0.0

    def __init__(self, max_iter: int = 80) -> None:
        self.max_iter = int(max_iter)

    @property
    def kernel_iters(self) -> int:
        return self.max_iter

    def utau(
        self,
        u_t: np.ndarray,
        y: Union[float, np.ndarray],
        nu: float,
        dpds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        U = np.asarray(u_t, dtype=np.float64)
        yy = np.broadcast_to(
            np.asarray(y, dtype=np.float64), U.shape).copy()
        ut, _, _ = solve_utau(U, yy, nu, max_iter=self.max_iter, tol=0.0)
        return ut


WallLaw = Union[ViscousLaw, MuskerLaw]

#: Config-name registry (internal_geometry.<type>.surfel.law — consumed by
#: surfel_boundary.py).
WALL_LAWS: Dict[str, Type] = {
    "musker": MuskerLaw,
    "laminar": ViscousLaw,
}
