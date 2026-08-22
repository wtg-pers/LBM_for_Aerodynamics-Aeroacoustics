"""Sustained numerical trip — forced transition strip (patch 66).

The IC seed (initial_perturbation, patch 58) is one-shot and washes out
convectively (patches 59/65): the forward boundary layer returns to a
quiet laminar-like state, separates at x/c ~ 0.53 and the form drag of
the separated rear half is the entire Cd excess (65). This module is
the seed's TIME-SUSTAINED extension — the numerical analog of a wind-
tunnel trip strip (Schlatter & Orlu-style forcing): a small, localized,
divergence-free body-force fluctuation applied EVERY substep in a band
near the leading edge, so the resolved boundary layer downstream keeps
turbulent content instead of relaminarizing.

NOT a transition model: no model equations — the disturbance is
physical forcing; SGS + the surfel wall law handle the rest.

Force field (modes shared with the seed's build_modes — eps _|_ k so
the forcing is solenoidal; k_z quantized to the span for wrap
continuity; everything deterministic from the seed integer):

    a(x, t) = A_eff * sqrt(2/M) * env(x,y)
              * sum_m eps_m cos(k_m . x - omega_m t + phi_m)
    df_i    = w_i * 3 * (c_i . a)        (first-order forcing, rho ~ 1)

Determinism contract (the MPI load-bearing part): a() is a function of
GLOBAL L0-lu coordinates and GLOBAL L0 time (t_l0 = level substep count
* 2^-level), so every level and every rank evaluates the same physical
field in its own frame — slab windows pass their wrap-resolved global
z rows, and ranks agree on shared/ghost rows with ZERO communication
(gate T3: 1 GPU == mpirun -n 2, f bitwise, trip ON).

Level scaling: amp_lu is the acceleration in L0 lattice units;
A_eff(level) = amp_lu * 2^-level (dt^2/dx unit scaling — the same
per-level rule as body_force). Coarse-level copies inside the refined
region are overwritten by F2C like the seed's (registered approximation
carried over, patch 58); box spill into the body lands on dead cells,
whose advect sources are exactly zero (dV = g = 0).

Config block (top level, like initial_perturbation):

    "trip_forcing": {
        "enabled": True,
        "amp_lu": 1e-3,           # acceleration, L0 lattice units
        "box_lu": [x0,x1, y0,y1], # global L0 lu; z is ALWAYS full span
        "taper_lu": 1.0,          # cosine edge on x/y (58's window)
        "lambda_lu": [1.5, 4.0],  # mode wavelengths, L0 lu
        "n_modes": 16,
        "seed": 20260822,
        "u_ref_lu": 0.0866025,    # convective scale for omega
        "omega_scale": 1.0,       # omega_m = scale*2*pi*u_ref/lambda_m
    }
"""

from __future__ import annotations

import numpy as np

from src.utilities.initial_perturbation import build_modes


class TripForcing:
    """Per-sim trip applier. Geometry/phase caches build LAZILY on the
    first apply() — replicated MPI build sims never advance, so they
    never pay the cache (the same reason kernel.Q is lazy, 64 sec. 13).
    """

    def __init__(self, xp, cfg: dict, xg, yg, zg, level: int,
                 lattice) -> None:
        self.xp = xp
        self.cfg = dict(cfg)
        # host copies of the 1-D GLOBAL L0-lu cell-center coordinates of
        # this sim's local axes (slab passes wrap-resolved z rows)
        self._xg = np.asarray(xg, dtype=np.float64)
        self._yg = np.asarray(yg, dtype=np.float64)
        self._zg = np.asarray(zg, dtype=np.float64)
        self.level = int(level)
        self._w = np.asarray(lattice.w.get() if hasattr(lattice.w, 'get')
                             else lattice.w, dtype=np.float64)
        c = np.asarray(lattice.c.get() if hasattr(lattice.c, 'get')
                       else lattice.c, dtype=np.float64)
        self._c = c.T if c.shape[0] == 3 else c        # -> (27, 3)
        self._built = False
        self._sl = None            # None after build = box outside sim

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._built = True
        xp, cfg = self.xp, self.cfg
        x0, x1, y0, y1 = [float(v) for v in cfg['box_lu']]
        ii = np.flatnonzero((self._xg >= x0) & (self._xg <= x1))
        jj = np.flatnonzero((self._yg >= y0) & (self._yg <= y1))
        if ii.size == 0 or jj.size == 0:
            return                                 # sim outside the strip
        sx = slice(int(ii[0]), int(ii[-1]) + 1)
        sy = slice(int(jj[0]), int(jj[-1]) + 1)
        self._sl = (sx, sy, slice(None))

        k, eps, phi = build_modes(
            cfg['seed'], cfg['n_modes'],
            cfg['lambda_lu'][0], cfg['lambda_lu'][1], cfg['span_z_lu'])
        lam = 2.0 * np.pi / np.linalg.norm(k, axis=1)
        omega = (float(cfg.get('omega_scale', 1.0)) * 2.0 * np.pi
                 * float(cfg['u_ref_lu']) / lam)   # rad per L0 step

        xg = self._xg[sx][:, None, None]
        yg = self._yg[sy][None, :, None]
        zg = self._zg[None, None, :]
        ph = (k[:, 0, None, None, None] * xg[None]
              + k[:, 1, None, None, None] * yg[None]
              + k[:, 2, None, None, None] * zg[None]
              + phi[:, None, None, None])
        self._ph = xp.asarray(ph)                  # (M,)+strip, f64
        self._omega = xp.asarray(omega)
        self._eps = xp.asarray(eps)                # (M, 3)

        # cosine-tapered x/y window (58's _win); z uniform — a trip
        # strip spans the wing. Amplitude folds the level force scaling.
        t = float(cfg.get('taper_lu', 1.0))

        def _win(g, lo, hi):
            edge = np.minimum((g - lo) / t, (hi - g) / t).clip(0.0, 1.0)
            w = 0.5 * (1.0 - np.cos(np.pi * edge))
            w[(g < lo) | (g > hi)] = 0.0
            return w

        env = (_win(self._xg[sx], x0, x1)[:, None, None]
               * _win(self._yg[sy], y0, y1)[None, :, None])
        amp = (float(cfg['amp_lu']) * np.sqrt(2.0 / k.shape[0])
               * 2.0 ** (-self.level))
        self._ampenv = xp.asarray(amp * env)       # (nsx, nsy, 1)
        # w_i * 3 * c_i, folded: df = wc3 @ a
        self._wc3 = xp.asarray(3.0 * self._w[:, None] * self._c)  # (27,3)

    # ------------------------------------------------------------------
    def apply(self, f_post, step: int) -> None:
        """Add the trip kick to the post-collision field (std layout).

        step = the sim's OWN substep counter (pre-advance value); global
        L0 time = step * 2^-level, so all levels/ranks agree on t.
        """
        if not self._built:
            self._build()
        if self._sl is None:
            return
        xp = self.xp
        t_l0 = float(step) * (2.0 ** (-self.level))
        cosv = xp.cos(self._ph - self._omega[:, None, None, None] * t_l0)
        a = xp.tensordot(self._eps, cosv, axes=([0], [0]))   # (3,)+strip
        a *= self._ampenv[None]
        df = xp.tensordot(self._wc3, a, axes=([1], [0]))     # (27,)+strip
        sub = f_post[(slice(None),) + self._sl]
        sub += df.astype(f_post.dtype, copy=False)
