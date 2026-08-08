"""Solenoidal initial-condition perturbation (seed-discrimination arm).

Purpose (patch_notes/surfel/57 sec. 4): the Re ladder found the stack's
solution Re-INVARIANT and half-attached because nothing seeds resolved
turbulence (clean inflow, thin span). This module plants a reproducible,
divergence-free velocity perturbation in the initial condition ONLY —
zero runtime contact, zero gate contact.

Construction: a finite sum of Fourier modes

    u'(x) = sigma_u * sqrt(2/M) * sum_m  eps_m  cos(k_m . x + phi_m)

with polarization eps_m PERPENDICULAR to k_m, so each mode is exactly
solenoidal (div u' = -(k.eps) sin(...) = 0 analytically). Everything is
a deterministic function of GLOBAL L0-lattice coordinates and the seed,
so every MLG level evaluates the SAME physical field in its own frame —
no interpolation, level-consistent by construction.

Registered approximations (patch 58): the amplitude envelope is a
cosine-tapered BOX (not a wall-distance band — spill into the body
interior lands on dead cells and is zeroed at the first advect);
k_z is quantized to the L0 span for z-wrap continuity (finer levels'
node-based span differs by one coarse cell — a transient seam the
first steps wash out); coarse levels alias sub-Nyquist modes (their
copy of the seed lives mostly in the excised region and is overwritten
by F2C within one coarse step).
"""

from __future__ import annotations

import numpy as np


def build_modes(seed: int, n_modes: int, lam_min: float, lam_max: float,
                span_z: float):
    """Deterministic mode set. Wavelengths in L0 lu; k_z quantized to
    the L0 span (2*pi*m/span_z) for wrap continuity."""
    rng = np.random.default_rng(int(seed))
    lam = rng.uniform(float(lam_min), float(lam_max), n_modes)
    kmag = 2.0 * np.pi / lam
    kdir = rng.normal(size=(n_modes, 3))
    kdir /= np.linalg.norm(kdir, axis=1, keepdims=True)
    k = kdir * kmag[:, None]
    # quantize k_z to the periodic span (nearest harmonic, sign kept)
    dkz = 2.0 * np.pi / float(span_z)
    k[:, 2] = np.round(k[:, 2] / dkz) * dkz
    # polarization: random vector projected off k, normalized
    eps = rng.normal(size=(n_modes, 3))
    kk = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-30)
    eps -= (eps * kk).sum(axis=1, keepdims=True) * kk
    eps /= np.maximum(np.linalg.norm(eps, axis=1, keepdims=True), 1e-30)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    return k, eps, phi


def evaluate(xp, cfg: dict, xg, yg, zg):
    """u'(x_global) on the given coordinate grids (broadcastable, L0 lu).

    cfg keys: sigma_u [lu vel], box_lu [x0,x1,y0,y1,z0,z1], taper_lu,
    lambda_lu [min,max], n_modes, seed, span_z_lu.
    Returns (3,)+shape float32 array in xp namespace.
    """
    k, eps, phi = build_modes(cfg['seed'], cfg['n_modes'],
                              cfg['lambda_lu'][0], cfg['lambda_lu'][1],
                              cfg['span_z_lu'])
    k = xp.asarray(k)
    eps = xp.asarray(eps)
    phi = xp.asarray(phi)
    shape = xp.broadcast_shapes(xg.shape, yg.shape, zg.shape)
    u = xp.zeros((3,) + shape, dtype=xp.float64)
    for m in range(k.shape[0]):
        ph = k[m, 0] * xg + k[m, 1] * yg + k[m, 2] * zg + phi[m]
        c = xp.cos(ph)
        for d in range(3):
            u[d] += eps[m, d] * c
    amp = float(cfg['sigma_u']) * np.sqrt(2.0 / k.shape[0])

    # cosine-tapered box envelope (x/y only; z is periodic — no taper)
    x0, x1, y0, y1, z0, z1 = [float(v) for v in cfg['box_lu']]
    t = float(cfg.get('taper_lu', 2.0))

    def _win(g, lo, hi):
        w = xp.ones_like(g, dtype=xp.float64)
        w = xp.where(g < lo, 0.0, w)
        w = xp.where(g > hi, 0.0, w)
        rise = (g - lo) / t
        fall = (hi - g) / t
        edge = xp.minimum(rise, fall).clip(0.0, 1.0)
        return w * 0.5 * (1.0 - xp.cos(np.pi * edge))

    env = _win(xg, x0, x1) * _win(yg, y0, y1)
    zin = xp.where((zg >= z0) & (zg <= z1), 1.0, 0.0)
    return (u * (amp * env * zin)[None]).astype(xp.float32)


def apply_to_level(xp, f, collision, pert_cfg: dict,
                   origin=(0.0, 0.0, 0.0), dx: float = 1.0,
                   rho0: float = 1.0, u0=(0.0, 0.0, 0.0)) -> int:
    """Overwrite f with eq(rho0, u0 + u') inside the perturbation box.

    Level-local wiring: cell centers sit at integer local indices; the
    global coordinate is origin + index*dx. Only the local sub-box
    intersecting cfg['box_lu'] is touched (one-off, low memory).
    Returns the number of perturbed cells (0 = box outside this level).
    """
    shape = f.shape[1:]
    x0, x1, y0, y1, z0, z1 = [float(v) for v in pert_cfg['box_lu']]
    lo = [x0, y0, z0]
    hi = [x1, y1, z1]
    sl = []
    for ax in range(3):
        a = int(np.floor((lo[ax] - origin[ax]) / dx)) - 1
        b = int(np.ceil((hi[ax] - origin[ax]) / dx)) + 2
        a = max(a, 0)
        b = min(b, shape[ax])
        if b <= a:
            return 0
        sl.append(slice(a, b))
    sl = tuple(sl)

    idx = [xp.arange(s.start, s.stop, dtype=xp.float64) for s in sl]
    xg = origin[0] + idx[0][:, None, None] * dx
    yg = origin[1] + idx[1][None, :, None] * dx
    zg = origin[2] + idx[2][None, None, :] * dx
    du = evaluate(xp, pert_cfg, xg, yg, zg)

    u0 = xp.asarray(u0, dtype=xp.float32)
    u_sub = u0[:, None, None, None] + du
    rho_sub = xp.full(du.shape[1:], rho0, dtype=xp.float32)
    f_sub = collision.compute_equilibrium(rho_sub, u_sub)
    f[(slice(None),) + sl] = f_sub.astype(f.dtype, copy=False)
    return int(np.prod(du.shape[1:]))
