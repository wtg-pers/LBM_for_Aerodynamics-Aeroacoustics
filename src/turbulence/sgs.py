"""Sub-grid stress (SGS) eddy-viscosity formulas.

CPU-side reference implementations for the two supported models:

    Smagorinsky (1963):
        nu_t = (Cs * Delta)^2 * |S|

    WALE (Nicoud & Ducros 1999):
        nu_t = (Cw * Delta)^2 * (Sd:Sd)^{3/2} / [ (S:S)^{5/2} + (Sd:Sd)^{5/4} ]

        with the traceless squared-gradient operator
            Sd_ab = 0.5 (g^2_ab + g^2_ba) - (1/3) delta_ab g^2_cc,
            g_ab  = du_a/dx_b   (velocity-gradient tensor),
            g^2_ab = g_ac g_cb.

|S| is recovered from f^neq inside the cumulant kernel (see strain_rate.py),
so the Smagorinsky path is essentially free.

WALE requires the full velocity gradient, which is computed by central FD on
neighbour velocities. This is done with a separate gradient pass / inline in
the kernel using shared-memory neighbour reads.

These numpy reference paths are used only by unit tests; the kernel inlines
the same algebra in CUDA C.
"""

from __future__ import annotations

import numpy as np


# Default model constants (config-overridable).
DEFAULT_CS = 0.17    # Lilly 1967 (Kolmogorov K0=1.4 → Cs ≈ 0.173); the
                     # canonical isotropic-turbulence value rounded to 0.17.
                     # For wall-bounded / cylinder LES, prefer Cs=0.10
                     # (Frohlich 1998, Stiebler 2011) or use WALE / dynamic.
DEFAULT_CW = 0.5     # WALE (Nicoud & Ducros 1999)
DEFAULT_CS_MAX = 0.5     # Dynamic Smagorinsky clipping  Cs²_max = 0.25
DEFAULT_DYN_ALPHA_SQ = 3.0  # (Δ̂/Δ)² for tensor-product 3-point box test filter

VALID_MODELS = ("off", "smagorinsky", "wale", "dyn_smag")


def parse_sgs_config(config: dict) -> dict:
    """Validate and canonicalise the top-level config["sgs"] block.

    Schema:
        config["sgs"] = {
            "enabled": bool          (default: False)
            "model":   str           (default: "off"; off|smagorinsky|wale)
            "Cs":      float         (default: DEFAULT_CS = 0.17)
            "Cw":      float         (default: DEFAULT_CW = 0.5)
        }

    Returns a dict with all keys present and validated. If the block is
    missing, returns the SGS-disabled canonical dict. enabled=True with
    model="off" is treated as a misconfiguration (raises ValueError).
    """
    raw = config.get("sgs", {})
    if not isinstance(raw, dict):
        raise TypeError(f"config['sgs'] must be a dict, got {type(raw).__name__}")

    enabled = bool(raw.get("enabled", False))
    model = str(raw.get("model", "off")).lower()
    if model not in VALID_MODELS:
        raise ValueError(
            f"sgs.model must be one of {VALID_MODELS}, got '{model}'"
        )
    if enabled and model == "off":
        raise ValueError("sgs.enabled=True requires sgs.model in {smagorinsky, wale}")
    if not enabled:
        model = "off"

    Cs = float(raw.get("Cs", DEFAULT_CS))
    Cw = float(raw.get("Cw", DEFAULT_CW))
    Cs_max = float(raw.get("Cs_max", DEFAULT_CS_MAX))
    alpha_sq = float(raw.get("alpha_sq", DEFAULT_DYN_ALPHA_SQ))
    if Cs <= 0 or Cw <= 0:
        raise ValueError(f"sgs.Cs and sgs.Cw must be positive (got Cs={Cs}, Cw={Cw})")
    if Cs_max <= 0 or alpha_sq <= 0:
        raise ValueError(
            f"sgs.Cs_max and sgs.alpha_sq must be positive "
            f"(got Cs_max={Cs_max}, alpha_sq={alpha_sq})"
        )

    # wall_damp_cells (patch 12 follow-up): zero nu_t within N cells of a
    # SOLID body before it enters collision. The gradient/test-filter
    # stencils read the raw staircase velocity jump at wall-adjacent
    # cells; |S| ~ u/dx there is a discretisation artifact, not resolved
    # turbulence, and drives nu_t to O(100-1000) x nu_mol at tau -> 0.5
    # (f1g probe: alpha=10 wing, L3 shell p95 = 66x, max = 891x ->
    # hyper-viscous wall layer, lift collapse). 0 = off (default).
    wall_damp_cells = int(raw.get("wall_damp_cells", 0))
    if wall_damp_cells < 0:
        raise ValueError(f"sgs.wall_damp_cells must be >= 0, got {wall_damp_cells}")

    return {"enabled": enabled, "model": model,
            "Cs": Cs, "Cw": Cw,
            "Cs_max": Cs_max, "alpha_sq": alpha_sq,
            "wall_damp_cells": wall_damp_cells}


# ---- Smagorinsky --------------------------------------------------------

def nu_t_smagorinsky(s_norm: np.ndarray, dx: float, Cs: float = DEFAULT_CS) -> np.ndarray:
    """nu_t = (Cs * Delta)^2 * |S|."""
    return (Cs * dx) ** 2 * s_norm


# ---- WALE ---------------------------------------------------------------

def _g2_traceless_2d(gxx, gxy, gyx, gyy):
    """Return Sd_ab = 0.5 (g^2_ab + g^2_ba) - (1/3) delta_ab g^2_cc, 2D."""
    g2_xx = gxx * gxx + gxy * gyx
    g2_yy = gyx * gxy + gyy * gyy
    g2_xy = gxx * gxy + gxy * gyy
    g2_yx = gyx * gxx + gyy * gyx
    trace_g2 = g2_xx + g2_yy
    sd_xx = g2_xx - trace_g2 / 3.0
    sd_yy = g2_yy - trace_g2 / 3.0
    sd_xy = 0.5 * (g2_xy + g2_yx)
    return sd_xx, sd_yy, sd_xy


def _g2_traceless_3d(g):
    """g is (3, 3) tensor or array; returns Sd as 6 components symmetric tensor.

    Accepts numpy array of shape (3, 3, ...) or (3, 3) and returns
    (Sd_xx, Sd_yy, Sd_zz, Sd_xy, Sd_xz, Sd_yz).
    """
    # g2_ab = sum_c g_ac g_cb
    g2 = np.einsum('ac...,cb...->ab...', g, g)
    trace_g2 = g2[0, 0] + g2[1, 1] + g2[2, 2]
    sd_xx = g2[0, 0] - trace_g2 / 3.0
    sd_yy = g2[1, 1] - trace_g2 / 3.0
    sd_zz = g2[2, 2] - trace_g2 / 3.0
    sd_xy = 0.5 * (g2[0, 1] + g2[1, 0])
    sd_xz = 0.5 * (g2[0, 2] + g2[2, 0])
    sd_yz = 0.5 * (g2[1, 2] + g2[2, 1])
    return sd_xx, sd_yy, sd_zz, sd_xy, sd_xz, sd_yz


def nu_t_wale_2d(gxx, gxy, gyx, gyy, dx: float, Cw: float = DEFAULT_CW,
                 eps: float = 1e-12) -> np.ndarray:
    """WALE eddy viscosity in 2D from velocity gradient tensor.

    S_ab = 0.5 (g_ab + g_ba)
    Sd from _g2_traceless_2d
    nu_t = (Cw Delta)^2 (Sd:Sd)^{3/2} / [(S:S)^{5/2} + (Sd:Sd)^{5/4}]
    """
    s_xx = gxx
    s_yy = gyy
    s_xy = 0.5 * (gxy + gyx)
    s_dd = s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy   # S:S

    sd_xx, sd_yy, sd_xy = _g2_traceless_2d(gxx, gxy, gyx, gyy)
    sd_dd = sd_xx * sd_xx + sd_yy * sd_yy + 2.0 * sd_xy * sd_xy   # Sd:Sd

    num   = sd_dd ** 1.5
    den   = s_dd ** 2.5 + sd_dd ** 1.25 + eps
    return (Cw * dx) ** 2 * num / den


def nu_t_wale_3d(g: np.ndarray, dx: float, Cw: float = DEFAULT_CW,
                 eps: float = 1e-12) -> np.ndarray:
    """WALE eddy viscosity in 3D. `g` is (3, 3, ...) gradient tensor."""
    s = 0.5 * (g + np.swapaxes(g, 0, 1))
    s_dd = (s * s).sum(axis=(0, 1))   # S:S = sum S_ab^2

    sd_xx, sd_yy, sd_zz, sd_xy, sd_xz, sd_yz = _g2_traceless_3d(g)
    sd_dd = (sd_xx * sd_xx + sd_yy * sd_yy + sd_zz * sd_zz
             + 2.0 * (sd_xy * sd_xy + sd_xz * sd_xz + sd_yz * sd_yz))

    num   = sd_dd ** 1.5
    den   = s_dd ** 2.5 + sd_dd ** 1.25 + eps
    return (Cw * dx) ** 2 * num / den
