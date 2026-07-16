"""ALM regularization-kernel factory (G-beta0, patch_notes/alm_beta_kernel).

THE single source of truth for the ALM kernel family eta(r): the spreading
RawKernels, the sampling RawKernel, the numpy reference/fallback paths, the
radial-truncation renormalization and the smearing-correction deficit kernel
all derive their kernel-dependent pieces from ONE spec object, so the
physics-code unity principle holds by construction (01_design.md Part I).

Families:
  gaussian     eta = exp(-r^2/eps^2) / (pi^{3/2} eps^3), truncated at
               n_cut*eps + discrete renormalization (production default;
               the generated CUDA sources are BYTE-IDENTICAL to the
               pre-refactor hardcoded ones -> bit parity, G-beta0 gate).
  wendland     psi31(q) = (1-q)^4 (4q+1), q = r/R_s, exactly 0 beyond R_s.
               R_s = sqrt(15/2)*eps fixes the SECOND MOMENT equal to the
               gaussian's (3/2 eps^2) -> eps-equivalence (01_design.md #12).
               eta = 21/(2 pi R_s^3) * psi(q).
  winckelmans  high-order algebraic (Winckelmans-Leonard 1993),
               eta = 15/(8 pi eps^3) * (1 + r^2/eps^2)^{-7/2}; second
               moment (3/2) eps^2 -> sigma = eps trivially. Kept as an
               A/B control only (01_design.md #7b: algebraic tail).

Conventions shared by every family:
  - config 'eps' keeps its production meaning (the gaussian width policy,
    eps = max(0.25c, 2 dx)); each family maps eps to its own support.
  - support_factor k: cutoff/support radius = k * eps. This IS the old
    n_cut for the gaussian; compact families OVERRIDE it with their exact
    support (config gaussian_cutoff is ignored there).
  - discrete renormalization (divide by the truncated grid sum) stays for
    every family: compactness removes the truncation arbitrariness, not
    the sub-cell sum variation (01_design.md #4).
  - CUDA snippets are ASCII-only ([[feedback_cuda_kernel_ascii]]) and keep
    the fixed-order accumulation of the surrounding kernels.
"""

from typing import Optional

import numpy as np

_SQRT_7_5 = float(np.sqrt(7.5))            # Wendland R_s / eps (eps-equivalence)


class ALMKernelSpec:
    """One kernel family: host constants + numpy shapes + CUDA snippets.

    The CUDA snippet fields are exact source-text fragments substituted into
    the RawKernel templates (spreading.py / interpolation.py). For the
    gaussian they reproduce the pre-refactor sources byte-for-byte.
    """

    def __init__(self, name: str, support: Optional[float], norm_const: float,
                 cuda_w_sample: str, cuda_eta_iso: str, cuda_eta_radial: str,
                 cuda_eta_aniso: str):
        self.name = name
        self._support = support            # None -> config-driven (gaussian)
        #: eta = norm_const / (eps^3) * shape   (aniso: /(eps_c eps_t eps_r))
        self.norm_const = float(norm_const)
        self.cuda_w_sample = cuda_w_sample
        self.cuda_eta_iso = cuda_eta_iso
        self.cuda_eta_radial = cuda_eta_radial
        self.cuda_eta_aniso = cuda_eta_aniso

    # ---- host-side ------------------------------------------------------

    def support_factor(self, n_cut_cfg: float) -> float:
        """Cutoff/support radius in units of eps.

        gaussian: the config gaussian_cutoff (an arbitrary truncation);
        compact families: their EXACT support (config value ignored);
        winckelmans: config-driven like the gaussian (its algebraic tail
        has no natural cutoff - 01_design.md #7b documents the loss).
        """
        return float(n_cut_cfg) if self._support is None else self._support

    def np_shape(self, xp, d_sq, eps_sq):
        """Unnormalized kernel shape from squared distance / squared width.

        gaussian bit-contract: exactly ``xp.exp(-d_sq / eps_sq)`` (the
        pre-refactor expression at every call site).
        """
        if self.name == "gaussian":
            return xp.exp(-d_sq / eps_sq)
        if self.name == "wendland":
            q = xp.sqrt(d_sq / eps_sq) / _SQRT_7_5
            q = xp.minimum(q, 1.0)
            om = 1.0 - q
            return (om * om) * (om * om) * (4.0 * q + 1.0)
        if self.name == "winckelmans":
            b = 1.0 + d_sq / eps_sq
            return 1.0 / (b * b * b * xp.sqrt(b))
        raise AssertionError(self.name)

    def np_shape_arg(self, xp, arg):
        """Unnormalized shape from the ellipsoidal argument
        arg = sum_axis (d_axis/eps_axis)^2 (anisotropic path)."""
        if self.name == "gaussian":
            return xp.exp(-arg)
        if self.name == "wendland":
            q = xp.minimum(xp.sqrt(arg) / _SQRT_7_5, 1.0)
            om = 1.0 - q
            return (om * om) * (om * om) * (4.0 * q + 1.0)
        if self.name == "winckelmans":
            b = 1.0 + arg
            return 1.0 / (b * b * b * xp.sqrt(b))
        raise AssertionError(self.name)

    def np_norm(self, eps):
        """Continuous normalization 1/(width^3-scaled constant).

        gaussian keeps the exact pre-refactor expression
        ``1.0 / (np.pi ** 1.5 * eps ** 3)`` (bit contract at the CPU
        reference sites); other families use norm_const / eps^3 (the
        constant already absorbs their support scaling)."""
        if self.name == "gaussian":
            return 1.0 / (np.pi ** 1.5 * eps ** 3)
        return self.norm_const / eps ** 3

    def np_norm3(self, e1, e2, e3):
        """Anisotropic normalization (three widths)."""
        if self.name == "gaussian":
            return 1.0 / (np.pi ** 1.5 * e1 * e2 * e3)
        return self.norm_const / (e1 * e2 * e3)

    def deficit_K(self, xp, d, eps):
        """Smearing-correction deficit kernel K(d; eps) = 1 - Gamma_enc/Gamma
        of the kernel-smeared straight trailed vortex (01_design.md #8).
        Even in d; the semi-infinite (rotor-plane) case carries the same
        factor by the end-plane mirror symmetry of a straight core.

        gaussian: exp(-(d/eps)^2)  (Lamb-Oseen; bit-contract with the
                  pre-refactor _viscous_core_correction / influence_matrix).
        winckelmans: 1/(1 + (d/eps)^2)^2  (01_design.md #7b filament
                  integral; identical via the 2D-projection route:
                  eta2 = (2/pi)(s^2+1)^{-3} -> F = 1 - (1+x^2)^{-2}).
        wendland: closed form derived in 03_correction_derivation.md by
                  projecting eta3 along the filament (Gamma_enc of the 2D
                  core), verified against double quadrature to ~1e-12:
                    x = |d|/R_s (R_s = sqrt(7.5) eps), a = sqrt(1-x^2)
                    K = [a(945 - 2625a^2 + 2359a^4 - 663a^6)
                         - 105 x^6 (8+x^2) ln((1+a)/x)] / 16
                  K(0) = (945-2625+2359-663)/16 = 1 exactly; K = 0 for
                  x >= 1 (compact: the far field is EXACT, so the
                  correction acts only inside the support).
        """
        if self.name == "gaussian":
            return xp.exp(-(d / eps) ** 2)
        if self.name == "winckelmans":
            return 1.0 / (1.0 + (d / eps) ** 2) ** 2
        if self.name == "wendland":
            x = xp.minimum(xp.abs(d) / (_SQRT_7_5 * eps), 1.0)
            a = xp.sqrt(xp.maximum(1.0 - x * x, 0.0))
            a2 = a * a
            # x=0: L is finite under the clamp and multiplied by x^6 = 0
            L = xp.log((1.0 + a) / xp.maximum(x, 1e-300))
            poly = a * (945.0 + a2 * (-2625.0 + a2 * (2359.0 - 663.0 * a2)))
            return (poly - 105.0 * x ** 6 * (8.0 + x * x) * L) / 16.0
        raise NotImplementedError(
            f"deficit kernel for '{self.name}' not derived "
            "(patch_notes/alm_beta_kernel)")


# --- CUDA snippet definitions -------------------------------------------
# gaussian snippets reproduce the pre-refactor sources BYTE-FOR-BYTE
# (G-beta0 gate pins the generated-source sha256).

_GAUSSIAN = ALMKernelSpec(
    "gaussian", None, 1.0 / (np.pi ** 1.5),
    cuda_w_sample="        double w = exp(-d2 / e2);",
    cuda_eta_iso=(
        "    // Gaussian kernel: eta = (1 / (pi^{3/2} * eps^3)) * "
        "exp(-d_sq / eps_sq)\n"
        "    double norm = inv_pi32 / (eps * eps * eps);     // [1/lu^3]\n"
        "    double eta = norm * exp(-d_sq / eps_sq);         // [1/lu^3]"),
    cuda_eta_radial=(
        "    double norm = inv_pi32 / (eps * eps * eps);\n"
        "    double eta = norm * exp(-d_sq / eps_sq) * scales[m];"
        "   // renormalised"),
    cuda_eta_aniso=(
        "    double norm = inv_pi32 / (epc * ept * epr);\n"
        "    double eta = norm * exp(-arg);"),
)

# Wendland psi_{3,1}(q) = (1-q)^4 (4q+1), q = r/R_s, R_s = sqrt(7.5) eps.
# In-kernel q comes from the cutoff radius already in scope (r_cut = R_s for
# this family since support_factor IS the support): q = d/r_cut. norm input
# ('inv_pi32' parameter slot) carries 21/(2 pi (sqrt(7.5))^3) so that
# norm = <input>/(eps^3) = 21/(2 pi R_s^3).
_W_NORM = 21.0 / (2.0 * np.pi * _SQRT_7_5 ** 3)

_WENDLAND = ALMKernelSpec(
    "wendland", _SQRT_7_5, _W_NORM,
    cuda_w_sample=(
        "        double q = sqrt(d2 / r2);\n"
        "        double om = 1.0 - q;\n"
        "        double w = (om * om) * (om * om) * (4.0 * q + 1.0);"),
    cuda_eta_iso=(
        "    // Wendland psi31: eta = 21/(2 pi R_s^3) * (1-q)^4 (4q+1), "
        "q = d/R_s\n"
        "    double norm = inv_pi32 / (eps * eps * eps);\n"
        "    double q = sqrt(d_sq) / r_cut;\n"
        "    double om = 1.0 - q;\n"
        "    double eta = norm * (om * om) * (om * om) * (4.0 * q + 1.0);"),
    cuda_eta_radial=(
        "    double norm = inv_pi32 / (eps * eps * eps);\n"
        "    double q = sqrt(d_sq) / r_cut;\n"
        "    double om = 1.0 - q;\n"
        "    double eta = norm * (om * om) * (om * om) * (4.0 * q + 1.0)"
        " * scales[m];"),
    cuda_eta_aniso=(
        "    double norm = inv_pi32 / (epc * ept * epr);\n"
        "    double q = sqrt(arg) / n_cut;\n"
        "    double om = 1.0 - q;\n"
        "    double eta = norm * (om * om) * (om * om) * (4.0 * q + 1.0);"),
)

# Winckelmans-Leonard high-order algebraic: eta = 15/(8 pi eps^3) *
# (1 + d^2/eps^2)^{-7/2}. A/B control only (algebraic tail: 01_design #7b).
_WINCK = ALMKernelSpec(
    "winckelmans", None, 15.0 / (8.0 * np.pi),
    cuda_w_sample=(
        "        double bw = 1.0 + d2 / e2;\n"
        "        double w = 1.0 / (bw * bw * bw * sqrt(bw));"),
    cuda_eta_iso=(
        "    // Winckelmans high-order algebraic: eta = 15/(8 pi eps^3) *"
        " (1+d^2/eps^2)^{-7/2}\n"
        "    double norm = inv_pi32 / (eps * eps * eps);\n"
        "    double bw = 1.0 + d_sq / eps_sq;\n"
        "    double eta = norm / (bw * bw * bw * sqrt(bw));"),
    cuda_eta_radial=(
        "    double norm = inv_pi32 / (eps * eps * eps);\n"
        "    double bw = 1.0 + d_sq / eps_sq;\n"
        "    double eta = norm / (bw * bw * bw * sqrt(bw)) * scales[m];"),
    cuda_eta_aniso=(
        "    double norm = inv_pi32 / (epc * ept * epr);\n"
        "    double bw = 1.0 + arg;\n"
        "    double eta = norm / (bw * bw * bw * sqrt(bw));"),
)

_SPECS = {"gaussian": _GAUSSIAN, "wendland": _WENDLAND,
          "winckelmans": _WINCK}

#: module default: the production gaussian. Every entry point that takes an
#: optional kernel_spec falls back to this, so pre-refactor callers/gates
#: are untouched (and bit-identical).
GAUSSIAN = _GAUSSIAN


def get_alm_kernel(name: Optional[str]) -> ALMKernelSpec:
    """Resolve a kernel family by config name (None/'gaussian' -> gaussian)."""
    key = (name or "gaussian").lower()
    if key not in _SPECS:
        raise ValueError(
            f"actuator_line.kernel.type '{name}' unknown; "
            f"choose one of {sorted(_SPECS)}")
    return _SPECS[key]
