"""
C81 Airfoil Table Loader (Mach-indexed) for the Actuator Line Model

The classic "C81" airfoil-deck format (used by CAMRAD II, RCAS, CHARM, ...)
stores CL, CD, CM as functions of:
    - α    (angle of attack)   [degrees],  full ±180° range
    - Mach (free-stream Mach)  [dimensionless]

This is the natural fit for high-tip-speed rotors (e.g. HART-II, M_tip≈0.64):
in an Actuator-Line model the LBM grid never carries the blade-section
velocity, so the section relative Mach enters the loads *only* through this
table.  The LBM itself resolves only the low-Mach induced/wake field — which
is why a weakly-compressible solver remains valid for the flow even though
the blade-relative Mach is transonic.

C81 deck layout (this loader, validated against NACA23012.C81)
==============================================================
    line 0 : "<name>,<...>NLMC NALC NLMD NALD NLMM NALM"  (6×I2 packed counts)
    then three tables, each:
        Mach header :  NM machs   (wraps every 9 values)
        NA rows     :  α  +  NM coefficients  (wraps every 9 values)
    table order : CL, CD, CM.  A trailing free-text reference block is ignored.

Integration with the ALM (constant-chord rotors)
================================================
`actuator_line.py` queries loads through `polar_query(α_deg, Re)` — it is
Reynolds-indexed and never passes Mach.  For a *constant-chord* blade
(HART-II: c = 0.121 m everywhere) Re and Mach are in 1:1 correspondence:

    M = u_rel / a = (Re · ν / c) / a = [ν / (c · a)] · Re   ≡   k · Re

so `make_c81_polar_query()` returns a closure that recovers Mach from Re and
queries the table — *no change to actuator_line.py is required*.  For HART-II
k = ν/(c·a) = 1.50e-5/(0.121·341.6) ≈ 3.63e-7  (Re_tip≈1.77e6 → M≈0.64). ✓

The equivalence is scaling-invariant: ν/(c·a) computed from physical units
equals (ν_lu/c_lu)·(dx/dt)/a, so the closure is exact under standard LBM
convective/acoustic scaling (lattice Re = physical Re).  Variable-chord
rotors must instead pass Mach to the ALM directly (future ALM extension).

References:
    - NACA23012.C81 provenance: Lim, J. W., "Improved Performance Prediction
      for Bo105 Model Rotor in Cruise Using CFD," 37th European Rotorcraft
      Forum, Gallarate, Italy, Sep. 2011.

Author: LBM Development Team
Date: 2026-06
"""

import re
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import cupy as _cp                     # optional GPU backend
except Exception:                          # pragma: no cover — CPU-only env
    _cp = None


def _array_module(x):
    """Return cupy for a cupy array, else numpy (BEM GPU-resident dispatch)."""
    if _cp is not None and isinstance(x, _cp.ndarray):
        return _cp
    return np

try:
    import numpy.typing as npt
    _Arr = "npt.NDArray"
except Exception:  # pragma: no cover
    _Arr = "np.ndarray"

Scalar = Union[float, "np.ndarray"]


# =============================================================================
# §1. Low-level parser
# =============================================================================
def _read_data_floats(path: str) -> Tuple[str, list]:
    """Read a C81 deck: return (header_line, flat_list_of_float_tokens).

    Streaming stops at the first line whose leading token is not a float
    (the trailing free-text reference block), so any commentary after the
    CM table is safely ignored.
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError(f"Empty C81 file: {path}")
    header = lines[0]
    vals: list = []
    for ln in lines[1:]:
        toks = ln.split()
        if not toks:
            continue
        try:
            float(toks[0])
        except ValueError:
            break  # reached the reference / comment block
        for t in toks:
            vals.append(float(t))
    return header, vals


def _parse_header_counts(header: str) -> Tuple[str, Dict[str, int]]:
    """Parse the packed 6×I2 dimension specifier at the end of the header.

    Returns (airfoil_name, {NLMC, NALC, NLMD, NALD, NLMM, NALM}).
    """
    m = re.search(r"(\d{12})\s*$", header)
    if not m:
        raise ValueError(
            f"Cannot find 12-digit C81 dimension specifier in header: "
            f"{header!r}"
        )
    d = m.group(1)
    keys = ["NLMC", "NALC", "NLMD", "NALD", "NLMM", "NALM"]
    counts = {k: int(d[i:i + 2]) for k, i in zip(keys, range(0, 12, 2))}
    name = header[: m.start()].strip().rstrip(",").strip()
    return name, counts


def parse_c81(path: str) -> Dict[str, object]:
    """Parse a C81 deck into Mach/AoA grids and CL/CD/CM coefficient tables.

    Returns a dict with keys:
        name                         : airfoil name        [str]
        mach_cl, mach_cd, mach_cm    : Mach grids          [-]
        aoa_cl,  aoa_cd,  aoa_cm     : AoA grids           [deg]
        CL, CD, CM                   : arrays [n_aoa, n_mach]
    """
    header, vals = _read_data_floats(path)
    name, c = _parse_header_counts(header)

    it = iter(vals)

    def take(n: int) -> list:
        try:
            return [next(it) for _ in range(n)]
        except StopIteration:
            raise ValueError(
                f"C81 parse underflow in {path}: ran out of values while "
                f"reading a block of {n}. Header counts={c}"
            )

    def read_table(n_mach: int, n_aoa: int, label: str
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mach = np.asarray(take(n_mach), dtype=np.float64)
        aoa = np.empty(n_aoa, dtype=np.float64)
        data = np.empty((n_aoa, n_mach), dtype=np.float64)
        for i in range(n_aoa):
            row = take(1 + n_mach)
            aoa[i] = row[0]
            data[i, :] = row[1:]
        # --- integrity checks ---
        if not np.all(np.diff(aoa) > 0):
            raise ValueError(f"C81 {label}: AoA not strictly increasing")
        if not np.all(np.diff(mach) > 0):
            raise ValueError(f"C81 {label}: Mach not strictly increasing")
        return mach, aoa, data

    mach_cl, aoa_cl, CL = read_table(c["NLMC"], c["NALC"], "CL")
    mach_cd, aoa_cd, CD = read_table(c["NLMD"], c["NALD"], "CD")
    mach_cm, aoa_cm, CM = read_table(c["NLMM"], c["NALM"], "CM")

    return {
        "name": name,
        "mach_cl": mach_cl, "aoa_cl": aoa_cl, "CL": CL,
        "mach_cd": mach_cd, "aoa_cd": aoa_cd, "CD": CD,
        "mach_cm": mach_cm, "aoa_cm": aoa_cm, "CM": CM,
    }


# =============================================================================
# §2. Single coefficient table (α × Mach bilinear interpolation, clamped)
# =============================================================================
class _C81Table:
    """One coefficient table C(α, M) with clamped bilinear interpolation.

    Inputs outside the tabulated range are clamped to the grid bounds
    (no extrapolation): α is already full ±180°; Mach clamps to [M_min, M_max].
    """

    def __init__(self, label: str, aoa: np.ndarray, mach: np.ndarray,
                 data: np.ndarray) -> None:
        self.label = label
        self.aoa = aoa                    # [deg]
        self.mach = mach                  # [-]
        self.data = data                  # [n_aoa, n_mach]
        self._interp = RegularGridInterpolator(
            (aoa, mach), data, method="linear",
            bounds_error=False, fill_value=None,  # None+clamp ⇒ no real extrap
        )
        self._gpu = None                  # lazy (aoa,mach,data) cupy mirror

    def _bilinear(self, alpha_deg, mach, xp):
        """Clamped bilinear on the rectilinear (α,M) grid — xp∈{numpy,cupy}.

        searchsorted + explicit weights; mathematically identical to
        RegularGridInterpolator(method='linear') on this grid (verified
        <1e-12), so the GPU-resident BEM path (cupy inputs) matches the CPU
        reference. Non-uniform α/M grids handled via searchsorted (no index
        arithmetic). See patch_notes/hpc_upgrade/05_p2_bem_gpu_resident_design.md.
        """
        if xp is _cp:
            if self._gpu is None:         # upload grid once (static)
                self._gpu = (_cp.asarray(self.aoa), _cp.asarray(self.mach),
                             _cp.asarray(self.data))
            aoa, mach_g, data = self._gpu
        else:
            aoa, mach_g, data = self.aoa, self.mach, self.data
        a = xp.clip(alpha_deg, aoa[0], aoa[-1])
        M = xp.clip(mach, mach_g[0], mach_g[-1])
        na = aoa.shape[0]; nm = mach_g.shape[0]
        ia = xp.clip(xp.searchsorted(aoa, a) - 1, 0, na - 2)
        im = xp.clip(xp.searchsorted(mach_g, M) - 1, 0, nm - 2)
        a0 = aoa[ia]; a1 = aoa[ia + 1]
        m0 = mach_g[im]; m1 = mach_g[im + 1]
        ta = (a - a0) / (a1 - a0)
        tm = (M - m0) / (m1 - m0)
        c00 = data[ia, im]; c10 = data[ia + 1, im]
        c01 = data[ia, im + 1]; c11 = data[ia + 1, im + 1]
        return (c00 * (1 - ta) * (1 - tm) + c10 * ta * (1 - tm)
                + c01 * (1 - ta) * tm + c11 * ta * tm)

    def __call__(self, alpha_deg: Scalar, mach: Scalar) -> Scalar:
        xp = _array_module(alpha_deg) if _array_module(alpha_deg) is _cp \
            else _array_module(mach)
        if xp is _cp:
            # GPU-resident path: explicit bilinear (RegularGridInterpolator is
            # CPU/scipy-only). Bit-close to the numpy reference below.
            a = _cp.asarray(alpha_deg, dtype=_cp.float64)
            M = _cp.asarray(mach, dtype=_cp.float64)
            a_b, M_b = _cp.broadcast_arrays(a, M)
            return self._bilinear(a_b, M_b, _cp)
        # CPU reference path (unchanged — bit-identical to prior behavior).
        a = np.clip(np.asarray(alpha_deg, dtype=np.float64),
                    self.aoa[0], self.aoa[-1])
        M = np.clip(np.asarray(mach, dtype=np.float64),
                    self.mach[0], self.mach[-1])
        a_b, M_b = np.broadcast_arrays(a, M)
        pts = np.stack([a_b.ravel(), M_b.ravel()], axis=-1)
        out = self._interp(pts).reshape(a_b.shape)
        return float(out) if out.ndim == 0 else out


# =============================================================================
# §3. Public polar set
# =============================================================================
class C81PolarSet:
    """Mach-indexed airfoil polar (CL/CD/CM) loaded from a C81 deck.

    Example:
        >>> polar = C81PolarSet.from_file("NACA23012.C81")
        >>> cl, cd, cm = polar.query(alpha_deg=5.0, mach=0.4)
    """

    def __init__(self, parsed: Dict[str, object]) -> None:
        self.name = str(parsed["name"])
        self._cl = _C81Table("CL", parsed["aoa_cl"], parsed["mach_cl"],
                              parsed["CL"])
        self._cd = _C81Table("CD", parsed["aoa_cd"], parsed["mach_cd"],
                              parsed["CD"])
        self._cm = _C81Table("CM", parsed["aoa_cm"], parsed["mach_cm"],
                              parsed["CM"])

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "C81PolarSet":
        return cls(parse_c81(path))

    # -- queries -----------------------------------------------------------
    def query(self, alpha_deg: Scalar, mach: Scalar
              ) -> Tuple[Scalar, Scalar, Scalar]:
        """Return (CL, CD, CM) at angle(s) of attack [deg] and Mach [-]."""
        return (self._cl(alpha_deg, mach),
                self._cd(alpha_deg, mach),
                self._cm(alpha_deg, mach))

    def get_CL(self, alpha_deg: Scalar, mach: Scalar) -> Scalar:
        return self._cl(alpha_deg, mach)

    def get_CD(self, alpha_deg: Scalar, mach: Scalar) -> Scalar:
        return self._cd(alpha_deg, mach)

    def get_CM(self, alpha_deg: Scalar, mach: Scalar) -> Scalar:
        return self._cm(alpha_deg, mach)

    # -- grid info ---------------------------------------------------------
    @property
    def mach_range(self) -> Tuple[float, float]:
        return float(self._cl.mach[0]), float(self._cl.mach[-1])

    @property
    def aoa_range(self) -> Tuple[float, float]:
        return float(self._cl.aoa[0]), float(self._cl.aoa[-1])

    def info(self) -> str:
        return (
            f"C81PolarSet('{self.name}'): "
            f"CL[{self._cl.data.shape}] CD[{self._cd.data.shape}] "
            f"CM[{self._cm.data.shape}], "
            f"Mach∈[{self.mach_range[0]:.2f},{self.mach_range[1]:.2f}], "
            f"AoA∈[{self.aoa_range[0]:.1f},{self.aoa_range[1]:.1f}]deg"
        )


# =============================================================================
# §4. ALM integration — Re→Mach closure (constant-chord rotors)
# =============================================================================
def make_c81_polar_query(
    polar: C81PolarSet,
    chord: float,
    nu: float,
    sound_speed: float,
) -> Callable[..., Tuple[Scalar, Scalar]]:
    """Build a `polar_query(alpha_deg, Re) -> (CL, CD)` closure for the ALM.

    Recovers the local Mach from the Reynolds number assuming a constant
    chord: M = [ν / (c·a)] · Re.  Drop-in compatible with the 2-argument
    `polar_query` signature in `actuator_line.py` (no ALM code change).

    Args:
        polar:        C81PolarSet (Mach-indexed CL/CD/CM)
        chord:        blade chord  [m]   (constant — rectangular planform)
        nu:           kinematic viscosity  [m²/s]
        sound_speed:  free-stream speed of sound a  [m/s]

    Returns:
        polar_query(alpha_deg, Re) -> (CL, CD).
        The closure carries `.mach_per_Re` (= k) for inspection/validation.
    """
    if chord <= 0 or nu <= 0 or sound_speed <= 0:
        raise ValueError("chord, nu, sound_speed must be positive")
    k = nu / (chord * sound_speed)        # [s/m · ... ] → Mach per unit Re

    def polar_query(alpha_deg: Scalar, Re: Scalar) -> Tuple[Scalar, Scalar]:
        mach = k * np.asarray(Re, dtype=np.float64)
        cl = polar.get_CL(alpha_deg, mach)
        cd = polar.get_CD(alpha_deg, mach)
        return cl, cd

    polar_query.mach_per_Re = k           # type: ignore[attr-defined]
    polar_query.polar_set = polar         # type: ignore[attr-defined]
    # _C81Table broadcasts (α, M) arrays through one RegularGridInterpolator
    # call, so whole-blade batched lookups are exact (same interpolant/clamp).
    polar_query.supports_batch = True     # type: ignore[attr-defined]
    return polar_query


def make_c81_polar_query_mach(
    polar: C81PolarSet,
) -> Callable[..., Tuple[Scalar, Scalar]]:
    """Build a Mach-native `polar_query(alpha_deg, Re, mach) -> (CL, CD)` closure.

    For tapered / variable-chord rotors the constant-chord Re→Mach trick of
    `make_c81_polar_query()` is invalid (chord varies along span), so the ALM
    computes the section Mach directly and passes it here.  The `mach` argument
    is used for the (alpha, M) table lookup; `Re` is accepted for signature
    compatibility but not used for the Mach recovery.

    The ALM (`actuator_line.py`) detects the `mach` parameter via signature
    inspection and supplies it per element; decks/closures without a `mach`
    parameter keep the original Reynolds-only behavior.

    Args:
        polar:  C81PolarSet (Mach-indexed CL/CD/CM)

    Returns:
        polar_query(alpha_deg, Re, mach) -> (CL, CD).
    """
    def polar_query(
        alpha_deg: Scalar, Re: Scalar, mach: Scalar = 0.0,
    ) -> Tuple[Scalar, Scalar]:
        m = np.asarray(mach, dtype=np.float64)
        cl = polar.get_CL(alpha_deg, m)
        cd = polar.get_CD(alpha_deg, m)
        return cl, cd

    polar_query.polar_set = polar         # type: ignore[attr-defined]
    polar_query.wants_mach = True         # type: ignore[attr-defined]
    polar_query.supports_batch = True     # type: ignore[attr-defined] (see above)
    return polar_query


# =============================================================================
# §5. Standalone validation
# =============================================================================
def _validate(path: str) -> None:
    import numpy as np
    print("=" * 72)
    polar = C81PolarSet.from_file(path)
    print(polar.info())
    print(f"  Mach grid: {polar._cl.mach}")
    print(f"  AoA  grid: {polar._cl.aoa[0]:.1f} .. {polar._cl.aoa[-1]:.1f} "
          f"deg  ({polar._cl.aoa.size} pts)")

    # --- low-Mach airfoil sanity (NACA23012: cambered, α0≈-1.2°) ----------
    print("\n[Low-speed (M=0) airfoil sanity]")
    aa = np.linspace(-4, 8, 25)
    cl0 = polar.get_CL(aa, 0.0)
    # zero-lift angle
    i = np.where(np.diff(np.sign(cl0)))[0]
    if i.size:
        j = i[0]
        a0 = aa[j] - cl0[j] * (aa[j + 1] - aa[j]) / (cl0[j + 1] - cl0[j])
    else:
        a0 = float("nan")
    # lift-curve slope near α=0  [1/deg]
    cl_slope = (polar.get_CL(2.0, 0.0) - polar.get_CL(-2.0, 0.0)) / 4.0
    print(f"  CL(0°,M=0)      = {polar.get_CL(0.0, 0.0):+.4f}  (expect ~+0.10..0.14)")
    print(f"  α0 (zero-lift)  = {a0:+.3f} deg  (expect ~ -1.0..-1.3)")
    print(f"  CLα near 0      = {cl_slope:.4f} /deg = {cl_slope*np.degrees(1):.3f} /rad"
          f"  (expect ~0.10 /deg, 2π/rad≈0.110)")
    print(f"  CL_max (M=0)    = {polar.get_CL(np.linspace(0,20,201),0.0).max():+.3f}"
          f"  (expect ~1.2..1.6)")
    print(f"  CD_min (M=0)    = {polar.get_CD(np.linspace(-4,8,121),0.0).min():.5f}"
          f"  (expect ~0.006..0.010)")

    # --- compressibility trend (Prandtl–Glauert: CLα grows with Mach) -----
    print("\n[Compressibility trend — CLα(M) at α≈4°]")
    for M in [0.0, 0.3, 0.5, 0.6, 0.8]:
        sl = (polar.get_CL(5.0, M) - polar.get_CL(3.0, M)) / 2.0
        print(f"  M={M:.1f}:  CL(4°)={polar.get_CL(4.0,M):+.4f}   "
              f"CLα≈{sl:.4f}/deg   CD(4°)={polar.get_CD(4.0,M):.5f}")

    # --- HART-II Re→Mach closure check ------------------------------------
    print("\n[HART-II Re→Mach closure]")
    R, c, RPM = 2.0, 0.121, 1041.0
    nu, a = 1.50e-5, 341.6
    omega = RPM * 2 * np.pi / 60.0
    V_tip = omega * R
    Re_tip = V_tip * c / nu
    q = make_c81_polar_query(polar, chord=c, nu=nu, sound_speed=a)
    M_tip_direct = V_tip / a
    M_tip_via_Re = q.mach_per_Re * Re_tip
    print(f"  k = ν/(c·a)        = {q.mach_per_Re:.3e}  Mach/Re")
    print(f"  Re_tip             = {Re_tip:.3e}")
    print(f"  M_tip (V_tip/a)    = {M_tip_direct:.4f}")
    print(f"  M_tip (k·Re_tip)   = {M_tip_via_Re:.4f}   (must match ✓)")
    cl_tip, cd_tip = q(6.0, Re_tip)
    print(f"  q(α=6°, Re_tip) -> CL={cl_tip:+.4f}, CD={cd_tip:.5f}")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    default = ("aeromechanics_workshop/references/Z01_data/NACA23012.C81")
    _validate(sys.argv[1] if len(sys.argv) > 1 else default)
