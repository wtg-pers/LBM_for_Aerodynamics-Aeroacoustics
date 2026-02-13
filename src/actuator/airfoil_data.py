"""
Airfoil Aerodynamic Polar Data for Actuator Line Model

This module manages airfoil aerodynamic coefficient data (CL, CD) as functions
of angle of attack (α) and Reynolds number (Re), required for BEM force
calculations in the Actuator Line model.

Physical Context:
================
In the BEM approach (Watanabe et al. Eq. 9-10), the lift and drag forces on
each blade element are computed as:

    F_L = 0.5 · ρ · u_rel² · c_a · Δr · CL(Re, α)   [N]     (Eq.9)
    F_D = 0.5 · ρ · u_rel² · c_a · Δr · CD(Re, α)   [N]     (Eq.10)

where CL and CD are functions of:
    - α  (angle of attack)  [degrees]
    - Re (Reynolds number) = u_rel · c_a / ν  [dimensionless]

Data Flow (two interoperable representations):
===============================================

    ┌──────────────────────┐      to_grid_format()      ┌────────────────────┐
    │  AirfoilDatabase     │ ──────────────────────────→ │ Grid Format        │
    │  (OOP, per-Re polar) │                             │ (Alpha, Re,        │
    │  - Viterna extrap.   │ ←────────────────────────── │  CL_grid, CD_grid) │
    │  - Cubic spline      │      from_grid()            │ BEMT-compatible    │
    │  - XFOIL/CSV I/O     │                             │ make_polar_query() │
    └──────────────────────┘                             └────────────────────┘
           ↑                                                      ↑
    load_xfoil() / load_csv()                        gen_airfoil_polar()
    create_naca0012_polar()                          load_airfoil_from_csv()

Compatibility with BEMT Lab:
    - make_polar_query()  : identical bilinear interpolation closure
    - _bounds()           : identical 1D bracket helper
    - gen_airfoil_polar() : wraps neuralfoil/aerosandbox (= gen_airfoil_db)
    - load_airfoil_from_csv() : pandas-based multi-Re CSV loader
    - Grid tuple (Alpha, Re, CL, CD) : bidirectional conversion

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Eq. 9-10)
    - Drela M., "XFOIL: An Analysis and Design System ...", 1989
    - Viterna L.A. & Corrigan R.D., "Fixed Pitch Rotor Performance ...", 1982

Author: LBM Development Team
Date: 2026-02
"""

import os
import csv
import warnings
from typing import (
    TYPE_CHECKING, Optional, List, Dict, Tuple, Union, Callable
)

import numpy as np
from scipy import interpolate as sp_interp

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# §1. Grid-Based Polar Query (BEMT-Compatible)
# =============================================================================
# Ported from: bemt_lab/models/bemt_model.py → _bounds(), make_polar_query()
# These functions are the PRIMARY interface used by the Actuator Line model
# at runtime: each marker particle calls query(alpha_deg, Re) every timestep.

def _bounds(vec: np.ndarray, q: float) -> Tuple[int, int, float]:
    """Find bracketing indices and interpolation weight for 1D sorted array

    Given a sorted 1D array `vec` and a query value `q`, returns indices
    (j1, j2) and weight t such that:
        value ≈ (1-t) · vec[j1] + t · vec[j2]

    Boundary handling: clamps to edges (t=0 at low end, t=1 at high end).

    Args:
        vec: Sorted 1D array (ascending)
        q: Query value (same units as vec)

    Returns:
        (j1, j2, t): Lower index, upper index, interpolation weight  [dimensionless]

    Note:
        Identical to bemt_lab/models/bemt_model.py::_bounds()
    """
    if q <= vec[0]:
        return 0, min(1, len(vec) - 1), 0.0
    if q >= vec[-1]:
        return max(len(vec) - 2, 0), len(vec) - 1, 1.0
    j2 = int(np.searchsorted(vec, q, side='right'))
    j1 = j2 - 1
    t = (q - vec[j1]) / (vec[j2] - vec[j1] + 1e-12)
    return j1, j2, float(t)


def make_polar_query(
    alpha_grid: 'npt.NDArray',
    re_grid: 'npt.NDArray',
    CL_grid: 'npt.NDArray',
    CD_grid: 'npt.NDArray'
) -> Callable[[float, float], Tuple[float, float]]:
    """Create bilinear interpolation query from 2D polar grids

    This is the CORE query function used by both BEMT and Actuator Line models.
    Returns a lightweight closure that performs O(1) bilinear lookup.

    Grid Format (BEMT convention):
        alpha_grid[i, j] = α for (Re index i, α index j)   [degrees]
        re_grid[i, j]    = Re for (Re index i, α index j)   [dimensionless]
        CL_grid[i, j]    = CL at (Re_i, α_j)                [dimensionless]
        CD_grid[i, j]    = CD at (Re_i, α_j)                [dimensionless]

    Interpolation Method:
        Bilinear in (α, Re) space — fast and sufficient for BEM,
        where α and Re vary smoothly along the blade span.

    Args:
        alpha_grid: 2D meshgrid of angles of attack  [degrees]
        re_grid:    2D meshgrid of Reynolds numbers   [dimensionless]
        CL_grid:    2D array of lift coefficients     [dimensionless]
        CD_grid:    2D array of drag coefficients     [dimensionless]

    Returns:
        query(alpha_deg, Re) → (CL, CD) closure

    Example:
        >>> packs = gen_airfoil_polar("naca0012", Re_target=1e5)
        >>> query = make_polar_query(*packs[0])
        >>> cl, cd = query(alpha_deg=8.0, Re=1.2e5)

    Note:
        Identical interface to bemt_lab/models/bemt_model.py::make_polar_query()
    """
    # Extract 1D axes from meshgrid
    a = np.asarray(alpha_grid[0, :], dtype=np.float64)  # [degrees]
    R = np.asarray(re_grid[:, 0], dtype=np.float64)     # [dimensionless]
    CL = np.asarray(CL_grid, dtype=np.float64)          # [dimensionless]
    CD = np.asarray(CD_grid, dtype=np.float64)           # [dimensionless]

    # Handle transposed grids (safety)
    if CL.shape == (a.size, R.size):
        CL, CD = CL.T, CD.T

    # Sort axes ascending
    ai = np.argsort(a);  a = a[ai];   CL = CL[:, ai]; CD = CD[:, ai]
    Ri = np.argsort(R);  R = R[Ri];   CL = CL[Ri, :]; CD = CD[Ri, :]

    def query(alpha_deg: float, Re: float) -> Tuple[float, float]:
        """Bilinear interpolation of (CL, CD) at given (α, Re)

        Args:
            alpha_deg: Angle of attack  [degrees]
            Re: Reynolds number  [dimensionless]

        Returns:
            (CL, CD)  [dimensionless, dimensionless]
        """
        # Wrap α to [-180, 180]  [degrees]
        a0 = (float(alpha_deg) + 180.0) % 360.0 - 180.0

        j1, j2, ta = _bounds(a, a0)
        k1, k2, tr = _bounds(R, float(Re))

        # Bilinear interpolation
        CL0 = (1 - ta) * CL[k1, j1] + ta * CL[k1, j2]
        CL1 = (1 - ta) * CL[k2, j1] + ta * CL[k2, j2]
        CD0 = (1 - ta) * CD[k1, j1] + ta * CD[k1, j2]
        CD1 = (1 - ta) * CD[k2, j1] + ta * CD[k2, j2]

        return float((1 - tr) * CL0 + tr * CL1), \
               float((1 - tr) * CD0 + tr * CD1)

    return query


# =============================================================================
# §2. Polar Generation (neuralfoil / aerosandbox)
# =============================================================================
# Ported from: bemt_lab/airfoils/database.py → gen_airfoil_db()
# Enhanced: cleaner docstring, explicit unit annotations, error messages.

def gen_airfoil_polar(
    airfoil_name: str = "naca0012",
    Re_target: float = 1e5,
    alphas: Optional['npt.NDArray'] = None,
    mode: str = "asb",
    ncrit: float = 9.0,
    coordinates: Optional['npt.NDArray'] = None,
    dat_path: Optional[str] = None,
) -> List[Tuple['npt.NDArray', 'npt.NDArray', 'npt.NDArray', 'npt.NDArray']]:
    """Generate airfoil polar data using neuralfoil / aerosandbox

    Creates a 2D (α, Re) grid covering Re_target/2 to Re_target*2,
    returning the standard BEMT pack format.

    Re Coverage Strategy:
        Re_low  = Re_target / 2   [dimensionless]
        Re_high = Re_target * 2   [dimensionless]
        Step    = 10^(order-1)    (e.g., Re=1e5 → step=1e4)

    Args:
        airfoil_name: Airfoil name for aerosandbox  (e.g., "naca0012")
        Re_target:    Target Reynolds number  [dimensionless]
        alphas:       Angle of attack array   [degrees]
                      Default: np.linspace(-180, 180, 361)
        mode:         Generation backend
                      "asb"  — aerosandbox built-in library
                      "user" — from coordinate array (requires `coordinates`)
                      "dat"  — from .dat file (requires `dat_path`)
        ncrit:        Critical N-factor for transition  [dimensionless]
        coordinates:  Airfoil coordinates for mode="user"  [(N,2) array]
        dat_path:     Path to .dat file for mode="dat"

    Returns:
        List containing one tuple: [(Alpha_grid, Re_grid, CL_grid, CD_grid)]
        All grids have shape (n_Re, n_alpha).

    Raises:
        ImportError: If neuralfoil or aerosandbox not installed

    Example:
        >>> packs = gen_airfoil_polar("naca0012", Re_target=1e5)
        >>> query = make_polar_query(*packs[0])
        >>> cl, cd = query(8.0, 1e5)

    Note:
        Equivalent to bemt_lab/airfoils/database.py::gen_airfoil_db()
    """
    if alphas is None:
        alphas = np.linspace(-180, 180, 361)  # [degrees]

    # Re range: half-decade around target
    Re_low = Re_target / 2                     # [dimensionless]
    Re_high = Re_target * 2                    # [dimensionless]
    order = int(np.floor(np.log10(Re_target)))
    order_magnitude = 10 ** (order - 1)        # step size

    Res = np.arange(Re_low, Re_high + order_magnitude, order_magnitude)
    Alpha, Re = np.meshgrid(alphas, Res)       # 2D meshgrid

    if mode == "asb":
        try:
            import aerosandbox as asb
        except ImportError:
            raise ImportError(
                "aerosandbox required for mode='asb'. "
                "Install: pip install aerosandbox"
            )
        af = asb.Airfoil(airfoil_name)
        aero = af.get_aero_from_neuralfoil(
            alpha=Alpha.ravel(),
            Re=Re.ravel(),
            mach=0,
            model_size="xxlarge"
        )

    elif mode == "user":
        try:
            import neuralfoil as nf
        except ImportError:
            raise ImportError(
                "neuralfoil required for mode='user'. "
                "Install: pip install neuralfoil"
            )
        if coordinates is None:
            raise ValueError("coordinates required for mode='user'")
        aero = nf.get_aero_from_coordinates(
            coordinates=coordinates,
            alpha=Alpha.ravel(),
            Re=Re.ravel(),
            model_size="xxlarge",
            n_crit=ncrit
        )

    elif mode == "dat":
        try:
            import neuralfoil as nf
        except ImportError:
            raise ImportError(
                "neuralfoil required for mode='dat'. "
                "Install: pip install neuralfoil"
            )
        if dat_path is None:
            raise ValueError("dat_path required for mode='dat'")
        aero = nf.get_aero_from_dat_file(
            filename=dat_path,
            alpha=Alpha.ravel(),
            Re=Re.ravel(),
            model_size="xlarge",
            n_crit=ncrit
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'asb', 'user', or 'dat'.")

    CL_grid = aero["CL"].reshape(Alpha.shape)  # [dimensionless]
    CD_grid = aero["CD"].reshape(Re.shape)      # [dimensionless]

    return [(Alpha, Re, CL_grid, CD_grid)]


# =============================================================================
# §3. CSV / File I/O (BEMT-Compatible)
# =============================================================================
# Ported from: bemt_lab/io/dataset.py → load_airfoil_from_csv()

def load_airfoil_from_csv(
    csv_path: str,
    alpha_col: str = 'AoA(deg)',
    Re_col: str = 'Re',
    CL_col: str = 'cl',
    CD_col: str = 'cd',
) -> List[Tuple['npt.NDArray', 'npt.NDArray', 'npt.NDArray', 'npt.NDArray']]:
    """Load multi-Re airfoil polar from CSV into BEMT grid format

    Reads a CSV file with columns (Re, AoA, CL, CD) and constructs
    the standard 2D grid pack.

    CSV Format:
        Re,AoA(deg),cl,cd
        50000,-10.0,-0.60,0.035
        50000, -8.0,-0.55,0.028
        ...
        100000,-10.0,-0.70,0.028
        ...

    Args:
        csv_path:  Path to CSV file
        alpha_col: Column name for angle of attack  [degrees]
        Re_col:    Column name for Reynolds number   [dimensionless]
        CL_col:    Column name for lift coefficient   [dimensionless]
        CD_col:    Column name for drag coefficient   [dimensionless]

    Returns:
        List with one tuple: [(Alpha_grid, Re_grid, CL_grid, CD_grid)]
        Compatible with make_polar_query().

    Note:
        Equivalent to bemt_lab/io/dataset.py::load_airfoil_from_csv()
        Uses pandas if available, falls back to csv module.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        Re_values = np.sort(df[Re_col].unique())
        Alpha_values = np.sort(df[alpha_col].unique())
        n_Re, n_A = len(Re_values), len(Alpha_values)

        CL = np.zeros((n_Re, n_A))  # [dimensionless]
        CD = np.zeros((n_Re, n_A))  # [dimensionless]

        mapR = {re: i for i, re in enumerate(Re_values)}
        mapA = {a: j for j, a in enumerate(Alpha_values)}

        for _, row in df.iterrows():
            i = mapR[row[Re_col]]
            j = mapA[row[alpha_col]]
            CL[i, j] = row[CL_col]
            CD[i, j] = row[CD_col]

        Alpha_full = np.tile(Alpha_values, (n_Re, 1))        # [degrees]
        Re_full = np.tile(Re_values[:, None], (1, n_A))       # [dimensionless]

        return [(Alpha_full, Re_full, CL, CD)]

    except ImportError:
        # Fallback: pure csv module
        Re_data: Dict[float, Dict[str, List[float]]] = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    Re = float(row[Re_col])
                    if Re not in Re_data:
                        Re_data[Re] = {'alpha': [], 'CL': [], 'CD': []}
                    Re_data[Re]['alpha'].append(float(row[alpha_col]))
                    Re_data[Re]['CL'].append(float(row[CL_col]))
                    Re_data[Re]['CD'].append(float(row[CD_col]))
                except (ValueError, KeyError):
                    continue

        # Build grid from parsed data
        Re_values = np.array(sorted(Re_data.keys()))
        # Assume all Re share the same alpha set (sorted)
        first_Re = Re_values[0]
        Alpha_values = np.array(sorted(Re_data[first_Re]['alpha']))
        n_Re, n_A = len(Re_values), len(Alpha_values)

        CL = np.zeros((n_Re, n_A))
        CD = np.zeros((n_Re, n_A))
        for i, Re in enumerate(Re_values):
            sort_idx = np.argsort(Re_data[Re]['alpha'])
            CL[i, :] = np.array(Re_data[Re]['CL'])[sort_idx]
            CD[i, :] = np.array(Re_data[Re]['CD'])[sort_idx]

        Alpha_full = np.tile(Alpha_values, (n_Re, 1))
        Re_full = np.tile(Re_values[:, None], (1, n_A))

        return [(Alpha_full, Re_full, CL, CD)]


def load_airfoil_sections_from_csv(
    base_dir: str,
    n_sections: int,
    filename_fmt: str = "{k}section.csv",
) -> List[Tuple['npt.NDArray', 'npt.NDArray', 'npt.NDArray', 'npt.NDArray']]:
    """Load multiple airfoil section polars from numbered CSV files

    Args:
        base_dir:     Directory containing CSV files
        n_sections:   Number of sections to load
        filename_fmt: Filename format with {k} placeholder (1-indexed)

    Returns:
        List of (Alpha, Re, CL, CD) packs, one per section

    Note:
        Equivalent to bemt_lab/io/dataset.py::load_airfoil_sections_from_csv()
    """
    out = []
    for k in range(1, n_sections + 1):
        path = os.path.join(base_dir, filename_fmt.format(k=k))
        out += load_airfoil_from_csv(path)
    return out


# =============================================================================
# §4. Single Reynolds Number Polar (OOP)
# =============================================================================

class AirfoilPolarData:
    """Aerodynamic polar data for a single Reynolds number

    Stores CL(α) and CD(α) for one Re condition, with cubic spline
    interpolation and optional Viterna extrapolation to ±180°.

    Physical Units:
        α  : angle of attack       [degrees]
        CL : lift coefficient       [dimensionless]
        CD : drag coefficient       [dimensionless]
        CM : moment coefficient     [dimensionless]  (optional)
        Re : Reynolds number        [dimensionless]

    Example:
        >>> polar = AirfoilPolarData(
        ...     alpha=np.linspace(-10, 20, 31),
        ...     CL=CL_array, CD=CD_array, Re=1e5, name="NREL-S826"
        ... )
        >>> cl, cd = polar.get_coefficients(alpha=5.0)
    """

    def __init__(
        self,
        alpha: 'npt.NDArray',
        CL: 'npt.NDArray',
        CD: 'npt.NDArray',
        Re: float,
        CM: Optional['npt.NDArray'] = None,
        name: str = "Unknown",
        extrapolate: bool = True,
        alpha_stall_pos: Optional[float] = None,
        alpha_stall_neg: Optional[float] = None,
    ) -> None:
        """Initialize polar data for a single Reynolds number

        Args:
            alpha:           Angle of attack array    [degrees]
            CL:              Lift coefficient array    [dimensionless]
            CD:              Drag coefficient array    [dimensionless]
            Re:              Reynolds number           [dimensionless]
            CM:              Moment coeff. (optional)  [dimensionless]
            name:            Airfoil name (e.g., "NREL-S826")
            extrapolate:     If True, apply Viterna extrapolation to ±180°
            alpha_stall_pos: Positive stall angle      [degrees]
                             (auto-detected from CL_max if None)
            alpha_stall_neg: Negative stall angle      [degrees]
                             (auto-detected from CL_min if None)
        """
        # Store raw data
        self.alpha_raw = np.asarray(alpha, dtype=np.float64)  # [degrees]
        self.CL_raw = np.asarray(CL, dtype=np.float64)       # [dimensionless]
        self.CD_raw = np.asarray(CD, dtype=np.float64)        # [dimensionless]
        self.CM_raw = (
            np.asarray(CM, dtype=np.float64) if CM is not None else None
        )
        self.Re = float(Re)   # [dimensionless]
        self.name = name

        if len(self.alpha_raw) != len(self.CL_raw) or \
           len(self.alpha_raw) != len(self.CD_raw):
            raise ValueError(
                f"Array length mismatch: alpha({len(self.alpha_raw)}), "
                f"CL({len(self.CL_raw)}), CD({len(self.CD_raw)})"
            )

        # Sort by alpha ascending
        sort_idx = np.argsort(self.alpha_raw)
        self.alpha_raw = self.alpha_raw[sort_idx]
        self.CL_raw = self.CL_raw[sort_idx]
        self.CD_raw = self.CD_raw[sort_idx]
        if self.CM_raw is not None:
            self.CM_raw = self.CM_raw[sort_idx]

        # Detect stall angles  [degrees]
        self.alpha_stall_pos = (
            alpha_stall_pos if alpha_stall_pos is not None
            else self._detect_stall_positive()
        )
        self.alpha_stall_neg = (
            alpha_stall_neg if alpha_stall_neg is not None
            else self._detect_stall_negative()
        )

        # Key aerodynamic parameters
        self.CL_max = float(np.max(self.CL_raw))    # [dimensionless]
        self.CL_min = float(np.min(self.CL_raw))    # [dimensionless]
        self.CD_min = float(np.min(self.CD_raw))     # [dimensionless]
        self.CL_alpha = self._compute_lift_slope()   # [1/degree]

        # Viterna extrapolation to ±180° (essential for startup / yaw)
        if extrapolate:
            self._viterna_extrapolation()
        else:
            self.alpha = self.alpha_raw.copy()
            self.CL = self.CL_raw.copy()
            self.CD = self.CD_raw.copy()
            self.CM = (
                self.CM_raw.copy() if self.CM_raw is not None else None
            )

        # Build cubic spline interpolators
        self._build_interpolators()

    # ─────────────────────────────────────────────────────────────────────
    # §4.1 Core API
    # ─────────────────────────────────────────────────────────────────────

    def get_coefficients(
        self,
        alpha: Union[float, 'npt.NDArray']
    ) -> Tuple[Union[float, 'npt.NDArray'], Union[float, 'npt.NDArray']]:
        """Get CL and CD at given angle(s) of attack

        Args:
            alpha: Angle of attack  [degrees]  (scalar or array)

        Returns:
            (CL, CD)  [dimensionless, dimensionless]
        """
        alpha = np.asarray(alpha, dtype=np.float64)
        cl = self._interp_CL(alpha)
        cd = self._interp_CD(alpha)
        cd = np.maximum(cd, 1e-6)  # CD > 0 (physical constraint)

        if cl.ndim == 0:
            return float(cl), float(cd)
        return cl, cd

    def get_CL(self, alpha: Union[float, 'npt.NDArray']
               ) -> Union[float, 'npt.NDArray']:
        """Get lift coefficient only  [dimensionless]"""
        cl, _ = self.get_coefficients(alpha)
        return cl

    def get_CD(self, alpha: Union[float, 'npt.NDArray']
               ) -> Union[float, 'npt.NDArray']:
        """Get drag coefficient only  [dimensionless]"""
        _, cd = self.get_coefficients(alpha)
        return cd

    def get_CM(self, alpha: Union[float, 'npt.NDArray']
               ) -> Optional[Union[float, 'npt.NDArray']]:
        """Get moment coefficient  [dimensionless], or None"""
        if self._interp_CM is None:
            return None
        alpha = np.asarray(alpha, dtype=np.float64)
        cm = self._interp_CM(alpha)
        return float(cm) if cm.ndim == 0 else cm

    # ─────────────────────────────────────────────────────────────────────
    # §4.2 Stall Detection & Lift Slope
    # ─────────────────────────────────────────────────────────────────────

    def _detect_stall_positive(self) -> float:
        """Detect positive stall angle from CL_max position  [degrees]"""
        return float(self.alpha_raw[np.argmax(self.CL_raw)])

    def _detect_stall_negative(self) -> float:
        """Detect negative stall angle from CL_min position  [degrees]"""
        return float(self.alpha_raw[np.argmin(self.CL_raw)])

    def _compute_lift_slope(self) -> float:
        """Compute dCL/dα near α = 0  [1/degree]

        Thin airfoil theory predicts: 2π/rad ≈ 0.1097/deg
        """
        mask = (self.alpha_raw >= -5.0) & (self.alpha_raw <= 5.0)
        if np.sum(mask) < 2:
            return 2.0 * np.pi / 180.0  # thin airfoil default  [1/degree]
        coeffs = np.polyfit(self.alpha_raw[mask], self.CL_raw[mask], 1)
        return float(coeffs[0])  # [1/degree]

    # ─────────────────────────────────────────────────────────────────────
    # §4.3 Viterna-Corrigan Extrapolation
    # ─────────────────────────────────────────────────────────────────────

    def _viterna_extrapolation(self, AR: float = 10.0) -> None:
        """Apply Viterna-Corrigan extrapolation to ±180°

        Beyond stall, airfoil behaves increasingly like a flat plate:
            CL = A1 · sin(2α) + A2 · cos²(α) / sin(α)   [dimensionless]
            CD = B1 · sin²(α) + B2 · cos(α)              [dimensionless]

        Coefficients are calibrated to match CL/CD at the stall angle,
        ensuring continuity across the stall transition.

        Args:
            AR: Aspect ratio for CD_max estimation  [dimensionless]
                CD_max = 1.11 + 0.018·AR (flat plate normal to flow)
        """
        CD_max = 1.11 + 0.018 * AR  # [dimensionless]

        # ── Positive side: stall → 180° ──
        alpha_s = np.radians(self.alpha_stall_pos)  # [radians]
        CL_s = float(np.interp(
            self.alpha_stall_pos, self.alpha_raw, self.CL_raw
        ))
        CD_s = float(np.interp(
            self.alpha_stall_pos, self.alpha_raw, self.CD_raw
        ))

        sa, ca = np.sin(alpha_s), np.cos(alpha_s)
        B1_pos = CD_max
        B2_pos = (CD_s - CD_max * sa**2) / ca if abs(ca) > 1e-10 else 0.0
        A1_pos = (
            (CL_s - CD_max / 2.0 * np.sin(2 * alpha_s))
            * sa / (ca**2)
        ) if abs(ca) > 1e-10 else CD_max / 2.0
        A2_pos = (
            (CL_s - A1_pos * np.sin(2 * alpha_s))
            * sa / (ca**2)
        ) if abs(ca) > 1e-10 else 0.0

        # ── Negative side: -180° → stall ──
        alpha_s_neg = np.radians(self.alpha_stall_neg)
        CL_s_neg = float(np.interp(
            self.alpha_stall_neg, self.alpha_raw, self.CL_raw
        ))
        CD_s_neg = float(np.interp(
            self.alpha_stall_neg, self.alpha_raw, self.CD_raw
        ))

        sa_neg = np.sin(abs(alpha_s_neg))
        ca_neg = np.cos(alpha_s_neg)
        B1_neg = CD_max
        B2_neg = (
            (CD_s_neg - CD_max * sa_neg**2) / ca_neg
        ) if abs(ca_neg) > 1e-10 else 0.0
        A1_neg = (
            (CL_s_neg + CD_max / 2.0 * np.sin(2 * alpha_s_neg))
            * sa_neg / (ca_neg**2)
        ) if abs(ca_neg) > 1e-10 else CD_max / 2.0
        A2_neg = (
            (CL_s_neg - A1_neg * np.sin(2 * alpha_s_neg))
            * sa_neg / (ca_neg**2)
        ) if abs(ca_neg) > 1e-10 else 0.0

        # ── Build extended arrays ──
        mask_original = (
            (self.alpha_raw >= self.alpha_stall_neg) &
            (self.alpha_raw <= self.alpha_stall_pos)
        )

        # Post-stall positive region
        alpha_ext_pos = np.arange(
            self.alpha_stall_pos + 1.0, 180.1, 1.0
        )  # [degrees]
        a_rad_p = np.radians(alpha_ext_pos)
        CL_ext_pos = np.where(
            np.abs(np.sin(a_rad_p)) > 0.05,
            A1_pos * np.sin(2 * a_rad_p)
            + A2_pos * np.cos(a_rad_p)**2 / np.sin(a_rad_p),
            A1_pos * np.sin(2 * a_rad_p)  # drop singular term near 0°/180°
        )
        CD_ext_pos = np.maximum(
            B1_pos * np.sin(a_rad_p)**2
            + B2_pos * np.cos(a_rad_p),
            1e-4
        )

        # Post-stall negative region
        alpha_ext_neg = np.arange(
            -180.0, self.alpha_stall_neg, 1.0
        )  # [degrees]
        a_rad_n = np.radians(alpha_ext_neg)
        CL_ext_neg = np.where(
            np.abs(np.sin(a_rad_n)) > 0.05,
            A1_neg * np.sin(2 * a_rad_n)
            + A2_neg * np.cos(a_rad_n)**2 / np.sin(a_rad_n),
            A1_neg * np.sin(2 * a_rad_n)  # drop singular term near 0°/±180°
        )
        CD_ext_neg = np.maximum(
            B1_neg * np.sin(a_rad_n)**2
            + B2_neg * np.cos(a_rad_n),
            1e-4
        )

        # Concatenate: [-180° … stall⁻] + [stall⁻ … stall⁺] + [stall⁺ … 180°]
        self.alpha = np.concatenate([
            alpha_ext_neg, self.alpha_raw[mask_original], alpha_ext_pos
        ])  # [degrees]
        self.CL = np.concatenate([
            CL_ext_neg, self.CL_raw[mask_original], CL_ext_pos
        ])  # [dimensionless]
        self.CD = np.concatenate([
            CD_ext_neg, self.CD_raw[mask_original], CD_ext_pos
        ])  # [dimensionless]
        self.CD = np.maximum(self.CD, 1e-4)  # enforce CD > 0

        # CM: nearest-neighbor extrapolation (no Viterna model for moment)
        if self.CM_raw is not None:
            self.CM = np.interp(self.alpha, self.alpha_raw, self.CM_raw)
        else:
            self.CM = None

    # ─────────────────────────────────────────────────────────────────────
    # §4.4 Interpolator Construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_interpolators(self) -> None:
        """Build cubic spline interpolators for CL(α), CD(α), CM(α)"""
        # Remove duplicate alpha values (can occur at stall joins)
        _, unique_idx = np.unique(self.alpha, return_index=True)
        unique_idx = np.sort(unique_idx)

        alpha_u = self.alpha[unique_idx]
        CL_u = self.CL[unique_idx]
        CD_u = self.CD[unique_idx]

        self._interp_CL = sp_interp.interp1d(
            alpha_u, CL_u, kind='cubic',
            bounds_error=False, fill_value=(CL_u[0], CL_u[-1])
        )
        self._interp_CD = sp_interp.interp1d(
            alpha_u, CD_u, kind='cubic',
            bounds_error=False, fill_value=(CD_u[0], CD_u[-1])
        )

        if self.CM is not None:
            CM_u = self.CM[unique_idx]
            self._interp_CM = sp_interp.interp1d(
                alpha_u, CM_u, kind='cubic',
                bounds_error=False, fill_value=(CM_u[0], CM_u[-1])
            )
        else:
            self._interp_CM = None

    # ─────────────────────────────────────────────────────────────────────
    # §4.5 Info / Display
    # ─────────────────────────────────────────────────────────────────────

    def get_info(self) -> str:
        """Return human-readable summary"""
        ld_max = self.CL_max / max(self.CD_min, 1e-6)
        return (
            f"AirfoilPolarData: {self.name}\n"
            f"  Re = {self.Re:.0f}  [dimensionless]\n"
            f"  α range: [{self.alpha[0]:.1f}°, {self.alpha[-1]:.1f}°]  "
            f"({len(self.alpha)} points)\n"
            f"  CL_max = {self.CL_max:.4f} at α ≈ "
            f"{self.alpha_stall_pos:.1f}°\n"
            f"  CD_min = {self.CD_min:.6f}\n"
            f"  dCL/dα = {self.CL_alpha:.4f} [1/deg]  "
            f"(thin airfoil: {2*np.pi/180:.4f})\n"
            f"  L/D_max = {ld_max:.1f}"
        )

    def __repr__(self) -> str:
        return (
            f"AirfoilPolarData('{self.name}', Re={self.Re:.0f}, "
            f"α=[{self.alpha[0]:.0f}°..{self.alpha[-1]:.0f}°])"
        )


# =============================================================================
# §5. Multi-Reynolds Number Database (OOP)
# =============================================================================

class AirfoilDatabase:
    """Collection of airfoil polar data across multiple Reynolds numbers

    Provides two query modes:
        1. OOP:  db.get_coefficients(alpha, Re)  — log-linear Re interpolation
        2. BEMT: query = db.to_query()            — bilinear grid closure

    And bidirectional conversion with BEMT grid format:
        db.to_grid_format()  → (Alpha, Re, CL, CD)
        AirfoilDatabase.from_grid(Alpha, Re, CL, CD)  → db

    Example:
        >>> db = AirfoilDatabase("NREL-S826")
        >>> db.add_polar(polar_Re50k)
        >>> db.add_polar(polar_Re100k)
        >>> cl, cd = db.get_coefficients(alpha=8.0, Re=1.2e5)
        >>>
        >>> # Or BEMT-compatible query
        >>> query = db.to_query()
        >>> cl, cd = query(8.0, 1.2e5)
    """

    def __init__(self, name: str = "Unknown") -> None:
        """Initialize empty airfoil database

        Args:
            name: Airfoil name (e.g., "NREL-S826", "NACA0012")
        """
        self.name = name
        self._polars: Dict[float, AirfoilPolarData] = {}  # Re → polar
        self._sorted_Re: List[float] = []

    # ─────────────────────────────────────────────────────────────────────
    # §5.1 Polar Management
    # ─────────────────────────────────────────────────────────────────────

    def add_polar(self, polar: AirfoilPolarData) -> None:
        """Add a polar for a specific Reynolds number"""
        Re = polar.Re
        if Re in self._polars:
            warnings.warn(f"Overwriting polar at Re={Re:.0f}")
        self._polars[Re] = polar
        self._sorted_Re = sorted(self._polars.keys())

    def add_polar_from_arrays(
        self,
        alpha: 'npt.NDArray',
        CL: 'npt.NDArray',
        CD: 'npt.NDArray',
        Re: float,
        **kwargs
    ) -> None:
        """Create and add a polar from arrays"""
        polar = AirfoilPolarData(
            alpha=alpha, CL=CL, CD=CD, Re=Re,
            name=self.name, **kwargs
        )
        self.add_polar(polar)

    @property
    def num_polars(self) -> int:
        """Number of Re datasets available"""
        return len(self._polars)

    @property
    def Re_range(self) -> Tuple[float, float]:
        """Available Reynolds number range  [dimensionless]"""
        if not self._sorted_Re:
            return (0.0, 0.0)
        return (self._sorted_Re[0], self._sorted_Re[-1])

    # ─────────────────────────────────────────────────────────────────────
    # §5.2 OOP Query (log-linear Re interpolation)
    # ─────────────────────────────────────────────────────────────────────

    def get_coefficients(
        self,
        alpha: Union[float, 'npt.NDArray'],
        Re: float
    ) -> Tuple[Union[float, 'npt.NDArray'], Union[float, 'npt.NDArray']]:
        """Get CL, CD with log-linear Re interpolation

        Interpolation strategy:
            - 1 Re  : use directly (no interpolation)
            - exact : use matching polar
            - between: log-linear blend (CL varies ~linearly with log(Re))
            - outside: clamp to nearest available Re

        Args:
            alpha: Angle of attack  [degrees]
            Re:    Reynolds number   [dimensionless]

        Returns:
            (CL, CD)  [dimensionless, dimensionless]
        """
        if not self._polars:
            raise RuntimeError("No polar data loaded.")

        # Single Re → no interpolation
        if len(self._sorted_Re) == 1:
            return self._polars[self._sorted_Re[0]].get_coefficients(alpha)

        # Exact match
        if Re in self._polars:
            return self._polars[Re].get_coefficients(alpha)

        # Clamp to range
        Re_lo, Re_hi = self._sorted_Re[0], self._sorted_Re[-1]
        if Re <= Re_lo:
            return self._polars[Re_lo].get_coefficients(alpha)
        if Re >= Re_hi:
            return self._polars[Re_hi].get_coefficients(alpha)

        # Find bracketing Re values
        for i in range(len(self._sorted_Re) - 1):
            if self._sorted_Re[i] <= Re <= self._sorted_Re[i + 1]:
                Re1 = self._sorted_Re[i]
                Re2 = self._sorted_Re[i + 1]
                break

        # Log-linear weight: w = log(Re/Re1) / log(Re2/Re1)
        w = (np.log(Re) - np.log(Re1)) / (np.log(Re2) - np.log(Re1))

        cl1, cd1 = self._polars[Re1].get_coefficients(alpha)
        cl2, cd2 = self._polars[Re2].get_coefficients(alpha)

        cl = (1.0 - w) * np.asarray(cl1) + w * np.asarray(cl2)
        cd = (1.0 - w) * np.asarray(cd1) + w * np.asarray(cd2)

        if np.ndim(cl) == 0:
            return float(cl), float(cd)
        return cl, cd

    # ─────────────────────────────────────────────────────────────────────
    # §5.3 BEMT-Compatible Conversion
    # ─────────────────────────────────────────────────────────────────────

    def to_grid_format(
        self,
        alpha_range: Tuple[float, float] = (-180.0, 180.0),
        n_alpha: int = 361
    ) -> Tuple['npt.NDArray', 'npt.NDArray', 'npt.NDArray', 'npt.NDArray']:
        """Export as BEMT-compatible grid format (Alpha, Re, CL, CD)

        Samples all polars onto a uniform (α, Re) grid.
        Output is directly compatible with make_polar_query().

        Args:
            alpha_range: (α_min, α_max)  [degrees]
            n_alpha:     Number of α points

        Returns:
            (Alpha_grid, Re_grid, CL_grid, CD_grid)  — all shape (n_Re, n_alpha)
        """
        if not self._polars:
            raise RuntimeError("No polar data loaded.")

        alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        Res = np.array(self._sorted_Re)

        Alpha, Re_grid = np.meshgrid(alphas, Res)
        CL_grid = np.zeros_like(Alpha)  # [dimensionless]
        CD_grid = np.zeros_like(Alpha)  # [dimensionless]

        for i, Re in enumerate(Res):
            cl, cd = self._polars[Re].get_coefficients(alphas)
            CL_grid[i, :] = cl
            CD_grid[i, :] = cd

        return Alpha, Re_grid, CL_grid, CD_grid

    def to_query(self, **kwargs
                 ) -> Callable[[float, float], Tuple[float, float]]:
        """Export as BEMT-compatible bilinear query closure

        Returns:
            query(alpha_deg, Re) → (CL, CD)

        Example:
            >>> query = db.to_query()
            >>> cl, cd = query(8.0, 1.2e5)
        """
        grid = self.to_grid_format(**kwargs)
        return make_polar_query(*grid)

    def to_bemt_pack(self, **kwargs
                     ) -> List[Tuple['npt.NDArray', 'npt.NDArray',
                                     'npt.NDArray', 'npt.NDArray']]:
        """Export as BEMT pack list: [(Alpha, Re, CL, CD)]

        Directly compatible with bemt_model functions that expect
        airfoil_data_list input.
        """
        return [self.to_grid_format(**kwargs)]

    @classmethod
    def from_grid(
        cls,
        alpha_grid: 'npt.NDArray',
        re_grid: 'npt.NDArray',
        CL_grid: 'npt.NDArray',
        CD_grid: 'npt.NDArray',
        name: str = "FromGrid",
        extrapolate: bool = False,
    ) -> 'AirfoilDatabase':
        """Create AirfoilDatabase from BEMT grid format

        Args:
            alpha_grid: 2D meshgrid of α   [degrees]
            re_grid:    2D meshgrid of Re   [dimensionless]
            CL_grid:    2D array of CL      [dimensionless]
            CD_grid:    2D array of CD      [dimensionless]
            name:       Airfoil name
            extrapolate: Apply Viterna to each Re slice

        Returns:
            AirfoilDatabase instance
        """
        alphas = np.asarray(alpha_grid[0, :], dtype=np.float64)
        Res = np.asarray(re_grid[:, 0], dtype=np.float64)
        CL = np.asarray(CL_grid, dtype=np.float64)
        CD = np.asarray(CD_grid, dtype=np.float64)

        # Handle transposed grids
        if CL.shape == (alphas.size, Res.size):
            CL, CD = CL.T, CD.T

        db = cls(name=name)
        for i, Re in enumerate(Res):
            db.add_polar_from_arrays(
                alpha=alphas, CL=CL[i, :], CD=CD[i, :],
                Re=float(Re), extrapolate=extrapolate
            )
        return db

    @classmethod
    def from_bemt_pack(
        cls,
        pack: Tuple['npt.NDArray', 'npt.NDArray',
                     'npt.NDArray', 'npt.NDArray'],
        name: str = "FromBEMT",
        extrapolate: bool = False,
    ) -> 'AirfoilDatabase':
        """Create from BEMT data pack tuple

        Convenience for from_grid() using the standard
        (Alpha, Re, CL, CD) tuple from gen_airfoil_polar().
        """
        return cls.from_grid(*pack, name=name, extrapolate=extrapolate)

    # ─────────────────────────────────────────────────────────────────────
    # §5.4 File I/O
    # ─────────────────────────────────────────────────────────────────────

    def load_xfoil(self, filepath: str, Re: Optional[float] = None,
                   **kwargs) -> None:
        """Load polar from XFOIL output file

        Standard format:
            Header with "Calculated polar for: <name>"
            "Re = X.XXX e Y" line
            Data: alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
        """
        alpha_list: List[float] = []
        cl_list: List[float] = []
        cd_list: List[float] = []
        cm_list: List[float] = []
        Re_parsed = None
        name_parsed = self.name
        in_data = False
        separator_count = 0

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                if 'Calculated polar for:' in line:
                    name_parsed = line.split(
                        'Calculated polar for:'
                    )[1].strip()

                if 'Re =' in line:
                    try:
                        parts = line.split('Re =')[1].strip().split()
                        if 'e' in parts or 'E' in parts:
                            e_idx = next(
                                i for i, p in enumerate(parts)
                                if p.lower() == 'e'
                            )
                            mantissa = float(parts[e_idx - 1])
                            exponent = float(parts[e_idx + 1])
                            Re_parsed = mantissa * 10**exponent
                        else:
                            Re_parsed = float(parts[0])
                    except (ValueError, IndexError, StopIteration):
                        pass

                if '------' in line:
                    separator_count += 1
                    # XFOIL format: data starts after the first separator
                    # (column header line precedes it, then dashes, then data)
                    if separator_count >= 1:
                        in_data = True
                    continue

                if in_data and line:
                    try:
                        parts = line.split()
                        if len(parts) >= 3:
                            alpha_list.append(float(parts[0]))
                            cl_list.append(float(parts[1]))
                            cd_list.append(float(parts[2]))
                            if len(parts) >= 5:
                                cm_list.append(float(parts[4]))
                    except ValueError:
                        continue

        if not alpha_list:
            raise ValueError(f"No data in XFOIL file: {filepath}")

        Re_final = Re if Re is not None else Re_parsed
        if Re_final is None:
            raise ValueError(f"Cannot determine Re from: {filepath}")

        if name_parsed != self.name and self.name == "Unknown":
            self.name = name_parsed

        polar = AirfoilPolarData(
            alpha=np.array(alpha_list),
            CL=np.array(cl_list),
            CD=np.array(cd_list),
            Re=Re_final,
            CM=np.array(cm_list) if cm_list else None,
            name=self.name,
            **kwargs
        )
        self.add_polar(polar)

    def load_csv(
        self, filepath: str, Re: float,
        alpha_col: str = 'alpha', CL_col: str = 'CL',
        CD_col: str = 'CD', CM_col: Optional[str] = 'CM',
        delimiter: str = ',', **kwargs
    ) -> None:
        """Load single-Re polar from CSV file"""
        data: Dict[str, List[float]] = {
            alpha_col: [], CL_col: [], CD_col: []
        }

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            has_cm = CM_col is not None and CM_col in fieldnames
            if has_cm:
                data[CM_col] = []

            for row in reader:
                try:
                    data[alpha_col].append(float(row[alpha_col]))
                    data[CL_col].append(float(row[CL_col]))
                    data[CD_col].append(float(row[CD_col]))
                    if has_cm:
                        data[CM_col].append(float(row[CM_col]))
                except (ValueError, KeyError):
                    continue

        polar = AirfoilPolarData(
            alpha=np.array(data[alpha_col]),
            CL=np.array(data[CL_col]),
            CD=np.array(data[CD_col]),
            Re=Re,
            CM=(np.array(data[CM_col])
                if has_cm and data.get(CM_col) else None),
            name=self.name,
            **kwargs
        )
        self.add_polar(polar)

    def load_bemt_csv(
        self, filepath: str,
        alpha_col: str = 'AoA(deg)',
        Re_col: str = 'Re',
        CL_col: str = 'cl',
        CD_col: str = 'cd',
        **kwargs
    ) -> None:
        """Load multi-Re polar from BEMT-format CSV

        Reads a CSV with (Re, AoA, cl, cd) columns where multiple
        Re values are in the same file.

        Args:
            filepath:  Path to CSV file
            alpha_col: Column for angle of attack  [degrees]
            Re_col:    Column for Reynolds number   [dimensionless]
            CL_col:    Column for lift coefficient
            CD_col:    Column for drag coefficient
        """
        packs = load_airfoil_from_csv(
            filepath,
            alpha_col=alpha_col,
            Re_col=Re_col,
            CL_col=CL_col,
            CD_col=CD_col
        )
        # Convert pack → per-Re polars
        if packs:
            Alpha, Re, CL, CD = packs[0]
            Res = Re[:, 0]         # [dimensionless]
            alphas = Alpha[0, :]   # [degrees]
            for i, Re_val in enumerate(Res):
                self.add_polar_from_arrays(
                    alpha=alphas,
                    CL=CL[i, :],
                    CD=CD[i, :],
                    Re=float(Re_val),
                    **kwargs
                )

    # ─────────────────────────────────────────────────────────────────────
    # §5.5 Info / Display
    # ─────────────────────────────────────────────────────────────────────

    def get_info(self) -> str:
        """Return summary of all loaded polars"""
        lines = [
            f"AirfoilDatabase: {self.name}",
            f"  Polars: {self.num_polars}",
        ]
        for Re in self._sorted_Re:
            p = self._polars[Re]
            ld = p.CL_max / max(p.CD_min, 1e-6)
            lines.append(
                f"    Re = {Re:>10.0f}: "
                f"α=[{p.alpha_raw[0]:+6.1f}°..{p.alpha_raw[-1]:+6.1f}°] "
                f"CL_max={p.CL_max:.3f}  CD_min={p.CD_min:.5f}  "
                f"L/D_max={ld:.0f}"
            )
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"AirfoilDatabase('{self.name}', {self.num_polars} polars)"


# =============================================================================
# §6. Built-in Airfoil Data (Testing & Validation)
# =============================================================================

def create_flat_plate_polar(Re: float = 1e5) -> AirfoilPolarData:
    """Flat plate polar from thin airfoil theory

    CL = 2π · sin(α)  [dimensionless]
    CD = CD_0 + CL² / (π · AR · e)  [dimensionless]

    Args:
        Re: Reynolds number  [dimensionless]
    """
    alpha = np.linspace(-15, 15, 61)           # [degrees]
    alpha_rad = np.radians(alpha)               # [radians]
    CL = 2.0 * np.pi * np.sin(alpha_rad)       # [dimensionless]
    CD_0 = 0.01 * (1e5 / max(Re, 1e3))**0.2    # friction estimate
    CD = CD_0 + CL**2 / (np.pi * 10.0 * 0.95)  # AR=10, e=0.95

    return AirfoilPolarData(
        alpha=alpha, CL=CL, CD=CD, Re=Re,
        name="Flat Plate (Theory)", extrapolate=True
    )


def create_naca0012_polar(Re: float = 1e5) -> AirfoilPolarData:
    """Approximate NACA0012 polar (XFOIL-representative at Re ≈ 10⁵)

    For production: use actual XFOIL data via db.load_xfoil().
    """
    alpha = np.array([
        -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16
    ], dtype=np.float64)  # [degrees]
    CL = np.array([
        -0.95, -0.85, -0.72, -0.55, -0.38, -0.19, 0.0, 0.19, 0.38,
         0.55, 0.72, 0.85, 0.95, 0.92, 0.82
    ], dtype=np.float64)  # [dimensionless]
    CD = np.array([
        0.030, 0.022, 0.017, 0.014, 0.012, 0.011, 0.010, 0.011, 0.012,
        0.014, 0.017, 0.022, 0.030, 0.045, 0.070
    ], dtype=np.float64)  # [dimensionless]

    return AirfoilPolarData(
        alpha=alpha, CL=CL, CD=CD, Re=Re,
        name="NACA0012 (approx)", extrapolate=True
    )


def create_nrel_s826_database() -> AirfoilDatabase:
    """NREL-S826 database for NTNU Blind Test 1 validation

    Approximate polars at Re = 50k, 100k, 200k.
    For production: load actual XFOIL data.

    Returns:
        AirfoilDatabase with 3 polars
    """
    db = AirfoilDatabase("NREL-S826")

    alpha = np.array([
        -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    ], dtype=np.float64)  # [degrees]

    # ── Re = 50,000 ──
    db.add_polar_from_arrays(alpha, CL=np.array([
        -0.60, -0.55, -0.40, -0.22, -0.05, 0.15, 0.38, 0.58, 0.78,
         0.95, 1.05, 1.10, 1.05, 0.95, 0.85, 0.75
    ]), CD=np.array([
        0.035, 0.028, 0.022, 0.018, 0.016, 0.015, 0.015, 0.016, 0.018,
        0.022, 0.028, 0.038, 0.052, 0.075, 0.100, 0.130
    ]), Re=5e4)

    # ── Re = 100,000 ──
    db.add_polar_from_arrays(alpha, CL=np.array([
        -0.70, -0.62, -0.45, -0.25, -0.05, 0.20, 0.45, 0.68, 0.90,
         1.08, 1.22, 1.28, 1.20, 1.05, 0.90, 0.78
    ]), CD=np.array([
        0.028, 0.022, 0.017, 0.014, 0.012, 0.011, 0.011, 0.012, 0.014,
        0.018, 0.023, 0.032, 0.045, 0.065, 0.090, 0.120
    ]), Re=1e5)

    # ── Re = 200,000 ──
    db.add_polar_from_arrays(alpha, CL=np.array([
        -0.78, -0.68, -0.50, -0.28, -0.05, 0.22, 0.50, 0.75, 0.98,
         1.18, 1.32, 1.38, 1.30, 1.12, 0.95, 0.80
    ]), CD=np.array([
        0.024, 0.018, 0.014, 0.011, 0.010, 0.009, 0.009, 0.010, 0.012,
        0.015, 0.020, 0.028, 0.040, 0.058, 0.082, 0.110
    ]), Re=2e5)

    return db

def create_polar_from_config(config: dict) -> Callable:
    """Create polar query function from configuration
    
    Args:
        config: airfoil_polar configuration dictionary
        
    Returns:
        polar_query: Callable(alpha_deg, Re) → (CL, CD)
    """
    method = config.get("method", "neuralfoil")
    
    if method == "neuralfoil":
        airfoil_name = config.get("airfoil_name", "naca0012")
        Re_target = config.get("Re_target", 1e5)
        mode = config.get("mode", "asb")
        ncrit = config.get("ncrit", 9.0)
        dat_path = config.get("dat_path", None)
        coordinates = config.get("coordinates", None)
        
        packs = gen_airfoil_polar(
            airfoil_name=airfoil_name,
            Re_target=Re_target,
            mode=mode,
            ncrit=ncrit,
            dat_path=dat_path,
            coordinates=coordinates,
        )
        return make_polar_query(*packs[0])
    
    elif method == "flat_plate":
        Re_target = config.get("Re_target", 1e5)
        polar = create_flat_plate_polar(Re=Re_target)
        return polar.get_coefficients
    
    elif method == "database":
        preset = config.get("preset", "nrel_s826")
        if preset == "nrel_s826":
            db = create_nrel_s826_database()
        elif preset == "naca0012_approx":
            db = AirfoilDatabase("NACA0012")
            db.add_polar(create_naca0012_polar(Re=1e5))
        else:
            raise ValueError(f"Unknown preset: {preset}")
        return db.to_query()
    
    elif method == "csv":
        csv_path = config["csv_path"]
        packs = load_airfoil_from_csv(
            csv_path,
            alpha_col=config.get("alpha_col", "AoA(deg)"),
            Re_col=config.get("Re_col", "Re"),
            CL_col=config.get("CL_col", "cl"),
            CD_col=config.get("CD_col", "cd"),
        )
        return make_polar_query(*packs[0])
    
    else:
        raise ValueError(f"Unknown polar method: {method}")
    
def gen_airfoil_polar_extended(
    airfoil_name: str = "naca0012",
    Re_target: float = 1e5,
    Re_min: Optional[float] = None,
    Re_max: Optional[float] = None,
    Re_steps: Optional[int] = None,
    alphas: Optional['npt.NDArray'] = None,
    mode: str = "asb",
    ncrit: float = 9.0,
    coordinates: Optional['npt.NDArray'] = None,
    dat_path: Optional[str] = None,
) -> List[Tuple['npt.NDArray', 'npt.NDArray', 'npt.NDArray', 'npt.NDArray']]:
    """Extended polar generation with configurable Re range

    Enhanced version of gen_airfoil_polar() with explicit Re range control.

    Re Range Determination (priority order):
        1. If Re_min & Re_max given: use [Re_min, Re_max]
        2. Else: use [Re_target/2, Re_target*2] (default behavior)

    Args:
        airfoil_name: Airfoil name for aerosandbox  [string]
        Re_target:    Target Reynolds number        [dimensionless]
        Re_min:       Minimum Re (optional)         [dimensionless]
        Re_max:       Maximum Re (optional)         [dimensionless]
        Re_steps:     Number of Re points (optional, auto if None)
        alphas:       Angle of attack array         [degrees]
        mode:         "asb" | "user" | "dat"
        ncrit:        Critical N-factor             [dimensionless]
        coordinates:  Airfoil coords for mode="user"
        dat_path:     .dat file path for mode="dat"

    Returns:
        [(Alpha_grid, Re_grid, CL_grid, CD_grid)]

    Example:
        >>> # Custom Re range: 100 to 1000
        >>> packs = gen_airfoil_polar_extended(
        ...     "naca0012", Re_target=300,
        ...     Re_min=100, Re_max=1000, Re_steps=10
        ... )
    """
    if alphas is None:
        alphas = np.linspace(-180, 180, 361)  # [degrees]

    # ── Determine Re range ──
    if Re_min is not None and Re_max is not None:
        # User-specified range
        Re_low = Re_min
        Re_high = Re_max
        if Re_steps is not None:
            Res = np.linspace(Re_low, Re_high, Re_steps)
        else:
            # Auto step based on order of magnitude
            order = int(np.floor(np.log10(Re_target)))
            step = max(10 ** (order - 1), 10)  # minimum step = 10
            Res = np.arange(Re_low, Re_high + step, step)
    else:
        # Default: half-decade around target
        Re_low = Re_target / 2
        Re_high = Re_target * 2
        order = int(np.floor(np.log10(max(Re_target, 10))))
        step = max(10 ** (order - 1), 10)
        Res = np.arange(Re_low, Re_high + step, step)

    # Ensure at least 2 Re values
    if len(Res) < 2:
        Res = np.array([Re_low, Re_high])

    Alpha, Re = np.meshgrid(alphas, Res)  # [degrees], [dimensionless]

    # ── Generate polars ──
    if mode == "asb":
        try:
            import aerosandbox as asb
        except ImportError:
            raise ImportError(
                "aerosandbox required for mode='asb'. "
                "Install: pip install aerosandbox"
            )
        af = asb.Airfoil(airfoil_name)
        aero = af.get_aero_from_neuralfoil(
            alpha=Alpha.ravel(),
            Re=Re.ravel(),
            mach=0,
            model_size="xxlarge"
        )

    elif mode == "user":
        try:
            import neuralfoil as nf
        except ImportError:
            raise ImportError(
                "neuralfoil required for mode='user'. "
                "Install: pip install neuralfoil"
            )
        if coordinates is None:
            raise ValueError("coordinates required for mode='user'")
        aero = nf.get_aero_from_coordinates(
            coordinates=coordinates,
            alpha=Alpha.ravel(),
            Re=Re.ravel(),
            model_size="xxlarge",
            n_crit=ncrit
        )

    elif mode == "dat":
        try:
            import neuralfoil as nf
        except ImportError:
            raise ImportError(
                "neuralfoil required for mode='dat'. "
                "Install: pip install neuralfoil"
            )
        if dat_path is None:
            raise ValueError("dat_path required for mode='dat'")
        aero = nf.get_aero_from_dat_file(
            filename=dat_path,
            alpha=Alpha.ravel(),
            Re=Re.ravel(),
            model_size="xlarge",
            n_crit=ncrit
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'asb', 'user', or 'dat'.")

    CL_grid = aero["CL"].reshape(Alpha.shape)  # [dimensionless]
    CD_grid = aero["CD"].reshape(Alpha.shape)  # [dimensionless]

    return [(Alpha, Re, CL_grid, CD_grid)]


class MultiAirfoilPolarManager:
    """Manager for multiple airfoil polar datasets

    Enables blade sections to use different airfoils by maintaining
    a dictionary of airfoil_name → polar_query mappings.

    Data Flow:
        Config → MultiAirfoilPolarManager → polar_query(alpha, Re, airfoil)
                                          or queries[airfoil](alpha, Re)

    Example:
        >>> manager = MultiAirfoilPolarManager()
        >>> manager.add_airfoil_from_neuralfoil("naca0012", Re_target=300)
        >>> manager.add_airfoil_from_neuralfoil("nrel_s826", Re_target=1e5)
        >>>
        >>> # Query with airfoil name
        >>> cl, cd = manager.query("naca0012", alpha=5.0, Re=200)
        >>>
        >>> # Or get individual query function
        >>> query = manager.get_query("naca0012")
        >>> cl, cd = query(5.0, 200)
    """

    def __init__(self) -> None:
        """Initialize empty manager"""
        self._databases: Dict[str, AirfoilDatabase] = {}
        self._queries: Dict[str, Callable] = {}
        self._default_airfoil: Optional[str] = None

    def add_airfoil(
        self,
        name: str,
        database: AirfoilDatabase,
        set_default: bool = False
    ) -> None:
        """Add an airfoil database

        Args:
            name: Airfoil identifier (used in blade sections)
            database: AirfoilDatabase instance
            set_default: If True, use as fallback for unknown airfoils
        """
        self._databases[name] = database
        self._queries[name] = database.to_query()
        if set_default or self._default_airfoil is None:
            self._default_airfoil = name

    def add_airfoil_from_neuralfoil(
        self,
        name: str,
        Re_target: float = 1e5,
        Re_min: Optional[float] = None,
        Re_max: Optional[float] = None,
        Re_steps: Optional[int] = None,
        mode: str = "asb",
        ncrit: float = 9.0,
        dat_path: Optional[str] = None,
        set_default: bool = False,
    ) -> None:
        """Generate and add airfoil polar using neuralfoil/aerosandbox

        Args:
            name: Airfoil identifier (e.g., "naca0012", "nrel_s826")
            Re_target: Target Reynolds number  [dimensionless]
            Re_min: Minimum Re (optional)      [dimensionless]
            Re_max: Maximum Re (optional)      [dimensionless]
            Re_steps: Number of Re points
            mode: "asb" | "user" | "dat"
            ncrit: Critical N-factor
            dat_path: Path to .dat file (for mode="dat")
            set_default: Use as fallback for unknown airfoils
        """
        packs = gen_airfoil_polar_extended(
            airfoil_name=name,
            Re_target=Re_target,
            Re_min=Re_min,
            Re_max=Re_max,
            Re_steps=Re_steps,
            mode=mode,
            ncrit=ncrit,
            dat_path=dat_path,
        )
        db = AirfoilDatabase.from_bemt_pack(packs[0], name=name)
        self.add_airfoil(name, db, set_default=set_default)

    def add_airfoil_from_flat_plate(
        self,
        name: str = "flat_plate",
        Re_target: float = 1e5,
        set_default: bool = False,
    ) -> None:
        """Add flat plate theory polar

        Args:
            name: Airfoil identifier
            Re_target: Reference Re for drag scaling
            set_default: Use as fallback
        """
        polar = create_flat_plate_polar(Re=Re_target)
        db = AirfoilDatabase(name)
        db.add_polar(polar)
        self.add_airfoil(name, db, set_default=set_default)

    def add_airfoil_from_csv(
        self,
        name: str,
        csv_path: str,
        alpha_col: str = "AoA(deg)",
        Re_col: str = "Re",
        CL_col: str = "cl",
        CD_col: str = "cd",
        set_default: bool = False,
    ) -> None:
        """Add airfoil from CSV file

        Args:
            name: Airfoil identifier
            csv_path: Path to CSV file
            alpha_col, Re_col, CL_col, CD_col: Column names
            set_default: Use as fallback
        """
        db = AirfoilDatabase(name)
        db.load_csv(
            csv_path,
            alpha_col=alpha_col,
            Re_col=Re_col,
            CL_col=CL_col,
            CD_col=CD_col,
        )
        self.add_airfoil(name, db, set_default=set_default)

    def get_query(self, airfoil_name: str) -> Callable:
        """Get polar query function for specific airfoil

        Args:
            airfoil_name: Airfoil identifier

        Returns:
            query(alpha_deg, Re) → (CL, CD)
        """
        # Normalize name (case-insensitive)
        name_lower = airfoil_name.lower()
        for key in self._queries:
            if key.lower() == name_lower:
                return self._queries[key]

        # Fallback to default
        if self._default_airfoil is not None:
            warnings.warn(
                f"Unknown airfoil '{airfoil_name}', using default "
                f"'{self._default_airfoil}'"
            )
            return self._queries[self._default_airfoil]

        raise KeyError(f"Airfoil '{airfoil_name}' not found and no default set")

    def query(
        self,
        airfoil_name: str,
        alpha: float,
        Re: float
    ) -> Tuple[float, float]:
        """Query CL, CD for specific airfoil

        Args:
            airfoil_name: Airfoil identifier
            alpha: Angle of attack  [degrees]
            Re: Reynolds number     [dimensionless]

        Returns:
            (CL, CD)  [dimensionless, dimensionless]
        """
        query_func = self.get_query(airfoil_name)
        return query_func(alpha, Re)

    def to_unified_query(self) -> Callable:
        """Create unified query function accepting airfoil name

        Returns:
            query(alpha_deg, Re, airfoil_name) → (CL, CD)

        Note:
            If airfoil_name is None or empty, uses default airfoil.
        """
        def unified_query(
            alpha_deg: float,
            Re: float,
            airfoil_name: Optional[str] = None
        ) -> Tuple[float, float]:
            if airfoil_name is None or airfoil_name == "":
                airfoil_name = self._default_airfoil
            return self.query(airfoil_name, alpha_deg, Re)
        return unified_query

    @property
    def airfoil_names(self) -> List[str]:
        """List of registered airfoil names"""
        return list(self._databases.keys())

    @property
    def default_airfoil(self) -> Optional[str]:
        """Name of default airfoil"""
        return self._default_airfoil

    def get_info(self) -> str:
        """Human-readable summary"""
        lines = [
            "MultiAirfoilPolarManager",
            "=" * 60,
            f"  Registered airfoils: {len(self._databases)}",
            f"  Default: {self._default_airfoil}",
            "",
        ]
        for name, db in self._databases.items():
            Re_lo, Re_hi = db.Re_range
            marker = " (default)" if name == self._default_airfoil else ""
            lines.append(f"  [{name}]{marker}")
            lines.append(f"      Re range: {Re_lo:.0f} ~ {Re_hi:.0f}")
            lines.append(f"      Polars: {db.num_polars}")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"MultiAirfoilPolarManager({self.airfoil_names})"


def create_polar_from_config(
    config: Dict
) -> Tuple[Callable, Optional[MultiAirfoilPolarManager]]:
    """Create polar query function(s) from configuration

    Supports both single-airfoil and multi-airfoil configurations.

    Config Formats:
        1. Single airfoil (legacy compatible):
            {
                "method": "neuralfoil",
                "airfoil_name": "naca0012",
                "Re_target": 300,
                ...
            }

        2. Multiple airfoils:
            {
                "method": "multi",
                "default": "naca0012",
                "airfoils": {
                    "naca0012": {"Re_target": 300, "mode": "asb"},
                    "nrel_s826": {"Re_target": 1e5, "mode": "asb"},
                    "custom": {"csv_path": "data/custom.csv"}
                }
            }

    Args:
        config: airfoil_polar configuration dictionary

    Returns:
        (polar_query, manager)
        - polar_query: Callable for single airfoil query
        - manager: MultiAirfoilPolarManager (None if single airfoil)

    Raises:
        ValueError: Invalid configuration
        ImportError: Missing required packages
    """
    method = config.get("method", "neuralfoil")

    # ═══════════════════════════════════════════════════════════════════
    # Multi-airfoil mode
    # ═══════════════════════════════════════════════════════════════════
    if method == "multi":
        manager = MultiAirfoilPolarManager()
        airfoils_cfg = config.get("airfoils", {})
        default_name = config.get("default", None)

        if not airfoils_cfg:
            raise ValueError("'airfoils' dict required for method='multi'")

        for name, af_cfg in airfoils_cfg.items():
            af_method = af_cfg.get("method", "neuralfoil")
            is_default = (name == default_name) or (default_name is None)

            if af_method == "neuralfoil":
                manager.add_airfoil_from_neuralfoil(
                    name=af_cfg.get("airfoil_name", name),
                    Re_target=af_cfg.get("Re_target", 1e5),
                    Re_min=af_cfg.get("Re_min"),
                    Re_max=af_cfg.get("Re_max"),
                    Re_steps=af_cfg.get("Re_steps"),
                    mode=af_cfg.get("mode", "asb"),
                    ncrit=af_cfg.get("ncrit", 9.0),
                    dat_path=af_cfg.get("dat_path"),
                    set_default=is_default,
                )
            elif af_method == "flat_plate":
                manager.add_airfoil_from_flat_plate(
                    name=name,
                    Re_target=af_cfg.get("Re_target", 1e5),
                    set_default=is_default,
                )
            elif af_method == "csv":
                manager.add_airfoil_from_csv(
                    name=name,
                    csv_path=af_cfg["csv_path"],
                    alpha_col=af_cfg.get("alpha_col", "AoA(deg)"),
                    Re_col=af_cfg.get("Re_col", "Re"),
                    CL_col=af_cfg.get("CL_col", "cl"),
                    CD_col=af_cfg.get("CD_col", "cd"),
                    set_default=is_default,
                )
            else:
                raise ValueError(f"Unknown method '{af_method}' for airfoil '{name}'")

        # Return unified query (3-arg) and manager
        return manager.to_unified_query(), manager

    # ═══════════════════════════════════════════════════════════════════
    # Single airfoil modes
    # ═══════════════════════════════════════════════════════════════════
    if method == "neuralfoil":
        airfoil_name = config.get("airfoil_name", "naca0012")
        Re_target = config.get("Re_target", 1e5)
        Re_min = config.get("Re_min")
        Re_max = config.get("Re_max")
        Re_steps = config.get("Re_steps")
        mode = config.get("mode", "asb")
        ncrit = config.get("ncrit", 9.0)
        dat_path = config.get("dat_path")
        coordinates = config.get("coordinates")

        packs = gen_airfoil_polar_extended(
            airfoil_name=airfoil_name,
            Re_target=Re_target,
            Re_min=Re_min,
            Re_max=Re_max,
            Re_steps=Re_steps,
            mode=mode,
            ncrit=ncrit,
            coordinates=coordinates,
            dat_path=dat_path,
        )
        query = make_polar_query(*packs[0])
        return query, None

    elif method == "flat_plate":
        Re_target = config.get("Re_target", 1e5)
        polar = create_flat_plate_polar(Re=Re_target)
        return polar.get_coefficients, None

    elif method == "database":
        preset = config.get("preset", "nrel_s826")
        if preset == "nrel_s826":
            db = create_nrel_s826_database()
        elif preset == "naca0012_approx":
            db = AirfoilDatabase("NACA0012")
            db.add_polar(create_naca0012_polar(Re=1e5))
        else:
            raise ValueError(f"Unknown preset: {preset}")
        return db.to_query(), None

    elif method == "csv":
        csv_path = config["csv_path"]
        packs = load_airfoil_from_csv(
            csv_path,
            alpha_col=config.get("alpha_col", "AoA(deg)"),
            Re_col=config.get("Re_col", "Re"),
            CL_col=config.get("CL_col", "cl"),
            CD_col=config.get("CD_col", "cd"),
        )
        return make_polar_query(*packs[0]), None

    else:
        raise ValueError(
            f"Unknown polar method: '{method}'. "
            f"Available: 'neuralfoil', 'flat_plate', 'database', 'csv', 'multi'"
        )