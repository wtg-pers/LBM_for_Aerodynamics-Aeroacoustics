"""
Generate a Mach-indexed C81 airfoil deck from NeuralFoil (+ Viterna to +/-180).

Produces a deck readable by `C81PolarSet.from_file` (see c81_loader.py), so a
constant-chord rotor can use the existing `airfoil_polar` method "c81" with a
synthetic, Mach-dependent polar.  Reusable for any NeuralFoil-supported airfoil
(NACA0012 for Caradonna-Tung; RC-series for HVAB once coordinates are supplied).

Per Mach: NeuralFoil over the attached-flow range, then AirfoilPolarData's
Viterna extrapolation fills +/-180 deg.  CL/CD drive the ALM forces; CM is
carried for completeness (unused by thrust/torque).

Usage:
    python src/actuator/c81_from_neuralfoil.py naca0012 data/airfoils/NACA0012.C81
"""
import os
import sys

import numpy as np


# Default grids (nM, nA must stay < 100 for the 2-digit C81 header fields)
MACH_GRID = np.round(np.arange(0.0, 0.81, 0.1), 3)          # 0.0 .. 0.8  (9)
AOA_GRID = np.concatenate([                                  # 61 points
    np.arange(-180.0, -30.0, 10.0),       # -180 .. -40  (coarse)
    np.arange(-30.0, 30.1, 2.0),          # -30 .. 30    (fine, attached)
    np.arange(40.0, 180.1, 10.0),         # 40 .. 180    (coarse)
])
NF_RANGE = (-24.0, 24.0, 1.0)             # NeuralFoil sampling (attached)


def _resolve_airfoil(airfoil_ref):
    """Resolve a NAME (asb/UIUC database) or a coordinate .dat PATH to an
    asb.Airfoil.  A path lets us use custom sections (e.g. RC(6)-08 from the
    NASA-TM-4264 design coordinates, or HVAB FileShare sections)."""
    import aerosandbox as asb
    if isinstance(airfoil_ref, str) and os.path.isfile(airfoil_ref):
        coords = np.loadtxt(airfoil_ref, skiprows=1)   # Selig .dat (skip name)
        nm = os.path.splitext(os.path.basename(airfoil_ref))[0]
        return asb.Airfoil(name=nm, coordinates=coords)
    return asb.Airfoil(airfoil_ref)


def _nf_polar(airfoil_ref, mach, Re):
    """NeuralFoil CL/CD/CM over the attached range at one Mach.

    airfoil_ref: airfoil name (asb database) OR a coordinate .dat path.
    """
    a = np.arange(*NF_RANGE)
    aero = _resolve_airfoil(airfoil_ref).get_aero_from_neuralfoil(
        alpha=a, Re=Re, mach=float(mach), model_size="large")
    cm = aero.get("CM")
    return a, np.asarray(aero["CL"]), np.asarray(aero["CD"]), \
        (np.asarray(cm) if cm is not None else None)


def generate(airfoil_name, out_path, Re=2.0e6,
             mach_grid=MACH_GRID, aoa_grid=AOA_GRID, dat=None):
    """Build a Mach x AoA C81 deck and write it to out_path.

    airfoil_name: deck name (also used as asb lookup if `dat` is None).
    dat: optional coordinate .dat path (custom section); overrides the name
         for the NeuralFoil lookup while keeping `airfoil_name` as the deck id.
    """
    from src.actuator.airfoil_data import AirfoilPolarData

    airfoil_ref = dat if dat else airfoil_name
    nM, nA = len(mach_grid), len(aoa_grid)
    CL = np.zeros((nA, nM)); CD = np.zeros((nA, nM)); CM = np.zeros((nA, nM))
    for j, M in enumerate(mach_grid):
        a_nf, cl, cd, cm = _nf_polar(airfoil_ref, M, Re)
        pol = AirfoilPolarData(a_nf, cl, cd, Re=Re,
                               CM=cm, name=airfoil_name, extrapolate=True)
        CL[:, j] = pol.get_CL(aoa_grid)
        CD[:, j] = pol.get_CD(aoa_grid)
        try:
            CM[:, j] = pol.get_CM(aoa_grid)
        except Exception:
            CM[:, j] = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _write_c81(out_path, airfoil_name.upper(), mach_grid, aoa_grid, CL, CD, CM)
    return out_path


def _write_c81(path, name, mach, aoa, CL, CD, CM):
    """Write a C81 deck round-trippable by c81_loader.C81PolarSet.from_file."""
    nM, nA = len(mach), len(aoa)
    assert nM < 100 and nA < 100, "C81 header fields are 2-digit"
    hdr = "%-28s%02d%02d%02d%02d%02d%02d" % (
        (name + ",")[:28], nM, nA, nM, nA, nM, nA)
    lines = [hdr]
    for tab in (CL, CD, CM):
        lines.append(" ".join("%.6g" % m for m in mach))
        for i in range(nA):
            lines.append(" ".join(["%.6g" % aoa[i]]
                                  + ["%.6g" % v for v in tab[i, :]]))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")))
    name = sys.argv[1] if len(sys.argv) > 1 else "naca0012"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/airfoils/%s.C81" % name.upper()
    dat = sys.argv[3] if len(sys.argv) > 3 else None   # optional coordinate .dat
    p = generate(name, out, dat=dat)
    print("wrote:", p)

    # round-trip validation
    from src.actuator.c81_loader import C81PolarSet
    pol = C81PolarSet.from_file(p)
    print(pol.info())
    a0 = None
    aa = np.linspace(-6, 6, 25); cl0 = pol.get_CL(aa, 0.0)
    k = np.where(np.diff(np.sign(cl0)))[0]
    if k.size:
        i = k[0]; a0 = aa[i] - cl0[i]*(aa[i+1]-aa[i])/(cl0[i+1]-cl0[i])
    slope = (pol.get_CL(2.0, 0.0) - pol.get_CL(-2.0, 0.0)) / 4.0
    print("  M=0: a0=%s deg, CLa=%.4f/deg, CLmax=%.3f, CDmin=%.5f"
          % ("%.3f" % a0 if a0 is not None else "n/a", slope,
             pol.get_CL(np.linspace(0, 18, 181), 0.0).max(),
             pol.get_CD(np.linspace(-6, 6, 61), 0.0).min()))
    for M in (0.4, 0.6):
        print("  M=%.1f: CL(5deg)=%.4f, CLa=%.4f/deg"
              % (M, pol.get_CL(5.0, M),
                 (pol.get_CL(5.0, M) - pol.get_CL(-5.0, M)) / 10.0))
