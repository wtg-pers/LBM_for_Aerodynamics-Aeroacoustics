"""
Richardson extrapolation + Grid Convergence Index (GCI) for grid-convergence
studies (Roache 1998; ASME V&V20).

Given a quantity (e.g. C_T) at >=2 grids of resolution D (cells per L_char,
so grid spacing h ~ 1/D), estimate the grid-converged value and its
uncertainty:

  - 3 grids: observed order of convergence p is solved from the data (handles
    non-constant refinement ratios via Roache's iterative formula); the finest
    pair is Richardson-extrapolated; GCI gives a numerical-uncertainty band.
  - 2 grids: p must be assumed (default 2 = nominally second-order).

Usage (programmatic):
    from src.utilities.richardson import grid_convergence
    grid_convergence([32, 36, 40], [0.00553, 0.00528, 0.00510], target=0.00473)

CLI:
    python src/utilities/richardson.py --D 32 36 40 --f 0.00553 0.00528 0.00510 \\
        --target 0.00473 --label "C_T (M=0.877)"
"""
import argparse
import math


def _solve_order(e21, e32, r21, r32, tol=1e-8, itmax=200):
    """Roache's iterative solve for observed order p (non-uniform ratios)."""
    s = 1.0 if (e32 / e21) > 0 else -1.0
    p = 2.0
    for _ in range(itmax):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return p


def grid_convergence(D, f, target=None, assumed_p=2.0, Fs=1.25):
    """Richardson extrapolation + GCI.

    Args:
        D:  list of resolutions (cells per L_char); larger D = finer.
        f:  list of the quantity at each D (same order).
        target: optional reference value to compare the extrapolation against.
        assumed_p: order used when only 2 grids are given.
        Fs: GCI safety factor (1.25 for 3 grids, 3.0 for 2 grids per Roache).

    Returns dict with: order p, extrapolated f0, gci (%), and (if target)
    the extrapolated and finest-grid errors vs target.
    """
    pairs = sorted(zip(D, f))                      # coarse -> fine
    Ds = [d for d, _ in pairs]
    fs = [v for _, v in pairs]
    h = [1.0 / d for d in Ds]                       # grid spacing ~ 1/D

    out = {"D": Ds, "f": fs}
    if len(Ds) >= 3:
        f1, f2, f3 = fs[-3], fs[-2], fs[-1]         # coarse, mid, fine
        r21 = h[-3] / h[-2]
        r32 = h[-2] / h[-1]
        e21, e32 = (f2 - f1), (f3 - f2)
        if e21 == 0 or e32 == 0:
            p = assumed_p
        else:
            p = _solve_order(e21, e32, r21, r32)
        f0 = f3 + (f3 - f2) / (r32 ** p - 1.0)
        gci = Fs * abs((f3 - f2) / f3) / (r32 ** p - 1.0)
        out.update(order=p, f0=f0, gci_pct=100 * gci,
                   monotonic=(e21 * e32 > 0))
    elif len(Ds) == 2:
        f1, f2 = fs[-2], fs[-1]
        r = h[-2] / h[-1]
        p = assumed_p
        f0 = f2 + (f2 - f1) / (r ** p - 1.0)
        gci = 3.0 * abs((f2 - f1) / f2) / (r ** p - 1.0)
        out.update(order=p, f0=f0, gci_pct=100 * gci, assumed_order=True)
    else:
        out.update(order=None, f0=fs[-1], gci_pct=float("nan"))

    if target is not None and out.get("f0") is not None:
        out["target"] = target
        out["err_extrap_pct"] = 100 * (out["f0"] - target) / target
        out["err_finest_pct"] = 100 * (fs[-1] - target) / target
    return out


def report(res, label="quantity"):
    print("=" * 60)
    print(" Grid-convergence: %s" % label)
    print("=" * 60)
    for d, v in zip(res["D"], res["f"]):
        print("   D=%-4d  %s = %.6f" % (d, label, v))
    if res.get("order") is not None:
        kind = " (assumed)" if res.get("assumed_order") else " (observed)"
        print("   ----")
        print("   order p%s      = %.2f" % (kind, res["order"]))
        print("   extrapolated f0   = %.6f" % res["f0"])
        print("   GCI (uncertainty) = %.2f%%" % res["gci_pct"])
        if "monotonic" in res and not res["monotonic"]:
            print("   [warn] non-monotonic convergence -> p/extrapolation unreliable")
    if "target" in res:
        print("   ----")
        print("   target            = %.6f" % res["target"])
        print("   finest-grid error = %+.1f%%" % res["err_finest_pct"])
        print("   extrapolated error= %+.1f%%" % res["err_extrap_pct"])
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=float, nargs="+", required=True)
    ap.add_argument("--f", type=float, nargs="+", required=True)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--label", default="C_T")
    ap.add_argument("--assumed-p", type=float, default=2.0)
    a = ap.parse_args()
    if len(a.D) != len(a.f):
        raise SystemExit("--D and --f must have equal length")
    report(grid_convergence(a.D, a.f, a.target, a.assumed_p), a.label)


if __name__ == "__main__":
    main()
