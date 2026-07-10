"""Phase 1e gate — scoped coupling API vs original: BIT-EXACT.

(e1) f_coarse_prev passed as the pre-extracted sub-volume must give the SAME
     fine strips as the full-array path (identical values are read).
(e2) f_coarse_is_sub + strips_out must reproduce the strips the default
     full-volume C2F writes; pre-strided F2C + return_excised must reproduce
     the block the default F2C writes. All comparisons bit-exact on GPU
     (same math, different access paths).

Run from repo root:  python patch_notes/hpc_upgrade/gates/eso_coupling_scoped_gate.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import cupy as cp
from src.lattice import get_lattice
from src.grid.overlap_manager import OverlapRegion, IndexBox
from src.grid.level_scaling import LevelScaler
from src.grid.interpolation import CubicInterpolation
from src.grid.coupling import GridCoupling

COARSE = (28, 26, 24)
FINE_REGION = IndexBox(8, 19, 7, 17, 6, 15)


def build():
    lat = get_lattice('D3Q27', cp, dtype=cp.float32)
    region = OverlapRegion(0, COARSE, FINE_REGION, overlap_width=2)
    scaler = LevelScaler(tau_0=0.6, num_levels=2)
    return GridCoupling(cp, lat, region, scaler, CubicInterpolation(),
                        filter_level=1), region


def main():
    cpl, region = build()
    rng = cp.random.default_rng(2)
    Q = 27
    f_coarse = (rng.random((Q,) + COARSE, dtype=cp.float32) * 0.02 + 0.03)
    f_prev = (rng.random((Q,) + COARSE, dtype=cp.float32) * 0.02 + 0.03)
    Nf = region.fine_shape
    f_fine0 = (rng.random((Q,) + Nf, dtype=cp.float32) * 0.02 + 0.03)

    sub_sl = (slice(None),) + cpl.coarse_sub_spatial_slices
    ok = True

    for half in (True, False):
        # -- reference: original full-array call --
        fA = f_fine0.copy()
        cpl.coarse_to_fine(f_coarse, fA, is_half_step=half,
                           f_coarse_prev=(f_prev if half else None))
        # -- (e1): f_prev as sub-volume --
        fB = f_fine0.copy()
        cpl.coarse_to_fine(f_coarse, fB, is_half_step=half,
                           f_coarse_prev=(cp.ascontiguousarray(f_prev[sub_sl])
                                          if half else None),
                           f_coarse_prev_is_sub=half)
        e1 = bool(cp.all(fA == fB))
        # -- (e2): coarse sub + strips_out --
        strips: list = []
        cpl.coarse_to_fine(cp.ascontiguousarray(f_coarse[sub_sl]), None,
                           is_half_step=half,
                           f_coarse_prev=(cp.ascontiguousarray(f_prev[sub_sl])
                                          if half else None),
                           f_coarse_is_sub=True, f_coarse_prev_is_sub=half,
                           strips_out=strips)
        fC = f_fine0.copy()
        for sp_sl, vals in strips:
            fC[(slice(None),) + sp_sl] = vals
        e2 = bool(cp.all(fA == fC))
        print(f"  [C2F half={half}] e1 f_prev-sub bit-exact={e1}   "
              f"e2 sub+strips bit-exact={e2}  (strips={len(strips)})")
        ok = ok and e1 and e2

    # -- F2C --
    fcA = f_coarse.copy()
    cpl.fine_to_coarse(f_fine0, fcA)
    at_sl = (slice(None),) + cpl.fine_at_coarse_spatial_slices
    block = cpl.fine_to_coarse(cp.ascontiguousarray(f_fine0[at_sl]), None,
                               f_fine_is_at_coarse=True, return_excised=True)
    fcB = f_coarse.copy()
    fcB[(slice(None),) + cpl.excised_spatial_slices] = block
    e3 = bool(cp.all(fcA == fcB))
    print(f"  [F2C] pre-strided + return_excised bit-exact={e3}")
    ok = ok and e3

    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
