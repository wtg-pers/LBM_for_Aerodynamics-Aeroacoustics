"""G-M2a — MLG rank-local coupling: 2-rank vs 1-rank BIT, all three axes.

Synthetic 2-level MLG (periodic coarse 36x32x32 + nested fine box, esoteric
cumulant on both levels, real GridCoupling with the fused rescale kernel).
Both runs execute the IDENTICAL production schedule per coarse step:

    save f_prev(coarse sub) -> L0 kernel -> L1 kernel -> C2F(half, f_prev)
    -> L1 kernel -> C2F(full) -> F2C

Reference (1 rank) drives the real GridCoupling through the esoteric-branch
API (gather sub -> strips_out -> scatter), i.e. the production
_coupling_c2f/_coupling_f2c flow. The decomposed run uses Partition1D per
level (fine cuts DERIVED from coarse cuts), HaloBandExchangerV1 ring
exchange, and RankLocalCouplingV1. Owned cells must match bit-for-bit.

Run from repo root:  python patch_notes/hpc_upgrade/gates/mgpu_m2a_gate.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.lattice import get_lattice
from src.grid.overlap_manager import OverlapRegion, IndexBox
from src.grid.level_scaling import LevelScaler
from src.grid.interpolation import CubicInterpolation
from src.grid.coupling import GridCoupling
from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
from src.kernels.esoteric_d3q27 import (
    esoteric_gather_std, esoteric_scatter_std,
    esoteric_gather_std_region, esoteric_scatter_std_region)
from src.parallel import Partition1D, HaloBandExchangerV1, LoopbackTransport
from src.parallel.mlg_coupling import RankLocalCouplingV1, fine_range_from_coarse

COARSE = (36, 32, 32)
FBOX = IndexBox(8, 27, 6, 25, 6, 25)      # fine_region (coarse coords)
NSTEPS = 5                                 # coarse steps
NR = 2
U0 = 0.04

_CXl = [0, 1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0, 1,-1,1,-1,1,-1,1,-1]
_CYl = [0, 0,0,1,-1,0,0, 1,1,-1,-1,0,0,0,0, 1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1]
_CZl = [0, 0,0,0,0,1,-1, 0,0,0,0,1,1,-1,-1, 1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1]
CX = cp.asarray(_CXl, cp.float32); CY = cp.asarray(_CYl, cp.float32); CZ = cp.asarray(_CZl, cp.float32)
W = cp.asarray([8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8, cp.float32)


def feq_of_u(u):
    cu = (CX[:, None, None, None]*u[0] + CY[:, None, None, None]*u[1]
          + CZ[:, None, None, None]*u[2])
    usq = u[0]**2 + u[1]**2 + u[2]**2
    return (W[:, None, None, None]*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
            ).astype(cp.float32)


def tgv_u(xc, yc, zc):
    """Analytic TGV velocity at coarse-coordinate grids (broadcastable)."""
    k = 2*np.pi/COARSE[0]
    ux = U0*cp.cos(k*xc)*cp.sin(k*yc)*cp.sin(k*zc)
    uy = -U0*cp.sin(k*xc)*cp.cos(k*yc)*cp.sin(k*zc)
    uz = cp.zeros(cp.broadcast(xc, yc, zc).shape, cp.float32)
    return cp.stack([cp.broadcast_to(ux, uz.shape),
                     cp.broadcast_to(uy, uz.shape), uz])


def build_coupling():
    lat = get_lattice('D3Q27', cp, dtype=cp.float32)
    region = OverlapRegion(0, COARSE, FBOX, overlap_width=2)
    scaler = LevelScaler(tau_0=0.6, num_levels=2)
    gc = GridCoupling(cp, lat, region, scaler, CubicInterpolation(),
                      filter_level=1)
    taus = [scaler.get_level_units(k).tau for k in range(2)]
    return gc, region, taus


def init_f(region):
    """Global physical f0 for both levels (co-located analytic TGV)."""
    x = cp.arange(COARSE[0], dtype=cp.float32)
    y = cp.arange(COARSE[1], dtype=cp.float32)
    z = cp.arange(COARSE[2], dtype=cp.float32)
    f0c = feq_of_u(tgv_u(x[:, None, None], y[None, :, None], z[None, None, :]))
    ox, oy, oz = region.fine_origin_coarse
    nx, ny, nz = region.fine_shape
    fx = ox + cp.arange(nx, dtype=cp.float32)/2.0
    fy = oy + cp.arange(ny, dtype=cp.float32)/2.0
    fz = oz + cp.arange(nz, dtype=cp.float32)/2.0
    f0f = feq_of_u(tgv_u(fx[:, None, None], fy[None, :, None],
                         fz[None, None, :]))
    return f0c, f0f


class Lvl:
    """One esoteric level buffer (global or rank-local)."""

    def __init__(self, f0_phys, tau):
        self.mem = esoteric_scatter_std(cp, f0_phys, 0)
        self.t = 0
        s = f0_phys.shape[1:]
        n = int(np.prod(s))
        self.dims = s
        self.omega = 1.0/tau
        self.nt = cp.zeros(n, cp.int8)
        self.bc0 = cp.zeros(n, cp.float32)
        self.bcr = cp.ones(n, cp.float32)
        self.rho = cp.empty(s, cp.float32)
        self.u = cp.empty((3,)+s, cp.float32)

    def advance(self, ker):
        nx, ny, nz = self.dims
        ker.launch(self.mem, self.rho, self.u, self.nt, self.bcr,
                   self.bc0, self.bc0, self.bc0, self.omega, 1.0, 1.0,
                   nx, ny, nz, t_step=self.t, force=None)
        self.t += 1


def run_reference(gc, region, taus, f0c, f0f):
    ker = EsotericCumulantKernelD3Q27()
    L0, L1 = Lvl(f0c, taus[0]), Lvl(f0f, taus[1])
    box = gc.coarse_sub_spatial_slices
    for _ in range(NSTEPS):
        fprev = esoteric_gather_std_region(cp, L0.mem, L0.t, box)
        L0.advance(ker)
        L1.advance(ker)
        for half in (True, False):
            f_sub = esoteric_gather_std_region(cp, L0.mem, L0.t, box)
            strips = []
            gc.coarse_to_fine(f_sub, None, is_half_step=half,
                              f_coarse_prev=(fprev if half else None),
                              f_coarse_is_sub=True, f_coarse_prev_is_sub=True,
                              strips_out=strips)
            for sp_sl, vals in strips:
                esoteric_scatter_std_region(cp, L1.mem, vals, L1.t, sp_sl)
            if half:
                L1.advance(ker)
        f_at = esoteric_gather_std_region(
            cp, L1.mem, L1.t, gc.fine_at_coarse_spatial_slices)
        block = gc.fine_to_coarse(f_at, None, f_fine_is_at_coarse=True,
                                  return_excised=True)
        esoteric_scatter_std_region(cp, L0.mem, block, L0.t,
                                    gc.excised_spatial_slices)
    return (esoteric_gather_std(cp, L0.mem, L0.t),
            esoteric_gather_std(cp, L1.mem, L1.t))


def wrap_slice(f, part):
    """Local physical slice (owned +- ghost, periodic wrap along axis)."""
    lo = part.own_start - part.ghost
    idx = (np.arange(lo, lo + part.local_shape[part.axis])
           % f.shape[1 + part.axis])
    return cp.take(f, cp.asarray(idx), axis=1 + part.axis)


def run_split(gc, region, taus, f0c, f0f, axis):
    fdc = region.fine_domain_coarse            # box INCLUDING overlap
    box_lo = (fdc.x_start, fdc.y_start, fdc.z_start)[axis]
    box_hi = (fdc.x_end, fdc.y_end, fdc.z_end)[axis]
    nf_ax = region.fine_shape[axis]
    pcs, pfs = [], []
    for r in range(NR):
        pc = Partition1D(COARSE, NR, r, axis=axis)
        f0_, fc_ = fine_range_from_coarse(pc, box_lo, box_hi, nf_ax)
        pf = Partition1D.from_range(region.fine_shape, NR, r, axis,
                                    f0_, fc_)
        pcs.append(pc); pfs.append(pf)

    lb = LoopbackTransport()
    ker = EsotericCumulantKernelD3Q27()
    L0 = [Lvl(wrap_slice(f0c, pcs[r]), taus[0]) for r in range(NR)]
    L1 = [Lvl(wrap_slice(f0f, pfs[r]), taus[1]) for r in range(NR)]
    ex0 = [HaloBandExchangerV1(pcs[r], lb, cp) for r in range(NR)]
    ex1 = [HaloBandExchangerV1(pfs[r], lb, cp) for r in range(NR)]
    rlc = [RankLocalCouplingV1(gc, pcs[r], pfs[r], cp) for r in range(NR)]

    def sync(exs, lv):
        for r in range(NR):
            exs[r].post(lv[r].mem, lv[r].t)
        for r in range(NR):
            exs[r].complete(lv[r].mem, lv[r].t)

    for _ in range(NSTEPS):
        sync(ex0, L0)
        fprev = [None]*NR
        for r in range(NR):
            reg = rlc[r].coarse_block_region_local()
            if reg is not None:
                fprev[r] = esoteric_gather_std_region(
                    cp, L0[r].mem, L0[r].t, reg)
            L0[r].advance(ker)
        sync(ex1, L1)
        for r in range(NR):
            L1[r].advance(ker)
        sync(ex0, L0)                    # fresh coarse ghosts for C2F read
        for r in range(NR):
            rlc[r].c2f(L0[r].mem, L1[r].mem, L0[r].t, L1[r].t,
                       is_half_step=True, f_prev_sub_loc=fprev[r])
        sync(ex1, L1)
        for r in range(NR):
            L1[r].advance(ker)
        for r in range(NR):
            rlc[r].c2f(L0[r].mem, L1[r].mem, L0[r].t, L1[r].t,
                       is_half_step=False)
        sync(ex1, L1)                    # fresh fine ghosts for F2C read
        for r in range(NR):
            rlc[r].f2c(L1[r].mem, L0[r].mem, L1[r].t, L0[r].t)
    # assemble owned
    def assemble(lvs, parts):
        outs = [esoteric_gather_std(cp, lvs[r].mem, lvs[r].t)
                [(slice(None),) + parts[r].owned_local()] for r in range(NR)
                if parts[r].own_count > 0]
        return cp.concatenate(outs, axis=1 + axis)
    return assemble(L0, pcs), assemble(L1, pfs)


def main():
    gc, region, taus = build_coupling()
    f0c, f0f = init_f(region)
    fC_ref, fF_ref = run_reference(gc, region, taus, f0c, f0f)
    ok = True
    for axis, nm in ((0, "x"), (1, "y"), (2, "z")):
        fC, fF = run_split(gc, region, taus, f0c, f0f, axis)
        bc = bool(cp.all(fC == fC_ref)); bf = bool(cp.all(fF == fF_ref))
        print(f"  axis={nm}: coarse bit={bc} (max|d|={float(cp.max(cp.abs(fC-fC_ref))):.2e})"
              f"  fine bit={bf} (max|d|={float(cp.max(cp.abs(fF-fF_ref))):.2e})")
        ok = ok and bc and bf
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
