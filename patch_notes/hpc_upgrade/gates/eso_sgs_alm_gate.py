"""Phase 1c gate — SGS (smagorinsky kernel-level, dyn_smag Simulation-level)
and ALM 2-pass (constant body force), esoteric vs standard.

A) smagorinsky: esoteric cumulant(sgs=smag) vs standard fused cumulant(smag)
   + roll streaming — macro fields match to fp32 tolerance; nu_t compared too.
B) dyn_smag end-to-end: two real Simulation objects (standard fused vs
   LBM_ESOTERIC=1), periodic TGV, same sgs_cfg — rho/u match after N steps.
C) ALM 2-pass plumbing: sim.al_model sentinel + monkeypatched
   _compute_body_force = constant Fx field; esoteric-with-alm vs standard
   default path (same monkeypatch) — u must match (Guo correction + source).

Run from repo root:  python patch_notes/hpc_upgrade/gates/eso_sgs_alm_gate.py
"""
import os
os.environ.setdefault("LBM_ESOTERIC", "0")   # set per-case below
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp

_CXl = [0, 1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0, 1,-1,1,-1,1,-1,1,-1]
_CYl = [0, 0,0,1,-1,0,0, 1,1,-1,-1,0,0,0,0, 1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1]
_CZl = [0, 0,0,0,0,1,-1, 0,0,0,0,1,1,-1,-1, 1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1]
CX = cp.asarray(_CXl, cp.float32); CY = cp.asarray(_CYl, cp.float32); CZ = cp.asarray(_CZl, cp.float32)
W = cp.asarray([8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8, cp.float32)
N_AX = 20


def feq(rho, u):
    cu = (CX[:, None, None, None]*u[0] + CY[:, None, None, None]*u[1] + CZ[:, None, None, None]*u[2])
    usq = u[0]**2 + u[1]**2 + u[2]**2
    return (W[:, None, None, None]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)).astype(cp.float32)


def tgv_f0():
    x = cp.arange(N_AX, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    k = 2*np.pi/N_AX
    u0 = cp.stack([0.05*cp.cos(k*X)*cp.sin(k*Y)*cp.sin(k*Z),
                   -0.05*cp.sin(k*X)*cp.cos(k*Y)*cp.sin(k*Z),
                   cp.zeros((N_AX,)*3, cp.float32)])
    return feq(cp.ones((N_AX,)*3, cp.float32), u0)


def case_A_smag(nsteps=20, Cs=0.17):
    from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
    from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
    from src.kernels.esoteric_d3q27 import esoteric_scatter_std
    Nn = N_AX**3
    f0 = tgv_f0()
    # reference
    coll = CumulantCollideKernelD3Q27(sgs_model="smagorinsky")
    f = f0.copy(); fp = cp.empty_like(f)
    rho = cp.empty((N_AX,)*3, cp.float32); u = cp.empty((3,)+(N_AX,)*3, cp.float32)
    nut_r = cp.empty(Nn, cp.float32)
    recs = []
    for _ in range(nsteps):
        coll.launch(f, fp, rho, u, None, 1.6, 1.0, 1.0, Nn, Cs=Cs, nu_t_out=nut_r)
        recs.append((rho.copy(), u.copy(), nut_r.copy()))
        fnew = cp.empty_like(f)
        for q in range(27):
            fnew[q] = cp.roll(fp[q], (_CXl[q], _CYl[q], _CZl[q]), axis=(0, 1, 2))
        f = fnew
    # esoteric
    mem = esoteric_scatter_std(cp, f0, 0)
    nt = cp.zeros(Nn, cp.int8); bc = cp.zeros(Nn, cp.float32); bcr = cp.ones(Nn, cp.float32)
    nut_e = cp.empty(Nn, cp.float32)
    ker = EsotericCumulantKernelD3Q27(sgs_model="smagorinsky")
    dmax = numax = 0.0
    for k in range(nsteps):
        ker.launch(mem, rho, u, nt, bcr, bc, bc, bc, 1.6, 1.0, 1.0,
                   N_AX, N_AX, N_AX, t_step=k, force=None, Cs=Cs, nu_t_out=nut_e)
        rr, ur, nr = recs[k]
        dmax = max(dmax, float(cp.max(cp.abs(rr - rho))), float(cp.max(cp.abs(ur - u))))
        numax = max(numax, float(cp.max(cp.abs(nr - nut_e))))
    ok = dmax < 5e-4 and numax < 5e-5
    print(f"  [A smag]     max|d(rho,u)|={dmax:.3e}  max|d nu_t|={numax:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def _build_sim(coll_kind, sgs_cfg=None):
    from src.lattice import get_lattice
    from src.macroscopic.compute import Macroscopic
    from src.collision.cumulant import CumulantCollision
    from src.streaming.stream import StreamingPull
    from src.boundary.domain_bc_manager import DomainBCManager
    from src.solver.simulation import Simulation
    S = (N_AX,)*3
    lat = get_lattice('D3Q27', cp, dtype=cp.float32)
    coll = CumulantCollision(cp, lat)
    bcs = {f: {"location": f, "method": "periodic"}
           for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    sim = Simulation(cp, Macroscopic(cp, lat), coll,
                     StreamingPull(cp, lat, S),
                     DomainBCManager(cp, lat, bcs, S, verbose=False),
                     tau=1.0/1.6, domain_shape=S, sgs_cfg=sgs_cfg)
    return sim, coll


def case_B_dynsmag(nsteps=15):
    from src.turbulence.sgs import parse_sgs_config
    sgs = parse_sgs_config({"model": "dyn_smag"})
    f0 = tgv_f0()
    outs = {}
    for label, env in (("std", "0"), ("eso", "1")):
        os.environ["LBM_ESOTERIC"] = env
        sim, coll = _build_sim("cumulant", sgs_cfg=sgs)
        sim.set_distribution(f0.copy())
        for _ in range(nsteps):
            sim.advance()
        outs[label] = (sim.rho.copy(), sim.u.copy(),
                       getattr(sim, "_use_esoteric", False))
    drho = float(cp.max(cp.abs(outs["std"][0] - outs["eso"][0])))
    du = float(cp.max(cp.abs(outs["std"][1] - outs["eso"][1])))
    ok = outs["eso"][2] and (not outs["std"][2]) and drho < 5e-4 and du < 5e-4
    print(f"  [B dyn_smag] eso_on={outs['eso'][2]} max|drho|={drho:.3e} max|du|={du:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def case_C_alm(nsteps=15, Fx=1e-5):
    f0 = tgv_f0()
    S = (N_AX,)*3
    Ffield = cp.zeros((3,) + S, dtype=cp.float32)
    Ffield[0] = Fx
    outs = {}
    for label, env in (("std", "0"), ("eso", "1")):
        os.environ["LBM_ESOTERIC"] = env
        sim, coll = _build_sim("cumulant")
        sim.set_distribution(f0.copy())
        sim.al_model = object()                       # dispatch sentinel
        sim._compute_body_force = lambda u, rho: Ffield
        for _ in range(nsteps):
            sim.advance()
        outs[label] = (sim.rho.copy(), sim.u.copy(),
                       getattr(sim, "_use_esoteric", False))
    drho = float(cp.max(cp.abs(outs["std"][0] - outs["eso"][0])))
    du = float(cp.max(cp.abs(outs["std"][1] - outs["eso"][1])))
    # sanity: force actually accelerated the flow
    ux_mean = float(cp.mean(outs["eso"][1][0]))
    ok = outs["eso"][2] and drho < 5e-4 and du < 5e-4 and ux_mean > 0.5*nsteps*Fx
    print(f"  [C alm-2pass] eso_on={outs['eso'][2]} max|drho|={drho:.3e} max|du|={du:.3e} "
          f"<ux>={ux_mean:.2e} (>= {0.5*nsteps*Fx:.1e}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"[eso_sgs_alm_gate] {N_AX}^3 periodic TGV")
    a = case_A_smag()
    b = case_B_dynsmag()
    c = case_C_alm()
    print("  RESULT:", "PASS" if (a and b and c) else "FAIL")
    sys.exit(0 if (a and b and c) else 1)


if __name__ == "__main__":
    main()
