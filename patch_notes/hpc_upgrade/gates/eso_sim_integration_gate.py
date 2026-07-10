"""Phase 1a gate (integration) — Esoteric Pull through the real Simulation.

Confirms the wiring: LBM_ESOTERIC=1 -> set_distribution() calls _init_esoteric
-> advance() dispatches to _advance_esoteric, on a periodic BGK box (all FLUID).
Checks _use_esoteric flag, mass conservation, and NaN-freeness.

Run from repo root on a CUDA box:
    python patch_notes/hpc_upgrade/gates/eso_sim_integration_gate.py
"""
import os
os.environ["LBM_ESOTERIC"] = "1"
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.lattice import get_lattice
from src.macroscopic.compute import Macroscopic
from src.collision.bgk import BGKCollision
from src.collision.cumulant import CumulantCollision
from src.streaming.stream import StreamingPull
from src.boundary.domain_bc_manager import DomainBCManager
from src.solver.simulation import Simulation

SHAPE = (24, 24, 24)
NSTEPS = 30
U0 = 0.05


def run_case(coll_cls, label, want_cumulant):
    lat = get_lattice('D3Q27', cp, dtype=cp.float32)
    macro = Macroscopic(cp, lat)
    coll = coll_cls(cp, lat)
    stream = StreamingPull(cp, lat, SHAPE)
    bcs = {f: {"location": f, "method": "periodic"}
           for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    bcm = DomainBCManager(cp, lat, bcs, SHAPE, verbose=False)
    sim = Simulation(cp, macro, coll, stream, bcm, tau=1.0, domain_shape=SHAPE)

    Nx, Ny, Nz = SHAPE
    x = cp.arange(Nx, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    k = 2 * np.pi / Nx
    ux = U0 * cp.cos(k*X) * cp.sin(k*Y) * cp.sin(k*Z)
    uy = -U0 * cp.sin(k*X) * cp.cos(k*Y) * cp.sin(k*Z)
    uz = cp.zeros_like(ux)
    u0 = cp.stack([ux, uy, uz])
    f0 = coll.compute_equilibrium(cp.ones(SHAPE, cp.float32), u0).astype(cp.float32)

    sim.set_distribution(f0)
    m0 = float(cp.sum(sim.f))
    for _ in range(NSTEPS):
        sim.advance()
    m1 = float(cp.sum(sim.f))

    nan = bool(cp.any(~cp.isfinite(sim.rho))) or bool(cp.any(~cp.isfinite(sim.u)))
    drift = (m1 - m0) / m0
    ok = (sim._use_esoteric and sim._esoteric_is_cumulant == want_cumulant
          and not nan and abs(drift) < 1e-5 and sim.f_post is None
          and sim._esoteric_step == NSTEPS)
    print(f"  [{label}] use_eso={sim._use_esoteric} is_cumulant={sim._esoteric_is_cumulant} "
          f"f_post={sim.f_post} drift={drift:+.2e} |u|max={float(cp.max(cp.abs(sim.u))):.4f} "
          f"NaN={nan} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"[eso_sim_integration_gate] {SHAPE}, {NSTEPS} steps, periodic, LBM_ESOTERIC=1")
    ok_bgk = run_case(BGKCollision, "BGK", False)
    ok_cum = run_case(CumulantCollision, "cumulant", True)
    print("  RESULT:", "PASS" if (ok_bgk and ok_cum) else "FAIL")
    sys.exit(0 if (ok_bgk and ok_cum) else 1)


if __name__ == "__main__":
    main()
