"""R3-2 twin gate — eso_mem_force vs mem_force_d3q27 (standard path).

CLOSES THE 3rd-review gap: eso_mem_force was the only new physics-producing
code in the track without an equivalence gate. What was verified before was
decomposition-invariance (1-rank == 2-rank), NOT that the force extraction
itself matches the standard-path convention (a wrong slot rule can be
rank-invariant too). This gate compares the two PRODUCTION extraction
implementations on bit-identical states.

Setup: one small periodic box + solid sphere (cell-center PIP), cumulant,
built twice through the real Simulation — standard two-buffer path
(HWBBKernelD3Q27 wall BC + f_post) and Esoteric Pull path (implicit HWBB).
Both start from the same f0.

Per sampled step, four claims at their pre-argued grades:

  [P] field parity (precondition):  gather_std(eso.mem, t) vs std.f on
      solid-source-masked entries + rho/u on fluid cells. Grade:
      fp-lastbit ACCUMULATION (< 1e-5 after ~24 steps) — the standard
      fused cumulant and the esoteric cumulant are different codegens of
      the same collision, so std-vs-eso was never a bit pair (matches the
      eso_cumulant_equiv_gate grade). An implicit-bounce scheme error
      shows as O(f_i - f_opp) ~ 1e-2 here (and did: this gate caught the
      2-step-delayed bounce + the IC seeding gap).
  [A1] extraction twin ON THE SAME STATE (THE claim):  reconstruct the
      completed step's outgoing populations from esoteric memory with the
      solid-agnostic gather + roll(-c_q) (g_q(x) = gather[q](x + c_q);
      at toward-solid links this recovers the deposit — pure streaming
      relation, independent of the mem-force slot table), then compare a
      fixed-order f64 host sum of the standard MEM formula against the
      production eso_mem_force on that same memory:
      ||dF|| <= 1e-11 * ||F||  (f64-lastbit class). A slot/link-set
      convention error shows up as O(2f) ~ 1e-1 per link.
  [A2] cross-path consistency:  eso_mem_force vs the f64 host reference
      from the STANDARD path's own (f_post, needs_bounce). Bounded by the
      [P] parity noise: ||dF|| <= 2 * par * n_links.
  [B] standard-kernel accumulation grade:  production MEMForceKernelD3Q27
      (f32 atomicAdd, --use_fast_math) vs the same f64 host reference.
      Bounded by f32 accumulation:  ||dF|| <= 32 * eps32 * sum|terms|.
      DIAGNOSTIC tier by declaration -- this leg just pins the grade.

Run from repo root on a CUDA box:
    python patch_notes/hpc_upgrade/gates/eso_mem_force_twin_gate.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

SHAPE = (32, 32, 32)
NSTEPS = 24
U0 = 0.05
SPHERE_C = (13.5, 15.5, 17.5)     # off-center + off-lattice: no symmetry
SPHERE_R = 5.3
EPS32 = np.float32(1.0) * 2.0 ** -23


def build_sim(esoteric: bool):
    """One production Simulation (cumulant, periodic box, sphere HWBB)."""
    os.environ["LBM_ESOTERIC"] = "1" if esoteric else "0"
    import cupy as cp
    from src.lattice import get_lattice
    from src.macroscopic.compute import Macroscopic
    from src.collision.cumulant import CumulantCollision
    from src.streaming.stream import StreamingPull
    from src.boundary.domain_bc_manager import DomainBCManager
    from src.boundary.wall import HalfwayBounceBack
    from src.solver.simulation import Simulation

    lat = get_lattice('D3Q27', cp, dtype=cp.float32)
    macro = Macroscopic(cp, lat)
    coll = CumulantCollision(cp, lat)
    stream = StreamingPull(cp, lat, SHAPE)
    bcs = {f: {"location": f, "method": "periodic"}
           for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    bcm = DomainBCManager(cp, lat, bcs, SHAPE, verbose=False)

    # cell-center PIP sphere (the only validated mask rule in this repo)
    Nx, Ny, Nz = SHAPE
    ii, jj, kk = cp.meshgrid(cp.arange(Nx, dtype=cp.float64),
                             cp.arange(Ny, dtype=cp.float64),
                             cp.arange(Nz, dtype=cp.float64), indexing='ij')
    solid = ((ii - SPHERE_C[0]) ** 2 + (jj - SPHERE_C[1]) ** 2
             + (kk - SPHERE_C[2]) ** 2) <= SPHERE_R ** 2
    wall = HalfwayBounceBack(cp, lat, solid)

    sim = Simulation(cp, macro, coll, stream, bcm, tau=1.0,
                     domain_shape=SHAPE, obstacle_bc=wall)

    # asymmetric IC: uniform inflow + shear perturbations (all 3 F components
    # nonzero so the comparison is conditioned in every axis)
    x = cp.arange(Nx, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    k1 = 2 * np.pi / Nx
    ux = U0 * (1.0 + 0.2 * cp.sin(k1 * Z) * cp.cos(2 * k1 * Y))
    uy = 0.3 * U0 * cp.sin(k1 * X + 0.7)
    uz = 0.2 * U0 * cp.cos(k1 * Y + 1.3) * cp.sin(k1 * X)
    u0 = cp.stack([ux, uy, uz]).astype(cp.float32)
    f0 = coll.compute_equilibrium(
        cp.ones(SHAPE, cp.float32), u0).astype(cp.float32)
    sim.set_distribution(f0.copy())
    return sim, wall, solid


def host_ref_f64(f_post_np: np.ndarray, nb_np: np.ndarray, c: np.ndarray):
    """Fixed-order f64 host sum of the standard MEM formula (the shared
    reference both production kernels are judged against)."""
    F = np.zeros(3, np.float64)
    sum_abs = 0.0
    for q in range(1, 27):
        s = float(np.sum(f_post_np[q][nb_np[q]], dtype=np.float64))
        a = float(np.sum(np.abs(f_post_np[q][nb_np[q]]), dtype=np.float64))
        for d in range(3):
            F[d] += 2.0 * float(c[d, q]) * s
        sum_abs += 2.0 * a * float(np.max(np.abs(c[:, q])))
    return F, sum_abs


def main():
    import cupy as cp
    from src.kernels.esoteric_d3q27 import eso_mem_force, esoteric_gather_std

    print(f"[eso_mem_force_twin_gate] {SHAPE} periodic + sphere r={SPHERE_R}, "
          f"{NSTEPS} steps, cumulant tau=1.0")
    sim_std, wall_std, solid = build_sim(esoteric=False)
    sim_eso, _, _ = build_sim(esoteric=True)
    assert not sim_std._use_esoteric and sim_eso._use_esoteric
    n_solid = int(cp.sum(solid))
    n_links = int(cp.sum(wall_std.needs_bounce))
    print(f"  solid cells={n_solid}  bounce links={n_links}")

    c_np = cp.asnumpy(wall_std.c)
    nb_np = cp.asnumpy(wall_std.needs_bounce)
    opp_np = cp.asnumpy(wall_std.opp)
    fluid3 = ~cp.asnumpy(solid)
    Nx, Ny, Nz = SHAPE
    full_bounds = [(0, Nx), (0, Ny), (0, Nz)]
    # f-level compare mask: esoteric_gather_std is a solid-agnostic pure
    # permutation, so entries whose SOURCE cell is solid (x - c_q solid,
    # i.e. needs_bounce[opp(q)][x]) map to the stale normal slot the fixed
    # kernel no longer reads — exclude them from the f compare. The bounced
    # populations are still fully checked through rho/u (macro consumes
    # every loaded fhn, bounce branches included).
    fmask = np.empty((27,) + SHAPE, bool)
    for q in range(27):
        fmask[q] = fluid3 & ~nb_np[int(opp_np[q])]

    ok = True
    max_par = 0.0
    max_relA1 = 0.0
    max_ratA2 = 0.0
    max_ratB = 0.0
    for step in range(1, NSTEPS + 1):
        sim_std.advance()
        sim_eso.advance()

        # [P] field parity (bit): f on solid-source-masked entries + macro
        f_eso_std = esoteric_gather_std(
            cp, sim_eso.f.reshape(27, Nx, Ny, Nz), sim_eso._esoteric_step)
        df = np.abs(cp.asnumpy(
            f_eso_std - sim_std.f.reshape(27, Nx, Ny, Nz)))
        par = float(np.max(df[fmask]))
        dr = np.abs(cp.asnumpy(sim_eso.rho - sim_std.rho))
        du = np.abs(cp.asnumpy(sim_eso.u - sim_std.u))
        par = max(par, float(np.max(dr[fluid3])),
                  float(np.max(du[:, fluid3])))
        max_par = max(max_par, par)

        # f64 host reference from the standard path's own state
        f_post_np = cp.asnumpy(sim_std.f_post).reshape(27, Nx, Ny, Nz)
        F_ref, sum_abs = host_ref_f64(f_post_np, nb_np, c_np)
        F_norm = float(np.linalg.norm(F_ref))

        # production eso_mem_force (f64)
        F_eso = np.asarray(eso_mem_force(
            cp, sim_eso.f.reshape(27, Nx, Ny, Nz),
            sim_eso._eso_node_type, sim_eso._esoteric_step, full_bounds),
            dtype=np.float64)

        # [A1] SAME-STATE twin: outgoing populations recovered from eso
        # memory by the streaming relation g_q(x) = gather[q](x + c_q)
        # (solid-agnostic primitives only), fed to the standard formula.
        g_np = cp.asnumpy(f_eso_std)
        f_post_eso = np.empty_like(g_np)
        for q in range(27):
            f_post_eso[q] = np.roll(
                g_np[q], shift=(-int(c_np[0, q]), -int(c_np[1, q]),
                                -int(c_np[2, q])), axis=(0, 1, 2))
        F_eso_ref, _ = host_ref_f64(f_post_eso, nb_np, c_np)
        relA1 = (float(np.linalg.norm(F_eso - F_eso_ref))
                 / max(float(np.linalg.norm(F_eso_ref)), 1e-30))
        max_relA1 = max(max_relA1, relA1)

        # [A2] cross-path: bounded by the field-parity noise
        bandA2 = 2.0 * max(par, 1e-30) * n_links
        dA2 = float(np.linalg.norm(F_eso - F_ref))
        max_ratA2 = max(max_ratA2, dA2 / bandA2)

        # [B] production standard kernel (f32 atomicAdd)
        F_std = np.asarray(wall_std_force(sim_std, wall_std), np.float64)
        bandB = 32.0 * float(EPS32) * max(sum_abs, 1e-30)
        dB = float(np.linalg.norm(F_std - F_ref))
        max_ratB = max(max_ratB, dB / bandB)

        step_ok = (par < 1e-5 and relA1 < 1e-11 and dA2 <= bandA2
                   and dB <= bandB)
        ok = ok and step_ok
        if step % 4 == 0 or not step_ok:
            print(f"  step {step:3d}: par={par:.1e}  "
                  f"F_ref=({F_ref[0]:+.6e},{F_ref[1]:+.6e},{F_ref[2]:+.6e})  "
                  f"relA1={relA1:.2e}  dA2/band={dA2 / bandA2:.2e}  "
                  f"dB/band={dB / bandB:.2e}"
                  f"{'' if step_ok else '  <-- FAIL'}")

    print(f"  [P]  field parity max = {max_par:.3e}  "
          f"(claim: fp-lastbit accumulation, < 1e-5)")
    print(f"  [A1] eso_mem_force vs f64 ref on SAME state  max rel = "
          f"{max_relA1:.3e}  (claim: f64-lastbit class, < 1e-11)")
    print(f"  [A2] eso_mem_force vs std-state f64 ref  max |dF|/band = "
          f"{max_ratA2:.3e}  (claim: parity-noise band, <= 1)")
    print(f"  [B]  mem_force_d3q27 vs f64 ref  max |dF|/band = {max_ratB:.3e}"
          f"  (claim: f32-accumulation band, <= 1)")
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def wall_std_force(sim_std, wall_std):
    """Production standard extraction, exactly as ForceManager drives it:
    MomentumExchangeForce.compute(f_post) -> MEMForceKernelD3Q27."""
    import cupy as cp
    from src.lattice import get_lattice
    from src.utilities.force_calculator import MomentumExchangeForce
    global _MEM_STD
    try:
        _MEM_STD
    except NameError:
        lat = get_lattice('D3Q27', cp, dtype=cp.float32)
        _MEM_STD = MomentumExchangeForce(cp, lat, wall_std.solid_mask,
                                         wall_bc=wall_std)
    return _MEM_STD.compute(sim_std.f_post)


if __name__ == "__main__":
    main()
