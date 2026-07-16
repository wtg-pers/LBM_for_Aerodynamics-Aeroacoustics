"""G-beta0 — ALM kernel abstraction refactor gate (01_design.md #13).

Claims, each at its pre-argued grade:

  [S] SOURCE parity (bit-by-construction): the four generated gaussian CUDA
      sources (spread / spread_radial / spread_aniso / sample) are
      BYTE-IDENTICAL to the pre-refactor hardcoded sources — sha256 pinned
      from baseline_0716. Identical source => identical SASS => bit.
  [G] GOLDEN parity on fixed synthetic inputs (pre-refactor outputs stored
      in data/gbeta0_golden.npz):
        - sampling num/den, radial scales, deficit K: bit (max|d| == 0;
          deterministic paths: fixed-order tree / shape-fixed reductions)
        - spread F_grid (plain/radial/aniso): rel <= 1e-14 (f64 atomicAdd
          order is run-nondeterministic; the ladder grades ALM F_grid
          fp-lastbit — G-M3 measures ~6.5e-17)
  [U] UNIT sanity of the new families (wendland / winckelmans):
        - integral eta dV = 1 (radial quadrature, < 1e-10)
        - second moment = (3/2) eps^2 at eps-equivalence (< 1e-10)
        - wendland support: eta exactly 0 beyond sqrt(7.5) eps
        - winckelmans deficit K = 1/(1+(d/eps)^2)^2 vs the filament
          integral formula (closed form check)
  [E] END-TO-END smoke for the new families on the golden inputs: sampling
      den > 0, F_grid finite, and total deposited force within 2% of the
      marker sum (continuous-normalization discretization error).

Run from repo root:
    python patch_notes/alm_beta_kernel/gates/gbeta0_kernel_abstraction_gate.py
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "gbeta0_golden.npz")

PINNED = {
    "spread": "7eb9297a4e04ba8f48df1daf810924a330eeac67b4b90c6f9c0dffb74eb6d404",
    "spread_radial":
        "6a964b402d65f1570734345e77cef4becb0bb8b24bcfbf53aa57b125b2ff220e",
    "spread_aniso":
        "23c63721facf2ea6d978fd3da9ef980047bd2863df040be731d31c5d64ae1a02",
    "sample": "bc578ca91b9fca8de4ebd0f0395ad38f4f5bd35e480928c85b161d921b61ef1e",
}


def main():
    import cupy as cp
    from src.actuator import spreading as sp
    from src.actuator import interpolation as it
    from src.actuator.alm_kernel import GAUSSIAN, get_alm_kernel

    ok = True

    # ---- [S] source parity -------------------------------------------------
    gen = {
        "spread": sp._bake(sp._SPREAD_KERNEL_TMPL, "{{ETA_ISO}}",
                           GAUSSIAN.cuda_eta_iso),
        "spread_radial": sp._bake(sp._SPREAD_KERNEL_RADIAL_TMPL,
                                  "{{ETA_RADIAL}}", GAUSSIAN.cuda_eta_radial),
        "spread_aniso": sp._bake(sp._SPREAD_KERNEL_ANISO_TMPL,
                                 "{{ETA_ANISO}}", GAUSSIAN.cuda_eta_aniso),
        "sample": it._SAMPLE_KERNEL_TMPL.replace("{{W_SAMPLE}}",
                                                 GAUSSIAN.cuda_w_sample),
    }
    for k, src in gen.items():
        h = hashlib.sha256(src.encode()).hexdigest()
        good = h == PINNED[k]
        ok &= good
        print(f"  [S] {k:14s} {'BYTE-IDENTICAL' if good else 'DIVERGED!'}")

    # ---- [G] golden parity -------------------------------------------------
    g = np.load(GOLD)
    pos, eps, forces = g["pos"], g["eps"], g["forces"]
    u_field = cp.asarray(g["u_field"])
    SHAPE = tuple(u_field.shape[1:])
    active = np.ones(len(eps), bool)
    n_cut = 3.0

    num, den = it._sample_markers_rawkernel(u_field, pos, eps, cp, n_cut)
    d_num = float(np.max(np.abs(num - g["num"])))
    d_den = float(np.max(np.abs(den - g["den"])))

    Fg = sp.spread_forces_to_grid_gpu(SHAPE, pos, forces, eps, cp,
                                      marker_active=active, n_cut=n_cut)
    rt = {"axis": g["axis"], "hub": g["hub"], "r_root": 2.0, "r_tip": 14.0}
    Fg_r = sp.spread_forces_to_grid_gpu(SHAPE, pos, forces, eps, cp,
                                        marker_active=active, n_cut=n_cut,
                                        radial_trunc=rt)
    aniso = {"ec": g["ec"], "et": g["et"], "er": g["er"],
             "eps_c": eps.copy(), "eps_t": 0.6 * eps, "eps_r": 1.3 * eps}
    Fg_a = sp.spread_forces_to_grid_gpu(SHAPE, pos, forces, eps, cp,
                                        marker_active=active, n_cut=n_cut,
                                        aniso=aniso)
    scales = sp.compute_radial_scales_batch(
        cp, SHAPE, pos, eps, active, g["axis"], g["hub"], 2.0, 14.0,
        n_cut=n_cut)
    K = GAUSSIAN.deficit_K(np, g["d"], 3.1)

    def rel(a, b):
        return float(cp.max(cp.abs(a - cp.asarray(b)))
                     / max(float(cp.max(cp.abs(cp.asarray(b)))), 1e-300))

    d_sc = float(cp.max(cp.abs(scales - cp.asarray(g["scales"]))))
    d_K = float(np.max(np.abs(K - g["K_def"])))
    r_f = rel(Fg, g["F_grid"])
    r_fr = rel(Fg_r, g["F_grid_radial"])
    r_fa = rel(Fg_a, g["F_grid_aniso"])
    bit_ok = (d_num == 0.0 and d_den == 0.0 and d_sc == 0.0 and d_K == 0.0)
    rel_ok = max(r_f, r_fr, r_fa) <= 1e-14
    ok &= bit_ok and rel_ok
    print(f"  [G] sample num/den d=({d_num:.1e},{d_den:.1e})  "
          f"scales d={d_sc:.1e}  K d={d_K:.1e}  (claim: bit)")
    print(f"  [G] F_grid rel: plain={r_f:.2e} radial={r_fr:.2e} "
          f"aniso={r_fa:.2e}  (claim: <= 1e-14, atomicAdd order)")

    # ---- [U] unit sanity ---------------------------------------------------
    from scipy import integrate
    for name in ("wendland", "winckelmans"):
        s = get_alm_kernel(name)
        e = 2.7
        # wendland: exact support; winckelmans: algebraic tail -> integrate
        # to infinity (quad handles the decay via variable transformation)
        rmax = s.support_factor(3.0) * e if name == "wendland" else np.inf
        I0 = integrate.quad(
            lambda r: 4 * np.pi * r * r * float(s.np_norm(e))
            * float(s.np_shape(np, np.float64(r * r), e * e)),
            0, rmax, limit=400)[0]
        I2 = integrate.quad(
            lambda r: 4 * np.pi * r ** 4 * float(s.np_norm(e))
            * float(s.np_shape(np, np.float64(r * r), e * e)),
            0, rmax, limit=400)[0]
        u0 = abs(I0 - 1.0)
        u2 = abs(I2 - 1.5 * e * e) / (1.5 * e * e)
        good = u0 < 1e-10 and u2 < 1e-10
        ok &= good
        print(f"  [U] {name:12s} |int eta - 1|={u0:.2e}  "
              f"|M2/(1.5 eps^2)-1|={u2:.2e}  -> {'OK' if good else 'FAIL'}")
    sw = get_alm_kernel("wendland")
    beyond = sw.np_shape(np, (np.sqrt(7.5) * 2.7 * 1.0001) ** 2, 2.7 ** 2)
    ok &= (beyond == 0.0)
    print(f"  [U] wendland beyond-support shape = {beyond} (claim: exactly 0)")
    swk = get_alm_kernel("winckelmans")
    dd = np.linspace(0.1, 10, 33)
    K_wk = swk.deficit_K(np, dd, 2.0)
    F_fil = dd ** 2 * (dd ** 2 + 2 * 4.0) / (dd ** 2 + 4.0) ** 2
    d_wk = float(np.max(np.abs(K_wk - (1.0 - F_fil))))
    ok &= d_wk < 1e-14
    print(f"  [U] winckelmans deficit vs filament closed form: "
          f"max|d|={d_wk:.2e}")

    # ---- [E] end-to-end smoke for the new families -------------------------
    for name in ("wendland", "winckelmans"):
        s = get_alm_kernel(name)
        ncut = s.support_factor(8.0 if name == "winckelmans" else 3.0)
        n2, w2 = it._sample_markers_rawkernel(u_field, pos, eps, cp, ncut,
                                              kernel_spec=s)
        F2 = sp.spread_forces_to_grid_gpu(SHAPE, pos, forces, eps, cp,
                                          marker_active=active, n_cut=ncut,
                                          kernel_spec=s)
        tot_dep = -cp.asnumpy(F2.sum(axis=(1, 2, 3)))
        tot_ref = forces.sum(axis=0)
        cons = float(np.max(np.abs(tot_dep - tot_ref)
                            / np.maximum(np.abs(tot_ref), 1e-30)))
        good = (np.all(w2 > 0) and np.all(np.isfinite(n2))
                and bool(cp.all(cp.isfinite(F2))) and cons < 0.02)
        ok &= good
        print(f"  [E] {name:12s} den>0={bool(np.all(w2 > 0))}  "
              f"force-conservation rel={cons:.2e} (claim < 2e-2)  "
              f"-> {'OK' if good else 'FAIL'}")

    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
