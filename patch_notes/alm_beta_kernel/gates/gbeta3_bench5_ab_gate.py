"""G-beta3 — bench5 A/B: gaussian vs wendland (01_design.md #14.4).

Two production-shaped pairs, both esoteric (production path), NSTEPS
coarse steps each:

  pure : bench5_purealm_m3      vs bench5_purealm_wendland
         (kernel effect on spreading + sampling only)
  archB: bench5_archb_m3        vs bench5_archb_wendland
         (FULL beta chain: + radial-trunc renormalization with
          kernel-consistent scales + kleine-straight solve on the DERIVED
          Wendland deficit K — 03_correction_derivation.md)

PHYSICS DIFFERENCES ARE EXPECTED (that is the experiment); the gate
asserts sanity, not equality:

  [K] the kernel actually switched: model.n_cut == sqrt(7.5) (eps-equiv
      support) on wendland runs, == 3.0 (config gaussian_cutoff) on
      gaussian runs.
  [A] stability: no NaN in the spread force trace or fields, all 4 runs.
  [B] conservation (wendland runs): total deposited force == -sum(marker
      F_global) to rel < 1e-3 (continuous-normalization discretization;
      G-beta0 [E] measured ~2e-5 synthetic).
  [C] A/B thrust-trace tail: median rel diff < 0.15 (sanity band at
      eps-equivalence; REPORTED, not a physics claim).
  [D] spanwise F_n (blade 0, final step): corr(gaussian, wendland) > 0.9
      (shape sanity) + REPORT of the peak-region (r/R 0.85-0.97) mean rel
      diff — the quantity the D40 matrix will judge.

Run from repo root:
    python patch_notes/alm_beta_kernel/gates/gbeta3_bench5_ab_gate.py
"""
import gc
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

NSTEPS = int(os.environ.get("SMOKE_STEPS", "20"))
SQRT_7_5 = float(np.sqrt(7.5))


def run(cfg_path):
    import cupy as cp
    os.environ["LBM_ESOTERIC"] = "1"
    for k in ("MLG_CUDA_GRAPH", "MLG_PROFILE", "MLG_NVTX"):
        os.environ.pop(k, None)
    sys.argv = ["gbeta3", "--config", cfg_path,
                "--gpu", "0", "--no-vtk", "--no-force"]
    from src.io.args_parser import parse_args
    from src.solver.setup import SimulationSetup
    from src.solver.initializer import SolverInitializer
    a = parse_args()
    s = SimulationSetup(a)
    s.start_log_capture()
    mlg = s.build_simulation()
    SolverInitializer(s).initialize(mlg, a)
    s.stop_log_capture()
    mlg._graph_enabled = False

    alm_lev = None
    for k in range(mlg.num_levels):
        if getattr(mlg.get_level(k), "al_model", None) is not None:
            alm_lev = mlg.get_level(k)
    assert alm_lev is not None
    model = alm_lev.al_model

    F_hist, nan = [], False
    for _ in range(NSTEPS):
        mlg.advance()
        Fg = model._F_grid
        F = [float(cp.sum(Fg[d])) for d in range(3)]
        F_hist.append(F)
        nan = nan or any(not np.isfinite(v) for v in F)
    cp.cuda.runtime.deviceSynchronize()
    nan = nan or bool(cp.any(~cp.isfinite(alm_lev.rho)))

    n0 = len(model.rotor.blades[0].marker_r)
    out = {
        "kernel": model._alm_kernel.name,
        "n_cut": float(model.n_cut),
        "F": np.array(F_hist),
        "nan": bool(nan),
        "dep": np.array([float(cp.sum(model._F_grid[d])) for d in range(3)]),
        # deposited force = -ramp * sum(markers): the startup ramp scales
        # _F_grid AFTER spreading (actuator_line step, ramp=rev 1), while
        # _last_forces_global stores the UNramped marker forces.
        "ramp": float(model._ramp_factor),
        "mark": model._last_forces_global.sum(axis=0).astype(np.float64),
        "r_R": (model.rotor.blades[0].marker_r
                / float(model.rotor.radius)).copy(),
        "F_n": np.asarray(model._last_bem_result.F_n[:n0], dtype=np.float64),
    }
    del mlg, s, alm_lev, model, Fg
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return out


def main():
    print(f"[gbeta3_bench5_ab] gaussian vs wendland, {NSTEPS} coarse steps, "
          "esoteric production path")
    pairs = {
        "pure": ("configs/hpc_bench/bench5_purealm_m3.py",
                 "configs/hpc_bench/bench5_purealm_wendland.py"),
        "archB": ("configs/hpc_bench/bench5_archb_m3.py",
                  "configs/hpc_bench/bench5_archb_wendland.py"),
    }
    ok = True
    for tag, (cfg_g, cfg_w) in pairs.items():
        g = run(cfg_g)
        w = run(cfg_w)

        # [K] kernel actually switched
        k_ok = (g["kernel"] == "gaussian" and abs(g["n_cut"] - 3.0) < 1e-12
                and w["kernel"] == "wendland"
                and abs(w["n_cut"] - SQRT_7_5) < 1e-12)
        # [A] stability
        a_ok = (not g["nan"]) and (not w["nan"])
        # [B] conservation on the wendland run (deposited = -ramp * markers)
        rmark = w["ramp"] * w["mark"]
        ref = np.maximum(np.abs(rmark), 1e-30)
        cons = float(np.max(np.abs(w["dep"] + rmark) / ref))
        b_ok = cons < 1e-3
        # [C] thrust-trace tail
        mg = np.linalg.norm(g["F"], axis=1)
        mw = np.linalg.norm(w["F"], axis=1)
        tail = slice(NSTEPS // 4, None)
        rel = np.abs(mw[tail] - mg[tail]) / np.maximum(mg[tail], 1e-30)
        med, mx = float(np.median(rel)), float(np.max(rel))
        c_ok = med < 0.15 and mg[-1] > 0
        # [D] spanwise F_n
        corr = float(np.corrcoef(g["F_n"], w["F_n"])[0, 1])
        pk = (g["r_R"] >= 0.85) & (g["r_R"] <= 0.97)
        pk_rel = float(np.mean(
            (w["F_n"][pk] - g["F_n"][pk])
            / np.maximum(np.abs(g["F_n"][pk]), 1e-30)))
        # sanity threshold 0.9: shape similarity, not equality — measured
        # 0.965 (archB) / see log (pure) at eps-equivalence, 20-step ramp
        d_ok = corr > 0.9 and np.all(np.isfinite(w["F_n"]))

        ok &= k_ok and a_ok and b_ok and c_ok and d_ok
        print(f"  [{tag}]")
        print(f"    [K] n_cut g={g['n_cut']:.4f} w={w['n_cut']:.4f} "
              f"-> {'OK' if k_ok else 'FAIL'}")
        print(f"    [A] NaN g={g['nan']} w={w['nan']} "
              f"-> {'OK' if a_ok else 'FAIL'}")
        print(f"    [B] wendland conservation rel={cons:.2e} (<1e-3) "
              f"-> {'OK' if b_ok else 'FAIL'}")
        print(f"    [C] thrust tail rel diff median={med:.3e} max={mx:.3e} "
              f"(sanity <0.15) -> {'OK' if c_ok else 'FAIL'}")
        print(f"    [D] spanwise corr={corr:.5f} (>0.9)  peak-region "
              f"(0.85-0.97R) mean rel dF_n={pk_rel:+.3e}  "
              f"-> {'OK' if d_ok else 'FAIL'}")

    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
