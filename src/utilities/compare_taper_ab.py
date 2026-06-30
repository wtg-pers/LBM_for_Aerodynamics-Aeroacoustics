#!/usr/bin/env python
"""
ALM Stage B (ε 팁 테이퍼) A/B 비교 — 한 번에 실행되는 스크립트.

baseline(A, epsilon_mode=default)과 taper(B, epsilon_mode=tip_taper) 두 런 결과를
받아서 다음을 한 번에 수행:
  1) C_T 비교 (rotor_performance.csv, 마지막 N rev 평균) vs 실험값
  2) spanwise 분포 비교 (spanwise_post.compute 재사용; eps_lu/phi/M2CL/F_n)
  3) A/B 오버레이 플롯 PNG 저장
  4) 팁 영역(r/R≥0.9) 수치 요약 출력

사용법 (리눅스 터미널에서 바로):
    # 기본 폴더명(M0.877 light)으로:
    python src/utilities/compare_taper_ab.py

    # 폴더/옵션 지정:
    python src/utilities/compare_taper_ab.py \
        --A result_ct_t08.0_M877_mlg4_D32_light \
        --B result_ct_t08.0_M877_mlg4_D32_light_taper \
        --mtip 0.877 --avg-revs 3 --exp-ct 0.00473

CWD 무관하게 동작 (스크립트가 repo 루트를 sys.path에 추가). result 폴더 경로는
현재 작업 디렉터리(보통 repo 루트) 기준 상대경로 또는 절대경로 모두 가능.
"""
import argparse
import os
import sys

# --- repo 루트를 import 경로에 추가 (CWD 무관 동작) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas가 필요합니다:  pip install pandas")

from src.utilities.spanwise_post import compute  # noqa: E402


def ct_from_perf(result_dir, avg_revs):
    """rotor_performance.csv에서 마지막 avg_revs rev 평균 → C_T(rotorcraft), T_lu."""
    path = os.path.join(result_dir, "csv", "rotor_performance.csv")
    if not os.path.isfile(path):
        return None, None, f"없음: {path}"
    df = pd.read_csv(path)
    if df.empty or "revolutions" not in df.columns:
        return None, None, f"데이터 없음: {path}"
    w = df[df["revolutions"] >= df["revolutions"].max() - avg_revs]
    if w.empty:
        w = df
    T = w["thrust_lu"].mean()
    # C_T(rotorcraft) = thrust_lu / (rho_ref * area_lu * tip_speed_lu^2)
    #                   (output_manager.py:619)
    denom = w["rho_ref"].mean() * w["area_lu"].mean() * w["tip_speed_lu"].mean() ** 2
    ct = T / denom if denom else float("nan")
    return ct, T, None


def geom_eps(result_dir):
    """blade_geometry.csv(설정 시점 기록)에서 (r_R, epsilon_lu, chord_lu) 반환 or None.

    per-step blade CSV의 eps_lu가 없는 (Stage B 이전) 런에서도 테이퍼 적용 여부를
    확인할 수 있는 경로. 이 파일은 setup 시점에 항상 기록됨.
    """
    path = os.path.join(result_dir, "csv", "blade_diagnostics", "blade_geometry.csv")
    if not os.path.isfile(path):
        return None
    g = pd.read_csv(path)
    col = "epsilon_lu" if "epsilon_lu" in g.columns else (
        "eps_lu" if "eps_lu" in g.columns else None)
    if col is None or "r_R" not in g.columns:
        return None
    g = g.sort_values("r_R")
    return g["r_R"].values, g[col].values


def tip_summary(df, r_min=0.9):
    """r/R >= r_min 영역의 대표값(팁 마커)을 dict로."""
    tip = df[df["r_R"] >= r_min]
    if tip.empty:
        tip = df.tail(1)
    last = df.iloc[-1]  # 최외측 마커
    out = {"r_R_tip": float(last["r_R"])}
    for c in ("eps_lu", "phi", "alpha", "M2CL", "CL", "F_n"):
        if c in df.columns:
            out[c] = float(last[c])
    return out


def main():
    ap = argparse.ArgumentParser(
        description="ALM Stage B ε-taper A/B 비교 (C_T + spanwise + 오버레이 플롯)")
    ap.add_argument("--A", default="result_ct_t08.0_M877_mlg4_D32_light",
                    help="baseline 결과 폴더 (epsilon_mode=default)")
    ap.add_argument("--B", default="result_ct_t08.0_M877_mlg4_D32_light_taper",
                    help="taper 결과 폴더 (epsilon_mode=tip_taper)")
    ap.add_argument("--mtip", type=float, default=0.877,
                    help="팁 Mach (local Mach + M^2*CL 활성화)")
    ap.add_argument("--avg-revs", type=float, default=3.0,
                    help="평균낼 마지막 회전수")
    ap.add_argument("--umax", type=float, default=0.1, help="u_max_lu (scaling)")
    ap.add_argument("--exp-ct", type=float, default=0.00473,
                    help="실험 C_T 기준값 (CT M0.877 = 0.00473)")
    ap.add_argument("--outdir", default="aeromechanics_workshop/temp_results",
                    help="플롯/CSV 출력 폴더")
    ap.add_argument("--la", default="A base", help="A 라벨 (출력/플롯)")
    ap.add_argument("--lb", default="B taper", help="B 라벨 (출력/플롯)")
    args = ap.parse_args()
    LA, LB = ("%-7s" % args.la)[:7], ("%-7s" % args.lb)[:7]

    os.makedirs(args.outdir, exist_ok=True)

    # ── 1) C_T 비교 ───────────────────────────────────────────────
    print("=" * 64)
    print(" C_T 비교  (마지막 %.0f rev 평균, rotorcraft 정의)" % args.avg_revs)
    print("=" * 64)
    ct_a, T_a, err_a = ct_from_perf(args.A, args.avg_revs)
    ct_b, T_b, err_b = ct_from_perf(args.B, args.avg_revs)
    for tag, ct, T, err in [(LA, ct_a, T_a, err_a),
                            (LB, ct_b, T_b, err_b)]:
        if err:
            print("  %s  : %s" % (tag, err))
        else:
            dev = (ct - args.exp_ct) / args.exp_ct * 100.0
            print("  %s  : C_T=%.5f  (T_lu=%.4f, exp 대비 %+.1f%%)"
                  % (tag, ct, T, dev))
    print("  %s  : C_T=%.5f" % ("exp    ", args.exp_ct))
    if ct_a and ct_b:
        print("  ΔC_T(B−A) = %+.5f  (%+.1f%%p of exp)"
              % (ct_b - ct_a, (ct_b - ct_a) / args.exp_ct * 100.0))
    print()

    # ── 1b) ε 테이퍼 적용 확인 (blade_geometry.csv) ───────────────
    print("=" * 64)
    print(" ε 테이퍼 적용 확인  (blade_geometry.csv: inboard vs 팁)")
    print("=" * 64)
    for tag, d in [(LA, args.A), (LB, args.B)]:
        ge = geom_eps(d)
        if ge is None:
            print("  %s  : epsilon_lu 없음 (구버전 런 or 파일 부재)" % tag)
            continue
        rR, eps = ge
        inb = eps[rR < 0.7]
        inb_v = float(inb.mean()) if len(inb) else float(eps[0])
        print("  %s  : inboard(r/R<0.7)≈%.3f   팁≈%.3f   (테이퍼면 팁<inboard)"
              % (tag, inb_v, float(eps[-1])))
    print()

    # ── 2) spanwise 분포 ──────────────────────────────────────────
    dfa = compute(args.A, args.avg_revs, args.mtip, args.umax)
    dfb = compute(args.B, args.avg_revs, args.mtip, args.umax)
    if dfa is None or dfb is None:
        miss = args.A if dfa is None else args.B
        print("[경고] blade_diagnostics 데이터가 없습니다: %s" % miss)
        print("       (런이 충분히 진행됐는지/폴더명이 맞는지 확인)")
        return

    # 비교용 spanwise CSV 저장
    for tag, df in [("A", dfa), ("B", dfb)]:
        cols = [c for c in ["r_R", "eps_lu", "alpha", "phi", "CL", "CD",
                            "mach", "M2CL", "F_n", "F_theta"] if c in df.columns]
        outp = os.path.join(args.outdir, "spanwise_%s_%s.csv"
                            % (tag, os.path.basename(os.path.normpath(
                                args.A if tag == "A" else args.B))))
        df[cols].to_csv(outp, index=False)

    # ── 3) 팁 요약 ────────────────────────────────────────────────
    print("=" * 64)
    print(" 팁(최외측 마커) 비교")
    print("=" * 64)
    sa, sb = tip_summary(dfa), tip_summary(dfb)
    keys = [k for k in ("eps_lu", "phi", "alpha", "M2CL", "CL", "F_n")
            if k in sa or k in sb]
    print("  %-8s %12s %12s %12s" % ("", LA, LB, "Δ(B−A)"))
    for k in keys:
        va, vb = sa.get(k), sb.get(k)
        sva = "%12.4f" % va if va is not None else "%12s" % "—"
        svb = "%12.4f" % vb if vb is not None else "%12s" % "—"
        sd = "%12.4f" % (vb - va) if (va is not None and vb is not None) else "%12s" % "—"
        print("  %-8s %s %s %s" % (k, sva, svb, sd))
    print("  (팁 r/R ≈ %.3f)" % sa["r_R_tip"])
    print()

    # ── 4) A/B 오버레이 플롯 ──────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [c for c in ["eps_lu", "phi", "M2CL", "F_n"]
              if c in dfa.columns and c in dfb.columns]
    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        ax = [ax]
    # NOTE: plot labels are ASCII only — matplotlib's default font lacks Korean
    # glyphs (and clusters often have no Korean font). Console output stays Korean.
    titles = {"eps_lu": "eps_lu [lu] (taper check)", "phi": "inflow angle phi [deg]",
              "M2CL": "M^2 * CL (sectional loading)", "F_n": "F_n [lattice]"}
    for col, a in zip(panels, ax):
        a.plot(dfa["r_R"], dfa[col], "-o", ms=3, label=args.la)
        a.plot(dfb["r_R"], dfb[col], "-s", ms=3, label=args.lb)
        a.set(xlabel="r/R", title=titles.get(col, col))
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("ALM Stage B eps-taper: A vs B", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(args.outdir, "AB_taper_compare.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("저장: %s" % png)
    print("저장: %s/spanwise_{A,B}_*.csv" % args.outdir)


if __name__ == "__main__":
    main()
