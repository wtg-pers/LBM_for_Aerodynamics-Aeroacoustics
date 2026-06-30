#!/usr/bin/env python
"""ALM 속도 샘플러 A/B/C 3-way 비교 — 팁 유입 결손의 b1/b2 분해.

세 런(동일 조건, 샘플러 mode만 다름):
  A  = gaussian   (현행 ±3ε)
  B3 = mask_disk  (off-disk b1 제거)
  B1 = point      (b1+b2 제거)

분해 (spanwise 각 마커 + 팁):
  b1 (off-disk)        = A  − B3
  b2 (radial smooth)   = B3 − B1
  total (b)            = A  − B1
주: φ(inflow angle)는 Prandtl 무관(velocity triangle 단계) → 샘플러 효과를 직접 반영.

수행:
  1) C_T 비교 (rotor_performance.csv 마지막 N rev 평균; 폴더 입력시) ± 실험값
  2) spanwise 병합 + b1/b2/total 분해 CSV
  3) 팁(최외측 마커) 분해표 + φ 회복 판정 (--phi-target / --alpha-target)
  4) 오버레이 플롯(φ, α, M2CL, eps_lu, 3곡선) + 분해 플롯(b1·b2 vs r/R)

사용법 (런 후):
  python src/utilities/compare_sampler_abc.py \
      --A  result_hvab_hover_c10.0_M650_mlg4_D32_light_sampA_gauss \
      --B3 result_hvab_hover_c10.0_M650_mlg4_D32_light_sampB3_mask \
      --B1 result_hvab_hover_c10.0_M650_mlg4_D32_light_sampB1_point \
      --mtip 0.65 --avg-revs 3 --alpha-target 3.0

각 입력은 result 폴더(→ spanwise_post.compute) 또는 spanwise CSV 파일 경로 모두 허용.
CWD 무관(스크립트가 repo 루트를 sys.path에 추가).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pandas/numpy 필요:  pip install pandas numpy")

from src.utilities.spanwise_post import compute  # noqa: E402

# compare_taper_ab.py의 C_T 추출을 재사용
try:
    from src.utilities.compare_taper_ab import ct_from_perf  # noqa: E402
except Exception:
    ct_from_perf = None


def load_spanwise(path, avg_revs, mtip, umax):
    """입력이 CSV면 직접 read, result 폴더면 spanwise_post.compute. → DataFrame|None."""
    if path is None:
        return None
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif os.path.isdir(path):
        df = compute(path, avg_revs, mtip, umax)
    else:
        print("  [경고] 입력 없음/형식불명: %s" % path)
        return None
    if df is None or "r_R" not in df.columns:
        return None
    return df.sort_values("r_R").reset_index(drop=True)


def aligned(dfs):
    """세 DataFrame이 동일 r_R(마커)인지 확인하고 정렬 정합."""
    lens = {len(d) for d in dfs if d is not None}
    if len(lens) != 1:
        print("  [경고] 마커 수 불일치 %s — r_R 교집합으로 정렬" % lens)
        common = set(dfs[0]["r_R"].round(6))
        for d in dfs[1:]:
            if d is not None:
                common &= set(d["r_R"].round(6))
        dfs = [d[d["r_R"].round(6).isin(common)].reset_index(drop=True)
               if d is not None else None for d in dfs]
    return dfs


def main():
    ap = argparse.ArgumentParser(
        description="ALM 샘플러 A/B/C 3-way 비교 (b1/b2 분해 + 팁 φ 회복 판정)")
    base = "result_hvab_hover_c10.0_M650_mlg4_D32_light"
    ap.add_argument("--A",  default=base + "_sampA_gauss",
                    help="A=gaussian 폴더 또는 spanwise CSV")
    ap.add_argument("--B3", default=base + "_sampB3_mask",
                    help="B-iii=mask_disk 폴더/CSV")
    ap.add_argument("--B1", default=base + "_sampB1_point",
                    help="B-i=point 폴더/CSV")
    ap.add_argument("--mtip", type=float, default=0.65)
    ap.add_argument("--avg-revs", type=float, default=3.0)
    ap.add_argument("--umax", type=float, default=0.1)
    ap.add_argument("--exp-ct", type=float, default=None,
                    help="실험 C_T(있으면 편차 출력)")
    ap.add_argument("--alpha-target", type=float, default=None,
                    help="팁 운동량이론 기대 α[deg] (회복 판정용, 예: 3.0)")
    ap.add_argument("--phi-target", type=float, default=None,
                    help="팁 기대 φ[deg] (회복 판정용)")
    ap.add_argument("--outdir", default="aeromechanics_workshop/temp_results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    LBL = {"A": "gaussian", "B3": "mask_disk", "B1": "point"}

    # ── 1) C_T 3-way ──────────────────────────────────────────────
    print("=" * 72)
    print(" C_T 비교  (마지막 %.0f rev 평균, rotorcraft 정의)" % args.avg_revs)
    print("=" * 72)
    cts = {}
    for tag, d in (("A", args.A), ("B3", args.B3), ("B1", args.B1)):
        if ct_from_perf and os.path.isdir(d):
            ct, T, err = ct_from_perf(d, args.avg_revs)
        else:
            ct, T, err = None, None, "폴더 아님/perf 없음"
        cts[tag] = ct
        if ct is None:
            print("  %-9s : %s" % (LBL[tag], err))
        else:
            dev = ("  (exp %+.1f%%)" % ((ct - args.exp_ct) / args.exp_ct * 100)
                   if args.exp_ct else "")
            print("  %-9s : C_T=%.5f  (T_lu=%.4f)%s" % (LBL[tag], ct, T, dev))
    if args.exp_ct:
        print("  %-9s : C_T=%.5f" % ("exp", args.exp_ct))
    if cts["A"] and cts["B1"]:
        print("  ΔC_T: B3−A=%+.5f,  B1−A=%+.5f  (point/mask가 낮으면 팁 de-loading↑)"
              % (cts["B3"] - cts["A"], cts["B1"] - cts["A"]))
    print()

    # ── 2) spanwise 로드 + 분해 ───────────────────────────────────
    dA = load_spanwise(args.A, args.avg_revs, args.mtip, args.umax)
    dB3 = load_spanwise(args.B3, args.avg_revs, args.mtip, args.umax)
    dB1 = load_spanwise(args.B1, args.avg_revs, args.mtip, args.umax)
    if any(x is None for x in (dA, dB3, dB1)):
        miss = [t for t, x in (("A", dA), ("B3", dB3), ("B1", dB1)) if x is None]
        print("[경고] spanwise 로드 실패: %s — 폴더/CSV 경로·런 진행 확인" % miss)
        return
    dA, dB3, dB1 = aligned([dA, dB3, dB1])

    r = dA["r_R"].values
    QTY = [c for c in ("phi", "alpha", "F_n", "M2CL", "CL") if c in dA.columns]
    merged = pd.DataFrame({"r_R": r})
    for q in QTY:
        a, b3, b1 = dA[q].values, dB3[q].values, dB1[q].values
        merged["%s_A" % q] = a
        merged["%s_B3" % q] = b3
        merged["%s_B1" % q] = b1
        merged["%s_b1_offdisk" % q] = a - b3      # A − B3
        merged["%s_b2_radial" % q] = b3 - b1       # B3 − B1
        merged["%s_total" % q] = a - b1            # A − B1
    if "eps_lu" in dA.columns:
        merged["eps_lu"] = dA["eps_lu"].values
    out_csv = os.path.join(args.outdir, "sampler_abc_decomposition.csv")
    merged.to_csv(out_csv, index=False)

    # ── 3) 팁 분해표 + 판정 ───────────────────────────────────────
    print("=" * 72)
    print(" 팁(최외측 마커, r/R=%.4f) 분해   [b1=A−B3, b2=B3−B1, total=A−B1]" % r[-1])
    print("=" * 72)
    print("  %-7s %9s %9s %9s | %9s %9s %9s"
          % ("qty", "A gauss", "B3 mask", "B1 point",
             "b1 offdsk", "b2 radial", "total"))
    i = -1  # 팁 마커 (positional)
    for q in QTY:
        a = merged["%s_A" % q].iloc[i]
        b3 = merged["%s_B3" % q].iloc[i]
        b1 = merged["%s_B1" % q].iloc[i]
        print("  %-7s %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f"
              % (q, a, b3, b1, a - b3, b3 - b1, a - b1))
    print()

    # 판정: 팁 φ/α 회복
    print("=" * 72)
    print(" 판정 — 팁 유입 회복")
    print("=" * 72)
    if "phi" in QTY:
        pa = merged["phi_A"].iloc[i]; p3 = merged["phi_B3"].iloc[i]; p1 = merged["phi_B1"].iloc[i]
        print("  팁 φ:  A=%.2f°  →  B3=%.2f°  →  B1=%.2f°  (회복=커질수록 좋음)"
              % (pa, p3, p1))
        if args.phi_target is not None:
            print("        목표 φ≈%.2f° | 잔차 A=%+.2f B3=%+.2f B1=%+.2f"
                  % (args.phi_target, pa - args.phi_target,
                     p3 - args.phi_target, p1 - args.phi_target))
    if "alpha" in QTY:
        aa = merged["alpha_A"].iloc[i]; a3 = merged["alpha_B3"].iloc[i]; a1 = merged["alpha_B1"].iloc[i]
        print("  팁 α:  A=%.2f°  →  B3=%.2f°  →  B1=%.2f°  (운동량 기대 쪽=작아질수록 좋음)"
              % (aa, a3, a1))
        if args.alpha_target is not None:
            tot = aa - a1
            rec = (aa - a1) / (aa - args.alpha_target) * 100 if (aa - args.alpha_target) else float("nan")
            print("        목표 α≈%.2f° | A 잔차=%+.2f°, B1이 회복한 비율=%.0f%% of 결손"
                  % (args.alpha_target, aa - args.alpha_target, rec))
            b1c = (aa - a3); b2c = (a3 - a1)
            if tot:
                print("        분해: off-disk(b1)=%.0f%%, radial(b2)=%.0f%% of total Δα"
                      % (b1c / tot * 100, b2c / tot * 100))
    print()

    # ── 4) 플롯 ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # ASCII-only labels (클러스터 한글 폰트 부재). 콘솔은 한글 유지.
    panels = [c for c in ("phi", "alpha", "M2CL", "eps_lu")
              if c in dA.columns]
    titles = {"phi": "inflow angle phi [deg]", "alpha": "AoA alpha [deg]",
              "M2CL": "M^2*CL (loading)", "eps_lu": "eps_lu [lu]",
              "F_n": "F_n [lattice]"}
    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        ax = [ax]
    for col, a in zip(panels, ax):
        a.plot(dA["r_R"], dA[col], "-o", ms=3, label="A gaussian")
        a.plot(dB3["r_R"], dB3[col], "-s", ms=3, label="B3 mask_disk")
        a.plot(dB1["r_R"], dB1[col], "-^", ms=3, label="B1 point")
        if col == "alpha" and args.alpha_target is not None:
            a.axhline(args.alpha_target, ls="--", c="k", lw=1, label="momentum target")
        a.set(xlabel="r/R", title=titles.get(col, col))
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("ALM sampler A/B/C: gaussian vs mask_disk vs point",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1png = os.path.join(args.outdir, "sampler_abc_overlay.png")
    fig.savefig(p1png, dpi=130); plt.close(fig)

    # 분해 플롯 (φ b1/b2 vs r/R)
    if "phi" in QTY:
        fig2, a2 = plt.subplots(figsize=(6, 4))
        a2.plot(r, merged["phi_b1_offdisk"], "-o", ms=3, label="b1 off-disk (A-B3)")
        a2.plot(r, merged["phi_b2_radial"], "-s", ms=3, label="b2 radial (B3-B1)")
        a2.plot(r, merged["phi_total"], "-^", ms=3, label="total (A-B1)")
        a2.axhline(0, c="k", lw=0.7)
        a2.set(xlabel="r/R", ylabel="delta phi [deg]",
               title="phi deficit decomposition (sampler)")
        a2.grid(alpha=0.3); a2.legend(fontsize=8)
        fig2.tight_layout()
        p2png = os.path.join(args.outdir, "sampler_abc_decomp_phi.png")
        fig2.savefig(p2png, dpi=130); plt.close(fig2)
        print("저장: %s" % p2png)

    print("저장: %s" % p1png)
    print("저장: %s" % out_csv)


if __name__ == "__main__":
    main()
