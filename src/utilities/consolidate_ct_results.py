#!/usr/bin/env python
"""
CT hover ALM 결과 통합 비교 — 모든 result 폴더를 자동 탐색해 한 표/CSV/플롯으로.

각 폴더에서:
  - C_T(rotorcraft) = thrust_lu/(rho_ref·area_lu·tip_speed_lu²), 마지막 N rev 평균
  - 폴더명에서 설정 태그 파싱 (Prandtl 모드 / taper / n_radial / extent / SGS / mtip)
  - 팁(최외측 마커) phi·alpha·eps·M2CL·F_n (spanwise_post.compute 재사용)

출력:
  - 콘솔 통합 표 (C_T 정렬) + 누락(pending) 폴더 표시
  - <outdir>/ct_consolidated.csv  (마스터 표)
  - <outdir>/ct_consolidated_CT.png    (C_T 막대 + exp 선)
  - <outdir>/ct_consolidated_spanwise.png (F_n·phi(r/R) 오버레이)

사용법 (결과 폴더가 있는 디렉터리 = 보통 repo 루트에서):
    python src/utilities/consolidate_ct_results.py
    python src/utilities/consolidate_ct_results.py --glob "result_ct_t08.0_M877_*" --avg-revs 3
    python src/utilities/consolidate_ct_results.py --glob "result_ct_*"   # M0.44 포함 전체
"""
import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas 필요:  pip install pandas")

from src.utilities.spanwise_post import compute  # noqa: E402


def parse_tags(folder):
    """폴더명 → 설정 태그 dict + 짧은 라벨."""
    name = os.path.basename(folder.rstrip("/"))
    t = {}
    t["prandtl"] = ("OFF" if "noprandtl" in name
                    else ("stdR" if "prtipR" in name else "legacy"))
    t["taper"] = "taper" if "taper" in name else "-"
    mn = re.search(r"_n(\d+)\b", name)
    t["n_radial"] = int(mn.group(1)) if mn else 40
    t["extent"] = "extended" if "extended" in name else "light"
    t["sgs"] = "off" if "nosgs" in name else "on"
    mm = re.search(r"_M(\d+)_", name)
    t["mtip"] = round(int(mm.group(1)) / 1000, 3) if mm else None
    # 짧은 라벨
    lbl = t["prandtl"]
    if t["taper"] != "-":
        lbl += "+taper"
    if t["extent"] == "extended":
        lbl += " ext"
    if t["n_radial"] != 40:
        lbl += " n%d" % t["n_radial"]
    t["label"] = lbl
    return name, t


def ct_of(folder, avg_revs):
    """rotor_performance.csv → (C_T rotorcraft, T_lu, n_rev) or (None,None,None)."""
    path = os.path.join(folder, "csv", "rotor_performance.csv")
    if not os.path.isfile(path):
        return None, None, None
    df = pd.read_csv(path)
    if df.empty or "revolutions" not in df.columns:
        return None, None, None
    nrev = float(df["revolutions"].max())
    w = df[df["revolutions"] >= nrev - avg_revs]
    if w.empty:
        w = df
    T = w["thrust_lu"].mean()
    den = w["rho_ref"].mean() * w["area_lu"].mean() * w["tip_speed_lu"].mean() ** 2
    return (T / den if den else float("nan")), T, nrev


def tip_metrics(folder, avg_revs, mtip):
    """최외측 마커의 phi/alpha/eps/M2CL/F_n + 전체 spanwise DataFrame."""
    try:
        df = compute(folder, avg_revs, mtip, 0.1)
    except Exception:
        df = None
    if df is None or df.empty:
        return {}, None
    last = df.iloc[-1]
    m = {k: float(last[k]) for k in ("phi", "alpha", "eps_lu", "M2CL", "F_n")
         if k in df.columns}
    return m, df


def main():
    ap = argparse.ArgumentParser(description="CT hover ALM 결과 통합 비교")
    ap.add_argument("--glob", default="result_ct_t08.0_M877_*",
                    help="결과 폴더 glob 패턴")
    ap.add_argument("--avg-revs", type=float, default=3.0)
    ap.add_argument("--exp-ct", type=float, default=0.00473, help="실험 C_T (M0.877)")
    ap.add_argument("--mtip", type=float, default=0.877)
    ap.add_argument("--outdir", default="aeromechanics_workshop/temp_results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    folders = sorted(d for d in glob.glob(args.glob) if os.path.isdir(d))
    if not folders:
        print("폴더 없음:", args.glob)
        return

    rows, spans = [], []
    for f in folders:
        name, t = parse_tags(f)
        ct, T, nrev = ct_of(f, args.avg_revs)
        tm, df = tip_metrics(f, args.avg_revs, t.get("mtip") or args.mtip)
        rows.append({
            "label": t["label"], "prandtl": t["prandtl"], "taper": t["taper"],
            "n_radial": t["n_radial"], "extent": t["extent"], "sgs": t["sgs"],
            "mtip": t["mtip"], "n_rev": round(nrev, 2) if nrev else None,
            "C_T": round(ct, 5) if ct else None,
            "dev_%": round((ct - args.exp_ct) / args.exp_ct * 100, 1) if ct else None,
            "tip_phi": round(tm.get("phi"), 3) if "phi" in tm else None,
            "tip_eps": round(tm.get("eps_lu"), 3) if "eps_lu" in tm else None,
            "folder": name,
        })
        if df is not None:
            spans.append((t["label"], df))

    tab = pd.DataFrame(rows).sort_values(
        "C_T", na_position="last").reset_index(drop=True)

    # ── 콘솔 표 ──
    print("=" * 92)
    print(" CT hover 결과 통합  (exp C_T=%.5f, 마지막 %.0f rev 평균)"
          % (args.exp_ct, args.avg_revs))
    print("=" * 92)
    show = ["label", "prandtl", "taper", "n_radial", "extent",
            "C_T", "dev_%", "tip_phi", "tip_eps", "n_rev"]
    with pd.option_context("display.width", 160,
                           "display.max_rows", None):
        print(tab[show].to_string(index=False))
    print("\n  exp: C_T=%.5f" % args.exp_ct)
    missing = tab[tab["C_T"].isna()]
    if len(missing):
        print("  [데이터 없음/미실행]:", ", ".join(missing["folder"]))

    csv_out = os.path.join(args.outdir, "ct_consolidated.csv")
    tab.to_csv(csv_out, index=False)
    print("\n저장:", csv_out)

    # ── 플롯 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = tab[tab["C_T"].notna()]
    if len(done):
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(done)), 4.5))
        colors = {"legacy": "tab:gray", "stdR": "tab:blue", "OFF": "tab:red"}
        bars = ax.bar(done["label"], done["C_T"],
                      color=[colors.get(p, "tab:green") for p in done["prandtl"]])
        ax.axhline(args.exp_ct, color="k", ls="--", lw=1.5,
                   label="exp %.5f" % args.exp_ct)
        for b, d in zip(bars, done["dev_%"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    "%+.0f%%" % d, ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("C_T (rotorcraft)")
        ax.set_title("CT hover M0.877 — C_T vs exp")
        ax.legend()
        plt.xticks(rotation=35, ha="right", fontsize=8)
        fig.tight_layout()
        p1 = os.path.join(args.outdir, "ct_consolidated_CT.png")
        fig.savefig(p1, dpi=130)
        plt.close(fig)
        print("저장:", p1)

    if spans:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for lbl, df in spans:
            ax[0].plot(df["r_R"], df["F_n"], "-o", ms=2.5, label=lbl)
            if "phi" in df.columns:
                ax[1].plot(df["r_R"], df["phi"], "-o", ms=2.5, label=lbl)
        ax[0].set(xlabel="r/R", ylabel="F_n [lattice]", title="Section thrust force")
        ax[1].set(xlabel="r/R", ylabel="phi [deg]", title="Inflow angle")
        for a in ax:
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
        fig.tight_layout()
        p2 = os.path.join(args.outdir, "ct_consolidated_spanwise.png")
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        print("저장:", p2)


if __name__ == "__main__":
    main()
