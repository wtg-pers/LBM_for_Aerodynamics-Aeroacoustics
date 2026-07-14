"""Collective sweep vs references: HVAB experiment + KSAS paper curves.

References (digitized/measured, aeromechanics_workshop validation_datas):
  - EXP Run44 hover polar (collective, CT, CT/sigma, CP, FM)
  - KSAS CFD/CSD elastic-ext FM-CT/sigma (Fig8), Jain HeliOS FM-CT/sigma
  - KSAS Fig10a/b spanwise M2Cn/M2Cc @ c10 (EXP-Transition / Elastic-Ext /
    Elastic-Fully-turb / Rigid-Ext) + sectional NF EXP
Ours: sweep_analysis/sweep_summary.csv (+ spanwise_c{deg}.csv if present —
regenerate with the current postprocess to get them).

    python src/utilities/compare_sweep_vs_refs.py \
        --sweep aeromechanics_workshop/HVAB/0713_multi_gpu/sweep_analysis
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VD = os.path.join("aeromechanics_workshop", "HVAB",
                  "0701_marker_tests_and_shareddata", "validation_datas")


def rd(path, cols):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip() for h in rows[0]]
    ix = [hdr.index(c) for c in cols]
    out = []
    for r in rows[1:]:
        try:
            out.append([float(r[i]) for i in ix])
        except (ValueError, IndexError):
            pass
    return np.array(out)


def rd_curves(path, ycol):
    rows = list(csv.reader(open(path)))
    out = {}
    for r in rows[1:]:
        if len(r) < 3:
            continue
        try:
            x, c, y = float(r[0]), r[1].strip(), float(r[2])
        except ValueError:
            continue
        out.setdefault(c, []).append((x, y))
    return {c: np.array(sorted(v)) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=os.path.join(
        "aeromechanics_workshop", "HVAB", "0713_multi_gpu", "sweep_analysis"))
    a = ap.parse_args()
    out = os.path.join(a.sweep, "compare")
    os.makedirs(out, exist_ok=True)

    ours = rd(os.path.join(a.sweep, "sweep_summary.csv"),
              ["deg", "C_T", "C_P", "FM", "CT_over_sigma"])
    exp = rd(os.path.join(VD, "experiment_measured",
                          "hvab_M065_NATURAL_Run44_reference.csv"),
             ["collective_deg", "CT", "CT_sigma", "CP", "FM"])
    ksas = rd(os.path.join(VD, "cfd_digitized",
                           "KSAS_CFDCSD_FM_CTsigma_approx.csv"),
              ["CT_sigma", "FM"])
    helios = rd(os.path.join(VD, "cfd_digitized",
                             "Jain_Helios_FM_CTsigma_approx.csv"),
                ["CT_sigma", "FM"])

    # ── Fig A: FM vs CT/sigma ──
    plt.figure(figsize=(8.5, 5.6))
    plt.plot(exp[:, 2], exp[:, 4], "k-o", ms=4, label="HVAB EXP (Run44)")
    plt.plot(ksas[:, 0], ksas[:, 1], "^--", color="tab:green", ms=6,
             label="KSAS CFD/CSD elastic (dig.)")
    plt.plot(helios[:, 0], helios[:, 1], "v--", color="tab:purple", ms=6,
             label="Jain HeliOS (dig.)")
    plt.plot(ours[:, 4], ours[:, 3], "s-", color="tab:red", ms=7,
             label="present LBM-ALM (rigid, pure)")
    for d, x, y in zip(ours[:, 0], ours[:, 4], ours[:, 3]):
        plt.annotate(f"{int(d)}°", (x, y), textcoords="offset points",
                     xytext=(6, -10), color="tab:red")
    plt.xlabel(r"$C_T/\sigma$"); plt.ylabel("FM")
    plt.title("HVAB hover polar — sweep vs experiment & published CFD")
    plt.grid(alpha=0.3); plt.legend()
    plt.savefig(os.path.join(out, "FM_vs_CTsigma_refs.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # ── Fig B: collective vs CT / CP / FM (EXP overlay) ──
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, oc, ec, lab in ((axs[0], 1, 1, r"$C_T$"),
                            (axs[1], 2, 3, r"$C_P$"),
                            (axs[2], 3, 4, "FM")):
        ax.plot(exp[:, 0], exp[:, ec], "k-o", ms=4, label="EXP Run44")
        ax.plot(ours[:, 0], ours[:, oc], "s-", color="tab:red", ms=6,
                label="present")
        ax.set_xlabel("collective [deg]"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("collective sweep vs experiment")
    plt.savefig(os.path.join(out, "collective_vs_refs.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # deltas table
    print("deg |   CT(ours)  CT(exp)   dCT%  |  FM(ours) FM(exp)")
    for d, ct, cp_, fm, cts in ours:
        e = exp[np.isclose(exp[:, 0], d)]
        if len(e):
            print(f"{int(d):3d} |  {ct:.5f}  {e[0,1]:.5f}  "
                  f"{100*(ct/e[0,1]-1):+5.1f}%  |  {fm:.4f}  {e[0,4]:.4f}")

    # ── Fig C/D: spanwise @ c10 (needs spanwise_c10.csv from postprocess) ──
    sp10 = os.path.join(a.sweep, "spanwise_c10.csv")
    if os.path.exists(sp10):
        mine = rd(sp10, ["r_R", "M2Cn", "M2Cc"])
        exp_nf = rd(os.path.join(VD, "experiment_measured",
                                 "hvab_M065_coll10_sectionalNF_EXP.csv"),
                    ["r_R", "M2CN"])
        for key, col, fig10, ename in (
                ("M2Cn", 1, "KSAS_Fig10a_M2Cn_c10.csv", exp_nf),
                ("M2Cc", 2, "KSAS_Fig10b_M2Cc_c10.csv", None)):
            curves = rd_curves(os.path.join(VD, "cfd_digitized", fig10), key)
            plt.figure(figsize=(9, 5.4))
            for cname, arr in curves.items():
                sty = "k-o" if "EXP" in cname else "--"
                plt.plot(arr[:, 0], arr[:, 1], sty, ms=4, label=f"KSAS {cname}")
            if ename is not None:
                plt.plot(ename[:, 0], ename[:, 1], "kd", ms=7, mfc="none",
                         label="EXP sectional NF")
            plt.plot(mine[:, 0], mine[:, col], "-", color="tab:red", lw=2,
                     label="present pure ALM (10°)")
            spb = os.path.join(a.sweep, "spanwise_archB10.csv")
            if os.path.exists(spb):
                mb = rd(spb, ["r_R", "M2Cn", "M2Cc"])
                plt.plot(mb[:, 0], mb[:, col], "-", color="tab:orange",
                         lw=2, label="present archB (trunc+Kleine, 10°)")
            plt.xlabel("r/R"); plt.ylabel(rf"$M^2 c_{{{key[-1].lower()}}}$")
            plt.title(f"spanwise {key} @ collective 10° — vs KSAS Fig10")
            plt.grid(alpha=0.3); plt.legend(fontsize=9)
            plt.savefig(os.path.join(out, f"spanwise_{key}_c10_refs.png"),
                        dpi=130, bbox_inches="tight")
            plt.close()
        print(f"spanwise overlays written")
    else:
        print("spanwise_c10.csv not found — rerun the (updated) postprocess "
              "on the cluster and copy sweep_analysis/spanwise_c*.csv")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
