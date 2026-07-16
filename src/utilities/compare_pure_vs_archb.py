"""Pure vs archB collective sweeps + references — final combined set.

Outputs into 0713_multi_gpu/sweep_analysis/compare2/:
  polar_pure_archb_refs.png, dCT_vs_collective.png,
  spanwise_M2Cn_grid.png (2x2, pure vs archB per collective + EXP@10)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.path.join("aeromechanics_workshop", "HVAB", "0713_multi_gpu")
VD = os.path.join("aeromechanics_workshop", "HVAB",
                  "0701_marker_tests_and_shareddata", "validation_datas")
OUT = os.path.join(B, "sweep_analysis", "compare2")
os.makedirs(OUT, exist_ok=True)

pure = np.genfromtxt(os.path.join(B, "sweep_analysis", "sweep_summary.csv"),
                     delimiter=",", names=True)
arb = np.genfromtxt(os.path.join(B, "sweep_analysis_archb",
                                 "sweep_summary.csv"),
                    delimiter=",", names=True)
exp = np.genfromtxt(os.path.join(VD, "experiment_measured",
                                 "hvab_M065_NATURAL_Run44_reference.csv"),
                    delimiter=",", names=True)
ksas = np.genfromtxt(os.path.join(VD, "cfd_digitized",
                                  "KSAS_CFDCSD_FM_CTsigma_approx.csv"),
                     delimiter=",", names=True, usecols=(0, 1),
                     invalid_raise=False)

# polar
plt.figure(figsize=(8.5, 5.6))
plt.plot(exp["CT_sigma"], exp["FM"], "k-o", ms=4, label="HVAB EXP (Norman, natural)")
plt.plot(ksas["CT_sigma"], ksas["FM"], "^--", color="tab:green", ms=6,
         label="KSAS CFD/CSD elastic (dig.)")
plt.plot(pure["CT_over_sigma"], pure["FM"], "s-", color="tab:red", ms=7,
         label="present pure ALM")
plt.plot(arb["CT_over_sigma"], arb["FM"], "D-", color="tab:orange", ms=6,
         label="present archB (trunc+Kleine)")
for d, x, y in zip(pure["deg"], pure["CT_over_sigma"], pure["FM"]):
    plt.annotate(f"{int(d)}°", (x, y), textcoords="offset points",
                 xytext=(5, 5), color="tab:red", fontsize=9)
plt.xlabel(r"$C_T/\sigma$"); plt.ylabel("FM"); plt.grid(alpha=0.3)
plt.title("HVAB hover polar — pure vs archB vs references")
plt.legend()
plt.savefig(os.path.join(OUT, "polar_pure_archb_refs.png"), dpi=130,
            bbox_inches="tight")
plt.close()

# delta CT + vs experiment
e_ct = np.interp(pure["deg"], exp["collective_deg"], exp["CT"])
d_pure = 100 * (pure["C_T"] / e_ct - 1)
d_arb = 100 * (arb["C_T"] / e_ct - 1)
corr = 100 * (arb["C_T"] / pure["C_T"] - 1)
plt.figure(figsize=(8, 5))
plt.plot(pure["deg"], d_pure, "s-", color="tab:red", label="pure vs EXP")
plt.plot(arb["deg"], d_arb, "D-", color="tab:orange", label="archB vs EXP")
plt.plot(arb["deg"], corr, "v--", color="tab:blue",
         label="Kleine correction effect (archB/pure - 1)")
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("collective [deg]"); plt.ylabel(r"$\Delta C_T$ [%]")
plt.title("CT overprediction vs loading — kernel-bias signature")
plt.grid(alpha=0.3); plt.legend()
plt.savefig(os.path.join(OUT, "dCT_vs_collective.png"), dpi=130,
            bbox_inches="tight")
plt.close()

# spanwise grid
exp_nf = np.genfromtxt(os.path.join(VD, "experiment_measured",
                                    "hvab_M065_coll10_sectionalNF_EXP.csv"),
                       delimiter=",", names=True)
fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for ax, deg in zip(axs.ravel(), (6, 8, 10, 12)):
    p_ = np.genfromtxt(os.path.join(B, "sweep_analysis",
                                    f"spanwise_c{deg}.csv"),
                       delimiter=",", names=True)
    a_ = np.genfromtxt(os.path.join(B, "sweep_analysis_archb",
                                    f"spanwise_c{deg}.csv"),
                       delimiter=",", names=True)
    ax.plot(p_["r_R"], p_["M2Cn"], "-", color="tab:red", lw=2, label="pure")
    ax.plot(a_["r_R"], a_["M2Cn"], "-", color="tab:orange", lw=2,
            label="archB")
    if deg == 10:
        ax.plot(exp_nf["r_R"], exp_nf["M2CN"], "kd", ms=7, mfc="none",
                label="EXP sectional")
    ax.set_title(f"collective {deg}°"); ax.grid(alpha=0.3)
    ax.set_ylabel(r"$M^2 c_n$"); ax.legend(fontsize=9)
for ax in axs[1]:
    ax.set_xlabel("r/R")
fig.suptitle("spanwise loading: pure vs archB across collective")
plt.savefig(os.path.join(OUT, "spanwise_M2Cn_grid.png"), dpi=130,
            bbox_inches="tight")
plt.close()

print("deg | dCT_pure  dCT_archB  corr_effect | FM pure/archB/EXP")
e_fm = np.interp(pure["deg"], exp["collective_deg"], exp["FM"])
for i in range(len(pure)):
    print(f"{int(pure['deg'][i]):3d} | {d_pure[i]:+7.1f}%  {d_arb[i]:+7.1f}%"
          f"   {corr[i]:+6.2f}%   | {pure['FM'][i]:.4f}/{arb['FM'][i]:.4f}"
          f"/{e_fm[i]:.4f}")
print(f"wrote {OUT}/")
