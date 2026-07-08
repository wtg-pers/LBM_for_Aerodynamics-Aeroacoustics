"""Decompose the ALM tip over-loading into (1) induced-angle deficit (alpha too
high, F1) and (2) deck CL over-prediction, using the MEASURED sectional Cn and
the TM-4264 experimental RC(6)-08 polar as the truth for CL(alpha)."""
import os, sys, csv
import numpy as np
REPO="/home/wtg/01_python_project/test_0124"
sys.path.insert(0, os.path.join(REPO,"src/utilities")); sys.path.insert(0, os.path.join(REPO,"src"))
from compare_spanwise import load_spanwise
from actuator.c81_loader import C81PolarSet
MTIP,UMAX=0.65,0.1

# ALM baseline (uniform+gauss)
s=load_spanwise(os.path.join(REPO,"aeromechanics_workshop/HVAB/260701/case1_marker_test_with_nasa_c81/pureALM_csv"),3.0)
Ml=s["u_rel"]*(MTIP/UMAX); ar=np.radians(s["alpha"])
alm_Cn = s["CL"]*np.cos(ar)+s["CD"]*np.sin(ar)          # section normal-force coeff
alm_M2Cn = Ml**2*alm_Cn

# measured M2Cn -> Cn_exp (Cn = M2Cn / M_loc^2, M_loc = mtip*r/R)
exd=[r for r in csv.DictReader(open(os.path.join(REPO,
    "aeromechanics_workshop/HVAB/nasa_hvab_airloads_ref/hvab_M065_coll10_sectionalNF_EXP.csv")))
    if r["M2CN"].strip() not in ("","nan")]
exR=np.array([float(r["r_R"]) for r in exd]); exM=np.array([float(r["M2CN"]) for r in exd])
exCn = exM/(MTIP*exR)**2

# TM-4264 experimental RC(6)-08 polar (M0.63) -> invert CL(alpha)
tm=[r for r in csv.DictReader(open(os.path.join(REPO,
    "aeromechanics_workshop/HVAB/nasa_hvab_airloads_ref/RC608_TM4264_M063_polar_EXP.csv")))]
tm_a=np.array([float(r["alpha_deg"]) for r in tm]); tm_cl=np.array([float(r["Cl"]) for r in tm])
def alpha_for_cl(cl): return float(np.interp(cl, tm_cl, tm_a))   # exp: what alpha gives this CL
# OVERFLOW deck (what our ALM uses at the tip, RC6-08)
ovf=C81PolarSet.from_file(os.path.join(REPO,"data/airfoils/hvab_nasa_overflow/RC6-08_OVERFLOW_SA.c81"))

print("Tip decomposition (baseline uniform+gauss). M_loc=mtip*r/R.")
print(f"{'r/R':>6}{'exp Cn':>8}{'ALM Cn':>8}{'ALM α':>7}{'α_exp*':>8}{'Δα(F1)':>8}{'deckCL@αALM':>11}{'expCL@αALM':>11}")
for xr, cne in zip(exR, exCn):
    if xr < 0.85: continue
    i = np.argmin(np.abs(s["r_R"]-xr))
    aA = s["alpha"][i]; cnA = alm_Cn[i]; M = MTIP*xr
    a_exp = alpha_for_cl(cne)          # exp: alpha that gives the measured Cn
    dalpha = aA - a_exp                # induced-angle deficit (ALM alpha too high)
    deck_cl = ovf.get_CL(aA, M)        # deck CL at ALM's alpha
    exp_cl  = float(np.interp(aA, tm_a, tm_cl))  # TM-4264 CL at ALM's alpha
    print(f"{xr:6.3f}{cne:8.3f}{cnA:8.3f}{aA:7.2f}{a_exp:8.2f}{dalpha:+8.2f}{deck_cl:11.3f}{exp_cl:11.3f}")
print("\nΔα(F1)=ALM α − (α that would give the MEASURED Cn) = induced-angle deficit.")
print("deckCL vs expCL @ same α = deck over-prediction (contributor 2).")
