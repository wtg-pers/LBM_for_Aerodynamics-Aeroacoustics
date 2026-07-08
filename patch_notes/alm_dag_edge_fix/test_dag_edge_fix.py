"""Validate the reimplemented _viscous_core_correction (edge-based, Eq.17-18).
Calls the REAL edited method (stub self) on (a) smooth synthetic Gamma and
(b) the actual HVAB run Gamma."""
import os, sys, csv
import numpy as np
REPO="/home/wtg/01_python_project/test_0124"
sys.path.insert(0, os.path.join(REPO,"src")); sys.path.insert(0, os.path.join(REPO,"src/utilities"))
from actuator.actuator_line import ActuatorLineModel
from compare_spanwise import load_spanwise

class Stub: pass
def wcorr(r,eps,Gam,chord,dr,target="inviscid",factor=0.25):
    s=Stub(); s._eps_corr_target=target; s._eps_opt_factor=factor
    return ActuatorLineModel._viscous_core_correction(s,r,eps,Gam,chord,dr)

print("="*70)
print("(A) SMOOTH synthetic Γ — validate sign / smoothness / circulation")
print("="*70)
rR=np.linspace(0.25,1.0,48); r=rR*128.0; dr=float(np.mean(np.diff(r)))
eps=np.full(48,2.3); chord=np.full(48,3.0)
# blade-like: rises from root, peaks ~0.9, finite-but-small at tip (like HVAB)
Gam=0.35*np.sin(np.pi*np.clip((rR-0.25)/0.72,0,1))**0.7
Gam[-1]*=0.9   # finite tip
w=wcorr(r,eps,Gam,chord,dr)
Gw=np.diff(np.concatenate(([0.0],Gam,[0.0])))
print(f"ΣΓw (edge, incl. tip/root) = {Gw.sum():+.2e}  (must be ~0: circulation conserved)")
print(f"{'r/R':>6}{'Γ':>7}{'w_corr':>10}")
for st in (0.30,0.50,0.70,0.85,0.95,1.00):
    i=np.argmin(np.abs(rR-st)); print(f"{rR[i]:6.2f}{Gam[i]:7.3f}{w[i]:+10.5f}")
mono = np.all(np.diff(w[rR>0.6])>-1e-4)
print(f"팁으로 갈수록 w 증가(단조)? {mono};  w 부호 팁={np.sign(w[-1]):+.0f}(>0=다운워시=de-load ✓)")

print("\n"+"="*70)
print("(B) ACTUAL HVAB run Γ (pureALM) — magnitude vs needed ~0.0053")
print("="*70)
T2="aeromechanics_workshop/HVAB/260630/260630_results_nasa_c81"
sp=load_spanwise(os.path.join(REPO,T2,"pureALM_csv"),3.0)
geo={float(x["r_lu"]):float(x["chord_lu"]) for x in csv.DictReader(open(os.path.join(REPO,T2,"pureALM_csv/blade_diagnostics/blade_geometry.csv")))}
rr=sp["r_lu"]; epsr=sp["eps_lu"]; chordr=np.array([geo[min(geo,key=lambda g:abs(g-ri))] for ri in rr])
Gamr=0.5*chordr*sp["u_rel"]*sp["CL"]; drr=float(np.mean(np.diff(rr)))
wr=wcorr(rr,epsr,Gamr,chordr,drr)
print(f"{'r/R':>6}{'Γ':>7}{'NEW w_corr':>12}{'(old ≈)':>9}")
oldw={0.90:0.00133,0.95:-0.00236,0.97:0.00103,0.99:0.00061}
for st in (0.85,0.90,0.95,0.97,0.99):
    i=np.argmin(np.abs(sp['r_R']-st))
    print(f"{st:6.2f}{Gamr[i]:7.3f}{wr[i]:+12.5f}{oldw.get(round(st,2),float('nan')):9.5f}")
print(f"\n필요 다운워시 ~0.0053. NEW 팁 w_corr = {wr[-1]:+.5f} (old +0.00061, ~{wr[-1]/0.00061:.0f}x 강화)")

print("\n(C) 회귀: scalar vs array dr, opt target, no-crash")
w_sc=wcorr(r,eps,Gam,chord,dr)                       # scalar dr
w_ar=wcorr(r,eps,Gam,chord,np.full(48,dr))           # array dr (uniform → same)
print(f"scalar vs array dr 최대차 = {np.max(np.abs(w_sc-w_ar)):.2e} (0이어야)")
w_opt=wcorr(r,eps,Gam,chord,dr,target="opt")
print(f"target=opt 정상 실행, 팁 w={w_opt[-1]:+.5f} (inviscid보다 작아야: {abs(w_opt[-1])<abs(w[-1])})")
