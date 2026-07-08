"""Diagnose why the Dag correction is weak on the light grid.
(1) dag vs pureALM tip alpha/u_n (did the correction change anything?)
(2) compute the Dag w_corr directly from the tip loading and compare to the
    downwash NEEDED to fix the ~+3deg tip induced deficit.
(3) test the user hypothesis: does a tip Gamma=0 node change dGamma/dr at the tip?"""
import os, sys, csv
import numpy as np
REPO="/home/wtg/01_python_project/test_0124"
sys.path.insert(0, os.path.join(REPO,"src/utilities"))
from compare_spanwise import load_spanwise

T2="aeromechanics_workshop/HVAB/260630/260630_results_nasa_c81"
sp=load_spanwise(os.path.join(REPO,T2,"pureALM_csv"),3.0)
sd=load_spanwise(os.path.join(REPO,T2,"dag_csv"),3.0)
print("(1) Task2 dag vs pureALM at tip — did Dag change the inflow?")
print(f"{'r/R':>6}{'α_pure':>8}{'α_dag':>8}{'Δα':>7}{'un_pure':>9}{'un_dag':>9}{'Δun':>9}")
for st in (0.90,0.95,0.99):
    ap=np.interp(st,sp['r_R'],sp['alpha']); ad=np.interp(st,sd['r_R'],sd['alpha'])
    up=np.interp(st,sp['r_R'],sp['u_n']); ud=np.interp(st,sd['r_R'],sd['u_n'])
    print(f"{st:6.2f}{ap:8.2f}{ad:8.2f}{ad-ap:+7.2f}{up:9.5f}{ud:9.5f}{ud-up:+9.5f}")

# (2) compute Dag w_corr from pureALM loading (lattice units)
import csv as _c
geo={float(x["r_lu"]):float(x["chord_lu"]) for x in _c.DictReader(open(os.path.join(REPO,T2,"pureALM_csv/blade_diagnostics/blade_geometry.csv")))}
r=sp["r_lu"]; eps=sp["eps_lu"]; CL=sp["CL"]; urel=sp["u_rel"]; utheta=sp["u_theta"]
chord=np.array([geo[min(geo,key=lambda g:abs(g-ri))] for ri in r])
Gam=0.5*chord*urel*CL
dr=np.mean(np.diff(r))
inv4pi=1/(4*np.pi)
def dag_wcorr(rr,Gam,eps,dr,tip_node=False):
    if tip_node:
        rr=np.concatenate([rr,[rr[-1]+0.5*dr]]); Gam=np.concatenate([Gam,[0.0]]); eps=np.concatenate([eps,[eps[-1]]])
    g=np.gradient(Gam,rr); N=len(rr); w=np.zeros(N)
    for i in range(N):
        d=rr[i]-rr; far=np.abs(d)>1e-9; sd=np.where(far,d,1.0)
        ker=np.where(far,np.exp(-(d/eps)**2)/sd,0.0)
        w[i]=-inv4pi*np.sum(g*ker)*dr
    return w
w=dag_wcorr(r,Gam,eps,dr)
wt=dag_wcorr(r,Gam,eps,dr,tip_node=True)  # user hypothesis: add Gamma=0 tip node
print("\n(2)(3) Dag w_corr [lattice u_n units] + tip-node test:")
print(f"{'r/R':>6}{'Γ':>8}{'dΓ/dr':>9}{'ε':>6}{'w_dag':>10}{'w_dag+tipnode':>14}{'un(actual)':>11}")
for st in (0.90,0.95,0.97,0.99):
    i=np.argmin(np.abs(sp['r_R']-st))
    dgdr=np.gradient(Gam,r)[i]
    print(f"{st:6.2f}{Gam[i]:8.3f}{dgdr:9.4f}{eps[i]:6.2f}{w[i]:10.5f}{wt[i] if i<len(wt) else float('nan'):14.5f}{sp['u_n'][i]:11.5f}")
# needed downwash for Δα~3deg at tip
utan_tip=abs(utheta[-1])+abs(urel[-1])  # ~ tangential scale
print(f"\n필요 다운워시(Δα=3° 회복, u_tan≈{abs(urel[-1]):.3f}): Δu_n ≈ {abs(urel[-1])*np.tan(np.radians(3)):.5f}")
print(f"Dag가 주는 팁 w_corr ≈ {w[-1]:.5f}  →  {'충분' if abs(w[-1])>0.003 else '너무 작음(약함 확인)'}")
