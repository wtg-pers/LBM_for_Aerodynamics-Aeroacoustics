"""Audit our Dag implementation vs the paper Eq.17 (Dag & Sorensen 2020).
Paper: wake vortices at N+1 EDGES, Gamma_w(j)=Gamma(j)-Gamma(j-1) [incl. root & TIP
edges -> tip vortex = -Gamma_tip], prefactor +1/(4pi), kernel exp(-(d/eps)^2)/d.
Our code: dGamma/dr at N MARKERS (misses edges/tip vortex), prefactor -1/(4pi).
Apply both to the HVAB tip loading and compare to the ~0.005 downwash needed."""
import os, sys, csv
import numpy as np
REPO="/home/wtg/01_python_project/test_0124"
sys.path.insert(0, os.path.join(REPO,"src/utilities"))
from compare_spanwise import load_spanwise
T2="aeromechanics_workshop/HVAB/260630/260630_results_nasa_c81"
sp=load_spanwise(os.path.join(REPO,T2,"pureALM_csv"),3.0)
geo={float(x["r_lu"]):float(x["chord_lu"]) for x in csv.DictReader(open(os.path.join(REPO,T2,"pureALM_csv/blade_diagnostics/blade_geometry.csv")))}
r=sp["r_lu"]; eps=sp["eps_lu"]; CL=sp["CL"]; urel=sp["u_rel"]
chord=np.array([geo[min(geo,key=lambda g:abs(g-ri))] for ri in r])
Gam=0.5*chord*urel*CL
N=len(r); dr=np.mean(np.diff(r)); inv4pi=1/(4*np.pi)

def OURS(sign=-1):   # marker dGamma/dr (our current code)
    g=np.gradient(Gam,r); w=np.zeros(N)
    for i in range(N):
        d=r[i]-r; far=np.abs(d)>1e-9; sd=np.where(far,d,1.0)
        w[i]=sign*inv4pi*np.sum(g*np.where(far,np.exp(-(d/eps)**2)/sd,0.0))*dr
    return w

def PAPER(sign=+1):  # Eq.17: N+1 edge vortices, Gamma_w = Gamma(j)-Gamma(j-1)
    # edges (0..N): root=r[0]-dr/2, interior midpoints, tip=r[-1]+dr/2
    edges=np.concatenate(([r[0]-0.5*dr],0.5*(r[:-1]+r[1:]),[r[-1]+0.5*dr]))
    Gpad=np.concatenate(([0.0],Gam,[0.0]))          # Gamma(0)=Gamma(N+1)=0
    Gw=np.diff(Gpad)                                  # Gamma(j)-Gamma(j-1), len N+1
    eps_e=np.concatenate(([eps[0]],0.5*(eps[:-1]+eps[1:]),[eps[-1]]))
    w=np.zeros(N)
    for i in range(N):
        d=r[i]-edges; far=np.abs(d)>1e-9; sd=np.where(far,d,1.0)
        w[i]=sign*inv4pi*np.sum(Gw*np.where(far,np.exp(-(d/eps_e)**2)/sd,0.0))
    return w

w_ours=OURS(-1)
w_paper_p=PAPER(+1)   # paper sign
w_paper_m=PAPER(-1)   # paper formula but our sign
need=abs(urel[-1])*np.tan(np.radians(3))   # downwash needed for ~3deg tip deficit
print(f"팁 Δα~3° 회복에 필요한 다운워시(+) ≈ {need:.5f}\n")
print(f"{'r/R':>6}{'Γ':>7}{'OURS(-)':>10}{'PAPER(+)':>10}{'PAPER(-)':>10}")
for st in (0.85,0.90,0.95,0.97,0.99):
    i=np.argmin(np.abs(sp['r_R']-st))
    print(f"{st:6.2f}{Gam[i]:7.3f}{w_ours[i]:10.5f}{w_paper_p[i]:10.5f}{w_paper_m[i]:10.5f}")
print(f"\n판정: 팁서 (+)=다운워시=de-load(맞는 방향), 크기 ~{need:.4f} 근처여야 정상.")
print(f"  OURS 팁: {w_ours[-1]:+.5f}   PAPER(+) 팁: {w_paper_p[-1]:+.5f}   PAPER(-) 팁: {w_paper_m[-1]:+.5f}")
# net trailed circulation check
edges=np.concatenate(([r[0]-0.5*dr],0.5*(r[:-1]+r[1:]),[r[-1]+0.5*dr]))
Gw=np.diff(np.concatenate(([0.0],Gam,[0.0])))
print(f"\nΣΓw (paper, 끝점 포함) = {Gw.sum():.3e} (물리적으로 0; 순환보존). "
      f"Σ(dΓ/dr)·dr (ours) = {np.sum(np.gradient(Gam,r))*dr:+.4f} (=Γ_tip−Γ_root≠0, 팁와류 누락)")
