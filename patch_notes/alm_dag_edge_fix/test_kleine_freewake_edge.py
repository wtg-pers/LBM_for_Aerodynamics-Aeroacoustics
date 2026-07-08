"""Validate the edge-based free-wake fix: in the STRAIGHT semi-infinite limit,
B_edge @ E must reproduce the edge-based influence_matrix (the free-wake just
curves those filaments). Also checks _edge_positions_3d and _shed_edge_idx."""
import os, sys, csv
import numpy as np
REPO="/home/wtg/01_python_project/test_0124"
sys.path.insert(0,os.path.join(REPO,"src")); sys.path.insert(0,os.path.join(REPO,"src/utilities"))
from actuator.smearing_correction import influence_matrix, edge_operator, freewake_influence
from actuator.actuator_line import ActuatorLineModel
from compare_spanwise import load_spanwise

# real run geometry
T2="aeromechanics_workshop/HVAB/260630/260630_results_nasa_c81"
sp=load_spanwise(os.path.join(REPO,T2,"pureALM_csv"),3.0)
r=sp["r_lu"]; eps=sp["eps_lu"]; n=len(r); dr=float(np.mean(np.diff(r)))

print("(1) import/syntax OK (edge_operator, freewake_influence imported)")

# --- straight-limit: markers along x; filaments trail in +y (streamwise);
#     downwash along z. B_edge@E should == influence_matrix (edge) ---
ctrl=np.column_stack([r, np.zeros(n), np.zeros(n)])          # markers on x-axis
edge_pos=ActuatorLineModel._edge_positions_3d(ctrl)          # (n+1,3)
E,r_edge,eps_edge=edge_operator(r,eps,dr)
# semi-infinite straight filament from each edge, trailing +y (2 rings + tail)
D=5.0
rings=[edge_pos.copy(), edge_pos+np.array([0,D,0])]          # newest-first
A_ref=influence_matrix(r,eps,dr)
for ax in ([0,0,1.0],[0,0,-1.0]):
    B=freewake_influence(ctrl, rings, eps_edge, np.array(ax,float))   # (n, n+1)
    A_free=B@E
    err=np.max(np.abs(A_free-A_ref))/(np.max(np.abs(A_ref))+1e-30)
    print(f"   axial={ax}: max rel|B@E − influence_matrix| = {err:.3e}")

print("\n(2) _edge_positions_3d: n markers → n+1 edges, 끝점 외삽 확인")
print(f"   markers x[0,1,-2,-1]={ctrl[[0,1,-2,-1],0]}")
print(f"   edges   x[0,1,-2,-1]={edge_pos[[0,1,-2,-1],0]}  (edge[0]<marker[0], edge[-1]>marker[-1])")
print(f"   edge[0]={edge_pos[0,0]:.3f} vs r[0]-dr/2={r[0]-dr/2:.3f}; edge[-1]={edge_pos[-1,0]:.3f} vs r[-1]+dr/2={r[-1]+dr/2:.3f}")

print("\n(3) _shed_edge_idx: all / tip")
class B: pass
bl=B(); bl.marker_r=r; bl.marker_active=np.ones(n,bool); bl.r_tip=r[-1]/0.9922
m=ActuatorLineModel.__new__(ActuatorLineModel)
for spec,exp in [("all",None),("tip",[n])]:
    m._kleine_wake_markers=spec
    got=m._shed_edge_idx(bl)
    print(f"   wake_markers={spec!r:6} → {got}  (기대 {exp})")
# tip-only A: only tip edge => carries −Γ_tip
m._kleine_wake_markers="tip"; e_idx=m._shed_edge_idx(bl)
B_tip=freewake_influence(ctrl, [edge_pos[e_idx], edge_pos[e_idx]+np.array([0,D,0])], eps_edge[e_idx], np.array([0,0,1.0]))
A_tip=B_tip@E[e_idx,:]
print(f"   tip-only A shape={A_tip.shape}, 팁 마커 자기유도 A_tip[-1,-1]={A_tip[-1,-1]:+.4f} (팁와류만)")
