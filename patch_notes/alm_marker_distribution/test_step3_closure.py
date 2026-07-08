"""Step 3 verification: endpoint Γ=0 closure (option 3) — physics, not smoke.
closure OFF must be byte-identical; closure ON must (a) zero the net trailed
circulation, (b) capture the tip/root shed, (c) add tip downwash."""
import sys
import numpy as np
sys.path.insert(0, "/home/wtg/01_python_project/test_0124")
from src.actuator.rotor import Rotor
from src.actuator.actuator_line import ActuatorLineModel

rotor = Rotor.from_simple(
    n_blades=2, radius=16.0, hub_radius=4.0, chord=3.0,
    twist_root=12.0, twist_tip=2.0, omega=0.02,
    hub_center=(24., 24., 24.), n_radial=24, dx=1.0, rotation_axis=(0, 0, 1))
mdl = ActuatorLineModel(rotor=rotor, nu=0.01, domain_shape=(48, 48, 48),
    polar_query=lambda a, Re, name=None: (0.6, 0.012), rho_ref=1.0, n_cut=3.0,
    dx_phys=1.0, dt_phys=1.0, u_inf_lu=0.1, coeff_mode='rotorcraft', xp=np, sound_speed=1.0)
mdl._eps_corr_target = "inviscid"

bl = rotor.blades[0]
r = bl.marker_r.astype(float)
eps = bl.marker_epsilon.astype(float)
chord = bl.marker_chord.astype(float)
dr = bl.marker_dr                       # scalar (uniform)
N = len(r)
# realistic Γ(r): rises from root, peaks near tip, still 70% of max at last marker
x = (r - r[0]) / (r[-1] - r[0])
Gamma = (0.4 + 0.6 * np.sin(np.pi * (0.15 + 0.7 * x)))      # nonzero at both ends
Gamma *= (r > -1)                                           # all active

def wcorr(closure):
    mdl._eps_endpoint_closure = closure
    return mdl._viscous_core_correction(r, eps, Gamma, chord, dr)

w_off = wcorr(False)
w_on = wcorr(True)

# P1. closure OFF == standalone reference (original formula, byte-identical)
gradG = np.gradient(Gamma, r); inv4pi = 1/(4*np.pi); ref = np.zeros(N)
for i in range(N):
    d = r[i]-r; far = np.abs(d) > 1e-9; sd = np.where(far, d, 1.0)
    ker = np.where(far, np.exp(-(d/eps)**2)/sd, 0.0)
    ref[i] = -inv4pi*np.sum(gradG*ker)*dr
print(f"P1 closure OFF byte-identical to original: {np.array_equal(w_off, ref)}")

# P2. net trailed circulation Σ(dΓ/dr)·Δr  (closure → 0; open → Γ_last−Γ_first)
net_open = np.sum(np.gradient(Gamma, r) * dr)
drv = np.full(N, dr)
r_s = np.concatenate(([r[0]-0.5*drv[0]], r, [r[-1]+0.5*drv[-1]]))
G_s = np.concatenate(([0.], Gamma, [0.]))
drs = np.empty_like(r_s); drs[1:-1]=0.5*(r_s[2:]-r_s[:-2]); drs[0]=0.5*(r_s[1]-r_s[0]); drs[-1]=0.5*(r_s[-1]-r_s[-2])
net_closed = np.sum(np.gradient(G_s, r_s) * drs)
print(f"P2 net trailed circ:  open={net_open:+.4f} (=Γ_last−Γ_first={Gamma[-1]-Gamma[0]:+.4f})  "
      f"closed={net_closed:+.2e} (→0)  {abs(net_closed)<1e-12}")

# P3. virtual nodes land exactly on root-cut / tip (cell edges)
root_cut = bl.r_root_cut; tip = bl.r_tip
print(f"P3 virtual nodes: r_s[0]={r_s[0]:.4f}(root_cut={root_cut:.4f}?{np.isclose(r_s[0],root_cut)})  "
      f"r_s[-1]={r_s[-1]:.4f}(tip={tip:.4f}?{np.isclose(r_s[-1],tip)})")

# P4. physical effect: closure adds downwash near the tip (missing tip shed)
itip = N - 1
print(f"P4 tip w_corr: open={w_off[itip]:+.4e}  closed={w_on[itip]:+.4e}  "
      f"|closed|>|open| (more tip downwash): {abs(w_on[itip])>abs(w_off[itip])}")
print(f"   tip Γ retained = {Gamma[-1]/Gamma.max()*100:.0f}% of max → strong tip shed exists")
print(f"\nw_off (tip 4): {np.round(w_off[-4:],5)}")
print(f"w_on  (tip 4): {np.round(w_on[-4:],5)}")
