"""Q2 verification: free-wake tip-only shedding.
(a) shedding subset → ring point count; (b) free-wake runs end-to-end for
'all' and 'tip'; (c) 'all'+scalar dr keeps the legacy A path; (d) shed_idx resolve."""
import sys
import numpy as np
sys.path.insert(0, "/home/wtg/01_python_project/test_0124")
from src.actuator.rotor import Rotor
from src.actuator.actuator_line import ActuatorLineModel

Nx = Ny = Nz = 48
u = np.zeros((3, Nx, Ny, Nz))
zz, yy, xx = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
rcyl = np.sqrt((xx-24.)**2 + (yy-24.)**2)
u[2] = np.where(rcyl < 16, -0.08, 0.0)

def build():
    rotor = Rotor.from_simple(n_blades=2, radius=16.0, hub_radius=4.0, chord=3.0,
        twist_root=12.0, twist_tip=2.0, omega=0.02, hub_center=(24.,24.,24.),
        n_radial=20, dx=1.0, rotation_axis=(0,0,1))
    mdl = ActuatorLineModel(rotor=rotor, nu=0.01, domain_shape=(Nx,Ny,Nz),
        polar_query=lambda a,Re,name=None:(0.6,0.012), rho_ref=1.0, n_cut=3.0,
        dx_phys=1.0, dt_phys=1.0, u_inf_lu=0.1, coeff_mode='rotorcraft', xp=np, sound_speed=1.0)
    mdl._eps_corr = True; mdl._eps_corr_method = "kleine"
    mdl._kleine_wake_mode = "free"; mdl._polar_wants_mach = False
    return mdl, rotor

# (d) shed_idx resolution
mdl, rotor = build(); bl = rotor.blades[0]; npb = rotor.markers_per_blade
for spec, exp in (("all", None), ("tip", 1), (3, 3), (0.9, None)):
    mdl._kleine_wake_markers = spec
    idx = mdl._shed_idx(bl)
    got = "None" if idx is None else len(idx)
    print(f"(d) wake_markers={str(spec):5} -> shed {got} markers"
          + (f" (idx {idx})" if idx is not None and len(idx)<=3 else ""))

# (a,b) run free-wake end-to-end for all / tip
print()
for markers in ("all", "tip"):
    mdl, rotor = build()
    mdl._kleine_wake_markers = markers
    Fg = None
    for s in range(6):
        Fg = np.asarray(mdl.step(u, 1.0))
    ne = mdl._kleine_wake[0].rings[0].shape[0]
    nrings = len(mdl._kleine_wake[0].rings)
    A = mdl._kleine_A_free.get(0)
    wc = mdl._last_bem_result.w_corr
    print(f"(a/b) markers={markers:4}: ring_points={ne} (npb={npb}) nrings={nrings} "
          f"A.shape={None if A is None else A.shape} F_finite={np.all(np.isfinite(Fg))} "
          f"w_corr_finite={np.all(np.isfinite(wc))} |w_corr|_tip={abs(wc[npb-1]):.3e}")

# (c) 'all'+scalar dr uses legacy A = dr*(B@G) — verify against manual recompute
mdl, rotor = build(); mdl._kleine_wake_markers = "all"
for s in range(4):
    mdl.step(u, 1.0)
# rebuild A manually from the cached pieces and compare
from src.actuator.smearing_correction import freewake_influence, _gradient_matrix
bl = rotor.blades[0]; n = rotor.markers_per_blade
r = bl.marker_r; eps = bl.marker_epsilon; dr = bl.marker_dr
ctrl = mdl._last_positions[0:n]
axis = -np.sign(rotor.omega)*np.asarray(rotor.rotation_axis, float)
B = freewake_influence(ctrl, mdl._kleine_wake[0].rings, eps, axis)
A_legacy = dr * (B @ _gradient_matrix(r))
A_code = mdl._kleine_A_free[0]
print(f"\n(c) 'all'+scalar A == legacy dr*(B@G): maxabs={np.max(np.abs(A_code-A_legacy)):.2e} "
      f"(scalar dr={np.ndim(dr)==0})")
