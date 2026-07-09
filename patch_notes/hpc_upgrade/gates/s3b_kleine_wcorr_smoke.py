"""S3b integration smoke: _kleine_w_corr full path, GPU (ALM_CORR_GPU=1) vs CPU.

Drives the real method with a minimal fake self/rotor/blade + a synthetic free
wake, exercising: free-wake GPU build (A resident) -> device conversion ->
warm-start -> cl/dcl closures (GPU C81 LUT) -> correct_noniterative (GPU
linalg) -> safety net -> D2H return. Verifies it RUNS and w_gpu ~ w_cpu (the
free-wake + solve reduction-order differs ~1e-11, not a bug).
"""
import sys, os, glob, types
import numpy as np
import cupy as cp
sys.path.insert(0, os.path.abspath("src"))
from actuator.actuator_line import ActuatorLineModel
from actuator.smearing_correction import FreeWake, edge_operator
from actuator.c81_loader import C81PolarSet, make_c81_polar_query_mach

deck = sorted(glob.glob("data/airfoils/hvab_nasa_overflow/*RC4-10*.c81"))[0]
pq = make_c81_polar_query_mach(C81PolarSet.from_file(deck))

n = 12
r = np.linspace(1.0, 8.0, n)
chord = np.full(n, 0.5)
dr = float(np.mean(np.diff(r)))
eps = np.full(n, 2.0)
twist = np.full(n, 10.0)
active = np.ones(n, bool)
u_n = np.full(n, 1.2)
u_theta = np.full(n, 0.3)
CL = np.full(n, 0.4)
cos_sweep = 1.0
ne = n + 1                                        # trailed edges (N+1)

# synthetic helical free wake: L rings of ne edge-nodes
E, r_edge, eps_edge = edge_operator(r, eps, dr)
def ring(a):
    c, s = np.cos(-a * 0.2), np.sin(-a * 0.2)
    pts = np.stack([r_edge, np.zeros(ne), np.zeros(ne)], 1)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    pts = pts @ Rz.T; pts[:, 2] -= a * 0.15
    return pts
positions = np.stack([r, np.zeros(n), np.full(n, 0.05)], 1)   # blade markers

def make_fake():
    wake = FreeWake(n_w=60)
    wake.rings = [ring(a) for a in range(40)]     # filled wake
    rotor = types.SimpleNamespace(omega=2.0, rotation_axis=np.array([0.,0,1]),
                                  thrust_axis=np.array([0.,0,-1]), n_blades=1)
    f = types.SimpleNamespace(
        rotor=rotor, nu=1e-5, a_phys=340.0, dx_phys=1e-3, dt_phys=1e-6,
        _polar_wants_mach=True, _multi_airfoil=False, polar_query=pq,
        _kleine_wake={0: wake}, _kleine_wake_mode="free",
        _kleine_A_free={}, _kleine_A={}, _kleine_rebuild_every=1,
        _kleine_wake_steps=0, _kleine_wake_markers=None,
        _kleine_gamma_prev={}, _kleine_w_prev={}, _kleine_fallback_count=0,
        _eps_corr_smooth=0, _last_positions=positions.copy(),
        _pf_bem=False, _pf_shapes=None,
        _pf_acc={'freewake':0.,'solve':0.,'polar':0.,'other':0.,'n':0},
        _polar_groups={}, _polar_qf_cache={})
    for m in ('_lookup_cl_cd', '_shed_edge_idx', '_smooth_active', '_kleine_w_corr'):
        setattr(f, m, types.MethodType(getattr(ActuatorLineModel, m), f))
    blade = types.SimpleNamespace(
        marker_r=r, marker_epsilon=eps, marker_twist=twist,
        marker_chord=chord, marker_active=active, r_tip=8.0)
    return f, blade

def run(env):
    os.environ['ALM_CORR_GPU'] = env
    f, blade = make_fake()
    return f._kleine_w_corr(0, blade, u_n, u_theta, cos_sweep, chord, dr, CL, active, n)

w_cpu = run('0')
w_gpu = run('1')
print(f"w_cpu type={type(w_cpu).__name__}  w_gpu type={type(w_gpu).__name__}")
assert isinstance(w_gpu, np.ndarray), "GPU path must D2H return numpy to caller"
d = np.abs(w_cpu - w_gpu).max()
print(f"|w|~{np.abs(w_cpu).max():.3e}   w_gpu vs w_cpu max|Δ| = {d:.3e}")
ok = (d < 1e-9) and np.all(np.isfinite(w_gpu))
print("RESULT:", "PASS (GPU path runs, ~CPU)" if ok else "FAIL")
sys.exit(0 if ok else 1)
