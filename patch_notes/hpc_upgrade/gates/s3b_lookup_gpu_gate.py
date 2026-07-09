"""S3b gate: ActuatorLineModel._lookup_cl_cd xp-agnostic, GPU vs CPU.

Exercises the xp mechanics (zeros/asarray/arange/valid-mask/group-index/batch
query) via the single-airfoil path on a real NASA C81 deck. Multi-airfoil is
the same per-group qf calls (each proven bit-close in S1). Fake self/blade so
no full rotor build needed.
"""
import sys, os, glob, types
import numpy as np
import cupy as cp
sys.path.insert(0, os.path.abspath("src"))
from actuator.actuator_line import ActuatorLineModel
from actuator.c81_loader import C81PolarSet, make_c81_polar_query_mach

deck = sorted(glob.glob("data/airfoils/hvab_nasa_overflow/*RC4-10*.c81"))[0]
pq = make_c81_polar_query_mach(C81PolarSet.from_file(deck))

n = 48
rng = np.random.default_rng(5)
alpha = rng.uniform(-6, 14, n)
Re = rng.uniform(2e5, 2e6, n)
u_rel = rng.uniform(0.5, 3.0, n)
Mach = rng.uniform(0.05, 0.65, n)
active = np.ones(n, bool); active[:2] = False
u_rel[5] = 0.0                                   # a marker below the 1e-10 gate

fake_self = types.SimpleNamespace(_multi_airfoil=False, polar_query=pq)
fake_blade = types.SimpleNamespace(marker_active=active)
lookup = ActuatorLineModel._lookup_cl_cd               # unbound

# CPU
CL_c, CD_c = lookup(fake_self, alpha, Re, u_rel, Mach, fake_blade, n)
# GPU (cupy inputs)
g = lambda x: cp.asarray(x)
CL_g, CD_g = lookup(fake_self, g(alpha), g(Re), g(u_rel), g(Mach), fake_blade, n)
assert isinstance(CL_g, cp.ndarray), "GPU path did not return cupy"

dCL = np.abs(CL_c - cp.asnumpy(CL_g)).max()
dCD = np.abs(CD_c - cp.asnumpy(CD_g)).max()
print(f"deck={os.path.basename(deck)}  active={active.sum()}/{n}  (1 u_rel=0 gated)")
print(f"CL_c range {CL_c[active].min():.3f}..{CL_c[active].max():.3f}")
print(f"CL max|Δ| = {dCL:.3e}   CD max|Δ| = {dCD:.3e}")
print(f"inactive+gated zero (CPU): CL {np.all(CL_c[~active]==0)}, marker5 CL={CL_c[5]:.1e}")
ok = max(dCL, dCD) < 1e-12
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
