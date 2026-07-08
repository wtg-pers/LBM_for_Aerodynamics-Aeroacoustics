"""End-to-end: model.step() with non-uniform marker_dr + Dağ correction ON.
Confirms the per-marker dr flows through BOTH the force calc and the correction."""
import sys
import numpy as np
sys.path.insert(0, "/home/wtg/01_python_project/test_0124")
from src.actuator.rotor import Rotor
from src.actuator.actuator_line import ActuatorLineModel

xp = np
Nx = Ny = Nz = 48
R = 16.0
HUB = (24.0, 24.0, 24.0)
N_CUT = 3.0

# synthetic downwash field (u_z = -0.1 inside a disk-ish region, 0 outside)
u = np.zeros((3, Nx, Ny, Nz))
zz, yy, xx = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
rcyl = np.sqrt((xx - HUB[0])**2 + (yy - HUB[1])**2)
u[2] = np.where(rcyl < R, -0.1, 0.0)

def pq(alpha_deg, Re, name=None):
    return (0.6, 0.012)

def build_and_step(dist, side="both"):
    rotor = Rotor.from_simple(
        n_blades=2, radius=R, hub_radius=4.0, chord=3.0,
        twist_root=12.0, twist_tip=2.0, omega=0.02,
        hub_center=HUB, n_radial=20, dx=1.0, rotation_axis=(0, 0, 1),
    )
    for b in rotor.blades:                              # re-place markers
        b.generate_markers(20, distribution=dist, cosine_side=side)
        b.set_lattice_spacing(dx=1.0)
    mdl = ActuatorLineModel(
        rotor=rotor, nu=0.01, domain_shape=(Nx, Ny, Nz), polar_query=pq,
        rho_ref=1.0, n_cut=N_CUT, dx_phys=1.0, dt_phys=1.0,
        u_inf_lu=0.1, coeff_mode='rotorcraft', xp=xp, sound_speed=1.0,
    )
    mdl._eps_corr = True                                # Dağ correction ON
    mdl._eps_corr_method = "dag"
    mdl._eps_corr_target = "inviscid"
    Fg = np.asarray(mdl.step(u, 1.0))
    md = rotor.blades[0].marker_dr
    return Fg, md, mdl._last_bem_result

print("dist        dr_type   Σdr      finite  Σ|F|        w_corr(tip)")
for dist, side in (("uniform", "both"), ("cosine", "both"),
                   ("cosine", "tip"), ("endpoint", "both")):
    Fg, md, bem = build_and_step(dist, side)
    dr_type = "scalar" if np.ndim(md) == 0 else "array"
    sdr = float(np.sum(md)) if np.ndim(md) else float(md * 20)
    finite = bool(np.all(np.isfinite(Fg)))
    sF = float(np.sum(np.abs(Fg)))
    wtip = float(bem.w_corr[19]) if bem.w_corr is not None else float('nan')
    tag = f"{dist}/{side}" if dist == "cosine" else dist
    print(f"{tag:11} {dr_type:7}  {sdr:6.2f}   {finite!s:5}   {sF:.4e}   {wtip:+.3e}")
print("\nPASS if: all finite, Σ|F|>0, uniform=scalar / cosine,endpoint=array, "
      "Dağ w_corr nonzero (correction ran with per-marker dr).")
