"""
Sphere + 2 counter-rotating rotors — the small functional rig.

A compact stand-in for octo8: a bluff body with rotors above it, blowing down
onto it. Same interactions (rotor wake -> body -> outwash), ~1.5 M cells
instead of 66 M, so every capability can be exercised and re-exercised cheaply
before it is pointed at the real case.

    z
    ^     ( rotor A )   ( rotor B )     <- counter-rotating, side by side
    |          \\           /
    |            ~~ wake ~~
    |              (  )                  <- sphere
    +----> y

Layout (L0 lattice units, D_rotor = 16 cells):
    rotor hubs   (40, 30, 52) and (40, 58, 52)   28 lu apart = 1.75 D
    sphere       centre (40, 44, 28), radius 7 lu
    clearance    rotor plane sits 1.0 D above the sphere's top

Variants, chosen so each new capability gets a test the moment it lands:
    mlg="off"     uniform  — the reference every other variant is judged against
    mlg="single"  one fine box over sphere+rotors (multi-rotor fine-level ALM)
    mlg="blocks"  one fine block PER ROTOR at L2 (several blocks under ONE L1)
    mlg="nested"  one L1 PER ROTOR, each holding its own L2 (several blocks at
                  BOTH levels — the topology where a fine block's parent is
                  itself one of several)
    mlg="cut"     like "single" but the L1 floor is RAISED into the sphere, so
                  the body straddles the refinement interface — the octo8
                  airframe situation at rig scale. Rejected outright unless
                  mlg.wall_coupling.mode='exclude'; judged against "single",
                  which is the same case with the body fully enclosed.

The two rotors spin opposite ways on purpose: their torques cancel, so the
sphere sees no net swirl and any asymmetry in the answer is a bug, not physics.
"""
import numpy as np

# ── rotor (a small 2-bladed prop; flat-plate polar keeps the rig fast) ──
R_PHYS = 0.1143
D_PHYS = 2 * R_PHYS
RPM = 5000.0
OMEGA = RPM * 2 * np.pi / 60.0
D_LU = 16                                  # rotor diameter in L0 cells

_SECS = [{"r": rR * R_PHYS, "chord": 0.15 * R_PHYS, "twist": 12.0,
          "airfoil": "fp", "active": rR >= 0.2}
         for rR in np.linspace(0.0, 1.0, 11)]

# ── geometry, all in L0 lattice units ──────────────────────────────
NX, NY, NZ = 80, 88, 96
HUBS = [(40.0, 30.0, 52.0), (40.0, 58.0, 52.0)]
SPIN = ("ccw", "cw")                       # counter-rotating
# radius/height chosen so the sphere clears the L1 edge by more than
# 0.5*L_body — closer than that and the coupling interface sits inside
# the boundary layer (mlg region-padding guidance).
SPHERE_C, SPHERE_R = (40.0, 44.0, 28.0), 7.0

# one box over everything that matters
L1_BOX = {"x_min": 22, "x_max": 58, "y_min": 14, "y_max": 74,
          "z_min": 12, "z_max": 62}
# per-rotor boxes: disjoint with room for both coupling bands between them
L2_BOXES = [{"name": "rotorA", "x_min": 30, "x_max": 50,
             "y_min": 20, "y_max": 40, "z_min": 44, "z_max": 60},
            {"name": "rotorB", "x_min": 30, "x_max": 50,
             "y_min": 48, "y_max": 68, "z_min": 44, "z_max": 60}]


# "cut": the L1 floor sits at z=22, inside the sphere (which spans z 21..35).
# The C2F band (fine_domain..fine_region = z 20..22 coarse) then contains
# solid cells on the z_low face — the configuration _check_body_vs_coupling_band
# rejects, and the one wall-aware coupling exists to make runnable.
L1_BOX_CUT = dict(L1_BOX, z_min=22)


def build(mlg: str = "off", wall_bc: str = "hwbb", n_rev: float = 0.5,
          sphere: bool = True, wall_coupling: dict | None = None,
          forces: bool = False):
    """mlg ∈ {"off", "single", "blocks", "nested", "cut"}."""
    u_max = 0.1
    steps_rev = int(round(2 * np.pi * (D_LU / 2) / u_max))     # 503

    al = {
        "enabled": True,
        "rotors": [
            {"name": f"rotor{'AB'[i]}_{SPIN[i]}",
             "rotor": {"rpm": (1.0 if SPIN[i] == "cw" else -1.0) * RPM,
                       "radius": R_PHYS,
                       "omega": (1.0 if SPIN[i] == "cw" else -1.0) * OMEGA,
                       "n_blades": 2, "hub_center": list(h),
                       "rotation_axis": [0, 0, -1], "thrust_direction": [0, 0, 1],
                       "theta_0": float(i * np.pi / 4), "blade": {"sections": _SECS},
                       "grid": {"n_radial": 16}, "epsilon_chord_factor": 0.25}}
            for i, h in enumerate(HUBS)],
        "gaussian_cutoff": 3.0, "rho_ref": 1.0, "coeff_mode": "rotorcraft",
        "ramp_steps": steps_rev // 2, "prandtl_loss": False,
    }

    if mlg == "single":
        levels = [{}, {"regions": [dict(L1_BOX, name="rig")]}]
        num_levels = 2
    elif mlg == "cut":
        levels = [{}, {"regions": [dict(L1_BOX_CUT, name="rig")]}]
        num_levels = 2
    elif mlg == "nested":
        # L1 split per rotor, each with its own L2 child. fine_domain_coarse
        # of the two L1 boxes must stay disjoint: [18,42] vs [46,70].
        levels = [{},
                  {"regions": [
                      {"name": "armA", "x_min": 28, "x_max": 52,
                       "y_min": 20, "y_max": 40, "z_min": 42, "z_max": 62},
                      {"name": "armB", "x_min": 28, "x_max": 52,
                       "y_min": 48, "y_max": 68, "z_min": 42, "z_max": 62}]},
                  {"regions": [
                      # must CONTAIN the rotor disk (hub +- R = 8), else the
                      # rotor stays on L1 and L2 excises part of it — the
                      # placement guard rejects exactly that.
                      {"name": "rotorA", "x_min": 32, "x_max": 48,
                       "y_min": 22, "y_max": 38, "z_min": 46, "z_max": 58},
                      {"name": "rotorB", "x_min": 32, "x_max": 48,
                       "y_min": 50, "y_max": 66, "z_min": 46, "z_max": 58}]}]
        num_levels = 3
    elif mlg == "blocks":
        levels = [{}, {"regions": [dict(L1_BOX, name="rig")]},
                  {"regions": [dict(b) for b in L2_BOXES]}]
        num_levels = 3
    else:
        levels, num_levels = [], 1

    eq = lambda loc: {"location": loc, "method": "eq",
                      "velocity": [0.0, 0.0, 0.0]}
    boundaries = {k: eq(k) for k in ("xmin", "xmax", "ymin", "ymax", "zmax")}
    boundaries["zmin"] = {"location": "zmin", "method": "sponge",
                          "velocity": [0.0, 0.0, 0.0], "density": 1.0,
                          "thickness": 10, "strength": 0.1}

    geom = ({"sphere": {"enabled": True, "center": SPHERE_C,
                        "radius": SPHERE_R, "wall_bc": wall_bc}}
            if sphere else {"type": "none"})

    tag = f"rig_{mlg}" + ("" if sphere else "_nobody")
    return {
        "simulation": {"device_mode": "gpu", "precision": "float32",
                       "dimension": 3, "lattice_model": "D3Q27",
                       "collision_model": "cumulant"},
        "physics": {"rho": 1.225, "U_inf": 0.0, "nu": 1.461e-5,
                    "L_char": D_PHYS, "flow_direction": [0.0, 0.0, -1.0]},
        "grid": {"Nx": NX, "Ny": NY, "Nz": NZ, "resolution": D_LU},
        "numerics": {"acoustic_scaling": False, "u_max": u_max,
                     "c_s_phys": 340.3, "collision": "cumulant"},
        "boundaries": boundaries,
        "internal_geometry": geom,
        "mlg": ({"enabled": True, "num_levels": num_levels, "overlap_width": 2,
                 "interpolation": "cubic", "filter_level": 1, "levels": levels,
                 **({"wall_coupling": wall_coupling} if wall_coupling else {})}
                if num_levels > 1 else {"enabled": False}),
        "sgs": {"enabled": False, "model": "off"},
        "airfoil_polar": {"method": "flat_plate", "Re_target": 1e5},
        "actuator_line": al,
        "conservation": {"enabled": True, "verbose": 0, "log_to_csv": True},
        "convergence": {"enabled": False},
        "force_calculation": {"enabled": bool(forces)},
        "output": {"output_dir": f"./result_{tag}/vtk",
                   "checkpoint_dir": f"./result_{tag}/checkpoints",
                   "csv_dir": f"./result_{tag}/csv", "clear_previous": True,
                   "vtk": {"enabled": True, "precision": "float32",
                           "variables": ["density", "velocity",
                                         "velocity_magnitude"]},
                   "checkpoint": {"enabled": True, "keep_last_n": 1}},
        "time": {"max_steps": int(n_rev * steps_rev),
                 "output_interval": max(1, steps_rev // 8),
                 "logging_interval": max(1, steps_rev // 20),
                 "checkpoint_interval": max(1, steps_rev // 2),
                 "conservation_interval": max(1, steps_rev // 4)},
    }


if __name__ == "__main__":
    for m in ("off", "single", "blocks"):
        c = build(m)
        n = c["grid"]["Nx"] * c["grid"]["Ny"] * c["grid"]["Nz"]
        print(f"  {m:7s} L0 {n/1e6:.2f}M cells, "
              f"{c['mlg'].get('num_levels', 1)} level(s), "
              f"{c['time']['max_steps']} steps")
