"""bench5 archB WENDLAND, n_radial=24 — D40-defect discriminator smoke.

Matches the D40 case-4' marker-spacing regime on the bench5 grid: with
eps/dx = 3.4 the Wendland support is R_s = sqrt(7.5)*eps ~ 9.3 fine lu;
n_radial=24 gives dr = 128/24 ~ 5.3 fine lu, i.e. R_s/dr ~ 1.8 — the same
sparse-coupling regime as D40 N64 (R_s/dr ~ 1.9) where the compact deficit
K couples each tip marker to only ~1 neighbour per side. Hypothesis: the
kleine-straight solve develops an odd-even tip mode there (the case4w
sawtooth), seeded blade-selectively by the 2-rank fp asymmetry.
Not a physics case.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, radial_truncation=True,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_radial=24,
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_archb_wendland_n24",
)
config["actuator_line"]["kernel"] = {"type": "wendland"}
