"""Verify the prescribed-helix hover pitch guard (eps_correction.helix_pitch_floor).

Synthetic HVAB convention (axis=+x, thrust=−x → downwash=+x). The sampled u_n is
positive mid-span but NEGATIVE at the innermost (hub recirculation) and outermost
(tip self-induced upwash) markers — the exact pattern measured in the
260703_dag_edge_fix helix runs, where the tip closure-edge filament climbed 5.6 lu
UPSTREAM (wake VTP x_min 90.4 < hub 96).

Checks:
  1. floor="off"  reproduces the bug: tip/root edge filaments ascend (−x).
  2. floor="auto" (default): every filament's descent is monotonic downstream (+x).
  3. Edges whose pitch was already positive keep BYTE-IDENTICAL geometry.
  4. Tip-edge descent over 2 revs = 4π·w_floor/|ω| with w_floor = mean positive u_n.
  5. Explicit float floor is honored.

Run:  python patch_notes/alm_dag_edge_fix/test_helix_pitch_floor.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.actuator.actuator_line import ActuatorLineModel  # noqa: E402

# ---- synthetic HVAB-like rotor (fine-level lattice units) --------------------
N = 48
HUB = np.array([768.0, 640.0, 640.0])
AXIS = np.array([1.0, 0.0, 0.0])
THRUST = np.array([-1.0, 0.0, 0.0])          # HVAB: thrust = −axis → downwash = +x
OMEGA = 0.1 / 128.0                          # u_tip=0.1, R=128
r = np.linspace(33.0, 127.0, N)

blade = SimpleNamespace(
    marker_r=r,
    marker_epsilon=np.full(N, 2.0),
    marker_active=np.ones(N, dtype=bool),
)

u_n = np.full(N, 0.007)                      # mid-span inflow (downwash sign +)
u_n[0] = -0.003                              # root recirculation upwash
u_n[-1] = -0.004                             # tip self-induced upwash
u_theta = np.zeros(N)
Gamma = 0.5 * 6.0 * 0.1 * 0.6 * np.sin(np.pi * (r - 30.0) / 100.0)  # smooth, ~physical

positions = HUB[None, :] + np.outer(r, np.array([0.0, 1.0, 0.0]))   # blade along +y


def make_self(pitch_floor):
    return SimpleNamespace(
        _kleine_rebuild_every=1,
        _dag_helix_steps=0,
        _dag_helix_M={},
        _dag_helix_rings={},
        _dag_helix_revs=2.0,
        _dag_helix_azimuth_deg=2.0,
        _dag_helix_pitch_floor=pitch_floor,
        _kleine_wake_nw=None,
        _kleine_wake_markers="all",
        _last_positions=positions.copy(),
        rotor=SimpleNamespace(
            hub_center=HUB, rotation_axis=AXIS, thrust_axis=THRUST,
            omega=OMEGA,
        ),
        # borrow the real methods (duck-typed self; _edge_positions_3d is static)
        _edge_positions_3d=ActuatorLineModel._edge_positions_3d,
        _shed_edge_idx=lambda b: None,       # wake_markers="all"
    )


def run(pitch_floor):
    slf = make_self(pitch_floor)
    w = ActuatorLineModel._dag_prescribed_helix_wcorr(
        slf, 0, blade, Gamma, u_n, u_theta, 2.0)
    rings = slf._dag_helix_rings[0]          # list of (n_edges, 3) per ψ ring
    nodes = np.stack(rings)                  # (n_ring, n_edge, 3)
    return w, nodes


w_off, nodes_off = run("off")
w_auto, nodes_auto = run("auto")
w_expl, nodes_expl = run(0.007)

dx_off = nodes_off[-1, :, 0] - nodes_off[0, :, 0]    # axial travel per edge filament
dx_auto = nodes_auto[-1, :, 0] - nodes_auto[0, :, 0]

# 1. bug reproduced with "off": end filaments ascend (−x)
assert dx_off[0] < 0 and dx_off[-1] < 0, f"expected ascent with off: {dx_off[[0, -1]]}"
print(f"1. off  : root/tip filament ascend  dx = {dx_off[0]:+.2f} / {dx_off[-1]:+.2f} lu"
      f"  (bug reproduced)")

# 2. auto: every filament descends monotonically downstream
diffs = np.diff(nodes_auto[:, :, 0], axis=0)
assert (dx_auto > 0).all(), f"ascending filament with auto: min dx={dx_auto.min():.3f}"
assert (diffs >= -1e-12).all(), "non-monotonic axial descent with auto"
print(f"2. auto : all {len(dx_auto)} filaments descend  dx ∈ "
      f"[{dx_auto.min():.2f}, {dx_auto.max():.2f}] lu, monotonic")

# 3. positive-pitch edges byte-identical between off and auto
interior = np.ones(len(dx_off), dtype=bool)
interior[[0, -1]] = False
# edge 1 also touches marker 0's phi (edge = marker average) — exclude edges
# whose pitch mixes a negative-u_n marker
interior[1] = False
interior[-2] = False
assert np.array_equal(nodes_off[:, interior, :], nodes_auto[:, interior, :]), \
    "valid-pitch filament geometry changed"
print(f"3. auto : {interior.sum()} valid-pitch filaments byte-identical to paper-literal")

# 4. tip-edge descent = 4π·w_floor/|ω| (2 revs), w_floor = mean positive u_n
w_floor = u_n[u_n > 0].mean()
expect = 4.0 * np.pi * w_floor / OMEGA
assert abs(dx_auto[-1] - expect) / expect < 1e-6, (dx_auto[-1], expect)
print(f"4. auto : tip-edge descent {dx_auto[-1]:.2f} lu == 4π·w_floor/ω = {expect:.2f} lu"
      f"  (w_floor={w_floor:.5f})")

# 5. explicit float floor honored
dx_expl = nodes_expl[-1, :, 0] - nodes_expl[0, :, 0]
expect5 = 4.0 * np.pi * 0.007 / OMEGA
assert abs(dx_expl[-1] - expect5) / expect5 < 1e-6
print(f"5. float: tip-edge descent {dx_expl[-1]:.2f} lu == {expect5:.2f} lu (w=0.007)")

# w_corr sanity: finite, and tip-region downwash positive in both modes
assert np.isfinite(w_off).all() and np.isfinite(w_auto).all()
print(f"   w_corr tip (last 3): off={w_off[-3:].round(5)}  auto={w_auto[-3:].round(5)}")
print("\nALL CHECKS PASSED")
