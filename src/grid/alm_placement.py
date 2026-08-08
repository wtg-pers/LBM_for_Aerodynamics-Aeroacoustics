"""
ALM ↔ MLG placement geometry — which refinement level owns each rotor.

Pure numpy, no cupy, no Simulation: importable and unit-testable without a GPU.

WHY THIS EXISTS
---------------
An actuator-line model injects its body force into exactly ONE level's grid.
MLG's F2C coupling then overwrites the coarse `excised` region with fine-level
data (`multi_level_grid.py`, "F→C feedback"). A rotor whose force is deposited
on a level that a FINER level excises therefore has its momentum deleted every
coarse step — silently, because the ALM computes its forces from blade-element
theory and never checks whether the fluid received them. Measured symptom
(2-rotor probe, 300 steps): marker-sampled induced axial velocity exactly
0.000e+00 vs 7.99e-4 for the uniform twin, with thrust reading +30% high.

So placement is a correctness concern, not a convenience, and it gets its own
module with its own tests.

THE RULE (see `select_level`)
-----------------------------
Selection (hard — decides the level):
    the rotor DISK box (hub ± swept extents, NO epsilon) must lie inside the
    level's `excised` box.
Validation (loud — never decides):
    the SUPPORT box (disk ± n_cut·eps) is checked against the coupling band
    and the fine domain, and reported.

Epsilon is deliberately absent from the selection test. A static sweep over
every rotor config in the repo (190 of them) showed that
  - disk-in-excised reproduces the legacy hub-point selection for 190/190, and
  - adding the epsilon support DOWNGRADES 13 production configs by one level
    (the `farfield40` campaign overshoots its excised edge by ~3.7 fine cells).
Selection must not silently coarsen an existing config, so the support width
informs a warning instead.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "LevelBox", "RotorExtent", "PlacementReport",
    "level_boxes_l0", "block_boxes_l0", "rotor_extent", "select_level",
    "excision_conflicts", "band_report",
]

_AXES = ("x", "y", "z")
_FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


# =============================================================================
# Geometry containers — everything in L0 lattice units
# =============================================================================
@dataclass(frozen=True)
class LevelBox:
    """One MLG level's geometry expressed in L0 lattice units.

    excised_lo/hi   the region the fine grid OWNS; the coarse level's data here
                    is overwritten by F2C every coarse step.
    domain_lo/hi    excised plus the overlap band (the fine array's full extent).
    band            per-face band width [L0 lu]; 0 on flush faces, where the
                    fine face IS a domain face and coupling skips it.
    """
    level: int
    excised_lo: np.ndarray
    excised_hi: np.ndarray
    domain_lo: np.ndarray
    domain_hi: np.ndarray
    band: Dict[str, float]
    dx: float
    uid: int = 0            # block uid; 0 on the whole-level legacy path
    name: str = ""

    def contains_box(self, lo: np.ndarray, hi: np.ndarray,
                     *, excised: bool = True) -> bool:
        a = self.excised_lo if excised else self.domain_lo
        b = self.excised_hi if excised else self.domain_hi
        return bool(np.all(lo >= a) and np.all(hi <= b))

    def overlaps_box(self, lo: np.ndarray, hi: np.ndarray) -> bool:
        return bool(np.all(hi >= self.excised_lo) and np.all(lo <= self.excised_hi))

    def margins(self, lo: np.ndarray, hi: np.ndarray,
                *, excised: bool = True) -> Dict[str, float]:
        """Signed clearance per face [L0 lu]; negative = the box sticks out."""
        a = self.excised_lo if excised else self.domain_lo
        b = self.excised_hi if excised else self.domain_hi
        out = {}
        for d, ax in enumerate(_AXES):
            out[f"{ax}_min"] = float(lo[d] - a[d])
            out[f"{ax}_max"] = float(b[d] - hi[d])
        return out


@dataclass(frozen=True)
class RotorExtent:
    """A rotor's axis-aligned footprint in L0 lattice units."""
    name: str
    hub: np.ndarray
    disk_lo: np.ndarray
    disk_hi: np.ndarray
    support_lo: np.ndarray
    support_hi: np.ndarray
    # Per-marker pieces, so the support can be re-evaluated at the dx of
    # whichever level the rotor ends up on. epsilon has a 2*dx floor, so an
    # L0-derived support is a large over-estimate for a fine level (4x at L2)
    # and would raise false band warnings.
    reach: np.ndarray = None        # (N,3) per-marker per-axis swept reach
    eps_chord: np.ndarray = None    # (N,) factor*chord, WITHOUT the 2dx floor
    n_cut_aniso: float = 3.0

    def support_at(self, dx: float):
        """(lo, hi) of the Gaussian support if this rotor sits on a dx grid."""
        if self.reach is None or self.eps_chord is None:
            return self.support_lo, self.support_hi
        eps = np.maximum(self.eps_chord, 2.0 * float(dx))
        half = (self.reach + self.n_cut_aniso * eps[:, None]).max(axis=0)
        return self.hub - half, self.hub + half


@dataclass
class PlacementReport:
    """Outcome of resolving one rotor onto a level."""
    rotor: str
    level: int
    uid: int = 0
    reason: str = ""
    band_overshoot: Dict[str, float] = field(default_factory=dict)
    domain_overshoot: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Level geometry
# =============================================================================
def level_boxes_l0(overlap_mgr, level_origins: Sequence[Sequence[float]],
                   level_spacings: Sequence[Sequence[float]],
                   num_levels: int) -> List[LevelBox]:
    """Express every MLG level's excised/domain extent in L0 lattice units.

    Args:
        overlap_mgr:    OverlapManager; `get_region(k-1)` holds level k's geometry.
        level_origins:  per-level origin in L0 lu (setup._mlg_level_origins).
        level_spacings: per-level spacing in L0 lu (setup._mlg_level_spacings).
        num_levels:     total level count.

    Returns one LevelBox per level; level 0 spans the whole domain and has no
    band (there is nothing coarser to couple to).
    """
    boxes: List[LevelBox] = []
    for k in range(num_levels):
        if k == 0:
            # Level 0's extent is only needed as an "everything" sentinel.
            boxes.append(LevelBox(
                level=0,
                excised_lo=np.full(3, -np.inf), excised_hi=np.full(3, np.inf),
                domain_lo=np.full(3, -np.inf), domain_hi=np.full(3, np.inf),
                band={f: 0.0 for f in _FACES}, dx=1.0))
            continue

        region = overlap_mgr.get_region(k - 1)
        po = np.asarray(level_origins[k - 1], dtype=np.float64)
        pd = np.asarray(level_spacings[k - 1], dtype=np.float64)
        fr, fd = region.fine_region, region.fine_domain_coarse

        ex_lo = po + np.array([fr.x_start, fr.y_start, fr.z_start]) * pd
        ex_hi = po + np.array([fr.x_end, fr.y_end, fr.z_end]) * pd
        dm_lo = po + np.array([fd.x_start, fd.y_start, fd.z_start]) * pd
        dm_hi = po + np.array([fd.x_end, fd.y_end, fd.z_end]) * pd

        # Band width per face, in L0 lu. Flush faces have none by design
        # (OverlapRegion.flush_faces / coupling skips them).
        flush = getattr(region, "flush_faces", {}) or {}
        band = {}
        for d, ax in enumerate(_AXES):
            band[f"{ax}_min"] = 0.0 if flush.get(f"{ax}_min") else float(ex_lo[d] - dm_lo[d])
            band[f"{ax}_max"] = 0.0 if flush.get(f"{ax}_max") else float(dm_hi[d] - ex_hi[d])

        boxes.append(LevelBox(level=k, excised_lo=ex_lo, excised_hi=ex_hi,
                              domain_lo=dm_lo, domain_hi=dm_hi, band=band,
                              dx=float(level_spacings[k][0])))
    return boxes


def block_boxes_l0(blocks) -> List[LevelBox]:
    """Same thing, one box per BLOCK — the multi-block form of level_boxes_l0.

    A rotor is owned by the finest BLOCK that contains it, not merely by the
    finest level: with several blocks on a level, only one of them excises the
    rotor, and the others are irrelevant to it.
    """
    boxes: List[LevelBox] = []
    for b in blocks:
        if b.parent is None or b.region is None:
            boxes.append(LevelBox(
                level=b.level, uid=b.uid, name=b.name,
                excised_lo=np.full(3, -np.inf), excised_hi=np.full(3, np.inf),
                domain_lo=np.full(3, -np.inf), domain_hi=np.full(3, np.inf),
                band={f: 0.0 for f in _FACES}, dx=b.spacing))
            continue
        po = np.asarray(b.parent.origin, dtype=np.float64)
        pd = float(b.parent.spacing)
        fr, fd = b.region.fine_region, b.region.fine_domain_coarse
        ex_lo = po + np.array([fr.x_start, fr.y_start, fr.z_start]) * pd
        ex_hi = po + np.array([fr.x_end, fr.y_end, fr.z_end]) * pd
        dm_lo = po + np.array([fd.x_start, fd.y_start, fd.z_start]) * pd
        dm_hi = po + np.array([fd.x_end, fd.y_end, fd.z_end]) * pd
        flush = getattr(b.region, "flush_faces", {}) or {}
        band = {}
        for d, ax in enumerate(_AXES):
            band[f"{ax}_min"] = 0.0 if flush.get(f"{ax}_min") else float(ex_lo[d] - dm_lo[d])
            band[f"{ax}_max"] = 0.0 if flush.get(f"{ax}_max") else float(dm_hi[d] - ex_hi[d])
        boxes.append(LevelBox(level=b.level, uid=b.uid, name=b.name,
                              excised_lo=ex_lo, excised_hi=ex_hi,
                              domain_lo=dm_lo, domain_hi=dm_hi,
                              band=band, dx=float(b.spacing)))
    return boxes


# =============================================================================
# Rotor geometry
# =============================================================================
def rotor_extent(model, name: str = "rotor",
                 dx_model_to_l0: float = 1.0) -> RotorExtent:
    """Swept-disk AABB and Gaussian-support AABB of one ALM model, in L0 lu.

    Decomposing each marker offset into an axial part (along the shaft) and an
    in-plane radius gives the exact swept extent for ANY shaft orientation, so
    coning, prebend and tilted shafts are handled without special cases:

        disk_half[d] = max_j [ r_j·sqrt(1 - a_hat[d]^2) + |axial_j|·|a_hat[d]| ]

    The support box adds the kernel reach, per marker. `n_cut` is taken from the
    MODEL (`model.n_cut`), not from config `gaussian_cutoff`: compact kernel
    families override it with their own support factor.

    Args:
        dx_model_to_l0: scale from the model's own lattice units to L0 lu
                        (1.0 when the model already lives on L0).
    """
    rotor = model.rotor
    hub = np.asarray(rotor.hub_center, dtype=np.float64)
    pos = np.asarray(rotor.get_all_marker_positions(), dtype=np.float64)

    a_hat = np.asarray(rotor.rotation_axis, dtype=np.float64)
    n = np.linalg.norm(a_hat)
    a_hat = a_hat / n if n > 0 else np.array([0.0, 0.0, 1.0])

    rel = pos - hub[None, :]
    axial = rel @ a_hat                                   # (N,)
    radial_vec = rel - axial[:, None] * a_hat[None, :]
    radial = np.linalg.norm(radial_vec, axis=1)           # (N,)

    # per-axis reach of a swept circle of radius r at axial offset a
    inplane = np.sqrt(np.maximum(0.0, 1.0 - a_hat ** 2))  # (3,)
    reach = radial[:, None] * inplane[None, :] + np.abs(axial)[:, None] * np.abs(a_hat)[None, :]

    disk_half = reach.max(axis=0)

    eps = np.asarray(rotor.get_all_marker_epsilon(), dtype=np.float64)
    n_cut = float(getattr(model, "n_cut", 3.0))
    aniso = getattr(model, "_aniso", None)
    a_fac = 1.0
    if isinstance(aniso, dict) and aniso:
        vals = [float(v) for v in aniso.values()
                if isinstance(v, (int, float))]
        if vals:
            a_fac = max(1.0, max(vals))
    support_half = (reach + (n_cut * a_fac) * eps[:, None]).max(axis=0)

    # epsilon without the 2*dx floor, so support_at() can re-apply the floor
    # of whichever level the rotor is placed on.
    blade0 = rotor.blades[0]
    fac = float(getattr(blade0, 'epsilon_chord_factor', 0.25))
    chord = np.asarray(getattr(blade0, 'marker_chord', []), dtype=np.float64)
    eps_c = (np.tile(fac * chord, rotor.n_blades) if chord.size
             else np.zeros_like(eps))

    s = float(dx_model_to_l0)
    hub_l0 = hub * s
    return RotorExtent(
        name=name, hub=hub_l0,
        disk_lo=hub_l0 - disk_half * s, disk_hi=hub_l0 + disk_half * s,
        support_lo=hub_l0 - support_half * s, support_hi=hub_l0 + support_half * s,
        reach=reach * s, eps_chord=eps_c * s, n_cut_aniso=n_cut * a_fac)


# =============================================================================
# Selection and validation
# =============================================================================
def select_level(ext: RotorExtent, boxes: Sequence[LevelBox]) -> PlacementReport:
    """Finest level whose EXCISED box contains the rotor disk.

    Levels are strictly nested, so scanning finest → coarsest and taking the
    first hit yields the finest containing level. Level 0 always matches.
    """
    for box in sorted(boxes, key=lambda b: -b.level):
        if box.level == 0:
            continue
        if box.contains_box(ext.disk_lo, ext.disk_hi, excised=True):
            return PlacementReport(rotor=ext.name, level=box.level,
                                   uid=box.uid,
                                   reason="disk inside excised box")
    tight = min((b for b in boxes if b.level > 0),
                key=lambda b: min(b.margins(ext.disk_lo, ext.disk_hi).values()),
                default=None)
    why = "no fine level contains the rotor disk"
    if tight is not None:
        m = tight.margins(ext.disk_lo, ext.disk_hi)
        worst = min(m, key=m.get)
        why += (f" (closest: L{tight.level}, face {worst} short by "
                f"{-m[worst]:.3f} L0 lu)")
    return PlacementReport(rotor=ext.name, level=0, reason=why)


def excision_conflicts(ext: RotorExtent, level: int,
                       boxes: Sequence[LevelBox]) -> List[Tuple[int, Dict[str, float]]]:
    """Finer levels that excise part of this rotor's support — a hard error.

    A finer level overlapping the rotor means the ALM's force is deposited on a
    grid whose values are replaced by that finer grid every coarse step. Partial
    overlap clips the rotor; full overlap deletes it outright. Either way the
    momentum never reaches the fluid.
    """
    out = []
    for box in boxes:
        if box.level <= level:
            continue
        if box.overlaps_box(ext.support_lo, ext.support_hi):
            lo = np.maximum(ext.support_lo, box.excised_lo)
            hi = np.minimum(ext.support_hi, box.excised_hi)
            out.append((box.level,
                        {ax: float(max(0.0, hi[d] - lo[d]))
                         for d, ax in enumerate(_AXES)}))
    return out


def band_report(ext: RotorExtent, box: LevelBox) -> PlacementReport:
    """Support box vs the coupling band and the fine domain — warning material.

    Overshoot into the band means part of the Gaussian is deposited where C2F
    overwrites from the parent each sub-step; overshoot past the fine DOMAIN
    means that part is lost entirely. Reported in fine cells, since that is the
    unit the config author reasons in.
    """
    rep = PlacementReport(rotor=ext.name, level=box.level)
    if box.level == 0:
        return rep
    dx = box.dx if box.dx > 0 else 1.0
    # Evaluate the support with the epsilon this rotor will actually have on
    # THIS level, not the L0 one.
    _lo, _hi = ext.support_at(dx)
    m_ex = box.margins(_lo, _hi, excised=True)
    m_dm = box.margins(_lo, _hi, excised=False)
    for face in _FACES:
        if m_ex[face] < 0.0 and box.band.get(face, 0.0) > 0.0:
            rep.band_overshoot[face] = float(-m_ex[face] / dx)
        if m_dm[face] < 0.0:
            rep.domain_overshoot[face] = float(-m_dm[face] / dx)
    return rep
