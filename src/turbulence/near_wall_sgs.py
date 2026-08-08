"""Near-wall eddy-viscosity reconstruction (CAMWA 2024 Eq.39) — W3c.

Wall-modelled LES needs the SGS field near the wall to be consistent
with the wall model, not with the bulk Smagorinsky estimate: the
staircase velocity jump at an immersed wall is read as resolved strain
and inflates nu_t, while the wall model separately assumes a transport
level of its own. Liang et al. (Comput. Math. Appl. 158 (2024) 21-35,
doi:10.1016/j.camwa.2024.01.008) replace nu_t in the near-wall band by
the van-Driest-damped mixing length, capped by the Smagorinsky length:

    nu_t = min[(kappa d_w)^2, (C_smag Delta)^2] . S . {1 - exp[-(y+/25)^3]}

Their section 5.3 is the reason this matters for us: with the wall model
applied to the VELOCITY only (no reconstruction) the instantaneous field
collapses onto the time-averaged one and the slat vortex oscillation
disappears — i.e. the wall model kills the unsteadiness that carries the
acoustics. Only with the reconstruction do the tonal peaks reappear.

Implementation note: our SGS pre-pass already writes the Smagorinsky
field nu_t_smag = (C_smag Delta)^2 |S|, so |S| need not be recomputed —
the whole formula collapses to a per-cell RESCALING of that field,

    nu_t <- nu_t_smag . min[(kappa d_w)^2 / (C_smag Delta)^2, 1]
                      . {1 - exp[-(y+/25)^3]}

which costs one indexed multiply over the band and needs no stencil.
The band index set and the wall distance are built once at setup.

The eddy viscosity is a turbulence closure (hence this module's home),
driven by u_tau supplied by the wall model in src/boundary/.

Author: LBM Development Team
Date: 2026-08 (wall_model track W3c)
"""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

#: von Karman constant of CAMWA Eq.(39).
KAPPA = 0.4184
#: van Driest constant of CAMWA Eq.(39) (cubic LES form, not the
#: squared RANS form of their Eq.(25) which uses A+ = 26).
A_PLUS = 25.0


class NearWallEddyViscosity:
    """CAMWA Eq.(39) reconstruction over a precomputed near-wall band.

    Args:
        xp:            array module (cupy on the production path).
        wall_distance: (*shape,) float wall distance d_w [lattice units].
            Cells with d_w <= 0 or non-finite are excluded from the band.
        band_cells:    only cells with d_w <= band_cells are rebuilt
            (outside, the plain SGS field is kept). Default 5.
        C_smag:        the SGS constant the nu_t field was built with —
            MUST match sgs_cfg["Cs"], else the |S| back-out is wrong.
        delta:         SGS filter width [lattice units] (1.0 = grid).
        kappa, a_plus: model constants (defaults = the reference's).
    """

    def __init__(
        self,
        xp: 'ModuleType',
        wall_distance: 'npt.NDArray',
        *,
        C_smag: float,
        band_cells: float = 5.0,
        delta: float = 1.0,
        kappa: float = KAPPA,
        a_plus: float = A_PLUS,
    ) -> None:
        self.xp = xp
        self.C_smag = float(C_smag)
        self.delta = float(delta)
        self.kappa = float(kappa)
        self.a_plus = float(a_plus)
        self.band_cells = float(band_cells)

        dw = xp.asarray(wall_distance, dtype=xp.float64).ravel()
        band = xp.isfinite(dw) & (dw > 0.0) & (dw <= self.band_cells)
        self.idx = xp.where(band)[0]
        self.d_w = dw[self.idx]
        # length-scale cap: min[(kappa d_w)^2, (C_smag Delta)^2] / (C_smag Delta)^2
        ls = self.kappa * self.d_w
        cap = self.C_smag * self.delta
        self._len_ratio = xp.minimum((ls / cap) ** 2, 1.0)
        self.n_band = int(self.idx.size)
        self._scale_buf = None

    # ------------------------------------------------------------------
    def apply(
        self,
        nu_t_flat: 'npt.NDArray',
        u_tau: Union[float, 'npt.NDArray'],
        nu: float,
    ) -> None:
        """Rescale nu_t in place over the band (see module doc).

        Args:
            nu_t_flat: (N,) SGS eddy viscosity == (C_smag delta)^2 |S|.
            u_tau:     scalar (homogeneous wall) or (N,) field, from the
                       wall model of the PREVIOUS step (one-step lag is
                       intentional and harmless — the wall model's own
                       input filter is far slower).
            nu:        molecular viscosity [lu^2/lt].
        """
        if self.n_band == 0:
            return
        xp = self.xp
        ut = u_tau[self.idx] if hasattr(u_tau, 'shape') and \
            getattr(u_tau, 'ndim', 0) > 0 else u_tau
        if not xp.any(xp.asarray(ut) > 0.0):
            return                      # wall model not yet active
        yplus = self.d_w * ut / nu
        damp = 1.0 - xp.exp(-(yplus / self.a_plus) ** 3)
        nu_t_flat[self.idx] = (nu_t_flat[self.idx]
                               * (self._len_ratio * damp)).astype(
                                   nu_t_flat.dtype)

    # ------------------------------------------------------------------
    def factor(
        self,
        u_tau: Union[float, 'npt.NDArray'],
        nu: float,
    ) -> 'npt.NDArray':
        """The band's multiplicative factor (host reference / gate)."""
        yplus = self.d_w * u_tau / nu
        return self._len_ratio * (1.0 - self.xp.exp(
            -(yplus / self.a_plus) ** 3))

    def scale_field(
        self,
        n_cells: int,
        u_tau: Union[float, 'npt.NDArray'],
        nu: float,
    ) -> Optional['npt.NDArray']:
        """(N,) float32 nu_t multiplier, 1.0 outside the band.

        For SGS models whose nu_t is computed INSIDE the fused collision
        kernel (smagorinsky): the kernel takes this as `nut_scale`.
        Returns None while the wall model has not produced a u_tau yet
        (caller then leaves the SGS field untouched). The band is fixed,
        so only band entries are rewritten after the first call.
        """
        xp = self.xp
        if self.n_band == 0 or not float(xp.max(xp.asarray(u_tau))) > 0.0:
            return None
        if self._scale_buf is None or self._scale_buf.size != n_cells:
            self._scale_buf = xp.ones(n_cells, dtype=xp.float32)
        self._scale_buf[self.idx] = self.factor(u_tau, nu).astype(
            xp.float32)
        return self._scale_buf

    def get_info(self) -> str:
        return (f"Near-wall SGS reconstruction (CAMWA Eq.39): "
                f"band {self.n_band:,} cells (d_w <= {self.band_cells}), "
                f"kappa={self.kappa}, A+={self.a_plus}, "
                f"C_smag={self.C_smag}, Delta={self.delta}")


def slab_wall_distance(
    xp: 'ModuleType',
    shape,
    axis: int,
    wall_lo: float,
    wall_hi: float,
) -> 'npt.NDArray':
    """Wall distance for a plane-channel testbed (two slab walls).

    d_w = min(y - wall_lo, wall_hi - y) at cell centres along `axis`;
    non-positive inside the solid slabs (excluded by the band filter).
    """
    dims = tuple(int(s) for s in shape)
    c = xp.arange(dims[axis], dtype=xp.float64)
    d = xp.minimum(c - wall_lo, wall_hi - c)
    view = [1] * len(dims)
    view[axis] = dims[axis]
    return xp.broadcast_to(d.reshape(view), dims).copy()
