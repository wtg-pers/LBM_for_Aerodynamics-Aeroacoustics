"""Interpolated Bounce-Back (Bouzidi-Firdaouss-Lallemand, linear form).

Implements the fixed-wall IBB of Bouzidi et al. (2001, Phys. Fluids 13,
3452), Eqs. 5a-5b. The public interface mirrors `HalfwayBounceBack` so the
two are interchangeable in `setup.py::_setup_boundaries`.

Per boundary link (fluid node x_f, direction c_i pointing into solid) we
store q ∈ (0, 1] = |x_f → wall| / |x_f → x_solid|. With q = 0.5 the formula
reduces exactly to half-way bounce-back — the sentinel value used for links
without a geometry-specific q.

Update rule (stationary wall), applied to post-streaming f at the fluid
node after BCs/streaming, using the post-collision f_post at time t:

    q < 1/2:  f_ī(x_f) = 2q·f̃_i(x_f) + (1 − 2q)·f̃_i(x_f − c_i)
    q ≥ 1/2:  f_ī(x_f) = (1/(2q))·f̃_i(x_f) + ((2q−1)/(2q))·f̃_ī(x_f)

where ī = opp[i] and f̃ ≡ f_post.

Current scope: 2D only (the 3D HWBB CUDA kernel path is preserved
unchanged for existing 3D configs).

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class InterpolatedBounceBack:
    """Linear Bouzidi IBB for stationary (no-slip) walls.

    Same-shape attributes as HalfwayBounceBack so downstream code
    (MomentumExchangeForce, simulation loop) does not need changes:

        solid_mask         : (shape,) bool
        needs_bounce       : (Q, shape) bool — link mask
        n_boundary_links   : int
        c, opp, Q, dim     : lattice views

    Extra:
        q_fraction         : (Q, shape) float32 — q per link, 0.5 default
    """

    def __init__(
        self,
        xp: "ModuleType",
        lattice,
        solid_mask: "npt.NDArray",
        q_fraction: Optional["npt.NDArray"] = None,
        use_sparse: Optional[bool] = None,
    ) -> None:
        """
        Args:
            xp:         Array module (numpy or cupy).
            lattice:    Lattice (D2Q9 supported).
            solid_mask: Boolean mask, True = solid.
            q_fraction: Optional (Q, *shape) float array. Any value not in
                        (0, 1] is treated as "unknown" and forced to 0.5.
                        If None, q = 0.5 everywhere (→ HWBB).
            use_sparse: If True, after building the dense q_fraction the
                        boundary-link triple (link_cell, link_dir, link_q)
                        is materialised and the dense (Q, *shape) array is
                        released. Default: True for 3D (memory critical),
                        False for 2D (existing dense kernel path is fine).
        """
        self.xp = xp
        self.lattice = lattice
        self.solid_mask = xp.asarray(solid_mask, dtype=bool)
        self.c = xp.asarray(lattice.c)
        self.opp = xp.asarray(lattice.opp)
        self.Q = lattice.Q
        self.dim = lattice.dim

        if self.dim not in (2, 3):
            raise NotImplementedError(
                f"InterpolatedBounceBack supports 2D and 3D; dim={self.dim}."
            )

        self._use_sparse = (self.dim == 3) if use_sparse is None else bool(use_sparse)

        self._precompute_boundary_links()

        target_shape = (self.Q,) + tuple(self.solid_mask.shape)
        if q_fraction is None:
            self.q_fraction = xp.full(target_shape, 0.5, dtype=xp.float32)
        else:
            q = xp.asarray(q_fraction, dtype=xp.float32)
            if tuple(q.shape) != target_shape:
                raise ValueError(
                    f"q_fraction shape {tuple(q.shape)} != expected {target_shape}"
                )
            # Any non-physical q → sentinel 0.5 (HWBB fallback).
            bad = (q <= 0.0) | (q > 1.0)
            if bool(xp.any(bad)):
                q = xp.where(bad, xp.float32(0.5), q)
            self.q_fraction = q

        self._sanitize_q_for_solid_upstream()

        # Sparse boundary-link representation (link_cell, link_dir, link_q).
        # Built unconditionally so the kernel always has access; in sparse
        # mode the dense q_fraction is then released to free GPU memory.
        self._build_sparse_links()
        if self._use_sparse:
            # Release dense q_fraction (Q * prod(shape) * 4 bytes). For 3D
            # Re=3900 v2 (51M cells) this frees ~5.5 GB.
            self.q_fraction = None

    # ──────────────────────────────────────────────────────────────
    # Setup helpers
    # ──────────────────────────────────────────────────────────────

    def _precompute_boundary_links(self) -> None:
        """Fill needs_bounce[i, ...] (fluid node, direction into solid)."""
        xp = self.xp
        c = self.c
        solid = self.solid_mask
        shape = solid.shape
        dim = self.dim

        self.needs_bounce = xp.zeros((self.Q,) + shape, dtype=bool)
        for i in range(self.Q):
            if i == 0:
                continue
            if dim == 2:
                cx, cy = int(c[0, i]), int(c[1, i])
                shifted = xp.roll(xp.roll(solid, -cx, axis=0), -cy, axis=1)
            else:  # 3D
                cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
                shifted = xp.roll(
                    xp.roll(xp.roll(solid, -cx, axis=0), -cy, axis=1),
                    -cz, axis=2,
                )
            self.needs_bounce[i] = (~solid) & shifted

        self.n_boundary_links = int(xp.sum(self.needs_bounce))

    def _sanitize_q_for_solid_upstream(self) -> None:
        """Force q = 0.5 on links whose q<1/2 branch would read a solid node.

        The Bouzidi q<1/2 term needs the post-collision distribution at
        x_f − c_i (one step further into the fluid). In thin regions
        (airfoil TE, narrow gaps) that node may itself be solid. For those
        links we fall back to q = 0.5 → HWBB.
        """
        xp = self.xp
        solid = self.solid_mask
        n_fixed = 0

        for i in range(1, self.Q):
            if self.dim == 2:
                cx, cy = int(self.c[0, i]), int(self.c[1, i])
                upstream_is_solid = xp.roll(
                    xp.roll(solid, +cx, axis=0), +cy, axis=1
                )
            else:  # 3D
                cx, cy, cz = (int(self.c[0, i]), int(self.c[1, i]),
                               int(self.c[2, i]))
                upstream_is_solid = xp.roll(
                    xp.roll(xp.roll(solid, +cx, axis=0), +cy, axis=1),
                    +cz, axis=2,
                )
            bad = (
                self.needs_bounce[i]
                & (self.q_fraction[i] < 0.5)
                & upstream_is_solid
            )
            if bool(xp.any(bad)):
                n_fixed += int(xp.sum(bad))
                self.q_fraction[i] = xp.where(
                    bad, xp.float32(0.5), self.q_fraction[i],
                )

        self._n_links_fixed = n_fixed

    def _build_sparse_links(self) -> None:
        """Materialise sparse boundary-link arrays from dense (needs_bounce, q_fraction).

        Output (stored on self):
            link_cell (n_links,) int32 -- flat cell index in C-contiguous layout
                                          of self.solid_mask
            link_dir  (n_links,) int8  -- direction q in 1..Q-1
            link_q    (n_links,) float32 -- q value in (0, 1]
            n_links   int

        Each entry corresponds to a fluid node x_f whose direction-q
        neighbour is solid; the IBB kernel writes f[opp(q), x_f] using
        link_q[link_idx] as the Bouzidi q for that link.
        """
        xp = self.xp
        nb = self.needs_bounce          # (Q, *shape) bool
        q  = self.q_fraction            # (Q, *shape) float32
        Q  = self.Q

        # Flatten spatial dims to a single axis.
        N_cells = 1
        for s in self.solid_mask.shape:
            N_cells *= int(s)
        nb_flat = nb.reshape(Q, N_cells)
        q_flat  = q.reshape(Q, N_cells)

        # Collect (q_dir, cell_idx) of all True entries.
        link_dir, link_cell = xp.where(nb_flat)
        link_q = q_flat[link_dir, link_cell]

        self.link_cell = link_cell.astype(xp.int32)
        self.link_dir  = link_dir.astype(xp.int8)
        self.link_q    = link_q.astype(xp.float32)
        self.n_links   = int(self.link_cell.size)

    # ──────────────────────────────────────────────────────────────
    # Step application
    # ──────────────────────────────────────────────────────────────

    def apply(self, f: "npt.NDArray", f_post: "npt.NDArray") -> None:
        """Apply Bouzidi linear IBB (stationary wall).

        Must run AFTER streaming. Writes f[ī] at fluid nodes adjacent to
        solid using the post-collision f_post of the same step.

        Args:
            f:      Post-streaming distribution (Q, *shape). Modified in place.
            f_post: Post-collision distribution at time t (pre-streaming).
                    Required — the q<1/2 branch also reads f_post[i, x_f − c_i].
        """
        if f_post is None:
            raise ValueError(
                "InterpolatedBounceBack.apply() requires f_post "
                "(post-collision distribution)."
            )
        if self.q_fraction is None:
            raise RuntimeError(
                "InterpolatedBounceBack.apply() Python path requires the "
                "dense q_fraction array, but the instance was built with "
                "use_sparse=True. Use the GPU kernel via Simulation, or "
                "construct with use_sparse=False to keep the dense array."
            )

        xp = self.xp
        opp = self.opp
        c = self.c

        for i in range(1, self.Q):
            mask = self.needs_bounce[i]
            if not bool(xp.any(mask)):
                continue

            i_opp = int(opp[i])
            q_i = self.q_fraction[i]

            # q ≥ 1/2 branch — local (x_f only)
            mask_ge = mask & (q_i >= 0.5)
            if bool(xp.any(mask_ge)):
                q_ge = q_i[mask_ge]
                two_q = 2.0 * q_ge
                f[i_opp][mask_ge] = (
                    (1.0 / two_q) * f_post[i][mask_ge]
                    + ((two_q - 1.0) / two_q) * f_post[i_opp][mask_ge]
                )

            # q < 1/2 branch — needs upstream f_post[i] at x_f - c_i
            mask_lt = mask & (q_i < 0.5)
            if bool(xp.any(mask_lt)):
                if self.dim == 2:
                    cx, cy = int(c[0, i]), int(c[1, i])
                    # shift(+c_i) so that shifted[x_f] = f_post[i, x_f − c_i]
                    f_upstream = xp.roll(
                        xp.roll(f_post[i], +cx, axis=0), +cy, axis=1,
                    )
                else:  # 3D
                    cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
                    f_upstream = xp.roll(
                        xp.roll(xp.roll(f_post[i], +cx, axis=0), +cy, axis=1),
                        +cz, axis=2,
                    )
                q_lt = q_i[mask_lt]
                f[i_opp][mask_lt] = (
                    2.0 * q_lt * f_post[i][mask_lt]
                    + (1.0 - 2.0 * q_lt) * f_upstream[mask_lt]
                )

    def reset_solid_nodes(
        self,
        f: "npt.NDArray",
        rho: Optional["npt.NDArray"] = None,
        u: Optional["npt.NDArray"] = None,
    ) -> None:
        """Reset distributions inside solid nodes to rest-state equilibrium."""
        xp = self.xp
        w = xp.asarray(self.lattice.w)
        if rho is None:
            rho_solid = 1.0
        else:
            rho_solid = rho[self.solid_mask] if hasattr(rho, "__getitem__") else rho
        for i in range(self.Q):
            f[i][self.solid_mask] = w[i] * rho_solid

    def apply_with_reset(
        self,
        f: "npt.NDArray",
        f_post: Optional["npt.NDArray"] = None,
    ) -> None:
        """apply() + reset_solid_nodes(). Signature matches HWBB."""
        self.apply(f, f_post)
        self.reset_solid_nodes(f)

    # ──────────────────────────────────────────────────────────────
    # HWBB-compatible helpers
    # ──────────────────────────────────────────────────────────────

    def mask_velocity(self, u: "npt.NDArray") -> None:
        for d in range(self.dim):
            u[d][self.solid_mask] = 0.0

    def mask_density(self, rho: "npt.NDArray", rho_solid: float = 1.0) -> None:
        rho[self.solid_mask] = rho_solid

    def get_solid_mask(self) -> "npt.NDArray":
        return self.solid_mask

    def get_info(self) -> str:
        xp = self.xp
        n_solid = int(xp.sum(self.solid_mask))
        n_total = int(self.solid_mask.size)
        lines = [
            "Interpolated Bounce-Back (Bouzidi linear):",
            f"  Solid nodes: {n_solid:,} ({100 * n_solid / n_total:.2f}%)",
            f"  Boundary links: {self.n_boundary_links:,}",
        ]
        if self.n_boundary_links > 0:
            if self.q_fraction is not None:
                qb = self.q_fraction[self.needs_bounce]
            else:
                qb = self.link_q
            q_mean = float(xp.mean(qb))
            q_min = float(xp.min(qb))
            q_max = float(xp.max(qb))
            n_hwbb = int(xp.sum(qb == xp.float32(0.5)))
            lines.append(
                f"  q-fraction: mean={q_mean:.3f}, min={q_min:.3f}, "
                f"max={q_max:.3f}"
            )
            lines.append(
                f"  Links at HWBB sentinel (q=0.5): "
                f"{n_hwbb:,} / {self.n_boundary_links:,}"
            )
        if getattr(self, "_n_links_fixed", 0) > 0:
            lines.append(
                f"  Links reset due to solid upstream: {self._n_links_fixed}"
            )
        return "\n".join(lines)
