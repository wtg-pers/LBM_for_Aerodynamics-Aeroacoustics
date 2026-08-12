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

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class InterpolatedBounceBack:
    """Linear Bouzidi IBB for stationary (no-slip) walls.

    Same-shape attributes as HalfwayBounceBack so downstream code
    (MomentumExchangeForce, simulation loop) does not need changes:

        solid_mask         : (shape,) bool
        needs_bounce       : (Q, shape) bool — link mask, or None (see below)
        n_boundary_links   : int
        c, opp, Q, dim     : lattice views

    Extra:
        q_fraction         : (Q, shape) float32 — q per link, 0.5 default,
                             or None in link mode
        link_cell/link_dir/link_q/n_links : the sparse boundary-link triple

    Two construction modes
    ----------------------
    `__init__` (dense) takes the (Q,)+shape `q_fraction` array; used by 2D,
    whose CUDA kernels index it directly, and by gates.

    `from_links` (sparse) takes the link triple straight from
    `compute_boundary_links` + a `*_links` q core and NEVER allocates
    anything of size (Q,)+shape — neither `needs_bounce` nor `q_fraction`.
    Both are then None, and `n_boundary_links == n_links`. This is what 3D
    production builds use: the dense pair costs 135 B/cell to carry ~10
    links per 1000 cells, and at v2's L1 (108.6 M cells) that alone exceeds
    a 24 GB card. Consumers that genuinely need the dense mask (2D kernels,
    the WFB link-geometry sibling) call `materialize_needs_bounce()`
    explicitly, so the cost is never paid by accident.
    """

    def __init__(
        self,
        xp: "ModuleType",
        lattice,
        solid_mask: "npt.NDArray",
        q_fraction: Optional["npt.NDArray"] = None,
        use_sparse: Optional[bool] = None,
        periodic_axes: Optional[Tuple[int, ...]] = None,
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
            periodic_axes: axes whose wrap is a contract — see
                q_fraction.compute_needs_bounce. None = legacy torus.
                Also governs the solid-upstream sanitiser: an upstream
                node that would leave the box on a non-periodic axis is
                treated like a solid upstream (q<1/2 demoted to 0.5)
                instead of being read through the wrap.
        """
        self.xp = xp
        self.lattice = lattice
        self.solid_mask = xp.asarray(solid_mask, dtype=bool)
        self.c = xp.asarray(lattice.c)
        self.opp = xp.asarray(lattice.opp)
        self.Q = lattice.Q
        self.dim = lattice.dim
        self._periodic_axes = periodic_axes

        if self.dim not in (2, 3):
            raise NotImplementedError(
                f"InterpolatedBounceBack supports 2D and 3D; dim={self.dim}."
            )

        self._use_sparse = (self.dim == 3) if use_sparse is None else bool(use_sparse)
        self._link_mode = False

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
    # Sparse (link) construction
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_links(
        cls,
        xp: "ModuleType",
        lattice,
        solid_mask: "npt.NDArray",
        link_cell: "npt.NDArray",
        link_dir: "npt.NDArray",
        link_q: Optional["npt.NDArray"] = None,
        periodic_axes: Optional[Tuple[int, ...]] = None,
    ) -> "InterpolatedBounceBack":
        """Build from the sparse link triple — no (Q,)+shape allocation.

        `link_cell`/`link_dir` must come from `compute_boundary_links` (or an
        equivalent enumeration in the same direction-major, ascending-flat-
        cell order); `link_q` is the matching per-link q from one of the
        `compute_q_fraction_*_links` cores, or None for q = 0.5 everywhere
        (≡ HWBB).

        Produces exactly the same `link_cell`/`link_dir`/`link_q` a dense
        `__init__` with the equivalent q_fraction would — the same
        non-physical-q clamp and the same solid-upstream sanitisation are
        applied, per link (gate: ibb_sparse identity).
        """
        self = cls.__new__(cls)
        self.xp = xp
        self.lattice = lattice
        self.solid_mask = xp.asarray(solid_mask, dtype=bool)
        self.c = xp.asarray(lattice.c)
        self.opp = xp.asarray(lattice.opp)
        self.Q = lattice.Q
        self.dim = lattice.dim
        self._periodic_axes = periodic_axes
        if self.dim != 3:
            raise NotImplementedError(
                "InterpolatedBounceBack.from_links is 3D-only "
                f"(dim={self.dim}); 2D uses the dense q_fraction kernels."
            )
        self._use_sparse = True
        self._link_mode = True
        # Dense forms deliberately absent — see the class docstring.
        self.needs_bounce = None
        self.q_fraction = None

        self.link_cell = xp.asarray(link_cell).astype(xp.int32)
        self.link_dir = xp.asarray(link_dir).astype(xp.int8)
        self.n_links = int(self.link_cell.size)
        self.n_boundary_links = self.n_links

        if link_q is None:
            q = xp.full(self.n_links, xp.float32(0.5), dtype=xp.float32)
        else:
            q = xp.asarray(link_q).astype(xp.float32)
            if int(q.size) != self.n_links:
                raise ValueError(
                    f"link_q size {int(q.size)} != n_links {self.n_links}"
                )
            # Any non-physical q -> sentinel 0.5 (HWBB fallback).
            bad = (q <= 0.0) | (q > 1.0)
            if bool(xp.any(bad)):
                q = xp.where(bad, xp.float32(0.5), q)
        self.link_q = q

        self._sanitize_links_for_solid_upstream()
        return self

    def _sanitize_links_for_solid_upstream(self) -> None:
        """Per-link twin of `_sanitize_q_for_solid_upstream` (see it).

        The upstream node x_f - c_i is taken with per-axis wrap, matching the
        `xp.roll` the dense sanitiser uses. With explicit `periodic_axes`, an
        upstream that would leave the box on a non-periodic axis is demoted
        exactly like a solid upstream — the wrapped cell is a stranger, not
        one step further into the fluid.
        """
        xp = self.xp
        if self.n_links == 0:
            self._n_links_fixed = 0
            return
        nx, ny, nz = (int(s) for s in self.solid_mask.shape)
        cell = self.link_cell.astype(xp.int64)
        gx = cell // (ny * nz)
        gy = (cell // nz) % ny
        gz = cell % nz
        d = self.link_dir.astype(xp.int64)
        c = self.c.astype(xp.int64)
        ux_raw = gx - c[0][d]
        uy_raw = gy - c[1][d]
        uz_raw = gz - c[2][d]
        ux = ux_raw % nx
        uy = uy_raw % ny
        uz = uz_raw % nz
        upstream_bad = self.solid_mask.reshape(-1)[
            (ux * ny + uy) * nz + uz]
        if self._periodic_axes is not None:
            for ax, (u_raw, n_ax) in enumerate(
                    ((ux_raw, nx), (uy_raw, ny), (uz_raw, nz))):
                if ax in self._periodic_axes:
                    continue
                upstream_bad = upstream_bad | (u_raw < 0) | (u_raw >= n_ax)
        bad = (self.link_q < 0.5) & upstream_bad
        n_fixed = int(xp.sum(bad))
        if n_fixed:
            self.link_q = xp.where(bad, xp.float32(0.5), self.link_q)
        self._n_links_fixed = n_fixed

    def materialize_needs_bounce(self) -> "npt.NDArray":
        """Build (and cache) the dense (Q,)+shape link mask from the links.

        Explicit because it costs Q bytes per cell (27 B/cell in 3D). Only
        the dense consumers — 2D CUDA kernels, `compute_link_wall_geometry`
        (WFB) — should ask for it.
        """
        if self.needs_bounce is not None:
            return self.needs_bounce
        xp = self.xp
        shape = tuple(int(s) for s in self.solid_mask.shape)
        N = 1
        for s in shape:
            N *= s
        nb = xp.zeros((self.Q, N), dtype=bool)
        if self.n_links:
            nb[self.link_dir.astype(xp.int64),
               self.link_cell.astype(xp.int64)] = True
        self.needs_bounce = nb.reshape((self.Q,) + shape)
        return self.needs_bounce

    # ──────────────────────────────────────────────────────────────
    # Setup helpers
    # ──────────────────────────────────────────────────────────────

    def _precompute_boundary_links(self) -> None:
        """Fill needs_bounce[i, ...] (fluid node, direction into solid).

        Delegates to q_fraction.compute_needs_bounce — the single link
        enumeration — so the seam (periodic_axes) policy cannot drift
        between the dense and sparse construction paths.
        """
        from src.boundary.q_fraction import compute_needs_bounce

        self.needs_bounce = compute_needs_bounce(
            self.xp, self.lattice, self.solid_mask,
            periodic_axes=self._periodic_axes,
        )
        self.n_boundary_links = int(self.xp.sum(self.needs_bounce))

    def _sanitize_q_for_solid_upstream(self) -> None:
        """Force q = 0.5 on links whose q<1/2 branch would read a solid node.

        The Bouzidi q<1/2 term needs the post-collision distribution at
        x_f − c_i (one step further into the fluid). In thin regions
        (airfoil TE, narrow gaps) that node may itself be solid. For those
        links we fall back to q = 0.5 → HWBB. With explicit periodic_axes,
        an upstream that would leave the box on a non-periodic axis is
        demoted the same way (the roll would read the opposite face).
        """
        xp = self.xp
        solid = self.solid_mask
        dim = self.dim
        n_fixed = 0

        for i in range(1, self.Q):
            if dim == 2:
                cx, cy = int(self.c[0, i]), int(self.c[1, i])
                upstream_bad = xp.roll(
                    xp.roll(solid, +cx, axis=0), +cy, axis=1
                )
                c_i = (cx, cy)
            else:  # 3D
                cx, cy, cz = (int(self.c[0, i]), int(self.c[1, i]),
                               int(self.c[2, i]))
                upstream_bad = xp.roll(
                    xp.roll(xp.roll(solid, +cx, axis=0), +cy, axis=1),
                    +cz, axis=2,
                )
                c_i = (cx, cy, cz)
            if self._periodic_axes is not None:
                # upstream x - c_i leaves the box at x[ax] = 0 (c > 0)
                # or x[ax] = N-1 (c < 0); |c| <= 1 for D2Q9/D3Q27.
                # roll always copies, so writing here is safe.
                for ax in range(dim):
                    if ax in self._periodic_axes:
                        continue
                    comp = int(c_i[ax])
                    if comp == 0:
                        continue
                    sl = [slice(None)] * dim
                    sl[ax] = 0 if comp > 0 else -1
                    upstream_bad[tuple(sl)] = True
            bad = (
                self.needs_bounce[i]
                & (self.q_fraction[i] < 0.5)
                & upstream_bad
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
                "dense q_fraction array, but this instance has none — it was "
                "built either with use_sparse=True or via from_links(). Use "
                "the GPU kernel via Simulation (it consumes the link triple), "
                "or construct densely with use_sparse=False."
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
