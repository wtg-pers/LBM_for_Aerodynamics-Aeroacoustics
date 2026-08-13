"""
Field unit conversion for output channels — the single source of truth.

output.units selects what VALUES the field writers emit:

    "phys" (default)  physical fields — p_prime_pa [Pa], velocity_ms [m/s], ...
    "lu"              raw lattice fields — density, velocity [dx/dt], ...

Only field VALUES change. COORDINATES stay in the global L0-lu frame in
every mode, so .vti/.vth overlays, the geometry outline, and probe/config
coordinates keep matching ParaView regardless of the unit choice.

Conversions (rho0 = 1, isothermal EOS p = cs^2 rho), at level k with
block spacing s = 2^-k (in L0 units):

    p_prime_pa    = cs^2 (rho_lu - 1) * rho_phys * (dx/dt)^2   [level-invariant]
    velocity_ms   = u_lu * dx/dt                               [level-invariant]
    nu_t_m2s      = nu_t_lu * (dx^2/dt) * s                    [level-DEPENDENT]
    body_force_nm3= f_lu * (rho_phys dx/dt^2) / s              [level-DEPENDENT]

dx/dt is level-invariant (dx and dt halve together per level), so p' and
u take ONE constant across the whole AMR hierarchy. dx^2/dt and dx/dt^2
are NOT invariant — the writers pass each block's spacing so nu_t and
body_force come out right on fine levels. The constants are ALWAYS
printed in the setup summary — in 'lu' mode as the ParaView Calculator
recipe, in 'phys' mode as a record of what was applied.

The probe channel (probes.csv) is always physical [Pa] and is not
affected by this switch. Checkpoints always stay lattice (restart
correctness). ALM marker/wake VTP stay lattice (diagnostic frames).

Author: LBM Development Team
Date: 2026-08
"""

from typing import Any, List, Tuple

__all__ = ["FieldUnits", "assert_series_units"]


def assert_series_units(sample_path: str, mode: str) -> None:
    """Hard-error when an existing .vti series conflicts with `mode`.

    Restarting (or re-running without clear_previous) APPENDS to the
    on-disk series: the PVD/vtm index merges old and new files, so a
    units switch would put lattice `density` and physical `p_prime_pa`
    inside ONE time series with no visible marker — every downstream
    consumer (ParaView coloring, spectra scripts) silently reads a
    mixed-unit signal. The checkpoint itself is unit-agnostic (lattice
    f only), so the fix is never "can't restart" — it is: keep the
    original units, or point the outputs at a fresh directory.

    `sample_path` is one existing file of the series; unreadable or
    marker-free files skip the check (never block a run on a stale
    partial file).
    """
    try:
        with open(sample_path, 'rb') as f:
            head = f.read(4096).decode('ascii', errors='ignore')
    except OSError:
        return
    if 'Name="density"' in head:
        existing = 'lu'
    elif 'Name="p_prime_pa"' in head:
        existing = 'phys'
    else:
        return
    if existing == mode:
        return
    raise ValueError(
        f"output.units='{mode}' would append to an existing series "
        f"written with units='{existing}' ({sample_path}). Mixing units "
        "inside one time series is not allowed. Either keep units="
        f"'{existing}', or write to a fresh output directory "
        "(--results-dir / output.output_dir), or clear the old output "
        "(--clear).")


class FieldUnits:
    """Field-value unit converter shared by every VTK-family writer.

    mode 'lu'  -> convert() is the identity.
    mode 'phys'-> known fields are scaled AND renamed (the name carries
    the unit, so a .vti can never claim lattice values in Pa clothing).
    Unknown names pass through untouched in both modes.
    """

    #: name -> (phys name, scale kind)
    _PHYS = {
        'density':    ('p_prime_pa',     'pressure'),
        'velocity':   ('velocity_ms',    'velocity'),
        'nu_t':       ('nu_t_m2s',       'viscosity'),
        'body_force': ('body_force_nm3', 'force_density'),
    }

    def __init__(self, mode: str, uc: Any) -> None:
        """
        Args:
            mode: 'lu' or 'phys' (validated by the config parser).
            uc: UnitConverter (dx_phys, dt_phys, cs, rho_phys).
        """
        assert mode in ('lu', 'phys'), mode
        self.mode = mode
        vel = uc.dx_phys / uc.dt_phys
        self.vel_conv = vel                                  # [m/s per lu]
        self.p_conv = (uc.cs ** 2) * uc.rho_phys * vel * vel  # [Pa per drho]
        self.nu_conv = uc.dx_phys ** 2 / uc.dt_phys           # [m^2/s per lu]
        self.force_conv = uc.rho_phys * uc.dx_phys / uc.dt_phys ** 2  # [N/m^3]
        self.rho0 = 1.0
        self._uc = uc

    def convert(self, name: str, arr: Any,
                spacing: float = 1.0) -> Tuple[str, Any]:
        """(name, numpy array) -> (output name, output array).

        Identity in 'lu' mode and for names without a physical mapping
        (e.g. solid_mask). Never modifies `arr` in place.

        Args:
            spacing: The block's spacing in L0 units (2^-level). Only
                nu_t (x spacing) and body_force (/ spacing) depend on it;
                pressure/velocity are level-invariant and ignore it.
        """
        if self.mode == 'lu' or name not in self._PHYS:
            return name, arr
        out_name, kind = self._PHYS[name]
        if kind == 'pressure':
            return out_name, (arr - self.rho0) * self.p_conv
        if kind == 'velocity':
            return out_name, arr * self.vel_conv
        if kind == 'viscosity':
            return out_name, arr * (self.nu_conv * spacing)
        return out_name, arr * (self.force_conv / spacing)

    def summary_lines(self) -> List[str]:
        """Setup-summary block: applied units or the conversion recipe."""
        uc = self._uc
        lines = [
            f" Units  : output.units = '{self.mode}'  "
            f"(dx = {uc.dx_phys:.6e} m, dt(L0) = {uc.dt_phys:.6e} s, "
            f"dx/dt = {self.vel_conv:.4f} m/s — level-invariant)",
        ]
        if self.mode == 'phys':
            lines += [
                f"          fields: p_prime_pa = cs^2*(rho-1)*{uc.rho_phys}"
                f"*(dx/dt)^2  [x{self.p_conv:.6e}]",
                f"                  velocity_ms [x{self.vel_conv:.6e}], "
                f"nu_t_m2s [x{self.nu_conv:.6e} * 2^-level]",
                "          (p'/u: ONE constant for every AMR level; "
                "nu_t/body_force applied per level)",
                "          (coordinates stay L0-lu; checkpoints/markers "
                "stay lattice)",
            ]
        else:
            lines += [
                "          ParaView recipe (lattice -> physical; p'/u "
                "constants valid on EVERY AMR level):",
                f"            p' [Pa]  = (density - 1) * {self.p_conv:.6e}",
                f"            u [m/s]  = velocity * {self.vel_conv:.6e}",
                f"            nu_t [m2/s] = nu_t * {self.nu_conv:.6e}"
                " * 2^-level  (level-dependent!)",
            ]
        return lines
