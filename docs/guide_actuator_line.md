# Actuator Line Model (ALM) Guide

Covers the `actuator_line` and `airfoil_polar` config sections: wind turbine rotor simulation using body forces projected onto the LBM flow field.

## Overview

The Actuator Line Method represents rotor blades as lines of body forces distributed along the span. At each timestep:

1. Sample flow velocity at marker positions
2. Compute relative velocity, angle of attack (BEM)
3. Look up CL, CD from airfoil polar
4. Compute lift/drag forces per marker
5. Project forces onto the lattice via Gaussian kernel
6. Add body force to the LBM collision step

## Quick Start

```python
actuator_line = {
    "enabled": True,
    "coeff_mode": "auto",
    "units": {"dx_phys": 0.001, "dt_phys": 2.89e-6},
    "rotors": [
        {
            "name": "main_rotor",
            "rotor": {
                "n_blades": 3,
                "hub_center": [0.05, 0.08, 0.08],   # [m]
                "omega": 628.3,                       # [rad/s]
                "rotation_axis": [1, 0, 0],
                "blade": {
                    "sections": [
                        {"r": 0.0,   "chord": 0.006, "twist": 25.0, "airfoil": "naca0012", "active": False},
                        {"r": 0.007, "chord": 0.006, "twist": 25.0, "airfoil": "naca0012", "active": True},
                        {"r": 0.037, "chord": 0.003, "twist": 10.0, "airfoil": "naca0012", "active": True},
                    ],
                },
                "grid": {"n_radial": 20},
            },
        },
    ],
}

airfoil_polar = {
    "method": "neuralfoil",
    "airfoil_name": "naca0012",
    "Re_target": 1e5,
    "mode": "asb",
}
```

---

## Configuration Structure

### Top-Level Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | False | Enable/disable ALM |
| `coeff_mode` | str | `"auto"` | Force coefficient convention (see below) |
| `gaussian_cutoff` | float | 3.0 | Gaussian kernel cutoff radius (in ε units) |
| `rho_ref` | float | 1.0 | Reference density [kg/m³] |

### Units (Required)

Physical-lattice unit conversion factors. Required for all ALM simulations.

```python
"units": {
    "dx_phys": 0.001,        # [m/lu] Physical grid spacing
    "dt_phys": 2.89e-6,      # [s/lt] Physical time step
    "nu_phys": 1.5e-5,       # [m²/s] Physical kinematic viscosity (optional)
},
```

Derived conversions:
```
u_phys = u_lu × dx_phys / dt_phys    [m/s]
ω_lu = ω_phys × dt_phys              [rad/lt]
R_lu = R_phys / dx_phys              [lu]
```

### Coefficient Mode

Controls how thrust/torque coefficients are non-dimensionalized.

| Mode | Convention | Typical Use |
|------|-----------|-------------|
| `"auto"` | Auto-detect from config | Default |
| `"wind_turbine"` | C_T = T / (0.5 ρ U² πR²) | Horizontal axis wind turbines |
| `"rotorcraft"` | C_T = T / (ρ (ωR)² πR²) | Helicopters, drones, hovering rotors |

---

## Rotor Configuration

### Single Rotor (Legacy)

```python
actuator_line = {
    "enabled": True,
    "hub_center_phys": [0.05, 0.08, 0.08],
    "radius_phys": 0.037,
    "n_blades": 2,
    "omega_rpm": 8000,
    "rotation_axis": "x",
    ...
}
```

### Multi-Rotor (Recommended)

```python
actuator_line = {
    "enabled": True,
    "rotors": [
        {"name": "rotor_0", "rotor": { ... }},
        {"name": "rotor_1", "rotor": { ... }},
    ],
}
```

### Per-Rotor Parameters

```python
"rotor": {
    "n_blades": 3,                          # Number of blades
    "hub_center": [0.05, 0.08, 0.08],       # [m] Hub position (physical)
    "omega": 628.3,                          # [rad/s] Angular velocity
    "theta_0": 0.0,                          # [rad] Initial azimuth
    "rotation_axis": [1, 0, 0],              # Rotation axis vector
    "blade": { ... },                        # Blade geometry
    "grid": {"n_radial": 20},                # Markers per blade
}
```

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `n_blades` | int | [-] | Number of blades |
| `hub_center` | list(3) | [m] | Hub position in physical coords |
| `omega` | float | [rad/s] | Angular velocity (+ = CCW from upstream) |
| `theta_0` | float | [rad] | Initial azimuth angle of blade 0 |
| `rotation_axis` | list/str | [-] | Axis of rotation |
| `grid.n_radial` | int | [-] | Number of markers per blade |

### Rotation Axis

| Value | Direction | Use Case |
|-------|-----------|----------|
| `[1, 0, 0]` or `"x"` | X-axis | HAWT with wind in x |
| `[0, 0, 1]` or `"z"` | Z-axis | Vertical rotor |
| `"hawt_z"` | Z-axis (HAWT preset) | HAWT with wind in z |
| `[0.707, 0, 0.707]` | Custom tilt | Tilted rotor |

---

## Blade Geometry

### Section-Based Definition

```python
"blade": {
    "sections": [
        {"r": 0.000, "chord": 0.006, "twist": 25.0, "airfoil": "naca0012", "active": False},
        {"r": 0.007, "chord": 0.006, "twist": 25.0, "airfoil": "naca0012", "active": False},
        {"r": 0.00700001, "chord": 0.006, "twist": 25.0, "airfoil": "naca0012", "active": True},
        {"r": 0.037, "chord": 0.003, "twist": 10.0, "airfoil": "naca0012", "active": True},
    ],
}
```

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `r` | float | [m] | Radial position from hub center |
| `chord` | float | [m] | Local chord length |
| `twist` | float | [deg] | Local twist angle (positive = into wind) |
| `airfoil` | str | — | Airfoil identifier for polar lookup |
| `active` | bool | — | Whether this section generates forces |

Sections are interpolated linearly between stations to assign properties to each marker.

### Inactive Sections

Sections near the hub (circular cross-section, nacelle attachment) should be marked `active: False`. These markers exist for geometric continuity but produce no aerodynamic forces.

### Root Cutoff Pattern

```python
# Hub attachment: no force
{"r": 0.0,     ..., "active": False},
{"r": R*0.15,  ..., "active": False},
# Active blade starts here
{"r": R*0.15 + 1e-6, ..., "active": True},
{"r": R,       ..., "active": True},
```

---

## Airfoil Polar

Defines how CL and CD are obtained for given angle of attack and Reynolds number.

```python
airfoil_polar = {
    "method": "neuralfoil",
    "airfoil_name": "naca0012",
    "Re_target": 1e5,
    "mode": "asb",
}
```

### Available Methods

| Method | Description | Speed | Accuracy |
|--------|-------------|-------|----------|
| `"neuralfoil"` | Neural-network-based prediction | Fast | Good |
| `"prescribed"` | Fixed CL, CD values | Instant | User-defined |
| `"flat_plate"` | Thin airfoil theory (CL = 2πα) | Instant | Low Re only |

### NeuralFoil Parameters

| Key | Type | Description |
|-----|------|-------------|
| `airfoil_name` | str | NACA code (e.g., `"naca0012"`, `"naca2412"`) |
| `Re_target` | float | Target Reynolds number |
| `mode` | str | `"asb"` (AeroSandbox) or `"xfoil"` |
| `ncrit` | float | Transition parameter (default 9.0) |

### Prescribed Polar

```python
airfoil_polar = {
    "method": "prescribed",
    "CL_fixed": 1.0,
    "CD_fixed": 0.02,
}
```

Useful for debugging and validation (known forces).

---

## Force Projection

Forces are projected from blade markers onto the LBM grid via a regularized Gaussian kernel:

```
f_body(x) = Σ_markers F_marker × η_ε(|x − x_marker|)

η_ε(r) = (1 / (ε³ π^{3/2})) × exp(−r²/ε²)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epsilon_factor` | ε = factor × Δx | 2.0 |
| `gaussian_cutoff` | Kernel cutoff at N × ε | 3.0 |

Larger ε gives smoother but more diffuse forces. Smaller ε gives sharper forces but may cause noise.

---

## Output

### Marker VTP Files

When VTK is enabled, marker positions and BEM data are written as `.vtp` files:

```
vtk/markers/
├── markers_00001000.vtp
├── markers_00002000.vtp
└── markers.pvd            # Time-series
```

ParaView visualization:
- Apply "Glyph" filter with arrows to show force vectors
- Color by `alpha`, `CL`, `CD`, `blade_id`

### Rotor Performance CSV

```
csv/rotor_performance.csv:
step, time_lt, time_phys, revolutions, thrust_lu, torque_lu, power_lu, C_T, C_P, ...
```

### Terminal Summary

At simulation end:
```
Rotor Performance Summary:
  Revolutions: 12.5
  Mean C_T: 0.0123
  Mean C_P: 0.0045
  Figure of Merit: 0.72
```

---

## Domain Sizing for ALM

| Dimension | Guideline |
|-----------|-----------|
| Upstream (inlet to hub) | 3D–5D |
| Downstream (hub to outlet) | 8D–12D |
| Lateral (hub to wall) | 4D–6D |
| Vertical (hub to wall) | 4D–6D |
| Resolution | D/Δx ≥ 40 (rotor diameter / grid spacing) |

D = rotor diameter. Ensure the rotor tip is at least 2D from any wall.

---

## Disable

```python
actuator_line = {"enabled": False}
airfoil_polar = {}
```
