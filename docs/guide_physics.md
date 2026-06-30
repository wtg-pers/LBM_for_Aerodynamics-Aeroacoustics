# Physics Parameters Guide

Covers the `simulation.physics` config section: Reynolds number, relaxation time, viscosity, Mach number, and unit conversion.

## Config Template

```python
"physics": {
    "Re": 100,                              # [-] Reynolds number
    "tau": 0.59,                            # [-] Relaxation time
    "nu_lu": 0.03,                          # [Δx²/Δt] Kinematic viscosity
    "u_ref_lu": 0.05,                       # [Δx/Δt] Reference velocity
    "L_ref_lu": 30.0,                       # [Δx] Reference length
    "initial_flow_velocity": [0.05, 0, 0],  # [Δx/Δt] Initial velocity
    "U_ref": 10.0,                          # [m/s] Physical velocity (ALM only)
},
```

---

## Parameter Reference

| Key | Type | Unit | Required | Description |
|-----|------|------|----------|-------------|
| `Re` | float | [-] | Yes | Reynolds number |
| `tau` | float | [Δt] | Yes | Relaxation time (must be > 0.5) |
| `nu_lu` | float | [Δx²/Δt] | Yes | Kinematic viscosity in lattice units |
| `u_ref_lu` | float | [Δx/Δt] | Yes | Reference velocity in lattice units |
| `L_ref_lu` | float | [Δx] | Yes | Reference length in lattice units |
| `omega` | float | [-] | No | Relaxation rate (= 1/τ, auto-derived) |
| `U_ref` | float | [m/s] | ALM only | Physical reference velocity |
| `initial_flow_velocity` | float/list | [Δx/Δt] | No | Initial velocity field |

---

## Fundamental Relations

```
Speed of sound:   cs = 1/√3 ≈ 0.5774                [Δx/Δt]
Viscosity:        ν = cs² × (τ − 0.5) = (τ − 0.5)/3  [Δx²/Δt]
Reynolds:         Re = u_ref × L_ref / ν
Mach:             Ma = u_ref / cs = u_ref × √3
```

---

## Setup Procedure

### Step 1: Choose physical conditions

```python
RE = 100         # Target Reynolds number
U_INLET = 0.05   # Inlet velocity [Δx/Δt] (keep Ma < 0.1)
L_REF = 30        # Reference length [lu] (diameter, channel height)
```

### Step 2: Derive lattice parameters

```python
import numpy as np
NU_LU = U_INLET * L_REF / RE         # Viscosity
TAU = 0.5 + 3.0 * NU_LU             # Relaxation time
MA = U_INLET * np.sqrt(3.0)         # Mach number

assert TAU > 0.5, f"UNSTABLE: τ = {TAU}"
assert MA < 0.3,  f"COMPRESSIBLE: Ma = {MA}"
```

### Step 3: Write config

```python
"physics": {
    "Re": RE, "tau": TAU, "nu_lu": NU_LU,
    "u_ref_lu": U_INLET, "L_ref_lu": float(L_REF),
    "initial_flow_velocity": [U_INLET, 0.0, 0.0],
},
```

---

## Stability Requirements

| Condition | Requirement | Consequence of Violation |
|-----------|-------------|--------------------------|
| τ > 0.5 | **Mandatory** | Negative viscosity → divergence |
| τ < ~2.0 | Recommended | Large τ → accuracy degradation |
| Ma < 0.3 | **Mandatory** | Compressibility errors ∝ Ma² |
| Ma < 0.1 | Recommended | High-accuracy incompressible flow |
| u_ref < 0.1 | Recommended | Practical lattice velocity limit |

### τ Range Characteristics

| τ Range | ν Value | Characteristics |
|---------|---------|-----------------|
| 0.501–0.55 | Very small | High Re, may be unstable → use Cumulant |
| 0.55–0.7 | Moderate | Typical working range |
| 0.7–1.0 | Large | Low Re, very stable |
| > 1.0 | Very large | Very low Re, accuracy may degrade |

---

## Initial Velocity

```python
# Scalar: x-direction only
"initial_flow_velocity": 0.05,

# Vector: per-component
"initial_flow_velocity": [0.05, 0.0, 0.0],   # x-direction flow
"initial_flow_velocity": [0.0, 0.0, 0.0],    # Quiescent
```

---

## Physical Unit Conversion

Required only for Actuator Line Model (ALM). For standard flows, lattice units suffice.

```python
# In actuator_line.units:
"units": {
    "dx_phys": 0.001,       # [m/lu] Physical grid spacing
    "dt_phys": 2.89e-6,     # [s/lt] Physical time step
},
```

Conversion: `u_phys = u_lu × Δx/Δt`, `ν_phys = ν_lu × Δx²/Δt`

---

## Quick Reference: Example Parameters

| Re | u_ref | L_ref | ν_lu | τ | Ma |
|----|-------|-------|------|---|-----|
| 20 | 0.02 | 30 | 0.0300 | 0.590 | 0.035 |
| 100 | 0.05 | 30 | 0.0150 | 0.545 | 0.087 |
| 200 | 0.05 | 40 | 0.0100 | 0.530 | 0.087 |
| 1000 | 0.05 | 60 | 0.0030 | 0.509 | 0.087 |

For high Re (τ → 0.5), use the **Cumulant** collision model.
