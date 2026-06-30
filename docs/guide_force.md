# Force Calculation Guide

Covers the `force_calculation` config section: aerodynamic force measurement on internal obstacles using the Momentum Exchange Method (MEM).

## Config

```python
force_calculation = {
    "enabled": True,
    "interval": 10,              # Compute every N steps
    "start_step": 100,           # Skip initial transient

    "reference": {
        "rho": 1.0,              # [-] Reference density
        "velocity": 0.05,        # [Δx/Δt] Reference velocity
        "char_length": 20,       # [Δx] Characteristic length (D)
        "span_length": 3,        # [Δx] Span for 2D coefficient
    },

    "log": {
        "enabled": True,
        "filename": "force_history",
    },
}
```

## Parameters

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `enabled` | bool | — | Enable force calculation |
| `interval` | int | [steps] | Computation frequency |
| `start_step` | int | [steps] | Skip initial transient |

### Reference Parameters (for non-dimensionalization)

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `rho` | float | [-] | Reference density (typically 1.0) |
| `velocity` | float | [Δx/Δt] | Reference velocity (inlet velocity) |
| `char_length` | float | [Δx] | Characteristic length (object diameter D) |
| `span_length` | float | [Δx] | Span: 1 for 2D, Nz for quasi-2D, actual span for 3D |

## Force Coefficients

```
Reference area:  A = char_length × span_length

Drag coefficient:  C_D = F_x / (0.5 × ρ × U² × A)
Lift coefficient:  C_L = F_y / (0.5 × ρ × U² × A)
```

## Strouhal Number

Computed automatically at simulation end from FFT of the Cl time series:

```
St = f × D / U
```

where f is the dominant frequency of Cl oscillation.

## Output

- CSV: `csv/force_history.csv` with per-step Cd, Cl values
- Terminal: summary with mean ± std, Strouhal number

## Reference Values (Cylinder)

| Re | Cd | St | Flow Type |
|----|----|----|-----------|
| 20 | 2.00 | — | Steady, symmetric |
| 40 | 1.50 | — | Steady, symmetric |
| 100 | 1.33 | 0.164 | Unsteady, Kármán vortex |
| 200 | 1.34 | 0.197 | Unsteady, 3D transition |

## Prerequisites

- `internal_geometry` must define an obstacle
- CLI must not have `--no-force` flag

## Disable

```python
force_calculation = {"enabled": False}
```
