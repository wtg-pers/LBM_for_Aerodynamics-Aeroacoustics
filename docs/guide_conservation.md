# Conservation Monitoring Guide

Covers the `conservation` config section: mass conservation tracking during simulation.

## Overview

The conservation monitor tracks the total mass (Σρ) in the fluid domain over time, comparing it against the initial value to detect drift. This is essential for validating boundary condition quality and solver stability.

## Config

```python
conservation = {
    "enabled": True,
    "check_interval": 100,    # Check every N steps
    "verbose": 0,             # Terminal output: 0=silent, 1=simple, 2=detailed
    "log_to_csv": True,       # Write history to CSV
}
```

## Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | True | Enable mass conservation monitoring |
| `check_interval` | int | output_interval | How often to compute mass (in steps) |
| `verbose` | int | 0 | Terminal verbosity: 0=silent, 1=summary, 2=per-check |
| `log_to_csv` | bool | True | Write `mass_conservation.csv` |

## How It Works

At initialization:
```
M₀ = Σ ρ(x, t=0)    over all fluid nodes (excluding solid)
```

At each check_interval:
```
M(t) = Σ ρ(x, t)
drift = (M(t) − M₀) / M₀ × 100%
```

## Output

### Progress Bar

Mass drift is displayed in real-time in the tqdm progress bar:
```
 45%|████▌     |4500/10000 42.1step/s [01:47, ρ=1.0144, drift=+1.44%]
```

### CSV File

`csv/mass_conservation.csv` contains per-check records:
```csv
step,mass,drift_percent
0,36000.000000,0.000000
100,36001.234567,0.003429
200,36002.456789,0.006824
...
```

### Final Summary

At simulation end, a conservation analysis is printed:
```
[8] Final Conservation Analysis
  Initial mass: 36000.000000
  Final mass:   36518.157920
  Drift: +1.4393%
```

## Interpreting Mass Drift

| Drift | Assessment |
|-------|------------|
| < 0.1% | Excellent |
| 0.1–1% | Normal (regularized inlet/outlet characteristic) |
| 1–3% | Acceptable for long runs with inlet/outlet BCs |
| > 5% | Investigate BC settings or stability |

### Common Causes of Drift

| Cause | Solution |
|-------|----------|
| Inlet/outlet BC mismatch | Ensure inlet velocity and outlet ρ are physically consistent |
| High outlet k value | Reduce k (e.g., 0.1 → 0.01) |
| τ close to 0.5 | Increase τ or use Cumulant collision |
| Small domain | Increase domain size (reduce BC influence) |

### Zero Drift Cases

The following BC combinations produce zero or near-zero drift:
- All-wall (closed box): drift = 0 (exactly conserved)
- Periodic boundaries: drift = 0
- Sponge + wall: typically very low drift

## Solid Mask

If an internal obstacle exists, solid nodes are **excluded** from the mass sum. This ensures the drift measurement reflects only fluid dynamics, not the obstacle presence.

## Disable

```python
conservation = {"enabled": False}
```

Mass monitoring is skipped entirely. The progress bar will not show drift.
