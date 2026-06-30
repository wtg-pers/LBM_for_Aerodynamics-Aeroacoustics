# Convergence Detection Guide

Covers the `convergence` config section: automatic steady-state detection, divergence detection, and termination actions.

## Config

```python
convergence = {
    "enabled": True,
    "cauchy": {
        "window_size": "auto",     # Window: "auto" or integer
        "epsilon": 1e-5,           # Energy threshold (Path A)
        "Cd_epsilon": 1e-3,        # Drag threshold (Path B)
        "n_required": 3,           # Consecutive passes required
    },
    "on_converged": "checkpoint_and_stop",
    "on_diverged": "stop_with_checkpoint",
    "on_max_steps": "continue",
}
```

## Convergence Paths (Auto-Selected)

The solver automatically chooses the convergence criterion based on the simulation setup:

| Path | Condition | Criterion | Monitored Only |
|------|-----------|-----------|----------------|
| **A** | No obstacle or no force calc | Energy (ε) | — |
| **B** | Obstacle + force calc active | Cd (ε_Cd) | Energy, Cl |

### Path A: Energy-Based

```
E(t) = Σ|u|² / N_fluid

ε_cauchy = |mean(E, window_new) − mean(E, window_old)| / mean(E, window_old)

Converged when: ε_cauchy < epsilon for n_required consecutive checks
```

### Path B: Drag-Based

```
Cd(t) from force_calculation

ε_cauchy = |mean(Cd, window_new) − mean(Cd, window_old)| / mean(Cd, window_old)

Converged when: ε_cauchy < Cd_epsilon for n_required consecutive checks
```

## Parameters

### Cauchy Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window_size` | str/int | `"auto"` | Sample window for mean comparison |
| `epsilon` | float | 1e-5 | Energy convergence threshold |
| `Cd_epsilon` | float | 1e-3 | Drag coefficient threshold |
| `n_required` | int | 3 | Consecutive passes for final verdict |

### Window Size

| Setting | Behavior |
|---------|----------|
| `"auto"` | Computed from T_conv = L_ref/u_ref × coverage_factor (recommended) |
| Integer | Manual number of samples |

Auto window: `window = T_conv × 50`, `half_window = window / 2`

### Threshold Guidelines

| epsilon (energy) | Use Case |
|-----------------|----------|
| 1e-4 | Loose (fast exploration) |
| **1e-5** | **Standard** |
| 1e-6 | Strict (precision validation) |

| Cd_epsilon (drag) | Use Case |
|------------------|----------|
| 1e-2 | Loose |
| **1e-3** | **Standard** |
| 3e-4 | Strict (publication quality) |

### n_required

Prevents false convergence from transient oscillations.

| Value | Characteristic |
|-------|---------------|
| 1 | Fast detection, risk of false positive |
| **3** | **Recommended default** |
| 5 | Conservative, very reliable |

## Actions

### on_converged

| Value | Behavior |
|-------|----------|
| `"checkpoint_and_stop"` | Save checkpoint, then stop (default, recommended) |
| `"stop"` | Stop immediately (no checkpoint) |
| `"continue"` | Keep running past convergence |

### on_diverged

| Value | Behavior |
|-------|----------|
| `"stop_with_checkpoint"` | Save checkpoint for diagnosis, then stop (default) |
| `"stop"` | Stop immediately |

### on_max_steps

| Value | Behavior |
|-------|----------|
| `"continue"` | Keep running if not converged (default) |
| `"warn"` | Print warning, then continue |
| `"stop"` | Stop at max_steps |

## Divergence Detection

The solver automatically monitors for:
- ρ < 0 or ρ > 3 (unphysical density)
- |u| > 1 (superluminal lattice velocity)
- NaN or Inf values

When detected, the configured `on_diverged` action is triggered.

## Disable

```python
convergence = {"enabled": False}
```

The simulation runs until `max_steps` with no automatic termination.
