# Boundary Conditions Guide

Covers the `boundaries` config section: all available face boundary conditions including sponge layers.

## Config Template

```python
boundaries = {
    "inlet":  {"location": "xmin", "method": "regularized_inlet",
               "velocity": 0.05, "rho": 1.0},
    "outlet": {"location": "xmax", "method": "regularized_outlet",
               "rho": 1.0, "k": 0.1},
    "wall_s": {"location": "ymin", "method": "regularized_wall"},
    "wall_n": {"location": "ymax", "method": "regularized_wall"},
    "wall_z0": {"location": "zmin", "method": "regularized_wall"},
    "wall_z1": {"location": "zmax", "method": "regularized_wall"},
}
```

---

## Locations

| Location | Face | Coordinate |
|----------|------|------------|
| `xmin` | Inlet (upstream) | x = 0 |
| `xmax` | Outlet (downstream) | x = Nx-1 |
| `ymin` | Bottom wall | y = 0 |
| `ymax` | Top wall | y = Ny-1 |
| `zmin` | Front (3D) | z = 0 |
| `zmax` | Back (3D) | z = Nz-1 |

---

## Available Methods

### 1. Regularized Inlet — Velocity Dirichlet

Prescribes a fixed velocity at the inlet face.

```python
{
    "location": "xmin",
    "method": "regularized_inlet",
    "velocity": 0.05,    # [Δx/Δt] Velocity magnitude (face-normal direction)
    "rho": 1.0,          # [-] Reference density (default 1.0)
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `velocity` | float | — | Inlet velocity [Δx/Δt] |
| `rho` | float | 1.0 | Reference density [-] |

**Physics**: Reconstructs the distribution function using regularized non-equilibrium stress tensor. Density is computed from known populations; velocity is prescribed.

**Notes**:
- `velocity` is a scalar (normal component to the face)
- Ensure Ma = velocity × √3 < 0.3

---

### 2. Regularized Outlet — Pressure Dirichlet

Prescribes a target density (= pressure) at the outlet with relaxation.

```python
{
    "location": "xmax",
    "method": "regularized_outlet",
    "rho": 1.0,          # [-] Target density
    "k": 0.1,            # [-] Relaxation parameter (0 < k ≤ 1)
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rho` | float | 1.0 | Target outlet density [-] |
| `k` | float | 0.1 | Relaxation coefficient |

**Physics**: `ρ_outlet = ρ_interior + k × (ρ_target − ρ_interior)`. Velocity is extrapolated from the interior.

**Relaxation parameter `k`**:

| k | Behavior | Use Case |
|---|----------|----------|
| 0.01 | Very soft, minimal reflections | Unsteady flows, vortex shedding |
| **0.1** | **Balanced (default)** | **Most cases** |
| 0.5 | Fast enforcement | Steady flows, quick convergence |
| 1.0 | Hard (instantaneous) | May cause wave reflections |

---

### 3. Regularized Wall — No-Slip

Zero-velocity wall boundary condition.

```python
{
    "location": "ymin",
    "method": "regularized_wall",
}
```

No additional parameters. The distribution function is reconstructed with u = 0 using the regularized stress tensor approach.

---

### 4. Sponge Layer — Non-Reflecting Buffer Zone

Volume-based damping that drives the flow toward a freestream state in a buffer zone near the boundary. **Not a face BC** — operates on a configurable-thickness region.

```python
{
    "location": "xmax",
    "method": "sponge",
    "velocity": 0.05,       # [Δx/Δt] Freestream velocity U∞
    "rho": 1.0,             # [-] Freestream density ρ∞
    "thickness": 20,        # [lu] Buffer zone depth
    "strength": 0.5,        # [-] Maximum damping coefficient σ_max
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `velocity` | float/list | — | Freestream velocity U∞ [Δx/Δt] |
| `rho` | float | 1.0 | Freestream density ρ∞ [-] |
| `thickness` | int | 20 | Buffer zone depth L [lu] |
| `strength` | float | 0.5 | Maximum damping σ_max (0 < σ ≤ 1) |

**Physics**: In the buffer zone, the distribution function is blended toward equilibrium:

```
f(x) ← f(x) + σ(x) · [f_eq(ρ∞, U∞) − f(x)]

Damping profile (quadratic):
σ(x) = σ_max × (d / L)²

d = distance from inner edge of sponge zone
L = sponge thickness
```

At the inner edge (d=0), σ=0 (no damping). At the boundary face (d=L), σ=σ_max (maximum damping).

**Implementation detail**: The sponge face also gets a Neumann base BC to handle streaming wrap-around artifacts. The sponge damping is applied after all face/corner BCs as a separate phase.

**Parameter selection**:

| Parameter | Guideline |
|-----------|-----------|
| `thickness` | 10–30 lu (≥ 5% of domain length along that axis) |
| `strength` | 0.3–0.7 typical. Higher = stronger absorption but may cause internal reflections near the inner edge |

**When to use sponge**:

| Scenario | Recommended BC |
|----------|---------------|
| Channel flow (inlet/outlet) | `regularized_inlet` + `regularized_outlet` |
| External flow, far-field | `sponge` on outlet and/or lateral faces |
| Aeroacoustics (wave absorption) | `sponge` on all non-inlet faces |
| Vortex shedding (clean outflow) | `sponge` on outlet |

---

### 5. Equilibrium Inlet (Legacy)

```python
{"location": "xmin", "method": "equilibrium", "velocity": 0.05}
```

Sets f = f_eq(ρ=1, u=velocity). Ignores non-equilibrium part. May cause mass drift. **Use `regularized_inlet` instead.**

---

### 6. Neumann Outlet (Legacy)

```python
{"location": "xmax", "method": "neumann"}
```

Zero-gradient extrapolation: ∂f/∂n = 0. May cause density drift. **Use `regularized_outlet` or `sponge` instead.**

---

## Method Comparison

| Method | Accuracy | Stability | Mass Consv. | Reflections | Recommended |
|--------|----------|-----------|-------------|-------------|-------------|
| `regularized_inlet` | High | High | Good | Low | ✓ Inlet |
| `regularized_outlet` | High | High | Fair | Low (k-dependent) | ✓ Outlet |
| `regularized_wall` | High | High | Good | N/A | ✓ Walls |
| `sponge` | High | High | Fair | Very low | ✓ Far-field, acoustics |
| `equilibrium` | Medium | Medium | Poor | Medium | Legacy only |
| `neumann` | Medium | Medium | Poor | Medium | Legacy only |

---

## Edge & Corner Treatment

Where two faces meet (edge) or three faces meet (corner), the solver automatically applies **equilibrium reconstruction**: f = f_eq(ρ, u). No user configuration needed.

---

## Recommended BC Combinations

### Channel Flow (Poiseuille)

```python
"inlet":  → regularized_inlet (velocity)
"outlet": → regularized_outlet (rho=1.0, k=0.1)
"walls":  → regularized_wall (all remaining faces)
```

### External Flow (Cylinder/Sphere)

```python
"inlet":    → regularized_inlet (velocity)
"outlet":   → regularized_outlet or sponge
"far-field":→ regularized_inlet (same velocity as inlet) or sponge
```

### Aeroacoustic Simulation

```python
"inlet":  → regularized_inlet (velocity)
"outlet": → sponge (thickness=30, strength=0.5)
"lateral":→ sponge (thickness=20, strength=0.3)
```

---

## Mass Conservation Notes

The regularized inlet/outlet combination typically produces a small mass drift (+1–2% over long simulations). This is a characteristic of the BC approximation, not a solver error. To minimize drift:
- Reduce outlet `k` (softer relaxation)
- Increase domain size (less BC influence on interior)
- Use sponge layers instead of outlet (better mass behavior for external flows)
