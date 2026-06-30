# Solver Configuration Guide

Covers the `simulation` config section: device, lattice model, collision operator, and domain.

## Config Template

```python
simulation = {
    "device_mode": "gpu",        # "gpu" (CuPy) or "cpu" (NumPy)
    "device_id": 0,              # GPU index (multi-GPU systems)
    "dimension": 3,              # 2 or 3
    "lattice_model": "D3Q27",    # "D2Q9", "D3Q19", "D3Q27"
    "collision_model": "bgk",    # "bgk" or "cumulant"
    # "omega_bulk": 1.0,         # Cumulant only: bulk viscosity rate
    # "omega_high": 1.0,         # Cumulant only: high-order rate
    "domain": {
        "Nx": 200, "Ny": 100, "Nz": 50,
    },
    "physics": { ... },          # See guide_physics.md
    "time": { ... },             # See guide_output.md
}
```

---

## Device

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `device_mode` | str | — | `"gpu"` (CuPy) or `"cpu"` (NumPy) |
| `device_id` | int | 0 | GPU index for multi-GPU systems |

CLI override: `--gpu 1` or `--list-gpus` to show available devices.

---

## Lattice Model

| Model | Dim | Velocities | Use Case |
|-------|-----|------------|----------|
| `D2Q9` | 2D | 9 | 2D flows (channel, cylinder) |
| `D3Q19` | 3D | 19 | 3D flows (less memory, slightly less accurate) |
| `D3Q27` | 3D | 27 | 3D flows (recommended, best accuracy/stability) |

Dimension and lattice must be consistent:
- `dimension: 2` requires `D2Q9`
- `dimension: 3` requires `D3Q19` or `D3Q27`

---

## Collision Model

| Model | Key | Best For | Notes |
|-------|-----|----------|-------|
| BGK | `"bgk"` | Low Re, simple flows | Single relaxation time (SRT) |
| Cumulant | `"cumulant"` | High Re, turbulent | Galilean invariant, D3Q27 only |

### Cumulant Parameters (optional)

| Key | Default | Description |
|-----|---------|-------------|
| `omega_bulk` | 1.0 | Bulk viscosity relaxation rate |
| `omega_high` | 1.0 | Higher-order moment relaxation rate |

### Selection Guide

| Condition | Recommended |
|-----------|-------------|
| Re < 200, laminar | BGK |
| Re > 200, unsteady | Cumulant |
| 2D simulation | BGK (Cumulant is 3D only) |
| Stability priority | Cumulant |

---

## Domain

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `Nx` | int | [lu] | Streamwise grid points |
| `Ny` | int | [lu] | Wall-normal grid points |
| `Nz` | int | [lu] | Spanwise grid points (3D only) |

### Sizing Guidelines

| Flow Type | Nx | Ny | Nz |
|-----------|----|----|-----|
| Channel (Poiseuille) | 4H–10H | H (channel height) | 3–H |
| Cylinder/sphere | 15D–25D | 10D–20D | 3 (quasi-2D) or πD |
| Wind turbine (ALM) | 10D–15D | 8D–12D | 8D–12D |

H = channel height, D = object diameter.
