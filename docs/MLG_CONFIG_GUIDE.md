# Multi-Level Grid (MLG) Configuration Guide

## Overview

MLG enables local grid refinement within the LBM solver. Fine grids overlap
coarse grids at 2× resolution per level, connected by two-way coupling
(Coarse→Fine interpolation + Fine→Coarse restriction).

```
Level 0 (dx=1.0):  |========================|  Full domain
Level 1 (dx=0.5):      |==============|        2× resolution
Level 2 (dx=0.25):        |========|            4× resolution
Level 3 (dx=0.125):          |====|             8× resolution
```

## Quick Start

Add an `mlg` section to your config file:

```python
mlg = {
    "enabled": True,
    "num_levels": 2,
    "overlap_width": 2,
    "interpolation": "cubic",
    "filter_level": 1,
    "levels": [
        {},  # Level 0 = full domain (always empty)
        {
            "region": {
                "x_min": 40, "x_max": 80,
                "y_min": 5,  "y_max": 25,
                "z_min": 2,  "z_max": 8,
            },
        },
    ],
}
```

No changes to `main.py` are needed. The solver automatically detects MLG
and switches to nested time-stepping.

## Configuration Parameters

### Top-level

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `False` | Enable/disable MLG |
| `num_levels` | int | 1 | Total number of grid levels (including Level 0) |
| `overlap_width` | int | 2 | Overlap buffer width in coarse cells |
| `interpolation` | str | `"cubic"` | Spatial interpolation scheme |
| `filter_level` | int | 1 | F→C low-pass filter depth |

### Interpolation Schemes

| Scheme | Accuracy | Stencil | Mass Conservation |
|--------|----------|---------|-------------------|
| `"cubic"` | 4th order | 4-point symmetric | ✓ Good |
| `"compact_second_order"` | 2nd order | 2-point mean | ✗ Causes mass loss |

**Recommendation**: Always use `"cubic"` unless debugging.

### Filter Levels

| Level | Neighbors | Use Case |
|-------|-----------|----------|
| 0 | None (no filter) | Testing only |
| 1 | 6 face neighbors | Default, recommended |
| 2 | 18 face+edge neighbors | Stronger smoothing |

### Overlap Width

| Width | Fine cells | Interpolation support |
|-------|------------|----------------------|
| 1 | 2 | Insufficient for cubic (needs 4-point stencil) |
| **2** | **4** | **Minimum for cubic interpolation** |
| 3 | 6 | Extra buffer, slightly better accuracy |

**Recommendation**: Use `overlap_width: 2` (minimum for cubic).

## Region Specification

### All coordinates are in Level 0 (physical) units

```python
"levels": [
    {},  # Level 0 = full domain
    {"region": {"x_min": 40, "x_max": 80, ...}},  # L0 coords
    {"region": {"x_min": 45, "x_max": 75, ...}},  # L0 coords (NOT parent local!)
    {"region": {"x_min": 50, "x_max": 70, ...}},  # L0 coords
]
```

The solver automatically converts L0 coordinates to parent-local coordinates
internally. You never need to compute local indices yourself.

### Nesting Rule

Each level's region must be **strictly inside** its parent's region:

```
✓ Valid:   L1 x[40,80],  L2 x[45,75],  L3 x[50,70]
✗ Invalid: L1 x[40,80],  L2 x[35,75]   ← L2 extends beyond L1
```

### Distance from Domain Boundaries

Fine regions should maintain a buffer from domain boundaries:

```python
# For overlap_width=2, keep at least 2 cells from each wall:
"y_min": 2,          # NOT 0 (would clip overlap)
"y_max": Ny - 3,     # NOT Ny-1
```

If the fine region + overlap extends beyond the domain, the overlap is
automatically clipped with a warning. This reduces interpolation accuracy
at that face.

## Physics Scaling

The solver automatically computes per-level physics parameters:

| Parameter | Level k formula | Example (k=2, τ₀=0.59) |
|-----------|----------------|------------------------|
| dx | dx₀ / 2^k | 0.25 |
| dt | dt₀ / 2^k | 0.25 |
| τ | 2τ_{k-1} − 0.5 | 0.86 |
| ν | Same as Level 0 | 0.03 (preserved) |
| u | Same as Level 0 | 0.02 (convective scaling) |

Viscosity is continuous across all levels by construction.

## Computational Cost

Each level k performs 2^k sub-steps per coarse step:

```
Level 0:  N₀ nodes × 1 step  =  N₀ updates
Level 1:  N₁ nodes × 2 steps =  2·N₁ updates
Level 2:  N₂ nodes × 4 steps =  4·N₂ updates
Level 3:  N₃ nodes × 8 steps =  8·N₃ updates
```

Total sub-steps per coarse step = 2^M − 1 (M = num_levels).

**Rule of thumb**: The finest level dominates the computational cost.
Keep the finest level's region as small as possible.

## Memory Estimate

Each level stores `f` with shape `(Q, Nx, Ny, Nz)` in float64:

```
Memory per level = Q × Nx × Ny × Nz × 8 bytes
D3Q27: Q=27, so 216 bytes per node
```

## Output

### VTK Files

```
vtk/
├── vth/
│   └── lbm_00005000.vth              ← Open in ParaView
├── level0/
│   └── lbm_00005000_level0.vti
└── level1/
    └── lbm_00005000_level1.vti
```

- Open `.vth` files in ParaView for AMR visualization
- Select multiple `.vth` files for time-series animation
- Individual level `.vti` files can be opened separately

### Checkpoint

Checkpoints automatically save all levels' distribution functions.
Restart restores all levels. If the checkpoint has fewer levels than
the current config, missing levels are initialized with equilibrium.

## CLI Flags

```bash
# Default: compact summary to terminal, detailed log to file
python main.py --config configs/my_mlg_config.py

# Verbose: echo detailed setup log to terminal
python main.py --config configs/my_mlg_config.py --verbose

# Restart from checkpoint
python main.py --config configs/my_mlg_config.py --restart-latest --extend 5000
```

## Example Configs

| Config | Levels | Purpose |
|--------|--------|---------|
| `mlg_poiseuille_3d.py` | 2 | Basic validation |
| `mlg_poiseuille_4level.py` | 4 | Multi-level nesting test |
| `mlg_poiseuille_4level_uniform_yz.py` | 4 | Wall resolution study |

## Current Limitations

1. **No fine-level obstacle**: Obstacles exist only on Level 0
2. **No fine-level wall BC**: Fine grids touching domain walls don't apply wall BC
3. **3D only**: D3Q27 tested; D2Q9 and D3Q19 not yet verified
4. **Static refinement**: Fine regions are fixed at simulation start
5. **ParaView seam**: AMR visualization shows visual seams (not numerical errors)
