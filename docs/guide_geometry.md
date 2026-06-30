# Internal Geometry Guide

Covers the `internal_geometry` config section: immersed obstacles (cylinder, sphere, airfoil) with Halfway Bounce-Back wall treatment.

## No Obstacle

```python
internal_geometry = {"type": "none"}
# or
internal_geometry = {}
```

## Cylinder

2D or 3D. In 3D, the cylinder extends infinitely along the z-axis.

```python
internal_geometry = {
    "cylinder": {
        "enabled": True,
        "center_x": 50,      # [lu] Center x
        "center_y": 50,       # [lu] Center y
        "diameter": 20,       # [lu] Diameter
    },
}
```

## Sphere (3D only)

```python
internal_geometry = {
    "sphere": {
        "enabled": True,
        "center_x": 50, "center_y": 50, "center_z": 25,
        "diameter": 20,
    },
}
```

## NACA Airfoil (2D only)

```python
internal_geometry = {
    "airfoil": {
        "enabled": True,
        "naca": "0012",
        "chord": 60,
        "center_x": 80, "center_y": 50,
        "angle_of_attack": 5.0,   # [degrees]
    },
}
```

## Placement Guidelines

| Rule | Guideline |
|------|-----------|
| Upstream distance | center_x ≈ Nx/4 to Nx/5 |
| Lateral centering | center_y = Ny/2 |
| Blockage ratio | β = D/Ny < 5% (i.e., Ny ≥ 20D) |
| Wake resolution | At least 10D–15D downstream of object |
| Grid resolution | D ≥ 20 lu for acceptable surface accuracy |

## Wall Method

All obstacles use **Halfway Bounce-Back** (2nd-order accurate), applied automatically. Curved surfaces are represented by staircase approximation at the lattice resolution.

## Force Measurement

Enable `force_calculation` to measure drag/lift on the obstacle. See [guide_force.md](guide_force.md).

## Current Limitations

- Only one obstacle type per simulation
- Obstacles exist on Level 0 only (fine-level obstacle support planned)
- Staircase surface approximation (no interpolated bounce-back yet)
