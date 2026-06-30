# Output & Checkpoint Guide

Covers the `output` config section and `simulation.time` timing parameters: VTK visualization, checkpoint restart, CSV logging, and directory structure.

## Config

```python
output = {
    "output_dir": "./results/vtk",
    "checkpoint_dir": "./results/checkpoints",
    "csv_dir": "./results/csv",
    "clear_previous": True,

    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 0,
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3,
    },
}
```

## Directories

| Key | Default | Description |
|-----|---------|-------------|
| `output_dir` | `"./results/vtk"` | VTK file directory |
| `checkpoint_dir` | `"./results/checkpoints"` | Checkpoint directory |
| `csv_dir` | `"./results/csv"` | CSV log directory |
| `clear_previous` | `False` | Delete old results on fresh start |

CLI overrides: `--output-dir`, `--checkpoint-dir`, `--csv-dir`, `--clear`

### Directory Structure

```
results/
├── vtk/
│   ├── lbm_00000000.vti            # Single grid
│   ├── simulation.pvd              # Time-series (open in ParaView)
│   ├── vth/                        # MLG only
│   │   └── lbm_00000000.vth
│   ├── level0/                     # MLG only
│   │   └── lbm_00000000_level0.vti
│   └── markers/                    # ALM only
│       └── markers_00000000.vtp
├── checkpoints/
│   └── checkpoint_00005000.npz
└── csv/
    ├── setup_log.txt               # Detailed setup log
    ├── mass_conservation.csv
    ├── force_history.csv           # If force_calculation enabled
    └── rotor_performance.csv       # If ALM enabled
```

## VTK Output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | True | Enable VTK output |
| `precision` | str | `"float32"` | `"float32"` or `"float64"` |
| `compression_level` | int | 0 | 0–9 (0=none, 9=max) |

Output frequency is set in `simulation.time.output_interval`.

### VTK Fields

| Field | Type | Unit |
|-------|------|------|
| density | Scalar | [-] |
| velocity | Vector(3) | [Δx/Δt] |
| velocity_magnitude | Scalar | [Δx/Δt] |
| solid_mask | Integer | 0=fluid, 1=solid |
| body_force | Vector(3) | ALM force (if active) |

### ParaView

- **Single grid**: Open `simulation.pvd` for time-series animation
- **MLG**: Select all `.vth` files in `vtk/vth/` for time-series
- **ALM markers**: Open `.vtp` files alongside flow field

Disable: `--no-vtk` CLI flag

## Checkpoint

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | True | Enable checkpoints |
| `keep_last_n` | int | 3 | Keep only last N files (0=keep all) |

Checkpoint frequency: `simulation.time.checkpoint_interval`.

### File Format

NumPy compressed `.npz`. Contains:

| Key | Content |
|-----|---------|
| `f` | Distribution function (Level 0) |
| `rho`, `u` | Macroscopic fields |
| `step` | Current step number |
| `tau`, `config_json` | Metadata |
| `f_level_1`, `f_level_2`, ... | MLG fine level distributions |
| `num_levels` | MLG level count |

### Restart

```bash
python main.py --config ... --restart-latest                    # Latest checkpoint
python main.py --config ... --restart ./checkpoints/ckpt.npz    # Specific file
python main.py --config ... --restart-latest --extend 10000     # Add 10000 steps
python main.py --config ... --restart-latest --max-steps 50000  # Run until 50000
```

## Time Settings

```python
"time": {
    "max_steps": 50000,           # Total steps
    "output_interval": 500,       # VTK every N steps
    "checkpoint_interval": 5000,  # Checkpoint every N steps
    "probe_interval": 10,         # Force/conservation check interval
},
```

### Interval Guidelines

| Setting | Quick Test | Normal | Precision |
|---------|-----------|--------|-----------|
| output_interval | 100 | 500–1000 | 100–500 |
| checkpoint_interval | 1000 | 5000 | 2000 |
| probe_interval | 10 | 10–50 | 1–10 |

## CLI Reference

```bash
--no-vtk          # Skip VTK output
--no-force        # Skip force calculation
--clear           # Clear previous results
--verbose / -v    # Echo detailed setup log to terminal
--gpu N           # Select GPU device
--list-gpus       # Show available GPUs
```
