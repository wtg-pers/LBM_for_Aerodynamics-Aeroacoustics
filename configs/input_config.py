"""
User parameter settings
"""
Nx = 100
Ny = 40
Nz = 40

simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz
    },
    "physics": {
        "Re": 100.0,
        "u_init": 0.1,
        "characteristic_length": 20
    },
    "time": {
        "max_steps": 20000,
        "output_interval": 200,         # VTK output interval
        "checkpoint_interval": 2000     # Checkpoint save interval
    }
}

boundaries = {
    "west": {"type": "inlet", "location": 0, "velocity": 0.1, "profile": "uniform"},
    "east": {"type": "outlet", "location": Nx - 1, "rho": 1.0, "k": 0.1},
    "south": {"type": "wall", "location": 0, "method": "bouzidi", "q": 0.5},
    "north": {"type": "wall", "location": Ny - 1, "method": "bouzidi", "q": 0.5},
    "bottom": {"type": "wall", "location": 0, "method": "bouzidi", "q": 0.5},
    "top": {"type": "wall", "location": Nz - 1, "method": "bouzidi", "q": 0.5}
}


# =============================================================================
# Output Configuration
# =============================================================================
output = {
    "vtk": {
        "enabled": True,
        "precision": "float32",     # 'float32' or 'float64'
        "compression_level": 6,     # 0 (none) to 9 (max compression)
        "variables": ["density", "pressure", "velocity", "velocity_magnitude", "solid_mask"]
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3           # Number of checkpoints to keep (0=keep all)
    }
}


# 3. 최종적으로 ConfigLoader가 가져갈 딕셔너리 통합
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "output": output
    # 필요 시 추가 섹션
}