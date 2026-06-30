# Multi-GPU Domain Decomposition Design

**Date:** 2026-04-12
**Target:** 43M+ nodes, x-axis decomposition, CuPy + NCCL

---

## 1. Architecture Overview

```
GPU 0                    GPU 1                    GPU 2
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ f[27,Nx0+2,  │  NCCL  │ f[27,Nx1+2,  │  NCCL  │ f[27,Nx2+2,  │
│     Ny, Nz]  │◄──────►│     Ny, Nz]  │◄──────►│     Ny, Nz]  │
│  ┌────────┐  │        │  ┌────────┐  │        │  ┌────────┐  │
│  │ halo=1 │  │        │  │ halo=1 │  │        │  │ halo=1 │  │
│  │ local  │  │        │  │ local  │  │        │  │ local  │  │
│  │ halo=1 │  │        │  │ halo=1 │  │        │  │ halo=1 │  │
│  └────────┘  │        │  └────────┘  │        │  └────────┘  │
└──────────────┘        └──────────────┘        └──────────────┘
  x: [0, Nx0)            x: [Nx0, Nx0+Nx1)      x: [Nx0+Nx1, Nx_total)
```

### 1.1 Decomposition Strategy

- **Axis:** x-axis only (simplest, load-balanced for channel/pipe flows)
- **Halo width:** 1 cell (D3Q27 stencil requires 1 neighbor)
- **Local domain:** `(Nx_local + 2*halo, Ny, Nz)` per GPU
- **Communication:** NCCL AllGather or P2P Send/Recv via CuPy

### 1.2 Memory Layout Per GPU

```python
# Global domain: (Nx_total, Ny, Nz)
# GPU rank r owns x-indices [x_start, x_start + Nx_local)
# Local array includes halos: (Nx_local + 2, Ny, Nz)
#
# Index mapping:
#   local_x = 0          → left halo  (from rank r-1)
#   local_x = 1..Nx_local → owned cells
#   local_x = Nx_local+1 → right halo (from rank r+1)
```

---

## 2. Communication Protocol

### 2.1 Halo Exchange (per LBM step)

After each `advance()` call, exchange boundary planes:

```
Rank r sends:
  f[:, 1, :, :]         → Rank r-1 (right halo of left neighbor)
  f[:, Nx_local, :, :]  → Rank r+1 (left halo of right neighbor)

Rank r receives:
  f[:, 0, :, :]         ← Rank r-1 (left halo)
  f[:, Nx_local+1, :, :] ← Rank r+1 (right halo)
```

**Data volume:** `27 × Ny × Nz × 4B (float32)` per face
- For 43M nodes (360×360×330): `27 × 360 × 330 × 4 = 12.8 MB` per face
- Two faces per rank: ~25.6 MB total per step

### 2.2 NCCL Communication

```python
import cupy
from cupy.cuda import nccl

# Initialize (once)
comm = nccl.NcclCommunicator(n_gpus, rank_id, unique_id)

# Halo exchange (each step)
# Send left boundary, receive right halo (pipelined)
comm.send(f_send_left, peer=rank-1, stream=stream)
comm.recv(f_recv_left, peer=rank-1, stream=stream)
comm.send(f_send_right, peer=rank+1, stream=stream)
comm.recv(f_recv_right, peer=rank+1, stream=stream)
stream.synchronize()
```

### 2.3 Esoteric Pull Compatibility

Esoteric Pull uses single-buffer in-place streaming with parity-dependent
slot access. Halo exchange must account for:

1. **Parity synchronization:** All GPUs must be at the same parity (even/odd step)
2. **Exchange physical f:** Use `esoteric_gather_physical()` at halo cells,
   exchange in standard D3Q27 ordering, then `esoteric_scatter_physical()` on receiving side
3. **Alternative:** Exchange raw Esoteric buffer at boundary planes (simpler,
   but requires matching parity — which is guaranteed by lockstep execution)

**Recommended:** Exchange raw Esoteric buffer directly (no gather/scatter overhead).
Both GPUs are at the same parity in lockstep mode, so the raw buffer is consistent.

---

## 3. MLG Compatibility

### 3.1 Per-Level Decomposition

Each MLG level is independently decomposed along x:

```
Level 0 (coarse):  Nx_c / n_gpus per GPU
Level 1 (fine):    Nx_f / n_gpus per GPU (Nx_f = 2*Nx_c)
```

Each GPU runs its own `MultiLevelGrid` with local domain sizes.

### 3.2 Coupling at GPU Boundaries

MLG coupling (C2F, F2C) requires data from neighboring cells for:
- **Spatial interpolation (C2F):** 4th-order cubic needs 3 coarse cells → 2-cell overlap
- **Low-pass filter (F2C):** 1-2 passes → 1-2 cell dependency

At GPU boundaries, coupling data may extend into neighbor's domain.
**Solution:** Extend halo width for MLG levels:
- Level 0 (coarse): halo = 2 (for cubic interpolation in C2F)
- Level k (fine): halo = 1 (streaming only; coupling uses level k-1 halo data)

### 3.3 Overlap Region Constraints

The MLG `fine_region` should not straddle GPU boundaries. If a fine region
spans multiple GPUs:
- Split fine_region at GPU boundaries
- Each GPU owns the fine_region portion within its x-range
- Halo exchange for fine level covers coupling overlap

---

## 4. File Structure

```
src/
  parallel/
    __init__.py
    domain_decomposition.py   — DomainDecomposition class
    halo_exchange.py          — HaloExchange (NCCL communication)
    partition.py              — Partition info (rank, x_start, Nx_local)
```

### 4.1 Key Interfaces

```python
class Partition:
    """Per-GPU domain partition info."""
    rank: int
    n_ranks: int
    x_start: int      # Global x-index of first owned cell
    nx_local: int      # Number of owned cells in x
    ny: int            # Full y-extent (not decomposed)
    nz: int            # Full z-extent (not decomposed)
    halo_width: int    # 1 for D3Q27

class HaloExchange:
    """NCCL-based halo exchange for Esoteric Pull arrays."""
    def __init__(self, partition: Partition, comm: NcclCommunicator): ...
    def exchange(self, f: cp.ndarray) -> None: ...

class DomainDecomposition:
    """Multi-GPU simulation orchestrator."""
    def __init__(self, config, n_gpus): ...
    def advance(self) -> None:
        self.sim.advance()     # Local LBM step
        self.halo.exchange(self.sim.f)  # NCCL halo exchange
    def gather_field(self, field_name) -> cp.ndarray:
        """Gather full field to rank 0 for output."""
```

---

## 5. Implementation Phases

### Phase 1: Single-Level Decomposition
- `Partition` + `HaloExchange` + `DomainDecomposition`
- Test: Poiseuille channel with 2 GPUs
- Validate: profile matches single-GPU result

### Phase 2: Esoteric Pull Integration
- Raw buffer exchange at halo planes
- Parity-locked execution
- Test: Sphere drag with 2 GPUs

### Phase 3: MLG Integration
- Extended halo for coarse level (width=2)
- Per-level halo exchange
- Coupling at GPU boundaries
- Test: 2-level MLG with 2 GPUs

### Phase 4: ALM Integration
- Marker positions may span multiple GPUs
- Interpolation: each GPU interpolates markers within its domain
- Spreading: each GPU spreads markers within its domain + halo
- Global reduction for markers near GPU boundaries

---

## 6. Performance Estimates

### 6.1 Communication Overhead

For 43M nodes (360×360×330) on 4 GPUs:
- Per-GPU: ~10.8M nodes
- Halo face: 27 × 360 × 330 × 4B = 12.8 MB
- PCIe 4.0 x16: ~25 GB/s → 0.5 ms per face
- NVLink: ~600 GB/s → 0.02 ms per face
- **Communication/computation ratio:** ~5% (PCIe) or <1% (NVLink)

### 6.2 Expected Scaling

- Single GPU (1.46M nodes): 3075 MLUPS @ 0.47 ms/step
- 4 GPUs (43M nodes): ~3000 MLUPS per GPU × 4 = ~12000 MLUPS
- With communication: ~11000 MLUPS (92% efficiency, PCIe)
