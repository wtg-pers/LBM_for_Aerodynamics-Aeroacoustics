"""Task 1 — cluster MPI/mpi4py/CuPy verification (scp this to the 4x4090 box).

Verifies, in one shot, everything Task 1 needs before multi-GPU work:
  * mpi4py imports and COMM_WORLD wiring works
  * which MPI library mpi4py actually linked (CUDA-aware hint)
  * HOST-STAGING CuPy halo exchange works  (the always-available path)
  * DEVICE-DIRECT CuPy halo exchange        (works only on CUDA-aware MPI)

Run (pairs up ranks 0<->1, 2<->3, ... so n must be even):
    # host-staging only (safe; never aborts):
    mpirun -n 2 python3 task1_cluster_verify.py
    mpirun -n 4 python3 task1_cluster_verify.py

    # also probe device-direct (may SIGSEGV/abort if NOT CUDA-aware -> that
    # abort IS the answer: device pointers can't be shipped, use host staging):
    TEST_DEVICE_DIRECT=1 mpirun -n 2 python3 task1_cluster_verify.py

Interpretation:
    device_direct=PASS  -> CUDA-aware MPI: Task 4 halo can ship device pointers.
    device_direct=FAIL/abort -> NOT CUDA-aware: Task 4 halo must host-stage.
"""
import os
import sys
import numpy as np
# Open MPI 5.x routes CUDA-aware (GPUDirect) through the UCX PML, but a UCX built
# without multi-thread support can't be selected if MPI is initialized with
# MPI_THREAD_MULTIPLE (mpi4py's default). Our halo exchange is single-threaded per
# rank, so request only 'serialized' -> UCX PML is selectable -> CUDA-aware works.
# Override from the shell with MPI4PY_RC_THREAD_LEVEL=... if needed.
import mpi4py
mpi4py.rc.thread_level = os.environ.get("MPI4PY_RC_THREAD_LEVEL", "serialized")
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
peer = rank ^ 1  # pair (0,1), (2,3), ...

if size % 2 != 0:
    if rank == 0:
        print("ERROR: run with an even -n (ranks pair up 0<->1, 2<->3).")
    sys.exit(2)

# ---- 0) environment / which MPI did mpi4py link? -------------------------
libver = MPI.Get_library_version().strip().splitlines()[0]
if rank == 0:
    print(f"[env] mpi4py size={size}  MPI='{libver}'")
    # best-effort CUDA-aware hints from env (OpenMPI/UCX expose these)
    hints = {k: v for k, v in os.environ.items()
             if any(s in k for s in ("CUDA", "UCX", "GDR", "MPICH_GPU"))}
    if hints:
        print(f"[env] cuda/ucx-ish env: {hints}")

try:
    import cupy as cp
except Exception as e:  # noqa: BLE001
    if rank == 0:
        print(f"[FATAL] cupy import failed: {type(e).__name__}: {e}")
    sys.exit(3)

# Map rank -> GPU using the NODE-LOCAL rank (robust on multi-node; == rank on a
# single node). To restrict which physical GPUs are used, launch with e.g.
#   CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 ...   (ranks land on physical GPU 0 and 1)
# getDeviceCount() already reflects the CUDA_VISIBLE_DEVICES mask.
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK",      # Open MPI
                 os.environ.get("MV2_COMM_WORLD_LOCAL_RANK",        # MVAPICH
                 os.environ.get("SLURM_LOCALID", rank))))           # SLURM / fallback
ndev = cp.cuda.runtime.getDeviceCount()
dev = local_rank % ndev
cp.cuda.Device(dev).use()
# every rank reports its pin so GPU assignment is visible
print(f"[env] rank{rank} local_rank{local_rank} -> dev{dev} (visible GPUs={ndev})", flush=True)

N = 1 << 16
send_d = cp.arange(N, dtype=cp.float32) + rank * 1000.0
expected = cp.arange(N, dtype=cp.float32) + peer * 1000.0

# ---- A) host staging (always works) --------------------------------------
comm.Barrier()
send_h = cp.asnumpy(send_d)
recv_h = np.empty(N, dtype=np.float32)
comm.Sendrecv(send_h, dest=peer, sendtag=0, recvbuf=recv_h, source=peer, recvtag=0)
ok_A = bool(cp.allclose(cp.asarray(recv_h), expected))
ok_A_all = comm.allreduce(ok_A, op=MPI.LAND)
if rank == 0:
    print(f"[A host-staging ] all-ranks ok={ok_A_all}", flush=True)

if os.environ.get("TEST_DEVICE_DIRECT", "0") != "1":
    if rank == 0:
        print(f"RESULT host_staging={'PASS' if ok_A_all else 'FAIL'} "
              f"device_direct=SKIPPED(set TEST_DEVICE_DIRECT=1)", flush=True)
    sys.exit(0)

# ---- B) device direct (CUDA-aware only) ----------------------------------
comm.Barrier()
recv_d = cp.empty(N, dtype=cp.float32)
ok_B, err_B = False, ""
try:
    comm.Sendrecv(send_d, dest=peer, sendtag=1, recvbuf=recv_d, source=peer, recvtag=1)
    cp.cuda.runtime.deviceSynchronize()
    ok_B = bool(cp.allclose(recv_d, expected))
except Exception as e:  # noqa: BLE001
    ok_B, err_B = False, f"{type(e).__name__}: {e}"
ok_B_all = comm.allreduce(ok_B, op=MPI.LAND)
if rank == 0:
    print(f"[B device-direct] all-ranks ok={ok_B_all} err={err_B!r}", flush=True)
    print(f"RESULT host_staging={'PASS' if ok_A_all else 'FAIL'} "
          f"device_direct={'PASS' if ok_B_all else 'FAIL/UNSUPPORTED'}", flush=True)
