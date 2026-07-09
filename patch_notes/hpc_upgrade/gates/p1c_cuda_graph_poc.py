"""Phase 1c PoC: CUDA Graph capture/replay of a many-small-kernel sequence.

Validates the mechanism the MLG launch-bound advance() would use:
 (1) capture a long sequence of small kernel launches, replay → bit-identical
 (2) graph replay removes per-launch overhead (wall speedup)
 (3) whether allocation DURING capture works (Stage A coupling allocs slabs)
"""
import cupy as cp
import time

n = 64 * 64 * 64                     # modest array (launch-bound regime)
K = 120                             # ~ MLG kernels/coarse-step order
a0 = cp.arange(n, dtype=cp.float64)
work = cp.RawKernel(r'''
extern "C" __global__ void bump(double* x, long long n) {
    long long i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i]*1.0000001 + 1.0;
}''', 'bump')
threads = 256; blocks = (n + threads - 1)//threads

def run_direct(x):
    for _ in range(K):
        work((blocks,), (threads,), (x, n))

# ---- reference (direct) ----
x_ref = a0.copy()
run_direct(x_ref); cp.cuda.runtime.deviceSynchronize()

# ---- (1)+(2) capture & replay ----
s = cp.cuda.Stream(non_blocking=True)
x_g = a0.copy()
with s:
    s.begin_capture()
    for _ in range(K):
        work((blocks,), (threads,), (x_g, n), stream=s)
    graph = s.end_capture()
    graph.launch(stream=s)
s.synchronize()
d = float(cp.abs(x_ref - x_g).max())
print(f"(1) graph replay vs direct: max|Δ| = {d:.3e}  {'OK' if d==0 else 'MISMATCH'}")

# timing: direct vs graph replay (both K launches)
for _ in range(3): run_direct(x_ref)
cp.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
for _ in range(50): run_direct(x_ref)
cp.cuda.runtime.deviceSynchronize()
t_direct = (time.perf_counter()-t0)/50*1e3
for _ in range(3):
    with s: graph.launch(stream=s)
s.synchronize()
t0 = time.perf_counter()
with s:
    for _ in range(50): graph.launch(stream=s)
s.synchronize()
t_graph = (time.perf_counter()-t0)/50*1e3
print(f"(2) {K} launches: direct {t_direct:.3f} ms  graph {t_graph:.3f} ms  speedup {t_direct/t_graph:.2f}x")

# ---- (3) allocation during capture? ----
s2 = cp.cuda.Stream(non_blocking=True)
alloc_ok = None
try:
    y = a0.copy()
    with s2:
        s2.begin_capture()
        scratch = cp.empty(n, dtype=cp.float64)   # alloc INSIDE capture
        scratch[:] = y
        work((blocks,), (threads,), (scratch, n), stream=s2)
        y[:] = scratch
        g2 = s2.end_capture()
        g2.launch(stream=s2)
    s2.synchronize()
    alloc_ok = "WORKS"
except Exception as e:
    alloc_ok = f"FAILS ({type(e).__name__}: {str(e)[:60]})"
print(f"(3) allocation during capture: {alloc_ok}")
print("VERDICT:", "graph capture viable" if d == 0 else "FAIL")
