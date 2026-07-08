"""MLG coarse-step profiler — env-gated, GPU-synced per-section wall clock.

HPC-upgrade Phase 0 instrument (patch_notes/hpc_upgrade/PLAN.md §0). Decomposes
one coarse timestep into per-level LBM advance / coupling / f_prev copies so the
**launch-bound vs kernel-bound** verdict can be made: compare each level's
synced-wall against its bandwidth floor (cells * 216 B * 2 / GPU_BW). If wall
>> floor, the coarse step is dominated by kernel-launch / CPU-GPU sync, not
bandwidth -> Phase 1c (CUDA Graphs + ALM overlap) is the top speed lever.

Design mirrors the ALM_PROFILE convention in actuator_line.py:
  * inert unless MLG_PROFILE is set (one bool test + a shared no-op CM per
    section; zero allocation on the hot production path)
  * GPU-synced wall clock per named section (sums are perturbed by the extra
    per-section sync — that is the point of nsys cross-check; use for relative
    attribution + the floor comparison, not absolute throughput)
  * running means printed every MLG_PROFILE_EVERY coarse steps
  * MLG_NVTX=1 additionally emits NVTX ranges so `nsys profile` gives the
    un-perturbed kernel/gap timeline with the same section names.

Usage:
    MLG_PROFILE=1 python main.py --config configs/hpc_bench/bench5_baseline.py
    MLG_NVTX=1 nsys profile -t cuda,nvtx -o bench5 \
        python main.py --config configs/hpc_bench/bench5_baseline.py
    # combine with the ALM phase profiler for the full step budget:
    ALM_PROFILE=1 MLG_PROFILE=1 python main.py --config ...
"""
import os
import time
from contextlib import contextmanager

_ENABLED = "MLG_PROFILE" in os.environ
_NVTX = os.environ.get("MLG_NVTX", "0") == "1"
_EVERY = int(os.environ.get("MLG_PROFILE_EVERY", "100"))

_cp = None
_nvtx = None
if _ENABLED or _NVTX:
    try:
        import cupy as _cp  # noqa: F811
        if _NVTX:
            from cupy.cuda import nvtx as _nvtx  # noqa: F811
    except Exception:  # noqa: BLE001 — CPU backend or no cupy: sync/nvtx off
        _cp = None
        _nvtx = None
        _NVTX = False


def _sync() -> None:
    if _cp is not None:
        _cp.cuda.Stream.null.synchronize()


class _Noop:
    """Shared zero-cost context manager returned when profiling is disabled."""

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NOOP = _Noop()


class _Profiler:
    def __init__(self) -> None:
        self._acc: dict[str, float] = {}   # name -> total seconds
        self._cnt: dict[str, int] = {}     # name -> total calls
        self._order: list[str] = []        # stable table order (first-seen)
        self._steps = 0                    # coarse steps completed

    @contextmanager
    def timed(self, name: str):
        if _nvtx is not None:
            _nvtx.RangePush(name)
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            dt = time.perf_counter() - t0
            if name not in self._acc:
                self._acc[name] = 0.0
                self._cnt[name] = 0
                self._order.append(name)
            self._acc[name] += dt
            self._cnt[name] += 1
            if _nvtx is not None:
                _nvtx.RangePop()

    def step_done(self) -> None:
        self._steps += 1
        if _EVERY > 0 and self._steps % _EVERY == 0:
            self.report()

    def report(self) -> None:
        if not self._steps:
            return
        total = sum(self._acc.values())
        print(
            f"\n[MLG_PROFILE] mean over {self._steps} coarse steps "
            f"(GPU-synced wall; per-section sync inflates sums — "
            f"use for attribution, cross-check absolutes with nsys):"
        )
        print(f"  {'section':<16}{'calls/step':>11}{'ms/step':>11}{'% step':>9}")
        for name in self._order:
            ms = self._acc[name] / self._steps * 1e3
            calls = self._cnt[name] / self._steps
            pct = (self._acc[name] / total * 100.0) if total else 0.0
            print(f"  {name:<16}{calls:>11.1f}{ms:>11.2f}{pct:>8.1f}%")
        print(
            f"  {'TOTAL':<16}{'':>11}"
            f"{total / self._steps * 1e3:>11.2f}{100.0:>8.1f}%"
        )


profiler = _Profiler()


def timed(name: str):
    """Context manager timing `name`; a shared no-op unless MLG_PROFILE is set."""
    return profiler.timed(name) if _ENABLED else _NOOP


def step_done() -> None:
    """Mark one coarse step complete (triggers the periodic report)."""
    if _ENABLED:
        profiler.step_done()


def report() -> None:
    """Force-print the current running means (e.g. at finalize)."""
    if _ENABLED:
        profiler.report()
