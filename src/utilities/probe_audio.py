"""
probes.csv -> p' -> WAV: render a pressure-probe time series audible.

The probe channel (src/io/probe_writer.py) records p [Pa] at fs = 1/dt(L0),
which for acoustic-scaled runs is far above the audio band — so the series
can be resampled straight down to a sound card rate. Pipeline:

    1. load probes.csv (restart overlaps deduped: last write wins per step)
    2. drop the initial transient (--t-start), subtract the mean -> p'(t)
    3. FFT resample (brickwall anti-alias) onto the output rate
    4. peak-normalize to -1 dBFS, write 16-bit WAV (stdlib `wave`)

`--slow S` time-stretches by S (duration xS, frequencies /S) for content
that sits above the audible band. Reported per probe: p'_rms and SPL
(dB re 20 uPa) — the "is this audible" number.

Usage (from the repo root):
    python -m src.utilities.probe_audio results/csv/probes.csv
    python -m src.utilities.probe_audio results/csv/probes.csv \
        --probe 0 --t-start 0.5 --rate 48000 --slow 1 --out results/audio

numpy + stdlib only (cluster-safe, Python 3.9).

Author: LBM Development Team
Date: 2026-08
"""

import argparse
import os
import wave
from typing import List, Optional, Tuple

import numpy as np


def load_probe_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read probes.csv -> (steps, time_s, p[n_rows, n_probes]).

    Restarted runs may re-record a step range already on disk (restart
    from an interval checkpoint older than the crash point). Dedupe by
    step index, LAST occurrence wins (the rerun supersedes), then verify
    the remaining series is uniform — resampling assumes it.
    """
    data = np.genfromtxt(path, delimiter=',', names=True)
    if data.ndim == 0:
        data = data.reshape(1)
    cols = list(data.dtype.names)
    if cols[:2] != ['step', 'time_s']:
        raise ValueError(f"{path}: expected 'step,time_s,p0_pa,...' header, "
                         f"got {cols[:3]}")
    steps = data['step'].astype(np.int64)
    # last occurrence per step: scan reversed, np.unique keeps the FIRST
    # hit it sees, which in reversed order is the latest write
    _, ridx = np.unique(steps[::-1], return_index=True)
    sel = np.sort(steps.size - 1 - ridx)
    steps = steps[sel]
    t = data['time_s'][sel]
    p = np.column_stack([data[c][sel] for c in cols[2:]])
    d = np.diff(steps)
    if d.size and not np.all(d == d[0]):
        raise ValueError(
            f"{path}: non-uniform step spacing after dedupe "
            f"(gaps at steps {steps[:-1][d != d[0]][:5]}...) — "
            "the series has holes; re-record or trim to a uniform span")
    return steps, t, p


def fft_resample(x: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a uniform series to n_out points (brickwall anti-alias)."""
    X = np.fft.rfft(x)
    n_keep = n_out // 2 + 1
    if n_keep <= X.size:          # downsample: truncate spectrum
        Y = X[:n_keep]
    else:                          # upsample: zero-pad spectrum
        Y = np.zeros(n_keep, dtype=X.dtype)
        Y[:X.size] = X
    return np.fft.irfft(Y, n=n_out) * (n_out / x.size)


def write_wav(path: str, x: np.ndarray, rate: int) -> None:
    """Peak-normalize to -1 dBFS and write mono 16-bit PCM."""
    peak = float(np.max(np.abs(x))) or 1.0
    pcm = np.clip(x / peak * 0.891, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Render probes.csv pressure series as WAV audio")
    ap.add_argument('csv', help="path to probes.csv")
    ap.add_argument('--probe', default='all',
                    help="probe index, comma list, or 'all' (default)")
    ap.add_argument('--t-start', type=float, default=0.0,
                    help="discard t < this [s] (startup transient)")
    ap.add_argument('--rate', type=int, default=48000,
                    help="WAV sample rate [Hz] (default 48000)")
    ap.add_argument('--slow', type=float, default=1.0,
                    help="time-stretch factor: duration xS, freq /S")
    ap.add_argument('--out', default=None,
                    help="output dir (default: alongside the CSV)")
    args = ap.parse_args(argv)

    steps, t, p = load_probe_csv(args.csv)
    keep = t >= args.t_start
    if keep.sum() < 16:
        raise ValueError(
            f"only {int(keep.sum())} samples at t >= {args.t_start}s — "
            "nothing to render")
    t, p = t[keep], p[keep]
    dt = float(np.median(np.diff(t)))
    fs_sim = 1.0 / dt
    T = t[-1] - t[0]

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    idx = (range(p.shape[1]) if args.probe == 'all'
           else [int(s) for s in args.probe.split(',')])
    print(f"probes.csv: {p.shape[0]} samples, fs_sim = {fs_sim:.4g} Hz, "
          f"T = {T:.4g} s, {p.shape[1]} probe(s)")
    n_out = max(16, int(round(T * args.rate * args.slow)))
    for i in idx:
        pp = p[:, i] - p[:, i].mean()
        rms = float(np.sqrt(np.mean(pp ** 2)))
        spl = 20.0 * np.log10(rms / 2e-5) if rms > 0 else float('-inf')
        y = fft_resample(pp, n_out)
        path = os.path.join(out_dir, f"mic{i}.wav")
        write_wav(path, y, args.rate)
        print(f"  mic{i}: p'_rms = {rms:.4e} Pa ({spl:.1f} dB SPL), "
              f"{n_out / args.rate:.2f} s @ {args.rate} Hz"
              + (f" (x{args.slow:g} slow)" if args.slow != 1.0 else "")
              + f" -> {path}")


if __name__ == '__main__':
    main()
