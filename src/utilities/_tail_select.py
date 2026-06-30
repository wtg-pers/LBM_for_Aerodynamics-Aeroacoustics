"""Robust converged-tail selection for restarted runs.

Restart resets the per-run counters (`step`, `revolutions`, `time_lt`) back toward
zero, so the naive `rev >= rev_max - avg_revs` window can (a) span a counter reset
(mixing two run segments) or (b) silently average the whole short last segment
(including the post-restart transient) when `rev_max < avg_revs`.  It also reports
a misleading "revolutions" if the restarted segment's time base is rescaled.

`tail_mask` selects the tail of the FINAL run-segment only, and supports an explicit
step-based window (`avg_steps`) as a fallback when the revolutions axis is unreliable.
Used by hover_fm_post.py and compare_spanwise.py.
"""
import numpy as np


def last_segment_start(step, rev=None) -> int:
    """Index where the final run-segment starts.

    A restart resets a counter back toward zero.  Restarts seen in practice reset
    *either* the step counter (rev continues) *or* the revolutions/time counter
    (step continues via --max-steps) -- so a STRICT decrease in EITHER column marks
    a new segment.  Equal values are NOT a reset (blade_diagnostics repeats step/rev
    once per blade)."""
    step = np.asarray(step, dtype=float)
    rev = np.asarray(rev, dtype=float) if rev is not None else None
    start = 0
    for i in range(1, len(step)):
        reset = step[i] < step[i - 1]
        if rev is not None and rev[i] < rev[i - 1] - 1e-9:
            reset = True
        if reset:
            start = i
    return start


def tail_mask(step, rev, avg_revs, avg_steps=None):
    """Boolean mask over the full arrays selecting the converged tail.

    Restricts to the final run-segment (post last reset), then:
      - avg_steps set -> last `avg_steps` by the step column (step-based window);
      - else          -> last `avg_revs` by the rev column within that segment;
                         if the segment spans < avg_revs (or rev is non-monotonic),
                         fall back to the whole final segment (info['short_tail']).

    Returns (mask, info).  info: n, restarted, short_tail, rev_span, step_lo/hi.
    """
    step = np.asarray(step, dtype=float)
    rev = np.asarray(rev, dtype=float)
    n = len(step)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask, {"n": 0, "restarted": False, "short_tail": True,
                      "rev_span": 0.0, "step_lo": float("nan"),
                      "step_hi": float("nan")}
    s0 = last_segment_start(step, rev)
    seg_step = step[s0:]
    seg_rev = rev[s0:]
    short = False
    if avg_steps is not None:
        sel = seg_step >= (seg_step[-1] - float(avg_steps))
    else:
        span = float(seg_rev[-1] - seg_rev[0])
        monotonic = bool(np.all(np.diff(seg_rev) >= -1e-9))
        if span >= avg_revs and monotonic:
            sel = seg_rev >= (seg_rev[-1] - avg_revs)
        else:
            sel = np.ones(len(seg_rev), dtype=bool)   # too short -> whole segment
            short = True
    mask[s0:] = sel
    sel_rev = rev[mask]
    return mask, {
        "n": int(mask.sum()),
        "restarted": bool(s0 > 0),
        "short_tail": bool(short),
        "rev_span": float(sel_rev[-1] - sel_rev[0]) if mask.sum() else 0.0,
        "step_lo": float(step[mask][0]) if mask.sum() else float("nan"),
        "step_hi": float(step[mask][-1]) if mask.sum() else float("nan"),
    }
