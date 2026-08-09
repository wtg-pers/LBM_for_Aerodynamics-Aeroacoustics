"""rig_cut geometry under the DEFAULT strict policy — must be REJECTED.

The "before" half of the wall-aware coupling pair. Identical to
`rig_cut.py` except that `mlg.wall_coupling` is absent, so the default
strict policy applies and the build must fail with:

    ValueError: Level 1: obstacle intersects the C2F/F2C coupling band on
    face(s): z_low (90 solid cells). ...

Keep it: it is the one-command reproduction of the rejection that
patch_notes/wall_coupling/01 is about, and it fails at build time so it
costs no GPU. A silent success here means the guard has been weakened.

    LBM_ESOTERIC=1 python main.py --config configs/testrig/_rig_cut_strict.py \\
        --gpu 0 --max-steps 2        # expected: ValueError, not a run
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _rig_base import build  # noqa: E402

config = build(mlg="cut", forces=True)
