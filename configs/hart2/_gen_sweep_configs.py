"""
Generate the HART-II hover collective-sweep config set (Task 3).

Writes one production wrapper per collective in this directory:
    hart2_hover_c03.py ... hart2_hover_c16.py   (collective 3..16 deg @ 0.75R)

Each is a thin wrapper around _hart2_hover_base.build_config(collective).
Re-run to regenerate:  python configs/hart2/_gen_sweep_configs.py
"""
import os

COLLECTIVES = list(range(3, 17))   # 3,4,...,16 deg
HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = '''"""
HART-II hover, collective = {c}.0 deg @ 0.75R (production, Task 3 sweep point).

    python main.py --config configs/hart2/hart2_hover_c{c:02d}.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _hart2_hover_base import build_config

config = build_config(collective_deg={c}.0)
'''

if __name__ == "__main__":
    written = []
    for c in COLLECTIVES:
        path = os.path.join(HERE, "hart2_hover_c%02d.py" % c)
        with open(path, "w") as f:
            f.write(TEMPLATE.format(c=c))
        written.append(os.path.basename(path))
    print("Wrote %d sweep configs:" % len(written))
    print("  " + "  ".join(written))
    print("\nCluster run example:")
    print("  for c in %s; do" % " ".join("%02d" % c for c in COLLECTIVES))
    print("    python main.py --config configs/hart2/hart2_hover_c${c}.py")
    print("  done")
