"""stl_sphere_gridcheck twin with wall_bc='ibb' — dry-run q build check.

    python -m to_claude.stl_grid_check --config configs/stl/stl_sphere_gridcheck_ibb.py
Expected: every level with solid nodes prints "Bouzidi IBB (ray-triangle q
from STL, ...)" and n_miss=0; f stays unallocated.
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "stl_sphere_gridcheck.py")
_spec = importlib.util.spec_from_file_location("gridcheck_ibb_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

config["internal_geometry"]["stl"]["wall_bc"] = "ibb"
