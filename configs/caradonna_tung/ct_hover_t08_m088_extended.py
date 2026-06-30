"""
Caradonna-Tung hover — θ=8°, M_tip=0.877, EXTENDED fine regions (DGX Spark).

Tests the leading hypothesis for the tip over-loading: the trailed tip vortex
leaves the finest MLG box (light L3 = only 0.25D downstream) and is lost before
it can induce downwash back at the tip.  Here dx is UNCHANGED from 'light'
(D=32) but the finest level follows the descending/contracting tip vortex
(downstream 0.25D -> 1.0D, lateral ±0.6D -> ±0.7D) -> isolates EXTENT from dx.

If the tip inflow angle recovers (phi: 0 -> positive) and C_T drops toward the
experimental 0.00473, the limit was fine-region EXTENT (fixable); if not, it is
the ALM tip model itself.

~80M cells, ~33 GB -> needs the DGX Spark (128 GB unified memory); slower
wall-clock (bandwidth-bound) but fits.  SGS off (not the driver; saves nothing
but keeps it clean — flip use_sgs=True if desired).

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_extended.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877, preset="extended",
                      use_sgs=False)
