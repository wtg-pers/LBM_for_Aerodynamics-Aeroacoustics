"""Sphere + 2 counter-rotating rotors — MLG variant 'blocks'. See _rig_base."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _rig_base import build

config = build(mlg="blocks")
