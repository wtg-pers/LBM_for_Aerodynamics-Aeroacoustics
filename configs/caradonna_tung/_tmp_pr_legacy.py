import os,sys
sys.path.insert(0,os.path.dirname(__file__))
from _ct_hover_base import build_config
config=build_config(collective_deg=8.0,mtip=0.439,smoke=True)  # prandtl_loss=True (legacy R-eps)
config["time"]["logging_interval"]=1
_f="result_ct_smoke_prLeg"
config["output"]["output_dir"]="./%s/vtk"%_f
config["output"]["checkpoint_dir"]="./%s/checkpoints"%_f
config["output"]["csv_dir"]="./%s/csv"%_f
