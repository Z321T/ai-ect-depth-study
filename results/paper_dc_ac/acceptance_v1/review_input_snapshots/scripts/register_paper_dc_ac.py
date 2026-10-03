"""Freeze the exploratory E2 experiment after actual acceptance."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ect.paper_dc_ac_registration import register_experiment

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--acceptance',required=True,type=Path)
    args=parser.parse_args()
    registry=register_experiment(args.project_root,args.output,args.acceptance)
    print(json.dumps(registry,indent=2))
