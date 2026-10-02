"""Evaluate the registered models with fixed repeated noise; never train or tune."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ect.evaluation import evaluate_registered


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--registry',default='results/experiments/formal_v1/registry.json')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--cache',default='data/processed/grouped_v1')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--output',required=True)
    p.add_argument('--acceptance',help='Verified validation acceptance JSON required for test')
    a=p.parse_args()
    report=evaluate_registered(a.project_root,a.registry,a.split,a.output,a.cache,a.device,a.acceptance,
                              progress=lambda x:print(json.dumps(x),flush=True))
    print(json.dumps({k:report[k] for k in ('status','split','sample_count','model_count','prediction_count','heldout_test_evaluated')},indent=2))


if __name__=='__main__':
    main()
