"""Register or run the fixed train-only paper BN-buffer experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ect.paper_bn import register_experiment, run_recalibration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    commands=parser.add_subparsers(dest='command',required=True)
    register=commands.add_parser('register')
    register.add_argument('--output',required=True,type=Path)
    register.add_argument('--acceptance',required=True,type=Path)
    run=commands.add_parser('run')
    run.add_argument('--origin',required=True,type=Path)
    run.add_argument('--output',required=True,type=Path)
    run.add_argument('--registry',type=Path)
    run.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    run.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.command=='register':
        result=register_experiment(args.project_root,args.output,args.acceptance)
        print(json.dumps(dict(status=result['status'],runs=len(result['runs'])),indent=2))
    else:
        report=run_recalibration(args.project_root,args.origin,args.output,device=args.device,
                                smoke=args.smoke,registry_path=args.registry)
        print(json.dumps({k:report[k] for k in ('status','seed','sample_counts','metrics','calibration','elapsed_seconds')},indent=2))


if __name__=='__main__':
    main()
