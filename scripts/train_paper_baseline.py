"""Train the registered finite-budget paper reconstruction on train/validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ect.paper_training import train_paper_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config',type=Path,default=Path('config/paper_baseline_v1.json'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'])
    parser.add_argument('--seed',type=int,choices=[0,1,2])
    parser.add_argument('--smoke',action='store_true',help='2 epochs, 2 train/validation rows per class; acceptance only')
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads((args.config if args.config.is_absolute() else root/args.config).read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    if 'seed' not in config:
        parser.error('--seed or a registered per-run config is required')
    if args.smoke:
        config.update(purpose='paper_runner_smoke_validation_only',max_epochs=2,validation_every_epochs=1,
                      train_per_class=2,validation_per_class=2)
    def progress(entry):
        if entry['validation'] is not None or entry['epoch']==1:
            print(json.dumps({k:entry[k] for k in ('epoch','train_loss','online_train_accuracy','validation','elapsed_seconds')}),flush=True)
    report = train_paper_baseline(root,config,args.output,progress=progress,
        registry_path=args.registry,device_override=args.device)
    print(json.dumps({k:report[k] for k in ('status','selected_epoch','sample_counts','metrics','elapsed_seconds')},indent=2))


if __name__=='__main__':
    main()
