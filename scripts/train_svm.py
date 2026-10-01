"""Run fixed SVM candidates on train and validation, preserving heldout test."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ect.baseline import train_svm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, default=Path('config/svm_v1.json'))
    parser.add_argument('--cache', type=Path, default=Path('data/processed/grouped_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/baselines/svm_pilot_v1'))
    parser.add_argument('--full-train', action='store_true', help='Use all train rows, retaining the candidate grid')
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    if args.full_train:
        config['train_per_class'] = None
    def progress(record):
        print(json.dumps({k: record[k] for k in ('index', 'parameters', 'fit_seconds', 'converged')} |
                         {'validation_macro_f1': record['metrics']['macro_f1']}), flush=True)
    report = train_svm(root, config, args.output, args.cache, progress=progress)
    print(json.dumps({'selected_candidate': report['selected_candidate'], 'sample_counts': report['sample_counts'],
                      'elapsed_seconds': report['elapsed_seconds'], 'heldout_test_evaluated': False,
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
