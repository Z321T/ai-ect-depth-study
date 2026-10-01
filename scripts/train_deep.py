"""Train CNN/ResNet on frozen train/validation; never score heldout test."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ect.deep import train_deep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, default=Path('config/deep_v1.json'))
    parser.add_argument('--cache', type=Path, default=Path('data/processed/grouped_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['cnn', 'resnet'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--augment', action='store_true', help='ResNet 50%% clean / 50%% 10..30 dB')
    parser.add_argument('--smoke', action='store_true', help='2 epochs, 10 rows/class in train AND validation; not a comparison')
    args = parser.parse_args()
    root = args.project_root.resolve()
    path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(path.read_text())
    for key in ('model', 'device', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.augment:
        config['augmentation'] = True
    if args.smoke:
        config.update(purpose='smoke_validation_only', train_per_class=10, validation_per_class=10, max_epochs=2, patience=2)
    def progress(epoch):
        print(json.dumps({key: epoch[key] for key in ('epoch', 'train_loss', 'selection_score', 'augmented_count')}), flush=True)
    report = train_deep(root, config, args.output, args.cache, progress=progress)
    print(json.dumps({key: report[key] for key in ('status', 'selected_epoch', 'sample_counts', 'model_roundtrip_verified', 'heldout_test_evaluated')}, indent=2))


if __name__ == '__main__':
    main()
