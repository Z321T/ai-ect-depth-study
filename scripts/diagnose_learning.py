"""Diagnose frozen formal_v1 models on train/validation only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ect.learning_diagnostics import diagnose_learning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--registry', default='results/experiments/formal_v1/registry.json')
    parser.add_argument('--cache', default='data/processed/grouped_v1')
    args = parser.parse_args()
    report = diagnose_learning(ROOT, args.output, args.device, args.registry, args.cache,
                               progress=lambda message: print(message, flush=True))
    print(f'Complete: {report["prediction_count"]} predictions; test not evaluated', flush=True)


if __name__ == '__main__':
    main()
