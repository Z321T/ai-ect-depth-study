#!/usr/bin/env python3
"""Run the registered fixed BN ablation on a completed E2 selected model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ect.paper_dc_ac_bn import run_bn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', '--root', dest='root',
                        default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--origin', '--origin-run', dest='origin', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    report = run_bn(args.root, args.origin, args.output, args.device)
    print(json.dumps(dict(protocol=report['protocol'], status=report['status'],
        selected_epoch=report['selected_epoch'], sample_counts=report['sample_counts'],
        metrics=report['metrics']), allow_nan=False))


if __name__ == '__main__':
    main()
