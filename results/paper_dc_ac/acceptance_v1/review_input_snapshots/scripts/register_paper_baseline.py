"""Freeze three paper-baseline runs after verified runner acceptance; no training."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.ect.paper_registration import register_paper_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--config', default='config/paper_baseline_v1.json')
    parser.add_argument('--cache', default='data/processed/grouped_v1')
    parser.add_argument('--output', default='results/paper_baseline/finite_v1')
    parser.add_argument('--acceptance', required=True, help='Verified runner acceptance JSON')
    args = parser.parse_args()
    registry = register_paper_baseline(args.project_root, args.output, args.acceptance,
                                       args.config, args.cache)
    print(json.dumps({'status': registry['status'], 'runs': len(registry['runs']),
                      'test_classification_enabled': False}, indent=2))


if __name__ == '__main__':
    main()
