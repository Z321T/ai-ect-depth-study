"""Create cache from immutable raw files and frozen manifests, on CPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ect.prepare import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests/grouped_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/grouped_v1"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report", type=Path, default=Path("results/preprocessing/prepare_run.json"))
    args = parser.parse_args()
    meta = prepare_dataset(args.project_root, args.manifest_dir, args.output, args.batch_size)
    report = args.report if args.report.is_absolute() else args.project_root / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"sample_counts": meta["sample_counts"], "elapsed_seconds": meta["elapsed_seconds"],
                      "cache": str(args.output), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
