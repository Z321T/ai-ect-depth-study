"""Run from project root: python scripts/audit_data.py."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.ect.audit import audit_arrays


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/MDDECT_v1_train.npy"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/MDDECT_v1_test.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit"))
    args = parser.parse_args()
    paths = {"train": args.train, "test": args.test}
    arrays = {name: np.load(path, allow_pickle=False, mmap_mode="r") for name, path in paths.items()}
    report, rows = audit_arrays(arrays)
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    report["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
    report["inputs"] = {name: {"path": str(path), "size_bytes": path.stat().st_size,
                              "sha256": file_sha256(path)} for name, path in paths.items()}
    report["code_sha256"] = {str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
                             for path in [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "src/ect/audit.py"]}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report["samples_sha256"] = file_sha256(args.output / "samples.csv")
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"datasets": report["datasets"], "overlap": report["overlap"],
                      "person_components": report["person_components"]}, indent=2))
    print(f"Audit saved to {args.output}")


if __name__ == "__main__":
    main()
