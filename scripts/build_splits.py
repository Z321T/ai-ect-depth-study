"""Build the current MDDECT v1 grouped protocol from verified audit artifacts."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_data import file_sha256
from src.ect.audit import validate_sample_rows
from src.ect.splits import build_grouped_manifest
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("results/data_audit"))
    parser.add_argument("--output", type=Path, default=Path("manifests/grouped_v1"))
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    audit_path = args.audit_dir / "audit.json"
    samples_path = args.audit_dir / "samples.csv"
    audit = json.loads(audit_path.read_text())
    if audit.get("samples_sha256") != file_sha256(samples_path):
        raise ValueError("Audited sample table changed or audit schema is outdated; rerun audit")
    if audit["cross_class_duplicate_groups"] or any(s["nonfinite_values"] for s in audit["datasets"].values()):
        raise ValueError("Resolve class conflicts or nonfinite data before creating splits")
    for entry in audit["inputs"].values():
        if file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Input file changed after audit")
    with samples_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("person", "angle", "direction", "repeat", "class_index", "multiplicity"):
            row[key] = int(row[key])
    if len(rows) != sum(s["sample_count"] for s in audit["datasets"].values()):
        raise ValueError("Incomplete audited sample table")
    arrays = {name: np.load(entry["path"], allow_pickle=False, mmap_mode="r")
              for name, entry in audit["inputs"].items()}
    validate_sample_rows(rows, arrays)
    components = audit["person_components"]
    required = [i for i, c in enumerate(components) if c == ["test:0"]]
    singles = [i for i, c in enumerate(components) if len(c) == 1 and c[0].startswith("train:")]
    if len(required) != 1 or len(singles) != 9 or sorted(map(len, components)) != [1] * 10 + [10] * 2:
        raise ValueError("This policy requires the audited MDDECT v1 component structure")
    random.Random(args.seed).shuffle(singles)
    assignment = {i: "train" for i in range(len(components))}
    for i in required + singles[:2]:
        assignment[i] = "test"
    for i in singles[2:4]:
        assignment[i] = "validation"
    manifest, summary = build_grouped_manifest(rows, components, assignment)
    if summary["sample_counts"] != {"train": 24000, "validation": 3200, "test": 4800}:
        raise ValueError("Unexpected unique counts; audit and split policy need review")
    expected = {"train": 1200, "validation": 160, "test": 240}
    for name, counts in summary["class_counts"].items():
        if counts != {i: expected[name] for i in range(20)}:
            raise ValueError("Unexpected class coverage")
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, records in manifest.items():
        path = args.output / f"{name}.jsonl"
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        files[path.name] = file_sha256(path)
    summary.update(schema_version=1, protocol="grouped_v1", seed=args.seed,
                   class_mapping_status="unverified", inputs=audit["inputs"],
                   audit_sha256=file_sha256(audit_path), samples_sha256=file_sha256(samples_path),
                   manifest_sha256=files, components=components,
                   code_sha256={"scripts/build_splits.py": file_sha256(Path(__file__)),
                                "src/ect/splits.py": file_sha256(Path(__file__).resolve().parents[1] / "src/ect/splits.py")},
                   limitations=["Exact waveform and nominal person component isolation only.",
                                "No claim of independence of specimens, near duplicates or actual acquisition sources."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("sample_counts", "groups", "people")}, indent=2))
    print(f"Manifests saved to {args.output}")


if __name__ == "__main__":
    main()
