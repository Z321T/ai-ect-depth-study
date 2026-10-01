"""Tiny real files for integrity and train-only preprocessing integration tests."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.ect.audit import waveform_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_dataset(root):
    root = Path(root)
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    data = np.empty((3, 1, 1, 1, 2, 1250, 2), dtype=np.float32)
    time = np.arange(1250) / 2500
    for person, offset in enumerate([10, 1000, 2000]):
        for cls in range(2):
            data[person, 0, 0, 0, cls, :, 0] = offset + cls + np.sin(2 * np.pi * 50 * time)
            data[person, 0, 0, 0, cls, :, 1] = offset * 2 + cls + np.cos(2 * np.pi * 50 * time)
    path = raw / "fixture.npy"
    np.save(path, data)
    folder = root / "manifests/fixture"
    folder.mkdir(parents=True)
    summary = {"schema_version": 1, "protocol": "fixture", "class_mapping_status": "unverified",
               "inputs": {"train": {"path": "data/raw/fixture.npy", "sha256": sha(path)}},
               "manifest_sha256": {}, "sample_counts": {}, "class_counts": {},
               "components": [[f"train:{p}"] for p in range(3)]}
    for person, split in enumerate(["train", "validation", "test"]):
        records = []
        for cls in range(2):
            origin = dict(source="train", person=person, angle=0, direction=0, repeat=0, class_index=cls)
            records.append(dict(split=split, group_id=person, class_index=cls,
                                wave_sha256=waveform_hash(data[person, 0, 0, 0, cls]),
                                representative=origin, origins=[origin]))
        file = folder / f"{split}.jsonl"
        file.write_text("".join(json.dumps(row) + "\n" for row in records))
        summary["manifest_sha256"][file.name] = sha(file)
        summary["sample_counts"][split] = 2
        summary["class_counts"][split] = {"0": 1, "1": 1}
    (folder / "summary.json").write_text(json.dumps(summary))
    return data, folder


def change_manifest(folder, split, edit):
    path = folder / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    edit(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest_sha256"][path.name] = sha(path)
    summary_path.write_text(json.dumps(summary))
