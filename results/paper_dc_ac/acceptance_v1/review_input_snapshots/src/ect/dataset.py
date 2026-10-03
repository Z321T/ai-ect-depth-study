"""CPU-only access to immutable raw scans using verified grouped manifests."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .audit import waveform_hash
from .integrity import file_sha256

SPLITS = ("train", "validation", "test")
INDEX_AXES = ("person", "angle", "direction", "repeat", "class_index")


class ManifestDataset:
    def __init__(self, project_root, manifest_dir="manifests/grouped_v1"):
        self.arrays = {}
        try:
            self._load(project_root, manifest_dir)
        except BaseException:
            self.close()
            raise

    def _load(self, project_root, manifest_dir):
        self.root = Path(project_root).resolve()
        folder = Path(manifest_dir)
        self.folder = folder.resolve() if folder.is_absolute() else (self.root / folder).resolve()
        self.summary_path = self.folder / "summary.json"
        self.summary = json.loads(self.summary_path.read_text())
        self.arrays = {}
        for source, info in self.summary["inputs"].items():
            path = (self.root / info["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Raw data path must stay within project root")
            if file_sha256(path) != info["sha256"]:
                raise ValueError(f"Raw input checksum mismatch: {source}")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            self.arrays[source] = data
            if data.ndim != 7 or data.shape[-2:] != (1250, 2) or any(n == 0 for n in data.shape):
                raise ValueError("Raw input requires seven axes ending in 1250,2")
            if not np.issubdtype(data.dtype, np.floating):
                raise ValueError("Raw input must use floating-point signals")
        self.records = {}
        person_group = {}
        for group, members in enumerate(self.summary["components"]):
            for person in members:
                if person in person_group:
                    raise ValueError("Repeated person in components")
                person_group[person] = group
        hashes, people, groups, original = {}, {}, {}, set()
        for split in SPLITS:
            path = self.folder / f"{split}.jsonl"
            if file_sha256(path) != self.summary["manifest_sha256"][path.name]:
                raise ValueError(f"Manifest checksum mismatch: {split}")
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if not records or len(records) != self.summary["sample_counts"][split]:
                raise ValueError("Manifest sample count mismatch")
            expected = {int(k): v for k, v in self.summary["class_counts"][split].items()}
            if dict(Counter(row["class_index"] for row in records)) != expected:
                raise ValueError("Manifest class counts mismatch")
            hashes[split], people[split], groups[split] = set(), set(), set()
            for row in records:
                if row["split"] != split:
                    raise ValueError("Record belongs to another split")
                if row["wave_sha256"] in hashes[split]:
                    raise ValueError("Duplicate waveform within manifest")
                hashes[split].add(row["wave_sha256"])
                groups[split].add(row["group_id"])
                if row["representative"]["class_index"] != row["class_index"]:
                    raise ValueError("Representative class differs from record class")
                if row["representative"] not in row["origins"]:
                    raise ValueError("Representative is not a recorded original index")
                for origin in row["origins"]:
                    source = origin["source"]
                    if source not in self.arrays:
                        raise ValueError("Unknown raw source")
                    index = tuple(origin[k] for k in INDEX_AXES)
                    if any(type(i) is not int or i < 0 or i >= n
                           for i, n in zip(index, self.arrays[source].shape[:5])):
                        raise ValueError("Original index outside raw source")
                    if origin["class_index"] != row["class_index"]:
                        raise ValueError("Original class differs from record class")
                    person = f"{source}:{origin['person']}"
                    if person_group.get(person) != row["group_id"]:
                        raise ValueError("Original person does not belong to record group")
                    people[split].add(person)
                    key = (source,) + index
                    if key in original:
                        raise ValueError("Repeated original index")
                    original.add(key)
            self.records[split] = records
        for i, first in enumerate(SPLITS):
            for second in SPLITS[i + 1:]:
                if hashes[first] & hashes[second] or people[first] & people[second] or groups[first] & groups[second]:
                    raise ValueError("Waveform, person or group crosses splits")
        expected_origins = sum(int(np.prod(a.shape[:5])) for a in self.arrays.values())
        if len(original) != expected_origins:
            raise ValueError("Incomplete original index coverage")

    def close(self):
        for data in self.arrays.values():
            if not data._mmap.closed:
                data._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def iter_batches(self, split, batch_size=256):
        if split not in SPLITS or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Known split and positive integer batch size required")
        records = self.records[split]
        for start in range(0, len(records), batch_size):
            selected = records[start:start + batch_size]
            scans = []
            for row in selected:
                origin = row["representative"]
                scan = self.arrays[origin["source"]][tuple(origin[k] for k in INDEX_AXES)]
                if waveform_hash(scan) != row["wave_sha256"]:
                    raise ValueError("Actual waveform hash differs from manifest")
                scans.append(scan)
            yield np.stack(scans), np.array([r["class_index"] for r in selected], dtype=np.int64)
