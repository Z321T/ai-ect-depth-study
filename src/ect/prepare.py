"""Publish a reproducible raw-unit cache; normalize only with frozen train stats."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import tempfile
import time
import numpy as np
import scipy
from .dataset import ManifestDataset, SPLITS
from .integrity import file_sha256
from .preprocessing import FIR_COEFFICIENTS, downsample_iq, fit_channel_stats, standardize


def prepare_dataset(project_root, manifest_dir="manifests/grouped_v1",
                    output_dir="data/processed/grouped_v1", batch_size=256):
    root = Path(project_root).resolve()
    output = Path(output_dir)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Cache output must be inside project root")
    if output.exists():
        raise FileExistsError(f"Cache already exists: {output}; use a new --output directory")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive integer batch size required")
    started = time.perf_counter()
    dataset = ManifestDataset(root, manifest_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with dataset, tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "cache"
        stage.mkdir()
        files = {}
        for split in SPLITS:
            n = len(dataset.records[split])
            x_path, y_path = stage / f"{split}_x.npy", stage / f"{split}_y.npy"
            x = y = None
            try:
                x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n, 250, 2))
                y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(n,))
                offset = 0
                for raw, labels in dataset.iter_batches(split, batch_size):
                    count = len(labels)
                    x[offset:offset + count] = downsample_iq(raw)
                    y[offset:offset + count] = labels
                    offset += count
                if offset != n:
                    raise ValueError("Incomplete cache")
                x.flush(); y.flush()
            finally:
                for mapped in (x, y):
                    if mapped is not None:
                        mapped._mmap.close()
            for path in (x_path, y_path):
                files[path.name] = file_sha256(path)
        train = np.load(stage / "train_x.npy", allow_pickle=False, mmap_mode="r")
        try:
            stats = fit_channel_stats(train[start:start + batch_size] for start in range(0, len(train), batch_size))
        finally:
            train._mmap.close()
        stats.update(train_cache_sha256=files["train_x.npy"],
                     train_manifest_sha256=dataset.summary["manifest_sha256"]["train.jsonl"])
        normalization = stage / "normalization.json"
        normalization.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
        files[normalization.name] = file_sha256(normalization)
        code_folder = Path(__file__).resolve().parent
        metadata = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": dataset.summary["protocol"], "class_mapping_status": "unverified",
            "summary_path": str(dataset.summary_path.relative_to(root)),
            "summary_sha256": file_sha256(dataset.summary_path),
            "manifest_sha256": dataset.summary["manifest_sha256"], "inputs": dataset.summary["inputs"],
            "sample_counts": dataset.summary["sample_counts"], "class_counts": dataset.summary["class_counts"],
            "files_sha256": files, "cache_units": "raw input units, not standardized",
            "x_dtype": "float32", "y_dtype": "int64", "signal_shape": [250, 2],
            "filter": {"method": "resample_poly", "up": 1, "down": 5, "axis": -2,
                       "input_hz": 2500, "output_hz": 500, "numtaps": 101,
                       "cutoff_half_amplitude_hz": 250, "window": "kaiser", "beta": 5,
                       "padtype": "line", "coefficients": FIR_COEFFICIENTS.tolist()},
            "batch_size": batch_size,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "code_sha256": {f"src/ect/{name}": file_sha256(code_folder / name)
                            for name in ("dataset.py", "preprocessing.py", "prepare.py", "integrity.py")},
            "elapsed_seconds": time.perf_counter() - started,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError("Output appeared during preparation; refusing overwrite")
        stage.rename(output)
    return metadata


class PreparedDataset:
    """Checked read-only cache; batches are independent writable NumPy arrays."""
    def __init__(self, project_root, cache_dir="data/processed/grouped_v1"):
        root = Path(project_root).resolve()
        folder = Path(cache_dir)
        self.folder = folder.resolve() if folder.is_absolute() else (root / folder).resolve()
        self.metadata = json.loads((self.folder / "metadata.json").read_text())
        meta = self.metadata
        required = {f"{split}_{kind}.npy" for split in SPLITS for kind in ("x", "y")}
        required.add("normalization.json")
        if set(meta["files_sha256"]) != required:
            raise ValueError("Prepared cache checksum table must be complete")
        summary_path = root / meta["summary_path"]
        if file_sha256(summary_path) != meta["summary_sha256"]:
            raise ValueError("Split summary checksum mismatch")
        summary = json.loads(summary_path.read_text())
        if summary["manifest_sha256"] != meta["manifest_sha256"] or summary["inputs"] != meta["inputs"]:
            raise ValueError("Cache provenance differs from split summary")
        for name, digest in meta["manifest_sha256"].items():
            if file_sha256(summary_path.parent / name) != digest:
                raise ValueError("Manifest checksum mismatch")
        for info in meta["inputs"].values():
            if file_sha256(root / info["path"]) != info["sha256"]:
                raise ValueError("Raw input checksum mismatch")
        for name, digest in meta["files_sha256"].items():
            if file_sha256(self.folder / name) != digest:
                raise ValueError(f"Prepared cache checksum mismatch: {name}")
        self.stats = json.loads((self.folder / "normalization.json").read_text())
        if self.stats["fitted_split"] != "train" or self.stats["ddof"] != 0:
            raise ValueError("Normalization must be fitted on train with ddof=0")
        if self.stats["train_cache_sha256"] != meta["files_sha256"]["train_x.npy"] or self.stats["train_manifest_sha256"] != meta["manifest_sha256"]["train.jsonl"]:
            raise ValueError("Normalization does not match train provenance")
        self.arrays = {}
        self._mappings = []
        try:
            for split in SPLITS:
                x = np.load(self.folder / f"{split}_x.npy", allow_pickle=False, mmap_mode="r")
                self._mappings.append(x)
                y = np.load(self.folder / f"{split}_y.npy", allow_pickle=False, mmap_mode="r")
                self._mappings.append(y)
                n = summary["sample_counts"][split]
                if x.shape != (n, 250, 2) or x.dtype != np.float32 or y.shape != (n,) or y.dtype != np.int64:
                    raise ValueError("Prepared array shape/dtype differs from protocol")
                self.arrays[split] = (x, y)
            if self.stats["count_per_channel"] != summary["sample_counts"]["train"] * 250:
                raise ValueError("Incorrect normalization training point count")
        except BaseException:
            self.close()
            raise

    def close(self):
        for mapped in self._mappings:
            if not mapped._mmap.closed:
                mapped._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def iter_batches(self, split, batch_size=256, normalize=True):
        if split not in SPLITS or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Known split and positive integer batch size required")
        x, y = self.arrays[split]
        for start in range(0, len(x), batch_size):
            signals = np.array(x[start:start + batch_size], copy=True)
            if normalize:
                signals = standardize(signals, self.stats)
            yield signals, np.array(y[start:start + batch_size], copy=True)
