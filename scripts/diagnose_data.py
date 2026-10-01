"""Verify prepared arrays and export training-only signal and filter diagnostics."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect_matplotlib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz
from src.ect.audit import waveform_hash
from src.ect.integrity import file_sha256
from src.ect.prepare import PreparedDataset
from src.ect.preprocessing import FIR_COEFFICIENTS, fit_channel_stats, standardize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache", type=Path, default=Path("data/processed/grouped_v1"))
    parser.add_argument("--output", type=Path, default=Path("results/preprocessing"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    store = PreparedDataset(root, args.cache)
    report = {"metadata_sha256": file_sha256(store.folder / "metadata.json"),
              "script_sha256": file_sha256(__file__), "splits": {}, "processed_overlap": {}}
    summary_path = root / store.metadata["summary_path"]
    hashes = {}
    for split, (x, y) in store.arrays.items():
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite cached data")
        actual_counts = {str(k): v for k, v in sorted(Counter(map(int, y)).items())}
        if actual_counts != store.metadata["class_counts"][split]:
            raise ValueError("Cache labels differ from manifest class counts")
        records = [json.loads(line) for line in (summary_path.parent / f"{split}.jsonl").read_text().splitlines()]
        if not np.array_equal(y, [r["class_index"] for r in records]):
            raise ValueError("Cache label order differs from manifest")
        hashes[split] = {waveform_hash(scan) for scan in x}
        report["splits"][split] = {"shape": list(x.shape), "dtype": str(x.dtype),
                                    "finite": True, "class_counts": actual_counts,
                                    "unique_processed_waveforms": len(hashes[split])}
        if len(hashes[split]) != len(x):
            raise ValueError("Duplicate waveforms introduced by preprocessing")
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = len(hashes[first] & hashes[second])
        report["processed_overlap"][f"{first}/{second}"] = overlap
        if overlap:
            raise ValueError("Processed waveforms cross splits")
    zstats = fit_channel_stats(batch for batch, _ in store.iter_batches("train"))
    report["train_normalized"] = {k: zstats[k] for k in ("mean", "std", "count_per_channel")}
    if not np.allclose(zstats["mean"], 0, atol=1e-6) or not np.allclose(zstats["std"], 1, atol=1e-6):
        raise ValueError("Training normalization failed verification")

    x, y = store.arrays["train"]
    class_indices = sorted(map(int, np.unique(y)))
    rng = np.random.default_rng(20261001)
    chosen = {cls: int(rng.choice(np.flatnonzero(y == cls))) for cls in class_indices}
    records = [json.loads(line) for line in (summary_path.parent / "train.jsonl").read_text().splitlines()]
    report["training_examples"] = [{"class_index": cls, "row_index": i,
                                     "raw_wave_sha256": records[i]["wave_sha256"],
                                     "representative": records[i]["representative"]} for cls, i in chosen.items()]
    rows = (len(class_indices) + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(16, rows * 2.6), squeeze=False)
    for axis, cls in zip(axes.flat, class_indices):
        scan = x[chosen[cls]]
        axis.plot(np.arange(250) / 500, scan[:, 0], label="I", linewidth=0.8)
        axis.plot(np.arange(250) / 500, scan[:, 1], label="Q", linewidth=0.8)
        axis.set_title(f"Class index {cls}", fontsize=10)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.set_xlabel("Time (s)"); axis.set_ylabel("Dataset units")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Training signals after anti-alias downsampling (class semantics unverified)")
    fig.tight_layout(); fig.savefig(output / "training_signals.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(rows, 5, figsize=(16, rows * 2.6), squeeze=False)
    for axis, cls in zip(axes.flat, class_indices):
        scan = x[chosen[cls]].astype(np.float64)
        centered = scan - scan.mean(axis=0)
        axis.plot(np.arange(250) / 500, centered[:, 0], label="I", linewidth=0.8)
        axis.plot(np.arange(250) / 500, centered[:, 1], label="Q", linewidth=0.8)
        axis.set_title(f"Class index {cls}", fontsize=10)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.set_xlabel("Time (s)"); axis.set_ylabel("AC (dataset units)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Same training examples, per-channel mean removed for display only; model inputs unchanged")
    fig.tight_layout(); fig.savefig(output / "training_signals_ac.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(rows, 5, figsize=(16, rows * 2.6), squeeze=False)
    for axis, cls in zip(axes.flat, class_indices):
        ids = np.flatnonzero(y == cls)
        for i in rng.choice(ids, size=min(8, len(ids)), replace=False):
            scan = standardize(x[int(i)], store.stats)
            axis.plot(scan[:, 0], scan[:, 1], alpha=0.55, linewidth=0.8)
        axis.set_title(f"Class index {cls}", fontsize=10)
        axis.set_xlabel("I (train z-score)"); axis.set_ylabel("Q (train z-score)")
    fig.suptitle("Training I/Q trajectories: 8 examples per index, fixed seed")
    fig.tight_layout(); fig.savefig(output / "training_iq.png", dpi=140); plt.close(fig)

    frequency, response = freqz(FIR_COEFFICIENTS, worN=4096, fs=2500)
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(frequency, 20 * np.log10(np.maximum(np.abs(response), 1e-8)))
    axis.axvline(250, color="tab:red", linestyle="--", label="250 Hz half-amplitude point")
    axis.set(xlim=(0, 1250), ylim=(-110, 5), xlabel="Frequency (Hz)", ylabel="Gain (dB)",
             title="101-tap Kaiser FIR, beta=5; output sample rate 500 Hz")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output / "filter_response.png", dpi=160); fig.savefig(output / "filter_response.svg"); plt.close(fig)
    svg = output / "filter_response.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    ac_power = np.mean((x.astype(np.float64) - x.mean(axis=1, keepdims=True, dtype=np.float64)) ** 2, axis=(1, 2))
    total_power = np.mean(x.astype(np.float64) ** 2, axis=(1, 2))
    positive = ac_power > 0
    report["train_ac_power"] = {"zero_power_samples": int((~positive).sum()),
                                  "median_ac_power": float(np.median(ac_power)),
                                  "median_total_to_ac_db": float(np.median(10 * np.log10(total_power[positive] / ac_power[positive])))}
    (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "normalization.json").write_text(json.dumps(store.stats, indent=2, allow_nan=False) + "\n")
    store.close()
    print(json.dumps({"normalized_train": report["train_normalized"],
                      "processed_overlap": report["processed_overlap"],
                      "train_ac_power": report["train_ac_power"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
