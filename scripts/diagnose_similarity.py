"""Diagnose selected split near-duplicate candidates in raw-unit cache."""
import argparse
from contextlib import contextmanager, ExitStack
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits
from src.ect.integrity import file_sha256
from src.ect.similarity import center_iq, nearest_neighbors


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


@contextmanager
def _cache(root, folder, query_split="validation", reference_split="train"):
    """PreparedDataset-compatible reader; only open/hash selected split signals.

    Verify the frozen summary, selected manifests and selected cache x/y bytes.
    Raw source identities are inherited from prepare metadata, not re-read here.
    Full PreparedDataset checks all three splits and raw inputs; this reader's
    narrower validation scope is explicitly recorded in summary.json.
    """
    meta = json.loads((folder / "metadata.json").read_text())
    summary_path = root / meta["summary_path"]
    if file_sha256(summary_path) != meta["summary_sha256"]:
        raise ValueError("Split summary checksum mismatch")
    summary = json.loads(summary_path.read_text())
    for key in ("manifest_sha256", "inputs", "sample_counts", "protocol"):
        if summary[key] != meta[key]:
            raise ValueError(f"Cache provenance differs from split summary: {key}")
    required = {f"{split}_{kind}.npy" for split in ("train", "validation", "test") for kind in ("x", "y")}
    required.add("normalization.json")
    if set(meta["files_sha256"]) != required:
        raise ValueError("Prepared cache checksum table must be complete")
    if meta["signal_shape"] != [250, 2] or meta["cache_units"] != "raw input units, not standardized":
        raise ValueError("Raw-unit [250,2] prepared cache required")
    records, arrays = {}, {}
    with ExitStack() as stack:
        for split in (reference_split, query_split):
            manifest = summary_path.parent / f"{split}.jsonl"
            if file_sha256(manifest) != meta["manifest_sha256"][manifest.name]:
                raise ValueError("Manifest checksum mismatch")
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            n = meta["sample_counts"][split]
            if len(rows) != n or any(row["split"] != split for row in rows):
                raise ValueError("Manifest row count/split differs from cache")
            records[split] = rows
            pair = []
            for kind in ("x", "y"):
                path = folder / f"{split}_{kind}.npy"
                if file_sha256(path) != meta["files_sha256"][path.name]:
                    raise ValueError(f"Prepared cache checksum mismatch: {path.name}")
                mapped = np.load(path, allow_pickle=False, mmap_mode="r")
                stack.callback(mapped._mmap.close)
                expected_shape = (n, 250, 2) if kind == "x" else (n,)
                expected_dtype = np.float32 if kind == "x" else np.int64
                if mapped.shape != expected_shape or mapped.dtype != expected_dtype:
                    raise ValueError("Prepared array shape/dtype differs from protocol")
                pair.append(mapped)
            if not np.array_equal(pair[1], [row["class_index"] for row in rows]):
                raise ValueError("Cached labels differ from frozen manifest")
            arrays[split] = pair
        yield meta, records, arrays


def _statistics(distances, indices, counts, labels, reference_labels):
    finite = distances[np.isfinite(distances)]
    levels = (0, .01, .05, .25, .5, .75, .95, .99, 1)
    candidates = counts > 0
    return {
        "applicable_query_count": int(np.count_nonzero(indices >= 0)),
        "not_applicable_query_count": int(np.count_nonzero(indices < 0)),
        "nearest_distance_quantiles": {str(level): float(np.quantile(finite, level)) if len(finite) else None for level in levels},
        "candidate_query_count": int(np.count_nonzero(candidates)),
        "candidate_pair_count": int(np.sum(counts)),
        "nearest_candidate_same_class_count": int(np.count_nonzero(labels[candidates] == reference_labels[indices[candidates]])),
    }


def _write_csv(path, result, records, meta, query_split="validation", reference_split="train"):
    base = ["query_split", "reference_split", "query_row", "query_manifest_sha256", "query_wave_sha256", "query_class_index", "query_representative", "query_origins", "query_ac_norm"]
    suffixes = ["reference_split", "reference_row", "distance", "applicable", "candidate", "candidate_pair_count", "reference_manifest_sha256", "reference_wave_sha256", "reference_class_index", "same_class", "reference_representative", "reference_origins"]
    fields = base + [f"{kind}_{name}" for kind in ("raw", "shape") for name in suffixes]
    compact = lambda obj: json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, query in enumerate(records[query_split]):
            row = dict(query_split=query_split, reference_split=reference_split,
                       query_row=i, query_manifest_sha256=meta["manifest_sha256"][f"{query_split}.jsonl"],
                       query_wave_sha256=query["wave_sha256"], query_class_index=query["class_index"],
                       query_representative=compact(query["representative"]), query_origins=compact(query["origins"]),
                       query_ac_norm=float(result.query_ac_norm[i]))
            for kind in ("raw", "shape"):
                index = int(getattr(result, f"{kind}_index")[i])
                distance = getattr(result, f"{kind}_distance")[i]
                count = int(getattr(result, f"{kind}_candidate_count")[i])
                details = dict(reference_split=reference_split, reference_row=index, distance=float(distance) if index >= 0 else "",
                               applicable=index >= 0, candidate=count > 0, candidate_pair_count=count,
                               reference_manifest_sha256=meta["manifest_sha256"][f"{reference_split}.jsonl"])
                if index >= 0:
                    reference = records[reference_split][index]
                    details.update(reference_wave_sha256=reference["wave_sha256"],
                                   reference_class_index=reference["class_index"],
                                   same_class=query["class_index"] == reference["class_index"],
                                   reference_representative=compact(reference["representative"]),
                                   reference_origins=compact(reference["origins"]))
                row.update({f"{kind}_{key}": value for key, value in details.items()})
            writer.writerow(row)


def _plot(path, result, queries, references, query_split="validation", reference_split="train"):
    if not (np.any(result.raw_candidate_count) or np.any(result.shape_candidate_count)):
        return False
    # Keep matplotlib's incidental configuration files outside the project.
    with tempfile.TemporaryDirectory(prefix="ect-similarity-mpl-") as config:
        os.environ.setdefault("MPLCONFIGDIR", config)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
        for col, (kind, threshold) in enumerate((("raw", .001), ("shape", .01))):
            distances = getattr(result, f"{kind}_distance")
            finite = distances[np.isfinite(distances)]
            positive = finite[finite > 0]
            axis = axes[0, col]
            if len(positive):
                low, high = min(float(positive.min()), threshold) * .8, max(float(positive.max()), threshold) * 1.2
                bins = np.geomspace(low, high, 40)
                axis.hist(positive, bins=bins, color="#386c82")
                axis.set_xscale("log")
            axis.axvline(threshold, color="#b74731", linestyle="--", label=f"candidate <= {threshold:g}")
            axis.set_title(f"{kind}: {query_split} nearest {reference_split} distance (zeros: {np.count_nonzero(finite == 0)})")
            axis.set_ylabel("Queries")
            axis.legend(fontsize=8)
            candidate_rows = np.flatnonzero(getattr(result, f"{kind}_candidate_count") > 0)
            axis = axes[1, col]
            if not len(candidate_rows):
                axis.text(.5, .5, f"No {kind} threshold candidates", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            qi = int(candidate_rows[np.argmin(distances[candidate_rows])])
            ri = int(getattr(result, f"{kind}_index")[qi])
            q = center_iq(queries[qi:qi + 1])[0]
            r = center_iq(references[ri:ri + 1])[0]
            if kind == "shape":
                q /= result.query_ac_norm[qi]
                r /= result.reference_ac_norm[ri]
            for channel, color in enumerate(("#386c82", "#b74731")):
                axis.plot(q[:, channel], color=color, label=f"{query_split} ch{channel}")
                axis.plot(r[:, channel], color=color, linestyle="--", alpha=.7, label=f"{reference_split} ch{channel}")
            axis.set_title(f"{kind} candidate: {query_split} row {qi} / {reference_split} row {ri}; d={distances[qi]:.6g}")
            axis.set_xlabel("Time index (no alignment)")
            axis.set_ylabel("Joint unit AC" if kind == "shape" else "AC, raw units (display only)")
            axis.legend(fontsize=8)
        fig.suptitle(f"{reference_split} / {query_split} candidates: similarity alone does not establish leakage")
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/grouped_v1"))
    parser.add_argument("--query-split", choices=("validation", "test"), default="validation",
                        help="Query split (default: validation)")
    parser.add_argument("--reference-split", choices=("train", "validation"), default="train",
                        help="Reference split (default: train); validation requires test queries")
    parser.add_argument("--output", type=Path,
                        help="Output directory (default: results/similarity/{reference}_{query}_v1)")
    parser.add_argument("--query-block", type=int, default=256)
    parser.add_argument("--reference-block", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if (args.query_split, args.reference_split) not in (("validation", "train"), ("test", "train"), ("test", "validation")):
        parser.error("Allowed query/reference split pairs: validation/train, test/train, test/validation")
    selected_splits = (args.reference_split, args.query_split)
    unread_split = next(split for split in ("train", "validation", "test") if split not in selected_splits)
    root = args.project_root.resolve()
    folder = (root / args.cache).resolve()
    output_path = args.output if args.output is not None else Path(f"results/similarity/{args.reference_split}_{args.query_split}_v1")
    output = (root / output_path).resolve()
    if not output.is_relative_to(root / "results/similarity"):
        parser.error("Output must be within project results/similarity/")
    if min(args.query_block, args.reference_block, args.threads) < 1:
        parser.error("Block sizes and threads must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        with _cache(root, folder, args.query_split, args.reference_split) as (meta, records, arrays), threadpool_limits(limits=args.threads):
            verified = time.perf_counter() - started
            queries, qlabels = arrays[args.query_split]
            references, rlabels = arrays[args.reference_split]
            def progress(done, total, seconds):
                print(f"Nearest neighbors {done}/{total}; search {seconds:.2f}s", file=sys.stderr, flush=True)
            result = nearest_neighbors(queries, references, query_block=args.query_block,
                                       reference_block=args.reference_block, progress=progress)
            _write_csv(output / "nearest.csv", result, records, meta, args.query_split, args.reference_split)
            plotted = _plot(output / "candidates.png", result, queries, references, args.query_split, args.reference_split)
            report = {
                "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "query_split": args.query_split, "reference_split": args.reference_split,
                "signal_splits_read": list(selected_splits),
                "query_count": len(queries), "reference_count": len(references),
                "exhaustive_pair_count": len(queries) * len(references),
                "thresholds": {"raw": .001, "shape": .01},
                "distances": {"raw": "||q-r||_2 / ||q-mean_t(q)||_2, raw input units, direct differences",
                              "shape": "per-channel stable float64 time centering, joint unit L2, direct candidate differences"},
                "allowed_invariances": {"shape": ["per-channel DC offset", "joint positive scale"], "raw": []},
                "zero_ac_policy": "Zero-AC queries: both metrics not applicable, row -1 and empty CSV distance; zero-AC references excluded only from shape",
                "tie_policy": "Smallest zero-based original reference row for exact computed distance ties",
                "candidate_count_definition": "All pairs at distance <= threshold; query counts count at least one such reference. CSV class agreement is for the nearest pair only, computed after label-free search.",
                "blocks": {"query": args.query_block, "reference": args.reference_block},
                "raw": _statistics(result.raw_distance, result.raw_index, result.raw_candidate_count, qlabels, rlabels),
                "shape": _statistics(result.shape_distance, result.shape_index, result.shape_candidate_count, qlabels, rlabels),
                "zero_ac_reference_count": int(np.count_nonzero(result.reference_ac_norm == 0)),
                "cache": str(folder), "cache_metadata_sha256": file_sha256(folder / "metadata.json"),
                "cache_files_sha256": {f"{split}_{kind}.npy": meta["files_sha256"][f"{split}_{kind}.npy"] for split in selected_splits for kind in ("x", "y")},
                "summary_sha256": meta["summary_sha256"],
                "manifest_sha256": {f"{split}.jsonl": meta["manifest_sha256"][f"{split}.jsonl"] for split in selected_splits},
                "raw_inputs_declared_by_cache": meta["inputs"], "class_mapping_status": meta["class_mapping_status"],
                "integrity_scope": f"Verified summary, {args.reference_split}/{args.query_split} manifests and cache x/y bytes and label ordering. Complete prepared-cache checksum table required. No {unread_split} signal or raw input file read. Raw source hashes inherited from bound prepare metadata.",
                "code_sha256": {name: file_sha256(PROJECT_ROOT / name) for name in ("src/ect/similarity.py", "scripts/diagnose_similarity.py", "src/ect/integrity.py")},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                "platform": platform.platform(), "cpu_count": os.cpu_count(), "device": "cpu",
                                "requested_threads": args.threads, "threadpools": threadpool_info()},
                "timing_seconds": {"integrity_and_load": verified, "search": result.elapsed_seconds,
                                   "total": time.perf_counter() - started},
                "artifacts_sha256": {path.name: file_sha256(path) for path in (output / "nearest.csv", output / "candidates.png") if path.exists()},
                "figure_created": plotted,
                "interpretation": f"{args.query_split}-to-{args.reference_split} data similarity audit only. Shape proximity is a candidate, not a leakage determination. No split changes, class semantics, test classification metrics or source-independence claims.",
            }
            _json(output / "summary.json", report)
        print(json.dumps({"output": str(output), "raw": report["raw"], "shape": report["shape"], "timing_seconds": report["timing_seconds"]}, indent=2))
    except Exception as exc:
        _json(output / "failure.json", {"exception": type(exc).__name__, "message": str(exc), "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
