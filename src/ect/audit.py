"""Audit seven-axis MDDECT arrays without assigning unverified class semantics."""
from collections import Counter, defaultdict
import hashlib
import numpy as np


def waveform_hash(scan):
    """Byte identity including shape and dtype; not a near-duplicate measure."""
    h = hashlib.sha256()
    h.update(f"{scan.shape}:{scan.dtype.str}:".encode("ascii"))
    h.update(scan.tobytes(order="C"))
    return h.hexdigest()


def validate_sample_rows(rows, arrays):
    """Prove full original-index coverage and rehash each actual source scan."""
    seen = set()
    counts = Counter()
    axes = ("person", "angle", "direction", "repeat", "class_index")
    for row in rows:
        source = row["source"]
        if source not in arrays:
            raise ValueError("Unknown source file")
        index = tuple(row[key] for key in axes)
        if row["person_id"] != f"{source}:{index[0]}":
            raise ValueError("Person identity does not match original index")
        if any(not isinstance(i, int) or i < 0 or i >= n
               for i, n in zip(index, arrays[source].shape[:5])):
            raise ValueError("Original index outside source array")
        key = (source,) + index
        if key in seen:
            raise ValueError("Repeated original index")
        seen.add(key)
        counts[source] += 1
        if waveform_hash(arrays[source][index]) != row["wave_sha256"]:
            raise ValueError("Recorded waveform hash does not match original index")
    for source, data in arrays.items():
        if counts[source] != int(np.prod(data.shape[:5])):
            raise ValueError("Incomplete original-index coverage")


def audit_arrays(arrays):
    """Return JSON-compatible statistics and a row per original scan.

    First five axes are person, angle, direction, repeat, class index.
    Final two axes are time and the two I/Q channels.
    """
    rows = []
    by_hash = defaultdict(list)
    stats = {}
    people = set()
    for source, data in arrays.items():
        if data.ndim != 7 or data.shape[-1] != 2 or any(n == 0 for n in data.shape):
            raise ValueError("MDDECT requires seven nonempty axes and two channels")
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError("MDDECT values must be floating point")
        count = int(np.prod(data.shape[:5]))
        finite_count = zero_count = constant_count = 0
        sums = np.zeros(2, dtype=np.float64)
        squares = np.zeros(2, dtype=np.float64)
        channel_count = np.zeros(2, dtype=np.int64)
        minimum = np.full(2, np.inf)
        maximum = np.full(2, -np.inf)
        for person in range(data.shape[0]):
            people.add(f"{source}:{person}")
            block = data[person].reshape(-1, data.shape[-2], 2)
            finite = np.isfinite(block)
            finite_count += int(finite.sum())
            zero_count += int(np.count_nonzero(np.all(block == 0, axis=(1, 2))))
            constant_count += int(np.count_nonzero(np.all(block == block[:, :1], axis=(1, 2))))
            for channel in range(2):
                values = block[..., channel][finite[..., channel]].astype(np.float64)
                channel_count[channel] += values.size
                if values.size:
                    sums[channel] += values.sum()
                    squares[channel] += np.dot(values, values)
                    minimum[channel] = min(minimum[channel], values.min())
                    maximum[channel] = max(maximum[channel], values.max())
            for local in np.ndindex(data.shape[1:5]):
                index = (person,) + local
                row = dict(zip(("person", "angle", "direction", "repeat", "class_index"), index))
                row.update(source=source, person_id=f"{source}:{person}",
                           wave_sha256=waveform_hash(data[index]))
                rows.append(row)
                by_hash[row["wave_sha256"]].append(row)
        means = np.divide(sums, channel_count, out=np.zeros(2), where=channel_count > 0)
        variance = np.divide(squares, channel_count, out=np.zeros(2), where=channel_count > 0) - means ** 2
        hashes = {r["wave_sha256"] for r in rows if r["source"] == source}
        stats[source] = {
            "shape": list(data.shape), "dtype": str(data.dtype), "sample_count": count,
            "samples_per_class": count // data.shape[4],
            "unique_waveforms": len(hashes), "duplicate_excess": count - len(hashes),
            "nonfinite_values": int(data.size) - finite_count,
            "zero_samples": zero_count, "constant_samples": constant_count,
            "channel_mean_finite": [float(means[c]) if channel_count[c] else None for c in range(2)],
            "channel_std_finite": [float(np.sqrt(max(0, variance[c]))) if channel_count[c] else None for c in range(2)],
            "channel_min_finite": [float(minimum[c]) if channel_count[c] else None for c in range(2)],
            "channel_max_finite": [float(maximum[c]) if channel_count[c] else None for c in range(2)],
        }

    parent = {p: p for p in people}

    def root(person):
        while person != parent[person]:
            parent[person] = parent[parent[person]]
            person = parent[person]
        return person

    conflicts = cross_person = overlap_unique = 0
    overlap_counts = Counter()
    duplicate_examples = []
    for digest, group in by_hash.items():
        members = sorted({r["person_id"] for r in group})
        for member in members[1:]:
            parent[root(member)] = root(members[0])
        if len(group) > 1:
            conflicts += len({r["class_index"] for r in group}) > 1
            cross_person += len(members) > 1
            if len(duplicate_examples) < 10:
                duplicate_examples.append({"wave_sha256": digest, "occurrences": group})
        if len({r["source"] for r in group}) > 1:
            overlap_unique += 1
            overlap_counts.update(r["source"] for r in group)
    components = defaultdict(list)
    for person in sorted(people):
        components[root(person)].append(person)
    components = sorted(components.values(), key=lambda c: c[0])
    for row in rows:
        row["multiplicity"] = len(by_hash[row["wave_sha256"]])
    report = {
        "schema_version": 2,
        "hash_definition": "SHA256(shape:dtype: + full C-order scan bytes)",
        "datasets": stats,
        "total_unique_waveforms": len(by_hash),
        "cross_class_duplicate_groups": int(conflicts),
        "cross_person_duplicate_groups": int(cross_person),
        "overlap": {"unique_waveforms": overlap_unique,
                    **{f"{name}_samples": overlap_counts[name] for name in arrays}},
        "person_components": components,
        "duplicate_examples": duplicate_examples,
        "limitations": ["Exact byte identity only; near duplicates and collection provenance not established.",
                        "Class indices have no verified semantic mapping in this audit."],
    }
    return report, rows
