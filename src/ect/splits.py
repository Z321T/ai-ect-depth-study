"""Build deduplicated splits from complete person connected components."""
from collections import Counter


def build_grouped_manifest(rows, components, assignment):
    if set(assignment) != set(range(len(components))):
        raise ValueError("Every component must have exactly one split assignment")
    names = ("train", "validation", "test")
    if not set(assignment.values()) <= set(names):
        raise ValueError("Unknown split name")
    person_group = {}
    for group, members in enumerate(components):
        for person in members:
            if person in person_group:
                raise ValueError("Person belongs to multiple components")
            person_group[person] = group
    if {r["person_id"] for r in rows} != set(person_group):
        raise ValueError("Manifest rows and component people do not match")

    unique = {}
    original_indices = set()
    for row in rows:
        if row["person_id"] != f"{row['source']}:{row['person']}":
            raise ValueError("Person identity does not match original index")
        original_index = tuple(row[key] for key in ("source", "person", "angle", "direction", "repeat", "class_index"))
        if original_index in original_indices:
            raise ValueError("Repeated original index")
        original_indices.add(original_index)
        group = person_group[row["person_id"]]
        split = assignment[group]
        digest = row["wave_sha256"]
        origin = {key: row[key] for key in ("source", "person", "angle", "direction", "repeat", "class_index")}
        if digest in unique:
            previous = unique[digest]
            if previous["split"] != split:
                raise ValueError("Identical waveform crosses splits")
            if previous["class_index"] != row["class_index"]:
                raise ValueError("Identical waveform has a class conflict")
            if previous["group_id"] != group:
                raise ValueError("Shared waveform was not linked in person components")
            previous["origins"].append(origin)
        else:
            unique[digest] = {"split": split, "group_id": group,
                              "class_index": row["class_index"], "wave_sha256": digest,
                              "representative": origin, "origins": [origin]}
    manifest = {name: [] for name in names}
    for digest in sorted(unique):
        record = unique[digest]
        manifest[record["split"]].append(record)
    summary = {
        "sample_counts": {name: len(records) for name, records in manifest.items()},
        "class_counts": {name: dict(sorted(Counter(r["class_index"] for r in records).items()))
                         for name, records in manifest.items()},
        "groups": {name: [i for i in assignment if assignment[i] == name] for name in names},
        "people": {name: sorted(p for p, i in person_group.items() if assignment[i] == name) for name in names},
        "exact_waveform_overlap": 0,
        "person_overlap": 0,
    }
    return manifest, summary
