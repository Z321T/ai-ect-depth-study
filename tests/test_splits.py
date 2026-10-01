import unittest


class SplitTests(unittest.TestCase):
    def build(self, rows, components, assignment):
        try:
            from src.ect.splits import build_grouped_manifest
        except ImportError:
            self.fail("build_grouped_manifest has not been implemented")
        return build_grouped_manifest(rows, components, assignment)

    def row(self, person, digest, cls=0):
        source, index = person.split(":")
        return dict(person_id=person, source=source, person=int(index), angle=0,
                    direction=0, repeat=0, class_index=cls, wave_sha256=digest)

    def test_duplicates_within_group_are_kept_once_with_provenance(self):
        rows = [self.row("train:0", "a"), self.row("train:1", "a"),
                self.row("train:2", "b"), self.row("test:0", "c")]
        manifest, summary = self.build(rows, [["train:0", "train:1"], ["train:2"], ["test:0"]],
                                       {0: "train", 1: "validation", 2: "test"})
        self.assertEqual(len(manifest["train"]), 1)
        self.assertEqual(len(manifest["train"][0]["origins"]), 2)
        self.assertEqual(summary["sample_counts"], {"train": 1, "validation": 1, "test": 1})

    def test_incorrect_components_cannot_allow_leaking_waveform(self):
        rows = [self.row("train:0", "a"), self.row("test:0", "a")]
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            self.build(rows, [["train:0"], ["test:0"]], {0: "train", 1: "test"})

    def test_conflicting_class_indices_cannot_be_silently_deduplicated(self):
        rows = [self.row("train:0", "a", 0), self.row("train:1", "a", 1)]
        with self.assertRaisesRegex(ValueError, "class conflict"):
            self.build(rows, [["train:0", "train:1"]], {0: "train"})

    def test_unassigned_components_cannot_drop_samples(self):
        with self.assertRaisesRegex(ValueError, "Every component"):
            self.build([self.row("train:0", "a")], [["train:0"]], {})

    def test_person_identity_must_match_loading_index(self):
        row = self.row("train:0", "a")
        row["person"] = 1
        with self.assertRaisesRegex(ValueError, "identity"):
            self.build([row], [["train:0"]], {0: "train"})

    def test_repeating_origin_is_not_counted_as_full_provenance(self):
        row = self.row("train:0", "a")
        with self.assertRaisesRegex(ValueError, "Repeated original index"):
            self.build([row, row.copy()], [["train:0"]], {0: "train"})


if __name__ == "__main__":
    unittest.main()
