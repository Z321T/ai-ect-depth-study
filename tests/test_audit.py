"""Protect the evidence used to determine whether evaluation splits are independent."""
import unittest
import numpy as np


class AuditTests(unittest.TestCase):
    def audit(self, arrays):
        try:
            from src.ect.audit import audit_arrays
        except ImportError:
            self.fail("audit_arrays has not been implemented")
        return audit_arrays(arrays)

    def test_duplicate_counts_include_all_test_occurrences(self):
        train = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 2, 1, 2, 2)
        test = np.repeat(train[:, :, :, :1], 2, axis=3)
        report, rows = self.audit({"train": train, "test": test})
        self.assertEqual(report["datasets"]["train"]["unique_waveforms"], 2)
        self.assertEqual(report["datasets"]["test"]["duplicate_excess"], 1)
        self.assertEqual(report["overlap"]["unique_waveforms"], 1)
        self.assertEqual(report["overlap"]["test_samples"], 2)
        self.assertEqual(len(rows), 4)

    def test_shared_waveforms_join_people_transitively(self):
        data = np.arange(24, dtype=np.float32).reshape(3, 1, 1, 2, 1, 2, 2)
        data[1, 0, 0, 0] = data[0, 0, 0, 0]
        data[2, 0, 0, 0] = data[1, 0, 0, 1]
        report, _ = self.audit({"train": data})
        self.assertEqual(report["person_components"], [["train:0", "train:1", "train:2"]])

    def test_conflicting_class_indices_are_reported(self):
        data = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 1, 2, 2, 2)
        data[0, 0, 0, 0, 1] = data[0, 0, 0, 0, 0]
        report, _ = self.audit({"train": data})
        self.assertEqual(report["cross_class_duplicate_groups"], 1)

    def test_nonfinite_values_and_constant_scans_are_visible(self):
        data = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 2, 1, 2, 2)
        data[0, 0, 0, 0, 0] = 0
        data[0, 0, 0, 1, 0, 0, 0] = np.nan
        report, _ = self.audit({"train": data})
        stats = report["datasets"]["train"]
        self.assertEqual(stats["nonfinite_values"], 1)
        self.assertEqual(stats["zero_samples"], 1)
        self.assertEqual(stats["constant_samples"], 1)

    def test_incorrect_axis_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "seven"):
            self.audit({"train": np.zeros((2, 3), dtype=np.float32)})

    def test_hash_cannot_claim_another_loading_index(self):
        from src.ect import audit
        self.assertTrue(hasattr(audit, "validate_sample_rows"), "Sample validation missing")
        data = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 2, 1, 2, 2)
        _, rows = self.audit({"train": data})
        rows[0]["wave_sha256"] = rows[1]["wave_sha256"]
        with self.assertRaisesRegex(ValueError, "waveform hash"):
            audit.validate_sample_rows(rows, {"train": data})

    def test_equal_row_count_does_not_prove_complete_origins(self):
        from src.ect import audit
        self.assertTrue(hasattr(audit, "validate_sample_rows"), "Sample validation missing")
        data = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 2, 1, 2, 2)
        _, rows = self.audit({"train": data})
        with self.assertRaisesRegex(ValueError, "Repeated original index"):
            audit.validate_sample_rows([rows[0], rows[0].copy()], {"train": data})


if __name__ == "__main__":
    unittest.main()
