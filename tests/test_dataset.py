import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from fixture_data import make_dataset, change_manifest


class DatasetTests(unittest.TestCase):
    def store(self, root, folder):
        try:
            from src.ect.dataset import ManifestDataset
        except ImportError:
            self.fail("ManifestDataset is missing")
        return ManifestDataset(root, folder)

    def test_loads_exact_source_indices_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, folder = make_dataset(tmp)
            store = self.store(tmp, folder)
            batches = list(store.iter_batches("validation", batch_size=1))
            self.assertEqual(len(batches), 2)
            np.testing.assert_array_equal(batches[1][0][0], data[1, 0, 0, 0, 1])
            np.testing.assert_array_equal(batches[1][1], [1])
            batches[0][0][0] = 0
            self.assertNotEqual(float(next(store.iter_batches("validation"))[0][0, 0, 0]), 0)
            arrays = list(store.arrays.values())
            store.close()
            self.assertTrue(all(array._mmap.closed for array in arrays))

    def test_modified_raw_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            with Path(tmp, "data/raw/fixture.npy").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.store(tmp, folder)

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            with (folder / "train.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.store(tmp, folder)

    def test_actual_waveform_must_match_claim_even_if_file_checksum_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            change_manifest(folder, "train", lambda rows: rows[0].update(wave_sha256="0" * 64))
            with self.assertRaisesRegex(ValueError, "waveform hash"):
                store = self.store(tmp, folder)
                with store:
                    list(store.iter_batches("train"))

    def test_representative_class_cannot_disagree_with_record_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            change_manifest(folder, "train", lambda rows: rows[0]["representative"].update(class_index=1))
            with self.assertRaisesRegex(ValueError, "class"):
                self.store(tmp, folder)

    def test_manifest_validation_failure_closes_opened_raw_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            (folder / "train.jsonl").write_text("changed")
            original = np.load
            opened = []
            def capture(*args, **kwargs):
                mapped = original(*args, **kwargs)
                opened.append(mapped)
                return mapped
            with patch.object(np, "load", side_effect=capture):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    self.store(tmp, folder)
            self.assertTrue(opened)
            self.assertTrue(all(array._mmap.closed for array in opened))
