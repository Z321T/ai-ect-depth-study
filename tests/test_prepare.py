import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from fixture_data import make_dataset
from src.ect.preprocessing import downsample_iq


class PrepareTests(unittest.TestCase):
    def module(self):
        try:
            from src.ect import prepare
        except ImportError:
            self.fail("Preparation pipeline is missing")
        return prepare

    def test_only_training_signals_determine_stats_and_cache_keeps_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, folder = make_dataset(tmp)
            output = Path(tmp, "cache")
            result = self.module().prepare_dataset(tmp, folder, output, batch_size=1)
            stats = json.loads((output / "normalization.json").read_text())
            expected = downsample_iq(data[0].reshape(2, 1250, 2))
            np.testing.assert_allclose(stats["mean"], expected.mean(axis=(0, 1), dtype=np.float64), atol=1e-7)
            np.testing.assert_allclose(stats["std"], expected.std(axis=(0, 1), dtype=np.float64), rtol=1e-7)
            self.assertEqual(stats["fitted_split"], "train")
            self.assertEqual(result["sample_counts"], {"train": 2, "validation": 2, "test": 2})
            np.testing.assert_array_equal(np.load(output / "validation_y.npy"), [0, 1])
            np.testing.assert_array_equal(np.load(output / "train_x.npy"), expected)

    def test_existing_cache_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            output = Path(tmp, "cache"); output.mkdir()
            with self.assertRaises(FileExistsError):
                self.module().prepare_dataset(tmp, folder, output)

    def test_prepared_reader_uses_train_stats_and_detects_changed_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            module = self.module()
            output = Path(tmp, "cache")
            module.prepare_dataset(tmp, folder, output)
            reader = module.PreparedDataset(tmp, output)
            x, y = next(reader.iter_batches("train"))
            np.testing.assert_allclose(x.mean(axis=(0, 1)), 0, atol=1e-6)
            np.testing.assert_allclose(x.std(axis=(0, 1)), 1, atol=1e-6)
            self.assertGreater(float(next(reader.iter_batches("validation"))[0].mean()), 100)
            reader.close()
            with (output / "train_x.npy").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.PreparedDataset(tmp, output)

    def test_missing_required_checksums_are_rejected(self):
        for missing in ("validation_y.npy", "normalization.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                _, folder = make_dataset(tmp)
                module = self.module()
                output = Path(tmp, "cache")
                module.prepare_dataset(tmp, folder, output)
                metadata_path = output / "metadata.json"
                metadata = json.loads(metadata_path.read_text())
                del metadata["files_sha256"][missing]
                metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "checksum.*complete"):
                    module.PreparedDataset(tmp, output)

    def test_partial_mapping_creation_failure_closes_files_and_cleans_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            module = self.module()
            original = np.lib.format.open_memmap
            opened = []
            def fail_second(path, **kwargs):
                if str(path).endswith("train_y.npy"):
                    raise OSError("Injected mapping failure")
                mapped = original(path, **kwargs)
                if str(path).endswith("train_x.npy"):
                    opened.append(mapped)
                return mapped
            with patch.object(np.lib.format, "open_memmap", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "Injected"):
                    module.prepare_dataset(tmp, folder, Path(tmp, "cache"))
            self.assertTrue(opened[0]._mmap.closed)
            self.assertFalse(Path(tmp, "cache").exists())
            self.assertEqual(list(Path(tmp).glob(".prepare-*")), [])

    def test_reader_context_closes_all_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            module = self.module()
            output = Path(tmp, "cache")
            module.prepare_dataset(tmp, folder, output)
            with module.PreparedDataset(tmp, output) as reader:
                arrays = [array for pair in reader.arrays.values() for array in pair]
                self.assertEqual(len(next(reader.iter_batches("train"))[1]), 2)
            self.assertTrue(all(array._mmap.closed for array in arrays))
            reader.close()

    def test_reader_partial_open_failure_closes_already_opened_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            module = self.module()
            output = Path(tmp, "cache")
            module.prepare_dataset(tmp, folder, output)
            original = np.load
            opened = []
            def fail_second(path, **kwargs):
                if str(path).endswith("train_y.npy"):
                    raise OSError("Injected read failure")
                mapped = original(path, **kwargs)
                opened.append(mapped)
                return mapped
            with patch.object(np, "load", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "Injected"):
                    module.PreparedDataset(tmp, output)
            self.assertTrue(opened[0]._mmap.closed)
