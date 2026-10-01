import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fixture_data import make_dataset
from src.ect.prepare import prepare_dataset, PreparedDataset
from src.ect.features import extract_features


CONFIG = {'schema_version': 1, 'protocol': 'svm_features_v1',
          'purpose': 'development_validation_only', 'seed': 20261001,
          'threads': 1, 'train_per_class': None,
          'candidates': [{'kernel': 'linear', 'C': 1.}, {'kernel': 'rbf', 'C': 1., 'gamma': 'scale'}]}


class BaselineTests(unittest.TestCase):
    def module(self):
        from src.ect import baseline
        return baseline

    def test_sampling_is_stratified_reproducible_and_preserves_order(self):
        module = self.module()
        labels = np.repeat(np.arange(3), 10)
        first = module.stratified_rows(labels, 3, 12)
        np.testing.assert_array_equal(first, module.stratified_rows(labels, 3, 12))
        np.testing.assert_array_equal(np.bincount(labels[first]), [3, 3, 3])
        self.assertTrue(np.all(np.diff(first) > 0))
        np.testing.assert_array_equal(module.stratified_rows(labels, None, 12), np.arange(30))

    def test_train_only_scaler_sealed_test_and_roundtrip_predictions(self):
        module = self.module()
        class Guard(dict):
            def __getitem__(self, key):
                if key == 'test':
                    raise AssertionError('Heldout test accessed')
                return super().__getitem__(key)
        class Sealed(PreparedDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.arrays = Guard(self.arrays)
            def iter_batches(self, split, *args, **kwargs):
                if split == 'test':
                    raise AssertionError('Heldout test accessed')
                return super().iter_batches(split, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            with patch.object(module, 'PreparedDataset', Sealed):
                report = module.train_svm(tmp, CONFIG, output, cache_dir='cache')
            self.assertFalse(report['heldout_test_evaluated'])
            self.assertEqual(report['sample_counts'], {'train': 2, 'validation': 2})
            with PreparedDataset(tmp, 'cache') as store:
                train = extract_features(store.arrays['train'][0])
                validation = extract_features(store.arrays['validation'][0])
            model = module.load_svm_model(output)
            np.testing.assert_allclose(model['pipeline'].named_steps['scale'].mean_, train.mean(axis=0))
            self.assertEqual(model['pipeline'].named_steps['scale'].n_samples_seen_, 2)
            with (output / 'validation_predictions.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            np.testing.assert_array_equal(model['pipeline'].predict(validation), [int(r['predicted_class_index']) for r in rows])
            self.assertEqual([int(r['row_index']) for r in rows], [0, 1])
            expected = max(range(len(report['candidates'])), key=lambda i: (
                report['candidates'][i]['metrics']['macro_f1'], report['candidates'][i]['metrics']['accuracy'], -i))
            self.assertEqual(report['selected_candidate'], expected)
            with (output / 'model.joblib').open('ab') as handle:
                handle.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                module.load_svm_model(output)

    def test_existing_run_is_not_overwritten(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, 'run'); output.mkdir()
            with self.assertRaises(FileExistsError):
                module.train_svm(tmp, CONFIG, output)

    def test_failed_run_is_recorded_without_success_artifacts(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            with patch.object(module, 'extract_features', side_effect=RuntimeError('Injected feature failure')):
                with self.assertRaisesRegex(RuntimeError, 'Injected'):
                    module.train_svm(tmp, CONFIG, output, cache_dir='cache')
            failure = json.loads((output / 'failure.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['config'], CONFIG)
            self.assertFalse((output / 'report.json').exists())
            self.assertFalse((output / 'model.joblib').exists())

    def test_invalid_sampling_or_candidate_is_rejected(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.stratified_rows(np.array([0, 1]), 0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            config = {**CONFIG, 'candidates': [{'kernel': 'rbf', 'C': -1., 'gamma': 'scale'}]}
            with self.assertRaises(ValueError):
                module.train_svm(tmp, config, Path(tmp, 'run'))

    def test_rbf_uses_supported_uncalibrated_default_without_deprecation(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            config = {**CONFIG, 'candidates': [CONFIG['candidates'][1]]}
            report = module.train_svm(tmp, config, output, cache_dir='cache')
            self.assertEqual(report['candidates'][0]['warnings'], [])
            model = module.load_svm_model(output)['pipeline'].named_steps['svm']
            self.assertFalse(hasattr(model, 'predict_proba'))

    def test_running_job_claims_output_before_fitting(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            def concurrent_attempt(record):
                if record['index'] == 0:
                    with self.assertRaises(FileExistsError):
                        module.train_svm(tmp, CONFIG, output, cache_dir='cache')
            report = module.train_svm(tmp, CONFIG, output, cache_dir='cache', progress=concurrent_attempt)
            self.assertEqual(report['status'], 'complete')
            self.assertFalse((output / 'failure.json').exists())

    def test_model_accepts_crlf_feature_source_but_rejects_actual_code_changes(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            module.train_svm(tmp, CONFIG, output, cache_dir='cache')
            source = Path(module.__file__).with_name('features.py')
            code = source.read_bytes()
            original = Path.read_bytes
            def crlf(path):
                return code.replace(b'\n', b'\r\n') if path == source else original(path)
            with patch.object(Path, 'read_bytes', crlf):
                module.load_svm_model(output)
            def changed(path):
                return code + b'\n# Actual code change\n' if path == source else original(path)
            with patch.object(Path, 'read_bytes', changed):
                with self.assertRaisesRegex(ValueError, 'Feature implementation'):
                    module.load_svm_model(output)

    def test_partial_publication_failure_removes_success_artifacts(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            original = Path.rename
            def fail_prediction_publish(path, target):
                if Path(target) == output / 'validation_predictions.csv':
                    raise OSError('Injected publication failure')
                return original(path, target)
            with patch.object(Path, 'rename', fail_prediction_publish):
                with self.assertRaisesRegex(OSError, 'Injected publication'):
                    module.train_svm(tmp, CONFIG, output, cache_dir='cache')
            self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])

    def test_success_and_failure_markers_cannot_be_loaded_together(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            module.train_svm(tmp, CONFIG, output, cache_dir='cache')
            (output / 'failure.json').write_text('{}')
            with self.assertRaises(ValueError):
                module.load_svm_model(output)

    def test_generated_json_keeps_checksums_after_git_line_ending_normalization(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            output = Path(tmp, 'run')
            original = Path.write_text
            def windows_default(path, text, *args, **kwargs):
                if path.suffix == '.json' and kwargs.get('newline') != '\n':
                    text = text.replace('\n', '\r\n')
                return original(path, text, *args, **kwargs)
            with patch.object(Path, 'write_text', windows_default):
                module.train_svm(tmp, CONFIG, output, cache_dir='cache')
            for path in output.glob('*.json'):
                path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
            module.load_svm_model(output)
