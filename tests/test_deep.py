import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fixture_data import make_dataset
from src.ect.prepare import prepare_dataset, PreparedDataset


CONFIG = dict(schema_version=1, protocol='deep_ac_v1', purpose='smoke_validation_only',
              model='cnn', augmentation=False, seed=0, sampling_seed=20261001,
              device='cpu', threads=1, batch_size=2, max_epochs=2, patience=2,
              learning_rate=0.001, weight_decay=0.0001,
              train_per_class=None, validation_per_class=None)


class DeepTests(unittest.TestCase):
    def module(self):
        from src.ect import deep
        return deep

    def test_sealed_test_training_noise_validation_and_checkpoint_roundtrip(self):
        m = self.module()
        class Guard(dict):
            def __getitem__(self, key):
                if key == 'test':
                    raise AssertionError('Heldout classification accessed')
                return super().__getitem__(key)
        class Sealed(PreparedDataset):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.arrays = Guard(self.arrays)
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            for model, augmentation in [('cnn', False), ('resnet', True)]:
                run = Path(tmp, model)
                config = CONFIG | dict(model=model, augmentation=augmentation)
                with patch.object(m, 'PreparedDataset', Sealed):
                    report = m.train_deep(tmp, config, run, 'cache')
                self.assertFalse(report['heldout_test_evaluated'])
                self.assertEqual(report['sample_counts'], {'train': 2, 'validation': 2})
                self.assertEqual(set(report['validation_metrics']), {'clean', '30', '20', '10'})
                self.assertEqual(report['selected_epoch'], max(report['epochs'], key=lambda e:
                    (*m.selection_key(e['validation_metrics'], augmentation), -e['epoch']))['epoch'])
                bundle = m.load_deep_model(run, device='cpu')
                self.assertEqual(bundle['normalization']['fitted_split'], 'train')
                self.assertTrue(all(t.device.type == 'cpu' for t in bundle['model'].state_dict().values()))
                with PreparedDataset(tmp, 'cache') as store:
                    with (run / 'validation_predictions.csv').open() as handle:
                        predictions = list(csv.DictReader(handle))
                    for condition, snr in [('clean', None), ('30', 30), ('20', 20), ('10', 10)]:
                        records = [json.loads(line) for line in (folder / 'validation.jsonl').read_text().splitlines()]
                        actual = m.predict_raw(bundle, store.arrays['validation'][0],
                            [r['wave_sha256'] for r in records], snr_db=snr, seed=10, namespace='validation', batch_size=1)
                        selected = [int(r['predicted_class_index']) for r in predictions if r['condition'] == condition]
                        np.testing.assert_array_equal(actual, selected)
                self.assertGreater(report['parameter_update_l2'], 0)
                with (run / 'model.pt').open('ab') as handle:
                    handle.write(b'tampered')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    m.load_deep_model(run)

    def test_selection_is_fixed_clean_or_equal_four_condition_average(self):
        m = self.module()
        metric = {'clean': {'macro_f1': .8, 'accuracy': .9},
                  '30': {'macro_f1': .6, 'accuracy': .7},
                  '20': {'macro_f1': .4, 'accuracy': .5},
                  '10': {'macro_f1': .2, 'accuracy': .3}}
        np.testing.assert_allclose(m.selection_key(metric, False), [.8, .9])
        np.testing.assert_allclose(m.selection_key(metric, True), [.5, .6])

    def test_failure_record_and_atomic_directory_claim(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp, 'failed')
            with patch.object(m, 'PreparedDataset', side_effect=ValueError('broken cache')):
                with self.assertRaisesRegex(ValueError, 'broken cache'):
                    m.train_deep(tmp, CONFIG, run)
            self.assertEqual(json.loads((run / 'failure.json').read_text())['status'], 'failed')
            self.assertFalse((run / 'report.json').exists())
            with self.assertRaises(FileExistsError):
                m.train_deep(tmp, CONFIG, run)
            with self.assertRaises((ValueError, FileNotFoundError)):
                m.load_deep_model(run)

    def test_invalid_config_rejected_before_claim(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            for field, value in [('batch_size', 0), ('seed', -1), ('device', 'gpu'),
                                 ('learning_rate', float('nan')), ('augmentation', 'yes'),
                                 ('model', 'unknown'), ('max_epochs', True)]:
                run = Path(tmp, field)
                with self.assertRaises(ValueError):
                    m.train_deep(tmp, CONFIG | {field: value}, run)
                self.assertFalse(run.exists())

    def test_labels_must_match_manifest_not_only_cached_checksum(self):
        m = self.module()
        class Wrong(PreparedDataset):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                x, y = self.arrays['train']
                self.arrays['train'] = x, np.array(y[::-1])
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            with patch.object(m, 'PreparedDataset', Wrong):
                with self.assertRaisesRegex(ValueError, 'manifest'):
                    m.train_deep(tmp, CONFIG, 'run', 'cache')

    def test_zero_ac_validation_snr_marked_not_applicable(self):
        m = self.module()
        class ConstantValidation(PreparedDataset):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                x, y = self.arrays['validation']
                changed = np.array(x)
                changed[0] = [1., 2.]
                self.arrays['validation'] = changed, y
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            with patch.object(m, 'PreparedDataset', ConstantValidation):
                report = m.train_deep(tmp, CONFIG | {'max_epochs': 1}, 'run', 'cache')
            for condition in ['30', '20', '10']:
                self.assertEqual(report['validation_noise_summary'][condition]['not_applicable_count'], 1)
                self.assertLess(report['validation_noise_summary'][condition]['max_snr_error_db'], 1e-8)
            with Path(tmp, 'run/validation_predictions.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row['snr_applicable'] == 'False' for row in rows if row['row_index'] == '0' and row['condition'] != 'clean'))

    def test_near_tie_device_difference_is_reported_not_checkpoint_failure(self):
        m = self.module()
        reference = np.array([[1., 1. + 1e-8], [3., 1.]], dtype=np.float64)
        cpu = np.array([[1. + 1e-8, 1.], [3., 1.]], dtype=np.float64)
        comparison = m.compare_logits(reference, cpu)
        self.assertTrue(comparison['within_tolerance'])
        self.assertEqual(comparison['argmax_difference_count'], 1)
        self.assertLess(comparison['max_abs_error'], 2e-8)

    def test_reload_rejects_contradictory_epoch_configuration_and_model_metadata(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            report = m.train_deep(tmp, CONFIG, 'run', 'cache')
            path = Path(tmp, 'run/report.json')
            for field, value in [('selected_epoch', 999), ('config_sha256', '0' * 64),
                                 ('classes', [0, 1, 2]), ('schema_version', 999)]:
                changed = report | {field: value}
                path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    m.load_deep_model(path.parent)
            changed = json.loads(json.dumps(report))
            changed['epochs'][report['selected_epoch']]['validation_metrics']['clean']['accuracy'] = .123
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                m.load_deep_model(path.parent)


if __name__ == '__main__':
    unittest.main()
