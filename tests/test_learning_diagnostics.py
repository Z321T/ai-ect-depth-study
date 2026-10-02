import importlib
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


class DiagnosticInferenceTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('src.ect.learning_diagnostics')

    def bundle(self):
        model = nn.Sequential(nn.BatchNorm1d(2), nn.AdaptiveAvgPool1d(1),
                              nn.Flatten(), nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].running_mean.copy_(torch.tensor([8., -8.]))
            model[0].running_var.copy_(torch.tensor([.1, 10.]))
            model[3].weight.copy_(torch.eye(2))
        model.train()
        return dict(model=model, normalization={'mean': [0., 0.], 'std': [1., 1.]})

    def test_batch_statistics_are_diagnostic_and_original_state_never_changes(self):
        m = self.module()
        bundle = self.bundle()
        before = copy.deepcopy(bundle['model'].state_dict())
        raw = np.random.default_rng(42).normal(size=(6, 250, 2)).astype(np.float32)
        frozen = m.diagnostic_logits(bundle, raw, batch_size=3)
        counterfactual = m.diagnostic_logits(bundle, raw, batch_size=3, batch_stats=True)
        self.assertGreater(float(np.max(np.abs(frozen - counterfactual))), 1.)
        self.assertTrue(bundle['model'].training)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, bundle['model'].state_dict()[key]), key)
        self.assertTrue(all(p.grad is None for p in bundle['model'].parameters()))

    def test_mixed_batch_order_is_reproducible_and_restored_to_manifest_rows(self):
        m = self.module()
        bundle = self.bundle()
        raw = np.random.default_rng(7).normal(size=(7, 250, 2)).astype(np.float32)
        actual = m.diagnostic_logits(bundle, raw, batch_size=3, batch_stats=True, order_seed=20261002)
        np.testing.assert_array_equal(actual, m.diagnostic_logits(bundle, raw, batch_size=3,
            batch_stats=True, order_seed=20261002))
        order = np.random.default_rng(20261002).permutation(len(raw))
        expected = m.diagnostic_logits(bundle, raw[order], batch_size=3, batch_stats=True)
        np.testing.assert_allclose(actual[order], expected, rtol=1e-6, atol=1e-6)
        frozen = m.diagnostic_logits(bundle, raw, batch_size=3)
        np.testing.assert_allclose(frozen, m.diagnostic_logits(bundle, raw, batch_size=3,
            order_seed=20261003), rtol=1e-6, atol=1e-6)

    def test_invalid_input_and_nonfinite_outputs_are_rejected(self):
        m = self.module()
        b = self.bundle()
        for bad in (np.zeros((0, 250, 2)), np.zeros((2, 224, 2)), np.full((2, 250, 2), np.nan)):
            with self.assertRaises(ValueError):
                m.diagnostic_logits(b, bad)
        for bad in (0, True):
            with self.assertRaises(ValueError):
                m.diagnostic_logits(b, np.zeros((2, 250, 2)), batch_size=bad)
        with torch.no_grad():
            b['model'][3].weight.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            m.diagnostic_logits(b, np.zeros((2, 250, 2)))

    def test_signal_summary_preserves_large_dc_and_resolves_tiny_ac(self):
        m = self.module()
        t = np.linspace(-1., 1., 250)
        raw = np.empty((2, 250, 2), dtype=np.float64)
        raw[:, :, 0] = 1000. + 1e-4*t
        raw[:, :, 1] = -2000. + 2e-4*t
        s = m.signal_summary(raw, {'mean': [1000., -2000.], 'std': [1., 2.]})
        self.assertEqual(s['sample_count'], 2)
        self.assertAlmostEqual(s['raw_channel_mean'][0], 1000.)
        self.assertGreater(s['normalized_joint_ac_rms_quantiles']['0.5'], 0.)
        self.assertLess(s['normalized_joint_ac_rms_quantiles']['0.5'], .001)


class DiagnosticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fixture_evaluation import make_evaluation
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.registry = make_evaluation(cls.root)
        project = Path(__file__).resolve().parents[1]
        for relative in ('src/ect/learning_diagnostics.py', 'scripts/diagnose_learning.py'):
            path = cls.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((project/relative).read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_registered_diagnostic_never_indexes_test_and_csv_counts_match(self):
        m = importlib.import_module('src.ect.learning_diagnostics')
        original = m.PreparedDataset
        class TrainValidationOnly(dict):
            def __getitem__(self, key):
                if key == 'test':
                    raise AssertionError('Test samples must not be indexed')
                return super().__getitem__(key)
        class Guarded(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.arrays = TrainValidationOnly(self.arrays)
        with patch.object(m, 'PreparedDataset', Guarded):
            report = m.diagnose_learning(self.root, 'diagnostic', device='cpu',
                registry=str(self.registry.relative_to(self.root)), cache_dir='cache')
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['test_classification_evaluated'])
        self.assertEqual(report['prediction_count'], 112)
        self.assertTrue(report['checkpoint_files_unchanged'])
        with (self.root/'diagnostic/predictions.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 112)
        self.assertEqual({r['split'] for r in rows}, {'train', 'validation'})
        for entry in report['metrics']:
            selected = [r for r in rows if (r['split'],r['run_id'],r['mode']) ==
                        (entry['split'],entry['run_id'],entry['mode'])]
            acc = sum(r['true_class'] == r['predicted_class'] for r in selected)/len(selected)
            self.assertEqual(acc, entry['metrics']['accuracy'])
        with self.assertRaises(FileExistsError):
            m.diagnose_learning(self.root, 'diagnostic', device='cpu')

    def test_failure_is_recorded_and_no_success_report_remains(self):
        m = importlib.import_module('src.ect.learning_diagnostics')
        with patch.object(m, 'load_registered', side_effect=ValueError('invalid binding')):
            with self.assertRaisesRegex(ValueError, 'invalid binding'):
                m.diagnose_learning(self.root, 'failed', device='cpu',
                    registry=str(self.registry.relative_to(self.root)), cache_dir='cache')
        self.assertTrue((self.root/'failed/failure.json').exists())
        self.assertFalse((self.root/'failed/report.json').exists())


if __name__ == '__main__':
    unittest.main()
