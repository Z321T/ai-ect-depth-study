"""CPU contract tests for identity-keyed crops and probability averaging."""
import importlib
import json
import os
import subprocess
import sys
import unittest

import numpy as np
import torch
from torch import nn


class SmallClassifier(nn.Module):
    """Real affine classifier with explicit inference contract checks."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 3)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[.25, -.5], [-.5, .25], [.75, .125]]))
            self.classifier.bias.copy_(torch.tensor([-1., 2., -.5]))

    @property
    def device(self):
        return self.classifier.weight.device

    def forward(self, inputs):
        assert inputs.shape[1:] == (2, 224)
        assert inputs.dtype == torch.float32
        assert not torch.is_grad_enabled()
        assert all(not m.training for m in self.modules())
        return self.classifier(inputs[:, :, 0])


class BNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(.8)
        self.head = nn.Linear(2, 3)
        self.register_buffer('forward_calls', torch.tensor(0))

    def forward(self, inputs):
        assert not torch.is_grad_enabled()
        assert all(not m.training for m in self.modules())
        self.forward_calls.add_(1)
        return self.head(self.dropout(self.bn(inputs)).mean(dim=2))


class BadOutputClassifier(SmallClassifier):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.calls = 0

    def forward(self, inputs):
        outputs = super().forward(inputs)
        self.calls += 1
        if self.kind == 'nan':
            return outputs * float('nan')
        if self.kind == 'inf':
            return outputs * float('inf')
        if self.kind == 'rank':
            return outputs[:, 0]
        if self.kind == 'rows':
            return outputs[:0]
        if self.kind == 'classes':
            return outputs[:, :1]
        if self.kind == 'integer':
            return outputs.to(torch.int64)
        if self.kind == 'list':
            return outputs.tolist()
        if self.kind == 'varying':
            return outputs if self.calls == 1 else outputs[:, :2]
        raise AssertionError('unknown bad output fixture')


class PaperCropsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        try:
            self.api = importlib.import_module('src.ect.paper_crops')
        except ModuleNotFoundError as exc:
            if exc.name != 'src.ect.paper_crops':
                raise
            self.fail('paper_crops module is missing')
        self.ids = ['a', 'b', 'c']
        t = np.arange(250, dtype=np.float64)
        self.raw = np.stack([np.column_stack((t, -2 * t)) + [1000 * i, -300 * i]
                             for i in range(3)])
        self.stats = {'mean': [13., -26.], 'std': [2., 4.]}

    def test_identity_sha256_pcg64_reference_stream(self):
        # Independently generated compact-JSON/SHA256/PCG64 protocol fixtures.
        np.testing.assert_array_equal(
            self.api.crop_starts(self.ids, 10, 'training', epoch=0), [[1], [23], [21]])
        np.testing.assert_array_equal(
            self.api.crop_starts(self.ids, 10, 'validation'),
            [[14, 22, 15, 20, 16, 25, 9, 24, 20, 3],
             [18, 22, 6, 5, 5, 15, 4, 16, 7, 16],
             [20, 3, 1, 0, 8, 6, 15, 25, 26, 18]])

    def test_starts_cover_both_endpoints_and_have_integer_shapes(self):
        ids = [f'sample-{i}' for i in range(2000)]
        for namespace, epoch, count in [('training', 0, 1), ('validation', None, 10)]:
            starts = self.api.crop_starts(ids, 20261002, namespace, epoch)
            self.assertEqual(starts.shape, (2000, count))
            self.assertEqual(starts.dtype, np.int64)
            self.assertEqual(set(starts.flat), set(range(27)))
            # A broad distribution check catches strongly biased/nonrandom crops.
            frequencies = np.bincount(starts.ravel(), minlength=27)
            self.assertTrue(np.all(frequencies > starts.size / 27 * .5))
            self.assertTrue(np.all(frequencies < starts.size / 27 * 1.5))

    def test_training_crops_copy_exact_slices(self):
        before = self.raw.copy()
        self.raw.flags.writeable = False
        starts = self.api.crop_starts(self.ids, 10, 'training', 0)
        crops = self.api.training_crops(self.raw, self.ids, 10, 0)
        self.assertEqual(crops.shape, (3, 224, 2))
        self.assertEqual(crops.dtype, np.float32)
        for i, start in enumerate(starts[:, 0]):
            np.testing.assert_array_equal(crops[i], self.raw[i, start:start + 224].astype(np.float32))
        self.assertFalse(np.shares_memory(crops, self.raw))
        crops[0, 0, 0] = -999
        np.testing.assert_array_equal(self.raw, before)

    def test_validation_crops_copy_exact_ten_slices(self):
        before = self.raw.copy()
        starts = self.api.crop_starts(self.ids, 10, 'validation')
        crops = self.api.validation_crops(self.raw, self.ids)
        self.assertEqual(crops.shape, (3, 10, 224, 2))
        self.assertEqual(crops.dtype, np.float32)
        for i in range(3):
            for j, start in enumerate(starts[i]):
                np.testing.assert_array_equal(crops[i, j], self.raw[i, start:start + 224].astype(np.float32))
        self.assertFalse(np.shares_memory(crops, self.raw))
        np.testing.assert_array_equal(self.raw, before)

    def test_last_legal_crop_ends_at_point_249(self):
        ids = [f'end-{i}' for i in range(200)]
        raw = np.repeat(self.raw[:1], len(ids), axis=0)
        starts = self.api.crop_starts(ids, 7, 'training', 2)
        rows = np.flatnonzero(starts[:, 0] == 26)
        self.assertGreater(len(rows), 0)
        crops = self.api.training_crops(raw, ids, 7, 2)
        for row in rows:
            np.testing.assert_array_equal(crops[row], raw[row, 26:250])
            np.testing.assert_array_equal(crops[row, -1], raw[row, 249])

    def test_ten_random_crops_allow_repetition_and_are_not_sorted(self):
        starts = self.api.crop_starts([f'id-{i}' for i in range(30)], 10, 'validation')
        self.assertTrue(any(len(set(row)) < 10 for row in starts))
        self.assertTrue(np.any(np.diff(starts, axis=1) < 0))

    def test_crop_order_partitions_and_duplicate_ids_are_invariant(self):
        order = [2, 0, 1]
        for function, args in [(self.api.training_crops, (77, 8)),
                               (self.api.validation_crops, (77,))]:
            full = function(self.raw, self.ids, *args)
            permuted = function(self.raw[order], [self.ids[i] for i in order], *args)
            np.testing.assert_array_equal(permuted, full[order])
            pieces = [function(self.raw[i:i + 1], [self.ids[i]], *args) for i in range(3)]
            np.testing.assert_array_equal(np.concatenate(pieces), full)
            duplicate = function(np.repeat(self.raw[:1], 2, axis=0), ['a', 'a'], *args)
            np.testing.assert_array_equal(duplicate[0], duplicate[1])

    def test_epoch_seed_identity_and_namespace_change_streams(self):
        ids = [f'id-{i}' for i in range(64)]
        base = self.api.crop_starts(ids, 10, 'training', 0)
        for seed, epoch, names in [(11, 0, ids), (10, 1, ids),
                                  (10, 0, [name + '-other' for name in ids])]:
            self.assertFalse(np.array_equal(base, self.api.crop_starts(names, seed, 'training', epoch)))
        self.assertFalse(np.array_equal(base[:, 0], self.api.crop_starts(ids, 10, 'validation')[:, 0]))
        np.testing.assert_array_equal(base, self.api.crop_starts(ids, 10, 'training', 0))

    def test_global_numpy_rng_is_untouched(self):
        before = np.random.get_state()
        self.api.training_crops(self.raw, self.ids, 10, 0)
        self.api.validation_crops(self.raw, self.ids)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_process_hash_seed_does_not_change_starts(self):
        code = ("from src.ect.paper_crops import crop_starts; import json; "
                "print(json.dumps(crop_starts(['é', 'a'], 10, 'training', 2).tolist()))")
        outputs = []
        for seed in ('1', '999'):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1')
            outputs.append(json.loads(subprocess.check_output(
                [sys.executable, '-B', '-c', code], env=env, text=True)))
        self.assertEqual(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[0], self.api.crop_starts(['é', 'a'], 10, 'training', 2))

    def test_uint32_seed_bounds_and_numpy_integer_parameters(self):
        for seed in (0, 2**32 - 1, np.uint32(10)):
            self.api.validation_crops(self.raw, self.ids, seed)
        self.api.training_crops(self.raw, self.ids, np.uint32(10), np.int64(0))
        for seed in (-1, 2**32, True, np.bool_(False), 1.5, '10', None, np.nan, np.inf):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.api.crop_starts(self.ids, seed, 'validation')

    def test_namespace_and_epoch_contract_rejects_test(self):
        for namespace, epoch in [('training', None), ('training', -1), ('training', True),
                                 ('training', 1.5), ('training', np.inf), ('validation', 0),
                                 ('test', None), ('', None), (None, None), ([], None)]:
            with self.subTest(namespace=namespace, epoch=epoch), self.assertRaises(ValueError):
                self.api.crop_starts(self.ids, 10, namespace, epoch)

    def test_ids_must_be_a_nonempty_sequence_of_nonblank_strings(self):
        for ids in ([], 'abc', b'abc', None, [1], [''], ['  '], [None], [True], [['a']]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.api.crop_starts(ids, 10, 'validation')
        for ids in ([], ['a'], ['a', 'b', 'c', 'd']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.api.validation_crops(self.raw, ids)

    def test_crops_reject_wrong_shapes_types_and_nonfinite_values(self):
        invalid = [np.zeros((0, 250, 2)), np.zeros((3, 249, 2)), np.zeros((3, 250, 1)),
                   np.zeros((250, 2)), np.zeros((3, 250, 2, 1)),
                   self.raw.astype(complex), self.raw.astype(str), self.raw.astype(object),
                   np.ones((3, 250, 2), dtype=bool)]
        for value in (np.nan, np.inf, -np.inf, 1e300):
            raw = self.raw.copy()
            raw[0, 249, 1] = value
            invalid.append(raw)
        for raw in invalid:
            for function, args in [(self.api.training_crops, (10, 0)),
                                   (self.api.validation_crops, ())]:
                with self.subTest(shape=raw.shape, dtype=raw.dtype), self.assertRaises(ValueError):
                    function(raw, self.ids, *args)

    def test_ten_crop_probabilities_match_hand_calculation_not_logit_mean(self):
        raw = self.raw[:1].copy()
        before = raw.copy()
        model = SmallClassifier().train()
        probabilities = self.api.ten_crop_probabilities(model, raw, ['a'], self.stats, batch_size=3)
        starts = self.api.crop_starts(['a'], 10, 'validation')[0]
        weight = np.array([[.25, -.5], [-.5, .25], [.75, .125]])
        bias = np.array([-1., 2., -.5])
        logits = np.array([((raw[0, start] - [13., -26.]) / [2., 4.]) @ weight.T + bias
                           for start in starts])
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected = (exp / exp.sum(axis=1, keepdims=True)).mean(axis=0)
        np.testing.assert_allclose(probabilities, expected[None], atol=2e-7, rtol=0)
        mean_logits = logits.mean(axis=0)
        exp_mean = np.exp(mean_logits - mean_logits.max())
        wrong = exp_mean / exp_mean.sum()
        self.assertGreater(np.max(np.abs(expected - wrong)), .05)
        self.assertGreater(np.count_nonzero(probabilities > 0), 1)
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertEqual(probabilities.dtype, np.float32)
        np.testing.assert_array_equal(raw, before)
        self.assertTrue(model.training)

    def test_probability_order_partition_and_inference_batch_size_are_invariant(self):
        model = SmallClassifier()
        full = self.api.ten_crop_probabilities(model, self.raw, self.ids, self.stats)
        for size in (1, 3, 7, 10, 128, np.int64(5)):
            actual = self.api.ten_crop_probabilities(model, self.raw, self.ids, self.stats, batch_size=size)
            np.testing.assert_allclose(actual, full, atol=2e-7, rtol=0)
        order = [2, 0, 1]
        permuted = self.api.ten_crop_probabilities(model, self.raw[order], [self.ids[i] for i in order], self.stats)
        np.testing.assert_allclose(permuted, full[order], atol=2e-7, rtol=0)
        pieces = [self.api.ten_crop_probabilities(model, self.raw[i:i + 1], [self.ids[i]], self.stats)
                  for i in range(3)]
        np.testing.assert_allclose(np.concatenate(pieces), full, atol=2e-7, rtol=0)
        np.testing.assert_allclose(full.sum(axis=1), 1., atol=2e-7, rtol=0)

    def test_clone_preserves_original_bn_parameters_gradients_buffers_and_modes(self):
        model = BNClassifier().train()
        model.dropout.eval()  # Deliberately mixed submodule modes must be retained.
        with torch.no_grad():
            model.bn.running_mean.copy_(torch.tensor([5., -7.]))
            model.bn.running_var.copy_(torch.tensor([2., 3.]))
            model.bn.num_batches_tracked.fill_(17)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        states = {key: value.clone() for key, value in model.state_dict().items()}
        gradients = [p.grad.clone() for p in model.parameters()]
        modes = [module.training for module in model.modules()]
        rng_before = torch.get_rng_state().clone()
        first = self.api.ten_crop_probabilities(model, self.raw, self.ids, self.stats, batch_size=3)
        second = self.api.ten_crop_probabilities(model, self.raw, self.ids, self.stats, batch_size=7)
        np.testing.assert_allclose(first, second, atol=2e-7, rtol=0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, states[key]), key)
        for parameter, grad in zip(model.parameters(), gradients):
            self.assertTrue(torch.equal(parameter.grad, grad))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_inference_rejects_invalid_batch_sizes_and_models(self):
        for size in (0, -1, True, np.bool_(False), 1.5, '2', None, np.nan, np.inf):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.api.ten_crop_probabilities(SmallClassifier(), self.raw, self.ids, self.stats, batch_size=size)
        with self.assertRaises((ValueError, TypeError)):
            self.api.ten_crop_probabilities(None, self.raw, self.ids, self.stats)

    def test_inference_rejects_invalid_input_ids_and_seeds(self):
        model = SmallClassifier()
        for raw, ids, seed in [(self.raw[:, :249], self.ids, 10),
                               (self.raw * np.nan, self.ids, 10), (self.raw, ['a'], 10),
                               (self.raw, self.ids, -1), (self.raw, self.ids, 2**32)]:
            with self.subTest(seed=seed, ids=ids), self.assertRaises(ValueError):
                self.api.ten_crop_probabilities(model, raw, ids, self.stats, seed)

    def test_inference_rejects_invalid_normalization_and_standardization_overflow(self):
        invalid = [None, {}, {'mean': [0, 0]}, {'mean': [0], 'std': [1, 1]},
                   {'mean': [0, 0], 'std': [0, 1]}, {'mean': [0, 0], 'std': [-1, 1]},
                   {'mean': [np.nan, 0], 'std': [1, 1]},
                   {'mean': [0, 0], 'std': [np.inf, 1]},
                   {'mean': [0, 0], 'std': [1e-300, 1e-300]}]
        for stats in invalid:
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                self.api.ten_crop_probabilities(SmallClassifier(), self.raw, self.ids, stats)

    def test_inference_rejects_nonfinite_and_malformed_logits(self):
        for kind in ('nan', 'inf', 'rank', 'rows', 'classes', 'integer', 'list', 'varying'):
            model = BadOutputClassifier(kind).train()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.api.ten_crop_probabilities(model, self.raw, self.ids, self.stats, batch_size=3)
            self.assertTrue(model.training)
            self.assertEqual(model.calls, 0)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))

    def test_nonfinite_model_parameters_are_rejected_without_mutation(self):
        model = SmallClassifier()
        with torch.no_grad():
            model.classifier.bias[0] = float('inf')
        with self.assertRaises(ValueError):
            self.api.ten_crop_probabilities(model, self.raw, self.ids, self.stats)
        self.assertTrue(torch.isinf(model.classifier.bias[0]))


if __name__ == '__main__':
    unittest.main()
