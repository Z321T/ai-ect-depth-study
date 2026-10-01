"""Scientific invariants for identity-keyed, AC-referenced noise."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


class NoiseTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('src.ect.noise')
        except ModuleNotFoundError as exc:
            if exc.name != 'src.ect.noise':
                raise
            self.fail('AC noise implementation is missing')

    def signals(self, count=4):
        t = np.arange(250, dtype=np.float64)
        wave = np.column_stack((np.sin(t * .13), 3 * np.cos(t * .07)))
        return np.stack([wave * (i + 1) + [17, -31] for i in range(count)])

    def test_joint_ac_power_exact_snr_and_channel_dc_preservation(self):
        x = self.signals()
        before = x.copy()
        snrs = [-5., 10., 20., 30.]
        noisy, info = self.module().add_ac_noise(x, ['a', 'b', 'c', 'd'], snrs, 7, 'validation')
        residual = noisy - x
        ac_power = np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=(1, 2))
        noise_power = np.mean(residual ** 2, axis=(1, 2))
        np.testing.assert_allclose(10 * np.log10(ac_power / noise_power), snrs, atol=1e-11, rtol=0)
        np.testing.assert_allclose(residual.mean(axis=1), 0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(noisy.mean(axis=1), x.mean(axis=1), atol=3e-14, rtol=0)
        np.testing.assert_allclose(info['ac_power'], ac_power, rtol=1e-13)
        np.testing.assert_allclose(info['noise_power'], noise_power, rtol=1e-13)
        self.assertEqual(info['snr_applicable'], [True] * 4)
        self.assertEqual(info['snr_db'], snrs)
        self.assertEqual(noisy.dtype, np.float64)
        np.testing.assert_array_equal(x, before)
        self.assertFalse(np.shares_memory(noisy, x))

    def test_scalar_snr_and_float32_readonly_input(self):
        x = self.signals(2).astype(np.float32)
        before = x.copy()
        x.flags.writeable = False
        scalar, info = self.module().add_ac_noise(x, ['a', 'b'], 20, 0, 'validation')
        vector, _ = self.module().add_ac_noise(x, ['a', 'b'], [20., 20.], 0, 'validation')
        np.testing.assert_array_equal(scalar, vector)
        np.testing.assert_array_equal(x, before)
        self.assertEqual(scalar.dtype, np.float64)
        self.assertEqual(info['snr_db'], [20., 20.])

    def test_exact_constants_are_unchanged_and_snr_is_not_applicable(self):
        x = np.broadcast_to(np.array([0.1, -1e12]), (3, 250, 2)).copy()
        noisy, info = self.module().add_ac_noise(x, ['a', 'b', 'c'], 10, 3, 'validation')
        np.testing.assert_array_equal(noisy, x)
        self.assertEqual(info['ac_power'], [0.] * 3)
        self.assertEqual(info['noise_power'], [0.] * 3)
        self.assertEqual(info['snr_applicable'], [False] * 3)
        self.assertEqual(info['snr_db'], [10.] * 3)

    def test_ac_power_is_stable_with_large_dc_offset(self):
        x = np.tile(np.array([[0., 0.], [1., 2.], [0., 0.]]), (1, 1, 1)) + 2. ** 40
        _, info = self.module().add_ac_noise(x, ['a'], 10., 0, 'validation')
        # Channel variances are 2/9 and 8/9: joint power is 5/9.
        self.assertAlmostEqual(info['ac_power'][0], 5 / 9, places=15)

    def test_one_constant_channel_still_receives_jointly_calibrated_noise(self):
        x = self.signals(1)
        x[:, :, 1] = 8.
        noisy, info = self.module().add_ac_noise(x, ['a'], 20, 0, 'validation')
        residual = noisy - x
        self.assertGreater(np.mean(residual[:, :, 1] ** 2), 0)
        self.assertAlmostEqual(10 * np.log10(info['ac_power'][0] / np.mean(residual ** 2)), 20., places=11)

    def test_identity_reordering_partitioning_and_repetition_are_identical(self):
        x = self.signals()
        ids, snrs = ['a', 'b', 'c', 'd'], [30., 20., 10., 0.]
        module = self.module()
        full, info = module.add_ac_noise(x, ids, snrs, 9, 'validation')
        order = [3, 1, 0, 2]
        permuted, perm_info = module.add_ac_noise(x[order], [ids[i] for i in order], [snrs[i] for i in order], 9, 'validation')
        np.testing.assert_array_equal(permuted, full[order])
        for key in ('ac_power', 'noise_power', 'snr_applicable', 'snr_db'):
            self.assertEqual(perm_info[key], [info[key][i] for i in order])
        for i in range(4):
            single, _ = module.add_ac_noise(x[i:i + 1], ids[i:i + 1], snrs[i], 9, 'validation')
            np.testing.assert_array_equal(single[0], full[i])
        repeated, _ = module.add_ac_noise(np.repeat(x[:1], 2, axis=0), ['a', 'a'], 30., 9, 'validation')
        np.testing.assert_array_equal(repeated[0], repeated[1])

    def test_seed_identity_namespace_and_condition_separate_noise_streams(self):
        x = self.signals(1)
        module = self.module()
        baseline, _ = module.add_ac_noise(x, ['a'], 20., 9, 'validation')
        for identity, snr, seed, namespace in [('b', 20., 9, 'validation'), ('a', 20., 10, 'validation'),
                                               ('a', 20., 9, 'other'), ('a', 10., 9, 'validation')]:
            changed, _ = module.add_ac_noise(x, [identity], snr, seed, namespace)
            self.assertFalse(np.array_equal(changed, baseline))
            if snr != 20.:
                self.assertLess(abs(np.corrcoef((changed - x).ravel(), (baseline - x).ravel())[0, 1]), .3)
        # Delimiter-containing identities must not alias different tuple keys.
        a, _ = module.add_ac_noise(x, ['b|c'], 20, 0, 'a')
        b, _ = module.add_ac_noise(x, ['c'], 20, 0, 'a|b')
        self.assertFalse(np.array_equal(a, b))

    def test_global_numpy_rng_state_is_untouched(self):
        state = np.random.get_state()
        self.module().add_ac_noise(self.signals(1), ['a'], 10., 3, 'validation')
        self.module().augment_training(self.signals(1), ['a'], 3, 2)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_fresh_processes_with_different_python_hash_seeds_are_identical(self):
        code = '''
import hashlib
import json
import numpy as np
from src.ect.noise import add_ac_noise, augment_training
x = np.arange(48, dtype=np.float64).reshape(4, 6, 2)
ids = ['sample-0', 'sample-1', 'sample-2', 'sample-3']
results = [add_ac_noise(x, ids, 20., 17, 'validation'),
           augment_training(x, ids, 17, 4)]
print(json.dumps([(hashlib.sha256(noisy.tobytes()).hexdigest(), info)
                  for noisy, info in results], sort_keys=True, allow_nan=False))
'''
        outputs = []
        for hash_seed in ['1', '999']:
            result = subprocess.run([sys.executable, '-c', code],
                                    cwd=Path(__file__).resolve().parents[1],
                                    env=dict(os.environ, PYTHONHASHSEED=hash_seed),
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stderr, '')
            outputs.append(json.loads(result.stdout))
        self.assertEqual(outputs[0], outputs[1])

    def test_small_nonzero_ac_is_not_treated_as_constant(self):
        x = (self.signals(1) - [17., -31.]) * 1e-12
        noisy, info = self.module().add_ac_noise(x, ['tiny'], 20, 0, 'validation')
        self.assertEqual(info['snr_applicable'], [True])
        self.assertGreater(info['ac_power'][0], 0)
        self.assertAlmostEqual(10 * np.log10(info['ac_power'][0] / np.mean((noisy - x) ** 2)), 20., places=12)

    def test_invalid_signals_are_rejected(self):
        module = self.module()
        for x in (np.zeros((250, 2)), np.zeros((1, 250, 3)), np.zeros((1, 1, 2)),
                  np.zeros((0, 250, 2)), np.full((1, 250, 2), np.nan),
                  np.full((1, 250, 2), np.inf), np.ones((1, 250, 2), dtype=complex)):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                module.add_ac_noise(x, ['a'], 10., 0, 'validation')

    def test_invalid_identity_seed_namespace_and_snr_are_rejected(self):
        module, x = self.module(), self.signals(2)
        valid = dict(signals=x, sample_ids=['a', 'b'], snr_db=10., seed=0, namespace='validation')
        bad_values = {'sample_ids': [[], ['a'], ['a', 'b', 'c'], 'ab', ['a', None]],
                      'seed': [-1, True, 1.5, '1'], 'namespace': ['', ' ', None, 2],
                      'snr_db': [None, np.nan, np.inf, -np.inf, [10.], [10., 20., 30.],
                                 [[10., 20.]], [10., np.nan], '10', True]}
        for key, values in bad_values.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    module.add_ac_noise(**dict(valid, **{key: value}))

    def test_augmentation_clean_rows_metadata_and_noisy_snr(self):
        x = self.signals(512)
        before = x.copy()
        noisy, info = self.module().augment_training(x, [f'sample-{i}' for i in range(len(x))], 13, 2)
        mask = np.asarray(info['augmented'], dtype=bool)
        self.assertGreater(mask.mean(), .4)
        self.assertLess(mask.mean(), .6)
        requested = info['snrs']
        self.assertEqual(info['snr_db'], requested)
        self.assertEqual(info['snr_applicable'], mask.tolist())
        np.testing.assert_array_equal(noisy[~mask], x[~mask])
        self.assertTrue(all(s is None for s, chosen in zip(requested, mask) if not chosen))
        snrs = np.array([s for s, chosen in zip(requested, mask) if chosen])
        self.assertTrue(np.all((snrs >= 10.) & (snrs <= 30.)))
        self.assertGreater(snrs.mean(), 18.)
        self.assertLess(snrs.mean(), 22.)
        self.assertGreater(np.count_nonzero(snrs < 15.), 20)
        self.assertGreater(np.count_nonzero(snrs > 25.), 20)
        actual = np.mean((noisy - x) ** 2, axis=(1, 2))
        np.testing.assert_allclose(info['noise_power'], actual, rtol=1e-13, atol=0)
        self.assertTrue(np.all(actual[~mask] == 0))
        np.testing.assert_allclose(10 * np.log10(np.asarray(info['ac_power'])[mask] / actual[mask]), snrs, atol=1e-11, rtol=0)
        np.testing.assert_allclose((noisy - x).mean(axis=1), 0, atol=2e-13, rtol=0)
        np.testing.assert_array_equal(x, before)

    def test_augmentation_is_identity_epoch_keyed_and_batch_independent(self):
        x = self.signals(40)
        ids = [f'sample-{i}' for i in range(len(x))]
        module = self.module()
        full, info = module.augment_training(x, ids, 13, 2)
        order = np.arange(len(x))[::-1]
        permuted, perm_info = module.augment_training(x[order], [ids[i] for i in order], 13, 2)
        np.testing.assert_array_equal(permuted, full[order])
        for key in ('augmented', 'snrs', 'snr_db', 'ac_power', 'noise_power', 'snr_applicable'):
            self.assertEqual(perm_info[key], [info[key][i] for i in order])
        for i in range(len(x)):
            single, single_info = module.augment_training(x[i:i + 1], ids[i:i + 1], 13, 2)
            np.testing.assert_array_equal(single[0], full[i])
            self.assertEqual(single_info['snrs'], info['snrs'][i:i + 1])
        for seed, epoch in [(14, 2), (13, 3)]:
            changed, changed_info = module.augment_training(x, ids, seed, epoch)
            self.assertFalse(np.array_equal(changed, full))
            self.assertNotEqual(changed_info['snrs'], info['snrs'])

    def test_augmentation_constant_rows_remain_unchanged(self):
        x = np.full((40, 250, 2), .1)
        noisy, info = self.module().augment_training(x, [f'constant-{i}' for i in range(40)], 13, 2)
        np.testing.assert_array_equal(noisy, x)
        self.assertTrue(any(info['augmented']))
        self.assertEqual(info['snr_applicable'], [False] * 40)
        self.assertEqual(info['ac_power'], [0.] * 40)
        self.assertEqual(info['noise_power'], [0.] * 40)

    def test_mixed_batch_masks_support_training_counters_and_json_reports(self):
        x = self.signals(80)
        x[::3] = [.1, -7.]
        noisy, info = self.module().augment_training(x, [f'mixed-{i}' for i in range(len(x))], 13, 2)
        # Exercise the actual consumer's masks without requiring Torch.
        augmented = np.asarray(info['augmented'])
        applicable = np.asarray(info['snr_applicable'])
        self.assertEqual(augmented.dtype, np.dtype(bool))
        self.assertEqual(applicable.dtype, np.dtype(bool))
        ac_power = np.asarray(info['ac_power'])
        np.testing.assert_array_equal(applicable, augmented & (ac_power > 0))
        zero_ac_count = int(np.sum(np.logical_and(augmented, np.logical_not(applicable))))
        self.assertGreater(zero_ac_count, 0)
        self.assertGreater(int(np.sum(augmented)) - zero_ac_count, 0)
        np.testing.assert_array_equal(noisy[~applicable], x[~applicable])
        self.assertEqual(json.loads(json.dumps(info, allow_nan=False)), info)

    def test_augmentation_rejects_invalid_epoch_seed_signals_and_identities(self):
        module, x = self.module(), self.signals(2)
        for epoch in [-1, True, 1.5, None]:
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                module.augment_training(x, ['a', 'b'], 0, epoch)
        for signals, ids, seed in [(x, ['a'], 0), (x, ['a', 'b'], -1),
                                   (x * np.nan, ['a', 'b'], 0)]:
            with self.assertRaises(ValueError):
                module.augment_training(signals, ids, seed, 0)


if __name__ == '__main__':
    unittest.main()
