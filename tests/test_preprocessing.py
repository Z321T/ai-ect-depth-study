import unittest
import numpy as np


class PreprocessingTests(unittest.TestCase):
    def module(self):
        try:
            from src.ect import preprocessing
        except ImportError:
            self.fail("Preprocessing implementation is missing")
        return preprocessing

    def test_preserves_dc_channels_and_does_not_mutate_input(self):
        signals = np.broadcast_to(np.array([1.06, 1.02], dtype=np.float32), (3, 1250, 2)).copy()
        before = signals.copy()
        result = self.module().downsample_iq(signals)
        self.assertEqual(result.shape, (3, 250, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, before[:, ::5], rtol=0, atol=2e-7)
        np.testing.assert_array_equal(signals, before)

    def test_keeps_low_frequency_and_suppresses_aliasing(self):
        t = np.arange(1250) / 2500
        x = np.stack([np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 600 * t)], axis=-1)
        y = self.module().downsample_iq(x)
        np.testing.assert_allclose(y[20:-20, 0], x[::5][20:-20, 0], atol=0.003)
        self.assertLess(float(np.sqrt(np.mean(y[20:-20, 1] ** 2))), 0.01)

    def test_invalid_shape_and_nonfinite_input_are_rejected(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, "1250"):
            module.downsample_iq(np.zeros((250, 2)))
        x = np.zeros((1250, 2)); x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            module.downsample_iq(x)

    def test_stable_channel_stats_do_not_depend_on_batch_partition(self):
        x = 1e8 + np.arange(2000, dtype=np.float64).reshape(4, 250, 2) / 1000
        module = self.module()
        stats = module.fit_channel_stats([x[:1], x[1:3], x[3:]])
        np.testing.assert_allclose(stats["mean"], x.mean(axis=(0, 1)), atol=1e-7)
        np.testing.assert_allclose(stats["std"], x.std(axis=(0, 1)), rtol=1e-7)
        self.assertEqual(stats["count_per_channel"], 1000)
        single = module.fit_channel_stats([x])
        np.testing.assert_allclose(stats["std"], single["std"], rtol=1e-7)

    def test_standardize_uses_supplied_train_statistics(self):
        module = self.module()
        train = np.arange(2000, dtype=np.float32).reshape(4, 250, 2)
        stats = module.fit_channel_stats([train])
        normalized = module.standardize(train, stats)
        np.testing.assert_allclose(normalized.mean(axis=(0, 1)), [0, 0], atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=(0, 1)), [1, 1], atol=1e-6)
        shifted = module.standardize(train + 1000, stats)
        self.assertGreater(float(shifted.mean()), 1)

    def test_empty_and_zero_variance_train_statistics_are_rejected(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, "empty"):
            module.fit_channel_stats([])
        with self.assertRaisesRegex(ValueError, "variance"):
            module.fit_channel_stats([np.ones((2, 250, 2))])
        with self.assertRaisesRegex(ValueError, "variance"):
            module.fit_channel_stats([np.full((4, 250, 2), 0.1)])
        with self.assertRaisesRegex(ValueError, "variance"):
            module.fit_channel_stats([np.full((2, 250, 2), 0.1)] * 2)
