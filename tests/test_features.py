import unittest
import numpy as np


class FeatureTests(unittest.TestCase):
    def module(self):
        from src.ect import features
        return features

    def test_known_statistics_and_closed_triangle_geometry(self):
        module = self.module()
        x = np.array([[[0., 0.], [1., 0.], [0., 1.]]])
        before = x.copy()
        result = module.extract_features(x)
        self.assertEqual(result.shape, (1, 25))
        values = dict(zip(module.FEATURE_NAMES, result[0]))
        self.assertAlmostEqual(values['I_mean'], 1 / 3)
        self.assertAlmostEqual(values['I_std'], np.sqrt(2 / 9))
        self.assertAlmostEqual(values['I_ptp'], 1.)
        self.assertAlmostEqual(values['IQ_covariance'], -1 / 9)
        self.assertAlmostEqual(values['IQ_correlation'], -0.5)
        self.assertAlmostEqual(values['trajectory_length'], 1 + np.sqrt(2))
        self.assertAlmostEqual(values['endpoint_distance'], 1.)
        self.assertAlmostEqual(values['signed_closed_area'], 0.5)
        np.testing.assert_array_equal(x, before)

    def test_constants_have_finite_statistics(self):
        module = self.module()
        result = module.extract_features(np.full((2, 250, 2), 0.1))
        values = dict(zip(module.FEATURE_NAMES, result[0]))
        self.assertTrue(np.isfinite(result).all())
        for name in ('I_std', 'I_skew', 'I_excess_kurtosis', 'IQ_correlation', 'signed_closed_area'):
            self.assertEqual(values[name], 0.)

    def test_geometric_features_keep_translation_and_scale_as_defined(self):
        module = self.module()
        t = np.linspace(0, 2 * np.pi, 250)
        x = np.stack((np.sin(t), np.cos(t)), axis=-1)[None]
        a = dict(zip(module.FEATURE_NAMES, module.extract_features(x)[0]))
        b = dict(zip(module.FEATURE_NAMES, module.extract_features(x * 3 + [10, -20])[0]))
        self.assertAlmostEqual(b['I_mean'], a['I_mean'] * 3 + 10)
        self.assertAlmostEqual(b['I_std'], a['I_std'] * 3)
        self.assertAlmostEqual(b['IQ_correlation'], a['IQ_correlation'])
        self.assertAlmostEqual(b['trajectory_length'], a['trajectory_length'] * 3)
        self.assertAlmostEqual(b['signed_closed_area'], a['signed_closed_area'] * 9)

    def test_batch_partition_is_consistent_and_invalid_inputs_rejected(self):
        module = self.module()
        x = np.random.default_rng(0).normal(size=(4, 250, 2))
        np.testing.assert_array_equal(module.extract_features(x),
                                      np.concatenate([module.extract_features(x[:1]), module.extract_features(x[1:])]))
        for bad in (np.zeros((250, 2)), np.zeros((1, 1, 2)), np.zeros((1, 250, 3)), np.full((1, 250, 2), np.nan)):
            with self.assertRaises(ValueError):
                module.extract_features(bad)
