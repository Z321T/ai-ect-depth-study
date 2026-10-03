"""Independent numeric and boundary checks for the full250 DC/AC protocol."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


def hand_fixture():
    """Wave means (2,10), (6,14), (10,18); alternating AC amplitudes."""
    sign = np.tile([-1.0, 1.0], 125)
    means = np.array([[2, 10], [6, 14], [10, 18]], dtype=np.float64)
    amplitudes = np.array([[1, 2], [2, 4], [3, 6]], dtype=np.float64)
    return means[:, None, :] + sign[None, :, None] * amplitudes[:, None, :]


class DcAcTests(unittest.TestCase):
    def module(self):
        try:
            from src.ect import dc_ac
        except ImportError:
            self.fail("DC/AC implementation is missing")
        return dc_ac

    def test_fit_matches_hand_calculated_population_statistics(self):
        # Catches pointwise variance, ddof=1, and per-wave AC normalization.
        stats = self.module().fit_dc_ac(hand_fixture(), batch_size=2)
        np.testing.assert_allclose(stats["mu_d"], [6, 14], rtol=0, atol=1e-14)
        np.testing.assert_allclose(stats["sigma_d"], [math.sqrt(32 / 3)] * 2,
                                   rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(stats["sigma_a"],
                                   [math.sqrt(14 / 3), math.sqrt(56 / 3)],
                                   rtol=1e-14, atol=1e-14)
        self.assertEqual(stats["sample_count"], 3)
        self.assertEqual(stats["points_per_channel"], 750)
        self.assertEqual(stats["channels"], 2)
        self.assertEqual(stats["signal_length"], 250)

    def test_transform_matches_hand_values_and_preserves_dc_and_relative_ac(self):
        # Removing DC, missing sqrt2, or scaling each wave separately must fail.
        module = self.module()
        raw = hand_fixture()
        z = module.transform_dc_ac(raw, module.fit_dc_ac(raw))
        self.assertEqual(z.shape, (3, 250, 2))
        self.assertEqual(z.dtype, np.float32)
        np.testing.assert_allclose(z.mean(axis=1, dtype=np.float64),
                                   [[-math.sqrt(3) / 2] * 2, [0, 0],
                                    [math.sqrt(3) / 2] * 2], rtol=0, atol=2e-6)
        np.testing.assert_allclose(z[:, 1] - z[:, 0],
                                   np.repeat(np.array([[math.sqrt(3 / 7)],
                                                       [2 * math.sqrt(3 / 7)],
                                                       [3 * math.sqrt(3 / 7)]]), 2, axis=1),
                                   rtol=2e-7, atol=2e-7)
        expected_first = (-4 / math.sqrt(32 / 3) - 1 / math.sqrt(14 / 3)) / math.sqrt(2)
        self.assertAlmostEqual(float(z[0, 0, 0]), expected_first, delta=1e-7)
        np.testing.assert_allclose(z.std(axis=(0, 1), dtype=np.float64), [1, 1],
                                   rtol=0, atol=2e-6)

    def test_fit_partition_is_stable_for_large_dc_and_small_variance(self):
        # Raw second moments or merging uncentered means loses small variance.
        module = self.module()
        index = np.arange(37, dtype=np.float64)
        means = np.stack([1e12 + index / 4, 1e12 - index / 2], axis=-1)
        raw = means[:, None, :] + np.tile([-0.25, 0.25], 125)[None, :, None]
        for batch_size in (1, 2, 7, 256):
            with self.subTest(batch_size=batch_size):
                stats = module.fit_dc_ac(raw, batch_size=batch_size)
                np.testing.assert_allclose(stats["mu_d"], [1e12 + 4.5, 1e12 - 9],
                                           rtol=0, atol=0)
                np.testing.assert_allclose(stats["sigma_d"],
                                           [math.sqrt(7.125), math.sqrt(28.5)],
                                           rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(stats["sigma_a"], [0.25, 0.25],
                                           rtol=0, atol=1e-14)

    def test_transform_partition_and_order_are_exactly_invariant(self):
        module = self.module()
        raw = hand_fixture()
        stats = module.fit_dc_ac(raw)
        together = module.transform_dc_ac(raw, stats)
        separate = np.concatenate([module.transform_dc_ac(raw[i:i + 1], stats)
                                   for i in range(len(raw))])
        np.testing.assert_array_equal(together, separate)
        np.testing.assert_array_equal(module.transform_dc_ac(raw[::-1], stats), together[::-1])

    def test_memmap_streaming_does_not_mutate_data_or_statistics(self):
        module = self.module()
        raw = hand_fixture().astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npy"
            np.save(path, raw)
            mapped = np.load(path, mmap_mode="r")
            try:
                stats = module.fit_dc_ac(mapped, batch_size=1)
                before = copy.deepcopy(stats)
                restored = json.loads(json.dumps(stats, allow_nan=False))
                module.validate_stats(restored)
                z = module.transform_dc_ac(mapped, restored)
                np.testing.assert_array_equal(mapped, raw)
                np.testing.assert_array_equal(z, module.transform_dc_ac(raw, stats))
                self.assertEqual(stats, before)
                self.assertEqual(restored, stats)
            finally:
                mapped._mmap.close()

    def test_validation_uses_frozen_train_stats_and_retains_large_shift(self):
        module = self.module()
        raw = hand_fixture()
        stats = module.fit_dc_ac(raw)
        before = copy.deepcopy(stats)
        z = module.transform_dc_ac(raw, stats)
        shifted = module.transform_dc_ac(raw + np.array([100, -200]), stats)
        np.testing.assert_allclose((shifted.astype(np.float64) - z).mean(axis=(0, 1)),
                                   [100 / math.sqrt(64 / 3), -200 / math.sqrt(64 / 3)],
                                   rtol=2e-7, atol=2e-6)
        self.assertEqual(stats, before)

    def test_full250_mean_is_used_before_a_biased224_crop(self):
        # Re-estimating DC on cropped data would give an incorrect offset.
        module = self.module()
        stats = module.fit_dc_ac(hand_fixture())
        raw = np.zeros((1, 250, 2), dtype=np.float64)
        raw[:, :26] = 250
        z = module.transform_dc_ac(raw, stats)
        self.assertEqual(z[:, 26:].shape, (1, 224, 2))
        expected = ((26 - 6) / math.sqrt(32 / 3) - 26 / math.sqrt(14 / 3)) / math.sqrt(2)
        np.testing.assert_allclose(z[:, 26:, 0], expected, rtol=1e-7, atol=1e-7)
        with self.assertRaises(ValueError):
            module.transform_dc_ac(raw[:, 26:], stats)

    def test_zero_scales_are_floored_and_produce_finite_outputs(self):
        module = self.module()
        raw = np.broadcast_to([5.0, -2.0], (2, 250, 2)).copy()
        stats = module.fit_dc_ac(raw, batch_size=1)
        self.assertEqual(stats["sigma_d"], [0, 0])
        self.assertEqual(stats["sigma_a"], [0, 0])
        self.assertEqual(stats["scale_d"], [1e-12, 1e-12])
        self.assertEqual(stats["scale_a"], [1e-12, 1e-12])
        self.assertEqual(stats["floor_d"], [True, True])
        self.assertEqual(stats["floor_a"], [True, True])
        module.validate_stats(stats)
        np.testing.assert_array_equal(module.transform_dc_ac(raw, stats), np.zeros_like(raw))
        self.assertTrue(np.isfinite(module.transform_dc_ac(raw + 1, stats)).all())

    def test_each_channel_and_component_is_floored_independently(self):
        module = self.module()
        raw = np.zeros((2, 250, 2), dtype=np.float64)
        raw[:, :, 0] = np.array([0, 2])[:, None]
        raw[:, :, 1] = np.tile([-1.0, 1.0], 125)
        stats = module.fit_dc_ac(raw)
        self.assertEqual(stats["sigma_d"], [1, 0])
        self.assertEqual(stats["sigma_a"], [0, 1])
        self.assertEqual(stats["floor_d"], [False, True])
        self.assertEqual(stats["floor_a"], [True, False])
        np.testing.assert_allclose(module.transform_dc_ac(raw, stats)[0, :2],
                                   [[-1 / math.sqrt(2), -1 / math.sqrt(2)],
                                    [-1 / math.sqrt(2), 1 / math.sqrt(2)]],
                                   rtol=0, atol=1e-7)

    def test_fit_is_train_only_and_rejects_invalid_batch_sizes(self):
        module = self.module()
        for split in ("validation", "test", None, "Train"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                module.fit_dc_ac(hand_fixture(), fitted_split=split)
        for size in (0, -1, 1.5, True, "2", None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                module.fit_dc_ac(hand_fixture(), batch_size=size)

    def test_fit_and_transform_reject_empty_malformed_and_nonfinite_inputs(self):
        module = self.module()
        stats = module.fit_dc_ac(hand_fixture())
        invalid = [np.empty((0, 250, 2)), np.zeros((250, 2)), np.zeros((1, 224, 2)),
                   np.zeros((1, 250, 1)), np.zeros((1, 1, 250, 2)),
                   np.full((1, 250, 2), "1"), np.zeros((1, 250, 2), dtype=complex),
                   np.zeros((1, 250, 2), dtype=bool), np.zeros((1, 250, 2), dtype=object)]
        for value in (np.nan, np.inf, -np.inf):
            raw = hand_fixture()
            raw[-1, -1, -1] = value
            invalid.append(raw)
        for raw in invalid:
            with self.subTest(shape=raw.shape, dtype=raw.dtype), self.assertRaises(ValueError):
                module.fit_dc_ac(raw, batch_size=1)
            with self.subTest(shape=raw.shape, dtype=raw.dtype), self.assertRaises(ValueError):
                module.transform_dc_ac(raw, stats)

    def test_validate_stats_rejects_incomplete_nonjson_and_incoherent_schema(self):
        module = self.module()
        stats = module.fit_dc_ac(hand_fixture())
        module.validate_stats(stats)
        invalid = [None, [], "stats"]
        for key in stats:
            incomplete = copy.deepcopy(stats)
            del incomplete[key]
            invalid.append(incomplete)
        for key, value in [
                ("extra", 1), ("schema_version", 2), ("schema_version", True),
                ("schema_version", 1.0), ("protocol", "other"), ("fitted_split", "validation"),
                ("sample_count", 0), ("sample_count", True), ("sample_count", 3.0),
                ("points_per_channel", 749), ("points_per_channel", 750.0),
                ("channels", 1), ("channels", True), ("signal_length", 224),
                ("epsilon", 0), ("epsilon", 1e-8), ("epsilon", True),
                ("mu_d", [np.nan, 0]), ("mu_d", [0, np.inf]), ("mu_d", [True, 0]),
                ("mu_d", [0]), ("mu_d", (6, 14)), ("mu_d", ["6", 14]),
                ("mu_d", np.array([6, 14])), ("sigma_d", [-1, 1]),
                ("sigma_a", [1, -1]), ("sigma_a", [np.inf, 1]),
                ("scale_d", [0, 1]), ("scale_a", [1, np.nan]),
                ("scale_a", [1, 1]), ("scale_d", [1, 1]),
                ("floor_d", [True, False]), ("floor_a", [False, 0]),
                ("floor_d", [False]), ("floor_a", (False, False))]:
            malformed = copy.deepcopy(stats)
            malformed[key] = value
            invalid.append(malformed)
        for index, malformed in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                module.validate_stats(malformed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                module.transform_dc_ac(hand_fixture(), malformed)


if __name__ == "__main__":
    unittest.main()
