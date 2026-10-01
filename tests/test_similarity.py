import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class SimilarityTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("src.ect.similarity")
        except ModuleNotFoundError:
            self.fail("Independent similarity module is missing")

    def test_copies_and_relative_raw_distance(self):
        q = np.array([[[1., 2.], [3., 6.], [5., 4.]]])
        refs = np.concatenate([q + 0.0001, q + 4, q], axis=0)
        result = self.module().nearest_neighbors(q, refs, reference_block=1)
        self.assertEqual(result.raw_index.tolist(), [2])
        self.assertEqual(result.raw_distance.tolist(), [0.])
        self.assertEqual(result.raw_candidate_count.tolist(), [2])
        one = self.module().nearest_neighbors(q, refs[:1])
        expected = np.linalg.norm(q - refs[:1]) / np.linalg.norm(q - q.mean(axis=1, keepdims=True))
        self.assertAlmostEqual(one.raw_distance[0], expected, places=14)

    def test_offset_and_joint_positive_scale_preserve_shape(self):
        q = np.array([[[1., 2.], [3., 6.], [5., 4.]]])
        refs = q * 3 + [200., -500.]
        result = self.module().nearest_neighbors(q, refs)
        self.assertLess(result.shape_distance[0], 1e-14)
        self.assertEqual(result.shape_candidate_count.tolist(), [1])
        self.assertGreater(result.raw_distance[0], .001)

    def test_shape_does_not_allow_sign_shift_or_separate_channel_scale(self):
        q = np.array([[[1., 2.], [3., 6.], [5., 4.], [2., 7.]]])
        for refs in (-q, np.roll(q, 1, axis=1), q * [2., 1.]):
            with self.subTest(refs=refs):
                result = self.module().nearest_neighbors(q, refs)
                self.assertGreater(result.shape_distance[0], .01)
                self.assertEqual(result.shape_candidate_count.tolist(), [0])

    def test_zero_ac_is_not_applicable_and_constant_references_excluded(self):
        variable = np.array([[[0., 0.], [1., 2.], [2., 0.]]])
        constant = np.broadcast_to([1e12, -1e12], variable.shape).copy()
        queries = np.concatenate([constant, variable])
        refs = np.concatenate([constant, variable])
        result = self.module().nearest_neighbors(queries, refs)
        self.assertEqual(result.raw_index.tolist(), [-1, 1])
        self.assertEqual(result.shape_index.tolist(), [-1, 1])
        self.assertTrue(np.isnan(result.raw_distance[0]))
        self.assertTrue(np.isnan(result.shape_distance[0]))
        self.assertEqual(result.shape_candidate_count.tolist(), [0, 1])
        no_refs = self.module().nearest_neighbors(variable, constant)
        self.assertEqual(no_refs.shape_index.tolist(), [-1])
        self.assertTrue(np.isnan(no_refs.shape_distance[0]))
        self.assertTrue(np.isfinite(no_refs.raw_distance[0]))

    def test_large_dc_small_difference_uses_actual_difference_and_stable_mean(self):
        q = 1e12 + np.array([[[0., 2.], [.001, -.003], [.005, .009], [-.004, .001]]])
        refs = np.concatenate([q + .002, q + .00025])
        result = self.module().nearest_neighbors(q, refs)
        centered = (q - q[:, :1]) - (q - q[:, :1]).mean(axis=1, keepdims=True)
        expected = np.linalg.norm(q - refs[1:2]) / np.linalg.norm(centered)
        self.assertEqual(result.raw_index.tolist(), [1])
        self.assertAlmostEqual(result.raw_distance[0], expected, places=14)
        self.assertGreater(result.raw_distance[0], 0)
        self.assertEqual(self.module().center_iq(q).tolist(), centered.tolist())

    def test_block_invariance_and_smallest_index_ties_against_exhaustive_oracle(self):
        rng = np.random.default_rng(25)
        queries = rng.normal(size=(9, 17, 2)) + [1e6, -1e6]
        refs = rng.normal(size=(23, 17, 2)) + [1e6, -1e6]
        refs[2] = queries[0]
        refs[19] = refs[2]
        queries[-1] = 5
        refs[-1] = 2
        before = queries.copy()
        module = self.module()
        centered_q = module.center_iq(queries).reshape(9, -1)
        centered_r = module.center_iq(refs).reshape(23, -1)
        qnorm = np.linalg.norm(centered_q, axis=1)
        rnorm = np.linalg.norm(centered_r, axis=1)
        for qb, rb in ((1, 1), (4, 7), (50, 50)):
            result = module.nearest_neighbors(queries, refs, query_block=qb, reference_block=rb)
            self.assertEqual(result.shape_index[0], 2)
            self.assertEqual(result.raw_index[0], 2)
            for i in range(8):
                raw = np.linalg.norm((queries[i] - refs).reshape(len(refs), -1), axis=1) / qnorm[i]
                valid = np.flatnonzero(rnorm > 0)
                shape = np.linalg.norm(centered_q[i] / qnorm[i] - centered_r[valid] / rnorm[valid, None], axis=1)
                self.assertEqual(result.raw_index[i], np.argmin(raw))
                self.assertEqual(result.shape_index[i], valid[np.argmin(shape)])
                self.assertAlmostEqual(result.raw_distance[i], np.min(raw), places=13)
                self.assertAlmostEqual(result.shape_distance[i], np.min(shape), places=13)
            if qb == 1:
                first = result
            else:
                for name in ("raw_index", "raw_distance", "shape_index", "shape_distance", "raw_candidate_count", "shape_candidate_count"):
                    np.testing.assert_array_equal(getattr(first, name), getattr(result, name))
        np.testing.assert_array_equal(queries, before)

    def test_shape_near_ties_are_recomputed_instead_of_rounded_to_zero(self):
        rng = np.random.default_rng(2)
        q = rng.normal(size=(1, 250, 2))
        refs = np.repeat(q, 4, axis=0)
        refs[0, 15, 0] += 1e-9
        refs[1, 15, 0] += 1e-10
        refs[2, 15, 0] += 1e-11
        refs[3] = refs[2]
        for block in (1, 2, 9):
            result = self.module().nearest_neighbors(q, refs, reference_block=block)
            self.assertEqual(result.shape_index.tolist(), [2])
            self.assertGreater(result.shape_distance[0], 0)

    def test_inclusive_threshold_counts_all_pairs_with_direct_distance(self):
        q = np.array([[[0., 1.], [2., 0.], [-1., -2.]]])
        refs = np.concatenate([q + .01, q + .01, q + .02])
        module = self.module()
        raw = np.linalg.norm(q - refs[:1]) / np.linalg.norm(module.center_iq(q))
        exact_shape = np.linalg.norm(module.center_iq(q) / np.linalg.norm(module.center_iq(q))
                                     - module.center_iq(refs[:1]) / np.linalg.norm(module.center_iq(refs[:1])))
        for block in (1, 10):
            result = module.nearest_neighbors(q, refs, raw_threshold=raw, shape_threshold=exact_shape, reference_block=block)
            self.assertEqual(result.raw_candidate_count.tolist(), [2])
            self.assertGreaterEqual(result.shape_candidate_count[0], 2)
            self.assertEqual(result.raw_index.tolist(), [0])

    def test_invalid_inputs_and_options_are_rejected(self):
        good = np.ones((2, 4, 2))
        for bad in (np.ones((4, 2)), np.ones((1, 4, 3)), np.empty((0, 4, 2)), np.ones((2, 0, 2)), good.astype(complex), np.full_like(good, np.nan), np.full_like(good, np.inf), np.array([[['x', 'y']]])):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.module().nearest_neighbors(bad, good)
        with self.assertRaises(ValueError):
            self.module().nearest_neighbors(good, np.ones((2, 5, 2)))
        for opts in ({"query_block": 0}, {"reference_block": 1.5}, {"query_block": True}, {"raw_threshold": -1}, {"shape_threshold": np.nan}, {"raw_threshold": np.inf}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                self.module().nearest_neighbors(good, good, **opts)


class SimilarityCLITests(unittest.TestCase):
    def run_fixture(self, root, *options):
        script = Path(__file__).resolve().parents[1] / "scripts/diagnose_similarity.py"
        return subprocess.run([sys.executable, str(script), "--project-root", str(root), "--cache", "cache", "--output", "results/similarity/check", *options], cwd="/tmp", capture_output=True, text=True)

    def test_cli_is_cwd_independent_traces_candidates_and_never_opens_test_signal(self):
        from fixture_data import make_dataset
        from src.ect.prepare import prepare_dataset
        script = Path(__file__).resolve().parents[1] / "scripts/diagnose_similarity.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, folder = make_dataset(root)
            prepare_dataset(root, folder, "cache")
            # No test signal file at all: CLI must succeed using only train/validation.
            (root / "cache/test_x.npy").unlink()
            (root / "cache/test_y.npy").unlink()
            cmd = [sys.executable, str(script), "--project-root", str(root), "--cache", "cache", "--output", "results/similarity/check", "--query-block", "1", "--reference-block", "1"]
            run = subprocess.run(cmd, cwd="/tmp", capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            output = root / "results/similarity/check"
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["query_count"], 2)
            self.assertEqual(report["reference_count"], 2)
            self.assertEqual(report["signal_splits_read"], ["train", "validation"])
            self.assertEqual(report["thresholds"], {"raw": .001, "shape": .01})
            self.assertIn("src/ect/similarity.py", report["code_sha256"])
            self.assertIn("scripts/diagnose_similarity.py", report["code_sha256"])
            self.assertEqual(len(report["cache_metadata_sha256"]), 64)
            self.assertEqual(report["shape"]["candidate_query_count"], 2)
            self.assertTrue((output / "candidates.png").exists())
            with (output / "nearest.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(len(row["query_wave_sha256"]) == 64 for row in rows))
            self.assertTrue(all(json.loads(row["query_origins"]) for row in rows))
            self.assertTrue(all(len(row["shape_reference_manifest_sha256"]) == 64 for row in rows))
            snapshot = (output / "summary.json").read_bytes()
            again = subprocess.run(cmd, cwd="/tmp", capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((output / "summary.json").read_bytes(), snapshot)

    def test_changed_cache_is_rejected_and_failure_recorded(self):
        from fixture_data import make_dataset
        from src.ect.prepare import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, folder = make_dataset(root)
            prepare_dataset(root, folder, "cache")
            with (root / "cache/validation_x.npy").open("ab") as handle:
                handle.write(b"tampered")
            run = self.run_fixture(root)
            self.assertNotEqual(run.returncode, 0)
            failure = json.loads((root / "results/similarity/check/failure.json").read_text())
            self.assertIn("checksum", failure["message"])
            self.assertFalse((root / "results/similarity/check/summary.json").exists())

    def test_zero_ac_report_has_null_quantiles_and_no_figure(self):
        from fixture_data import make_dataset, sha
        from src.ect.prepare import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, folder = make_dataset(root)
            prepare_dataset(root, folder, "cache")
            path = root / "cache/validation_x.npy"
            np.save(path, np.ones((2, 250, 2), dtype=np.float32))
            metadata_path = root / "cache/metadata.json"
            meta = json.loads(metadata_path.read_text())
            meta["files_sha256"][path.name] = sha(path)
            metadata_path.write_text(json.dumps(meta))
            run = self.run_fixture(root)
            self.assertEqual(run.returncode, 0, run.stderr)
            output = root / "results/similarity/check"
            report = json.loads((output / "summary.json").read_text())
            self.assertFalse(report["figure_created"])
            self.assertFalse((output / "candidates.png").exists())
            self.assertEqual(report["shape"]["not_applicable_query_count"], 2)
            self.assertTrue(all(value is None for value in report["raw"]["nearest_distance_quantiles"].values()))
            with (output / "nearest.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["shape_reference_row"], "-1")
            self.assertEqual(rows[0]["shape_distance"], "")
            self.assertEqual(rows[0]["shape_candidate"], "False")


if __name__ == "__main__":
    unittest.main()
