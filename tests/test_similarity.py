import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
    def run_fixture(self, root, *options, output="results/similarity/check"):
        script = Path(__file__).resolve().parents[1] / "scripts/diagnose_similarity.py"
        cmd = [sys.executable, str(script), "--project-root", str(root), "--cache", "cache"]
        if output is not None:
            cmd.extend(["--output", output])
        return subprocess.run([*cmd, *options], cwd="/tmp", capture_output=True, text=True)

    def test_test_queries_use_only_selected_signals_and_trace_selected_identities(self):
        from fixture_data import make_dataset, sha
        from src.ect.prepare import prepare_dataset
        for reference_split, excluded in (("train", "validation"), ("validation", "train")):
            with self.subTest(reference_split=reference_split), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, folder = make_dataset(root)
                prepare_dataset(root, folder, "cache")
                rng = np.random.default_rng(77)
                references = rng.normal(size=(2, 250, 2)).astype(np.float32)
                meta_path = root / "cache/metadata.json"
                meta = json.loads(meta_path.read_text())
                for split, signals in ((reference_split, references), ("test", references[::-1])):
                    path = root / f"cache/{split}_x.npy"
                    np.save(path, signals)
                    meta["files_sha256"][path.name] = sha(path)
                meta_path.write_text(json.dumps(meta))
                # Missing unselected signals/manifests and raw inputs must not matter.
                for kind in ("x", "y"):
                    (root / f"cache/{excluded}_{kind}.npy").unlink()
                (folder / f"{excluded}.jsonl").unlink()
                (root / "data/raw/fixture.npy").unlink()
                run = self.run_fixture(root, "--query-split", "test", "--reference-split", reference_split,
                                       "--query-block", "1", "--reference-block", "1")
                self.assertEqual(run.returncode, 0, run.stderr)
                output = root / "results/similarity/check"
                report = json.loads((output / "summary.json").read_text())
                self.assertEqual(report["query_split"], "test")
                self.assertEqual(report["reference_split"], reference_split)
                self.assertEqual(report["signal_splits_read"], [reference_split, "test"])
                self.assertEqual((report["query_count"], report["reference_count"], report["exhaustive_pair_count"]), (2, 2, 4))
                self.assertEqual(report["cache"], str(root / "cache"))
                self.assertEqual(report["cache_metadata_sha256"], sha(meta_path))
                self.assertEqual(report["cache_files_sha256"], {
                    f"{split}_{kind}.npy": meta["files_sha256"][f"{split}_{kind}.npy"]
                    for split in (reference_split, "test") for kind in ("x", "y")})
                self.assertEqual(report["manifest_sha256"], {
                    f"{split}.jsonl": meta["manifest_sha256"][f"{split}.jsonl"]
                    for split in (reference_split, "test")})
                self.assertIn(f"{reference_split}/test manifests", report["integrity_scope"])
                self.assertIn(f"No {excluded} signal", report["integrity_scope"])
                self.assertIn("raw input file", report["integrity_scope"])
                self.assertIn("test", report["interpretation"])
                self.assertIn(reference_split, report["interpretation"])
                self.assertIn("No split changes", report["interpretation"])
                self.assertIn("test classification metrics", report["interpretation"])
                query_rows = [json.loads(line) for line in (folder / "test.jsonl").read_text().splitlines()]
                ref_rows = [json.loads(line) for line in (folder / f"{reference_split}.jsonl").read_text().splitlines()]
                with (output / "nearest.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)
                for qi, row in enumerate(rows):
                    self.assertEqual(row["query_split"], "test")
                    self.assertEqual(row["reference_split"], reference_split)
                    self.assertEqual(int(row["query_row"]), qi)
                    self.assertEqual(row["query_manifest_sha256"], sha(folder / "test.jsonl"))
                    self.assertEqual(row["query_wave_sha256"], query_rows[qi]["wave_sha256"])
                    self.assertEqual(int(row["query_class_index"]), query_rows[qi]["class_index"])
                    self.assertEqual(json.loads(row["query_representative"]), query_rows[qi]["representative"])
                    self.assertEqual(json.loads(row["query_origins"]), query_rows[qi]["origins"])
                    for kind in ("raw", "shape"):
                        ri = 1 - qi
                        self.assertEqual(row[f"{kind}_reference_split"], reference_split)
                        self.assertEqual(int(row[f"{kind}_reference_row"]), ri)
                        self.assertEqual(float(row[f"{kind}_distance"]), 0)
                        self.assertEqual(row[f"{kind}_candidate_pair_count"], "1")
                        self.assertEqual(row[f"{kind}_reference_manifest_sha256"], sha(folder / f"{reference_split}.jsonl"))
                        self.assertEqual(row[f"{kind}_reference_wave_sha256"], ref_rows[ri]["wave_sha256"])
                        self.assertEqual(int(row[f"{kind}_reference_class_index"]), ref_rows[ri]["class_index"])
                        self.assertEqual(json.loads(row[f"{kind}_reference_representative"]), ref_rows[ri]["representative"])
                        self.assertEqual(json.loads(row[f"{kind}_reference_origins"]), ref_rows[ri]["origins"])
                        self.assertEqual(row[f"{kind}_same_class"], "False")
                self.assertTrue(report["figure_created"])

    def test_invalid_split_pairs_are_rejected_before_creating_output(self):
        pairs = (("validation", "validation"), ("train", "train"), ("train", "validation"),
                 ("test", "test"), ("validation", "test"))
        for query, reference in pairs:
            with self.subTest(query=query, reference=reference), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.run_fixture(root, "--query-split", query, "--reference-split", reference)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertNotIn("unrecognized arguments", run.stderr)
                self.assertIn("split", run.stderr)
                self.assertFalse((root / "results").exists())

    def test_default_output_is_named_reference_then_query(self):
        from fixture_data import make_dataset
        from src.ect.prepare import prepare_dataset
        cases = (((), "train_validation_v1", "validation", "train"),
                 (("--query-split", "test"), "train_test_v1", "test", "train"),
                 (("--query-split", "test", "--reference-split", "validation"),
                  "validation_test_v1", "test", "validation"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, folder = make_dataset(root)
            prepare_dataset(root, folder, "cache")
            for options, name, query, reference in cases:
                with self.subTest(output=name):
                    run = self.run_fixture(root, *options, output=None)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    report = json.loads((root / f"results/similarity/{name}/summary.json").read_text())
                    self.assertEqual(report["query_split"], query)
                    self.assertEqual(report["reference_split"], reference)

    def test_selected_cache_and_manifest_tampering_are_rejected(self):
        from fixture_data import make_dataset
        from src.ect.prepare import prepare_dataset
        for reference in ("train", "validation"):
            for relative in ("cache/test_x.npy", "cache/test_y.npy", f"cache/{reference}_x.npy",
                             f"cache/{reference}_y.npy", "manifests/fixture/test.jsonl",
                             f"manifests/fixture/{reference}.jsonl", "manifests/fixture/summary.json"):
                with self.subTest(reference=reference, path=relative), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    _, folder = make_dataset(root)
                    prepare_dataset(root, folder, "cache")
                    with (root / relative).open("ab") as handle:
                        handle.write(b"tampered")
                    run = self.run_fixture(root, "--query-split", "test", "--reference-split", reference)
                    self.assertNotEqual(run.returncode, 0)
                    output = root / "results/similarity/check"
                    self.assertTrue((output / "failure.json").exists(), run.stderr)
                    failure = json.loads((output / "failure.json").read_text())
                    self.assertIn("checksum", failure["message"].lower())
                    self.assertFalse((output / "summary.json").exists())

    def test_unselected_entries_are_still_required_in_full_checksum_table(self):
        from fixture_data import make_dataset
        from src.ect.prepare import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, folder = make_dataset(root)
            prepare_dataset(root, folder, "cache")
            path = root / "cache/metadata.json"
            meta = json.loads(path.read_text())
            del meta["files_sha256"]["train_x.npy"]
            path.write_text(json.dumps(meta))
            run = self.run_fixture(root, "--query-split", "test", "--reference-split", "validation")
            self.assertNotEqual(run.returncode, 0)
            output = root / "results/similarity/check"
            self.assertTrue((output / "failure.json").exists(), run.stderr)
            failure = json.loads((output / "failure.json").read_text())
            self.assertIn("checksum table must be complete", failure["message"])
            self.assertFalse((output / "summary.json").exists())

    def test_manifest_identity_is_bound_to_summary_and_labels_match_selected_rows(self):
        from fixture_data import make_dataset, sha
        from src.ect.prepare import prepare_dataset
        for mutation in ("identity", "labels", "row_split"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, folder = make_dataset(root)
                prepare_dataset(root, folder, "cache")
                meta_path = root / "cache/metadata.json"
                meta = json.loads(meta_path.read_text())
                if mutation == "identity":
                    meta["manifest_sha256"]["test.jsonl"] = "0" * 64
                    message = "provenance"
                elif mutation == "labels":
                    path = root / "cache/test_y.npy"
                    np.save(path, np.array([1, 0], dtype=np.int64))
                    meta["files_sha256"][path.name] = sha(path)
                    message = "labels"
                else:
                    path = folder / "test.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["split"] = "validation"
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    meta["manifest_sha256"][path.name] = sha(path)
                    summary_path = folder / "summary.json"
                    summary = json.loads(summary_path.read_text())
                    summary["manifest_sha256"] = meta["manifest_sha256"]
                    summary_path.write_text(json.dumps(summary))
                    meta["summary_sha256"] = sha(summary_path)
                    message = "count/split"
                meta_path.write_text(json.dumps(meta))
                run = self.run_fixture(root, "--query-split", "test", "--reference-split", "validation")
                self.assertNotEqual(run.returncode, 0)
                output = root / "results/similarity/check"
                self.assertTrue((output / "failure.json").exists(), run.stderr)
                failure = json.loads((output / "failure.json").read_text())
                self.assertIn(message, failure["message"])
                self.assertFalse((output / "summary.json").exists())

    def test_test_zero_ac_rows_keep_selected_split_identity(self):
        from fixture_data import make_dataset, sha
        from src.ect.prepare import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, folder = make_dataset(root)
            prepare_dataset(root, folder, "cache")
            path = root / "cache/test_x.npy"
            np.save(path, np.ones((2, 250, 2), dtype=np.float32))
            meta_path = root / "cache/metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["files_sha256"][path.name] = sha(path)
            meta_path.write_text(json.dumps(meta))
            run = self.run_fixture(root, "--query-split", "test", "--reference-split", "validation")
            self.assertEqual(run.returncode, 0, run.stderr)
            output = root / "results/similarity/check"
            report = json.loads((output / "summary.json").read_text())
            self.assertFalse(report["figure_created"])
            self.assertEqual(report["raw"]["not_applicable_query_count"], 2)
            self.assertEqual(report["shape"]["not_applicable_query_count"], 2)
            with (output / "nearest.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                self.assertEqual(row["query_split"], "test")
                self.assertEqual(row["reference_split"], "validation")
                for kind in ("raw", "shape"):
                    self.assertEqual(row[f"{kind}_reference_split"], "validation")
                    self.assertEqual(row[f"{kind}_reference_manifest_sha256"], sha(folder / "validation.jsonl"))
                    self.assertEqual(row[f"{kind}_reference_row"], "-1")
                    self.assertEqual(row[f"{kind}_distance"], "")
                    self.assertEqual(row[f"{kind}_reference_wave_sha256"], "")

    def test_plot_uses_selected_split_titles_and_labels(self):
        cli = importlib.import_module("scripts.diagnose_similarity")
        with tempfile.TemporaryDirectory() as config, patch.dict("os.environ", {"MPLCONFIGDIR": config}):
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.axes import Axes
            from matplotlib.figure import Figure
        refs = np.random.default_rng(1).normal(size=(2, 250, 2))
        result = importlib.import_module("src.ect.similarity").nearest_neighbors(refs, refs)
        titles, labels = [], []
        axis_title, figure_title, axis_plot = Axes.set_title, Figure.suptitle, Axes.plot
        def record_axis_title(axis, title, *args, **kwargs):
            titles.append(title)
            return axis_title(axis, title, *args, **kwargs)
        def record_figure_title(figure, title, *args, **kwargs):
            titles.append(title)
            return figure_title(figure, title, *args, **kwargs)
        def record_plot(axis, *args, **kwargs):
            labels.append(kwargs.get("label", ""))
            return axis_plot(axis, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Axes, "set_title", record_axis_title), \
                patch.object(Figure, "suptitle", record_figure_title), patch.object(Axes, "plot", record_plot):
            path = Path(tmp) / "candidates.png"
            try:
                plotted = cli._plot(path, result, refs, refs, "test", "validation")
            except TypeError as exc:
                self.fail(f"Plot must support selected splits: {exc}")
            self.assertTrue(plotted)
            self.assertTrue(path.exists())
        self.assertTrue(any("test" in title and "validation" in title for title in titles))
        self.assertTrue(any("test row" in title and "validation row" in title for title in titles))
        self.assertIn("test ch0", labels)
        self.assertIn("validation ch0", labels)
        self.assertFalse(any("train" in text for text in titles + labels))

    def test_json_uses_explicit_lf_under_windows_default_translation(self):
        cli = importlib.import_module("scripts.diagnose_similarity")
        write_text = Path.write_text
        def windows_default(path, text, *args, **kwargs):
            if kwargs.get("newline") != "\n":
                text = text.replace("\n", "\r\n")
            return write_text(path, text, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, "write_text", windows_default):
            path = Path(tmp) / "summary.json"
            cli._json(path, {"query_split": "test", "reference_split": "validation"})
            content = path.read_bytes()
            self.assertNotIn(b"\r", content)
            self.assertTrue(content.endswith(b"\n"))
            self.assertEqual(json.loads(content)["query_split"], "test")

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
