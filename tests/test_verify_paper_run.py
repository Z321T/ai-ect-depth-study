import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixture_data import make_dataset
from src.ect.prepare import prepare_dataset


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n')


class VerifyPaperRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import verify_paper_run
        cls.verifier = verify_paper_run
        # The producer is used only to create a tiny real checkpoint/CSV fixture.
        from src.ect.paper_training import train_paper_baseline
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        _, manifests = make_dataset(cls.fixture.name)
        prepare_dataset(cls.fixture.name, manifests, 'cache')
        config = json.loads((ROOT / 'config/paper_baseline_v1.json').read_text())
        config.update(purpose='paper_runner_smoke_validation_only', seed=0, device='cpu',
                      max_epochs=2, validation_every_epochs=1, batch_size=2, cache_dir='cache')
        train_paper_baseline(cls.fixture.name, config, 'run')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(self.fixture.name, self.root, dirs_exist_ok=True)
        self.run = self.root / 'run'
        self.output = self.root / 'proof.json'
        self.report = json.loads((self.run / 'report.json').read_text())

    def verify(self, **kwargs):
        return self.verifier.verify_paper_run(self.root, self.run, self.output, **kwargs)

    def save_report(self):
        write(self.run / 'report.json', self.report)

    def refresh_artifact(self, name):
        self.report['files_sha256'][name] = sha(self.run / name)
        self.save_report()

    def assert_rejected(self, **kwargs):
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
            self.verify(**kwargs)
        self.assertFalse(self.output.exists(), 'Failure must never publish a success JSON')

    def test_verifier_api_exists(self):
        from scripts.verify_paper_run import verify_paper_run
        self.assertTrue(callable(verify_paper_run))

    def test_real_cpu_both_is_independent_and_read_only(self):
        before = {p.name: sha(p) for p in self.run.iterdir() if p.is_file()}
        from src.ect import paper_crops, paper_training
        # Poison every producer inference path. None can provide verification evidence.
        with (patch.object(paper_training, 'predict_paper', side_effect=AssertionError('producer')),
             patch.object(paper_training, 'load_paper_model', side_effect=AssertionError('loader')),
             patch.object(paper_crops, 'crop_starts', side_effect=AssertionError('crop producer')),
             patch.object(paper_crops, 'ten_crop_probabilities', side_effect=AssertionError('producer'))):
            result = self.verify(batch_size=3)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['prediction_counts'], {'train': 2, 'validation': 2})
        self.assertEqual(result['group_counts'], {'train': {'0': 2}, 'validation': {'1': 2}})
        self.assertLessEqual(result['maximum_metric_error'], 1e-12)
        self.assertEqual(result['device'], 'cpu')
        self.assertTrue(result['determinism']['deterministic_algorithms'])
        self.assertFalse(result['determinism']['tf32'])
        self.assertFalse(result['flags']['test_classification_evaluated'])
        self.assertEqual(result['script']['text'], Path(self.verifier.__file__).read_text())
        self.assertEqual(result['script']['sha256'], sha(self.verifier.__file__))
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(before, {p.name: sha(p) for p in self.run.iterdir() if p.is_file()})

    def test_validation_only_never_maps_train_or_test_for_inference(self):
        load = np.load
        opened = []
        def selected_only(path, *args, **kwargs):
            self.assertTrue(Path(path).name.startswith('validation_'))
            mapped = load(path, *args, **kwargs)
            opened.append(mapped)
            return mapped
        with patch.object(np, 'load', side_effect=selected_only):
            result = self.verify(split='validation', batch_size=1)
        self.assertEqual(result['prediction_counts'], {'validation': 2})
        self.assertTrue(all(m._mmap.closed for m in opened))
        self.assertEqual(set(result['metrics']), {'validation'})

    def test_determinism_is_set_before_checkpoint_load(self):
        self.assert_rejected(split='test')
        original = torch.load
        def check(*args, **kwargs):
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            return original(*args, **kwargs)
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        with patch.object(torch, 'load', side_effect=check):
            self.verify(split='train')

    def test_artifact_source_and_cache_tampering_rejected(self):
        for section, key in [('files_sha256', 'model.pt'), ('code_sha256', 'src/ect/paper_models.py'),
                             ('code_sha256_lf', 'src/ect/paper_models.py'),
                             ('manifest_sha256', 'train.jsonl')]:
            original = copy.deepcopy(self.report)
            self.report[section][key] = '0' * 64
            self.save_report()
            with self.subTest(section=section):
                self.assert_rejected()
            self.report = original
        self.report['cache_metadata_sha256'] = '0' * 64
        self.save_report()
        self.assert_rejected()

    def test_incomplete_or_failed_report_never_publishes_proof(self):
        for field, value in [('status', 'running'), ('heldout_test_evaluated', True),
                             ('model_roundtrip_verified', False), ('classes', [0, 2]),
                             ('files_sha256', {}), ('config_sha256', '0' * 64)]:
            original = copy.deepcopy(self.report)
            self.report[field] = value
            self.save_report()
            with self.subTest(field=field):
                self.assert_rejected()
            self.report = original
        self.save_report()
        write(self.run / 'failure.json', {'status': 'failed'})
        self.assert_rejected()

    def test_running_marker_blocks_complete_report_and_publish_race(self):
        marker = self.run / 'running.marker'
        marker.write_text('running')
        with self.assertRaisesRegex(ValueError, 'running|Running|Incomplete'):
            self.verify()
        self.assertFalse(self.output.exists())
        marker.unlink()
        original = self.verifier.independent_probabilities
        def interrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            marker.write_text('running')
            return result
        with patch.object(self.verifier, 'independent_probabilities', side_effect=interrupt):
            self.assert_rejected()

    def test_rows_and_csv_identity_are_verified_even_with_forged_checksums(self):
        for edit in ('duplicate', 'reordered', 'identity', 'label', 'prediction'):
            name = 'validation_predictions.csv'
            original = (self.run / name).read_bytes()
            with (self.run / name).open() as handle:
                rows = list(csv.DictReader(handle))
            if edit == 'duplicate':
                rows[1]['row_index'] = rows[0]['row_index']
            elif edit == 'reordered':
                rows.reverse()
            elif edit == 'identity':
                rows[0]['wave_sha256'] = 'f' * 64
            elif edit == 'label':
                rows[0]['class_index'] = str(1 - int(rows[0]['class_index']))
            else:
                rows[0]['predicted_class_index'] = str(1 - int(rows[0]['predicted_class_index']))
            with (self.run / name).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.refresh_artifact(name)
            with self.subTest(edit=edit):
                self.assert_rejected()
            (self.run / name).write_bytes(original)
            self.refresh_artifact(name)
        write(self.run / 'train_rows.json', [1, 0])
        self.refresh_artifact('train_rows.json')
        self.assert_rejected(split='validation')

    def test_metrics_and_group_metrics_are_recomputed(self):
        for group in (False, True):
            original = copy.deepcopy(self.report)
            metric = (self.report['group_metrics']['train']['0'] if group
                      else self.report['metrics']['train'])
            metric['per_class']['0']['precision'] += .125
            self.save_report()
            with self.subTest(group=group):
                self.assert_rejected()
            self.report = original

    def test_numpy_metrics_include_absent_classes_and_zero_division(self):
        value = self.verifier.numpy_metrics(np.array([0, 0, 1]), np.array([0, 2, 2]), 3)
        self.assertEqual(value['confusion_matrix'], [[1, 0, 1], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(value['per_class']['2']['support'], 0)
        self.assertEqual(value['per_class']['1']['precision'], 0)
        self.assertAlmostEqual(value['macro_f1'], 2 / 9)

    def test_literal_unicode_crop_starts_normalization_and_probability_average(self):
        starts = [[23, 16, 11, 18, 10, 8, 25, 24, 9, 7],
                  [2, 1, 21, 13, 24, 9, 18, 1, 9, 15]]
        raw = np.arange(2 * 250 * 2, dtype=np.float32).reshape(2, 250, 2) / 31
        mean, std = np.array([.12, -.77]), np.array([3.7, 1.23])
        captured = []
        class Recorder(torch.nn.Module):
            def forward(self, batch):
                captured.append(batch.cpu().numpy().copy())
                score = batch[:, 0, 0] - batch[:, 1, -1]
                return torch.stack((score, -score), dim=1)
        actual = self.verifier.independent_probabilities(
            Recorder(), raw, ['身份/alpha', 'beta'], mean, std, torch.device('cpu'), 3)
        expected_inputs = np.stack([((raw[i, start:start + 224].astype(np.float64) - mean) / std)
                                    .astype(np.float32).T for i in range(2) for start in starts[i]])
        np.testing.assert_array_equal(np.concatenate(captured), expected_inputs)
        scores = expected_inputs[:, 0, 0] - expected_inputs[:, 1, -1]
        expected = torch.softmax(torch.from_numpy(np.stack((scores, -scores), axis=1)), 1)
        np.testing.assert_allclose(actual, expected.numpy().reshape(2, 10, 2).mean(axis=1), rtol=1e-7)

    def test_marker_appearing_during_final_hash_check_blocks_publication(self):
        original = self.verifier.sha256
        report_calls = 0
        def late_marker(path):
            nonlocal report_calls
            if Path(path) == self.run / 'report.json':
                report_calls += 1
                if report_calls == 2:
                    (self.run / 'running.marker').write_text('incomplete')
            return original(path)
        with patch.object(self.verifier, 'sha256', side_effect=late_marker):
            self.assert_rejected()

    def test_loaded_forward_code_must_match_hashed_source(self):
        model = self.verifier.PaperResNeXt1D
        original = model.forward
        def same_predictions(self, inputs):
            return original(self, inputs)
        # Same behavior and unchanged disk SHA cannot hide a different loaded code object.
        with patch.object(model, 'forward', same_predictions):
            self.assert_rejected()

    def make_full_report(self):
        """Relabel tiny artifacts solely to test identity rejection, never train full."""
        config = json.loads((ROOT / 'config/paper_baseline_v1.json').read_text()) | {'seed': 0}
        cache = self.root / config['cache_dir']
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.root / 'cache', cache)
        self.report.update(config=config, purpose=config['purpose'], cache_path=config['cache_dir'],
                           config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True,
                               separators=(',', ':')).encode()).hexdigest(), selected_epoch=10,
                           execution_registration=None)
        first = copy.deepcopy(self.report['epochs'][0])
        selected_metrics = {k: self.report['metrics']['validation'][k] for k in ('accuracy', 'macro_f1')}
        self.report['epochs'] = [first | {'epoch': epoch, 'validation':
            selected_metrics if epoch % 10 == 0 else None} for epoch in range(1, 1001)]
        checkpoint = torch.load(self.run / 'model.pt', weights_only=True, map_location='cpu')
        checkpoint.update(config=config, selected_epoch=10)
        torch.save(checkpoint, self.run / 'model.pt')
        self.refresh_artifact('model.pt')
        return config

    def test_full_purpose_null_or_fake_registration_rejected(self):
        self.make_full_report()
        for index, registration in enumerate((None, {},
                {'registry_path': 'bogus.json', 'registry_sha256': '0' * 64},
                {'protocol': 'paper_baseline_finite_v1', 'seed': 0})):
            self.output = self.root / f'proof-{index}.json'
            self.report['execution_registration'] = registration
            self.save_report()
            with self.subTest(registration=registration):
                self.assert_rejected()

    def test_smoke_must_have_explicit_null_registration(self):
        for index, registration in enumerate(({}, {'seed': 0}, 'fake', 'missing')):
            self.output = self.root / f'proof-{index}.json'
            if registration == 'missing':
                self.report.pop('execution_registration', None)
            else:
                self.report['execution_registration'] = registration
            self.save_report()
            with self.subTest(registration=registration):
                self.assert_rejected()

    def registered_fixture(self):
        """Controlled JSON identity chain around two-class CPU smoke weights.

        Synthetic history and acceptance ONLY exercise registration identity.
        No full training or GPU acceptance is performed or claimed here. Only
        the historical design hash is substituted for this tiny data identity.
        """
        config = self.make_full_report()
        directory = self.root / 'registration'
        (directory / 'configs').mkdir(parents=True)
        self.run.rename(directory / 'resnext_s0')
        self.run = directory / 'resnext_s0'
        base_path = self.root / 'config/paper_baseline_v1.json'
        base_path.parent.mkdir()
        shutil.copyfile(ROOT / 'config/paper_baseline_v1.json', base_path)
        cache = self.root / config['cache_dir']
        metadata = json.loads((cache / 'metadata.json').read_text())
        files = {'config/paper_baseline_v1.json': sha(base_path)}
        components = {}
        for name in ('report.json', 'verification.json'):
            relative = f'results/paper_components/acceptance_v1/{name}'
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            files[relative] = sha(target)
            components[name] = sha(target)
        design = json.loads((ROOT / 'results/paper_components/baseline_design_v1.json').read_text())
        design.update(cache_metadata_sha256=sha(cache / 'metadata.json'),
                      manifest_sha256=metadata['manifest_sha256'])
        design_path = self.root / 'results/paper_components/baseline_design_v1.json'
        write(design_path, design)
        design_sha = sha(design_path)
        self.design_guard = patch.object(self.verifier, 'DESIGN_SHA256', design_sha, create=True)
        self.design_guard.start()
        self.addCleanup(self.design_guard.stop)
        files['results/paper_components/baseline_design_v1.json'] = design_sha
        # Independently enumerate the registered 13 paths, including verify CLI.
        sources = {f'src/ect/{name}.py': sha(ROOT / f'src/ect/{name}.py') for name in (
            'paper_training', 'paper_registration', 'paper_models', 'paper_crops', 'devices',
            'baseline', 'prepare', 'dataset', 'preprocessing', 'integrity')}
        sources.update({f'scripts/{name}.py': sha(ROOT / f'scripts/{name}.py') for name in (
            'train_paper_baseline', 'register_paper_baseline', 'verify_paper_run')})
        acceptance_path = self.root / 'controlled_acceptance.json'
        artifact = self.root / 'controlled_review.txt'
        artifact.write_text('Synthetic identity fixture, not actual CUDA acceptance.\n')
        self.acceptance = dict(schema_version=1, status='verified',
            protocol='paper_baseline_runner_acceptance_v1', test_classification_enabled=False,
            checks={name: True for name in ('cpu_training', 'cuda_training',
                'independent_crop_reload', 'csv_metrics', 'full_regression', 'independent_code_review')},
            code_sha256=sources, files_sha256={'controlled_review.txt': sha(artifact)})
        write(acceptance_path, self.acceptance)
        files.update(self.acceptance['files_sha256'])
        files['controlled_acceptance.json'] = sha(acceptance_path)
        files[(cache / 'metadata.json').relative_to(self.root).as_posix()] = sha(cache / 'metadata.json')
        files[metadata['summary_path']] = metadata['summary_sha256']
        for name, digest in metadata['files_sha256'].items():
            files[(cache / name).relative_to(self.root).as_posix()] = digest
        for name, digest in metadata['manifest_sha256'].items():
            files[(Path(metadata['summary_path']).parent / name).as_posix()] = digest
        for info in metadata['inputs'].values():
            files[info['path']] = info['sha256']
        runs = []
        for seed in (0, 1, 2):
            resolved = config | {'seed': seed}
            path = directory / 'configs' / f'resnext_s{seed}.json'
            write(path, resolved)
            relative = path.relative_to(self.root).as_posix()
            files[relative] = sha(path)
            runs.append(dict(run_id=f'resnext_s{seed}', seed=seed, config_path=relative,
                config_file_sha256=sha(path), resolved_config_sha256=hashlib.sha256(json.dumps(
                    resolved, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                output_dir=f'registration/resnext_s{seed}', status_at_registration='registered'))
        self.registry = dict(schema_version=1, protocol=config['protocol'],
            status='frozen_execution_registration', registered_utc='2026-10-02T00:00:00+00:00',
            registration_dir='registration', test_classification_enabled=False,
            base_config_path='config/paper_baseline_v1.json', base_config_sha256=sha(base_path),
            design_path='results/paper_components/baseline_design_v1.json', design_sha256=design_sha,
            acceptance_path='controlled_acceptance.json', acceptance_sha256=sha(acceptance_path),
            code_sha256=sources, data_identity=dict(cache_dir=config['cache_dir'],
                cache_metadata_sha256=sha(cache / 'metadata.json'),
                cache_files_sha256=metadata['files_sha256'], manifest_sha256=metadata['manifest_sha256'],
                summary_path=metadata['summary_path'], summary_sha256=metadata['summary_sha256'],
                raw_inputs=metadata['inputs'], sample_counts=metadata['sample_counts'],
                normalization_sha256=sha(cache / 'normalization.json')), files_sha256=files, runs=runs)
        self.report['execution_registration'] = dict(registry_path='registration/registry.json',
            registry_sha256='', protocol=config['protocol'], registered_utc=self.registry['registered_utc'],
            **{key: runs[0][key] for key in ('run_id', 'seed', 'config_path', 'config_file_sha256',
                'resolved_config_sha256', 'output_dir')}, acceptance_sha256=sha(acceptance_path))
        self.resign_registry()

    def resign_registry(self):
        path = self.root / 'registration/registry.json'
        write(path, self.registry)
        self.report['execution_registration']['registry_sha256'] = sha(path)
        self.save_report()

    def test_controlled_full_identity_proof_hashes_every_registry_file_and_source(self):
        self.registered_fixture()
        from src.ect import paper_training, paper_registration, paper_crops
        with (patch.object(paper_registration, 'validate_registered_run', side_effect=AssertionError('producer')),
              patch.object(paper_training, 'predict_paper', side_effect=AssertionError('producer')),
              patch.object(paper_training, 'load_paper_model', side_effect=AssertionError('producer')),
              patch.object(paper_crops, 'ten_crop_probabilities', side_effect=AssertionError('producer'))):
            result = self.verify()
        for relative, digest in self.registry['files_sha256'].items():
            self.assertEqual(result['files_sha256'][str(self.root / relative)], digest)
        for relative, digest in self.registry['code_sha256'].items():
            self.assertEqual(result['files_sha256'][str(ROOT / relative)], digest)
        self.assertEqual(result['execution_registration'], self.report['execution_registration'])
        self.assertTrue(result['flags']['execution_registration_verified'])
        self.assertEqual(result['prediction_counts'], {'train': 2, 'validation': 2})

    def test_smoke_fake_real_registry_cannot_authorize_full_identity(self):
        self.registered_fixture()
        # A valid registry still cannot be attached to a smoke report.
        self.report['config']['purpose'] = 'paper_runner_smoke_validation_only'
        self.report['purpose'] = self.report['config']['purpose']
        self.report['config_sha256'] = hashlib.sha256(json.dumps(self.report['config'],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.save_report()
        self.assert_rejected()

    def test_full_report_registration_fields_are_not_trusted(self):
        self.registered_fixture()
        original = copy.deepcopy(self.report)
        changes = [('registry_sha256', '0' * 64), ('seed', 1), ('run_id', 'resnext_s1'),
                   ('output_dir', 'run'), ('config_path', 'registration/configs/resnext_s1.json'),
                   ('acceptance_sha256', '0' * 64), ('registered_utc', 'forged'), ('extra', True)]
        for index, (field, value) in enumerate(changes):
            self.report = copy.deepcopy(original)
            self.report['execution_registration'][field] = value
            self.output = self.root / f'proof-{index}.json'
            self.save_report()
            with self.subTest(field=field):
                self.assert_rejected()

    def test_resigned_registry_corruption_rejected_independently(self):
        self.registered_fixture()
        original = copy.deepcopy(self.registry)
        mutations = [lambda r: r.update(design_sha256='0' * 64),
            lambda r: r.update(base_config_sha256='0' * 64),
            lambda r: r.update(test_classification_enabled=True),
            lambda r: r['runs'][2].update(seed=1),
            lambda r: r['runs'][1].update(output_dir='arbitrary'),
            lambda r: r['data_identity']['sample_counts'].update(train=1),
            lambda r: r['data_identity'].update(normalization_sha256='0' * 64),
            lambda r: r['code_sha256'].pop('scripts/verify_paper_run.py'),
            lambda r: r['code_sha256'].update({'scripts/verify_paper_run.py': '0' * 64}),
            lambda r: r['files_sha256'].pop('controlled_review.txt'),
            lambda r: r['files_sha256'].update({'controlled_review.txt': '0' * 64}),
            lambda r: r['files_sha256'].update({'./controlled_review.txt': sha(self.root / 'controlled_review.txt')})]
        for index, mutate in enumerate(mutations):
            self.registry = copy.deepcopy(original)
            mutate(self.registry)
            self.output = self.root / f'proof-{index}.json'
            self.resign_registry()
            with self.subTest(index=index):
                self.assert_rejected()

    def test_all_three_config_files_are_read_even_for_seed_zero(self):
        self.registered_fixture()
        path = self.root / 'registration/configs/resnext_s2.json'
        config = json.loads(path.read_text())
        config['seed'] = 1
        write(path, config)
        self.registry['files_sha256']['registration/configs/resnext_s2.json'] = sha(path)
        self.registry['runs'][2]['config_file_sha256'] = sha(path)
        self.resign_registry()
        self.assert_rejected()

    def test_acceptance_artifacts_and_source_table_cannot_be_forged(self):
        self.registered_fixture()
        original = copy.deepcopy(self.acceptance)
        for index, mutate in enumerate((
                lambda a: a['code_sha256'].pop('scripts/verify_paper_run.py'),
                lambda a: a['files_sha256'].update({'controlled_review.txt': '0' * 64}),
                lambda a: a['checks'].update(cuda_training=False))):
            acceptance = copy.deepcopy(original)
            mutate(acceptance)
            path = self.root / 'controlled_acceptance.json'
            write(path, acceptance)
            self.registry['acceptance_sha256'] = sha(path)
            self.registry['files_sha256']['controlled_acceptance.json'] = sha(path)
            self.report['execution_registration']['acceptance_sha256'] = sha(path)
            self.output = self.root / f'proof-{index}.json'
            self.resign_registry()
            with self.subTest(index=index):
                self.assert_rejected()

    def test_registry_failure_and_drift_during_inference_block_publication(self):
        self.registered_fixture()
        (self.root / 'registration/failure.json').write_text('{}')
        self.assert_rejected()
        (self.root / 'registration/failure.json').unlink()
        original = self.verifier.independent_probabilities
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            (self.root / 'controlled_review.txt').write_text('changed')
            return result
        with patch.object(self.verifier, 'independent_probabilities', side_effect=corrupt):
            self.assert_rejected()

    def test_link_commit_survives_temporary_unlink_oserror(self):
        unlink = Path.unlink
        def cleanup_error(path, *args, **kwargs):
            if path.name.startswith('.paper-proof-'):
                raise OSError('temporary cleanup denied after commit')
            return unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', cleanup_error):
            result = self.verify()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(len(list(self.root.glob('.paper-proof-*'))), 1)

    def test_link_failure_never_publishes_even_if_temp_cleanup_fails(self):
        unlink = Path.unlink
        def cleanup_error(path, *args, **kwargs):
            if path.name.startswith('.paper-proof-'):
                raise OSError('temporary cleanup denied before commit')
            return unlink(path, *args, **kwargs)
        with (patch.object(self.verifier.os, 'link', side_effect=OSError('link failed')),
              patch.object(Path, 'unlink', cleanup_error)):
            with self.assertRaises(OSError):
                self.verify()
        self.assertFalse(self.output.exists())

    def test_loaded_literal_defaults_are_verified_beyond_code_objects(self):
        function = self.verifier._paper_models._same_padding
        original = function.__defaults__
        try:
            function.__defaults__ = (2,)  # same __code__, different dilation default
            with self.assertRaisesRegex(ValueError, 'default'):
                self.verifier.verify_loaded_inference_sources()
        finally:
            function.__defaults__ = original

    def test_loaded_keyword_defaults_are_verified(self):
        function = self.verifier._devices.seed_everything
        original = function.__kwdefaults__
        try:
            function.__kwdefaults__ = {'forged': 1}
            with self.assertRaisesRegex(ValueError, 'default'):
                self.verifier.verify_loaded_inference_sources()
        finally:
            function.__kwdefaults__ = original

    def test_history_and_checkpoint_metadata_are_independently_checked(self):
        for edit in ('selected', 'missing', 'schedule', 'history_metric', 'tie'):
            original = copy.deepcopy(self.report)
            if edit == 'selected':
                self.report['selected_epoch'] = 999
            elif edit == 'missing':
                self.report['epochs'].pop()
            elif edit == 'schedule':
                self.report['epochs'][0]['learning_rate'] = 4e-6
            elif edit == 'history_metric':
                self.report['epochs'][self.report['selected_epoch'] - 1]['validation']['accuracy'] = .123
            else:
                # Manufacture an accuracy tie whose first epoch must win.
                self.report['epochs'][0]['validation'] = copy.deepcopy(self.report['epochs'][1]['validation'])
                self.report['selected_epoch'] = 2
            self.save_report()
            with self.subTest(edit=edit):
                self.assert_rejected()
            self.report = original
        self.save_report()
        saved = torch.load(self.run / 'model.pt', weights_only=True, map_location='cpu')
        saved['normalization']['mean'][0] += 1
        torch.save(saved, self.run / 'model.pt')
        self.refresh_artifact('model.pt')
        self.assert_rejected()

    def test_cache_checksum_and_manifest_label_order_checked(self):
        path = self.root / 'cache/validation_y.npy'
        labels = np.load(path)[::-1].copy()
        np.save(path, labels)
        self.assert_rejected()
        metadata_path = self.root / 'cache/metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['files_sha256'][path.name] = sha(path)
        write(metadata_path, metadata)
        self.report['cache_metadata_sha256'] = sha(metadata_path)
        self.save_report()
        self.assert_rejected()

    def test_output_must_be_new_and_outside_run(self):
        self.output.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.verify()
        self.assertEqual(self.output.read_text(), 'keep')
        with self.assertRaises(ValueError):
            self.verifier.verify_paper_run(self.root, self.run, self.run / 'proof.json')
        self.assertFalse((self.run / 'proof.json').exists())

    def test_cli_cpu_and_invalid_split(self):
        command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/verify_paper_run.py'),
                   '--project-root', str(self.root), '--run', 'run', '--output', str(self.output),
                   '--device', 'cpu', '--split', 'validation', '--batch-size', '2']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text())['prediction_counts'], {'validation': 2})
        result = subprocess.run(command + ['--split', 'test'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
