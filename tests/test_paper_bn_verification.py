"""Independent BN verification: real fixtures, adversarial identity and state tests."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
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
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n', encoding='utf-8')


class IndependentCalibrationTests(unittest.TestCase):
    def module(self):
        from scripts import verify_paper_bn
        return verify_paper_bn

    def test_manual_training_crops_normalization_and_original_state(self):
        torch.set_num_threads(1)
        raw = np.arange(1000, dtype=np.float32).reshape(2, 250, 2) / 31
        mean, std = np.array([.12, -.77]), np.array([3.7, 1.23])
        captured = []
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2, momentum=.01, eps=.001)
            def forward(self, batch):
                captured.append(batch.cpu().numpy().copy())
                return self.bn(batch)
        original = Recorder().eval()
        original.bn.running_mean.fill_(91)
        before = copy.deepcopy(original.state_dict())
        calibrated, summary = self.module().independent_calibration(
            original, raw, ['身份/alpha', 'beta'], mean, std, torch.device('cpu'))
        expected = np.stack([((raw[i, start:start + 224].astype(np.float64) - mean) / std)
                             .astype(np.float32).T for i, start in enumerate([11, 9])])
        np.testing.assert_array_equal(np.concatenate(captured), expected)
        np.testing.assert_allclose(calibrated.bn.running_mean.numpy(), expected.mean((0, 2)), rtol=1e-6)
        np.testing.assert_allclose(calibrated.bn.running_var.numpy(),
                                   expected.transpose(1, 0, 2).reshape(2, -1).var(1, ddof=1), rtol=1e-6)
        self.assertEqual(summary['sample_count'], 2)
        self.assertEqual(summary['batch_count'], 1)
        self.assertEqual(summary['last_batch_size'], 2)
        self.assertFalse(calibrated.training)
        self.assertEqual(calibrated.bn.momentum, .01)
        self.assertFalse(original.training)
        self.assertTrue(all(torch.equal(before[k], original.state_dict()[k]) for k in before))

    def test_manifest_order_tail_is_retained_and_batches_have_equal_weight(self):
        raw = np.repeat(np.arange(130, dtype=np.float32)[:, None, None], 500, axis=1).reshape(130, 250, 2)
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2, momentum=.01)).eval()
        calibrated, summary = self.module().independent_calibration(
            model, raw, [str(i) for i in range(130)], np.zeros(2), np.ones(2), torch.device('cpu'))
        self.assertEqual(summary['batch_count'], 2)
        self.assertEqual(summary['last_batch_size'], 2)
        np.testing.assert_allclose(calibrated[0].running_mean.numpy(), [96., 96.], rtol=1e-6)
        self.assertEqual(calibrated[0].num_batches_tracked.item(), 2)

    def test_nonfinite_inputs_are_rejected(self):
        raw = np.zeros((2, 250, 2), dtype=np.float32)
        raw[1, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.module().independent_calibration(torch.nn.Sequential(torch.nn.BatchNorm1d(2)),
                raw, ['a', 'b'], np.zeros(2), np.ones(2), torch.device('cpu'))

    def test_bn_affine_parameter_changes_are_rejected(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2)).eval()
        calibrated = copy.deepcopy(model)
        with torch.no_grad():
            calibrated[0].weight.add_(.001)
        with self.assertRaisesRegex(ValueError, 'parameter|Parameter'):
            self.module().compare_bn_states(model, calibrated, calibrated, True)

    def test_same_device_requires_exact_buffers_but_cross_device_is_bounded(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2)).eval()
        replayed = copy.deepcopy(model)
        replayed[0].running_mean.fill_(1.)
        calibrated = copy.deepcopy(replayed)
        calibrated[0].running_mean.add_(1e-6)
        with self.assertRaisesRegex(ValueError, 'exact|Exact'):
            self.module().compare_bn_states(model, calibrated, replayed, True)
        result = self.module().compare_bn_states(model, calibrated, replayed, False)
        self.assertGreater(result['maximum_error'], 0)
        self.assertFalse(result['exact'])
        calibrated[0].running_mean.add_(.01)
        with self.assertRaisesRegex(ValueError, 'buffer|Buffer'):
            self.module().compare_bn_states(model, calibrated, replayed, False)

    def test_batch_counter_is_exact_even_across_devices(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2)).eval()
        saved = copy.deepcopy(model)
        saved[0].num_batches_tracked.add_(1)
        with self.assertRaisesRegex(ValueError, 'counter|Counter|buffer'):
            self.module().compare_bn_states(model, saved, model, False)


class BNVerifierAPITests(unittest.TestCase):
    def test_public_api_is_present(self):
        from scripts import verify_paper_bn
        self.assertTrue(callable(getattr(verify_paper_bn, 'verify_paper_bn', None)),
                        'Independent BN verification API must be implemented')

    def test_output_and_run_must_be_root_contained(self):
        from scripts import verify_paper_bn
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'outside|Outside'):
                verify_paper_bn.verify_paper_bn(root, root / 'run', root.parent / 'proof.json')
            with self.assertRaisesRegex(ValueError, 'outside|Outside'):
                verify_paper_bn.verify_paper_bn(root, root.parent / 'run', root / 'proof.json')

    def test_existing_output_cannot_be_overwritten(self):
        from scripts import verify_paper_bn
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = root / 'proof.json'
            proof.write_text('existing')
            with self.assertRaises(FileExistsError):
                verify_paper_bn.verify_paper_bn(root, root / 'run', proof)
            self.assertEqual(proof.read_text(), 'existing')

    def test_failure_and_running_markers_are_refused_before_load(self):
        from scripts import verify_paper_bn
        for marker in ('failure.json', 'running.marker'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = root / 'run'
                run.mkdir()
                (run / marker).write_text('failed or running')
                with self.assertRaisesRegex(ValueError, 'Failed|Running|failed|running'):
                    verify_paper_bn.verify_paper_bn(root, run, root / 'proof.json')
                self.assertFalse((root / 'proof.json').exists())

    def test_verifier_source_changed_after_import_cannot_be_claimed_as_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp)
            for folder in ('scripts', 'src'):
                shutil.copytree(ROOT / folder, copied / folder, ignore=shutil.ignore_patterns('__pycache__'))
            code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import verify_paper_bn as verifier
path = Path(verifier.__file__)
path.write_bytes(path.read_bytes() + b"\\n# changed after import\\n")
try:
    verifier.verify_loaded_sources()
except ValueError as error:
    assert "changed since import" in str(error), str(error)
else:
    raise AssertionError("Stale verifier source was accepted")
'''
            result = subprocess.run([sys.executable, '-B', '-c', code, tmp],
                                    cwd=tmp, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_same_behavior_replaced_calibration_function_cannot_provide_evidence(self):
        from scripts import verify_paper_bn
        original = verify_paper_bn.independent_calibration
        def replacement(*args, **kwargs):
            return original(*args, **kwargs)
        with patch.object(verify_paper_bn, 'independent_calibration', replacement):
            with self.assertRaisesRegex(ValueError, 'Loaded verification function'):
                verify_paper_bn.verify_loaded_sources()


class RegistrationIdentityTests(unittest.TestCase):
    """Registration boundary tests; synthetic JSON never stands for GPU acceptance."""
    def setUp(self):
        from scripts import verify_paper_bn
        self.verifier = verify_paper_bn
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'registry.json'
        self.report = dict(purpose='full', config=copy.deepcopy(verify_paper_bn.CONFIG),
            code_sha256={name: sha(ROOT / name) for name in verify_paper_bn.SOURCES},
            execution_registration={'registry_path': 'registry.json', 'registry_sha256': '', 'run_id': 'resnext_s0'})
        self.anchor = self.root / 'anchor.json'
        self.anchor.write_text('anchor')
        self.registry = dict(schema_version=1, status='frozen_execution_registration',
            protocol='paper_bn_recalibration_v1', config=copy.deepcopy(self.report['config']),
            created_utc='2026-10-03T00:00:00Z', test_classification_enabled=False,
            code_sha256=self.report['code_sha256'], files_sha256={'anchor.json': sha(self.anchor)}, runs=[])

    def verify(self, anchors=None):
        write(self.path, self.registry)
        self.report['execution_registration']['registry_sha256'] = sha(self.path)
        def checked(path, expected=None):
            actual = sha(path)
            self.assertTrue(expected is None or actual == expected)
            return actual
        return self.verifier.check_registration(self.root, self.root / 'new', self.report,
            self.root / 'origin', {}, checked, anchors or {})

    def test_registry_config_boolean_alias_is_rejected(self):
        self.registry['config']['calibration_seed'] = False
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            self.verify()

    def test_registry_must_explicitly_seal_test_classification(self):
        for value in (True, 0, None):
            self.registry['test_classification_enabled'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Frozen'):
                self.verify()

    def test_registry_complete_sources_and_file_anchor_coverage_required(self):
        source = self.registry['code_sha256']
        self.registry['code_sha256'] = {}
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.verify()
        self.registry['code_sha256'] = source
        with self.assertRaisesRegex(ValueError, 'registration anchor'):
            self.verify({str(self.root / 'missing.json'): 'f' * 64})

    def test_registry_requires_all_three_complete_run_entries(self):
        with self.assertRaisesRegex(ValueError, 'Three complete'):
            self.verify()


class BNVerifierIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import verify_paper_bn
        from src.ect.paper_training import train_paper_baseline
        from src.ect.paper_bn import run_recalibration
        cls.verifier = verify_paper_bn
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        _, manifests = make_dataset(cls.fixture.name)
        prepare_dataset(cls.fixture.name, manifests, 'cache')
        config = json.loads((ROOT / 'config/paper_baseline_v1.json').read_text())
        config.update(purpose='paper_runner_smoke_validation_only', seed=0, device='cpu',
                      max_epochs=2, validation_every_epochs=1, batch_size=2, cache_dir='cache')
        train_paper_baseline(cls.fixture.name, config, 'origin')
        # Producers create evidence only; the verifier must remain independent.
        run_recalibration(cls.fixture.name, 'origin', 'new', device='cpu', smoke=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(self.fixture.name, self.root, dirs_exist_ok=True)
        self.run, self.origin = self.root / 'new', self.root / 'origin'
        self.output = self.root / 'proof.json'
        self.report = json.loads((self.run / 'report.json').read_text())

    def verify(self, **kwargs):
        return self.verifier.verify_paper_bn(self.root, self.run, self.output, **kwargs)

    def save(self):
        write(self.run / 'report.json', self.report)

    def refresh(self, name):
        self.report['files_sha256'][name] = sha(self.run / name)
        self.save()

    def rejected(self):
        with self.assertRaises((ValueError, RuntimeError, OSError, KeyError, TypeError)):
            self.verify()
        self.assertFalse(self.output.exists(), 'Rejected evidence cannot publish a proof')

    def test_real_cpu_checkpoint_and_both_split_predictions_are_verified_independently(self):
        from src.ect import paper_bn, paper_training, paper_crops, preprocessing, baseline
        poison = AssertionError('Producer routine used by independent verifier')
        before = {str(p.relative_to(self.root)): sha(p) for folder in (self.origin, self.run)
                  for p in folder.iterdir() if p.is_file()}
        load = np.load
        opened = []
        def sealed(path, *args, **kwargs):
            self.assertFalse(Path(path).name.startswith('test_'), 'Test arrays must never be mapped')
            result = load(path, *args, **kwargs)
            opened.append(result)
            return result
        with (patch.object(paper_bn, 'recalibrate_bn', side_effect=poison),
              patch.object(paper_bn, 'load_recalibrated_model', side_effect=poison),
              patch.object(paper_training, 'load_paper_model', side_effect=poison),
              patch.object(paper_training, 'predict_paper', side_effect=poison),
              patch.object(paper_training, 'metrics', side_effect=poison),
              patch.object(paper_crops, 'training_crops', side_effect=poison),
              patch.object(paper_crops, 'crop_starts', side_effect=poison),
              patch.object(paper_crops, 'ten_crop_probabilities', side_effect=poison),
              patch.object(preprocessing, 'standardize', side_effect=poison),
              patch.object(baseline, 'classification_metrics', side_effect=poison, create=True),
              patch.object(np, 'load', side_effect=sealed)):
            proof = self.verify()
        self.assertEqual(proof['status'], 'verified')
        self.assertEqual(proof['protocol'], 'paper_bn_recalibration_v1')
        self.assertEqual(proof['prediction_counts'], {'train': 2, 'validation': 2})
        self.assertEqual(proof['maximum_metric_error'], 0)
        self.assertEqual(proof['calibration_buffer_max_abs_error'], 0)
        self.assertTrue(proof['same_device_calibration_exact'])
        self.assertEqual(proof['checks']['prediction_class_mismatches'], 0)
        self.assertFalse(proof['checks']['test_classification_enabled'])
        self.assertTrue(all(m._mmap.closed for m in opened))
        self.assertEqual(proof['verification_script']['sha256'], sha(self.verifier.__file__))
        self.assertEqual(proof['verification_script']['text'], Path(self.verifier.__file__).read_text())
        self.assertEqual(json.loads(self.output.read_text()), proof)
        self.assertEqual(before, {str(p.relative_to(self.root)): sha(p) for folder in (self.origin, self.run)
                                 for p in folder.iterdir() if p.is_file()})

    def test_forged_file_hash_cannot_hide_learned_parameter_or_bn_buffer_changes(self):
        original = torch.load(self.run / 'model.pt', weights_only=True)
        for kind in ('parameter', 'running_mean', 'num_batches_tracked'):
            saved = copy.deepcopy(original)
            key = next(k for k in saved['state_dict'] if k.endswith('weight' if kind == 'parameter' else kind))
            saved['state_dict'][key].add_(1)
            torch.save(saved, self.run / 'model.pt')
            self.refresh('model.pt')
            with self.subTest(kind=kind):
                self.rejected()

    def test_checkpoint_boolean_substitutes_for_integer_metadata_are_rejected(self):
        original = torch.load(self.run / 'model.pt', weights_only=True)
        for kind in ('schema_version', 'calibration_seed', 'deterministic', 'selected_epoch'):
            self.output = self.root / f'{kind}-proof.json'
            saved = copy.deepcopy(original)
            if kind == 'schema_version': saved[kind] = True
            elif kind == 'selected_epoch':
                # The fixture selected epoch is one: bool equality must not hide this.
                saved[kind] = True
            elif kind == 'calibration_seed': saved['config'][kind] = False
            else: saved['config'][kind] = 1
            torch.save(saved, self.run / 'model.pt')
            self.refresh('model.pt')
            with self.subTest(kind=kind): self.rejected()

    def test_original_checkpoint_schema_boolean_is_rejected_even_with_forged_origin_hashes(self):
        path = self.origin / 'model.pt'
        saved = torch.load(path, weights_only=True)
        saved['schema_version'] = True
        torch.save(saved, path)
        origin_report = json.loads((self.origin / 'report.json').read_text())
        origin_report['files_sha256']['model.pt'] = sha(path)
        write(self.origin / 'report.json', origin_report)
        self.report['origin_report_sha256'] = sha(self.origin / 'report.json')
        self.report['origin_model_sha256'] = sha(path)
        checkpoint = torch.load(self.run / 'model.pt', weights_only=True)
        checkpoint['origin_model_sha256'] = sha(path)
        torch.save(checkpoint, self.run / 'model.pt')
        self.refresh('model.pt')
        self.rejected()

    def test_report_environment_must_record_one_thread(self):
        self.report['environment']['threads'] = 2
        self.save()
        self.rejected()

    def test_rows_csv_order_labels_identity_and_predictions_are_checked_even_with_forged_hashes(self):
        path = self.run / 'validation_predictions.csv'
        original = path.read_bytes()
        for kind in ('order', 'duplicate', 'label', 'identity', 'group', 'prediction'):
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            if kind == 'order': rows.reverse()
            elif kind == 'duplicate': rows[1]['row_index'] = rows[0]['row_index']
            elif kind == 'label': rows[0]['class_index'] = str(1 - int(rows[0]['class_index']))
            elif kind == 'identity': rows[0]['wave_sha256'] = 'e' * 64
            elif kind == 'group': rows[0]['group_id'] = 'other'
            else: rows[0]['predicted_class_index'] = str(1 - int(rows[0]['predicted_class_index']))
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.refresh(path.name)
            with self.subTest(kind=kind): self.rejected()
            path.write_bytes(original)
            self.refresh(path.name)
        write(self.run / 'train_rows.json', [1, 0])
        self.refresh('train_rows.json')
        self.rejected()

    def test_per_class_group_and_baseline_metrics_are_recomputed(self):
        original = copy.deepcopy(self.report)
        for kind in ('overall', 'class', 'group', 'baseline'):
            if kind == 'overall': self.report['metrics']['train']['accuracy'] += .125
            elif kind == 'class': self.report['metrics']['train']['per_class']['0']['precision'] += .125
            elif kind == 'group': self.report['group_metrics']['train']['0']['macro_f1'] += .125
            else: self.report['baseline_metrics']['train']['accuracy'] += .125
            self.save()
            with self.subTest(kind=kind): self.rejected()
            self.report = copy.deepcopy(original)

    def test_report_rejects_missing_hashes_wrong_origin_configuration_and_false_calibration(self):
        original = copy.deepcopy(self.report)
        changes = [('code_sha256', {}), ('files_sha256', {}), ('origin_report_sha256', '0' * 64),
            ('origin_model_sha256', '0' * 64), ('heldout_test_evaluated', True), ('status', 'running'),
            ('config', self.report['config'] | {'calibration_seed': 1}), ('execution_registration', {}),
            ('calibration', self.report['calibration'] | {'parameters_unchanged': False}),
            ('calibration', self.report['calibration'] | {'sample_count': 3}),
            ('calibration', self.report['calibration'] | {'changed_buffers': []}),
            ('normalization', self.report['normalization'] | {'mean': [0., 0.]})]
        for key, value in changes:
            self.report[key] = value
            self.save()
            with self.subTest(key=key, value=value): self.rejected()
            self.report = copy.deepcopy(original)

    def test_determinism_is_configured_before_either_checkpoint_is_loaded(self):
        original = torch.load
        calls = []
        def load(*args, **kwargs):
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            calls.append(Path(args[0]).parent.name)
            return original(*args, **kwargs)
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        with patch.object(torch, 'load', side_effect=load): self.verify()
        self.assertEqual(calls, ['new', 'origin'])

    def test_same_behavior_patched_loaded_model_and_verifier_helpers_are_rejected(self):
        model = self.verifier.old.PaperResNeXt1D
        forward = model.forward
        def same(self, x): return forward(self, x)
        with patch.object(model, 'forward', same): self.rejected()
        original = self.verifier.numpy_metrics
        def same_metrics(*args, **kwargs): return original(*args, **kwargs)
        with patch.object(self.verifier, 'numpy_metrics', same_metrics): self.rejected()

    def test_output_inside_either_run_is_rejected(self):
        for folder in (self.origin, self.run):
            self.output = folder / 'proof.json'
            self.rejected()

    def test_atomic_publication_never_overwrites_concurrently_created_output(self):
        original = self.verifier.os.link
        def competitor(source, target):
            Path(target).write_text('concurrent proof')
            return original(source, target)
        with patch.object(self.verifier.os, 'link', side_effect=competitor):
            with self.assertRaises(FileExistsError): self.verify()
        self.assertEqual(self.output.read_text(), 'concurrent proof')
        self.assertFalse(list(self.root.glob('.paper-bn-proof-*')))

    def test_marker_or_artifact_change_during_final_hash_check_blocks_publication(self):
        for mutation in ('marker', 'artifact'):
            # Trigger drift through actual file IO without changing verifier helpers.
            read = Path.open
            reads = 0
            def observe(path, *args, **kwargs):
                nonlocal reads
                if path == self.run / 'report.json' and args and args[0] == 'rb':
                    reads += 1
                    if reads == 2:
                        if mutation == 'marker': (self.run / 'running.marker').write_text('running')
                        else:
                            with read(self.run / 'train_predictions.csv', 'ab') as handle: handle.write(b'changed')
                return read(path, *args, **kwargs)
            with patch.object(Path, 'open', observe): self.rejected()
            (self.run / 'running.marker').unlink(missing_ok=True)

    def test_cli_real_smoke_and_no_test_option(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_paper_bn.py'),
            '--project-root', str(self.root), '--run', 'new', '--output', 'cli-proof.json', '--device', 'cpu'],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)['prediction_counts'], {'train': 2, 'validation': 2})
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_paper_bn.py'),
            '--run', 'new', '--output', 'unused.json', '--split', 'test'], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
    def test_real_cuda_replay_and_cuda_checkpoint_cpu_migration(self):
        from src.ect.paper_bn import run_recalibration
        run_recalibration(self.root, 'origin', 'cuda-new', device='cuda', smoke=True)
        self.run = self.root / 'cuda-new'
        self.report = json.loads((self.run / 'report.json').read_text())
        proof = self.verify(device='cuda')
        self.assertEqual(proof['device'], 'cuda')
        self.assertTrue(proof['same_device_calibration_exact'])
        self.assertEqual(proof['calibration_buffer_max_abs_error'], 0)
        self.output = self.root / 'transfer-proof.json'
        proof = self.verify(device='cpu')
        self.assertEqual(proof['device'], 'cpu')
        self.assertFalse(proof['same_device_calibration_exact'])
        self.assertEqual(proof['checks']['prediction_class_mismatches'], 0)
