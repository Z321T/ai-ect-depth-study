"""Independent E2 verification tests; producer helpers never build expectations."""
import copy
import csv
import hashlib
import importlib
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
from src.ect.devices import seed_everything
from src.ect.paper_models import PaperResNeXt1D

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, allow_nan=False) + '\n')


class IndependentStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'scripts/verify_paper_dc_ac.py').exists(),
                        'E2 independent verifier is missing')
        self.v = importlib.import_module('scripts.verify_paper_dc_ac')

    def test_full_train_population_and_ac_power_are_hand_calculated(self):
        # Sample DCs 1, 5 / 10, 14, with residual alternating +/-2 / +/-3.
        raw = np.zeros((2, 250, 2), dtype=np.float32)
        raw[:, :, 0] = np.array([1, 5])[:, None] + np.tile([-2, 2], 125)
        raw[:, :, 1] = np.array([10, 14])[:, None] + np.tile([-3, 3], 125)
        stats = self.v.fit_statistics(raw)
        self.assertEqual(stats['sample_count'], 2)
        self.assertEqual(stats['points_per_channel'], 500)
        np.testing.assert_allclose(stats['mu_d'], [3, 12], rtol=0, atol=0)
        np.testing.assert_allclose(stats['sigma_d'], [2, 2], rtol=0, atol=0)
        np.testing.assert_allclose(stats['sigma_a'], [2, 3], rtol=0, atol=0)

    def test_zero_scales_use_explicit_floor(self):
        raw = np.full((3, 250, 2), 7, np.float32)
        stats = self.v.fit_statistics(raw)
        self.assertEqual(stats['floor_d'], [True, True])
        self.assertEqual(stats['floor_a'], [True, True])
        self.assertEqual(stats['scale_d'], [1e-12, 1e-12])
        np.testing.assert_array_equal(self.v.transform_full250(raw, stats), 0)

    def test_transform_preserves_relative_ac_and_dc(self):
        raw = np.zeros((2, 250, 2), np.float32)
        raw[0] = np.tile([[-1, -1], [1, 1]], (125, 1))
        raw[1] = 4 + 2 * raw[0]
        stats = self.v.fit_statistics(raw)
        z = self.v.transform_full250(raw, stats)
        np.testing.assert_allclose(z.mean(axis=1), [[-2**-.5]*2, [2**-.5]*2], atol=1e-6)
        self.assertAlmostEqual(float(z[1, :, 0].std() / z[0, :, 0].std()), 2, places=6)
        np.testing.assert_array_equal(z[:1], self.v.transform_full250(raw[:1], stats))

    def test_full250_required_and_invalid_values_rejected(self):
        for raw in (np.zeros((0, 250, 2)), np.zeros((1, 224, 2)),
                    np.full((1, 250, 2), np.nan), np.full((1, 250, 2), np.inf)):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                self.v.fit_statistics(raw)

    def test_statistics_schema_tamper_rejected(self):
        stats = self.v.fit_statistics(np.ones((2, 250, 2), np.float32))
        for field, value in [('schema_version', True), ('sample_count', True),
                             ('fitted_split', 'validation'), ('epsilon', 1e-6),
                             ('mu_d', [1]), ('scale_a', [0, 0]),
                             ('floor_d', [0, 0]), ('points_per_channel', 2)]:
            bad = copy.deepcopy(stats)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.v.validate_statistics(bad)
        stats['extra'] = 1
        with self.assertRaises(ValueError):
            self.v.validate_statistics(stats)

    def test_inference_uses_full250_dc_before_manual_crops(self):
        class ObservingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = []
            def forward(self, x):
                self.seen.append(x.cpu().numpy().copy())
                return torch.stack((x[:, 0].mean(1), x[:, 1].mean(1)), 1)
        raw = np.zeros((1, 250, 2), np.float32)
        raw[:, 224:, 0] = 250
        stats = dict(schema_version=1, protocol='dc_ac_full250_v1', fitted_split='train',
                     sample_count=1, points_per_channel=250, epsilon=1e-12,
                     mu_d=[0., 0.], sigma_d=[1., 1.], sigma_a=[2., 2.],
                     scale_d=[1., 1.], scale_a=[2., 2.], floor_d=[False, False],
                     floor_a=[False, False], channels=2, signal_length=250)
        model = ObservingModel()
        identity = 'a' * 64
        self.v.independent_probabilities(model, raw, [identity], stats, torch.device('cpu'), 3)
        key = json.dumps(['validation', 10, identity, None], separators=(',', ':')).encode()
        starts = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest(), 'big'))).integers(0, 27, 10)
        d = raw.astype(np.float64).mean(1, keepdims=True)
        full = ((d + (raw - d) / 2) / np.sqrt(2)).astype(np.float32)
        expected = np.stack([full[0, s:s+224].T for s in starts])
        np.testing.assert_array_equal(np.concatenate(model.seen), expected)

    def test_initial_hash_binds_name_shape_dtype_and_bytes(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.z = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
                self.a = torch.nn.Parameter(torch.tensor([[3.]], dtype=torch.float32))
        h = hashlib.sha256()
        h.update(b'{"dtype":"float32","name":"a","shape":[1,1]}\n')
        h.update(np.array([[3.]], np.float32).tobytes() + b'\n')
        h.update(b'{"dtype":"float64","name":"z","shape":[1]}\n')
        h.update(np.array([2.], np.float64).tobytes() + b'\n')
        self.assertEqual(self.v.initial_parameters_sha256(Tiny()), h.hexdigest())

    def test_verify_run_api_available(self):
        self.assertTrue(callable(getattr(self.v, 'verify_run', None)), 'Missing read-only verify_run')


class RunVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = importlib.import_module('scripts.verify_paper_dc_ac')
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        root = Path(cls.fixture.name)
        _, manifests = make_dataset(root)
        prepare_dataset(root, manifests, 'cache')
        # Fixture creation may use the producer; verification may not.
        from src.ect.paper_dc_ac_training import train_paper_dc_ac
        config = json.loads((ROOT/'config/paper_baseline_v1.json').read_text())
        config.update(protocol='paper_dc_ac_finite_v1', normalization_protocol='dc_ac_full250_v1',
            purpose='paper_dc_ac_smoke_validation_only', seed=0, device='cpu', max_epochs=2,
            validation_every_epochs=1, batch_size=2, cache_dir='cache')
        train_paper_dc_ac(root, config, 'run')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(self.fixture.name, self.root, dirs_exist_ok=True)
        self.run = self.root/'run'
        self.report = json.loads((self.run/'report.json').read_text())

    def save(self):
        write(self.run/'report.json', self.report)

    def refresh(self, name):
        self.report['files_sha256'][name] = digest(self.run/name)
        if name == 'normalization_dc_ac.json':
            self.report['normalization_sha256'] = digest(self.run/name)
        self.save()

    def reject(self, **kwargs):
        with self.assertRaises((ValueError, RuntimeError, OSError, KeyError, TypeError)):
            self.v.verify_run(self.root, self.run, **kwargs)

    def test_real_cpu_exact_metrics_and_read_only(self):
        before = {p.name:digest(p) for p in self.run.iterdir() if p.is_file()}
        from src.ect import dc_ac, paper_dc_ac_training, paper_crops
        with (patch.object(dc_ac, 'fit_dc_ac', side_effect=AssertionError('producer fit')),
             patch.object(dc_ac, 'transform_dc_ac', side_effect=AssertionError('producer transform')),
             patch.object(paper_dc_ac_training, 'load_paper_dc_ac_model', side_effect=AssertionError('producer loader')),
             patch.object(paper_dc_ac_training, 'predict_paper_dc_ac', side_effect=AssertionError('producer predict')),
             patch.object(paper_crops, 'validation_crops', side_effect=AssertionError('producer crops'))):
            proof = self.v.verify_run(self.root, self.run)
        self.assertEqual(proof['status'], 'verified')
        self.assertEqual(proof['prediction_counts'], {'train':2, 'validation':2})
        self.assertEqual(proof['maximum_metric_error'], 0)
        self.assertEqual(proof['metrics'], self.report['metrics'])
        self.assertEqual(proof['run_dir'], 'run')
        self.assertFalse(proof['heldout_test_evaluated'])
        self.assertFalse(proof['checks']['test_classification_evaluated'])
        self.assertTrue(all(v is True for k, v in proof['checks'].items() if k != 'test_classification_evaluated'))
        self.assertEqual(hashlib.sha256(proof['script']['text'].encode()).hexdigest(), proof['script']['sha256'])
        self.assertEqual(before, {p.name:digest(p) for p in self.run.iterdir() if p.is_file()})

    def test_failure_and_running_markers(self):
        for name in ('failure.json', 'running.marker'):
            (self.run/name).write_text('{}')
            self.reject()
            (self.run/name).unlink()

    def test_report_contract_tampering(self):
        for key, value in [('schema_version', True), ('status', 'running'),
                           ('heldout_test_evaluated', True), ('selected_epoch', 99),
                           ('model_roundtrip_verified', False), ('initial_parameters_sha256', '0'*64),
                           ('execution_registration', {}), ('normalization_sha256', '0'*64)]:
            saved = copy.deepcopy(self.report)
            self.report[key] = value
            self.save()
            with self.subTest(key=key):
                self.reject()
            self.report = saved

    def test_source_and_artifact_checksum_tampering(self):
        for field in ('code_sha256', 'code_sha256_lf', 'files_sha256'):
            saved = copy.deepcopy(self.report)
            self.report[field][next(iter(self.report[field]))] = '0'*64
            self.save()
            with self.subTest(field=field):
                self.reject()
            self.report = saved

    def test_transform_optimizer_and_initial_update_metadata_tampering(self):
        for key, value in [('input_transform_order', ['identity_crop', 'full250_dc_ac']),
                           ('normalization_provenance', {}), ('optimizer', {'name':'SGD'}),
                           ('parameter_update_l2', -1), ('selection_rule', 'last epoch')]:
            saved = copy.deepcopy(self.report)
            self.report[key] = value
            self.save()
            with self.subTest(key=key):
                self.reject()
            self.report = saved

    def test_selected_checkpoint_bn_history_tamper_rejected(self):
        entry = self.report['epochs'][self.report['selected_epoch']-1]
        entry['bn'][next(iter(entry['bn']))]['variance_min'] += .1
        self.save()
        self.reject()

    def test_metric_confusion_boolean_rejected(self):
        self.report['metrics']['train']['confusion_matrix'][0][0] = True
        self.save()
        self.reject()

    def test_failure_during_atomic_publication_never_creates_proof(self):
        proof = self.v.verify_run(self.root, self.run)
        output = self.root/'proof.json'
        (self.run/'failure.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.v.publish_proof(self.root, self.run, output, proof)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.paper-dc-ac-proof-*')))

    def test_runtime_model_forward_poison_rejected(self):
        with patch.object(PaperResNeXt1D, 'forward', lambda self, x: torch.zeros((len(x), 2))):
            self.reject()

    def test_acceptance_proof_rejects_pruned_flags_and_evidence(self):
        proof = self.v.verify_run(self.root, self.run)
        files = {}
        for label, value in proof['checked_files_sha256'].items():
            path = Path(label)
            base = self.root if path.is_relative_to(self.root) else ROOT
            files[path.relative_to(base).as_posix()] = value
        def checked(path, expected=None):
            value = digest(path)
            if expected is not None and value != expected:
                raise ValueError('checksum')
            return value
        self.v.check_acceptance_proof(self.root, proof, self.report, self.run, 'cpu', files, checked,
                                     bn=(self.report['protocol'] == 'paper_dc_ac_bn_v1'))
        for key in ('checks', 'checked_files_sha256', 'script', 'config', 'metrics', 'source_sha256'):
            bad = copy.deepcopy(proof)
            bad[key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.v.check_acceptance_proof(self.root, bad, self.report, self.run, 'cpu', files, checked,
                                             bn=(self.report['protocol'] == 'paper_dc_ac_bn_v1'))
        bad = copy.deepcopy(proof)
        bad['checked_files_sha256'].pop(str(self.run/'model.pt'))
        with self.assertRaises(ValueError):
            self.v.check_acceptance_proof(self.root, bad, self.report, self.run, 'cpu', files, checked,
                                         bn=(self.report['protocol'] == 'paper_dc_ac_bn_v1'))

    def test_normalization_provenance_and_refit_tampering(self):
        original = json.loads((self.run/'normalization_dc_ac.json').read_text())
        for field in ('mu_d', 'sample_count', 'provenance'):
            artifact = copy.deepcopy(original)
            if field == 'mu_d':
                artifact['statistics']['mu_d'][0] += 1
                self.report['normalization'] = artifact['statistics']
            elif field == 'sample_count':
                artifact['statistics']['sample_count'] += 1
                artifact['statistics']['points_per_channel'] += 250
                self.report['normalization'] = artifact['statistics']
            else:
                artifact['provenance']['fit_source_sha256'] = '0'*64
            write(self.run/'normalization_dc_ac.json', artifact)
            self.refresh('normalization_dc_ac.json')
            with self.subTest(field=field):
                self.reject()
            self.report['normalization'] = original['statistics']

    def test_checkpoint_schema_and_metadata_tampering(self):
        saved = torch.load(self.run/'model.pt', weights_only=True)
        for key, value in [('schema_version', True), ('selected_epoch', 99), ('extra', 1),
                           ('normalization', {}), ('classes', [0, 2])]:
            bad = copy.deepcopy(saved)
            bad[key] = value
            torch.save(bad, self.run/'model.pt')
            self.refresh('model.pt')
            with self.subTest(key=key):
                self.reject()

    def test_csv_identity_and_prediction_tampering_even_with_updated_sha(self):
        path = self.run/'train_predictions.csv'
        original = path.read_text()
        for key, value in [('row_index', '1'), ('wave_sha256', '0'*64), ('group_id', '9'),
                           ('class_index', '9'), ('predicted_class_index', '9')]:
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            rows[0][key] = value
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=self.v.CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            self.refresh(path.name)
            with self.subTest(key=key):
                self.reject()
            path.write_text(original)

    def test_metrics_groups_and_history_tampering(self):
        for field in ('metrics', 'group_metrics', 'epochs'):
            saved = copy.deepcopy(self.report)
            if field == 'metrics':
                self.report[field]['train']['accuracy'] += .1
            elif field == 'group_metrics':
                self.report[field]['validation']['unknown'] = self.report['metrics']['validation']
            else:
                self.report[field][0]['learning_rate'] *= 2
            self.save()
            with self.subTest(field=field):
                self.reject()
            self.report = saved

    def test_device_change_is_explicit_and_still_exact(self):
        self.report['environment']['device'] = 'cuda'
        self.save()
        self.reject()
        proof = self.v.verify_run(self.root, self.run, allow_device_change=True)
        self.assertFalse(proof['same_device'])
        self.assertEqual(proof['maximum_metric_error'], 0)

    def test_cli_atomic_no_overwrite(self):
        output = self.root/'proof.json'
        command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/verify_paper_dc_ac.py'),
                   '--project-root', str(self.root), '--run', 'run', '--device', 'cpu', '--output', str(output)]
        done = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        before = output.read_bytes()
        done = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(output.read_bytes(), before)


class BNVerificationTests(RunVerificationTests):
    @classmethod
    def setUpClass(cls):
        cls.v = importlib.import_module('scripts.verify_paper_dc_ac')
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        root = Path(cls.fixture.name)
        _, manifests = make_dataset(root)
        prepare_dataset(root, manifests, 'cache')
        from src.ect.paper_dc_ac_training import train_paper_dc_ac
        from src.ect.paper_dc_ac_bn import run_bn
        config = json.loads((ROOT/'config/paper_baseline_v1.json').read_text())
        config.update(protocol='paper_dc_ac_finite_v1', normalization_protocol='dc_ac_full250_v1',
            purpose='paper_dc_ac_smoke_validation_only', seed=0, device='cpu', max_epochs=2,
            validation_every_epochs=1, batch_size=2, cache_dir='cache')
        train_paper_dc_ac(root, config, 'origin')
        run_bn(root, 'origin', 'run', device='cpu')

    def test_bn_api_and_dispatch_replay_exact_buffers(self):
        self.assertTrue(callable(getattr(self.v, 'verify_bn_run', None)), 'BN verifier missing')
        from src.ect import paper_dc_ac_bn
        with (patch.object(paper_dc_ac_bn, 'load_bn_model', side_effect=AssertionError('producer BN load')),
              patch.object(paper_dc_ac_bn, 'calibrate_dc_ac_bn', side_effect=AssertionError('producer CMA'))):
            proof = self.v.verify_run(self.root, self.run)
        self.assertEqual(proof['protocol'], 'paper_dc_ac_bn_v1')
        self.assertTrue(proof['same_device_calibration_exact'])
        self.assertEqual(proof['calibration_buffer_max_abs_error'], 0)
        self.assertEqual(proof['origin_run'], 'origin')

    def test_bn_parameter_change_rejected_after_rehash(self):
        checkpoint = torch.load(self.run/'model.pt', weights_only=True)
        key = next(k for k in checkpoint['state_dict'] if k.endswith('weight'))
        checkpoint['state_dict'][key].view(-1)[0] += .1
        torch.save(checkpoint, self.run/'model.pt')
        self.refresh('model.pt')
        self.reject()

    def test_bn_counter_buffer_change_rejected_after_rehash(self):
        checkpoint = torch.load(self.run/'model.pt', weights_only=True)
        key = next(k for k in checkpoint['state_dict'] if k.endswith('num_batches_tracked'))
        checkpoint['state_dict'][key] += 1
        torch.save(checkpoint, self.run/'model.pt')
        self.refresh('model.pt')
        self.reject()

    def test_bn_calibration_fixed_protocol_and_origin_tampering(self):
        for key, value in [('origin_model_sha256', '0'*64), ('origin_report_sha256', '0'*64),
                           ('bn_execution_registration', {}), ('bn_before', {}), ('bn_after', {})]:
            saved = copy.deepcopy(self.report)
            self.report[key] = value
            self.save()
            with self.subTest(key=key):
                self.reject()
            self.report = saved
        for key, value in [('calibration_seed', 1), ('order', 'validation'), ('passes', 2)]:
            saved = copy.deepcopy(self.report)
            self.report['calibration']['config'][key] = value
            self.save()
            with self.subTest(key=key):
                self.reject()
            self.report = saved

    def test_bn_origin_failure_blocks_verification(self):
        (self.root/'origin/failure.json').write_text('{}')
        self.reject()


class WholeTrainSmokeTests(unittest.TestCase):
    def test_capped_smoke_still_fits_all_clean_training_rows(self):
        from scripts.verify_paper_dc_ac import verify_run
        from src.ect.paper_dc_ac_training import train_paper_dc_ac
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, manifests = make_dataset(root)
            expanded = np.repeat(raw, 2, axis=3)
            expanded[:, :, :, 1] += 11
            np.save(root/'data/raw/fixture.npy', expanded)
            from src.ect.audit import waveform_hash
            summary = json.loads((manifests/'summary.json').read_text())
            summary['inputs']['train']['sha256'] = digest(root/'data/raw/fixture.npy')
            for person, split in enumerate(('train', 'validation', 'test')):
                rows = [json.loads(line) for line in (manifests/f'{split}.jsonl').read_text().splitlines()]
                added = copy.deepcopy(rows)
                for cls, row in enumerate(added):
                    row['representative']['repeat'] = 1
                    row['origins'][0]['repeat'] = 1
                    row['wave_sha256'] = waveform_hash(expanded[person, 0, 0, 1, cls])
                path = manifests/f'{split}.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows+added))
                summary['manifest_sha256'][path.name] = digest(path)
                summary['sample_counts'][split] = 4
                summary['class_counts'][split] = {'0':2, '1':2}
            write(manifests/'summary.json', summary)
            prepare_dataset(root, manifests, 'cache')
            config = json.loads((ROOT/'config/paper_baseline_v1.json').read_text())
            config.update(protocol='paper_dc_ac_finite_v1', normalization_protocol='dc_ac_full250_v1',
                purpose='paper_dc_ac_smoke_validation_only', seed=0, device='cpu', max_epochs=2,
                validation_every_epochs=1, batch_size=2, cache_dir='cache', train_per_class=1, validation_per_class=1)
            report = train_paper_dc_ac(root, config, 'run')
            proof = verify_run(root, 'run')
            self.assertEqual(proof['prediction_counts']['train'], 2)
            self.assertEqual(report['normalization']['sample_count'], 4)
            full = np.load(root/'cache/train_x.npy').astype(np.float64)
            np.testing.assert_allclose(report['normalization']['mu_d'], full.mean(1).mean(0), rtol=1e-10, atol=1e-12)


@unittest.skipUnless(torch.cuda.is_available(), 'Requires actual GPU execution context')
class CUDAInteropTests(unittest.TestCase):
    def test_actual_cuda_training_bn_and_cpu_migration(self):
        from scripts.verify_paper_dc_ac import verify_run
        from src.ect.paper_dc_ac_training import train_paper_dc_ac
        from src.ect.paper_dc_ac_bn import run_bn
        with tempfile.TemporaryDirectory() as tmp:
            _, manifests = make_dataset(tmp)
            prepare_dataset(tmp, manifests, 'cache')
            config = json.loads((ROOT/'config/paper_baseline_v1.json').read_text())
            config.update(protocol='paper_dc_ac_finite_v1', normalization_protocol='dc_ac_full250_v1',
                purpose='paper_dc_ac_smoke_validation_only', seed=0, device='cuda', max_epochs=2,
                validation_every_epochs=1, batch_size=2, cache_dir='cache')
            train_paper_dc_ac(tmp, config, 'origin')
            run_bn(tmp, 'origin', 'bn', device='cuda')
            for run in ('origin', 'bn'):
                same = verify_run(tmp, run, device='cuda')
                transferred = verify_run(tmp, run, device='cpu', allow_device_change=True)
                self.assertEqual(same['prediction_counts'], {'train':2, 'validation':2})
                self.assertEqual(same['maximum_metric_error'], 0)
                self.assertEqual(transferred['metrics'], same['metrics'])
                self.assertEqual(transferred['maximum_metric_error'], 0)
                if run == 'bn':
                    self.assertTrue(same['same_device_calibration_exact'])
                    self.assertEqual(same['calibration_buffer_max_abs_error'], 0)


if __name__ == '__main__':
    unittest.main()
