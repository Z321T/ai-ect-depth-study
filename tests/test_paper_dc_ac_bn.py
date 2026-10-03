"""Train-only fixed BN sidecar: real inputs, BN updates and artifact checks."""
import copy
import importlib.util
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

from src.ect.paper_crops import training_crops
from src.ect.prepare import PreparedDataset, prepare_dataset
from fixture_data import make_dataset

ROOT = Path(__file__).resolve().parents[1]


class PaperDCACBNTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('src.ect.paper_dc_ac_bn'),
                             'E2 fixed BN sidecar is not implemented')
        from src.ect import paper_dc_ac_bn
        return paper_dc_ac_bn

    def model(self, device='cpu'):
        torch.manual_seed(3)
        return torch.nn.Sequential(torch.nn.Conv1d(2, 3, 1),
            torch.nn.BatchNorm1d(3, momentum=.01), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
            torch.nn.Linear(3, 2)).to(device).eval()

    def stats(self):
        return dict(schema_version=1, protocol='dc_ac_full250_v1', fitted_split='train',
            sample_count=192, points_per_channel=48000, epsilon=1e-12,
            mu_d=[.1, -.2], sigma_d=[2., 3.], sigma_a=[.5, 1.5],
            scale_d=[2., 3.], scale_a=[.5, 1.5], floor_d=[False, False],
            floor_a=[False, False], channels=2, signal_length=250)

    def manual_pass(self, original, raw, ids, stats):
        expected = copy.deepcopy(original).train()
        for bn in expected.modules():
            if isinstance(bn, torch.nn.BatchNorm1d):
                bn.reset_running_stats()
                bn.momentum = None
        device = next(expected.parameters()).device
        with torch.no_grad():
            for start in range(0, len(raw), 128):
                values = raw[start:start + 128].astype(np.float64)
                dc = values.mean(axis=1, keepdims=True)
                full = (((dc - stats['mu_d']) / stats['scale_d'] +
                         (values - dc) / stats['scale_a']) / np.sqrt(2)).astype(np.float32)
                crop = training_crops(full, ids[start:start + 128], 0, 0)
                expected(torch.from_numpy(np.ascontiguousarray(crop.transpose(0, 2, 1))).to(device))
        return expected

    def test_fixed_protocol_and_configuration(self):
        m = self.module()
        self.assertEqual(m.PROTOCOL, 'paper_dc_ac_bn_v1')
        c = m.fixed_config()
        self.assertEqual(c['calibration_batch_size'], 128)
        self.assertEqual(c['calibration_seed'], 0)
        self.assertEqual(c['calibration_epoch'], 0)
        self.assertEqual(c['order'], 'manifest')
        self.assertEqual(c['passes'], 1)
        self.assertFalse(c['test_classification_enabled'])
        altered = c | {'passes': True}
        self.assertFalse(m.config_matches(altered))

    def test_state_comparison_rejects_learned_and_non_bn_changes(self):
        m = self.module()
        original = self.model()
        original.register_buffer('untouched', torch.tensor(7))
        changed = copy.deepcopy(original)
        changed[1].running_mean.add_(1)
        self.assertEqual(m.state_changes(original, changed), ['1.running_mean'])
        changed.untouched.add_(1)
        with self.assertRaisesRegex(ValueError, 'Non-BN'):
            m.state_changes(original, changed)
        changed = copy.deepcopy(original)
        with torch.no_grad():
            changed[1].weight.add_(1)
        with self.assertRaises(ValueError):
            m.state_changes(original, changed)

    def test_full250_transform_precedes_crop_and_tail64_uses_batch_cma(self):
        m = self.module()
        raw = np.random.default_rng(1).normal(size=(192, 250, 2)).astype('f4')
        raw[:, :26, 0] += 80
        raw[128:, :, 1] += 30
        ids = [str(i) for i in range(len(raw))]
        original = self.model()
        expected = self.manual_pass(original, raw, ids, self.stats())
        actual, summary = m.calibrate_dc_ac_bn(original, raw, ids, self.stats())
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)
        self.assertEqual(summary['sample_count'], 192)
        self.assertEqual(summary['batch_count'], 2)
        self.assertEqual(summary['last_batch_size'], 64)
        self.assertEqual(int(actual[1].num_batches_tracked), 2)
        self.assertEqual(actual[1].momentum, .01)
        self.assertFalse(actual.training)

    def test_original_modes_gradients_state_and_stats_are_preserved(self):
        m = self.module()
        original = self.model()
        original[2].train()
        for parameter in original.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = copy.deepcopy(original.state_dict())
        modes = [x.training for x in original.modules()]
        stats = self.stats()
        frozen = copy.deepcopy(stats)
        raw = np.random.default_rng(2).normal(size=(40, 250, 2)).astype('f4')
        initial_raw = raw.copy()
        updated, summary = m.calibrate_dc_ac_bn(original, raw, [str(i) for i in range(40)], stats)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, original.state_dict()[name]), name)
        self.assertEqual(modes, [x.training for x in original.modules()])
        for name, parameter in original.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(updated.named_parameters())[name]))
            self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))
        np.testing.assert_array_equal(raw, initial_raw)
        self.assertEqual(stats, frozen)
        self.assertTrue(summary['parameters_unchanged'])
        self.assertTrue(summary['original_state_unchanged'])

    def test_nontrain_invalid_inputs_untracked_bn_and_dropout_are_rejected(self):
        m = self.module()
        original = self.model()
        raw = np.ones((2, 250, 2), np.float32)
        for split in ('validation', 'test'):
            with self.assertRaises(ValueError):
                m.calibrate_dc_ac_bn(original, raw, ['a', 'b'], self.stats(), split=split)
        for values, ids in ((raw, ['a']), (raw[:0], []), (raw * np.nan, ['a', 'b']),
                            (raw[:, :224], ['a', 'b']), (raw.astype(complex), ['a', 'b']),
                            (raw, 'ab')):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                m.calibrate_dc_ac_bn(original, values, ids, self.stats())
        for model in (torch.nn.BatchNorm1d(2, track_running_stats=False),
                      torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Dropout())):
            with self.assertRaises(ValueError):
                m.calibrate_dc_ac_bn(model, raw, ['a', 'b'], self.stats())

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires accessible actual CUDA')
    def test_real_cuda_manual_tail_cma_and_cpu_state_portability(self):
        m = self.module()
        original = self.model('cuda')
        raw = np.random.default_rng(3).normal(size=(192, 250, 2)).astype('f4')
        ids = [str(i) for i in range(len(raw))]
        expected = self.manual_pass(original, raw, ids, self.stats())
        updated, summary = m.calibrate_dc_ac_bn(original, raw, ids, self.stats())
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, updated.state_dict()[name]), name)
        self.assertTrue(summary['parameters_unchanged'])
        self.assertTrue(all(torch.isfinite(value).all() for value in updated.cpu().state_dict().values()))

    def fixture(self, root, device='cpu'):
        from src.ect import paper_dc_ac_training as training
        _, manifests = make_dataset(root)
        prepare_dataset(root, manifests, 'cache')
        base = json.loads((ROOT / 'config/paper_baseline_v1.json').read_text())
        config = base | dict(protocol='paper_dc_ac_finite_v1',
            normalization_protocol='dc_ac_full250_v1', purpose='paper_dc_ac_smoke_validation_only',
            seed=0, device='cpu', max_epochs=2, batch_size=2,
            validation_every_epochs=1, cache_dir='cache')
        return training.train_paper_dc_ac(root, config, 'origin', device_override=device)

    def test_cli_help(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/recalibrate_paper_dc_ac_bn.py'),
            '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--origin', result.stdout)

    def test_real_two_class_smoke_preserves_history_selection_and_normalization(self):
        m = self.module()
        self.assertTrue(callable(getattr(m, 'run_bn', None)), 'BN runner must exist')
        class Guard(dict):
            def __getitem__(self, key):
                if key == 'test':
                    raise AssertionError('test classification accessed')
                return super().__getitem__(key)
        class Sealed(PreparedDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.arrays = Guard(self.arrays)
        with tempfile.TemporaryDirectory() as tmp:
            old = self.fixture(tmp)
            origin = Path(tmp, 'origin')
            hashes = {name: m.file_sha256(origin / name) for name in (*m.ARTIFACTS, 'report.json')}
            with patch.object(m, 'PreparedDataset', Sealed):
                report = m.run_bn(tmp, 'origin', 'new', device='cpu')
            self.assertEqual(report['protocol'], 'paper_dc_ac_bn_v1')
            for field in ('config', 'config_sha256', 'epochs', 'selected_epoch', 'normalization',
                          'sample_counts', 'classes', 'initial_parameters_sha256'):
                self.assertEqual(report[field], old[field], field)
            self.assertEqual(report['calibration']['config'], m.fixed_config())
            self.assertEqual(report['calibration']['sample_count'], 2)
            self.assertFalse(report['heldout_test_evaluated'])
            self.assertEqual(report['baseline_metrics'], old['metrics'])
            self.assertEqual(m.load_bn_model(tmp, 'new')['report'], report)
            for name, digest in hashes.items():
                self.assertEqual(m.file_sha256(origin / name), digest, name)
            for name in ('normalization_dc_ac.json', 'train_rows.json', 'validation_rows.json'):
                self.assertEqual((origin / name).read_bytes(), Path(tmp, 'new', name).read_bytes())
            with self.assertRaises(FileExistsError):
                m.run_bn(tmp, 'origin', 'new')
            with self.assertRaises(ValueError):
                m.run_bn(tmp, 'origin', 'origin/nested')

    def test_loader_rejects_semantic_tamper_even_with_rehashed_artifacts(self):
        m = self.module()
        self.assertTrue(callable(getattr(m, 'load_bn_model', None)), 'BN loader must exist')
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            report = m.run_bn(tmp, 'origin', 'new', device='cpu')
            run = Path(tmp, 'new')
            saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
            for mutation in ('parameter', 'counter', 'selected_epoch', 'config'):
                altered = copy.deepcopy(saved)
                if mutation == 'parameter':
                    key = next(name for name in altered['state_dict'] if name.endswith('weight'))
                    altered['state_dict'][key].add_(1)
                elif mutation == 'counter':
                    key = next(name for name in altered['state_dict'] if name.endswith('num_batches_tracked'))
                    altered['state_dict'][key].add_(1)
                elif mutation == 'selected_epoch':
                    altered['selected_epoch'] = True
                else:
                    altered['config']['protocol'] = 'paper_dc_ac_bn_v1'
                torch.save(altered, run / 'model.pt')
                changed = copy.deepcopy(report)
                changed['files_sha256']['model.pt'] = m.file_sha256(run / 'model.pt')
                m.write_json(run / 'report.json', changed)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    m.load_bn_model(tmp, 'new')
            torch.save(saved, run / 'model.pt')
            report['files_sha256']['model.pt'] = m.file_sha256(run / 'model.pt')
            for field, value in (('epochs', []), ('baseline_metrics', {}),
                    ('code_sha256', {}), ('selected_epoch', True), ('normalization_sha256', '0' * 64)):
                changed = copy.deepcopy(report)
                changed[field] = value
                m.write_json(run / 'report.json', changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    m.load_bn_model(tmp, 'new')

    def test_failure_is_marked_and_no_complete_report_remains(self):
        m = self.module()
        self.assertTrue(callable(getattr(m, 'run_bn', None)), 'BN runner must exist')
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            origin_hash = m.file_sha256(Path(tmp, 'origin/model.pt'))
            with patch.object(torch, 'save', side_effect=OSError('injected write failure')):
                with self.assertRaisesRegex(OSError, 'injected'):
                    m.run_bn(tmp, 'origin', 'new', device='cpu')
            self.assertTrue(Path(tmp, 'new/failure.json').exists())
            self.assertFalse(Path(tmp, 'new/report.json').exists())
            self.assertEqual(m.file_sha256(Path(tmp, 'origin/model.pt')), origin_hash)
            with self.assertRaises(ValueError):
                m.load_bn_model(tmp, 'new')

    def test_loader_rejects_origin_drift_and_running_or_failure_markers(self):
        m = self.module()
        self.assertTrue(callable(getattr(m, 'run_bn', None)), 'BN runner must exist')
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            m.run_bn(tmp, 'origin', 'new', device='cpu')
            for name in ('failure.json', 'running.marker'):
                path = Path(tmp, 'new', name)
                path.write_text('{}')
                with self.subTest(marker=name), self.assertRaises(ValueError):
                    m.load_bn_model(tmp, 'new')
                path.unlink()
            with Path(tmp, 'origin/model.pt').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Origin'):
                m.load_bn_model(tmp, 'new')

    def test_full_origin_requires_main_registry_binding_fixed_bn_and_sources(self):
        m = self.module()
        from src.ect import paper_dc_ac_training as training
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = dict(purpose=training.FULL_PURPOSE, sample_counts={'train': 24000, 'validation': 3200},
                       execution_registration=None)
            with self.assertRaisesRegex(ValueError, 'registration'):
                m._origin_registration(root, root / 'origin', old)
            registry = dict(bn_calibration=m.fixed_config(), code_sha256=m.source_hashes())
            registry['bn_calibration']['passes'] = True
            m.write_json(root / 'registry.json', registry)
            old['execution_registration'] = dict(registry_path='registry.json',
                registry_sha256=m.file_sha256(root / 'registry.json'))
            with self.assertRaisesRegex(ValueError, 'fixed BN'):
                m._origin_registration(root, root / 'origin', old)
            registry['bn_calibration'] = m.fixed_config()
            registry['code_sha256'].pop('src/ect/paper_dc_ac_bn.py')
            m.write_json(root / 'registry.json', registry)
            old['execution_registration']['registry_sha256'] = m.file_sha256(root / 'registry.json')
            with self.assertRaisesRegex(ValueError, 'sidecar sources'):
                m._origin_registration(root, root / 'origin', old)

    def test_loader_rejects_changed_rows_normalization_or_state_dtype(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            report = m.run_bn(tmp, 'origin', 'new', device='cpu')
            run = Path(tmp, 'new')
            for name, value in (('train_rows.json', [1, 0]), ('validation_rows.json', [True, 1]),
                                ('normalization_dc_ac.json', {})):
                original = (run / name).read_bytes()
                m.write_json(run / name, value)
                changed = copy.deepcopy(report)
                changed['files_sha256'][name] = m.file_sha256(run / name)
                m.write_json(run / 'report.json', changed)
                with self.subTest(artifact=name), self.assertRaises(ValueError):
                    m.load_bn_model(tmp, 'new')
                (run / name).write_bytes(original)
            saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
            name = next(iter(saved['state_dict']))
            saved['state_dict'][name] = saved['state_dict'][name].double()
            torch.save(saved, run / 'model.pt')
            report['files_sha256']['model.pt'] = m.file_sha256(run / 'model.pt')
            m.write_json(run / 'report.json', report)
            with self.assertRaisesRegex(ValueError, 'dtype'):
                m.load_bn_model(tmp, 'new')

    def test_failed_success_marker_removal_remains_unloadable(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            unlink = Path.unlink
            def fail_unlink(path, *args, **kwargs):
                if path.name == 'running.marker':
                    raise OSError('injected marker removal failure')
                return unlink(path, *args, **kwargs)
            with patch.object(Path, 'unlink', fail_unlink):
                with self.assertRaisesRegex(OSError, 'injected marker'):
                    m.run_bn(tmp, 'origin', 'new', device='cpu')
            self.assertTrue(Path(tmp, 'new/running.marker').exists())
            self.assertTrue(Path(tmp, 'new/failure.json').exists())
            self.assertFalse(Path(tmp, 'new/report.json').exists())
            with self.assertRaises(ValueError):
                m.load_bn_model(tmp, 'new')

    def test_edited_imported_sidecar_or_cli_cannot_be_claimed_as_executed_source(self):
        for target in ('src/ect/paper_dc_ac_bn.py', 'scripts/recalibrate_paper_dc_ac_bn.py'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                copied = Path(tmp)
                for folder in ('src', 'scripts'):
                    shutil.copytree(ROOT / folder, copied / folder,
                                    ignore=shutil.ignore_patterns('__pycache__'))
                registry = 'results/experiments/paper_baseline_v1/registry.json'
                (copied / registry).parent.mkdir(parents=True)
                shutil.copy2(ROOT / registry, copied / registry)
                code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.ect import paper_dc_ac_bn as m
path=Path(sys.argv[1])/sys.argv[2]
path.write_bytes(path.read_bytes()+b'\\n# edited after import\\n')
try:
    m.source_hashes()
except ValueError:
    pass
else:
    raise AssertionError('stale imported source was attributed to edited disk source')
'''
                result = subprocess.run([sys.executable, '-B', '-c', code, tmp, target],
                                        cwd=tmp, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_source_drift_during_intervention_invalidates_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp)
            for folder in ('src', 'scripts'):
                shutil.copytree(ROOT / folder, copied / folder,
                                ignore=shutil.ignore_patterns('__pycache__'))
            registry = 'results/experiments/paper_baseline_v1/registry.json'
            (copied / registry).parent.mkdir(parents=True)
            shutil.copy2(ROOT / registry, copied / registry)
            (copied / 'config').mkdir()
            shutil.copy2(ROOT / 'config/paper_baseline_v1.json', copied / 'config/paper_baseline_v1.json')
            shutil.copy2(ROOT / 'tests/fixture_data.py', copied / 'fixture_data.py')
            code = '''import json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import torch
from src.ect import paper_dc_ac_bn as m
from src.ect import paper_dc_ac_training as parent
from src.ect.prepare import prepare_dataset
from fixture_data import make_dataset
root=Path(sys.argv[1])
_, manifests=make_dataset(root)
prepare_dataset(root,manifests,'cache')
base=json.loads((root/'config/paper_baseline_v1.json').read_text())
cfg=base|dict(protocol='paper_dc_ac_finite_v1',normalization_protocol='dc_ac_full250_v1',
    purpose=parent.SMOKE_PURPOSE,seed=0,device='cpu',max_epochs=2,batch_size=2,
    validation_every_epochs=1,cache_dir='cache')
parent.train_paper_dc_ac(root,cfg,'origin')
digest=m.file_sha256(root/'origin/model.pt')
save=torch.save
def drift(*args,**kwargs):
    save(*args,**kwargs)
    path=root/'scripts/recalibrate_paper_dc_ac_bn.py'
    path.write_bytes(path.read_bytes()+b'\\n# edited during execution\\n')
with patch.object(torch,'save',drift):
    try: m.run_bn(root,'origin','new',device='cpu')
    except ValueError: pass
    else: raise AssertionError('source drift was accepted')
assert (root/'new/failure.json').exists()
assert not (root/'new/report.json').exists()
assert digest==m.file_sha256(root/'origin/model.pt')
'''
            result = subprocess.run([sys.executable, '-B', '-c', code, tmp], cwd=tmp,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires accessible actual CUDA')
    def test_real_cuda_origin_bn_run_and_cpu_migration(self):
        m = self.module()
        from src.ect import paper_dc_ac_training as training
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp, device='cuda')
            report = m.run_bn(tmp, 'origin', 'new', device='cuda')
            self.assertEqual(report['environment']['device'], 'cuda')
            cpu = m.load_bn_model(tmp, 'new', device='cpu')
            gpu = m.load_bn_model(tmp, 'new', device='cuda')
            with PreparedDataset(tmp, 'cache') as store:
                folder = Path(tmp) / Path(store.metadata['summary_path']).parent
                for split in ('train', 'validation'):
                    ids = [json.loads(line)['wave_sha256'] for line in (folder / f'{split}.jsonl').read_text().splitlines()]
                    raw = store.arrays[split][0]
                    a = training.predict_paper_dc_ac(cpu, raw, ids, batch_size=128)
                    b = training.predict_paper_dc_ac(gpu, raw, ids, batch_size=128)
                    np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
                    self.assertTrue(np.isfinite(a).all())


if __name__ == '__main__':
    unittest.main()
