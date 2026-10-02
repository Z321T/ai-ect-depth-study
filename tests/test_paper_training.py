import copy
import csv
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
from src.ect.prepare import prepare_dataset, PreparedDataset

BASE = json.loads((Path(__file__).resolve().parents[1] / 'config/paper_baseline_v1.json').read_text())
SMOKE = BASE | dict(purpose='paper_runner_smoke_validation_only', seed=0,
                    device='cpu', max_epochs=2, batch_size=2,
                    validation_every_epochs=1, cache_dir='cache')


class PaperTrainingTests(unittest.TestCase):
    def module(self):
        from src.ect import paper_training
        return paper_training

    def test_original_lr_milestones_are_not_rescaled_and_selection_uses_accuracy(self):
        m = self.module()
        self.assertEqual([m.learning_rate_for_epoch(SMOKE, e) for e in [1, 1000, 4999, 5000, 7500]],
                         [4e-5, 4e-5, 4e-5, 4e-6, 4e-7])
        records = [dict(epoch=1, validation=None),
                   dict(epoch=2, validation=dict(accuracy=.8, macro_f1=.1)),
                   dict(epoch=3, validation=dict(accuracy=.8, macro_f1=.9)),
                   dict(epoch=4, validation=dict(accuracy=.7, macro_f1=1.))]
        self.assertEqual(m.select_epoch(records)['epoch'], 2)

    def test_real_cpu_training_seals_test_preserves_crop_identity_and_reloads(self):
        m = self.module()
        class Guard(dict):
            def __getitem__(self, key):
                if key == 'test':
                    raise AssertionError('test classification accessed')
                return super().__getitem__(key)
        class Sealed(PreparedDataset):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.arrays = Guard(self.arrays)
        with tempfile.TemporaryDirectory() as tmp:
            _, manifests = make_dataset(tmp)
            prepare_dataset(tmp, manifests, 'cache')
            calls = []
            original = m.training_crops
            def observe(raw, ids, seed, epoch):
                actual = original(raw, ids, seed, epoch)
                again = original(raw[::-1], ids[::-1], seed, epoch)[::-1]
                np.testing.assert_array_equal(actual, again)
                calls.append((seed, epoch))
                return actual
            with patch.object(m, 'PreparedDataset', Sealed), patch.object(m, 'training_crops', observe):
                report = m.train_paper_baseline(tmp, SMOKE, 'run')
            self.assertEqual(calls, [(0, 0), (0, 1)])
            self.assertFalse(report['heldout_test_evaluated'])
            self.assertEqual(report['sample_counts'], dict(train=2, validation=2))
            self.assertGreater(report['parameter_update_l2'], 0)
            self.assertEqual(report['optimizer'], dict(name='Adam', betas=[.9, .999], eps=1e-8, weight_decay=0.))
            self.assertEqual(report['selected_epoch'], m.select_epoch(report['epochs'])['epoch'])
            bundle = m.load_paper_model(Path(tmp, 'run'))
            self.assertEqual(bundle['normalization']['fitted_split'], 'train')
            self.assertTrue(all(t.device.type == 'cpu' for t in bundle['model'].state_dict().values()))
            with PreparedDataset(tmp, 'cache') as store:
                for split in ('train', 'validation'):
                    records = [json.loads(s) for s in (manifests / f'{split}.jsonl').read_text().splitlines()]
                    actual = m.predict_paper(bundle, store.arrays[split][0], [r['wave_sha256'] for r in records], batch_size=2)
                    with Path(tmp, 'run', f'{split}_predictions.csv').open() as handle:
                        rows = list(csv.DictReader(handle))
                    np.testing.assert_array_equal(actual.argmax(1), [int(r['predicted_class_index']) for r in rows])
                    self.assertEqual([int(r['row_index']) for r in rows], [0, 1])
                    self.assertEqual([r['wave_sha256'] for r in rows], [r['wave_sha256'] for r in records])

    def test_interval_records_all_epochs_without_early_stop(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            report = m.train_paper_baseline(tmp, SMOKE | dict(max_epochs=4, validation_every_epochs=2), 'run')
            self.assertEqual([e['epoch'] for e in report['epochs']], [1, 2, 3, 4])
            self.assertEqual([e['epoch'] for e in report['epochs'] if e['validation'] is not None], [2, 4])
            self.assertTrue(all(e['learning_rate'] == 4e-5 for e in report['epochs']))

    def test_invalid_config_and_full_run_without_registry_do_not_claim(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            for field, value in [('batch_size', 0), ('max_epochs', True), ('seed', -1),
                                 ('learning_rate', .001), ('patience', 8), ('noise_augmentation', True),
                                 ('selection', ['macro_f1']), ('learning_rate_milestones_epochs', [500,750]),
                                 ('test_classification_enabled', True), ('device', 'gpu')]:
                path = Path(tmp, field)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    m.train_paper_baseline(tmp, SMOKE | {field:value}, path)
                self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, 'registration'):
                m.train_paper_baseline(tmp, BASE | dict(seed=0), 'full')
            self.assertFalse(Path(tmp, 'full').exists())

    def test_manifest_labels_and_failure_marker_and_directory_ownership(self):
        m = self.module()
        class Wrong(PreparedDataset):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                raw, labels = self.arrays['train']
                self.arrays['train'] = raw, np.array(labels[::-1])
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            with patch.object(m, 'PreparedDataset', Wrong), self.assertRaisesRegex(ValueError, 'manifest'):
                m.train_paper_baseline(tmp, SMOKE, 'run')
            self.assertEqual(json.loads(Path(tmp, 'run/failure.json').read_text())['status'], 'failed')
            self.assertFalse(Path(tmp, 'run/report.json').exists())
            with self.assertRaises(FileExistsError):
                m.train_paper_baseline(tmp, SMOKE, 'run')
            with self.assertRaisesRegex(ValueError, 'Failed'):
                m.load_paper_model(Path(tmp, 'run'))

    def test_rejects_weight_metadata_source_and_success_failure_mixtures(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            report = m.train_paper_baseline(tmp, SMOKE, 'run')
            output = Path(tmp, 'run')
            for field, value in [('selected_epoch', 999), ('config_sha256', '0'*64),
                                 ('code_sha256_lf', {}), ('classes', [0,1,2]), ('schema_version', 2)]:
                changed = copy.deepcopy(report)
                changed[field] = value
                m.write_json(output / 'report.json', changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    m.load_paper_model(output)
            m.write_json(output / 'report.json', report)
            with (output / 'model.pt').open('ab') as handle:
                handle.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                m.load_paper_model(output)
            m.write_json(output / 'failure.json', {'status':'failed'})
            with self.assertRaisesRegex(ValueError, 'Failed'):
                m.load_paper_model(output)

    def test_execution_registration_is_rechecked_before_success_publication(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            # The registry boundary is isolated here; real registry validation
            # and the full budget are covered by registration integration tests.
            config = SMOKE | dict(purpose=m.FULL_PURPOSE)
            with patch.object(m,'validate_config'), patch('src.ect.paper_registration.validate_registered_run',
                    side_effect=[{'registry_sha256':'before'}, {'registry_sha256':'after'}]):
                with self.assertRaisesRegex(ValueError, 'registration changed'):
                    m.train_paper_baseline(tmp, config, 'run', registry_path='registry.json')
            self.assertTrue(Path(tmp,'run/failure.json').exists())
            self.assertFalse(Path(tmp,'run/report.json').exists())

    def test_source_edited_after_import_cannot_be_reported_as_executed(self):
        self.module()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp)
            shutil.copytree(root/'src',copied/'src',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(root/'scripts',copied/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
            _, folder = make_dataset(copied)
            prepare_dataset(copied,folder,'cache')
            code = """import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from src.ect import paper_training as m
config=json.loads(sys.argv[2])
p=Path(sys.argv[1],'src/ect/paper_crops.py')
p.write_bytes(p.read_bytes()+b'\\n\\ndef training_crops(*a, **kw): raise RuntimeError(\"changed\")\\n')
try: m.train_paper_baseline(sys.argv[1],config,'run')
except ValueError as e:
    assert 'import' in str(e).lower(),str(e)
else: raise AssertionError('stale imported source was accepted')
"""
            result = subprocess.run([sys.executable,'-B','-c',code,tmp,json.dumps(SMOKE)],
                                    cwd=tmp,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_dependency_imported_before_runner_cannot_hide_edited_disk_source(self):
        self.module()
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            copied=Path(tmp)
            shutil.copytree(root/'src',copied/'src',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(root/'scripts',copied/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
            _, folder=make_dataset(copied)
            prepare_dataset(copied,folder,'cache')
            code="""import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from src.ect import paper_crops
p=Path(sys.argv[1],'src/ect/paper_crops.py')
p.write_bytes(p.read_bytes()+b'\\n\\ndef training_crops(*a, **kw): raise RuntimeError(\"changed\")\\n')
from src.ect import paper_training as m
try: m.train_paper_baseline(sys.argv[1],json.loads(sys.argv[2]),'run')
except ValueError as e:
    assert 'import' in str(e).lower(),str(e)
else: raise AssertionError('stale dependency code was accepted')
"""
            result=subprocess.run([sys.executable,'-B','-c',code,tmp,json.dumps(SMOKE)],cwd=tmp,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_warm_dependency_default_arguments_match_recorded_source(self):
        self.module()
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            copied=Path(tmp)
            shutil.copytree(root/'src',copied/'src',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(root/'scripts',copied/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
            code="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from src.ect import paper_models
p=Path(sys.argv[1],'src/ect/paper_models.py')
p.write_text(p.read_text().replace('dilation: int = 1','dilation: int = 2'))
from src.ect import paper_training as m
try: m.verify_imported_sources()
except ValueError as e: assert 'default' in str(e).lower(),str(e)
else: raise AssertionError('stale defaults accepted')
"""
            result=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=tmp,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_failed_final_progress_and_unwritable_failure_marker_never_load_success(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp,folder,'cache')
            original = m.write_json
            def faulty(path,value):
                path=Path(path)
                if path.name=='failure.json' or (path.name=='progress.json' and value.get('status')=='complete'):
                    raise PermissionError('storage is unwritable')
                return original(path,value)
            with patch.object(m,'write_json',faulty),self.assertRaises(PermissionError):
                m.train_paper_baseline(tmp,SMOKE,'run')
            with self.assertRaises((ValueError,FileNotFoundError)):
                m.load_paper_model(Path(tmp,'run'))

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires accessible actual CUDA')
    def test_real_cuda_training_and_cpu_reload(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, folder = make_dataset(tmp)
            prepare_dataset(tmp, folder, 'cache')
            report = m.train_paper_baseline(tmp, SMOKE, 'run', device_override='cuda')
            self.assertEqual(report['environment']['device'], 'cuda')
            self.assertGreater(report['parameter_update_l2'], 0)
            bundle = m.load_paper_model(Path(tmp, 'run'), 'cpu')
            self.assertEqual(next(bundle['model'].parameters()).device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
