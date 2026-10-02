"""Synthetic acceptance evidence tests; these fixtures are NOT runner acceptance."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from fixture_data import make_dataset
from src.ect.prepare import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / 'config/paper_baseline_v1.json').read_text())
CHECKS = ('cpu_training', 'cuda_training', 'independent_crop_reload', 'csv_metrics',
          'full_regression', 'independent_code_review')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class PaperRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real CPU-produced reports/checkpoints/independent predictions are
        # reused. CUDA metadata is simulated ONLY in this CPU unit fixture;
        # it is never presented as an actual CUDA runner acceptance.
        from src.ect.paper_training import train_paper_baseline
        from scripts.verify_paper_run import verify_paper_run
        cls.prototype=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.prototype.cleanup)
        root=Path(cls.prototype.name)
        _, manifests=make_dataset(root)
        prepare_dataset(root,manifests,BASE['cache_dir'])
        config=BASE | dict(purpose='paper_runner_smoke_validation_only',seed=0,max_epochs=2,
                          validation_every_epochs=1,train_per_class=2,validation_per_class=2)
        for device, folder in [('cpu','runner_smoke'),('cuda','runner_smoke_cuda')]:
            run=root/'results'/folder
            train_paper_baseline(root,config,run,device_override='cpu')
            if device=='cuda':
                report=json.loads((run/'report.json').read_text())
                report['environment'].update(device='cuda',device_name='CPU-only synthetic CUDA metadata fixture')
                write(run/'report.json',report)
            path=root/'results'/f'{folder}_verification.json'
            proof=verify_paper_run(root,run,path,device='cpu',split='both')
            if device=='cuda':
                proof['device']='cuda'
                proof['device_request']='cuda'
                proof['fixture_only']=True
                write(path,proof)

    def module(self):
        from src.ect import paper_registration
        return paper_registration

    def fixture(self, root):
        """Use real cache validation and only synthetic acceptance/design JSON."""
        m = self.module()
        shutil.copytree(self.prototype.name,root,dirs_exist_ok=True)
        cache = root / BASE['cache_dir']
        metadata = json.loads((cache/'metadata.json').read_text())
        config = root / 'config/paper_baseline_v1.json'
        config.parent.mkdir()
        config.write_bytes((ROOT / 'config/paper_baseline_v1.json').read_bytes())
        component = root / 'results/paper_components/acceptance_v1'
        write(component / 'report.json', {'fixture_only': True})
        write(component / 'verification.json', {'fixture_only': True})
        design = json.loads((ROOT / 'results/paper_components/baseline_design_v1.json').read_text())
        design['cache_metadata_sha256'] = sha(cache / 'metadata.json')
        design['manifest_sha256'] = metadata['manifest_sha256']
        design['component_acceptance_report_sha256'] = sha(component / 'report.json')
        design['component_acceptance_verification_sha256'] = sha(component / 'verification.json')
        design_path = root / 'results/paper_components/baseline_design_v1.json'
        write(design_path, design)
        runner = importlib.import_module('src.ect.paper_training')
        sources = {f'src/ect/{name}': sha(importlib.import_module('src.ect.' + Path(name).stem).__file__)
                   for name in runner.CODE_FILES}
        sources['src/ect/paper_registration.py'] = sha(m.__file__)
        for name in ('train_paper_baseline.py', 'register_paper_baseline.py','verify_paper_run.py'):
            sources[f'scripts/{name}'] = sha(ROOT / 'scripts' / name)
        artifact_table={}
        runs=[]
        for device,folder in [('cpu','runner_smoke'),('cuda','runner_smoke_cuda')]:
            run=root/'results'/folder
            proof_path=root/'results'/f'{folder}_verification.json'
            proof=json.loads(proof_path.read_text())
            proof['project_root']=str(root)
            proof['run_path']=str(run)
            updated={}
            old_root=Path(self.prototype.name)
            for name,digest in proof['files_sha256'].items():
                p=Path(name)
                updated[str(root/p.relative_to(old_root)) if p.is_relative_to(old_root) else name]=digest
            proof['files_sha256']=updated
            write(proof_path,proof)
            for p in [run/'report.json',*[run/n for n in runner.ARTIFACTS],proof_path]:
                artifact_table[p.relative_to(root).as_posix()]=sha(p)
            runs.append(dict(device=device,run_dir=run.relative_to(root).as_posix(),
                             verification_path=proof_path.relative_to(root).as_posix()))
        acceptance = dict(schema_version=1, status='verified',
                          protocol='paper_baseline_runner_acceptance_v1',
                          test_classification_enabled=False, fixture_only=True,
                          checks={key: True for key in CHECKS}, code_sha256=sources,
                          files_sha256=artifact_table,runs=runs)
        acceptance_path = root / 'results/runner_acceptance.json'
        write(acceptance_path, acceptance)
        anchor = patch.object(m, 'DESIGN_SHA256', sha(design_path))
        anchor.start()
        self.addCleanup(anchor.stop)
        return m, acceptance_path, root / 'results/registration'

    def register(self, root):
        m, acceptance, output = self.fixture(root)
        registry = m.register_paper_baseline(root, output, acceptance)
        return m, acceptance, output, registry

    def test_module_exposes_independent_registration_interface(self):
        self.assertTrue(callable(getattr(self.module(), 'register_paper_baseline', None)))
        self.assertTrue(callable(getattr(self.module(), 'validate_registered_run', None)))

    def test_unrelated_artifact_and_boolean_checks_cannot_replace_real_runner_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            m, acceptance, output=self.fixture(root)
            value=json.loads(acceptance.read_text())
            value.pop('runs')
            unrelated=root/'unrelated.json'
            write(unrelated,{'unrelated':True})
            value['files_sha256']={'unrelated.json':sha(unrelated)}
            write(acceptance,value)
            with self.assertRaisesRegex(ValueError,'CPU/CUDA|runner evidence|smoke'):
                m.register_paper_baseline(root,output,acceptance)

    def test_smoke_proof_semantics_and_bound_verifier_source_are_required(self):
        changes=[('report_sha256','0'*64),('prediction_counts',{'train':999,'validation':2}),
                 ('device','cuda'),('flags',{}),('source_sha256',{}),('script',{}),
                 ('determinism',{'tf32':True})]
        for field,value in changes:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                m,acceptance,output=self.fixture(root)
                content=json.loads(acceptance.read_text())
                path=root/content['runs'][0]['verification_path']
                proof=json.loads(path.read_text())
                proof[field]=value
                write(path,proof)
                content['files_sha256'][path.relative_to(root).as_posix()]=sha(path)
                write(acceptance,content)
                with self.assertRaises(ValueError):
                    m.register_paper_baseline(root,output,acceptance)

    def test_independent_metric_rounding_uses_declared_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            m,acceptance,output=self.fixture(root)
            content=json.loads(acceptance.read_text())
            path=root/content['runs'][0]['verification_path']
            proof=json.loads(path.read_text())
            proof['metrics']['train']['accuracy']+=1e-14
            proof['maximum_metric_error']=1e-14
            write(path,proof)
            content['files_sha256'][path.relative_to(root).as_posix()]=sha(path)
            write(acceptance,content)
            self.assertEqual(m.register_paper_baseline(root,output,acceptance)['status'],'frozen_execution_registration')

    def test_registration_freezes_three_exact_configs_and_validation_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output, registry = self.register(root)
            self.assertEqual(registry['protocol'], 'paper_baseline_finite_v1')
            self.assertEqual(registry['status'], 'frozen_execution_registration')
            self.assertFalse(registry['test_classification_enabled'])
            self.assertEqual(len(registry['runs']), 3)
            before = {p.relative_to(root).as_posix(): (sha(p), p.stat().st_mtime_ns)
                      for p in root.rglob('*') if p.is_file()}
            for seed, run in enumerate(registry['runs']):
                self.assertEqual(run['run_id'], f'resnext_s{seed}')
                self.assertEqual(json.loads((root / run['config_path']).read_text()), BASE | {'seed': seed})
                self.assertEqual(root / run['output_dir'], output / f'resnext_s{seed}')
                result = m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': seed},
                                                   BASE['cache_dir'], run['output_dir'])
                self.assertEqual(result['registry_sha256'], sha(output / 'registry.json'))
                self.assertEqual(result['run_id'], run['run_id'])
            after = {p.relative_to(root).as_posix(): (sha(p), p.stat().st_mtime_ns)
                     for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(registry['acceptance_sha256'], sha(acceptance))
            self.assertFalse((output / 'resnext_s0').exists())
            with self.assertRaises(FileExistsError):
                m.register_paper_baseline(root, output, acceptance)

    def test_acceptance_requires_exact_checks_complete_code_and_all_artifacts(self):
        mutations = [('status', 'pending'), ('schema_version', True),
                     ('protocol', 'another'), ('test_classification_enabled', True),
                     ('checks', {key: key != 'cuda_training' for key in CHECKS}),
                     ('checks', {key: 1 for key in CHECKS}), ('checks', {}),
                     ('code_sha256', {}), ('files_sha256', {})]
        for field, value in mutations:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, acceptance, output = self.fixture(root)
                changed = json.loads(acceptance.read_text())
                changed[field] = value
                write(acceptance, changed)
                with self.assertRaises(ValueError):
                    m.register_paper_baseline(root, output, acceptance)
                self.assertFalse(output.exists())

    def test_failed_acceptance_artifact_owner_rejected_even_with_valid_sha(self):
        for relative in ('results/failure.json', 'results/runner_smoke/failure.json'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, acceptance, output = self.fixture(root)
                write(root / relative, {'status': 'failed'})
                with self.assertRaisesRegex(ValueError, '[Ff]ail'):
                    m.register_paper_baseline(root, output, acceptance)
                self.assertFalse(output.exists())

    def test_artifact_corruption_and_missing_artifacts_rejected(self):
        for remove in (False, True):
            with self.subTest(remove=remove), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, acceptance, output = self.fixture(root)
                artifact = root / 'results/runner_smoke/report.json'
                if remove:
                    artifact.unlink()
                else:
                    artifact.write_text('changed')
                with self.assertRaises(ValueError):
                    m.register_paper_baseline(root, output, acceptance)

    def test_original_config_and_design_cannot_be_replaced(self):
        for field, value in [('max_epochs', 10), ('purpose', 'smoke'), ('device', 'cpu'),
                             ('learning_rate_milestones_epochs', [500, 750]), ('training_seeds', [0])]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, acceptance, output = self.fixture(root)
                write(root / 'config/paper_baseline_v1.json', BASE | {field: value})
                with self.assertRaises(ValueError):
                    m.register_paper_baseline(root, output, acceptance)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            design = root / 'results/paper_components/baseline_design_v1.json'
            changed = json.loads(design.read_text())
            changed['execution_registry_bound'] = True
            write(design, changed)
            with self.assertRaises(ValueError):
                m.register_paper_baseline(root, output, acceptance)

    def test_root_shadow_sources_never_substitute_for_executed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            fake = root / 'src/ect/paper_models.py'
            fake.parent.mkdir(parents=True)
            fake.write_text('shadow is not executed')
            changed = json.loads(acceptance.read_text())
            changed['code_sha256']['src/ect/paper_models.py'] = sha(fake)
            write(acceptance, changed)
            with self.assertRaises(ValueError):
                m.register_paper_baseline(root, output, acceptance)

    def test_validation_rejects_config_seed_output_cache_and_config_file_tampering(self):
        mutations = [BASE | {'seed': 0, 'device': 'cpu'}, BASE | {'seed': True},
                     BASE | {'seed': 9}, BASE | {'seed': 0, 'max_epochs': 10}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, _, output, registry = self.register(root)
            run = registry['runs'][0]
            for config in mutations:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    m.validate_registered_run(root, output / 'registry.json', config,
                                               BASE['cache_dir'], run['output_dir'])
            with self.assertRaises(ValueError):
                m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                           BASE['cache_dir'], 'results/other')
            write(root / run['config_path'], BASE | {'seed': 0, 'device': 'cuda'})
            with self.assertRaises(ValueError):
                m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                           BASE['cache_dir'], run['output_dir'])

    def test_cache_identity_rejected_even_for_identical_copy(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, _, output, registry = self.register(root)
            shutil.copytree(root / BASE['cache_dir'], root / 'cache_copy')
            with self.assertRaises(ValueError):
                m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                           'cache_copy', registry['runs'][0]['output_dir'])

    def test_validation_rechecks_all_execution_sources_after_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, _, output, registry = self.register(root)
            actual_sha = m.file_sha256
            for label in registry['code_sha256']:
                path = (Path(importlib.import_module('src.ect.' + Path(label).stem).__file__).resolve()
                        if label.startswith('src/ect/') else ROOT / label)
                def drift(candidate, selected=path):
                    return '0' * 64 if Path(candidate).resolve() == selected else actual_sha(candidate)
                with self.subTest(label=label), patch.object(m, 'file_sha256', drift), self.assertRaises(ValueError):
                    m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                               BASE['cache_dir'], registry['runs'][0]['output_dir'])

    def test_registry_incomplete_hash_tables_and_mixed_failure_are_rejected(self):
        for field in ('code_sha256', 'files_sha256', 'runs'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, _, output, registry = self.register(root)
                changed = copy.deepcopy(registry)
                changed[field] = {} if field != 'runs' else []
                write(output / 'registry.json', changed)
                with self.assertRaises(ValueError):
                    m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                               BASE['cache_dir'], registry['runs'][0]['output_dir'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, _, output, registry = self.register(root)
            write(output / 'failure.json', {'status': 'failed'})
            with self.assertRaisesRegex(ValueError, '[Ff]ail'):
                m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                           BASE['cache_dir'], registry['runs'][0]['output_dir'])

    def test_publication_failure_writes_marker_before_cleanup_even_if_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            original = m.write_json
            def fail_publish(path, value):
                original(path, value)
                if Path(path).name == 'registry.json':
                    raise OSError('publication failure')
            with patch.object(m, 'write_json', fail_publish), patch.object(Path, 'unlink', side_effect=OSError('cleanup failure')):
                with self.assertRaises(OSError):
                    m.register_paper_baseline(root, output, acceptance)
            self.assertEqual(json.loads((output / 'failure.json').read_text())['status'], 'failed')
            with self.assertRaises(ValueError):
                m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                           BASE['cache_dir'], output / 'resnext_s0')

    def test_paths_outside_root_and_run_path_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            with self.assertRaises(ValueError):
                m.register_paper_baseline(root, root.parent / 'escaped-registration', acceptance)
            registry = m.register_paper_baseline(root, output, acceptance)
            changed = copy.deepcopy(registry)
            changed['runs'][0]['output_dir'] = '../escaped-run'
            write(output / 'registry.json', changed)
            with self.assertRaises(ValueError):
                m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                           BASE['cache_dir'], root.parent / 'escaped-run')

    def test_cli_help_runs_without_training(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/register_paper_baseline.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--acceptance', result.stdout)

    def test_acceptance_omitted_source_or_wrong_source_hash_is_rejected(self):
        for mode in ('omitted', 'wrong'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, acceptance, output = self.fixture(root)
                value = json.loads(acceptance.read_text())
                key = 'scripts/register_paper_baseline.py'
                if mode == 'omitted':
                    del value['code_sha256'][key]
                else:
                    value['code_sha256'][key] = '0' * 64
                write(acceptance, value)
                with self.assertRaisesRegex(ValueError, 'source'):
                    m.register_paper_baseline(root, output, acceptance)

    def test_validation_rechecks_acceptance_artifacts_components_cache_and_raw(self):
        labels = ['results/runner_acceptance.json', 'results/runner_smoke/report.json',
                  'results/paper_components/acceptance_v1/report.json',
                  'results/paper_components/acceptance_v1/verification.json',
                  'results/paper_components/baseline_design_v1.json',
                  'config/paper_baseline_v1.json', 'manifests/fixture/train.jsonl',
                  'manifests/fixture/test.jsonl', 'manifests/fixture/summary.json',
                  'data/raw/fixture.npy', BASE['cache_dir'] + '/metadata.json',
                  BASE['cache_dir'] + '/normalization.json', BASE['cache_dir'] + '/test_x.npy']
        for label in labels:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, _, output, registry = self.register(root)
                with (root / label).open('ab') as handle:
                    handle.write(b'\nchanged')
                with self.assertRaises(ValueError):
                    m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                               BASE['cache_dir'], registry['runs'][0]['output_dir'])

    def test_lost_atomic_directory_claim_preserves_winner_and_does_not_mark_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            original = Path.mkdir
            def competing_claim(path, *args, **kwargs):
                if path == output:
                    original(path)
                    (path / 'winner.txt').write_text('other owner')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'mkdir', competing_claim), self.assertRaises(FileExistsError):
                m.register_paper_baseline(root, output, acceptance)
            self.assertEqual((output / 'winner.txt').read_text(), 'other owner')
            self.assertFalse((output / 'failure.json').exists())
            self.assertFalse((output / 'registry.json').exists())

    def test_registration_under_failed_parent_is_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, _ = self.fixture(root)
            parent = root / 'failed_registration'
            write(parent / 'failure.json', {'status': 'failed'})
            with self.assertRaisesRegex(ValueError, '[Ff]ail'):
                m.register_paper_baseline(root, parent / 'nested', acceptance)
            self.assertFalse((parent / 'nested').exists())

    def test_registry_paths_cannot_redirect_run_config_or_output_within_root(self):
        for key, value in [('config_path', 'config/paper_baseline_v1.json'),
                           ('output_dir', 'results/another'), ('seed', 2),
                           ('config_file_sha256', '0' * 64), ('resolved_config_sha256', '0' * 64)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, _, output, registry = self.register(root)
                changed = copy.deepcopy(registry)
                changed['runs'][0][key] = value
                write(output / 'registry.json', changed)
                with self.assertRaises(ValueError):
                    m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                               BASE['cache_dir'], registry['runs'][0]['output_dir'])

    def test_acceptance_artifact_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            value = json.loads(acceptance.read_text())
            value['files_sha256'] = {'../outside.json': '0' * 64}
            write(acceptance, value)
            with self.assertRaisesRegex(ValueError, 'inside project root'):
                m.register_paper_baseline(root, output, acceptance)

    def test_metadata_stays_equal_after_mutable_training_report_and_progress_appear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, _, output, registry = self.register(root)
            run = registry['runs'][0]
            args = (root, output / 'registry.json', BASE | {'seed': 0},
                    BASE['cache_dir'], run['output_dir'])
            before = m.validate_registered_run(*args)
            write(root / run['output_dir'] / 'progress.json', {'status': 'running', 'epoch': 10})
            write(root / run['output_dir'] / 'report.json', {'fixture_only': True, 'status': 'complete'})
            write(output / 'execution_status.json', {'resnext_s0': 'complete'})
            self.assertEqual(m.validate_registered_run(*args), before)
            actual_sha = m.file_sha256
            source = Path(m.__file__).resolve()
            def drift(path):
                return '0' * 64 if Path(path).resolve() == source else actual_sha(path)
            with patch.object(m, 'file_sha256', drift), self.assertRaises(ValueError):
                m.validate_registered_run(*args)

    def test_registration_calls_runner_import_identity_guard_before_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            runner = importlib.import_module('src.ect.paper_training')
            with patch.object(runner, 'verify_imported_sources', create=True,
                              side_effect=ValueError('imported source identity differs')):
                with self.assertRaisesRegex(ValueError, 'imported source identity'):
                    m.register_paper_baseline(root, output, acceptance)
            self.assertFalse(output.exists())

    def test_running_registration_or_acceptance_owner_cannot_authorize_training(self):
        for location in ('results/registration', 'results/runner_smoke', 'results'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                m, _, output, registry = self.register(root)
                (root / location / 'running.marker').write_text('running')
                with self.assertRaisesRegex(ValueError, '[Rr]unning'):
                    m.validate_registered_run(root, output / 'registry.json', BASE | {'seed': 0},
                                               BASE['cache_dir'], registry['runs'][0]['output_dir'])

    def test_active_training_output_marker_is_allowed_but_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, _, output, registry = self.register(root)
            run = registry['runs'][0]
            args = (root, output / 'registry.json', BASE | {'seed': 0},
                    BASE['cache_dir'], run['output_dir'])
            initial = m.validate_registered_run(*args)
            run_dir = root / run['output_dir']
            run_dir.mkdir()
            (run_dir / 'running.marker').write_text('Active training; no success report yet')
            self.assertEqual(m.validate_registered_run(*args), initial)
            write(run_dir / 'failure.json', {'status': 'failed'})
            with self.assertRaisesRegex(ValueError, '[Ff]ailed'):
                m.validate_registered_run(*args)

    def test_claim_is_marked_running_and_success_registry_is_last_without_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, acceptance, output = self.fixture(root)
            original = m.write_json
            seen = []
            def observe(path, value):
                path = Path(path)
                seen.append(path.name)
                if path.name.startswith('resnext_s'):
                    self.assertTrue((output / 'running.marker').exists())
                    self.assertFalse((output / 'registry.json').exists())
                if path.name == 'registry.json':
                    self.assertFalse((output / 'running.marker').exists())
                original(path, value)
            with patch.object(m, 'write_json', observe):
                m.register_paper_baseline(root, output, acceptance)
            self.assertEqual(seen[-1], 'registry.json')
            self.assertFalse((output / 'running.marker').exists())


if __name__ == '__main__':
    unittest.main()
