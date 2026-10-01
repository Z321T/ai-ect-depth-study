import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.ect.integrity import file_sha256


ROOT = Path(__file__).resolve().parents[1]


class RegistrationTests(unittest.TestCase):
    def module(self):
        path = ROOT / 'scripts/register_experiments.py'
        self.assertTrue(path.exists(), 'Registration implementation missing')
        spec = importlib.util.spec_from_file_location('registration', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, root):
        root = Path(root)
        config = root / 'config'
        config.mkdir()
        plan = json.loads((ROOT / 'config/experiment_v1.json').read_text())
        (config / 'experiment_v1.json').write_text(json.dumps(plan))
        (config / 'deep_v1.json').write_bytes((ROOT / 'config/deep_v1.json').read_bytes())
        mapping = config / 'class_mapping.json'
        mapping.write_bytes((ROOT / 'config/class_mapping.json').read_bytes())
        cache = root / 'cache'
        cache.mkdir()
        (cache / 'metadata.json').write_text('{}')
        code = root / 'src/ect'
        code.mkdir(parents=True)
        for name in ['deep.py','models.py','devices.py','noise.py','baseline.py','prepare.py','dataset.py','preprocessing.py','integrity.py','similarity.py','features.py']:
            (code / name).write_bytes((ROOT / 'src/ect' / name).read_bytes())
        scripts = root / 'scripts'
        scripts.mkdir()
        for name in ['register_experiments.py','diagnose_similarity.py','train_deep.py']:
            (scripts / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
        metadata = {'protocol':'grouped_v1', 'summary_sha256':'summary', 'manifest_sha256':{'train.jsonl':'a','validation.jsonl':'b','test.jsonl':'c'},
            'files_sha256':{f'{s}_{k}.npy':f'{s}_{k}' for s in ['train','validation','test'] for k in ['x','y']},
            'inputs':{'train':{'sha256':'raw'}}, 'sample_counts':{'train':20,'validation':20,'test':20}}
        for relative, reference in zip(plan['audit_reports'], ['train','validation']):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            artifact=path.parent/'nearest.csv'
            artifact.write_text('fixture\n')
            audit = dict(schema_version=1,query_split='test', reference_split=reference, query_count=20, reference_count=20,
                exhaustive_pair_count=400, thresholds={'raw':.001,'shape':.01},
                cache_metadata_sha256=file_sha256(cache/'metadata.json'), summary_sha256='summary',
                manifest_sha256={f'{s}.jsonl':metadata['manifest_sha256'][f'{s}.jsonl'] for s in ['test',reference]},
                cache_files_sha256={f'{s}_{k}.npy':metadata['files_sha256'][f'{s}_{k}.npy'] for s in ['test',reference] for k in ['x','y']},
                code_sha256={'src/ect/similarity.py':file_sha256(code/'similarity.py'),
                    'scripts/diagnose_similarity.py':file_sha256(scripts/'diagnose_similarity.py')},
                artifacts_sha256={'nearest.csv':file_sha256(artifact)},figure_created=False,
                raw_inputs_declared_by_cache=metadata['inputs'],
                raw={'candidate_pair_count':0,'candidate_query_count':0},
                shape={'candidate_pair_count':0,'candidate_query_count':0})
            path.write_text(json.dumps(audit))
        baseline = root / plan['baseline_report']
        baseline.parent.mkdir(parents=True,exist_ok=True)
        for name in ['model.joblib','train_rows.json','validation_predictions.csv']:
            (baseline.parent/name).write_text('fixture artifact\n')
        baseline.write_text(json.dumps({'status':'complete','heldout_test_evaluated':False,
            'sample_counts':{'train':20,'validation':20}, 'cache_metadata_sha256':file_sha256(cache/'metadata.json'),
            'manifest_sha256':metadata['manifest_sha256'],
            'files_sha256':{name:file_sha256(baseline.parent/name) for name in ['model.joblib','train_rows.json','validation_predictions.csv']}}))
        class Store:
            folder = cache
            def __init__(self,*a,**kw): self.metadata=metadata
            def __enter__(self): return self
            def __exit__(self,*a): pass
        return plan, Store

    def test_frozen_configs_registered_before_training_without_test_classification(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, store=self.fixture(tmp)
            with patch.object(m,'PreparedDataset',store):
                registry=m.register(tmp,cache_dir='cache')
            self.assertEqual(len(registry['runs']),9)
            self.assertFalse(registry['test_classification_evaluated'])
            self.assertEqual({r['seed'] for r in registry['runs']},{0,1,2})
            for run in registry['runs']:
                config=json.loads((Path(tmp)/run['config_path']).read_text())
                self.assertIsNone(config['train_per_class'])
                self.assertIsNone(config['validation_per_class'])
                self.assertEqual(config['max_epochs'],30)
                self.assertEqual(config['device'],'auto')
                self.assertEqual(file_sha256(Path(tmp)/run['config_path']),run['config_file_sha256'])
                self.assertEqual(run['status_at_registration'],'registered')
            with patch.object(m,'PreparedDataset',store), self.assertRaises(FileExistsError):
                m.register(tmp,cache_dir='cache')

    def test_candidates_or_mismatched_data_identity_prevent_registration(self):
        m=self.module()
        for field in ['candidate','cache','threshold','algorithm','baseline']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                plan,store=self.fixture(tmp)
                path=Path(tmp)/plan['audit_reports'][0]
                obj=json.loads(path.read_text())
                if field=='candidate': obj['shape']['candidate_pair_count']=1
                elif field=='cache': obj['cache_metadata_sha256']='wrong'
                elif field=='threshold': obj['thresholds']['shape']=.02
                elif field=='algorithm': obj['code_sha256']['src/ect/similarity.py']='wrong'
                else:
                    path=Path(tmp)/plan['baseline_report']
                    obj=json.loads(path.read_text());obj['heldout_test_evaluated']=True
                path.write_text(json.dumps(obj))
                with patch.object(m,'PreparedDataset',store), self.assertRaises(ValueError):
                    m.register(tmp,cache_dir='cache')
                self.assertFalse(Path(tmp,'results/experiments/formal_v1/registry.json').exists())

    def test_missing_corrupt_or_failed_svm_cannot_authorize_registration(self):
        m=self.module()
        for fault in ['weight_missing','csv_changed','failure']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                plan,store=self.fixture(tmp)
                folder=(Path(tmp)/plan['baseline_report']).parent
                if fault=='weight_missing': (folder/'model.joblib').unlink()
                elif fault=='csv_changed': (folder/'validation_predictions.csv').write_text('changed\n')
                else: (folder/'failure.json').write_text('{}')
                with patch.object(m,'PreparedDataset',store), self.assertRaises(ValueError):
                    m.register(tmp,cache_dir='cache')

    def test_cleanup_failure_revokes_registry_success_marker(self):
        m=self.module()
        original=tempfile.TemporaryDirectory
        class FailedCleanup(original):
            def __exit__(self,*args):
                super().__exit__(*args)
                raise OSError('cleanup fault')
        with original() as tmp:
            _,store=self.fixture(tmp)
            with patch.object(m,'PreparedDataset',store), patch.object(m.tempfile,'TemporaryDirectory',FailedCleanup):
                with self.assertRaisesRegex(OSError,'cleanup fault'):
                    m.register(tmp,cache_dir='cache')
            folder=Path(tmp,'results/experiments/formal_v1')
            self.assertTrue((folder/'failure.json').exists())
            self.assertFalse((folder/'registry.json').exists())

    def test_incomplete_audit_artifacts_and_provenance_are_rejected(self):
        m=self.module()
        for fault in ['missing_table','missing_csv','summary','cli','inputs']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                plan,store=self.fixture(tmp)
                path=Path(tmp)/plan['audit_reports'][0]
                audit=json.loads(path.read_text())
                if fault=='missing_table': audit.pop('artifacts_sha256')
                elif fault=='missing_csv': audit['artifacts_sha256']={}
                elif fault=='summary': audit['summary_sha256']='wrong'
                elif fault=='cli': audit['code_sha256']['scripts/diagnose_similarity.py']='wrong'
                else: audit['raw_inputs_declared_by_cache']={}
                path.write_text(json.dumps(audit))
                with patch.object(m,'PreparedDataset',store), self.assertRaises(ValueError):
                    m.register(tmp,cache_dir='cache')


if __name__=='__main__':
    unittest.main()
